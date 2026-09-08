---
title: "vLLM Scheduler：每步 token 预算、抢占与结果对账"
---

# vLLM Scheduler：每步 token 预算、抢占与结果对账

> **源码基线**：`vllm-project/vllm@199cb9b964822e59ab9b58d88e7be31eb419a2ae`（main 快照，2026-09-07 UTC）
> **主题**：一次 `schedule()` 怎样在 running decode 与 waiting prefill 之间分配预算，取得 KV/encoder 容量，再根据执行结果修正进度。
> **适用范围**：V1 `Scheduler` / `AsyncScheduler` 的队列、token/input/spec/encoder 预算、抢占、输出与完成；设备行列及异步执行归 11/12，KV block/hash/refcount 算法归 08，采样和投机正确性归 14/16。
> **最近更新**：2026-09-08。按固定源码与测试静态核验；数值例用于重放控制流，未实跑模型、GPU 或 connector。

## 1. 一步只能算 6 个 token，先给谁

设每步 token budget 和 input budget 都为 6，最多 3 个 running 请求；开启 chunked prefill，关闭长 prefill 限额，无 spec、媒体或 prefix hit，KV 足够。R 已在 decode：已知 21 个 token、算过 20 个，差 1 个；P、Q 按顺序等待，prompt 分别长 10、3。每步结果返回后再排下一步，R 在这三步内不结束。

| step | 先扫描 running | 剩余预算如何准入 waiting | 发出计划 | 结果返回后的变化 |
|---|---|---|---|---|
| 1 | R 得 1，预算 6→5 | P 要 10，只取 5，预算归零；Q 留 waiting | R:1，P:5 | R 多一个输出；P 只完成前 5 个 prompt token，无采样输出 |
| 2 | R 得 1；P 已是 running，取剩余 5 | 预算用完，Q 继续等 | R:1，P:5 | P 完成 prompt，得到第一个输出 token |
| 3 | R、P 各差 1，预算 6→4 | Q 取 3，预算剩 1 | R:1，P:1，Q:3 | Q 完成 prompt，得到第一个输出；无需为了凑满预算再造工作 |

这里“发出 5 个 token”是计算 5 个输入位置，不是向用户交付 5 个新 token。`num_computed_tokens` 在提交计划时已经推进，表中“结果返回”才确定采样输出。若不开 chunked prefill，等待中的长请求放不下会停止本轮 waiting 扫描；不能从本例推成“总会越过长请求，让后面的短请求先走”。`test_schedule_order` 用 800/800/10/10 的请求和 1024 预算验证了这个差别。

<!-- 图1 spec：输入是 R 差1、P差10、Q差3和预算6；沿一次 schedule 的实际顺序显示6→5→0，输出R1/P5和仍waiting的Q；矩形依次表示准入动作与结果对账，非时间比例图。 -->
```mermaid
flowchart TB
    I["step 1：预算 6，KV 足够<br/>running R 差 1；waiting P 差 10、Q 差 3"] --> R["R：裁剪为 1，取得 KV slots<br/>登记 R:1，预算 6 → 5"]
    R --> P["P：chunk 裁剪为 5<br/>取得 KV slots 后转 RUNNING"]
    P --> O["登记 P:5，预算 5 → 0<br/>Q 仍在 waiting"]
    O --> C["封装计划 R:1、P:5<br/>computed 和 in-flight 先增加"]
    C --> U["用这份计划与执行结果对账<br/>R 交付新 token，P 仍在 prefill"]
    classDef acc fill:#dbeafe,stroke:#2563eb,color:#0f172a
    class R,P,O acc
```

这就是连续调度的基本单位：**每步重新分配要计算的位置数**。固定成员、等整个 batch 全部完成才换人的方案，无法在下一个 step 立即把空出的机会给新请求；这是从当前 running-first / waiting-admission 控制流重建的设计理由，不是源码中的历史决策记录。源码明确采用统一进度模型，不为 prefill 和 decode 建两套调度阶段；chunked prefill、prefix hit、spec validation 都是在追赶同一请求的目标进度。

但“每步重排”也把 Scheduler 放进延迟关键路径。token 数、可驻留 request slot、KV blocks 分别受限；多发 token 可能拉长一步，多占 KV 可能引发重算，多接纳请求也不等于降低排队时间。后文沿本例逐项加入这些限制。

## 2. 调度器记住了什么

| 状态 | 用途 | 不能混同的边界 |
|---|---|---|
| `requests` | request id 到 `Request` 的映射 | connector 延迟释放时，终态对象仍可能在映射中 |
| `waiting`、`skipped_waiting` | 准入候选，以及等待依赖/本轮约束而跳过的请求 | 队列位置不是额外的 `RequestStatus`；blocked 不等于 finished |
| `running` | 已准入、占活跃 slot 的请求 | 不保证每步都执行；当步执行子集看 `num_scheduled_tokens` |
| `num_tokens_with_spec` | prompt + 已返回 output + 当前 draft 的长度 | draft 还没有通过验证 |
| `num_computed_tokens`、`num_in_flight_tokens` | 已提交进度，以及尚未返回的执行位置数 | computed 含乐观推进，不等于全部已确认的 KV |
| `num_output_placeholders`、`num_stale_output_tokens` | 预期输出数量，以及抢占前仍待排空的执行位置数 | 两者单位/消减规则不同，不能互相代替 |
| 本步 maps | 新 blocks、每请求 token 数、encoder 项、scheduled spec | 联合资源成功后才登记；异步 KV load 可只持 blocks 而不执行 |
| `finished_req_ids`、`reset_preempted_req_ids` | 下游清理旧请求镜像的生命周期增量 | 放进 output 后换新 set，不能原地 `clear()` 破坏旧计划 |

请求状态区分普通等待、三个 blocked waiting、running、preempted 和终态；`RequestStatus.is_finished()` 将枚举中 `PREEMPTED` 之后的值视为终态。图中的资源驻留说明不是新的枚举。

<!-- 图2 spec：输入是新请求；展示三个异步等待原因、准入运行、抢占重排和终止路径；终态后connector可留对象，KV fence可继续留块；不把skipped队列误画为enum。 -->
```mermaid
stateDiagram-v2
    state "WAITING" as W
    state "等待 grammar" as G
    state "等待 remote KV" as K
    state "等待 streaming 输入" as S
    state "RUNNING" as R
    state "PREEMPTED" as P
    state "终态" as F
    [*] --> W: 新请求
    [*] --> G: grammar 异步构建
    G --> W: ready
    W --> K: 保留 blocks 并发起 load
    P --> K: 恢复时异步 load
    K --> W: 首次请求 transfer 完成
    K --> P: 被抢占请求 transfer 完成
    W --> R: 预算与资源允许
    P --> R: 普通 stale 排空后恢复<br/>drop 模式可提前恢复
    R --> P: KV 不足
    R --> S: 可续会话等待新输入
    S --> W: 输入到达
    W --> F: abort 或 error
    G --> F: abort 或 grammar error
    K --> F: abort 或 transfer error
    S --> F: 会话结束或 abort
    R --> F: stop、length、abort 或 error
    P --> F: stale 输出触发 stop 或 abort
    F --> [*]: connector 允许后删除对象<br/>KV 回池还可能等 fence
```

五条不变量贯穿后文：已发 token 总量不超上限；token/input budget 非负；running 数不超 slot 上限；发给 runner 的计划只能包含本步能执行的联合保留；返回结果必须按产生它的那份计划对账。源码在封装 output 前直接断言前几项。`running` 数可以大于当步 scheduled 请求数，因此不能拿整个 running 列表当作模型输入。

## 3. 一个请求的 token 数怎样逐项被裁剪

### 3.1 两种 token 预算和 request slot

每轮 `token_budget = max_num_scheduled_tokens`，`input_budget = max_num_batched_tokens`。二者通常相等；模型会在执行中追加输入位置时，调度上限可以更小。若 speculative 配置要求 `draft_slots = max_num_new_slots_for_drafting`，每接纳一个请求，input budget 扣掉的是 `num_new_tokens + draft_slots`，token budget 只扣 `num_new_tokens`。例如 token budget 6、input budget 8、每请求 draft slots 2：第一个请求取 4 后，剩 token=2、input=2；第二个请求因 `input_budget <= draft_slots` 停止，即使还剩 token 预算也不能入场。

running 的候选量按 `num_tokens_with_spec + num_output_placeholders - num_computed_tokens` 计算。随后按实际顺序处理：长 prefill threshold → token/input 剩余额度 → model length（留本步采样位置）→ Mamba split → encoder 边界 → MTP prefill lookahead。普通自回归每步采样位置数为 1，diffusion 为 0，不能把“总要再留一个 bonus token”当作所有模型的通则。

候选量为零时，running 循环通常 `continue`：可能前一步仍在途、已经到长度上限、encoder budget/cache 不足，或没有足够预算跨过对齐/预读边界；后面的请求仍可运行。源码明确指出这放松了严格 FCFS。V2 + PP + async 还检查 `next_decode_eligible_step`，同一请求两次 decode 至少间隔 PP size 个调度 step；达到输出上限的 placeholder guard 则避免确定无用的额外一步。

waiting 除 token 外还检查 `len(running) + num_waiting_for_streaming_input`：暂停等输入的 streaming session 仍占 runner slot。`max_num_seqs` 是驻留/执行容量约束，前端 `max_num_queued_reqs/tokens` admission 是另一道入口限流，见 [[02_engineering/03_infer_frameworks/vllm/03_vllm_request_semantics_analysis|请求语义]]，两者不能替代。

### 3.2 speculative 也花预算，且 shape 不能随意截断

running 只将批准区间内的 draft 写入 `scheduled_spec_decode_tokens`，然后清空 request 的旧 draft，等 `update_draft_token_ids()` 或 async worker 更新。prefill chunk 不接收 draft：现有测试以 prompt 80、预算 50、draft 3 逐步验证 **50 → 30 → 1+3**；第二步是剩余 30 个 prompt 位置，不能混入 3 个 draft 而变成 33。投机的 propose/verify/accept 分布推导属于 [[02_engineering/03_infer_frameworks/vllm/16_vllm_speculative_decoding_analysis|投机解码]]。

另一个分支发生在 waiting 请求只差 1 个位置时，例如 33-token prompt 命中 32-token prefix。若已有 running decode 或命中进度非零，batch 尚无已排 prefill，使用固定 K 的自回归 spec，且模型长度与预算容得下，Scheduler 可将它补成 `1+K` 行并附 `[-1] * K`，保持 uniform decode，便于 full CUDA Graph。它不是已经产生了 K 个真实 draft。容量不足以保留整个 `1+K` 时，本轮先不准入；已有 prefill、dynamic K 或 diffusion 时不套这条 padding 规则。

**Mamba 对齐后的修正不同于预算不足。** 已补成 `1+3=4` 行的请求，若 split 裁到 1、2、3 中任何一个正数，最终都回退为 **1 行并清掉 padding 标志**，不附 spec placeholders。否则 sampler 按 draft 数推导的 row window 与实际 query 行数不一致；回归测试明确覆盖三个裁剪值。对齐直接得到零则本轮停止准入。动态 spec 在计划结尾按本步 scheduled request 数查询下一步 K；它是下一轮 draft 数选择，不能追溯改写本轮已批准区间。

### 3.3 encoder 预算决定 decoder 能走到哪里

在主例中加入一幅图：P 的媒体占位从位置 4 开始，需要 6 个 encoder embeddings，本步 encoder budget 只有 4，起点为 0、无预读 shift。即使 decoder 获得 5-token 候选区间，也只能取前 4 个文本位置。若下一步起点已在 4，encoder 仍不可用，就取零；running 跳过 P，继续尝试后面的请求。

`_try_schedule_encoder_inputs()` 只检查本步 token 区间（含 drafter read-ahead）覆盖的媒体项，区分已缓存、同一步重复 hash、远端 EC cache 命中和新计算。新计算同时受 encoder compute budget 与 encoder cache 容量约束，通常整个媒体项一起编码；远端加载仍占 cache 容量但不扣本地编码 compute。`disable_chunked_mm_input` 还会把跨不完整媒体项的区间退到该项之前。encoder-decoder 在 decoder 进度为零时先保证 encoder 输入，已有 decoder 进度后不按普通 decoder 媒体占位重复处理。

旧例 `test_schedule_partial_requests` 仍很有区分力：3 个 800-token 请求、媒体区间从 100 起长 600、token/encoder budget 各 1024，第一步排 **800/100/100**；结果返回后第二步排 **1/700/0**。第三个请求还在 running，却没有本步执行项。这也说明“encoder 是 forward 前的附加工作”不够准确：它先裁剪整个调度区间。encoder/媒体算子的设备执行见 [[02_engineering/03_infer_frameworks/vllm/15_vllm_multimodal_execution_analysis|多模态执行]]。

EAGLE 类方法的 prefill lookahead 通常为 1；multi-module MTP 为 spec 数。这个 shift 同时影响 encoder 提前调度、延后释放与 chunk 末端：若 prompt 10、lookahead 3、候选先算 8，会只留下 2 个已知输入供下轮 drafter 预读，因此 `_reserve_prefill_lookahead()` 将本轮退到 7，留下完整 3 个。要么完成 prefill，要么留够预读窗口；不能让尾部 MTP 模块过早改读采样 draft 并污染其 KV。编码后的 cache 也要等已确认进度越过媒体末端加 lookahead 才释放，不能只看包含 placeholders 的乐观 computed。

### 3.4 Mamba split 保证缓存的是哪个位置的状态

Mamba `align` 模式保存的是某个确切 token 边界后的递归状态。可复用的完整块槽 p 必须代表计算完 `(p+1)*block_size` 个 token 的状态；把在 364 处结束的中间状态标成 state@1600，会让命中它的后续请求从错误状态恢复。普通 attention 的 token KV 与这种递归状态不能用同一“随便切一个 chunk”的假设。

`_mamba_block_aligned_split()` 先合并已有进度、本地命中、外部命中得到 start，只在 prefill/重放旧输出期间裁剪。中间 chunk 向块边界对齐；若物理块大于整个配置允许的 chunk，允许先以私有 running state 小步前进，再停在下一个边界。还有几个必须检查的提前停止点：从块中部恢复后的下一个整块边界、最后可缓存块边界、细粒度 prefix hit 所需的 prompt 最后 hash 边界、按块向下对齐的 shared-prefix 分叉点。不能把所有情况简写为“永远按 block_size 向下取整”。

<!-- 图3 spec：两个具体query分别重放共用checkpoint校验；1984→3602的initial/checkpoint列均1因而拒绝，0→100的列为-1/0且满足hash与16对齐因而可导出96；明确输入、算式、判定与下一步，不是二维KV布局。 -->
```mermaid
flowchart TB
    subgraph A["块中部恢复：必须先停在 3200"]
        direction TB
        I["start 1984，end 3602<br/>Mamba block 1600"] --> C["initial 列 = floor(1983/1600) = 1<br/>checkpoint 列 = ceil(3602/1600) - 2 = 1"]
        C --> F["1 不大于 1：checkpoint 与 initial 槽冲突<br/>内部导出无效，先算 3200 - 1984 = 1216"]
    end
    subgraph B["可导出内部 checkpoint 的 query"]
        direction TB
        J["start 0，end 100，checkpoint 96<br/>hash 8，Mamba block 64，alignment 16"] --> K["initial 列 = floor(-1/64) = -1<br/>checkpoint 列 = ceil(100/64) - 2 = 0<br/>0 大于 -1，两个槽不冲突"]
        K --> V["起点按 8 对齐，96 至少距起点 8<br/>0 小于 96 小于 100，96 可被 16 整除"]
        V --> O["校验有效：本步导出 state@96<br/>并完成 query 到 100"]
    end
    A ~~~ B
    classDef acc fill:#dbeafe,stroke:#2563eb,color:#0f172a
    classDef warn fill:#ffedd5,stroke:#ea580c,color:#0f172a
    class V,O acc
    class F warn
```

图例来自 `test_partial_checkpoint_resume_stops_at_mamba_block_boundary`：prompt=3602、start=1984、block=1600，先算 `3200-1984=1216`。即使启用内部 checkpoint，该位置对应的 checkpoint 列会与 initial-state 列冲突，仍不能跨过 3200。相反，支持导出 checkpoint 的 backend 可在最后一次 prefill 内同时保存中间状态，免去某些额外切分；这必须通过共用 `is_mamba_prefill_checkpoint_valid()`：起点 hash 对齐、checkpoint 严格在 query 内、离起点至少一个 hash block、相对起点满足 backend alignment，而且 checkpoint 列必须在 initial-state 列之后。

例如测试中的 start=0、end=100、hash=8、Mamba block=64、alignment=16：checkpoint=96 有效，88 因不满足相对起点 16 对齐而无效；alignment 未声明也无效。Kimi K3 KDA metadata 构建实际使用 **该层 `kv_cache_spec.block_size`** 计算 checkpoint 列，并调用同一校验器；不能拿全局配置块大小替代所有层。Scheduler 当前选择第一个 Mamba spec 的 checkpoint alignment，源码仍有“不同 Mamba spec 对齐要求”的支持 TODO；此处不推成任意混合后端均已支持。

新基线还分开 `use_eagle` 与 `use_eagle_block_drop`：前者决定 hidden-state drafter / 预读语义，后者才决定丢弃易变的尾部 prefix block，并传给 KV manager 与 split/checkpoint 计算。禁用 block drop 不会同时关闭 EAGLE。测试用 prompt=3602、block=1600、无内部 checkpoint，开启 drop 首次停在 1600，关闭则停在 3200。块分配与 checkpoint 的物理保存、partial-tail hash/CoW 仍由 08 页展开。

## 4. KV 不够时，怎样撤回本步已经选中的请求

running 请求的 token 区间算好后，`allocate_slots()` 尝试落实逻辑 KV。失败会从 running 选 victim：FCFS 从列表尾部取；priority 按 `(priority, arrival_time)` 最大者取，数值越大优先级越低，同优先级晚到者先被选。priority 排序主要决定等待队列与 victim，不能假定 running 列表每轮都重新全排序。

若 victim 恰好已经在本步更早登记，必须从 scheduled running、token map、new block map、spec map 删除它，退回 `restored_tokens` 的 token budget、`restored_tokens + draft_slots` 的 input budget，并退回其本步已排 encoder embeddings 的 compute budget。移除列表前方 victim 时还要调小扫描游标，避免漏掉下一个请求。随后释放 victim 的 KV/encoder 引用，改成 PREEMPTED、computed 归零、清空 draft 与 placeholders，放回 waiting 队首，累计 preemption 并发出 reset id；继续尝试为当前请求分配，直到成功或当前请求自己也成为 victim。

现有 priority 测试给出可重放的容量例：block_size=16，总 6 块含 1 个 null，实际可用 5 块。低优先级 L 的 32-token prompt 先占 2 块，输出后下一步扩成 3 块；随后高优先级 H 的 32-token prompt 占 2 块，正好用完。再下一步先选中的 L 尚在这 3 块内，H 则需要第 3 块：此时抢占 L，**撤销刚登记的 L:1**，归还预算与 L 的 3 块，H 才能继续。最终计划里不能同时有“执行 L”与“L 的旧 KV 已释放”。

<!-- 图4 spec：采用priority测试的5个可用块与token预算200；L先登记1却随后成为victim，展示撤销L1、budget199→200、释放L3块，再给H第3块并输出仅H1；free块0→3→2，不画物理布局。 -->
```mermaid
flowchart TB
    I["步前：L 占 3 块，H 占 2 块，free 0<br/>token budget 200；L 优先级低"] --> L["先为 L 登记本步 1 token<br/>无需新块，budget 200 → 199"]
    L --> H["H 也要 1 token，但须增加第 3 块<br/>free 0，allocate_slots 失败"]
    H --> V["选 L 为 victim，撤销本步 L:1<br/>移出本步 token 与 blocks 计划项<br/>归还预算 199 → 200"]
    V --> F["释放 L 的 3 块，free 0 → 3<br/>L 改 PREEMPTED，computed 归零"]
    F --> A["H 取得 1 个新块，free 3 → 2<br/>登记 H:1，budget 200 → 199"]
    A --> O["最终计划只有 H:1，H 共占 3 块<br/>L 等待重算，本轮不再准入 waiting"]
    classDef acc fill:#dbeafe,stroke:#2563eb,color:#0f172a
    classDef warn fill:#ffedd5,stroke:#ea580c,color:#0f172a
    class A,O acc
    class V,F warn
```

这段可以用“本步预留—撤回—提交”理解，但源码并没有把它正式称为事务，也没有通用的异常回滚系统。不是任意阶段异常都能自动恢复所有资源：它实现的是这些确定分支中的显式撤回。普通同步可运行请求只有 KV 成功后才扣预算、进入 scheduled maps；waiting 的 KV 分配失败还会撤销 encoder cache manager 的临时 touch 并停止准入。

本轮一旦发生 preemption，就不再接纳 waiting 新请求。**分析推断**：这避免在刚为已有工作回收容量的同一步又引入新竞争者；源码有 guard，没有写这条理由。暂停全部时 token budget 为零；非完全暂停但不允许新请求时也不进入 waiting admission。

抢占不是把 KV swap 到 CPU 保存。V1 设计文档明确移除了 swapped preemption 与 `--swap-space`，改用 prefix caching 加 recompute：恢复时重查可用前缀，再计算缺失部分。computed 归零意味着重新建立有效进度，未必意味着所有历史 token 都要从零做前向，但重算绝不是免费暂停。

## 5. waiting：就绪、slot 与容量同时满足才入场

waiting 扫描先在普通与 skipped 队列中按策略挑候选。grammar 尚未就绪、remote KV 尚未完成、streaming 输入未到的请求继续跳过；grammar 编译异常进入 request-level error 路径。新的 LoRA 会超出本步 `max_loras`、connector 暂不能确定命中数、EC 预取尚未就绪，也会移到本步 skipped 队列再尝试其它请求；pass 末把跳过项放回 skipped 前部。相反，slot 耗尽、等待长请求的不可切分区间放不下、KV 分配失败会停止这次 waiting 扫描。**跳过与停止不是同一策略。**

真正可准入的候选先查本地 prefix，再查 connector 的外部 prefix，确定从哪里继续；重启/重放时使用整个 `request.num_tokens`，不仅是原 prompt 长度。接着裁剪 token/encoder、计算 lookahead/cross-attention slots，再调用 KV allocation。成功后才从队列弹出、加入 running、按 WAITING/PREEMPTED 分别标 new/resumed、记 scheduled maps、扣预算并写回 computed 起点。

异步 KV load 是这个顺序中的明确例外：它先**只保留传输需要的 blocks**，本轮新执行 token=0，spec lookahead slots 留待以后分配；分配时考虑其它 in-flight prefill 尚需的容量，避免无法抢占的 load 把后续完成空间占尽。allocation 成功后从队列取出，却转为 `WAITING_FOR_REMOTE_KVS` 放回 skipped，写入预计命中进度并直接继续扫描：不进入 running，不写 scheduled-token map，不扣本轮执行预算。这个 computed 值在 transfer ready 前不能当作已加载成功的 KV。

worker 报告接收完成后才缓存有效前缀、promote 为 WAITING 或 PREEMPTED，并重新参加准入。全 prompt 命中仍留最后一个 token 重算以取得 logits。需要清零的新块若正被异步 load 覆写，本步跳过 zeroing，避免两条写入互相竞争；加载失败后只保留有效前缀，其余部分重新计算前补回清零要求。传输协议细节见 [[02_engineering/03_infer_frameworks/vllm/22_vllm_disaggregated_kv_serving_analysis|分离式 KV Serving]]。

DP prefill balancing 可让 Core 对某些 step 传入 `throttle_prefills`：存在需要保护的 decode、且上次放行不是容量饱和时，running prefill 暂停、waiting 本地 prefill 延后，decode 继续。没有 decode 工作可保护时仍允许 prefill，避免白跑 dummy；它并非简单的“每隔固定 N 步才能处理 prompt”。Core 的全局 unfinished 同步是另一机制：当前基线在 step 1 及 `dp_sync_interval` 倍数同步，调用与 wave 完成见 [[02_engineering/03_infer_frameworks/vllm/06_vllm_engine_architecture_analysis|Engine 架构]]。

## 6. 计划发出后，进度怎样变成事实

### 6.1 原计划与结果配对

`SchedulerOutput` 携带 new request 首次数据、cached request 差量、每请求 token 数、spec tokens、encoder 项、共同前缀、finished/reset ids，以及 connector metadata、待清零块与 CoW copy 工作。V2 把 resumed 请求并入 new 数据恢复完整 token 历史；V1 在 cached delta 标记 resume。KV connector 的精确 block snapshot 只供 Scheduler 构造 metadata，发往 worker 前会清掉。runner 的 compact/stable row 更新属于 15/16，本页不推断它们的设备布局。

output 先保留原始进度，随后 `_update_after_schedule()` 才增加 computed 和 in-flight。这使下一次 schedule 能立即排后续 prompt chunk；未来 spec rejection 再回退。routed-expert 返回还会先快照 block IDs，防止异步抢占后无法按原执行读取结果。EngineCore 负责保留这份计划并与对应 future FIFO 配对，详见 06 页。

`AsyncScheduler` 对非 partial-prefill 增加本步预期的 sampled + scheduled spec 数为 placeholders，设置下一轮 spec placeholder 列表；grammar 依赖尚未返回 token 时设置 pending 标志，由 Core 延后生成相应 mask/采样。它不是把未知 token 当作已知文本，而是给下一轮调度提供位置数量。新的 decode 资格 step、KV cache 可确认边界和输出上限 guard 都要使用这些计数。

以普通自回归的一次 spec 为例：已知 token 数 21、computed=20，draft=3，本轮排 4。提交后 computed=24、in-flight 加 4，async placeholders 加 4；假设结果为“接受 1 个 draft + 1 个采样 token”，则接受 draft 数=`2-1=1`、拒绝数=`3-1=2`。结果对账先把 in-flight 减 4，computed 回退到 22，placeholders 因 rejection 从 4 减到 2，再因交付 2 个 token 减到 0；已知 token 数变成 23，下一轮又差 1。若中间发生抢占，这些回退不能照搬，见下一节。

### 6.2 三种迟到/失败结果，不能都叫“丢弃 stale”

`update_from_output()` 按传入计划的 `num_scheduled_tokens` 遍历，先减少仍存在请求的 in-flight 与 stale 份额，再检查是否受 KV load failure 影响、是否已删除/终态，最后通过 runner 的 `req_id_to_index` 取结果。队列位置不能当作结果行号；abort/finished 不会被旧结果重新激活。

| 返回结果类型 | token 是否交付 | 计数与后续动作 |
|---|---|---|
| 正常结果 | 交付实际接受 token | 回退 spec rejection，再扣 async placeholders；可推进 grammar 与 stop |
| 普通 KV 压力抢占前的 stale | 可交付，仍可触发 stop | computed/placeholders 已在 preempt 重置，不再扣 rejection 或交付数量；未排空前暂不恢复 |
| 特殊 drop-mode stale | 全部跳过 | 排空 in-flight/stale 份额；不给用户交付、不推进 grammar、不修改已重置进度 |
| KV load 失败影响的本步结果 | 跳过该结果 | 先截断到有效 prefix，再按 recompute/fail 策略恢复或终止 |
| 对象已删除或终态 | 忽略 | 不重新入队，不复活；对象可能仍因 connector 保留 |

普通 preempt 把 `num_stale_output_tokens` **赋值为**当前 in-flight，不累加，因为在途总量已经包含未排空的旧份额。waiting 看见可交付 stale 尚未排空就跳过该请求，避免恢复时重采同一个位置；结果仍按原序交付，保持 spec 接受行为。AsyncScheduler 仅在更新前状态仍为 RUNNING 时 cache 新确认的 blocks，PREEMPTED stale 不会提交到已释放的旧 KV。

`reset_prefix_cache` 需要同一步恢复，以及 connector `requires_kv_delivery` 的 KV hand-off 情况，使用 drop 模式：旧 KV 已不能支持这些 token 的交付语义，或相同位置已经重采，所以不交付旧结果。多次抢占时，尚未排空的 drop 份额仍保持 drop，不能改回可交付。现有 async 测试覆盖普通 KV 压力、重复 reset、PP 多步在途、producer/consumer 不同 hand-off 要求，并检查 token 恰好交付一次、无 placeholder 下溢、无位置重复采样。

<!-- 图5 spec：输入是原计划S和结果O；先排空本步in-flight，再区分失效KV、终态、drop stale、可交付stale和正常结果；只有正常分支回退已重置前不存在的计数，两个可交付分支汇合stop；负向分支输出明确。 -->
```mermaid
flowchart TB
    I["原计划 S + 返回结果 O<br/>按 request id 配对"] --> D["对象仍存在时：减本步 in-flight<br/>若有 stale，同时排空其份额"]
    D --> K{"结果依赖失败 KV load？"}
    K -->|是| F["跳过 token<br/>有效前缀重算或 FINISHED_ERROR"]
    K -->|否| T{"对象已删除或终态？"}
    T -->|是| X["忽略，不复活"]
    T -->|否| S{"stale 模式？"}
    S -->|drop| X
    S -->|普通可交付| P["保留 token<br/>不再扣 reset 后的计数"]
    S -->|非 stale| R["回退 rejected draft<br/>交付时扣 placeholders"]
    P --> O["追加 token，检查 grammar 与 stop<br/>继续运行或进入完成路径"]
    R --> O
    classDef acc fill:#dbeafe,stroke:#2563eb,color:#0f172a
    classDef warn fill:#ffedd5,stroke:#ea580c,color:#0f172a
    class P,R,O acc
    class F,X warn
```

KV load failure 不是计算结果只“过时”：它依赖的数据无效。`_update_requests_with_invalid_blocks()` 先扣除本步乐观 scheduled 数，扫描可能含外部 KV 的 prefix，遇到首个坏块把 computed 截到该块起点。例如已加载前 99 块、首个坏块索引 50，就只承认前 50 块，不能把本步生成 token 交付。同步 recompute 保留请求 RUNNING 和已分配 blocks，重算坏块及后缀；共享坏块只标一份重算，其它依赖请求仍跳过本步结果。fail 策略驱逐坏块及依赖后缀的 cache 身份，并统一返回 `FINISHED_ERROR`。

异步 load 尚未缓存，recompute 把请求记入失败接收集合，等传输真正结束才缓存成功前缀并重新准入；一个有效位置也没有时释放原分配。当前 invalid-block 扫描直接解包单组 block IDs，源码留有 hybrid allocator 支持 TODO，因此不能把这套恢复规则宣称成任意 hybrid group 都已覆盖。对应测试分别验证同步保留 blocks、fail 驱逐污染 cache、异步不缓存坏块。

### 6.3 stop、终态和物理回收是不同完成点

实际输出逐 token 追加，按 EOS、stop token、模型长度/max tokens 的顺序检查，再经过 min_tokens 门槛判断配置的序列重复终止；触发后裁掉同一返回块里多余 token。pooling 有结果即停止；encoder-only 实例要消费完整 prompt 后才能结束，不能首个媒体项算完就结束。grammar 只推进真正需要约束的输出部分，拒绝实际 token 或编译失败走请求级 ERROR；文本 stop 字符串与协议 finish 的前端语义见 03，采样/grammar 算法见 [[02_engineering/03_infer_frameworks/vllm/14_vllm_sampling_structured_output_analysis|采样与结构化输出]]。部分 prefill 不产生用户采样输出，代码有相应断言。

结果确实执行后才 `_free_encoder_inputs()`；确认进度是 computed 减 placeholders，还须越过媒体末端与 drafter lookahead。对 encoder-decoder，decoder 已开始意味着 cross-attention KV 已缓存，可释放 encoder 输出。若 resumable 请求暂时结束，`_handle_stopped_request()` 会接续已排队的新输入或进入 WAITING_FOR_STREAMING_REQ；这时并非终态，不走最终释放。

外部 abort 的 `finish_requests()` 先移除 running/waiting/skipped 中有效请求，再设置终态并调用统一释放。`_free_request()` 通知 KV/EC connector、释放 encoder 引用、登记 finished ids；一般释放 blocks 并删除 request mapping。connector 要求 delay 时，对象已终止、不参与 admission，却仍驻留并持有 blocks，直到 receive/send 完成；producer 的 partial Mamba tail 还可能在 finalize/store 完成前继续保留，这个缓存细节归 08/22。

即使 connector 已允许 `_free_blocks()` 删除 request mapping，物理 blocks 仍可能等待执行 fence 才回池。该 defer gate 在当前生产路径是 **KV consumer connector 且 `max_concurrent_batches > 1`**，防止新 load 覆盖仍被旧 batch 写入的块，不是所有异步模式无条件延迟。`finished_req_ids` 用来让 worker 清镜像，不是“物理 blocks 已空闲”的证明；具体 Core 队列与 fence 时序已在 06 页重放。

## 7. 成本与可观察的边界

| 现象或约束 | 机制上的原因/后果 | 不能直接推出 |
|---|---|---|
| waiting 与 queue time 上升 | 到达工作未被 admission 及时消化，可能受 token、slot、KV 或异步依赖限制 | 不能只归因于 kernel 变慢 |
| KV usage 接近满且 preemption 增长 | 容量回收进入运行路径，已有请求可能需要重算 | 不等于必须关闭 prefix cache |
| ITL 上升、队列却稳定 | step 工作量、执行或 CPU/GPU overlap 可能变化 | 单项指标不能定位 runner/kernel |
| 局部候选为零但后来请求执行 | running `continue`、blocked waiting skip 放松严格先来先服务 | 不承诺所有请求公平，也不保证无饥饿 |
| 本步剩余 token 仍有请求未入场 | input draft reserve、slot、LoRA、encoder、对齐或队首不可切分工作也能阻塞 | token 预算不是唯一容量 |
| spec / async 开启后状态复杂 | rejection、placeholder、stale 跨多个已发 step 对账 | 更多 overlap 不等于无同步或必然更快 |
| 某批回到 piecewise/eager | 当前 batch/backend 不满足 full graph 条件 | 不等于编译整体失效 |
| 用户结束但 blocks 未回池 | connector 与 in-flight fence 各有完成条件 | finished 状态不是物理回收完成 |

表中定位是分析推断。源码分别记录 running、waiting、skipped waiting、KV usage、preemption、TTFT、ITL 和排队时刻；这些指标需要一起解释，定义与排查见 [[02_engineering/03_infer_frameworks/vllm/23_vllm_observability_reliability_analysis|可观测性与可靠性]]。配置还拒绝 `max_num_batched_tokens < max_num_seqs`，以及关闭 chunked prefill 时 batch token 上限小于最大模型长度的组合，避免配置本身令长请求无从准入。

调度、分页、async 与 graph 不是必须同时开启的一组开关。它们可以独立配置或回退；共同参与时则要在同一 token 进度和容量边界保持一致：KV 不足改写当步计划，placeholder 改变未返回进度，graph dispatcher 按最终 batch 形状挑路径。本页能证明的是 Scheduler 在其逻辑资源视角内按这些规则形成计划与更新状态，不是 GPU 无故障、采样分布正确或 KV 永无碎片的保证；设备与 graph 能力另见 [[02_engineering/03_infer_frameworks/vllm/19_vllm_compilation_cudagraph_analysis|编译与 CUDA Graph]]。

## 8. 源码阅读路线

以下均为本基线实际打开的定位符；测试只静态阅读，没有声称在本机运行通过。

1. 状态与计划：`vllm/v1/request.py::Request`、`RequestStatus`；`vllm/v1/core/sched/output.py::SchedulerOutput`，先区分数量、状态与每步 maps。
2. 从主例重放完整循环：`vllm/v1/core/sched/scheduler.py::Scheduler.schedule`；`tests/v1/core/test_scheduler.py::test_schedule_order`、`test_schedule_partial_requests`，核对 running-first 与 break/continue。
3. 预算和配置前提：`vllm/config/scheduler.py::SchedulerConfig.verify_max_model_len`；`vllm/config/speculative.py::SpeculativeConfig.use_eagle`、`SpeculativeConfig.use_eagle_block_drop`；预算字段与开关不互相代替。
4. encoder / MTP 边界：`vllm/v1/core/sched/scheduler.py::Scheduler._try_schedule_encoder_inputs`、`Scheduler._reserve_prefill_lookahead`、`Scheduler._free_encoder_inputs`，连读调度与回收 shift。
5. spec 行数：`tests/v1/core/test_scheduler.py::test_no_spec_tokens_scheduled_for_prefill_chunks`、`test_spec_decode_padding_first_decode_step`、`test_spec_decode_padding_dropped_when_recurrent_alignment_clips`，验证 50/30/4 与残缺 padding 的修正。
6. Mamba split 与共同校验：`vllm/v1/core/sched/scheduler.py::Scheduler._mamba_block_aligned_split`；`vllm/v1/kv_cache_interface.py::is_mamba_prefill_checkpoint_valid`、`get_mamba_prefill_checkpoint_position`；`tests/v1/core/test_mamba_align_chunk_split.py::test_partial_checkpoint_resume_stops_at_mamba_block_boundary`、`test_disabling_eagle_block_drop_keeps_the_trailing_cache_boundary`。
7. checkpoint 的另一侧：`vllm/models/kimi_k3/nvidia/kda_metadata.py::KimiK3KDAMetadataBuilder.build`；`vllm/v1/core/single_type_kv_cache_manager.py::MambaManager._needs_internal_checkpoint`，验证层 spec 块大小与 initial/checkpoint 槽限制。
8. 抢占与乐观推进：`vllm/v1/core/sched/scheduler.py::Scheduler._preempt_request`、`Scheduler._update_after_schedule`；`tests/v1/core/test_scheduler.py::test_priority_scheduling_preemption`；`docs/design/metrics.md` 的 Removed Metrics，确认 recompute 取代旧 swap。
9. 异步结果：`vllm/v1/core/sched/async_scheduler.py::AsyncScheduler._update_after_schedule`、`AsyncScheduler._update_request_with_output`；`vllm/v1/core/sched/scheduler.py::Scheduler.update_from_output`；`tests/v1/core/test_async_scheduler.py` 的 KV-pressure、reset-prefix-cache 与 mid-handoff 测试。
10. 远端 KV：`vllm/v1/core/sched/scheduler.py::Scheduler._update_requests_with_invalid_blocks`、`Scheduler._handle_invalid_blocks`、`Scheduler._update_waiting_for_remote_kv`、`Scheduler._try_promote_blocked_waiting_request`；`tests/v1/kv_connector/unit/test_invalid_blocks_correctness.py` 的 sync recompute/fail 与 async recompute 测试。
11. 完成：`vllm/v1/core/sched/scheduler.py::Scheduler.finish_requests`、`Scheduler._free_request`、`Scheduler._free_request_blocks`、`Scheduler._update_from_kv_xfer_finished`；`vllm/v1/core/sched/utils.py::check_stop`；`tests/v1/core/test_deferred_block_free.py`，与 06 页的完成时序对照。
12. 观察成本：`vllm/v1/metrics/stats.py::SchedulerStats`、`IterationStats.update_from_output`、`IterationStats.update_from_events`；配置/设备动态 graph 的前提接续 19、23 页。

## Related Pages

- [[02_engineering/03_infer_frameworks/vllm/02_vllm_architecture_overview_analysis|vLLM 架构概览]] — 把本页每步调度放回请求入口、资源控制与设备执行的完整路径。
- [[02_engineering/03_infer_frameworks/vllm/06_vllm_engine_architecture_analysis|vLLM Engine 架构]] — 解释谁提交计划、保存 future、按顺序回传结果，以及 Core 与前端的不同完成点。
- [[02_engineering/03_infer_frameworks/vllm/08_vllm_kv_cache_management_analysis|vLLM KV Cache 管理]] — 展开 `allocate_slots` 后面的 blocks、prefix cache、CoW 与延迟回收算法。
- [[02_engineering/03_infer_frameworks/vllm/11_vllm_model_runner_v1_analysis|Model Runner V1]] / [[02_engineering/03_infer_frameworks/vllm/12_vllm_model_runner_v2_analysis|Model Runner V2]] — 对照本页计划如何变成 compact/stable row 和设备输入，说明真正的异步执行约束。
- [[02_engineering/03_infer_frameworks/vllm/16_vllm_speculative_decoding_analysis|vLLM 投机解码]] — 深入本页只计算数量与回退的 draft/verify/accept 正确性。
- [[02_engineering/03_infer_frameworks/vllm/22_vllm_disaggregated_kv_serving_analysis|vLLM 分离式 KV Serving]] — 展开 remote-KV 等待、加载失败与 connector 延迟释放协议。
