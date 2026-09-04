---
title: "Megatron-LM 跨轴通信掩盖：时间线、资源竞争与诊断"
---

# Megatron-LM 跨轴通信掩盖：时间线、资源竞争与诊断

> **源码基线**：`NVIDIA/Megatron-LM@85902ef599ea4eb06ada7567a479c524b605767a`（`dev`，2026-09-01）。
> **阅读前置**：先从 [[12_megatron_tp_analysis]]、[[13_megatron_cp_analysis]]、[[14_megatron_ep_analysis]]、[[15_megatron_pp_schedulers_analysis]]、[[16_megatron_distributed_optimizer_analysis]] 中选择已开启并行轴的本地机制。
> **回答的问题**：TP/CP/EP/PP/DP 都可异步通信时，哪些窗口能同时存在，哪些 stream 会争抢同一资源，以及“开关已开但吞吐没变”应如何定位？
> **所有权**：本页只拥有跨轴时间线、PP×DP 对齐接口、stream/网络/显存竞争、组合失效条件与诊断顺序。五个轴的完整本地实现归 12–16，本页不复制它们的代码、通信量推导或配置全表。
> **最近更新**：2026-09-03。将原逐轴实现大全收缩为跨轴组合 owner；轴内独有增量已并回 12–16。

---

## 1. 背景：五个 overlap 都成立，不等于一起开就能叠加

单轴页回答的是“这次 collective 何时发起、在哪里等待”。真实的 Megatron step 却可以同时包含 TP AG/RS、CP KV 搬运、EP A2A、PP P2P，以及 DP 梯度 RS/参数 AG。每条路单独看都有合法的并发窗口，但组合后仍有三个上限：

1. **依赖上限**：只有不消费该通信结果的计算才能做掩盖物；
2. **资源上限**：不同 CUDA stream 仍可以争抢 SM、copy engine 和同一物理链路；
3. **头尾上限**：第一段通信之前没有前驱计算，最后一段之后没有后继计算，这些 exposed tail 无法靠“多开一个异步开关”消失。

因此评估的对象不是开了几个 flag，而是最终 critical path 上还露出多少通信、多少计算被通信拖慢，以及为了保持在飞状态多付了多少显存。

### 1.1 五个轴的本地 owner（一行导航）

| 轴 | 本地触发点 → 等待点 | 唯一机制 owner |
|---|---|---|
| TP | TE user buffer 在 linear/GEMM 内做 pipelined 或 bulk AG/RS，依赖结果前收口 | [[12_megatron_tp_analysis]] §4.2 |
| CP | TE 内核按 attention 块调度；原生 eager fallback 按 KV head 双缓冲 | [[13_megatron_cp_analysis]] §3 |
| EP | dispatcher 负责 token A2A；共享专家与 A2A stream 旋钮也属 EP | [[14_megatron_ep_analysis]] §5–§8 |
| PP | VPP 调度槽中延迟 P2P handle 的 `wait`；combined-1F1B 调度跨 microbatch 的 F/B | [[15_megatron_pp_schedulers_analysis]] §6–§7 |
| DP | backward hook 在 bucket ready 时发梯度 RS/AR，forward pre-hook 等当前参数 AG 并预取下一桶 | [[16_megatron_distributed_optimizer_analysis]] §3.2–§3.7 |

本表是导航，不是第六份机制说明。以下只讨论它们的交界面。

---

## 2. 为什么这么设计：重叠窗口位于不同抽象层，统一“先发后等”不足以表达

五条路并不共享一个异步通信框架：

- TP 的依赖位于一次 GEMM 内部，Megatron 只初始化 TE user buffer 并传参（`megatron/training/initialize.py:188-249`；`megatron/core/extensions/transformer_engine.py:815-848`）。
- CP 的并发单位是 attention 块或 KV-head slice，原生路径的 `wait→swap→prefetch next→compute current` 循环在 `megatron/core/transformer/dot_product_attention_context_parallel.py:181-224`。
- EP combined-1F1B 要用另一个 microbatch 的 attention/MLP 掩盖 A2A，其 docstring 明确写出“一个 microbatch 的 forward 与另一个的 backward 并行”（`megatron/core/pipeline_parallel/combined_1f1b.py:51-64`）。
- PP 的窗口在 stage 边界，由 VPP 稳态循环保存 P2P request，到数据被消费或源缓冲被释放时才等待（`megatron/core/pipeline_parallel/schedules.py:1849-2180`）。
- DP 的窗口位于反向 bucket 就绪序与前向 bucket 消费序，不是 layer 中的某个 GEMM（`megatron/core/distributed/param_and_grad_buffer.py:611-680`）。

> [!note] 本页的架构归纳
> 源码分别呈现了上述五种载体，但没有一处自述“为什么不建统一 overlap framework”。本页据依赖所在的抽象层重建该结论：GEMM chunk、attention slice、microbatch pair、pipeline slot 与 DDP bucket 不能共用同一个 ready/wait 协议。这是有源码锚点的推断，不是作者的设计自述。

### 2.1 判断两条 overlap 能否组合的四个问题

| 问题 | 必须说清的状态 | 如果答不出来 |
|---|---|---|
| 什么时候 ready？ | 是 GEMM chunk、head、layer、microbatch、stage 还是 bucket 就绪 | 无法确定真正的 dispatch 点 |
| 谁会消费结果？ | 找到首个需要输出的算子/调度槽 | 无法确定最晚的 wait 点 |
| 用什么掩盖？ | 掩盖计算不能依赖未完成的通信结果 | 所谓 overlap 只是换了等待位置 |
| 两者争什么？ | 区分进程组/CUDA stream 与物理 NIC、NVLink、SM、缓冲 | 容易把“并发排队”误读为“并行执行” |

---

## 3. 一个训练 step 中的跨轴候选时间线

下表是**依赖顺序**，不是假设每个作业都开启五轴的固定 trace。具体分支由 PP/VPP、是否 MoE、`cp_comm_type` 与分片策略决定。

| 阶段 | 前台工作 | 可能在后台的通信 | 必须收口的边界 |
|---|---|---|---|
| 1. 进入新 chunk / 新前向 | 上一个 model chunk 或 optimizer 尾部工作 | DP 参数 AG 预取 | 该 module 的 forward pre-hook 在读参数前 `finish_param_sync` |
| 2. dense attention / MLP 前向 | 当前 layer 的 GEMM/attention | TP user-buffer AG/RS；CP 的下一个 attention 块或 KV-head AG | 当前 GEMM/head 真正需要对应输入时 |
| 3. MoE 层稳态 | microbatch `i` 的 backward attention/MLP 与 `i+1` 的 forward 子节点 | 对侧 microbatch 的 EP dispatch/combine A2A | 专家 GEMM 消费 dispatched token、或 residual 消费 combined output 之前 |
| 4. PP stage 边界 | 下一个 forward/backward 调度槽 | 上一槽的 activation/gradient P2P | 接收张量被计算消费、或异步 send 的源存储被释放之前 |
| 5. transformer backward | 从输出向输入的 dgrad/wgrad | 已就绪 DP bucket 的 RS/AR；TP 反向 AG/RS | 下一个依赖梯度的计算或 step 尾 `finish_grad_sync` |
| 6. PP cooldown / 梯度收尾 | 剩余 backward 槽与未同步 model chunk | 对齐后的 DP grad reduce、剩余 P2P | 调度器显式启用梯度同步并派发遗留 chunk（`megatron/core/pipeline_parallel/schedules.py:2162-2171`） |
| 7. optimizer 边界 | 梯度校验与参数更新 | 不应再有未被 owner 持有的必需梯度通信 | optimizer 读梯度之前所有必需同步完成；step 内部归 [[26_megatron_optimizer_step_internals_deepdive]] |

这条时间线的关键不是“越早 dispatch 越好”。PP 调度器的注释反而明说异步通信往往会拖慢计算，因此要在 PP group 上对齐参数/梯度通信，减少 microbatch 时间不匹配导致的空等（`megatron/core/pipeline_parallel/schedules.py:1438-1455`、`:1540-1557`）。

### 3.1 PP×DP：“对齐”是调度回调，不是额外 collective

训练接线层只在模型是 Megatron-FSDP/DDP 且 `overlap_grad_reduce=True` 时接管 `no_sync_func`；若另有自定义 `no_sync_func` 会被 `assert` 拒绝（`megatron/training/training.py:4371-4378`）。当 `align_grad_reduce=True` 时，它再把各 model chunk 的 `start_grad_sync` 绑到 `config.grad_sync_func`；参数侧对称地在 `overlap_param_gather and align_param_gather` 时绑定 `start_param_sync`（`:4379-4386`）。

回调最终由 PP 时间表调用：前向预处理在各 stage 对齐地预取后续 chunk 参数（`megatron/core/pipeline_parallel/schedules.py:1436-1455`），反向后处理在对齐槽中调 `grad_sync_func`（`:1540-1557`）。所以：

- `align_grad_reduce` 决定 DP reduce 由各 stage 自行抢跑，还是交给 PP 时间表对齐派发；实际行为由上述 training wiring 与 schedule 调用点共同确定。
- `grad_sync_func` 是 PP 调度器反向调 DP 同步的接口，它自身不创建 bucket 或 collective（`megatron/core/model_parallel_config.py:207-211`）。
- 参数侧的 bucket/AG、`align_param_gather` 和 `param_sync_func` 仍归 [[16_megatron_distributed_optimizer_analysis]]；本页只拥有它们与 PP schedule 的握手。

> [!contradiction] `align_grad_reduce` 字段注释的方向疑似反写
> `megatron/training/config/common_config.py:83-86` 给出默认值 `True`，却写成“if not set”时所有 PP stages 同时 reduce、“otherwise”各自发起；这与执行接线相反：只有 `args.align_grad_reduce=True` 才注入 `grad_sync_func`（`megatron/training/training.py:4379-4382`），随后 PP schedule 在对齐槽调用它（`megatron/core/pipeline_parallel/schedules.py:1540-1557`）。因此本页以 wiring + schedule 作为当前行为证据；字段声明只证明默认值和这处源码内部冲突。

### 3.2 TP×CP：CP 先改变本地 token shape，TP user buffer 再据此初始化

TP user buffer 的第一维不是全局 `seq_length × micro_batch_size`，而是再除以 `context_parallel_size`（`megatron/training/initialize.py:211-220`）。这是一个真正的跨轴数据契约：CP 决定每个 rank 的 token shape，TP overlap 用该 shape 预注册 TE 缓冲。它不意味着 TP collective 与 CP collective 会自动彼此掩盖；只能说两者在形状上已正确接线。TP user-buffer 的完整契约见 [[12_megatron_tp_analysis]] §4.2。

### 3.3 PP×EP×FSDP：细粒度 schedule 绕开 wrapper 时，分片生命周期必须补回去

combined-1F1B 直接调度 layer 子节点，会绕过 Megatron-FSDP wrapper 的常规 forward hook。无 PP 宿主在进入 schedule 前显式 `_replace_param_with_raw_if_needed()`（`megatron/core/pipeline_parallel/combined_1f1b.py:67-74`），并由 layer schedule plan 补挂 reshard hook（`megatron/core/models/common/model_chunk_schedule_plan.py:377`）。而 VPP 多 chunk 路径尚未处理这个生命周期，因此显式拒绝 `virtual_pipeline_model_parallel_size > 1` + FSDP + EP overlap（`megatron/core/pipeline_parallel/combined_1f1b.py:203-214`）。

这里的通用判据是：**某个 overlap schedule 若绕过参数/梯度 wrapper，必须重新审计 all-gather、release、pre-backward 和 post-backward 的 owner**。FSDP 的完整状态机归 [[36_megatron_fsdp_analysis]]，本页只记它与 combined schedule 的断面。

### 3.4 TP×EP：普通 linear 的 UB overlap 不自动适用于 expert linear

TE 桥接在 `is_expert` 时显式将 user-buffer pipelined AG/RS 关闭（`megatron/core/extensions/transformer_engine.py:829-843`）。这阻止了一个常见误读：同一层里的 dense attention TP overlap 成立，不代表 MoE expert tensor parallel 会复用同一条 UB 路径。EP/ETP 通信应回到 [[14_megatron_ep_analysis]] 和 combined schedule 判断，而不是在全局 trace 中把两条线合并。

---

## 4. 资源竞争：stream 分开了依赖队列，没有凭空增加硬件

### 4.1 源码能直接证明的三个竞争信号

1. PP 调度器两处注释都说“asynchronous communication tends to slow down compute”，并以跨 PP stage 同时派发来降低不匹配空等（`megatron/core/pipeline_parallel/schedules.py:1438-1455`、`:1540-1557`）。
2. DDP 在 `CUDA_DEVICE_MAX_CONNECTIONS=1` 下会合并连续的小 bucket，注释给出的原因是多个 back-to-back communication kernels 会阻碍与 compute kernel 的重叠（`megatron/core/distributed/param_and_grad_buffer.py:1811-1818`）。
3. EP 允许将 A2A communication stream 设为 CUDA 高优先级：combined schedule 在两个宿主中透传字段（`megatron/core/pipeline_parallel/combined_1f1b.py:67`、`:203`），`set_streams` 以 `torch.cuda.Stream.priority_range()` 创建该流（`megatron/core/pipeline_parallel/utils.py:350-362`）。完整旋钮归 [[14_megatron_ep_analysis]] §8.4。

> [!note] 运行时推断
> 上述代码能证明“异步通信可拖慢计算”、“连续通信 kernel 可破坏 overlap”与“A2A 可提高 stream priority”。由此可推得：进程组不同或 CUDA stream 不同，只表示调度队列可独立推进，不表示它们拥有独立 NIC/NVLink/SM。具体争用哪项硬件必须以目标机器的 profiler trace 为准，源码没有对任意拓扑做这个保证。

### 4.2 三类不能靠更多 stream 解决的瓶颈

| 瓶颈 | trace 特征 | 优先检查 |
|---|---|---|
| 带宽饱和 | 多条 collective 时间重叠，但单条的 duration 同时拉长，compute 也被拖慢 | TP/CP/EP/PP/DP 的进程组是否落在同一跨节点物理链路；先只保留一条跨机 overlap 做对照 |
| kernel 过碎 | 大量小 collective 背靠背，中间没有计算 kernel 取得进展 | DP 的 bucket/`num_buckets`、PP 的 P2P 粒度、是否在单连接顺序下造成通信连发 |
| 在飞缓冲过多 | 吞吐略升或不变，但峰值显存上升/OOM | PP recv buffer、TP user buffer、CP KV 双缓冲、EP 跨 microbatch 激活与 DP RS/AG 中间状态是否在同一时刻存活 |

DP 已给出一个可复用的治理模式：fp32-accumulation RS 在派发新 bucket 前排空已发起的前驱 bucket，从而给中间 all-to-all 输出的在飞数量设上界（`megatron/core/distributed/distributed_data_parallel.py:350-365`；`megatron/core/distributed/param_and_grad_buffer.py:665-680`）。完整机制归 [[16_megatron_distributed_optimizer_analysis]] §3.7。

### 4.3 显存与暴露通信是同一个旋钮的两面

`ep_overlap_early_attn_memory_release` 的 docstring 直接写明这笔交易：EP overlap 可在前向 module 分配多于反向 module 释放的显存时抬高峰值；把 attention backward 提前能更早释放激活，但会使 `moe_combine_fwd` 与 `moe_dispatch_bwd` 重新暴露（`megatron/core/model_parallel_config.py:355-366`）。该开关的调度 owner 是 [[15_megatron_pp_schedulers_analysis]] §7；本页只用它说明全局优化目标必须同时包含 wall-clock 和 peak memory。

---

## 5. 组合开关：先证明单轴有效，再每次增加一条交界

### 5.1 轴内开关仍由轴内 owner 解释

| 轴 | 最小激活链 | 不在本页重复的细节 |
|---|---|---|
| TP | `sequence_parallel` → `tp_comm_overlap` → 所需 bulk/pipelined 子开关 | TE user-buffer 命名、GEMM 依赖与所有 `tp_comm_*` 契约见 [[12_megatron_tp_analysis]] |
| CP | `context_parallel_size>1` + 与 backend/模型匹配的 `cp_comm_type` | p2p/all_gather/a2a/a2a+p2p 及 eager fallback 见 [[13_megatron_cp_analysis]] |
| EP | `expert_model_parallel_size>1` + dispatcher/backend；需 combined 时再开 `overlap_moe_expert_parallel_comm` | dispatcher、shared expert、A2A priority 见 [[14_megatron_ep_analysis]]；combined schedule 见 [[15_megatron_pp_schedulers_analysis]] |
| PP | VPP 宿主 + `overlap_p2p_comm=True` + `batch_p2p_comm=False` | request 生命周期、warmup/steady/cooldown 与 recv buffer 见 [[15_megatron_pp_schedulers_analysis]] |
| DP | `overlap_grad_reduce`；需参数预取时再开 `overlap_param_gather` | bucket 分组/排序、ZeRO 分片和完成点见 [[16_megatron_distributed_optimizer_analysis]] |

### 5.2 仅有两个配置实体由本页拥有

跨轴对齐需要一个控制位和一个回调槽：`align_grad_reduce` 选择是否让 PP stages 同时发梯度 reduce，`grad_sync_func` 让 PP 时间表能调用 DP 的异步梯度同步。它们的字段契约见 §10。

`delay_wgrad_compute` 与 `ep_overlap_early_attn_memory_release` 不再归本页：它们改变 combined-1F1B 的局部节点顺序，owner 是 [[15_megatron_pp_schedulers_analysis]]。`high_priority_a2a_comm_stream` 也不归本页：它是 EP A2A stream 资源旋钮，owner 是 [[14_megatron_ep_analysis]]。

### 5.3 建议的启用顺序

1. 保留一份所有 overlap 都关闭的可复现 baseline，记录 step time、峰值显存和主要 collective duration。
2. 只开一个轴，在 trace 中确认 dispatch 时机、实际并发区间与 exposed tail。
3. 对该轴调到“计算能推进、collective 也不变成延迟小包”的粒度。
4. 加入第二轴，对比两条 collective 的 duration 和同期 compute duration 是否同时拉长。
5. 只在 PP×DP 组合中调对齐点；不要用 `align_grad_reduce` 去试图修复 TP/EP/CP 的局部窗口。
6. 最后复查 peak memory，确认吞吐收益没有被 OOM、更激进重计算或更小 microbatch 抵消。

---

## 6. 硬约束：先区分“不能跑”与“能跑但没收益”

### 6.1 不满足就报错或被强制改写

| 组合 | 硬门 | 证据 |
|---|---|---|
| TP overlap | 必须开 SP，且初始化环境必须可导入 TE 与 YAML | `megatron/training/arguments.py:1542-1545`；`megatron/training/initialize.py:192-201` |
| DP param gather | 需 distributed optimizer、Megatron-FSDP 或 `dist_muon` 之一，且必须同时开 grad reduce overlap | `megatron/training/arguments.py:1077-1085` |
| PP P2P overlap | 只在 interleaved/VPP schedule 启用；`batch_p2p_comm` 与它互斥 | `megatron/training/arguments.py:1049-1071`；`megatron/core/model_parallel_config.py:380-388` |
| EP combined overlap | torch ≥ 2.6、EP>1、dispatcher 为 `alltoall`/`flex`；PP>1 时必须有 VPP | `megatron/core/transformer/transformer_config.py:3583-3600` |
| delayed wgrad | `delay_wgrad_compute` 必须与 combined overlap 同开；它与 `overlap_dispatch_backward_with_experts_wgrad` 互斥 | `megatron/core/transformer/transformer_config.py:3712-3734` |
| early attention release | 必须已开 combined overlap | `megatron/core/transformer/transformer_config.py:3736-3740` |
| dynamic/hybrid CP × PP | 当前 schedule 明写 per-token loss + no PP，混合 CP 调度器仍留有 PP TODO | `megatron/core/pipeline_parallel/schedules.py:387-390`；`megatron/core/pipeline_parallel/hybrid_cp_schedule.py:190` |
| EP overlap × VPP × FSDP | VPP 多 chunk 路径显式 assert 不支持 | `megatron/core/pipeline_parallel/combined_1f1b.py:203-214` |

### 6.2 正确性仍成立，但 overlap 可能实际为零

| 症状 | 已知的静默/警告型原因 | 第一个核对点 |
|---|---|---|
| PP 开关在 CLI 中设了，trace 里仍是同步 P2P | 没有 VPP 时 arguments 会将 `overlap_p2p_comm=False` 并打 warning | `megatron/training/arguments.py:1061-1071` |
| DP 参数 AG 总在 module 前停顿 | 参数注册序与实际 forward 序不一致，下一桶可被错误预取，源码会警告它损害 overlap | `megatron/core/distributed/param_and_grad_buffer.py:638-646` |
| TP 通信与 wgrad 串行 | `CUDA_DEVICE_MAX_CONNECTIONS` 未按预期安排 kernel 下发；该要求是收益必需、不是正确性必需 | `megatron/core/tensor_parallel/layers.py:699-706`、`:761-775` |
| 某个 TP linear 不再出现 UB 并发 | user-buffer name 不在支持集合时，TE 桥接 warning 后关闭该层 overlap | `megatron/core/extensions/transformer_engine.py:805-813` |
| NCCLEP combined 正确但 A2A 仍全暴露 | `moe_ncclep_static_shape=False` 导致 device-to-host sync 串行化 1F1B，源码明确 warning “loses the overlap benefit” | `megatron/core/transformer/transformer_config.py:3672-3685` |
| 新开一轴后通信和计算都变慢 | 异步 work 已排队，但物理资源过载；这不是开关未生效 | 对比增量 trace 中 collective duration 和同期 GEMM duration，不只看是否交叠 |
| 有明显交叠但 step time 几乎不变 | 可掩盖的独立计算窗口小于通信，头/尾仍在 critical path | 分别计算 dispatch→首个 compute、真正交叠段、最后一个 wait 的时长 |

表中最后两行是 profiler 层的运行推断，不是 Megatron 对某种硬件的固定保证。

---

## 7. 诊断梯子：从“分支有没有激活”走到“关键路径缩短了多少”

### 7.1 第一层：先证明配置合法

按 §6.1 检查 assert/ValueError 链，再搜索运行时 warning。不要先从 NCCL 带宽推断：若非交错 PP 已把 P2P overlap 强制关闭，或 NCCLEP 被 D2H sync 串行化，任何网络调参都不会创造窗口。

### 7.2 第二层：在 trace 里找到 dispatch、overlap 和 wait 三个点

对每条开启的路径只回答三个问题：

- collective 是否在预期的 layer/head/microbatch/stage/bucket ready 点发起？
- dispatch 与 wait 之间是否有**不依赖通信输出**的 compute kernel 获得实际执行？
- 最后一个 wait 后面是谁继续占据 critical path？

只看 CUDA stream 上两段色块水平重合是不够的。如果 GEMM 和 collective duration 都比单轴 baseline 长，这是资源争用，不是额外收益。

### 7.3 第三层：用增量实验隔离冲突轴

| 对照 | 要回答的问题 |
|---|---|
| all off → 只开 A | A 是否真正缩短 step，其显存成本多大？ |
| 只开 A → A+B | B 是否缩短新的 exposed tail，还是把 A 和 compute 同时拖慢？ |
| A+B → A+B+对齐 | PP×DP 的 stage skew 是否下降，是否只是把通信峰值挪到了另一时刻？ |
| 吞吐最优 → 显存安全版 | 降低在飞数或提前释放后，增加的 exposed communication 是否小于避免 OOM/重计算的收益？ |

### 7.4 第四层：再调粒度和优先级

先调 DP bucket / PP microbatch 这类决定窗口大小的参数，再调 A2A high-priority stream 或 HybridEP SM 预算这类决定资源分配的参数。如果前三层还没证明存在独立计算窗口，提高 stream priority 只会改变谁等谁，不会减少必须完成的工作。

---

## 8. 典型组合的阅读与调试入口

| 场景 | 先稳定的本地窗口 | 再审计的交界 |
|---|---|---|
| Dense TP×PP×DP | 12 的 TP UB，15 的 VPP/P2P，16 的 bucket | PP stage 的 DP grad/param 对齐；跨机 P2P 与 DP collective 是否互相拉长 |
| 长上下文 TP×CP×DP | 13 的 `cp_comm_type`，12 的 CP-aware UB shape，16 的 bucket | TP/CP 是否同时占用高频链路；KV buffer + TP UB + DP 在飞状态的峰值 |
| MoE EP×PP×DP | 14 的 dispatcher/backend，15 的 combined schedule，16 的 regular/expert-DP bucket | A2A、P2P 与 DP reduce 的 duration 是否一起拉长；是否需要 stage 对齐 |
| MoE EP×FSDP | 14 的 EP 本地路径，36 的 FSDP unit 状态机 | combined schedule 是否绕过 wrapper hook；当前是否命中 VPP>1 显式禁用边界 |
| dynamic/hybrid CP | 13/29 的动态分组与 packed sequence | 先保持 no PP 的已支持边界，不把它与 PP overlap 强行组合 |

---

## 9. 不变量、代价与故意不做的事

### 9.1 不变量

- **Overlap 不自动减少通信量。** 它改的是 dispatch/wait 时机。若同时换了 CP 算法、EP dispatcher 或 DP 分片策略，通信量的变化应归新算法，不应算给 overlap。
- **每个异步路径必须有唯一的 completion owner。** TP 在 TE linear，CP 在 attention 循环，PP 在 request 生命周期，EP 在 schedule node，DP 在 bucket group。全局页不新增第二个 `wait`。
- **进程组是正确性域，不是带宽预留。** 两个 collective 在不同 group 上合法，仍可能经过同一物理链路。

### 9.2 代价

- 异步发起需要 request、event、双缓冲或预取 buffer 活得更久，峰值显存会上升。
- 重叠路径对调度顺序更敏感：DP 参数注册序错配会损害预取，TP 需要预期的 kernel 下发顺序，PP 需要保留 send 源存储到 request 完成。
- 提前释放显存可能故意缩小 overlap 窗口；正确的目标是可运行约束下的稳定 step time，不是事件图上最大的彩色重叠面积。

### 9.3 故意不做

- 本页不提供一份“所有 overlap 全开”的通用配置，因为拓扑、模型 shape、后端和显存余量都是判据。
- 本页不复制 TP/CP/EP/PP/DP 的代码摘录；如果诊断需要追一个 handle，应回到 §1.1 的 owner。
- 本页不绕过当前明确的不支持组合：dynamic/hybrid CP × PP 与 EP overlap × VPP × FSDP 都应等源码边界改变后再重新评估。

---

## 10. 配置契约：跨轴梯度对齐的两个接口

### `DistributedInitConfig`（`megatron/training/config/common_config.py`，1 项）

| 字段 | 类型 | 默认 | 契约 | 行 |
|---|---|---|---|---|
| `align_grad_reduce` | `bool` | `True` | 开启时注入 `grad_sync_func`，让 PP schedule 在对齐槽发梯度 reduce；关闭时不注入。字段 docstring 的方向与执行接线冲突，见 §3.1。 | `:83`；行为见 `megatron/training/training.py:4379-4382` 与 `megatron/core/pipeline_parallel/schedules.py:1540-1557` |

### `ModelParallelConfig`（`megatron/core/model_parallel_config.py`，1 项）

| 字段 | 类型 | 默认 | 契约 | 行 |
|---|---|---|---|---|
| `grad_sync_func` | `Optional[Callable]` | `None` | 发起异步梯度同步（例如 distributed-optimizer reduce-scatter）的回调；接收待同步参数迭代器，由 PP schedule 在对齐槽调用。 | `:207-211` |

> 所有 TP 本地 overlap 字段已归 [[12_megatron_tp_analysis]]；`high_priority_a2a_comm_stream` 已归 [[14_megatron_ep_analysis]]；`delay_wgrad_compute` 与 `ep_overlap_early_attn_memory_release` 已归 [[15_megatron_pp_schedulers_analysis]]。完整的唯一 owner 映射见 `docs/coverage/megatron-lm.yaml`。

---

## 11. 发展趋势

> [!note] 推断
> 以下是从当前守卫、新配置与 TODO 归纳的工程方向，不是 Megatron 的公开路线图。

- **Overlap 的评价从“通信是否异步”转向“在飞状态是否有界”。** DP 前驱 RS 排空（`megatron/core/distributed/param_and_grad_buffer.py:665-680`）和 EP 早释放 attention 激活（`megatron/core/model_parallel_config.py:355-366`）都在为峰值显存付出部分并发。
- **资源控制从 boolean 开关向优先级与预算细分。** `high_priority_a2a_comm_stream` 和 HybridEP 的 SM 预算将“能不能并发”拆成“并发时谁先取得资源”；但收益仍必须在具体拓扑上测量。
- **跨轴难点越来越集中在 wrapper/schedule 接口。** PP×DP 依赖 `grad_sync_func`，EP×FSDP 需补 reshard hook，dynamic CP×PP 仍留 TODO；后续审计应先查这些边界，而不是重新遍历所有 collective。

---

## Related Pages

- [[12_megatron_tp_analysis]] —— TP AG/RS、SP、TE user buffer 与全部 `tp_comm_*` 配置的本地 owner。
- [[13_megatron_cp_analysis]] —— CP 四种通信类型、TE 透传与原生 eager all-gather 双缓冲的 owner。
- [[14_megatron_ep_analysis]] —— EP dispatcher、shared expert、A2A stream priority 与 SM 预算的 owner。
- [[15_megatron_pp_schedulers_analysis]] —— P2P request 生命周期、VPP 与 combined-1F1B 时间表的 owner。
- [[16_megatron_distributed_optimizer_analysis]] —— DP bucket、grad reduce、param gather 与前驱 RS 排空的 owner。
- [[36_megatron_fsdp_analysis]] —— Megatron-FSDP unit、hook、reshard 与双缓冲状态机，用于审计 EP overlap 接入边界。
- [[30_comm_compute_overlap_analysis]] —— 跨框架的通信-计算掩盖对比；本页只对 Megatron 的跨轴组合负责。
