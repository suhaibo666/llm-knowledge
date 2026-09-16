# vLLM 01–26 修复验收（2026-09-16）

对象是 [2026-09-15 全域评审](2026-09-15-vllm-domain-review.md) 之后的修复结果：26 篇正文、index 与 changelog 的工作区改动（+1725/−471，未提交）。验收由 10 个与写作无关的评审者执行，方式是逐条核对上一轮发现的状态，并把**新增文字**当成新稿重新核验——改稿是新错误的主要来源。协调者复核了全部 P0 和本轮新增的关键 P1，并亲自复现了构建失败。

- **源码**：`vllm-project/vllm@199cb9b964822e59ab9b58d88e7be31eb419a2ae`，只读，HEAD 未移动。
- **明细**：[`2026-09-16-vllm-domain-review-recheck/`](2026-09-16-vllm-domain-review-recheck/) 十份分组报告，含逐条状态表、新发现证据与建议。

## 0. 结论

| 维度 | 上一轮 | 本轮 |
|---|---|---|
| 页面判定 | PASS 5 / REJECT 21 | **PASS 或 ACCEPT 17 篇**（04–11、13–16、21–23、25、26）；**条件通过 4 篇**（12、17、18、20，各 1–2 条新 P1）；**REJECT 3 篇**（02、19、24）；**待规划裁决 2 篇**（01、03） |
| P0 | 7 条 | **全部修复并抽查属实**；本轮无新 P0 事实错误 |
| P1 | 约 60 条 | 修复约 45 条；**新增 10 条**（多为改稿引入）；**未修 10 条左右**，其中 8 条写作者按规范上交规划批准 |
| 互推无主功能点（§4.2 的 21 项） | 全部断链 | **17 项真正闭合**（owner 写出内容且推出方指过去），2 项部分，2 项待规划 |
| 域级约定 | 页头日期三种、24–26 锚点风格不一、index 箭头错 | **全部修好**：26 页统一 2026-09-07 UTC，24–26 收敛为 `path::symbol`，index 的 23/26/20-21 依赖方向与 MRV2 默认提示已改 |
| 机械门禁 | T0 四项全绿 | T0 四项仍全绿，但 **`python -m tools.mkdocs_site.cli build --changed` 失败** |

修复质量整体是高的：五组报告明确写“新增文字逐条回源码核验，无回归”，包括 13/18/25/26 四页直接从 REJECT 翻成 PASS，08 的六条 P1 全部实修（准入上限公式、HMA 解析六行表、hash 合同、partial hit 门、Mamba 1600 三步表、spec 可缓存尾部三分支都能按算例复放）。

## 1. 合并前必须处理（阻塞）

### B1 三处 wikilink 锚点不存在，站点构建中断

`check_links --strict` 报 0，但 `python -m tools.mkdocs_site.cli build --changed` 抛 `tools.mkdocs_site.wikilinks.LinkResolutionError` 并中止。构建是 all-or-nothing，`.github/workflows/pages.yml` 部署的正是这套栈，三处必须一起修：

| 位置 | 链接写的锚点 | 目标页实际标题 |
|---|---|---|
| 24 L120 | `14#4.3 扩展接口与不支持项` | `### 4.3 logits processor 的变体集合从哪里枚举出来` |
| 25 L315 | `13#3.1 EngineCoreClient 是一组实现，不是一个进程` | `### 3.1 进程、对象与状态归属` |
| 25 L403 | `26#6. Future、广播顺序与 FIFO 不变量` | `## 6. Future：响应没有请求 ID 时怎样保持配对` |

根因是 13 和 26 在本轮改了标题，而引用方沿用旧标题。**T0 查不出这一类**：`check_links` 的 `stale_section` 只看正文里紧邻的纯文本 `§N`，不解析 wikilink 内的 `#标题`。建议把“改动含带锚点链接或改过标题时补跑一次 `--changed` 构建”写进流程，并扩展 `check_links` 覆盖该类锚点（已另开任务）。

### B2 改稿引入的新 P1（10 条）

| # | 页 §节 | 问题 | 源码事实 |
|---|---|---|---|
| 1 | 20 §5.3、§13 | 新写的 `rocm_aiter_ops.do_fused_qk_norm_rope_and_cache` 在基线中不存在，§13 阅读路线锚点因此失效 | 实链为 impl `do_qk_norm_rope_kvcache_update` → `_aiter_ops.py::rocm_aiter_ops.do_qk_norm_rope_kvcache_update` → `fused_qk_norm_rope_and_cache` → AITER `fused_qk_norm_rope_cache_pts_quant_shuffle`（协调者已复核：`grep` 全库无 `do_fused_qk_norm_rope_and_cache`） |
| 2 | 20 §3.5 | 说 21 的 lowering pass 只把“实际选中实现”的 UUID 纳入 | `VllmIRLoweringPass.uuid` 对每个注册 op 的**全部** provider 求哈希；与 21 正确的表述冲突 |
| 3 | 24 §3、§4、§6.1 | 把 `vllm.logits_processors` 组的加载写成 MRV1 runner 独占 | 另有前端链：`InputProcessor._validate_params` → `SamplingParams.verify` 无条件 `_validate_logits_processors` → `_load_logitsprocs_plugins`，同样 `raise RuntimeError`，失败表现为请求校验期错误 |
| 4 | 17 §6.1 | 说普通 linear 参数“可能”豁免加载跟踪 | 09 §2.8 本轮已改成无条件豁免并指出 `UnquantizedLinearMethod` 覆写该 hook；同一事实两页相反 |
| 5 | 18（三处） | 写 `EngineCore.reinitialize_distributed` / `commit_prepared_elastic_ep` | 两者都在 `DPEngineCoreProc`（`vllm/v1/engine/core.py:2302/2356`，协调者已复核）；23 §6.3 写法正确，owner 页写错 |
| 6 | 12 L229 | 说 `align` 只缓存落在块边界上的步末 Mamba 状态 | 它委托的 08 §5.1.3 按 hash 边界记账，`_cache_partial_tail_block` 的 partial-tail 分支要求 `num_tokens % block_size != 0` |
| 7 | 12 L608 | `max_concurrent_batches` 写成“异步 MRV2 为 PP+1，否则 PP” | async + MRV1 且 PP≤1 时为 2（`config/vllm.py:570-577`），06 §4.1 新表正确 |
| 8 | 19 §1.2 图 1 | 仍画 sampled token 同址写回 `InputBuffers` | 实际写 `RequestState.last_sampled_tokens / all_token_ids`；`InputBuffers` 无 sampled 字段，与本轮加强后的 12 §2.6 冲突 |
| 9 | 19 §6.5 图 2 | 仍把 `resolve_cudagraph_mode_and_sizes` 画在首次 dummy forward 之前 | 实际顺序是 `profile_run`（首次编译）→ 最小 KV profiling（resolve 第一次）→ 真实 KV 初始化（resolve 第二次），与 12 §2.9 冲突 |
| 10 | 19 §4.3、§9.1 | 把 `compile_sizes` 必须匹配 capture ladder 的 `ValueError` 写成通用约束 | `CudagraphDispatcher` 只由 MRV1 与两个 drafter 构造，默认 MRV2 路径不生效 |

第 8–10 条是上一轮就提出、本轮未动的 P1，因此 19 维持 REJECT。

### B3 02 的架构检查仍未通过

02 的 P0 修得干净（默认 `step_with_batch_queue` 路径、两图、§2.4 调用树、§2.3 收尾全部对上源码），但 item 5 场景清单与 item 8 读者交接原样未动：

- `test_abort_defers_free` 仍写成普遍的延迟释放，与 06 §6.2 的门（KV consumer connector 且 `max_concurrent_batches > 1`）冲突；
- §5.3 仍把 gRPC / Rust / Omni 交给 13，而 13 在本轮 +283 行之后 `grpc`、`omni` 命中数仍为 0；
- 全页对 14 的链接仍是 0 次，15、16 在架构里没有落点；
- 缺 scale-out 场景（`VLLM_ENABLE_SCALE_OUT_ENDPOINTS`、`--tokens-only`、render → token-in 往返）；
- §5.6 仍未写 DP=2 默认起 2 个 API server 走 `run_multi_api_server`，也未链接 13。

## 2. 重复解释：唯一变差的一项

上一轮的 P1-7（stale output 与完成信号在 06、07 各写一遍）没有修，而且**变成三份**：新写的 07 §8.1.2 又复述了一遍 `finished_req_ids` 语义与“已发布集合不得 `clear()`”规则，其中两份在同一页。类似地，07 §2.1 现在完整复述了已由 06 §4.1 独占的 async 解析条件，且全页没有指向 §4.1 的指针；默认 `non_block` 提交序散落在 7 页，启动顺序图仍有 6–7 处。修 owner 缺口时同步删掉推出方的正文，比补一句链接更重要。

## 3. 按规范上交规划批准的项（不算失败，但仍开放）

写作者按 CLAUDE.md 的“不擅自改页面边界”把下列交给 `planning-codebase-analysis`，changelog 已记录：

- “调用 API / 使用特性”这一步的使用说明 owner（01 覆盖缺口、03 的分类）；因此 01 的使用说明判定仍 FAIL，03 作为分析页 PASS、作为使用说明 FAIL；
- P/D proxy 路由段、配置解析总链、Ray/external_launcher 监督与 Ray DP actor 生命周期、tool/reasoning parser、pooling 执行、非 CUDA 平台、gRPC/Rust frontend、`n>1`/beam search、spec 下 `logprob_token_ids`。

其中两处是页内可自修的部分，不必等规划：22 §2、§14「它不是什么」与 §14.4 仍断言 13 拥有 P/D 路由，同页末行又写“全库无人展开”，三处自相矛盾；23 L347 仍把 Ray/external-launcher 监督指向 18 和 13，两页都没有该内容。建议现在改成“本域暂无 owner，已提交规划”这类如实表述。

域级评审者另建议在 index「目录范围」加一句覆盖边界说明，否则这 8 项已知缺口对读者不可见。

## 4. 已确认修好的域级项

- 26 页页头基线统一为 `2026-09-07 UTC`（此前 19/20 写 09-08、21–23 写 09-06），`最近更新` 统一为 2026-09-15。
- 24–26 的稳定源码路线从点分 qualname 收敛为 `path::symbol`（24: 24 处、25: 22 处、26: 14 处，点分写法为 0）。
- index 修正 23 的依赖方向、把 26 前移到 18 之前、纠正 20/21 的依赖倒置，并加了“默认先进入 MRV2（12），11 为对照与回落”的提示。
- 升级为“完整”的 e2e 阶段：KV 预算（08 §3.1.3 有公式、算例与 startup plan）、`SchedulerOutput` 生产端 22 字段（07 §8.1.1–8.1.3，逐字段与源码清点一致）、prompt logprobs（14 §3.8）、KV events（23 §5.3）、connector 遥测矩阵（23 §5.2）、Elastic EP 编排（18 与 13 §4.6 闭环）。
- 06 现在独占 `async_scheduling` → `max_concurrent_batches` → `step_fn`/Scheduler 类这条选择轴，02、11、12、19 都指向 06 §4.1，六页口径一致。

## 5. 建议的收口顺序

1. **B1 三处锚点**，然后重跑 `python -m tools.mkdocs_site.cli build --changed` 取构建证据；这是合并前的硬条件。
2. **B2 十条新 P1**：1–7 为单句到单段的改写；8–10 是 19 的两张图与一处约束范围，需要重绘并与 12 对齐。
3. **B3 02 的 item 5/8**，同时按 §2 删掉重复正文（07 §8.1.2、07 §2.1、06 §5.1/§6.1 与 07 §8.3/§8.5 的取舍）。
4. 22、23 的页内矛盾改成如实表述；index 补覆盖边界说明。
5. 剩余 P2（各组共约 60 条，明细在分组报告里）与规划裁决项。

本轮仍为静态读码：未运行 vLLM、GPU、跨机传输或故障注入；除本报告与明细归档外，没有修改 wiki 页面，也没有提交或推送。
