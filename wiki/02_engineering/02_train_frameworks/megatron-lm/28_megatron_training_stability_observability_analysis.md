---
title: "Megatron-LM 训练稳定性与可观测性深度解析"
---

# Megatron-LM 训练稳定性与可观测性深度解析

> **源码基线**:`NVIDIA/Megatron-LM@71092579522a12522d9f323ae180c9825d01928a`(`dev`,2026-08-27)
> **重定基线**:2026-08-28 由 `ee3f1ffa…`(2026-05-19)推进,跨 578 个提交;本页全部 `path:line` 已在新基线下逐条重核。
> 核心文件:`megatron/core/rerun_state_machine.py`、`megatron/core/fault_injector.py`、`megatron/core/energy_monitor.py`、`megatron/core/timers.py`、`megatron/core/optimizer/qk_clip.py`、`megatron/core/optimizer/grad_scaler.py`、`megatron/core/optimizer/clip_grads.py`、`megatron/core/transformer/moe/moe_logging.py`、`megatron/core/transformer/moe/router_replay.py`;训练循环日志在 `megatron/training/training.py`
> 配套阅读:`16_megatron_distributed_optimizer_analysis.md`、五份并行分析、`27_megatron_tp_fsdp_resharding_supplements_analysis.md`
> 定位:系统性专题。前面所有文档讲"怎么把模型并行训起来、训得快";本文讲**怎么让它训得稳、出问题怎么发现、看哪些指标判断健康**。
> **叙事顺序**:本页按五拍组织——背景 → 为什么这么设计(含被否掉的替代)→ 实现思路与细节 → 约束 → 发展趋势。
> **最近更新**:2026-08-28。按五拍重排章节顺序;机制正文与既有引用未改。

> [!update] 该特性自 `dev@232c478d4`(2026-06-16)引入,行号已重核至基线 `71092579`。
> 本页已对齐 `ee3f1ff..HEAD` 的稳定性/可观测性增量。新增机制:**梯度范数超阈跳步**(§1.2)、**MTP 训练稳定性套件**(detach / 隔离 / 独立缩放 / 独立裁剪,新增 §1.8)、**MoE aux/z-loss 在 TP>1 下的梯度缩放修正**与 **DSA indexer loss 跨 micro-batch 平均**(§1.7)。可观测性侧补充 RerunStateMachine 去 stat 系统调用(§1.4)、MoE logging 的 `record/report` 生命周期(§2.2)、seqlen 统计保真与混合模型显式进程组(§2.5)。各处以特性引入日期 `[!update]` 标注;2026-08-28 重定基线后,全页行号(含这些小节)统一以 `71092579` 为准。重核发现一处结论已不成立,以 `[!contradiction]` 标在 §1.7(aux/z-loss 的 TP 缩放修正被 #5542 改写)。

---

## 0. 背景与设计取舍

### 0.1 背景:大规模长时训练的三类系统性风险

大规模训练(几千卡、数月)面对三类系统性风险:

1. **数值不稳定** —— 梯度/激活溢出、loss 尖峰、attention logit 爆炸。
2. **硬件故障与静默错误** —— GPU 宕机(显性)、**静默数据损坏 SDC**(GPU 算错但不报错)。
3. **不可观测** —— 训练跑着但不知道健不健康、慢在哪、为什么发散。

Megatron 对应有三套机制,本文依次拆解:

| 风险 | 机制 | 代码 |
|------|------|------|
| 数值不稳定 | loss scaling、梯度裁剪、**梯度范数超阈跳步**(§1.2)、NaN/Inf 检查、QK-clip、**MTP 稳定性套件**(§1.8) | `megatron/core/optimizer/grad_scaler.py`、`megatron/core/optimizer/clip_grads.py`、`megatron/core/optimizer/optimizer.py`、`megatron/core/optimizer/qk_clip.py`、`megatron/core/transformer/multi_token_prediction.py` |
| 硬件故障 / SDC | RerunStateMachine、fault injector、NTP 容错 | `megatron/core/rerun_state_machine.py`、`megatron/core/fault_injector.py`、`megatron/core/distributed/nonuniform_tp.py` |
| 可观测性 | Timer 系统、MoE 逐层指标、能耗监控、TB/wandb 日志 | `megatron/core/timers.py`、`megatron/core/transformer/moe/moe_logging.py`、`megatron/core/energy_monitor.py` |


### 0.2 为什么这么设计:三条路各自否掉了什么

上表三套机制都不是唯一解。下面逐条给出源码陈述的理由与**被否掉的替代**;源码沉默处整段标为推断。

**① SDC 归因用"重跑 + 带容差比对",而不是"重算一遍比位"。**
`validate_result` 的 docstring 把整个判据写死成三条重跑比对:「an **irreproducible** result is detected by rerunning the iteration **locally (same GPU)** and verifying the result is different;a **mismatching** result is detected by rerunning the iteration on a **different GPU** …;an **expected** result is detected by rerunning … and verifying the result is the **same**」(`megatron/core/rerun_state_machine.py:508-515`)。关键在于它**不要求逐位相同**:`tolerance` 参数的说明是「tolerance used in combination with `comparison_func` to determine reproducibility of results. Default is no tolerance (deterministic calculations)」(`:483-484`),而 docstring 里的示例直接给 `tolerance=0.001`,注释即「max 0.1% difference in results due to non-determinism」(`:503`)。类级 Caveats 更把这条抬成前提:「computations are **NOT** required to be deterministic, i.e. results may vary slightly across re-runs of the iteration」——真正被要求确定的只有**控制流**(`:169-177`)。

**② 第一级判据由调用方给函数,而不是引擎内置阈值;而且 `DISABLED` 模式复用同一入口。**
`validate_result(result, rejection_func, message, comparison_func=None, tolerance=0.0, fatal=True)`(`megatron/core/rerun_state_machine.py:464-471`):判"这个结果算不算异常"的 `rejection_func`(例:`torch.isnan`)、描述串(例:`"spiky loss"`)、比对函数与容差,全部由调用点提供。而在默认的 `RerunMode.DISABLED` 下,这条路径**仍然执行** `rejection_func`,并在 `fatal=True` 时抛 `RuntimeError`——注释写明动机:「This is a **backward-compatible behavior for infs and NaNs**」(`:517-537`)。这正是 §1.3 的 `--check-for-nan-in-loss-and-grad` 与 §1.4 的多级归因共用同一入口的原因:关掉归因不等于关掉检查。

**③ MoE aux/z-loss 的跨 rank 缩放:闭式因子 `×|tp_cp|` 被"实测有效 token 数"取代。**
这是本页范围内最清楚的一次"替代被否":
- **旧法**(#5047,`9ef8a2a42`,2026-06-02,commit message 即 "Fix MoE aux_loss / z_loss gradient scaling with TP > 1"):`calculate_per_token_loss` 下把 aux/z-loss 预乘 `num_local_tokens * self.tp_cp_group.size()`,用一个**闭式常数**抵消 `finalize_model_grads` 里的全局除法(推导见 §1.7 的 `[!update]`)。
- **新法**(`7f9175207`,#4359,2026-06-17):把本域的有效 token 数装进张量,沿 `aux_loss_scale_reduce_groups` **逐组 `all_reduce` 求和**后直接相乘(`megatron/core/transformer/moe/router.py:605-622`)。
- **源码自陈的理由就写在注释里**:「Use the reduced count directly: with **THD padding or dynamic CP**, valid token counts can differ by rank/group, so `local_num_tokens * group_size` is **not generally correct**」(`megatron/core/transformer/moe/router.py:599-604`)。
- → 判据是**前提失效**而不是精度不够:闭式因子只在"同组各 rank 有效 token 数相等"时成立,THD packing 与 dynamic CP 直接打破了这个前提,于是宁可每个 aux-loss 域每步多付一次 `all_reduce`(`:620-621`),也不留一个在新特性下**静默**算错的常数。

**④ MTP 的解耦是三层可叠加开关,而"把 output layer 整体 functional 化隔离"这条路试过并被撤下。**
#5080(`d23ca853f`)曾新增独立开关 `mtp_isolated_loss`,用 `torch.func.functional_call` 配 `{name: param.detach()}` 把 output layer 的**全部参数/buffer** 隔离;#5223(`7eedac586`,2026-06-11,"[Dev] Cherry-pick MTP detach heads")把它整体删除,只保留"`detach()` 共享的 `output_weight` 张量"这一最小手术,并把能力并进 `mtp_detach_heads`。细节与行号见 §1.8 (b) 的 `[!deprecated]`。

> [!note] 推断
> 源码陈述的是**事实**:重跑三段式流程与 `tolerance` 的存在、`DISABLED` 下仍跑 `rejection_func` 的向后兼容注释、闭式因子在 THD/dynamic-CP 下"not generally correct"、`mtp_isolated_loss` 被删除。**把这些读成"逐位比对因此不可行"、"宁可多付一次 all_reduce 也不要静默错误"、"最小手术优于全隔离"这三层取舍判据,是本页的重建**,源码从未这样自陈。要引用这几条判断,请回到 `megatron/core/rerun_state_machine.py:169-175`、`:483-484`、`:503`、`:508-515`、`:517-537`、`megatron/core/transformer/moe/router.py:599-604`、`:605-622` 这几个 locator,不要引用本段推断。
> 另有一处**归属存疑**,不改既有 callout、仅在此标出:§1.7 的 `[!contradiction]` 把 aux-loss 缩放的改写记在 #5542(`d1384c2d9`)名下;按 `git log -S "so local_num_tokens * group_size is not generally correct" -- megatron/core/transformer/moe/router.py`,引入该注释与 `all_reduce` 路径的实际是 `7f9175207`(#4359,2026-06-17),`d1384c2d9`(#5542,2026-07-14)只在其上加了 14 行的 padding 排除(`valid_token_count`)。结论(闭式因子已被推翻)不变,变的是归属的 PR。

---

## 1. 训练稳定性机制

### 1.1 数值溢出防护:loss scaling + inf/nan 跳步

(详见 `16_megatron_distributed_optimizer_analysis.md` §13–§14,此处定位为"稳定性"视角)

fp16 训练里梯度可能**下溢成 0** 或**上溢成 inf**。两道防线:
- **Loss scaling**:loss 乘 `S`,把小梯度抬出下溢区;`DynamicGradScaler` 自适应 `S`(`megatron/core/optimizer/grad_scaler.py:64`)。
- **inf/nan 跳步**:`optimizer.step()` 第一步 `prepare_grads` 扫描梯度,发现非有限值 → **`found_inf_flag=True` → 整步丢弃**(参数不动),`DynamicGradScaler` 随后按 `backoff_factor` 调小 `S`。

这是 fp16 训练的安全阀:偶发溢出不会污染参数,只是浪费一步。

### 1.2 梯度尖峰防护:全局梯度裁剪

`megatron/core/optimizer/clip_grads.py` / `MegatronOptimizer.clip_grad_norm`:算全局梯度范数 `‖g‖`(需跨 TP×PP all-reduce),超过 `--clip-grad`(默认 1.0)就等比缩小。挡住偶发的梯度尖峰,防止单步大跳导致发散。

> [!update] 梯度范数超阈跳步(#3460) — 该特性自 `dev@232c478d4`(2026-06-16)引入,行号已重核至基线 `71092579`。
> 在裁剪之上新增**第二道闸**:`OptimizerConfig.grad_norm_skip_threshold`(`megatron/core/optimizer/optimizer_config.py:394`,默认 `float('inf')` 即关闭)。
> 在 `ChainedOptimizer.step()`(`megatron/core/optimizer/optimizer.py:2013`,跳步判据在 `:2081-2090`)里,算完主梯度范数 `grad_norm` 并完成裁剪后,若 `grad_norm > config.grad_norm_skip_threshold`(且存在 main 参数),则打 `INFO` 日志并置 `should_skip_update=True`,使 `update_successful = False`、**不调 `step_with_ready_grads()`** —— 整步丢弃,参数不动。
> - 与 §1.1 inf/nan 跳步的区别:那是因**非有限值**而跳;这里是因**有限但过大**的范数而跳(裁剪还嫌不够,疑似该步本身被污染)。
> - 与 §1.2 裁剪的区别:裁剪是把范数**缩到阈值后照常更新**;超阈跳步是**直接放弃这一步**。
> - 判据用的是"主梯度范数"(`get_grad_norm()` 已排除 §1.8 的 `mtp` 独立范数组),且仅在 `main_params` 非空时触发。
> - 目前**只有配置项、无 CLI 开关**,需经 `OptimizerConfig` 注入;只在 `ChainedOptimizer.step` 路径生效(Megatron 主训练栈即走此路径)。被跳的步计入 `skipped iters`(§3.2)。

### 1.3 NaN / Inf 显式检查

`--check-for-nan-in-loss-and-grad`:每步显式检查 loss 与梯度是否含 NaN/Inf。`megatron/core/rerun_state_machine.py` 即使没调 `validate_result` 也会在这个选项下兜底检查(`:583`)。发现后可选择**报错退出**或交给 RerunStateMachine 归因(§1.4)。FSDP 侧另有 `_check_nan_in_grad`。

### 1.4 RerunStateMachine —— 静默数据损坏(SDC)归因

`megatron/core/rerun_state_machine.py`(1425 行)。**这是 Megatron 最有特色的稳定性机制**,专治"GPU 算错了但不崩"的 SDC。

**问题**:出现 NaN 或 loss 尖峰时,根因可能是 ——(a) 真实的数据驱动尖峰(无害);(b) 某 GPU 偶发位翻转(transient);(c) 某 GPU 硬件坏了(persistent)。光看数值分不清。

**做法 —— 多级重跑归因**:在关键计算点调 `validate_result(...)`(`:464`,带一句描述如 `"spiky loss"` 和一个拒绝函数)。一旦结果被判"异常":

```
异常结果
   │
   ├─ 在同一 GPU 原地重跑 forward-backward
   │     ├─ 结果不同 ──► TRANSIENT_ERROR(不可复现 → 偶发位翻转/SDC)
   │     └─ 结果相同 ──► 继续
   │                       │
   │     从 checkpoint 在不同 GPU 上重跑
   │           ├─ 结果不同 ──► PERSISTENT_ERROR(原 GPU 硬件故障)
   │           └─ 结果相同 ──► CORRECT_RESULT(真实尖峰,无害,放行)
```

`RerunDiagnostic`(`:60`)三种结论;`RerunState`(`:82`)六个状态机状态管这套多级重跑(原地重跑 → checkpoint 重跑 → 换 GPU 重跑)。识别出故障 GPU 后 `should_checkpoint_and_exit`(`:400`)让作业**存档退出**,由调度器换节点重启。

**三种模式**(`RerunMode`,`:74`):
- `DISABLED` —— 默认关闭(实验特性)。
- `VALIDATE_RESULTS` —— 开启上述归因。
- `REPORT_DETERMINISM_STATS` —— 只统计计算的确定性,不重跑归因。

> 源码自带 DISCLAIMER:这是 alpha 级实验特性,标记的"故障节点"应再用标准诊断套件确认。

> [!update] 去掉 validate_result 的 stat 系统调用(#5107) — 该特性自 `dev@232c478d4`(2026-06-16)引入,行号已重核至基线 `71092579`。
> 上文对 RerunStateMachine 流程的描述在 `dev@232c478d4` 仍然成立(行号微移:`validate_result`→`:464`、`RerunDiagnostic`→`:60`、`RerunMode`→`:74`、`RerunState`→`:82`、`should_checkpoint_and_exit`→`:400`、check-for-nan 兜底注释→`:583`)。一处实现变化:
> - 旧实现每次 `validate_result` 都用 `inspect.currentframe()` + `inspect.getframeinfo()` 取调用点的 `filename/lineno`,后者会触发**文件系统 stat 系统调用**;在高频校验下成为热点。现已删去 `import inspect` 与整段取帧逻辑(`megatron/core/rerun_state_machine.py:959` 的 `_get_validation_call_info`)。
> - `Caller` 具名元组(`:46`)不再含 `filename/lineno`,只保留 `(message, rank)`。确定性统计(`REPORT_DETERMINISM_STATS`)的日志改为按校验描述串定位 ——`"From validation call '<message>'"`(`:1012`/`:1020`),取代原来的 `From <file>, line <n>`。
> - 结论:这是**纯性能/可观测性优化**,不改变 transient/persistent/correct 的三级归因语义;唯一影响是日志里用"校验描述"而非"文件:行号"来标识每个校验点(因此校验描述串应起得可读、可区分)。

### 1.5 QK-clip —— 注意力 logit 稳定

`megatron/core/optimizer/qk_clip.py`。已知的一类训练不稳:attention 的 `Q·Kᵀ` logit 数值越训越大,softmax 进饱和区、梯度异常。`clip_qk(model)`:
- 遍历各层 `self_attention`,读 `core_attention.current_max_attn_logits`(前向时记录的本层最大 attention logit)。
- 跨 DP(含 CP)组 `all_reduce(MAX)` 得全局最大 logit。
- `log_max_only=False` 时调 `clip_qk()` 实际裁剪;`log_max_only=True` 则**只监控不裁剪**(把最大 logit 当指标看)。

返回的 `log_max_attention_logit` 既是稳定手段、也是一个可观测指标。

### 1.6 容错:NTP 与故障注入

- **NTP(Nonuniform TP)**:TP 组留备用 rank,核心 rank 故障时重分片续训,免整体重启 —— 详见 `27_megatron_tp_fsdp_resharding_supplements_analysis.md` §4。
- **`megatron/core/fault_injector.py`**(233 行):**主动注入故障**用于测试容错路径 —— 验证 RerunStateMachine、NTP 等机制是否真能正确响应。是"测试稳定性机制本身"的工具。

### 1.7 MoE 稳定性

MoE 有独有的不稳定源 —— 路由:
- **router fp32**:`--moe-router-dtype fp32` —— 路由 logit 保持 fp32(README 强调:高专家数下 bf16 路由精度不足,专家输出按路由分加权累加会放大误差)。
- **负载均衡损失**:`aux_loss` 等防止专家路由坍塌(`14_megatron_ep_analysis.md` §7)。
- **`megatron/core/transformer/moe/router_replay.py`**(208 行):记录/重放路由决策 —— 用于复现和调试路由相关的不确定性。

> [!update] aux_loss / z_loss 在 TP>1 下的梯度缩放修正(#5047) — 该特性自 `dev@232c478d4`(2026-06-16)引入,行号已重核至基线 `71092579`。
> `--calculate-per-token-loss` 模式下,`finalize_model_grads` 会把每个参数梯度统一除以 `total_global_tokens`(全局非 padding token 数)。但 router 权重标了 `sequence_parallel=True`,各 TP rank 只在自己的**序列分片**上算偏梯度、再由 `_allreduce_non_tensor_model_parallel_grads` 在 TP 组内**求和**。把 `total_global_tokens` 按 router 的本地 token 数展开:
> $$
> \begin{aligned}
> \text{total\_global\_tokens}
> &= \text{num\_micro\_batches}\times \text{dp\_size}\times\big(\text{num\_local\_tokens}\times\lvert\text{tp\_cp}\rvert\big)
> \end{aligned}
> $$
> 旧代码只乘 `num_local_tokens`,在 `tp_cp_group.size()>1` 时 aux/z-loss 梯度被额外缩小了 `|tp_cp|` 倍 —— TP/CP 越大,负载均衡损失越被稀释。
> 修正:`aux_loss` 与 `z_loss` 预乘改为 `num_local_tokens * self.tp_cp_group.size()`(`megatron/core/transformer/moe/router.py:546`、`:587`,行号对应旧基线 `232c478d4`),恰好抵消上式中的 `|tp_cp|`,使有效缩放回到目标 `1/(num_micro_batches·dp_size)`,与 `!calculate_per_token_loss` 路径一致、且对 TP/CP 配置不变。(z_loss 系数另有 `/tp_cp_group.size()` 是独立的**前向**修正:z_loss 在每个 TP+CP rank 的本地 logits 上独立计算,需按 TP+CP 求平均而非求和。)回归测试见 `tests/.../test_aux_loss.py::TestPerTokenAuxLoss`。

> [!contradiction] 上一条的**修正手法**在基线 `71092579` 下已被推翻(#5542 `d1384c2d9`,[codex] Exclude padding tokens from MoE routing)。
> `aux_loss` 的预乘不再是 `num_local_tokens * tp_cp_group.size()`,而是把本域的**有效 token 数**放进一个张量、沿 `aux_loss_scale_reduce_groups` 逐组 `all_reduce` 求和后直接相乘(`megatron/core/transformer/moe/router.py:598-624`)。
> 源码给出的理由就写在注释里:*with THD padding or dynamic CP, valid token counts can differ by rank/group, so local_num_tokens * group_size is not generally correct*(`:602-604`)—— 即 THD padding / 动态 CP 下各 rank 的有效 token 数**不再相等**,`×|tp_cp|` 这个闭式因子本身就不成立。
> `z_loss` 侧同样改写:`calculate_per_token_loss` 分支现直接挂 `z_loss_sum`(不再乘 `num_local_tokens × |tp_cp|`,`:666-671`);只有 `!calculate_per_token_loss` 分支保留了 `moe_z_loss_coeff / tp_cp_group.size()` 这一**前向**修正(`:673`)。
> 因此上文的代数推导只对旧基线 `232c478d4` 有效;§3.4「解读 aux/z-loss 需注意 TP 缩放」的口径提示同样只适用于 `232c478d4` 到 #5542 之间的版本。

> [!update] DSA indexer loss 跨 micro-batch 平均(#4070) — 该特性自 `dev@232c478d4`(2026-06-16)引入,行号已重核至基线 `71092579`。
> 新的实验性注意力变体 **DSA(Dynamic Sparse Attention,`experimental_attention_variant='dsa'`)** 引入一个 **indexer 辅助损失**,经 `DSAIndexerLossAutoScaler`(`megatron/core/transformer/experimental_attention_variant/dsa.py:1149`)注入梯度。此前它**未按 micro-batch 数归一**,导致其相对主损失的尺度随梯度累积步数漂移。
> 修正:在 `forward_step_calc_loss`(`megatron/core/pipeline_parallel/schedules.py:377–394`)按与 MTP loss 相同的方式设缩放 —— `calculate_per_token_loss` 时设 `loss_scale`,否则设 `loss_scale / num_microbatches`。属于"辅助损失正确归一"一类,与 §1.8 的 MTP loss 缩放同源。

### 1.8 MTP(多 token 预测)训练稳定性套件

> [!update] MTP 稳定性套件(#3456 / #5080 / #3459 / #4116) — 该特性自 `dev@232c478d4`(2026-06-16)引入,行号已重核至基线 `71092579`。
> 注:#5080 引入的独立开关 `mtp_isolated_loss` 在 `dev@232c478d4` 已被**合并进 `mtp_detach_heads`** 并移除(见下文 (b) 的 `[!deprecated]`),故 HEAD 上实际是"`mtp_detach_heads` + `mtp_grad_scale_func` + `mtp` 独立裁剪组"三件套。

MTP(Multi-Token Prediction,详见 GPT/DeepSeek 系列)在主模型之外挂若干 MTP head,用一个**辅助损失**让模型一次预测多个未来 token。问题在于:MTP loss 的梯度默认会**回流主模型与共享权重**(embedding、output projection),与主 LM loss 抢梯度、互相干扰;且 MTP head 的梯度尺度常与主干不同,统一裁剪/缩放会失真。`dev` 分支为此补了一套**可叠加**的控制开关,核心是"把 MTP 这条支路从主图上逐级解耦,并给它独立的损失缩放与梯度裁剪"。

**(a) `mtp_detach_heads` —— 切断 MTP→主模型的梯度回流(#3456,并在 #5223 合并 #5080 的隔离能力)**

`TransformerConfig.mtp_detach_heads`(`megatron/core/transformer/transformer_config.py:89`,默认 `False`)。开启后在三处 `detach()`,使 MTP loss **只训练 MTP head 自身**,不更新主模型:
- `MultiTokenPredictionBlock.forward`:取出本 stage 的 `hidden_states` 后 `detach()`(`megatron/core/transformer/multi_token_prediction.py:2931`)。
- `MultiTokenPredictionLayer._get_embeddings`:`decoder_input = embedding(...).detach()`,切断对**共享 embedding** 的梯度(`:2137`)。注意紧接着对 `hidden_states` 做 `make_viewless_tensor` 后,若它已不 `requires_grad` 会显式 `requires_grad_(True)` —— 因为 `detach()` 后张量 `_base` 为 `None`、`make_viewless_tensor` 退化为 no-op,而激活重计算(`CheckpointFunction`)要求至少一个输入可导,这里补回以保住到 MTP 层参数的梯度通路。
- `process_mtp_loss`:`output_weight.detach()`,切断对**共享 output projection** 的梯度(`:1772`)。
- **在线 RL 支持**(原属 #5080,现并入本开关):`process_mtp_loss` 允许 `labels=None`(RL 时主 LM head 输出 logits 供外部 RL loss 用),此时从 `input_ids` 左移一位自行派生 MTP 标签(`label[i]=input_id[i+1]`,`megatron/core/transformer/multi_token_prediction.py:1746–1761`),让 MTP 辅助损失在不触碰主模型的前提下照常训练。

**(b) ~~`mtp_isolated_loss`~~ —— 已被 (a) 合并并移除(#5080 引入 → #5223 撤下)**

> [!deprecated] 2026-06-16:`mtp_isolated_loss` 配置在 `dev@232c478d4` **已不存在**。#5080 曾新增该开关,在 (a) 之上更进一步,用 `torch.func.functional_call` 配 `{name: param.detach()}` 把 **output layer 的全部参数/buffer** 也隔离(`output_layer_for_mtp`)。随后 #5223([Dev] Cherry-pick MTP detach heads,`7eedac586`)做了**收敛**:删除 `mtp_isolated_loss` 配置与该 `functional_call` 全隔离逻辑,把"切断对共享 `output_weight` 的梯度"与"`labels=None` 派生标签"两项能力**直接挂到 `mtp_detach_heads`** 上(见 (a))。因此 HEAD 上:① 不再有独立的 `mtp_isolated_loss`;② 不再对 output layer 的**全部可学习参数**做 functional 化隔离,只 `detach()` 共享的 `output_weight` 张量。文档保留此项以记录该演进;若读 `dev@232c478d4` 源码,请勿再找 `mtp_isolated_loss`。(2026-08-28 重核:基线 `71092579` 下全仓 `git grep mtp_isolated_loss` 仍为 0 命中,本条依然成立。)

**(c) `mtp_grad_scale_func` —— MTP loss 独立的损失缩放(#3459)**

`ModelParallelConfig.mtp_grad_scale_func`(`megatron/core/model_parallel_config.py:195`,默认 `None`)。此前 MTP loss 与主 loss 共用 `grad_scale_func`;现在可单独给 MTP loss 指定缩放函数。落地在 `megatron/core/pipeline_parallel/schedules.py` 新增的 `_get_mtp_loss_scale(config, device)`(`:266`):
- 优先用 `mtp_grad_scale_func()`;否则回退 `grad_scale_func(torch.ones(1))`;再否则取 `1`。
- 结果会校验必须是标量 / size-1 张量,并搬到 output 张量所在 device,经 `MTPLossAutoScaler.set_loss_scale` 注入(`megatron/core/pipeline_parallel/schedules.py:365–375`)。
- 意义:fp16/bf16 下 MTP 支路可用与主 loss **不同的 loss scale**,避免辅助损失把主 loss 的动态缩放带偏。

**(d) `mtp` 独立梯度裁剪组(#4116)**

当 `mtp_detach_heads=True` 时,MTP head 的梯度已与主干解耦,但若仍并入全局范数一起裁剪,二者尺度差异会互相污染。本 PR 在优化器侧引入**命名梯度范数组**机制:
- 建块时给 MTP 参数打标签:`MultiTokenPredictionBlock.__init__` 中 `for param in self.parameters(): param.grad_norm_group = 'mtp'`(`megatron/core/transformer/multi_token_prediction.py:2790`)。
- 优化器侧新增基础设施(`megatron/core/optimizer/optimizer.py`):常量 `MTP_GRAD_NORM_GROUP='mtp'`、`SEPARATE_GRAD_NORM_GROUPS`、`GRAD_NORM_GROUP_ATTR`;辅助函数 `_get_param_grad_norm_group` / `_is_separate_grad_norm_group` / `_validate_grad_norm_group`;以及 `copy_optimizer_param_metadata`(建主 fp32 副本时把 `grad_norm_group` 标签一并复制,否则副本丢标签)。
- 范数计算分流:原 `get_main_grads_for_grad_norm` 重构为 `get_grads_for_grad_norm(grad_norm_group=None)` —— 传 `None` 取**主组**梯度(已**排除** `mtp` 组),传 `'mtp'` 只取该组。`clip_grad_norm` / `ChainedOptimizer.step` 据此把 `main_params` 与各 `grad_norm_group` 分别算范数、**各自按 `clip_grad` 独立裁剪**(`megatron/core/optimizer/optimizer.py:2053–2079`)。
- 跨 rank 一致性:`has_grad_norm_group()` 用一次全局 `all_reduce(MAX)` 判断"是否有任一 rank 持有该组参数"并缓存,保证按组归约的集合通信在各 rank 间不失配(某 rank 本地无 mtp 分片、对端有);`LayerWiseDistributedOptimizer` 重写该方法走全局 `group=None` 归约(与其 dist-opt 全局归约范式一致)。
- 与 §1.2 跳步的衔接:超阈跳步判据用的是**主组** `grad_norm`(不含 mtp),即 MTP head 的大范数不会误触发整步丢弃。

**叠加关系小结(HEAD = `dev@232c478d4`)**:`mtp_detach_heads` 是主开关 —— 切断 MTP→主模型/共享 embedding/共享 `output_weight` 的梯度回流,并支持在线 RL 的 `labels=None` 派生(已吸收 #5080 原 `mtp_isolated_loss` 的能力,见 (b))。在此之上可叠加 `mtp_grad_scale_func`(独立损失缩放);且 `mtp_detach_heads=True` 时自动启用 `mtp` 独立梯度裁剪组(独立梯度裁剪)。三者共同把 MTP 这条辅助支路在**前向图、损失缩放、梯度裁剪**三个层面与主模型解耦,显著降低 MTP 对主训练稳定性的干扰。

---

## 2. 监控基础设施

### 2.1 Timer 系统(`megatron/core/timers.py`)

`Timer` / `DummyTimer`(`megatron/core/timers.py:109` / `:78`,共同基类 `TimerBase` 在 `:35`)。给训练各阶段计时:`forward-compute`、`backward-compute`、`optimizer`、`batch-generator`、`forward-backward`、`optimizer-clip-main-grad`、`optimizer-count-zeros` 等(前几份文档的 `config.timers('...')` 调用即此)。

要点:
- **`log_level` 分级**:每个 timer 有 log level,低于阈值的用 `DummyTimer`(零开销空实现)—— 细粒度计时不污染生产性能。
- **`barrier` 选项**:计时前可选 `torch.distributed.barrier()`,得到对齐的、可跨 rank 比较的耗时(用于发现 straggler);不加 barrier 则是本 rank 异步耗时。

### 2.2 MoE 逐层指标(`megatron/core/transformer/moe/moe_logging.py`)

`megatron/core/transformer/moe/moe_logging.py`(745 行)有两个全局 tracker:
- **`MoEMetricsTracker`**:逐层收集 MoE 指标(各层 aux loss、z-loss 等),`--moe-per-layer-logging` 开启。能看出**哪一层**路由出问题,而不只是全局平均。
- **`MoEOverloadFactorTracker`**(`:95`):跟踪**专家过载因子**(overload factor)——`max_expert_load / mean_load`,即 `14_megatron_ep_analysis.md` §7.1 的负载不均衡因子 `f`。`--log-moe-overload-factor` 开启;跨 `tp_ep` 与 `expt_dp` 组做 MAX 规约,反映最坏专家的过载程度。

> [!update] MoE logging 的 record/report 生命周期(#3431) — 该特性自 `dev@232c478d4`(2026-06-16)引入,行号已重核至基线 `71092579`。
> `megatron/core/transformer/moe/moe_logging.py` 在 `ee3f1ff..HEAD` 间**内容无净变化**(仍 745 行),上述两个 tracker 描述在 `dev@232c478d4` 依然准确。补充其 #3431 重构后的标准用法,便于对照源码:
> - **生命周期**:每步 `record(name, value, layer_number, num_layers, reduce_group=...)` 在 router 前向时**按层累加**到 `MetricEntry`(`megatron/core/transformer/moe/moe_logging.py:435` 起的 `MoEMetricsTracker`)→ 步末一次 `report(loss_scale=1/num_microbatches, iteration, writer=..., per_layer_logging=...)` 统一**跨 rank 同步 + 聚合 + 写 TB/W&B + 清零**。全局单例经 `get_moe_metrics_tracker()` 取得。
> - **PP 对齐**:无 MoE 层的 PP rank 需 `force_initialize=True` 预建大小为 `num_layers(+mtp_num_layers)` 的零张量,否则跨 PP 的 `all_reduce` 会因张量尺寸不一致而挂死。
> - **归约语义**:`MetricEntry` 带 `reduce_group`(求和,如 `tp_cp`)、`avg_group`(求平均)、`needs_dp_avg`(再跨 DP 平均)三档;以 `"loss"` 结尾的指标会并入训练循环的 `total_loss_dict` 而**不**重复打到控制台串。

### 2.3 能耗监控(`megatron/core/energy_monitor.py`)

`megatron/core/energy_monitor.py`(95 行):采集 GPU 能耗,算每步/每 token 的能量 —— 大规模训练的成本与碳足迹指标。

### 2.4 日志后端:TensorBoard / wandb / one_logger

`megatron/training/training.py` 的 `training_log` 把指标同时写三处:`writer`(TensorBoard)、`wandb_writer`、`one_logger`。各指标有独立开关(`--log-loss-scale-to-tensorboard`、`--log-throughput` 等)。控制台还会拼一行 `log_string` 打印关键指标。

### 2.5 杂项可观测性修正

> [!update] 保真序列长度统计(#95654c956) — 该特性自 `dev@232c478d4`(2026-06-16)引入,行号已重核至基线 `71092579`。
> `train_step` 返回的 `seqlen_sum_this_global_batch` / `seqlen_squared_sum_this_global_batch` 用于**变长序列感知的吞吐 / FLOP 估算**(`megatron/training/training.py:3513` 经 `seqlen_squared_sum_in_batch` 进入 attention FLOP 项)。旧代码在**非变长**(未走 `wrap_data_iterator`)路径下把这两个统计丢成 `_, _`,导致 FLOP/吞吐日志退化或失真。
> 修正(`megatron/training/training.py:3029–3030`):非变长路径显式回填闭式值 `seq_length * global_batch_size` 与 `seq_length² * global_batch_size`,保证两条路径都给出正确的 seqlen 统计、`--log-throughput` 的 TFLOP/s 估算保真。属于纯可观测性修复,不改训练数值。

> [!update] 混合模型分阶段日志改传显式进程组(#4781) — 该特性自 `dev@232c478d4`(2026-06-16)引入,行号已重核至基线 `71092579`。
> 混合(Hybrid,如 Mamba/attention 混排)模型在 `select_pipeline_segment` 里对每个 PP 段做布局日志。此前该日志依赖全局 `parallel_state` 取 TP / DP-CP 组,在自定义进程组拓扑下可能取错组、或在多 `ProcessGroupCollection` 场景下不一致。
> 修正:`HybridModel` 经 `_hybrid_logging_pg_kwargs(pg_collection)`(`megatron/core/models/hybrid/hybrid_model.py:44`,调用点 `:204`)抽出 `tp` / `dp_cp` 组,显式透传给 `select_pipeline_segment(..., tp_group=, dp_cp_group=)`(`megatron/core/models/hybrid/hybrid_layer_allocation.py:342`),并校验两者"要么都给、要么都不给"。是分阶段日志在显式进程组拓扑下的健壮性修正。

---

## 3. 指标目录(按类别)

下面是训练循环(`megatron/training/training.py` 的 `training_log`)实际产出、可用于分析的指标。

### 3.1 损失与收敛

| 指标 | 含义 | 健康信号 |
|------|------|---------|
| `lm loss`(及各 loss key) | 语言模型损失,跨 DP 平均 | 平滑下降、无持续尖峰 |
| MoE `aux loss` / `z-loss` | 负载均衡 / logit 正则损失(可逐层) | 稳定在小值 |
| MTP loss | 多 token 预测损失 | —— |
| 验证 loss | eval 集损失 | 与训练 loss 不剧烈背离 |

### 3.2 数值健康

| 指标 | 含义 | 健康信号 |
|------|------|---------|
| `grad norm` | 全局梯度范数 | 稳定区间;偶发尖峰被裁剪正常,持续上升危险 |
| `loss scale` | 当前 loss 缩放(fp16) | 稳定在某高值;频繁腰斩 = 频繁溢出,不健康 |
| `num zeros in grad` | 梯度中零元素数(`--log-num-zeros-in-grad`) | 突增 → 可能下溢 / dead neuron |
| `nan iters` | 出现 NaN 的迭代计数 | 应 ≈ 0 |
| `skipped iters` | 因 inf/nan 跳过的迭代数 | 应 ≈ 0;偏多说明数值不稳 |
| `max attention logit` | QK-clip 监控的最大注意力 logit | 不应无界增长 |

> [!update] 新增数值健康信号(#3460 / #4116) — 该特性自 `dev@232c478d4`(2026-06-16)引入,行号已重核至基线 `71092579`。
> - `skipped iters` 的口径扩展:除 inf/nan 跳步外,现在还包含 §1.2 的**梯度范数超阈跳步**(`grad_norm_skip_threshold`)。若启用了该阈值,`skipped iters` 偏多既可能是数值溢出、也可能是主梯度范数频繁超阈 —— 二者都指向"该降学习率 / 查数据 / 调阈值"。
> - 启用 MTP `mtp` 独立裁剪组(§1.8d)后,MTP head 的梯度范数被**单独**计算与裁剪;主 `grad norm` 指标因此**只反映主模型**(已排除 mtp 组),读 grad norm 时需注意它不再涵盖 MTP head。

### 3.3 吞吐与效率

| 指标 | 含义 |
|------|------|
| `throughput per GPU (TFLOP/s/GPU)` | 单卡算力利用(`--log-throughput`) |
| `tokens/s`、迭代耗时 | 端到端吞吐 |
| Timer 分解 | `forward-compute` / `backward-compute` / `optimizer` / `batch-generator` 各占多少 → 定位瓶颈 |
| `advanced iters` | 真正推进了的迭代数 |

### 3.4 MoE 专属

| 指标 | 含义 | 健康信号 |
|------|------|---------|
| 逐层 aux loss | 哪一层路由不均(`--moe-per-layer-logging`) | 各层都小 |
| **expert overload factor** `f` | 最忙专家 / 均值(`--log-moe-overload-factor`) | 接近 1 = 均衡;远大于 1 = 路由坍塌 |

> [!update] 解读 aux/z-loss 需注意 TP 缩放(#5047) — 该特性自 `dev@232c478d4`(2026-06-16)引入,行号已重核至基线 `71092579`。
> 若在 `--calculate-per-token-loss` 下用 TP>1 训练:旧版本里 aux/z-loss 的**梯度**被额外缩小了 `tp_cp_group.size()` 倍(§1.7),日志里的 loss 数值看似正常、但其对路由的实际约束被稀释 —— 表现为"aux loss 不高却仍路由不均 / overload factor 偏大"。升级到 `dev@232c478d4` 后该缩放已修正,横向对比历史曲线时需考虑这一口径变化。**再一次口径变化**:基线 `71092579` 下 #5542 又把这套修正整体改写(见 §1.7 的 `[!contradiction]`),因此 `232c478d4` 与 `71092579` 之间也存在一次不可直接对比的断点。

### 3.5 系统

| 指标 | 含义 |
|------|------|
| GPU 显存(allocated / reserved) | 距 OOM 的余量 |
| 能耗 / 每 token 能量 | `energy_monitor` |
| `consumed samples` / `consumed tokens` | 训练进度 |
| RerunStateMachine 归因记录 | transient / persistent error 计数 → 硬件健康 |

---

## 4. 实战:用指标判断训练健康

```
收敛健康?     看 lm loss —— 平滑下降、无持续尖峰即健康
              偶发单点尖峰 + 能自行恢复 → 多半无害(RerunStateMachine 可归因为 CORRECT_RESULT)

数值健康?     grad norm 稳定区间 + loss scale 不频繁腰斩 + skipped/nan iters ≈ 0
              loss scale 反复减半 → fp16 频繁溢出,考虑 bf16 或调初始 scale
              grad norm 持续走高 → 学习率过大 / 数据问题,可能将发散

性能健康?     throughput 平稳;掉速 → 看 Timer 分解(计算?通信?数据加载?)
              加 barrier 的 timer 跨 rank 比较 → 找 straggler

MoE 健康?     overload factor 接近 1;若远大于 1 → 调大 aux_loss 系数 / 换均衡策略

硬件健康?     开 RerunStateMachine(VALIDATE_RESULTS):
              transient error → 偶发,关注但未必换卡
              persistent error → 该 GPU 故障,checkpoint-and-exit 换节点
```

> [!update] 新增稳定性手段的实战切入 — 该特性自 `dev@232c478d4`(2026-06-16)引入,行号已重核至基线 `71092579`。
> - **梯度尖峰**裁剪仍救不回时:设 `grad_norm_skip_threshold`(§1.2),让主梯度范数超阈的"坏步"被整步丢弃,而非缩放后照常更新 —— 适合偶发数据异常 / 疑似污染步。被丢的步计入 `skipped iters`。
> - **MTP 训练干扰主干**:`lm loss` 抖动疑似来自 MTP 支路时,按 §1.8 逐级解耦 —— `mtp_detach_heads`(切回流,含共享 `output_weight` 与在线 RL `labels=None` 派生)→ 视需要 `mtp_grad_scale_func`(独立损失缩放)。`mtp_detach_heads=True` 自动启用 `mtp` 独立梯度裁剪组,主 `grad norm` 与 MTP head 不再互相污染。(注:HEAD 已无独立的 `mtp_isolated_loss`,其能力并入 `mtp_detach_heads`。)

---

## 5. 约束:这些机制的前提、代价与失效条件

前四节几乎只谈"能发现什么"。这一节把前提、代价与失效条件集中列出,每条带 locator。

**① RerunStateMachine 是 alpha 级实验特性,且默认关闭。**
文件顶部的 DISCLAIMER 原话:「alpha-level code … has **not been tested at scale** so should not be assumed to be accurate. Nodes flagged by this code as potentially faulty should be subjected to **standard diagnostic test suites** for a definitive diagnosis. Also note that experimental features **may break existing APIs**」(`megatron/core/rerun_state_machine.py:21-30`)。默认模式是 `RerunMode.DISABLED`(§1.4)。

**② 它对训练循环有一条硬假设:控制流必须可复现。**
Caveats(1):重跑要求「execution (flow control) of the iteration is **deterministic** w.r.t. the state captured by the rerun state」,更具体是「a re-run of the iteration yields the **same calls to `validate_result()`** as in the initial run」;计算本身则不要求确定(`megatron/core/rerun_state_machine.py:169-175`)。这意味着任何"按运行期条件跳过/追加校验点"的写法都会破坏归因。自定义状态还需自行提供 `state_save_func` / `state_restore_func`(`:135-141`,示例见 `:143-164`)。

**③ 它只能重跑当前这一步。**
Caveats(2)明写:「The re-run logic is **currently only able to re-run the current step**. It may be that an unexpected result (e.g. spiky loss) is the result of a calculation that happened at a **previous** iteration. The current implementation **will not catch such issues**」(`megatron/core/rerun_state_machine.py:177-180`)。即"上一步埋的雷、这一步炸"这类问题,本机制查不出。

**④ 归因不是在线修复,而是"存档 + 特定退出码 + 由调度器重启"。**
`EXIT_CODE_RESUME_TO_DISAMBIGUATE = 16`(需重启以消歧)、`EXIT_CODE_FAILED_ON_RESULT_VALIDATION = 17`(校验失败)(`megatron/core/rerun_state_machine.py:36-40`)。要用起来,外层作业管理器必须认识这两个退出码——这是一条**框架之外**的前提。

**⑤ QK-clip 的代价按"层数 × 每步"计。**
`clip_qk` 对每个 attention 模块各做一次 `all_reduce(MAX)`(DP+CP 组)并各做一次 `.item()`(设备→主机同步)(`megatron/core/optimizer/qk_clip.py:50-57`)。另有两条静默边界:前向没记录 `current_max_attn_logits` 的层被 `continue` **直接跳过**(`:47-49`);`log_max_only=True` 时必须手工把它置 `None`,否则日志值会是陈旧值——注释即「When qk-clip is disabled, `clip_qk()` is not called and would otherwise **never reset** `current_max_attn_logits`」(`:58-65`)。

**⑥ 带 barrier 的 Timer 是一把双刃剑。**
`Timer` 类注释:「If this flag is passed, then all the caller processes will wait till all reach the timing routine. **It is up to the user to make sure all the ranks in `barrier_group` call it otherwise, it will result in a hang**」;且 `barrier_group` 默认 `None`,在 torch 分布式里等于**全局通信域**(`megatron/core/timers.py:113-118`、`:132`)。§2.1 说的"对齐后可跨 rank 比较"要按这个语义理解:它引入真实同步点,既改变被测对象,又有挂死风险。

**⑦ 能耗监控依赖 NVML,缺库时静默降级;且要求全 rank 参与。**
`pynvml` 导入失败时只是把 `has_nvml` 置 `False`,不报错(`megatron/core/energy_monitor.py:8-19`);类文档要求「**All ranks** in the process group are expected to call functions `lap()` and `get_total()`. Energy is monitored across all ranks and aggregated with an **all-reduce**」(`:22-27`)——漏调即失配。

**⑧ MoE 逐层日志要求全 PP rank 参与集合通信。**
无 MoE 层的 PP rank 必须 `force_initialize=True` 预建等长零张量,否则跨 PP 的 `all_reduce` 会因张量尺寸不一致而**挂死**(§2.2 的 `[!update]`)。

**⑨ aux-loss 缩放要求 `reduce_group` 存在,并按域付 all_reduce。**
`assert reduce_group is not None, "reduce_group is required for aux-loss scaling"`(`megatron/core/transformer/moe/router.py:618`);随后对 `aux_loss_scale_reduce_groups` 里**每个组**各做一次 `all_reduce`(`:620-621`)。这是 §0.2 ③ 那次改写付出的固定成本。

**⑩ 梯度范数超阈跳步只有配置项、且只在一条路径生效。**
`OptimizerConfig.grad_norm_skip_threshold` 默认 `float('inf')` 即关闭,docstring 只说「Skip gradient update if the gradient norm exceeds this threshold. Disabled by default.」(`megatron/core/optimizer/optimizer_config.py:394-397`);基线下 `megatron/training/` 里 `git grep grad_norm_skip_threshold` **零命中**——确实没有 CLI 开关,必须经 `OptimizerConfig` 注入(与 §1.2 `[!update]` 一致)。

**⑪ 指标口径存在版本断点,不可跨版本直接对比曲线。** aux/z-loss 的缩放在 `232c478d4` 前后、以及 `232c478d4` 与 `71092579` 之间各有一次口径变化(§1.7 的 `[!contradiction]`、§3.4 的 `[!update]`);启用 `mtp` 独立裁剪组后主 `grad norm` 不再涵盖 MTP head(§3.2 的 `[!update]`)。

---

## 6. 发展趋势

每条都锚在基线源码能读到的 DISCLAIMER / Caveats / 提交历史上;**方向解读是本页的推断,不是作者自陈**。

**① 重跑归因将从"单步"扩到"多步"——这是源码自己写的 future work。**
Caveats(2)的结尾原话:「**We're planning to add the capability to re-run multiple steps in a future implementation.**」(`megatron/core/rerun_state_machine.py:177-180`)。它正对着 §5 ③ 那条约束:当前查不出"上一步埋雷"。

**② RerunStateMachine 仍在 alpha,API 可能破坏性变更。** DISCLAIMER 明说「experimental features may break existing APIs」(`megatron/core/rerun_state_machine.py:29`),并要求把它的结论交给标准诊断套件复核(`:25-27`)。→ 本页 §1.4 的流程描述应按"会变"来读。

**③ 辅助损失的跨 rank 归一正在从"闭式因子"整体转向"实测计数"。**
同一段代码三个月内被改了三轮:#5047(`9ef8a2a42`,2026-06-02)引入闭式 `×|tp_cp|` → #4359(`7f9175207`,2026-06-17)换成沿 `aux_loss_scale_reduce_groups` 逐组 `all_reduce` 实测 → #5542(`d1384c2d9`,2026-07-14)再补 `valid_token_count` 把 padding token 排除在外。驱动力写在注释里:THD packing 与 dynamic CP 让"各 rank token 数相等"这个隐含前提不再成立(`megatron/core/transformer/moe/router.py:599-604`)。→ 推断:只要变长/打包/动态 CP 继续铺开,其余仍依赖闭式并行因子的归一化(z-loss 的 `!calculate_per_token_loss` 分支仍保留 `moe_z_loss_coeff / tp_cp_group.size()`,§1.7)迟早会走同一条路。

**④ MTP 相关开关正在收敛而不是扩张。** #5080 的 `mtp_isolated_loss` 被 #5223 删除并并入 `mtp_detach_heads`(§0.2 ④、§1.8 (b));基线下全仓 `git grep mtp_isolated_loss` 仍为 0 命中。→ 推断:MTP 的稳定性控制面倾向"一个主开关 + 自动派生行为",而非继续增加正交开关。

**⑤ 稳定性/可观测性代码里几乎没有 TODO,信息只能从提交历史读。**
对 `megatron/core/rerun_state_machine.py`、`fault_injector.py`、`energy_monitor.py`、`timers.py`、`optimizer/qk_clip.py`、`transformer/moe/moe_logging.py`、`transformer/moe/router_replay.py` 七个文件做 `git grep -n -E "TODO|FIXME|deprecat|WIP"`,基线下**只命中 `rerun_state_machine.py:21` 与 `:29` 两条 DISCLAIMER**。因此本节没有"某某待办"型趋势可写——这本身是个结论:这批模块的演进线索只存在于 PR 历史里。

---

## 7. 小结

- **数值稳定三道防线**:loss scaling(抬出下溢区)+ inf/nan 跳步(丢弃坏步)+ 全局梯度裁剪(削尖峰);MoE 另加 router fp32;attention 另加 QK-clip。
- **硬件 / SDC 防线**:`RerunStateMachine` 用**多级重跑归因**区分 transient(偶发位翻转)/ persistent(故障 GPU)/ correct(真实尖峰)—— 专治"算错但不报错"的静默数据损坏;NTP 提供 TP 容错;`fault_injector` 测试这些机制本身。
- **可观测三件套**:分级带 barrier 的 `Timer` 系统(性能分解、找 straggler)、MoE 逐层 + 过载因子 tracker、energy monitor;统一写 TensorBoard/wandb/one_logger。
- **核心指标**:收敛看 `lm loss`;数值健康看 `grad norm` / `loss scale` / `skipped & nan iters` / `num zeros`;性能看 `throughput` + Timer 分解;MoE 看 `overload factor`;硬件看 RerunStateMachine 归因。

> [!update] `ee3f1ff..HEAD` 增量补强 — 该特性自 `dev@232c478d4`(2026-06-16)引入,行号已重核至基线 `71092579`。
> - **数值稳定**新增"梯度范数超阈跳步"(§1.2,`grad_norm_skip_threshold`)作为裁剪之外的第四道闸:坏步整步丢弃而非缩放。
> - **MTP 稳定性套件**(§1.8):`mtp_detach_heads`(切回流 + 共享 output_weight 隔离 + 在线 RL `labels=None` 派生,已合并 #5080 原 `mtp_isolated_loss` 的能力)/ `mtp_grad_scale_func`(独立损失缩放)/ `mtp` 独立梯度裁剪组,把 MTP 辅助支路在前向图、损失缩放、梯度裁剪三层与主模型解耦。
> - **辅助损失正确性**:aux/z-loss 在 TP>1 + per-token-loss 下的梯度缩放修正(§1.7);DSA indexer loss 跨 micro-batch 平均(§1.7)。
> - **可观测性**:RerunStateMachine 去 stat 系统调用、确定性日志改按校验描述标识(§1.4);MoE logging 的 `record/report` 生命周期(§2.2);seqlen 统计保真、混合模型分阶段日志改传显式进程组(§2.5)。

---

*生成依据:`Megatron-LM` `dev` 分支 `71092579`(2026-08-27;由 `ee3f1ff` 重定基线而来,特性增量来自 `dev@232c478d4`)。全页行号(含 `[!update]` 小节)已统一重核至 `71092579`。`RerunStateMachine` 为实验特性。训练循环日志位于 `megatron/training/training.py`(在 `megatron/core` 之外)。*

## Related Pages

- [[16_megatron_distributed_optimizer_analysis]] · [[27_megatron_tp_fsdp_resharding_supplements_analysis]] · [[30_megatron_rl_posttraining_consistency_analysis]]
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]]
