# Megatron-LM 训练稳定性与可观测性深度解析

> 代码基准:`Megatron-LM/` 子仓库 `dev` 分支,commit `ee3f1ff`
> 核心文件:`megatron/core/rerun_state_machine.py`、`fault_injector.py`、`energy_monitor.py`、`timers.py`、`optimizer/qk_clip.py`、`optimizer/grad_scaler.py`、`optimizer/clip_grads.py`、`transformer/moe/moe_logging.py`、`transformer/moe/router_replay.py`;训练循环日志在 `megatron/training/training.py`
> 配套阅读:`optimizer_internals_analysis.md`、五份并行分析、`tp_fsdp_resharding_supplements_analysis.md`
> 定位:系统性专题。前面所有文档讲"怎么把模型并行训起来、训得快";本文讲**怎么让它训得稳、出问题怎么发现、看哪些指标判断健康**。

---

## 0. 总览

大规模训练(几千卡、数月)面对三类系统性风险:

1. **数值不稳定** —— 梯度/激活溢出、loss 尖峰、attention logit 爆炸。
2. **硬件故障与静默错误** —— GPU 宕机(显性)、**静默数据损坏 SDC**(GPU 算错但不报错)。
3. **不可观测** —— 训练跑着但不知道健不健康、慢在哪、为什么发散。

Megatron 对应有三套机制,本文依次拆解:

| 风险 | 机制 | 代码 |
|------|------|------|
| 数值不稳定 | loss scaling、梯度裁剪、NaN/Inf 检查、QK-clip | `grad_scaler.py`、`clip_grads.py`、`qk_clip.py` |
| 硬件故障 / SDC | RerunStateMachine、fault injector、NTP 容错 | `rerun_state_machine.py`、`fault_injector.py`、`nonuniform_tp.py` |
| 可观测性 | Timer 系统、MoE 逐层指标、能耗监控、TB/wandb 日志 | `timers.py`、`moe_logging.py`、`energy_monitor.py` |

---

## 1. 训练稳定性机制

### 1.1 数值溢出防护:loss scaling + inf/nan 跳步

(详见 `optimizer_internals_analysis.md` §3–4,此处定位为"稳定性"视角)

fp16 训练里梯度可能**下溢成 0** 或**上溢成 inf**。两道防线:
- **Loss scaling**:loss 乘 `S`,把小梯度抬出下溢区;`DynamicGradScaler` 自适应 `S`(`grad_scaler.py:64`)。
- **inf/nan 跳步**:`optimizer.step()` 第一步 `prepare_grads` 扫描梯度,发现非有限值 → **`found_inf_flag=True` → 整步丢弃**(参数不动),`DynamicGradScaler` 随后按 `backoff_factor` 调小 `S`。

这是 fp16 训练的安全阀:偶发溢出不会污染参数,只是浪费一步。

### 1.2 梯度尖峰防护:全局梯度裁剪

`clip_grads.py` / `MegatronOptimizer.clip_grad_norm`:算全局梯度范数 `‖g‖`(需跨 TP×PP all-reduce),超过 `--clip-grad`(默认 1.0)就等比缩小。挡住偶发的梯度尖峰,防止单步大跳导致发散。

### 1.3 NaN / Inf 显式检查

`--check-for-nan-in-loss-and-grad`:每步显式检查 loss 与梯度是否含 NaN/Inf。`rerun_state_machine.py` 即使没调 `validate_result` 也会在这个选项下兜底检查(`:583`)。发现后可选择**报错退出**或交给 RerunStateMachine 归因(§1.4)。FSDP 侧另有 `_check_nan_in_grad`。

### 1.4 RerunStateMachine —— 静默数据损坏(SDC)归因

`rerun_state_machine.py`(1425 行)。**这是 Megatron 最有特色的稳定性机制**,专治"GPU 算错了但不崩"的 SDC。

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

### 1.5 QK-clip —— 注意力 logit 稳定

`optimizer/qk_clip.py`。已知的一类训练不稳:attention 的 `Q·Kᵀ` logit 数值越训越大,softmax 进饱和区、梯度异常。`clip_qk(model)`:
- 遍历各层 `self_attention`,读 `core_attention.current_max_attn_logits`(前向时记录的本层最大 attention logit)。
- 跨 DP(含 CP)组 `all_reduce(MAX)` 得全局最大 logit。
- `log_max_only=False` 时调 `clip_qk()` 实际裁剪;`log_max_only=True` 则**只监控不裁剪**(把最大 logit 当指标看)。

返回的 `log_max_attention_logit` 既是稳定手段、也是一个可观测指标。

### 1.6 容错:NTP 与故障注入

- **NTP(Nonuniform TP)**:TP 组留备用 rank,核心 rank 故障时重分片续训,免整体重启 —— 详见 `tp_fsdp_resharding_supplements_analysis.md` §2。
- **`fault_injector.py`**(233 行):**主动注入故障**用于测试容错路径 —— 验证 RerunStateMachine、NTP 等机制是否真能正确响应。是"测试稳定性机制本身"的工具。

### 1.7 MoE 稳定性

MoE 有独有的不稳定源 —— 路由:
- **router fp32**:`--moe-router-dtype fp32` —— 路由 logit 保持 fp32(README 强调:高专家数下 bf16 路由精度不足,专家输出按路由分加权累加会放大误差)。
- **负载均衡损失**:`aux_loss` 等防止专家路由坍塌(`ep_analysis.md` §4)。
- **`router_replay.py`**(207 行):记录/重放路由决策 —— 用于复现和调试路由相关的不确定性。

---

## 2. 监控基础设施

### 2.1 Timer 系统(`timers.py`)

`Timer` / `DummyTimer`(`timers.py:35`)。给训练各阶段计时:`forward-compute`、`backward-compute`、`optimizer`、`batch-generator`、`forward-backward`、`optimizer-clip-main-grad`、`optimizer-count-zeros` 等(前几份文档的 `config.timers('...')` 调用即此)。

要点:
- **`log_level` 分级**:每个 timer 有 log level,低于阈值的用 `DummyTimer`(零开销空实现)—— 细粒度计时不污染生产性能。
- **`barrier` 选项**:计时前可选 `torch.distributed.barrier()`,得到对齐的、可跨 rank 比较的耗时(用于发现 straggler);不加 barrier 则是本 rank 异步耗时。

### 2.2 MoE 逐层指标(`moe_logging.py`)

`moe_logging.py`(745 行)有两个全局 tracker:
- **`MoEMetricsTracker`**:逐层收集 MoE 指标(各层 aux loss、z-loss 等),`--moe-per-layer-logging` 开启。能看出**哪一层**路由出问题,而不只是全局平均。
- **`MoEOverloadFactorTracker`**(`:95`):跟踪**专家过载因子**(overload factor)——`max_expert_load / mean_load`,即 `ep_analysis.md` §4 的负载不均衡因子 `f`。`--log-moe-overload-factor` 开启;跨 `tp_ep` 与 `expt_dp` 组做 MAX 规约,反映最坏专家的过载程度。

### 2.3 能耗监控(`energy_monitor.py`)

`energy_monitor.py`(95 行):采集 GPU 能耗,算每步/每 token 的能量 —— 大规模训练的成本与碳足迹指标。

### 2.4 日志后端:TensorBoard / wandb / one_logger

`megatron/training/training.py` 的 `training_log` 把指标同时写三处:`writer`(TensorBoard)、`wandb_writer`、`one_logger`。各指标有独立开关(`--log-loss-scale-to-tensorboard`、`--log-throughput` 等)。控制台还会拼一行 `log_string` 打印关键指标。

---

## 3. 指标目录(按类别)

下面是训练循环(`training.py` 的 `training_log`)实际产出、可用于分析的指标。

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

---

## 5. 小结

- **数值稳定三道防线**:loss scaling(抬出下溢区)+ inf/nan 跳步(丢弃坏步)+ 全局梯度裁剪(削尖峰);MoE 另加 router fp32;attention 另加 QK-clip。
- **硬件 / SDC 防线**:`RerunStateMachine` 用**多级重跑归因**区分 transient(偶发位翻转)/ persistent(故障 GPU)/ correct(真实尖峰)—— 专治"算错但不报错"的静默数据损坏;NTP 提供 TP 容错;`fault_injector` 测试这些机制本身。
- **可观测三件套**:分级带 barrier 的 `Timer` 系统(性能分解、找 straggler)、MoE 逐层 + 过载因子 tracker、energy monitor;统一写 TensorBoard/wandb/one_logger。
- **核心指标**:收敛看 `lm loss`;数值健康看 `grad norm` / `loss scale` / `skipped & nan iters` / `num zeros`;性能看 `throughput` + Timer 分解;MoE 看 `overload factor`;硬件看 RerunStateMachine 归因。

---

*生成依据:`Megatron-LM` `dev` 分支 `ee3f1ff`。源码行号以该 commit 为准。`RerunStateMachine` 为实验特性。训练循环日志位于 `megatron/training/training.py`(在 `megatron/core` 之外)。*

## Related Pages

- [[optimizer_internals_analysis]] · [[tp_fsdp_resharding_supplements_analysis]] · [[rl_posttraining_consistency_analysis]]
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]]
