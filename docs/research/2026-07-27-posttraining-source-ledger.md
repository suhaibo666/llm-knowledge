# 2026-07-27 起 LLM 后训练来源台账

> 用途：为 `wiki/03_posttraining/` 的 D02–D12 提供统一、可复核的论文版本、技术报告与源码定位。
> 框架快照日期：2026-07-27；D12 Kimi K3 报告增补核验：2026-07-28。
> 证据口径：A 为论文正文或固定 commit 源码；B 为同版本官方文档、测试或示例；C 为项目方声明；D 为本文机制推导。

## 1. 固定源码快照

| 框架 | 官方仓库 | 固定 commit | 主要用途 | 核验 |
|---|---|---|---|---|
| verl | [verl-project/verl](https://github.com/verl-project/verl) | `983cb0f24443f87b3d161fad318445130a620b07` | D02、D04、D06、D07 | detached HEAD，工作树干净 |
| slime | [THUDM/slime](https://github.com/THUDM/slime) | `aaf5c2092b01219fa0d5c2d323741d409086ca32` | D06、D08 | detached HEAD，工作树干净 |
| AReaL | [areal-project/AReaL](https://github.com/areal-project/AReaL) | `b23fa6cf9c8edfebcf055079ab78913128bc4579` | D03、D04、D06、D09、D11 | detached HEAD，工作树干净 |
| ROLL | [alibaba/ROLL](https://github.com/alibaba/ROLL) | `370cb24c1036ea9145365478fcc40612b2186fc8` | D03、D06、D10、D11 | detached HEAD，工作树干净 |
| Kimi K3 报告仓库 | [MoonshotAI/Kimi-K3](https://github.com/MoonshotAI/Kimi-K3) | `0797decb18ab079de86f991b87a64b81ec15a3c2` | D01–D05、D11、D12 | 官方报告仓库；无核心 RL 训练源码 |

### 1.1 D02–D12 证据覆盖

| 文档 | 主要一手证据 |
|---|---|
| D02 | DeepSeekMath、DAPO、Dr. GRPO、GSPO、SAO 固定论文版本；verl `core_algos.py` |
| D03 | RAGEN、Agent Lightning、SAO；三框架 sandbox/agent 示例 |
| D04 | AReaL、StreamRL、AsyncFlow、RollPacker、三组 TIM 工作 |
| D05 | 四框架 control/data/weight 源码链；AsyncFlow、RollPacker 的队列与长尾机制 |
| D06 | 四个固定 commit 的统一 P1–P4 能力审计 |
| D07 | verl stable 主链与 experimental fully async 源码 |
| D08 | slime Megatron/SGLang、DataSource、权重传输与 async producer 源码 |
| D09 | AReaL trainer、freshness manager、v2 services、Hermes 与 Ascend 分支文档 |
| D10 | ROLL RLVR/Agentic pipeline、Strategy、resource mapping、weight group 与 NPU platform |
| D11 | 四框架证据加 torch_npu、HCCL、MindSpeed、vLLM-Ascend、SGLang NPU 官方文档 |
| D12 | Kimi K3 Technical Report `0797decb` §4.1、§4.2、§5.3、Appendix F；官方固定仓库树与本地 PDF hash |

## 2. 算法来源

| 主题 | 固定来源与承重定位 | 支持的结论 | 证据 |
|---|---|---|---|
| GRPO | [DeepSeekMath, arXiv:2402.03300v3](https://arxiv.org/abs/2402.03300v3)，§4.1.1，Eq. 21，Algorithm 1 | 同题组采样、组内 reward 标准化、去 critic、token ratio 与直接 KL | A |
| R1-Zero/R1 | [DeepSeek-R1, arXiv:2501.12948v2](https://arxiv.org/abs/2501.12948v2)，§2.2–2.3，Table 2 | 规则奖励、纯 RL 与 cold-start 多阶段路线的边界 | A |
| DAPO | [DAPO, arXiv:2503.14476v2](https://arxiv.org/abs/2503.14476v2)，§3.1–3.4，Eq. 8–13，Table 1 | Clip-Higher、动态采样、token-level loss、overlong shaping | A |
| Dr. GRPO | [Understanding R1-Zero-Like Training, arXiv:2503.20783v2](https://arxiv.org/abs/2503.20783v2)，§3.1–3.2，Fig. 4–5，Table 2 | response-length divisor 与 group-std 会改变有效权重；提出无偏归一化 | A |
| GSPO | [GSPO, arXiv:2507.18071v2](https://arxiv.org/abs/2507.18071v2)，§4.1–4.3，Eq. 5、10、13、17，Fig. 1 | sequence likelihood ratio、整条 response clipping、GSPO-token 等价条件 | A |
| SAO | [SAO, arXiv:2607.07508v1](https://arxiv.org/abs/2607.07508v1)，§3.1–3.3，Eq. 3–5，Fig. 2 | 单 rollout、rollout log-prob、双侧 token mask、value 模型补偿 | A |

## 3. Agentic 与 Credit 来源

| 主题 | 固定来源与承重定位 | 支持的结论 | 证据 |
|---|---|---|---|
| 长轨迹稳定性 | [RAGEN, arXiv:2504.20073v2](https://arxiv.org/abs/2504.20073v2)，§3–4，Fig. 2 | StarPO、Echo Trap 诊断、trajectory filtering 与 critic/gradient stabilization | A |
| Agent–trainer 解耦 | [Agent Lightning, arXiv:2508.03680v1](https://arxiv.org/abs/2508.03680v1)，§3.2–3.3，Fig. 2 | 以 MDP 数据接口解耦 agent 执行；分层 credit 把 episode return 分到 LLM call | A |
| 单样本 agentic RL | [SAO, arXiv:2607.07508v1](https://arxiv.org/abs/2607.07508v1)，§3，Fig. 2 | 不等同“取消 baseline”；以 critic 承担单样本高方差问题 | A |
| coding sandbox | AReaL `examples/sandbox_daytona/reward_example.py:11`；ROLL `roll/pipeline/rlvr/rewards/code_sandbox_reward_worker.py`；slime `examples/coding_agent_rl/swe.py:70-364` | 代码任务的环境、超时、patch 与 grader 属于 rollout 数据面 | A/B |

## 4. Async、Freshness 与 TIM 来源

| 主题 | 固定来源与承重定位 | 支持的结论 | 证据 |
|---|---|---|---|
| Fully async | [AReaL, arXiv:2505.24298v5](https://arxiv.org/abs/2505.24298v5)，§3–4 | generation 与 training 解耦，以 version staleness 控制吞吐/偏差 | A |
| 异构解耦 | [StreamRL, arXiv:2504.15930v1](https://arxiv.org/abs/2504.15930v1)，§3–4 | rollout/training 解耦、跨资源池与通信成本转移 | A |
| 流式生产消费 | [AsyncFlow, arXiv:2507.01663v1](https://arxiv.org/abs/2507.01663v1)，§3–4 | storage/transfer 解耦、partial rollout、producer-consumer 与 bounded staleness | A |
| freshness-preserving tail control | [RollPacker, arXiv:2509.21009v1](https://arxiv.org/abs/2509.21009v1)，§3–4 | 通过尾部打包减少同步 batch 长尾，而非允许任意旧样本 | A |
| TIM 因果诊断 | [Diagnosing TIM, arXiv:2605.14220v1](https://arxiv.org/abs/2605.14220v1)，§3.1–3.2、§4.1–4.2，Fig. 2–6 | VeXact 零 mismatch 对照；小数值漂移可独立触发 collapse；补丁不能替代诊断基线 | A |
| TIM 动态耦合 | [Beyond Precision, arXiv:2602.01826v1](https://arxiv.org/abs/2602.01826v1)，§3、§4.1–4.4，Fig. 7–13 | mismatch 与 update size/response length 动态耦合；LR 调度与 IS 处理不同问题 | A |
| inference policy 目标 | [MIPI/MIPU, arXiv:2606.29526v1](https://arxiv.org/abs/2606.29526v1)，§4.1，Algorithm 1，Table 1–2 | 同参数仍可能有 train/inference policy 差异；candidate update 需接受测试 | A |

## 5. Kimi K3 技术报告基线

| 来源 | 固定版本与承重定位 | 支持的结论 | 证据 |
|---|---|---|---|
| Kimi K3 Technical Report | [MoonshotAI/Kimi-K3 `0797decb`](https://github.com/MoonshotAI/Kimi-K3/commit/0797decb18ab079de86f991b87a64b81ec15a3c2)，47 页；PDF SHA-256 `fd6ee35c07766a5eb6104235f1b407e4329f969e3482b8c42937c7b5f2b3efe1`；本地 `raw/01_theory/01_models/moonshot_kimi/Kimi_K3_Technical_Report_2026-07-28.md` | 固定 2026-07-28 官方报告正文；仓库未含核心 RL 训练实现 | A/C |
| 三阶段后训练 | §4.1、pp.12–14；Eq. 15–16 | SFT → 3 领域 × 3 effort 专家 RL → MOPD；reasoning budget、GRM、deployment-aware QAT 与 draft fine-tuning | A |
| Partial rollout | §4.1.2，p.13 | \(\lambda NK\) phase gate、prompt 内 \(K\) group、跨 iteration resume 与 extreme off-policy 声明 | A/C |
| White-box environments | §4.2.1–4.2.7、pp.14–16；Fig. 9–10 | 动态 harness、任务合成、kernel/personal-assistant/AET/web-dev verifier 设计 | A |
| 1M Agentic RL infra | §5.3、pp.21–22 | co-location、external KV write-back、auto-throttling、gradient-buffer reuse、AgentENV lifecycle | A/C |
| XTML trajectory protocol | Appendix F、pp.46–47；Fig. 16 | option zones、think/response/tools channels、preserved thinking、typed tool calls | A |

报告没有披露 \(N,K,\lambda,\tau,\sigma,R_{\max}\)、完整 stale-data regularizer、RL 总计算量、组件消融或 trainer/rollout/weight-sync 源码。D12 中由机制推出的 schema 和设计检查项统一标为综合判断，不反写成项目方实现事实。

## 6. 四框架源码地图

### 6.1 verl

- 入口/编排：`verl/trainer/main_ppo.py:34,103,167-168`；`verl/trainer/ppo/ray_trainer.py:286,772,1380`。
- 数据：`verl/protocol.py:318,721,781,963,971` 的 `DataProto` 及变换。
- 一轮迭代：`verl/trainer/ppo/ray_trainer.py:1488,1642,1665,1690-1691`。
- 算法：`verl/trainer/ppo/core_algos.py:1279-1358,1538-1594,2007,2292-2456`。
- TIM correction：`verl/trainer/ppo/rollout_corr_helper.py:554-601`。
- 权重：`verl/workers/engine_workers.py:705-725,783-787`；`verl/workers/rollout/vllm_rollout/vllm_rollout.py:271-320`。
- fully async 边界：`verl/experimental/fully_async_policy/README.md:64`；`verl/experimental/fully_async_policy/fully_async_main.py:25-29,222`。

### 6.2 slime

- 入口/同步主循环：`train.py:9-93`。
- Ray role：`slime/ray/rollout.py:427,552-680`；`slime/ray/actor_group.py:13,130-178`。
- 数据面：`slime/rollout/data_source.py:17,50,168,225`；`slime/ray/rollout.py:711-866`。
- SGLang 生成：`slime/rollout/sglang_rollout.py:153,224,294,375,617`。
- Megatron update：`slime/backends/megatron_utils/actor.py:364-539,567-627`。
- weight transport：`slime/backends/megatron_utils/actor.py:150-181`；`slime/ray/actor_group.py:161-268`。
- fully async：`slime/rollout/fully_async_rollout.py:76-256`；官方 smoke test 入口 `examples/fully_async/README.md:3-83`。

### 6.3 AReaL

- 训练主链：`areal/trainer/rl_trainer.py:105,307-434,605-816`。
- rollout/任务：`areal/infra/controller/rollout_controller.py:74`；`areal/infra/workflow_executor.py:263,747`。
- freshness：`areal/infra/staleness_manager.py:20-181`，核心 capacity 公式在 `92-112`。
- 三策略概率：`docs/en/best_practices/algo_perf.md:54-81`。
- v2 服务：`areal/v2/training_service/controller/controller.py:31`；`areal/v2/agent_service/protocol.py:29-318`。
- 权重服务：`areal/v2/weight_update/gateway/app.py:170-757`，AWEX/disk/colocate 三路径。
- NPU：`docs/en/tutorial/installation_npu.md:1-182`；`areal/infra/platforms/npu.py:14-30`。

### 6.4 ROLL

- RLVR：`roll/pipeline/rlvr/rlvr_pipeline.py:121,197-374,459-700`。
- async 分支：`roll/pipeline/rlvr/rlvr_pipeline.py:454,477-565`。
- Agentic：`roll/pipeline/agentic/agentic_pipeline.py` 与 `agentic_rollout_pipeline.py`。
- Strategy：`roll/distributed/strategy/factory.py:11-28`；`roll/distributed/strategy/strategy.py:52-408`；vLLM/SGLang/Megatron/FSDP2 实现。
- 资源映射：`roll/distributed/scheduler/resource_manager.py:12-142`。
- 权重同步：`roll/distributed/executor/model_update_group.py:9-37`。
- NPU：`roll/platforms/__init__.py:16-43`；`roll/platforms/npu.py:11-105`；`roll/distributed/scheduler/protocol.py:230-240,378-386`。

## 7. 使用约束

1. “支持”必须分接口存在、调用链可达、正确性验证和目标规模性能四级。
2. 论文 benchmark 数字不能脱离版本、模型、硬件、序列和采样条件。
3. 当前台账只证明固定快照中的事实；30 天后引用快速变化源码前应重验。
4. `file:line` 只有与 40 位 commit 一起使用才是稳定证据。
5. 技术报告中的项目级设计与自报规模不自动等于公开源码支持；K3 不加入 D06 的四框架 P1–P4 矩阵。
