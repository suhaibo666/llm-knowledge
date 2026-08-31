---
title: "D06 工业后训练框架对比"
---

# D06 工业后训练框架对比

> **阶段**：S02–S05
> **文档编号**：D06
> **快照日期**：2026-07-27；slime 列于 2026-08-14 单独重验
> **证据基线**：verl `983cb0f`、slime `681b3adc`、AReaL `b23fa6c`、ROLL `370cb24`
> **结论先行**：verl、slime、AReaL、ROLL 不是同一目标函数下的排行榜；它们分别优化通用可组合性、Megatron+SGLang 深集成、fully async 服务化和多 Strategy/异构硬件。
> **阅读导航**：[[01_posttraining_infra_mechanism_analysis|上一篇 D05]] · [[10_verl_end_to_end_iteration_analysis|下一篇 D07]]

---

## 1. 固定比较基线

| 框架 | commit | 本文角色 |
|---|---|---|
| verl | `983cb0f24443f87b3d161fad318445130a620b07` | 主 baseline，先贯通稳定同步链 |
| slime | `681b3adca54105d5ecd3fb822fa0dc58a427e0f9` | Megatron+SGLang 性能、训推一致性与稳定性对照；2026-08-14 重验 |
| AReaL | `b23fa6cf9c8edfebcf055079ab78913128bc4579` | fully async、agent service、freshness 对照 |
| ROLL | `370cb24c1036ea9145365478fcc40612b2186fc8` | 多 Strategy、资源映射、Ascend 对照 |

“更快”只在同模型、同硬件、同并行、同序列、同有效样本数和同 freshness 下有意义。本文不做脱离条件的性能排名。

## 2. 一页机制矩阵

| 维度 | verl | slime | AReaL | ROLL |
|---|---|---|---|---|
| 控制面 | Ray PPO trainer 与 role worker | 轻量 train loop + Ray groups | controller/workflow + v2 services | pipeline + Cluster/schedulers |
| 训练后端 | FSDP/Megatron 等 | Megatron 主路径 | FSDP/Megatron/Archon | HF/FSDP2/Megatron |
| rollout | vLLM/SGLang 等 | SGLang 深集成 | SGLang/vLLM remote | vLLM/SGLang Strategy |
| 核心数据 | `DataProto` | `Sample` + DataSource/buffer | trajectory + workflow task | `DataProto` + remote storage |
| 权重面 | worker/rollout update | NCCL/tensor/disk/delta | AWEX/disk/colocate gateway | `ModelUpdateGroup` |
| 同步主线 | 成熟 | 成熟 | 可配为同步 | 成熟 |
| fully async | experimental 独立路径 | warm queue rollout path | 核心设计与 freshness manager | `async_pipeline` 与 agentic 分支 |
| Agentic | rollout interface/loop 扩展 | custom generate + agent harness | workflow + online agent service | env manager + tools + proxy |
| TIM 工具 | correction/bypass/metrics | rollout log-prob、temperature/top-p replay、routing replay、TIS/OPSM 与 GLM-5 strict gate | behave/proximal 分离 | train-infer corrections |
| NPU | 官方 Ascend 扩展列出 vLLM/SGLang + FSDP/FSDP2/Megatron/MindSpeed；固定 upstream commit 仍需逐后端核验 | 本快照非主路径 | 独立 Ascend 文档/分支 + vLLM-Ascend | 主树 platform 抽象与 vLLM-Ascend |

## 3. 四级支持证据

| 级别 | 定义 | 本文标记 |
|---|---|---|
| P1 | 配置、类、adapter 存在 | 接口 |
| P2 | 固定 commit 调用链与示例可达 | 功能 |
| P3 | 有一致性、恢复或 correctness 测试 | 正确性 |
| P4 | 目标规模且条件完整的性能证据 | 性能 |

重要结论：

- “有 NPU platform 类”通常只是 P1；
- “有 end-to-end example”可到 P2；
- “GPU/NPU 曲线相近”可作为局部 P3/P4，但不能外推所有模型；
- README 宣称属于 C 级来源，必须回到源码/测试。

## 4. 控制面与可修改性

### 4.1 verl

| 维度 | 摘要 |
|---|---|
| 优点 | `RayPPOTrainer` 主循环可读地串起 reward/advantage/actor update/weight refresh；`DataProto` 贯穿 role 边界；算法 registry 已含 GRPO、GSPO、SAPO、CISPO、REINFORCE 等十余种 estimator/policy-loss |
| 代价 | abstraction 层和配置面较大；stable 同步链与 experimental fully async 是两条架构路径；理解性能问题需要跨 trainer、worker、rollout backend |

详见 [[01_verl_architecture_overview_analysis|verl 架构总览]]（hybrid-controller、`fit` 主循环、`DataProto`）与 [[15_verl_rl_algorithms_analysis|verl RL 算法]]（registry 机制、config key→代码锚点）。

### 4.2 slime

优点：

- `train.py:49-93` 的主循环非常直接；
- 深押 Megatron+SGLang，可以暴露上游后端的高级能力；
- weight transport 与 custom rollout 的扩展面清晰。

代价：

- 单 rollout backend 的设计降低通用后端可替换性；
- fully async 当前主要让 rollout producer/queue 跨轮保温；`train_async.py` 只做一拍 generate/train overlap，换权重前仍等待生成完成；
- README 中生态项目能力不能自动算进 slime core。

### 4.3 AReaL

优点：

- freshness admission control 是一等对象；
- workflow/trajectory 与 agent service 边界清晰；
- v2 weight-update gateway 将 pair、version 与 transfer mode 显式化。

代价：

- 论文、legacy controller 和 v2 service 同时存在，阅读时必须分代；
- 服务化增加部署、鉴权、网络和故障恢复面；
- fully async correctness 更依赖 version/log-prob contract。

### 4.4 ROLL

优点：

- Strategy 将训练/推理操作统一到 worker interface；
- `device_mapping` 能表达 colocated 与 disaggregated；
- 主树内有 NPU platform、dtype 修正和 vLLM-Ascend worker 选择。

代价：

- 多后端 patch 与版本兼容矩阵较大；
- Strategy 屏蔽接口，不会自动屏蔽 kernel、collective、dtype 和性能差异；
- RLVR `async_pipeline` 与 Agentic async 的数据语义需分别读。

## 5. Async 语义对照

| 框架 | 可确认实现 | 不是 |
|---|---|---|
| verl | experimental fully async policy 拥有单独 main/queue/rollouter/trainer | stable `RayPPOTrainer.fit` 自动 fully async |
| slime | background asyncio worker 持续产组、逻辑 backpressure、surplus 跨调用保温；另有一拍 generate/train overlap | 无版本 admission 上限的任意 replay，或生成中途换权重 |
| AReaL | producer admission + version staleness + workflow executor | 单纯把 `asyncio` 包在 rollout 外 |
| ROLL | pipeline flag 控制 generate/model update 与 scheduler pause | 所有 agent env 都天然 on-policy |

因此对“异步程度”不应只打一个勾。至少分：

```text
phase overlap
producer warm queue
partial rollout
bounded version lag
independent training service
independent inference service
```

## 6. 算法扩展面

| 目标 | verl | slime | AReaL | ROLL |
|---|---|---|---|---|
| 新 loss | `core_algos.py` registry | Megatron loss/ppo utils | PPO actor/loss config | actor worker + strategy |
| 新 rollout | rollout class/agent loop | `--rollout-function-path` | `RolloutWorkflow`/agent | scheduler/env manager |
| 新 reward | reward manager/function | RM hub/custom function | workflow reward | reward worker |
| 新 weight transport | worker/rollout path | updater class | gateway adapter | update group/strategy |
| 新硬件 | worker/backend 适配 | 训练/serving 双栈 | platform + branch/image | platform + strategy |

对工业修改，建议先选“最小变更面”而非“功能最多”：

- loss 研究与通用 baseline：verl；
- Megatron+SGLang 大规模路径：slime；
- 在线 agent 与 bounded async：AReaL；
- 多后端、异构与 Ascend 主树实验：ROLL。

## 7. 正确性与运营检查

| 检查 | verl | slime | AReaL | ROLL |
|---|---|---|---|---|
| policy version 字段 | 部分路径需追 meta | rollout id/weight version | 一等字段 | global step/model update |
| TIM correction | helper 与 loss 接口 | log-prob/routing replay | 双 ratio 指标 | correction utility |
| fault injection | 部分 worker/test | health monitor/CI injection | controller/service recovery | scheduler/cluster 机制 |
| checkpoint queue state | 需按路径核验 | DataSource save/load | recovery/version hook | pipeline checkpoint |
| weight atomicity | backend-dependent | engine pause/flush/version compare | gateway last-version commit | all worker refs |

这张表表示源码中的机制入口，不表示完成了统一规模的故障注入验证。

## 8. 选型树

```mermaid
flowchart TD
    Q["首要约束"] --> B["通用算法与可读主链"]
    Q --> P["Megatron SGLang 极致路径"]
    Q --> A["在线 Agent Fully Async"]
    Q --> H["多后端与 Ascend"]
    B --> V["verl"]
    P --> S["slime"]
    A --> R["AReaL"]
    H --> O["ROLL"]
```

真实项目常用“主框架 + 对照框架”：

- 用 verl 建 correctness baseline；
- 用 slime/AReaL 验证 overlap 收益；
- 用 ROLL/AReaL Ascend 路径做 NPU gap analysis。

## 9. 当前证据边界

- 本文是源码静态审计，不是四框架同条件 benchmark。
- slime README 列出的外部生态项目不计入 core P2。
- [Ascend 的 verl 安装指南](https://ascend.github.io/docs/sources/_generated/sources/verl/get_start/install_guidance.html)是 2026-05-20 的当前扩展能力证据；它不自动证明固定 upstream commit 的每种组合都达到 P3/P4。
- AReaL NPU 支持文档明确指向维护分支；主分支的 platform 类不能替代该分支验证。
- ROLL NPU 主树有较多 P1/P2 证据，但目标模型的 P3/P4 仍需按官方 Ascend guide 和真实硬件复测。
- 所有快速变化结论在 2026-08-26 后引用前应重验。

> [!warning] 有效期临近,且 verl 列已有确证的过期项(2026-08-11 状态检查)
> 本页快照基线为 2026-07-27,自设复核期 **2026-08-26,距今约 2 周**。逾期未重验则本页矩阵应降级为「历史快照」而非现状描述。
>
> **本次已确证 verl 列至少有两项过期**(依据 verl 官方 v0.8.0 release notes 与 v0.7 release blog,2026-08-11 核):
> 1. **入口脚本已更名**:"`main_ppo.py` is deprecated with a warning in favor of `main_ppo_sync.py`"。本页「控制面」一行的描述以 legacy 入口为准。
> 2. **数据面新增了 TransferQueue 通路**:v0.8.0 交付了 "New sync trainer with **TransferQueue** to decouple control flow and data flow in the single controller",而本页「核心数据」一行只记了 `DataProto`。
>
> **同时确证一项"未过期"**——避免反向误判:v0.7 blog 曾称计划"make this the default transmission method in v0.8",但 **v0.8.0 实际未使 TransferQueue 成为默认**(release notes 注明 "TBD: Fully async trainer with TransferQueue will be in next release")。故本页把 verl 数据面记为 `DataProto` 在**默认路径**上仍然正确,只是不再完整。展开见 [[16_verl_v1_transfer_queue_analysis]]。
>
> **重验时的已知障碍**:四框架的 commit 比对需要访问 GitHub,本次会话所处环境无法访问,故只能从官方文档侧做部分核验。slime / AReaL / ROLL 三列本次**未做任何重验**。
>
> **后续状态（2026-08-14）**：slime 本地 `origin/main@681b3adc` 已完成源码重验，本页 slime commit、TIM 与 async 语义已更新；AReaL / ROLL 仍维持 2026-07-27 快照。slime 的完整证据链见 [[slime/index]] 独立知识域。
>
> 另有两项已知边界,重验时一并处理:
> 1. **verl 列的基线与本库 verl 深潜页不一致**——本页 verl 列锁 `983cb0f`,而 [[02_engineering/04_posttrain_frameworks/verl/index|verl 分析域]] 已更新到另一冻结基线。跨页引用 verl 结论时须先对齐基线。
> 2. **对比集是四框架封闭集**（verl / slime / AReaL / ROLL）。Miles、SkyRL、NeMo-RL 在全库零命中,PRIME-RL 未进入本页。矩阵不覆盖 = 未评估,不等于不存在或不重要。

## Related Pages

- [[01_posttraining_infra_mechanism_analysis|D05 后训练 Infra 核心机制]]
- [[10_verl_end_to_end_iteration_analysis|D07 verl 端到端训练迭代]]
- [[01_slime_architecture_overview_analysis|D08 slime 架构]]
- [[21_areal_async_architecture_analysis|D09 AReaL 架构]]
- [[22_roll_strategy_and_ascend_analysis|D10 ROLL 与 Ascend]]
