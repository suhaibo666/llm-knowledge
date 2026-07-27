# 03 LLM 后训练纵向学习域

> **当前阶段**：S00–S05 全部完成
> **快照日期**：2026-07-27
> **主线**：Reasoning RL + Agentic/Coding RL + 工业级训练系统
> **核心框架**：verl、slime、AReaL、ROLL
> **硬件视角**：NVIDIA/CUDA 主线与 Ascend/NPU 映射

---

## 1. 为什么建立一个统一领域

过去的后训练资料分别位于：

- `wiki/01_theory/04_posttraining/`：算法、论文和对齐方法；
- `wiki/02_engineering/04_posttrain_frameworks/`：verl、RL infra 和 sandbox。

这种分类适合按学科查资料，却会割裂一次真实 RL 迭代。算法中的 importance ratio、group sampling、sequence clipping 和 staleness 假设，必须与 rollout、buffer、权重同步、训练—推理一致性和资源调度一起分析。

因此，2026 后训练前沿研究新增的 D00–D11 全部写入本目录。旧页面保持原位，通过链接复用，不迁移、不复制。

本领域要建立的不是论文清单，而是三种连续能力：

1. 从公式识别算法对在线数据的真实要求；
2. 从配置和入口追踪工业框架的一次完整训练迭代；
3. 从 CUDA 基线判断 Ascend 适配的功能、正确性与性能缺口。

---

## 2. 从这里开始

1. 先读 [[03_posttraining/00_posttraining_source_reading_guide|D00 LLM 后训练前沿源码学习路线]]，了解完整顺序和六级能力门槛。
2. 再读 [[03_posttraining/01_posttraining_frontier_map_analysis|D01 后训练前沿全景地图]]，建立算法—数据—系统—硬件的统一坐标。
3. 按 D02 → D11 顺序进入机制、源码与硬件深挖。

如果已经熟悉 policy gradient，可使用 D00 中的短路线：`D00 → D01 → D02 → D04 → D05 → D07 → D11`。

---

## 3. S00–S05 研究阶段

| 阶段 | 研究目标 | 主要产出 | 完成标志 | 状态 |
|---|---|---|---|---|
| S00 | 建立 2026-07-27 快照、固定框架 baseline、统一入口 | D00、D01、index | 能解释研究边界、阅读顺序、四框架角色和证据规则 | 已完成 |
| S01 | 建立算法与 Infra 的统一坐标系 | D02–D05 | 能从公式推到数据 schema，并分析 freshness、TIM 和系统闭环 | 已完成 |
| S02 | 建立框架矩阵并贯通 verl | D06 首版、D07 | 能从配置追完一次 verl 迭代及 weight refresh | 已完成 |
| S03 | 用 slime 与 AReaL 对照性能和 fully async | D08、D09，更新 D06 | 能比较同步、流式和 fully async 的收益来源与代价 | 已完成 |
| S04 | 深挖 ROLL、异构和 Ascend | D10、D11，更新 D06 | 能给出 CUDA–Ascend 组件差距与验证矩阵 | 已完成 |
| S05 | 综合、验收和持续追踪 | 复核 D00、D01、D06 | 所有结论有固定版本，快速变化页面有 staleness 标记 | 已完成 |

---

## 4. D00–D11 文档顺序

| 编号 | 文档 | 阶段 | 角色 | 状态 |
|---|---|---|---|---|
| D00 | [[03_posttraining/00_posttraining_source_reading_guide|LLM 后训练前沿源码学习路线]] | S00/S05 | 总入口、能力门槛和阅读方法 | 已完成 |
| D01 | [[03_posttraining/01_posttraining_frontier_map_analysis|后训练前沿全景地图]] | S00 | 当前前沿、基线和深挖队列 | 已完成 |
| D02 | [[03_posttraining/02_reasoning_rl_algorithm_evolution_analysis|Reasoning RL 算法演进]] | S01 | GRPO、DAPO、GSPO 及后续演进 | 已完成 |
| D03 | [[03_posttraining/03_agentic_rl_algorithm_analysis|Agentic RL 算法与环境]] | S01 | trajectory、reward、credit 与 agent runtime | 已完成 |
| D04 | [[03_posttraining/04_on_policy_off_policy_staleness_analysis|On-policy、Off-policy 与 Staleness]] | S01 | policy lag、correction 与 TIM | 已完成 |
| D05 | [[03_posttraining/05_posttraining_infra_mechanism_analysis|后训练 Infra 核心机制]] | S01 | control/data/weight 三平面 | 已完成 |
| D06 | [[03_posttraining/06_framework_comparison|工业后训练框架对比]] | S02/S05 | 四框架的统一机制矩阵 | 已完成 |
| D07 | [[03_posttraining/07_verl_end_to_end_iteration_analysis|verl 端到端训练迭代]] | S02 | 主基线的真实源码调用链 | 已完成 |
| D08 | [[03_posttraining/08_slime_architecture_analysis|slime 高性能与异步架构]] | S03 | 性能、数据面与 staleness 对照 | 已完成 |
| D09 | [[03_posttraining/09_areal_async_architecture_analysis|AReaL Fully Async 与 Agentic 架构]] | S03 | fully async 与服务化 agent loop | 已完成 |
| D10 | [[03_posttraining/10_roll_strategy_and_ascend_analysis|ROLL Strategy、异构与 Ascend]] | S04 | 多后端 Strategy 和 NPU 专项 | 已完成 |
| D11 | [[03_posttraining/11_cuda_ascend_posttraining_stack_comparison|CUDA–Ascend 后训练栈对照]] | S04 | 跨硬件的能力与差距矩阵 | 已完成 |

---

## 5. 四个框架的研究分工

| 框架 | 研究角色 | 重点问题 | 对应文档 |
|---|---|---|---|
| verl | 主基线 | 一次训练迭代、角色编排、算法扩展、weight sync/reshard | D07 |
| slime | 性能与前沿对照 | Megatron + SGLang、DataSource/buffer、weight transport、async producer | D08 |
| AReaL | Fully async/Agentic 对照 | 微服务、Hermes、policy lag、agent trajectory | D09 |
| ROLL | 多后端/异构/Ascend 专项 | Strategy、AutoDeviceMapping、RLVR 与 Agentic async 差异 | D10 |

四者不按单一性能榜排名。D06 已在固定版本与 freshness 语义下比较机制与证据；真实性能仍需同条件实测。

---

## 6. 既有知识入口

这些页面仍是有效的背景材料，但快速演进的实现结论需要在 D02–D10 中重新绑定版本。

### 算法与对齐背景

- [[01_theory/04_posttraining/index|旧后训练理论目录]]
- [[01_theory/04_posttraining/grpo_analysis|GRPO 原理与实现]]
- [[01_theory/04_posttraining/dapo_analysis|DAPO 深度解析]]
- [[01_theory/04_posttraining/gspo_analysis|GSPO 深度解析]]
- [[01_theory/04_posttraining/reward_hacking_defense_analysis|Reward Hacking 防御]]

### Infra 与源码背景

- [[02_engineering/04_posttrain_frameworks/index|旧后训练框架目录]]
- [[02_engineering/04_posttrain_frameworks/verl/index|verl 既有分析索引]]
- [[02_engineering/04_posttrain_frameworks/rl_infra_efficiency_analysis|RL Infra 效率分析]]
- [[02_engineering/04_posttrain_frameworks/rl_sandbox_design_analysis|RL Sandbox 设计]]

现有 verl 系列使用较早 baseline `8a694930`，应作为历史解释而非 2026-07-27 当前源码事实。

---

## 7. 维护规则

- 新增的本研究文档只进入 `wiki/03_posttraining/`。
- 旧理论/工程页面只链接，不在新目录复制全文。
- 阶段使用 S00–S05，文档使用 D00–D11；文件名两位数字前缀与阅读顺序一致。
- 论文结论绑定 arXiv ID 和版本；代码结论绑定 repo、branch、commit、日期和 `file:line`。
- 区分来源事实、机制推导、项目方声明和研究判断。
- 性能数字必须附硬件、模型、并行、batch、序列长度和对照条件。
- 快速演进页面超过 30 天未核验时标记 staleness。
- 每完成一个阶段，更新本页状态、D00 路线和 `wiki/changelog.md`。

---

## Related Pages

- [[03_posttraining/00_posttraining_source_reading_guide|D00 LLM 后训练前沿源码学习路线]]
- [[03_posttraining/01_posttraining_frontier_map_analysis|D01 后训练前沿全景地图]]
- [知识库总索引](../index.md)
- [[01_theory/04_posttraining/index|旧后训练理论目录]]
- [[02_engineering/04_posttrain_frameworks/index|旧后训练框架目录]]
