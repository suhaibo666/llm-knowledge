# LLM 后训练前沿与工业源码研究设计

> **设计日期**：2026-07-27
> **知识库基线**：`llm-knowledge@c884b9d`（`main`）
> **研究目标**：建立可持续更新的后训练技术地图，并形成能够追踪工业级源码主链路的学习体系
> **研究主线**：Reasoning RL + Agentic/Coding RL
> **平台边界**：NVIDIA/CUDA 为上游基线，昇腾/NPU 为适配与差异分析对象

## 1. 目标与完成定义

本研究服务两个核心目标：

1. **前沿认知**：理解当前 LLM 后训练的算法谱系、关键争议、系统约束和工业演进方向，能够判断新工作究竟改进了算法、数据、环境、调度还是底层执行。
2. **源码理解**：能够从公开入口追踪一次完整训练迭代，解释数据、控制流、张量、进程、资源和权重如何在训练与推理组件之间流动。

研究完成后，读者应能：

- 用统一坐标系比较 Reasoning RL 与 Agentic RL 的主要算法；
- 解释 on-policy、off-policy、staleness、importance sampling 和 train-inference mismatch 的关系；
- 追踪 `prompt → rollout → reward → advantage → update → weight sync` 的端到端链路；
- 比较 verl、slime、AReaL 和 ROLL 的核心抽象与系统取舍；
- 定位吞吐、显存、通信、长尾和容错瓶颈；
- 说明 CUDA 基线能力在昇腾/NPU 上的对应实现、适配点和缺口；
- 按学习路线独立阅读新论文、新框架和新版本，而不是依赖二手总结。

## 2. 研究边界

### 2.1 必要基础层

只补足理解前沿所需的基础，不写传统对齐算法百科：

- SFT、reward model、rule-based verifier 与 model-based judge；
- PPO、policy gradient、advantage estimation、KL control；
- DPO 类离线偏好优化的定位与边界；
- on-policy、off-policy、importance sampling 与 policy lag。

### 2.2 前沿算法层

围绕两条主线组织：

- **Reasoning RL**：GRPO、DAPO、GSPO、RLVR、采样策略、难度调度、长推理、token/sequence-level objective、训练—推理不一致；
- **Agentic RL**：多轮 rollout、工具调用、环境反馈、trajectory/step-level credit assignment、长时程任务、异步训练和策略陈旧度。

算法清单不以名称堆砌为目标。只有同时满足“问题清楚、机制可验证、实现或实验可追踪”的工作才进入主干；其余工作进入前沿雷达候选区。

### 2.3 Infra 机制层

重点研究决定工业可用性的机制：

- colocated、hybrid 与 disaggregated 部署；
- 同步、流水化、部分异步和 fully-async 训练；
- rollout 调度、continuous batching、动态 batching 与长尾治理；
- 权重同步、参数格式转换、训练—推理重分片和版本管理；
- reward/verifier 流水线与异步计算；
- sandbox/environment 的隔离、扩缩容和失败处理；
- checkpoint、容错、背压、可观测性和数值一致性；
- 显存、通信、吞吐、GPU 利用率与成本模型。

### 2.4 框架源码层

研究样本固定为四个层次：

| 定位 | 框架 | 主要研究价值 |
|---|---|---|
| 主基线 | verl | 完整生态、HybridFlow、算法覆盖、训练与 rollout 编排 |
| 性能主对照 | slime | Megatron + SGLang、训练推理解耦、高性能与异步路径 |
| Agentic/异步主对照 | AReaL | fully-async RL、agent runtime 解耦、服务化接入与 staleness 控制 |
| 专题对照 | ROLL | 多角色架构、多后端 Strategy、异构资源映射与昇腾适配 |

必要时下钻 Megatron-Core、FSDP、Ray、SGLang 和 vLLM，但只追踪后训练关键路径，不扩展成通用训练或推理框架全量分析。

### 2.5 硬件映射层

- NVIDIA/CUDA 作为算法、通信、推理和性能实现基线；
- 昇腾/NPU 按组件逐项映射可直接复用、需要适配和当前缺失的能力；
- 重点比较集合通信、显存管理、推理引擎、权重同步、算子支持、动态 shape、性能分析和故障诊断。

## 3. 总体方法：双轨交叉

### 3.1 横向前沿雷达

持续维护以下分类：

- 算法与 objective；
- 数据生成、采样和 curriculum；
- reward、verifier 与 reward hacking；
- Agent environment 与 sandbox；
- 同步/异步 RL Infra；
- 训练框架、推理引擎和硬件支持；
- 工业公开实践与可验证性能数据。

雷达记录“发布日期、来源版本、解决问题、核心机制、证据强度、公开实现、与既有工作的差异、是否进入深挖队列”。

### 3.2 纵向源码解剖

每个框架至少追踪一条真实可执行路径：

`配置入口 → driver/trainer → 资源编排 → rollout → reward → advantage → actor update → 权重同步 → 下一轮 rollout`

源码分析必须继续下钻：

- 关键类、函数和数据结构；
- batch、trajectory、token 和张量 shape 的变化；
- Ray actor、进程组、device mesh 或 RPC 的边界；
- 模型权重的所有权、版本和格式；
- 并行策略、重分片、offload/reload 与通信；
- 背压、超时、异常、重试和 checkpoint；
- 性能瓶颈、隐藏约束和可扩展点。

### 3.3 交叉矩阵

横向算法与纵向源码通过以下固定字段关联：

| 字段 | 回答的问题 |
|---|---|
| Algorithm | 数学目标和采样假设是什么 |
| Component | 需要哪些训练、推理、reward 和 environment 组件 |
| Control flow | 同步、流水化还是异步 |
| Data flow | sample、trajectory、logprob、advantage 和权重如何移动 |
| Framework | 哪些框架原生支持，哪些通过扩展支持 |
| Code locator | 当前固定 commit 中的真实入口与调用链 |
| Cost | 显存、通信、计算、延迟和稳定性代价 |
| Hardware mapping | CUDA 与昇腾实现的对应关系和缺口 |

## 4. 交付物结构

研究成果沉淀到现有 `llm-knowledge`，采用“总览 + 专题 + 学习路线”结构。

### 4.1 总览与算法专题

计划页面：

- `wiki/01_theory/04_posttraining/posttraining_frontier_map_analysis.md`
- `wiki/01_theory/04_posttraining/reasoning_rl_algorithm_evolution_analysis.md`
- `wiki/01_theory/04_posttraining/agentic_rl_algorithm_analysis.md`
- `wiki/01_theory/04_posttraining/on_policy_off_policy_staleness_analysis.md`

### 4.2 Infra 与框架专题

计划页面：

- `wiki/02_engineering/04_posttrain_frameworks/posttraining_infra_mechanism_analysis.md`
- `wiki/02_engineering/04_posttrain_frameworks/framework_comparison.md`
- `wiki/02_engineering/04_posttrain_frameworks/verl/verl_end_to_end_iteration_analysis.md`
- `wiki/02_engineering/04_posttrain_frameworks/slime/slime_architecture_analysis.md`
- `wiki/02_engineering/04_posttrain_frameworks/areal/areal_async_architecture_analysis.md`
- `wiki/02_engineering/04_posttrain_frameworks/roll/roll_strategy_and_ascend_analysis.md`
- `wiki/02_engineering/04_posttrain_frameworks/cuda_ascend_posttraining_stack_comparison.md`

### 4.3 学习路线

计划页面：

- `wiki/02_engineering/04_posttrain_frameworks/posttraining_source_reading_guide.md`

学习路线不按固定天数切分，而按可验证能力分级：

1. 能解释核心数学对象；
2. 能画出单次训练迭代；
3. 能追踪 verl 主链路；
4. 能比较同步与全异步实现；
5. 能分析权重同步和资源布局；
6. 能独立评估新框架与昇腾适配成本。

所有新页面必须加入领域 `index.md`，建立双向 `[[wiki links]]`，并更新 `wiki/changelog.md`。

## 5. 证据与版本标准

### 5.1 来源优先级

技术事实优先使用一手来源：

1. 论文正式版本或 arXiv 最新版本；
2. 官方源码仓库与固定 commit；
3. 官方文档、release note、设计文档和 issue/PR；
4. 作者公开演讲或技术博客；
5. 二手文章仅用于发现线索，不作为关键结论的唯一证据。

无法从公开一手来源验证的工业说法必须标为“公开信息不足”或“推断”，不能写成事实。

### 5.2 基线规则

- 论文记录 arXiv ID、版本号与发布日期；
- 框架记录仓库、分支、commit 和核验日期；
- 代码结论引用仓库相对路径与准确行号；
- 性能数字必须同时记录硬件、模型、并行配置、batch、序列长度和对照基线；
- 快速演进的框架在正式写作前重新核验远端状态；
- 源码与文档冲突时以固定 commit 的真实代码为准，并显式记录冲突。

### 5.3 事实与推断分离

每个非平凡结论区分：

- **来源事实**：论文、代码或官方文档明确给出的内容；
- **机制推导**：由公式、调用链或资源布局推导出的结果；
- **研究判断**：对工业意义、适用范围和未来趋势的判断。

## 6. 单篇分析模板

每个算法、Infra 或框架专题统一采用以下骨架：

1. **一条主线**：用一至两句话说明核心设计赌注；
2. **问题与失败模式**：原方案在哪里失效；
3. **机制**：数学、状态机、数据流或控制流；
4. **真实链路**：入口、关键跳转、数据结构和 owner；
5. **证据**：源码位置、实验表格、消融或版本记录；
6. **为什么不采用显而易见的替代方案**；
7. **代价与边界**：质量、吞吐、显存、通信、稳定性和复杂度；
8. **CUDA/昇腾映射**；
9. **最小源码阅读路径**；
10. **Related Pages**。

不以函数签名罗列、README 复述或论文摘要改写替代机制分析。

## 7. 分阶段研究顺序

### 阶段 0：建立当前快照

- 盘点现有知识库并标记已覆盖、陈旧、矛盾和空白内容；
- 核验 2026-07-27 时点的论文、框架和官方动态；
- 固定四个框架及关键依赖的源码基线；
- 形成前沿候选清单与证据清单。

### 阶段 1：建立统一坐标系

- 补齐必要基础；
- 形成 Reasoning RL 与 Agentic RL 算法演进图；
- 深挖 on-policy/off-policy、staleness、importance sampling 与 train-inference mismatch；
- 输出前沿地图初版。

### 阶段 2：贯通 verl

- 从真实配置和训练入口开始；
- 完整追踪单次训练迭代；
- 分析 worker/engine、DataProto、资源布局、rollout、reward、advantage、更新与权重同步；
- 标注算法扩展点、性能瓶颈和昇腾相关路径。

### 阶段 3：对照 slime 与 AReaL

- slime 聚焦 Megatron + SGLang、高性能生成、训练推理解耦和异步路径；
- AReaL 聚焦 fully-async、Agent runtime 服务化、policy lag 和 Agentic RL；
- 使用同一交叉矩阵与 verl 对照，避免按各自 README 的宣传口径比较。

### 阶段 4：ROLL 与昇腾专题

- ROLL 聚焦多后端 Strategy、资源映射和昇腾支持；
- 建立 CUDA—昇腾组件矩阵；
- 对比通信、推理、权重同步、算子和性能工具链差异；
- 区分“框架已声明支持”与“源码和示例已闭环验证”。

### 阶段 5：综合与持续追踪

- 完成框架对比和学习路线；
- 更新索引、交叉链接与 changelog；
- 对超过 30 天未核验的快速演进页面标记 staleness；
- 新论文或版本先进入雷达，满足证据标准后再进入主干专题。

## 8. 验证与质量门槛

每一阶段完成前必须验证：

- 所有框架结论均绑定固定 commit；
- 所有关键调用链至少包含入口、编排、执行和回流四类定位点；
- 所有算法公式和实验结论绑定论文版本与章节、公式或表格；
- 性能结论包含完整对照条件，不使用脱离 baseline 的倍数；
- CUDA 与昇腾对比区分已验证事实、官方声明和研究推断；
- 所有新增 `[[wiki links]]` 目标存在；
- 所有新增 Mermaid 图逐块检查并尽可能实际渲染；
- 无未完成占位符、空章节、冲突标记和尾随空白；
- `git diff --check` 通过；
- 暂存区只包含本研究产生的文件，不混入工作区现有修改。

## 9. 风险与处理原则

| 风险 | 处理原则 |
|---|---|
| 框架快速变化 | 写作前重新固定 commit；页面头部记录核验日期 |
| README 与代码不一致 | 代码为准，README 作为定位线索并记录冲突 |
| 工业数据不可公开验证 | 明确标为公开信息不足，不用传闻填补 |
| 算法名称多而实质接近 | 按 objective、采样、credit assignment 和系统假设归类 |
| 异步性能与模型质量混淆 | 分开讨论 sample efficiency、hardware efficiency 和 wall-clock |
| 昇腾支持停留在声明层 | 分为声明、代码路径、示例、CI/测试和实测五级证据 |
| 研究范围失控 | 下钻依赖时只追踪当前端到端链路的承重组件 |

## 10. 非目标

本轮默认不包含：

- 全量复现所有算法；
- 大规模多机训练和性能调优；
- 对所有后训练框架做穷举式比较；
- 通用 Megatron、FSDP、vLLM 或 SGLang 教程；
- 没有公开证据支持的闭源系统逆向猜测；
- 自动创建周期性任务或外部通知。

当源码事实存在争议、文档与实现冲突或 CUDA—昇腾行为无法仅靠静态分析确认时，再单独提出最小验证实验。

## 11. 提交策略

- 研究设计单独提交；
- 后续按“前沿地图、verl 主链路、异步框架对照、昇腾专题、综合学习路线”分批提交；
- 每批提交前只暂存本研究文件，保留用户已有的未提交修改；
- 不在未获得明确要求时推送远端。
