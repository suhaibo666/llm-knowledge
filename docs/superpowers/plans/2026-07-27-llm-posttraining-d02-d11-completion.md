# LLM 后训练 D02–D11 全量深挖 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在统一的 `wiki/03_posttraining/` 中完成 D02–D11，形成从前沿算法、在线 RL 数据语义、Infra 机制到 verl/slime/AReaL/ROLL 源码与 CUDA–Ascend 映射的可顺序学习体系。

**Architecture:** 先固定论文与四个源码仓库的可复现快照，再以同一组字段贯通算法和系统：优化单位、policy version、数据 owner、控制流、权重面、正确性不变量、成本和硬件映射。框架页面分别从真实入口追踪一条端到端链路，D06 汇总为统一矩阵，D00/D01/index 在全部证据落定后回写终版。

**Tech Stack:** Markdown、Obsidian Wiki Links、Mermaid、LaTeX、Git、PowerShell、GitHub 官方源码、arXiv 固定版本、PyPDF/pypdf 文本抽取

## Global Constraints

- 所有新增研究文档统一位于 `wiki/03_posttraining/`，D02–D11 文件名和编号不得改变。
- 研究快照日期固定为 `2026-07-27`；论文绑定 arXiv ID、版本和章节/公式/表格，源码绑定仓库、分支、commit 和 `file:line`。
- 四框架源码 baseline 固定为：verl `983cb0f24443f87b3d161fad318445130a620b07`、slime `aaf5c2092b01219fa0d5c2d323741d409086ca32`、AReaL `b23fa6cf9c8edfebcf055079ab78913128bc4579`、ROLL `370cb24c1036ea9145365478fcc40612b2186fc8`。
- 不使用脱离硬件、模型、并行、batch、序列和 freshness 条件的性能倍数。
- 区分来源事实、项目方声明、机制推导和研究判断；README 与源码冲突时以固定 commit 的可达代码为准。
- 每篇页面包含上一页/下一页、`## Related Pages` 和至少一个既有页面回链。
- 新 Mermaid 图遵守 `CLAUDE.md`：节点 id 使用英文数字，标签不含裸 `[] () {} |`，有文字的连线使用管道标签。
- 旧理论/工程页面只增加必要回链或过时标记，不迁移、不复制、不删除。
- 交付前无空章节、冲突标记、尾随空白或未完成占位词；全部当前内部链接存在。

---

### Task 1: 建立可核验的论文与源码证据面

**Files:**
- Create: `docs/research/2026-07-27-posttraining-source-ledger.md`
- External read-only checkouts: `E:/97-codes/torch_parallel/posttraining-research-sources/{verl,slime,AReaL,ROLL}`
- External paper cache: Windows temporary directory下的 `posttraining-papers/`

**Interfaces:**
- Consumes: S00 baseline、D01 候选清单、论文官方页面和四个官方仓库
- Produces: D02–D11 共用的版本表、论文定位符表、源码入口/owner/通信/测试 locator 表

- [x] **Step 1: 核验 2026-07-27 前沿来源**

仅搜索论文官方页、作者/项目官方仓库和官方文档。记录 title、arXiv version、发布日期、问题、机制、证据与是否有公开实现。核心集合至少覆盖 GRPO/DeepSeek-R1、DAPO、GSPO、Agentic credit、SAO、AReaL、StreamRL、AsyncFlow、RollPacker 和 TIM。

- [x] **Step 2: 获取四个固定源码快照**

对每个仓库执行只读 clone/fetch，并 checkout Global Constraints 中的 40 位 commit。运行：

```powershell
git -C <repo> rev-parse HEAD
git -C <repo> status --short
```

预期：HEAD 与固定 commit 完全一致，工作树无修改。

- [x] **Step 3: 建立源码地图**

每个仓库先记录两层目录，再定位：

```text
配置/命令入口 → controller/trainer → role/worker
→ rollout/agent → reward/advantage → policy update
→ weight publish/load → next rollout
```

每条链记录真实 `file:line`、关键 schema、owner、RPC/collective/queue、同步点和公开测试/示例。

- [x] **Step 4: 下载并提取承重论文**

将 PDF 提取为带 `===== PAGE N =====` 的 UTF-8 文本；逐篇核对章节、公式、图和表格，不从摘要推断实验条件。

- [x] **Step 5: 写 source ledger 并验证**

ledger 按 `Algorithms / Agentic / Async Infra / TIM / Framework Code` 分类，包含 baseline、locator、用于哪篇 D 文档和证据等级。运行：

```powershell
rg -n 'arXiv|commit|file:line|§|Table|Eq' docs/research/2026-07-27-posttraining-source-ledger.md
```

预期：每个 D02–D11 都有一手来源入口。

### Task 2: 完成 D02–D04 算法与在线数据语义

**Files:**
- Create: `wiki/03_posttraining/02_reasoning_rl_algorithm_evolution_analysis.md`
- Create: `wiki/03_posttraining/03_agentic_rl_algorithm_analysis.md`
- Create: `wiki/03_posttraining/04_on_policy_off_policy_staleness_analysis.md`

**Interfaces:**
- Consumes: Task 1 论文 ledger 与框架算法实现 locator
- Produces: D05 的数据/控制/权重面不变量，D06 的算法能力字段

- [x] **Step 1: 写 D02**

用统一公式比较 GRPO、DAPO、GSPO 及进入主干的后续方法；每个方法回答优化单位、ratio/clip 粒度、advantage、采样结构、证据、替代方案、系统要求和限制。加入“算法名变化 vs 真正改变的统计量”矩阵。

- [x] **Step 2: 写 D03**

定义 episode/trajectory/turn/action/token 五层 schema；分析 outcome/process reward、step/trajectory credit、group barrier、single-rollout、环境等待和 coding sandbox。以 SAO 等固定版本工作说明为什么 Agentic RL 会反向改变 optimizer。

- [x] **Step 3: 写 D04**

严格区分 system async、policy lag、off-policy、TIM；给出 behavior/old/current/reference policy 和 ratio 的关系，定义 version distance、wall-clock age 与 update count。比较 correction、丢弃/截断、staleness bound、同步长尾治理和 fully async。

- [x] **Step 4: 验证算法文档**

检查公式符号一致、每个结果数字带表格/条件、每篇都有上一页/下一页与 Related Pages，且不将异步直接等同 off-policy。

### Task 3: 完成 D05–D07 Infra、框架矩阵与 verl 主链

**Files:**
- Create: `wiki/03_posttraining/05_posttraining_infra_mechanism_analysis.md`
- Create: `wiki/03_posttraining/06_framework_comparison.md`
- Create: `wiki/03_posttraining/07_verl_end_to_end_iteration_analysis.md`

**Interfaces:**
- Consumes: D02–D04 的算法不变量、verl 固定源码 locator
- Produces: 四框架统一坐标、slime/AReaL/ROLL 对照字段

- [x] **Step 1: 写 D05**

以 control/data/weight 三平面解释同步、流水、部分异步、fully async；覆盖 placement 与 parallelism、continuous/dynamic batching、reward/environment、weight sync/reshard/version commit、backpressure、checkpoint、故障和可观测性。

- [x] **Step 2: 写 D07**

从 verl 当前真实训练入口追踪一轮迭代，至少覆盖配置、Ray trainer、资源池/worker、DataProto、rollout、reward、advantage、actor update 和 weight refresh；指出 stable 与 experimental fully async 的边界、训练/推理 log-prob 路径和扩展算法的最小修改面。

- [x] **Step 3: 写 D06 首版并由 D07 回填**

建立角色模型、训练后端、rollout、数据面、权重面、异步、Agentic、容错、TIM、Ascend 和证据等级矩阵。任何“支持”按接口/功能/正确性/性能四级标记。

- [x] **Step 4: 验证 verl locator**

抽查入口、编排、执行、回流、权重同步至少各两个 locator；文件存在且行号内容与正文一致。

### Task 4: 完成 D08–D10 三个对照框架源码深挖

**Files:**
- Create: `wiki/03_posttraining/08_slime_architecture_analysis.md`
- Create: `wiki/03_posttraining/09_areal_async_architecture_analysis.md`
- Create: `wiki/03_posttraining/10_roll_strategy_and_ascend_analysis.md`
- Modify: `wiki/03_posttraining/06_framework_comparison.md`

**Interfaces:**
- Consumes: D05 的统一机制、D06 字段、三仓固定源码 locator
- Produces: D11 的 CUDA–Ascend 组件事实和 D06 终版矩阵

- [x] **Step 1: 写 D08**

追踪 slime 的启动、Ray placement、Megatron actor、SGLang rollout、data buffer/TransferQueue、weight update 和 staleness；区分核心仓库能力与扩展/官方声明，并解释吞吐来自哪个 overlap 或解耦点。

- [x] **Step 2: 写 D09**

追踪 AReaL 2.0 training/inference/agent/weight-update 服务、Hermes loop、trajectory、staleness control 和 async trainer；明确论文代际与当前源码差异，定位 Ascend branch 仅能从公开证据确认的范围。

- [x] **Step 3: 写 D10**

追踪 ROLL 的 workflow/controller、role、Strategy、AutoDeviceMapping、rollout/reward/update/weight sync；分别陈述普通 RLVR 与 Agentic async 状态，定位 Ascend 配置、设备抽象、通信/推理依赖和公开示例/测试。

- [x] **Step 4: 更新 D06**

将 D08–D10 的源码事实回填矩阵，标注 README–code 冲突、未公开实现与需要运行实验才能确认的结论。

### Task 5: 完成 D11 并综合更新学习体系

**Files:**
- Create: `wiki/03_posttraining/11_cuda_ascend_posttraining_stack_comparison.md`
- Modify: `wiki/03_posttraining/00_posttraining_source_reading_guide.md`
- Modify: `wiki/03_posttraining/01_posttraining_frontier_map_analysis.md`
- Modify: `wiki/03_posttraining/index.md`
- Modify: `wiki/index.md`
- Modify: `wiki/changelog.md`
- Modify: `wiki/01_theory/04_posttraining/grpo_analysis.md`
- Modify: `wiki/01_theory/04_posttraining/dapo_analysis.md`
- Modify: `wiki/01_theory/04_posttraining/gspo_analysis.md`
- Modify: `wiki/02_engineering/04_posttrain_frameworks/verl/index.md`
- Modify: `wiki/02_engineering/04_posttrain_frameworks/rl_infra_efficiency_analysis.md`
- Modify: `wiki/02_engineering/04_posttrain_frameworks/rl_sandbox_design_analysis.md`

**Interfaces:**
- Consumes: D02–D10 全部结论
- Produces: 可从首页顺序阅读、可持续追踪的 D00–D11 完整知识域

- [x] **Step 1: 写 D11**

按 PyTorch/device、collective、parallelism、train engine、rollout engine、weight sync、kernel、dynamic shape、profiling、fault diagnosis、containers/versioning 建立 CUDA–Ascend 矩阵；每格分为可复用、需要适配、公开证据不足。

- [x] **Step 2: 更新 D00/D01/index**

将所有“计划”状态改为已完成；D00 增加每篇最小源码路径和最终能力验收；D01 更新前沿结论与 D02–D11 链接；index 记录完成状态、baseline 和 30 天 staleness 规则。

- [x] **Step 3: 更新全局导航、旧页面回链和 changelog**

页面数更新为 13；新增 D00–D11 快速导航；旧 GRPO/DAPO/GSPO/verl/infra/sandbox 页面增加统一领域回链或旧 baseline 标记；changelog 只记录本研究新增与修订。

### Task 6: 全量机械验证、源抽查与提交

**Files:**
- Verify all files from Tasks 1–5

**Interfaces:**
- Consumes: 完整 D00–D11
- Produces: 可提交、无悬挂链接和无证据占位的最终快照

- [x] **Step 1: 检查编号、导航和内部链接**

确认 `00_` 到 `11_` 连续，所有当前 Markdown/Wiki 链接目标存在，上一页/下一页闭环。

- [x] **Step 2: 检查来源与源码 locator**

每篇至少有 baseline；框架页随机抽查不少于五个 `file:line`；算法页随机抽查不少于三个 §/Eq/Table/Fig 定位符。

- [x] **Step 3: 检查 Mermaid**

定位本次所有 Mermaid 块，逐块按 `CLAUDE.md` 清单检查；若本地有 `mmdc` 则实渲，否则记录人工检查结果。

- [x] **Step 4: 检查格式与范围**

运行：

```powershell
rg -n '待补|待定|<<<<<<<|=======|>>>>>>>' wiki/03_posttraining docs/research
git diff --check
git status --short -uall
```

预期：占位/冲突扫描无输出；diff check 为 0；变更只包含本计划、ledger、D00–D11、必要索引/回链/changelog。

- [x] **Step 5: 提交**

仅暂存本计划产生的文件，提交信息：

```text
docs(posttraining): complete D02-D11 deep dives
```
