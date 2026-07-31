# 知识库整改 P5:后训练三域整合 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 解散 `wiki/03_posttraining/`(14 篇纵向学习域):D00-D12 按主题分发进功能树,GRPO 三写归一,verl 双基线整合,阅读路线降为 `courses/posttraining_frontier.md`;三个重构目录施行分段编号;全程 broken=0。

**Spec:** `docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md` §3.2(逐页处置表)、§5(分段编号)、§6(课程页规则)。

**编辑总规则(P3/P4 全部沉淀):** 只搬运不新造;判重必须找到承接页实际段落;被并方独有事实逐字落地+溯源;"全重复"判断必被审查验伪,清单型/过渡型/编译器视角型内容最易误杀;冲突留双+`[!contradiction]`;无源可疑转 `[!todo]` 不整删;删除决定推翻既往裁定必须 changelog 披露;量化措辞不升级认识论地位;行号按目标页页头基线核;裸基名链接默认;台账逐节;checker broken=0 每 commit。

---

## Task 1: 分支+基线
- [ ] main 最新(≥172b0de)、干净;`git checkout -b reorg/p5`;checker 记录(应 pages=375 broken=0);pytest 77

## Task 2: 纯迁移批(D01/D03/D04/D08-D12,一个 commit)
- [ ] `03_posttraining/01_posttraining_frontier_map_analysis.md`→`01_theory/04_posttraining/posttraining_frontier_map_analysis.md`;`03_...`→`agentic_rl_algorithm_analysis.md`、`04_...`→`on_policy_off_policy_staleness_analysis.md` 同目录;`08/09/10/11_*.md`→`02_engineering/04_posttrain_frameworks/`(slime/areal/roll/cuda_ascend_stack 去编号名);`12_kimi_k3_posttraining_case_study_analysis.md`→`01_theory/01_models/moonshot_kimi/`
- [ ] 入链改写(裸基名);两域 index 补条目;D04 §7 与 `01_theory/04_posttraining/tim_causal_chain_analysis.md` 补双向链(spec 点名);checker=0

## Task 3: D02 演进权威页 + GRPO 三写归一(一个 commit)
- [ ] `02_reasoning_rl_algorithm_evolution_analysis.md`→`01_theory/04_posttraining/reasoning_rl_algorithm_evolution_analysis.md`,定位为算法演进权威页;§3.6 K3-MOPD 收缩为一句+`[[kimi_k3_posttraining_case_study_analysis]]`
- [ ] 论文页瘦身三篇:`grpo_analysis`(165)/`dapo_analysis`(187)/`gspo_analysis`(133)——与 D02 §3.x 重叠的公式推导/动机叙述收缩为指向 D02 的一句话,保留论文元数据、原始实验数字、D02 没有的原文细节;逐页台账(重叠段必须在 D02 找到实际对应才可收缩)
- [ ] `02_engineering/04_posttrain_frameworks/verl/verl_rl_algorithms_analysis.md`(389)§3/§4/§7 数学部分同规程收缩指 D02,保留"注册表选型机制+config key→代码锚点"(spec 定位);台账
- [ ] checker=0

## Task 4: D05/D06 迁入+收缩;weight sync 划界(一个 commit)
- [ ] `05_posttraining_infra_mechanism_analysis.md`→`04_posttrain_frameworks/posttraining_infra_mechanism_analysis.md`;§7 sandbox 收缩为接口视角+`[[rl_sandbox_design_analysis]]`;§4 backpressure 收缩+`[[rl_infra_efficiency_analysis]]`(收缩前先验两页覆盖,独有逐字回流)
- [ ] `06_framework_comparison.md`→`04_posttrain_frameworks/rl_framework_comparison.md`(comparison 类型);§4.1 verl 段压缩为矩阵一行+链接
- [ ] `02_train_frameworks/megatron-lm/megatron_rl_posttraining_consistency_analysis.md`(207)与 `megatron_vllm_weight_sync_analysis.md`(182)与 D05 §6 三方划界补互指(不合并,spec 定位=补链)
- [ ] checker=0

## Task 5: D07 verl 端到端整合(一个 commit,基线敏感)
- [ ] `07_verl_end_to_end_iteration_analysis.md`→`04_posttrain_frameworks/verl/verl_end_to_end_iteration_analysis.md`,基线 `983cb0f` 为准
- [ ] 与 `verl_ray_trainer_analysis.md`(354,基线 `8a694930`)逐节台账:重叠(fit 主循环/角色资源池/一步 PPO 字段流转)以 D07 为准合并,ray_trainer 独有段(若有,逐字)并入 D07 后删除或留残页(按余量,台账定);**基线冲突处理**:两版描述矛盾时以 983cb0f 为准但保留 `[!contradiction]` 记录旧基线行为(若差异是版本演进)
- [ ] D07 §3 DataProto 收缩为契约表+`[[verl_dataproto_analysis]]`;§6 权重刷新收缩时序+`[[verl_rollout_resharding_analysis]]`(先验覆盖)
- [ ] verl/ 其余 8 篇页头加基线横幅:`> [!note] 本页基线 verl \`8a694930\`;端到端迭代以 [[verl_end_to_end_iteration_analysis]](\`983cb0f\`)为准,机制若有出入以新基线页为先` ;verl/index 同步
- [ ] checker=0

## Task 6: 位置错位页归位(一个 commit)
- [ ] `01_theory/04_posttraining/RL_PPO_Loss_and_GRPO_Analysis.md`(232,TorchTitan+vLLM 源码级)→`02_engineering/04_posttrain_frameworks/rl_ppo_loss_and_grpo_analysis.md`(snake_case 一并修);入链改写
- [ ] `02_engineering/04_posttrain_frameworks/batch_invariance_guide.md`(427)→`02_engineering/07_training_reliability/`;tools/batch_invariance_demo.py 的引用路径不变(脚本在 tools/);入链改写;两域 index 同步
- [ ] checker=0

## Task 7: 课程页 + 03_posttraining 删除 + index 重建(一个 commit)
- [ ] `wiki/courses/posttraining_frontier.md`:纯导读——从 `00_posttraining_source_reading_guide.md`(301)吸收阅读路线与六级能力门槛(导读级),从 `03_posttraining/index.md` 吸收 S00-S05 阶段叙述;按新位置的编号链接排 D01→D12 阅读序;能力门槛/路线之外的实质内容若 D00 独有,先逐字落最贴近页(候选:`posttraining_frontier_map_analysis`)再删
- [ ] `git rm` D00 与 03_posttraining/index.md,目录清空删除;入链全改(checker --json);wiki/index.md:删 03_posttraining 域行、courses 区补行、两域页数重算;`01_theory/04_posttraining/index.md` 与 `04_posttrain_frameworks/index.md` 全面重建(条目+一句话+新链接)
- [ ] checker=0

## Task 8: 三目录分段编号(一个 commit)
- [ ] 按 spec §5 段位约定对 `01_theory/04_posttraining/`(约 19 页)、`02_engineering/04_posttrain_frameworks/`(根,约 10 页)、`verl/`(10 页)施行:index 先定段位表(0 导览/1 主线按学习序/2 深潜专题/3 方法论实践),git mv,全库改链;`moonshot_kimi/`、`07_training_reliability/` 本次只补条目不编号(P7 全库推广时处理)
- [ ] checker=0;pytest 77

## Task 9: 阶段门(控制者亲自执行)
- [ ] pytest+checker 终值;`grep 03_posttraining` 残留核(changelog 豁免);changelog P5 条目;merge --no-ff 回 main+push;roadmap 标 ✅

## 风险注记
- Task 3/5 是内容敏感点(算法公式与双基线),审查按 P3/P4 强度;Task 2/6 纯迁移轻审
- 并行会话:开工前确认 main 未漂移
