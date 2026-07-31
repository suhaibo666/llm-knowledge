# Knowledge Base Changelog

All source ingestions and significant wiki updates are logged here.

> 本文件为只追加的历史日志：各条目按**写入当时**的状态记载，其中的文件路径、行数等均以当时为准，**不随后续目录迁移回写**。查当前路径请以各域 index 为准。

---

## 2026-07-31：知识库结构整改 P6 Task 5 —— NPU 三页划界 + 中重叠七组扫尾

**Type**: Structure Reorg / Dedup（设计：`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md` §3.5/§3.6；计划：`docs/superpowers/plans/2026-07-31-kb-reorg-p6-p7-finale.md` Task 5）

### 一、npu/ 三页 vs 上游新页划界

`02_compile_stack/04_inductor/npu/` 的 `20_npu_lowering_guide.md`(884)/`22_npu_fusion_passes_deepdive.md`(425)/`30_npu_vs_upstream_fusion_passes.md`(287) 逐节核对 vs 上游 `10_fx_lowering_to_inductor_ir_analysis.md`（原 C17，GraphLowering/call_function 决策树/lowering 注册 API）与 `03_graph_ir_and_passes/` pass 页（`21_fx_graph_editing_primitives_and_invariants_analysis.md` FX 改图原语与不变量、`12_graph_effects_alias_mutation_and_order_analysis.md` functionalization）：

- `20_npu_lowering_guide.md`(884→869) §1.1「什么是 Lowering」的通用 FX→IR 管线示意图（与 §1-§5 讲的是同一套 `GraphLowering` 机制）收缩为一句 + 链接指向 `10_fx_lowering_to_inductor_ir_analysis`；§1.2 起的 torch_npu monkey-patch 策略、§2-§9（架构对比表、fallback 分流、专有 IR 节点、配置体系、v2.7.1 源码复核）**全部核实为 NPU 特有内容，未删改**。
- `22_npu_fusion_passes_deepdive.md`(425→428) §7「改图操作原语与 pass 通用原理」核对：`replace_all_uses_with`/`erase_node`/insertion point 语义及事务化改图状态机是 PyTorch FX **通用**设计，已在 §7 顶部新增划界声明指向 `21_fx_graph_editing_primitives_and_invariants_analysis`（原语通用语义）与 `12_graph_effects_alias_mutation_and_order_analysis`（post-grad 图 functionalization 基础）；§7.1 原语表『代表用处』列（`ascend_graph_pass.py` file:line）、§7.3 `view_fold_pass` 逐 pass 走查（含 DAG 扇出 mermaid）、§7.4 三条贯穿原理判定为 **NPU 应用侧独有内容，全部保留**（该表/走查在上游页无对应物——是"用这些通用原语具体怎么改某个 NPU pass"而非重讲原语本身）。
- `30_npu_vs_upstream_fusion_passes.md`(287→288) 核实为**已是范本**：全篇以 file:line 逐层对照上游三阶段 pass（§3.1-3.3 已链 `30_pre_grad_passes_guide`/`31_joint_graph_passes_guide`/`32_post_grad_passes_guide`），无需收缩；§3.5 补一句 + 链接 `10_fx_lowering_to_inductor_ir_analysis`（lowering/decomposition 通用机制侧）。
- **双向链回补**：`10_fx_lowering_to_inductor_ir_analysis`、`21_fx_graph_editing_primitives_and_invariants_analysis` 此前均未反向链接 npu/ 三页，补 Related Pages 各一条。

三页合计 1596→1585 行；无独有内容删除，全部核实后原样保留。

### 二、中重叠 7 组现状核查（spec §3.6 残余）

逐组 grep 现状，只补缺失的双向链，已处理的记录出处：

| 组 | 现状 | 处置 |
|---|---|---|
| vLLM compilation ↔ `03_runtime_graphs` | 正向链（vLLM→`10_pytorch_cuda_graphs_complete_guide`/`11_torch_compile_npugraphs_deepdive`）P3 已补；反向链缺失 | 补 2 条反向链（CUDA 侧 + NPU 侧回指 vLLM 分段 CUDA Graph 应用实例） |
| vLLM IR/fusion ↔ pass 页 | 原声明仍在：`vllm_ir_and_fusion_passes_analysis` 已链 `22_pattern_expression_and_matcher_engine_analysis`/`24_graph_pass_pipeline_ordering_and_fixpoint_analysis`/`32_post_grad_passes_guide`；`24_...§14` 跨框架对照表已含 vLLM/sglang/npu 三个代表页双向链 | 已完成，无需改动 |
| sglang ↔ vllm | 健康范本核实：两页头部即互相声明"对照面"，`24_...§14` 表格双向收录 | 已完成，无需改动 |
| TIM 分层 | `26_tim_causal_chain_analysis` 头部四环因果链声明完整，与 `25_on_policy_off_policy_staleness_analysis` §7 边界区分明确 | 已完成于 P5 |
| operator_optimization ↔ kernel 页 | 六页归一（Roofline/执行模型双权威 + NPU 段划界） | 已完成于本 P6 Task 4（commit `dbaa37e`） |
| D05 ↔ sandbox/infra | `01_posttraining_infra_mechanism_analysis` §4/§7 与 `11_rl_sandbox_design_analysis`/`12_rl_infra_efficiency_analysis` 逐句对照、独有内容迁移、三方划界声明均已生效并核实链接有效 | 已完成于 P5 Task 4 |
| megatron_precision_cudagraph ↔ Guide | 核查发现**未补链**（P3 遗留待办未落地）：`23_megatron_precision_cudagraph_fusion_analysis.md` Related Pages 无任何指向 `10_pytorch_cuda_graphs_complete_guide` 的链接 | 补双向链（训练框架应用实例 ↔ CUDA Graph 通用机制权威页） |

**验收**：`python tools/check_links.py` pages=373、broken=0（ambiguous=70 不变，属既有裸 index 基线，P7 Task 8 范围）；`python -m pytest -q` 77 passed。

---

## 2026-07-31：知识库结构整改 P6 完成（高重叠清零）

**Type**: Structure Reorg（设计 §3.5/§3.6；P0-P7 的第七段）

- **Ring Attention/CP 四写归一**：新建理论权威页（674 行，骨架来源逐节标注、公式口径冲突并列披露含一处源内不一致 [!contradiction]）；四框架页收缩为实现差异（DSv4 特有零删减经审查回补后达成）。
- **横向页矩阵化**：通信掩盖页 271→124（纯对比矩阵+扩 MindSpeed 列）；分布式优化器横向页收缩；**Megatron 优化器三页 1118→984 合并为单页**；FSDP 四页补分工声明。
- **Roofline/执行模型归一**：六页 2046→1922，执行模型权威=cuda_execution_model_guide，Roofline 权威=operator_optimization_guide §2；昇腾页各留划界。
- **npu 三页对上游划界**；中重叠七组全部补齐双向链（含 P3 承诺未落地的一处）。
- 至此设计盘点的 **13 组高重叠全部清零**。全程 broken=0；wiki 374→373 页。

## 2026-07-31：知识库结构整改 P6 Task 4 —— Roofline / GPU 执行模型归一

**Type**: Structure Reorg / Dedup（设计：`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md` §3.5；计划：`docs/superpowers/plans/2026-07-31-kb-reorg-p6-p7-finale.md` Task 4）

`05_gpu_kernel/` 内 Roofline 讲了 ~4 遍、GPU 执行模型讲了 ~3 遍的六页归一，定两个权威：

- **执行模型权威 = `cuda_execution_model_guide.md`**（280→282 行）：核实其 Grid/Block/Warp/Thread/SM 讲解已完整覆盖 `gpu_kernel_guide.md` §01 与 `triton_00_gpu_essentials_guide.md` §2 直觉一的全部内容，无独有段需吸收；页头新增"本页地位"权威声明。
- **Roofline 权威 = `operator_optimization_guide.md` §2**（834→760 行，含新增 §2.5）：吸收 `triton_00_gpu_essentials_guide.md` §3 的独有内容——用 Triton 官方 `01/02/03-*.py` benchmark 公式逐项手算 AI 的完整推导（向量加法/融合 softmax/矩阵乘三例，含源码行号引用），逐字迁入新增 §2.5；§2 头新增权威声明。核实 `triton_06_optimization_profiling_guide.md` §1 与 `gpu_kernel_guide.md` 均无 Roofline 独有内容（前者的流程图与优化闭环已被 operator_optimization_guide §3/§8 覆盖，后者仅 §10 诊断清单一句提及）。
- **四页收缩为指针**：`gpu_kernel_guide.md`(303→299 行) §01 执行层级模型（含 Grid/Tiling 关系一节，该点已由 `cuda_gemm_kernel_analysis` §1 更深覆盖，改指该页）收缩为一句 + 双链接；§02 内存层级核实为无重复的唯一详解，未改动。`triton_00_gpu_essentials_guide.md`(119→90 行) §2 直觉三 + 原 §3 Demo 收缩为一段结论 + 链接（保留"一眼判别法"一句作为课程起点的独立可读锚点，§1/直觉一/直觉二/§3 分工表/§4 动手验证均未改，学习路线连贯性验证通过）。`triton_06_optimization_profiling_guide.md`(297→276 行) §1 的 roofline 循环流程图 + 长引述收缩为一句 + 链接，保留"优化杠杆速查表"（页内导航，非概念复述）。
- **§6 昇腾段 vs `ascend_kernel_execution_model_analysis.md` 判定**：后者独有内容（四类单元显式缓冲链完整图示、CUDA/Ascend 逐项对位表、GEMM 四层结构对照、片上缓冲预算账本、非 GEMM 算子按类分派表、FlashAttention Cube-Vector 融合、训练层三条主线）远超 50%，判定**各留 + 划界**（非合并）：两页互加"与 XX 的划界"声明——`operator_optimization_guide.md` §6.1/§6.2（834 行版本内约 149→95 行，含 §6 全节）的 AI Core 结构图/存储层次图/CopyIn-Compute-CopyOut 完整代码收缩为摘要 + 指向该深度页，保留独有的 Tiling 约束数值推导（L0A 容量→M 维上限 512）、DataCopy 32-byte 对齐要求、§6.3 GPU 经验迁移 checklist 与 AICPU/host CPU fallback 辨析（后两者深度页未覆盖，全部原样保留）；`ascend_kernel_execution_model_analysis.md`(213→215 行) 页头新增划界声明。
- **索引**：`05_gpu_kernel/index.md` 补 `operator_optimization_guide` 页面列表行（此前缺失，未入索引）；`cuda_execution_model_guide`/`gpu_kernel_guide`/`triton_00`/`triton_06`/`ascend_kernel_execution_model_analysis` 的 Related Pages 互补权威页链接。
- **净效果**：六页合计 2046→1922 行（−124，新增 sourced 内容的同时净收缩）；无内容丢失，独有段全部逐字保留或迁移。
- **验收**：`python tools/check_links.py` pages=373、broken=0（ambiguous=70 为既有裸 index 基线，不属本次范围）；`python -m pytest -q` 77 passed。

---

## 2026-07-31：知识库结构整改 P6 Task 3（二）—— Megatron 分布式优化器三页合并 + 横向页矩阵化

**Type**: Structure Reorg / Dedup（计划：`docs/superpowers/plans/2026-07-31-kb-reorg-p6-p7-finale.md` Task 3 第二部分）

- **三页合并为一**：`megatron-lm/megatron_ddp_optimizer_analysis.md`(446)、`megatron-lm/megatron_optimizer_internals_analysis.md`(227)逐字并入 `megatron-lm/16_megatron_distributed_optimizer_analysis.md`(原 445→合并后 984 行),文件名保留最通用者。以 `megatron_ddp_optimizer_analysis` 的 ZeRO 0-3 阶梯框架(§0-§5,含「阶段①-④」命名,供全库既有引用免改)为骨架;`megatron_optimizer_internals_analysis` 的优化器类层次/混合精度/step 五步/Loss Scaling/梯度裁剪/LR 调度整体并入为新 §6-§11;本页原稿(通信组定义/FP8-FP4/CPU offload/三种 FSDP 实现对比/Layer-Wise+Muon 集成)保留为 §2.8-§2.11、§12-§14,真正重复的段落(通信量 P 记号复述、类继承结构的简化版)删除、仅保留 2026-06-16 增量更新的独有更正。三页内部原有的跨页 `[[wiki link]]` 互指(如"详见 xxx §A.7 的更新")已改写为同页 §N 引用。
- **两个 torchtitan FSDP 四页补分工声明**(不合并,`02_train_frameworks/torchtitan/11_torchtitan_fsdp_analysis.md`/`25_torchtitan_simple_fsdp_analysis.md`/`20_torchtitan_fsdp_prefetch_overlap_memory_analysis.md`/`21_torchtitan_hsdp_backward_overlap_analysis.md` 各页头补一句"四页分工"声明,标杆篇/深挖伴篇/HSDP 展开篇/编译器路径替代方案互指)。
- **横向页收缩**:`32_distributed_optimizer_deepdive.md`(194→185 行)§一 的 ZeRO-1 因果链完整推导(与合并后的 megatron 页 §阶段② 重复)收缩为一段 + 指针,保留 §二(梯度累积 K 倍通信量表)、§三(FSDP2/Megatron/MindSpeed 三方对比矩阵,核心矩阵型内容)、§四(MindSpeed param 临时化,全库唯一出处,未改)、§五/§六(Adam vs Muon 内存与系统影响,通用理论、未在任何子页找到对应,未改)、§七(选型决策树)原样保留;补 §六.3 指向合并后 megatron 页 §14.1 的 Muon 具体解法指针;补齐全页此前缺失的 `## Related Pages` 区块(6 条)。
- **入链改写**:全库 12 个引用 `megatron_ddp_optimizer_analysis`/`megatron_optimizer_internals_analysis` 的文件(含 `megatron-lm/index.md` 的系列计数 18→16、`torchtitan/index.md`、`19_megatron_dist_checkpointing_analysis.md`、`17_megatron_parallelism_orchestration_analysis.md`、`15_megatron_pp_schedulers_analysis.md`、`12_megatron_tp_analysis.md`、`27_megatron_tp_fsdp_resharding_supplements_analysis.md`、`23_megatron_precision_cudagraph_fusion_analysis.md`、`28_megatron_training_stability_observability_analysis.md`、`11_torchtitan_fsdp_analysis.md`、本 changelog 历史条目的链接目标)全部改指 `16_megatron_distributed_optimizer_analysis`,原两处独立 Related Pages 行合并去重;`megatron-lm/index.md` 移除两行系列表格行(补一句"已并入"说明)。
- **验收**:`python tools/check_links.py` pages=373(375−2)、broken=0;`python -m pytest -q` 77 passed。

---

## 2026-07-31：知识库结构整改 P6 Task 3（一）—— 通信掩盖横向页收缩为跨框架矩阵

**Type**: Structure Reorg / Dedup（计划：`docs/superpowers/plans/2026-07-31-kb-reorg-p6-p7-finale.md` Task 3 第一部分）

- `02_train_frameworks/30_comm_compute_overlap_analysis.md`(271→124 行)由"基于 Megatron/torchtitan 两框架源码分析"的机制正文页,收缩为 **Megatron-LM/torchtitan/MindSpeed 三框架计算通信掩盖跨维度对比矩阵页**:与三个框架的通信掩盖权威机制页([[20_megatron_comm_overlap_analysis]] 740 行、[[24_torchtitan_comm_optimizations_overlap_analysis]] 170 行、[[11_mindspeed_comm_overlap_analysis]] 461 行)逐节核对,机制正文找到子页对应即收缩为一句 + 链接;子页缺失的机制段逐字下沉(注明来源):
  - `20_megatron_comm_overlap_analysis.md` 新增 §5.8(Shared Expert 独立 stream 状态机,`moe_shared_expert_overlap`)、§5.3 补 Layer→5 子节点拆解/`stream_acquire_context()`/镜像层配对三段、§5.6 补 DeepEP/HybridEP 硬件后端速查表(图片迁至 `megatron-lm/assets/`)。
  - `14_torchtitan_pp_analysis.md` §7.2/§7.3 补 I/W 拆分"非按模型结构"要点与 `OVERLAP_F_B` 源码 + "非真并发"澄清。
  - `24_torchtitan_comm_optimizations_overlap_analysis.md` §3.3 补 Graph Trainer FSDP AG/RS 重排 pass 一句。
  - 新增 §四"MindSpeed 掩盖机制概览"(此前该页完全未提及 MindSpeed)。
- 保留三子页均无的合成视图:概念分层图(§一)、combined_1f1b vs ZBV/DualPipeV 架构差异分析(新 §三,含 MindSpeed fb-overlap+DualPipeV 定位)、Sub-Layer 级掩盖可达性矩阵(§四,补 MindSpeed 列)、框架差异总结表(§五,补 MindSpeed 行)。
- `31_comm_compute_fusion_guide.md` 与本页头各补一句"融合"(单 kernel)vs"掩盖"(独立调度重叠)边界声明,互指。
- **验收**:`python tools/check_links.py` pages=375、broken=0;`python -m pytest -q` 77 passed。

---

## 2026-07-31：知识库结构整改 P6 Task 2 —— CP/Ring Attention 归一

**Type**: Structure Reorg / Dedup（设计：`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md` §3.5；计划：`docs/superpowers/plans/2026-07-31-kb-reorg-p6-p7-finale.md` Task 2）

- **新建** `01_theory/06_distributed_parallelism/20_ring_attention_and_context_parallel_analysis.md`（655 行）：抽取四份框架 CP 分析页（`13_megatron_cp_analysis` 391、`13_torchtitan_cp_analysis` 345、`20_mindspeed_context_parallel_analysis` 420、`35_deepseek_v4_context_parallel_analysis` 853，合计 2009 行）里重复讲的通用机制——CP 动机（attention `O(S²)` 墙）、与 TP/PP/DP/EP 的组合关系、序列切分（朴素连续切分 + 折叠/头尾配对负载均衡定量证明 + PTRR 任意稀疏掩码均衡 + RoPE 切分不变量）、因果 mask 三分支裁剪、四种通信调度（Ring P2P + online-softmax、All-gather 双缓冲、Ulysses 头维换轴、分层混合 N 级分组构造）、通信量代数统一对比、Dynamic CP 通用机制。逐段择"最深最完整版本"逐字为骨架并注明来源：MindSpeed 提供负载均衡定量证明 + RoPE 不变量 + 因果三分支裁剪 + online-softmax 公式 + Ulysses 全套机制 + Hybrid 量化论证；torchtitan 提供 Ring 主循环伪代码 + 通信掩盖时序 + 反向双环原理 + PTRR；DeepSeek-V4 提供 Native CP AllGather 代码 + 分层分组构造代码 + 通信量统一公式；Megatron-LM 提供 CP 动机 + 并行组合关系。其余版本的独有补充/口径差异（如 Ulysses 通信量"2 次"vs"4 次"计数粒度、通信量公式是否显式含 `/TP` 因子）逐段并注，不强行合并掩盖。
- **四页收缩**（通用段替换为一句定位 + 链接，页头补划界声明，各自实现差异/源码走读/性能数据/配置全保留）：
  - `13_megatron_cp_analysis.md` 391→131 行：保留 `cp_comm_type` 四选一配置接口、TE 透传架构、选型决策树、Dynamic CP 的 dispatcher 兼容/CUDA Graph 守卫等 Megatron 特有源码细节。
  - `13_torchtitan_cp_analysis.md` 345→181 行：保留 SDPA-ring 与 FlexAttention-allgather 双路径架构（torchtitan 独有）、DTensor dispatcher 接线、`functional collectives`/`AsyncCollectiveTensor` 异步实现（不手写 CUDA stream）。
  - `20_mindspeed_context_parallel_analysis.md` 420→345 行：保留五算法运行期分派脊柱、Ring **双环**（outer/inner window + 双 dKV 反向环）、**Adaptive CP**（调度驱动 + rank 重映射）、**KV-cache CP**（显存换反向通信）——后三项四框架中仅 MindSpeed 独有。
  - `35_deepseek_v4_context_parallel_analysis.md` 853→613 行：只裁掉与理论页字面重复的 Native CP AllGather 源码 walkthrough、Hierarchical CP 分组构造代码、标准 attention 通信量公式；MLA 降低 CP 通信量 ~128 倍的推导、CSA/HCA 压缩注意力与 CP 交互的论文↔代码 gap 审计（本页核心贡献）、RoPE 的 CP 感知、TE `cp_stream` 双缓冲机制、Dynamic CP 对 MLA 的不支持、CP 与 EP 带宽竞争等 DSv4 特有内容零删减。
- **索引与入链**：`06_distributed_parallelism/index.md` 补条目（页面列表 + 建议阅读顺序 + 显存通信总账 CP 行 + 关联域指向四份框架页）；`megatron-lm/index.md`、`torchtitan/index.md`、`mindspeed/index.md` 三处描述同步指向理论页；四页 Related Pages 与理论页 Related Pages 双向互链。
- **验收**：`python tools/check_links.py` pages=375（374+1）、broken=0、orphans=0；`ambiguous`/`bare_index` 69→70（新页 Related Pages 末尾沿用同目录 8 个既有页面的裸 `[[index]]` 惯例，与本次改动前基线持平 +1，属 P7 Task 8 全库裸 index 清零范围，非本次引入的新问题）。`python -m pytest tools/ -q` 77 passed。

## 2026-07-31：P6 Task 2 逐句复核修复（六项）

**Type**: Correction（对上一条目 P6 Task 2 归一结果的逐句审查，`git show 70d096e:...` 逐字取回原文核对）

复核发现归一/收缩过程中丢失或失真的六处，逐项修复（不改上一条历史记录，按追加惯例新增本条）：

1. **A1 回补**：旧 `35_deepseek_v4_context_parallel_analysis.md` §2.4.2 的 All-gather"关键缺陷"半句——KV buffer 显存代价公式（原 `2×S×B×Hₖ×D`）在收缩时全库丢失。补回理论页 §6.2「显存代价」新增段（All-gather 机制的通用属性，非 DSv4 专属）。
2. **A2 回补**：旧 DSv4 §2.4.4 的 a2a+p2p 三级递进数据流（Level 1 Pair A2A(2-GPU) → Level 2 Quad A2A(4-GPU，建立在 Level 1 之上) → Level 3 跨节点 P2P 环）及"NVLink 承担大部分通信量"的关键优势说明，收缩后两处均只剩两级描述。逐字补回理论页 §8.2 末尾新增「执行序」段。
3. **虚假指针修正**：`35_deepseek_v4_context_parallel_analysis.md:565` 与理论页 §11 声称 `resolve_cp_group`/`PackedSeqParams` 源码"已并入 `13_megatron_cp_analysis.md` §3"，但收缩时该源码级细节（`packed_seq_params.py:23-24`/`:69`、`transformer_engine.py:1798`、#5215 修复 `:1886`、`GPTModel`/`GatedDeltaNet`/MTP 消费者清单）被误删且全库无处收留。采用方案 (a)：逐字恢复进 `13_megatron_cp_analysis.md` §3.1「机制（源码）」，两处指针随之变为如实表述（391→131→135 行）。
4. **理论页 §9 补充口径差异第三条**：核实 DSv4 与 MindSpeed 的 Ulysses/a2a 通信量公式在扣除已知的 TP 因子、fwd/bwd 口径两项调整后仍残留 **cp 倍**未归因差异（Ring 行同样两项调整后可精确抵消至 2 倍，Ulysses/a2a 行不能）——新增 `[!contradiction]` callout 如实披露，不假装已解释，提示读者两页公式不可直接换算。
5. **符号消歧**：§9 开头补充说明本节 $h$ 指头数，与 §0 全局记号表的 $h$（隐藏维度）是两个不同的量（本节 $h\times d$ 才对应 §0 的 $h$）。
6. **引用来源订正**：理论页 §4.1 "从朴素 $cp\cdot$(全块)降到约一半"一句的引用来源从 `20_mindspeed_context_parallel_analysis.md §4.3` 订正为 `§4.3/§4.4`（三分支裁剪机制本体在 §4.3，但该精确量化措辞的原始出处是 §4.4 通信量代数节的 `[!tip]` 优化点 callout）。

**验收**：`python tools/check_links.py` pages=375、broken=0（不变）；`python -m pytest tools/ -q` 77 passed。

---

## 2026-07-31：知识库结构整改 P5 完成（后训练三域整合收官）

**Type**: Structure Reorg（设计：`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md` §3.2；P0-P7 的第六段）

- **`03_posttraining/` 纵向域解散**：D01-D12 按主题分发进功能树（理论算法→`01_theory/04_posttraining/`，框架与 infra→`02_engineering/04_posttrain_frameworks/`，K3 案例→`moonshot_kimi/`）；D00 阅读路线降为 `courses/posttraining_frontier.md` 纯导读页。
- **GRPO 三写归一**：D02 定位为算法演进权威页；grpo/dapo/gspo 论文页瘦身为"论文档案"（元数据+实验+特有公式）；verl 页保工程锚点——审查中回补 J_GRPO 的 KL 组合项等三处符号级细节。
- **verl 双基线整合**：D07（`983cb0f`）入主 verl/ 为端到端主链，ray_trainer 保留为 legacy 深潜页；**源码核实发现 `trainer.use_v1` 默认翻转**（旧教学主链在新基线非默认路径），三处 [!contradiction] 双记；其余 8 篇挂基线横幅。
- **错位页归位**：RL_PPO_Loss（实现分析）→框架域；batch_invariance_guide→训练可靠性域。
- **三目录分段编号**：01_theory/04_posttraining（17 页）、04_posttrain_frameworks 根（9 页）、verl/（10 页）按 0/1/2/3 段位编号，485 处链接改写。
- 全程 broken=0；wiki 375→374 页（净 −2 +1 课程页）。

## 2026-07-31：知识库结构整改 P5 Task 8（三目录分段编号）

**Type**: Naming Convention（设计：`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md` §5；计划：`docs/superpowers/plans/2026-07-31-kb-reorg-p5-posttraining.md` Task 8；P4 Task 9.5 同款规程）

对 P5 已重构定型的三个目录施行 §5 分段编号（文件名两位数字前缀 = 段位：段 0(01-09)入门导览、段 1(10-19)核心机制主线（按流水线/学习顺序）、段 2(20-29)深潜/专题、段 3(30-39)方法论/对照/工程实践；`index.md` 不编号）：`01_theory/04_posttraining/`、`02_engineering/04_posttrain_frameworks/`（根）、`02_engineering/04_posttrain_frameworks/verl/`。每目录先在 index.md 定段位表，再 `git mv`，全库改链，checker=0。判段按内容实质而非文件名后缀/既有体裁标签。`moonshot_kimi/`、`07_training_reliability/` 本次不编号（P7 全库推广时处理）。

### 段位表 1：`01_theory/04_posttraining/`（17 篇内容页）

| 段 | 编号区间 | 页面 |
|---|---|---|
| 0 | 01 | posttraining_frontier_map_analysis |
| 1 | 10–13 | instructgpt_rlhf_analysis、ppo_analysis、dpo_analysis、reasoning_rl_algorithm_evolution_analysis |
| 2 | 20–29 | grpo_analysis、dapo_analysis、gspo_analysis、rloo_analysis、agentic_rl_algorithm_analysis、on_policy_off_policy_staleness_analysis、tim_causal_chain_analysis、vapo_analysis、rlhf_foundations_analysis、kimi_k1.5_analysis |
| 3 | 30–31 | preference_optimization_analysis、reward_hacking_defense_analysis |

段 1 按学习序排列——InstructGPT 三步 RLHF 基础 → PPO-Clip → DPO → Reasoning RL 算法演进权威页（`13`，P5 Task 3 已定为 GRPO/DAPO/GSPO/SAO 谱系统一权威页）。段 2 恰好填满 10 格：GRPO/DAPO/GSPO/RLOO 四篇论文档案（Task 3 已瘦身为元数据/原始实验数字，公式与动机指向 `13`）+ agentic RL/staleness/TIM 因果链三个专题 + VAPO/RLHF 多论文综合/Kimi K1.5 三篇档案页。

### 段位表 2：`02_engineering/04_posttrain_frameworks/`（根，9 篇内容页；`verl/` 子目录见表 3）

| 段 | 编号区间 | 页面 |
|---|---|---|
| 0 | 01 | posttraining_infra_mechanism_analysis |
| 1 | 10–12 | rl_ppo_loss_and_grpo_analysis、rl_sandbox_design_analysis、rl_infra_efficiency_analysis |
| 2 | 20–22 | slime_architecture_analysis、areal_async_architecture_analysis、roll_strategy_and_ascend_analysis |
| 3 | 30–31 | rl_framework_comparison、cuda_ascend_posttraining_stack_comparison |

### 段位表 3：`verl/`（10 篇内容页）

| 段 | 编号区间 | 页面 |
|---|---|---|
| 0 | 01–02 | verl_architecture_overview_analysis、verl_quickstart_guide |
| 1 | 10–15 | verl_end_to_end_iteration_analysis、verl_single_controller_analysis、verl_dataproto_analysis、verl_workers_engine_analysis、verl_rollout_resharding_analysis、verl_rl_algorithms_analysis |
| 2 | 20 | verl_ray_trainer_analysis |
| 3 | 30 | verl_optimization_analysis |

段 1 按 index.md 既有「五条平面」表的管线顺序排列（入口/驱动当前基线主链 → 控制面 → 数据面 → 计算面 → 生成面 → 算法面）。

**判段说明**（内容实质优先于文件名后缀/既有体裁标签，同 P4 Task 9.5 规程）：

- `posttraining_infra_mechanism_analysis`（D05）：`_analysis` 后缀但内容是 control/data/weight 三平面模型这一域级总览，与 `posttraining_frontier_map_analysis` 在理论域的段 0 定位同构，判入段 0；与之相邻的 `rl_framework_comparison`（明确 comparison 类）分列段 3，两者不会同段竞争。
- `preference_optimization_analysis`：`_analysis` 后缀但内容是 DPO/IPO/SimPO/ORPO/KTO/MODPO 横向对比表，按内容判入段 3（对照类）而非段 1/2。
- `reward_hacking_defense_analysis`：内容是环境/penalty/inoculation prompting/agentic safety 四层防御体系，属方法论/工程实践，判入段 3。
- `rlhf_foundations_analysis`：`_analysis` 后缀但结构是 ReMax/Weak-to-Strong/RM Overoptimization/RigorLLM 等多篇独立论文的综合档案，与 grpo/dapo/gspo/rloo 四篇论文档案同类，判入段 2 而非段 1。
- `verl_ray_trainer_analysis`：verl/index.md「深挖实现」分组的一员，但该页自身层次列与页头横幅均明确自称 `8a694930` **legacy 深潜**（当前默认主链已切至 `verl_end_to_end_iteration_analysis` 的 `983cb0f`），故不与同组其余四篇（single_controller/dataproto/workers_engine/rollout_resharding）一起入段 1 主线，单独判入段 2。
- `verl_rl_algorithms_analysis`：虽在 verl/index.md 旧「算法与优化」分组与 `verl_optimization_analysis` 并列，但按「五条平面」表它是管线终点的算法面，判入段 1 主线；`verl_optimization_analysis` 是横切性能/显存方法论，判入段 3。

### 执行

`git mv` 36 个文件（17+9+10，均为「两位数字前缀 + 原基名」，不改文件名其余部分——含 `kimi_k1.5_analysis.md` 的点号未消除，naming §5 命名统一属 P7 范围非本任务）。全库改写裸基名 `[[wikilink]]` 链接 **485 处**（53 个文件，含 `courses/posttraining_frontier.md` 全部 D01–D12 阅读序链接、三目录自身 index.md 与页内互链、`wiki/index.md` 快速导航两行）。`wiki/changelog.md` 中 **6 处历史条目共 20 个 wikilink**（2026-05-24 交叉引用行、2026-06-22 verl 系列创建条目 9 处、2026-06-24 GLM-5 校验行、2026-07-25 TIM 首篇创建行、2026-07-27 D01 创建行，以及本 P5 序列内 Task 2 自身条目 1 处）按"历史不回写"惯例降级为反引号 + 去向说明；Task 3/4/5/7 自身条目中的相关链接此前已用反引号包裹（元描述文本），未被本次渲染判定为活链接，无需处理。

**顺带修订（Task 7 changelog 条目审查移交的两处）**：pages 口径句"计划文本写'375−2=373'"改为"按本次执行前的推算 375−2（未计新增课程页 +1）"，不改历史行结构，只修正误引；六级能力门槛压缩措辞补一句"压缩保留主干，个别子弹与前置/任务两列未 1:1 保留，实质由链接的深潜页承载"。

**验收**：`tools/check_links.py` broken=0、orphans=0（pages=374，纯改名不增减文件数）；`ambiguous`=69、`bare_index`=69（与 Task 7 基线持平，本次未引入）。`python -m pytest -q` 77 passed。

**自查**：三张段位表在编号前均先读取各页正文首屏/已有 index 描述定段（详见上方「判段说明」五条），非凭文件名猜测；`python tools/check_links.py --json` 全量 JSON 核对 broken 列表为空；链接重写脚本按 `check_links.py` 同款 fence/inline-code 豁免逻辑逐行处理，避免误改示例代码块内的字面 `[[...]]` 文本；`grep -rn` 全库确认无 markdown 式 `](...)` 链接指向本次改名的 36 个文件（本库仅用 Obsidian `[[wikilink]]`，无需额外处理）。

---

## 2026-07-31：知识库结构整改 P5 Task 7（课程页化，解散 03_posttraining 域）

**Type**: Course-page Consolidation + Directory Removal + Index Rebuild（设计：`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md` §3.2/§6；计划：`docs/superpowers/plans/2026-07-31-kb-reorg-p5-posttraining.md` Task 7）

**新建** `wiki/courses/posttraining_frontier.md`：纯导读页，结构参照 `courses/torch_compile_end_to_end.md`（角色横幅 → 这门课是什么 → 阅读路线 → 六级能力门槛 → 与功能树的关系 → Related Pages）。从 `03_posttraining/00_posttraining_source_reading_guide.md`（D00，301 行）吸收阅读路线骨架（三原则、最短闭环 mermaid 流程图、"只想先抓主干"短路线）与六级能力门槛（L1–L6，整体保留为表，每级压缩为核心问题 + 验收物两列；压缩保留主干，个别子弹与前置/任务两列未 1:1 保留，实质由链接的深潜页承载）；从 `03_posttraining/index.md` 吸收 S00–S05 阶段叙述，作为阅读路线的六个分段小节（S00 基线与地图…S05 综合验收）；阅读序按 D01→D12 的新位置排列（链接用现基名，Task 8 分段编号后再改一轮，本次不编号）；另从 `03_posttraining/index.md` §5「四个框架的研究分工」吸收一张精简表（verl/slime/AReaL/ROLL 研究角色对照），放在 S03/S04 阅读分组之后。

**D00 独有内容台账**（先核实再删，逐节判定）：

| D00 章节 | 处置 | 理由 |
|---|---|---|
| §0 怎样使用这条路线、§1 阅读顺序表 + 最短闭环 + 抓主干短路线 | 吸收入课程页 | 阅读路线骨架，课程页核心内容 |
| §2 六级能力门槛 L1–L6 | 吸收入课程页（压缩为表） | 门槛定义本身是导读性内容 |
| §3 论文阅读方法（五问法 + 两个笔记模板） | 不迁移，无单独落地 | 通用研究方法论，不含后训练领域独有事实；与本库 `.claude/skills/source-faithful-analysis` 方法论同类，非本域专属 |
| §4 源码阅读方法（owner 追踪六步 + 六问表 + 证据记录模板 Repository/Branch/Commit/…） | 不迁移，无单独落地 | 同上；证据记录模板已被 spec §6.3「代码分析页头钉基线」规则取代 |
| §5 工业实现四级"支持"口径（P1 接口/P2 功能/P3 正确性/P4 性能） | 不迁移，无单独落地 | **核实结论**：已独立复现于 `rl_framework_comparison.md`（D06）§3「四级支持证据」（几乎同一张表，本域权威定义）；`cuda_ascend_posttraining_stack_comparison.md`（D11）§13 进一步实例化为 CUDA→Ascend 专用的 M1–M4 迁移验收；两页均不因 D00 删除而失去定义来源 |
| §6 学习过程实践题（按 S00–S05 阶段） | 不迁移，无单独落地 | 绑定即将解散的 S00–S05 阶段标签，为个人学习进度工具，非可复用领域事实 |
| §7 版本与复习节奏（快速变化页面 30 天复核；K3 `0797decb` 复核触发条件） | 不迁移，无单独落地 | CLAUDE.md 已有通用 staleness 复核规则；K3 复核触发条件是操作性元数据非域事实 |
| §8 最小源码路径与终局验收（各框架最小入口→调用链） | 不迁移，无单独落地 | **核实结论**：verl 入口（`main_ppo.py`→`RayPPOTrainer.fit`）已在 `verl_end_to_end_iteration_analysis`/`verl_ray_trainer_analysis` 逐行覆盖；slime（`train.py`→DataSource/Megatron/SGLang）、AReaL（`PPOTrainer`→workflow/staleness）、ROLL（RLVR/Agentic pipeline→Strategy/device mapping）均已在各自深挖页以真实 `file:line` 覆盖，细致度远超 D00 原一句话摘要 |

`03_posttraining/index.md` 除 S00–S05 阶段叙述外，§1「为什么建立统一领域」并入课程页「这门课是什么」；§4 D00–D12 顺序表、§6 既有知识入口均与 D01（`posttraining_frontier_map_analysis`）自身文档顺序表/既有知识复用规则重复，不重复迁移；§7 维护规则绑定即将解散的目录本身，作废不迁移。

**删除**：`git rm wiki/03_posttraining/00_posttraining_source_reading_guide.md wiki/03_posttraining/index.md`，`wiki/03_posttraining/` 目录随之清空自动移除。

**入链改写**：D00/`03_posttraining/index` 的全部活链接改指 `[[courses/posttraining_frontier]]`，涉及 `24_kimi_k3_posttraining_case_study_analysis.md`（阅读导航"回到 D00"）、`posttraining_frontier_map_analysis.md`（阅读导航 + 文档顺序表第 1 行 + Related Pages，3 处）、`cuda_ascend_posttraining_stack_comparison.md`（Related Pages）、`dapo_analysis.md`/`grpo_analysis.md`/`gspo_analysis.md`/`rl_infra_efficiency_analysis.md`/`rl_sandbox_design_analysis.md`/`verl/index.md`（各 1 处 Related Pages）。`wiki/changelog.md` 中 2026-07-27 历史条目的 2 处活链接（`03_posttraining/index`、`00_posttraining_source_reading_guide`）按"历史不回写"惯例降级为反引号 + 去向说明。

**`wiki/index.md` 重建**：删除"03 后训练纵向学习域"整段；courses 表新增 `posttraining_frontier` 行；快速导航"LLM 后训练前沿 D00–D05"/"D06–D12"两行合并改写为"D01–D06"/"D07–D12"（各 7/6 项，去掉已删除的 `03_posttraining/index`、`00_posttraining_source_reading_guide`，D01–D06 行首改指课程页）；"PPO/GRPO RL 训练"行沿用 Task 6 已完成的 `rl_ppo_loss_and_grpo_analysis` 改名。**页数重算**（覆盖全部因 P5 迁移产生偏差的行，不止 Task 7 直接改动的两域）：模型 56→57、Kimi 13→14（D12 迁入，Task 2）、后训练对齐 15→18（+D01/D02/D03/D04，−RL_PPO，Task 2/3/6）、后训练框架 14→21（+D05/D06/D07/D08/D09/D10/D11/RL_PPO，−batch_invariance，Task 2/4/5/6；verl 子计数 10→11）、训练可靠性 4→5（+batch_invariance，Task 6）；两处"统计于"日期改 2026-07-31。

**`01_theory/04_posttraining/index.md` 与 `04_posttrain_frameworks/index.md`**：两域此前在 Task 2/3/4/6 已逐批为迁入页补全条目 + 一句话定位（"后训练前沿整合"/"后训练框架源码对照"/"RL 算法源码实现"三张表），本次核对确认条目完整、无缺漏；标题/摘要行改为反映扩容后的真实覆盖范围（前者从"LLM 对齐与偏好优化"扩为"LLM 后训练算法理论"）；两处新增课程页入口段落（仿 `01_ai_frameworks/index.md` 既有惯例："阅读路线入口…不计入下表"）+ 关联域各补一行指向 `courses/posttraining_frontier`；日期改 2026-07-31。

**验收**：`tools/check_links.py` broken=0、orphans=0，`ambiguous`/`bare_index`=69（与基线持平，非本次引入）；pages=374。**pages 对账与计划口径的差异**：按本次执行前的推算 375−2（未计新增课程页 +1），但 Task 7 同时创建 1 篇新文件（课程页）与删除 2 篇（D00 + `03_posttraining/index`），净变化为 375+1−2=374，本报告按实际文件系统状态汇报 374，已在交付报告中显式披露此算术口径差异。`python -m pytest -q` 77 passed。

**自查**：`grep -rn "03_posttraining" wiki/` 除本 changelog 历史条目（已降级反引号）与新课程页/两域 index 的解释性反引号文本外 0 处活 `[[...]]` 链接命中；D00 §5/§8 的"不迁移"判定均先用 grep 核实目标内容已在 `rl_framework_comparison.md`/`cuda_ascend_posttraining_stack_comparison.md`/各框架深挖页独立存在，未凭印象判断。

---

## 2026-07-31：知识库结构整改 P5 Task 6（错位页归位）

**Type**: Misplaced-page Relocation（设计：`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md` §3.2；计划：`docs/superpowers/plans/2026-07-31-kb-reorg-p5-posttraining.md` Task 6）

两篇此前放错域的页面归位，均为纯搬运（正文零改动，仅补链接）：

- `git mv wiki/01_theory/04_posttraining/RL_PPO_Loss_and_GRPO_Analysis.md` → `wiki/02_engineering/04_posttrain_frameworks/rl_ppo_loss_and_grpo_analysis.md`（snake_case 化）。该页是 TorchTitan + vLLM 的 PPO Loss / GRPO 流程源码级实现分析，此前误放理论目录，实为框架工程分析。与 verl 域已有的 `verl_rl_algorithms_analysis.md`（同为源码级 PPO/GRPO loss 分析，框架为 verl core_algos 注册表）互补一句话双向链接。
- `git mv wiki/02_engineering/04_posttrain_frameworks/batch_invariance_guide.md` → `wiki/02_engineering/07_training_reliability/batch_invariance_guide.md`。该页讲批次不变性/确定性算子实现（源自 DeepSeek V4 报告 §3.3 + DeepGEMM 源码），主题属确定性/可靠性问题域而非后训练框架，与 `determinism_and_numerical_reliability_analysis.md` 问题 2（训推数值不一致 / batch 不变性）互为系统侧上游与算子层实现细化的关系，双向补链；`tools/batch_invariance_demo.py` 引用路径为仓库根相对路径文本，两个新旧目录深度相同（均为 `wiki/02_engineering/<domain>/`），无需改写。

**入链改写（裸基名）**：`RL_PPO_Loss_and_GRPO_Analysis` → `rl_ppo_loss_and_grpo_analysis`，涉及 `23_glm5_posttraining_deepdive.md`（2 处）、`tim_causal_chain_analysis.md`、`rl_infra_efficiency_analysis.md`、`rl_sandbox_design_analysis.md`、`determinism_and_numerical_reliability_analysis.md`、`07_training_reliability/index.md`、`wiki/index.md`。`batch_invariance_guide` 基名不变（同名文件仅换目录），裸基名链接天然不受影响；唯一一处路径限定链接 `01_ai_frameworks/index.md` 的 `[[04_posttrain_frameworks/batch_invariance_guide]]` 改为裸基名 `[[batch_invariance_guide]]`（同域内唯一同名文件，不存在歧义）。`wiki/changelog.md` 中 1 处 2026-05-24 历史活链接（`RL_PPO_Loss_and_GRPO_Analysis`）按"历史不回写"惯例降级为反引号 + 去向说明。

**索引同步**：`01_theory/04_posttraining/index.md` 移除 RL_PPO 行；`02_engineering/04_posttrain_frameworks/index.md` 「数值与确定性」小节（原仅 batch_invariance_guide 一行）改为「RL 算法源码实现」小节收纳新迁入的 `rl_ppo_loss_and_grpo_analysis`，并注明 batch_invariance_guide 去向；`07_training_reliability/index.md` 问题地图第 2 行「详见」列补 `[[batch_invariance_guide]]`，新增「第四篇：batch 不变性算子实现」小节介绍其归位背景与来源（独立于本域原 wanka 综述素材）。

**验收**：`tools/check_links.py` broken=0、orphans=0（pages=375，与基线一致，纯搬运不增删文件）；`python -m pytest -q` 77 passed。

---

## 2026-07-31：知识库结构整改 P5 Task 5（verl 端到端整合，双基线调和）

**Type**: Content Consolidation + Baseline Reconciliation（设计：`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md` §3.2；计划：`docs/superpowers/plans/2026-07-31-kb-reorg-p5-posttraining.md` Task 5）

`git mv wiki/03_posttraining/07_verl_end_to_end_iteration_analysis.md` → `wiki/02_engineering/04_posttrain_frameworks/verl/verl_end_to_end_iteration_analysis.md`（D07，基线 verl `983cb0f`，225 行）。verl 域此前已有 `verl_ray_trainer_analysis.md`（354 行，基线 `8a694930`）逐方法源码走读 `RayPPOTrainer.fit`；两页覆盖同一主题但基线不同、深度不同，本次逐节台账后**不合并、保留双页**并调和。

**与 `verl_ray_trainer_analysis.md` 逐节台账**：重叠区（fit 主循环、一步 PPO 字段流转、advantage/loss 数学）以 D07 为准，不回填 ray_trainer 的逐行细节（D07 §4/§5 的高层表已经是权威账本）。ray_trainer §2（Role 枚举/`ResourcePoolManager`/`init_workers` 的 `create_colocated_worker_cls`/`WorkerDict` colocate 机制）、§3–§6（fit() 逐行追踪、`_balance_batch`、时序图、dispatch 表）在 `983cb0f` 本地 checkout（`E:\97-codes\torch_parallel\verl`）核对后确认为**独有、且体量远超 40%**（全篇 354 行里仅"fit 主循环顺序/advantage 数学"这类主题级重叠，method 级源码走读几乎全部独有）——判定**保留残页**：`verl_ray_trainer_analysis.md` 不删除，作为 `8a694930` legacy 深潜companion 页保留;仅将其中直接回答 D07 §2 自身"colocate or disaggregate"清单的**核心机制结论**（Role 枚举、`need_critic`/`need_reference_policy` 判定、`create_colocated_worker_cls` 合并 WorkerDict 机制、`ref_in_actor`）逐字并入 D07 §2，全部按 `983cb0f` 重新核对行号（utils.py:27-107、single_controller/ray/base.py:185-233、ray_trainer.py:343-360,772-907，与 `8a694930` 相应位置逐一比对，多数行号完全一致，`fit()` 因 `983cb0f` 新增代码整体下移 21 行）。

**基线冲突（`[!contradiction]` 双记）**：核对 `main_ppo.py` 发现两基线间存在真实机制反转——`trainer.use_v1` 默认值从 `8a694930` 的 `false`（legacy `RayPPOTrainer.fit` 为默认执行路径）反转为 `983cb0f` 的 `true`（`config/ppo_trainer.yaml:201`→`:219`；`main_ppo.py:184-193` 默认改道 `TaskRunnerV1`/TransferQueue）。这不是页面撰写错误而是版本演进，按规程在 D07 §1、`verl_ray_trainer_analysis.md` §1 版本定位note、`verl/index.md`「HEAD 架构演进提示」三处以 `[!contradiction]` 双记：`RayPPOTrainer.fit`（本系列 9+1 篇文档共同的教学主链）在 `983cb0f` 已非默认路径，需显式 `trainer.use_v1=false` 才会执行；`TaskRunnerV1`/TransferQueue 路径本知识库尚无覆盖。`@deprecated` 装饰器本身两基线间未变（均在 `ray_trainer.py:285`）。

**D07 §3/§6 收缩（先验两专页覆盖，Task 4 同款流程）**：`verl_dataproto_analysis.md`（325 行）与 `verl_rollout_resharding_analysis.md`（347 行）均已全面覆盖 D07 §3/§6 的全部事实（前者到方法级，后者到 CUDA IPC bucket/CheckpointEngine 两条路径的机制级），未发现 D07 独有细节。§3（DataProto）57→70 行原文压缩为「容器契约表 + 四条不变量」+ `[[verl_dataproto_analysis]]` 链接；§6（权重刷新）保留原有 `983cb0f` 专属行号（`engine_workers.py:705-725,783-787`、`vllm_rollout.py:271-320,278`）改写为时序代码块，补 `[[verl_rollout_resharding_analysis]]` 链接。

**verl 域其余 8 篇（`verl_architecture_overview_analysis`/`verl_quickstart_guide`/`verl_single_controller_analysis`/`verl_dataproto_analysis`/`verl_workers_engine_analysis`/`verl_rollout_resharding_analysis`/`verl_rl_algorithms_analysis`/`verl_optimization_analysis`）页头加基线横幅**：`> [!note] 本页基线 verl \`8a694930\`；端到端迭代以 [[verl_end_to_end_iteration_analysis]]（基线 \`983cb0f\`）为准，两基线间机制差异以新基线页为先。`；`verl_ray_trainer_analysis.md`（D07 对手方，保留残页）同样加此横幅，另加上述 `[!contradiction]` 版本反转记录。

**verl/index.md 重建**：新增「端到端主链（当前基线）」表段承接 D07；「由浅入深三层」「深挖实现」「算法与优化」三表标注基线 `8a694930`；「五条平面」入口/驱动行、「经典 RL 数据流」承接句、Related Pages 均补 D07 双链接；「HEAD 架构演进提示」callout 补 `use_v1` 反转记录；页数 9→10；原对 D07 的 `[[03_posttraining/07_verl_end_to_end_iteration_analysis]]` 引用改裸基名 `[[verl_end_to_end_iteration_analysis]]`。`04_posttrain_frameworks/index.md` 子目录表 verl 行页数同步改「9 篇 8a694930 深潜 + 端到端主链页基线 983cb0f」。

**入链改写（裸基名）**：`03_posttraining/07_verl_end_to_end_iteration_analysis` → `verl_end_to_end_iteration_analysis`，涉及 `posttraining_frontier_map_analysis.md`、`cuda_ascend_posttraining_stack_comparison.md`、`posttraining_infra_mechanism_analysis.md`、`rl_framework_comparison.md`（阅读导航+正文两处）、`slime_architecture_analysis.md`（阅读导航+正文两处）、`00_posttraining_source_reading_guide.md`、`03_posttraining/index.md`、`wiki/index.md`。D07 自身阅读导航（`[[rl_framework_comparison|上一篇 D06]]` · `[[slime_architecture_analysis|下一篇 D08]]`）此前已是裸基名，无需改写。

**验收**：`tools/check_links.py` broken=0、orphans=0（pages=375，与基线一致，本次只搬运不新增/删除文件）；`python -m pytest -q` 77 passed。

**自查**：verl 本地 git checkout（`E:\97-codes\torch_parallel\verl`）同时含 `983cb0f24443f87b3d161fad318445130a620b0` 与 `8a694930` 两个 commit，本次全部行号引用与「机制未变/已反转」判断均用 `git show <sha>:<path>` 现场核对，未依赖两页文本互证兜底。ray_trainer 页保留判定基于逐节主题映射后的独有内容占比估算（远超 40% 阈值），非精确计数；D07 §2 新增内容与 D07 §3/§6 收缩前均逐一核对对应专页/源码覆盖,未发现事实丢失。`grep -rn '03_posttraining/07_verl_end_to_end_iteration_analysis' wiki/` 除 `changelog.md`（历史记录，规则豁免不回写）与 `verl/index.md` 的一处历史路径纯文本提及（非 wiki-link）外 0 处活链接命中。

---

## 2026-07-31：知识库结构整改 P5 Task 4（D05/D06 迁入收缩 + weight sync 三方划界）

**Type**: Content Consolidation + Boundary Linking（设计：`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md` §3.2；计划：`docs/superpowers/plans/2026-07-31-kb-reorg-p5-posttraining.md` Task 4）

`git mv wiki/03_posttraining/05_posttraining_infra_mechanism_analysis.md` → `wiki/02_engineering/04_posttrain_frameworks/posttraining_infra_mechanism_analysis.md`；`git mv wiki/03_posttraining/06_framework_comparison.md` → `wiki/02_engineering/04_posttrain_frameworks/rl_framework_comparison.md`（comparison 类型命名）。两文件标题保留 D05/D06 前缀（沿用 D01–D04 迁移惯例），页头阅读导航与内部互链改裸基名。

**D05 §7（Reward/Environment/Sandbox）逐句对照 `rl_sandbox_design_analysis.md`（190 行）**：两页实为不同来源（D05 引 Kimi K3 Technical Report `0797decb`；rl_sandbox 引 RollArt/ProRL/Anthropic），逐句核对后**未发现字面重复**——reward-as-service 契约字段与「需要防」六条威胁清单是 D05 独有的三平面接口定义，予以保留；K3 white-box harness 版本化段落与 Fork/Pause/Resume/Snapshot 语义段落判定为「生产 sandbox 实现细节」而非「三平面接口」，逐字回流至 `rl_sandbox_design_analysis.md` 新增 §2.1「K3 案例：harness 版本化与故障恢复语义」（同步补 `关键资料`/`入库日期`/`Related Pages`）。D05 §7 更名「...的接口视角」，末尾改为指向 §2.1 的一句链接。262 行 → 该节从 26 行压缩到约 17 行，无事实删除。

**D05 §4（数据面 backpressure）逐句对照 `rl_infra_efficiency_analysis.md`（301 行）**：同样未发现字面重复——buffer schema 与四层容量定义（并发/staleness/内存/状态）是 D05 独有的 data plane 接口定义，予以保留；AReaL `StalenessManager` 源码引用（`areal/infra/staleness_manager.py:80-112`）与 K3 cache-pressure-aware admission 信号组合段落判定为「准入控制机制实现」，逐字回流至 `rl_infra_efficiency_analysis.md` 新增「优化 6: Admission-aware Backpressure」（`## 2. 五个核心优化`→`## 2. 核心优化`，因新增第 6 项去掉"五个"；同步补 `关键资料`/`入库日期`/`Related Pages`）。D05 §4 末尾改为指向该节的一句链接。

**D06 §4.1（verl 控制面）压缩**：先验 `verl_architecture_overview_analysis.md`（229 行）覆盖情况——`RayPPOTrainer.fit` 主循环（其 §6）、`DataProto` 贯穿角色边界（其 §3）均逐字覆盖；"算法 registry 含 GRPO/GSPO/SAPO/CISPO/REINFORCE"中的 SAPO 一项在该页未直接出现，改核 `verl_rl_algorithms_analysis.md`（已含 SAPO 注册表条目）覆盖，判定为全覆盖。原「优点/代价」两段散文压缩为一个二行摘要表 + 双链接（`verl_architecture_overview_analysis` 架构总览 + `verl_rl_algorithms_analysis` registry 机制），4.2/4.3/4.4（slime/AReaL/ROLL）不动。反向在两篇 verl 页 Related Pages 补 D06 回链。

**weight sync 三方划界（只补链不合并，三页正文除本条声明外不动）**：`02_train_frameworks/megatron-lm/30_megatron_rl_posttraining_consistency_analysis.md`（207 行,Megatron 训练侧 refit/训推一致性）、`33_megatron_vllm_weight_sync_analysis.md`（182 行,verl 在 Megatron+vLLM 场景下的 Gather-Broadcast-Load 同步实现）与 D05 §6（weight publish 三平面协议）互相在页头/相应节插入同一句「三方分工」声明：D05=三平面机制视角（框架无关）；megatron 两页=训练侧/verl 具体实现；`verl_rollout_resharding_analysis`=verl 自身 resharding/3D-HybridEngine 实现（该页仅在声明句中提及，未改动）。三页 Related Pages 互补链接。

**入链改写**：`03_posttraining/05_posttraining_infra_mechanism_analysis`→`posttraining_infra_mechanism_analysis`、`03_posttraining/06_framework_comparison`→`rl_framework_comparison` 两个路径的全部 wiki-link 目标改为裸基名，涉及 `wiki/index.md`、`03_posttraining/index.md`、`00_posttraining_source_reading_guide.md`、`posttraining_frontier_map_analysis.md`、`on_policy_off_policy_staleness_analysis.md`、`agentic_rl_algorithm_analysis.md`、`dapo_analysis.md`、`24_kimi_k3_posttraining_case_study_analysis.md`、`07_verl_end_to_end_iteration_analysis.md`、`verl/index.md`、`slime_architecture_analysis.md`、`roll_strategy_and_ascend_analysis.md`、两文件自身的阅读导航与 Related Pages。`03_posttraining/index.md` D05/D06 两行链接目标随批量改写同步生效。`04_posttrain_frameworks/index.md` 新增「后训练框架源码对照」表两行（posttraining_infra_mechanism_analysis / rl_framework_comparison），迁入来源脚注 D08–D11 改为 D05–D11，`Coding RL Sandbox 与 Infra` 表两行补来源(Kimi K3/AReaL)与主题(K3 案例/admission-aware backpressure)。

**验收**：`tools/check_links.py` broken=0（pages=375，与基线一致）；`python -m pytest -q` 77 passed。

**自查**：D05 §4/§7、D06 §4.1 三处压缩均先做逐句/逐段对照，实测无字面重复，因此本次"收缩"不含事实删除——D05 两节的所有独有事实全部逐字迁移到承接页（rl_sandbox_design_analysis §2.1、rl_infra_efficiency_analysis「优化 6」）并补溯源，D06 §4.1 的收缩基于承接页已覆盖的验证结论（含 SAPO 单项二次核实）。weight sync 三方划界未改动任何一页的既有分析正文，只加边界声明与互链。`grep -rn '03_posttraining/05_posttraining_infra_mechanism_analysis\|03_posttraining/06_framework_comparison' wiki/` 0 处活链接命中。

---

## 2026-07-31：知识库结构整改 P5 Task 3（D02 演进权威页 + GRPO 三写归一）

**Type**: Content Consolidation（设计：`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md` §3.2；计划：`docs/superpowers/plans/2026-07-31-kb-reorg-p5-posttraining.md` Task 3）

`git mv wiki/03_posttraining/02_reasoning_rl_algorithm_evolution_analysis.md` → `wiki/01_theory/04_posttraining/reasoning_rl_algorithm_evolution_analysis.md`，定为 GRPO/DAPO/Dr.GRPO/GSPO/SAO 公式演进与工程语义的**统一权威页**（页头新增"定位"行）。§3.6 K3-MOPD 收缩为一句 + [[24_kimi_k3_posttraining_case_study_analysis|D12]] 链接：先核实 D12 §1/§2.2/§4 已逐字覆盖 §3.6 的 MOPD 公式（Eq. 15 与 §3.6 完全一致）、reasoning-effort 预算约束公式与"top-k 蒸馏无明确优势"结论，无独有事实需回流。205→187 行。

**GRPO/DAPO/GSPO 三篇论文页瘦身**（逐节台账，收缩前均在 D02 找到实际对应段；D02 未给出的公式变体一律保留）：

| 页面 | 收缩(D02 对应) | 保留(论文特有/D02 未给出的公式) | 行数 |
|---|---|---|---|
| `grpo_analysis` | Key Innovation 段落(D02 §3.1 $\hat A_i$ 公式)、GRPO Objective 的 clip-surrogate 结构(D02 §2)、Why GRPO Works Well 动机叙述(D02 §1/§3.1)、Practical Implementation 玩具伪代码(改指 verl `core_algos.py:268/1279` 真实代码锚点) | 论文元数据、KL 低方差无偏估计量(D02 未给)、DeepSeek-R1-Zero 训练配置表/涌现行为/性能、R1 全流程、GRPO vs DPO 对比表、Impact | 165→119 |
| `dapo_analysis` | Clip-Higher 与 Dynamic Sampling 两段动机叙述(公式与 D02 §2/§3.2 重复)、Relationship to Other Methods 表(与 D02 §4 重复) | 论文元数据、"30 分失败三症状"诊断段、Token-Level Loss 的 $J_{DAPO}$/$J_{GRPO}$ 显式求和公式(D02 无)、Overlong Reward Shaping 分段惩罚公式(D02 无)、DAPO Algorithm 伪代码、Training Configuration/Progressive Results 两张原始数字表、KL 移除动机、数据集细节、Key Insights | 187→157 |
| `gspo_analysis` | Problem 段 5 点缺陷列表压缩为一段(D02 §3.4)、Sequence-Level Ratio 公式与 GSPO Objective(D02 §3.4/§2 完全一致)、Key Difference 表与 Relationship to Other Methods 表(均与 D02 §4 重复) | 论文元数据、Gradient Comparison 的 $\nabla J$ 显式梯度公式(D02 无)、GSPO-token stop-gradient 变体公式(D02 无)、Clipping Range Difference 原始超参数表(3e-4/4e-4)、Empirical Results 原始实验数据、Why GSPO Matters | 133→88 |

**verl `verl_rl_algorithms_analysis.md` §3.2/§4.1/§4.2 数学部分收缩指 D02**（保留全部代码锚点、14 优势估计器/11 策略损失清单、注册表机制、config key 映射——spec 点名保护项）：GRPO 组内归一化公式（§3.2）、vanilla PPO clip 基础结构（§4.1，dual-clip 扩展因 D02 未给出而保留）、GSPO 序列比公式（§4.2，stop-grad 实现技巧因 D02 未给出而保留）分别收缩为指向 D02 §3.1/§2/§3.4 的一句话，`core_algos.py:xxx` 代码块与行号全部原样保留。§7"与 RL 文献的对应"由散文列表改写为"verl 选型→文献→D02 对应→论文页"表格（轻改，DAPO"在 verl 里不是新损失"的实现洞察保留为表内备注；表中特别标注 verl `sapo`(arXiv 2511.20347)与 D02 §3.5 的 SAO(arXiv 2607.07508)是不同算法，避免同名混淆）。389→382 行（净减 7 行：删除 3 处重复 LaTeX 展示块，新增 §7 对应表 + Related Pages 补链）。

**入链改写**：全库 12 处 `[[03_posttraining/02_reasoning_rl_algorithm_evolution_analysis...]]` 目标改为裸基名 `[[reasoning_rl_algorithm_evolution_analysis]]`，涉及 `wiki/index.md`、`03_posttraining/index.md`、`00_posttraining_source_reading_guide.md`、`agentic_rl_algorithm_analysis.md`(D03 阅读导航+Related Pages)、`posttraining_frontier_map_analysis.md`(D01 阅读导航+文档顺序表)、`on_policy_off_policy_staleness_analysis.md`(D04 Related Pages)、`24_kimi_k3_posttraining_case_study_analysis.md`(D12 Related Pages)。D02 自身页内指向 grpo/dapo/gspo 的 3 处路径限定链接改裸基名（同目录）。`01_theory/04_posttraining/dapo_analysis.md` 中指向仍在 `03_posttraining/` 的 D05 链接（Task 4 才迁移）保持路径限定不变。

**Index 更新**：`01_theory/04_posttraining/index.md` "GRPO 系列"表前新增定位说明（D02=权威页，四篇论文页=元数据/实验数字档案）；"后训练前沿整合"表插入 D02 行（D01→D02→D03→D04 顺序），迁入来源脚注补 D02。`03_posttraining/index.md` D02 行改指新路径（索引本身按计划保留到 Task 7 删除）。

**验收**：`tools/check_links.py` broken=0（pages=375，与 P4/Task 2 基线一致；ambiguous=69/bare_index=69 均为存量未变）；`python -m pytest -q` 77 passed。

**自查**：`grep -rn '\[\[03_posttraining/02_reasoning_rl_algorithm_evolution_analysis' wiki/` 0 处活链接命中（唯一残留的 3 处字符串是本条目自身的反引号内说明文字，非 `[[wikilink]]`，`check_links.py` 的行内代码豁免规则不计入 broken）；本次改动前该旧路径从未出现在此前的历史 changelog 条目中，无需"历史不回写"降级处理。

---

## 2026-07-31：知识库结构整改 P5 Task 2（后训练三域整合，纯迁移批 D01/D03/D04/D08–D12）

**Type**: Pure Migration（设计：`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md` §3.2；计划：`docs/superpowers/plans/2026-07-31-kb-reorg-p5-posttraining.md` Task 2）

`wiki/03_posttraining/` 解散工作的开局批次，把 8 篇纵向学习域文档 `git mv` 到功能树对应位置并去编号前缀：

| 旧路径 | 新路径 |
|---|---|
| `03_posttraining/01_posttraining_frontier_map_analysis.md` | `01_theory/04_posttraining/posttraining_frontier_map_analysis.md` |
| `03_posttraining/03_agentic_rl_algorithm_analysis.md` | `01_theory/04_posttraining/agentic_rl_algorithm_analysis.md` |
| `03_posttraining/04_on_policy_off_policy_staleness_analysis.md` | `01_theory/04_posttraining/on_policy_off_policy_staleness_analysis.md` |
| `03_posttraining/08_slime_architecture_analysis.md` | `02_engineering/04_posttrain_frameworks/slime_architecture_analysis.md` |
| `03_posttraining/09_areal_async_architecture_analysis.md` | `02_engineering/04_posttrain_frameworks/areal_async_architecture_analysis.md` |
| `03_posttraining/10_roll_strategy_and_ascend_analysis.md` | `02_engineering/04_posttrain_frameworks/roll_strategy_and_ascend_analysis.md` |
| `03_posttraining/11_cuda_ascend_posttraining_stack_comparison.md` | `02_engineering/04_posttrain_frameworks/cuda_ascend_posttraining_stack_comparison.md` |
| `03_posttraining/12_kimi_k3_posttraining_case_study_analysis.md` | `01_theory/01_models/moonshot_kimi/24_kimi_k3_posttraining_case_study_analysis.md` |

正文零改动（仅两处例外，见下）。**入链改写**：全库 109 处 `[[...]]` 目标从旧编号基名/`03_posttraining/NN_...` 路径限定形式改为新裸基名，涉及 26 个文件（含 `wiki/index.md`、`wiki/03_posttraining/` 内未迁移的 D00/D02/D05/D06/D07/index、moonshot_kimi 五篇 K3 页、`grpo_analysis`/`gspo_analysis`、`rl_infra_efficiency_analysis`/`rl_sandbox_design_analysis`/`verl/index`）；`03_posttraining/index.md` 与 `00_posttraining_source_reading_guide.md` 中指向这 8 篇的行同步改指新位置，但索引本身按计划保留到 Task 7 才删除。`wiki/changelog.md` 中 3 处写入当时的历史活链接（本文件之前记载 D01/D12 新增的条目）按"历史不回写"惯例降级为反引号 + 去向说明，不当作活链接维护。

**例外 1（spec 点名的良性分层，只补链不动正文）**：`on_policy_off_policy_staleness_analysis.md`（D04）§7 TIM 小节新增一句指向 `tim_causal_chain_analysis`（历史活链接，已于 2026-07-31 因 kb-reorg P5 Task 8 再编号为 [[26_tim_causal_chain_analysis]]，按"历史不回写"惯例降级为反引号）的速览提示；`tim_causal_chain_analysis.md` Related Pages 反向补一条指向 `on_policy_off_policy_staleness_analysis` 的链接，说明其覆盖"TIM 与 staleness/off-policy 关系"这一上层概念坐标。

**例外 2（三个承接目录 index 补条目）**：`01_theory/04_posttraining/index.md` 新增"后训练前沿整合"小节（3 条：posttraining_frontier_map/agentic_rl_algorithm/on_policy_off_policy_staleness）；`02_engineering/04_posttrain_frameworks/index.md` 新增"后训练框架源码对照"小节（4 条：slime/areal/roll/cuda_ascend_stack）；`01_theory/01_models/moonshot_kimi/index.md` 在既有 K3 报告行后新增一条专属条目指向 `24_kimi_k3_posttraining_case_study_analysis`。

**验收**：`tools/check_links.py` broken=0（pages=375，与 P4 收官基线一致）；`python -m pytest tools/ -q` 77 passed。

---

## 2026-07-30：知识库结构整改 P4 Task 10（课程页化 + 19 号目录解散 + 全 index 终校，P4 收官）

**Type**: Course-page Consolidation + Directory Removal + Index Audit（设计：
`docs/superpowers/plans/2026-07-30-kb-reorg-p4-ai-frameworks.md` Task 10；
`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md` §6 课程页规则）

**课程页化**：新建 `wiki/courses/torch_compile_end_to_end.md`（spec §6 定义的纯导读页——只含
阅读顺序、链接与一句话导读，不承载正文）：开篇(是什么/前置/三段式流水线速览) → 按目录顺序
(0 前置 eager_runtime → 1 Dynamo → 2 AOTAutograd → 3 Graph IR/Passes → 4 Inductor(内含五阶段
一览表) → 5 编译缓存 → 6 调试诊断 → 7 CUDA Graphs → 8 图导出/算子扩展 → 9 分布式原语 → 10
训练扩展收尾)、段位递进(0→1→2→3)排布的阅读路线表(链接全用编号后新基名，一句话导读复用各
目录自己 index.md 已有的表述) → 三条捷径 → labs 使用说明(六卷 demo 入口表 + 运行契约 + PASS/
BLOCKED/FAIL 语义) → 与功能树的关系说明(courses/ 是纯索引层，功能树为唯一权威)。

**删除清单**（导读/动机价值并入课程页后删除源文件，`git rm`）：

| 文件 | 行数 | 去向 |
|---|---|---|
| `19_torch_compile_end_to_end/00_torch_compile_end_to_end_index.md` | 221 | A-F 六卷学习路径故事线 → 课程页开篇 + 阅读路线表 |
| `19_torch_compile_end_to_end/00_pytorch_graph_series_index.md` | 302 | 卷 C Part I-IV 结构 → 课程页 §3 Graph IR / §4 Inductor 分节 |
| `19_torch_compile_end_to_end/01_graph_ir_motivation_and_taxonomy.md`（C01） | 372 | "为什么需要图 IR"动机段 → 课程页 §3 末尾导读级要点；逐项核对确认全篇技术细节（autograd Edge、FX Node opcode、AOT partition、Inductor lowering、Scheduler 工厂、torch.cond HOP、`Graph.lint`、复杂度分析、reverse-topological order）均已被 `03_graph_ir_and_passes`/`01_eager_runtime/05_autograd_engine` 对应深潜页独立引用同一批源码定位，无需逐字搬运 |
| `02_compile_stack/04_inductor/02_torch_compile_architecture.md`（overview 四写之一） | 151 | 五阶段全景表 → 课程页 §4 Inductor 小节；`04_inductor/index.md` 补一段简短 is-what/why 段落承接 overview 角色，不留空洞 |

`19_torch_compile_end_to_end/index.md`（孤儿页）随目录一并 `git rm -r` 删除，19 号目录清空。

**入链修复**：一次性脚本批量重写全库 127 处 `[[...]]` 目标（94 个文件），全部改指
`courses/torch_compile_end_to_end` 或 `02_compile_stack/04_inductor/index`；随后逐文件人工核查
消除批量替换产生的相邻重复条目（约 12 处，如同一 Related Pages 列表里两个旧索引坍缩成同一个
课程页链接）；`01_ai_frameworks/index.md` 删除"过渡期"19 号指向块，课程入口改指
`[[courses/torch_compile_end_to_end|torch.compile 端到端课程]]`；`wiki/index.md` 新增 courses
入口小节。changelog 历史条目中 4 处指向已删除页的活链接按"历史不回写"惯例降级为惰性反引号 +
去向说明（不计入上方 127 处）。

**pytest 侧修复**：`tools/labs_torch_compile/demo_manifest.json` 的 `c01` 条目 `page` 字段改指
`torch_compile_end_to_end.md`；`test_volume_demo_contract.py` 的 `_page_root()` 为 `volume=="C" and
page_id=="c01"` 特判路由到 `wiki/courses/`，末尾兜底分支从"回退到 19 号目录"改为
`raise AssertionError`（19 号目录已删除，理应不可达）；`CourseMarkdownContractTest._course_pages()`
的 `course_root` glob 目标同步改为 `wiki/courses`。labs README/`NATIVE_BACKEND_RUNBOOK.md` 对
`00_pytorch_graph_series_index` 的 2 处引用同步改指课程页。

**遗留小修**（上一任务审查移交的 5 处旧基名/层次标注遗留，均为反引号/prose 引用，未被
`check_links.py` 覆盖，故未在此前批量修复中被发现）：`04_inductor/index.md` 表格单元格 3 个旧
基名补前缀；`15_inductor_compile_fx_orchestration_analysis.md`"最后更新"注里 7 个旧基名补前缀/
改指现存后继页（`scheduler_analysis` 已随 Task 8 判重删除，改指其内容实际归宿
`13_scheduler_dependency_graph_fusion_and_ordering_analysis`）；`npu/30_npu_vs_upstream_fusion_passes.md`
的 `npu_compile_paths_overview.md` 引用补 `01_` 前缀；`npu/32_npu_debug_guide.md` 页头"层次"由
"quick start"改为"方法论/排查实践(段 3)"，与 Task 9.5 changelog 记载的编号判段结论对齐（该页头
此前遗漏同步）。

**全 index 终校**：`01_ai_frameworks/index.md` 五层表填平 Task 7/Task 4 完成后仍标"待填充"的
Graph IR/Passes 与调试诊断两格；`02_compile_stack/index.md`、`04_inductor/index.md`（Task 8 自查
点名重点通读，确认无遗留问题）及其余 26 个模块/层 index 逐一核对：脚本核实每个 index.md 均已
提及其目录下全部同级 `.md` 文件（0 处缺失）；`04_inductor/index.md`、`02_aot_autograd/index.md`、
`01_fx_export_extensibility/index.md` 三个当日实际编辑过内容的模块 index 补"最后更新"日期至
2026-07-30；`01_eager_runtime/index.md`/`02_compile_stack/index.md`/`04_export_and_distributed/index.md`
三个 Task 2 建立的架构层壳 index 确认按设计无日期字段，非遗漏。`wiki/index.md`：AI框架页数按
`find wiki/02_engineering/01_ai_frameworks -name "*.md" ! -name SUPERSEDED.md` 实数重算，
174→**150**（子域 TorchInductor 39→36、运行时图 10→12，均为 Task 3-9.5 期间累积漂移的一次性
对账，非本任务新增改动）；统计日期同步到 2026-07-30。

**自查披露**：`grep -rn "19_torch_compile_end_to_end" wiki/ --include="*.md" | grep -v changelog`
非零命中——8 处，均为 Task 3（A 卷迁移）遗留的历史出处脚注（形如"本节内容原属 P4 知识库整改
被删除的 A 卷回顾页(`19_torch_compile_end_to_end/aXX_....md`)"），在反引号内、非 `[[wikilink]]`、
明确注明"已删除"，起 changelog 同等的溯源作用；均早于本任务存在（`git show e5cc60a` 可验证），
本任务未新增此类残留，按惯例不追溯改写，留痕供控制者复核。

**P4 阶段小结**（Task 1-10 全阶段，本任务收官）：`01_ai_frameworks` 从 18 个平铺目录重组为 5
个架构层两级目录；`19_torch_compile_end_to_end`（63 篇/2.05 万行）经 A→E→B→D→C→F 六卷逐批
判重解散，独有内容验证后并入功能树对应页，课程导读价值最终归一到单一 `courses/` 纯导读页；
`checker` 全程 `broken=0`（Task 1 基线 pages=398 → 本任务终态 pages=375，净 -23 主要来自判重
删除的重复大文与本任务的 4 页课程化收口，非内容流失）；`pytest tools/` 全程 77 passed。

**校验**：`python tools/check_links.py`：pages=375，broken=0，ambiguous=69，bare_index=69，
orphans=0；`pytest tools/ -q`：77 passed。

## 2026-07-30：知识库结构整改 P4 Task 9.5（目录内分段编号，用户追加需求）

**Type**: Naming Convention（设计：`docs/superpowers/plans/2026-07-30-kb-reorg-p4-ai-frameworks.md` Task 9.5、
`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md` §5）

对 `01_ai_frameworks` 下 19 个已定型模块目录（含 2 个硬件子目录）逐目录施行两位数字段位前缀命名：
段 0（01-09）入门/导览、段 1（10-19）核心机制主线（按该子系统执行流水线/依赖顺序排列）、段 2
（20-29）深潜/专题、段 3（30-39）方法论/对照/工程实践；`index.md` 不编号，仅在其中新增"段位与
阅读顺序"小节定段位表。纯改名，不改内容；跨 8 个 commit 分批完成，每批 checker broken=0 +
pytest 77 后提交：

| 目录 | 篇数 | 段位分布 | 编号区间 |
|---|---|---|---|
| `01_eager_runtime/01_tensor_and_storage` | 2 | 0段1/1段1 | 01,10 |
| `01_eager_runtime/02_dispatcher_and_device` | 5 | 0段1/1段2/2段2 | 01,10-11,20-21 |
| `01_eager_runtime/03_op_registration` | 0（未编号） | — | 顶层 0 篇内容页；`npu/` 3 篇低于 ≥4 递归阈值，两者均不编号 |
| `01_eager_runtime/04_aten_op_execution` | 2 | 0段1/1段1 | 01,10 |
| `01_eager_runtime/05_autograd_engine` | 3 | 0段1/1段1/2段1 | 01,10,20 |
| `01_eager_runtime/06_nn_module_system` | 2 | 0段1/1段1 | 01,10 |
| `01_eager_runtime/07_memory_amp_profiler` | 2 | 0段1/1段1 | 01,10 |
| `02_compile_stack/01_dynamo` | 14 | 0段1/1段9/2段3/3段1 | 01,10-18,20-22,30 |
| `02_compile_stack/02_aot_autograd` | 6 | 0段1/1段4/2段1 | 01,10-13,20 |
| `02_compile_stack/03_graph_ir_and_passes` | 12 | 1段6/2段6 | 10-15,20-25 |
| `02_compile_stack/04_inductor` | 23 | 0段2/1段6/2段9/3段6 | 01-02,10-15,20-28,30-35 |
| `02_compile_stack/04_inductor/npu` | 12 | 0段1/1段3/2段5/3段3 | 01,10-12,20-24,30-32 |
| `02_compile_stack/05_codegen_backends/mlir` | 4 | 0段1/1段2/3段1 | 01,10-11,30 |
| `02_compile_stack/05_codegen_backends/mlir/npu` | 0（未编号） | — | 2 篇低于 ≥4 递归阈值，不编号 |
| `02_compile_stack/06_compile_cache` | 4 | 1段4 | 10-13 |
| `02_compile_stack/07_debugging` | 10 | 1段10 | 10-19（见下方说明） |
| `03_runtime_graphs/cuda` | 3 | 0段1/1段1/2段1 | 01,10,20 |
| `03_runtime_graphs/npu` | 6 | 0段1/1段2/2段2/3段1 | 01,10-11,20-21,30 |
| `04_export_and_distributed/01_fx_export_extensibility` | 3 | 0段1/1段1/2段1 | 01,10,20 |
| `04_export_and_distributed/02_distributed_primitives` | 4 | 0段1/1段1/2段2 | 01,10,20-21 |
| `05_other_frameworks` | 1 | 1段1 | 10 |

共 118 篇内容页改名，全库改写裸基名 `[[wikilink]]` 链接约 1570 处（含 labs `demo_manifest.json` 的 `page`
字段与 `test_volume_demo_contract.py` 硬编码文件名字面量同步，涉及原 B/C/D/E/F 卷 page_id）。

**判段说明**（内容实质优先于文件名后缀/既有体裁标签）：

- `01_eager_runtime/04_aten_op_execution/adding_an_aten_operator_guide.md`：`_guide` 后缀但内容
  是 native_functions.yaml 速查实操，按内容实质入段 0（非默认的段 2/3）。
- `02_compile_stack/04_inductor/npu/npu_compile.md`、`.../npu_debug_guide.md`：目录旧三层
  （overview/quick start/deep dive）均标两者为 quick start；按内容实质改判——`npu_compile`
  是编译工作流/Autotune/精度校验等机制性叙述入段 1，`npu_debug_guide` 是排查方法论入段 3。
- `03_runtime_graphs/npu/npugraphs_make_graphed_callables_deep_dive.md`：旧体裁标 quick start，
  内容是窄 API 的六阶段实现级深挖，按内容实质改判段 2。
- `02_compile_stack/07_debugging`：十篇不做 quickstart/深潜/方法论层级切分——本目录的"核心机制
  主线"就是排查工作流本身，全部落段 1（10-19）恰好填满整段。编号顺序采用 index.md 原有的
  "建议阅读顺序"，而非旧卷内编号 D07/E01-E09 顺序：`14_compiled_artifact_lifecycle`（原 D07）
  页头仍标其原 D 卷前置为 `cudagraph_trees_...`、且被 `10_observability` 页头引用为"前置"，但
  在本目录实际教学顺序中排第 5 位（失败分层定位之后、minifier/bisector 工具之前）——是本次编号
  中唯一一处"页头前置字段"与"实际阅读顺序"不一致的目录，过程中发现后已改正（初次按 D07 在前
  的旧卷顺序命名，随即按 index.md 建议阅读顺序重排，同批提交前完成，无需回滚）。

**验证口径**：任务书给出的验证命令
`find wiki/02_engineering/01_ai_frameworks -name "*.md" ! -name "index.md" ! -path "*19_torch*" |
grep -v -E "/[0-9]{2}_"` 因按**完整路径**匹配（父目录如 `01_eager_runtime/`、`02_compile_stack/`
本身即带两位数字前缀），对本次未编号的 5 个文件（3 篇 `03_op_registration/npu/` + 2 篇
`05_codegen_backends/mlir/npu/`）不会显式列出——命令按此口径运行确为空。若改按**文件基名**匹配
（`basename | grep -v -E "^[0-9]{2}_"`），会精确列出这 5 个文件；它们是符合 spec §5"npu/cuda 子
目录页多（≥4 内容页）时递归适用，页少不编"规则的预期例外，非遗漏。

**校验**：`python tools/check_links.py`：pages=379，broken=0（8 个 commit 均独立核验）；
`pytest tools/ -q`：77 passed（8 个 commit 均独立核验）。

---

## 2026-07-30：知识库结构整改 P4 Task 9（19 号 F 卷分发，6 篇 1504 行）

**Type**: Volume Migration + Boundary Reconciliation（设计：`docs/superpowers/plans/2026-07-30-kb-reorg-p4-ai-frameworks.md` Task 9）

f01-f07（除已在 Task 6 随 D06 迁走的 f08）从 `19_torch_compile_end_to_end/` 分发进功能树，去
`fNN_` 前缀，六个子任务各一 commit：

| 篇 | 去向 | 处置 |
|---|---|---|
| f01（316 行）Compiled Autograd | `01_eager_runtime/05_autograd_engine/compiled_autograd_analysis.md` | 与 `autograd_engine_analysis.md`（481 行）互补划界：后者讲 eager 模式 `Engine::execute` 直接执行反向 DAG,本页讲同一引擎驱动 Python tracer 把运行时反向录制成 FX 图再编译——同一引擎的两种运行模式,无字面重复,两页页头各补显式 `[!note]` 分工声明 + 互指,eager 页新增 §12 |
| f02（226 行）Activation Checkpoint | `02_compile_stack/02_aot_autograd/activation_checkpoint_recompute_and_compile_analysis.md` | 与 C10（`saved_tensors_recompute_and_runtime_abi_analysis.md`,433 行）互指划界:本页站用户 API/策略层（`torch.utils.checkpoint`、Selective AC policy）,C10 站 partitioner 源码/runtime ABI 层,补齐 spec 附录 A 缺口 |
| f03（230 行）DDP Compile Boundaries + f04（221 行）FSDP/DTensor | `04_export_and_distributed/02_distributed_primitives/`（`ddp_compile_boundaries_and_optimizer_analysis.md`/`fsdp_dtensor_and_distributed_graphs_analysis.md`） | 与 `c10d_ddp_fsdp_dtensor_analysis.md`（433 行）三方互指划界:该页讲 DDP/FSDP/DTensor 原语本身（Reducer 分桶、FlatParameter shard/unshard、placement 传播）,零涉及 `torch.compile`;两篇讲原语与编译器相遇时新增的边界（DDPOptimizer bucket 切图、`use_orig_params=True`、Dynamo skip frame）;三页头各补 `[!note]`,索引新增"原语本身 vs compile 边界"小节 |
| f05（255 行）Custom Operator/Fake Kernel | `04_export_and_distributed/01_fx_export_extensibility/custom_operators_fake_kernels_and_decompositions_analysis.md` | 逐节判重 vs `fx_graph_export_and_custom_ops_analysis.md`（315 行）§7:该页 §7 是 FX/export/functorch 全景 deep dive 中约 40 行的 custom_op 注册概述（公开重导出入口）,本页是 13 节的"编译器边界契约"深度分析（fake kernel 正确性、mutation/version、decomposition/lowering/fallback 选择、失败定位分层）,独有内容占比远超 50%,判定**保留独立页**,不并入;仅在该页 §7 末补一条深度指路链接 |
| f06（260 行）Custom Backend/Device Integration | `01_eager_runtime/02_dispatcher_and_device/custom_backends_and_device_integration_analysis.md` | 三方划界 vs `privateuse1_device_integration_analysis.md`（278 行,eager/dispatcher 层设备接入,零涉及 compile,边界声明即可）+ `codegen_extension_guide.md`（247 行,Inductor codegen backend 注册实操指南,与本页 §5/§6/§8 有真实重叠）——**收缩重叠段**:§5/§6/§8 的 `register_backend_for_device`/`DeviceOpOverrides`/`BackendFeature` 具体注册细节改为指向该指南,保留本页独有的三层 backend 划分框架与 `DeviceInterface` |
| f07（292 行）AOTInductor 打包部署 | `02_compile_stack/04_inductor/aotinductor_packaging_and_deployment_analysis.md` | 纯平移,无判重需要 |

**B02 `use_aoti` todo 解答**（`backend_modes_options_stances_and_fullgraph_analysis.md` §14.2,
Task 5 遗留,原 `[!todo]`；**同日据 spec 复核修正**,见下方"复核修正"段）：读 F07 源码后对照
本地 pinned pytorch checkout 核实——`_TorchCompileAOTInductorWrapper.__call__` 最终调用与普通
JIT 编译相同的 `compile_fx(...)`（`torch/__init__.py:3016-3054` → 父类 `__call__`）；
`aoti_compile_and_package` 底层的 `compile_fx_aot`（`torch/_inductor/compile_fx.py:2221`）内部
同样调用 `compile_fx(...)`（`:2282-2284`）。两条路径在 `compile_fx` 内部汇合于同一段代码：
`V.aot_compilation` 为真时直接返回 `CompiledAOTI(filename=compiled_fn, device_type=graph.device_type)`
（`compile_fx.py:1849-1859`），是 `compile_fx_aot` 断言并解包 `.filename` 的同一个
`CompiledAOTI` 类型（`:2285-2289`）——两条路径汇合于同一套 AOT 代码生成/artifact 机制,但
`CompiledAOTI` 是否被真正装载成可调用 runner **不对称**：`__post_init__` 只有在
`config.enable_autograd_for_aot`（默认 `False`,`config.py:1696`）为真时才构造
`AOTIModelContainerRunner{Cuda,Xpu,Cpu}`（F07 §8 讲的同一个 `model_container_runner.cpp`
runner）并装进 `current_callable`；只有 use_aoti 路径会用 `config.patch("enable_autograd_for_aot", True)`
显式临时打开这个门（因为它必须马上把结果当 compiled callable 用），plain 的
export/package 路径不打这个 patch,`current_callable` 保持 `None`（但该路径本就只取
`.filename`，不依赖 `current_callable`）。差异还在前端捕获来源（Dynamo 运行时捕获 vs
`torch.export.export()` 产出的 `ExportedProgram`）与是否额外 `package_aoti` 打包成可跨进程
加载的 `.pt2` 归档。结论写入 B02 §14.2 note，双向加回链；旧 `[!todo]` 原文保留、标注已解答
（本文件"不回写活链接"惯例，`f07...` 引用降级为惰性反引号）。

**复核修正**（同日,spec 审查发现并修复,1 commit）：上一段与 B02 §14.2 note 最初表述为
"两条路径共享完全相同的 C ABI runner、`CompiledAOTI` 本身就是可直接调用对象"，忽略了
`enable_autograd_for_aot` 门控——已按上文改为不对称表述，并据此修正 F07/B02 两页互指
Related Pages 的措辞。另据复核，f06 迁移时 §5 收缩丢失了 `register_backend_for_device` 的
`device_custom_pass: CustomGraphModulePass`/`device_custom_config: ConfigModule` 两个可选
参数与其写入的独立全局注册表 `custom_backend_passes`/`custom_backend_codegen_configs`
事实（原引用 `common.py:389-418`/`:419-434` 经核精确）——已逐字取自 f06 迁移前版本
（`git show ffdff43:19_torch_compile_end_to_end/f06_...md` §5）落回
`codegen_extension_guide.md` §3 表格行 + §4 补充段；§6 被压缩掉的 `DeviceOpOverrides` 专项
方法名（`cpp_aoti_stream_guard`/`aoti_get_stream`/`tma_descriptor_helpers`/`cpp_scratch`,
`common.py:321-382`）也补回该指南 §5。校验：`checker` pages=379 broken=0；`pytest tools/ -q`
77 passed。

**入链修复**：六批共修复 20+ 处 inbound wikilink 到新基名（各篇自身前置/后续互链、course
index 表格逐行更新为新基名 + 迁移/划界说明、`00_torch_compile_end_to_end_index.md` §8）。
`docs/superpowers/specs/2026-07-28-torch-compile-end-to-end-learning-series-design.md` 是历史
设计规格（先例：B/C/D/E 迁移均未回写），本次同样不touch。

**labs 同步**（最后一个 commit 一并完成，先例见 Task 6 d8c1a5b）：`demo_manifest.json` 七个
F 卷 `page` 字段去前缀；`test_volume_demo_contract.py` 新增 `_F_PAGE_ROOTS`（f01-f08 共 8 个
page_id、7 个不同目录,f03/f04 同目录,含 Task 6 已迁的 f08）接入 `_page_root()`；
`test_call_chain_pages_have_source_walkthroughs` 的 f01 target_pages 条目改指新目录/新基名，
移除不再使用的 `course_root` 局部变量。

**校验**：`python tools/check_links.py`：pages=379（无净变化，纯移动+判重非删除），broken=0
（每个子 commit 独立核验）；`pytest tools/ -q`：77 passed（labs 同步 commit 后核验，中间过程
commit 未逐一跑 pytest，随 D 卷先例批量核验于最后一个 commit）。

---

## 2026-07-30：知识库结构整改 P4 Task 8 组 D 步骤 4（C21 vs codegen 碎片四页，收尾）

**Type**: Cross-link Reconciliation（设计：`docs/superpowers/plans/2026-07-30-kb-reorg-p4-ai-frameworks.md` Task 8 组 D）

C21（`codegen_kernel_mapping_autotuning_and_provenance_analysis.md`，505 行）定位为
codegen/kernel 映射/autotuning/provenance 的**总线页**（覆盖面广、单点纵深浅）。逐页核对
四篇碎片，均判定独有内容 >50%，**保留为专项，不合并**：

| 碎片页 | 独有度判定 | 依据 |
|---|---|---|
| `inductor_codegen_analysis.md`（283 行，含 Task 6 回补的 §7 CPU CodeGen） | 独有 | §7 CPU kernel 类体系（`CppKernel`/`CppVecKernel`/`CppTile2DKernel`）与向量化判定完全不被 C21 覆盖；§2.2 后端文件清单、§4.4 wrapper `IndentedBuffer` 段拆分、§4.3 `TritonKernel.codegen_kernel()` 内部结构均比 C21 对应段落更具体 |
| `inductor_gpu_kernel_dispatch_model.md`（98 行） | 独有 | `program_id→offset→index→mask` kernel 骨架、`IterationRanges` 树、`select_tiling` 打分算法、`GridExpr`/Y-Z 溢出族，全篇贯穿 NPU 对照，C21 §6"Loop codegen"仅 10 行概念性带过 |
| `inductor_reduction_codegen_deep_analysis.md`（89 行） | 独有 | persistent/looped/split/cooperative reduction 四种形态的具体代码生成与阈值，C21 未涉及 reduction codegen 细节 |
| `inductor_autotuning_analysis.md`（152 行，含 Task 6 回补的 §六 ChoiceCaller/TuningProcessPool，本任务组 D 步骤 2 新增 §七 CoordescTuner） | 独有 | `CachingAutotuner` 生命周期、config 启发式、`AttrsDescriptor`、Triton 编译链、NPU 对照表，C21 §8"两层autotuning"只有 20 行提及两层机制存在 |

**处置**：不做内容合并，只补齐此前缺失的双向交叉链接——C21 Related Pages 新增到
`inductor_gpu_kernel_dispatch_model`/`inductor_reduction_codegen_deep_analysis` 的链接
（此前只链 `inductor_codegen_analysis`/`inductor_autotuning_analysis`）；后两篇各自新增
回链 C21。四篇碎片页 Task 6 回补的独有内容（CPU codegen、ChoiceCaller/TuningProcessPool）
确认原样保留，未被本任务触碰或覆盖。

**校验**：`python tools/check_links.py`：pages=379（无净变化，纯加链接），broken=0；
`pytest tools/ -q`：77 passed。

**Task 8 组 D 全组小结**（步骤 1-4 共 4 commit）：C18-C21 从 `19_torch_compile_end_to_end`
纯移动至 `04_inductor/`；`PyTorch_Inductor_Technical_Analysis.md`（1699 行）与
`scheduler_analysis.md`（964 行）两个大型综合参考页判重后完全解体，独有内容分别落地到
`codegen_extension_guide`/`inductor_autotuning_analysis`/`post_grad_passes_guide`/C20 四页；
`inductor_memory_management_analysis.md`（278 行）与
`wrapper_execution_memory_allocation_and_reuse_analysis.md`（D05，228 行）并入 C19 成为
内存主题统一入口；C21 与四篇 codegen 碎片页确认分工清晰、只补链接。全组共删除 4 个页面
（1699+964+278+228 = 3169 行），净新增/改写约 1500 行精炼内容，checker 全程 broken=0。

**追记（复核披露，2026-07-30，不改上方历史行）**：事后复核发现上方步骤 2（8e6a6aa，解体
`PyTorch_Inductor_Technical_Analysis.md`/`scheduler_analysis.md`，本小结未单列 commit 但已
计入上面的删除统计）存在两处需要补救的问题，另有两处内容确认为已核实但归一时未落地：

1. **§9 整删决定推翻了 P3（97840f0）的隔离保留裁定**：97840f0 已明确裁定 addmulnorm 自定义
   融合规则教学"整节仅作历史材料保留"（unresolved-quarantine），8e6a6aa 却以"已自我标注为
   unverified/fictional"为由整删、未见零痕迹处置的说明。现补救：`codegen_extension_guide.md`
   新增 §11 隔离存根，记录该教学曾存在、§9.4 现有融合规则引用（`post_grad.py`/`mm_plus_mm.py`
   等）真实可核、教学主体未验证且含虚构文件名，并指路 `git show 6579658:.../PyTorch_Inductor_Technical_Analysis.md`
   查原文。
2. **§8"虚构"定性亦有夸大**：8e6a6aa 的 commit message 把整个 §8 后端扩展教学描述为
   "uncited/fictional examples"，但复核发现 §8.2 核心数据结构一节确有真实引用
   （`torch/_inductor/codegen/common.py:313`，`DeviceCodegen`/`device_codegens` 的真实定义
   行）；"虚构"定性对 §8 整节而言不准确，已在上述 §11 存根中一并披露口径偏差。
3. **C20（`scheduler_dependency_graph_fusion_and_ordering_analysis.md`）"逐字并入"表述不准
   确**：§18/§19 原标注"从 `scheduler_analysis.md` §7/§9 逐字并入"，实际该页解体时省略了
   §7.3 的 `can_fuse` 合法性判定流程图、§4 的 11 类核心类结构类图，§8.1 的 `min_order`/
   `max_order` O(1) 粗筛 rationale 也未落地——均非逐字。现已回补三处（§1.1 新增类图、§18.3
   新增流程图并保留 P3 当年对该图"单一阈值流程"误导提示的修正注、原 §5 补 `min_order`/
   `max_order` 说明），三处新回补内容保留原文 `L<line>` 引用形式；措辞相应改为"改写并入
   （示意图与部分代码曾省略，经复核回补）"。系统性被剥离的其余 file:line 引用不做全量恢复。
4. **两处已核实但未落地的内容，本次一并补上**：`saved_tensors_recompute_and_runtime_abi_analysis.md`
   （C10）§11.1"重放 view"句后加 `[!todo]` 指出该机制由 `gen_alias_from_base`
   （`functional_utils.py:315`，`runtime_wrappers.py` 六处调用，均已按当前 pinned checkout
   核验）实现，原 `aotautograd_analysis` §10.2 曾展开、归一时未落地；
   `buffer_liveness_memory_planning_and_reuse_analysis.md`（C19）§18.1 回补两个被丢的
   `output_code.py:785-813`/`:817-840` 定位符（对照已删除的 D05 原文位置核实）。

校验：`python tools/check_links.py` pages=379，broken=0；`pytest tools/ -q` 77 passed。

## 2026-07-30：知识库结构整改 P4 Task 8 组 D 步骤 3（C19+D05 内存归一）

**Type**: Redundancy Merge（设计：`docs/superpowers/plans/2026-07-30-kb-reorg-p4-ai-frameworks.md` Task 8 组 D）

C19（`buffer_liveness_memory_planning_and_reuse_analysis.md`，编译期 realize/last-use/reuse
权威页）确立为内存主题骨架，吸收两页独有内容后二者删除：

- **`inductor_memory_management_analysis.md`（278 行）**：C19 此前完全不覆盖运行期
  `CUDACachingAllocator`（层 2）与 CUDA Graphs `cudagraph_trees` 私有池（层 3）——**独有，
  逐字改写并入 C19 新增 §16**（三层总览 + 层 2 段大小档位 + 层 3 树结构/共享私有池/地址
  稳定/checkpoint/graph partition 集成）；其 §2.6 池初始化大小的完整推导 + 真实
  `test_memory_planning.py` 实例，深于 C19 原有 §7-8 对同一批类的引用式带过——**独有，
  并入 C19 新增 §17**。该页 §2.1-2.5（编译期规划本身）与 C19 §1-15 重叠，未重复搬运。
- **`wrapper_execution_memory_allocation_and_reuse_analysis.md`（D05，228 行）**：boxed
  calling convention（`_BoxedCallable`/`CompiledFxGraph.__call__`）与通信 buffer 独立池
  两个事实 C19 完全未提——**独有，并入 C19 新增 §18.1/§18.2**；`AllocateLine.plan` 的
  完整 7 步决策序列比 C19 §6 的概念性描述更程序化——**并入 C19 新增 §18.3**。该页 §1-3/§9-14
  （boxed wrapper 宏观框架、liveness/reuse 三层区分、复杂度、常见误解）与 C19 §1-2/§6/§13
  重叠，未重复搬运。

**`inductor_memory_allocation_guide.md`（209 行）判定**：>50% 独有（分配器选型对照表、
`memory_stats`/snapshot 实测复现代码、§5 完整的内存越界/踩踏排查流程含
`compute-sanitizer` 工具链、§7 与外部报告的差异订正）——**保留**，6 处指向已删除
`inductor_memory_management_analysis` 的引用改指 C19 对应新章节（§16/§17）。

**入链修复**：C19 自身新增 3 处 Related Pages；`inductor_memory_management_analysis` 删除后
6 个外部文件改指；`wrapper_execution_memory_allocation_and_reuse_analysis` 删除后 7 个外部
文件改指（含 `04_inductor/index.md` 两行表格合并为一行、`00_torch_compile_end_to_end_index.md`
D05 行改指）；changelog 4 处历史活链接降级为反引号。

**配套改动**：`demo_manifest.json` 移除 d05 条目——其页面已并入 C19，而清单要求
`page` 字段全局唯一（C19 自己的 c19 条目已占用该文件名）；`test_volume_demo_contract.py`
的 `expected_ids`/条目计数（55→54）与 `_D_PAGE_ROOTS` 同步；原 d05 demo 用例
（`demo_d_artifact_runtime.py --case wrapper_memory_reuse`）仍可运行，C19 §18.3 末尾以
指路形式保留命令（C 卷页本就无需通过 manifest 强制"配套 Demo"小节）。

**校验**：`python tools/check_links.py`：pages 381→379，broken=0；`pytest tools/ -q`：77 passed。

## 2026-07-30：知识库结构整改 P4 Task 8 组 C（C17 vs lowering_analysis 归一）

**Type**: Move + Redundancy Merge（设计：`docs/superpowers/plans/2026-07-30-kb-reorg-p4-ai-frameworks.md` Task 8 组 C）

C17（435 行）`git mv` 为 `02_compile_stack/04_inductor/fx_lowering_to_inductor_ir_analysis.md`
（同目录内平移，标题保留原"17 ·"前缀）。

**vs `lowering_analysis.md`（448 行）判重**：两页体裁互补——C17 是"GraphLowering 怎样把
FX 解释成 IR"的机制权威页（Interpreter 状态模型、决策树式 `call_function` 选择、
realization 时机），`lowering_analysis` 是函数/API 级完整参考。用本地 pinned pytorch
checkout（`e8f97c1a6e...`）逐条核验后：

- §一/§二/§四/Call Chain/Data Flow/Key Design Decisions/Beginner Summary：与 C17 §1-§4
  概念重叠，且部分表述（"fallback 兜底保证编译永不失败"）已被 C17 §6 证伪，页内自身也已用
  `[!注]` 标注这些旧结论不成立并指回 C17——删除，不搬运。
- §2.2/2.2.1/2.2.2 完整注册 API 面（`register_pointwise`/`register_foreach_pointwise`/
  `add_needs_realized_inputs`/`add_layout_constraint`/`make_fallback` 等 C17 §4 未点名的
  API）+ 两个实操接入示例（纯 fallback 接入、复用已有 lowering 组合新 op）：**独有，
  改写并入 C17 新增 §4.3**，全部函数位置逐一核验现存（行号相对旧参考漂移 200-1500 行
  不等，以函数名定位为准，页内注明）。
- §三"Lowering 做了什么优化"完整八类目录（Pointwise 融合基础/View 零拷贝家族/Reduction
  优化含 Welford 与 OnlineSoftmax/常量折叠提升/智能 Fallback/Foreach 水平融合/量化 op
  融合/Layout 约束优化）：**独有，改写并入 C17 新增 §16**，逐函数核验存在
  （`make_pointwise`/`make_reduction`/`var_mean_welford_`/`promote_constants`/
  `maybe_layout_constraints` 等）。
- §五"自己做 Lowering 要关注的优化点"三级检查清单（基础/高优先级/进阶）+ Questions &
  Uncertainties（complex tensor 支持不完整、unbacked symbol slice/select 复杂、
  OnlineSoftmax 不支持 split reduction）：**独有，改写并入 C17 新增 §17**。

**入链修复**：C17 基名改名影响 12 个文件；`lowering_analysis` 删除后 14 个外部文件
（含 `PyTorch_Inductor_Technical_Analysis`、`torch_compile_architecture`、
`inductor_reduction_codegen_deep_analysis`、`npu_lowering_guide` 等）重定向到
[[10_fx_lowering_to_inductor_ir_analysis]]；changelog 1 处历史活链接降级为反引号
（另 1 处已是逐项反引号包裹，未受影响）。

**配套改动**：`demo_manifest.json` c17 `page` 字段与 `test_volume_demo_contract.py` 的
`_C_PAGE_ROOTS` 同步新增 `c17 → 04_inductor`。

**校验**：`python tools/check_links.py`：pages 384→383，broken=0；`pytest tools/ -q`：77 passed。

## 2026-07-30：知识库结构整改 P4 Task 8 组 B（C09/C10 vs aotautograd_analysis 1460 行归一）

**Type**: Move + Redundancy Merge + New Specialist Page（设计：`docs/superpowers/plans/2026-07-30-kb-reorg-p4-ai-frameworks.md` Task 8 组 B）

C09（555 行）`git mv` 为 `02_compile_stack/02_aot_autograd/aotautograd_joint_forward_backward_graphs_analysis.md`；
C10（491 行）`git mv` 为同目录 `saved_tensors_recompute_and_runtime_abi_analysis.md`。标题保留原
"09 ·"/"10 ·" 编号前缀（沿用 Task 5/7 惯例：文件名去前缀、标题不去）。

**vs `aotautograd_analysis.md`（1460 行 = 原 1127 + Task 3 补的 §13）逐节台账**：

| 节 | 判定 | 处置 |
|---|---|---|
| §1-§2 概览/工作流 | 与 C09 §1 自带 mermaid、页头「课程主线与本页分工」表完全重叠 | 删除，不搬运 |
| §3 Phase1 捕获+元数据（含 §3.3 InputAliasInfo/OutputAliasInfo/OutputType dataclass 字段列表） | 用本地 pinned pytorch checkout（`e8f97c1a6e...`）核验：字段基本准确但行号漂移 140-450 行；[[12_graph_effects_alias_mutation_and_order_analysis]]（C05）已用相同源码位置覆盖同一组 dataclass | 删除，不重复搬运 |
| §4/§8 functionalization 包装器链、子类解包 | 与 C05 已覆盖的 dedupe/synthetic-base wrapper 顺序重叠（C05 已引用 `runtime_wrappers.py:1586/1844` 等同一组行号） | 删除 |
| §5 Phase3 分区与编译 | 伪代码/无精确行号，被 C09 §5-§11 的逐行验证内容全面超越 | 删除 |
| §6 decomposition/常量折叠/DCE/pattern matching | 不属于 C09/C10 主题，分别已被 [[15_graph_normalization_decomposition_and_functionalization_analysis]]/`pattern_expression_and_matcher_engine_analysis`/`dead_code_topology_and_effect_order_analysis`（Task 7 新页）覆盖 | 删除 |
| §7 Phase5 运行时包装器链 + `_HANDLER_MAP` | 核验 `AOTDedupeWrapper`/`AOTSyntheticBaseWrapper` 已被 C05 覆盖；但 post-compile 链（`RuntimeWrapper:189`/`AOTDispatchSubclassWrapper:1406`/`FunctionalizedRngRuntimeWrapper:1212`/`AOTDispatchAutograd:3624`）与 `_HANDLER_MAP`/`OutputType→Handler` 派发表（`runtime_wrappers.py:320-345`）经核验后确认未被任何现存页覆盖 | **独有，逐字改写并入 C10 新增 §11.1** |
| §9 AOTConfig/ViewAndMutationMeta/AOTState 数据结构 | ViewAndMutationMeta 行号（446-475）与 C09 已有引用一致；AOTConfig/AOTState 字段无深层机制增量 | 删除 |
| §10.1 激活检查点/重计算 | 与 C10 §5-§8 min-cut 内容重叠 | 删除 |
| §10.2 视图重放优化（`gen_alias_from_base`） | 判定为 C05 alias territory，本任务范围外，暂未新落地（记为待核验缺口） | 未迁移，留 `[!todo]` 级别观察项（未写入正文，本条目记录） |
| §10.3/§10.4 静态输入优化/AOTAutogradCache | AOTAutogradCache 已有专页 [[11_aotautograd_cache_analysis]] | 删除 |
| §11.1/§11.2 输入别名/输出别名限制 | synthetic base 已被 C05 覆盖；输出别名 assert 为单行低价值 | 删除 |
| §11.3 自定义 autograd 函数检测 | 核验：`_is_result_of_custom_autograd_fn` 现已内联（非独立函数），逻辑见 `collect_metadata_analysis.py:479-485`，其对 `OutputType.custom_function_view` 分类的影响见 `:490-497`；未被任何现存页覆盖 | **独有，改写并入 C09 新增 §17** |
| §11.4 `aot_export` 元数据变异禁令 | 核验现行位置 `aot_autograd.py:651-657`（原引用漂移约 350 行）；未被覆盖 | **独有，并入 C09 新增 §17** |
| §12/附录 总结、术语表、参考资料 | 通用摘要，无独有事实 | 删除 |
| §13 ProxyTensor/FakeTensor（Task 3 从已删 A04 页迁入） | 本页此前称"未展开"；核验全部 13.1-13.11 引用行号在当前基线漂移 5-20 行内，内容仍准确 | **完整独立成页** `dispatch_modes_proxytensor_faketensor_analysis.md`，不塞进 C09（主题是 dispatch-mode 机制而非 joint 图提取本身），逐字保留 |

**vs `fx_graph_construction_and_transformation_analysis.md`（269 行，Task 7 瘦身后的 AOT 残留页）**：
§3.1（通用 Proxy/tracer 建 Node 路径）核验已被 [[10_fx_graph_core_data_model_analysis]] 覆盖（同引用
`torch/fx/proxy.py:600-635`），删除；§3.2-3.4（joint 构造、partition 提取、跨图 ABI）与 C09 §2-§11
重叠，删除；§4.1（min-cut 选择保存/重算，**计划点名"两轮审查确认为最深版本"**）核验独有事实
`activation_memory_budget=0/1` 两个边界的快速路径行为（`partitioners.py:3471-3480`，直接读源码确认：
`==0` 返回 `node_info.inputs`，`==1` 直接返回 `solve_min_cut` 结果不再进 knapsack）——**逐字改写并入
C10 §7 新增段**；§4.2-4.3 与 C10 §8-§10 重叠，删除；§10「残留提醒」两条中"saved tensor 不是跨图
Node 引用"已是 C10 §16 原文，删除；"bw 不是反向边图"与"阅读图时的四问"改写并入 C09 新增 §18；
§11 源码导航表全部行已是正文内联引用，删除独立表格。

**遗留两小项处理**：①该页原 §3.3 "有一处链接被 Task 7 规范化"——核实为历史记录性描述，指向已完成
的 Task 7 改动，不需要额外动作，此处记录确认。②"GraphNode 非独立节点类型"一句从
`26aeabb`（Task 6 commit，该 commit 时 fx_graph_construction 文件仍含 §2.1 全文）取回原文，逐字补入
[[10_fx_graph_core_data_model_analysis]] §2.2（C02）Node 定义段末尾，标注取回来源与日期。

**入链修复**：C09/C10 基名改名共影响 24 个文件的裸链接/路径限定链接（图系列两个 00 索引表格行、
自身互指、`graph_stage_boundaries_identity_and_provenance_analysis` 前置行、`aot_autograd_quickstart`、
`aotautograd_cache_analysis`、`memory_amp_profiler`/`12_activation_checkpointing_analysis` 等）；
`aotautograd_analysis` 删除后 15 个外部文件重定向（按引用内容精确改指 C09/C10/新
`dispatch_modes_proxytensor_faketensor_analysis` 三个目标之一，而非笼统指回同一处）；
`fx_graph_construction_and_transformation_analysis` 删除后 2 个外部文件改指；`02_aot_autograd/index.md`
两处表格重写；changelog 3 处历史活链接按"历史不回写"惯例降级为反引号+去向说明。

**配套改动**：`demo_manifest.json` c09/c10 `page` 字段与 `test_volume_demo_contract.py` 的
`_C_PAGE_ROOTS` 同步新增 `c09/c10 → 02_aot_autograd`。

**校验**：`python tools/check_links.py`：pages 385→384（净减 1，两页删除、一页新增），broken=0；
`pytest tools/ -q`：77 passed。

## 2026-07-30：知识库结构整改 P4 Task 8 组 A（C04 + 动态形状归一）

**Type**: Move + Redundancy Merge（设计：`docs/superpowers/plans/2026-07-30-kb-reorg-p4-ai-frameworks.md` Task 8 组 A）

`19_torch_compile_end_to_end/04_symbolic_shapes_guards_and_graph_reuse.md`（415 行）`git mv`
到 `02_compile_stack/01_dynamo/symbolic_shapes_guards_and_graph_reuse_analysis.md`，作为符号
形状系统（ShapeEnv/SymNode/guard 生成/backed·unbacked 判定）的**概念权威页**。

**vs `dynamic_shapes_full_analysis.md`（460 行）判重**：两页覆盖同一主题但视角不同——C04 是
按机制分层的概念权威叙述，旧页是按 Dynamo→ShapeEnv→Guard→Inductor 链路组织的纵深稿。逐节核对
后确认旧页 §1-§4、§5.2-5.5 的大部分事实（EQUALS_MATCH、ShapeEnv 字段表、guard 三层生成、
automatic_dynamic_shapes 渐进策略、关键源码索引表）已被 C04 现有正文以不同措辞覆盖，真正独有
的三处逐字/改写并入：
1. `DimDynamic` 五值枚举（DYNAMIC/DUCK/STATIC/UNBACKED/INFER_STRIDE，`torch/fx/experimental/
   symbolic_shapes.py:1988`，据当前基线 `e8f97c1a6e...` 重新核验行号，原页引用的 1967 已漂移）
   ——并入 C04 §7 新增子节「维度分配策略：`DimDynamic`」。
2. `recompile_limit=8` 与 `EQUALS_MATCH`（`torch/_dynamo/guards.py:2772`，原引用 2638 已漂移）
   ——并入 C04 §2 静态特化缺点段。
3. `matmul` 端到端案例（Stage 0-5，用户代码→Dynamo 捕获→ShapeEnv 状态→guard 生成→Inductor
   codegen→第二次调用命中缓存）——改写为 C04 新增 §15，SizeArg/buffer_reuse_key 引用行号据
   当前基线重新核验（`common.py:286`、`wrapper.py:123`，原引用 291/100 已漂移）。
其余内容判定为同一事实的不同措辞，不重复搬运。旧页 mermaid 状态图（automatic dynamic 四次调用
时间线）与 C04 §8 结论重叠且与 b09 §9 的状态机图功能重复，判定为可省略的重复可视化，未搬运。

**与 b09/unbacked_symint/inductor_codegen_dynamic_shape 划界**：四页互相新增分工声明——
[[17_dynamic_shapes_generalization_and_fallback_analysis]]（b09）聚焦 Dynamo 侧自动泛化行为
（`frame_state`/`mark_dynamic`）；C04 是符号系统概念权威页；[[25_unbacked_symint_analysis]] 聚焦
unbacked 专项（`torch._check`/`guard_or_*`/size-oblivious）；[[24_inductor_codegen_dynamic_shape_analysis]]
聚焦符号如何流入 Inductor kernel/wrapper。四页页头/正文均补互链，不复述彼此机制细节。

**入链修复**：`dynamic_shapes_full_analysis` 删除后，18 处活链接改指新页——
`fx_graph_cache_analysis`、`dynamo_pgo_cache_analysis`（2 处）、`aotautograd_cache_analysis`、
`04_inductor/index`（表格行改为一行式说明指向 `01_dynamo/index`）、
`inductor_codegen_dynamic_shape_analysis`、`inductor_quickstart`（2 处）、
`npu_compile_paths_overview`、`npu_inductor_linearize_dynamic_shape_analysis`、
`torch_compile_architecture`（2 处）、`npu_inductor_optimization_analysis`（2 处），另 10 处
`04_symbolic_shapes_guards_and_graph_reuse` 裸链（图系列两个 00 索引表格行、C02/C03 前后篇
导航、b09、`06_compile_cache/index`）改指新基名。`wiki/changelog.md` 历史条目中 1 处活链接
（原 2026-07 中旬 NPU codegen 页条目）按"历史不回写"惯例降级为反引号 + 去向说明。

**配套改动**：`tools/labs_torch_compile/demo_manifest.json` 的 c04 `page` 字段与
`test_volume_demo_contract.py` 的 `_C_PAGE_ROOTS` 同步新增 `c04 → 01_dynamo`。

**校验**：`python tools/check_links.py`：pages 386→385，broken=0；`pytest tools/ -q`：77 passed。

## 2026-07-30：知识库结构整改 P4 Task 7 组 4（control_flow_capture_analysis vs C06 收尾判重）

**Type**: Redundancy Review + Boundary Clarification（设计：`docs/superpowers/plans/2026-07-30-kb-reorg-p4-ai-frameworks.md` Task 7；Task 5 遗留收尾）

`02_compile_stack/01_dynamo/control_flow_capture_analysis.md`（204 行）vs
`00_torch_compile_end_to_end_index`（已于 P4 Task 10 删除，导读价值并入 [[courses/torch_compile_end_to_end]]）卷 B 的 b04 判重已在
Task 5 完成（零重叠）；本组补做 vs C06
（[[13_structured_outputs_higher_order_and_nested_graphs_analysis]]）的判重，Task 7 组 1-3
未涉及的最后一项遗留。

**判定**：两页体裁不同——本页讲**捕获前端**：Dynamo 字节码符号执行期间"控制流该不该
入图、走哪条路径"（`speculate_subgraph`/`generic_jump`/graph break，均是 `torch/_dynamo/`
内部机制，源码引用 `torch/_dynamo/variables/higher_order_ops.py`、
`torch/_dynamo/symbolic_convert.py`）；C06 讲**IR 层结构**：不论谁捕获（Dynamo/`make_fx`/
`torch.export`），outer/child GraphModule 的 ownership、pytree、DCE 递归边界怎样表达
（源码引用 `torch/_higher_order_ops/`、`torch/fx/`）。逐节核对确认本页 §1-§4（两条路径
框架、`speculate_subgraph` 四步机制、`cond` 深挖的 Dynamo 侧投机/checkpoint-rollback、
控制流 HOP 家族、原生 `if/for/while` 字节码分流、"trace 两支/编译两支/运行一支"三个
误解辨析、路径选型表）在 C06 中**零重叠**——C06 全篇未提及 `speculate_subgraph`、
`VariableTracker`、`generic_jump`、`symbolic_convert`、graph break 中任何一个术语。

**唯一重叠段**：本页 §2.4「下游:HOP 在编译后端的处理」的 `FakeTensorMode`/
`ProxyTorchDispatchMode` 两条 bullet（`cond.py:408/403`，讲 `trace_cond`/FakeTensor merge
机制）与 C06 §6 及其"源码跟读"§1-§3 讲的是**同一段源码**（locator 一致），但 C06 是完整
源码跟读、本页只是两行下游提及——收缩本页这两条为互指 C06，保留 §2.4 另外两条独有 bullet
（`py_functionalize_impl`/Inductor `Conditional` IR，C06 未涉及）。

**互链**：本页页头新增与 C06 的划界说明段；页尾 Related Pages 加一条指向 C06；C06 §6 开头
加一句指回本页（"Dynamo 如何决定该不该走 cond 是更早一层问题"），Related Pages 加一条
反向链接。本组是四组中改动量最小的一组，符合预期（该页"大概率保留，只补划界"）。

**校验**：`python tools/check_links.py`：pages 386→386，broken=0；`pytest tools/ -q`：
77 passed。

---

## 2026-07-30：知识库结构整改 P4 Task 7 组 2（pass 方法论归一：C13/C15/C16 吸收两旧页后删除）

**Type**: Redundancy Consolidation（设计：`docs/superpowers/plans/2026-07-30-kb-reorg-p4-ai-frameworks.md` Task 7）

`04_inductor/fx_pass_optimization_methodology.md`（349 行）+
`04_inductor/torch_upstream_pass_deepdive.md`（232 行）逐节比对 [[22_pattern_expression_and_matcher_engine_analysis]]（C13）/
[[24_graph_pass_pipeline_ordering_and_fixpoint_analysis]]（C15）/
[[25_graph_rewrite_legality_validation_and_complexity_analysis]]（C16）。判重结论：两旧页
的机制细节（PatternEntry 三型、序列化缓存、pre/joint/post_grad 三阶段执行顺序、
`b2b_gemm`/`fused_int_mm_mul` 等具体 pass 的门控与合法性）绝大部分已被 C13/C15/C16
以**更细粒度的当前基线 locator**覆盖（例如 C13 §15 复杂度已含旧页没有的 mutation-region
扫描成本、sort-key 比较成本项）；旧页 §3.1-3.4「pass 全集目录」与 `pre_grad_passes_guide`/
`joint_graph_passes_guide`/`post_grad_passes_guide`（各 800+ 行的 Inductor 阶段实操指南，
不属本目录范围）逐条重复，判定冗余不迁移。

**逐条核实并逐字迁移的独有事实**（迁移前用本地 pinned checkout `E:/97-codes/torch_parallel/p`
@ `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`——与 C13/C15/C16 页头基线完全一致——重新核验
了每条 locator，不是照搬旧页的 `9922478dffa` 基线数字；drift 处以本次核验结果为准）：

- **C13 新增两节**：`fwd_only` vs `joint_fwd_bwd` 两种 trace 模式（`pattern_matcher.py:2583/2613`，
  各自决定 pattern 匹配推理图还是训练图）；声明式 pattern vs 手工 `graph.find_nodes` 遍历
  两条并存改图路线 + "归一化 pass 为融合 pass 铺路"技巧。
- **C15 新增/扩写六处**：①pre-grad 的 `config.pattern_matcher` 门控（`config.py:290`）+
  `is_predispatch` 分岔到 `_run_pre_dispatch_passes`/`default_pass_list`
  （`pre_grad.py:200-224,353-361`）；②`lazy_init()` 在 pre_grad（`pre_grad.py:174-182`，
  imports `apply_gumbel_max_trick`/`efficient_conv_bn_eval`/`split_cat`）与 joint
  （`joint_graph.py:81-89`，imports `_pad_mm_init`/`_sfdp_init`/`_misc_patterns_init`）
  两处的具体注册内容；③post-grad 三桶 `pass_patterns[0]→[1]→[2]` 严格顺序
  （`post_grad.py:85-88`）+ auto_chunker 必须早于 pad_mm 的源码注释原文
  （`joint_graph.py:733-734`）+ `reinplace_inplaceable_ops` 具名尾部锚点
  （`post_grad.py:451-452`）；④`GraphTransformObserver` 完整机制新增专节——类定位
  （`torch/fx/passes/graph_transform_observer.py:22`）、`dynamo_timed(f"pass.{subsystem}.
  {passname}")` 计时格式、按名 `config.disabled_passes` 与按子系统 `CompilerBisector.
  disable_subsystem` 两级禁用（79-116）；⑤`GroupBatchFusionBase`/`GroupFusion`/
  `BatchFusion`/`register_fusion` 机制新增专节——与 §2.1 起的 `PatternExpr` 声明式匹配
  并存的另一套改图基础设施（`group_batch_fusion.py:101-159,1488,1615,1664,1677`）；
  ⑥六个"为什么"pass 设计检查框架，插入 §11 决策树之后并逐条标注在本系列的既有落点
  （不重开一套独立标准）。三则源码取证式脚注核实后**在当前基线仍然成立**，随对应
  driver 小节落地：`binary_folding.py` 实际归属 freezing 而非 `pre_grad_passes()`
  （`freezing_patterns.py:98-101,115`、`config.py:1670`）；`decompose_mem_bound_mm.py`
  只被 joint 借用 `check_device` 一个辅助函数，不是 joint pass 本身（`joint_graph.py:40,989`）；
  `b2b_gemm`/`micro_pipeline_tp` 是仅有的两个不经 `GraphTransformObserver` 包裹的
  post-grad pass（`post_grad.py:266-270`）；`check_shape_cuda_and_fused_int_mm_mul_enabled`
  全仓库零引用、确系孤儿函数，`config.decompose_mem_bound_mm` 名不副实——真正门控是
  `torch._C._has_mkldnn` + `post_grad_fusion_options` 字典 key（`post_grad.py:833-838,2048`、
  `config.py:1636`）。
- **C15 新增 §14「跨框架方法论对照」**（明确标注基线独立于本页，torch_npu/vLLM/SGLang
  三个基线不随 upstream 同步）：四家现状对照表 + "融合朝向谱系"一句话读法 + 两种本页
  未覆盖的下游落地形态（rewrite-existing-op、fallback/换手工算子）。这是本组唯一保留
  跨基线内容的一节，按计划指令"四家现状归纳...逐字并入对应 C 篇"执行，不因体裁不完全
  匹配而丢弃。
- **C16 §5 新增"三条可复用的安全判据"**：算子类别不变式、边局部改写+纯函数前提、单用户
  门槛判据——三条判据在 upstream/torch_npu/vLLM/SGLang 独立收敛，故上调为通用安全底线，
  vendor 专属举例（torch_npu `fold_cat`/`fold_squeeze`）未随之带入以保持 C16 纯 upstream
  机制页定位。**C16 §8 差异测试矩阵新增"收益"行**（pass 开/关 A/B、命中计数、端到端性能）。

**结构性冗余、判定不迁移**：§0-§0.8 八阶段"是什么/为什么放这里/适合做/不适合做"的教学
式 prose + 代码示例（C15 §1/§11 的表格化处理已覆盖同等结论，旧页的 pedagogical 铺陈
判定为表达形式而非独有事实）；序列化 pattern 缓存机制（C13 §13 已近逐句覆盖，含同一
`PYTORCH_GEN_PATTERNS` 环境变量）；三种 `PatternEntry`/lowering-pattern vs graph/
replacement-pattern 落地形态区分（C13 §11 locator 更细）；§3.1-3.4 pass 全集目录（与
三份 stage guide 冗余，非本目录职责）。

**两页删除**：`fx_pass_optimization_methodology.md`（349 行）、`torch_upstream_pass_
deepdive.md`（232 行）。**入链改指**：非 changelog 入链共 16 处（`dynamo_pass_methodology`/
`codegen_extension_guide`/`decomposition_passes_guide`/`joint_graph_passes_guide`/
`lowering_analysis`/`post_grad_passes_guide`/`pre_grad_passes_guide` 各 1 处一致模式的
Related Pages 单行，批量改指 C15；`04_inductor/index.md` 表格两行合并为一行更新描述；
`sglang/index.md` 两处 `../../` 相对路径链接顺带改为裸基名（分别指向 C13/C15，消除既有
`../` 违规）；`sglang_compilation_passes_analysis.md`/`vllm_ir_and_fusion_passes_analysis.md`
各 2-3 处按内容落点分别改指 C13（引擎机制）或 C15 §14（方法论/跨框架对照)）。changelog
历史条目（2026-07-22、2026-07-20 各一条）中的 3 处活链接按"历史不回写"惯例降级为反引号
+ 说明去向。

**校验**：`python tools/check_links.py`：pages 388→386（两页删除），broken=0；
`pytest tools/ -q`：77 passed。

---

## 2026-07-30：知识库结构整改 P4 Task 7 组 3（decomposition_passes_guide vs C08 判重，保留双页）

**Type**: Redundancy Review（no merge）（设计：`docs/superpowers/plans/2026-07-30-kb-reorg-p4-ai-frameworks.md` Task 7）

`04_inductor/decomposition_passes_guide.md`（163 行）逐节核对 vs
[[15_graph_normalization_decomposition_and_functionalization_analysis]]（C08，图规范化/Decomposition/
Functionalization 机制骨架）：guide 的 §1（decomposition table 结构与 mermaid）、§3（`register_
decomposition`/`decompositions`/`select_decomp_table`/`torch._decomp.get_decompositions`/
`compile_fx(decompositions=table)` API 表）、§4（注册并加入编译的可运行代码示例）、§5（写
decomposition 前必须回答的算子集收益/语义等价/动态形状三组问题，含 dtype promotion、整数
溢出、NaN/Inf、复数、低精度误差、RNG 顺序、高阶梯度等细项）、§7（六条验证清单）、§8（五条
反模式）在 C08 中**逐句核实均不存在**——C08 是捕获期机制解释（为什么/怎样发生），guide 是
面向开发者的 API 参考+可运行示例+checklist（如何写一个新 decomposition），两者体裁不同，
guide 独有内容占比远超 50% 判据。§2（适合/不适合做的理由）与 C08 §4（收益/代价）主题重叠
但角度不同（guide 是"该不该做"的决策框架，C08 是"是什么"的描述），判定为互补而非重复，
不删减。

**判定**：按计划判据（独有 >50% 时保留为 guide）保留双页，不合并、不删除。双向互链已存在
（guide 页头「课程分工」→ C08；C08 Related Pages → guide），本次只把 C08 → guide 的反向链接
补充一句说明区分二者体裁，避免后续误判为待归一的重复对。

**校验**：`python tools/check_links.py`：pages 388→388，broken=0（本组无内容迁移，仅一行
Related Pages 描述补充）。

---

## 2026-07-30：知识库结构整改 P4 Task 7 组 1（fx_graph_construction_and_transformation_analysis 判重归一）

**Type**: Redundancy Consolidation（设计：`docs/superpowers/plans/2026-07-30-kb-reorg-p4-ai-frameworks.md` Task 7）

`02_compile_stack/02_aot_autograd/fx_graph_construction_and_transformation_analysis.md`（2026-07-23
综合报告快照，601 行）自带「阅读定位与迁移去向」节，按其自述逐节判重 vs Task 7 Step 1
迁入的 C02/03/05/12/13/14/16：

- **§2 FX 对象模型、边和图序**：逐段核对（`Graph`/`Node`/`GraphModule` 三对象表、
  `_update_args_kwargs`/`torch/csrc/fx/node.cpp` C++ 热点实现、`Graph.nodes` 链表遍历、
  `graph.lint()`、链表 vs 普通 list 的设计动机）均已存在于 [[10_fx_graph_core_data_model_analysis]]
  （locator 一致或仅个位数行号漂移，判定为 drift 非新事实）。唯一未被覆盖的一句独有澄清——
  "`users` 反向邻接"与"AOTAutograd backward `GraphModule`"是两个不同概念、不要混为一谈——
  逐字并入该页 §4「两种"反向"不要混为一谈」小节，并加一句到 `19_torch_compile_end_to_end/09_aotautograd_joint_forward_backward_graphs`（历史活链接，该页已于 2026-07-30 移动为 [[11_aotautograd_joint_forward_backward_graphs_analysis]]，按"历史不回写"惯例降级为反引号）的互指。原 §2 删除。
- **§5-§6 PatternExpr/PatternMatcherPass**：`CallFunction`/`Arg`/`KeywordArg`/`Ignored`/
  `MultiOutputPattern` 语义表、候选桶注册与逆序匹配、三类 Entry（`LoweringPatternEntry`/
  `GraphPatternEntry`/`ReplacementPatternEntry`）、pre/joint/post 阶段收尾差异，均已存在于
  [[22_pattern_expression_and_matcher_engine_analysis]] 且粒度更细（如该页额外覆盖了旧页没有的
  serialized/precompiled pattern 缓存机制）。原 §5-§6 删除，无独有内容需要迁移。
- **§7-§8 DCE、拓扑与保序**：`Graph.eliminate_dead_code()` 判定条件、`Node.is_impure()`
  规则、DCE 安全前提、`stable_topological_sort` 的 `pending/ready/waiting/cursor` 状态机，
  均已存在于 [[23_dead_code_topology_and_effect_order_analysis]]（该页 §7 的算法级走读甚至更完整）。
  原 §7-§8 删除。
- **§9 复杂度分析（通用表 + §9.1 pattern 总体复杂度）**：已被 [[10_fx_graph_core_data_model_analysis]]
  §11、[[22_pattern_expression_and_matcher_engine_analysis]] §15（含旧页没有的 mutation-region 扫描
  成本 `M(v)`、非 `call_function` root 扫描成本 `H`、sort-key 比较成本 `Lcmp` 等更严格的界）、
  [[23_dead_code_topology_and_effect_order_analysis]] §11 分别覆盖，且新页复杂度分析普遍更严格
  （旧页多处用未加限定的 `O(N+E)`，新页显式标注仅在 arity/sort-key 有界时才成立）。§9.2
  （AOT 全链路复杂度，含 min-cut max-flow 项）核实已存在于
  `19_torch_compile_end_to_end/09_aotautograd_joint_forward_backward_graphs` §14 与
  `10_saved_tensors_recompute_and_runtime_abi` §14 各自的复杂度节。原 §9（含 9.1/9.2）删除。
- **§10 不变量与排查清单**：§10.1 八条通用不变量已被 [[21_fx_graph_editing_primitives_and_invariants_analysis]]
  §14（10 项检查清单）覆盖；其中两条 AOT 特有提醒（"不把 bw 叫反向边图"、"不把 saved tensor
  理解为跨图 Node 引用"）保留在本页新 §10「残留提醒」。§10.2「阅读图时的四问」判定为跨
  FX 数据模型/pattern/AOT 三个语境的综合辨析工具，不属于单一目的地页，整节保留。
- **§11 关键源码导航**：通用行（Node/Graph/lint/DCE/Proxy tracing/pattern 相关）已核实存在于
  对应课程页各自的源码路径引用中；AOT 特有行（joint capture/partitioners/runtime_wrappers）
  与本页保留的 §3/§4 正文内联引用重复，压缩为一张 4 行 mini 表，标题改「AOT 特有部分」。

**属 AOTAutograd 特有、本组不处理**：§1 中 2/7 条（AOTAutograd 两张独立 GraphModule、
recompute 非特殊节点类型）、§3（joint→fw/bw 构造与跨图 ABI）、§4（recompute 选择/构图/reorder，
含 min-cut partition §4.1 细节）——按计划要求原样保留，留给 Task 8 与
`09_aotautograd_joint_forward_backward_graphs`/`10_saved_tensors_recompute_and_runtime_abi`
（C09/C10）归一；本组未改动这两节正文一字。

页面净变化：601 → 269 行；标题从「FX Graph 构图与改图机制 — AOTAutograd 正反向分图、
PatternMatcher、DCE 与保序」改为「AOTAutograd Joint Graph 构造与 Recompute — 正反向分图、
Saved-Tensor ABI 与 Min-Cut Partition」以反映瘦身后范围；文件名不变（`fx_graph_construction_
and_transformation_analysis.md`，Task 8 可能进一步处置）。

**入链改指**（9 处非 changelog 入链逐条核实内容落点后精确改）：
`aotautograd_analysis.md` 描述句删去已迁移的"复杂度"表述；`02_aot_autograd/index.md`
两处表格行更新为反映瘦身范围+互指新目录；`04_inductor/index.md`「FX Passes」表格行改指
[[02_compile_stack/03_graph_ir_and_passes/index]]；`torch_upstream_pass_deepdive.md`
Related Pages 一行改指同一索引；`fx_graph_export_and_custom_ops_analysis.md` Related Pages
一行改指同一索引（AOT 部分保留旁注）；`01_fx_export_extensibility/index.md` 两处（页面列表表格行 +
Related Pages 一行）同样改指该索引，注明 AOT 分图见 `02_aot_autograd/index`。

新目录 `03_graph_ir_and_passes/index.md` 补充「与旧页的关系」一节说明本次归一决策；
`02_aot_autograd/index.md` 新增一句互指该索引。

**校验**：`python tools/check_links.py`：pages 388→388，broken=0；`pytest tools/ -q`：77 passed。

---

## 2026-07-30：知识库结构整改 P4 Task 7 Step 1（19 号 C 卷第一批迁入 03_graph_ir_and_passes，12 篇）

**Type**: Structure Reorg（设计：`docs/superpowers/plans/2026-07-30-kb-reorg-p4-ai-frameworks.md` Task 7）

**迁移**：`19_torch_compile_end_to_end/`下 C 卷 21 篇中的 12 篇（FX 数据模型/改图原语/pattern/DCE/保序部分：C02,03,05,06,07,08,11,12,13,14,15,16）`git mv` 到新建 `02_compile_stack/03_graph_ir_and_passes/`，去数字前缀规范重命名（新基名全库唯一，无冲突）；C 卷其余 9 篇（C01 动机、C04 动态形状、C09/C10 AOT joint/recompute、C17-C21 Inductor IR/Scheduler/Codegen）留 Task 8：

- `02_fx_graph_core_data_model.md` → `fx_graph_core_data_model_analysis.md`
- `03_graph_values_metadata_and_signatures.md` → `graph_values_metadata_and_signatures_analysis.md`
- `05_graph_effects_alias_mutation_and_order.md` → `graph_effects_alias_mutation_and_order_analysis.md`
- `06_structured_outputs_higher_order_and_nested_graphs.md` → `structured_outputs_higher_order_and_nested_graphs_analysis.md`
- `07_graph_capture_frontends_and_tracing.md` → `graph_capture_frontends_and_tracing_analysis.md`
- `08_graph_normalization_decomposition_and_functionalization.md` → `graph_normalization_decomposition_and_functionalization_analysis.md`
- `11_graph_stage_boundaries_identity_and_provenance.md` → `graph_stage_boundaries_identity_and_provenance_analysis.md`
- `12_fx_graph_editing_primitives_and_invariants.md` → `fx_graph_editing_primitives_and_invariants_analysis.md`
- `13_pattern_expression_and_matcher_engine.md` → `pattern_expression_and_matcher_engine_analysis.md`
- `14_dead_code_topology_and_effect_order.md` → `dead_code_topology_and_effect_order_analysis.md`
- `15_graph_pass_pipeline_ordering_and_fixpoint.md` → `graph_pass_pipeline_ordering_and_fixpoint_analysis.md`
- `16_graph_rewrite_legality_validation_and_complexity.md` → `graph_rewrite_legality_validation_and_complexity_analysis.md`

**入链修复**：42 个文件、179 处 `[[...]]` 目标随重命名替换（含裸基名与 `19_torch_compile_end_to_end/`-限定两种旧写法，后者一并去路径前缀改裸基名）；两个 00 系列索引（`00_pytorch_graph_series_index.md`、`00_torch_compile_end_to_end_index.md`）按 Task 6（D 卷）先例逐行替换链接目标，不做 B/E 式整段压缩（因 C 卷本次只分发一半，Part IV 与 C01/04/09/10 仍在原目录）；changelog.md 内唯一一处历史提及（反引号引用，非 `[[...]]` 活链接）按惯例不回写。

**Labs 同步**：`demo_manifest.json` 12 条 C 卷 `page` 字段（c02/c03/c05/c06/c07/c08/c11-c16）同步改名；`test_volume_demo_contract.py` 的 `_page_root()` 新增 `_C_PAGE_ROOTS` 字典分支（C 卷仅部分迁移，未列出的 c-id 仍回退旧目录，与 D 卷四散模式同构）；`CourseMarkdownContractTest._course_pages()` 同步纳入 `03_graph_ir_and_passes/*.md`。

新目录 `03_graph_ir_and_passes/index.md` 建成实质内容（替换占位）；`02_compile_stack/index.md`、`01_ai_frameworks` 相关导航行同步更新。

**校验**：`python tools/check_links.py`：pages 388→388（纯移动不改变页数），broken=0；`pytest tools/ -q`：77 passed。

---

## 2026-07-30：知识库结构整改 P4 Task 6（19 号 D 卷分发,7 篇 1888 行）

**Type**: Redundancy Consolidation + Physical Move（设计：`docs/superpowers/plans/2026-07-30-kb-reorg-p4-ai-frameworks.md` Task 6）

逐篇处置(每篇/每组一 commit,均去 `dNN_`/`f08_` 前缀):

- **D01**(335 行)→`02_compile_stack/04_inductor/inductor_compile_fx_orchestration_analysis.md`。与`inductor_compiler_pipeline_analysis.md`(921 行,原索引"脊柱文档")逐节判重:该页 §1-§7 逐阶段走读(Dynamo/AOTAutograd/Decomposition/FX Passes/Lowering/Scheduler/CodeGen)已被 04_inductor 目录各阶段专题页(`pre_grad_passes_guide`/`joint_graph_passes_guide`/`post_grad_passes_guide`/`decomposition_passes_guide`/`lowering_analysis`/`scheduler_analysis`/`inductor_codegen_analysis`)、01_dynamo、02_aot_autograd 目录各专题页更深入地覆盖(逐关键词核实,深度均超过原页对应节),判定为冗余,不强并;§0 全景图与§8 设计哲学/§9 文件速查表判定为独有综合价值,吸收为 D01 新增 §0(mermaid 全景图 + 各阶段深挖入口导航表)与 §15/§16。921 页删除,19 条入链逐条改指(9 条改指 D01 并按目标页语境调整描述文字,6 条因链接方本身就是该阶段权威页而判定自引用冗余后直接删除,2 条改指更贴切的具体目标——`decomposition_passes_guide`/`wrapper_execution_memory_allocation_and_reuse_analysis`,2 条改指`02_compile_stack/04_inductor/index`导航)。
- **D02**(315 行)→`02_compile_stack/02_aot_autograd/aot_runtime_wrappers_and_lazy_backward_compile_analysis.md`。纯平移,无判重对象(spec 缺口补齐)。
- **D03**(211 行)→`02_compile_stack/04_inductor/async_compile_workers_and_module_loading_analysis.md`。纯平移。
- **D04**(219 行)→内容改写为`02_compile_stack/06_compile_cache/index.md`的 overview 主体:D04 的 14 节源码级分析(七层 cache 对照表、FXGraphCache key 构造、guard-vs-key 双层策略、序列化边界、AOTAutograd cache 在 FXGraphCache 之上、失效非广播事件、不变量、复杂度、常见误解)置于 index 顶部作 §1-14,原 91 行导航(四层教学表、专题页表、生命周期阅读顺序、课程边界、审计边界)完整保留于后作 §15-19(未删减,含 PGO 行等 D04 未覆盖的互补内容)。D04 文件删除,12 条入链改指`02_compile_stack/06_compile_cache/index`(路径限定,因"index"歧义)。
- **D05**(224 行)→`02_compile_stack/04_inductor/wrapper_execution_memory_allocation_and_reuse_analysis.md`。纯平移;不做内存归一(留 Task 8 与 C19 一并),页头加互指说明,并与`inductor_memory_management_analysis`/`inductor_memory_allocation_guide`加双向回链。
- **D06**(320 行)+**F08**(322 行)→`03_runtime_graphs/cuda/`(`cudagraph_trees_warmup_record_and_replay_analysis.md`/`training_inference_cudagraph_and_freezing_analysis.md`)。与`PyTorch_CUDA_Graphs_Complete_Guide.md`"方式2"/"综合比较"节判重:该节内容为伪代码级使用示例与四种用法对比表,未引用`cudagraph_trees.py`/`freezing.py`任何一行;D06/F08 是源码行级机制分析(`CUDAGraphNode`/`CUDAGraphTreeManager`状态机、`freezing.py`变换链与所有权后果),独有内容占比 >>50%,按判定标准保留为专题页,双向加互指(Guide 方式2/综合比较节 + Related Pages 新增两条,`cuda/index.md`新增两行)。
- **D07**(264 行)→`02_compile_stack/07_debugging/compiled_artifact_lifecycle_and_runtime_failures_analysis.md`。纯平移;`07_debugging/index.md`补为第 10 篇(原"九篇"改"十篇"),插入 aotautograd 失败定位之后、minifier/bisector 之前(生命周期状态机为失败定位提供时序坐标),去除因迁入而冗余的旧"本卷前置卷 D"Related Pages 行。

**labs 同步**:`demo_manifest.json`7 条 D 卷 page 字段(d01-d07)去前缀更新,D04 字段改为`index.md`(因内容并入索引页);`test_volume_demo_contract.py`的`_page_root`新增`_D_PAGE_ROOTS`分支(D 卷四散至 04_inductor/02_aot_autograd/06_compile_cache/03_runtime_graphs::cuda/07_debugging 四个目录,不同于 B/E 整卷单目录迁移,故改按`page_id`路由)与 F08 特判;`CourseMarkdownContractTest.test_call_chain_pages_have_source_walkthroughs`的`target_pages`同步更新三个已迁移页面的根目录。

**校验**:`python tools/check_links.py`:pages 390→388(921 与 D04 两页删除,净 -2),broken=0;`pytest tools/ -q`:77 passed(labs 同步后恢复,过程中三个 manifest/call-chain 测试短暂失败,已在本任务收尾前修复)。

**追记(复核修正,2026-07-30,不改上方历史行)**:

(a) 上文 D01 条"2 条改指更贴切的具体目标——`decomposition_passes_guide`/`wrapper_execution_memory_allocation_and_reuse_analysis`"经复核有误:对照 d8c1a5b 实际 diff,该 commit 只改了 `decomposition_passes_guide.md` 一处(旧链 `[[inductor_compiler_pipeline_analysis]]` → `[[15_inductor_compile_fx_orchestration_analysis]]` — compile_fx 把 decomposition table 传给 AOTAutograd 的调用点);`wrapper_execution_memory_allocation_and_reuse_analysis.md` 在 d8c1a5b 中未被触碰(该页由更早的 01c1422 移入本目录,与本次改指无关)。实际应为"1 条改指更贴切的具体目标"。

(b) "19 条入链"为**页数**口径(`git grep -l`,d8c1a5b^ 下命中 19 个非 changelog 文件);若按**出现次数**(`git grep -o '\[\[inductor_compiler_pipeline_analysis'`,同一基线,排除 changelog.md 本身)计,实际为 **30** 次。两个数字口径不同,均属实,原文未注明口径,此处补注。

(c) "纯平移四篇"(D02/D03/D05/D07,各条独立标注"纯平移")中,**D05 并非字节对字节平移**:对照 01c1422 实际 diff,`wrapper_execution_memory_allocation_and_reuse_analysis.md` 净 +5 行(6 insertions/1 deletion)——除去 `d05_` 前缀迁移外,页头新增了一段与 `inductor_memory_management_analysis`/`inductor_memory_allocation_guide` 的互指说明(该 commit 消息本身已披露"页头加互指说明",但顶部"纯平移"表述未与此区分,此处复核标注更精确)。D02/D03/D07 复核 diffstat 确认为无额外插入的纯移动。

(d) 本次(kb-reorg P4 Task 6 复核修复)在 D01 判重时被跳过、未落地的 3 处原文事实,已按"逐字为基底+溯源"原则回补:`inductor_compiler_pipeline_analysis.md` §7.4(CPU CodeGen:`CppKernel`/`CppVecKernel`/`CppTile2DKernel` 类体系与向量化判定)→ [[20_inductor_codegen_analysis]] 新增 §7;§7.5(`ChoiceCaller`/`TritonTemplateCaller` 编译期算法选择基础设施、`TuningProcessPool` 子进程隔离)→ [[21_inductor_autotuning_analysis]] 新增 §六,并恢复一条指向 [[15_inductor_compile_fx_orchestration_analysis]] 的 Related Pages 链接(替代被删的旧链);§4.3.7 `dedup_reduce_scatters`(reduce_scatter 线性可加性融合)→ [[32_post_grad_passes_guide]] Pass 19-21 节补齐第 4 个成员。三处原文均无固定源码基线声明,回补时对照本地 pinned checkout(`E:/97-codes/torch_parallel/p`,`e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`)核实所有行号引用,发现原文行号普遍漂移(如 `ir.py:5582` 实为 `ChoiceCaller` 所在 `ir.py:6185` 与 `TritonTemplateCaller` 所在 `select_algorithm.py:3347` 两处的合并粗写),均以 `[!correction]` 就地标注,不改写原文陈述本身。校验:`python tools/check_links.py` pages 388→388(纯编辑,broken=0);`pytest tools/ -q` 77 passed。

---

## 2026-07-30：知识库结构整改 P4 Task 5 spec 审查修复（bytecode_analysis 两算法补落点 + 加速数字弃置例外可见化）

**Type**: Redundancy Consolidation 修复（spec 审查发现 2 项：824c5a2 抽验通过、989080c 修剪无删失，另 2 项需修）

**修复 1(实质缺口)**：`git show d0b998f:...PyTorch_Dynamo_Technical_Analysis.md` §3.2「字节码分析技术」的 `remove_dead_code()`(活代码可达性,跟随跳转目标与异常表条目)与 `stacksize_analysis()`(≤100 轮定点迭代)两个算法此前全库无落点——97d19ce 的判重台账遗漏了这两个具体函数(只覆盖了架构层面的"死代码消除/栈大小分析"存在性,未展开算法机制)。按 B04 §14 吸收 a03 `bytecode_transformation.py` 内容的同款模式,在 [[12_instruction_translator_and_bytecode_state_machine_analysis]] 新增 §15,以原文算法主张为基底,对照本地 pinned 源码(`E:/97-codes/torch_parallel/p`,`e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`)的 `torch/_dynamo/bytecode_analysis.py` 补全精确行号引用（原文无定位符,引用补全属允许的增强,非新造断言）：

- §15.1 `remove_dead_code()`(`torch/_dynamo/bytecode_analysis.py:69-125`)：`find_live_code`(`:74-89`)可达性遍历,含跳转目标空值断言(`:82-87`,原文简化版缺失)；Python 3.11+ 异常表 start/end 回填(`:93-123`,原文完全未覆盖)；调用点 `convert_frame.py:979`。
- §15.2 `stacksize_analysis()`(`:249-291`)：`StackSize`区间(`:224-247`)+ 三类传播边——顺序执行(`:266-270`)、跳转(`:271-276`)、异常表(`:277-281`,原文未覆盖)；`fixed_point`收敛判据(`:260-262`)与失败断言(`:286-289`,原文未覆盖)；结果写回 `co_stacksize`(`:290`，`bytecode_transformation.py:1875`)。
- [!correction] 原文简化伪代码的定点循环声明 `changed` 却从未赋值为 `True`，若照抄会导致收敛判断恒假、循环跑满 100 轮不提前退出；订正为当前源码的 `FixedPointBox` 判据，不迁移原文这段有缺陷的伪代码。
- §9"删除死bytecode和无意义跳转"补一句指向 §15 的前向指针。

**修复 2(例外可见化)**：97d19ce 判重台账里"§6.1 加速数字无源、与 e07 论点抵触、不落地"的例外此前只记录在 changelog，页面本身不可见。在 [[17_compile_latency_cache_and_steady_state_performance_analysis]] §1（"平均耗时没有诊断价值"论点处）补一句注，明示该旧页给出的"2-3x/4x/1.5-2x"数字已按记录弃置、未作 `[!todo]` 保留。

**校验**：`python tools/check_links.py`：pages 390→390（纯编辑）,broken=0；`pytest tools/ -q`：77 passed（新增 §15 的 mermaid/list-marker/locator-length 质量门禁通过，所有引用跨度 ≤57 行，未触发 100 行上限）。

---

## 2026-07-30：知识库结构整改 P4 Task 5 Step 4-5（control_flow_capture_analysis 判重结论 + dynamo_pass_methodology 与 B10 互链）

**Type**: Redundancy Consolidation（设计：`docs/superpowers/plans/2026-07-30-kb-reorg-p4-ai-frameworks.md` Task 5）

**Step 4 判重结论**：`control_flow_capture_analysis.md`(204 行,已在 01_dynamo/)vs B04(`instruction_translator_and_bytecode_state_machine_analysis`)——关键词核查（`generic_jump`/`FOR_ITER`/`speculate_subgraph`/`higher_order`/`torch.cond`/`CondHigherOrderVariable`/`POP_JUMP_IF`/`evaluate_expr`/`guard_bool`/`install_subgraph` 在 B04 全部零命中）确认**零重叠**：B04 讲字节码解释器的通用状态机（`run→step→dispatch`，以 `CALL` 为例），本页讲控制流两条路径的专题机制（HOP `speculate_subgraph`/`torch.cond` 深挖、原生 `if/for` 的 `generic_jump`/`FOR_ITER`），两页视角完全互补。按任务指示**不动本页内容**（Step 2 已做的入链机械修复保留）。与 C06 的判重按计划原文注明留 Task 7 收尾。

**Step 5 判重结论**：`dynamo_pass_methodology.md`(152 行,已在 01_dynamo/)vs B10(`backend_contract_and_custom_backend_analysis`)——两页主题重叠（都在讲 Dynamo backend 扩展）但**层次不同**：B10 是契约机制的源码级深挖（registry/gm 形态/example inputs/返回值语义/mode-options 传递/可选 context 接口/failure 分层，全部带 `file:line`），本页是"要不要在 Dynamo 做这件事"的开发决策指南（适合/不适合决策表、可运行注册代码示例、改图排错流程、离开 Dynamo 的路由表），两者是 CLAUDE.md Page Types 里 Entity(deep dive) vs Guide 的正常搭配,不构成重复共存。**收窄重叠表述**：§1 删除与 B10 §1 重复的契约机制复述,改为精简结论 + 指向 B10;§4 API 速查表后加指针注明源码级实现见 B10 §2/§7;§6 改图规则清单后加指针注明完整正确性清单见 B10 §13、IR 方言边界见 B10 §10。**新增互链**：两页此前互相都没有链接对方（各自 Related Pages 检索确认零命中）——本任务修复,双向加回链。

**校验**：`python tools/check_links.py`：pages 390→390（零删除、纯编辑）,broken=0；`pytest tools/ -q`：77 passed。

---

## 2026-07-30：知识库结构整改 P4 Task 5 Step 3（删除 torch_compile_source_analysis.md，593 行，独有内容并入 B01/B02）

**Type**: Redundancy Consolidation（设计：`docs/superpowers/plans/2026-07-30-kb-reorg-p4-ai-frameworks.md` Task 5）

**判重**：`04_inductor/torch_compile_source_analysis.md`（593 行，`torch.compile()` 函数体逐段源码解析）与 B01/B02 通读比对——**与 A 卷/PyTorch_Dynamo_Technical_Analysis 不同,本页确有实质独有内容**（关键词 grep 核实 `3.15`/`GIL_DISABLED`/`sysconfig`/`_log_api_usage_once`/`CompilerBisector`/`guard_filter_fn`/`use_aoti`/`is_exporting`/`_in_hop_compile`/`_TorchCompileAOTInductorWrapper`/`apply_mode`/`apply_options`/`list_mode_options` 在 B01/B02 及全库均无匹配或仅部分匹配），逐字迁入并对照本地 pinned pytorch checkout（`E:/97-codes/torch_parallel/p`）核验行号：

- §4.1 Python 版本兼容性检查（3.15+ 拒绝、free-threaded GIL<3.13.3 拒绝、`_log_api_usage_once` 遥测）→ [[10_torch_compile_api_and_first_call_lifecycle_analysis]] 新增 §13.1，核验定位 `torch/__init__.py:3267-3280`
- §4.6 `torch.export` 兼容性（`is_exporting()`/`_in_hop_compile()` 短路为 no-op）→ 同页新增 §13.2，核验定位 `torch/__init__.py:3350-3359`
- §4.4 CompilerBisector 二分调试的 API 入口钩子（`bisect_backend := CompilerBisector.get_backend()` 覆盖 backend + vLLM 自定义 backend 保护条件）→ [[22_backend_modes_options_stances_and_fullgraph_analysis]] 新增 §13，核验定位 `torch/__init__.py:3330-3342`；与 [[15_minifier_repro_and_compiler_bisector_analysis]] §7（bisector 内部二分算法）互补,双向加回链
- §4.5 特殊选项提取 + §4.7/§5 三个 wrapper 类的方法级实现（`_TorchCompileInductorWrapper.apply_mode/apply_options/get_compiler_config/reset`、CUDA<12.6 CUPTI workaround、`_TorchCompileAOTInductorWrapper` 子类的 `cpp_wrapper`/`aot_inductor.package`/`V.set_aot_compilation`、`_TorchCompileWrapper` 的 `lookup_backend`+kwargs 透传）→ 同页新增 §14，核验定位 `torch/__init__.py:2907-3096` 区间多段
- [!todo] §14.2 迁移时发现:`use_aoti=True` 是从 `torch.compile()` **JIT 入口**直接触发的 AOTInductor 打包路径,而 `f07_aotinductor_packaging_and_deployment_analysis`（2026-07-30 起随 kb-reorg P4 Task 9 迁入并改名为 [[28_aotinductor_packaging_and_deployment_analysis]]，本条历史记载不回写活链接）§2-§3 记录的公开入口 `aoti_compile_and_package` 明确要求 `ExportedProgram`(export 驱动、部署前离线完成)。两条路径是否共享下游产物、`use_aoti` 这条 JIT 捷径的运维定位,F07 尚未覆盖——本任务范围内未展开核实,双向加回链留待后续处理。**已于 kb-reorg P4 Task 9(2026-07-30)核实解答**：两者在 `compile_fx`/`CompiledAOTI` 层汇合于同一段代码，但 `CompiledAOTI` 装载成可调用 runner 这一步受 `enable_autograd_for_aot` 门控、并不对称（只有 `use_aoti` 路径强制打开），差异还在前端捕获来源（Dynamo 运行时捕获 vs `ExportedProgram`）与是否额外 `package_aoti` 打包成 `.pt2`；详见 [[22_backend_modes_options_stances_and_fullgraph_analysis]] §14.2 note。
- §6 编译模式对照表（mode/CUDA Graphs/Triton autotune 布尔矩阵）与 B02 §3 的同一组事实（散文体）重复,不迁移；§7 能力范围与限制、§8 使用示例是已覆盖机制的摘要/演示,不迁移；§2/§3/§4.2/§4.3 是与 B01/B02 完全重复的调用栈图与参数说明,不迁移。

**删除**：`git rm` 该页。**入链修复**：`04_inductor/dynamic_shapes_full_analysis.md` 改指 [[10_torch_compile_api_and_first_call_lifecycle_analysis]]；`04_inductor/index.md` 移除本页表格行,改为一行式说明指向 [[02_compile_stack/01_dynamo/index]]。

**校验**：`python tools/check_links.py`：pages 391→390，broken=0，orphans=0；`pytest tools/ -q`：77 passed（含新增 §13/§14 的 mermaid/list-marker/locator-length 质量门禁）。

---

## 2026-07-30：知识库结构整改 P4 Task 5 Step 2（删除 PyTorch_Dynamo_Technical_Analysis.md，2018 行）

**Type**: Redundancy Consolidation（设计：`docs/superpowers/plans/2026-07-30-kb-reorg-p4-ai-frameworks.md` Task 5）

**判重**：`02_compile_stack/01_dynamo/PyTorch_Dynamo_Technical_Analysis.md`（2018 行，早期"演示代码 + 简化伪源码"风格全景页）逐节通读（§1 概述、§2 核心架构 7 阶段、§3 技术实现细节、§4 典型示例代码、§5 核心模块架构与工作流程、§6 生态角色与应用、§7 覆盖度声明、总结）后与 B 卷十篇逐一核对，**全篇无独有事实需要迁移**：

- §2/§3/§5（eval-frame 拦截、字节码分析转换、符号执行、变量跟踪、Guard 系统、FX 图构建、后端编译 7 阶段机制，含模块依赖图与调用链图）均为 B01-B10 的低精度复述——本页用"简化伪代码"且无 `file:line` 定位符，B 卷十篇逐条给出精确源码定位。抽样核实：本页 §3.1 PEP 523 C 扩展入口链描述（`dynamo_custom_eval_frame_shim → dynamo__custom_eval_frame → set_eval_frame`）缺少 [[11_eval_frame_callback_and_code_cache_analysis]] §2 已给出的更完整更准确的三态协议（`None`/`False`/callable，定位 `torch/csrc/dynamo/eval_frame.c:518-533`/`:616-638`）；本页 §7 列的"未覆盖组件"（VariableBuilder、Source 系统、高阶操作）实际均已被 [[13_variable_tracker_source_and_python_object_model_analysis]]、[[21_control_flow_capture_analysis]] 覆盖。
- §4 典型示例代码分析（简单函数/nn.Module/动态形状/graph break/循环编译走读,均为未验证的插图式伪代码）被 `tools/labs_torch_compile/demo_b_dynamo_capture.py` 的十个真实可运行 case（compile_lifecycle/backend_modes_fullgraph/eval_frame_cache/bytecode_state_machine/variable_source_guards/output_graph_side_effects/guards_recompile/graph_break_resume/dynamic_shapes/custom_backend_contract）与 B 卷各篇「源码跟读」「配套 Demo」小节取代，不重复落地插图代码。
- §6.1"性能提升"给出的"典型模型 2-3 倍/Transformer 4 倍/小模型 1.5-2 倍"加速比**无源可查**（`raw/` 下仅 `dynamo.eddx`/`torch.compile.eddx` 图示源，非可引用的实测数据）；且与 [[17_compile_latency_cache_and_steady_state_performance_analysis]] §1 的核心论点（"平均耗时没有诊断价值，必须拆分测量场景"）直接抵触，判断为不应作为事实迁移的营销式泛化断言，不落地（不同于常规"无源可疑声明转 [!todo]"处理，因为该断言的方法论本身已被后继页的论点否定，保留只会误导读者）。
- §6.5"性能优化技术"（算子融合/内存布局/循环展开/并行化）实际描述的是 Inductor 层优化,不属于 Dynamo 机制,且与 B 卷主题不符,留待后续 Inductor 相关任务处理,本任务不落地。
- §5.1 模块依赖关系图（`torch._dynamo` 包结构树）经抽样核对本地 pinned pytorch checkout（`eval_frame.py`/`config.py`/`convert_frame.py`/`bytecode_analysis.py`/`bytecode_transformation.py`/`codegen.py`/`backends/`/`variables/` 均存在）大体准确，但属于可从源码树直接重建的编排性示意图（非独有事实），且 B 卷各篇「源码阅读顺序」「源码补充」小节已提供更具体的逐机制文件路径，不单独迁移。

**删除**：`git rm` 该页。**入链修复**（18 个外部文件的活链接，均为泛指性"参见 Dynamo 帧评估/字节码/Guard"式指针，按 Task 3 a05 先例改指 [[02_compile_stack/01_dynamo/index]]）：`pytorch_dispatcher_analysis`、`npu_operator_graph_eligibility_guide`（2 处）、`op_registration_pipeline_analysis`、`aotautograd_analysis`、`dynamic_shapes_full_analysis`、`inductor_compiler_pipeline_analysis`、`inductor_memory_management_analysis`、`torch_compile_source_analysis`、`unbacked_symint_analysis`、`torch_mlir_pass_pipeline_analysis`、`vllm/index`、`vllm_compilation_cudagraph_analysis`（2 处）、`vllm_ir_and_fusion_passes_analysis`、`wiki/index`；`dynamo_pgo_cache_analysis` 按其"VariableBuilder/guard 的宿主"原描述精确改指 [[13_variable_tracker_source_and_python_object_model_analysis]] + [[15_guards_cache_lookup_and_recompilation_analysis]] 两篇。域内三篇（`control_flow_capture_analysis`、`dynamo_pass_methodology`、`dynamo_quickstart`）的 Related Pages 按原描述拆成对应的具体 B 卷页链接（如"帧评估、字节码符号执行、Guard 与重编译"拆为三条精确链接），比泛指索引更精确。`wiki/changelog.md` 里 3 处写入当时的历史活链接（2026-06-30/2026-06-12 更早条目）按"历史不回写"惯例降级为惰性反引号 + 去向说明；另 2 处（2026-07-17 前后两条）本就是反引号包裹的非活链接，未受影响。

**索引重建**：`02_compile_stack/01_dynamo/index.md` 页面列表从 4 行重建为 13 行（quickstart + B01-B10 + control_flow 专题 + dynamo_pass_methodology development guide），移除 `PyTorch_Dynamo_Technical_Analysis` 行；页头摘要与最后更新同步。

**校验**：`python tools/check_links.py`：pages 392→391，broken=0，orphans=0；`pytest tools/ -q`：77 passed。

---

## 2026-07-30：知识库结构整改 P4 Task 5 Step 1（19 号 B 卷迁入 01_dynamo，10 篇 2653 行）

**Type**: Structure Reorg（设计：`docs/superpowers/plans/2026-07-30-kb-reorg-p4-ai-frameworks.md` Task 5）

**迁移**：`19_torch_compile_end_to_end/b01-b10`（API 与首次编译生命周期、backend 参数/stance/fullgraph、eval-frame callback/code cache、字节码符号执行、VariableTracker/来源、OutputGraph/side effects、guard/cache/recompile、graph break/resume、动态形状泛化/fallback、backend contract，共 2653 行）`git mv` 到 `02_compile_stack/01_dynamo/`，去 `bNN_` 前缀规范重命名（新基名全库唯一，无冲突）：

- `b01_torch_compile_api_and_first_call_lifecycle_analysis.md` → `torch_compile_api_and_first_call_lifecycle_analysis.md`
- `b02_backend_modes_options_stances_and_fullgraph_analysis.md` → `backend_modes_options_stances_and_fullgraph_analysis.md`
- `b03_eval_frame_callback_and_code_cache_analysis.md` → `eval_frame_callback_and_code_cache_analysis.md`
- `b04_instruction_translator_and_bytecode_state_machine_analysis.md` → `instruction_translator_and_bytecode_state_machine_analysis.md`
- `b05_variable_tracker_source_and_python_object_model_analysis.md` → `variable_tracker_source_and_python_object_model_analysis.md`
- `b06_output_graph_side_effects_and_graph_emission_analysis.md` → `output_graph_side_effects_and_graph_emission_analysis.md`
- `b07_guards_cache_lookup_and_recompilation_analysis.md` → `guards_cache_lookup_and_recompilation_analysis.md`
- `b08_graph_break_resume_functions_and_partial_graphs_analysis.md` → `graph_break_resume_functions_and_partial_graphs_analysis.md`
- `b09_dynamic_shapes_generalization_and_fallback_analysis.md` → `dynamic_shapes_generalization_and_fallback_analysis.md`
- `b10_backend_contract_and_custom_backend_analysis.md` → `backend_contract_and_custom_backend_analysis.md`

**入链修复**：十篇互链（前置/后续 + Related Pages）随重命名全库替换（裸基名，新基名唯一）；`00_torch_compile_end_to_end_index.md` 卷 B 表按 Task 3/4 先例压缩为一行式指向（完整表格重建留 Task 10）；`00_pytorch_graph_series_index.md` 未引用卷 B，无需改动；`02_compile_stack/07_debugging/` 内 `dynamo_explain_and_graph_break_diagnosis_analysis.md`/`guard_failure_and_recompile_diagnosis_analysis.md`、`19_torch_compile_end_to_end/d04_compile_cache_hierarchy_keys_and_invalidation_analysis.md`/`f06_custom_backends_and_device_integration_analysis.md` 对十篇的引用改指新基名；`wiki/changelog.md` 里 Task 3 条目（本文件写入当时的历史记载，按本文件"不随后续迁移回写"惯例）中对 `b01`/`b03`/`b04` 的活链接降级为惰性反引号 + 说明新名，不当作活链接维护。

**Labs 同步**：`tools/labs_torch_compile/demo_manifest.json` 的 10 条 B 卷 `page` 字段同步改名（page_id/volume/script/case 不变，55 条不变）；`test_volume_demo_contract.py` 的 `_page_root(labs_root, volume)` 新增 B→`01_dynamo` 分支（Task 4 预留的同一模式）；`CourseMarkdownContractTest._course_pages()` 同步纳入 `01_dynamo/*.md`（含卷内既有的 quickstart/pass-methodology/control-flow/旧大文页，质量门禁全部通过）；`test_call_chain_pages_have_source_walkthroughs`/`test_b07_cache_miss_formula_contains_all_addends` 的 b04/b07 路径改指新家。

**校验**：`python tools/check_links.py`：pages 392→392（纯移动不改变页数），broken=0；`pytest tools/ -q`：77 passed。

---

## 2026-07-30：知识库结构整改 P4 Task 4（19 号 E 卷迁入 07_debugging，9 篇 2391 行 + 吸收旧 debug 页）

**Type**: Structure Reorg + Redundancy Consolidation（设计：`docs/superpowers/plans/2026-07-30-kb-reorg-p4-ai-frameworks.md` Task 4）

**迁移**：`19_torch_compile_end_to_end/e01-e09`（观测台账、Dynamo explain/graph break、guard failure/recompile、AOTAutograd/Inductor 失败分层定位、minifier/repro/bisector、正确性验证方法论、compile 延迟/cache/稳态性能、kernel/fusion/内存/硬件性能归因、生产上线/fallback/监控，共 2391 行）`git mv` 到 `02_compile_stack/07_debugging/`，去 `eNN_` 前缀规范重命名（新基名全库唯一，无冲突）：

- `e01_observability_logs_counters_and_artifact_map_analysis.md` → `observability_logs_counters_and_artifact_map_analysis.md`
- `e02_dynamo_explain_and_graph_break_diagnosis_analysis.md` → `dynamo_explain_and_graph_break_diagnosis_analysis.md`
- `e03_guard_failure_and_recompile_diagnosis_analysis.md` → `guard_failure_and_recompile_diagnosis_analysis.md`
- `e04_aotautograd_and_inductor_failure_localization_analysis.md` → `aotautograd_and_inductor_failure_localization_analysis.md`
- `e05_minifier_repro_and_compiler_bisector_analysis.md` → `minifier_repro_and_compiler_bisector_analysis.md`
- `e06_compiled_correctness_validation_methodology_analysis.md` → `compiled_correctness_validation_methodology_analysis.md`
- `e07_compile_latency_cache_and_steady_state_performance_analysis.md` → `compile_latency_cache_and_steady_state_performance_analysis.md`
- `e08_kernel_fusion_memory_and_hardware_performance_analysis.md` → `kernel_fusion_memory_and_hardware_performance_analysis.md`
- `e09_production_rollout_fallback_and_monitoring_analysis.md` → `production_rollout_fallback_and_monitoring_analysis.md`

**删除并吸收**：`02_compile_stack/04_inductor/Pytorch_Compile_Debug_Analysis.md`（558 行，E 卷的压缩前身）逐节判重后删除。机制性内容（Dynamo/FX/Guards/AOT/Inductor 各阶段原理与定位方法、通用决策树前 4 步）已被上述九篇更严谨的源码级分析取代，不重复落地；但该页提供的**可运行排查脚本**、**分布式专属决策分支**与**kernel/CUDA 层崩溃诊断**九篇均未覆盖，逐字迁入新建的 `02_compile_stack/07_debugging/index.md` 附录：`run_debug.sh`（环境变量一键启动）、`export_capture.py`/`capture_backend.py`（DIY GraphModule 捕获，与官方 `after_dynamo`/`after_aot` repro 生成器正交互补）、`collect_artifacts.sh`、`diff_rank_logs.sh`（多 rank 日志对比，九篇未提供任何等价工具）、决策树第 5 分支（仅分布式场景复现）、kernel/CUDA 崩溃关键词与修复清单（segfault/OOM/launch failed/arch mismatch，[[18_kernel_fusion_memory_and_hardware_performance_analysis]] 只覆盖性能不覆盖崩溃）、工程化使用流程、小技巧、upstream issue 附件清单（与 [[15_minifier_repro_and_compiler_bisector_analysis]] §11 的通用 repro 标准互补，不重复）。

**入链修复**：九篇互链（前置/后续 + Related Pages）随重命名全库替换（裸基名，新基名唯一）；`00_torch_compile_end_to_end_index.md` 卷 E 表按 Task 3 先例压缩为一行式指向（完整表格重建留 Task 10）；`b01/b02/b07/b08/b09/b10/d03/d05/d06/d07/f01/f03/f04/f07/f08` 对九篇的引用改指新基名；`04_inductor/index.md`、`inductor_memory_allocation_guide.md`、`inductor_quickstart.md`（3 处）、`npu/npu_debug_guide.md`、`torch_compile_architecture.md`（2 处）、`19_torch_compile_end_to_end/11_graph_stage_boundaries_identity_and_provenance.md` 对旧 debug 页的引用改指 `02_compile_stack/07_debugging/index`；changelog 历史条目（2026-07 更早的一条）按本文件「历史不回写」惯例改为惰性反引号 + 说明去向，不当作活链接维护。

**Labs 同步**：`tools/labs_torch_compile/demo_manifest.json` 的 9 条 E 卷 `page` 字段同步改名（page_id/volume/script/case 不变，55 条不变）；`test_volume_demo_contract.py` 新增 `_page_root(labs_root, volume)` 按卷解析页面目录（E→`07_debugging`，其余仍在 `19_torch_compile_end_to_end`，为 Task 5-9 逐卷迁移预留同一模式）；`CourseMarkdownContractTest._course_pages()` 同步纳入 `07_debugging/*.md`，保持 list-marker/mermaid/locator 质量门禁覆盖迁移后的九篇。

**校验**：`python tools/check_links.py`：pages 393→392（-1 为旧 debug 页删除，九篇是移动不减少），broken=0；`pytest tools/ -q`：77 passed。

---

## 2026-07-30：知识库结构整改 P4 Task 3（19 号 A 卷删除，5 篇 1790 行）

**Type**: Redundancy Consolidation + Structure Reorg（设计：`docs/superpowers/plans/2026-07-30-kb-reorg-p4-ai-frameworks.md` Task 3；P4 阶段首个编辑任务）

**删除**：`19_torch_compile_end_to_end/a01-a05`（Tensor/Storage/View、operator/schema/dispatcher/autograd、Python frame/code object/bytecode、dispatch mode/ProxyTensor/FakeTensor、eager-capture-compile-replay 成本模型，共 1790 行）。这五篇是"执行模型前置基础回顾"，判重发现约 60-70% 内容与 `01_eager_runtime` 各功能页重复，但**逐句核查后确认每篇均有编译器视角的独有分析**（不可盲信"回顾=重复"），已逐字迁入对应功能页新增小节，不是简单删除：

- a01 → [[01_eager_runtime/01_tensor_and_storage/10_tensor_impl_and_storage_analysis]] §13（differentiable view/DifferentiableViewMeta、mutation 状态机、复杂度记账、常见误解、view→Autograd→编译器的源码跟读）
- a02 → [[10_pytorch_dispatcher_analysis]] §12（ADInplaceOrView 分层、mutation 算子 rebase 样本、Dispatcher/Autograd Edge/FX data edge 三种"边"辨析）
- a03 → `b04_instruction_translator_and_bytecode_state_machine_analysis`（P4 Task 5 起更名为 [[12_instruction_translator_and_bytecode_state_machine_analysis]]） §14 + `b03_eval_frame_callback_and_code_cache_analysis`（P4 Task 5 起更名为 [[11_eval_frame_callback_and_code_cache_analysis]]） §13（`bytecode_transformation` 重组子系统、code object/frame/instruction 定义表、C-hook 与 ConvertFrame 边界）
- a04 → `aotautograd_analysis` §13（P4 Task 8 起独立成页 [[10_dispatch_modes_proxytensor_faketensor_analysis]]）（`__torch_function__`/`__torch_dispatch__`/ProxyTensor/FakeTensor 四层分工、`track_tensor_tree`、FakeTensorMode 状态、decomposition 落点、数据相关 operator 边界）
- a05 → `b01_torch_compile_api_and_first_call_lifecycle_analysis`（P4 Task 5 起更名为 [[10_torch_compile_api_and_first_call_lifecycle_analysis]]） §12 + [[17_compile_latency_cache_and_steady_state_performance_analysis]] §12-§16（七阶段成本词汇表、cache-entry 查找与 backend handoff 源码补充；参数化成本模型/break-even/四层 cache 对照表与 e07 既有的测量方法论合并，不重复落地两处）

**入链修复**：`f05`（a02/a04 引用改指 pytorch_dispatcher_analysis / aotautograd_analysis）、`b01`/`e07`/`d06`（a05 引用改指内容实际落点）；卷内 a01-a05 互链随整卷删除一并消失；`00_torch_compile_end_to_end_index.md` 的"卷 A"表按 Task 3 约定不做整体重排（留给 Task 10），仅去除失效行并加一行去向说明，避免 broken>0。

**Labs 同步**：`tools/labs_torch_compile/demo_manifest.json` 移除 a01-a05 五条 page 映射（60→55 条）；`test_volume_demo_contract.py` 同步更新 `expected_ids`/条目计数与 `test_call_chain_pages_have_source_walkthroughs` 的 target_pages；`demo_a_execution_model.py` 脚本与其两个 `VolumeABContractTest` 保留（脚本级机制验证，不依赖已删除的课程页）。

**校验**：迁移前逐条 grep+读段落核实独有性（而非凭印象判重）；迁移时对照本地 pinned PyTorch(`9922478`) 抽样核验源码定位符，发现 a05 一处 `output_graph.py` 引用行号漂移（原 A 卷标注的 `e8f97c1a...` 版本与当前 pinned 版本不一致），已重新定位到 `call_user_compiler`/`_call_user_compiler`/`BackendCompilerFailed` 的准确行号后再落地。`python tools/check_links.py`：pages 398→393，broken=0，orphans=0（基线持平）；`pytest tools/ -q`：77 passed。

## 2026-07-30：知识库结构整改 P3 完成（runtime_graphs 去重收官）

**Type**: Structure Reorg（设计：`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md` §3.3；P0-P7 的第四段）

- **06_graphs 目录 14 页 → 10 页**：删 cuda/README、npu/README（overview 三写归一）、CUDA_Graphs_Timing_Diagrams（时序图内联进 Complete_Guide 并以 mermaid 替换其 ASCII 重复图）、npugraphs_memory_reuse_analysis（Graph Tree 双写并入 torch_compile_npugraphs_deep_dive）；comparison 633→145 行只留真差异表。
- **每次合并均经逐句对抗审查**：三轮回补（8 组机制事实、6 组分析事实、硬件约束转 [!todo]）、一处推断性时序事件删除、一处"理论→实测"措辞漂移回退——详见本页下方各 Task 条目与 git 历史。
- CLAUDE.md「Never delete」条款同步为「Merge over coexist」（完整修订仍在 P7）。
- 全程 broken=0；本域净删约 2900 行重复内容。

## 2026-07-30：知识库结构整改 P3 Task 4(NPU Graph Tree 双写合并)

**Type**: Redundancy Consolidation(设计:`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md`;P3 阶段最大编辑)

**合并**：`06_graphs/npu/npugraphs_memory_reuse_analysis.md`(1698 行)并入 `torch_compile_npugraphs_deep_dive.md`(2397→2714 行,净增 317)。两页讲同一套 NPU Graph Tree 机制,§三「NPU Graph Tree 核心机制」与被并页的「Graph Tree 机制」「内存复用策略」「@torch.compile 场景案例」约 800 行重叠;被并页内「关键代码解析(合并自 memory_management)」节为上次合并未合净的残留,一并处理。

**逐节处置**（被并页每个 `##`/`###` 节）：
- `目录`/`概述` — 纯框架文字,丢弃(信息已含于主干开篇)。
- `核心架构组件`(NPUGraph/NPUCachingAllocator/NPUGraphTreeManager/NPUGraphNode 四个 C++/Python 结构) — NPUGraph、PrivatePool 为独有,搬入新增 §3.5.1-3.5.2(表格化,无损);Manager/Node 类结构与主干 §2.5.1、§3.2 重叠,丢弃。
- `内存池管理`(MempoolId_t/生命周期/PrivatePool) — 独有,搬入 §3.5.1(含 capture_begin 池注册代码节选、Capture vs Replay 差异表、replay 可跨流回放的事实)。
- `Graph Tree 机制`(核心概念/树节点结构/路径管理与状态切换) — 核心概念的"树形峰值内存=max 而非 sum"公式独有,搬入 §3.1 补充段;树节点结构与主干重复,丢弃;路径管理与状态切换的 `_run`/`apply_checkpoint` 源码与主干 §3.2.2-3.2.3 重叠但含更细粒度行号引用,细粒度引用被 §3.6 新流程图吸收,原始代码块丢弃。
- `Capture 与 Replay 流程` — capture_begin/replay 完整 ACL 样板代码与主干 §2.7 概念重叠,丢弃;差异表与"可跨流回放"独有事实已搬入 §3.5.1。
- `内存复用策略`(Liveness/StorageWeakRefWrapper/Alias Detection) — 全部独有,搬入新增 §3.7.1-3.7.3(无损)。
- `详细案例分析:@torch.compile 场景` — 旗舰案例,与主干既有的"训练循环 A→B→C"场景不同(本例为 graph-break 分支场景);代码示例+完整时序图+效率对比搬入新增 §3.8(精简,舍弃与其自身内部重复的 4 张辅助图:Liveness T0-T3、Checkpoint/Restore 图解、Phase1-4 图解、总内存变化图,因其事实已被时序图/§3.6/§3.7 覆盖);两张拓扑图(代码执行流程图、Graph Tree 节点关系)舍弃,拓扑事实由时序图 Note 与案例文字保留。
- `关键源码解析`(Graph Tree 创建流程/add_function 流程/NPUGraphNode 内存管理) — 前两者与主干 §2.6、§3.2.2 逐字重复,丢弃;第三者的 `_record`/`run` 与主干 §2.7-2.8 重复丢弃,`__init__` 中父子 liveness 对比片段独有,搬入 §3.7.1。
- `内存复用可视化总结` — 与时序图/效率对比重复,除"内存复用效率对比"(独有的定量对比,搬入 §3.8)外丢弃。
- `关键代码解析(合并自 memory_management)`(TreeManager 生命周期/分配器检查点结构/Warmup 与静态输入/静态输入优化) — 全部独有:TreeManagerContainer 生命周期搬入新增 §3.4;BlockState/SegmentState/PrivatePoolState 搬入 §3.5.3;NPUWarmupNode.run 与主干 §3.2 warmup 概念重叠部分丢弃,`_use_npu_memory_pool_manager` 事实已隐含于既有描述;静态输入优化(`npugraph_managed_idxs` 等)搬入 §3.7.4。「使用建议补充」为通用泛泛建议,丢弃。
- `总结`(核心机制回顾/三分类/触发条件/最佳实践/待探索问题) — 前三者与新搬入内容重复,丢弃;最佳实践中"减少 graph break/稳定输入/避免频繁路径切换"三点折入 §3.8 结尾一句;`待探索问题`(stale_storages 恒空、clear_path_state 为空操作)独有,搬入 §3.6。
- `参考文档` — 源码文件列表并入主干 §六"核心实现分布在"(新增 NPUGraph.h/cpp、NPUCachingAllocator.cpp 两行);外部链接与主干无实质差异,丢弃。

**主干新增结构**：§三新增 3.4 TreeManagerContainer 单例生命周期、3.5 C++ 层数据结构(NPUGraph/PrivatePool/Checkpoint 快照)、3.6 Checkpoint 恢复流程与内存复用三分类、3.7 内存复用策略(Liveness/弱引用/别名检测)、3.8 案例分析(graph-break 分支完整生命周期);3.1 补充内存优化公式;3.2.3 加一句指向 §3.6 的前向引用。全部新增 mermaid(1 flowchart + 1 sequenceDiagram)经 `@mermaid-js/mermaid-cli` 实渲验证。

**收尾**：5 处入站 wikilink 改指(`12_activation_checkpointing_analysis`/`aclgraph_deep_analysis`/`aclgraph_multistream_rng_analysis`/`npugraphs_make_graphed_callables_deep_dive`/主干自身的 Related Pages 自链接一并清除);`06_graphs/npu/index.md` 删行+承接说明+日期 bump 至 2026-07-30。体量:两页合计 1698+317=2015 行变动,净删 1698-317=1381 行(≥1200 目标);主干净增 317 行(≤500 目标)。链接检查 broken 0→0、orphans 1→1(与本次改动无关的既有孤儿页)。

**追记(质量审查回补)**：上表 2714 行为本条目写入当时的统计;后续质量审查中对本次合并做了三处小修正回补(§3.6 措辞澄清、Related Pages 去重等),回补后终值为 **2809 行**(非 2714)。审查过程中还发现 §3.8 案例分析里有一处推断性事件描述并非源码可验证事实,已就地删除——记此一笔作为本次合并的自纠记录。

---

## 2026-07-30：知识库结构整改 P3 Task 5(deep_dive 内两处收缩 + 三项遗留修正)

**Type**: Redundancy Consolidation(设计:`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md`;P3 阶段收尾编辑)

**A. §四「与 make_graphed_callables 的对比」收缩**：`torch_compile_npugraphs_deep_dive.md` §四(90 行)只留 4.1 功能对比表 + 到 [[20_npugraphs_make_graphed_callables_deepdive]] 的链接(90→18 行,净删 72)。逐段核对：4.2「实现对比」两幅 ASCII 流程图——`make_graphed_callables` 六步图被承接页「二、完整实现流程(六阶段)」(490-716 行)以远更详细的代码粒度完整覆盖，`torch.compile(backend="npugraphs")` 六步图被本文档自身 §1.3 流程图 + §2.3-2.8 完整覆盖(均原地保留未删)，故两图均丢弃不搬运；4.3「执行时序对比」sequenceDiagram 被本文档自身 §3.2.5(A→B→C 完整执行场景，含 generation/warmup/record/execute 状态转换)以更细粒度完整覆盖，同样丢弃不搬运。核对结论：§四无独有内容需要搬运，承接页 `npugraphs_make_graphed_callables_deep_dive.md` 本次未变动(668→668)。

**B. 附录 A「mode="reduce-overhead" 完整编译流程与双路径对比」收缩**：整改决策(spec §3.3)以 `aclgraph_deep_analysis.md` 为 reduce-overhead 捕获路径权威页。附录 A(759 行)逐段核对：
- 一、mode 参数澄清(1.1 mode→config 表、1.2 mode/backend 关系) — 承接页原无 mode 参数说明,独有,搬入承接页新增 §1.5「mode 参数与两条路径的触发关系」。
- 二、整体架构对比(2.1/2.2 双路径 ASCII+mermaid 架构图) — 路径 B 部分与本文档自身 §1-2 完全重叠丢弃;路径 A 部分(Wrapper→compile_fx→AOT→Codegen→cudagraph_post_compile)的阶段划分与源码定位独有,浓缩为 §4.4 的 Phase 表(不逐字保留原 ASCII/mermaid 图,信息已被表格等价表达,避免图文双份冗余)。
- 三、路径 A 完整调用链路(3.1-3.7,各 Phase 完整代码清单) — 逐 Phase 源码位置(文件+行号)独有,浓缩进 §4.4 Phase 表;3.6 npugraphify monkey-patch 入口代码与承接页 §二·差异1 现有代码逐字重复,丢弃、原地加引用。
- 四、路径 A 完整调用栈 / 五、路径 B 调用栈(对比参考) — 与三、八两节内容重复(调用链路的另一种呈现),丢弃。
- 六、关键代码路径差异分析(6.1 编译产物对比/6.2 录制内容差异/6.3 回退差异) — 独有,浓缩进 §4.4 对比表(录制内容/NPU 利用率/编译耗时/回退行为/典型定位五维)。
- 七、Wrapper 类对比(7.1/7.2) — 独有,并入新增 §1.5。
- 八、执行时序对比(8.1/8.2 两幅 sequenceDiagram) — 与三、四节的 Phase 文字描述重复(可视化重述而非新事实),且承接页已有 §1.3 通用捕获/回放时序图,丢弃不搬运。
- 九、性能特性对比(9.1 综合对比表/9.2 适用场景) — 独有,浓缩进 §4.4 对比表 + 选型清单。
- 十、max-autotune 额外优化 — 独有,并入 §1.5 mode 表备注。
- 十一、对比总结(11.1 一句话/11.2 决策流程图/11.3 代码示例/11.4 文件索引) — 11.1/11.2 改写为 §4.4 结尾的一句话总结+选型清单;11.3 代码示例与本文档自身 §5.2 最佳实践重复,丢弃;11.4 文件索引表独有,搬入 §4.4。

附录 A 本体替换为一段摘要(路径 A/B 核心差异一段话 + 链接到 [[10_aclgraph_deep_analysis]] §1.5/§4.4，759→12 行，净删 747)。

**承接页接收清单**：
- `npugraphs_make_graphed_callables_deep_dive.md`：净增 0 行(668→668，§四无独有内容)。
- `aclgraph_deep_analysis.md`：净增 63 行(569→632)，新增 §1.5「mode 参数与两条路径的触发关系」(并入「一、路径概述」)与 §4.4「与 backend="npugraphs" 路径(路径 B)的对比」(并入「四、这条路径为什么会存在」)，均为小节融入而非尾部堆贴；同步合并 Related Pages 中两行重复的 `[[11_torch_compile_npugraphs_deepdive]]` 为一行(见下 C.1)。

**C. 三项遗留小修正**(上一任务质量审查发现)：
1. `aclgraph_deep_analysis.md` Related Pages 两行 `[[11_torch_compile_npugraphs_deepdive]]` 合并为一行，注释合并为"NPU Graphs 与 torch.compile 集成深度分析；§3.4-3.8 内存管理与复用；reduce_overhead vs npugraphs 对比"。
2. `torch_compile_npugraphs_deep_dive.md` §3.6 加一句区分 checkpoint 恢复三步机制与分类表 dead 行 `_npu_npuCachingAllocator_raw_delete` 为两层释放，不引入新机制断言。
3. 本 changelog 上条(P3 Task 4)追记回补后终值 2809 行(非 2714)与 §3.8 自纠记录(即上方"追记"段)。

**体量**：`torch_compile_npugraphs_deep_dive.md` 2809→1990 行(净删 819 = §四 -72 + 附录A -747)；`aclgraph_deep_analysis.md` 569→632 行(净增 63)；`npugraphs_make_graphed_callables_deep_dive.md` 668→668 行(不变)。三页合计 4046→3290，净删 756 行。链接检查 broken 0→0、ambiguous 70→70、bare_index 70→70、orphans 1→1(与本次改动无关，四项均与改动前基线一致)。全部新增/删除的 mermaid 块经 fence 计数校验闭合平衡(无需重渲，本次未新增 mermaid 内容，只有整块删除)。

**追记(复核回补)**：上方 B 段"六、…独有,浓缩进 §4.4 对比表"与"九、…独有,浓缩进 §4.4 对比表 + 选型清单"两处表述不够精确——复核发现浓缩过程连带丢了 6 组分析性事实(Inductor 各编译阶段"对性能的影响"列、6.3 三类回退场景的路径 A 特有行为、7.1 Wrapper 的 options 处理/reset 行为两行、7.2 `backend="npugraphs"` 时 `mode` 参数的 TypeError footgun、9.1 首次执行延迟与内存占用两个对比维度、9.2 路径 A/B 的具体选型场景列表)，已在后续 commit 中以表行/脚注形式逐字回补至 `aclgraph_deep_analysis.md` §1.5/§4.4；更准确的表述应为"部分浓缩,若干维度经复核回补"。另，上方"三、"段对附录 §3.6 的删除依据也不完整：其 `npugraphify` 函数体代码不仅与 `aclgraph_deep_analysis.md` §二·差异1 的 monkey-patch 注册代码重复，也与 `torch_compile_npugraphs_deep_dive.md` 自身 §2.4.1「npugraphify 函数（utils/_graph_tree.py）」的函数体重复，是双重重复而非仅与差异1重复。

---

## 2026-07-29：知识库结构整改 P0–P2(工具、快速止血、图源入库)

**Type**: Structure Reorg(设计:`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md`;这是 P0–P7 七阶段的前三段,后续 P3+ 将做内容去重与目录重组)

- **新增链接检查器** `tools/check_links.py`(+8 项测试):broken/ambiguous/裸 index/孤儿页四类检查,`--strict` 做阶段门禁;基线与终值存档于 `docs/research/2026-07-29-linkcheck-*.json`。
- **坏链清零**:broken 138 → 0。其中 113 处为指向已删审计产物的 `[[correction_report]]` 标注(删除时甄别出 14 处含实质技术警示的,以纯文本注回补);其余 24 处为陈旧目录名遗留、相对路径深度错、示例文本误解析等,逐处改指现路径/转义/登记 Knowledge Gap。
- **工作产物清理**:`docs/audits/`(55MB,107 文件)移出工作区并 gitignore;`torch_compile_debug/` 调试残留删除;`raw/_ingest/` 施工单与 wanka 源汇编稿归位 `docs/research/`;重复 docx 经核验后删除;demo 脚本迁 `tools/`。
- **图表可再生**:`.html2md` 中 24 个图表源 html 与 8 个渲染脚本入库 `tools/figs/`、`tools/html2md/`(修正六个脚本的硬编码路径深度);端到端验证 html 与 PNG 再生均字节一致。
- **文档除假**:README 与 `wiki/index.md` 重写/修正(删幻影目录树、全表页数按实数重算并注明统计口径);本 changelog 头部新增"按写入当时状态记载、不随迁移回写"政策。

---

## 2026-07-29：`torch.compile` A→F 六卷 CUDA-first 配套 Demo

**Type**: Executable Teaching Labs + Evidence Contract（固定实现审计仍以 PyTorch
`e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52` 为准；当前本机为 PyTorch `2.9.1+cpu`，
不把 CPU、FakeTensor、generated code 或源码可达性冒充 CUDA 实测。）

- **建立六个卷级入口与 45 个 case**：新增统一 harness，以及 A/B/C/D/E/F 六个入口。
  A 覆盖执行模型，B 覆盖 Dynamo，C 编排原有 21 篇深度 Labs，D 覆盖 artifact/cache/runtime，
  E 覆盖诊断、复现、正确性与性能，F 覆盖 compiled autograd、checkpoint、distributed、
  custom op/backend 与 AOTInductor。
- **固定运行与失败合同**：统一支持 `--list --json`、可重复 `--case`、`--device`、`--seed`
  和隔离的 `--output-dir`；`PASS/BLOCKED/FAIL` 分别使用退出码 `0/3/2`，CLI 合同错误为
  `4`。能力缺失在 case 正文前阻断，异常和子进程非零退出不会被吞成成功。
- **形成 60 页唯一映射**：`labs/demo_manifest.json` 把 A01–F08 全部正文映射到真实入口和
  case；A/B/D/E/F 的 39 篇正文在最终 `Related Pages` 前新增可复制命令、证据字段与
  CUDA 边界。C01–C21 保留原有逐篇 Lab，由 C 卷入口分主题编排，不复制或弱化原证据。
- **真实预检结果**：本机实际通过 A 的 4 个 CPU case、B 的 10 个 case、C 的 6 个编排
  case、D 的 3 个 CPU case、E 的 7 个 CPU case、F 的 4 个 CPU case；CUDA/Triton、
  native compiler、Linux distributed 和多卡路径按声明返回 `BLOCKED`。修复了
  after-Dynamo repro 误传 `make_fx` 图的问题，并把 Windows Gloo 不可运行边界改为
  声明式 Linux gate。
- **验证与审计闭合**：Labs 合同由 42 项增至 63 项并全部通过；审计工具 90/90 通过；
  六入口 subprocess JSON 清单、60 页映射、Markdown 回链、CLI 退出码、C 子进程证据传播
  均有自动测试。统一课程账本重建为 6,483/6,483 个 decisions，validation error 为 0；
  `[S]/[R]/[I]/[M]/[B]` 为 1,287/366/3,675/19/43。
- **保留验收边界**：当前没有 CUDA receipt，因此 GPU kernel 数、显存峰值、加速比、
  autotune winner、CUDAGraph replay、多卡 FSDP/DTensor 与 AOTI package load 仍未升级为
  `PASS`，必须在目标 CUDA/Linux 环境重新运行对应 case。

---

## 2026-07-28：`torch.compile` 端到端 A→F 课程补齐

**Type**: Source-faithful Learning Series + Evidence Closure（固定源码基线为 PyTorch
`e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`；本阶段只完善原理和源码链路，不新增 demo，
不把 CPU 环境观察外推为 native/CUDA 实测。）

- **新增课程域 `19_torch_compile_end_to_end/00_torch_compile_end_to_end_index`**（该目录已于
  P4 Task 10 整体解散删除，导读价值并入 [[courses/torch_compile_end_to_end]]）：
  形成 A→F 六卷编号路径；新写 A01–A05、B01–B10、D01–D07、E01–E09、F01–F08 共
  39 篇正文，并将 `00_pytorch_graph_series_index`
  的 C01–C21 正文、Labs 和证据资产实体并入同一目录；旧目录 16 已删除，原 01–21
  文件顺序和内容身份继续保留。
- **从 eager 贯通生产运行**：卷 A 建立 Tensor/storage/layout、dispatcher/autograd、
  Python frame、dispatch modes 和成本模型；卷 B 追踪 `torch.compile`、eval-frame、
  bytecode、VariableTracker/Source、OutputGraph、guards、graph break、dynamic shape 与
  backend；卷 D 衔接 `compile_fx`、AOT runtime wrapper、异步编译、cache、wrapper memory、
  CUDAGraph Trees 与 artifact lifecycle。
- **补齐工程验收闭环**：卷 E 按 capture/AOT/lowering/codegen/load/runtime 分层组织日志、
  explain、guard/recompile、minifier/bisector、正确性、冷启动/热缓存/稳态、fusion/memory/
  hardware 和 production rollout；明确“能编译”“能加载”“数值正确”“性能获益”是四个门槛。
- **补齐高级边界**：卷 F 区分 Compiled Autograd 与 AOTAutograd，解释 activation
  checkpoint 与 AOT recompute 的叠加、DDP/FSDP/DTensor 的图和 rank state、custom op
  完整编译契约、Dynamo backend 与 Inductor device backend、AOTInductor 的 PT2/C ABI/
  constant ownership，以及 training/inference/freezing/CUDA Graph 的正交组合。
- **纠正历史路径和关键误解**：minifier 当前入口核准为
  `torch/_functorch/fx_minifier.py`；AOTInductor package 入口核准为
  `torch/_inductor/__init__.py` 与 `torch/_inductor/package/package.py`。明确 JIT cache
  不是部署 ABI、freezing 不是 `eval()`、CUDA Graph static input 首先约束地址而非值、
  partitioned forward 的 saved activation 不能全部当作 backward static input。
- **统一课程与证据闭合**：端到端 manifest 已逐篇纳入 C00–C21，包含
  61 个编号入口/正文和 1 个 C00 支持索引；6,312 个 claim units 全部有决定，其中
  `[S]/[R]/[I]/[M]/[B]` 分别为 1,287/366/3,552/19/43，另有 1,045 个非断言；
  validation error 为 0。卷 C 自身账本也已随实体迁移重建并复核。
- **导航闭合**：框架总索引、Dynamo、AOTAutograd、Inductor、runtime memory、FX/export/
  extensibility、distributed primitives、C 卷图编译主线与 compile cache
  均已建立课程回链。当前主机无 MSVC/CUDA/Triton；未新增或伪造 native runtime receipt。

---

## 2026-07-28（二）：K3 开源栈源码审计 + 稳定性专题横切

**Type**: Source Audit（代码）+ Deep Dive + Cross-Document Correction（把 K3 随发布开放的仓库从“报告项目级说法”推进到源码级，并把散落各页的稳定性机制横切成一页。）

- **新增 [[01_theory/01_models/moonshot_kimi/27_moonep_analysis]]**：固定 `MoonshotAI/MoonEP@0f385f03`（2026-07-28，MIT）做 file:line 级审计。拆出在线规划的五步算法（全局直方图 → `balance` 守恒 → 贪心配额 → 逐专家分配 → 精确落点 + dedup），并从算法本身推出 README 只给结论的那条约束——贪心“一次填满最空接收方”⇒ 每 rank 至多从一个远端 home group 接收 ⇒ 训练时 `B=E/R` 足够。记录工程形态（几乎全部 kernel 用 CUTLASS Python DSL 写，C++ 仅 60 行 pybind + VMM/IPC/multicast；planner 是单个 cooperative kernel，计划全程不下 GPU）、梯度回收闭环、以及源码注释里写明的被否决方案（远端写清零 vs 本地清零的 NVLink 预算权衡，`grad_reduce.py:24-30`）。
- **新增 [[01_theory/01_models/moonshot_kimi/26_kimi_k3_open_source_stack_analysis]]**：按 GitHub API 逐仓核对 `created_at`/`pushed_at`，把仓库分成三类（生产栈组件 / K3 自己写的作品 / 评测配套）。**更正一条流传口径**：FlashKDA 建于 2026-04-20、最新 commit `d2ff19a`（2026-05-26），**并非随 K3 新开源、也未为 K3 更新**；真正落在发布窗口的新仓是 MoonEP、AgentENV、minitriton、nano-kpu、PerceptionBench。补齐 AgentENV 的 overlaybd 与 memory ballooning（报告未提）、minitriton 与 nano-kpu 的免责声明与“综合估算而非 P&R”口径。
- **新增 [[01_theory/01_models/moonshot_kimi/25_kimi_k3_stability_analysis]]**：按“哪条轴会失稳”重组报告 §2.2/§2.3/§2.4/§2.5/§3.2/§3.3/§3.4/§4.1.2/§4.1.4 与 Appendix C，得出主线——**K3 拒绝的每个替代方案都是“用质量或超参换稳定”，被采纳的机制几乎都同时提升质量或降低开销**（aux loss、sign 更新、BIP、hard clamp、无界 SwiGLU、SigLIP 初始化、WSD 七处取舍指向同一方向）。
- **修正 [[01_theory/01_models/moonshot_kimi/23_kimi_k3_infra_deepdive]] §2.1**：原记“Per-Head Muon 如何与 K2 MuonClip 组合仍未知”过窄；报告 §3.3（p.11）明写三者并用（Per-Head Muon + K2 weight-clipping + QB），并给出 cosine + 1% warmup、weight decay 0.1、8k→64k 预训练。“未知”收窄到联合消融与 clip 阈值。§5 事实边界表的 MoonEP 行标为已兑现、AgentENV 行补仓库口径。
- **补齐 [[01_theory/01_models/moonshot_kimi/22_kimi_k3_architecture_deepdive]] 两处缺失的“为什么”**：MoonViT-V2 从零训练的**首要动机是训练稳定性**（SigLIP 初始化的 MoonViT-3D 梯度范数持续偏高且频繁 spike，Fig. 6），且视觉质量持平；Block AttnRes 在 K3 的精确配置为 8 块 × 12 层、计入 embedding 共 9 块，开销从 `O(Ld)` 降到 `O(Nd)`。
- **`D12`（`03_posttraining/12_kimi_k3_posttraining_case_study_analysis`，历史活链接，已于 2026-07-31 因 kb-reorg P5 迁移为 [[24_kimi_k3_posttraining_case_study_analysis]]，按"历史不回写"惯例降级为反引号）新增 §9.1**：AgentENV 仓库侧口径，并标注与报告延迟数字（133/49 ms vs `<100`/`<50 ms`）的口径差异，证据等级升至“可下载实现 + README 自报”，未升 P2/P3。
- **补写 [[01_theory/01_models/moonshot_kimi/22_kimi_k3_architecture_deepdive]] §六 SiTU（2026-07-28 追加）**：把原来 5 行的条目扩成完整一节，并新增自绘四联图 `assets/kimi_k3_fig_situ_range.png/.svg`（按报告 §2.3.2 与 Appendix B 公式数值绘制，非复制 Fig. 4）。补出报告 Appendix B 的设计目标原文（bound the SwiGLU product **without discarding the characteristic shape of Swish**——保住原点近线性与消失负尾）、只 cap 线性因子而保留 sigmoid 的理由、Eq. 18 的一阶等价与 Eq. 19 的界，以及 hard clamp 被否决的原因（饱和边界外梯度归零）。**值域主结果**：预激活不变；门支 `(−0.2785,+∞) → (−0.2698, 4)`（下确界只动 3%，cap 实际只作用于正半轴）；up 支 `ℝ → ±25`；输出 `ℝ → (−100,100)`。另加三条本库推算：四角点显示 ±100 两端都由门支饱和到 4 驱动、门支负半轴对输出量级贡献上限仅 6.74；cap 保留线性值的比例只依赖 `z/β`（0.25β→98.0%、1β→76.2%、2β→48.2%），据此说明 `β₁=4` vs `β₂=25` 意味着门被管得比 up 严约 6 倍（取值理由报告未给，标 [推断]）。
- **证据边界（显眼的缺席）**：K2 有“15.5T tokens 零 loss spike”，**K3 报告没有等价陈述**——无训练 loss 曲线、无 spike 计数、无容错章节；全文唯一 spike 级实测证据是 Fig. 6 的 MoonViT 梯度范数对比。因此“K3 稳定性机制更系统”可说，“K3 训练更稳定”不可说。

---

## 2026-07-28：Kimi K3 技术报告回填后训练统一学习域

**Type**: Source Ingest + Industrial Case Study + Cross-Document Correction（固定官方报告 `0797decb`，将算法、trajectory、environment、Infra 与部署精度放回同一个 `wiki/03_posttraining/` 闭环。）

- **原始来源**：新增 `raw/01_theory/01_models/moonshot_kimi/Kimi_K3_Technical_Report_2026-07-28.pdf`，SHA-256 `fd6ee35c07766a5eb6104235f1b407e4329f969e3482b8c42937c7b5f2b3efe1`；来源台账补 §4.1、§4.2、§5.3 与 Appendix F 的精确定位。
- **新增 D12 `03_posttraining/12_kimi_k3_posttraining_case_study_analysis`**（历史活链接，已于 2026-07-31 因 kb-reorg P5 迁移为 [[24_kimi_k3_posttraining_case_study_analysis]]，按"历史不回写"惯例降级为反引号）：串起 SFT → 九个 domain/effort 专家 → MOPD，澄清 partial rollout 保留 prompt 内 \(K\) group，并分析 white-box environment、XTML preserved thinking、MXFP4/MXFP8 QAT、draft model、external KV pool 与 AgentENV。
- **回填统一主线**：更新 D00–D05 与 D11；把 K3 作为项目级工业案例，而不是没有训练源码证据的“第五个开源框架”；D00 与领域/全局索引扩展为 D00–D12 连续编号。
- **修正事实边界**：量化 scheme 一致只消除该维度 TIM；K3 partial rollout 不是 fully async；Figure 8、MOPD、GRM、external KV 与 AgentENV 均保留未披露超参数、消融或运行条件。
- **K3 旧页同步**：总览、架构、Infra 与 Moonshot 索引从“报告/权重待发布”更新为 2026-07-28 固定报告，并把后训练机制统一链接到 D12。

---

## 2026-07-28：PyTorch 图编译系列源码级重构与审计闭合

**Type**: Source-faithful Refactor + Final Audit（固定源码基线为 PyTorch
`e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`；本轮冻结演示扩展，集中完成原理与源码机制说明。）

- **建立编号化学习主线**：形成 `00` 总索引与 `01–21` 正文，按“图语义基础 → 捕获与
  AOT 正反向构图 → 安全改图与 PatternMatcher → Inductor IR/Scheduler/Codegen”
  顺序组织；总索引新增源码阅读方法与各阶段 driver 入口。
- **补强源码机制**：所有正文均从概念解释推进到真实调用链、读写状态、不变量、
  consumer、设计取舍和参数化复杂度；重点闭合 FX use-def、AOT fw/bw runtime ABI、
  saved tensor/recompute、PatternExpr 匹配、DCE/稳定拓扑、lowering/realization、
  Scheduler dependency/fusion 以及 codegen/autotune/cache 边界。
- **纠正关键误解**：fw/bw 是两张独立 FX Graph，没有跨图 Node 边；recompute 是
  partitioner 向 bw fresh graph 复制节点；PatternMatcher 按注册 root 候选索引匹配；
  DCE 必须结合副作用判定；Scheduler topo 与 fusion candidate 复杂度均按真实实现参数化，
  不再笼统写成“逐 pattern 扫整图”或“必然二次”。
- **课程证据闭合**：3134/3134 个课程 claim decisions，0 个 validation error；
  916 个固定源码证据、366 个当前环境实测、1284 个带已验证父结论的推论，41 个环境阻塞
  保持显式，不把 generated-only 或 blocked 能力升级成 native runtime 事实。
- **历史材料隔离闭合**：28 篇旧页完整保留；2190/2190 个历史 claim 有处置，
  91 个纠正，2099 个未证实结论全部 `retain-quarantined`；新课程未导入任何 unresolved
  history claim。
- **验证结果**：审计工具 90 项测试与课程合同 42 项测试均通过，21/21 个既有
  runtime producer receipts 成功；本机缺少 MSVC/CUDA/Triton，原生 C++ kernel 与
  CUDA/Triton autotune 仍保持 `BLOCKED`，没有扩展演示 demo。

---

## 2026-07-27：完成 LLM 后训练 D02–D11 算法、Infra、框架源码与 CUDA–Ascend 深挖

**Type**: Deep Dive + Source Audit（在统一 `wiki/03_posttraining/` 中完成 S01–S05，固定论文版本和四框架 commit，贯通算法统计语义、在线数据、系统与硬件。）

- **算法与数据语义**：新增 D02–D04，比较 GRPO、DAPO、Dr. GRPO、GSPO、SAO，定义 Agentic trajectory/reward/credit schema，并严格拆分 system async、policy lag、off-policy 与 training–inference mismatch。
- **Infra 与主框架**：新增 D05–D07，以 control/data/weight 三平面建立工业机制模型；完成四框架矩阵，并从 verl `main_ppo.py` 追踪 `RayPPOTrainer.fit`、`DataProto`、advantage、actor update 与 rollout weight refresh。
- **三个源码对照**：新增 D08–D10，固定 slime `aaf5c20`、AReaL `b23fa6c`、ROLL `370cb24`；澄清 TransferQueue 不属于 slime core，分别追踪 slime async producer、AReaL freshness/weight services、ROLL Strategy/device mapping/NPU platform。
- **CUDA–Ascend**：新增 D11，按 device、collective、train/rollout、weight sync、kernel、dynamic shape、profiling 和版本矩阵建立 M1–M4 迁移验收；纳入 2026-07-27 的 vLLM-Ascend、SGLang NPU、MindSpeed 与 HCCL 公开进展。
- **证据与导航**：新增 `docs/research/2026-07-27-posttraining-source-ledger.md`；D00、D01、领域索引和全局索引更新为 D00–D11 顺序入口；旧 GRPO/DAPO/GSPO/verl/infra/sandbox 页面增加统一领域回链。

---

## 2026-07-27：建立 LLM 后训练统一纵向学习域与 S00 前沿快照

**Type**: Research Baseline + Learning Route（将 Reasoning RL、Agentic/Coding RL、Infra、工业框架源码与 CUDA/Ascend 映射纳入同一研究闭环。）

- **新增统一入口**：建立 `03_posttraining/index`（历史活链接，该域已于 2026-07-31 因 kb-reorg P5 解散并删除，S00–S05 阶段叙述并入 [[courses/posttraining_frontier]]，按"历史不回写"惯例降级为反引号），后续 D00–D11 新研究统一写入 `wiki/03_posttraining/`；旧理论与工程页面保持原位，通过链接复用，不再把 RL 算法和 Infra 分散承载。
- **新增 D00 `00_posttraining_source_reading_guide`**（历史活链接，该页已于 2026-07-31 因 kb-reorg P5 解散删除，阅读路线骨架 + 六级能力门槛并入 [[courses/posttraining_frontier]]，按"历史不回写"惯例降级为反引号）：按 D00 → D11 固定推荐阅读顺序，定义 S00–S05 六个研究阶段、六级可验证能力门槛，以及论文、源码、工业“支持等级”和 CUDA→Ascend 适配的阅读方法。
- **新增 D01 `01_posttraining_frontier_map_analysis`**（历史活链接，已于 2026-07-31 因 kb-reorg P5 迁移为 `posttraining_frontier_map_analysis`、同日 Task 8 再编号为 [[01_posttraining_frontier_map_analysis]]，按"历史不回写"惯例降级为反引号）：以优化粒度、on-policy/freshness、训练—推理一致性和 Agentic 环境四组张力组织前沿地图；固定 verl `983cb0f`、slime `aaf5c20`、AReaL `b23fa6c`、ROLL `370cb24` 的 2026-07-27 源码快照。
- **研究分工**：verl 作为主基线，slime 作为性能/前沿对照，AReaL 作为 fully async/Agentic 对照，ROLL 作为多后端、异构和 Ascend 专项；不使用脱离模型、硬件、配置和 freshness 条件的总榜。

---

## 2026-07-26：图编译知识体系重构复核——课程主线通过，历史无损迁移仍未验收

**Type**: Source-faithful Refactor + Design Conformance Review（不删除旧页；固定源码审计基线为 PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`。本机 Lab 使用 PyTorch `2.9.1+cpu`、`torch.version.git_version=5811a8d7da873dd699ff6687092c225caffcf1bb`，两条基线分开记录。）

- **重审 `00_pytorch_graph_series_index`**（该页已于 P4 Task 10 删除，导读价值并入
  [[courses/torch_compile_end_to_end]]）**与 21 篇专题**：Part I 建立图 IR、FX 数据模型、值/metadata/signature、符号形状、effect/alias/mutation、结构化输出与 higher-order graph；Part II 解释捕获、规范化、AOTAutograd joint→fw/bw、saved tensor/recompute ABI 与跨阶段 provenance；Part III 解释 FX 改图原语、`PatternExpr`、DCE/稳定拓扑排序、pass 顺序/fixpoint、合法性与复杂度；Part IV 贯通 FX lowering、Inductor IR、buffer/liveness、Scheduler 与 codegen/autotune。21 篇正文的 323 个完整 repository-relative `file:line` 定位在固定 checkout 上全部路径存在且行号有效。
- **可执行证据升级**：Lab 目录现有 18 个机制/贯穿脚本和 1 个 9-test 自动合同入口；合同覆盖四种捕获、FX 不变量、effect/alias/DCE、AOT joint/fw/bw/recompute、PatternMatcher、pass/topology、Part III rewrite、Part IV IR/Scheduler/provenance 与统一模型 bundle。AOT Lab 现用同次 partition 的 lab-only origin token 建立 joint→fw/bw 精确 old-to-new 映射，并把 saved-slot fw value/bw placeholder 绑定到同一 joint origin；artifact manifest 明确整体 continuity 为 `partial`，不再把独立前端捕获和独立 backend 捕获写成一条单次编译链。本轮复跑 `Ran 9 tests`，结果 `OK`；审计工具 8 项测试也全部通过。
- **Part III 贯穿改写**：`add(matmul(x, weight), bias) → addmm(bias, x, weight)` 仅在 rank、dtype、shape/无 broadcast 等合法性成立时执行；数值、一阶梯度、`gradcheck`、shape、alias relation、输入 mutation relation、非法 broadcast 拒绝、失败原子性与第二次运行零改动均有 assertion。
- **Part IV 证据边界**：真实执行 GraphLowering、Scheduler、dependency/fusion/reorder、external matmul 与 `eigvals` fallback；custom lowering 到达 `ComputedBuffer`；生成 wrapper/C++ source 并完成 Scheduler→FX→Python provenance join。当前 Windows CPU 缺 MSVC `cl` 且无 CUDA，因此 native pointwise/reduction kernel、真实 fusion 性能、物理 allocator peak 与 Triton autotune 没有实测；mock/no-op 捕获的 codegen 产物明确标记为 generated-not-executed，Scheduler group 数也不再误写成 native kernel 数。
- **迁移与导航**：28/28 篇 manifest 页面均有“页面角色与审计状态”说明并保留全文；其中 19 篇旧主干页面不再整页 blanket deprecated，另 9 篇 runtime/checkpoint/cache/memory 专题补齐独有职责、基线和未闭合边界。综合报告增加旧节→新系列去向表；同步更新框架总索引、AOTAutograd、Inductor、FX/export、runtime memory 和 compile cache 六个索引，并补齐相关回链。
- **历史审计 gate 仍打开**：manifest 已扩为 28 页、2,514 条 inventory records、2,022 条 heading/code/locator ledger rows，围栏不平衡为 0；但仍有 832 个 `TBD` destination、1,041 个 unresolved-like row（1,030 `unresolved`、1 `not_semantically_audited`、10 `needs_manual_resolution`）、0 个真实 destination anchor。故当前新课程可作为固定基线下的已验证主线使用，但整个重构不能宣称满足“历史资料逐结构单元无损迁移”或 authoritative 发布门槛。完整结论见 `docs/audits/pytorch_graph_series/2026-07-26/design_conformance_review.md`。

---

## 2026-07-25：训推不一致（TIM）因果链——第 1 块首页与 raw 摄入清单

**Type**: Source Ingestion + Deep Dive（严格路线：所有断言带一手来源与 §/Fig/Table/Eq 级定位符；未核实项显式标注，不做推测补齐。）

- **新增 `tim_causal_chain_analysis`**（515 行，历史活链接，已于 2026-07-31 因 kb-reorg P5 Task 8 再编号为 [[26_tim_causal_chain_analysis]]，按"历史不回写"惯例降级为反引号）：打通本库此前断掉的一环——「kernel 非确定性 → logprob 偏差 → 重要性比方差放大 → 训练崩溃」。上游（浮点非确定性、batch 不变性）已由 [[determinism_and_numerical_reliability_analysis]] 覆盖，下游（loss spike 治理）已由 [[training_dynamics_stability_analysis]] 覆盖，本页补中间两环与算法侧修法全谱。
- **归因框架**：采用 Qwen 2512.01374 §2.4 的二因子分解（训推数值分歧 × 策略陈旧度）作为全页坐标系，据此把 PPO clip、TIS/TRM/ALP/MIPU、batch-invariant kernel、TBIK、FP16、MoE 路由回放归位到各自作用的因子上。
- **六类病因逐条落定位符**：引擎间 kernel 差异、batch 触发不同 tiling、**TP size 改变累加顺序**（本库此前未覆盖）、BF16 尾数不足、MoE 路由分歧、量化 rollout。其中 TBIK Table 1/2 的对照实验证明「只做 batch 不变，对跨 TP 数值发散几乎无效」（Llama-3.1-8B 上 BIO 的 27.54 甚至高于 BF16 的 26.48）。
- **崩溃形态学**：首次在本库区分 recomputation 与 bypass 两条路径的不同崩溃曲线（多阶段 vs 单阶段、是否伴随 loss spike），并记录一条对工程有直接影响的发现——**K1/K3 KL 对 recomputation 型崩溃是盲的**（前 700 步几乎平坦而 reward 已在退化），而 recomputation 正是 verl 等框架的常见默认路径。
- **确定性税汇总**：batch-invariant GEMM 194 vs cuBLAS 527 TFLOPS；单个确定性请求混入 11 请求批次使整批吞吐掉 56%；TBIK 端到端 22%–63%；vLLM+TorchTitan bitwise 一致 RL run 慢 2.4×。并指明三篇系统侧论文**无一测量 RL 闭环端到端代价**，列为头号 open question。
- **三处 `> [!contradiction]` 标注**：① TIM 根因是精度（2510.26788）还是优化（2602.01826）——两篇互相点名，本页据 $C\cdot T^2$ 的乘积结构论证二者数学上不互斥；② MoE 的 Routing Replay 是否必要——GSPO §5.3 主张可取代 vs 同作者组 2512.01374 §4.4 回到「必需」，并如实标注后者**未点名**前者；③ 确定性是否必须付性能代价——实测派 vs 2606.00279，本页指出后者目标是可审计而非不变性，且「无性能代价」这一标题级主张零测量。
- **两处 raw 原文核实**（直接读 `raw/` 中已有 PDF）：DeepSeek-V4 §3.3 的 dual-kernel strategy 抵消的是**放弃 split-KV 后 decoding attention 的 wave-quantization 损失**，不是 matmul 固定 tiling 的损失（matmul 侧是 DeepGEMM 端到端替换 + 放弃 split-k 后另做优化）；GSPO §5.3 的 Routing Replay 表述与 10% 专家漂移数字逐句核对。
- **更新 [[01_theory/04_posttraining/index]]**：新增「训推一致性（TIM）与 RL 稳定性」小节；**首次建立 Knowledge Gaps 节**，记录 9 条确认无一手来源的缺口（崩溃阈值、尾部刻画、逐位置增长曲线、极端 token 频率、重尾→熵坍塌因果、RL 闭环确定性税、VeXact 开销、RL 阶段路由坍塌、vLLM logprobs RFC）；标注三条旧目录结构下的失效链接并给出替代。
- **新增 `raw/_ingest/INGEST_MANIFEST_block1_tim.md` 与 `fetch_block1_tim_sources.ps1`**：29 项已核实来源（26 篇 PDF + 3 份官方文档）的摄入清单与下载脚本，含目标路径、arXiv ID、一句话定位、保真度标注，以及 7 项需浏览器手动存档的博客/issue。**容器侧 arXiv 被代理阻断，PDF 需在本机执行脚本落盘。**
- **Mermaid 三块**：均按 CLAUDE.md 清单逐条自检，并用 mermaid-cli 实渲通过。

---

## 2026-07-23：FX Graph 构图、AOT 正反向分图与 PatternMatcher 改图机制报告

**Type**: Deep Dive（按 PyTorch `ea5655fcebf` 固定基线核对 FX、AOTAutograd 与 Inductor 源码。）

- **新增 `fx_graph_construction_and_transformation_analysis`**（历史活链接，该页已于 2026-07-30 判重删除，AOT 特有内容并入 [[11_aotautograd_joint_forward_backward_graphs_analysis]]/[[12_saved_tensors_recompute_and_runtime_abi_analysis]]，FX 数据模型部分此前已并入 [[10_fx_graph_core_data_model_analysis]]，按"历史不回写"惯例降级为反引号）：统一解释 `Graph` 侵入式双向链表、`Node.args/kwargs`、`_input_nodes/users` 与查找辅助表，区分图序、依赖边和 `GraphModule.forward` 生成代码。
- **补全 AOTAutograd 正反向构图**：从 joint function tracing、partition 分类与子图抽取，解释 fw 额外输出 → runtime context → bw placeholder 的跨图 ABI；明确 fw/bw 无对象级 Node 边。
- **补全 recompute 机制**：说明 min-cut 保存/重算选择、普通 forward 节点复制进 bw，以及 backward recompute reorder 如何缩短临时值生命周期。
- **补全 PatternMatcher、DCE 与保序**：覆盖 PatternExpr 子类的图场景、候选桶与逆序匹配、三类 PatternEntry、mutation/stream 边界、dead node 定义、稳定拓扑排序、lint/recompile 检查点。
- **复杂度模型**：给出构图、候选检索、匹配/替换、DCE、lint、稳定拓扑排序、fw/bw 抽取、min-cut 与全 pass 管线的参数化复杂度；同步更新 AOTAutograd、Inductor、FX/export 索引与双向链接。

---

## 2026-07-22：Torch Compile 八阶段 Pass 开发方法论与 Dynamo/Codegen 缺口补齐

**Type**: Deep Dive + Correction（按 PyTorch `9922478dffa` 固定基线复核并补齐 pass 放置方法、关键 API、注册示例与阶段注意事项。）

- **方法论升级**：重写 `fx_pass_optimization_methodology`（2026-07-30 起并入 [[24_graph_pass_pipeline_ordering_and_fixpoint_analysis]]，详见该日期 changelog 条目）的权威主干，完整覆盖 Dynamo → Pre-Grad → AOT/Decomposition → Joint → Post-Grad → Lowering → Scheduler → Codegen；每阶段均回答“是什么、为什么、适合做什么、为什么不放相邻阶段”，新增选择表、放置规则、六问设计法和验证矩阵。
- **补齐三块缺失内容**：新增 [[30_dynamo_pass_methodology]]（backend callable/`register_backend` 边界）、[[33_decomposition_passes_guide]]（decomp table、AOT 注入位置、注册/选择方法）、[[34_codegen_extension_guide]]（`BaseScheduling` + Wrapper + `DeviceOpOverrides` + `register_backend_for_device`）。
- **三阶段 Pass 指南纠错**：[[30_pre_grad_passes_guide]] 订正 non-functional/non-normalized IR、真实执行顺序、`pre_grad_custom_pass(Graph)->None` 和缺失 `PatternMatcherPass` import；[[31_joint_graph_passes_guide]] 订正 `pass_patterns` 所属模块、两轮顺序、Graph hook 契约和“空 hook 确保加载”错误；[[32_post_grad_passes_guide]] 补真实全流程、三轮 pattern、inference-aware hook、通信 bucketing 与 reinplace 尾部不变量。
- **Lowering/Scheduler/Codegen 纠错**：`lowering_analysis`（历史活链接，该页已于 2026-07-30 判重并入 [[10_fx_lowering_to_inductor_ir_analysis]]，按"历史不回写"惯例降级为反引号）订正“Post-Grad 在 Lowering 之后”的错误顺序并补 `register_lowering`/fallback API 示例；`scheduler_analysis`（历史活链接，该页已于 2026-07-30 判重删除并入 [[13_scheduler_dependency_graph_fusion_and_ordering_analysis]]，按"历史不回写"惯例降级为反引号）明确真实接口是 `_pre/_post_fusion_custom_pass(list[BaseSchedulerNode]) -> list[...]`，将 `GraphLowering`/`node.fusable` 旧示例标为 deprecated；[[20_inductor_codegen_analysis]] 更新固定基线入口并链接完整扩展指南。
- **动态形状方法修正**：把“遇到 SymInt 一律跳过”改为“符号恒等或 ShapeEnv/guard 可证明则支持，无法证明才拒绝”。同步更新 Dynamo/Inductor 索引与交叉链接。

---

## 2026-07-22：CUDA GEMM / 非 GEMM / Ascend 算子三篇生产级 Kernel 资料入库

**Type**: Source Ingestion（按 `llm-knowledge` 的 raw→wiki→index→backlink→changelog 流程导入用户提供的三份 HTML。）

- **原始资料归档**：新增 `raw/02_engineering/05_gpu_kernel/{cuda_gemm_final,cuda_nonmatmul_kernels_final,ascend_kernels}.html`，保持下载文件字节不变；SHA-256 分别为 `56f589…f85b0`、`ba1ce1…db02`、`1a6b9c…f7f0`。
- **新增 [[cuda_gemm_kernel_analysis]]**：以 SM80 / A100 代表性配置串起 Grid→CTA→Warp→MMA、M/N 空间切块与 K 时间归约、`cp.async` 完成语义、每线程约 232 寄存器账本、shared-memory epilogue 与生产级 kernel 骨架。
- **新增 [[cuda_nonmatmul_kernels_analysis]]**：以 roofline + 五类数据依赖为统一分类，覆盖 elementwise、reduction、norm、FlashAttention、stencil、scan、gather/scatter/sort，明确 shape 会让同一算子跨 compute-/memory-/latency-bound 阵营。
- **新增 [[ascend_kernel_execution_model_analysis]]**：把 CUDA 两篇映射到 DaVinci AI Core 的 Cube/Vector/Scalar/MTE、GM→L1→L0→UB 显式缓冲链、Queue 双缓冲、FixPipe，以及 compute / memory / communication 三条训练优化主线；明确其为平台对照材料、非官方文档。
- **可核验性与图形**：三页逐章附 raw HTML 行号范围和快照哈希；19 个内嵌 SVG 已渲染为 PNG。更新 GPU Kernel、工程与总索引，并向 [[gpu_kernel_guide]]、[[cuda_execution_model_guide]]、[[triton_03_matmul_guide]]、[[operator_optimization_guide]]、[[13_mindspeed_ascend_affinity_analysis]]、[[21_npu_inductor_optimization_analysis]] 补回链；未发现需标记的既有内容冲突。

---

## 2026-07-20（四次更新）：[[22_npu_fusion_passes_deepdive]] §5 后端级融合大幅展开——加「决策链 + 代价模型」

**Type**: Update（应用户「§5 后端 pass 需展开:当前后端优化都做了哪些?怎么建模选择最终的融合方式?」。1 路 source-audit agent 核 scheduler/select_algorithm/tiling 决策链 + 本人抽验 7 处载荷 file:line。）

- **§5 从 3 小节（CATLASS/DVM/can_fuse）扩为 8 小节**：新增 §5.0 决策链（mermaid：后端路由→合法性→排序→收益实测→tiling / GEMM 分支 autotune→EVG）、§5.1 后端路由（`choose_node_backend` + `TORCHINDUCTOR_NPU_BACKEND`）、**§5.3 融合收益建模（核心）**、§5.4 GEMM 实现+epilogue autotune、§5.5 tiling 编译期穷举、§5.7 四类建模范式小结。原 can_fuse/CATLASS/DVM 归位重编号。
- **核心回答「怎么建模选最终融合」**：三段式=**启发式定合法性 + 实测定收益/实现 + 编译期穷举定形状**。真正的实测代价模型只有两处——③收益 `speedup_by_fusion` 编译 fused/unfused 真机实测 `ms_fused < ms1+ms2`（`scheduler.py:541`）；④GEMM 多模板 `finalize_as_caller` 选最快实现（`:440-441`，同时决定融不融与选哪个实现，真机 AICore profiling `do_batch_profiling`）。tiling 靠 UB 公式 `max_numel_threshold=ub_size//ptr//dtype` 编译期穷举（`tile_generator.py:47-48`）。
- **两处源码校正**：① `score_fusion`/`score_fusion_memory` **NPU 未覆写**（全库无 `def score_fusion`），NPU 只改邻近门 `are_long_distant_nodes`（64→20，仅 A5）——不能说「NPU 自定义融合打分」；② AKG 在本 baseline **实际停用**（mfusion 路径警告 "not supported currently"），不写成活跃后端。两个实测开关（`CATLASS_EPILOGUE_FUSION`、`TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING`）默认均关。
- 7 处载荷 file:line 已抽验（`scheduler.py:541/440-441/202-203`、`tile_generator.py:47-48`、`npu_combined_scheduling.py:40-43`、score_fusion 无覆写、`are_long_distant_nodes:39`）；新 mermaid 过本库规范校验。
- **追加（应用户「default 路径下走 CATLASS 分支怎么判？直接白名单吗」）**：§5.1 补「CATLASS 分支三时刻路由链」——**不是直接白名单**。① lowering 时 `tuned_mm` 过外层 3 门（连续性/非零/`use_catlass_template`）+ 内层 6 门（白名单 `mm/addmm/bmm` + size 阈值 + 非 ROCm + dtype + `use_max_autotune` + backend/库），过了才 `add_catlass_gemm_choices` 展开一批候选（`kernel/mm.py:79-135`、`utils.py:236-265`、`gemm_template.py:189,227-247`）；② autotune 真机实测选中，胜出才产 `CATLASSTemplateBuffer`；③ scheduler `is_catlass_template` 仅类型 dispatch（`catlass_scheduling.py:70-73`）。结论：白名单只是资格门之一，最终判据是 autotune 实测；无 max-autotune 时根本不生成 CATLASS 候选。所有 file:line 本人实读。

---

## 2026-07-20（三次更新）：工业界 FX Pass 全景——上游全集 + vLLM/SGLang 现状 + 开发方法论（4 页，3 路并发 agent）

**Type**: Deep Dive（应用户「是否有 upstream pass 分析全集?补一个 torch_upstream_pass_deepdive;vllm/sglang 是否也有大量 pass?并发总结;并归纳 pass 优化开发方法论」。3 路并发 source-audit agent 逐仓核源：upstream `9922478dffa`、vLLM `97a98006b0`、SGLang `d6ef68881e`。）

- **新增 `torch_upstream_pass_deepdive`**（04_inductor，2026-07-30 起判重删除、机制层并入 [[22_pattern_expression_and_matcher_engine_analysis]]/[[24_graph_pass_pipeline_ordering_and_fixpoint_analysis]]，详见该日期 changelog 条目）：上游 Inductor pass「全集 + 机制」总纲，补上三份 stage 指南缺的**机制层**——`PatternMatcherPass` 声明式引擎（声明→trace→匹配→改写）、三种 `PatternEntry`（lowering/graph/replacement）、`fwd_only`/`joint_fwd_bwd` 两种 trace、序列化 pattern 缓存（30 个 `_sfdp`）、三阶段驱动器 + `*_custom_*_pass` 钩子 + `GraphTransformObserver`，及 pre/joint/post_grad 全集目录。6 处载荷 `file:line` 已实读复核（`PatternMatcherPass.apply:2352` + 跨 mutation/stream 护栏 2400/2409、`LoweringPatternEntry.apply:1156`↔`graph.py:1372-1376` passthrough、序列化缓存 import 1973-1981、post_grad 三桶 85-89、`_sfdp_init:1487`）。
- **新增 [[sglang_compilation_passes_analysis]]** + 新建 `sglang/` 目录与 index：**核心发现——SGLang `srt/compilation/` 是 vLLM piecewise-cudagraph 管线的近逐文件 fork，但融合 pass 被整个抽空，真实图重写 pass 数=0**；唯一的 `FixFunctionalizationPass.__call__` 是 no-op（`fix_functionalization.py:34-37` 只 `count+=1`，`pass_manager.py:47-51` `configure()` 只挂它、`passes` 恒空）——两处均本人实读复核。含两条 compile 路径、split_ops 切图、CUDA/NPU/XPU piecewise backend 差异。
- **更新 [[vllm_ir_and_fusion_passes_analysis]]**：新增 §3.5「Pass 全家福 + 三大融合维度」——回答「vLLM 有没有大量 pass：有，约 23 个」（16 融合 + 6 IR/utility + 1 pre-grad），建在 torch `pattern_matcher` 上，朝**厂商 kernel**（FlashInfer/cutlass/symm_mem/AITER）融合，三维度 upstream 没有：**集合通信 / 量化 / KV-cache 写入**；补 pre-grad 钩子 + `compile_range` 门控 + 三种 pattern 注册形态。
- **新增 `fx_pass_optimization_methodology`**（04_inductor，2026-07-30 起判重删除、内容并入 [[24_graph_pass_pipeline_ordering_and_fixpoint_analysis]] §14 等处，详见该日期 changelog 条目）：跨四家归纳的 pass 开发方法论——四家现状对照表、四个决策问题（在哪做 Q1 / 匹配什么 Q2 / 怎么落地 Q3 / 怎么保证对 Q4）、**融合朝向谱系**（codegen→厂商库→预融合 kernel，解释 vLLM 写十几个而 SGLang 一个不写）、工业界工程护栏（uuid=源码 hash / 可 bisect / 门控分层 / pattern 对 custom_ops 鲁棒 / 序列化缓存）、开发 checklist、反模式（搬框架≠搬融合、pass 会腐化、跨类别改写会错）。
- **联动**：04_inductor/index 加两页；03_infer_frameworks/index 加 SGLang 子框架行；各页 Related 交叉链接。

---

## 2026-07-20（二次更新）：[[22_npu_fusion_passes_deepdive]] 新增 §7「改图操作原语与 pass 通用原理（机制总纲）」

**Type**: Update（应用户「补充这个改图的操作，以及 pass 主要操作的原理是什么」——承接对 `view_fold_pass` 的连续追问：多个 view 如何变一个、如何保证等价、DAG 扇出/多后继怎么处理。）

- **新增 §7（机制总纲）**：把 26 个 pass 共用的改图机制抽出来单讲。§7.1 **FX 改图操作原语表**（`replace_input_with` 边局部 / `replace_all_uses_with` 全局 / `call_function` / `inserting_before` / `erase_node` / `propagate_fake_tensor` / `eliminate_dead_code`，全文 132 处调用，每行带代表 `file:line`）+ 数据模型（`node.args` 与 `node.meta["val"]` FakeTensor）；§7.2 **pass 四步通用套路**（定位→改写[指针重接 | 造等价新子图]→维护 meta→DCE 清理）；§7.3 **`view_fold_pass` 全走查**（t0/t1/t2 拓扑序传递塌缩 + A/B/C/D DAG 扇出的边局部安全 + 等价性论证）；§7.4 **三条贯穿原理**（纯函数⇒边局部安全 & 单用户门槛判据、拓扑序⇒一趟塌缩、等价来自算子类别不变式、meta 一等公民 + 静态 shape 门槛）。
- **核心结论**：「多变一」= 指针重接让末端算子直连源头、中间节点变孤儿再 DCE，**不是生成合并算子**；view_fold **不需要单用户前提**（边局部改写对 DAG 扇出天然安全），而 `fold_cat`/`fold_squeeze` 因会改动前驱本身才查单用户（`:287`/`:580`）。
- §1 加「机制总纲建议先读 §7」前向指针；§3 `view_fold_pass` 表行加「全走查见 §7.3」。§7.1 全部 `file:line` 经 grep/直读复核。
- **补图**：§7.3 增 before/after mermaid 图直观展示 `replace_input_with` 的**边局部**改写——处理 B 时只把 B 的入边从 A 改指 x（橙色），`C→A`/`D→A` 两条边纹丝不动、A（绿色）因 D 仍引用而存活；对照 `replace_all_uses_with` 的全局替换。图经本库 mermaid 规范校验（无嵌套定界符/管道标签合规）。

---

## 2026-07-20：新增 [[22_npu_fusion_passes_deepdive]] —— 自定义融合 Pass 逐个深挖（场景·问题·优化·效果）

**Type**: Deep Dive（应用户反馈「[[30_npu_vs_upstream_fusion_passes]] 对每个 pass 的 why/场景/效果讲得不够——需要具体代码场景、为什么这么优化、优化带来什么效果」。三路并发 source-audit agent 逐函数读 `ascend_graph_pass.py`(2548 行)全部 26 个 pass + helper。）

- **新增 [[22_npu_fusion_passes_deepdive]]**：对 26 个自定义 pass（4 PRE + 22 POST）+ 3 个后端融合机制（CATLASS EVG epilogue / DVM 图级分区融合 / `NPUTritonScheduling.can_fuse`）逐个给「**触发场景（含 before 代码）→ 待优化问题 → 优化机制（after）→ 效果**」四拍，每条带 `file:line`。重点展开 9 个「真·融合」pass：`masked_add_compose`（互补掩码相加→单 where）、`bool_cast_mul_to_where`（bool cast×→where）、`sign_diff_hamming_fuse`（符号位汉明距离 6 算子链→gt/gt/ne/sum）、`batch_embedding_fusion`（N 段 embed+reduce→单次 reshape→embedding→reduce）、`cat_to_view`/`repeat_to_expand`（cat/repeat→零拷贝 view/expand/roll）等。
- **效果口径诚实边界**（核心方法论）：逐函数 grep 确认**全文除 CATLASS `catlass_epilogue_fusion_counter` 外无任何计数器/benchmark**，故所有效果均为**结构性**（少 N kernel / 少一次拷贝 / 转零拷贝 / int32 索引），非实测加速；多个 pass 带**中文 docstring 直述动机**（标「原文」，可信度最高），而源文件**无硬件注释**，凡「因为达芬奇/UB/i64」因果均标 [硬件推断]；`fold_four_op`/`fold_where` 经 `get_binary_fold_result` 会留一个 clone、非零成本。
- **校正**：`fusion_attention_v3_pass` 是 **基础版 `npu_fusion_attention.default` → `v3.default`**（本 baseline 无 `_v2`），非「v2→v3」——同步订正 [[30_npu_vs_upstream_fusion_passes]] §6 措辞。
- **联动**：[[30_npu_vs_upstream_fusion_passes]] §3.4 加深挖页指针、Related 补链；更新 [[04_inductor/npu/index]] deep dive 表。

---

## 2026-07-17（五次更新）：新增 [[30_npu_vs_upstream_fusion_passes]] —— torch_npu vs 上游 Inductor 融合 Pass 全流程对照

**Type**: Deep Dive（应用户「总结为一个新页，从 FX pass 到后端全流程比较 torch_npu 与 upstream 融合 pass 的差异、谁有谁无及原因，稽核源码」。逐行核验两套 checkout：torch_npu `b3c8a815b`(v2.7.1) + upstream `9922478dffa`(main)，三路并发 source-audit agent + 主体自审。）

- **新增 [[30_npu_vs_upstream_fusion_passes]]**：主线「torch_npu 不重写上游 pass 管线，而是三处介入 + 重活下沉后端」。逐层对照 pre_grad / joint_graph / post_grad / lowering-decomp / 后端 scheduler；三张「谁有谁无谁不同」总表；`is_gpu`/`GPU_TYPES` 门控如何决定上游 pass 在 NPU 上跑不跑（`patch_is_gpu` 追加 `"npu"`，但硬编码 `.is_cuda` 的 pad_mm/b2b_gemm/decompose_mm 仍不跑）；26 个 `ascend_custom_passes` 全清单（4 PRE + 22 POST，仅推理）；根因收敛到 Cube 专用单元 / ACLNN 手工库 / 达芬奇布局约束 / 集成方式四条。
- **源码级校正（source-wins）**：① `patch_pattern_mm_plus_mm` 是**删除**上游 `mm_plus_mm`（注释「torch_npu does not support」）而非添加——纠正 [[01_npu_compile_paths_overview]] §2.5 旧记法；② fallback 实测 **932**(340+592) 而非旧「约 963」；③ persistent reduction 内置后端是**阈值门控非恒关**（恒关的是实验性 Linearize 后端）；④ [[21_npu_inductor_optimization_analysis]] §12.4 自定义 pass 清单已过时（`unfold_dual_reduction_pass` 不在此 head，新增 `sign_diff_hamming_fuse_pass`/`batch_embedding_fusion_pass` 等）；⑤ `pattern_match/npu_fusion_attention_graph.py` 在此 checkout **未接线**（无生产代码 import），真正生效的 attention 改写是 v2→v3 的 `fusion_attention_v3_pass`。
- **联动**：更新 [[04_inductor/npu/index]] deep dive 表；为 [[01_npu_compile_paths_overview]] 与 [[21_npu_inductor_optimization_analysis]] 补 `## Related Pages` 反向链接。

---

## 2026-07-17（四次更新）：GDN/KDA 线性注意力从公式到训推融合 Kernel 的完整闭环

**Type**: Deep Dive（承接用户连续追问：QKVABZ、$a/b/z$ 设计、RNN 中的 $t$、chunk size 与 $t$ 的关系、chunk 数学等价性、仿射状态是否必须保序，以及当前训练/Prefill/Decode kernel 融合。）

- **新增 [[20_gdn_kda_linear_attention_analysis]]**：从 $x_t\rightarrow q,k,v,a,b,z$ 开始，逐步解释 raw $a\rightarrow g=\log\alpha$、$b\rightarrow\beta$、$z$ 输出门的职责分离；给出 GDN 标量 decay 与 KDA 逐通道 decay 的五步递推、统一仿射式 $S_t=A_tS_{t-1}+B_t$、$C=3$ 展开，以及 chunk 摘要 $(P,R)$ 的保序结合复合。明确纠正“chunk 状态矩阵直接相乘”和“有结合律即可乱序”两个误解。
- **新增 [[21_gdn_kda_kernel_implementation_analysis]]**：训练侧固定 FLA `ccb0ff944cbf`，拆解 autograd chunk forward/backward、gate+cumsum、KKT+solve-tril+W/U、状态 scan、输出和反向重算；推理侧固定 SGLang `7824903417b7`，拆解 QKVABZ 投影融合、Prefill $C=64$ chunk pipeline、Decode fused recurrent 五步、GDN packed-decode 与 speculative verify。明确 SGLang 是推理基线，不用其 forward-only 代码冒充训练反向。
- **原始来源与联动**：新增 raw 快照 `Gated_Delta_Networks-2412.06464v3.pdf`；修正 [[12_kimi_linear_analysis]] KDA 公式中 $S_{-1}$ 的下标笔误为 $S_{t-1}$；更新 Moonshot/Kimi 与模型总索引，并为 [[22_kimi_k3_architecture_deepdive]] 补充双向入口。
- **后续补充：TND/THD packed 输入**：[[21_gdn_kda_kernel_implementation_analysis]] 新增 §八，明确 TND/THD 的 $T=\sum_iL_i$、外层 batch=1 与 `cu_seqlens` 状态边界；逐段追踪 Megatron-LM `dev@232c478d43ce` 的 `T×1×D → per-sequence CP→HP → 1×T×N×d → boundary-aware short conv/chunk GDN → per-sequence HP→CP`，并解释为何每条 packed 序列必须独立重置 RNN 状态、chunk 不能跨 pack 边界，以及当前 batch、CP 对齐、FLA 与 inference 限制。
- **TND 代码级追踪**：同页 §8.6 补充最小等价伪代码，并把边界隔离落实到三层实现：Megatron `_unpack_sequence` 与逐序列 CP↔HP、FLA causal-conv 的 `bos/eos + boundary_check`、GDN Triton state kernel 的 `N×head` program grid 与每序列独立 chunk-state 槽。

---

## 2026-07-17(三次更新): [[23_kimi_k3_infra_deepdive]] 新增 §四「负载建模」—— 逐模块 roofline 记账与 bound 判定

**Type**: Update(应用户「探讨引入 KDA 后各部分的负载瓶颈与 bound;训练侧按 8K→64K→256K→1M 逐模块计算说明;不压缩合并入库」)

- **推理侧(§4.1-4.4)**:decode 逐部件字节/FLOP 记账——KDA 状态 R/W(~2 FLOPs/B 带宽 bound 但噪声级,Kimi-Linear 模板 40MB/token)、Gated MLA KV 扫描(1M=1.15GB/层/序列,**与论文实测 TPOT 1.84ms@1M 交叉验证吻合**)、LatentMoE 权重扫描(全量 1.49TB/步 ≈ 4.8ms 地板/64 卡,MXFP4 砍半的对象)、EP a2a 同步延迟(~120 次/步,supernode 的真正含义);prefill 三段迁移:冷算 170 PFLOPs@1M → FlashKDA 的"小矩阵效率" bound → 命中 >90% 后变"恢复带宽" bound(50-200ms vs 冷算三分钟)。
- **训练侧(§4.5-4.7)**:8 个模块逐个"公式→代数→判 bound"——专家 GEMM t≈2500/专家 compute bound(与推理相反);EP a2a 16ms vs 27ms/层的重叠余量(无 LatentMoE 则 64ms 反超成主 bound);MLA 注意力 ∝L(256K 交叉、1M 占 78%,373ms vs 27ms/层);KDA 常数但 bwd 状态检查点 32GB/层/序列;AttnRes 1M 显存税 8.6GB/卡;**Per-Head Muon 的 NS 正交化代价 ∝min²×max,按头切片砍 64×**(整矩阵 11 TFLOPs vs 64 片共 0.17 TFLOPs);PP 气泡 (p−1)/m 随 m∝1/L 崩塌 → 拓扑必须 PP→CP 重排;激活显存 1TB/序列@1M 决定 CP 下限。含四档 L 的 binding constraint 矩阵与两个反直觉结论(a2a 随 L 变轻;训推瓶颈迁移方向相反,3:1 同时钉住两头)。
- 全部模型维度假设(d=8192、48+16 层、d_lat=2048、激活 50B)与硬件锚点(H200/H20/H800 公开规格)显式标注 [推断];§五清单新增 6 行负载建模敏感参数,报告发布后代入公式即可整体刷新。原 §四顺延为 §五。
- 后续补充:§4.5 账本总纲下新增「记账单位说明」——"常数/∝L"均指每 token 成本;GEMM 每 token 恒定(按序列算总量仍 ∝L),注意力是唯一 token 两两交互、每 token ∝L 的模块;并注明两个二阶效应(GEMM 利用率、MoE t 判据)与口径稳健性检查。

---

## 2026-07-17(二次更新): K3 结构页吸收 GDN→KDA 单 token 数据流图;总体结构图恢复为官方原图

**Type**: Update(应用户「吸收 gdn-qkvabz-dataflow.svg 并详细解释;总体结构图直接复用原报告的,不要自绘」)

- **[[22_kimi_k3_architecture_deepdive]] 新增 §2.3「单 token 数据流:q/k/v/a/b/z 六路信号如何各司其职」**:融合投影切六路信号的分工表、状态更新五步拆解("误差修正"读法,并给出与 §2.2 闭式公式的代数等价展开)、GDN→KDA 需改的三处(每 head 标量门→Diag 逐通道门、投影/卷积布局、输出门定位),源码定位补齐(`fla/ops/gated_delta_rule/naive.py:31,54` vs `fla/ops/kda/naive.py:30-31` 等);原 §2.3/2.4 顺延为 §2.4/2.5。图源 `assets/kimi_k3_fig_gdn_qkvabz_dataflow.{svg,png}`(图内标注实现基线 SGLang main@78249034,以图注为准)。
- **官方架构图原图可渲染了**:从博客 CSS bundle(`78620784116c2822.css`)抽出全部 89 条 `BlockAttnRes-module` 规则,与官方内嵌 SVG 拼装成自包含 wrapper `assets/kimi_k3_official_arch_render.html`,按暗色主题 2× 渲染为 `assets/kimi_k3_official_arch.png`;[[14_kimi_k3_analysis]] 的总体结构图改用该官方原图,自绘重绘版 `kimi_k3_arch_redrawn.{svg,png}` 删除。原图证实:每个子层是常规残差 (+) 与 AttnRes (w,α) 取回并存,最终 Output 前还有一次 (w,α) 聚合;MoE/KDA 放大面板中的低秩梯形投影清晰可见。

---

## 2026-07-17: 新增 Kimi K3 收录(三页)—— 首个开源 3T 级模型的发布报告 + 结构变化 + 训推 infra

**Type**: Ingest(应用户「kimi3 发布了,总结模型报告 + 结合开源源码分析结构变化点与训推 infra」。**关键事实：K3 权重承诺于 2026-07-27 前发布，完整技术报告尚无明确发布日期**——当前“报告”实体 = 官方发布博客，已快照落 `raw/01_theory/01_models/moonshot_kimi/Kimi_K3_blog_2026-07-16.{txt,html}`；结构机制的源码证据来自官方声明 K3 所基于的组件仓库，均已实际克隆/打开核验)

- **新页**:[[14_kimi_k3_analysis]](发布总结:2.8T/896选16/1M 上下文,33 项基准全表与口径脚注,官方自报"位列 Fable 5 与 GPT 5.6 Sol 之后",preserved thinking history 等限制)、[[22_kimi_k3_architecture_deepdive]](六大变化点各按动机→机制→证据→为何不选替代:KDA 3:1 混合、Gated MLA+NoPE、AttnRes、Stable LatentMoE+Quantile Balancing、SiTU、2.8T/1M/视频)、[[23_kimi_k3_infra_deepdive]](Per-Head Muon、静态 shape 全平衡 EP、INT4→MXFP4/MXFP8 QAT 演进、Mooncake >90% 命中、KDA prefix caching 进 vLLM 的 PR 链 #27654/#42406、FlashKDA CUTLASS kernel、64+ 卡超节点账)。
- **可读性修订**：结合 K3 官方 Tech Blog（当前公开“报告”）重写三页的主线与长段落，明确区分官方事实、组件证据和推断；将 KDA、AttnRes、Per-Head Muon、MXFP4 容量与 Mooncake 成本公式改为独立公式块并补齐符号说明；价格统一写作 `USD/MTok`，避免美元符号与 Markdown LaTeX 定界符冲突。同时订正发布时间口径：7 月 27 日是权重发布期限，不是技术报告的官方承诺日期。
- **源码核验基线**:Kimi-Linear @`8c1d85e` + HF 48B config/modeling @`e1df551a` + fla @`b328e7c`(KDA 公式↔`fla/ops/kda/naive.py:59-63` 逐行对照;27 层实际 20 KDA:7 MLA;MLA `assert use_nope`);Attention-Residuals @`85e2231`(仓库仅 README+论文,伪代码 README.md:52-91;arXiv 2603.15031v1 全套消融);FlashKDA @`d2ff19a`(CHUNK=16 双 kernel,H20 1.85–2.31×)。
- **图**:9 张入 `moonshot_kimi/assets/`——官方基准图 2 张(PNG 原件)+ 官方内嵌架构 SVG 原件(aria-label 即 "Block Attention Residuals architecture diagram")+ 按官方风格重绘架构主图 + KDA/AttnRes/LatentMoE/KDA-prefix-cache/Mooncake/MXFP4-QAT 自绘深色 SVG(均 2× 渲染 PNG,已逐张目检)。
- **索引联动**:[[moonshot_kimi/index]] 家族表/时间线/论文索引增 K3 与 AttnRes 行,知识缺口标注 AttnRes 已覆盖;[[01_theory/01_models/index]] Kimi 段增 4 行(含补录 12_kimi_linear_analysis);[[12_kimi_linear_analysis]]/[[13_kimi_k2_5_analysis]]/[[11_kimi_k2_analysis]] 增后继回链。
- **待回填**：激活参数、层数、SiTU/Quantile Balancing/Per-Head Muon 精确定义与训练规模；三页中所有 `[推断]` 项须在完整技术报告发布后核对（报告日期未定，缺口清单见 [[23_kimi_k3_infra_deepdive]] §4）。

---

## 2026-07-16: 新增 [[21_aclgraph_multistream_rng_analysis]] —— ACLGraph 多流依赖与 graph-safe RNG

**Type**: Ingest / codebase deep dive（应用户连续咨询，将“多流入图、`wait_stream`/Event、遗漏 join、通信流场景、随机数入图与算子适配”去重后落库）

- **基线**：`torch_npu v2.7.1@b3c8a815`、其记录的 `op-plugin@6ef73e399`、PyTorch `main@2b460d01`。当前 op-plugin 工作树仍是较旧 `b7a17be...`，算子引用统一从固定 git 对象核验。
- **核心结论**：多流不是多个图 capture 后合并，而是 Event Record/Wait 把 side stream 纳入同一个 `model_ri_` 并在 capture end 前返回主流；`wait_stream` 只建立“源流调用点之前 → 目标流调用点之后”的局部单向偏序。RNG 也不是 CPU 完全不可用，而是 replay 不重跑 host 取 seed 逻辑，故用 device seed/offset tensor + intragraph offset + wholegraph increment 保证每次 replay 推进 Philox counter。
- **源码展开**：记录 `NPUGraph` 单模型 capture/replay、Stream/Event fork/join、HCCL 独立流及 allocator `recordStream`；以 native dropout 串起 secondary stream 与 Tensor RNG API；按固定 op-plugin 源码区分明确支持、部分重载/SoC 分支和仍走 `philox_engine_inputs()` 的路径。
- **测试审计**：保留 torch_npu 已有多流、RNG functional/distribution 测试证据，同时标出测试清单与固定 op-plugin 中 `bernoulli_` 等旧标量路径的版本/dispatch 张力；补充逐 overload、连续 replay、自定义 generator、无 join 负例、多副流 RNG 与 ACLGraph+HCCL 验收矩阵。
- **去重联动**：原 [[10_aclgraph_deep_analysis]] 只保留总览并链接专题；[[01_aclgraph]]、[[npu_operator_graph_eligibility_guide]] 与 NPU Graphs index 增加反链，不复制机制正文。
- **二次扩写**：按用户反馈恢复对话中的机制细节：显式 Event 与 `wait_stream` 的差别、capture/replay 逐阶段时序、三流间接加入与逐级返回、HCCL compute→comm→compute 双向依赖、数值化 Philox offset 演算、counter 计算边界、dropout 捕获/重放时间线，以及可直接转为用例的 RNG/多流验收模板。专题页由 312 行扩至 465 行，仍低于单页 500 行拆分线。

---

## 2026-07-16: 新增 [[inkling_analysis]] — Thinking Machines Inkling (975B-A41B) 收录,含 thinking_machines 新目录

**Type**: Ingest(应用户「今天 Thinking Machines 发布开源模型,分析并落库」。**关键事实: 无正式技术报告/论文**——arXiv 无;"technical report" = 官方公告 + HF 模型卡 + config.json,已全部落 `raw/01_theory/01_models/thinking_machines/`)

- **新源文件(raw/)**: `Inkling_config.json`(架构 ground truth,HF `thinkingmachines/Inkling`)、`Inkling_HF_model_card.md`(完整基准表 + Apache 2.0 许可 + 安全评估)、`Inkling_official_announcement_2026-07-15.html`(训练配方 + 设计哲学)。
- **新页**: [[inkling_analysis]](主线: 不抄 DeepSeek 作业的多模态开源 MoE,赌可定制底座而非榜首)+ [[thinking_machines/index]];[[01_theory/01_models/index]] 增 Thinking Machines 段。
- **本库对 config 核验的架构差异化要点**(全部带行号):
  1. **抛 RoPE**: 学习式相对位置编码(`d_rel=16, rel_extent=1024`)+ >128K logit 缩放外推到 1M。
  2. **抛 MLA**: 滑窗/全局 5:1 交错(`local_layer_ids` 55 层 + 11 全局),**非对称 KV 头**(全局 8 / 滑窗 16,窗口 512)。
  3. **SConv**(核 4)+ **encoder-free 四模态**(vision hMLP patchify 40×40 / audio 离散 dMel 16 级)+ **8 层 MTP**(vs DeepSeek/Hy3 的 1)。
  4. 路由沿用 sigmoid 免辅助损失,但 `route_scale=8.0` 异常大;Muon+Adam 混合训练 + muP。
- **影响力判断**(§五,已与事实分离): 前 OpenAI CTO 首发即开源(Apache 2.0)的象征意义;Tinker 微调变现的商业模式创新;抗审查/校准的差异化卡位;encoder-free 在视觉基准上已见代价(MMMU Pro 落后)。
- **联动反链**: [[hy3_analysis]](保守 vs 差异化对照)、[[12_deepseek_v3_analysis]](选择性继承)、[[13_kimi_k2_5_analysis]](多模态路线对照 + K2.5 合成数据冷启动)、[[21_hw_friendly_llm_codesign_analysis]](NVFP4 部署)。
- **校验**: config 行号逐一对 raw/ 核对;基准表从 HF 模型卡原表摘录(非二手);mermaid 结构图按库规范自查(首行 flowchart TB、英文 id、subgraph 标题无 `[]|`、标签无裸定界符)通过;无技术报告故"为什么"部分推断已显式标注。页 <300 行。

---

## 2026-07-15: 新增 [[21_torch_npu_upstream_adaptation_analysis]] — Ascend out-of-tree 适配与 PyTorch upstream 差异全景

**Type**: Ingest / codebase comparison（应用户“结合工作区 torch_npu 和 PyTorch 源码梳理 Ascend 适配主要点与上游差异”。基线：`torch_npu v2.7.1@b3c8a815`、其记录的 `op-plugin@6ef73e399`、PyTorch `main@2b460d01`，均按本地源码逐点核实行号）

- **中心结论**：torch_npu 应拆成三层评价——PrivateUse1/autoload/Guard/Hooks/Allocator/Dispatcher/AMP/c10d/Dynamo/Inductor registry 等**标准插件面**；ACL/CANN、私有 format、ACLNN、HCCL、Ascend codegen/ACLGraph 等**硬件实现面**；改写 Dynamo rule/Variable、Inductor lowering/scheduler/wrapper、cudagraph tree、distributed/FSDP 等**兼容补丁面**。前两层不是“落后”，第三层才是升级风险主来源。
- **口径订正**：旧页“Ascend 三路径 vs 社区统一 Triton”已加 `[!deprecated]`；当前 upstream CUDA/XPU 也采用 combined scheduling 混合多 codegen，差异应看公开注册接口与私有 patch 的边界。
- **upstream 新方向**：核实 AcceleratorHooks/`torch.accelerator`、实验性 Python PrivateUse1 hooks/guard、Inductor custom pass/config 与 device-module 自动 codegen 注册、`CUDAGraphPolicy`、distributed backend entry point、OpenReg 可执行规格；据此给出 P0-P2 收敛路线。
- **现场约束**：torch_npu 严格 pin PyTorch 2.7.1，而对照为 upstream main，页内明确区分版本差异与硬件差异；op-plugin 工作树未停在 gitlink commit，正文只引用 git 对象中可核验的记录版本配置。
- **联动**：更新 `01_dispatcher_and_device/index`、AI framework 总索引；为 [[11_privateuse1_device_integration_analysis]] 增反链；[[01_npu_compile_paths_overview]] 增过时口径提示与反链。

---

## 2026-07-15: 新增 [[21_hw_friendly_llm_codesign_analysis]] — NVIDIA 硬件友好 LLM 设计指南(软硬协同,系列第一篇)

**Type**: Ingest(应用户「总结该 blog 并加入知识库」。源 = NVIDIA Developer Blog 2026-07-10,HTML 快照已存 `raw/01_theory/06_distributed_parallelism/`;正文经本地 HTML→文本提取逐行核验,7 条 Guideline、公式、Table 1/2、Fig. 4/5/7/9 图注均照原文录入,未依赖 WebFetch 小模型转述——两次转述在"对齐 128/256/512"表述上确有出入,以原文为准)

- **落位裁定**: 本库 `05_inference` 域定义为 CoT/RAG/Agent(推理=reasoning,见 01_theory/index),故本页落 **06_distributed_parallelism**(兄弟页 EP/PP/TP 直接对应博客 §五/§六);raw 同步新建 `06_distributed_parallelism/` 目录。
- **页面要点**: 主线"模型超参(H/H'/L/对齐/精度)= 部署性能参数";roofline 记账 + Table 2 小 K 反例(H'=512 全程 memory-bound)+ Fig. 4 阈值(80% 吞吐需 K>3072/N>2560,GB300+NVFP4)+ tile 量化(128/256/512)+ NVFP4 双层缩放(16 值 micro-block E4M3 + FP32)+ 宽 EP 的 GEMM-M 公式 + CPP/Helix。§7 单列**立场评注**(NVIDIA 硬件本位、阈值不可跨硬件搬运)并对照 [[hy3_analysis]] 专家维 1536 与 K>3072 阈值的张力。
- **联动**: 06 index 页面列表增行、日期 bump;[[14_expert_parallel_analysis]] / [[15_pipeline_parallel_analysis]] Related 区各加反链;mermaid 决策图按库规范自查通过。

---

## 2026-07-14 (二): 新增 [[stem_sparse_attention_analysis]] — 混元 Stem 免训练稀疏注意力 (arXiv 2603.06274v1)

**Type**: Ingest(应用户提问「Hy3 提出的推理 attn 结构是什么」——Phase 5 按需生长。溯源链: 中文报道 → 腾讯技术工程公众号(2026-06-26,经 53AI/搜狐转载)→ **论文本体 PDF 已下载入 raw/ 并逐节核读** p1-7)

- **定性**: Stem 是**推理服务栈的免训练 prefill 优化插件,不在 Hy3 开源权重内**(`modeling_hy_v3.py` 纯稠密 GQA,README 部署配方无 Stem)——回答了"推理的 attn 结构"与开源模型结构的边界问题。
- **机制**(全部对论文核验): TPD 预算按 query 位置从 k_start 线性衰减到 μ·k_start(Eq. 3, μ=0.7),理论依据是因果信息流的递归误差放大(Eq. 1/Fig. 2/Fig. 3);OAM 选块度量 = QK^T + 0.2·max(0, log‖V‖₂)(Eq. 7),由最小化稀疏-稠密输出重构误差推导(Eq. 5-6);块大小 128、恒保 4 初始+4 局部块、下限 54 块(§3.1/Alg. 1)。
- **证据**: LongBench 25-31% 预算近稠密(Table 2)、RULER 25% 预算稀疏方法最高分(Table 4)、可叠加 DeepSeek-V3.2 DSA/MiniCPM-4.1 再压 15-18% 预算(Table 3)、等预算消融 TPD +2~3.4 分(Table 5)、128K H20 prefill 1540→420ms=3.7×(Fig. 1/§3.3)。
- **口径修正**: 新闻稿"3.6 倍首字延迟"与论文 Fig. 1 的 3.7× 不一致,页内以论文为准并注明;"ICML-26 接收"仅新闻口径,PDF 页眉仍为 Preprint,两说并存;HPC-BSA 算子 ~3× 加速为公众号口径(论文用 MIT-BSA),已标二手。
- **联动**: [[hy3_analysis]] §2.4 增服务栈指针 callout + Related 反链;tencent_hunyuan/index 与 01_models/index 增行;mermaid 流程图按库规范自查通过。

---

## 2026-07-14: 新增 [[hy3_analysis]] — 腾讯混元 Hy3 (295B-A21B) 收录,含 tencent_hunyuan 新目录

**Type**: Ingest(应用户「增加最新的 Hy3 technical report」。**关键事实: 截至 2026-07-14 Hy3 无 arXiv 正式论文**——arXiv API 与 HF papers 索引均核验无果;"technical report" 实体 = 模型卡 + 开源工件 + 官方博客/新闻稿,已全部落入 `raw/01_theory/01_models/tencent_hunyuan/`)

- **新源文件(raw/)**: GitHub README EN/CN(@ `8a12d9af87c6`, 2026-07-06)、`config.json`(HF `tencent/Hy3`)、`chat_template.jinja`、transformers `modeling_hy_v3.py`(@ `295cee3e1d00`)、官方榜图 benchmark.png / benchmark-appendix.png(约 40 基准 × 11 模型全量矩阵)。
- **新页**: [[hy3_analysis]](主线: 架构冻结、全靠后训练的性价比 Agent 模型)+ [[tencent_hunyuan/index]];[[01_theory/01_models/index]] 增 Hunyuan 段;[[12_deepseek_v3_analysis]] Related 区加反向链接。
- **本库独立核验的三个关键发现**:
  1. **preview 与正式版 `config.json` 逐字段完全一致**(实测 diff)——三个月提升纯来自后训练,榜单增量(DeepSWE 0.9→28.0、USAMO 37.3→72.0 等)构成一组罕见的"纯后训练 ablation"。
  2. **路由 = DeepSeek-V3 免辅助损失配方原样采用**(sigmoid L310 + 选择期偏置 L312-313 + 原始分加权),非标处仅 `router_scaling_factor: 2.826`;注意力为 GQA+QK-Norm(保守派,未用 MLA)。
  3. **[!contradiction] 已挂**: preview 官方博客称"differentiated expert size + P-Penalty Loss",但开源工件为均匀 1536 专家(config + `HYV3Experts` 单张量存储 L324-333)——按工件优先原则记录两说。
- **校验**: 全部 config/modeling/chat-template 行号逐一对 raw/ 内文件核对;榜图数字从 PNG 原图读取(非二手转述);mermaid 结构图按库规范逐条自查(首行 `flowchart TB`、英文 id、标签无裸 `[]()|`、subgraph 标题合规、单独 `end`)通过;API 定价仅有二手来源,已标"存疑待核"。页 <300 行。

---

## 2026-07-07: [[longcat_2_analysis]] 订正 ScMoE 结构描述 + 新增 §5.5「计算-通信重叠调度（SBO / 训练双 chunk）」

**Type**: Update（应用户「2.0 模型结构介绍有点问题、并行策略设置未介绍」。源忠实——窗口拓扑据 SGLang `sglang-longcat-pr/longcat_flash.py:429-461` 逐行核验；SBO 阶段级调度据 LongCat-Flash 技术报告 arXiv 2509.01322 §5 并明确标注来源与推断边界）

- **订正结构描述的「fork 点」易错点**：`clone` 发生在 `attn0`(MLA₁) **之后**（`longcat_flash.py:449`），故 Fig-1 图注与「一句话读结构」原先把 `attn0` 计入「与 MoE 并行的稠密链」不准——已改为 **attn0 是 fork 前的共享前置**，可掩盖 MoE 通信的**重叠窗口 = 稠密FFN₁ + MLA-attn1 + 稠密FFN₂**（不含 attn0）。Fig-1 PNG 本身画法正确（attn0 在 fork 之上），**无需重绘**，仅订正文字；§2.3 note 补「窗口要说精确」段。
- **新增 §5.5**：ScMoE 短路只建立**数据依赖上的自由度**，窗口内部「怎么切/谁盖谁」是调度层选择、训练与推理各一套——
  - **推理 SBO 四阶段**（含 ASCII 图）：① MLA₁ → ② 稠密FFN₁+**MLA₂.QKV 投影段**掩 dispatch → ③ **MoE GEMM 裸露**（靠 wide EP/EP128 压薄）→ ④ **MLA₂.核心+输出投影段**+稠密FFN₂ 掩 combine。**命名消歧 callout**：SBO 的「Attn0/Attn1」= MLA₂ 的两个 phase，≠ 层内两个 MLA 块（`self_attn[0/1]`）。列出相对「稠密FFN 掩 dispatch、MLA₂ 掩 combine」粗说法的三处精确修正。
  - **训练 token 维双 chunk 互掩**（Flash 报告 §2.2「token 维细粒度切分并发」）：两 chunk 与 dense FFN、彼此 overlap，**连 GEMM 也参与掩盖**——与推理「GEMM 裸露」相反。
- **联动更新**：§5.2「ScMoE 调度」bullet 改指 §5.5；页头「维度」补「并行与重叠调度」；[[longcat_flash_analysis]] §五 SBO bullet 补四阶段/双 chunk 摘要并交叉链到本页 §5.5。
- **校验**：两处 ASCII 图为纯 `code` 块（非 mermaid，无渲染风险）；`longcat_flash.py:433/449/456/460/467-492` 定位符逐一对源核对；页仍 <500 行。

---

## 2026-07-06: [[25_unbacked_symint_analysis]] 增补 §10 —— unbacked 处理的最新进展（`guard_size_oblivious` → 显式 size-oblivious 原语族）

**Type**: Update（应用户「根据最新技术更新知识库」。源忠实——全部新断言据 pinned pytorch checkout `torch/fx/experimental/symbolic_shapes.py` 逐个核验签名/行号/docstring/`__all__` 导出，只扩展不删除）

- **§7 API 表补 5 条**（均带全路径）：`statically_known_false`、`guard_or_true`、`optimization_hint(x, fallback)`、`sym_and`/`sym_or`；并加一行指针指向 §10。
- **新增 §10「从 `guard_size_oblivious` 到显式 size-oblivious 推理原语」**：
  - **旧机制** `guard_size_oblivious`（`:534`）——对 size-like unbacked 隐式临时设值域 `[2,Inf]`，docstring 自承 "we may diverge in behavior"，隐式/难推理。
  - **新原语族**（逐个核验行号）：`guard_or_false`(`:1573`)/`guard_or_true`(`:1580`)/`statically_known_true`(`:1648`)/`statically_known_false`(`:1621`)/`optimization_hint`(`:155`)/`sym_and`(`:1672`)/`sym_or`(`:1698`)，均在 `__all__` 导出；三档分工（静态保守 / 有默认分支 / 仅优化不影响正确性）。
  - **迁移规模实测**（当前 checkout `torch/`）：`guard_or_false|true` **~366 处/44 文件**（decomp、`_refs`、`_meta_registrations`、Inductor `lowering/ir`、**DTensor** `_view_ops.py` 单文件 ~38 处）vs `guard_size_oblivious` **~18 处/9 文件** —— 数量级反转，佐证「显式 guard_or_* 已成默认范式、旧接口沦为残留」，并呼应 DTensor+dynamic shape 正被改造为 unbacked-safe。
  - **选型决策 mermaid 图** + 「是否必须解决 unbacked」取舍（graph break 是合法逃生口；仅 `fullgraph=True`/export/AOTInductor/整图 CUDA Graph 穿过数据相关 op 时才必须）。
- **校验**：新 API 的签名/行号/`__all__` 全部对 pinned checkout 源码核对；mermaid 块按本库规范逐条自查（首行 `flowchart TD`、英文 id、矩形/菱形标签无裸 `[]()`、连线标签无引号/括号/`|`）通过；页头「最后更新」改 2026-07-06 并注明增补范围与定位符基准；§10 内部锚点因 heading 含 en-dash 会被 GitHub slugger 吞成 `20252026`，已改 heading 为 ASCII 连字符使 `#...2025-2026...` 锚点可解析。页仍 <500 行。

---

## 2026-07-06: [[07_training_reliability/index]] 簇补 7 张机制图示（提升可读性）

**Type**: Enrich（应用户「三类文章基于原始 technical report 补图示、提升可读性」。据原文机制 + 其引用的技术报告绘制）

- 手绘 HTML+CSS/SVG → 本库无头 Edge 2× 渲染 **7 图**（源 `.html2md/figs/training_reliability_figs.html`，gitignored）：
  - **确定性&数值页**（[[determinism_and_numerical_reliability_analysis]]）：`tr_det_fig1` 浮点非确定性五层来源 · `tr_det_fig2` 长链累加病(顺序 O(n·ε) 吞位,BF16 Σ1000×1.0=256)与药(树形 O(log n·ε) + DeepSeek FP8 两级累加) · `tr_det_fig3` SDC 四层检测 + Gemini split-phase 确定性重放闭环。
  - **容错&恢复页**（[[fault_tolerance_and_recovery_analysis]]）：`tr_ft_fig1` 恢复粒度坐标系(8 环恢复链路 + Job/Pod/进程内/Step 各级砍环时间轴) · `tr_ft_fig2` hang 症状/病灶空间分离 + Flight Recorder seq 对账定位。
  - **训练动力学页**（[[training_dynamics_stability_analysis]]）：`tr_dyn_fig1` spike/NaN 排查决策树(确定性重放为核心分岔) · `tr_dyn_fig2` spike 治理四层防线(架构/优化器/数据/运维)。
- 各图嵌入对应小节（背景 / 如何发现 / 排查思路 / 解决方案）。**校验**：7 张 PNG 逐张实渲肉眼核对（恢复链路时间轴、seq 对账暗框、决策树双通道分支均正常，无溢出/无裸定界符）；图片用标准 `![](assets/*.png)`（SVG 渲染，非 mermaid）。

---

## 2026-07-06: 新建 [[07_training_reliability/index]] 簇 —— 摄入《万卡训练确定性与可靠性深度分析》(9 问题域·多来源综述)

**Type**: Ingest（应用户「把这份基于 LongCat 衍生的稳定性训练文档吸收到知识库」。源忠实——二手综述的结构化摄入，机制/数字/命令/代码忠实原文，交叉链到已有一手页）

**源**：用户提供的多来源综述 `raw/02_engineering/wanka_determinism_reliability_deep_analysis.md`（已存 raw、747 行），综合 Gemini 1.0/2.5 · Llama 3 · ByteRobust(SOSP'25) · MegaScale(NSDI'24) · Aegis(NSDI'25) · C4(HPCA'25) · DeepSeek-V3(ISCA'25) · Thinking Machines「Defeating Nondeterminism」· Anthropic postmortem · 华为 CloudMatrix · 美团 LongCat-2.0 博客 + Megatron-LM/NVRx/torch_npu 代码。

- **新建 `02_engineering/07_training_reliability/` 簇（index + 3 内容页）**，按原文四部分/9 问题拆解：
  - [[07_training_reliability/index]]（**coordinator 手写的 exemplar**）：问题地图（9 问题×两主线）+「确定性是故障定界的地基」主线 + 趋势与开放问题（原文第四部分）+ 与本库已有页的交叉表。
  - [[determinism_and_numerical_reliability_analysis]]（问题 1-4）：浮点非确定性五层来源（atomicAdd/split-K/通信规约树/MoE 排序/框架随机性）、batch 不变性与 RL 确定性（Thinking Machines、Anthropic top-k 事故、TIS）、低精度长链累加（pairwise/树形、FP32 main_grad、DeepSeek FP8 两级累加/DeepGEMM、Kahan）、SDC 四层检测体系（压测/统计/ABFT+DP hash/确定性重放）。
  - [[fault_tolerance_and_recovery_analysis]]（问题 5-8）：goodput/ETTR + 五级恢复坐标系（Job/Pod/Node/进程/Step + 算子链路级）+ 各家术语对照（华为 MindIO TFT 的 TTP/UCE/ARF、NVRx in-process restart、torchft、Gemini slice 弹性…）、hang/straggler（flight recorder/栈聚类/straggler 打分）、Checkpoint（异步+原子提交/本地分层/临终/数据回放）、网络链路（PFC 风暴/ECMP hash/链路级快恢/流量工程）。
  - [[training_dynamics_stability_analysis]]（问题 9）：loss spike/NaN 四类根因、分层监控+前兆指标、排查决策树、四层防线（QK-Norm/z-loss/soft-capping/EGS、MuonClip/AdaGC/ZClip、数据指纹、运维自动化）、2026 前沿（Muon 路线共识、DeepSeek-V4 Anticipatory Routing/mHC、Kimi K2.5 与 GLM-5 的 RL 稳定性与问题 2 合流）。
- **并行 writer-agent 契约**：3 内容页由 3 个 subagent 并行写，各读 raw 指定行段（Part1/2/3），严格「不加源外事实、保留全部数字/env-var/代码/出处、只用给定交叉链、无 mermaid」，结构化回报。

**整合**：[[02_engineering/index]] 子领域表加 07 行；[[index]] 目录树加 07、工程域表加「训练可靠性 4」行、按主题查找加一行。**校验**：3 页 grep 确认关键 env-var/数字/机制在位（CUBLAS_WORKSPACE_CONFIG、TORCH_NCCL_TRACE_BUFFER_SIZE、五级坐标系、MuonClip/Anticipatory Routing/129.3 MWh 等）；4 页全部 `[[链接]]` 机械核对**零死链**；4 页 grep 确认**零 mermaid**（全用 ASCII/代码/表）；抽读 determinism 页头+§1 核对源忠实与房风格。

---

## 2026-07-06: 新建 [[longcat_flash_analysis]] —— 摄入 LongCat-Flash（560B/27B MoE，ScMoE + 零计算专家首创）

**Type**: Ingest（应用户「把 LongCat-Flash 也摄入知识库」。LongCat-2.0 的架构前身，源忠实 + 抓本质）

**源（source-faithful）**：LongCat-Flash Technical Report **arXiv 2509.01322v1**（2025-09-01）+ released `meituan-longcat/LongCat-Flash-Chat/config.json`（含 `modeling_longcat_flash.py`）。config 逐字段核对；报告按 §/Eq./Table 定位（§2.1 零计算专家 Eq.1-5、§2.2 ScMoE Fig.4、§2.3 方差对齐 Eq.6-8、§2.4 MLA/MTP、§3.1 超参迁移 Table1 + 模型生长、§3.2 稳定性、Table2/3 评测）。

- **新建深挖页 [[longcat_flash_analysis]]**（主线「用零计算专家 + ScMoE 短路把 560B 的激活压到 ~27B」）：
  - **架构**：MLA(64h, q_lora1536/kv_lora512/nope128+rope64/v128) · **零计算专家**（512 FFN + **256 identity**、top-12、激活 18.6–31.3B 动态、PID 控偏置 + 设备级均衡损失）· **ScMoE 短路**（前块稠密 FFN ∥ 当前 MoE dispatch/combine 通信、TPOT 较 DeepSeek-V3 ↓~50%、质量中性 Fig.4）· MTP（单稠密头、接受率 >90%）。
  - **缩放/稳定性**：μP 超参迁移(s=8, 代理宽 768, Table1) · 模型生长初始化(14→28, r=2) · Router 稳定(Rg<0.1 + PID) · hidden z-loss(Eq.10) · Adam eps 1e-16 · 确定性 + SDC 检测。
  - **预训练**：20T tokens/30 天/98.48% 可用率；三段课程(通用→STEM&code 70%→长上下文 8k→32k 80B→128k 20B)；13-gram + BGE-m3>0.9 去污染。
  - **Infra/推理**：SBO(NVLink TP ∥ RDMA EP)；**H800 >100 TPS、$0.7/M**；投机解码。
  - **Agentic**：τ²-Bench 67.7 / VitaBench 24.30(30+ 工具、60+ 轮)为长板；Base/Chat 评测表(Table2/3)。
  - **§八 Flash→2.0 演进对照表**：Flash(560B/27B·28 层·MLA 全注意力·512+256 专家·128K·H800) → 2.0(1.6T/48B·38 层·MLA+**LSA**·768+128·**N-gram**·1M·**国产 ASIC**)。
  - [!correction] 订正本人先前假设：**Flash 亦用 MLA**（非 MHA/GQA）——config `attention_method:"MLA"` 坐实；ScMoE/零计算专家为两代共享、SGLang `longcat_flash.py` 同一份代码。

**整合**：[[meituan_longcat/index]] §一家族表 Flash 行改为已摄入、§五缺口更新；[[01_theory/01_models/index]] LongCat 区加 Flash 行；[[index]] 模型 30→31、LongCat 子域 2→3、导航加 [[longcat_flash_analysis]]；[[longcat_2_analysis]] Related 加前身回链。**校验**：9 个交叉链接目标本会话/前序 grep 核对存在；纯文本+表+ASCII，无 mermaid；数值带 arXiv §/Eq./Table 或 config 定位。

---

## 2026-07-06: 用开源推理代码升级 [[longcat_2_analysis]] 架构描述 + 3 张代码级模型结构图

**Type**: Enrich（应用户「longcat2.0 开源了推理代码，结合推理代码完善模型结构描述 + 详细绘制模型结构图，每步数据流用 SVG→PNG」。源升级：**codebase 一手 > 博客二手**）

**源（source-faithful, codebase）**：官方 2026-07 放出 `config.json` + 194 分片权重（HF `meituan-longcat/LongCat-2.0`，`GIT_LFS_SKIP_SMUDGE` clone）；GPU 推理经 SGLang **PR #30042** @ `HarryWu99/sglang@c6c36d9`。逐文件核对 `longcat_flash.py`（模型/ScMoE/MoE，1093 行）、`n_gram_embedding.py`（:134-175）、`nsa_indexer.py`（SI/CLI，:493/:539-559）、`config.json`（60 行逐字段）、README（SGLang 部署 + LSA/N-gram 说明）。

- **3 张代码级结构图**（手绘 HTML+CSS/SVG → 本库无头 Edge 2× 渲染；源 `.html2md/figs/longcat2_architecture.html`）：`assets/longcat2_arch_fig1.png`（整体前向 + 单层 ScMoE 短路放大）、`fig2`（N-gram Embedding 数据流）、`fig3`（MLA + LSA 数据流）。逐张实渲肉眼核对。
- **§1.1 核心参数表全部落实**（每条带 config.json 行号或 `文件:行`）：38 层 · hidden 8192 · 64 heads · MLA(q_lora 1536/kv_lora 512/nope128+rope64/v128) · 稠密 FFN 12288 · **768 路由 + 128 零计算(identity)专家 / top-12** · 专家 FFN 2048 · vocab 163840 · RMSNorm/SiLU · RoPE-YaRN(factor 120) · N-gram 16 路哈希 · LSA index_topk 2048 / local 1024 / init 16 / cli_factor 2 · MTP 3-step。
- **§1.2 换成 3 图 + 代码级结构总览**；**§2 三处订正**（`[!important]`）：① 注意力实为 **MLA**（LSA = MLA 骨干 + DSA 式索引器，非全新注意力）；② 层是 **ScMoE 短路**（2×(MLA+稠密FFN) ∥ MoE，`longcat_flash.py:449-460`），非「注意力→MoE」常规块；③ **零计算专家确有其事**（`zero_expert_num:128 identity`），激活参数随 token 动态。
- **§2.1/2.2/2.3 加「代码补充」**：LSA 的 SI（force-keep 16 sink+1024 local = 50%，印证官方图）/ CLI（cli_factor=2、缓存索引集印证 Q4）/ HI（SGLang 未实现）；N-gram（16 路多项式哈希→查表→投影→**mean**）；ScMoE（`LongcatFlashMoE` 768+128、top-12、`zero_experts_compute_triton`）。
- **[!contradiction] 订正**：§9.1 zero-compute experts 由「未证实」→**代码定案为常设机制**、二手「33–56B 区间」方向可信；新增「注意力实为 MLA」订正。§6 补**推理 FP8**（`LongCat-2.0-FP8` + bf16 KV，`longcat_flash.py:697-808`），训练精度仍未披露。§9.2 结构项**全部划除**（已回填 §1.1），仅留训练侧未披露。
- **家族 index**：§四 开源状态**翻篇**（未放出 → 已开源：权重 + config + SGLang 推理码）；§一/§二/§三 补 38 层 / MLA / ScMoE 短路 / 推理 FP8 / 128 零计算专家。

**整合**：改动集中在 [[longcat_2_analysis]]、[[meituan_longcat/index]]，新增 3 图。**校验**：3 PNG 实渲核对；图片用标准 `![](assets/*.png)`（非 mermaid）；config.json 行号与 `longcat_flash.py`/`nsa_indexer.py`/`n_gram_embedding.py` 关键行本会话开文件核对；SGLang 建模码不入本库（仅引 `file:line`）。

---

## 2026-07-03: [[35_inductor_memory_allocation_guide]] 新增 §5「内存越界/踩踏排查」— 补第二份社区材料的缺口(纠错版)

**Type**: Enrich（应用户"第二份专家社区材料的第三部分——内存踩踏检测——库里缺,补上,一定忠于事实"。经 gap 分析:材料前两部分已被 `inductor_memory_management_analysis`（历史活链接，该页已于 2026-07-30 判重删除并入 [[12_buffer_liveness_memory_planning_and_reuse_analysis]]，按"历史不回写"惯例降级为反引号）覆盖且更准,仅第三部分是真缺口;材料本身有错,按源码收敛后再入库）

**源（source-faithful）**：pytorch @ `5f6df46744a` 逐行核对——`config.py:232-237`（`size_asserts`/`nan_asserts`/`scalar_asserts` 默认值）、`codegen/wrapper.py:1793-1827`（`assert_size_stride` 生成 + 延迟到首个 kernel 前）、`ir.py:7817/7845/9152`、`utils.py:161`（`GPU_ALIGN_BYTES=16`）、`codegen/triton.py:5458`（`mask=xindex<xnumel`）、`codegen/common.py:2789`（核内 NaN 断言）。

- **guide 新增 §5 内存越界/踩踏排查**（原 §5/§6 顺延为 §6/§7）：§5.1 自动核内置防护（Triton `mask` + `assert_size_stride`/`assert_alignment`/scalar·nan_asserts,多为默认 ON）· §5.2 真正越界来源（自定义算子/手写 Triton/错误 stride·offset/unbacked symint）· §5.3 工具（`compute-sanitizer` 取代 cuda-memcheck + `CUDA_LAUNCH_BLOCKING`）· §5.4 排查步骤。
- [!correction] **订正该社区材料**（源 > 材料，均已核实）：① 「Inductor 对越界无内置保护」**错**——mask + size/alignment/scalar 断言多为默认 ON;② 「规划池用 `_cuda_beginAllocateToPool` 申请」**错**——该 API 全库仅在 `cudagraph_trees.py`（CUDA Graphs 私有池）,规划池是普通 `empty_strided`;③ `cuda-memcheck` 已被 `compute-sanitizer` 取代。**材料对的部分**（保留）：16 字节对齐确有其事（`GPU_ALIGN_BYTES=16`）。前两部分（池初始化/复用）不收录——深挖页已覆盖且更源忠实。

**整合**：[[04_inductor/index]] guide 条目补「越界/踩踏排查」;guide Related 增 [[23_inductor_gpu_kernel_dispatch_model]]/`Pytorch_Compile_Debug_Analysis`(该页已于 P4 Task 4 判重删除，内容并入 [[02_compile_stack/07_debugging/index]])/[[25_unbacked_symint_analysis]] 回链。**校验**：所有 `file:line`/常量值本会话开文件核对;新增段无 mermaid。

---

## 2026-07-03: 深化 [[longcat_2_analysis]] —— 补官方原图(LSA/N-gram/MOPD) + 读图问答 + 开源状态 + 完整大纲审计 + MOPD 订正

**Type**: Enrich（应用户连续追问：架构更细解读 / 是否有开源代码 / 放原图并结合图说明「LSA 复用是缓存还是重算」/ 完善缺失内容。源忠实——官方架构图与图注为一手证据，纠正二手误读。提交 `305dac1`）

- **补 3 张官方原图**（美团 S3 CDN 下载 SVG → 本库无头 Edge 管线 2× 转 PNG、白底，与既有页一致；SVG 原图并存作 source）：`assets/lsa_overview.{svg,png}` · `ngram_embedding_overview.{svg,png}` · `mopd_overview.{svg,png}`。渲染工具 `.html2md/svg2png.mjs`（gitignored，不入库）。
- **LSA §2.1 读图深化**：据 LSA 图确证结构——Full KV 分 **Streaming(绿)→Contiguous KV(~50% 预算)** 与 **Non-Streaming(黄)→Block Indexer→Token Indexer 两级 top-k→Non-Contiguous KV(~50% 预算)**，右 **Reuse Layer 无索引器**、标注 "Directly Reusing the Indices from the Owner Layer"。新增「读图问答」回答四问：① streaming token = sink + 近窗连续段（约半预算），非纯滑窗；② 层次化 = 块→token 两级选择（共享参数）；③ CLI = Owner 算一次、多 Reuse 层复用；④ **复用是缓存非重算**（Reuse 层结构上无索引器；缓存的是 top-k 索引集合而非注意力结果，每层仍算自己的 Attn；证据「amortize indexing cost」+ MTP「reusing the index set」）。LSA 动机补为「定点修 DSA Lightning Indexer 的**输出不连续 + 二次方打分**两短板」。
- **N-gram §2.2 读图**：据图补机制——当前位置取 2/3/4/5-gram，各过 Hash+Embedding+Projection（多张哈希表）再与 Base Embedding 相加；动机补「MoE 稀疏度已过甜点区(~97%)、挪 135B 到 N-gram 收益远超标准专家」。
- **[!contradiction] MOPD 订正（源 > 二手，超越本 changelog 2026-07-02 条目的「多目标策略分布」）**：官方架构图副标题为 **"Multi-Teacher On-Policy Distill(ation)"（多教师在线策略蒸馏）**，据此订正早期二手误读「Multi-Objective Policy Distribution」。三组 teacher 原子能力据图列全（Agent: Tool Use/API Parsing/Self-Correction；Reasoning: Multi-Hop/STEM/**Adaptive Computation**；Interaction: 指令遵循/人类对齐/幻觉抑制）。
- **家族 index 补「四、开源状态」**：已核实 **LongCat-2.0 仓库 main 仅 README+LICENSE(MIT)+figures、无 config.json/建模代码/权重**（HF 下载 0、weights coming soon）；**架构前身 LongCat-Flash-Chat 完全开源（79K+ 下载，含 ScMoE / zero-compute experts）**，是当前唯一可读参考实现；2.0 新增的 LSA / N-gram / MOPD 无开源代码。
- **完整大纲审计补缺**：据博客全章节大纲补 §5.2「推理·模型专属优化」（absorb computation / pipelining indexer / KVP / ScMoE 调度）、§5.4 weight prefetch、§8「官方能力演示 3 场景（Codebase Migration / Agentic & Research / Content Generation）」；§9.2 审计确认 layers/dims/heads/vocab/activation/norm/RoPE/数据配比/tokenizer/吞吐 **全部 not stated**（非漏读，已在页内声明）。

**整合**：改动集中在 [[longcat_2_analysis]] 与 [[meituan_longcat/index]]。**校验**：3 张 PNG 均实渲肉眼核对（LSA/N-gram/MOPD 内容正确）；图片用标准 `![](assets/*.png)`（非 mermaid，无定界符风险）；§5 重编号后 §9 未动、页内/index 的「§9」引用仍有效；无新增死链。

---

## 2026-07-02: 新建 [[meituan_longcat/index]] + [[longcat_2_analysis]] —— 美团 LongCat-2.0（1.6T/48B MoE，国产 ASIC 全栈）

**Type**: Ingest（应用户「分析 longcat 2.0 的模型结构、训练、AI infra、低精度、稳定性、效果，录入知识库」。源忠实 + 抓本质）

**源（source-faithful）**：官方 Tech Blog `longcat.chat/blog/longcat-2.0`（**JS 渲染 SPA**，直取仅得标题）→ 经**渲染代理提取 + 三源交叉核对**（渲染博客文本 · HF/GitHub `README.md` · DeepWiki 镜像），并与二手报道对比去伪。Baseline = 访问日期 **2026-07-02**；权重/config.json「coming soon」、正式技术报告未见 → 页头与 §9 已标注保真度与未披露项，待 raw 源到位回填精确基线。

- **新建 family index [[meituan_longcat/index]]**：LongCat 家族（LongCat-Flash 前身 → LongCat-2.0）总览 + 一页速览 + 与 GLM-5/DeepSeek-V3/V4/Kimi-K2 的稀疏注意力/优化器/低精度/硬件定位对照表 + 知识缺口。
- **新建深挖页 [[longcat_2_analysis]]**（主线「在国产 AI ASIC 上把 1.6T MoE 推到近前沿 Agentic Coding」）：
  - **架构**：LSA 稀疏注意力三正交索引（SI 硬件对齐连续访问 / CLI 跨相邻层复用+跨层蒸馏 / HI 块级粗筛→token 细选 training-free）；**N-gram Embedding 135B, n=5**（与 MoE 正交稀疏维扩参、空间约 100×、<10% 预算、降大 batch 解码 I/O）；**ScMoE**（per-core 显式控制→dense/MoE 分支全并行）；**MTP 3-step**（第 2/3 步复用第 1 步 LSA 索引）。
  - **预训练**：>35T tokens；**Muon 大规模**（TP 适配 + DP 状态去冗 + 对称矩阵乘 kernel）；数百亿 token **原生 1M**（all-gather CP 扩到 512+）。
  - **后训练**：**MOPD** 多目标策略分布，融合 **Agent/Reasoning/Interaction** 三组 teacher expert 群蒸馏。
  - **AI Infra**：**6D 并行 = 5D(TP/CP/EP/DP/PP) + EMBP**（专并行 135B N-gram）；superpod ≤48 机 all-to-all + 跨 pod RoCE（+30%）+ 总体 +35% 吞吐；推理 **PD 分离**（prefill CPP+Attention SP / decode KVP+EP128，KV 走 200Gbps 网卡）；**Super Kernels + L2 预取 + EPLB**。
  - **低精度**：[!contradiction] **博客不讲 FP8/FP4 量化**——其「精度」叙事是国产 ASIC 上的**数值可靠性/确定性**（确定性算子覆盖 Embedding/FA/LSA/MoE + 二叉树分段累加降 FP 误差 + 对齐高精度基线验证）。与 DeepSeek-V3(FP8)/GLM-5(INT4 QAT) 是不同侧面。
  - **稳定性**：>35T tokens **零回滚/无不可恢复 spike**；bit-flip 检测 + 端到端自动故障识别/流量切换/恢复。
  - **效果**：全评测表（LongCat-2.0 vs Gemini 3.1 Pro / GPT-5.5 / Claude Opus 4.6/4.7/4.8）——**SWE-bench Pro 59.5 > GPT-5.5 58.6**、Terminal-Bench 2.1 70.8、GPQA-diamond 88.9；整体落后 Claude Opus 4.8。
  - **§9 源忠实修正**：[!contradiction] 二手报道称「动态激活 33–56B / zero-compute experts」——博客只提训练期 padding→zero-expert（省显存），激活即 ~48B，疑似把 LongCat-Flash 机制张冠李戴；[!contradiction] 训练算力 README「加速器·小时」vs 博客渲染「天」24× 分歧，FLOPs 粗算支持「小时」。

**整合**：[[01_theory/01_models/index]] 新增「LongCat / Meituan」家族区；[[index]]（总索引）模型行 28→30、加「LongCat (美团)」子行与「按主题查找」条目、更新日期至 2026-07-02；两新页与 [[01_glm_5_analysis]]/[[12_deepseek_v3_analysis]]/[[13_deepseek_v4_analysis]]/[[11_kimi_k2_analysis]]/[[11_muon_analysis]]/[[14_expert_parallel_analysis]] 等互链。**校验**：全用 ASIC——图表用 **ASCII**（与 GLM-5/Kimi 同系列风格，零 mermaid 定界符风险）；跨链目标经 grep 核对；因源为渲染提取，数值保真度与未披露项已在页头/§9 显式声明，不臆造未披露量。

---

## 2026-07-01: 新建 [[06_distributed_parallelism/index]] 分布式并行原理簇 —— 原语→DP→TP/SP/CP→EP→PP→ZeRO 全景（理论层）

**Type**: New（应用户"在 01_theory 加分布式并行原理解读，从分布式原语→TP→EP→PP→ZeRO 等基本概念；演示图用 SVG→PNG"。抓本质 + 引擎无关的原理层，与已有工程页分工）

**定位**：新建理论簇 `01_theory/06_distributed_parallelism/`，**原理（principle）层、引擎无关**——只讲「为什么这么切、代价函数长什么样、为什么不选替代」，两根主线贯穿全簇：**$\alpha$-$\beta$ 通信代价模型** + **显存账本（参数/梯度/优化器态/激活）**；「源码怎么实现」一律交叉链接到 [[02_engineering/index]] 已有的源级页（`[[15_distributed_primitives/index]]`、[[megatron-lm/index]]、[[torchtitan/index]] 等），不重复。填补「理论层无分布式并行原理页」的空白。

- **新增 index + 6 内容页**：
  - [[10_collectives_analysis]] — 六大原语语义、$\alpha$-$\beta(-\gamma)$ 模型、核心恒等式 **all-reduce = reduce-scatter + all-gather**、ring 每卡搬运 $2(N{-}1)/N\cdot M$ 的带宽最优性、ring vs tree、all-to-all/p2p 代价（全簇「代价词汇表」）。
  - [[11_data_parallel_analysis]] — DP：复制模型/切数据、all-reduce 梯度的等价性、通信 $\propto\Psi$ 与 batch/卡数无关、$16\Psi$ 显存账本（引出 ZeRO）、分桶重叠 + 梯度累积。
  - [[12_zero_fsdp_analysis]] — ZeRO 1/2/3 逐级切优化器态/梯度/参数、通信 vs DP 增量（1/2 免费、3 多 ~50% AG）、ZeRO-3 = FSDP 的 unshard→compute→reshard。
  - [[13_tensor_sequence_parallel_analysis]] — TP（Megatron 列切→行切 + f/g 共轭算子、每层 4 次 all-reduce、只敢机内）、SP（拆 all-reduce 为 RS+AG，零额外通信换激活显存）、CP（ring-attention 交换 KV 攻长序列）。
  - [[14_expert_parallel_analysis]] — EP：路由 + 两次 all-to-all（分发/回收）、负载不均与容量因子、分层 a2a。
  - [[15_pipeline_parallel_analysis]] — PP：microbatching、气泡率 $(P{-}1)/(m{+}P{-}1)$、GPipe vs 1F1B（同气泡、显存 $\propto m$ vs $\propto P$）vs interleaved（真降气泡）、zero-bubble。
- **演示图 9 张 SVG→PNG**（手绘 HTML+SVG，走 `.html2md/render_figs.mjs` 无头 Edge 2× 截图）：六原语语义、ring all-reduce 分解、DP 数据流、Megatron 列/行切+f/g、TP+SP 激活切分、ring-attention、EP 三段 a2a+负载不均、GPipe/1F1B 甘特气泡对比、ZeRO 0/1/2/3 显存分区、N 维正交布局（DP2×PP2×TP4）。原理演示图统一走 SVG（按用户约定：代码调用/类/逻辑图才用 mermaid）。

**工具改动**：`render_figs.mjs` 加 `FIGS_OUT` 环境变量支持自定义输出目录（默认仍指 GLM assets，向后兼容），本簇渲染到 `06_distributed_parallelism/assets/`。

**整合**：[[01_theory/index]] 子领域表加「06 分布式并行原理」一行；`[[15_distributed_primitives/index]]` 与 [[06_auto_parallel/index]] 各加回链（理论↔实现互指）。**校验**：9 张 PNG 逐张实渲肉眼核对（SVG 经 Edge 所见即所得，天然规避 mermaid 定界符坑）；本簇内 `[[链接]]` 与指向工程页的跨域链接经 grep/文件核对存在。

---

## 2026-06-30: 新建 [[35_inductor_memory_allocation_guide]] + 深挖页补「池大小如何确定」—— 吸收外部专家报告

**Type**: Ingest + Enrich（应用户"把外部报告 `deep-research-report.md` 的原理分析风格吸收进库 + 回答 pool 初始化大小如何确定 + 补例子/演示图"。源忠实 + 抓本质）

**源（source-faithful）**：pytorch @ `5f6df46744a`，逐行核对 `codegen/memory_planning.py`（`AllocationPool`/`get_symbolic_size`/`allocate_at_end`/`codegen_create`）、`c10/core/AllocatorConfig.h:16-24`（段大小常量）、`c10/cuda/CUDACachingAllocator.cpp:3063/3697`（`round_size`/`get_allocation_size`）、`codegen/wrapper.py:1520`（`alloc_from_pool`=`torch.ops.inductor._alloc_from_pool`）、`torch/csrc/inductor/inductor_ops.cpp:36/129`、`test/inductor/test_memory_planning.py:108-142`（真实 codegen 实例）。

- **深挖页 `inductor_memory_management_analysis`（历史活链接，该页已于 2026-07-30 判重删除并入 [[12_buffer_liveness_memory_planning_and_reuse_analysis]]，按"历史不回写"惯例降级为反引号）新增**:
  - **§2.6 池的初始化大小如何确定**:Inductor `AllocationPool` 大小=编译期 `root.get_symbolic_size()`（`TemporalSplit` 取最大、`SpatialSplit`=`align(left)+right`）、`allocate_at_end` 末尾追加扩容、`codegen_create` 出扁平 1-D buffer;带 `test_memory_planning.py` 真实实例（`pool1 = empty_strided_cuda((4*s27*s77 + align(4*s77*s77),),(1,))` + 两个 `alloc_from_pool`）+ **字节布局 ASCII 图**。
  - **§3 物理段大小**:`empty_strided` 落 `CUDACachingAllocator` 后按 `get_allocation_size` 取段——≤1 MiB→2 MiB、1–10 MiB→20 MiB、≥10 MiB→2 MiB 倍数（`AllocatorConfig.h:16-24`），解释 `reserved` 远大于 `allocated` 的原因。
- **新建 guide [[35_inductor_memory_allocation_guide]]**（吸收报告骨架：角色边界→分配全过程 sequence 图→分配器对照表→`memory_stats`/snapshot 实测复现→实践建议）。
  - [!correction] **订正原报告 4 处**（源 > 报告）：① 池化 `memory_planning` 非默认、仅 inference（默认是逐 buffer 复用）;② 实验开关应 `memory_planning=True` 而非 `memory_efficient_fusion`;③ 数分配次数用 `allocation.all.allocated`/`segment.all.allocated` 而非 `allocation.all.current`;④ `expandable_segments` 是 native 子开关、非独立后端。报告对 `alloc_from_pool` 的描述确认正确。

**整合**：[[04_inductor/index]] 新增两页条目;两页互相回链;深挖页 §2.6/§3 增补。**校验**：所有新 `file:line`/常量值本会话开文件核对（含 `AllocatorConfig.h:16-24` 数值、`test_memory_planning.py:108` 的 `@config.patch(memory_planning=True)`）；guide 的 1 个 sequenceDiagram 按规范扫（消息无 `[]`/`()`/`|`，participant 用英文 id + alias）；字节布局用 ASCII 非 mermaid。

---

## 2026-06-30: 新建 `inductor_memory_management_analysis`（历史活链接，该页已于 2026-07-30 判重删除并入 [[12_buffer_liveness_memory_planning_and_reuse_analysis]]，按"历史不回写"惯例降级为反引号） — torch.compile 内存分配管理(全栈三层)

**Type**: New（应用户提问"torch.compile 的 memory alloc 管理怎么做"→评估知识库覆盖发现"零件散在 3 个域、无统一脊柱、cudagraph_trees 与 codegen 复用链是短板"→开新页补齐。源忠实 + 抓本质）

**源（source-faithful）**：pytorch 本地 checkout @ `5f6df46744a`（trunk, 2026-06-29）。两个并行 writer-agent 分别深挖**编译期**（`codegen/wrapper.py`·`_inductor/memory.py`·`codegen/memory_planning.py`·`config.py`）与 **CUDA Graphs**（`cudagraph_trees.py`·`cudagraph_utils.py`·`compile_fx.py`），每条 `file:line` 开文件核对；coordinator 抽检 `wrapper.py:2480`、`memory.py:1016`、`config.py:252-268`、`cudagraph_trees.py:2301-2302` 全部吻合。

- **新增** `inductor_memory_management_analysis`（历史活链接，该页已于 2026-07-30 判重删除并入 [[12_buffer_liveness_memory_planning_and_reuse_analysis]]，按"历史不回写"惯例降级为反引号）：主线"三层叠加"——
  - **层 1 编译期**：默认 `memory_plan_reuse` 两遍把 `Allocate`+`Free` 改写成 `Reuse`（峰值感知 `should_reuse_buffer`；同形状指针别名 / 异形状 `reinterpret_tensor`，`wrapper.py:2436/956/4043`）；scheduler `compute_last_usage`+`free_buffers` 决定释放时机（`scheduler.py:8731/8742`）；`reorder_for_peak_memory` 多拓扑序选最低峰值（`memory.py:1016`，扫描线估峰）；可选池化 `MemoryPlanner` 时分/空分打包（`memory_planning.py:675`，`memory_planning` 默认关）。
  - **层 2 运行期**：`empty_strided` 落 `CUDACachingAllocator` block/segment 缓存池（复用既有深页 [[10_caching_allocator_autocast_profiler_analysis]]，强调"编译期逻辑复用 + 运行期物理复用叠加"）。
  - **层 3 CUDA Graphs**：`cudagraph_trees` 跨图共享 `graph_pool_handle` 私有池 + 地址稳定（static/managed idx，`:1019/1932`）+ checkpoint 重建分配器簿记（`:3135`）+ graph partition 切出 cudagraph-unsafe 算子（`scheduler.py:8856`）。
  - [!correction] 据 `5f6df46744a` **订正 `scheduler_analysis`（历史活链接，该页已于 2026-07-30 判重删除并入 [[13_scheduler_dependency_graph_fusion_and_ordering_analysis]]，按"历史不回写"惯例降级为反引号）两处行号**：`reorder_for_peak_memory` 实定义在 `memory.py:1016`（非 `scheduler.py:2986`）；`mutation_renames` 在 `scheduler.py:4197/4770`（非 `:2913-2928`）——符号名对、行号随版本漂移；并澄清 `memory.py` 是"区间+扫描线估峰驱动重排"而非区间图着色（着色在 `memory_planning.py`，默认关）。

**整合**：[[04_inductor/index]] 概览区新增本页；`PyTorch_Inductor_Technical_Analysis`（§6/§7 概念版，该页已于 2026-07-30 判重删除，§6/§7 被本页确认冗余未迁移，按"历史不回写"惯例降级为反引号）、[[10_caching_allocator_autocast_profiler_analysis]]（层 2）各加回链；本页另链 `scheduler_analysis`（历史活链接，该页已于 2026-07-30 判重删除并入 [[13_scheduler_dependency_graph_fusion_and_ordering_analysis]]，按"历史不回写"惯例降级为反引号）/[[20_inductor_codegen_analysis]]/[[21_control_flow_capture_analysis]]。**校验**：3 个 mermaid 块按本库规范逐条扫（subgraph 标题无 `[]`/`|`、各 `end` 单独闭合、连线标签无引号/括号/`|`、节点标签无裸 `[]()`）；交叉链接目标经 grep 确认存在。

---

## 2026-06-30: 新建 [[21_control_flow_capture_analysis]] — Dynamo 控制流捕获两条路径(HOP 投机子图 vs 原生字节码特化)

**Type**: New（应用户提问"torch.compile 编译流程里 cond 入图怎么做的" → 追问"是否覆盖所有控制流入图情况" → "总结一个章节专门介绍控制流"。源忠实 + 抓本质）

**源（source-faithful）**：pytorch 本地 checkout @ `5f6df46744a`（trunk, 2026-06-29），逐一开文件核对 `torch/_dynamo/variables/higher_order_ops.py`、`torch/_dynamo/symbolic_convert.py`、`torch/_higher_order_ops/cond.py`、`torch/_inductor/ir.py` 的引用行。

- **新增** [[21_control_flow_capture_analysis]]（02_dynamo deep dive）：核心论点——Dynamo 对控制流有**两条互不桥接的路径**。
  - **路径 A 显式 HOP**：`speculate_subgraph`（`higher_order_ops.py:2004`）统一引擎四步（开子 tracer→内联→freevar lifting→收尾）；`cond` 深挖（常量谓词特化短路 `:2419`、checkpoint/rollback 投机两分支 `:2475-2552`、`_merge_graph_inputs` 合并签名 `:1287`、`_ALLOW_FALLBACK_TO_EAGER=False` 禁 graph break `:2378`）；控制流 HOP 家族表（cond/switch/while_loop/map/scan/associative_scan，子图结构均经投机/install 锚点核对）；下游 dispatch（`cond.py:403/408/710` Proxy/Fake/functionalize + `ir.py:10700` `Conditional`）。
  - **路径 B 原生控制流**：`generic_jump`（`symbolic_convert.py:714`）四种结局（常量拍平/SymBool guard 特化/数据依赖切图/`fullgraph` 硬报错）；`FOR_ITER`（`:2485`）循环展开。
  - [!correction] **纠正常见误解**：Dynamo **不会**自动把数据依赖 `if` 转成 `cond`——源码里只有"切图"或"报错提示手写 `torch.cond`"两条出路（`symbolic_convert.py:769`/`:937`）。

**整合**：`[[02_dynamo/index]]` 页面列表新增本页；`PyTorch_Dynamo_Technical_Analysis`（该页已于 P4 Task 5 判重删除，内容并入 [[02_compile_stack/01_dynamo/index]]）Related Pages 加回链。**校验**：所有 `file:line` 均本会话内开文件核对；3 个 mermaid 块按本库规范逐条扫（标签无裸 `[]()`、特殊形状无嵌套定界符、连线文字无引号/括号/`|`）；交叉链接目标经 glob 确认存在。

**追加（同日，应用户连续追问澄清编译期/运行期边界）**：
- `PyTorch_Dynamo_Technical_Analysis`（该页已于 P4 Task 5 判重删除，内容并入 [[02_compile_stack/01_dynamo/index]]）§6.6「动态控制流」加 `> [!deprecated]` 指引转向本页（原演示内容按 never-delete 保留）。
- 本页新增 **§2.5「trace 两支 / 编译两支 / 运行只跑一支」**：拆解三个常见误解——① 「捕获条件」是把 `pred` 接成 cond 节点运行时输入（`pred.as_proxy()` `:2588`），非 trace 期选支；② 「Dynamo 编译两个子图」不准——Dynamo 只 trace、产 **1 张父图**（两子图为嵌套 `GraphModule`），编译成 kernel 是下游 Inductor 一次编译产两段；③ 按 pred 选支是 cond 算子 lowering 在**运行期**做（`cond_op_dense` `cond.py:310-313`）。附 `cond` vs graph-break 六维对照表。新增锚点 `cond.py:301-313`、`higher_order_ops.py:2588` 均本会话开文件核对。

---

## 2026-06-29: 新建投机推理专题 [[speculative_decoding/index]] — DSpark 论文 + DeepSpec 开源仓 + MTP→DFlash→DSpark 演进

**Type**: Ingest（应用户"分析 dspark 论文原理 + 结合开源 dspark 仓 + 总览投机推理演进 mtp/dflash/dspark 区别，归纳入库"。源忠实 + 抓本质）

> [!correction] **arXiv 编号订正（源 > 转述）**：用户给的 **arXiv:2606.19348 经核对是 DeepSeek-V4 模型论文**（本库已审计，见 [[30_deepseek_v4_audit_analysis]]），**不是 DSpark**。DSpark 是挂在 V4 checkpoint 上的投机解码草稿模块，其论文以 `DSpark_paper.pdf` 随开源仓 **`github.com/deepseek-ai/DeepSpec`** 发布（标题 *DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation*，Cheng et al., 北大+DeepSeek-AI）。HF 模型卡 `DeepSeek-V4-Pro-DSpark` 引用 2606.19348 指的是**底座模型**。

**源（source-faithful）**：克隆 `DeepSpec` @ `dd854392`（main, 2026-06-28）到 `E:\97-codes\torch_parallel\DeepSpec`，PDF 抽成页码标记文本逐页核对；论文公式与代码 `file:line` 双向交叉核对（Eq.5 ↔ `markov_head.py:8`、Eq.7 ↔ `modeling.py:268/293`、Eq.8 ↔ `loss.py:69`、Eq.2-3 ↔ `modeling.py:241/104-113`）。

- **新建目录** `wiki/02_engineering/03_infer_frameworks/speculative_decoding/`（3 页）：
  - [[index]]（本专题总览/演进survey）：投机解码在 `L=(T_draft+T_verify)/τ` 上的三代演进——① 自回归（MTP/Eagle3，升 τ 但 T_draft∝γ）→ ② 并行（Medusa/DFlash，降 T_draft 但后缀崩塌）→ ③ DSpark（半自回归补 τ + 置信度调度降有效 T_verify）。四代横向对比表 + 三者区别本质。
  - [[dspark_analysis]]（论文深挖，exemplar）：两大部件——**半自回归生成**（并行 DFlash 骨干 + Markov/RNN 串行头，Eq.4-6；接受长度相对 Eagle3 +30.9%、DFlash +16.3%）+ **置信度调度验证**（置信头 Eq.7-8 + STS 校准 + 硬件感知前缀调度器 Alg.1，按 SPS 负载曲线全局贪心、早停保无偏）。生产相对 MTP-1 提速 60%–85%（V4-Flash）/57%–78%（V4-Pro）。
  - [[deepspec_codebase_analysis]]（源码级）：一套 `Qwen3DSparkTrainer` 同产三草稿模型——**DFlash = DSpark 关掉串行/置信头的消融**（`config/dflash/*:18-26`，无独立 modeling）；训练前向链、三项损失 ↔ Eq.9-12、推理拒绝采样路径。**关键边界**：开源仓只到「置信头 + 静态阈值裁剪 + bsz=1」，Algorithm 1 多请求调度器/异步 ZOS/变长内核是生产专属。

**整合**：[[03_infer_frameworks/index]] 新增"投机推理"子目录；[[vllm_speculative_decoding_analysis]]（已含 dflash/mtp proposer）加 [[dspark_analysis]] 回链；[[12_deepseek_v3_analysis]]（MTP 起源）、[[13_deepseek_v4_analysis]]（底座模型）各加回链。**校验**：三页所有 `file:line` 已逐一开文件核对；交叉链接经 grep 确认目标存在。

---

## 2026-06-29: 新建 [[24_megatron_linear_cross_entropy_analysis]] — 融合线性交叉熵("chunk loss")源码级深挖

**Type**: New（应用户提问"当前 Megatron 的 chunk loss 如何实现",读 Megatron-LM `dev@232c478d4` 源码后沉淀,带具体源码实现分析）

- **新增** [[24_megatron_linear_cross_entropy_analysis]]:Megatron 版"chunk loss" = `cross_entropy_fusion_impl='linear'` 融合线性 CE。
  - **配置三档**(`model_parallel_config.py:257,262`):`native`/`te` 接收**已物化的 logits**只融 softmax+NLL(`language_module.py:157,180`);**`linear`** 把 LM-head matmul 也融进核、logits 从不物化。
  - **选路**:`gpt_model.py:157-160` 算 `fuse_linear_cross_entropy`,`:263` 把输出层换成 `LinearCrossEntropyModule`,`:799-802` 以 `output_cross_entropy_loss=True` 直接吐 loss。
  - **省显存本质**(`fused_linear_cross_entropy.py:161-181/197-223`):`save_for_backward` 只存 `hidden + max + sum-exp`(各 O(N))、**不存 logits**,反向从统计量按块重算 → 峰值 `O(N·V)→O(N·d)+O(N)`。
  - **Blackwell 融合核**(`linear_cross_entropy/blackwell/`):`entry.py:147-151` 按 `vocab_per_split=512` 切 `num_splits` 块、逐块 online-softmax(`fwd_mainloop.py:40-53`),`:246/253` 跨 TP `all_reduce(MAX/SUM)`;**硬件门控仅算力 10.x**(`:34-40` 非 Blackwell `raise`)——[!warning] 标注。
  - **对照** MindSpeed `chunk_loss`(序列维框架层 autograd,可移植 NPU)vs Megatron `linear`(词表维 kernel 融合,绑 Blackwell);同属 Flash-Attention 式"online-softmax + 不物化大矩阵 + 反向重算"。

**整合**:[[megatron-lm/index]] 专题深挖区(融合算子项下)新增条目;[[12_mindspeed_memory_optimization_analysis]] §8 chunk-loss 增"跨框架对照"回链。**校验**:`model_parallel_config.py:257/262`、`gpt_model.py:157-160/263/799-802`、`fused_linear_cross_entropy.py:34-40/161-181`、`blackwell/entry.py:147-151/246/253`、`language_module.py:157/180` 均逐一开文件核对;交叉链接 4 个目标经 find 确认存在。

---

## 2026-06-26: 新建 [[cuda_execution_model_guide]] — Grid·Block·Warp·Thread·SM 执行模型（概念→深入）

**Type**: Ingest（应用户"Grid→Block→Warp→Thread→SM 这条映射链不清楚、会阻塞 GPU 编程理解，从概念到深入解释并入库"。**铁律：真实可靠 + demo**）

**源（source-faithful）**：以 **NVIDIA CUDA C++ Programming Guide v12.9.1（archive）** 为权威源——`§Thread Hierarchy`（Programming Model）+ `§Hardware Implementation / SIMT Architecture`，关键事实逐条 WebFetch 核验后引用（warp=32、Block≤1024 因驻留同一 SM、Block 必须独立执行、warp 一次执行一条公共指令、发散逐路径串行、Independent Thread Scheduling 自 Volta/CC7.0、Cluster 自 CC9.0 最多 8 Block）。**注意**：v13.3 已把单页指南拆成多页，原 `index.html` 仅剩目录；故锚定有完整内容的 archive v12.9.1。

- **新建** `wiki/02_engineering/05_gpu_kernel/cuda_execution_model_guide.md`：
  - 主线：逻辑层级（Grid→Block→Thread，你写的）↔ 物理层级（GPU→SM→Warp，硬件跑的）互相映射，钥匙是 **Warp**。
  - 概念层（公司类比 + 索引变量 + Thread ID 公式）→ 物理层（Block→SM 驻留、SM→Warp 切分）→ 深入层（warp 事实派生：①分支发散 ②合并访问 ③占用率 + ④`__syncthreads` 仅 Block 内 ⑤Block 独立=可扩展）→ 映射到 Triton（program≈block、num_warps、threadIdx 不可见）→ 常见误解纠正（以源为准）。
  - **3 个可运行 demo**：`whoami.cu`（printf 看 Block 被切成 32 一组 warp）、`devinfo.cu`（`cudaGetDeviceProperties` 查真实 SM 数/warpSize/上限）、`triton_whoami.py`（`TRITON_INTERPRET=1` 看 program_id≈blockIdx，无 GPU 可跑）。
- **整合**：`05_gpu_kernel/index.md` 页面列表新增本页（置于 gpu_kernel_guide 前，作地基）；[[triton_00_gpu_essentials_guide]]（正文执行层级处 + 相关页面）与 [[triton_01_programming_model_guide]]（相关页面）双向补链；`wiki/index.md` GPU Kernel 计数 10→11 + 主题导航新增「GPU 执行模型」行。
- **校验**：本页外链（gpu_kernel_guide / triton_00 / triton_01 / triton_04 / triton_05 / index）均指向已存在页；**0 悬挂链**。

---

## 2026-06-26: 新建「Triton 学习路线」系列(9 页) — 小白→会写·会调·会优化·会debug 全能专家

**Type**: Ingest + New domain（应用户"以 Triton 为切入点，整理 GPU 编程要素 + Triton 路线，手把手从小白到全能专家，输出学习资料入库"。**铁律：内容真实可靠、每教程带可运行 demo**）

**源与方法（source-faithful）**：父目录无 Triton checkout，故按本库「上游 checkout 放父目录」惯例**浅克隆官方 `triton-lang/triton` 到 `../triton`**，钉死基线 **`main @ 70e0929`（2026-06-25）, v3.8.0**，作 `file:line` 可核验定位符。每个 demo 逐字锚定官方 tutorial，绝不凭记忆写。

- **新建子目录** `wiki/02_engineering/05_gpu_kernel/triton/`，9 页：
  - [[index]] — 学习路线总索引（主线：Triton=block-level 编程，编译器自动管 coalescing/shared mem/warp 划分；四能力闭环图）
  - [[triton_00_gpu_essentials_guide]] — L0 地基：执行/内存层级 + roofline；**demo 用官方 benchmark 的 GB/s vs TFLOPS 公式手算算术强度判 bound**（锚 `01:128`/`02:225`/`03:438`）
  - [[triton_01_programming_model_guide]] — L1 会写①：SPMD 五件套；向量加法（锚 `01-vector-add.py:29-75`）。**协调者自写的校准 exemplar**
  - [[triton_02_fused_softmax_guide]] — L1 会写②：reduction + fusion 省带宽（锚 `02-fused-softmax.py:42-174`）
  - [[triton_03_matmul_guide]] — L1 会写③+优化：多维指针算术/`tl.dot`/fp32 累加器/L2 grouping（锚 `03-matrix-multiplication.py:232-320`，A100 220→245 TFLOPS @ `:145`）
  - [[triton_04_autotune_guide]] — L2 会调：`Config`/`num_warps`/`num_stages`/`key`（锚 `runtime/autotuner.py:351,334-340,408` + matmul `:228-231`）
  - [[triton_05_debug_guide]] — L3 会debug：`TRITON_INTERPRET`（CPU 串行模拟，`knobs.py:471`/`interpreter.py:1410`）+ `device_print`/`static_print`/`static_assert`/`device_assert`（`core.py:3398/3414/3428/3478`）；越界 bug→修复 demo
  - [[triton_06_optimization_profiling_guide]] — L4 会优化：roofline 驱动 + proton（`09-persistent-matmul.py` 真实用法）+ FlashAttention online-softmax（锚 `06-fused-attention.py:69-110`，HBM 流量 O(N²)→O(N·d)）
  - [[triton_knowledge_map]] — 总纲：四能力知识点清单 + 分级自测 + 进阶（tutorials 04-11 + gluon）+ 真实资源
- **生产方式**：协调者自写 index/00/01/knowledge_map + 5 个并行 writer-agent（严格契约：各锚定指定真实 tutorial 文件、以 01 为模板、mermaid 图、demo 忠实源 API）。**抽查定位符均真实**（`06:84 alpha=tl.math.exp2`、`autotuner.py:351` 默认值、`05-layer-norm.py` 锁式并行归约）。
- **整合**：更新 `05_gpu_kernel/index.md`（新增 Triton 表）、`wiki/index.md`（GPU Kernel 计数 1→10 + Triton 子条目 + 主题导航）；交叉链向 [[gpu_kernel_guide]]/[[30_triton_vs_mlir_backend_analysis]]/[[20_inductor_codegen_analysis]]/[[21_inductor_autotuning_analysis]]/[[26_flex_attention_analysis]]。
- **校验**：9 页内部 `[[链接]]` 全部互指存在页；外链目标均已存在；**0 悬挂链**。

---

## 2026-06-25: 清理 `Megatron-LM_Distributed_Parallel_Exam` 遗留悬挂链(11 处/9 文件)

**Type**: Maintenance（该页早先已删除、内容分发至各分析页;全库尚残留 11 处指向它的悬挂链,逐条按主题重指到真实后继页）

- **重指(按内容去向,非一键)**：
  - 重计算/卸载/resharding 类 → [[18_megatron_recompute_analysis]]（activation_checkpointing 的 Q12/Q13/Q30）
  - FP8 / CUDA Graph 类 → [[23_megatron_precision_cudagraph_fusion_analysis]]（low_precision Q15、transformer_engine Q14/Q15）
  - Muon/Layer-Wise 优化器 → [[16_megatron_distributed_optimizer_analysis]]（11_muon_analysis）
  - 泛 5D 并行综合 → [[17_megatron_parallelism_orchestration_analysis]]（distributed_optimizer/memory_optimization/moe_training/tflops 的「相关页面」、`wiki/index.md`「Megatron 分布式」导航）
- **散文出处改写**：low_precision §4.1、transformer_engine §10 的「来自 `...Exam.md` Q19/Q14：」改为中性引导句（删去已失效的源文件名,内容保留）。
- **校验**：全 wiki **0 处**仍指向 `Megatron-LM_Distributed_Parallel_Exam`（changelog 4 条历史记录按惯例保留）；5 个后继目标页均存在,**未引入新 dangling link**。

---

## 2026-06-25: DeepSeek-V4 工程页(TP/CP)整理入 megatron-lm/ + 双向交叉链接 + 源码审核

**Type**: Reorg + Audit（应用户"把 02_train 下重复的 deepseek-v4 内容合并/挪过来,模型分析放一起,重复删除、不重复挪,并审核内容"）

**核查结论(先判定再动)**：`02_train_frameworks/` 下两篇 V4 页面经核查**不是重复**——它们是**真实的 Megatron-LM 源码级框架分析**（每节带 `megatron/core/...py:line`，与本地 `../Megatron-LM` 源码核对通过），与论文级模型页 [[23_deepseek_v4_cp_analysis]] **角度互补**（论文*算法* ↔ 框架*实现*）；TP 页在模型侧无对应。故**无可删的真重复**；缺的是两侧**互相没有交叉链接**。按 wiki「理论/工程」分层 + 用户选定「留工程层 + 双向交叉链接」执行：

- **移库（`git mv` 保留历史）**：`34_deepseek_v4_tensor_parallel_analysis.md`、`35_deepseek_v4_context_parallel_analysis.md` + 7 张图 `assets/deepseek_v4_*_fig*.png` 从 `02_train_frameworks/` 移入 `02_train_frameworks/megatron-lm/`（与其余 Megatron 页同处；`assets/` 相对引用保持有效）。
- **补基线头**：两页原**无 commit 基线**，补 `源基线: Megatron-LM dev @ 232c478d4 (2026-06-16)` + 维度(工程实现) + 与模型页分工说明。
- **补 `## 相关页面`**：两页原**缺交叉引用节**（违反 wiki 规则），补全——双向链接模型页（[[23_deepseek_v4_cp_analysis]]/[[13_deepseek_v4_analysis]]/[[26_deepseek_v4_technical_deepdive]]/[[25_mhc_analysis]]）+ 同目录 Megatron 页。
- **模型页反向链接**：[[23_deepseek_v4_cp_analysis]] 的「相关页面」新增「框架实现」小节，指向两篇工程页（论文算法 ↔ Megatron 实现对照）。
- **索引**：`02_train_frameworks/index.md` 删去两条目（已移子目录）；`megatron-lm/index.md`「专题深挖」表新增两条目（带跨目录指回模型页）。
- **内容审核（对照 `../Megatron-LM` @ 232c478d4 抽查）**：✅ `deepseek_v4_hybrid_attention.py:92` `get_pg_size(tp)==1`、`:447` `parallel_mode='duplicated'`；✅ `hyper_connection.py:193/243` mHC 用 `nn.Linear`+`sequence_parallel`（非 TP-sharded）；✅ `experts.py:346` routed-expert `tp_group.size()>1` 约束；✅ `shared_experts.py:123` 标准 TP；✅ §九 特征5「CSA+CP 两阶段压缩 KV all-gather 尚未实现」仍成立（`csa.py` 的 `cp_group` 仅用于 RoPE CP 感知，page §5.1 已记，**非**压缩 KV 通信）。结论：内容真实、源码 grounded、非臆造非重复；仅**行号随源码漂移数行**（已在页头标注）。

---

## 2026-06-25: DeepSeek-V4 全系列对正式发表版 arXiv:2606.19348v1 审计 / 核对 / 订正

**Type**: Reconciliation（应用户"分析 deepseek v4 输出一份报告，文章地址 arxiv 2606.19348"，并选定「审计 + 核对 + 订正」。既有 ~7 篇 V4 页面是论文正式上 arXiv 前 2 天(2026-04-24)基于无编号预发布 PDF / AI 合成笔记写成，故以正式版逐项复核）

**方法**：下载正式版 PDF → 与本地预发布 `raw/.../DeepSeek_V4.pdf` 双双抽成页码标记文本 → diff + 逐项核对超参/基准/章节/机制；4 个并行只读审计 agent 锚定 `GROUND_TRUTH` 事实表逐页查证，关键臆造断言由协调者亲自 grep 正式版复核（`DualPath`=0、`Highly Compressed`=0 vs `Heavily`=7、`task_classifier`=0、`n log n`=0、`INT8`=0 vs `MXFP4`=1、`ablation`=0）。

- **新增** [[30_deepseek_v4_audit_analysis]]：审计报告（核对基线 arXiv:2606.19348v1, 2026-04-26）——逐页裁决表 + 核对通过事实(超参/效率/基准全一致) + 章节号位移映射 + 论文中**不存在**的臆造清单(按出处反证)。
- **核对通过(数字全对)**：超参表(层/维/专家/压缩率)、头条效率(Pro 27%/10%、Flash 10%/7%)、Table 1 基座(MMLU-Pro 65.5/68.3/73.5)、Table 6 后训练(LiveCodeBench 93.5、MRCR 83.5)。
- **订正(Tier-A，论文真源页面)**：
  - [[13_deepseek_v4_analysis]] —— 重订基线头；FP4 标注为**后训练 §5.2.1**(两组件 FP4 + 索引分数 BF16)；正文臆造的「DualPath 推理框架」加 `> [!contradiction]` 标注；评测表「顶尖」措辞按 Table 6 校准为有基线的相对表述；「51×/2048×」标注为本页推导。
  - [[23_deepseek_v4_cp_analysis]] —— **章节号位移订正**(CP §3.5.3→§3.4.3、推理框架 §3.6/§3.6.1/§3.6.2→§3.5/§3.5.1/§3.5.2、Muon §3.4.1、mHC §3.4.2，共 ~13 处)；重订基线头；footer 旧路径修正。
  - [[24_deepseek_v4_fp4_qat_analysis]] —— 出处 §3.7(两版皆无)订正为 **§5.2.1**；旧路径 `raw/05_model_families/`→`raw/01_theory/01_models/`；「三个组件 FP4」订正为「2 组件 FP4 + 索引分数 BF16」；效果表 ~75%/~2× 标注为估算(论文仅明述 top-k 2×、99.7% 召回)。
  - [[25_mhc_analysis]] —— 基本一致；补注消融数据源自 mHC 论文(arXiv:2512.24880v2)而非 V4。
- **加警示横幅，建议整页重写(Tier-B，预发布 AI 生成、无任何论文引用，确认系统性臆造)**：
  - [[26_deepseek_v4_technical_deepdive]] —— DSA=CSA+HCA 倒置、Highly/Heavily 名错、HCA 10% vs m′=128、臆造分层调度、臆造 MoE 任务路由、DualPath、MLA「V3 引入」(应 V2)、Sinkhorn 100(应 20)。
  - [[27_deepseek_v4_implementation_deepdive]] —— 专家 128/激活 8(应 256·384/6)、HCA 0.1、Sinkhorn 100、「Muon」实为 Adam(无 Newton-Schulz)、量化写成 INT8(应 FP4)、臆造 PCA KV 压缩/DualPath。
  - [[28_deepseek_v4_architecture_analysis]] —— 128 专家/K=8–12、5%·35% 任务自适应、领域命名专家、O(n log n)、DualPath/SNIC/CNIC、图缺 MTP、CSA/HCA 误并为一层。
- **Tier-A 遵循 wiki「不删除、仅标注」规则**：臆造内容保留原文 + `> [!warning]`/`> [!contradiction]` 标注 + 指向审计报告。
- **Tier-B 整页重写完成**（用户确认"基于正式版全部重写 3 篇"后执行）：3 个并行 writer-agent 锚定核验过的机制简报
  （`MECH_BRIEF`：CSA/HCA/DSA Eq 9–27、Muon Alg 1/Eq 28、mHC Eq 1–8、§4.2.1 配置）逐方程重写，每条断言带 §/Eq/page；
  协调者抽查定位符 + 机械校验（横幅已除、`DualPath`/`INT8`/`Highly`/`task_classifier`/`n log n` 清零、跨链无悬挂、图内无 `[[…]]` 泄漏）。
  - [[26_deepseek_v4_technical_deepdive]]（288 行）—— CSA/HCA/DSA/MLA 四机制「动机→机制(LaTeX)→证据→为何不选替代」对比。
  - [[27_deepseek_v4_implementation_deepdive]]（362 行）—— 五大组件逐方程伪代码，常量取自 §4.2.1，Flash|Pro 双值。
  - [[28_deepseek_v4_architecture_analysis]]（237 行）—— 复刻 Figure 2/3/4 + §4.2.1 配置表，含 MTP、首-2-层非对称、CSA/HCA 交错。
- **更新** `deepseek/index.md`：V4 专题表加「核对状态」列 + 审计报告入口；3 篇 Tier-B 状态→「✅ 已据正式版整页重写」；最后更新日期→2026-06-25。

---

## 2026-06-25: 仓库自带技能 `.claude/skills/source-faithful-analysis/` — 「源忠实分解」方法论

**Type**: Meta（应用户"把分析方法论作为 llm-knowledge **自带的 skill** 放进仓库，而非写成归档知识目录"。先建 `methodology/` docs，按反馈改为仓库内置 Claude Code 技能）

- 新增**仓库自带技能** `.claude/skills/source-faithful-analysis/`：`SKILL.md`（含 frontmatter，Claude Code 打开本仓库即自动加载）+ `references/{codebase,paper,general,parallel-agent-contract}.md`（按来源类型的定位符/摄入配方/本质清单/专属红旗 + 并行 writer-agent 契约）。镜像全局同名技能（由原 `source-faithful-codebase-analysis` + `source-faithful-paper-analysis` 合并而来）。
- `.gitignore`：`.claude/` 改为 `.claude/*` + `!.claude/skills/`——本地 settings 仍忽略，但**签入仓库自带技能**。
- `CLAUDE.md` 第 4 层与「## Analysis & Decomposition Methodology」节改指该自带技能；CLAUDE.md 管*结构与约定*、技能管*分析与分解的过程*，Ingest/Query Workflow 是其落地实例。
- 取代了本会话早先签入的 `methodology/` 文档目录（已删除）；GLM-5 (2602.15763) 系列即该方法论的范例产出。

---

## 2026-06-24: AI infra 三页补「掩盖 / 缓存」图示（5 张时间线 / 复用图）

**Type**: Update（应用户"AI infra 深挖里涉及掩盖、缓存的优化点都配个图示，便于分析理解"。掩盖用时间线(Gantt 前后对比)、缓存用复用流 / 前缀树 / 内存层）

- [[22_glm5_training_infra_deepdive]] §3.3 新增 **图 3**：计算-通信掩盖时间线——②双缓冲(累积‖梯度同步)、③Muon(本地计算‖分片all-gather)、④激活offload(计算‖搬运)、⑥延迟wgrad(填气泡)、⑦层级all-to-all(节点内‖节点间)，各自把"什么藏进计算"。
- [[24_glm5_agentic_rl_deepdive]] 新增 **图 1**（PD 解耦时间线：混部 prefill 抢占 decode vs 解耦后 decode 连续）+ **图 3**（DP-aware routing 的 KV 前缀复用：朴素 O(总上下文) vs 一致性哈希亲和 O(增量)）；原图 1/2 顺延为 **图 2/4** 以保持阅读序。
- [[26_glm5_low_precision_chip_deepdive]] §4.3 新增 **图 3**：昇腾掩盖与缓存——左 Lightning Indexer/MLAPO(Vector‖Cube)/异步调度(D2H‖decode准备)/FlashComm(拆AllReduce) 计算掩盖访存通信，右 RadixCache 前缀共享 + Prefix Cache KV 外溢到系统内存。
- **工具链**：`figstyle.css` 增加时间线(`.tl`/`.tl-bar`)与内存层(`.tier`)样式；新增图源 `glm5_infra_overlap` / `glm5_agentic_cache` / `glm5_chip_overlap_cache`.html（gitignored），渲染 5 张 PNG 到 `assets/`。**校验**：5 图逐张肉眼查无溢出/残留链接语法；agentic 页图号顺延后阅读序连续。

---

## 2026-06-24: [[20_glm5_architecture_deepdive]] 补完整模型结构图（§1.1，config 实据）

**Type**: Update（应用户"architecture 里缺一张完整模型结构图"。拉取 released `zai-org/GLM-5` config.json 作实据，新增"宏观层栈 + 单 MoE 解码层放大"的结构图 + config 超参表 + 层数 contradiction）

- 新增 §1.1「完整模型结构（GlmMoeDsa）」+ 图 `assets/glm5_architecture_fig3.png`：左=宏观栈(Embedding→Dense×3→MoE×75→Final RMSNorm→LM Head+MTP)，右=单层放大(子层A DSA 注意力：MLA-256 低秩+Muon Split→lightning indexer top-2048→稀疏注意力；子层B MoE：Router→top-8/256+1 共享→加权合并)。
- 超参全部取自 released config：hidden 6,144 · **78 层**(前 3 dense + 75 MoE) · qk/v head_dim 256 · kv_lora 512+rope 64=**576** · 256 专家 top-8 + 1 共享 · DSA index_topk 2,048 · MTP×1。
- `> [!contradiction]`：论文 §2.1 称 **80** 层、开源权重 **78** 层，以权重为准；原 DSA 续训图 caption 顺延为「图 3」。

---

## 2026-06-24: GLM-5 论文逐章深挖补齐 6 篇 + 流程图工具链 + 索引整合

**Type**: New + Deepen（应用户"针对每个章节做 deepdive，解释原理/效果/为什么，并补流程图(SVG→PNG)"。在 [[20_glm5_architecture_deepdive]] 校准页之上，6 个并行 writer-agent 各写一篇深挖页 + 各自流程图 HTML，coordinator 统一渲染 14 图并整合）

- **新增 6 篇深挖页**（源基线 arXiv 2602.15763v2，逐节 原理/效果/为什么 + §/页码引用）：
  - [[21_glm5_data_deepdive]] — §2.2–2.3 数据（双分类器漏斗 + 三段式上下文扩展，2 图）
  - [[22_glm5_training_infra_deepdive]] — §2.4 显存五件套 + 长序列并行（2 图）
  - [[23_glm5_posttraining_deepdive]] — §3.1–3.5 SFT(三思考模式)/GRPO+IcePop/General RL/跨阶段蒸馏（2 图）
  - [[24_glm5_agentic_rl_deepdive]] — §3.6+§4 slime/全异步解耦 RL/三类环境构造（2 图）
  - [[25_glm5_training_stability_deepdive]] — 跨章稳定性主线（失配×噪声×故障：TITO/双边IS/staleness/优化器reset/确定性topk，2 图）
  - [[26_glm5_low_precision_chip_deepdive]] — §2.4.3+§3.6.2+§5 INT4 QAT→FP8→W4A8 + 昇腾三支柱（2 图）
- **流程图工具链**：新增 `.html2md/render_figs.mjs`（复用 Edge/puppeteer 2× 截图）+ `figs/figstyle.css`；图源 HTML 在 gitignored `.html2md/figs/`，14 张 PNG 落 `assets/`（house 风格:奶白卡片 + 彩色圆角节点 + 灰箭头）。
- **整合**：父索引 [[zhipu_glm/index]] 新增「§四之补 GLM-5 论文深挖页矩阵」(7 页表) + §六 GLM-5 行改指矩阵；概要页 [[01_glm_5_analysis]] 补「逐章深挖」Related 段，并对 §五 估算基准加 `> [!contradiction]` 用 Table 7 真值订正（SWE-bench Verified 77.8 / τ²-Bench 89.7 / AA Index 50 等）。
- **校验**：7 页 + 索引/概要的 `[[]]` 链接脚本提取，同系列 7 个 `glm5_*_deepdive` + 既有 [[11_muon_analysis]]/`grpo_analysis`（历史活链接，已于 2026-07-31 因 kb-reorg P5 Task 8 再编号为 [[20_grpo_analysis]]，按"历史不回写"惯例降级为反引号）/[[14_megatron_ep_analysis]]/[[verl/index]]/[[13_low_precision_training_analysis]] 等均存在，0 悬空；14 图 `assets/*` 引用解析正常；agentic_rl 两图 note 内误写的 `[[]]` 已改纯文本并重渲。

---

## 2026-06-24: 新增 [[20_glm5_architecture_deepdive]] 并入 GLM 索引

**Type**: New（GLM-5 架构深挖页:论文 §2.1 的"规模 × 长上下文成本"权衡——744B/40B MoE 扩专家减层、MLA→Muon Split→MLA-256、MTP 参数共享、DSA 两阶段续训与高效注意力消融,含 2 图）

**整合**:父索引 [[zhipu_glm/index]] §六 论文索引 GLM-5 行新增架构深挖链接(概要 [[01_glm_5_analysis]] + 架构深挖 [[20_glm5_architecture_deepdive]])。**校验**:2 张图 `assets/glm5_architecture_fig{1,2}.png` 引用解析正常;`## Related` 段含同系列深挖页前向引用(6 个 `glm5_*_deepdive` 为规划中页面,标记式前向链接)+ 既有页([[01_glm_5_analysis]]/[[11_muon_analysis]]/[[12_deepseek_v3_analysis]]/[[20_deepseek_moe_analysis]])均存在。

---

## 2026-06-23: [[16_megatron_distributed_optimizer_analysis]] 新增 §2.7「bucketing 算法与 overlap 调度」(机制级深挖)

**Type**: Update（应用户提问"Megatron distributed optimizer 如何 bucket、如何调计算与 bucket 让计算 overlap 掉参数通信",现读 Megatron-LM `dev@232c478d4` 源码后补全;既有 §2.1–2.3 只到高层轮廓）

- **新增**(原 `megatron_ddp_optimizer_analysis.md`,2026-07-31 并入 [[16_megatron_distributed_optimizer_analysis]])§2.7,补三件源码层细节:
  - **分桶算法**:逆序贪心 `_compute_default_per_buffer_param_layout`(`param_and_grad_buffer.py:891-939`)—— `params[::-1]` 逆序(backprop 序,末层落 bucket 0)、累计 ≥ bucket_size 即封桶;三级结构 Buffer / Bucket / BucketGroup(后者为一次 NCCL collective 粒度,`_coalescing_manager` 合并)。
  - **bucket_size 调参**:默认 `max(40M, 1M·dp_size)`(`distributed_data_parallel.py:68-69`)、ring 每 rank 报文 = `bucket_size/dp_size`(`..._config.py:61`)、distopt 可分片约束 `numel % dp == 0`(`param_and_grad_buffer.py:1059`)、`pad_buckets_for_high_nccl_busbw` 凑 2^16。
  - **双向 overlap 的 hook 调度**:反向 backward-post-hook → `register_grad_ready` golden-count 满才发异步 RS(`param_and_grad_buffer.py:802-824`);前向 forward-pre-hook(`distributed_data_parallel.py:413`)→ `finish_param_sync` wait 本组 AG + 预取 `next_param_gather_bucket_group`(`param_and_grad_buffer.py:496/:531`),链按前向序串于 `distributed_data_parallel.py:295-308`。附理想时间线 ASCII 图 + 调节点表(bucket_size / 桶序=执行序 / align_param_gather / 头尾暴露)。
  - **补澄清(应用户追问"register_grad_ready 是否先填桶再统一通信")**:`register_grad_ready` 是**就绪计数器而非填数据**——填数据是同一 hook 内前一步 `param.main_grad.add_(param.grad.data)`(`distributed_data_parallel.py:469`,`main_grad` 是扁平 buffer 的视图,梯度原地累加,无"搬进桶"动作);"桶满"= 该组成员梯度全算齐(`per_param == golden`,`param_and_grad_buffer.py:822`),非攒够字节(桶大小/成员初始化即定死);golden count 可 >1(参数被多次消费,首 batch 记录,`:273-276`)。已并入 §2.7 反向 overlap 小段。
- **基线**:Megatron-LM `dev@232c478d4`(2026-06-16);全部 `file:line` 现读现核。**索引**:[[megatron-lm/index]] 加 `> [!update] 2026-06-23` note。**纯增不删**:既有 §2.1–2.6 原样保留。

---

## 2026-06-23: MindSpeed 全 5 篇按「每特性四件套」再深挖(图示 + 优化点 callout + 源码,融合算子说明融合内容)

**Type**: Deepen（用户第三轮反馈"每个都比较浅显,每种优化特性最好补对应图示和优化点说明;融合算子说明融合了哪些内容 + 补源码解读"。先把 affinity 页定为新标尺(每算子四件套:融合内容→before/after 图示→`[!tip] 优化点`→源码解读),再以它为 in-house exemplar 并行重写其余 4 篇。源码核对 @ MindSpeed 1432cb09 / MindSpeed-LLM 0c16322d)

- **[[13_mindspeed_ascend_affinity_analysis]]**(468→**662 行**,10 图,12 优化点 callout):融合算子每个补**融合内容**(N 个散算子→1 核)——GMM(E 次切片+GEMM→1 变长分组,反向 dgrad 累加进 main_grad)、SwiGLU(chunk+SiLU+⊙→`npu_swiglu`)、RMSNorm(x²+mean+rsqrt+×→`npu_rms_norm`)、RoPE(rotate_half+cos/sin→`npu_rotary_position_embedding`)、Softmax(scale+mask+softmax 7 趟→`npu_scaled_masked_softmax`,带 fp16/sk≤4096 硬约束)、MoE-permute、Flash-Attention(**O(S²)→O(S),S×S 不物化**)、Fused-EMA-AdamW(一核回写 param/m/v/s);每个带 before/after 图 + 量化优化点。
- **[[20_mindspeed_context_parallel_analysis]]**(373→**420 行**,7 优化点 callout):Ulysses/Ring-双环/Hybrid/Adaptive/KV-cache/2·cp 负载均衡 每变体补量化优化点(通信量比、overlap、straggler 消除)。
- **[[10_mindspeed_parallelism_analysis]]**(469→**495 行**,18 优化点 callout,~20 图):TP-2D/非对齐/vocab/PP 划分/MoE-EP/LayerZeRO/Custom-FSDP/分层解耦(U-split/VDP/VTP)每特性补图 + 量化优化点。
- **[[11_mindspeed_comm_overlap_analysis]]**(406→**461 行**,9 优化点 callout):MC2/CoC/MoE-overlap/fb-overlap/alltoall-MC2/DualPipeV/RiPipe/optimize-p2p/async-log 每特性补时序图 + 量化优化点(气泡比、隐藏率)。
- **[[12_mindspeed_memory_optimization_analysis]]**(248→**422 行**,15 优化点 callout):重计算/Swap/reuse-fp32/MoE-zero-mem/压缩/virtual-opt/chunk-loss 每特性补 before/after 图 + 省显存 Δ 公式。

**校验**:各 agent 透明纠正若干行号(mc2 CoC 互斥 `:21-22`、planner greedy `:127-142`、flexible_schedules 路径 `core/pipeline_parallel/`、compress pdf/ratio),coordinator 抽样复核均命中;5 页 `[[]]` 链接脚本提取确认 0 悬空。MindSpeed 系列累计约 **2460 行**(index 除外)。

---

## 2026-06-23: [[33_fault_recovery_relink_comparison]] 深挖「重新建链全过程」+「进程状态管理」(§5/§6)

**Type**: Expand（应用户"再深入解读重新建链过程,以及各训练进程状态如何管理"。读 MindSpeed TTP 状态机源码 + ARF 清理/重建回调 + Megatron Wrapper finalize,补两节深度)

- **§5 重新建链完整过程**:MindSpeed ARF 的有序回调链(`stop→clean→rebuild_group→repair`,注册序 `tft_train_initialize.py:97-107`)逐步拆解 + 时序图——`stop_device` / `torch_sync` / `unset_gather_handle` 置空旧异步句柄(`tft_stop_clean.py:76-88`)/ UCE 检查迁坏 HBM 张量(`:36-49,60-74`)/ 逐组 `reinit_process_group(rebuild_link=True)`(`tft_arf_group_repair.py:31-98`);对比 Megatron NVRx `Wrapper` 的 abort→finalize(`destroy_state`)→rank_assignment(RESERVE 热备)→initialize(`inprocess_restart.py:25-29,50-67,80-125`)。
- **§6 训练进程状态管理**:MindSpeed 自研 TTP 的显式 `WorkerStatus` 状态机(INIT→NORMAL→{ABNORMAL/FAULT/PAUSE}→STOPPED,`core/ttp/constants.py:5-12`)、rank0 `TTPController._worker_status` + 心跳带 status/iteration(`comm/controller.py:53`、`comm/heartbeat.py:102-116`)、`_on_worker_fault` 广播 PAUSE(`controller.py:535-542`);并给出「故障时哪些状态丢/留」对照表(Megatron destroy&reload vs MindSpeed clean&repair-in-place)。

**校验**:`controller.py:535-542`(PAUSE 广播)、`constants.py:5/53`、`tft_stop_clean.py`、`tft_arf_group_repair.py:31-98`、`inprocess_restart.py:25-125` 均逐一开文件核对。页面 130→约 260 行。

---

## 2026-06-23: 新建「训练快恢与重新建链」跨框架对比页(Megatron/MindSpeed/MindFormers)

**Type**: New（应用户提问"故障节点更换涉及重新建链,Megatron/MindSpeed/MindFormers 各怎么做,结论要有事实依据"。3 个并行 research-agent 分别读三仓容错代码,coordinator 抽样核验关键引用后落页;wiki 此前无容错/快恢专题)

- **新增** [[33_fault_recovery_relink_comparison]]:跨框架快恢与「重新建链」机制对比,源码核对 @ Megatron `232c478d4` / MindSpeed `1432cb09` / MindSpeed-LLM `0c16322d` / MindFormers `01e71622` / torch_npu。
  - **Megatron-LM**:委托 NVRx,`--inprocess-restart` 进程内重启——abort NCCL(`inprocess_restart.py:93-98`)→ `destroy_model_parallel`(`training.py:286-292`)→ 新 `PrefixStore(str(iteration), store)` 命名空间重跑 `init_process_group`(`training.py:1088-1090`、`initialize.py:316-333`);热备 reserve rank 顶替。
  - **MindSpeed-LLM**:MindIO TFT / **ARF 空中加油**——`arf_rebuild_process_group_callback`(`tft_arf_group_repair.py:31,47`)调 `torch_npu reinit_process_group(rebuild_link=True)` → `abort_hccl_comm("reinit")`(`torch_npu .../distributed_c10d.py:346-372`)**原地重建** HCCL(PG 对象存活);故障 rank 优化器态从同伴 DP **replica** 拷回(`tft_replica_group.py:26`、`tft_optimizer_data_repair.py:86-175`);另有 elastic scale-in/out 全重建。
  - **MindFormers**:**不自实现**,委托 MindSpore runtime + MindIO——仅 `_tft_handler.init`(`build_context.py:346-352`)使能 ARF、包优化器、reboot 节点跳 barrier(`version_control.py:289-301`);重新建链在闭源 runtime 内。
  - 显式标注三处**闭源边界**(NVRx / MindIO `mindio_ttp` / MindSpore runtime),区分"框架 Python 可见" vs "运行时黑盒"。

**整合**:父索引 [[02_engineering/02_train_frameworks/index]] 页面列表新增条目。**校验**:抽样复核 `tft_arf_group_repair.py:47`(`reinit_process_group(rebuild_link=True)`)、`distributed_c10d.py:346/370`(`abort_hccl_comm`)、`build_context.py:346-352`(`_tft_handler.init`)、`inprocess_restart.py:93-98`(abort Compose)、`training.py:1088-1090`(PrefixStore)均逐一开文件命中。

---

## 2026-06-23: [[14_megatron_ep_analysis]] 新增「DeepEP 通信量图解」三图(§③.3.5)

**Type**: Update（应用户"图示解释 DeepEP 通信量分析,用 SVG 画图转 PNG 放进 wiki"。承接本轮对话链:核实 Megatron flex dispatcher 通信量估计 → 深挖 DeepEP `intra_dispatch` NVLink 扇出内核路径 → 本次落图）

- **配图基线**:为给出可核验 `file:line`,本地浅克隆 **DeepEP @ `af9a040`**(`main`,2026-06-15;`csrc/kernels/legacy/` v1 `Buffer` 内核 = `--moe-flex-dispatcher-backend deepep` 路径)。
- **新增** [[14_megatron_ep_analysis]] §③.3.5 三图(SVG 手绘 → headless Edge 2× 元素截图 → PNG,存 `megatron-lm/assets/megatron_ep_analysis_deepep_fig{1,2,3}.png`):
  - 图 1:标准 AllToAll(按专家,跨界冗余 k 次)vs DeepEP fused_dispatch(按节点,一次 RDMA + 节点内 NVLink 扇出);跨节点流量 ∝ |R(t)| ≤ k。
  - 图 2:两级通信量分解 RDMA = |R(t)|·M / NVLink = [Σ(gₙ−1)+g_s]·M,「省 1 跳 IB ⇄ 多 1 跳 NVLink」严格相等;源码对应 `notify_dispatch`(`internode.cu:314` 每节点 `total_count` / `:313` 每卡 `per_nvl_rank_count`)+ `SourceMeta` 位图(`:22`)+ `kRDMAAndNVLForwarder` 逐卡选通(`:971`)。
  - 图 3:2 node × 2 GPU 数值走查(token X→{E1,E3,E5,E6}),跨节点 IB 2M→1M、节点内 NVLink 1M→2M,IB 加速比 (k/P)/(1−(1−1/P)ᵏ)(P=2,k=4→2.13×、topk=8→≈4×)。
- **源码纠正(code wins)**:DeepEP 落地卡 = 与源卡「同号」的 NVL rank(`internode.cu:826`),未必是目标卡;故节点内实际 NVLink = gₙ − 𝟙[同号落地卡∈目标],§③.3.2 理想公式的「−1」为上界(最好情形)。已在图 3 与正文标注。
- **工具**:复用 gitignored `.html2md/`(puppeteer-core→Edge)新增 `deepep_figs/render.mjs` 元素截图脚本。**索引**:[[megatron-lm/index]] 加 `> [!update] 2026-06-23` note。

---

## 2026-06-23: MindSpeed 系列从「分类大纲」深挖到机制级(+CP 专页,4 篇重写到 megatron 深度)

**Type**: Deepen（用户反馈"大纲有了,针对特性的 deep dive 分析缺少,参考 megatron-llm / torchtitan"。校准:`14_megatron_ep_analysis`(753 行,单机制)/`13_megatron_cp_analysis`(391)是深度标尺——每机制需 命题→源码片段+`file:line`→数据流图→通信/显存代数→权衡。原 4 篇每机制仅 ~15-30 行,远不及。并行 writer-agent 逐页重写,coordinator 抽样核验引用)

- **新增** [[20_mindspeed_context_parallel_analysis]](373 行,对标 [[13_megatron_cp_analysis]]):CP 家族专页——分派脊柱 `dot_product_attention.py:134-322` 路由五变体;**Ulysses**(头维 `single_all_to_all` 换轴 `:83-108`,通信量 vs Ring 推为 `(2/cp)·(a/a_kv)`)、**Ring/双环**(KV 环形 P2P + online-softmax `utils.py:77-119`、因果块跳过 `ring_context_parallel.py:16-33`、双环窗口 `model_parallel_utils.py:121-212`)、**Hybrid**(Ulysses×Ring 2D)、**Adaptive**、**KV-cache**;含 2·cp 因果负载均衡(`get_batch_utils.py:244-263`)与各变体整除约束(Ulysses `a%(cp·TP)==0`、megatron-cp `seq%(2cp)==0` 等)。
- **重写加深** [[10_mindspeed_parallelism_analysis]](241→469 行):CP 段收为指针(导向新 CP 专页),腾出篇幅深挖 **TP-2D**(`linear_2d_split_along_first_dim.py:128-150` AG→MM→RS 二维流水)、**PP 划分**(noop/非对齐/布局)、**MoE-EP**(GMM 原语 `gmm/experts.py:124-151`、tp-extend-ep、专家放置贪心重排 `expert_placement/planner.py:111-150`)、**LayerZeRO3**(`zero3/fsdp.py`)/Custom-FSDP、**分层解耦训练**(U-split/VDP/VTP)。
- **重写加深** [[11_mindspeed_comm_overlap_analysis]](267→406 行):三母题各配时序图+代数——*算子融合*(MC2 `npu_all_gather_base_mm`/`npu_mm_reduce_scatter_base`、alltoall-MC2 `npu_alltoallv_gmm`)、*软件流水*(CoC chunk 双流 `coc_utils.py:200-248`、MoE async-handle、fb-overlap 跨微批 + WeightGradStore 解耦 `:203-205`)、*换调度*(DualPipeV 7 段 `:310-344`、RiPipe、optimize-p2p)。
- **重写加深** [[13_mindspeed_ascend_affinity_analysis]](233→468 行,代码块 2→15、配图 4):补足代码走读——op_builder JIT(`builder.py:65-77` `cpp_extension.load`、GMM 三 dispatch key→CANN `GroupedMatmul`)、融合算子(`npu_swiglu`/`npu_scaled_masked_softmax`/`npu_rms_norm`/`npu_fusion_attention`)、HCCL buffer/QoS、`npu_apply_fused_ema_adamw`;保留 affinity 勘误([!warning])。
- [[12_mindspeed_memory_optimization_analysis]] 维持(原已达深度标尺)。

**整合**:[[mindspeed/index]] 四大类表新增「并行·CP 深挖」行;父索引 [[02_engineering/02_train_frameworks/index]](4→5 篇)与总索引 [[index]] 领域总览(MindSpeed 6)/快速导航(增 CP 页)同步。**校验**:各 agent 透明纠正了若干行号(expert_placement 路径、U-shaped loss 行、gmm dispatch 行),coordinator 抽样复核 `planner.py:111`(greedy)、`schedules.py:300`(U-loss)、`gmm.py:153`(@impl PrivateUse1)、`mc2_fuse_a2a.py:39`(npu_alltoallv_gmm)均命中;5 页 `[[]]` 链接经脚本提取确认 0 悬空。

---

## 2026-06-23: 新建「MindSpeed × MindSpeed-LLM 昇腾训练加速特性」子目录(index + 4 篇深挖)

**Type**: New domain（应用户目标"分析 MindSpeed+MindSpeed-LLM 的训练优化特性:并行/计算通信掩盖/内存优化/昇腾亲和,总结进知识库"。源码核对 @ MindSpeed `master 1432cb09`(patches Megatron core_r0.17.0)+ MindSpeed-LLM `master 0c16322d`;4 篇深挖由并行 writer-agent 各读一域源码产出,coordinator 校验引用与链接）

新建 `wiki/02_engineering/02_train_frameworks/mindspeed/`:
- [[mindspeed/index]](知识地图):**猴补丁式 Megatron 加速层**定位、`MindSpeedFeature` 契约(register_args/register_patches/validate、O0/O1/O2 优化等级门控 `feature.py:12-20`)、`create_features_list()` ~70 特性总账(`features_manager/__init__.py:367-398`)、两层结构(core 通用 + LLM 模型/任务层)、四大类罗盘。
- [[10_mindspeed_parallelism_analysis]](241 行):CP(Ulysses 头切 all-to-all / Ring / 自适应 / KV-cache,2·CP 因果负载均衡)、TP(非对齐线性 / TP-2D `[h/y,E/x]` / vocab ReplaceIndexPut)、PP 划分(noop/布局/非对齐/num-layer-list)、MoE-EP(tp-extend-ep + GMM 原语 + 专家放置 EMA 预测)、DP/分布式(LayerZeRO3 / Custom-FSDP)、分层解耦训练(U-split/VDP/VTP)。
- [[11_mindspeed_comm_overlap_analysis]](267 行):两大母题——chunk-GEMM 流水异步通信(CoC/MoE-overlap)与 matmul+集合通信单核融合(MC2 `npu_all_gather_base_mm`/`npu_mm_reduce_scatter_base`、alltoall-MC2 `npu_alltoallv_gmm`);PP 换调度消/填气泡(DualPipeV 7 段 / RiPipe 重算填泡 / optimize-p2p)。**勘误**:async-log-allreduce 掩盖的是 loss 日志 all-reduce,非梯度规约。
- [[12_mindspeed_memory_optimization_analysis]](248 行):统一原语 `untyped_storage().resize_(0)` + 反向重填;重计算(激活/norm/按 PP-rank/block-uniform)、Swap(smart-swap/swap-attention saved_tensors_hooks/swap-optimizer 态常驻 CPU)、reuse-fp32-param(fp32↔bf16 共享存储 `reuse_data_ptr`)、MoE-zero-memory、压缩(HANS/换尾数)、virtual-optimizer、chunk-loss。
- [[13_mindspeed_ascend_affinity_analysis]](233 行):**算子替换层**——op_builder JIT(`cpp_extension.load` 编 CANN 核)、融合算子(GMM 三 dispatch key→`GroupedMatmul`、`npu_swiglu`/`npu_scaled_masked_softmax`/`npu_rms_norm`)、Flash-Attention(`npu_fusion_attention` SBH/TND)、HCCL buffer/QoS 调优、融合优化器(`npu_apply_fused_ema_adamw`)。**重要勘误**:`AffinityFeature` 并非 CPU 绑核(两仓 grep `sched_setaffinity/numa` 皆空),而是 VocabParallel 交叉熵的 NPU 亲和改写(`affinity.py:13-17` 补丁 `calculate_predicted_logits`)——已据此修正 index 描述。

**整合**:父索引 [[02_engineering/02_train_frameworks/index]] 子目录表新增 `[[mindspeed/index]]` 行并更新日期;总索引 [[index]] 目录树 + 领域总览(MindSpeed 5 篇)+ 快速导航「昇腾训练加速」行同步。**校验**:抽样核对各页 `file:line`(affinity 勘误、recompute `resize_(0)`、op_builder `load`、`npu_swiglu`、Ulysses forward 均逐一开文件确认);5 页全部 `[[]]` 链接经脚本提取后确认目标存在,0 悬空(路径式 `[[megatron-lm/index]]` 等按 Obsidian 后缀匹配解析)。

---

## 2026-06-23: 新建「MindFormers PyNative 专家并行(EP)实现与通信量」专页(训练框架域 +1)

**Type**: New（应用户提问"结合 MindFormers PyNative 代码分析 EP 实现、各方案通信量,尤其 deredundancyEP 与 zeroredundancyEP";既有 [[mindformers_moe_token_dispatcher_analysis]] 只覆盖 **Graph 模式**的去冗余 dispatcher,PyNative 路径与 zero_redundancy 均为真空。源码核对 @ MindFormers `01e71622` master）

- **新增** [[mindformers_pynative_ep_analysis]]:
  - **源码勘误(code wins)**:PyNative 路径只有 `alltoall`(`ExpertParallel`)与 `alltoall_deredundancy`(`DeredundancyExpertParallel`)两种;**`zero_redundancy` 在 PyNative 不存在**,仅 Graph 路径有 `MoEAlltoAllZeroRedundancyTokenDispatcher`(证据:`pynative/config/config.py:418-423` 选项表、`parallelize.py:967-975` 选择器、pynative 子树 grep 零命中)。
  - **基础 `alltoall`**:单层 flat EP、逐 (token,expert) 对 all-to-all(`_permute:320` 每槽一行 ⇒ 同 rank 多专家=重复发送,量 ∝ k);`_build_resort_index:107-145` host 端从计数矩阵重建重排索引,省一次 routing-map a2a + 两次 sort。
  - **去冗余 `deredundancy`**:两级 oep(跨机,步长取 rank)/iep(机内 8 卡);跨机只走 AllGather+ReduceScatter(每 token 跨机恰 1 次,与 k 无关),机内 AlltoAllV 精确落专家卡;`config.py:425-433` 强校验 `EP≥npu_nums_per_device`。
  - **零冗余(对照)**:`mint.any` 按目标 rank 去重(`token_dispatcher.py:193-208`),冗余从 k 降到"不同 rank 数",收端本地复制。
  - **`OverlapExpertParallel`**:A/B/C/D 同步钩子 + 异步 a2a 做通算重叠(不改通信量)。
  - 含**三方案通信量总对照表**(拓扑 / 重复次数 / 与 k 关系 / 集合通信 / D2H / 适用瓶颈)。

**整合**:姊妹篇 [[mindformers_moe_token_dispatcher_analysis]] footer 增「PyNative 对照」回链。**校验**:新页 `file:line` 均按当前 checkout 逐一核对;交叉链接 [[mindformers_moe_token_dispatcher_analysis]] / [[14_megatron_ep_analysis]] / [[15_torchtitan_ep_analysis]] / [[20_deepseek_moe_analysis]] 经 glob 确认存在。

**目录重构(同日)**:为 MindFormers 单建子目录 `02_engineering/02_train_frameworks/mindformers/`,收纳两篇(PyNative EP + Graph 去冗余 dispatcher)及其 7 张图(移入 `mindformers/assets/`,`assets/…figN.png` 相对引用随之保持有效),新建 [[mindformers/index]] 知识地图。父索引 [[02_engineering/02_train_frameworks/index]] 子目录表新增 `[[mindformers/index]]` 行、移除两篇的单独条目;总索引 [[index]] 目录树/领域总览(MindFormers 2 篇)/MoE 快速导航同步。`[[bare filename]]` 链接按文件名解析,移动后不失效。

---

## 2026-06-22: vLLM 系列补「图改写机制深挖」专页 + 调度页补 prefill/decode 与 PD 分离(系列增至 12 篇 + index)

**Type**: Expand（沉淀对话中的源码级追问:图模式 pass 机制 / vllm_ir 自定义算子 / RMSNorm+quant 融合全程 / prefill-decode 切换与 PD 分离;源码核对 @ vLLM `485bbe1c6`）

- **新增** [[vllm_ir_and_fusion_passes_analysis]]([[vllm_fused_ops_and_kernels_analysis]] 的机制深挖伴篇):① vLLM IR 层 `vllm_ir`(`torch.library` 自建命名空间、`CompositeExplicitAutograd` 不分解 + fake ⇒ 被 Dynamo 保留为 opaque 节点、为何不挂 `aten`、provider/lowering);② `PostGradPassManager` pass 流水线与 `-O` 档默认表;③ 经 `backends.py:966` 挂进 Inductor `post_grad_custom_post_pass` 生效;④ RMSNorm+FP8 量化从「用户模型代码 → eager 双 kernel(HBM 往返)→ 手写融合 kernel `_C.rms_norm_static_fp8_quant`」全程走查 + before/after FX 图。
- **扩充** [[vllm_scheduler_analysis]] §3.12:prefill/decode 在单实例内"不切换"(统一 `num_computed_tokens` 追赶 + 混批),与集群级 **PD 分离**(KV 连接器跨实例)的不同场景对照与两种相反哲学。

**整合**:[[vllm/index]] 支柱三新增 IR/Pass 页(11→12 篇)、父索引 [[02_engineering/03_infer_frameworks/index]](→12+index)与总索引 [[index]](推理框架 14 / vLLM 13)计数同步;[[vllm_fused_ops_and_kernels_analysis]] 回链伴篇。校验:新页 `file:line` 均核对,`[[]]` 链接全部解析。

---

## 2026-06-22: 新建「自动并行」域 + 业界研究综述罗盘(1 篇 + index)

**Type**: New domain（应用户调研需求"业界自动并行研究现状、主流开源库与论文、从哪几个方面建模分析搜索较优并行策略"；wiki 此前无自动并行专题,grep 仅在 Megatron/torchtitan 页零散提及"并行策略",raw/ 亦无对口源论文,故基于公开论文/文档 Web 检索后综合成域）

新建目录 `wiki/02_engineering/06_auto_parallel/`:
- [[auto_parallel_survey_analysis]](罗盘综述):**通用流水线**(策略表示→代价模型→搜索算法→运行时,含 mermaid)、**7 大技术谱系**(算子级搜索 FlexFlow/OptCNN → 编译器传播 GSPMD/PartIR → 联合分层 **Alpa**(inter-op DP + intra-op ILP) → 显存感知 Galvatron/**Aceso** → 原语+约束 **nnScaler** → 异构/动态 Metis/Astra/Sailor → 框架原生 DTensor/veScale/OneFlow-SBP/MindSpore)、**4 个建模维度**(搜索空间 / 代价模型含 α-β 通信与显存约束 LaTeX / 硬件拓扑 / 优化目标)、**5 类搜索算法**(精确 ILP/DP/MILP · 元启发 MCMC/MCTS · 贪心传播 · 分解剪枝 · 模拟器在环)、**关键洞察**(分解是核心招式、代价模型准确性>搜索算法先进性、传播 vs 全局搜索分野)、2024–2026 趋势(显存-并行协同/异构/框架原生/4D→5D MoE)。
- [[06_auto_parallel/index]] 域索引:罗盘速览 + 后续按系统拆页规划(alpa/nnscaler/galvatron/gspmd/dtensor)。

**整合**:父索引 [[02_engineering/index]] 子领域表新增 `06_auto_parallel` 行;综述页交叉链接 [[megatron-lm/index]](手工 5D 对照组/执行后端)、[[torchtitan/index]](DTensor 原生)、[[10_mindspore_compiler_analysis]](传播范式)、[[31_comm_compute_fusion_guide]](overlap 实测)、[[32_distributed_optimizer_deepdive]](ZeRO/FSDP 分片)。**校验**:2 页均含 `## Related Pages`,跨链目标页经 glob 确认存在,0 悬空链接;论文出处以 Sources 段外链给出(Alpa/GSPMD/nnScaler/PartIR/Galvatron-BMW/综述/DTensor 等)。

---

## 2026-06-22: vLLM 系列补「算子融合与 Triton Kernel」专页(系列增至 11 篇 + index)

**Type**: Expand（应用户提问"融合算子/Triton 等算子特性有介绍吗"——既有 10 篇仅在注意力/量化页顺带提及,无专篇;补 [[vllm_fused_ops_and_kernels_analysis]] 填补真空）

新增 [[vllm_fused_ops_and_kernels_analysis]](「特性优化」支柱):**CustomOp 多实现派发**(`model_executor/custom_op.py` native/cuda/triton + `custom_ops` 开关,Inductor 下默认走 native 交其自动融合)、**torch.compile 融合 Pass**(`compilation/passes/fusion/`:RMS+quant、SiluMul+quant、AllReduce+RMSNorm/async-TP、attention+quant、SP,经 `PostGradPassManager` 挂进 Inductor `post_grad_custom_post_pass`)、**fused_moe**(grouped GEMM + Triton/CUTLASS/DeepGEMM oracle 派发 + `configs/E=*,N=*,device=*.json` autotune)。与 [[vllm_compilation_cudagraph_analysis]](图捕获)、[[vllm_quantization_analysis]](量化 GEMM)、[[vllm_attention_backends_analysis]](Triton 注意力)形成"被引用→展开"分工;跨域对照 [[21_megatron_fusion_operators_analysis]] / [[23_torchtitan_compute_memory_optimizations_analysis]]。

**整合**:[[vllm/index]] 支柱三增列本页(10→11 篇)、父索引 [[02_engineering/03_infer_frameworks/index]] 与总索引 [[index]] 计数同步;[[vllm_compilation_cudagraph_analysis]] 融合 pass 处回链本页。校验:本页 14 个 `[[]]` 链接全部解析,`file:line` 经核对 @ `485bbe1c6`。

---

## 2026-06-22: 新建 vLLM 推理引擎源码级分析系列(10 篇 + index)

**Type**: New series（对标 [[torchtitan/index]] 的深度/格式/出处严谨度;源码基准 vLLM `main` @ `485bbe1c6`(2026-06-21),源码 `E:\97-codes\torch_parallel\vllm`,聚焦 **V1 引擎**;10 个并发 agent 各写一篇 + 整合 index/parent-index/changelog/交叉链接）

新建目录 `wiki/02_engineering/03_infer_frameworks/vllm/`,按用户视角的「调度 → 模型库 → 特性优化」三支柱,每篇以「Overview → Quick Start → Deep Dive」三维展开,所有非平凡论断带 `file.py:line` 出处:

- **调度(3 篇)**:[[vllm_engine_architecture_analysis]](脊梁篇:解耦双进程 + `EngineCore.step()` 四段忙循环 + ZMQ IPC + Executor→Worker 扇出)、[[vllm_scheduler_analysis]](连续批处理 token 级、`schedule()` 先 running 后 waiting、分块预填充、抢占/重算)、[[vllm_kv_cache_management_analysis]](分页块、BlockPool 引用计数/LRU 驱逐、`allocate_slots`、块哈希前缀缓存、混合 KV、显存 profiling 定块数)
- **模型库(2 篇)**:[[vllm_model_library_analysis]](模型定义约定 `*ForCausalLM`、懒注册表、惰性流式权重加载 + `packed_modules_mapping`、TP 感知层库)、[[vllm_attention_backends_analysis]]("写 KV + 调后端"两步走、`AttentionMetadata` 桥、PagedAttention 间接寻址、统一变长注意力、FA/FlashInfer/Triton/MLA)
- **特性优化(5 篇)**:[[vllm_feature_optimizations_overview]](特性总表 + 深挖结构化输出/LoRA/分离式 KV 连接器/KV 卸载)、[[vllm_speculative_decoding_analysis]](draft+verify、n-gram/EAGLE/Medusa/MTP、拒绝采样无偏、调度 lookahead/回退)、[[vllm_quantization_analysis]](`QuantizeMethodBase` 插件框架、FP8/AWQ/GPTQ/FP4、加载期 Marlin repack、KV 量化)、[[vllm_distributed_inference_analysis]](5 维 rank 张量切 TP/PP/EP/DP、`GroupCoordinator`、PP `batch_queue` 虚拟流水线、MoE DP-attention+EP+EPLB)、[[vllm_compilation_cudagraph_analysis]](`@support_torch_compile`→VllmBackend(Inductor)、**分段 CUDA Graph** 注意力切出、`cudagraph_mode` 五态、运行时按形状 dispatch replay)

**HEAD 关键事实(各页据 `485bbe1c6` 源码核实,与多数旧博客不符)**:
- **V0 独立引擎已移除**:`vllm/engine/llm_engine.py:6` 现仅为 `LLMEngine = V1LLMEngine` 别名;今天 `from vllm import LLMEngine` 拿到的是 V1 兼容外壳,底层跑 V1 `EngineCore`。
- **注意力模块已重构**:无顶层 `vllm/attention/`;注意力层在 `vllm/model_executor/layers/attention/`,V1 后端/metadata 在 `vllm/v1/attention/`。
- **调度统一**:无独立 prefill/decode 阶段,二者统一为 `num_computed_tokens` 追赶 `num_tokens_with_spec`;分块预填充只是 `min(剩余 prompt, token 预算)` 的自然结果,无独立代码路径。
- **KV 卸载非独立子系统**:注册名 `OffloadingConnector`,与分离式推理共用 `KVConnectorBase_V1` 抽象;前缀缓存(GPU 内)/KV 卸载(下沉 CPU/盘)/分离式(跨实例)三者正交可叠加。

**整合**:[[vllm/index]] 知识地图(四支点设计哲学 / 三支柱 10 篇表 / 一条请求穿三支柱全景 mermaid / 关键设计速览 / 阅读路径);父索引 [[02_engineering/03_infer_frameworks/index]] 新增 vLLM 子框架行、总索引 [[index]] 更新目录树/计数/快速导航。**校验**:10 页 + index 全部含 `## Related Pages` 且回链 [[vllm/index]];sibling slug 与文件名一一对应;18 个跨域目标页(megatron_inference_engine / mooncake / deepseek_v3 / gpu_kernel_guide / CUDA Graphs / torch.compile 栈等)均经 glob 确认存在,0 悬空链接。

---

## 2026-06-22: 新建 verl(HybridFlow)RLHF 框架源码级分析系列(9 篇 + index)

**Type**: New series（对标 [[torchtitan/index]] 的深度/格式/出处严谨度;源码基准 verl `main` @ `8a694930`,源码 `E:\97-codes\torch_parallel\verl`;9 个并发 agent 各写一篇 + 整合 index/parent-index/changelog/交叉链接）

新建目录 `wiki/02_engineering/04_posttrain_frameworks/verl/`(verl 是 RL **后训练(RLHF)**编排框架,归入「后训练框架」而非「训练框架」——后者 Megatron-LM/torchtitan 为预训练并行框架,是 verl 的训练后端),从「架构→实现→优化」「overview→quickstart→deep dive」由浅入深拆 9 篇,每篇所有非平凡论断均带 `file.py:line` 出处:

- **入门两篇**:`verl_architecture_overview_analysis`(HybridFlow 混合控制器、五平面、五角色、v0/v1 入口、master 架构图)、`verl_quickstart_guide`(安装/Hydra 启动/config 体系/一次 GRPO 端到端走查/后端切换旋钮)
- **实现五篇**:`verl_single_controller_analysis`(`@register`+8 种 Dispatch、`DP_COMPUTE_PROTO` chunk/concat、RayWorkerGroup/colocate)、`verl_dataproto_analysis`(`DataProto`/`BatchData`/`DataProtoFuture`)、`verl_ray_trainer_analysis`(`RayPPOTrainer.fit()` 逐步追踪 + 数据流时序图)、`verl_workers_engine_analysis`(`TrainingWorker`/`ActorRolloutRefWorker` + `BaseEngine` 模板方法 + FSDP/Megatron 引擎)、`verl_rollout_resharding_analysis`(vLLM/SGLang 异步 server + 3D-HybridEngine:`get_per_tensor_param`+`CheckpointEngine`+CUDA-IPC bucketed transfer)
- **算法与优化两篇**:`verl_rl_algorithms_analysis`(`core_algos` 14 种优势估计 + 11 种 policy loss + KL k1/k2/k3,均含 LaTeX)、`verl_optimization_analysis`(placement/offload/序列打包/Ulysses SP/异步 RL 旋钮目录)

> 以上 9 篇标题内的历史活链接已于 2026-07-31 因 kb-reorg P5 Task 8 施行分段编号（`verl_architecture_overview_analysis`→[[01_verl_architecture_overview_analysis]]、`verl_quickstart_guide`→[[02_verl_quickstart_guide]]、`verl_single_controller_analysis`→[[11_verl_single_controller_analysis]]、`verl_dataproto_analysis`→[[12_verl_dataproto_analysis]]、`verl_ray_trainer_analysis`→[[20_verl_ray_trainer_analysis]]、`verl_workers_engine_analysis`→[[13_verl_workers_engine_analysis]]、`verl_rollout_resharding_analysis`→[[14_verl_rollout_resharding_analysis]]、`verl_rl_algorithms_analysis`→[[15_verl_rl_algorithms_analysis]]、`verl_optimization_analysis`→[[30_verl_optimization_analysis]]；现况总览见 [[verl/index]]），按"历史不回写"惯例本条目内原 9 处链接降级为反引号。

**HEAD 关键勘误(各页已标注,与多数博客的「经典 HybridFlow」描述不符)**:
- `RayPPOTrainer` 已 `@deprecated`(`ray_trainer.py:285`)但默认 `trainer.use_v1=false` 仍走它;新路径为 `TaskRunnerV1`+TransferQueue+`AgentLoopManager`。
- **无独立 `CriticWorker`/`RewardModelWorker` 类**:critic = 带 value head 的 `TrainingWorker`,reward 走 `workers/reward_manager` + `experimental/reward_loop`。
- rollout 退役 SPMD 同步模式,改异步 server(`ServerAdapter.generate_sequences` 直接 raise),生成由 `LLMServerManager`/`AgentLoopManager` 驱动。
- `Role` enum 实际在 `trainer/ppo/utils.py:27`(ray_trainer 仅 re-export);`compute_policy_loss`(core_algos:1203)已废弃,实际分发走 `workers/utils/losses.py` 的 `get_policy_loss_fn`。

**整合**:[[verl/index]] 知识地图(五平面表/9 篇三层表/五角色表/RL 数据流图/与训练后端的 cross-domain 链接);父索引 [[02_engineering/04_posttrain_frameworks/index]] 新增 verl 子目录行、总索引 [[index]] 更新目录树/计数/快速导航;9 篇互链 + 跨域链(→ [[11_torchtitan_fsdp_analysis]]/[[megatron-lm/index]]/[[32_distributed_optimizer_deepdive]] 等)。**校验**:9 页全部含 `## Related Pages`、均回链 [[verl/index]];所用 sibling slug 与文件名一一对应;跨域目标页均存在,0 悬空链接。

---

## 2026-06-17: 内置 default 后端「真·Split-Tiling」页改名对称（`npu_inductor_splittiling_backend_analysis`）

**Type**: Rename（应用户「内置后端加 `_splittiling` 对称区分」；仅改唯一真·Split-Tiling 页，MLIR/DVM/总览/通用页保持原名以免误标；`git mv` + 全 wiki `[[link]]` 同步）

为与实验性 [[23_npu_inductor_linearize_backend_analysis]] 成「方案对称对」，把内置 default 后端唯一描述 **Triton / Split-Tiling 路径**的深度页改名：
- `npu_triton_backend_deep_analysis` → [[11_npu_inductor_splittiling_backend_analysis]]

**刻意未改名（非单一方案，加后缀会误标）**：[[01_npu_compile_paths_overview]]（Triton/ACLGraph/MLIR 三路径）、[[10_npu_inductor_backend_analysis]]（5 后端融合规则）、[[21_npu_inductor_optimization_analysis]]（跨 Triton/MLIR/DVM）、[[20_npu_lowering_guide]] / [[12_npu_compile]] / [[32_npu_debug_guide]]（通用）。

**同步**：该页有 10 处跨目录入链（04_inductor / 05_codegen_backends / 07_op_registration / changelog），全部 `[[link]]` 用 perl 同步；页头加 `> [!note]` 指向实验 Linearize 对照页；[[04_inductor/npu/index]] 行标「内置 default（Split-Tiling）」。校验：0 残留旧名、0 新增悬空链接。

---

## 2026-06-17: 20_megatron_comm_overlap_analysis §5.6.1 补 DeepEP/HybridEP 两级通信模型

**Type**: Expand(承 §③.3,把两级模型按"加速通信"角度补到通信掩盖页;纯增,交叉引用避免重复)

**背景**:`14_megatron_ep_analysis` §③.3 已落地两级通信量公式与数值走查;通信掩盖页 §5.6（DeepEP/HybridEP 后端）此前只说"降 A2A 绝对耗时与 SM 占用",未解释**为什么**能降。

**新增([[20_megatron_comm_overlap_analysis]] §5.6.1)**:
- 两级拆分(`num_tokens_per_rdma_rank`/node→`inter_dispatch`/IB + `num_tokens_per_rank`/GPU→`intra_dispatch`/NVLink,双 buffer `num_rdma_bytes`/`num_nvl_bytes`,`fused_a2a.py:62/135`)+ 去冗余规则(跨 node 只发一次)。
- 关键式:跨节点 `∝|R(t)|`、IB 加速比 $\frac{k/P}{1-(1-1/P)^k}$(2 node topk4→2.13×、topk8→4×);完整走查指向 [[14_megatron_ep_analysis]] §③.3,不重复。
- **与"掩盖"的关系**:§5.6 去冗余/两级降 A2A 绝对耗时 + §5.1 1F1B 把剩余 A2A 掩盖到计算后;并接 §5.7 的 `high_priority_a2a_comm_stream` / `moe_hybridep_num_sms_preprocessing` 调尾延迟。

**校验**:LaTeX、`path:line`、`[[link]]` 按页约定;无删改既有内容。

---

## 2026-06-17: NPU 实验后端 3 页按「方案」改名（`npu_inductor_linearize_*`）

**Type**: Rename（应用户「区分方案」要求，避免与 torch_npu 内置后端页混淆；`git mv` + 全 wiki `[[link]]` 同步）

把实验性 `npu_inductor_2.9.0`（**Linearize 方案**）的页统一为 `npu_inductor_linearize_*` 前缀，与内置 default（**Split-Tiling**）的 `npu_inductor_*`/`npu_triton_*` 区分：
- `npu_inductor_dynamic_shape_analysis` → [[24_npu_inductor_linearize_dynamic_shape_analysis]]
- `npu_inductor_vs_builtin_comparison` → [[31_npu_inductor_linearize_vs_builtin_comparison]]
- [[23_npu_inductor_linearize_backend_analysis]] 本已合规，不变。

全 wiki `[[link]]`（index / changelog / 三页互链）同步更新；校验：0 处残留旧名、0 新增悬空链接。

---

## 2026-06-17: Inductor 分析「完整录入」扩写 — NPU 实验后端拆 3 页（§0 对标 + §1 三方 output code）+ 上游融合补全

**Type**: Expand（应用户「完整录入、不过度裁剪、查知识熵减」要求；回读原始《npu_inductor 设计与对标分析》§0/§1 精确还原；纯增不删既有结构）

**背景**：同日先前的「Inductor 后端分析合入」条为控冗余而**压缩过度，存在明显知识熵减**——丢了 §0 全套实测对标表、§1 GPU/内置/本后端三方 output code 逐行对比、四遍折叠的 dual-decomp 实例、动态 shape 三情形 A/B/C 完整代码、上游 G2 融合的 prologue/epilogue/foreach/proximity 细节。本次按「完整录入」扩写还原。

**NPU 侧：1 页 → 3 页系列**：
- [[23_npu_inductor_linearize_backend_analysis]] 扩为完整版：装配顺序全表、Linearize 恒等式 + `_apply_linearize` 主干 + 四遍折叠表 + **dual-decomp 折叠实例**（softmax-bw + sum(0) + permute，4 独立轴 → 2 基础轴的完整除/模映射 + 地址文本修复）、索引线性化全 6 pass、40-CU group dispatch prologue 完整代码、融合门控（病灶数据 + 别名坑）、r 轴 rsplit（partial + combine）、**全 5 处类型降型**、白名单 lowering + 算子专项、**完整可优化点**。
- [[24_npu_inductor_linearize_dynamic_shape_analysis]]（新）—— 编译一次 vs gears 分桶、签名 numel/divisor 代码、header 三件套、**三情形 A/B/C 完整代码 + 对照表**、配套（fold_trivial / 符号 split / static split block）、permute 产物。
- [[31_npu_inductor_linearize_vs_builtin_comparison]]（新，comparison）—— **§1 三方 output code 逐行对比**（GPU / 内置 Split-Tiling / 本后端 Linearize 完整 kernel + 逐项差异表）+ **§0 全套实测**（torchbench 34 模型总体 / 逐模型、京东 OneRec 4 backbone、test_all 60 算子 case）+ 逐维综合矩阵。

**上游侧补全**：`scheduler_analysis`（历史活链接，该页已于 2026-07-30 判重删除，本条所述 §7.6 内容已逐字并入 [[13_scheduler_dependency_graph_fusion_and_ordering_analysis]] §18.6，按"历史不回写"惯例降级为反引号）新增 §7.6——组兼容（numel/rnumel + tiling 一致）、proximity 门控（>64）、模板 prologue/epilogue 融合、foreach 融合、`_LoopMutationTracker` 回滚 + 循环重排；`> [!note]` 指向 NPU 后端的 read 门控 / proximity 收 20。

**索引/校验**：[[04_inductor/npu/index]] 补 2 新页行；3 个 NPU 页互链 + backlink 既有内置后端页；无悬空链接；§0/§1 数据均回读原始设计文档精确还原并标注口径（本库未独立复跑）。

---

## 2026-06-17: 14_megatron_ep_analysis §③.3 补「两级通信量公式 + 数值走查 + all2allv 澄清」

**Type**: Expand(对照 `Megatron-LM` `dev@232c478d4` 源码 `fused_a2a.py` / `token_dispatcher.py` 核实;单页增补,纯增)

**背景**:用户追问 DeepEP/HybridEP 的**具体通信量公式**、**两级通信如何进行**,以及节点间是否用 all2allv。原 §③.3 只有一张概念示意图,缺公式与逐字节走查。

**新增([[14_megatron_ep_analysis]] §③.3 下 4 个子小节)**:
- **③.3.1 两级 dispatch 机制(源码)**:`get_dispatch_layout` 的双计数 `num_tokens_per_rdma_rank`(每 node→`inter_dispatch`/RDMA)+ `num_tokens_per_rank`(每 GPU→`intra_dispatch`/NVLink);双 buffer `num_rdma_bytes`/`num_nvl_bytes`(`fused_a2a.py:62/135/168`);asymmetric-domain forwarding 规则(跨 node 只发一次,落地 NVLink fan-out)。
- **③.3.2 通信量公式**:逐 token 两级分解 $\text{RDMA}=|R(t)|M$、$\text{NVLink}=[\sum(g_n-1)+g_s]M$;聚合式 + IB 加速比 $\frac{k/P}{1-(1-1/P)^k}$;与 §2.4.1 标准 A2A `4·S·B·H·K·(E−1)/E²` 对齐。
- **③.3.3 数值走查**:2 node×2 GPU、8 专家、EP=4、topk=4 逐字节例子(token X→{E1,E3,E5,E6}),标准 2M vs DeepEP 1M 跨节点对照,代入加速比 2.13×(topk8→4×)。
- **③.3.4 all2allv 澄清**:标准 `MoEAlltoAllTokenDispatcher` 用 NCCL all2allv(`token_dispatcher.py:703`);DeepEP/HybridEP **非 collective**,是 `buffer.dispatch()`(`fused_a2a.py:160`)的 NVSHMEM 单边 RDMA(IBRC/IBGDA)+ permute 融合 + 两级 —— 语义是变长 A2A,实现非 all2allv,故能 node 级去冗余。

**校验**:LaTeX 公式块、源码行号、`[[link]]` 均按本页约定;无删改既有内容。

---

## 2026-06-17: Inductor 后端分析合入 — NPU 实验性 Linearize 后端 + 上游 GPU 派发/reduction/autotune 基线（4 新页 + 1 增补）

**Type**: Add & Augment（对照本地源码逐文件核实：`npu_inductor_2.9.0` 包 + upstream PyTorch 2.9.0 `E:\97-codes\pytorch\pytorch\torch\_inductor`；6 个只读 agent 取证 + 既有页去重；纯增不删既有结构）

**背景**：用户在 pytorch 工作区完成了对 NPU 实验性后端 `npu_inductor_2.9.0`（独立 monkey-patch 包，≠ torch_npu 内置 `_inductor`）及其上游 GPU 基线的源码级分析，需按知识库约定（NPU↔upstream 分开、overview→quickstart→deepdive、零冗余）合入。只读 agent 确认：既有 `04_inductor/npu/` 8 页全部讲 **torch_npu 内置**后端（Split-Tiling/CATLASS，v2.7.1.post5），**零提及** Linearize / `npu_inductor_2.9.0`；既有上游页已覆盖后端注册/融合/动态 shape 符号化，**缺** GPU kernel 派发模型、reduction codegen、autotune 三块（后者即本域 index 列出的「Inductor autotuning」空白）。

**新增 4 页**：
- [[23_npu_inductor_linearize_backend_analysis]]（NPU）—— 实验性 `npu_inductor_2.9.0`：import 即 patch + `disable_register_inductor_npu()` 关掉内置后端、Linearize（多维→40-CU `bin[40,1,1]` group dispatch）+ 索引线性化、编译一次动态 shape（3 情形）、`NPU_MAX_FUSED_READS` 融合门控、r 轴 rsplit、类型降型、白名单 lowering、与内置后端逐维对比、可优化点。
- [[23_inductor_gpu_kernel_dispatch_model]]（upstream）—— GPU kernel 骨架（`program_id→offset→mask`，无循环）、`IterationRanges` 树、stride-1 tiling、`Grid1D/2D/2DWithYZOverflow/CooperativeReductionGrid`。
- [[22_inductor_reduction_codegen_deep_analysis]]（upstream）—— persistent/looped/split/cooperative reduction（semaphore barrier）、block ptr/TMA。
- [[21_inductor_autotuning_analysis]]（upstream）—— `CachingAutotuner` 生命周期、config 启发式、`config_of`/AttrsDescriptor、`make_launcher`、`triton.compile(ASTSource,GPUTarget)`→PTX/cubin、`DeviceProperties`（填补「Inductor autotuning」空白）。

**增补 1 处**：[[24_inductor_codegen_dynamic_shape_analysis]] 新增 §2.4——`s0→ks0` 重命名 + `signature_to_meta._decide_tl_dtype` 把动态 `ks*` 升 `tl.int64` 防 `ks0*ks1` 溢出；加 `> [!contradiction]` 指向 NPU 后端的 i32 降型（GPU↔NPU 动态 shape 整型的根本分歧）。

**索引/空白更新**：[[04_inductor/index]] 加「codegen 派发与运行时（GPU 基线）」分组 3 行；[[04_inductor/npu/index]] 加实验后端行 + 头注（区分内置/实验、PyTorch 2.9.0 基线）；[[01_ai_frameworks/index]] 空白「Inductor autotuning」「NPU Monkey Patch 演进追踪 v2.9.0」标 ✅ 并指向新页。

**交叉引用**：4 新页均含 `## Related Pages` + `[[wikilink]]`；NPU 页 §六对比直接 backlink 既有 [[11_npu_inductor_splittiling_backend_analysis]]/[[01_npu_compile_paths_overview]]/[[21_npu_inductor_optimization_analysis]]（内置后端细节，不复述）；上游 3 页互链并指向既有 `scheduler_analysis`（历史活链接，该页已于 2026-07-30 判重删除并入 [[13_scheduler_dependency_graph_fusion_and_ordering_analysis]]，按"历史不回写"惯例降级为反引号）/`dynamic_shapes_full_analysis`（历史活链接，该页已于 2026-07-30 并入 [[20_symbolic_shapes_guards_and_graph_reuse_analysis]]，按"历史不回写"惯例降级为反引号）/`PyTorch_Inductor_Technical_Analysis`（历史活链接，该页已于 2026-07-30 判重删除，按"历史不回写"惯例降级为反引号）。

**核验**：所有代码引用带 `file:line`（npu_inductor 包 + upstream `torch/_inductor/`）；零冗余（内置后端/上游已覆盖部分一律 cross-link 而非复述）；纯增，未删改既有页结构。源分析底稿在 pytorch 工作区 `npu_inductor_2.9.0/triton-backend-analysis/`。

---

## 2026-06-17: SimpleFSDP 页 §5 深挖(编译流程 + 两个通信 pass + 掩盖机制)

**Type**: Expand(对照 torchtitan `main` @ `61c010fcb` `experiments/graph_trainer/` 源码逐行核实;纯增不改既有结构)

**背景**:`[[25_torchtitan_simple_fsdp_analysis]]` 原 §5 偏"概念入门",未讲清编译流程 / 通信 pass / 加 pass 阶段 / 掩盖机制。用户追问后,读透 `compile.py` / `passes.py` / `fsdp_passes.py` / `trainer.py` / `make_fx_tracer.py`,把 §5 从一节扩成 5.1–5.4 源码级深挖:

- **5.1 编译流程**:`aot_fx_trace` 首步 `minimal_fx_tracer`(make_fx)把 fwd+loss+bwd 追成**一张 joint FX 图**(redistribute 落成 `all_gather_into_tensor`/`reduce_scatter_tensor` 节点),`apply_graph_passes` 跑 `compile_time_passes` 流水线改写图(只跑一次),之后每步 `run_traced` 复用;给出 10 步 pass 流水线,标出两个通信 pass 在**第 6/7 位**(显存策略之后、inductor 之前)。
- **5.2 通信 pass ①** `reassign_collective_pgs_pass`:把 AG 改派到额外 NCCL PG(同 ranks、`use_local_synchronization`)→ 独立 CUDA 流 → **AG∥RS∥compute**(等价 FSDP2 多流)。
- **5.3 通信 pass ②** `joint_transformer_block_bucketing_reordering_pass`(`JointManualOverlapScheduler`):按 block/方向/FSDP2 参数序**分桶**(每 block 合 1 AG+1 RS)+ `overlap_deps` **重排**(AG 逆序预取、RS 延后 wait 越过计算)。
- **5.4** 一图收束 + 纠正:`autobucketing_/transformer_block_bucketing` 是已废弃 JIT 后端的非 joint 版,默认 aot_fx_trace 走 joint 版。

**更新**:`[[25_torchtitan_simple_fsdp_analysis]]` §5 重写、复核表补 8 条(编译流程 + 两 pass)、§9 小结补编译流程条;同步源文档 `llm_repo/torchtitan/docs/parallelism-analysis/simple-fsdp.md`。页头日期 → 2026-06-17。

**追加(同日,应用户「补充通信粒度 + 配图」)**:新增 **§5.5 通信粒度**——讲清 SimpleFSDP **trace 时逐参数(一参一次,无 eager 分组)→ 编译期 bucketing pass 按 block 合成每块 1 AG+1 RS**;「分层统一通信」是编译期优化产物、非天生(不开 compile 即退化逐参数)。修正 §6 对比表「通信单位」行。新增 **2 张 SVG→PNG 机制图**(入 `torchtitan/assets/`):`simple-fsdp-compile-flow`(编译流程 + 10 步 pass 流水线,高亮第 6/7 通信 pass)、`simple-fsdp-bucketing-overlap`(逐参数 → 每块 1AG+1RS 的三流并发时间线)。复核表补「参数化逐参数 getter」行。

---

## 2026-06-16: Megatron-LM 知识库去重整合 + 命名对齐 torchtitan(删 1 · 改名 21 · 索引收敛)

**Type**: Refactor & Dedup(用户授权"只删重复的 md";4 个只读 agent 产出重复度矩阵 → 合并唯一独有内容 → 删冗余文件 → 全量改名 + 链接修复;允许删除既有文档)

**背景**:megatron-lm 目录此前**命名两代混杂**——旧代 CamelCase/前缀混乱(`Megatron-LM_MoE_Zero_Redundancy_Analysis`、`Megatron_LM_TFLOPS_Analysis`)+ 新"源码级系统分析系列"无前缀(`ep_analysis`/`tp_analysis`…),且疑似存在重复知识。4 个只读 agent 逐页对照后结论:**两代多为"深版 + 精简digest 指针"的互补关系,而非重复**,仅 1 页是真正被涵盖的冗余。

**① 去重删除(1 文件)**:
- 删除 `Megatron-LM_MoE_Zero_Redundancy_Analysis.md` —— 其零冗余 AllToAll / 七阶段 dispatcher / MoE Folding 知识已被 [[14_megatron_ep_analysis]] **完全涵盖且更深**(该旧页 2026-06-16 自身更新note 也已指向 ep_analysis)。删除前把其**唯一独有教学资产**「EP=4、num_experts=4、topk=2 的逐 token 数值走查(routing_map 矩阵 + A2A 传输矩阵 + 反向 A2A + 加权 unpermute)」并入 [[14_megatron_ep_analysis]] §②.3.1。

**② 命名对齐 torchtitan(改名 21 文件)**:本目录全部页统一为 `megatron_<topic>_analysis`(小写 snake_case,对齐 `torchtitan_<topic>_analysis` 风格)。
- 新系列 19 页加前缀:`ep_analysis`→`14_megatron_ep_analysis`、`tp_analysis`→`12_megatron_tp_analysis`、`cp_analysis`→`13_megatron_cp_analysis`、`pp_schedulers_analysis`→`15_megatron_pp_schedulers_analysis`、`ddp_optimizer_analysis`→`megatron_ddp_optimizer_analysis`、`recompute_analysis`、`optimizer_internals_analysis`、`precision_cudagraph_fusion_analysis`、`training_stability_observability_analysis`、`rl_posttraining_consistency_analysis`、`inference_engine_analysis`、`model_structure_analysis`、`dataset_analysis`、`packed_dataset_dynamic_cp_analysis`、`dist_checkpointing_analysis`、`parallelism_orchestration_analysis`、`pp_supplements_analysis`、`tp_fsdp_resharding_supplements_analysis`、`moe_training_optimization_report` 均加 `megatron_` 前缀。
- 旧 CamelCase 2 页规整:`Megatron_LM_TFLOPS_Analysis`→`32_megatron_tflops_analysis`、`Megatron_vLLM_Weight_Sync_Analysis`→`33_megatron_vllm_weight_sync_analysis`。
- 已合规 5 页不动:`20_megatron_comm_overlap_analysis`、`21_megatron_fusion_operators_analysis`、`22_megatron_memory_optimization_analysis`、`16_megatron_distributed_optimizer_analysis`、`25_megatron_nonuniform_tp_analysis`。
- **链接修复**:全 wiki(208 个 md)用 `[[<basename><delimiter>` 锚定的 perl 替换更新所有 `[[wiki link]]` + 反引号/散文中的 `*.md` 文件名提及;锚定保证不误伤 `[[23_deepseek_v4_cp_analysis]]`/`[[12_torchtitan_tp_analysis]]`/`[[25_megatron_nonuniform_tp_analysis]]` 等近名页。

**③ 修复历史悬空链接**:`[[llm_parallelism_analysis]]`(该页从未以 .md 形式存在,仅旧 .html;changelog 早有记录)全 wiki 重指向 [[15_megatron_pp_schedulers_analysis]](正反向 DAG + 调度,最贴近其原意);涉及 megatron-lm/index、父级 `02_train_frameworks/index`、torchtitan 多页。

**④ 索引收敛**:[[megatron-lm/index]] 原"Core Topics"旧分组中 3 行(并行/MoE 行)已与下文 18 篇系列重复——重构为「全景报告(capstone)+ 专题深挖(系列外深版:distributed_optimizer / memory / fusion / comm_overlap / nonuniform_tp / tflops / vllm_weight_sync)」,移除重复行;Related Pages 去重;加「去重与命名整合」note 说明深版/digest 互补关系。父 index 同步基线 `ee3f1ff`→`232c478d4` 并移除悬空行。

**保留判定(审计结论:深版,非重复,不删)**:`21_megatron_fusion_operators_analysis`(融合算子全目录,precision §3 是其 digest)、`25_megatron_nonuniform_tp_analysis`(NTP 容错深版且更准确,tp_fsdp_resharding §2 是其 digest)、`16_megatron_distributed_optimizer_analysis`(FP8/FP4 量化 + CPU-offload + 三种 FSDP 对比 + §A.7 Muon,多页反向引用)、`22_megatron_memory_optimization_analysis`(显存 survey,与 recompute 互补);`32_distributed_optimizer_deepdive`(父目录,跨框架对比)。

**校验**:目录文件 28→27;**全 wiki 0 处仍指向旧 megatron 基名**、0 处残留旧 `.md` 散文提及(changelog 历史条目按惯例保留原名不改写);全量 `[[link]]` 一致性检查通过,**未引入任何新 dangling link**(`Megatron-LM_Distributed_Parallel_Exam`/`npugraphs_memory_analysis`/`scaling_laws_for_transfer_analysis` 为既有遗留,非本次)。

---

## 2026-06-16: Megatron-LM 知识库对照 `dev@232c478d4` 全量刷新(22 页 · 7 维 + 模型结构 · 9 并行 agent)

**Type**: Update & Verify(更新上层 `Megatron-LM` 源码 `dev` 分支 `77c0f8cb3`→`232c478d4`,FF 306 commits;再对照 wiki 基线 `ee3f1ff`→`232c478d4`(298 commits)逐页核实增量并纠错;铁律＝纯增不删、每处增补带 `> [!update] 2026-06-16` + `path:line` + `(#PR)`、行号以当前 dev 复核)

**背景**:Megatron-LM 源码级系统分析系列 18 篇初版基于 `dev@ee3f1ff`(2026-05-19)。上层仓库本地 HEAD 落后,先 fast-forward 到 `232c478d4`(2026-06-16,今日);该区间 298 个非合并 commit 覆盖用户点名的 7 个维度(并行/显存/计算/通信/低精度/训练稳定性/RL)+ 一批模型结构新增。9 个并行 agent 按**互不相交的页分组**(无文件冲突)逐页对照当前源码核实。

**各维度新增要点**:
- **并行**:1F1B `mtp_post_process` 重排序 + combined-1F1B 释放 loss-node 输入(#4695/#4909/#4511);HyperCommGrid 命名视图异构并行(#5148)、bridge 跨网格 P2P 专用 pg(#5234)、训练循环迁移 `pg_collection`(#5259/#5250/#5006);动态 CP per-microbatch CP 度 + TE CP-group 还原修复(#4226/#5215/#5123);非融合 cross-entropy 接收 `tp_group`(#5128)。
- **通信**:DeepEP v2 flex dispatcher(`deepepv2`/ElasticBuffer,#4793)、THD 下 deepep/hybridep(#4816)、高优先级 A2A 流 + HybridEP 预处理 SM(#4694)、dispatch 排空前驱 RS(#4940)、双 buffer wgrad 竞态修复(#5222)、A2A-Overlap for Megatron-FSDP(#3797)、移除 HybridEP IB guardrail(#4846/#4719/#4718)。
- **显存**:Paged Stashing 落地(#4247/#5003)、NCCL UB 内存池反注册(#4492)、细粒度 offload in-flight 节流(#4692)、显存估算计入 EP(#4687)、移除 checkpoint-time cache reclaim(#5170)、提前 del output(#4742)、FSDP double-buffer IMA 修复(#4810);GDN 整模块选择性重计算(#5296/#4715)、HybridModel 重计算(#4496)、MTP 重计算 + 训练 CG 修复(#4593/#3919)。
- **计算**:TE op-fuser GroupedMLP 融合(#4636)、TEFusedDenseMLP Dense+Grouped GEMM SM100+(#4318)、ScaledSReLU/ClampedSwiGLU(#4859/#5130)、DSv4 Hybrid Attention 融合 kernel(#4894)、冻结线性 dgrad fold(#5092)、mHC 融合 kernel 多后端(#4624)、融合 MLA 权重梯度 hook 修复(#5273)。
- **低精度**:MXFP8/NVFP4 param-gather 修复(#4994/#4800/#4358/#4852/#4562)、opt-in MXFP8 LM-head(#4825)、CUDA Graph API 拆解 impl/modules/inference-scope(#4292)、CG 覆盖按 max_tokens(#4214)、TE 2.15/2.16(#4682/#4992)。
- **训练稳定性**:grad-norm 超阈值跳过整步(#3460)、MoE aux/z-loss TP>1 梯度缩放修复(#5047)、DSA indexer loss 跨 mb 平均(#4070)、MoE logging 重构(#3431)、MTP 稳定性套件(#3456/#3459/#4116/#5080)、RerunStateMachine 去 stat 系统调用(#5107)。
- **RL/训推一致**:`--rl-inference-parsers` 接入 MRL(#4768)、Refit 重构统一 CopyService(#4762)、policy-epoch 权重时效(#4533)、logprob 0-token 切片修复(#5167);推理高层 API `MegatronLLM`/`MegatronAsyncLLM`(#4697)、进程级 `InferenceMode`(#4617)、MTP/prefix-cache 统计持久化 + 接受率指标(#4101/#3458)、mixed-prefill CG 分布(#3509)、非均匀 PP 的 KV layer_map(#4775)。
- **模型结构(最大新增)**:DeepSeek-V4 hybrid(DSA 学习索引器 top-k 稀疏 + CSA/HCA 压缩注意力 + hash 路由,#5042/#5130/#5018/#5142/#3026;TP=1、暂无推理路径)、GDN 序列打包(#2645)、Mamba conv 直接 mixer 参数(#4899)、mHC 支持 HybridModel(#4949)、Step-3.5-Flash 逐头注意力门控(#4841)、Qwen3.5/Qwen3-30B(#4776/#4751/#5012);VarlenDataset(#4832)、get_batch 整合 + SFT THD PP(#4103)。

**纠正的知识库错误(4 处实质性,均已源码核实)**:
1. **Muon/ZeRO 框架过时**:Muon 现经 `LayerWiseDistributedOptimizer` + 独立 `DistributedOptimizer` 经 `ChainedOptimizer` 串联,**与 ZeRO 切分共存**(#4509/#4771);`--layer-wise-distributed-optimizer` flag **不存在**(由 `--optimizer muon --use-distributed-optimizer` 触发);`optimizer/muon.py` 是 28 行兼容 shim,真实现在 `emerging_optimizers.py::TensorParallelMuon`(此归属错误自 `ee3f1ff` 即存在)。
2. **`mtp_isolated_loss` 已移除**:#5080 引入后被 #5223 合并进 `mtp_detach_heads` 并删除,HEAD 无此配置。
3. **moe_layer `train()` 重写已删除**:dispatcher 改由 `InferenceMode.is_active()` 在 `MoELayer.forward` 判定(#4617),不再按 train/eval 切换。
4. **GDN 统一 A2A(#4913)未在当前源码**:被后续 dev↔main 合并回退,GDN 前向仍用 per-section A2A 循环(标 `[!contradiction]`)。
- 另修正多处 `path:line` 行号漂移(`optimizer.py`/`cuda_graphs.py`/`hyper_comm_grid.py`/`transformer_layer.py`/`bridge_communicator.py` 等,均因新增代码下移),逐处以当前 dev 行号标注。

**索引更新**:[[megatron-lm/index]] 系列标题基线 `ee3f1ff`→`232c478d4`,新增「ee3f1ff→232c478d4 增量刷新」`[!update]` 总览块(7 维 + 4 处纠错 + 交叉链接)。

**校验**:22 页改动 +641/−10(删除仅为标注重排,无内容删除);全量 `[[wiki link]]` 一致性检查通过,本次**未引入任何 dangling link**(`llm_parallelism_analysis` 等为既有遗留,非本次新增)。上层 `Megatron-LM` 本地改动(`.agents/.claude` 符号链接型变更、未跟踪 html)未受 FF 影响。

---

## 2026-06-16: torchtitan 新增 HSDP 反向 + 性能手段 + SimpleFSDP(4 页 + 3 图)

**Type**: Expand(对照 torchtitan `main` @ `61c010fcb` / PyTorch 2.9.1 源码逐行核实;5 个并行 research agent 摸清缺口 + 4 个并行 agent 机械转换入库;纯增不改既有)

**背景**:torchtitan 上游迭代后,既有 8 篇(01–06 + AC + FSDP 预取)未覆盖一批性能手段。审计(对照 HEAD `61c010fcb`)确认遗漏:HSDP 反向双流掩盖、低精度(Float8/MXFP8)、算子融合、编译、对称内存、Async-TP、MinimalAsyncEP、full_dtensor、SimpleFSDP。

**新增(4 页,均带 file:line 源码复核表)**:
- [[21_torchtitan_hsdp_backward_overlap_analysis]] — HSDP 反向 reduce-scatter 与 all-reduce 双流掩盖:`foreach_reduce` 跨流编排、host 端算子下发顺序、AR∥RS 并发的正确性证明、reduce 路径 fp32 暂存与显存峰值;**带 3 张 SVG→PNG 机制图**(时间线 / 正确性分解 / 显存)
- [[23_torchtitan_compute_memory_optimizations_analysis]] — 算力/显存:Float8 rowwise/MXFP8、FusedSwiGLU/MoE grouped GEMM/FusedQKV、逐 block 编译即融合、融合 Adam、ChunkedCELoss、CPU offload
- [[24_torchtitan_comm_optimizations_overlap_analysis]] — 通信:跨维度计算-通信掩盖矩阵、Async-TP 微流水、对称内存、MinimalAsyncEP、full_dtensor SPMD
- [[25_torchtitan_simple_fsdp_analysis]] — SimpleFSDP(graph_trainer 实验,arXiv:2411.00284):分片即 DTensor `redistribute` 进图、编译器 pass 分桶重叠;与 FSDP2 eager 多流编排逐项对比

**纠正的常见误解(已写入)**:① Float8 rowwise 下通信仍高精度,本版无 fp8 all-gather;② torchtitan 核心循环不在 microbatch 间延迟 FSDP 规约(每 mb 都 reduce-scatter);③ MinimalAsyncEP 不做通信-计算重叠;④ `set_symm_mem_for_comm` 在 torch 2.9.1 不存在(追踪更新版 torch)。

**索引更新**:[[torchtitan/index]] 头块基准 `cf3c4312`→`61c010fcb`、计数 9→12、新增「深挖伴篇(3)」补 HSDP +「性能手段与编译器路线(3)」分区 + Related Pages;[[02_engineering/02_train_frameworks/index]] torchtitan 行计数 7→12、基准更新、日期 2026-06-16。资源:3 张 HSDP 机制图(SVG+PNG)入 `torchtitan/assets/`。

**校验**:4 页无残留相对链接 / 旧页脚,HSDP 三图引用完整;13 项 rel-link→`[[wiki]]` 映射逐条核对。来源同步自 `E:\97-codes\llm_repo\torchtitan\docs\parallelism-analysis\{07-hsdp-backward-overlap, 08-compute-and-memory-optimizations, 09-comm-optimizations-and-overlap, simple-fsdp}.md`。

---

## 2026-06-15: 补齐 eager 运行时地基 — 新增 7 模块 21 页(Workflow 编排 + 源码逐行核实)

**Type**: Ingest & Expand（Workflow：覆盖审计 → 架构全图 → 缺口路线图；再 7 模块流水线 research→并行写 3 层→校验；铁律＝纯增不改既有、对照 `E:\97-codes\pytorch\pytorch` v2.13.0a0 核实行号、不破坏既有 wikilink）

**审计结论**：01_ai_frameworks 此前几乎全是 torch.compile 编译栈 + dispatcher/op 注册,**整层 eager 运行时地基缺失**（用户点名 autograd 引擎、tensor 表达机制）。13 个 agent 对照源码核实后产出 8 项优先级缺口路线图。

**新增（7 模块,P0+P1,纯增无删改）**：
- **[P0] `[[00_tensor_and_storage/index]]`**：张量表达机制 — `Tensor=intrusive_ptr<TensorImpl>`、Storage/视图别名、sizes/strides/dtype、张量上的 DispatchKeySet（overview + quickstart + deepdive）
- **[P0] `[[10_eager_autograd/index]]`**：eager 反向引擎 — Node/Edge DAG、多线程 Engine、AccumulateGrad、SavedVariable、自定义 Function；**含与 03_aot_autograd 的对照表**（运行时磁带 vs 编译期联合图）
- **[P1] `[[11_aten_op_execution/index]]`**：ATen 算子定义/执行 — native_functions.yaml、torchgen、结构化 kernel、boxing（07 的上游通用版）
- **[P1] `[[12_nn_module_system/index]]`**：torch.nn — Module/Parameter/state_dict/hooks/容器/lazy/Optimizer
- **[P1] `[[13_runtime_memory_amp_profiler/index]]`**：缓存分配器 / AMP+GradScaler / Kineto Profiler
- **[P1] `[[14_fx_export_and_extensibility/index]]`**：torch.fx / torch.export / torch.library custom_op / functorch
- **[P1] `[[15_distributed_primitives/index]]`**：c10d/DDP/FSDP/DTensor/TP/PP（[[02_train_frameworks/index]] 的底座）

**索引与规划**：域 [[01_ai_frameworks/index]] 重构为「两条主轴（eager 地基 / 编译栈）+ 三层功能目录」,「知识空白」扩写为带优先级的规划路线图（P2 序列化遗留 `16_serialization_and_legacy`、各新模块 NPU 特化 `npu/` 下沉等留痕）。

**校验**：21 页结构齐全（头块/层次/Related Pages）；域内 317 条 wikilink 全解析；独立抽查 tensor/autograd 关键 citation（`node.h:112`、`variable.h:229-230`、`TensorImpl.h:510` 等）与源码一致。后续对抗式全量校验按「可信度分级」原则从略（写手已逐行核实）。

---

## 2026-06-15: 全模块分层与 NPU 分离审计修复(Workflow 编排 + 对抗式验证)

**Type**: Audit & Fix（Workflow:9 模块并发审计 → 每条发现对抗式验证 → 7 模块并发修复;铁律＝保留独有信息、只去重叠/迁移 NPU、不重命名）

**审计**:对 01_ai_frameworks 全部模块核查三项——冗余/雷同、overview→quick start→deep dive 分层、NPU↔upstream 分离;32 个 agent,每条可执行发现经对抗式验证(默认怀疑,排除「`input` 含 npu 子串」等假阳性),确认 16 条。

**修复（7 模块,纯增改无删除/重命名）**:
- **04_inductor**:`Pytorch_Compile_Debug_Analysis` 残留的 NPU 平台声明 + 脚本块(`DEVICE_TYPE=npu` / ASCEND/HCCL env)清除 → 纯 upstream(NPU 调试已在 [[32_npu_debug_guide]])
- **06_graphs**:`cuda/README` 移除混入的「NPU Graphs 对比/系统要求/参考资源」段 + 失实目录树 → 纯 CUDA,对比指向 [[30_comparison]];与 cuda/index(导航)分工
- **05_codegen_backends/mlir**:新建 `torch_mlir_quickstart`(quick start 层),`torch_mlir_pass_pipeline_analysis` §0 去重定位说明
- **03_aot_autograd**:index 补「模块概述」(定义/栈位置/三职责),quickstart §1 精简为「快速导航」
- **07_op_registration/npu**:index 补「整体架构」(算子生命周期 + 三维度依赖),`npu_operator_graph_eligibility_guide` §7 去 aclop/aclnn 重述、加交叉引用
- **08_kernel_optimization**:`operator_optimization_guide` 加「文档结构与阅读路径」+ §2.2/§6 GPU↔NPU 对标标注
- **09_other_frameworks**:`mindspore_compiler_analysis` 补「快速理解」(quick start)+ §5.3 标注昇腾 NPU 特化

**校验**:01_ai_frameworks 全量 wikilink 零断链;`cuda/README` 与 Debug 页 NPU 残留清零。

---

## 2026-06-15: 04_inductor 由浅入深重构 + NPU/upstream 彻底分离

**Type**: Restructure（4 agent 并发；铁律＝保留全部独有 upstream 信息、只去重叠、迁移 NPU；对照 pytorch/torch_npu 源码核实）

**由浅入深三层闭环**：
- overview：`torch_compile_architecture` 就地重写为「Inductor / torch.compile 概览」（153 行：是什么→在 torch.compile 中的位置→五阶段一览→核心概念→最小例子旅程→导航）
- quick start：`inductor_quickstart`
- deep dive：端到端 `inductor_compiler_pipeline_analysis`（脊柱）+ 后端/IR 深度 `PyTorch_Inductor_Technical_Analysis` + 各阶段（lowering/scheduler/codegen）/ passes / 动态形状 / 专题(flex/source/debug)

**去冗余**：`PyTorch_Inductor_Technical_Analysis` 2527→1699 行——删除与 pipeline 重复的 stage 流程走读，重定位为「Inductor 后端选择与 IR 优化深度」（后端选择/配置、IR 数据结构、融合成本模型与坐标下降 autotune、常量折叠、内存规划/内存池、CUDA Graphs 集成、后端扩展，均 pipeline 未展开）。

**NPU / upstream 分离**：
- Technical 的 NPU 适配（后端注册 `NPUDeviceOpOverrides`、初始化 hook、RNG patch、`config.device="npu"` 等）→ 迁入 `npu/NPU_Inductor_Backend_Analysis`（→2440 行）
- `Pytorch_Compile_Debug_Analysis` §11「NPU 特有调试」（~326 行）→ 新建 `npu/npu_debug_guide.md`（quick start），原页 897→575 行变纯 upstream
- `scheduler_analysis` §9.3「新设备 backend 注册」示例泛化为设备无关（`MyDeviceScheduling`/占位 `mydevice`），NPU 真实实现指向 npu/
- 各 upstream 文档 NPU 提及降至仅指针级：Technical 66→4、Debug→7（共享双平台脚本）、pipeline/source＝0
- 复核确认 passes 三件套、动态形状三篇此前的「NPU 计数」实为 `input` 子串假阳性，本就纯 upstream

**索引**：`04_inductor/index` 与 `04_inductor/npu/index` 重排为 overview→quick start→deep dive，分组清晰。

**校验**：01_ai_frameworks 全量 wikilink 零断链；13 个 `PyTorch_Inductor_Technical_Analysis` 入站链接保持有效（未改名无需 repoint）。

---

## 2026-06-14: PrivateUse1 接入面 9 接入点补「为什么·深入（根本原因）」

**Type**: Page Deepening（9 个 agent 并发挖根因；基于本地 checkout pytorch `trunk/6f26be8` + torch_npu 逐条核对 `file:line`，配 RFC/PR/dev-discuss/官方博客；遵「只扩展不删除」铁律，原「为什么」一句全部保留）

**更新页**：`02_engineering/01_ai_frameworks/01_dispatcher_and_device/privateuse1_device_integration_analysis`（212→277 行）——为 §1 设计哲学、9 个接入点、§12 运行时组件各加一节根本原因分析：
- **§1 设计哲学**：DispatchKey 是稀缺 64-bit 资源（backend 槽 ≤16 且已满）→ 预留 PrivateUse1/2/3 匿名占位 key；对比"全员上游"撞 key/带宽墙、"各自 fork"碎片化；placeholder→100+ PR 升一等公民；Authenticity/dogfooding
- **Device**：c10 不链接加速器库 → 按 DeviceType O(1) 虚分派；`exchange_device` 专为 RAII 恢复设计、`device_count` noexcept；CUDA12 eager-context 与 fork 防护
- **Guard**：编译期不知后端 → `DeviceGuardImplInterface` 注册表；`InlineDeviceGuard<CUDAGuardImpl>`(去虚化) vs `VirtualGuardImpl`(虚分发) 的性能/可扩展分界
- **Hooks**：编译期解耦下的控制反转（IoC），补「无 device key 可路由」的通用路径（Generator/pin_memory/init）
- **Operators**：分层 alias key + redispatch（反向白送 / composite 自动分解 / cpu_fallback）+ 不可约核心算子
- **AMP**：autocast 作为独立 dispatch key，cast 是算子属性而非设备属性，上游定策略后端填 dtype
- **Autoload**：注册依赖 import 的鸡蛋问题、`entry_points(group="torch.backends")`、隐式加载代价
- **Profiler**：`ProfilerStubs` 依赖倒置、event 异步计时、legacy fallback vs kineto `IActivityProfiler` 门槛（含一处精度澄清）
- **Distributed**：集合通信语义/实现解耦、`Work` 异步句柄、device→backend 解析
- **CI**：扩展点无编译期强约束 + 第三方管不住上游 → OpenReg 作可执行规格
- **§12 运行时组件（逐组件根因）**：Allocator（caching/`recordStream` 防 use-after-free；OpenReg 仅落接口、caching 仍 TODO）、Host pinned（DMA 要 page-locked）、Stream（值类型 + StreamId 位编码 vs `pack3` 三字段）、Event（lazy 创建 + EventPool）、Generator（graph-safe RNG：seed/offset 放设备内存防 replay 不推进）、Serialization（`register_package` + `TensorBackendMetaRegistry` 持久化设备私有 format）、Exception（C 错误码→`c10::Error` + 异步 `PeekAtLastError`）

**校验**：抽样核对 `file:line`（`DeviceGuardImplInterface.h:382/200/388`、`DispatchKey.h:332/345`、`autocast_mode.h:478`、`torch/__init__.py:3025`、`profiler.py:315`、`run_test.py:982-986` 等）通过；页 259 行，未触 500 行拆分阈值。

---

## 2026-06-13: 补全各模块 quick start 层(overview→quick start→deep dive 三层闭环)

**Type**: Layering Completion（5 个 agent 并发创建；所有 API/config/env/flag 对照 pytorch upstream / torch_npu v2.7.1 源码逐一核实、引用 path，无杜撰）

**新增 quick start 页（代码核实）**：
- `04_inductor/inductor_quickstart` —— `torch.compile` 参数（mode/dynamic/fullgraph/options）、`_inductor.config` 关键项、TORCH_LOGS、缓存、mode 选型
- `02_dynamo/dynamo_quickstart` —— `explain()`、graph break 定位、`fullgraph`、guards/recompiles、disable/allow_in_graph/reset
- `03_aot_autograd/aot_autograd_quickstart` —— `backend="aot_eager"` + `TORCH_LOGS=aot_graphs/aot_joint_graph`、partitioner（min-cut vs default）、`aot_function`
- `01_dispatcher_and_device/device_integration_quickstart` —— PrivateUse1 最小接入（基于 torch_openreg）、9 接入点、dispatch 排查命令
- `05_codegen_backends/mlir/npu/npu_mlir_quickstart` —— 启用 MLIR 后端（`TORCHINDUCTOR_NPU_BACKEND`）、anir config、bishengir flags、autotune、精度校验

**索引分层**：01-08 各模块索引补「层次」列与 quick start 入口，形成 overview（索引/概览页）→ quick start → deep dive 三层闭环；根索引已有「知识分层约定」。

**inductor 去冗余确认**：Backend_Analysis/Mechanism、scheduler_fusion 已于前次合并；`inductor_compiler_pipeline_analysis` 与 `PyTorch_Inductor_Technical_Analysis` 互补保留（流程 vs 综合参考，服务不同读者），`torch_compile_architecture`↔pipeline 为有意的 overview↔deepdive 分层。

**校验**：01_ai_frameworks 全量 wikilink 零断链（唯一 `[[maybe_unused]]` 为 C++ 代码块内属性，非链接）。

---

## 2026-06-13: 冗余文档合并 + overview→quick start→deep dive 分层

**Type**: Redundancy Consolidation（4 簇并发分析 + 4 个执行 agent 保守合并；铁律＝保留全部独有信息、仅去重叠；不重命名既有 deepdive 以护 basename 链接）

**合并（7 篇并入，内容无损）**：
- 04_inductor/npu：`NPU_Inductor_Backend_Mechanism`（25-35% 重叠）→ `NPU_Inductor_Backend_Analysis`（并入 MultiTemplateBuffer / Prologue Fusion / 4 实战场景 / 融合性能 / 配置；2265 行）
- 04_inductor：`scheduler_fusion_strategies` → `scheduler_analysis`（并入自定义融合 Pass + 排查指南）
- 05_codegen_backends/mlir/npu：`npu_mlir_backend_deep_analysis` + `npu_mlir_pipeline_analysis`（65-75% 重叠）→ `npu_mlir_backend_technical_analysis`（并入社区遵循/打破、三层 Pass、15 patch 分组、双通道 fallback、六阶段主线、演进建议；1400 行）
- 06_graphs/cuda：`SUMMARY`（99% 同 README）→ `README`
- 06_graphs/npu：`npugraphs_memory_management_analysis`（60%）→ `npugraphs_memory_reuse_analysis`；`torch_compile_mode_reduce_overhead_vs_backend_npugraphs`（45%）→ `torch_compile_npugraphs_deep_dive`（附录 A：双路径对比）

**保留（互补不冗余）**：inductor 通用「管线 / 技术分析」二分、passes 三件套、动态形状三件套、MLIR 通用三篇、`aclgraph` + `aclgraph_deep_analysis`（天然 overview/deepdive）。

**分层**：各模块索引新增「层次」列（overview→quick start→deep dive）；根索引补「知识分层约定」章节；硬件子索引按层次重排。

**收尾**：全部入站 wikilink repoint（含跨域 megatron `[[SUMMARY]]`→`[[06_graphs/cuda/README]]`）；01_ai_frameworks 内容页 52→45；全量 wikilink 零断链；`wiki/index.md` 计数更新（45/21/10/4）。

---

## 2026-06-13: 知识诊断与自动修复（对照 upstream + torch_npu v2.7.1 源码）

**Type**: Source-Verified Correction（9 个只读 agent 全量核验 + 5 个修复 agent 精准订正；基准 pytorch upstream / torch_npu v2.7.1.post5 / op-plugin）

**删除杜撰**：
- CUDA Graphs「方式5: experimental 参数」整段系完全杜撰（torch.compile 无 experimental 参数，签名仅 backend/mode/options…）——跨 4 文件删除约 674 行（Complete_Guide / Timing_Diagrams / SUMMARY / README），并重排方式编号、删相关表列/场景/TOC；伪代码 C CUDA API 名改为 PyTorch `CUDAGraph` 方法。
- MLIR 后端「`_triton.has_triton = lambda: False` 强制禁用 Triton（npu_inductor_plugin.py:68-69）」系杜撰：该处实为 `atexit.register(shutdown_compile_workers)`，且 ascend_npu_ir 插件无 has_triton 赋值；真实门控为 `_inductor/utils.py:25-63 patch_has_triton()`，对 NPU **返回 True**。订正跨 3 文件的代码块/表/叙述（改为「MLIR 后端改用 MLIR codegen 旁路 Triton，has_triton 仍 True」）。

**订正过时数值/路径**：
- NPU aten fallback 计数：859 / ~635(289+346) → 实测 **963**（TORCH_NATIVE 348 + NPU_EXTRA 615，截至 v2.7.1），跨 3 文件。
- MLIR npu 行数：npu_inductor_plugin.py 474→461、inductor_patch/lowering.py 7440→7505、mlir.py 469→141。
- Dynamo 页错误源码路径前缀 `file:///e:/97-codes/torch_parallel/pytorch` → `E:\97-codes\pytorch\pytorch`（24 处）。
- AOTAutograd「Phase 0: create_aot_state」误述 → 标注为 aot_function 内部初始化、非独立编译阶段。
- inductor `select_decomp_table()` 位置 compile_fx.py:2686 → decomposition.py:972。
- post_grad FSDP2 pass（remove_fsdp2_unsharded_param_graph_input_usage）当前源码无 → 标注已移除/重构。
- op-plugin 手写 opapi 计数 352 → ~356（补严格 `*KernelNpuOpApi.cpp` 命名约 300 的口径）；CUDACombinedScheduling 行号 23→24。
- comparison.md NPU 捕获时序图：aclopExecute→aclnn 记录、删 aclmdlRIInstantiate（model_ri 于 capture_begin 创建）、修 ` ```mermer ` 渲染错误、过时 contradiction 标注转 note。

**标注/软化**：torch-mlir 上游路径标注「本地不可验证」；过期日期（2026-05-08 等）软化为「截至 …」；tilelang/mindspore 页头加「概念级、本地无源码」说明；operator_optimization 区分 AICPU 与 host CPU fallback。

**核验准确（未改）**：dispatcher（DispatchKeySet/优先级）、op-plugin 注册链路与 NPUGraph.cpp 行号、NPU Graphs 9 篇中 8 篇、inductor 多数概念页；`isnan` 确认为真实逻辑缺陷。

**校验**：01_ai_frameworks 全量 wikilink 零断链。

---

## 2026-06-13: 01_ai_frameworks 按 PyTorch 架构重组（功能目录 + 硬件子目录）

**Type**: Structural Reorganization（目录重构，git mv 保留历史；内容不变）

**动机**：原结构（cudagraphs/inductor/mlir/op_plugin + 根级散页）将通用机制与 CUDA/NPU 硬件特定内容混排。改为按 PyTorch 编译/运行时架构分功能目录，硬件特定内容下沉到各功能目录的 `npu/`、`cuda/` 子目录。

**新结构**：`01_dispatcher_and_device/`、`02_dynamo/`、`03_aot_autograd/`、`04_inductor/`(+`npu/`)、`05_codegen_backends/mlir/`(+`npu/`)、`06_graphs/`(`cuda/`+`npu/`)、`07_op_registration/npu/`、`08_kernel_optimization/`、`09_other_frameworks/`。

**迁移**：52 内容页 + 3 `.py` 经 `git mv` 迁移；重写 16 个 `index.md`（每目录一入口，含硬件分层约定）；裸 `[[index]]`→`[[01_ai_frameworks/index]]`（18 处）；修 `operator_optimization_guide` 相对/路径限定链接；更新 `wiki/index.md` 顶层入口与页数（52）；修 changelog 历史 `[[op_plugin/index]]`→新路径。

**校验**：全库 wikilink 扫描，重组区零真实断链。

---

## 2026-06-13: 升级 NPU Inductor 优化思想页 —— 新增 §十二「实战：从源码看优化案例」

**Type**: Source-Verified Augmentation（本地 `pta_suhaibo/torch_npu` checkout **v2.7.1** / commit `8bcbe1939` 逐行核验，可 `git grep` 对照；区别于 §一–§十一 基于来源文档的指示性行号）

**更新文件**：

- `inductor/npu_inductor_optimization_analysis.md` —— 新增 §十二「实战：从源码看优化案例」：① `mm`/`addmm`→CATLASS 全链路（decomposition 排除 → 连续守卫 → Cube 模板门控 → autotune → epilogue → ACLNN 兜底）；② 规约类（`mean` 全程 fp32 / `tile_generator` UB 公式 `max_numel_threshold = ub_size//ptr//dtype` / 何时关 persistent / `native_layer_norm` 条件退 ACLNN / cumsum int64→int32）；③ elementwise（`tl_math.*` 覆写 / `expm1` 分解）；④ 融合 pass 范式（`register_custom_pass` 二维注册表 + `is_inference_check` 门控 + `SHUT_DOWN_FX_PASS_LIST` 开关；14+ fold pass 清单；dtype_optimal int64→int32 / fold_sink_view / unfold_dual_reduction 三个范式）；⑤ 案例→硬件思想映射表。同步更新页首代码位置 note（§十二 行号已核验）。

---

## 2026-06-13: 补充 inductor fallback 与 aclgraph 捕获门禁（当前源码 a6655d4 复核）

**Type**: Source Re-verification（基于 torch_npu 当前源码 `a6655d4` + pytorch fork `9922478` 的逐行复核；非 `raw/` 源，行号以该 commit 为准；既有相关页多基于 2.7/2.7.1，故为「校正 + 补充」）

**扩展既有页面（不新建、不删除原文）**：

- `02_engineering/01_ai_frameworks/inductor/npu_lowering_guide.md` —— 新增 §9「当前源码复核（a6655d4）」：① 校正 `_register_npu_inductor_fallbacks` 当前为**纯黑名单**（白名单语义已移至 `ascend_npu_ir` 后端，两后端策略相反）② `FALLBACK_LIST` 两半（`TORCH_NATIVE` = GPU 也 fallback 的复杂算法 / `NPU_EXTRA` = 昇腾 triton 未支持）③ `TORCH_NATIVE` ↔ 上游 `make_fallback` **六分类**映射（含 `# 5) Impossible (missing triton features)`）④ 校正间接访存当前为 `INDIRECT_MEM_FALLBACK_LIST`**黑名单** + A2/A3 vs A5 ⑤ `embedding+sum` 融合收益**实测**（eager 440 / fallback 1209 / 融合 260 us，反驳「融合没收益」）⑥ `isnan` 疑似 bug（`:1011` 条件比错变量）
- `02_engineering/01_ai_frameworks/cudagraphs/npugraphs/aclgraph_deep_analysis.md` —— 新增「差异 8：aclop/aclnn 捕获门禁」：只有 aclnn 能入图、aclop 因运行时 JIT 被禁（`OpCommand.cpp:135-139`）、internal_format 放大、`capture_begin` 前置（`TASK_QUEUE_ENABLE≠2` / 非默认流 / `IsCaptureSupported`）、RNG 捕获期禁用

**交叉引用**：

- `npu_lowering_guide` ↔ `aclgraph_deep_analysis` 互加 Related Pages —— aclnn/aclop 是 inductor fallback 关与 aclgraph 捕获关的**公共枢纽**（fallback 到 aclop 会同时破坏融合与捕获）；`npu_lowering_guide` 增链 `npu_inductor_optimization_analysis`

**矛盾标注（保留双方）**：

- `cudagraphs/npugraphs/comparison.md` 捕获时序图（`:236` `aclopExecute (记录到图)`、`:246` 独立 `aclmdlRIInstantiate()`）与源码不符：捕获期 aclop 被禁、`model_ri` 于 `capture_begin` 即创建（三级 API）——已在 `comparison.md` 与 `aclgraph_deep_analysis.md` 双向 `> [!contradiction]` 标注

---

## 2026-06-13: 新增 NPU Inductor 优化思想全景（硬件驱动）

**Type**: Knowledge Synthesis（源自外部文档体系 GitCode `anyrenwei/Ascend-Related-Docs` 的 `ascend/torch_inductor/inductor/` 全系列，6 个并行子代理跨 02–09 + tiling-comparison/dynamic-shape/refactor-design 提取 80+ 优化点后综合；非 `raw/` 源，基于 torch_npu 2.7 分支；达芬奇架构为背景知识，行号指示性）

**新增文件**：

- `02_engineering/01_ai_frameworks/inductor/npu_inductor_optimization_analysis.md` —— 把抽象「优化思想」逐条落到达芬奇硬件特性上，按「**硬件特性 → 优化思想 → 实际案例**」组织、跨 Triton/MLIR/DVM 三后端：① 编译时驱动（两阶段 tiling/TileGenerator）② 塞满 UB·仅 persistent 规约（UB 公式 block + no-loop）③ 连续访存（golden_var_list + 显式 permute）④ Cube 专用模板（CATLASS + EVG epilogue）⑤ fp32 中间精度（sum/mean/tanh clamp/bf16 promote）⑥ 能力门控 + 分解阶梯（~635 fallback、decomp 13/9/45）⑦ 可信硬件度量（AICore profiler 计时）⑧（非硬件）工程优化（origin tracking O(n²)→O(n)）；收尾「动态 shape 是编译时驱动的反噬」+ 四改进方向。含 2 个 Mermaid（AI Core 结构 / 硬件→思想→案例映射）

**索引与交叉引用**：

- `inductor/index.md` —— NPU 后端节新增该页（标注「优化思想全景 why」，与既有「what/how」页互补）；最后更新 2026-06-13
- `inductor/npu_inductor_splittiling_backend_analysis.md` —— Related Pages 新增反链（本页「why」与该页「what/how」互补）

**矛盾标注（保留双方）**：

- fallback / patch 计数口径差异——本页（2.7 来源）fallback ~635 / patch 30+，本库 [[11_npu_inductor_splittiling_backend_analysis]]（v2.7.1 源码核查）fallback 859 / patch 35+；已在页内 `> [!contradiction]` 标注，深入以 v2.7.1 源码页为准

---

## 2026-06-13: 新增 PyTorch Dispatcher 算子分发机制深度分析

**Type**: Knowledge Synthesis（源自 PyTorch 源码 `c10/` + `aten/` + `torch/csrc/` 的问答整理稿；本机未装 torch，§11 代码示例输出为手算预期值）

**新增文件**：

- `02_engineering/01_ai_frameworks/pytorch_dispatcher_analysis.md` —— 覆盖 Dispatcher 设计动机、核心数据结构（DispatchKey/DispatchKeySet/OperatorEntry/KernelFunction boxed-unboxed）、调用流程 + redispatch 洋葱、**深入①** requires_grad 算子 Python→CUDA 逐层调用栈、分发顺序四要素（枚举优先级 / TLS / fallthrough-fallback / alias key）、**深入②** native_functions.yaml + torchgen 代码生成、C++→Python 类关系、注册与调用接口、自定义分发（`__torch_function__` / `__torch_dispatch__` / `TorchDispatchMode` / 自定义后端 key）、**深入③** 三个可运行示例（FlopCounterMode / relu→gelu 替换 / LoggingTensor 子类）

**索引与交叉引用**：

- `02_engineering/01_ai_frameworks/index.md` —— 新增「核心运行时（Dispatcher）」子节；最后更新 2026-06-13
- `02_engineering/01_ai_frameworks/inductor/aotautograd_analysis.md` —— Related Pages 新增反链（AOTAutograd 用 `__torch_dispatch__` 追踪联合图，是该机制的直接消费者）

---

## 2026-06-12: 入图判别页勘误——§8.2 修正「Inductor 边界 vs Triton 语言上限」的归因

**Type**: Errata(PyTorch 上游源码核查 `pytorch/torch/_inductor/`,逐条带 文件:行号):原表述把「只能降解为 loop IR」隐含归因到 Triton 语言能力,经核查应修正为 **TorchInductor 自动 lowering+codegen 的设计边界**——Triton 语言本身能手写 matmul/flash-attention(triton tutorials + PyTorch 的 `.jinja` 模板本身即手写 Triton);仅复数/稀疏/部分 fp8 那类才是真后端特性缺失

**更新文件**:

- `op_plugin/npu_operator_graph_eligibility_guide.md` §8.2 —— ① 开头改用 Inductor loop-level IR 的准确定义(`ir.py:989 class Loops` docstring;`Pointwise/Reduction/Scatter/Scan/Sort`)+ `make_fallback→FallbackKernel`(`ir.py:8765`)机制;② 新增两个 callout:「关键澄清:Inductor 边界 ≠ Triton 上限」(mm 模板即手写 Triton,`mm.py:85`+`triton_mm.py.jinja`)与「对 NPU 适配的含义」(fallback 多为工程投入问题,torch_npu `_inductor/lowering.py:227-994` 即在补 NPU lowering;第二关是「软」边界,区别于第一/三关的硬约束);③ 标注 sort/topk/conv-backward/cumsum 为**条件性** fallback(`ir.Sort`/`ir.Scan` 可 codegen),非无条件退回

---

## 2026-06-12: 入图判别页深化——补「三关硬性不变量(为什么进不去)」+「新算子前瞻判据」(§8/§9)

**Type**: Knowledge Synthesis(机制根因核查,逐条带 文件:行号):aclop 不可 capture 的根因 = `aclopCompileAndExecute` 运行时编译+执行融合 + 释放 GIL(OpParamMaker.cpp:144 注释) + OOM 重试 host 控制流,而 aclnn 是两段式(GetWorkspaceSize 算 tiling / aclnnXxx 只塞预编译 kernel)纯异步 task;inductor FALLBACK_LIST 两类根因 = IR 表达力边界(TORCH_NATIVE,GPU 也 fallback) vs 昇腾 intrinsic 缺口(NPU_EXTRA,超越函数/位运算 GPU 当 pointwise 但 triton-ascend 缺 libdevice);dynamo = 输出元数据须可符号推导

**更新文件**:

- `op_plugin/npu_operator_graph_eligibility_guide.md` —— 新增 §8「三关的硬性不变量:为什么进不去」(不变量层层收紧:第一关形状可预测 / 第二关计算可表达 / 第三关执行可录制;aclnn-only 铁律的 `aclopCompileAndExecute` 根因;`allow_internal_format=False` 为何救场;TORCH_NATIVE vs NPU_EXTRA 的「通用限制 vs 昇腾待补齐」之分;A2/A3-SIMD vs A5-SIMT 间接访存硬件根因)+ §9「面向新算子的前瞻判据」(决策树 Mermaid + 三关自检 checklist);原速查表顺延为 §10;目录补两项;新增 [[25_unbacked_symint_analysis]] 交叉引用

---

## 2026-06-12: 新增 op-plugin 算子接入域(3 篇 + 目录)——配置分类 / 注册链路 / 入图判别

**Type**: Knowledge Synthesis(源自 `E:\97-codes\pytorch\torch_npu` 当前 checkout 的多代理源码核查:op-plugin codegen、torchnpugen、_inductor、NPUGraph、_meta_registrations 等,逐条带 文件:行号 证据)

**新增文件**(`02_engineering/01_ai_frameworks/op_plugin/` 为新建目录,4 篇 `.md`):

- `op_plugin/index.md` —— 域入口:从 yaml 到入图的一图概览 + 三篇导航 + 「这一域回答什么」对照表
- `op_plugin/op_plugin_config_and_classification_guide.md` —— config 五文件字段;official/custom/symint(正交维度纠正)/quant;acl_op(aclop) vs op_api(aclnn);gen_opapi 结构化 vs 手写适配(「过适配」澄清);看一条 func 配置就分类的四维速查表
- `op_plugin/op_registration_pipeline_analysis.md` —— 两段 codegen 串联;生成产物(RegisterNPU.cpp/CustomRegisterSchema.cpp/custom_ops.py);**TORCH_LIBRARY=静态初始化「库加载即注册」**;编译期→加载期(import torch_npu 时 dlopen libtorch_npu.so 触发静态初始化)→运行期时间线;acl_op/op_api 运行时三层选择;official/custom 两条完整调用链
- `op_plugin/npu_operator_graph_eligibility_guide.md` —— 入图四路线总览;非 torchair 三关递进流水线(dynamo meta / inductor lowering+fallback / aclgraph aclnn-only 铁律);每关判别命令(TORCH_LOGS、has_kernel_for_dispatch_key、lowering.fallbacks、allow_internal_format);op_api/acl_op 贯穿主线

**索引与交叉引用**:

- `01_ai_frameworks/index.md` —— 子目录表新增 `[[07_op_registration/npu/index]]`;页面列表新增「op-plugin 算子接入」区(3 行);页头摘要与最后更新改 2026-06-12
- 交叉引用:三篇互链,并 link 到既有 [[01_npu_compile_paths_overview]] / [[11_npu_inductor_splittiling_backend_analysis]] / [[10_aclgraph_deep_analysis]] / `PyTorch_Dynamo_Technical_Analysis`（该页已于 P4 Task 5 判重删除，内容并入 [[02_compile_stack/01_dynamo/index]]） / [[20_npu_lowering_guide]]。入图判别页明确定位为「判别视角」,与既有「路径实现全景」页互补、不重复

---

## 2026-06-12: FSDP 深挖篇勘误——"分配 ≠ 新建":两层复用与社区机制(§5.5)

**Type**: Errata + Knowledge Synthesis(源码新核 5 处:`init_all_gather_outputs` 早退守卫、`alloc/free_storage`=`resize_`、`_set_unshard_async_op` 跨流碎片说明、`set_custom_all_gather`/`allocate()` 钩子、`set_allocate_memory_from_process_group`)

**更新文件**:

- `torchtitan/20_torchtitan_fsdp_prefetch_overlap_memory_analysis.md` —— ① §5.2 修正误导表述:"+p" 是显存占用增量而非"每次新分配"(逐参数 buffer 张量仅首迭代创建,此后 storage resize 0↔满;扁平 buffer 物理块稳态来自 caching allocator 池命中,无 cudaMalloc);② 新增 §5.5 勘误与补充:两层既有复用、FSDP 为何不自管持久池(allocator 等效/跨流 event/尺寸不齐/reserved 反升)、社区机制清单(storage-resize、expandable_segments、async_op 挪流、custom allocate 钩子、PG 缓冲注册、MemPool、compile 消 resize+copy、Megatron 持久缓冲先例)、自建持久池的场景判断(NPU 栈最值得);③ §7 复核表扩 5 行;页头日期更新

---

## 2026-06-11: torchtitan 系列新增两篇深挖伴篇(FSDP 预取/掩盖/显存、激活重计算 AC)

**Type**: Knowledge Synthesis(源自 torchtitan `cf3c4312` + PyTorch 2.9.1 源码逐行核验的问答整理稿,配 SVG→PNG 机制图)

**新增文件**(2 篇 `.md` + 16 个图文件,`torchtitan/assets/` 为新建目录):

- `02_engineering/02_train_frameworks/torchtitan/20_torchtitan_fsdp_prefetch_overlap_memory_analysis.md` —— [[11_torchtitan_fsdp_analysis]] 的深挖伴篇(2 图):串行 vs 多流预取掩盖时序、唯一跨流同步点 `wait_event(_fsdp_collectives.py:361)`、copy-in 三步(narrow 视图巧思 + `_foreach_copy_` 方向)、flat 双缓冲 ping-pong(为何延迟释放)、"完整参数 ≤2 份不会 3 份"的 reshard-先于-unshard 时序证明、CI/AG/CO 各阶段显存账
- `02_engineering/02_train_frameworks/torchtitan/22_torchtitan_ac_analysis.md` —— 激活重计算原理 + 代码解读(6 图):AC vs DCP 两种 checkpoint 区分、`checkpoint_wrapper` 接口链路(module 在两次 `next(gen)` 之间跑)、票据机制(`weak_holders`/`recomputed`/`recomp_counter` 下标对齐,发票→重算绑票→兑票)、SAC 双 dispatch mode 缓存回放 + torchtitan policy(奇偶 mm/SDPA/comm 恒存)+ attention 端到端走查、显存预估三法(full 手算 / SAC 加总 save-op / memory_budget Pareto)、粒度控制五法(含 config 驱动模块级方案)、横跨 autograd(`saved_tensors_hooks`)×dispatch(`TorchDispatchMode`)两核心、`ActivationCheckpointConfig` 全字段速查

**索引与交叉引用**:

- `torchtitan/index.md` —— 新增「深挖伴篇」表(2 行);系列篇数 7→9;并行施加管线 `apply_ac()` 挂链;Related Pages 补两页;最后更新 2026-06-11
- `11_torchtitan_fsdp_analysis.md` —— Related Pages 首行新增深挖伴篇反链
- `01_theory/02_pretraining/12_activation_checkpointing_analysis.md` —— Related Pages 首行新增 [[22_torchtitan_ac_analysis]](工程侧非重入/SAC,与该页 Megatron 重入路径互补)

---

## 2026-06-09: HTML 报告转 Markdown 并替换原 HTML(SVG/CSS 图 → PNG)

**Type**: 格式迁移 + 文件替换(为移动端阅读把 9 篇 HTML 报告转为 Markdown,图渲染为内嵌 PNG;转换验证无误后删除原 HTML,仓库只保留 Markdown)

**转换方式**: 用无头 Edge(`puppeteer-core`)加载每页 → 强制 light 配色并禁用 reveal 滚动动画 → 对每个图形元素按元素截图为 2× PNG(完整保留 CSS 变量配色与字体)→ DOM 规范化(callout→blockquote、TOC→列表、`<pre>`/`white-space:pre` 容器→围栏代码块、标题副标题、figure→`<img>`)→ Turndown + GFM 转 Markdown。图片存于各目录 `assets/`。

**新增文件**(9 篇 `.md` + 44 张 PNG,取代同名 `.html`):

- `02_engineering/02_train_frameworks/`:`21_async_collective_tensor_deepdive.md`(4图)、`30_comm_compute_overlap_analysis.md`(7)、`35_deepseek_v4_context_parallel_analysis.md`(6)、`34_deepseek_v4_tensor_parallel_analysis.md`(1)、`32_distributed_optimizer_deepdive.md`(7)、`20_megatron_pp_parallelism_analysis.md`(4)、`mindformers_moe_token_dispatcher_analysis.md`(7)、`22_muon_sharded_hsdp_analysis.md`(6)
- `02_engineering/05_gpu_kernel/`:`gpu_kernel_guide.md`(2,`tier-diagram` 与 FlashAttention `fa-flow` 两张 CSS 图)

**删除文件**: 上述 9 篇对应的 `.html` 原件(`21_async_collective_tensor_deepdive.html` 等 8 篇 + `gpu_kernel_guide.html`)。

**索引与链接更新**:

- 全库 Obsidian 维基链接统一从 `[[*.html]]`(及 `[[*.html|别名]]`)改写为 `[[*]]`,共 13 处,分布于:`02_engineering/index.md`、`02_train_frameworks/index.md`、`05_gpu_kernel/index.md`、`torchtitan/index.md`、`megatron-lm/index.md`,以及 torchtitan `cp/ep/fsdp/pp/tp` 五篇分析页与 `16_megatron_distributed_optimizer_analysis.md` 的交叉引用
- `wiki/index.md`(总索引)— 目录树补入 `torchtitan/` 与 `05_gpu_kernel/`;领域总览表新增「torchtitan(7)」「GPU Kernel(1)」两行;去掉「后训练框架(预留)」过时标注
- 相关 index 的「最后更新」统一改为 2026-06-09

**说明**: 原 `.html` 已删除,仓库仅保留 Markdown(移动端阅读首选)。转换为忠实迁移——所有 SVG/CSS 图表已逐张校验为 PNG,代码块围栏完整,技术内容未改写。

---

## 2026-06-06: MindFormers MoE 去冗余 Token Dispatcher 源码图解(HTML)

**Type**: Knowledge Synthesis(基于 MindFormers `master` `mindformers/parallel_core/training_graph/transformer/moe/token_dispatcher.py` 中 `MoEAlltoAllDeredundencyTokenDispatcher` 的源码级图解,对照 torchtitan `token_dispatcher.py` 的 AllToAll/DeepEP 路径)

**新增文件**:

- `wiki/02_engineering/02_train_frameworks/mindformers_moe_token_dispatcher_analysis.html` — 7 张手绘 SVG + 逐行代码解读的深色报告。覆盖:两级专家并行布局(oep 跨机 / iep 机内 + 专家按节点分块 `[a,b)`)、一个 token 的两跳旅程、dispatch 全流程 6 个集合通信(3×AllGather + 3×AlltoAllV)、去冗余四步(sort→mask→NonZero→IndexSelect)、计数转置 `[B]` 与 D2H 为何免/需(`iepones` 常量 vs `exsl/exrl` 变长 + `Depend` overlap)、combine 的 `ReduceScatter` top-k 求和(零画板)、combine 梯度反向 adjoint(RS↔AG、scatter↔gather、`mul(probs)` 分叉出 dprobs 回流 router)

**索引更新**:

- `wiki/02_engineering/02_train_frameworks/index.md` — 页面列表新增 `mindformers_moe_token_dispatcher_analysis.html`;最后更新日期改为 2026-06-06

**主线观点**: 两级 EP 的设计目标是把「不规则 + D2H」关进快的机内 NVLink(变长 AlltoAllV),跨机 IB 只走定形、免 D2H 的规则 collective(AllGather/ReduceScatter);去冗余 = 跨机全量 AllGather + 本地 mask 筛选,以通信冗余换取规则性与零 D2H。

**交叉引用**: 与 [[torchtitan/15_torchtitan_ep_analysis]](token all-to-all dispatch/combine、DeepEP/HybridEP)、[[21_async_collective_tensor_deepdive]](ACT 延迟 wait)、[[30_comm_compute_overlap_analysis]](DeepEP/HybridEP 通信掩盖)互为对照(MindSpore 静态图 vs PyTorch eager+compile)。

---

## 2026-05-24: Coding LLM RL「三块脏活」分析(3 篇)

**Type**: Knowledge Synthesis（基于 Anthropic Claude 4.5 model card、Anthropic reward hacking 论文、RollArt/ProRL Agent/RollPacker 等 RL infra 论文及姚顺宇张小珺访谈的综合分析）

**新增文件**:

- `wiki/01_theory/04_posttraining/reward_hacking_defense_analysis.md` — Reward Hacking 防御四层体系：环境加固 / reward penalty / Inoculation Prompting（接种式提示）/ Post-RL agentic safety；含 Anthropic 2025-11 misalignment 泛化论文要点与 Claude 4.5 各档 hacking 率数据（Opus 4.5 18.2% > Sonnet 4.5 12.8% > Haiku 4.5 12.6%）
- `wiki/02_engineering/04_posttrain_frameworks/rl_sandbox_design_analysis.md` — 生产级 RL Sandbox 设计：10 万级并发、Firecracker microVM 选型对比、Disaggregated 架构（training/inference/sandbox 三集群分离）、Rollout 三阶段（init/exec/eval）独立调度
- `wiki/02_engineering/04_posttrain_frameworks/rl_infra_efficiency_analysis.md` — RL Infra 效率五项核心优化：异步训练（off-policy staleness 权衡）、长尾治理（redundant rollouts / trajectory 调度 / 早停 / timeout）、硬件感知调度（H800 prefill / H20 decode）、in-flight reward、environment 池十万级；附「为什么 coding 是第一个起飞领域」整体行业判断

**索引更新**:

- `wiki/01_theory/04_posttraining/index.md` — 新增「对齐安全」小节，加入 reward_hacking_defense_analysis；最后更新日期改为 2026-05-24
- `wiki/02_engineering/04_posttrain_frameworks/index.md` — 拆分为「数值与确定性」「Coding RL Sandbox 与 Infra」两小节，新增 rl_sandbox_design_analysis 与 rl_infra_efficiency_analysis；最后更新日期改为 2026-05-24
- `wiki/index.md` — 后训练对齐页数 13→14、后训练框架页数 1→3；快速导航加入「Coding RL『脏活』系列」一行；最后更新日期改为 2026-05-24
- `wiki/02_engineering/04_posttrain_frameworks/batch_invariance_guide.md` — 相关页面增加 RL Sandbox 与 Infra 两页 backlink

**主线观点**: Coding 大模型训练的护城河来自三块「脏活」——Sandbox 决定能不能稳定跑、RL Infra 决定能跑多大多快、Reward Hacking 防御决定训出来的是不是你想要的。三者强耦合，单点短板即整体瓶颈；国内玩家真实差距在 infra 与 reward 体系而非算法。

**交叉引用**: 三篇互链，并与 `grpo_analysis` / `ppo_analysis` / `dapo_analysis` / `gspo_analysis` / `rlhf_foundations_analysis` / `kimi_k1.5_analysis`（以上 6 处历史活链接已于 2026-07-31 因 kb-reorg P5 Task 8 分段编号，现况见 [[01_theory/04_posttraining/index]]，按"历史不回写"惯例降级为反引号）/ [[batch_invariance_guide]] / `RL_PPO_Loss_and_GRPO_Analysis`（历史活链接，已于 2026-07-31 因 kb-reorg P5 先迁移改名为 `rl_ppo_loss_and_grpo_analysis`、同日 Task 8 再编号为 [[10_rl_ppo_loss_and_grpo_analysis]]，按"历史不回写"惯例降级为反引号）等既有页交叉引用。

---

## 2026-05-22: torchtitan 多维并行体系源码级分析(7 篇)

**Type**: Knowledge Synthesis(基于 torchtitan `main` @ `cf3c4312` 与 PyTorch 2.9.1 FSDP2/DTensor/pipelining 内核的源码级分析)

**新增目录**: `wiki/02_engineering/02_train_frameworks/torchtitan/`

**新增文件**:

- `torchtitan/index.md` — torchtitan 多维并行知识地图:设计哲学(一组 GPU 多重视图)、三张 DeviceMesh、并行施加管线、组合建议
- `torchtitan/10_torchtitan_parallel_dims_analysis.md` — 并行基座:`ParallelDims` 维度约束、`build_mesh` 三张逻辑 mesh(dataloading/dense/sparse)、`fake` backend、mesh 查询接口
- `torchtitan/11_torchtitan_fsdp_analysis.md` — **标杆篇** DP/FSDP2:`FSDPParam` 逐参数切分、`FSDPParamGroup` 分组、all-gather 预取(隐式/显式)、五条 CUDA stream 异步编排、reduce-scatter 梯度规约、反向钩子链
- `torchtitan/12_torchtitan_tp_analysis.md` — TP:`distribute_tensor` 切分、`redistribute` 通信选择、列并行→行并行配对、Sequence Parallel、Async TP(`_micro_pipeline_tp` inductor pass)、Loss Parallel
- `torchtitan/13_torchtitan_cp_analysis.md` — CP:`_context_parallel_shard` 序列切分、HeadTail/PTRR 负载均衡、Ring Attention K/V 环形轮转、在线 softmax 合并、通信掩盖
- `torchtitan/14_torchtitan_pp_analysis.md` — PP:`_split_module` 模型切分、P2P send/recv、调度气泡对比(GPipe/1F1B/Interleaved/ZBV/DualPipeV)、action-based runtime、Zero Bubble(I/W 拆分)
- `torchtitan/15_torchtitan_ep_analysis.md` — EP:`ExpertParallel` 专家权重 `Shard(0)`、token all-to-all dispatch/combine、`AsyncCollectiveTensor` 延迟 wait、shared_experts 通信掩盖、DeepEP/HybridEP、`edp_mesh` FSDP

**统一分析粒度**: 每篇按 `fully_shard` 标杆粒度展开——参数/数据切分 → 通信原语 → 通信掩盖 → 异步实现 → 反向传播,带 `文件:行号` 引用与 ASCII 流程图。

**索引更新**:

- `wiki/02_engineering/02_train_frameworks/index.md` — 子目录表与页面列表加入 `torchtitan/index` 条目

**交叉引用**: torchtitan 系列与 Megatron-LM 源码级系列([[12_megatron_tp_analysis]]/[[13_megatron_cp_analysis]]/[[14_megatron_ep_analysis]]/[[15_megatron_pp_schedulers_analysis]]/[[16_megatron_distributed_optimizer_analysis]])互为对照(PyTorch-native vs CUDA/Megatron 生态),并与 [[21_async_collective_tensor_deepdive]]、[[30_comm_compute_overlap_analysis]] 等既有页交叉引用。

---

## 2026-05-22: Dynamic Shape 体系补充：Unbacked SymInt、XBLOCK 选择机制、GPU vs NPU 对比

**Type**: Knowledge Synthesis（对话探讨中发现的 wiki 空白，补充入库）

**新增文件**:

- `wiki/02_engineering/01_ai_frameworks/inductor/unbacked_symint_analysis.md`
  - **§1**: Backed vs Unbacked 根本区别，产生 unbacked symbol 的 op 类型（nonzero/item/where/unique/masked_select 等）
  - **§2**: 为什么 Guard 机制对 unbacked 无效（执行时机不同）
  - **§3**: ShapeEnv 内部两类约束的存储——`var_to_range`（backed）vs `deferred_runtime_asserts`（unbacked），`guard_or_defer_runtime_assert` 分流逻辑
  - **§4**: Inductor codegen 中 unbacked symbol 在 wrapper 的体现（`u0 = buf0.size(0)` 先读取，后断言）
  - **§5**: `torch._check()` 的三种效果：值域细化、符号替换（消灭 unbacked）、条件记忆（解决控制流）
  - **§6**: `GuardOnDataDependentSymNode` 错误的触发机制（`bool(u0 > 4)` 的调用链），与 backed symbol 的对比
  - **§7**: 相关 API 速查表（`_check`、`_check_is_size`、`constrain_range`、`mark_unbacked`、`statically_known_true`、`guard_or_false`）
  - **§8**: 常见误用与修法（Python 切片、item() 控制流、empty_strided + unbacked stride）
  - **§9**: Backed vs Unbacked 全链路对比表

**更新文件**:

- `wiki/02_engineering/01_ai_frameworks/inductor/inductor_codegen_dynamic_shape_analysis.md`
  - 新增 **§9 XBLOCK 选择机制与 Dynamic Shape 性能代价**：
    - 候选值范围（32–4096，2 的幂次，`TRITON_MAX_BLOCK['X']=4096`）
    - 三种模式对比：heuristics（运行时 lambda）、autotune（benchmark 候选集）、静态特化
    - Dynamic shape 下 hint 截断问题：autotune 候选集基于编译期 hint 生成，运行时大 shape 的最优 XBLOCK 可能不在候选列表
    - 不同 op 类型的影响程度（Pointwise 轻微 / Reduction 中等 / GEMM 严重）
    - `tl.constexpr` 的本质：每个不同 XBLOCK 值对应一个独立 PTX kernel binary

- `wiki/02_engineering/01_ai_frameworks/inductor/npu_compile_paths_overview.md`
  - 新增 **§九 GPU vs NPU Dynamic Shape 难易度对比**：
    - GPU：SIMT 天然参数化，SymInt+ShapeEnv 与硬件特性匹配，主要代价是 CUDA Graph 不兼容和 autotune hint 截断
    - NPU 三层结构性困难：① Cube Core 刚性 tiling 对齐 → padding 破坏 fusion；② ACLGraph 需预知 shape → dynamic shape 下 graph 无法复用；③ 859 op fallback 绕过 SymInt 体系
    - 本质定性：GPU 是软件/编译层问题，NPU 是硬件架构层问题
    - NPU dynamic shape 实践建议（shape bucketing、torchair 路径、避免 dynamic+ACLGraph 组合）

**索引更新**:

- `wiki/02_engineering/01_ai_frameworks/inductor/index.md` — 新增 `unbacked_symint_analysis` 和 `npu_compile_paths_overview` 条目

---

## 2026-05-20: Megatron Nonuniform Tensor Parallelism (NTP) 深度分析

**Type**: Knowledge Synthesis（基于 Megatron-LM dev 分支源码分析）

**新增文件**:

- `wiki/02_engineering/02_train_frameworks/megatron-lm/25_megatron_nonuniform_tp_analysis.md`
  - **§1**: NTP 概念——TP 组级 GPU 故障容错，不同 DP 副本使用不同大小 TP group
  - **§2**: 设计动机——三种故障应对方案对比（全停重启 vs 全量降级 vs NTP），适用场景（硬件故障应急/异构拓扑部署）
  - **§3**: 实现机制——通信组重配置（冷重启 + sys.exit(0)）、参数 split 元数据（ntp_map 仅设 send_splits/recv_splits、不动参数数据）、梯度同步三阶段流程（Spare→Core all-to-all → DP sync → Core→Extra post-sync reshard）、Buffer/Bucket 适配、Transformer Engine userbuffer 适配
  - **§4**: 关键约束——不做参数 resharding、不做优化器状态转换、不做 checkpoint 转换、reduced 副本 OOM 风险、计算不均衡与尾延迟
  - **§5**: 与 Megatron 主流程关系——完全 opt-in/non-intrusive，无侵入 pretrain_gpt/checkpointing/distrib_optimizer/transformer_config
  - **§6**: 总结——NTP 是梯度级 DDP shim，只做通信组重建 + 两次 all-to-all + bucket group 时序控制
  - 含 1 幅 Mermaid 序列图（三阶段梯度同步流程）

**交叉引用更新**:

- `megatron-lm/index.md` — Distributed Parallelism 表格新增条目，Knowledge Gaps 更新（fault tolerance 标记为已解决，NTP checkpoint 转换标记为新 gap），Cross-Domain Links 新增

---
## 2026-05-22(修订): 分布式优化器字节数订正 + 全系列符号记号统一

- **Fixed【矛盾点订正】**: 经源码二次核查,标准 bf16 训练每参数模型态为 **18 字节**(非原稿的 16):bf16 训练强制 fp32 梯度累积(`arguments.py:1296-1310`、`param_and_grad_buffer.py:812`),梯度 buffer 为 fp32(4 字节)。
  - 影响并修正 4 篇:`ddp_optimizer_analysis`(重写,ZeRO 表重算:18Ψ / 6Ψ+12Ψ/dp / 2Ψ+16Ψ/dp / 18Ψ/dp)、`optimizer_internals_analysis`、`ep_analysis`、`pp_schedulers_analysis`。
  - 与存量 `32_distributed_optimizer_deepdive.html`(Adam 18 字节)一致,矛盾消除。
- **Changed【符号记号统一】**: 全 18 篇统一记号:
  - 并行度 → `tp`/`pp`/`cp`/`ep`/`dp`/`vp`/`etp`/`edp`(消除 `p`/`t`/`e`/`v`/`N` 单字母混用)
  - 张量维 → `s`(序列)/`b`(批)/`h`(hidden)/`h_ffn`(FFN 中间维);消除 `S`/`H`/`B` 大写不一致与 `H` 一词二义
  - 参数量统一 `Ψ`(原 `N_params` 并入)
  - 保留:`B`=反向算子(PP F/B/W)、`B`=RowParallel 权重矩阵(TP)、`S`=loss scale(优化器)—— 不同概念,非重复记号
- **Updated**: `ddp_optimizer_analysis.md` 加更正记录头注。

## 2026-05-22: Megatron-LM 源码级系统分析系列(18 篇)入库

- **Source**: Megatron-LM `dev` 分支 commit `ee3f1ff` 源码(代码分析,非 `raw/` 论文)
- **Created**: `wiki/02_engineering/02_train_frameworks/megatron-lm/` 下新增 18 篇 `*_analysis.md`:
  - 并行轴(5):`pp_schedulers_analysis`、`ep_analysis`、`tp_analysis`、`cp_analysis`、`ddp_optimizer_analysis`
  - 编排与补遗(3):`parallelism_orchestration_analysis`、`pp_supplements_analysis`、`tp_fsdp_resharding_supplements_analysis`
  - 性能基建(3):`recompute_analysis`、`optimizer_internals_analysis`、`precision_cudagraph_fusion_analysis`
  - 系统专题(3):`training_stability_observability_analysis`、`rl_posttraining_consistency_analysis`、`inference_engine_analysis`
  - 数据/模型/存档(4):`dataset_analysis`、`packed_dataset_dynamic_cp_analysis`、`model_structure_analysis`、`dist_checkpointing_analysis`
- **Key topics**: PP 5 调度器与气泡推导、EP 3 dispatcher、TP/SP、CP 4 种 cp_comm_type、ZeRO 0-3、进程组编排、Megatron-FSDP、激活重计算、优化器内部、FP8/CUDA Graph/融合、RerunStateMachine(SDC 归因)、RL 训推一致性、推理引擎、序列打包与动态 CP、模型结构(MLA/MoE Router/Mamba)、分布式 checkpoint
- **Companion artifact**: `_pp_sim.py` — PP 调度模拟器,逐 op 解算精确流水线时空图
- **Updated**: `megatron-lm/index.md` — 新增"源码级系统分析系列"章节;Knowledge Gaps 中 Context Parallelism / checkpoint format / Sequence Parallelism 三项标记为已解决
- **Updated**: `02_engineering/02_train_frameworks/index.md` — megatron-lm 子目录条目补注
- **Note**: 18 篇互为 `[[wiki link]]` 交叉引用,自成体系;来源为源码而非论文;ASCII 时空图保留原始可验证形式(未转 Mermaid)

## 2026-05-19: 分片 Muon 与双网格 HSDP 技术报告入库

**Type**: User Contribution（手动入库，分布式优化器技术分析报告）

**入库文件**:

- `wiki/02_engineering/02_train_frameworks/22_muon_sharded_hsdp_analysis.html`
  - **§1**: Muon 算法核心原理（Nesterov 动量 + Newton-Schulz 正交化、按 head/expert 粒度）
  - **§2**: 分片 Muon 的挑战与 all-to-all 解法（gather → N-S → scatter 流程、批量化同形状张量、通信异步化）
  - **§3**: 双网格 HSDP 设计（非专家窄网格 FSDP + 专家宽网格 EP、CP/EP 维度解耦）
  - **§4**: TP 场景覆盖情况分析（Column/Row-parallel 下 N-S 的正确性条件）
  - **§5**: 异步流水线 Gantt 图（顺序执行 vs 通信计算重叠，约 33% 耗时节省）
  - **§6**: 非专家权重分工 N-S 优化方案（消除 k 倍计算冗余，通信量约减半）
  - **§7**: 方案对比总结表（Cursor vs nanoGPT-speedrun vs 分工优化提案）
  - 含 4 幅 SVG 图表（all-to-all N-S 流程、双网格拓扑、异步流水线 Gantt、分工优化对比）

**交叉引用更新**:

- `02_train_frameworks/index.md` — 页面列表新增条目
- `01_theory/02_pretraining/11_muon_analysis.md` — Related Pages 新增回链

---

## 2026-05-19: 算子调优体系指南入库

**Type**: User Contribution（手动入库，算子开发与性能优化指南）

**入库文件**:

- `wiki/02_engineering/01_ai_frameworks/operator_optimization_guide.md`
  - **§1**: 算子编程体系概览（GPU: CUDA/CUTLASS/Triton/TileLang/TVM + NPU: AscendC/TBE/CANN）
  - **§2**: Roofline 性能分析模型（A100/H100/910B Ridge Point、Nsight/msprof Profiling 指标）
  - **§3**: GPU Memory Bound 优化（融合、算法变形、向量化访存）与 Compute Bound 优化（Tiling、软件流水线、Tensor Core）
  - **§4**: 融合算子识别与设计（决策矩阵、常见 Pattern、FX Graph 替换）
  - **§5**: 等价替换寻找方法（数学变形、算法层替换、AutoTuning）
  - **§6**: 昇腾 NPU 优化路径（Da Vinci 架构、AscendC 三段流水、GPU 经验适配表）
  - **§7**: 与 torch.compile 的关系（各框架接入方式、Custom Op 注册）
  - **§8**: GPU/NPU 完整优化工作流（Profile → Roofline → 优化 → 注册 → 验证）

**交叉引用更新**:

- `01_ai_frameworks/index.md` — 编译优化表格新增条目

---

## 2026-05-19: Pin Memory 与内存语义通信分析入库

**Type**: User Contribution（手动入库，综合深度分析）

**入库文件**:

- `wiki/02_engineering/pin_memory_and_memory_semantics_analysis.md`
  - **§1**: Pin Memory 与内存语义通信基础概念（DMA、RDMA Write/Read/Atomic、ibv_reg_mr）
  - **§2**: 传统消息语义 vs 内存语义的局限性分析（NCCL 固定成员、P2P 内存拷贝开销、NIC 厂商碎片化）
  - **§3**: Pin Memory 在 PyTorch DataLoader / DeepSpeed ZeRO-Offload / vLLM KV Offload 中的应用
  - **§4**: 内存语义通信在 vLLM P/D 分离 / Mooncake TransferEngine / DeepSeek DeepEP / 3FS / RLHF 权重同步中的应用
  - **§5**: 两种 Pin Memory 层次区分（CPU DRAM vs GPU HBM Registration）
  - **§6**: 社区应用全景总结与核心趋势

**交叉引用更新**:

- `02_engineering/index.md` — 页面列表新增条目

---

## 2026-05-17: Megatron-LM Pipeline Parallelism 分析报告

**Type**: New Page

**新增文件**:

- `wiki/02_engineering/02_train_frameworks/20_megatron_pp_parallelism_analysis.html`
  - **§1**: PP 进程组与拓扑 (`parallel_state.py`) — 4 类进程组、UCC/NCCL 双后端、辅助通信组
  - **§2**: 4 种调度策略总览 — Non-Interleaved 1F1B / Interleaved 1F1B (VPP) / Combined 1F1B / No-Pipelining
  - **§3**: Non-Interleaved 1F1B 详细执行流 — Warmup/Steady/Cooldown 三阶段公式、SVG 流水线时序图
  - **§4**: Interleaved 1F1B (VPP) — 微批量到 Chunk 的映射表、Bubble 验证与计算并行度提升
  - **§5**: P2P 通信原语 — `isend/irecv` vs `batch_isend_irecv`、交替组策略 (PP=2 优化)、`P2PCommunicator` 封装
  - **§6**: Combined 1F1B (EP 通算重叠) — `AbstractSchedulePlan` 层级别交错、AllGather/ReduceScatter 重叠
  - **§7**: Bubble 分析与通信量公式 — 气泡比推导、非交错 vs 交错对比、P2P 通信量公式
  - **§8**: 激活检查点与内存优化 — Partial AC、Deallocate Pipeline Outputs、Defer Embedding Wgrad
  - **§9**: 细粒度激活卸载 — `PipelineOffloadManager`、D2H/H2D 双流、`post_warmup_callback` 自适应调参
  - **§10**: 配置推荐与决策树 — 4 种典型场景配置速查表

---

## 2026-05-16: DeepSeek-V4 CP 分析报告准确性修正

**Type**: Correction（对已有文档进行准确性审核并修正错误）

**修正文件**:

- `wiki/01_theory/01_models/deepseek/23_deepseek_v4_cp_analysis.md`
  - **修正 1**：源文件路径错误 — `raw/05_model_families/deepseek/DeepSeek_V4.pdf` → `raw/01_theory/01_models/deepseek/DeepSeek_V4.pdf`
  - **修正 2**：Stage 1 Step 3 压缩输出数量错误 — `(c+1) 个 compressed entries` → `1（CSA 重叠窗口）或 2（HCA 无重叠）个 boundary compressed entries`（原公式与 2c tokens / ratio c 的数学不一致）
  - **修正 3**：Stage 2 All-Gather 输出长度公式错误 — `总长度 = P × c`（与 S 无关的常数，量级完全错误）→ `总长度 ≈ S/c，即 P × S/(P·c)`

- `wiki/02_engineering/02_train_frameworks/35_deepseek_v4_context_parallel_analysis.html`
  - **修正 1（§5.2 callout）**：删除误导性"CSA 与 CP 的序列分片策略兼容"表述，改为明确说明当前代码是功能降级版（AllGather 缺失、fill_value 边界填充）
  - **修正 2（§6.3）**：`h_k = 1（MQA）` → `MLA 低秩潜变量压缩效果，通信量等效于 MQA`，避免将 MLA 误称为 MQA
  - **修正 3（§6.4）**：CSA CP 通信量公式标注为"论文设计目标，当前代码未完全实现"，补充实际代码行为（CSA 层跨 rank 压缩 KV 通信量为 0）
  - **修正 4（§5.5 Gap 4）**：P2P 数据量 `ratio × hidden_size` 改为按 Compressor 输入维度分情况讨论，加 ⚠️ 提示需确认实际维度
  - **修正 5（§9.1 特征 5）**："CSA 压缩与 CP 天然兼容" → "CSA 压缩与 CP 的兼容尚未完整实现"，如实反映当前代码状态

---

## 2026-05-15: 知识库目录结构重整

**Type**: Reorganization

**变更内容**:

- **新建 `mlir/` 子目录** (`wiki/02_engineering/01_ai_frameworks/mlir/`)
  - 从 `01_ai_frameworks/` 移入: `mlir_core_concepts.md`、`torch_mlir_pass_pipeline_analysis.md`、`triton_vs_mlir_backend_analysis.md`
  - 从 `inductor/` 移入: `npu_mlir_backend_deep_analysis.md`、`npu_mlir_pipeline_analysis.md`、`npu_mlir_backend_technical_analysis.md`
- **模型论文归位**:
  - `29_engram_analysis.md` → `deepseek/`
  - `10_moba_analysis.md`、`12_kimi_linear_analysis.md` → `moonshot_kimi/`
- **跨域移动**:
  - `31_comm_compute_fusion_guide.md` → `02_train_frameworks/`
  - `mooncake_analysis.md` → `03_infer_frameworks/`
  - `batch_invariance_guide.md` → `04_posttrain_frameworks/`
- **索引更新**: 所有受影响的 `index.md` 已同步更新

## 2026-05-15: DeepSeek-V4 Tensor Parallel 分析重大修正（基于 Megatron-LM dev 源码）

**Type**: Correction / Rewrite（基于实际源码的全面重写，纠正此前推断性分析中的重大错误）

**修正文件**:

- `wiki/02_engineering/02_train_frameworks/34_deepseek_v4_tensor_parallel_analysis.html`
  - **纠正 1**：DSv4 Hybrid Attention 实际强制 `TP size = 1`（`assert get_pg_size(self.pg_collection.tp) == 1`），此前错误推断为 Column+Row Parallel 切分
  - **纠正 2**：`q_down_proj` 为 `tp_group=None` + `parallel_mode="duplicated"`，不是 ColumnParallel
  - **纠正 3**：Compressor (`linear_wkv`, `linear_wgate`) 和 CSAIndexer (`linear_wq_b`, `linear_weights_proj`) 均为 `parallel_mode="duplicated"`，不产生 TP 通信
  - **纠正 4**：mHC 使用原生 `nn.Linear`（非 TP-sharded），依赖 `sequence_parallel` 属性进行梯度同步，不是 Column+Row Parallel
  - **纠正 5**：Routed Expert 的 fused `TEGroupedMLP` 不支持 `TP > 1`（`experts.py:328-329`），此前错误推断为 ETP 切分
  - **修正 6**：通信量分析全面重写——当前实现下 Attention、mHC、Routed Expert 的 TP 通信均为 0，主要跨 rank 通信仅剩 EP All-to-All 和 CP Ring-AG
  - 新增明确的源码引用（文件路径 + 行号）
  - 新增 "关键发现对比表" 和 "Future Work" 章节

**源码依据**:

- `/Users/suhaibo/97-llm/Megatron-LM/megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py:87-88, 172-195, 421-454`
- `/Users/suhaibo/97-llm/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa.py:288-309, 451-474`
- `/Users/suhaibo/97-llm/Megatron-LM/megatron/core/transformer/hyper_connection.py:150-151, 187-200`
- `/Users/suhaibo/97-llm/Megatron-LM/megatron/core/transformer/moe/experts.py:328-329`
- `/Users/suhaibo/97-llm/Megatron-LM/megatron/core/transformer/moe/shared_experts.py:112-159`
- `/Users/suhaibo/97-llm/Megatron-LM/megatron/core/models/gpt/experimental_attention_variant_module_specs.py:183-196`

---

## 2026-05-15: 新增"为什么 V4 选择 TP=1"架构分析章节

**Type**: Enhancement（在修正后的文档中新增深度分析章节）

**更新文件**:

- `wiki/02_engineering/02_train_frameworks/34_deepseek_v4_tensor_parallel_analysis.html`
  - 新增章节 **"为什么 V4 选择 TP=1：架构与工程考量"**（位于"关键发现"与"Attention 层"之间）
  - 从 5 个维度系统分析 TP=1 的深层原因：
    1. 压缩操作的全局性（Compressor softmax 归约、Indexer Top-K 无法在 TP 边界分片）
    2. q_down_proj 的 duplicated 设计（需要完整 hidden_states 同时供给 q 和 kv 压缩路径）
    3. o_group_proj 的不可分性（Grouped LoRA einsum 需要完整 attention 输出）
    4. 延续 V3 的"弱化 TP，强化 EP+DP"设计哲学
    5. 通信开销与计算密度的权衡（长序列下 AG/RS 数据量远大于收益）
  - 新增 V3 vs V4 并行策略对比表
  - 阐明"接口预留、实现待定"的工程策略

---

## 2026-05-15: DeepSeek-V4 Context Parallelism 实现深度分析（基于 Megatron-LM dev 源码）

**Type**: Knowledge Synthesis（基于 Megatron-LM dev 分支 CP 实现源码，新建 HTML 深度分析）

**入库文件**:

- `wiki/02_engineering/02_train_frameworks/35_deepseek_v4_context_parallel_analysis.html`
  - 9 节深度分析：CP 进程组拓扑（含 Hierarchical CP）、4 种 CP 通信类型（p2p/all_gather/a2a/a2a+p2p）、Native CP 实现（AttentionFuncionWithContextParallel autograd.Function）、TransformerEngine CP 支持（cp_stream Ring Attention）、DSv4 CP 适配（RoPE cp_group、CSA p2p、Dynamic CP 限制）、通信量分析（MLA 使 CP 通信降低 ~128x）、Overlap 机制（TE P2P vs Native AllGather）、Dynamic CP 运行时机制、配置推荐
  - **新增 5.5 节**：CSA/HCA CP 论文设计与代码实现的 Gap 分析——基于 csa.py 源码审计，指出 `cp_comm_type` 参数未实际使用、`_overlap_transform` 跨 rank 依赖未解决（fill_value 填充边界）、压缩 KV 的 AllGather 缺失、P2P 与计算掩盖可行性分析
  - **新增 2.4 节**：四种 CP 方法的 QKV 交互图示——4 幅 SVG 详细展示 p2p（Q 固定，K/V 轮转）、all_gather（聚合完整 K/V）、a2a（All-to-All 交换序列/Head 维度）、a2a+p2p（分层 NVLink A2A + IB P2P）的 token 收发细节与计算数据布局
  - 含 9 幅 SVG 图表（新增 4 幅交互图示）

**源码依据**:

- `/Users/suhaibo/97-llm/Megatron-LM/megatron/core/transformer/dot_product_attention_context_parallel.py` — Native CP autograd.Function，AllGather/ReduceScatter 实现
- `/Users/suhaibo/97-llm/Megatron-LM/megatron/core/extensions/transformer_engine.py` — TE CP 初始化，cp_stream，cp_comm_type 配置
- `/Users/suhaibo/97-llm/Megatron-LM/megatron/core/parallel_state.py` — CP group 创建，Hierarchical CP `create_hierarchical_groups`
- `/Users/suhaibo/97-llm/Megatron-LM/megatron/core/transformer/experimental_attention_variant/deepseek_v4_hybrid_attention.py` — DSv4 CP 集成，RoPE cp_group，Dynamic CP 不支持
- `/Users/suhaibo/97-llm/Megatron-LM/megatron/core/transformer/experimental_attention_variant/csa.py` — CSA 默认 `cp_comm_type="p2p"`
- `/Users/suhaibo/97-llm/Megatron-LM/megatron/core/transformer/transformer_config.py` — CP 通信类型定义与校验
- `/Users/suhaibo/97-llm/Megatron-LM/megatron/core/model_parallel_config.py` — Dynamic CP 配置

**交叉引用更新**:

- `02_train_frameworks/index.md` — 页面列表新增 35_deepseek_v4_context_parallel_analysis.html 条目

---

## 2026-05-14: DeepSeek-V4 Tensor Parallel 切分方案 HTML 深度分析

**Type**: Knowledge Synthesis（基于 Megatron-LM dev 分支实现 + V4 架构特性，新建 HTML 深度分析）

**入库文件**:

- `wiki/02_engineering/02_train_frameworks/34_deepseek_v4_tensor_parallel_analysis.html`
  - 8 节深度分析：V4 架构概览与 TP 必要性、CSA/HCA Attention 层 TP 列行并行策略、MoE ETP 切分（共享专家+路由专家）、mHC 流形约束超连接的切分特殊性、逐层通信量统一公式推导、TP Bulk vs Pipelined Overlap 掩盖方案、TP×EP×PP×CP 四维协同调度、配置决策树
  - 含 5 幅 SVG 图表：CSA TP 数据流、MoE Expert ETP 切分、Bulk Overlap 原理、四维并行通信组拓扑、TP 配置决策树

**交叉引用更新**:

- `02_train_frameworks/index.md` — 页面列表新增 34_deepseek_v4_tensor_parallel_analysis.html 条目

---

## 2026-05-14: 分布式优化器深度分析 HTML 入库

**Type**: Ingestion（HTML 深度分析文档入库，无需新建 .md）

**入库文件**:

- `wiki/02_engineering/02_train_frameworks/32_distributed_optimizer_deepdive.html`
  - 7 节深度分析：ZeRO 分片体系通信量等价性、梯度累积对 ZeRO-1/2 的差异化影响 (K×P)、FSDP2/Megatron/MindSpeed 三方对比、MindSpeed param 临时化与 zero-copy、Adam vs Muon 优化器内存估算 (18→14 bytes/param)、Muon Newton-Schulz 对 ZeRO 切分的根本性挑战、选型决策树
  - 含 6 幅 SVG 图表：DDP vs ZeRO 通信量、梯度累积通信差异、Overlap 机制对比、MindSpeed 内存布局、Element-wise vs 矩阵运算、选型决策树

**交叉引用更新**:

- `megatron-lm/index.md` — Memory & Compute Optimization 节新增 HTML 文件条目
- `megatron-lm/16_megatron_distributed_optimizer_analysis.md` — Related Pages 新增链接
- `02_train_frameworks/index.md` — 页面列表新增条目

---

## 2026-05-13: torch_npu torch.compile 三条路径深度分析

**Type**: Knowledge Synthesis（基于 torch_npu 源码级分析，新建 4 页 + 更新 3 页）

**新建页面**:

- `wiki/02_engineering/01_ai_frameworks/inductor/npu_compile_paths_overview.md`
  - NPU torch.compile 路径总览：三条路径 (Triton/default、ACLGraph、MLIR) 全景对比
  - 与社区 (CUDA/XPU) 的核心差异：monkey-patching 策略、fallback 机制、调度器继承
  - 当前适配的收益：快速迭代、硬件特性直达、多路径冗余保障
  - 演进路线：v2.7.1 (35+ patches) → v2.9.0 (~10) → master (~8)，`_compat.inductor` 兼容层、条件化 patch 管理

- `wiki/02_engineering/01_ai_frameworks/inductor/npu_inductor_splittiling_backend_analysis.md`
  - Triton/Inductor default 路径深度分析（Path 1）
  - Monkey Patch 五类分类：调度器重写、代码生成、wrapper 层、 lowering 规则、Triton 集成
  - `NPUCombinedScheduling` 继承 `CUDACombinedScheduling`，组合 CATLASS + Triton + NoLinearTriton 三种调度器
  - `golden_var_list` / `unified-axis` 逻辑：SIMD/SIMT 混合执行的统一轴选择机制
  - `NPUIndexTritonKernel` 特殊索引 kernel、35+ monkey patches 逐版本演进
  - `lowering_fallback_list.py`：859 aten ops + 135 prims ops 强制 fallback 到 ACLNN
  - 与社区逻辑对比：继承为主、局部重写，演进方向是减少侵入式 patch

- `wiki/02_engineering/01_ai_frameworks/cudagraphs/npugraphs/aclgraph_deep_analysis.md`
  - ACLGraph 深度分析（Path 2）
  - CANN `aclmdlRI*` API 图捕获/重放机制（`AclmdlRICaptureBegin`、`AclmdlRIExecuteAsync`）
  - `NpuGraphOpHandler` 插件框架：FA3 等特殊融合算子在图捕获期的参数预分配
  - Super Kernel (`AclskOptimize`)：CANN 特有的多 kernel 合并优化
  - `StaticKernelCompiler`：预热期预编译 ACLNN kernel，确保捕获确定性
  - 与社区 CUDAGraph 差异：ACLGraph 是 CANN 运行时原语，NPU Graphs 是 PyTorch 层封装
  - 演进方向：统一 NPU Graphs/ACLGraph 接口，向社区 `torch.cuda.CUDAGraph` API 对齐

- `wiki/02_engineering/01_ai_frameworks/inductor/npu_mlir_backend_deep_analysis.md`
  - MLIR 路径深度分析（Path 3）
  - `has_triton = False` 禁用 Triton，启用 MLIR codegen 路径
  - IR 回溯机制：patch `ir.Loops.create` 附加 `traced_graph` 元数据，实现 FX Graph 重建
  - Bisheng 编译器 (`bishengir-opt` + `bishengir-compile`)：华为私有编译器管线
  - Scheduler patch：修改融合规则适应 MLIR 路径需求
  - `auto_fallback` 模式：编译失败自动回退到 FX Graph，双通道容错
  - 与社区逻辑差异：MLIR 路径在社区不存在，是 NPU 特有方案
  - 演进方向：`torch_npu._compat.inductor` 兼容层、条件化 patch、社区 MLIR 接口标准化

**更新页面**:

- `wiki/02_engineering/01_ai_frameworks/inductor/index.md`
  - 新增 2 个深度分析页面条目（npu_inductor_splittiling_backend_analysis、npu_mlir_backend_deep_analysis）

- `wiki/02_engineering/01_ai_frameworks/cudagraphs/npugraphs/index.md`
  - 新增 1 个深度分析页面条目（aclgraph_deep_analysis）

- `wiki/02_engineering/01_ai_frameworks/index.md`
  - 新增 4 个页面条目（npu_compile_paths_overview、npu_inductor_splittiling_backend_analysis、npu_mlir_backend_deep_analysis、aclgraph_deep_analysis）
  - 更新知识空白（新增 3 项：monkey patch 演进追踪、CATLASS/CK 生态、IR 回溯通用性）

- `wiki/changelog.md`（本条目）

**知识来源**:
  - `torch_npu` 源码：`torch_npu/_inductor/`（monkey-patch、lowering、codegen/triton.py、codegen/wrapper.py）
  - `torch_npu` 源码：`torch_npu/utils/_graph_tree.py`（ACLGraph 捕获/重放）
  - `torch_npu` 源码：`torch_npu/dynamo/__init__.py`（backend 注册）
  - `torch_npu` 文档：`docs/torch_npu_compile_path_*.md`（三条路径原始分析报告）
  - PyTorch 社区源码：`torch/_inductor/`（CUDA 参考实现）

---

## 2026-05-12: DL 编译优化趋势与通算融合知识补充

**Type**: Knowledge Synthesis（基于社区分析 + DeepSeek V4 wiki 内容，新建 4 页 + 更新 3 页）

**新建页面**:

- `wiki/02_engineering/01_ai_frameworks/flex_attention_analysis.md`
  - FlexAttention（PyTorch 2.4+）范式：从 Pattern Matching（_sfdp_init）到语义驱动代码生成
  - BlockMask 机制：block 粒度稀疏结构编译时分析，FULL/PARTIAL/EMPTY 分类
  - score_mod：内联注意力权重修改（ALiBi、Soft-cap、Temperature）
  - 与 _sfdp_init 的详细对比，典型 LLM 模型映射表
  - 局限与未来方向（torch.export AOT、NPU 支持）

- `wiki/02_engineering/01_ai_frameworks/tilelang_analysis.md`
  - TileLang 的定位：填补图 Pass（太高层）和 Kernel（太低层）之间的 Gap
  - DeepSeek V4 mHC 融合 kernel：RMSNorm+Linear+Sinkhorn-Knopp 片上融合，读写量降 3×
  - Host Codegen：<1μs kernel launch overhead（vs Python wrapper 的 5-20μs）
  - Z3 SMT 求解器：整数约束的编译时自动验证
  - TileLang vs Triton DSL 对比，tile-level IR 生态（FlexAttention/Linalg Tiling/CuTe）
  - 对图 Pass 体系的影响：tile-level IR 作为图 Pass 与 Kernel 的解耦层

- `wiki/02_engineering/01_ai_frameworks/31_comm_compute_fusion_guide.md`
  - 通算融合四层次模型（手动→半自动→框架感知→全自动）
  - **WaveEP（DeepSeek V4）**：wave-based 细粒度 EP 调度原理、CUDA Stream 架构、wave 粒度权衡
  - 实测性能：一般推理 1.50-1.73×，RL rollout 高达 1.96×
  - **DeepEP**：fine-grained SM control，FusedDispatch（Permute+A2A+Unpermute），HybridEP（NVLink/IB 异构）
  - TP/PP/CP/DP 各维度通算重叠机制（Pipelined AG、DualPipe、Ring Attention 双缓冲）
  - MLIR Mesh Dialect 的通算 IR 作用：async token + chunk-level 依赖
  - WaveEP 编译化路径：wave IR 表示 → Cost Model → TileLang 绑定 → DTensor 集成

- `wiki/02_engineering/01_ai_frameworks/mindspore_compiler_analysis.md`
  - MindSpore 编译流水线：ANF 图 → MindCompiler → AKG → CANN 后端
  - ANF（Administrative Normal Form）IR：函数式表示，高阶函数支持，vs FX Graph 对比
  - MindCompiler Pass：代数化简、常量折叠、算子融合白名单匹配
  - AKG Polyhedral 自动 Kernel 生成：loop tiling/vectorization/fusion，昇腾 NPU 特化
  - **ParallelAuto**：DP 递归规划自动并行策略搜索，vs Alpa 对比
  - MindSpore 2.x 动静统一（@jit 装饰器）
  - 与 PyTorch Inductor 的 Pass 体系逐类对比，优劣评价

**更新页面**:

- `wiki/02_engineering/01_ai_frameworks/mlir_core_concepts.md`（新增"补充"章节）
  - **MLIR Mesh Dialect**：通信作为 IR 一等公民，async token + chunk 依赖分析，对 WaveEP 编译化的意义
  - **IREE**：Flow/Stream/HAL Dialect 三层架构，与 torch-mlir 的组合使用，vs torch.compile 对比
  - **StableHLO**：跨框架稳定 IR 锚点，通信算子标准化，GSPMD 自动并行的 IR 基础
  - **Triton 3.x MLIR 迁移**：Triton Dialect + TritonGPU Dialect，H100 TMA 异步 copy，与 Linalg Pass 的潜在集成

- `wiki/02_engineering/01_ai_frameworks/index.md`
  - 新增 3 个优化页面条目（flex_attention、tilelang、comm_compute_fusion）
  - 新增 1 个架构页面条目（mindspore_compiler）
  - 更新 mlir_core_concepts 描述（含新增内容）
  - 更新知识空白（新增 3 项）

- `wiki/changelog.md`（本条目）

**知识来源**:
  - PyTorch 官方文档（FlexAttention, DTensor, torch.export）
  - `wiki/01_theory/01_models/deepseek/13_deepseek_v4_analysis.md`（WaveEP、TileLang、DeepEP）
  - `wiki/02_engineering/02_train_frameworks/megatron-lm/moe_training_optimization_report.md`（DeepEP/HybridEP）
  - `wiki/02_engineering/02_train_frameworks/megatron-lm/20_megatron_comm_overlap_analysis.md`（多维通算重叠）
  - MLIR 官方文档（Mesh Dialect RFC，IREE，StableHLO）
  - Triton GitHub（triton-lang/triton MLIR 迁移 PR）
  - MindSpore 官方文档（ParallelAuto，AKG，ANF IR）

---

## 2026-05-12: Megatron-LM MoE 训练优化技术全景分析

**Type**: Knowledge Synthesis + Research（源码级分析, 新建 3 个 Wiki 页面）

- **Created**:
  - `wiki/02_engineering/02_train_frameworks/megatron-lm/16_megatron_distributed_optimizer_analysis.md` — 分布式优化器深度分析（ZeRO-1/2 分片机制, Reduce-Scatter/All-Gather 通信, FP8/FP4 量化参数, CPU Offloading 双模式）
  - `wiki/02_engineering/02_train_frameworks/megatron-lm/22_megatron_memory_optimization_analysis.md` — 显存优化全景分析（NCCL Pool, MoE Paged Stash 三级溢出, Fine-Grained Activation Offloading, Buffer 复用, FP8/FP4 精度, Resharding）
  - `wiki/02_engineering/02_train_frameworks/megatron-lm/21_megatron_fusion_operators_analysis.md` — 融合算子优化分析（Bias+Activation 融合 6 种, Fused LayerNorm/Softmax, MoE 专用融合 4 种, Communication Fusion, FP8 Input Store, Triton/CUTLASS/cuTile kernel 层次）
- **Updated**: `megatron-lm/index.md` — 新增 "Memory & Compute Optimization" 章节, 更新 Knowledge Gaps 和 Cross-Domain Links
- **Supplemented**: FSDP2 适配分析 — Megatron 三种梯度/参数分片方案对比（`DistributedOptimizer` vs `TorchFullyShardedDataParallel` vs `MegatronFSDP`），FSDP Unit 机制, ZeRO 分片谱系, NCCL UserBuffer 优化, Delayed Wgrad Overlap, 与 EP/Activation Checkpointing/CUDA Graph 的协同
- **Supplemented**: CP 源码分析 — Ring Attention with AllGather pipeline, zigzag mask conversion, KV buffer double buffering, CP 正反向通信量公式
- **Supplemented**: 通信量/通信组全面分析（正反向） — 6 个并行维度（TP/PP/EP/CP/DP/DistOpt）的通信组层级关系图、通信原语映射、正反向通信量公式推导、通信时序图、Bucket 粒度与 NCCL 带宽、统一通信量总览表、671B MoE 典型通信量排序
- **Removed**: `Megatron-LM_Distributed_Parallel_Exam.md` — 内容分发至各分析页面（SP/TP 边界、TP autograd Function、Dynamic CP、MoE Router/Folding、FSDP+TP/EP 拓扑、Layer-Wise Optimizer、Grouped GEMM、通信组具体示例）
- **Updated**: `megatron-lm/index.md` — 移除已删除 exam 页面引用
- **Design doc**: `docs/superpowers/specs/2026-05-12-moe-training-optimization-report-design.md`

**Key sources analyzed**:
  - `megatron/core/optimizer/distrib_optimizer.py` — 分布式优化器主类 (~2800 lines)
  - `megatron/core/fusions/` — 13 个融合算子文件（@jit_fuser, Triton, CUTLASS/cuTile）
  - `megatron/core/nccl_allocator.py`, `moe/paged_stash.py`, `fine_grained_activation_offload.py` — 显存优化
  - `megatron/core/transformer/dot_product_attention_context_parallel.py` — CP Ring Attention
  - `megatron/core/transformer/moe/fused_a2a.py` — DeepEP/HybridEP 通信融合
  - `megatron/core/distributed/param_and_grad_buffer.py` — 参数/梯度 Buffer 管理

---

## 2026-05-11: torch.compile Dynamic Shape 全链路技术分析

**Type**: Knowledge Synthesis（PyTorch 主分支源码级调研）

- **Created**: `wiki/02_engineering/01_ai_frameworks/inductor/dynamic_shapes_full_analysis.md` — Dynamic Shape 全链路分析（中文）
- **Updated**: `wiki/02_engineering/01_ai_frameworks/inductor/index.md` — 编译阶段表格新增条目

**Key topics**:
  - **Why static-only**: Guard system bakes concrete integers → every shape change triggers recompilation
  - **ShapeEnv architecture**: `_init()` core data structures (`var_to_range`, `replacements`, `divisible`, `deferred_runtime_asserts`), backpropagation of constraints
  - **DimDynamic**: DYNAMIC/DUCK/STATIC/UNBACKED/INFER_STRIDE policies, how `mark_dynamic()` and `assume_static_by_default` control symbol allocation
  - **Guard system**: `_maybe_guard_rel()` → equality replacement + range refinement, three-layer guard architecture (ShapeEnv → GuardBuilder.SHAPE_ENV → runtime asserts)
  - **Correctness guarantees**: `assert_size_stride()` runtime validation, `exclusion_constraints` for automatic_dynamic recompilation
  - **SymInt/SymNode**: Python-level symbolic integer wrapping sympy.Expr, transparent tracking of all shape arithmetic
  - **automatic_dynamic_shapes**: Progressive dynamism — static first, recompile with dynamic on wobble, exclusion guards preserve static cache

- Cross-referenced with `[[24_inductor_codegen_dynamic_shape_analysis]]`, `[[02_torch_compile_architecture]]`, `[[PyTorch_Dynamo_Technical_Analysis]]`

---

## 2026-05-11: PyTorch Inductor 端到端编译管线源码分析

**Type**: Knowledge Synthesis（PyTorch 主分支源码级调研）

- **Created**: `wiki/02_engineering/01_ai_frameworks/inductor/inductor_compiler_pipeline_analysis.md` — PyTorch Inductor 后端编译流程深度分析（中文）
- **Updated**: `wiki/02_engineering/01_ai_frameworks/inductor/index.md` — 架构与流程表格新增条目
- **Cross-referenced**: 新页面与现有 10 个分阶段分析页面建立双向链接（`[[aotautograd_analysis]]`, `[[30_pre_grad_passes_guide]]`, `[[31_joint_graph_passes_guide]]`, `[[32_post_grad_passes_guide]]`, `[[lowering_analysis]]`, `[[scheduler_analysis]]`, `[[20_inductor_codegen_analysis]]`, `[[PyTorch_Dynamo_Technical_Analysis]]`, `[[PyTorch_Inductor_Technical_Analysis]]`, `[[02_torch_compile_architecture]]`）

**Key topics**:
  - **§1 Dynamo**: PEP 523 帧拦截、符号化执行字节码（`InstructionTranslator`）、VariableTracker 体系、Guards 机制（C++ `RootGuardManager`）、Graph Break 处理
  - **§2 AOT Autograd**: 前向/反向追踪、Joint Graph、Functionalization、Min-cut 分区算法、激活值保存 vs 重新计算权衡
  - **§3 Decomposition**: Core ATen + Inductor 分解表、条件化分解（形状/设备/类型）
  - **§4 FX Passes**: Pre-grad（normalization、group_batch_fusion、fuse_fx、efficient_conv_bn_eval）/ Joint-graph（constant_fold_uniform_value、remove_no_ops、pattern matching、replace_random）/ Post-grad（reorder_for_locality、mkldnn_fusion、b2b_gemm、micro_pipeline_tp、collectives bucketing、reinplace）— 逐 pass 源码级分析
  - **§5 Lowering**: `lowerings` 字典、`TensorBox/StorageBox`、IR 原语（Pointwise/Reduction/Scan/TemplateBuffer）、`register_lowering` 装饰器
  - **§6 Scheduler**: 依赖分析（`compute_dependencies`）、融合算法（`fuse_nodes`/`can_fuse`/`can_fuse_vertical`）、Combo Kernel、图分区（CUDAGraph）
  - **§7 CodeGen**: Triton/C++ Kernel 生成、Tiling 策略、Autotuning 子进程（`TuningProcessPool`）、AOTI C++ Wrapper、两层架构（Kernel + Wrapper）
  - **§8 设计哲学**: 分层解耦、函数化→优化→inplace、融合优先、延迟决策

- 重写 `wiki/02_engineering/01_ai_frameworks/torch_mlir_pass_pipeline_analysis.md`
  — **核心修正**: torch-mlir 可以通过 `torch.compile(model, backend=custom_mlir_backend)` 的自定义 backend 方式使用，入口是 `stateless_fx_import(gm)`——它直接接收 Dynamo 捕获的 `torch.fx.GraphModule`，不需要 `torch.export`。
  — 三条路径: A(`torch.compile`→Inductor→Triton，不走MLIR) / B(`torch.compile`+torch-mlir backend，走MLIR，本文) / C(`torch.compile`→NPU MLIR，monkey-patch)
  — 文档主体: Layer 0 `stateless_fx_import(gm)` → Layer 1 `torchdynamo-export-to-torch-backend-pipeline` (4-7 Pass) → Layer 2 `torch-backend-to-linalg-on-tensors-backend-pipeline` (18 Pass) → Layer 3 Linalg→GPU (上游 MLIR 概述)
- 更新 `wiki/02_engineering/01_ai_frameworks/inductor/npu_mlir_pipeline_analysis.md`
  — 新增 "NPU Codegen 内部的 MLIR Pass 分解" 小节，详细列出 Stage 6a→6e 的五个子阶段及每个子阶段内部的 Pass 序列
  — 补充 `torch-lower-to-backend-contract` 在 NPU 场景中的具体 Pass 序列及每个 Pass 的作用

- 重写 `wiki/02_engineering/01_ai_frameworks/torch_mlir_pass_pipeline_analysis.md`
  — **根本性重写**: 不再分析 torch-mlir 独立路径 (fx.py/export_and_import)，而是追踪 `torch.compile` → Inductor → NPU MLIR 的实际代码路径。
  — 六阶段流水线: Stage 0 Dynamo 图捕获 → Stage 1 FX Graph 预处理 (npu_optimize_fx_graph, parallel_scheduler_pass) → Stage 2 AOT Autograd (wrap_compiler 注入) → Stage 3 Decomposition (NPU 选择性禁用) → Stage 4 Inductor Lowering (TracedGraph 三层耦合) → Stage 5 Scheduler 融合 (NPU 修改规则) → Stage 6 NPU MLIR Codegen (5 子阶段: FX 重建→FxImporter→LowerToBackendContract→Bisheng 降级→毕昇编译)
  — 基于 `torch_npu` 源码: `npu_inductor_plugin.py`、`codegen/mlir.py`、`inductor_patch/lowering.py`、`inductor_patch/ir.py`、`utils.py`、`torch_mlir_patch.py`、`mlir_compiler.py`
  — 35+ Pass 总结表，标注每个 Pass 的 IR 层级、实现语言、核心作用、是否为 NPU 特有
  — 核心设计权衡: Python 前端承担编译责任、TracedGraph "夹带私货"代价、双编译器分工、Fallback 双通道

## 2026-05-11: torch.compile → MLIR 完整 Pass 管线分析 (基于源码追踪重写) [已被上述版本取代]

- 重写 `wiki/02_engineering/01_ai_frameworks/torch_mlir_pass_pipeline_analysis.md`
  — **核心修正**: 基于 `torch.compile` → MLIR 的实际 Python → C++ 调用链完整追踪。从 `fx.py:export_and_import()` → `_module_lowering()` → `lower_mlir_module()` 逐函数追踪，确定实际执行的两级 MLIR pipeline:
  — **Stage 1** (`torchdynamo-export-to-torch-backend-pipeline`): `torch-match-quantized-custom-ops` → `Inliner` → `ReduceOpVariants` → `Canonicalizer` → [可选 Decompose→Recompose→Canonicalizer]，共 4-7 Pass
  — **Stage 2** (`torch-backend-to-linalg-on-tensors-backend-pipeline`): `RestructureNonConstantAxes` → `FuseQuantizedOps` → `ConvertTorchToTMTensor` → `Canonicalizer` → `ConvertTorchToLinalg`(9 组 pattern) → `Canonicalizer` → `ConvertTorchToSCF` → `ConvertTorchToArith` → `ConvertTorchToTensor` → `ConvertTorchConversionToMLProgram` → `memref::ExpandOps` → `Canonicalizer` → `memref::ResolveShapedTypeResultDims` → `CSE` → `FuncBackendTypeConversion` → `Canonicalizer` → `FinalizingBackendTypeConversion` → `VerifyLinalgOnTensorsBackendContract`，共 18 Pass
  — 基于源码: `python/torch_mlir/fx.py`、`python/torch_mlir/compiler_utils.py`、`lib/Dialect/Torch/Transforms/Passes.cpp`、`lib/Dialect/TorchConversion/Transforms/Passes.cpp`、`lib/Conversion/TorchToLinalg/TorchToLinalg.cpp`
  — 每次 Canonicalizer (共 5 次) 标注了其消除的特定碎屑类型
  — 文档结构: §1 Dynamo Export 管线 6 个 Pass 三维分析 (Inliner→ReduceOpVariants→Canonicalizer→[Decompose→Recompose→Canonicalizer]) + ConvertTorchToLinalg 概述；§2 TorchScript 管线完整执行顺序；§3 架构转变分析 "前端承担编译责任" (TorchScript 2019 vs Dynamo Export 2023 哲学对比表)；§4 两条管线的共享组件 (ReduceOpVariants / DecomposeComplexOps / Canonicalizer / satisfiesBackendContract)；§5 LowerToBackendContract 迭代引擎深度分析；§6 设计方案总结对比表；§7 与 Triton 对比。
  — 基于 `Passes.cpp` 中 `createTorchDynamoExportToTorchBackendPipeline` 和 `createTorchScriptModuleToTorchBackendPipeline` 的精确源码，阐明两条管线的 18 个 Pass 差异及其根本原因。
- 更新 `wiki/02_engineering/01_ai_frameworks/mlir_core_concepts.md` — Related Pages 新增交叉引用
- 更新 `wiki/02_engineering/01_ai_frameworks/index.md` — 编译架构页面列表新增条目

## 2026-05-11: MLIR Pass 设计哲学补充 + torch-mlir Pass 源码实例

- 更新 `wiki/02_engineering/01_ai_frameworks/mlir_core_concepts.md`
  — 新增 §4.1 四种 Pass 作用域的设计哲学（安全性/可组合性/并行调度/测试调试）、"为什么不像 Triton 做全局优化"分析、与 Eager Mode 概念对应表；新增 §4.2 上游 MLIR ElementwiseOpFusion 源码解析（`areElementwiseOpsFusable`、`fuseElementwiseOps`、融合前后 IR 对比、与 Triton 融合检查项一一对应）；新增 torch-mlir FuseQuantizedOps 实例（Dialect 级 Pass，量化链融合）
- 更新 `wiki/02_engineering/01_ai_frameworks/triton_vs_mlir_backend_analysis.md`
  — 新增社区活跃度章节（`llvm/torch-mlir` main 分支每日活跃，SHARK-Turbine 已迁移）


- 新建 `wiki/02_engineering/01_ai_frameworks/inductor/npu_mlir_pipeline_analysis.md`
  — NPU MLIR 六阶段适配全景 (Dynamo→AOT→Decomp→Lowering→Scheduler→Codegen)，GPU Triton vs NPU MLIR 逐阶段对比。"改了什么、为什么在这一层、怎么改的"。
  核心内容: 三层 Pass 架构 (FX/Inductor/毕昇)、15 个 Monkey Patch 五组分类、编译模式状态机、Fallback 双通道、Autotune 60 配置
- 重写 `wiki/02_engineering/01_ai_frameworks/inductor/npu_compile.md`（原为 10 行存根）
  — 完整 NPU 编译工作流: 三种编译模式 (auto_fallback/default/complete_fallback)、毕昇编译器接口 (-enable-hfusion-compile 等)、60 维 Autotune、在线精度对比 (ANIR_ONLINE_ACC_COMP)、芯片感知 (910B1/310B1/910_9391)
- 更新 `inductor/index.md`、`01_ai_frameworks/index.md`、`npu_mlir_backend_technical_analysis.md`、`npu_lowering_guide.md` 交叉引用

## 2026-05-08: 知识库目录结构重构

**Type**: Infrastructure — 从旧编号体系迁移至 Theory/Engineering 双层结构

### 新结构

```
raw/ & wiki/ 镜像
├── 01_theory/           # 理论研究 (原 llm/ 域 + 模型家族)
│   ├── 01_models/       # 模型架构 + 模型家族 (原 01_architecture + 05_model_families + 07_multimodal)
│   ├── 02_pretraining/  # 预训练技术 (原 02_training)
│   ├── 03_sft/          # SFT + 低参微调 (新建，预留)
│   ├── 04_posttraining/ # 后训练对齐 (原 03_alignment)
│   └── 05_inference/    # 推理技术 (原 04_reasoning + 08_agents)
└── 02_engineering/      # 工程实现 (原 torch_compile/ + 06_infra + 10/11)
    ├── 01_ai_frameworks/    # AI框架 (原 torch_compile/)
    ├── 02_train_frameworks/ # 训练框架 (原 06_infra + 10_train_framework)
    ├── 03_infer_frameworks/ # 推理框架 (原 11_infer_framework)
    └── 04_posttrain_frameworks/ # 后训练框架 (新建，预留)
```

### 变更内容

- 迁移 ~99 raw PDFs + ~102 wiki 页面至新结构
- 80 个文件中的 `[[wiki links]]` 路径批量更新（Python 脚本）
- 新建 5 个 index.md；重写 wiki/index.md 和 7 个领域 index
- 更新 CLAUDE.md、README.md
- 旧编号体系 (01-11) 完全废弃

## 2026-05-09: Triton vs Torch-MLIR 编译后端对比 + MLIR 基础概念

- 新建 `wiki/02_engineering/01_ai_frameworks/triton_vs_mlir_backend_analysis.md`
  — Triton 与 Torch-MLIR 在 Dynamo→AOT Eager→Decomposition→Lowering→Scheduler→Codegen 六个阶段的概念级对等映射表和优劣势分析
- 新建 `wiki/02_engineering/01_ai_frameworks/mlir_core_concepts.md`
  — MLIR 三核心机制: Dialect 词汇表、Pass 变换引擎、IR 注册链路 (TableGen→C++→MLIRContext)，含递降完整示例
- 更新 `wiki/02_engineering/01_ai_frameworks/index.md`、`inductor/index.md` 和 `npu_mlir_backend_technical_analysis.md` 的交叉引用

## 2026-05-08: 训练/推理框架目录页创建

- 新建 `wiki/llm/10_train_framework/index.md`（对应 `raw/10_train_framework/`：megatron.eddx, mindformers.eddx）
- 新建 `wiki/llm/11_infer_framework/index.md`（对应 `raw/11_infer_framework/`，当前为空）
- 更新 `wiki/llm/index.md`、`wiki/index.md`、`CLAUDE.md` 目录结构

## 2026-05-06: GLM/GLM-5 技术路线摄入

**Type**: Source Ingestion (GLM Series)

### 下载的新 Raw 文件

- `raw/05_model_families/zhipu_glm/GLM-5_Vibe_Coding_to_Agentic_Engineering-2602.15763.pdf`

### 创建的 Wiki 页面

- **Created**: `wiki/llm/05_model_families/zhipu_glm/01_glm_5_analysis.md` — GLM-5 Vibe Coding 到 Agentic Engineering（中文）
- **Created**: `wiki/llm/05_model_families/zhipu_glm/10_glm_5v_turbo_analysis.md` — GLM-5V-Turbo 原生多模态 Agent（中文）
- **Created**: `wiki/llm/05_model_families/zhipu_glm/index.md` — GLM 技术路线总览

**Key topics (01_glm_5_analysis)**:
  - 744B/40B MoE (256 专家，8 激活)，80 层
  - Muon Split: per-head 独立正交化，MLA 匹配 GQA-8 性能
  - MLA-256: head dim 192→256，头数减少 1/3，解码计算降低
  - MTP 参数共享 (3 层)，Accept Length 2.76
  - DSA 稀疏注意力：20B tokens 适配，计算减少 1.5-2×，无损
  - 28.5T tokens 预训练，200K 上下文 mid-training
  - 异步 RL 基础设施：TITO gateway + Direct Double-sided Importance Sampling
  - Reasoning RL: GRPO + IcePop，训练-推理不匹配缓解
  - Agentic RL: 10K+ SWE + Terminal + Search 环境
  - 国产 GPU 全栈适配 (7 大平台)
  - SWE-bench ~65, τ²-Bench ~60, HLE ~30

**Key topics (10_glm_5v_turbo_analysis)**:
  - CogViT 视觉编码器：两阶段预训练 (蒸馏 MIM + 对比图文)
  - NaFlex 可变分辨率，64K batch, 80 亿中英图文对
  - MMTP 多模态 MTP：`<|image|>` 共享 token 方案
  - 30+ 任务联合 RL：感知/推理/Agent 全面提升
  - 大规模多模态 RL 基础设施：四维重新设计
  - ImageMining 基准：30.7 分
  - Design2Code 94.8, BrowseComp-VL 51.9, OSWorld 62.3
  - 纯文本编码能力保持 (CC-Backend 22.8, CC-Frontend 68.4)

---

## 2026-05-06: Kimi K2 & K2.5 技术路线摄入

**Type**: Source Ingestion (Kimi K2/K2.5)

### 创建的 Wiki 页面

- **Created**: `wiki/llm/05_model_families/moonshot_kimi/11_kimi_k2_analysis.md` — Kimi K2 开放 Agent 智能（中文）
- **Created**: `wiki/llm/05_model_families/moonshot_kimi/13_kimi_k2_5_analysis.md` — Kimi K2.5 视觉 Agent 智能（中文）
- **Updated**: `wiki/llm/05_model_families/moonshot_kimi/index.md` — 论文索引更新，K2/K2.5 标记为已摄入

**Key topics (11_kimi_k2_analysis)**:
  - 1.04T/32.6B MoE，384 专家 (sparsity=48)，64 注意力头
  - MuonClip 优化器：QK-Clip 解决 logits 爆炸，15.5T token 零 loss spike
  - 稀疏度扩展定律：sparsity 48 vs 8 节省 1.69× FLOPs
  - 大规模 Agentic 数据合成：23,000+ 工具，模拟+真实沙盒
  - RL 框架：RLVR + 自批评 Rubric 奖励，覆盖可验证和主观任务
  - SWE-bench 65.8、τ²-Bench 66.1、AIME 2024 69.6
  - Agent 能力超越 Claude Opus 4 和 GPT-4.1

**Key topics (13_kimi_k2_5_analysis)**:
  - MoonViT-3D 视觉编码器：原生分辨率，3D 时空编码，4× 时间压缩
  - 早期融合 + 低视觉比例 (10%:90%) 优于晚期融合
  - Zero-Vision SFT：仅用文本 SFT 激活视觉能力
  - 联合多模态 RL：视觉 RL 提升文本性能 (MMLU-Pro +1.7%)
  - Agent Swarm：可训练编排器 + 冻结子智能体，BrowseComp 60.6%→78.4%
  - Toggle 算法：token 减少 25-30%，性能影响可忽略
  - DEP 训练基础设施：多模态训练效率达纯文本 90%
  - LVBench 75.9%、OCRBench 92.3%、BrowseComp 78.4%

---

## 2026-05-06: Kimi/Moonshot AI 技术路线批量摄入 (4 篇核心论文)

**Type**: Source Ingestion (Kimi 技术路线)

### 下载的新 Raw 文件

- `raw/05_model_families/moonshot_kimi/Kimi_k1.5_Scaling_RL-2501.12599.pdf`
- `raw/05_model_families/moonshot_kimi/Mooncake_KVCache_Disaggregated-2407.00079.pdf`
- `raw/05_model_families/moonshot_kimi/MoBA_Mixture_of_Block_Attention-2502.13189.pdf`
- `raw/05_model_families/moonshot_kimi/Kimi_Linear_Attention-2510.26692.pdf`

### 创建的 Wiki 页面

- **Created**: `wiki/llm/06_infra/mooncake_analysis.md` — Mooncake KVCache 中心化分离式服务架构（中文）
- **Created**: `wiki/llm/01_architecture/10_moba_analysis.md` — MoBA 混合块注意力机制（中文）
- **Created**: `wiki/llm/01_architecture/12_kimi_linear_analysis.md` — Kimi Linear/KDA 线性注意力架构（中文）
- **Created**: `wiki/llm/03_alignment/kimi_k1.5_analysis.md` — Kimi k1.5 RL 缩放定律（中文）
- **Created**: `wiki/llm/05_model_families/moonshot_kimi/index.md` — Kimi 技术路线总览

**Key topics (mooncake_analysis)**:
  - Prefill/Decode/KVCache 三池分离架构
  - Chunked Pipeline Parallelism (CPP) 替代跨节点 SP
  - Layer-wise Prefill：KVCache 传输与计算重叠
  - 缓存感知全局调度 + 热点块迁移
  - 预测性早期拒绝解决负载波动
  - 真实负载吞吐量提升 75%，模拟场景 525%

**Key topics (10_moba_analysis)**:
  - 将 MoE 原理应用于注意力机制
  - Query 动态路由到 KV Block (top-k 选择)
  - 块路由：mean_pool(K) 亲和度 + 因果掩码
  - MoBA/Full 混合预训练 (90%/10%)
  - 1M 序列 6.5x 加速，10M 序列 16x 加速
  - 已部署支持 Kimi 长上下文请求

**Key topics (12_kimi_linear_analysis)**:
  - KDA: Kimi Delta Attention (通道级细粒度遗忘门)
  - 约束 DPLR 结构，消除数值不稳定，Kernel 速度 ~2x
  - 3:1 KDA-MLA 混合架构，MLA 层使用 NoPE
  - KV Cache 减少 75%，1M 解码 6x 加速
  - 在预训练/SFT/长上下文/RL 场景下均超越全注意力
  - 开源 KDA Kernel + vLLM 集成 + Checkpoints

**Key topics (kimi_k1.5_analysis)**:
  - 在线镜像下降变体 (类似 GRPO，理论来源不同)
  - 128K 上下文 RL 训练，上下文长度是关键扩展维度
  - Partial Rollout + 混合部署 (Megatron ↔ vLLM via Mooncake)
  - Long2Short 蒸馏 (模型合并/拒绝采样/DPO/RL)
  - 长度惩罚渐进式引入，防止过度思考
  - AIME 77.5、MATH-500 96.2、Codeforces 94th percentile

---

## 2026-05-06: 低精度训练与 Transformer Engine 知识整合

**Type**: Knowledge Synthesis（Megatron-LM 源码 + TE GitHub 仓库 + DeepSeek-V4 FP4 QAT）

- **Created**: `wiki/llm/02_training/13_low_precision_training_analysis.md` — Megatron 低精度训练全栈分析（中文）
- **Created**: `wiki/llm/02_training/14_transformer_engine_analysis.md` — NVIDIA Transformer Engine 技术分析（中文）
- **Updated**: `wiki/llm/index.md` — Optimizers & Training Algorithms 表格新增 3 条目
- **Updated**: `wiki/llm/06_infra/megatron-lm/index.md` — Knowledge Gaps 更新（TE 集成、低精度训练标记为已解决），Cross-Domain Links 扩展

**Key topics (13_low_precision_training_analysis)**:
  - 精度格式全览（FP32 → BF16 → FP16 → FP8 → MXFP8 → FP4）
  - 五种 FP8 Recipe（tensorwise/delayed/blockwise/mxfp8/custom）及对比
  - FP8 Primary Weights（fp8_param_gather）显存节省分析（6N → 5N bytes）
  - first_last_layers_bf16 首末层 BF16 保护机制
  - TP 通信与 FP8 协同（User Buffer, Pipelined/Bulk Overlap）
  - FP4 QAT（DeepSeek-V4 方案）：无损反量化原理、STE 训练、推理部署
  - MoE + 低精度（Grouped GEMM FP8, Router Fusion, DeepEP A2A）
  - Scaling MoE 论文精度实践总结
  - 配置速查表

**Key topics (14_transformer_engine_analysis)**:
  - TE 两层架构（Python API + C++/CUDA Kernel）
  - 精度格式矩阵：FP8(E4M3/E5M2/HYBRID) / MXFP8 / NVFP4 / BF16/FP16
  - Recipe 系统（DelayedScaling → Float8CurrentScaling → MXFP8BlockScaling → NVFP4BlockScaling2D）
  - Quantizer 体系（Float8CurrentScalingQuantizer / Float8Quantizer / MXFP8Quantizer）
  - Scale 计算核心公式 + 边界情况处理
  - FP8GlobalStateManager：全局 buffer 批量 amax reduce + 激活重计算支持
  - C++ Kernel 层（quantize/dequantize/gemm/grouped_gemm/融合算子）
  - CommOverlap 体系（CommOverlapHelper/CommOverlap/CommOverlapP2P + NVSHMEM）
  - Megatron 集成桥接（TELinear/TELayerNormColumnParallelLinear/TENorm + FP8 recipe 映射）
  - CUDA Graphs + FP8 协同
  - 环境变量与调试指南

---

## 2026-05-07: 知识库索引体系重构 — overview.md → index.md

**Type**: Infrastructure

- **Renamed** all `overview.md` → `index.md`: `llm/`, `llm/06_infra/megatron-lm/`, `torch_compile/`
- **Renamed** `*_overview.md` → `index.md`: `moonshot_kimi/kimi_overview.md`, `zhipu_glm/glm_overview.md`
- **Created** 13 new `index.md` files for directories lacking one:
  - `wiki/index.md` — 知识库总索引（全新）
  - `llm/01_architecture/index.md`, `llm/02_training/index.md`, `llm/03_alignment/index.md`
  - `llm/04_reasoning_and_retrieval/index.md` (stub), `llm/05_model_families/index.md`, `llm/05_model_families/deepseek/index.md`
  - `llm/06_infra/index.md`, `llm/07_multimodal/index.md` (stub), `llm/08_agents/index.md` (stub)
  - `torch_compile/cudagraphs/index.md`, `torch_compile/cudagraphs/npugraphs/index.md`, `torch_compile/inductor/index.md`
- **Updated** all cross-references (~50 files): `overview` → `index`, `kimi_overview` → `index`, `glm_overview` → `index`
- **Updated** `CLAUDE.md` — Page Types, Naming Conventions, Directory Layout, all workflows

---

## 2026-05-06: Wiki 目录重组 — torch_compile 独立为顶级域

**Type**: Infrastructure

- **Moved** `wiki/llm/02_training/torch_compile/` → `wiki/torch_compile/`
- **Rationale**: 与 `raw/09_pytorch/00_compile/` 对齐，torch_compile 作为独立领域不再嵌套在 LLM training 下
- **Updated** all cross-references (~35 files): `llm/02_training/torch_compile/` → `torch_compile/`
- **Updated** `CLAUDE.md` — Directory Layout 反映新结构

---

## 2026-05-06: Raw 目录结构更新 — 新增 09_pytorch

**Type**: Infrastructure

- **Added** `raw/09_pytorch/00_compile/` — 5 PyTorch compile 内部源码分析图（.eddx 格式）：
  - `torch.compile.eddx` — torch.compile 整体架构
  - `dynamo.eddx` — Dynamo 图捕获
  - `AOTautograd.eddx` — AOT Autograd 前向/反向分离
  - `inductor-lowering.eddx` — Inductor IR Lowering 流程
  - `aoteager精度比对.eddx` — AOT Eager 精度对比
- **Updated** `CLAUDE.md` — Directory Layout 同步更新（raw/ 新增 09_pytorch, wiki/ 反映实际重组后的结构）

---

## 2026-04-29: LLM 并行计算依赖分析（HTML）

**Type**: Knowledge Synthesis（Megatron-LM 源码验证）

- **Created**: `wiki/llm/06_infra/llm_parallelism_analysis.html` — LLM 正反向计算依赖 + 并行策略通信分析（中文）
- **Updated**: `wiki/llm/06_infra/megatron-lm/index.md` — Distributed Parallelism 表格新增条目 + Knowledge Gaps 更新
- **Key topics**:
  - 单层 Transformer Decoder 前向/反向算子 DAG（SVG 依赖图 + 关系表）
  - Megatron-LM 源码级验证: `ColumnParallelLinear` / `RowParallelLinear` / `LinearWithGradAccumulationAndAsyncCommunication`
  - TP (Tensor Parallelism) f/g 算子通信模式、SP (Sequence Parallelism) AG+RS 数据流
  - EP (Expert Parallelism) AllToAll dispatch/combine + 内部 TP 通信
  - CP (Context Parallelism) Ring Attention vs Ulysses 对比
  - 组合并行 (TP+SP+CP+EP+PP) 完整前向执行顺序表
  - 计算通信重叠: async grad AllReduce, Ring Attention P2P overlap, DDP bucket overlap
  - CSS `white-space: pre` 修复, 12 代码块 Python 格式化 + 语法高亮

---

## 2026-04-29: DeepSeek-V4 Raw → Wiki 知识整合

**Type**: Knowledge Integration（Raw MD 文件与 Wiki 合并/去重）

将 `raw/05_model_families/deepseek/` 下 9 个 V4 相关 MD 文件与 Wiki 现有内容整合：

- **Created**: `wiki/llm/05_model_families/deepseek/24_deepseek_v4_fp4_qat_analysis.md` — FP4 QAT 完整分析（全新主题）
- **Moved (3 files)**:
  - `28_deepseek_v4_architecture_analysis.md` — V4 架构 ASCII 结构图（50KB 补充参考）
  - `27_deepseek_v4_implementation_deepdive.md` — V4 核心组件伪代码实现（34KB 补充参考）
  - `26_deepseek_v4_technical_deepdive.md` — CSA/HCA/DSA/MLA 对比深度解析（42KB 补充参考）
- **Updated (merged unique content)**:
  - `13_deepseek_v4_analysis.md` — 新增 §Compressed KV 数值示例、DualPath 推理框架、Think Modes、Pro-Max 评测
  - `mHC.md` — 扩展 §动态与静态系数（完整公式 3-8、对比表、训练细节）
  - `23_deepseek_v4_cp_analysis.md` — 新增 §9 实现细节（Fused Select-and-Pad、Top-K Selector、传统 CP 对比表）
- **Cross-references**: 所有新/更新页面双向链接已更新

---

## 2026-04-29: Activation Checkpointing（重计算）完整分析

**Type**: Knowledge Synthesis（PyTorch autograd 机制 + Megatron-LM 源码分析）

- **Created**: `wiki/llm/02_training/12_activation_checkpointing_analysis.md` — 激活重计算完整分析（中文）
- **Updated**: `wiki/llm/index.md` — Optimizers & Training Algorithms 表格新增条目
- **Updated**: `wiki/llm/06_infra/megatron-lm/Megatron-LM_Distributed_Parallel_Exam.md` — Q12 考点添加交叉引用
- **Key topics**:
  - autograd `ctx.save_for_backward` 机制与 `torch.no_grad` 干预原理
  - ctx 中 tensor 激活值 vs 元信息的二分法（重计算只消除前者）
  - View/Cast/Slice 算子的反向机制：仅依赖元信息，ctx 不存储 tensor
  - View chain 问题与 Megatron `make_viewless_tensor` 的切断方案
  - Megatron 三层 checkpoint 架构：CheckpointFunction → CheckpointWithoutOutput/te_checkpoint → TransformerBlock 调度
  - `distribute_saved_activations` 的 TP 切分/聚合机制
  - `CheckpointWithoutOutput` 的 zero-copy storage sharing 和 `CheckpointManager`
  - Uniform vs Block 调度策略、逐层 checkpoint 的必要性（vs 整 model 一层）
  - Selective recomputation 的子模块级选择依据与 Decoder 层激活值依赖全景
  - 理论激活值开销公式与估算范例

---

## 2026-04-28: DeepSeek-V4 CP 深度分析

---

## 2026-04-28: DeepSeek-V4 CP 深度分析

**Type**: Source Ingestion (扩展已有 V4 分析)

- **Source**: `raw/05_model_families/deepseek/DeepSeek_V4.pdf` §3.5.3, §3.6, §4.1
- **Created**: `wiki/llm/05_model_families/deepseek/23_deepseek_v4_cp_analysis.md` — DeepSeek-V4 Context Parallelism 深度分析（中文）
- **Updated**: `wiki/llm/05_model_families/deepseek/13_deepseek_v4_analysis.md` — CP 节扩展并添加指向新页面链接
- **Key topics**:
  - Packed sequences 数据格式与 CP 的三个矛盾（跨 rank 文档切断、压缩窗口跨边界、压缩输出长度不可预测）
  - 两阶段通信协议形式化描述（Stage 1 P2P O(c) 常数通信 + Stage 2 All-Gather 压缩 KV）
  - 通信量开销公式推导与数值估算（CSA ~51× 减少, HCA ~2048× 减少 vs 标准 CP）
  - 三层 sample 可见性控制（sample-level attention mask → block-level causal → precomputed rules / Top-K selector）
  - 训练 vs 推理尾部 token 处理策略对比（丢弃 vs State Cache vs 重计算）
  - CSA 重叠窗口对 CP 边界的额外影响
  - 完整 packed sequences × CP × 压缩的数值示例

---

## 2026-04-24: Wiki Directory Restructure

**Type**: Infrastructure

Restructured `wiki/llm/` to mirror `raw/` classification (01-08), consolidating related content:

- **Created** subdirectories under `wiki/llm/`:
  - `01_architecture/` — Transformer, scaling laws, memory architectures
  - `02_training/` — Optimizers, initialization, training precision
  - `03_alignment/` — RLHF, DPO, GRPO, PPO, and related methods
  - `04_reasoning_and_retrieval/` — Reserved for CoT, verification, RAG
  - `05_model_families/deepseek/` — All DeepSeek model analyses
  - `06_infra/megatron-lm/` — Distributed training, MoE infrastructure
  - `07_multimodal/` — Reserved for vision-language, audio-language
  - `08_agents/` — Reserved for agentic AI, tool use
- **Moved** `wiki/torch_compile/` → `wiki/torch_compile/`
- **Moved** `wiki/megatron-lm/` → `wiki/llm/06_infra/megatron-lm/`
- **Moved** `mHC.md` → `wiki/llm/05_model_families/deepseek/mHC.md`
- **Updated** all path-based wiki links across the entire wiki

---

## 2026-04-16: Wiki Schema & Structure Initialization

**Type**: Infrastructure

Created the wiki schema and structural pages:

- Created `CLAUDE.md` — wiki maintenance schema and rules
- Created `wiki/llm/index.md` — LLM domain knowledge map
- Created `wiki/megatron-lm/overview.md` — Megatron-LM domain knowledge map
- Created `wiki/torch_compile/index.md` — torch compile domain knowledge map
- Created `wiki/changelog.md` — this file

---

## Pre-Changelog Entries (Historical Reconstruction)

The following pages were created before the changelog was established. Dates are approximate.

### ~2026-03: MoE & Distributed Training

- Created `wiki/megatron-lm/Megatron-LM_MoE_Zero_Redundancy_Analysis.md` — Source: `raw/Scalable Training of Moe Models with Megatron core-2603.07685v2.pdf`
- Created `wiki/megatron-lm/Megatron-LM_Distributed_Parallel_Exam.md` — Comprehensive exam covering 5D parallelism

### ~2026-02: Muon Optimizer

- Created `wiki/llm/11_muon_analysis.md` — Source: `raw/MUON IS SCALABLE FOR LLM TRAINING-2502.16982v1.pdf`
- Created `wiki/megatron-lm/Megatron_LM_TFLOPS_Analysis.md` — TFLOPS estimation methodology

### ~2026-01: DeepSeek & Memory Architectures

- Created `wiki/llm/29_engram_analysis.md` — Source: `raw/Engram_paper.pdf`
- Created `wiki/llm/18_deepseek_math_v2_analysis.md` — Self-verifiable math reasoning

### ~2025-12: Weight Initialization & KIMI

- Created `wiki/llm/10_llm_initiliaze_analysis.md` — Dense & MoE initialization

---

## 2026-04-17: mHC Source Ingestion

**Type**: Source Ingestion

- **Source**: `raw/mHC-2512.24880v2.pdf` (DeepSeek-AI, arXiv:2512.24880v2)
- **Created**: `wiki/llm/mHC.md` — Manifold-Constrained Hyper-Connections analysis (in Chinese)
- **Updated**: `wiki/llm/index.md` — Added mHC entry and cross-domain links
- **Cross-referenced**: Added backlinks to `11_muon_analysis.md`, `10_llm_initiliaze_analysis.md`, `Megatron-LM_MoE_Zero_Redundancy_Analysis.md`
- **Key topics**: doubly stochastic matrix, Sinkhorn-Knopp projection, residual stream expansion, DeepSeek-V3 MoE, kernel fusion, selective recomputing

### ~2025-11: Training-Inference Integration

- Created `wiki/megatron-lm/Megatron_vLLM_Weight_Sync_Analysis.md` — verl Megatron + vLLM weight sync

### ~2025-10: Torch Compile & NPU

- Created `wiki/torch_compile/inductor/` — 17 pages covering Dynamo, AOT Autograd, Inductor, NPU backends
- Created `wiki/torch_compile/cudagraphs/` — CUDA Graphs guides and NPU Graphs deep dives

## 2026-04-20: DeepSeek Model Family Batch Ingestion (Part 1/4)

**Type**: Source Ingestion

- **Source**: `raw/05_model_families/deepseek/DeepSeek_LLM-2401.02954.pdf` (DeepSeek-AI, arXiv:2401.02954)
- **Created**: `wiki/llm/10_deepseek_llm_analysis.md` — DeepSeek LLM analysis
- **Updated**: `wiki/llm/index.md` — Added DeepSeek model family section
- **Key topics**: scaling laws with non-embedding FLOPs/token representation, data quality impact on model/data allocation, multi-step LR scheduler, GQA, bilingual pre-training, SFT+DPO alignment

- **Source**: `raw/05_model_families/deepseek/DeepSeek_V2-2405.04434.pdf` (DeepSeek-AI, arXiv:2405.04434)
- **Created**: `wiki/llm/11_deepseek_v2_analysis.md` — DeepSeek-V2 analysis
- **Key topics**: MLA (Multi-head Latent Attention), low-rank KV joint compression, decoupled RoPE, DeepSeekMoE, device-limited routing, three-level auxiliary losses, token dropping, GRPO, two-stage RL

- **Source**: `raw/05_model_families/deepseek/DeepSeek_V3-2412.19437.pdf` (DeepSeek-AI, arXiv:2412.19437)
- **Created**: `wiki/llm/12_deepseek_v3_analysis.md` — DeepSeek-V3 analysis
- **Key topics**: FP8 mixed precision training, fine-grained quantization (tile/block-wise), DualPipe pipeline parallelism, auxiliary-loss-free load balancing, Multi-Token Prediction (MTP), cross-node all-to-all communication kernels, inference deployment with redundant experts, R1 distillation

- **Source**: `raw/05_model_families/deepseek/DeepSeek_R1-2501.12948.pdf` (DeepSeek-AI, arXiv:2501.12948)
- **Created**: `wiki/llm/14_deepseek_r1_analysis.md` — DeepSeek-R1 analysis
- **Key topics**: pure RL reasoning without SFT, GRPO, emergent self-verification/reflection, "aha moment", multi-stage pipeline (cold start → RL → SFT → RL), distillation to Qwen/Llama, rule-based rewards, language consistency reward

**Remaining**: Coder, Coder-V2, Math, MoE, Prover, VL

---

## 2026-04-24: DeepSeek-V4 Source Ingestion

**Type**: Source Ingestion

- **Source**: `raw/05_model_families/deepseek/DeepSeek_V4.pdf` (DeepSeek-AI, 2025)
- **Created**: `wiki/llm/13_deepseek_v4_analysis.md` — DeepSeek-V4 analysis (in Chinese)
- **Updated**: `wiki/llm/index.md` — Added V4 to DeepSeek model family section
- **Updated**: `wiki/llm/12_deepseek_v3_analysis.md` — Added backlink to V4
- **Updated**: `wiki/llm/11_deepseek_v2_analysis.md` — Added backlink to V4
- **Cross-referenced**: `mHC.md`, `11_muon_analysis.md`, `12_deepseek_v3_analysis.md`, `11_deepseek_v2_analysis.md`
- **Key topics**: CSA (Compressed Sparse Attention), HCA (Heavily Compressed Attention), hybrid attention architecture, DSA (DeepSeek Sparse Attention), Lightning Indexer, million-token context, mHC integration, Muon optimizer, Anticipatory Routing, SwiGLU clamping, wave-based EP overlap, TileLang kernels, FP4 QAT, heterogeneous KV cache management, on-disk KV cache storage

---

## 2026-04-21: DeepSeek Model Family Batch Ingestion (Part 2/4)

**Type**: Source Ingestion

- **Source**: `raw/05_model_families/deepseek/DeepSeek_Coder-2401.14196.pdf` (DeepSeek-AI, arXiv:2401.14196)
- **Created**: `wiki/llm/15_deepseek_coder_analysis.md` — DeepSeek-Coder analysis
- **Key topics**: repository-level code corpus, dependency parsing, topological sort, Fill-in-the-Middle (FIM), 87 programming languages, 16K context, GQA

- **Source**: `raw/05_model_families/deepseek/DeepSeek_Coder_V2-2406.11931.pdf` (DeepSeek-AI, arXiv:2406.11931)
- **Created**: `wiki/llm/16_deepseek_coder_v2_analysis.md` — DeepSeek-Coder-V2 analysis
- **Key topics**: MoE code model, 338 languages, 128K context, 6T additional tokens, YaRN extension, GRPO with reward model, SWE-bench >10%

- **Source**: `raw/05_model_families/deepseek/DeepSeek_Math-2402.03300.pdf` (DeepSeek-AI, arXiv:2402.03300)
- **Created**: `wiki/llm/17_deepseek_math_analysis.md` — DeepSeekMath analysis
- **Key topics**: 120B math tokens from Common Crawl, iterative fastText pipeline, GRPO origin, unified RL paradigm, MATH 51.7%

- **Source**: `raw/05_model_families/deepseek/DeepSeek_MoE-2401.06066.pdf` (DeepSeek-AI, arXiv:2401.06066)
- **Created**: `wiki/llm/20_deepseek_moe_analysis.md` — DeepSeekMoE architecture analysis
- **Key topics**: fine-grained expert segmentation, shared expert isolation, expert-level/device-level balance loss, 2B/16B/145B scales

- **Source**: `raw/05_model_families/deepseek/DeepSeek_Prover-2408.08152.pdf` (DeepSeek-AI, arXiv:2408.08152)
- **Created**: `wiki/llm/22_deepseek_prover_analysis.md` — DeepSeek-Prover-V1.5 analysis
- **Key topics**: Lean 4 theorem proving, truncate-and-resume mechanism, RMaxTS Monte-Carlo tree search, thought-augmented proofs, RLPAF

- **Source**: `raw/05_model_families/deepseek/DeepSeek_VL-2403.05525.pdf` (DeepSeek-AI, arXiv:2403.05525)
- **Created**: `wiki/llm/21_deepseek_vl_analysis.md` — DeepSeek-VL analysis
- **Key topics**: hybrid vision encoder (SigLIP + SAM), 576 visual tokens, modality warm-up, 70% text preservation, real-world VL taxonomy

- **Note**: `raw/05_model_families/deepseek/DeepSeek_VL2-2412.10322.pdf` was identified as an unrelated physics paper (arXiv:2412.10322v1, hep-lat). No genuine DeepSeek-VL2 source was found.

**Remaining**: None (DeepSeek model family complete)

---

## 2026-04-21: Architecture Foundations & Alignment Methods Batch Ingestion

**Type**: Source Ingestion

### Architecture Foundations (01_architecture/)

- **Source**: `raw/01_architecture/Attention_Is_All_You_Need-1706.03762.pdf` (Vaswani et al., Google, NIPS 2017)
- **Created**: `wiki/llm/attention_is_all_you_need_analysis.md` — Transformer architecture analysis
- **Key topics**: scaled dot-product attention, multi-head attention, positional encoding, encoder-decoder structure, self-attention vs RNN/CNN complexity, O(1) path length

- **Source**: `raw/01_architecture/Scaling_Laws_for_Neural_Language_Models-2001.08361.pdf` (Kaplan et al., OpenAI, 2020)
- **Created**: `wiki/llm/scaling_laws_analysis.md` — Neural scaling laws analysis
- **Key topics**: power-law scaling (L ~ N^-0.076, D^-0.095, C^-0.050), compute-optimal training (N~C^0.73), sub-linear data scaling (D~N^0.74), early stopping, critical batch size, architecture independence

- **Source**: `raw/01_architecture/Long_Context_Scaling_Law-2503.04725.pdf` (Chen et al., MIT, NeurIPS 2025)
- **Created**: `wiki/llm/long_context_scaling_law_analysis.md` — Long-context mutual information scaling
- **Key topics**: bipartite mutual information (I_BP ~ L^beta), L2M condition, history state requirements, Transformer vs SSM long-context capability

- **Skipped**: `raw/01_architecture/Scaling_Laws_for_Transfer-2002.05102.pdf` — PDF contains unrelated mathematics paper (Hurwitz actions on reflection groups)

### Alignment & Preference Optimization (03_alignment/)

- **Source**: `raw/03_alignment/PPO_Proximal_Policy_Optimization-1707.06347.pdf` (Schulman et al., OpenAI, 2017)
- **Created**: `wiki/llm/ppo_analysis.md` — PPO algorithm analysis
- **Key topics**: PPO-Clip objective, surrogate loss, multiple epochs on same data, GAE advantage estimation, KL constraint

- **Source**: `raw/03_alignment/InstructGPT_RLHF-2203.02155.pdf` (Ouyang et al., OpenAI, 2022)
- **Created**: `wiki/llm/instructgpt_rlhf_analysis.md` — RLHF pipeline analysis
- **Key topics**: three-step RLHF (SFT→RM→PPO), KL penalty against SFT, 1.3B > 175B GPT-3, helpful/honest/harmless criteria

- **Source**: `raw/03_alignment/DPO_Direct_Preference_Optimization-2305.18290.pdf` (Rafailov et al., Stanford, 2023)
- **Created**: `wiki/llm/dpo_analysis.md` — DPO algorithm analysis
- **Key topics**: closed-form policy-reward relationship, binary cross-entropy replaces RLHF, no sampling during training

- **Created**: `wiki/llm/preference_optimization_analysis.md` — DPO family comparison
- **Covers**: IPO (squared loss), SimPO (no ref model, length-normalized), ORPO (monolithic), KTO (binary labels, prospect theory), MODPO (multi-objective)

- **Source**: `raw/03_alignment/DeepSeek_R1_Reasoning_via_RL-2501.12948.pdf` (DeepSeek-AI, 2025)
- **Created**: `wiki/llm/grpo_analysis.md` — GRPO algorithm analysis
- **Key topics**: group-relative advantages, no value function, pure RL for reasoning, DeepSeek-R1-Zero emergent behaviors

**Updated**: `wiki/llm/index.md` — Added Architecture Foundations, Scaling Laws, and Alignment sections

---

## 2026-04-21: Alignment Methods Batch Ingestion (Part 2)

**Type**: Source Ingestion

### Advanced RL Algorithms

- **Source**: `raw/03_alignment/DAPO_Decoupled_Clip_Dynamic_Sampling-2503.14476.pdf` (ByteDance Seed, Tsinghua AIR, 2025)
- **Created**: `wiki/llm/dapo_analysis.md` — DAPO algorithm analysis
- **Key topics**: decoupled clipping (eps_low=0.2, eps_high=0.28), dynamic sampling (filter accuracy 0/1), token-level policy gradient loss, soft overlong punishment, AIME 50 with Qwen2.5-32B, open-source RL system

- **Source**: `raw/03_alignment/GSPO_Group_Sequence_Policy_Optimization-2507.18071.pdf` (Qwen Team, Alibaba, 2025)
- **Created**: `wiki/llm/gspo_analysis.md` — GSPO algorithm analysis
- **Key topics**: sequence-level importance ratio, fixes GRPO's token-level instability, length-normalized sequence likelihood, stabilizes MoE RL training, Qwen3 improvements

- **Source**: `raw/03_alignment/RLOO_REINFORCE_Leave_One_Out-2402.14740.pdf` (Cohere For AI, 2024)
- **Created**: `wiki/llm/rloo_analysis.md` — RLOO algorithm analysis
- **Key topics**: REINFORCE with leave-one-out baseline, no value function needed, theoretical foundation for GRPO, 2.5x faster than PPO

- **Source**: `raw/03_alignment/VAPO_Value_Augmented_Proximal_Policy_Optimization-2504.05118.pdf` (ByteDance Seed, 2025)
- **Created**: `wiki/llm/vapo_analysis.md` — VAPO framework analysis
- **Key topics**: value-model-based RL, AIME 60.4 (SOTA), addresses value bias/length heterogeneity/reward sparsity, 5000 steps to SOTA, zero crashes

### RLHF Foundations & Advanced Methods

- **Created**: `wiki/llm/rlhf_foundations_analysis.md` — Comprehensive coverage of:
  - **ReMax** (arXiv:2310.10505): Simplified RLHF using REINFORCE, exploits fast simulation/deterministic transitions/trajectory rewards
  - **Weak-to-Strong Generalization** (OpenAI, arXiv:2312.09390): Can weak model supervision elicit strong model capabilities? Analogy to superhuman alignment
  - **Scaling Laws for RM Overoptimization** (OpenAI, arXiv:2210.10760): Goodhart's Law in RLHF, predictable scaling of overoptimization, best-of-n vs RL
  - **Learning to Summarize** (OpenAI, arXiv:2009.01325): First RLHF for summarization, precursor to InstructGPT
  - **Fine-Tuning from Human Preferences** (OpenAI, arXiv:1909.08593): Earliest RLHF work, stylistic control and summarization
  - **RigorLLM** (arXiv:2403.13031): Resilient guardrails against adversarial attacks, energy-based data generation, minimax optimization

**Updated**: `wiki/llm/index.md` — Added DAPO, GSPO, RLOO, VAPO, and RLHF Foundations entries

**Digestion progress**: 3/4 architecture papers, **20/20 alignment papers digested** (complete)

## Related Pages

- [[01_theory/index]]
- [[02_engineering/02_train_frameworks/megatron-lm/index]]
- [[02_engineering/01_ai_frameworks/index]]
