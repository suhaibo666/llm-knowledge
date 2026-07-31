# 知识库整改 P6-P7:横向页收缩 + 全库收尾 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** P6 消除最后 2 组高重叠(横向页复述、Roofline/CP 多写)+ npu 划界 + 中重叠扫尾;P7 全库编号/命名推广、裸 index 清零、changelog 归档、CLAUDE.md 完整修订、README 定稿。完成后整改全部落地。

**Spec:** `2026-07-29-llm-knowledge-reorg-design.md` §3.5/§3.6(P6)、§4/§5/§6(P7)。

**编辑总规则:** P3-P5 全部沉淀条款继续适用(判重找实际段落/逐字+溯源/清单型防误杀/无源转 todo/推翻裁定须披露/checker=0 每 commit)。

---

## P6(分支 reorg/p6)

### Task 1: 分支+基线(main ≥69876bf;checker 374/0;pytest 77)

### Task 2: CP/Ring Attention 归一
- [ ] 抽通用机制页 `01_theory/06_distributed_parallelism/ring_attention_and_context_parallel_analysis.md`:以四页(`megatron-lm/megatron_cp_analysis` 391、`torchtitan/torchtitan_cp_analysis` 345、`mindspeed/mindspeed_context_parallel_analysis` 420、`megatron-lm/deepseek_v4_context_parallel_analysis` 853)中讲**通用机制**的段(序列切分/因果裁剪/通信量与 S/CP 代数/Ring 调度)为素材,择最深版本逐字为骨架,其余版本差异并注;四页收缩为各自实现差异+指向通用页;mindspeed 页已有划界声明为范本;台账逐节
- [ ] checker=0;commit

### Task 3: 顶层横向页收缩 + Megatron 优化器三页合并
- [ ] `02_train_frameworks/comm_compute_overlap_analysis.md`(271)收缩为纯对比矩阵页,机制正文下沉 megatron-lm/torchtitan/mindspeed 对应页(下沉前验收接页覆盖,独有逐字);`comm_compute_fusion_guide.md`(361)与 overlap 页补"融合 vs 掩盖"划界互指
- [ ] `distributed_optimizer_deep_dive.md`(194)同规程收缩为横向对比;megatron-lm 内 `megatron_distributed_optimizer_analysis`(445)/`megatron_ddp_optimizer_analysis`(446)/`megatron_optimizer_internals_analysis`(227)三页逐节台账合并为一页(以引用最全者为骨架)
- [ ] torchtitan 四页 FSDP 补显式分工声明(每页头一句);checker=0;commit(可拆二)

### Task 4: Roofline/执行模型归一(05_gpu_kernel)
- [ ] `operator_optimization_guide.md`(834,P4 迁入)§2 Roofline/§3 GPU 路径与 `gpu_kernel_guide.md`(303)§01/02、`cuda_execution_model_guide.md`(280)、`triton/triton_00_gpu_essentials_guide.md`(119)§2、`triton_06_optimization_profiling_guide.md`(297)台账归一:执行模型权威=cuda_execution_model_guide(吸收其它版本独有),Roofline 权威=operator_optimization_guide 对应节(其余只链);§6 昇腾段与 `ascend_kernel_execution_model_analysis.md`(213)合并判定(独有>50% 留)
- [ ] checker=0;commit

### Task 5: npu 三页划界 + 中重叠扫尾
- [ ] `02_compile_stack/04_inductor/npu/` 的 `npu_lowering_guide`(884)/`npu_fusion_passes_deepdive`(425)/`npu_vs_upstream_fusion_passes`(287)与 `10_fx_lowering_to_inductor_ir_analysis`(原C17)/`03_graph_ir_and_passes` pass 页划界:上游机制重叠段收缩指新页,NPU 特有全留;台账
- [ ] 中重叠 7 组扫尾(spec §3.6 残余):逐组检查现状(P3-P5 已处理多组),未补双向链的补齐(vLLM compilation↔runtime_graphs、TIM 分层等);一句话+链接式收缩仅对仍在复述的
- [ ] checker=0;commit

### Task 6: P6 阶段门(控制者):changelog 阶段条目、merge、push、roadmap

## P7(分支 reorg/p7)

### Task 7: 全库编号/命名推广(分批 commit)
- [ ] 未编号目录施行段位编号(≥4 内容页的):`01_theory/01_models/`(根 3 页豁免?按数)及 deepseek/(21)、moonshot_kimi/(14)、zhipu_glm/(10)、tencent_hunyuan、meituan_longcat、thinking_machines(小目录按门槛豁免)、`02_pretraining/`(7)、`06_distributed_parallelism/`(9)、`02_train_frameworks/` 根(收缩后)与 megatron-lm/(30)、torchtitan/(13)、mindspeed/(6)、mindformers/(3 豁免)、`03_infer_frameworks/` vllm/(13)、speculative_decoding/sglang(豁免)、`05_gpu_kernel/` 根与 triton/(9)、`06_auto_parallel/`(豁免)、`07_training_reliability/`(5)
- [ ] snake_case 残余清理(编号时一并):`RL_Training_Inference_Precision_Analysis`、`Engram_Analysis`、`mHC`、`kimi_k*.5`(点号)、`PyTorch_CUDA_Graphs_Complete_Guide`、`CUDA_Graphs_Timing…`(已删)、`NPU_Inductor_Backend_Analysis`、`NPU_MLIR_Backend…` 等 P4 未触的大写页
- [ ] 每批 checker=0+pytest

### Task 8: 链接治理收尾
- [ ] 裸 `[[index]]` 69 处 → 路径限定(检查器口径清零 ambiguous/bare_index)
- [ ] 正文句主位裸编号链接补显示名(P4 审查估计 ~120-150 处,`[[NN_xxx|语义名]]`)
- [ ] checker:broken=0 且 ambiguous=0 bare_index=0;commit

### Task 9: changelog 归档 + CLAUDE.md 完整修订 + README/index 定稿
- [ ] changelog 按季度拆 `wiki/changelog/2026Q2.md` 等,主文件留当季+政策头
- [ ] CLAUDE.md 全面修订(spec §6 逐条):courses 规则、页面类型表(+_quickstart/_deepdive/编号约定,禁 README)、链接规则(§4 新规)、Related Pages 3-7 精选制、索引维护(只表格不深树/不写精确页数)、源政策(代码页钉基线)、保留 Mermaid 规范与 >500 行拆分提议;"合并优于并存"已入(P3),核对表述
- [ ] README 定稿;wiki/index 终校(courses 区/域表/页数口径)
- [ ] checker+pytest;commit

### Task 10: 终门(控制者):全库终检(broken=0/ambiguous=0)、changelog 总结条目、merge、push、roadmap 全 ✅、spec 附执行后记
