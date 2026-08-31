---
title: "LLM Knowledge Wiki — 知识库总索引"
---

# LLM Knowledge Wiki — 知识库总索引

> 最后更新: 2026-08-27

---

## 目录结构

目录结构以文件系统与各域 index 为准；本索引只维护下方领域总览。

---

## 领域总览

### 01 理论研究

| 领域 | 入口 | 页面数 | 状态 |
|------|------|--------|------|
| 模型 | [[01_theory/01_models/index]] | 67 | 活跃 |
| └─ DeepSeek | [[01_theory/01_models/deepseek/index]] | 22 | 活跃 |
| └─ Kimi | [[01_theory/01_models/moonshot_kimi/index]] | 15 | 活跃 |
| └─ GLM | [[01_theory/01_models/zhipu_glm/index]] | 12 | 活跃 |
| └─ Qwen | [[01_theory/01_models/alibaba_qwen/index]] | 6 | 活跃 |
| └─ LongCat (美团) | [[01_theory/01_models/meituan_longcat/index]] | 3 | 活跃 |
| 预训练 | [[01_theory/02_pretraining/index]] | 7 | 活跃 |
| SFT & 低参微调 | [[01_theory/03_sft/index]] | 1 | 待建设 |
| 后训练对齐 | [[01_theory/04_posttraining/index]] | 18 | 活跃 |
| 推理技术 | [[01_theory/05_inference/index]] | 1 | 待建设 |
| 分布式并行理论 | [[01_theory/06_distributed_parallelism/index]] | 9 | 活跃 |

> 页面数为递归统计（含各级 index，不含 SUPERSEDED 存根）；模型域各行于 **2026-08-27** 随 GLM/Qwen/Kimi/DeepSeek 新发布模型摄入重新统计，其余行沿用 2026-08-04（kb-reorg P7 收尾）的统计。

### 02 工程实现

| 领域 | 入口 | 页面数 | 状态 |
|------|------|--------|------|
| PyTorch | [[02_engineering/01_pytorch/index]] | 148 | 活跃 |
| └─ TorchInductor | [[02_engineering/01_pytorch/02_compile_stack/04_inductor/index]] | 36 | 活跃 |
| └─ 运行时图(CUDA/NPU) | [[02_engineering/01_pytorch/03_runtime_graphs/index]] | 12 | 活跃 |
| └─ Codegen 后端(MLIR) | [[02_engineering/01_pytorch/02_compile_stack/05_codegen_backends/mlir/index]] | 8 | 活跃 |
| 训练框架 | [[02_engineering/02_train_frameworks/index]] | 67 | 活跃 |
| └─ Megatron-LM | [[02_engineering/02_train_frameworks/megatron-lm/index]] | 27 | 活跃 |
| └─ torchtitan | [[02_engineering/02_train_frameworks/torchtitan/index]] | 24 | 活跃 |
| └─ MindFormers | [[02_engineering/02_train_frameworks/mindformers/index]] | 3 | 活跃 |
| └─ MindSpeed | [[02_engineering/02_train_frameworks/mindspeed/index]] | 6 | 活跃 |
| 推理框架 | [[02_engineering/03_infer_frameworks/index]] | 20 | 活跃 |
| └─ vLLM | [[02_engineering/03_infer_frameworks/vllm/index]] | 13 | 活跃 |
| 后训练框架 | [[02_engineering/04_posttrain_frameworks/index]] | 47 | 活跃 |
| └─ verl (HybridFlow) | [[02_engineering/04_posttrain_frameworks/verl/index]] | 15 | 活跃 |
| └─ slime | [[02_engineering/04_posttrain_frameworks/slime/index]] | 21 | 活跃 |
| GPU Kernel | [[02_engineering/05_gpu_kernel/index]] | 17 | 活跃 |
| └─ Triton 学习路线 | [[02_engineering/05_gpu_kernel/triton/index]] | 9 | 活跃 |
| 自动并行 | [[02_engineering/06_auto_parallel/index]] | 2 | 活跃 |
| 训练可靠性 | [[02_engineering/07_training_reliability/index]] | 5 | 活跃 |

> 页面数为递归统计（含各级 index，不含 SUPERSEDED 存根）；训练框架与 torchtitan 于 2026-08-27 随 TorchTitan `a3168782c` 机制级复审重新统计为 67/24 页；后训练框架于 2026-08-28 随 verl `254a23ed` 全域复审增至 47 页，其中 `verl/` 15 页、`slime/` 21 页；其余工程域沿用最近一次域级统计。

### courses 课程入口

跨领域的纯导读页(只含阅读顺序、链接与一句话导读,正文归属对应功能树,详见
`CLAUDE.md`/spec §6 课程页规则):

| 课程 | 入口 | 覆盖范围 |
|------|------|---------|
| torch.compile 端到端 | [[courses/torch_compile_end_to_end]] | `01_pytorch` 四层功能树(eager 地基→Dynamo→AOTAutograd→Graph IR/Passes→Inductor→缓存→调试→运行时图→导出/分布式) |
| LLM 后训练前沿 | [[courses/posttraining_frontier]] | `01_theory/04_posttraining`(算法)+ `02_engineering/04_posttrain_frameworks`(框架源码,含 `verl/`、`slime/`)+ `moonshot_kimi`(K3 工业案例)三处功能树(D01→D12:全景地图→算法演进→Agentic→staleness→Infra→框架对比→verl 主链→slime→AReaL→ROLL→CUDA–Ascend→K3 综合案例) |

---

## 快速导航

### 按主题查找

| 主题 | 主要页面 |
|------|---------|
| Transformer 原理 | [[attention_is_all_you_need_analysis]] |
| 缩放定律 | [[scaling_laws_analysis]], [[long_context_scaling_law_analysis]] |
| 优化器 | [[11_muon_analysis]] |
| 低精度训练 | [[13_low_precision_training_analysis]], [[14_transformer_engine_analysis]], [[24_deepseek_v4_fp4_qat_analysis]] |
| 对齐/RLHF | [[10_instructgpt_rlhf_analysis]], [[12_dpo_analysis]], [[20_grpo_analysis]], [[11_ppo_analysis]] |
| DeepSeek 模型 | [[13_deepseek_v4_analysis]], [[12_deepseek_v3_analysis]], [[14_deepseek_r1_analysis]] |
| Qwen 模型 | [[01_theory/01_models/alibaba_qwen/index]], [[10_qwen3_8_analysis]] |
| LongCat (美团) | [[meituan_longcat/index]], [[longcat_flash_analysis]], [[longcat_2_analysis]] |
| Megatron 分布式 | [[17_megatron_parallelism_orchestration_analysis]], [[20_megatron_comm_overlap_analysis]] |
| MoE | [[14_megatron_ep_analysis]], [[20_deepseek_moe_analysis]] |
| MoE 专家并行 (MindFormers) | [[mindformers/index]], [[mindformers_pynative_ep_analysis]], [[mindformers_moe_token_dispatcher_analysis]] |
| 昇腾训练加速 (MindSpeed) | [[mindspeed/index]], [[10_mindspeed_parallelism_analysis]], [[20_mindspeed_context_parallel_analysis]], [[11_mindspeed_comm_overlap_analysis]], [[12_mindspeed_memory_optimization_analysis]], [[13_mindspeed_ascend_affinity_analysis]] |
| torch.compile | [[courses/torch_compile_end_to_end]], [[02_compile_stack/01_dynamo/index]], [[02_compile_stack/04_inductor/index]] |
| CUDA/NPU Graphs | [[10_pytorch_cuda_graphs_complete_guide]], [[11_torch_compile_npugraphs_deepdive]] |
| Triton kernel 入门→专家 | [[triton_10_programming_model_guide]], [[triton_12_matmul_guide]], [[triton_13_autotune_guide]], [[triton_30_optimization_profiling_guide]], [[triton_31_knowledge_guide]] |
| GPU/NPU 执行模型与 GEMM | [[10_cuda_execution_model_guide]], [[20_cuda_gemm_kernel_analysis]], [[22_ascend_kernel_execution_model_analysis]] |
| 非 GEMM Kernel 优化 | [[21_cuda_nonmatmul_kernels_analysis]], [[01_gpu_kernel_guide]], [[triton_01_gpu_essentials_guide]] |
| vLLM 推理引擎 | [[vllm/index]], [[10_vllm_engine_architecture_analysis]], [[11_vllm_scheduler_analysis]], [[12_vllm_kv_cache_management_analysis]], [[01_vllm_feature_optimizations_guide]] |
| vLLM 图编译/算子融合 | [[24_vllm_fused_ops_and_kernels_analysis]], [[25_vllm_ir_and_fusion_passes_analysis]], [[23_vllm_compilation_cudagraph_analysis]] |
| PPO/GRPO RL 训练 | [[10_rl_ppo_loss_and_grpo_analysis]], [[20_rl_training_inference_precision_analysis]] |
| RL 训练框架 (verl/HybridFlow) | [[02_engineering/04_posttrain_frameworks/verl/index|verl 分析域]], [[01_verl_architecture_overview_analysis]], [[20_verl_ray_trainer_analysis]], [[15_verl_rl_algorithms_analysis]] |
| Coding RL「脏活」系列 | [[31_reward_hacking_defense_analysis]], [[11_rl_sandbox_design_analysis]], [[12_rl_infra_efficiency_analysis]] |
| LLM 后训练前沿 D01–D06 | [[courses/posttraining_frontier]], [[01_posttraining_frontier_map_analysis]], [[13_reasoning_rl_algorithm_evolution_analysis]], [[24_agentic_rl_algorithm_analysis]], [[25_on_policy_off_policy_staleness_analysis]], [[01_posttraining_infra_mechanism_analysis]], [[30_rl_framework_comparison]] |
| LLM 后训练前沿 D07–D12 | [[10_verl_end_to_end_iteration_analysis]], [[slime/index]], [[01_slime_architecture_overview_analysis]], [[30_slime_rollout_optimization_analysis]], [[17_slime_train_inference_consistency_analysis]], [[31_slime_posttraining_stability_analysis]], [[25_vime_vllm_backend_support_analysis]], [[21_areal_async_architecture_analysis]], [[22_roll_strategy_and_ascend_analysis]], [[31_cuda_ascend_posttraining_stack_comparison]], [[24_kimi_k3_posttraining_case_study_analysis]] |
| 万卡训练确定性与可靠性 | [[07_training_reliability/index]], [[10_determinism_and_numerical_reliability_analysis]], [[11_fault_tolerance_and_recovery_analysis]], [[12_training_dynamics_stability_analysis]] |

### 按原始来源

| 来源类型 | 位置 |
|---------|------|
| 学术论文 | `raw/01_theory/` |
| 训练/推理框架图表 (.eddx) | `raw/02_engineering/02_train_frameworks/` |
| PyTorch 内部图表 (.eddx) | `raw/02_engineering/01_ai_frameworks/` |

---

## 维护说明

- 本索引随知识库同步更新
- 新建页面后需在对应 `index.md` 中添加条目
- 变更历史见 [[changelog]]
