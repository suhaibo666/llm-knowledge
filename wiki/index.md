# LLM Knowledge Wiki — 知识库总索引

> 最后更新: 2026-07-22

---

## 目录结构

```
wiki/
├── index.md                           # ← 本文件
├── changelog.md                       # 变更日志
├── 01_theory/                         # 理论研究
│   ├── 01_models/                     # 模型架构 + 模型家族
│   │   ├── index.md
│   │   ├── deepseek/
│   │   ├── moonshot_kimi/
│   │   └── zhipu_glm/
│   ├── 02_pretraining/                # 预训练技术
│   ├── 03_sft/                        # SFT + 低参微调
│   ├── 04_posttraining/               # 后训练对齐 (RLHF/DPO/GRPO)
│   └── 05_inference/                  # 推理技术 (CoT/RAG/Agent)
└── 02_engineering/                    # 工程实现
    ├── 01_ai_frameworks/              # AI框架 (PyTorch compile)
    │   ├── cudagraphs/
    │   ├── inductor/
    │   └── mlir/
    ├── 02_train_frameworks/           # 训练框架 (Megatron-LM / torchtitan / MindFormers / MindSpeed)
    │   ├── megatron-lm/
    │   ├── torchtitan/
    │   ├── mindformers/               # MoE 专家并行(PyNative + Graph)
    │   └── mindspeed/                 # 昇腾 MindSpeed 训练加速特性(并行/掩盖/内存/亲和)
    ├── 03_infer_frameworks/           # 推理框架
    │   └── vllm/                       # vLLM V1 引擎(12 篇 + index)
    ├── 04_posttrain_frameworks/       # 后训练框架 (verl / RLHF Infra)
    │   └── verl/                      # verl (HybridFlow) 源码级分析
    ├── 05_gpu_kernel/                 # GPU/NPU Kernel 工程 (含 triton/ 学习路线)
    ├── 06_auto_parallel/             # 自动并行策略搜索(综述罗盘)
    └── 07_training_reliability/       # 万卡训练确定性与可靠性(9 问题域·多来源综述)
```

---

## 领域总览

### 01 理论研究

| 领域 | 入口 | 页面数 | 状态 |
|------|------|--------|------|
| 模型 | [[01_theory/01_models/index]] | 31 | 活跃 |
| └─ DeepSeek | [[01_theory/01_models/deepseek/index]] | 19 | 活跃 |
| └─ Kimi | [[01_theory/01_models/moonshot_kimi/index]] | 6 | 活跃 |
| └─ GLM | [[01_theory/01_models/zhipu_glm/index]] | 3 | 活跃 |
| └─ LongCat (美团) | [[01_theory/01_models/meituan_longcat/index]] | 3 | 活跃 |
| 预训练 | [[01_theory/02_pretraining/index]] | 7 | 活跃 |
| SFT & 低参微调 | [[01_theory/03_sft/index]] | 0 | 待建设 |
| 后训练对齐 | [[01_theory/04_posttraining/index]] | 14 | 活跃 |
| 推理技术 | [[01_theory/05_inference/index]] | 1 | 待建设 |

### 02 工程实现

| 领域 | 入口 | 页面数 | 状态 |
|------|------|--------|------|
| AI框架 | [[02_engineering/01_ai_frameworks/index]] | 45 | 活跃 |
| └─ TorchInductor | [[02_engineering/01_ai_frameworks/04_inductor/index]] | 21 | 活跃 |
| └─ 运行时图(CUDA/NPU) | [[02_engineering/01_ai_frameworks/06_graphs/index]] | 10 | 活跃 |
| └─ Codegen 后端(MLIR) | [[02_engineering/01_ai_frameworks/05_codegen_backends/mlir/index]] | 4 | 活跃 |
| 训练框架 | [[02_engineering/02_train_frameworks/index]] | 8 | 活跃 |
| └─ Megatron-LM | [[02_engineering/02_train_frameworks/megatron-lm/index]] | 6 | 活跃 |
| └─ torchtitan | [[02_engineering/02_train_frameworks/torchtitan/index]] | 7 | 活跃 |
| └─ MindFormers | [[02_engineering/02_train_frameworks/mindformers/index]] | 2 | 活跃 |
| └─ MindSpeed | [[02_engineering/02_train_frameworks/mindspeed/index]] | 6 | 活跃 |
| 推理框架 | [[02_engineering/03_infer_frameworks/index]] | 14 | 活跃 |
| └─ vLLM | [[02_engineering/03_infer_frameworks/vllm/index]] | 13 | 活跃 |
| 后训练框架 | [[02_engineering/04_posttrain_frameworks/index]] | 12 | 活跃 |
| └─ verl (HybridFlow) | [[02_engineering/04_posttrain_frameworks/verl/index]] | 9 | 活跃 |
| GPU Kernel | [[02_engineering/05_gpu_kernel/index]] | 14 | 活跃 |
| └─ Triton 学习路线 | [[02_engineering/05_gpu_kernel/triton/index]] | 9 | 活跃 |
| 自动并行 | [[02_engineering/06_auto_parallel/index]] | 1 | 活跃 |
| 训练可靠性 | [[02_engineering/07_training_reliability/index]] | 4 | 活跃 |

---

## 快速导航

### 按主题查找

| 主题 | 主要页面 |
|------|---------|
| Transformer 原理 | [[attention_is_all_you_need_analysis]] |
| 缩放定律 | [[scaling_laws_analysis]], [[long_context_scaling_law_analysis]] |
| 优化器 | [[muon_analysis]] |
| 低精度训练 | [[low_precision_training_analysis]], [[transformer_engine_analysis]], [[deepseek_v4_fp4_qat_analysis]] |
| 对齐/RLHF | [[instructgpt_rlhf_analysis]], [[dpo_analysis]], [[grpo_analysis]], [[ppo_analysis]] |
| DeepSeek 模型 | [[deepseek_v4_analysis]], [[deepseek_v3_analysis]], [[deepseek_r1_analysis]] |
| LongCat (美团) | [[meituan_longcat/index]], [[longcat_flash_analysis]], [[longcat_2_analysis]] |
| Megatron 分布式 | [[megatron_parallelism_orchestration_analysis]], [[megatron_comm_overlap_analysis]] |
| MoE | [[megatron_ep_analysis]], [[deepseek_moe_analysis]] |
| MoE 专家并行 (MindFormers) | [[mindformers/index]], [[mindformers_pynative_ep_analysis]], [[mindformers_moe_token_dispatcher_analysis]] |
| 昇腾训练加速 (MindSpeed) | [[mindspeed/index]], [[mindspeed_parallelism_analysis]], [[mindspeed_context_parallel_analysis]], [[mindspeed_comm_overlap_analysis]], [[mindspeed_memory_optimization_analysis]], [[mindspeed_ascend_affinity_analysis]] |
| torch.compile | [[torch_compile_architecture]], [[PyTorch_Dynamo_Technical_Analysis]], [[PyTorch_Inductor_Technical_Analysis]] |
| CUDA/NPU Graphs | [[PyTorch_CUDA_Graphs_Complete_Guide]], [[torch_compile_npugraphs_deep_dive]] |
| Triton kernel 入门→专家 | [[triton_01_programming_model_guide]], [[triton_03_matmul_guide]], [[triton_04_autotune_guide]], [[triton_06_optimization_profiling_guide]], [[triton_knowledge_map]] |
| GPU/NPU 执行模型与 GEMM | [[cuda_execution_model_guide]], [[cuda_gemm_kernel_analysis]], [[ascend_kernel_execution_model_analysis]] |
| 非 GEMM Kernel 优化 | [[cuda_nonmatmul_kernels_analysis]], [[gpu_kernel_guide]], [[triton_00_gpu_essentials_guide]] |
| vLLM 推理引擎 | [[vllm/index]], [[vllm_engine_architecture_analysis]], [[vllm_scheduler_analysis]], [[vllm_kv_cache_management_analysis]], [[vllm_feature_optimizations_overview]] |
| vLLM 图编译/算子融合 | [[vllm_fused_ops_and_kernels_analysis]], [[vllm_ir_and_fusion_passes_analysis]], [[vllm_compilation_cudagraph_analysis]] |
| PPO/GRPO RL 训练 | [[RL_PPO_Loss_and_GRPO_Analysis]], [[RL_Training_Inference_Precision_Analysis]] |
| RL 训练框架 (verl/HybridFlow) | [[verl/index]], [[verl_architecture_overview_analysis]], [[verl_ray_trainer_analysis]], [[verl_rl_algorithms_analysis]] |
| Coding RL「脏活」系列 | [[reward_hacking_defense_analysis]], [[rl_sandbox_design_analysis]], [[rl_infra_efficiency_analysis]] |
| 万卡训练确定性与可靠性 | [[07_training_reliability/index]], [[determinism_and_numerical_reliability_analysis]], [[fault_tolerance_and_recovery_analysis]], [[training_dynamics_stability_analysis]] |

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
