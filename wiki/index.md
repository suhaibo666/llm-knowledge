# LLM Knowledge Wiki — 知识库总索引

> 最后更新: 2026-05-07

---

## 目录结构

```
wiki/
├── index.md                    # ← 本文件 — 知识库总索引
├── changelog.md                # 变更日志
├── llm/                        # LLM 训练与优化
│   ├── index.md                # LLM 领域总索引
│   ├── 01_architecture/        # Transformer, 缩放定律, 记忆架构
│   │   └── index.md
│   ├── 02_training/            # 优化器, 初始化, 低精度训练, 数值精度
│   │   └── index.md
│   ├── 03_alignment/           # RLHF, DPO, GRPO, PPO 等对齐方法
│   │   └── index.md
│   ├── 04_reasoning_and_retrieval/ # CoT, RAG, 验证 (待建设)
│   │   └── index.md
│   ├── 05_model_families/      # 按组织的模型技术报告
│   │   ├── index.md
│   │   ├── deepseek/           # DeepSeek 系列
│   │   │   └── index.md
│   │   ├── moonshot_kimi/      # Kimi 系列
│   │   │   └── index.md
│   │   └── zhipu_glm/         # GLM 系列
│   │       └── index.md
│   ├── 06_infra/               # 分布式训练基础设施
│   │   ├── index.md
│   │   └── megatron-lm/        # Megatron-LM
│   │       └── index.md
│   ├── 07_multimodal/          # 多模态 (待建设)
│   │   └── index.md
│   └── 08_agents/              # Agent (待建设)
│       └── index.md
├── torch_compile/              # PyTorch 编译栈
│   ├── index.md
│   ├── cudagraphs/             # CUDA/NPU Graphs
│   │   ├── index.md
│   │   └── npugraphs/          # NPU Graphs
│   │       └── index.md
│   └── inductor/               # TorchInductor
│       └── index.md
└── changelog.md
```

---

## 领域总览

| 领域 | 入口 | 页面数 | 状态 |
|------|------|--------|------|
| **LLM 训练与优化** | [[llm/index]] | 70+ | 活跃 |
| ├─ 基础架构 | [[llm/01_architecture/index]] | 5 | 活跃 |
| ├─ 训练技术 | [[llm/02_training/index]] | 5 | 活跃 |
| ├─ 对齐与偏好优化 | [[llm/03_alignment/index]] | 11 | 活跃 |
| ├─ 推理与检索 | [[llm/04_reasoning_and_retrieval/index]] | 0 | 待建设 |
| ├─ 模型家族 | [[llm/05_model_families/index]] | 30+ | 活跃 |
| │  ├─ DeepSeek | [[llm/05_model_families/deepseek/index]] | 18 | 活跃 |
| │  ├─ Kimi | [[llm/05_model_families/moonshot_kimi/index]] | 4 | 活跃 |
| │  └─ GLM | [[llm/05_model_families/zhipu_glm/index]] | 3 | 活跃 |
| ├─ 训练基础设施 | [[llm/06_infra/index]] | 6 | 活跃 |
| │  └─ Megatron-LM | [[llm/06_infra/megatron-lm/index]] | 6 | 活跃 |
| ├─ 多模态 | [[llm/07_multimodal/index]] | 0 | 待建设 |
| └─ Agent | [[llm/08_agents/index]] | 0 | 待建设 |
| **PyTorch 编译栈** | [[torch_compile/index]] | 30+ | 活跃 |
| ├─ CUDA/NPU Graphs | [[torch_compile/cudagraphs/index]] | 10 | 活跃 |
| │  └─ NPU Graphs | [[torch_compile/cudagraphs/npugraphs/index]] | 8 | 活跃 |
| └─ TorchInductor | [[torch_compile/inductor/index]] | 17 | 活跃 |

---

## 快速导航

### 按主题查找

| 主题 | 主要页面 |
|------|---------|
| **Transformer 原理** | [[attention_is_all_you_need_analysis]] |
| **缩放定律** | [[scaling_laws_analysis]], [[long_context_scaling_law_analysis]] |
| **优化器** | [[muon_analysis]] |
| **低精度训练** | [[low_precision_training_analysis]], [[transformer_engine_analysis]], [[deepseek_v4_fp4_qat_analysis]] |
| **对齐/RLHF** | [[instructgpt_rlhf_analysis]], [[dpo_analysis]], [[grpo_analysis]], [[ppo_analysis]] |
| **DeepSeek 模型** | [[deepseek_v4_analysis]], [[deepseek_v3_analysis]], [[deepseek_r1_analysis]] |
| **Megatron 分布式** | [[Megatron-LM_Distributed_Parallel_Exam]], [[megatron_comm_overlap_analysis]] |
| **MoE** | [[Megatron-LM_MoE_Zero_Redundancy_Analysis]], [[deepseek_moe_analysis]] |
| **torch.compile** | [[torch_compile_architecture]], [[PyTorch_Dynamo_Technical_Analysis]], [[PyTorch_Inductor_Technical_Analysis]] |
| **CUDA/NPU Graphs** | [[PyTorch_CUDA_Graphs_Complete_Guide]], [[torch_compile_npugraphs_deep_dive]] |
| **PPO/GRPO RL 训练** | [[RL_PPO_Loss_and_GRPO_Analysis]], [[RL_Training_Inference_Precision_Analysis]] |

### 按原始来源泛型

| 来源类型 | 位置 |
|---------|------|
| **学术论文** | `raw/01_architecture/` ~ `raw/08_agents/` |
| **PyTorch 内部图表** (.eddx) | `raw/09_pytorch/00_compile/` |
| **NVIDIA Megatron/TE** | `raw/06_moe_and_distributed/` |

---

## 维护说明

- 本索引随知识库同步更新
- 新建页面后需在对应 `index.md` 中添加条目
- 变更历史见 [[changelog]]
