# GLM/智谱 AI 技术路线总览

> GLM (General Language Model) 是智谱 AI (Zhipu AI) 与清华大学 KEG 实验室开发的大语言模型系列。

---

## 一、GLM 模型家族

| 模型 | 发布时间 | 参数量 | 核心能力 | arXiv |
|------|---------|--------|---------|-------|
| **GLM-130B** | 2022.10 | 130B | 首个双语开源 LLM | 2210.02414 |
| **ChatGLM/GLM-4** | 2024.06 | - | 消费级部署，多轮对话 | 2406.12793 |
| **CodeGeeX** | 2023.06 | - | 代码生成，多语言 | 2306.03078 |
| **CogView** | 2021.05 | - | 文本到图像生成 | 2105.13290 |
| **CogVideo** | 2022.04 | - | 文本到视频生成 | 2204.14230 |
| **CharacterGLM** | 2023.11 | - | 角色扮演对话 | 2311.16832 |
| **GLM-4 Voice** | 2024.12 | - | 语音理解与生成 | 2412.02612 |
| **GLM-TTS** | 2025.12 | - | 文本到语音合成 | 2512.14291 |
| **GLM-5** | 2026.02 | 744B/40B | Vibe Coding → Agentic Engineering | 2602.15763 |
| **GLM-5.1** | 2026.04 | 754B | 最新迭代 | - |
| **GLM-5V-Turbo** | 2026.04 | - | 原生多模态 Agent | 2604.26752 |

---

## 二、技术演进时间线

```
2021.05  CogView
    │      └── 文本到图像生成
    │
2022.04  CogVideo
    │      └── 文本到视频生成
    │
2022.10  GLM-130B
    │      └── 首个中英双语开源 LLM
    │      └── 130B 参数
    │
2023.06  CodeGeeX
    │      └── 多语言代码生成
    │
2023.11  CharacterGLM
    │      └── 角色扮演对话系统
    │
2024.06  ChatGLM / GLM-4
    │      └── 消费级部署优化
    │      └── 多轮对话能力
    │
2024.12  GLM-4 Voice
    │      └── 语音理解与生成
    │
2025.12  GLM-TTS
    │      └── 文本到语音合成
    │
2026.02  GLM-5
    │      └── 744B/40B MoE (256 专家)
    │      └── DSA 稀疏注意力
    │      └── 异步 RL 基础设施
    │      └── 28.5T tokens 预训练
    │      └── Muon Split + MLA-256
    │      └── 国产 GPU 全栈适配
    │
2026.04  GLM-5.1
    │      └── 754B 参数
    │      └── 最新迭代
    │
2026.04  GLM-5V-Turbo
           └── 原生多模态 Agent
           └── CogViT 视觉编码器
           └── MMTP 多模态多 Token 预测
           └── 30+ 任务联合 RL
           └── ImageMining 基准
```

---

## 三、核心技术栈

### 3.1 架构演进

```
GLM-130B (Dense)
    │
    ▼
GLM-4 (优化部署)
    │
    ▼
GLM-4.5 (MoE, ARC 能力统一)
    │
    ▼
GLM-5 (744B/40B MoE)
    ├── Muon Split: per-head 正交化
    ├── MLA-256: head dim 192→256
    ├── MTP 参数共享 (3 层)
    ├── DSA 稀疏注意力
    └── 256 专家，8 激活
    │
    ▼
GLM-5.1 (754B)
    │
    ▼
GLM-5V-Turbo (多模态)
    ├── CogViT (两阶段预训练)
    ├── MMTP (多模态 MTP)
    └── 30+ 任务联合 RL
```

### 3.2 训练基础设施

```
GLM-5 训练基础设施:
├── 内存优化
│   ├── Flexible MTP placement
│   ├── Pipeline ZeRO2 gradient sharding
│   ├── Zero-redundant Muon communication
│   ├── Pipeline activation offloading
│   └── Sequence-chunked output projection
│
├── 并行优化
│   ├── Efficient deferred weight gradient
│   └── Efficient long-sequence training
│
└── 精度
    └── INT4 QAT (bitwise-identical)
```

### 3.3 RL 框架

```
GLM-5 RL 框架:
├── Reasoning RL
│   ├── GRPO + IcePop
│   ├── 训练-推理不匹配缓解
│   └── 混合领域 (数学+科学+代码+TIR)
│
├── Agentic RL
│   ├── 异步解耦框架
│   ├── TITO gateway
│   ├── Direct Double-sided Importance Sampling
│   └── 10K+ SWE + Terminal + Search 环境
│
├── General RL
│   ├── Foundational correctness
│   ├── Emotional intelligence
│   └── Task-specific quality
│
└── On-Policy Cross-Stage Distillation
    └── 防止灾难性遗忘
```

---

## 四、GLM-5 关键技术

### 4.1 Muon Split

**问题**：MLA 的 576 维 latent KV-cache 在 Muon 优化器下无法匹配 GQA-8 性能。

**方案**：将 up-projection 矩阵按头拆分，独立正交化。

**效果**：MLA 性能匹配 GQA-8，训练过程**无需 logits clipping**。

### 4.2 DSA 稀疏注意力

通过 Continued Pre-Training 从 dense 转换：
- Warmup: 1000 steps
- Sparse Adaptation: 20B tokens
- 注意力计算减少 **1.5-2×**
- **无损**：唯一可应用于所有层的高效注意力方案

### 4.3 国产 GPU 适配

- 华为 Ascend、摩尔线程、海光、寒武纪、昆仑芯、沐曦、燧原
- 从底层 kernel 到上层推理框架全栈优化

---

## 五、GLM-5V-Turbo 关键技术

### 5.1 CogViT 视觉编码器

| 阶段 | 方法 | 关键配置 |
|------|------|---------|
| Stage 1 | 蒸馏式掩码图像建模 | SigLIP2 + DINOv3 双教师, 35% 掩码, Muon |
| Stage 2 | 对比式图文预训练 | NaFlex 可变分辨率, 64K batch, 80 亿图文对 |

### 5.2 MMTP 多模态多 Token 预测

采用 `<|image|>` 共享 token 方案：
- 无需跨 pipeline 传播视觉 embedding
- 兼容 sequence/context parallelism
- 训练 loss 更低，收敛更稳定

### 5.3 联合多模态 RL

30+ 任务类别联合优化，感知/推理/Agent 能力全面提升：
- 2D grounding: +4.8%
- 视频理解: +5.6%
- 3D grounding: +7.7%
- OCR: +4.2%
- 图表理解: +7.7%
- GUI Agent: +4.9%

---

## 六、关键论文索引

| 论文 | arXiv | Wiki 页面 |
|------|-------|----------|
| GLM-130B | 2210.02414 | 待摄入 |
| ChatGLM/GLM-4 | 2406.12793 | 待摄入 |
| CodeGeeX | 2306.03078 | 待摄入 |
| CogView | 2105.13290 | 待摄入 |
| CogVideo | 2204.14230 | 待摄入 |
| CharacterGLM | 2311.16832 | 待摄入 |
| GLM-4 Voice | 2412.02612 | 待摄入 |
| GLM-TTS | 2512.14291 | 待摄入 |
| **GLM-5** | **2602.15763** | **[[glm_5_analysis]]** |
| **GLM-5V-Turbo** | **2604.26752** | **[[glm_5v_turbo_analysis]]** |

---

## 七、知识缺口

以下 raw 文件尚未摄入到 wiki：

- GLM-130B (2210.02414) — 首个双语开源 LLM
- ChatGLM/GLM-4 (2406.12793) — 消费级部署
- CodeGeeX (2306.03078) — 代码生成
- CogView (2105.13290) — 文本到图像
- CogVideo (2204.14230) — 文本到视频
- CharacterGLM (2311.16832) — 角色扮演
- GLM-4 Voice (2412.02612) — 语音
- GLM-TTS (2512.14291) — TTS

---

## Related Pages

- [[llm/overview]]
- [[llm/05_model_families/moonshot_kimi/kimi_overview]]
- [[llm/06_infra/megatron-lm/overview]]
