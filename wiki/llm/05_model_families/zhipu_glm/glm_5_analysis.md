# GLM-5: Vibe Coding 到 Agentic Engineering

> **论文**: GLM-5: from Vibe Coding to Agentic Engineering
> **作者**: GLM-5 Team (Zhipu AI & Tsinghua University)
> **arXiv**: 2602.15763 (2026-02)
> **开源**: https://github.com/zai-org/GLM-5
> **模型**: https://huggingface.co/zai-org/GLM-5

---

## 一、模型架构

### 1.1 核心参数

| 参数 | GLM-4.5 | GLM-5 | 变化 |
|------|---------|-------|------|
| 总参数量 | 355B | **744B** | ↑ 110% |
| 激活参数量 | 32B | **40B** | ↑ 25% |
| 专家总数 | - | **256** | 新增 |
| 每 Token 激活专家 | - | 8 | 新增 |
| 层数 | - | **80** | 减少通信开销 |
| 训练 Token | - | **28.5T** | 大规模扩展 |

### 1.2 架构设计

```
┌──────────────────────────────────────────────┐
│                GLM-5 Architecture             │
├──────────────────────────────────────────────┤
│  Input → Embedding                           │
│    │                                         │
│    ▼                                         │
│  Transformer Block × 80                      │
│  ┌────────────────────────────────────────┐ │
│  │  Multi-latent Attention (MLA)          │ │
│  │  - Muon Split: per-head orthogonalize  │ │
│  │  - MLA-256: head dim 192→256           │ │
│  │  - heads 减少 1/3，解码计算降低        │ │
│  └────────────────────────────────────────┘ │
│    │                                         │
│    ▼                                         │
│  ┌────────────────────────────────────────┐ │
│  │  MoE Layer                             │ │
│  │  - 256 experts, 8 active + 1 shared    │ │
│  │  - NO expert grouping                  │ │
│  └────────────────────────────────────────┘ │
│    │                                         │
│    ▼                                         │
│  ┌────────────────────────────────────────┐ │
│  │  MTP (Multi-Token Prediction)          │ │
│  │  - 3 层参数共享                        │ │
│  │  - 推理时预测 4 tokens                 │ │
│  │  - Accept Length 2.76 (vs V3.2 2.55)  │ │
│  └────────────────────────────────────────┘ │
└──────────────────────────────────────────────┘
```

### 1.3 关键架构创新

#### Muon Split

**问题**：MLA 的 576 维 latent KV-cache 在 Muon 优化器下无法匹配 GQA-8 性能。

**方案**：将 up-projection 矩阵按头拆分，独立正交化：

$$W^{UQ}, W^{UK}, W^{UV} \rightarrow \text{per-head matrices} \rightarrow \text{independent orthogonalization}$$

**效果**：不同注意力头的投影权重以不同尺度更新，MLA 性能匹配 GQA-8，且训练过程**无需 logits clipping**。

#### MLA-256

- Head dimension: 192 → **256**
- 注意力头数：减少 **1/3**
- 训练计算和参数量不变，**解码计算降低**

#### MTP 参数共享

- 训练时共享 **3 层 MTP** 参数
- 推理时预测 **4 tokens**（speculative decoding）
- Accept Length: **2.76**（DeepSeek-V3.2: 2.55）

### 1.4 DSA (DeepSeek Sparse Attention)

通过 Continued Pre-Training 从 dense 模型转换：

| 阶段 | 配置 |
|------|------|
| Warmup | 1000 steps, 14 sequences × 202,752 tokens, LR 5e-3 |
| Sparse Adaptation | 20B tokens, 同 mid-training 配置 |

**效果**：长上下文性能接近 MLA，注意力计算减少 **1.5-2×**。

| Benchmark (128K) | MLA | DSA |
|------------------|-----|-----|
| MQ-NIAH | 100.0 | 100.0 |
| MV-NIAH | 95.5 | **97.0** |
| SQuAD | 79.7 | **86.0** |
| HotpotQA | 66.3 | 63.0 |

---

## 二、预训练

### 2.1 数据规模

| 数据类型 | 说明 |
|---------|------|
| Web | DCLM 分类器 + World Knowledge classifier |
| Code | **28% 增长**，低资源语言专用分类器 |
| Math & Science | LLM 评分，chunk-and-aggregate 长文档评分 |

**总训练量**：**28.5T tokens**

### 2.2 Mid-Training

| 阶段 | 上下文 | Token 数 |
|------|--------|---------|
| Stage 1 | 32K | 1T |
| Stage 2 | 128K | 500B |
| Stage 3 | **200K** | 50B |

**软件工程数据**：~10M issue-PR pairs，~160B unique tokens

### 2.3 训练基础设施

**内存优化**：
- Flexible MTP placement
- Pipeline ZeRO2 gradient sharding
- Zero-redundant Muon communication
- Pipeline activation offloading
- Sequence-chunked output projection

**并行优化**：
- Efficient deferred weight gradient computation
- Efficient long-sequence training (workload-aware reordering, dynamic attention redistribution)

**INT4 QAT**：SFT 阶段应用，训练和推理 **bitwise-identical**

---

## 三、后训练

### 3.1 SFT

**三类数据**：
1. General Chat：QA、写作、角色扮演、翻译、多轮对话
2. Reasoning：数学、编程、科学推理
3. Coding & Agent：前后端工程、工具调用、编码/搜索 Agent

**三种 Thinking 特性**：
- Interleaved Thinking
- Preserved Thinking
- **Turn-level Thinking**：per-turn 控制推理

**最大上下文**：**202,752 tokens**

### 3.2 Reasoning RL

**算法**：GRPO + IcePop（训练-推理不匹配缓解）

$$\mathcal{L}(\theta) = -\mathbb{E}_{x,\{y_i\}}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\operatorname{pop}(\rho_{i,t},1/\beta,\beta) \cdot \min(r_{i,t}\hat{A}_{i,t},\operatorname{clip}(r_{i,t},1-\epsilon_{low},1+\epsilon_{high})\hat{A}_{i,t})\right]$$

**训练-推理不匹配比率**：
$$\rho_{i,t} = \frac{\pi_{\theta_{old}}^{\text{train}}(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_{old}}^{\text{infer}}(y_{i,t}|x,y_{i,<t})}$$

**超参数**：$\beta=2, \epsilon_{low}=0.2, \epsilon_{high}=0.28$，group size=32，batch size=32

**DSA RL 关键发现**：
- 使用 **torch.topk**（确定性）而非 CUDA non-deterministic topk
- RL 期间**冻结 indexer 参数**

**混合领域 RL**：数学 + 科学 + 代码 + TIR（工具集成推理）

### 3.3 Agentic RL

**异步解耦框架**：
- Multi-Task Rollout Orchestrator
- Token-in-Token-out (TITO) gateway
- Direct Double-sided Importance Sampling
- DP-aware routing（最大化 KV-cache 复用）

**训练环境**：
- 10K+ 真实 SWE 任务
- Terminal 任务
- 高难度多跳搜索任务

### 3.4 General RL

**三维度优化**：
1. Foundational correctness：指令遵循、逻辑一致性、事实准确性
2. Emotional intelligence：情感智能
3. Task-specific quality：任务特定质量

**On-Policy Cross-Stage Distillation**：防止灾难性遗忘

---

## 四、Agentic 能力

### 4.1 软件工程

- 端到端软件开发能力
- 10K+ 真实 GitHub issue-PR 环境
- 长程任务规划与自纠正

### 4.2 工具使用

- MCP 工具调用
- 搜索 Agent
- 多步工具编排

### 4.3 长程一致性

- Vending-Bench 2：模拟自动售货机业务运营（最终余额 $4,432，开源模型 #1）
- CC-Bench-V2：前端、后端、长程任务

---

## 五、性能基准

### 5.1 Agentic & Coding

| Benchmark | GLM-5 | GLM-4.7 | Claude Opus 4.5 | GPT-5.2 |
|-----------|-------|---------|-----------------|---------|
| **SWE-bench Verified** | **~65** | ~50 | ~69 | ~70 |
| **SWE-bench Multilingual** | **~70** | ~50 | ~78 | ~72 |
| **Terminal-Bench 2.0** | **~45** | ~35 | ~59 | ~54 |
| **BrowseComp** | **~50** | ~35 | ~58 | ~59 |
| **MCP-Atlas** | **领先** | - | - | - |
| **τ²-Bench** | **~60** | ~45 | ~59 | - |
| **Vending Bench 2** | **$4,432** | - | ~$5,000 | - |

### 5.2 推理与 STEM

| Benchmark | GLM-5 | GLM-4.7 | Claude Opus 4.5 | GPT-5.2 |
|-----------|-------|---------|-----------------|---------|
| **HLE** | **~30** | ~20 | ~35 | ~35 |
| **GPQA-Diamond** | **~75** | ~65 | ~78 | ~80 |

### 5.3 综合排名

- **Artificial Analysis Intelligence Index v4.0**：**50 分**（开源模型首次达到 50）
- **LMArena Text Arena**：开源模型 **#1**
- **LMArena Code Arena**：开源模型 **#1**
- 平均比 GLM-4.7 提升 **~20%**

---

## 六、高效注意力变体对比

GLM-5 团队对比了多种高效注意力方案：

| 方法 | RULER@128K | MRCR@128K | 特点 |
|------|-----------|-----------|------|
| Full Attention | 75.28 | 35.39 | 基准 |
| SWA Interleave | 44.93 (↓30.35) | 28.83 (↓6.56) | 灾难性下降 |
| SWA Pattern (Search) | 69.59 (↓5.69) | 33.58 (↓1.81) | 搜索最优层模式 |
| GDN | 64.00 (↓11.28) | 30.22 (↓5.17) | 线性注意力 |
| SimpleGDN | 67.03 (↓8.25) | 31.27 (↓4.12) | 简化 GDN |
| **DSA** | **~75** (≈0) | **~35** (≈0) | **无损** |

**结论**：DSA 是唯一无损的高效注意力方案，可应用于所有层。

---

## 七、国产 GPU 适配

GLM-5 从第一天起全栈适配国产 GPU 生态：

- **华为 Ascend**
- **摩尔线程 Moore Threads**
- **海光 Hygon**
- **寒武纪 Cambricon**
- **昆仑芯 Kunlunxin**
- **沐曦 MetaX**
- **燧原 Enflame**

覆盖从底层 kernel 到上层推理框架的深度优化。

---

## 八、GLM 技术演进

```
GLM-130B (2210) → 首个双语开源 LLM
    │
ChatGLM/GLM-4 (2406) → 消费级部署
    │
GLM-4.5 → ARC 能力统一 (Agentic+Reasoning+Coding)
    │
GLM-4.7 → 358B MoE, 改进的 Agent 能力
    │
GLM-5 (2602) → 744B/40B MoE, DSA, 异步 RL
    │
GLM-5.1 → 754B, 最新迭代
    │
GLM-5V-Turbo (2604) → 多模态 Agent, CogViT
```

---

## Related Pages

- [[llm/overview]]
- [[llm/06_infra/megatron-lm/overview]]
- [[zhipu_glm/glm_overview]]
