# Kimi/Moonshot AI 技术路线总览

> Kimi 是 Moonshot AI（月之暗面）开发的大语言模型系列，以长上下文能力和推理能力著称。
> 最后更新: 2026-07-31（kb-reorg P7 Task 7:目录内分段编号 + `kimi_k2.5_analysis` 消点号→`kimi_k2_5_analysis`）

---

## 页面列表(按层次)

> **段位与阅读顺序**(kb-reorg P7 Task 7,2026-07-31):段 1(10-19)核心机制主线——按发布时间交错排列旗舰模型与其奠基的注意力机制创新(MoBA→K2→Kimi Linear/KDA→K2.5→K3,与下方「技术演进时间线」一致);段 2(20-29)深潜/专题——K3 各切面的源码级/机制级深挖(KDA 机制与 kernel 实现、K3 架构/infra/后训练案例/稳定性/开源栈、MoonEP 专家负载均衡)。

| 页面 | 层次 | 核心主题 |
|------|------|---------|
| [[10_moba_analysis]] | 核心机制(段 1) | MoBA:混合块注意力,亚二次 $O(N \cdot B \cdot k)$,已生产部署 |
| [[11_kimi_k2_analysis]] | 核心机制(段 1) | Kimi K2:1T MoE Agentic 智能,Moonlight 架构 |
| [[12_kimi_linear_analysis]] | 核心机制(段 1) | Kimi Linear:KDA 线性注意力,3:1 KDA-MLA 混合,6x 解码加速 |
| [[13_kimi_k2_5_analysis]] | 核心机制(段 1) | Kimi K2.5:1.1T MoE,视觉 Agent 智能融合 |
| [[14_kimi_k3_analysis]] | 核心机制(段 1) | Kimi K3:2.78T MoE 旗舰,KDA+AttnRes+Stable LatentMoE,1M 上下文 |
| [[20_gdn_kda_linear_attention_analysis]] | 深潜(段 2) | Gated Delta Networks + KDA 机制深挖:从 QKVABZ 到 chunkwise 仿射扫描 |
| [[21_gdn_kda_kernel_implementation_analysis]] | 深潜(段 2) | GDN/KDA 训练与推理 Kernel 实现(FLA/SGLang) |
| [[22_kimi_k3_architecture_deepdive]] | 深潜(段 2) | K3 架构深挖:Attention Residuals、Stable LatentMoE、KDA:Gated MLA 混合比 |
| [[23_kimi_k3_infra_deepdive]] | 深潜(段 2) | K3 训练/推理基础设施深挖 |
| [[24_kimi_k3_posttraining_case_study_analysis]] | 深潜(段 2) | K3 后训练案例(原 D12):九专家 RL、MOPD、partial rollout、white-box environment、QAT、百万 token 状态管理 |
| [[25_kimi_k3_stability_analysis]] | 深潜(段 2) | K3 训练稳定性:七条失稳轴与被拒绝的替代方案 |
| [[26_kimi_k3_open_source_stack_analysis]] | 深潜(段 2) | K3 开源栈全景,逐仓证据等级核对 |
| [[27_moonep_analysis]] | 深潜(段 2) | MoonEP:动态冗余专家的全平衡 EP,源码级审计 |

---

## 一、Kimi 模型家族

| 模型 | 发布时间 | 参数量 | 核心能力 | arXiv |
|------|---------|--------|---------|-------|
| **Kimi Chat** | 2023.03 | - | 200K+ 上下文 | - |
| **k1.5** | 2025.01 | - | RL 缩放、长 CoT 推理 | 2501.12599 |
| **K2** | 2025.07 | 1T MoE | 开放 Agent 智能 | 2507.20534 |
| **K2.5** | 2026.02 | 1.1T MoE | 视觉 Agent 智能 | 2602.02276 |
| **K2.6** | 2026.04 | 1.1T MoE | 开源编码迭代 | 待发布 |
| **K3** | 2026.07 | **2.78T MoE**（104.2B 激活） | KDA+AttnRes+Stable LatentMoE；1M 上下文；原生视觉/视频 | Technical Report 2026-07-28 |

---

## 二、技术演进时间线

```
2023.03  Kimi Chat 发布
    │      └── 200K+ 上下文能力
    │
    ▼
2024.07  Mooncake (推理服务架构)
    │      └── KVCache 中心化分离架构
    │      └── Prefill/Decode 池分离
    │      └── 开源: github.com/kvcache-ai/Mooncake
    │
    ▼
2025.01  k1.5 (RL 缩放定律)
    │      └── 在线镜像下降变体 (类似 GRPO)
    │      └── 128K 上下文 RL 训练
    │      └── Long2Short 方法
    │      └── Partial Rollout + 混合部署
    │
    ▼
2025.02  MoBA (混合块注意力)
    │      └── 将 MoE 原理应用于注意力
    │      └── 1M 序列 6.5x 加速
    │      └── 已部署支持 Kimi 长上下文请求
    │
    ▼
2025.04  Kimi VL (视觉语言模型)
    │      └── 多模态理解
    │
    ▼
2025.04  Kimi Audio (音频模型)
    │      └── 音频理解和生成
    │
    ▼
2025.07  K2 (1T MoE Agent 模型)
    │      └── Moonlight 架构
    │      └── 强大推理和 Agent 能力
    │
    ▼
2025.10  Kimi Linear (线性注意力)
    │      └── KDA: Kimi Delta Attention
    │      └── 3:1 KDA-MLA 混合架构
    │      └── 1M 解码 6x 加速，KV Cache 减少 75%
    │
    ▼
2026.02  K2.5 (视觉 Agent 智能)
    │      └── 1.1T MoE 参数
    │      └── 视觉 + Agent 能力融合
    │
    ▼
2026.03  Attention Residuals (AttnRes 论文, 2603.15031)
    │      └── 残差流 → 深度方向 softmax attention
    │      └── 在 Kimi-Linear 48B + 1.4T tokens 验证 (等效 1.25× 算力)
    │
    ▼
2026.07  K3 (首个开源 3T 级旗舰)
           └── 2.8T MoE (Stable LatentMoE, 896 选 16)
           └── KDA : Gated MLA = 3:1 + AttnRes
           └── 1M 上下文、原生视觉/视频
           └── MXFP4 权重 + MXFP8 激活 (SFT 起 QAT)
           └── 三领域 × 三 effort RL experts → MOPD
           └── Partial rollout + 外部 KV 池 + 可恢复 AgentENV
           └── 相对 K2 约 2.5× scaling 效率
```

---

## 三、核心技术栈

### 3.1 注意力机制演进

```
标准 Softmax Attention (O(N²))
    │
    ├── MoBA (2025.02) → 亚二次 O(N·B·k)
    │      └── 将 MoE 原理应用于注意力
    │      └── 动态 top-k 块选择
    │      └── 保留 Softmax，兼容性强
    │
    └── Kimi Linear/KDA (2025.10) → 线性 O(T)
           └── 通道级细粒度遗忘门
           └── Delta Rule + DPLR 优化
           └── 3:1 KDA-MLA 混合架构
```

### 3.2 推理服务架构

```
Mooncake (2024.07)
├── Prefill Pool (计算优化)
├── Decode Pool (吞吐优化)
├── KVCache Pool (容量优化)
└── Conductor (全局调度)
     ├── 缓存感知调度
     ├── Layer-wise Prefill
     ├── CPP 分块流水线并行
     └── 预测性早期拒绝
```

### 3.3 RL 训练框架

```
k1.5 RL 框架 (2025.01)
├── 在线镜像下降变体
├── 128K 上下文扩展
├── 长度惩罚 (渐进式)
├── Partial Rollout
├── 混合部署 (Megatron ↔ vLLM)
└── Long2Short 蒸馏

K3 后训练闭环 (2026.07)
├── SFT agent cold start + XTML
├── 三领域 × low/high/max = 九个 RL experts
├── Reasoning Effort RL + Agentic GRM
├── Partial rollout: 达到 λNK 后暂停；同 prompt 的 K 条完成后 dispatch
├── MOPD: student on-policy token 上汇合九个 teachers
└── 1M infra: external KV pool / auto-throttling / AgentENV
```

K3 没有把 partial rollout 换成一套完全异步算法：它仍按迭代和组边界更新，只允许未完成轨迹跨迭代续跑。完整机制、公式、证据边界与源码缺口统一见 [[24_kimi_k3_posttraining_case_study_analysis|D12]]。

---

## 四、关键技术对比

### 4.1 注意力机制对比

| 特性 | MoBA | Kimi Linear (KDA) |
|------|------|-------------------|
| 复杂度 | 亚二次 $O(N \cdot B \cdot k)$ | 线性 $O(T)$ |
| 注意力形式 | 保留 Softmax，稀疏化 | 线性近似 + Delta Rule |
| 状态大小 | 完整 KV Cache (稀疏) | 固定 $d_k \times d_v$ |
| 适用场景 | 长上下文 prefill | 超长序列/Agent/流式 |
| 部署状态 | 已生产部署 | 开源 + vLLM 集成 |

### 4.2 RL 算法对比

| 特性 | Kimi k1.5 | Kimi K3 | DeepSeek-R1 | OpenAI o1 |
|------|-----------|---------|-------------|-----------|
| RL 主线 | 在线镜像下降变体 | 九专家 RL → MOPD | GRPO | 未公开 |
| 最大上下文 | 128K | 1M | 128K | 未公开 |
| 多模态 | ✅ | ✅ | ❌ | ❌ |
| Partial Rollout | ✅ | ✅，跨迭代续跑但保留 $K$-response completion/dispatch boundary | ❌ | 未公开 |
| 推理预算档 | 未统一为产品档 | low / high / max | 未公开 | 未公开 |

---

## 五、Kimi 技术路线全景图

```
┌─────────────────────────────────────────────────────────────────┐
│                        Kimi 技术全景                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  应用层: Kimi Chat / API / Agent                                │
│       │                                                         │
│       ▼                                                         │
│  模型层: K3 (2.78T MoE) ← K2.5 (1.1T MoE) ← K2 ← k1.5         │
│       │                                                         │
│       ▼                                                         │
│  注意力层:                                                       │
│       ├── MoBA (长上下文 Prefill 优化)                           │
│       ├── KDA (线性注意力，恒定状态)                             │
│       └── MLA (全局注意力，3:1 混合)                             │
│       │                                                         │
│       ▼                                                         │
│  训练层:                                                         │
│       ├── RL 框架 (在线镜像下降 + 长度惩罚)                       │
│       ├── Long2Short 蒸馏                                        │
│       ├── Partial Rollout                                        │
│       └── K3: 九专家分化 → MOPD 在线汇合                         │
│       │                                                         │
│       ▼                                                         │
│  推理层: Mooncake                                                │
│       ├── Prefill/Decode 池分离                                  │
│       ├── KVCache 分布式缓存                                     │
│       └── 缓存感知调度                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 六、关键论文索引

| 论文 | arXiv | Wiki 页面 |
|------|-------|----------|
| Mooncake: KVCache-centric Disaggregated Architecture | 2407.00079 | [[02_engineering/03_infer_frameworks/mooncake_analysis]] |
| MoBA: Mixture of Block Attention | 2502.13189 | [[10_moba_analysis]] |
| Kimi k1.5: Scaling RL with LLMs | 2501.12599 | [[01_theory/04_posttraining/29_kimi_k1_5_analysis]] |
| Kimi VL Technical Report | 2504.07491 | 待摄入 |
| Kimi Audio Technical Report | 2504.18425 | 待摄入 |
| Kimi K2: Open Agentic Intelligence | 2507.20534 | [[11_kimi_k2_analysis]] |
| Kimi Linear: Expressive Efficient Attention | 2510.26692 | [[12_kimi_linear_analysis]] |
| Gated Delta Networks + KDA 机制深挖 | 2412.06464 / 2510.26692 | [[20_gdn_kda_linear_attention_analysis]] |
| GDN/KDA 训练与推理 Kernel 实现 | FLA / SGLang | [[21_gdn_kda_kernel_implementation_analysis]] |
| Kimi K2.5: Visual Agentic Intelligence | 2602.02276 | [[13_kimi_k2_5_analysis]] |
| Attention Residuals | 2603.15031 | [[22_kimi_k3_architecture_deepdive]] §4(独立页待建) |
| Kimi K3: Open Frontier Intelligence Technical Report | 官方报告 2026-07-28 | [[14_kimi_k3_analysis]] · [[22_kimi_k3_architecture_deepdive]] · [[23_kimi_k3_infra_deepdive]] · [[25_kimi_k3_stability_analysis]] · [[24_kimi_k3_posttraining_case_study_analysis\|D12]] |
| K3 后训练案例（kb-reorg P5 迁入,原 D12） | 官方报告 2026-07-28 | [[24_kimi_k3_posttraining_case_study_analysis]] — 九专家 RL、MOPD、partial rollout、white-box environment、QAT 与百万 token 状态管理的统一综合案例 |
| K3 训练稳定性（横切报告 §2.3/§2.4/§2.5/§3.2/§3.3/§4.1.2/App. C） | 官方报告 2026-07-28 | [[25_kimi_k3_stability_analysis]] |
| K3 开源栈全景（逐仓 GitHub API 核对） | 仓库快照 2026-07-28 | [[26_kimi_k3_open_source_stack_analysis]] |
| MoonEP：动态冗余专家的全平衡 EP | `MoonEP@0f385f03` | [[27_moonep_analysis]] |

---

## 七、知识缺口

以下 raw 文件尚未摄入到 wiki：

- Kimi VL(2504.07491,待摄入) — 视觉语言模型技术报告
- Kimi Audio(2504.18425,待摄入) — 音频模型技术报告

仍待补齐或独立摄入的 Moonshot AI 材料：
- Kimi-Dev (2509.23045) — Agentless Training
- Kimina-Prover (2504.11354) — 形式推理
- Attention Residuals (2603.15031) — 机制/消融/开销已在 [[22_kimi_k3_architecture_deepdive]] §4 深度覆盖(2026-07-17,含源码核查);独立 `attnres_analysis` 页待建
- Kimi K3 技术报告已于 2026-07-28 摄入；尚缺核心 RL trainer/rollout 源码、训练与生产拓扑、关键超参数及隔离消融，统一追踪在 [[24_kimi_k3_posttraining_case_study_analysis|D12]]
- K3 开源栈已于 2026-07-28 逐仓核对（见 [[26_kimi_k3_open_source_stack_analysis]]）：**MoonEP 已做源码级审计**（[[27_moonep_analysis]]），AgentENV 升至 README 级。**仍无任何开源实现的**：K3 backbone、trainer/rollout、以及 Gated MLA（含 512 head-dim）与 AttnRes 两类 kernel
- AgentENV（`kvcache-ai/AgentENV`）待做 commit 级代码审计；minitriton / nano-kpu 待实际复跑（当前仅按 README 记录，属 K3 能力证物而非基础设施）
- K3 各稳定性组件的**隔离消融**（RMSNorm / SiTU / QB / Per-Head Muon 各自贡献）报告未给；K3 也**没有** K2 那样的“零 loss spike”级训练稳定性陈述，边界见 [[25_kimi_k3_stability_analysis]] §4
- G1 (2505.13426) — VLM 感知 + RL
- WorldVQA (2602.02537) — 多模态评测
- Pixel-Level VLM (2601.19228) — 像素级感知

---

## Related Pages

- [[01_theory/index]]
- [[01_theory/01_models/attention_is_all_you_need_analysis]]
- [[01_theory/04_posttraining/20_grpo_analysis]]
- [[24_kimi_k3_posttraining_case_study_analysis]] — K3 后训练算法、1M Agentic RL infra 与源码边界
- [[02_engineering/02_train_frameworks/megatron-lm/index]]
- [[20_gdn_kda_linear_attention_analysis]] — 从 QKVABZ 到 chunkwise 仿射扫描
- [[21_gdn_kda_kernel_implementation_analysis]] — 训练、Prefill、Decode 融合实现
- [[25_kimi_k3_stability_analysis]] — K3 稳定性栈：七条失稳轴与被拒绝的替代方案
- [[26_kimi_k3_open_source_stack_analysis]] — K3 开源栈全景与各仓证据等级
- [[27_moonep_analysis]] — MoonEP 源码级审计（全平衡 EP）
