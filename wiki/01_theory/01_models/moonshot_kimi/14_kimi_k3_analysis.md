# Kimi K3：以约 2.5× 的整体扩展效率迈向 3T 级开放模型

> **来源基线**：官方 Kimi K3 Tech Blog（2026-07-16）与 [Kimi K3 Technical Report `0797decb`](https://github.com/MoonshotAI/Kimi-K3/commit/0797decb18ab079de86f991b87a64b81ec15a3c2)（2026-07-28）；模型权重固定到 [Hugging Face `9f62e4e9`](https://huggingface.co/moonshotai/Kimi-K3/tree/9f62e4e9fffbd0a83ddd60e1c209d828994b3569)。
> - 一手来源：[kimi.com/blog/kimi-k3](https://www.kimi.com/blog/kimi-k3)，本地博客快照为 `raw/01_theory/01_models/moonshot_kimi/Kimi_K3_blog_2026-07-16.txt`；报告本地快照为 `raw/01_theory/01_models/moonshot_kimi/Kimi_K3_Technical_Report_2026-07-28.pdf`；官方内嵌架构图原件为 `assets/kimi_k3_official_arch.svg`。
> - 结构组件的论文与源码证据：见 [[22_kimi_k3_architecture_deepdive]]。
> - 训练与推理基础设施：见 [[23_kimi_k3_infra_deepdive]]。
> **维度**：模型发布总览。本页回答“K3 是什么、表现如何、官方如何定位”；机制层面的“为什么”由两篇 deep dive 展开。
> **更新**：2026-07-28，已回填正式报告中的模型结构、后训练与 Infra 事实；后训练统一案例见 [[24_kimi_k3_posttraining_case_study_analysis]]。

> [!deprecated]
> 本页 2026-07-17 初版的“技术报告与权重尚未发布”状态已被 2026-07-28 官方报告和权重取代。仍未披露的训练超参数与源码边界保留在第七节。

---

## 一、主线

**K3 的主线不只是“把模型做大”，而是提高单位算力所能换取的能力。** 官方将 KDA、AttnRes、高稀疏 Stable LatentMoE，以及训练和数据配方的共同收益，概括为相对 Kimi K2 **约 2.5× 的整体 scaling efficiency**（报告 §3.2、Fig. 7，pp.10–11；博客 `:104-105`）。

这里的“2.5×”不是某一项 benchmark 的提升，也不是单纯的训练吞吐倍率。正式报告把它画成 held-out OOD validation loss 对训练 FLOPs 的拟合曲线，但没有给出各结构、数据和训练配方的隔离归因；因此不能把它外推为系统吞吐、训练成本或任一单组件的倍率。

2.8T 总参数可以理解为这套效率改造进一步规模化后的结果。官方也没有回避差距：K3 的整体表现仍落后于 Claude Fable 5 和 GPT 5.6 Sol，但在其评测套件中持续领先其他被测模型（博客 `:76-77`）；Limitations 又进一步承认，实际用户体验与这两款闭源模型之间仍有可感知差距（博客 `:552-559`）。

## 二、关键规格（架构数值对照 K2，产品能力对照 K2.5）

| 维度 | Kimi K3 | K2 / K2.5 基线（按行注明） | 出处 |
|---|---|---|---|
| 总参数 | **2.78T**（约 2.8T） | K2：1.04T | 报告 Table 1，p.11 |
| 激活参数 | **104.2B** | K2：32.6B | 报告 Table 1，p.11 |
| Backbone 层数 | **93**（69 KDA + 24 Gated MLA） | K2：61 MLA | 报告 Table 1，p.11 |
| MoE | **Stable LatentMoE,896 专家选 16**(稀疏度 56) | K2：384 选 8 + 1 共享(稀疏度 48) | 报告 §2.3，pp.6–8；Table 1，p.11 |
| 注意力 | **KDA : Gated MLA = 3:1 混合** | K2：纯 MLA(61 层) | 报告 §2.1，pp.3–6；Table 1，p.11 |
| 跨层连接 | **AttnRes**(深度方向 softmax 注意力) | K2：标准残差 | 报告 §2.2，p.6；Table 1，p.11 |
| 上下文 | **1M 原生** | K2.5：256K | 报告 §3.4，p.12；[[13_kimi_k2_5_analysis]] §1.1 |
| 模态 | 原生文本/图像/**视频**同模型 | K2.5：原生视觉(MoonViT-3D) | 报告 §2.4，pp.9–10；[[13_kimi_k2_5_analysis]] §1.3 |
| 量化 | **MXFP4 routed-expert 权重 + MXFP8 输入激活，SFT/RL 全程 QAT** | K2.5：INT4 post-training QAT | 报告 §4.1.4，p.14；[[13_kimi_k2_5_analysis]] |
| 训练稳定性组件 | Per-Head Muon、Quantile Balancing、SiTU | K2：MuonClip | 报告 §2.3、§2.5，pp.6–11；[[11_kimi_k2_analysis]] |
| 思考模式 | `low/high/max` effort；**preserved thinking history** | K2.5：thinking + instant 双模式 | 报告 §4.1.2，p.13；Appendix F，p.47；[[13_kimi_k2_5_analysis]] |
| API 定价 | 输入 3.00 USD/MTok；缓存命中输入 0.30 USD/MTok；输出 15.00 USD/MTok | — | 博客 §Availability（`:223-227`） |

**整体结构图**（官方博客内嵌架构图**原图**：SVG 原件 + 官方页面 CSS 按暗色主题渲染，渲染 wrapper 存于 `assets/kimi_k3_official_arch_render.html`，可复现）：

![Kimi K3 官方架构图：右侧主干自下而上为 Embedding → Block n−3/n−2/n−1 → 当前块，块内重复单元为 KDA→(+)→Stable LatentMoE→(+) ×3 加 Gated MLA→(+)→Stable LatentMoE ×1；每个子层既有常规残差 (+)，又有一个 (w, α) 单元从历史块与 Embedding 做深度注意力取回，网络最终 Output 前还有一次 (w, α) 聚合。左上放大面板为 Stable LatentMoE：输入经 Router 与降维 Linear（梯形）分发给 Shared Expert（绿）与 Routed Expert（蓝紫），聚合后经升维 Linear 与 Norm 相加输出。左下放大面板为 KDA：q/k 走 Linear→Conv→L2，v 走 Linear→Conv，另有低秩瓶颈（梯形对）产生细粒度遗忘门与 σ 门、以及低秩 σ 输出门与 Norm 后的逐元素乘。](assets/kimi_k3_official_arch.png)

> 官方内嵌 SVG 的 aria-label 直接写着 **"Block Attention Residuals architecture diagram"**——AttnRes 在官方叙事中的地位可见一斑。

## 三、基准结果

K3 的所有结果都使用 `reasoning effort=max`、`temperature=1.0`、`top-p=1.0`。不同 benchmark 分别运行在 KimiCode、Claude Code 或 Codex harness 下，因此分数不仅反映基础模型，也包含 agent harness 的影响（博客 §Footnotes，`:495-539`）。主要对手包括 Claude Fable 5（max，可能 fallback）、GPT 5.6 Sol（max）、Claude Opus 4.8（max）、GPT 5.5（xhigh）和 GLM-5.2（max）。

![官方 Coding 基准：K3 在 Program Bench 和 SWE Marathon 排名第一，在 Terminal Bench 2.1、FrontierSWE 与 Kimi Code Bench 2.0 排名第二。](assets/kimi_k3_bench_coding.png)

![官方 General 与 Visual Agents 基准：K3 在 Automation Bench、SpreadsheetBench 2 和 BrowseComp 排名第一，多项知识工作与视觉任务进入前三。](assets/kimi_k3_bench_agents.png)

在博客给出的 33 项完整评测中，K3 的第一名项目包括：SWE Marathon 42.0、Program Bench 77.8、BrowseComp 91.2、DeepSearchQA 95.0（F1）、Automation Bench 30.8、SpreadsheetBench 2 34.8、OmniDocBench 91.1，以及与 Fable 5 并列的 ZeroBench 23.0。

**分域画像**：

- **长程任务是最突出的优势。** SWE Marathon 得分 42.0，明显高于 Fable 5 的 35.0；FrontierSWE 得分 81.2，仅次于 Fable 5 的 86.6，并高于 GPT 5.6 Sol 的 71.3。这与官方“训练特别强调长程高难任务”的说明一致。
- **检索与浏览能力进入第一梯队。** BrowseComp 得分 91.2。脚注还披露：使用 1M 上下文且完全不做 context management 时，K3 仍能得到 90.4；采用 300K token 触发的上下文压缩策略后为 91.2（博客 `:538-539`）。这说明 1M 原生窗口至少让“无需压缩直接浏览”成为可行基线。
- **经典编码仍略逊于最强闭源模型。** DeepSWE 为 67.5，低于 GPT 5.6 Sol 的 73.0 和 Fable 5 的 70.0。
- **知识推理强弱并存。** GPQA-Diamond 达到 93.5，但 HLE-Full 为 43.5，与 Fable 5 的 53.3 仍有明显差距。
- **视觉任务整体较强。** OmniDocBench 以 91.1 排名第一；MathVision 在不使用和使用 Python 时分别为 94.3、97.8，接近第一梯队。

**读表时需要保留四项口径限制。** Fable 5 的结果来自第三方评测，且可能包含 fallback；在 PostTrain Bench 中，如果 Fable 5 拒答，评测会回退到 Opus 4.8；GLM-5.2 以及部分 Claude/GPT 分数引用自各家官方页面或 Artificial Analysis，并非 Moonshot 自测；DeepSWE 官方榜单使用 mini-SWE-agent harness 时，K3 得分为 67.3，而不是上图中的 67.5（博客 §Footnotes）。

## 四、能力案例（官方自报，尚未经第三方复现）

1. **Kernel 优化。** 官方设置了四项任务，覆盖 AttnRes、KDA 和 512 head-dimension MLA kernel，并分别在 H200 与“另一厂商 GPGPU”上运行；每个模型获得独立沙箱和 24 小时时间。K3 与 Fable 5 表现接近，并明显优于 Opus 4.8 和 GPT 5.6 Sol。博客还称，K3 开发后期的大部分 kernel 优化由早期版 K3 自己完成。任务本身也从侧面揭示了 K3 kernel 栈的关键组成（博客 §Kernel Optimization，`:164-169`）。
2. **GPU 编译器。** K3 从零构建 MiniTriton：以 MLIR 为基础设计 tile-level IR，并实现 PTX code generation。官方报告其 roofline 微基准可达到或超过 Triton、`torch.compile`，且能够端到端训练 nanoGPT 并正常收敛（§GPU Compiler Development）。
3. **芯片设计。** 在 48 小时自主运行中，K3 使用开源 EDA 与 Nangate 45 nm 工艺库，为运行自家架构的 nano 模型设计芯片。官方给出的结果包括 4 mm² 面积、100 MHz 时序收敛、146 万个标准单元、0.277 MB SRAM，以及带融合反量化的 INT4 MAC；仿真解码吞吐为 8,700 token/s（§Chip Design）。
4. **科研复现。** 在天体物理 I–Love–Q 关系任务中，K3 交叉核验 20 多篇论文和 300 多个物态方程，产出 3,000 多行代码。官方称约两小时完成了资深研究员通常需要一至两周的工作（§Coding for Research）。
5. **知识工作与视频编辑。** K3 生成了一份覆盖 42 年 ASIC 行业史的交互式报告，经历 120 多轮递归改进和 2,800 多次检索；另一个案例把 56 份素材自动剪成 teaser，官方估计相当于熟练剪辑师一至两个工作日的工作量（§Knowledge Work、§Video Editing）。

## 五、使用限制（官方 Limitations）

1. **对 thinking history 敏感。** K3 使用 preserved thinking history 模式训练。如果 harness 没有完整回传历史思考内容，或者会话中途从其他模型切换到 K3，生成质量可能变得高度不稳定。官方建议使用经过验证的 harness（如 Kimi Code），并避免中途切换模型。
2. **可能过度主动。** 面对小问题或含糊意图时，K3 可能替用户做出超出预期的决定。部署方需要在 system prompt 或 `AGENTS.md` 中明确权限与行为边界。
3. **用户体验仍有差距。** 官方承认，K3 与 Fable 5、GPT 5.6 Sol 之间仍存在可感知的 UX 差距（博客 `:552-559`）。

## 六、发布与获取

- 渠道：kimi.com、Kimi Work（3.1.0 及以上）、Kimi Code（通过 `/model` 选择 `kimi-k3`）和 Kimi API（模型名 `kimi-k3`）。
- 价格：缓存命中输入为 0.30 USD/MTok，未命中输入为 3.00 USD/MTok，输出为 15.00 USD/MTok。官方 API 由 Mooncake 的分离式推理架构支撑，coding 负载的缓存命中率超过 90%（博客 `:223-227`）。若按恰好 90% 命中估算，平均输入成本上界约为 `0.9 × 0.30 + 0.1 × 3.00 = 0.57 USD/MTok`；机制见 [[23_kimi_k3_infra_deepdive]] §3。
- 时间线：官方技术报告与完整模型权重均已发布；报告固定于 2026-07-28 的 `0797decb`，权重快照固定于 Hugging Face `9f62e4e9`。K3 的 effort-conditioned 后训练覆盖 `low/high/max`（报告 §4.1.2，p.13；Appendix F，p.47）。

## 七、技术报告已确认与仍未披露的边界

报告已确认 104.2B 激活参数、93 层、7,168 hidden dimension、69 KDA + 24 Gated MLA、MoonViT-V2、Per-Head Muon、Quantile Balancing、SiTU-GLU 和 Stable LatentMoE 的机制主体（报告 §2、§3，pp.3–12；Table 1，p.11）。

后训练则明确为 SFT → 三领域 × 三 effort 的九个 RL experts → MOPD，并公开 partial rollout、reasoning budget、Agentic GRM、white-box environments、MXFP4/MXFP8 QAT、draft model 和 1M Agentic RL Infra（报告 §4.1–4.2，pp.12–16；§5.3，pp.21–22；Appendix F，pp.46–47）。完整机制统一放在 [[24_kimi_k3_posttraining_case_study_analysis|D12]]，避免在模型目录再形成一套割裂的 RL 资料。

仍未披露的关键项包括：RL 的 \(N,K,\lambda,\tau,\sigma,R_{\max}\)、总 RL FLOPs、逐 token stale-data regularizer 的完整公式、GRM 训练细节、MOPD 隔离消融、训练/rollout/weight-sync 源码，以及若干 Infra 指标的硬件和分位数条件。现有 `[推断]` 必须逐项对照正式报告；报告仍未给出的内容不能自动升级为事实。

## Related Pages

- [[22_kimi_k3_architecture_deepdive]] — 按“动机 → 机制 → 证据 → 替代方案”分析 KDA、Gated MLA、AttnRes、Stable LatentMoE、SiTU 与规模变化
- [[23_kimi_k3_infra_deepdive]] — Per-Head Muon、MXFP4/MXFP8 QAT、全平衡 EP、Mooncake、KDA prefix cache 进 vLLM、64+ 卡超节点
- [[25_kimi_k3_stability_analysis]] — 稳定性栈横切：七条失稳轴、七处被拒绝的替代方案，以及“K3 没有 K2 那样的零 spike 陈述”这条边界
- [[26_kimi_k3_open_source_stack_analysis]] — 随发布开放的仓库全景（含“FlashKDA 并非本次新开源”的更正）与各仓证据等级
- [[27_moonep_analysis]] — MoonEP 源码级审计：报告 §5.2.1 的七条说法逐条兑现
- [[24_kimi_k3_posttraining_case_study_analysis]] — 九专家 RL、MOPD、partial rollout、white-box environment 与 1M Agentic RL Infra 的统一案例
- [[12_kimi_linear_analysis]] — KDA 与 3:1 混合架构的原始论文(K3 注意力主干的前身)
- [[13_kimi_k2_5_analysis]] — 直接前代(1.04T MoE + MoonViT 原生视觉)
- [[11_kimi_k2_analysis]] — 2.5× scaling 效率宣称的对照基线(MuonClip、15.5T tokens)
- [[10_moba_analysis]] — Moonshot 更早的长上下文注意力路线(MoBA,已被 KDA 路线接替)
- [[moonshot_kimi/index]] — Kimi/Moonshot 技术路线总览
