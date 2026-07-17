# Kimi K3 — 用「结构效率 ×2.5」逼近闭源前沿的首个开源 3T 级模型

> **来源基线**(2026-07-16 发布,**完整技术报告与权重尚未放出**,将于 2026-07-27 前随权重发布):
> - 一手来源:官方发布博客 [kimi.com/blog/kimi-k3](https://www.kimi.com/blog/kimi-k3),本地快照 `raw/01_theory/01_models/moonshot_kimi/Kimi_K3_blog_2026-07-16.txt`(逐字稿,下称"博客");官方内嵌架构图原件 `assets/kimi_k3_official_arch.svg`
> - 结构组件的机制/源码证据:见 [[kimi_k3_architecture_deepdive]](KDA/AttnRes/LatentMoE 逐项溯源)
> - 训推 infra:见 [[kimi_k3_infra_deepdive]]
> **维度**: Entity 总览(发布报告级)。本页回答"K3 是什么、表现如何、官方怎么定位";机制级"为什么"在两个 deepdive 页。
> **更新**: 2026-07-17 初版(基于发布博客;技术报告发布后需回填激活参数、层数、训练配方等缺口)

---

## 一、主线

**K3 的主线不是"更大",而是"每单位算力换更多智能"**:官方将 KDA + AttnRes + 高稀疏 Stable LatentMoE 三项结构改动加上训练/数据配方,合并量化为——

> "these structural changes yield an approximate **2.5× improvement in overall scaling efficiency compared to Kimi K2**, allowing the model to convert compute into intelligence more effectively"(博客 §An Open 3T-Class Model)

2.8T 总参数是这个效率红利之上的规模化结果,而非卖点本身。官方同时罕见地自报位次:"整体性能仍落后于最强闭源模型 Claude Fable 5 和 GPT 5.6 Sol,但稳定超过其余被测模型"(博客开篇原句),Limitations 里再次承认 UX 差距。

## 二、关键规格(vs 直接前代 K2.5)

| 维度 | Kimi K3 | Kimi K2.5(对照) | 出处 |
|---|---|---|---|
| 总参数 | **2.8T**(首个开源 3T 级) | 1.04T | 博客开篇;[[kimi_k2.5_analysis]] §1.1 |
| 激活参数 | **未披露**(待技术报告) | 32B | 博客(无此数) |
| MoE | **Stable LatentMoE,896 专家选 16**(稀疏度 56) | 384 选 8 + 1 共享(稀疏度 48) | 博客 §Architecture;K2.5 README |
| 注意力 | **KDA : Gated MLA = 3:1 混合** | 纯 MLA(61 层) | 博客架构图 3×/1× 标注 |
| 跨层连接 | **AttnRes**(深度方向 softmax 注意力) | 标准残差 | 博客 §Architecture |
| 上下文 | **1M 原生** | 256K | 博客开篇 |
| 模态 | 原生文本/图像/**视频**同模型 | 原生视觉(MoonViT 400M) | 博客 §Video Editing |
| 量化 | **MXFP4 权重 + MXFP8 激活,SFT 起 QAT** | 原生 INT4(post-training QAT) | 博客 §Architecture |
| 训练稳定性组件 | Per-Head Muon、Quantile Balancing、SiTU | MuonClip | 博客 §Architecture;[[kimi_k2_analysis]] |
| 思考模式 | 默认 max effort;**preserved thinking history 模式训练** | thinking + instant 双模式 | 博客开篇、§Limitations |
| API 定价 | 输入 $3.00/M(缓存命中 $0.30/M)、输出 $15.00/M | — | 博客 §Availability |

**整体结构图**(按官方博客内嵌架构图重绘,原件见 assets):

![Kimi K3 整体结构:中间主干为重复单元 KDA→Stable LatentMoE ×3 + Gated MLA→Stable LatentMoE ×1,每个子层带 AttnRes 的 (α,w) 深度注意力系数,从 Block n−1/n−2/n−3 与 Embedding 选择性取回表征;左放大面板为 KDA 内部(q/k/v 各自 Linear+Conv,q/k 加 L2 归一,σ 门 α_t/β_t,输出门),右放大面板为 Stable LatentMoE(Linear 降维→Router→896 选 16 + Shared Expert→Linear 升维)](assets/kimi_k3_arch_redrawn.png)

> 官方内嵌 SVG 的 aria-label 直接写着 **"Block Attention Residuals architecture diagram"**——AttnRes 在官方叙事中的地位可见一斑。

## 三、基准结果

评测口径:K3 全部 reasoning effort=max、temperature=1.0、top-p=1.0;按基准分别用 KimiCode / Claude Code / Codex harness(博客 §Footnotes)。对手:Claude Fable 5(max, with fallback)、GPT 5.6 Sol(max)、Claude Opus 4.8(max)、GPT 5.5(xhigh)、GLM-5.2(max)。

![官方 Coding 基准图:DeepSWE 67.5(第3)、Terminal Bench 2.1 88.3(第2)、FrontierSWE 81.2(第2)、Program Bench 77.8(第1)、Kimi Code Bench 2.0 72.9(第2)、SWE Marathon 42.0(第1,Fable 5 仅 35.0)](assets/kimi_k3_bench_coding.png)

![官方 General/Visual Agents 基准图:GDPval-AA v2 Elo 1668(第3)、AA-Briefcase Elo 1548(第2)、Automation Bench 30.8(第1)、JobBench 52.9(第2)、SpreadsheetBench 2 34.8(第1)、BrowseComp 91.2(第1)、CharXiv w/tool 91.3(第2)、Zerobench w/tool 41.0(并列第2)](assets/kimi_k3_bench_agents.png)

**K3 登顶项**(全表 33 项,博客 §Full Benchmark Table):SWE Marathon 42.0、Program Bench 77.8、BrowseComp 91.2、DeepSearchQA 95.0(F1)、Automation Bench 30.8、SpreadsheetBench 2 34.8、OmniDocBench 91.1、ZeroBench 23.0(与 Fable 5 并列)。

**分域画像**:

- **长程任务是最大亮点**:SWE Marathon 42.0 全场第一(Fable 5 35.0);FrontierSWE 81.2 仅次于 Fable 5(86.6)、大幅超 GPT 5.6 Sol(71.3)。与其"训练特别强调长程高难任务"的自述一致(§Limitations)。
- **检索/浏览第一梯队**:BrowseComp 91.2 全场最高;脚注披露**用 1M 上下文、完全不做 context 管理时 90.4**(300K 触发 compaction 反而 91.2)——1M 原生上下文让"无压缩暴力浏览"成为可行基线(博客 §Footnotes/BrowseComp)。
- **经典编码略逊半档**:DeepSWE 67.5 vs GPT 5.6 Sol 73.0 / Fable 5 70.0。
- **知识推理**:GPQA-Diamond 93.5;HLE-Full 43.5 与 Fable 5(53.3)差距明显。
- **视觉**:OmniDocBench 91.1 第一;MathVision 94.3/97.8(裸考/带 python)接近第一梯队。

**读表注意**(博客 §Footnotes):Fable 5 由第三方评测、"结果可能含 fallback";PostTrain Bench 中 Fable 5 拒答自动回落 Opus 4.8 作答;GLM-5.2/部分 Claude/GPT 分数引自各家官方页与 artificialanalysis.ai 而非自测;DeepSWE 官方榜单口径下 K3 为 67.3(mini-SWE-agent harness)。

## 四、能力叙事(官方案例,未经第三方复现)

1. **Kernel 优化**:四项任务(AttnRes、KDA、512 head-dim MLA kernel)跨 H200 与"另一厂商 GPGPU",每模型独立沙箱 24h;K3 与 Fable 5 相当、明显超 Opus 4.8/GPT 5.6 Sol;并披露"K3 开发后期,团队大部分 kernel 优化由早期版 K3 完成"(博客 §Kernel Optimization)。**任务选材本身泄露了 K3 自家 kernel 栈的组成**。
2. **GPU 编译器**:从零构建 MiniTriton(tile 级 IR over MLIR + PTX codegen),roofline 微基准持平或超 Triton/torch.compile,端到端撑起 nanoGPT 训练收敛(§GPU Compiler Development)。
3. **芯片设计**:48h 自主运行,开源 EDA + Nangate 45nm,为"跑自家架构 nano 模型"设计芯片:4mm²、100MHz 时序收敛、1.46M 标准单元、0.277MB SRAM、INT4 MAC + 融合反量化,仿真 8,700 tokens/s 解码(§Chip Design)。
4. **科研复现**:天体物理 I–Love–Q 关系,交叉验证 20+ 论文、300+ 物态方程、3,000+ 行代码,约 2h 完成"资深研究员 1-2 周"工作量(§Coding for Research)。
5. **知识工作/视频**:42 年 ASIC 行业交互式报告(120+ 轮递归自改进、2.8k+ 检索);56 素材自剪 teaser(≈熟练剪辑师 1-2 个工作日)(§Knowledge Work、§Video Editing)。

## 五、使用限制(官方 Limitations,部署方必读)

1. **对 thinking history 敏感**:K3 以 preserved thinking history 模式训练——harness 不完整回传历史思考内容、或会话中途从其他模型切到 K3,"生成质量可能高度不稳定";建议用验证过的 harness(如 Kimi Code)、避免中途切换。
2. **过度主动**:遇小问题/意图含糊可能替用户做预期外决定;需在 system prompt 或 AGENTS.md 显式约束。
3. **UX 差距自认**:与 Fable 5 / GPT 5.6 Sol 相比仍有可感知差距。

## 六、发布与获取

- 渠道:kimi.com、Kimi Work(≥3.1.0)、Kimi Code(`/model` 选 kimi-k3)、Kimi API(`kimi-k3`)。
- 价格:$0.30/M(缓存命中输入)/ $3.00/M(未命中输入)/ $15.00/M(输出);官方 API 由 Mooncake 驱动,**coding 负载缓存命中率 >90%** ⇒ 有效输入价 ≈$0.57/M(博客 §Availability;机制见 [[kimi_k3_infra_deepdive]] §3)。
- 时间线:权重 2026-07-27 前 + 技术报告同步;低/高思考档后续推出。

## 七、待技术报告回填的缺口

激活参数量、总层数、隐藏维度、视觉塔细节、预训练 token 数、Per-Head Muon/Quantile Balancing/SiTU/Stable LatentMoE 的精确定义与消融、RL 配方。**本页与两个 deepdive 页中所有标注 [推断] 的内容都应在报告发布后核对。**

## Related Pages

- [[kimi_k3_architecture_deepdive]] — 六大结构变化点逐项:动机→机制→证据→为何不选替代(KDA/Gated MLA/AttnRes/Stable LatentMoE/SiTU/规模)
- [[kimi_k3_infra_deepdive]] — Per-Head Muon、MXFP4/MXFP8 QAT、全平衡 EP、Mooncake、KDA prefix cache 进 vLLM、64+ 卡超节点
- [[kimi_linear_analysis]] — KDA 与 3:1 混合架构的原始论文(K3 注意力主干的前身)
- [[kimi_k2.5_analysis]] — 直接前代(1.04T MoE + MoonViT 原生视觉)
- [[kimi_k2_analysis]] — 2.5× scaling 效率宣称的对照基线(MuonClip、15.5T tokens)
- [[moba_analysis]] — Moonshot 更早的长上下文注意力路线(MoBA,已被 KDA 路线接替)
- [[moonshot_kimi/index]] — Kimi/Moonshot 技术路线总览
