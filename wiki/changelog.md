# Knowledge Base Changelog

All source ingestions and significant wiki updates are logged here.

---

## 2026-07-06: [[unbacked_symint_analysis]] 增补 §10 —— unbacked 处理的最新进展（`guard_size_oblivious` → 显式 size-oblivious 原语族）

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

## 2026-07-03: [[inductor_memory_allocation_guide]] 新增 §5「内存越界/踩踏排查」— 补第二份社区材料的缺口(纠错版)

**Type**: Enrich（应用户"第二份专家社区材料的第三部分——内存踩踏检测——库里缺,补上,一定忠于事实"。经 gap 分析:材料前两部分已被 [[inductor_memory_management_analysis]] 覆盖且更准,仅第三部分是真缺口;材料本身有错,按源码收敛后再入库）

**源（source-faithful）**：pytorch @ `5f6df46744a` 逐行核对——`config.py:232-237`（`size_asserts`/`nan_asserts`/`scalar_asserts` 默认值）、`codegen/wrapper.py:1793-1827`（`assert_size_stride` 生成 + 延迟到首个 kernel 前）、`ir.py:7817/7845/9152`、`utils.py:161`（`GPU_ALIGN_BYTES=16`）、`codegen/triton.py:5458`（`mask=xindex<xnumel`）、`codegen/common.py:2789`（核内 NaN 断言）。

- **guide 新增 §5 内存越界/踩踏排查**（原 §5/§6 顺延为 §6/§7）：§5.1 自动核内置防护（Triton `mask` + `assert_size_stride`/`assert_alignment`/scalar·nan_asserts,多为默认 ON）· §5.2 真正越界来源（自定义算子/手写 Triton/错误 stride·offset/unbacked symint）· §5.3 工具（`compute-sanitizer` 取代 cuda-memcheck + `CUDA_LAUNCH_BLOCKING`）· §5.4 排查步骤。
- [!correction] **订正该社区材料**（源 > 材料，均已核实）：① 「Inductor 对越界无内置保护」**错**——mask + size/alignment/scalar 断言多为默认 ON;② 「规划池用 `_cuda_beginAllocateToPool` 申请」**错**——该 API 全库仅在 `cudagraph_trees.py`（CUDA Graphs 私有池）,规划池是普通 `empty_strided`;③ `cuda-memcheck` 已被 `compute-sanitizer` 取代。**材料对的部分**（保留）：16 字节对齐确有其事（`GPU_ALIGN_BYTES=16`）。前两部分（池初始化/复用）不收录——深挖页已覆盖且更源忠实。

**整合**：[[04_inductor/index]] guide 条目补「越界/踩踏排查」;guide Related 增 [[inductor_gpu_kernel_dispatch_model]]/[[Pytorch_Compile_Debug_Analysis]]/[[unbacked_symint_analysis]] 回链。**校验**：所有 `file:line`/常量值本会话开文件核对;新增段无 mermaid。

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

**整合**：[[01_theory/01_models/index]] 新增「LongCat / Meituan」家族区；[[index]]（总索引）模型行 28→30、加「LongCat (美团)」子行与「按主题查找」条目、更新日期至 2026-07-02；两新页与 [[glm_5_analysis]]/[[deepseek_v3_analysis]]/[[deepseek_v4_analysis]]/[[kimi_k2_analysis]]/[[muon_analysis]]/[[expert_parallel_analysis]] 等互链。**校验**：全用 ASIC——图表用 **ASCII**（与 GLM-5/Kimi 同系列风格，零 mermaid 定界符风险）；跨链目标经 grep 核对；因源为渲染提取，数值保真度与未披露项已在页头/§9 显式声明，不臆造未披露量。

---

## 2026-07-01: 新建 [[06_distributed_parallelism/index]] 分布式并行原理簇 —— 原语→DP→TP/SP/CP→EP→PP→ZeRO 全景（理论层）

**Type**: New（应用户"在 01_theory 加分布式并行原理解读，从分布式原语→TP→EP→PP→ZeRO 等基本概念；演示图用 SVG→PNG"。抓本质 + 引擎无关的原理层，与已有工程页分工）

**定位**：新建理论簇 `01_theory/06_distributed_parallelism/`，**原理（principle）层、引擎无关**——只讲「为什么这么切、代价函数长什么样、为什么不选替代」，两根主线贯穿全簇：**$\alpha$-$\beta$ 通信代价模型** + **显存账本（参数/梯度/优化器态/激活）**；「源码怎么实现」一律交叉链接到 [[../02_engineering/index]] 已有的源级页（[[15_distributed_primitives/index]]、[[megatron-lm/index]]、[[torchtitan/index]] 等），不重复。填补「理论层无分布式并行原理页」的空白。

- **新增 index + 6 内容页**：
  - [[collectives_analysis]] — 六大原语语义、$\alpha$-$\beta(-\gamma)$ 模型、核心恒等式 **all-reduce = reduce-scatter + all-gather**、ring 每卡搬运 $2(N{-}1)/N\cdot M$ 的带宽最优性、ring vs tree、all-to-all/p2p 代价（全簇「代价词汇表」）。
  - [[data_parallel_analysis]] — DP：复制模型/切数据、all-reduce 梯度的等价性、通信 $\propto\Psi$ 与 batch/卡数无关、$16\Psi$ 显存账本（引出 ZeRO）、分桶重叠 + 梯度累积。
  - [[zero_fsdp_analysis]] — ZeRO 1/2/3 逐级切优化器态/梯度/参数、通信 vs DP 增量（1/2 免费、3 多 ~50% AG）、ZeRO-3 = FSDP 的 unshard→compute→reshard。
  - [[tensor_sequence_parallel_analysis]] — TP（Megatron 列切→行切 + f/g 共轭算子、每层 4 次 all-reduce、只敢机内）、SP（拆 all-reduce 为 RS+AG，零额外通信换激活显存）、CP（ring-attention 交换 KV 攻长序列）。
  - [[expert_parallel_analysis]] — EP：路由 + 两次 all-to-all（分发/回收）、负载不均与容量因子、分层 a2a。
  - [[pipeline_parallel_analysis]] — PP：microbatching、气泡率 $(P{-}1)/(m{+}P{-}1)$、GPipe vs 1F1B（同气泡、显存 $\propto m$ vs $\propto P$）vs interleaved（真降气泡）、zero-bubble。
- **演示图 9 张 SVG→PNG**（手绘 HTML+SVG，走 `.html2md/render_figs.mjs` 无头 Edge 2× 截图）：六原语语义、ring all-reduce 分解、DP 数据流、Megatron 列/行切+f/g、TP+SP 激活切分、ring-attention、EP 三段 a2a+负载不均、GPipe/1F1B 甘特气泡对比、ZeRO 0/1/2/3 显存分区、N 维正交布局（DP2×PP2×TP4）。原理演示图统一走 SVG（按用户约定：代码调用/类/逻辑图才用 mermaid）。

**工具改动**：`render_figs.mjs` 加 `FIGS_OUT` 环境变量支持自定义输出目录（默认仍指 GLM assets，向后兼容），本簇渲染到 `06_distributed_parallelism/assets/`。

**整合**：[[01_theory/index]] 子领域表加「06 分布式并行原理」一行；[[15_distributed_primitives/index]] 与 [[06_auto_parallel/index]] 各加回链（理论↔实现互指）。**校验**：9 张 PNG 逐张实渲肉眼核对（SVG 经 Edge 所见即所得，天然规避 mermaid 定界符坑）；本簇内 `[[链接]]` 与指向工程页的跨域链接经 grep/文件核对存在。

---

## 2026-06-30: 新建 [[inductor_memory_allocation_guide]] + 深挖页补「池大小如何确定」—— 吸收外部专家报告

**Type**: Ingest + Enrich（应用户"把外部报告 `deep-research-report.md` 的原理分析风格吸收进库 + 回答 pool 初始化大小如何确定 + 补例子/演示图"。源忠实 + 抓本质）

**源（source-faithful）**：pytorch @ `5f6df46744a`，逐行核对 `codegen/memory_planning.py`（`AllocationPool`/`get_symbolic_size`/`allocate_at_end`/`codegen_create`）、`c10/core/AllocatorConfig.h:16-24`（段大小常量）、`c10/cuda/CUDACachingAllocator.cpp:3063/3697`（`round_size`/`get_allocation_size`）、`codegen/wrapper.py:1520`（`alloc_from_pool`=`torch.ops.inductor._alloc_from_pool`）、`torch/csrc/inductor/inductor_ops.cpp:36/129`、`test/inductor/test_memory_planning.py:108-142`（真实 codegen 实例）。

- **深挖页 [[inductor_memory_management_analysis]] 新增**:
  - **§2.6 池的初始化大小如何确定**:Inductor `AllocationPool` 大小=编译期 `root.get_symbolic_size()`（`TemporalSplit` 取最大、`SpatialSplit`=`align(left)+right`）、`allocate_at_end` 末尾追加扩容、`codegen_create` 出扁平 1-D buffer;带 `test_memory_planning.py` 真实实例（`pool1 = empty_strided_cuda((4*s27*s77 + align(4*s77*s77),),(1,))` + 两个 `alloc_from_pool`）+ **字节布局 ASCII 图**。
  - **§3 物理段大小**:`empty_strided` 落 `CUDACachingAllocator` 后按 `get_allocation_size` 取段——≤1 MiB→2 MiB、1–10 MiB→20 MiB、≥10 MiB→2 MiB 倍数（`AllocatorConfig.h:16-24`），解释 `reserved` 远大于 `allocated` 的原因。
- **新建 guide [[inductor_memory_allocation_guide]]**（吸收报告骨架：角色边界→分配全过程 sequence 图→分配器对照表→`memory_stats`/snapshot 实测复现→实践建议）。
  - [!correction] **订正原报告 4 处**（源 > 报告）：① 池化 `memory_planning` 非默认、仅 inference（默认是逐 buffer 复用）;② 实验开关应 `memory_planning=True` 而非 `memory_efficient_fusion`;③ 数分配次数用 `allocation.all.allocated`/`segment.all.allocated` 而非 `allocation.all.current`;④ `expandable_segments` 是 native 子开关、非独立后端。报告对 `alloc_from_pool` 的描述确认正确。

**整合**：[[04_inductor/index]] 新增两页条目;两页互相回链;深挖页 §2.6/§3 增补。**校验**：所有新 `file:line`/常量值本会话开文件核对（含 `AllocatorConfig.h:16-24` 数值、`test_memory_planning.py:108` 的 `@config.patch(memory_planning=True)`）；guide 的 1 个 sequenceDiagram 按规范扫（消息无 `[]`/`()`/`|`，participant 用英文 id + alias）；字节布局用 ASCII 非 mermaid。

---

## 2026-06-30: 新建 [[inductor_memory_management_analysis]] — torch.compile 内存分配管理(全栈三层)

**Type**: New（应用户提问"torch.compile 的 memory alloc 管理怎么做"→评估知识库覆盖发现"零件散在 3 个域、无统一脊柱、cudagraph_trees 与 codegen 复用链是短板"→开新页补齐。源忠实 + 抓本质）

**源（source-faithful）**：pytorch 本地 checkout @ `5f6df46744a`（trunk, 2026-06-29）。两个并行 writer-agent 分别深挖**编译期**（`codegen/wrapper.py`·`_inductor/memory.py`·`codegen/memory_planning.py`·`config.py`）与 **CUDA Graphs**（`cudagraph_trees.py`·`cudagraph_utils.py`·`compile_fx.py`），每条 `file:line` 开文件核对；coordinator 抽检 `wrapper.py:2480`、`memory.py:1016`、`config.py:252-268`、`cudagraph_trees.py:2301-2302` 全部吻合。

- **新增** [[inductor_memory_management_analysis]]：主线"三层叠加"——
  - **层 1 编译期**：默认 `memory_plan_reuse` 两遍把 `Allocate`+`Free` 改写成 `Reuse`（峰值感知 `should_reuse_buffer`；同形状指针别名 / 异形状 `reinterpret_tensor`，`wrapper.py:2436/956/4043`）；scheduler `compute_last_usage`+`free_buffers` 决定释放时机（`scheduler.py:8731/8742`）；`reorder_for_peak_memory` 多拓扑序选最低峰值（`memory.py:1016`，扫描线估峰）；可选池化 `MemoryPlanner` 时分/空分打包（`memory_planning.py:675`，`memory_planning` 默认关）。
  - **层 2 运行期**：`empty_strided` 落 `CUDACachingAllocator` block/segment 缓存池（复用既有深页 [[caching_allocator_autocast_profiler_analysis]]，强调"编译期逻辑复用 + 运行期物理复用叠加"）。
  - **层 3 CUDA Graphs**：`cudagraph_trees` 跨图共享 `graph_pool_handle` 私有池 + 地址稳定（static/managed idx，`:1019/1932`）+ checkpoint 重建分配器簿记（`:3135`）+ graph partition 切出 cudagraph-unsafe 算子（`scheduler.py:8856`）。
  - [!correction] 据 `5f6df46744a` **订正 [[scheduler_analysis]] 两处行号**：`reorder_for_peak_memory` 实定义在 `memory.py:1016`（非 `scheduler.py:2986`）；`mutation_renames` 在 `scheduler.py:4197/4770`（非 `:2913-2928`）——符号名对、行号随版本漂移；并澄清 `memory.py` 是"区间+扫描线估峰驱动重排"而非区间图着色（着色在 `memory_planning.py`，默认关）。

**整合**：[[04_inductor/index]] 概览区新增本页；[[PyTorch_Inductor_Technical_Analysis]]（§6/§7 概念版）、[[caching_allocator_autocast_profiler_analysis]]（层 2）各加回链；本页另链 [[scheduler_analysis]]/[[inductor_codegen_analysis]]/[[control_flow_capture_analysis]]。**校验**：3 个 mermaid 块按本库规范逐条扫（subgraph 标题无 `[]`/`|`、各 `end` 单独闭合、连线标签无引号/括号/`|`、节点标签无裸 `[]()`）；交叉链接目标经 grep 确认存在。

---

## 2026-06-30: 新建 [[control_flow_capture_analysis]] — Dynamo 控制流捕获两条路径(HOP 投机子图 vs 原生字节码特化)

**Type**: New（应用户提问"torch.compile 编译流程里 cond 入图怎么做的" → 追问"是否覆盖所有控制流入图情况" → "总结一个章节专门介绍控制流"。源忠实 + 抓本质）

**源（source-faithful）**：pytorch 本地 checkout @ `5f6df46744a`（trunk, 2026-06-29），逐一开文件核对 `torch/_dynamo/variables/higher_order_ops.py`、`torch/_dynamo/symbolic_convert.py`、`torch/_higher_order_ops/cond.py`、`torch/_inductor/ir.py` 的引用行。

- **新增** [[control_flow_capture_analysis]]（02_dynamo deep dive）：核心论点——Dynamo 对控制流有**两条互不桥接的路径**。
  - **路径 A 显式 HOP**：`speculate_subgraph`（`higher_order_ops.py:2004`）统一引擎四步（开子 tracer→内联→freevar lifting→收尾）；`cond` 深挖（常量谓词特化短路 `:2419`、checkpoint/rollback 投机两分支 `:2475-2552`、`_merge_graph_inputs` 合并签名 `:1287`、`_ALLOW_FALLBACK_TO_EAGER=False` 禁 graph break `:2378`）；控制流 HOP 家族表（cond/switch/while_loop/map/scan/associative_scan，子图结构均经投机/install 锚点核对）；下游 dispatch（`cond.py:403/408/710` Proxy/Fake/functionalize + `ir.py:10700` `Conditional`）。
  - **路径 B 原生控制流**：`generic_jump`（`symbolic_convert.py:714`）四种结局（常量拍平/SymBool guard 特化/数据依赖切图/`fullgraph` 硬报错）；`FOR_ITER`（`:2485`）循环展开。
  - [!correction] **纠正常见误解**：Dynamo **不会**自动把数据依赖 `if` 转成 `cond`——源码里只有"切图"或"报错提示手写 `torch.cond`"两条出路（`symbolic_convert.py:769`/`:937`）。

**整合**：[[02_dynamo/index]] 页面列表新增本页；[[PyTorch_Dynamo_Technical_Analysis]] Related Pages 加回链。**校验**：所有 `file:line` 均本会话内开文件核对；3 个 mermaid 块按本库规范逐条扫（标签无裸 `[]()`、特殊形状无嵌套定界符、连线文字无引号/括号/`|`）；交叉链接目标经 glob 确认存在。

**追加（同日，应用户连续追问澄清编译期/运行期边界）**：
- [[PyTorch_Dynamo_Technical_Analysis]] §6.6「动态控制流」加 `> [!deprecated]` 指引转向本页（原演示内容按 never-delete 保留）。
- 本页新增 **§2.5「trace 两支 / 编译两支 / 运行只跑一支」**：拆解三个常见误解——① 「捕获条件」是把 `pred` 接成 cond 节点运行时输入（`pred.as_proxy()` `:2588`），非 trace 期选支；② 「Dynamo 编译两个子图」不准——Dynamo 只 trace、产 **1 张父图**（两子图为嵌套 `GraphModule`），编译成 kernel 是下游 Inductor 一次编译产两段；③ 按 pred 选支是 cond 算子 lowering 在**运行期**做（`cond_op_dense` `cond.py:310-313`）。附 `cond` vs graph-break 六维对照表。新增锚点 `cond.py:301-313`、`higher_order_ops.py:2588` 均本会话开文件核对。

---

## 2026-06-29: 新建投机推理专题 [[speculative_decoding/index]] — DSpark 论文 + DeepSpec 开源仓 + MTP→DFlash→DSpark 演进

**Type**: Ingest（应用户"分析 dspark 论文原理 + 结合开源 dspark 仓 + 总览投机推理演进 mtp/dflash/dspark 区别，归纳入库"。源忠实 + 抓本质）

> [!correction] **arXiv 编号订正（源 > 转述）**：用户给的 **arXiv:2606.19348 经核对是 DeepSeek-V4 模型论文**（本库已审计，见 [[deepseek_v4_audit_report]]），**不是 DSpark**。DSpark 是挂在 V4 checkpoint 上的投机解码草稿模块，其论文以 `DSpark_paper.pdf` 随开源仓 **`github.com/deepseek-ai/DeepSpec`** 发布（标题 *DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation*，Cheng et al., 北大+DeepSeek-AI）。HF 模型卡 `DeepSeek-V4-Pro-DSpark` 引用 2606.19348 指的是**底座模型**。

**源（source-faithful）**：克隆 `DeepSpec` @ `dd854392`（main, 2026-06-28）到 `E:\97-codes\torch_parallel\DeepSpec`，PDF 抽成页码标记文本逐页核对；论文公式与代码 `file:line` 双向交叉核对（Eq.5 ↔ `markov_head.py:8`、Eq.7 ↔ `modeling.py:268/293`、Eq.8 ↔ `loss.py:69`、Eq.2-3 ↔ `modeling.py:241/104-113`）。

- **新建目录** `wiki/02_engineering/03_infer_frameworks/speculative_decoding/`（3 页）：
  - [[index]]（本专题总览/演进survey）：投机解码在 `L=(T_draft+T_verify)/τ` 上的三代演进——① 自回归（MTP/Eagle3，升 τ 但 T_draft∝γ）→ ② 并行（Medusa/DFlash，降 T_draft 但后缀崩塌）→ ③ DSpark（半自回归补 τ + 置信度调度降有效 T_verify）。四代横向对比表 + 三者区别本质。
  - [[dspark_analysis]]（论文深挖，exemplar）：两大部件——**半自回归生成**（并行 DFlash 骨干 + Markov/RNN 串行头，Eq.4-6；接受长度相对 Eagle3 +30.9%、DFlash +16.3%）+ **置信度调度验证**（置信头 Eq.7-8 + STS 校准 + 硬件感知前缀调度器 Alg.1，按 SPS 负载曲线全局贪心、早停保无偏）。生产相对 MTP-1 提速 60%–85%（V4-Flash）/57%–78%（V4-Pro）。
  - [[deepspec_codebase_analysis]]（源码级）：一套 `Qwen3DSparkTrainer` 同产三草稿模型——**DFlash = DSpark 关掉串行/置信头的消融**（`config/dflash/*:18-26`，无独立 modeling）；训练前向链、三项损失 ↔ Eq.9-12、推理拒绝采样路径。**关键边界**：开源仓只到「置信头 + 静态阈值裁剪 + bsz=1」，Algorithm 1 多请求调度器/异步 ZOS/变长内核是生产专属。

**整合**：[[03_infer_frameworks/index]] 新增"投机推理"子目录；[[vllm_speculative_decoding_analysis]]（已含 dflash/mtp proposer）加 [[dspark_analysis]] 回链；[[deepseek_v3_analysis]]（MTP 起源）、[[deepseek_v4_analysis]]（底座模型）各加回链。**校验**：三页所有 `file:line` 已逐一开文件核对；交叉链接经 grep 确认目标存在。

---

## 2026-06-29: 新建 [[megatron_linear_cross_entropy_analysis]] — 融合线性交叉熵("chunk loss")源码级深挖

**Type**: New（应用户提问"当前 Megatron 的 chunk loss 如何实现",读 Megatron-LM `dev@232c478d4` 源码后沉淀,带具体源码实现分析）

- **新增** [[megatron_linear_cross_entropy_analysis]]:Megatron 版"chunk loss" = `cross_entropy_fusion_impl='linear'` 融合线性 CE。
  - **配置三档**(`model_parallel_config.py:257,262`):`native`/`te` 接收**已物化的 logits**只融 softmax+NLL(`language_module.py:157,180`);**`linear`** 把 LM-head matmul 也融进核、logits 从不物化。
  - **选路**:`gpt_model.py:157-160` 算 `fuse_linear_cross_entropy`,`:263` 把输出层换成 `LinearCrossEntropyModule`,`:799-802` 以 `output_cross_entropy_loss=True` 直接吐 loss。
  - **省显存本质**(`fused_linear_cross_entropy.py:161-181/197-223`):`save_for_backward` 只存 `hidden + max + sum-exp`(各 O(N))、**不存 logits**,反向从统计量按块重算 → 峰值 `O(N·V)→O(N·d)+O(N)`。
  - **Blackwell 融合核**(`linear_cross_entropy/blackwell/`):`entry.py:147-151` 按 `vocab_per_split=512` 切 `num_splits` 块、逐块 online-softmax(`fwd_mainloop.py:40-53`),`:246/253` 跨 TP `all_reduce(MAX/SUM)`;**硬件门控仅算力 10.x**(`:34-40` 非 Blackwell `raise`)——[!warning] 标注。
  - **对照** MindSpeed `chunk_loss`(序列维框架层 autograd,可移植 NPU)vs Megatron `linear`(词表维 kernel 融合,绑 Blackwell);同属 Flash-Attention 式"online-softmax + 不物化大矩阵 + 反向重算"。

**整合**:[[megatron-lm/index]] 专题深挖区(融合算子项下)新增条目;[[mindspeed_memory_optimization_analysis]] §8 chunk-loss 增"跨框架对照"回链。**校验**:`model_parallel_config.py:257/262`、`gpt_model.py:157-160/263/799-802`、`fused_linear_cross_entropy.py:34-40/161-181`、`blackwell/entry.py:147-151/246/253`、`language_module.py:157/180` 均逐一开文件核对;交叉链接 4 个目标经 find 确认存在。

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
- **整合**：更新 `05_gpu_kernel/index.md`（新增 Triton 表）、`wiki/index.md`（GPU Kernel 计数 1→10 + Triton 子条目 + 主题导航）；交叉链向 [[gpu_kernel_guide]]/[[triton_vs_mlir_backend_analysis]]/[[inductor_codegen_analysis]]/[[inductor_autotuning_analysis]]/[[flex_attention_analysis]]。
- **校验**：9 页内部 `[[链接]]` 全部互指存在页；外链目标均已存在；**0 悬挂链**。

---

## 2026-06-25: 清理 `Megatron-LM_Distributed_Parallel_Exam` 遗留悬挂链(11 处/9 文件)

**Type**: Maintenance（该页早先已删除、内容分发至各分析页;全库尚残留 11 处指向它的悬挂链,逐条按主题重指到真实后继页）

- **重指(按内容去向,非一键)**：
  - 重计算/卸载/resharding 类 → [[megatron_recompute_analysis]]（activation_checkpointing 的 Q12/Q13/Q30）
  - FP8 / CUDA Graph 类 → [[megatron_precision_cudagraph_fusion_analysis]]（low_precision Q15、transformer_engine Q14/Q15）
  - Muon/Layer-Wise 优化器 → [[megatron_distributed_optimizer_analysis]]（muon_analysis）
  - 泛 5D 并行综合 → [[megatron_parallelism_orchestration_analysis]]（distributed_optimizer/memory_optimization/moe_training/tflops 的「相关页面」、`wiki/index.md`「Megatron 分布式」导航）
- **散文出处改写**：low_precision §4.1、transformer_engine §10 的「来自 `...Exam.md` Q19/Q14：」改为中性引导句（删去已失效的源文件名,内容保留）。
- **校验**：全 wiki **0 处**仍指向 `Megatron-LM_Distributed_Parallel_Exam`（changelog 4 条历史记录按惯例保留）；5 个后继目标页均存在,**未引入新 dangling link**。

---

## 2026-06-25: DeepSeek-V4 工程页(TP/CP)整理入 megatron-lm/ + 双向交叉链接 + 源码审核

**Type**: Reorg + Audit（应用户"把 02_train 下重复的 deepseek-v4 内容合并/挪过来,模型分析放一起,重复删除、不重复挪,并审核内容"）

**核查结论(先判定再动)**：`02_train_frameworks/` 下两篇 V4 页面经核查**不是重复**——它们是**真实的 Megatron-LM 源码级框架分析**（每节带 `megatron/core/...py:line`，与本地 `../Megatron-LM` 源码核对通过），与论文级模型页 [[deepseek_v4_cp_analysis]] **角度互补**（论文*算法* ↔ 框架*实现*）；TP 页在模型侧无对应。故**无可删的真重复**；缺的是两侧**互相没有交叉链接**。按 wiki「理论/工程」分层 + 用户选定「留工程层 + 双向交叉链接」执行：

- **移库（`git mv` 保留历史）**：`deepseek_v4_tensor_parallel_analysis.md`、`deepseek_v4_context_parallel_analysis.md` + 7 张图 `assets/deepseek_v4_*_fig*.png` 从 `02_train_frameworks/` 移入 `02_train_frameworks/megatron-lm/`（与其余 Megatron 页同处；`assets/` 相对引用保持有效）。
- **补基线头**：两页原**无 commit 基线**，补 `源基线: Megatron-LM dev @ 232c478d4 (2026-06-16)` + 维度(工程实现) + 与模型页分工说明。
- **补 `## 相关页面`**：两页原**缺交叉引用节**（违反 wiki 规则），补全——双向链接模型页（[[deepseek_v4_cp_analysis]]/[[deepseek_v4_analysis]]/[[deepseek_v4_technical_deep_dive]]/[[mHC]]）+ 同目录 Megatron 页。
- **模型页反向链接**：[[deepseek_v4_cp_analysis]] 的「相关页面」新增「框架实现」小节，指向两篇工程页（论文算法 ↔ Megatron 实现对照）。
- **索引**：`02_train_frameworks/index.md` 删去两条目（已移子目录）；`megatron-lm/index.md`「专题深挖」表新增两条目（带跨目录指回模型页）。
- **内容审核（对照 `../Megatron-LM` @ 232c478d4 抽查）**：✅ `deepseek_v4_hybrid_attention.py:92` `get_pg_size(tp)==1`、`:447` `parallel_mode='duplicated'`；✅ `hyper_connection.py:193/243` mHC 用 `nn.Linear`+`sequence_parallel`（非 TP-sharded）；✅ `experts.py:346` routed-expert `tp_group.size()>1` 约束；✅ `shared_experts.py:123` 标准 TP；✅ §九 特征5「CSA+CP 两阶段压缩 KV all-gather 尚未实现」仍成立（`csa.py` 的 `cp_group` 仅用于 RoPE CP 感知，page §5.1 已记，**非**压缩 KV 通信）。结论：内容真实、源码 grounded、非臆造非重复；仅**行号随源码漂移数行**（已在页头标注）。

---

## 2026-06-25: DeepSeek-V4 全系列对正式发表版 arXiv:2606.19348v1 审计 / 核对 / 订正

**Type**: Reconciliation（应用户"分析 deepseek v4 输出一份报告，文章地址 arxiv 2606.19348"，并选定「审计 + 核对 + 订正」。既有 ~7 篇 V4 页面是论文正式上 arXiv 前 2 天(2026-04-24)基于无编号预发布 PDF / AI 合成笔记写成，故以正式版逐项复核）

**方法**：下载正式版 PDF → 与本地预发布 `raw/.../DeepSeek_V4.pdf` 双双抽成页码标记文本 → diff + 逐项核对超参/基准/章节/机制；4 个并行只读审计 agent 锚定 `GROUND_TRUTH` 事实表逐页查证，关键臆造断言由协调者亲自 grep 正式版复核（`DualPath`=0、`Highly Compressed`=0 vs `Heavily`=7、`task_classifier`=0、`n log n`=0、`INT8`=0 vs `MXFP4`=1、`ablation`=0）。

- **新增** [[deepseek_v4_audit_report]]：审计报告（核对基线 arXiv:2606.19348v1, 2026-04-26）——逐页裁决表 + 核对通过事实(超参/效率/基准全一致) + 章节号位移映射 + 论文中**不存在**的臆造清单(按出处反证)。
- **核对通过(数字全对)**：超参表(层/维/专家/压缩率)、头条效率(Pro 27%/10%、Flash 10%/7%)、Table 1 基座(MMLU-Pro 65.5/68.3/73.5)、Table 6 后训练(LiveCodeBench 93.5、MRCR 83.5)。
- **订正(Tier-A，论文真源页面)**：
  - [[deepseek_v4_analysis]] —— 重订基线头；FP4 标注为**后训练 §5.2.1**(两组件 FP4 + 索引分数 BF16)；正文臆造的「DualPath 推理框架」加 `> [!contradiction]` 标注；评测表「顶尖」措辞按 Table 6 校准为有基线的相对表述；「51×/2048×」标注为本页推导。
  - [[deepseek_v4_cp_analysis]] —— **章节号位移订正**(CP §3.5.3→§3.4.3、推理框架 §3.6/§3.6.1/§3.6.2→§3.5/§3.5.1/§3.5.2、Muon §3.4.1、mHC §3.4.2，共 ~13 处)；重订基线头；footer 旧路径修正。
  - [[deepseek_v4_fp4_qat_analysis]] —— 出处 §3.7(两版皆无)订正为 **§5.2.1**；旧路径 `raw/05_model_families/`→`raw/01_theory/01_models/`；「三个组件 FP4」订正为「2 组件 FP4 + 索引分数 BF16」；效果表 ~75%/~2× 标注为估算(论文仅明述 top-k 2×、99.7% 召回)。
  - [[mHC]] —— 基本一致；补注消融数据源自 mHC 论文(arXiv:2512.24880v2)而非 V4。
- **加警示横幅，建议整页重写(Tier-B，预发布 AI 生成、无任何论文引用，确认系统性臆造)**：
  - [[deepseek_v4_technical_deep_dive]] —— DSA=CSA+HCA 倒置、Highly/Heavily 名错、HCA 10% vs m′=128、臆造分层调度、臆造 MoE 任务路由、DualPath、MLA「V3 引入」(应 V2)、Sinkhorn 100(应 20)。
  - [[deepseek_v4_implementation_details]] —— 专家 128/激活 8(应 256·384/6)、HCA 0.1、Sinkhorn 100、「Muon」实为 Adam(无 Newton-Schulz)、量化写成 INT8(应 FP4)、臆造 PCA KV 压缩/DualPath。
  - [[deepseek_v4_architecture_diagrams]] —— 128 专家/K=8–12、5%·35% 任务自适应、领域命名专家、O(n log n)、DualPath/SNIC/CNIC、图缺 MTP、CSA/HCA 误并为一层。
- **Tier-A 遵循 wiki「不删除、仅标注」规则**：臆造内容保留原文 + `> [!warning]`/`> [!contradiction]` 标注 + 指向审计报告。
- **Tier-B 整页重写完成**（用户确认"基于正式版全部重写 3 篇"后执行）：3 个并行 writer-agent 锚定核验过的机制简报
  （`MECH_BRIEF`：CSA/HCA/DSA Eq 9–27、Muon Alg 1/Eq 28、mHC Eq 1–8、§4.2.1 配置）逐方程重写，每条断言带 §/Eq/page；
  协调者抽查定位符 + 机械校验（横幅已除、`DualPath`/`INT8`/`Highly`/`task_classifier`/`n log n` 清零、跨链无悬挂、图内无 `[[…]]` 泄漏）。
  - [[deepseek_v4_technical_deep_dive]]（288 行）—— CSA/HCA/DSA/MLA 四机制「动机→机制(LaTeX)→证据→为何不选替代」对比。
  - [[deepseek_v4_implementation_details]]（362 行）—— 五大组件逐方程伪代码，常量取自 §4.2.1，Flash|Pro 双值。
  - [[deepseek_v4_architecture_diagrams]]（237 行）—— 复刻 Figure 2/3/4 + §4.2.1 配置表，含 MTP、首-2-层非对称、CSA/HCA 交错。
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

- [[glm5_training_infra_deepdive]] §3.3 新增 **图 3**：计算-通信掩盖时间线——②双缓冲(累积‖梯度同步)、③Muon(本地计算‖分片all-gather)、④激活offload(计算‖搬运)、⑥延迟wgrad(填气泡)、⑦层级all-to-all(节点内‖节点间)，各自把"什么藏进计算"。
- [[glm5_agentic_rl_deepdive]] 新增 **图 1**（PD 解耦时间线：混部 prefill 抢占 decode vs 解耦后 decode 连续）+ **图 3**（DP-aware routing 的 KV 前缀复用：朴素 O(总上下文) vs 一致性哈希亲和 O(增量)）；原图 1/2 顺延为 **图 2/4** 以保持阅读序。
- [[glm5_low_precision_chip_deepdive]] §4.3 新增 **图 3**：昇腾掩盖与缓存——左 Lightning Indexer/MLAPO(Vector‖Cube)/异步调度(D2H‖decode准备)/FlashComm(拆AllReduce) 计算掩盖访存通信，右 RadixCache 前缀共享 + Prefix Cache KV 外溢到系统内存。
- **工具链**：`figstyle.css` 增加时间线(`.tl`/`.tl-bar`)与内存层(`.tier`)样式；新增图源 `glm5_infra_overlap` / `glm5_agentic_cache` / `glm5_chip_overlap_cache`.html（gitignored），渲染 5 张 PNG 到 `assets/`。**校验**：5 图逐张肉眼查无溢出/残留链接语法；agentic 页图号顺延后阅读序连续。

---

## 2026-06-24: [[glm5_architecture_deepdive]] 补完整模型结构图（§1.1，config 实据）

**Type**: Update（应用户"architecture 里缺一张完整模型结构图"。拉取 released `zai-org/GLM-5` config.json 作实据，新增"宏观层栈 + 单 MoE 解码层放大"的结构图 + config 超参表 + 层数 contradiction）

- 新增 §1.1「完整模型结构（GlmMoeDsa）」+ 图 `assets/glm5_architecture_fig3.png`：左=宏观栈(Embedding→Dense×3→MoE×75→Final RMSNorm→LM Head+MTP)，右=单层放大(子层A DSA 注意力：MLA-256 低秩+Muon Split→lightning indexer top-2048→稀疏注意力；子层B MoE：Router→top-8/256+1 共享→加权合并)。
- 超参全部取自 released config：hidden 6,144 · **78 层**(前 3 dense + 75 MoE) · qk/v head_dim 256 · kv_lora 512+rope 64=**576** · 256 专家 top-8 + 1 共享 · DSA index_topk 2,048 · MTP×1。
- `> [!contradiction]`：论文 §2.1 称 **80** 层、开源权重 **78** 层，以权重为准；原 DSA 续训图 caption 顺延为「图 3」。

---

## 2026-06-24: GLM-5 论文逐章深挖补齐 6 篇 + 流程图工具链 + 索引整合

**Type**: New + Deepen（应用户"针对每个章节做 deepdive，解释原理/效果/为什么，并补流程图(SVG→PNG)"。在 [[glm5_architecture_deepdive]] 校准页之上，6 个并行 writer-agent 各写一篇深挖页 + 各自流程图 HTML，coordinator 统一渲染 14 图并整合）

- **新增 6 篇深挖页**（源基线 arXiv 2602.15763v2，逐节 原理/效果/为什么 + §/页码引用）：
  - [[glm5_data_deepdive]] — §2.2–2.3 数据（双分类器漏斗 + 三段式上下文扩展，2 图）
  - [[glm5_training_infra_deepdive]] — §2.4 显存五件套 + 长序列并行（2 图）
  - [[glm5_posttraining_deepdive]] — §3.1–3.5 SFT(三思考模式)/GRPO+IcePop/General RL/跨阶段蒸馏（2 图）
  - [[glm5_agentic_rl_deepdive]] — §3.6+§4 slime/全异步解耦 RL/三类环境构造（2 图）
  - [[glm5_training_stability_deepdive]] — 跨章稳定性主线（失配×噪声×故障：TITO/双边IS/staleness/优化器reset/确定性topk，2 图）
  - [[glm5_low_precision_chip_deepdive]] — §2.4.3+§3.6.2+§5 INT4 QAT→FP8→W4A8 + 昇腾三支柱（2 图）
- **流程图工具链**：新增 `.html2md/render_figs.mjs`（复用 Edge/puppeteer 2× 截图）+ `figs/figstyle.css`；图源 HTML 在 gitignored `.html2md/figs/`，14 张 PNG 落 `assets/`（house 风格:奶白卡片 + 彩色圆角节点 + 灰箭头）。
- **整合**：父索引 [[zhipu_glm/index]] 新增「§四之补 GLM-5 论文深挖页矩阵」(7 页表) + §六 GLM-5 行改指矩阵；概要页 [[glm_5_analysis]] 补「逐章深挖」Related 段，并对 §五 估算基准加 `> [!contradiction]` 用 Table 7 真值订正（SWE-bench Verified 77.8 / τ²-Bench 89.7 / AA Index 50 等）。
- **校验**：7 页 + 索引/概要的 `[[]]` 链接脚本提取，同系列 7 个 `glm5_*_deepdive` + 既有 [[muon_analysis]]/[[grpo_analysis]]/[[megatron_ep_analysis]]/[[verl/index]]/[[low_precision_training_analysis]] 等均存在，0 悬空；14 图 `assets/*` 引用解析正常；agentic_rl 两图 note 内误写的 `[[]]` 已改纯文本并重渲。

---

## 2026-06-24: 新增 [[glm5_architecture_deepdive]] 并入 GLM 索引

**Type**: New（GLM-5 架构深挖页:论文 §2.1 的"规模 × 长上下文成本"权衡——744B/40B MoE 扩专家减层、MLA→Muon Split→MLA-256、MTP 参数共享、DSA 两阶段续训与高效注意力消融,含 2 图）

**整合**:父索引 [[zhipu_glm/index]] §六 论文索引 GLM-5 行新增架构深挖链接(概要 [[glm_5_analysis]] + 架构深挖 [[glm5_architecture_deepdive]])。**校验**:2 张图 `assets/glm5_architecture_fig{1,2}.png` 引用解析正常;`## Related` 段含同系列深挖页前向引用(6 个 `glm5_*_deepdive` 为规划中页面,标记式前向链接)+ 既有页([[glm_5_analysis]]/[[muon_analysis]]/[[deepseek_v3_analysis]]/[[deepseek_moe_analysis]])均存在。

---

## 2026-06-23: [[megatron_ddp_optimizer_analysis]] 新增 §2.7「bucketing 算法与 overlap 调度」(机制级深挖)

**Type**: Update（应用户提问"Megatron distributed optimizer 如何 bucket、如何调计算与 bucket 让计算 overlap 掉参数通信",现读 Megatron-LM `dev@232c478d4` 源码后补全;既有 §2.1–2.3 只到高层轮廓）

- **新增** [[megatron_ddp_optimizer_analysis]] §2.7,补三件源码层细节:
  - **分桶算法**:逆序贪心 `_compute_default_per_buffer_param_layout`(`param_and_grad_buffer.py:891-939`)—— `params[::-1]` 逆序(backprop 序,末层落 bucket 0)、累计 ≥ bucket_size 即封桶;三级结构 Buffer / Bucket / BucketGroup(后者为一次 NCCL collective 粒度,`_coalescing_manager` 合并)。
  - **bucket_size 调参**:默认 `max(40M, 1M·dp_size)`(`distributed_data_parallel.py:68-69`)、ring 每 rank 报文 = `bucket_size/dp_size`(`..._config.py:61`)、distopt 可分片约束 `numel % dp == 0`(`param_and_grad_buffer.py:1059`)、`pad_buckets_for_high_nccl_busbw` 凑 2^16。
  - **双向 overlap 的 hook 调度**:反向 backward-post-hook → `register_grad_ready` golden-count 满才发异步 RS(`param_and_grad_buffer.py:802-824`);前向 forward-pre-hook(`distributed_data_parallel.py:413`)→ `finish_param_sync` wait 本组 AG + 预取 `next_param_gather_bucket_group`(`param_and_grad_buffer.py:496/:531`),链按前向序串于 `distributed_data_parallel.py:295-308`。附理想时间线 ASCII 图 + 调节点表(bucket_size / 桶序=执行序 / align_param_gather / 头尾暴露)。
  - **补澄清(应用户追问"register_grad_ready 是否先填桶再统一通信")**:`register_grad_ready` 是**就绪计数器而非填数据**——填数据是同一 hook 内前一步 `param.main_grad.add_(param.grad.data)`(`distributed_data_parallel.py:469`,`main_grad` 是扁平 buffer 的视图,梯度原地累加,无"搬进桶"动作);"桶满"= 该组成员梯度全算齐(`per_param == golden`,`param_and_grad_buffer.py:822`),非攒够字节(桶大小/成员初始化即定死);golden count 可 >1(参数被多次消费,首 batch 记录,`:273-276`)。已并入 §2.7 反向 overlap 小段。
- **基线**:Megatron-LM `dev@232c478d4`(2026-06-16);全部 `file:line` 现读现核。**索引**:[[megatron-lm/index]] 加 `> [!update] 2026-06-23` note。**纯增不删**:既有 §2.1–2.6 原样保留。

---

## 2026-06-23: MindSpeed 全 5 篇按「每特性四件套」再深挖(图示 + 优化点 callout + 源码,融合算子说明融合内容)

**Type**: Deepen（用户第三轮反馈"每个都比较浅显,每种优化特性最好补对应图示和优化点说明;融合算子说明融合了哪些内容 + 补源码解读"。先把 affinity 页定为新标尺(每算子四件套:融合内容→before/after 图示→`[!tip] 优化点`→源码解读),再以它为 in-house exemplar 并行重写其余 4 篇。源码核对 @ MindSpeed 1432cb09 / MindSpeed-LLM 0c16322d)

- **[[mindspeed_ascend_affinity_analysis]]**(468→**662 行**,10 图,12 优化点 callout):融合算子每个补**融合内容**(N 个散算子→1 核)——GMM(E 次切片+GEMM→1 变长分组,反向 dgrad 累加进 main_grad)、SwiGLU(chunk+SiLU+⊙→`npu_swiglu`)、RMSNorm(x²+mean+rsqrt+×→`npu_rms_norm`)、RoPE(rotate_half+cos/sin→`npu_rotary_position_embedding`)、Softmax(scale+mask+softmax 7 趟→`npu_scaled_masked_softmax`,带 fp16/sk≤4096 硬约束)、MoE-permute、Flash-Attention(**O(S²)→O(S),S×S 不物化**)、Fused-EMA-AdamW(一核回写 param/m/v/s);每个带 before/after 图 + 量化优化点。
- **[[mindspeed_context_parallel_analysis]]**(373→**420 行**,7 优化点 callout):Ulysses/Ring-双环/Hybrid/Adaptive/KV-cache/2·cp 负载均衡 每变体补量化优化点(通信量比、overlap、straggler 消除)。
- **[[mindspeed_parallelism_analysis]]**(469→**495 行**,18 优化点 callout,~20 图):TP-2D/非对齐/vocab/PP 划分/MoE-EP/LayerZeRO/Custom-FSDP/分层解耦(U-split/VDP/VTP)每特性补图 + 量化优化点。
- **[[mindspeed_comm_overlap_analysis]]**(406→**461 行**,9 优化点 callout):MC2/CoC/MoE-overlap/fb-overlap/alltoall-MC2/DualPipeV/RiPipe/optimize-p2p/async-log 每特性补时序图 + 量化优化点(气泡比、隐藏率)。
- **[[mindspeed_memory_optimization_analysis]]**(248→**422 行**,15 优化点 callout):重计算/Swap/reuse-fp32/MoE-zero-mem/压缩/virtual-opt/chunk-loss 每特性补 before/after 图 + 省显存 Δ 公式。

**校验**:各 agent 透明纠正若干行号(mc2 CoC 互斥 `:21-22`、planner greedy `:127-142`、flexible_schedules 路径 `core/pipeline_parallel/`、compress pdf/ratio),coordinator 抽样复核均命中;5 页 `[[]]` 链接脚本提取确认 0 悬空。MindSpeed 系列累计约 **2460 行**(index 除外)。

---

## 2026-06-23: [[fault_recovery_relink_comparison]] 深挖「重新建链全过程」+「进程状态管理」(§5/§6)

**Type**: Expand（应用户"再深入解读重新建链过程,以及各训练进程状态如何管理"。读 MindSpeed TTP 状态机源码 + ARF 清理/重建回调 + Megatron Wrapper finalize,补两节深度)

- **§5 重新建链完整过程**:MindSpeed ARF 的有序回调链(`stop→clean→rebuild_group→repair`,注册序 `tft_train_initialize.py:97-107`)逐步拆解 + 时序图——`stop_device` / `torch_sync` / `unset_gather_handle` 置空旧异步句柄(`tft_stop_clean.py:76-88`)/ UCE 检查迁坏 HBM 张量(`:36-49,60-74`)/ 逐组 `reinit_process_group(rebuild_link=True)`(`tft_arf_group_repair.py:31-98`);对比 Megatron NVRx `Wrapper` 的 abort→finalize(`destroy_state`)→rank_assignment(RESERVE 热备)→initialize(`inprocess_restart.py:25-29,50-67,80-125`)。
- **§6 训练进程状态管理**:MindSpeed 自研 TTP 的显式 `WorkerStatus` 状态机(INIT→NORMAL→{ABNORMAL/FAULT/PAUSE}→STOPPED,`core/ttp/constants.py:5-12`)、rank0 `TTPController._worker_status` + 心跳带 status/iteration(`comm/controller.py:53`、`comm/heartbeat.py:102-116`)、`_on_worker_fault` 广播 PAUSE(`controller.py:535-542`);并给出「故障时哪些状态丢/留」对照表(Megatron destroy&reload vs MindSpeed clean&repair-in-place)。

**校验**:`controller.py:535-542`(PAUSE 广播)、`constants.py:5/53`、`tft_stop_clean.py`、`tft_arf_group_repair.py:31-98`、`inprocess_restart.py:25-125` 均逐一开文件核对。页面 130→约 260 行。

---

## 2026-06-23: 新建「训练快恢与重新建链」跨框架对比页(Megatron/MindSpeed/MindFormers)

**Type**: New（应用户提问"故障节点更换涉及重新建链,Megatron/MindSpeed/MindFormers 各怎么做,结论要有事实依据"。3 个并行 research-agent 分别读三仓容错代码,coordinator 抽样核验关键引用后落页;wiki 此前无容错/快恢专题)

- **新增** [[fault_recovery_relink_comparison]]:跨框架快恢与「重新建链」机制对比,源码核对 @ Megatron `232c478d4` / MindSpeed `1432cb09` / MindSpeed-LLM `0c16322d` / MindFormers `01e71622` / torch_npu。
  - **Megatron-LM**:委托 NVRx,`--inprocess-restart` 进程内重启——abort NCCL(`inprocess_restart.py:93-98`)→ `destroy_model_parallel`(`training.py:286-292`)→ 新 `PrefixStore(str(iteration), store)` 命名空间重跑 `init_process_group`(`training.py:1088-1090`、`initialize.py:316-333`);热备 reserve rank 顶替。
  - **MindSpeed-LLM**:MindIO TFT / **ARF 空中加油**——`arf_rebuild_process_group_callback`(`tft_arf_group_repair.py:31,47`)调 `torch_npu reinit_process_group(rebuild_link=True)` → `abort_hccl_comm("reinit")`(`torch_npu .../distributed_c10d.py:346-372`)**原地重建** HCCL(PG 对象存活);故障 rank 优化器态从同伴 DP **replica** 拷回(`tft_replica_group.py:26`、`tft_optimizer_data_repair.py:86-175`);另有 elastic scale-in/out 全重建。
  - **MindFormers**:**不自实现**,委托 MindSpore runtime + MindIO——仅 `_tft_handler.init`(`build_context.py:346-352`)使能 ARF、包优化器、reboot 节点跳 barrier(`version_control.py:289-301`);重新建链在闭源 runtime 内。
  - 显式标注三处**闭源边界**(NVRx / MindIO `mindio_ttp` / MindSpore runtime),区分"框架 Python 可见" vs "运行时黑盒"。

**整合**:父索引 [[02_engineering/02_train_frameworks/index]] 页面列表新增条目。**校验**:抽样复核 `tft_arf_group_repair.py:47`(`reinit_process_group(rebuild_link=True)`)、`distributed_c10d.py:346/370`(`abort_hccl_comm`)、`build_context.py:346-352`(`_tft_handler.init`)、`inprocess_restart.py:93-98`(abort Compose)、`training.py:1088-1090`(PrefixStore)均逐一开文件命中。

---

## 2026-06-23: [[megatron_ep_analysis]] 新增「DeepEP 通信量图解」三图(§③.3.5)

**Type**: Update（应用户"图示解释 DeepEP 通信量分析,用 SVG 画图转 PNG 放进 wiki"。承接本轮对话链:核实 Megatron flex dispatcher 通信量估计 → 深挖 DeepEP `intra_dispatch` NVLink 扇出内核路径 → 本次落图）

- **配图基线**:为给出可核验 `file:line`,本地浅克隆 **DeepEP @ `af9a040`**(`main`,2026-06-15;`csrc/kernels/legacy/` v1 `Buffer` 内核 = `--moe-flex-dispatcher-backend deepep` 路径)。
- **新增** [[megatron_ep_analysis]] §③.3.5 三图(SVG 手绘 → headless Edge 2× 元素截图 → PNG,存 `megatron-lm/assets/megatron_ep_analysis_deepep_fig{1,2,3}.png`):
  - 图 1:标准 AllToAll(按专家,跨界冗余 k 次)vs DeepEP fused_dispatch(按节点,一次 RDMA + 节点内 NVLink 扇出);跨节点流量 ∝ |R(t)| ≤ k。
  - 图 2:两级通信量分解 RDMA = |R(t)|·M / NVLink = [Σ(gₙ−1)+g_s]·M,「省 1 跳 IB ⇄ 多 1 跳 NVLink」严格相等;源码对应 `notify_dispatch`(`internode.cu:314` 每节点 `total_count` / `:313` 每卡 `per_nvl_rank_count`)+ `SourceMeta` 位图(`:22`)+ `kRDMAAndNVLForwarder` 逐卡选通(`:971`)。
  - 图 3:2 node × 2 GPU 数值走查(token X→{E1,E3,E5,E6}),跨节点 IB 2M→1M、节点内 NVLink 1M→2M,IB 加速比 (k/P)/(1−(1−1/P)ᵏ)(P=2,k=4→2.13×、topk=8→≈4×)。
- **源码纠正(code wins)**:DeepEP 落地卡 = 与源卡「同号」的 NVL rank(`internode.cu:826`),未必是目标卡;故节点内实际 NVLink = gₙ − 𝟙[同号落地卡∈目标],§③.3.2 理想公式的「−1」为上界(最好情形)。已在图 3 与正文标注。
- **工具**:复用 gitignored `.html2md/`(puppeteer-core→Edge)新增 `deepep_figs/render.mjs` 元素截图脚本。**索引**:[[megatron-lm/index]] 加 `> [!update] 2026-06-23` note。

---

## 2026-06-23: MindSpeed 系列从「分类大纲」深挖到机制级(+CP 专页,4 篇重写到 megatron 深度)

**Type**: Deepen（用户反馈"大纲有了,针对特性的 deep dive 分析缺少,参考 megatron-llm / torchtitan"。校准:`megatron_ep_analysis`(753 行,单机制)/`megatron_cp_analysis`(391)是深度标尺——每机制需 命题→源码片段+`file:line`→数据流图→通信/显存代数→权衡。原 4 篇每机制仅 ~15-30 行,远不及。并行 writer-agent 逐页重写,coordinator 抽样核验引用)

- **新增** [[mindspeed_context_parallel_analysis]](373 行,对标 [[megatron_cp_analysis]]):CP 家族专页——分派脊柱 `dot_product_attention.py:134-322` 路由五变体;**Ulysses**(头维 `single_all_to_all` 换轴 `:83-108`,通信量 vs Ring 推为 `(2/cp)·(a/a_kv)`)、**Ring/双环**(KV 环形 P2P + online-softmax `utils.py:77-119`、因果块跳过 `ring_context_parallel.py:16-33`、双环窗口 `model_parallel_utils.py:121-212`)、**Hybrid**(Ulysses×Ring 2D)、**Adaptive**、**KV-cache**;含 2·cp 因果负载均衡(`get_batch_utils.py:244-263`)与各变体整除约束(Ulysses `a%(cp·TP)==0`、megatron-cp `seq%(2cp)==0` 等)。
- **重写加深** [[mindspeed_parallelism_analysis]](241→469 行):CP 段收为指针(导向新 CP 专页),腾出篇幅深挖 **TP-2D**(`linear_2d_split_along_first_dim.py:128-150` AG→MM→RS 二维流水)、**PP 划分**(noop/非对齐/布局)、**MoE-EP**(GMM 原语 `gmm/experts.py:124-151`、tp-extend-ep、专家放置贪心重排 `expert_placement/planner.py:111-150`)、**LayerZeRO3**(`zero3/fsdp.py`)/Custom-FSDP、**分层解耦训练**(U-split/VDP/VTP)。
- **重写加深** [[mindspeed_comm_overlap_analysis]](267→406 行):三母题各配时序图+代数——*算子融合*(MC2 `npu_all_gather_base_mm`/`npu_mm_reduce_scatter_base`、alltoall-MC2 `npu_alltoallv_gmm`)、*软件流水*(CoC chunk 双流 `coc_utils.py:200-248`、MoE async-handle、fb-overlap 跨微批 + WeightGradStore 解耦 `:203-205`)、*换调度*(DualPipeV 7 段 `:310-344`、RiPipe、optimize-p2p)。
- **重写加深** [[mindspeed_ascend_affinity_analysis]](233→468 行,代码块 2→15、配图 4):补足代码走读——op_builder JIT(`builder.py:65-77` `cpp_extension.load`、GMM 三 dispatch key→CANN `GroupedMatmul`)、融合算子(`npu_swiglu`/`npu_scaled_masked_softmax`/`npu_rms_norm`/`npu_fusion_attention`)、HCCL buffer/QoS、`npu_apply_fused_ema_adamw`;保留 affinity 勘误([!warning])。
- [[mindspeed_memory_optimization_analysis]] 维持(原已达深度标尺)。

**整合**:[[mindspeed/index]] 四大类表新增「并行·CP 深挖」行;父索引 [[02_engineering/02_train_frameworks/index]](4→5 篇)与总索引 [[index]] 领域总览(MindSpeed 6)/快速导航(增 CP 页)同步。**校验**:各 agent 透明纠正了若干行号(expert_placement 路径、U-shaped loss 行、gmm dispatch 行),coordinator 抽样复核 `planner.py:111`(greedy)、`schedules.py:300`(U-loss)、`gmm.py:153`(@impl PrivateUse1)、`mc2_fuse_a2a.py:39`(npu_alltoallv_gmm)均命中;5 页 `[[]]` 链接经脚本提取确认 0 悬空。

---

## 2026-06-23: 新建「MindSpeed × MindSpeed-LLM 昇腾训练加速特性」子目录(index + 4 篇深挖)

**Type**: New domain（应用户目标"分析 MindSpeed+MindSpeed-LLM 的训练优化特性:并行/计算通信掩盖/内存优化/昇腾亲和,总结进知识库"。源码核对 @ MindSpeed `master 1432cb09`(patches Megatron core_r0.17.0)+ MindSpeed-LLM `master 0c16322d`;4 篇深挖由并行 writer-agent 各读一域源码产出,coordinator 校验引用与链接）

新建 `wiki/02_engineering/02_train_frameworks/mindspeed/`:
- [[mindspeed/index]](知识地图):**猴补丁式 Megatron 加速层**定位、`MindSpeedFeature` 契约(register_args/register_patches/validate、O0/O1/O2 优化等级门控 `feature.py:12-20`)、`create_features_list()` ~70 特性总账(`features_manager/__init__.py:367-398`)、两层结构(core 通用 + LLM 模型/任务层)、四大类罗盘。
- [[mindspeed_parallelism_analysis]](241 行):CP(Ulysses 头切 all-to-all / Ring / 自适应 / KV-cache,2·CP 因果负载均衡)、TP(非对齐线性 / TP-2D `[h/y,E/x]` / vocab ReplaceIndexPut)、PP 划分(noop/布局/非对齐/num-layer-list)、MoE-EP(tp-extend-ep + GMM 原语 + 专家放置 EMA 预测)、DP/分布式(LayerZeRO3 / Custom-FSDP)、分层解耦训练(U-split/VDP/VTP)。
- [[mindspeed_comm_overlap_analysis]](267 行):两大母题——chunk-GEMM 流水异步通信(CoC/MoE-overlap)与 matmul+集合通信单核融合(MC2 `npu_all_gather_base_mm`/`npu_mm_reduce_scatter_base`、alltoall-MC2 `npu_alltoallv_gmm`);PP 换调度消/填气泡(DualPipeV 7 段 / RiPipe 重算填泡 / optimize-p2p)。**勘误**:async-log-allreduce 掩盖的是 loss 日志 all-reduce,非梯度规约。
- [[mindspeed_memory_optimization_analysis]](248 行):统一原语 `untyped_storage().resize_(0)` + 反向重填;重计算(激活/norm/按 PP-rank/block-uniform)、Swap(smart-swap/swap-attention saved_tensors_hooks/swap-optimizer 态常驻 CPU)、reuse-fp32-param(fp32↔bf16 共享存储 `reuse_data_ptr`)、MoE-zero-memory、压缩(HANS/换尾数)、virtual-optimizer、chunk-loss。
- [[mindspeed_ascend_affinity_analysis]](233 行):**算子替换层**——op_builder JIT(`cpp_extension.load` 编 CANN 核)、融合算子(GMM 三 dispatch key→`GroupedMatmul`、`npu_swiglu`/`npu_scaled_masked_softmax`/`npu_rms_norm`)、Flash-Attention(`npu_fusion_attention` SBH/TND)、HCCL buffer/QoS 调优、融合优化器(`npu_apply_fused_ema_adamw`)。**重要勘误**:`AffinityFeature` 并非 CPU 绑核(两仓 grep `sched_setaffinity/numa` 皆空),而是 VocabParallel 交叉熵的 NPU 亲和改写(`affinity.py:13-17` 补丁 `calculate_predicted_logits`)——已据此修正 index 描述。

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

**整合**:姊妹篇 [[mindformers_moe_token_dispatcher_analysis]] footer 增「PyNative 对照」回链。**校验**:新页 `file:line` 均按当前 checkout 逐一核对;交叉链接 [[mindformers_moe_token_dispatcher_analysis]] / [[megatron_ep_analysis]] / [[torchtitan_ep_analysis]] / [[deepseek_moe_analysis]] 经 glob 确认存在。

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

**整合**:父索引 [[02_engineering/index]] 子领域表新增 `06_auto_parallel` 行;综述页交叉链接 [[megatron-lm/index]](手工 5D 对照组/执行后端)、[[torchtitan/index]](DTensor 原生)、[[mindspore_compiler_analysis]](传播范式)、[[comm_compute_fusion_guide]](overlap 实测)、[[distributed_optimizer_deep_dive]](ZeRO/FSDP 分片)。**校验**:2 页均含 `## Related Pages`,跨链目标页经 glob 确认存在,0 悬空链接;论文出处以 Sources 段外链给出(Alpa/GSPMD/nnScaler/PartIR/Galvatron-BMW/综述/DTensor 等)。

---

## 2026-06-22: vLLM 系列补「算子融合与 Triton Kernel」专页(系列增至 11 篇 + index)

**Type**: Expand（应用户提问"融合算子/Triton 等算子特性有介绍吗"——既有 10 篇仅在注意力/量化页顺带提及,无专篇;补 [[vllm_fused_ops_and_kernels_analysis]] 填补真空）

新增 [[vllm_fused_ops_and_kernels_analysis]](「特性优化」支柱):**CustomOp 多实现派发**(`model_executor/custom_op.py` native/cuda/triton + `custom_ops` 开关,Inductor 下默认走 native 交其自动融合)、**torch.compile 融合 Pass**(`compilation/passes/fusion/`:RMS+quant、SiluMul+quant、AllReduce+RMSNorm/async-TP、attention+quant、SP,经 `PostGradPassManager` 挂进 Inductor `post_grad_custom_post_pass`)、**fused_moe**(grouped GEMM + Triton/CUTLASS/DeepGEMM oracle 派发 + `configs/E=*,N=*,device=*.json` autotune)。与 [[vllm_compilation_cudagraph_analysis]](图捕获)、[[vllm_quantization_analysis]](量化 GEMM)、[[vllm_attention_backends_analysis]](Triton 注意力)形成"被引用→展开"分工;跨域对照 [[megatron_fusion_operators_analysis]] / [[torchtitan_compute_memory_optimizations_analysis]]。

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

- **入门两篇**:[[verl_architecture_overview_analysis]](HybridFlow 混合控制器、五平面、五角色、v0/v1 入口、master 架构图)、[[verl_quickstart_guide]](安装/Hydra 启动/config 体系/一次 GRPO 端到端走查/后端切换旋钮)
- **实现五篇**:[[verl_single_controller_analysis]](`@register`+8 种 Dispatch、`DP_COMPUTE_PROTO` chunk/concat、RayWorkerGroup/colocate)、[[verl_dataproto_analysis]](`DataProto`/`BatchData`/`DataProtoFuture`)、[[verl_ray_trainer_analysis]](`RayPPOTrainer.fit()` 逐步追踪 + 数据流时序图)、[[verl_workers_engine_analysis]](`TrainingWorker`/`ActorRolloutRefWorker` + `BaseEngine` 模板方法 + FSDP/Megatron 引擎)、[[verl_rollout_resharding_analysis]](vLLM/SGLang 异步 server + 3D-HybridEngine:`get_per_tensor_param`+`CheckpointEngine`+CUDA-IPC bucketed transfer)
- **算法与优化两篇**:[[verl_rl_algorithms_analysis]](`core_algos` 14 种优势估计 + 11 种 policy loss + KL k1/k2/k3,均含 LaTeX)、[[verl_optimization_analysis]](placement/offload/序列打包/Ulysses SP/异步 RL 旋钮目录)

**HEAD 关键勘误(各页已标注,与多数博客的「经典 HybridFlow」描述不符)**:
- `RayPPOTrainer` 已 `@deprecated`(`ray_trainer.py:285`)但默认 `trainer.use_v1=false` 仍走它;新路径为 `TaskRunnerV1`+TransferQueue+`AgentLoopManager`。
- **无独立 `CriticWorker`/`RewardModelWorker` 类**:critic = 带 value head 的 `TrainingWorker`,reward 走 `workers/reward_manager` + `experimental/reward_loop`。
- rollout 退役 SPMD 同步模式,改异步 server(`ServerAdapter.generate_sequences` 直接 raise),生成由 `LLMServerManager`/`AgentLoopManager` 驱动。
- `Role` enum 实际在 `trainer/ppo/utils.py:27`(ray_trainer 仅 re-export);`compute_policy_loss`(core_algos:1203)已废弃,实际分发走 `workers/utils/losses.py` 的 `get_policy_loss_fn`。

**整合**:[[verl/index]] 知识地图(五平面表/9 篇三层表/五角色表/RL 数据流图/与训练后端的 cross-domain 链接);父索引 [[02_engineering/04_posttrain_frameworks/index]] 新增 verl 子目录行、总索引 [[index]] 更新目录树/计数/快速导航;9 篇互链 + 跨域链(→ [[torchtitan_fsdp_analysis]]/[[megatron-lm/index]]/[[distributed_optimizer_deep_dive]] 等)。**校验**:9 页全部含 `## Related Pages`、均回链 [[verl/index]];所用 sibling slug 与文件名一一对应;跨域目标页均存在,0 悬空链接。

---

## 2026-06-17: 内置 default 后端「真·Split-Tiling」页改名对称（`npu_inductor_splittiling_backend_analysis`）

**Type**: Rename（应用户「内置后端加 `_splittiling` 对称区分」；仅改唯一真·Split-Tiling 页，MLIR/DVM/总览/通用页保持原名以免误标；`git mv` + 全 wiki `[[link]]` 同步）

为与实验性 [[npu_inductor_linearize_backend_analysis]] 成「方案对称对」，把内置 default 后端唯一描述 **Triton / Split-Tiling 路径**的深度页改名：
- `npu_triton_backend_deep_analysis` → [[npu_inductor_splittiling_backend_analysis]]

**刻意未改名（非单一方案，加后缀会误标）**：[[npu_compile_paths_overview]]（Triton/ACLGraph/MLIR 三路径）、[[NPU_Inductor_Backend_Analysis]]（5 后端融合规则）、[[npu_inductor_optimization_analysis]]（跨 Triton/MLIR/DVM）、[[npu_lowering_guide]] / [[npu_compile]] / [[npu_debug_guide]]（通用）。

**同步**：该页有 10 处跨目录入链（04_inductor / 05_codegen_backends / 07_op_registration / changelog），全部 `[[link]]` 用 perl 同步；页头加 `> [!note]` 指向实验 Linearize 对照页；[[04_inductor/npu/index]] 行标「内置 default（Split-Tiling）」。校验：0 残留旧名、0 新增悬空链接。

---

## 2026-06-17: megatron_comm_overlap_analysis §5.6.1 补 DeepEP/HybridEP 两级通信模型

**Type**: Expand(承 §③.3,把两级模型按"加速通信"角度补到通信掩盖页;纯增,交叉引用避免重复)

**背景**:`megatron_ep_analysis` §③.3 已落地两级通信量公式与数值走查;通信掩盖页 §5.6（DeepEP/HybridEP 后端）此前只说"降 A2A 绝对耗时与 SM 占用",未解释**为什么**能降。

**新增([[megatron_comm_overlap_analysis]] §5.6.1)**:
- 两级拆分(`num_tokens_per_rdma_rank`/node→`inter_dispatch`/IB + `num_tokens_per_rank`/GPU→`intra_dispatch`/NVLink,双 buffer `num_rdma_bytes`/`num_nvl_bytes`,`fused_a2a.py:62/135`)+ 去冗余规则(跨 node 只发一次)。
- 关键式:跨节点 `∝|R(t)|`、IB 加速比 $\frac{k/P}{1-(1-1/P)^k}$(2 node topk4→2.13×、topk8→4×);完整走查指向 [[megatron_ep_analysis]] §③.3,不重复。
- **与"掩盖"的关系**:§5.6 去冗余/两级降 A2A 绝对耗时 + §5.1 1F1B 把剩余 A2A 掩盖到计算后;并接 §5.7 的 `high_priority_a2a_comm_stream` / `moe_hybridep_num_sms_preprocessing` 调尾延迟。

**校验**:LaTeX、`path:line`、`[[link]]` 按页约定;无删改既有内容。

---

## 2026-06-17: NPU 实验后端 3 页按「方案」改名（`npu_inductor_linearize_*`）

**Type**: Rename（应用户「区分方案」要求，避免与 torch_npu 内置后端页混淆；`git mv` + 全 wiki `[[link]]` 同步）

把实验性 `npu_inductor_2.9.0`（**Linearize 方案**）的页统一为 `npu_inductor_linearize_*` 前缀，与内置 default（**Split-Tiling**）的 `npu_inductor_*`/`npu_triton_*` 区分：
- `npu_inductor_dynamic_shape_analysis` → [[npu_inductor_linearize_dynamic_shape_analysis]]
- `npu_inductor_vs_builtin_comparison` → [[npu_inductor_linearize_vs_builtin_comparison]]
- [[npu_inductor_linearize_backend_analysis]] 本已合规，不变。

全 wiki `[[link]]`（index / changelog / 三页互链）同步更新；校验：0 处残留旧名、0 新增悬空链接。

---

## 2026-06-17: Inductor 分析「完整录入」扩写 — NPU 实验后端拆 3 页（§0 对标 + §1 三方 output code）+ 上游融合补全

**Type**: Expand（应用户「完整录入、不过度裁剪、查知识熵减」要求；回读原始《npu_inductor 设计与对标分析》§0/§1 精确还原；纯增不删既有结构）

**背景**：同日先前的「Inductor 后端分析合入」条为控冗余而**压缩过度，存在明显知识熵减**——丢了 §0 全套实测对标表、§1 GPU/内置/本后端三方 output code 逐行对比、四遍折叠的 dual-decomp 实例、动态 shape 三情形 A/B/C 完整代码、上游 G2 融合的 prologue/epilogue/foreach/proximity 细节。本次按「完整录入」扩写还原。

**NPU 侧：1 页 → 3 页系列**：
- [[npu_inductor_linearize_backend_analysis]] 扩为完整版：装配顺序全表、Linearize 恒等式 + `_apply_linearize` 主干 + 四遍折叠表 + **dual-decomp 折叠实例**（softmax-bw + sum(0) + permute，4 独立轴 → 2 基础轴的完整除/模映射 + 地址文本修复）、索引线性化全 6 pass、40-CU group dispatch prologue 完整代码、融合门控（病灶数据 + 别名坑）、r 轴 rsplit（partial + combine）、**全 5 处类型降型**、白名单 lowering + 算子专项、**完整可优化点**。
- [[npu_inductor_linearize_dynamic_shape_analysis]]（新）—— 编译一次 vs gears 分桶、签名 numel/divisor 代码、header 三件套、**三情形 A/B/C 完整代码 + 对照表**、配套（fold_trivial / 符号 split / static split block）、permute 产物。
- [[npu_inductor_linearize_vs_builtin_comparison]]（新，comparison）—— **§1 三方 output code 逐行对比**（GPU / 内置 Split-Tiling / 本后端 Linearize 完整 kernel + 逐项差异表）+ **§0 全套实测**（torchbench 34 模型总体 / 逐模型、京东 OneRec 4 backbone、test_all 60 算子 case）+ 逐维综合矩阵。

**上游侧补全**：[[scheduler_analysis]] 新增 §7.6——组兼容（numel/rnumel + tiling 一致）、proximity 门控（>64）、模板 prologue/epilogue 融合、foreach 融合、`_LoopMutationTracker` 回滚 + 循环重排；`> [!note]` 指向 NPU 后端的 read 门控 / proximity 收 20。

**索引/校验**：[[04_inductor/npu/index]] 补 2 新页行；3 个 NPU 页互链 + backlink 既有内置后端页；无悬空链接；§0/§1 数据均回读原始设计文档精确还原并标注口径（本库未独立复跑）。

---

## 2026-06-17: megatron_ep_analysis §③.3 补「两级通信量公式 + 数值走查 + all2allv 澄清」

**Type**: Expand(对照 `Megatron-LM` `dev@232c478d4` 源码 `fused_a2a.py` / `token_dispatcher.py` 核实;单页增补,纯增)

**背景**:用户追问 DeepEP/HybridEP 的**具体通信量公式**、**两级通信如何进行**,以及节点间是否用 all2allv。原 §③.3 只有一张概念示意图,缺公式与逐字节走查。

**新增([[megatron_ep_analysis]] §③.3 下 4 个子小节)**:
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
- [[npu_inductor_linearize_backend_analysis]]（NPU）—— 实验性 `npu_inductor_2.9.0`：import 即 patch + `disable_register_inductor_npu()` 关掉内置后端、Linearize（多维→40-CU `bin[40,1,1]` group dispatch）+ 索引线性化、编译一次动态 shape（3 情形）、`NPU_MAX_FUSED_READS` 融合门控、r 轴 rsplit、类型降型、白名单 lowering、与内置后端逐维对比、可优化点。
- [[inductor_gpu_kernel_dispatch_model]]（upstream）—— GPU kernel 骨架（`program_id→offset→mask`，无循环）、`IterationRanges` 树、stride-1 tiling、`Grid1D/2D/2DWithYZOverflow/CooperativeReductionGrid`。
- [[inductor_reduction_codegen_deep_analysis]]（upstream）—— persistent/looped/split/cooperative reduction（semaphore barrier）、block ptr/TMA。
- [[inductor_autotuning_analysis]]（upstream）—— `CachingAutotuner` 生命周期、config 启发式、`config_of`/AttrsDescriptor、`make_launcher`、`triton.compile(ASTSource,GPUTarget)`→PTX/cubin、`DeviceProperties`（填补「Inductor autotuning」空白）。

**增补 1 处**：[[inductor_codegen_dynamic_shape_analysis]] 新增 §2.4——`s0→ks0` 重命名 + `signature_to_meta._decide_tl_dtype` 把动态 `ks*` 升 `tl.int64` 防 `ks0*ks1` 溢出；加 `> [!contradiction]` 指向 NPU 后端的 i32 降型（GPU↔NPU 动态 shape 整型的根本分歧）。

**索引/空白更新**：[[04_inductor/index]] 加「codegen 派发与运行时（GPU 基线）」分组 3 行；[[04_inductor/npu/index]] 加实验后端行 + 头注（区分内置/实验、PyTorch 2.9.0 基线）；[[01_ai_frameworks/index]] 空白「Inductor autotuning」「NPU Monkey Patch 演进追踪 v2.9.0」标 ✅ 并指向新页。

**交叉引用**：4 新页均含 `## Related Pages` + `[[wikilink]]`；NPU 页 §六对比直接 backlink 既有 [[npu_inductor_splittiling_backend_analysis]]/[[npu_compile_paths_overview]]/[[npu_inductor_optimization_analysis]]（内置后端细节，不复述）；上游 3 页互链并指向既有 [[scheduler_analysis]]/[[dynamic_shapes_full_analysis]]/[[PyTorch_Inductor_Technical_Analysis]]。

**核验**：所有代码引用带 `file:line`（npu_inductor 包 + upstream `torch/_inductor/`）；零冗余（内置后端/上游已覆盖部分一律 cross-link 而非复述）；纯增，未删改既有页结构。源分析底稿在 pytorch 工作区 `npu_inductor_2.9.0/triton-backend-analysis/`。

---

## 2026-06-17: SimpleFSDP 页 §5 深挖(编译流程 + 两个通信 pass + 掩盖机制)

**Type**: Expand(对照 torchtitan `main` @ `61c010fcb` `experiments/graph_trainer/` 源码逐行核实;纯增不改既有结构)

**背景**:`[[torchtitan_simple_fsdp_analysis]]` 原 §5 偏"概念入门",未讲清编译流程 / 通信 pass / 加 pass 阶段 / 掩盖机制。用户追问后,读透 `compile.py` / `passes.py` / `fsdp_passes.py` / `trainer.py` / `make_fx_tracer.py`,把 §5 从一节扩成 5.1–5.4 源码级深挖:

- **5.1 编译流程**:`aot_fx_trace` 首步 `minimal_fx_tracer`(make_fx)把 fwd+loss+bwd 追成**一张 joint FX 图**(redistribute 落成 `all_gather_into_tensor`/`reduce_scatter_tensor` 节点),`apply_graph_passes` 跑 `compile_time_passes` 流水线改写图(只跑一次),之后每步 `run_traced` 复用;给出 10 步 pass 流水线,标出两个通信 pass 在**第 6/7 位**(显存策略之后、inductor 之前)。
- **5.2 通信 pass ①** `reassign_collective_pgs_pass`:把 AG 改派到额外 NCCL PG(同 ranks、`use_local_synchronization`)→ 独立 CUDA 流 → **AG∥RS∥compute**(等价 FSDP2 多流)。
- **5.3 通信 pass ②** `joint_transformer_block_bucketing_reordering_pass`(`JointManualOverlapScheduler`):按 block/方向/FSDP2 参数序**分桶**(每 block 合 1 AG+1 RS)+ `overlap_deps` **重排**(AG 逆序预取、RS 延后 wait 越过计算)。
- **5.4** 一图收束 + 纠正:`autobucketing_/transformer_block_bucketing` 是已废弃 JIT 后端的非 joint 版,默认 aot_fx_trace 走 joint 版。

**更新**:`[[torchtitan_simple_fsdp_analysis]]` §5 重写、复核表补 8 条(编译流程 + 两 pass)、§9 小结补编译流程条;同步源文档 `llm_repo/torchtitan/docs/parallelism-analysis/simple-fsdp.md`。页头日期 → 2026-06-17。

**追加(同日,应用户「补充通信粒度 + 配图」)**:新增 **§5.5 通信粒度**——讲清 SimpleFSDP **trace 时逐参数(一参一次,无 eager 分组)→ 编译期 bucketing pass 按 block 合成每块 1 AG+1 RS**;「分层统一通信」是编译期优化产物、非天生(不开 compile 即退化逐参数)。修正 §6 对比表「通信单位」行。新增 **2 张 SVG→PNG 机制图**(入 `torchtitan/assets/`):`simple-fsdp-compile-flow`(编译流程 + 10 步 pass 流水线,高亮第 6/7 通信 pass)、`simple-fsdp-bucketing-overlap`(逐参数 → 每块 1AG+1RS 的三流并发时间线)。复核表补「参数化逐参数 getter」行。

---

## 2026-06-16: Megatron-LM 知识库去重整合 + 命名对齐 torchtitan(删 1 · 改名 21 · 索引收敛)

**Type**: Refactor & Dedup(用户授权"只删重复的 md";4 个只读 agent 产出重复度矩阵 → 合并唯一独有内容 → 删冗余文件 → 全量改名 + 链接修复;允许删除既有文档)

**背景**:megatron-lm 目录此前**命名两代混杂**——旧代 CamelCase/前缀混乱(`Megatron-LM_MoE_Zero_Redundancy_Analysis`、`Megatron_LM_TFLOPS_Analysis`)+ 新"源码级系统分析系列"无前缀(`ep_analysis`/`tp_analysis`…),且疑似存在重复知识。4 个只读 agent 逐页对照后结论:**两代多为"深版 + 精简digest 指针"的互补关系,而非重复**,仅 1 页是真正被涵盖的冗余。

**① 去重删除(1 文件)**:
- 删除 `Megatron-LM_MoE_Zero_Redundancy_Analysis.md` —— 其零冗余 AllToAll / 七阶段 dispatcher / MoE Folding 知识已被 [[megatron_ep_analysis]] **完全涵盖且更深**(该旧页 2026-06-16 自身更新note 也已指向 ep_analysis)。删除前把其**唯一独有教学资产**「EP=4、num_experts=4、topk=2 的逐 token 数值走查(routing_map 矩阵 + A2A 传输矩阵 + 反向 A2A + 加权 unpermute)」并入 [[megatron_ep_analysis]] §②.3.1。

**② 命名对齐 torchtitan(改名 21 文件)**:本目录全部页统一为 `megatron_<topic>_analysis`(小写 snake_case,对齐 `torchtitan_<topic>_analysis` 风格)。
- 新系列 19 页加前缀:`ep_analysis`→`megatron_ep_analysis`、`tp_analysis`→`megatron_tp_analysis`、`cp_analysis`→`megatron_cp_analysis`、`pp_schedulers_analysis`→`megatron_pp_schedulers_analysis`、`ddp_optimizer_analysis`→`megatron_ddp_optimizer_analysis`、`recompute_analysis`、`optimizer_internals_analysis`、`precision_cudagraph_fusion_analysis`、`training_stability_observability_analysis`、`rl_posttraining_consistency_analysis`、`inference_engine_analysis`、`model_structure_analysis`、`dataset_analysis`、`packed_dataset_dynamic_cp_analysis`、`dist_checkpointing_analysis`、`parallelism_orchestration_analysis`、`pp_supplements_analysis`、`tp_fsdp_resharding_supplements_analysis`、`moe_training_optimization_report` 均加 `megatron_` 前缀。
- 旧 CamelCase 2 页规整:`Megatron_LM_TFLOPS_Analysis`→`megatron_tflops_analysis`、`Megatron_vLLM_Weight_Sync_Analysis`→`megatron_vllm_weight_sync_analysis`。
- 已合规 5 页不动:`megatron_comm_overlap_analysis`、`megatron_fusion_operators_analysis`、`megatron_memory_optimization_analysis`、`megatron_distributed_optimizer_analysis`、`megatron_nonuniform_tp_analysis`。
- **链接修复**:全 wiki(208 个 md)用 `[[<basename><delimiter>` 锚定的 perl 替换更新所有 `[[wiki link]]` + 反引号/散文中的 `*.md` 文件名提及;锚定保证不误伤 `[[deepseek_v4_cp_analysis]]`/`[[torchtitan_tp_analysis]]`/`[[megatron_nonuniform_tp_analysis]]` 等近名页。

**③ 修复历史悬空链接**:`[[llm_parallelism_analysis]]`(该页从未以 .md 形式存在,仅旧 .html;changelog 早有记录)全 wiki 重指向 [[megatron_pp_schedulers_analysis]](正反向 DAG + 调度,最贴近其原意);涉及 megatron-lm/index、父级 `02_train_frameworks/index`、torchtitan 多页。

**④ 索引收敛**:[[megatron-lm/index]] 原"Core Topics"旧分组中 3 行(并行/MoE 行)已与下文 18 篇系列重复——重构为「全景报告(capstone)+ 专题深挖(系列外深版:distributed_optimizer / memory / fusion / comm_overlap / nonuniform_tp / tflops / vllm_weight_sync)」,移除重复行;Related Pages 去重;加「去重与命名整合」note 说明深版/digest 互补关系。父 index 同步基线 `ee3f1ff`→`232c478d4` 并移除悬空行。

**保留判定(审计结论:深版,非重复,不删)**:`megatron_fusion_operators_analysis`(融合算子全目录,precision §3 是其 digest)、`megatron_nonuniform_tp_analysis`(NTP 容错深版且更准确,tp_fsdp_resharding §2 是其 digest)、`megatron_distributed_optimizer_analysis`(FP8/FP4 量化 + CPU-offload + 三种 FSDP 对比 + §A.7 Muon,多页反向引用)、`megatron_memory_optimization_analysis`(显存 survey,与 recompute 互补);`distributed_optimizer_deep_dive`(父目录,跨框架对比)。

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
- [[torchtitan_hsdp_backward_overlap_analysis]] — HSDP 反向 reduce-scatter 与 all-reduce 双流掩盖:`foreach_reduce` 跨流编排、host 端算子下发顺序、AR∥RS 并发的正确性证明、reduce 路径 fp32 暂存与显存峰值;**带 3 张 SVG→PNG 机制图**(时间线 / 正确性分解 / 显存)
- [[torchtitan_compute_memory_optimizations_analysis]] — 算力/显存:Float8 rowwise/MXFP8、FusedSwiGLU/MoE grouped GEMM/FusedQKV、逐 block 编译即融合、融合 Adam、ChunkedCELoss、CPU offload
- [[torchtitan_comm_optimizations_overlap_analysis]] — 通信:跨维度计算-通信掩盖矩阵、Async-TP 微流水、对称内存、MinimalAsyncEP、full_dtensor SPMD
- [[torchtitan_simple_fsdp_analysis]] — SimpleFSDP(graph_trainer 实验,arXiv:2411.00284):分片即 DTensor `redistribute` 进图、编译器 pass 分桶重叠;与 FSDP2 eager 多流编排逐项对比

**纠正的常见误解(已写入)**:① Float8 rowwise 下通信仍高精度,本版无 fp8 all-gather;② torchtitan 核心循环不在 microbatch 间延迟 FSDP 规约(每 mb 都 reduce-scatter);③ MinimalAsyncEP 不做通信-计算重叠;④ `set_symm_mem_for_comm` 在 torch 2.9.1 不存在(追踪更新版 torch)。

**索引更新**:[[torchtitan/index]] 头块基准 `cf3c4312`→`61c010fcb`、计数 9→12、新增「深挖伴篇(3)」补 HSDP +「性能手段与编译器路线(3)」分区 + Related Pages;[[02_engineering/02_train_frameworks/index]] torchtitan 行计数 7→12、基准更新、日期 2026-06-16。资源:3 张 HSDP 机制图(SVG+PNG)入 `torchtitan/assets/`。

**校验**:4 页无残留相对链接 / 旧页脚,HSDP 三图引用完整;13 项 rel-link→`[[wiki]]` 映射逐条核对。来源同步自 `E:\97-codes\llm_repo\torchtitan\docs\parallelism-analysis\{07-hsdp-backward-overlap, 08-compute-and-memory-optimizations, 09-comm-optimizations-and-overlap, simple-fsdp}.md`。

---

## 2026-06-15: 补齐 eager 运行时地基 — 新增 7 模块 21 页(Workflow 编排 + 源码逐行核实)

**Type**: Ingest & Expand（Workflow：覆盖审计 → 架构全图 → 缺口路线图；再 7 模块流水线 research→并行写 3 层→校验；铁律＝纯增不改既有、对照 `E:\97-codes\pytorch\pytorch` v2.13.0a0 核实行号、不破坏既有 wikilink）

**审计结论**：01_ai_frameworks 此前几乎全是 torch.compile 编译栈 + dispatcher/op 注册,**整层 eager 运行时地基缺失**（用户点名 autograd 引擎、tensor 表达机制）。13 个 agent 对照源码核实后产出 8 项优先级缺口路线图。

**新增（7 模块,P0+P1,纯增无删改）**：
- **[P0] [[00_tensor_and_storage/index]]**：张量表达机制 — `Tensor=intrusive_ptr<TensorImpl>`、Storage/视图别名、sizes/strides/dtype、张量上的 DispatchKeySet（overview + quickstart + deepdive）
- **[P0] [[10_eager_autograd/index]]**：eager 反向引擎 — Node/Edge DAG、多线程 Engine、AccumulateGrad、SavedVariable、自定义 Function；**含与 03_aot_autograd 的对照表**（运行时磁带 vs 编译期联合图）
- **[P1] [[11_aten_op_execution/index]]**：ATen 算子定义/执行 — native_functions.yaml、torchgen、结构化 kernel、boxing（07 的上游通用版）
- **[P1] [[12_nn_module_system/index]]**：torch.nn — Module/Parameter/state_dict/hooks/容器/lazy/Optimizer
- **[P1] [[13_runtime_memory_amp_profiler/index]]**：缓存分配器 / AMP+GradScaler / Kineto Profiler
- **[P1] [[14_fx_export_and_extensibility/index]]**：torch.fx / torch.export / torch.library custom_op / functorch
- **[P1] [[15_distributed_primitives/index]]**：c10d/DDP/FSDP/DTensor/TP/PP（[[02_train_frameworks/index]] 的底座）

**索引与规划**：域 [[01_ai_frameworks/index]] 重构为「两条主轴（eager 地基 / 编译栈）+ 三层功能目录」,「知识空白」扩写为带优先级的规划路线图（P2 序列化遗留 `16_serialization_and_legacy`、各新模块 NPU 特化 `npu/` 下沉等留痕）。

**校验**：21 页结构齐全（头块/层次/Related Pages）；域内 317 条 wikilink 全解析；独立抽查 tensor/autograd 关键 citation（`node.h:112`、`variable.h:229-230`、`TensorImpl.h:510` 等）与源码一致。后续对抗式全量校验按「可信度分级」原则从略（写手已逐行核实）。

---

## 2026-06-15: 全模块分层与 NPU 分离审计修复(Workflow 编排 + 对抗式验证)

**Type**: Audit & Fix（Workflow:9 模块并发审计 → 每条发现对抗式验证 → 7 模块并发修复;铁律＝保留独有信息、只去重叠/迁移 NPU、不重命名）

**审计**:对 01_ai_frameworks 全部模块核查三项——冗余/雷同、overview→quick start→deep dive 分层、NPU↔upstream 分离;32 个 agent,每条可执行发现经对抗式验证(默认怀疑,排除「`input` 含 npu 子串」等假阳性),确认 16 条。

**修复（7 模块,纯增改无删除/重命名）**:
- **04_inductor**:`Pytorch_Compile_Debug_Analysis` 残留的 NPU 平台声明 + 脚本块(`DEVICE_TYPE=npu` / ASCEND/HCCL env)清除 → 纯 upstream(NPU 调试已在 [[npu_debug_guide]])
- **06_graphs**:`cuda/README` 移除混入的「NPU Graphs 对比/系统要求/参考资源」段 + 失实目录树 → 纯 CUDA,对比指向 [[comparison]];与 cuda/index(导航)分工
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
- 05_codegen_backends/mlir/npu：`npu_mlir_backend_deep_analysis` + `npu_mlir_pipeline_analysis`（65-75% 重叠）→ `NPU_MLIR_Backend_Technical_Analysis`（并入社区遵循/打破、三层 Pass、15 patch 分组、双通道 fallback、六阶段主线、演进建议；1400 行）
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

- fallback / patch 计数口径差异——本页（2.7 来源）fallback ~635 / patch 30+，本库 [[npu_inductor_splittiling_backend_analysis]]（v2.7.1 源码核查）fallback 859 / patch 35+；已在页内 `> [!contradiction]` 标注，深入以 v2.7.1 源码页为准

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

- `op_plugin/npu_operator_graph_eligibility_guide.md` —— 新增 §8「三关的硬性不变量:为什么进不去」(不变量层层收紧:第一关形状可预测 / 第二关计算可表达 / 第三关执行可录制;aclnn-only 铁律的 `aclopCompileAndExecute` 根因;`allow_internal_format=False` 为何救场;TORCH_NATIVE vs NPU_EXTRA 的「通用限制 vs 昇腾待补齐」之分;A2/A3-SIMD vs A5-SIMT 间接访存硬件根因)+ §9「面向新算子的前瞻判据」(决策树 Mermaid + 三关自检 checklist);原速查表顺延为 §10;目录补两项;新增 [[unbacked_symint_analysis]] 交叉引用

---

## 2026-06-12: 新增 op-plugin 算子接入域(3 篇 + 目录)——配置分类 / 注册链路 / 入图判别

**Type**: Knowledge Synthesis(源自 `E:\97-codes\pytorch\torch_npu` 当前 checkout 的多代理源码核查:op-plugin codegen、torchnpugen、_inductor、NPUGraph、_meta_registrations 等,逐条带 文件:行号 证据)

**新增文件**(`02_engineering/01_ai_frameworks/op_plugin/` 为新建目录,4 篇 `.md`):

- `op_plugin/index.md` —— 域入口:从 yaml 到入图的一图概览 + 三篇导航 + 「这一域回答什么」对照表
- `op_plugin/op_plugin_config_and_classification_guide.md` —— config 五文件字段;official/custom/symint(正交维度纠正)/quant;acl_op(aclop) vs op_api(aclnn);gen_opapi 结构化 vs 手写适配(「过适配」澄清);看一条 func 配置就分类的四维速查表
- `op_plugin/op_registration_pipeline_analysis.md` —— 两段 codegen 串联;生成产物(RegisterNPU.cpp/CustomRegisterSchema.cpp/custom_ops.py);**TORCH_LIBRARY=静态初始化「库加载即注册」**;编译期→加载期(import torch_npu 时 dlopen libtorch_npu.so 触发静态初始化)→运行期时间线;acl_op/op_api 运行时三层选择;official/custom 两条完整调用链
- `op_plugin/npu_operator_graph_eligibility_guide.md` —— 入图四路线总览;非 torchair 三关递进流水线(dynamo meta / inductor lowering+fallback / aclgraph aclnn-only 铁律);每关判别命令(TORCH_LOGS、has_kernel_for_dispatch_key、lowering.fallbacks、allow_internal_format);op_api/acl_op 贯穿主线

**索引与交叉引用**:

- `01_ai_frameworks/index.md` —— 子目录表新增 [[07_op_registration/npu/index]];页面列表新增「op-plugin 算子接入」区(3 行);页头摘要与最后更新改 2026-06-12
- 交叉引用:三篇互链,并 [[link]] 到既有 [[npu_compile_paths_overview]] / [[npu_inductor_splittiling_backend_analysis]] / [[aclgraph_deep_analysis]] / [[PyTorch_Dynamo_Technical_Analysis]] / [[npu_lowering_guide]]。入图判别页明确定位为「判别视角」,与既有「路径实现全景」页互补、不重复

---

## 2026-06-12: FSDP 深挖篇勘误——"分配 ≠ 新建":两层复用与社区机制(§5.5)

**Type**: Errata + Knowledge Synthesis(源码新核 5 处:`init_all_gather_outputs` 早退守卫、`alloc/free_storage`=`resize_`、`_set_unshard_async_op` 跨流碎片说明、`set_custom_all_gather`/`allocate()` 钩子、`set_allocate_memory_from_process_group`)

**更新文件**:

- `torchtitan/torchtitan_fsdp_prefetch_overlap_memory_analysis.md` —— ① §5.2 修正误导表述:"+p" 是显存占用增量而非"每次新分配"(逐参数 buffer 张量仅首迭代创建,此后 storage resize 0↔满;扁平 buffer 物理块稳态来自 caching allocator 池命中,无 cudaMalloc);② 新增 §5.5 勘误与补充:两层既有复用、FSDP 为何不自管持久池(allocator 等效/跨流 event/尺寸不齐/reserved 反升)、社区机制清单(storage-resize、expandable_segments、async_op 挪流、custom allocate 钩子、PG 缓冲注册、MemPool、compile 消 resize+copy、Megatron 持久缓冲先例)、自建持久池的场景判断(NPU 栈最值得);③ §7 复核表扩 5 行;页头日期更新

---

## 2026-06-11: torchtitan 系列新增两篇深挖伴篇(FSDP 预取/掩盖/显存、激活重计算 AC)

**Type**: Knowledge Synthesis(源自 torchtitan `cf3c4312` + PyTorch 2.9.1 源码逐行核验的问答整理稿,配 SVG→PNG 机制图)

**新增文件**(2 篇 `.md` + 16 个图文件,`torchtitan/assets/` 为新建目录):

- `02_engineering/02_train_frameworks/torchtitan/torchtitan_fsdp_prefetch_overlap_memory_analysis.md` —— [[torchtitan_fsdp_analysis]] 的深挖伴篇(2 图):串行 vs 多流预取掩盖时序、唯一跨流同步点 `wait_event(_fsdp_collectives.py:361)`、copy-in 三步(narrow 视图巧思 + `_foreach_copy_` 方向)、flat 双缓冲 ping-pong(为何延迟释放)、"完整参数 ≤2 份不会 3 份"的 reshard-先于-unshard 时序证明、CI/AG/CO 各阶段显存账
- `02_engineering/02_train_frameworks/torchtitan/torchtitan_ac_analysis.md` —— 激活重计算原理 + 代码解读(6 图):AC vs DCP 两种 checkpoint 区分、`checkpoint_wrapper` 接口链路(module 在两次 `next(gen)` 之间跑)、票据机制(`weak_holders`/`recomputed`/`recomp_counter` 下标对齐,发票→重算绑票→兑票)、SAC 双 dispatch mode 缓存回放 + torchtitan policy(奇偶 mm/SDPA/comm 恒存)+ attention 端到端走查、显存预估三法(full 手算 / SAC 加总 save-op / memory_budget Pareto)、粒度控制五法(含 config 驱动模块级方案)、横跨 autograd(`saved_tensors_hooks`)×dispatch(`TorchDispatchMode`)两核心、`ActivationCheckpointConfig` 全字段速查

**索引与交叉引用**:

- `torchtitan/index.md` —— 新增「深挖伴篇」表(2 行);系列篇数 7→9;并行施加管线 `apply_ac()` 挂链;Related Pages 补两页;最后更新 2026-06-11
- `torchtitan_fsdp_analysis.md` —— Related Pages 首行新增深挖伴篇反链
- `01_theory/02_pretraining/activation_checkpointing_analysis.md` —— Related Pages 首行新增 [[torchtitan_ac_analysis]](工程侧非重入/SAC,与该页 Megatron 重入路径互补)

---

## 2026-06-09: HTML 报告转 Markdown 并替换原 HTML(SVG/CSS 图 → PNG)

**Type**: 格式迁移 + 文件替换(为移动端阅读把 9 篇 HTML 报告转为 Markdown,图渲染为内嵌 PNG;转换验证无误后删除原 HTML,仓库只保留 Markdown)

**转换方式**: 用无头 Edge(`puppeteer-core`)加载每页 → 强制 light 配色并禁用 reveal 滚动动画 → 对每个图形元素按元素截图为 2× PNG(完整保留 CSS 变量配色与字体)→ DOM 规范化(callout→blockquote、TOC→列表、`<pre>`/`white-space:pre` 容器→围栏代码块、标题副标题、figure→`<img>`)→ Turndown + GFM 转 Markdown。图片存于各目录 `assets/`。

**新增文件**(9 篇 `.md` + 44 张 PNG,取代同名 `.html`):

- `02_engineering/02_train_frameworks/`:`async_collective_tensor_deep_dive.md`(4图)、`comm_compute_overlap_analysis.md`(7)、`deepseek_v4_context_parallel_analysis.md`(6)、`deepseek_v4_tensor_parallel_analysis.md`(1)、`distributed_optimizer_deep_dive.md`(7)、`megatron_pp_parallelism_analysis.md`(4)、`mindformers_moe_token_dispatcher_analysis.md`(7)、`muon_sharded_hsdp_report.md`(6)
- `02_engineering/05_gpu_kernel/`:`gpu_kernel_guide.md`(2,`tier-diagram` 与 FlashAttention `fa-flow` 两张 CSS 图)

**删除文件**: 上述 9 篇对应的 `.html` 原件(`async_collective_tensor_deep_dive.html` 等 8 篇 + `gpu_kernel_guide.html`)。

**索引与链接更新**:

- 全库 Obsidian 维基链接统一从 `[[*.html]]`(及 `[[*.html|别名]]`)改写为 `[[*]]`,共 13 处,分布于:`02_engineering/index.md`、`02_train_frameworks/index.md`、`05_gpu_kernel/index.md`、`torchtitan/index.md`、`megatron-lm/index.md`,以及 torchtitan `cp/ep/fsdp/pp/tp` 五篇分析页与 `megatron_distributed_optimizer_analysis.md` 的交叉引用
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

**交叉引用**: 与 [[torchtitan/torchtitan_ep_analysis]](token all-to-all dispatch/combine、DeepEP/HybridEP)、[[async_collective_tensor_deep_dive]](ACT 延迟 wait)、[[comm_compute_overlap_analysis]](DeepEP/HybridEP 通信掩盖)互为对照(MindSpore 静态图 vs PyTorch eager+compile)。

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

**交叉引用**: 三篇互链，并与 [[grpo_analysis]] / [[ppo_analysis]] / [[dapo_analysis]] / [[gspo_analysis]] / [[rlhf_foundations_analysis]] / [[kimi_k1.5_analysis]] / [[batch_invariance_guide]] / [[RL_PPO_Loss_and_GRPO_Analysis]] 等既有页交叉引用。

---

## 2026-05-22: torchtitan 多维并行体系源码级分析(7 篇)

**Type**: Knowledge Synthesis(基于 torchtitan `main` @ `cf3c4312` 与 PyTorch 2.9.1 FSDP2/DTensor/pipelining 内核的源码级分析)

**新增目录**: `wiki/02_engineering/02_train_frameworks/torchtitan/`

**新增文件**:

- `torchtitan/index.md` — torchtitan 多维并行知识地图:设计哲学(一组 GPU 多重视图)、三张 DeviceMesh、并行施加管线、组合建议
- `torchtitan/torchtitan_parallel_dims_analysis.md` — 并行基座:`ParallelDims` 维度约束、`build_mesh` 三张逻辑 mesh(dataloading/dense/sparse)、`fake` backend、mesh 查询接口
- `torchtitan/torchtitan_fsdp_analysis.md` — **标杆篇** DP/FSDP2:`FSDPParam` 逐参数切分、`FSDPParamGroup` 分组、all-gather 预取(隐式/显式)、五条 CUDA stream 异步编排、reduce-scatter 梯度规约、反向钩子链
- `torchtitan/torchtitan_tp_analysis.md` — TP:`distribute_tensor` 切分、`redistribute` 通信选择、列并行→行并行配对、Sequence Parallel、Async TP(`_micro_pipeline_tp` inductor pass)、Loss Parallel
- `torchtitan/torchtitan_cp_analysis.md` — CP:`_context_parallel_shard` 序列切分、HeadTail/PTRR 负载均衡、Ring Attention K/V 环形轮转、在线 softmax 合并、通信掩盖
- `torchtitan/torchtitan_pp_analysis.md` — PP:`_split_module` 模型切分、P2P send/recv、调度气泡对比(GPipe/1F1B/Interleaved/ZBV/DualPipeV)、action-based runtime、Zero Bubble(I/W 拆分)
- `torchtitan/torchtitan_ep_analysis.md` — EP:`ExpertParallel` 专家权重 `Shard(0)`、token all-to-all dispatch/combine、`AsyncCollectiveTensor` 延迟 wait、shared_experts 通信掩盖、DeepEP/HybridEP、`edp_mesh` FSDP

**统一分析粒度**: 每篇按 `fully_shard` 标杆粒度展开——参数/数据切分 → 通信原语 → 通信掩盖 → 异步实现 → 反向传播,带 `文件:行号` 引用与 ASCII 流程图。

**索引更新**:

- `wiki/02_engineering/02_train_frameworks/index.md` — 子目录表与页面列表加入 `torchtitan/index` 条目

**交叉引用**: torchtitan 系列与 Megatron-LM 源码级系列([[megatron_tp_analysis]]/[[megatron_cp_analysis]]/[[megatron_ep_analysis]]/[[megatron_pp_schedulers_analysis]]/[[megatron_ddp_optimizer_analysis]])互为对照(PyTorch-native vs CUDA/Megatron 生态),并与 [[async_collective_tensor_deep_dive]]、[[comm_compute_overlap_analysis]] 等既有页交叉引用。

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

- `wiki/02_engineering/02_train_frameworks/megatron-lm/megatron_nonuniform_tp_analysis.md`
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
  - 与存量 `distributed_optimizer_deep_dive.html`(Adam 18 字节)一致,矛盾消除。
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

- `wiki/02_engineering/02_train_frameworks/muon_sharded_hsdp_report.html`
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
- `01_theory/02_pretraining/muon_analysis.md` — Related Pages 新增回链

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

- `wiki/02_engineering/02_train_frameworks/megatron_pp_parallelism_analysis.html`
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

- `wiki/01_theory/01_models/deepseek/deepseek_v4_cp_analysis.md`
  - **修正 1**：源文件路径错误 — `raw/05_model_families/deepseek/DeepSeek_V4.pdf` → `raw/01_theory/01_models/deepseek/DeepSeek_V4.pdf`
  - **修正 2**：Stage 1 Step 3 压缩输出数量错误 — `(c+1) 个 compressed entries` → `1（CSA 重叠窗口）或 2（HCA 无重叠）个 boundary compressed entries`（原公式与 2c tokens / ratio c 的数学不一致）
  - **修正 3**：Stage 2 All-Gather 输出长度公式错误 — `总长度 = P × c`（与 S 无关的常数，量级完全错误）→ `总长度 ≈ S/c，即 P × S/(P·c)`

- `wiki/02_engineering/02_train_frameworks/deepseek_v4_context_parallel_analysis.html`
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
  - 从 `inductor/` 移入: `npu_mlir_backend_deep_analysis.md`、`npu_mlir_pipeline_analysis.md`、`NPU_MLIR_Backend_Technical_Analysis.md`
- **模型论文归位**:
  - `Engram_Analysis.md` → `deepseek/`
  - `moba_analysis.md`、`kimi_linear_analysis.md` → `moonshot_kimi/`
- **跨域移动**:
  - `comm_compute_fusion_guide.md` → `02_train_frameworks/`
  - `mooncake_analysis.md` → `03_infer_frameworks/`
  - `batch_invariance_guide.md` → `04_posttrain_frameworks/`
- **索引更新**: 所有受影响的 `index.md` 已同步更新

## 2026-05-15: DeepSeek-V4 Tensor Parallel 分析重大修正（基于 Megatron-LM dev 源码）

**Type**: Correction / Rewrite（基于实际源码的全面重写，纠正此前推断性分析中的重大错误）

**修正文件**:

- `wiki/02_engineering/02_train_frameworks/deepseek_v4_tensor_parallel_analysis.html`
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

- `wiki/02_engineering/02_train_frameworks/deepseek_v4_tensor_parallel_analysis.html`
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

- `wiki/02_engineering/02_train_frameworks/deepseek_v4_context_parallel_analysis.html`
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

- `02_train_frameworks/index.md` — 页面列表新增 deepseek_v4_context_parallel_analysis.html 条目

---

## 2026-05-14: DeepSeek-V4 Tensor Parallel 切分方案 HTML 深度分析

**Type**: Knowledge Synthesis（基于 Megatron-LM dev 分支实现 + V4 架构特性，新建 HTML 深度分析）

**入库文件**:

- `wiki/02_engineering/02_train_frameworks/deepseek_v4_tensor_parallel_analysis.html`
  - 8 节深度分析：V4 架构概览与 TP 必要性、CSA/HCA Attention 层 TP 列行并行策略、MoE ETP 切分（共享专家+路由专家）、mHC 流形约束超连接的切分特殊性、逐层通信量统一公式推导、TP Bulk vs Pipelined Overlap 掩盖方案、TP×EP×PP×CP 四维协同调度、配置决策树
  - 含 5 幅 SVG 图表：CSA TP 数据流、MoE Expert ETP 切分、Bulk Overlap 原理、四维并行通信组拓扑、TP 配置决策树

**交叉引用更新**:

- `02_train_frameworks/index.md` — 页面列表新增 deepseek_v4_tensor_parallel_analysis.html 条目

---

## 2026-05-14: 分布式优化器深度分析 HTML 入库

**Type**: Ingestion（HTML 深度分析文档入库，无需新建 .md）

**入库文件**:

- `wiki/02_engineering/02_train_frameworks/distributed_optimizer_deep_dive.html`
  - 7 节深度分析：ZeRO 分片体系通信量等价性、梯度累积对 ZeRO-1/2 的差异化影响 (K×P)、FSDP2/Megatron/MindSpeed 三方对比、MindSpeed param 临时化与 zero-copy、Adam vs Muon 优化器内存估算 (18→14 bytes/param)、Muon Newton-Schulz 对 ZeRO 切分的根本性挑战、选型决策树
  - 含 6 幅 SVG 图表：DDP vs ZeRO 通信量、梯度累积通信差异、Overlap 机制对比、MindSpeed 内存布局、Element-wise vs 矩阵运算、选型决策树

**交叉引用更新**:

- `megatron-lm/index.md` — Memory & Compute Optimization 节新增 HTML 文件条目
- `megatron-lm/megatron_distributed_optimizer_analysis.md` — Related Pages 新增链接
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

- `wiki/02_engineering/01_ai_frameworks/comm_compute_fusion_guide.md`
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
  - `wiki/01_theory/01_models/deepseek/deepseek_v4_analysis.md`（WaveEP、TileLang、DeepEP）
  - `wiki/02_engineering/02_train_frameworks/megatron-lm/moe_training_optimization_report.md`（DeepEP/HybridEP）
  - `wiki/02_engineering/02_train_frameworks/megatron-lm/megatron_comm_overlap_analysis.md`（多维通算重叠）
  - MLIR 官方文档（Mesh Dialect RFC，IREE，StableHLO）
  - Triton GitHub（triton-lang/triton MLIR 迁移 PR）
  - MindSpore 官方文档（ParallelAuto，AKG，ANF IR）

---

## 2026-05-12: Megatron-LM MoE 训练优化技术全景分析

**Type**: Knowledge Synthesis + Research（源码级分析, 新建 3 个 Wiki 页面）

- **Created**:
  - `wiki/02_engineering/02_train_frameworks/megatron-lm/megatron_distributed_optimizer_analysis.md` — 分布式优化器深度分析（ZeRO-1/2 分片机制, Reduce-Scatter/All-Gather 通信, FP8/FP4 量化参数, CPU Offloading 双模式）
  - `wiki/02_engineering/02_train_frameworks/megatron-lm/megatron_memory_optimization_analysis.md` — 显存优化全景分析（NCCL Pool, MoE Paged Stash 三级溢出, Fine-Grained Activation Offloading, Buffer 复用, FP8/FP4 精度, Resharding）
  - `wiki/02_engineering/02_train_frameworks/megatron-lm/megatron_fusion_operators_analysis.md` — 融合算子优化分析（Bias+Activation 融合 6 种, Fused LayerNorm/Softmax, MoE 专用融合 4 种, Communication Fusion, FP8 Input Store, Triton/CUTLASS/cuTile kernel 层次）
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

- Cross-referenced with `[[inductor_codegen_dynamic_shape_analysis]]`, `[[torch_compile_architecture]]`, `[[PyTorch_Dynamo_Technical_Analysis]]`

---

## 2026-05-11: PyTorch Inductor 端到端编译管线源码分析

**Type**: Knowledge Synthesis（PyTorch 主分支源码级调研）

- **Created**: `wiki/02_engineering/01_ai_frameworks/inductor/inductor_compiler_pipeline_analysis.md` — PyTorch Inductor 后端编译流程深度分析（中文）
- **Updated**: `wiki/02_engineering/01_ai_frameworks/inductor/index.md` — 架构与流程表格新增条目
- **Cross-referenced**: 新页面与现有 10 个分阶段分析页面建立双向链接（`[[aotautograd_analysis]]`, `[[pre_grad_passes_guide]]`, `[[joint_graph_passes_guide]]`, `[[post_grad_passes_guide]]`, `[[lowering_analysis]]`, `[[scheduler_analysis]]`, `[[inductor_codegen_analysis]]`, `[[PyTorch_Dynamo_Technical_Analysis]]`, `[[PyTorch_Inductor_Technical_Analysis]]`, `[[torch_compile_architecture]]`）

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
- 更新 `inductor/index.md`、`01_ai_frameworks/index.md`、`NPU_MLIR_Backend_Technical_Analysis.md`、`npu_lowering_guide.md` 交叉引用

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
- 更新 `wiki/02_engineering/01_ai_frameworks/index.md`、`inductor/index.md` 和 `NPU_MLIR_Backend_Technical_Analysis.md` 的交叉引用

## 2026-05-08: 训练/推理框架目录页创建

- 新建 `wiki/llm/10_train_framework/index.md`（对应 `raw/10_train_framework/`：megatron.eddx, mindformers.eddx）
- 新建 `wiki/llm/11_infer_framework/index.md`（对应 `raw/11_infer_framework/`，当前为空）
- 更新 `wiki/llm/index.md`、`wiki/index.md`、`CLAUDE.md` 目录结构

## 2026-05-06: GLM/GLM-5 技术路线摄入

**Type**: Source Ingestion (GLM Series)

### 下载的新 Raw 文件

- `raw/05_model_families/zhipu_glm/GLM-5_Vibe_Coding_to_Agentic_Engineering-2602.15763.pdf`

### 创建的 Wiki 页面

- **Created**: `wiki/llm/05_model_families/zhipu_glm/glm_5_analysis.md` — GLM-5 Vibe Coding 到 Agentic Engineering（中文）
- **Created**: `wiki/llm/05_model_families/zhipu_glm/glm_5v_turbo_analysis.md` — GLM-5V-Turbo 原生多模态 Agent（中文）
- **Created**: `wiki/llm/05_model_families/zhipu_glm/index.md` — GLM 技术路线总览

**Key topics (glm_5_analysis)**:
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

**Key topics (glm_5v_turbo_analysis)**:
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

- **Created**: `wiki/llm/05_model_families/moonshot_kimi/kimi_k2_analysis.md` — Kimi K2 开放 Agent 智能（中文）
- **Created**: `wiki/llm/05_model_families/moonshot_kimi/kimi_k2.5_analysis.md` — Kimi K2.5 视觉 Agent 智能（中文）
- **Updated**: `wiki/llm/05_model_families/moonshot_kimi/index.md` — 论文索引更新，K2/K2.5 标记为已摄入

**Key topics (kimi_k2_analysis)**:
  - 1.04T/32.6B MoE，384 专家 (sparsity=48)，64 注意力头
  - MuonClip 优化器：QK-Clip 解决 logits 爆炸，15.5T token 零 loss spike
  - 稀疏度扩展定律：sparsity 48 vs 8 节省 1.69× FLOPs
  - 大规模 Agentic 数据合成：23,000+ 工具，模拟+真实沙盒
  - RL 框架：RLVR + 自批评 Rubric 奖励，覆盖可验证和主观任务
  - SWE-bench 65.8、τ²-Bench 66.1、AIME 2024 69.6
  - Agent 能力超越 Claude Opus 4 和 GPT-4.1

**Key topics (kimi_k2.5_analysis)**:
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
- **Created**: `wiki/llm/01_architecture/moba_analysis.md` — MoBA 混合块注意力机制（中文）
- **Created**: `wiki/llm/01_architecture/kimi_linear_analysis.md` — Kimi Linear/KDA 线性注意力架构（中文）
- **Created**: `wiki/llm/03_alignment/kimi_k1.5_analysis.md` — Kimi k1.5 RL 缩放定律（中文）
- **Created**: `wiki/llm/05_model_families/moonshot_kimi/index.md` — Kimi 技术路线总览

**Key topics (mooncake_analysis)**:
  - Prefill/Decode/KVCache 三池分离架构
  - Chunked Pipeline Parallelism (CPP) 替代跨节点 SP
  - Layer-wise Prefill：KVCache 传输与计算重叠
  - 缓存感知全局调度 + 热点块迁移
  - 预测性早期拒绝解决负载波动
  - 真实负载吞吐量提升 75%，模拟场景 525%

**Key topics (moba_analysis)**:
  - 将 MoE 原理应用于注意力机制
  - Query 动态路由到 KV Block (top-k 选择)
  - 块路由：mean_pool(K) 亲和度 + 因果掩码
  - MoBA/Full 混合预训练 (90%/10%)
  - 1M 序列 6.5x 加速，10M 序列 16x 加速
  - 已部署支持 Kimi 长上下文请求

**Key topics (kimi_linear_analysis)**:
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

- **Created**: `wiki/llm/02_training/low_precision_training_analysis.md` — Megatron 低精度训练全栈分析（中文）
- **Created**: `wiki/llm/02_training/transformer_engine_analysis.md` — NVIDIA Transformer Engine 技术分析（中文）
- **Updated**: `wiki/llm/index.md` — Optimizers & Training Algorithms 表格新增 3 条目
- **Updated**: `wiki/llm/06_infra/megatron-lm/index.md` — Knowledge Gaps 更新（TE 集成、低精度训练标记为已解决），Cross-Domain Links 扩展

**Key topics (low_precision_training_analysis)**:
  - 精度格式全览（FP32 → BF16 → FP16 → FP8 → MXFP8 → FP4）
  - 五种 FP8 Recipe（tensorwise/delayed/blockwise/mxfp8/custom）及对比
  - FP8 Primary Weights（fp8_param_gather）显存节省分析（6N → 5N bytes）
  - first_last_layers_bf16 首末层 BF16 保护机制
  - TP 通信与 FP8 协同（User Buffer, Pipelined/Bulk Overlap）
  - FP4 QAT（DeepSeek-V4 方案）：无损反量化原理、STE 训练、推理部署
  - MoE + 低精度（Grouped GEMM FP8, Router Fusion, DeepEP A2A）
  - Scaling MoE 论文精度实践总结
  - 配置速查表

**Key topics (transformer_engine_analysis)**:
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

- **Created**: `wiki/llm/05_model_families/deepseek/deepseek_v4_fp4_qat_analysis.md` — FP4 QAT 完整分析（全新主题）
- **Moved (3 files)**:
  - `deepseek_v4_architecture_diagrams.md` — V4 架构 ASCII 结构图（50KB 补充参考）
  - `deepseek_v4_implementation_details.md` — V4 核心组件伪代码实现（34KB 补充参考）
  - `deepseek_v4_technical_deep_dive.md` — CSA/HCA/DSA/MLA 对比深度解析（42KB 补充参考）
- **Updated (merged unique content)**:
  - `deepseek_v4_analysis.md` — 新增 §Compressed KV 数值示例、DualPath 推理框架、Think Modes、Pro-Max 评测
  - `mHC.md` — 扩展 §动态与静态系数（完整公式 3-8、对比表、训练细节）
  - `deepseek_v4_cp_analysis.md` — 新增 §9 实现细节（Fused Select-and-Pad、Top-K Selector、传统 CP 对比表）
- **Cross-references**: 所有新/更新页面双向链接已更新

---

## 2026-04-29: Activation Checkpointing（重计算）完整分析

**Type**: Knowledge Synthesis（PyTorch autograd 机制 + Megatron-LM 源码分析）

- **Created**: `wiki/llm/02_training/activation_checkpointing_analysis.md` — 激活重计算完整分析（中文）
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
- **Created**: `wiki/llm/05_model_families/deepseek/deepseek_v4_cp_analysis.md` — DeepSeek-V4 Context Parallelism 深度分析（中文）
- **Updated**: `wiki/llm/05_model_families/deepseek/deepseek_v4_analysis.md` — CP 节扩展并添加指向新页面链接
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

- Created `wiki/llm/muon_analysis.md` — Source: `raw/MUON IS SCALABLE FOR LLM TRAINING-2502.16982v1.pdf`
- Created `wiki/megatron-lm/Megatron_LM_TFLOPS_Analysis.md` — TFLOPS estimation methodology

### ~2026-01: DeepSeek & Memory Architectures

- Created `wiki/llm/Engram_Analysis.md` — Source: `raw/Engram_paper.pdf`
- Created `wiki/llm/deepseek_math_v2.md` — Self-verifiable math reasoning

### ~2025-12: Weight Initialization & KIMI

- Created `wiki/llm/llm_initiliaze_analysis.md` — Dense & MoE initialization

---

## 2026-04-17: mHC Source Ingestion

**Type**: Source Ingestion

- **Source**: `raw/mHC-2512.24880v2.pdf` (DeepSeek-AI, arXiv:2512.24880v2)
- **Created**: `wiki/llm/mHC.md` — Manifold-Constrained Hyper-Connections analysis (in Chinese)
- **Updated**: `wiki/llm/index.md` — Added mHC entry and cross-domain links
- **Cross-referenced**: Added backlinks to `muon_analysis.md`, `llm_initiliaze_analysis.md`, `Megatron-LM_MoE_Zero_Redundancy_Analysis.md`
- **Key topics**: doubly stochastic matrix, Sinkhorn-Knopp projection, residual stream expansion, DeepSeek-V3 MoE, kernel fusion, selective recomputing

### ~2025-11: Training-Inference Integration

- Created `wiki/megatron-lm/Megatron_vLLM_Weight_Sync_Analysis.md` — verl Megatron + vLLM weight sync

### ~2025-10: Torch Compile & NPU

- Created `wiki/torch_compile/inductor/` — 17 pages covering Dynamo, AOT Autograd, Inductor, NPU backends
- Created `wiki/torch_compile/cudagraphs/` — CUDA Graphs guides and NPU Graphs deep dives

## 2026-04-20: DeepSeek Model Family Batch Ingestion (Part 1/4)

**Type**: Source Ingestion

- **Source**: `raw/05_model_families/deepseek/DeepSeek_LLM-2401.02954.pdf` (DeepSeek-AI, arXiv:2401.02954)
- **Created**: `wiki/llm/deepseek_llm_analysis.md` — DeepSeek LLM analysis
- **Updated**: `wiki/llm/index.md` — Added DeepSeek model family section
- **Key topics**: scaling laws with non-embedding FLOPs/token representation, data quality impact on model/data allocation, multi-step LR scheduler, GQA, bilingual pre-training, SFT+DPO alignment

- **Source**: `raw/05_model_families/deepseek/DeepSeek_V2-2405.04434.pdf` (DeepSeek-AI, arXiv:2405.04434)
- **Created**: `wiki/llm/deepseek_v2_analysis.md` — DeepSeek-V2 analysis
- **Key topics**: MLA (Multi-head Latent Attention), low-rank KV joint compression, decoupled RoPE, DeepSeekMoE, device-limited routing, three-level auxiliary losses, token dropping, GRPO, two-stage RL

- **Source**: `raw/05_model_families/deepseek/DeepSeek_V3-2412.19437.pdf` (DeepSeek-AI, arXiv:2412.19437)
- **Created**: `wiki/llm/deepseek_v3_analysis.md` — DeepSeek-V3 analysis
- **Key topics**: FP8 mixed precision training, fine-grained quantization (tile/block-wise), DualPipe pipeline parallelism, auxiliary-loss-free load balancing, Multi-Token Prediction (MTP), cross-node all-to-all communication kernels, inference deployment with redundant experts, R1 distillation

- **Source**: `raw/05_model_families/deepseek/DeepSeek_R1-2501.12948.pdf` (DeepSeek-AI, arXiv:2501.12948)
- **Created**: `wiki/llm/deepseek_r1_analysis.md` — DeepSeek-R1 analysis
- **Key topics**: pure RL reasoning without SFT, GRPO, emergent self-verification/reflection, "aha moment", multi-stage pipeline (cold start → RL → SFT → RL), distillation to Qwen/Llama, rule-based rewards, language consistency reward

**Remaining**: Coder, Coder-V2, Math, MoE, Prover, VL

---

## 2026-04-24: DeepSeek-V4 Source Ingestion

**Type**: Source Ingestion

- **Source**: `raw/05_model_families/deepseek/DeepSeek_V4.pdf` (DeepSeek-AI, 2025)
- **Created**: `wiki/llm/deepseek_v4_analysis.md` — DeepSeek-V4 analysis (in Chinese)
- **Updated**: `wiki/llm/index.md` — Added V4 to DeepSeek model family section
- **Updated**: `wiki/llm/deepseek_v3_analysis.md` — Added backlink to V4
- **Updated**: `wiki/llm/deepseek_v2_analysis.md` — Added backlink to V4
- **Cross-referenced**: `mHC.md`, `muon_analysis.md`, `deepseek_v3_analysis.md`, `deepseek_v2_analysis.md`
- **Key topics**: CSA (Compressed Sparse Attention), HCA (Heavily Compressed Attention), hybrid attention architecture, DSA (DeepSeek Sparse Attention), Lightning Indexer, million-token context, mHC integration, Muon optimizer, Anticipatory Routing, SwiGLU clamping, wave-based EP overlap, TileLang kernels, FP4 QAT, heterogeneous KV cache management, on-disk KV cache storage

---

## 2026-04-21: DeepSeek Model Family Batch Ingestion (Part 2/4)

**Type**: Source Ingestion

- **Source**: `raw/05_model_families/deepseek/DeepSeek_Coder-2401.14196.pdf` (DeepSeek-AI, arXiv:2401.14196)
- **Created**: `wiki/llm/deepseek_coder_analysis.md` — DeepSeek-Coder analysis
- **Key topics**: repository-level code corpus, dependency parsing, topological sort, Fill-in-the-Middle (FIM), 87 programming languages, 16K context, GQA

- **Source**: `raw/05_model_families/deepseek/DeepSeek_Coder_V2-2406.11931.pdf` (DeepSeek-AI, arXiv:2406.11931)
- **Created**: `wiki/llm/deepseek_coder_v2_analysis.md` — DeepSeek-Coder-V2 analysis
- **Key topics**: MoE code model, 338 languages, 128K context, 6T additional tokens, YaRN extension, GRPO with reward model, SWE-bench >10%

- **Source**: `raw/05_model_families/deepseek/DeepSeek_Math-2402.03300.pdf` (DeepSeek-AI, arXiv:2402.03300)
- **Created**: `wiki/llm/deepseek_math_analysis.md` — DeepSeekMath analysis
- **Key topics**: 120B math tokens from Common Crawl, iterative fastText pipeline, GRPO origin, unified RL paradigm, MATH 51.7%

- **Source**: `raw/05_model_families/deepseek/DeepSeek_MoE-2401.06066.pdf` (DeepSeek-AI, arXiv:2401.06066)
- **Created**: `wiki/llm/deepseek_moe_analysis.md` — DeepSeekMoE architecture analysis
- **Key topics**: fine-grained expert segmentation, shared expert isolation, expert-level/device-level balance loss, 2B/16B/145B scales

- **Source**: `raw/05_model_families/deepseek/DeepSeek_Prover-2408.08152.pdf` (DeepSeek-AI, arXiv:2408.08152)
- **Created**: `wiki/llm/deepseek_prover_analysis.md` — DeepSeek-Prover-V1.5 analysis
- **Key topics**: Lean 4 theorem proving, truncate-and-resume mechanism, RMaxTS Monte-Carlo tree search, thought-augmented proofs, RLPAF

- **Source**: `raw/05_model_families/deepseek/DeepSeek_VL-2403.05525.pdf` (DeepSeek-AI, arXiv:2403.05525)
- **Created**: `wiki/llm/deepseek_vl_analysis.md` — DeepSeek-VL analysis
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
