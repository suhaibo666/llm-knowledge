# D 组独立审阅：采样、投机解码、约束解码

> **状态**：第一版（进行中）。已完成全部来源开读与主要结论；算例脚本复算、SVG 渲染目检与 T0 检查在后续版本补记。
> **审阅者**：独立审阅（非作者、非本批协调者）。审阅日期 2026-09-18。
> **页面**（均在 `wiki/01_theory/05_inference/`）：`17_sampling_decoding_analysis.md`（含 `assets/17_sampling_decoding.svg/.mjs`）、`23_speculative_decoding_analysis.md`、`24_speculative_decoding_variants_analysis.md`、`32_constrained_decoding_analysis.md`。

## 1. 已打开的来源（固定版本）

| 来源 | 版本 | 打开方式 | 用于 |
|---|---|---|---|
| Holtzman 等，The Curious Case of Neural Text Degeneration | arXiv:1904.09751v2 | PDF 逐页文本 | 17 |
| Hugging Face Transformers | v4.57.1 = commit `8cb5963cc22174954e7dca2c0a3320b7dc2f4edc`（GitHub API 核对：annotated tag v4.57.1 → 该 commit，2025-10-14，"Release: v4.57.1"） | raw.githubusercontent.com 下载 `generation/logits_process.py`（3231 行）、`beam_search.py`（1002 行）、`stopping_criteria.py`、`utils.py`、`configuration_utils.py` | 17 |
| Leviathan、Kalman、Matias，Speculative Decoding | arXiv:2211.17192v2 | PDF 逐页文本（共享缓存） | 23、24 |
| Draft & Verify | arXiv:2309.08168v2 | PDF 逐页文本 | 24 |
| Medusa | arXiv:2401.10774v1 | PDF 逐页文本 | 24 |
| EAGLE | arXiv:2401.15077v1 | PDF 逐页文本 | 24 |
| SpecInfer | arXiv:2305.09781v4 | PDF 逐页文本 | 24 |
| Willard & Louf，Efficient Guided Generation | arXiv:2307.09702v1 | PDF 逐页文本 | 32 |
| XGrammar | arXiv:2411.15100v1 | PDF 逐页文本 | 32 |

`raw/01_theory/05_inference/` 下页首引用的来源索引全部存在：`The_Curious_Case_of_Neural_Text_Degeneration-1904.09751.md`、`Hugging_Face_Transformers_Generation-v4.57.1.md`、`Speculative_Decoding-2211.17192.md`、`Draft_and_Verify-2309.08168.md`、`Medusa-2401.10774.md`、`EAGLE-2401.15077.md`、`SpecInfer-2305.09781.md`、`Efficient_Guided_Generation-2307.09702.md`、`XGrammar-2411.15100.md`。

## 2. 结论表

| page | beat2 | delete-code | figure-trigger | algorithm-replay | spot-check | verdict | note |
|---|---|---|---|---|---|---|---|
| 17_sampling_decoding_analysis | pass | pass | transform（softmax/温度/截断）+ state（beam） | pass（SVG 随机支路措辞需改） | 2/3（beam 源码锚点失效且指向已弃用类；温度锚点不支持 T=0 转 greedy） | REJECT | 最小失败单元：§4 beam 的 Transformers 源码证据（P1）；其余为 P2 |
| 23_speculative_decoding_analysis | pass | pass | transform（接受/拒绝/残差）+ state（KV 提交） | pass | 3/3 | PASS（带 P2） | 公式、记号、算例与 Leviathan v2 一致 |
| 24_speculative_decoding_variants_analysis | FAIL §1/§3（D&V、EAGLE 的正确性口径误述） | pass | layout（树 mask/位置）+ transform（验证） | FAIL §2 | 1/3（D&V 口径与论文不符；EAGLE §2.2 不存在且漏树验证；Medusa §3.1.2/§3.3.1/§3.3.3 相符） | REJECT | P0：D&V 被写成仅 greedy；P1：EAGLE 验证描述；P1：树 mask/位置/输出未重放 |
| 32_constrained_decoding_analysis | 部分（§2 缺“逐步 mask≠全局条件分布”） | pass | transform（mask/归一化）+ state（解析状态） | pass（部分：图未含被禁 `"}`/EOS 与概率） | 3/3 | REJECT | 最小失败单元：§2 缺分布失真说明（P1）；其余 P2 |

## 3. 逐页发现

### 3.1 `17_sampling_decoding_analysis.md`

**P1-17-1（已确认）beam search 的源码证据锚点失效，且指向固定版本中已弃用、不在 `generate()` 实际路径上的类。**
- 位置：`wiki/01_theory/05_inference/17_sampling_decoding_analysis.md:140`，“这不是必须使用的统一公式，而是 Transformers 固定版本 `BeamHypotheses.add` 的实现形式。其 `BeamSearchScorer.process` 先按本轮候选排名扫描……[来源：`BeamSearchScorer.process`](…/beam_search.py#L2639-L2646)”；`:153` “按该固定 Transformers 实现，本轮排名前二的是 `AA`、`AB`……”。
- 证据：在 `8cb5963c` 下 `src/transformers/generation/beam_search.py` 只有 1002 行，`#L2639-L2646` 不存在（`BeamSearchScorer.process` 实际在 L219–L322，EOS 排名门在 L272–L281）。`BeamSearchScorer.__init__`（L175–L177）与 `BeamHypotheses.__init__`（L921–L923）都会记录 “is deprecated and will be removed in v4.62.0”。该版本 `utils.py` 的 `GENERATION_MODES_MAPPING`（L132–L143）把 `BEAM_SEARCH`/`BEAM_SAMPLE` 映射到 `GenerationMixin._beam_search`；实际路径是 `_get_top_k_continuations` → `_get_running_beams_for_next_iteration` → `_update_finished_beams`：`top_num_beam_mask` 定义在 L3191，使用在 L3081（`did_top_num_beams_just_finished = next_token_hits_stopping_criteria & top_num_beam_mask[None, :]`），长度惩罚在 L3085（`topk_log_probs / ((cur_len + 1 - decoder_prompt_len) ** length_penalty)`）；`early_stopping` 的 True/False/"never" 语义在 `_check_early_stop_heuristic`（L2911 起）与 `_beam_search_has_unfinished_sequences`（L2958 起）。
- 影响：页面描述的**行为**（只有前 $B$ 名的 EOS 进入完成集合、$S/t^\alpha$、三种早停语义）在实际路径上同样成立，算例结论不变；问题在于证据：一个载荷性源码引用的行号区间不存在，而且把已弃用类当成“固定版本的实现”。
- 建议：锚点改为 `generation/utils.py::GenerationMixin._beam_search`、`::_update_finished_beams`（`top_num_beam_mask`）与 `::_check_early_stop_heuristic`；若保留 `BeamSearchScorer`/`BeamHypotheses`，注明它们在 v4.57.1 已弃用、`generate()` 不经过它们，并修正 `beam_search.py` 的行号。

**P2-17-2（已确认）“$T=0$ 转到 greedy”与所引来源不符。**
- 位置：`17_sampling_decoding_analysis.md:20`，“$T=0$ 不是上式的合法除法，而是转到 greedy 的离散选择。[来源：`TemperatureLogitsWarper`]”。
- 证据：`logits_process.py` `TemperatureLogitsWarper.__init__`（L277–L286）在 temperature 不是严格正的 float 时抛 `ValueError`，对 0.0 追加提示 “If you're looking for greedy decoding strategies, set `do_sample=False`”——被引符号是**拒绝** $T=0$，不是转换。另一处 `utils.py` L1782–L1785（`_prepare_generation_config` 合并模型默认值的分支）有 “edge case”：`temperature == 0.0` 时把 `do_sample` 设为 False——只在该配置路径成立。
- 建议：写成“该版本的 warper 拒绝 $T\le0$，要求显式 `do_sample=False` 走 greedy；在合并模型默认生成配置的路径中，`temperature=0.0` 会被改成 `do_sample=False`；许多服务引擎把 $T=0$ 约定为 greedy”。

**P2-17-3（已确认）SVG 随机支路把候选集写成“抽到”。**
- 位置：`assets/17_sampling_decoding.svg` 中间下栏文字“候选：截断后集合；抽到 A、B”（由 `assets/17_sampling_decoding.mjs:117` 生成）。
- 问题：一次抽样只输出一枚 token；“抽到 A、B”读起来像两枚都被抽出。应为“只可能抽到 A 或 B（概率 0.731/0.269）”。

**P2-17-4（已确认）处理器顺序只说“以引擎为准”，但本页已固定的 v4.57.1 有明确顺序，且温度与 top-p 的交互是基础直觉。**
- 位置：`17_sampling_decoding_analysis.md:92`。
- 证据：`utils.py::_get_logits_processor`（约 L1150–L1316）：重复惩罚、禁词、min-length/min-new-tokens、suppress 等 processor 在前；`do_sample` 时 warper 依次为 temperature → top-k → top-p → min-p → typical → epsilon → eta；`TopPLogitsWarper` 在（可能已被 top-k 置 $-\infty$ 的）分数上重新 softmax 后判定。
- 建议：给出该固定版本的顺序作为一个实例；补一句“top-k 候选集不随温度变，top-p 候选集随温度变”：本页数值下 $T=0.5$、$p=0.8$ 只保留 A（0.864704≥0.8），$T=1$ 保留 A、B。

**P2-17-5（已确认）可复现性段落未点明 batch 组成与 greedy 同样受影响。**
- 位置：`17_sampling_decoding_analysis.md:159` 与 `:166`（表中 greedy 的“不能保证”一格只列“全序列最优或多样性”）。
- 问题：段落已正确指出 seed 只固定随机流起点、数值/硬件/并行归约会改变 logits 末位；但服务场景最常见的不确定来源是**同一请求在不同 batch 组成/大小下归约顺序不同**，这会同样翻转 greedy 的近似同分 argmax。建议在段落与表中 greedy 一栏补充“跨 batch/硬件逐 token 一致性”。

### 3.2 `23_speculative_decoding_analysis.md`

记号与论文一致：$p$=目标（$M_p$），$q$=草稿（$M_q$），接受 $\min(1,p/q)$、残差 $\mathrm{norm}(\max(0,p-q))$、全接受后从 $p_{\gamma+1}$ 取一枚、每轮 1 到 $\gamma+1$ 枚、$E[N]=(1-\alpha^{\gamma+1})/(1-\alpha)$、Theorem 3.8 的 $(1-\alpha^{\gamma+1})/((1-\alpha)(\gamma c+1))$ 均与 Leviathan v2 §2.1、§2.3 Algorithm 1、§3.1 Eq. (1)、§3.3 Theorem 3.8、Appendix A.1 一致。KV 提交表的逻辑结论正确。未发现 P0/P1。

**P2-23-1（已确认）未写出 $\beta=\sum_x\min(p,q)$，证明中的 0.30 也未说明就是拒绝概率。**
- 位置：`23_speculative_decoding_analysis.md:69`（`0.30\,r_2(v)`）、`:90`（“令 $\alpha$ 为各位置接受概率的共同均值”）。
- 证据：论文 Theorem 3.5 $\beta=\sum_x\min(p(x),q(x))$，Corollary 3.6 $\alpha=E(\min(p,q))$；Appendix A.1 指出残差归一化常数恰为 $1-\beta$。本例 $\sum\min(p_2,q_2)=0.70$，拒绝概率 $q_2(B)\times(1-0.4)=0.30$，恰等于残差总质量。页面直接把 0.30 当作拒绝概率系数，读者需自行推出这一关键等式；§5 的 $\alpha$ 也没有与 $p,q$ 连起来。
- 建议：在 §3 补一句“位置 2 的接受概率 $\beta=\sum\min=0.70$，拒绝概率 $1-\beta=0.30$ 恰是残差归一化常数”，并在 §5 说明 $\alpha=E[\beta]$。

**P2-23-2（已确认）$q$ 的条件写得比证明需要的更严。**
- 位置：`23_speculative_decoding_analysis.md:32`，“$p$ 和 $q$ 必须都是同一请求所要求的最终抽样规则处理后的分布”。
- 证据：论文 §2.2 为简化假定 $p,q$ 都已按抽样方法调整；但 Appendix A.1 对**任意** $q$ 成立（§3.6：“for any choice of approximation model $M_q$ without restriction”）。精确性只要求比值与残差中用的 $q$ 就是草稿实际抽样所用的分布；例如 greedy 草稿（one-hot $q$）配随机目标依然精确。对 $p$ 的要求（必须是目标最终处理后的分布）写得正确。
- 建议：改为“$p$ 必须是目标请求最终处理后的分布；$q$ 必须与草稿实际抽样所用分布一致（论文为简化假定两者按同一抽样方法调整）”。

**P2-23-3（已确认）KV 表的“本轮开始”状态只适用于 prefill 后的第一轮。**
- 位置：`23_speculative_decoding_analysis.md:78`、`:82` 与 `:84`，以及图中“目标并行求 p1 至 p4”。
- 问题：页面假定本轮开始时已有 $h$ 的全部 KV，验证只算 A、B、C 的 KV；而本轮结束时发布 $h,A,C$、KV 只有 $h,A$。所以下一轮开始时 KV 比已发布序列少一枚，与本轮假设不同。稳态下，验证前向的输入是“最后一枚已提交 token + $\gamma$ 枚草稿”，共 $\gamma+1$ 个位置，产出 $p_1..p_{\gamma+1}$；本页的起始假设只在 prefill 已给出 $p_1$ 的首轮成立。
- 建议：补一句稳态口径，避免读者把两种起点混用。

**P2-23-4（建议）** Mermaid 已重放控制路径、输出与 KV 边界；可在节点上标出 $a_2(B)=0.4$、$u_2=0.7$、$r_2=(5/6,0,1/6)$，并写出“$\min+$残差$=p_2$”这一不变量，使图自身可复算。

### 3.3 `24_speculative_decoding_variants_analysis.md`

**P0-24-1（已确认）把 Draft & Verify 的正确性口径写成仅 greedy，与固定版本论文不符（遗漏导致的误述）。**
- 位置：`24_speculative_decoding_variants_analysis.md:19`（表格“论文中与目标的关系”：“Draft & Verify 的 Algorithm 2 明确是 **greedy**，验证时用完整模型 `argmax` 修正”）、`:24`（“保证的是其 deterministic greedy 输出；若把它改成温度采样，不能沿用同一个证明”）、`:75`（“不改变其完整模型 greedy 校验的正确性口径”）；`raw/01_theory/05_inference/Draft_and_Verify-2309.08168.md` 的说明也沿用此口径。
- 证据（2309.08168v2）：§3.2 原文在介绍 Algorithm 2 后紧接着说完整的**基于采样**的解码过程见 Appendix K；§4.2 的主实验同时报告 greedy（temperature 0.0）与随机采样（temperature 0.2/0.6）；Appendix K Algorithm 4 用 $r\ge\min(1,p_{\text{full}}/p_{\text{draft}})$ 拒绝，并从 $\mathrm{norm}(\max(0,p_{\text{full}}-p_{\text{draft}}))$ 重抽，即 speculative sampling；Appendix J.2 声称采样设置下输出与自回归解码**同分布**（引 Leviathan 的证明）；Appendix B 给出采样用 top_p 0.85/0.95。
- 影响：本页主旨是“草稿来源与验证规则配对”，D&V 是其中的主要例子。读者会误以为跳层自投机只有 greedy 精确保证、采样没有论文支撑。每句单看都在描述 Algorithm 2，但表格那一栏和 §4 的定性是对**整篇论文**保证范围的误述。
- 建议：改为“Algorithm 2（正文）为 greedy 的逐位置 argmax 校验；Appendix K Algorithm 4 用 speculative sampling（$\min(1,p/q)$ 接受 + 正差残差），论文据此声称采样输出同分布；§3.4 的草稿提前退出只影响草稿成本”，并同步修正 raw 索引。

**P1-24-2（已确认）EAGLE 的验证描述：所引 §2.2 不存在，且遗漏了 EAGLE 的树草稿与逐层递归验证。**
- 位置：`24_speculative_decoding_variants_analysis.md:7`（页首“§2.2、§3–4、Fig. 6–7”）、`:22`（“EAGLE §2.2 沿用 speculative sampling 的修正”）、`:69`（“其 §2.2 明确把目标接受概率 $\min(1,p/q)$ 与正差残差放回最终验证”）；raw 索引 `EAGLE-2401.15077.md` 也写“§2.2”。
- 证据（2401.15077v1）：§2 “Preliminaries” 下没有编号子节，接受率/残差出现在它的 “Speculative sampling” 段落，并声称 greedy 与非 greedy 两种设置下分布不变。§4.1：“EAGLE generates a tree-structured draft”，Fig. 7 用 3 次前向起草 10 个 token 的树。§4.3 Verification phase：用 tree attention 一次前向，“At every level of the draft tree, we recursively apply speculative sampling algorithms … consistent with SpecInfer … and SpecTr”。
- 影响：本页 §3 强调树验证不能套用线性链规则，却把 EAGLE 的正确性写成线性 $\min(1,p/q)$+残差，且全页不提 EAGLE 也是树草稿。读者会得到“EAGLE=线性草稿+线性验证”的错误图景，也无法把 EAGLE 与 §3 的树验证表对上。
- 建议：定位改为“§2（Preliminaries，Speculative sampling 段）与 §4.3”；表格与 §3 写明 EAGLE 起草 token 树，并在每层递归应用 speculative sampling（与 SpecInfer 的多候选验证同类）。

**P1-24-3（已确认）原理图没有重放最小例子的树 mask、位置编号和验证输出（rubric 第 5 项）。**
- 位置：`24_speculative_decoding_variants_analysis.md:37`（图规格承诺“两侧下方同时展示树 attention mask 的允许访问”）、`:39–54`（Mermaid）、`:56`。
- 问题：(1) 只给出 `AB` 一行可见性，没有给展平节点 `[A, X, AB, AY]`（加前缀 $h$）的完整 mask。按祖先规则应为：A 见 {h, A}；X 见 {h, X}；AB 见 {h, A, AB}；AY 见 {h, A, AY}。(2) 位置编号只有文字规则（§3.1.2 “positional indices … adjusted in line with this structure”），没有实例：A、X 应同为 $n$，AB、AY 同为 $n+1$，而非展平下标 $n..n+3$。(3) 没有把教学目标路径 A→Y 走到输出：greedy 校验下，线性布局接受 A、在位置 2 拒 B，再由目标给出修正 Y，共发布 2 枚；树布局接受 A、Y，再从 $p(\cdot\mid h,A,Y)$ 取 bonus，共发布 3 枚。图中 “线性最多走四层” 与 “按对应验证规则发布” 都没有给出本例结果。按 rubric，文字（`:35`）不能替代图的重放。
- 已给出的部分是正确的：`AB` 的上下文为 $h,A,B$ 而不是 $h,A,X,B,Y$；两种布局都是 5 个逻辑条件位置。
- 建议：补一张 mask 矩阵加位置编号的小图，并在两条支路末端写出本例发布的 token。

**P2-24-4（已确认）Medusa-2 会改变目标本身。** `:21` 只写“Medusa-1 冻结基座，Medusa-2 联训”。Medusa v1 摘要把 Medusa-1 描述为 lossless；§3.2.2 的 Medusa-2 联训 backbone，需要专门配方才能保住原能力。因此即使用精确拒绝采样，Medusa-2 保持的是**联训后** backbone 的分布，不是原模型的。建议在正确性口径里点明这一点。

**P2-24-5（已确认）SpecInfer 定理的措辞。** `:60` “其 Theorem 4.2 证明后者的分布正确性”：v4 §4.3 陈述了 Theorem 4.2（式 (6)），证明写着 “presented in [28]”。建议改为“陈述/给出……证明见其引文”。§4.1–4.2 的 tree attention 与 topology-aware causal mask 与页面描述一致。

### 3.4 `32_constrained_decoding_analysis.md`

**P1-32-1（已确认）没有说明“逐步 mask + 局部重归一化 ≠ 在约束下对模型分布取条件”（分布失真）。**
- 位置：`32_constrained_decoding_analysis.md:20–29`：把 $p_{s_t}(v)$ 称为“约束分布”，并说“只是将本步无效候选概率置零，并保留有效项间的相对权重”。
- 问题：逐步归一化得到的序列概率是 $\prod_t p(y_t\mid y_{<t})/Z_t$，其中 $Z_t$ 是第 $t$ 步有效 token 的总质量；它一般不等于 $P_{\text{model}}(y)/P_{\text{model}}(L)$。那些后续合法延续在模型下概率很低的前缀会被高估。用本页玩具例说明：抽到 `on` 后下一步只剩 `"}`，重归一化后概率为 1，无论模型原本给 `"}` 多少质量；若模型在 `on` 后只给 `"}` 1%（例如想续写 `only`），全局条件分布应大幅压低这条路径，逐步 mask 却不会。此外，同一字符串 `{"mode":"on"}` 可由两条 token 路径（`on`+`"}` 与 `on"}`）生成，字符串级概率是 0.665241+0.090031=0.755272，对 `off` 的 0.244728——token 化歧义也属于“约束分布”的一部分。两篇来源只在单步层面描述 mask：Willard & Louf §2.1 称之为 “unnormalized conditional distributions”，XGrammar §2.1 说 “preserving the relative probabilities of other valid tokens”。页面照搬单步表述，却没提醒序列级的差别；而相邻的 23 页恰恰强调“分布保持”，读者很容易误以为约束采样等于条件采样。
- 建议：§2 增一段“单步保持相对权重，序列级不等于条件分布”，用玩具例量化，并说明精确条件采样需要前瞻或重要性/SMC 类修正（Willard & Louf §2 提到 SMC 采样 [Lew et al., 2023]）。

**P2-32-2（已确认）空候选段落与本页 EOS 模型自相矛盾。** `:71` “若 $A(s_t)=\varnothing$ ……若在接受状态，正常结束可以是唯一合法动作”：本页 §1（`:16`）规定 EOS 只在接受状态允许，所以接受态且无后续字节时 $A=\{\text{EOS}\}\neq\varnothing$。建议分开写：“接受且不可继续 → $A=\{\text{EOS}\}$”与“真正的空集（词表无法覆盖所需字节、长度预算耗尽等）→ 报错/回退”。

**P2-32-3（已确认）Outlines 索引的描述含糊，缺朴素方案。** `:14` “为正则构造 token 到有限状态转移的索引”：论文的关键是 Algorithm 4 的映射 $\sigma:Q\to\mathcal P(V)$（FSM 状态 → 允许 token 集），把每步 $O(N)$ 的全词表扫描变成平均 $O(1)$ 查表（§1、§3）；PDA 扩展为 $\sigma:Q\times\Gamma_\epsilon\to\mathcal P(V)$（§4.1）。建议写出“状态→token 集”方向与 $O(N)$ 朴素做法（beat-1/beat-2）。

**P2-32-4（已确认）XGrammar 转述不准。** `:16` “token 可能跨字符甚至 Unicode 字节边界”：原文是 token “may break a Unicode character”（§2.1），且支持 “tokens containing sub-UTF8 characters”（§3）。字节是最小单位，谈不上“跨字节边界”；应写“可能只包含某个字符的一部分 UTF-8 字节”。`:69` 把 context-independent tokens 译作“上下文无关 token”，与“上下文无关文法（CFG）”撞词，建议改为“上下文独立（只由栈顶节点决定）的 token”。

**P2-32-5（建议）** Mermaid 已重放状态分支与接受；可补上第一步同样被 mask 的 `"}`、`EOS`，以及三项重归一化概率，使图自身覆盖 §2 的变换。

## 4. 抽查的锚点

| 页 | 来源与定位 | 页面所说 | 结果 |
|---|---|---|---|
| 17 | Holtzman v2 §3.1 Eq. (2)–(3) | top-p 取累计 ≥p 的最小集合并重归一化 | 相符 |
| 17 | Holtzman v2 §5.2 + Appendix A（Beam width effect） | 增大 beam 宽度仍可能重复/过短 | 相符（过短出自 §5.2/App. A：“average length gets shorter as b increases”） |
| 17 | Transformers `8cb5963c` `RepetitionPenaltyLogitsProcessor`（L297–L406） | 正分除、负分乘、至多一次、decoder-only 含 prompt | 相符（`torch.where(score < 0, score * penalty, score / penalty)`） |
| 17 | 同上 `TopPLogitsWarper`（L464–L528） | top-p 保留最小集合 | 相符（升序累加，删除 `cumprobs <= 1-top_p`，本例保留 A、B） |
| 17 | 同上 `TemperatureLogitsWarper`（L231–L294） | $T=0$ 转 greedy | **不符**（`__init__` 对 $T\le0$ 抛错；见 P2-17-2） |
| 17 | 同上 `beam_search.py#L2639-L2646` | `BeamSearchScorer.process` | **锚点不存在**；符号存在但已弃用，实际路径在 `utils.py`（见 P1-17-1） |
| 23 | Leviathan v2 §2.3 Algorithm 1 | 接受/残差/bonus，1..γ+1 枚 | 相符 |
| 23 | Leviathan v2 §3.1 Eq. (1)、§3.3 Theorem 3.8 | $E[N]$、速度比 | 相符 |
| 23 | Leviathan v2 Appendix A.1 | 单位置分布等式 | 相符 |
| 24 | Draft & Verify v2 §3.2、Algorithm 2 | 仅 greedy 口径 | **不符**（§3.2 指向 Appendix K 的采样版 Algorithm 4；§4.2 主实验含 T=0.2/0.6） |
| 24 | Medusa v1 §3.1.2 / §3.3.1 / §3.3.3 | 祖先 mask 与位置；typical acceptance 放松分布匹配；校准集贪心加点 | 相符 |
| 24 | EAGLE v1 “§2.2” | 线性 $\min(1,p/q)$+残差 | **定位不存在**；内容在 §2，但 EAGLE 实际用树草稿与 §4.3 的逐层递归验证 |
| 24 | SpecInfer v4 Algorithm 2、Theorem 4.2 | VerifyGreedy/VerifyStochastic；随机验证分布正确 | 相符（证明在其引文 [28]） |
| 32 | Willard & Louf v1 §1、§3 Algorithm 3–4、§4.1 | FSM 索引与 CFG/PDA 扩展 | 相符（描述偏含糊，见 P2-32-3） |
| 32 | XGrammar v1 §2.1 Fig. 2、§2.2 | 设 $-\infty$ 保留相对概率；PDA 与多栈 | 相符 |
| 32 | XGrammar v1 §3.1、§3.3、§3.5 | 上下文独立/依赖 token、持久栈回滚、CPU mask 与 GPU 推理重叠 | 相符 |

## 5. 待补（进行中）

- 用 `.venv` Python 逐项复算 17/23/24/32 全部数字，结果补入第 6 节。
- 渲染 `17_sampling_decoding.svg` 做目检；T0 `check_links/check_math/check_markdown/check_assets` 定点运行；index 行与页题核对。
