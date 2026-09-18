# H 组：05_inference 全域覆盖、跨页一致性、导航与来源审阅

> **状态**：进行中（首版落盘，后续核查完成后原位更新）。
> **组别**：H — 覆盖、跨页一致性、导航与来源（全域视角；逐页技术深度由其他组负责）。
> **范围**：`wiki/01_theory/05_inference/` 全部 29 篇正文与 `index.md`；`wiki/01_theory/index.md`、`wiki/index.md`、`wiki/changelog.md` 的本批改动；`raw/01_theory/05_inference/` 新增 47 个来源页与 `raw/README.md` 改动；与 `02_engineering/03_infer_frameworks/`、`01_theory/06_distributed_parallelism/`、`05_gpu_kernel/` 等既有页的重叠。
> **打开的来源（含版本）**：arXiv abs 页 35 个（提交历史逐版核对，见 §5）；DistServe OSDI'24 正式版 PDF（§1 goodput 定义）；FlashAttention arXiv:2205.14135v2（§3.2 Theorem 2）；本库既有页 `02_engineering/03_infer_frameworks/01_llm_inference_technology_stack_analysis.md`、`vllm/04_vllm_performance_tuning_guide.md`、`vllm/16_vllm_speculative_decoding_analysis.md`、`05_gpu_kernel/01_gpu_kernel_guide.md`、`06_distributed_parallelism/20_ring_attention_and_context_parallel_analysis.md`（git diff）、`raw/` 既有来源页 4 个。
> **复算/机械核对**：29 页站内链接图（正文链接与 Related Pages 分开统计，入链/出链）；index 行标题与页面 H1/front-matter 逐一比对；`find` 复算页数；47 个新 raw 文件与 `raw/README.md` 逐一对账；页首引用的 40 个 `raw/...` 路径逐一 `test -e`；11 页 KV/GEMM 账与 30 页 KV 账抽算。机械门禁（check_links/math/markdown/assets、mkdocs build --changed、MathJax）已由协调者确认全绿，本组不重复（任务 6 跳过）。
> **不在本组范围**：逐页公式推导、算例与论文锚点的深度核对（其他组）。本组只在跨页一致性需要时抽查个别技术点。

## 1. 结论总览

全域 29 篇在 **PCP/DCP/CPP 定义、online softmax/LSE 记号（18 ↔ 28 ↔ 06/20）、投机解码记号（23 ↔ 24）、KV 容量公式的因子 2** 上相互一致；arXiv 版本与日期抽查 35/35 全部正确；页首引用的 raw 路径 40/40 存在；`wiki/index.md` 页数 30 与 `find` 一致。

主要问题集中在四处：

1. **核心指标定义跨页冲突**：TTFT 的起点在 11（含客户端排队，从到达计）与 40（从实际发出计）不同，41 又沿用 11 的口径；"有效吞吐/goodput" 在 11、40、DistServe、vLLM/AIPerf 之间用词冲突且未对照（P0 + P1）。
2. **本域与既有工程页之间几乎没有接线**：29 篇页面除 06/20 → 18 外**没有任何**来自工程域或其他理论域的入链；既有工程页仍各自讲原理，其中一处（05_gpu_kernel 指南的 FlashAttention IO 复杂度）与 18 页冲突且是错的。
3. **本域内部导航断在"后续专题"占位**：计划 T99 声称已检查的关键链有 9 条前向链接实际缺失；多处"后续专题/量化算法专题/后续评测页"仍是 A 批写作时的占位文字而未改成链接。
4. **状态与标题**：index 与上级索引、changelog 把全部 29 篇标为"交叉审阅通过/已完成"，而证据账显示 23/29 篇的"非作者审阅"由本波协调者完成，不满足 rubric 的独立审阅要求；index 有 2 行标题与页面 H1 不一致。

## 2. 逐页判定（仅 H 组维度）

说明：beat2 / delete-code / figure-trigger / algorithm-replay 属逐页技术组职责，本组不评（记"—"）。H 组以"跨页一致性 / 导航 / 来源"三项给出本组维度的判定；最终 verdict 需与各技术组合并。

| page | beat2 | delete-code | figure-trigger | algorithm-replay | spot-check（H：raw 路径/版本） | verdict（H 维度） | note |
|---|---|---|---|---|---|---|---|
| 01_inference_overview | — | — | — | — | raw ✓ / 版本 ✓ | PASS（带 P2） | 唯一全连通入口页；其余页大多只从它可达 |
| 10_prefill_decode | — | — | — | — | ✓ | PASS（带 P2） | 3 处"后续专题"占位未链接（→12/15/17） |
| 11_inference_cost_model | — | — | — | — | ✓ | **REJECT** | P0 TTFT 起点与 40 冲突；P1 goodput 用语；不链 40 |
| 12_kv_cache | — | — | — | — | ✓（DeepSeek-V2 用旧 raw 页） | PASS（带 P2） | 不链 13；标题与 index 行不一致（P1 记在 index） |
| 13_paged_kv_attention | — | — | — | — | ✓ | PASS（带 P2） | 位置 0 起算，与 10/12/16 的 1 起算不同 |
| 14_prefix_caching | — | — | — | — | ✓（SGLang commit ✓） | **REJECT（存疑待码核）** | P1 全前缀命中 $m_i=L_i$ 与 10 的"logits 属于已处理位置"冲突 |
| 15_continuous_batching | — | — | — | — | ✓ | PASS（带 P2） | Related 标签与 13 标题不一致 |
| 16_chunked_prefill | — | — | — | — | vLLM 文档用 `latest` 未版本化（P2） | PASS（带 P2） | index 行标题不一致（P1 记在 index） |
| 17_sampling_decoding | — | — | — | — | ✓ | PASS（带 P2） | 不链 23、32（"后续专题"占位） |
| 18_efficient_attention | — | — | — | — | ✓ | PASS（带 P2） | 在线 softmax 唯一 owner 成立；不链 27/28/29；外域 05_gpu_kernel 指南与其冲突（P1 记在外域） |
| 19_inference_quantization | — | — | — | — | ✓ | PASS（带 P2） | 不链 20、21（占位文字） |
| 20_quantization_methods | — | — | — | — | ✓ | PASS（带 P2） | $s$ 记号与 19 的 scale 冲突（P2） |
| 21_kv_compression | — | — | — | — | DeepSeek-V2 用新 raw 页（与 12 重复来源页） | PASS（带 P2） | |
| 22_kv_tiering_transfer | — | — | — | — | Mooncake raw 未记 v1 | PASS（带 P2） | 不链 26；$L,S$ 记号复用 |
| 23_speculative_decoding | — | — | — | — | ✓ | PASS（带 P2） | 未给 $\beta=\sum\min(p,q)$ 接受率式（P2） |
| 24_speculative_variants | — | — | — | — | ✓ | PASS（带 P2） | MTP 草稿源无出链 |
| 25_multi_lora_serving | — | — | — | — | LoRA raw 未记 v1 | PASS（带 P2） | |
| 26_pd_disaggregation | — | — | — | — | Mooncake raw 未记 v1 | PASS（带 P2） | 引 DistServe 却不出现 goodput（并入 P1 goodput 条） |
| 27_pcp | — | — | — | — | ✓ | PASS | |
| 28_dcp | — | — | — | — | ✓ | PASS（带 P2） | $b_{kv}$ 与 30 同名异义 |
| 29_cpp | — | — | — | — | ✓ | PASS | |
| 30_parallelism_composition | — | — | — | — | ✓ | PASS（带 P2） | "MoE 专题"未链 31 |
| 31_moe_inference | — | — | — | — | ✓ | PASS | |
| 32_constrained_decoding | — | — | — | — | raw 页缺作者（P2） | PASS（带 P2） | 仅 01 入链 |
| 33_multimodal | — | — | — | — | raw 页缺作者（P2） | PASS（带 P2） | 仅 01 入链 |
| 34_hybrid_state | — | — | — | — | ✓（vLLM commit ✓） | PASS（带 P2） | 把 SWA 可达范围指向 12，但 12 明确不覆盖（见覆盖缺口 G1） |
| 35_execution_optimization | — | — | — | — | ✓ | PASS（带 P2） | 仅 01 入链 |
| 40_benchmarking_guide | — | — | — | — | AIPerf URL 与 11 不同（P2） | **REJECT** | P0 TTFT 起点与 11 冲突；"原始有效输出吞吐"用语 |
| 41_optimization_guide | — | — | — | — | ✓ | PASS（带 P2） | TTFT 口径随 11，与 40 不同（由 P0 统一修） |
| index | — | — | — | — | — | **REJECT** | P1 两行标题不符；P1 审阅状态过度声明 |

## 3. 发现（按严重度）

### P0

**H-P0-1 TTFT（及 E2E）起点在 11 与 40 两个"指标页"中定义不同** — 已确认

- `wiki/01_theory/05_inference/11_inference_cost_model_analysis.md:14`："一次流式请求至少有到达、实际发出、首个内容响应和末个内容响应四个时间点。令第 $i$ 个请求在客户观测点的到达时刻为 $t_{i,\mathrm{arr}}$"；`:20-26` 定义 $\operatorname{TTFT}_i=t_{i,\mathrm{first}}-t_{i,\mathrm{arr}}$、$\operatorname{E2E}_i=t_{i,\mathrm{last}}-t_{i,\mathrm{arr}}$；`:182-189` 把 TTFT 分解为 $W_{i,\mathrm{client}}+W_{i,\mathrm{server}}+\dots$，并写"$W_{i,\mathrm{client}}$ 可能来自压测客户端并发阀……若一个报告只从'已实际发出'开始计时，它可能把客户端排队移出 TTFT"。即 11 的 TTFT **包含**客户端并发阀排队。
- `wiki/01_theory/05_inference/40_inference_benchmarking_guide.md:34`："客户端被并发阀挡住的等待是 $t_i^{\mathrm{send}}-t_i^{\mathrm{plan}}$；客户端可见 TTFT 是 $t_i^{\mathrm{first}}-t_i^{\mathrm{send}}$"。即 40 的 TTFT **不含**客户端排队；而 40 的 `:9`、`:16` 同时声明"TTFT、ITL……的指标与资源定义归 11""各指标的事件定义……见 11"。
- `41_inference_optimization_guide.md:14`、`:56` 又按 11 的口径把"计划发出……客户端与网络 0.3 s"计入 TTFT。
- 外域旁证：`02_engineering/03_infer_frameworks/vllm/04_vllm_performance_tuning_guide.md:126-130` 的 vLLM bench 口径是"TTFT 实际发送前开始计时""E2EL……不含客户端 semaphore 排队"，客户端排队单列 `client_queue_time`，与 40 一致、与 11 相反。
- **问题**：本域最核心的指标在"定义 owner（11）"与"测量 owner（40）"之间出现两个不同起点；读者会学到两个 TTFT（以及两个 E2E）。按简报规则属 P0。
- **修正建议**：由 11（计划 §7 指定的指标 owner）选定唯一的 TTFT/E2E 起点并命名另一个（建议与 vLLM bench/AIPerf 一致取 $t^{\mathrm{send}}$，另设"含客户端排队的 TTFT/E2E"= 从 $t^{\mathrm{plan}}$ 起算）；40、41 改为引用该定义，41 场景 A 的分解同步改名。

### P1

**H-P1-1 goodput / "有效吞吐" 三处用语互相冲突，且全域不出现 "goodput" 一词** — 已确认

- `11:55`："$\Theta_{\mathrm{good}}$ 是本文采用的**有效输出吞吐**"——token/s，且要求"成功且满足预先声明的 SLO 与质量门"（`:50-51`）。
- `40:45`：$Q_{\mathrm{good}}$ 是"满足预定延迟 SLO 的请求吞吐"（req/s），并说明它与 11 的量不同；但 `40:57` 表行"**原始有效输出吞吐** | 196.67 token/s"指的是**全部成功**输出（即 11 的 $\Theta_{\mathrm{out}}$），把 11 中专指"过 SLO 与质量门"的"有效"一词用成了"成功"。
- `26_prefill_decode_disaggregation_analysis.md` 以 DistServe 为主来源，全文不出现 goodput；而 DistServe OSDI'24 §1（正式版 PDF 第 2 页）把 per-GPU goodput 定义为"the maximum request rate that can be served adhering to the SLO attainment goal (say, 90%) for each GPU provisioned"，论文标题即 "Goodput-optimized"。40 的决策规则（`:14` "找到满足 TTFT 与流内间隔 SLO 的最大成功请求率"，`:51` "$f_{\mathrm{good}}\ge90\%$"）正是这一概念，却未命名、未归属。
- 外域：vLLM 04（`:34`、`:133`）用 "SLO goodput / Request goodput"（req/s）。
- **问题**：学习者在工程页遇到的 goodput 在本域找不到对照；本域内部"有效"一词两义。
- **修正建议**：11 显式引入 "goodput"，并用一张小表对照三种量：token 级 good 吞吐（本域 $\Theta_{\mathrm{good}}$，可含质量门）、固定负载下的请求级 goodput（AIPerf/vLLM bench，= 40 的 $Q_{\mathrm{good}}$）、DistServe 的"达标率约束下的最大请求率"（每 GPU）；40 把"原始有效输出吞吐"改为"成功输出吞吐（$\Theta_{\mathrm{out}}$）"；26 在资源配比处点出 DistServe 的优化目标是 per-GPU goodput。

**H-P1-2 全前缀命中 $m_i=L_i$ 与"logits 属于已处理位置"冲突** — 存疑（引擎侧证据待补，见 §3 更新）

- `14_prefix_caching_analysis.md:17`："命中后，新请求可把匹配位置当作已有 K/V，只对未命中的后缀作前向计算"；`:69` 允许 $0\le m_i\le L_i$。
- `10_prefill_decode_analysis.md:55`："logits 属于已处理的输入位置，选出的 token 才是下一位置的输入"。
- **问题**：若整条 prompt 都命中（$m_i=L_i$），本轮没有任何已处理位置，也就没有可用于采样首个输出 token 的 logits——除非另外缓存了末位置 logits。14 没有写这个条件，与 10 的时序规则矛盾。
- **修正建议**：14 写明"为得到首 token 的 logits，至少要重算最后一个 prompt 位置（常见做法是把可用命中上限设为 $L_i-1$），或另存末位置 logits"，并把 $m_i\le L_i$ 改为带条件的 $m_i\le L_i-1$。

**H-P1-3 本域与既有工程/理论页几乎零接线；现有页继续各讲一遍原理** — 已确认

- 机械统计：29 篇中只有 18 从 `06_distributed_parallelism/20_ring_attention_and_context_parallel_analysis.md:280` 获得一条域外入链（外加 changelog）；其余 28 篇域外入链为 0。`02_engineering/03_infer_frameworks/index.md:32` 只链到 05 index（且标签仍为"推理技术理论"）。
- 计划 §6 要求"每个任务都更新……必要的已有反向链接"；证据账 `:8` 却记"本轮没有修改原有工程页"，`:53` 对 06/21 的 CPP 交接链接写"T29 写通用原理后再考虑只补交接链接"——至今未补。
- 读者确实需要的回链（均为既有页正在讲同一原理或同一指标的位置）：`01_llm_inference_technology_stack_analysis.md` §四 指标表与 §五 学习路径（→11/40/41 与 10–18）；`vllm/04` 指标口径（→11/40，且见 P0）；`vllm/16` §2.1–2.3 完整重推接受/残差（→23/24）；`vllm/07`（→15/16）、`vllm/08`（→12/13/14）、`vllm/10`（→18）、`vllm/14`（→17/32）、`vllm/15`（→33）、`vllm/17`（→19/20）、`vllm/18`（→27/28/30）、`vllm/19`（→35）、`vllm/22`（→22/26）、`vllm/23`（→11/40）；`speculative_decoding/index.md`、`dspark_analysis.md`（→23/24）；`06/21_hw_friendly_llm_codesign_analysis.md` §六（→29）；`mooncake_analysis.md`（↔22/26，双向都缺）。
- 另有过期归属指针：`02_engineering/02_train_frameworks/mindspeed/20_mindspeed_context_parallel_analysis.md:144`（"online-softmax 合并公式均已归一到理论页 §5"）、`torchtitan/13_torchtitan_cp_analysis.md:14` 仍把 online softmax 归 06/20，而 06/20 §5.2 现已把通用推导交给 18。

**H-P1-4 既有工程页对 FlashAttention IO 复杂度的解释与 18 冲突且有误** — 已确认

- `wiki/02_engineering/05_gpu_kernel/01_gpu_kernel_guide.md:252`："FA 利用 online softmax……HBM 流量降为 O(N·d)，匹配 Q/K/V/O 本身的大小。IO 复杂度从 O(N²) 降至 O(N²·d / M)"。
- `wiki/01_theory/05_inference/18_efficient_attention_analysis.md:71`："普通物化 attention 为 $\Theta(Nd+N^2)$，该 FlashAttention 算法为 $\Theta(N^2d^2/M)$"。
- **证据**：FlashAttention arXiv:2205.14135v2 §3.2 Theorem 2（PDF 第 5–6 页）："Standard attention (Algorithm 0) requires Θ(Nd + N²) HBM accesses, while FlashAttention (Algorithm 1) requires Θ(N²d²M⁻¹) HBM accesses"，条件 $d\le M\le Nd$；Proposition 3 说明只有 $M=\Theta(Nd)$ 时才到 $\Omega(Nd)$。
- **问题**：同一概念在库里有两份解释，旧页自相矛盾（O(Nd) 与 O(N²d/M) 并列）且与论文不符，新 owner 页正确。
- **修正建议**（归 05_gpu_kernel owner）：改为 $\Theta(N^2d^2M^{-1})$ 并注明 $d\le M\le Nd$，链接 18 作为推导 owner。

**H-P1-5 计划声称已检查的关键阅读链有 9 条前向链接缺失；多处仍是"后续专题"占位** — 已确认

计划 §9 T99 勾选"检查关键链：10→12→13→14；10→15→16；19→20/21；17→23→24；16/18→27/28/29；22→26；11→40→41"。按链接图实测（正文 + Related Pages）：

| 链 | 实际 | 占位证据 |
|---|---|---|
| 12→13 | 缺 | 12 的正文只链 10，Related 只有 10 与外域 |
| 10→15 | 缺 | `10:61` "批调度和 token 预算属于后续调度专题" |
| 19→20、19→21 | 缺 | `19:9` "量化方案搜索、校准与训练归量化算法专题；KV 的长上下文质量归 KV 专题"；`19:78`、`:86`、`:98`、`:132` 同类 |
| 17→23 | 缺 | `17:9` "投机解码的接受／拒绝修正归后续专题……归约束解码专题"；`17:124`、`:170` 同类（也缺 17→32） |
| 16→28、18→27/28/29 | 缺 | 18 只在 `:83` 链 06/20 |
| 22→26 | 缺 | `22:9` "Prefill/Decode 部署及池间资源配比归分离式推理页" |
| 11→40 | 缺 | `11:9` "实验设计、工具和统计结论属于后续评测页"；`:207` "应在后续评测页按冻结工作负载取得" |

另：`10:9`、`10:22`（"其成立条件见后续 KV 专题"）、`30:9`（"动态专家负载均衡留给 MoE 专题"）亦未链接。结果是 20、26、31、32、33、34、35 七篇只有 01 一个域内入链（20 仅 01；24 仅 01/23；25 仅 01/41）。

**H-P1-6 index 两行标题与页面 H1 不一致** — 已确认

- `index.md:20` "KV Cache：复用依据与容量" ↔ 页面 H1 "KV Cache：为什么历史可以复用，以及它占多少内存"。
- `index.md:24` "Chunked Prefill：长输入的分步执行" ↔ H1 "Chunked Prefill：把长输入拆到轮次之间，同时保留因果历史"。
- 另 10、11、13、15、17 五行只取冒号前短名（与 18–41 行取全名不一致，P2）。Related Pages 中 12 页的标签出现三种写法（"复用依据与容量""历史复用与容量""KV Cache 基础"），13 页有"Paged KV Cache 与 PagedAttention""Paged KV 与块表"两种。

**H-P1-7 审阅状态过度声明** — 已确认

- `index.md:9`："2026-09-17 完成并交叉审阅全部 29 篇正文"；29 行全部"已完成，交叉审阅通过"；`wiki/index.md:32` 与 `wiki/01_theory/index.md:17` "29 篇正文已完成"；`wiki/changelog.md` 本批条目"各页的教学算例、公式和图均经非作者审阅"。
- 证据账 `docs/research/2026-09-17-inference-principles-source-ledger.md:64-94`：29 行中 23 行的"非作者审阅"是"协调者"。rubric（`skills/source-faithful-analysis/references/page-review-rubric.md` "Who runs it"）要求"The reviewer is never the writer……the coordinator dispatches one independent reviewer per wave"；计划 §8.3 自己规定"未获得独立结论的页面标'已撰写/待审'，不冒称最终完成"。
- **修正建议**：本轮独立审阅闭环前，index 与上级索引状态改为"已撰写，独立审阅中"，changelog 删去"均经非作者审阅"的表述或加限定。

### P2

- **H-P2-1 记号跨页复用（已确认）**：$b_{kv}$ 在 `28:53` 指"一层、每个 KV head、每 token 的 K/V 字节"，在 `30:19-21` 指"每 token、全部层的 KV 字节"；$B$ 分别是块长（13、34）、token 预算（16）、beam 宽（17）、位宽（19）、key 块数（18）、块名（22）、LoRA 矩阵（25）；$d$ 是 head dim（11、30）、decode 数（16）、DCP 度（28）；$L$ 是层数（11、12、21）、启动开销（22）、序列长度（34）；$S$ 是序列长度（12、27）、同步成本（22）；$s$ 在 19 是量化 scale、在 20 是通道缩放；$\alpha$ 是长度惩罚（17）、迁移强度（20）、接受率（23）。每页局部都有定义，不构成错误；建议在 11 或 index 加一张"本域通用记号"表，至少统一 KV 公式记号（11 的 $2LH_{kv}db$ 与 12 的 $SLH_{kv}(d_k+d_v)b$ 已一致，28/30 的 $b_{kv}$ 应改名）。
- **H-P2-2 位置编号起点不一（已确认）**：10、12、16、21、23 从 1 起；13、22、27、28、29 从 0 起。
- **H-P2-3 措辞与本域立场不一（已确认）**：`26:8` "将计算密集的 prompt 处理和反复读取历史的逐 token 生成交给不同实例"，而 10:69、11:170、16:16 明确拒绝"prefill 必然算力受限"的无条件说法；建议 26 加"通常"或引 11 的条件。
- **H-P2-4 SWA 指针落空（已确认）**：`34:29` "窗口的可达范围见 [[12_kv_cache_analysis|KV Cache 基础]]"，但 `12:63` 明说滑动窗口"须重新分析……不能直接照搬本页"。见覆盖缺口 G1。
- **H-P2-5 index 超出"只维护本级入口表"（已确认）**：`index.md:9` 指向 `docs/research/` 工作账；`:50` 重复 PCP/DCP/CPP 全称与"DCP 不指 Dynamic CP"（27/28/29 已拥有）；可保留分段说明，其余建议移除或压缩为一句范围说明。
- **H-P2-6 外部入口陈旧（已确认）**：`README.md:36` 仍为 "推理技术（待建设）"；`02_engineering/03_infer_frameworks/index.md:32` 标签"推理技术理论"；`wiki/01_theory/index.md:7` 导语仍说"推理技术"。
- **H-P2-7 无课程串联（建议）**：`wiki/courses/` 三门课均不经过本域；本域加 vLLM 工程域正是"推理端到端"阅读路径的天然材料。计划写明本批不新增课程，故只记建议。

## 4. 覆盖缺口（首版，后续补充外域核对）

（待补：逐项 grep 全库确认是否已有其他页覆盖后定级。）

## 5. 来源与版本核对

### 5.1 arXiv 版本/日期逐版核对（35 个 id，全部通过）

对 `https://arxiv.org/abs/<id>` 的 Submission history 逐版核对页首与 raw 页所写"版本 + 日期"：1706.03762v7（2023-08-02）、1712.05877v1（2017-12-15）、1904.09751v2（2020-02-14）、1911.02150v1（2019-11-06）、2106.09685v1（2021-06-17）、2204.14198v1（2022-04-29）、2205.14135v2（2022-06-23）、2208.07339v2（2022-11-10）、2210.17323v1（2022-10-31）、2211.17192v2（2023-05-18）、2304.08485v2（2023-12-11）、2305.09781v4（2024-04-01）、2305.13245v1（2023-05-22）、2306.00978v1（2023-06-01）、2306.14048v1（2023-06-24）、2307.08691v1（2023-07-17）、2307.09702v1（2023-07-19，v1 标题 "Efficient Guided Generation for LLMs" 与 raw 一致）、2308.16369v1（2023-08-31）、2309.06180v1（2023-09-12）、2309.08168v2（2024-05-20）、2309.17453v2（2023-11-21）、2310.18547v1（2023-10-28）、2311.03285v1（2023-11-06）、2312.00752v1（2023-12-01）、2312.07104v1（2023-12-12，v1 标题 "Efficiently Programming Large Language Models using SGLang" 与 14 一致）、2401.10774v1（2024-01-19）、2401.15077v1（2024-01-26）、2402.02750v1（2024-02-05）、2403.19887v1（2024-03-28）、2405.04434v1（2024-05-07）、2407.00079v1（2024-06-24）、2411.15100v1（2024-11-22）、2412.19437v1（2024-12-27）、2608.26523v1（2026-08-27，作者 Yan Shi、Xiaochao Wang、Jingchun Gao 等与 raw 一致）。另查 DistServe arXiv 版 2401.09670（本域用 OSDI 正式版，合理）。

### 5.2 raw 路径与 raw/README

- 29 篇页首引用的 40 个 `raw/...` 完整路径全部存在；短名引用（`FlashAttention2-2307.08691.md` 等）亦均存在。
- 47 个新增未跟踪 raw 文件（简报所说 48 行含 `raw/README.md` 本身的修改）全部登记在 `raw/README.md`。

### 5.3 来源问题（P2，已确认）

- **重复来源页**：DeepSeek-V2 同时有 `raw/01_theory/01_models/deepseek/DeepSeek_V2-2405.04434.md`（12 使用）与新增 `raw/01_theory/05_inference/DeepSeekV2_MLA-2405.04434.md`（21 使用），两者都进了 `raw/README.md`。建议 21 改用既有页，删去新页（或在既有页补 v1 与本批取证范围）。
- **既有 raw 页未记所钉版本**：Mooncake（`raw/01_theory/01_models/moonshot_kimi/Mooncake_KVCache_Disaggregated-2407.00079.md` 只有无版本链接，"最后更新 2025-09-03"= v4 FAST'25 改版）被 22/26 以 v1 §3/§4.2/§5 引用；LoRA raw（无版本，最后更新 = v2）被 25 以 v1 引用；DeepSeek-V2/V3 raw 同理（12、31 引 v1）。页面自身链接带 v1，不致错；但沿 raw 进入的读者会落到章节已重排的最新版。
- **raw 页字段不全**：`Flamingo-2204.14198.md`、`XGrammar-2411.15100.md`、`Visual_Instruction_Tuning-2304.08485.md` 无作者；`DistServe-OSDI2024.md`、`FlexGen-ICML2023.md`、`SmoothQuant-ICML2023.md` 无日期行（仅会议名）。`raw/README.md:9` 声明每页含"规范标题、arXiv/官方链接、提交日期、主分类、作者与摘要"，新页采用了另一套"取证范围"格式。
- **同一动态文档两个 URL**：AIPerf Metrics Reference 在 11 与 `InferenceCostModelSources-20260917.md` 用 `/aiperf/dev/reference/...`，在 40/41 与 `InferenceBenchmarkMethodology-20260917.md` 用 `/aiperf/reference/...`。
- **可版本化却用 latest**：16 与 `vLLMChunkedPrefillDocs-20260917.md` 引 `docs.vllm.ai/en/latest/configuration/optimization/`（仅访问快照），而 35 对同一页面用了版本化的 `/en/v0.20.1/configuration/optimization/`；建议 16 同样钉到版本路径。19 的 TensorRT "Working with Quantized Types" 已声明无 11.3.0 固定路径，合理。

## 6. 已核对锚点

| 用途 | 来源 | 定位 | 结果 |
|---|---|---|---|
| H-P1-1 goodput 定义 | DistServe OSDI'24 正式版 | §1（PDF 第 2 页） | 原文为"maximum request rate…adhering to the SLO attainment goal (say, 90%) for each GPU" |
| H-P1-4 IO 复杂度 | FlashAttention 2205.14135v2 | §3.2 Theorem 2、Proposition 3（PDF 第 5–6 页） | $\Theta(Nd+N^2)$ vs $\Theta(N^2d^2M^{-1})$，$d\le M\le Nd$；18 正确，05_gpu_kernel 指南不符 |
| H-P0-1 旁证 | 本库 `vllm/04_vllm_performance_tuning_guide.md` | `:126-130` 指标口径表 | TTFT 从实际发送计、E2E 不含客户端排队 |
| 在线 softmax 归属 | `06/20_ring_attention…` git diff | §5.2 | 已改为引用 18，记号 $(m,\ell,u)$ 与 18/28 一致 |
| 35 个 arXiv id | arxiv.org/abs | Submission history | 35/35 版本日期一致 |
