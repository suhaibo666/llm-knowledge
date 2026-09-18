# 组 B 独立审阅：KV 状态、分页、前缀缓存与分层迁移

> **审阅组**：B — KV cache state, paging, prefix caching, tiering（独立审阅，非作者）。
> **页面**：`wiki/01_theory/05_inference/` 下 `12_kv_cache_analysis.md`、`13_paged_kv_attention_analysis.md`（含 `assets/paged_attention_block_refcount.{svg,mjs}`）、`14_prefix_caching_analysis.md`（含 `assets/prefix_cache_identity_radix.{svg,mjs}`）、`22_kv_tiering_transfer_analysis.md`。
> **审阅日期**：2026-09-18。状态：全部核验完成（结论见 §1 判定表与 §5 摘要）。

## 0. 已打开的来源与复算范围

**论文（均下载 PDF 并生成分页文本后逐段阅读，未用摘要或博客）**

| 来源 | 固定版本 | 实际阅读的位置 |
|---|---|---|
| PagedAttention | arXiv:2309.06180v1 | §1（p.2）、§2.1–2.3、§3/§3.1、Fig. 3、§4.1 Eq. (4) 与脚注 1、§4.2、§4.3、§4.4（parallel sampling / Fig. 8、beam search、shared prefix）、§4.5、§4.6、§5.1–5.2、§7.1–7.3、Fig. 18–19、§8 |
| Multi-Query Attention | arXiv:1911.02150v1 | §2.4、§2.4.1、§3、§3.1、§4.1 |
| GQA | arXiv:2305.13245v1 | §1、§2.1、§2.2、Fig. 2、§3.1 |
| DeepSeek-V2 | arXiv:2405.04434v1 | §2.1、§2.1.1–2.1.4、Eq. (1)–(19)、Table 1（p.6–8） |
| Transformer | arXiv:1706.03762v7 | §3.1、§3.2.3（因果掩码） |
| SGLang | arXiv:2312.07104v1 | §2.1、§2.3 与 Fig. 2、§5、§5.1、§5.2、Fig. 6、Algorithm 1、§6.3.2、§6.4.1–6.4.4 |
| Mooncake | arXiv:2407.00079v1（2024-06-24） | §2、§3、Fig. 3、Fig. 4 及其 ∗/† 注、§4.1、§4.2、Algorithm 1、§5.1、§5.2 |
| FlexGen | ICML 2023（PMLR 202）正式 PDF | 摘要、§1（含三层存储图）、§4、§4.1、§4.2（Algorithm 1、Tensor placement、Computation delegation） |

**源码（只读）**：SGLang `0b3bb0cbe31873994c9f989fddfe2f87ca839fdd`（`v0.5.15.post1`，commit 日期 2026-07-13）：`python/sglang/srt/mem_cache/radix_cache.py::{RadixKey, RadixKey.match, RadixKey.child_key, RadixKey.page_aligned, TreeNode, RadixCache.match_prefix, RadixCache._match_prefix_helper, RadixCache._split_node, RadixCache.insert, RadixCache.cache_finished_req, RadixCache.evict, RadixCache.inc_lock_ref, RadixCache.dec_lock_ref, RadixCache._update_leaf_status}`；`python/sglang/srt/managers/schedule_policy.py::{match_prefix_for_req, PrefillAdder._req_inc_lock_ref}`；`python/sglang/srt/managers/schedule_batch.py::Req._compute_max_prefix_len`；`python/sglang/srt/server_args.py` 的 `radix_eviction_policy` 默认值。旁证：vLLM `199cb9b`（brief 所列 main 快照）`vllm/v1/core/kv_cache_manager.py` 中全命中时的 `max_cache_hit_length`。

**raw 索引**：页首所引 `raw/01_theory/05_inference/{PagedAttention-2309.06180, MultiQueryAttention-1911.02150, GroupedQueryAttention-2305.13245, SGLang_RadixAttention-2312.07104, FlexGen-ICML2023}.md`、`raw/01_theory/01_models/deepseek/DeepSeek_V2-2405.04434.md`、`raw/01_theory/01_models/moonshot_kimi/Mooncake_KVCache_Disaggregated-2407.00079.md` 全部存在。

**复算（Python / Node，脚本在 scratchpad）**

- T12：无缓存 $4+5+6=15$、有缓存 $4+1+1=6$；单层单头因果 query–key 对 $10+15+21=46$ 与 $10+5+6=21$；$S=6,L=2,d_k=d_v=8,b=2$ 下 MHA/GQA/MQA 每 token $256/128/64$ B、六位置 $1536/768/384$ B。全部正确。
- T13：$B=4$ 时 $t=6\Rightarrow l=1,o=2$；slot 公式 $4\cdot1+2=6$；A 的 $S=7\Rightarrow n_{\mathrm{block}}=2,w_{\mathrm{tail}}=1$；四个快照的块表与引用 `P7:1,P1:1 → 2,2 → P7:2,P1:1,P3:1 → P7:1,P1:1,P3:0` 与论文 Fig. 8 的 CoW 顺序一致。全部正确。
- T14：$10+9+6=25$、新算 $10+1+2=13$、命中 $0+8+4=12$。在“每条请求查找前，前一条已登记”的前提下正确（前提未写明，见 F14-4）。
- T22：$M=2\,\mathrm{MiB}$、$R=1\,\mathrm{GiB/s}$ 时 $M/R=2^{21}/2^{30}\,\mathrm{s}=1.953125\,\mathrm{ms}$，加 $0.5\,\mathrm{ms}$ 得 $2.453125\,\mathrm{ms}$；单位全程二进制，一致。
- 图：两个 `.mjs` 复制到 scratchpad 重新生成后与仓库 `.svg` 逐字节相同；两张 SVG 与 T12、T22 的 Mermaid 已用 headless Chrome 实际渲染目检（Mermaid 解析成功，无溢出/重叠）。
- 机械门：`check_math`、`check_markdown`、`check_assets` 对四页 `--strict` 均 0 error / 0 warning；`check_links --strict` 全库 482 页 broken/ambiguous/bare_index/stale_section/orphans 全 0。

## 1. 判定表

| page | beat2 | delete-code | figure-trigger | algorithm-replay | spot-check | verdict | note |
|---|---|---|---|---|---|---|---|
| 12_kv_cache_analysis | pass | pass | timing（缓存逐轮追加） | pass（Mermaid 复放 ABCD→EFG、写 KV 与停止边界） | 5/5 | REJECT（仅机械 P1） | 正文技术内容可通过；唯一 P1 是 index 行标题与页标题不一致（F12-1），其余为 P2 |
| 13_paged_kv_attention_analysis | pass | pass | layout + 状态转移 | pass（SVG 四阶段复放 B=4 块表/CoW/ref=0） | 7/7 | PASS（带 P2） | 算例、图与论文 Fig. 8 一致；定位与记号有 P2 |
| 14_prefix_caching_analysis | FAIL §5 淘汰规则 | pass | transform（radix 匹配/插入） | pass（最终树 + 逐请求命中长度；无分裂/逐出态，P2） | 论文 6 处内容属实（§5.1 漏写“叶子”，§6.4.1 定位错挂）；源码 hop-walk 通过 | REJECT | F14-1 全命中边界；F14-2 淘汰只取叶子的前缀闭包不变量缺失 |
| 22_kv_tiering_transfer_analysis | pass | pass | timing / 状态与完成边界 | pass（Mermaid 复放 B0/B1 的 G0→C0→G1 与两个失败分支） | Mooncake 4/5、FlexGen 2/2 | REJECT | F22-1：把 Mooncake §4.2 prefill 侧逐层等待挪到 decode 批次内（P0） |

## 2. 分页面发现

### 12_kv_cache_analysis.md

总体：问题→朴素方案→因果归纳论证→算例→容量公式→头布局→边界的顺序清楚，适合初学者；复用条件（同模型、同前缀 token、同位置/注意力规则，且不保证浮点逐位相同）表述准确；MQA/GQA 事实（GQA-1=MQA、GQA-H=MHA、需权重或 uptraining）与论文一致；MLA 缓存内容（$d_c$ 维联合 latent + $d_h^R$ 维共享解耦 RoPE key，Table 1 每 token $(d_c+d_h^R)l$ 元素、按元素计与精度无关）正确。

- **F12-1｜P1（机械）｜已确认**｜`wiki/01_theory/05_inference/index.md:20` 的入口写作 `KV Cache：复用依据与容量`，而页标题（`12_kv_cache_analysis.md:2,5`）是“KV Cache：为什么历史可以复用，以及它占多少内存”。brief 将 index/header 不一致列为 P1。这个别名在本域约 10 处链接中沿用（如 10、11、14、15、17、22 页及 `wiki/changelog.md:47`）。成本最低的修法是把页标题改成与该别名一致；若保留现标题，则同步 index 行。
- **F12-2｜P2｜已确认**｜`12_kv_cache_analysis.md:85`“论文 §2.1.2–2.1.3 解释了为什么位置编码不能简单并入低秩上投影”。缓存内容写对了，但 RoPE 不兼容的原因被含糊转述。DeepSeek-V2 §2.1.3（p.8）的原因是：若对 $k^C_t$ 施加 RoPE，$W^{UK}$ 会与随位置变化的 RoPE 矩阵耦合，推理时无法再把 $W^{UK}$ 吸收进 $W^Q$（§2.1.2 的权重吸收），因而“must recompute the keys for all the prefix tokens”。这正是缓存侧需要另存共享 $k^R_t$ 的原因。建议用一句话写出“吸收 + 重算前缀 key”这一因果，并补 Table 1 的 DeepSeek-V2 取值 $d_c=4d_h$、$d_h^R=d_h/2$，即 $4.5d_hl$、等效 2.25 组 GQA。可顺带用本页教学参数给一行 MLA 对照：$d_h=8$ 时每层每 token 36 个元素，两层 bf16 为 144 B/token，六位置 864 B。
- **F12-3｜P2｜已确认**｜`12_kv_cache_analysis.md:89`“分页、跨请求复用、**压缩**、卸载改变**状态的物理管理和驻留方式**”。KV 量化和 token 选择会改变保存的信息（有损），不只是物理管理。这与同组 `22_kv_tiering_transfer_analysis.md:79`（“量化、删除某些 token……信息已改变”）以及 21 页的三轴定义矛盾。修复：把“压缩”单列为“改变数值精度或保留位置（近似）”。
- **F12-4｜P2｜已确认**｜`12_kv_cache_analysis.md:7` 页首“文献基线”未列正文 `:16` 实际引用的 Transformer arXiv:1706.03762v7（§3.1、§3.2.3 已核对，掩码论述属实）。另外，DeepSeek-V2 在 raw 中有两份索引：页首指向 `raw/01_theory/01_models/deepseek/DeepSeek_V2-2405.04434.md`，21 页用新建的 `raw/01_theory/05_inference/DeepSeekV2_MLA-2405.04434.md`。同一论文存在两个来源索引，建议合并或互指。
- **F12-5｜P2｜已确认**｜`12:83`“采用不同头布局需要相应模型权重或转换训练”说法正确，但对初学者可再具体一句：GQA v1 §2.1 的 uptraining 是把各组 K/V 投影 mean-pool，再用约 5% 原预训练算力继续训练（§1、§2.1）；MQA 原文（§4.1）的模型则从头训练。
- **F12-6｜P2｜已确认**｜编号约定跨页不一：12 用 1 起算（位置 1–6，沿用 PagedAttention §2 的 $x_1..x_n$），13 明确用 0 起算（`13:22`），22 也用 0 起算（位置 0–7）。各页都自洽，但初学者在 12→13 之间易混。建议 13 的第 2 节加一句与 12 的换算。

### 13_paged_kv_attention_analysis.md

总体：技术内容与 PagedAttention v1 一致。三类浪费（预留、内部碎片、外部碎片）、逻辑/物理块与 #filled、Eq. (4) 块式注意力、Fig. 8 的“首个写者 CoW、另一分支见 ref=1 后原地写”、fork/append/free、20–26% kernel 延迟、块长 16 的条件、§8 的适用边界都已核对属实。B=4 算例与 SVG 四阶段逐项一致，并清楚区分了“数据仍在 / 可再分配 / 被引用”。

- **F13-1｜P2｜已确认**｜`13:61` 用“[来源：PagedAttention §4.4–4.5]”支持“引用到 0 后可 free 并再存其他请求 KV”。该语义实际在 §4.3（“its KV blocks can be freed to store the KV cache of other requests”）与 §4.4 beam search（“frees all physical blocks whose reference counts reach 0”）；§4.5 讲的是 FCFS、all-or-nothing 抢占与 swap/recompute，并不支持这一句。另外，页首 `:7` 列了 Fig. 6、7、9，正文并未引用；§4.5 在正文只出现在这一处错挂的“§4.4–4.5”里。建议改定位为 §4.3–4.4，并删去页首未用的图号，或在正文用上它们。
- **F13-2｜P2｜已确认**｜`13:14`“即使实际长度预先已知，整块连续区间在请求存活期间也不能借给短请求使用”标为 §3。原句在 §1（p.2：“even if the actual length is known a priori, the pre-allocation is still inefficient”）；§3.1 只有相近表述。建议补 §1 定位。
- **F13-3｜P2｜已确认**｜同一页里 $B$ 既是块长（`13:22,46`“块长 $B=4$”），又是请求名（“B 由相同的已处理历史分叉而来”“B 的块表”），SVG 图例也同时出现“块长 B = 4”与“请求 B”。初学者读 $\lfloor t/B\rfloor$ 时容易混淆。建议把两条分支改名为论文的 A1/A2，或改用 A/C。
- **F13-4｜P2｜已确认**｜`13:32`“$o=2$ … 写入或读取物理块 $P1$ 的第 2 格”。0 起算的 offset 2 在中文里容易读成“第 2 格”（1 起算）。建议写作“offset 2（第 3 格）”。
- **F13-5｜P2｜已确认**｜`13:52`“直接写 $P1$ 会改掉 B 的历史”。A 写入的是 P1 的空槽 offset 3，并不覆盖 B 已有的位置 4–6。真正的冲突是两条分支的位置 7 都要写同一个槽，结果会互相覆盖，或让对方的逻辑块读到不属于自己的 token。建议改为“两条分支的位置 7 都映射到 P1 offset 3”。
- **F13-6｜P2｜已确认**｜`13:89`“块表数量随 $\lceil S/B\rceil$ 增长”。随 $\lceil S/B\rceil$ 增长的是每张块表的**表项数**；块表张数取决于请求数以及按层/头的建表方式（§4.1 脚注 1）。
- **F13-7｜P2｜已确认（图）**｜SVG 第③阶段把“A 触发 CoW（P1 ref 2→1）”和“B 随后原地写”合成一个终态。论文 Fig. 8 的关键在顺序：A 先检查 ref>1，分配 P3 并复制，递减到 1；B 随后看到 ref=1 才原地写。正文 `:52` 讲清了这个顺序，图只给终态。建议拆成 ③a/③b，或在③中加一个“ref 2→1”的中间标注。
- **F13-8｜P2｜已确认**｜`13:42`“每个逻辑块表项还**须**知道已有多少有效格”。论文 Fig. 6 的块表确实记录 #filled；但 §4.3 同时说明块从左到右填满，只有全部前序块满了才分配新块。因此只有尾块可能不满，各块的有效格数由序列长度 $S$ 唯一确定。逐表项计数是一种表示，不是必要条件。建议改为“须能得知有效长度（逐项计数或序列长度）”。
- **F13-9｜P2｜已确认**｜`index.md:21` 的别名“分页 KV 与 PagedAttention”只是页标题冒号前的部分，而 index 其他行多用完整标题。

### 14_prefix_caching_analysis.md

总体：身份键（token ID、位置/注意力语义、模型/adapter、多模态/检索上下文、租户隔离）论证到位，且没有把 vLLM 的块哈希链或 SGLang 的页对齐写成通用定律（`:39,41` 都明确限定）。SGLang 论文事实（radix tree、节点引用计数、计数为零才可逐出、LRU、Algorithm 1 按匹配长度排序与 `evictable_size()+available_size()` 选批）与源码 hop-walk 都已核对属实。但有两处影响“怎样正确命中/淘汰”的缺口。

- **F14-1｜P1｜已确认**｜`14:17`“命中后，新请求可把匹配位置当作已有 K/V，**只对未命中的后缀作前向计算**”，以及 `14:65–69` 的 $N_{\mathrm{new}}=\sum_i(L_i-m_i)$，定义域写作 $0\le m_i\le L_i$。缺少的条件是：KV cache 只保存各层 K/V，不保存最后位置的输出隐藏状态或 logits。因此当整个输入都命中（$m_i=L_i$）时，仍须至少重算最后一个输入位置，才能得到第一个输出 token 的分布（除非另外缓存 logits）。按页内公式，这种情况的新算数是 0，而实际至少是 1。证据有三：（1）逻辑必然性，与 12/10 页“`G` 已采样但无 `KV[7]`”是同一边界的反面；（2）固定 SGLang 快照 `schedule_batch.py::Req._compute_max_prefix_len` 中 `max_prefix_len = input_len - 1`，注释“the matched length is at most 1 less than the input length…”，`schedule_policy.py::match_prefix_for_req` 用它截断 `num_matched_prefix_tokens`；（3）旁证 vLLM `kv_cache_manager.py` 的 “When all tokens hit the cache, we must recompute the last token to obtain logits … max_cache_hit_length = request.num_tokens - 1”。本页 A/B/C 算例恰好没有触发这个边界，但公式和 §1 的表述对全命中是错的。修复：把可用命中上限写成 $m_i\le L_i-1$（需要生成时），或在公式下加一句“全命中仍须重算末位置以得 logits”。
- **F14-2｜P1｜已确认**｜`14:59`、`14:74` 只写到“计数为零的节点才可被逐出”“LRU 驱逐未被运行请求引用的节点”，缺少 RadixAttention 淘汰规则中决定正确性的那一半：**只逐出叶子，并递归向上**（SGLang v1 §5.1：“an LRU eviction policy that recursively evicts leaf nodes”）。原因在于前缀闭包不变量：一个缓存节点只有在其全部祖先的 KV 都还在时才可能被命中（命中必须是从位置 0 起的连续前缀）。若逐出内部节点（例如本例的 `H0…H3`），它下面的 `A0 A1`、`B0` 的 KV 仍占容量，却永远无法命中。固定源码 `RadixCache.evict` 只从 `evictable_leaves` 取候选；删除叶子后，只有当父节点变成无子且 `lock_ref==0` 时，父节点才入堆（`_update_leaf_status`、`_delete_leaf`）。这是 §5“保留成本”这一主要单元的 beat-2：规则为何成立没有交代，读者可能以为任意无引用节点都可逐出。修复：补一段“叶子优先 + 前缀闭包”的理由，最好在图或算例中演示一次逐出（例如容量不足时先逐出 `C0 C1` 或 `A0 A1`，而不是 `S`/`H`）。
- **F14-3｜P2｜已确认**｜`14:74`“论文也说明其当时实现只匹配已有 cache tree、不会把 waiting queue 中尚未插入的请求当作可共享前缀”，挂在 `[来源：SGLang §5、Algorithm 1]` 下。该说明实际在 §6.4.1（p.11：“we only compare a pending request with the existing cache tree. We do not construct a tree index for all requests in the waiting queue”）。内容属实，定位应补 §6.4.1。
- **F14-4｜P2｜已确认**｜`14:49–57` 的算例只写“A、B、C 依次到达”，没有写明**登记时点**：B 要命中 8 个位置，前提是 A 的 KV 在 B 查找前已插入树中。论文 Algorithm 1 只在请求完成后 `T.insert(req)`（p.8）。固定快照插入得更早：未完成请求的 prefill 结束后，`scheduler_components/batch_result_processor.py` 就调用 `maybe_cache_unfinished_req(req, tree_cache)`；分块 prefill 则在每块之后经 `scheduler.py::Scheduler.stash_chunked_request` 插入。因此若 A 仍在排队或 prefill 中，B 在两种设计下都会 miss；若 A 已完成 prefill 但仍在 decode，按论文算法 B 会 miss，按该快照 B 可以命中（§6.4.1 所述局限）。建议在算例前加一句“每条请求在后一条查找前已完成并登记”。01 页的同类算例就写明了“A 的四位置 prompt KV 已完成且可被索引后 C 才到达”。
- **F14-5｜P2｜已确认**｜`14:74` 对调度只列出论文做法，没有给 beat-2：论文 §5.2 的理由是避免“cache thrashing”，§6.4.1 / Fig. 10 显示 LPM 优于 FCFS/Random（简单任务如 MMLU 差别不大）。本页也没有提到最长前缀优先的代价（短命中请求可能饥饿、偏离 FCFS 公平）。建议各补一句。
- **F14-6｜P2｜存疑**｜`14:39`“命中后都应能回到原始 token/身份作一致性验证，避免把摘要碰撞当作内容相等”。这是合理的设计建议，但写成了“都应”。实际系统常见的做法是使用抗碰撞哈希而不逐 token 复核；SGLang 的 radix 匹配本身就直接比较原始 token。建议改为“要么用抗碰撞摘要，要么保留原 token 以便复核”。
- **F14-7｜P2｜已确认**｜`14:76`“论文 §6.4 的实验分别改变可共享长度和访问模式……支持‘命中长度、复用频率和容量共同决定收益’”。§6.4.3 改变的是可共享/不可共享长度和“每篇文章的问题数”，并没有改变容量。“容量”属于推断，应标注为分析推断。
- **F14-8｜P2｜已确认**｜`14:8` 源码基线写“（detached 于 `v0.5.15.post1`，2026-09-17；…符号列表）”，但该 commit 的日期是 2026-07-13，2026-09-17 是访问日期。页首还塞入了符号列表，并与“文献基线”合成 5 行页首。按 maintaining skill，页首日期应为基线日期，符号应放入正文的源码阅读路线。
- **F14-9｜P2｜已确认**｜`14:41` 把“匹配终止于压缩边中间时切分节点”与 `page_size` 截断一并称为“该快照的表示与对齐策略”。页对齐确属实现选择（SGLang v1 §5.1 的页大小就是 1 token）；但边中间分叉必须切分，是 radix（压缩前缀）树的定义性质，快照特有的只是“在 match 阶段而非 insert 阶段切分”。建议区分这两点，以免读者以为切分是 SGLang 特例。另外，图只给最终树，没有展示 A→B→C 过程中的两次切分（A 插入一条 10 token 的边，B 在 8 处切分，C 在 4 处切分），可作为 P2 改进。
- **F14-10｜P2｜已确认**｜`14:19`“PagedAttention §4.4 … 不定义如何发现跨请求内容相同”在字面上属实，但漏掉了对初学者有用的前史：同一节的 “Shared prefix” 段（p.8）已提出跨请求共享，做法是由服务方为**预定义**的共享前缀预留物理块，用户 prompt 映射到这些块，最后一块标记 CoW。SGLang v1 §5 开头也说既有系统的复用 “often requires manual configurations”。补一句“手工预定义 → 自动发现（radix 或哈希）”，能说清 Prefix Caching 相对分页多出了什么。

### 22_kv_tiering_transfer_analysis.md

总体：“身份命中 / 传输完成 / 消费可读”三分法清楚；发布与回收的安全不变量被明确标为教学推演；FlexGen 事实（三层存储、面向延迟不敏感的吞吐任务、wg/wc/wd·hg/hc/hd·cg/cc/cd 驻留比例、Algorithm 1 重叠下一层权重、下一批 cache 装载与上一批存储）属实；Mooncake Fig. 3（CPU 分页块、含前缀 hash、Messenger RDMA 传输）、Fig. 4 第 4 步（KV 全部到达 decode 节点 CPU DRAM 后才加入 continuous batching）、§5.1（TTFT = 排队 + 预测 prefill，传输时间受拥塞影响）、§5.2（远端命中不一定比重算快；热点复制）属实。

- **F22-1｜P0｜已确认**｜`22:16`“所有 KV 已进入 decode 节点 CPU DRAM 后，请求才进入其 continuous batching。**进入批次后，某层 attention 计算仍须等该层异步装载完成。**前一句是请求准入边界，后一句是每层 GPU 读取边界”，标注来源为 `[Mooncake §3、Fig. 3–4，§4.2]`。Mooncake v1 中的逐层 launch/wait 属于 **prefill 实例**：§4.2 标题就是“Layer-wise Prefill”，原文为“Before each layer's attention computation begins, the model waits for the asynchronous loading of that layer's KVCache…”；Fig. 4 注明“(∗) For prefill instances, the load and store operations … are performed layer-by-layer”。对 decode 实例，论文只写“(†) For decoding instances, asynchronous loading is performed concurrently with GPU decoding to prevent GPU idle time”，并未描述进入批次后的逐层等待。页面因此把 prefill 侧机制安到了 decode 批次内，属于来源未作的断言。`22:26`（“逐层等待装载”）与 `22:45` 的泛化表述本身可以保留，但应指明是 prefill 侧。修复：改为“Mooncake 在 **prefill 实例**上逐层等待复用前缀 KV 的异步装载（§4.2、Fig. 4 ∗）；decode 侧只说明异步装载与解码重叠（Fig. 4 †）”。prefill 侧“前缀命中→CPU→GPU 逐层装载→该层可读”恰好是本页论点最直接的论文证据。**跨页提示**：`26_prefill_decode_disaggregation_analysis.md:66`“`D` 的‘可读’对应 §4.2 逐层加载等待”有同样的错配，请负责 26 的组核对。
- **F22-2｜P2｜已确认**｜`22:65`“忽略重叠时，恢复路径的**下界式**教学估计为 $T\approx Q+L+M/R+S$”。忽略重叠的串行相加，对重叠后的真实关键路径是偏保守的**上界**；只有“用标称带宽、不计重试/争用”才构成下界。两种说法混在一起。建议删去“下界式”，或分开写明“不计争用时偏乐观、不计重叠时偏保守”。
- **F22-3｜P2｜已确认**｜`22:49–59` 只比较“迁回 GPU / 重算 / 留在原层”三类路径。所引 FlexGen §4.2 还给出第四种选择：**计算下放**（Computation delegation），即 KV 在 CPU 时直接在 CPU 上算 attention，只搬激活，I/O 约降为原来的 $1/s$（论文示例 $s\ge512$）。建议在路径表补一行，或至少在正文点明。
- **F22-4｜P2｜已确认**｜`22:71` 强调“有效带宽”但没有举例。PagedAttention v1 §7.3 / Fig. 19 给了现成的来源证据：块越小，CPU↔GPU 小传输越多，PCIe 有效带宽越低；块较小时重算更划算，块较大时 swap 更划算。建议引用一句，把块粒度与传输效率连起来，连接 13 页的块大小讨论。
- **F22-5｜P2｜已确认**｜`22:32–43` 的 Mermaid 渲染正常，内容与正文一致；但未写“图的规格”，也未使用本域的 `classDef` 配色（默认紫色全图，没有区分主路径与失败分支）。另外，读回阶段 C0 作为源在 G1 装载期间也须受保护，这一点在正文 §5 写到了，图里未体现。
- **F22-6｜P2｜已确认**｜同组记号冲突：`22:65` 用 $L$ 表示“传输启动及索引开销”、$S$ 表示“同步/映射成本”，而 12（`12:67`）与 13（`13:75`）中 $L$ 是层数、$S$ 是位置/token 数。按 12→13→22 的阅读顺序容易误读。建议 22 改用 $T_{\mathrm{launch}}$、$T_{\mathrm{sync}}$ 之类的记号。
- **F22-7｜P2｜已确认**｜页首 `22:8` 与表 `22:22` 直接使用“目录命中”，“目录”（记录某身份的 KV 副本在哪一层、哪个节点的位置索引）没有定义，与 14 页“内容索引”的关系也没交代。建议首次出现时加一句定义，并链接 14。

## 3. 已抽查锚点（逐条记录）

| 页面 | 来源与定位 | 页面主张 | 结果 |
|---|---|---|---|
| 12 | PagedAttention v1 §2.2（p.3） | 同一 token 在不同位置/历史下 KV 不同；只算新位置 K/V | 属实 |
| 12 | MQA v1 §2.4（p.3–4）、§3（p.5） | 增量注意力拼接 prev_K/prev_V；MQA 各头共享单组 K/V | 属实 |
| 12 | GQA v1 §2.1–2.2、Fig. 2（p.1–2） | GQA-1=MQA，GQA-H=MHA；需 mean-pool + uptraining | 属实 |
| 12 | DeepSeek-V2 v1 §2.1.2–2.1.3、Table 1（p.7–8） | 缓存 $c^{KV}$ 与共享 $k^R$；$(d_c+d_h^R)l$ 元素 | 属实（RoPE 原因转述含糊，见 F12-2） |
| 12 | Transformer v7 §3.1、§3.2.3 | 因果掩码阻止向左信息流 | 属实（未列入页首，见 F12-4） |
| 13 | PagedAttention v1 §3.1、Fig. 3 | 预留 / 内部碎片（请求结束才显露）/ 外部碎片 | 属实（“即使长度已知”一句出自 §1，见 F13-2） |
| 13 | §4.1 Eq. (4)、脚注 1 | 块式注意力；K/V 可整体或按层/头分表 | 属实 |
| 13 | §4.2 | 逻辑块左到右填充，块表记录物理块与 #filled | 属实 |
| 13 | §4.4、Fig. 8 | P7/P1 ref=2；A1 写时分配 P3、复制、ref 2→1，A2 原地写 | 属实（与页面 A/B 顺序一致） |
| 13 | §5.2 | fork / append / free | 属实 |
| 13 | §7.1、§7.2、Fig. 18 | 20–26% 更高 kernel 延迟；默认块长 16 的条件 | 属实 |
| 13 | §8 | 非 LLM、算力受限负载可能因间接寻址变慢 | 属实 |
| 14 | SGLang v1 §2.1 | KV 只依赖此前 token，同前缀可复用 | 属实（Fig. 2 实际位于 §2.3，页面写法可接受） |
| 14 | §5、§5.1 | radix tree；LRU 递归逐出叶子；节点引用计数为 0 才可逐出 | 部分：页面漏写“叶子”（F14-2） |
| 14 | Algorithm 1（p.8） | 按匹配长度排序；`evictable_size()+available_size()` 选批；完成后减引用并插入 | 属实 |
| 14 | §6.4.1、§6.4.3 | 仅匹配现有树，不索引 waiting queue；共享长度/结构实验 | 内容属实，定位错挂 §5（F14-3）；“容量”为推断（F14-7） |
| 14 | PagedAttention v1 §2.2 | token 在不同位置/历史下 KV 不同（身份键依据） | 属实 |
| 14 | PagedAttention v1 §4.4（parallel sampling、Shared prefix） | 已共享块续写靠引用计数与 CoW；不定义跨请求发现 | 属实（漏提“预定义共享前缀”前史，F14-10） |
| 14 | 源码 `RadixKey`、`child_key`、`_check_compatible` | token IDs + `extra_key` 命名空间，不同 `extra_key` 不共享节点 | 属实 |
| 14 | 源码 `match_prefix` → `page_aligned` → `_match_prefix_helper` → `_split_node` | `page_size>1` 时先截断；边中间结束时切分 | 属实 |
| 14 | 源码 `inc_lock_ref` / `dec_lock_ref` / `evict` / `_update_leaf_status` | 沿路径到根加减 `lock_ref`；只从可逐出叶子取候选，父节点无子且无锁后入堆 | 属实（策略可配，默认 `lru`） |
| 14 | 源码 `match_prefix_for_req` + `Req._compute_max_prefix_len` | （页面未提）命中上限 $L-1$ | 支持 F14-1 |
| 22 | Mooncake v1 §3、Fig. 3 | CPU 分页块、含前缀 hash、Messenger 传输 | 属实 |
| 22 | Fig. 4 第 4 步 | KV 全部到达 decode CPU DRAM 后才入批 | 属实 |
| 22 | §4.2、Fig. 4 ∗/† | 页面称入批后逐层等待装载 | **不属实：逐层等待属于 prefill 实例**（F22-1） |
| 22 | §5.1 | TTFT = 排队 + 预测 prefill；传输受拥塞影响 | 属实 |
| 22 | §5.2 | 远端命中可能不如重算；热点复制 | 属实 |
| 22 | FlexGen §1 | 面向延迟不敏感的吞吐任务；GPU/CPU/磁盘三层 | 属实 |
| 22 | FlexGen §4.2、Algorithm 1 | 驻留比例；重叠下一层权重、下一批 cache 装载与上一批存储 | 属实 |

## 4. 分组边界与一致性

- 13（物理块）、14（内容身份与索引）、22（跨层迁移）的归属总体遵守计划 §7，没有三次重述整套缓存机制。13 与 14 都明确把“无引用块能否按内容命中”交给身份层，这一点做得好。
- 不一致：12 把“压缩”归为物理管理，与 22、21 矛盾（F12-3）；0/1 起算混用（F12-6）；22 对 Mooncake 逐层等待的错配也出现在 26（F22-1 跨页提示）。
- 12 与 21 都给出 MLA 的 $L(d_c+d_h^R)$，两处数值一致，属轻度重复，可接受。KV-head 在 TP 下的复制问题已由 30 页覆盖，12 无需重复。

## 5. 摘要（P0/P1 清单）

| 级别 | 位置 | 问题 | 证据 |
|---|---|---|---|
| P0 | `22_kv_tiering_transfer_analysis.md:16` | 把 Mooncake 的逐层 launch/wait 说成 decode 入批后的每层读取边界 | v1 §4.2“Layer-wise Prefill”与 Fig. 4 ∗ 只针对 prefill 实例；decode 仅有 Fig. 4 † 的异步装载与解码重叠 |
| P1 | `14_prefix_caching_analysis.md:17,65–69` | 全命中边界：$m_i\le L_i$、“只算未命中后缀”，漏掉末位置须重算以得 logits | SGLang `Req._compute_max_prefix_len`（`input_len - 1`）；vLLM `max_cache_hit_length = num_tokens - 1` |
| P1 | `14_prefix_caching_analysis.md:59,74` | 淘汰规则缺“只逐出叶子、递归向上”及前缀闭包理由（§5 beat-2） | SGLang v1 §5.1“recursively evicts leaf nodes”；`RadixCache.evict` 只取 `evictable_leaves` |
| P1（机械） | `index.md:20` 对 `12_kv_cache_analysis.md:2,5` | index 行标题与页标题不一致 | 文本比对 |

其余均为 P2：定位错挂、记号冲突、beat-2 可补强、图的中间态与样式。13 页无 P0/P1。
