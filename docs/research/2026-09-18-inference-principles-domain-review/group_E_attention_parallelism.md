# 05 推理原理独立审阅 — E 组：高效 Attention 与推理并行（PCP/DCP/CPP/组合）

> **状态**：审阅进行中（中间稿，防中断先落盘）。已完成：来源开读、锚点核验、文字/公式/表逐句核对、机械门禁；进行中：Python 复算脚本归档、SVG 实际渲染目检。
> **审阅人**：独立审阅（非作者、非本批协调者）。日期 2026-09-18。
> **页面**：`wiki/01_theory/05_inference/` 下 `18_efficient_attention_analysis.md`、`27_prefill_context_parallelism_analysis.md`（+`assets/27_pcp_query_kv_layout.svg`）、`28_decode_context_parallelism_analysis.md`（+`assets/28_dcp_kv_merge.svg`）、`29_chunked_pipeline_parallelism_analysis.md`（+`assets/29_cpp_stage_chunk_timeline.svg`）、`30_inference_parallelism_composition_analysis.md`（+`assets/30_parallelism_two_workloads.svg/.mjs`、`assets/inference_parallel_axes.mjs`）；附带 `wiki/01_theory/06_distributed_parallelism/20_ring_attention_and_context_parallel_analysis.md` §5.2 交接（`git diff`）。

## 已实际打开的来源（版本）

| 来源 | 版本/定位 | 取得方式 |
|---|---|---|
| FlashAttention | arXiv 2205.14135**v2**（arXiv 提交记录：v2 2022-06-23 17:53 UTC）；§2.2 Algorithm 0、§3.1 Algorithm 1/Theorem 1、§3.2 Theorem 2/Proposition 3/Fig. 2、§5 | PDF 分页文本 `SCRATCH/papers/2205.14135v2.txt` |
| FlashAttention-2 | arXiv 2307.08691**v1**（2023-07-17）；§2.3.1、§3.1/§3.1.1、§3.2、§3.3 | `SCRATCH/papers/2307.08691v1.txt` |
| Flash-Decoding 作者文章 | princeton-nlp.github.io/flash-decoding（2023-10-12） | curl 原 HTML 转文本 |
| flash-attention 源码 | `Dao-AILab/flash-attention@6d673cd9610172431bf1b7786d10918dba2783c0`（GitHub API：提交信息 “Bump to v2.2.0”，2023-09-05；`flash_attn/__init__.py` `__version__ = "2.2.0"`） | raw.githubusercontent.com 下载 4 个文件 |
| vLLM Context Parallel Deployment | docs.vllm.ai `v0.20.0`（页尾 January 29, 2026）：Prefill CP / Decode CP / Case study | curl 原 HTML |
| vLLM DCP 技术博客 | vllm.ai/blog/2026-08-07-decode-context-parallelism：§3、§4、§4.1、§5.3、§5.4 | curl 原 HTML |
| vLLM-Ascend CP 设计文档 | `v0.21.0rc` developer_guide/Design_Documents/context_parallel：What is CP、Device Distribution、Block Table、DCP、PCP | curl 原 HTML |
| vLLM-Ascend CP 用户指南 | `v0.21.0rc` user_guide/feature_guide/context_parallel：Supported Scenarios、How to use、Constraints | curl 原 HTML |
| vLLM-Ascend Dynamic CPP | `v0.21.0rc` 设计文档（Problem Statement、Quadratic Latency Model、Runtime Phase、Constraints）；另开同版本用户指南 `dynamic_chunk_pipeline_parallel.html`（Supported Scenarios） | curl 原 HTML |
| VPP | arXiv 2608.26523**v1**（2026-08-27）：Abstract、§1、§2.3、§2.4、§4 | `SCRATCH/papers/2608.26523v1.txt` |
| NVIDIA 软硬协同文章 | 2026-07-10，库内快照 `raw/01_theory/06_distributed_parallelism/NVIDIA_HW_Friendly_LLM_CoDesign_2026-07-10.html`：Large EP、Design for pipeline parallelism（Fig. 8–9、Guideline 6）、Hybrid parallel strategies（Guideline 7）；另取 Fig. 9 原图目检 | 库内快照 + 原图 |
| Sarathi | arXiv 2308.16369v1 §4.2、Fig. 6 | `SCRATCH/papers/2308.16369v1.txt` |

`raw/` 可发现性：各页页首引用的 `raw/01_theory/05_inference/FlashAttention-2205.14135.md`、`FlashAttention2-2307.08691.md`、`FlashDecoding-2023.md`、`vLLM_Context_Parallel-v0.20.0.md`、`vLLM_Ascend_Context_Parallel-v0.21.0rc.md`、`vLLM_Decode_Context_Parallel-20260807.md`、`VPP_Chunked_Pipeline_Parallelism-2608.26523.md`、`vLLM_Ascend_Dynamic_CPP-v0.21.0rc.md` 及 `raw/01_theory/06_distributed_parallelism/NVIDIA_HW_Friendly_LLM_CoDesign_2026-07-10.html` 均存在；29 页正文引用的 Sarathi 也有 `raw/01_theory/05_inference/Sarathi-2308.16369.md`。

## 复算范围（全部独立计算）

- 18：两块 $(m,\ell,u)$、重标定因子 $1/2$、$\ell=3,u=11,o=11/3$；直接法 $22/6$；LSE 式 $L_A=\ln2,L_B=\ln4$，权重 $1/3,2/3$。全部正确。
- 27：连续切分 $1+2=3$、$3+4=7$；head-tail $\{0,3\},\{1,2\}$ 得 $1+4=5$、$2+3=5$；all-gather 接收量 $S(1-1/p)$。全部正确。
- 28：rank0 $m=0,\ell=2,u=4,o=2$；rank1 $m=\ln2,\ell=3/2,u=5,o=10/3$；合并 $\ell=5/2,u=7,o=14/5$；直接法权重 $(1,2,1,1)$ 得 $14/5$；简单平均 $8/3$；LSE 权重 $2/5,3/5$。全部正确。
- 29：3×6 网格忙格 12、总格 18、空格 6、比例 $1/3$；$(P-1)/(M+P-1)$ 在均衡等时假设下成立；两条偏序逐格满足；第 $j$ 块有效 q-k 对 $c^2(j-1)+c(c+1)/2$。全部正确。
- 30：$b_{kv}=2\cdot16\cdot1\cdot64\cdot2=4096$ B；A $131072\times4$ KiB $=512$ MiB；B $64\times2048\times4$ KiB $=512$ MiB；TP4+DCP4 每卡 128 MiB；PCP2×DCP4 每卡 64 MiB；DP8 每卡 $8\times8=64$ MiB；两组 TP4+DCP4 各 32 请求每卡 64 MiB。全部正确。
- 06 §5.2 torchtitan 合并式：`out - sigmoid(block_lse-lse)*(out-block_out)` 与 `lse - logsigmoid(lse-block_lse)` 分别等于按 $e^{lse}$、$e^{block\_lse}$ 加权的归一化输出与 $\log(e^{lse}+e^{block\_lse})$，代数核对正确。

## 机械门禁（轻量）

- `check_links --strict`：482 页，broken/ambiguous/bare_index/stale_section/orphans 全 0。
- `check_math --strict`、`check_markdown --strict`、`check_assets --strict`：`wiki/01_theory/05_inference/`（30 文件）与 06/20 页均 0 error、0 warning。

## 结论表

| page | beat2 | delete-code | figure-trigger | algorithm-replay | spot-check | verdict | note |
|---|---|---|---|---|---|---|---|
| 18_efficient_attention_analysis | pass | pass | transform | pass | 3/3（另加 4 处） | PASS（带 P2） | 11/3 与 LSE 复算、FA/FA2/Flash-Decoding/固定源码均相符；Theorem 2 缺解读、记号不一致为 P2 |
| 27_prefill_context_parallelism_analysis | pass | pass | layout | **FAIL §3** | 3/3（另加 3 处） | REJECT | §3 head-tail 分配与顺序恢复置换只在文字/表中重放，SVG 仅一行脚注“5 : 5”（P1） |
| 28_decode_context_parallelism_analysis | pass | pass | transform | pass | 3/3（另加 4 处） | PASS（带 P2） | 14/5、8/3 与图一致；“去除 8 份冗余”措辞、记号冲突为 P2 |
| 29_chunked_pipeline_parallelism_analysis | pass | pass | timing | pass（待渲染终检） | 3/3（另加 2 处） | PASS（带 P2） | 时间线/气泡/偏序全对；Fig. 9 “GB300” 来源未载、Prefiller 推荐引错文档、页首漏 Sarathi 为 P2 |
| 30_inference_parallelism_composition_analysis | pass | pass | layout | pass | 3/3（另加 3 处） | PASS（带 P2） | 容量账全对，三关分离清楚；个别措辞与引文定位为 P2 |

## 分页发现

### 18 高效 Attention

总体：技术内容正确、忠实；在线 softmax 推导、IO 定理条件、exact 与浮点的区分都准确。未发现 P0/P1。

- **P2 · 已确认** · `wiki/01_theory/05_inference/18_efficient_attention_analysis.md:71` “普通物化 attention 为 $\Theta(Nd+N^2)$，该 FlashAttention 算法为 $\Theta(N^2d^2/M)$”。问题：只列两式，未给出读者比较所需的解读——两者 $N^2$ 项之比为 $d^2/M$，论文紧接定理即写明 $d$ 取 64–128、$M$ 约 100KB 时 $d^2\ll M$，故 HBM 访问少很多倍（2205.14135v2 §3.2 Theorem 2 后一段）。基础读者无法从式子自行得出“为何更少”。建议补一句 $d^2/M$ 的比较及其条件。
- **P2 · 已确认** · `18_efficient_attention_analysis.md:17`、`:75`。§1 称 FA2 §3.1.1 说明“延迟归一化”，§4 称“额外归一化操作都影响最终时间”，但全文未给出这一设计的理由：FA2 §3.1 明言非 matmul FLOP 在 A100 上约贵 16 倍（312 vs 19.5 TFLOPs/s），因此只在循环末尾除以 $\ell$。且 `:75` 该句只引 FA2 §3.2（并行），与“额外归一化操作”所据的 §3.1 不对应。建议在 §2 的 $o=u/\ell$ 处补一句理由并改引 §3.1。
- **P2 · 已确认** · 记号：`:15` 用 $N_q\times N_k$，`:79` 改用 “逻辑分数形状为 $S_q\times S_k$”，而 $S$ 在 `:15` 已是分数矩阵。建议统一为 $N_q\times N_k$。

### 06/20 §5.2 交接（本批改动）

- 数学无丢失：旧 §5.2 的增量式（归一化输出按 $e^{m_{prev}-m}\ell_{prev}/\ell$ 加权）等价于 18 §2 的“反复应用”合并与 §3 的 LSE 加权式；新文对 torchtitan `lse` 是对数量、与 18 的 $\ell$ 不同的说明正确；“fp32 累加不保证逐位相等”比旧文“避免误差”更准确。
- **P2 · 已确认** · 入站说明过时：`wiki/02_engineering/02_train_frameworks/mindspeed/20_mindspeed_context_parallel_analysis.md:11` 仍称其为 06 “§5.2 在线 softmax 公式”的骨架来源；`wiki/02_engineering/02_train_frameworks/torchtitan/13_torchtitan_cp_analysis.md:14` 仍称 online softmax 属 06 理论页。交接后通用公式归 05/18，两处归属描述应更新（链接本身不断）。
- **P2 · 已确认** · `wiki/01_theory/06_distributed_parallelism/20_ring_attention_and_context_parallel_analysis.md:280` 新 §5.2 删去了原“> 骨架取自 `20_mindspeed_context_parallel_analysis.md` §4.4”来源注，而该页页首约定“每节骨架……逐段注明来源”。建议保留一行来源注（或注明本节已改为交接段）。
- 旁注（**既有问题，非本批引入**）：同页 `:601` §9 称“本页分别在 §5.2、§6.2、§7.2 与 §8 保留各机制……公式”，但 §5.2 在改前改后都不含通信量公式。

### 27 PCP

总体：PCP 定义、两条路径、正确性条件、all-gather 接收量、Ascend 的 head-tail/逐层 all-gather/弃 Ring 理由均与来源一致；“实现支持 ≠ 定义”落实到位。

- **P1 · 已确认（rubric check 5）** · `27_prefill_context_parallelism_analysis.md:31`、`:33`、`:37`；SVG 末行文字“本例连续切分负载 3 : 7；head-tail 可变成 5 : 5，但需重排位置。”问题：§3 的因果负载均衡是一个独立的分区+置换算法（head-tail：rank0 取 $\{0,3\}$、rank1 取 $\{1,2\}$，输出须由卡内次序 $(o_0,o_3),(o_1,o_2)$ 置换回 $(o_0,\ldots,o_3)$）。页内只用文字重放；原理图只画连续切分，其“位置顺序恢复”框在连续切分下是平凡拼接，未暴露 head-tail 下真正需要的恢复置换（即 `pcp_allgather_restore_idx` 所解决的问题），5:5 也只以脚注出现。按 rubric“每条独立算法须复用同一例子并有可追踪的 lane/图”，§3 不满足；本审阅简报亦要求核对 SVG 是否重放 3/7→5/5。最小修复：在 SVG 增加 head-tail lane（同 4 位置、两卡分配、各行可见集合、5:5、置换恢复箭头）。
- 无其他实质问题。`:43` “至少还要接收约 $S(1-1/p)$”是在 all-gather 视图内的尺寸账，成立。

### 28 DCP

总体：“不能简单平均”的数学、query 汇集/局部统计/输出回 head 布局三段通信、交错放置、`dcp_size<=tp_size/H` 的性质（实现选择而非数学上界）都与 vLLM v0.20.0、DCP 博客 §4.1、Ascend 设计文档一致。

- **P2 · 已确认** · `28_decode_context_parallelism_analysis.md:55` “MLA 有效 $H=1$、TP8 时用 DCP8 去除 8 份冗余，以及 GQA $H=4$、TP8 时用 DCP2 去除 2 份冗余”。来源原文为 “8x KV cache duplication”“duplication is 2x”，DCP 把 8 倍/2 倍重复降为 1 倍；“去除 8 份冗余”会被读成 8 份多余副本（实为 7 份）。建议改为“把 8 倍（2 倍）重复降为 1 倍”。
- **P2 · 已确认** · 记号冲突：`28:53` 的 $b_{kv}$ 定义为“一层、每个 KV head、每 token 的 K/V 字节”，`30:19` 的 $b_{kv}$ 是“单 token 所有层所有 KV head 的 K/V 字节”（4 KiB）；`28:53` 又用 $d$ 表示 DCP 路数，而 18/30 页 $d$ 是 head dim。同组相邻页同符号异义，建议 28 改用 $b_{kv}^{\text{layer,head}}$ 与 $n_{dcp}$ 之类。
- **P2 · 建议** · `:16`、`:37`：解释正确，但可把正确权重显式写出——两卡局部指数和在共同基准下为 $2:3$，正确权重 $2/5,3/5$，简单平均隐含 $1/2,1/2$；只有各分片全局指数和相等时简单平均才恰好正确。对初学者更直观。

### 29 CPP

总体：CPP 定义（层 stage × 上下文 chunk）、两条依赖（上游同块激活、同 stage 先前块 KV）、每 stage 仅持本段层 KV、均衡假设下的气泡式及其适用条件、等长块不等时（Sarathi Fig. 6、VPP §2.4/§4、Ascend Problem Statement）均正确；Dynamic CPP 与 VPP 的区分准确。

- **P2 · 已确认（来源未载）** · `29_chunked_pipeline_parallelism_analysis.md:64` “NVIDIA 原文据其 DeepSeek-R1 256K prefill、GB300 场景展示 CPP……”。Fig. 9 原图标题为 “DeepSeek-R1 Prefill ISL 256k”，caption 与 alt 文本只写 DeepSeek-R1、256K、PP 1→32，**均未写硬件**；GB300 只见于该文 GEMM 实测段。把 Fig. 9 归到 GB300 是过度归因（疑自 06/21 页页首“数字均基于 GB300”泛化而来）。非承重，建议删去“GB300”或标为推断。
- **P2 · 已确认** · `:66` “vLLM-Ascend 动态 CPP 也推荐用于 Prefiller”引的是 Dynamic CPP **设计文档**；该推荐实际出自同版本**用户指南** Supported Scenarios（“It is better to be used in PD disaggregation scenarios”），设计文档无此句；raw 索引 `vLLM_Ascend_Dynamic_CPP-v0.21.0rc.md` 也只登记设计文档。建议改引用户指南并补入 raw 索引。
- **P2 · 已确认** · `:7` 页首基线未列 Sarathi v1，而 `:58` 正文引用 Sarathi v1 §4.2/Fig. 6（raw 索引存在）。建议补入页首。
- **P2 · 建议（beat-2 可量化）** · `:12–16`、`:35`：“为何要 CPP”只有定性说明，未与基线对比。同一教学假设下，单卡或不分块 PP 处理整段 prompt 需 $P\cdot M=12$ 格，CPP 为 $M+P-1=6$ 格；补此对照即可让读者看到 CPP 缩短 TTFT 的来源与上限 $PM/(M+P-1)$。
- **P2 · 已确认** · 措辞：`:20` “允许一个 stage 在某格只处理一个 chunk”应为“每格至多处理一个 chunk”；`:54` “最后 stage 只完成 $C_1$ 时可以有这一块的中间 logits”——此时 $C_1$ 各位置已过全部层，得到的是这些位置的最终层输出（若需 prompt logprobs 可算其 logits），“中间 logits”易误解。

### 30 并行组合

总体：容量账全部正确；TP 在 KV head 少于 TP 度时的复制、DCP 复用 TP rank、PCP 扩设备域（`world_size=tp*pcp`、`cp_size=pcp*dcp`）、Ascend 约束、NVIDIA 宽 EP / CPP / Helix 的场景限定均与来源一致；“数学可组合 / 实现支持 / 性能合适”三关在 §5 逐关给出具体内容，达成本页特有验收。

- **P2 · 已确认** · `30_inference_parallelism_composition_analysis.md:66` 把 `cp_size = pcp_size * dcp_size` 与“Device Distribution”一起引向**用户指南** URL；该式实际在**设计文档** “Block Table” 节，“Device Distribution” 也是设计文档章节。建议改为分别引设计文档 Block Table/Device Distribution 与用户指南 How to use/Constraints。
- **P2 · 已确认** · `:37` “推理前向无跨副本梯度归约”——推理本无梯度，此句对比训练 DP 才有意义；建议改为“DP 副本间在前向中无需任何 attention 通信”。
- **P2 · 已确认** · `:58` “一层内部可能由 attention 和 FFN 喜欢不同的布局”语句不通，应为“一层内的 attention 与 FFN 可能适合不同布局”。
- 记号冲突见 28 节（$b_{kv}$）。

## 锚点抽查记录

| 页 | 来源与定位 | 页面所称 | 结果 |
|---|---|---|---|
| 18 | 2205.14135v2 §2.2 Algorithm 0 | 标准实现把 $S$、$P$ 写到 HBM | 相符 |
| 18 | 2205.14135v2 §3.1 Algorithm 1 / Theorem 1；§3.2 Theorem 2（$d\le M\le Nd$） | 分块+统计合并、$O(N^2d)$ FLOPs、$\Theta(Nd+N^2)$ vs $\Theta(N^2d^2/M)$ | 相符 |
| 18 | 2205.14135v2 §3.2 Fig. 2（GPT-2 medium、1024、A100，fwd+bwd；块大小 >256 转其他瓶颈）；§5 跨架构可移植性 | 同左 | 相符 |
| 18 | 2307.08691v1 §2.3.1、§3.1.1（未缩放 $\tilde O$、只存 logsumexp）、§3.2（沿序列长度并行、batch×heads 小时提高占用） | 同左 | 相符 |
| 18 | Flash-Decoding 文章 “Multi-head attention for decoding”“A faster attention for decoding”（三步：切 KV、并行算局部 attention 并写每行每 split 的 LSE、按 LSE 归约；两 kernel） | 同左 | 相符 |
| 18 | `csrc/flash_attn/src/flash_fwd_kernel.h::softmax_rescale_o`（维护 `scores_max`/`scores_sum`，`exp2f((max_prev-max_cur)*scale)` 重标定 `acc_o`） | 块合并 | 相符 |
| 18 | `flash_attn/flash_attn_interface.py::flash_attn_with_kvcache` docstring：`num_splits` “split the key/value into this many chunks along the sequence” | 同左 | 相符 |
| 18 | `flash_fwd_launch_template.h::run_flash_splitkv_fwd`（`num_splits>1` 时启动 `flash_fwd_splitkv_combine_kernel`）；`flash_fwd_kernel.h::combine_attn_seqk_parallel`（对各 split LSE 求 logsumexp，按 `expf(lse-lse_logsum)` 缩放累加 `Oaccum`） | split-KV 恢复 | 相符 |
| 27 | vLLM v0.20.0 “Prefill Context Parallel”：两策略 partial Q/full KV、partial Q/partial KV；“Both approaches are under active development”；TTFT 目标 | 同左 | 相符 |
| 27 | Ascend v0.21.0rc 设计文档 PCP：补齐到 2·pcp_size、切 2p 段首尾配对、`pcp_allgather_restore_idx`、`_update_tokens_for_pcp`；仅当前层 all-gather KV 用后即弃；Ring 因开发复杂度高、重叠收益有限未选 | 同左 | 相符 |
| 27 | Ascend 用户指南 How to use：`world_size = tensor_parallel_size * prefill_context_parallel_size` | 同左 | 相符 |
| 28 | vLLM v0.20.0 “Decode Context Parallel”：TP 先切 H；重复 tp_size/H 倍；dcp ∈ [1, tp_size/H]；超出在数学上可行但非 attention 层用途不明；交错放置；case study（R1 TP8→DCP8、Qwen3-235B H=4 TP8→DCP2）；MLA/GQA 支持、部分 backend 支持 MTP | 同左 | 相符（措辞 P2 见上） |
| 28 | vLLM DCP 博客 §4.1：AllGather Q → Compute → AllGather+ReduceScatter（`cp_lse_ag_out_rs`）；MLA 可 opt-in 复制 query 投影跳过 Q all-gather | 同左 | 相符 |
| 28 | Ascend 设计文档 DCP / Block Table：GQA 沿 head 维 all-gather Q、`cp_lse_ag_out_rs`、all-to-all 替代；`block_size % cp_kv_cache_interleave_size == 0` | 同左 | 相符 |
| 29 | NVIDIA “Design for pipeline parallelism”：CPP 同时切层与上下文 chunk（Fig. 8）；Fig. 9 DeepSeek-R1 256K prefill、PP 增大 FTL 降而 tokens/s/GPU 近常数；Guideline 6；PP 在 P/D 分离背景下相关 | 同左 | 相符，但“GB300”无出处（P2） |
| 29 | VPP 2608.26523v1 §1、§2.4（CPP 定义；等长块后块更慢致气泡；DCPP）；§4（DCPP 调度开销） | 同左 | 相符 |
| 29 | Ascend Dynamic CPP 设计文档：启动 profiling 64 个 chunk 尺寸、二次模型、运行时校准；Constraints：PP>1、chunked prefill 必需、启动开销数十秒 | 同左 | 相符；“推荐用于 Prefiller”在用户指南（P2） |
| 29 | Sarathi 2308.16369v1 §4.2 / Fig. 6：后块重读前块 KV | 同左 | 相符 |
| 30 | Ascend 设计文档 Block Table：`cp_size = pcp_size * dcp_size`；Device Distribution：PCP 新通信域、DCP 复用 TP；用户指南 Constraints（MLA/GQA 的 dcp 约束、KV 传输需 interleave=block_size） | 同左 | 相符（引文定位 P2） |
| 30 | NVIDIA “Hybrid parallel strategies”：低并发下 attention 与 FFN 分别并行；TP 只扩到 KV head 数；Helix 沿序列切 KV 后同批 GPU 做 TP×EP；NVLink 吸收通信 | 同左 | 相符 |
| 30 | vLLM v0.20.0 PCP “under active development” | 同左 | 相符 |
