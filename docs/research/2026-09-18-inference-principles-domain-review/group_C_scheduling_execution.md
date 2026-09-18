# 推理原理域独立审阅 — C 组：调度、Chunked Prefill、P/D 分离、执行优化

- **审阅日期**：2026-09-18。审阅者未参与这些页面的写作；wave 账本中的“非作者审阅 PASS”未被采信。
- **页面**（均在 `wiki/01_theory/05_inference/`）：`15_continuous_batching_analysis.md`、`16_chunked_prefill_analysis.md`、`26_prefill_decode_disaggregation_analysis.md`、`35_inference_execution_optimization_analysis.md`（含 `assets/35_decode_execution_overlap.svg` 与 `.mjs`）。
- **实际打开的来源（版本）**：
  - Orca，OSDI 2022 正式 PDF（USENIX `osdi22-yu.pdf`，论文集 pp. 521–538；下文页码均为论文集页码，PDF 第 N 页对应论文集第 519+N 页）。
  - PagedAttention，arXiv:2309.06180v1。
  - Sarathi，arXiv:2308.16369v1；对照用 Sarathi-Serve，arXiv:2403.02310v1。
  - DistServe，OSDI 2024 正式 PDF（USENIX `osdi24-zhong-yinmin.pdf`，pp. 193–210；下文页码均为论文集页码，PDF 第 N 页对应论文集第 191+N 页）。
  - Mooncake，arXiv:2407.00079v1。
  - vLLM 文档源文件 **v0.20.1 tag**（GitHub raw：`docs/configuration/optimization.md`、`docs/design/{torch_compile,cuda_graphs,fusions,dbo,optimization_levels}.md`）；vLLM main 检出 `199cb9b9`（2026-09-07）的 `docs/configuration/optimization.md`；2026-09-18 在线 `docs.vllm.ai/en/latest/configuration/optimization/`。
  - PyTorch **v2.8.0 tag** `docs/source/notes/cuda.rst`（“Asynchronous execution / CUDA streams / CUDA Graphs / Constraints / Usage with multiple streams / Graph memory management”）。
- **复算**：用 `.venv` Python 独立复算 T15 四轮账本、T16 两条路线（外加页面遗漏的第三条路线）、T26 交接时间线与容量上界、T35 的 12→10 依赖调度和缓冲生命周期（脚本 `SCRATCH/groupC/recompute.py`）。页面上的所有算术数字**全部复算一致**；问题出在解释与来源归属，不在算术。
- **图**：用仓库自带 mermaid 11.17.2 与 headless Chrome 实际渲染 15/16/26 的 3 个 Mermaid 块（均解析通过）并目检；把 35 的 SVG 渲染为 PNG 目检，并逐条核对 `<text>`、`<path>` 坐标与 `.mjs` 生成逻辑。
- **机械门禁**：`check_links --strict`（482 页）broken/ambiguous/bare_index/stale_section/orphans 全为 0；`check_math`、`check_markdown`、`check_assets` 以 `--strict` 对 4 页检查，0 error、0 warning。
- **raw 可发现性**：页首引用的 8 个 raw 索引路径全部存在。Mooncake 的 raw 索引未钉住 v1（见 P2-26-4）。
- **hop-walk**：四页都没有对源码行为作出论断（35 页只引用版本化文档），故不适用，表中不列。
- **页首约定**：四页都有四行页首（文献基线或文档基线 / 主题 / 适用范围 / 最近更新）；Related Pages 各 5 条，每条附一句说明。

## 裁定表

| page | beat2 | delete-code | figure-trigger | algorithm-replay | spot-check | verdict | note |
|---|---|---|---|---|---|---|---|
| 15_continuous_batching_analysis | pass（§1 selective batching 除外） | pass | timing | pass（Mermaid 逐轮重放四轮账本） | 9/9（有一处页码 P2） | **REJECT** | P1：§1 第 26 行只点名 selective batching，未解释其机制与依据；页面自己的第 2、3 轮混合批恰恰依赖它。修一段即可。 |
| 16_chunked_prefill_analysis | **FAIL §3–§4** | pass | timing + transform（因果 mask/历史 KV） | **FAIL §2**（图只重放 token 计数，未画历史 KV 读取） | 4/4 | **REJECT** | P1 ×3：算例把“分块”与“优先级”混为一谈；Sarathi 的 piggyback/decode-maximal batching 未解释，ITL 框架来源未交代；决定性的历史注意力没有原理图。 |
| 26_prefill_decode_disaggregation_analysis | pass | pass | timing | pass（时序图重放 8 MiB 交接），但 D 侧“可读”的来源映射错误 | 5/7 | **REJECT** | P0：把 Mooncake §4.2（prefill 侧逐层 load/store）当作 D 侧的逐层装载等待。P1：误述 DistServe §6.3 的实验条件，漏掉按带宽放置；全页没有 goodput。 |
| 35_inference_execution_optimization_analysis | pass | pass | timing（+ 缓冲生命周期） | pass（SVG 有依赖箭头与生命周期条） | 12/12 | **PASS（带 P2）** | vLLM v0.20.1 与 PyTorch 2.8 的论断逐条吻合。P2：图捕获约束漏了“捕获内禁止 CPU–GPU 同步”；SVG 的 C→J 箭头落在 V 上并压住标签。 |

---

## 15 — Continuous Batching 与请求调度

总体：三类调度粒度划分、状态机、联合准入、Orca 的 FCFS/预留与 PagedAttention §4.5 的引擎策略归属都清楚，账本可审计。唯一的实质缺口是 Orca 的第二项核心技术。

### P1-15-1（已确认）selective batching 只点名，未解释

- **位置**：`wiki/01_theory/05_inference/15_continuous_batching_analysis.md:26`
- **原文**：「Orca 还专门引入 selective batching 处理这些形状差异。」
- **问题**：Orca 的贡献是两项：iteration-level scheduling **和** selective batching。本页只写了后者的名字和“处理形状差异”这一目的，没有写它是怎样执行的，也没有写为什么这样做成立。它的机制是：非 Attention 算子（Linear、LayerNorm、Add、GeLU）把本轮所有请求的 token 压平成 $[\sum L, H]$，按 token 成批计算；只有 Attention 按请求拆开（Split → 每请求 Attention，读 K/V manager 中的历史 → Merge）。成立的依据是 Attention 不带参数，拆开它几乎不损失“权重读取一次、多请求复用”的批处理收益。页面自己的账本第 2 轮（R1 decode + R2 prefill）和第 3 轮（R1 decode + R3 两 token prefill）正是 Orca C2 所说“不能直接组 batch”的混合形状，读者读完不知道这样的一轮是怎么执行的。基础读者会期待这里有这条标准直觉，因此定为 P1。
- **证据**：
  - Orca §3 C2（p. 525）：列出三种不能组 batch 的情形。
  - Orca §3 S2 与 Fig. 5（pp. 525–526）：“flattening … $[\sum L,H]=[5,H]$”；“splits the batch and processes each request individually for the Attention operation while applying token-wise … batching to other operations”；输入为 $[7,H]$。
  - Orca §1（p. 522）：“Since the Attention operation is not associated with any model parameters, applying batching to Attention has no benefit …”
  - Orca §6.1（p. 530）：与 FT 效率相当。
  - PagedAttention v1 §4.3（p. 6）：“concatenates all the input tokens of the current iteration … as one sequence”。
- **建议**：在 §1 补一段 selective batching 的机制与依据。最好在图或表中给第 3 轮加一行：`[b, D, E]` 压平为 $[3,H]$ 进入 Linear/MLP；Attention 拆开计算，R1 读 A、B、a 的 KV，R3 在 D、E 上做因果注意力。

### P2-15-2（已确认）页码

- **位置**：`:70`
- **原文**：「[来源事实：Orca §4.2，Algorithm 1，p. 527]」
- **问题**：§4.2 从 p. 527 开始，但 Algorithm 1 及 `max_tokens` 预留那段（第 23–26 行）在 p. 528。
- **建议**：改为 “pp. 527–528”。

### P2-15-3（已确认）指代不清

- **位置**：`:133`
- **原文**：「暂不接纳新请求，直到被抢占序列完成后再恢复它们」
- **问题**：“它们”指代不清。PagedAttention §4.5（p. 8）的原话是：“vLLM stops accepting new requests until all preempted sequences are completed. Once a request completes, its blocks are freed …, and the blocks of a preempted sequence are brought back in”。
- **建议**：改写为“停止接纳新请求，直到被抢占序列全部完成；其他请求完成释放块后，再把被抢占序列的块换回继续执行”。可顺带补一句：同一请求的多条序列（如 beam）作为 sequence group 一起被抢占或重调度。

### P2-15-4（已确认）读者正文出现内部任务编号

- **位置**：`:141`
- **原文**：「因此 T15 的账本…」
- **问题**：“T15”是计划内部编号，不应出现在读者正文里。另外，把 Orca §6.2 称为“反例提醒”不贴切，那是作者给出的适用性保留。
- **建议**：改为“本页账本”“作者的保留说明”。

### P2-15-5（建议）原理图可读性

- **位置**：`:97–116`
- **问题**：Mermaid 纵向链基本是表格的重排。它能通过重放要求，但若改成“请求 × 轮次”的占用泳道（请求位与 KV 条），可以一眼看出“R2 释放 → R3 加入”和 KV 6/6 的顶格。
- **说明**：非缺陷，可选改进。

**复算**：四轮的输入 token 为 2/2/3/1，轮中 KV 为 2/4/6/3，释放依次为 R2:1、R1:4、R3:3，与页面逐项一致；第 2 轮 R3 只被请求数上限挡住（token 与 KV 都还放得下），与正文一致。

**初学者可读性**：先给问题和三种粒度，再给最小账本，顺序很好。§2 的分段状态式对初学者偏重，但每一项都有文字解释。主要缺口仍是 P1-15-1：读者不知道不同阶段、不同长度的请求如何在同一轮里执行。

---

## 16 — Chunked Prefill

总体：§2 的可见集合公式、跨块 KV 重读、vLLM 策略的“某引擎策略”标注都正确。但页面核心的比较算例有混淆变量，Sarathi 本身的机制没有讲，决定性不变量没有图。

### P1-16-1（已确认）算例把“分块”与“调度优先级”两个变量混在一起比较

- **位置**：`:45–50`、`:64`
- **原文**：「整段策略令已就绪的 $D_1,D_2$ 至少多等一轮」
- **问题**：
  - “整段”一栏默认采用 **prefill 优先**：L 的 8 token 占满轮 1，已就绪且更早到达的 D1、D2 等待，这也违背 Orca 的 iteration-level FCFS。“分块”一栏则是 **decode 优先**。
  - 在页面自己的 B=8、K=14 下，被省略的“整段 + decode 优先”路线：轮 1 只跑 D1、D2（2 token；2+8>8，L 放不进，等待）；轮 2 跑 L 的 8 token。这条路线同样在轮 1 完成 D1、D2、轮 2 完成 L 的 prompt，轮次结果与分块完全相同，轮中 KV 峰值只有 **8**（分块 10，整段 prefill 优先 14）。
  - 因此，“D1、D2 至少多等一轮”是 prefill 优先带来的，不是“整段”本身的性质；14 对 10 的峰值差来自驻留顺序（D1、D2 在 L 写满之前已经释放），也不是分块带来的。页面 §2 自己也说分块不减少最终 KV。
  - D1、D2 只做一次 decode 就结束，这也掩盖了分块真正解决的两难：decode 持续存在时，不分块的 decode 优先会在 $B-d<I$ 时让 L 一直进不来，prefill 优先则让 decode 停顿，分块使两者每轮都能推进。
  - 算例没有时间模型，也无法体现 ITL 收益。
- **证据**：
  - 复算脚本输出：整段 prefill 优先峰值 14；分块 decode 优先峰值 10；整段 decode 优先峰值 8；三者窗口均为 10 token、均在 2 轮完成。
  - DistServe §2.3（p. 196）：“Prioritizing tasks in either phase adversely affects the latency of the other”。
  - Sarathi-Serve v1 摘要：stall-free schedules。
- **建议**：把整段栏标为“整段 + prefill 优先”，补“整段 + decode 优先”第三栏；让 D1、D2 持续 ≥2 轮，或让 $I>B-d$，以显示不分块时 L 进不了批；把 KV 峰值差明确归因于执行顺序；可加一个每轮时长模型（如 $t=a+b\cdot\text{tokens}$）来展示最大 ITL。

### P1-16-2（已确认）Sarathi 的核心机制与动机未解释，ITL 框架来源未交代

- **位置**：`:16`、`:68`、§3 `:31–41`；框架见 `:14`
- **原文**：`:16`「…并据此提出 chunked-prefills 与 decode-maximal batching」
- **问题**：
  - decode-maximal batching / piggybacking 是混合批的收益来源，全页没有解释：一个 prefill chunk 加上用 decode 填满其余槽位；所有线性层融合为一次 GEMM，decode token 搭乘计算受限的 prefill 复用同一次权重读取；Attention 分开计算。§3 只写了 $d+c_q\le B$ 和政策，`:41` 转述 vLLM 文档时也漏掉了它的第二条收益“locating compute-bound (prefill) and memory-bound (decode) requests to the same batch”。
  - Sarathi v1 的动机是 decode 低效和流水线气泡（吞吐），不是 ITL。页面 §1 的“长前向拉长 token 间隔”和 §3 的 token 预算/decode 优先，实际来自 Sarathi-Serve（stall-free batching + token budget）与 vLLM 文档。页面唯一的论文基线却是 Sarathi v1，也没有引用 Sarathi-Serve。
  - 我没有找到明确把 Sarathi-Serve 观点写成 Sarathi v1 的句子，因此不定 P0；但读者很容易误以为所引论文提出分块是为了保护 ITL。
- **证据**：
  - Sarathi v1 摘要与 §1（pp. 1–2）；§3.2 与 Fig. 5（pp. 4–5，PP 气泡）；§4.3、§4.3.1（pp. 6–7）：“fuse all the linear operations, while letting the attention computations … happen separately”。
  - Sarathi v1 Table 2（p. 7）：decode 每 token 12.49 ms → 1.2 ms。
  - Sarathi-Serve v1 摘要（p. 1）：“inspired by the techniques we originally proposed for optimizing throughput in Sarathi … leverages chunked-prefills from Sarathi to create stall-free schedules”。
- **建议**：补一段 piggyback/decode-maximal batching（为什么混批划算）；说明 Sarathi v1 的吞吐与 PP 气泡动机；为 stall-free/token budget/ITL 框架补上 Sarathi-Serve 来源（arXiv id+版本，并建 raw 索引），或把它标为分析推断与 vLLM 文档政策。

### P1-16-3（已确认，rubric）决定性不变量没有原理图

- **位置**：§2 `:20–29`；图 `:52–62`
- **问题**：页面标题的“同时保留因果历史”以及计划中的专项验收“分块不能隐去历史注意力”，是本页的决定性不变量。masking 在机制 profile 中明确属于算法触发项，但唯一的图只重放每轮 token 与 KV 计数。它没有画出块 2 的 query $x_5..x_8$ 读取 $x_1..x_4$ 的 KV 并在块内做因果三角（即 Sarathi Fig. 6 的 mask），也没有体现重读代价（Sarathi §4.2：N 块时首块 KV 被读 N 次；DistServe §2.3 的 $O(N^2)$）。
- **建议**：对同一个 8-token、c=4 的例子加一个 mask/KV 读取小图：块 1 为 4×4 下三角；块 2 为 4×8，其中前 4 列全可见、后 4 列为下三角。或至少在 B2 节点标出“读取历史 KV 4”，再配 mask 网格。

### P2-16-4（已确认）术语未定义

- **位置**：`:68`
- **原文**：「其具体 256-token 例子与 tile 结论」
- **问题**：“tile 结论”对基础读者是未定义术语。Sarathi §4.4 与 Fig. 7（p. 8）的内容是：matmul tile 量化，因此 chunk 加上搭载的 decode 数应是 tile 的整数倍（如 256−(B−1)）；128→256 token 耗时 +27%，257 token 再 +32%。
- **建议**：用一句话解释，或删去。

### P2-16-5（已确认）页首章节与正文不一致

- **位置**：`:7`
- **问题**：页首只钉住 Sarathi “§4.2–4.4”，正文 `:16` 还引用了 §3.1–3.3。
- **建议**：同步页首。

### P2-16-6（已确认）来源事实与分析备注混在一起

- **位置**：`:27`
- **原文**：「浮点实现不保证逐位相同。[来源事实：Sarathi v1 §4.2，Fig. 6]」
- **问题**：Sarathi 只说 “mathematically equivalent”；浮点一句是分析备注，却放在“来源事实”引注之前。
- **建议**：移出引注或标为分析推断。

### P2-16-7（已确认）读者正文出现内部任务编号

- **位置**：`:9`、`:39`
- **问题**：出现 “T27/T29/T14/T15”。
- **建议**：改用页面名或链接。

### P2-16-8（已确认）索引别名与页面标题不一致

- **位置**：`wiki/01_theory/05_inference/index.md:24`
- **问题**：别名「Chunked Prefill：长输入的分步执行」与页面 H1「Chunked Prefill：把长输入拆到轮次之间，同时保留因果历史」不一致。这是计划名与最终标题之差，含义不冲突，故定 P2 而非 P1。15 的索引行（`index.md:23`）用的是截短标题，可接受。

### P2-16-9（已确认）vLLM 文档基线未钉版本

- **位置**：`:7`
- **问题**：vLLM 文档以 “latest + 访问日期” 作为基线。本次核对，v0.20.1 tag 源文件、main@199cb9b9 源文件、2026-09-18 在线 latest 三处文字一致，页面转述准确。
- **建议**：可改钉 v0.20.1（35 页已使用该版本），便于复现。

**复算**：两条路线窗口都是 10 token；轮中 KV 为整段 12→14、分块 10→8；终态 L=8。与页面一致（另见 P1-16-1 的第三条路线）。

**初学者可读性**：§1 的直觉（长前向拉长 token 间隔）好懂，§2 的可见集合公式简洁。但“为什么把 prefill 块和 decode 放进同一批反而更省”（piggyback）没有讲；“算术强度”“tile”在首次出现处既未解释，也未就地链接 11 页的 Roofline 小节（11 页只出现在文末 Related Pages）。读者能学会“怎么切”，却学不到“为什么这样混批划算”。

---

## 26 — Prefill/Decode 分离

总体：交接边界（收全 / 可读 / 首个 decode 完成）、DistServe §2.3 被否决的替代方案、pull 缓冲、Mooncake 提前拒绝与负载振荡、P/D 与 PCP/DCP 的区分，都写得清楚，时间线算术正确。问题集中在两处来源映射和一个缺失的核心概念。

### P0-26-1（已确认）把 Mooncake §4.2 的 prefill 侧逐层加载说成 D 侧的“可读”条件

- **位置**：`:27`、`:66`
- **原文**：`:27`「…后请求进入 D 的 continuous batch，实际逐层 GPU 计算前还等待对应装载。」；`:66`「`D` 的“可读”对应 §4.2 逐层加载等待。」
- **问题**：Mooncake v1 §4.2 “Layer-wise Prefill”（p. 7）讲的是 **prefill 实例**：“Before each layer's attention computation begins, the model waits for the asynchronous loading of that layer's KVCache … After the attention calculation is complete, asynchronous storage of that layer's KVCache is launched”。这是前缀缓存 load/store 与 prefill 计算的重叠。关于 decode 实例，论文只在 Fig. 4 注（†）（p. 6）写了 “For decoding instances, asynchronous loading is performed concurrently with GPU decoding to prevent GPU idle time”，以及 §3 第 4 步（p. 6）“After all the KVCache is received in the CPU DRAM of the decoding node, the request joins the next batch”。论文从未描述 D 侧的逐层等待。页面把论文没有的主张归给了 §4.2；按 brief 规则，这属于 P0。
- **建议**：D 侧改引 Fig. 4（†）的“与 decode 并发的异步装载”；把 §4.2 的逐层等待挪到 P 侧（KVCache Reuse / Incremental Prefill）。D 侧“装载完成才可读”作为通用可见性条件保留，但标为分析推断。

### P1-26-2（已确认）误述 DistServe §6.3 的实验条件，漏掉按带宽放置

- **位置**：`:82`
- **原文**：「DistServe 的 Fig. 10 在其 OPT 模型、ShareGPT、所用高速网络下观察到传输延迟占比很小；…不能推出低带宽节点间…也可忽略」
- **问题**：DistServe 的实验恰恰是在**低带宽节点间网络**上做的：跨节点只有 25 Gbps。传输占比小，是因为 §4.2 的放置算法把同一 stage 的 P/D 段放在同一节点，走 NVLink。“按集群带宽放置两阶段”是 DistServe 摘要列出的核心贡献之一，全页没有提到。页面的警示句方向（“不能外推”）本身合理，但给出的条件与论文相反，也漏掉了真正起作用的手段。
- **证据**：
  - §6.1（p. 201）：“The cross-node bandwidth is 25Gbps … we use the low node-affinity placement algorithm”。
  - §6.3（p. 203）：“less than 0.1% of the total latency … over 95% of requests experience a delay of less than 30ms, despite our testbed having only limited cross-node bandwidth. This is due to the algorithm described in §4.2 … enabling the use of intra-node NVLINK”。
  - §4.2（pp. 199–200）与 Algorithm 2。
  - §3.3（p. 198）：OPT-66B 512-token KV ≈ 1.13 GB；10 rps 需要 11.3 GB/s ≈ 90 Gbps。
- **建议**：把这句改为“在 25 Gbps 节点间网络上，靠同节点 NVLink 放置得到”；在 §4 或 §5 补一段“放置是控制传输代价的主杠杆”；可复用 §3.3 的带宽算术作为第二个可复算例子。

### P1-26-3（缺失已确认；严重度为审阅判断）全页没有 goodput

- **问题**：DistServe 的优化目标是 per-GPU goodput，即在 SLO 达成率目标（如 90%）下每 GPU 可服务的最大请求率。Mooncake §2 同样以 goodput 为目标，并只计入完整完成的请求。本页 §4 的容量式是吞吐的必要条件（页面自称“尚未保证尾延迟”），读者因此看不到两件事：
  - 为什么在 SLO 约束下，即使混部的原始吞吐更高，分离仍可能胜出；
  - DistServe 自己给出的收益边界：离线、吞吐导向场景下 chunked-prefill with piggyback 可能更好；只有很少 GPU 时，设计空间受限。
- **证据**：
  - DistServe §1（p. 194）goodput 定义；§2（p. 195）；§4 与 Algorithm 1（pp. 198–199，以 goodput/num_gpus 选配置）。
  - DistServe §1 的最小例子（p. 194）：混部 1.6 rps/GPU；仅 prefill 5.6、仅 decode 10；2P1D 合计 10 rps，即 3.3 rps/GPU，2.1×。
  - DistServe §7（pp. 204–205）：吞吐导向与资源受限场景。
  - Mooncake v1 §2（p. 4）。
- **本域已有的相关定义**：11 页的“有效输出吞吐 $\Theta_{\mathrm{good}}$”（11:55）和 40 页的 $Q_{\mathrm{good}}$（40:45）都是在给定负载下实测的达标量；DistServe 的 goodput 是“达成率 ≥ 目标时可承受的最大请求率（按每 GPU 计）”，属于容量量。二者不能混用。
- **建议**：给出 DistServe 式 goodput 定义，链接 11 与 40 页，并注明上述区别（AIPerf 的 Goodput 同样是实测量）。在 §5 补上 DistServe §7 的边界。

### P2-26-4（已确认）页首章节不全；Mooncake raw 索引未钉 v1

- **位置**：`:7`
- **问题**：Mooncake 页首写 “§2–5”，但正文还引用 §6.2–6.4（`:78`）和 §7（`:82`）。raw 索引 `raw/01_theory/01_models/moonshot_kimi/Mooncake_KVCache_Disaggregated-2407.00079.md` 链接的是不带版本的 arXiv id，且写着“最后更新 2025-09-03”，没有记录页面所用的 v1。

### P2-26-5（已确认）两个例子的带宽设定容易混淆

- **位置**：`:31` 与 `:76`
- **问题**：§3 时间线假设单次传输有效 1 GiB/s，§4 又说“网络给该服务 80 MiB/s”。两者是独立的教学设定，但没有说明，读者容易当成同一条链路。
- **建议**：注明两者相互独立。

### P2-26-6（已确认）遗漏项列举不全

- **位置**：`:48`
- **原文**：「仅用 12 ms prefill 与 2 ms decode…账目便少了本例的 8.8125 ms、2.153125 ms 和 4 ms」
- **问题**：“12+2” 也漏了 3 ms 的 P 排队。
- **建议**：一并列出，或改说“只计两端计算”。

**复算**：23.8125 / 25.965625 / 29.965625 / 31.965625 ms，传输 8.8125 ms、装载 2.153125 ms；2P+1D 上界 min(8,5,10)=5，2P+2D 上界 min(8,10,10)=8，全部一致。P/D 与 PCP/DCP 的区分（`:84`）正确。

**初学者可读性**：职责表、交接时间线和时序图对初学者很友好，“首个 decode 完成 ≠ TTFT”的提醒也到位。但缺少 goodput 这把标准尺子，读者无法把“配比”与 SLO 联系起来（P1-26-3）。

---

## 35 — 推理执行优化

总体：四类优化分别节省什么、代价是什么，写得清楚；vLLM v0.20.1 与 PyTorch 2.8 的论断逐条吻合；12→10 的依赖演算与生命周期正确。SVG 满足“有依赖和生命周期，而不只是并排色块”的专项验收：U→C、C→汇合、J→CPU 都有箭头，B、通信载荷、结果都有生命周期条。

### P2-35-1（已确认）图捕获约束漏了“禁止 CPU–GPU 同步”

- **位置**：`:37`
- **问题**：约束列举漏掉“捕获内禁止 CPU–GPU 同步操作（如 `.item()`）”，“动态形状被禁止”也只是隐含表达。PyTorch 2.8 “Constraints” 原文：“Ops that synchronize the CPU with the GPU (e.g., `.item()` calls) are prohibited”；“Dynamic shapes are prohibited”。推理中，采样侧的 `.item()` 和数据相关的尺寸是最常见的捕获破坏者。
- **建议**：补一句。

### P2-35-2（已确认）SVG 的 C→J 箭头指错对象并压住标签

- **位置**：`assets/35_decode_execution_overlap.svg`
- **问题**：C→J 路径 `M 848 552 V 479 H 923 V 465` 的箭头落在 V 的右下沿（x=923，而 J 从 x=934 起），水平段还穿过“C 与 V 汇合”标签（x≈922、y=476）。目检渲染图，容易读成 C→V。
- **建议**：把箭头终点移到 J 框（x≈974），并移开标签。

### P2-35-3（存疑，轻微）“写 B”画在 CPU 泳道

- **位置**：`:45` 及 SVG
- **问题**：“更新静态输入缓冲 B”画在 CPU 泳道上。实际多为流上排队的 H2D 拷贝；pinned staging buffer 也要等拷贝完成才能被 CPU 覆写，这是另一条生命周期边。
- **建议**：可加注，不影响结论。

**复算**：串行 12；重叠后 C 为 [5,7)、V 为 [5,8)、J 为 [8,9)、CPU 消费为 [9,10)，总长 10；B 生命周期 [2,8]，通信缓冲到 7，结果在 9 可见。页面、`.mjs` 断言、SVG 坐标三者一致。

**初学者可读性**：从“一次 decode 的 CPU/GPU 交接”讲起，四类费用表清楚。vLLM 专有名词较密，但每处都标了版本和适用范围，guard 等编译概念也链接到了 PyTorch 工程页。可读性合格。

---

## 锚点抽查记录

| 页面 | 来源与定位 | 页面主张 | 结果 |
|---|---|---|---|
| 15 | Orca §2 Fig. 1（pp. 522–523） | iteration 定义；initiation 一次处理全部输入，increment 每轮一个 token | ✓ |
| 15 | Orca §3 S1 Fig. 4（p. 525） | 选请求 → 引擎只跑一轮 → 收结果；新请求等一个 iteration 后可被考虑；完成即返回 | ✓ |
| 15 | Orca §3 C2/S2 Fig. 5（pp. 525–526） | selective batching 存在，用于处理形状差异 | ✓（但未解释，见 P1-15-1） |
| 15 | Orca §4.2 Algorithm 1（pp. 527–528） | iteration-level FCFS 定义；`max_bs`；首次调度预留 `max_tokens`；死锁动机 | ✓（页码 P2） |
| 15 | Orca §6.2 Fig. 10（p. 532） | 调大 max_bs 提高吞吐而未伤延迟，但对任意硬件/模型/负载无保证 | ✓ |
| 15 | PagedAttention v1 §2.3（p. 3）、§4.3（p. 6）、§4.5（p. 8）、§6.2（p. 11） | 逐轮增删、专用 kernel 免 padding；逐轮选候选并分配块；FCFS、all-or-nothing、swap/recompute、抢占后停收新请求；超过容量后延迟发散 | ✓（§4.5 措辞 P2） |
| 16 | Sarathi v1 §3.1–3.3（pp. 4–5） | LLaMA-13B/A6000 下 prefill 与 decode 的算术强度和吞吐差异 | ✓（§3.2 的 PP 气泡动机未写出） |
| 16 | Sarathi v1 §4.2 Fig. 6（p. 6） | 跨块因果 mask，数学等价；小块降低算术强度；重读先前块 KV | ✓ |
| 16 | Sarathi v1 §4.3–4.4（pp. 6–8） | 小 chunk 可搭载更多 decode、但 prefill 效率更低；tile 量化 | ✓（转述正确但单薄） |
| 16 | vLLM Optimization and Tuning “Chunked Prefill”（v0.20.1 源、main@199cb9b9 源、在线 latest） | decode 优先；按 `max_num_batched_tokens` 余量排 prefill；放不下自动切块；小预算利 ITL、大预算利 TTFT | ✓ 三处一致 |
| 26 | DistServe §2.3（pp. 195–196） | 分块、顺序执行、优先级三种替代方案的缺陷 | ✓ |
| 26 | DistServe §4.3 Fig. 6（p. 200） | D 侧 pull，P 的 GPU 作排队缓冲；FCFS | ✓ |
| 26 | DistServe §6.3 Fig. 10（p. 203）+ §6.1（p. 201） | 在“高速网络”下传输占比很小 | ✗（实为 25 Gbps 节点间 + NVLink 放置，见 P1-26-2） |
| 26 | Mooncake v1 §3 步骤 1–4、Fig. 4（pp. 5–6） | Conductor 选 P/D；Messenger 逐层流式送到 D 的 CPU；KV 收全后进入 continuous batch | ✓ |
| 26 | Mooncake v1 §4.2（p. 7） | D 侧“逐层加载等待” | ✗（原文是 prefill 侧，见 P0-26-1） |
| 26 | Mooncake v1 §5.1（pp. 8–9）、§6.2–6.4（pp. 11–13） | cache-aware 的 prefill 选择；P 已完成后被 D 拒绝会浪费算力；提前拒绝引起负载振荡；基于预测的提前拒绝 | ✓ |
| 35 | vLLM v0.20.1 `optimization.md`：Optimization Levels；CPU Resources / Performance Impact | -O1 为 PIECEWISE、-O2 为 FULL_AND_PIECEWISE、-O3 目前同 -O2；CPU 影响调度与输入/输出处理 | ✓ |
| 35 | vLLM v0.20.1 `torch_compile.md`：Compilation Cache、Computation Graph Processing、Cudagraph Capture、compile_sizes | 接收请求前完成编译；按 attention 切图；piecewise 图；`cudagraph_capture_sizes`；专用尺寸需要调优时间 | ✓ |
| 35 | vLLM v0.20.1 `cuda_graphs.md`：CudagraphModes、BatchDescriptor、兼容性表 | 分派键四字段；FULL_DECODE_ONLY；FULL_AND_PIECEWISE 内存与捕获时间最高；默认值依赖 piecewise compilation | ✓ |
| 35 | vLLM v0.20.1 `fusions.md` 与 `dbo.md` | AllReduce+RMSNorm 的硬件/token 数/TP 组合限制；RoPE+KV 仅 ROCm/AITER；DBO 仅 DP+EP、两个 microbatch、两个线程、MoE 内 yield | ✓ |
| 35 | PyTorch v2.8.0 `cuda.rst`：CUDA streams、CUDA Graphs（Why / API / Constraints / multiple streams / memory management） | 同流按序、跨流须显式同步与 `record_stream`；固定地址与长期存活的输入输出；CPU 工作不被捕获；图私有内存池与共享顺序约束 | ✓ |

## 跨页一致性与边界

- 15 与 16 的预算记号（$B_{\mathrm{tok}}$ 与 $B$、$K$）可对应，互相引用正确。
- 16、26、29 对“分块 / P/D 分离 / PCP / DCP / CPP”的分界一致。
- 26 与 35 对 vLLM 行为都标注为“某引擎 / 某版本”，没有冒充通用原理。
- 未发现组内数字或定义相互矛盾。
- 35 与 15 的“下一轮元数据依赖本轮采样结果”一致。
- **组外连带问题（请转交负责 22 页的组）**：P0-26-1 的同一误读也出现在 `wiki/01_theory/05_inference/22_kv_tiering_transfer_analysis.md:16`（「进入批次后，某层 attention 计算仍须等该层异步装载完成…[Mooncake §3、Fig. 3–4，§4.2]」）和 `:45`（「Mooncake 的逐层等待就是这一边界的具体设计证据。[Mooncake §4.2]」）。26 页的表述看来沿用自 22 页，修正时两页应一起改，否则仍是两份互相印证的错误来源。
