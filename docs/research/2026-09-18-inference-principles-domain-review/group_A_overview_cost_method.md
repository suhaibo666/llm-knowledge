# 推理原理域独立审阅 — A 组：全景、成本模型、方法

> **状态**：终稿，2026-09-18。
> **审阅者**：独立审阅者，未参与这些页面的写作。
> **页面**（均在 `wiki/01_theory/05_inference/`）：`01_inference_overview_analysis.md`、`10_prefill_decode_analysis.md`、`11_inference_cost_model_analysis.md`、`40_inference_benchmarking_guide.md`、`41_inference_optimization_guide.md`。
> **合同**：`docs/superpowers/plans/2026-09-17-inference-principles-implementation-plan.md` §6 T01/T10/T11/T40/T41、§7；`docs/research/2026-09-17-inference-principles-source-ledger.md`；`page-review-rubric.md`、`paper.md`、`mechanism-analysis.md`。

## 已打开的来源（均为本次实际打开）

| 来源 | 版本 / 获取方式 | 读过的位置 |
|---|---|---|
| Williams, Waterman, Patterson, *Roofline*, CACM 52(4) | 页首所列 PDF（PDF 第 1–4 页即刊物 pp.65–68），生成带页码的文本 | p.66 operational intensity 定义；p.67 min(峰值, 带宽×强度) 公式、x 轴、ridge point；p.68 X2/X4 ridge 与上界表述 |
| PagedAttention | arXiv:2309.06180v1，带页码的文本 | §2.1 Eq.(1)–(3)；§2.2、§2.3 全文；§4.3；§4.4 Shared prefix；§4.5 |
| Orca | OSDI'22 正式 PDF（usenix.org/system/files/osdi22-yu.pdf） | §3 S1；§4.2 与 Algorithm 1 |
| SGLang | arXiv:2312.07104v1 | §2.1；§5、§5.1、§5.2 |
| Transformer | arXiv:1706.03762v7 | §3.1、§3.2.1 Eq.(1)、§3.2.3 |
| Punica | arXiv:2310.18547v1 | §3、§4（SGMV 语义、Fig. 3） |
| S-LoRA | arXiv:2311.03285v1 | §5、§5.1、§5.2、§5.3 |
| NVIDIA NIM LLM Benchmarking Metrics | 1.0.0（页脚 “Last updated on Apr 01, 2026”），curl 抓取原始 HTML | TTFT、e2e_latency、ITL、TPS、RPS 全节 |
| NVIDIA AIPerf Metrics Reference | `/aiperf/reference/…` 与 `/aiperf/dev/reference/…` 两个通道，2026-09-18 抓取 | TTFT、ITL、ICL、Output Token Throughput（含 Per User）、Good Request Count/Fraction、Goodput、Request Latency、Replay Schedule Lag |
| NVIDIA AIPerf Load Generator Options | 2026-09-18 抓取 | Request Scheduling Options、Arrival Pattern、Concurrency behavior |
| NVIDIA GenAI-Perf README | Triton 用户指南页，2026-09-18 抓取 | 概述、Metrics 表、Profiling Options |
| MLCommons MLPerf Inference Rules | `inference_rules.adoc` @ `d3eba2f21026d868ad65cdcad2bb81e4a17ce3d3`（raw.githubusercontent） | §1 Definitions；§3 Scenarios 表和尾分位样本数表；Benchmarks 表中的 LLM TTFT/TPOT 约束；Load Generator / LoadGen Operation；FAQ LLM 节 |
| vLLM Context Parallel Deployment | docs.vllm.ai/en/v0.20.0 | “Prefill Context Parallel”、“Decode Context Parallel” |

注：动态文档是 2026-09-18 抓取，页面引用的是 2026-09-17 快照。两者间的差异无法完全排除，但本文涉及的每一条都已写明与页面转述一致还是不一致。

**raw 可发现性**：各页页首引用的 `raw/` 路径全部存在：`raw/01_theory/05_inference/{PagedAttention-2309.06180, Orca-2022, SGLang_RadixAttention-2312.07104, Roofline-2009, InferenceCostModelSources-20260917, InferenceBenchmarkMethodology-20260917, Punica-2310.18547, S_LoRA-2311.03285, vLLM_Context_Parallel-v0.20.0}.md`，以及 `raw/01_theory/01_models/Attention_Is_All_You_Need-1706.03762.md`。

**复算**：用 `.venv` Python 运行 `SCRATCH/groupA/recompute.py`，逐项复算以下内容：
- 01：18→14 个 prefill 位置，21→17 个理想 slot。
- 10：四输入三输出的 KV 账，终态为 6 = I+O−1。
- 11：权重 13.0385 GiB、KV 524,288 B/token、A/B/C 的六个 KV 数及三个合计和三个差额；三行 GEMM 的 F/D/I/Roofline 与 ridge 50；单 query attention 强度（MHA bf16 为 1 FLOP/B，GQA-8 为 4）；4h-FFN 参数量的合理性。
- 40：600 请求表全部 7 行。
- 41：0.3+0.2+1.9 与 55+65。

**除 11 的 “128.03 MiB” 应为 128.04 MiB 外，全部吻合。** I+O−1 另做了独立推演：prefill 写入 I 个位置；第 t 个输出（t<O）在下一轮作为输入，写入 1 个位置；第 O 个输出不再回送。故总数为 I+(O−1)，与 10 的表和 01 的 21 个 slot 一致。

**机械检查（只读）**：
- `check_links --strict`：482 页，broken/ambiguous/bare_index/stale_section/orphans 全为 0。
- 对 `wiki/01_theory/05_inference`（30 个文件）运行 `check_math --strict`、`check_markdown --strict`、`check_assets --strict`：均为 0 error、0 warning。
- `--changed` 下出现的 6 条数学警告都在其他组的审阅报告里，不在本组页面。

## 判定表

| page | beat2 | delete-code | figure-trigger | algorithm-replay | spot-check | verdict | note |
|---|---|---|---|---|---|---|---|
| 01_inference_overview_analysis | pass | pass | timing | pass | 4/4 | PASS（带 P2） | 两处引用标记的作用域，另有少量措辞与术语问题 |
| 10_prefill_decode_analysis | pass | pass | timing | pass | 3/3 | REJECT | P1，§4 末段（10:69）：没有给出 prefill/decode 资源特性的标准直觉，全页也没有链到 11 |
| 11_inference_cost_model_analysis | pass | pass | transform | pass | 5/5（1 条转述偏弱，P2） | REJECT | P1，§1（11:14–30）：TTFT 起点定义与所引工具口径及 40 页冲突 |
| 40_inference_benchmarking_guide | pass | pass | transform（决策账） | pass | 4/4 | REJECT | P1，§3–§4（40:34、40:51）：TTFT 从实际发出起算并据此判 SLO；与 11/41 冲突，且并发阀生效时漏计客户端排队（coordinated omission） |
| 41_inference_optimization_guide | pass | pass | none（决策流程，已附流程图） | n/a | 5/5 | PASS（带 P2） | 11/40 统一 TTFT 起点后需同步措辞 |

hop-walk 不适用：五页都没有声称代码行为；41 对 vLLM 的引用是版本化文档，已核对。

## 发现（按页）

### 跨页 P1（11 / 40 / 41）：TTFT 起点在三页互相矛盾，40 的 SLO 判定会发生 coordinated omission（已确认）

**三页原文：**
- **11**
  - 11:14：“一次流式请求至少有到达、实际发出、首个内容响应和末个内容响应四个时间点。令第 $i$ 个请求在客户观测点的到达时刻为 $t_{i,\mathrm{arr}}$”。
  - 11:16–21：“常用的请求级定义是：… $\operatorname{TTFT}_i=t_{i,\mathrm{first}}-t_{i,\mathrm{arr}}$”。
  - 11:184–189：把 $W_{i,\mathrm{client}}$（“可能来自压测客户端并发阀”）计入 TTFT，并警告“若一个报告只从‘已实际发出’开始计时，它可能把客户端排队移出 TTFT”。
- **40**
  - 40:34：“客户端被并发阀挡住的等待是 $t_i^{\mathrm{send}}-t_i^{\mathrm{plan}}$；客户端可见 TTFT 是 $t_i^{\mathrm{first}}-t_i^{\mathrm{send}}$”。
  - 40:51：在线决策门限“每请求客户端 TTFT 不超过 250 ms”用的正是这个从 send 起算的量。
- **41**
  - 41:14：“一条请求的客户端 TTFT 可沿计划发出、实际发出、服务端接收、排队、prefill、首 token 返回逐段记录”。
  - 41:56：单请求分解里含“客户端与网络 0.3 s”。
  - 即 41 的“客户端 TTFT”从计划发出起算，与 11 一致，与 40 相反。

**问题：**
- 同一个名字“（客户端）TTFT”，在 11 和 41 里包含客户端并发阀的等待，在 40 里不包含。
- 11 把自己从“到达”起算的公式称为“常用的”，紧接着引 NIM 作“来源事实”，而 NIM 是从提交请求起算。11 没说明：只要存在客户端排队，两者就不同。
- 40 在 40:22 自己设想了“若同时限制并发，客户端可能在计划时刻之后才发出请求”，却仍按 send 起算来判 SLO。被并发阀推迟的请求，其等待在 SLO 判定中不计入——这就是 coordinated omission，会让过载的配置通过门限。40 只要求“报告漂移”，没有要求把漂移计入 SLO，也没有要求漂移为零。

**证据（两种起点都合法，但必须分开命名）：**
- 从 send 起算（工具口径）：
  - NIM 1.0.0 的 TTFT：“the time it takes from submitting the query to receiving the first token”。
  - AIPerf 的 TTFT：“after sending a request”，公式为 `ttft_ns = request.content_responses[0].perf_ns - request.start_perf_ns`。
  - GenAI-Perf Metrics 表：“Time between when a request is sent and when its first response is received”。
- 从计划时刻起算（MLPerf 口径）：MLPerf Inference Rules @d3eba2f2，Load Generator / LoadGen Operation：“Latency is defined as the time from when the LoadGen was scheduled to pass a query to the SUT, to the time it receives a reply.”

**建议：**
- 由指标所有者 11 同时定义两个量，例如 $\operatorname{TTFT}^{\mathrm{sched}}=t^{\mathrm{first}}-t^{\mathrm{plan}}$ 与 $\operatorname{TTFT}^{\mathrm{send}}=t^{\mathrm{first}}-t^{\mathrm{send}}$，并注明 NIM/AIPerf/GenAI-Perf 报告后者、MLPerf 用前者。
- 40 和 41 引用同一套符号。
- 40 的开环 SLO 判定改用计划时刻起点，或者规定 send 口径只在漂移可以忽略时有效，并点名 coordinated omission。

### 01_inference_overview_analysis

算例、图和来源都核对无误。01 对全局关系、三本账和四个边界（准入、轮中计算、交付、释放）讲得清楚；beat-2（位置数不等于时间，命中不等于可读）到位。

- **P2 存疑（引用作用域）01:28**：原文“B 的 8-token prefill 可以按 Chunked Prefill 分成两个轮次，为 A 的 decode 留插入点；分块不会删掉 B 的历史依赖。[来源事实：SGLang v1 §5]；[PagedAttention v1 §2.2–2.3]”。
  - SGLang v1 §5 讲的是 RadixAttention（前缀 KV 复用、LRU、引用计数）；PagedAttention §2.2–2.3 讲的是 prompt/generation 两阶段与 iteration-level batching。两者都没有讲 chunked prefill。
  - 标记紧跟在分块句之后，读者可能以为分块出自这两篇。
  - 建议把标记移到“命中”句后，或给分块句加 Sarathi（`raw/01_theory/05_inference/Sarathi-2308.16369.md` 已存在）。若协调者认定这属于“归因于来源而来源未说”，可以升级处理。
- **P2 存疑（引用作用域）01:59**：原文“调度器可能让某些请求先到先服务、保护短请求延迟，或给长 prompt 分块；这些是政策选择… [来源事实：Orca §3–4.2]”。Orca §4.2 支持 iteration-level FCFS、`max_bs` 和 K/V slot 预留，但不讨论保护短请求或给 prompt 分块。建议把后两项标为本页推断，或另引来源。
- **P2 已确认（措辞）01:55**：“若 C 的 KV 在远端，索引命中也不代表传输完成后已经可读取”逻辑错位，应为“索引命中不代表 KV 已传输完成、可以读取”。
- **P2 已确认（术语）01:14/01:20**：“KV”首次出现时没有定义也没有链接（第一个到 12 的链接在 01:61）；“隔离身份”没有解释。建议在首现处加一句说明，或链接 [[12_kv_cache_analysis]]。
- **P2 已确认（页首）01:7**：页首列了 NIM LLM Metrics 1.0.0（TTFT、ITL、TPS），正文没有任何地方引用它（指标交给 11）。可以删掉，或改成“指标定义见 11”。

### 10_prefill_decode_analysis

四输入三输出账本、逻辑形状（4×4 下三角、1×5、1×6）、“G 已输出而 KV7 不存在”的边界，三者与图完全一致。Transformer 与 PagedAttention 的三处定位都确实说了页面所称的内容。

- **P1 已确认（缺少基础概念，也没有指向）10:69**：原文“本页没有给出‘prefill 总是算力受限、decode 总是带宽受限’的无条件结论。PagedAttention §2.2 报告的是典型服务情形；…性能账应在专页按工作负载计算与测量。”
  - **问题**：
    - 页面只说不给无条件结论，却没有给出有条件的标准直觉，也没说 PagedAttention 报告的“典型情形”是什么。
    - “专页”没有链接；全页正文和 Related Pages 都没有链到 11。
    - 初学者读完不知道：为什么 prefill 能高效利用 GPU，为什么 decode 常受显存带宽约束，为什么合批有用。
    - 本页标题问的正是两阶段为何有不同的执行节奏，读者必然期待这个直觉。组内重点也要求“至少给出清晰指向”。
  - **证据**：
    - PagedAttention v1 §2.2（PDF p.3），prompt phase：“can be parallelized using matrix-matrix multiplication operations. Therefore, this phase can efficiently use the parallelism inherent in GPUs”。
    - 同节，generation phase：“often uses matrix-vector multiplication, which is less efficient. As a result, this phase severely underutilizes GPU computation and becomes memory-bound”。
    - §2.3：“the overhead of moving weights is amortized across the requests in a batch”。
    - 11 §4 已有定量账：m=512 时强度 442.8 FLOP/B，高于 ridge 50；m=1 时为 1.0；m=32 时为 31.7。
  - **建议**：
    - 在 §3 或 §4 加 2–3 句有条件的直觉：
      - 已知 prompt 走矩阵乘矩阵，权重读取摊到多个 token 上。
      - batch=1 的 decode 走矩阵乘向量，每步都要读全部权重和历史 KV，计算却很少。
      - 合批能摊销权重读取，但摊销不了各序列自己的 KV。
      - 实际瓶颈随长度、batch、量化和设备而变。
    - 链接 [[11_inference_cost_model_analysis|成本模型]] §4，并把 11 加入 Related Pages。
- **P2 已确认（导航）10:71–77**：Related Pages 五项中有三项是工程页，没有 11、15、17。页首 10:9 说“KV 的复用和容量、采样规则、服务调度分别由后续专题负责”，却只链接了 12。计划 T99 的关键链写明“10→15→16”，本页没有到 15 的链接。建议补上 [[15_continuous_batching_analysis]]、[[17_sampling_decoding_analysis]]、[[11_inference_cost_model_analysis]]。
- **P2 已确认（论证方向）10:20**：“前面的位置在同一层内不必等后面的位置算完”方向不对。因果掩码本来就保证前面不依赖后面。prefill 能并行的关键是：每个位置在第 $l$ 层只依赖第 $l-1$ 层、位置 $\le i$ 的输出，而对已知 prompt 这些输出同时可得，所以同层内**没有任何位置需要等其他位置**。建议改写。
- **P2 已确认（来源转述）10:65**：PagedAttention §2.2 的终止条件是 “the sequence reaches a maximum length (specified by users or limited by LLMs)” 或 `<eos>`。页面写成“输出长度上限”，漏掉了模型最大序列长度这一项。

### 11_inference_cost_model_analysis

公式和数字几乎全部正确：
- $2mkn$、流量下界、$I=F/D$、$\min(\Pi_{\mathrm{peak}},B_{\mathrm{mem}}I)$、ridge 50、$2LH_{\mathrm{kv}}db$、I+O 与 I+O−1、各合计与差额，都复算通过。
- “虚构设备”自洽：100/2=50，且 31.7 < 50 < 442.8。
- 瓶颈表述都带有条件。
- Roofline p.67–68 的转述准确：min 公式、以 DRAM 字节为横轴、kernel 上界。

- **P1 已确认**：见上文跨页 P1（11:14–30、11:184–189）。
- **P2 已确认（算术）11:167**：原文“128.03 MiB”。$D=2(4096+16384)+2\cdot4096\cdot16384=134{,}258{,}688$ B $=128.0391$ MiB，四舍五入应为 **128.04 MiB**。图中没有出现这个数。
- **P2 已确认（来源转述偏弱）11:89、11:172**：原文“它提供阶段语义，而非‘prefill 或 decode 必然受限于某资源’的定律”与“原文也只在其服务讨论的条件下描述 memory-bound 的 decoder phase”。PagedAttention §2.2 直接写了 generation phase “severely underutilizes GPU computation and becomes memory-bound”，§2.3 写了合批摊销权重搬运；原文唯一的限定是 “often uses matrix-vector multiplication”。建议先如实转述原文结论，再补本页的适用条件。本页 §4 的条件化论证本身是对的。
- **P2 已确认（术语与归属）11:42–55**：11 是指标所有者，却只定义了按 token 计的 $\Theta_{\mathrm{good}}$（“本文采用的有效输出吞吐”）。通行的 goodput 按请求计：AIPerf 的 Goodput = `good_request_count / benchmark_duration_seconds`。请求级的 $Q_{\mathrm{good}}$、$f_{\mathrm{good}}$ 只在 40:36–45 定义。两页互相注明“不是同一个量”，并不矛盾，但指标定义分散在两页。建议 11 同时给出请求级 goodput（注明 AIPerf 口径），40 直接引用。
- **P2 已确认（缺少排队直觉）11:174–191**：计划 T11 §5 要求讲“排队与执行及延迟吞吐取舍”。页面给了 TTFT 分解和 batch 取舍，但缺两点：
  - 并发、吞吐与时延的基本关系：Little 定律 $L=\lambda W$；闭环下 $X=C/\overline{E2E}$。
  - 到达率逼近容量时，排队时延会急剧上升。
  - 这恰好也能解释 40:22 所说“闭环会随服务变慢自动降低到达率”的机制。
  - 页面没有使用 Little 定律，因此不存在误用。
- **P2 建议（可选）11:172**：单 query attention 的强度常数可以直接写出：$\approx 4SdH_q/(2SdH_{\mathrm{kv}}b)=2H_q/(H_{\mathrm{kv}}b)$，即 bf16 MHA 为 1 FLOP/B，GQA 组大小为 8 时为 4。并点明：合批能摊销权重读取，但摊销不了各序列自己的 KV 读取。
- **P2 已确认（符号）11:64–65、11:128–130**：“C”既是并发数，又是情形名（“情形 C … C32”，“$C=32$”）。建议把并发改记为 $N_c$ 之类。
- **P2 已确认（来源通道）11:7 与 40:7/41:7**：11 引的是 AIPerf `/aiperf/dev/reference/…`，40/41 引的是 `/aiperf/reference/…`。所查条目内容一致，但建议统一通道，并在 raw 索引中注明。

### 40_inference_benchmarking_guide

以下内容核对正确：
- 600 请求算例全部复算正确：11,800/60=196.67；560/600=93.33%；500/600=83.33%；9.33 与 8.33 req/s；90% 门限选 A。
- 尝试分母、失败原因、请求级分位数、warmup 与冷热、配对与重复运行的要求都对。
- AIPerf 的 Good Request Fraction（`attempted = request_count + error_request_count`）与 Goodput 转述准确。
- MLPerf §3 场景转述准确：Server/Interactive 为 “according to a Poisson distribution”，Offline 为 “sends all samples … in a single query”。
- 页面没有引用 MLPerf 的具体 TTFT/TPOT 数值，因此不需要核对那一栏。

- **P1 已确认**：见上文跨页 P1（40:34、40:51）。
- **P2 已确认（到达过程）40:22**：原文“固定请求率是按事先给出的时间表发出请求”，把按速率生成与 fixed schedule 混在了一起。
  - AIPerf Load Generator 中，`--request-rate` 按 arrival pattern 生成请求（默认 `poisson`，另有 `constant`、`gamma`）；`--fixed-schedule` 才是 “Replay requests at exact timestamps”。
  - MLPerf Server 同样用 Poisson。
  - 页面全篇没有提到到达分布。同一平均速率下，Poisson 与恒定间隔的排队尾部不同，所以分布本身应列入负载合同。
- **P2 已确认（窗口边界）40:36**：“失败数 $N_{\mathrm{fail}}=N-N_{\mathrm{ok}}$”隐含所有尝试请求都已到终态；在稳态窗口里，跨越 $T_1$ 的在途请求会被误记为失败。§4 的例子用“窗口包含全部终态”回避了这点，但 §3 的通用定义应写明处理方式。MLPerf LoadGen 的做法是：达到最短时长后停止发请求，再 “waits for all queries to complete”。
- **P2 已确认（分位数统计）40:47**：只说“从请求级样本求 P50/P95/P99”，没有样本量要求。
  - MLPerf 规则 §3 给出了尾分位所需的推理数（99% 置信度下，90%ile 为 24,576，99%ile 为 270,336），并采用 early stopping。
  - 41 的三个场景都以 100 条请求的 P95 作门限，而这个 P95 只由最慢的约 5 条决定。
  - 建议补一句样本量或置信区间的要求，或要求报告多次运行的区间（§5 第 2 步已部分覆盖）。
- **P2 已确认（ITL 口径）40:47、40:51**：
  - 推荐的“每请求平均 ITL”会把一次请求内的长停顿平均掉。而 40:113 在 Related Pages 里把 16 页当作“可检验的 TTFT/ITL 取舍”，chunked prefill 要消除的恰恰是这种 decode 停顿。
  - AIPerf 另有 ICL（逐块间隔的完整分布）；GenAI-Perf 的 ITL 是逐响应取值。
  - 建议评测调度改动时，同时报告逐 token 或逐块间隔的尾部或最大停顿。
  - 另外，“每请求平均 ITL”就是 11 所称的 TPOT，建议统一称呼。
- **P2 已确认（决策规则措辞）40:14**：原文“在给定…请求到达率下，找到满足 TTFT 与流内间隔 SLO 的最大成功请求率”有两种读法：一是固定到达率、比较各配置（§4 的做法）；二是扫描到达率、求可维持的最大速率（MLPerf Server 的指标 “Maximum Poisson throughput parameter supported”）。建议把两种决策形式分开写。

### 41_inference_optimization_guide

三个场景的数字都标为教学设定，算术正确。以下几处论证完整：“单请求分解不能把各段 P95 相加”“一次只改一个变量”“不把论文加速倍数相乘”。与 11/40 及机制页（14、22、25、27、28、29、30）的边界一致。vLLM v0.20.0 PCP 原文 “Both approaches are under active development.” 与 41:30 相符。

- **P2 已确认（措辞）41:82**：“KV 压缩降低容量后可能引入解压/质量成本”会读成“降低容量”，应为“降低 KV 占用后”。
- **P2 已确认（house）41:7**：页首字段名是“来源基线”，其余四页用的是“文献基线”。
- **P2 建议 41:64–68**：场景 B 中 100 条请求有 80 条共享前缀，TTFT P95 由最慢的约 5 条决定，这几条可能恰好来自 20 条未命中的请求。建议在假设阶段先核对 P95 尾部由哪类请求构成，否则 ≤750 ms 的门限可能与缓存无关。这与本页 §1“不能用平均替代长尾”的原则一致。
- 随跨页 P1 同步：41:14 的“客户端 TTFT”须与 11/40 统一后的定义一致。

### 索引（P2）

- `index.md:18–19`：10、11 的入口别名是截短的标题（“自回归生成与 Prefill / Decode”“推理性能与资源成本模型”），页面标题还带副标题。别名是标题前缀，语义不冲突。
- `index.md:9,17–19,44`：状态写的是“交叉审阅通过”，与本次独立审阅对 10、11、40 的 REJECT 不一致，修订后需更新。

## 锚点抽查记录

| 页面 | 来源与定位 | 结果 |
|---|---|---|
| 01 | PagedAttention v1 §2.1–2.2（prompt/generation；终态 token 不再回送） | 一致 |
| 01 | PagedAttention v1 §2.3 “After each iteration, completed requests are removed from the batch, and new ones are added”；§4.3 每轮选序列、分配块、完成后释放 | 一致 |
| 01 | Orca §3 S1（选请求→执行一轮→收结果，每轮可重选） | 一致；§3–4.2 不含 01:59 的“保护短请求”和“分块”（P2） |
| 01 | SGLang v1 §5/§5.1（radix 树前缀复用、LRU、引用计数） | 一致；不含 01:28 的分块句（P2） |
| 10 | Transformer v7 §3.1 “prevent positions from attending to subsequent positions” | 一致 |
| 10 | Transformer v7 §3.2.3（decoder 自注意力掩码；与 encoder-decoder attention 的区分） | 一致 |
| 10 | PagedAttention v1 §2.2（prompt phase；逐轮输入 $x_{n+t}$；只新算 $k_{n+t},v_{n+t}$；两种终止条件） | 一致；终止条件转述偏窄（P2） |
| 11 | Roofline CACM p.67 的公式、x 轴、ridge；p.68 的上界 | 一致 |
| 11 | NIM 1.0.0：TTFT、E2E、ITL（GenAI-Perf 式 $(e2e-TTFT)/(O-1)$，LLMPerf 含 TTFT）、TPS 两种分母、每用户 TPS | 一致；TTFT 起点见跨页 P1 |
| 11 | AIPerf：ITL 排除 TTFT，为 $(latency-TTFT)/(OSL-1)$；Output Token Throughput Per User = 1/ITL | 一致 |
| 11 | GenAI-Perf：列出 TTFT、ITL、请求与输出 token 吞吐；负载由用户指定（并发、请求率） | 一致 |
| 11 | PagedAttention v1 §2.2 | 内容存在，但页面转述偏弱（P2） |
| 40 | MLPerf @d3eba2f2 §3 Scenarios（Server/Interactive 用 Poisson；Offline 一次提交） | 一致 |
| 40 | AIPerf Load Generator：request-rate / concurrency-only / fixed-schedule 三种模式；并发上限会阻塞按计划到期的请求 | 一致（“事先给出的时间表”措辞见 P2） |
| 40 | AIPerf Good Request Fraction（错误请求计入 attempted）与 Goodput（good 数 / 时长） | 一致 |
| 40 | NIM 1.0.0 TTFT（从提交起算） | 与 40 的 send 口径一致，但与 11/41 冲突（P1） |
| 41 | vLLM v0.20.0 “Both approaches are under active development.” | 一致 |
| 41 | Punica v1 §4 SGMV `Y[s[i]:s[i+1]] += X[s[i]:s[i+1]] @ W[i]` | 一致 |
| 41 | S-LoRA v1 §5.1 Unified Paging、§5.2 Prefetching and Overlapping | 一致 |
| 41 | PagedAttention v1 §4.4 Shared prefix（前缀块映射，最后一块 CoW） | 一致 |
| 41 | SGLang v1 §5 RadixAttention | 一致 |
