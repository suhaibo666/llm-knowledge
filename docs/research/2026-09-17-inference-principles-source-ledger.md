# 推理基础原理：来源与归属证据账

日期：2026-09-17。对应执行计划 `docs/superpowers/plans/2026-09-17-inference-principles-implementation-plan.md`。本文件记录已核验来源、版本、算例和审阅证据，不是新的读者目录。动态官方文档采用 2026-09-17 访问快照，论文采用下表所列固定版本。

## 工作区与共享文件边界

- 起始分支为 `main`，工作区已有另一任务的未提交修改，集中在 slime 后训练域、图形生成器、课程与 `docs/radar/watchlist.yaml`。`wiki/index.md` 也已有未提交修改；本任务只改理论域“推理技术”一行及该行页数的复算注记，不回写该文件其余段落。`wiki/01_theory/index.md` 的起始 05 行为“推理技术：CoT、RAG、Agent、验证”；05 原 index 是 Agent 待建设页，指向已迁移的旧 `raw/08_agents/` 路径。
- 本轮没有修改原有工程页或其他任务内容。为消除 T18 与 06 对通用在线 softmax 推导的重复，只定点修改 `wiki/01_theory/06_distributed_parallelism/20_ring_attention_and_context_parallel_analysis.md` §5.2：06 保留 Ring/CP 特有通信与局部合并，通用数值推导交给 T18 并补链接。没有清理、暂存、提交、推送、合并或发布。

## 页面与来源进度

下表汇总 29 篇正文所据的原始论文、规范或固定版本官方文档；教学算例均非运行实测。详细定位和链接同时写在各页页首与 raw 来源索引。

| ID | 核心问题 | 来源类型和实际入口 | 版本与定位 | 可支撑内容、推断限制 | 状态 |
|---|---|---|---|---|---|
| 01 | 请求全程如何分层？ | PagedAttention、Orca、SGLang 原文；NIM 指标文档；具体原文链接与 raw 索引见对应正文页首 | 2309.06180v1 §2/§4；OSDI22 §3/§4.2；2312.07104v1 §5；NIM 1.0.0 | A/B/C 的 18→14 个 prefill 位置与 21→17 理想 KV slot 为教学推演；命中要求身份一致且 KV 可读。 | 原文已打开；非作者审阅 PASS |
| 10 | 已知 prompt 与未知输出如何执行？ | 论文：`raw/01_theory/01_models/Attention_Is_All_You_Need-1706.03762.md` → [官方 PDF](https://arxiv.org/pdf/1706.03762v7)；`raw/01_theory/05_inference/PagedAttention-2309.06180.md` → [官方 PDF](https://arxiv.org/pdf/2309.06180v1) | 1706.03762v7 §3.1–3.2.3、Eq. (1)；2309.06180v1 §2.1–2.3、Eq. (1)–(3) | 因果掩码、逐位置条件概率、prompt phase、逐轮生成及 KV 形成顺序。四输入三输出与逻辑张量形状是教学推演，非运行实测。 | 原文已打开；非作者审阅 PASS |
| 11 | 推理成本如何计量？ | 论文：`raw/01_theory/05_inference/Roofline-2009.md`；官方指标：`InferenceCostModelSources-20260917.md`；PagedAttention v1 §2.2 | Roofline p.67–68；NIM 1.0.0 TTFT/E2E/ITL/TPS；AIPerf/GenAI-Perf 2026-09-17 快照 | 固定 7B 模型、虚构设备、A/B/C 容量和 GEMM 账为教学推导；KV 已写位置用 $I+O-1$，预留位另列。Roofline 是性能上界，不是测量。 | 原文已打开；非作者审阅 PASS |
| 12 | KV 何时可复用？ | 论文：`raw/01_theory/05_inference/PagedAttention-2309.06180.md`、`MultiQueryAttention-1911.02150.md`、`GroupedQueryAttention-2305.13245.md` 与 `raw/01_theory/01_models/deepseek/DeepSeek_V2-2405.04434.md` | 2309.06180v1 §2.1–2.2；1911.02150v1 §2.4、§3；2305.13245v1 §2.2、Fig. 2；2405.04434v1 §2.1.1–2.1.4、Table 1 | 历史 KV 复用、逐轮追加和 MHA/MQA/GQA/MLA 表示差异。15 对 6 的位置计算、46 对 21 的位置对数及字节表为教学推演，非实测。 | 原文已打开；非作者审阅 PASS |
| 13 | 分页 KV 如何映射？ | 论文：`raw/01_theory/05_inference/PagedAttention-2309.06180.md` → [2309.06180v1](https://arxiv.org/html/2309.06180v1) | §3/Fig. 3；§4.1 Eq. (4)/Fig. 5；§4.2–4.4/Fig. 8；§5.2；§7.1–7.2/Fig. 18；§8 | B=4 的块表、offset、分叉 CoW 与 ref=0 末态为教学推演；旧字节是否仍在只作存储层推断。内容身份与命中归 14，不把论文基线等同当前 vLLM。 | 已打开原文；正文、SVG 与图源已撰写；非作者审阅通过 |
| 14 | 前缀如何跨请求共享？ | `raw/01_theory/05_inference/SGLang_RadixAttention-2312.07104.md`；PagedAttention v1；固定 SGLang `0b3bb0cbe31873994c9f989fddfe2f87ca839fdd` | SGLang v1 §2.1、§5 Algorithm 1/Fig. 6、§6.4；PagedAttention v1 §4.4；源码 `RadixKey`、`match_prefix`、`inc_lock_ref`、`dec_lock_ref`、`evict` | 前缀身份、Radix 匹配、物理引用与淘汰分开；A/B/C 命中长度是教学推演，不冒充引擎缓存命中率。 | 原文与固定源码已打开；非作者审阅 PASS |
| 15 | 如何逐轮组批？ | `raw/01_theory/05_inference/Orca-2022.md`；PagedAttention v1 | Orca OSDI 2022 §3、§4.2 Algorithm 1/FCFS；PagedAttention v1 §4.5 | 请求到达、逐轮准入、token/KV 预算与结束释放；四轮账是教学推演，不把 FCFS 等同 continuous batching 定义。 | 原文已打开；非作者审阅 PASS |
| 16 | 长 prompt 如何分块？ | `raw/01_theory/05_inference/Sarathi-2308.16369.md`；`vLLMChunkedPrefillDocs-20260917.md` | Sarathi v1 §3.1–3.3、§4.2 Fig. 6、§4.3–4.4；vLLM Optimization and Tuning 2026-09-17 快照 | 连续分块的因果历史与跨块 KV、decode 混批和策略边界；B=8、K=14 的双路线 10-token 时间线为教学推演。 | 原文已打开；非作者审阅 PASS |
| 17 | logits 如何变成 token？ | `raw/01_theory/05_inference/The_Curious_Case_of_Neural_Text_Degeneration-1904.09751.md`；`Hugging_Face_Transformers_Generation-v4.57.1.md` | Holtzman v2 §3.1；Transformers v4.57.1 固定 commit `8cb5963cc22174954e7dca2c0a3320b7dc2f4edc` 的 `TemperatureLogitsWarper`、`RepetitionPenaltyLogitsProcessor`、`BeamSearchScorer.process`、`BeamHypotheses` | softmax、温度、top-k/p、重复惩罚、beam/EOS；EOS 只有排名前 $B$ 才进入完成集合为该实现合同，不当通用定律。 | 原文与固定源码已打开；非作者审阅 PASS |
| 18 | 分块 attention 如何降 IO？ | 论文：`raw/01_theory/05_inference/FlashAttention-2205.14135.md`、`FlashAttention2-2307.08691.md`；作者文章：`FlashDecoding-2023.md`；官方实现 `Dao-AILab/flash-attention@6d673cd9610172431bf1b7786d10918dba2783c0` | 2205.14135v2 §2.2、§3.1–3.2、§5；2307.08691v1 §2.3、§3.1–3.2；作者文 2023-10-12；代码 `softmax_rescale_o`、`flash_attn_with_kvcache`、`combine_attn_seqk_parallel` | 分块统计、IO 条件、短 query 的 split-KV。两块 $11/3$ 是教学复算；实数等价不保证浮点逐位相等。在线 softmax 推导归 18，06 保留 Ring/CP 通信。 | 原文与固定源码已打开；非作者审阅 PASS |
| 19 | 量化怎样改变误差与成本？ | `raw/01_theory/05_inference/Quantization_and_Training_of_Neural_Networks-1712.05877.md`、`LLM_int8-2208.07339.md`、`TensorRT_Quantization_Schemes-11.3.0.md` | Jacob v1 §2.1–2.4、§3.1；LLM.int8 v2 §3.1–3.2、§4.1–4.2/Appendix D；TensorRT Quantization Schemes/Accuracy 11.3.0，Working with Quantized Types 2026-09-17 快照 | affine 编码、粒度、舍入/截断、离群值与执行成本；五元素三路径和非对称示例是教学数据，格式支持与数值可表示分开。 | 原文与官方文档已打开；非作者审阅 PASS |
| 20 | 校准怎样控制误差？ | GPTQ、AWQ、SmoothQuant 原论文；具体原文链接与 raw 索引见对应正文页首 | 2210.17323v1 §3–4；2306.00978v1 §2–3；ICML23 §3–5 | 三种算法的校准目标与误差路径分开；小矩阵复算不代表模型质量或 GPU 加速。 | 原文已打开；非作者审阅 PASS |
| 21 | KV 怎样压缩或选择？ | KIVI、H2O、StreamingLLM、DeepSeek-V2 原论文；具体原文链接与 raw 索引见对应正文页首 | 2402.02750v1 §3–4；2306.14048v1 §3–5；2309.17453v2 §3–4；2405.04434v1 §2.1 | 量化、选择保留与架构性 latent 压缩分开；精确复用与近似质量不混写。 | 原文已打开；非作者审阅 PASS |
| 22 | KV 怎样跨层迁移？ | Mooncake、FlexGen 原论文；具体原文链接与 raw 索引见对应正文页首 | 2407.00079v1 §3/§4.2/§5；ICML23 §1/§4.1–4.2 | 目录命中、数据传完、设备可读取是不同边界；传输时间按教学条件复算。 | 原文已打开；非作者审阅 PASS |
| 23 | 投机怎样保持目标分布？ | Leviathan 等投机解码原论文；具体原文链接与 raw 索引见对应正文页首 | 2211.17192v2 §2.1–2.3/Algorithm 1/Appendix A.1 | 小词表接受/拒绝残差分布逐项复算；分布等价不等于逐次 token 相同。 | 原文已打开；非作者审阅 PASS |
| 24 | 草稿和树验证怎样选择？ | Draft & Verify、Medusa、EAGLE、SpecInfer 与原始投机论文；具体原文链接与 raw 索引见对应正文页首 | 2309.08168v2 §3；2401.10774v1 §3；2401.15077v1 §2–4；2305.09781v4 §3–4 | 草稿来源、树结构、验证规则配对；不同变体不一概宣称无损。 | 原文已打开；非作者审阅 PASS |
| 25 | 多 LoRA 怎样共享基座？ | LoRA、Punica、S-LoRA 原论文；具体原文链接与 raw 索引见对应正文页首 | 2106.09685v1 §3；2310.18547v1 §2–4；2311.03285v1 §4–5 | 基座批共享与逐行 adapter 增量分开；搬运/分页额外成本计入。 | 原文已打开；非作者审阅 PASS |
| 26 | P/D 如何分离和交接？ | DistServe、Mooncake 原论文；具体原文链接与 raw 索引见对应正文页首 | OSDI24 §2.3/§3–6；2407.00079v1 §2–5 | P/D 交接的排队、传输和可读完成条件分开；不把分离等同 PCP/DCP。 | 原文已打开；非作者审阅 PASS |
| 27 | PCP 如何分担 prefill？ | vLLM、vLLM-Ascend 官方文档；具体原文链接与 raw 索引见对应正文页首 | v0.20.0 Prefill CP；Ascend v0.21.0rc Device Distribution/Prefill CP | PCP2 教学依赖逐 query 核对；实现支持不当作原理定义。 | 原文已打开；非作者审阅 PASS |
| 28 | DCP 如何分担 decode KV？ | vLLM、vLLM-Ascend 官方文档及 vLLM 技术文章；具体原文链接与 raw 索引见对应正文页首 | v0.20.0 Decode CP；Ascend v0.21.0rc Block Table/Decode CP；2026-08-07 文章 §3–5 | 局部 softmax 统计合并与未切分结果复算；复用 TP ranks 是实现选择。 | 原文已打开；非作者审阅 PASS |
| 29 | CPP 如何做层间流水？ | NVIDIA 软硬协同文章、VPP 论文、Ascend CPP 文档；具体原文链接与 raw 索引见对应正文页首 | 2026-07-10 Fig. 8–9；2608.26523v1 §1/§2.4；Ascend v0.21.0rc Runtime/Constraints | chunk 的层间和同层历史依赖及气泡模型按均衡教学假设核对。 | 原文已打开；非作者审阅 PASS |
| 30 | 并行轴如何组合？ | vLLM/Ascend CP、NVIDIA 软硬协同来源；具体原文链接与 raw 索引见对应正文页首 | vLLM v0.20.0；Ascend v0.21.0rc；NVIDIA 2026-07-10 | 并行轴的数学可组合、实现支持、性能合适分开；容量算例复算。 | 原文已打开；非作者审阅 PASS |
| 31 | MoE 路由如何影响推理？ | DeepSeek-V3 Technical Report 原论文；具体原文链接与 raw 索引见对应正文页首 | 2412.19437v1 §3.4.1–3.4.2 | 逻辑专家和物理副本分开；8 token 的 4/4→6/2→4/4 是教学负载。 | 原文已打开；非作者审阅 PASS |
| 32 | 如何约束生成格式？ | Efficient Guided Generation、XGrammar 原论文；具体原文链接与 raw 索引见对应正文页首 | 2307.09702v1 §1–4；2411.15100v1 §2–3 | token 边界的 grammar mask 与归一化概率复算；格式合法不保证语义。 | 原文已打开；非作者审阅 PASS |
| 33 | 多模态推理如何组织？ | LLaVA、Flamingo 原论文；具体原文链接与 raw 索引见对应正文页首 | 2304.08485v2 §4.1/Fig.1；2204.14198v1 §3.1/Fig.4–6 | 视觉注入与 cross-attention 分支分开；教学占位数不等于 encoder 工作量。 | 原文已打开；非作者审阅 PASS |
| 34 | 混合模型状态如何协作？ | Jamba、Mamba 原论文；具体原文链接与 raw 索引见对应正文页首 | 2403.19887v1 §1–3；2312.00752v1 §2–3/Eqs.2a–2b | 逐 token KV 与压缩循环状态生命周期分开；回滚需快照/重放。 | 原文已打开；非作者审阅 PASS |
| 35 | 推理执行优化怎样组合？ | vLLM V1、PyTorch CUDA 官方文档；具体原文链接与 raw 索引见对应正文页首 | vLLM v0.20.1 Optimization；PyTorch 2.8 CUDA streams/graphs | 融合、编译、图捕获和重叠的节省项及额外条件逐项区分。 | 原文已打开；非作者审阅 PASS |
| 40 | 性能结论如何可信？ | MLCommons MLPerf、NVIDIA AIPerf/NIM 官方文档；具体原文链接与 raw 索引见对应正文页首 | MLCommons 固定提交 d3eba2f21026d868ad65cdcad2bb81e4a17ce3d3 §3；NIM 1.0.0；AIPerf 2026-09-17 快照 | 600 请求的尝试分母、good 比率与 token/s 为教学数字；未运行压测。 | 原文已打开；非作者审阅 PASS |
| 41 | 如何从瓶颈选方案？ | PagedAttention、SGLang、Punica、S-LoRA 原论文及 vLLM/AIPerf 文档；具体原文链接与 raw 索引见对应正文页首 | 2309.06180v1 §2–4；2312.07104v1 §5；2310.18547v1 §3–4；2311.03285v1 §5.1–5.2；vLLM v0.20.0 | 三个场景是观测→单变量假设→配对实验→预设门限的教学方案，没有 B 配置实测。 | 原文已打开；非作者审阅 PASS |


## 旧内容守恒与交接

| 旧解释 | 独有内容与当前去向 | 本轮决定 |
|---|---|---|
| `wiki/01_theory/05_inference/index.md` 原 Agent 待建设页 | Agent/ReAct、CoT/RAG 后续线索；旧 `raw/08_agents/` 路径已非当前来源位置 | 在新 index 的范围边界保留线索，并指向现存 `raw/01_theory/05_inference/`；没有创建 Agent 正文 |
| `wiki/01_theory/06_distributed_parallelism/20_ring_attention_and_context_parallel_analysis.md` | Ring/CP 的跨卡通信、局部 softmax 统计合并和原有算例 | 保留跨卡通信与既有例子；T18 负责单次 attention 的在线 softmax 数值推导并回链 06，不迁移或删除旧内容 |
| `wiki/01_theory/06_distributed_parallelism/21_hw_friendly_llm_codesign_analysis.md` §六 | CPP 的论文特有 Guideline 6、Fig. 9 数据及 Helix 对照 | 保留；T29 写通用原理后再考虑只补交接链接，不迁移实测 |
| `wiki/02_engineering/03_infer_frameworks/vllm/08_vllm_kv_cache_management_analysis.md` | 固定 vLLM 基线的块池、hash、引用计数、命中/释放 | 保留；12–14 讲跨引擎原理，不复制该页字段和配置矩阵 |
| `wiki/02_engineering/03_infer_frameworks/vllm/17_vllm_quantization_analysis.md` | 量化格式、loader、kernel 与配置边界 | 保留；19–20 负责表示、校准和误差原理 |
| `wiki/02_engineering/03_infer_frameworks/vllm/16_vllm_speculative_decoding_analysis.md` | 固定实现中的 proposer、verifier、GPU/CPU 状态结算 | 保留；23–24 负责通用接受/拒绝与草稿设计 |

没有目标已验收的新页时不删除上述旧内容；所有后续去重须再列入站链接与来源支持的订正。

## 本次验证与待审边界

### 2026-09-17 并发撰写批次的审阅进度

| 页面 | 非作者审阅 | 来源抽查、算例与图 | 结论 |
|---|---|---|---|
| T10 Prefill / Decode | T13 作者 | Transformer v7 与 PagedAttention v1 的 3 个定位；四输入三输出及 logits/KV 顺序复核，Mermaid 实际渲染 | PASS：新采样 token 尚未形成 KV 的边界明确 |
| T11 成本模型 | 协调者 | NIM 1.0.0、AIPerf、Roofline p.67–68 抽查；A/B/C、$I+O-1$ 与预留容量复算；独立 raw 与 Mermaid 修订后目检 | PASS：教学推导与实测分开，瓶颈有条件限定 |
| T12 KV Cache | T13 作者 | PagedAttention、MQA、GQA、DeepSeek-V2 原文抽查；15/6、46/21 与字节账复核，Mermaid 实际渲染 | PASS：容量公式标明架构和布局假设 |
| T13 分页 KV | 协调者 | PagedAttention v1 §4.1 Eq. (4)、§4.3、§4.4 Fig. 8 抽查；B=4 算例、四阶段 SVG 复算目检 | PASS：数据仍在、可复用和被引用分开 |
| T14 Prefix Caching | 协调者 | SGLang v1 §5 Algorithm 1/Fig. 6、§6.4，PagedAttention v1 §4.4，SGLang 固定源码抽查；A/B/C 命中和 SVG 目检 | PASS：身份、匹配、引用与淘汰分开 |
| T15 Continuous Batching | 协调者 | Orca OSDI §3/§4.2 Algorithm 1、PagedAttention v1 §4.5 抽查；四轮 token/KV 账复核，Mermaid 实际渲染 | PASS：政策与机制分开，状态释放可对账 |
| T16 Chunked Prefill | 协调者 | Sarathi v1 §4.2 Fig. 6/§4.3–4.4 和 vLLM 官方文档抽查；两路线同为 10 输入 token，KV 10/12/14 峰值复核，Mermaid 实际渲染 | PASS：历史依赖保留，TTFT/ITL 取舍有条件限定 |
| T17 采样与解码 | 协调者 | Holtzman v2 §3.1、Transformers v4.57.1 固定源码抽查；softmax 与 beam 逐项计算；EOS 排名门订正，SVG 目检 | PASS：实现 EOS 合同不冒充通用策略 |
| T18 高效 Attention | T11 作者 | FlashAttention v2 §3.1–3.2、FA2 v1 §3.2、Flash-Decoding 作者文与固定代码符号抽查；$11/3$ 独立计算，Mermaid 渲染；06 §5.2 去重后复审 | PASS：通用在线 softmax 推导归 T18，06 保留 Ring/CP 交接 |
| T19 推理量化 | 协调者 | Jacob v1、LLM.int8 v2、TensorRT 11.3.0 原文抽查；三路径及非对称最近偶数舍入复算，SVG 目检 | PASS：数值表示与引擎可执行性分开 |
| T01 全景 | spec_lora 作者 | PagedAttention、Orca、SGLang 3/3；18→14、21→17 复算，因果时间图渲染 | PASS：三请求共享身份与 KV 可读性限定明确 |
| T20 量化方法 | 协调者 | GPTQ/AWQ/SmoothQuant 原文抽查；Hessian、SSE 与量化账复算，图渲染 | PASS：校准算法和部署收益分开 |
| T21 KV 压缩 | 协调者 | KIVI/H2O/StreamingLLM/DeepSeek-V2 原文抽查；示例数值复算，图渲染 | PASS：表示压缩、淘汰和架构压缩分开 |
| T22 KV 分层 | 协调者 | Mooncake/FlexGen 原文抽查；传输时间复算，图渲染 | PASS：命中、搬完、可消费分开 |
| T23 投机解码 | 协调者 | Leviathan v2 原文与小词表分布复算，图渲染 | PASS：接受/拒绝概率保持目标分布 |
| T24 投机变体 | 协调者 | Draft & Verify/Medusa/EAGLE/SpecInfer 原文抽查，图渲染 | PASS：变体各自验证合同明确 |
| T25 多 LoRA | 协调者 | LoRA/Punica/S-LoRA 原文抽查，图渲染 | PASS：adapter 身份与输出行逐一对应 |
| T26 P/D 分离 | 协调者 | DistServe/Mooncake 原文抽查；交接时间复算，图渲染 | PASS：传输与排队开销计入 |
| T27 PCP | 协调者 | vLLM/Ascend 版本文档核对；3/7→5/5 算例、SVG 目检 | PASS：每个 query 的依赖完整 |
| T28 DCP | 协调者 | vLLM/Ascend 文档及技术文章核对；14/5 与 8/3 复算、SVG 目检 | PASS：局部归一化正确合并 |
| T29 CPP | 协调者 | NVIDIA/VPP/Ascend 来源核对；12 busy/18 slots 气泡账复算、SVG 目检 | PASS：时间线满足阶段依赖 |
| T30 并行组合 | 协调者 | vLLM/Ascend 来源核对；512 MiB 总量/64 MiB 每卡复算、SVG 目检 | PASS：数学、支持与收益条件分开 |
| T31 MoE 推理 | 协调者 | DeepSeek-V3 v1 §3.4 抽查；4/4→6/2→4/4 算例，Mermaid 渲染 | PASS：专家副本和逻辑路由分开 |
| T32 约束生成 | 协调者 | Outlines/XGrammar 原文抽查；0.665241/0.244728/0.090031 复算，Mermaid 渲染 | PASS：跨 token 语法与格式/语义边界明确 |
| T33 多模态 | 协调者 | LLaVA/Flamingo 原文抽查；9 self-KV 输入位置与 latent 支路核对，Mermaid 渲染 | PASS：两种连接范式分开 |
| T34 混合状态 | 协调者 | Jamba/Mamba 原文抽查；公式定位更正，Mermaid 渲染 | PASS：循环状态的回滚边界明确 |
| T35 执行优化 | 协调者 | PyTorch/vLLM 固定文档核对；12→10 时间线、SVG 目检 | PASS：依赖/生命周期与重叠分开 |
| T40 评测方法 | spec_lora 作者 | MLCommons 固定提交、AIPerf/NIM 3/3；600 请求算例与图渲染 | PASS：尝试分母及门限可复核，数字为教学例 |
| T41 优化决策 | parallel_axes 作者 | SGLang/Punica/S-LoRA 3/3；三情景图渲染，AIPerf 分母核对 | PASS：观测、单变量、配对、门限与回滚闭环 |

以下是前一轮两篇正文的历史验证记录；本批整合后的检查另记于其后。

- T0：`check_links --strict` 对 455 页报告 broken、ambiguous、bare_index、stale_section、orphans 均为 0；`check_math`、`check_markdown`、`check_assets` 对 53 个当前改动 Markdown 文件均为 0 error、0 warning；定点 `git diff --check` 为 0。当前改动集合含另一任务的文件，以上结果是整集合结果。
- 教学算术独立复核：三轮无缓存位置数 15、有缓存新位置数 6；稠密因果位置对数 46 与 21；所声明参数下 MHA/GQA/MQA 的六位置 KV 理想容量分别为 1536/768/384 B。
- 当前工作区的 `build --changed` 在另一任务的 `slime/14_slime_megatron_training_analysis.md:60` 指向不存在的 `15_slime_loss_parallelism_analysis` 标题锚点处停止，未检查到本域。没有改动该页。
- 为验证本域，临时从 `HEAD` 建隔离快照，只叠加本次两篇正文、05 索引与两个上级索引，使用仓库现成依赖构建：455 页，broken links、missing anchors/assets/legacy routes、orphans 均为 0；新两页共 38 个公式的 MathJax 渲染通过。临时快照在校验后自动清理。
- 两幅 Mermaid 原理图已用仓库所带 Mermaid 与无头浏览器实际渲染并目检：输入、每轮待处理 token、KV 写入、停止边界及缓存/重算两条路线一致。此条记录写入时两篇尚待非作者审阅；该审阅现已完成，见上表。未运行模型、GPU 或性能实测。

### A 批整合后复核

- 本批 10 篇正文、6 个 Mermaid 原理图和 4 张带生成源的 SVG 均已核对。6 个 Mermaid 块用仓库自带浏览器运行时解析；T11、T15、T16、T18 等实际渲染并目检，SVG 的数字、引用和边界亦已目检。T16 两方案同为 10 个本窗口输入 token：整段在轮 1 用满 8 的预算，使两个已就绪 decode 等到轮 2；分块在轮 1 同批处理它们与 L 的首 4 个 token。轮中 KV 最高分别为 14 与 10 slot，终态均为 L 的 8 slot。T19 非对称例用十进制最近偶数舍入复算，代码 `(0,2,3,5,13)`、还原 `(-1.2,-0.4,0,0.8,4.0)`、绝对误差 `(0,0,0.2,0.1,0.2)`。
- 当前工作区 `python tools/check_links.py --strict` 检查 463 页，broken、ambiguous、bare_index、stale_section、orphans 全 0；`check_math`、`check_markdown`、`check_assets` 对 76 个当前改动 Markdown 文件均 0 error、0 warning，定点 `git diff --check` 通过。76 文件包含其他任务的未提交改动；上述结果是实际当前工作区的检查，不声称其他任务内容已经过本批事实审阅。
- 当前工作区的全库 Mermaid corpus 在另一个任务的 `wiki/02_engineering/03_infer_frameworks/vllm/20_vllm_fused_ops_and_kernels_analysis.md:462` 因含括号的边标签解析失败；本批 6 个 Mermaid 块定点解析通过。当前工作区的 `build --changed` 仍在另一任务的 `slime/14_slime_megatron_training_analysis.md:60` 失效锚点处停止；本批未改这些页面。
- 从 `HEAD` 临时建立干净快照，仅叠加本批 `05_inference/`、06 §5.2 和上级索引，使用仓库现成依赖完整构建：463 页，broken links、missing anchors/assets/legacy routes、orphans 全 0；随后用同一快照运行 MathJax 浏览器检查，本批 10 页共 532 个公式 PASS。临时快照校验后自动清理。该结果证明本批页面可建站和渲染，不代表当前含其他任务改动的工作树已完整构建通过。

### 全域 29 篇完成后的最终复核

- 当前工作区 `check_links --strict` 覆盖 482 页，broken、ambiguous、bare_index、stale_section、orphans 均为 0；`check_math --changed --strict`、`check_markdown --changed --strict`、`check_assets --changed --strict` 各检查 125 个当前改动 Markdown 文件，均 0 error、0 warning；`git diff --check` 通过。125 文件包含另一任务的未提交内容，检查结果不等于对该任务的事实审阅。
- 29 篇正文和本级 index 合计 30 个 Markdown 文件。当前工作区的 `build --changed` 仍被另一任务 `slime/14_slime_megatron_training_analysis.md:60` 的失效锚点阻断；全库 Mermaid corpus 仍有另一任务的 vLLM fused ops 图语法错误。本批未改这些页面。
- 从 `HEAD` 建隔离快照，仅叠加整个 `05_inference/`、06 §5.2 改动和两个上级索引，完整构建 482 页成功；broken links、missing anchors/assets/legacy routes、orphans 均为 0。用该构建对本域 29 篇逐页运行浏览器 MathJax：28 篇含公式，共 1191 条公式 PASS；定点运行仓库自带 Mermaid 浏览器运行时，21 个图块解析 PASS。带生成源的 SVG 已逐图目检，正文算例与图的数字经上表非作者复算。
- 所有 29 篇正文由非作者审阅得出 PASS；T01、T40、T41 的图和来源经过修订与复审。页面数字均明确为教学推演；未运行模型、GPU、多机传输或真实性能压测，也未暂存、提交、推送、合并或发布。
