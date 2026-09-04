---
title: "Megatron-LM 推理引擎深度解析(Inference Engine)"
---

# Megatron-LM 推理引擎深度解析(Inference Engine)

> **源码基线**:`NVIDIA/Megatron-LM@85902ef599ea4eb06ada7567a479c524b605767a`(`dev`,2026-09-01)
> **重定基线**：2026-09-01 由 `71092579`（2026-08-27）推进，跨 7 个提交；本页落在本轮改动文件上的引用已按 difflib 逐行对齐重定位（含裸续引 `:NNN`），指向历史基线（`ee3f1ff` / `232c478d4`）的引用按原样冻结、未参与重定位。
> **重定基线**:2026-08-28 由 `ee3f1ffa…`(2026-05-19)推进,跨 578 个提交;本页全部 `path:line` 形式的引用已在新基线下逐条重核;**代码块内被点名的符号与不带行号的裸路径不在该次扫描口径内**,已知漏网处已于 2026-08-28 单独更正。
> 核心文件:`megatron/core/inference/` 下 `engines/`(`megatron/core/inference/engines/dynamic_engine.py` 2614 行、`megatron/core/inference/engines/static_engine.py`)、`megatron/core/inference/contexts/dynamic_context.py`(4021 行)、`megatron/core/inference/contexts/kv_block_allocator.py`、`megatron/core/inference/scheduler.py`
> 配套阅读:`30_megatron_rl_posttraining_consistency_analysis.md`(RL rollout 用的就是本引擎)、`23_megatron_precision_cudagraph_fusion_analysis.md`、`14_megatron_ep_analysis.md`
> 定位:系统性专题。`30_megatron_rl_posttraining_consistency_analysis.md` 把推理引擎当作 RL rollout 的积木一笔带过,本文拆开它内部。
> **叙事顺序**:本页按五拍组织——背景 → 为什么这么设计(含被否掉的替代)→ 实现思路与细节 → 约束 → 发展趋势。
> **最近更新**:2026-08-28。按五拍重排章节顺序;机制正文与既有引用未改。

---

## 0. 总览

推理(自回归生成)与训练是**两种不同的负载**:训练是一次大前向+反向、batch 固定、要可微;推理是**逐 token 生成**、请求动态来去、不需反向、要低延迟高吞吐。所以 Megatron 有一套独立的推理引擎(`megatron/core/inference/`),配 `transformer_impl='inference_optimized'`(见 `30_megatron_rl_posttraining_consistency_analysis.md` §4)。

两个引擎变体,本文当作"调度器/dispatcher"那样逐个解读:

| 引擎 | 代码 | 批处理 | KV cache | 用途 |
|------|------|--------|----------|------|
| **Static** | `megatron/core/inference/engines/static_engine.py`(406 行) | 定长批,同进同出 | 简单 | 离线批量推理 |
| **Dynamic** | `megatron/core/inference/engines/dynamic_engine.py`(2614 行) | **连续批处理**(in-flight) | **块级**(paged 式) | 在线服务、RL rollout |

> [!update] 该特性自 `dev@232c478d4`(2026-06-16)引入,行号已重核至基线 `71092579`。新增**高层封装 API**(#4697,`megatron/core/inference/apis/`)。本文 §3~§9 描述的 `DynamicInferenceEngine` / `DynamicInferenceContext` / `TextGenerationController` / `model_inference_wrappers` 现被官方降格为**底层积木**;典型用法改用两个 vLLM 风格的门面类:
> - **`MegatronLLM`**(同步,`megatron/core/inference/apis/llm.py`):`generate(prompts, sampling_params)` 一行出结果,单 prompt 也**总是返回 `list[DynamicInferenceRequest]`**;含 `pause`/`unpause`/`suspend`/`resume`/`shutdown` 生命周期与 `with` 上下文管理器。
> - **`MegatronAsyncLLM`**(异步,`megatron/core/inference/apis/async_llm.py`):额外提供 `serve(ServeConfig(...))` 起 **OpenAI 兼容 HTTP 服务**;**强制 `use_coordinator=True`**(direct 模式会在 `__init__` 抛 `ValueError`,因 direct 模式的同步 `engine.generate()` 与调用方 asyncio loop 冲突)。
> - **`ServeConfig`**(`megatron/core/inference/apis/serve_config.py`):`host`/`port`(默认 `0.0.0.0:5000`)/`parsers`/`verbose`/`frontend_replicas=4`。`SamplingParams`、`DynamicInferenceRequest(Record)` 从 `megatron.core.inference` 重导出(`megatron/core/inference/apis/__init__.py`)。
>
> 两种执行模式:`use_coordinator=False`(**direct**,调用方自己管数据分片)/ `use_coordinator=True`(**coordinator**,引擎跨 DP 副本路由请求,HTTP serving 必需)。调用方仍须**自行** `initialize_megatron(...)` + 建模型 + `model.eval()`。引擎层对应新增 `step_modern`(`megatron/core/inference/engines/dynamic_engine.py:2197`)/`step_legacy`(`:2203`)/`async_step`(`:2160`)分步入口。已知限制(见 `core/inference/README.md`):coordinator 模式下 `engine.reset()` 不安全(会重绑 `_cond`/`_state_events` 致死锁、并静默把 `use_coordinator` 置 `False`);HTTP 前端固定在 global rank 0;响应 `"model"` 字段恒为 `"EMPTY"`。RL 权重热更新门面(`suspend_for_refit`/`update_weights_from_collective`/`resume_after_refit`)与 `megatron serve` CLI 列入 roadmap、尚未落地。
>
> 本文后续各节仍**有效且更精确**——它们正是这层门面背后的实现细节。下文 path:line 中,`megatron/core/inference/engines/dynamic_engine.py` 的部分行号已随版本漂移(如 §7 的 `create_cuda_graphs` 现为 `:367`(`ee3f1ff` 为 `:325`,`232c478d4` 为 `:363`))。

---

## 1. 背景:自回归推理为何需要专门的引擎

### 1.1 KV cache,以及它带来的三个新问题

一次生成 = 给定 prompt,逐个吐 token,每个新 token 都要 attend 到**之前所有 token**。朴素做法每步把整条序列重新前向 —— `O(S²)` 重复计算。于是有 **KV cache**:把每个 token 算过的 Key/Value 存下来,新 token 只算自己的、复用缓存。

但 KV cache 带来三个新问题,正是推理引擎要解决的:

1. **两相负载迥异**:处理 prompt(prefill)和逐 token 生成(decode)的计算特征完全不同(§1.2)。
2. **显存碎片**:每条序列的 KV cache 长度不同、还在增长,预分配定长连续缓冲极浪费(§5)。
3. **批利用率低**:不同请求长度不一、来去时间不一,定长批里短请求早早算完却要空等最长的(§4)。

### 1.2 Prefill 与 Decode:两相,两种瓶颈

```
请求:prompt = [t0 t1 t2 t3]  →  生成 [g0 g1 g2 ...]

① Prefill:  一次前向处理整个 prompt(4 个 token)
            → 填满 prompt 的 KV cache,产出第一个 token g0
            计算特征:大 GEMM,【计算受限 compute-bound】

② Decode:   逐 token,每步只前向 1 个 token,attend 到全部已缓存 KV
            g0 → g1 → g2 → ...  每步产 1 token
            计算特征:GEMM 极小(1 token),却要从 HBM 读整个 KV cache
                     【显存带宽受限 memory-bound】
```

关键洞察:
- **Prefill 算力密集**,一次喂很多 token,GPU 算得满。
- **Decode 带宽密集**,每步只 1 token,GEMM 小到喂不饱 GPU,瓶颈在"把模型权重 + KV cache 从 HBM 搬进来"。**decode 是自回归推理的真正瓶颈**。
- **batching 对 decode 至关重要**:多条序列一起 decode,模型权重只从 HBM 读一次就摊给整批 → 带宽利用率上去了。这是连续批处理(§4)的根本动因。

---

## 2. 为什么这么设计:自研引擎 + 一个进程级"我在推理"开关

摆在面前有两条更省事的路:**①** 训练框架根本不碰生成,把权重导出去交给 TensorRT-LLM / vLLM 这类专用推理栈;**②** 就用训练那套前向,靠 `model.eval()` + `torch.no_grad()` 把它当推理跑。两条 Megatron 都没走。源码陈述了其中三条理由,第四条源码沉默,由本页重建并标为推断。

**① `engines/` 里确实留过"转发给 TensorRT-LLM"的槽位,但它从未落地就被整文件删除。**
早期 `megatron/core/inference/engines/` 是四个文件并列:`__init__.py`、`abstract_engine.py`、`mcore_engine.py`、`trt_llm_engine_wrapper.py`。`TRTLLMEngineWrapper(AbstractEngine)` 从头到尾是个桩——`generate()` 直接 `return prompts`、`is_model_trt_llm_exportable()` 恒 `return False`,两个方法都挂着 `# TODO : Will use high level apis to implement this` / `# TODO : Need to implement this`。提交 `ca9edbef9`(2024-06-07,commit message 即 "Refactor ammo")把该文件整体删除,此后 `engines/` 只剩 static / dynamic 两个**自研**引擎(§3、§4)。
今天源码走的是另一条路——**接口对齐 vLLM、实现留在树内**:门面类提供「a vLLM-style `generate(prompts, sampling_params)` API」(`megatron/core/inference/README.md:3`);roadmap 里的 `megatron serve` CLI「mirrors `vllm serve`」(`:69`);连 HTTP 前端的 rank 放置都说要「mirroring how vLLM's `--headless` is invoked today」(`:82`)。

**② "现在是不是在推理"必须是一个独立的进程级标志,`self.training` / `no_grad` / `inference_context` 三个现成信号全部被点名否掉。**
`InferenceMode` 的 docstring 把话说死:需要区分推理与非推理(「e.g. training, RL logprobs」)路径的模块「should read `InferenceMode.is_active()` **rather than relying on** `self.training`, `torch.is_grad_enabled()`, or `inference_context is not None`」(`megatron/core/inference/utils.py:20-26`)。注意它把 **RL logprob 重算**与 training 并列为**非推理**路径——而 RL 训练相恰恰是用 `eval()` + `no_grad` 跑的,三个旧信号在那里全部指向"推理"。
**被否掉的替代就写在历史里**:提交 `925422cd8`(2026-05-13,#4617,commit message 即 "One single flag that determines if we are in inference")之前,`MoELayer` 用 `nn.Module` 的 mode 切换来选 dispatcher——它重写了 `def train(self, mode: bool = True)`,`mode` 为真换回训练 dispatcher、为假换成推理 dispatcher,`forward` 里则判 `not self.training`。该 commit 把这段 `train()` 重写**整段删除**,改成在 `forward` 入口读 `InferenceMode.is_active()` 选 dispatcher(基线 `megatron/core/transformer/moe/moe_layer.py:716`、`:625`),同批还改了 `gpt_model`(`megatron/core/models/gpt/gpt_model.py:339`、`:421`、`:437`、`:686`)、`transformer_layer`(`megatron/core/transformer/transformer_layer.py:773`、`:999`、`:1137`、`:1777`)等 15 个文件。引擎侧对称地在进出推理时置位/清位(`megatron/core/inference/engines/dynamic_engine.py:296`、`:806`、`:857`,`megatron/core/inference/engines/static_engine.py:133`)。
→ 决定取舍的判据是**让"模式"有唯一的真相源**:模块散着各自猜模式,只要有一个猜错就静默走错 kernel;改成单一标志后,判错会同时错在所有模块,更容易发现,也让 RL 那种"eval + no_grad 但不是推理"的第三种状态第一次可表达。

**③ KV 显存是上来就吃掉的一整块,再在内部切块,而不是随用随 `torch.empty`。**
`DynamicInferenceContext` 的 docstring 明写:块级 KV cache 的「memory buffer is allocated **up front**(size `buffer_size_gb` if `unified_memory_level == 0`, or `buffer_size_gb + paused_buffer_size_gb` if `unified_memory_level == 1`), that is divided into blocks and dynamically assigned to requests. At any given step, **any unassigned blocks equate to unused space**」(`megatron/core/inference/contexts/dynamic_context.py:229-243`)。它同时把代价说了出来:没分出去的块就是纯浪费。§5 讲的是"块怎么切",这里补的是"为什么整块预留"。

**④ 异步门面宁可在 `__init__` 硬失败,也不做两套 event loop 的桥接。**
`MegatronAsyncLLM.__init__` 上方的注释直接给出理由:direct 模式会「invoke the synchronous `engine.generate()` from inside the caller's asyncio loop, which collides with the engine's loop-bound internal state(`_cond`, `_state_events`)」,而 coordinator 模式通过 `start_listening_to_data_parallel_coordinator` 把这些原语重绑到一个守护线程的 loop 上,「and avoids the conflict」;紧接着 `if not use_coordinator: raise ValueError(...)`(`megatron/core/inference/apis/async_llm.py:42-55`)。README 补上这是**暂时**的:「Tracked for an upstream `engine.async_generate(...)`(or engine loop-rebinding)fix」(`megatron/core/inference/README.md:75`)。
→ 判据是**把一类难查的运行期故障换成一次构造期报错**:桥接做错的表现是 `RuntimeError: This event loop is already running` 或挂死,而硬失败发生在第一行。

> [!note] 推断
> 源码陈述的是**事实**:TRT-LLM wrapper 是个桩且被删除、README 反复对标 vLLM 接口、`InferenceMode` 点名否掉三个旧信号、KV buffer 预分配、异步门面拒绝 direct。**"Megatron 因此选择了自研引擎而非外部推理栈"这层因果由本页承担**——源码从未写过一句"我们决定不用 TensorRT-LLM"。上面 ①④ 两段末尾的"判据"("唯一真相源"、"把运行期故障换成构造期报错")同样是本页的重建,不是作者自陈。要引用这几条判断,请回到 `megatron/core/inference/README.md:3`、`:69`、`:82`、`megatron/core/inference/utils.py:20-26`、`megatron/core/inference/apis/async_llm.py:42-55` 这几个 locator,不要引用本段推断。

---

## 3. 引擎变体① StaticInferenceEngine —— 定长批

`megatron/core/inference/engines/static_engine.py`。一批请求**同时开始、同时结束**:凑齐一个固定大小的 batch,prefill,然后一起 decode,直到**最长的那条**生成完,整批才返回。

- 优点:实现简单,无动态调度。
- 缺点:**短请求空等长请求** —— batch 里某条 20 token 就停了,但要陪最长的 2000 token 跑完,那段 GPU 槽位全空转。
- 适用:离线批量推理(所有 prompt 已知、长度相近、不在乎单条延迟)。

---

## 4. 引擎变体② DynamicInferenceEngine —— 连续批处理

`megatron/core/inference/engines/dynamic_engine.py`。类注释点明:"in-flight batching and a dynamic block-level KV cache (similar to paged attention)"。

### 4.1 连续批处理(continuous / in-flight batching)

不再"同进同出"。每个 decode step 之后:
- **已完成的序列立即离开**,腾出槽位。
- **等待队列里的新请求立即补进来**(可能先做它的 prefill)。

```
定长批:    [seq A ████████████████]   A 早完但占着槽位空等 ───────┐
            [seq B ████]              B 完了也空等                │
            [seq C ████████████████████████]  最长的 C            │
            └──────────── 整批一起返回,前面大片空转 ──────────────┘

连续批处理:[seq A ████] →(A 完,新请求 D 补入)→ [seq D ████████]
            [seq B ████████] →(B 完,E 补入)→ [seq E ██████]
            [seq C ████████████████████████]
            └─ GPU 槽位始终被填满,无空转 ─┘
```

效果:GPU 利用率从"被最长序列拖累"变成"始终满载"。这是在线服务吞吐的关键。

### 4.2 Chunked Prefill

问题:一个长 prompt 的 prefill 是个大前向,会**长时间霸占 GPU**,正在 decode 的其他请求被卡住、延迟飙升。

`enable_chunked_prefill`:把长 prefill **切成若干 chunk**,一个 chunk 一个 chunk 地做,**穿插在 decode step 之间** —— prefill 不再独占,decode 请求的延迟被拉平。`chunked_prefill_request_id` 跟踪正在分块的那个请求,把它钉在等待队列头部。

---

## 5. 块级 KV Cache(paged-attention 式)

`megatron/core/inference/contexts/dynamic_context.py`(`DynamicInferenceContext`)+ `megatron/core/inference/contexts/kv_block_allocator.py`(`KVBlockAllocator`)。

### 5.1 动机:显存碎片

每条序列要存 KV cache,长度**不一且持续增长**。若按"最大长度预分配一块连续显存":
- 长度 100 的请求占了 4096 的坑 → 浪费 97%。
- 连续分配 → 外部碎片,大请求进不来。

### 5.2 解法:固定大小的块

KV cache 显存切成大量**固定大小的块(block)**。一条序列的 KV cache = **一串块的列表(block table)**,块之间在物理显存上**不必连续**:

```
物理 KV 显存:  [blk0][blk1][blk2][blk3][blk4][blk5][blk6]...

seq A 的 KV  →  block table = [blk0, blk3, blk5]   ← 逻辑连续,物理分散
seq B 的 KV  →  block table = [blk1, blk2]
                序列增长时 KVBlockAllocator 按需再分一个块
                序列结束时块立即回收进空闲池
```

这就是 vLLM **PagedAttention** 的思想:按需分块、近零浪费、无外部碎片。Mamba 类模型的状态另有 `MambaSlotAllocator`(状态空间模型的"KV"是固定大小的循环状态,按 slot 分配)。

### 5.3 背压:overflow 异常

显存有限,块会用完。`DynamicInferenceContext` 定义了一组 `ContextOverflowError`(`megatron/core/inference/contexts/dynamic_context.py:107`):`RequestOverflowError`(请求数超,`:127`)、`TokenOverflowError`(`:133`)、`MaxSequenceLengthOverflowError`(`:139`)、`BlockOverflowError`(块用尽,`:146`)。引擎据此**背压** —— 新请求进不来就留在等待队列,而不是 OOM 崩溃。

> [!update] 该特性自 `dev@232c478d4`(2026-06-16)引入,行号已重核至基线 `71092579`。**超长 `num_tokens_to_generate` 改为「钳制」而非「拒绝」**(#5181,`megatron/core/inference/engines/dynamic_engine.py:1098-1109`),对齐 vLLM 行为。旧逻辑:`prompt_len + num_tokens_to_generate > max_sequence_length` 直接把请求标 `FAILED` + `MaxSequenceLengthOverflowError`。新逻辑:仅当 `num_tokens_to_generate < 0` 或 prompt 本身已超长(`remaining_tokens < 0`)才 FAILED;若只是请求的生成长度超过剩余预算,则**钳到 `remaining_tokens`** 并(rank 0)`warnings.warn`,请求照常受理。意义:与其它推理框架行为一致,长请求不再被硬拒。

> [!update] 该特性自 `dev@232c478d4`(2026-06-16)引入,行号已重核至基线 `71092579`。**非均匀 PP 下 KV `layer_map` 的尺寸修正**(#4775,`megatron/core/inference/contexts/dynamic_context.py:385`)。纯 Transformer 模型原按 `num_layers // pp_size` 估算本 rank 的注意力层数,在 `account_for_embedding/loss_in_pipeline_split`、首尾 PP 段不等分、或自定义 `pipeline_model_parallel_layout` 时会**算错**,导致 `append_key_value_cache` 抛 `KeyError`。修复改调与 `TransformerBlock` 同源的 `get_num_layers_to_build(model_config, vp_stage=None, pp_rank=...)`(`pp_rank` 取自 `pg_collection.pp`),使 §5.2 的 block table / `layer_map` 在非均匀流水线切分下也对齐。

> [!update] 该特性自 `dev@232c478d4`(2026-06-16)引入,行号已重核至基线 `71092579`。**修正推理元数据张量 dtype**(#4855,`megatron/core/inference/contexts/dynamic_context.py` / `megatron/core/inference/contexts/gpu_view.py`)。`token_to_block_idx` 等按 token 计数的索引字段从 `int32` 改为 `int64`(超长序列时 int32 会溢出),CPU bookkeeping buffer 的字节偏移与 8 字节对齐(含 Mamba 段 `batch_indices_decode` 为 int64、其余 int32)相应重排。属底层正确性修复,不改 §5 的分块语义。

---

## 6. Prefix Caching

`enable_prefix_caching`。若多个请求**共享 prompt 前缀**,这段前缀的 KV 只需算一次、块可**跨请求共享**。

```
请求1:[系统提示 ……][用户问题 A]
请求2:[系统提示 ……][用户问题 B]
       └─ 同一段前缀 ─┘
       这段的 KV 块算一次,两请求共享 → 省 prefill 计算 + 省显存
```

- LRU 驱逐:`prefix_cache_lru_clock` 单调时钟给块排序,显存紧时淘汰最久未用的前缀块。
- 命中统计:`prefix_cache_hits`、`prefix_cache_blocks_matched`。
- **对 RL rollout 收益巨大**:GRPO 对同一 prompt 采样多条 response(`14_megatron_ep_analysis.md` / `30_megatron_rl_posttraining_consistency_analysis.md`),prompt 前缀的 KV 算一次共享给所有 rollout;聊天服务里共享的 system prompt 同理。

> [!update] 该特性自 `dev@232c478d4`(2026-06-16)引入,行号已重核至基线 `71092579`。**prefix cache / MTP 统计改为「整个引擎生命周期累计」**(#4101,`megatron/core/inference/engines/dynamic_engine.py`)。`DynamicInferenceContext.prefix_cache_hits` / `prefix_cache_blocks_matched` 是**每 step 清零**的瞬时量;引擎现把它们累加进生命周期级累加器 `self._prefix_cache_hits` / `_prefix_cache_blocks_matched`(`:348-349`,每步 `+=` 后把 context 计数清零,`:2016-2017`),`get_metrics` 上报累计值(`inference/prefix_cache_hits` 等,`:2064`)。意义:metric 不再只反映最后一步,而是反映整个服务期的真实命中。

---

## 7. CUDA Graph 在推理里

decode step 是"逐 token、kernel 小而多",CPU 启动开销占比极高 —— 正是 CUDA Graph 的主场(见 `23_megatron_precision_cudagraph_fusion_analysis.md` §4)。

但 CUDA Graph 要求形状固定,而连续批处理的"当前 batch 里有几个请求"是变化的。`DynamicInferenceEngine` 的解法(`create_cuda_graphs`,`megatron/core/inference/engines/dynamic_engine.py:367`):
- 预先枚举一组**典型的 batch 维度**(`cuda_graph_batch_dimensions_list`,不同 request count)。
- **为每种 batch size 各捕获一张图**,warmup 阶段录完。
- 运行时按当前实际请求数,**选最接近的那张图**重放(不足部分 padding)。
- `inference_cuda_graph_scope`(`layer` / `block`)控制图化粒度;`use_cuda_graphs_for_non_decode_steps` 决定 prefill 步是否也图化;MTP 另有独立的图 warmup。

> [!update] 该特性自 `dev@232c478d4`(2026-06-16)引入,行号已重核至基线 `71092579`。**CUDA Graph 尺寸分布从「线性」改为「指数递减 + 混合 prefill 网格」**(#3509,`InferenceConfig.cuda_graph_sizing_distribution`、`megatron/core/inference/config.py:120` 的 `CudaGraphSizingDistribution` 枚举、CLI `--inference-dynamic-batching-cuda-graph-sizing-distribution`)。本节"为多种 batch size 各捕一张图"的结论不变,变的是**枚举哪些尺寸**:
> - **`EXPONENTIAL`(新默认)**:token 数从 `cuda_graph_max_tokens` 起**逐次减半**直到 `tp_size`(log 间距),总图数约 `log2(max_tokens)`,每个尺度的相对 padding 有界(最坏约 2×)。
> - **`LINEAR`(旧行为)**:`[1,2,4] + range(8,256,8) + range(256,max+1,16)`,高端图更密。
> - 混合 prefill/decode 另按 `cuda_graph_mixed_prefill_count`(默认 16)走**网格**枚举。动机:旧线性分布在大 `max_tokens` 下图数爆炸且高端浪费,指数分布用更少的图覆盖更宽的请求规模区间。`create_cuda_graphs` 现位于 `megatron/core/inference/engines/dynamic_engine.py:367`(`232c478d4` 为 `:363`,原 §7 引用的 `ee3f1ff` `:325`)。

> [!update] 该特性自 `dev@232c478d4`(2026-06-16)引入,行号已重核至基线 `71092579`。**MTP 推测解码:术语更名 + 按位接受率指标**。`num_mtp_heads` 全面更名为 `num_mtp_depths`(#4101,`megatron/core/inference/text_generation_controllers/text_generation_controller.py`、`megatron/core/inference/engines/dynamic_engine.py`);推测解码接受统计从两个标量改为**按位置(per-position)张量** `_spec_tokens_proposed_per_pos` / `_spec_tokens_accepted_per_pos`(长度 = `num_speculative_tokens`,索引 i 对应 MTP 第 i 个 draft token),`get_metrics` 既报聚合 `inference/spec_decode_acceptance_rate` 也报逐位接受率;prefill 请求被排除出分母(MTP 只对 decode 请求提议)。配套 #3458 在训练侧 MTP 模块(`megatron/core/transformer/multi_token_prediction.py`)加了 per-layer loss / 接受率计数器。意义:可定位"哪一深度的 draft token 接受率塌掉",指导 `num_speculative_tokens` 调参。

> [!update] 该特性自 `dev@232c478d4`(2026-06-16)引入,行号已重核至基线 `71092579`。**Nemotron 的 prefill engine step 优化**(#4764,`megatron/core/inference/contexts/attention_context/mamba_metadata.py`、`megatron/core/ssm/mamba_mixer.py`、`megatron/core/ssm/ops/causal_conv1d_varlen.py`)。混合(Mamba)模型在 CUDA Graph 兼容的固定尺寸 buffer 下,中间状态提取元数据改用 `padded_prefill_count * MAX_INTERMEDIATE_OFFSETS_PER_REQUEST` 作为上界、并把 fill 操作限定在 `[:max_count]` 区间(而非整个 buffer),减少混合 prefill/decode 图里的无效填充开销;同时移除了 `megatron/core/transformer/moe/token_dispatcher_inference.py` 里一处冗余逻辑。属 §4.2 chunked/混合 prefill 在 SSM 模型上的性能修补。

---

## 8. Scheduler 与请求池

`megatron/core/inference/scheduler.py`。`Scheduler` 维护几个请求池:

```
新请求 ──► active_request_pool   (在批,正在跑)
           waiting_request_pool  (排队,批满了进不去)
           completed_request_pool(已完成,待取回)

每个 step:active 里完成的 → completed;从 waiting 提升新请求填补空位
```

`add_request` 按当前批是否已满,决定进 active 还是 waiting。配 `AsyncStream` 支持**流式输出**(token 边生成边返回)。

---

## 9. 周边

| 组件 | 作用 |
|------|------|
| `model_inference_wrappers/` | 把模型包装成推理接口,走 `inference_optimized` 路径 |
| `text_generation_controllers/` | decode 主循环、采样调用、detokenize |
| `sampling/` + `megatron/core/inference/sampling_params.py` | greedy / top-k / top-p / temperature;`return_log_probs`(RL 要的 logprob) |
| `text_generation_server/` | HTTP 服务,OpenAI 兼容的 `chat/completions` 端点 |
| `megatron/core/inference/data_parallel_inference_coordinator.py` | 多 DP rank 推理的请求协调 |
| `megatron/core/inference/unified_memory.py` | KV cache 用统一内存,引擎 `suspend`/`resume` 时把 KV 换出(RL 里训练相不需要推理引擎时腾显存) |
| `megatron/core/inference/async_stream.py` / `megatron/core/inference/engines/async_zmq_communicator.py` | 异步、流式、ZMQ 通信 |

`suspend`/`resume`(`megatron/core/inference/engines/dynamic_engine.py:799`/`:850`,共用 `suspend_resume_ctx` `:738`)对 **RL collocated 部署**很关键:训练相和推理相在同一批卡上轮流跑,推理引擎 suspend 时删 CUDA Graph、把 KV cache 换出统一内存,让出显存给训练;resume 时再恢复(见 `30_megatron_rl_posttraining_consistency_analysis.md` §7)。

> [!update] 该特性自 `dev@232c478d4`(2026-06-16)引入,行号已重核至基线 `71092579`。**「是否在推理」统一为一个进程级全局开关 `InferenceMode`**(#4617,`megatron/core/inference/utils.py:20`)。引擎进入推理时 `InferenceMode.set_active()`(`megatron/core/inference/engines/dynamic_engine.py:296`/`:857`、`megatron/core/inference/engines/static_engine.py:133`)、退出时 `unset_active()`(`megatron/core/inference/engines/dynamic_engine.py:806`),并提供 `with InferenceMode.active():` 上下文管理器。**为什么**:模型各模块此前靠 `self.training` / `torch.is_grad_enabled()` / `inference_context is not None` 来猜"现在是不是推理",这三者都不可靠(尤其 RL 训练相用 `eval()`+`no_grad` 重算 logprob 时会被误判)。改为单一标志后,`gpt_model`、`attention`、`moe_layer`/`router`/`experts`、`mamba_*`、`transformer_layer`(`inference_fuse_tp_communication`)等全部改读 `InferenceMode.is_active()` 决定走推理 kernel/dispatcher 还是训练路径。**对本文与 RL 页的影响**:`30_megatron_rl_posttraining_consistency_analysis.md` §4 所述 MoE 推理 dispatcher 的切换**不再**由 `MoELayer.train()` 重写驱动(该重写已删),而由 `MoELayer.forward` 入口的 `InferenceMode.is_active()` 决定——详见该页 §4 的 `[!deprecated]` 批注。

---

## 10. 约束:引擎不做什么、以什么为代价

前面九节几乎全是收益。这一节把代价、前提与失效条件集中列出,每条带 locator。

**① 调用方责任(门面类不替你做的事)。**
`megatron/core/inference/README.md:55-59` 列了三条硬前提:构造**之前**必须自己 `initialize_megatron(...)`(完整的 Megatron 分布式初始化);构造**之前**必须自己 `model.eval()`——「The class does not toggle model state」;`pause`/`unpause`/`suspend`/`resume` 这组生命周期方法**要求 `use_coordinator=True`**,direct 模式下抛 `RuntimeError`(实现见 `megatron/core/inference/apis/_llm_base.py:353-355` 的 `_assert_coordinator`)。coordinator 模式下 `generate(...)` 还只在 primary rank 有效,非 primary 直接 `RuntimeError`(`megatron/core/inference/apis/_llm_base.py:347-351`)。

**② `MegatronAsyncLLM` 不支持 direct 模式。** 构造期 `ValueError`,原因见 §2 ④(`megatron/core/inference/apis/async_llm.py:42-55`;`serve()` 处另有一次同类校验,`:187`)。

**③ coordinator 模式下 `engine.reset()` 不安全,且是两种不同的坏法。**
README 把两条失效路径拆得很细(`megatron/core/inference/README.md:77-80`):*死锁*——`reset()` 是**重绑**而非原地修改 `_cond` / `_state_events`,已挂起在旧对象上的协程再也等不到 `notify_all()` / `set()`,下一次 `generate()` 直接挂住;*静默损坏*——`reset()` 顺手把 `self.use_coordinator` 置 `False`,失败请求处理、调度通知、`suspend()` 状态机全部悄悄改走 direct 分支,不挂死但行为错、更难查。示例脚本因此禁掉 `--inference-repeat-n > 1` 与 `--use-coordinator` 同开。direct 模式的 reset 是安全的。

**④ HTTP 服务两处写死。** 前端固定在 global rank 0,`ServeConfig` 没有 per-rank 的 `role` 开关,放置只能靠 launcher 控(`megatron/core/inference/README.md:82`);响应里 `"model"` 恒为 `"EMPTY"`,既不回显也不校验请求里的 `model` 字段,也没有 `GET /v1/models`(`:84`)。

**⑤ KV 池是"上来就吃掉"的定额,不是弹性的。**
默认 `buffer_size_gb = 20`(GB,`megatron/core/inference/config.py:160`)、`block_size_tokens = 256`(`:157`)、`unified_memory_level = 0`(`:197`)。`unified_memory_level` 为 0 时 paused buffer **含在** `buffer_size_gb` 里,为 1 才额外用 CPU 内存(`:161-173`)。`max_requests` 的实际上限「primarily limited by the combination of `buffer_size_gb` and `max_sequence_length`」(`:185-188`),`max_tokens` 则「primarily limited by prefill activation memory usage」(`:191-194`)——§4/§5 的"连续批处理 + 按需分块"并不能突破这两条预算。

**⑥ 块池里有一块永远不可用。** `KVBlockAllocator` 留了一个 `dummy_block_idx`,可用块数是 `total_count - 1`;`active_count = total_count - paused_count - 1` 且 `assert self.active_count >= 1`,即 `paused_count` 必须严格小于 `total_count - 1`(`megatron/core/inference/contexts/kv_block_allocator.py:45-48`)。

**⑦ 投机解码(MTP)的两条硬边界。** `num_speculative_tokens` 必须非负(`megatron/core/inference/engines/dynamic_engine.py:239`),且**严格小于** `block_size_tokens`(`megatron/core/inference/contexts/dynamic_context.py:309-312`)——一次投机的 token 不能跨出一个块。

**⑧ Flash MLA 把块大小钉死在 64。** 模型是 MLA 且开 `cache_mla_latents` 时,`block_size_tokens != 64` 直接 assert 失败,错误信息里直接给出要改的 CLI 开关(`megatron/core/inference/contexts/dynamic_context.py:288-291`)。这和 ⑤ 的默认值 256 冲突,MLA 模型必须显式改配置。

**⑨ suspend 让出的显存是真删掉状态,不是换页。** `suspend()` 先 `InferenceMode.unset_active()`,再进 `suspend_resume_ctx` 调 `deallocate_inference_state_buffers()`(`megatron/core/inference/engines/dynamic_engine.py:799-812`);此后再去碰 context 的张量状态,得到的是 `TensorStateDeallocatedError`——它被明确归类为一种 `ContextOverflowError`,注释即「Context's tensor state is currently deallocated, such as when the engine has been suspended」(`megatron/core/inference/contexts/dynamic_context.py:165-169`)。§9 说的"让出显存给训练相"要按这个语义理解:不是暂停,是拆掉再重建。

**⑩ 背压不是无限排队。** §5.3 那组 `ContextOverflowError`(`megatron/core/inference/contexts/dynamic_context.py:107`)只保证"不 OOM 崩溃",请求该失败还是失败——`ActiveRequestCountOverflowError` 就发生在 warmup 请求数超过 `max_requests` 时(`:152-162`)。

---

## 11. 发展趋势

以下每条都锚在基线源码里能读到的 `TODO` / `DeprecationWarning` / README roadmap 上;**方向解读是本页的推断,不是作者自陈**。

**① Static 引擎正在下线,而且已经被 Dynamic 引擎"借壳"。**
§3 把 Static 与 Dynamic 当两个平行变体讲;基线源码里 `StaticInferenceEngine.__init__` 无条件发 `DeprecationWarning`,原话是「`StaticInferenceEngine` will be deprecated in a future version of Megatron-core. Please directly use `DynamicInferenceEngine` instead. **`StaticInferenceEngine` currently uses `DynamicInferenceEngine` under the hood.**」(`megatron/core/inference/engines/static_engine.py:64-69`);走 `legacy=True` 的老路径另发一条「The static engine will be deprecated and removed in the future version of megatron-core. Switch to DynamicInferenceEngine.」(`:60-62`)。→ §3 描述的**定长批语义**仍然成立,但它已经不是一条独立的实现路径了。

**② 三个 step 入口会收敛回一个 `step()`。**
`step_legacy()` 自己写了下线版本:「`step_legacy()` is deprecated and will be removed in `megatron-core` 0.16. Please use `step_modern()` going forward, **which will eventually be renamed to `step()`**」(`megatron/core/inference/engines/dynamic_engine.py:2203-2211`)。§0 的 [!update] 列出的 `step_modern` / `step_legacy` / `async_step` 三入口是过渡态。

**③ README 自陈的四条 roadmap。**(`megatron/core/inference/README.md:61-71`)
- **Dynamic streaming**:离线流式已可用 `engine.async_step()`;HTTP 流式还缺 coordinator / `InferenceClient` 协议携带**部分输出**(现在只能传最终 request record)(`:65`)。
- **Weight update APIs**:计划把现有 resharding/refit 原语包成 `suspend_for_refit()` / `update_weights_from_collective()` / `resume_after_refit()`,给"rollout 之间换权重"的 RL 流程用(`:67`)。这条直接落在本文 §9 的 suspend/resume 与 `30_megatron_rl_posttraining_consistency_analysis.md` 的 refit 之间。
- **`megatron serve` CLI**:单二进制启动器,复用 `MegatronAsyncLLM.serve(...)`,含单机与多机 headless 模式,「mirrors `vllm serve`」(`:69`)。
- **Config-based model construction**:`MegatronLLM(model="...")` 式构造 + 模型 recipe/ckpt 解析,把"自己建模型"从 §10 ① 的调用方责任里拿掉(`:71`)。

**④ direct 模式的异步 generate 已经有指定修法。**
`megatron/core/inference/apis/_llm_base.py:417-418` 挂着「TODO: replace with an upstream `engine.async_generate` so direct-mode async generate doesn't block the caller's event loop.」——正对着 §10 ② 那条约束。

**⑤ 采样/logprob 侧的已知缺口。** `megatron/core/inference/contexts/dynamic_context.py:3889` 的「TODO: @wdykas support top-n log probs.」;`text_generation_controller.py:1166` 与 `:2379` 两处同文的「TODO(ksanthanam): Evaluate whether it makes more sense to sample on 1 rank」——采样是否收拢到单 rank 仍未定。

---

## 12. 小结

- **推理 ≠ 训练**:逐 token 生成、请求动态、无反向 → 专门的引擎(`inference/`)+ `inference_optimized` 实现。
- **两相**:prefill 算力受限(整 prompt 一次前向),decode 带宽受限(逐 token,读整个 KV cache)—— decode 是真瓶颈,靠 batching 摊薄权重读取。
- **两个引擎**:Static(定长批,同进同出,离线用)vs **Dynamic(连续批处理 + 块级 KV cache,在线/RL 用)**。
- **连续批处理**:完成即离开、等待即补入,GPU 不再被最长序列拖累。
- **块级 KV cache(paged)**:KV 切固定块、按需分配、block table 串起来 → 消除显存碎片;Mamba 用 slot 分配。
- **Prefix caching**:共享前缀的 KV 块跨请求复用,LRU 驱逐 —— RL 多采样、聊天共享 system prompt 收益巨大。
- **CUDA Graph**:为多种 batch size 各捕一张图,运行时选最近的重放,对付 decode 的小 kernel 开销。
- **Scheduler + overflow 背压**:active/waiting/completed 池,块用尽则背压排队而非 OOM。
- **suspend/resume**:RL collocated 部署下推理引擎让出显存给训练相。

---

*生成依据:`Megatron-LM` `dev` 分支 `85902ef599ea4eb06ada7567a479c524b605767a`(2026-09-01;由 `71092579` 重定基线而来,更早一次为 2026-08-28 由 `ee3f1ff` 推进)。源码行号以该 commit 为准。配套文档:`30_megatron_rl_posttraining_consistency_analysis.md`、`23_megatron_precision_cudagraph_fusion_analysis.md`、`14_megatron_ep_analysis.md`。*

---

## 配置契约：`InferenceSetupConfig`

本页正文讲推理引擎的**机制**——Static/Dynamic 引擎、连续批处理、块级 KV cache、chunked prefill。本节给它的**配置面**：`InferenceSetupConfig` 是把这些机制暴露给用户的那一层，经 [[41_megatron_config_surface_analysis]] §2 的工厂自动转成 CLI。

**下表直接取自 `megatron/training/config/inference_config.py` 的类体**。注意它叫 `InferenceSetupConfig` 而非 `InferenceConfig`——它管的是**起一个推理会话需要的装配参数**（引擎类型、批与序列上界、采样默认值、服务端口），不是 `megatron/core/inference` 内部那些运行期结构的配置。两者的分界正是 [[41_megatron_config_surface_analysis]] §4 说的「args → 配置对象桥接」那一层。


### `InferenceSetupConfig`（`megatron/training/config/inference_config.py`，37 项）

| 字段 | 类型 | 默认 | 契约 | 行 |
|---|---|---|---|---|
| `inference_batch_times_seqlen_threshold` | `int` | `-1` | If (batch-size * sequence-length) is smaller than this threshold then batches will not be split up for pipelining. Requires setting --pipeline-model-parallel… | `:50` |
| `max_tokens_to_oom` | `int` | `12000` | Maximum number of tokens during inference (# in prompt + # to generate). Allows us to throw an error before OOM crashes server. | `:55` |
| `output_bert_embeddings` | `bool` | `False` | Output Bert embeddings (via mean pooling) from model, rather than its binary head output or entire hidden batch. | `:59` |
| `bert_embedder_type` | `Literal['megatron', 'huggingface']` | `'megatron'` | Select either Megatron or Huggingface as the Bert embedder. | `:63` |
| `use_legacy_static_engine` | `bool` | `False` | Use legacy static engine. (Current static engine uses dynamic engine under the hood.) | `:70` |
| `inference_max_requests` | `int` | `8` | Maximum number of requests for inference. | `:73` |
| `inference_max_seq_length` | `int` | `2560` | Maximum sequence length expected for inference (prefill + decode). | `:76` |
| `inference_dynamic_batching` | `bool` | `False` | Enable dynamic batching mode. | `:81` |
| `inference_dynamic_batching_buffer_size_gb` | `float` | `40.0` | Amount of on-GPU memory allocated for the KV cache. The total amount of memory allocated for the KV cache (CPU + GPU memory) depends on the value set for the… | `:84` |
| `inference_dynamic_batching_paused_buffer_size_gb` | `float \| None` | `None` | Amount of memory reserved for paused requests in the dynamic inference context. Active requests are paused when there are not enough active blocks available … | `:89` |
| `inference_dynamic_batching_mamba_memory_ratio` | `float \| None` | `None` | Percentage of memory buffer to allocate for Mamba states. If not specified, allocates Mamba state tensors for each KV cache block. Only used for hybrid models. | `:94` |
| `inference_dynamic_batching_block_size` | `int` | `256` | KV cache block size. It should be a multiple of 256. | `:98` |
| `inference_dynamic_batching_max_requests` | `int \| None` | `None` | Override the inference context's `max_requests`. By default, `max_requests` is set to the number of blocks in the context's memory buffer. | `:101` |
| `inference_dynamic_batching_max_tokens` | `int \| None` | `None` | Override the inference context's default `max_tokens`. | `:105` |
| `inference_dynamic_batching_num_cuda_graphs` | `int` | `16` | Maximum number of cuda graphs to capture, where the cuda graph batch sizes range from 1 to `max_requests`. The user can also pass -1, in which case we automa… | `:108` |
| `inference_dynamic_batching_track_paused_request_events` | `bool` | `False` | Track paused request ids by adding 'paused' events to each request's event history. This has a very minor impact on latency. | `:113` |
| `inference_dynamic_batching_track_generated_token_events` | `bool` | `False` | Track per-token events with timestamps for each generated token. When enabled, each generated token creates a GENERATED_TOKEN event with a timestamp, useful … | `:117` |
| `inference_dynamic_batching_unified_memory_level` | `Literal[0, 1]` | `0` | Set unified memory usage within the dynamic inference context. The levels are: 0) no unified memory, 1) allocate `memory_buffer` in unified memory. | `:121` |
| `inference_dynamic_batching_cuda_graph_mixed_prefill_count` | `int` | `16` | Number of mixed prefill requests to capture in a cuda graph. | `:125` |
| `inference_dynamic_batching_sampling_backend` | `Literal['torch', 'flashinfer']` | `'torch'` | Which sampling kernels to use during inference. Falls back to "torch" with a warning if "flashinfer" is requested but the package is not installed. | `:135` |
| `inference_dynamic_batching_async_sched_mode` | `Literal['legacy', 'serial']` | `'legacy'` | Async scheduling mode for dynamic batching. "legacy" (default) preserves the existing resolve-before-prepare path. "serial" speculatively prepares and forwar… | `:139` |
| `inference_dynamic_batching_logprobs_mode` | `Literal['raw_logprobs', 'processed_logprobs']` | `'raw_logprobs'` | How returned inference log-probs are computed engine-wide. "raw_logprobs" (default) uses the unmodified model logits; "processed_logprobs" uses temperature a… | `:144` |
| `decode_only_cuda_graphs` | `bool` | `False` | Only use cuda graphs for decode-only steps, not prefill and mixed steps. | `:152` |
| `inference_dynamic_batching_enable_prefix_caching` | `bool` | `False` | Enable/disable prefix caching for dynamic batching inference. When disabled, KV cache blocks cannot be shared between requests with identical prompt prefixes. | `:169` |
| `inference_dynamic_batching_prefix_caching_eviction_policy` | `Literal['ref_zero', 'lru']` | `'ref_zero'` | Eviction policy for prefix caching blocks. "ref_zero" (default) immediately returns blocks to the free pool when ref_count hits 0. "lru" keeps blocks cached … | `:173` |
| `inference_dynamic_batching_prefix_caching_coordinator_policy` | `Literal['longest_prefix', 'first_prefix_block', 'round_robin']` | `'first_prefix_block'` | Coordinator routing policy for prefix caching. "first_prefix_block" (default) routes based on the first block hash only. "longest_prefix" routes to the rank … | `:178` |
| `inference_dynamic_batching_prefix_caching_routing_alpha` | `float` | `0.5` | Weight for prefix-aware routing score: score = alpha * match + (1 - alpha) * normalized_load. Higher alpha favors prefix cache hits; lower alpha favors load … | `:185` |
| `inference_dynamic_batching_prefix_caching_mamba_gb` | `float \| None` | `None` | GPU memory budget (in GB) for the Mamba state cache used by prefix caching on hybrid models. When set, Mamba states at block boundaries are cached for reuse. | `:189` |
| `inference_logging_step_interval` | `int` | `0` | Step interval for logging inference metrics. Default to 0 to disable inference logging. | `:195` |
| `inference_text_gen_server_logging` | `bool` | `False` | Enable per-request logging in the inference text generation server. | `:198` |
| `inference_wandb_logging` | `bool` | `False` | Enable inference wandb logging. | `:201` |
| `inference_coordinator_port` | `int \| None` | `None` | This port will be used to setup the inference coordinator on node-0. | `:206` |
| `inference_use_synchronous_zmq_collectives` | `bool` | `False` | Use synchronous ZMQ collectives for inference. Helps in reducing performance variability for MoEs. | `:209` |
| `inference_disable_ep_consensus` | `bool` | `False` | Skip the EP-group consensus all-reduce in the inference engine control loop and step on local state only. Only safe when EP coordination is not required (e.g… | `:213` |
| `mamba_inference_conv_states_dtype` | `Literal['bf16', 'fp16', 'fp32']` | `'bf16'` | Dtype for the Mamba inference conv states tensor. | `:221` |
| `mamba_inference_ssm_states_dtype` | `Literal['bf16', 'fp16', 'fp32']` | `'bf16'` | Dtype for the Mamba inference SSM states tensor. | `:224` |
| `use_flashinfer_fused_rope` | `bool` | `False` | Use flashinfer's fused rope implementation. Mirrors `--use-flashinfer-fused-rope`. | `:239` |

> 该类共 44 个字段，本表收 37 项；其余 7 项已在别处归属：主要归 本页他处 4 项、[[23_megatron_precision_cudagraph_fusion_analysis]] 2 项、[[30_megatron_rl_posttraining_consistency_analysis]] 1 项（完整归属见 `docs/coverage/megatron-lm.yaml`）。

---

## 配置契约：推理侧的 core 字段

本页前一节给的是 `InferenceSetupConfig`（训练侧的装配参数）。本节补的是散在 `TransformerConfig` 里、但只在推理路径生效的字段——`inference_*` 一组与 `flash_decode`、`symmetric_ar_type`、`nccl_all_reduce_for_prefill`、`mlp_chunks_for_prefill`。

**两组字段分属不同 config 类不是历史遗留**：`InferenceSetupConfig` 描述「起一个推理会话要什么」，而这里几项描述「模型在推理时的行为差异」，后者必须和模型配置同生命周期——它们会影响 kernel 选择与通信路径，不能在会话级别改。

**下表直接取自 `megatron/core/transformer/transformer_config.py` 的类体**。




### `TransformerConfig`（`megatron/core/transformer/transformer_config.py`，7 项）

| 字段 | 类型 | 默认 | 契约 | 行 |
|---|---|---|---|---|
| `flash_decode` | `bool` | `False` | Use the optimized flash decoding kernel during inference. | `:1330` |
| `inference_sampling_seed` | `int` | `42` | Random seed to use for sampling during inference. | `:1349` |
| `symmetric_ar_type` | `Optional[Literal['two_shot', 'one_shot', 'multimem_all_reduce']]` | `None` | What type of symmetric all reduce to use. The default is None which is no use of symmetric memory. | `:1352` |
| `nccl_all_reduce_for_prefill` | `bool` | `False` | If True, use NCCL all-reduce kernels when symmetric all-reduce is enabled. | `:1357` |
| `inference_disable_triton_nvls_kernels` | `bool` | `False` | If true, disables the use of Triton NVLS kernels during inference. | `:1366` |
| `inference_moe_disable_fused_quant_kernels` | `bool` | `False` | When False (default), use fused kernels that combine permute/activation with MXFP8 quantization + swizzle into a single kernel launch. Only applies when fp8_… | `:1379` |
| `mlp_chunks_for_prefill` | `int` | `1` | The number of chunks along the sequence dimension to use for MLP computation during prefill. | `:1428` |

> 该类共 266 个字段，本表收 7 项；其余 259 项已在别处归属：主要归 [[10_megatron_model_structure_analysis]] 92 项、[[14_megatron_ep_analysis]] 38 项、[[23_megatron_precision_cudagraph_fusion_analysis]] 38 项、[[21_megatron_fusion_operators_analysis]] 26 项，另散见 20 页（完整归属见 `docs/coverage/megatron-lm.yaml`）。

## Related Pages

- [[10_megatron_model_structure_analysis]] — 提供推理引擎所装载模型与 attention 变体的结构前置。
- [[23_megatron_precision_cudagraph_fusion_analysis]] — 展开低精度与 CUDA Graph 对推理执行面的约束。
- [[30_megatron_rl_posttraining_consistency_analysis]] — 说明 rollout 推理与训练模型之间的 logprob、reshard 与 refit 边界。
- [[33_megatron_rl_runtime_analysis]] — 展开推理引擎在 RL rollout runtime 中的调用位置。
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]] — 返回全部 35 篇内容页的主题索引。

