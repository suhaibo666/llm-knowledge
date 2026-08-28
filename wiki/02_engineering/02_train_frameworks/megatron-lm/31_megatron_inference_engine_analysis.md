---
title: "Megatron-LM 推理引擎深度解析(Inference Engine)"
---

# Megatron-LM 推理引擎深度解析(Inference Engine)

> **源码基线**:`NVIDIA/Megatron-LM@71092579522a12522d9f323ae180c9825d01928a`(`dev`,2026-08-27)
> **重定基线**:2026-08-28 由 `ee3f1ffa…`(2026-05-19)推进,跨 578 个提交;本页全部 `path:line` 已在新基线下逐条重核。
> 核心文件:`megatron/core/inference/` 下 `engines/`(`megatron/core/inference/engines/dynamic_engine.py` 2614 行、`megatron/core/inference/engines/static_engine.py`)、`megatron/core/inference/contexts/dynamic_context.py`(4021 行)、`megatron/core/inference/contexts/kv_block_allocator.py`、`megatron/core/inference/scheduler.py`
> 配套阅读:`30_megatron_rl_posttraining_consistency_analysis.md`(RL rollout 用的就是本引擎)、`23_megatron_precision_cudagraph_fusion_analysis.md`、`14_megatron_ep_analysis.md`
> 定位:系统性专题。`30_megatron_rl_posttraining_consistency_analysis.md` 把推理引擎当作 RL rollout 的积木一笔带过,本文拆开它内部。

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

## 1. 动机:自回归推理为何需要专门的引擎

一次生成 = 给定 prompt,逐个吐 token,每个新 token 都要 attend 到**之前所有 token**。朴素做法每步把整条序列重新前向 —— `O(S²)` 重复计算。于是有 **KV cache**:把每个 token 算过的 Key/Value 存下来,新 token 只算自己的、复用缓存。

但 KV cache 带来三个新问题,正是推理引擎要解决的:

1. **两相负载迥异**:处理 prompt(prefill)和逐 token 生成(decode)的计算特征完全不同(§2)。
2. **显存碎片**:每条序列的 KV cache 长度不同、还在增长,预分配定长连续缓冲极浪费(§5)。
3. **批利用率低**:不同请求长度不一、来去时间不一,定长批里短请求早早算完却要空等最长的(§4)。

---

## 2. Prefill 与 Decode:两相,两种瓶颈

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

decode step 是"逐 token、kernel 小而多",CPU 启动开销占比极高 —— 正是 CUDA Graph 的主场(见 `23_megatron_precision_cudagraph_fusion_analysis.md` §2)。

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

## 10. 小结

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

*生成依据:`Megatron-LM` `dev` 分支 `71092579`(2026-08-27;由 `ee3f1ff` 重定基线而来)。源码行号以该 commit 为准。配套文档:`30_megatron_rl_posttraining_consistency_analysis.md`、`23_megatron_precision_cudagraph_fusion_analysis.md`、`14_megatron_ep_analysis.md`。*

## Related Pages

- [[30_megatron_rl_posttraining_consistency_analysis]] · [[23_megatron_precision_cudagraph_fusion_analysis]] · [[14_megatron_ep_analysis]] · [[10_megatron_model_structure_analysis]]
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]]
