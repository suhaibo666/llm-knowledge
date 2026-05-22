# Megatron-LM 推理引擎深度解析(Inference Engine)

> 代码基准:`Megatron-LM/` 子仓库 `dev` 分支,commit `ee3f1ff`
> 核心文件:`megatron/core/inference/` 下 `engines/`(`dynamic_engine.py` 2446 行、`static_engine.py`)、`contexts/dynamic_context.py`(3785 行)、`contexts/kv_block_allocator.py`、`scheduler.py`
> 配套阅读:`rl_posttraining_consistency_analysis.md`(RL rollout 用的就是本引擎)、`precision_cudagraph_fusion_analysis.md`、`ep_analysis.md`
> 定位:系统性专题。`rl_posttraining_consistency_analysis.md` 把推理引擎当作 RL rollout 的积木一笔带过,本文拆开它内部。

---

## 0. 总览

推理(自回归生成)与训练是**两种不同的负载**:训练是一次大前向+反向、batch 固定、要可微;推理是**逐 token 生成**、请求动态来去、不需反向、要低延迟高吞吐。所以 Megatron 有一套独立的推理引擎(`megatron/core/inference/`),配 `transformer_impl='inference_optimized'`(见 `rl_posttraining_consistency_analysis.md` §4)。

两个引擎变体,本文当作"调度器/dispatcher"那样逐个解读:

| 引擎 | 代码 | 批处理 | KV cache | 用途 |
|------|------|--------|----------|------|
| **Static** | `static_engine.py`(403 行) | 定长批,同进同出 | 简单 | 离线批量推理 |
| **Dynamic** | `dynamic_engine.py`(2446 行) | **连续批处理**(in-flight) | **块级**(paged 式) | 在线服务、RL rollout |

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

`static_engine.py`。一批请求**同时开始、同时结束**:凑齐一个固定大小的 batch,prefill,然后一起 decode,直到**最长的那条**生成完,整批才返回。

- 优点:实现简单,无动态调度。
- 缺点:**短请求空等长请求** —— batch 里某条 20 token 就停了,但要陪最长的 2000 token 跑完,那段 GPU 槽位全空转。
- 适用:离线批量推理(所有 prompt 已知、长度相近、不在乎单条延迟)。

---

## 4. 引擎变体② DynamicInferenceEngine —— 连续批处理

`dynamic_engine.py`。类注释点明:"in-flight batching and a dynamic block-level KV cache (similar to paged attention)"。

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

`contexts/dynamic_context.py`(`DynamicInferenceContext`)+ `kv_block_allocator.py`(`KVBlockAllocator`)。

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

显存有限,块会用完。`DynamicInferenceContext` 定义了一组 `ContextOverflowError`(`dynamic_context.py:106`):`RequestOverflowError`(请求数超)、`TokenOverflowError`、`BlockOverflowError`(块用尽)、`MaxSequenceLengthOverflowError`。引擎据此**背压** —— 新请求进不来就留在等待队列,而不是 OOM 崩溃。

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
- **对 RL rollout 收益巨大**:GRPO 对同一 prompt 采样多条 response(`ep_analysis.md` / `rl_posttraining_consistency_analysis.md`),prompt 前缀的 KV 算一次共享给所有 rollout;聊天服务里共享的 system prompt 同理。

---

## 7. CUDA Graph 在推理里

decode step 是"逐 token、kernel 小而多",CPU 启动开销占比极高 —— 正是 CUDA Graph 的主场(见 `precision_cudagraph_fusion_analysis.md` §2)。

但 CUDA Graph 要求形状固定,而连续批处理的"当前 batch 里有几个请求"是变化的。`DynamicInferenceEngine` 的解法(`create_cuda_graphs`,`dynamic_engine.py:325`):
- 预先枚举一组**典型的 batch 维度**(`cuda_graph_batch_dimensions_list`,不同 request count)。
- **为每种 batch size 各捕获一张图**,warmup 阶段录完。
- 运行时按当前实际请求数,**选最接近的那张图**重放(不足部分 padding)。
- `inference_cuda_graph_scope`(`layer` / `block`)控制图化粒度;`use_cuda_graphs_for_non_decode_steps` 决定 prefill 步是否也图化;MTP 另有独立的图 warmup。

---

## 8. Scheduler 与请求池

`scheduler.py`。`Scheduler` 维护几个请求池:

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
| `sampling/` + `sampling_params.py` | greedy / top-k / top-p / temperature;`return_log_probs`(RL 要的 logprob) |
| `text_generation_server/` | HTTP 服务,OpenAI 兼容的 `chat/completions` 端点 |
| `data_parallel_inference_coordinator.py` | 多 DP rank 推理的请求协调 |
| `unified_memory.py` | KV cache 用统一内存,引擎 `suspend`/`resume` 时把 KV 换出(RL 里训练相不需要推理引擎时腾显存) |
| `async_stream.py` / `async_zmq_communicator.py` | 异步、流式、ZMQ 通信 |

`suspend`/`resume`(`dynamic_engine.py:719/768`)对 **RL collocated 部署**很关键:训练相和推理相在同一批卡上轮流跑,推理引擎 suspend 时删 CUDA Graph、把 KV cache 换出统一内存,让出显存给训练;resume 时再恢复(见 `rl_posttraining_consistency_analysis.md` §7)。

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

*生成依据:`Megatron-LM` `dev` 分支 `ee3f1ff`。源码行号以该 commit 为准。配套文档:`rl_posttraining_consistency_analysis.md`、`precision_cudagraph_fusion_analysis.md`、`ep_analysis.md`。*

## Related Pages

- [[rl_posttraining_consistency_analysis]] · [[precision_cudagraph_fusion_analysis]] · [[ep_analysis]] · [[model_structure_analysis]]
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]]
