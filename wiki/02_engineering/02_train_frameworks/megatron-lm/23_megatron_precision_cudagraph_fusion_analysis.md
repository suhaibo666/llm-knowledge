---
title: "Megatron-LM FP8 精度 · CUDA Graph · 算子融合 深度解析"
---

# Megatron-LM FP8 精度 · CUDA Graph · 算子融合 深度解析

> **源码基线**：`NVIDIA/Megatron-LM@71092579522a12522d9f323ae180c9825d01928a`（`dev`，2026-08-27）
> **重定基线**：2026-08-28 由 `ee3f1ffa2acd18131ab67cabab4cec45283512ab`（2026-05-19）推进，跨 578 个提交；本页全部 `path:line` 已在新基线下逐条重核。
> 核心文件:`megatron/core/fp8_utils.py`、`megatron/core/fp4_utils.py`、`megatron/core/enums.py`、`megatron/core/transformer/cuda_graphs.py`、`megatron/core/full_cuda_graph.py`、`fusions/`、`megatron/core/num_microbatches_calculator.py`
> 配套阅读:五份并行文档 + `18_megatron_recompute_analysis.md` + `16_megatron_distributed_optimizer_analysis.md`
> 定位:"第二层补遗"第③份。这三块是与并行轴正交的**性能基建** —— 不改变并行策略,而是在精度、内核调度、内核形态三个层面榨吞吐与显存。

---

## 0. 总览

| 主题 | 解决的瓶颈 | 一句话 |
|------|-----------|--------|
| **FP8/FP4 低精度** | 显存 + 算力 + 通信 | 用 8/4 bit 做 GEMM 与通信,三重收益 |
| **CUDA Graph** | CPU 内核启动开销 | 把一串 kernel 录成一张图,一次重放 |
| **算子融合** | 内核启动 + HBM 读写 | 多个小算子合成一个 kernel |

README(MoE)把 MoE 训练的瓶颈归为三堵墙:**显存墙、通信墙、计算效率墙**。本文这三块分别针对它们。

---

## 1. FP8 / FP4 低精度训练

### 1.1 动机

bf16 已是主流,但 Hopper/Blackwell 的 Tensor Core 对 **FP8** 还能再快一档,且 8 bit 比 16 bit 再省一半。MoE 大模型尤其吃这个 —— GEMM 多、激活大、EP 通信重。FP4 更激进(Blackwell)。

注意:**实际的 FP8 GEMM 内核在 TransformerEngine 里**,Megatron 侧 `megatron/core/fp8_utils.py` 负责**选 recipe、建量化上下文、管 FP8 张量**。

### 1.2 四种 FP8 recipe(`megatron/core/enums.py:12` `Fp8Recipe`)

```python
class Fp8Recipe(str, Enum):
    delayed   = "delayed"      # 延迟缩放:用 amax 历史窗口定 scale
    tensorwise= "tensorwise"   # 整张量一个 scale(per-tensor)
    blockwise = "blockwise"    # 分块缩放:激活 1×128、权重 128×128
    mxfp8     = "mxfp8"        # 微缩放:1×32 一组,E8M0 scale
    # 还有 custom
```

| recipe | 缩放粒度 | 平台 | 定位 |
|--------|---------|------|------|
| delayed | per-tensor + amax 历史 | Hopper | 早期方案,需维护 amax history;`recompute`/A2A-overlap 有兼容限制 |
| tensorwise | 整张量 | Hopper, Blackwell | 保守,初期试验 |
| **blockwise** | 1×128 / 128×128 | Hopper | **生产首选**,DeepSeek-V3 级别已验证 |
| mxfp8 | 1×32 微块 | Blackwell | GB200 原生硬件支持 |

缩放粒度越细(blockwise/mxfp8),越能容忍张量内数值动态范围差异,精度越稳。`get_fp8_recipe`(`megatron/core/fp8_utils.py:739`)按配置返回 recipe,`get_fp8_context` 把一段 GEMM 包进对应的量化上下文。`get_fp8_align_size`(`:343`)给出 mxfp8 需要的对齐尺寸(故有 `--moe-router-padding-for-fp8`)。

### 1.3 FP8 的三重收益

FP8 同时砸向三堵墙:

| 墙 | FP8 收益 | 机制 |
|----|---------|------|
| **显存** | 激活省 ~50% | 线性层输入存 FP8 而非 bf16;FP8 primary weight 免 bf16 拷贝 |
| **算力** | GEMM 更快 | Hopper/Blackwell 的 FP8 Tensor Core 比 bf16 快 |
| **通信** | EP dispatch 省 50% | token 以 FP8 做 all-to-all(`14_megatron_ep_analysis.md`);参数 all-gather FP8(`--fp8-param-gather`) |

### 1.4 与并行轴的交织

FP8 不是孤立特性,它**贯穿前面所有文档**:
- **TP**:`ColumnParallel`/`RowParallel` 的 GEMM 走 FP8;`megatron/core/fp8_utils.py` 有 `is_column_parallel_linear`/`is_row_parallel_linear` 判定。
- **EP**:dispatch 的 A2A 用 FP8,通信量砍半(`14_megatron_ep_analysis.md`);`combined_1f1b` 的 fp8 上下文(`15_megatron_pp_schedulers_analysis.md` 调度器⑤)。
- **DP/ZeRO**:`--fp8-param-gather` 让参数 all-gather 走 FP8(`quantize_param_shard`、`post_all_gather_processing`,`megatron/core/fp8_utils.py:659/674`)。
- **重计算**:fp8 下用 `te_checkpoint`(`18_megatron_recompute_analysis.md` §3.4);delayed scaling 与某些 selective 重计算互斥。
- **首尾层**:`is_first_last_bf16_layer`(`:698`)—— 首尾层常保留 bf16(对精度最敏感)。

FP4(`megatron/core/fp4_utils.py`、`Fp4Recipe`)同理,更激进,Blackwell 专属。

---

## 2. CUDA Graph

### 2.1 动机:CPU 内核启动开销

GPU 执行每个 kernel 前,CPU 要先"启动"它(launch)。当模型由**大量小 kernel**组成 —— 尤其细粒度 MoE(几百个小专家 GEMM、路由、置换)—— CPU 来不及把 kernel 一个个塞给 GPU,**GPU 在 kernel 之间出现空隙、干等 CPU**。Nsight 时间线上表现为 kernel 间的缝隙。

**CUDA Graph**:把一段固定的 kernel 序列**录制(capture)成一张图**,之后整张图**一次性重放(replay)**,绕过逐 kernel 的 CPU 启动 —— 消除启动开销、消除 CPU 抖动(配合 `--manual-gc` 更稳)。

### 2.2 三种实现(`--cuda-graph-impl`)

| impl | 粒度 | 实现 |
|------|------|------|
| `local` | 每层一张图 | MCore 自带的图管理器(`megatron/core/transformer/cuda_graphs.py`) |
| `transformer_engine` | 每层一张图 | 用 TE 的 `make_graphed_callables()` |
| `full_iteration` | 整个前向+反向一张图 | 整步录成单图,消除最多 |

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。CUDA Graph API 已重构(#4292，与基线里的孪生 PR #4293 同内容；当前 `dev@232c478d4` 已生效)。本表的 "三种实现" 论断**在重构后依然成立**——`cuda_graph_impl` 是一个 `Literal['none','local','transformer_engine','full_iteration']`(`megatron/core/transformer/transformer_config.py:1148`)，`full_iteration` 确为一个独立的 **impl** 值(而非旧版的某个 scope)。但旧的单一旋钮 `--cuda-graph-scope` 已被**拆成三个正交字段**，需补充说明:
> - **`--cuda-graph-impl`**：选实现/总粒度 —— `none`(eager) / `local` / `transformer_engine` / `full_iteration`。
> - **`--cuda-graph-modules`**(由 `--cuda-graph-scope` 改名，`megatron/core/transformer/enums.py:CudaGraphModule`，`megatron/core/transformer/cuda_graph_config.py`)：在 `local` / `transformer_engine` 的**逐层图内部**选**捕获哪些子区域** —— `attn` / `mlp` / `moe` / `moe_router` / `moe_preprocess` / `mamba`；**留空 = 整层捕获**。`full_iteration` 下此字段**必须为空**。
> - **`--inference-cuda-graph-scope`**(`megatron/core/transformer/enums.py:InferenceCudaGraphScope`)：推理图的归属边界 —— `none`(eager) / `layer`(TransformerLayer/MambaLayer 边界) / `block`(TransformerBlock/HybridBlock 边界)。`local` 默认 `layer`，其它 impl 默认 `none`(`ALLOWED_INFERENCE_SCOPES`)。
> - 兼容迁移：旧值 `full_iteration` → `--cuda-graph-impl=full_iteration`；旧值 `full_iteration_inference` → `--inference-cuda-graph-scope=block`；`full` → 空 modules(整层)。`--cuda-graph-scope` 与旧 `CudaGraphScope` 枚举仅为旧 checkpoint 反序列化保留(`megatron/training/arguments.py`/`transformer_config.__post_init__` 自动迁移并告警)。
> - 训练校验也随之改写：`--cuda-graph-impl=full_iteration` 要求 `--no-check-for-nan-in-loss-and-grad`；`--inference-cuda-graph-scope=block` + fp8 仅支持 `--transformer-impl=inference_optimized` 且 `--fp8-recipe=mxfp8`(`megatron/training/arguments.py:validate_args`)。

`megatron/core/transformer/cuda_graphs.py` 的机制:`_CudaGraphRunner` 包住一个可图化的模块;`_CudagraphGlobalRecord`(`:345`)记录所有 runner 的创建顺序;`create_cudagraphs`(`:497`)在**第一个训练步**真正录图(被 `15_megatron_pp_schedulers_analysis.md` 的 `megatron/core/pipeline_parallel/schedules.py` 在步末调用 —— 前五份文档里 `megatron/core/pipeline_parallel/schedules.py` 出现的 `create_cudagraphs()` 就是它)。`TensorReusePool`(`:161`)在图之间复用张量缓冲。

> [!deprecated] 该机制在基线 `71092579` 下已不存在（`TensorReusePool` 与 `_determine_if_first_last_layer_of_this_vp_chunk` 在 `71092579` 全仓库零命中，由 #5451 *Optimize memory usage of partial CUDA graphs* 删除），以上关于 `TensorReusePool` 与 VP-chunk 判定的描述对应旧基线 `ee3f1ff`。
> - **`TensorReusePool` 已删除**，图间缓冲复用改为**引用计数**方案：`CudagraphBufferMetadata`(`megatron/core/transformer/cuda_graphs.py:146`)的 `cudagraph_reuse_ref_count` / `capture_reuse_count`(`:156`/`:157`)配合模块级 `fwd_buffer_reuse_ref_count` / `bwd_buffer_reuse_ref_count`(`:341`/`:342`)判定某个 buffer 能否被后一张图直接写入或复用。"在图之间复用张量缓冲" 这一**意图**仍成立，但已无 Pool 这个对象。
> - **`_determine_if_first_last_layer_of_this_vp_chunk` 已删除**，改由 `annotate_first_last_layer(layers)`(`:247`)在建块时直接给每层打 `is_first_layer` / `is_last_layer` 标记(`megatron/core/transformer/transformer_block.py:389`、`megatron/core/models/hybrid/hybrid_block.py:766`)，`_CudaGraphRunner` 从 base module 上读取(`:721`)。
> - 本页其余 `megatron/core/transformer/cuda_graphs.py` 符号在新基线下仍在，位置为：`_ensure_generator_state_is_cudagraph_safe`→`:269`、`_CudagraphGlobalRecord`→`:345`(其 `create_cudagraphs` classmethod 在 `:370`)、模块级 `create_cudagraphs()`→`:497`、`_CudaGraphRunner`→`:674`。

### 2.3 约束:动态形状

CUDA Graph 要求**每次重放的张量形状、地址固定**。问题:
- dropless MoE 的每个专家收到的 token 数**随路由动态变化** → MoE 层形状不定 → **无法图化**。
- 解法:① 设 `--moe-expert-capacity-factor` + `--moe-pad-expert-input-to-capacity` 让 MoE 形状静态(代价:丢/填 token);② 或只图化 attention(`--cuda-graph-modules attn`),MoE 层不动。

`megatron/core/transformer/cuda_graphs.py` 还要处理 RNG 状态(`_ensure_generator_state_is_cudagraph_safe`,`:269`)—— dropout 的随机数生成器在图重放下必须可控。与 VPP 配合时还要判定层属于哪个 VP chunk(`_determine_if_first_last_layer_of_this_vp_chunk`,`:249`)。

> [!deprecated] 该机制在基线 `71092579` 下已不存在（`_determine_if_first_last_layer_of_this_vp_chunk` 全仓库零命中，由 #5451 删除），上句描述对应旧基线 `ee3f1ff`。新基线改为 `annotate_first_last_layer(layers)`(`megatron/core/transformer/cuda_graphs.py:247`)在建块时直接标注 `is_first_layer` / `is_last_layer`。

推理另有 `--inference-cuda-graph-scope=layer|block`。

---

## 3. 算子融合(`fusions/`)

### 3.1 动机:内核启动 + HBM 往返

两个相邻算子(如 `bias add` 后接 `GeLU`)若各是一个 kernel:① 两次启动开销;② 中间结果要**写回 HBM 再读出**(显存带宽是稀缺资源)。**融合**把它们写成**一个 kernel**:中间结果只在寄存器/共享内存里流转,不落 HBM,启动也只一次。

收益:更少 kernel 启动 + 更少 HBM 流量 → 直接提速,尤其对 memory-bound 的逐元素算子。

### 3.2 `fusions/` 清单

| 融合 kernel | 融合了什么 |
|------------|-----------|
| `fused_bias_gelu` / `fused_bias_geglu` / `fused_bias_swiglu` | bias 加法 + 激活函数(GeLU/GeGLU/SwiGLU) |
| `fused_bias_dropout` | bias + dropout(+ 残差) |
| `fused_softmax` | scale + mask + softmax(attention) |
| `fused_layer_norm` | LayerNorm / RMSNorm |
| `fused_cross_entropy` / `fused_linear_cross_entropy` | 交叉熵(后者连输出投影一起融) |
| `fused_pad_routing_map` | MoE 路由图的 FP8 对齐填充 |
| `fused_indices_converter` | MoE 路由索引转换 |
| `fused_mla_yarn_rope_apply` | MLA 的 YaRN RoPE 应用 |
| `fused_mhc_kernels` | hyper connections |
| `fused_weighted_squared_relu` | 加权 squared-ReLU 激活 |

### 3.3 MoE 专用融合(README)

MoE 是"小算子最多"的地方,有三个关键融合开关:
- `--moe-grouped-gemm`:把 `E/e` 个专家的 GEMM 批成**一次 grouped GEMM**(`14_megatron_ep_analysis.md` §2.4)。
- `--moe-router-fusion`:路由投影 + top-k + softmax + aux loss 融成少数 kernel。
- `--moe-permute-fusion`:token 置换/反置换融合。

它们直接对应 README"计算效率墙"的解法。

---

## 4. 附:`num_microbatches_calculator`

`megatron/core/num_microbatches_calculator.py`:由 `global_batch_size`、`micro_batch_size`、`data_parallel_size` 算出每步的 microbatch 数:

```
num_microbatches = global_batch_size / (micro_batch_size · data_parallel_size)
```

这个 `num_microbatches` 正是 `15_megatron_pp_schedulers_analysis.md` 里反复出现的 `m`(梯度累加步数 / 流水线 microbatch 数)。它还支持 **batch size ramp-up**(训练初期用小 global batch、逐步增大)。是连接"数据并行配置"与"PP 调度"的小齿轮。

---

## 5. 小结

- **FP8/FP4**:用低精度做 GEMM 与通信,**三重收益**(显存 ~50%、算力更快、EP/DP 通信砍半);4 种 recipe(delayed/tensorwise/blockwise/mxfp8),生产首选 **blockwise**(Hopper)/ **mxfp8**(Blackwell);实际内核在 TE,Megatron 管 recipe 与上下文;**贯穿所有并行轴**。
- **CUDA Graph**:把 kernel 序列录成图、一次重放,消除 CPU 启动开销与抖动;3 种粒度(local / transformer_engine / full_iteration);**动态形状是死穴** —— dropless MoE 需固定容量或只图化 attention。
- **算子融合**:多算子合成一 kernel,省启动 + 省 HBM 往返;`fusions/` 一堆逐元素/归一化/交叉熵融合;MoE 三件套 `grouped-gemm` / `router-fusion` / `permute-fusion`。
- 三者都与并行轴**正交**,是 kernel/精度层面的提速,叠加在并行策略之上。

至此"第二层补遗"3 份文档全部完成:① 激活重计算、② 优化器内部、③ FP8 精度 + CUDA Graph + 算子融合。

---

## 6. 增量更新（ee3f1ff → dev@232c478d4）

> 基线 `ee3f1ff`(2026-05-19) 之后、当前 `dev@232c478d4`(2026-06-16) 的精度/CUDA-Graph 新增机制与勘误。§1–§4 原论断在新基线 `71092579` 下经重核仍成立，行号已全部推进(`megatron/core/enums.py` 的 `Fp8Recipe` 仍在 `megatron/core/enums.py:12`；`get_fp8_align_size`@`megatron/core/fp8_utils.py:343`、`quantize_param_shard`@`:659`、`post_all_gather_processing`@`:674`、`is_first_last_bf16_layer`@`:698`；`get_fp8_recipe` 由 `:536` 位移到 `:739`(`HAVE_TE` 分支)/`:885`(无 TE 的桩函数))。

### 6.1 MXFP8 LM-head 输出投影（opt-in）

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。
> 新增 `fp8_output_proj`(`megatron/core/transformer/transformer_config.py:705`，#4825)：把**词表输出投影(LM head)**也放进 MXFP8 autocast 跑。原本 §1.5 提过"首尾层常保留 bf16"，本特性是其**反向 opt-in** —— 显式让最后的 output projection 走 MXFP8。
> - 仅当 `fp8=True` 且 `fp8_recipe='mxfp8'` 时生效(否则构造期报错)。
> - 实现：`GPTModel` 的 `output_layer` 在 `is_mxfp8_output_proj_active(config)`(`megatron/core/fp8_utils.py:717`)为真时换成 `TELMHeadColumnParallelLinear`(`megatron/core/extensions/transformer_engine.py:1350`)而非普通 `ColumnParallelLinear`(`megatron/core/models/gpt/gpt_model.py`)。

### 6.2 MXFP8/FP4 param-gather 一连串修复 + 新旋钮

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。
> §1.3/§1.4 提到的 `--fp8-param-gather`(参数 all-gather 走 FP8)在 mxfp8/nvfp4 下有多处数值与流程修复：
> - **`reuse_grad_buf_for_mxfp8_param_ag`**(`megatron/core/optimizer/optimizer_config.py:180`，#4994/#4800)：复用 grad buffer 做 mxfp8 参数 all-gather。新增 `MegatronOptimizer.prepare_model_params_for_param_sync()`(`megatron/core/optimizer/optimizer.py`)，`ChainedOptimizer` 重写它，在显式 DDP param-sync 前每个 model chunk **只 stage 一次**(`zero_grad_buffer` + `_copy_main_params_to_param_buffer`)；并禁止与 `overlap_param_gather_with_optimizer_step` 同用。修了 "DP overlap 关闭时 mxfp8 param gather 数值错误"。
> - **eval 期强制 param-AG 后的后处理**(#4562)：把 quantize/transpose 等 `post_all_gather_processing` 从 DDP 内联挪到 `megatron/core/distributed/param_and_grad_buffer.py`，确保 eval 里强制全量 all-gather 后参数的 FP8/FP4 量化态正确(`megatron/core/distributed/param_and_grad_buffer.py`、`megatron/training/training.py`)。
> - **FP4 param gather 适配 NVFP4 混精**(#4358，`megatron/core/extensions/transformer_engine.py`、`megatron/core/quantization/utils.py`)：让 `--fp4-param-gather` 在 NVFP4 recipe 的**混合精度**(部分层 fp4、部分 fp8/bf16)下工作；`get_quant_config_or_none` 容忍 `module_path=None`。覆盖 attention/MLP/MoE/MLA/MTP/SSM 多处 quant 配置接线。
> - **Megatron-FSDP MXFP8 转置权重缓冲**(#4852，`megatron/core/distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py`)：持久化转置权重 buffer 的**非对称单元(asymmetrical units)**，修 FSDP 下 mxfp8 转置权重的分片/持久化。

### 6.3 推理 CUDA Graph 覆盖：max_requests → max_tokens

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。
> 新增 `cuda_graph_all_prefills` / `--inference-cuda-graph-all-prefills`(`megatron/core/inference/config.py:245`，#4214)。原先 prefill/mixed 推理图的 batch 上界受 `max_requests*(num_speculative_tokens+1)` 约束；开启后 prefill/mixed 图捕获**扩展到覆盖整个 `max_tokens` 预算**(decode-only 图仍按旧上界)。同时删除旧的 `--inference-dynamic-batching-cuda-graph-max-tokens`(默认 16384)旋钮，改由上述 token 预算逻辑推导。这呼应 §2 "动态形状是死穴" —— 通过把图覆盖按 **token 数**(而非请求数)分档，让变长 prefill 也能命中图。

### 6.4 与融合页的交叉补充

> [!update] 该特性自 `dev@232c478d4`（2026-06-16）引入，行号已重核至基线 `71092579`。
> §3 算子融合的增量(TE op-fuser 把 grouped MLP 的 GEMM+激活+GEMM 整链融合、ScaledSReLU/Clamped-SwiGLU、`TEFusedDenseMLP` 在 SM100+/MXFP8 触发 CuTe GEMM-SwiGLU 融合、mHC 多后端重写、DSv4 稀疏注意力融合 kernel、TE 版本依赖)详见 [[21_megatron_fusion_operators_analysis]] §7。这些融合与本页 FP8/MXFP8 精度强相关(多数融合 kernel 的收益正建立在 MXFP8 量化 epilogue 上)。

---

*生成依据:`Megatron-LM` `dev` 分支 `71092579`。源码行号以该 commit 为准。FP8/FP4 的 GEMM 内核位于 TransformerEngine。配套文档:五份并行分析 + `18_megatron_recompute_analysis.md` + `16_megatron_distributed_optimizer_analysis.md`。*

## Related Pages

- [[14_megatron_ep_analysis]] · [[18_megatron_recompute_analysis]] · [[16_megatron_distributed_optimizer_analysis]]
- [[21_megatron_fusion_operators_analysis]]
- [[10_pytorch_cuda_graphs_complete_guide]] — CUDA Graph 通用机制权威页（capture/replay、地址不变式、失败与回退；本页 §2 是训练框架侧的具体应用）
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]]
