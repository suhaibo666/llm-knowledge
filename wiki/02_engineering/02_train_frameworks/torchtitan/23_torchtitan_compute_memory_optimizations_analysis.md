# 计算与显存优化：量化、显式融合、编译、CUDA Graph 与训练态内存

> **代码基准**：pytorch/torchtitan `main` @ `a3168782c9a3a2e40afbd0de114818b96e2bda6e`（2026-08-27）
> **最后更新**：2026-08-27 · **系列**：[[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]]
>
> **主线**：TorchTitan 的计算/显存优化不是一个总开关，而是四个接线面：建模前改写配置树，模块层选择量化或显式融合实现，区域编译决定 compiler 边界，训练步运行时再由 CUDA Graph、优化器、Chunked Loss 与 FSDP offload 管理 kernel 和存活张量。能否叠加，取决于这些接线面是否仍保持同一参数布局、SPMD 契约与固定输入形状。
>
> 主要源文件：`torchtitan/components/quantization/`、`torchtitan/models/common/dist_gemm.py`、`torchtitan/overrides/`、`torchtitan/distributed/{compile,cudagraph}.py`、`torchtitan/components/{loss,optimizer}/`。

---

## 1. 先纠正旧版知识

旧基线 `61c010fcb` 的总体判断——“TorchTitan 主要负责接线，底层计算交给 PyTorch/TorchAO/Inductor”——仍然成立，但以下具体断言已经失效：

1. **低精度不再只有 Float8 与 MXFP8。** 当前还有 NVFP4 converter；它保留 BF16 权重，在前向、反向中即时量化激活、权重和梯度，并且只支持 NVIDIA SM100+（`torchtitan/components/quantization/nvfp4.py:7-16`、`torchtitan/components/quantization/nvfp4.py:267-297`）。
2. **FusedSwiGLU 不再用 `einsum`。** 当前 `w13` 是标准 `Linear`，一次 GEMM 后拆 gate/up，再调用自定义 Triton `silu_and_mul`（`torchtitan/overrides/fused_swiglu.py:450-475`）。
3. **“TorchTitan 唯一自写 Triton kernel 在 MinimalAsyncEP”已经错误。** 当前还有 fused SwiGLU、FusedGroupedExperts、DeepSeek MLA 的 Triton custom op，以及可选 Helion RoPE（`torchtitan/overrides/fused_swiglu.py:20-29`、`torchtitan/overrides/fused_mla.py:895-900`、`torchtitan/overrides/helion_rope.py:1028-1049`）。
4. **优化器文件已迁移为 package。** 入口现在是 `components/optimizer/optimizer.py`，package 同时导出 scheduler/container/helper（`torchtitan/components/optimizer/__init__.py:7-21`）。
5. **核心 Trainer 已内置 CUDA Graph，且默认未禁用。** 这条 eager forward+backward capture 路径独立于 `torch.compile(mode="reduce-overhead")` 的内部 graph capture（`torchtitan/config/configs.py:77-86`）。

> [!deprecated] 旧 FusedSwiGLU 与 CUDA Graph 描述
> 旧页的 `einsum("...d,hgd->...hg")` 前向与“核心 Trainer 没有 CUDA Graph”不能再作为当前行为。以下章节只描述新实现。

---

## 2. 优化发生在哪一层

| 层 | 当前入口 | 改变什么 | 不自动改变什么 |
|---|---|---|---|
| 配置树改写 | Float8/MXFP8/NVFP4 converter | 把匹配的 `Linear.Config` 或 expert config 换成量化子类 | 不自动改变 TP/FSDP 布局 |
| 模块实现选择 | FusedQKV、fused SwiGLU、fused MLA、dist-GEMM | 参数组织、kernel 边界、collective 与 GEMM 的相邻关系 | 不等于编译器自动发现这些结构 |
| 区域编译 | `CompileConfig` + `apply_compile` | 逐 TransformerBlock 编译；loss 可独立编译 | 不等于核心 CUDA Graph wrapper |
| 训练步运行时 | CUDA Graph、optimizer、Chunked Loss、CPU offload | launch 开销、优化器状态、logits 峰值、设备驻留 | 不改变模型数学定义 |

量化 converter 在 `model_registry` 中按用户顺序执行，返回已改写的 `ModelSpec`；Trainer 随后才在 meta device 上 build 模型（`torchtitan/models/llama3/__init__.py:369-389`、`torchtitan/trainer.py:350-365`）。量化必须排在 LoRA converter 前，否则配置校验直接报错（`torchtitan/models/utils.py:27-46`）。

这解释了为什么 TorchTitan 的“优化开关”经常表现为**配置类型变化**而非模型建好后的 monkey patch：参数布局和 sharding 可以在 build/parallelize 阶段一起确定。

---

## 3. 低精度训练：Float8、MXFP8 与 NVFP4

### 3.1 共同接线：替换叶子，保留并行骨架

`QuantizationConverter` 是 `ModelConfigConverter` 的子类，并显式携带“model 是否会被 compile”的信息；源码同时把 compile 作为量化取得竞争性性能的预期运行方式（`torchtitan/components/quantization/__init__.py:13-31`）。`ShardingConfig` 本身不携带 dtype，量化与参数/激活 placement 被设计成正交关系（`torchtitan/protocols/sharding.py:59-75`）。

因此，低精度的主要收益是改变本地 GEMM 的数值格式和 kernel；不能仅凭“Linear 已量化”推导出 TP/FSDP wire format 也同步变成 FP8/FP4。NVFP4 更明确地规定：stock TP all-gather/reduce-scatter 仍传 BF16，不传 FP4 code（`torchtitan/components/quantization/nvfp4.py:9-16`）。

### 3.2 Float8：成熟范围最宽，但 converter 仍是 rowwise-only

当前 Float8 Linear 暴露 `rowwise` 与 `rowwise_with_gw_hp` 两个配方；硬件门槛是 NVIDIA SM89+ 或 AMD gfx942+，旧卡只能用 eager emulation 做测试，而 emulation 与 compile 不兼容（`torchtitan/components/quantization/float8.py:53-96`）。converter 从 TorchAO 解析配方，并为 `rowwise` 打开 Inductor 的 precision-cast 兼容开关（`torchtitan/components/quantization/float8.py:98-123`）。

转换范围由两层过滤决定：Linear 的输入/输出维度必须都是 16 的倍数，且 FQN 不得命中过滤表；`auto_filter_small_kn` 还可以交给 TorchAO 按 GEMM 尺寸避开不划算的小矩阵（`torchtitan/components/quantization/utils.py:16-30`、`torchtitan/components/quantization/float8.py:125-170`）。

MoE 不是走 dense Linear converter，而是覆写 `GroupedExperts._grouped_mm`，调用 TorchAO 的“量化后 scaled grouped MM”；同时要求 dispatcher 能把每个专家 token 数 padding 到 16 的倍数（`torchtitan/components/quantization/float8.py:176-220`、`torchtitan/components/quantization/float8.py:246-265`）。

### 3.3 MXFP8：SM100、块缩放、显式 padding 合同

MXFP8 dense converter 是 `Linear.Config -> MXFP8Linear.Config` 的纯叶子替换，要求 SM100+；未 compile 只会警告，不会拒绝运行（`torchtitan/components/quantization/mx.py:50-98`）。MoE 路径当前只暴露 `mxfp8_rceil`，默认每专家 padding 到 32；CuTeDSL kernel 在 SM100 上可能要求 128，配置把这个倍数留给调用方（`torchtitan/components/quantization/mx.py:149-187`）。

量化 MoE 只能接支持 padding 的 `TorchAOTokenDispatcher` 或 `HybridEPTokenDispatcher`；其它 dispatcher 会在 converter 阶段报错，而不是运行时静默换实现（`torchtitan/components/quantization/utils.py:33-64`）。

### 3.4 NVFP4：BF16 主权重 + 即时 FP4 GEMM

NVFP4 与“把 checkpoint 永久压成 FP4”不同。`NVFP4Linear` 继承 TorchAO training module，权重仍是 BF16；前向调用 `nvfp4_linear`，将输入、权重和反向相关量即时量化（`torchtitan/components/quantization/nvfp4.py:84-92`、`torchtitan/components/quantization/nvfp4.py:234-241`）。所以它主要降低 GEMM 计算格式，**不能据此声称持久参数、优化器状态或 TP 通信缩成 4 bit**。

它的并行接线比 MXFP8 更显式：`Config.build` 读取 stock Linear 的 colwise/rowwise `ShardingConfig`，把量化 op 包进 `LocalMapConfig`；rowwise 输入/输入梯度保持最后一维 shard，colwise 输入保持 replicate、输入梯度标成 partial（`torchtitan/components/quantization/nvfp4.py:113-157`）。随机舍入 seed 是每 rank 本地状态，而 Hadamard sign vector 固定且各 TP rank 相同，不需要广播（`torchtitan/components/quantization/nvfp4.py:42-47`、`torchtitan/components/quantization/nvfp4.py:210-232`）。

边界是硬约束：

- 全局 `in_features`/`out_features` 必须是 128 的倍数；TP 切分后的本地 GEMM 维度仍由 TorchAO kernel 再校验（`torchtitan/components/quantization/nvfp4.py:94-111`）。
- 只支持 CUDA SM100+，TorchAO 必须包含 NVFP4 training prototype；compile 是性能建议而非构造前提（`torchtitan/components/quantization/nvfp4.py:267-297`）。
- 空 `fqns` 表示“转换所有 Linear”，不是“不转换”；因此真实 Llama 配方显式排除 `lm_head`，并可保留最后 15% decoder layers 为 BF16 tail（`torchtitan/components/quantization/nvfp4.py:247-277`、`torchtitan/models/llama3/config_registry.py:227-247`）。

---

## 4. 当前显式融合算子

### 4.1 stock 路径已经包含的融合边界

- `FusedQKVLinear` 用一个 `wqkv` Linear 生成 Q/K/V，再 reshape/split；它通过 state-dict hooks 继续保存 stock `wq/wk/wv` 键（`torchtitan/models/common/attention.py:737-768`、`torchtitan/models/common/attention.py:779-807`）。
- dense attention 的 SDPA 路径显式按 cuDNN、Flash、Math 顺序选择 backend；Varlen 路径调用 PyTorch 的 varlen attention op（`torchtitan/models/common/attention.py:397-440`、`torchtitan/models/common/attention.py:195-219`）。
- `GroupedExperts` 不是逐专家 Python 循环，而是三次 `torch._grouped_mm`，以累计 token offsets 划分每个专家的行区间；`_grouped_mm` 同时是 Float8/MXFP8 替换 scaled grouped MM 的稳定 seam（`torchtitan/models/common/moe.py:55-120`）。
- `RMSNorm` 直接复用 `nn.RMSNorm`；TorchTitan 没有为 stock norm 维护另一份 Triton 实现（`torchtitan/models/common/nn_modules.py:134-148`）。

### 4.2 FusedSwiGLU：一份标准 Linear 权重 + 自定义激活 kernel

当前 `FusedSwiGLU` 的 `w13.weight` 物理形状是 `(2 * hidden_dim, dim)`，gate/up 行交错。一次标准 Linear 之后把末维还原成 `(hidden_dim, 2)`，再由 `torchtitan::silu_and_mul` 完成 pointwise 融合（`torchtitan/overrides/fused_swiglu.py:33-37`、`torchtitan/overrides/fused_swiglu.py:450-475`）。保留标准 Linear 很重要：量化 converter 仍能替换 `w13`，而 TP `Shard(0)` 也会给每 rank 分到匹配的 gate/up 行。

checkpoint 仍暴露 stock `w1.weight`/`w3.weight`：保存时拆 `w13`，加载时重新交错合并，因此启用 override 不要求转换旧 checkpoint（`torchtitan/overrides/fused_swiglu.py:477-504`）。

MoE 的 `FusedGroupedExperts` 采用相同思路：一次 grouped GEMM 同时算 gate/up，按 offsets 跳过 padding 的无效行，再接一次 down-projection grouped GEMM；EP shard 专家轴、TP shard hidden 轴，显式大小为 2 的 gate/up 轴不分片（`torchtitan/overrides/fused_swiglu.py:600-658`、`torchtitan/overrides/fused_swiglu.py:692-703`）。

### 4.3 其它 opt-in override：不要误认为默认启用

- Helion RoPE 是 `CosSinRoPE`/`ComplexRoPE` 的 exact override；不支持的输入回退 PyTorch 实现，显式请求但未安装 Helion 会报 `ImportError`（`torchtitan/overrides/helion_rope.py:972-1025`、`torchtitan/overrides/helion_rope.py:1028-1049`）。
- Fused MLA 只针对 DeepSeek 风格 `Attention.Config`，要求 `ComplexRoPE`；CPU 输入回退 stock forward，CUDA 路径用 fused Q/KV RoPE assembly custom ops（`torchtitan/overrides/fused_mla.py:803-825`、`torchtitan/overrides/fused_mla.py:847-899`）。

这些 override 说明“显式融合”是模型/config 类型级选择，不是导入 TorchTitan 就自动打开全部 kernel。

---

## 5. dist-GEMM：把 TP collective 折进 GEMM

dist-GEMM 是当前新增的显式模块后端，与编译器的 Async TP pass 不是同一入口。`tp_gemm_backend` 只有 `default` 与 `dist_gemm`；后者通过 symmetric-memory op 把 all-gather/reduce-scatter 与相邻 GEMM 融合（`torchtitan/models/common/config_utils.py:62-68`）。

真实调用链是：

```text
model_registry(tp_gemm_backend="dist_gemm")
  -> make_gqa_config: AllGatherFusedQKVLinear + RowParallelLinear
  -> make_ffn_config: DistGEMMFeedForward
  -> decoder_sharding 删除已被模块接管的 boundary redistribution
  -> forward 内从 current spmd_types mesh 取得 TP group
  -> distributed/linear.py 的 autograd Function
  -> torch.ops.symm_mem.fused_all_gather_matmul /
     torch.ops.symm_mem.fused_matmul_reduce_scatter
```

attention 端要求 fused QKV：一次 all-gather 喂给 `wqkv`，`wo` 将本地 GEMM 的 partial 直接 reduce-scatter 回 sequence shard（`torchtitan/models/common/config_utils.py:187-229`、`torchtitan/models/common/dist_gemm.py:113-175`）。FFN 端一次 all-gather 同时喂 `w1/w3`，`w2` 再 fused reduce-scatter（`torchtitan/models/common/dist_gemm.py:178-231`）。

底层 autograd 不是只优化 forward。`AllGatherLinear` 前向只保存 gathered input 的 K-shard，避免保存完整输入；反向用 fused reduce-scatter 算 dgrad，再 all-gather 该 K-shard 算 wgrad（`torchtitan/distributed/linear.py:47-78`、`torchtitan/distributed/linear.py:122-180`）。双投影版本一次 gather 喂两份权重，并把两份 dgrad 合成一次 reduce-scatter（`torchtitan/distributed/linear.py:185-210`、`torchtitan/distributed/linear.py:226-279`）。

这条路径有明确边界：必须是 `spmd_types` 且启用 Sequence Parallel；TP=1 时只警告并回退 stock projection；symmetric-memory workspace 当前假定同组 op 串行，不能在不同 stream 上并发占用同一 offset（`torchtitan/models/common/dist_gemm.py:53-110`、`torchtitan/distributed/linear.py:14-25`）。sharding 层会删除已由 dist-GEMM 接管的 attention/FFN boundary all-gather，防止重复通信（`torchtitan/models/common/decoder_sharding.py:226-244`、`torchtitan/models/common/decoder_sharding.py:329-343`）。

FusedSwiGLU 可以与它叠加，但必须选择专门的 `dist_gemm_fused_swiglu` override；这会把单 `w13` Linear 与 all-gather 融合，再把 `w2` 与 reduce-scatter 融合（`torchtitan/overrides/fused_swiglu.py:507-558`、`torchtitan/overrides/fused_swiglu.py:580-597`）。更完整的通信调度与 Async TP 对照见 [[24_torchtitan_comm_optimizations_overlap_analysis]]。

---

## 6. `torch.compile` 与核心 CUDA Graph 是两条独立路径

### 6.1 regional compile：逐 TransformerBlock，而非整模型

Llama 当前顺序是 TP/SPMD sharding contract → AC wrapper → compile → FSDP；`apply_compile` 对每个 TransformerBlock 调 `compile(fullgraph=True)`（`torchtitan/models/llama3/parallelize.py:40-55`、`torchtitan/distributed/compile.py:39-72`）。重复 block 可以复用编译结果，同时把 FSDP wrapper 留在 compiled region 外。

`CompileConfig.components` 默认包含 `model` 和 `loss`；leaf loss 在选择 `loss` 时独立 compile，不会把 `lm_head` 的 Chunked Loss 循环一起编进 loss function（`torchtitan/config/configs.py:295-315`、`torchtitan/components/loss.py:234-260`、`torchtitan/components/loss.py:551-570`）。FlexAttention 自身也持有一个 Inductor 编译入口，并关闭其内部 Triton CUDA graphs，避免把这点与 Trainer 的 core graph 混为一谈（`torchtitan/models/common/attention.py:222-259`）。

Async TP 已经归入 compile 配置：只有 `compile.enable=True`、`components` 含 `model` 时才能打开；`apply_compile` 会给 TP group 注册 symmetric memory，并设置 Inductor `_micro_pipeline_tp`（`torchtitan/config/configs.py:295-315`、`torchtitan/distributed/compile.py:75-96`）。这与 §5 的显式 dist-GEMM module backend 是两个独立机制。

### 6.2 核心 CUDA Graph：capture 整个 forward+backward body

`training.disable_cuda_graphs=False` 是默认值。非 PP 路径把 `_forward_backward_body` 包成 `wrap_with_cuda_graph`；这个 body 同时包含模型前向、loss 与 `loss.backward()`（`torchtitan/trainer.py:579-588`、`torchtitan/trainer.py:703-728`）。

wrapper 首次在独立 stream warmup，第二次 capture，之后把新 tensor 输入 copy 到静态 buffer 并 replay；shape、dtype、device、非 tensor 值和 `BlockMask` 结构都必须保持不变（`torchtitan/distributed/cudagraph.py:189-240`、`torchtitan/distributed/cudagraph.py:264-325`）。graph 开启时 `zero_grad(set_to_none=False)` 保留梯度存储，日志若要跨 replay 保存 loss 会先 clone graph-owned output（`torchtitan/trainer.py:774-778`、`torchtitan/trainer.py:830-848`）。

当前限制来自 capture 本身：

- 仅 NVIDIA CUDA；ROCm/CPU 会警告并退回 eager（`torchtitan/distributed/cudagraph.py:335-351`）。
- PP 尚不支持；EP 只允许无 host sync 的 MinimalAsyncEP，或设置 `non_blocking_capacity_factor` 的 HybridEP（`torchtitan/trainer.py:165-196`）。
- auxiliary pytree 与 `BlockMask` 结构必须跨 step 不变（`torchtitan/distributed/cudagraph.py:39-98`）。

因此，“开启 compile”与“开启 core CUDA Graph”既不互相包含，也不互相替代；默认配置甚至允许前者关闭、后者开启。

---

## 7. 优化器：package 化、多类型分组与 fused state

`OptimizersContainer` 现在允许同一个 model part 内按参数 FQN 正则使用不同 optimizer；规则按列表顺序 first-match-wins，同一 optimizer 名的多个 param group 会合进一个 optimizer instance（`torchtitan/components/optimizer/optimizer.py:40-67`、`torchtitan/components/optimizer/optimizer.py:79-100`）。参数名以 canonical FQN 写入 group，供扁平化 checkpoint state 使用（`torchtitan/components/optimizer/optimizer.py:168-216`）。

`implementation` 默认仍是 `fused`，并将 `fused`/`foreach` kwargs 交给 PyTorch optimizer；`fused_opt_states_bf16` 只支持 Adam/AdamW，通过 step pre-hook 预建 BF16 momentum/variance，让 fused CUDA kernel 进入混合精度 state 路径（`torchtitan/components/optimizer/optimizer.py:102-165`、`torchtitan/components/optimizer/optimizer.py:332-371`）。不要把它理解为 TorchTitan 自写了一套 Adam kernel。

工厂当前支持 `Adam`、`AdamW` 与 `DistMuon`，并且每个 optimizer 名还可接收一次性的 factory kwargs（`torchtitan/components/optimizer/optimizer.py:127-150`、`torchtitan/components/optimizer/optimizer.py:218-243`）。DistMuon 的二维 mesh、bucket 与 collective 机制不在本页重复，见 [[26_torchtitan_flex_shard_dist_muon_analysis]]。

---

## 8. Chunked Loss：省的是 logits 峰值，不是 decoder activation

`ChunkedLossWrapper` 让 decoder 跳过 `lm_head`，返回 hidden states；Trainer 再把真实 `lm_head` 注入 loss wrapper（`torchtitan/models/common/decoder.py:297-308`、`torchtitan/trainer.py:495-520`）。loss 沿本地 token 维等分 hidden/labels，逐 chunk 执行 `lm_head + inner loss`，所以任一时刻只保留一个 chunk 的大词表 logits（`torchtitan/components/loss.py:513-552`、`torchtitan/components/loss.py:605-647`）。本地 token 数不能被 `num_chunks` 整除时直接失败（`torchtitan/components/loss.py:614-623`）。

训练态下，每个 detached chunk 单独反向到 hidden leaf；`GradAccumulator` 预分配 FP32 buffer，原地写入 chunk grad，最后由自定义 autograd Function 把完整 hidden gradient 接回 decoder graph（`torchtitan/components/loss.py:649-708`、`torchtitan/components/loss.py:718-729`、`torchtitan/components/loss.py:780-817`）。验证态 hidden 不需要 grad，因此不建 accumulator，也不做 chunk backward（`torchtitan/components/loss.py:587-603`、`torchtitan/components/loss.py:715-719`）。

FSDP `lm_head` 在 chunk 循环前禁止 forward/backward reshard，并关闭前 N-1 个 chunk 的 grad sync；最后一个 chunk 恢复 sync 后触发合并的 reduce-scatter，循环后再恢复策略并 reshard（`torchtitan/components/loss.py:666-714`）。

> [!warning] 当前源码内部有一处文字与执行逻辑矛盾
> `ChunkedLossWrapper` 类 docstring 的 `torchtitan/components/loss.py:534-538` 写“reduce-scatter fires per-chunk”，但实际分支及其注释在 `:666-714` 明确关闭前 N-1 个 chunk 的 gradient sync，并称其为“single reduce-scatter at the last chunk”。本页按可执行逻辑描述；该 docstring 应视为待修正文档债务。

---

## 9. CPU offload、混合精度与训练步内存

`training.enable_cpu_offload` 的配置语义是 offload 参数、梯度和 optimizer state；Llama 的实现把 `CPUOffloadPolicy` 注入每个 FSDP unit（`torchtitan/config/configs.py:72-75`、`torchtitan/distributed/fsdp.py:223-232`）。Trainer 在此模式下以 CPU 作为参数初始化设备、保留 GPU buffer device，并额外启用 CPU distributed backend（`torchtitan/trainer.py:390-400`、`torchtitan/trainer.py:628-637`）。因此 offload 是 FSDP 生命周期策略，不是一次性的 `model.cpu()`。

混合精度要区分两个配置：`training.dtype=bfloat16` 是参数、梯度、optimizer state 全 BF16 的 full-BF16 训练；`mixed_precision_param=bfloat16` 与 `mixed_precision_reduce=float32` 则交给 FSDP/autocast 策略（`torchtitan/config/configs.py:89-108`）。Llama 即使 FSDP shard degree 为 1 也调用 `apply_fsdp_to_decoder`，借此安装 `MixedPrecisionPolicy`；该 policy 明确设置 param dtype、reduce dtype 且不在 wrapper 边界强制 cast forward inputs（`torchtitan/models/llama3/parallelize.py:57-78`、`torchtitan/distributed/fsdp.py:223-228`）。

训练步还做两项峰值控制：先把 gradient-accumulation 所需 microbatch 保留在 CPU，处理每组时才搬到 device；loss/grad finiteness 用 device-side async assert，在 optimizer 更新前停止非有限 step，避免每步 host sync（`torchtitan/trainer.py:788-834`、`torchtitan/trainer.py:878-889`）。

---

## 10. 选择与失败边界

| 目标 | 当前机制 | 必要边界 |
|---|---|---|
| 降低 dense GEMM 精度 | Float8 / MXFP8 / NVFP4 converter | Float8 维度 16 对齐；MXFP8/NVFP4 要 SM100；NVFP4 本地 GEMM 维度还要 128 对齐 |
| 降低 MoE grouped GEMM 精度 | Float8/MXFP8 expert converter | dispatcher 必须支持 padding；MXFP8 padding 倍数需匹配 kernel |
| 合并 gate/up 与激活 | fused SwiGLU / FusedGroupedExperts override | opt-in；checkpoint hooks 保持 stock keys |
| overlap TP collective 与 GEMM | dist-GEMM | `spmd_types` + SP + CUDA symmetric memory；与 compile Async TP 是不同入口 |
| 编译 pointwise/epilogue | per-block compile | `compile.enable` 且 `components` 含 `model`；FlexAttention backend 有额外约束 |
| 降低 eager launch 开销 | core CUDA Graph | 固定输入 metadata；无 PP；EP dispatcher 不得 host sync；仅 NVIDIA CUDA |
| 降低词表 logits 峰值 | Chunked Loss | 本地 token 数可整除 chunk 数；省 logits，不替代 AC |
| 降低 GPU 状态驻留 | FSDP CPU offload | 额外 CPU 内存/传输成本；通过 FSDP policy 生效 |

量化 recipe 是否更快、BF16 tail 比例是否稳定、dist-GEMM 是否胜过编译器 Async TP，都不能仅从接线源码得出；它们依赖模型 shape、硬件、TorchAO/PyTorch 版本和通信拓扑，需要目标集群 benchmark。

---

## 11. 小结

- TorchTitan 的核心角色是把 converter、模块 override、SPMD/FSDP 契约和运行时组合起来，而非复制底层 kernel 库。
- 当前低精度主线是 Float8、MXFP8、NVFP4；NVFP4 保留 BF16 主权重与 BF16 TP collective，不等价于 4-bit 模型状态压缩。
- 显式融合已扩展到标准 Linear 版 FusedSwiGLU、FusedGroupedExperts、fused MLA、Helion RoPE 与 dist-GEMM；旧 `einsum`/“无自写 Triton”结论已失效。
- per-block compile、compile Async TP、显式 dist-GEMM 与 core CUDA Graph 是四个不同层级的机制，不能用一个“compile 已开”笼统代替。
- 训练态显存主要由 Chunked Loss、FSDP mixed precision/offload、optimizer state dtype 与 AC 共同控制；各自优化的是不同存活区间。

## Related Pages

- [[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]] —— 系列入口、页面边界与当前基线。
- [[11_torchtitan_fsdp_analysis]] —— MixedPrecisionPolicy、CPUOffloadPolicy 与 FSDP unit 生命周期。
- [[12_torchtitan_tp_analysis]] —— TP/SP 布局、loss parallel 与 compile Async TP 的完整调用链。
- [[22_torchtitan_ac_analysis]] —— decoder activation 的重计算策略；与 Chunked Loss 优化不同对象。
- [[24_torchtitan_comm_optimizations_overlap_analysis]] —— symmetric memory、dist-GEMM、Async TP 与 EP overlap 的通信侧深挖。
- [[26_torchtitan_flex_shard_dist_muon_analysis]] —— DistMuon 的 mesh、bucket 与分布式 optimizer 机制。
- [[27_torchtitan_graph_trainer_compiler_runtime_analysis]] —— 实验 Graph Trainer 的编译/图运行时，与核心 Trainer CUDA Graph 对照。
