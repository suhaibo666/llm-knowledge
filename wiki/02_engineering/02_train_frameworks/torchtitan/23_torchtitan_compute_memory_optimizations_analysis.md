# 计算与显存优化 —— 低精度 · 算子融合 · 编译 · 优化器 · ChunkedCE · CPU offload(源码级)

> **代码基准**:torchtitan `main` @ `61c010fcb` · PyTorch `2.9.1`;低精度依赖 torchao
> **最后更新**:2026-06-16 · **系列**:torchtitan 多维并行源码级分析(见 [[torchtitan/index]])
>
> 本篇聚焦**算力/显存**侧的非并行性能手段;**通信**侧(对称内存、Async-TP、重叠矩阵、MinimalAsyncEP)见 [[24_torchtitan_comm_optimizations_overlap_analysis]]。
> 行号约定:torchtitan 以 `torchtitan/torchtitan/` 为根;PyTorch 2.9.1 以 `[pt]` 前缀。torchao 未安装在本机,fp8 内部 casting 只在调用点引用。

---

## 0. 全景:三类"省算力/省显存"的手段

```
省算力(吞吐):
  ① 低精度 GEMM     Float8(rowwise)/ MXFP8  ── torch._scaled_mm / _scaled_grouped_mm,fp8 tensorcore
  ② 显式融合算子    FusedSwiGLU(gate+up 合一)/ MoE grouped GEMM / FusedQKVLinear / RMSNorm(aten)
  ③ 编译即融合      regional torch.compile 逐 TransformerBlock,inductor 做 pointwise/epilogue 融合
  ④ 融合优化器      torch.optim Adam(fused=True,默认)单 kernel 全参数更新

省显存:
  ⑤ ChunkedCELoss   O(B·L·V) -> O(B·L/N·V),不物化完整 logits
  ⑥ CPU offload     分片 param/grad/optim 落 CPU,D2H/H2D 与计算重叠
  ⑦ 混合精度        MixedPrecisionPolicy(param=bf16, reduce=fp32),非 autocast
```

关键定位:**torchtitan 是薄封装**——低精度靠 torchao,融合 GEMM 靠 `torch._grouped_mm` / `torch._scaled_mm`,pointwise 融合靠 inductor,融合优化器靠 `torch.optim`。torchtitan 负责的是**何时、对哪些模块、按什么顺序**接入。

---

## 1. 低精度训练:Float8(rowwise)与 MXFP8

### 1.1 接入方式:在 config 上量化,不是在 model 上 swap

新版 torchtitan(`#3127 "quantize on config instead of on model"`)用**配置树转换器**:量化在**建模前**完成。`QuantizationConverter` 遍历模型 `Config` 树,把 `Linear.Config` 替换成 `Float8Linear.Config` / `MXFP8Linear.Config`,于是模型**直接以量化模块建出来**,不再做建好后 `convert_to_float8_training` 的 module swap。

- 转换:`components/quantization/float8.py:151` `for fqn, linear_config, parent, attr in model_config.traverse(Linear.Config):` → 过滤后写回 `Float8Linear.Config(..., _torchao_config=self.torchao_config)`(`:160-163`)。
- torchao 模块本体:`float8.py:27` `class Float8Linear(TorchAOFloat8Linear, Module)`;config 由**配方名**构造:`float8.py:108-110` `TorchAOFloat8LinearConfig.from_recipe_name(cfg.recipe_name)`。
- 时机:转换器在 `model_registry()` 里跑(`models/llama3/__init__.py:379-382`),**早于** TP/AC/compile/FSDP。模型随后在 meta device 上 build 成 `Float8Linear`(`trainer.py:285`)。

### 1.2 转换什么 / 排除什么

`utils.py:module_filter_fn`(`utils.py:16-30`):

- **维度必须是 16 的倍数**(fp8 tensorcore 硬约束):`in_features % 16 == 0 and out_features % 16 == 0`(`utils.py:23-25`),否则**静默跳过**。
- **`filter_fqns` 黑名单**跳过(`utils.py:28`)。真实配置排除 LM head / router:llama3-405B `filter_fqns=["output"]`(`config_registry.py:193`);deepseek-v3-671B `["output", "router.gate"]`(`config_registry.py:154`)。
- `auto_filter_small_kn` 哨兵切换到 torchao 的按 GEMM 尺寸自动过滤(小 GEMM 在 H100 上不划算,跳过)(`float8.py:121-140`)。

### 1.3 缩放配方:只有 rowwise(无 tensorwise)

`Float8LinearConverter.Config.recipe_name`(`float8.py:58`):

```python
recipe_name: Literal["rowwise", "rowwise_with_gw_hp"] = "rowwise"
```

- **本版 Linear 转换器只暴露 rowwise**(每行/每列一个 scale),`tensorwise`(整张量一个 scale)**不可选**。`rowwise_with_gw_hp` = 权重梯度(wgrad)GEMM 保高精度、fwd 与 dgrad 走 fp8。
- rowwise 带一个 inductor 兼容开关:`float8.py:115-118` `torch._inductor.config.emulate_precision_casts = True`(规避 pytorch/pytorch#150859)。
- **MXFP8 = 块缩放(microscaling)**:1×32 元素共享一个 scale,数据 `e4m3fn` + scale `e8m0`(`mxfp8.md:35`),粒度比 rowwise 更细。

### 1.4 算力收益:fp8 tensorcore GEMM

低精度把 GEMM 跑在 **fp8 tensorcore** 上:激活/权重动态量化到 fp8 → `torch._scaled_mm`(稠密)/ `torch._scaled_grouped_mm`(MoE)→ 净加速(`mxfp8.md:36-37`)。三段 GEMM(fwd / dgrad / wgrad)的 fp8 casting 在 torchao 内部,torchtitan 只通过 `rowwise_with_gw_hp` 选择"让 wgrad 留高精度"。

> 收益前提(`float8.md:3`):只有当**大多数 GEMM 足够大**,fp8 tensorcore 的加速才能盖过动态量化的开销——所以才有 `filter_fqns` / `auto_filter_small_kn`。

### 1.5 通信:rowwise 下**通信仍是高精度**(纠正常见误解)

**本版没有 fp8 weight all-gather。** 全仓 grep `enable_fsdp_float8_all_gather` / `precompute_float8_dynamic_scale_for_fsdp` **无匹配**。文档明说:`float8.md:53` "for float8 with rowwise scaling, all distributed communication is done in high precision";`mxfp8.md:208` 同。FSDP all-gather 仍按 `mixed_precision_param`(bf16)走(`parallelize.py:90`)。

> 含义:历史上"fp8 all-gather 省通信"只对 **tensorwise** 成立;本版 rowwise-only,**通信量不减**,低精度的收益是**纯算力**。这一点与很多旧资料不同,务必区分。

### 1.6 硬件/编译约束

- Float8 需 **SM89(H100/Ada)+**:`float8.py:85-91` 否则 `ValueError`;MoE fp8 同(`float8.py:222-223`)。
- MXFP8 需 **SM100(Blackwell B200)+**:`mx.py:70-71`(prototype,`torchao.prototype`)。
- **compile 强相关**:`float8.md:33` "torch.compile fuses the float8 scaling/casting kernels",是有竞争力性能的前提;MoE fp8 直接 warn 要求 compile(`float8.py:225-229`)。`emulate` 模式与 compile 不兼容(`float8.py:68-72`)。

### 1.7 MoE 专家的低精度

单独的 `Float8GroupedExpertsConverter` / `MXFP8GroupedExpertsConverter`(`float8.py:204-248` / `mx.py:139-192`)把 `GroupedExperts` 换成 torchao 量化子类,让 grouped GEMM 走 `torch._scaled_grouped_mm`,并把 token dispatcher 换成 **padded** 变体(fp8 `PAD_MULTIPLE=16`,mxfp8 `pad_multiple=32`,sm100 CuTeDSL 要 128)。

---

## 2. 显式融合算子

### 2.1 FusedSwiGLU:gate+up 合成一个 GEMM(`#3638`,override)

stock FeedForward **没有显式融合**:`feed_forward.py:53-54` `self.w2(F.silu(self.w1(x)) * self.w3(x))`——三个独立 GEMM + eager 的 `silu*up`(elementwise 融合交给 compile)。

opt-in 的 `overrides/fused_swiglu.py` 把 **gate(w1)与 up(w3)合成一个 GEMM**:`w13` 形状 `(hidden, 2, dim)`(`:82`),前向一次 einsum(`:89-93`):

```python
gate, up = torch.einsum("...d,hgd->...hg", x, self.w13).unbind(-1)
return self.w2(F.silu(gate) * up)
```

- 收益:up-projection 从两次 GEMM 变一次,减一半 kernel 启动;`(hidden,2,dim)` 布局连续、转置无开销(`:14-31`)。
- **checkpoint 互通**(`#3638` 的重点):活参是 `w13`,但存/读 checkpoint 时拆成 stock 的 `w1.weight`/`w3.weight`——`register_state_dict_post_hook` 存时拆(`:95-105`)、`register_load_state_dict_pre_hook` 读时合(`:107-119`),与 stock FeedForward / HF 适配互换。
- TP 把 `w13` 按 `Shard(0)`(显式 `2` 维给每 rank 匹配的 gate/up 切片,Megatron 列并行)(`:159-168`)。
- 选择:override 注册(`@override("fused_swiglu", ...)` `:122-127`),`--override.imports torchtitan.overrides.fused_swiglu` 激活。

### 2.2 MoE grouped GEMM:一个 kernel 算完本地所有专家

专家计算用 **`torch._grouped_mm` + `offs`**(每专家累积 token 偏移),不是循环、不是 bmm。`models/common/moe.py:80-96`:

```python
offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
h_RF = F.silu(torch._grouped_mm(x_RD.bfloat16(), w1_EFD.bfloat16().transpose(-2,-1), offs=offsets_E))
h_RF = h_RF * torch._grouped_mm(x_RD.bfloat16(), w3_EFD..., offs=offsets_E)
return torch._grouped_mm(h_RF, w2_EDF..., offs=offsets_E).type_as(x_RD)
```

`offs` 让一次 kernel 完成所有本地专家的 matmul(每专家吃自己那段连续 token)。EP/TP 下 DTensor 权重先 `.to_local()`(动态 per-rank token 数无法用 DTensor 表达)(`moe.py:67-78`)。低精度时换成 `torch._scaled_grouped_mm`(§1.7)。

### 2.3 FusedQKVLinear / RMSNorm / RoPE / attention

| 算子 | 是否融合 | 机制 | 源码 |
|---|---|---|---|
| QKV 投影 | **融合** | 单 `wqkv` GEMM 再按 R 维切,省 2 次 kernel 启动 | `attention.py:608-650` |
| RMSNorm | **融合(aten)** | `nn.RMSNorm` → `torch.rms_norm` 单融合 aten 算子,**非** 自写 Triton | `nn_modules.py:151-165`、`[pt]nn/functional.py` |
| RoPE | **不融合** | `view_as_complex` 复乘 / `cos+rotate_half·sin`,elementwise 融合交给 compile | `rope.py:240-251/329-343` |
| SDPA | **融合(flash/cuDNN)** | `F.scaled_dot_product_attention` + 显式 `sdpa_kernel([CUDNN,FLASH,MATH])` | `attention.py:256-309` |
| FlexAttention | **融合(compiled Triton)** | `torch.compile(flex_attention)` + `max_autotune`,生成融合 Triton | `attention.py:167-253` |
| Varlen | **融合(FA3)** | `varlen_attn`,Hopper 上强制 FA3 | `attention.py:73-164` |
| gpt_oss swiglu | 省中间分配 | `torch.addcmul` 替 `out*(x+1)`,免物化大中间(`#3454`) | `gpt_oss/moe.py:43-50` |

> 唯一的**自写 Triton kernel** 在 `distributed/minimal_async_ep/kernels.py`(MoE permute/combine,见 [[24_torchtitan_comm_optimizations_overlap_analysis]] §5),不是通用模型算子。

---

## 3. 融合优化器

`components/optimizer.py`,**默认 `fused=True`**:

```python
implementation: Literal["for-loop","foreach","fused","fused_opt_states_bf16"] = "fused"  # :102-118
fused = config.implementation in ("fused", "fused_opt_states_bf16")
return {"fused": fused, "foreach": config.implementation == "foreach"}                    # :133-146
```

- 这些 kwarg 注入每个 param-group(`:187-194`),按 `(正则, 优化器名)` 分组,共享同一优化器类型的参数合进一个 `torch.optim.Adam/AdamW`(`:206-216`)。靠 **PyTorch 内置 fused/foreach Adam**,torchtitan 无自写优化器 kernel。
- 性能:`fused`=单 CUDA kernel 完成全部参数 Adam 更新(启动/访存最少);`foreach`=每个数学步一个 `_foreach` 横向融合;`for-loop`=逐参数最慢(`:110-117`)。
- `fused_opt_states_bf16`:fp32 参数 + bf16 动量/方差,走融合 kernel 的混合精度路径(`FusedAdamMathFunctorMP`),step 前钩子预建 bf16 state(`:310-349`)。

---

## 4. 编译即融合:regional 逐 TransformerBlock compile

`distributed/compile.py`。**逐 block compile,不是整模型**:

```python
for layer_id, transformer_block in model.layers.named_children():
    transformer_block.compile(backend=backend, fullgraph=True)   # :55-56
```

- 为何 regional(`:36-39`):所有 block 结构相同 → Dynamo/Inductor **只编译一个 block** 复用到每层,省编译时间;block 边界外保留 FSDP2 的 all-gather/reshard 边界(`parallelize.py:72-74` "after AC wrapping and before FSDP")。
- dynamo flags:`capture_scalar_outputs=True`(`:42`,MoE token-choice 的数据依赖动态形状);`skip_fwd_side_effects_in_bwd_under_checkpoint=True`(`:48-50`,容忍 AC 重算时 RoPE cache 等前向副作用)。
- **融合在哪发生**:默认 `backend="inductor"`,每个被 compile 的 block 由 inductor 做 pointwise/epilogue 融合 + Triton codegen——stock FeedForward 的 `silu*up`、RoPE elementwise、norm 等都在这里被融合。
- regional inductor:`_maybe_regional_inductor_backend`(`:61-96`)在非 inductor 外层后端下,只把标注 `compile_with_inductor` 的 FlexAttention 区域降到 inductor(`#3563`)。
- gate:`compile_config.enable and "model" in compile_config.components`(`parallelize.py:65-67`)。配置 `compile.enable=False` / `components=["model","loss"]` / `backend="inductor"`(`configs.py:258-264`)。
- **依赖关系**:Float8 高性能要 compile(§1.6);Async-TP **必须** compile(见 [[24_torchtitan_comm_optimizations_overlap_analysis]] §3)。

---

## 5. 省显存:ChunkedCELoss

`components/loss.py`。`ChunkedCELoss` **从不物化完整 `[B,L,V]` logits**:模型前向 `_skip_lm_head=True` 出 `[B,L,D]` hidden(`trainer.py:437/445`),loss 把**序列维**切 `num_chunks`(默认 8,`:477`),逐 chunk 跑 `lm_head + cross_entropy`:

- 峰值从 `O(B·L·V)` 降到 `O(B·L/N·V)`(`:442-443`),同时只有一个 chunk 的 logits+softmax 存活。
- 每 chunk `detach().requires_grad_()` 成叶子,`lm_head(h_chunk) → ce → backward → grad_accumulator.add(h_chunk.grad) → h_chunk.grad=None`(`:586-604`)。
- **与 FSDP 协同**:循环前关 `lm_head` 的 reshard 与 grad-sync,使权重整轮不重复 all-gather、最后一个 chunk 才触发**一次** reduce-scatter(`:580-583`);循环后恢复并 reshard(`:606-610`)。
- `GradAccumulator`(`:319-434`):预分配一块 fp32 零缓冲,按 seq_dim 原地切片写入(`:403-409`),捕获首 chunk 的 DTensor placement(如 TP 轴 `Partial(sum)`)在 `result()` 重包成 DTensor(`:413-433`),让 TP 规约只发生一次。
- no_grad 快路径(`#3652`):仅 `requires_grad` 时才建 accumulator(`:566-572`),验证/推理直接返回 `total_loss`。

---

## 6. 省显存:CPU offload + 混合精度 + 训练循环细节

### 6.1 CPU offload

`training.enable_cpu_offload`(`configs.py:46-49`)→ `fsdp_config["offload_policy"]=CPUOffloadPolicy()`(`fsdp.py:144-145`)。分片 param/grad/optim 常驻 CPU/pinned,all-gather 把分片拷上 GPU 算、reduce-scatter 结果 D2H 拷回。

- 配套:权重 CPU 初始化、buffer 上 GPU(`trainer.py:321-323`);加 gloo CPU backend 让 offload 张量也能集合通信(`trainer.py:563`)。
- **D2H 重叠规则**(`[pt]_fsdp_collectives.py:592-606`):仅**非梯度累积**时才非阻塞 overlap D2H(累积时 CPU add 依赖拷贝结果);非阻塞时记 `grad_offload_event`,optimizer 前等它。

### 6.2 混合精度(非 autocast)

`MixedPrecisionPolicy(param_dtype, reduce_dtype, cast_forward_inputs=False)`(`fsdp.py:136-140`),`param=bf16 / reduce=fp32`(`configs.py:58-70`)。核心训练循环**不用 `torch.autocast`**(只在 DDP/单卡路径与 MoE 路由强制 fp32 score 处用)。这与 [[21_torchtitan_hsdp_backward_overlap_analysis]] §9 的 fp32 reduce 暂存一致。

### 6.3 训练循环其它手段(含一处重要纠正)

- **梯度累积**:`gradient_accumulation_steps = global_batch // (local_batch × batch_degree)`(`trainer.py:348-351`),循环内逐 microbatch `backward`,循环后一次 clip+step(`:799-809`)。microbatch 在 **CPU 收集、逐个上 GPU**,省累积显存(`:574-577`)。
- ⚠️ **纠正常见假设**:torchtitan 核心训练循环**不**在 microbatch 间延迟 FSDP 规约——全仓无 `no_sync` / `set_requires_gradient_sync(False)` / `set_is_last_backward` 在 microbatch 循环里。**每个 microbatch 的反向都触发 reduce-scatter**(只有 `ChunkedCELoss` 在 *chunk* 级做了合并,§5)。
- 全局 token 缩放:loss 用 `reduction="sum"`,再除以 **batch mesh 上 all-reduce 出的全局有效 token 数**(`trainer.py:766-781`、`loss.py:289-292`);因此 `disable_fsdp_gradient_division` 关掉 FSDP 按 world_size 除(见 [[21_torchtitan_hsdp_backward_overlap_analysis]] §7)。
- 确定性代价:`set_determinism`(`utils.py:124-168`)开 `use_deterministic_algorithms`、关 cuDNN benchmark/flex max-autotune,注释明示"expect perf degradation"。

---

## 7. 源码复核小结

| 断言 | 位置 | 结果 |
|---|---|---|
| 量化在 config 树上转换(非 model swap) | `components/quantization/float8.py:151-163` | OK |
| Float8 Linear 只 rowwise(无 tensorwise) | `float8.py:58` | OK |
| 维度需 %16,filter_fqns 排除 output/router | `utils.py:23-28`、`config_registry.py:154/193` | OK |
| **rowwise 通信仍高精度,无 fp8 all-gather** | `float8.md:53`、grep 无 `*float8_all_gather*` | OK |
| Float8 需 SM89 / MXFP8 需 SM100 | `float8.py:85-91`、`mx.py:70-71` | OK |
| MoE 专家 grouped GEMM = `torch._grouped_mm`+offs | `models/common/moe.py:80-96` | OK |
| FusedSwiGLU gate+up 合一 GEMM + checkpoint 互通 | `overrides/fused_swiglu.py:82-119` | OK |
| RMSNorm = `torch.rms_norm` 融合 aten;RoPE 不融合 | `nn_modules.py:151-165`、`rope.py:240-343` | OK |
| FusedQKVLinear 单 wqkv GEMM | `attention.py:608-650` | OK |
| 优化器默认 `fused=True` | `components/optimizer.py:102-146` | OK |
| 逐 TransformerBlock regional compile | `distributed/compile.py:55-56` | OK |
| ChunkedCELoss O(BLV)->O(BL/N·V) + 单次 RS | `components/loss.py:442-443/580-604` | OK |
| **microbatch 间不延迟 FSDP 规约(每 mb 都 RS)** | 全仓无 `no_sync`/`set_requires_gradient_sync` 于 `trainer.py` | OK |
| CPU offload = CPUOffloadPolicy + gloo + D2H 重叠 | `fsdp.py:144-145`、`[pt]_fsdp_collectives.py:592-606` | OK |

---

## 8. 小结

- **算力**:① 低精度 GEMM(Float8 rowwise / MXFP8 块缩放,`_scaled_mm`/`_scaled_grouped_mm`,SM89/SM100,需 compile;**通信仍高精度**);② 显式融合算子(FusedSwiGLU gate+up 合一、MoE `torch._grouped_mm`、FusedQKVLinear、RMSNorm aten);③ 编译即融合(逐 block inductor);④ 融合 Adam(默认 fused)。
- **显存**:⑤ ChunkedCELoss(不物化完整 logits + chunk 级单次 RS);⑥ CPU offload(D2H 与计算重叠);⑦ 混合精度(MixedPrecisionPolicy,非 autocast)。
- **薄封装哲学**:重活在 torchao / `torch._grouped_mm` / inductor / `torch.optim`;torchtitan 决定接入时机、模块范围、与并行/编译的顺序。
- 通信侧的性能手段(对称内存、Async-TP 微流水、跨维度重叠矩阵、MinimalAsyncEP)见 [[24_torchtitan_comm_optimizations_overlap_analysis]]。

---

## Related Pages

- [[24_torchtitan_comm_optimizations_overlap_analysis]] —— 通信侧性能手段(对称内存/Async-TP/重叠矩阵),与本篇互补
- [[11_torchtitan_fsdp_analysis]] —— FSDP2 标杆篇
- [[21_torchtitan_hsdp_backward_overlap_analysis]] —— HSDP 反向 fp32 reduce 暂存与峰值
- [[22_torchtitan_ac_analysis]] —— 激活重计算(另一类省显存手段)
- [[21_megatron_fusion_operators_analysis]] —— Megatron 融合算子对照(跨框架)
- [[torchtitan/index]] —— torchtitan 多维并行知识地图
