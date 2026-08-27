---
title: "TorchTitan 计算与显存优化：先改表示，再编译区域，最后固定训练步"
---

# TorchTitan 计算与显存优化：先改表示，再编译区域，最后固定训练步

> **论点式副标题**：TorchTitan 没有一个包办吞吐与显存的“优化总开关”。当前主链依次在配置树上换计算表示或冻结训练状态、在 TransformerBlock/loss 上划编译区域、在 Trainer 中固定 forward+loss+backward 的 CUDA Graph 边界，并在图外以 chunked loss、optimizer state 和有限性闸门管理训练态存活区间。
>
> **代码基准**：pytorch/torchtitan `main` @ `a3168782c9a3a2e40afbd0de114818b96e2bda6e`（2026-08-26）
> **本页回答**：Float8/MXFP8/NVFP4、LoRA converter 与显式 fused override 改了什么；compile component/region 如何接线；core CUDA Graph 如何 capture/replay；optimizer、gradient accumulation、Chunked Loss 与 finite gate 如何决定峰值和更新边界。
> **Sibling 边界**：AC 的 decoder activation 重计算见 22 页；FSDP mixed precision/offload 见 11 页；dist-GEMM、Async TP 和 EP overlap 的通信调度见 24 页；GraphTrainer 的整图内存策略见 27 页。本页只解释这些机制与标准 Trainer 计算/显存控制面的交界。

---

## 1. Overview

### 1.1 背景、问题与 thesis

“训练更快/更省显存”至少包含四个不同问题：GEMM 用什么数值格式，哪些算子共享一次读写或 collective，编译器看到多大的重复区域，以及反向前后哪些张量必须继续存活。若把它们压成一个布尔开关，就无法判断收益来自哪里，也无法解释为什么某个组合在特定硬件、并行布局或输入形状下失败。

当前代码给出的主线是：

1. **converter/override 先改变模型表示**：converter 在模型 build 前替换配置叶子；override 在 model config 已接收 job-level 设置之后、任何组件 build 之前改写整棵配置树，可把 FFN/experts 变成结构不同但协议兼容的实现（`torchtitan/protocols/model.py:19-33`、`torchtitan/models/llama3/__init__.py:369-389`、`torchtitan/trainer.py:334-357`）。
2. **compile 选择区域，不负责所有优化**：Llama 先施加 SPMD/TP、再 AC、逐 TransformerBlock compile，最后 FSDP；loss 可作为独立 component 编译（`torchtitan/models/llama3/parallelize.py:40-78`、`torchtitan/components/loss.py:234-260`）。
3. **CUDA Graph 是独立运行时边界**：core Trainer 只包装 model+loss+backward callable，输入 copy-in、optimizer、checkpoint、metrics 和 validation 留在图外（`torchtitan/trainer.py:579-588`、`torchtitan/trainer.py:703-728`、`torchtitan/trainer.py:774-940`）。
4. **训练态显存由存活区间而非单一 dtype 决定**：gradient accumulation 输入留在 CPU 到使用时才搬运；Chunked Loss 限制大词表 logits 峰值；optimizer state dtype 与 FSDP/AC 则管理另外的状态类别（`torchtitan/trainer.py:785-834`、`torchtitan/components/loss.py:513-551`、`torchtitan/components/optimizer/optimizer.py:102-165`）。

### 1.2 四个接线面

| 接线面 | 当前入口 | 改变的对象 | 明确不等价于 |
|---|---|---|---|
| 配置表示 | Float8/MXFP8/NVFP4/LoRA converter、fused override | Linear/experts/FFN 的 Config 类型、可训练状态与本地 kernel | 自动改变 TP/FSDP wire dtype |
| 编译区域 | `CompileConfig.components`、`apply_compile()` | TransformerBlock、leaf loss、可选 Async TP compiler pass | core CUDA Graph capture |
| 固定执行 | `wrap_with_cuda_graph()` | forward+loss+backward 的 kernel launch 序列与静态存储 | 整个 train step 或 PP schedule |
| 训练状态 | Chunked Loss、optimizer、CPU staging、finite gate | logits、梯度、optimizer state、输入与指标的存活期 | decoder activation 重计算或权重 checkpoint |

### 1.3 关键图：优化顺序也是组合契约

```text
recipe
  -> model_registry(converters=[...])       # 改配置叶子
  -> Trainer override pass                  # 可换整个 FFN/experts 结构
  -> meta model build
  -> model.parallelize                      # SPMD/TP layouts
  -> activation checkpoint wrapper
  -> compile each TransformerBlock          # 可同时配置 compiler Async TP
  -> FSDP wrapper / mixed-precision policy
  -> materialize + init
  -> optimizer / scheduler
  -> Trainer runtime
       CPU microbatch staging
       -> CUDA Graph(eager warmup -> capture -> replay fwd/loss/bwd)
       -> grad clip + finite gate
       -> optimizer/scheduler step
```

这条顺序解释了一个关键判据：如果某项优化需要改变参数组织或 sharding contract，它必须在 build/parallelize 前后拥有明确 seam；如果只降低 Python/kernel launch，它应位于保持参数语义不变的运行时边界。不能期待 compile 自动发现所有结构融合，也不能用 converter 去表达跨步状态机。

图中的当前锚点依次是：registry 执行 converter（`torchtitan/models/llama3/__init__.py:369-389`），Trainer 执行 override 与 meta build（`torchtitan/trainer.py:334-365`），Llama 执行 sharding→AC→compile→FSDP（`torchtitan/models/llama3/parallelize.py:40-78`），最后 Trainer 物化后构造 optimizer（`torchtitan/trainer.py:472-493`、`torchtitan/trainer.py:533-543`）。

### 1.4 快速选择：先说清楚瓶颈

| 观测到的瓶颈 | 优先核对 | 先不要假设 |
|---|---|---|
| dense/MoE GEMM 吞吐不足 | hardware、shape 对齐、Float8/MXFP8/NVFP4 converter | 低精度一定降低通信或 optimizer state |
| 重复 block 的 Python/pointwise 开销 | `compile.enable`、`components`、backend | compile 自动捕获完整训练循环 |
| eager launch 开销 | CUDA Graph guards 与输入 metadata 稳定性 | CUDA Graph 能容纳 host sync/PP schedule |
| 大词表 logits 峰值 | `ChunkedLossWrapper.num_chunks` | chunked loss 会重算 decoder activation |
| optimizer state 占用 | fused BF16 states、FSDP CPU offload、optimizer 类型 | fused Adam 是 TorchTitan 自写 kernel |
| 只训练低秩 adapter | LoRA converter 的 target、冻结范围与 TP placement | 当前已有 LoRA-only/PEFT checkpoint export |
| TP 通信暴露 | compile Async TP 或 dist-GEMM | 两者是同一个入口或可以盲目叠加 |

---

## 2. 配置树改写：量化 converter 与显式融合

### 2.1 背景与问题

低精度和结构融合若在模型已经物化后 monkey-patch，容易使 checkpoint key、参数初始化、TP/FSDP sharding 与真实模块类型彼此脱节。TorchTitan 要解决的是：在分配真实存储前改变本地计算实现，同时尽量保留模型协议和并行骨架。

### 2.2 为什么选择“改 Config，再 build”

选中路线是让 `ModelConfigConverter` 遍历/替换配置树，模型 registry 按 recipe 给出的顺序执行 converter，然后才把结果封装进 `ModelSpec`（`torchtitan/protocols/model.py:19-33`、`torchtitan/models/llama3/__init__.py:369-389`）。明显替代方案是对已建 `nn.Module` 原地替换；它更直观，却必须重新处理 meta init、state dict hooks 和 sharding metadata。这里的决定标准是：优化后的模块仍要从同一 Config/Module 协议进入 parallelize 与 checkpoint。

converter 与 override 不是同义词。converter 通常把 `Linear.Config` 叶子换成量化子类；FusedSwiGLU 则把两个 projection 的 FFN 结构改成一份 `w13` 参数和一次标准 Linear，再接自定义 SiLU×mul kernel，同时用 state-dict hooks 对外保持 `w1/w3` 键（`torchtitan/overrides/fused_swiglu.py:450-504`）。这种显式结构选择不会等价于“编译器也许会融合”。

### 2.3 当前实现与状态

**共同控制面。** `QuantizationConverter.Config` 显式携带 `model_compile_enabled`，因为量化模块预期在 compile 下才有竞争性性能；这个布尔值不是 Trainer 自动回填，代表 recipe 会从 `CompileConfig` 计算后传入 converter（`torchtitan/components/quantization/__init__.py:13-31`、`torchtitan/models/llama3/config_registry.py:106-137`）。量化必须出现在 LoRA converter 之前，否则 registry 验证直接失败（`torchtitan/models/utils.py:27-46`）。

**Float8。** dense converter 当前提供 `rowwise` 与 `rowwise_with_gw_hp` recipe；硬件门槛是 NVIDIA SM89+ 或 AMD gfx942+，旧硬件只有 eager emulation 测试路径，且 emulation 与 model compile 不兼容（`torchtitan/components/quantization/float8.py:53-96`）。它只转换 in/out features 都被 16 整除且未被 FQN filter 排除的 Linear；可选 TorchAO auto-filter 还能跳过小 K/N、预期不划算的 GEMM（`torchtitan/components/quantization/utils.py:16-30`、`torchtitan/components/quantization/float8.py:125-173`）。MoE 不走 dense leaf swap，而是把 `GroupedExperts._grouped_mm` 换成 TorchAO scaled grouped MM，并强制 dispatcher 提供 16 对齐 padding（`torchtitan/components/quantization/float8.py:176-220`、`torchtitan/components/quantization/float8.py:216-266`）。

**MXFP8。** dense converter 是 `Linear.Config -> MXFP8Linear.Config` 叶子替换，要求 SM100；未 compile 只警告性能，不拒绝构造（`torchtitan/components/quantization/mx.py:50-98`）。MoE converter 当前只暴露 `mxfp8_rceil`，默认 per-expert padding multiple 为 32，但注释指出 SM100 CuTeDSL kernel 可能要求 128（`torchtitan/components/quantization/mx.py:149-187`）。dispatcher 替换只接受 TorchAO/HybridEP 这两类能表达 padding 的实现，否则 converter 阶段报错（`torchtitan/components/quantization/utils.py:33-64`）。

**NVFP4。** 它不是把 checkpoint 永久压成 4 bit：`NVFP4Linear` 保留 BF16 weight，在调用 TorchAO `nvfp4_linear` 时即时量化 activation、weight 与 backward 相关量；TP block boundary 仍使用 stock BF16 all-gather/reduce-scatter（`torchtitan/components/quantization/nvfp4.py:7-16`、`torchtitan/components/quantization/nvfp4.py:234-241`）。opaque Triton autograd op 被注册为 local-safe，stock colwise/rowwise sharding 被折进 `LocalMapConfig`，因此 SPMD type checker仍能验证输入与输入梯度布局（`torchtitan/components/quantization/nvfp4.py:67-82`、`torchtitan/components/quantization/nvfp4.py:113-157`）。

### 2.4 成本、失败与组合边界

- Float8 dense converter 明确用统一 filter 跳过任一维度不被 16 整除的 Linear；当前 MXFP8 dense converter 并未调用这个 filter，而是直接替换选中的/全部 Linear，把更细的 shape 可行性留给 TorchAO 实现（`torchtitan/components/quantization/float8.py:125-173`、`torchtitan/components/quantization/mx.py:79-98`）。NVFP4 则在 TorchTitan Config 层硬性要求 global in/out features 被 128 整除，TP 后的 local GEMM 维度仍由 TorchAO kernel 再校验（`torchtitan/components/quantization/nvfp4.py:94-111`）。
- MXFP8/NVFP4 都要求 SM100，Float8 的平台面更宽；三者都依赖 TorchAO，缺包/缺 prototype 是硬失败（`torchtitan/components/quantization/float8.py:77-103`、`torchtitan/components/quantization/mx.py:65-77`、`torchtitan/components/quantization/nvfp4.py:280-297`）。
- NVFP4 的空 `fqns` 表示转换所有 Linear，不表示关闭。Llama recipe 显式用 `fqns=["layers"]` 保留 `lm_head`，另一个 recipe 只转换前 85% layers、保留 BF16 tail（`torchtitan/components/quantization/nvfp4.py:267-318`、`torchtitan/models/llama3/config_registry.py:120-161`）。CPU tests 固定了“不转换 lm_head”和 BF16 tail 边界（`tests/unit_tests/cpu/test_quantization.py:72-168`）。
- FusedSwiGLU 当前不是旧版 `einsum`：`w13` 是 `(2*hidden, dim)` 的标准 Linear，输出再 unflatten/unbind 并调用 fused activation；需要专门 override 才启用（`torchtitan/overrides/fused_swiglu.py:450-475`、`torchtitan/overrides/fused_swiglu.py:580-597`）。MoE 版本则显式保留大小为 2 的 gate/up 轴，EP shard expert、TP shard hidden（`torchtitan/overrides/fused_swiglu.py:600-658`、`torchtitan/overrides/fused_swiglu.py:692-703`）。

低精度是否更快不能仅由“converter 已执行”推出。shape、硬件、TorchAO/PyTorch 版本、量化/反量化开销与 BF16 tail 都要在目标集群 benchmark；源码只定义可执行边界。

### 2.5 有锚点的演进

提交 `bcc09297a1d892e0ccf6a41fc367e76aab02475b` 把 NVFP4 作为与 Float8/MXFP8 同级 converter 引入，理由是保留 stock sharding/ BF16 collective 的 leaf-swap 组合性，而把收益集中在 FP4 compute。当前代码仍把动态 sign vector 标为未来扩展，现状是各 TP rank 用同一硬编码 Hadamard basis、随机舍入 seed 各 rank 独立（`torchtitan/components/quantization/nvfp4.py:38-47`、`torchtitan/components/quantization/nvfp4.py:210-232`）。

> [!deprecated] 已失效的低精度/融合心智模型
> “低精度只有 Float8/MXFP8”“NVFP4 会把 TP wire 和 optimizer state 都变成 FP4”“FusedSwiGLU 仍是 einsum”都不符合当前 HEAD。量化首先是 Config leaf swap；NVFP4 保留 BF16 主权重/边界通信；当前 fused FFN 使用标准 Linear + 自定义激活 kernel。

### 2.6 LoRA：冻结也是配置树语义，不是训练后再过滤参数

#### ① 背景 / 问题

LoRA 不只是在 Linear 旁边增加两个低秩矩阵。标准 Trainer 会在模型并行化、物化之后从 `requires_grad` 参数构造 optimizer；如果 converter 漏冻 embedding、norm 或 composite module 的直属参数，训练仍会悄悄更新 base model。TP 下 adapter 还必须继承 base projection 的输入/输出切分，否则低秩支路与 base 输出无法相加。

#### ② 为什么选择 Config converter，而不是 build 后挂 adapter

当前实现从叶到根改写 `Module.Config`：目标 Linear 被替换为“当前 owner 的 LoRA 子类 Config”，非目标 Config 被替换为仍保持原类型关系的冻结子类（`torchtitan/components/lora.py:115-138`、`torchtitan/components/lora.py:186-227`）。直观替代是在完整 `nn.Module` 建好后遍历名字、挂 adapter 并批量切 `requires_grad`；它会晚于配置级量化、参数初始化和 sharding metadata 的确定时机，也正是更容易漏掉 composite/root 直属参数的路径。判据是：训练状态、模块实现与布局都应在 meta build 前成为同一份构造契约。

提交 `74181b1e873a7be7a7d736bc5b1fa5d760d948e1` 的正文把这一选择写成“LoRA 与量化共享 converter 列表”，并要求量化先改 owner、LoRA 再继承该 owner；提交 `b3213ef8c6445fb2fb0bca63b0a2027f5c966753` 随后修复了首版只冻 Linear、导致 embedding/norm 仍可训练的问题。这两次演进说明冻结范围不是外围 optimizer filter，而是 converter 的正确性条件。

#### ③ 当前实现 / 状态与布局

`LoRAConverter.Config` 默认 `rank=8, alpha=16`，`target_modules` 按 FQN 最后一段匹配；`None` 表示所有 Linear，空列表表示一个都不匹配（`torchtitan/components/lora.py:141-175`）。匹配项通过 `_get_lora_cls(cfg._owner)` 动态继承已经被前序 converter 选中的 Linear 类型，冻结 base 参数，建立 Kaiming 初始化的 `lora_a` 与零初始化的 `lora_b`，forward 计算 `base + alpha/rank * B(A(x))`；因此初始化时 adapter 分支为零（`torchtitan/components/lora.py:61-112`）。

TP placement 由 base weight 推导：colwise base 的 A replicated、B 跟随输出维 shard；rowwise base 的 A 跟随输入维 shard、B replicated，使低秩支路产生与 base 相同的局部/partial 输出语义（`torchtitan/components/lora.py:23-54`）。随后 converter 反向遍历整棵 Config 树：只有匹配项成为 LoRA Config，其余节点连 root/composite 的直属参数都在 build 时冻结（`torchtitan/components/lora.py:196-227`）。CPU tests 验证最终只有 `lora_a/lora_b` 可训练，并覆盖 root/composite 参数与原 Config 类型检查（`tests/unit_tests/cpu/test_lora.py:21-58`、`tests/unit_tests/cpu/test_lora.py:118-232`）。

当前 Llama recipe 先做 Float8 emulation，再做 LoRA；registry 明确拒绝“LoRA 后再量化”（`torchtitan/models/llama3/config_registry.py:165-179`、`torchtitan/models/utils.py:27-46`）。集成矩阵把该 recipe 扩展到 `spmd_types + TP2 + PP2`、8 GPU real process group，证明的是这一个组合有回归入口，不是所有量化/模型/并行组合均已验证（`torchtitan_recipes/tests/features.py:400-408`、`tests/integration_tests/features.py:284-290`）。

#### ④ 成本 / 约束 / 失败边界

- target 只按 FQN **最后一段**匹配；未匹配名字只 warning，不失败。命名漂移可能让预期 adapter 缺席（`torchtitan/components/lora.py:203-233`）。空 target 列表会把整棵模型冻结却不创建 adapter，converter 本身没有“至少一个可训练参数”的 guard。
- dynamic subclass 按 parent class 缓存并注册为可 import symbol；这是为了复用和序列化可解析，不代表任意第三方 Linear Config 都自动满足 LoRA 的 `in_features/out_features/sharding_config` 契约（`torchtitan/components/lora.py:57-71`、`torchtitan/components/lora.py:109-112`）。
- HEAD 没有 LoRA-only 或 PEFT adapter checkpoint 路径。标准 checkpointer 的 `ModelWrapper` 展平整个 `model.state_dict()`，所以 base 与 adapter 一起进入普通 DCP 状态；LoRA converter 自身也只返回替换后的 model Config（`torchtitan/components/checkpointer/base.py:86-136`、`torchtitan/components/lora.py:196-235`）。不能把历史分支里出现过的 partial-save/PEFT 工作写成当前功能。
- 已有集成项使用 Float8 **emulation** 且关闭 CUDA Graph；它不能证明真实低精度 kernel、HF export、MoE LoRA 或 core CUDA Graph 的组合成熟度（`torchtitan/models/llama3/config_registry.py:165-179`、`torchtitan_recipes/tests/features.py:400-408`）。

#### ⑤ 有锚点的演进

当前可证实的方向是把 LoRA 收敛到通用 `ModelConfigConverter + Module.Config` 协议，并补齐“只允许 adapter 可训练”的类型/测试边界；`b3213ef8c` 的修复与现行全树 `recurse=True` 遍历共同支撑这一点。源码与可达提交没有 LoRA-only export 已落地或何时落地的承诺，因此它仍应列为缺口，而不是路线图。

---

## 3. Regional compile：组件选择、区域顺序与 Async TP

### 3.1 背景与问题

整模型 compile 能给编译器更大视野，但分布式 wrapper、重复 decoder blocks、动态 MoE 路由和 FlexAttention backend 会扩大 trace/重编译/失败面。相反，逐算子 compile 边界太碎，无法消除 block 内 pointwise 和数据流开销。

### 3.2 为什么选择逐 TransformerBlock

当前选择是把重复的 TransformerBlock 作为 fullgraph 区域，同时让 FSDP wrapper 留在区域外。明显替代是“FSDP 后整模型 compile”；`apply_compile()` 的 docstring 明列它为替代方案，但当前实现遍历 `model.layers`，逐 block 调 `.compile(fullgraph=True)`（`torchtitan/distributed/compile.py:39-72`）。决定标准是让同构 blocks 复用编译结构，并把分布式生命周期与 compiled compute region 分开。

提交 `5242bdf7151d8a57ca59ea54c656165d855772be` 进一步解释了统一 regional 路线的背景：FSDP 已改为 layer-level 后，dense/MoE 可以共用 per-layer compile；MoE 所需的 scalar-output capture 对 dense model 无害，不再需要 experts-only 子边界。

### 3.3 当前调用链与状态

**配置门。** `CompileConfig` 默认关闭，`components` 默认包含 `model` 与 `loss`；`enable_async_tensor_parallel=True` 只有在 compile enabled 且 `model` 被选中时才合法（`torchtitan/config/configs.py:295-315`）。

**模型顺序。** Llama 先执行模型 sharding，随后 AC wrapper，再 `apply_compile()`，最后 `apply_fsdp_to_decoder()`；因此 compile trace 看见 AC 包装后的 block，但 FSDP 是外层分布式 wrapper（`torchtitan/models/llama3/parallelize.py:40-78`）。这条顺序不是所有模型细节的替代说明；AC/FSDP 各自机制见 sibling 页。

**区域内容。** `apply_compile()` 打开 MoE data-dependent scalar capture，并接受 AC recompute 不重放 Python forward side effects 的 eager/compile 差异，再选择 backend、逐 layer fullgraph compile（`torchtitan/distributed/compile.py:49-72`）。leaf loss 通过 `BaseLoss._maybe_compile()` 独立 compile；`ChunkedLossWrapper` 只把 compile config 传给 inner loss，`lm_head + chunk loop` 并未整体变成 loss compiled region（`torchtitan/components/loss.py:234-260`、`torchtitan/components/loss.py:513-570`）。

**FlexAttention backend。** default Inductor 直接处理 Flex region；若 outer backend 是 `aot_eager`，TorchTitan 用 `regional_inductor` 只标记 FlexAttention 子区域交给 Inductor。其它 non-Inductor backend 不会静默退化成 eager aten，而是直接失败（`torchtitan/distributed/compile.py:99-150`；选择行为由 `tests/unit_tests/gpu/test_compile_regional_inductor.py:36-62` 覆盖）。

**Async TP。** compiler pass 不再由 parallelism config 单独启动。`apply_compile()` 取 backend-aware dense TP mesh，在 exact TP group 上启用 symmetric memory，并设置 Inductor `_micro_pipeline_tp`（`torchtitan/distributed/compile.py:39-52`、`torchtitan/distributed/compile.py:75-96`）。提交 `737594746fda65a6d94dc9482ef07863a80c8588` 的正文明确记录了移动原因：旧接线没有跨 SPMD backend 选对 dense TP group；把它归入 compile 才能在编译区域建立正确 process group。

### 3.4 成本、失败与组合边界

- `compile.enable=True` 不保证 `model` 被编译；还要看 `components`。同理，只选择 `loss` 不会启用 Async TP（`torchtitan/config/configs.py:295-315`）。
- 每个 block 都以 `fullgraph=True` 提交，且 MoE 的 data-dependent scalar outputs 由全局 Dynamo 开关显式捕获（`torchtitan/distributed/compile.py:54-70`）。**知识库推断**：设计意图是守住完整 block 边界，而不是依赖 graph break 自动切成许多小图；具体失败文本仍由当前 PyTorch compiler 决定。
- `skip_fwd_side_effects_in_bwd_under_checkpoint=True` 明确接受 eager AC 与 compile AC 在 Python side-effect 重放上的差异；依赖 forward Python mutation 得到语义结果的模块不能假设二者等价（`torchtitan/distributed/compile.py:57-64`）。
- Async TP 是 compiler micro-pipeline；dist-GEMM 是模型模块显式把 all-gather/reduce-scatter 折入相邻 GEMM。后者只在 `spmd_types + Sequence Parallel` 合法，TP=1 时警告回退 stock projection（`torchtitan/models/common/config_utils.py:62-68`、`torchtitan/models/common/dist_gemm.py:53-110`）。两者不是同一开关，组合收益需 benchmark，通信细节见 [[02_engineering/02_train_frameworks/torchtitan/24_torchtitan_comm_optimizations_overlap_analysis|通信优化与 overlap]]。

### 3.5 有锚点的演进

`compile.py` 仍有两个上游 TODO：移除 `FakeTensorMode.__init__` monkeypatch，以及等 PyTorch 自动为 Async TP process group 注册 symmetric memory（`torchtitan/distributed/compile.py:23-28`、`torchtitan/distributed/compile.py:83-93`）。这表明当前接线仍承担上游缺口，不应把这些全局开关视为永久公共 API。

---

## 4. Core CUDA Graph：固定 forward+loss+backward，而非整个 train step

### 4.1 背景与问题

regional compile 优化图内程序，却不必然消除 eager Trainer 每个 microbatch 的 CPU launch 开销。CUDA Graph 要解决的是重复 replay 一段固定 kernel/collective 序列；它要求地址/shape/控制结构稳定，因此不能直接包住动态 dataloader、checkpoint、metrics 或 Python PP schedule。

### 4.2 为什么选择 combined fwd/bwd capture boundary

当前选择把 `_forward_backward_body(inputs, labels, global_valid_tokens, extra_kwargs)` 包成一段 graph；明显替代是捕获整个 `train_step()`，或只捕获 model forward。前者包含输入搬运、optimizer 和动态副作用，后者仍留下 loss/backward launches。提交 `f84224af0995debb4b32bb1a0050796ab9135c49` 把 PP 不支持解释为“需要不同 capture boundary”：应按 stage action 捕获、保持 Python schedule/P2P eager，而不是改变模型语义。

### 4.3 当前 capture/replay 状态机

1. Trainer 默认 `disable_cuda_graphs=False`；非 PP 构造时用 `wrap_with_cuda_graph()` 替换 fwd/bwd callable，该 callable内部执行 model、loss 与 `loss.backward()`（`torchtitan/config/configs.py:77-86`、`torchtitan/trainer.py:579-588`、`torchtitan/trainer.py:703-728`）。
2. wrapper 在非 NVIDIA CUDA 上警告并原样返回 eager callable；GPU manager 为所有 wrapper 共享 graph memory pool 和 stream，并在 Trainer close 时统一 teardown（`torchtitan/distributed/cudagraph.py:125-186`、`torchtitan/distributed/cudagraph.py:335-355`、`torchtitan/trainer.py:1015-1023`）。
3. 第一次调用在共享独立 stream 上 eager warmup；第二次调用在同一 pool/stream capture，然后当前调用也 replay；后续调用把动态 tensor copy 到 capture 保存的静态输入，再 replay graph（`torchtitan/distributed/cudagraph.py:237-244`、`torchtitan/distributed/cudagraph.py:292-325`）。
4. tensor 输入数、shape、dtype、device 必须保持不变；non-tensor 值必须相等。`BlockMask` 还会被专门 flatten，叶子数与 context 跨 step 变化就失败（`torchtitan/distributed/cudagraph.py:39-98`、`torchtitan/distributed/cudagraph.py:264-290`）。
5. graph output 引用静态存储，下次 replay 会覆盖。Trainer 只在日志步 clone 第一个 detached loss，并原地累加其余 loss；非日志步不保存输出（`torchtitan/distributed/cudagraph.py:335-340`、`torchtitan/trainer.py:835-848`）。提交 `2807d3f550fe27db18bd9395ba63176364eaed6d` 的正文说明这消除了每个 microbatch 无条件 clone。

### 4.4 Guards、成本与失败边界

- 配置验证拒绝 PP；EP 只接受没有 CPU synchronization 的 MinimalAsyncEP，或配置 `non_blocking_capacity_factor` 的 HybridEP（`torchtitan/trainer.py:165-196`）。CPU tests 覆盖 PP 拒绝、支持/不支持 EP dispatcher 与显式禁用 graph（`tests/unit_tests/cpu/test_config_manager.py:164-241`）。
- graph 开启时 `zero_grad(set_to_none=False)` 保留梯度 storage；这牺牲 set-to-none 的某些内存/稀疏分配行为，换取 replay 地址稳定（`torchtitan/trainer.py:774-780`）。
- 输入 tensor 虽可 copy-in，但 metadata/辅助 pytree 结构不能动态变化；“固定 token budget”不自动保证自定义模型的所有 extra kwargs 都稳定（`torchtitan/distributed/cudagraph.py:264-325`）。
- graph capture 与 `torch.compile(mode="reduce-overhead")` 各自可能使用 CUDA Graph，当前 core wrapper 明确独立；不能从 `compile.enable` 推出 core graph 开/关（`torchtitan/config/configs.py:77-86`）。

### 4.5 有锚点的演进

初始 CUDA Graph 提交把 FSDP `reshard_after_forward=always` 列为已知限制，但当前 HEAD 已修复相关 Chunked Loss 路径：wrapper 在 FSDP idle 状态显式 `lm_head.unshard()`，避免 eager warmup 的 all-gather event 泄入 capture state（`torchtitan/components/loss.py:666-714`）。提交 `04326fc8e60c27953339c81539c3083babbce226` 记录了该故障与修复，所以“CUDA Graph 永远不支持 FSDP always-reshard”已不是现行结论。源码仍把 PP 支持留在配置 guard 中，当前不能提前宣称 stage-level capture 已实现。

---

## 5. 训练步内存与更新：输入、Chunked Loss、optimizer 和 finite gate

### 5.1 背景与问题

即使 GEMM 已低精度、block 已 compile，训练仍可能被输入累积、全词表 logits、optimizer moments 和失败 step 的同步检查支配。这里的问题不是选择另一个 kernel，而是确定哪些张量何时创建、何时释放、何时允许修改参数。

### 5.2 为什么把控制留在 graph 外

当前设计先收齐一个 optimizer step 的所有 microbatch，但 tensor 保持在 CPU；逐 accumulation group 搬到 device、完成 fwd/bwd 后释放，再统一 clip/finite gate 和 optimizer update（`torchtitan/trainer.py:639-666`、`torchtitan/trainer.py:774-889`）。明显替代是一边从 loader 取数据一边更新，或把 optimizer/checkpoint 也塞进 CUDA Graph。前者会在中途耗尽时形成半个更新，后者把动态 state mutation 固定进 capture。决定标准是：完整 step 的原子性优先，图内只保留稳定 compute body。

### 5.3 当前状态与调用链

**Gradient accumulation/input staging。** Trainer 在 CPU 上预取 `gradient_accumulation_steps × num_pp_microbatches`，先统计非 ignore label，再在 DP batch mesh 上得到 device-resident global valid-token count；每组使用时才 `.to(device)`（`torchtitan/trainer.py:785-834`）。它降低 GPU 上同时存活的输入组数，但 CPU 仍要容纳完整 step 的预取数据。

**Chunked Loss。** 模型跳过 `lm_head` 并返回 hidden states，Trainer 将真实 head 注入 wrapper（`torchtitan/trainer.py:495-520`）。wrapper 沿本地 token 维把 hidden/labels 等分，逐 chunk 执行 `lm_head + inner loss`，因此一次只物化一个 chunk 的大词表 logits；它不重算 decoder activation（`torchtitan/components/loss.py:513-551`、`torchtitan/components/loss.py:575-647`）。训练时每个 detached hidden chunk 单独 backward，FP32 `GradAccumulator` 拼回完整 hidden grad，再由自定义 autograd Function把梯度送回 decoder graph（`torchtitan/components/loss.py:649-729`、`torchtitan/components/loss.py:760-817`）。

**FSDP/Chunked Loss 协调。** `lm_head` 在 chunk loop 前保持 unsharded，前 N-1 chunks 关闭 grad sync，最后一个 chunk 恢复后只触发一次合并 reduce-scatter，结束再恢复 reshard 策略（`torchtitan/components/loss.py:666-714`）。类 docstring 的“reduce-scatter per chunk”与可执行逻辑矛盾；本页按后者描述（矛盾位置 `torchtitan/components/loss.py:534-551`）。

**Optimizer。** Trainer 在 model parallelize/materialize 后才 build optimizer，使它绑定最终参数对象（`torchtitan/trainer.py:533-543`）。`OptimizersContainer` 以 FQN regex first-match-wins 分配参数；同 optimizer name 的 groups 合入一个 instance，不同 name 可共存，构造后验证所有 trainable parameter 恰好被覆盖（`torchtitan/components/optimizer/optimizer.py:40-100`、`torchtitan/components/optimizer/optimizer.py:168-243`、`torchtitan/components/optimizer/optimizer.py:270-300`）。当前 factory 只列 `Adam`、`AdamW`、`DistMuon`（`torchtitan/components/optimizer/optimizer.py:141-150`）；DistMuon 的二维 mesh/bucket/collective 见 [[02_engineering/02_train_frameworks/torchtitan/26_torchtitan_flex_shard_dist_muon_analysis|Flex Shard 与 DistMuon]]。

**Optimizer state 与有限性闸门。** implementation 默认 `fused`，实际把 `fused/foreach` kwargs 交给 PyTorch optimizer；`fused_opt_states_bf16` 用 step pre-hook 在 Adam/AdamW lazy init 前建立 BF16 moments，仍调用 PyTorch fused kernel（`torchtitan/components/optimizer/optimizer.py:102-165`、`torchtitan/components/optimizer/optimizer.py:332-371`）。所有 microbatch 完成后才 clip grad；loss finite 沿 loss/PP mesh 传播，与 world-reduced grad norm 合成 device-side `_assert_async`，通过后才等待 checkpoint staging 并 step optimizer/scheduler（`torchtitan/trainer.py:850-889`）。

### 5.4 成本、失败与组合边界

- Chunked Loss 省的是 logits 峰值，不是 decoder activation；local token length 不能整除 chunk 数就失败（`torchtitan/components/loss.py:605-623`；CPU test 在 `tests/unit_tests/cpu/test_loss.py:760-768`）。FP32 grad accumulator 自身也占一份 hidden-gradient buffer，chunk 不是免费。
- optimizer regex 规则按顺序 first-match；空匹配直接 `ValueError`，遗漏/重复最终由参数 identity 集合不等断言拦截（`torchtitan/components/optimizer/optimizer.py:168-216`、`torchtitan/components/optimizer/optimizer.py:270-280`）。
- `fused_opt_states_bf16` 只对 Adam/AdamW 注册 hook；它降低 moments dtype，不改变参数/梯度 dtype，也不等价于 full-BF16 training（`torchtitan/components/optimizer/optimizer.py:109-123`、`torchtitan/components/optimizer/optimizer.py:332-371`）。
- finite gate 避免逐 microbatch D2H，但只能报告某个范围内发生非有限，不能精确指出哪个 microbatch。提交 `da3be38c13945610ae1cfedada13a0fb1c111a20` 明确把诊断粒度作为消除同步的代价。
- `training.enable_cpu_offload` 由 FSDP policy 管理参数、梯度和 optimizer state，增加 CPU memory/传输成本；不是 `model.cpu()`（`torchtitan/config/configs.py:72-75`、`torchtitan/models/llama3/parallelize.py:68-78`）。其生命周期由 [[02_engineering/02_train_frameworks/torchtitan/11_torchtitan_fsdp_analysis|FSDP2]] 负责，本页不重复展开。

### 5.5 有锚点的演进

提交 `632f67f128c31a18ff9717e988c6ee5b5ec7499c` 引入 mixed optimizer 的直接动机，是让 router 等参数能使用与其余权重不同的 optimizer，同时仍由容器统一 checkpoint/scheduler/metrics。当前 package 路径已变为 `components/optimizer/optimizer.py`，`__init__.py` 继续导出容器、scheduler 与 helper（`torchtitan/components/optimizer/__init__.py:7-21`）；旧的单文件 `components/optimizer.py` 不再是源码权威。

---

## 6. 组合判据：优化对象、边界与被否替代

### 6.1 背景与问题

多个优化都声称“降低显存/提高吞吐”，但它们可能作用于同一 GEMM、同一 collective 或同一静态 storage。未经证据就全部开启，既可能重复工作，也可能因硬件、layout 或 capture guard 直接失败。

### 6.2 为什么按“被优化对象”而不是开关名决策

选中判据是先标出瓶颈对象和生命周期，再选择一个拥有该边界的机制。明显替代是依据功能名称叠加：quantization + compile + CUDA Graph + dist-GEMM 全开。源码并没有给出这一组合的普适胜出结论；converter、compiler pass、module backend 与 runtime capture 分属不同层，收益只能在具体 shape/topology 上验证。

### 6.3 当前组合地图

| 机制 | 优化对象 | 关键前提 | fallback/失败 |
|---|---|---|---|
| Float8 dense/MoE | GEMM 数值格式 | SM89/gfx942、16 对齐；MoE dispatcher padding | emulation 仅 eager 测试；缺能力失败 |
| MXFP8 | block-scaled GEMM | SM100、padding/kernel 对齐 | 未 compile 警告性能；缺能力失败 |
| NVFP4 | local GEMM 的动态 FP4 训练表示 | SM100、TorchAO prototype、128 对齐；TP 用 local SPMD region | BF16 wire 不变；不满足维度失败 |
| LoRA converter | 可训练参数规模与低秩 adapter compute | Config 树、target FQN、量化先于 LoRA、兼容 TP placement | 未匹配只 warning；普通 DCP 保存全模型；无 LoRA-only export |
| FusedSwiGLU | gate/up 参数与 pointwise 边界 | 对应 exact override、兼容 sharding/checkpoint | 不启 override 即 stock FFN |
| regional compile | 重复 block/leaf loss 图 | component 被选中、fullgraph 可捕获 | graph break/不支持 backend 显式失败 |
| compiler Async TP | TP collective/GEMM micro-pipeline | model compile + dense TP mesh + symmetric memory | TP 不启用则 setup no-op |
| dist-GEMM | 显式 module 内 collective+GEMM | `spmd_types` + SP + TP | TP=1 警告回退；其它前提失败 |
| core CUDA Graph | eager launch 与静态 buffer 复用 | NVIDIA CUDA、固定 metadata、无 PP/host sync | 非 NVIDIA eager fallback；guard 组合失败 |
| Chunked Loss | 全词表 logits 峰值 | local token 可整除 chunks | 不减少 decoder activation |
| fused BF16 optimizer states | Adam moments | Adam/AdamW + fused kernel | 其它 optimizer 不注册该 BF16-state hook |

### 6.4 失败边界与验证责任

- 配置/构造期 guard 证明的是“可以接线”，不是“更快”。Float8 auto-filter、小 BF16 tail、chunk count、compile backend 和 optimizer state dtype 都需要目标模型/硬件 benchmark。
- 量化保持 sharding skeleton 不意味着每个 opaque kernel 都能由 DTensor/SPMD 自动推导；NVFP4 正因 opaque autograd op 才显式建立 local-map/type contract（`torchtitan/components/quantization/nvfp4.py:67-82`、`torchtitan/components/quantization/nvfp4.py:113-157`）。
- core CUDA Graph 与 regional compile 可同时存在，但两者各自的 graph/capture guard 都必须满足；“compile 通过”不证明 replay 输入稳定。
- AC、Chunked Loss、FSDP offload分别优化 decoder activation、logits、参数/梯度/state 驻留，不能互相替代。机制边界分别见 [[02_engineering/02_train_frameworks/torchtitan/22_torchtitan_ac_analysis|Activation Checkpointing]] 与 [[02_engineering/02_train_frameworks/torchtitan/11_torchtitan_fsdp_analysis|FSDP2]]。

### 6.5 有锚点的趋势与旧断言纠正

当前 HEAD 支持的、应替换旧知识的结论是：

| 旧断言 | 当前事实 |
|---|---|
| 低精度只有 Float8/MXFP8 | 已有 NVFP4 converter；BF16 主权重与 BF16 TP boundary 保持不变 |
| FusedSwiGLU 使用 einsum | 当前是标准 `w13` Linear + fused SiLU×mul Triton op |
| core Trainer 没有 CUDA Graph | 默认未禁用，包住 forward+loss+backward；PP/host-sync 路径受 guard |
| compile 与 CUDA Graph 是同一开关 | 二者配置、区域、状态机独立，可能同时存在 |
| Async TP 属于 parallelism 入口 | 已移到 `CompileConfig.enable_async_tensor_parallel` 并由 `apply_compile()` 配置 |
| FSDP always-reshard 永久阻止 CUDA Graph | Chunked Loss 已显式 idle-unshard 修复该 capture isolation 问题 |
| optimizer 源码仍在 `components/optimizer.py` | 当前实现位于 `components/optimizer/` package；容器只封装 PyTorch/DistMuon factories |
| LoRA 只是 build 后过滤 optimizer 参数，且已有 PEFT/adapter-only 保存 | 当前在 Config 树中替换目标并冻结其余节点；普通 DCP 仍展平完整 model state |

源码没有给出“dist-GEMM 将取代 compiler Async TP”或“NVFP4 将取代 Float8/MXFP8”的路线图。本页不做此类外推；它们是不同硬件/shape/调度假设下的并存策略。

---

## Related Pages

- [[02_engineering/02_train_frameworks/torchtitan/01_torchtitan_trainer_quickstart|Trainer 生命周期]] —— 本页所有优化在标准 Trainer 中的构造顺序、单步边界与异常收尾。
- [[02_engineering/02_train_frameworks/torchtitan/11_torchtitan_fsdp_analysis|FSDP2 参数与状态生命周期]] —— MixedPrecisionPolicy、CPU offload、reshard 与 optimizer 后建原因。
- [[02_engineering/02_train_frameworks/torchtitan/12_torchtitan_tp_analysis|Tensor Parallel]] —— 参数/activation placement、Loss Parallel 与 compile Async TP 的 TP 语义。
- [[02_engineering/02_train_frameworks/torchtitan/22_torchtitan_ac_analysis|Activation Checkpointing]] —— decoder activation 的重计算预算，与 Chunked Loss 优化不同存活对象。
- [[02_engineering/02_train_frameworks/torchtitan/24_torchtitan_comm_optimizations_overlap_analysis|通信优化与 overlap]] —— symmetric memory、dist-GEMM、Async TP、EP overlap 的通信侧机制。
- [[02_engineering/02_train_frameworks/torchtitan/26_torchtitan_flex_shard_dist_muon_analysis|Flex Shard 与 DistMuon]] —— 本页仅链接的分布式 optimizer mesh/bucket/collective。
- [[02_engineering/02_train_frameworks/torchtitan/27_torchtitan_graph_trainer_compiler_runtime_analysis|GraphTrainer 编译运行时]] —— 对照 core Trainer 的局部 graph：更大 joint graph 与内存策略的实验控制面。
