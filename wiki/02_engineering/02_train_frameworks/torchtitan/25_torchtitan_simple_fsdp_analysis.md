---
title: "SimpleFSDP：把参数物化与梯度规约放进 joint graph"
---

# SimpleFSDP：把参数物化与梯度规约放进 joint graph

> **论点**：GraphTrainer 的 SimpleFSDP 不是 eager FSDP2 的精简 wrapper，而是一个编译器可见性契约：分片 DTensor 是持久状态，参数 property 在 forward 内显式完成 unshard，DTensor autograd 在 joint forward/backward 图中显式产生梯度规约；GraphTrainer 因而可以跨前后向统一做保存/重算、分桶、重排与可选重叠。换来的不是“免费加速”，而是更严格的 trace、布局和组合边界。
>
> **代码基准**：pytorch/torchtitan `main` @ `a3168782c9a3a2e40afbd0de114818b96e2bda6e`（2026-08-26）
>
> **本页回答**：为什么选择 graph-visible collectives；参数状态与 joint trace 如何相接；reshard、bucket、reorder 如何工作；HSDP、EP、PP、compile 到底支持到哪里；哪些正确性/性能结论已有测试，哪些仍是空白。
>
> **兄弟页边界**：eager FSDP2 生命周期属于 [[02_engineering/02_train_frameworks/torchtitan/11_torchtitan_fsdp_analysis|TorchTitan FSDP2 分析]]；GraphTrainer 的 tracer、artifact 与执行器属于 [[02_engineering/02_train_frameworks/torchtitan/27_torchtitan_graph_trainer_compiler_runtime_analysis|GraphTrainer 编译器与运行时]]；本页只追 SimpleFSDP 如何把 collective 暴露给这些机制。

---

## 1. Overview：这里要解决的是“编译器看不见 FSDP 调度”

### ① 背景 / 问题

Eager FSDP2 擅长在运行时用参数状态机、hook 与通信 stream 管理 unshard、reshard 和梯度规约；但 GraphTrainer 想在一张 forward + loss + backward 图上统一改变通信顺序、保存边界和计算重叠。若 collective 仍藏在 wrapper 的运行时控制面，FX pass 就没有可重排的节点。

当前实验的训练闭包直接执行 model、loss 与 `torch.autograd.grad(loss, params)`，返回 loss 和全部参数梯度；模型参数由 tracer 作为静态状态输入，而不是普通用户输入（`torchtitan/experiments/graph_trainer/trainer.py:75-103`）。这定义了问题尺度：不是只编译单个 module forward，而是捕获一整个 fwd/bwd step。

### ② 为什么选择 SimpleFSDP，而不是直接复用 eager FSDP2

选择标准是 **collective 是否能成为 joint FX graph 中可识别、可改写的普通节点**。SimpleFSDP 把参数读取写成 DTensor `redistribute`，让 autograd 从 placement 推导反向规约；显而易见的替代方案 eager FSDP2 则把成熟的动态生命周期留在运行时，但不能直接给本实验的全图 pass 同等显式控制。

演进证据与这个判据一致：提交 `c8d2d71812b3ad51a001277e4f7ae388cff35647` 合并 simple_fsdp 与 compiler_toolkit，commit body 把目标写成对性能、数值与可调试性的显式 compiler-stack 控制。这个历史动机不能证明当前性能更快；它只解释为何本实验接受更窄的静态契约。

### ③ 实现思路与可追踪调用链

| 层次 | 当前对象 | 图中可见的结果 |
|---|---|---|
| 持久状态 | DP-sharded / replicated DTensor 参数 | 参数 placeholder 与 placement |
| 参数访问 | 动态 property 调用 `ReplicateComputation` | all-gather / wait / reconstruction，或无通信复制 |
| backward | `Partial(sum)` gradient placement | reduce-scatter；replicate mode 下是 all-reduce |
| GraphTrainer | fwd + loss + `autograd.grad` joint trace | forward AG、backward AG、backward gradient collective |
| graph passes | pattern、bucket、reorder、memory policy | 去重、保存/重算、预取与可选跨流重叠 |

当前 Llama 接线是 FQN 标注 → TP → `apply_simple_fsdp` → compile；注释还说明 degree 1 也必须包装，以便参数 mixed-precision cast 生效（`torchtitan/experiments/graph_trainer/llama3/parallelize.py:53-70`）。DeepSeek V3 是 TP/EP → SimpleFSDP → 可选 eager EP chunking → compile（`torchtitan/experiments/graph_trainer/deepseek_v3/parallelize.py:59-77`）。

Quick Start 心智模型：

```text
inner TP/EP 参数（可选）
  -> data_parallel：叠加 outer DP placement，保存分片状态
  -> 参数 property：计算前 materialize 成 DP-replicated 参数
  -> forward + loss + autograd.grad：同时捕获计算与反向规约
  -> memory / FSDP / EP / TP passes
  -> run_traced：执行显式 backward 图，再把 grads 累积回 live params
```

首个 AOT step 在训练 context 中 trace、构造并应用 pass，随后执行图并把返回的 grads 手工累积到 live parameters（`torchtitan/experiments/graph_trainer/trainer.py:206-262`）。

### ④ 约束、代价与失败边界

这条路径属于 `torchtitan/experiments/graph_trainer`，不是 core Trainer 的 FSDP 实现。`GraphTrainer.forward_backward_step` 在启用普通 PP 或 compile mode 不是 `aot_fx_trace` 时回退父 Trainer（`torchtitan/experiments/graph_trainer/trainer.py:136-148`）。因此“GraphTrainer 总是 joint trace”与“SimpleFSDP 已替换 core eager FSDP2”都不成立。

GraphTrainer 配置转换还强制 `spmd_backend="partial_dtensor"`；紧邻 TODO 明说 CP placements 在这个后端不会应用，须等迁到 `spmd_types`（`torchtitan/experiments/graph_trainer/configs.py:239-258`）。全局默认已是 `spmd_types`，不等于本实验也已迁移（`torchtitan/config/configs.py:174-180`）。

### ⑤ 发展趋势

**有锚点的推断**：`to_graph_trainer_config()` 的 CP TODO 表明迁到 `spmd_types` 是明确缺口，但源码没有承诺时间或最终接口；不能据此写成近期路线图（`torchtitan/experiments/graph_trainer/configs.py:252-258`）。

---

## 2. 参数状态：分片存储，属性读取时物化

### ① 背景 / 问题

若把完整参数长期保存在 module 上，编译器虽然容易捕获计算，却失去 FSDP 的显存收益；若由外部 hook 临时替换参数，图又看不到物化过程。SimpleFSDP 需要同时保留“持久分片”与“访问时完整参数”两种视图。

### ② 为什么选择动态 property，而不是 forward hook

决定性标准仍是参数读取必须发生在被 trace 的 Python 计算路径内。实现没有使用 `nn.utils.parametrize` 或 forward hook，而是缓存动态 `SimpleFSDP*` 子类，把每个参数名替换成读取 `_parameters[pn]` 后运行 parametrization 的 property（`torchtitan/experiments/graph_trainer/simple_fsdp.py:130-164`）。

替代方案 hook 更接近 eager FSDP2 的运行时状态机，但会重新引入图外时序；property 的代价则是动态类、pickle/state-dict 兼容和全局启停状态都要由实验自己负责。

### ③ 实现、状态与布局

`data_parallel()` 将 mode 映射为持久 placement：`replicate` 使用 `Replicate()`，`fully_shard` 使用 `Shard(shard_dim)`，`hybrid_shard` 使用 `[Replicate(), Shard(shard_dim)]`，且 hybrid 强制要求 2D mesh（`torchtitan/experiments/graph_trainer/simple_fsdp.py:253-272`）。

它遍历 module 的直接参数：普通 Tensor 走 `distribute_tensor`，已有 TP/EP DTensor 走 `_distribute_dtensor`；随后重新注册 `nn.Parameter`，并跳过已含 `SimpleFSDP` 类名的 module，避免二次包装（`torchtitan/experiments/graph_trainer/simple_fsdp.py:274-293`）。最后为各 module 安装 `ReplicateComputation` property（`torchtitan/experiments/graph_trainer/simple_fsdp.py:295-318`）。

`state_dict` 路径直接访问 `_parameters`，不会触发 property 物化；源码同时留有 DCP `get_model_state_dict` 仍可能触发 getter 的 TODO（`torchtitan/experiments/graph_trainer/simple_fsdp.py:135-143`）。动态类被放进 `sys.modules`，以供 pickle/GraphPickler 找回（`torchtitan/experiments/graph_trainer/simple_fsdp.py:145-164`）。

即使 NGPU=1，包装也有语义：测试验证底层 `_parameters["weight"]` 保持 FP32，property 读取和 forward 输出是 BF16（`torchtitan/experiments/graph_trainer/tests/test_simple_fsdp.py:35-76`）。提交 `533795ee330fed4dca25fd777940e2968b56927c` 的 commit body 明确把无条件包装归因于单卡 mixed-precision parity，而非单卡通信。

### ④ 约束、代价与失败边界

`disable_active_parametrization()` 只为 forward 外的初始化、检查与调试暂时返回原始分片参数，并用 `finally` 恢复全局开关（`torchtitan/experiments/graph_trainer/simple_fsdp.py:29-39`；`torchtitan/experiments/graph_trainer/simple_fsdp.py:239-247`）。它不是训练期“关闭 FSDP 通信”的配置。

动态 property 也意味着直接通过公开属性观察参数时看到的是物化/转 dtype 后的视图；若工具需要持久 shard，必须走 `_parameters` 或受支持的 state 路径。DCP TODO 说明这部分互操作仍没有被源码宣布为完全解决。

### ⑤ 发展趋势

**有锚点的推断**：DCP TODO 指向 state-dict 工具避免 getter 的后续工作，但当前没有实现或兼容承诺（`torchtitan/experiments/graph_trainer/simple_fsdp.py:138-143`）。

---

## 3. Collective 状态机：unshard 与 reduce 为什么会进入 backward 图

### ① 背景 / 问题

仅把 forward all-gather 暴露给 FX 还不够；要在 joint graph 内调度反向，梯度从 replicated 计算视图回到持久 shard 的规约也必须由可追踪算子表达。

### ② 为什么选择 DTensor placement autograd

SimpleFSDP 不手写 backward hook，而让 `redistribute` 的 placement 变化描述数学关系：计算 placement 全是 `Replicate`，本地计算视图的 gradient placement 全是 `Partial(sum)`（`torchtitan/experiments/graph_trainer/simple_fsdp.py:167-185`）。显而易见的替代是手写 gradient hook 发 collective；它更直接，却又把反向时序藏回 Python runtime。

这里的判据是“同一状态转换既能执行，也能由 autograd/FX 看见”，不是减少 API 数量。

### ③ 实现与真实调用链

纯 DP 路径先把持久 DTensor `redistribute` 到全 `Replicate`，同时指定 `forward_dtype` 与 `backward_dtype`，再 `to_local(grad_placements=Partial(sum))`（`torchtitan/experiments/graph_trainer/simple_fsdp.py:225-231`）。由此得到：

- `Shard -> Replicate` 的 forward 是 all-gather，backward 将 Partial 梯度规约回 Shard，即 reduce-scatter；
- `Replicate -> Replicate` 没有 forward unshard，backward 的 Partial 梯度回到 replicated state，即 all-reduce；
- hybrid placement 在 replicate 与 shard 两个 DP 维上组合规约。

这三类 DDP/FSDP/HSDP 语义由实现注释直接列出（`torchtitan/experiments/graph_trainer/simple_fsdp.py:187-193`），具体 dtype 与 to-local/rewrap 路径位于 `torchtitan/experiments/graph_trainer/simple_fsdp.py:194-231`。

有 TP/EP 时，getter 先把组合 DTensor 的 local shard 重包到纯 DP mesh，执行 DP redistribute，再将结果按 non-DP mesh 重新包装，因此计算仍保留 inner TP/EP DTensor 语义（`torchtitan/experiments/graph_trainer/simple_fsdp.py:194-224`）。

包装已有 inner DTensor 时，`_distribute_dtensor()` 拼接 outer DP 与 inner mesh；若 outer 与 inner 同时 shard 同一 tensor dim，则 outer placement 用 `_StridedShard(split_factor=inner shard count)` 表达嵌套切分（`torchtitan/experiments/graph_trainer/simple_fsdp.py:48-98`）。它先分发 local tensor，再以拼接 mesh 和组合 placements 重建 DTensor（`torchtitan/experiments/graph_trainer/simple_fsdp.py:100-127`）。

### ④ 约束、代价与失败边界

实现断言 non-DP mesh 不超过两维，当前只覆盖 DP + TP、DP + EP 或 DP + EP + TP 这种 inner 维数量（`torchtitan/experiments/graph_trainer/simple_fsdp.py:187-224`）。这不是任意 N-D SPMD 通用层。

旧心智模型中的局部 `full_dtensor=True` 已不存在：`data_parallel()` 与 `ReplicateComputation` 当前签名没有该开关，纯 DP 路径总会 `to_local`，有 inner mesh 才重包为 non-DP DTensor（`torchtitan/experiments/graph_trainer/simple_fsdp.py:167-185`；`torchtitan/experiments/graph_trainer/simple_fsdp.py:253-259`）。提交 `601cf4d2304e4cfbb05e7b2865bb116a61d81a94` 是删除旧全局 full-DTensor 后端的演进证据，不能当现行可配模式。

### ⑤ 发展趋势

本单元没有当前 TODO 承诺扩展到任意 mesh 维数；因此不把“将支持 N-D”写成趋势。

---

## 4. Joint trace：backward hook 被替换为显式 graph nodes

### ① 背景 / 问题

参数 property 只保证 collective 出现在 Python 执行中；还需要 tracer 同时展开 backward，并在后续 step 用 live 参数执行已变换的图，否则 pass 操作的只是一次性 fake graph。

### ② 为什么选择一次捕获 fwd/loss/bwd

分开 trace forward 与 backward 的明显替代方案更简单，但 memory policy 无法统一判断“前向保存 unshard 输出”还是“反向重做 unshard”，bucket 也容易跨方向误配。当前设计选择 joint graph，以 forward/backward metadata 区分方向并统一调度。

提交 `9ed1a028dfe5b110904439b6b1f9919ffa4f9090` 的 commit body 正面解释了这一选择：旧的两个方向分开 bucketing 被一个 joint scheduler 取代，SAC recompute 节点仍凭 backward metadata 路由到反向。

### ③ Tracer、backward nodes 与 runtime

`minimal_fx_tracer` 展开 module/optimizer state，检查用户 pytree leaf，只接受 Tensor 与有限 primitive；wrapper subclass 上被标 dynamic 的维度会报错（`torchtitan/experiments/graph_trainer/make_fx_tracer.py:390-431`）。随后它 unwrap tensor subclass、fakeify inputs，在 patched backward 和临时 state reparameterization 下执行 non-strict `make_fx`（`torchtitan/experiments/graph_trainer/make_fx_tracer.py:432-505`）。

Tracer 把 forward metadata 复制到相应 backward node，最后返回含 graph、state FQN/spec 与重包装信息的 `TracedResult`（`torchtitan/experiments/graph_trainer/make_fx_tracer.py:507-526`）。SimpleFSDP 的 property 调用发生在这段闭包内，所以 all-gather/reduce-scatter/all-reduce 是 graph nodes，而不是 Trainer 事后插入的 hook。

`run_traced` 每次从 live module 重新提取 state，并在 `torch.no_grad()` 下执行：因为图里已有显式 backward，再让 eager autograd 记录会产生冗余图并持有中间值（`torchtitan/experiments/graph_trainer/make_fx_tracer.py:532-553`；`torchtitan/experiments/graph_trainer/make_fx_tracer.py:566-608`）。

### ④ 约束、代价与失败边界

运行时 FQN/input-spec validation 默认关闭以避免开销，调用者必须保持 kwargs 的 trace-time 顺序（`torchtitan/experiments/graph_trainer/make_fx_tracer.py:555-559`）。这比 eager FSDP2 的动态 module 执行契约更脆弱。

Optimizer 当前不在这张 joint graph 中：`TracedResult` 的源码 TODO 明说尚未 trace optimizer state（`torchtitan/experiments/graph_trainer/make_fx_tracer.py:312-323`）。因此“forward + backward + optimizer 全步编译”是过度表述。

### ⑤ 发展趋势

**有锚点的推断**：optimizer-state TODO 表示它是已知缺口；但当前 GraphTrainer 仍由 Trainer 在图外执行 optimizer，不能把 TODO 视为功能承诺（`torchtitan/experiments/graph_trainer/make_fx_tracer.py:312-323`）。

---

## 5. Reshard：wrapper 标志变成保存 / 重算决策

### ① 背景 / 问题

FSDP 的核心交换是显存对通信：保留 forward 物化出的完整参数可避免 backward 再次 all-gather，但占用更多峰值显存；释放它则要在 backward 重算 unshard。

### ② 为什么交给 memory policy，而不是 wrapper 尾部立即 reshard

SimpleFSDP 本身只表达状态转换，不接收 `reshard_after_forward` 参数。GraphTrainer 已拥有整张 fwd/bwd 图和 SAC 决策，因此把“保留哪个 reconstruction 输出”统一交给 memory policy，比 property 在 forward 末尾不可见地丢弃参数更符合全图判据。

明显替代方案是照 eager FSDP2 由 wrapper 状态机立即 reshard；它的运行时语义成熟，但不会自然与 GraphTrainer 的 rematerialization pass 合并。

### ③ 当前实现与图形差异

默认 memory policy 读取 `fsdp_reshard_after_forward`：不 reshard 时，强制保存 FSDP unshard 输出；需要 reshard 时，不强制保存，使 SAC/rematerialization 可在 backward 再执行 unshard（`torchtitan/experiments/graph_trainer/memory_policy.py:344-363`）。

pattern helper 把 unshard 定义为 parameter shard → all-gather → wait → reconstruction chain，并选择最后一个非-view reconstruction 作为输出；没有 all-gather 的参数直接对应 placeholder（`torchtitan/experiments/graph_trainer/fsdp_patterns.py:147-190`）。

真实 GraphTrainer 测试固定了图序：`never` 只出现按层正序的 forward AG；`always` 还出现按层逆序的 backward AG（`torchtitan/experiments/graph_trainer/tests/test_passes.py:1960-2008`）。这比“reshard=True 会释放参数”更精确：本实验观测的是保存边界和 backward 是否重新物化。

### ④ 约束、代价与失败边界

GraphPP 明确拒绝 `fsdp_reshard_after_forward="always"`，只接受 `default/never`，并要求 `aot_fx_trace` 与 runtime PP schedule（`torchtitan/experiments/graph_trainer/graph_pp/pipeline.py:40-63`）。因此普通 GraphTrainer 的 memory policy 语义不能无条件外推到 GraphPP。

Pattern matcher 依赖当前 DTensor trace 形状；源码 TODO 希望上游以后直接标注 unshard/reduce region（`torchtitan/experiments/graph_trainer/fsdp_patterns.py:175-190`；`torchtitan/experiments/graph_trainer/fsdp_patterns.py:202-236`）。trace 形状变化可能使结构识别失败，这是显式的维护成本。

### ⑤ 发展趋势

**有锚点的推断**：上游 region annotation TODO 指向更稳健的边界标记，但当前所有 pass 仍必须按结构匹配，不能提前按“已有 annotation”描述。

---

## 6. Bucket、reorder 与 overlap：默认项和 opt-in 必须分开

### ① 背景 / 问题

单个参数一条 AG/RS chain 会产生小 collective；只做 bucket 又可能把通信放错时机。GraphTrainer 需要同时解决 payload 聚合、forward/backward 方向隔离、参数顺序、launch/wait 移动和 stream 归属。

### ② 为什么选择编译 pass，而不是 ParamGroup 内置调度

选中路线是先保留细粒度 graph nodes，再由 joint scheduler 按 module/FQN 与方向聚合。替代方案是在 SimpleFSDP wrapper 内预先建立 ParamGroup；那样更像 eager FSDP2，但 pass 看不到可按实际 joint graph/SAC 结果重组的细节。

决定性标准不只吞吐，还包括数值顺序。提交 `2f1ca1d6b18212427419e4dd0987f80a642fa2bf` 的 commit body 说明：FX 执行序与 eager FSDP2 managed-parameter 注册序不同，会改变 NCCL bucket 内 bf16 reduction offset 与累加顺序；当前实现因此按 traced state FQN 恢复 eager packing order，并在 64/128/256 GPU 做过该提交级验证。

### ③ 默认 pipeline 与真实重排

启用 graph passes 后，pipeline 先做 DCE/canonicalize 与重复 unshard chain 去重，再做 memory/offload/SAC、EP 隔离，随后 FSDP bucket/reorder，最后进入 EP/TP/Inductor 阶段（`torchtitan/experiments/graph_trainer/passes.py:201-209`；`torchtitan/experiments/graph_trainer/passes.py:255-349`）。Precompile artifact 已完成 compile-time transforms，训练时默认只追加 runtime CUDA Graph pass（`torchtitan/experiments/graph_trainer/passes.py:410-448`）。

`deduplicate_fsdp_unshard_chains_pass` 合并同一 flat parameter 因多次 getter 产生的重复 AG/wait/reconstruction chain（`torchtitan/experiments/graph_trainer/fsdp_passes.py:74-118`）。

默认 joint scheduler 分别 bucket forward AG、backward AG 与 backward RS，不让 AG 配对跨过前后向边界（`torchtitan/experiments/graph_trainer/fsdp_passes.py:543-593`）。其重排维护独立 forward/backward AG buffer；RS 正向遍历以延迟 wait，AG 逆向遍历以提前 launch 并把 wait 留在首次依赖前（`torchtitan/experiments/graph_trainer/fsdp_passes.py:398-521`）。

两种 overlap 是 opt-in：

- `enable_fsdp_ag_rs_overlap=False` 默认关闭；开启后把 AG 重派到同 ranks 的额外 PG，而每个 NCCL PG 有独立 stream，且重派必须发生在 bucketing 前（`torchtitan/experiments/graph_trainer/configs.py:149-153`；`torchtitan/experiments/graph_trainer/fsdp_passes.py:136-139`；`torchtitan/experiments/graph_trainer/fsdp_passes.py:198-228`）。
- `enable_fsdp_dense_region_overlap=False` 默认关闭；开启后 pass 把 AG/RS launch 移进相邻 dense region，保留 wait 到依赖边界（`torchtitan/experiments/graph_trainer/configs.py:155-163`；`torchtitan/experiments/graph_trainer/fsdp_passes.py:1008-1040`）。

提交 `52a292d2977690d407bd81781de932f7f7dc56c5` 的 commit body 证实 AG/RS extra-PG overlap 从无条件列表移到默认 false 的 config gate；旧页若把它写成默认行为，已被当前代码推翻。

### ④ 约束、代价与失败边界

Dense-region scheduler 只和 `layers.*.moe` graph chunking 组合；其他 EP overlap 形态会 warning 并禁用它（`torchtitan/experiments/graph_trainer/passes.py:302-335`）。strict scheduler 无法完成所有合法移动时会抛错，不会静默接受部分重排（`torchtitan/experiments/graph_trainer/fsdp_passes.py:1301-1334`）。

最重要的 HSDP 边界是：`fsdp_patterns` 能识别 reduce-scatter、all-reduce 与 HSDP 的 AR→RS 尾链，供 GraphPP 切分（`torchtitan/experiments/graph_trainer/fsdp_patterns.py:202-236`）；但当前默认 joint scheduler 的文档与 RS-delay matcher只明确处理 bucketed AG/RS（`torchtitan/experiments/graph_trainer/fsdp_passes.py:426-464`；`torchtitan/experiments/graph_trainer/fsdp_passes.py:543-593`）。

旧页引用的 `b94c11a63`、`63a758ccc` 不在当前 baseline 的祖先链上；它们不能证明 HEAD 已有 HSDP all-reduce bucketing/reorder。当前源码也没有相应 pass-level HSDP AR 测试。正确结论是“结构识别覆盖 AR→RS，默认 scheduler 的 AR 优化证据不足”。

### ⑤ 发展趋势

本单元只有 pattern annotation TODO 可支持方向性推断；没有当前祖先提交或 TODO 承诺 HSDP AR bucketing，所以不把非祖先实验提交包装成趋势。

---

## 7. HSDP、EP、PP 与 compile：支持不是一个布尔值

### ① 背景 / 问题

“SimpleFSDP 能构造某个 mesh”只证明状态布局能建立，不证明 joint trace、pass、数值、GraphPP runtime 和端到端 CI 都覆盖同一组合。组合判断必须逐层拆开。

### ② 为什么采用分层判据，而不是一张支持表打勾

本页用四级证据：构造 guard → graph pattern/pass → 数值单测 → integration matrix。显而易见的替代是一看到配置项或 mesh branch 就写“支持”；它会把 dormant code、skip 测试和生产验证混为一谈。

### ③ 当前组合状态与调用链

| 组合 | 当前 source fact | 证据边界 |
|---|---|---|
| HSDP storage | `hybrid_shard` = replicate + shard；`apply_simple_fsdp` 在 replicate 与 shard/CP 同时存在时选择它 | `torchtitan/experiments/graph_trainer/simple_fsdp.py:253-272`; `torchtitan/experiments/graph_trainer/common_utils.py:421-448` |
| HSDP numerics | toy Linear 对比 eager FSDP2、SimpleFSDP eager 与 `torch.compile(aot_eager)`，20 steps exact loss equality | `torchtitan/experiments/graph_trainer/tests/test_numerics.py:677-818`；不是 full GraphTrainer pass/perf 测试 |
| EP + E-FSDP | routed experts 先在 `[dp_replicate, efsdp]` 或 `[efsdp]` 单独包装，必要时专家权重沿 dim 1 shard，再包装整模型 | `torchtitan/experiments/graph_trainer/common_utils.py:450-487` |
| expert bucket | 默认计划在 `efsdp` 开启时把 MoE dense 与 routed experts 分为不同 bucket | `torchtitan/experiments/graph_trainer/common_utils.py:349-387` |
| ordinary PP | `GraphTrainer.forward_backward_step` 直接回退父 Trainer | `torchtitan/experiments/graph_trainer/trainer.py:136-148` |
| GraphPP | 独立构建 fw/full_bw/bw_di/bw_dw/unshard/reduce_grad graph modules | `torchtitan/experiments/graph_trainer/graph_pp/graph_builder.py:760-768` |
| compile | `aot_fx_trace` 不先包 model；真实 fwd/bwd 在 Trainer trace。JIT 已 deprecated | `torchtitan/experiments/graph_trainer/compile.py:87-119` |
| CP | GraphTrainer 强制 partial-DTensor，CP placement 当前不应用 | `torchtitan/experiments/graph_trainer/configs.py:252-258` |

GraphPP 在切图前无条件做 normalize/DCE/dedup，再按 `enable_passes` 决定是否应用 GraphTrainer passes（`torchtitan/experiments/graph_trainer/graph_pp/graph_builder.py:779-832`）。提交 `0ea4b78c71153695f609073ec152999aab8fa2f6` 的 commit body 说明其选择了显式 UNSHARD 与 REDUCE_GRAD 子图，并保证 no-FSDP backward 的 pre-reduce cast 仍处在正确边界；这不是 ordinary PP 自动获得 joint graph 的证据。

当前 integration matrix 中，AOT Llama FSDP+TP 有启用项，而 CP 组合因 GraphTrainer 尚未迁移 `spmd_types` 被禁用（`torchtitan/experiments/graph_trainer/tests/integration_tests.py:195-232`）。DeepSeek 的多项 FSDP+TP+EP/overlap 组合因 EP all-to-all instability 或 cyclic bucket 被禁用，但 GraphPP FSDP4+EP2 的多个 schedule 有启用项，MinimalAsyncEP FSDP4+EP4 也有启用项（`torchtitan/experiments/graph_trainer/tests/integration_tests.py:407-596`）。Qwen3 MoE FSDP+TP+EP 有启用项但关闭 cudagraph（`torchtitan/experiments/graph_trainer/tests/integration_tests.py:622-637`）。

### ④ 约束、代价与失败边界

EP graph chunking 在 TP degree 大于 1 时直接报错（`torchtitan/experiments/graph_trainer/passes.py:220-230`）。Integration matrix 中 disabled/skip 项是当前失败边界，不应被“代码有 branch”覆盖。

GraphPP 与普通 GraphTrainer 是两个入口；GraphPP 的组合成功不能反向证明普通 PP 走了 GraphTrainer joint step。类似地，HSDP toy convergence 不能证明默认 joint scheduler 已对 HSDP all-reduce 做 bucket/reorder，更不能证明吞吐收益。

### ⑤ 发展趋势

**有锚点的推断**：CP disabled 注释与 conversion TODO 都指向 `spmd_types` migration；DeepSeek disabled cases 则记录当前运行不稳定。二者都是修复方向的证据，不是支持承诺（`torchtitan/experiments/graph_trainer/tests/integration_tests.py:22-24`；`torchtitan/experiments/graph_trainer/tests/integration_tests.py:426-503`）。

---

## 8. 正确性与性能证据：能证明什么，不能证明什么

### ① 背景 / 问题

图级通信重排最危险的失败不是崩溃，而是梯度归属、bucket offset、dtype 或保存边界悄悄变化。另一方面，一张 profile 截图也不能替代稳定的性能矩阵。

### ② 为什么把正确性 parity 放在吞吐主张之前

当前设计把参数 order 对齐 eager FSDP2，说明数值顺序本身就是决策标准，而非只看理论 overlap。明显替代是按 FX 执行序最大化局部方便性；提交 `2f1ca1d6b...` 已记录它会改变多机 bf16 NCCL accumulation order，因此被否决。

### ③ 当前已有验证

多 GPU trace 测试使用 `partial_dtensor`，对 Llama3、Qwen3 与 DeepSeek V3 检查图中确有 AG/RS，并在 5 steps 上比较 eager 执行同一个 SimpleFSDP model 与 `run_traced` 的 loss/grad exact equality（`torchtitan/experiments/graph_trainer/tests/test_trace_module.py:1813-1966`）。它验证 trace/runtime parity，但参考侧不是 core eager FSDP2；GPT-OSS case 仍因 dtype mismatch skip（`torchtitan/experiments/graph_trainer/tests/test_trace_module.py:1968-1969`）。

独立 numerics 测试在 toy Linear 上比较 eager FSDP2、SimpleFSDP 与 compiled SimpleFSDP 的 replicate/fully_shard/hybrid_shard 收敛，并逐 step `torch.equal`（`torchtitan/experiments/graph_trainer/tests/test_numerics.py:677-818`）。它补了 eager FSDP2 baseline，但不是大模型或默认 pass pipeline。

真实 GraphTrainer pass 测试覆盖 `never/always` 的 AG 层序与 backward re-AG（`torchtitan/experiments/graph_trainer/tests/test_passes.py:1832-2008`）。另一个测试只比较 `loss.backward` 与 `autograd.grad` 在 SimpleFSDP Llama 上的精确峰值字节相等（`torchtitan/experiments/graph_trainer/tests/test_trace_module.py:2163-2258`）；它不是 SimpleFSDP 相对 eager FSDP2 的显存优势证明。

### ④ 约束、代价与失败边界

当前 HEAD 没有统一的 eager FSDP2 vs GraphTrainer 吞吐/峰值显存/HSDP pass-level benchmark。extra PG 与 dense-region overlap 默认关闭，也意味着不能把其潜在 overlap 写成默认收益。

Commit body 中的 64/128/256 GPU 数值验证属于特定顺序修复的历史证据，不是当前全部组合的持续性能保证。Integration matrix 也同时包含大量 disabled cases，尤其 DeepSeek EP overlap 与 AutoParallel；文档应保留这些负证据（`torchtitan/experiments/graph_trainer/tests/integration_tests.py:426-503`；`torchtitan/experiments/graph_trainer/tests/integration_tests.py:723-781`）。

### ⑤ 发展趋势

本基线没有性能 gate 或 benchmark TODO 能支持“即将成为默认 Trainer”的判断。合理且有限的推断是：SimpleFSDP 当前主要价值在 **让通信成为可编程 graph IR**；是否胜过 eager FSDP2，必须按模型、拓扑、reshard 与 opt-in overlap 实测。

---

## 9. 结论：应替换的旧心智模型

### ① 背景 / 问题

旧页最容易把实验路径、已删除后端、非祖先提交与 core Trainer 行为混在一起，得到一个“更轻、更快、组合全开”的 SimpleFSDP。

### ② 为什么采用 live-vs-legacy 结论

本页只把当前 HEAD 可执行路径写成现状；历史 commit 只解释选择原因。明显替代是从旧实现或实验分支补齐当前源码缺口，但那会破坏 frozen-baseline 可复核性。

### ③ 当前可用结论

1. SimpleFSDP 的核心不是 wrapper API，而是 **sharded DTensor state + graph-visible property materialization + placement-driven backward reduce**（`torchtitan/experiments/graph_trainer/simple_fsdp.py:167-231`）。
2. GraphTrainer 捕获 fwd/loss/bwd，默认做 unshard 去重与 joint AG/RS bucketing/reorder；extra PG 与 dense-region 调度都是 opt-in（`torchtitan/experiments/graph_trainer/passes.py:255-349`）。
3. Reshard 是 memory policy 的保存/重算选择，不是当前 `data_parallel()` 参数（`torchtitan/experiments/graph_trainer/memory_policy.py:344-363`）。
4. 普通 PP 回退 core Trainer；GraphPP 是独立显式子图路径。GraphTrainer 仍强制 partial-DTensor，CP 当前不成立（`torchtitan/experiments/graph_trainer/trainer.py:136-148`；`torchtitan/experiments/graph_trainer/configs.py:252-258`）。
5. HSDP storage 和 toy numerics 有证据，AR→RS pattern 也可识别；但默认 scheduler 的 HSDP AR bucketing/reorder 与性能仍是验证空白（`torchtitan/experiments/graph_trainer/fsdp_patterns.py:202-236`；`torchtitan/experiments/graph_trainer/tests/test_numerics.py:800-818`）。

### ④ 约束、代价与失败边界

不要再写：SimpleFSDP 使用 backward hook；局部 `full_dtensor` 仍可配置；optimizer 已进入 joint graph；普通 PP 也走 GraphTrainer joint step；extra-PG overlap 默认开启；非祖先 HSDP bucket 提交代表当前能力。这六项都与当前实现或 baseline ancestry 冲突。

### ⑤ 发展趋势

**有锚点的推断**：最明确的演进压力是 GraphTrainer 迁移 `spmd_types`、减少结构 pattern 对上游 trace 形状的依赖，以及补齐组合测试；源码没有证据支持何时完成，更没有证据支持替换 core eager FSDP2。

---

## Related Pages

- [[02_engineering/02_train_frameworks/torchtitan/11_torchtitan_fsdp_analysis|TorchTitan FSDP2 分析]] —— 对照 eager FSDP2 的参数状态机、hook 与运行时通信生命周期。
- [[02_engineering/02_train_frameworks/torchtitan/16_torchtitan_spmd_types_analysis|TorchTitan SPMD Types 分析]] —— 解释全局默认 `spmd_types` 与 GraphTrainer 仍用 partial-DTensor 的边界。
- [[02_engineering/02_train_frameworks/torchtitan/20_torchtitan_fsdp_prefetch_overlap_memory_analysis|TorchTitan FSDP 预取、重叠与内存]] —— 对照 eager FSDP 的 reshard、prefetch 与显存窗口。
- [[02_engineering/02_train_frameworks/torchtitan/21_torchtitan_hsdp_backward_overlap_analysis|TorchTitan HSDP 反向重叠分析]] —— 展开 HSDP AR/RS 拓扑与当前验证边界。
- [[02_engineering/02_train_frameworks/torchtitan/23_torchtitan_compute_memory_optimizations_analysis|TorchTitan 计算与显存优化分析]] —— 对照编译、重计算与显存策略的全局配置边界。
- [[02_engineering/02_train_frameworks/torchtitan/27_torchtitan_graph_trainer_compiler_runtime_analysis|TorchTitan GraphTrainer 编译器与运行时分析]] —— 深入 tracer、pass pipeline、precompile artifact 与 GraphPP。
- [[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]] —— 本系列入口与功能树位置。
