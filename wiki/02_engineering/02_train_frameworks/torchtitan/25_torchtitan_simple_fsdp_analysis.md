# SimpleFSDP：把参数物化与梯度规约变成 joint FX 图里的 collective

> **代码基准**：torchtitan `main` @ `a3168782c9a3a2e40afbd0de114818b96e2bda6e`
> **最后更新**：2026-08-27 · **状态**：`torchtitan/experiments/graph_trainer` 实验路径
>
> **中心结论**：SimpleFSDP 不是“轻量版 FSDP2 wrapper”，而是一种编译器契约：持久状态保存为 DP-sharded DTensor，模块属性 getter 在计算前用 `redistribute(Replicate)` 物化参数，并用 `Partial(sum)` 描述本地梯度；GraphTrainer 再把 forward、loss、backward 与这些 collective 一次性 trace 成 joint FX 图，由 pass 做去重、分桶、预取和可选跨流重叠。它用编译期全局可见性替换 FSDP2 的运行时 hook/stream 编排，代价是静态 trace 契约、实验性组合边界，以及当前 GraphTrainer 仍被固定在 `partial_dtensor`。

---

## 1. 先清除旧基准留下的两个错误前提

### 1.1 当前既没有全局 `full_dtensor` 后端，也没有 SimpleFSDP 局部开关

全局 `ParallelismConfig.spmd_backend` 现在只接受 `partial_dtensor` 与 `spmd_types`，默认值是 `spmd_types`；配置校验也只允许这两项（`torchtitan/config/configs.py:174-180,261-266`）。提交 `601cf4d23` 删除了全局 `full_dtensor` 后端，同时删除了 SimpleFSDP 的局部 `full_dtensor` 参数，这是演进证据，不是另一条仍可配置的现状。

当前 `data_parallel()` 的完整签名只有 `model/device_mesh/mode/mp_policy/shard_dim`，`ReplicateComputation` 的构造参数也没有 `full_dtensor`（`torchtitan/experiments/graph_trainer/simple_fsdp.py:167-185,253-259`）。无 TP/EP 时，物化后的 DP DTensor **总会**执行 `to_local(grad_placements=Partial(sum))`；有 TP/EP 时则先 `to_local`，再只按非 DP mesh 重包成 DTensor（`torchtitan/experiments/graph_trainer/simple_fsdp.py:194-231`）。所以旧页“`full_dtensor=True` 保持整网 DTensor”及“nD 并行只是不支持这个开关”的描述已经失效。

### 1.2 默认后端与 GraphTrainer 实际后端不能混写

虽然全局默认已经是 `spmd_types`，`to_graph_trainer_config()` 仍明确把 GraphTrainer 配置替换为 `spmd_backend="partial_dtensor"`；紧邻 TODO 说明原因是 partial-DTensor 不会应用 `ShardingConfig` 声明的 CP placements，必须等 GraphTrainer 迁到 `spmd_types` 才能重新启用 CP（`torchtitan/experiments/graph_trainer/configs.py:239-258`）。因此本页的当前主线是 **SimpleFSDP + partial-DTensor GraphTrainer**，不能借全局默认值宣称 CP 已可用。

AutoParallel 也不是旧 `full_dtensor` 的新名字。模型 registry 在 `enable_autoparallel` 时改走独立的 `parallelize_autoparallel_*` 分支，否则才走本页的手工 TP/EP + SimpleFSDP 分支（`torchtitan/experiments/graph_trainer/llama3/__init__.py:20-27`）。AutoParallel 配置本身声明它替代 manual TP/FSDP/EP，且只支持 `aot_fx_trace`（`torchtitan/experiments/graph_trainer/configs.py:177-189`）；它不应再被解释成 `data_parallel(..., full_dtensor=True)`。

---

## 2. 当前真实接线：先模型并行，再 SimpleFSDP，最后编译

Llama 路径先给模块标注 FQN，若启用 TP 则先做模型并行，随后**无条件**调用 `apply_simple_fsdp()`，最后调用 `apply_compile()`（`torchtitan/experiments/graph_trainer/llama3/parallelize.py:53-70`）。DeepSeek V3 同样先做 TP/EP，再做 SimpleFSDP；它只在二者之间多了 MoE 标注，且在 compile 前可插入 eager EP chunking（`torchtitan/experiments/graph_trainer/deepseek_v3/parallelize.py:59-77`）。

```text
模型构造 / FQN 标注
  → TP 或 EP 先产生 inner DTensor（若启用）
  → apply_simple_fsdp：把 DP placement 叠到参数状态上
  → apply_compile：aot_fx_trace 模式不包 model
  → 首个真实 batch：forward + loss + autograd.grad joint trace
  → graph passes
  → 后续 step 复用变换后的 FX 图
```

“无条件”不是说单卡会通信：GraphTrainer 保留 degree-1 的 `fsdp` mesh，仍通过 SimpleFSDP getter 应用参数混合精度；Llama 接线的注释明确给出这个理由（`torchtitan/experiments/graph_trainer/llama3/parallelize.py:58-61`）。对应单 rank CPU 测试验证：底层 `_parameters["weight"]` 保持 FP32，而属性读取和 forward 输出变为 BF16（`torchtitan/experiments/graph_trainer/tests/test_simple_fsdp.py:35-76`）。

### 2.1 DP 模式与 MoE 专家分片

`data_parallel()` 将三种 mode 映射为三种持久参数 placement；`hybrid_shard` 强制要求 2D mesh（`torchtitan/experiments/graph_trainer/simple_fsdp.py:253-272`）。

| mode | 持久参数 placement | 梯度同步语义 |
|---|---|---|
| `replicate` | `(Replicate(),)` | DDP 式 all-reduce |
| `fully_shard` | `(Shard(shard_dim),)` | FSDP 式 reduce-scatter |
| `hybrid_shard` | `(Replicate(), Shard(shard_dim))` | HSDP 式 replicate/shard 组合规约 |

`apply_simple_fsdp()` 在有 `dp_replicate` 且又有 `dp_shard`（代码也预留 `cp_enabled`）时选 `hybrid_shard`，仅 replicate 时选 `replicate`，否则选 `fully_shard`（`torchtitan/experiments/graph_trainer/common_utils.py:421-448`）。但这里出现 `cp_enabled` 只是 mesh 选择代码，不推翻 §1.2 的当前 CP 失效边界。

EP 开启且模型是 `Decoder` 时，每层 `moe.routed_experts.inner_experts` 会先在 EDP/EFSDP mesh 上单独包装；若 E-FSDP ranks 多于 expert 数，专家权重改沿 dim 1 分片，随后整模型再走普通 DP mesh（`torchtitan/experiments/graph_trainer/common_utils.py:450-483`）。这使 dense 参数与 routed-expert 参数可以拥有不同 DP group，而不是强迫所有权重共享一个 FSDP group。

---

## 3. 参数状态机：分片存储，访问时物化，反向按 placement 规约

### 3.1 包装阶段

`data_parallel()` 遍历每个 module 的直接参数：普通 Tensor 用 `distribute_tensor`，已有 TP/EP DTensor 则用 `_distribute_dtensor`；分片结果重新注册为 `nn.Parameter`，已经带 `SimpleFSDP` 类名的 module 会跳过，避免重复包装（`torchtitan/experiments/graph_trainer/simple_fsdp.py:274-293`）。之后每个 module 安装一个 `ReplicateComputation` getter（`torchtitan/experiments/graph_trainer/simple_fsdp.py:295-318`）。

这个 getter 不是 `nn.utils.parametrize`。实现按 `(原始类, 参数名集合)` 缓存动态 `SimpleFSDP*` 子类，把每个参数名变成读取 `_parameters[pn]` 后调用 parametrization 的 property，并把动态类暴露进 `sys.modules` 供 pickle/GraphPickler 解析（`torchtitan/experiments/graph_trainer/simple_fsdp.py:130-164`）。`state_dict` 直接访问 `_parameters`，因此可以保存分片状态而不触发参数物化；源码也指出 DCP 的 `get_model_state_dict` 仍有待避免 getter 的 TODO（`torchtitan/experiments/graph_trainer/simple_fsdp.py:138-143`）。

### 3.2 计算阶段

`ReplicateComputation` 把计算 placement 设为各 DP mesh 维都 `Replicate`，把本地梯度 placement 设为各维 `Partial(sum)`（`torchtitan/experiments/graph_trainer/simple_fsdp.py:167-185`）。无模型并行轴时，getter 执行：

```python
replicated = x.redistribute(
    placements=[Replicate()] * dp_mesh.ndim,
    forward_dtype=param_dtype,
    backward_dtype=reduce_dtype,
)
local_param = replicated.to_local(
    grad_placements=[Partial(reduce_op="sum")] * dp_mesh.ndim,
)
```

这段现行实现位于 `simple_fsdp.py:225-231`。当持久 placement 是 `Shard` 时，前向 redistribute 产生 all-gather，反向把 Partial 梯度规约回 Shard，形成 reduce-scatter；持久 placement 是 `Replicate` 时形成 all-reduce；HSDP 混合二者。源码在 materialization 入口直接列出了 DDP/FSDP/HSDP 的这三种规约语义（`torchtitan/experiments/graph_trainer/simple_fsdp.py:187-190`）。`forward_dtype` 与 `backward_dtype` 把参数通信/计算 dtype 与梯度规约 dtype 放进同一个可追踪 redistribute（`torchtitan/experiments/graph_trainer/simple_fsdp.py:202-214,225-231`）。

`disable_active_parametrization()` 暂时让 getter 原样返回分片 `x`，并在 `finally` 恢复全局开关；注释把它限定为 forward 外的检查、调试和初始化，不是训练时关闭通信的模式（`torchtitan/experiments/graph_trainer/simple_fsdp.py:29-39,239-247`）。

### 3.3 与 TP/EP 的嵌套

已有 inner DTensor 时，`_distribute_dtensor()` 拼接 outer DP mesh 与 inner TP/EP mesh；若两层都切同一个 tensor dim，则用 `_StridedShard(split_factor=inner shard count)` 表达先 inner、再 DP 的嵌套切分（`torchtitan/experiments/graph_trainer/simple_fsdp.py:48-94`）。包装时先在 outer mesh 上分发 local tensor，再用拼接 mesh 与组合 placements 构造状态 DTensor（`torchtitan/experiments/graph_trainer/simple_fsdp.py:100-127`）。

物化时实现暂时把组合 DTensor 的 local shard 重包到纯 DP mesh，做 DP all-gather，再把 local 完整参数重包回 non-DP mesh；源码限制 non-DP mesh 最多两维，即只覆盖 DP + EP、TP 或 EP+TP（`torchtitan/experiments/graph_trainer/simple_fsdp.py:187-224`）。输出保留的是 inner TP/EP DTensor 语义，不是已删除的“全模型 full-DTensor 后端”。

---

## 4. Trace/runtime 契约：collective 为什么能被 pass 看见

`GraphTrainerCompileConfig.mode` 默认是 `aot_fx_trace`，`jit` 已标为 deprecated（`torchtitan/experiments/graph_trainer/configs.py:65-74`）。在该模式下 `apply_compile()` 刻意不包装 model，只声明真正捕获会发生在训练时，或从 precompile artifact 懒加载（`torchtitan/experiments/graph_trainer/compile.py:98-119`）。因此被 trace 的入口不是孤立 forward，而是 Trainer 对真实输入构造的完整一步。

`make_fwd_bwd_step()` 的闭包执行 model forward、loss，并用 `torch.autograd.grad(loss, params)` 返回 `[loss] + grads`；model 本身不作为用户输入，而由 tracer 把参数/缓冲穿成静态图输入（`torchtitan/experiments/graph_trainer/trainer.py:75-103`）。首步在 `train_context()` 内调用 `minimal_fx_tracer`，随后构造并一次性应用 pass pipeline；之后 `run_traced` 复用图并把显式 grads 累积回 live 参数（`torchtitan/experiments/graph_trainer/trainer.py:206-262`）。SimpleFSDP getter 在这个闭包内执行，所以 redistribute 的 all-gather/reduce-scatter 也成为 FX 节点，而不是藏在 module hook 中。

Tracer 的静态边界是：module state 与用户输入先 flatten，DTensor 等 tensor subclass 被拆成 plain tensors并记录重包布局，然后在 fake tensor + non-strict `make_fx` 下临时把 live module state 替换为图输入（`torchtitan/experiments/graph_trainer/make_fx_tracer.py:390-470,477-526`）。用户 pytree leaf 只能是 Tensor 或有限 primitive，wrapper subclass 上的 marked dynamic dim 会报错（`torchtitan/experiments/graph_trainer/make_fx_tracer.py:403-431`）。

每次执行会重新采样 live state，并在 `torch.no_grad()` 下运行，因为图里已经含显式 backward；但 FQN 与 input pytree spec 的 runtime 校验默认关闭，调用方还必须保持 kwargs 的 trace-time 顺序（`torchtitan/experiments/graph_trainer/make_fx_tracer.py:532-559,566-608`）。Optimizer 仍不在 GraphTrainer 的 traced state 中，`TracedResult` 源码对此保留明确 TODO（`torchtitan/experiments/graph_trainer/make_fx_tracer.py:312-323`）。完整 Trainer 控制面见 [[02_engineering/02_train_frameworks/torchtitan/27_torchtitan_graph_trainer_compiler_runtime_analysis|TorchTitan GraphTrainer 编译器与运行时分析]]。

---

## 5. `reshard_after_forward`：不是 SimpleFSDP 构造参数，而是图内保存/重算策略

持久参数始终是 §3 的分片状态，getter 每次被图执行时仍会物化参数。配置中的 `fsdp_reshard_after_forward` 取 `default/always/never`，语义是显存与通信的权衡（`torchtitan/config/configs.py:147-159`）；它没有被传入当前 `data_parallel()`。

GraphTrainer 的默认 memory policy 先解析该配置：当策略是不 reshard 时，它把 FSDP unshard 输出列为强制保存节点；当策略要求 reshard 时则不强制保存，使 backward 可通过 rematerialization 再次执行 unshard（`torchtitan/experiments/graph_trainer/memory_policy.py:344-363`）。被保存/复算的准确边界是 all-gather/wait reconstruction chain 的最后一个非-view 输出；无 all-gather 的 replicated/local 参数则没有这个 unshard 输出（`torchtitan/experiments/graph_trainer/fsdp_patterns.py:147-190`）。

端到端 pass 测试把这个差异固定为图顺序：`never` 只有前向层序 all-gather；`always` 还应出现按逆层序执行的 backward all-gather（`torchtitan/experiments/graph_trainer/tests/test_passes.py:1960-2008`）。所以旧页若把 reshard 说成 `data_parallel` wrapper 在 forward 尾部立即“丢完整参数”，会错过当前由 SAC/memory policy 决定保存还是重算的真实机制。

---

## 6. 当前 FSDP pass：默认、可选与顺序必须分开

当 `compile.enable_passes` 为真时，普通 aot trace 路径先做 DCE、canonicalize 与重复 unshard chain 去重；随后是 memory policy、CPU offload、SAC，再进入 EP/FSDP/TP 通信调度和最终 Inductor（`torchtitan/experiments/graph_trainer/passes.py:201-209,255-268,269-348`）。Precompile artifact 已在生成阶段完成 compile-time 变换，训练时默认 pipeline 只再追加 CUDA Graph capture，而不会重跑这些变换（`torchtitan/experiments/graph_trainer/passes.py:410-448`）。

当前有四类 FSDP 相关动作：

| 动作 | 当前启用条件 | 做什么 |
|---|---|---|
| `deduplicate_fsdp_unshard_chains_pass` | enabled pipeline 的规范化阶段 | 同一 flat param 被多次 getter 读取时，把重复 AG/wait reconstruction chain 合并为第一个（`torchtitan/experiments/graph_trainer/fsdp_passes.py:74-118`） |
| `joint_transformer_block_bucketing_reordering_pass` | enabled pipeline 默认追加 | joint graph 内按 module、方向分别 bucket forward AG、backward AG、backward RS，并产生预取顺序（`torchtitan/experiments/graph_trainer/passes.py:274-286`; `torchtitan/experiments/graph_trainer/fsdp_passes.py:543-593`） |
| `reassign_collective_pgs_pass` | `enable_fsdp_ag_rs_overlap=True`，默认关闭 | 把 FSDP AG 改派到同 ranks 的额外 PG；每个 NCCL PG 有独立 stream，因此可与原 PG 上的 RS 重叠，且必须在 bucketing 前执行（`torchtitan/experiments/graph_trainer/configs.py:149-153`; `torchtitan/experiments/graph_trainer/fsdp_passes.py:136-139,189-232`） |
| `schedule_fsdp_comms_to_dense_regions_pass` | `enable_fsdp_dense_region_overlap=True`，默认关闭 | 把 bucketed AG/RS launch 物理移动到相邻 dense attention 区域，保留 wait 到首次使用前，并把 RS wait 下沉（`torchtitan/experiments/graph_trainer/configs.py:155-163`; `torchtitan/experiments/graph_trainer/fsdp_passes.py:1008-1040`） |

这里修正旧页“两个通信 pass 固定处于第 6/7 位”的结论：bucketing/reordering 是 enabled pipeline 的默认项，但额外 PG 重分配由默认关闭的显式开关控制，dense-region scheduler 也是新增的显式开关。`compile_time_passes()` docstring 仍写着可用 `disable_passes` 关闭 PG 重分配，但实际 pass 列表只有在 `enable_fsdp_ag_rs_overlap` 为真时才追加；现状应以配置默认值与可执行的 gate 为准（`torchtitan/experiments/graph_trainer/passes.py:158-161,274-286`）。

### 6.1 默认 bucketing/reordering 实际安排了什么

Joint scheduler 在一张 forward+backward 图上工作，并把两个方向分开 bucket，防止 AG 配对跨过 forward/backward 边界；bucket 内参数顺序来自 traced state FQN，以匹配 eager FSDP2 的 first-seen parameter order（`torchtitan/experiments/graph_trainer/fsdp_passes.py:265-296,299-369`）。这解释了为什么 SimpleFSDP 可以从“每个属性 getter 一条 materialization chain”提升到“按 transformer scope 聚合通信”：分组是编译 pass 的产物，不是参数 wrapper 自带的 ParamGroup。

重排阶段先构造 overlap dependencies：RS 以正向拓扑遍历延后 wait，让规约跨过后续 backward work；AG 逆序遍历，并为 forward/backward 保持独立缓冲，把 start 提前、wait 保留到依赖前（`torchtitan/experiments/graph_trainer/fsdp_passes.py:398-416,426-521`）。若启用 dense-region scheduler，它进一步规定 AG(i) 落到相邻层 dense 区域、RS(i) 落到下一段 backward dense 区域；strict 模式无法完成所有合法移动时会抛错，而不是静默给出部分调度（`torchtitan/experiments/graph_trainer/fsdp_passes.py:1017-1040,1301-1334`）。

---

## 7. 组合边界与失败模式

| 组合/边界 | 当前结论 | 证据 |
|---|---|---|
| TP/EP | 手工路径支持 outer DP 叠最多两维 inner mesh；更高维直接 assertion | `torchtitan/experiments/graph_trainer/simple_fsdp.py:187-224` |
| EP + E-FSDP | routed experts 单独走 `efsdp` mesh，bucket plan 也可把 expert bucket 从 MoE dense 部分拆开 | `torchtitan/experiments/graph_trainer/common_utils.py:450-483`; `torchtitan/experiments/graph_trainer/tests/test_passes.py:3603-3621` |
| CP | 当前 GraphTrainer 转换强制 partial-DTensor，源码 TODO 明示 CP placements 不生效；不是已验证组合 | `torchtitan/experiments/graph_trainer/configs.py:252-258` |
| PP | 普通 `GraphTrainer.forward_backward_step` 在 PP 开启时回退父 Trainer；GraphPP 是独立路径 | `torchtitan/experiments/graph_trainer/trainer.py:136-148` |
| GraphPP + reshard | 只接受 `default/never`，显式拒绝 `always`，并要求 `aot_fx_trace` 与 runtime PP schedule | `torchtitan/experiments/graph_trainer/graph_pp/pipeline.py:40-63` |
| EP graph chunking + TP | TP degree大于 1 会在 pass 构造时抛错 | `torchtitan/experiments/graph_trainer/tests/test_passes.py:3529-3545` |
| dense-region FSDP + EP overlap | 只与 `layers.*.moe` 的 graph chunking 组合；其他 EP overlap 形态会 warning 后禁用该 scheduler | `torchtitan/experiments/graph_trainer/passes.py:302-335` |
| AutoParallel | 独立 ILP SPMD 路径；Llama 当前拒绝 DDP/CP/PP，DeepSeek 还拒绝 TP 并要求 2D EFSDP+EP mesh | `torchtitan/experiments/graph_trainer/llama3/parallelize_autoparallel.py:38-68`; `torchtitan/experiments/graph_trainer/deepseek_v3/parallelize_autoparallel.py:91-134` |

不要从“代码能构造某个 mesh”外推出数值/性能已验证。当前 multi-GPU trace 测试用 `partial_dtensor` 对 Llama3、Qwen3、DeepSeek V3 检查图里确有 AG/RS，并在 5 个 step 上把 traced loss/grad 与参考路径做 exact equality；GPT-OSS FSDP case 仍因 scatter dtype mismatch 被 skip（`torchtitan/experiments/graph_trainer/tests/test_trace_module.py:1813-1930,1932-1969`）。Pass 层另有重复 unshard 2→1、PG 重写且 AG 数量不变、真实 GraphTrainer 层序预取等测试（`torchtitan/experiments/graph_trainer/tests/test_passes.py:159-217,327-371,1832-2008`）。

---

## 8. 结论

1. **SimpleFSDP 的权威状态是“sharded DTensor storage + dynamic getter”**：forward 物化 Replicate 参数，反向从 Partial 梯度回到持久 placement；当前 getter 总是退出 DP DTensor 域，不再存在局部 `full_dtensor` 分支（`torchtitan/experiments/graph_trainer/simple_fsdp.py:167-231,253-319`）。
2. **GraphTrainer 才是它的通信调度器**：完整 fwd/loss/bwd trace 让 collective 成为 FX 节点，默认做去重和 joint bucketing/reordering，额外 PG 与 dense-region 调度则是明确的 opt-in（`torchtitan/experiments/graph_trainer/trainer.py:206-262`; `torchtitan/experiments/graph_trainer/passes.py:255-348`）。
3. **reshard 是图内内存策略，不是 wrapper 布尔参数**：不 reshard 强制保存 unshard 输出，reshard 允许 backward 重新物化；这决定显存/通信交换，也改变 backward AG 的图形（`torchtitan/experiments/graph_trainer/memory_policy.py:344-363`; `torchtitan/experiments/graph_trainer/tests/test_passes.py:1960-2008`）。
4. **当前最大组合缺口是后端迁移**：全局默认虽为 `spmd_types`，GraphTrainer 仍强制 `partial_dtensor`，CP 因而不能按声明布局执行；AutoParallel 与 GraphPP 各自是另一路径，不是对这一缺口的隐式修复（`torchtitan/config/configs.py:174-180`; `torchtitan/experiments/graph_trainer/configs.py:239-258`）。

---

## Related Pages

- [[02_engineering/02_train_frameworks/torchtitan/11_torchtitan_fsdp_analysis|TorchTitan FSDP2 分析]] —— eager FSDP 参数状态机与运行时 hook/stream 对照
- [[02_engineering/02_train_frameworks/torchtitan/16_torchtitan_spmd_types_analysis|TorchTitan SPMD Types 分析]] —— 默认 `spmd_types`、兼容 `partial_dtensor` 与声明式布局
- [[02_engineering/02_train_frameworks/torchtitan/20_torchtitan_fsdp_prefetch_overlap_memory_analysis|TorchTitan FSDP 预取、重叠与内存]] —— eager FSDP 的 unshard、prefetch 与显存生命周期
- [[02_engineering/02_train_frameworks/torchtitan/24_torchtitan_comm_optimizations_overlap_analysis|TorchTitan 通信优化与重叠]] —— Async-TP、symmetric memory、EP dispatcher 与跨维组合
- [[02_engineering/02_train_frameworks/torchtitan/27_torchtitan_graph_trainer_compiler_runtime_analysis|TorchTitan GraphTrainer 编译器与运行时分析]] —— joint trace、pass pipeline、precompile 与 GraphPP 控制面
- [[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]] —— 系列入口与功能树定位
