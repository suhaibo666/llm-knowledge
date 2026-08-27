# 激活重计算：策略对象、逐块包装与编译器内存预算

> **TorchTitan 源码基线**：`pytorch/torchtitan` `main` @ `a3168782c9a3a2e40afbd0de114818b96e2bda6e`（2026-08-27）
> **PyTorch 内部机制固定基线**：`pytorch/pytorch` tag `v2.9.1`。本页用它解释 non-reentrant checkpoint 与 SAC 的底层协议；这不是对 TorchTitan 所依赖 PyTorch HEAD 的推断。
> **范围**：训练时的 activation checkpointing（AC，亦称 activation recomputation）。权重/优化器状态的持久化 checkpoint 是另一条机制。

## 0. 核心结论

当前 TorchTitan 不再用一个 `apply_ac(model, ac_mode, ...)` 函数分派三种模式，而是把 AC 建模为配置可构造的策略对象：`FullAC` 和 `SelectiveAC` 在 eager 模型层边界插入 non-reentrant checkpoint wrapper，`MemoryBudgetAC` 则不包装层，只设置编译器分区器的全局 activation-memory budget。三者共享 `ActivationCheckpointing.apply()` 的模型约定或生命周期，但并不共享同一种执行机制（`torchtitan/distributed/activation_checkpoint.py:112-164,166-182,185-287,290-330`）。

这一区分决定了正确的接线顺序：模型先完成 TP/SPMD parallelize，再应用 AC 策略，再 compile，最后 FSDP；PP 则先切成 stage-local model parts，再对每个 part 走同一模型并行化函数（`torchtitan/models/llama3/parallelize.py:40-78`; `torchtitan/distributed/pipeline_parallel.py:96-123`）。AC 优化的是单步前后向的激活驻留量；训练状态能否跨进程重启由独立的 `CheckpointManager` 配置和 `components/checkpointer/` 包负责（`torchtitan/trainer.py:65-107`; `torchtitan/components/checkpointer/__init__.py:7-29`）。

---

## 1. 旧知识迁移：哪些已经失效

> [!deprecated] 旧版通用 `apply_ac` 入口已失效
> 当前通用实现是 `ActivationCheckpointing`、`FullAC`、`SelectiveAC`、`MemoryBudgetAC` 类层次，并由显式 union 暴露 `selective`、`full`、`memory-budget`、`none` 四个配置分支（`torchtitan/distributed/activation_checkpoint.py:112-182,185-341`）。历史重构落在 commit `c5d93d109`；现状仍以上述当前源码为准。

> [!deprecated] 旧配置字段不可照搬
> 基类现在只有 `preserve_rng_state`、`determinism_check`、`debug`；`early_stop` 不再是配置项，Full/Selective wrapper 都固定传 `early_stop=False`（`torchtitan/distributed/activation_checkpoint.py:122-140,172-181,278-287`）。Selective 的形状强制重算入口现在是 `force_recompute_mm_shapes_by_fqns`，默认匹配 `moe.router.gate`，不是旧的按 op 名称列表（`torchtitan/distributed/activation_checkpoint.py:194-207`）。

> [!deprecated] `torchtitan/components/checkpoint.py` 已不是当前权重 checkpoint 所在地
> 组件配置遵循“配置靠近组件”的约定，当前 checkpoint owner 指向 `components/checkpointer/dcp.py`；公共导出集中在 `components/checkpointer/__init__.py`（`torchtitan/config/configs.py:7-22`; `torchtitan/components/checkpointer/__init__.py:7-29`）。历史迁移 commit 为 `6bc2108e9`。

唯一需要保留的例外是 Flux：它仍有一个**模型私有** `apply_ac()`，只要 `ac_config` 非空就对 `double_blocks`、`single_blocks` 做 full checkpoint，并固定 `preserve_rng_state=True`；它不解释 Full/Selective/MemoryBudget 的策略差异（`torchtitan/models/flux/parallelize.py:42-53,170-185`）。因此“通用旧入口已删除”不等于“当前树中完全没有名为 `apply_ac` 的函数”。

---

## 2. 两种 checkpoint 不是一件事

| 机制 | 被节省/持久化的状态 | 时间尺度 | 当前配置入口 |
|---|---|---|---|
| Activation checkpointing | 前向中间激活少存或不存，反向时重算 | 一个训练 step 内 | `Trainer.Config.activation_checkpoint`，默认 `SelectiveAC.Config`（`torchtitan/trainer.py:105-107`） |
| Weight/state checkpoint | 模型、优化器、数据加载等训练状态写入持久存储，用于恢复 | 跨 step、进程或作业 | `Trainer.Config.checkpoint`，默认 `CheckpointManager.Config`（`torchtitan/trainer.py:65-104`） |

名称相似不代表数据流相交：AC wrapper 改写 forward/backward 的保存与重算协议；`CheckpointManager` 是训练组件，导出 stateful、interval 和 DCP helper 等持久化能力（`torchtitan/components/checkpointer/__init__.py:7-29`）。

---

## 3. 当前配置与真实调用链

### 3.1 配置是可构造 union

`ActivationCheckpointConfig` 不是一个带 `mode` 字符串的大 dataclass，而是四路显式 subcommand union：

| 选择 | 构造结果 | 关键效果 |
|---|---|---|
| `selective` | `SelectiveAC.Config` | wrapper + op 级保存/重算策略 |
| `full` | `FullAC.Config` | wrapper + 块内全部激活重算 |
| `memory-budget` | `MemoryBudgetAC.Config` | 设置编译器分区预算，不插 eager wrapper |
| `none` | `None` | 不应用 AC |

union 定义及其 CLI 名称见 `torchtitan/distributed/activation_checkpoint.py:333-341`；Trainer 的默认值是 selective（`torchtitan/trainer.py:105-107`）。这种类型层分派让每种策略只暴露自己真正拥有的字段，而不是让无效字段组合延迟到运行时。

### 3.2 eager 模型的顺序

Llama 3 的当前顺序是：

1. 先按 TP/SPMD 配置 parallelize 模型（`torchtitan/models/llama3/parallelize.py:40-44`）。
2. 若 AC 非空，调用 `ac_config.build(job_config, parallel_dims).apply(model)`（`torchtitan/models/llama3/parallelize.py:46-47`）。
3. 再按 compile 配置编译 transformer blocks（`torchtitan/models/llama3/parallelize.py:49-55`）。
4. 最后应用 FSDP/HSDP（`torchtitan/models/llama3/parallelize.py:57-78`）。

这样 AC 看到的是已具有 TP/SPMD 语义的 block，而 compile 又能捕获 checkpoint 边界。PP 不改变该局部顺序：pipeline builder 先得到 `model_parts`，随后逐个把 `ac_config` 交给模型的 `parallelize_fn`（`torchtitan/distributed/pipeline_parallel.py:96-123`）。Qwen3.5 还把同一策略分别用于 language model 与 vision encoder，再分别 compile（`torchtitan/models/qwen3_5/parallelize.py:73-90`）。

### 3.3 基类规定模型边界

`ActivationCheckpointing.apply()` 先执行 Dynamo LRU workaround，再取得 `model.get_submodule("layers")`；随后只遍历 `layers.named_children()`，以 `layers.<id>` 作为 FQN 调 `_wrap_block()` 并注册回原位置（`torchtitan/distributed/activation_checkpoint.py:145-163`）。所以通用 AC 的结构契约是“模型具有 `layers` 容器，容器的直接 children 是 checkpoint block”，不是任意递归包裹所有叶子。

这也是 PP 能自然复用的原因：stage-local part 只包含本 stage 的层，但仍保留模型 parallelize 函数预期的层容器；AC 的状态边界因而跟 stage-local block 对齐，而不跨 stage 保存 autograd graph（`torchtitan/distributed/pipeline_parallel.py:96-123`）。

---

## 4. 三种策略实际上做什么

### 4.1 FullAC：块是重算边界

`FullAC.Config` 不增加字段；每个 block 用 PyTorch distributed checkpoint wrapper 包起来，透传 RNG、determinism 和 debug 配置，并固定 non-reentrant `early_stop=False`（`torchtitan/distributed/activation_checkpoint.py:166-182`）。这意味着反向触发该块重算时不会因“本次已恢复完所需 saved tensors”而提前停止：执行边界更规整，但可能多做后半段计算。

FullAC 的价值是规则、可预测；代价是块内昂贵算子也会重算。它适合显存压力优先、或 selective 策略难以稳定描述自定义算子的场景。

### 4.2 SelectiveAC：默认省下昂贵计算，但显式处理例外

SelectiveAC 仍在每个 block 建立 non-reentrant checkpoint 边界，但给 wrapper 传入由 `create_selective_checkpoint_contexts(policy_fn)` 生成的 context（`torchtitan/distributed/activation_checkpoint.py:278-287`）。policy 的关键不是“列一个不重算白名单”这么简单，而是以下状态机：

- forward 与 recompute 各自维护 mm/linear 次数，避免两个阶段共用计数造成漂移（`torchtitan/distributed/activation_checkpoint.py:243-267`）。
- CUDA 到 CPU 的 `_to_copy` 必须保存，否则 recompute 会重做 offload 副作用（`torchtitan/distributed/activation_checkpoint.py:247-255`）。
- 用户按 FQN 选择 Linear 模块，代码抽取其 `(in_features, out_features)`；随后任何匹配该矩阵形状的 `aten.mm` 或 `aten.linear` 都倾向重算。配置文档明确警告：同形状的非目标 matmul 也会被命中（`torchtitan/distributed/activation_checkpoint.py:194-207,220-241,256-263`）。
- 默认 save-op 集合里的算子通常 `MUST_SAVE`，但 mm/linear 每第二次调用会 `PREFER_RECOMPUTE`；不在集合中的普通算子也 `PREFER_RECOMPUTE`（`torchtitan/distributed/activation_checkpoint.py:269-274`）。这个“交替 matmul”启发式保留部分昂贵 GEMM，同时继续换取显存。

默认 save-op 集合覆盖 SDPA 各后端、低精度 softmax 路径的 `max`、FlexAttention、linear、可能不确定的 top-k，以及 reduce-scatter/all-to-all 通信输出；DeepEP、HybridEP、Inductor HOP 和 torch-attn varlen 算子只在运行环境可解析时加入（`torchtitan/distributed/activation_checkpoint.py:31-69`）。可选 op 解析失败会被安静跳过，而不是让通用训练因未安装扩展而失败（`torchtitan/distributed/activation_checkpoint.py:72-94`）。

策略并非封死：子类可以覆盖 `get_save_ops()` 改变默认保存集合（`torchtitan/distributed/activation_checkpoint.py:209-212`）。明显替代方案是在核心文件硬编码每个模型的 op 列表；override seam 把模型/实验差异留在策略扩展点，同时保持通用 policy 的计数与副作用规则一致。

### 4.3 MemoryBudgetAC：把选择交给编译器分区器

MemoryBudgetAC 的 `apply()` 不遍历或包装 `layers`。它设置 `torch._functorch.config.activation_memory_budget`；预算 `0` 表示只保存编译区域输入、区域内激活全重算，`1` 表示保存运行时优化分区器选出的全部激活，默认是 `0.5`（`torchtitan/distributed/activation_checkpoint.py:290-305,319-330`）。配置在构造时强制预算落在 `[0, 1]`（`torchtitan/distributed/activation_checkpoint.py:315-317`）。

可选的 `visualize_memory_budget_pareto` 会创建输出目录，并启用 functorch 的 Pareto 图 dump；这同样是进程级配置副作用，不是某个 block 的局部 wrapper（`torchtitan/distributed/activation_checkpoint.py:307-313,319-328`）。Trainer 因而要求它同时满足 `compile.enable=True` 且模型声明 compile 组件，否则初始化直接报错（`torchtitan/trainer.py:156-163`）。

明显替代方案是先插 eager selective wrapper、再让编译器二次决定保存集；当前实现选择二者互斥，避免两层重算边界让分区器看不到完整 compiled region。

---

## 5. 共享边界、约束与容易误配之处

### 5.1 Dynamo 缓存 workaround 是全局副作用

所有通用策略进入 `apply()` 时都会调用 `_disable_dynamo_lru_cache()`；MemoryBudgetAC 也显式调用它（`torchtitan/distributed/activation_checkpoint.py:145-152,319-321`）。原因是 SAC + PP + FlexAttention 可能在同一代码对象上交替出现 dynamic/static graph；当前 workaround 关闭 Dynamo LRU，使其用插入顺序命中缓存（`torchtitan/distributed/activation_checkpoint.py:97-109`）。这不是单模型局部状态，多个 trainer 共进程时需要把它视为全局编译行为变化。

### 5.2 FlexAttention 与 `spmd_types` 的 selective 组合被拒绝

Trainer 的 typechecking 明确拒绝 `SelectiveAC.Config` + FlexAttention + `spmd_types`，错误信息要求改用 full/none 或非 Flex attention（`torchtitan/trainer.py:141-154`）。因此“Selective 默认可用于所有 attention 后端”是错误断言；save-op 集合中列出 Flex op，不代表所有分布式类型路径已被支持。

### 5.3 SelectiveAC 的 `debug=True` 是底层协议冲突

当前 SelectiveAC 暴露基类 `debug`，又无条件给 wrapper 传自定义 `context_fn`（`torchtitan/distributed/activation_checkpoint.py:122-140,278-287`）。但固定 PyTorch 2.9.1 的 non-reentrant 实现明确拒绝 `debug=True` 与非默认 `context_fn` 同时使用（`[PyTorch 2.9.1] torch/utils/checkpoint.py:1496-1501`）。所以在这条固定基线上，该组合会失败；配置类型本身没有提前排除它。

### 5.4 RNG 与确定性检查解决的是不同问题

`preserve_rng_state` 控制重算是否回放 forward 的 CPU/device RNG；PyTorch 2.9.1 在进入 recompute 前恢复保存的 RNG state（`[PyTorch 2.9.1] torch/utils/checkpoint.py:1524-1555`）。`determinism_check` 则让 checkpoint frame 比较原 saved tensor 与重算 tensor 的元数据（`[PyTorch 2.9.1] torch/utils/checkpoint.py:822-916`）。前者避免 dropout 等随机路径漂移，后者发现重算结果结构不一致；二者不可互相替代。

---

## 6. PyTorch 2.9.1 固定基线：wrapper 下面的协议

本节只解释 TorchTitan wrapper 所依赖的底层形状，并明确冻结在 PyTorch `v2.9.1`。

### 6.1 wrapper 保持模块与 state_dict 透明

PyTorch distributed `CheckpointWrapper` 把原模块存入 `_checkpoint_wrapped_module`，通过 state-dict hooks 去掉/补回该前缀，并把属性访问转发给被包装模块（`[PyTorch 2.9.1] torch/distributed/algorithms/_checkpoint/checkpoint_wrapper.py:35-102,114-173`）。它默认选择 `CheckpointImpl.NO_REENTRANT`，forward 只是在原模块调用外包一层 `torch.utils.checkpoint.checkpoint`（`[PyTorch 2.9.1] torch/distributed/algorithms/_checkpoint/checkpoint_wrapper.py:114-173`）。

因此 AC wrapper 改变执行/保存语义，但不应改变逻辑参数名。这对后续 FSDP 按 FQN 管理参数和权重 checkpoint 恢复尤其重要。

### 6.2 non-reentrant checkpoint 是“前向建票据，反向按需重放”

PyTorch 2.9.1 的 non-reentrant 路径先创建 generator frame，进入 forward context 执行用户函数，再完成 generator 收尾（`[PyTorch 2.9.1] torch/utils/checkpoint.py:470-508,1557-1573`）。forward 的 saved-tensor pack hook 不保存真实 tensor，而是创建 `_Holder`；第一次 unpack 时才运行 recompute，并从对应 holder 取回本次重算张量（`[PyTorch 2.9.1] torch/utils/checkpoint.py:1127-1171`）。

重算 pack hook 以保存顺序对齐 holder，检测“重算保存得更多”等不一致；默认 early-stop 会在已恢复完所需张量时抛内部停止信号（`[PyTorch 2.9.1] torch/utils/checkpoint.py:1068-1114`）。TorchTitan Full/Selective 固定 `early_stop=False`，正是主动放弃这一提前退出优化，以获得完整 block 重放（`torchtitan/distributed/activation_checkpoint.py:172-181,278-287`）。

### 6.3 SAC 是成对 dispatch mode，不是另一个 autograd

`CheckpointPolicy` 有 MUST/PREFER × SAVE/RECOMPUTE 四个返回值；其中 MUST 语义不能被后续编译器覆盖（`[PyTorch 2.9.1] torch/utils/checkpoint.py:1247-1277`）。forward caching mode 对 SAVE op 缓存输出，recompute cached mode 按调用顺序弹出缓存；选择 RECOMPUTE 的 op 才实际再次执行（`[PyTorch 2.9.1] torch/utils/checkpoint.py:1295-1362`）。两个 mode 共享 storage，并作为 forward/recompute context 对返回（`[PyTorch 2.9.1] torch/utils/checkpoint.py:1365-1448`）。

所以 TorchTitan SelectiveAC 的 policy 只是决定每个 op 走“缓存复用”还是“重新执行”；non-reentrant holder/ticket 仍负责把反向请求接回对应的重算 frame。

---

## 7. GraphTrainer：相关但独立的 memory policy

GraphTrainer 不应被描述成调用上述 eager `ActivationCheckpointing.apply()`。它的模型 parallelize 路径接受 `ac_config` 参数但实际流程是 annotate、TP、simple FSDP、compile，没有调用 `ac_config.build(...).apply(...)`（`torchtitan/experiments/graph_trainer/llama3/parallelize.py:28-70`）。其内存策略是编译 joint graph 上的标签 pass。

GraphTrainer 配置暴露 `default`、`full`、`eager`、`sac_and_offload` 四个 `memory_policy`，并分别说明默认 selective、全重算、模拟 eager SAC、以及 SAC + CPU offload 的含义（`torchtitan/experiments/graph_trainer/configs.py:97-117`）。pass 给 FX joint graph 节点写 `MUST_SAVE`、`MUST_RECOMPUTE`、`MUST_CPU_OFFLOAD` 标签，并复用通用 `_get_default_save_ops()`（`torchtitan/experiments/graph_trainer/memory_policy.py:7-48`）。

四个策略在 registry 中分别注册：default 还强制保存 FSDP unshard 通信，full 只保留层边界，eager 模拟交替 matmul，sac_and_offload 添加 CPU offload 选择（`torchtitan/experiments/graph_trainer/memory_policy.py:344-430`）。这条机制拥有整张 joint graph，能看见跨 eager block 的数据依赖；代价是依赖 compile 图捕获与节点标注，不是通用 eager trainer 的直接替换。

---

## 8. 选择指南

| 目标/约束 | 优先策略 | 原因与边界 |
|---|---|---|
| 需要最直接、规则的显存换算力 | FullAC | 每块完整重放；昂贵 op 也重算 |
| eager trainer，希望保留部分 GEMM/attention/通信结果 | SelectiveAC | op policy 可扩展；需避开当前 FlexAttention + `spmd_types` 禁配 |
| 已 compile，希望用连续预算探索保存/重算点 | MemoryBudgetAC | 预算交给编译器分区器；必须启用 compile |
| 使用 GraphTrainer joint graph | GraphTrainer memory policy | 它是图标签 pass，不走 eager AC wrapper |
| 使用 Flux 当前 parallelize | 谨慎 | 非空 AC 配置都会落到其私有 full wrapper，策略名不被区分 |

设计上没有“免费显存”：FullAC 提高重算 FLOPs；SelectiveAC 保存昂贵结果却增加 activation cache；MemoryBudgetAC 把决策质量交给编译器图与成本模型；GraphTrainer 获得全图视野但要求另一条训练运行时。正确比较应同时观测峰值显存、step time、compile 时间和实际重算算子，而不是只看配置名称。

---

## 9. 小结

- 当前通用入口是策略对象 union，不是旧 `apply_ac` mode switch。
- Full/Selective 是逐 block non-reentrant wrapper；MemoryBudget 是编译器全局预算，机制不可混写。
- 真实顺序是 TP/SPMD → AC → compile → FSDP；PP 对每个 stage-local model part 重走该顺序。
- Selective 的核心状态是 forward/recompute 分离计数、默认 save-op 集合、同形状强制重算与副作用 op 保护。
- activation checkpoint 与训练状态 checkpoint 只共享“checkpoint”一词；owner、数据流和时间尺度均不同。
- GraphTrainer 的 memory policy 是 joint graph 标签 pass，应作为相关但独立机制阅读。

## Related Pages

- [[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]] —— 系列入口与当前源码基线。
- [[11_torchtitan_fsdp_analysis]] —— AC wrapper 之后的参数分片与通信生命周期。
- [[12_torchtitan_tp_analysis]] —— AC 之前的 TP/SPMD 布局与边界 collective。
- [[23_torchtitan_compute_memory_optimizations_analysis]] —— compile、fused operator 与其他计算/显存优化的组合边界。
- [[24_torchtitan_comm_optimizations_overlap_analysis]] —— Selective save-op 集合涉及的通信与 overlap 背景。
- [[27_torchtitan_graph_trainer_compiler_runtime_analysis]] —— GraphTrainer joint graph、编译与运行时主线。
- [[01_theory/02_pretraining/12_activation_checkpointing_analysis|Activation Checkpointing 理论]] —— 重计算的通用成本模型与算法背景。
