---
title: "激活重计算：策略对象、逐块缓存边界与编译器内存预算"
---

# 激活重计算：策略对象、逐块缓存边界与编译器内存预算

> **代码基准**：pytorch/torchtitan `main` @ `a3168782c9a3a2e40afbd0de114818b96e2bda6e`（2026-08-26）
> **最后更新**：2026-08-27 · **系列**：[[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]]
>
> **本页回答**：当前 `none/full/selective/memory-budget` 四个配置分支怎样落到三类 policy object，AC 为什么安装在 Module layout 之后、compile/FSDP 之前，Selective 如何按 op、FQN 和 matmul shape 决定保存或重算，以及 RNG、MoE/EP、SPMD typechecking、GraphTrainer 的真实组合边界。
>
> **边界**：本页只分析一个训练 step 内的 activation 保存/重算；权重与优化器的持久化 checkpoint 不是同一机制。PyTorch checkpoint wrapper 的 holder/cache 内部实现也不冒充 TorchTitan 代码；这里只写 TorchTitan 当前传入的 policy、状态与 guard。

---

## 1. Overview

### ① 背景/问题

激活重计算不是一个布尔开关。Full AC 希望只保留 block 边界并在 backward 完整重放；Selective AC 希望缓存昂贵或有副作用的 op 输出、重算其余 op；MemoryBudget AC 则让编译器分区器在整个 compiled region 上选择保存集。把三者塞进 `mode: str` 与一组共享字段，会允许大量没有语义的字段组合，也很难让模型或硬件实验替换 selective save policy。

### ② 为什么这么设计

当前选择是 `Configurable` policy hierarchy 加显式 union；明显替代方案是旧的 `apply_ac(model, mode, ...)` free function。提交 `c5d93d1098fe` 的正文明确把 AC 定位为硬件/内存预算 knob，而非模型属性：policy class 让每条策略拥有相关字段，并以继承 `SelectiveAC.get_save_ops()` 作为 typed extension point，不需要 registry 或字符串间接层。

### ③ 实现思路与细节

| 配置分支 | 实际对象 | 当前机制 | 默认 Trainer 状态 |
|---|---|---|---|
| `none` | `None`，不是 policy class | model parallelize 函数跳过 `build().apply()` | 非默认 |
| `full` | `FullAC.Config → FullAC` | 每个 transformer block 装 full checkpoint wrapper | 非默认 |
| `selective` | `SelectiveAC.Config → SelectiveAC` | 每块 wrapper 内用 op policy 决定保存/重算 | **默认** |
| `memory-budget` | `MemoryBudgetAC.Config → MemoryBudgetAC` | 写 functorch process-global memory budget，不包 block | 非默认 |

union 的四个 tyro subcommand 定义在 `torchtitan/distributed/activation_checkpoint.py:333-341`；Trainer 默认构造 `SelectiveAC.Config`（`torchtitan/trainer.py:65-107`）。`None` 的实际分支由模型 parallelize 中的 `if ac_config is not None` 实现（`torchtitan/models/llama3/parallelize.py:40-55`）。

可追踪调用链是：

```text
Trainer.Config.activation_checkpoint
  -> model_spec.parallelize_fn 或 PP stage-local parallelize_fn
  -> model.parallelize               布局与状态声明先落地
  -> ac_config.build().apply          wrapper 或全局预算
  -> apply_compile                    捕获 AC 后的模型
  -> apply_fsdp_to_decoder            最后安装 FSDP
```

Llama 3 的当前顺序逐行固定在 `torchtitan/models/llama3/parallelize.py:40-78`；PP 先切 `model_parts`，再对每个 stage/chunk 调同一个 parallelize function（`torchtitan/distributed/pipeline_parallel.py:96-123`）。

### ④ 约束/边界

通用 `ActivationCheckpointing.apply()` 假定模型有 `layers` submodule，并只包装其直接 children；它不是递归包所有叶模块（`torchtitan/distributed/activation_checkpoint.py:146-163`）。MemoryBudget 重写 `apply()`，所以不要求 `layers`，也不安装 block wrapper（`torchtitan/distributed/activation_checkpoint.py:290-330`）。

seed-checkpoint 创建路径会跳过整个 model `parallelize_fn`，因此 AC、compile、nD parallelism 和 mixed precision 都不安装（`torchtitan/trainer.py:472-490`）。“Trainer 配了 AC”不等于所有构建模式都执行了 AC。

### ⑤ 发展趋势（有锚点的推断）

提交 `c5d93d1098fe` 已用 policy hierarchy 替代 flat config；当前 union 与调用点继续沿用该结构（`torchtitan/distributed/activation_checkpoint.py:112-163`、`:333-341`）。**推断**：扩展方向是增加 policy subclass 或覆盖 save set，而不是恢复 mode switch；源码没有承诺新增第五种策略。

---

## 2. 安装时机：为什么是 layout → AC → compile → FSDP

### ① 背景/问题

AC wrapper 需要看到最终 transformer block 的 forward；compile 需要看到 checkpoint/context 边界；FSDP 又要包住完整的 block 调用和重算。若先 compile 再插 wrapper，compiled graph 看不到 AC；若 AC 早于 Module layout，重算时可能重走未安装的 TP/SPMD boundary；若 FSDP 先于所有改写，wrapper 与参数 materialization 的嵌套关系更难推理。

### ② 为什么这么设计

选择让模块布局先落地、AC 再定义重算边界、compile 捕获二者、FSDP 最外层管理参数。明显替代方案是把 AC 放在 compile 之后或 FSDP 之后。源码注释直接把 compile 时机描述为“after AC wrapping and before FSDP”（`torchtitan/models/llama3/parallelize.py:46-58`）；这是显式生命周期契约，不是知识库猜测。

### ③ 实现思路与细节

- `spmd_types` 或 TP 开启时先调用 `model.parallelize(parallel_dims)`，把 state/input/output contract 降成实际 forward（`torchtitan/models/llama3/parallelize.py:40-44`）。
- AC 非空时只把 `dump_folder` 交给 `Config.build()`，随后 `apply(model)`；旧页若写成 `build(job_config, parallel_dims)` 已不符合 HEAD（`torchtitan/models/llama3/parallelize.py:46-47`）。
- model compile 随后逐 block 应用，FSDP 最后使用 dense/sparse mesh 安装（`torchtitan/models/llama3/parallelize.py:49-78`）。
- Qwen3.5 同样先 layout、再用同一 AC policy 分别 apply language model 与可选 vision encoder、再分别 compile，最后才进入 FSDP（`torchtitan/models/qwen3_5/parallelize.py:55-104`）。
- PP 路径先 split，再对每个 `model_part` 执行相同局部顺序，并把改写后的 part 回填 stage（`torchtitan/distributed/pipeline_parallel.py:96-123`）。

### ④ 约束/边界

该顺序是默认 Trainer 的 common decoder parallelize contract，不代表所有模型/实验 trainer。GraphTrainer 的 Llama 路径忽略传入的 `ac_config`，实际顺序是 FQN annotate → TP → SimpleFSDP → compile（`torchtitan/experiments/graph_trainer/llama3/parallelize.py:28-70`）。Flux 则只把“非空配置”解释为私有 full wrapper，且在自身 layout 前应用（`torchtitan/models/flux/parallelize.py:42-58`、`:170-185`）；不能从 common decoder 顺序推断所有模型都走同一 owner。

PP 只把 block 边界局部化到 stage/model chunk；没有跨 pipeline stage 的 checkpoint wrapper。**知识库推断**：这是 split-first、part-local parallelize 调用链的直接结果（`torchtitan/distributed/pipeline_parallel.py:96-123`），不是额外的跨 stage AC 算法。

### ⑤ 发展趋势（有锚点的推断）

提交 `c5d93d1098fe` 把 callsite 统一为 `ac_config.build(...).apply(model)`，并把 MemoryBudget 的 compile validation 移到 Trainer Config；common decoder 当前仍遵循 AC-before-compile/FSDP（`torchtitan/models/llama3/parallelize.py:40-78`）。**推断**：该顺序是公共 decoder 主线，但 Flux 的当前反例说明尚未完成全模型统一。

---

## 3. Full 与公共 wrapper contract：规则重放、RNG 和 determinism

### ① 背景/问题

最容易预测的显存换算力方案是只保存 block 边界，backward 时完整重放 block。但随机算子必须尽量复现 forward，重算输出结构也需要可选检查；否则 dropout、随机路由或数据依赖分支可能让重算图漂移。

### ② 为什么这么设计

FullAC 选择整块 non-reentrant wrapper，并固定 `early_stop=False`；明显替代方案是允许 wrapper 恢复完所需 tensor 后提前停止。**知识库推断**：完整重放使 block 级执行边界更规则，代价是可能多重算后半段 op。当前源码只证明 `early_stop=False` 被固定传入，不声称它一定更快（`torchtitan/distributed/activation_checkpoint.py:166-182`）。

### ③ 实现思路与细节

公共 Config 有三个 knobs：`preserve_rng_state=True`、`determinism_check="default"`、`debug=False`（`torchtitan/distributed/activation_checkpoint.py:120-140`）。Full 和 Selective 都把三者原样交给 PyTorch wrapper，并固定 `early_stop=False`（`torchtitan/distributed/activation_checkpoint.py:173-182`、`:278-287`）。

- `preserve_rng_state` 的 TorchTitan 契约是要求 wrapper stash/restore RNG，默认开启但可能变慢（`torchtitan/distributed/activation_checkpoint.py:122-128`）。
- `determinism_check` 与 `debug` 的具体合法值和检查协议委托给所安装的 PyTorch；TorchTitan 不在本文件内枚举或预验证它们（`torchtitan/distributed/activation_checkpoint.py:130-140`）。
- `None` 不构建 wrapper，因此这三个字段不存在运行时作用；MemoryBudget 虽因继承在 Config 表面拥有它们，但其 `apply()` 不读取这些字段（`torchtitan/distributed/activation_checkpoint.py:290-330`）。

GPU unit test 用相同 toy block 比较 No AC、Selective、FQN shape 强制重算和 Full 的 backward FLOPs：Full 的重算量最大，且各策略梯度与 reference 对齐（`tests/unit_tests/gpu/test_activation_checkpoint.py:46-96`、`:152-216`）。

### ④ 约束/边界

`preserve_rng_state=True` 只覆盖 wrapper 所支持的 RNG 回放，不会把非随机但 backend-nondeterministic 的 op 自动变确定。MoE `topk` 因而由 selective policy 单独 MUST_SAVE，而不是靠 RNG state（`torchtitan/distributed/activation_checkpoint.py:39-53`）。

TorchTitan 当前没有在 Config 层验证任意 `determinism_check` 字符串，也没有根据 backend 自动关闭 `debug`；这些参数的最终兼容性属于所安装 PyTorch wrapper。页面不再沿用未在当前 TorchTitan baseline 冻结的外部 PyTorch 行号来宣称额外 guard。

### ⑤ 发展趋势（有锚点的推断）

提交 `c5d93d1098fe` 删除了 `early_stop` 用户字段并把两条 eager policy 都固定为 false；HEAD 仍保持这一行为（`torchtitan/distributed/activation_checkpoint.py:166-182`、`:278-287`）。**推断**：当前扩展面集中在 save policy 与 RNG/determinism knobs，而不是暴露 wrapper 的全部底层参数。

---

## 4. Selective：op 保存集、FQN→shape 与每次调用的 cache policy

### ① 背景/问题

Full AC 会重算 attention、GEMM 和通信，节省显存但增加昂贵工作。Selective 的目标是在同一个 block checkpoint 内保存高成本或不能安全重做的 op 输出，同时重算便宜 op。难点是同一 op 会出现多次、forward 与 recompute 的调用计数必须一致，模型配置又更容易按模块 FQN 指定例外而不是直接写底层 op 次序。

### ② 为什么这么设计

当前路线是集中式默认 op save set + 可覆盖 `get_save_ops()` + FQN 匹配转成 matmul RHS shape；明显替代方案是每个模型维护自己的 op list，或旧的按层频率 selective。提交 `114151ad4d3c` 明确删除 layer-frequency 与 per-model save list，把 per-op SAC 集中到公共策略；提交 `c5d93d1098fe` 又以 subclass override 取代 registry/string indirection。

### ③ 实现思路与细节

默认 save set 由 PyTorch `get_default_op_list().compute_intensive_ops` 加显式 compute/comm ops 组成（`torchtitan/distributed/activation_checkpoint.py:31-94`）：

- SDPA variants、FlexAttention、linear、低精度缩放用 max、topk、可选 Inductor HOP 与 varlen attention；
- reduce-scatter、AllToAll，以及环境中已注册的 DeepEP/HybridEP dispatch/combine；
- 可选 op 通过逐段 `getattr` 解析，缺扩展时安静跳过，而不是让配置构造失败（`torchtitan/distributed/activation_checkpoint.py:72-94`）。

FQN/shape 路径不是精确模块白名单：

1. 默认 substring `moe.router.gate` 与每个 `layers.<id>` base FQN 拼接，遍历 block 内 `named_modules()`（`torchtitan/distributed/activation_checkpoint.py:194-230`）。
2. 命中对象必须是 `nn.Linear`，否则报错；代码把 weight `(out,in)` 归一成 mm RHS `(in,out)`（`torchtitan/distributed/activation_checkpoint.py:231-241`）。
3. 后续任何 RHS shape 相同的 `aten.mm` 或 `aten.linear` 都 `PREFER_RECOMPUTE`，不限于原 FQN（`torchtitan/distributed/activation_checkpoint.py:256-266`）。GPU test 验证 gate 与 wq 同为 `(512,512)` 时会一起重算（`tests/unit_tests/gpu/test_activation_checkpoint.py:218-267`）。

cache policy 的 TorchTitan 状态是：`context_fn` 每次调用 `_get_custom_policy()`，新建 `{forward_mm_count, recompute_mm_count}`；两个阶段独立计数。save set 中的 op 通常 `MUST_SAVE`，但每第二个 mm/linear 改为 `PREFER_RECOMPUTE`；其他 op 默认也 `PREFER_RECOMPUTE`（`torchtitan/distributed/activation_checkpoint.py:243-287`）。实际 tensor cache 的存取顺序由 `create_selective_checkpoint_contexts()` 实现，TorchTitan 只提供 policy，不维护第二套 cache 容器。

### ④ 约束/边界

- FQN 匹配用 substring `any(f in fqn)`，不是 exact/glob；同 shape 碰撞是明确文档化行为（`torchtitan/distributed/activation_checkpoint.py:194-207`、`:226-237`）。
- forward/recompute 虽分开计数，但策略仍依赖 op 调用序一致；`determinism_check` 只是传给 wrapper，不能把动态控制流自动变稳定。
- CUDA→CPU `_to_copy` 总是 `MUST_SAVE`，避免在 recompute 重做例如 AllToAll metadata 的 D2H 同步（`torchtitan/distributed/activation_checkpoint.py:243-255`）。
- 子类可覆盖 save set，但 Config union 默认只暴露公共 `SelectiveAC.Config`；自定义 subclass 应在 Python config 中提供自己的 Config（`torchtitan/distributed/activation_checkpoint.py:112-118`、`:209-212`）。

### ⑤ 发展趋势（有锚点的推断）

从 `114151ad4d3c` 的集中式 policy 到 `c5d93d1098fe` 的 subclass seam，演进已从“模型各写 op list”转向“公共默认 + typed override”。**推断**：模型特例更可能通过 override/FQN shape 进入，而不是恢复 layer-frequency 模式；HEAD 没有 layer-frequency 配置字段（`torchtitan/distributed/activation_checkpoint.py:185-212`）。

---

## 5. MoE 与 EP：保存路由/通信，Minimal 强制 full

### ① 背景/问题

MoE 重算比 dense block 更敏感：router 若在 recompute 给出不同 expert assignment，梯度会沿另一条路径；EP dispatch/combine 若被重做会再次通信；某些 dispatcher 还使用固定对称内存和有状态 ping-pong buffer，不能任意套 selective cache 假设。

### ② 为什么这么设计

Selective 对 standard/DeepEP/Hybrid 的路线是保存路由关键结果和昂贵通信输出；MinimalAsyncEP 则直接要求整块 full recompute。明显替代方案是让所有 dispatcher 共用同一 SAC op list。当前决定性标准是 backend 是否有能安全表达为 save-op 的调用边界，或其 buffer 生命周期是否专为完整重放设计。

历史证据更直接：提交 `7074b056aa5e` 记录 topk 在部分 backend 上不确定会令 SAC 路由漂移，因此把 `aten.topk` 加入 save set；提交 `7b579addea35` 把 MinimalAsyncEP 的设计前提列为“only for full recompute”，并用双 buffer ping-pong 避免 recompute 时复制接收 buffer。

### ③ 实现思路与细节

- router `topk` 是 `MUST_SAVE`，使 recompute 复用 forward 的 expert assignment（`torchtitan/distributed/activation_checkpoint.py:39-53`）。
- standard EP 的 `all_to_all_single` 和 reduce-scatter 输出进入默认 save set；AllToAll split metadata 的 CUDA→CPU copy 也在运行 policy 中强制保存（`torchtitan/distributed/activation_checkpoint.py:60-70`、`:243-255`）。
- DeepEP/HybridEP ops 仅在对应 `torch.ops` 已注册时加入；缺可选依赖会静默跳过（`torchtitan/distributed/activation_checkpoint.py:64-94`）。这说明 policy 有组合入口，不等于任意扩展版本都已测试。
- 当前 Qwen3 DeepEP v2 recipe 明确使用 `SelectiveAC.Config`，H100 集成项再把它组合为 FSDP4+EP4（`torchtitan/models/qwen3/config_registry.py:417-459`、`torchtitan_recipes/tests/h100.py:89-95`、`tests/integration_tests/h100.py:73-78`）。
- Minimal 配置更新要求 EP>1，并硬性检查 eager `FullAC.Config` 或 GraphTrainer `compile.memory_policy == "full"`；否则立即报错（`torchtitan/distributed/minimal_async_ep/api.py:84-140`）。当前 H100 Minimal recipe 也显式设置 FullAC（`torchtitan_recipes/tests/h100.py:63-77`）。

### ④ 约束/边界

FullAC 没有 selective save context。**知识库推断**：因此 block 内 router 和 EP communication 会随完整 forward 重放；这正是 Minimal buffer 设计所接受的契约，但会增加其他 dispatcher 的重通信成本（`torchtitan/distributed/activation_checkpoint.py:166-182`、`torchtitan/distributed/minimal_async_ep/api.py:131-140`）。

DeepEP Selective 的证据边界是一个当前 recipe 与 H100 集成配置，不是所有 DeepEP 拓扑的数值证明。Hybrid 当前 H100 recipe 保留 DeepSeek debug 基配置的 SelectiveAC 并启用 compile，但集成描述主要验证 FSDP+HybridEP+compile，不应扩写为专门 SAC correctness suite（`torchtitan/models/deepseek_v3/config_registry.py:55-73`、`torchtitan_recipes/tests/h100.py:80-86`、`tests/integration_tests/h100.py:49-55`）。

当前 AC unit test 只用 toy linear/MoE FQN 验证 FLOPs、memory、gradients 与 shape policy，没有真实 EP collective（`tests/unit_tests/gpu/test_activation_checkpoint.py:17-96`、`:218-267`）。

### ⑤ 发展趋势（有锚点的推断）

`7074b056aa5e` 通过保存 topk 修复 SAC 路由漂移；`7b579addea35` 则选择让 Minimal 只支持 full recompute。**推断**：当前演进准则是按 backend 的副作用/缓冲契约选择“保存”或“整块重放”，而不是强迫所有 EP 后端共享一种 SAC 行为。

---

## 6. SPMD typechecking、FlexAttention、PP 与 Dynamo cache

### ① 背景/问题

Selective wrapper、FlexAttention HOP、SPMD checker 与 PP dynamic microbatch graph 会同时改变执行上下文。save set 中出现 FlexAttention op，并不意味着 checker 能理解所有缓存/重算边界；Dynamo 在同一 code object 上保留 static/dynamic 两张图时，还可能选择与 SAC cache 不匹配的图。

### ② 为什么这么设计

当前选择是对已知不安全组合早失败，并临时关闭 Dynamo LRU；明显替代方案是允许组合运行，等待晚期 assertion 或错误梯度。Trainer guard 只拒绝 `spmd_types + typechecking + SelectiveAC + FlexAttention`，不是拒绝所有 Selective+Flex（`torchtitan/trainer.py:141-154`）。

### ③ 实现思路与细节

- 若上述四个条件同时满足，Config 初始化要求改用 Full、None 或非 Flex attention（`torchtitan/trainer.py:141-154`）。
- SPMD typechecking 与 PP>1 还有独立 guard，不论 AC 策略都会失败（`torchtitan/trainer.py:129-139`）。不要把它误写成 Selective 专属限制。
- 任一 Full/Selective `apply()` 都先调用 `_disable_dynamo_lru_cache()`；MemoryBudget override 也显式调用（`torchtitan/distributed/activation_checkpoint.py:152-163`、`:319-330`）。
- workaround 的注释记录了 SAC+PP+Flex 下第二个 microbatch 触发 dynamic recompilation，LRU 可能选中缺少已缓存 symint 的图；关闭 LRU 后按 insertion order 选图（`torchtitan/distributed/activation_checkpoint.py:97-109`）。

### ④ 约束/边界

关闭 Dynamo LRU 是 process-global 副作用，不绑定某个 model；同进程多 trainer/多模型会共享该变化。`None` 不调用 policy `apply()`，因此也不会触发 workaround（`torchtitan/models/llama3/parallelize.py:46-47`）。

测试 helper 在启用 `spmd_types` typechecking 时直接把 AC 设为 None，因为 debug models 使用 FlexAttention，且 typechecking 另有 compile/PP 限制（`torchtitan_recipes/tests/__init__.py:20-30`）。这是测试矩阵的保守选择，不是框架声称 FullAC 也不支持。

当前集成基础设施还记录 FSDP+SelectiveAC backward recompute 在 Fake PG + spmd_types 下有 shard/storage shape mismatch，并把对应 test 保留在 real PG（`tests/integration_tests/__init__.py:65-85`）。这是一条测试后端限制，不能泛化成真实 PG 不支持。

### ⑤ 发展趋势（有锚点的推断）

Trainer 对 Selective+Flex+typechecking 留有显式 Enable TODO，Dynamo workaround 链接上游 issue（`torchtitan/trainer.py:141-154`、`torchtitan/distributed/activation_checkpoint.py:97-109`）。**推断**：两项都是待解除的兼容层，不是长期 API 承诺；源码没有完成日期。

---

## 7. MemoryBudget：编译器全局预算，不是第四种 block wrapper

### ① 背景/问题

逐 block policy 只能在 eager checkpoint 边界内选择保存集；编译器若拥有完整 region 的 forward/backward 图，可以用成本模型在更连续的内存预算上选 partition。把两层策略叠加会让 compiler 只看到已被 wrapper 切碎的区域，并产生难解释的双重重算边界。

### ② 为什么这么设计

MemoryBudget 选择只设置 functorch partitioner 的 process-global budget，不遍历 `layers`、不安装 wrapper；明显替代方案是先 Selective/Full 包 block 再让 compiler 二次 partition。**知识库推断**：当前互斥机制把保存/重算决策完整交给 compiler，决定性要求是 model compiled region 必须存在（`torchtitan/distributed/activation_checkpoint.py:290-330`、`torchtitan/trainer.py:156-163`）。

### ③ 实现思路与细节

- `memory_budget` 默认 0.5；0 表示 full compiled-region AC 的激活内存，1 表示默认 runtime-optimized strategy 的激活内存（`torchtitan/distributed/activation_checkpoint.py:296-305`）。
- Config `__post_init__` 要求 inclusive `[0,1]`；比较也会拒绝 NaN/±Inf。提交 `9854306021a9` 的正文解释早失败是为了防止非法值污染 process-global functorch config（`torchtitan/distributed/activation_checkpoint.py:315-317`）。
- 可选 Pareto visualization 创建 `{dump_folder}/memory_budget_pareto` 并设置 functorch 全局 dump 选项；最后写 `activation_memory_budget`（`torchtitan/distributed/activation_checkpoint.py:319-330`）。
- 默认 Trainer 和 Forge 都要求 `compile.enable` 且 `"model" in compile.components`，否则 Config 构造失败（`torchtitan/trainer.py:156-163`、`torchtitan/experiments/forge/engine.py:61-69`）。

### ④ 约束/边界

MemoryBudget Config 继承 `preserve_rng_state/determinism_check/debug`，但当前 `apply()` 不读取它们；不要把这些 eager wrapper knobs 写成 compiler budget 的有效控制面（`torchtitan/distributed/activation_checkpoint.py:120-140`、`:290-330`）。

budget 与 Pareto settings 是进程级 `torch._functorch.config`，不是 model-local 状态；同进程后续 compile 会看到这些值。当前源码没有 restore-on-exit 逻辑（`torchtitan/distributed/activation_checkpoint.py:319-330`）。

### ⑤ 发展趋势（有锚点的推断）

提交 `9854306021a9` 刚把范围检查从延迟 compiler 失败前移到 owning Config。**推断**：演进标准是阻止非法全局状态写入，而不是在 apply 时容错或 clamp；当前不存在自动修正预算的 fallback。

---

## 8. GraphTrainer：同名配置字段，独立的 joint-graph memory policy

### ① 背景/问题

GraphTrainer 继承默认 Trainer Config，因此仍能看到 `activation_checkpoint` 字段；但它的训练图拥有 forward+loss+backward joint graph，可以直接给 FX nodes 标记保存、重算或 CPU offload。若把它描述成调用 eager `ActivationCheckpointing.apply()`，会错误理解 FSDP 时序、RNG 处理和 MinimalAsyncEP 的配置条件。

### ② 为什么这么设计

GraphTrainer 选择在 joint graph 上做 memory-policy pass；明显替代方案是复用 eager block wrapper。全图路线能看见 layer boundary、FSDP unshard 与 backward consumer，代价是依赖 tracing/compile graph，不能直接作为默认 Trainer eager policy 的实现细节（`torchtitan/experiments/graph_trainer/memory_policy.py:201-220`、`:344-430`）。

### ③ 实现思路与细节

- Llama GraphTrainer parallelize 虽接受 `ac_config`，却没有 build/apply；实际执行 FQN annotate、TP、SimpleFSDP、compile（`torchtitan/experiments/graph_trainer/llama3/parallelize.py:28-70`）。
- `compile.memory_policy` 有 `default/full/eager/sac_and_offload`：default 保存 compute-intensive 与必要 FSDP unshard，full 标记整层重算，eager 模拟逐层交替 mm，offload 在 default SAC 后按 CPU budget 迁移保存激活（`torchtitan/experiments/graph_trainer/configs.py:97-117`、`torchtitan/experiments/graph_trainer/memory_policy.py:344-430`）。
- `full` 支持 `FQN_PATTERN::OP` save exceptions；非法 selector 或非 full policy 携带该字段会在 trace 前报错（`torchtitan/experiments/graph_trainer/memory_policy.py:75-128`）。
- Graph full policy把 tagged nondeterministic RNG op 设为 MUST_SAVE，因为 remat path 不回放 RNG state；其他节点默认 MUST_RECOMPUTE（`torchtitan/experiments/graph_trainer/memory_policy.py:131-166`）。
- 非 joint `forward-loss-backward` remat 若发现 recompute region 中 RNG op，会直接报错并要求移出或使用 joint graph；多个不相交 remat region也被拒绝（`torchtitan/experiments/graph_trainer/selective_activation_remat.py:101-150`）。

### ④ 约束/边界

GraphTrainer 的 `memory_policy=full` 与 eager `FullAC.Config` 语义目标相似但执行机制不同：前者是 node tags/remat，后者是 block checkpoint wrapper。MinimalAsyncEP guard 有意接受两者之一（`torchtitan/distributed/minimal_async_ep/api.py:131-140`）。

GraphTrainer 默认 memory policy 还根据 FSDP `reshard_after_forward` 决定是否强制保存 unshard nodes（`torchtitan/experiments/graph_trainer/memory_policy.py:344-363`）。因此不能把 eager `_get_default_save_ops()` 复制成 GraphTrainer 的完整行为；它只贡献基础 op set。

### ⑤ 发展趋势（有锚点的推断）

提交 `36d38d3575a4` 集成 remat AC；当前 registry 已允许新增 memory policy 而不修改 dispatcher（`torchtitan/experiments/graph_trainer/memory_policy.py:406-430`）。**推断**：GraphTrainer 的扩展面是 graph pass/registry，不是 eager policy subclass；两条路线不会因共享 save-op helper 就自动合并。

---

## 9. 选择准则与旧断言纠偏

### ① 背景/问题

只看策略名会忽略最关键的组合条件：是否 compile、是否需要全局 graph、是否有 EP stateful buffers、是否启用 SPMD checker，以及能否容忍通信重放。

### ② 为什么这么设计

应按“决策者是谁、边界多大、哪些副作用可重放”选择，而不是把 selective 当作总是优于 full。明显替代方案是以单一显存百分比排序；它忽略重算 FLOPs、通信、cache 驻留、compiler cost model 和进程级副作用。以下指南是基于当前 guard/call chain 的**知识库推断**。

### ③ 实现思路与细节

| 条件 | 当前选择 | 决定性理由 |
|---|---|---|
| 不需要 AC | `None` | 完全跳过 policy apply 与 Dynamo workaround |
| 规则 block 重放、MinimalAsyncEP | `FullAC` | Minimal 有硬 guard；计算/通信均可能重放 |
| 默认 eager dense、standard EP、代表 DeepEP | `SelectiveAC` | 默认 save set 保留昂贵 compute/comm，便宜 op 重算 |
| compiled model、连续预算探索 | `MemoryBudgetAC` | compiler partitioner 决策；必须 compile model component |
| GraphTrainer | `compile.memory_policy` | joint graph tags/remat，不走 eager policy apply |
| Flex + spmd typechecking | Full 或 None | Selective 有明确 Trainer guard |

旧断言应改为：

- 现行通用入口不是 `apply_ac(mode=...)`，而是三类 policy object 加 `None` union（`torchtitan/distributed/activation_checkpoint.py:112-163`、`:333-341`）。
- `early_stop` 不再是用户字段，Full/Selective 都固定 false（`torchtitan/distributed/activation_checkpoint.py:166-182`、`:278-287`）。
- Selective 不再有 layer-frequency mode 或每模型 op list；当前是公共 save set、subclass seam 与 FQN→shape 例外。
- MemoryBudget 不包 block，也不消费 RNG/determinism knobs；它写 compiler 全局配置。
- GraphTrainer memory policy 不是默认 Trainer AC 的第四个实现类。
- DeepEP 当前有 Selective 代表组合；MinimalAsyncEP 当前必须 full，不能把所有 EP backend 写成同一 AC 兼容矩阵。

### ④ 约束/边界

页面没有运行 GPU/多机实验；H100 recipe 与 integration entry 只说明组合被纳入提交矩阵。性能取舍必须同时测峰值显存、step time、compile 时间、重算 FLOPs、重通信量与 cache 驻留，源码不提供跨模型统一赢家。

Flux 仍有模型私有 `apply_ac()`，其行为不等同于通用 policy hierarchy（`torchtitan/models/flux/parallelize.py:42-58`、`:170-185`）。由于本页主线是默认 Trainer/common AC，Flux 只作为“树中仍可能有同名私有入口”的边界，不把它扩展成第四条通用路径。

### ⑤ 发展趋势（有锚点的推断）

当前明确的 TODO/演进锚点是 Selective+Flex+typechecking、Dynamo cache workaround、GraphTrainer registry 与 model-private Flux 例外，而不是增加统一 fallback。**推断**：在这些边界消失前，选择矩阵必须保持显式 guard 与测试证据，不能宣称四个配置分支可在所有模型/Trainer 间自由互换。

---

## Related Pages

- [[02_engineering/02_train_frameworks/torchtitan/11_torchtitan_fsdp_analysis|FSDP 分片与生命周期]] — AC wrapper 外层的参数 materialize、reshard 与 backward 时序。
- [[02_engineering/02_train_frameworks/torchtitan/15_torchtitan_ep_analysis|专家并行 EP]] — standard、DeepEP、HybridEP、MinimalAsyncEP 的通信与 buffer 契约。
- [[02_engineering/02_train_frameworks/torchtitan/16_torchtitan_spmd_types_analysis|SPMD Types]] — AC 重算期间必须保持的 current-mesh 与布局断言。
- [[02_engineering/02_train_frameworks/torchtitan/23_torchtitan_compute_memory_optimizations_analysis|计算与显存优化]] — compile、融合 kernel 与 AC 的组合选择。
- [[02_engineering/02_train_frameworks/torchtitan/24_torchtitan_comm_optimizations_overlap_analysis|通信优化与重叠]] — Selective 保存通信输出与 Full 重通信的成本背景。
- [[02_engineering/02_train_frameworks/torchtitan/27_torchtitan_graph_trainer_compiler_runtime_analysis|GraphTrainer 编译器运行时]] — joint graph memory policy 与 remat pass 的完整所有者。
- [[01_theory/02_pretraining/12_activation_checkpointing_analysis|Activation Checkpointing 理论]] — 重计算的通用时间—显存成本模型。
