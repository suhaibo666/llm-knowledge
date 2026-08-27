---
title: "GraphTrainer：用一张 joint FX 图接管训练步的编译控制面"
---

# GraphTrainer：用一张 joint FX 图接管训练步的编译控制面

> **论点**：GraphTrainer 的关键不是把 `torch.compile` 套到更大范围，而是把 forward、loss、backward 与分布式 collective 变成同一张可检查、可重排的 FX 程序；内存、offload、FSDP/TP/EP 调度、Inductor 和 CUDA Graph 因而共享一个控制面。这个选择换来跨系统可见性，也把动态图弹性、组合成熟度和 artifact 兼容责任转移给 tracer 与 pass pipeline。
>
> **代码基准**：pytorch/torchtitan `main` @ `a3168782c9a3a2e40afbd0de114818b96e2bda6e`（2026-08-26）
>
> **最后更新**：2026-08-27 · **状态**：`torchtitan/experiments/graph_trainer` 实验路径
>
> **本页回答**：为什么 joint fwd/loss/bwd graph 胜过 hooks/局部 compile；mode、tracer 与 live state 怎样衔接；pass 顺序为什么不能任意交换；AutoParallel 怎样把 solver placement 接到 local-tensor AOT 边界；precompile 与 GraphPP 各自移动了什么边界；哪些 README/API 主张已被当前可执行路径否定。
>
> **兄弟页边界**：SimpleFSDP 的参数/collective 状态机属于 [[02_engineering/02_train_frameworks/torchtitan/25_torchtitan_simple_fsdp_analysis|SimpleFSDP 分析]]；GraphPP 复用的 stage split 与 schedule 基座属于 [[02_engineering/02_train_frameworks/torchtitan/14_torchtitan_pp_analysis|PP 分析]]；本页只分析 GraphTrainer 的编译与运行时控制面。

---

## 1. Overview：问题不是缺少一个 compile 开关，而是控制面碎裂

### ① 背景 / 问题

Eager 分布式训练把 FSDP hook、autograd、activation checkpoint、`torch.compile` 和 CUDA Graph 放在不同拦截点。GraphTrainer 自己的 Manifesto 记录了这种组合脆弱性：AC 可能在 backward 重做 FSDP all-gather，compile graph break 又可能使 AC 失效并 OOM；eager 的细粒度调度主要依赖 `autograd.Function` 和 hooks，MoE microbatch overlap 很快变成回调时序问题（`torchtitan/experiments/graph_trainer/MANIFESTO.md:15-33`）。

另一个压力来自 kernel launch：Manifesto 把更快 accelerator 下的 CPU launch overhead 与 CUDA Graph 需求列为该实验的直接背景（`torchtitan/experiments/graph_trainer/MANIFESTO.md:3-13`）。这是上游给出的动机；它不等价于“GraphTrainer 已在所有硬件上更快”。

### ② 为什么选择一张 joint graph，而不是 hooks 或局部 model compile

**选中的路线**是捕获 forward + loss + backward 的单一 flat FX graph，让 backward computation、collective 与保存边界都成为普通节点，再把每种优化实现为同一表示上的 pass（`torchtitan/experiments/graph_trainer/MANIFESTO.md:35-56`）。**明显替代方案**是局部 `torch.compile(model)` 加 eager FSDP/AC hooks；它保留成熟的动态 runtime，却无法给一个 pass 同时看到前向保存、反向重算与通信 launch/wait。

决定性标准是 **依赖关系是否全局可见且可重排**，不是“编译范围越大越好”。提交 `c8d2d71812b3ad51a001277e4f7ae388cff35647` 的 commit body 把 simple_fsdp 与 compiler_toolkit 合并的目的明确写成对性能、数值和可调试性的显式 compiler-stack 控制；提交 `a91f57bd4238116c26fb0fd8bed9e0704372a97a` 则把 eager 难以跟上 accelerator scaling 写成 Manifesto 的核心理由。

### ③ 当前状态、关键对象与调用链

| 层次 | 当前对象 | 责任 |
|---|---|---|
| 配置 | `GraphTrainerCompileConfig` | 选择 active mode、memory policy、passes、Inductor、artifact |
| 捕获 | `minimal_fx_tracer` | 把 module state 与用户输入展开为纯函数图输入 |
| 图状态 | `TracedResult` | 保存 FX graph、fake inputs、pytree/subclass layout、state FQN |
| 变换 | ordered pass list | 规范化 → 内存/offload/remat → 通信 → Inductor → CUDA Graph |
| 执行 | `run_traced` | 每步重取 live state，以显式 backward 图执行并重包输出 |
| 梯度出口 | `accumulate_param_grads_` | 将图返回的 grads 写回 Trainer 持有的 live parameters |

最短的非 PP 调用链是：

```text
GraphTrainer.forward_backward_step
  -> model.preprocess_inputs                     # trace 外
  -> make_fwd_bwd_step(model, loss_fn)
  -> minimal_fx_tracer(..., module=model)        # 首步或无 artifact 时
       model -> annotated loss -> autograd.grad
  -> ordered graph passes
  -> run_traced(..., module=model)
  -> [loss, grads...] -> accumulate_param_grads_
  -> core Trainer: clip -> finite gate -> optimizer.step -> scheduler.step
```

生产闭包执行 model、annotated loss 与 `torch.autograd.grad`，显式返回 `[loss] + grads`（`torchtitan/experiments/graph_trainer/trainer.py:75-103`）；首步 trace、pass 与执行/梯度回写位于 `torchtitan/experiments/graph_trainer/trainer.py:206-262`。Optimizer 仍由 core Trainer 在图外执行（`torchtitan/trainer.py:850-889`）。

### ④ 约束、代价与失败边界

GraphTrainer 是实验 Trainer，不是 core Trainer 的默认实现。它继承 core `Trainer`，但只在无 PP 且 mode 为 `aot_fx_trace` 时接管 forward/backward；普通 PP 或其他 mode 直接调用父实现（`torchtitan/experiments/graph_trainer/trainer.py:106-148`）。

整步图还要求输入结构、state FQN、tensor subclass 重包信息与 trace-time 契约保持稳定。运行时验证默认关闭以减少每步开销，因此一部分正确性责任从框架检查移到了调用者（`torchtitan/experiments/graph_trainer/make_fx_tracer.py:532-559`）。

### ⑤ 发展趋势

**有锚点的推断**：Manifesto 将硬件异构、autoresearch 与可调试性列为 graph-first 的继续投入方向（`torchtitan/experiments/graph_trainer/MANIFESTO.md:58-70`）；这些是实验愿景，不是替换 core Trainer 的时间表或性能承诺。

---

## 2. Mode 与接线：active 主线只有 `aot_fx_trace`

### ① 背景 / 问题

GraphTrainer 历史上先后出现 JIT、旧 AOT export 与 `aot_fx_trace`；若把旧文档和当前配置混读，会误以为三条路径都仍可运行，甚至把 model-level JIT 的行为外推到 joint graph。

### ② 为什么保留 JIT 兼容、删除旧 AOT

当前选择是把 `aot_fx_trace` 设为默认 active path，只暂时保留 deprecated JIT。明显替代是长期维护旧 export-AOT、JIT 和 make-fx 三套编译控制面；提交 `1a22c2da13b6588798e7fbc14f8c933efbd7f94f` 的 commit body 说明旧 AOT 已因 disabled tests 和维护负担被删除，保留的是 active `aot_fx_trace` 与 deprecated JIT。

判据是 live tests 与维护成本，而不是名称中的 “AOT”。因此旧 `aot` 与当前 `aot_fx_trace` 不是别名。

### ③ 当前配置、Quick Start 与 dispatch

配置 literal 只有 `jit`、`aot_fx_trace` 与 `None`，默认是 `aot_fx_trace`；注释把它定义为 non-strict make-fx 的 fwd+loss+bwd capture，并把 JIT 标为 deprecated（`torchtitan/experiments/graph_trainer/configs.py:65-82`）。

最小入口可以使用 GraphTrainer 自己的 model/config registry：

```bash
MODULE=graph_trainer.llama3 \
CONFIG=graph_trainer_llama3_debugmodel \
./run_train.sh --compile.mode aot_fx_trace
```

README 给出同类 GraphTrainer module/config 启动方式（`torchtitan/experiments/graph_trainer/README.md:32-49`）；对应 config factory 先把 core config 转成 `GraphTrainer.Config`，再显式设 `GraphTrainerCompileConfig(enable=True)`（`torchtitan/experiments/graph_trainer/llama3/config_registry.py:31-34`）。

`apply_compile()` 在 JIT mode 对 model 调用 `torch.compile` custom backend；在 `aot_fx_trace` 下刻意原样返回 model，因为真正捕获发生在 `forward_backward_step`，artifact 也在那里惰性加载（`torchtitan/experiments/graph_trainer/compile.py:87-119`）。

### ④ 约束、代价与失败边界

当前 README 的 legacy details 仍写“两种 legacy modes”并列出已删除的 `--compile.mode aot`（`torchtitan/experiments/graph_trainer/README.md:16-22`）。这是 baseline 内的文档/代码冲突；现状必须以配置 literal 与删除提交为准。

> [!contradiction] 当前源码推翻旧 mode 表
> README 仍列 `aot`，但 `GraphTrainerCompileConfig.mode` 已不接受它；本页不把历史模式保留为 live 配置。

JIT 虽仍可配置，但 `apply_compile()` 发出 `FutureWarning`（`torchtitan/experiments/graph_trainer/compile.py:87-93`），integration matrix 又因上游 partitioner regression 把全部 JIT cases 统一禁用（`torchtitan/experiments/graph_trainer/tests/integration_tests.py:13-24`）。“可解析”不等于“当前有持续集成保障”。

`compile.enable=False` 或 mode 为 `None` 会让 model compile 成为 no-op，而 GraphTrainer 的 forward/backward 也回退父 Trainer（`torchtitan/experiments/graph_trainer/compile.py:74-85`；`torchtitan/experiments/graph_trainer/trainer.py:136-148`）。

### ⑤ 发展趋势

**有锚点的推断**：JIT warning 明确说未来移除，旧 AOT 已先行删除；因此 mode 正在收敛到 `aot_fx_trace`。源码没有给出 JIT 删除版本（`torchtitan/experiments/graph_trainer/compile.py:87-93`）。

---

## 3. Tracer 与 state：把有状态训练临时改写成纯函数

### ① 背景 / 问题

FX graph 需要显式输入，但 module parameters、buffers、DTensor 内层 tensors 和 PP/EP 的结构化输入并不是普通 positional tensors。直接闭包捕获 live tensors 会把地址或 rank-specific 状态烘进图；把 module/optimizer 对象当参数又不满足 pytree/make-fx 约束。

### ② 为什么选择 state extraction + 临时 reparameterization

选中路线是：闭包捕获 Python 对象，tracer 每次从 live objects 提取 tensor state，把 state 与用户输入 flatten/unwarp 后作为 graph inputs，再在 trace 期间临时换回 module。明显替代是 functionalize 整个 Trainer API 或把 module 本身放进 args；当前 tracer明确拒绝 args 中的 `nn.Module`（`torchtitan/experiments/graph_trainer/make_fx_tracer.py:348-376`）。

判断标准是图签名只含 make-fx 可处理的 tensor/primitive，同时保留 module FQN 和 tensor subclass 布局以便 runtime 重建。

### ③ `TracedResult`、trace 与 replay 状态机

`TracedResult` 保存 graph、fake example inputs、输入/输出 subclass layouts、pytree specs、tensor input indices 与 module state FQNs（`torchtitan/experiments/graph_trainer/make_fx_tracer.py:280-310`）。DTensor 等 traceable wrapper subclass 被递归拆成 plain tensor leaves，并记录 class、attrs、context、outer size/stride 以便重包（`torchtitan/experiments/graph_trainer/make_fx_tracer.py:53-131`）。

Trace 顺序是：

1. 运行可选 `prepare_inputs`，再提取 module/optimizer state，并 flatten state 与用户 pytree（`torchtitan/experiments/graph_trainer/make_fx_tracer.py:390-421`）。
2. 拒绝非法 leaf 与 wrapper subclass 上的 marked dynamic dims；unwrap subclass 后创建 FakeTensor/ShapeEnv（`torchtitan/experiments/graph_trainer/make_fx_tracer.py:403-442`）。
3. 在 `stateless._reparametrize_module`、可选 optimizer swap 与 patched backward 下运行用户闭包（`torchtitan/experiments/graph_trainer/make_fx_tracer.py:448-475`）。
4. 禁用 autograd multithreading，保证 backward trace 留在能看到 CooR ContextVar 的调用线程；用 non-strict `make_fx` 生成 graph（`torchtitan/experiments/graph_trainer/make_fx_tracer.py:477-505`）。
5. 将 forward 的 module/stack metadata 复制给关联 backward nodes，再构造 `TracedResult`（`torchtitan/experiments/graph_trainer/make_fx_tracer.py:507-526`）。

Replay 每步重新提取 live state、拼接用户输入、unwrap subclass，并在 `torch.no_grad()` 下执行 graph；因为 backward 已显式存在，若再让 eager autograd 记录会建立冗余图并持有 forward intermediates（`torchtitan/experiments/graph_trainer/make_fx_tracer.py:532-553`；`torchtitan/experiments/graph_trainer/make_fx_tracer.py:566-608`）。

模型 `preprocess_inputs` 在 trace 外执行，产出的 `inputs/labels/extra_kwargs` 才进入 joint graph；weight tying 则用 `named_parameters(remove_duplicate=False)` 保留重复参数条目（`torchtitan/experiments/graph_trainer/trainer.py:150-176`）。

### ④ 约束、代价与失败边界

所有 state/user pytree leaves 必须是 Tensor 或有限 primitive；自定义结构要注册 pytree node/constant（`torchtitan/experiments/graph_trainer/make_fx_tracer.py:401-418`）。wrapper subclass 的 dynamic dim 当前直接报错（`torchtitan/experiments/graph_trainer/make_fx_tracer.py:420-431`）。

`run_traced(..., _validate_runtime=False)` 默认不检查 runtime state FQN 与 pytree spec，并要求 kwargs 顺序保持 trace-time 插入顺序（`torchtitan/experiments/graph_trainer/make_fx_tracer.py:555-559`；`torchtitan/experiments/graph_trainer/make_fx_tracer.py:566-587`）。这是稳态开销换安全检查的明确取舍。

Tracer API 已能接收 optimizer，并按 param-group 顺序临时换入 optimizer params/state（`torchtitan/experiments/graph_trainer/make_fx_tracer.py:219-275`；`torchtitan/experiments/graph_trainer/make_fx_tracer.py:334-366`），但生产 GraphTrainer 调用只传 `module=model`（`torchtitan/experiments/graph_trainer/trainer.py:220-232`）。`num_static_inputs` 的 TODO 也明确写着 GraphTrainer 尚未 trace optimizer（`torchtitan/experiments/graph_trainer/make_fx_tracer.py:312-323`）。

### ⑤ 发展趋势

**有锚点的推断**：提交 `d4a3c63499dbeb7442a29e2bc5a0a3590af3bbdb` 已把 tracer 泛化到 module+optimizer roots，但当前 TODO 与生产 call site 表明 optimizer 进图仍未接通；Manifesto/README 的“可选 optimizer.step”只能视为 tracer 能力方向，不能写成当前训练链事实。

---

## 4. Pass pipeline：顺序本身就是正确性契约

### ① 背景 / 问题

Joint graph 同时含计算、保存候选、FSDP/EP/TP collective 与 terminal compiler region。若 pass 任意换序，offload 可能看不到 memory tags，bucketing 可能继承错误 process group，GraphPP partition 可能遇到非 canonical 图，terminal Inductor 后也不再有可安全改写的原 FX 结构。

### ② 为什么选择显式 ordered list，而不是独立开关各自生效

选中路线是构造一条 ordered callable list，再逐 pass 更新同一个 `GraphModule`；明显替代是 feature 开关各自注入 hook 或由多个编译器隐式排序。决策准则是依赖顺序可检查、可测试、可按 pass name 做 ablation。`apply_graph_passes()` 先精确过滤 disabled names，再按列表顺序执行，并可记录每步 graph diff（`torchtitan/experiments/graph_trainer/passes.py:451-532`）。

### ③ 当前顺序：memory → communication → terminal compiler → runtime capture

默认 compile-time pipeline 的主要相序是：

1. DCE、canonicalize、SimpleFSDP unshard-chain dedup（`torchtitan/experiments/graph_trainer/passes.py:201-209`）。
2. memory policy tagging、CPU offload rewrite、selective activation remat（`torchtitan/experiments/graph_trainer/passes.py:255-268`）。
3. 可选 EP chunk、EP PG isolation 与再次 DCE（`torchtitan/experiments/graph_trainer/passes.py:269-273`）。
4. 可选 FSDP extra-PG reassignment，随后默认 joint FSDP bucket/reorder（`torchtitan/experiments/graph_trainer/passes.py:274-286`）。
5. 可选 EP overlap schedule、dense-region FSDP schedule、Async TP（`torchtitan/experiments/graph_trainer/passes.py:288-338`）。
6. Regional 或 full Inductor 作为 terminal compile phase（`torchtitan/experiments/graph_trainer/passes.py:340-407`）。
7. 最后追加 CUDA Graph wrapper；有 artifact 时 compile-time passes 已在预编译阶段完成，runtime 只追加 CUDA Graph（`torchtitan/experiments/graph_trainer/passes.py:410-448`）。

Pass-order 单测逐项断言 canonicalize < dedup < memory，offload < remat，EP chunk/PG isolation < FSDP bucket < EP schedule < full Inductor（`torchtitan/experiments/graph_trainer/tests/test_passes.py:3481-3527`）。这说明相序不是文档建议，而是测试契约。

Memory policy 本身以 node metadata 表达保存/重算；默认策略在 FSDP 不 reshard 时强制保存 unshard 输出（`torchtitan/experiments/graph_trainer/memory_policy.py:344-363`）。CPU offload 只选 forward 中已有 MUST_SAVE 或 backward consumer 的候选，按 tensor 大小和 CPU budget 筛选，并跳过最后一层 heuristic（`torchtitan/experiments/graph_trainer/cpu_offload.py:273-341`）。

CUDA Graph pass 是 all-or-nothing wrapper：不兼容时 warning 后保留原 graph；兼容时第一次运行 warmup、第二次 record/replay、以后 replay（`torchtitan/experiments/graph_trainer/cudagraph.py:551-605`）。它把 `TracedResult.num_static_inputs` 对应的 state tensors 视为稳定地址，其余 tensor inputs copy 入固定 buffers（`torchtitan/experiments/graph_trainer/passes.py:439-447`；`torchtitan/experiments/graph_trainer/cudagraph.py:164-225`）。

### ④ 约束、代价与失败边界

`enable_passes=False` 时，非 PP GraphTrainer 不构造任何默认 pass，连 runtime CUDA Graph 也不会追加，因为 pass application 整体位于这个 gate 内（`torchtitan/experiments/graph_trainer/trainer.py:234-250`）。GraphPP 是例外：partition 所需 DCE/canonicalize/dedup 无视该 gate（见 §8）。

Regional mode 只编译标注区域，FlexAttention 必须走 regional Inductor 以匹配 eager numerics；full mode 把整张 graph 收为一个 compiled region，源码明确要求它是最后一个 FX-level pass（`torchtitan/experiments/graph_trainer/passes.py:352-407`；`torchtitan/experiments/graph_trainer/inductor_passes.py:329-409`）。

Extra-PG FSDP overlap 与 dense-region FSDP schedule 默认都是 false；后者改变 collective placement，且只在特定 EP graph-chunking 组合下生效（`torchtitan/experiments/graph_trainer/configs.py:149-163`）。因此“GraphTrainer 默认把所有通信都重叠”是错误结论。

### ⑤ 发展趋势

**有锚点的推断**：CUDA Graph compatibility helper 留有移除 all-or-nothing path 的 TODO（`torchtitan/experiments/graph_trainer/cudagraph.py:410-425`），暗示更细粒度 capture 仍在演进；当前实现仍是整张 graph wrapper，不能按未来 piecewise 机制描述。

---

## 5. SimpleFSDP、TP、EP：并行优化为何必须先进入 IR

### ① 背景 / 问题

Joint graph 只有在 collective 真正成为 nodes 时才有调度价值。若 FSDP 仍藏在 hooks、TP 仍只由局部 compile 控制、EP microbatch 仍在 Python 循环里，GraphTrainer 只能看见算子岛，无法跨通信/计算移动依赖。

### ② 为什么先 parallelize，再捕获整步

选中路线是模型先应用 TP/EP 与 SimpleFSDP，使 DTensor redistribution/collective 出现在实际 forward/backward，再由 Trainer 捕获。替代方案是 trace 原始 dense model 后让 pass 自动推导所有并行布局；当前 manual path 没有这样做，AutoParallel 是另一个明确配置的实验分支（`torchtitan/experiments/graph_trainer/configs.py:177-189`）。

判据是 trace 必须看到与真实参数 mesh/layout 一致的 collective，而不是在 dense IR 上猜测通信。

### ③ 当前接线与三类 pass

Llama 的 manual 路径是 FQN annotation → TP → 无条件 SimpleFSDP → compile（`torchtitan/experiments/graph_trainer/llama3/parallelize.py:53-70`）；DeepSeek V3 是 TP/EP → SimpleFSDP → 可选 eager EP chunking → compile（`torchtitan/experiments/graph_trainer/deepseek_v3/parallelize.py:59-77`）。

- **SimpleFSDP**：参数 property 用 traceable DTensor redistribution 表达 unshard，backward placement 表达 gradient reduction；joint passes 才能去重、bucket 和移动这些 nodes（`torchtitan/experiments/graph_trainer/simple_fsdp.py:167-231`）。
- **Async TP**：`apply_compile()` 先在 backend-aware dense TP mesh 上启用 symmetric-memory 支撑，随后 compile-time list 在 FSDP/EP schedule 后追加 `async_tensor_parallel_pass`（`torchtitan/experiments/graph_trainer/compile.py:74-80`；`torchtitan/experiments/graph_trainer/passes.py:337-338`）。
- **EP overlap**：graph strategy先标 chunk dimension、按 static-input boundary 拆图，再 isolate EP PG、做 FSDP bucket 与 EP schedule；eager strategy只填已有 chunk metadata（`torchtitan/experiments/graph_trainer/passes.py:210-300`）。

E-FSDP 开启时，默认 bucket plan 还会把 MoE dense/shared-expert 区域与 routed experts 分开，以匹配不同 expert storage group（`torchtitan/experiments/graph_trainer/passes.py:176-199`；`torchtitan/experiments/graph_trainer/tests/test_passes.py:3603-3621`）。

### ④ 约束、代价与失败边界

Graph EP chunking 在 TP degree 大于 1 时直接抛错：DTensor lowering 后是 physical TP-local tensors，源码说尚未证明再切 chunk 等价；调用方必须用 TP=1 或 eager chunking（`torchtitan/experiments/graph_trainer/passes.py:219-230`）。

Dense-region FSDP scheduler 与 EP overlap 只在 graph chunking 根为 `layers.*.moe` 时组合，否则 warning 并禁用该 scheduler（`torchtitan/experiments/graph_trainer/passes.py:302-335`）。

Async TP 虽有 live pass，但当前 H100 integration case因 fused collective-matmul stride/meta mismatch 被禁用（`torchtitan/experiments/graph_trainer/tests/integration_tests.py:689-719`）。API 存在不能代替端到端负证据。

### ⑤ 发展趋势

**有锚点的推断**：graph EP+TP guard 与 Async-TP disabled TODO 都给出明确待解边界，但没有承诺统一解决方案或时间（`torchtitan/experiments/graph_trainer/passes.py:223-229`；`torchtitan/experiments/graph_trainer/tests/integration_tests.py:712-718`）。

---

## 6. AutoParallel：solver 选择 placement，GraphTrainer 仍拥有 train-step 图

### ① 背景 / 问题

**背景推断**：Manual TP/FSDP/EP 要由模型作者逐模块指定布局；模型、mesh 或算子改变时，recipe 也要同步维护。AutoParallel 试图把这部分改成“模型图 + mesh + 输入/输出约束 → solver placement”，但 GraphTrainer 仍需要捕获 loss/backward 并运行自己的 pass pipeline。提交 `0fae1675e855f384ce644563d16f3f0c9a316218` 的 commit body 因而把它定义为 GraphTrainer `aot_fx_trace` 的一等集成，而不是独立 trainer：AutoParallel 解 SPMD placement并返回 AOT-backed module，GraphTrainer 再 trace 完整 train step。

### ② 为什么选择 solver module 边界，而不是让 GraphTrainer pass 自己猜布局

选中的路线把两个决策分层：AutoParallel 根据模型级 constraints 求参数、输入与输出 placement；GraphTrainer 消费已并行化 module，继续负责 loss/backward、memory/communication passes 和 terminal compiler。明显替代是沿用 §5 的手写模型 recipe，或让 GraphTrainer 在 joint graph 上同时搜索所有布局。当前仓库只把 `enable_autoparallel` 描述为 ILP solver-based SPMD sharding，并强制它搭配 `aot_fx_trace`；没有暴露 solver objective 或证明全局最优性的源码，因此后两点不能从 “AutoParallel” 名称推断（`torchtitan/experiments/graph_trainer/configs.py:177-189`）。

提交 `0fae1675e855f384ce644563d16f3f0c9a316218` 给出的决策准则是复用 GraphTrainer 正常 train-step pipeline，同时让 placement search 成为可替换的 parallelization mode。**推断**：把 AOT module 当边界而非把 solver 嵌入 GraphTrainer passes，也隔离了布局搜索与后续图调度；源码未声称这会降低总编译时间或一定优于 manual recipe。

### ③ 实现思路 / 状态 / 调用链

模型 registry 只在 `compile_config.enable_autoparallel` 为 true 时 lazy-import AutoParallel 分支，否则仍走 manual parallelize；Llama 与 DeepSeek 各自执行这一 dispatch（`torchtitan/experiments/graph_trainer/llama3/__init__.py:20-27`；`torchtitan/experiments/graph_trainer/deepseek_v3/__init__.py:20-27`）。两条 solver 路径的约束不同：

| 路径 | solver mesh 与约束 | AOT 输出边界 |
|---|---|---|
| Llama dense | 从可用的 `dp_replicate/fsdp/tp` axes 建 dense mesh；当前先拒绝 DDP/CP/PP。token 与 position 在 FSDP axis `Shard(0)`、TP axis `Replicate()`；输出在 FSDP axis `Shard(0)`、TP vocab axis `Shard(1)`（`torchtitan/experiments/graph_trainer/llama3/parallelize_autoparallel.py:53-68`；`torchtitan/experiments/graph_trainer/llama3/parallelize_autoparallel.py:105-138`） | TP 开启时附加 `AutoParallelModelOutput(tp_mesh, Shard(1))`，把 local logits 重包成 vocab-sharded DTensor，供 `loss_parallel()` 继续识别布局（`torchtitan/experiments/graph_trainer/llama3/parallelize_autoparallel.py:145-158`；`torchtitan/experiments/graph_trainer/autoparallel_api.py:85-90`） |
| DeepSeek sparse | 强制且仅取二维 `("efsdp", "ep")` sparse mesh，拒绝 DDP/CP/PP/TP；输入和输出都在两轴 `Shard(0)`，并以 `dynamic=True` 建 solver（`torchtitan/experiments/graph_trainer/deepseek_v3/parallelize_autoparallel.py:106-134`；`torchtitan/experiments/graph_trainer/deepseek_v3/parallelize_autoparallel.py:163-197`） | EFSDP/EP 都是 data-parallel factors，local logits 与 local labels 直接进入 loss，不做 DTensor output rewrap（`torchtitan/experiments/graph_trainer/deepseek_v3/parallelize_autoparallel.py:201-209`） |

两条路径都先 `add_*_constraints`，再调用 `optimize_placement()`，最后把 placement 交给 `apply_placement_for_fx_module`（`torchtitan/experiments/graph_trainer/llama3/parallelize_autoparallel.py:129-158`；`torchtitan/experiments/graph_trainer/deepseek_v3/parallelize_autoparallel.py:180-209`）。后者先 materialize sharded parameter/buffer dict，再以同一个 compiler function AOT compile joint forward/backward descriptors（`torchtitan/experiments/graph_trainer/autoparallel_api.py:75-101`）。

Runtime forward 按 graph 记录的 FQN 重取 parameter/buffer；若它们是 DTensor，就把 `.to_local()` payload 与 flattened user inputs 组成 boxed args，调用 AOT function（`torchtitan/experiments/graph_trainer/autoparallel_api.py:30-44`；`torchtitan/experiments/graph_trainer/autoparallel_api.py:103-130`）。**推断**：这条 local-tensor AOT boundary 避免让 solver 已决定的 DTensor wrapper 再参与 AOT graph 输入语义；源码明确的事实仅是它确实 localize state。Llama 的 local vocab logits 随后按 TP mesh 大小恢复 global shape，并以 `run_check=False`、显式 contiguous stride 构造 DTensor；DeepSeek 不提供 adapter 时原样返回 plain tensor（`torchtitan/experiments/graph_trainer/autoparallel_api.py:56-72`）。

### ④ 约束 / 成本 / 失败边界

API 存在不等于成熟可用。DeepSeek 当前不直接并行化 TorchTitan model：其 token dispatcher 含 solver 尚不支持的 `aten::div.Tensor_mode`，所以另建 meta device 上的 `autoparallel._testing.models.dsv3` 模型并补回 TorchTitan MoE attributes；import 失败消息明确要求在把依赖移到 supported namespace 前不要视为稳定 production dependency（`torchtitan/experiments/graph_trainer/deepseek_v3/parallelize_autoparallel.py:41-88`；`torchtitan/experiments/graph_trainer/deepseek_v3/parallelize_autoparallel.py:147-161`、`211-220`）。

更强的负证据来自 integration matrix：Llama FSDP2+TP2 因 AutoParallel/PyTorch 单维 strategy API skew 被 `disabled=True`；注释还记录 FlexAttention BlockMask 无法穿过 graph capture，所以 case 已降到 SDPA。DeepSeek EFSDP4+EP2 则因 meta trace 与 CUDA runtime 的 FakeTensor device mismatch 被 `disabled=True`（`torchtitan/experiments/graph_trainer/tests/integration_tests.py:723-756`；`torchtitan/experiments/graph_trainer/tests/integration_tests.py:760-781`）。因此 baseline 中这两个 AutoParallel integration definitions **全部 disabled**；这证明“当前无 enabled 端到端保障”，不证明算法概念不可行。

历史也显示边界持续受上游 churn 影响：可达提交 `f59f47ec494659b1179b021b62e3b5f39886fb45` 曾因 private expected-input helper 被移除而改成从 graph placeholder 计数；当前实现保留了这一兼容方案（`torchtitan/experiments/graph_trainer/autoparallel_api.py:103-115`）。可达提交 `f7a8e22e39db41f36b5668a36263b1d113c57a95` 又因新的 strategy API skew 禁用 Llama case。README 或 CLI flag 都不能覆盖这些当前负证据。

### ⑤ 发展趋势

**有锚点的推断**：DeepSeek import guard 指向把 DSv3 model/annotation 移出 `_testing`，两个 integration TODO 分别等待 strategy API migration 与 FakeTensor device regression 修复（`torchtitan/experiments/graph_trainer/deepseek_v3/parallelize_autoparallel.py:48-54`；`torchtitan/experiments/graph_trainer/tests/integration_tests.py:735-740`、`763-780`）。这些是明确修复压力，不是成熟度或重新启用时间表。

---

## 7. Precompile artifact：移动编译成本，不移动运行时真相

### ① 背景 / 问题

每个 rank 都重复 trace、Inductor compile，会放大启动成本；但直接序列化某个 rank 的 graph 又可能烘入 rank coordinate、ProcessGroup 名称、SymInt 具体值和 CUDA Graph 内存地址。

### ② 为什么选择 CooR + fake PG + GraphPickler

选中路线是在单 GPU 上用 compile-on-one-rank 和 fake process group 生成 rank-agnostic graph，把 DeviceMesh 留作 runtime graph input，再让所有训练 ranks 加载同一 artifact。明显替代是每 rank 独立编译，或序列化 rank-specific callable；前者浪费编译时间，后者不能安全跨 rank。

提交 `fbdc4c94fe967c8bfce2fe55cf63f8e89087bb46` 的 commit body 将 CooR 方案概括为“单 GPU trace distributed ops、artifact 由所有 torchrun ranks 加载”；提交 `1c4e18b92ef3473586fa8a28dcb61e13bdb6b904` 又解释 `aot_fx_trace` 必须用 GraphPickler 保留 SymInt，避免 plain pickle 把 embedding vocab offset 固化为 trace-rank 常量。

### ③ 生成、序列化与加载链

Precompile 入口要求显式 `dp_shard`，因为单进程无法从实际 world size 推断 `-1`；它按所有并行度计算虚拟 world，创建 rank 0 fake PG，并在 parallelize 阶段全局启用 CooR（`torchtitan/experiments/graph_trainer/precompile_main.py:47-111`）。模型仍走正常 model-specific `parallelize_fn`，但 init weights 期间临时关闭 CooR，因为 DTensor RNG 尚不支持该模式（`torchtitan/experiments/graph_trainer/precompile_main.py:120-159`）。

随后它构造 dummy inputs、调用同一个 `make_fwd_bwd_step` 与 `minimal_fx_tracer`，应用 compile-time passes；CUDA Graph 被排除，因为必须在各 runtime rank 的内存/stream 环境中 capture（`torchtitan/experiments/graph_trainer/precompile_main.py:187-208`；`torchtitan/experiments/graph_trainer/precompile_main.py:284-324`）。

Artifact 保存 serialized GraphModule、state FQNs、subclass layouts、output spec 与 tensor-input indices；compiled Triton artifacts已烘进 graph（`torchtitan/experiments/graph_trainer/precompile.py:149-174`）。GraphPickler 保存 SymInt 与受过滤 metadata；`user_inputs_spec` 刻意不序列化，因为 FlexAttention BlockMask context 含不可 pickle 的 `_MaskModWrapper`（`torchtitan/experiments/graph_trainer/precompile.py:176-214`）。

训练首步按 model/config/dims 计算 fingerprint，检查固定 key 后加载 `TracedResult`（`torchtitan/experiments/graph_trainer/trainer.py:178-204`）。DeviceMesh 是 runtime placeholder，因此加载不需要重写 PG 名（`torchtitan/experiments/graph_trainer/precompile.py:281-308`）；默认 pipeline 检测 artifact 后只补 CUDA Graph pass（`torchtitan/experiments/graph_trainer/passes.py:410-448`）。

### ④ 约束、代价与失败边界

Fingerprint docstring 声称捕获所有影响 compiled output 的因素，但实现只 hash model shape/dtype、ParallelDims、部分 compile/EP fields、PyTorch version 与 CUDA capability（`torchtitan/experiments/graph_trainer/precompile.py:33-85`）。当前 config 中的 `enable_passes`、`disable_passes`、`inductor_compilation`、CPU-offload knobs 和 FSDP overlap flags并未全部进入 hash（`torchtitan/experiments/graph_trainer/configs.py:82-169`）。这是当前 source 内部的覆盖缺口，不能把 fingerprint 描述成完整缓存 key。

> [!contradiction] Fingerprint 注释强于当前实现
> “captures everything” 是函数说明；实际字段枚举没有覆盖全部 pass-shaping knobs。本文按实现收窄 stale-artifact 保证。

Mismatch 默认抛错，但环境变量可显式绕过；legacy artifact 没有 fingerprint 时只 warning 并继续（`torchtitan/experiments/graph_trainer/precompile.py:104-143`）。绕过意味着调用者自行承担 stale artifact 风险。

当前还有一个更直接的 trace-contract 冲突：core Trainer 把 `global_valid_tokens` 保持为 device scalar tensor（`torchtitan/trainer.py:787-806`），而 precompile 入口在提交 `73aed7f6c09c04041dae2d2c185bb4c6384ebb3f` 后又构造 Python float，并保留“runtime 也是 float”的过时注释（`torchtitan/experiments/graph_trainer/precompile_main.py:210-225`）。本页不推断具体故障形态，但这两个当前调用点已经不再表达同一种输入契约。

CooR precompile 还明确拒绝 CP（`torchtitan/experiments/graph_trainer/precompile_main.py:243-250`）。测试 runner 的 Llama FSDP+TP definition启用，但 DeepSeek FSDP+TP+EP precompile 因 data-dependent unbacked SymInt 被禁用（`torchtitan/experiments/graph_trainer/tests/run_precompile_tests.py:35-88`）。

### ⑤ 发展趋势

**有锚点的推断**：precompile/training 共享 build/parallelize/init 的 TODO 与 CP TODO 都指向减少两条 setup 路径漂移（`torchtitan/experiments/graph_trainer/precompile_main.py:120-121`；`torchtitan/experiments/graph_trainer/precompile_main.py:243-250`）。当前 token-count 类型冲突正是这类漂移的实例，但源码没有承诺修复版本。

---

## 8. Ordinary PP 与 GraphPP：同一 Trainer，两个执行所有者

### ① 背景 / 问题

Pipeline schedule 需要按 microbatch/stage action 交错 forward、backward、P2P、unshard 和 reduce。把整个多 stage 作业捕获为一张普通 non-PP joint graph，会丢失 schedule runtime 的 action 边界；完全重写 PP 则重复 TorchTitan 已有 stage split 和 PyTorch schedule。

### ② 为什么复用 PP schedule、只替换 stage compute

GraphPP 选择让 upstream PP 继续拥有 microbatch split、action order、P2P 与 stage metadata，GraphTrainer 只为每个本地 stage 建立 callable bundle并注册 action handlers（`torchtitan/experiments/graph_trainer/graph_pp/runner.py:306-327`）。明显替代是 graph compiler 自己实现完整 PP scheduler；当前设计以复用调度正确性和限制改动面为判据。

提交 `21e7954944ea84484b7804a2119ab7b42e699478` 的 commit body 直接说明：GraphPipelineRuntime 执行 upstream runtime schedule actions，而 graph construction 使用 upstream metadata、GraphTrainer trace 与 pre-partition passes。

### ③ Stage 构建、partition 与 runtime action

GraphTrainer model registry把 `pipelining_fn` 指向 `graph_pipeline_llm`（`torchtitan/experiments/graph_trainer/llama3/__init__.py:30-48`；`torchtitan/experiments/graph_trainer/deepseek_v3/__init__.py:30-52`）。Core Trainer 在 PP enabled 时调用该函数取得 schedule 与 local model parts（`torchtitan/trainer.py:426-456`）。

`graph_pipeline_llm` 复用通用 metadata、FQN stage split、rank-to-stage mapping 与 schedule builder；每个 local part 仍先应用 model-specific TP/EP/SimpleFSDP，再包装 `GraphPipelineStage`，最后注册 graph runtime（`torchtitan/experiments/graph_trainer/graph_pp/pipeline.py:98-180`）。

每个 stage 用代表性输入 trace：last stage 对 scalar loss 求 param/input grads；non-last stage对输出与 downstream-provided output grads 求 param/input grads（`torchtitan/experiments/graph_trainer/graph_pp/graph_builder.py:909-1015`）。Pass 后 partition 为 forward/backward，并继续抽取可选 FSDP UNSHARD/REDUCE_GRAD 与 dI/dW pieces（`torchtitan/experiments/graph_trainer/graph_pp/graph_builder.py:1026-1085`）。

Runtime 为 `FORWARD/FULL_BACKWARD/UNSHARD/RESHARD/REDUCE_GRAD/BACKWARD_INPUT/BACKWARD_WEIGHT/OVERLAP_F_B` 注册 handlers（`torchtitan/experiments/graph_trainer/graph_pp/runner.py:709-743`）。例如 forward 确保参数已 unshard，再执行 stage graph；reduce-grad 执行专用 graph并按 microbatch 数缩放；RESHARD action只清空缓存的 unsharded parameter values（`torchtitan/experiments/graph_trainer/graph_pp/runner.py:436-465`；`torchtitan/experiments/graph_trainer/graph_pp/runner.py:563-582`）。

### ④ 约束、代价与失败边界

GraphPP 要求 `aot_fx_trace`、拒绝 precompile artifact、拒绝 `fsdp_reshard_after_forward="always"`，并只接受 PyTorch runtime schedule subclasses（`torchtitan/experiments/graph_trainer/graph_pp/pipeline.py:40-63`）。

即使 `enable_passes=False`，GraphPP 仍无条件执行 DCE、canonicalize 与 SimpleFSDP unshard dedup，因为 partition 假设 canonical graph；只有后续可选 GraphTrainer passes 才受 gate 控制（`torchtitan/experiments/graph_trainer/graph_pp/graph_builder.py:779-832`）。

GraphPP 当前明确不 capture CUDA Graph：provider 会 warning，要求等待独立 runtime integration（`torchtitan/experiments/graph_trainer/graph_pp/graph_builder.py:1281-1295`）。因此 ordinary GraphTrainer 的 “compile-time artifact + runtime CUDA Graph” 不能套用到 GraphPP。

`GraphTrainer.forward_backward_step` 遇到 PP 虽回退父 Trainer，但这不表示 GraphPP 回到 eager stage compute：真正执行所有者已经是 model spec 返回的 `GraphPipelineRuntime`（`torchtitan/experiments/graph_trainer/trainer.py:136-148`；`torchtitan/experiments/graph_trainer/graph_pp/pipeline.py:160-180`）。

### ⑤ 发展趋势

**有锚点的推断**：GraphPP provider 的 CUDA Graph warning 与 pipeline 的 precompile “not yet” guard 给出明确缺口；源码未承诺具体方案。当前不应把 README 的未来 composability 写成 live 能力。

---

## 9. Integration 证据：enabled 与 disabled 同样重要

### ① 背景 / 问题

实验目录很容易用 README feature list制造“FSDP+TP+EP+PP+CP+CUDA Graph 全组合”的印象；实际组合能力由 config guards、enabled tests、disabled TODO 与模型后端共同决定。

### ② 为什么以负证据校准 feature list

选中判据是：只有当前调用链可达且 integration definition 未禁用，才写成已覆盖组合；代码 branch 只能证明实现意图，README checkmark 只能证明文档主张。明显替代是只列成功项，它会隐藏最可能踩中的 hang、cyclic graph 与 backend conflict。

### ③ 当前测试矩阵能支持的结论

| 组合 | 当前矩阵状态 | 证据 |
|---|---|---|
| Llama AOT FX FSDP4+TP2 | enabled | `torchtitan/experiments/graph_trainer/tests/integration_tests.py:219-232` |
| Llama FSDP4+TP2 + SAC/offload | enabled | `torchtitan/experiments/graph_trainer/tests/integration_tests.py:233-248` |
| Llama/Qwen FSDP+TP+CP | disabled | GraphTrainer 未迁 `spmd_types`，且 Flex CP regional-Inductor 另有错误（`torchtitan/experiments/graph_trainer/tests/integration_tests.py:195-217`; `torchtitan/experiments/graph_trainer/tests/integration_tests.py:603-620`） |
| DeepSeek FSDP+TP+EP | disabled variants | flaky/hanging all-to-all 或 cyclic FSDP bucket region（`torchtitan/experiments/graph_trainer/tests/integration_tests.py:426-467`） |
| DeepSeek GraphPP FSDP4+EP2 | enabled for Interleaved1F1B/ZBV/DualPipeV | `torchtitan/experiments/graph_trainer/tests/integration_tests.py:504-563` |
| DeepSeek MinimalAsyncEP | enabled | `torchtitan/experiments/graph_trainer/tests/integration_tests.py:580-596` |
| Qwen3 MoE FSDP4+TP2+EP4 | enabled，显式关闭 CUDA Graph | `torchtitan/experiments/graph_trainer/tests/integration_tests.py:622-637` |
| Async TP | disabled | Inductor stride/meta mismatch（`torchtitan/experiments/graph_trainer/tests/integration_tests.py:689-719`） |
| AutoParallel Llama/DeepSeek | disabled | PyTorch API skew 或 FakeTensor device mismatch（`torchtitan/experiments/graph_trainer/tests/integration_tests.py:723-781`） |

CP 冲突还有配置层证据：`to_graph_trainer_config()` 强制 `spmd_backend="partial_dtensor"`，注释说明该后端不应用 `ShardingConfig` 的 CP placements（`torchtitan/experiments/graph_trainer/configs.py:239-258`）；integration 顶部因此全局设置 `_CP_DISABLED=True`（`torchtitan/experiments/graph_trainer/tests/integration_tests.py:22-28`）。全局 TorchTitan 默认 `spmd_types` 不会覆盖这个转换。

### ④ 约束、代价与失败边界

Enabled integration definition说明该组合进入当前测试列表，不等于本页执行过 GPU 测试，也不提供吞吐/峰值显存结论。Disabled 项则是明确负证据：在当前 baseline 上不能作为受保障路径。

README 仍宣称 full step 可选 optimizer、precompile fingerprint 检测 stale artifact、FSDP+TP+EP composability（`torchtitan/experiments/graph_trainer/README.md:7-14`）；本页分别用生产 call site、fingerprint field coverage 和 integration matrix对这些主张做了降级。README 是设计入口，不是比可执行源码更高的事实层。

### ⑤ 发展趋势

**有锚点的推断**：integration TODO 指向 CP backend migration、EP all-to-all/FSDP cyclic region、Async-TP meta kernel 与 AutoParallel API 对齐；它们构成当前最清晰的修复压力，但不应转写成 release roadmap（`torchtitan/experiments/graph_trainer/tests/integration_tests.py:407-503`；`torchtitan/experiments/graph_trainer/tests/integration_tests.py:712-780`）。

---

## 10. 结论：应替换的旧心智模型

### ① 背景 / 问题

旧页容易把 Manifesto 愿景、generic tracer 能力、README 旧模式和当前 production path 合并成一个“整步全部进图、任意组合可编译”的 GraphTrainer。

### ② 为什么采用 live-vs-legacy 结论

本页以当前 call site 与 enabled/disabled tests 为现状，以 reachable commit body 解释选择原因。明显替代是用旧 AOT、generic optimizer tracer 或 future GraphPP RFC 补齐当前缺口；这种写法无法在 frozen baseline 复核。

### ③ 当前可用结论

1. Active 主线是 non-PP `aot_fx_trace` joint fwd/loss/bwd；旧 AOT 已删除，JIT deprecated 且 integration disabled（`torchtitan/experiments/graph_trainer/configs.py:65-82`；`torchtitan/experiments/graph_trainer/tests/integration_tests.py:16-20`）。
2. Module state进入 graph，optimizer 虽被 generic tracer API 支持，却没有接入生产 GraphTrainer；clip、finite gate、optimizer/scheduler 都在 core Trainer 图外（`torchtitan/experiments/graph_trainer/trainer.py:220-262`；`torchtitan/trainer.py:850-889`）。
3. Ordered pass pipeline 的核心价值是让 memory/offload/remat、FSDP/TP/EP 与 terminal compiler共享依赖图；CUDA Graph 是最后的 runtime wrapper（`torchtitan/experiments/graph_trainer/passes.py:201-448`）。
4. Precompile 用 CooR/GraphPickler 移走 compile cost，但 fingerprint 覆盖不完整，CP、token-count trace contract 与部分 MoE SymInt 仍是边界（`torchtitan/experiments/graph_trainer/precompile.py:33-85`；`torchtitan/experiments/graph_trainer/precompile_main.py:210-250`）。
5. Ordinary PP 的 Trainer method 回退不等于 GraphPP eager 执行；GraphPP 由独立 stage graph provider/runtime 接管，并明确不支持 precompile、reshard always 与 CUDA Graph（`torchtitan/experiments/graph_trainer/graph_pp/pipeline.py:40-63`；`torchtitan/experiments/graph_trainer/graph_pp/graph_builder.py:1281-1295`）。

### ④ 约束、代价与失败边界

不要再写：旧 `aot` 仍是当前 mode；optimizer step 已进入 production joint graph；`enable_passes=False` 仍会自动 capture CUDA Graph；precompile fingerprint 完整覆盖所有 compiler knobs；GraphTrainer CP 可用；ordinary PP 与 GraphPP 是同一 forward-backward call path。这六项都被当前实现或 tests 否定。

### ⑤ 发展趋势

**有锚点的推断**：当前最明确的演进方向是 mode 收敛、GraphTrainer 迁移 `spmd_types`、GraphPP runtime CUDA Graph/precompile、以及减少 precompile/setup 漂移。它们都有 warning/TODO/guard 锚点，但源码没有证据支持何时成为 core Trainer 默认。

---

## Related Pages

- [[02_engineering/02_train_frameworks/torchtitan/01_torchtitan_trainer_quickstart|TorchTitan Trainer Quick Start]] —— 对照 core Trainer 的数据、optimizer、checkpoint 与训练循环所有权。
- [[02_engineering/02_train_frameworks/torchtitan/14_torchtitan_pp_analysis|TorchTitan PP 分析]] —— GraphPP 复用的 stage split、microbatch 与 PyTorch schedule 基座。
- [[02_engineering/02_train_frameworks/torchtitan/15_torchtitan_ep_analysis|TorchTitan EP 分析]] —— dispatcher、E-FSDP 与 GraphTrainer EP chunk/overlap 的边界。
- [[02_engineering/02_train_frameworks/torchtitan/16_torchtitan_spmd_types_analysis|TorchTitan SPMD Types 分析]] —— 解释 partial-DTensor/`spmd_types` 双后端及 CP 冲突。
- [[02_engineering/02_train_frameworks/torchtitan/22_torchtitan_ac_analysis|TorchTitan Activation Checkpoint 分析]] —— 对照 eager AC 与 graph tensor-granularity memory policy。
- [[02_engineering/02_train_frameworks/torchtitan/25_torchtitan_simple_fsdp_analysis|TorchTitan SimpleFSDP 分析]] —— collective 如何进入 joint graph，并被 bucket/reorder。
- [[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]] —— 本系列入口与功能树定位。
