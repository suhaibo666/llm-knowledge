# GraphTrainer：把训练控制面下沉为可变换的 FX 图

> **代码基准**：torchtitan `main` @ `a3168782c9a3a2e40afbd0de114818b96e2bda6e`
> **最后更新**：2026-08-27 · **状态**：`torchtitan/experiments/graph_trainer` 实验路径
>
> **中心结论**：核心 Trainer 把并行、autograd、compile 和 CUDA Graph 作为彼此协作的运行时系统；GraphTrainer 则先把 forward + loss + backward 捕获成一张显式 FX 图，再在图上安排内存、FSDP/TP/EP 通信、Inductor 区域与 CUDA Graph。它的优势是全局可见性和 pass 可组合性，代价是需要更严格的静态契约，且 GraphPP、CP、precompile 等组合仍有明确实验边界。

---

## 1. 为什么另起一条 GraphTrainer 路线

GraphTrainer 的动机不是“把 `torch.compile(model)` 再包一层”。它认为 eager 训练的多个系统各有拦截点：FSDP hooks、autograd、activation checkpoint、compile 和 CUDA Graph 的边界互相影响；只有把完整训练计算放进同一 IR，编译器才能看见通信、重计算、offload 与 kernel 之间的统一依赖。Manifesto 将其定义为“捕获 forward、loss、backward，以及可选 optimizer step 后再变换的一张 FX 图”（`torchtitan/experiments/graph_trainer/MANIFESTO.md:3-13`）。

当前**生产 GraphTrainer 接线**需要进一步限定：`make_fwd_bwd_step()` 返回 `[loss]+grads`，使用 `torch.autograd.grad` 显式产出参数梯度；`optimizer.step()` 仍由父 Trainer 在图外执行（`torchtitan/experiments/graph_trainer/trainer.py:75-103,251-262`,`torchtitan/trainer.py:850-890`）。底层 `minimal_fx_tracer` 已有 optimizer state swap 能力，但源码仍标注 GraphTrainer 尚未追踪 optimizer state（`torchtitan/experiments/graph_trainer/make_fx_tracer.py:320-321,337-364`）。因此“可选 optimizer 进图”是 tracer 能力/方向，不应写成当前默认训练链已启用。

两条路线的控制边界可以概括为：

| 路线 | 编译器看到什么 | FSDP/AC/通信由谁排程 | optimizer |
|---|---|---|---|
| 核心 Trainer | 逐 block 或 forward/backward body | eager FSDP hooks、模块 wrapper、autograd engine | 图外 |
| GraphTrainer `aot_fx_trace` | forward + loss + backward + 显式 param grads | FX passes 看到并移动图内 collective/重计算 | 当前仍图外 |

## 2. 一次非 PP 训练步怎样变成图

`GraphTrainer` 继承核心 `Trainer`，仍复用数据、checkpoint、optimizer 和日志生命周期；只在无 PP 且 `compile.mode="aot_fx_trace"` 时接管 forward/backward（`torchtitan/experiments/graph_trainer/trainer.py:106-148`）。

第一次真实输入到来时，调用链是：

```text
GraphTrainer.forward_backward_step
  → model.preprocess_inputs
  → make_fwd_bwd_step(model,loss_fn)
  → minimal_fx_tracer(...,module=model)
       捕获 model forward → loss → autograd.grad
  → 构造并执行 graph pass pipeline
  → 缓存 TracedResult

后续 step
  → run_traced(cached_graph, live module state)
  → 得到 loss 与显式 grads
  → accumulate_param_grads_ 写回 live param.grad
  → 父 Trainer 做 clip / optimizer / scheduler
```

预处理仍在 trace 外先执行，因此模型的 mask、输入结构和并行预处理契约没有被 GraphTrainer 绕过（`torchtitan/experiments/graph_trainer/trainer.py:150-176`）。首次 trace 在 `train_context()` 内进行，参数/缓冲被 tracer 作为静态图输入穿过；得到的图随后经过所选 pass pipeline（`torchtitan/experiments/graph_trainer/trainer.py:206-250`）。执行也重新进入 `train_context()`,输出梯度显式累积到 live 参数（`torchtitan/experiments/graph_trainer/trainer.py:251-262`）。

这种显式梯度出口很重要：如果只依赖 traced graph 内的 `.grad` side effect，FX 执行后无法保证它还写回 Trainer 正在持有的参数。Chunked loss 也因此有专用 wrapper，把 lm-head param grads 作为 autograd 输出而不是隐式副作用（`torchtitan/experiments/graph_trainer/configs.py:278-290`,`torchtitan/experiments/graph_trainer/chunked_loss.py:15-62`）。

## 3. `aot_fx_trace` 是默认，JIT/AOT 旧入口不再是主线

`GraphTrainerCompileConfig.mode` 默认 `aot_fx_trace`;配置仍保留 `jit`，但 README 把 JIT 和旧 AOT 标成待删除的 legacy modes（`torchtitan/experiments/graph_trainer/configs.py:66-82`,`torchtitan/experiments/graph_trainer/README.md:16-22`）。当前 integration test 还因上游 partitioner 回归整体禁用 JIT（`torchtitan/experiments/graph_trainer/tests/integration_tests.py:16-20`）。

这三者不能混写：

- `jit` 是模型级 `torch.compile()` 和 custom backend 的旧路径。
- 旧 `aot` 曾显式导出 joint graph，现已从当前配置 literal 移除。
- `aot_fx_trace` 使用 non-strict `minimal_fx_tracer` 直接得到训练 joint graph，不依赖 AOTAutograd 再做 forward/backward partition。

`apply_compile()` 在 `aot_fx_trace` 下刻意不包 model；真正捕获延迟到第一次训练输入，precompile artifact 也在同一位置懒加载（`torchtitan/experiments/graph_trainer/compile.py:56-122`）。这样 trace 能以真实 batch 结构和 Trainer 的 loss 语义为入口，而不是只看孤立的 model forward。

## 4. Pass pipeline：先规范化，再做内存/通信，最后选执行后端

当前默认 pipeline 不是一串无序开关。`compile_time_passes()` 给出稳定的相序（`torchtitan/experiments/graph_trainer/passes.py:140-170,201-349`）：

1. **规范化**：dead-code elimination、canonicalize、去重 FSDP unshard chain。
2. **内存策略**：逐 tensor 标注保存/重算/offload，应用 CPU offload，再做 selective activation remat。
3. **EP/FSDP/TP 调度**：可选 EP graph chunking、独立 EP process group、FSDP bucket/reorder、EP overlap schedule、dense-region FSDP 调度与 Async TP。
4. **终结编译**：regional 或 full Inductor；之后 FX 图不再是权威 IR。
5. **运行时捕获**：可选 CUDA Graph，必须在需要内存中 recapture 的运行 rank 执行。

默认 FSDP bucket 计划按 TransformerBlock 和 MoE 结构构造；若启用 overlap，会在 bucketing 前给 collective 换独立 process group/stream，使 bucket 继承新的通信域（`torchtitan/experiments/graph_trainer/passes.py:158-199,274-285`）。Async TP 则作为显式 pass 追加，而非核心 Trainer 的 compile 配置副作用（`torchtitan/experiments/graph_trainer/passes.py:337-348`）。

终结阶段支持两种粒度：

| 模式 | 行为 | 边界 |
|---|---|---|
| `regional`（默认） | 只把标注区域交给 Inductor；FlexAttention 必须区域编译 | 图的其余控制仍保留 FX 可见性 |
| `full` | 整张 train-step 图交给 Inductor | 必须最后执行，之后不能再按原 FX 节点变换 |

两条路径的选择与 FlexAttention/RMSNorm 标注逻辑在 `torchtitan/experiments/graph_trainer/passes.py:352-407`。CUDA Graph pass 在有 precompile artifact 时仍要于训练 rank 追加，因为序列化时无法保存某个进程内的 CUDA Graph capture（`torchtitan/experiments/graph_trainer/passes.py:410-448`）。

## 5. 图内并行：SimpleFSDP、TP 与 EP overlap

### 5.1 SimpleFSDP 是 FSDP 的图表示

GraphTrainer 的 SimpleFSDP 把参数 unshard / grad reduce 作为 collective 节点暴露给 FX，因此后续 pass 能分桶、改 process group、移动通信到相邻 dense 区域，而不依赖 eager hook 的运行时先后。机制细节和与 FSDP2 的差异见 [[25_torchtitan_simple_fsdp_analysis]]；本页只强调它为何是 GraphTrainer pass pipeline 的前提。

### 5.2 Async TP 在 joint graph 上匹配

GraphTrainer 的 Async TP pass 给图中 TP collective/matmul 做 micro-pipeline 变换（`torchtitan/experiments/graph_trainer/passes.py:95-131`）。它与核心 Trainer 的 `torchtitan/distributed/compile.py` 接线不是同一个入口：前者是 joint FX graph pass，后者是在逐 block compile 前打开 Inductor 配置。两者目标相近，控制面与可见图范围不同。

### 5.3 EP overlap 用 chunking 改写依赖图

启用 EP overlap 后，trace-input preparer先给 token-grid 维度写 symbolic-shape 元数据，graph chunk pass 再据此拆分 live-in/live-out；随后 isolate EP process group、重排 FSDP bucket，并把 token exchange 与可重叠计算排进图（`torchtitan/experiments/graph_trainer/README.md:141-182`,`torchtitan/experiments/graph_trainer/passes.py:210-300`）。

当前 graph chunking 明确拒绝 TP>1，因为 DTensor lowering 后的物理 TP-local tensor 和布局 helper 还未证明可以等价拆分；调用者须改用 TP=1 或 eager chunking（`torchtitan/experiments/graph_trainer/passes.py:213-230`）。`enable_fsdp_dense_region_overlap` 与 EP overlap 也只有在特定 MoE-only graph chunk 配置下组合，否则会被忽略并告警（`torchtitan/experiments/graph_trainer/passes.py:302-335`）。

## 6. GraphPP：复用 eager 切 stage，用图替换 stage 内执行

GraphPP 没有重新发明 PP schedule。它先复用 TorchTitan 的 module FQN stage split 与 PyTorch runtime schedules，再为每个本地 stage 用代表性 microbatch 构造 graph bundle；steady state 的 PP action 只调用预建 callable（`torchtitan/experiments/graph_trainer/README.md:63-105`）。

构建阶段：

1. 校验必须是 `aot_fx_trace`、不可加载 precompile artifact、不可 `fsdp_reshard_after_forward="always"`，且 schedule 必须是 runtime schedule（`torchtitan/experiments/graph_trainer/graph_pp/pipeline.py:40-63`）。
2. 根据显式 `module_fqns_per_model_part` 或自动 LLM 层分配生成 stage，按本 PP rank 找出本地 virtual stages（`torchtitan/experiments/graph_trainer/graph_pp/pipeline.py:98-132`）。
3. 每个 model part 先走模型自己的 parallelize，再包装为 `GraphPipelineStage`（`torchtitan/experiments/graph_trainer/graph_pp/pipeline.py:131-158`）。
4. 建立 PyTorch PP schedule，注册 `GraphTrainerStageGraphProvider`，得到 `GraphPipelineRuntime`（`torchtitan/experiments/graph_trainer/graph_pp/pipeline.py:160-180`）。

当前 runtime schedule 覆盖 `Interleaved1F1B`、`ZBVZeroBubble` 与 `DualPipeV`；stage graph bundle可包含 forward、full backward、拆分 dI/dW、FSDP `UNSHARD/REDUCE_GRAD`，以及 `OVERLAP_F_B` 的 multiplexed graph（`torchtitan/experiments/graph_trainer/README.md:87-105`）。

GraphPP 与非 PP GraphTrainer 仍不是完全同一条执行链：`GraphTrainer.forward_backward_step()` 检测到 PP 会回退父 Trainer，真正的图执行由 GraphPipelineRuntime 接管（`torchtitan/experiments/graph_trainer/trainer.py:136-148`）。

## 7. Precompile：把编译成本移到单 rank，但 artifact 是易失品

无 PP 路径可在单 GPU 预编译 joint FX graph并写入 disk storage；训练时先按 model、compile config、parallel dims 计算 fingerprint，再加载已编译 artifact（`torchtitan/experiments/graph_trainer/trainer.py:178-204`）。artifact 与 PyTorch/CUDA、模型和并行配置绑定，任一改变都要重新生成（`torchtitan/experiments/graph_trainer/README.md:224-236`）。

它省的是每个训练 rank 重复编译，不是让 artifact 跨拓扑任意复用。GraphPP 当前直接拒绝 `precompile_artifact_dir`，因为 stage-local trace/graph construction 仍发生在 runtime（`torchtitan/experiments/graph_trainer/graph_pp/pipeline.py:45-50`）。

## 8. 现状边界与证据冲突

| 边界 | 当前可执行结论 | 证据 |
|---|---|---|
| 核心 GraphTrainer optimizer | tracer 支持 optimizer state，但生产 `make_fwd_bwd_step` 只捕获到 grads，optimizer 图外 | `torchtitan/experiments/graph_trainer/trainer.py:75-103,251-262`;`torchtitan/experiments/graph_trainer/make_fx_tracer.py:320-321` |
| JIT | 配置仍在，integration 整体禁用，README 标 legacy | `torchtitan/experiments/graph_trainer/tests/integration_tests.py:16-20`;`torchtitan/experiments/graph_trainer/README.md:16-22` |
| CP | README composability 表写 ✅，但转换强制 `partial_dtensor`，integration 明确 `_CP_DISABLED=True` | `torchtitan/experiments/graph_trainer/configs.py:239-258`;`torchtitan/experiments/graph_trainer/tests/integration_tests.py:22-28`;`torchtitan/experiments/graph_trainer/README.md:294-315` |
| GraphPP + precompile | 当前不支持 | `torchtitan/experiments/graph_trainer/graph_pp/pipeline.py:45-50` |
| GraphPP + FSDP reshard always | 拒绝；GraphPP 假设 default/never 的 ZeRO-2-style 路径 | `torchtitan/experiments/graph_trainer/graph_pp/pipeline.py:52-56` |
| Graph EP chunk + TP>1 | 拒绝，尚未证明 lower 后切分等价 | `torchtitan/experiments/graph_trainer/passes.py:213-230` |
| GraphPP + EP overlap / CUDA Graph | README 标为后续工作，不能当成已组合能力 | `torchtitan/experiments/graph_trainer/README.md:107-109` |

这里应以**可执行源码与测试开关**覆盖 README 状态表：CP 的表格很可能反映目标或某阶段结果，但当前转换函数强制 `partial_dtensor`，而当前 CP 只接受 `spmd_types`;所以本基线不能宣称 GraphTrainer CP 已启用。类似地，README 顶部写“可选 optimizer.step 进图”应解释为 tracer 方向，而不是当前 GraphTrainer production step。

## 9. 何时选它

- 如果目标是稳定训练主线、广泛模型与组合支持，优先核心 [[01_torchtitan_trainer_quickstart|Trainer]]。
- 如果要研究“通信节点怎样与计算重排”“逐 tensor 重算/offload”“整步 CUDA Graph/Inductor”或 GraphPP，GraphTrainer 提供更完整、可检查的 IR。
- 如果依赖 CP、GraphPP precompile 或未经测试的 EP+TP overlap 组合，应先按本页边界审计当前 commit，而不是依据实验目录的总览表直接启用。

## Related Pages

- [[25_torchtitan_simple_fsdp_analysis]] —— SimpleFSDP 如何把分片 collective 表达进图
- [[24_torchtitan_comm_optimizations_overlap_analysis]] —— 核心 Trainer 与 GraphTrainer 的通信重叠对照
- [[14_torchtitan_pp_analysis]] —— eager PP stage/schedule 基座，GraphPP 复用它的切分和调度
- [[22_torchtitan_ac_analysis]] —— eager AC/SAC 与 graph tensor-granularity memory policy 的边界
- [[15_torchtitan_ep_analysis]] —— eager token dispatcher 与 GraphTrainer EP overlap 的作用层次
- [[01_torchtitan_trainer_quickstart]] —— 核心 Trainer 推荐入口和训练生命周期
- [[torchtitan/index]] —— TorchTitan 知识地图与代码演进审计
