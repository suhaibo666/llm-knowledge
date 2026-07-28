# PyTorch Graph Compiler 系列 Part IV 审计

审计日期：2026-07-26  
审计范围：设计规格 §5.5、§6、§9 Batch 4、§12.7–12.9；页面 17–21；原始 Lab `labs/part4_inductor.py`  
补充复核：审计过程中新增的 `labs/part4_artifact_bundle.py`、`labs/series_artifact_bundle.py` 与 `labs/test_series_contract.py`  
权威源码基线：`E:/97-codes/torch_parallel/p`，commit `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`  
运行时环境：Windows 11、Python `3.13.5`、PyTorch `2.9.1+cpu`，runtime git `5811a8d7da873dd699ff6687092c225caffcf1bb`，CUDA unavailable，MSVC `cl` unavailable，Triton unavailable

## 1. 结论

**Part IV 原始交付不通过设计验收。**

页面 17–21 的源码机制说明总体有价值，尤其 Scheduler、memory planning 和 codegen 的主干叙述大多可由 pinned source 支持；但设计规格要求的是“机制说明 + 每页规定实验 + 可检查 artifact + 贯穿模型 + 反向 provenance”。原始 `part4_inductor.py` 只真实验证了 external matmul，并把 pointwise C++ 编译标记为缺少 `cl`；它没有生成 IR、索引、liveness、peak、dependency/fusion group、wrapper、custom lowering 或 provenance artifact。因此：

- §12.7“贯穿模型可生成完整阶段 artifact bundle”在原始交付中失败；
- §12.9“从 Scheduler/kernel 反向定位至 FX 和用户源码”在原始交付中失败；
- §12.8 属于 Part III，本报告不重复判定；
- “本机缺少 `cl`”是**真实 kernel 编译/执行**的有效 blocker，但不是 GraphLowering、Scheduler、IR dump、dependency/fusion、静态 liveness/peak 估算、fallback/custom lowering 或 external wrapper/provenance 的 blocker。页面 19、20 和系列索引把 blocker 扩大到了本可运行的阶段。

审计过程中新增的两个 artifact 脚本明显改善了当前工作树：

- provenance 文件最初缺失；设置 `trace.provenance_tracking_level=1` 后，当前 `PartFourArtifactContractTest` 已通过；
- `series_artifact_bundle.py` 当前也能生成 frontend、AOT 和 backend 文件，contract test 已通过；
- 但 backend source generation 通过伪造 compiler 探测并把 `AsyncCompile.cpp_pybinding` 替换为 no-op kernel 实现，只能归类为 **mock-compile 条件下真实观察到的 source codegen，kernel 未编译、未执行**；
- frontend 的 `UnifiedGraphModel` 与 backend 的独立 `backend_core` 不是同一个逐阶段下沉的 Graph，HOP 又是单独 export 变体；当前 `stage_node_mapping.json` 自己也声明是 semantic stage records，而非跨阶段身份映射；
- liveness、Scheduler dependency/fusion group、split/copy/fusion/liveness/provenance 综合表仍不完整，真实 C++ kernel、性能、Triton autotune 仍未验证。

所以，**新增补丁把 §12.7 和 §12.9 从“缺失”推进到“部分满足”，尚不足以把 Part IV 判为通过。**

## 2. 证据等级

本报告严格区分以下证据：

| 标记 | 含义 |
|---|---|
| `[D]` | 设计规格或页面当前文本 |
| `[S]` | 在 pinned PyTorch source commit 上实际打开并核对的实现 |
| `[R]` | 在上述 Windows CPU PyTorch 2.9.1 环境真实运行并观察到 |
| `[M]` | 在 mock compiler/no-op kernel 条件下真实运行到 source codegen；不等于编译或执行 kernel |
| `[B]` | 环境阻塞，未声称已验证 |

“源码支持某机制”不能替代“Lab 实测某行为”；`[M]` 也不能替代真实 compiler/device lane 的 `[R]`。

## 3. 按严重度列出的发现

### P0

#### P0-1：原始 Part IV Lab 与五页规定实验严重不匹配

`part4_inductor.py:5-39` 只有一个 pointwise+reduction 函数及 `max_fusion_size` 对照，且两条路径都在 C++ 编译前因缺 `cl` 阻塞；`part4_inductor.py:42-55` 只有 external matmul 数值检查。它没有覆盖：

- unsupported op 与 custom lowering；
- elementwise/broadcast/transpose/reduction/matmul 的 IR 与索引核对；
- fusion、save/recompute、view/copy 的生命周期与峰值；
- dependency/fusion groups、关闭 reorder、kernel/performance 对照；
- fusion group 的 kernel、wrapper、autotune choices 与源码映射。

页面 18 完全没有 Lab 小节。页面 17、19、20、21 的 Lab 小节只是复述同一旧脚本的成功或 blocker，不能替代各页规格中的实验。

#### P0-2：`cl` blocker 被错误扩大，导致可运行证据被省略

`cl` 缺失确实会阻止 CPU pointwise 的 generated C++ 编译/加载/执行，审计也复现了 `Compiler: cl is not found`。但在同一环境中，以下阶段均实际可运行：

- `make_fx` 后直接构造并运行 `GraphLowering`；
- 构造 `Scheduler`、检查 reads/writes、last-use、fusion group；
- 生成 `ir_pre_fusion.txt` 和 `ir_post_fusion.txt`；
- 运行静态 peak-memory estimator；
- 观察 view 的 `ReinterpretView` 与 copy 的 `ComputedBuffer`；
- 观察 broadcast index；
- 演示 unsupported custom op 的 missing-lowering、implicit fallback 与 registered lowering；
- 对 external matmul 调用 `GraphLowering.codegen()`，生成 Python wrapper 和 user-source line map。

因此：

- 页面 19 `:197-201` 不发布任何 reuse/peak 数字是过度保守；它应区分 scheduler 静态估算与真实 allocator peak；
- 页面 20 `:190-195` 暗示缺 `cl` 使 dependency/fusion 数量不可得，是阶段边界错误；Scheduler fusion 发生在 codegen/compile 之前；
- 系列索引 `:93-95` 说当前只覆盖到 external Inductor 路径，遗漏了本机可观测的内部 IR/Scheduler 阶段。

#### P0-3：原始交付不满足完整 bundle 与反向 provenance

设计 §6 要求同一个贯穿模型包含 parameter、buffer、view/mutation、dynamic shape、structured output、可微 matmul/pointwise/reduction 和 HOP，并产出 Python source、Dynamo/FX、functional ATen、AOT joint/fw/bw、post-grad FX、IR、Scheduler、kernel/wrapper 及 split/copy/fusion/liveness/provenance 表。

原始 Part IV 页面和 `part4_inductor.py` 没有这份 bundle，也没有 kernel → Scheduler → IR → FX → Python 的可检查链。因此原始交付对 §12.7、§12.9 都是直接失败，而非仅仅“设备阻塞”。

当前新增的 `series_artifact_bundle.py` 生成了很多文件，但链条被人为拆开：

- `UnifiedGraphModel` 负责 symbolic/Dynamo/export/functional/AOT；
- 单独重写的 `backend_core` 才进入 backend codegen；
- `HigherOrderBranch` 是独立 export；
- `stage_node_mapping.json` 是各阶段 node 清单和文字 transition，不是同一 node/value 的可验证映射；
- backend provenance JSON 映射 post-grad FX 与生成的 C++ source label，但没有显式 Scheduler/IR stable ID；
- 没有规定的 liveness 综合表。

故当前状态只能判定 §12.7、§12.9 **部分满足**。

#### P0-4：页面把“部分运行”称作“已验证 Lab”，但未满足正式 Lab 契约

设计 §3.5 要求每个 Lab 至少具备最小输入、明确命令、预期 graph/output、一个正确案例、一个错误/边界案例、artifact 位置、baseline/environment，并真实运行。原始页面没有给出明确命令和 artifact 目录；缺 `cl` 是环境 blocker，不是刻意构造且具有教学含义的 semantic/error boundary。标题“已验证 Lab”容易让读者误以为页面规定实验已经完成。

### P1

#### P1-1：页面 17 的概念清单仍有实质缺口

已有 GraphLowering Interpreter、lowering registry、fallback/ExternKernel、pointwise/reduction、layout、realization 与 provenance 概述；但规格中的 lowering-pattern、device/mutation 处理，以及 decomposition、post-grad fusion、lowering 三者的责任边界仅零散出现，没有形成可核查机制链。Lab 没有 unsupported/custom-lowering 三态对照。

#### P1-2：页面 18 没有设计要求的 IR/index 实验

页面介绍了主要 IR 家族，但：

- `Scan` 仅被列名，没有结构或实际 dump；
- LoopBody/iteration domain/index expression 没有由具体输入反推出地址表达式；
- 没有 elementwise、broadcast、transpose、reduction、matmul 对照矩阵；
- 没有 Lab、命令、输出或 artifact。

此外其 declaration locator 表覆盖范围不完整：表中列出的范围没有实际包含 TemplateBuffer、ExternKernel、FallbackKernel；layout 范围也没有包含 NonOwningLayout。

#### P1-3：页面 19 的源码说明较强，但没有 lifecycle/peak 实验

页面正确区分 scheduler peak reorder、wrapper reuse planning 与 pooled static planner，也列出 alias/mutation/output/external 限制；但：

- in-place reuse 的具体生成条件和 wrapper 结果只停留在概述；
- 没有 fusion、save/recompute、view/copy 四组生命周期表；
- 没有 logical live bytes、static estimated peak、wrapper reuse、physical allocator peak 的分列；
- 当前新 Part IV artifact 中看到 `buf1 = buf0  # reuse` 只是单个例子，不能替代设计矩阵。

#### P1-4：页面 20 缺 collective/device/multi-output/locality 的闭环

Scheduler pipeline、dependency 类型、fusion rounds、cycle legality 与 priority 主干基本有 pinned source 支持；但 collective dependencies、multi-output/device boundary 和 locality reorder 只部分覆盖。原始 Lab 没有打印 dependency group，也没有关闭 reorder，更没有 kernel/performance 对照。当前新增 artifact 只用 `max_fusion_size=64` 与 `1` 证明 Scheduler 是否形成 fused node，没有 reorder 或性能证据。

#### P1-5：页面 21 的 provenance 仍主要是概念链

backend dispatch、AlgorithmSelectorCache、CachingAutotuner、wrapper/cache/provenance 的源码说明大体成立，但：

- loop ordering、tiling、mask、reduction codegen 和 wrapper allocation/launch 缺少具体 artifact 注释；
- 页面引用 `scheduler.py:9749-9795` 作为 group 到 kernel 主定位不精确；该范围主要是 codegen 入口/partition wrapper，关键 node dispatch 位于约 `9968-10080`；
- 原始 Lab 没有 kernel、wrapper 或 line mapping；
- 当前 mock sourcegen 能生成 `captured_cpp_kernel.cpp`、`output_code.py` 和 provenance JSON，但 kernel 没有被 compiler 接受或执行；
- Triton/autotune 仍正确标成未测试，不能计作 Lab 完成。

#### P1-6：新增 contract 主要检查“存在性”，不足以证明内容语义

`PartFourArtifactContractTest` 验证状态字段与文件非空；它没有断言：

- IR 中具体 reduction/broadcast index；
- pre/post fusion 的成员集合和 dependency 边；
- wrapper reuse/launch 语句；
- provenance JSON 是否同时覆盖 kernel label、FX node 和 user source；
- custom lowering 与 unsupported/fallback 基线之间的差异；
- mock codegen 输出的数值正确性。

`UnifiedSeriesArtifactContractTest` 同样主要验证文件存在与 feature 清单。feature 名称存在不证明同一 Graph 贯穿所有阶段。

#### P1-7：Batch 4 的 runtime/cache 回链更新不完整

Inductor index 与系列 index 已有链接；runtime memory index 未发现本系列新增回链，compile-cache 目录没有 `index.md`，页面 21 只链接了一个 cache 子页。设计 §9/§10 要求的六类索引更新尚未完整落实。

### P2

- 页面 17、21 有 Lab runtime header，页面 18–20 缺同等级的环境/命令/状态声明，证据标签不一致。
- 原始 Lab 不固定 seed，输出虽是数值相等判断，但 artifact 复查不够稳定。
- `part4_artifact_bundle.py` 的 `codegen_only_status=generated_not_executed` 基本诚实；更精确的名称应包含 `mock_compile`，避免读者把 no-op kernel 路径理解成真实加载。
- external eigvals fallback 和 matmul 虽真实执行，但没有保存对应 IR/wrapper/provenance artifact。
- changelog/index 对“21 页、13 个 Lab”的计数不应被读作 13 个 Lab 均满足 §3.5；建议同时发布 per-Lab acceptance 状态。

## 4. 每页缺口矩阵

| 页面 | 规格重点 | 源码说明覆盖 | 原始 Lab/artifact | 当前补丁增量 | 判定 |
|---|---|---|---|---|---|
| 17 FX→IR | pointwise/reduction/matmul/unsupported/custom lowering；职责边界 | GraphLowering、registry、fallback、Extern、realization 较好；lowering-pattern、device/mutation、三阶段职责部分缺失 | 仅 external matmul 成功；pointwise 被 `cl` 阻塞；无 unsupported/custom/artifact | custom lowering 进入 IR；eigvals fallback 执行；但无 missing-lowering 对照，custom 走 mock codegen | **部分，不通过 Lab** |
| 18 IR/index | elementwise/broadcast/transpose/reduction/matmul 的 IR 与索引 | IR family 主干有；Scan、LoopBody/index 具体性不足；locator 不全 | **没有 Lab 小节** | 新 trace 只覆盖小型 reduction/custom；没有五类矩阵或 matmul index | **失败** |
| 19 liveness/peak | fusion、save/recompute、view/copy 生命周期和峰值 | 三类 memory planning 区分较好；in-place reuse 机制偏浅 | 以缺 `cl` 为由不发布任何数据 | wrapper 中可见一次 reuse；无 save/recompute、view/copy、peak 表 | **失败** |
| 20 Scheduler | dependency/fusion groups；关闭 fusion/reorder；kernel/perf | pipeline、deps、fusion legality/priority 较好；collective/device/multi-output/locality 部分 | 有 max-fusion 代码，但未得到 group/kernel/perf | pre/post IR 表明 max-fusion 对 fused node 的影响；无 reorder、性能、完整 dependency dump | **部分，不通过 Lab** |
| 21 codegen/provenance | kernel、wrapper、autotune choices、source mapping | backend/autotune 源码概述较好；具体 loop/wrapper artifact 缺失；一处 locator 偏移 | 只有 external matmul 布尔值和 blocker | provenance 缺件已修复；mock source/wrapper 可检查；真实 C++ compile/execute 与 Triton autotune 未做 | **部分，不通过完整验收** |

## 5. 原始 `part4_inductor.py` 对正式 Lab 契约的符合度

| §3.5 要求 | 现状 | 判定 |
|---|---|---|
| 最小可运行输入 | 两个小函数和 tensor 输入 | 满足 |
| 明确运行命令 | 页面只引用脚本路径，没有命令 | 不满足 |
| 预期 graph/output | 只列旧运行输出，没有预期 IR/graph | 不满足 |
| 至少一个正确案例 | external matmul eager/compiled 相等 | 满足 |
| 至少一个错误/边界案例 | 缺 compiler 是环境异常，不是所要求的语义边界；没有 unsupported/custom 对照 | 不满足 |
| artifact 保存位置 | 没有 | 不满足 |
| baseline/environment | 页面 17/21 部分记录 runtime；各页不一致，脚本自身不写 manifest | 部分 |
| 真实运行且不伪造 | external matmul 已运行；pointwise 与 Triton 明确未冒充成功 | 部分满足，诚实但覆盖不足 |

审计复跑旧脚本得到：

```text
pointwise_inductor_status=blocked_missing_msvc_cl
matmul_output_matches=True
matmul_generated_kernels=0
triton_autotune_tested=False
```

## 6. 新增 remediation 的精确边界

### 6.1 已经真实改善的部分

`part4_artifact_bundle.py` 当前能够：

- `[R]` 真实执行 external matmul 并与 eager 对比；
- `[R]` 真实执行 eigvals fallback 并与 eager 对比；
- `[M]` 在 monkeypatch compiler 探测与 no-op `cpp_pybinding` 下走过 Dynamo→Inductor source generation；
- `[M]` 保存 FX readable/runnable/transformed、pre/post fusion IR、wrapper、captured C++ source；
- `[M]` 比较 `max_fusion_size=64` 和 `1` 的 fused scheduler 形态；
- `[M]` 注册 custom lowering 并确认其 IR 不含 fallback/extern/custom-op 名称；
- `[M]` 保存 provenance JSON。

本次审计开始时 provenance 文件缺失；加入 `trace.provenance_tracking_level=1` 后，当前复跑：

```text
test_part_four_emits_inspectable_artifacts_without_claiming_execution ... ok
Ran 1 test in 6.421s
OK
```

`series_artifact_bundle.py` 当前复跑：

```text
test_unified_model_emits_frontend_aot_and_backend_stage_artifacts ... ok
Ran 1 test in 6.057s
OK
```

这两项是有效回归保护。

### 6.2 不能从这些测试推出的结论

`_run_codegen_only()` 明确做了以下替换：

- compiler existence/version 返回 mock 值；
- compiler 名称固定为 `cl`；
- `AsyncCompile.cpp_pybinding` 捕获 C++ 字符串并返回 no-op callable；
- compiled wrapper 被调用，但生成的 C++ translation unit 没有交给 C++ compiler，kernel body 没有执行。

因此它证明“Inductor 在该 runtime 中构造了 source/wrapper/provenance artifact”，不证明：

- source 能由 MSVC 编译；
- ABI、allocation 和 launch 在真实 extension 中正确；
- kernel 数值正确；
- kernel 数量或性能；
- Triton candidate selection/autotuning。

建议状态值改为：

```text
source_codegen_observed_with_mock_compile_kernel_not_executed
```

并在 artifact manifest 中记录所有 monkeypatch。

### 6.3 统一 bundle 的剩余断点

新增 bundle 对 §6 的字段覆盖比原始交付完整，但不是同一计算图的连续 lowering：

```text
UnifiedGraphModel
  ├─ symbolic / Dynamo / export / functional ATen / AOT joint-fw-bw
  └─ 未直接送入 backend

backend_core（另写的等价前缀）
  └─ post-grad FX / IR / scheduler / mock-generated source

HigherOrderBranch
  └─ 独立 export 变体
```

要满足 §12.7，应把 AOT partition 实际产出的 fw/bw GraphModule（至少明确选择的 forward graph）继续送入 GraphLowering，并用 manifest 记录 graph hash、placeholder/output ABI 和每一跳的 value 映射；不应靠重写一个语义相似的 `backend_core` 接续。

## 7. 本机无 `cl` 条件下的可行性实验

以下是审计 scratch experiment 的 `[R]` 观察。它们证明方案可行，但由于当前没有保存为正式 Lab artifact，不能直接算作页面验收完成。

| 实验 | 观察 |
|---|---|
| GraphLowering pointwise/reduction | `make_fx` 后直接 `GraphLowering.run()` 成功；生成 `ComputedBuffer`/`Reduction`，origin 包含 add/relu/sin/mul/sum |
| Scheduler-only fusion | 两个独立 pointwise 输出在 `max_fusion_size=64` 时形成一个 `FusedSchedulerNode`，在 `1` 时为两个 `SchedulerNode` |
| Debug artifact | `DebugContext` + trace 配置生成 `ir_pre_fusion.txt`、`ir_post_fusion.txt`，不需要 `cl` |
| Unsupported/custom lowering | implicit fallback 产生 `FallbackKernel`/`MultiOutput`；关闭 implicit fallback 得到 missing-lowering error；注册 lowering 后变为 `ComputedBuffer(Pointwise)` |
| Static liveness/peak | 对 reduction→sin 两节点取得 reads/writes/last_usage；`estimate_peak_memory` 返回 input-specific 估算 `512` bytes、timeline `[512, 32, 0]` |
| View/copy | 纯 permute 为零 schedulable op，输出 `ReinterpretView`；contiguous clone 产生 `ComputedBuffer(Pointwise)` 和 contiguous `FixedLayout` |
| Broadcast index | `(3,1)+(1,4)` 的依赖索引分别读取 `c0`、`c1`，输出线性索引为 `4*p0+p1` |
| External wrapper/provenance | matmul 的 `GraphLowering.codegen()` 无 `cl` 也能生成 `extern_kernels.mm(...)` wrapper 和指向 user model 的 line map；无 generated loop kernel |
| 真正 pointwise codegen | 仍在 compiler 检查处失败：`Compiler: cl is not found` |

关键边界是：

```text
GraphLowering / Scheduler / static memory analysis / external wrapper
    可在当前机器运行

generated C++ extension compile + kernel execute
    需要 MSVC cl

Triton kernel + launch autotuning
    需要可用 CUDA/Triton 设备 lane
```

## 8. 已核对的 pinned-source 定位

以下定位均在 `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52` 上实际打开核对，不是从页面照抄。行号以该 commit 为准。

| 机制 | 定位 | 核对结论 |
|---|---|---|
| GraphLowering 是 FX Interpreter | `torch/_inductor/graph.py:386-395` | 类继承关系成立 |
| call_function lowering/fallback 顺序 | `torch/_inductor/graph.py:1402-1546` | registered/user lowering 与 fallback 分派主干 |
| origin 传播 | `torch/_inductor/graph.py:1925-1992` | `run_node` 设置 current origins/context |
| graph codegen orchestration | `torch/_inductor/graph.py:2991-3008` | Scheduler 与 wrapper/codegen 入口 |
| compile-to-module 边界 | `torch/_inductor/graph.py:3050-3067` | source generation 后才进入 module compile/load |
| lowering registration | `torch/_inductor/lowering.py:481-553` | broadcast/type promotion/validation wrapper |
| fallback handler | `torch/_inductor/lowering.py:2714-2729` | fallback 构造 `FallbackKernel` |
| Loops/Pointwise/Reduction | `torch/_inductor/ir.py:1057-1119`, `1219-1428` | loop IR 家族成立 |
| TensorBox/StorageBox realization | `torch/_inductor/ir.py:10547-10607` | lazy box 与 realize 路径 |
| IR origins | `torch/_inductor/ir.py:589-645` | origin/meta 字段 |
| Operation dependency protocol | `torch/_inductor/ir.py:956-1018` | reads/writes/dependency 入口 |
| Layout family | `torch/_inductor/ir.py:4399-4472`, `4726-4755`, `4962` | Fixed/Flexible/NonOwning 分散定义 |
| Buffer/OperationBuffer/ComputedBuffer | `torch/_inductor/ir.py:5160-5435` | storage 与 schedulable operation 的组合 |
| Template/Extern/Fallback/MultiOutput | `torch/_inductor/ir.py:5876`, `6995`, `9314`, `10144` | 页面 18 的单一区间未覆盖这些声明 |
| Scheduler pipeline/last use | `torch/_inductor/scheduler.py:4235-4410` | operation 建图、reorder 与 last-use 主干 |
| IR op→SchedulerNode 映射 | `torch/_inductor/scheduler.py:4641-4651` | node 类型选择 |
| dependencies/alias setup | `torch/_inductor/scheduler.py:4689-4760` | reads/writes、alias/mutation 依赖 |
| DCE/topological schedule | `torch/_inductor/scheduler.py:5055-5129` | DCE 与拓扑顺序 |
| iterative fusion rounds | `torch/_inductor/scheduler.py:5268-5301` | 最多多轮融合 |
| candidate grouping | `torch/_inductor/scheduler.py:6830-6884` | fusion candidate window/group |
| cycle check | `torch/_inductor/scheduler.py:6886-6930` | DFS cycle legality |
| can_fuse legality | `torch/_inductor/scheduler.py:7818-7905` | rollback、stream/mempool 等限制 |
| fusion score | `torch/_inductor/scheduler.py:8631-8675` | profitability/priority |
| last use/free | `torch/_inductor/scheduler.py:8959-8995` | buffer last-use/free |
| static peak estimator | `torch/_inductor/memory.py:1016-1105` | 不依赖真实 allocator |
| reuse key | `torch/_inductor/codegen/wrapper.py:108-138` | device/dtype/size/alignment/stream/mempool 条件 |
| peak-aware reuse | `torch/_inductor/codegen/wrapper.py:963-1015` | wrapper reuse planning |
| alias/MultiOutput/ReuseLine | `torch/_inductor/codegen/wrapper.py:1088-1188` | wrapper allocation/reuse 表达 |
| planning vs reuse | `torch/_inductor/codegen/wrapper.py:2526-2582` | wrapper planning 分派 |
| pooled planner phases | `torch/_inductor/codegen/memory_planning.py:665-683`, `777-817` | greedy/static planning |
| memory pool modes | `torch/_inductor/config.py:248-268` | 配置值 |
| fusion thresholds | `torch/_inductor/config.py:988-1015` | fusion 相关配置 |
| backend contract | `torch/_inductor/codegen/common.py:309-318`, `389-434` | scheduling/codegen 接口 |
| built-in dispatch | `torch/_inductor/codegen/common.py:515-613` | backend-specific dispatch |
| origin→wrapper | `torch/_inductor/scheduler.py:9053-9069` | wrapper context 中保存 origin |
| codegen entry 与实际 node dispatch | `torch/_inductor/scheduler.py:9749-9755`, `9968-10080` | 页面 21 应补充后一区间 |
| AlgorithmSelectorCache | `torch/_inductor/select_algorithm.py:3949-4039` | choice lookup/benchmark/cache |
| multi-template/fallback | `torch/_inductor/select_algorithm.py:4196-4255` | algorithm-choice 分支 |
| CachingAutotuner | `torch/_inductor/runtime/triton_heuristics.py:531-620`, `1789-1859`, `2412-2525` | runtime autotuning/cache |
| pointwise heuristic registry | `torch/_inductor/runtime/triton_heuristics.py:4365-4438` | pointwise heuristic 配置 |

## 9. 最小修订与实验计划

### Wave 1：先修证据标签，不等待设备

1. 将页面 17、19、20、21 的“已验证 Lab”拆为：
   - 本机真实执行；
   - source-backed；
   - mock source-codegen；
   - blocked device/compiler lane。
2. 每页添加精确命令、runtime/source baseline、artifact 相对目录和 expected assertions。
3. 明确写出：缺 `cl` 只阻塞 generated C++ compile/execute，不阻塞 GraphLowering/Scheduler。

### Wave 2：建立无 `cl` 的 CPU internal-stage Lab

把本报告 §7 的 scratch experiments 固化为一个可复跑脚本与 artifact bundle：

1. page 17：pointwise、reduction、external matmul、missing lowering、implicit fallback、custom lowering 五态；
2. page 18：elementwise/broadcast/transpose/view-copy/reduction/matmul 的 IR class、layout、iteration/index 表；
3. page 19：逐节点 reads/writes/last-use、logical bytes、static estimated peak、wrapper reuse；另接 AOT saved/recompute 两种 fw/bw；
4. page 20：保存 dependency edges、pre/post fusion groups；分别切换 fusion 和 peak-memory/locality reorder，先比较 Scheduler 结构；
5. page 21：对 external matmul 保存真实 wrapper/line map，对 generated pointwise 保存明确标注的 mock source/provenance。

每个 case 都应有一个正确断言和一个语义边界/错误断言，而不是拿“compiler 未安装”充当边界案例。

### Wave 3：补真实设备 lane

1. Windows + MSVC：真实编译和执行 CPU generated C++，验证数值、kernel count、wrapper allocation/reuse 和性能；
2. CUDA + Triton：记录 algorithm choices、candidate timing、selected config、generated kernel、launch 与 cache hit；
3. 禁止把 mock sourcegen 的 artifact 复制到真实 lane 目录；manifest 必须记录 compiler/device 和 execution status。

### Wave 4：打通同一贯穿 Graph

1. 由 `UnifiedGraphModel` 产生 AOT joint/fw/bw；
2. 将实际 partition 后的 forward（以及可行时 backward）GraphModule继续送入 post-grad/GraphLowering，而不是重写 `backend_core`；
3. 为每阶段写 graph hash、placeholder/output ABI、stable semantic value ID；
4. 生成规定的 split/copy/fusion/liveness/provenance 表；
5. provenance 至少包含：

```text
user source span
→ Dynamo/FX node
→ functional/AOT node
→ post-grad FX node
→ IR operation/buffer
→ Scheduler node/fusion group
→ wrapper launch / generated source label
```

### Wave 5：补 Batch 4 导航与最终 gate

更新 runtime memory 与 compile-cache 回链，建立 cache index 或明确目录入口；然后运行：

- 两个 artifact contract；
- 内容语义 assertions，而非仅文件存在；
- markdown/link/Related Pages 检查；
- 独立审计确认 §12.7、§12.9。

## 10. 验收状态汇总

| 条款 | 原始交付 | 当前补丁后 | 说明 |
|---|---|---|---|
| §5.5 pages 17–21 内容 | 部分 | 部分 | 源码说明主干较好，指定 Lab 仍未闭环 |
| §6 贯穿模型/bundle | 失败 | 部分 | 文件显著增加，但 frontend/backend 不是同一实际 Graph 链；缺综合表 |
| Batch 4 IR/Scheduler/kernel artifacts | 失败 | 部分 | IR/Scheduler/mock source 有；真实 generated kernel 与完整 matrices 无 |
| Batch 4 end-to-end provenance | 失败 | 部分 | provenance JSON 已补；Scheduler/IR stable mapping 与真实 kernel execution 无 |
| Batch 4 runtime/cache links | 部分 | 部分 | Inductor 有，runtime/cache 不完整 |
| §12.7 完整阶段 artifact bundle | 失败 | **部分，未通过** | contract 文件存在不等于同一 Graph 贯穿 |
| §12.8 Part III pass | 不在本报告范围 | 不在本报告范围 | 由 Part III 独立审计判定 |
| §12.9 Scheduler/kernel→FX/source | 失败 | **部分，未通过** | mock-generated source→post-grad FX→user source 可见；Scheduler/IR 显式映射和真实 kernel 缺失 |

最终判定：**Part IV 需要修订后复审；当前不应标记为设计完成。**
