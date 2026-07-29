# 11 · 图阶段边界、节点身份与 Provenance

> 前置：[[09_aotautograd_joint_forward_backward_graphs]]、[[10_saved_tensors_recompute_and_runtime_abi]]
> 当前实现基线：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`
> Lab 环境：PyTorch `2.9.1+cpu`
> 最后更新：2026-07-28

## 1. “同一个算子”不是“同一个 Node”

一段用户 `relu(x @ w + b)`可经历：

```text
Python source
→ Dynamo FX Node
→ decomposition 后多个 ATen FX Nodes
→ joint Node
→ fw Node 或 bw copied Node
→ post-grad replacement Nodes
→ lazy Inductor IR values
→ realized Operation/Buffer
→ fused Scheduler group
→ kernel/wrapper source lines
```

语义可追溯，但object identity不会贯穿。

## 2. 阶段地图

| 阶段 | 主数据结构 | 典型改变 |
|---|---|---|
| Dynamo | FX Graph + guards | Python分段、specialization |
| pre-grad | FX Graph | 高层canonicalization |
| AOT capture | joint FX Graph | functionalization、decomposition、grad trace |
| joint passes | joint FX Graph | partition前rewrite |
| partition | fresh fw/bw Graph | copy、save/recompute boundary |
| post-grad | 各自FX Graph | 低层pattern/device rewrite |
| GraphLowering | Interpreter env + Inductor IR | target lowering、lazy composition |
| realization | Buffer/Operation registration | 具名storage/operation |
| Scheduler | SchedulerNode/dependencies | DCE、fusion、reorder、liveness |
| codegen | backend kernel + wrapper IR/source | indexing/launch/allocation |

### 概念生命周期不等于某个 artifact 目录的连续编译链

上表描述编译系统可能经历的阶段，不自动证明一组 dump 是由前一个文件逐级喂给后一个文件
得到的。必须区分：

- **同一用户计算的独立 capture**：例如分别运行 `symbolic_trace`、`torch.export.export`、
  functional `make_fx`和 `aot_module`。它们可用于比较语义和结构，但前一份 GraphModule
  不是后一条 API 的输入；
- **同一次连续编译中的阶段变化**：例如一次 AOT `partition_fn`收到 joint，返回 fresh
  fw/bw，随后同一次调用触发 fw/bw compiler callback；或者一次 `torch.compile`后端调用
  从 post-grad FX 继续产生 IR、Scheduler group和generated source。

因此，连续性是需要 artifact 或 callback 关系证明的 claim，不能从文件名顺序推导。

## 3. 身份变化的五种模式

### 一对一但 fresh

partition `node_copy`：语义对应，但owner/object不同
（`torch/_functorch/partitioners.py:514-538`;
`torch/_functorch/partitioners.py:544-568`;
`torch/_functorch/partitioners.py:654-661`;
`torch/_functorch/partitioners.py:676-705`）。

### 一对多

decomposition将一个composite Node展开为多个primitive Nodes。

### 多对一

pattern replacement可用一个fused op替换子图；Scheduler fusion将多个operations组成一个
FusedSchedulerNode/kernel candidate。

### 一对零

DCE删除dead Node；view/lazy expression也可能不形成独立realized operation。

### 零对一

wrapper插入allocation/guard/copy/token；codegen产生辅助kernel或runtime assertion。

## 4. `node.meta`是provenance carrier，不是稳定schema

常见keys：

- `stack_trace`；
- `source_fn_stack`；
- `nn_module_stack`；
- `val`/`tensor_meta`；
- `from_node`/sequence number；
- `is_forward`/`is_backward`；
- `recompute`；
- stream/mempool/mutation region。

不同前端/stage保证的keys不同。pass应：

- 只依赖该stage contract保证的meta；
- replacement时有选择地传播；
- 不把debug meta当semantic edge；
- 对missing key有明确策略。

## 5. GraphLowering origin propagation

`run_node()`以当前FX Node为origin，并收集lowered arguments中的origins；在创建IRNode期间
安装origin、stream与mempool context
（`torch/_inductor/graph.py:1960-1992`）。

`IRNode`保存：

- origins set；
- origin_node；
- traceback；
- annotations；
- stream index；
- mempool
  （`torch/_inductor/ir.py:589-645`）。

因此一个IR value可继承多个upstream origins。

## 6. lazy realization如何改变映射

多个FX pointwise lowering可组合在同一lazy `StorageBox` expression中。只有被要求realize时，
`StorageBox.realize()`创建 `ComputedBuffer`，注册buffer与operation，并复制origin/trace/
stream/mempool
（`torch/_inductor/ir.py:10578-10607`）。

所以：

- 每个FX Node未必有buffer name；
- buffer creation时机可晚于该FX Node lowering；
- realized operation origins可覆盖一段FX链。

## 7. Scheduler fusion后的provenance

SchedulerNode包装realized operation。fusion group的nodes集合聚合origin sets。
wrapper codegen进入source context时，Scheduler按origin所属FX graph顺序选择latest origin
（`torch/_inductor/scheduler.py:9053-9069`）。

这个选择服务于generated source context/traceback，不表示kernel只来自latest Node。

## 8. Generated code mapping

理想mapping是many-to-many：

| 方向 | 用途 |
|---|---|
| Python → FX | 哪段用户代码被捕获 |
| FX → IR | 哪个lowering/realization来自哪些Node |
| IR → Scheduler | 哪个operation产生哪些buffers/deps |
| Scheduler → kernel | 哪些nodes被融合/模板化 |
| kernel line → origin | 生成代码/traceback定位用户源码 |

仅凭相似name反推不可靠；name会重命名、复制、去重。

`Graph.node_copy`会浅复制 meta（`torch/fx/graph.py:2386-2420`），由计算值改成 fresh
placeholder 的路径也继承 meta（`torch/_functorch/partitioners.py:544-550`）。

AOT partition 是本系列可以取得 exact old→new 证据的一段：Lab 在一次
`partition_fn`中向 joint `node.meta`注入唯一 audit token，`default_partition`返回后立即
在 fw/bw 上回收 token。这能映射本次运行中的真实对象，同时仍需声明 token 是 Lab 探针，
不是 production stable ID；fresh output Node则按结构重建处理。

## 9. Dump名称必须绑定stage

调试日志常见：

- Dynamo graph；
- pre-grad graph；
- joint graph；
- fw/bw graph；
- post-grad graph；
- IR pre-fusion/post-fusion；
- output code。

“这是fusion graph”不能只看文件名。FX post-grad仍是operator graph；真正Scheduler fusion在
lowering/realization之后。

## 10. Pass placement为何依赖stage identity

同一语义pattern：

- pre-grad可能看到 `nn.Linear`/高层op；
- joint看到forward与gradient；
- post-grad看到functional ATen；
- lowering pattern直接产生IR；
- Scheduler只看到buffer operation。

把规则移到另一stage，target、metadata、alias contract和cleanup ownership都会变化。

## 源码跟读：identity 在哪里断开，provenance 又怎样继续传播

### 1. FX 跨 Graph 复制：新 identity，浅复制 metadata

`Graph.node_copy` 先用调用者提供的 `arg_transform` 递归映射 args/kwargs，再调用目标 Graph
的 `create_node`；最后对 `node.meta` 做 `copy.copy`
（`torch/fx/graph.py:2386-2420`）。

这个 API 同时给出两项不同保证：

```text
结构 identity:
  result_node is not source_node
  result_node.graph is destination_graph
  result_node.args 引用 destination graph 中的映射 Node

provenance carrier:
  result_node.meta 是新的 dict（浅复制）
  meta 内嵌对象未承诺深复制或稳定 schema
```

因此 AOT partition 可以借 meta token 追踪 old→new，但不能把共享的 nested meta payload
误当成跨阶段不可变对象，也不能假设所有 pass 都调用 `node_copy`。decomposition、fusion、
fresh placeholder/output reconstruction 需要自己的 provenance 传播策略。

### 2. FX→Inductor：当前 FX Node 与输入 IR origins 被并集

`GraphLowering.run_node` 在 lowering 前建立 `origins = {current_fx_node}`；对 call_function，
它从已 lowered args/kwargs 收集 origins 并做并集
（`torch/_inductor/graph.py:1960-1977`）。随后在执行 lowering 的动态作用域中安装：

- `IRNode.current_origins(origins)`；
- current stream；
- current mempool；
- current FX Node
  （`torch/_inductor/graph.py:1986-1992`）。

IR constructor 不需要每个 lowering 手工传 origin 参数。`IRNode.current_origins` 将新集合与
外层 current origins 合并，并在 context 退出时恢复旧值
（`torch/_inductor/ir.py:613-621`）。同一模式也用于 stream/mempool
（`torch/_inductor/ir.py:623-645`）。

这个动态 context 解释了 provenance 为什么自然变成集合：若当前 FX op 的 lowering 组合了
多个已有 lazy IR values，新 IR 同时继承当前 callsite 与所有输入来源。

### 3. Lazy value→realized buffer：operation identity 到此才出现

`StorageBox.realize` 首先检查 data 是否已经 realized；对于 Pointwise/Reduction/Scan/Sort，
它创建 `ComputedBuffer` 与 `FlexibleLayout`，向 `V.graph` 注册 buffer 和 operation，然后
把 origins、origin node、traceback、stream、mempool 复制到新 realized object
（`torch/_inductor/ir.py:10578-10607`）。

状态变化是：

```text
realize 前
StorageBox.data = lazy Pointwise/Reduction/...
没有独立 registered operation name

realize 后
StorageBox.data = ComputedBuffer(name=...)
GraphLowering.buffers / operations 增加
origins 继续保留
```

所以“某 FX Node 没有独立 buffer”可能有两种完全不同的原因：

- 它的 lazy expression 被内联进后来 realized consumer；
- 它确实被折叠、删除或没有运行时 operation。

不能把“未单独 realize”直接称为 dead。dead 是可观察语义/依赖判定，lazy inline 是实现
表示选择。

### 4. Scheduler fusion：多个 operation identity 聚合成 group

SchedulerNode 从 registered operation 构造；fusion 后 group 的 `get_nodes()` 包含多个
成员，每个成员又携带 origin set。进入 wrapper source context 时，Scheduler 收集所有
origins，按各自 FX graph order 建 index，选择顺序最晚的 origin 作为当前 codegen context
（`torch/_inductor/scheduler.py:9053-9069`）。

“选择 latest origin”是为了给生成代码、traceback 或 profiler 一个代表位置；完整 group
仍可能来自多个 FX Nodes。把该代表 origin 当成唯一 source 会把 many-to-one fusion 错写成
one-to-one。

### 5. 一条 provenance 链应记录关系类型，而不只是 ID

| 边界 | 典型关系 | 可依赖的实现载体 |
|---|---|---|
| source→FX | one-to-many / segment | stack/source/module meta、guards |
| joint→fw/bw | one-to-zero/one/two | fresh Node + shallow-copied meta + ABI |
| FX→lazy IR | one-to-many / many-to-one | dynamic origin context |
| lazy IR→operation | inline 或 realize | `StorageBox.realize` registration |
| operation→fusion group | many-to-one | Scheduler group members/origin sets |
| group→source line | many-to-many + representative | wrapper context/line map |

因此一个可靠 artifact 至少要写出 `relationship`、source/destination stage、是否同一次运行、
是否保持 object identity、使用何种 token/origin 证据。仅存 `{old_name: new_name}` 无法区分
复制、融合、独立 recapture 和偶然同名。

### 源码边界

这些源码能证明 `node_copy`、origin context、realization 与 Scheduler representative
选择；它们没有承诺 `node.meta` 或 IR origin set 是跨版本稳定公共 schema。审计 token 和
hook 适合作为固定版本观测工具，不应成为生产缓存 key。生产 pass 需要使用所在 stage 明确
支持的 provenance 字段，并为缺失或多对多关系保留诚实表示。

## 11. Cache也不能只用Node identity

compile cache通常以code/guards、graph serialization、input metadata、config等形成key。
Node Python object只在一次Graph生命周期中稳定。跨进程/序列化/profiling correlation需用：

- source location；
- deterministic graph serialization；
- debug handle；
- origin set；
- stable hash/key protocol。

## 12. 已验证 Lab 与 artifact bundle

从知识库根目录运行：

```powershell
python -B wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\series_artifact_bundle.py `
  --output-dir wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\artifacts\end_to_end
```

`UnifiedGraphModel.forward`调用同一个 `backend_core`，稳定计算前缀包含 parameter、buffer、
view/mutation、matmul、pointwise、reduction 与 structured output；HOP branch 作为独立
export 变体，避免把 child GraphModule ownership 压扁。正例生成：

- model source、symbolic FX、Dynamo FX/guards；
- ExportedProgram/signature、functional ATen；
- AOT joint、fw、bw、exact partition mapping 与 partition ABI；
- post-grad FX、Inductor IR pre/post fusion；
- wrapper、captured C++ source 与 provenance JSON。

错误/边界检查包括 export range 越界、cond branch metadata 不一致、AOT cross-owner
Node 引用必须为 0。实测关键输出：

```text
forward_matches=True
gradient_matches=True
dynamic_export_has_range_constraints=True
export_out_of_range_rejected=True
aot_has_joint_forward_backward=True
aot_cross_graph_node_refs=0
aot_joint_partition_mapping_exact=True
aot_partition_to_compiler_callback_continuity=True
aot_saved_slot_binding_origins_match=True
artifact_bundle_continuity=partial
hop_branch_captured=True
hop_invalid_branch_rejected=True
backend_codegen_status=generated_not_executed
```

这里的 `partial`不是运行失败，而是对证据范围的准确描述。本次 manifest 列出的 artifacts
由一次脚本执行生成或刷新，但内部包含多次独立 capture：

- symbolic FX、Dynamo FX、ExportedProgram、functional ATen 与 AOT joint 都来自同一
  `UnifiedGraphModel`计算，却不是一条把前一个 GraphModule作为后一个输入的连续流水线；
- AOT joint→partition fw/bw→fw/bw compiler callback 是一条连续段；
- `backend/`是对 `backend_core`的独立 `torch.compile`运行；它保持同一计算前缀，但把
  parameter/buffer值作为显式输入，并没有直接消费上面的 `aot_forward.py`；
- 在 `backend/`内部，FX→IR→Scheduler→generated source 是同一次后端调用中的连续段。

`artifact_manifest.json`机器可读地记录上述 capture run、连续段、缺失的连续 edge 与
evidence boundary。由此不能声称“本 bundle 已把同一个 Dynamo Node 一路编译成同一个
kernel”；可以声称的是“同一计算已有分阶段实物，并且 AOT 段与后端段各自有连续性证据”。

为补上这个宽模型故意保留的 AOT→Inductor 断点，另有一个范围更窄、但对象链连续的
`extern_matmul_only` Lab：

```powershell
python -B wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\part2_continuous_aot_inductor.py `
  --output-dir wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\artifacts\part2_continuous_aot_inductor
```

它在同一次 `torch.compile` 中包装真实 partition、fw/bw compiler、`GraphLowering.__init__`
和 `Scheduler.__init__`，并用同一个 run-id 与 lab-only origin token 记录：

- forward partition 返回的 `GraphModule`、`Graph`、owner 直接进入 compiler callback，
  并在 `GraphLowering`入口保持同一对象；
- backward 在 partition 返回与 compiler callback 之间会重建 module/graph/owner；不能
  声称 backward 对象 identity 保持，但 token 集合完整保留；
- callback 到 `GraphLowering`入口的 backward module/graph/owner 再次保持；
- `GraphLowering`源码令 `self.orig_gm = gm.__copy__()`
  （`torch/_inductor/graph.py:583`），因此新的 owner 指向浅拷贝；
- 本次 Lab 到 Scheduler 时 module 与 Graph identity 仍保持，同时观察到
  `graph.owning_module`转向上述浅拷贝；这个 owner transition 是预期事实，不是追踪失败；
- 真实 Scheduler 构造发生在 `torch/_inductor/graph.py:2989`；
- 本次 fw 得到 `1 IR op / 1 Scheduler node / 2 reads / 1 write`，bw 得到
  `2 IR ops / 2 Scheduler nodes / 4 reads / 2 writes`，且 Scheduler origins 仍来自 AOT token。

数值与两组梯度均等于 eager，包装器都在 `finally` 中恢复。该 Lab 没有重捕获来冒充
连续性，也没有 mock compiler；它把“AOT partition→Inductor lowering→Scheduler”的特定
连续边提升为 `[R]`，并不把宽模型 bundle 的其他独立 capture 伪装成同一对象流水线。

该 Lab 只执行 ATen extern matmul 路径，没有编译 Inductor 生成的 native C++ kernel；
因此 native C++ correctness/performance 仍是 `[B]`。

证据边界必须保留：后端 source generation 通过 mock compiler discovery，并把
`AsyncCompile.cpp_pybinding`替成 no-op 来捕获 source。`backend/output_code.py`、
`captured_cpp_kernel.cpp`、IR 与 provenance 是真实生成的 `[M]` artifact，但 C++ translation
unit 没有交给 MSVC、kernel 没有执行，不能用来声称 native correctness/performance。

`stage_node_mapping.json`列出各 stage Node、capture run ID和relationship；其中把
independent recapture显式标为 `continuous_transition=false`。它本身主要是 semantic
stage records，不是凭 name 生成的 old/new Python object identity 表。
`aot_joint_to_fw_bw_node_mapping.json`则是同一次 AOT partition 的真实 old→new 表：
本次运行31个 joint Nodes中30个映射到至少一张提取图，剩余1个是被结构重建的 joint
output；12个进入 fw、21个进入bw、3个在两边均有 fresh destination。artifact同时记录
compiler callback destinations；即使本环境的 backward partition Graph 与 bw compiler
callback Graph不是同一个 Python对象，audit token集合仍完整保留。

AOT 的 runtime slot/ownership 证据在 `aot_partition_abi.json`，其中3条 saved output
slots分别绑定3个 fresh backward placeholders；每一对携带相同 joint audit origin，
且 cross-graph Node refs为0。完整 artifact 树与环境见
[`labs/README.md`](labs/README.md)。

post-grad→generated source 的 provenance 在
`backend/inductor_provenance_tracking_node_mappings.json`；它随 mock compiler/no-op
kernel 的 source-codegen 路径生成，证据等级是 `[M]`，不证明 native kernel 已编译或执行。

Scheduler/IR 尚无统一 stable ID；在取得新的连续性证据之前只能保留
many-to-many/部分映射，而不能凭 name 强行一一对应。

## 13. 排查一个生成kernel的顺序

1. 从wrapper/kernel name找到Scheduler codegen group；
2. 列出group内部 SchedulerNodes与buffer names；
3. 读取每个operation的IR origins；
4. 定位对应post-grad FX Nodes；
5. 通过stack/source meta回到Dynamo/user source；
6. 检查decomposition/partition/replacement造成的一对多、多对一；
7. 若找不到一一对应，不要强造映射。

## 14. 复杂度与存储

provenance传播成本不总是常数：

- 每个IRNode保存origin set，密集union可增长；
- fusion group聚合多个operation；
- 完整many-to-many map最坏可达 `Θ(N_source × N_generated)`；
- stack trace字符串和generated line maps也占空间。

工程实现常用set去重、只保留selected origin或按需debug，以控制编译内存。

## 15. 本篇结论

```text
identity:
  只在当前数据结构和生命周期内可靠

provenance:
  通过 meta、origin set、line map 与 signature
  维护跨阶段语义关系

correct debugging:
  使用 many-to-many stage map
  不依赖同名或同一 Python object

continuity:
  区分独立 recapture 与同一次编译内 transition
  只对有 callback、token 或 provenance artifact 的边声明连续
```

## 学习顺序

- 上一篇：[[10_saved_tensors_recompute_and_runtime_abi]]
- 下一篇：[[12_fx_graph_editing_primitives_and_invariants]]

## Related Pages

- [[19_torch_compile_end_to_end/00_pytorch_graph_series_index]]
- [[07_graph_capture_frontends_and_tracing]]
- [[09_aotautograd_joint_forward_backward_graphs]]
- [[10_saved_tensors_recompute_and_runtime_abi]]
- [[17_fx_lowering_to_inductor_ir]]
- [[21_codegen_kernel_mapping_autotuning_and_provenance]]
- [[Pytorch_Compile_Debug_Analysis]]
