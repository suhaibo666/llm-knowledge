# 08 · 图规范化、Decomposition 与 Functionalization

> 前置：[[07_graph_capture_frontends_and_tracing]]、[[05_graph_effects_alias_mutation_and_order]]
> 当前实现基线：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`
> Lab 环境：PyTorch `2.9.1+cpu`
> 最后更新：2026-07-28

## 1. 捕获成功为何还不够

前端产物可能包含：

- module/high-level composite ops；
- positional/keyword/default 参数的多种 spelling；
- view 与 in-place mutation；
- duplicated/aliased inputs；
- Python containers；
- backend不支持的 operator set；
- 等价但形态不同的表达式。

pass 若直接覆盖全部组合，规则数量爆炸且 legality 难证明。规范化的目的不是“优化一次”，
而是建立后续阶段可依赖的 canonical contract。

## 2. 五类规范化不要混成一类 pass

| 机制 | 改变什么 | 主要目的 |
|---|---|---|
| schema normalization | args/kwargs/default spelling | 统一调用 ABI |
| decomposition | composite op → 更小 ops | 收敛 operator set |
| functionalization | mutation/view → functional关系与边界输出 | 显式 alias/mutation |
| synthetic base/dedupe | 输入 identity/alias calling convention | 保留跨输入 alias |
| canonical graph passes | algebraic/layout/pattern形态 | 提高匹配与后端机会 |

它们可能在不同 capture/stage发生，不能画成所有程序必经的五个固定顺序框。

## 3. Schema normalization

`aten.add.Tensor(x, y, alpha=1)`可由位置参数、keyword/default 多种写法表示。pattern若按 raw
args比较，会把语义相同调用当成不同结构。

FX/PatternMatcher 可借 operator schema 补齐 default kwargs、flatten结构后递归比较。
当前 `_TargetArgsExpr`的 normalization/match逻辑位于
`torch/_inductor/pattern_matcher.py:881-898`、
`torch/_inductor/pattern_matcher.py:901-906`、
`torch/_inductor/pattern_matcher.py:909-935`、
`torch/_inductor/pattern_matcher.py:963-990`与
`torch/_inductor/pattern_matcher.py:991-1019`。

Schema normalization只统一 call signature，不做代数等价证明。

## 4. Decomposition

decomposition把一个 operator实现为其他 operators，例如 composite activation、loss 或
backward formula展开成后端更熟悉的原语。

收益：

- lowering只需覆盖收敛 operator set；
- 暴露跨 op pattern/fusion；
- 统一 autograd/functional行为；
- 后端可复用已有实现。

代价：

- Node 数增加；
- 可能丢失高层语义或高效专用 kernel机会；
- decomposition顺序影响 pattern；
- dynamic/alias/numerical behavior必须等价。

Decomposition不是 post-grad pattern replacement 的同义词。AOT capture通过 decomposition
table配置 tracing；Inductor也有自己的 decomposition/fallback决策。

## 5. Functionalization

functionalization追踪 alias并把 mutation转为 functional updates。AOT metadata先分析输入/
输出 mutation 与 alias，再决定 graph/runtime ABI
（`torch/_functorch/_aot_autograd/collect_metadata_analysis.py:252-274`;
`torch/_functorch/_aot_autograd/collect_metadata_analysis.py:276-289`;
`torch/_functorch/_aot_autograd/collect_metadata_analysis.py:291-320`;
`torch/_functorch/_aot_autograd/collect_metadata_analysis.py:447-475`;
`torch/_functorch/_aot_autograd/collect_metadata_analysis.py:488-510`;
`torch/_functorch/_aot_autograd/collect_metadata_analysis.py:760-805`）。

它不是简单字符串替换：

```text
x.add_(1)
```

要同时处理：

- x 的新值；
- 返回值与 x alias；
- 其他 views；
- input mutation是否用户可见；
- autograd版本计数；
- no-grad/inference语义。

## 6. Controlled mutation tail

当前 AOT functional graph contract允许受控 `copy_`尾部，用于 `keep_input_mutations`等路径；
若关闭 functionalization则不强制该 invariant
（`torch/_functorch/_aot_autograd/graph_capture.py:340-403`;
`torch/_functorch/_aot_autograd/graph_capture_wrappers.py:1030-1056`;
`torch/_functorch/_aot_autograd/graph_capture_wrappers.py:1058-1070`;
`torch/_functorch/_aot_autograd/graph_capture_wrappers.py:1088-1104`;
`torch/_functorch/_aot_autograd/graph_capture_wrappers.py:1130-1144`）。

所以准确表述是：

> functionalization把一般 alias/mutation转换为后端可处理的 functional关系，并通过
> signature/runtime wrapper或受控尾部恢复 mutation；不是保证图中绝无任何 effectful op。

## 7. Duplicate input 与 synthetic base

### duplicate object

`f(x, x)`两个参数位置引用同一 Tensor。若编译器当成独立值，mutation/alias会错。
`AOTDedupeWrapper`在 capture前规范化。

### aliased views

`f(base[:], base[1:])`是不同 Tensor对象但共享 storage。若发生 mutation，AOT可用
`AOTSyntheticBaseWrapper`合并为 synthetic base，再在图内重建 views
（`torch/_functorch/_aot_autograd/runtime_wrappers.py:1586-1608`;
`torch/_functorch/_aot_autograd/runtime_wrappers.py:1612-1639`;
`torch/_functorch/_aot_autograd/runtime_wrappers.py:1660-1689`;
`torch/_functorch/_aot_autograd/runtime_wrappers.py:1696-1716`;
`torch/_functorch/_aot_autograd/runtime_wrappers.py:1725-1747`;
`torch/_functorch/_aot_autograd/runtime_wrappers.py:1749-1766`;
`torch/_functorch/_aot_autograd/runtime_wrappers.py:1844-1863`;
`torch/_functorch/_aot_autograd/runtime_wrappers.py:1864-1880`;
`torch/_functorch/_aot_autograd/runtime_wrappers.py:1909-1938`;
`torch/_functorch/_aot_autograd/runtime_wrappers.py:1945-1960`;
`torch/_functorch/_aot_autograd/runtime_wrappers.py:1963-1978`;
`torch/_functorch/_aot_autograd/runtime_wrappers.py:1979-1999`;
`torch/_functorch/_aot_autograd/runtime_wrappers.py:2001-2018`）。

这两种机制解决 object identity 与 storage alias 的不同问题。

## 8. Lifted state 与 functional graph

- lifting：parameter/buffer/constant从隐式 module access变为显式 inputs；
- functionalization：mutation从隐式 write变为值关系/额外 outputs。

ExportGraphSignature同时记录 lifted inputs和 mutation outputs
（`torch/export/graph_signature.py:166-181`），但两个概念不能互相替代。

## 9. Canonicalization、constant folding 与 CSE

### canonicalization

把多个等价 spelling转成稳定形态，例如 argument order、reshape/view序列、dtype cast组合。

### constant folding

只有依赖 compile-time known values的子图可求值。symbolic value是否 constant需由
ShapeEnv/guards证明；不能把 tracing hint直接折叠。

### CSE

合并相同 pure expression需要确认：

- target pure；
- args/kwargs等价；
- alias/mutation/effect无差异；
- random/collective不可误合并；
- metadata与source mapping处理合理。

这些是图 pass，不应和 decomposition/functionalization混成一个不可分步骤。

## 10. 顺序为何影响 pattern 与 partition

```text
先 decomposition → pattern看到 primitive chain
先 pattern fusion → 保留高层专用 op

先 functionalization → mutation变为值关系，规则更易证明
过早去掉 high-level metadata → partition失去checkpoint/semantic提示
```

pass placement本质是选择一个 invariant最合适的层。详见
[[15_graph_pass_pipeline_ordering_and_fixpoint]]。

## 11. 与 AOT joint 的边界

metadata collection在 joint capture之前执行 forward分析，但明确没有 tracing
（`torch/_functorch/_aot_autograd/collect_metadata_analysis.py:167-242`）。

随后 AOT capture wrappers应用functionalization/effect/subclass等变换，再用 `make_fx`
捕获 joint graph
（`torch/_functorch/_aot_autograd/graph_capture.py:214-263`;
`torch/_functorch/_aot_autograd/graph_capture.py:472-536`）。

所以“metadata analysis生成 joint graph”是错误阶段图。

## 12. 对后端的承诺

规范化后端 contract需按具体入口声明，不能泛化为所有 FX：

- symbolic_trace可保留 call_module/in-place；
- export目标是更强 functional ATen/signature contract；
- AOT joint通常满足functionalization约束但可含受控copy/effect；
- post-grad图已更低层，但仍可能有extern/custom/HOP；
- GraphLowering面对缺失 lowering时还要decomposition/fallback/error。

## 源码跟读：AOT 捕获前后的规范化流水线

规范化最容易被误画成“对一张现成 Graph 顺序跑五个 pass”。AOT 的真实实现混合了
capture 前 wrapper、dispatcher 时 decomposition、capture 后 invariant check 和 runtime
post-compile wrapper。

```text
用户 flat_fn / flat_args
   │ metadata analysis
   ├─ dedupe wrapper
   ├─ synthetic-base wrapper
   ├─ create_functionalized_fn
   │
   ▼
make_fx / ProxyTensor dispatch
   ├─ decomposition table 命中则执行 decomposition callable
   └─ 否则记录原 operator
   ▼
functional FX graph
   ├─ assert invariant / DCE / recompile
   └─ compile
   ▼
runtime wrappers 恢复 duplicate、alias 与 mutation calling convention
```

### 1. Decomposition 在 dispatcher 截获时决定“记录原 op 还是执行替代函数”

ProxyTensor 提供 `decompose(table)` context manager，本质是临时启用当前 proxy mode 的
decomposition table（`torch/fx/experimental/proxy_tensor.py:173-181`）。

operator dispatch handler 在创建普通 proxy Node 之前调用 `maybe_handle_decomp`；若返回值
不是 `NotImplemented`，直接返回 decomposition 执行结果。未命中时，post-dispatch 路径
还可能调用 operator 自身 Composite decomposition
（`torch/fx/experimental/proxy_tensor.py:1280-1308`;
`torch/fx/experimental/proxy_tensor.py:1310-1324`）。

因此 decomposition 后图中出现的是 decomposition callable **执行时再次触发的算子**
Nodes，不是先创建 composite Node 再原地展开它。这个差异决定：

- 原 composite Node 可能根本不进入结果图；
- decomposition 的 Python/dispatcher 调用成本属于捕获过程；
- 输出 Node 数由替代函数实际执行路径决定；
- metadata/provenance 是否传播要看 tracing context，而不是事后 `replace_all_uses_with`。

### 2. Functionalization 也发生在“给 make_fx 的函数”这一层

AOT 的 `_prepare_graph_capture_tracing` 在 functionalization 开启时调用
`create_functionalized_fn`，把待追踪函数和输入描述一起改写
（`torch/_functorch/_aot_autograd/graph_capture.py:214-238`）。

该 wrapper 的 `_functionalized_f_helper` 先把输入递归包装成 functional tensors，再执行
原/joint function；它同时利用 `ViewAndMutationMeta` 判断哪些输入在图中发生 mutation
（`torch/_functorch/_aot_autograd/graph_capture_wrappers.py:851-885`;
`torch/_functorch/_aot_autograd/graph_capture_wrappers.py:887-905`）。

这就是 functionalization 不是字符串替换的源码证据：它在执行语义层观察 view/mutation，
让 make_fx 记录处理后的 functional operations，并生成 updated-input outputs/描述。

### 3. Duplicate input 与 storage-alias input 在 capture 前用不同 wrapper 处理

`AOTDedupeWrapper` 保存 `keep_arg_mask` 和 `add_dupe_map`：pre-compile 可删掉重复 argument，
post-compile/runtime 再按映射恢复原位置
（`torch/_functorch/_aot_autograd/runtime_wrappers.py:1586-1608`）。其策略还读取 input
mutation metadata，因为 mutated duplicate 不能和只读 duplicate 用同一简化处理
（`torch/_functorch/_aot_autograd/runtime_wrappers.py:1609-1625`）。

`AOTSyntheticBaseWrapper` 处理的不是相同 Python object，而是共享 storage 的 view inputs。
它调用 `merge_view_inputs` 得到 synthetic-base arguments 与重建信息；无需合并时直接关闭
post-compile 路径
（`torch/_functorch/_aot_autograd/runtime_wrappers.py:1844-1880`）。

所以：

```text
f(x, x)                    object identity duplicate → dedupe map
f(base[:], base[1:])       distinct Tensor, shared storage → synthetic base group
```

若用 Python identity 去处理第二类，会漏掉 alias；若把所有 storage-sharing inputs 当重复
参数删除，又会丢失各 view 的 size/stride/offset。

### 4. 捕获后 invariant check 说明 canonical contract 是显式建立的

AOT 在 functionalization enabled path 对 graph 调用 `assert_functional_graph`，分配
epilogue copy stream、包裹 sync control deps、DCE/recompile，再次检查 copy 数未变化
（`torch/_functorch/_aot_autograd/graph_capture.py:380-395`）。

这段顺序有两个重要含义：

- “functional graph”允许 contract 规定的少量 tail copy，不等于源码中完全没有 mutation；
- DCE 与 recompile 是该 capture stage 显式执行的收尾，不是 functionalization 或
  Graph edit API 自动附带的通用动作。

### 5. 为什么这些机制不能合并成一个 Graph-to-Graph pass

| 机制 | 必须看到的信息 | 只拿捕获后 Graph 是否足够 |
|---|---|---|
| schema normalization | operator schema + raw call spelling | 通常足够 |
| decomposition | dispatcher op、decomposition callable、执行 mode | 不总是；常在记录前发生 |
| functionalization | runtime alias/mutation dispatch | 不足 |
| dedupe | 原 calling convention 的 object identity | 捕获后可能已丢失 |
| synthetic base | 输入 storage/view relation | 仅 Node identity 不足 |
| canonical graph pass | 已建立的 graph invariant | 足够，但必须声明 stage |

把它们全放到图后处理会迫使 Graph metadata 重建捕获前的 object/storage/dispatch 信息；
把所有 canonical pass 又塞进 tracing wrapper，则会让 pass 难以独立验证和排序。当前分层
让每种机制在信息最完整的阶段工作，代价是 stage contract 和 wrapper ABI 必须被明确记录。

### 源码边界

以上链路解释 decomposition/functionalization/dedupe/synthetic-base 的发生位置与数据依赖。
它不意味着每个 AOT invocation 都启用所有 wrapper：inference/autograd、export、
disable_functionalization、subclass、mutation 类型等配置会选择不同分支。阅读具体 dump 时
必须以当次 `AOTConfig` 与 `ViewAndMutationMeta` 为准。

## 13. 复杂度与图规模

令输入图有 `V/E`，decomposition 后新增 `ΔV/ΔE`：

- schema/canonical spelling 的结构扫描通常与访问的 nodes/arguments 相关；
- decomposition 的输出规模决定后续所有 pass 成本；单个高层 op 可展开成多个 primitive，
  因而成本至少包含 `V+ΔV`，不能只报输入 `V`；
- functionalization 还维护 view replay、alias/mutation state 与 pytree I/O；meta kernel 和
  dispatcher 调用是外生成本；
- CSE/constant folding 的成本取决于 key/hash、常量 payload 及实际常量计算，不能只用
  graph edges 给出严格界；
- synthetic-base grouping 若需要比较大量输入 alias relation，候选数量与输入/alias 组规模
  也应单列。

常见 bounded-arity、有限 decomposition 表下结构遍历近输出图线性；没有算子分布、展开量
与 alias 分布时，期望复杂度未定义。

## 14. 已验证 Lab

从知识库根目录运行：

```powershell
python -B tools\labs_torch_compile\part2_capture_frontends.py
python -B tools\labs_torch_compile\part2_normalization.py
python -B tools\labs_torch_compile\series_artifact_bundle.py `
  --output-dir tools\labs_torch_compile\artifacts\end_to_end
```

`part2_capture_frontends.py`负责 symbolic_trace 与 make_fx 的 op 粒度对照；原版正文把这项
能力误写给 `part2_normalization.py`，现已纠正。后者真正验证：

- `torch.func.functionalize`将一个可支持的 in-place/view函数转换为 functional调用；
- 给 `make_fx`传 decomposition table前后 Node targets变化。

Lab会同时比较输出数值和输入 mutation，防止只看 graph string误判等价。

实测摘要：

```text
original_has_inplace=True
functional_has_outplace_add=True
functional_output_matches=True
functional_input_semantics_match=True
plain_has_silu=True
decomposed_has_silu=False
```

functionalized callable在边界仍恢复原函数对输入的可观察mutation，这正是“图内functional
关系”与“用户calling convention”需要分层的例子。

正例是 `silu` decomposition 与 functionalized output；错误/边界输入是对 view 做 in-place
mutation，脚本必须同时比较 output 和 input mutation，不能只 grep graph string。持久
artifact 位于 `tools/labs_torch_compile/artifacts/end_to_end/functional_aten.py`，自动合同
`EffectAndFunctionalizationContractTest`对上述等价性做 assertion。环境与命令见
[`tools/labs_torch_compile/README.md`](tools/labs_torch_compile/README.md)。

## 15. 选择机制

想新增优化时问：

1. 问题是 call spelling、operator set、mutation，还是代数 pattern？
2. 规则需要高层语义还是低层 primitive？
3. backend已有专用 op/template吗？
4. decomposition会增加多少 Node、丢失什么 metadata？
5. functionalization/runtime wrapper能否恢复用户 alias/mutation？
6. dynamic symbolic关系在何层可证明？
7. 该变换是否幂等，和邻近 pass会不会来回改写？

## 学习顺序

- 上一篇：[[07_graph_capture_frontends_and_tracing]]
- 下一篇：[[09_aotautograd_joint_forward_backward_graphs]]

## Related Pages

- [[19_torch_compile_end_to_end/00_pytorch_graph_series_index]]
- [[07_graph_capture_frontends_and_tracing]]
- [[05_graph_effects_alias_mutation_and_order]]
- [[09_aotautograd_joint_forward_backward_graphs]]
- [[15_graph_pass_pipeline_ordering_and_fixpoint]]
- [[decomposition_passes_guide]]
