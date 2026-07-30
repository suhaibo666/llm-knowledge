# 16 · Graph Rewrite 的合法性、验证与复杂度

> 前置：[[fx_graph_editing_primitives_and_invariants_analysis]]、[[graph_pass_pipeline_ordering_and_fixpoint_analysis]]
> 当前实现基线：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`
> Lab 环境：PyTorch `2.9.1+cpu`
> 最后更新：2026-07-28

## 1. 结构命中只是第一关

一个rewrite正确需同时满足：

```text
结构命中
∧ shape/dtype/device/layout合法
∧ alias/mutation/effect等价
∧ autograd等价
∧ dynamic guards充分
∧ 数值误差可接受
∧ stage invariants保持
```

任何一项失败都应保留原图。

## 2. Shape与broadcast

`add(matmul(x,w), b)`看似可融合，但需确认：

- matmul rank与batch semantics；
- b broadcast axes；
- symbolic equality/range可证明；
- empty/zero-size；
- output shape；
- replacement是否支持dynamic。

无法compile-time证明时，要么extra_check拒绝，要么由正确层产生runtime guard；不能用trace
hint假装恒真。

## 3. Dtype与数值

检查：

- type promotion；
- accumulation dtype；
- integer overflow；
- complex/conjugate；
- autocast；
- NaN/Inf/signed zero；
- reduction order与associativity；
- fast-math容忍度。

bitwise equality、`allclose`和训练收敛是不同validation等级。

## 4. Device、layout、stride与alignment

同shape Tensor可有不同：

- device；
- contiguous/channels-last；
- arbitrary stride/offset；
- overlapping storage；
- alignment；
- pinned memory。

replacement若隐含contiguous，必须插入合法copy/constraint或拒绝。Inductor GraphLowering在
调用lowering前可施加layout constraints；fallback backward还可能保守require contiguous
（`torch/_inductor/graph.py:1435-1465`;
`torch/_inductor/graph.py:1478-1515`）。

## 5. Alias、mutation与effect

验证：

- output alias输入吗；
- mutation目标/次数/顺序；
- view metadata；
- version counter；
- RNG/collective/I/O；
- stream/mempool；
- input/output escape。

只比较returned tensor数值无法发现alias contract改变。测试要比较data_ptr/storage relation与
mutation后的可观察状态。

### 三条可复用的安全判据

跨项目（upstream/torch_npu/vLLM/SGLang）反复独立收敛出同样三条判据，说明它们不是某个
实现的偶然选择，而是 alias/mutation 安全改写的通用底线：

1. **算子类别不变式**：只在共享某条不变式的一类算子内改写——例如 view 类＝扁平序/元素数
   不变，masked 类＝掩码互补，bool 索引＝按位选择。绝不跨类别改写（不碰 permute、真
   broadcast 这些会重排元素的算子）。这是"只改元数据、不重算数值"这条捷径成立的根据；
   一旦目标算子改变元素排布或数值语义，同一套"只改 meta"的改写就不再等价。
2. **边局部改写 + 纯函数前提**：post-functionalization 的图是 SSA 式纯算子图，改写只需
   动本节点的入边（`replace_input_with` 一类 API），对 DAG 的其余扇出天然安全——因为
   每个 consumer 各自持有自己的 `args` 引用，改一条边不会波及其他 consumer。
3. **单用户门槛判据**：改写只动"自己的入边"（换掉自己读哪个值）不需要检查 `users` 数；
   但如果改写要"吃掉/改动前驱节点本身"（把前驱内联进自己或删除前驱），就必须先确认
   前驱只有自己这一个 user，否则会破坏其他 consumer 仍需要的语义。这条判据把"何时必须
   查 `len(node.users)`"从直觉变成可检查的结构条件。

## 6. Autograd

forward等价不保证gradient等价。至少覆盖：

- first-order grads；
- multiple outputs/tangents；
- None/non-differentiable；
- gradgrad；
- input mutation与leaf rules；
- custom autograd.Function；
- saved tensor/version behavior。

可用eager differential、`gradcheck`、`gradgradcheck`，并比较compiled fw/bw。

## 7. Fake/meta checks与runtime guards

FakeTensor可低成本验证shape/dtype/device传播；ShapeEnv可证明symbolic predicates。
但Fake execution不验证：

- real numerical accuracy；
- backend kernel race；
- alias physical behavior；
- actual memory alignment；
- performance。

它是早期filter，不是最终proof。

## 8. Differential testing矩阵

维度至少包含：

- static/dynamic shapes；
- rank边界；
- empty/size-1/非整除；
- contiguous/transposed/sliced；
- dtype与autocast；
- CPU/GPU或目标backend；
- requires_grad组合；
- alias/non-alias inputs；
- mutation/no mutation；
- NaN/Inf/extreme values；
- deterministic/random seeds；
- **收益**：pass 开/关 A/B 对比——命中计数、kernel 数与端到端性能。前八项回答"改写是否
  正确"，这一项回答"改写是否值得"：结构性收益（少 N 个 kernel、少一次拷贝、转零拷贝）
  不是加速比，几乎没有 pass 自带 benchmark/counter，上线前必须用"开/关该 pass"实测，
  不能只凭"命中过"就假定有收益（对照 §10 性能值得性）。

## 9. 失败原子性

`extra_check`应在graph mutation前完成。若handler内部可能失败：

- 在临时Graph构造replacement；
- 全部成功后copy/commit；
- 或显式rollback；
- 永远不要捕获异常后留下半连接Node。

普通MatchContext rollback只覆盖pattern binding，不回滚handler graph surgery。

## 10. 性能值得性

合法rewrite也可能更慢。评估：

- kernel launch减少；
- materialization bytes；
- recompute FLOPs；
- register/shared memory压力；
- template/extern选择；
- compile/autotune cost；
- dynamic shape reuse；
- peak memory。

Scheduler fusion使用candidate/legality/score与部分benchmark，不是一个通用
`memory_saved > threshold && registers < limit`公式
（`torch/_inductor/scheduler.py:6395-6462`;
`torch/_inductor/scheduler.py:6830-6884`;
`torch/_inductor/scheduler.py:8610-8675`）。

## 11. 全链路复杂度

设FX `V/E`、候选 `C`、candidate `v`的 root bucket 为 `B(v)`、pattern大小 `K(p)`、
multi-output/user anchor探索 `A(p,v)`、replacement大小 `Q`、被重连 uses 为 `U`。
再设 `M(v)` 为首次mutation-region扫描对Node `v`的判定成本，`H`为非
`call_function` root lookup扫描相应op桶的总成本，`Lcmp`为一次Node sort-key字典序比较
扫描的key元素数上界，`Lkey(v)`为移动Node时复制/重建sort key的长度成本：

| 环节 | 典型结构成本 |
|---|---|
| 构图 | `O(V+E)`加实际trace computation |
| candidate lookup/sort | `O(Σv M(v)+H+C+C log C·Lcmp)`；已有mutation-region metadata时首项省略 |
| pattern match | `Σv Σp∈B(v) O(K(p)+A(p,v))` |
| replacement | `O(Q+U+local erase)`加handler/trace外生成本 |
| DCE | `O(V+E)` |
| stable topo | `O(V+E+Σv d(v)²+Σmoved Lkey(v))`；arity与sort-key长度都有界时才常见近线性 |
| lint | `O(V+E+target lookup)`量级 |
| recompile | `O(V + source size)` |
| partition extraction | `O(V+E)`，min-cut另计flow求解 |
| repeated pipeline | 乘实际round数 |

### 不能忽略的最坏项

- transitive ancestor sets可 `Θ(V²)`；
- matcher首次apply可扫描全图计算mutation region；非 `call_function` 的带target查询还会
  按注册root key过滤相应op桶，不能一概写成 `O(C)`候选查找
  （`torch/_inductor/pattern_matcher.py:2416-2425`、
  `torch/_inductor/pattern_matcher.py:2619-2645`、`torch/fx/graph.py:1360-1393`）；
- 反复在相邻 Node 间插入会增长层级式 sort key；若 key 长度不设界，candidate sort
  不能简写成严格 `O(C log C)`，stable topo移动Node时也会复制/重建key
  （`torch/fx/node.py:441-450`、`torch/csrc/fx/node.cpp:429-449`、
  `torch/csrc/fx/node.cpp:454-463`）；
- alias merge/all-pairs fusion可超线性；
- symbolic algebra不由V线性控制；
- autotune wall time由compile+benchmark candidates主导；
- nested graph需对所有children求和；
- trace本身执行的tensor计算可能远大于结构遍历。

`extra_check`、custom handler、FakeTensor/meta kernel 与 traced replacement 是任意外部
计算，不能塞进只依赖 `V/E`的严格上界。没有 bucket/fan-out/round 分布时，期望复杂度
未定义；应区分 bounded-arity常见成本与参数化严格界。

## 12. 验证层级

| 等级 | 证据 |
|---|---|
| L0 | graph lint/ownership/topology |
| L1 | Fake/meta/signature |
| L2 | forward differential |
| L3 | gradient/gradgrad |
| L4 | alias/mutation/effect |
| L5 | dynamic/randomized matrix |
| L6 | backend generated artifacts |
| L7 | benchmark与peak memory |

发布authoritative Lab应说明覆盖到哪一级。

## 源码跟读：合法性检查究竟在哪些层发生

不存在一个 `rewrite_is_legal(old, new)` 全能函数。当前实现把合法性分布在 matcher structural
checks、stage barrier、shape-specific retrace、lowering layout constraints 和最终测试中。

### 1. Matcher apply 先拒绝跨 mutation/stream/mempool 的结构命中

`PatternMatcherPass.apply` 得到 `Match` 后，会检查 matched Nodes 的 mutation region ID
是否唯一；随后把每个 Node 的 stream、mempool、mempool device 组成 context tuple，要求
全 match 一致
（`torch/_inductor/pattern_matcher.py:2653-2681`）。

只有这些 stage barrier 通过，才在 `guard_or_false` 下执行 entry `extra_check`，然后进入
Entry.apply；应用后还按 entry 类型刷新 mutation-region 信息
（`torch/_inductor/pattern_matcher.py:2682-2710`）。

这层能证明：

- replacement 没跨越已知 mutation barrier；
- matched operations 位于兼容 stream/mempool context；
- rule 自己声明的 extra predicate 成立。

它不能证明：

- 任意 storage alias 等价；
- 数值/梯度等价；
- 未被 region metadata 建模的 effect；
- backend kernel 在目标设备正确。

### 2. Traced replacement 会用真实 match 的 fake values 再 trace 一次

`register_replacement` 的初始 search pattern可能忽略被烧进 generic pattern 的 int/shape。
源码中的 `check_fn` 从 match kwargs 取 `meta["val"]` 等抽象值，检测 FakeTensorMode，并按
真实 match 重建 trace args
（`torch/_inductor/pattern_matcher.py:1828-1854`;
`torch/_inductor/pattern_matcher.py:1878-1903`）。

随后它对 search function 做 specific trace，重新转为 PatternExpr，并在实际 output root
再次 match。只有 specific match 与 user `extra_check` 都成功，才 trace replacement
function 并把结果放进 `match.replacement_graph`
（`torch/_inductor/pattern_matcher.py:1983-2002`;
`torch/_inductor/pattern_matcher.py:2012-2033`）。

这形成两级筛选：

```text
generic AST
  快速找结构候选
      │
      ▼
shape/type-specific search retrace
  确认当前 fake values 下搜索图仍同构
      │ extra_check
      ▼
replacement retrace
  构造当前命中对应的 replacement Graph
```

它比只用 generic pattern 安全，但仍是抽象执行/有限规则检查。若 `extra_check` 没覆盖
broadcast、alias 或 dynamic guard，二次 trace 不会自动补出完整证明。

### 3. Lowering 还会施加 layout constraint，说明 FX rewrite 通过不代表后端输入已合法

GraphLowering 对 fallback op 选择 layout constraint。backward 内置 ATen fallback 若没有
明确 tag，会保守要求 contiguous，源码说明否则某些 eager kernels 可能静默产生 accuracy
问题（`torch/_inductor/graph.py:1435-1464`）。

对已注册 lowering，`maybe_layout_constraints(target)` 可能：

- 用 Node `meta["eager_input_vals"]` 的 fake args 约束实际 IR args；
- 先按 operator schema normalize 三套参数；
- 或调用 target-specific layout constraint
  （`torch/_inductor/graph.py:1478-1515`）。

因此 pass 的 legality 分成至少两层：

```text
FX 层：
  数学/shape/alias/effect 上能否替换成 target B

Lowering 层：
  B 的实现需要何种 layout/stride，是否插 copy/realize/fallback
```

FX pass 可以选择更强 predicate提前拒绝，也可以依赖明确的 lowering constraint完成合法
layout conversion；但必须知道该 target 的后端 contract，不能默认相同 shape 就可直接调用。

### 4. Failure atomicity 必须位于第一次 Graph 写入之前

上述 matcher barrier 与 traced replacement `check_fn` 都发生在 Entry.apply 前，这是安全的
拒绝点。进入 GraphPatternEntry 自定义 handler 后，基础 matcher 不提供 Graph rollback。

可靠实现有两种：

```text
方案 A：纯分析
  先计算全部 predicate/guards/replacement plan
  → 全部通过后原地 create/reconnect/erase

方案 B：shadow GraphModule
  copy graph/state/signature
  → 在 shadow 上改写和验证
  → 成功后整体提交；失败丢弃
```

try/except 后删除“看起来是新建的 Nodes”并不稳健，因为 handler 可能已重连 old/new users、
改 companion metadata 或 owning module state。事务边界应覆盖所有被修改的数据结构。

### 5. 为什么 forward differential 不能替代 alias/grad/effect proof

两个 callable 可以返回数值相同 Tensor，但：

- 一个返回 input view，另一个返回 fresh allocation；
- 一个更新 input/version counter，另一个不更新；
- 一个消耗 RNG/执行 collective，另一个删除 effect；
- forward 相同，backward accumulation dtype/order 不同。

因此验证层级必须对应可观察合同：

| 合同 | 需要观测 |
|---|---|
| structure | lint、ownership、signature |
| abstract value | fake shape/dtype/device/layout |
| forward | real outputs、exception behavior |
| autograd | gradients、None slots、gradgrad、saved-version behavior |
| alias/mutation | storage/data_ptr/base、输入前后状态、version |
| effect | RNG state、collective/order、I/O/custom state |
| dynamic | ranges、guard fail/recompile、随机 shape matrix |
| backend | 真实生成、编译、加载、目标设备执行 |

一个实验覆盖 L0–L4，就应明确留下 L5 dynamic matrix、L6 native execution、L7 performance
为未关闭项；不能用代码生成 artifact 替代 kernel execution。

### 6. 全链路复杂度为何必须保留外生成本

源码可参数化结构成本，但以下回调没有只依赖 Graph `V/E` 的上界：

- `extra_check(match)` 可运行任意 Python/symbolic reasoning；
- specific search/replacement `trace_fn` 会执行 fake/meta operator；
- layout constraint 可 normalize 大型 pytree并插入 copy/realize；
- differential/gradcheck/randomized matrix 会执行真实 Tensor computation；
- backend compile/autotune 由候选 kernel、编译器和设备测量主导。

因此总成本应写成：

```text
T = T_graph_structure
  + Σ T_extra_check
  + Σ T_specific_trace
  + Σ T_replacement_trace
  + T_stage_cleanup
  + T_validation_matrix
  + T_backend_compile_and_benchmark
```

`O(V+E)`只可能描述其中某些结构扫描，不能代表“应用一个 pass 的端到端成本”。

### 源码边界

当前 matcher/GraphLowering 源码证明若干具体 guardrail，不构成任意 rewrite 的形式化证明。
authoritative claim 必须逐项说明：依赖哪个 stage invariant、哪个源码检查、哪个运行时验证，
以及仍未覆盖哪些输入域/设备/数值模式。

## 13. 已验证 Lab

从知识库根目录运行：

```powershell
python -B tools\labs_torch_compile\part3_end_to_end_pass.py `
  --output-dir tools\labs_torch_compile\artifacts\part3
```

pass只在可证明的窄域做：

```text
add(matmul(x, weight), bias) → addmm(bias, x, weight)
```

首版合法域是 rank-2、相同 dtype、`bias.shape`严格等于 matmul 输出 shape；不支持的 `(5,)`
broadcast bias 是错误/边界例，必须拒绝。分析和编辑在 deep-copied shadow GraphModule
完成，只有合法 rewrite 才把 fresh graph commit 回 caller，因此 reject 后 code 与 meta
都不变。

2026-07-26 实测并 assertion：

```text
legal_rewrite_applied=True
legal_has_addmm=True
illegal_broadcast_rewrite_applied=False
illegal_graph_unchanged=True
failure_atomicity_matches=True
forward_matches=True
gradient_matches=True
gradcheck_matches=True
shape_matches=True
alias_contract_matches=True
mutation_contract_matches=True
second_run_modified=False
second_run_code_unchanged=True
```

alias 与 mutation 是 eager-vs-rewrite relation/snapshot 差分，不再只是断言 rewritten output
“看起来不 alias”。`gradcheck`用 float64 输入。当前 legality predicate消费 real
`ShapeProp.tensor_meta`；本 Lab 对所列静态案例覆盖到 L0–L4。

当前 predicate 不是 dynamic FakeTensor/guard proof，也未覆盖 GPU/device-specific
accumulation、effectful inputs 或随机 shape 矩阵，因此不声称完成 L5–L7。

持久 artifact 位于 `tools/labs_torch_compile/artifacts/part3/`，包括 before/after/rejected graph、
`results.json`、环境与 manifest。自动合同 `EndToEndPassContractTest`失败会返回非零；
完整命令与证据等级见 [`tools/labs_torch_compile/README.md`](tools/labs_torch_compile/README.md)。

## 14. Review模板

```text
Stage:
Search pattern:
Replacement:
Input invariants:
Output/signature invariants:
Shape/dtype/device/layout proof:
Alias/mutation/effect proof:
Autograd proof:
Dynamic guard strategy:
Failure atomicity:
Complexity:
Test matrix:
Observed artifacts/performance:
```

## 学习顺序

- 上一篇：[[graph_pass_pipeline_ordering_and_fixpoint_analysis]]
- 下一篇：[[17_fx_lowering_to_inductor_ir]]

## Related Pages

- [[19_torch_compile_end_to_end/00_pytorch_graph_series_index]]
- [[fx_graph_editing_primitives_and_invariants_analysis]]
- [[pattern_expression_and_matcher_engine_analysis]]
- [[dead_code_topology_and_effect_order_analysis]]
- [[graph_pass_pipeline_ordering_and_fixpoint_analysis]]
- [[20_scheduler_dependency_graph_fusion_and_ordering]]
