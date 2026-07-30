# 15 · Graph Pass 流水线、顺序与 Fixed Point

> 前置：[[08_graph_normalization_decomposition_and_functionalization]]、[[14_dead_code_topology_and_effect_order]]
> 当前实现基线：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`
> Lab 环境：PyTorch `2.9.1+cpu`
> 最后更新：2026-07-28

## 1. Pass位置由可用不变量决定

| stage | 常见可见形态 | 适合 |
|---|---|---|
| Dynamo | Python捕获、高层ops、guards | graph break/specialization附近优化 |
| pre-grad | 高层FX、未显式backward | module/高层canonicalization |
| AOT decomposition | tracing operator set | composite→primitive |
| joint | forward+gradient同图 | save/recompute前联合优化 |
| post-grad | 独立functional ATen fw/bw | 低层pattern/device rewrite |
| lowering | target→Inductor IR | implementation/layout/fallback |
| Scheduler | realized buffer deps | fusion/reorder/liveness |
| codegen | backend kernel/wrapper | tiling/indexing/launch |

“能在某stage写出来”不等于“应放在那里”。

## 2. 三个Inductor FX driver不同

### pre-grad

可运行built-in/custom passes、configured matcher counter、stable sort、quant lifting、lint、
recompile（`torch/_inductor/fx_passes/pre_grad.py:336-433`）。

### joint

有ordered joint pattern lists、constant folding等，并根据match/change条件做stage cleanup
（`torch/_inductor/fx_passes/joint_graph.py:700-772`）。

### post-grad

有多轮pattern、device/communication/mutation-tail相关顺序与inference-aware hooks
（`torch/_inductor/fx_passes/post_grad.py:165-180`;
`torch/_inductor/fx_passes/post_grad.py:227-303`;
`torch/_inductor/fx_passes/post_grad.py:385-474`）。最后一段才覆盖 late
reinplace/alias 修正与 final lint/recompile；只读 main-pattern 范围不足以证明 mutation tail。

不能用一个“所有pass都pattern match→DCE→sort”图替代。

## 3. 同一语义规则不能原样跨 stage 搬运

以同一个代数变换为例：

```text
add(matmul(x, weight), bias) → addmm(bias, x, weight)
```

规则的数学含义没有变，但可匹配的 target spelling 由输入图契约决定：

| Lab 图契约 | 捕获方式 | 改写前 target | 改写后 target |
|---|---|---|---|
| front-end-like 高层 FX | `symbolic_trace` | `torch.matmul`、`torch.add` | `torch.addmm` |
| functional-ATen-like FX | `make_fx` | `aten.mm.default`、`aten.add.Tensor` | `aten.addmm.default` |

`symbolic_trace`通过 `Tracer.trace`记录执行并组装 `GraphModule`
（`torch/fx/_symbolic_trace.py:1361-1421`）；`make_fx`则返回一个用真实、Fake 或 symbolic
Tensor 执行后产生 FX 图的包装器
（`torch/fx/experimental/proxy_tensor.py:3312-3383`）。两种捕获入口本身已经足以产生不同
target vocabulary。

这正是 stage placement 必须显式声明输入不变量的原因。固定源码还直接警告 pre-grad IR
尚未 functionalize/normalize，规则必须自行处理 alias、mutation 与不同参数 schema
（`torch/_inductor/fx_passes/pre_grad.py:336-351`）；post-grad driver 接收的 IR 才声明为
normalized and functionalized
（`torch/_inductor/fx_passes/post_grad.py:165-171`）。

### 本页 Lab 的证据边界

`part3_passes.py`对上述两个 FX 图应用**同一语义、两份 stage-specific target contract**：

1. 用错误 stage 的 contract 时必须命中 `0` 次；
2. 用正确 contract 时各命中 `1` 次；
3. 两张改写图都与 eager 结果数值一致；
4. 第二次应用各命中 `0` 次，证明该规则在这两个受限输入上幂等。

这是可执行的 **stage-contract 模拟实验**，不是实际 `torch.compile` hook：

- 没有调用 `torch.compile`；
- 没有把规则注册进 `pre_grad_passes`或 `post_grad_passes`；
- 没有经过 AOTAutograd、functionalization 或 Inductor 的真实 pass ordering。

实验 matcher 刻意限制为无 kwargs 的二元 `matmul`和二维数值样例，用于隔离 target
vocabulary 问题。运行结果覆盖“pattern spelling 与输入图契约绑定”和“错误 contract
不命中”。

该模拟实验没有覆盖 broadcast、dynamic shape、alias 或 mutation，不能单独证明规则已经
安全接入真实 pre-grad/post-grad 流水线。真实 hook 集成还必须另外验证注册顺序、dynamic
shape、alias/mutation 与阶段 cleanup。

上述边界只描述 `part3_passes.py`。本页后面的
`part3_real_stage_hooks.py`已经通过 `torch._inductor.config` 的真实 custom-pass 入口补上
placement 证据。

该真实 hook Lab 仍未覆盖 shape、alias、mutation 等更广 legality 维度；这些边界留给第
16 篇的受限合同与后续目标设备验证。

## 4. Registration order

同root bucket中rules按registration order尝试。若先应用宽pattern，可能erase root，使后续
更专用rule无机会。优先级需要：

- 更具体rule先；
- canonicalization先于依赖canonical form的fusion；
- destructive rewrite后重新建立必要metadata；
- mutation/effect tail pass最后。

## 5. Single round、bounded repeat、fixed point

### single round

PatternMatcherPass一次snapshot candidates并逆序遍历；new roots不重访
（`torch/_inductor/pattern_matcher.py:2627-2645`）。

### bounded repeat

driver显式运行N次，例如pre-grad counter。

### fixed point

重复直到所有pass `modified=False`，但通常还有max steps防止不收敛。
FX `PassManager`默认 `steps=1`；只在configured steps内根据modified提前停止
（`torch/fx/passes/infra/pass_manager.py:154-195`;
`torch/fx/passes/infra/pass_manager.py:254-317`）。

`PassResult.modified=True`本身不触发无限迭代。

## 6. 幂等性

理想canonicalization：

```text
P(P(graph)) == P(graph)
```

fusion/replacement也应避免匹配自己的输出，除非driver有明确fixed-point语义。

非幂等风险：

- A→B与另一个pass B→A；
- 每轮插入新cast/view；
- name/meta变化让同语义Node重复生成；
- replacement root仍匹配search pattern。

## 7. Candidate lifecycle

一次Pattern apply：

1. 按registered root keys查询Graph side table；
2. snapshot与逆序sort；
3. 遍历candidate bucket；
4. match、safety、extra_check；
5. apply；
6. 不加入新candidates。

这使单轮复杂度和行为可控。需要新结果继续优化时，driver必须再调用一轮。

## 8. Pass冲突

常见冲突：

- decomposition暴露pattern，但也摧毁专用op；
- CSE合并sharing，改变 `_users`约束；
- DCE先删diagnostic/effect被误判Node；
- layout pass插copy使fusion pattern断开；
- partition前rewrite改变save cost；
- post-grad reinplace恢复mutation，后续rules需effect-aware。

解决方式是明确输入/输出invariant与ordering dependency，而不是只靠import顺序。

## 9. Recursion

stage driver可能递归处理HOP/subgraphs；单个PatternMatcherPass不自动代表整个nested tree。
每个child需独立Graph ownership、cleanup、lint/recompile。

## 10. Observer、counter与debug

可靠pass应记录：

- attempted/matched/applied；
- rejection原因；
- before/after Node数；
- stage与graph id；
- pattern name；
- generated artifact；
- compile time。

counter不是generic `changed` bit；例如joint path的count可能仅统计某类match，不能据此推断
其他custom pass是否修改图。

## 11. 选择stage的决策树

1. 是否需要Python/module语义？→ Dynamo/pre-grad。
2. 是否需要forward与backward同时可见？→ joint。
3. 是否要求functional ATen且独立fw/bw？→ post-grad。
4. 是否产生lazy loop/layout/extern implementation？→ lowering。
5. 是否依赖buffer reads/writes/fusion group？→ Scheduler。
6. 是否只与tile/index/mask/launch有关？→ codegen。

若多个stage都可实现，选择invariant最强、规则最局部、验证最容易的一层。

## 源码跟读：三个 driver 怎样把 stage contract 写进 pass 顺序

### 1. Pre-grad 源码首先警告“尚未 functional/normalized”

`pre_grad_passes` 的 docstring 明确要求规则自行处理 alias、mutation 和所有参数 schema，并
建议优先考虑 functionalization/normalization 后的 joint/post-grad
（`torch/_inductor/fx_passes/pre_grad.py:336-351`）。

实际顺序中，非 pre-dispatch 路径先做 numpy compatibility、`fuse_fx`，再强制
normalization pattern 优先；之后是 group batch fusion 与 configured pattern passes。
每个配置项的 `counter` 决定同一 matcher pass 运行次数
（`torch/_inductor/fx_passes/pre_grad.py:353-383`;
`torch/_inductor/fx_passes/pre_grad.py:384-402`）。

最后 custom passes 才运行，随后统一 stable topological sort、quant lift、lint、recompile
（`torch/_inductor/fx_passes/pre_grad.py:421-433`）。

这给出两个实现事实：

- counter 是显式 bounded repeat，不是 matcher 自己 fixed point；
- pre-grad cleanup 是 driver 尾声，不是每个 PatternEntry 自动执行。

### 2. Joint driver 把“必须先 canonicalize”和“有 change 才 cleanup”写进控制流

joint driver 一进入就调用 `canonicalize_aten_ir_passes`，注释标记“must occur before other
passes”；随后依次是 custom pre、noop removal、可选 constant folding、early patterns
（`torch/_inductor/fx_passes/joint_graph.py:700-733`）。

再往后是 AutoChunker、ordered `pass_patterns`、RNG replacement 与 custom post
（`torch/_inductor/fx_passes/joint_graph.py:735-766`）。只有 `count` 非零，才 stable sort、
lint、recompile（`torch/_inductor/fx_passes/joint_graph.py:768-772`）。

这里的 `count` 是 driver 自己累积的变化信号。新增 custom pass 若无条件 `count += 1`，
即使它内部没改图也会触发 cleanup；反过来，若某修改路径未正确计数，可能跳过预期尾声。
所以不能把任意 counter 当成精确“实际修改 Node 数”。

### 3. Post-grad 的输入契约最强，但尾部又主动重新引入 mutation

`post_grad_passes` 明确声明会分别在 fw、bw 调用，输入 IR 已 normalized/functionalized。
开头可做 DCE/locality，随后执行 custom pre
（`torch/_inductor/fx_passes/post_grad.py:165-199`）。

主 pattern 区先 group fusion、noop/assert removal、ordered pass lists，再执行配置化
post-grad patterns；custom post 在主 patterns 之后
（`torch/_inductor/fx_passes/post_grad.py:227-267`;
`torch/_inductor/fx_passes/post_grad.py:281-289`）。

之后 driver 仍有 random ordering、stable sort、constructor movement、FakeTensor metadata
增量更新等步骤
（`torch/_inductor/fx_passes/post_grad.py:292-303`）。更晚的 distributed/overlap 变换完成
后，源码明确要求 reinplace 等 mutation-introducing passes 保持最后，之后做 alias 修正与
HOP decomposition，最终 recompile + lint
（`torch/_inductor/fx_passes/post_grad.py:449-474`）。

所以“post-grad 是 functional graph”是该 driver **入口/大部分 pass 的 invariant**，不是
最终 graph 永远无 mutation。late reinplace 依赖前面的分析结果，在结束前受控恢复更高效
的 in-place 形式。

### 4. 为什么同一个规则跨 stage 不能只改注册位置

从三个 driver 可直接推出输入条件：

```text
pre-grad:
  高层/未规范，alias+mutation+schema 由规则承担

joint:
  forward+backward 同图，canonical ATen，partition 尚未发生

post-grad:
  fw 或 bw 独立，functional/normalized 入口，late tail 可 reinplace
```

同一个 `add(matmul(...), bias)`：

- pre-grad 可能是 Python function、module 或不同 schema；
- joint/post-grad 通常是 ATen overload，但 joint 还含 forward/backward sharing 与
  partition metadata；
- post-grad 已切成一侧，规则不能再查看跨 boundary save/recompute。

因此迁移规则至少要重审 target vocabulary、users 边界、metadata、alias/effect invariant、
是否影响 partition cost，以及 cleanup/observer 归属。

### 5. `PassManager` 的 fixed-point 语义是“整组 passes 有界重复”

FX `PassManager.__call__` 先解决 ordering constraints并检查 invariants，然后最多执行
`self.steps` 轮。每轮按顺序调用所有 passes，OR 累积 `res.modified`；GraphModule 结果会
recompile，可选地每个 pass 后检查。整轮没有修改才提前停止
（`torch/fx/passes/infra/pass_manager.py:254-302`;
`torch/fx/passes/infra/pass_manager.py:312-317`）。

所以：

```text
PatternMatcherPass.apply 单轮
  candidate snapshot，一次扫描

PassManager 一轮
  pass1.apply → pass2.apply → ... → passN.apply

PassManager fixed point
  重复“整组 passes”，最多 steps 轮
```

若 pass1 产生 pass2 可消费的形态，同轮即可继续；若 pass2 产生 pass1 的形态，要下一轮。
若 A→B 与 B→A 循环且都报告 modified，只有 `steps` 上限阻止无限振荡。

### 6. 设计一个可证明收敛的 pipeline

源码只提供 bounded execution 机制，不替用户证明收敛。规则设计应给出单调 measure：

| 规则类型 | 可用 measure |
|---|---|
| canonical spelling | 非 canonical Node 数严格下降 |
| decomposition | operator level/rank 单向下降 |
| fusion | 可融合 root 数或 operation 数下降 |
| cleanup | live pure/no-user Node 数下降 |
| bounded search | driver counter/steps 显式上限 |

若没有单调 measure，就必须将其定义为 single round/bounded heuristic，并验证最终形态，不应
宣称 fixed point。

### 7. Observer 的位置也是语义

三个 driver 都用 `GraphTransformObserver` 包裹 pass，但 observer 记录的是传给它的那次
Graph/GM 变化。它不会自动把独立 capture、AOT partition 或后端 lowering 串成连续对象链。
调试 artifact 要同时记录 stage、graph identity/run id、pass name、before/after 与 cleanup，
才能解释“这次命中发生在哪张图”。

### 源码边界

上述顺序绑定当前 SHA 和配置分支；实际启用哪些 pass 由 config、device、inference/training、
distributed 选项决定。文档可以陈述 driver 中的相对顺序与 invariant，不能无条件声称每次
编译都会执行列表中的所有可选步骤。

## 12. 复杂度

不同 stage 的图规模不同，应写为：

```text
Σgraph g Σpass q (
  candidate(q,g) + match(q,g) + rewrite(q,g) + cleanup(q,g)
)
```

- fixed `k`轮对相应 stage 乘 `k`；
- until-stable `r`轮乘实际 `r`，且 `r`必须有上界或单调 measure；
- pattern candidate snapshot降低单轮重入，但重复driver仍会重新查询/sort；
- custom handler、extra check、FakeTensor trace 与 backend compile 是外生成本。

没有 pass 命中率、graph size 与 convergence rounds 分布时，期望复杂度未定义。

证明收敛常用单调measure：

- 某类Node数严格下降；
- expression canonical rank下降；
- rewrite只从高level到低level；
- bounded counter。

## 13. 已验证 Lab

从知识库根目录运行：

```powershell
python -B tools\labs_torch_compile\part3_passes.py
python -B tools\labs_torch_compile\part3_end_to_end_pass.py `
  --output-dir tools\labs_torch_compile\artifacts\part3
python -B tools\labs_torch_compile\part3_real_stage_hooks.py `
  --output-dir tools\labs_torch_compile\artifacts\part3_real_stage_hooks
python -B tools\labs_torch_compile\series_artifact_bundle.py `
  --output-dir tools\labs_torch_compile\artifacts\end_to_end
```

`part3_passes.py`构造：

- 一个single-round pass；
- PassManager `steps=1`与 `steps=4`；
- 同一 `add(matmul)`融合在 front-end-like 与 functional-ATen-like 两种 target contract
  下的正例、错 stage 反例和二次运行幂等性检查；
- 两个互逆passes形成oscillation并由max steps截断。

贯穿 rewrite 作为正例必须第二次 apply 返回 `False`且 code 不变；oscillation pair 作为错误/
边界例每轮做 `add→sub→add`，无法收敛，只能被 `steps=4`截断。stage-contract 实验把
同一语义规则分别写成高层 FX 与 functional ATen target contract；交换 contract 时均为
零命中。

这些结果说明：把 rule 平移到另一个 stage 前，必须重写 pattern/legality contract。

实测：

```text
pass_manager_steps_1_count=1
pass_manager_steps_4_count=3
front_end_like_before_targets=matmul,add
functional_aten_like_before_targets=aten.mm.default,aten.add.Tensor
front_end_like_after_targets=addmm
functional_aten_like_after_targets=aten.addmm.default
stage_target_spelling_differs=True
stage_correct_contract_rewrites=True
stage_wrong_contract_rejected=True
stage_rewrite_idempotent=True
stage_outputs_match=True
stage_contract_kind=simulated_frontend_like_vs_functional_aten
actual_torch_compile_stage_hook_executed=False
oscillation_bounded_at_four=True
oscillation_final_target_is_add=True
second_run_modified=False
second_run_code_unchanged=True
```

`CountToThree`在第三次返回 stable 后提前停止，没有机械跑满4次；oscillation pair 则跑到
显式上限。新增 stage-contract 项由 `part3_passes.py`自身做 assertions；自动合同
`EditingAndPassManagerContractTest`与 `EndToEndPassContractTest`继续覆盖既有编辑、
PassManager 与贯穿 rewrite 合同。Part III artifact 位于 `tools/labs_torch_compile/artifacts/part3/`；环境、
命令与 stage artifact 见 [`tools/labs_torch_compile/README.md`](tools/labs_torch_compile/README.md)。

真实 hook Lab 另行安装 scoped config patch：

- pre-grad 输入是 Python/operator target，`operator.matmul + operator.add`命中一次并改成
  `torch.addmm`；把 functional-ATen contract 放到此阶段时命中 0；
- post-grad 输入是 functional ATen target，`aten.mm.default + aten.add.Tensor`命中一次
  并改成 `aten.addmm.default`；把 Python/operator contract 放到此阶段时命中 0；
- 两个 hook 都由真实 `torch.compile(..., backend="inductor", fullgraph=True)` driver 调用；
- 每个 hook 在同一张已改写 Graph 上立即再运行一次相同 rewrite，第二轮命中 0 且
  `graph.python_code()`不变；这才是 pass 幂等证据。随后 compiled callable 的第二次调用
  未重新编译是独立的 cache 观察，不能替代 pass 幂等；
- forward、输入梯度和权重梯度都与 eager 一致，config 在退出后恢复。

固定源码入口见 `torch/_inductor/fx_passes/pre_grad.py:421-426`、
`torch/_inductor/fx_passes/post_grad.py:190-198`、
`torch/_inductor/fx_passes/post_grad.py:281-289`。

2026-07-27 实测 pre/post correct hits 均为 1、wrong-stage hits 均为 0、second rewrite
hits 均为 0；该路径执行真实 Inductor extern addmm/mm。

该 Lab 没有生成、编译或执行 Inductor 生成的 native C++ kernel，因而不提供这一级证据。

## 学习顺序

- 上一篇：[[14_dead_code_topology_and_effect_order]]
- 下一篇：[[16_graph_rewrite_legality_validation_and_complexity]]

## Related Pages

- [[19_torch_compile_end_to_end/00_pytorch_graph_series_index]]
- [[08_graph_normalization_decomposition_and_functionalization]]
- [[13_pattern_expression_and_matcher_engine]]
- [[14_dead_code_topology_and_effect_order]]
- [[16_graph_rewrite_legality_validation_and_complexity]]
- [[fx_pass_optimization_methodology]]
