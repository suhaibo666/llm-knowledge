# Graph Compiler Foundations Labs

本目录把“源码审计基线”和“可运行 Lab 环境”明确分开：

- 实现结论审计：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`；
- 本机 Lab：Windows、Python `3.13.5`、PyTorch `2.9.1+cpu`，
  `torch.version.git_version=5811a8d7da873dd699ff6687092c225caffcf1bb`；
- 本机没有 CUDA 和 MSVC `cl`。因此 Triton/autotune 没有实测；CPU pointwise C++
  产物只做 codegen-only 捕获，不能称为已编译或已执行。

所有命令都从知识库根目录执行：

```powershell
cd E:\97-codes\torch_parallel\llm-knowledge
```

## 五个验收入口

### 1. 全系列贯穿模型

```powershell
python -B wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\series_artifact_bundle.py `
  --output-dir wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\artifacts\end_to_end
```

预期关键输出：

```text
forward_matches=True
gradient_matches=True
dynamic_export_has_range_constraints=True
aot_has_joint_forward_backward=True
aot_cross_graph_node_refs=0
aot_joint_partition_mapping_exact=True
aot_partition_to_compiler_callback_continuity=True
aot_saved_slot_binding_origins_match=True
artifact_bundle_continuity=partial
backend_codegen_status=generated_not_executed
hop_branch_captured=True
```

`artifacts/end_to_end/`包含模型源码、symbolic FX、Dynamo FX/guards、ExportedProgram
与 signature、functional ATen、AOT joint/fw/bw、partition ABI、HOP 子图、
post-grad FX、Inductor IR、pre/post-fusion Scheduler dump、wrapper/C++ source 与
provenance JSON。这里的“end-to-end”表示同一语义计算前缀的阶段覆盖：后端部分重新捕获
显式参数化的`backend_core`，并不是把前面记录的AOT forward `GraphModule`对象连续送入
Inductor。`aot_joint_to_fw_bw_node_mapping.json`使用同一次partition前注入、分图后读取的
lab-only origin token建立精确old-to-new映射；`artifact_manifest.json`把独立capture、
AOT连续段、backend连续段和缺失连续边分开列出。`stage_node_mapping.json`是语义阶段表，
会显式记录已验证transition与证据断点，不声称跨阶段 Python 对象 identity或单次编译的
逐节点连续映射。

### 2. 本机真实闭环补充

```powershell
python -B wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\part2_continuous_aot_inductor.py `
  --output-dir wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\artifacts\part2_continuous_aot_inductor

python -B wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\part3_real_stage_hooks.py `
  --output-dir wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\artifacts\part3_real_stage_hooks

python -B wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\part2_activation_peak.py `
  --output-dir wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\artifacts\part2_activation_peak
```

三个脚本只闭合本机能够直接实证的边界：

- `part2_continuous_aot_inductor.py`让一次真实`torch.compile`调用依次经过
  `CustomPartitionerFn`、AOT forward compiler callback、`compile_fx_inner`和
  `GraphLowering`，并在真实`Scheduler.__init__`完成后读取IR origin与依赖集合。固定
  extern-matmul例中，partition forward、compiler callback和GraphLowering入口看到同一
  `GraphModule`、`Graph`和owner identity；backward则在partition与callback之间重建
  module/Graph。两路到Scheduler都保留module、Graph和全量token，但
  `GraphLowering.__init__`内部的浅拷贝会把同一Graph的`owning_module`转移给
  `orig_gm`；artifact逐项记录这个owner transition以及Scheduler origin token、
  read/write dependency数量；
- `part3_real_stage_hooks.py`在真实Inductor driver中分别把pre-grad的
  `operator.matmul + operator.add`和post-grad的
  `aten.mm.default + aten.add.Tensor`改写为相应`addmm`。错误stage target零命中，
  forward/梯度正确；hook内把同一pass再次运行于已改写Graph并验证零rewrite、代码
  不变。第二次compiled函数调用命中缓存、不重编译是另一项独立观察；配置退出后恢复；
- `part2_activation_peak.py`在编译warm-up后用`saved_tensors_hooks`的真实
  pack/unpack事件分别测量“每个active saved value的logical tensor bytes之和”和
  “按untyped storage去重的backing-storage bytes”。pack保存的是detach结果，不持有
  hook输入tensor本身；不相交view回归会区分logical value与共享storage。固定例两种
  指标的high/low budget均为`768/512` bytes；它们都不是CUDA allocator peak。当前无CUDA，因此
  `physical_allocator_peak_bytes=null`且状态为`blocked_no_cuda`。

连续链和stage-hook例真实执行ATen extern `mm/addmm`，不使用mock compiler，但也不声称
执行了Inductor生成的native C++ kernel。专用manifest记录入口、命令、环境、结果和每个
非manifest产物的SHA-256。

### 3. Part III 贯穿改写

```powershell
python -B wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\part3_end_to_end_pass.py `
  --output-dir wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\artifacts\part3

python -B wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\part3_pattern.py `
  --output-dir wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\artifacts\part3_pattern
```

正例只接受 rank-2、相同 dtype、`bias.shape == matmul_output.shape`的
`add(matmul(x, weight), bias) → addmm(bias, x, weight)`。边界例使用 `(5,)` bias；
必须拒绝且 graph/code/meta 保持不变。脚本对数值、一阶梯度、`gradcheck`、shape、
alias relation、输入 mutation relation、失败原子性和第二次运行零改动做 assertion。

`part3_pattern.py`另行覆盖三类真实 entry：`GraphPatternEntry`执行 handler 图手术，
`ReplacementPatternEntry`执行 traced replacement，`LoweringPatternEntry`把 handler
延迟到`GraphLowering`并生成`ComputedBuffer(Pointwise)`。它还覆盖 unary、共享输入、
kwargs、`Ignored`、`MultiOutputPattern`、`extra_check`拒绝和二次运行零命中；这里只到
Inductor IR，`lowering_native_kernel_executed=False`。

### 4. Part IV 后端证据包

```powershell
python -B wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\part4_ir_scheduler_analysis.py `
  --output-dir wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\artifacts\part4_ir

python -B wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\part4_artifact_bundle.py `
  --output-dir wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs\artifacts\part4
```

它同时保留三种证据等级：

- `part4_ir_scheduler_analysis.py`：真实执行GraphLowering、Scheduler、dependency/fusion、
  reorder on/off与静态peak估计；当前固定case的顺序`[op0,op1,op2] → [op1,op0,op2]`、
  估计peak `264192 → 263172`，不请求native codegen；
- external matmul 与 `eigvals` fallback/extern 路径：真实执行并与 eager 比较，同时保存
  `eigvals`的`FallbackKernel` IR trace；
- pointwise/reduction：真实编译因缺 `cl`阻塞；
- fusion/custom-lowering：绕过 compiler discovery 并把 `cpp_pybinding`替换为 no-op，
  只捕获 IR、Scheduler、wrapper 与 C++ source，标记为
  `generated_not_executed`。

codegen-only对照中，Scheduler group数`1/2`、C++ loop计数`2/3`，但两边entry point均为
`1`，所以不把group数写成kernel数。`provenance_chain.json`实际连接Scheduler subnode、
post/pre FX node、C++ debug handle与Python stack trace；它不是runtime PC映射。两个脚本的
`environment.json`同时记录source locator基线`e8f97c…`与runtime torch
`5811a8…`，并明确`runtime_matches_source_baseline=false`。

### 5. 自动合同

```powershell
python -m unittest discover `
  -s wiki\02_engineering\01_ai_frameworks\19_torch_compile_end_to_end\labs `
  -p test_series_contract.py -v
```

合同失败会返回非零退出码；不能以“脚本打印了一组布尔量”替代 assertions。

## 页面到 Lab 的映射

| 页面 | 机制脚本 | 正例 | 错误或边界 | 持久证据 |
|---:|---|---|---|---|
| 01 | `part1_graph_taxonomy.py` | eager tape 与 FX program graph | 两种 Node 不共享 identity | `end_to_end/dynamo_fx.py`、`dynamo_guards.txt` |
| 02 | `part1_fx_core.py` | create/replace/recompile | live erase、cross-owner lint | stdout；贯穿图见 `end_to_end/symbolic_fx.py` |
| 03 | `part1_values_signatures.py` | parameter/buffer lifting | `ExportedProgram`不可直接调用 | `end_to_end/export_graph_signature.json` |
| 04 | `part1_symbolic_shapes.py` | automatic dynamic 与 range constraint | static specialization；越界由 export contract 拒绝 | `end_to_end/export_graph_signature.json` |
| 05 | `part1_effects_alias.py`、`part2_normalization.py` | impure DCE、alias、functionalization | pure dead node 与 view mutation | `end_to_end/functional_aten.py` |
| 06 | `part1_structured_hop.py` | tuple/dict 与 `cond` | 两个 branch contract 必须相同 | `end_to_end/hop_exported_program.py` |
| 07 | `part2_capture_frontends.py`、贯穿 bundle | 四种前端 | graph break/guard 边界 | `end_to_end/{symbolic_fx,dynamo_fx,exported_program}.py` |
| 08 | `part2_normalization.py` | functionalization/decomposition | mutating view | `end_to_end/functional_aten.py` |
| 09 | `part2_aot_graphs.py`、`part2_continuous_aot_inductor.py`、贯穿 bundle | joint/fw/bw mapping；extern-matmul的AOT fw直接进入真实Inductor | 无跨 Graph Node 引用；backward可重建Graph；不声称native kernel | `end_to_end/aot_*.py`、`artifacts/part2_continuous_aot_inductor/` |
| 10 | `part2_aot_graphs.py`、`part2_aot_recompute_analysis.py`、`part2_activation_peak.py` | min-cut budget与runtime saved-tensor logical peak对照 | save/recompute切换；physical allocator明确阻塞 | `artifacts/part2_recompute/`、`artifacts/part2_activation_peak/` |
| 11 | 贯穿 bundle、`part2_continuous_aot_inductor.py` | 原bundle保留两段证据；另以extern-matmul闭合AOT fw→Inductor单次链 | symbolic/export仍是独立capture；native kernel未执行 | `end_to_end/artifact_manifest.json`、`artifacts/part2_continuous_aot_inductor/continuity.json` |
| 12、14–16 | `part3_end_to_end_pass.py`、`part3_passes.py`、`part3_real_stage_hooks.py`、`part3_legality.py` | 合法rewrite、真实pre/post-grad hook、bounded oscillation | broadcast reject、failure atomicity、错stage零命中、同一pass二次运行与compiled函数缓存复用 | `artifacts/part3/`、`artifacts/part3_real_stage_hooks/` |
| 13 | `part3_pattern.py` | unary/shared/kwargs/multi-output 与三类 entry | sharing failure、`extra_check`拒绝、二次 apply=0、native kernel未执行 | `artifacts/part3_pattern/` |
| 17–21 | `part4_ir_scheduler_analysis.py`、`part4_artifact_bundle.py` | Pointwise/Reduction/Extern、dependency/fusion/reorder、external/custom lowering | view/copy、unsupported fallback、fusion limit、缺 compiler/GPU | `artifacts/part4_ir/`、`artifacts/part4/` |

这里的“边界”不是都等于异常：例如跨阶段 identity 不连续、fallback、不能证明 shape
而拒绝 rewrite，都属于机制的合法边界。页面若没有真实测量（如 GPU autotune 或 allocator
peak），会明确写“未验证”，不会由源码结构推测数值。

## Related Pages

- [[19_torch_compile_end_to_end/00_pytorch_graph_series_index]]
- [[16_graph_rewrite_legality_validation_and_complexity]]
- [[11_graph_stage_boundaries_identity_and_provenance]]
- [[21_codegen_kernel_mapping_autotuning_and_provenance]]
