# Torch Compile Pass 方法论与阶段指南重构实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立覆盖 Dynamo、Pre-Grad、Joint、Post-Grad、Decomposition、Lowering、Scheduler、Codegen 的优化选层方法论，并把各阶段指南修订为源码一致、包含关键 API 与接入示例的开发文档。

**Architecture:** 一个总方法论页负责跨阶段“是什么 / 为什么 / 适合什么 / 为什么不放相邻阶段”的决策；阶段页负责真实主干 Pass、API、注册方式、示例和护栏。所有断言以 `pytorch@9922478dffa` 为基线，索引和 changelog 负责把新增页面接回知识图谱。

**Tech Stack:** Markdown、Obsidian wikilink、Mermaid、PowerShell、Git、PyTorch 源码 checkout。

## Global Constraints

- PyTorch 源码核验基线固定为 `9922478dffa`，新增非平凡断言必须带已打开核验的 `file:line`。
- 保留用户已有的 `wiki/02_engineering/05_gpu_kernel/cuda_nonmatmul_kernels_analysis.md` 修改，不暂存、不覆盖。
- 遵守仓库“只扩展或注释、不静默删除旧内容”的规则；错误旧写法用 `> [!deprecated]` 或 `> [!contradiction]` 标注。
- 每个阶段页包含 `## Related Pages`，新增页面至少链接一个已有页面并获得索引反向链接。
- 新增 Mermaid 图遵守仓库标签和定界符规则；无法实渲染时逐块人工核验。
- 区分公开 API、下游扩展钩子和 `torch._inductor` 内部原型接口。
- 新增可运行 Python 示例必须语法有效；结构骨架或省略实现的示例显式标为“骨架示例”。
- 实施完成后使用一个聚合文档提交并推送 `origin/main`。

---

### Task 1: 固定源码证据与错误清单

**Files:**
- Read: `E:/97-codes/torch_parallel/pytorch/torch/_dynamo/backends/registry.py`
- Read: `E:/97-codes/torch_parallel/pytorch/torch/_inductor/fx_passes/pre_grad.py`
- Read: `E:/97-codes/torch_parallel/pytorch/torch/_inductor/fx_passes/joint_graph.py`
- Read: `E:/97-codes/torch_parallel/pytorch/torch/_inductor/fx_passes/post_grad.py`
- Read: `E:/97-codes/torch_parallel/pytorch/torch/_inductor/pattern_matcher.py`
- Read: `E:/97-codes/torch_parallel/pytorch/torch/_inductor/custom_graph_pass.py`
- Read: `E:/97-codes/torch_parallel/pytorch/torch/_inductor/decomposition.py`
- Read: `E:/97-codes/torch_parallel/pytorch/torch/_inductor/lowering.py`
- Read: `E:/97-codes/torch_parallel/pytorch/torch/_inductor/scheduler.py`
- Read: `E:/97-codes/torch_parallel/pytorch/torch/_inductor/codegen/common.py`
- Read: `E:/97-codes/torch_parallel/pytorch/torch/_inductor/codegen/simd.py`

**Interfaces:**
- Consumes: design spec §5 已知错误清单和固定 commit `9922478dffa`。
- Produces: 后续任务引用的真实符号名、签名、调用位置、门控和行号。

- [ ] **Step 1: 核验三个 FX 阶段驱动器和 custom hook**

Run:

```powershell
git -C E:\97-codes\torch_parallel\pytorch show 9922478dffa:torch/_inductor/fx_passes/pre_grad.py | rg -n "def pre_grad_passes|pre_grad_custom_pass|PRE_GRAD_PATTERNS|stable_topological_sort"
git -C E:\97-codes\torch_parallel\pytorch show 9922478dffa:torch/_inductor/fx_passes/joint_graph.py | rg -n "def joint_graph_passes|joint_custom_pre_pass|joint_custom_post_pass|pass_patterns|early_patterns"
git -C E:\97-codes\torch_parallel\pytorch show 9922478dffa:torch/_inductor/fx_passes/post_grad.py | rg -n "def post_grad_passes|post_grad_custom_pre_pass|post_grad_custom_post_pass|pass_patterns|reinplace_inplaceable_ops"
```

Expected: 每个驱动器、custom hook、pattern 容器和收尾调用均能在固定 commit 中定位。

- [ ] **Step 2: 核验 PatternMatcher 与自定义 Graph Pass 签名**

Run:

```powershell
git -C E:\97-codes\torch_parallel\pytorch show 9922478dffa:torch/_inductor/pattern_matcher.py | rg -n "class PatternMatcherPass|def register_graph_pattern|def register_lowering_pattern|def register_replacement|class CallFunction|class KeywordArg"
git -C E:\97-codes\torch_parallel\pytorch show 9922478dffa:torch/_inductor/custom_graph_pass.py | rg -n "class CustomGraphPass|class CustomSchedulerPass|CustomGraphPassCallable|CustomSchedulerPassCallable|def get_custom_graph_passes"
```

Expected: 确认 FX custom hook 接收 `torch.fx.Graph`，Scheduler custom pass 接收并返回 `list[BaseSchedulerNode]`。

- [ ] **Step 3: 核验 Decomposition、Lowering、Scheduler 和 Codegen 扩展点**

Run:

```powershell
git -C E:\97-codes\torch_parallel\pytorch show 9922478dffa:torch/_inductor/decomposition.py | rg -n "register_decomposition|decompositions|select_decomp_table"
git -C E:\97-codes\torch_parallel\pytorch show 9922478dffa:torch/_inductor/lowering.py | rg -n "def register_lowering|def register_pointwise|def make_fallback|def fallback_handler|lowerings"
git -C E:\97-codes\torch_parallel\pytorch show 9922478dffa:torch/_inductor/scheduler.py | rg -n "_pre_fusion_custom_pass|_post_fusion_custom_pass|def can_fuse|def fuse_nodes|def score_fusion"
git -C E:\97-codes\torch_parallel\pytorch show 9922478dffa:torch/_inductor/codegen/common.py | rg -n "def register_backend_for_device|class BackendFeature"
```

Expected: 所有计划写入 API 表的符号均存在；不存在的旧符号进入错误修订清单而非新正文。

- [ ] **Step 4: 核验 Dynamo Backend 入口**

Run:

```powershell
git -C E:\97-codes\torch_parallel\pytorch show 9922478dffa:torch/_dynamo/backends/registry.py | rg -n "def register_backend|def lookup_backend|_BACKENDS"
git -C E:\97-codes\torch_parallel\pytorch show 9922478dffa:torch/_dynamo/output_graph.py | rg -n "call_user_compiler|compiler_fn|example_inputs"
```

Expected: 文档可准确说明 `torch.compile(backend=callable)` 的 FX GraphModule 和 example inputs 交接边界。

---

### Task 2: 重构完整选层方法论

**Files:**
- Modify: `wiki/02_engineering/01_ai_frameworks/04_inductor/fx_pass_optimization_methodology.md`

**Interfaces:**
- Consumes: Task 1 的八阶段源码定位；现有 `torch_upstream_pass_deepdive.md`、NPU、vLLM、SGLang 分析。
- Produces: 后续所有阶段页共同引用的八阶段术语、选择判据和相邻阶段边界。

- [ ] **Step 1: 更新页头和主线**

把标题调整为完整 Torch Compile 优化选层方法论，页头写入 PyTorch baseline 和 2026-07-22 更新日期；主线明确“最早仍拥有所需语义、最晚已经建立所需不变量”的选层原则。

- [ ] **Step 2: 写八阶段总览矩阵**

矩阵固定包含：阶段、IR/可见范围、已经建立的不变量、适合的优化、主要代价、关键扩展入口。八行必须覆盖 Dynamo、Pre、Joint、Post、Decomposition、Lowering、Scheduler、Codegen。

- [ ] **Step 3: 为每个阶段补齐‘是什么 / 为什么’**

每阶段使用相同小节结构：

```markdown
### <阶段名>：<一句话核心取舍>

**是什么**：<IR、位置、可见范围>。

**为什么存在**：<该阶段解决的核心矛盾>。

**适合做**：<优化类型及代表案例>。

**不适合做**：<缺少的信息或正确性风险>。

**为什么不放前一层 / 后一层**：<相邻阶段取舍>。
```

- [ ] **Step 4: 增加选层决策树与跨阶段反模式**

决策顺序固定为：需要 Python/Module 语义 → Dynamo/Pre；需要联合前反向 → Joint；需要规范 ATen 与后端钩子 → Post；决定拆或保留复合算子 → Decomposition；产生实现 IR → Lowering；决定 Kernel 边界 → Scheduler；实现硬件指令和 Wrapper → Codegen。

- [ ] **Step 5: 修正动态形状泛化错误**

把“SymInt 一律跳过”改为：能用符号 Guard 证明安全的上游 Pass 可以继续；依赖 Python 整数、静态布局或具体 tiling 的 Pass 应保守跳过，并以 NPU 手工 Pass 作为后者案例。

---

### Task 3: 修订 Pre-Grad 阶段指南

**Files:**
- Modify: `wiki/02_engineering/01_ai_frameworks/04_inductor/pre_grad_passes_guide.md`

**Interfaces:**
- Consumes: `pre_grad.py` 驱动顺序、`PatternMatcherPass` 和 `pre_grad_custom_pass` 签名。
- Produces: 正确的 Pre-Grad 主干 Pass 表、API 表和两条接入路径。

- [ ] **Step 1: 纠正阶段定义**

统一写成“Torch IR，未函数化、未规范化”，保留高层 `torch.nn.functional`/Module 语义；增加“为什么适合高层融合和批量化、为什么不适合依赖纯函数不变量的激进改写”。

- [ ] **Step 2: 以驱动顺序重写主要 Pass 表**

表中覆盖 `numpy_compat_normalization`、`fuse_fx`、`normalization_pass`、`group_batch_fusion_passes(pre_grad=True)`、`PRE_GRAD_PATTERNS` 配置项、`efficient_conv_bn_eval`、Gumbel 优化和 `pre_grad_custom_pass`，每项说明作用、为什么位于 Pre-Grad、门控和顺序依赖。

- [ ] **Step 3: 重写关键 API 表**

覆盖 `PatternMatcherPass`、`register_graph_pattern`、`CallFunction`、`KeywordArg`、`Match`、`PRE_GRAD_PATTERNS`、`pre_grad_fusion_options`、`pre_grad_custom_pass`、`stable_topological_sort`、`graph.lint()` 和 `GraphModule.recompile()`；标注内部 API 稳定性。

- [ ] **Step 4: 修正注册示例**

旧 `joint_custom_pre_pass` 写法加 deprecated 标注；新示例显式导入 `PatternMatcherPass`，构造独立 `PatternMatcherPass`，通过 `config.pre_grad_custom_pass` 调用其 `apply(graph)`，不要求修改 PyTorch 内建 `PRE_GRAD_PATTERNS`。

- [ ] **Step 5: 增加 Pre-Grad 验证清单**

覆盖参数变体、别名/Mutation、FakeTensor meta、动态 shape Guard、Pattern 命中计数、开关对比和推理/训练测试。

---

### Task 4: 修订 Joint Graph 阶段指南

**Files:**
- Modify: `wiki/02_engineering/01_ai_frameworks/04_inductor/joint_graph_passes_guide.md`

**Interfaces:**
- Consumes: `joint_graph.py` 驱动顺序、`fwd_only`/`joint_fwd_bwd` trace 机制和 joint custom hooks。
- Produces: 解释“函数化 + 切分前全局可见性”的 Joint 方法论、正确 Pattern 注册示例。

- [ ] **Step 1: 补齐 Joint 阶段存在理由**

说明联合图同时可见前向、反向和保存值边界；函数化后可安全做纯函数图改写；一旦 partition 后便失去跨前反向机会。

- [ ] **Step 2: 以驱动顺序重写主要 Pass 表**

覆盖 `canonicalize_aten_ir_passes`、`joint_custom_pre_pass`、`remove_noop_ops`、`constant_fold_uniform_value`、`early_patterns`、`auto_chunker`、`pass_patterns` 中的 SDPA/pad_mm、`replace_random_passes` 和 `joint_custom_post_pass`，说明为什么规范化最先、auto_chunker 在 pad_mm 前。

- [ ] **Step 3: 重写 API 表与 trace 选择**

覆盖 `register_graph_pattern`、`register_replacement`、`fwd_only`、`joint_fwd_bwd`、`early_patterns`、`pass_patterns`、两个 joint custom hook、`statically_known_true`；明确训练和推理 Pattern 不能混用。

- [ ] **Step 4: 修正导入和加载示例**

对错误的 `pass_patterns` 导入及“设置空 hook 确保加载”加 deprecated 标注；给出一个通过 `joint_custom_post_pass` 运行独立 `PatternMatcherPass` 的下游示例，以及一个注册到内建 `pass_patterns[index]` 的上游骨架示例。

- [ ] **Step 5: 增加正确性与性能清单**

覆盖 forward/backward 数值与梯度、FakeTensor、alias、RNG 可复现、动态 shape Guard、partition 前后影响和真实模型命中率。

---

### Task 5: 修订 Post-Grad 阶段指南

**Files:**
- Modify: `wiki/02_engineering/01_ai_frameworks/04_inductor/post_grad_passes_guide.md`

**Interfaces:**
- Consumes: `post_grad.py` 驱动顺序、Graph/Lowering Pattern 差异、post custom hook 签名。
- Produces: 后端扩展最常用阶段的完整接入说明。

- [ ] **Step 1: 补齐 Post-Grad 选层理由**

说明 forward/backward 已切分、ATen 图已函数化和规范化、shape/device 更具体，适合设备、内存、通信和后端特化；同时明确不能再做跨前反向联合优化。

- [ ] **Step 2: 以源码顺序归纳主要 Pass**

覆盖 DCE、`reorder_for_locality`、custom pre hook、group/batch fusion、noop/assert 清理、三轮 `pass_patterns`、`POST_GRAD_PATTERNS`、`b2b_gemm`、micro-pipeline TP、collective bucketing、custom post hook、构造器迁移和最后执行的 `reinplace_inplaceable_ops`。

- [ ] **Step 3: 重写 API 表**

覆盖 `register_graph_pattern`、`register_lowering_pattern`、`register_replacement`、`post_grad_custom_pre_pass`、`post_grad_custom_post_pass`、`CustomInferenceAwareGraphPass`、`PatternMatcherPass` 和 `pass_patterns`；说明即时改 ATen 图与延迟到 Lowering 产 IR 的差别。

- [ ] **Step 4: 修正并给出双路径示例**

核验 OpOverload 使用 `.default`/具体 overload，修正 Graph Pass 输入类型；给出一个 custom post hook 的下游示例和一个 lowering-pattern 的上游骨架示例。

- [ ] **Step 5: 增加顺序与回滚护栏**

说明 normalization、通信重排、Mutation reinplace 的先后约束；加入按 pass 禁用、数值/梯度测试、A/B profiling 和缓存失效要求。

---

### Task 6: 新增 Decomposition 指南

**Files:**
- Create: `wiki/02_engineering/01_ai_frameworks/04_inductor/decomposition_passes_guide.md`

**Interfaces:**
- Consumes: `torch/_inductor/decomposition.py`、AOTAutograd decomposition table 和 NPU“保持整块”案例。
- Produces: “拆开以便融合”与“保持完整以走强 Kernel”的阶段方法论和注册示例。

- [ ] **Step 1: 写‘是什么 / 为什么’**

说明 Decomposition 决定后续图的算子粒度：拆解可暴露 pointwise/reduction 融合机会，但也可能破坏 SDPA、LayerNorm、厂商手工库等强算子的整体语义。

- [ ] **Step 2: 写关键 API 表**

覆盖 `register_decomposition`、decomposition table、`select_decomp_table`、条件化分解和从表中移除/保留算子的作用；标明注册目标必须是具体 `OpOverload`。

- [ ] **Step 3: 给出最小注册示例**

示例展示将一个复合 ATen op 分解成基础 ATen op，并说明该注册通常是上游/后端集成代码，不是普通业务代码应全局修改的稳定入口。

- [ ] **Step 4: 写选择与验证清单**

包含数值稳定性、autograd、dtype promotion、动态 shape、后端 fallback、Pattern 命中率和“分解前后 Kernel 数”验证。

---

### Task 7: 修订 Lowering 与 Scheduler 指南

**Files:**
- Modify: `wiki/02_engineering/01_ai_frameworks/04_inductor/lowering_analysis.md`
- Modify: `wiki/02_engineering/01_ai_frameworks/04_inductor/scheduler_analysis.md`

**Interfaces:**
- Consumes: Lowering 注册表、Scheduler 融合合法性/评分、custom Scheduler pass 和 Backend 接口。
- Produces: 两个低层阶段清晰分工及无冲突扩展示例。

- [ ] **Step 1: 为 Lowering 增加选层方法论**

解释它负责“选实现并产 Inductor IR”，适合 pointwise/reduction IR、模板、外部 Kernel 和 fallback；不适合继续做依赖高层语义的大范围 FX 图识别。

- [ ] **Step 2: 补齐 Lowering API 与示例**

API 表覆盖 `lowerings`、`register_lowering`、`register_pointwise`、`make_pointwise`、`fallback_handler`、`make_fallback`、`validate_ir`；给出一个简单 pointwise lowering 骨架并明确 broadcast/type-promotion 由 wrapper 处理。

- [ ] **Step 3: 修复 Scheduler custom pass 冲突**

保留 `_pre_fusion_custom_pass` / `_post_fusion_custom_pass` 与 `Callable[[list[BaseSchedulerNode]], list[BaseSchedulerNode]]`；将 `pre_fusion_custom_pass(GraphLowering)` 标为过时错误，不再把 `node.fusable` 当作固定公共属性。

- [ ] **Step 4: 重写 Scheduler API 和扩展示例**

API 表覆盖 `can_fuse`、`can_fuse_vertical`、`can_fuse_horizontal`、`score_fusion`、`get_possible_fusions`、`no_fuse_buffer_names`、两个 custom scheduler hook、`register_backend_for_device`；示例只演示安全过滤/排序节点，不宣称通过布尔属性强制非法融合。

- [ ] **Step 5: 增加 Scheduler 验证清单**

覆盖环检测、依赖、迭代空间、reduction、tiling、内存流量、编译时间、峰值显存和开关 A/B benchmark。

---

### Task 8: 新增 Dynamo 阶段方法论与接入指南

**Files:**
- Create: `wiki/02_engineering/01_ai_frameworks/02_dynamo/dynamo_pass_methodology.md`
- Modify: `wiki/02_engineering/01_ai_frameworks/02_dynamo/index.md`

**Interfaces:**
- Consumes: Dynamo Backend registry、OutputGraph→compiler_fn 调用和 FX GraphModule 接口。
- Produces: Dynamo 在完整选层方法论中的准确边界和可运行自定义 Backend 示例。

- [ ] **Step 1: 写 Dynamo 的‘是什么 / 为什么’**

说明 Dynamo 首要任务是从 Python 执行中捕获 FX 子图、建立 Guards、处理 graph break 并交给 Backend；适合保留 Python/Module 语义的规范化或分析，不适合依赖 ATen 函数化、设备布局和 Kernel 成本的融合。

- [ ] **Step 2: 写关键 API 表**

覆盖 `torch.compile(backend=...)`、Backend callable、`register_backend`、`lookup_backend`、`GraphModule.graph`、`graph.lint()`、`GraphModule.recompile()` 和 `torch._dynamo.explain`。

- [ ] **Step 3: 写可运行最小 Backend 示例**

示例接收 `(gm, example_inputs)`，在副本或明确控制的 GraphModule 上做一个局部 FX 改写，执行 `lint/recompile`，返回可调用对象；同时给出按名称注册 Backend 的内部扩展写法。

- [ ] **Step 4: 写 Guard 与 graph break 注意事项**

说明 Python 值特化、输入结构、动态 shape、重编译和跨 graph-break 不可见性如何限制此层 Pass。

---

### Task 9: 新增 Codegen 扩展指南并补导航

**Files:**
- Create: `wiki/02_engineering/01_ai_frameworks/04_inductor/codegen_extension_guide.md`
- Modify: `wiki/02_engineering/01_ai_frameworks/04_inductor/inductor_codegen_analysis.md`

**Interfaces:**
- Consumes: `BaseScheduling`/Backend 注册、`codegen_node`/`codegen_template`/Wrapper 机制。
- Produces: 明确“Codegen 兑现融合而非重新做 FX Pass”的扩展方法论。

- [ ] **Step 1: 写 Codegen 的‘是什么 / 为什么’**

说明 Codegen 接收已决定 Kernel 边界的 Scheduler Node，把迭代域、索引、mask、load/store、reduction 和调用封装转成设备代码与 Wrapper；其核心是硬件映射而非高层子图识别。

- [ ] **Step 2: 写扩展面 API 表**

覆盖 `BaseScheduling`、`codegen_node`、`codegen_template`、`codegen_combo_kernel`、`define_kernel`、`register_backend_for_device`、Wrapper Codegen 和 BackendFeature；标注哪些是设备后端必须实现、哪些只用于模板路径。

- [ ] **Step 3: 给出 Backend 骨架示例**

示例以“骨架示例”标注，展示 Scheduling 子类、Wrapper 子类和 `register_backend_for_device` 的连接关系；不使用省略号冒充可运行实现，未展开的方法明确列为实现清单。

- [ ] **Step 4: 补充选择与验证清单**

包括 tiling、vectorization、memory coalescing、mask、数值精度、编译产物缓存、Kernel 参数 ABI、JIT/AOT Wrapper 和设备实测。

- [ ] **Step 5: 在现有 Codegen 分析页加入开发导航**

页首与 Related Pages 指向 `[[codegen_extension_guide]]`，原页继续承担机制分析而非重复开发步骤。

---

### Task 10: 集成索引、交叉引用和 changelog

**Files:**
- Modify: `wiki/02_engineering/01_ai_frameworks/04_inductor/index.md`
- Modify: `wiki/02_engineering/01_ai_frameworks/02_dynamo/index.md`
- Modify: `wiki/changelog.md`
- Modify: Task 2–9 涉及页面的 `## Related Pages`

**Interfaces:**
- Consumes: 新增三个阶段指南和重构后的方法论页。
- Produces: 从总索引可发现、相邻阶段双向可导航的知识图谱。

- [ ] **Step 1: 更新 Inductor 索引**

在各编译阶段加入 `decomposition_passes_guide`，在 FX Passes 区更新总方法论描述，在 Codegen 区加入 `codegen_extension_guide`。

- [ ] **Step 2: 更新 Dynamo 索引**

把 `dynamo_pass_methodology` 标为 deep dive/guide，并说明其与 Inductor 总方法论的边界。

- [ ] **Step 3: 补齐相邻阶段双向链接**

链路固定为 Dynamo ↔ Pre ↔ Joint ↔ Post ↔ Decomposition ↔ Lowering ↔ Scheduler ↔ Codegen；每页同时链接总方法论和流水线总览。

- [ ] **Step 4: 追加 changelog**

记录日期、固定源码 baseline、新增页面、修正的主要错误和方法论覆盖范围，不宣称未实测的性能收益。

---

### Task 11: 全量验证、提交与推送

**Files:**
- Verify: Task 2–10 的全部 Markdown 文件
- Exclude: `wiki/02_engineering/05_gpu_kernel/cuda_nonmatmul_kernels_analysis.md`

**Interfaces:**
- Consumes: 所有文档改动。
- Produces: 可追溯的验证证据、聚合 commit 和 `origin/main` 推送。

- [ ] **Step 1: 验证八阶段结构与关键章节**

Run:

```powershell
rg -n "Dynamo|Pre-Grad|Joint Graph|Post-Grad|Decomposition|Lowering|Scheduler|Codegen" wiki/02_engineering/01_ai_frameworks/04_inductor/fx_pass_optimization_methodology.md
rg -n "关键 API|注册|接入|为什么|适合做|不适合做" wiki/02_engineering/01_ai_frameworks/04_inductor/pre_grad_passes_guide.md wiki/02_engineering/01_ai_frameworks/04_inductor/joint_graph_passes_guide.md wiki/02_engineering/01_ai_frameworks/04_inductor/post_grad_passes_guide.md wiki/02_engineering/01_ai_frameworks/04_inductor/decomposition_passes_guide.md wiki/02_engineering/01_ai_frameworks/04_inductor/lowering_analysis.md wiki/02_engineering/01_ai_frameworks/04_inductor/scheduler_analysis.md wiki/02_engineering/01_ai_frameworks/02_dynamo/dynamo_pass_methodology.md wiki/02_engineering/01_ai_frameworks/04_inductor/codegen_extension_guide.md
```

Expected: 八阶段均出现；每个阶段页都能定位 API、接入示例和方法论章节。

- [ ] **Step 2: 验证已知错误已被显式修订**

Run:

```powershell
rg -n "joint_custom_pre_pass = debug_graph_pass|config\.pre_fusion_custom_pass|node\.fusable|from torch\._inductor\.pattern_matcher import.*pass_patterns" wiki/02_engineering/01_ai_frameworks/04_inductor
```

Expected: 命中只存在于明确的 deprecated/contradiction 说明中；正确示例不再使用这些写法。

- [ ] **Step 3: 验证修改页面 wikilink**

Run the following PowerShell check from the repository root:

```powershell
$wikiRoot = (Resolve-Path 'wiki').Path
$all = Get-ChildItem $wikiRoot -Recurse -Filter '*.md'
$stems = @{}
$rels = @{}
foreach ($file in $all) {
    $stems[$file.BaseName] = $true
    $rel = $file.FullName.Substring($wikiRoot.Length + 1).Replace('\','/').Replace('.md','')
    $rels[$rel] = $true
}
$changed = @(
    'wiki/02_engineering/01_ai_frameworks/02_dynamo/dynamo_pass_methodology.md',
    'wiki/02_engineering/01_ai_frameworks/02_dynamo/index.md',
    'wiki/02_engineering/01_ai_frameworks/04_inductor/fx_pass_optimization_methodology.md',
    'wiki/02_engineering/01_ai_frameworks/04_inductor/pre_grad_passes_guide.md',
    'wiki/02_engineering/01_ai_frameworks/04_inductor/joint_graph_passes_guide.md',
    'wiki/02_engineering/01_ai_frameworks/04_inductor/post_grad_passes_guide.md',
    'wiki/02_engineering/01_ai_frameworks/04_inductor/decomposition_passes_guide.md',
    'wiki/02_engineering/01_ai_frameworks/04_inductor/lowering_analysis.md',
    'wiki/02_engineering/01_ai_frameworks/04_inductor/scheduler_analysis.md',
    'wiki/02_engineering/01_ai_frameworks/04_inductor/codegen_extension_guide.md',
    'wiki/02_engineering/01_ai_frameworks/04_inductor/inductor_codegen_analysis.md',
    'wiki/02_engineering/01_ai_frameworks/04_inductor/index.md'
)
$missing = @()
foreach ($path in $changed) {
    $text = Get-Content -Raw -Encoding utf8 $path
    foreach ($match in [regex]::Matches($text, '\[\[([^\]]+)\]\]')) {
        $target = $match.Groups[1].Value.Split('|')[0].Split('#')[0].Replace('\','/').Trim()
        $leaf = Split-Path $target -Leaf
        if (-not $stems.ContainsKey($leaf) -and -not ($rels.Keys | Where-Object { $_.EndsWith($target) })) {
            $missing += "$path -> $target"
        }
    }
}
$missing
if ($missing.Count -ne 0) { exit 1 }
```

Expected: no output and exit code 0.

- [ ] **Step 4: 验证 Markdown 与 Mermaid**

Run:

```powershell
git diff --check
rg -n "```mermaid|subgraph|-->\||\.->\|" wiki/02_engineering/01_ai_frameworks/04_inductor/fx_pass_optimization_methodology.md wiki/02_engineering/01_ai_frameworks/04_inductor/*.md wiki/02_engineering/01_ai_frameworks/02_dynamo/*.md
```

Expected: `git diff --check` exit code 0；逐个新增 Mermaid 块通过仓库定界符人工清单。

- [ ] **Step 5: 审核改动范围**

Run:

```powershell
git status --short
git diff --stat
git diff --name-only
```

Expected: 用户的 CUDA 文档仍为未暂存修改；本任务只修改计划列出的设计、计划、阶段页、索引和 changelog。

- [ ] **Step 6: 暂存本任务文件并确认排除用户修改**

Run:

```powershell
git add -- docs/superpowers/plans/2026-07-22-torch-compile-pass-methodology.md wiki/02_engineering/01_ai_frameworks/02_dynamo/dynamo_pass_methodology.md wiki/02_engineering/01_ai_frameworks/02_dynamo/index.md wiki/02_engineering/01_ai_frameworks/04_inductor/fx_pass_optimization_methodology.md wiki/02_engineering/01_ai_frameworks/04_inductor/pre_grad_passes_guide.md wiki/02_engineering/01_ai_frameworks/04_inductor/joint_graph_passes_guide.md wiki/02_engineering/01_ai_frameworks/04_inductor/post_grad_passes_guide.md wiki/02_engineering/01_ai_frameworks/04_inductor/decomposition_passes_guide.md wiki/02_engineering/01_ai_frameworks/04_inductor/lowering_analysis.md wiki/02_engineering/01_ai_frameworks/04_inductor/scheduler_analysis.md wiki/02_engineering/01_ai_frameworks/04_inductor/codegen_extension_guide.md wiki/02_engineering/01_ai_frameworks/04_inductor/inductor_codegen_analysis.md wiki/02_engineering/01_ai_frameworks/04_inductor/index.md wiki/changelog.md
git diff --cached --name-only
git diff --cached --check
```

Expected: cached 列表不包含 `wiki/02_engineering/05_gpu_kernel/cuda_nonmatmul_kernels_analysis.md`，cached check exit code 0。

- [ ] **Step 7: 创建聚合提交**

Run:

```powershell
git commit -m "docs(inductor): complete pass placement and extension guides"
```

Expected: commit exit code 0，提交包含计划文件、三个新阶段指南、五个修订阶段页、总方法论、索引和 changelog。

- [ ] **Step 8: 推送 main**

Run:

```powershell
git push origin main
```

Expected: push exit code 0，`origin/main` 前进到新的聚合提交；用户的未提交 CUDA 文档修改仍保留在工作区。
