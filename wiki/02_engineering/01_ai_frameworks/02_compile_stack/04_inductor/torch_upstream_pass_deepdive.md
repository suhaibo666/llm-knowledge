# torch 上游 Inductor Pass 全集与机制 — 一个 PatternMatcher 引擎，三处 IR 阶段落地

> **页面角色**：upstream pass全集、Pattern注册和阶段目录参考。
> **原始基线**：见下方`9922478dffa`；**当前审计基线**：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`。
> **课程分工**：本页保留大全式参考；PatternExpr、候选检索、匹配/替换和通用管线的当前主线见 [[19_torch_compile_end_to_end/13_pattern_expression_and_matcher_engine]] 与 [[19_torch_compile_end_to_end/15_graph_pass_pipeline_ordering_and_fixpoint]]。

> **Source baseline**: pytorch @ `9922478dffa`（main，2026-07-20 校验）
> **Dimension**: Deep Dive（mechanism-level）
> 本页是上游 Inductor FX pass 的「全集 + 机制」总纲：既给出 pre_grad / joint / post_grad 三阶段所有 pass 的目录（名字·干什么·门控·file:line），又补上三份 stage 指南 [[pre_grad_passes_guide]] / [[joint_graph_passes_guide]] / [[post_grad_passes_guide]] 缺失的「机制层」——pattern 如何被声明、trace、匹配、改写，三阶段如何被 compile_fx 串起来。与 [[npu_vs_upstream_fusion_passes]] 互为上游侧 / NPU 侧对照；工业界 pass 开发方法论归纳见 [[fx_pass_optimization_methodology]]。

---

## 1. 概览
### 一条主线（thesis）

**上游 Inductor 的所有图改写，几乎都由同一台 `PatternMatcherPass` 声明式引擎驱动；真正的多样性不在「匹配算法」，而在「这批 pattern 注册进哪个阶段的哪个 pass_dict、以什么 trace 方式生成、命中后是原地改 FX 图还是推迟到 lowering 才产 IR」。** 理解了「声明→trace→匹配→改写」这一条链路，和「pre_grad（Torch IR，未函数化）/ joint（aten IR，函数化后、切分前）/ post_grad（aten IR，切分后、lowering 前）」三个落地点，整个 pass 体系就是这台引擎在不同 IR 阶段的重复实例化。

### 编译流水线（pass 落在哪一段）

```
Dynamo 抓取 → Torch-IR GraphModule
        │
        ▼  compile_fx: run_pre_grad_passes → _recursive_pre_grad_passes  (compile_fx.py:506)
  ┌───────────────────────────────────────────────────────────────┐
  │ ① pre_grad_passes   (pre_grad.py:287)                          │
  │   IR = Torch IR,未函数化/未规范化;要自己处理别名与突变       │
  │   fuse_fx / normalization / group_batch / split_cat / gumbel   │
  └───────────────────────────────────────────────────────────────┘
        │
        ▼  AOTAutograd:trace 出 joint fwd+bwd 图(函数化 + 规范化,aten IR)
  ┌───────────────────────────────────────────────────────────────┐
  │ ② joint_graph_passes   (joint_graph.py:648)                    │
  │   在"联合图 / 切分之前"跑:SDPA 融合、pad_mm、常量折叠、       │
  │   replace_random、pointless_* 化简                              │
  └───────────────────────────────────────────────────────────────┘
        │
        ▼  partitioner 把联合图切成 forward GM + backward GM
  ┌───────────────────────────────────────────────────────────────┐
  │ ③ post_grad_passes   (post_grad.py:144)  —— 对 fwd、bwd 各跑一次 │
  │   mm_plus_mm / b2b_gemm / decompose_mem_bound_mm / reinplace /  │
  │   micro_pipeline_tp / group_batch(post) / 通信 bucketing        │
  └───────────────────────────────────────────────────────────────┘
        │
        ▼  GraphLowering:aten → Inductor IR
  ┌───────────────────────────────────────────────────────────────┐
  │ ④ lowering:普通 lowering + "lowering-pattern" 直通             │
  │   register_lowering_pattern 命中在 ③ 埋下的直通节点在此产 IR    │
  │   (graph.py:1372-1376)                                          │
  └───────────────────────────────────────────────────────────────┘
        │
        ▼  Scheduler / codegen(Triton / C++)
```

三阶段的调用点在 `compile_fx.py`：`_recursive_pre_grad_passes`（506，受 `config.use_pre_grad_passes` 门控 515）、`_recursive_joint_graph_passes`（528，`use_joint_graph_passes` 545，调 `joint_graph_passes` 558）、`_recursive_post_grad_passes`（570，`use_post_grad_passes` 576，调 `post_grad_passes` 582；在 lowering 前的 1380 触发）。三者都会**递归下钻子图**（HOP/invoke_subgraph 的 subgraph）后再处理外层图。

### 关键概念表

| 概念 | 是什么 | file:line |
|---|---|---|
| `PatternMatcherPass` | 一批 pattern 的注册表 + 驱动器；`apply(gm)` 遍历图节点做匹配改写 | pattern_matcher.py:2326 / apply 2352 |
| `PatternExpr` | pattern 语法树基类；子类 `CallFunction`/`KeywordArg`/`Arg`/`Ignored`/`MultiOutputPattern` 等 | pattern_matcher.py:476 |
| `PatternEntry` 三型 | `LoweringPatternEntry`（推迟到 lowering 产 IR）/`GraphPatternEntry`（即时改 FX 图）/`ReplacementPatternEntry`（贴入 traced 替换子图） | 1153 / 1167 / 1180 |
| `register_graph_pattern` | 装饰器，注册「命中后跑一段函数改图」的 pattern | pattern_matcher.py:2084 |
| `register_lowering_pattern` | 装饰器，注册「aten 子图 → 单个直接产 Inductor IR 的节点」 | pattern_matcher.py:2052 |
| `register_replacement` | 由 search_fn/replace_fn 两个「例子函数」trace 出 pattern+替换图 | pattern_matcher.py:1588 |
| `fwd_only` / `joint_fwd_bwd` | 两种 trace 方式：推理图 / 训练联合图 | 2583 / 2613 |
| `GraphTransformObserver` | 每个 pass 的包裹器：计时、日志、可按名/子系统禁用 | torch/fx/passes/graph_transform_observer.py:22 |
| `pre/joint/post_grad_custom_*_pass` | 三阶段的用户自定义 pass 钩子（config） | config.py:313 / 307-308 / 300-301 |

---

## 2. Pass 基础设施（现有 stage 指南缺失的机制层）
### 2.1 一次匹配的生命周期：声明 → trace → 匹配 → 改写

**声明**。所有 pattern 用 `PatternExpr` 语法树描述要匹配的子图：`CallFunction(target, *arg_patterns, **kwarg_patterns)`（pattern_matcher.py:845）是最常用的节点；叶子里 `KeywordArg("q")`（536）在命中时把该节点作为名为 `q` 的入参捕获交给 handler，`Arg`（511）按位置捕获，`Ignored`（521，`repr` 为 `*`）匹配但不传参，`MultiOutputPattern`（948）描述多输出子图。`PatternExpr._match`（482）是抽象方法，`match`（484）是入口。

**trace（仅 replacement 类）**。`register_replacement`（1588）不手写语法树，而是给两个「例子函数」：`search_fn`（要匹配的原样）与 `replace_fn`（替换成什么），外加 `example_inputs` 和 `trace_fn`。`trace_fn` 把 `search_fn` trace 成一个 FX `GraphModule`，再由 `fx_to_pattern`（2472）转成 `PatternExpr` 树（初始 trace 用 `ignore_types=(int,float,list,...)`，故形状不会烧死进 pattern，见 gen_pattern_and_search_gm 2031）。替换图不预先生成——命中后在 `check_fn`（1638）里用**真实形状重新 trace** `search_fn` 复核匹配（因为初始匹配忽略了 int 类型），复核过了才 trace `replace_fn` 得到 `match.replacement_graph`（1779）。一个刻意的取舍：`joint_fwd_bwd` 的训练 pattern 在推理模式下直接跳过（1809-1813）。

**匹配 + 改写**。`PatternMatcherPass.apply`（2352）是驱动器：它把已注册 pattern 按 `(op, target)` 建索引（`self.patterns` defaultdict，2337），用 `graph.find_nodes` 只取可能命中的节点（2373-2379），**逆拓扑序遍历**（`sorted(..., reverse=True)`，2383），对每个节点查 `patterns[(op,target)]`、`entry.pattern.match(node)`、跑 `extra_check`（用 `guard_or_false` 包裹，2423），命中就调 `entry.apply(...)`。整个过程包在 `GraphTransformObserver`（2382）里。两条重要的**安全护栏**：匹配若跨越突变区（mutation region）边界则丢弃（2400），跨越 stream 边界（`meta["custom"]["stream"]`）则丢弃（2409）——保证融合不会越过 in-place 突变或多流边界。

### 2.2 三种 `PatternEntry`：命中后到底做什么

`PatternEntry`（1123，dataclass：`pattern` + `extra_check` + `register` 1130）有三个落地子类，这是「两种 pattern 匹配」区别的根：

| 子类 | `apply` 干什么 | file:line |
|---|---|---|
| `LoweringPatternEntry` | 在匹配点插入一个 `call_function(handler, ...)` 节点（handler 打了 `_inductor_lowering_function` 标记），**把真正产 IR 的动作推迟到 lowering**；原节点擦除 | apply 1156 |
| `GraphPatternEntry` | **即时**调用 `handler(match, *args, **kwargs)` 直接改 FX 图 | apply 1174 |
| `ReplacementPatternEntry` | 用 `torch.fx.Interpreter`（`Replacer`）把 traced 替换子图**逐节点复制贴入**主图（`replace_with_graph` 1189） | 1180 |

三个注册入口一一对应：`register_lowering_pattern`（2052，装饰器 → `LoweringPatternEntry` 2069）、`register_graph_pattern`（2084 → `GraphPatternEntry` 2098）、`register_replacement`（1588，末尾建 `ReplacementPatternEntry` 并 `register` 到 pass_dict，1842-1848）。`register` 方法（1130）按 pattern 的 `fns`（即 target）把 entry 挂到 `pass_dicts[(op, target)]`，支持 `prepend` 插队和多 pass_dict 广播。

### 2.3 两种 trace 方式：`fwd_only` vs `joint_fwd_bwd`
`register_replacement` 的 `trace_fn` 决定 pattern 匹配的是推理图还是训练图：

- `fwd_only`（2583）：`make_fx(fn, decomps, tracing_mode="real")` 直接 trace，再跑 `remove_noop_ops` + `eliminate_dead_code`，得到「规范化推理图」。
- `joint_fwd_bwd`（2613）：走 `aot_function` + `default_partition`，用 `record_joint_graph`（2623）截下**联合 fwd+bwd 图**，跑 `remove_noop_ops` 与 `early_patterns.apply`（2649）后作为训练 pattern。SDPA、pad_mm 等都各注册 `_training`（joint_fwd_bwd）与 `_inference`（fwd_only）两版。

### 2.4 序列化 pattern 缓存（避免每次导入都重 trace）

从例子函数 trace 出 pattern 很贵，上游把它**预生成并落盘**。`gen_register_replacement`（1951）是带缓存的注册入口：

- 设了环境变量 `PYTORCH_GEN_PATTERNS` 时，调 `_serialize_pattern`（1855）重新 trace 并把 pattern 的 `PatternPrettyPrinter` 文本写进 `torch/_inductor/fx_passes/serialized_patterns/<search_fn名>.py`（路径常量 `SERIALIZED_PATTERN_PATH` 1935；文件头注明「auto-generated，用 torchgen/fuse/gen_patterns.py 重生成」1866-1867）。
- 正常运行时**不 trace**，直接 `importlib.import_module("torch._inductor.fx_passes.serialized_patterns.<name>")` 取出预生成的 `PatternExpr`（1973-1981），作为 `search_fn_pattern` 传给 `register_replacement`（2003）——省掉运行期 trace。

落盘的是 **`.py`（非 `.pt`）** 文件；目录里最典型的是 30 个 `_sfdp_pattern_*.py`（SDPA 各模板）。`test_serialized_patterns_up_to_date()` 用 `_known_precompiled_patterns`（1940）校验缓存与源码同步。

### 2.5 三阶段驱动器：如何被调用、如何收尾
三个 driver 函数结构一致——**入口先 `lazy_init` 一次性注册，中间按序跑 pass，尾部统一 `stable_topological_sort` + `lint` + `recompile`**——但落在不同 IR 阶段：

- **pre_grad**（`pre_grad_passes` pre_grad.py:287）：受 `config.pattern_matcher`（config.py:290，默认 `True`）门控；入口 `lazy_init()`（305 → def 174，导入 `split_cat`/`efficient_conv_bn_eval`/`apply_gumbel_max_trick`，fbcode 时另加 `fb`）。分两条互斥路径：`config.is_predispatch` 走 aten 预派发列表 `_run_pre_dispatch_passes`（199，`default_pass_list` 206）；否则走 OSS 路径：`numpy_compat_normalization`（316）→ `fuse_fx`（318）→ `normalization_pass`（320-322）→ `group_batch_fusion_passes(pre_grad=True)`（323-325）→ 遍历 `config.pre_grad_fusion_options` 应用 `PRE_GRAD_PATTERNS[name]`（326-348）→ `efficient_conv_bn_eval`（350）→ gumbel（353）。收尾：custom 钩子（357）、`stable_topological_sort`（362）、`quant_lift_up`（366）、`lint`/`recompile`（368-369）。
- **joint**（`joint_graph_passes` joint_graph.py:648）：`lazy_init`（660 → def 80，`@init_once_fakemode`，调 `_pad_mm_init`/`_sfdp_init`/`_misc_patterns_init` 86-88）→ `canonicalize_aten_ir_passes`（664，「必须最先」）→ joint_custom_pre 钩子（666）→ `remove_noop_ops`（674）→ `constant_fold_uniform_value`（677，门控 `joint_graph_constant_folding` config.py:1079）→ `early_patterns.apply`（681）→ auto_chunker（686，在 pad_mm 前）→ `pass_patterns` 循环（697-702，SDPA/pad_mm 在此 fire）→ `replace_random_passes`（704）→ joint_custom_post 钩子（709）→ 若有改动则 `stable_topological_sort`+`lint`（717-720）。
- **post_grad**（`post_grad_passes` post_grad.py:144）：DCE（156）→ `reorder_for_locality`（160，仅推理）→ post_grad_custom_pre 钩子（167）→ `remove_profiler_ops`（195）→ `group_batch_fusion_passes(pre_grad=False)`（201）→ `remove_noop_ops`/`remove_assert_ops`（204-205）→ **`pass_patterns[0→1→2]` 循环**（208-211）→ `POST_GRAD_PATTERNS` 循环（216-237）→ `b2b_gemm`（238）→ `micro_pipeline_tp`（241）→ 通信 bucketing（244+）→ post_grad_custom_post 钩子（253）→ `stable_topological_sort`（269）→ `move_constructors_to_gpu`（271）→ …→ `reinplace_inplaceable_ops`（423，「保持最后，因它引入突变」）→ `lint`/`recompile`（445-446）。

**`GraphTransformObserver`**（torch/fx/passes/graph_transform_observer.py:22）包裹每个 pass：`apply_graph_pass`（92）/`apply_gm_pass`（79）在 `with self:` 内跑，先 `_check_disable_pass`——可按 `config.disabled_passes`（按名）或 `CompilerBisector.disable_subsystem`（按 subsystem，如 `"post_grad_passes"`）禁用，这是 pass 级 bisect 调试的抓手，并用 `dynamo_timed(f"pass.{subsystem}.{passname}")` 计时。**`stable_topological_sort`**（pattern_matcher.py:2664）保证改图后节点顺序确定性，`graph.lint()` 校验图合法性。

### 2.6 自定义 pass 钩子（下游插入点）

三阶段都留了 config 钩子，经 `get_custom_graph_passes(...)` 取出后用 observer 跑：`config.pre_grad_custom_pass`（config.py:313；调用 pre_grad.py:357）、`config.joint_custom_pre_pass`/`joint_custom_post_pass`（307-308；调用 joint_graph.py:666/709）、`config.post_grad_custom_pre_pass`/`post_grad_custom_post_pass`（300-301；调用 post_grad.py:167/253）。post_grad 的两个钩子还支持 `CustomInferenceAwareGraphPass`（会 `functools.partial` 注入 `is_inference`，post_grad.py:170-172）。**这正是 torch_npu / vLLM / SGLang 三家都用来挂自己 pass 的官方插入点**（见 [[npu_vs_upstream_fusion_passes]] / [[vllm_ir_and_fusion_passes_analysis]] / [[sglang_compilation_passes_analysis]]）。[注：post_grad.py:201 里 `group_batch_fusion_passes` 复用了 observer 标签字符串 `"post_grad_custom_pre_pass"`，与真正的钩子同名，属标签复用，非同一 pass。]

---

## 3. Pass 全集目录
> 门控约定：`config.pattern_matcher`（默认 True）是三阶段 pattern 类 pass 的总闸；`pre_grad_fusion_options`/`post_grad_fusion_options` 默认 `{}`（config.py:362/366），故 group_batch 与 split_cat 家族**默认不做事**，需用户填 options 才逐个启用。

### 3.1 pre_grad（Torch IR，未函数化）

| Pass | 干什么 | 门控 | file:line |
|---|---|---|---|
| `numpy_compat_normalization` | 把 NumPy 风格 kwarg（`axis`→`dim` 等）改成 torch 规范名 | OSS 路径无条件 | 调用 pre_grad.py:316；类 misc_patterns.py:194 |
| `fuse_fx`（子驱动） | cat-sink + permute/linear/matmul 融合 +（freezing）去 Identity/conv-bn | `example_inputs` 非空 | def pre_grad.py:391 |
| ├ `sink_cat_after_pointwise` | 把 `cat` 后的逐元素激活下沉到 cat 之前 | 无条件 | pre_grad.py:702（调 396） |
| ├ `linear_permute_fusion` | `linear(...).permute(-1,-2)` → `linear_transpose` | `config.permute_fusion`（默认关，config.py:1518）且非 CPU | 765（调 402） |
| ├ `permute_linear_fusion` | `linear(x.permute)` → `transpose_linear` | 同上 | 806（调 404） |
| ├ `permute_matmul_fusion` | permute→bmm/matmul → `transpose_matmul` | 同上 | 840（调 406） |
| ├ `remove_identity` | 擦除 `nn.Identity` 模块节点 | `config.freezing`（默认关）+ grad 关 | 431（调 413） |
| └ `fuse_conv_bn` | 推理态 Conv+BN 折叠为单卷积 | `config.freezing` | 451（调 415） |
| `normalization_pass` | split/cat/stack/unbind 参数 schema 规范化（split_cat 家族首个） | options 含 `"normalization_pass"` | 应用 pre_grad.py:320-322；注册 split_cat.py |
| `group_batch_fusion_passes(pre_grad=True)` | 批式融合 batch_linear/layernorm/tanh/sigmoid/relu/… | 各融合由 `pre_grad_fusion_options` 成员启用 | 应用 pre_grad.py:323-325；驱动 group_batch_fusion.py:1416 |
| split_cat 家族（循环） | `merge_splits`/`split_cat`/`unbind_stack`/`merge_getitem_cat`/`mutate_cat`/`split_cat_to_slices`/… | 遍历 `pre_grad_fusion_options`（跳过 group-batch 名），每个可跑 `counter` 次 | pre_grad.py:326-348；`PRE_GRAD_PATTERNS` split_cat.py:48 |
| `efficient_conv_bn_eval_pass` | eval 态 Conv+BN 融合为高效 conv-bn-eval | 其 pattern 仅在 `efficient_conv_bn_eval_fx_passes`（默认关）时注册 | 应用 350-352；pattern efficient_conv_bn_eval.py:144 |
| `apply_gumbel_max_trick_pass` | `argmax(softmax/rand, -1)`（Gumbel-max）重写为优化式 | 无 extra_check | 应用 353-355；pattern apply_gumbel_max_trick.py:10 |
| `pre_grad_custom_pass`（钩子） | 用户自定义 pre_grad pass | `config.pre_grad_custom_pass` | 357-360 |

> `binary_folding.py` **不属于 pre_grad**：它经 `register_binary_folding_pattern` 注册进 **freezing** pattern 集，由 `config.enable_linear_binary_folding` 门控，`pre_grad_passes` 未引用它。[flagged]

### 3.2 joint_graph（aten IR，函数化后 · 切分前）

联合图级三个 `PatternMatcherPass` 对象：`early_patterns`（joint_graph.py:48）、`patterns`（=`pass_patterns[0]`，49）、匿名 `pass_patterns[1]`（55），均 `subsystem="joint_graph_passes"`（43-45）。

| Pass | 干什么 | 门控 / pass_dict | file:line |
|---|---|---|---|
| `canonicalize_aten_ir_passes` | 规范量化映射等，必须最先 | 无条件 | joint_graph.py:664（def 640） |
| `constant_fold_uniform_value` | 把 uniform 张量折叠成 `aten.full`（`UniformValueConstantFolder`） | `joint_graph_constant_folding`（config.py:1079） | 677-679；folder 253 |
| `early_patterns` / `pointless_view(_pair)` / `pointless_permute_pair` | 消去 no-op view / 互逆 permute | `config.pattern_matcher` | 应用 681；pattern 860/874/895 |
| `fix_iota_device` | `prims.iota` 改到消费者设备，避免主机-设备拷贝断 cudagraph | `patterns` | 724 |
| `pointless_convert` | 合并两次 `convert_element_type`（AMP 链） | `patterns` | 778 |
| `bmm_to_mm` | batch 静态为 1 时 `bmm`→`mm` | `patterns` | 916 |
| `mul_/div_softmax_pattern` | `scale(x)-amax(scale(x))` → 数值稳定的 `scale(x-amax(x))` | `pass_patterns[1]` + extra_check | 1049/1084 |
| `scatter_upon_const_tensor` | `full`+`scatter.value` → 逐元素 `where` | `optimize_scatter_upon_const_tensor` | 1126 |
| **SDPA 融合**（`_sfdp_*`） | `matmul(q,kᵀ)/scale → softmax → matmul(v)` → `_scaled_dot_product_attention`；共 **30** 个模板（`_sfdp_pattern_1..30`），每个含 training+inference 两版 | `_sfdp_init` 无条件注册进 `patterns`（无独立 `config.sdpa` 开关），随 `pass_patterns` 循环 fire | `_sfdp_init` fuse_attention.py:1487；`_sfdp_pattern_1` 43 |
| **pad_mm** | 把 mm/bmm/addmm 的 M/N/K 补齐到对齐倍数以吃满 tensor-core | `_pad_mm_init` 注册 3 组；实际生效受 `config.shape_padding`（config.py:1465）+ `should_pad_*` benchmark | `_pad_mm_init` pad_mm.py:920；`pad_mm` 828 |
| `replace_random_passes` | 把 eager `rand/randn/randint` 换成后端 `inductor_prims` RNG + seed/offset 融合 | `not config.fallback_random` | joint_graph.py:704；def replace_random.py:59 |
| `joint_custom_pre/post_pass`（钩子） | 用户自定义 joint pass | config.py:307-308 | 666 / 709 |

> `decompose_mem_bound_mm.py` **不是 joint pass**：它注册进 `construct_pattern_matcher_pass("decompose_mm_pass")`（post_grad 侧），joint 只借用了它的 `check_device`（joint_graph.py:39）。[verified]

### 3.3 post_grad（aten IR，切分后 · lowering 前）

三个有序桶 `pass_patterns = [PMP(), PMP(), PMP()]`（post_grad.py:85-89，注释「先 [0] 再 [1] 再 [2]」）；`register_lowering_pattern` 默认落 `pass_number=1`（888-906）。

| Pass | 干什么 | 类型 / 门控 | file:line |
|---|---|---|---|
| `reorder_for_locality` | 把生产者挪到消费者前（仅到首个 `copy_` 前）助融合 | 命令式；`is_inference and reorder_for_locality`（config.py:369） | 应用 161；fn 828 |
| `remove_profiler_ops` | 擦 `profiler._record_function_*`，不挡融合 | 命令式 | 应用 195；fn 92 |
| `group_batch_fusion_passes(pre_grad=False)` | post 级批/组融合（`batch_linear_post_grad` 等） | 命令式；`config.pattern_matcher` | 201-203；基类见 §3.4 |
| `remove_noop_ops` / `remove_assert_ops` | 去 no-op clone/alias；去 `_assert_tensor_metadata` | 命令式 | 204 / 205 |
| **`pass_pattern_{0,1,2}` 循环** | 三桶 PatternMatcherPass 依次 apply（下列 mm_plus_mm 等在此 fire） | **pattern-matcher** | 208-211 |
| `mm_plus_mm` | `add(mm(a,b), mm(c,d))` → 单核 `tuned_mm_plus_mm` | **lowering-pattern**（pass_patterns[1]）；extra_check 需 max_autotune | reg 946；kernel kernel/mm_plus_mm.py |
| `prepare_softmax` | softmax 前奏 `amax→sub→exp→sum` → `prepare_softmax_online` prim | replacement→pass_patterns[1]；`online_softmax`+triton | reg 813-825；模板 449-479 |
| `unfuse_bias_add` | `addmm` 拆回 `inp+(a@b)` 让 bias-add 融进逐元素 | graph-pattern（pass_patterns[2]） | 1741-1782 |
| `addmm`（再融合） | `add(mm(a,b),inp)` → `addmm` | graph-pattern（pass_patterns[2]） | 1809-1833 |
| `POST_GRAD_PATTERNS` 循环 | 用户 opt-in 的 split/cat/quant/`decompose_mm_pass` 桶 | **pattern-matcher**；`post_grad_fusion_options` | 216-237 |
| `decompose_mem_bound_mm` | 访存瓶颈的大 batch mm/bmm/addmm 拆成 broadcast-mul + sum（逐元素友好） | graph-pattern；需 `"decompose_mm_pass"` ∈ options | decompose_mem_bound_mm.py:224/243/267 |
| `b2b_gemm` | 背靠背 GEMM：`A@f(B@C)` 融合为自调优 Triton 模板 | **pattern-matcher**；`config.b2b_gemm_pass`（默认 False，config.py:293）；**无 observer 包裹** | 应用 238-239；pass obj b2b_gemm.py:36；handler 593 |
| `micro_pipeline_tp` | Async-TP：AG-matmul / matmul-RS 拆解重叠 | 命令式；`config._micro_pipeline_tp`（config.py:1168）；**无 observer** | 应用 241-242；fn micro_pipeline_tp.py:1261 |
| `fuse_ddp_communication` | DDP allreduce 通信 bucketing/融合 | 命令式；`config._fuse_ddp_communication` | 244-251 |
| `move_constructors_to_gpu` | 安全时把 CPU 构造张量搬上 GPU | 命令式；无条件 | 271-273；fn 2227 |
| `reinplace_inplaceable_ops` | 无后续 view 观察突变时重新原地化（index_put/scatter/functional collective/foreach） | 命令式；**保持最后**（引入突变） | 423-425；fn reinplace.py:1377 |
| `post_grad_custom_pre/post_pass`（钩子） | 用户自定义 post pass | config.py:300-301 | 167 / 253 |

> `fused_int_mm_mul`：`check_shape_cuda_and_fused_int_mm_mul_enabled`（post_grad.py:1878，门控 `config.force_fuse_int_mm_with_mul` config.py:376）在本 commit **无 `register_*` 引用，已成孤儿**；等效的 `_int_mm`+mul 融合现落在 quantization.py 的 smooth-quant 路径。[inferred]
> `config.decompose_mem_bound_mm`（config.py:1568）存在，但**不门控** `decompose_mem_bound_mm.py` 的 pattern（真正门控是 options 里的 `"decompose_mm_pass"`）。[inferred]

### 3.4 group_batch_fusion 的基类机制

`GroupBatchFusionBase`（group_batch_fusion.py:99）只定义 `match`/`fuse` 两个抽象方法；两个语义子类：`GroupFusion`（151，「任意形状，用 fbgemm.gmm 组融合」）与 `BatchFusion`（157，「同形状，用 bmm 批融合」）。注册用 `@register_fusion(name, pre_grad=...)`（116）填进 `PRE_GRAD_FUSIONS` / `POST_GRAD_FUSIONS`（112-113）两张表。驱动 `group_batch_fusion_passes`（1416）→ `generate_fusion_from_config`（1403，只挑 options 里已注册的名）→ 对每条规则 `apply_group_batch_fusion`（1355）：逆序遍历节点，`get_fusion_candidates` 收候选，`find_independent_subset_greedy`（1367，受 `min_fuse_set_size` 约束）贪心找互不依赖的子集，再 `rule.fuse`。这是「目录里 batch_* / group_* 那一排」背后的统一机制，区别于 §2.1 的 PatternExpr 声明式匹配。

---

## 4. 两种 pattern 匹配 与 pass 顺序约束
### 4.1 lowering-pattern vs graph/replacement-pattern

三阶段的所有 `PatternMatcherPass.apply` 都在 **FX 图 pass 阶段**完成匹配。真正的分野在**命中后何时、以何形态落地**：

- **graph-pattern / replacement-pattern**（`register_graph_pattern` / `register_replacement`）：命中即在 FX 图上改完——要么跑一段函数改图（`GraphPatternEntry`），要么贴入 traced 替换子图（`ReplacementPatternEntry`）。产物仍是 aten FX 图，后续正常 lowering。
- **lowering-pattern**（`register_lowering_pattern`）：命中时只在图上留一个 `call_function(handler,...)` 直通节点（`LoweringPatternEntry.apply` 1156）；handler 打了 `_inductor_lowering_function` 标记。真正产 Inductor IR 的动作**推迟到 `GraphLowering.call_function`**：遇到带该标记的 target 就直接 `target(*args, **kwargs)` 走「passthrough lowering」（graph.py:1372-1376），绕过普通 aten→IR lowering，一步生成融合 kernel 的 IR。`mm_plus_mm`、大量 mkldnn/量化融合走的正是这条——因为它们要产的是「一个自定义 Triton/外部 kernel」，而非另一串 aten 节点。

一句话：**匹配都在 post_grad（等）阶段，但 lowering-pattern 的「改写效果」延迟到 lowering 阶段兑现。**

### 4.2 顺序约束（为什么不能随便排）

- **桶内有序**：post_grad 的 `pass_patterns[0]→[1]→[2]` 严格顺序执行（注释 post_grad.py:84）；`[0]` 在 OSS 默认为空，由 freezing/早期量化填充。
- **规范化必须最前**：pre_grad 的 `normalization_pass` 是 split_cat 家族首个；joint 的 `canonicalize_aten_ir_passes` 标注「must occur before other passes」（joint_graph.py:663-664）——后续 pattern 依赖规范后的 schema。
- **依赖顺序**：joint 里 `auto_chunker` 必须在 `pad_mm` 之前（注释 684），免得 chunk 搜索还要处理 padding。
- **突变类保持最后**：post_grad 的 `reinplace_inplaceable_ops` 必须最后（注释 post_grad.py:421-422），因它引入 in-place 突变，会破坏之前 pass 依赖的「函数化」不变量。
- **每次改图后重排 + lint**：三 driver 收尾都 `stable_topological_sort` + `graph.lint()` + `recompile`，保证确定性与图合法。
- **引擎级护栏**：`PatternMatcherPass.apply` 逆拓扑序遍历（2383），且**丢弃跨突变区（2400）或跨 stream（2409）的匹配**——顺序 + 边界共同保证融合不越过 in-place 或多流边界。

---

## Related Pages

- [[19_torch_compile_end_to_end/00_pytorch_graph_series_index]] — 当前固定基线的图编译系统化课程入口
- [[fx_graph_construction_and_transformation_analysis]] — PatternExpr/候选桶/逆序匹配背后的 FX 数据结构，以及 DCE、保序和复杂度
- [[pre_grad_passes_guide]] — pre_grad 阶段逐 pass 细节
- [[joint_graph_passes_guide]] — joint 阶段（SDPA/pad_mm/常量折叠）细节
- [[post_grad_passes_guide]] — post_grad 阶段逐 pass 细节
- [[npu_vs_upstream_fusion_passes]] — 上游侧 ↔ NPU 侧融合 pass 对照
- [[vllm_ir_and_fusion_passes_analysis]] · [[sglang_compilation_passes_analysis]] — 工业界推理框架如何复用这套基础设施（post_grad 钩子）
- [[fx_pass_optimization_methodology]] — 从 npu/vllm/sglang/upstream 归纳的 pass 开发方法论
- [[02_compile_stack/04_inductor/index]] — lowering / Scheduler / codegen 下游各专题页导航
