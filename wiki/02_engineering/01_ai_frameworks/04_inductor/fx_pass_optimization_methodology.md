# FX Pass 优化开发方法论 — 从 upstream / torch_npu / vLLM / SGLang 四家现状归纳

> **Source baseline**: 归纳自四套已核源分析——upstream pytorch `9922478dffa`、torch_npu `b3c8a815b`(v2.7.1)、vLLM `97a98006b0`、SGLang `d6ef68881e`（均 2026-07-20 前后核验）
> **Dimension**: Methodology（跨源综合，非单源走读）
> 本页把 [[torch_upstream_pass_deepdive]]、[[npu_vs_upstream_fusion_passes]] / [[npu_fusion_passes_deepdive]]、[[vllm_ir_and_fusion_passes_analysis]]、[[sglang_compilation_passes_analysis]] 四份分析里反复出现的**共性与分歧**抽出来，归纳成一套「要不要做、在哪做、怎么做、怎么保证对、怎么可运维」的 pass 开发方法论。每条论断都能追回上述某页的已核 `file:line`（本页只给代表性引用，细节在各源页）。

---

## 1. 一条主线

**一个「pass 优化」本质是一个下注：「一段可被识别的子图，能被一段更便宜的等价子图（或一个更强的 kernel）替换」。** 工程上它永远拆成同样四个问题——

1. **在哪做（WHERE）**：落在哪个 IR 阶段 / 哪个钩子；
2. **匹配什么（WHAT）**：怎么声明「要识别的子图」；
3. **怎么落地（HOW）**：命中后改成什么形态、在哪一步兑现；
4. **怎么保证对（SAFE）**：靠什么不变式、护栏、门控保证等价与稳定。

四家现状的差异，几乎都能投影到这四个问题的不同选择上。下面先给一张全景对照，再逐问题归纳可复用的判据。

### 四家现状对照

| 维度 | upstream Inductor | torch_npu | vLLM | SGLang |
|---|---|---|---|---|
| **主引擎** | 声明式 `PatternMatcherPass`（自建） | 官方钩子 + 手工遍历（自建 `ascend_custom_passes` 注册表） | **复用** torch 的 `pattern_matcher` + `VllmInductorPass` 包装 | **fork vLLM 骨架，但抽空**融合层 |
| **落点** | pre/joint/post_grad + lowering | post_grad/pre_grad **custom 钩子**（仅推理为主） | post_grad_custom_post_pass 钩子 + pre_grad IR functionalization | post_grad_custom_post_pass 钩子（但 pass 为 no-op） |
| **融合朝向** | codegen（Triton/C++ 模板） | **厂商手工库 ACLNN** + Cube 模板/DVM | **厂商/手写 kernel**（FlashInfer/cutlass/symm_mem/AITER/`_C.*`） | 预融合 kernel + inductor 原生 `combo_kernels` |
| **真实图重写 pass 数** | 数十（三阶段全集） | 26 个自定义 + 后端融合 | ~16 融合 + ~7 IR/utility + 1 pre-grad | **0**（唯一的 fix_functionalization 是 no-op） |
| **新增的融合维度** | 算术/结构（GEMM/softmax/view-cat/conv-bn） | **硬件布局/dtype**（int64→int32、连续访存、Cube epilogue） | **集合通信 + 量化 + KV-cache 写入** | 无（下推到 kernel 层） |
| **代表页** | [[torch_upstream_pass_deepdive]] | [[npu_fusion_passes_deepdive]] | [[vllm_ir_and_fusion_passes_analysis]] | [[sglang_compilation_passes_analysis]] |

> 一句话读法：**upstream 造引擎、定义在哪三阶段落地；三家下游都从 `post_grad_custom_post_pass` 这个官方钩子接进去，然后各自决定「融合朝向什么」——这个选择直接决定了要不要写融合 pass（vLLM 写十几个、SGLang 一个不写）。**

---

## 2. 四个决策问题（可复用判据）

### Q1 · 落在哪一层？

上游把「改图」摊在五个可选落点上（[[torch_upstream_pass_deepdive]] §1 流水线），越早语义越高、越晚越规范：

| 落点 | IR 状态 | 适合放什么 | 代价 |
|---|---|---|---|
| **pre_grad** | Torch IR，未函数化/未规范化 | 高层语义融合（linear+permute、conv-bn）、batch 化 | 要自己处理别名与突变 |
| **joint** | aten，函数化后·切分前 | 跨 fwd/bwd 的融合（SDPA、pad_mm）、常量折叠 | 需 fake trace，训练/推理两版 pattern |
| **post_grad** | aten，切分后·lowering 前 | 设备/内存/通信优化、后端特化 | 已按 fwd/bwd 分开，shape 更具体 |
| **lowering** | aten→Inductor IR | 要「产一个自定义 kernel」的融合（lowering-pattern） | 只能产 IR，不能再改 aten 图 |
| **scheduler** | Inductor IR | 逐元素/规约的 kernel 级融合（can_fuse、epilogue） | 受 tiling/内存复用约束 |

**判据**：能在高层语义表达的融合尽量早放（信息最全）；依赖「已函数化、已规范化」的 pattern 放 joint/post_grad；**下游框架统一走 `post_grad_custom_post_pass` 钩子**（torch_npu/vLLM/SGLang 三家一致，[[torch_upstream_pass_deepdive]] §2.6）——因为这是唯一「不改上游源码就能插进优化管线」的官方插入点。第六个选择是**根本不做 FX pass**，把融合下推到 kernel + inductor `combo_kernels`（SGLang 路线，[[sglang_compilation_passes_analysis]] §3.3）。

### Q2 · 怎么声明匹配？

两条路线，判据清晰：

- **声明式 pattern**（`register_replacement` 从「例子函数」trace 出 pattern + 序列化缓存，或手写 `PatternExpr` 语法树）——**能用一对 `search_fn/replace_fn` 例子表达的融合走这条**。upstream 的 SDPA/pad_mm、vLLM 的全部融合 pass 都用它（vLLM 只在外面包 `VllmInductorPass` 做计时/uuid/计数，[[vllm_ir_and_fusion_passes_analysis]]）。优点：声明即正确、可序列化缓存避免每次 trace（[[torch_upstream_pass_deepdive]] §2.4）。
- **手工 `graph.find_nodes` 遍历**——**结构归一化、布局重排、无法用「一个子图例子」表达的改写走这条**。torch_npu 的 26 个 fold/融合 pass（[[npu_fusion_passes_deepdive]] §7）、vLLM 的 utility pass（noop_elimination/split_coalescing/scatter_split_replace/fix_functionalization）、SGLang 的 fix_functionalization 都是手工遍历。

> 一个反复出现的配套技巧：**为「归一化」单开前置 pass 给「融合」pass 铺路**。vLLM 的 `SplitCoalescingPass`/`ScatterSplitReplacementPass`、upstream 的 `normalization_pass`、torch_npu 的 `fold_sink_view` 都是——先把图整理成 pattern 能对上的规范形态，融合命中率才高。

### Q3 · 命中后怎么落地？

这是「两种 pattern 匹配」的真正分野（[[torch_upstream_pass_deepdive]] §4.1），判据由**「融合目标是什么」**决定：

| 落地形态 | 融合目标 | 代表 |
|---|---|---|
| **graph-pattern（即时改 FX 图）** | 另一串 aten 节点 | upstream `bmm_to_mm`；torch_npu 全部 fold |
| **lowering-pattern（推迟到 lowering 产 IR）** | 一个自定义 Triton/外部 kernel | upstream `mm_plus_mm`、b2b_gemm |
| **replacement（贴入 traced 子图）** | 一段等价子图 | upstream SDPA |
| **rewrite-existing-op（改写已有算子）** | 让已有 kernel 多干一步 | vLLM 把 quant 塞进 `unified_attention_with_output` 的 epilogue（[[vllm_ir_and_fusion_passes_analysis]]） |
| **fallback / 换手工算子** | 厂商手工库 | torch_npu 把 attention/通信 fallback 到 ACLNN；vLLM 换 FlashInfer/AITER |
| **push-to-kernel（不改图）** | 预融合 kernel + inductor combo | SGLang |

**核心洞察——融合「朝向什么」的谱系**：codegen（upstream 默认）→ 厂商手工库（torch_npu/vLLM，因硬件专用单元或厂商已调极致）→ 预融合 kernel（SGLang，干脆不写 pass）。**这条谱系解释了为什么同为推理框架，vLLM 写十几个融合 pass 而 SGLang 一个不写**：不是能力差异，而是「把融合放在编译层还是 kernel 层」的路线选择。

### Q4 · 怎么保证等价与稳定？

七条护栏，四家共享（细节见 [[npu_fusion_passes_deepdive]] §7.4 与 [[torch_upstream_pass_deepdive]] §4.2）：

1. **算子类别不变式**：只在共享某条不变式的一类算子内改写——view 类＝扁平序/元素数不变；masked＝掩码互补；bool×＝按位选择。**绝不跨类别改写**（不碰 permute/真广播这些会重排元素的算子）。这是「只改元数据、不重算数值」的根据。
2. **边局部改写 + 纯函数**：post-functionalization 的图是 SSA 式纯算子，`replace_input_with` 只动本节点入边，对 DAG 扇出天然安全（[[npu_fusion_passes_deepdive]] §7.3）。
3. **单用户门槛判据**：只改自己入边 → 不需要单用户；会「吃掉/改动前驱」→ 必须查单用户（torch_npu `fold_cat`/`fold_squeeze`）。
4. **引擎级边界护栏**：`PatternMatcherPass.apply` 逆拓扑序遍历，且**丢弃跨突变区、跨 stream 的匹配**（pattern_matcher.py:2400/2409）——保证融合不越过 in-place 或多流边界。
5. **meta 一等公民 + 静态 shape 门槛**：改写后必维护 `FakeTensor`（下游 lowering/tiling 靠它）；SymInt 动态维一律跳过。
6. **顺序约束**：规范化必须最前（`canonicalize_aten_ir_passes`）、依赖顺序（auto_chunker 在 pad_mm 前）、突变类保持最后（`reinplace_inplaceable_ops`）。
7. **收尾三件套**：`stable_topological_sort` + `graph.lint()` + `recompile`（+ DCE 删孤儿），保证确定性与图合法。

---

## 3. 工业界沉淀的工程护栏（可运维性）

「能融合」只是一半，「能上线、能调试、能回滚」是另一半。四家共同沉淀出这些：

- **uuid = 源码哈希 → pass 改动触发重编译**：torch_npu `get_hash_for_files`、vLLM/SGLang `hash_source`（inductor_pass.py）、upstream pattern 的序列化校验。改了 pass 逻辑而 cache 不失效会静默跑旧代码——这条是硬需求。
- **可禁用 / 可 bisect**：upstream `GraphTransformObserver` 支持按 subsystem（`"post_grad_passes"`）或按名禁用（`config.disabled_passes` / `CompilerBisector`）；torch_npu `SHUT_DOWN_FX_PASS_LIST=<name>`（含 `all`）；这是「怀疑某 pass 引入 bug/劣化」时的一键排查抓手。
- **可观测**：vLLM `VllmInductorPass` 提供计时（`time_and_log`）、全局 match 计数表、`dump_patterns`（把 Inductor pattern 反打成近似 Python）与前后图 dump；upstream `dynamo_timed(f"pass.{subsystem}.{name}")`。
- **门控分层**：总闸（`config.pattern_matcher`）→ 特性开关（`*_fusion_options` / `PassConfig`）→ 平台守卫（`is_cuda()`/`rocm_aiter_ops.is_enabled()`）→ 运行期尺寸（vLLM `is_applicable_for_range(compile_range)`、只在小 batch decode 的 token 区间触发重通信/KV 融合）。分层门控让一个融合能「按硬件、按特性、按 batch 尺寸」精确启停。
- **pattern 对 custom_ops 开关鲁棒**：vLLM `MatcherCustomOp.forward = custom if enabled else native`——同一 pattern 既能匹配「启用 custom op 的图」又能匹配「native 分解图」，避免因一个开关导致融合失配。
- **序列化 pattern 缓存**：upstream 把 SDPA 等 30 个 pattern 预 trace 落盘（`serialized_patterns/*.py`），运行期只 import 不 trace（[[torch_upstream_pass_deepdive]] §2.4）——从「例子函数 trace pattern」很贵，这是规模化的必要工程。

---

## 4. 开发一个新 pass 的流程（actionable checklist）

综合四家，落一个新融合 pass 的标准动作：

1. **定位可优化子图**：profiling 找「重复的小 kernel / 反复的 HBM 往返 / launch 密集 / 一个 kernel 结果立刻被下一个读回」。这是下注的依据，不是拍脑袋。
2. **Q1 选层**：能改上游就注册进对应阶段 pass_dict；不能改上游（下游框架）就挂 `post_grad_custom_post_pass` 钩子；若目标其实是「换个更强的 kernel」，考虑根本不写 pass（Q3 的 push-to-kernel）。
3. **Q2 选声明方式**：可用例子函数表达 → `register_replacement`（并考虑序列化缓存）；结构清理/布局重排 → 手工遍历；必要时先写一个前置**归一化** pass 铺路。
4. **Q3 选落地**：产 aten → graph-pattern；产自定义 kernel → lowering-pattern；让已有 kernel 多干一步 → rewrite-op；用厂商算子 → fallback/换 op。
5. **Q4 写不变式与 guard**：明确「我只在哪类算子内改、靠什么不变式等价」；加 dtype/shape/单用户/静态-shape 守卫；改写后 `propagate_fake_tensor`。
6. **收尾**：`stable_topological_sort` + `lint` + DCE。
7. **工程化**：`uuid`（源码哈希）+ 计时/计数 + 分层门控（config/平台/尺寸）+ 可禁用开关。
8. **验证**：数值等价（单测）+ **实测开关对比**（用 `SHUT_DOWN_FX_PASS_LIST` / config flag / bisect 关掉该 pass 跑 profiling）——因为几乎没有 pass 自带 benchmark 计数器，收益必须实测（见 §5）。

---

## 5. 反模式与诚实边界

- **效果多为结构性，真实收益要实测**：四家的 pass 里**几乎没有内建 benchmark/counter**（唯一的例外之一是 vLLM CATLASS-类 epilogue 与各 pass 的 match 计数）。「少 N 个 kernel / 少一次拷贝 / 转零拷贝」是结构性收益，不是加速比——上线前必须用「开/关该 pass」实测（[[npu_fusion_passes_deepdive]] §1 效果口径）。
- **搬框架容易，搬正确融合难**：SGLang 逐文件 fork 了 vLLM 的 pass 框架，却把融合内容整个抽空、`fix_functionalization` 沦为 no-op（[[sglang_compilation_passes_analysis]] §3.2）。**「有 pass 框架」≠「有融合优化」**——评估一个项目时要看 `configure()`/pass_dict 里实际挂了什么，而非目录里有多少文件。
- **pass 会腐化，以源码为准**：upstream 本 commit 里 `fused_int_mm_mul` 已成孤儿（无 register 引用）、`config.decompose_mem_bound_mm` 门控失效（[[torch_upstream_pass_deepdive]] §3.3）；torch_npu 的 `patch_pattern_mm_plus_mm` 是**删除**上游融合而非添加（[[npu_vs_upstream_fusion_passes]] §6）。民间说法/旧文档常与现状不符——**每条断言都要回源码那一行核**。
- **跨类别改写会错**：只在共享不变式的算子类内改；一旦碰会重排元素/改 stride 语义的算子（permute/transpose/真广播），「只改元数据」的等价前提就破了。
- **门控失配的静默坑**：pass 挂上了但因平台/尺寸/开关不满足而从不触发（vLLM 大量 `is_applicable_for_range`、平台守卫），或因 `custom_ops` 开关导致 pattern 失配——「pass 存在」不代表「pass 生效」，要用可观测手段确认命中数。

---

## Related Pages

- [[torch_upstream_pass_deepdive]] — 上游 Inductor pass 全集与 PatternMatcher 机制（本页 Q1–Q4 的引擎基线）
- [[npu_vs_upstream_fusion_passes]] · [[npu_fusion_passes_deepdive]] — torch_npu 侧：谁有谁无 + 26 个自定义 pass 的场景/问题/优化/效果 + 改图原语
- [[vllm_ir_and_fusion_passes_analysis]] — vLLM：复用 pattern_matcher、朝厂商 kernel 融合、通信/量化/KV-cache 三维度
- [[sglang_compilation_passes_analysis]] — SGLang：fork 骨架但不写融合、下推到 kernel 的反例
- [[scheduler_analysis]] — 后端 scheduler 融合（Q1 的 scheduler 落点）
