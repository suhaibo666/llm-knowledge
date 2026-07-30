# 知识库整改 P4:ai_frameworks 两级重组 + 19 号解散 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `01_ai_frameworks` 从 18 个平铺目录重组为 5 个架构层两级目录;19 号目录(63 篇/2.05 万行)解散——A 卷删、其余按卷分发进功能树并替换旧大文;labs 迁出 wiki;全程 broken=0。

**Architecture:** 先纯移动(git mv 16 目录 + 脚本改写 604 条路径限定链接,一次 commit 可整体回退),再按 A→E→B→D→C→F 卷分批编辑解散(每批走 P3 已验证的"台账+逐句判重+对抗审查"规程)。

**Tech Stack:** git mv / python 批量链接改写脚本 / tools/check_links.py 门禁 / P3 编辑惯例。

**Spec:** `docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md` §2(映射表)、§3.1(卷处置表);侦察数据:本文附录 A(逐文件映射定稿,含 3 个 spec 缺口的补齐决策)。

**编辑总规则(P3 沉淀,每个编辑任务适用):** 只搬运不新造;判重必须找到承接页实际对应段落;被并方独有事实逐字落地;冲突留双+`[!contradiction]`;无源可疑声明转 `[!todo]` 待核验而非删除;量化/断言措辞不得升级认识论地位;mermaid 过 CLAUDE.md 检查单+activate 栈平衡(能 `npx @mermaid-js/mermaid-cli` 实渲则实渲);裸基名链接默认、歧义才路径限定、禁 `../`;逐节处置台账为审查依据;每任务 checker broken=0 后 commit。

---

## Task 1: 开分支 + 基线

- [ ] `git checkout main && git pull`;确认 HEAD ≥ `0908b23`、工作区干净
- [ ] `git checkout -b reorg/p4`
- [ ] `python tools/check_links.py` 记录基线(应 pages=398 broken=0 ambiguous=70 bare_index=70 orphans=1);`python -m pytest tools/ -q`(8 passed)

## Task 2: 两级重组纯移动(一个大 commit,内容零改动)

**目录映射(spec §2 定稿):**

| 旧(01_ai_frameworks/ 下) | 新(01_ai_frameworks/ 下) |
|---|---|
| 00_tensor_and_storage | 01_eager_runtime/01_tensor_and_storage |
| 01_dispatcher_and_device | 01_eager_runtime/02_dispatcher_and_device |
| 07_op_registration | 01_eager_runtime/03_op_registration |
| 11_aten_op_execution | 01_eager_runtime/04_aten_op_execution |
| 10_eager_autograd | 01_eager_runtime/05_autograd_engine |
| 12_nn_module_system | 01_eager_runtime/06_nn_module_system |
| 13_runtime_memory_amp_profiler | 01_eager_runtime/07_memory_amp_profiler |
| 02_dynamo | 02_compile_stack/01_dynamo |
| 03_aot_autograd | 02_compile_stack/02_aot_autograd |
| (新建) | 02_compile_stack/03_graph_ir_and_passes |
| 04_inductor(含 npu/) | 02_compile_stack/04_inductor |
| 05_codegen_backends | 02_compile_stack/05_codegen_backends |
| 17_compile_cache | 02_compile_stack/06_compile_cache |
| (新建) | 02_compile_stack/07_debugging |
| 06_graphs | 03_runtime_graphs |
| 14_fx_export_and_extensibility | 04_export_and_distributed/01_fx_export_extensibility |
| 15_distributed_primitives | 04_export_and_distributed/02_distributed_primitives |
| 09_other_frameworks | 05_other_frameworks |
| 08_kernel_optimization | (迁出)02_engineering/05_gpu_kernel/(3 md 并入,目录删除) |
| 19_torch_compile_end_to_end | 原地不动,Task 3-10 解散(19 号在移动 commit 中保持原路径,避免双重改链) |

- [ ] **Step 1:** 按表逐条 `git mv`(16 个平移目录 + 08 的 3 个 md 到 05_gpu_kernel);labs 整目录 `git mv wiki/.../19_torch_compile_end_to_end/labs tools/labs_torch_compile`(152 跟踪文件;`.gitignore` L126-131 的 6 行 labs 路径同步改写为 tools/labs_torch_compile/...;物理移动被 ignore 的 artifacts 残留;清除 2 个空 children 目录)
- [ ] **Step 2:** 写一次性脚本 `tools/tmp_rewrite_paths.py`:对 wiki/**/*.md(含 courses 未来目录)与 docs/superpowers/specs|plans 中的**路径限定** `[[...]]` 链接与行内路径引用,按上表做前缀替换(19 号前缀不动);处理 `06_graphs`→`03_runtime_graphs` 等 604 条(侦察 §4.1 计数)。脚本打印每文件替换数;跑后人工抽查 10 处。
- [ ] **Step 3:** 各新架构层目录建壳 index.md(5 个,一句话+子目录表);被移动目录的旧 index.md 随目录走,内部自引路径由脚本已改;`01_ai_frameworks/index.md` 重写为 5 层导航(旧 18 行表替换,分层叙述保留但对齐新路径)
- [ ] **Step 4:** `python tools/check_links.py` **broken 必须=0**(此步是本计划最大风险点,不为 0 逐条修完再提交);`rm tools/tmp_rewrite_paths.py`
- [ ] **Step 5:** Commit `refactor: regroup 01_ai_frameworks into 5 architecture layers (pure moves)`;`git status` 干净

## Task 3: A 卷删除(5 篇 1790 行)

- [ ] 对 a01-a05 逐篇:checker --json 找入链(内部入链多来自 19 号索引,外部少量)→ 外部入链改指现存功能页(a01→`01_eager_runtime/01_tensor_and_storage/` 对应页、a02→`02_dispatcher_and_device`+`04_aten_op_execution` 对应页、a03/a04→`02_compile_stack/01_dynamo/index`(其前置概念由 b03/b04 承接)、a05→`01_dynamo/index`);课程页 Task 10 建成后可再优化指向,本任务只保证不断链。19 号索引内的入链本任务不动(索引页 Task 10 整体处置)
- [ ] **删除前逐篇过独有性**:A 卷是"基础回顾",但按 P3 教训不可盲信——每篇 grep 3-5 个核心术语到对应 eager_runtime 页确认覆盖;发现独有事实(如编译器视角的耗时模型 a05)先逐字搬到对应页或标注给课程页,台账记录
- [ ] `git rm` 5 篇;checker=0;commit `docs(compile): drop volume A recaps, redirect to eager_runtime modules`

## Task 4: E 卷迁入 07_debugging(9 篇)+ 删旧 debug 页

- [ ] `git mv` e01-e09 → `02_compile_stack/07_debugging/`(去 e 前缀重命名为规范名,如 `e01_observability_...` → `observability_logs_counters_analysis.md`,保 `_analysis` 后缀);建该目录 index.md
- [ ] `Pytorch_Compile_Debug_Analysis.md`(558)逐节判重 vs E 卷(它是 E 卷的压缩前身;其分布式排查脚本/决策树若 E 卷无对应,逐字搬入 e04 或 e09 对应新页)→ 台账 → `git rm`
- [ ] 入链修复;checker=0;commit

## Task 5: B 卷迁入 01_dynamo + 旧 Dynamo 大文处置

- [ ] `git mv` b01-b10 → `02_compile_stack/01_dynamo/`(去 b 前缀规范重命名)
- [ ] `PyTorch_Dynamo_Technical_Analysis.md`(2018)逐节判重 vs B 卷 10 篇(台账逐节,预期大部重复;独有段逐字并入对应 b 篇)→ `git rm`
- [ ] `torch_compile_source_analysis.md`(593)vs b01/b02 同规程合并删除;`control_flow_capture_analysis.md`(204)vs C06+b04 判重(C06 在 Task 7 才迁入,故本任务只做 vs b04 部分,C06 部分留 Task 7 收尾);`dynamo_pass_methodology.md`(152)vs b10 合并
- [ ] 01_dynamo/index.md 重建;入链修复;checker=0;commit

## Task 6: D 卷分发(7 篇)

| 篇 | 去向与合并对象 |
|---|---|
| d01(335) | `04_inductor/`,与 `inductor_compiler_pipeline_analysis.md`(921)二选一:以 d01 为骨架(基线新)吸收 921 页独有段后删 921 页 |
| d02(315) | `02_aot_autograd/`(runtime wrapper 族)——spec 缺口补齐 |
| d03(211) | `04_inductor/`(async compile)——缺口补齐 |
| d04(219) | 改写为 `06_compile_cache/index.md` 的 overview 主体(原 index 91 行导航保留在尾部) |
| d05(224) | `04_inductor/`,进入内存归一簇(Task 8 与 C19 一并归一,本任务先平移) |
| d06(320)+f08(322) | `03_runtime_graphs/cuda/`,与 `PyTorch_CUDA_Graphs_Complete_Guide` 对应节(方式2/综合比较)判重合并——注意 P3 已把 Guide 改为权威页,以 Guide 为主干吸收两篇独有内容后两篇删除或降为专题页(按独有量定,台账说明) |
| d07(264) | `07_debugging/`(runtime failure 族)——缺口补齐 |

- [ ] 逐篇执行(每篇台账);重命名去前缀;入链修复;checker=0;commit

## Task 7: C 卷第一批 → 03_graph_ir_and_passes(12 篇:02,03,05,06,07,08,11-16)

**缺口补齐:C07(捕获前端四路径)、C08(规范化/Decomp/Functionalization)均入本目录**(C07 讲四种前端对比属 IR 入口;C08 属规范化 pass)。

- [ ] `git mv` 12 篇入新目录(去数字前缀规范重命名,如 `02_fx_graph_core_data_model.md` → `fx_graph_data_model_analysis.md`;重命名映射在 commit message 列全)
- [ ] **归一旧页(本任务最重编辑,分四组,每组一 commit):**
  1. `fx_graph_construction_and_transformation_analysis.md`(601,19 条外链在身)vs C02/03/05/12/13/14/16:逐节台账;它已自带「阅读定位与迁移去向」节——按其自述拆解,独有段(尤其源码导航/排查清单)逐字分插对应 C 篇,外链逐条改指,删除
  2. pass 方法论组:`fx_pass_optimization_methodology`(349)+`torch_upstream_pass_deepdive`(232)vs C13/15/16 归一(方法论只留一处)
  3. `decomposition_passes_guide`(163)vs C08 归一
  4. `control_flow_capture_analysis` 的 C06 部分收尾(Task 5 遗留)
- [ ] 三份 stage guide(`pre/joint/post_grad_passes_guide`,2702 行)**不并入本目录**——它们是 Inductor 阶段实操指南,留在 `04_inductor/` 且在 Task 8 与 C15 划界(C15 讲通用流水线,三 guide 讲各阶段 pass 清单,补互指)
- [ ] 建 index.md;checker=0

## Task 8: C 卷第二批(9 篇:04→01_dynamo;09,10→02_aot_autograd;17-21→04_inductor)

- [ ] **C04 + 动态形状归一**:C04(415)入 `01_dynamo/`;与 b09(224)、`dynamic_shapes_full_analysis`(460)、`unbacked_symint_analysis`(368)、`inductor_codegen_dynamic_shape_analysis`(352)按 spec 定稿归一为三页:概念权威页(C04 为骨架吸收 dynamic_shapes_full 独有)+ unbacked 专项(保留)+ codegen 专项(保留,补互指);b09 与 C04 划界(Dynamo 侧行为 vs 符号系统)
- [ ] **C09/C10 vs `aotautograd_analysis.md`(1127)**:逐节台账,C09/C10 为骨架吸收 1127 页独有段(12 条外链改指)后删除;`aot_autograd_quickstart` 保留
- [ ] **C17 vs `lowering_analysis.md`(449)**:二选一,C17 骨架吸收后删 449 页
- [ ] **C18 vs `PyTorch_Inductor_Technical_Analysis.md`(1699)IR 节;C20 vs `scheduler_analysis.md`(964);C19+d05 vs 内存两页(278+208)+1699 页内存节;C21 vs codegen 四碎片(259+98+89+115)+`codegen_extension_guide`(223)**:逐组台账归一——1699 页在其 IR/融合/内存节被吸收后,剩余节(后端注册/常量折叠/自定义融合规则教学(含 [!todo] 隔离区))评估:仍有独有价值的节拆为小专题页或并入对应新页,空壳后删除;scheduler_analysis 独有的 §7 融合算法详解/§9 自定义指南逐字并入 C20
- [ ] 每组一 commit;checker=0

## Task 9: F 卷分发(6 篇:f01-f07 除 f08)

| 篇 | 去向 |
|---|---|
| f01(316) | `01_eager_runtime/05_autograd_engine/`,与 `autograd_engine_analysis`(481)互补划界(eager 引擎 vs compiled autograd),补显式分工声明 |
| f02(226) | `02_aot_autograd/`(recompute 族,与 C10 互指)——缺口补齐 |
| f03(230)/f04(221) | `04_export_and_distributed/02_distributed_primitives/`,与 `c10d_ddp_fsdp_dtensor_analysis`(433)划界(原语 vs compile 边界)补互指 |
| f05(255) | 与 `01_fx_export_extensibility/fx_graph_export_and_custom_ops_analysis`(315)判重合并 |
| f06(260) | 与 `02_dispatcher_and_device/privateuse1_device_integration_analysis`(278)+`codegen_extension_guide` 相关段判重合并 |
| f07(292) | `04_inductor/`(AOTInductor 打包部署)——缺口补齐 |

- [ ] 逐篇台账;重命名;入链;checker=0;commit

## Task 10: 课程页 + 19 号删除 + 全 index 重建 + 阶段门

- [ ] 创建 `wiki/courses/torch_compile_end_to_end.md`:纯导读(阅读顺序+链接+每篇一句话),吸收两个 00 索引的主线叙述与 C01(372,动机)的导读价值、`torch_compile_architecture.md`(151)的 overview 段;四页(两索引+C01+architecture)删除;19 号 `index.md`(孤儿)删除;19 号目录应已空,`git rm -r` 收尾
- [ ] `wiki/courses/` 建 index?(不建,courses 页少;wiki/index.md 加 courses 入口行)
- [ ] 全量 index 重建:5 个架构层壳 index、各模块 index(移动后自动继承的旧 index 逐个过一遍条目/路径/日期)、`01_ai_frameworks/index.md` 定稿、`wiki/index.md` 计数与结构行更新
- [ ] **阶段门**:pytest 8 passed;checker broken=0、orphans 不高于基线;`find wiki/02_engineering/01_ai_frameworks -name "*.md" ! -name "SUPERSEDED.md" | wc -l` 记录终值(预期 ~174−63+新增课程外迁移净变化,报告实数);changelog 阶段条目;merge 回 main(`--no-ff`)+push;roadmap 标 ✅
- [ ] **P6 待办登记**(roadmap):`04_inductor/npu/` 三页(npu_lowering_guide 884/npu_fusion_passes_deepdive 425/npu_vs_upstream 287)与 C17/C15 新页的划界

---

## 附录 A: spec 三缺口的补齐决策(本计划定稿)

1. C07 → `03_graph_ir_and_passes`(四种捕获前端对比,IR 入口视角);C08 → `03_graph_ir_and_passes`(规范化 pass 族,与 decomposition_passes_guide 归一)
2. d02→`02_aot_autograd`;d03→`04_inductor`;d05→`04_inductor`(内存簇);d07→`07_debugging`;f02→`02_aot_autograd`;f07→`04_inductor`
3. `04_inductor/npu/`(11 篇 6040 行)本阶段**只随目录平移**,与上游新页的划界记 P6 待办

## 附录 B: 风险与回退

- Task 2 是全计划最大单点(327 文件移动+604 链接改写):纯移动零内容改动,单 commit 可 `git revert`;checker=0 是硬闸
- 编辑任务(3-9)沿 P3 惯例逐对 commit,单对可回退
- 并行会话风险:P4 跨度长,开工前确认 main 无未拉取提交;每完成 2-3 个 Task 考虑 rebase main 检查漂移
