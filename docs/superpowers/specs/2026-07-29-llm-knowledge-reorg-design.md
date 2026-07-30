# llm-knowledge 知识库结构整改设计

> 日期:2026-07-29
> 状态:已经用户逐节确认
> 前置盘点:两份代理报告(结构盘点 + 内容重复专项),关键数据已内联本文,无需外部引用

## 0. 背景与目标

知识库现状:wiki 400 篇 / 12.5 万行,raw 129 个源文件。用户痛点与确认的整改约束:

| 决策项 | 结论 |
|---|---|
| 主要使用场景 | **人读为主**(Obsidian 浏览),agent 检索为辅 → 目录对人直观、链接克制 |
| 核心痛点 | **页面间内容重复**(同主题多目录并存,不知看哪个、更新哪个) |
| 破坏性尺度 | **允许合并与删页**(git 历史可追溯),同步修订 CLAUDE.md 的 "Never delete" 规则 |
| 权威归属模型 | **功能分类树为权威,课程/学习域降为纯导读页**(courses/ 层,无正文只有链接) |
| 工作产物处置 | **分类治理**:轻量流程文档留 docs/,重量审计产物删除+gitignore,图表源入库 tools/ |
| ai_frameworks 目录分组 | **按架构三层收敛为 5 个两级目录**(2026-07-29 追加确认):现 18 个平铺编号目录的编号顺序与逻辑分层(eager 地基/编译栈/图/扩展)脱节,重组为 5 个架构层目录、模块降一级并局部重编号 |

成功标准:

1. 一个主题在全库只有一个权威页;13 组高重叠全部消除
2. 坏链清零(现 160 处),`[[index]]` 裸引用清零(现 71 处)
3. 功能树是唯一内容入口;两条学习路线保留为 courses/ 导读页
4. 仓库工作区瘦身约 60MB;134 张 png 图表恢复可再生
5. CLAUDE.md 规则修订后,"新学习域重讲旧内容"这一复发根因被规则堵死

## 1. 现状诊断(整改依据,含数据锚点)

### 1.1 规模与失真

- wiki 实际 400 篇,README 声称 102 篇(差 4 倍);`wiki/index.md` 目录树含 3 个不存在的目录(`cudagraphs/`、`inductor/`、`mlir/`),页面计数最大偏差 −133,漏 `01_theory/06_distributed_parallelism` 域。
- `02_engineering/01_ai_frameworks/` 178 篇占全库 44.5%;其中 `19_torch_compile_end_to_end/` 单目录 63 篇 / 2 万行,内部三个索引并存(约定的 `index.md` 只覆盖 2/62 且零入链)。
- 编号跳空:`01_ai_frameworks` 下缺 16、18(历史合并入 19 号后未回填)。

### 1.2 五大重复组(内容重复专项结论)

| # | 重复组 | 卷入规模 | 关键事实 |
|---|---|---|---|
| 1 | `19_torch_compile_end_to_end/`(63 篇)vs `01_ai_frameworks` 旧树 | ~110 页 / 3.5 万行 | 新系列基线更新(PyTorch `e8f97c1a`)、粒度更好;旧大文未动:`PyTorch_Dynamo_Technical_Analysis`(2018 行)、`PyTorch_Inductor_Technical_Analysis`(1705 行)、`aotautograd_analysis`(1141 行)、`Pytorch_Compile_Debug_Analysis`(558 行) |
| 2 | 后训练三域(`01_theory/04_posttraining` 15 篇、`02_engineering/04_posttrain_frameworks` 14 篇、`03_posttraining` 14 篇) | ~20 页 / 4500 行 | GRPO/DAPO/GSPO 三写(论文页/D02/verl core_algos);verl 一次迭代双写且基线不同(`8a694930` vs `983cb0f`);weight sync 四写(含 megatron-lm 两页) |
| 3 | `06_graphs/` 目录内 | 12 页 / 8900 行 | cuda 侧 README 与 Complete_Guide H1 一字不差;Timing_Diagrams 与 Guide 同一四段结构;npu 侧 Graph Tree 双写 4095 行(`torch_compile_npugraphs_deep_dive` 2397 行 + `npugraphs_memory_reuse_analysis` 1698 行,后者留有"合并自 memory_management"痕迹);`aclgraph_deep_analysis` 与 deep_dive 附录 A 重复 |
| 4 | Megatron PP 三页 | 1740 行 | `02_train_frameworks/megatron_pp_parallelism_analysis`(740,目录错位)↔ `megatron-lm/megatron_pp_schedulers_analysis`(771,有 commit 基线 `ee3f1ff`)↔ `megatron_pp_supplements_analysis`(229) |
| 5 | 顶层横向页复述子页机制 | ~18 页 / 6500 行 | 通信掩盖(顶层 1+框架 3)、分布式优化器(顶层 1+Megatron 3)、Ring Attention 四写、FSDP 六处、Roofline 四写(`08_kernel_optimization/operator_optimization_guide` 834 行目录错位) |

另 7 组中重叠(需划界+补链,不需大合并):vLLM compilation↔upstream graphs、sglang↔vllm passes(健康范本)、TIM 因果链分层、D05↔sandbox/infra 页等。

**复发根因**:`03_posttraining/index.md` 与 `19_.../index.md` 用相同措辞承诺"旧页面保持原位,通过链接复用,不迁移、不复制",但正文均实质重讲旧页 → 必须在规则层堵死(见 §6)。

### 1.3 链接健康度

- 总链接 5787 条,均 14.5 条/页。
- 坏链 160 处/28 个目标:`[[correction_report]]` 113 处(目标在 `docs/audits/`,vault 外,全死链);示例文本误解析 16;末尾反斜杠 9;乱码 6;真实缺页 4(`scaling_laws_for_transfer_analysis`、`Kimi VL`、`Kimi Audio`、`llm_parallelism_analysis`);代码签名误解析 3;跨层+其他 9。
- `[[index]]` 歧义:56 个同名 index.md 承接 1015 条链接(17.5%),71 条为裸 `[[index]]`。
- `changelog.md` 2664 行/497 条链接,是 14 个假坏链目标的唯一来源(示例文本未转义)。
- Related Pages:65 篇缺失(16.3%),集中在 triton/(9)、deepseek/(8)、zhipu_glm/(7)——非"堆砌"而是缺失,规则本身需修订。
- 真孤儿页 2:`19_.../index.md`、`19_.../labs/NATIVE_BACKEND_RUNBOOK.md`。

### 1.4 层次错位与工作产物

- `docs/audits/` 55MB 审计产物已入 git(单 jsonl 最大 17MB);`docs/reports/` 一个 2.9MB docx 与 wiki 页面同主题重复。
- `.html2md/`(gitignored,52MB 主要是 node_modules)内含 wiki 全部 134 png/25 svg 的图表源 html + 转换脚本 → 图表现状不可再生。
- `torch_compile_debug/` 5 组 PyTorch 调试日志残留(2026-07-26)。
- `raw/_ingest/` 3 个文件是摄入施工单/脚本/changelog 草稿,非源材料;`raw/02_engineering/wanka_determinism_reliability_deep_analysis.md` 是分析页误放 raw。
- raw/wiki 镜像名存实亡:56 个 wiki 目录仅 15 个与 raw 对齐;75% wiki 目录的源是外部代码仓库;32 篇论文(OpenAI 11/Qwen 8/Google 6/Anthropic 3/MiniMax 3/Llama 1)未消化(本次不处理,见 §8)。

### 1.5 命名

- 7 套后缀并存:`_analysis/_guide/_quickstart/_deepdive/_deep_dive/_report/_methodology/_overview/_map/_model/_concepts/_details/_diagrams/_v2` 等,81 篇偏离约定(其中 19 号 21 篇纯序号页)。
- 17 篇非 snake_case(大写/点号/驼峰);wiki 内 4 个 README.md(规范外类型);`megatron-lm/` 目录连字符与同级不一致(本次不改目录名,影响面大收益小)。

## 2. 目标目录骨架

```
llm-knowledge/
├── raw/                        # 仅源材料;放弃 raw↔wiki 镜像约定
│   ├── 01_theory/              # 结构不变
│   └── 02_engineering/         # wanka_*.md 迁回 wiki/02_engineering/07_training_reliability/
├── wiki/
│   ├── index.md                # 只留域级表格(入口+一句话+状态),不画深层 ASCII 树、不写精确页数
│   ├── changelog.md            # 当季;历史归档 wiki/changelog/2026Q2.md 等
│   ├── 01_theory/
│   │   ├── 01_models/          # deepseek/ moonshot_kimi/ zhipu_glm/ meituan_longcat/ tencent_hunyuan/ thinking_machines/
│   │   ├── 02_pretraining/
│   │   ├── 03_sft/             # 空目录保留(标注待建设)
│   │   ├── 04_posttraining/    # 吸收 D02/D03/D04(算法演进为该目录权威演进页)
│   │   ├── 05_inference/       # 空目录保留
│   │   └── 06_distributed_parallelism/   # 新增 Ring Attention/CP 通用机制权威页
│   ├── 02_engineering/
│   │   ├── 01_ai_frameworks/   # 重组为 5 个架构层目录(两级);19 号解散
│   │   │   ├── 01_eager_runtime/           # 地基:tensor_storage/dispatcher_device/op_registration/aten/autograd/nn_module/memory_amp_profiler
│   │   │   ├── 02_compile_stack/           # 编译栈:dynamo/aot_autograd/graph_ir_and_passes(新,C卷)/inductor/codegen_backends/compile_cache/debugging(新,E卷)
│   │   │   ├── 03_runtime_graphs/          # cuda/ + npu/(原 06_graphs)
│   │   │   ├── 04_export_and_distributed/  # fx_export_extensibility + distributed_primitives
│   │   │   └── 05_other_frameworks/        # MindSpore 对照(原 09)
│   │   ├── 02_train_frameworks/    # 顶层横向页收缩为对比矩阵;megatron_pp_parallelism 并入 megatron-lm/
│   │   ├── 03_infer_frameworks/
│   │   ├── 04_posttrain_frameworks/  # 吸收 D05/D06/D08/D09/D10/D11;verl/ 吸收 D07
│   │   ├── 05_gpu_kernel/            # 吸收 operator_optimization_guide(Roofline 归一)
│   │   ├── 06_auto_parallel/
│   │   └── 07_training_reliability/  # 吸收 wanka_*.md
│   └── courses/                # 纯导读层:只有阅读顺序+链接+每篇一句话导读,禁止正文
│       ├── torch_compile_end_to_end.md
│       └── posttraining_frontier.md
├── docs/
│   ├── superpowers/{specs,plans}/   # 保留
│   ├── research/                    # 保留(源账本);raw/_ingest 3 文件迁入此处
│   └── (audits/ reports/ batch_invariance_demo.py 从工作区删除,git 历史可追溯)
├── tools/                      # 新建
│   ├── check_links.py          # Obsidian 链接解析:坏链/裸 index/孤儿页检查
│   ├── figs/                   # 自 .html2md/figs + deepep_figs 迁入(html 源+figstyle.css)
│   └── html2md/                # 转换脚本 convert*.mjs fix_links.mjs gen_pp_fig.mjs(不含 node_modules)
└── CLAUDE.md                   # 按 §6 修订
```

**01_ai_frameworks 旧目录 → 新位置映射**(链接修复与实施计划的依据):

| 旧目录 | 新位置 |
|---|---|
| `00_tensor_and_storage` | `01_eager_runtime/01_tensor_and_storage` |
| `01_dispatcher_and_device` | `01_eager_runtime/02_dispatcher_and_device` |
| `07_op_registration`(含 npu/) | `01_eager_runtime/03_op_registration` |
| `11_aten_op_execution` | `01_eager_runtime/04_aten_op_execution` |
| `10_eager_autograd` | `01_eager_runtime/05_autograd_engine` |
| `12_nn_module_system` | `01_eager_runtime/06_nn_module_system` |
| `13_runtime_memory_amp_profiler` | `01_eager_runtime/07_memory_amp_profiler` |
| `02_dynamo` | `02_compile_stack/01_dynamo` |
| `03_aot_autograd` | `02_compile_stack/02_aot_autograd` |
| (新建) | `02_compile_stack/03_graph_ir_and_passes`(吸收 C 卷 FX 数据模型/改图/pattern/pass 流水线篇) |
| `04_inductor`(含 npu/) | `02_compile_stack/04_inductor` |
| `05_codegen_backends`(mlir/) | `02_compile_stack/05_codegen_backends` |
| `17_compile_cache` | `02_compile_stack/06_compile_cache`(d04 变本目录 overview) |
| (新建) | `02_compile_stack/07_debugging`(吸收 E 卷 9 篇) |
| `06_graphs`(cuda/ npu/) | `03_runtime_graphs` |
| `14_fx_export_and_extensibility` | `04_export_and_distributed/01_fx_export_extensibility` |
| `15_distributed_primitives` | `04_export_and_distributed/02_distributed_primitives` |
| `09_other_frameworks` | `05_other_frameworks` |
| `08_kernel_optimization` | 删除,内容迁 `02_engineering/05_gpu_kernel/`(§3.5) |
| `19_torch_compile_end_to_end` | 解散(§3.1) |

C 卷 21 篇的目标目录补充说明:严格属于 Dynamo 符号形状的(04)入 `01_dynamo/`;AOT joint graph/重算(09、10)入 `02_aot_autograd/`;FX 数据模型/改图原语/pattern/pass 流水线/合法性(02、03、05、06、11–16)入新建 `03_graph_ir_and_passes/`;Inductor IR/内存/scheduler/codegen(17–21)入 `04_inductor/`;01(动机与分类)并入课程导读页。逐篇映射与合并对象在实施计划中定稿。

## 3. 去重迁移策略

总原则:**一主题一权威页**;新旧重叠时保留"基线更新、粒度更好"的版本,被并方删除;所有删除/改名经链接修复,提交前坏链为零。

### 3.1 组 1:19 号目录解散(最大迁移)

| 卷 | 动作 |
|---|---|
| A 卷(a01–a05 基础回顾) | **删除**。内容已存在于 `01_eager_runtime/` 各模块;课程页直接链功能树对应页 |
| B 卷(b01–b10 Dynamo) | 迁入 `02_compile_stack/01_dynamo/`;**删除** `PyTorch_Dynamo_Technical_Analysis.md`(2018 行);`dynamo_quickstart` 保留;`control_flow_capture_analysis`、`dynamo_pass_methodology` 与 B 卷对应篇合并 |
| C 卷(01–21 图编译) | 按 §2 映射分发;替换/合并旧页:动态形状四写归一(概念页+unbacked 专项+NPU 特化各一)、内存规划四页归一、`scheduler_analysis`(964)与 20 二选一、`lowering_analysis`(445)与 17 二选一、`inductor_compiler_pipeline_analysis`(921)与 d01 二选一、pass 六页(3453 行)按 pre/joint/post 阶段归一 |
| D 卷 | d01 入 `02_compile_stack/04_inductor/`(与 `inductor_compiler_pipeline_analysis` 二选一);d04 改写为 `02_compile_stack/06_compile_cache/` 的 overview;d06(cudagraph_trees)与 f08(训练/推理 cudagraph+freezing)迁入 `03_runtime_graphs/cuda/`,与 Complete_Guide 对应节合并 |
| E 卷(e01–e09 debug) | 迁入新建 `02_compile_stack/07_debugging/`;**删除** `Pytorch_Compile_Debug_Analysis.md`(558 行) |
| F 卷 | f01(compiled autograd)入 `01_eager_runtime/05_autograd_engine/` 与旧页互补划界;f03/f04 入 `04_export_and_distributed/02_distributed_primitives/`(讲"与 compile 的边界",与原语页显式分工);f05/f06 与 `01_fx_export_extensibility`/`02_dispatcher_and_device` 对应页合并 |
| 三个索引 + labs/ | 索引内容并入 `courses/torch_compile_end_to_end.md`;labs/(含 NATIVE_BACKEND_RUNBOOK、demo py 脚本、artifacts 空目录)迁到 `tools/labs_torch_compile/`,artifacts 空目录清除 |
| 旧枢纽页 | `torch_compile_architecture.md`(158,overview 四写之一)并入课程导读页后删除;`torch_compile_source_analysis.md`(593)与 b01/b02 合并 |

### 3.2 组 2:后训练三域整合

| 页面 | 动作 |
|---|---|
| D02 算法演进 | 迁入 `01_theory/04_posttraining/`,成为演进主线权威页;`grpo/dapo/gspo_analysis` 瘦身为论文特有内容(元数据、原始实验数字、原文细节),公式/动机指向 D02;D02 §3.6(K3-MOPD)收缩为一句+链接 D12 |
| D03(agentic RL)、D04(staleness) | 迁入 `01_theory/04_posttraining/`;D04 §7 与 `tim_causal_chain_analysis` 补双向链(良性分层保留) |
| D05(infra 三平面) | 迁入 `02_engineering/04_posttrain_frameworks/`;§7 sandbox、§4 backpressure 收缩为接口视角+外链 `rl_sandbox_design`/`rl_infra_efficiency` |
| D06 框架对比 | 迁入 `04_posttrain_frameworks/`,§4.1 verl 段压缩为矩阵一行 |
| D07 verl 端到端 | 迁入 `verl/`,以新基线 `983cb0f` 为准;与 `verl_ray_trainer_analysis` 重叠部分以 D07 为准合并;§3 DataProto/§6 权重刷新收缩为契约表+时序,细节外链 `verl_dataproto`/`verl_rollout_resharding` |
| D08/D09/D10/D11 | 迁入 `04_posttrain_frameworks/`(slime/AReaL/ROLL/CUDA-Ascend 栈对照) |
| D12 K3 案例 | 迁入 `01_theory/01_models/moonshot_kimi/` |
| D00 阅读路线 + 03 index | 并入 `courses/posttraining_frontier.md`;D01 前沿地图迁入 `01_theory/04_posttraining/` |
| verl/ 其余 8 篇 | 页头加基线横幅("基线 `8a694930`,端到端迭代以 [[新 D07 页]] 的 `983cb0f` 为准"),不做全量重核 |
| 位置错位页 | `RL_PPO_Loss_and_GRPO_Analysis`(TorchTitan+vLLM 源码级)迁入 `04_posttrain_frameworks/`;`batch_invariance_guide` 迁入 `07_training_reliability/`;megatron 两篇 weight sync 页与 D05 §6 划界补链 |

### 3.3 组 3:runtime_graphs(原 06_graphs)内部去重

- 删 `cuda/README.md`(288)、`npu/README.md`(125),导航职能归各自 index.md;npu 侧 overview 归 `aclgraph.md`。
- `CUDA_Graphs_Timing_Diagrams.md`(627)时序图内联进 `PyTorch_CUDA_Graphs_Complete_Guide` 对应四节后删除。
- npu Graph Tree 双写合并:以 `torch_compile_npugraphs_deep_dive` 为主干,吸收 `npugraphs_memory_reuse_analysis` 增量后删除后者;§四 对 make_graphed_callables 只留结论表。
- reduce-overhead 捕获路径以独立页 `aclgraph_deep_analysis`(570)为权威,deep_dive 附录 A 收缩为一句+链接(Graph Tree 与捕获路径是两个子主题,各留一页)。
- `comparison.md`(632)只留差异表,删除对两侧机制的复述。

### 3.4 组 4:Megatron PP 三页合并

以 `megatron-lm/megatron_pp_schedulers_analysis.md`(基线 `ee3f1ff`)为权威,吸收 `megatron_pp_parallelism_analysis.md`(740)与 `megatron_pp_supplements_analysis.md`(229)的增量后删除两者。

### 3.5 组 5:横向页收缩 + 机制归一

- `comm_compute_overlap_analysis`、`distributed_optimizer_deep_dive` 收缩为纯对比矩阵页,机制正文下沉至 megatron-lm/torchtitan/mindspeed 对应页;`comm_compute_fusion_guide` 与 overlap 页补交叉链澄清"融合 vs 掩盖"边界。
- Ring Attention/CP 通用机制抽为 `01_theory/06_distributed_parallelism/` 权威页;`megatron_cp`/`torchtitan_cp`/`mindspeed_context_parallel`/`deepseek_v4_context_parallel` 四页只留各自实现差异。
- `08_kernel_optimization/operator_optimization_guide.md`(834)整页迁入 `05_gpu_kernel/`,Roofline/执行模型与 `gpu_kernel_guide`/`cuda_execution_model_guide`/`triton_00`/`triton_06` 归一为"执行模型一页+Roofline 一页",其余页只链接;§6 昇腾段与 `ascend_kernel_execution_model_analysis` 合并;`08_kernel_optimization/` 剩余页(tilelang 等)一并迁入 `05_gpu_kernel/` 后该目录删除(两级重组后不产生编号空缺)。
- FSDP 六处、megatron 优化器三页:megatron-lm 内 `megatron_distributed_optimizer`/`megatron_ddp_optimizer`/`megatron_optimizer_internals` 三页合并为一页;torchtitan 四页 FSDP 保留(各讲一个专题,补显式分工声明);`15_distributed_primitives` 与 `02_train_frameworks` 执行既有待办"原语 vs 应用划界"。

### 3.6 中重叠 7 组

只做两件事:重叠段收缩为"一句话+链接",补双向链。以 sglang↔vllm passes 页的显式划界声明为范本。

## 4. 链接与索引治理

1. **`[[correction_report]]` 113 处**:审计过程标注整体移除(纠错结论已体现在正文;审计产物本身也将删除)。同类的 `[[demo_delivery_report_2026-07-29]]` 跨层链接一并处理。
2. **index 链接规则**:指向 index 的链接必须路径限定;修复全部 71 处裸 `[[index]]`;`[[01_theory/01_models]]`(指向目录)改指 index。
3. **其余坏链**:示例文本(16 处)反引号转义;末尾反斜杠(9 处)修复;乱码(6 处)修复;代码签名误解析(3 处)转义;真实缺页 4 处在对应 index 的 Knowledge Gaps 标注。
4. **Related Pages 规则**:改为"3–7 条精选,每条一句话说明关联;index 页豁免"。仅对本次迁移/合并涉及页补齐;其余 65 篇缺失不专项回补。
5. **changelog**:按季度切分(`wiki/changelog/2026Q2.md` 等),主文件只留当季;正文示例 `[[...]]` 全部转义。
6. **验收**:`tools/check_links.py` 报告坏链 0、裸 index 0、孤儿页 0(根 index 除外)。

## 5. 命名统一

- **目录内分段编号(2026-07-30 追加确认)**:内容页文件名加两位段位编号前缀,十位数字=类别/难度段,个位=段内学习顺序,让读者从文件名即可由浅入深:
  - 段 0(`01`–`09`):入门/导览(quickstart、knowledge map、overview 类)
  - 段 1(`10`–`19`):核心机制主线(按流水线/学习顺序排列的 `_analysis` 页)
  - 段 2(`20`–`29`):深潜/专题(`_deepdive`、专项分析、边角机制)
  - 段 3(`30`–`39`):方法论/对照/工程实践(开发 guide、comparison、排查实践)
  - `index.md` 不编号;某段超容时占用相邻空段并在该目录 index 的段位表注明;硬件子目录(npu/、cuda/)页多时同规则递归适用
  - 生效范围与时机:P4 收尾时对 `01_ai_frameworks` 全部定型目录统一编号;P7 推广到全库其余目录
- 合法后缀 6 种:`_analysis`、`_guide`、`_quickstart`、`_deepdive`(统一无下划线)、`comparison`、`index`。
- `_deep_dive` 5 篇改 `_deepdive`;`_report/_methodology/_overview/_map/_model/_concepts/_details/_diagrams/_v2` 等就近改名(多数在合并清单内自然消亡)。
- 非 snake_case 17 篇:删除清单外的改小写下划线;`kimi_k2.5_analysis`→`kimi_k2_5_analysis`、`kimi_k1.5_analysis`→`kimi_k1_5_analysis`(消点号);`mHC.md`→`mhc_analysis.md`;`Engram_Analysis.md`→`engram_analysis.md`;`RL_Training_Inference_Precision_Analysis` 等迁移时一并改名。
- wiki 内 README.md 取缔(4 个);19 号纯序号页迁移时按目标目录规范重命名(补后缀)。
- 目录名 `megatron-lm/` 连字符不改(影响面大、收益小,记录为已知偏离)。
- 所有改名/删页由链接修复脚本统一改写入链。

## 6. CLAUDE.md 规则修订

1. **Update Principles**:"Never delete" 改为"**合并优于并存**——发现主题重叠必须合并到权威页,被并页删除并在 changelog 记录(标明并入目标);删除前必须修复全部入链"。
2. **新增 Courses 规则(堵复发根因)**:学习域/系列课只能以 `wiki/courses/` 导读页创建;导读页只含阅读顺序、链接与每篇一句话导读,**禁止承载正文**;需要新正文时写入功能树并从课程页链接。
3. **溯源政策**:raw/ 只收文档类源(论文/文章/图表源);代码分析页必须在页头钉住"仓库 + commit 基线";删除"every claim traces back to raw/"的全称表述,放弃 raw↔wiki 镜像要求。
4. **Page Types**:登记 `_quickstart`、`_deepdive`;明令禁止 wiki 内 README.md;后缀白名单即 §5 六种。
5. **Cross-Reference Rules**:替换为 §4 规则(路径限定 index 链、Related Pages 3–7 条精选制、示例链接必须转义)。
6. **索引维护**:各 index.md 只维护本目录条目表;`wiki/index.md` 只留域级表格;README.md 不写精确页数。
7. 保留:Mermaid 规范、Ingest/Query Workflow 主体、>500 行拆分提议(41 篇存量超标页不专项处理,合并时顺带瘦身)。

## 7. 执行策略

分 7 阶段,每阶段结束跑 `tools/check_links.py` + git commit;在整改分支上进行,阶段可独立验收。

| 阶段 | 内容 | 风险 |
|---|---|---|
| P0 | 建 `tools/check_links.py`;建分支;基线统计存档 | 低 |
| P1 快速止血 | 删 `torch_compile_debug/`;`raw/_ingest`→`docs/research/`;`wanka_*.md` 迁回 wiki;git rm `docs/audits` `docs/reports` `docs/batch_invariance_demo.py`(demo 脚本移至 tools/)+gitignore;修 README/`wiki/index.md` 失真;修琐碎坏链(§4.3) | 低 |
| P2 图源入库 | `.html2md` 的 figs/脚本迁 `tools/`;抽验 1–2 张图可再生 | 低 |
| P3 06_graphs 去重 | §3.3,自包含练手 | 中 |
| P4 ai_frameworks 重组+19 号大迁移 | 先按 §2 映射表 git mv 完成两级重组(纯移动+修链),再按 §3.1 以 A→E→B→D→C→F 卷分批解散 19 号 | 高 |
| P5 后训练整合 | §3.2 + `courses/posttraining_frontier.md` | 中 |
| P6 横向页收缩 | §3.4 + §3.5 + §3.6 | 中 |
| P7 收尾 | 命名统一(§5);全部 index 重建;`courses/torch_compile_end_to_end.md` 定稿;CLAUDE.md/README 修订(§6);changelog 归档 | 中 |

预期结果:页面数 400 → 约 330;13 组高重叠清零;坏链 160 → 0;仓库工作区瘦身约 60MB;`01_ai_frameworks` 从 18 个平铺目录重组为 5 个架构层两级目录,目录顺序即阅读顺序。实施计划体量较大,writing-plans 阶段可按阶段拆成多份计划文档。

## 8. 范围外(明确不做)

- 32 篇未消化论文(OpenAI/Qwen/Google/Anthropic/MiniMax/Llama)的摄入
- `03_sft/`、`05_inference/` 空目录的内容建设
- verl 8 篇对新基线 `983cb0f` 的全量重核(只加横幅)
- 65 篇 Related Pages 缺失的专项回补(仅迁移页顺带补)
- 41 篇超 500 行存量页的专项拆分(合并涉及页顺带瘦身)
- `megatron-lm/` 目录改名
