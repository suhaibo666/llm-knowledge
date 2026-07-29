# `torch.compile` 端到端学习系列设计

**日期**：2026-07-28  
**状态**：设计已由用户确认，授权按 A→F 优先级直接实施  
**目标仓库**：`llm-knowledge`  
**源码主基线**：`E:/97-codes/torch_parallel/p` detached
`e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`  
**运行观察基线**：PyTorch `2.9.1+cpu` / git
`5811a8d7da873dd699ff6687092c225caffcf1bb`  
**新总入口**：
`wiki/02_engineering/01_ai_frameworks/19_torch_compile_end_to_end/`  
**Git 约束**：不创建、不暂存、不提交 commit

## 1. 背景

现有 `16_graph_compiler_foundations/00–21` 已经系统解释 FX、AOTAutograd、图改写、
Inductor IR、Scheduler 和 codegen。但是它的学习起点是“程序如何进入或已经进入某种图”，
不负责完整解释下列端到端问题：

```text
torch.compile API
→ 首次调用
→ Python frame 拦截
→ 字节码符号执行
→ graph break / guard / cache / recompile
→ FX / AOT / Inductor 图编译
→ 编译产物加载与 runtime wrapper
→ 调试、正确性和性能验收
→ 训练、分布式、扩展与部署
```

知识库已有 Dynamo、compile cache、debug、distributed、custom op 等页面，但它们属于不同
批次和源码基线。它们可以作为发现主题与历史线索，不能未经重新核验直接并入新权威主线。

## 2. 学习终点

读者完成六卷后，应能独立完成：

1. 从 `torch.compile()` 参数追到 `torch._dynamo.optimize()` 和 backend wrapper；
2. 解释一次 Python frame 的 cache lookup、symbolic execution、guard 生成与 code rewrite；
3. 区分 graph break、guard failure、backend failure、runtime failure 与性能退化；
4. 沿 Dynamo FX、AOT joint/fw/bw、Inductor IR、Scheduler、wrapper/kernel 追踪状态；
5. 解释首次编译、后续缓存命中、重新编译、eager fallback 和失效边界；
6. 使用正确的日志、artifact、minifier 和 bisector 定位失败阶段；
7. 分离 compile latency、warmup、steady-state、kernel、内存和分布式成本；
8. 判断 custom op/backend/device、Compiled Autograd、DDP/FSDP/DTensor 和 AOTInductor
   分别接入哪一层；
9. 设计 correctness、dynamic shape、performance 和 production rollout gate；
10. 明确 generated source、native compile、真实执行和测量证据之间的等级差异。

## 3. 总体信息架构

采用“总入口 + 六卷”的两级编号。现有卷 C 文件不改名、不复制，只在总入口中显示为
`C01–C21`，继续链接原 `16_graph_compiler_foundations/01–21`。

```text
00     torch.compile 端到端总索引
A01–A05  执行模型前置基础
B01–B10  torch.compile API 与 TorchDynamo
C01–C21  现有图编译核心课程
D01–D07  编译产物、缓存与运行时
E01–E09  调试、正确性与性能
F01–F08  训练、分布式、扩展与部署
```

不把 B/D/E/F 机械追加为原系列的 `22+`，因为 Dynamo 必须先于图编译核心学习；追加到
末尾会造成错误的前置依赖。

## 4. 单篇标准

每篇正文必须形成以下闭环：

1. **背景与动机**：哪种失败、成本或语义问题迫使系统引入当前机制；
2. **核心结论**：用一至两句话给出主线；
3. **对象与所有权**：关键类型、状态、生命周期与 owner；
4. **真实调用链**：从 driver 到当前机制，再到下游 consumer；
5. **状态机**：读取什么、写回什么、何时新建、何时失效；
6. **设计原因**：为什么不采用更直接的替代方案；
7. **不变量与失败边界**：assert、guard、fallback、graph break 或拒绝条件；
8. **复杂度与成本**：仅在可以参数化时给出，不用空泛 `O(N)`；
9. **常见误解**：纠正 API 名称或概念相似导致的错误推断；
10. **源码阅读入口**：固定 commit 下已打开核验的 repository-relative `file:line`；
11. **上下游关系**：前置、后续和 `## Related Pages`。

本阶段不新增演示 demo。现有 Lab 可以作为旁证链接，但新文档验收以源码机制、证据账本、
链接和结构门禁为主。后续恢复 demo 时另立计划。

## 5. 卷 A：执行模型前置基础

目录：`19_torch_compile_end_to_end/`

| 编号 | 文件 | 独立问题 |
|---:|---|---|
| A01 | `a01_tensor_storage_layout_and_views_analysis.md` | Tensor、Storage、layout、view/alias 如何构成编译器值语义 |
| A02 | `a02_operator_schema_dispatch_and_autograd_analysis.md` | operator schema、dispatcher 与 eager autograd 如何提供可捕获执行层 |
| A03 | `a03_python_frames_code_objects_and_bytecode_analysis.md` | frame、code object、instruction、value stack 为什么是 Dynamo 的捕获边界 |
| A04 | `a04_dispatch_modes_proxy_tensor_and_fake_tensor_analysis.md` | `__torch_function__`、`__torch_dispatch__`、ProxyTensor 与 FakeTensor 如何分工 |
| A05 | `a05_eager_capture_compile_and_replay_cost_model_analysis.md` | eager、capture、compile、load/replay 的生命周期和成本模型 |

## 6. 卷 B：`torch.compile` API 与 TorchDynamo

| 编号 | 文件 | 独立问题 |
|---:|---|---|
| B01 | `b01_torch_compile_api_and_first_call_lifecycle_analysis.md` | API 包装与第一次调用如何进入 Dynamo |
| B02 | `b02_backend_modes_options_stances_and_fullgraph_analysis.md` | backend、mode、options、stance、`fullgraph` 与 dynamic 参数如何改变策略 |
| B03 | `b03_eval_frame_callback_and_code_cache_analysis.md` | PEP 523 callback、code object cache 和 frame state 如何协作 |
| B04 | `b04_instruction_translator_and_bytecode_state_machine_analysis.md` | InstructionTranslator 如何执行 Python bytecode 的符号版本 |
| B05 | `b05_variable_tracker_source_and_python_object_model_analysis.md` | VariableTracker 与 Source 如何建模 Python 值、身份和来源 |
| B06 | `b06_output_graph_side_effects_and_graph_emission_analysis.md` | OutputGraph 如何收集 FX、guards、side effects 并提交 backend |
| B07 | `b07_guards_cache_lookup_and_recompilation_analysis.md` | guard 如何生成、执行、选择 cache entry 并触发重编译 |
| B08 | `b08_graph_break_resume_functions_and_partial_graphs_analysis.md` | graph break 如何切图、生成 resume code 并恢复 Python 状态 |
| B09 | `b09_dynamic_shapes_generalization_and_fallback_analysis.md` | static→dynamic 泛化、recompile limit 与 eager fallback |
| B10 | `b10_backend_contract_and_custom_backend_analysis.md` | backend 接收什么 FX/signature，返回 callable 的约束是什么 |

## 7. 卷 C：图编译核心

卷 C 复用：

`wiki/02_engineering/01_ai_frameworks/16_graph_compiler_foundations/00_pytorch_graph_series_index.md`

总入口增加桥接说明：

```text
B06 OutputGraph / B10 backend contract
→ C07 capture frontend
→ C08 normalization
→ C09–C10 AOT joint/fw/bw/runtime ABI
→ C12–C16 rewrite/pass
→ C17–C21 Inductor
→ D01 compile_fx runtime orchestration
```

卷 C 不重新编号文件，不修改已经冻结的历史页，只按需要更新索引回链。

## 8. 卷 D：编译产物、缓存与运行时

| 编号 | 文件 | 独立问题 |
|---:|---|---|
| D01 | `d01_inductor_compile_fx_orchestration_analysis.md` | `compile_fx` 如何编排 AOTAutograd 与 inner compile |
| D02 | `d02_aot_runtime_wrappers_and_lazy_backward_compile_analysis.md` | forward/backward wrapper、saved slots 与 lazy bw compile |
| D03 | `d03_async_compile_workers_and_module_loading_analysis.md` | async compile、worker、code cache 与 module load |
| D04 | `d04_compile_cache_hierarchy_keys_and_invalidation_analysis.md` | Dynamo/AOT/FXGraph/code/kernel/autotune cache 的 key 与失效 |
| D05 | `d05_wrapper_execution_memory_allocation_and_reuse_analysis.md` | wrapper 如何分配、复用、调用 kernel/extern 并返回结果 |
| D06 | `d06_cudagraph_trees_warmup_record_and_replay_analysis.md` | CUDAGraph Tree 的 warmup、record、replay、generation 与 liveness |
| D07 | `d07_compiled_artifact_lifecycle_and_runtime_failures_analysis.md` | 编译产物从创建、加载、调用到失效的故障边界 |

## 9. 卷 E：调试、正确性与性能

| 编号 | 文件 | 独立问题 |
|---:|---|---|
| E01 | `e01_observability_logs_counters_and_artifact_map_analysis.md` | 日志、counters、trace、dump 与 artifact 分别证明什么 |
| E02 | `e02_dynamo_explain_and_graph_break_diagnosis_analysis.md` | 如何从 explain 定位捕获失败和切图原因 |
| E03 | `e03_guard_failure_and_recompile_diagnosis_analysis.md` | 如何定位 guard failure、cache miss 和 recompile storm |
| E04 | `e04_aotautograd_and_inductor_failure_localization_analysis.md` | 如何区分 AOT、partition、lowering、scheduler、codegen 与 runtime failure |
| E05 | `e05_minifier_repro_and_compiler_bisector_analysis.md` | 如何产生可复现程序并二分失败子系统 |
| E06 | `e06_compiled_correctness_validation_methodology_analysis.md` | 数值、梯度、shape、alias、mutation、effect 如何验收 |
| E07 | `e07_compile_latency_cache_and_steady_state_performance_analysis.md` | 冷启动、热缓存、warmup、steady state 如何分开测量 |
| E08 | `e08_kernel_fusion_memory_and_hardware_performance_analysis.md` | kernel、fusion、memory、launch 与硬件瓶颈如何归因 |
| E09 | `e09_production_rollout_fallback_and_monitoring_analysis.md` | 灰度、fallback、版本、artifact、监控与回滚 gate |

## 10. 卷 F：训练、分布式、扩展与部署

| 编号 | 文件 | 独立问题 |
|---:|---|---|
| F01 | `f01_compiled_autograd_analysis.md` | Compiled Autograd 与 AOTAutograd 有何不同，如何捕获 eager backward engine |
| F02 | `f02_activation_checkpoint_recompute_and_compile_analysis.md` | 用户 checkpoint、AOT rematerialization 与 runtime memory 如何叠加 |
| F03 | `f03_ddp_compile_boundaries_and_optimizer_analysis.md` | DDP、optimizer、bucket/reducer 与 compile region 如何交互 |
| F04 | `f04_fsdp_dtensor_and_distributed_graphs_analysis.md` | FSDP/DTensor/collective 如何引入额外状态、effect 与图边界 |
| F05 | `f05_custom_operators_fake_kernels_and_decompositions_analysis.md` | custom op 如何提供 schema、fake/meta、autograd 与 decomposition |
| F06 | `f06_custom_backends_and_device_integration_analysis.md` | backend registry、DeviceInterface、lowering/codegen 扩展点 |
| F07 | `f07_aotinductor_packaging_and_deployment_analysis.md` | AOTInductor 与 `torch.compile` JIT 的产物、ABI 和部署差异 |
| F08 | `f08_training_inference_cudagraph_and_freezing_analysis.md` | training/inference、freezing、CUDAGraph 和 mutation 约束如何组合 |

模型级案例不另建 demo 页；每篇使用最小源码路径说明典型 Transformer 场景，并在
E09/F08 汇总端到端故障与性能决策树。

## 11. 证据设计

### 11.1 事实来源

优先级：

1. 固定 commit 的源码和同 commit 测试；
2. 固定源码注释与不变量检查；
3. 已有正式 runtime receipt；
4. PyTorch 官方文档，仅用于源码未声明的用户契约；
5. 现有 wiki 仅作主题线索；
6. 推论必须绑定一个或多个已验证父结论。

### 11.2 证据等级

沿用：

- `[S]`：固定源码事实；
- `[R]`：当前环境真实执行；
- `[I]`：具有父证据与推理说明；
- `[M]`：generated-only/mock；
- `[B]`：环境或能力阻塞。

本阶段冻结 demo，因此新增页主要使用 `[S]` 与 `[I]`；只有复用现有正式 receipt 时才能
使用 `[R]/[M]/[B]`。

### 11.3 定位要求

- 正文 source locator 使用 repository-relative `path:line`；
- 写入前打开目标范围核验；
- 正式 `[S]` evidence range 不超过 30 行；
- 任何版本不稳定 API 都绑定固定 SHA；
- 不把旧 wiki 的绝对路径或历史行号当作新证据。

## 12. 集成与迁移

- 新建 `19_torch_compile_end_to_end/index.md` 和 `00_torch_compile_end_to_end_index.md`；
- 更新 `01_ai_frameworks/index.md`；
- A/B/D/E/F 页面回链卷 C、Dynamo、AOT、Inductor、cache、runtime、distributed 等领域；
- 旧页不删除、不搬迁；
- 旧页与新结论冲突时增加 correction/contradiction callout，不静默覆盖；
- `wiki/changelog.md`按卷记录完成状态；
- 最终创建端到端系列 manifest、claim ledger、summary 与 delivery report。

## 13. 实施批次

1. **批次 0**：总入口、术语、文件矩阵、审计 manifest；
2. **批次 A**：A01–A05；
3. **批次 B**：B01–B10；
4. **批次 C**：接入现有 00–21，核对桥接；
5. **批次 D**：D01–D07；
6. **批次 E**：E01–E09；
7. **批次 F**：F01–F08；
8. **批次 G**：claim evidence、索引、changelog、最终报告与全量门禁。

每个批次完成后执行：

- 页面结构与最后 `Related Pages` 检查；
- wikilink/file link 解析；
- source locator 路径、行号和范围检查；
- Mermaid 人工规范扫描与可用时实渲；
- `git diff --check`；
- 固定源码 HEAD/clean 检查。

## 14. 验收标准

1. 总索引给出从 eager 到 production 的完整学习路径；
2. A/B/D/E/F 的每个独立问题都有一篇正文；
3. 卷 C 保持原编号、内容与证据，不复制或破坏现有链接；
4. 每篇包含背景、机制、调用链、状态、不变量、边界、设计原因和源码入口；
5. 所有源码断言绑定固定 checkout；
6. 所有新页进入父索引并有双向 Related 链接；
7. 旧页仍完整存在；
8. 不新增 demo，不把环境阻塞写成 runtime pass；
9. claim ledger 对所有事实单元有决定且 validation error 为 0；
10. 结构、链接、Mermaid、源码 locator 和 Markdown diff gate 均通过；
11. 最终报告明确完成项、修正知识、新增知识与保留边界；
12. Git index 为空，不创建 commit。

## 15. 设计自检

- 无 `TBD`、`TODO` 或未命名页面；
- A→B→C→D→E→F 的前置关系与文件编号一致；
- B10 与 F06 分工明确：B10 解释 backend callable 契约，F06 解释设备与后端实现接入；
- C 卷只桥接、不重写稳定编号；
- demo 延后与验收门禁一致；
- native CPU/CUDA/Triton 能力继续保持环境 gate；
- 用户已授权直接实施，不再设置确认暂停点。
