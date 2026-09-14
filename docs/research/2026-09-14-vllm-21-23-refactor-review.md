# vLLM 21–23 并发重构与独立审读

本轮范围为现有 21（IR 与融合 Pass）、22（分离式 KV Serving）、23（可观测性与可靠性）三页。用户指定以 Megatron-LM 12 的解释深度为参考，并授权子代理并发重构；页面边界、编号与功能树保持既有归属。

源码冻结为 `vllm-project/vllm@199cb9b964822e59ab9b58d88e7be31eb419a2ae`，本机 `/Users/suhaibo/97-llm/vllm`，`main`，提交日期 2026-09-06。原页已使用同一 commit，本轮不迁移基线、不更新 repository-wide radar。源码 checkout 仅作读取；开始时已有未跟踪 `artifacts/`，不纳入本轮操作。

三个写作代理各自拥有一页；协调者 `/root` 未参与三篇正文的初稿写作，负责源码抽查、原文保留审计、跨页衔接、最终审读与集成。方法为 `source-faithful-analysis` 的 codebase source pack、feature-analysis profile、基础五项 rubric 及 feature review；图形遵循知识库绘图与 Mermaid 技能。

## 内容合同

| 页面 | 主解释路径 | 保留边界与入向合同 |
|---|---|---|
| 21 | 双输出数值例子 → mutation 与 donation → functionalization / fusion / lowering → executable graph | 原有 26 条流程、16 个 PassConfig 字段、19 移交的 6 个 CompilationConfig 字段及已核实纠错；19 拥有 compile/cache/capture，20 拥有具体 kernel/provider |
| 22 | 同一 12-token、3-block 例子 → 各数据面匹配/分配/搬运/消费/释放 → 失败闭环 | 16 个注册 connector、native 以外 offload、MultiConnector、KV rank 约定与原有配置；07/08/13/15/18 保持相邻机制所有权 |
| 23 | 普通请求与抢占事件 → 同时间域区间 → logger/trace/benchmark 出口 → fatal 或受控恢复 | 原文设计文档冲突、NaN、采样/聚合限制与故障清理；新增承接15的多模态统计和22的KV传输观测 |

## 协调者已打开的证据

- 21：`vllm/compilation/passes/pass_manager.py::PostGradPassManager.configure / __call__`；`vllm/compilation/passes/ir/inplace_functionalization.py::VllmIRInplaceFunctionalizationPass.__call__`；`vllm/compilation/passes/ir/lowering_pass.py::VllmIRLoweringPass.lower_matched_op / uuid`；`vllm/compilation/passes/ir/clone_elimination.py::UnsafeCloneEliminationPass.__call__ / clone_preserves_layout / user_writes_to_node`；`vllm/compilation/passes/fusion/sequence_parallelism.py::SequenceParallelismPass / MiddleAllReduceRMSNormStaticNVFP4Pattern.register`；`vllm/config/compilation.py::CompilationConfig.__post_init__`；`tests/compile/passes/ir/test_inplace_functionalization.py` 的 donation、functional/mixed、重复使用与 piecewise 测试。
- 22：`vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py::NixlPullConnectorScheduler.get_num_new_matched_tokens`；`vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py::NixlBaseConnectorWorker._handle_failed_transfer`；`vllm/v1/core/sched/scheduler.py::Scheduler._update_waiting_for_remote_kv / _try_promote_blocked_waiting_request`；`vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py` 的 `get_num_new_matched_tokens / wait_for_layer_load / _pop_done_transfers`；`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py::MooncakeStoreWorker.start_load_kv`。
- 23：`vllm/v1/metrics/stats.py::RequestStateStats / IterationStats.update_from_output / update_from_events / update_from_finished_request`；`vllm/v1/engine/async_llm.py::AsyncLLM._run_output_handler`；`vllm/v1/engine/output_processor.py::OutputProcessor.do_tracing / _update_stats_from_output / _update_stats_from_finished`；`vllm/v1/metrics/loggers.py::StatLoggerManager.__init__ / record`；`vllm/config/observability.py::ObservabilityConfig`；`vllm/v1/fault_tolerance/engine_core_sentinel.py::EngineCoreSentinel.on_fault / handle_command / retry / fault_tolerant_wrapper`；`vllm/v1/worker/sentinel/gpu_worker_sentinel.py::WorkerSentinel.retry / _clean_worker_state`；`vllm/v1/engine/core.py::EngineCoreProc.run_engine_core / _send_engine_dead`。

## 原内容保留与实质订正

| 页面 | 原内容去向 | 实质新增或订正 |
|---|---|---|
| 21 | 双输出算例、容差、donation 与布局反例仍在前半篇；26 条流程与装配顺序、16/16 PassConfig、19 移交的 6 个 CompilationConfig 字段、range/uuid 成本与平台默认纠错移至后半篇；§9.1 入向合同保持 | 普通/量化 SP、AsyncTP、各融合族均给出输入与返回合同、注册依据、guard 和收益限制。torch-wrap=False 只令 IR 相关匹配失去对象；显式 auto_functionalized_v2 配置不被默认值覆盖；未知 HOP 按 writer 再判断 donation/后续使用/layout，不能一概写成保留 clone |
| 22 | 20 条接口流程与 owner 表从 §1.1–1.4 移至 §14.1–14.4；16 注册名/15 类、KVTransferConfig 13 字段、38 个家族—键配置项、环境变量、watchdog/heartbeat/HMA/weight epoch 纠错全部保留；§5.4、§11.1 稳定 | 六条数据路径使用相同 R，store 再分三种布局，Multi 分开 load 选择和 save 完成。MoRIIO 实际类为 MoRIIOWriter；READ/WRITE、Mooncake 错误分支只写源码真正发出的反馈。默认 fail 可以先 ERROR 终结请求，异步接收相关 block 仍延迟释放 |
| 23 | 原抢占时序、SLO 分类、NaN、采样/聚合/freshness、健康探针、FT 清理与无自动重放、metrics 设计文档冲突、原 8 个 wikilink 目的地保留 | 新增普通请求、MM 分摊、NIXL 指标、logger 选择、fatal 两路、ObservabilityConfig 13/13 和 FaultToleranceConfig 1/1；ParallelConfig 本页只覆盖 FT 2/61，按顶层类型声明计数，未声称整类已在本页写全 |

协调者对 HEAD 原稿与改稿复算 wikilink 目的地，三页均无旧目的地丢失；逐项阅读写作代理的概念保留账并与正文对照。22 的细账还枚举原文 901 个独有 inline-code/配置/锚点/wikilink 项，对消失字符串逐项说明改名、纠错或等价表达，不能用这个词项检查替代概念审读。正文中旧的其他页面行号引用已改为语义指向；三页均无遗留 `path:line` 引用，因此不触发 locator 条件门禁。

审读退回的具体问题包括：21 关闭 torch-wrap 的过强表述、未知 HOP/clone 的判断、融合图遗漏 NVFP4 scale；22 默认 fail 与 receive 收尾混同、内部 marker 被写成只读输入、invalid-block 专用指标的过度承诺；23 两个错误页面链接与调用树中错误的父子关系。修订由各页原写作者应用，协调者未代写正文以通过自己的 review。

## 独立审读与算例

| page | beat2 | hop-walk | delete-code | figure-trigger | algorithm-replay | spot-check | verdict | note |
|---|---|---|---|---|---|---|---|---|
| 21_vllm_ir_and_fusion_passes_analysis | pass | pass | pass | transform, layout | pass | 3/3 | PASS | feature: pass；双输出、输入旧值、scale 与 KV 副作用均保持可追踪 |
| 22_vllm_disaggregated_kv_serving_analysis | pass | pass | pass | transform, layout, timing, coupled-planes | pass | 3/3 | PASS | feature: pass；按实现区分命中承诺、地址分配、传输收尾、计算准入与源释放 |
| 23_vllm_observability_reliability_analysis | pass | pass | pass | transform, timing | pass | 3/3 | PASS | feature: pass；两个时钟域、统计出口与故障终态不互相冒充 |

三组专项 spot-check：

- 21：`VllmIRInplaceFunctionalizationPass.__call__` 验证后续输入旧值使用会失败、捐赠记录落在 placeholder；`VllmIRLoweringPass.lower_matched_op` 验证 fake 参数选实现及保护复制/替换；`UnsafeCloneEliminationPass.__call__` 验证 placeholder donation、后续 writer/user 与 layout 判定。另打开 manager、SP NVFP4 与 CompilationConfig 核验装配及已订正边界。
- 22：`NixlPullConnectorScheduler.get_num_new_matched_tokens` 核验普通 R 的 `(12, True)`；`NixlPullConnectorWorker._read_blocks` 核验 block 描述符、READ 提交与完成句柄登记，含全前缀命中的纯通知分支；`Scheduler.update_from_output / finish_requests / _update_waiting_for_remote_kv` 核验错误策略、延迟释放和成功后的末 token 重算。另打开 MoRIIO 与 store 对照不同数据路径。
- 23：`IterationStats.update_from_output / update_from_finished_request` 核验 TTFT、ITL、完成区间与均值；`OutputProcessor.do_tracing / propagate_error` 核验 stats 前置条件与异常抵达 collector；`EngineCoreSentinel.on_fault / retry` 核验 abort、DEAD/UNHEALTHY 分流和恢复完成。另打开 NIXL stats、logger 装配与默认 fatal 两条监督通道。

21 的 FP32 语义例子是 `u=(2,2,2,2)`，`epsilon=1e-6`，norm 两种权重对应约 0.999999875 与 1.999999750；不能推出 BF16 或融合 kernel 逐 bit 相等。22 普通三块传输保持 token 位置，成功后 12→11；MoRIIO READ 提前承诺 11，不能套用异步 waiting 模式。store 的 4 heads / store TP=2 / 3 chunks 得 6 个分片键，LBHNC 与 LBNHC 的段数按单 tensor、单 chunk、单 shard 比较为 1 与 4。

23 除手工对账外，从冻结源码 AST 提取实际统计类，以合成时间与空 LoRA 回调执行；协调者阅读并复跑该检查。普通请求得到 TTFT=0.090、queue=0.020、prefill=0.050、ITL=0.025/0.030、decode=0.055、inference=0.105、平均 TPOT=0.0275、E2E=0.145；抢占例保留 first scheduled=12、preemptions=1、queue/prefill/decode/inference=2/6/3/9。浮点断言用容差；这只是统计方法的局部执行，不是 vLLM 服务、GPU 或 IPC 测试。

## 图与机械验收

三页图数分别 11、10、5，共 26 幅 Mermaid。协调者用仓库 vendored Mermaid 与本地浏览器、`htmlLabels:false` 逐图渲染并查看 PNG；退回默认紫/黄配色、长英文标识及 block/shape 数字断行，作者修正后重渲染。图形使用逻辑变换、分片合同、时序和状态流；没有把两轴承载独立寻址决策的二维物理网格伪装成流程图，也没有新增 SVG 或生成器。

陌生读者检查分别要求：21 能指出替换前后 `(y,u)`、scale、分片 residual 与 KV dummy 的去向；22 能沿 R 的源 10/11/12、目标 40/41/42 找到提交者、外部依赖、完成门与最后 token 重算，并辨认各实现未闭合的错误分支；23 能从时间戳重建 0.125≠0.145、从两 worker 的 12/16ms 组时间重建 R 的 8ms 分摊估计，以及沿死亡检测走到等待者异常或受控恢复。

| 集成执行 | 结果 |
|---|---|
| `python tools/check_links.py --strict` | 452 页；broken、ambiguous、bare_index、stale_section、orphans 均为 0 |
| `python tools/check_math.py --changed --strict` | 6 个改动 Markdown，0 errors / 0 warnings |
| `python tools/check_markdown.py --changed --strict` | 6 个改动 Markdown，0 errors / 0 warnings |
| `python tools/check_assets.py --changed --strict` | 6 个改动 Markdown，0 errors / 0 warnings |
| `.venv/bin/python -m tools.mkdocs_site.cli build --changed` | 构建成功；5 个改动 wiki 路由的 broken_links、missing_anchors、missing_assets、missing_legacy_routes 均为 0；scoped orphans 按工具合同跳过，上面的全库检查已覆盖 |
| `node tools/mkdocs-site/mathjax-corpus.mjs --pages <21,22,23 三页>` | PASS；3 个请求页面全部检查，1 页中的 1 个公式渲染成功，其余示例使用行内代码/数值文本 |
| `git diff --check` | 无空白错误 |

首次构建使用当前 shell 的 `python`，缺少 `yaml` 而未启动；核实仓库既有 `.venv` 包含 yaml/mkdocs 后用该解释器完成构建，未安装或更改依赖。构建时 Material 输出关于未来 MkDocs 版本的上游提示，不是本轮页面验证失败。最后两处标签修订后重新构建并执行公式检查。

最终修改范围只有三篇正文、vLLM/index、wiki/changelog 和本报告；vLLM 仍为 26 篇正文加 1 篇 index，没有改变总索引计数或功能树。源码 HEAD 仍为冻结 commit，工作区仍仅有原先未跟踪的 `artifacts/`；未改源码、未提交或推送。

本轮不据源码与测试文件的阅读宣称已运行 GPU、跨机传输、OTLP 服务或设备故障注入；不对外部 PyTorch/NIXL/MoRI/AITER/FlashInfer/Mooncake 的内部执行和性能作已验证承诺。

## 用户反馈后的第二轮：机制解释与段落重写

用户指出首轮仍未说明当前 Pass 如何工作、融合依据、自定义 Pass 接入以及 vLLM IR/Pass 与 torch.compile 的关系，整篇段落写法也未充分参考 Megatron-LM 12。这个反馈表明上面的首轮 PASS 没有充分检验“读者能否理解并运用机制”；不能把首轮表格齐全、链接通过当成用户要求已经满足。上文保留为首轮审查历史，当前交付以本节的补充审查为准。

第二轮由 root 亲自重写21，原22/23作者并行改写各自全文。pass_extension_evidence 独立提供只读证据并终审21、23，rewrite_23 独立终审22；两名审查者均没有参与被审页面的写作。root只集成审查意见与机械门禁，没有给自己写的21作独立PASS。范文12的参考范围是连续因果段落：先交代当前输入或问题，再解释实际处理，随后说明结果、约束与成本；没有照搬章节标题，也没有把所有机制强制套成状态所有权模板。

### 本轮实质变化与内容保留

| 页面 | 实质变化 | 保留核验 |
|---|---|---|
| 21 | 开篇及§3区分FX容器、vLLM语义算子、Graph Pass、Torch backend与vLLM内部adaptor；§5.1沿Add+RMS的pattern/replacement/get_inputs解释四条注册、fake trace归一化、pm.apply改图与测试；§5.1.1区分计算等价、结构匹配、dtype extra_check、能力/range与性能依据；§7.2–7.3给出对象hook、真实规则装配、append位置、UUID/range和验证路线 | 第二轮前全部完整稳定源码锚点保留，5个旧wikilink目的地全部保留；26类流程、16个PassConfig、6个CompilationConfig面向与4个环境变量完整。§2/4/6/7/8改为过程段落，流程/装载/阶段表移到§11.2–11.3；§9.1入向标题稳定 |
| 22 | factory选择、请求/地址几何、调度匹配、六类数据面、store、Multi与成本按连续段落重写；新增NIXL region*B+blockID描述符配对、MoRI层与geometry缓存/session/offset/status的实际交接 | 第二轮前933个行内代码项、23个旧wikilink目的地保留；原配置与错误边界保留，注册/几何/Multi/扫描口径表移§14.5–14.8；§5.4和§11.1稳定 |
| 23 | 普通请求的首/后续/完成输出累积、logger factory装配与record/log/reset、MM三条统计链、NIXL成功/失败/控制数据、fatal和FT均改为连续机制段落；新核对text logger清零顺序 | 14个旧wikilink目的地保留；时钟/抢占算例、配置、NaN/trace/health/FT限制和已纠正文档冲突完整；5个原理图逐字不变 |

词项对账只用于发现可能丢失的概念，不替代机制审读。21删除的少数inline-code词面是检索命令、泛指摘录或改为普通文本的变量；六个曾缩写的完整源码锚点已恢复。三页的章节移动均用全库链接检查核对入向引用。内部保留账分别为 `/tmp/vllm21_followup_conservation.md`、`/tmp/vllm_22_style_conservation.md`、`/tmp/vllm23_paragraph_migration.md`；本表保存它们对交付有意义的结论。

### 独立核验与修正

21新增入口证据包括 `CompilationConfig.init_backend/__post_init__`、`TorchCompileWrapper.__init__`、`make_compiler`、三种 compiler adaptor、`VllmBackend.configure_post_pass`、`PostGradPassManager.add/__call__/uuid`、`InductorPass/CallableInductorPass`、`normalize_value`、`AddRMSNormPattern`、`VllmFusionPatternMatcherPass.register/_trace_fn/__call__`、`RMSNormStaticQuantPattern.register` 及 `_rms_input_weight_dtype_match`。测试亲读 `test_compile_ranges`、`test_pass_manager_uuid/test_bad_callable`、`test_add_rmsnorm_reshape_fusion`。自定义示例使用仓库实际测试的 `inductor_compile_config` 对象入口，不把配置创建当成已经触发编译，也没有把 TestBackend 的宽接口当成生产注入权限。

22审查者亲读 `_compute_desc_ids/_read_blocks`、`MoRIIOWriter._prepare_transfer_plan/_do_layer_write/_finalize_if_complete`、`_compute_block_transfer_offsets`、Multi选择/共享layout逻辑及三个MoRI布局/入队测试。23审查者亲读 `LoggingStatLogger.log/_update_stats/_reset`、`IterationStats` 首输出与结束计算、`EngineCoreSentinel.on_fault/retry`、`WorkerSentinel._clean_worker_state`，并重新执行原始AST方法的局部窗口验证：10秒、40/30 token得到4/3 token/s，随后四个窗口计数清零；完整log格式化的先后由源代码阅读核对。

独立审查退回并已修正：21将pre-grad图和donation异常限定到实际执行该hook的路径；extra users不再被断言一律不匹配；SM100阈值支持与`IS_DENSE=False`默认关闭SP分开。22附录补回MoRIIO READ的层前wait，避免只列WRITE。23将抢占Counter改为累计事件并说明窗口增量另算，调用树补出`Scheduler._preempt_request`后才记录PREEMPTED。

| 页面 | 审查者与关系 | beat2 / hop-walk / delete-code / algorithm-replay | spot-check | 最终结论 |
|---|---|---|---|---|
| 21 | pass_extension_evidence，未写21 | 全部pass；feature pass | 3/3并复核全部修正 | PASS |
| 22 | rewrite_23，未写22 | 全部pass；feature pass | 3/3并补充核对 | PASS |
| 23 | pass_extension_evidence，未写23 | 全部pass；feature pass | 3/3并复核两项修正 | PASS |

独立报告为 `/tmp/vllm21_followup_review.md`、`/tmp/vllm22_followup_review.md`、`/tmp/vllm23_followup_review.md`。最终审查判断依据实际段落、源码、同例重放与反例，不按字数、标题数、表格数或代码块数量判分。

### 第二轮机械验收与运行边界

21三个Python片段均通过语法编译检查；这是源码片段和接入配置的静态核验，未启动vLLM模型。21系统图新增pre hook执行前提后重渲染，发现英文标签折断即缩短标签，root复看最终PNG；其余10图、22的10图和23的5图与第二轮前逐字相同，独立审查者复看22/23既有渲染。总计仍为26幅Mermaid，没有新增资源文件。

第二轮已执行全库links（452页五项全0）、changed math/markdown/assets（6个Markdown，均0错误0警告）、站点构建（5个变更wiki路由，链接/锚点/资源/旧路由全0，scoped orphans按合同跳过）。最后一处摘要与图标签落盘后再次构建通过；三页公式渲染检查PASS（3个请求页全部检查，1页含1个公式）。最后执行git diff --check无空白错误，报告收口后再跑T0四项均通过。

源码HEAD及工作区再次核对：仍为冻结commit，只有既存未跟踪`artifacts/`；没有安装依赖或改源代码。未运行GPU、跨机、外部传输、OTLP服务或故障注入，也未宣称任何融合示例的端到端加速已验证。工作区修改范围保持三篇正文、域index、changelog与本报告，未提交或推送。


## 第三轮：用户十项审阅反馈与 SP 图修复（最终状态）

前两轮的 PASS 没有发现保护 clone 的错误、附录越过依赖边界的断言、失效的普通 § 引用以及文体和内容保留问题；它不能作为这些问题已被验证的证据。本轮以用户列出的 10 项和 SP 图媒介要求为明确验收范围，仅修订 21 号页，22/23 保持本轮开始时的内容。源码仍冻结在 `199cb9b964822e59ab9b58d88e7be31eb419a2ae`。

协调者修改正文，`sp_svg_revision` 独立制作 SP 布局图，`audit_user_ten_fixes` 不参与写作，负责对照修订前的 912 行稿、冻结源码与知识库规范逐项审读。初审没有直接给 PASS，而是退回系统图 clone 条件、QK 图依赖边界、删表时的独有匹配守卫、SiLU 算例映射与段落空行。补齐后，审阅者重新阅读补丁并查看实际渲染，用户 10 项与 SP SVG 范围内未剩未解决项。最终正文为 869 行；这只是版本记录，不是质量指标。

| 用户项 | 最终落点与具体修订 |
|---|---|
| 1 clone 条件 | §11 成本表与 §8.1 系统图均明确仅 inplace provider 插保护 clone，与 `IrOpImpl.func_impl_fn` 的早返回分支一致。 |
| 2 依赖边界 | 旧附录的 kernel 数量和通信调度断言删除；§6.2、6.3、6.7 及对应图统一限定本地调用、参数与返回，区分本地 `_C`、attention backend 和外部 AITER/PyTorch。 |
| 3 § 导航 | 清除全部旧 §8.6；AR、QK/KV、MLA dual、act、SP、AsyncTP 分别指向 6.2、6.3、6.5、6.1、6.6、6.7。全部普通 § 引用人工复核语义，另做存在性扫描；保留明确属于 19 号页的外页引用。 |
| 4 章节归属 | SP/AsyncTP 纳入 §6.6–6.7；5.1.1 降为四级标题；自定义 Pass 独立为 §10；8.2 由空洞清单改为装载与组成的解释。 |
| 5 单一信息源 | 旧11.2/11.3不再复述机制；新12.2仅26行源码入口导航。独有的 fused-add 注册优先、split 全 getitem/sizes 守卫、scatter 形态守卫与 NoOp shape 判定回到5/6节。 |
| 6 同例映射 | H=128、head_dim=64、MLA 8/4/2 与256/128/64明确由开篇四元素向量重复/旁路扩展；SP明确贡献与 residual 的角色不同；SiLU gate/value 对应 x/r，结果单独记 a_out。 |
| 7 正文与过程分离 | 清除“本轮/本次/亲读/已读”、验收措辞与陌生读者自检；最近更新只留日期；未运行状态仍保留。过程归本报告与 changelog。 |
| 8 文体与可读性 | 后半篇改为连续解释段落，Middle/static/NVFP4各自成段；中英留空格，首次定义RS/AG，图中文字同步调整。 |
| 9 前置链接 | 补FX改图、pattern matcher、effects/alias、functionalization、AOTAutograd、pre-grad与post-grad共7个PyTorch知识页入口，明确它们各自的源码基线。 |
| 10 所有权 | §8.4恢复 `tests/kernels/ir/test_layernorm.py::TestRMSNorm.test_impls` 与容差执行归20；明确本页只拥有 range 到达 `is_applicable_for_range` 后的判定，range生成/划分归19。 |
| SP SVG | §6.6使用 `assets/vllm_sp_token_partition.svg`，生成入口为 `tools/figs/svg/vllm_sp_token_partition.mjs`。两rank四token，展示 y 聚齐而 residual u 仍分片；这是布局示意，不是通信数值模拟器。 |

### 最终验证证据

- T0：全库链接检查452页，broken/ambiguous/bare_index/stale_section/orphans均0；math、markdown、assets各检查6个改动Markdown，均0 errors、0 warnings。
- 存在性扫描提取98处普通 § 引用，无缺失标题；该扫描不替代上面的逐条语义复核。PassConfig表16/16，内置实现导航26/26；与912行对照稿比较，完整 `*.py::symbol` 锚点无丢失，独有机制另由人工保留审查确认。
- 站点构建成功，5个改动wiki路由的broken_links、missing_anchors、missing_assets、missing_legacy_routes均0；scoped orphans按工具合同跳过，全库T0已检查orphan。
- MathJax检查21号页返回“no MathJax-bearing pages among the 1 requested”，不声称执行了公式渲染。
- 10个Mermaid图全部实际渲染成功；改动的QK边界、系统clone条件、SiLU结果图由独立审阅者查看PNG，SP SVG由制作代理、协调者与审阅者查看。SVG通过XML解析，生成器语法检查成功，重新生成内容完全一致。
- 正文3个Python代码块通过语法解析；没有执行vLLM设备运行测试。未运行CUDA/ROCm/XPU、分布式pytest或GPU benchmark，亦未验证外部AITER/FlashInfer/PyTorch内部调度。
- 冻结源码HEAD未变，源码checkout仍只有本轮开始时已有的未跟踪 `artifacts/`；`git diff --check`通过。

### 独立审读原始记录（保留初审与复审）

### vLLM 21 用户十项反馈独立复核

日期：2026-09-14。审阅者：audit_user_ten_fixes；未参与此页写作，仅报告发现。
文档：`wiki/02_engineering/03_infer_frameworks/vllm/21_vllm_ir_and_fusion_passes_analysis.md`。
对照稿：`/tmp/vllm21-before-user-review-fixes.md`。
冻结源码：`/Users/suhaibo/97-llm/vllm`，实际核对 HEAD 为 `199cb9b964822e59ab9b58d88e7be31eb419a2ae`，未移动 checkout。

加载并应用 source-fidelity、codebase、feature-analysis profile、base review rubric、feature review、drawing-wiki-figures。此次以用户十条及 SP 媒介要求逐条验收，未把上一轮 PASS 当证据。此报告不代替协调者运行的 T0/build，也不声称运行过 GPU 测试。

#### 逐条审阅记录（初审）

| 用户项 | 实际核查 | 初审状态 |
|---|---|---|
| 1 保护 clone 条件 | 新 §11 成本行明确只为 inplace provider clone，functional provider 直接 impl_fn；§2.3/§4.3/§7 一致。打开 `vllm/ir/op.py::IrOpImpl.func_impl_fn`，首个 not-inplace 分支直接 return。另查 §8.1 系统图仍写“选择实现并插保护 clone”，已要求同样加 inplace 条件。 | 正文通过；图待修复回看 |
| 2 外部依赖边界 | 旧 §11.2 的“四次通信/访存”“单个 AITER HIP kernel”“通信仍与 GEMM 串行”已删除，§12.2 只保留源码导航。§6.2 明确 FlashInfer/AITER 内部未审计；§6.7 限定 symm_mem 参数和返回，不断言 streams/调度。源码 QK 三合一 replacement 确有另一次 per_tensor_quant，正文没有把 quant 合入 kernel。QK图尚未可视化外部边界，已要求补图。 | 正文通过；QK图待回看 |
| 3 全部 § 引用 | 对全文每条 § 引用人工检查语义：旧 §8.6 为0；SP→6.6，AsyncTP→6.7，AR→6.2，QK/RoPE/KV→6.3，MLA双norm→6.5，custom Pass→10.1/10.2，装配→8.3。外页“19 §4.2”是明确外页引用，不按本页4.2误判。表前3行泛指§6有效但建议收紧6.1/6.4。 | 通过（精细导航建议待回看） |
| 4 章节归属 | SP和AsyncTP变体移入6.6/6.7；5.1.1采用####；自定义Pass独立为§10；8.2改为有实际装载解释的组成节；§7聚焦lowering/identity。 | 通过 |
| 5 重复与保留 | 旧两个大附录和阶段表不再作为第二解释源，12.2仅26行导航。主要机制回归§5/6/8。对照旧表发现两项有效独有信息尚需补正文：RMS+quant中fused-add先注册；SplitCoalescingPass全getitem user guard，以及scatter形态约束。源码确认其有效，已通知作者。 | 待两项保留补齐 |
| 6 同例贯穿 | 新§6引入四元素向量重复规则；H=128重复32次；Q/K每head64重复u/w16次，输出y重复16次；MLA q=8/kv=4分别重复u/w两次和一次，k_pe为x前两项，形状8+4+2=14。SP另明示rank1只是复用r数值，角色是贡献而非已有residual，规约后u与y数值一致。SiLU各变体仍缺gate/value到开篇值的明确映射，已建议补并不要把激活值混叫norm y。 | 已列出的SP/QKV/MLA通过；SiLU待回看 |
| 7 过程语与图规格 | 搜索“本轮、本次、亲读、已读、验收对象、陌生读者自检”均0；§2.1说明实际图变换，不再写问题/输入/验收；最近更新只保留日期。未运行验证状态仍明确保留。 | 通过 |
| 8 段落与缩写 | 后半篇正文中英spacing已改善；6.6首次引入AR/RS/AG给出中文含义，6.7复用。6.7 Middle/static/NVFP4三段初稿缺空行，Mermaid labels大量无空格，已通知协调者。 | 待排版回看 |
| 9 前置链接 | FX改图、PatternExpr、effects/alias、normalization/functionalization、AOTAutograd、pre-grad/post-grad在首次机制使用附近各有页面链接。打开目标标题及内容，主题与入口语义相符；正文说明各有PyTorch基线，不冒充当前vLLM第三方审计证据。 | 通过 |
| 10 两项所有权 | §8.4恢复“range到达is_applicable_for_range之后”的本页边界与19的生成/划分责任；恢复TestRMSNorm.test_impls，且新增ir_test_utils.assert_close的op.get_tolerance调用，明确声明与执行区别。两个测试实现亲读确认。 | 通过 |
| SP图SVG | 查看`/tmp/vllm_sp_token_partition.png`的实际渲染。两rank四token横向排列、H=4完整向量为一格；原图两rank各算4行，SP每rank算2行；y聚齐、residual u保持2行。p0/p1角色与First pattern一致，教学值与正文一致，未见重叠/裁切。资产为外部SVG+标准图片语法。 | 通过 |

#### 亲读源码锚点与结论

1. `vllm/ir/op.py::IrOpImpl.func_impl_fn`：只为 inplace provider 克隆 activation；functional直接调用。
2. `vllm/compilation/passes/ir/lowering_pass.py::VllmIRLoweringPass.lower_matched_op`：fake dispatch、补默认值、trace func_impl_fn、run_functional_passes=False。
3. `vllm/compilation/passes/fusion/sequence_parallelism.py::FirstAllReduceRMSNormPattern.register`：原图返回norm/all_reduce，replacement返回all_gather/reduce_scatter。
4. 同文件 `MiddleAllReduceRMSNormPattern.register`：rank-aware residual slice和最终cleanup注释，输出local residual。
5. `vllm/compilation/passes/fusion/rms_quant_fusion.py::RMSNormQuantFusionPass.__init__`：static、dynamic、group都先fused-add后pure RMS。
6. `vllm/compilation/passes/fusion/qk_norm_rope_kvcache_fusion.py::QkNormRopeKvCachePattern.replacement_non_fp8_quant_query/replacement_fp8_quant_query`：两个输出buffer、V旁路、dummy、额外query quant及scale回传；没有推断AITER内部kernel数量。
7. `vllm/compilation/passes/fusion/collective_fusion.py::GEMMReduceScatterPattern.register/AllGatherGEMMPattern.register/AsyncTPPass.is_applicable_for_range`：symm_mem两个方向的调用参数及返回；fullgraph assert后True，无自有token阈值。
8. `vllm/compilation/backends.py::VllmBackend.configure_post_pass` 与 `vllm/compilation/passes/pass_manager.py::PostGradPassManager.configure/__call__/add`：pre hook、ignore-prefix、追加用户Pass、固定尾部与range循环。
9. `vllm/compilation/passes/utility/split_coalescing.py::SplitCoalescingPass.__call__` 和 `scatter_split_replace.py::ScatterSplitReplacementPass.__call__`：前者全getitem guard、同input/sizes但不比较dim，后者限制特定rotary与user形态。
10. `tests/kernels/ir/test_layernorm.py::TestRMSNorm.test_impls` 与 `tests/ir/ir_test_utils.py::assert_close`：generate_inputs、supports_args skip、native/provider compare、从op取得容差。
11. `vllm/compilation/passes/utility/noop_elimination.py::NoOpEliminationPass.__call__/all_dims_equivalent`：reshape重绑与fake shape等价判断、slice_scatter取view替换。

#### 初审结论

REJECT（待小范围修复）：阻塞单元为§8.1系统图clone条件、§6.3图中的依赖边界、§6.1/6.3旧表有效信息保留、§6.1 SiLU同例映射，以及§6.7段落/图中文字spacing。其余项已逐条验证。复审只需确认这些补丁及图渲染，不重复泛化PASS。

#### 修复后最终复审

对作者随后落盘的补丁逐项重读，并实际查看最终 `21-figure-5.png`（QK）、`21-figure-9.png`（系统）、`21-figure-3.png`（SiLU），以及前述 SP SVG 渲染。初审记录保留作为修订轨迹；以下结论取代初审临时 REJECT。

- **1 已闭合**：§8.1 系统图现明确“inplace provider 才插保护 clone”，与 `func_impl_fn` 早退分支及成本行一致；图中文字清晰、无遮挡。
- **2 已闭合**：QK 图新增执行边界，正确区分本地 `_C.fused_qk_norm_rope`、带 KV 的 attention backend 分派与 AITER 三合一；不再用“一次 kernel”替代实际交接。图中可选 Q quant 为独立箭头。对应 `_C` 常量与两个 KV wrapper 已打开核对；没有把两合一统称为 AITER。旧附录过度断言全文已消除。
- **3 已闭合**：配置表前三项也收紧为 §6.1/§6.4。所有旧 §8.6 已归零，全部正文 § 引用此前已逐处按语义目标检查；后续补丁没有新增改号。
- **4 保持通过**：章节归属与 heading 层级如逐项表所述。
- **5 已闭合**：§6.1 补回每种 RMS quant 先注册 fused-add 再 pure RMS；§6.3 补回全 getitem users、同输入同 split sizes 与 scatter 形态 guard。补丁与已读代码吻合。两张旧大表中有据的独有约束已转入正文，来源锚点由正文及导航保留；外部 kernel 次数类无据内容以边界更正，不作为“保留信息”重新引入。
- **6 已闭合**：SiLU 明示 gate=x 重复32次、value=r重复32次，T=2/H=128，输出单独记 a_out；图同步改为高精度 a_out，未再和 norm y 混淆。各 quant 变体用同一个 activation，RMS、SP、QKV、MLA 的扩展映射继续成立。SP 并未把初始 r 误当残差输入。
- **7 保持通过**：过程词和审阅规格没有被补丁重新引回正文。
- **8 已闭合**：Middle/static FP8/NVFP4 已分为三个真实 Markdown 段落；Mermaid labels 添加中英文间隔。查看的 QK/系统/SiLU 图均无文字覆盖、裁切或未定义缩写阻塞。
- **9 保持通过**：前置链接按机制首次需要的位置落点，且保留各自基线说明。
- **10 保持通过**：range 所有权边界与 test_layernorm 容差执行证据都仍在 §8.4。
- **SP SVG 保持通过**：是布局图而非调用链替代，显示完整与分片的双输出区别；媒介、例子与文字对齐。

| page / review scope | beat2 | hop-walk | delete-code | figure-trigger | algorithm-replay | spot-check | feature | verdict | note |
|---|---|---|---|---|---|---|---|---|---|
| 21_vllm_ir_and_fusion_passes_analysis / 用户十项修订及新增 SP 图 | pass | pass | pass | transform, layout | pass | 11 组实际亲读源码锚点 | pass | PASS | 初审发现已逐项闭合；不是承袭上一轮 PASS |

**未解决项：无（本次用户十项与 SP SVG 范围）。** T0/build 由协调者独立执行并将其真实输出写入正式报告；此处的 PASS 是内容与图文复审，不声称 GPU、分布式性能或第三方 kernel 内部验证通过。协调者已报告 T0 四项为零；本审阅者没有重新运行该机械门禁，未将它作为源码正确性的替代证据。
