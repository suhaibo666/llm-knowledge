# vLLM 第二批：解释主线交付与迁移记录

> 编号说明（2026-09-08 后续整理）：本文记录交付过程，正文中的裸编号保留当时口径；文件名、链接路径已同步为连续编号后的当前值，当前导航以 vLLM 域索引为准。

日期：2026-09-08。本文件记录第二批核验、迁移和交付证据；正文及导航的唯一权威仍是 `wiki/02_engineering/03_infer_frameworks/vllm/`，本文件不建立另一套内容目录。

## 范围与执行约束

用户要求“开始处理第二批，要遵从知识库的构建逻辑”。沿用已批准规划中 D/E/F 的 12 篇现有页面：请求与调度 `04/10/11/12`，模型与设备执行 `13/14/15/16`，生成与模型特性 `18/19/20/21`。不新增专题、课程或编号；共享索引、迁移记录和变更记录由协调者统一维护。

源码只读固定于 `vllm-project/vllm@199cb9b964822e59ab9b58d88e7be31eb419a2ae`，即第一批已经采用的基线。旧稿在本批开始时快照至 `/tmp/vllm-wave2/original/`，逐页核对其独有信息；各写作者每次只编辑一篇，交付后由非作者协调者检查实际源码、解释和图示，再派发下一篇。工作区既有修改保持原状，不将其他领域的修改纳入本批内容承诺。

本批使用机制分析 profile，但按对象采用不同解释方式：接口页追踪输入和选择规则，调度页重放预算决策，并发页核对状态与可见时序，采样和量化页解释算法与数值过程。页头基线更新必须有相应的源码重核，不能作为单独交付。

## 逐页交付与旧内容去向

| 页面 | 原稿独有内容去向与本基线纠正 | 新增可重放解释 |
|---|---|---|
| 04 请求语义 | 原协议、生成/pooling/render/audio任务、模板与参数、stop与输出、live/legacy修正均保留在新§2–7；模型任务与HTTP路由没有混成同一列表。 | 普通chat往返、参数限额、跨增量stop缓冲、媒体位置配对和Render/Derender三路。 |
| 10 Engine | 原Client/进程/Executor选择、登记→发送→调度→归并→前端完成、队列、释放与故障边界完整保留；纠正普通stale仍可交付及defer需多批KV consumer，保留旧AsyncLLM wrapper文档冲突。 | 两个12-token prompt与10-token预算的FIFO；两份stale结算；fence2延迟释放；第一帧payload tracker保护可写buffer。 |
| 11 Scheduler | 原全部预算/队列/抢占/等待/结果/完成路径及旧02连续调度理由和信号均保留；补MTP read-ahead、Mamba checkpoint与failed KV load，区分普通/drop stale。 | R/P/Q预算6三步，L3/H2容量5撤回重分配，Mamba列1/1冲突与-1/0有效，spec拒绝回退及placeholder结算。 |
| 12 KV管理 | 原pool/ref/hash/free、容量预测、prefix/partial、hybrid与native CPU完整保留；旧02§3归§1–6；纠正普通CoW提前decref及finalized不等于设备完成。 | A/B共享与slot29；安全进度7；S/D copy引用；Mamba结束pin；packed stride12KiB与4可用块；lookup12→8→4→4；CPU双load。 |
| 13 模型库 | 原registry、构造/PP、loader、名称映射、packed/TP、LoRA、排障与限制全部保留；纠正auto Transformers顺序、普通linear的loaded-name豁免以及IPC后处理。 | 小Qwen2的QKV/gate-up rank分片与前向对应；IPC权重缓存share/copy/后处理/释放完整流程。 |
| 14 Attention | 原能力谓词、per-kind、显式/auto/旁路、spec/layout、metadata复用、代表backend与排障完整保留；明确worker内交集与跨worker同序一致，补B12X/SM90/SM120及indexer对齐。 | B5/A18/A19的槽453/210/211、完整历史与causal可见范围；manager64→kernel16地址不变且不改变回收粒度。 |
| 15 Runner V1 | 原persistent双层状态、增量事件、全部同行字段、processor/LoRA、streaming、异步身份、执行旁路与dummy生命周期均保留；补CoW接缝和GPU位置修正。 | A/X/B删空→压紧→换位的二维行图；5/50/51输入索引与prev2/-1设备接续；两个event分别说明host重用与CPU输出可见性。 |
| 16 Runner V2 | 原稳定row/双视图、staged/UVA、streaming、输入/输出、完整选择矩阵与graph地址生命周期保留；补受限DBO、pool至少2、双向blocker/replay与capture前预热。 | A3/B1不搬而idx1/3；ragged差量与深度2快照；微批只减未来query得到seq19/20；设备状态与worker/copy完成合流。 |
| 18 采样 | 原普通采样/custom、grammar准备/填mask/永久推进、成本与失败边界保留；更新Triton/CPU路径、空行恢复与thinking强制、GPU请求位置映射。 | 七词表逐步过滤；概率比/尾部质量、指数竞赛/Gumbel复算；ab/ac语法预览、回滚与实际输出推进。 |
| 19 多模态 | 原332行媒体/parser、两key、P0/P1/SHM、预算、cache引用、设备切片与六项对齐规则均有去向；明确miss不自动重试、最后occurrence释放与EC准入检查顺序。 | 一张图展开4占位、整item准入与分窗口E切片；稀疏mask前缀计数与M-RoPE delta=-2。 |
| 20 投机解码 | 原proposal/target约束、V1/V2 verification、双侧结算、stale/encoder、成本/词表/slot/custom边界均保留；补block真实算法、DFlash2条件q与adaptive真实GPU边界，纠正1GiB硬上限说法。 | 三词表AA的standard与block质量守恒；synthetic survival转换；DFlash2两步条件路径；computed12/total13；两请求预算2选择与cu0/3/4。 |
| 21 量化 | 原格式识别、名字映射、TP参数、post-load/late bias、候选选择、四类fallback与峰值均保留；纠正online按层组合与load计数完整性边界。 | AWQ 8×8双轴pack与zero轴；FP8四编码与输出误差；TP MAX共享scale；8/12等待late bias与Kernel失败回退。 |

## 旧 02 的迁移核对

旧 `02_vllm_system_design_principles_analysis.md` 本批保留；删除属于规划中的最终集中整合。并行第三批现已完成23的编译部分核验，记录见同日 `vllm-system-topics-wave` 报告；本批逐项确认D/E接收，不以第三批通过替代自己的源码核对。

| 旧02内容 | 唯一正文去向 | 核对证据与状态 |
|---|---|---|
| §1 每步动态请求/进度与三类稀缺资源 | 03§1–4的全局问题；11§1–3的预算例 | 11已按源码核对token/input/request上限，R/P/Q具体例保持原来的因果理由 |
| §1/§6 waiting、KV、preempt、ITL等信号与不能推出的结论 | 11§7；操作去05/06，信号定义去27 | 11保留完整信号表含义；指标不是单一根因证明 |
| §2.1 静态batch替代与逐步复用理由 | 11§1 | 原分析推断标签保留，由源码逐步预算支持 |
| §2.2–2.3 token/input/encoder/KV联合准入、配置拒绝、抢占与重算 | 11§2–5/§7 | 预算断言与5可用块优先级例已重放；撤回当步已选victim不是通用异常事务 |
| §3.1–3.2 连续最大预留的代价、按需block pool、容量预测、早期滑窗释放 | 12 | 12§1–3的请求例、块内碎片、max_model_len目标和9−2安全回收已核验 |
| §3.2–3.3 hash/refcount/可写性、append-only重复与末token重算 | 12 | 12§2/4保留双重身份、重复块与末token；hash完整单元与partial物理块、两类CoW及条件fence已核验 |
| §4.1–4.2 CPU间隙、提前提交与按序归并、placeholder真相 | 10§3–4；11§6 | 两批FIFO与普通/drop stale已独立核对；不给future附加未实现的全局原子提交 |
| §4.2–4.3 stable row/staged diff/host buffer race与async兼容 | 15§4–6；16 | 15的同行迁移与两个event已审；16的稳定row、深度2池化、微批切片及双向兼容矩阵已独立核验 |
| §5.1–5.2 backend能力、显式选择与自动候选、专用块限制 | 14§3–6 | 当前validator/selector/kernel size交集与layout已审；保守能力不等于最快承诺 |
| §5.2–5.3 runtime graph选择、启动时间/显存与fallback成本 | 23§1/3/6/9 | 第三批独立复核报告PASS；14只保留自身能力接缝 |
| §6 四机制可分别配置，却共享token/容量/inflight边界 | 11§7；10/12/15/16/23的相应实现 | 11/10/12/15/16的token/容量/inflight/完成边界均已审；四机制不必共同启用 |


## 索引整理与信息守恒

推理框架父索引仅保留本级2篇独立页和3个子域入口，移除对vLLM子页机制的重复枚举；原技术栈分层已有本级技术栈总览§1–4承接，SGLang当前只覆盖编译Pass、Mooncake为论文分析的范围仍在入口表中。没有新建课程或第二棵功能树。

vLLM域索引将原五类入口、四组owner表、四条路线和症状表合成每页一行的入口表：原26篇链接目标全部保留，首次运行/调优/排障/架构分别由01/05/06/03进入，具体机制按问题选择；阅读依赖改用语义名称。症状操作由05/06拥有，不在索引重写排障机制；训练器与外部编排的边界仍由29适用范围明确。各页题目随实质改写更新，路径与编号保持。总索引删除重复列出的vLLM子页快捷项，其内容仍由域索引的唯一入口表承接；递归重算推理框架35页、vLLM27页（均含index），没有复制下级机制目录。

## 覆盖与跨页接续

| 覆盖状态 | 本批范围与证据边界 |
|---|---|
| 已解释：请求与资源 | 04消息/参数/协议输出，10队列与完成，11多重预算/抢占/checkpoint，12pool/prefix/partial CoW/hybrid/CPU tier；普通、失败与异步分支分别重放。 |
| 已解释：模型与执行 | 13注册/名称映射/融合参数TP分片/IPC加载，14地址与backend能力，15 compact row与异步接续，16 stable row/staged/UVA/受限DBO；选择规则来自固定基线，不等于实测支持认证。 |
| 已解释：生成与数值 | 18过滤/随机采样/grammar，19媒体到embedding、两级缓存与dense/sparse位置，20标准与block验证/条件proposal/双侧结算/adaptive，21pack/zero/FP8/TP scale/在线转换/Kernel选择。 |
| 仅解释接口：LoRA | 13模型包装、15/16请求映射与执行限制；28拥有resolver/扩展入口，29拥有在线更新与缓存一致性接缝。没有把这些入口当作低秩Kernel数值过程的完整讲解。 |
| 明确缺口 | 具体VLM的resize/patching/vision tower/projector网络内部；LoRA低秩Kernel数值过程，以及同名热更、跨API请求和旧KV之间的端到端复现。未凭13通用模型注册或12名称hash补齐这些证据。 |
| 相邻页面接续 | MoE/EPLB的并行与迁移归22，算子内部归24；编译/图归23、IR归25，跨实例KV归26，观测与故障归27。第三批的独立报告提供其证据，本批仍单独核对交接处。 |
| 本轮排除 | GPTQ离线校准优化、外部第三方Kernel/网络库内部、训练器与外部编排算法；GPU/多机真实执行、模型下载、吞吐/延迟测量。教学数字和源码tests的静态阅读不是这些验证。 |

补入的新能力仍归原功能树页面，包括13 IPC、14 B12X/MLA sparse与虚拟块、19缓存miss/EC变体、20 block/adaptive与DFlash2、21按层组合量化；没有另建专题或课程，也没有用跨页链接代替本页应有解释。

## 非作者独立复核

审核使用 `skills/source-faithful-analysis/references/page-review-rubric.md`，每篇打开至少三个承重源码锚点，沿主路径核对语义跳转。触发算法或布局图的页面另检查实际渲染。

| page | beat2 | hop-walk | delete-code | figure-trigger | algorithm-replay | spot-check | verdict | note |
|---|---|---|---|---|---|---|---|---|
| 03_vllm_request_semantics_analysis | pass | pass | pass | transform, timing | pass | 3/3 | PASS | 聊天往返、stop跨增量缓冲和媒体offset配对可重放；Render三路互斥且保留目标校验 |
| 06_vllm_engine_architecture_analysis | pass | pass | pass | timing | pass | 3/3 | PASS | 原计划与future配对；普通/drop stale及GPU fence/传输tracker分别解释 |
| 07_vllm_scheduler_analysis | pass | pass | pass | transform, timing | pass | 3/3 | PASS | 五图独立目视；补全抢占撤回、checkpoint判定并修复宽图；failed load不等同普通stale |
| 08_vllm_kv_cache_management_analysis | pass | pass | pass | transform, timing, layout | pass | 3/3 | PASS | 七图独立目视；pool/partial/Mamba/CPU完成时刻分开；packed与共同边界可复算 |
| 09_vllm_model_library_analysis | pass | pass | pass | transform, timing | pass | 3/3 | PASS | Q/K/V与gate/up分别分片再写本地融合段；IPC copy与zero_copy完成边界可追踪 |
| 10_vllm_attention_backends_analysis | pass | pass | pass | transform | pass | 3/3 | PASS | 槽453/210/211与manager64虚拟拆分可复算；布局协商与局部能力guard分开 |
| 11_vllm_model_runner_v1_analysis | pass | pass | pass | transform, timing | pass | 3/3 | PASS | 三图独立目视；真实行与token索引同例可复算；专属图文一致性测试3/3通过 |
| 12_vllm_model_runner_v2_analysis | pass | pass | pass | transform, timing, layout | pass | 3/3 | PASS | 五图独立目视，图文测试4/4；UVA窗口与微批历史边界、worker/copy合流均清楚 |
| 14_vllm_sampling_structured_output_analysis | pass | pass | pass | transform, timing | pass | 3/3 | PASS | 概率比/尾部质量/指数竞赛/Gumbel可复算；grammar只按实际保留输出永久推进 |
| 15_vllm_multimodal_execution_analysis | pass | pass | pass | transform, timing | pass | 3/3 | PASS | dense/sparse span、缓存miss、整item与本步切片、M-RoPE坐标均可重放 |
| 16_vllm_speculative_decoding_analysis | pass | pass | pass | transform, timing | pass | 3/3 | PASS | 六图独立目视；9条proposal链精确枚举等于完整两token目标分布；synthetic另列目标 |
| 17_vllm_quantization_analysis | pass | pass | pass | transform, layout | pass | 3/3 | PASS | 四图独立目视；AWQ/FP8/TP逐步复算与测试2/2；latebias总量不冒充完整性保证 |

## 门禁与验证边界

最终检查覆盖本批12篇及共享导航；工作区其它任务的既有改动没有纳入本批内容承诺。

- 逐页非作者复核12/12 PASS；每页至少3个承重源码锚点。46个Mermaid与5个SVG共51图全部实际渲染、独立目视；图的数值与条件经源码复算。
- 15/16/21实际图文一致性Node测试分别3/4/2项，合计9项通过。另用独立有理数枚举20的9条proposal链，完整两token输出联合分布等于目标分布。
- 12页显式Python路径与首个qualified symbol机械检查436处均可解析；此检查不替代逐页语义审阅。无`path:line`形式代码引用，不触发legacy locator门禁。
- `check_links --strict`：448页，broken/ambiguous/bare_index/stale_section/orphans全部0。`check_math/check_markdown/check_assets --changed --strict`：各41个Markdown文件，0 errors、0 warnings。
- `mkdocs_site.cli build --changed`：37个变更路由的broken_links/missing_anchors/missing_assets/missing_legacy_routes全部0；构建范围内orphans按工具约定跳过，全库孤页由前项检查保证。
- 浏览器MathJax检查：本批12页全部检查，含公式的4页共87处公式通过。构建输出中的Material版本迁移提示不属于本批文档缺陷。
- 源码检出保持固定HEAD且工作树干净；知识库diff空白检查通过。未运行GPU、模型下载、在线推理、分布式通信或性能测量，教学数值不能表述为实测。

旧02仍保留旧基线，因此没有提前更新repository-wide radar baseline；最终集中整合时在删除/迁移及全域验收完成后处理。

## 独立复核的具体证据

以下按页保存协调者的实际检查记录；作者自检未替代独立判定。各记录内的“尚待/待集成”是审查过程的阶段状态，最终结果以上方12/12汇总及门禁为准。

### 04

协调者打开旧稿、新稿与逐节迁移报告；旧协议、generation/pooling/render/audio任务和所有live/legacy修正均有正文去向。主路径从消息/模板至EngineInput、参数归一、CoreRequest、前端状态、输出增量及协议完成闭合。

| page | beat2 | hop-walk | delete-code | figure-trigger | algorithm-replay | spot-check | verdict | note |
|---|---|---|---|---|---|---|---|---|
| 03_vllm_request_semantics_analysis | pass | pass | pass | transform, timing | pass | 3/3 | PASS | 聊天往返、stop跨增量缓冲和媒体offset配对可重放；Render三路互斥且保留目标校验 |

正式抽查：`vllm/renderers/hf.py::resolve_chat_template`；`vllm/entrypoints/serve/utils/api_utils.py::get_max_tokens`；`vllm/v1/engine/detokenizer.py::check_stop_strings`及BaseIncrementalDetokenizer更新/公开逻辑。

主路径另打开InputProcessor.process_inputs、AsyncLLM.add_request/_add_request/check_admission、RequestState.make_request_output、OutputProcessor.process_outputs，核对local注册、发送、stop完成与abort时序；没有将前端名额检查称为KV admission。旧内容逐节对比未发现丢弃。

四图均实际渲染。图1初审六泳道偏宽，作者压为四泳道并将首次await明确限定_add_request局部；重渲染后可读，无重叠。图2可由你好EN/D后文重建两字符缓冲与END截断；图3保留data/hash/position配对；图4content parts/features/token三路与Derender context边界清楚。

无GPU、HTTP/WebSocket或第三方tokenizer/音频运行验证；全库门禁待集成。

### 10

协调者对照旧稿完整状态/进程/故障边界与新稿；没有编辑正文。普通请求从前端登记、Client传输、输入线程预处理、主循环登记到schedule→execute→FIFO归并→用户输出闭合。两批例子按执行分支重放，不用测试的历史Finish Batch注释推断消费时刻。

| page | beat2 | hop-walk | delete-code | figure-trigger | algorithm-replay | spot-check | verdict | note |
|---|---|---|---|---|---|---|---|---|
| 06_vllm_engine_architecture_analysis | pass | pass | pass | timing | pass | 3/3 | PASS | 原计划与future配对；普通/drop stale及GPU fence/传输tracker分别解释 |

正式抽查：`vllm/v1/engine/core.py::EngineCore.step_with_batch_queue`；`vllm/v1/core/sched/scheduler.py::Scheduler._free_request_blocks/_drain_deferred_frees`及构造gate；`vllm/v1/engine/core.py::EngineCoreProc._send_msg_tracking_payload`。另打开VllmConfig.max_concurrent_batches、test_engine_core_concurrent_batches、前端登记与Scheduler.update_from_output等主路径。

5图全部实际渲染目视，普通4泳道、两批容量、stale两模式、fence2和首帧tracker均可读，无重叠裁切。图仅表达因果顺序，不声称比例时间或实测收益。静态核验未运行模型/GPU/多进程/ZMQ测试。

### 11

协调者已审完整新旧稿与源码。普通R/P/Q预算6、spec50/30/4、优先级victim L3/H2、async拒绝计数及Mamba切分与实现一致。原队列/预算/encoder/抢占/等待/释放/旧02动机和指标含义保留。

初审待修：figure check §3.4不能只以“内部checkpoint无效”跳过列冲突演算；§4撤回已登记L需可重放的独立图/分支；结果图先减计数需注明对象仍存在。已退给作者，正文因果与源码抽查通过，图修复复查后再作最终判定。

正式抽查：`vllm/v1/core/sched/scheduler.py::Scheduler.schedule`的running裁剪/priority撤回/waiting padded rows与async KV load；`vllm/v1/kv_cache_interface.py::is_mamba_prefill_checkpoint_valid`与`get_mamba_prefill_checkpoint_position`；`vllm/v1/core/sched/async_scheduler.py::AsyncScheduler._update_request_with_output`（连读Scheduler.update_from_output）。另开_mamba_block_aligned_split、对应resume/checkpoint tests、priority preemption test。

最终复查：作者已补全两组checkpoint列/对齐算例与抢占撤回图，结果图限定对象存在。图3过宽经二次版式修复为上下案例，协调者重新目视11-3/4/5，可读且数值未改；原图1/2已审。五图全部通过。beat2=pass；hop-walk=pass；delete-code=pass；figure-trigger=transform,timing；algorithm-replay=pass；spot-check=3/3；verdict=PASS。已授权12。

### 13

协调者审阅13全文与作者逐节迁移表；未编辑该正文。问题背景、最小Qwen2例子、模型选择/参数切片/后处理/IPC/LoRA的因果解释及负向边界均闭合。

| page | beat2 | hop-walk | delete-code | figure-trigger | algorithm-replay | spot-check | verdict | note |
|---|---|---|---|---|---|---|---|---|
| 09_vllm_model_library_analysis | pass | pass | pass | transform, timing | pass | 3/3 | PASS | Q/K/V与gate/up分别分片再写本地融合段；IPC copy与zero_copy完成边界可追踪 |

正式抽查：`vllm/model_executor/parameter.py::_ColumnvLLMParameter.load_qkv_weight/load_merged_column_weight`；`vllm/model_executor/model_loader/default_loader.py::DefaultModelLoader.track_weights_loading`；`vllm/model_executor/model_loader/weight_cache/ipc_loader.py::IpcModelLoader.load_model/_build_model`。均实际打开，支持切片、普通linear可能豁免和已处理模式后处理的正文结论。

主路径另打开loader注册/get_model、initialize_model、Qwen2构造与forward、WeightsMapper、BaseModelLoader返回、process_weights_after_loading、registry package hash与tests、IPC协议hash/allowlist及真实暖启测试、LoRA activate/unsupported wrapper分支。没有把loaded-name集合、checkpoint header hash或模型返回扩大为数值完整性证明。

图1从Q8×8/KV4×8/gate-up12×8重放rank1切片和目标offset，前向split边界一致。图2初次LR过宽，退回作者改TB；新图已重新渲染并目视检查，取state→注册→share/clone→后处理→物化→copy尽力release→eval全路径可读，无遮挡或裁切。图为操作/状态映射，不冒充二维storage布局或比例时间轴。

审核是源码与文档静态核对，不含GPU/CUDA IPC/LoRA执行。全库门禁待批次集成。

### 14

协调者审阅全文、旧稿逐节内容及实际源码，未修改正文。教学B5/A18/A19从query分段与seq_lens到块表/槽映射、缓存写入、causal读取和output闭合；3行Query不等于历史KV总长度。能力选择、layout解析、metadata复用和backend差异有各自限制与代价。

| page | beat2 | hop-walk | delete-code | figure-trigger | algorithm-replay | spot-check | verdict | note |
|---|---|---|---|---|---|---|---|---|
| 10_vllm_attention_backends_analysis | pass | pass | pass | transform | pass | 3/3 | PASS | 槽453/210/211与manager64虚拟拆分可复算；布局协商与局部能力guard分开 |

正式抽查：`vllm/model_executor/layers/attention/attention.py::Attention.forward/unified_attention_with_output`与KV dummy dependency；`vllm/v1/worker/utils.py::select_common_block_size`；`vllm/v1/kv_cache_interface.py::create_kv_cache_views`。另打开layout resolver、EngineCore初始化顺序、CUDA selector/validator、index_kpool alignment；selector可整除谓词与实际kernel整数相等规则没有混淆。

两图实际渲染目视通过，无交叠裁切；图表示地址/块号变换，不冒充真实二维字节布局。manager64→kernel16在两种地址公式下得到453/210/211，且不搬字节、不改变回收粒度。未运行第三方attention/GPU数值测试，表中能力仍受本基线和可选依赖限定。

### 15

完整新旧稿、作者逐节守恒报告已审；三张渲染图逐一目视：真实二维行字段随condense/swap共同搬动，索引展开5/50/51及async prev2/-1均能重放。正常正文比例可读，无文字裁切。独立运行专属Node图文一致性测试3/3通过。

正式重开三组承重锚点：`vllm/v1/worker/gpu_input_batch.py::InputBatch.swap_states/condense`；`vllm/v1/worker/gpu_model_runner.py::GPUModelRunner._prepare_inputs/_compute_prev_positions/_prepare_input_ids`；同文件`AsyncGPUModelRunnerOutput.get_output`与`GPUModelRunner.synchronize_input_prep`。与先前CoW helper、旧稿执行分支检查合并，验证批次身份、token/KV进度、host重用和输出可见性独立边界。LoRA展开、streaming更新、dummy/profile/capture、fallback与旧02异步成本均有去向。未运行vLLM/GPU测试。

beat2=pass；hop-walk=pass；delete-code=pass；figure-trigger=transform,timing；algorithm-replay=pass；spot-check=3/3；verdict=PASS。已授权接续16。

后续跨页复核：12发现CoW临时引用释放条件，协调者实际重开Scheduler.schedule与_free_cow_retained_blocks后请15作者最小收窄§3.2，已独立复读。现明确保护的是收集任务前的同一步调度期；普通配置取出任务即归还临时引用，KV consumer+多inflight才按copy fence延期，worker依靠zero→copy→forward流顺序。图及数值不变。

### 18

协调者打开旧稿、新稿与作者迁移报告，逐项核对普通采样、V1 custom、grammar ready/fill/commit、成本与证据限制。去掉代码块后，七词表过程与grammar例子仍自足。

| page | beat2 | hop-walk | delete-code | figure-trigger | algorithm-replay | spot-check | verdict | note |
|---|---|---|---|---|---|---|---|---|
| 14_vllm_sampling_structured_output_analysis | pass | pass | pass | transform, timing | pass | 3/3 | PASS | 概率比/尾部质量/指数竞赛/Gumbel可复算；grammar只按实际保留输出永久推进 |

正式抽查：`vllm/v1/sample/logits_processor/builtin.py::MinTokensLogitsProcessor._mask_stop_token_logits`与MRV2 `_bias_kernel`/对应test_gpu_logit_bias；`vllm/v1/worker/gpu/sample/gumbel.py::gumbel_noised_argmax`；`vllm/v1/core/sched/scheduler.py::Scheduler.update_from_output`的stop→trim→accept→error分支。

另打开MRV2 sampler顺序、badwords后缀/penalty kernel、PyTorch top-k/p排序、Triton dispatch和k门槛终止/split _combine保守回退、grammar request-position mapping和双向stream等待、XgrammarGrammar接受/回滚。正文不把优化kernel等价性、grammar库内部或empty-row处理夸大为通用保证。

教学值独立用标准math复算：top-k后B/C/D为0.140244/0.628532/0.231224；无top-k B/C/D/E为0.129250/0.579259/0.213097/0.078394；最终C/D为0.731059/0.268941；min-p门槛-0.302585。与正文舍入相符。

四张Mermaid全部渲染目视：七词表筛选、排序/主Triton/p-only三路、指数/Gumbel等价教学噪声、ab/ac状态与stop边界均可重建，无文字重叠或裁切。未运行GPU采样或第三方grammar。

### 19

协调者审阅旧稿和新稿，保留媒体加载/parser、processor key与encoder key、P0/P1、budget/freeable、设备输入和六项对齐规则；新增坐标与变体未挤掉旧独有边界。主路径由图片处理到展开token、feature、整item准入、encoder缓存、按窗gather/merge及模型位置闭合。

| page | beat2 | hop-walk | delete-code | figure-trigger | algorithm-replay | spot-check | verdict | note |
|---|---|---|---|---|---|---|---|---|
| 15_vllm_multimodal_execution_analysis | pass | pass | pass | transform, timing | pass | 3/3 | PASS | dense/sparse span、缓存miss、整item与本步切片、M-RoPE坐标均可重放 |

正式抽查：`vllm/multimodal/inputs.py::PlaceholderRange.get_embeds_indices_in_range`；`vllm/v1/core/sched/scheduler.py::Scheduler._try_schedule_encoder_inputs`；`vllm/model_executor/models/qwen2_vl.py::Qwen2VLForConditionalGeneration.get_mrope_input_positions/iter_mm_grid_thw`。另打开receiver汇总miss、Core ERROR发送、Async invalidate、EncoderRunner.gather_mm_embeddings完整路径及重复identifier最后occurrence释放。确认can_allocate在稀疏空窗与EC remote-hit之前，P0 miss恢复需要调用方重试。

4图实际渲染目视通过。图1四行E整item准入与两个窗口消费、图2无后台自动重执行、图3mask F/T/F/T/T前缀计数1:3和flattened位置4/5、图4网格1×2×2三轴位置与delta=-2均与源码演算一致，无重叠裁切。图表示索引/坐标变换与状态流，不冒充二维物理storage。未运行GPU、模型、媒体下载或EC传输测试。

### 20

协调者已读新稿全文、旧稿完整§1–9和作者前期合同。standard逐词质量/AA首拒、block的rho/h/残差、synthetic rates、proposer实际q、device12/total13与CPU结算、成本和chunk/adaptive边界都有实质解释。新稿保持原有V1/V2、grammar、preemption、confirmed encoder、同seed限制和有证据的优化方向。

正式锚点：`vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py::_compute_cumulative_log_p_kernel/_rejection_kernel/_resample_kernel`（前轮连续重开，非首拒即停、placeholder、末位rho、residual放缩与greedy分支）；`vllm/v1/worker/gpu/input_batch.py::_post_update_kernel`和`vllm/v1/worker/gpu/model_runner.py::GPUModelRunner.sample_tokens`（再次打开完整生成主路径）；`vllm/v1/worker/gpu/spec_decode/adaptive_verification.py::_assign_draft_token_budget/AdaptiveVerificationManager.compact_batch/reallocate_drafts`（重开完整函数）。另已读RejectionSampler整请求chunk实现和相应config guard。

独立使用Fraction精确枚举9条proposal链：AA的rho1=2/5、rho2=2/45、h1=13/43；A2/1/0概率2/45、13/45、2/3；补足两输出后的联合矩阵[[1/50,2/25,1/10],[3/100,3/25,3/20],[1/20,1/5,1/4]]，逐项等于p1×p2。脚本仅重放教学算法，不导入/执行vLLM。

初审要求：synthetic是独立接受长度控制算法，补s→c→前缀概率与平均输出的小图，明确target分布保证已改变。待作者渲染最终图与报告后判定。

最终：synthetic独立图已补，协调者实际渲染并逐一查看六图，概率质量、条件q、10/12/13三种进度与adaptive旧CPU边界均可重建；无交叠裁切，图不冒充测量时间。完整作者报告逐节守恒已核对。beat2=pass；hop-walk=pass；delete-code=pass；figure-trigger=transform,timing；algorithm-replay=pass；spot-check=3/3；verdict=PASS。授权21。未运行GPU/模型/外部suffix或性能测试。

### 16

协调者已完整重开`vllm/v1/worker/gpu/ubatch_utils.py`，核对切分、qsl重新定基、seq只减未来query、padding、一次整batch采样、输出按原微批顺序合并、GPU交接和异常可能挂起的边界。另重开`buffer_utils.py`的UvaBufferPool/StagedWriteTensor/FusedStagedWriter/_apply_write_kernel，核对轮换池无每槽event、默认max(2,n)、差量descriptor与多group指针/stride选择；前轮已读RequestState全部与Worker warmup→capture顺序。

提前查看作者两个SVG生成器与渲染：stable row A3/X2/B1删除X后A/B不搬，idxmapping[1,3]；二维切分U0 qsl[0,1,2]/seq[6,19]、U1 qsl[0,1]/seq[20]，数值与源码逐项相符。两图正常宽度可读，行列对齐、箭头不遮字、无裁切。尚待最终正文/报告及其余图检查。

正文初审：已读最终初稿全文，双层值状态、streaming重新加入、单/多group差量、正确CoW释放限定、三类host/device路径、GPU输入与采样回写、完整双向runner矩阵/独立async选择以及graph生命周期均自足。正式抽查三组以RequestState.add/remove、StagedWriteTensor/UvaBufferPool、_slice_input_batch/_slice_seq_lens为锚；另完整打开VllmConfig两套blocker/DBO/replay/use_v2判断和Worker.compile_or_warm_up_model，确认当前支持集合与warmup先于capture。

两项图初审返修：output图需要copy-ready与worker主分支完成合流后Engine消费；pool需独立深度2快照复用图，不能暗示轮转自己证明GPU完成。已发给作者。

最终复查：完整作者报告逐节守恒与旧02§4去向已审，两个SVG和三个Mermaid全部独立目视；新增pool前提与输出worker/copy合流已修正，正常宽度无重叠裁切。独立运行16图文一致性Node测试4/4通过。beat2=pass；hop-walk=pass；delete-code=pass；figure-trigger=transform,timing,layout；algorithm-replay=pass；spot-check=3/3；verdict=PASS。未运行GPU/vLLM或分布式测试。

### 21

协调者完整阅读旧稿、新稿全文和逐节守恒报告，主路径从checkpoint编码、scale、TP分片、方法绑定、post-load到运行Kernel闭合；原配置/loader/名字映射/参数维属性/四类fallback/峰值与Related均有明确去向。旧“预量化与在线量化整体互斥”已按新基线纠正为按层组合与冲突判定，没有以更新页头代替核验。

正式抽查三组：`auto_awq.py::_convert_awq_to_standard_format`全文；`online/fp8.py::_fp8_scale/_fp8_quant_per_channel`及per-tensor处理全文；`reload/layerwise.py::make_online_process_loader/finalize_layerwise_processing/_layerwise_process`全文。另打开`config/quantization.py::resolve_quantization_config`、quant_utils的amax归约helpers、静态FP8 strided-group CUDA Kernel、`scaled_fp8_conversion`及NVIDIA具体转换模板。静态量化乘倒数、channel真正除法与零scale保护的范围分开写，未从Python公式虚推所有设备中点行为。

独立复算：AWQ低位槽0/2/4/6/1/3/5/7经reverse索引0/4/1/5/2/6/3/7恢复0…7；0x75316420与0x76543210及qzeros转置一致。E4M3FN exponent/mantissa验证byte39/41/49/7e为1.125/2.25/4.5/448，scale .5，高精度3.828125、量化3.9375；K分片归约MAX224支持统一scale。late bias刷新总数为12，8/12等待；finalize对padding/首次缺权重与reload恢复的分支意味着总numel不构成一般完整性证明。新图保留这个限制。

实际读取生成器及测试，独立执行Node测试2/2通过；SVG与3张Mermaid共4图逐一目视，轴/位序/局部与全局scale/latebias与候选失败分支清晰，无裁切遮挡。generator只是教学标准布局与数值，正文没有冒充Marlin tile或设备实测。beat2=pass；hop-walk=pass；delete-code=pass；figure-trigger=transform,layout；algorithm-replay=pass；spot-check=3/3；verdict=PASS。未运行vLLM/GPU/TP collective或量化benchmark。

### 12

协调者已重开KVCacheManager.get_computed_blocks/allocate_slots、BlockPool.get_new_blocks/touch/is_block_writable/free_blocks、SingleTypeKVCacheManager._apply_cow/cache_blocks、take_kv_cache_block_copies与Scheduler._free_cow_retained_blocks。容量检查允许先回收旧窗口、finalized仅指不受draft拒绝影响的内容而非GPU已写完、CoW普通立即归还临时ref与有条件fence的区别均已核验。

另完整打开CPUOffloadingManager，包括MISS/HIT_PENDING/HIT、store_threshold/LRU保护候选、store ref=-1到0/失败移除、load ref0到1到0和reset；重开OffloadingConnectorScheduler.reset_cache与完成消息stale阈值，验证旧job跳过和flush IDs保留。packed groups/global spec/PP投影算法先前已读，将按最终正文示例复对。

实际读取12 SVG生成器、渲染并独立查看12-packed.png；group给定成员的教学page A2/B1/C4 KiB得到9/12/8，stride12，padding3/0/4，60KiB得到5 pool块扣null后4。图在同一字节范围上给出三种互斥group解释，比例来自输入，未将group误画为独立pool。文字bbox无越界，目视无交叠裁切。尚待正文/报告和其他算法图。

最终正文与图复核：完整阅读新306行正文并对照旧209行全文。普通10-token/block4位置9→slot29、两请求共享ref1→2→1→0、安全窗口进度9−2=7、hash2/group4的6-token部分命中、普通与有条件fence分支、Mamba运行时移hash与结束时精确块交接均可重放。正式三组抽查是KVCacheManager.allocate_slots/get_computed_blocks、BlockPool分配/touch/is_block_writable/free与CoW接缝、CPUOffloadingManager store/load/reset完整实现。

另独立打开Mamba.finalize_partial_tail_offload/cache_blocks/运行中partial处理，MooncakeStoreScheduler.register_finished_partial_tail/update_connector_output/has_pending_push_work，以及CPU copy handler wait_stream、结束event查询与buffer归还，核实三类完成条件不能相互替换。packed与approximate_gcd完整函数再核，候选3/4/5损失1/4/2，stride12KiB与4可用块相符；GLM5专用分组及PP最不利stage guard、tail alias也已重读。

6张Mermaid及1张SVG全部独立目视，数值、条件和文字可读，无裁切重叠；没有把CPU hash发布、GPU完成或connector保存完成画成同一时刻。初审发现两条不存在的28/29链接、is_writable方法误名及warning代替contradiction格式，已由作者修正，协调者逐项复查。旧02§3按需容量与前缀论证已归入本页§1–6；本页的普通CoW释放限定同步用于15/16接缝。beat2=pass；hop-walk=pass；delete-code=pass；figure-trigger=transform,timing,layout；algorithm-replay=pass；spot-check=3/3；正文与图判定=PASS。未运行GPU、CPU设备copy或外部connector测试。

作者逐节守恒报告已完整复读：旧12全部§1–8与7个Related目标，以及旧02§3均有定位；最后补入的块内碎片与max_model_len截断已逐句核查，reachable_block_mask锚点已修正。未发现未说明的旧内容丢失。最终verdict=PASS。

21最后将保留旧说法的基线纠正改为规范contradiction callout，内容与图不变，协调者复读确认。
