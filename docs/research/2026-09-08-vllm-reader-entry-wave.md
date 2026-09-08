# vLLM 第一批：读者入口交付与迁移记录

日期：2026-09-08。此文件是本批次工作记录与后续核验清单；页面内容与当前导航仍以 `wiki/02_engineering/03_infer_frameworks/vllm/` 为唯一权威。

## 范围与源码基线

用户要求按既有规划完成第一批 A/B/C：使用、性能评测调优、调试与可靠性。本批保留原路径，重构 `01`，新增 `05_vllm_performance_tuning_guide.md` 与 `06_vllm_debugging_troubleshooting_guide.md`，核验并调整 `27` 的机制边界；`03` 只补读者入口链接。

附件规划中的 `6b110badbb22d3f66c7218b71138f13b7a6b3419` 是旧基线。开始时工作区 `03`、域索引及 changelog 已记录重构统一采用 `199cb9b964822e59ab9b58d88e7be31eb419a2ae`；用户提供的 `/Users/suhaibo/97-llm/vllm` HEAD 正好匹配且工作树干净，因此沿用这一已确立基线。源码没有 fetch、checkout 或改写。其余 21 篇仍保留旧基线；本批没有更新全域 radar baseline。

开始时知识库已有其他任务的未提交修改（包括 `03`、域索引和 changelog）。本批在其上局部集成，没有回退、覆盖其他正文或提交这些修改。

## 页面边界与读者验收

| 页面 | 主问题与拥有范围 | 相邻权威页面 | 本批验收 |
|---|---|---|---|
| 01 使用指南 | 环境与安装、离线/在线调用、客户端、流式输出、常用配置及默认解析 | 04 输入语义；05 测量；06 排障 | 同一 Qwen 小模型串起环境、离线退出、在线服务与流式客户端；输出字段和失败结束可解释 |
| 05 性能评测与调优 | 负载合同、工具与指标、资源假设、单变量实验、质量/SLO/成本和回滚 | 11/12/16/23 等内部机制；27 信号产生 | 完整 A/B/回滚教学案例，固定负载与事先阈值，同时解释收益和 TTFT 代价 |
| 06 调试与排障 | 环境/日志/metrics/health/Profiler/trace 操作、分层定位、处置后验证 | 27 故障和观测机制；12 容量；23 编译 | 最大上下文 KV 容量失败贯穿采证、区分两种预算错误、只减长度、恢复验证与能力降级 |
| 27 可观测性与可靠性 | 事件→区间→聚合/trace；fatal/FT/NaN；时钟、采样、基数和可见性 | 06 操作；05 SLO实验；17 路由 | R 的抢占事件可重建统计；FT 从异常清理到拒绝/恢复/超时可追踪；health 与真实完成区分 |

所有运行示例经源码与测试合同静态核验；GPU、模型下载、在线服务、OTLP、Profiler 与故障注入未在本机执行。05 的性能数据及06的故障场景明确为教学推演，不是测量结果。01 的远程 wheel 可用性、模型 revision 与第三方依赖锁定也未验证。

## 旧 01 逐节迁移核对

原稿来自本批开始时的工作区，标题为“使用与优化指南：用可撤销实验寻找限制资源”。原始版本可在本次修改前的 Git 版本找到；本批未改该页路径。

| 原节与独有内容 | 最终去向 | 合并理由与状态 |
|---|---|---|
| §1 反对堆 flags、闭环、O2/balanced 与 budget 默认解析、help | 05§1；01§6 | 使用保留参数解释，实验方法归05；已迁入，默认重新核验 |
| §2 workload envelope 七维表 | 05§2 | 整体保留模型、请求、到达、SLO、吞吐、环境、运行状态；已迁入 |
| §2.1 correctness guard、canary、sampling与量化边界 | 05§2.1 | 保留正确性先于性能；纠正“仅改batch就应逐token一致”的普遍化，改为预声明实验门 |
| §3.1 离线批推理与chat/template边界 | 01§3 | 扩为可执行聊天/纯文本例子，解释结果顺序和返回结构；已保留 |
| §3.2 在线serve与generation config | 01§4–6 | 补客户端、非流式与流式消费、输出上限解析；已保留并扩写 |
| §4 三类benchmark矩阵与共享数据dispatch | 05§3 | 保留工具边界，新增固定batch命令；共享采样入口不等于相同sampling；已迁入 |
| §4.1 离线warmup/measurement命令与分离 | 05§3.1 | 重新核对warmup seed、固定sampling和计时窗口；已迁入 |
| §4.2 在线rate/burst/concurrency、sampling warning、warmup、seed/cache污染 | 05§3.2–3.3 | 保留并补Python/Rust路径、客户端排队、同一请求预热和共享前缀边界；已迁入 |
| §5 正确性/性能baseline、TTFT/ITL/TPOT/goodput与完整artifact | 05§2.1、§4 | 补分子分母、单token特例、goodput不含client queue；已迁入 |
| §6 八类资源限制、变量族、反证与机制owner | 05§5 | 原八类全部保留，操作性错误转06；已迁入 |
| §7 单变量实验卡、不可拆配置、交错重复、噪声 | 05§6 | 原字段保留，并由§7案例实际填写；已迁入 |
| §8 正确性/SLO/因果/容量/运维五面门及回滚触发 | 05§7.4、§8 | 五面表保留，重启和缓存状态恢复不丢失；已迁入 |
| §9 六个可复查问题 | 05§8 | 收束为实验交付证据；已迁入 |
| Related 02/04/11/12/22/27 | 05§1、§2.1、§5与Related；01相关入口 | 旧目标均仍能从其对应问题到达；没有删页或失效入链 |

## 旧 27 内容归位与修正

| 原内容 | 最终去向与状态 |
|---|---|
| §1–2 用户症状与资源状态、热循环成本和反馈分工 | 27§1–2 保留；抽象反馈示意改为§3两个可重建实例图 |
| §3.1 时钟、首次SCHEDULED与抢占区间 | 27§3.1 保留，新增R教学事件和每请求抢占histogram |
| §3.2 labels、trace parent/request ID、batch export | 27§3.2 保留；纠正TTFT/ITL并非只在完成时写，明确LoRA例外 |
| §3.3 症状分类 | 27§3.3 保留信号与故障域映射；06§2/§3.2/§4补操作 |
| §3.4 sentinel、Worker清理、fatal双通道、NaN策略 | 27§3.4 保留；新增默认关闭/单API/backend守卫、局部失败无回滚、旧请求不重放、health不查询所有FT状态 |
| §4.1–4.4 基数、1%/四次历史采样、stale值、多API/LoRA聚合、跨进程成本 | 27§4 全部保留；补KV residency默认关闭，启用后才采样 |
| §5 五步操作闭环 | 06§2完整案例、§3工具与§4分流；27§5改为机制测试合同与操作回链 |
| §6 设计文档与代码时间区间冲突、hidden metrics迁移 | 27§6 保留，不把过时设计文档改写为当前代码行为 |
| 原path:line与Related | 移除波动行号，27§7提供稳定符号路线；Engine边界仍可经正文AsyncLLM/Core路径和Serving页继续阅读 |

## 旧 02 保留与后续迁移清单

本批不删除或缩写旧02。03样稿已解释相邻全局动机，但这不等于旧02全部内容已经迁移；以下每项仍需对应工作包核验后才可标记完成。

| 旧节 | 现有相邻说明/计划去向 | 状态与下一步 |
|---|---|---|
| §1 每步动态batch、三类资源与观测 | 03§1、§3.2、§4.3；11/12 | 全局说明已存在；具体字段和旧证据保留02，待D核对 |
| §2 连续调度：静态替代、token预算、KV admission、抢占重算、测试 | 03§3.2/§4.3；11 | D待迁移核验，包括 max_num_scheduled_tokens 与输入budget区别 |
| §3 分页状态：pool/table、refcount、可写性、淘汰、append-only与末token重算 | 03§4.3；12 | D待迁移核验，旧02独有约束未删除 |
| §4 异步执行：batch queue、placeholder、持久row、host buffer race、显式/自动兼容 | 03§3.3/§4.5；15/16（10只保留协作） | E待迁移；旧基线stale-output描述不可代替新行为 |
| §5 能力合同：backend验证、显式拒绝/自动回退、graph模式与启动显存成本 | 03§4.6；14/23 | E/G待迁移，01/05只给选项与验证入口 |
| §6 四机制耦合与信号解释 | 03模块协作；11/12/15/16/23，观测含义27 | D/E/G待核对，05诊断表不是机制论证的替代 |

## 覆盖与后续工作登记

“已解释”只表示页头基线下存在正文，本表不宣称全域已经重新核验。入口范围优先普通文本/NVIDIA/Python；外部插件与其他后端只给边界，不作能力认证。

| 能力/读者需求 | 本批状态 | 权威位置或后续队列 |
|---|---|---|
| 环境、离线、HTTP、流式、常用参数 | 已补齐入口并核验 | 01；外部插件安装与模型依赖锁定未验证 |
| 三种benchmark、指标、负载、质量、因果与回滚 | 已补齐并核验 | 05；Rust benchmark完整路径本轮排除，示例固定Python |
| 安装/模板/容量/compile/hang分流、日志/health/metrics/Profiler/trace | 已补齐操作入口 | 06；真实设备和采证产物本轮未实跑 |
| 指标产生、抢占区间、FT/fatal、NaN与信号成本 | 已解释并按新基线核验 | 27；详细FT运维拓扑和所有backend恢复能力排除 |
| 全局架构 | 已有新基线样稿 | 03，本轮仅增01/05/06链接 |
| 请求、调度、KV、连续调度、chunked prefill、prefix caching | 旧基线已有机制解释，本批只解释配置/信号 | D：04/10/11/12；12接续null-block预留、max_model_len auto-fit与容量校验；11接续新budget/stale-output语义 |
| 模型、Attention Backend、两代Runner | 旧基线已有机制解释，本批仅选型边界 | E：13/14/15/16；04/13接续模板fallback与缺省上限优先级 |
| 采样、多模态、投机、量化 | 旧基线已有机制解释，本批仅质量/配置导航 | F：18/19/20/21 |
| 编译、CUDA Graph、融合算子、IR Pass | 旧基线已有机制解释，本批仅调参/排错 | G：23/24/25；23接续 enforce_eager 同时禁compile/graph的当前行为与诊断边界 |
| Serving、分布式、跨实例KV、插件、在线权重更新 | 旧基线已有机制解释，本批只给跳转 | H：17/22/26/28/29；22接续具体通信/hang拓扑，不能用本批单卡案例认证 |
| LoRA独立专题、MoE/EPLB独立分页 | 本轮排除，维持原归属 | E/H与G/H先提交覆盖缺口证据，再决定分页 |

## 非作者独立复核

三名写作者分别写01/05/06，协调者写27与共享集成。互审不编辑被审稿；协调者独立审核05。

| page | beat2 | hop-walk | delete-code | figure-trigger | algorithm-replay | spot-check | verdict | note |
|---|---|---|---|---|---|---|---|---|
| 01_vllm_feature_optimizations_guide | pass | pass | pass | none | n/a | 3/3 | PASS | 05作者复核：安装→离线→在线→流式与默认/硬上限 |
| 05_vllm_performance_tuning_guide | pass | pass | pass | none | n/a | 3/3 | PASS | 协调者复核：实证操作指南，算法归机制页；A/B/回滚与代价闭合 |
| 06_vllm_debugging_troubleshooting_guide | pass | pass | pass | none | n/a | 3/3 | PASS | 01作者复核：两类容量错误、Profiler路由与health局限 |
| 27_vllm_observability_reliability_analysis | pass | pass | pass | timing, transform | pass | 3/3 | PASS | 06作者复核并查看两张渲染图 |

正式抽查锚点：

- 01：`vllm/entrypoints/offline_utils.py::OfflineInferenceMixin._run_engine`；`vllm/entrypoints/openai/chat_completion/serving.py::OpenAIServingChat.chat_completion_stream_generator`；`tests/entrypoints/serve/utils/test_api_utils.py::TestGetMaxTokens`。
- 05：`vllm/benchmarks/serve.py::benchmark.limited_request_func / calculate_metrics`；`vllm/benchmarks/lib/endpoint_request_func.py::async_request_openai_completions`；`tests/benchmarks/test_serve_cli.py::test_bench_serve`。另外读了Rust dispatch、离线warmup/sampling以及arrival rescale分支。
- 06：`vllm/v1/core/kv_cache_utils.py::_check_enough_kv_cache_memory` 与null-block测试；`vllm/entrypoints/serve/profile/api_router.py::attach_router`；`vllm/v1/engine/async_llm.py::AsyncLLM.check_health / errored`。
- 27：`vllm/v1/metrics/stats.py::IterationStats.update_from_events / update_from_finished_request`；`vllm/v1/fault_tolerance/engine_core_sentinel.py::EngineCoreSentinel.on_fault / handle_command / retry`；`vllm/v1/worker/sentinel/gpu_worker_sentinel.py::WorkerSentinel.__init__ / retry`。

图示陌生读者检查：第一图可从R事件还原首次scheduled=12、queue=2/prefill=6/decode=3/inference=9，并区分wall-clock分支；第二图可从一次异常追到旧请求作废、DEAD拒绝、UNHEALTHY外部retry和成功/超时。两图已渲染，无文字遮挡、裁切，纵向拓扑不暗示比例时间轴。

## 机械门禁

- `check_links.py --strict`：448页，broken/ambiguous/bare_index/stale_section/orphans 全0。
- `check_math.py --changed --strict`、`check_markdown.py --changed --strict`、`check_assets.py --changed --strict`：18个Markdown文件，0错误0警告。此增量范围包括工作区原有修改；本批没有因此改写其他领域。
- `.venv/bin/python -m tools.mkdocs_site.cli build --changed`：作用域17页，broken_links/missing_anchors/missing_assets/missing_legacy_routes 全0；构建orphans为scoped skip，不冒称全站构建孤儿扫描。全库wikilink orphan门禁单独通过。
- `mathjax-corpus.mjs --pages`：01/03/05/06/27五篇通过，无MathJax公式。
- 27两张Mermaid已渲染并由作者与非作者目视检查；普通页面链接、返回字段和命令由独立审阅读者核验。
- 初次使用系统Python构建时缺PyYAML，已改用仓库已有 `.venv/bin/python` 完成构建；未安装或修改全局依赖。
- 本批没有新增 `path:line` 引用，27的旧引用已迁为稳定符号；未触及条件门禁所列实现工具、训练框架覆盖文件或课程，不运行无关工具测试。未提交、推送或部署。
