---
title: "vLLM 推理引擎：按问题与依赖组织的知识地图"
---

# vLLM 推理引擎：按问题与依赖组织的知识地图

> **核验基线**：`vllm-project/vllm@199cb9b964822e59ab9b58d88e7be31eb419a2ae`（2026-09-07 UTC）；各页适用范围与证据限制见页头和正文。
> **核验状态**：26篇正文统一到冻结基线；默认执行轴、跨页 owner、关键事实与稳定源码锚点已按全域复审结论收口。最后更新：2026-09-15。
> **目录范围**：26篇内容页 + 本索引，按原有相对顺序连续编号为01–26；旧版系统设计原则页已合并到架构与具体机制页。本页只维护本级入口及阅读依赖；使用、调优、排障、架构和具体机制分别从下表进入。

## 已知覆盖边界

下列主题在当前基线下**本域没有 owner 页面**。它们不是遗漏的链接，而是尚未分配归属的覆盖缺口，已提交 `planning-codebase-analysis` 裁决。右列给出**哪一页写下了这条缺口**，标 **无页面登记** 的表示本表是它在全域的唯一记录；在裁决之前，相关页面只作一行登记，不把读者指向不拥有该内容的页面。

| 尚无 owner 的主题 | 哪些页面在登记它 |
|---|---|
| “调用 API / 使用特性”这一步的使用说明归属 | **无页面登记**（03 §6 登记的是输入路径缺口，不是这一条） |
| P/D 实例之间的 proxy/router 路由（`examples/disaggregated/`） | 22 §2、§14.4；其中 Mooncake 示例 proxy 已由 [[20_mooncake_vllm_integration_analysis|Mooncake 的 vLLM 集成]] §9.1 拥有 |
| 配置解析总链 | **无页面登记**；各页只登记自己消费的字段 |
| Ray / external launcher 的监督拓扑与 Ray DP actor 生命周期（`CoreEngineActorManager`、`EngineCoreActor`，DP+MoE 时为 `DPMoEEngineCoreActor`） | 23 §6.2、13 §4.2 |
| tool / reasoning parser | **无页面登记**（03 只在行文中提到 parser，未登记缺口） |
| pooling 执行（`PoolingRunner` 内部） | 12 §2.6、§1.6 |
| 非 CUDA 平台 | 11 §2.10、02 §3.8（后者列出平台决定的下游选择点并声明本域只覆盖 CUDA） |
| gRPC / Rust frontend | 02 §5.3 |
| scale-out render → token-in 往返的部署编排 | 02 §5.5 |
| `n>1` / beam search | **无页面登记**（14 只把 `n>1` 作为 trace-replay 的排除项提及） |

## 读者入口

| 页面 | 读者问题与内容边界 | 阅读依赖 |
|---|---|---|
| [[01_vllm_feature_optimizations_guide|01 使用指南]] | 怎样准备环境、完成离线/在线调用并读懂流式输出？ | 无；随后读性能调优或调试排障 |
| [[04_vllm_performance_tuning_guide|04 性能评测与调优]] | 怎样固定负载与指标，提出调优假设并验证收益、代价和回滚？ | 使用指南；深入时接Scheduler、KV和设备执行 |
| [[05_vllm_debugging_troubleshooting_guide|05 调试与排障]] | 服务报错或卡住时，怎样采证、定位、处置并验证恢复？ | 使用指南；机制依据接可观测性与对应故障模块 |
| [[02_vllm_architecture_overview_analysis|02 软件架构]] | 设计压力怎样形成六个职责模块，各模块如何协作并支撑不同使用场景？ | 无；先看架构、启动与请求闭环，再进入各机制页 |

## 请求与资源：已有具体机制问题，从这里选择

| 页面 | 读者问题与内容边界 | 阅读依赖 |
|---|---|---|
| [[03_vllm_request_semantics_analysis|03 请求语义]] | 消息怎样经模板、参数与任务转换成为引擎请求，再恢复为用户输出？ | 架构概览 → Engine、采样与多模态 |
| [[06_vllm_engine_architecture_analysis|06 Engine架构]] | 请求怎样登记、提交与完成，两批在途工作怎样配对并释放资源？ | 请求语义 → Scheduler、Serving与分布式 |
| [[07_vllm_scheduler_analysis|07 Scheduler]] | 请求接入后，怎样持续调度、逐轮更新与交付结果，并完成取消、恢复、维护和资源回收？ | Engine → KV、Runner与生成特性 |
| [[08_vllm_kv_cache_management_analysis|08 KV Cache]] | 从建池到共享与回收，各子模块怎样交接块引用、设备索引和完成信号？ | Scheduler → Attention、Runner与分离式KV |
| [[13_vllm_serving_control_plane_analysis|13 Serving控制面与DP Coordinator]] | 负载反馈怎样选副本、wave 怎样唤醒与暂停，以及服务如何就绪、传播故障并关闭？ | 架构与Engine → 分布式、可观测性 |

## 模型与设备执行

> 普通生成在兼容配置下默认先进入 MRV2（12）；11 是 MRV1 的对照与回落路径。编号保留历史顺序，不代表必须先读 11。

| 页面 | 读者问题与内容边界 | 阅读依赖 |
|---|---|---|
| [[09_vllm_model_library_analysis|09 模型库与权重加载]] | checkpoint怎样经过模型选择、名称映射和rank分片成为可执行模型？ | 架构概览 → Attention、量化与在线更新 |
| [[10_vllm_attention_backends_analysis|10 Attention Backend]] | 本步 Query 怎样定位到具体 K/V 元素，backend 从哪些候选与选择轴中选出，布局与块粒度怎样接合？ | 模型库与KV → Runner、量化与编译图 |
| [[11_vllm_model_runner_v1_analysis|11 Model Runner V1]] | 请求行、token 行与 KV slot 怎样对应，压紧换行与设备结果怎样接续？ | 对照/回落路径；先读 Scheduler、KV与Attention，默认代际先读12 |
| [[12_vllm_model_runner_v2_analysis|12 Model Runner V2]] | 稳定请求行、差量写入、当步gather与ModelState怎样生成可重叠的设备执行？ | 默认入口；Scheduler、KV与Attention → 生成特性、编译图；11用于代际对照 |

## 生成与模型特性

| 页面 | 读者问题与内容边界 | 阅读依赖 |
|---|---|---|
| [[14_vllm_sampling_structured_output_analysis|14 采样与结构化输出]] | 一行logits怎样经过约束、过滤与随机选择变成已提交的token，grammar怎样按实际输出推进，logprobs、采样mask与分片采样各是哪条数据面？ | 请求语义与Runner → 投机解码 |
| [[15_vllm_multimodal_execution_analysis|15 多模态执行]] | 一张图片怎样经占位展开、两级缓存、整item准入和分步切片替换embedding，E何时可取又何时释放？ | 请求语义、Scheduler、模型库与Runner → 在线更新的缓存边界 |
| [[16_vllm_speculative_decoding_analysis|16 投机解码]] | drafter怎样构造并与target共用KV，draft怎样被验证、接受或替换，分布、进度与两处结算怎样保持一致？ | 采样、KV与Runner → 编译图、性能调优 |
| [[17_vllm_quantization_analysis|17 量化]] | 低精度数值、scale与打包布局怎样经过加载转换匹配可执行Kernel，MoE三维权重与KV scale参数怎样走同一条链？ | 模型库与Attention → 融合算子、IR |

## 系统专题与生产闭环

| 页面 | 读者问题与内容边界 | 阅读依赖 |
|---|---|---|
| [[26_vllm_multiproc_executor_rpc_deepdive|26 MultiprocExecutor专题]] | 模型并行workers为什么要锁步消费广播RPC，READY、有限共享内存、响应FIFO和shutdown怎样闭环？ | Engine → 分布式、Serving、Runner与可观测性 |
| [[18_vllm_distributed_inference_analysis|18 分布式推理]] | 六根并行轴怎样映射到rank与通信，PP反向采样、专家迁移、弹性扩缩容和微批怎样各自保持次序与完成点？ | Engine、MultiprocExecutor、模型库与Attention → Serving、分离式KV、在线更新 |
| [[19_vllm_compilation_cudagraph_analysis|19 编译与CUDA Graph]] | 动态shape怎样选择编译区域、捕获与重放，两代Runner的派发与encoder/speculator图各按什么键命中，不兼容时怎样回退？ | Attention与两代Runner → IR、融合算子 |
| [[21_vllm_ir_and_fusion_passes_analysis|21 IR与融合Pass]] | vLLM IR 与 torch.compile 如何衔接；Pass 怎样匹配和融合，用户如何接入规则、约束区间并验证结果？ | 量化与编译图 → 融合算子 |
| [[20_vllm_fused_ops_and_kernels_analysis|20 融合算子与Kernel]] | 融合具体减少哪些中间读写；provider与MoE backend怎样三层选择、何时被静默改写；rope、activation、Marlin与FP8权重在Kernel内怎样布局；专家与multi-LoRA怎样重排，何时不能使用？ | Attention、量化、编译与IR → 设备性能验证 |
| [[22_vllm_disaggregated_kv_serving_analysis|22 分离式KV Serving]] | 同一组 prompt blocks 在 pull、push、逐层读取与外部 store 中怎样交接、等待、重算和释放；16个注册 connector 各自支持到哪里？ | Scheduler、KV、Serving与分布式 → 可观测性 |
| [[23_vllm_observability_reliability_analysis|23 可观测性与可靠性]] | 一条请求的事件怎样算成延迟并进入 metrics、trace 或 benchmark；故障怎样通知等待者，哪些状态允许受控恢复？ | Scheduler、KV、Serving、MultiprocExecutor与分布式 → 调试与排障 |
| [[24_vllm_extension_plugin_system_analysis|24 扩展与插件]] | 同一套entry-point发现为什么要按进程、平台、请求、路由和运行时状态拆成不同生命周期？ | 请求语义、模型库与Serving → 在线更新、可观测性 |
| [[25_vllm_weight_transfer_online_update_analysis|25 在线权重更新]] | 请求跨版本时，pause、四类传输后端、原位写入、finish、版本和缓存怎样建立可见性边界？ | Engine、KV、模型库、Runner与分布式 → MultiprocExecutor、可观测性 |

## Related Pages

- [[02_engineering/03_infer_frameworks/index|推理框架目录]] — 回到各框架和专题的本级入口。
- [[02_engineering/03_infer_frameworks/01_llm_inference_technology_stack_analysis|大模型推理技术栈全景]] — 比较不同推理系统的能力边界。
- [[02_engineering/03_infer_frameworks/sglang/index|SGLang推理框架]] — 对照另一条编译实现路线。
- [[02_engineering/03_infer_frameworks/speculative_decoding/index|投机推理专题]] — 跨引擎理解draft/verify算法族。
- [[02_engineering/03_infer_frameworks/mooncake/index|Mooncake源码地图]] — 两个 Mooncake connector 越过边界后，Transfer Engine 与 Store 内部如何执行。
