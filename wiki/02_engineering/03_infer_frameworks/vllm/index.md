---
title: "vLLM 推理引擎：按问题与依赖组织的知识地图"
---

# vLLM 推理引擎：按问题与依赖组织的知识地图

> **核验基线**：`vllm-project/vllm@199cb9b964822e59ab9b58d88e7be31eb419a2ae`（2026-09-07 UTC）；各页适用范围与证据限制见页头和正文。
> **核验状态**：26篇正文已完成本轮重构与逐页核验。最后更新：2026-09-09。
> **目录范围**：26篇内容页 + 本索引，按原有相对顺序连续编号为01–26；旧版系统设计原则页已合并到架构与具体机制页。本页只维护本级入口及阅读依赖；使用、调优、排障、架构和具体机制分别从下表进入。

## 读者入口

| 页面 | 读者问题与内容边界 | 阅读依赖 |
|---|---|---|
| [[01_vllm_feature_optimizations_guide|01 使用指南]] | 怎样准备环境、完成离线/在线调用并读懂流式输出？ | 无；随后读性能调优或调试排障 |
| [[04_vllm_performance_tuning_guide|04 性能评测与调优]] | 怎样固定负载与指标，提出调优假设并验证收益、代价和回滚？ | 使用指南；深入时接Scheduler、KV和设备执行 |
| [[05_vllm_debugging_troubleshooting_guide|05 调试与排障]] | 服务报错或卡住时，怎样采证、定位、处置并验证恢复？ | 使用指南；机制依据接可观测性与对应故障模块 |
| [[02_vllm_architecture_overview_analysis|02 架构概览]] | 从一次模型计算到并发服务，各模块怎样完成一条请求？ | 无；随后按下列具体问题进入机制页 |

## 请求与资源：已有具体机制问题，从这里选择

| 页面 | 读者问题与内容边界 | 阅读依赖 |
|---|---|---|
| [[03_vllm_request_semantics_analysis|03 请求语义]] | 消息怎样经模板、参数与任务转换成为引擎请求，再恢复为用户输出？ | 架构概览 → Engine、采样与多模态 |
| [[06_vllm_engine_architecture_analysis|06 Engine架构]] | 请求怎样登记、提交与完成，两批在途工作怎样配对并释放资源？ | 请求语义 → Scheduler、Serving与分布式 |
| [[07_vllm_scheduler_analysis|07 Scheduler]] | Scheduler怎样把请求状态和多重资源约束变成本步计划，并按原计划对账？ | Engine → KV、Runner与生成特性 |
| [[08_vllm_kv_cache_management_analysis|08 KV Cache]] | 物理块怎样分配、共享、淘汰和回收，prefix/hybrid/offload怎样接合？ | Scheduler → Attention、Runner与分离式KV |
| [[13_vllm_serving_control_plane_analysis|13 Serving控制面]] | 服务怎样启动、就绪、路由、背压、传播故障并关闭？ | 架构与Engine → 分布式、可观测性 |

## 模型与设备执行

| 页面 | 读者问题与内容边界 | 阅读依赖 |
|---|---|---|
| [[09_vllm_model_library_analysis|09 模型库与权重加载]] | checkpoint怎样经过模型选择、名称映射和rank分片成为可执行模型？ | 架构概览 → Attention、量化与在线更新 |
| [[10_vllm_attention_backends_analysis|10 Attention Backend]] | 本步Query怎样找到完整KV历史，backend与布局如何满足当前能力要求？ | 模型库与KV → Runner、量化与编译图 |
| [[11_vllm_model_runner_v1_analysis|11 Model Runner V1]] | 请求压紧或换行时，哪些输入必须一起移动，怎样接续设备结果？ | Scheduler、KV与Attention → Runner V2、生成特性 |
| [[12_vllm_model_runner_v2_analysis|12 Model Runner V2]] | 稳定请求行、差量写入和当步gather怎样生成可重叠的设备执行？ | Runner V1及其前置 → 生成特性、编译图 |

## 生成与模型特性

| 页面 | 读者问题与内容边界 | 阅读依赖 |
|---|---|---|
| [[14_vllm_sampling_structured_output_analysis|14 采样与结构化输出]] | 一行logits怎样经过过滤与随机选择得到token，grammar怎样按实际输出推进？ | 请求语义与Runner → 投机解码 |
| [[15_vllm_multimodal_execution_analysis|15 多模态执行]] | 一张图片怎样经占位展开、缓存、整item准入和分步切片替换embedding？ | 请求语义、Scheduler、模型库与Runner → 在线更新的缓存边界 |
| [[16_vllm_speculative_decoding_analysis|16 投机解码]] | draft怎样被验证、接受或替换，分布、进度和KV怎样一起修正？ | 采样、KV与Runner → 编译图、性能调优 |
| [[17_vllm_quantization_analysis|17 量化]] | 低精度数值、scale与打包布局怎样经过加载转换匹配可执行Kernel？ | 模型库与Attention → 融合算子、IR |

## 系统专题与生产闭环

| 页面 | 读者问题与内容边界 | 阅读依赖 |
|---|---|---|
| [[18_vllm_distributed_inference_analysis|18 分布式推理]] | 并行轴怎样映射到rank与通信，专家迁移和微批怎样保持次序？ | Engine、模型库、Attention与Serving → 分离式KV、在线更新 |
| [[19_vllm_compilation_cudagraph_analysis|19 编译与CUDA Graph]] | 动态shape怎样选择编译区域、捕获与重放，并在不兼容时回退？ | Attention与两代Runner → IR、融合算子 |
| [[20_vllm_fused_ops_and_kernels_analysis|20 融合算子与Kernel]] | 融合具体减少哪些中间读写，专家与multi-LoRA计算怎样重排，何时不能使用？ | Attention、量化、编译与IR → 设备性能验证 |
| [[21_vllm_ir_and_fusion_passes_analysis|21 IR与融合Pass]] | 语义算子、原地修改和融合怎样安全改写并交给后端？ | 量化与编译图 → 融合算子 |
| [[22_vllm_disaggregated_kv_serving_analysis|22 分离式KV Serving]] | 跨Engine的KV怎样发现、传输、证明有效并在失败时解除持有？ | Scheduler、KV、Serving与分布式 → 可观测性 |
| [[23_vllm_observability_reliability_analysis|23 可观测性与可靠性]] | 指标、事件与追踪怎样产生，故障怎样传播、清理与恢复？ | 调试与排障 → Scheduler、KV、Serving与分布式 |
| [[24_vllm_extension_plugin_system_analysis|24 扩展与插件]] | 插件怎样按进程与应用生命周期发现、选择和初始化？ | 请求语义、模型库与Serving → 在线更新、可观测性 |
| [[25_vllm_weight_transfer_online_update_analysis|25 在线权重更新]] | 权重字节、rank更新、版本标签和缓存何时真正对请求可见？ | Engine、KV、模型库、Runner与分布式 → 可观测性 |
| [[26_vllm_multiproc_executor_rpc_deepdive|26 MultiprocExecutor专题]] | 本机workers怎样启动、广播同一RPC、形成背压、配对响应并在失败时收尾？ | Engine与分布式 → Serving、Runner与可观测性 |

## Related Pages

- [[02_engineering/03_infer_frameworks/index|推理框架目录]] — 回到各框架和专题的本级入口。
- [[02_engineering/03_infer_frameworks/01_llm_inference_technology_stack_analysis|大模型推理技术栈全景]] — 比较不同推理系统的能力边界。
- [[02_engineering/03_infer_frameworks/sglang/index|SGLang推理框架]] — 对照另一条编译实现路线。
- [[02_engineering/03_infer_frameworks/speculative_decoding/index|投机推理专题]] — 跨引擎理解draft/verify算法族。
- [[02_engineering/03_infer_frameworks/mooncake_analysis|Mooncake分离式推理]] — 对照论文层面的KV数据平面设计。
