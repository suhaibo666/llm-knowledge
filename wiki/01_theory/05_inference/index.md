---
title: "LLM 推理基础原理 — 目录索引"
---

# LLM 推理基础原理

本域解释引擎无关的生成语义、执行成本、状态管理和服务优化原理；具体源码、配置和兼容矩阵由[[02_engineering/03_infer_frameworks/index|推理框架工程域]]负责。下表列出本域 29 篇已完成正文，按阅读主题选择入口。

> **状态**：2026-09-17 完成并交叉审阅全部 29 篇正文。来源、算例和核验记录见 `docs/research/2026-09-17-inference-principles-source-ledger.md`。

## 分段与本级入口

`01` 是导论；`10–19` 是核心机制；`20–35` 是专题，因超过十篇延伸到相邻的 30 段；`40–41` 是方法页。编号是阅读组织，建设顺序按前置依赖。

| 编号 | 页面入口 | 读者问题 | 状态 |
|---|---|---|---|
| 01 | [[01_inference_overview_analysis|LLM 推理原理全景：一条请求如何变成可交付的 token]] | 一条请求经过哪些计算与服务环节？ | 已完成，交叉审阅通过 |
| 10 | [[10_prefill_decode_analysis|自回归生成与 Prefill / Decode]] | 已知输入与未知输出为何采用不同执行节奏？ | 已完成，交叉审阅通过 |
| 11 | [[11_inference_cost_model_analysis|推理性能与资源成本模型]] | 延迟、吞吐、显存和带宽怎样记账？ | 已完成，交叉审阅通过 |
| 12 | [[12_kv_cache_analysis|KV Cache：复用依据与容量]] | 历史 KV 何时可复用、容量如何增长？ | 已完成，交叉审阅通过 |
| 13 | [[13_paged_kv_attention_analysis|分页 KV 与 PagedAttention]] | 逻辑位置怎样映射到非连续物理 KV 块？ | 已完成，交叉审阅通过 |
| 14 | [[14_prefix_caching_analysis|Prefix Caching：跨请求前缀复用]] | 不同请求怎样安全复用同一前缀？ | 已完成，交叉审阅通过 |
| 15 | [[15_continuous_batching_analysis|Continuous Batching 与请求调度]] | 到达与结束时间不同的请求怎样逐轮同批？ | 已完成，交叉审阅通过 |
| 16 | [[16_chunked_prefill_analysis|Chunked Prefill：长输入的分步执行]] | 长输入怎样分步执行而保留历史依赖？ | 已完成，交叉审阅通过 |
| 17 | [[17_sampling_decoding_analysis|采样与解码策略]] | logits 怎样变成 token 序列？ | 已完成，交叉审阅通过 |
| 18 | [[18_efficient_attention_analysis|高效 Attention：分块、在线归一化与 IO]] | 分块和在线归一化怎样减少注意力 IO？ | 已完成，交叉审阅通过 |
| 19 | [[19_inference_quantization_analysis|推理量化基础：表示、误差与执行成本]] | 低精度表示的收益、误差和执行成本是什么？ | 已完成，交叉审阅通过 |
| 20 | [[20_quantization_methods_analysis|推理量化算法：校准、误差补偿与尺度迁移]] | 校准与误差控制算法怎样工作？ | 已完成，交叉审阅通过 |
| 21 | [[21_kv_compression_analysis|KV 压缩与选择性保留：精度、位置和表示三条轴]] | KV 怎样压缩或选择性保留？ | 已完成，交叉审阅通过 |
| 22 | [[22_kv_tiering_transfer_analysis|KV 分层存储与迁移：命中、完成和可读边界]] | KV 怎样跨存储层迁移？ | 已完成，交叉审阅通过 |
| 23 | [[23_speculative_decoding_analysis|投机解码基础：草稿、验证与分布保持]] | 草稿与目标模型怎样验证并维持分布？ | 已完成，交叉审阅通过 |
| 24 | [[24_speculative_decoding_variants_analysis|投机解码变体与草稿设计：候选从哪里来，怎样验证]] | 草稿与树验证有哪些取舍？ | 已完成，交叉审阅通过 |
| 25 | [[25_multi_lora_serving_analysis|Multi-LoRA Serving：共享基座下的多适配器执行]] | 多适配器怎样共享基座并执行？ | 已完成，交叉审阅通过 |
| 26 | [[26_prefill_decode_disaggregation_analysis|Prefill/Decode 分离：交接、排队与资源配比]] | Prefill 与 Decode 分离时如何交接状态？ | 已完成，交叉审阅通过 |
| 27 | [[27_prefill_context_parallelism_analysis|PCP：将长 Prefill 的 Query 工作分给多卡]] | PCP 怎样分担长输入的上下文计算？ | 已完成，交叉审阅通过 |
| 28 | [[28_decode_context_parallelism_analysis|DCP：Decode 时按序列分担 KV 与合并输出]] | DCP 怎样分担长 KV 的读取与归一化？ | 已完成，交叉审阅通过 |
| 29 | [[29_chunked_pipeline_parallelism_analysis|CPP：Prefill Chunk 穿过层间流水线]] | CPP 怎样把 prefill chunk 穿过层间流水线？ | 已完成，交叉审阅通过 |
| 30 | [[30_inference_parallelism_composition_analysis|推理并行组合：先算容量，再核对通信与布局]] | 推理并行轴怎样组合与布局？ | 已完成，交叉审阅通过 |
| 31 | [[31_moe_inference_analysis|MoE 推理：路由负载、专家副本与完成边界]] | MoE 路由偏斜怎样影响推理延迟？ | 已完成，交叉审阅通过 |
| 32 | [[32_constrained_decoding_analysis|约束解码：把格式规则落实到逐 token 状态]] | 格式约束怎样限制可采样 token？ | 已完成，交叉审阅通过 |
| 33 | [[33_multimodal_inference_analysis|多模态推理：媒体编码、运行状态与联合调度]] | 媒体编码、状态和语言生成怎样协作？ | 已完成，交叉审阅通过 |
| 34 | [[34_hybrid_state_inference_analysis|混合模型推理：KV 与循环状态协作]] | KV 与循环状态怎样一起推进和恢复？ | 已完成，交叉审阅通过 |
| 35 | [[35_inference_execution_optimization_analysis|推理执行优化：融合、编译、CUDA Graph 与重叠]] | 融合、编译、图和重叠各减少什么成本？ | 已完成，交叉审阅通过 |
| 40 | [[40_inference_benchmarking_guide|推理性能评测：从指标到可信结论]] | 怎样把性能指标测成可复核结论？ | 已完成，交叉审阅通过 |
| 41 | [[41_inference_optimization_guide|推理优化组合：从瓶颈证据到可回滚的方案]] | 怎样从观测瓶颈选择并验证优化组合？ | 已完成，交叉审阅通过 |

## 已有基础与范围边界

- [[01_theory/01_models/attention_is_all_you_need_analysis|Transformer 架构]]给出因果注意力的模型背景；[[01_theory/06_distributed_parallelism/index|分布式并行原理]]拥有通用 TP、PP、CP、EP 和通信代价。
- [[02_engineering/03_infer_frameworks/index|推理框架工程域]]保留具体引擎的源码路径与实现限制。本域的 PCP、DCP、CPP 分别是 Prefill Context Parallelism、Decode Context Parallelism、Chunked Pipeline Parallelism；DCP 在此不指 Dynamic CP。
- 原目录留下的 Agent、ReAct、CoT、RAG 和工具使用线索仍属后续范围，不混入本批 29 篇推理执行原理。ReAct 原论文的来源索引为 `raw/01_theory/05_inference/ReAct_Reasoning_Acting-2210.03629.md`；工具使用的对齐与 RLHF 方法由 [[01_theory/04_posttraining/index|后训练对齐]]承接。
