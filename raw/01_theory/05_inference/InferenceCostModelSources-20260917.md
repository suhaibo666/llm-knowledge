# 推理性能与资源成本模型：官方指标文档索引

> 本索引记录 `11_inference_cost_model_analysis.md` 使用的官方指标文档。它不代替原文；动态网页按下列访问快照核验。论文另用独立来源页：`raw/01_theory/05_inference/Roofline-2009.md` 与 `raw/01_theory/05_inference/PagedAttention-2309.06180.md`。

| 来源 | 冻结版本或快照 | 已核验定位 | 本页采用的证据范围 |
|---|---|---|---|
| NVIDIA NIM LLM Benchmarking Metrics | 文档版本 1.0.0，最后更新 2026-04-01；[官方页面](https://docs.nvidia.com/nim/benchmarking/llm/1.0.0/metrics.html) | TTFT、E2E、ITL、TPS、RPS 小节 | 流式指标的示例计时边界和 TPS 分母差异 |
| NVIDIA AIPerf Metrics Reference | 动态官方文档，访问快照 2026-09-17；[官方页面](https://docs.nvidia.com/aiperf/dev/reference/ai-perf-metrics-reference) | TTFT、ITL、Output Token Throughput Per User、E2E Output Token Throughput 小节 | 客户端 TTFT、流内 ITL 和单用户吞吐不可与系统吞吐混比 |
| NVIDIA GenAI-Perf | 动态官方文档，访问快照 2026-09-17；[官方页面](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_benchmark/genai-perf-README.html) | Overview、Metrics | LLM benchmark 的 TTFT、ITL、请求和输出 token 吞吐指标；并发或请求率属于负载输入 |

本页的显存公式、固定模型、虚构设备和 Roofline 数字为教学推导，非上述来源的模型配置、硬件规格或运行测量。
