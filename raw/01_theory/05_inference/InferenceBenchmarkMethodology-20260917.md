# 推理性能评测方法：官方规则与指标文档索引

> 本文件只登记 T40 使用的原始规则与官方文档，不复制其正文。MLPerf 规则固定到下列提交，其余动态网页按 2026-09-17 访问快照取证；本库的教学协议不是 MLPerf 正式提交。

| 来源 | 基线与原文 | 已读定位 | 支撑的边界 |
|---|---|---|---|
| MLCommons MLPerf Inference Rules | [官方规则，2026-09-17 固定提交](https://github.com/mlcommons/inference_policies/blob/d3eba2f21026d868ad65cdcad2bb81e4a17ce3d3/inference_rules.adoc) | §3 Scenarios 及规则开头对完整 run、质量、延迟要求的定义 | Server/Interactive、Offline 使用不同的负载生成和约束；不能拿离线吞吐替代在线 SLO 结果。本页不声称遵循完整提交规则。 |
| NVIDIA AIPerf Metrics Reference | [官方指标参考，2026-09-17 访问快照](https://docs.nvidia.com/aiperf/reference/ai-perf-metrics-reference) | Record/Aggregate/Derived、Goodput、Good Request Fraction、Error Request Count、TTFT/ITL | 请求级分布与窗口级速率分开；失败请求进入 attempted 分母，goodput 需要预先声明 SLO。 |
| NVIDIA AIPerf Load Generator Options | [官方负载生成参考，2026-09-17 访问快照](https://docs.nvidia.com/aiperf/benchmark-modes/load-generator-options-reference) | Request Scheduling Options：request rate、concurrency、fixed schedule | 固定到达率和闭环并发是不同负载；请求率与并发上限合用时须记录实际发出时刻。 |
| NVIDIA NIM LLM Benchmarking Metrics | [1.0.0 固定文档](https://docs.nvidia.com/nim/benchmarking/llm/1.0.0/metrics.html) | TTFT、ITL、TPS 与基准窗口口径 | 指标时间戳、流式间隔和吞吐分母必须在报告中定义，不能混用。 |

T40 的 600 请求、两个配置、20-token 输出和 SLO 结果是本库构造的教学数据，不是上述文档或任何硬件的测量结果。
