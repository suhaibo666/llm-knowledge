# vLLM Chunked Prefill 官方文档快照

> 动态文档索引；访问快照：2026-09-17。

| 项 | 值 |
|---|---|
| 文档 | [vLLM Optimization and Tuning：Chunked Prefill](https://docs.vllm.ai/en/latest/configuration/optimization/) |
| 访问定位 | `Chunked Prefill` 与 `Performance Tuning with Chunked Prefill` 小节 |
| 本库取证范围 | V1 在可用时默认启用；先排全部 pending decode，再在 `max_num_batched_tokens` 余量中安排 prefill；不能装入时切分；较小 budget 有利 ITL、较大 budget 有利 TTFT 的该版本文档建议。 |

该文档描述的是访问日的 vLLM 策略与调参建议。T16 不将其默认、数字示例或策略外推为通用机制。
