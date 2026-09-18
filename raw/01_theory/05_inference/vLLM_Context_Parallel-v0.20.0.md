# vLLM Context Parallel Deployment（v0.20.0）

- **来源**：vLLM 官方版本化文档，[Context Parallel Deployment](https://docs.vllm.ai/en/v0.20.0/serving/context_parallel_deployment/)；阅读快照 2026-09-17。
- **版本与定位**：文档路径 `v0.20.0`；“Prefill Context Parallel”“Decode Context Parallel”“Technical Discussions”。页尾标注 2026-01-29。
- **用途**：PCP 的 partial Q/full KV 与 partial Q/partial KV 两条策略；DCP 的序列维 KV 分片、TP 头数限制、交错放置、通信取舍与模型示例。
- **边界**：PCP 段明确说两条路径仍在开发；该页不是某后续版本全部 backend 的兼容性保证。DCP 中 `dcp_size <= tp_size/H` 是本版本的实现与配置选择，不是序列维切分的数学上界。

