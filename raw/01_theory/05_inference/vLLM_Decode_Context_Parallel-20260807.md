# Efficient Decode Context Parallelism with vLLM for Long Context Workloads

- **来源**：vLLM 官方[技术博客](https://vllm.ai/blog/2026-08-07-decode-context-parallelism)，2026-08-07；阅读快照 2026-09-17。
- **定位**：“Challenges of Serving Long Contexts”“What is DCP”“Decode Context Parallelism Process”“MLA Backend”“GQA Backend”。
- **用途**：DCP 复用本会重复的 KV 容量、`AllGather Q → local attention → AllGather + ReduceScatter` 的示意路径、MLA/GQA 实现范围和 TP/DCP 约束。
- **边界**：博客中的 B200/Kimi K2.6 性能数字有特定模型、精度、硬件和并发条件；不能推广到任意 DCP 部署。数值合并恒等式应由 attention 数学推导而非基准结果验证。

