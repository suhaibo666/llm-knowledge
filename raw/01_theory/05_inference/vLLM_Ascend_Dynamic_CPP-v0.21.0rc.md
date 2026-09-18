# vLLM-Ascend Dynamic Chunked Pipeline Parallel（v0.21.0rc）

- **来源**：vLLM-Ascend 官方版本化[设计文档](https://docs.vllm.ai/projects/ascend/en/v0.21.0rc/developer_guide/Design_Documents/dynamic_chunked_pipeline_parallel.html)；阅读快照 2026-09-17。
- **版本与定位**：路径 `v0.21.0rc`；“Problem Statement”“Solution Overview”“Quadratic Latency Model”“Runtime Phase”“Constraints”。
- **用途**：固定大小 chunk 在历史增长时可能耗时不等；一个工程变体以 profile、二次模型与运行时校准调整 chunk 长度；该版本要求 PP 和 chunked prefill。
- **边界**：它把工程特性称作 Dynamic CPP；本库 T29 的 CPP 基础定义为层 stage 与上下文 chunk 形成流水，动态调 chunk 只是其中一种均衡策略。文档的模型与开销不可当作所有 CPP 实现的性质。
