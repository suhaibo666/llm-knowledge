# vLLM-Ascend Context Parallel（v0.21.0rc）

- **来源**：vLLM-Ascend 官方版本化[设计文档](https://docs.vllm.ai/projects/ascend/en/v0.21.0rc/developer_guide/Design_Documents/context_parallel.html)和[用户指南](https://docs.vllm.ai/projects/ascend/en/v0.21.0rc/user_guide/feature_guide/context_parallel.html)；阅读快照 2026-09-17。
- **版本与定位**：路径 `v0.21.0rc`；设计文档 “Device Distribution”“Block Table”“Decode Context Parallel”“Prefill Context Parallel”；用户指南 “Supported Scenarios”“Constraints”。
- **用途**：区分 PCP 扩展设备域和 DCP 复用 TP 域；定位 PCP 的 head-tail 分配、逐层 all-gather KV、结果顺序恢复；定位 DCP 的 Q all-gather、局部 attention 与 `cp_lse_ag_out_rs` 合并。
- **边界**：实现分组、`cp_kv_cache_interleave_size`、具体 backend/组合支持属于该版本的工程合同，不构成 PCP/DCP 的通用定义。版本化文档是设计/使用说明，不等于对每一条代码执行路径的独立审计。

