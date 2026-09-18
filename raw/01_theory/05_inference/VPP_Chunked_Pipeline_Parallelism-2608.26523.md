# VPP: Virtual Pipeline Parallelism for Efficient Chunked Prefill in Long-Context LLM Inference

- **来源**：[arXiv:2608.26523v1](https://arxiv.org/html/2608.26523v1)，2026-08-27；[arXiv 元数据](https://arxiv.org/abs/2608.26523)与版本化 HTML 均列 Yan Shi、Xiaochao Wang、Jingchun Gao 等十名作者。
- **定位**：§1 Introduction；§2.4 Chunked Prefill Pipeline Parallelism；§4 的固定 chunk 成本动机、§5 的 VPP 调度；实验条件见 §6.1。
- **用途**：CPP 由分块 prefill 与 PP 组合而来；等长后续块会读取更长前缀，执行时间可能递增并产生气泡；动态改 chunk、虚拟 stage 是不同的后续均衡策略。
- **边界**：本库 T29 解释 CPP 基础，不把该论文的 VPP 结果写作 CPP 的普遍收益，也不把其特定 MoE/NPU 实验当成本页教学时间线。
