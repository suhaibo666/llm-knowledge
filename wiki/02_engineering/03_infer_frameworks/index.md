---
title: "推理框架 —— 目录索引"
---

# 推理框架 —— 目录索引

> 覆盖 LLM 推理技术栈、服务引擎、投机推理与分离式 KV 架构。
> **最后更新**：2026-09-08。本页只维护本级入口；各框架的机制归属、源码基线和核验范围见其域索引与正文。

## 总览与独立页面

| 页面 | 内容范围 | 来源与状态 |
|---|---|---|
| [[02_engineering/03_infer_frameworks/01_llm_inference_technology_stack_analysis|大模型推理技术栈全景]] | 模型制品、调度、KV、编译、Kernel、通信与服务的分层，以及主要推理框架的场景定位 | 技术栈总览；按页头观察时间与源码基线阅读 |
| [[02_engineering/03_infer_frameworks/mooncake_analysis|Mooncake 分离式推理]] | P/D 分离、分布式 KV Cache、RDMA 与存储数据平面 | arXiv:2407.00079；论文分析，不是引擎源码审计 |

## 子域入口

| 子域 | 入口 | 内容数量 | 覆盖边界 |
|---|---|---:|---|
| vLLM | [[02_engineering/03_infer_frameworks/vllm/index|vLLM 推理引擎知识地图]] | 25 篇正文 + index | 使用、调优、排障、架构与机制五类入口；逐页源码核验状态见域索引 |
| 投机推理 | [[02_engineering/03_infer_frameworks/speculative_decoding/index|投机推理专题]] | 2 篇正文 + index | 跨引擎的 draft/verify 技术专题；各算法与实现基线见正文 |
| SGLang | [[02_engineering/03_infer_frameworks/sglang/index|SGLang 推理框架]] | 1 篇正文 + index | 当前聚焦编译 Pass，与 vLLM 对照；尚非全系统源码覆盖 |

## Related Pages

- [[02_engineering/02_train_frameworks/index|训练框架目录]] — 追踪权重训练与训练侧的执行机制。
- [[02_engineering/04_posttrain_frameworks/index|后训练框架目录]] — 了解调用推理引擎的 rollout 与更新编排。
- [[02_engineering/01_pytorch/index|PyTorch 目录]] — 深入推理引擎依赖的框架、编译和 CUDA Graph 通用机制。
- [[02_engineering/05_gpu_kernel/index|GPU Kernel 开发]] — 接续具体 Kernel 内部的 tile、流水和设备资源分析。
- [[01_theory/01_models/index|模型架构]] — 解释 GQA、MLA、MoE 等模型结构本身。
- [[01_theory/05_inference/index|推理技术理论]] — 连接推理算法的理论层。
