# 推理框架 —— 目录索引

> 覆盖 LLM 推理技术栈、服务引擎、投机推理与分离式 KV 架构
> **最后更新**：2026-08-18

---

## 总览入口

| 页面 | 核心问题 |
|---|---|
| [[02_engineering/03_infer_frameworks/01_llm_inference_technology_stack_analysis|大模型推理技术栈全景]] | 从模型制品、调度、KV、编译、kernel、通信到 serving 的完整分层；Transformers、vLLM、SGLang、TensorRT-LLM、llama.cpp 与 TGI 的场景定位 |

## 子框架

| 子目录 | 入口 | 页数 | 核心主题 |
|---|---|---:|---|
| **vLLM** | [[02_engineering/03_infer_frameworks/vllm/index|vLLM 推理引擎知识地图]] | 12 + index | V1 EngineCore、continuous batching、paged KV、模型/注意力、投机、量化、并行、编译/CUDA Graph 与融合 kernel；架构入口已重验到 `f4b161d7fc` |
| **投机推理** | [[02_engineering/03_infer_frameworks/speculative_decoding/index|投机推理专题]] | 2 + index | MTP、EAGLE3、DFlash、DSpark 的 draft/verify 机制、接受率与代价模型 |
| **SGLang** | [[02_engineering/03_infer_frameworks/sglang/index|SGLang 推理框架]] | 1 + index | SGLang 编译 Pass 与 vLLM piecewise CUDA Graph 管线对照 |

## 独立页面

| 页面 | 来源 | 核心主题 |
|---|---|---|
| [[02_engineering/03_infer_frameworks/mooncake_analysis|Mooncake 分离式推理]] | Mooncake / Kimi | P/D 分离、分布式 KV Cache、RDMA 与存储数据平面 |

## Related Pages

- [[02_engineering/02_train_frameworks/index|训练框架目录]]
- [[02_engineering/01_ai_frameworks/index|AI 框架目录]]
- [[01_theory/05_inference/index|推理技术理论]]
- [[02_engineering/03_infer_frameworks/vllm/index|vLLM 推理引擎知识地图]]
