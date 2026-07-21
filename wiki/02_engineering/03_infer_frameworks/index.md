# 推理框架 — 目录索引

> 覆盖 LLM 推理框架的架构与实现（vLLM、TRT-LLM、Mooncake 等）
> 最后更新: 2026-06-22

---

## 子框架

| 子目录 | 入口 | 页数 | 核心主题 |
|------|------|------|---------|
| **vLLM** | [[vllm/index]] | 12 + index | vLLM V1 推理引擎源码级分析,三支柱:**调度**(连续批处理/分页 KV/抢占)、**模型库**(注册/层库/注意力后端)、**特性优化**(投机解码/量化/分布式/编译&CUDA Graph/算子融合&Triton);每篇 Overview→Quick Start→Deep Dive |
| **投机推理** | [[speculative_decoding/index]] | 2 + index | 投机解码草稿器演进 **MTP → Eagle3 → DFlash → DSpark**:三代如何依次拨动 `L=(T_draft+T_verify)/τ` 的三个杆;DSpark 论文(半自回归 + 置信度调度)深挖 + 开源仓 **DeepSpec** 源码级核对 |
| **SGLang** | [[sglang/index]] | 1 + index | SGLang 编译 Pass 与 torch.compile 适配:`srt/compilation/` 是 vLLM piecewise-cudagraph 管线的近逐文件 fork,**融合 pass 被抽空**(唯一 pass 为 no-op、真实图重写=0);与 [[vllm/index]] 互为对照 |

## 页面列表

| 页面 | 来源 | 核心主题 |
|------|------|---------|
| [[mooncake_analysis]] | Mooncake (Kimi) | 分离式推理架构, 分布式 KV Cache, Prefill/Decode 池分离, RDMA |

---

## 关联域

- [[../02_train_frameworks/index]] — 训练框架
- [[../01_ai_frameworks/index]] — AI 框架（推理图优化）
- [[../../01_theory/05_inference/index]] — 推理技术理论
