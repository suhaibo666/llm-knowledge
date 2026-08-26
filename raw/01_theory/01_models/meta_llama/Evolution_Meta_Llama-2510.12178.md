# Evolution of meta's llama models and parameter-efficient fine-tuning of large language models: a survey

> **本地 PDF 已于 2026-08-26 移除**：本库迁移至 GitHub 公开仓库后不再随库分发第三方论文原文，仅保留来源链接与元数据。
> 原文请从下方官方来源获取。

| 项 | 值 |
|---|---|
| arXiv | [arXiv:2510.12178](https://arxiv.org/abs/2510.12178) |
| PDF | https://arxiv.org/pdf/2510.12178 |
| 提交日期 | 2025-10-14 |
| 最后更新 | 2025-10-14 |
| 主分类 | cs.AI |
| 作者 | Abdulhady Abas Abdullah、Arkaitz Zubiaga、Seyedali Mirjalili　等 8 人 |
| 原文件名 | `Evolution_Meta_Llama-2510.12178.pdf` |

## 摘要

This review surveys the rapid evolution of Meta AI's LLaMA (Large Language Model Meta AI) series - from LLaMA 1 through LLaMA 4 and the specialized parameter-efficient fine-tuning (PEFT) methods developed for these models. We first describe the LLaMA family of foundation models (7B-65B to 288B parameters), their architectures (including native multimodal and Mixtureof-Experts variants), and key performance characteristics. We then describe and discuss the concept of PEFT, which adapts large pre-trained models by updating only a small subset of parameters, and review five PEFT methods that have been applied to LLaMA: LoRA (Low-Rank Adaptation), LLaMA-Adapter V1 and V2, LLaMA-Excitor, and QLoRA (Quantized LoRA). We discuss each method's mechanism, parameter savings, and example application to LLaMA (e.g., instruction tuning, multimodal tasks). We provide structured discussion and analysis of model and adapter architectures, parameter counts, and benchmark results (including examples where fine-tuned LLaMA models outperform larger baselines). Finally, we examine real-world use cases where LLaMA-based models and PEFT have been successfully applied (e.g., legal and medical domains), and we discuss ongoing challenges and future research directions (such as scaling to even larger contexts and improving robustness). This survey paper provides a one-stop resource for ML researchers and practitioners interested in LLaMA models and efficient fine-tuning strategies.
