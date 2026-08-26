# DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence

> **本地 PDF 已于 2026-08-26 移除**：本库迁移至 GitHub 公开仓库后不再随库分发第三方论文原文，仅保留来源链接与元数据。
> 原文请从下方官方来源获取。

> 原文件名未含 arXiv ID；经正文标题与本库 `docs/research/INGEST_MANIFEST_block1_tim.md` 记录核对为 arXiv:2606.19348。

| 项 | 值 |
|---|---|
| arXiv | [arXiv:2606.19348](https://arxiv.org/abs/2606.19348) |
| PDF | https://arxiv.org/pdf/2606.19348 |
| 提交日期 | 2026-04-26 |
| 最后更新 | 2026-04-26 |
| 主分类 | cs.CL |
| 作者 |  DeepSeek-AI、Anyi Xu、Bangcai Lin　等 319 人 |
| 原文件名 | `DeepSeek_V4.pdf` |

## 摘要

We present a preview version of DeepSeek-V4 series, including two strong Mixture-of-Experts (MoE) language models -- DeepSeek-V4-Pro with 1.6T parameters (49B activated) and DeepSeek-V4-Flash with 284B parameters (13B activated) -- both supporting a context length of one million tokens. DeepSeek-V4 series incorporate several key upgrades in architecture and optimization: (1) a hybrid attention architecture that combines Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA) to improve long-context efficiency; (2) Manifold-Constrained Hyper-Connections (mHC) that enhance conventional residual connections; (3) and the Muon optimizer for faster convergence and greater training stability. We pre-train both models on more than 32T diverse and high-quality tokens, followed by a comprehensive post-training pipeline that unlocks and further enhances their capabilities. DeepSeek-V4-Pro-Max, the maximum reasoning effort mode of DeepSeek-V4-Pro, redefines the state-of-the-art for open models, outperforming its predecessors in core tasks. Meanwhile, DeepSeek-V4 series are highly efficient in long-context scenarios. In the one-million-token context setting, DeepSeek-V4-Pro requires only 27% of single-token inference FLOPs and 10% of KV cache compared with DeepSeek-V3.2. This enables us to routinely support one-million-token contexts, thereby making long-horizon tasks and further test-time scaling more feasible. The model checkpoints are available at https://huggingface.co/collections/deepseek-ai/deepseek-v4.
