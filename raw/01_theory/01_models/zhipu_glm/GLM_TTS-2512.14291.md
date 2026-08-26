# GLM-TTS Technical Report

> **本地 PDF 已于 2026-08-26 移除**：本库迁移至 GitHub 公开仓库后不再随库分发第三方论文原文，仅保留来源链接与元数据。
> 原文请从下方官方来源获取。

| 项 | 值 |
|---|---|
| arXiv | [arXiv:2512.14291](https://arxiv.org/abs/2512.14291) |
| PDF | https://arxiv.org/pdf/2512.14291 |
| 提交日期 | 2025-12-16 |
| 最后更新 | 2025-12-16 |
| 主分类 | cs.SD |
| 作者 | Jiayan Cui、Zhihan Yang、Naihan Li　等 13 人 |
| 原文件名 | `GLM_TTS-2512.14291.pdf` |

## 摘要

This work proposes GLM-TTS, a production-level TTS system designed for efficiency, controllability, and high-fidelity speech generation. GLM-TTS follows a two-stage architecture, consisting of a text-to-token autoregressive model and a token-to-waveform diffusion model. With only 100k hours of training data, GLM-TTS achieves state-of-the-art performance on multiple open-source benchmarks. To meet production requirements, GLM-TTS improves speech quality through an optimized speech tokenizer with fundamental frequency constraints and a GRPO-based multi-reward reinforcement learning framework that jointly optimizes pronunciation, speaker similarity, and expressive prosody. In parallel, the system enables efficient and controllable deployment via parameter-efficient LoRA-based voice customization and a hybrid phoneme-text input scheme that provides precise pronunciation control. Our code is available at https://github.com/zai-org/GLM-TTS. Real-time speech synthesis demos are provided via Z.ai (audio.z.ai), the Zhipu Qingyan app/web (chatglm.cn).
