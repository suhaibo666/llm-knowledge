# CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Benchmarking on HumanEval-X

> **本地 PDF 已于 2026-08-26 移除**：本库迁移至 GitHub 公开仓库后不再随库分发第三方论文原文，仅保留来源链接与元数据。
> 原文请从下方官方来源获取。

> ⚠️ **ID 更正**：原文件名中的 `2306.03078` 实际对应无关论文《SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression》，正确 arXiv ID 为 `2303.17568`（已按论文标题在 arXiv 检索核对）。

| 项 | 值 |
|---|---|
| arXiv | [arXiv:2303.17568](https://arxiv.org/abs/2303.17568) |
| PDF | https://arxiv.org/pdf/2303.17568 |
| 提交日期 | 2023-03-30 |
| 最后更新 | 2024-07-10 |
| 主分类 | cs.LG |
| 作者 | Qinkai Zheng、Xiao Xia、Xu Zou　等 13 人 |
| 原文件名 | `CodeGeeX-2306.03078.pdf` |

## 摘要

Large pre-trained code generation models, such as OpenAI Codex, can generate syntax- and function-correct code, making the coding of programmers more productive and our pursuit of artificial general intelligence closer. In this paper, we introduce CodeGeeX, a multilingual model with 13 billion parameters for code generation. CodeGeeX is pre-trained on 850 billion tokens of 23 programming languages as of June 2022. Our extensive experiments suggest that CodeGeeX outperforms multilingual code models of similar scale for both the tasks of code generation and translation on HumanEval-X. Building upon HumanEval (Python only), we develop the HumanEval-X benchmark for evaluating multilingual models by hand-writing the solutions in C++, Java, JavaScript, and Go. In addition, we build CodeGeeX-based extensions on Visual Studio Code, JetBrains, and Cloud Studio, generating 4.7 billion tokens for tens of thousands of active users per week. Our user study demonstrates that CodeGeeX can help to increase coding efficiency for 83.4% of its users. Finally, CodeGeeX is publicly accessible and in Sep. 2022, we open-sourced its code, model weights (the version of 850B tokens), API, extensions, and HumanEval-X at https://github.com/THUDM/CodeGeeX.
