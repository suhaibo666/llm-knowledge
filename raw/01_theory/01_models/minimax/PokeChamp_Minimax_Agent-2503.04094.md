# PokéChamp: an Expert-level Minimax Language Agent

> **本地 PDF 已于 2026-08-26 移除**：本库迁移至 GitHub 公开仓库后不再随库分发第三方论文原文，仅保留来源链接与元数据。
> 原文请从下方官方来源获取。

| 项 | 值 |
|---|---|
| arXiv | [arXiv:2503.04094](https://arxiv.org/abs/2503.04094) |
| PDF | https://arxiv.org/pdf/2503.04094 |
| 提交日期 | 2025-03-06 |
| 最后更新 | 2025-03-06 |
| 主分类 | cs.LG |
| 作者 | Seth Karten、Andy Luu Nguyen、Chi Jin |
| 原文件名 | `PokeChamp_Minimax_Agent-2503.04094.pdf` |

## 摘要

We introduce PokéChamp, a minimax agent powered by Large Language Models (LLMs) for Pokémon battles. Built on a general framework for two-player competitive games, PokéChamp leverages the generalist capabilities of LLMs to enhance minimax tree search. Specifically, LLMs replace three key modules: (1) player action sampling, (2) opponent modeling, and (3) value function estimation, enabling the agent to effectively utilize gameplay history and human knowledge to reduce the search space and address partial observability. Notably, our framework requires no additional LLM training. We evaluate PokéChamp in the popular Gen 9 OU format. When powered by GPT-4o, it achieves a win rate of 76% against the best existing LLM-based bot and 84% against the strongest rule-based bot, demonstrating its superior performance. Even with an open-source 8-billion-parameter Llama 3.1 model, PokéChamp consistently outperforms the previous best LLM-based bot, Pokéllmon powered by GPT-4o, with a 64% win rate. PokéChamp attains a projected Elo of 1300-1500 on the Pokémon Showdown online ladder, placing it among the top 30%-10% of human players. In addition, this work compiles the largest real-player Pokémon battle dataset, featuring over 3 million games, including more than 500k high-Elo matches. Based on this dataset, we establish a series of battle benchmarks and puzzles to evaluate specific battling skills. We further provide key updates to the local game engine. We hope this work fosters further research that leverage Pokémon battle as benchmark to integrate LLM technologies with game-theoretic algorithms addressing general multiagent problems. Videos, code, and dataset available at https://sites.google.com/view/pokechamp-llm.
