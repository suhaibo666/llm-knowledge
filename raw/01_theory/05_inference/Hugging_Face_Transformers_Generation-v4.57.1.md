# Hugging Face Transformers Generation

> 本库仅保留官方文档与源码定位；该来源描述一个库版本的生成合同，不代表所有推理引擎。

| 项 | 值 |
|---|---|
| 发布基线 | [`transformers` v4.57.1](https://github.com/huggingface/transformers/tree/v4.57.1)，commit [`8cb5963`](https://github.com/huggingface/transformers/commit/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc)，2025-10-14 |
| 官方生成文档 | [GenerationConfig / text generation](https://huggingface.co/docs/transformers/v4.57.1/main_classes/text_generation) |
| 核心源码 | [`generation/logits_process.py`](https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/generation/logits_process.py)、[`generation/beam_search.py`](https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/generation/beam_search.py)、[`generation/stopping_criteria.py`](https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/generation/stopping_criteria.py) |
| 本库取证范围 | `TemperatureLogitsWarper`、`TopKLogitsWarper`、`TopPLogitsWarper`、`RepetitionPenaltyLogitsProcessor`；`BeamSearchScorer.process` 与 `BeamHypotheses.add/is_done`；`EosTokenCriteria` 与 `StopStringCriteria` |

本索引不代替官方文档或源码；引用具体合同前须重新打开对应符号。
