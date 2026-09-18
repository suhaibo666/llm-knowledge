# FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU

| 项 | 值 |
|---|---|
| 正式版本 | [ICML 2023，PMLR 202，原文](https://proceedings.mlr.press/v202/sheng23a/sheng23a.pdf) |
| 作者 | Ying Sheng、Lianmin Zheng、Binhang Yuan、Zhuohan Li、Max Ryabinin、Beidi Chen、Percy Liang、Christopher Ré、Ion Stoica、Ce Zhang |
| 本库访问 | 2026-09-17 |

## 本批核验位置

- §1：GPU、CPU、磁盘三层容量和带宽取舍；论文目标是对延迟不敏感的吞吐型任务。
- §4.1、Algorithm 1：权重、cache、激活的装载/存储与计算重叠，并在依赖边界同步。
- §4.2：权重、激活、cache 各自可选择驻留层，不能把所有对象统称为 KV。
- §5：压缩是改变 KV 表示的独立手段，不等同于无损卸载。

本文件只做来源索引；不存放第三方论文 PDF。对应原理页为 T22。
