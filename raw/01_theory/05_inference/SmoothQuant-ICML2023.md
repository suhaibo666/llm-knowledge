# SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models

| 项 | 值 |
|---|---|
| 正式版本 | [ICML 2023，PMLR 202，原文](https://proceedings.mlr.press/v202/xiao23c/xiao23c.pdf) |
| 作者 | Guangxuan Xiao、Ji Lin、Mickael Seznec、Hao Wu、Julien Demouth、Song Han |
| 本库访问 | 2026-09-17 |

## 本批核验位置

- §3、Fig. 2：激活离群值令 W8A8 的激活量化较难；论文实验条件不可外推。
- §4、Eq. (3)–(4)、Fig. 5：对角缩放的浮点等价式、校准样本上的尺度估计、迁移强度 α。
- §5、Tables/Figs：量化方案与硬件执行结果，等价变换不保证量化后逐位无误差。

本文件只做来源索引；不存放第三方论文 PDF。对应原理页为 T20。
