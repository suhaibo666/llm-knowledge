# LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale

- **作者**：Tim Dettmers 等
- **固定版本**：arXiv:2208.07339v2，2022-11-10；NeurIPS 2022 camera-ready
- **原文**：[arXiv PDF](https://arxiv.org/pdf/2208.07339v2)，[arXiv HTML](https://arxiv.org/html/2208.07339v2)
- **本库使用定位**：
  - §3.1，Eq. (7)：向量级缩放与反量化矩阵乘法；
  - §3.2，Eq. (8)：在该方法中把少量离群 feature dimension 留给 16-bit 路径；
  - §4.1–4.2：论文所测 Transformer 中的大幅值 feature 现象；
  - Appendix D：混合分解和内核实现会改变实际推理速度。
- **使用边界**：该论文的“约 0.1%”和阈值等是其模型与设置中的经验观察，不是任意模型都必然成立的分布定律。

