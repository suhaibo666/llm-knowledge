# Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference

- **作者**：Benoit Jacob 等
- **固定版本**：arXiv:1712.05877v1，2017-12-15；发表于 CVPR 2018
- **原文**：[arXiv PDF](https://arxiv.org/pdf/1712.05877v1)，[arXiv HTML](https://arxiv.org/html/1712.05877v1)
- **本库使用定位**：
  - §2.1：仿射量化关系 $r=S(q-Z)$、scale 与 zero-point 的含义；
  - §2.2：整数矩阵乘法、$M=S_1S_2/S_3$；
  - §2.3：zero-point 校正项和整数核心累加；
  - §2.4：uint8 乘法与 int32 accumulator、bias 的 scale；
  - §3.1：不同通道范围和离群权重会浪费量化精度。
- **使用边界**：本文以卷积网络的 8-bit 整数推理为对象。这里取其表示、缩放和累加的基础关系，不把实验结果泛化为所有 LLM 的吞吐率。

