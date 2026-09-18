# NVIDIA TensorRT Quantization Schemes

- **发布者**：NVIDIA
- **固定版本**：TensorRT 11.3.0 的 Quantization Schemes 与 Accuracy Considerations；Working with Quantized Types 为 2026-09-17 访问快照
- **原文**：[Quantization Schemes 11.3.0](https://docs.nvidia.com/deeplearning/tensorrt/11.3.0/inference-library/quantized-types-schemes.html)，[Accuracy Considerations 11.3.0](https://docs.nvidia.com/deeplearning/tensorrt/11.3.0/inference-library/accuracy-considerations.html)，[Working with Quantized Types（动态页，2026-09-17 快照）](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html)
- **本库使用定位**：
  - Quantization Schemes：INT8、INT4、FP8 的代码范围和缩放公式；
  - Working with Quantized Types：TensorRT 的对称量化约束及支持类型；
  - Accuracy Considerations：rounding、clamping 与 scale 的误差折中。
- **使用边界**：前两篇是版本化合同；第三篇没有可用的 11.3.0 固定 URL，故只能作为访问日快照。它们不等于通用 affine quantization 都能以任意非零 zero-point 在该版本运行。
