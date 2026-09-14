# DeepGEMM 26/09 Mega-mHC 来源索引

- 发布方：deepseek-ai / DeepGEMM。
- 官方发布说明：https://github.com/deepseek-ai/DeepGEMM/pull/432 ，发布/合并 2026-09-10，读取 2026-09-11（北京时间）。
- 声明：Mega-mHC 融合 Hyper-Connection 操作和 RMSNorm，报告 45%–85% speedup；页面未列完整逐点测量环境和延迟表。本索引不是独立复测记录。
- 代码补证基线：`deepseek-ai/DeepGEMM@39d8c4cacc2c07c1fa9921c6c29c6a8a2da75359`；源码不复制进 raw。稳定路径：`csrc/apis/mega_mhc.hpp`、`deep_gemm/include/deep_gemm/impls/sm100_mega_mhc.cuh`、`tests/test_mega_mhc.py`、`third-party/tilelang_ops/ref_mhc.py`。
- 概念核对：https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html ，Roofline 部分，读取 2026-09-11。
- 归纳页面：`wiki/01_theory/01_models/deepseek/19_deepseek_v4_1_flash_analysis.md` §2.5。HF 模型/技术报告原有冻结基线保持不变。
