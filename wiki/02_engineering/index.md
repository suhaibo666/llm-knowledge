# 工程实现 — 知识地图

覆盖大语言模型的工程基础设施：AI框架、训练框架、推理框架、后训练框架。

## 子领域

| 目录 | 核心主题 |
|------|---------|
| [[01_ai_frameworks/index]] | PyTorch 编译栈：Dynamo、Inductor、CUDA/NPU Graphs |
| [[02_train_frameworks/index]] | 训练框架：Megatron-LM 分布式、MindFormers |
| [[03_infer_frameworks/index]] | 推理框架：vLLM、TRT-LLM、Mooncake 分离式服务 |
| [[04_posttrain_frameworks/index]] | 后训练框架：RLHF 基础设施、对齐工具链（预留） |
| [[05_gpu_kernel/index]] | GPU Kernel 开发：执行层级、内存优化、Tensor Core/MMA、FlashAttention 链路、NPU 差异 |

## 页面列表

| 页面 | 来源 | 核心主题 |
|------|------|---------|
| [[05_gpu_kernel/gpu_kernel_guide]] | 综合深度分析 | GPU Kernel: 执行层级、内存优化、Tensor Core/MMA、torch.compile、FlashAttention、NPU 差异 |
| [[pin_memory_and_memory_semantics_analysis]] | 综合深度分析 | Pin Memory 与 RDMA 内存语义通信：PyTorch DataLoader、DeepSpeed ZeRO-Offload、vLLM KV Cache、Mooncake TransferEngine、DeepEP、3FS、NCCL 演进 |

## 关联域

- [[../01_theory/index]] — 理论研究（模型、预训练、后训练算法）

## Related Pages

- [[../index]] — 知识库总索引
- [[../changelog]] — 变更日志
