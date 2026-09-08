---
title: "大模型推理技术栈全景 —— 从模型文件到在线服务"
---

# 大模型推理技术栈全景 —— 从模型文件到在线服务

> **源码基线**：`vllm-project/vllm@199cb9b964822e59ab9b58d88e7be31eb419a2ae`（只读 `main` 快照，2026-09-07 UTC）。
> **主题**：从模型文件到在线服务的推理技术栈分层、框架定位与学习入口。
> **适用范围**：跨框架比较仍保留 2026-08-18 的观察结果；本次仅核验 vLLM 代码落点、依赖与导航，不更新其他框架的比较事实。
> **最近更新**：2026-09-08。

现代 LLM 推理不是“选一个框架”这么简单，而是一条跨越模型格式、调度、KV 内存、编译、内核、通信、服务与可观测性的流水线。vLLM 的价值在于把其中大多数层整合成一个可扩展的通用引擎。

---

## 一、先建立正确的分层模型

```mermaid
flowchart TB
  A["应用与协议层"] --> B["输入与采样语义层"]
  B --> C["请求调度层"]
  C --> D["KV 与显存管理层"]
  D --> E["模型执行层"]
  E --> F["图编译与算子层"]
  F --> G["GPU 内核与通信层"]
  G --> H["CUDA ROCm XPU TPU NPU Metal"]
  C --> I["分布式服务与 KV 数据平面"]
  A --> J["部署 可观测性 容错"]
```

| 层 | 核心问题 | 常见技术与组件 | vLLM 中的落点 |
|---|---|---|---|
| 模型与制品 | 权重如何表达、加载和复用 | Hugging Face config、Transformers、Safetensors、GGUF、量化权重 | 模型注册表、流式权重加载、量化插件 |
| 协议与输入 | HTTP、chat template、多模态、结构化输出如何统一 | OpenAI-compatible API、tokenizer、JSON Schema、grammar | FastAPI API server、InputProcessor、Structured Output |
| 调度 | 谁在本轮运行、运行多少 token | continuous batching、chunked prefill、priority、preemption | V1 Scheduler 的统一 token 预算 |
| KV 内存 | 长上下文如何避免碎片和重复计算 | paged KV、prefix cache、KV quantization、offload | BlockPool、KVCacheManager、KVConnector |
| 模型执行 | 一步前向如何分发到多卡 | TP、PP、DP、EP、CP、MoE dispatcher | Executor、Worker、Model Runner |
| 编译与内核 | 如何减少 Python、launch 和内存流量 | `torch.compile`、Inductor、CUDA Graph、FlashAttention、FlashInfer、Triton、CUTLASS | `VLLM_COMPILE`、CustomOp、融合 pass、attention backend |
| 数据平面 | Prefill 与 Decode、GPU 与外部缓存如何交换 KV | NIXL、Mooncake、LMCache、RDMA、P/D disaggregation | KV transfer connector |
| 运维 | 怎样扩缩容、限流、观测和定位 SLO | Prometheus、OpenTelemetry、router、Kubernetes | metrics、tracing、OpenAI server；更大规模控制面通常由外部系统补齐 |

这张表也解释了为什么“FlashAttention 很快”不等于“推理服务很快”：decode 可能受 CPU 调度和 kernel launch 限制，prefill 可能受算力和长尾限制，长上下文还可能先撞上 KV 容量；任何单点优化都只覆盖其中一段。

## 二、当前主流推理框架如何分工

| 技术栈 | 最适合的场景 | 关键优势 | 主要边界 |
|---|---|---|---|
| **Transformers `generate()`** | 算法验证、模型兼容性基线、低并发脚本 | 语义清晰、模型覆盖广、改模型最方便 | 不以高并发 continuous batching 和服务 SLO 为首要目标 |
| **vLLM** | 通用 GPU 在线服务、离线批推理、多种硬件和并行方式 | continuous batching、paged KV、OpenAI API、丰富量化/投机/分布式/编译能力 | 功能组合复杂，最优配置必须按模型、硬件与工作负载验证 |
| **SGLang** | 高性能在线服务、共享前缀明显的 agent/多轮工作负载 | RadixAttention、continuous batching、P/D 分离、结构化输出与广泛并行能力 | 同样快速演进；应按目标模型和后端实测，不宜只凭单项 benchmark 迁移 |
| **TensorRT-LLM** | NVIDIA GPU 上追求极致性能和深度定制 | NVIDIA 内核栈、KV manager、scheduler、PyExecutor、CUDA Graph、投机解码 | 硬件绑定更强，构建和部署复杂度通常更高 |
| **llama.cpp** | 本地、桌面、边缘、CPU/混合设备、GGUF 生态 | 纯 C/C++、部署轻、量化和硬件后端覆盖广 | 与大型数据中心 GPU serving 的优化目标不同 |
| **Text Generation Inference** | 维护既有 TGI 部署 | Hugging Face 生态与既有运维资产 | 官方已进入维护模式；新项目应优先评估 vLLM、SGLang 等活跃方案 |

上述定位来自各项目当前官方材料，而不是把 benchmark 排名外推成普遍结论：Transformers 的官方生成入口是 `generate()` / `GenerationConfig`；SGLang 官方列出 RadixAttention、P/D 分离、投机解码、连续批处理和多种并行；TensorRT-LLM 的官方架构把 LLM API、PyExecutor、scheduler、KV cache manager 与模型执行拆开；llama.cpp 官方强调 GGUF、量化及跨硬件后端；Hugging Face 已明确标注 TGI 进入维护模式。

### 2.1 一个实用选择树

```mermaid
flowchart TD
  A["你的主要目标"] --> B{"需要修改模型或验证算法"}
  B -->|"是"| C["Transformers 作为语义基线"]
  B -->|"否"| D{"本地或边缘优先"}
  D -->|"是"| E["llama.cpp 或设备原生栈"]
  D -->|"否"| F{"只部署 NVIDIA 且愿意深度优化"}
  F -->|"是"| G["并行评估 TensorRT LLM 与 vLLM"]
  F -->|"否"| H{"共享前缀或 agent 流量占主导"}
  H -->|"是"| I["并行评估 SGLang 与 vLLM"]
  H -->|"否"| J["优先用 vLLM 建通用基线"]
```

这棵树不是最终答案。最终选型需要固定模型、权重量化、输入/输出长度分布、并发模型、SLO 与成本口径，测出 TTFT、TPOT、吞吐、错误率和单位请求成本的 Pareto 前沿。

## 三、vLLM 在这条栈中的位置

vLLM 不是一个单纯的 PagedAttention kernel。当前源码把一条完整服务路径都纳入仓库：

1. **前端**：`LLM.generate()` 提供离线批推理，`vllm serve` 的普通 Python HTTP 路径启动 OpenAI-compatible 服务；入口分别见 `vllm/entrypoints/llm.py::LLM.generate`、`vllm/entrypoints/cli/serve.py::ServeSubcommand.cmd`，总体关系见仓库 `docs/design/arch_overview.md`。
2. **引擎控制面**：Renderer、InputProcessor、OutputProcessor 与 EngineCoreClient 衔接输入转换、流式输出和 EngineCore 通信；离线与异步入口分别装配同步 client 和异步多进程 client，不应画成同一种进程拓扑。见 `vllm/v1/engine/llm_engine.py::LLMEngine.__init__`、`vllm/v1/engine/async_llm.py::AsyncLLM.__init__`。
3. **调度与 KV**：Scheduler 用 token budget 统一描述 prefill/decode，KVCacheManager 与 BlockPool 负责分页块、前缀命中和回收；见 `vllm/v1/core/sched/scheduler.py::Scheduler.schedule`、`vllm/v1/core/kv_cache_manager.py::KVCacheManager.get_computed_blocks`、`KVCacheManager.allocate_slots` 与 `vllm/v1/core/block_pool.py::BlockPool.free_blocks`。
4. **执行**：Executor 经所选执行后端把 `SchedulerOutput` 交给 worker；GPU Worker 协调 PP 中间张量与 Model Runner 前向，采样另有调用入口，执行完成不能一概等同于最终输出已返回。见 `vllm/v1/executor/abstract.py::Executor.execute_model`、`Executor.sample_tokens` 与 `vllm/v1/worker/gpu_worker.py::Worker.execute_model`、`Worker.sample_tokens`。
5. **编译与内核**：顶层默认 `-O2`，为生产性能提供更多编译/融合默认项与 `FULL_AND_PIECEWISE` CUDA Graph；显式配置优先，具体平台和模型仍会限制可用路径，不能保证每批都执行 full graph。见 `vllm/config/vllm.py::VllmConfig.optimization_level`、`OPTIMIZATION_LEVEL_02`、`VllmConfig._apply_optimization_level_defaults` 与仓库 `docs/design/optimization_levels.md`。
6. **外部数据平面**：KV connector 工厂注册 LMCache、NIXL 与 Mooncake 实现，按配置延迟导入；对应依赖列于 `requirements/kv_connectors.txt`，并非所有安装方式都会默认装齐或启用。见 `vllm/distributed/kv_transfer/kv_connector/factory.py::KVConnectorFactory.register_connector`。

该基线的 CUDA requirements 固定 `torch==2.13.0`，公共依赖要求 `transformers>=5.10.4`、`tokenizers>=0.21.1`、`safetensors>=0.6.2`，并包含 FastAPI、PyZMQ 与 msgspec；CUDA 文件还列出 `flashinfer-python==0.6.18`、TVM FFI、TileLang、cuDNN frontend 与 CUTLASS DSL 等组件。这里描述的是仓库 `requirements/common.txt`、`requirements/cuda.txt` 的依赖声明，不是所有平台通用的安装清单：`setup.py::get_requirements` 按平台选文件，发布 wheel 时去掉非 PyPI 的 `flashinfer-cubin` pin，并按 CUDA 主版本调整 CUTLASS DSL 等依赖。

这支持一个**分析类比**：vLLM 更像“推理操作系统加集成发行版”——调度和内存策略在上层，具体算子按平台、模型、量化方式和可用依赖派发；依赖出现在清单里并不证明某个请求实际选用了它。

## 四、推理性能应该怎样拆解

| 指标 | 含义 | 常见主导因素 | 优化方向 |
|---|---|---|---|
| TTFT | 从请求到首 token | 排队、tokenize、prefill、KV 命中、长 prompt | prefix cache、chunked prefill、P/D 分离、请求路由 |
| TPOT | 相邻输出 token 的平均间隔 | decode batch、kernel launch、内存带宽、同步 | CUDA Graph、融合 kernel、投机解码、合适批量 |
| ITL 尾延迟 | 流式 token 间隔的 P95/P99 | 调度抖动、大 prefill 干扰、通信尾部 | token budget、prefill 限流、负载隔离 |
| 吞吐 | 每秒 token 或请求 | batch 利用率、GPU 利用率、并行效率 | continuous batching、DP/TP/EP、量化、异步流水 |
| 容量 | 可驻留权重和 KV | 权重精度、上下文、并发、块大小 | 权重/KV 量化、paged KV、offload、缩短 `max_model_len` |
| 正确性 | 输出是否等价且稳定 | chat template、generation config、量化、kernel 数值路径 | 固定采样语义、回归集、逐层与端到端校验 |

> [!warning] 三个常见误区
> 1. 吞吐最高的配置不一定满足交互式 P99；增大 `max_num_batched_tokens` 往往会用 TTFT 换吞吐。
> 2. TP 不是免费的容量扩展：它引入逐层 collective；模型能单卡装下时，DP 往往更适合扩请求吞吐。
> 3. 量化节省显存不等于必然加速；若目标硬件没有匹配 kernel，反量化或格式转换可能抵消收益。

## 五、建议学习路径

1. 先读 [[02_engineering/03_infer_frameworks/vllm/index|vLLM 推理引擎知识地图]]，把离线与在线拓扑分开。
2. 再读 [[vllm/06_vllm_engine_architecture_analysis|vLLM 引擎架构与请求生命周期]]，跟一次 `schedule → execute → sample → update`。
3. 用 [[vllm/01_vllm_feature_optimizations_guide|vLLM 使用指南]] 跑通离线推理与流式服务；建立 benchmark 基线和做单变量实验时转 [[vllm/04_vllm_performance_tuning_guide|性能调优指南]]，启动失败、报错或输出异常时转 [[vllm/05_vllm_debugging_troubleshooting_guide|调试与故障排查]]。
4. 按瓶颈进入 scheduler、KV、attention、quantization、speculative decoding、distributed 和 compilation 专题。
5. 比较编译实现时转 [[02_engineering/03_infer_frameworks/sglang/index|SGLang 编译 Pass]]；理解分离式推理的数据平面时转 [[mooncake_analysis|Mooncake 分离式推理]]。

## 主要外部来源

- [Hugging Face Transformers 文本生成](https://huggingface.co/docs/transformers/llm_tutorial)
- [SGLang 官方仓库](https://github.com/sgl-project/sglang)
- [TensorRT-LLM 架构总览](https://nvidia.github.io/TensorRT-LLM/developer-guide/overview.html)
- [llama.cpp 官方仓库](https://github.com/ggml-org/llama.cpp)
- [Hugging Face TGI 维护状态](https://huggingface.co/docs/inference-endpoints/engines/tgi)
- [vLLM 官方架构总览](https://docs.vllm.ai/en/latest/design/arch_overview/)

## Related Pages

- [[02_engineering/03_infer_frameworks/vllm/index|vLLM 推理引擎知识地图]] — 从本页的技术栈分层进入 vLLM 架构与机制专题；请求生命周期也可沿上文学习路径进入引擎页。
- [[vllm/01_vllm_feature_optimizations_guide|vLLM 使用指南]] — 首次安装、离线调用和在线流式服务的操作入口。
- [[vllm/04_vllm_performance_tuning_guide|vLLM 性能调优指南]] — 将本页的性能维度转成固定负载、基线测量和单变量实验。
- [[vllm/05_vllm_debugging_troubleshooting_guide|vLLM 调试与故障排查]] — 按最后完成的阶段定位安装、输入、显存、编译和进程通信故障。
- [[02_engineering/03_infer_frameworks/sglang/index|SGLang 推理框架]] — 对照现有的 SGLang 编译 Pass 与 torch.compile 适配分析，比较其编译路径与 vLLM 的接续关系。
- [[02_engineering/03_infer_frameworks/speculative_decoding/index|投机推理专题]] — 展开用提案与验证减少串行生成轮数的算法和系统代价。
- [[mooncake_analysis|Mooncake 分离式推理]] — 接续 KV 数据平面和分离式推理的系统设计。
