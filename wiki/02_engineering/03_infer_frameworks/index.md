---
title: "推理框架 —— 目录索引"
---

# 推理框架 —— 目录索引

> 覆盖 LLM 推理技术栈、服务引擎、投机推理与分离式 KV 架构
> **最后更新**：2026-08-30（vLLM 全域统一至 23 篇 + index）

---

## 模块定位：做什么 · 提供什么能力 · 边界在哪

**一句话**：推理框架把"训练好的权重"变成"能在延迟 SLO 下持续服务动态并发请求的在线系统"。它管的不是单次 forward 快不快，而是**在共享的显存与算力上，逐 token 地把有限资源分配给随机到达、长度未知的请求**。

**为什么必须独立成一层**：训练负载是静态可预测的——batch 固定、shape 固定、离线跑完即可；推理负载是**动态到达 + 输出长度未知 + 有尾延迟约束**。这三件事把训练栈里根本不存在的问题变成一等公民：

- KV cache 从"一个中间张量"变成**需要分配、回收、复用、抢占的长期显存资产**；
- 调度从"一个 epoch 循环"变成**每个 token step 都要重做一次的准入决策**；
- batch 从"静态维度"变成**每步都在变形的对象**，于是 CUDA Graph、attention 后端、量化 kernel 都必须按形态动态派发。

### 本域覆盖的系统与各自定位

本域不是单一产品的知识域。下表先说清**有哪些系统、各自解决什么、本库覆盖到什么程度**——覆盖度是高度不均衡的，如实标注在这里：

| 系统 | 在本域中的定位 | 本库覆盖 | 基线 |
|---|---|---|---|
| **vLLM** | 通用开源推理引擎的事实标准；本域**机制细节的主要样本**，从请求语义、EngineCore/Scheduler/paged KV 到 serving、采样、多模态、投机、量化、并行、compile 与在线更新全链路 | 23 篇 + index（**系统性源码覆盖**） | `vllm-project/vllm@6b110badbb22d3f66c7218b71138f13b7a6b3419`，23 篇已统一核验 |
| **SGLang** | 另一条工程路线；本库目前只切入其**编译 pass 体系**，与 vLLM 的 piecewise CUDA Graph 管线做对照 | 1 篇 + index（**单点切入，非全景**） | 见页头 |
| **Mooncake** | 以 KV Cache 为中心的**分离式服务架构**（论文级，非引擎实现）；prefill/decode 分离与跨实例 KV 传输的原始论证 | 1 篇（论文分析） | arXiv:2407.00079 |
| **投机推理** | **技术专题而非产品**：跨引擎的 draft/verify 演进主线（MTP → Eagle3 → DFlash → DSpark） | 2 篇 + index | DeepSpec `dd854392` 等，见页头 |
| **TensorRT-LLM / llama.cpp / TGI / Transformers** | 仅在技术栈全景页里做**场景定位对照**，本库**没有**源码级深挖 | 0 篇（仅出现在对照表） | — |

> 引用时请注意这个不均衡：能力清单里的源码锚点**绝大部分落在 vLLM**。把 vLLM 的某个实现读成"推理框架都这样"是错的——SGLang 的调度与 attention 后端组织方式并不相同，本库尚未覆盖到可以下这种断言的程度。

### 本域提供的能力

下表按**能力**组织（推理框架都必须解决的问题），"样本"列说明本库是拿谁的实现讲的。vLLM 源码锚点统一按冻结 checkout `vllm-project/vllm@6b110badbb22d3f66c7218b71138f13b7a6b3419`（2026-08-29）核对：

| 能力 | 具体提供什么 | 样本与源码锚点 | 详见 |
|---|---|---|---|
| **token 级准入与调度** | 每步重新决定：哪些请求进入本轮、各跑多少 token、占用哪些 block；chunked prefill 与 decode 的混排 | vLLM `vllm/v1/core/sched/scheduler.py` | [[11_vllm_scheduler_analysis]] |
| **请求与任务语义** | 把协议、render、generation/pooling/audio 任务压成稳定 Engine 合同，再恢复用户输出 | vLLM `vllm/renderers/`、`vllm/v1/engine/input_processor.py` | [[04_vllm_request_semantics_analysis]] |
| **分页 KV 显存管理** | 把 KV 当虚拟内存分页管理：block 分配/回收、prefix 复用、抢占与换出 | vLLM `vllm/v1/core/kv_cache_manager.py` | [[12_vllm_kv_cache_management_analysis]] |
| **引擎控制面** | 资源承诺的唯一提交者；隔离前端并发模型（同步/asyncio/ZMQ）与设备拓扑（uni/multiproc/Ray） | vLLM `vllm/v1/engine/core.py` | [[10_vllm_engine_architecture_analysis]] |
| **动态形态的执行** | 把变形的 batch 整理成设备可复用的输入；in-flight batch 与异步调度重叠 CPU/GPU | vLLM `vllm/v1/worker/` | [[15_vllm_model_runner_v2_analysis]] |
| **采样与结构化约束** | 把每请求 grammar 状态、logits 变换与 token selection 收敛成合法分布 | vLLM `vllm/v1/sample/`、`vllm/v1/structured_output/` | [[17_vllm_sampling_structured_output_analysis]] |
| **多模态执行** | 让媒体 preprocessing、processor/encoder cache、budget 与 token 位置保持一致 | vLLM `vllm/multimodal/`、`vllm/v1/worker/gpu/mm/` | [[18_vllm_multimodal_execution_analysis]] |
| **attention 后端派发** | 按 batch 形态、序列长度、量化格式在多种 attention 实现间选择 | vLLM `vllm/v1/attention/backends/` | [[14_vllm_attention_backends_analysis]] |
| **编译与图捕获** | piecewise CUDA Graph 消除 launch 开销；编译 pass 如何与动态形态共存 | vLLM `vllm/compilation/` **与 SGLang 对照** | [[23_vllm_compilation_cudagraph_analysis]] · [[sglang/index\|SGLang]] |
| **投机解码** | draft/verify 把 memory-bound 的 decode 换成 compute-bound，用接受率换吞吐 | vLLM `vllm/v1/spec_decode/` + **跨引擎专题** | [[20_vllm_speculative_decoding_analysis]] · [[speculative_decoding/index\|投机推理专题]] |
| **分离式服务** | prefill/decode 拆到不同实例、KV 跨实例传输，让两段各自按自己的瓶颈扩容 | vLLM KV connector **+ Mooncake 论文侧论证** | [[26_vllm_disaggregated_kv_serving_analysis]] · [[mooncake_analysis]] |
| **serving 控制面** | API server、launcher、DP 协调与路由；多实例的生命周期与故障传播 | vLLM `vllm/v1/engine/core_client.py` | [[16_vllm_serving_control_plane_analysis]] |
| **在线权重更新** | pause、传输 session、rank-local 更新、version 与 cache/runner 可见性 | vLLM weight-transfer、Engine utility 与 worker paths | [[29_vllm_weight_transfer_online_update_analysis]] |

### 不属于本模块的

- 模型结构本身（为什么用 GQA/MLA/MoE）→ [[01_theory/01_models/index|模型架构]]；
- kernel 层面的 tile/流水/寄存器账本 → [[02_engineering/05_gpu_kernel/index|GPU Kernel 开发]]；本域只关心"什么形态下调哪个 kernel"；
- 权重是怎么训出来的、RL rollout 怎么编排 → [[02_engineering/02_train_frameworks/index|训练框架]] 与 [[02_engineering/04_posttrain_frameworks/index|后训练框架]]。

### 与兄弟域的关系

本域是后训练域的 **rollout 引擎供应方**（verl 用 vLLM/SGLang、slime 用 SGLang）；向下依赖 [[02_engineering/01_pytorch/index|PyTorch]] 的编译栈与 CUDA Graph 能力；与 [[02_engineering/05_gpu_kernel/index|GPU Kernel]] 的分界是"谁在什么形态下调用哪个 kernel" vs "kernel 内部怎么写"。

---

## 总览入口

| 页面 | 核心问题 |
|---|---|
| [[02_engineering/03_infer_frameworks/01_llm_inference_technology_stack_analysis|大模型推理技术栈全景]] | 从模型制品、调度、KV、编译、kernel、通信到 serving 的完整分层；Transformers、vLLM、SGLang、TensorRT-LLM、llama.cpp 与 TGI 的场景定位 |

## 子框架

| 子目录 | 入口 | 页数 | 核心主题 |
|---|---|---:|---|
| **vLLM** | [[02_engineering/03_infer_frameworks/vllm/index|vLLM 推理引擎知识地图]] | 23 篇 + index | 以读者问题、设计约束、状态所有权与不变量为主线；23 篇统一固定到 `vllm-project/vllm@6b110badbb22d3f66c7218b71138f13b7a6b3419`，覆盖请求语义、Engine/资源/设备热路径、模型与专用化、规模化和生产闭环 |
| **投机推理** | [[02_engineering/03_infer_frameworks/speculative_decoding/index|投机推理专题]] | 2 + index | MTP、EAGLE3、DFlash、DSpark 的 draft/verify 机制、接受率与代价模型 |
| **SGLang** | [[02_engineering/03_infer_frameworks/sglang/index|SGLang 推理框架]] | 1 + index | SGLang 编译 Pass 与 vLLM piecewise CUDA Graph 管线对照 |

## 独立页面

| 页面 | 来源 | 核心主题 |
|---|---|---|
| [[02_engineering/03_infer_frameworks/mooncake_analysis|Mooncake 分离式推理]] | Mooncake / Kimi | P/D 分离、分布式 KV Cache、RDMA 与存储数据平面 |

## Related Pages

- [[02_engineering/02_train_frameworks/index|训练框架目录]]
- [[02_engineering/01_pytorch/index|AI 框架目录]]
- [[01_theory/05_inference/index|推理技术理论]]
- [[02_engineering/03_infer_frameworks/vllm/index|vLLM 推理引擎知识地图]]
