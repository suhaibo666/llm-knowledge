# vLLM 最新源码知识域分析设计

> **状态**：聊天蓝图已于 2026-08-29 获用户批准；Waves 1–6 已完成并于 2026-08-30 通过最终 scoped gate。
> **源码基线**：`vllm-project/vllm@6b110badbb22d3f66c7218b71138f13b7a6b3419`
> **版本标识**：冻结的 detached checkout，`v0.28.1rc0-80-g6b110badbb`，提交日期 2026-08-29
> **知识库位置**：`wiki/02_engineering/03_infer_frameworks/vllm/`
> **最终覆盖**：23 篇内容页 + `index.md`；覆盖矩阵 22/22 项 `covered`。

## 1. 决策与适用范围

本轮采用**架构优先的增量重规划**，不采用“给现有页面机械换行号”，也不把整个
vLLM 知识域推倒重写。现有页面中已经形成机制主线、状态所有权、替代方案和失败边界
的内容应保留；与最新源码冲突、越过概念所有权或只剩调用链搬运的部分重新分析。

本设计在新基线上的指导地位取代
`docs/superpowers/specs/2026-08-20-vllm-knowledge-domain-redesign.md`。旧文件保留为上一轮
`d66300a1` 重构的历史记录，不再指导 `6b110bad` 页面写作。所有本轮页面必须统一使用
本文件批准的 commit；不得在正文中混入旧基线行号。

目标读者是熟悉 Python、PyTorch 和基本 LLM 推理概念，但尚未掌握 vLLM 内部状态边界
的工程师。读者完成系列后应能：

1. 从请求能力和资源约束解释 vLLM 为什么形成当前分层；
2. 指出请求、token budget、逻辑/物理 KV、persistent row、权重版本和输出语义分别由谁拥有；
3. 沿一条真实生命周期验证状态在哪里建立、提交、转移和对外可见；
4. 判断一个改动属于哪层，依据是职责与边界，而不是额外的“去哪里加功能”目录；
5. 解释核心机制为何胜过直观替代方案，以及它支付的成本和失败条件。

## 2. 为什么需要重新规划

现有目录有 19 篇内容页和一个索引：18 篇主要固定在 `d66300a1`，请求全链路页固定在
`26858770`。最新基线相对本地 `26858770` 又前进 206 个提交，变化已经越过“只改行号”
的阈值：

- Model Runner V2 在 Triton 可用且能力检查通过时默认启用，V1 runner 变成条件回退；
- Renderer、Frontend task、Pooling、Transcription、Realtime 等公开语义继续扩展；
- KV layout、KV Connector、offload、Mooncake/NIXL 和混合状态模型持续演进；
- distributed executor、权重传输、sampling、structured output 与 multimodal runner 已形成独立状态所有者；
- 模型库、attention、编译和 kernel 目录有大量增删，旧定位符不能直接复用。

现有 `03_vllm_request_flow_walkthrough_analysis.md` 的问题也不是“731 行太长”，而是它
同时承担全局架构、服务启动、请求流、DeepSeek MLA/MoE、并行维度、排障、Python 语法
和函数索引。这些内容没有共同的状态所有者。页面需要按概念所有权重构，而不是按固定
行数拆分。

## 3. 备选方案与最终选择

| 方案 | 优点 | 主要问题 | 决策 |
|---|---|---|---|
| 只升级现有 19 页基线 | 改动最少 | 保留能力缺口、页面重叠与旧主路径假设 | 不采用 |
| 架构优先增量重规划 | 复用有效内容，同时修正边界和缺口 | 需要逐页重新核验与去重 | **采用** |
| 全部推倒重写 | 可以重新统一格式 | 浪费已有机制分析，链接风险和重复劳动最大 | 不采用 |

## 4. 系统中心命题

vLLM 不是一条“更快的 Transformer forward”调用链，而是一套把不确定请求转换为可重复
设备 step 的资源系统：接口层吸收协议和输入差异，Engine 层隔离并发与进程生命周期，
Scheduler/KV 层提交资源承诺，Executor/Worker 把承诺投射到设备拓扑，Model Runner 把
动态请求压成稳定 GPU 状态，模型与算子层再根据能力合同选择实现。跨实例 KV、在线权重
更新、插件和可观测性沿这些边界传递状态，但不改变每层的权威所有者。

这一命题是目录的主线。目录树、类继承和函数调用只作为验证它的证据，不能反过来充当
文档结构。

## 5. 能力图

| 能力组 | 用户可见能力 | 系统必须解决的问题 | 权威页面 |
|---|---|---|---|
| 接口与任务 | Offline `LLM`、OpenAI/Anthropic/Cohere、Render、Generate、Pooling、Transcription、Realtime | 不同协议和任务怎样形成稳定 Engine 请求与用户输出 | `01`、`04`、`16` |
| 在线资源控制 | continuous batching、chunked prefill、抢占、优先级、structured readiness | 每个 step 给哪些请求多少 token，何时能安全提交 | `10`、`11`、`17` |
| KV 与上下文 | paged KV、prefix cache、hybrid layout、offload、跨实例传输 | 不可预测序列怎样获得、复用、转移和释放状态 | `12`、`26` |
| 设备执行 | MRV2、persistent batch、async scheduling、sampling、CUDA Graph | 动态请求怎样变成地址稳定、可重叠的 GPU 工作 | `15`、`17`、`23` |
| 模型与模态 | model registry、weight ABI、LoRA、multimodal encoder、attention backend | 多模型、多模态和多硬件怎样服从统一执行合同 | `13`、`14`、`18` |
| 性能专用化 | speculative decoding、quantization、IR、fusion、specialized kernels | 何时通过额外状态或硬件特化换吞吐/延迟 | `20`、`21`、`24`、`25` |
| 扩展与规模化 | TP/PP/DP/EP/CP、DBO、plugins、disaggregated serving | 单机语义怎样扩展到多进程、多设备和外部组件 | `22`、`26`、`28` |
| 生产闭环 | metrics、traces、fault containment、sleep/pause、weight update | 如何观察承诺、隔离故障并安全改变在线状态 | `27`、`29` |

`wiki/02_engineering/07_training_reliability/20_batch_invariance_guide.md` 继续拥有 batch
invariance 与确定性问题；vLLM 目录只在相关页面保留实现接缝和链接。训练框架如何组织
rollout、optimizer 和版本推进仍由 post-training 知识域拥有；`29` 只拥有 vLLM 侧在线
权重更新协议。

## 6. 静态架构：责任、合同与状态所有权

### 6.1 接口与语义层

- **存在原因**：HTTP 协议、chat template、多模态媒体、离线对象和流式输出不应进入 GPU 调度循环。
- **提供能力**：请求验证、render/tokenize、Engine 输入构造、detokenize、stop 与协议输出。
- **边界合同**：协议/Prompt/媒体 → `EngineCoreRequest`；Engine token 输出 → 用户可见响应。
- **拥有状态**：协议语义、renderer/tokenizer、前端 `RequestState`、per-request output collector。
- **不拥有**：waiting/running、token budget、逻辑 KV 和 GPU batch。
- **证据入口**：`vllm/entrypoints/`、`vllm/renderers/`、`vllm/tasks.py`、
  `vllm/v1/engine/input_processor.py`、`vllm/v1/engine/output_processor.py`。

### 6.2 Engine 生命周期层

- **存在原因**：同步离线接口、async 在线接口、in-process/ZMQ/Ray 和多 API server 需要复用同一推理状态机。
- **提供能力**：engine 构造、进程启动、请求关联、传输、liveness、abort、shutdown 和背压。
- **边界合同**：前端 request/utility message ↔ EngineCore outputs/events。
- **拥有状态**：client transport、进程/任务生命周期、前端与 Engine 请求的关联。
- **不拥有**：admission policy、物理模型执行和协议渲染细节。
- **证据入口**：`vllm/v1/engine/llm_engine.py`、`async_llm.py`、`core_client.py`、
  `vllm/entrypoints/launchers/`。

### 6.3 资源控制层

- **存在原因**：动态请求、KV 容量、structured readiness 和公平性必须由唯一提交者协调。
- **提供能力**：request lifecycle、token admission、逻辑 KV 分配、抢占、完成与结果提交。
- **边界合同**：Engine 请求 + 上一步结果 → `SchedulerOutput`；runner 结果 → `EngineCoreOutputs`。
- **拥有状态**：waiting/running、token progress、逻辑 block table、grammar readiness、完成/取消状态。
- **不拥有**：GPU tensor 地址、HTTP stream 和模型权重。
- **证据入口**：`vllm/v1/engine/core.py`、`vllm/v1/core/sched/`、
  `vllm/v1/core/kv_cache_manager.py`、`block_pool.py`、`vllm/v1/structured_output/`。

### 6.4 分布式执行层

- **存在原因**：资源计划必须映射到 uni/multiprocess/Ray/external launcher 和多并行轴，而不能复制 Engine 语义。
- **提供能力**：worker 生命周期、RPC fan-out、collective、rank group、输出聚合和故障传播。
- **边界合同**：`SchedulerOutput` → 每 rank 设备执行；rank outputs → 聚合的 runner output/future。
- **拥有状态**：worker 拓扑、rank/group、通信顺序、模型权重和物理 KV 所在设备。
- **不拥有**：全局公平性、协议输出和模型 ABI 解释。
- **证据入口**：`vllm/v1/executor/`、`vllm/v1/worker/`、`vllm/distributed/`。

### 6.5 设备运行时层

- **存在原因**：逐 token 重建 Python/Tensor 状态和同步 CPU/GPU 会使控制开销进入关键路径。
- **提供能力**：persistent row、staged write、输入 gather、MM encoder、sampling、graph capture/replay。
- **边界合同**：逻辑 scheduler delta → 稳定设备 buffer；model output → token/pooling/auxiliary output。
- **拥有状态**：active request row、device-resident dynamic state、graph descriptor、当步 sampler/MM 状态。
- **不拥有**：全局 admission、协议 detokenization 和跨实例资源策略。
- **证据入口**：`vllm/v1/worker/gpu/`、`gpu_worker.py`、旧 runner
  `vllm/v1/worker/gpu_model_runner.py`。

### 6.6 模型与算子层

- **存在原因**：模型结构、权重格式、量化、attention layout 和硬件 kernel 组合远多于 Engine 可硬编码的范围。
- **提供能力**：模型检查/构造/加载、并行层、backend/provider 选择、compile/IR/fusion/kernel。
- **边界合同**：`VllmConfig` + checkpoint + device capability → 可执行 model/layer/backend。
- **拥有状态**：模型 ABI、参数/scale 语义、backend 能力、IR operation 和 kernel selection。
- **不拥有**：request lifecycle、token fairness 和服务进程拓扑。
- **证据入口**：`vllm/model_executor/`、`vllm/v1/attention/`、`vllm/compilation/`、
  `vllm/ir/`、`vllm/kernels/`。

### 6.7 横切数据面

KV Connector/offload、在线权重传输、插件、metrics 和 fault tolerance 跨过多层。它们必须
分别说明每一侧的 state owner 和提交协议，不能因为目录位于 `distributed/` 或 `v1/` 就被
误写成某一层的内部细节。

## 7. 动态生命周期

### 7.1 启动生命周期

`EngineArgs/config → launcher/client → EngineCore → Executor/Workers → model load → memory profile
→ KV initialization → compile/warmup → ready barrier`。

页面必须标出配置何时变成不可随意修改的运行时合同、物理容量何时反馈给 Scheduler，以及
API/EngineCore/Worker 的 ready 各自证明什么。启动进程数不能写成固定公式而忽略 executor
backend、DP coordinator 和外部 launcher。

### 7.2 在线请求生命周期

`协议请求 → Renderer/InputProcessor → 前端 RequestState + EngineCoreRequest →
EngineCoreClient → Scheduler/KV admission → Executor/Worker/MRV2 → model/attention/sampler →
Scheduler commit → OutputProcessor → stream`。

静态层图先解释谁拥有状态；生命周期图只解释状态如何跨边界。详细页不得重新复制每个
子系统的内部机制。

### 7.3 在线权重更新生命周期

`pause/drain/abort policy → initialize transfer → start transaction → transfer/update shards →
finish with weight version → cache/runner post-commit handling → resume`。

这一流程必须区分“字节已传完”“各 rank 已完成转换”“新版本已对请求可见”三个条件，
并核验失败、重复消息、稀疏更新和 cache 安全边界。

## 8. Live / legacy 边界

- `vllm.engine.LLMEngine` 与 `AsyncLLMEngine` 当前分别是 V1 `LLMEngine` / `AsyncLLM` 的别名；
  不再把独立 V0 engine 写成现行架构。
- **V1 engine** 与 **V1/V2 model runner** 是两个维度。Engine 仍称 V1；MRV2 在 Triton
  可用且 capability check 通过时默认选择，V1 runner 是条件回退。
- raw prompt 直接进入 `InputProcessor` 的部分路径已有 deprecation；新的 Renderer 边界要
  按当前代码说明，不能沿用旧 frontend 图。
- 官方 `docs/design/arch_overview.md` 同时包含当前 V1 进程说明和较旧的类描述。它用于
  设计意图证据；实际 live path 以本 commit 的 aliases、constructors 和 tests 为准。

## 9. 页面合同

以下所有页面都继承统一基线 `6b110badbb22d3f66c7218b71138f13b7a6b3419`。表中“必须
回答”同时是页面 completion test；如果读者在删除代码块后无法回答，该页面未完成。

### 9.1 入口与系统心智模型

| Path / 类型 / 动作 | 中心命题与必须回答 | 拥有 / 明确排除 | 主要证据入口、依赖与图 |
|---|---|---|---|
| `index.md` / Knowledge map / 最后重写 | 用问题、能力和阅读依赖导航整个域；各概念唯一 owner 是谁？ | 拥有导航和覆盖状态；不复述机制正文 | 依赖所有页面完成；静态能力到 owner 的小图可选 |
| `01_vllm_feature_optimizations_guide.md` / Guide / 保留重校 | 怎样跑通、测量并按瓶颈选择配置，而不是堆 flags？ | 拥有使用/benchmark/调优闭环；排除内部状态机 | `vllm/entrypoints/llm.py`、CLI、`vllm/benchmarks/`；配置决策表，不画内部调用图 |
| `02_vllm_system_design_principles_analysis.md` / Principles overview / 保留更新 | 动态请求为何要求连续调度、分页状态、异步执行和能力合同？ | 拥有全局约束和设计支点；排除完整分层与调用链 | config/core/metrics 与同 commit design docs；瓶颈→设计支点因果图 |
| `03_vllm_architecture_overview_analysis.md` / Architecture overview / 由旧 `03` 重命名重写 | 系统由哪些责任层组成、各层为何存在、谁拥有状态、一条请求如何穿过它们？ | 拥有静态分层与代表生命周期；排除 Scheduler/KV/MRV2 等内部证明 | entrypoints、engine/core、scheduler、worker/model runner、model executor；**静态层图 + 独立生命周期图** |
| `04_vllm_request_semantics_analysis.md` / Mechanism / 新增 | Generate、Pooling、Render、Transcription、Realtime 和不同协议如何归一为 Engine 输入并还原为用户输出？ | 拥有任务/协议/render/input/output 语义；排除 admission 和 GPU sampling | `vllm/tasks.py`、`entrypoints/`、`renderers/`、input/output processor；请求语义转换图 |

旧 `03` 中服务启动和进程拓扑迁入 `10/16`；DeepSeek MLA/MoE 只在 `13/14/24` 保留权威
解释；并行维度归 `22`；Python 语法表和机械函数索引删除。旧交互资产只有在新架构图
完成、链接全部迁移且确认无其他引用后才能移除或替换。

### 9.2 核心 Engine 与设备主线

| Path / 类型 / 动作 | 中心命题与必须回答 | 拥有 / 明确排除 | 主要证据入口、依赖与图 |
|---|---|---|---|
| `10_vllm_engine_architecture_analysis.md` / Deep dive / 重点更新 | Client、EngineCore 与 Executor 为什么分离，资源承诺在哪里提交？ | 拥有 Engine 内部对象/进程接缝；排除整个 codebase 分层和 HTTP 语义 | `llm_engine.py`、`async_llm.py`、`core_client.py`、`core.py`、executor；对象/进程所有权图 |
| `11_vllm_scheduler_analysis.md` / Deep dive / 重校 | 一次 schedule 怎样完成多资源 admission 事务并在 output 后提交？ | 拥有 waiting/running、token/encoder/spec budget、preempt/finish；排除物理 runner | `vllm/v1/core/sched/`、request、scheduler tests；请求状态机与一步事务图 |
| `12_vllm_kv_cache_management_analysis.md` / Deep dive / 重点更新 | 逻辑/物理 KV block、prefix cache、hybrid layout 和本地 offload 如何保持所有权正确？ | 拥有单 Engine KV 生命周期；排除跨 Engine 传输 | kv cache manager、block pool、kv cache utils、local offload；块状态/引用图 |
| `13_vllm_model_library_analysis.md` / Deep dive / 更新 | Registry、统一构造 ABI、权重映射、并行层和 LoRA 接合如何形成可执行模型？ | 拥有模型/权重 ABI；排除逐 step state、量化内部 kernel | model registry、model loader、linear layers、LoRA；模型构造与权重提交图可选 |
| `14_vllm_attention_backends_analysis.md` / Deep dive / 重点更新 | metadata、KV layout 和 capability negotiation 如何让动态调度选择专用 attention？ | 拥有 attention contract/backend selection；排除全局调度和 kernel 实现细节 | `vllm/v1/attention/`、attention layers、selector/tests；contract 边界图 |
| `15_vllm_model_runner_v2_analysis.md` / Deep dive / 重点重写 | 默认 MRV2 如何用 persistent row、staged write 与 async-first execution 重建设备热路径，何时回退 V1？ | 拥有 device request state、buffer/graph lifecycle；排除全局 admission | `vllm/v1/worker/gpu/`、`gpu_worker.py`、旧 runner、MRV2 design/tests；CPU/GPU overlap 时序图 |
| `16_vllm_serving_control_plane_analysis.md` / Deep dive / 重点重写 | launcher、API server、DP coordinator、Core clients 如何管理启动、路由、背压和故障域？ | 拥有 serving 拓扑/生命周期；排除请求渲染细节 | launchers、CLI/serve、coordinator、core client、fault tolerance；启动/ready 图 |
| `17_vllm_sampling_structured_output_analysis.md` / Deep dive / 新增 | logits 怎样经过 processor、penalty、top-k/top-p、grammar mask 和 sampler 仍保持分布正确？ | 拥有 token selection 与 structured constraint；排除 detokenization/协议输出 | `vllm/v1/sample/`、`vllm/v1/structured_output/`、config、scheduler/runner 接缝；采样流水线/grammar 状态图 |
| `18_vllm_multimodal_execution_analysis.md` / Deep dive / 新增 | 媒体怎样经 parse、processor cache、encoder budget/cache 和 device runner 变成模型输入？ | 拥有 modality preprocessing/feature/encoder state；排除具体 VLM 网络结构 | `vllm/multimodal/`、`vllm/v1/worker/gpu/mm/`、model interfaces、scheduler 接缝；媒体→feature→encoder→decoder 图 |

### 9.3 专项优化、规模化与生产能力

| Path / 类型 / 动作 | 中心命题与必须回答 | 拥有 / 明确排除 | 主要证据入口、依赖与图 |
|---|---|---|---|
| `20_vllm_speculative_decoding_analysis.md` / Deep dive / 更新 | 何时用 draft 成本换 target 串行步，怎样保持采样分布与 KV 正确？ | 拥有 propose/verify/accept 合同；排除一般 sampler 机制 | spec decode、scheduler、MRV2 sample/graph、tests；draft/verify 状态图 |
| `21_vllm_quantization_analysis.md` / Deep dive / 更新 | 格式、scale、load transform、hardware kernel 和 fallback 为什么必须联合决策？ | 拥有量化 ABI/派发；排除通用权重加载和 kernel 编程细节 | quant config/layers/kernels/model loader；三阶段加载图可选 |
| `22_vllm_distributed_inference_analysis.md` / Deep dive / 重点更新 | TP/PP/DP/EP/CP、DBO 和 executor 怎样映射到 rank state 与 collective 顺序？ | 拥有并行/通信语义；排除在线权重更新事务 | `vllm/distributed/`、executor/coordinator、parallel state/tests；rank tensor + DBO timeline |
| `23_vllm_compilation_cudagraph_analysis.md` / Deep dive / 重点更新 | 动态 shape 系统如何得到可编译、可捕获且地址稳定的执行区域？ | 拥有 compile/graph lifecycle；排除 IR op 语义和具体 kernel | compilation、MRV2 cudagraph utils、design/tests；eager/compile/piecewise/full 分发图 |
| `24_vllm_fused_ops_and_kernels_analysis.md` / Deep dive / 更新 | 何时融合和专用 kernel 真正减少 launch/访存，何时必须 fallback？ | 拥有 provider/kernel family/收益模型；排除 IR pass 时序 | kernels、fused MoE、custom op/IR providers、tests；kernel selection 图可选 |
| `25_vllm_ir_and_fusion_passes_analysis.md` / Deep dive / 更新 | IR 怎样保留稳定语义、表达 donation/alias 并按安全顺序改写和 lowering？ | 拥有 IR/pass/functionalization；排除模型图整体编译策略 | `vllm/ir/`、compilation passes、design/tests；pass pipeline 图 |
| `26_vllm_disaggregated_kv_serving_analysis.md` / Deep dive / 重点更新 | transferable groups、connector、lease 和 store 如何跨 Engine 转移状态并在故障时回收？ | 拥有跨实例 KV protocol；排除单 Engine block 生命周期 | KV transfer connectors、offload、Mooncake/NIXL/MoRIIO、tests；producer/consumer/lease 图 |
| `27_vllm_observability_reliability_analysis.md` / Deep dive / 更新 | metrics、events、traces 和 sentinels 怎样把 SLO 追溯到资源承诺与故障域？ | 拥有观测/故障反馈；排除服务路由机制 | metrics、fault tolerance、tracing/instrumentator、design/tests；反馈环图可选 |
| `28_vllm_extension_plugin_system_analysis.md` / Deep dive / 更新 | discovery、进程覆盖和多阶段初始化如何扩展核心而不隐式污染状态？ | 拥有 plugin ABI/lifecycle；排除模型内建 registry 与协议实现 | plugins、endpoint/platform/io/lora resolver docs/tests；plugin lifecycle 图 |
| `29_vllm_weight_transfer_online_update_analysis.md` / Deep dive / 新增 | pause/sleep、start/update/finish、weight version 和 cache/runner post-commit 怎样构成在线更新事务？ | 拥有 vLLM 侧权重传输与可见性；排除 trainer 算法/rollout 编排 | distributed weight transfer、LLM/AsyncLLM/Core utility methods、tests；版本提交状态机 |

## 10. 覆盖矩阵

在新基线下，旧页面不能因为文件存在就标为 `covered`；只有完成定位符重核、边界去重和
页面 completion test 后才能从 `planned` 改为 `covered`。

| 机制/生命周期 | 权威 owner | 其他页面只允许 | 最终状态 |
|---|---|---|---|
| 全系统静态分层 + 代表请求生命周期 | `03` | 一段局部定位 + 链接 | covered (Wave 1) |
| 性能/资源约束模型 | `02` | 当前机制的局部约束 | covered (Wave 2) |
| 协议、任务、render/input/output 语义 | `04` | serving/runner 的边界字段 | covered (Wave 2) |
| Engine client/core/executor 接缝 | `10` | 直接相邻接口 | covered (Wave 2) |
| token admission 与 request state | `11` | `SchedulerOutput` 消费合同 | covered (Wave 3) |
| 单 Engine KV block/prefix/offload | `12` | attention/connector 的消费合同 | covered (Wave 3) |
| 模型、权重、并行层、LoRA ABI | `13` | quant/runner 的接合点 | covered (Wave 4) |
| attention metadata/backend contract | `14` | runner/compile 的能力协商 | covered (Wave 4) |
| MRV2 device state 与执行 | `15` | scheduler delta/attention metadata 接口 | covered (Wave 3) |
| serving 启动、路由、背压和故障域 | `16` | 协议 handler 的一跳接口 | covered (Wave 2) |
| sampling 与 structured constraints | `17` | spec decode 的 verify 接缝 | covered (Wave 3) |
| multimodal preprocessing/encoder state | `18` | model/scheduler 的边界摘要 | covered (Wave 3) |
| speculative propose/verify | `20` | sampler/KV 的一跳合同 | covered (Wave 4) |
| quantization | `21` | model/kernel 接合点 | covered (Wave 4) |
| parallel/rank/collective | `22` | engine/executor 边界 | covered (Wave 5) |
| compile/CUDA Graph lifecycle | `23` | runner/backend capture contract | covered (Wave 4) |
| fused op/kernel provider | `24` | IR/model 的调用接缝 | covered (Wave 4) |
| IR/pass/lowering | `25` | compile page的阶段摘要 | covered (Wave 4) |
| 跨 Engine KV transfer/lease | `26` | 本地 KV/serving 边界 | covered (Wave 5) |
| metrics/traces/fault feedback | `27` | 调优页的指标链接 | covered (Wave 5) |
| plugin ABI/lifecycle | `28` | 对应 subsystem 的 extension boundary | covered (Wave 5) |
| 在线权重更新/版本可见性 | `29` | post-training 页的 vLLM 接缝 | covered (Wave 5) |

## 11. 尚未关闭的证据问题

以下是实施时必须从冻结源码和 tests 关闭的证据问题，不是允许自由发挥的 TODO：

1. MRV2 的默认路径、ROCm/Triton/feature fallback 与各 task 的准确能力矩阵；
2. Renderer 重构后 raw prompt、pre-rendered `EngineInput` 和远端 render endpoint 的 live/legacy 边界；
3. Pooling、Transcription、Realtime 是否共享同一 request/output contract，哪些机制只能做摘要而不能合并；
4. multimodal processor cache、encoder cache、GPU IPC lease 与 scheduler budget 的状态所有权是否需要进一步拆页；
5. online weight update 在 finish/version commit 后对 prefix/KV、quant post-process、spec draft weight 和 in-flight request 的准确处理；
6. transferable KV cache groups、hybrid DCP/Mamba state 和 connector 完成判定是否改变 `12/26` 的当前边界；
7. 官方 architecture/design docs 与 live aliases/constructors 冲突的具体位置和应显式纠正的旧说法。

若证据显示其中某项拥有独立 state owner、无法在批准页面内保持单一中心命题，必须返回本
设计做 material-drift 复核，不能由页面作者自行新增或挪动 owner。

## 12. 行文与证据合同

每个机制单元按以下因果顺序展开：

1. 背景：什么负载、瓶颈、正确性风险或兼容需求迫使它存在；
2. 为什么这样设计：当前路线及其胜过的直观替代，作者未自陈时明确标为分析推断；
3. 实现思路与细节：先讲因果机制和状态转换，再给最小调用链与证据；
4. 约束：不变量、代价、unsupported/fallback/assert/test 所揭示的边界；
5. 发展趋势：仅在 TODO、deprecation、RFC 或更新 commit 有锚点时保留，并标明推断。

源码引用是证据，不是叙事骨架：

- 每个非平凡事实都在 `6b110bad` 打开目标位置后给出准确 `file:line`；
- 官方 design docs 说明设计意图，code/tests 决定当前行为；发生冲突时并列写清；
- 不按函数体逐句搬运，不用签名列表组织章节；
- 引用代码仅限精确语法、顺序或 guard 本身承重的场景；
- **删除测试**：移除代码块后，正文仍能解释为何存在、状态怎样移动、什么会失效；
- 不设置固定页面数、行数、层数、代码块数或代码/解释比例；是否拆分只看概念所有权、
  中心命题和读者负担。

架构概览必须先给静态责任/状态所有权，再给动态生命周期。分层清楚后，读者自然能够
判断改动属于哪层；除非用户另行请求开发者扩展指南，不新增“去哪加功能”专章。

## 13. 图形合同

图只表达文字难以线性说明的关系。候选图已经写进页面合同；其中以下图为必需：

- `03`：静态责任层图、独立的在线请求生命周期图；
- `10`：Engine client/core/executor 对象与进程所有权图；
- `11`：请求状态机与一次 admission 事务；
- `12`：逻辑/物理 block、hash、refcount、free/evict 状态图；
- `15`：CPU step N+1 与 GPU step N overlap；
- `16`：launcher/API/Core/Worker ready 与故障传播；
- `17`：logits → constraint → sample 的数据/状态流；
- `18`：media → processor/cache → encoder → model input；
- `22`：rank/group 映射与 DBO 时间线；
- `26`：KV producer/consumer/lease/failure；
- `29`：在线权重版本事务。

同一关系只在 owner 页维护完整图，其他页面引用结论并链接。图形介质与视觉规范由
`drawing-wiki-figures` 路由；选择 Mermaid 时再调用 `writing-mermaid-diagrams`。图中源码
节点必须同样固定到本 commit，不能把旧交互图的 URL 或默认路径原样保留。

## 14. 实施顺序

顺序按概念依赖，而不是文件名：

1. **Wave 0 — 冻结与清单**：在独立只读 source worktree 固定 `6b110bad`；记录目录资产、
   backlinks、旧图依赖和 coverage matrix，不移动用户当前脏 checkout。
2. **Wave 1 — 标杆页**：完成 `03`。它校准静态分层、动态生命周期、图形和最小引用方式；
   用户验收后再扩展波次。
3. **Wave 2 — 入口与控制边界**：`02`、`04`、`10`、`16`。
4. **Wave 3 — 资源与设备热路径**：`11`、`12`、`15`、`17`、`18`。
5. **Wave 4 — 模型与专用化**：`13`、`14`、`20`、`21`、`23`、`24`、`25`。
6. **Wave 5 — 规模化和生产闭环**：`22`、`26`、`27`、`28`、`29`。
7. **Wave 6 — 使用与整合**：收缩/更新 `01`，最后更新 vLLM `index.md`、父索引、总索引
   必要计数、双向链接和 changelog。

每一波结束后更新 coverage matrix；只有 material drift 才重新请求批准。普通措辞、locator
移动和链接修正不触发重规划。

## 15. Material-drift 门

出现以下任一情况必须停止相关页面并回到用户：

- 需要改变统一源码基线；
- 发现某个权威概念必须移动 owner、合并或拆出新页面；
- `03` 的静态架构层无法解释 live path，系统中心命题需要修改；
- multimodal、non-generation task、weight transfer 或 KV connector 的证据推翻当前页面边界；
- 需要新建/移动 vLLM 目录或扩展到 vLLM-Ascend、production stack、训练框架实现；
- 用户受众或交付物从知识域分析扩展为 API 手册、部署手册或开发者扩展指南。

## 16. 变更安全

- 当前根知识库工作区存在用户未提交的 Megatron/Wiki 修改；vLLM 源码 checkout 也有未跟踪
  文件。实施必须使用隔离 worktree，不 reset、stash、覆盖或顺带提交这些内容。
- vLLM source worktree 固定到批准 commit 后只读；不得 fetch 后自动移动基线。
- `03` 重命名前先清点全库 backlinks；旧交互资产只有在新图可用、无引用且差异可审查时处理。
- index/changelog 使用精确路径部分暂存，不把其他会话的 Wiki 修改带入提交。

## 17. 完成与验证

完成不是“写满若干页”，而是 coverage matrix 中每项从 `planned/gap` 变为 `covered`，并满足：

1. 每页中心命题、reader question、owned/excluded concepts 与本设计一致；
2. 每个核心能力、代表生命周期、state owner 和机制只有一个权威页面；
3. 所有非平凡事实都使用新基线已打开核验的精确 locator；
4. 每页至少抽查 3 个承重 locator，重写页的关键调用链和失败边界做机械存在性检查；
5. 删除代码块后仍能复述设计逻辑、因果机制、不变量和失败条件；
6. 事实、官方设计意图和分析推断明确分开；
7. 新图完成渲染和人工目检，旧资产无悬空引用；
8. 验证范围限于本次重构的 `wiki/02_engineering/03_infer_frameworks/vllm/`、被它直接修改的
   父级/总索引与 changelog，以及这些文件引用的新增或改名资产：
   - 对上述页面做定向 wikilink 解析，验证目标存在、链接无歧义、旧 `03` slug 无残留；
   - 只在公式发生变化时，对变更文件运行路径限定的数学检查；
   - 只对新增或修改的图运行 Mermaid/图形渲染与人工目检；
   - 只对 vLLM 页面中的承重源码引用做基线存在性和抽样语义核验；
   - 对本任务文件运行定向 `git diff --check`。
9. 不运行全仓 `python -m pytest tools/`、全站 `npm run docs:test` 或与 vLLM 重构无关的
   仓库诊断；只有定向检查暴露共享工具回归，或用户明确扩大验证范围时，才升级验证。
10. vLLM index、父索引、总索引计数和 changelog 与最终页面集合一致。

本轮不得用页面数、行数、代码块数或引用数量代替上述完成证据。

最终 scoped gate 的结果是：23 篇内容页及 1 个域索引统一固定到上述 detached commit，
覆盖矩阵 22/22 项已关闭；Wave 6 完成使用指南、知识地图、父级/总索引、radar、回链和
changelog 的整合，因此不额外虚构一个机制 owner 行。
