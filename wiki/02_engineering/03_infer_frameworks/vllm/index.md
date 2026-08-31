---
title: "vLLM 推理引擎：按问题与依赖组织的知识地图"
---

# vLLM 推理引擎：按问题与依赖组织的知识地图

> **统一源码基线**：24 篇内容页全部按 `vllm-project/vllm@6b110badbb22d3f66c7218b71138f13b7a6b3419`（冻结的 detached checkout，提交时间 2026-08-29T02:40:53Z）核验。
> **覆盖范围**：24 篇内容页 + 本索引（`01`–`04`、`10`–`29`）。
> **导航原则**：先按读者问题找到唯一机制 owner，再沿“前置 → owner → 后续”阅读；本页只维护概念边界、依赖和覆盖状态，不复述 owner 页的机制证明。

## 一、先选择入口

| 你现在要回答什么 | 从这里开始 | 接下来 |
|---|---|---|
| 先跑通、测量并按瓶颈调优 | [[02_engineering/03_infer_frameworks/vllm/01_vllm_feature_optimizations_guide|使用与优化指南]] | 按诊断信号跳到下表中的机制 owner |
| 理解“为什么不是一次更快的 forward” | [[02_engineering/03_infer_frameworks/vllm/02_vllm_system_design_principles_analysis|系统设计原则]] | [[02_engineering/03_infer_frameworks/vllm/03_vllm_architecture_overview_analysis|架构概览]] |
| 建立完整责任层、状态 owner 与请求生命周期 | [[02_engineering/03_infer_frameworks/vllm/03_vllm_architecture_overview_analysis|架构概览]] | `04 → 10 → 11 → 12 → 15 → 16 → 18` |
| 查协议、任务、render 与用户输出语义 | [[02_engineering/03_infer_frameworks/vllm/04_vllm_request_semantics_analysis|请求语义]] | 生成选择看 `18`；媒体执行看 `19` |

## 二、概念 owner 与阅读依赖

### 2.1 入口、架构与控制边界

| Owner | 读者问题 | 本页拥有 | 前置 → 后续 |
|---|---|---|---|
| [[02_engineering/03_infer_frameworks/vllm/01_vllm_feature_optimizations_guide|01 使用与优化]] | 怎样建立基线并用可撤销实验找限制资源？ | 使用、benchmark、诊断与验收闭环 | 无 → 由症状选择机制页 |
| [[02_engineering/03_infer_frameworks/vllm/02_vllm_system_design_principles_analysis|02 系统设计原则]] | 动态请求为何迫使系统采用连续调度、分页状态、异步执行与能力合同？ | 全局约束与设计支点 | `01` 可选 → `03` |
| [[02_engineering/03_infer_frameworks/vllm/03_vllm_architecture_overview_analysis|03 架构概览]] | 系统由哪些责任层组成，状态怎样跨层提交和可见？ | 静态分层、层间合同、代表请求生命周期 | `02` → `04/10/11/12/15/16/22` |
| [[02_engineering/03_infer_frameworks/vllm/04_vllm_request_semantics_analysis|04 请求语义]] | 不同协议与任务在哪里合流，又在哪里恢复成用户输出？ | 协议、task、render/input/output 语义 | `03` → `10/17/18/19` |
| [[02_engineering/03_infer_frameworks/vllm/10_vllm_engine_architecture_analysis|10 Engine 所有权]] | Client、EngineCore、Executor 为何分离，资源承诺在哪里提交？ | Engine 对象/进程接缝与 request state | `03/04` → `11/17/22` |
| [[02_engineering/03_infer_frameworks/vllm/17_vllm_serving_control_plane_analysis|17 Serving 控制面]] | 服务怎样启动、ready、路由、背压、传播故障并关闭？ | launcher/API/Core/DP 拓扑与生命周期 | `03/10` → `22/27` |

### 2.2 请求与资源热路径

| Owner | 读者问题 | 本页拥有 | 前置 → 后续 |
|---|---|---|---|
| [[02_engineering/03_infer_frameworks/vllm/11_vllm_scheduler_analysis|11 Scheduler]] | 一个 step 怎样完成多资源 admission 并在 output 后提交？ | waiting/running、token/encoder/spec budget、抢占与 finish | `10` → `12/15/16/18/19/20/26` |
| [[02_engineering/03_infer_frameworks/vllm/12_vllm_kv_cache_management_analysis|12 KV Cache]] | 单 Engine 内逻辑/物理 block、prefix、hybrid layout 与 offload 怎样保持所有权正确？ | 本地 KV 生命周期 | `11` → `14/15/16/20/26` |
| [[02_engineering/03_infer_frameworks/vllm/15_vllm_model_runner_v1_analysis|15 Model Runner V1]] | 兼容路径怎样用紧凑 persistent batch 承接动态计划？ | `CachedRequestState`、compact row、condense/reorder、dummy 与 async barrier | `11/12/14` → `16/18/19/20/23` |
| [[02_engineering/03_infer_frameworks/vllm/16_vllm_model_runner_v2_analysis|16 Model Runner V2]] | 默认路径怎样把动态计划变成稳定、可重叠的设备状态？ | stable row、staged write、per-step gather 与 device buffer/graph 生命周期 | `11/12/14/15` → `18/19/20/23` |
| [[02_engineering/03_infer_frameworks/vllm/18_vllm_sampling_structured_output_analysis|18 采样与结构化输出]] | logits 怎样经约束与采样仍保持合法分布和 grammar 状态？ | 普通 token selection 与 structured constraint | `04/11/15/16` → `20` |
| [[02_engineering/03_infer_frameworks/vllm/19_vllm_multimodal_execution_analysis|19 多模态执行]] | 媒体怎样经过双层缓存、encoder budget 与位置合同进入模型？ | preprocessing、feature、encoder state 与对齐 | `04/11/13/15/16` → `29` 的 cache 审计 |

### 2.3 模型、设备优化与专用化

| Owner | 读者问题 | 本页拥有 | 前置 → 后续 |
|---|---|---|---|
| [[02_engineering/03_infer_frameworks/vllm/13_vllm_model_library_analysis|13 模型与权重 ABI]] | checkpoint 怎样变成各 rank 可执行模型并接合 LoRA？ | registry、构造、权重映射与并行层 ABI | `03` → `14/21/22/29` |
| [[02_engineering/03_infer_frameworks/vllm/14_vllm_attention_backends_analysis|14 Attention Backend]] | metadata、KV layout 与能力协商怎样选择专用 attention？ | attention 合同、backend selection 与 layout | `12/13` → `15/16/21/23/24` |
| [[02_engineering/03_infer_frameworks/vllm/20_vllm_speculative_decoding_analysis|20 投机解码]] | 何时 draft/verify 值得，怎样保持分布和 KV 正确？ | propose/verify/accept/rollback 合同 | `12/15/16/18` → `23/27` |
| [[02_engineering/03_infer_frameworks/vllm/21_vllm_quantization_analysis|21 量化 ABI]] | 格式、scale、加载转换、硬件 Kernel 与 fallback 为何必须联合决策？ | quantization ABI 与 dispatch | `13/14` → `24/25` |
| [[02_engineering/03_infer_frameworks/vllm/23_vllm_compilation_cudagraph_analysis|23 编译与 CUDA Graph]] | 动态 shape 怎样得到可编译、可捕获、地址稳定的执行区域？ | compile/cache/capture/replay 生命周期 | `14/15/16` → `24/25` |
| [[02_engineering/03_infer_frameworks/vllm/24_vllm_fused_ops_and_kernels_analysis|24 融合算子与 Kernel]] | 何时融合/专用 Kernel 真正省成本，何时必须 fallback？ | provider、Kernel family 与收益模型 | `14/21/23/25` → 设备级性能验证 |
| [[02_engineering/03_infer_frameworks/vllm/25_vllm_ir_and_fusion_passes_analysis|25 IR 与融合 Pass]] | IR 怎样固定语义并安全排序 functionalization、fusion 与 lowering？ | IR、alias/donation 与 pass 顺序 | `21/23` → `24` |

### 2.4 规模化、扩展与生产闭环

| Owner | 读者问题 | 本页拥有 | 前置 → 后续 |
|---|---|---|---|
| [[02_engineering/03_infer_frameworks/vllm/22_vllm_distributed_inference_analysis|22 分布式推理]] | 并行轴怎样映射到 rank/group/collective，DBO 怎样保持顺序？ | TP/PP/DP/EP/CP、executor 与 DBO | `10/13/14/17` → `23/24/26/29` |
| [[02_engineering/03_infer_frameworks/vllm/26_vllm_disaggregated_kv_serving_analysis|26 分离式 KV Serving]] | KV 怎样跨 Engine 转移、证明完成并在失败时回收？ | connector/store、transferable group 与 lease | `11/12/17/22` → `27` |
| [[02_engineering/03_infer_frameworks/vllm/27_vllm_observability_reliability_analysis|27 可观测性与可靠性]] | SLO 症状怎样闭环到资源承诺和进程故障域？ | metrics/events/traces/sentinel 反馈 | `11/12/17/22/26` → 生产处置与 `01` 调优闭环 |
| [[02_engineering/03_infer_frameworks/vllm/28_vllm_extension_plugin_system_analysis|28 扩展与插件]] | 第三方扩展怎样按进程/应用生命周期生效而不污染隐式状态？ | plugin ABI、发现、选择与初始化 | `04/13/17/22` → `27` |
| [[02_engineering/03_infer_frameworks/vllm/29_vllm_weight_transfer_online_update_analysis|29 在线权重更新]] | 权重字节、rank-local 更新、版本标签和 cache/runner 何时真正对请求可见？ | vLLM 侧 pause、transfer session、version commit 与失败边界 | `10/12/13/15/16/22` → `27` |

## 三、四条推荐学习路径

1. **Architecture**：`02 → 03 → 04 → 10 → 17`。先确认系统为何分层，再区分请求语义、Engine 状态与服务拓扑。
2. **Request / resource hot path**：`04 → 10 → 11 → 12 → 15 → 16 → 18`。`15/16` 先建立两代 runner 的状态布局差异；多模态请求在 `11` 后转入 `19`，投机请求在 `18` 后转入 `20`。
3. **Model / device optimization**：`13 → 14 → 15 → 16 → 21 → 23 → 25 → 24`。先固定模型和 backend 合同，再对照两代 runner 的设备状态，最后看低精度、编译图与 Kernel。
4. **Production / scale**：`17 → 22 → 26 → 27 → 28 → 29`。先建立进程与 rank 故障域，再看跨实例 KV、观测、扩展和在线更新。

## 四、按症状跳转

| 症状或目标 | 第一 owner | 一跳后续 |
|---|---|---|
| 协议行为、stop、tool/reasoning 或 pooling 输出不符预期 | `04` | token 选择看 `18`；媒体输入看 `19` |
| TTFT 高、长 prompt 阻塞 decode | `11` | `12`、`19`、`26` |
| TPOT 高、CPU/GPU 无法重叠 | `16` | 若配置回退 MRV1，先看 `15`；再看 `20`、`23`、`24` |
| 显存不足、prefix 命中异常或 eviction 抖动 | `12` | `21`、`22`、`26` |
| 模型、量化或 attention 组合不兼容 | `13` | `14`、`21`、`24` |
| 多 GPU 空转或 collective hang | `22` | `17`、`23`、`27` |
| 权重更新后输出、cache 或版本可见性异常 | `29` | `12`、`15/16`、`27` |
| 线上尾延迟、错误或 hang 难以归因 | `27` | 回到对应资源/进程 owner |

## 五、证据与边界口径

- 所有内容页使用同一冻结源码基线；正文中的 `file:line` 均相对 vLLM 仓库根。
- 当前行为由该 commit 的源码和测试决定；同 commit 的 `docs/design/` 只用于说明公开设计意图。
- 跨页只保留相邻合同和状态交接；机制、正确性证明、fallback 与失败边界只在上表指定的 owner 页维护。
- 训练器算法、rollout 编排与 policy 产生不属于本域；`29` 只拥有 vLLM 侧在线更新与可见性协议。

## Related Pages

- [[02_engineering/03_infer_frameworks/index|推理框架目录]] — vLLM 在整体推理框架技术栈中的位置。
- [[02_engineering/03_infer_frameworks/01_llm_inference_technology_stack_analysis|大模型推理技术栈全景]] — 与其他推理系统比较能力边界。
- [[02_engineering/03_infer_frameworks/sglang/index|SGLang 推理框架]] — 对照另一条调度与编译实现路线。
- [[02_engineering/03_infer_frameworks/speculative_decoding/index|投机推理专题]] — 跨引擎理解 draft/verify 算法族。
- [[02_engineering/03_infer_frameworks/mooncake_analysis|Mooncake 分离式推理]] — 对照论文层面的 KV 数据平面设计。
