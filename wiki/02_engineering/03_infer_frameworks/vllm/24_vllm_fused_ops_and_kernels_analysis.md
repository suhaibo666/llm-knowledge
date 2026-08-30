---
title: "vLLM 融合算子与 Kernel：用收益账本约束专用化与 fallback"
---

# vLLM 融合算子与 Kernel：用收益账本约束专用化与 fallback

> **读者问题**：把几个算子写进同一个 Kernel 就一定更快吗？vLLM 怎样在 native、vLLM C、AITER、Oink、Triton、FlashInfer/CUTLASS 等实现之间选择，又在 shape、dtype、硬件或 workspace 合同不成立时安全 fallback？
> **源码基线**：`vllm-project/vllm@6b110badbb22d3f66c7218b71138f13b7a6b3419`（`main`，提交时间 2026-08-29T02:40:53Z）
> **中心命题**：融合不是“Kernel 数越少越好”，而是一笔受语义合同约束的收益账：只有省掉的 launch、全局内存中间态与格式转换，大于更高寄存器/共享内存压力、workspace、shape/dtype/hardware 特化和维护成本时，专用实现才值得选择。vLLM 因而先稳定 op 语义，再把平台可用性、参数兼容性与部署能力拆成显式谓词；fallback 只能换成实现同一合同的 provider，不能顺手改变数值、布局或并行语义。
> **所有权边界**：本页拥有融合收益模型、provider/Kernel family、能力选择、workspace 与 fallback；只选 residual+RMSNorm(+quant) 和 fused MoE 两组代表族，不做 Kernel 名录。量化 pack/scale ABI 归 [[02_engineering/03_infer_frameworks/vllm/21_vllm_quantization_analysis|vLLM 量化设计]]，attention backend 合同归 [[02_engineering/03_infer_frameworks/vllm/14_vllm_attention_backends_analysis|vLLM Attention Backend]]，FX/IR pattern、alias/functionalization 与 pass 顺序归 [[02_engineering/03_infer_frameworks/vllm/25_vllm_ir_and_fusion_passes_analysis|vLLM IR 与融合 Pass]]。
> **最近更新**：2026-08-30。按 `6b110bad` 重建收益账、provider 选择、代表 Kernel family 与 fallback 边界。

## 1. 背景：融合优化先问“消掉了什么”

直观实现把 `add → RMSNorm → quant` 或 `route → permute → GEMM1 → activation → GEMM2 → unpermute/reduce` 分成独立 Kernel。每个边界都可能增加一次 host launch，并让生产者写出、消费者再读回中间 tensor；融合可以把局部值留在寄存器或 shared memory，也可以直接写入消费者要求的量化/布局格式。源码 benchmark 正是把 RMSNorm 后再 quant 的 unfused 函数与 fused 函数放在同一组 shape 网格中比较，而不是把“融合”当成先验胜者：输入覆盖 token 数 `1..1024`、多种 hidden size、residual 开关、dtype 与 group size（`benchmarks/fused_kernels/layernorm_rms_benchmarks.py:23-53`），unfused 路径显式先 norm 再 quant，fused 路径则一次调用 norm+quant op（`benchmarks/fused_kernels/layernorm_rms_benchmarks.py:56-140`）。

下面的账本是基于这些实现边界的**分析推断**；源码没有提供统一公式，也没有给出跨硬件通用阈值。

| 账本项 | 融合可能得到的收益 | 同时支付的成本 | 决策时必须带上的证据 |
|---|---|---|---|
| launch | 多个短 Kernel 合并为一个 launch | 更大 Kernel 可能拉长关键路径，削弱并行/重叠机会 | 同一真实 workload 的 launch trace 与端到端时间 |
| 中间态 | 少一次完整 tensor 写回与重读；可原地复用 output/residual | alias、stride 与生命周期变严格；错误复用会破坏语义 | bytes moved、copy event、alias/stride 边界测试 |
| 格式转换 | norm/activation 后直接产出 quantized 或 backend layout | dtype、scale 粒度、block shape 与 pack 进入 Kernel ABI | reference 数值、scale/layout 与边界 shape |
| 专用化 | 针对 token 数、hidden/expert shape 和硬件 tile 调优 | 代码、配置、编译 cache、回归矩阵与维护面扩张 | 覆盖 prefill/decode 与 TP/EP-local shape 的 benchmark |
| workspace | 两阶段计算可重用 scratch，避免反复分配 | 预留显存、capture 地址与异步完成前的存活期 | workspace 上界、复用顺序与峰值显存 |

因此应比较的是**一次完整调用的总成本**，不是源码中 op 的数量。仓库自带的 MoE 默认配置 benchmark 也把 tuned、old default、new default 在相同 `M/E/N/K/topk/dtype/block_shape` 下计时，并明确输出 Kernel 时间与 speedup（`benchmarks/kernels/benchmark_moe_defaults.py:4-16`；`benchmarks/kernels/benchmark_moe_defaults.py:34-77`）。这支持一个重要边界：配置变化可能改变胜者，仓库里存在 benchmark 驱动的选择依据，但源码没有发布一个可直接外推到任意部署的固定性能结论。

## 2. 为什么分成语义 op、provider 与 Kernel family

### 2.1 背景与替代方案

若模型层直接调用某个设备扩展，换硬件就要改模型；若只保留 native PyTorch，eager 路径又无法保证得到目标设备的专用布局与实现。vLLM 保留一个稳定语义边界，再让 provider 声明“在哪个平台可注册、哪些实参可执行”；`IrOp` 构造时总是注册 native implementation，并把它作为始终可用的语义实现（`vllm/ir/op.py:155-211`）。

这条结构胜过“一个全局 `if device` 选完所有 Kernel”，因为平台可用性和单次调用兼容性不是同一个问题。provider 注册接口把 `supported` 定义为硬件/库的静态门，把 `supports_args` 定义为 dtype/shape 等动态门，并要求每个实现保持 native 的签名与语义（`vllm/ir/op.py:244-273`）。这里的“为什么胜出”是依据接口分工重建的**分析推断**；源码只自陈最终合同。

### 2.2 实现机制：三层选择，不是一张名字表

| 层 | 输入 → 输出 | 拥有的决策 | 不拥有 | 承重证据 |
|---|---|---|---|---|
| layer / `CustomOp` | 稳定 layer 调用 + build platform → native 或平台方法 | 类级语义、CUDA/HIP/XPU/CPU/TPU/OOT 方法；对象构造时固定平台路径 | 同一对象每步按 shape 动态换 provider | `vllm/model_executor/custom_op.py:103-176`；`vllm/model_executor/custom_op.py:191-207` |
| `IrOp` provider | 同一个语义 op + priority + 当前 tensors → compatible impl | provider priority、静态可用性、每次调用的 argument predicate、native fallback | 图中 pattern 为什么/何时被改写 | `vllm/ir/op.py:327-366`；`vllm/ir/op.py:389-413` |
| Kernel family / oracle | dtype/quant/layout + local shape + hardware + parallel/routing feature → concrete Kernel class/config | tile/layout、workspace、实现能力与 family 内候选顺序 | checkpoint 如何形成 pack/scale；collective 的全局语义 | `vllm/model_executor/layers/fused_moe/modular_kernel.py:550-596`；`vllm/model_executor/layers/fused_moe/oracle/unquantized.py:208-325` |

`CustomOp` 被禁用时走可选编译的 `forward_native`；启用时按当前 build platform 绑定 `forward_hip/cpu/tpu/xpu/oot/cuda`，源码明确说明这里不支持动态 platform dispatch（`vllm/model_executor/custom_op.py:174-207`）。其注释还指出：在 opaque custom op 内部编译 native 并不能得到跨 op fusion，所以能展开时仍应展开（`vllm/model_executor/custom_op.py:209-216`）。这解释了为何“专用 Kernel 边界”和“编译器可见边界”要同时保留，而不能把所有算子都包成 opaque op。

`IrOp` 则在 priority 中逐项检查 `supports_args`；没有显式 priority 时用 native，priority 中没有全参数 provider 时会自动在末尾补 native 并告警（`vllm/ir/op.py:327-366`；`vllm/ir/op.py:389-413`）。平台默认还会考虑执行上下文：CUDA 在 Inductor 编译时默认 native，非 codegen 时默认 `vllm_c → native`，可选 Oink 再插到前面（`vllm/platforms/cuda.py:696-715`）；ROCm 只在 CUDA Graph、AITER flags 与设备条件同时成立时把 AITER RMSNorm 提到默认前面（`vllm/platforms/rocm.py:1101-1128`）。这是 provider 选择的执行上下文，不是 IR pass 顺序；后者仍由 page 25 拥有。

### 2.3 约束：fallback 的末项必须覆盖全部实参

dispatch 热路径要求 priority 最后一项支持所有实参，否则抛出 internal-bug 异常（`vllm/ir/op.py:327-366`）。测试同时固定两种行为：静态 unsupported provider 在设置 priority 时被过滤；只支持偶数 shape 的 provider 对奇数 shape 自动落到 native（`tests/ir/test_op.py:330-371`）。所以 fallback 不是捕获任意 Kernel error 后重试；它是在 launch 前依据已声明谓词选择同语义实现。

## 3. 代表族一：residual + RMSNorm（再接 quant）

### 3.1 背景与为什么值得融合

residual add 与 RMSNorm 都逐元素读取同一 token row；若分开执行，add 的完整输出既要落地又要被 norm 重读。融合实现可以在一次 traversal 中形成 residual、计算归一化并写出 norm output；再把 quant 合入时，还可避免 materialize 高精度 norm output。这里关于带宽收益的因果链是**分析推断**，其执行边界可由 benchmark 的 unfused/fused 对照直接核验（`benchmarks/fused_kernels/layernorm_rms_benchmarks.py:56-140`）。

融合不允许省掉可见语义。native reference 要求 residual 输出等于 `x + residual`，norm 输出等于该和的 RMSNorm；测试跨 token 数、hidden size、FP16/BF16/FP32 核验 shape/dtype/device 与数值（`tests/kernels/ir/test_layernorm.py:264-319`）。每个 provider 还必须与 native 结果相符，并验证 priority dispatch 与直接调用一致（`tests/kernels/ir/test_layernorm.py:321-348`）。

### 3.2 provider 选择怎样暴露真实边界

同一个 `fused_add_rms_norm` 不是“检测到 GPU 就调用最快扩展”这么简单：

- vLLM C provider 要求没有 `variance_size` override、weight 与 activation dtype 匹配，并声明 inplace；ROCm 遇到非 contiguous 输入时明确调用 native 后再 copy 回输入，而不是把不合法 stride 交给设备 Kernel（`vllm/kernels/vllm_c.py:47-74`）。
- AITER provider 额外只接受 FP16/BF16 activation，且同样拒绝 `variance_size` override（`vllm/kernels/aiter_ops.py:78-103`）。
- Oink provider 要求可视为二维、weight contiguous、input/residual shape 与 dtype 相同，并满足 256-bit vectorization stride；它同样声明 inplace（`vllm/kernels/oink_ops.py:50-58`；`vllm/kernels/oink_ops.py:90-127`）。

这些 guard 展示了融合的代价：为了少一次 launch/中间态，Kernel 把 dtype、stride、shape 与 alias 写入合同。测试不是只测一个 happy path；provider 注册表按平台核对 native/vLLM C/AITER/Oink 的可用性，并要求所有设备 provider 拒绝 `variance_size` override（`tests/kernels/ir/test_layernorm.py:211-230`；`tests/kernels/ir/test_layernorm.py:321-352`）。不满足时应选择 native 或另一个 compatible provider；若调用者强行绕过 `supports_args`，就不再属于安全 fallback 路径。

### 3.3 何时这类融合可能不划算

这是依据账本作出的**分析推断**：token row 很少时，省 launch 往往更重要；tensor 很大时，省 HBM 中间态更重要；但若 native/codegen 能把周边 op 一起优化，固定 opaque provider 反而可能丢掉更大的跨 op 机会。与此相呼应，CUDA 在 Inductor 模式默认 native，而在无 codegen 时优先 vLLM C（`vllm/platforms/cuda.py:696-715`）。最终阈值必须测量：仓库 benchmark 遍历 token、hidden、dtype、residual 与 group-size 组合，并把 fused/unfused 放进同一个 timer（`benchmarks/fused_kernels/layernorm_rms_benchmarks.py:174-285`），没有源码证据支持“一条 provider priority 对所有 shape 都最优”。

## 4. 代表族二：fused MoE 是组合收益，不是单个巨型 Kernel

### 4.1 背景与为什么不能只选一个 monolithic 实现

MoE 同时包含 routing、token 重排/通信、两次 expert GEMM、activation、router weight 与 combine/reduce。monolithic family 可以把 router 和 experts 一起消费，减少接口与中间态；modular family 则可独立组合 prepare/finalize 与 expert compute，并让 async all-to-all 或 shared experts overlap。源码把 monolithic `apply` 的输入提升为 router logits，并说明该形式用于 fused router+experts（`vllm/model_executor/layers/fused_moe/modular_kernel.py:1533-1563`）；modular 路径在 prepare 支持 async 时启动 async prepare、注册/执行 receive hook，再取得已重排 activation 与 metadata（`vllm/model_executor/layers/fused_moe/modular_kernel.py:1211-1283`）。

“总用 monolithic 以最小化 launch”会把 routing method、all-to-all、quantization、parallel mode 与 shared expert 能力锁进同一实现；“全部拆开”则放弃更深融合。源码没有写出这段备选论证，以上为**分析推断**。现行边界证明它选择兼容两者，但禁止随意混搭：prepare/finalize 与 experts 必须同为 monolithic 或同为 modular，否则构造直接报错（`vllm/model_executor/layers/fused_moe/modular_kernel.py:1600-1631`）。

### 4.2 oracle 不是按名称选，而是对部署谓词求交

MoE Kernel 的通用 `is_supported_config` 依次检查当前设备、gated activation、具体 activation、weight/activation quant scheme、parallel config、routing method、router-logits dtype、hidden shape、activation format、batch invariance 与 LoRA（`vllm/model_executor/layers/fused_moe/modular_kernel.py:550-596`）。因此决定 provider 的不是一个 `moe_backend` 字符串，而是至少以下状态：

| 决策维度 | 它为什么可能改变最佳或合法 family | 代表证据 |
|---|---|---|
| hardware/library | CUDA、ROCm、XPU、CPU 提供的 family 不同；同一 CUDA 世代也可能重排优先级 | unquantized oracle 在 ROCm 以 AITER 开头，在 CUDA 以 FlashInfer/Triton family 候选开头，且 Hopper 默认把两个 FlashInfer BF16 候选后移（`vllm/model_executor/layers/fused_moe/oracle/unquantized.py:48-99`） |
| dtype/quant/layout | provider 必须消费既有 weight/activation quant key 与 activation format | FP8 oracle 先构造 quant-specific family priority，再按 DeepEP layout、Hopper TP/EP 与平台重排（`vllm/model_executor/layers/fused_moe/oracle/fp8.py:69-133`） |
| local shape | TP 改写 partition 后的 intermediate size 可能触发对齐 guard | BF16 TRT-LLM LoRA 要求 per-partition intermediate size 为 128 的倍数，否则回 Triton（`vllm/model_executor/layers/fused_moe/oracle/unquantized.py:183-205`） |
| routing/parallel | monolithic router、EP/DP、all-to-all 与 batched activation format 必须一起兼容 | oracle 从 parallel config 先决定 standard/batched activation format，再逐 class 调 `is_supported_config`（`vllm/model_executor/layers/fused_moe/oracle/unquantized.py:238-284`） |
| feature contract | clamp、LoRA、batch invariance 或 deferred finalize 不能被静默遗漏 | config 明确要求不支持 SwiGLU clamp 的 backend 在 oracle 中被过滤（`vllm/model_executor/layers/fused_moe/config.py:1327-1330`） |

auto 模式依 priority 逐个检查 class，返回第一个 compatible candidate；全都失败才抛 `NotImplementedError`（`vllm/model_executor/layers/fused_moe/oracle/unquantized.py:286-325`）。显式指定 backend 的语义不同：指定 family 不兼容时立即 `ValueError`，不会悄悄换成用户未指定的 family（`vllm/model_executor/layers/fused_moe/oracle/unquantized.py:271-297`）。测试固定了两种代表 fallback：FlashInfer TRT-LLM monolithic 不支持但 modular 支持时留在同 family 改选 modular；DeepEP high-throughput 与该 BF16 path 不兼容时，auto 退到 Triton（`tests/kernels/moe/test_unquantized_backend_selection.py:266-302`；`tests/kernels/moe/test_unquantized_backend_selection.py:350-382`）。

### 4.3 workspace 与中间态复用：少一次 copy 仍要付显存合同

modular expert interface 要求 provider 根据 `M/N/K/topk/global/local experts/activation format` 报告两块 scratch 与最终 output shape；GEMM1 与 GEMM2 不同时存活，所以允许共享 workspace13（`vllm/model_executor/layers/fused_moe/modular_kernel.py:837-874`）。具体 allocator 按 chunked `M` 计算 scratch、按 full `M` 计算 final output，并让 GEMM1/3 与单 chunk output 共享一块较大 buffer（`vllm/model_executor/layers/fused_moe/modular_kernel.py:1120-1181`）。

这能减少分配和峰值，但不是“零中间态”：workspace2 仍必须与 common workspace 同时存活，provider 还必须准确报告上界。output buffer 只有 shape、dtype、device 与 contiguous 全部相符才可 alias；源码注释记录该 alias 用来去掉下游冗余 copy，并在 ROCm AITER enable 条件下才采用（`vllm/model_executor/layers/fused_moe/modular_kernel.py:1329-1365`）。所以 workspace/alias 优化的失败边界不是性能稍差而已：低估 shape 会越界，过早复用会破坏仍在飞行的计算，错用 output alias 会改变可见结果。

## 5. Selection 与 fallback：从“候选”到“可证明的实现”

### 5.1 推荐的选择顺序

1. **先固定语义**：output、可见 residual、dtype/scale/layout、router/reduce 状态与 alias 必须由 native/reference 或上层合同定义；provider 无权改写。
2. **过滤静态可用性**：平台、compute capability、扩展库与 build 决定 family 是否进入候选；unsupported provider 在 priority 安装时就过滤（`vllm/ir/op.py:389-413`）。
3. **过滤部署能力**：量化 key、parallel/routing、LoRA、batch invariance、activation format 与 hidden shape 共同决定 MoE class 能否实例化（`vllm/model_executor/layers/fused_moe/modular_kernel.py:550-596`）。
4. **检查每次调用实参**：普通 `IrOp` 再按 dtype、shape、stride 等 `supports_args` 选择 provider（`vllm/ir/op.py:327-361`）。
5. **在 compatible 集合内比较性能**：用真实 prefill/decode、TP/EP-local shape、CUDA Graph/async 条件测端到端，不用单个名字或单点 microbenchmark 替代部署分布。

第 5 步是本页依据 benchmark 结构给出的**分析建议**：RMSNorm benchmark 显式扫 shape/dtype/residual（`benchmarks/fused_kernels/layernorm_rms_benchmarks.py:41-53`），MoE benchmark 的配置键显式包含 `M/E/N/K/topk/dtype/block_shape`（`benchmarks/kernels/benchmark_moe_defaults.py:34-77`）。源码未实现一个统一 runtime autotuner，因此不能声称 vLLM 会为每次调用现场测出全局最快 Kernel。

### 5.2 四种结果必须区分

| 结果 | 正确行为 | 为什么不是同一类“fallback” |
|---|---|---|
| provider 静态不可用 | 从 priority 移除，再看下一项 | 当前进程根本没有可调用实现 |
| 当前实参不兼容 | 调用前选下一 provider，通常最终 native | 同一 op 的 shape/dtype/stride 局部边界 |
| auto MoE family 不兼容 | oracle 尝试下一 compatible family | 部署策略允许自动选择，语义与 quant/layout 合同仍不变 |
| 显式 family 不兼容或全候选失败 | `ValueError` / `NotImplementedError` | 静默改 family 会违背用户意图；没有同合同实现时不存在安全 fallback |

fallback 之后仍必须过 reference。融合 RMSNorm 测试以 native 为 oracle，对每个支持 provider 比较两个输出；不兼容参数先 skip，不把 crash 当作选择逻辑（`tests/kernels/ir/test_layernorm.py:321-352`）。MoE selection 测试则分别覆盖 platform 默认、显式 family、monolithic→modular 和跨 family fallback（`tests/kernels/moe/test_unquantized_backend_selection.py:170-197`；`tests/kernels/moe/test_unquantized_backend_selection.py:238-302`；`tests/kernels/moe/test_unquantized_backend_selection.py:350-382`）。

## 6. 约束、维护成本与验收

| 风险 | 必须守住的不变量 | 代价或失败边界 | 验收方式 |
|---|---|---|---|
| 省中间态改变数值 | fused output 与 reference 在该 dtype 容差内等价；可见 residual/reduce 状态不丢 | fusion boundary 可能改变 rounding；不能只看最终文本 | 每 provider reference 对照、边界 dtype/shape |
| inplace/alias | 只有声明并满足 shape/dtype/device/stride/lifetime 才复用 | 不兼容时 out-of-place/native；错误 alias 是 correctness bug | alias 与非 contiguous case、copy trace |
| tile/pack 特化 | TP/EP-local shape、block quant 与 provider layout 匹配 | padding、次级 family 或硬失败 | 不整除、空 token、最小/最大 token 测试 |
| workspace | scratch 上界正确，异步完成前保持存活，capture 时地址稳定 | 多占显存；错误复用会越界或读旧值 | 峰值显存、sanitizer/event、graph replay |
| provider 扩张 | 每个实现都维护语义、predicate、fake/compile/capture 与性能回归 | 新 provider 增加测试/配置/cache invalidation 面 | priority/filter 测试 + reference + 真实 workload benchmark |

维护成本也是收益账的一部分。IR provider 实现被 Dynamo 隐藏，因此 priority config 的 hash 还要显式纳入各实现 UUID，避免实现变化却复用旧 compile cache（`vllm/config/kernel.py:37-59`）；worker 初始化才导入当前平台 Kernel 并安装 priority（`vllm/config/kernel.py:70-105`；`vllm/v1/worker/worker_base.py:98`）。新增 provider 若只交一个快的 Kernel、却不补 argument predicate、reference 测试和 cache identity，就没有完成集成。

提交前的最小验收顺序应是：

1. native/reference 定义两个实现必须共同保持的数值、shape、dtype、layout 与副作用；
2. provider predicate 覆盖 hardware、dtype、shape、stride、parallel/routing 和 feature flags；
3. fallback 测试证明每个拒绝分支选择下一 compatible 实现或清楚硬失败；
4. workspace/alias 在 eager、compile/CUDA Graph 与 async 条件下验证生命周期；
5. 最后才用真实 workload 对比 launch、HBM/copy、workspace 峰值、Kernel time 与端到端 TPOT。

## 7. 有源码锚点的发展方向

> [!note] 分析推断
> 这里只从当前 TODO/临时分支外推维护压力，不把它写成已承诺 roadmap。

- unquantized MoE oracle 自陈：当前必须“偷看” prepare/finalize 才能决定 batched/standard activation format，等 TP 与 DP/EP selection 统一后可先选 prepare/finalize（`vllm/model_executor/layers/fused_moe/oracle/unquantized.py:241-249`）。这说明选择器正承受组件耦合压力；合理方向是让 format contract 更早成为显式输入，而不是继续在 provider 名单里堆特例。
- CUDA 的 Oink 环境变量被标注为待移除，用户可直接使用 IR op priority（`vllm/platforms/cuda.py:706-714`）。这指向一个更统一的 provider policy 面：平台提供默认，用户修改 priority，而 capability predicate 仍负责 correctness。

## Related Pages

- [[02_engineering/03_infer_frameworks/vllm/21_vllm_quantization_analysis|vLLM 量化设计]] — 拥有 weight/scale/zero 与 pack ABI；本页从已提交的表示接手 Kernel family 选择。
- [[02_engineering/03_infer_frameworks/vllm/25_vllm_ir_and_fusion_passes_analysis|vLLM IR 与融合 Pass]] — 拥有 pattern、alias/functionalization、pass 顺序与 lowering；本页只解释其产物怎样选择 provider。
- [[02_engineering/03_infer_frameworks/vllm/23_vllm_compilation_cudagraph_analysis|vLLM 编译与 CUDA Graph]] — 解释 native/codegen、opaque op、workspace 地址与 capture/replay 的生命周期边界。
- [[02_engineering/03_infer_frameworks/vllm/14_vllm_attention_backends_analysis|vLLM Attention Backend]] — attention metadata/KV layout 的能力协商在此；本页不把 attention backend 重列成 Kernel family。
- [[02_engineering/03_infer_frameworks/vllm/22_vllm_distributed_inference_analysis|vLLM 分布式推理]] — 拥有 TP/DP/EP 与 collective 顺序；本页只使用 local shape 和 parallel feature 作为 Kernel compatibility 输入。
