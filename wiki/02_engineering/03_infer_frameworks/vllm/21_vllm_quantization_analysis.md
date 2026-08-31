---
title: "vLLM 量化 ABI：格式、Scale、加载转换与 Kernel 必须联合决策"
---

# vLLM 量化 ABI：格式、Scale、加载转换与 Kernel 必须联合决策

> **读者问题**：为什么同样写着 W4A16 或 FP8 的 checkpoint，不能只按位宽选择一个 GEMM；vLLM 又怎样保证 checkpoint 的 pack/scale 语义、TP 后的局部形状、post-load 排列与最终设备 Kernel 始终是同一份合同？
> **源码基线**：`vllm-project/vllm@6b110badbb22d3f66c7218b71138f13b7a6b3419`（冻结的 detached checkout，提交时间 2026-08-29T02:40:53Z）
> **中心命题**：量化不是加载完成后的 dtype cast，而是一条逐步收紧的 ABI：configure 阶段确定 checkpoint/在线格式与全局能力边界，layer 构造阶段把格式变成 TP-local 参数与 scale 形状，post-load 阶段把加载表示提交为 Kernel 表示，runtime 只在能实现同一数值合同的 Kernel 之间派发。任一阶段若偷偷改变 pack axis、scale 粒度、zero-point、activation dtype 或分片语义，模型可能仍能运行却计算另一套数值；所以兼容谓词、转换和 fallback 必须联合决定。
> **所有权边界**：本页拥有量化 config → per-layer method → 参数/scale ABI → post-load transform → hardware dispatch/fallback，以及 packed mapping 对量化规则的消费接缝。通用 checkpoint 枚举、名称映射和参数分片提交归 [[02_engineering/03_infer_frameworks/vllm/13_vllm_model_library_analysis|vLLM 模型与权重 ABI]]；Kernel 内部算法、tile 和 provider 编程归 [[02_engineering/03_infer_frameworks/vllm/24_vllm_fused_ops_and_kernels_analysis|vLLM 融合算子与 Kernel]]；KV layout 与 attention backend 的完整协商归 `12/14`，本页只保留 scale 名称与量化能力接缝。
> **最近更新**：2026-08-30。按 `6b110bad` 重建 configure → create/load → post-load → dispatch/fallback 主线，并补齐在线量化、TP scale 一致性和显式失败边界。

## 1. 背景：低精度名字不是可执行格式

一个量化线性层至少要同时回答六个问题：权重的数值类型是什么、整数怎样 pack、scale 按 tensor/channel/group/block 哪个粒度解释、是否有 zero-point 或 activation-order metadata、TP 后本 rank 的 K/N 是多少、最终 Kernel 要读哪种排列。vLLM 把其中与实现无关的最小描述收进 `MPLinearLayerConfig`：全局/局部 weight shape、weight/activation type、group size、zero-point 与 `g_idx` 都是 Kernel 兼容谓词，而不是调优提示（`vllm/model_executor/kernels/linear/mixed_precision/MPLinearKernel.py:14-35`）。

基类也直接暴露同一生命周期：`create_weights()` 建立加载目标，`apply()` 消费 layer 上的表示；二者是抽象方法，且 `apply()` 只明确要求 create 已先发生。`process_weights_after_loading()` 则是默认 no-op 的可选 hook，具体 method 才用它完成转置、重排或量化（`vllm/model_executor/layers/quantization/base_config.py:20-72`）。这说明量化状态不是某个 CLI 字符串，而是跨模型构造与执行长期存活的 layer ABI。

| 决策点 | 输入 → 输出 | 本阶段必须固定的状态 | 不满足时的正确行为 |
|---|---|---|---|
| Configure | HF quant config / 在线配置 / 用户 override / 平台 → `QuantizationConfig` | 格式身份、activation dtype 范围、全局 capability、ignore 规则 | 配置不一致或平台不支持时尽早拒绝 |
| Create / load | config + layer prefix + 全局/TP-local shape → 参数容器；checkpoint tensor → 容器 | pack axis、scale/zero shape、logical shard identity、TP ownership | 显式 unquantized layer，或构造失败；不能猜 shape |
| Post-load | checkpoint/在线中间表示 → Kernel 表示（若 method 需要） | 转置、repack、在线 quant、scale 合并、替换后的参数身份 | 正常 loader 在返回模型前调用；绕过该顺序只会对依赖转换的 method 破坏 Kernel 合同，基类没有通用 `apply()` guard |
| Runtime dispatch | 已提交参数 + activation → output | Kernel 必须实现完全相同的数值与布局合同 | 换下一个兼容实现；没有兼容实现则硬失败 |

直观替代是“模型先按 BF16 构造并加载，再统一 cast，forward 时按位宽选 Kernel”。以下是**分析推断**：它失败的根因不是缺少某个量化方法，而是把四个提交点压成一个无类型边界——checkpoint pack 与 Kernel pack 可以不同，scale 可能跨 TP shard 求全局极值，融合 QKV 的三个逻辑矩阵也可能有独立 scale。现行接口将这些差异前移并显式化，代价是配置、层、loader seam 与 Kernel 都要维护同一 ABI。

## 2. Configure：先证明格式与部署能组成合同

### 2.1 checkpoint 身份、用户意图与在线格式不是同一个字段

对预量化 checkpoint，`ModelConfig` 先读 HF `quantization_config.quant_method`，再按有序 override 列表探测能兼容该 checkpoint 的实现；若用户显式 `--quantization` 与解析结果不同，直接报错，而不是强行用用户名字解释另一种字节布局（`vllm/config/model.py:1245-1324`）。例如 GPTQ 的 override 只接受 checkpoint 声明为 `gptq`，且用户选择必须仍在 GPTQ/Marlin/AutoGPTQ 兼容集合中（`vllm/model_executor/layers/quantization/auto_gptq.py:218-238`）。

在线量化走另一条入口：shorthand 先展开成 linear/MoE 的 weight `QuantKey`，显式 `quantization_config` 再按 layer kind 覆盖；若传统 checkpoint quant 名与在线配置混用，会在解析时拒绝（`vllm/config/quantization.py:114-148`；`vllm/config/quantization.py:158-194`）。加载侧也不伪造 checkpoint quant config：只有预量化格式从 HF config/file 构造 config；在线路径明确从 FP16/BF16 checkpoint 在加载期转换（`vllm/model_executor/model_loader/weight_utils.py:240-287`；`vllm/model_executor/model_loader/weight_utils.py:318-327`）。

这层拆分胜过“所有格式共用一个 `quantization` enum”。同名 checkpoint method 可能需要专用 parser，而在线 shorthand 描述的是目标 weight/activation 方案；把二者混为一谈会让“读取已有 scale”和“现场计算 scale”失去可区分的生命周期。

### 2.2 能力检查是多级谓词，不是 GPU 型号白名单

配置先经过三道全局门：方法名必须注册；平台的 `supported_quantization` 非空时必须包含该方法；实例化后的 config 还要同时满足设备 capability 与模型 activation dtype（`vllm/config/model.py:1326-1332`；`vllm/platforms/interface.py:962-969`；`vllm/config/vllm.py:751-784`）。ROCm/XPU 因此可以给出显式方法集合，而空集合的平台并不等于“全部 Kernel 都可用”，后面仍有 method/kernel 级检查（`vllm/platforms/rocm.py:513-537`；`vllm/platforms/xpu.py:103-129`）。

`QuantizationConfig` 自身声明支持的 activation dtype、最低 capability、配置文件来源和 layer → method 选择；checkpoint-compatible override 只是其中一个可选钩子（`vllm/model_executor/layers/quantization/base_config.py:87-159`；`vllm/model_executor/layers/quantization/base_config.py:179-192`）。因此“配置通过”只证明整个方法在当前部署上有可能成立，不证明每个 TP-local shape 已经找到 Kernel；shape 级证明必须等 layer 构造。

### 2.3 layer prefix 把模型命名带入量化规则

模型构造前，loader seam 把 HF→vLLM rename mapper 与 `packed_modules_mapping` 交给 quant config；采用 `SupportsQuant` 的模型在 `__new__` 中做同一件事（`vllm/model_executor/model_loader/utils.py:37-60`；`vllm/model_executor/model_loader/utils.py:265-286`；`vllm/model_executor/models/interfaces.py:1201-1233`）。以 Llama 为例，runtime 的 `qkv_proj` 和 `gate_up_proj` 分别对应 checkpoint 的三个、两个逻辑投影（`vllm/model_executor/models/llama.py:446-460`）。

本页只拥有 mapping 被量化规则消费后的不变量：融合层不能让 constituent shards 落入不同精度。skip matcher 先把 fused prefix 展开成逻辑 shard prefixes；只要部分 shard 命中、部分未命中就抛错（`vllm/model_executor/layers/quantization/utils/quant_utils.py:635-678`）。名称怎样递归找到 parameter、TP slice 怎样 copy 属于页面 `13`；但“融合后的一个 Kernel 调用必须看到一致 quant scheme”属于本页。

KV scale 只在这里保留一个窄加载接缝：base config 把 checkpoint 中旧的 fused `kv_scale`、ModelOpt 投影 scale 和若干 Q/K/V 命名统一映射到 attention layer 的 `q_scale/k_scale/v_scale`（`vllm/model_executor/layers/quantization/base_config.py:194-227`）。scale 最终对应哪种 KV dtype/layout、由哪个 attention backend 消费，仍由页面 `12/14` 拥有；量化 config 不应越过这条边界自行选择 attention 实现。

## 3. Create / load：先建立正确参数形状，再允许字节进入

### 3.1 layer 构造就是 ABI 实例化

`LinearBase` 在构造时把 `quant_config + prefix` 解析为一个 `quant_method`；没有 quant config 才使用普通 linear，而已提供 quant config 却无法为 linear 返回 method 会直接失败（`vllm/model_executor/layers/linear.py:242-276`）。具体 linear 随后把全局 input/output、当前 rank 的 partition sizes、parameter dtype 与自己的 weight loader 一起交给 `create_weights()`，forward 只调用已经绑定的方法（`vllm/model_executor/layers/linear.py:327-366`；`vllm/model_executor/layers/linear.py:392-403`）。

以 AutoGPTQ 为承重例而不是方法目录：它先用全局/局部 shape、quant type、activation dtype、group size、zero-point 与 `g_idx` 选择 Kernel，再创建 packed `qweight`、`scales`、`qzeros` 与 activation-order 参数；scale 在 row-parallel 下是复制还是分片也在这里决定（`vllm/model_executor/layers/quantization/auto_gptq.py:326-378`；`vllm/model_executor/layers/quantization/auto_gptq.py:380-445`）。所以 checkpoint tensor 能否被 copy 的前提，是参数容器已经编码了最终逻辑，而不是 loader 看到 tensor 后临时猜 pack axis。

通用加载边界只有一跳：default loader 把 tensor stream 交给 `model.load_weights()`；量化页不重复解释文件枚举、名称映射和 TP slice（`vllm/model_executor/model_loader/default_loader.py:414-445`）。本页关心的是 stream 写入量化方法创建的容器之后，容器仍只是“加载表示”，未必是“可执行表示”。

### 3.2 预量化与在线量化共享 ABI，但支付不同加载成本

| 路径 | 构造时的参数 | checkpoint 写入 | post-load 的职责 | 成本边界 |
|---|---|---|---|---|
| 预量化 | 直接建立 packed weight、scale、zero/metadata | 写入 checkpoint 已有的量化状态 | 标准化 bit order/axis，再做 Kernel repack | 文件更小，但 checkpoint format 必须精确匹配 |
| 在线量化 | 用 meta 参数描述 FP16/BF16 原始 weight | 按 layer materialize 原始 weight | 计算全局一致 scale、量化并替换参数、再做 Kernel repack | 不需预量化 checkpoint，但加载期有量化计算与瞬时双表示 |

在线 FP8 method 用 `uses_meta_device=True`，构造时只创建 meta weight 并注册 layerwise processing；真正 weight 在加载到该 layer 时 materialize（`vllm/model_executor/layers/quantization/online/fp8.py:114-155`）。per-tensor 实现随后从 weight 的 amax 计算 scale，必要时跨 TP 做 MAX all-reduce 以复现未分片 scale，生成 FP8 weight 并替换 weight/scale 参数（`vllm/model_executor/layers/quantization/utils/quant_utils.py:35-54`；`vllm/model_executor/layers/quantization/online/fp8.py:158-225`）。

这里 scale 不是附属 metadata，而是分布式数值状态。测试把同一全局 weight 沿 output/input 两个方向切 shard，要求在线量化后的局部 FP8 值与全局结果对应 slice 完全相等，且该共享的 scale 完全一致（`tests/quantization/test_online.py:301-368`）。**分析推断**：若省掉需要的 MAX collective，各 rank 仍会得到 shape 正确的 tensor，却使用不同量化格点；这类错误比 shape mismatch 更危险，因为它未必崩溃。

## 4. Post-load：把“已写入”提交为“可执行”

### 4.1 正常 loader 固定顺序，内部转换由 method 拥有

`BaseModelLoader.load_model()` 的正常顺序是 initialize → 通用 load → 在线 layerwise finalize（若需要）→ 全模型 quant post-load → `eval()`；源码注释明确把最后两步称为把 weight 处理成 Kernel format（`vllm/model_executor/model_loader/base_loader.py:42-82`）。这是 loader 在返回模型前提供的顺序，不是基类在 `apply()` 中执行的统一 guard：无需转换的 method 可以沿用默认 no-op，需要转换的 method 则依赖这条正常加载路径提交 Kernel 表示（`vllm/model_executor/layers/quantization/base_config.py:30-72`）。统一遍历对每个 `QuantizeMethodBase` 在目标设备上下文中调用 post-load；若转换替换了 Parameter，还会重新对齐 layer 的 TP rank/size（`vllm/model_executor/model_loader/utils.py:97-123`）。

CPU offload 不改变这个合同：post-load 会暂时把该 module 的 CPU 参数移到 target device，转换后再恢复原设备（`vllm/model_executor/model_loader/utils.py:154-189`）。它解决 GPU-only repack 无法处理 CPU tensor 的正确性问题，却会在加载期暂时占用设备内存；offload 不能被理解成 post-load 零显存成本。

AWQ 展示了为什么“checkpoint 已经量化”仍需要 post-load：checkpoint 沿 output dimension pack 且 bit order 非标准；method 先转成 GPTQ-like 的 input-packed 标准表示，再交给已选择的 Kernel 做自己的处理（`vllm/model_executor/layers/quantization/auto_awq.py:451-511`）。AutoGPTQ 的 method 则把 post-load 与 apply 都委托给同一个 Kernel 实例，确保产生布局的对象也是消费布局的对象（`vllm/model_executor/layers/quantization/auto_gptq.py:442-464`）。

### 4.2 在线量化为什么按 layer 提交

以下是从 meta/layerwise 设计重建的权衡（**分析推断**）：量化整个 BF16 模型后再释放原权重会把两份全模型表示同时留在峰值；在线 method 因而在 meta graph 上按 layer materialize、load、quantize/repack，再释放中间状态。基类把 `uses_meta_device` 的明确目的写成降低加载峰值（`vllm/model_executor/layers/quantization/base_config.py:23-28`）；回归测试的容量边界也承认单层转换期间 BF16 与 FP8 会短暂同时存活（`tests/quantization/test_fp8.py:236-250`）。这是一种把峰值从“全模型双份”压缩到“局部双份”的时间/内存交换，不是免费转换。

提交条件必须覆盖 layer 的全部可加载状态，而不只是 weight。reload 测试固定了一个曾经的失败边界：在线量化 layer 的 bias 比 meta weight 晚注册、晚加载时，不得提前 post-process；只有 bias 到达后才允许转换，否则会把尾随 bias 写进已经重排的 layer（`tests/model_executor/model_loader/test_reload.py:683-732`）。因此本阶段的不变量是：**所有参与数值或 Kernel 参数布局的输入都已到达，且 post-load 只把完整加载表示提交一次。**

## 5. Runtime dispatch / fallback：只能换实现，不能换语义

### 5.1 dispatch 选择的是“能实现该 ABI 的最优候选”

混合精度 linear 的候选按平台和预期性能排序；CUDA、ROCm、XPU、CPU 各有不同列表（`vllm/model_executor/kernels/linear/__init__.py:478-506`）。选择器依次应用 backend filter、禁用列表、compute capability 与 `can_implement(config)`，第一个全部通过的候选获选；全部失败时汇总原因并抛错（`vllm/model_executor/kernels/linear/__init__.py:780-842`）。scaled-mm 路径采用相同结构，并把平台支持与 shape/config 实现能力分开检查（`vllm/model_executor/kernels/linear/__init__.py:582-668`）。

`can_implement` 的粒度证明“同位宽”远远不够。Marlin 同时检查 CUDA、quant type、group size、activation order、TP-local K 是否被 group size 整除，并只对可修复的 tile misalignment 允许 padding（`vllm/model_executor/kernels/linear/mixed_precision/marlin.py:35-84`）；Exllama 还要求 activation 为 FP16、output 能按 pack factor 对齐，并拒绝 row-parallel 下的 activation reordering（`vllm/model_executor/kernels/linear/mixed_precision/exllama.py:18-75`）。这些是正确性谓词；候选顺序才是性能策略。

### 5.2 fallback 有四种语义，不能都叫“降级”

| 情况 | 行为 | 是否保持原量化数值合同 | 主要代价 / 证据 |
|---|---|---|---|
| 配置明确 ignore 某层 | 返回 unquantized method | 这是 config 声明的混合精度合同，不是事故 fallback | 多占权重内存；在线 config 对 linear/MoE 都显式这样处理（`vllm/model_executor/layers/quantization/online/base.py:157-181`） |
| 首选优化实现不兼容该 layer | 选择同格式的次级 method/kernel | 必须保持 weight/scale/zero 语义 | AutoAWQ 在 Marlin shape 不兼容时回到未优化 AWQ；MoE 回到 WNA16（`vllm/model_executor/layers/quantization/auto_awq.py:285-357`） |
| 候选 Kernel 不兼容 | 尝试优先级列表中的下一项 | 保持同一个 `MPLinearLayerConfig` | 可能降低吞吐；ROCm 测试固定 RDNA3 → Hybrid → Triton 的选择次序（`tests/kernels/quantization/test_w4a16_kernel_selection.py:24-71`） |
| 所有 Kernel 都不兼容 | 构造阶段硬失败 | 不允许悄悄换 scale、dtype 或 pack | 选择器报告逐候选失败原因（`vllm/model_executor/kernels/linear/__init__.py:813-842`） |

还有一种 correctness-first 的执行 fallback：启用 batch-invariant 模式时，在线 per-tensor FP8 若不是已知可保持该合同的 Cutlass 路径，会把 FP8 weight 按 scale 还原到 BF16 再做普通 linear（`vllm/model_executor/layers/quantization/online/fp8.py:227-254`）。它保留的是确定性/数值执行合同，代价是失去低精度 GEMM 的带宽与 Tensor Core 收益；这里只记录派发接缝，batch invariance 的完整定义仍由训练可靠性页面拥有。

> [!note] 分析推断
> 安全 fallback 的判据不是“结果看起来合理”，而是候选消费与当前 layer 完全相同的 pack、scale、zero-point、activation 与 TP-local shape。若 fallback 需要重新解释其中任一项，它就不是 runtime fallback，而是一条必须在 post-load 之前显式建模的新 ABI。

## 6. 正确性、性能与失败边界必须一起验收

| 压力 | 正确性不变量 | 为它支付的成本 | 失败边界 |
|---|---|---|---|
| fused QKV / gate-up | constituent shards 的 quant/ignore scheme 一致 | config 必须消费模型 mapping | 部分 shard 命中直接报错（`vllm/model_executor/layers/quantization/utils/quant_utils.py:639-668`） |
| TP scale | 需要全局统计的 scale 与未分片量化等价 | 某些 scheme 增加 MAX collective | 测试要求 FP8 值和 scale 精确相等（`tests/quantization/test_online.py:339-368`） |
| checkpoint pack ≠ Kernel pack | 对需要转换的 method，正常 loader 返回前提交 Kernel 表示 | 加载时 repack、padding、临时 buffer | AWQ 明确先转标准格式再交给 Kernel；基类不提供统一 guard（`vllm/model_executor/model_loader/base_loader.py:75-82`；`vllm/model_executor/layers/quantization/auto_awq.py:503-519`；`vllm/model_executor/layers/quantization/base_config.py:30-72`） |
| post-load 替换 Parameter | 新参数继承 layer 的 TP rank/size | 多一次 metadata reconciliation | loader 在转换后显式重写 TP state（`vllm/model_executor/layers/linear.py:291-304`；`vllm/model_executor/model_loader/utils.py:113-120`） |
| 在线量化峰值 | 同一 layer 的原权重全部到达后再量化 | 单层 BF16+低精度短时共存、加载计算 | late bias 回归测试阻止提前提交（`tests/model_executor/model_loader/test_reload.py:718-732`） |
| Kernel specialization | capability 与 shape 谓词全部成立 | 更多 provider、测试与 fallback 维护 | 没有兼容候选就硬失败，而不是静默换格式（`vllm/model_executor/kernels/linear/__init__.py:780-842`） |

推荐按 ABI 边界诊断，而不是按方法名排查：

1. **Configure**：核对 checkpoint `quant_method`、用户 override、在线 config 与最终 config class；先关闭身份冲突。
2. **Create/load**：检查目标 layer 的实际 `quant_method`、packed constituent 一致性、TP-local weight/scale/zero shape；“文件读到了”不代表参数 ABI 正确。
3. **Post-load**：确认在线量化或 repack 已完成、替换参数的 TP metadata 已恢复、无 meta/中间表示进入执行。
4. **Dispatch**：记录实际 Kernel 与被拒候选原因；分开验证 correctness fallback 和性能 fallback。
5. **数值与性能**：对固定输入比较高精度基线或未分片基线，再分别测 prefill/decode；不要用“模型能生成文本”替代 scale/pack 正确性。

本页刻意不提供“方法支持列表”：注册表本身已经包含 checkpoint methods、在线 shorthand 与 deprecated methods，且它们会随平台和 provider 变化（`vllm/model_executor/layers/quantization/__init__.py:12-52`；`vllm/model_executor/layers/quantization/__init__.py:108-180`）。稳定的知识不是名字数量，而是每个新增格式都必须穿过同一组 ABI 提交点和兼容谓词。

## Related Pages

- [[02_engineering/03_infer_frameworks/vllm/13_vllm_model_library_analysis|vLLM 模型与权重 ABI]] — 拥有通用 checkpoint tensor、名称映射、packed shard identity 与 TP 参数写入；本页从量化参数容器接手。
- [[02_engineering/03_infer_frameworks/vllm/24_vllm_fused_ops_and_kernels_analysis|vLLM 融合算子与 Kernel]] — 展开 provider、Kernel 内部优化与收益模型；本页只拥有量化兼容谓词和派发结果。
- [[02_engineering/03_infer_frameworks/vllm/22_vllm_distributed_inference_analysis|vLLM 分布式推理]] — 解释 TP/EP rank 与 collective 所有权；本页只证明 scale/pack 在这些分片下仍等价。
- [[02_engineering/03_infer_frameworks/vllm/14_vllm_attention_backends_analysis|vLLM Attention Backend]] — KV dtype、scale 与 attention backend 的完整能力协商在此展开。
- [[02_engineering/07_training_reliability/20_batch_invariance_guide|Batch Invariance]] — 定义量化派发为何有时必须放弃最快 Kernel 来守住批次不变性。
