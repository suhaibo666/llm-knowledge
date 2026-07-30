# F02 · Activation Checkpoint、AOT Recompute 与编译

> 卷别：F · 训练、分布式、扩展与部署  
> 固定源码：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`  
> 前置：[[20_compiled_autograd_analysis]]  
> 后续：[[ddp_compile_boundaries_and_optimizer_analysis]]  
> 最后更新：2026-07-30(kb-reorg P4 Task 9 迁入本目录,与 [[12_saved_tensors_recompute_and_runtime_abi_analysis]] 互指划界)

> [!note] 与 [[12_saved_tensors_recompute_and_runtime_abi_analysis]] 的分工
> 两页都讲"save vs recompute",但站在不同层：本页站在**用户 API 与策略**层——`torch.utils.checkpoint` 的 reentrant/non-reentrant 语义、Selective AC 的 `CheckpointPolicy`、RNG/device 契约,以及这些用户意图如何经 `node.meta["recompute"]` 影响 partitioner；[[12_saved_tensors_recompute_and_runtime_abi_analysis]] 站在**partitioner 源码与 runtime ABI**层——min-cut flow network 的拆点边语义、`default_partition`/`solve_min_cut` 的具体判断、fw/bw 之间 saved tensor 的真实 ABI 拼接与生成代码。理解顺序:先读本页知道"为什么/何时"发生重算,再读该页知道"partitioner 具体怎么切、runtime 怎么拼"。

## 1. 两种“重算”必须先分开

### 用户级 activation checkpoint

用户把函数包进`torch.utils.checkpoint`。forward不保存部分activation，backward时重新调用
该函数以重建saved tensors。

### AOTAutograd partition rematerialization

AOT已有joint fw/bw graph，partitioner在cut上决定哪些forward值作为fw额外输出保存，哪些
producer node复制进bw graph重算。

二者都用计算换内存，却发生在不同抽象层：

```text
用户 checkpoint region
→ trace 中的 recompute policy/meta
→ AOT joint graph
→ min-cut saved-vs-recompute
→ bw graph 中的复制节点
```

叠加时，用户policy会约束/引导partitioner，而不是简单“重算两遍”。

## 2. Reentrant 与 non-reentrant checkpoint

公开入口说明checkpoint通过在backward重新调用forward segment减少saved activations，并推荐
`use_reentrant=False`
（`torch/utils/checkpoint.py:355-383`）。

关键差别：

- non-reentrant会记录autograd graph，可在所有需要的saved tensor重建后early-stop；
- reentrant forward运行在`no_grad`下，backward重跑完整函数；
- 二者对backward API、嵌套结构、detached tensor和requires-grad有不同限制；
- 显式传 `use_reentrant`，避免版本默认变化。

差异说明见 `torch/utils/checkpoint.py:391-420` 与
`torch/utils/checkpoint.py:421-450`；入口分派见
`torch/utils/checkpoint.py:486-511`。

对`torch.compile`，non-reentrant通常更容易与图捕获和Selective AC组合，但必须依据目标版本
验证。

## 3. Non-reentrant 的 saved tensor 机制

它利用saved tensor hooks：

1. 原forward遇到save时只保留holder/metadata；
2. backward unpack holder时，若本graph task尚未recompute，则调用`recompute_fn`；
3. recompute期间再次save的Tensor按位置写入`recomputed[graph_task_id]`；
4. 检查数量和metadata；
5. 返回对应重建Tensor；
6. 满足所有需要的Tensor后可early-stop。

源码注释明确给出嵌套checkpoint、lifetime和early-stop规则
（`torch/utils/checkpoint.py:654-683` 与
`torch/utils/checkpoint.py:684-699` 与 `torch/utils/checkpoint.py:700-715`）。

unpack触发recompute与metadata检查见
`torch/utils/checkpoint.py:1158-1187` 与
`torch/utils/checkpoint.py:1188-1210`。

## 4. RNG、device 与外部状态

重算必须与原forward产生语义等价的中间值。checkpoint可保存/恢复RNG和device/autocast
context，但以下行为危险：

- forward与recompute走不同控制流；
- 依赖未保存的global mutable state；
- 在region中移动到新的device；
- 不可重放I/O、collective、hook；
- in-place影响region外对象；
- 随机算子未保持对应RNG state。

文档警告若recompute与原forward调用不同，可能错误或抛异常；这是checkpoint契约，不是
编译器可自动修复的问题
（`torch/utils/checkpoint.py:372-390`）。

## 5. Selective Activation Checkpoint

Selective AC通过`CheckpointPolicy`逐op决定：

- `MUST_SAVE` / `PREFER_SAVE`；
- `MUST_RECOMPUTE` / `PREFER_RECOMPUTE`。

policy context区分原forward与recompute；该对象的语义见
`torch/utils/checkpoint.py:1268-1283` 与 `torch/utils/checkpoint.py:1284-1299`。

Caching TorchDispatchMode在forward执行op并缓存允许保存的output；recompute mode按同一key
取回或重新执行。编译时policy也写入FX node的`meta["recompute"]`
（`torch/utils/checkpoint.py:1360-1382` 与
`torch/utils/checkpoint.py:1383-1405`）。

所以进入AOT图后，checkpoint意图以node metadata存在，partitioner能读取。

## 6. AOT partitioner 如何生成“带重算的反向图”

对joint graph中的forward node：

- 若被选为saved value，则它成为fw额外输出和bw placeholder；
- 若标为recompute且其值未跨cut保存，producer node被复制到bw图；
- bw中的梯度node消费这些重算结果；
- fw/bw之间仍没有直接FX边，ABI由fw output与bw placeholder位置连接。

partition后会提取fw/bw GraphModule，并对两图DCE；若有可重算RNG op还会functionalize RNG，
再重排bw以模拟autograd engine
（`torch/_functorch/partitioners.py:1782-1805`）。

这正是“recompute怎样加入反向图”的具体答案：**partition extraction在创建fresh bw nodes时
复制被选择的forward子图**，而不是运行时从fw图跳一条边回去。

## 7. 哪些东西不能随意重算

partitioner默认强制保存：

- 未被用户显式AC标记的collective输出；
- effectful op包装后的Tensor输出；
- forward/backward对同一primal mutation时的mutation source。

collective与effect约束见 `torch/_functorch/partitioners.py:2309-2335`；mutation source约束见
`torch/_functorch/partitioners.py:2354-2368`。

原因是重算可能重复通信、I/O、RNG或mutation，破坏effect次数与rank顺序。

## 8. Min-cut 的 save/recompute 模型

partitioner把joint graph转为flow network：

- 需要在backward可用的值连接sink；
- 禁止重算的node以无限容量连接source；
- 必须重算的node以无限容量连接sink侧；
- 保存某node的代价与Tensor大小、materialization和启发式有关；
- cut决定saved values。

`should_ban_recomputation`会拒绝MUST_SAVE、随机、compute-intensive、非allowlist或在backward
必须materialize等node
（`torch/_functorch/partitioners.py:2536-2552` 与
`torch/_functorch/partitioners.py:2553-2569`）。

node weight从估算大小出发，static-lifetime input可视为零保存成本，view可被偏向重算
（`torch/_functorch/partitioners.py:2601-2628` 与
`torch/_functorch/partitioners.py:2630-2639`）。

must-recompute通过无限容量边强制留在sink侧
（`torch/_functorch/partitioners.py:2676-2692` 与
`torch/_functorch/partitioners.py:2693-2708`）。

## 9. 反向图的可视形态

```text
fw:
  primals → a → b → output
              └──── saved: a

bw:
  placeholder(a), tangent
  b_recomputed = f(a)
  grad = backward_formula(b_recomputed, tangent)
```

若保存`b`，则bw直接有`placeholder(b)`，没有`f(a)`的复制节点。实际ABI还可能包含symints、
tokens、opaque objects和mutation更新值，不能只按Tensor列表理解。

## 10. 与 `torch.compile` 的交互

- Dynamo需要捕获checkpoint/HOP或允许边界；
- AOT读取recompute metadata并partition；
- Inductor分别编译fw/bw，包括bw中的重算node；
- backward可能lazy compile；
- Compiled Autograd还可能把局部AOT bw纳入更大反向图；
- CUDAGraph对动态shape、地址和mutation另有约束。

定位时必须标明问题发生在用户checkpoint replay、AOT partition选择、bw codegen还是CA
capture。

## 11. 成本模型

设保存activation总字节 \(M_s\)，重算FLOPs \(F_r\)，额外kernel/通信成本 \(T_r\)：

\[
M_{\text{peak}} \downarrow \quad\text{以}\quad
T_{\text{step}} \uparrow \approx T_r
\]

但峰值还取决于liveness、allocator、saved tensor释放时点与CUDAGraph pool。重算同一subgraph
也可能被fusion，使额外运行成本并非原forward片段的简单倍数。

## 12. 常见误解

- **“checkpoint和AOT rematerialization是同一个API。”** 前者是用户运行语义，后者是图cut。
- **“bw图通过边连回fw图执行。”** 两张FX图由saved ABI连接；重算node被复制到bw。
- **“所有便宜op都可重算。”** effect、collective、mutation和RNG有额外约束。
- **“reentrant与non-reentrant只差性能。”** autograd graph、API和嵌套语义不同。
- **“显存减少量等于少保存的Tensor字节。”** liveness、allocator和workspace也决定峰值。

## 配套 Demo

本页对应卷级入口 `tools/labs_torch_compile/demo_f_advanced_topics.py` 的 `checkpoint_recompute` 用例。默认以 CUDA 为验收设备：

```powershell
python -B tools\labs_torch_compile\demo_f_advanced_topics.py `
  --case checkpoint_recompute --device cuda `
  --output-dir tools\labs_torch_compile\artifacts\volume_demos\f02
```

先用 `--list --json` 查看用例声明的能力要求。无 CUDA 的机器可把 `--device` 改为 `cpu` 探索设备无关机制；CUDA/Triton/多卡专属用例会返回 `BLOCKED`，且不会执行用例正文。不要把 `BLOCKED` 写成 `PASS`。

重点读取 `summary.json` 与 `checkpoint_recompute/result.json`：`status` 区分 `PASS/BLOCKED/FAIL`，`environment` 固化运行环境，`observations` 保存本页机制的实测字段，`artifacts` 指向图代码、日志、trace 或进程证据。`PASS` 只表示该次运行中的断言通过，不外推到其他 PyTorch 版本、shape、dtype 或硬件。

## Related Pages

- [[00_torch_compile_end_to_end_index]]
- [[02_compile_stack/02_aot_autograd/index]] — 本模块 overview
- [[20_compiled_autograd_analysis]]
- [[ddp_compile_boundaries_and_optimizer_analysis]]
- [[12_saved_tensors_recompute_and_runtime_abi_analysis]] — partitioner 源码与 runtime ABI 层深析(本页用户 API/策略层的下游对应物,见页头分工声明)
- [[11_aotautograd_joint_forward_backward_graphs_analysis]]
- [[01_theory/02_pretraining/activation_checkpointing_analysis]]
