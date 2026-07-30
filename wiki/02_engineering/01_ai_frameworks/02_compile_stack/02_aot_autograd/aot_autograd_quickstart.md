# AOTAutograd Quick Start

> **页面角色**：AOTAutograd API quick start；示例是否在当前环境运行须以各代码块标注为准。
> **原始基线**：见下方页头；**当前审计基线**：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`。
> **审计状态**：已纳入 Batch 0，尚未逐代码块全部复跑；fw/bw 与 save/recompute 的已验证课程主线见 [[19_torch_compile_end_to_end/09_aotautograd_joint_forward_backward_graphs]] 和 [[19_torch_compile_end_to_end/10_saved_tensors_recompute_and_runtime_abi]]。

> 层次:quick start · 核验基准:PyTorch 上游(`E:\97-codes\pytorch\pytorch`) · 最后更新 2026-06-13
>
> 本页所有 API / backend / 日志键 / config 均对照真实源码核实,标注 `path:line`。深入原理见文末导航。

## 1. 快速导航
> 「AOTAutograd 是什么、在栈中的位置、三大职责(functionalization / joint graph / partition)」见 [[index]] 模块概述。本页聚焦上手:下面给出主要入口、核心概念与常用配置的索引,细节落在后续小节。

- **主要入口**:`aot_function` / `aot_module`(均在 `torch/_functorch/aot_autograd.py`)。走 `torch.compile` 时通常不直接调用,用 `backend="aot_eager"` 即可触发 AOTAutograd 全流程(见 §2);`aot_function` 的 docstring 写明它 ahead-of-time trace 前向 + 反向、生成联合图,再由 `partition_fn` 切分(`aot_autograd.py:723-728`)。
- **核心概念**:partitioner 有 `min_cut_rematerialization_partition`(默认,以重计算换显存)与 `default_partition`(多 save、少重算)两种(§3);看图用 `TORCH_LOGS` 的 `aot_graphs`(切分后的前/反向图)与 `aot_joint_graph`(切分前的联合图)(§2)。
- **常用配置**:`activation_memory_budget`(激活显存预算,调节 save vs 重算)、`AOT_PARTITIONER_DEBUG=1`(打印 partitioner 的切分决策),详见 §3。

下面从「看前/反向图最快的方式」开始。

## 2. 看前/反向图最快的方式

不必写任何 functorch 代码,直接用 Dynamo 的 `aot_eager` 后端(AOTAutograd + nop 编译器,纯调试用)配合 `TORCH_LOGS`:

```python
import torch

@torch.compile(backend="aot_eager")   # AOTAutograd 全流程,但不进 Inductor
def f(x, w):
    return torch.relu(x @ w).sum()

x = torch.randn(64, 128, requires_grad=True)
w = torch.randn(128, 256, requires_grad=True)
f(x, w).backward()
```

```bash
# 切分后的前向 + 反向图(给 Inductor 的就是它)
TORCH_LOGS="aot_graphs" python script.py
# 切分之前的联合图(调试 partitioner 用)
TORCH_LOGS="aot_joint_graph" python script.py
# 二者一起 + AOTAutograd 全量日志
TORCH_LOGS="aot,aot_joint_graph,aot_graphs" python script.py
```

`backend="aot_eager"` 真实存在,注册于 `torch/_dynamo/backends/debugging.py:428`(实现 `aot_eager` 函数 `:413-425`,用 `boxed_nop` 作前/反向编译器)。

`TORCH_LOGS` 键(全部核实于 `torch/_logging/_registrations.py`):

| 键 | 含义 | 来源 |
|----|------|------|
| `aot` | AOTAutograd 模块全量日志(`torch._functorch.aot_autograd` / `_aot_autograd`) | `_registrations.py:27` |
| `aot_graphs` | **切分后**的前向 + 反向 FX 图(默认 visible) | `_registrations.py:99-103` |
| `aot_joint_graph` | **切分前**的联合图,调试 partitioning 用 | `_registrations.py:104-107` |
| `aot_graphs_effects` | 含 effects 处理的前/反向图 | `_registrations.py:108-112` |

## 3. Partitioner:min-cut vs default
AOTAutograd 自带两个 partitioner,都在 `torch/_functorch/partitioners.py`:

| partitioner | 策略 | 来源 |
|-------------|------|------|
| `default_partition` | 贴近 eager:**前向算出的中间张量直接 save 给反向**,反向几乎不重算 | `partitioners.py:1248` |
| `min_cut_rematerialization_partition` | 在「save 张量(吃显存)」与「反向重计算(吃算力)」间求**最小割**,常用更少显存换带宽 | `partitioners.py:3726` |

**谁是默认?**

- `torch.compile`(Inductor 后端):默认用 **min-cut**(`torch/_inductor/compile_fx.py:30` 导入,`:2311-2313` 调用)。
- `backend="aot_eager"`:也用 **min-cut**(`debugging.py:423`)。
- 想看「不重计算」的 default 切法 → 用 `backend="aot_eager_default_partitioner"`(注册于 `debugging.py:433-435`,实现 `:430-432`,不传 `partition_fn` 即落到 `aot_function` 默认的 `default_partition`,见 `aot_autograd.py:706`)。
- 想用 Inductor 的 decomp + min-cut 但绕开 Inductor codegen → `backend="aot_eager_decomp_partition"`(`debugging.py:473-474`,实现 `:442-470`)。

**对比两种切分(最快验证)**

```bash
# min-cut(默认):反向图里会看到被重算的算子
TORCH_LOGS="aot_graphs" python script.py
# 改成 default:前向 save 更多张量,反向几乎不重算
# 把 @torch.compile(backend="aot_eager") 换成 backend="aot_eager_default_partitioner"
```
**观察重计算 / 调 min-cut**(config 在 `torch/_functorch/config.py`):

| 开关 | 默认 | 作用 | 来源 |
|------|------|------|------|
| `AOT_PARTITIONER_DEBUG=1`(env)→ `debug_partitioner` | `False` | 打印 partitioner 选择的 save/recompute 决策与节点统计 | `config.py:44` |
| `activation_memory_budget` | `1.0` | `0.0`=对整段做 activation checkpoint(最省显存、最多重算);`1.0`=纯 runtime 最优;`0.4`=只保留 40% 激活 | `config.py:202` |
| `activation_memory_budget_runtime_estimator` | `"flops"` | budget 下估算重算代价:`flops` / `profile` / `testing` | `config.py:209` |
| `activation_memory_budget_solver` | `"dp"` | 0-1 背包求解器:`dp` / `greedy` / `ilp` / `dp_knapsack_sliding_hirschberg` | `config.py:215` |
| `recompute_views` | `False` | view 是否总是重算(view 重算成本低,常设 True) | `config.py:175` |
```python
import torch._functorch.config as fc
fc.activation_memory_budget = 0.5   # 只保留约 50% 激活,其余反向重算
```

## 4. 直接用 functorch(脱离 torch.compile)

公共入口在 `functorch.compile`(`functorch/compile/__init__.py` 导出 `aot_function`、`aot_module`、`aot_module_simplified`、`nop`、`default_partition`、`min_cut_rematerialization_partition`)。

**最小示例**——打印联合切分后的前/反向图:

```python
import torch
from functorch.compile import aot_function

def fn(x):
    return x.sin().cos()

def print_compiler(fx_module, args):   # fw/bw 编译器:拿到 FX 图,返回可调用
    print(fx_module.code)
    return fx_module                    # 等价于 functorch.compile.nop

# bw_compiler 省略时回落到 fw_compiler(见签名默认值)
aot_fn = aot_function(fn, fw_compiler=print_compiler, bw_compiler=print_compiler)

x = torch.randn(4, 5, requires_grad=True)
aot_fn(x).sum().backward()             # 触发 trace + 切分 + 编译,打印前向、再打印反向
```

核实的真实签名(`torch/_functorch/aot_autograd.py`):

```python
# aot_autograd.py:702
def aot_function(fn, fw_compiler, bw_compiler=None,
                 partition_fn=default_partition,          # 默认 default,非 min-cut
                 decompositions=None, num_params_buffers=0,
                 keep_inference_input_mutations=False, inference_compiler=None,
                 *, dynamic=False, enable_log=True, ...): ...

# aot_autograd.py:837 —— 包装 nn.Module,内部转调 aot_function
def aot_module(mod, *args, **kwargs): ...

# aot_autograd.py:1538 —— 导出用,返回 (GraphModule, GraphSignature)
def aot_export_module(mod, args, *, decompositions=None, trace_joint,
                      output_loss_index=None, pre_dispatch=False,
                      dynamic_shapes=None, kwargs=None): ...
```

注意:`aot_export_module` 不在 `functorch.compile` 里,规范导入是 `from torch._functorch.aot_autograd import aot_export_module`(用例见 `torch/export/exported_program.py:340`);`trace_joint=True` 时返回联合前/反向图(`aot_autograd.py:1559`)。换 partitioner 直接传 `partition_fn=min_cut_rematerialization_partition`。

## 5. 常见现象速解
- **反向图里冒出 sin/relu 之类「本该在前向」的算子?** 这是 min-cut 的**重计算**:partitioner 判断「重算它」比「在前向存下它并跨边界传递」更省显存/带宽,于是没把它列入 saved tensors,而在反向就地重算。
- **saved tensors 与切分的关系**:前向图的额外输出 = 要 save 给反向的中间张量;default partitioner 倾向多 save(`partitioners.py:1263-1267`「stash 中间张量作为前向输出」),min-cut 在 save 与 recompute 间求最小割(`partitioners.py:3735-3736`「backward recomputes the forward,以带宽换显存」)。saved 越少 → 反向重算越多、峰值显存越低。
- **想让某类算子别被重算 / 多重算**:用 §3 的 `activation_memory_budget`(全局调档)或 `AOT_PARTITIONER_DEBUG=1` 先看清当前决策。
- **图里 in-place 操作不见了?** 已被 functionalization 改写成函数式 ATen(§1),属正常。

## 6. 深入阅读导航

- 原理深挖:[[aotautograd_analysis]](deep dive)— 五阶段全流程、数据结构、性能策略
- 联合图 passes:[[joint_graph_passes_guide]]
- 上游(图捕获):[[02_compile_stack/01_dynamo/index]]
- 下游(代码生成):[[02_compile_stack/04_inductor/index]]

## Related Pages

- [[19_torch_compile_end_to_end/00_pytorch_graph_series_index]] — 当前固定基线的图编译系统化课程入口
- [[01_ai_frameworks/index]]
- [[aotautograd_analysis]]
- [[joint_graph_passes_guide]]
- [[02_compile_stack/01_dynamo/index]]
- [[02_compile_stack/04_inductor/index]]
