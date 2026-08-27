# TorchTitan Trainer Quickstart：从 Python 配方到一次可恢复训练步

> **代码基准**：pytorch/torchtitan `main` @ `a3168782c9a3a2e40afbd0de114818b96e2bda6e`（2026-08-27）
> **维度**：Quickstart（入口与运行时主链）
> **最后更新**：2026-08-27 · **系列**：[[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]]
>
> **主线**：当前 TorchTitan 的推荐入口不再是“堆一串 CLI flag”，而是让一个 Python 函数返回完整 `Trainer.Config`。launcher 只负责选择配方与启动进程；`Configurable.build()` 把配方展开成模型、优化器、数据、checkpoint、validator 等组件，`Trainer` 再以固定生命周期组装它们。先掌握这条主链，再进入 TP/FSDP/GraphTrainer 等专项页。

---

## 1. Overview

### 1.1 最小启动命令

仓库自带的配置说明给出当前推荐调用方式（`torchtitan/config/README.md:3-11`）：

```bash
NGPU=4 \
MODULE=torchtitan_recipes.tests.features \
CONFIG=llama3_debugmodel_fsdp2_cp2 \
./run_train.sh
```

`run_train.sh` 把 `MODULE`/`CONFIG` 翻译成 `python -m torchtitan.train --module ... --config ...`；真实训练走 `torchrun`，`COMM_MODE=fake_backend` 则单进程模拟任意 world size，只跑一步做配置/模型校验（`run_train.sh:21-45`）。

> [!important] `NGPU` 不属于配方
> 并行度在 `Trainer.Config` 里，world size 由 launcher 环境决定；二者必须满足 [[10_torchtitan_parallel_dims_analysis]] 的乘积约束。配置文档明确要求 `NGPU` 与并行度乘积一致（`torchtitan/config/README.md:8-11`）。

### 1.2 入口文件地图

| 问题 | 从这里读 | 关键定位 |
|---|---|---|
| 如何选配方？ | `ConfigManager` | `torchtitan/config/manager.py:19-48` |
| 如何启动 Trainer？ | `train.main()` | `torchtitan/train.py:17-68` |
| 配置如何构造组件？ | `Configurable.Config.build()` | `torchtitan/config/configurable.py:134-179` |
| 模型/并行/组件如何组装？ | `Trainer.__init__()` | `torchtitan/trainer.py:274-628` |
| 一个优化步做什么？ | `Trainer.train_step()` | `torchtitan/trainer.py:774-940` |
| checkpoint/验证何时触发？ | `Trainer.train()` | `torchtitan/trainer.py:943-1003` |

---

## 2. 配置演进：full configuration 是程序，不是参数表

### 2.1 `--module` + `--config`

一个 run 由返回完整 `Trainer.Config` 的 Python 函数描述；函数同时选择模型、并行度与优化组件（`torchtitan/config/README.md:3-19`）。`ConfigManager` 先从 argv 中取出 `--module`/`--config`，导入目标模块并调用函数，再让兼容 CLI override 覆盖返回的对象（`torchtitan/config/manager.py:34-48`、`torchtitan/config/manager.py:48-159`）。

这意味着下面两种变化的推荐归属不同：

- **换集群/并行布局/训练组合**：写新的 recipe 函数。
- **实现新的能力**：新增或扩展一个组件 Config，再由 recipe 选择它。

配置文档明确冻结 `--section.option` CLI 的扩张；旧 flag 仍能覆盖配方只是兼容承诺，长期目标是只保留 `--module`/`--config`（`torchtitan/config/README.md:54-68`）。

### 2.2 `torchtitan` 与 `torchtitan_recipes` 的边界

`torchtitan` 包含模型定义和优化实现；同级 `torchtitan_recipes` 只选择组件组合。源码把二者分开，是因为 recipe 与集群/单次 run 绑定，演进速度和库 API 不同（`torchtitan/config/README.md:24-32`）。

当前 per-model `config_registry.py` 仍兼容，但文档已经把它标成旧位置，计划逐步迁往 `torchtitan_recipes`，且不打算为每个旧配置做双名 shim（`torchtitan/config/README.md:70-80`）。因此，新知识与新脚本不应再把 `models/<name>/config_registry.py` 当唯一配置权威。

> [!deprecated] “模型目录的 config registry 是唯一入口”已过时
> 它仍可运行，但推荐的全配置归属是独立 `torchtitan_recipes` 或任意外部可导入模块。

---

## 3. Launcher 主链：薄入口、统一收尾

`torchtitan.train.main()` 的真实调用链很短（`torchtitan/train.py:17-68`）：

```text
init_logger
  -> ConfigManager.parse_args()
  -> structured_logger.init
  -> config.build()                 # 构造 Trainer
  -> seed-checkpoint ? save(step=0) : trainer.train()
  -> trainer.close()
  -> destroy_process_group()
```

`config.build()` 不是手写在每个 config 类里的工厂。`Configurable` 在子类定义时自动给 Config 接上 build：默认构造对应实现类，也允许自定义 builder；同时拒绝 build kwargs 与 config 字段重名，避免同一参数有两份权威（`torchtitan/config/configurable.py:134-179`）。

launcher 还把 seed checkpoint 明确成特殊模式：只能 world size 1、必须开启 checkpoint，然后直接保存 step 0，不进入训练循环（`torchtitan/train.py:43-58`）。

---

## 4. Trainer 初始化：顺序本身就是不变量

### 4.1 先设备/分布式，再构建模型

`Trainer.__init__()` 先绑定 local device，再初始化 distributed 与 `ParallelDims`（`torchtitan/trainer.py:274-305`）。这必须早于模型/组件构造，因为：

- 配置与模型 sharding 需要 world size 和命名 mesh。
- 后续随机种子按 PP 等 mesh 轴区分（`torchtitan/trainer.py:320-332`）。
- logger 与 fake/real process group 都依赖已完成的分布式初始化。

然后 `model_config.update_from_config()` 把 job-level 配置写入模型子配置（包括 SPMD sharding contract），override 在所有组件 build 前统一执行，CUDA Graph 组合限制也在这里提前验证（`torchtitan/trainer.py:334-350`）。

### 4.2 meta build → 并行化 → materialize

模型先在 meta device 上构建并验证 Module protocol（`torchtitan/trainer.py:351-365`），再按是否开启 PP 分成两条路径：

- **无 PP**：调用模型的 `parallelize_fn` 施加 SPMD/AC/compile/FSDP，然后 `to_empty` 到目标设备并初始化状态（`torchtitan/trainer.py:472-493`）。
- **有 PP**：pipelining 函数先切 model parts，并对每个 part 施加相同并行化链；随后逐 part materialize/初始化（`torchtitan/trainer.py:426-471`）。

这个顺序避免先在单卡物化完整模型再分片，也保证 sharding wrapper 在真实参数分配前就位。具体并行顺序分别见 [[10_torchtitan_parallel_dims_analysis]] 与各维度页面。

### 4.3 优化器必须后建，checkpoint 必须再后建

优化器在模型完成并行化后才 build，因为它要看到最终 sharded 参数；LR scheduler 随后绑定这些 optimizer（`torchtitan/trainer.py:533-543`）。

tokenizer/dataloader 再按 DP rank 构造；checkpoint manager 最后拿到 dataloader、model parts、optimizers、LR schedulers、Trainer state 与可选 state-dict adapter（`torchtitan/trainer.py:546-577`）。这个依赖顺序解释了为什么 checkpoint 不是一个孤立 I/O 工具：它保存的是已经组装好的训练状态图。

### 4.4 运行上下文与 CUDA Graph

Trainer 创建 `SpmdContext`，把 dense/sparse mesh 和可选全局 type checker 接到每次前后向；若未禁用 CUDA Graph，则用 `wrap_with_cuda_graph()` 替换 `_forward_backward_body`（`torchtitan/trainer.py:579-588`）。

这是一项重要演进：CUDA Graph 已进入核心 Trainer，不再只是 GraphTrainer 实验件。它只包稳定的 forward/backward body，数据搬运、optimizer、checkpoint 与 validation 仍在图外；详细约束应与 [[23_torchtitan_compute_memory_optimizations_analysis]] 一起阅读。

---

## 5. 一个训练步的真实时间线

### 5.1 先收齐本步 microbatch，再算全局有效 token

`train_step()` 先按 gradient accumulation × PP microbatch 收集输入，统计非 ignore label；然后在 DP batch mesh 上规约 `global_valid_tokens`，并把该计数留在 device 上避免每步 D2H 同步（`torchtitan/trainer.py:774-806`）。

之后才逐 accumulation group 把 tensor 搬到 device，选择 PP 或非 PP forward/backward 路径（`torchtitan/trainer.py:808-834`）。这意味着配置里的训练批量语义已经从“样本数”转向 token 数：初始化阶段用 `num_tokens_per_train_step` 推导 accumulation steps（`torchtitan/trainer.py:406-425`）。

### 5.2 前后向只在 `SpmdContext` 中执行

非 PP 路径先调用模型的 `preprocess_inputs()`，让模型构造 mask、准备 CP 输入并标注 SPMD 类型，再进入 `train_context` 执行模型与 loss（`torchtitan/trainer.py:688-715`）。PP 路径对每个 microbatch 做同样预处理，然后把整个 schedule step 放入相同 context（`torchtitan/trainer.py:736-762`）。

因此，数据层输出并不是直接喂给模型；模型拥有“如何把一个 batch 变成当前 attention/SPMD 后端所需输入”的最后解释权。

### 5.3 optimizer 前做全局有限性闸门

所有 microbatch 完成后才 clip grad；loss finite 在 loss mesh 与 PP mesh 上传播，再与 grad norm finite 合并。`torch._assert_async` 把检查留在 device 顺序中，失败发生在 optimizer update 之前（`torchtitan/trainer.py:850-889`）。

紧接着先等待 checkpoint staging，再执行 optimizer 与 LR scheduler step（`torchtitan/trainer.py:887-890`）。这条顺序保证后台 checkpoint staging 不会与下一次状态修改竞争。

### 5.4 日志只规约需要的量

当本步需要日志时，Trainer 在 loss mesh 上求 global average/max loss 与 token count；TP rank 不重复规约同一 loss 语义，PP loss 则由拥有最后 stage 的 rank 传播（`torchtitan/trainer.py:891-940`）。

---

## 6. 外层训练循环：load 一次，save/validate 按步调度

`Trainer.train()` 在进入循环前加载指定 checkpoint，并记录 loaded step；这让“恢复后的第一步”也能正确触发一次性初始化/timeout 切换（`torchtitan/trainer.py:943-963`）。每步固定顺序是：

```text
GC -> train_step -> checkpoint.save -> optional validation -> profiler.step
```

对应源码为 `torchtitan/trainer.py:965-986`。首个相对 step 后，通信 timeout 从初始化/编译容忍值缩短为 steady-state 值（`torchtitan/trainer.py:988-997`），而不是只在全新训练的 step 1 生效。

checkpoint 在 validation 前触发，意味着验证失败时该训练步的可恢复状态已经提交；这是从代码顺序得到的工程含义，不是文档口号。

---

## 7. 当前最容易踩的四个坑

1. **照搬旧 CLI 大串参数**：仍兼容，但新特性不会继续扩 CLI；新 run 应写 full-config recipe（`torchtitan/config/README.md:54-68`）。
2. **把 config registry 当永久 API**：旧 per-model registry 正在迁往 `torchtitan_recipes`（`torchtitan/config/README.md:70-80`）。
3. **先建 optimizer 再并行化模型**：Trainer 明确反过来做，因为 optimizer 必须绑定最终 sharded params（`torchtitan/trainer.py:426-543`）。
4. **把 CUDA Graph 理解成整个 train step 捕获**：核心 Trainer 只包装 forward/backward body；optimizer、checkpoint、validation 仍是图外控制流（`torchtitan/trainer.py:579-588`、`torchtitan/trainer.py:850-890`）。

---

## 8. 小结

- 推荐入口是 Python full configuration：`--module` 选模块，`--config` 选返回 `Trainer.Config` 的函数。
- `train.main()` 保持很薄；可组合性的核心在 `Configurable.build()` 与 Trainer 固定生命周期。
- 初始化必须遵守“distributed/mesh → meta model → parallelize → materialize → optimizer → data/checkpoint/context”的依赖顺序。
- 一个训练步以全局有效 token 为归一化基准，在 optimizer 前做跨并行轴有限性闸门，并协调 checkpoint staging。
- 核心 Trainer 已接入 CUDA Graph、验证与组件化 checkpoint/data；这些不再只是实验目录功能。

## Related Pages

- [[02_engineering/02_train_frameworks/torchtitan/index|TorchTitan 知识地图]] —— 最新页面地图与代码基线。
- [[10_torchtitan_parallel_dims_analysis]] —— `Trainer.init_distributed()` 构造的命名 mesh 与并行约束。
- [[16_torchtitan_spmd_types_analysis]] —— `model.parallelize()` 和 `train_context` 背后的布局契约。
- [[11_torchtitan_fsdp_analysis]] —— meta 模型在物化前如何被 FSDP2 分片。
- [[23_torchtitan_compute_memory_optimizations_analysis]] —— core CUDA Graph、compile、低精度与显存优化。
- [[27_torchtitan_graph_trainer_compiler_runtime_analysis]] —— 实验 GraphTrainer 的 joint graph 控制面，与本页 eager Trainer 主链对照。
- [[25_torchtitan_simple_fsdp_analysis]] —— GraphTrainer 内把参数物化与梯度规约表达成 collective 的 SimpleFSDP 子机制。
