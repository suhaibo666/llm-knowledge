---
title: "Megatron-LM 最小训练：从 torchrun 到 checkpoint"
---

# Megatron-LM 最小训练：从 torchrun 到 checkpoint

> **源码基线**：`NVIDIA/Megatron-LM@85902ef599ea4eb06ada7567a479c524b605767a`（`dev`，2026-09-01）
> **学习前置**：先读 [[01_megatron_architecture_analysis]]，并完成仓库官方安装步骤。
> **回答的问题**：一轮最小 Megatron 训练怎样从进程组初始化走到梯度同步、参数更新和分布式 checkpoint 回读？
> **不覆盖**：本页只验证最小闭环；真实数据、并行算法和生产调优分别由 [[11_megatron_dataset_analysis]]、[[03_megatron_parallelism_geometry_quickstart]] 与专题页负责。
> **最后复核**：2026-09-03。

---

## 1. 为什么先跑这个例子

官方 quickstart 没有让新读者直接从八卡 LLaMA-3 配方开始，而是先安排一个“两张 GPU + mock data”的分布式训练环，用它验证安装和分布式环境；通过后才进入八卡 FP8 LLaMA-3 与真实数据预处理（`docs/get-started/quickstart.md:10-32`）。被否掉的路线是**用生产配方同时验证环境、数据、模型和性能栈**：一旦失败，很难判断问题属于哪一层。最小脚本把问题压缩为 NCCL、Megatron 进程组、模型/数据构造、一次 forward-backward 和 checkpoint 五类边界。

它不是“伪训练”调用图：脚本真的构造两层 GPT、包装 Megatron DDP、执行 5 次参数更新，并把 sharded state 保存后重新加载（`examples/run_simple_mcore_train_loop.py:54-76,207-261`）。但它使用 mock dataset、`Adam` 和 TP=2/PP=1，不代表生产配置的吞吐或精度选择。

## 2. 运行命令与成功信号

在 Megatron-LM 仓库根目录执行官方命令：

```bash
torchrun --nproc_per_node=2 examples/run_simple_mcore_train_loop.py
```

命令与“两张 GPU、mock data、先验证环境”的定位来自官方 quickstart（`docs/get-started/quickstart.md:14-20`）。脚本读取 `torchrun` 注入的 `RANK`、`WORLD_SIZE` 和 `LOCAL_RANK`，把当前进程绑定到 CUDA device 后初始化 NCCL（`examples/run_simple_mcore_train_loop.py:28-51`）；因此直接运行 `python examples/run_simple_mcore_train_loop.py` 不满足入口契约。

成功时应观察到两类输出：5 个 iteration 的 loss 字典，以及 checkpoint 回读后的 `Successfully loaded the model`（`examples/run_simple_mcore_train_loop.py:229-261`）。数值本身不是本页验收条件；真正的完成边界是 save 和 load 都返回，模型重新放回 CUDA device。

## 3. 一次真实执行的七个状态边界

| 阶段 | 输入状态 | 完成后可依赖的状态 | 证据 |
|---|---|---|---|
| 1. 分布式初始化 | `torchrun` 环境变量 | NCCL 默认组与 Megatron model-parallel groups 已建立 | `examples/run_simple_mcore_train_loop.py:28-51` |
| 2. 模型构造 | 两层 `TransformerConfig` | `GPTModel` 参数和 layer spec 已确定 | `examples/run_simple_mcore_train_loop.py:54-76` |
| 3. 样本流构造 | null tokenizer、mock dataset config | 可迭代的 batch 字典 | `examples/run_simple_mcore_train_loop.py:79-117` |
| 4. 并行包装与调度选择 | CUDA model、DDP config | 模型具有梯度同步接口；调度函数已选定 | `examples/run_simple_mcore_train_loop.py:207-227` |
| 5. forward-backward | 一个 microbatch | 本轮局部梯度和 loss 已产生 | `examples/run_simple_mcore_train_loop.py:229-242` |
| 6. 梯度完成与更新 | 尚可能未完成同步的梯度 | 跨相关组修正完成，`Adam.step()` 更新参数 | `examples/run_simple_mcore_train_loop.py:244-251` |
| 7. 持久化验证 | DDP 包装后的模型 | sharded state 已保存、加载并写回 model | `examples/run_simple_mcore_train_loop.py:163-204,253-261` |

这七步区分了“已经提交一次计算”和“模型状态已经可用于下一步”。尤其 `forward_backward_func` 返回不等于梯度已经满足所有并行组的不变量；脚本在 optimizer 前显式调用 `finalize_model_grads([gpt_model])`（`examples/run_simple_mcore_train_loop.py:233-249`）。

## 4. 沿源码走一遍

### 4.1 先建立组，再创建依赖组的状态

入口先初始化 TP=2、PP=1，再设置 model-parallel CUDA seed，之后才构造模型（`examples/run_simple_mcore_train_loop.py:207-213`）。这里有两条源码直接约束的依赖边：`model_parallel_cuda_manual_seed()` 的 docstring 明确要求它在 model parallel 初始化后调用，函数随后读取 TP/EP/ETP rank（`megatron/core/tensor_parallel/random.py:445-465`）；DDP 初始化会建立/读取 `ProcessGroupCollection` 的 DP 组并保存对应 handle（`megatron/core/distributed/distributed_data_parallel.py:95-120`），所以也必须位于进程组可用之后。脚本把模型构造放在 seed 之后是这个示例的具体顺序，本页不把这一步抬升成所有模型都不可颠倒的通用约束。两进程全部落入同一个 TP 组；该示例的 DP 度为 1，所以 DDP wrapper 在这里主要提供统一的梯度同步接口，而不是制造跨 rank 的 DP 流量。并行度推导见 [[03_megatron_parallelism_geometry_quickstart]]。

### 4.2 模型和数据刻意保持最小

`model_provider()` 只创建 2 层、hidden size 12、4 heads、词表 100 的 GPT，并先在 CPU 初始化，随后显式搬到 CUDA（`examples/run_simple_mcore_train_loop.py:54-74,211-213`）。数据侧使用 null-text tokenizer 和 `MockGPTDataset`；rank 0 编译 dataset helper 后用 barrier 等待，所有 rank 再构造相同的 train iterator（`:79-117`）。

被否掉的替代是先接真实 JSONL、tokenizer 模型和 `.bin/.idx` 数据：那会把数据格式问题混入环境验证。官方把真实数据预处理安排在最小训练和 LLaMA 示例之后，并明确由 `tools/preprocess_data.py` 生成 `.bin/.idx`（`docs/get-started/quickstart.md:30-64`）。

### 4.3 调度器选择与 DDP wrapper 是两个边界

脚本用 `DistributedDataParallelConfig` 关闭 overlap 和 distributed optimizer，再把 GPT 包成 Megatron DDP（`examples/run_simple_mcore_train_loop.py:215-223`）。紧接着 `get_forward_backward_func()` 根据当前 PP 状态返回调度器（`:225-227`）。前者决定参数/梯度同步接口，后者决定 microbatch 如何执行；二者不是同一个抽象。

`forward_step_func` 从 batch 取 tokens、position IDs、attention mask、labels 和 loss mask，调用模型并返回延迟求值的 loss 函数（`:120-160`）。主循环把它连同 iterator、model、序列长度和 microbatch size 交给调度器（`:229-242`），所以读者可以从入口逐跳走到 loss，而无需再搜索隐藏的示例 glue code。

### 4.4 梯度 ready 后才允许更新参数

调度器结束后，`finalize_model_grads` 完成 DP 梯度同步以及 TP 下非切分参数的修正，然后才调用 `optim.step()`（`examples/run_simple_mcore_train_loop.py:244-249`）。本例关闭通信 overlap，因而适合观察边界；生产配置如何把同步藏进反向和下一次前向，分别见 [[16_megatron_distributed_optimizer_analysis]] 与 [[20_megatron_comm_overlap_analysis]]。

### 4.5 save 成功不等于 load 已验证

保存函数先解开可能存在的 DDP wrapper，取 `sharded_state_dict` 后调用 distributed checkpoint save；加载函数用当前模型的 sharded state 作为模板，读取 checkpoint 并调用 `load_state_dict`（`examples/run_simple_mcore_train_loop.py:163-204`）。主程序随后把返回模型放回 device，并打印成功信号（`:253-261`）。因此本例的验收边界不是“目录出现”，而是**保存格式能被当前并行模型重新消费**。

## 5. 常见失败边界

| 现象 | 首先检查 | 原因边界 |
|---|---|---|
| 入口立即报环境变量缺失 | 是否通过 `torchrun` 启动 | rank/world/local-rank 在初始化函数中直接读取（`examples/run_simple_mcore_train_loop.py:40-46`） |
| CUDA/NCCL 初始化失败 | 两张可见 GPU、device 映射和 NCCL 环境 | 脚本固定使用 CUDA 与 NCCL（`:45-46`） |
| model-parallel 初始化报整除错误 | `WORLD_SIZE` 是否与 TP=2、PP=1 相容 | `world_size` 必须能被 `tp × pp × cp` 整除（`megatron/core/parallel_state.py:726-737`） |
| 数据阶段不同步 | rank 0 helper 编译是否完成 | 其他 rank 在 barrier 等待后才构造 dataset（`examples/run_simple_mcore_train_loop.py:90-111`） |
| 有 iteration 输出但 checkpoint 未完成 | 当前目录是否可创建 `ckpt`，save/load 是否均返回 | 路径固定为当前目录下 `ckpt`，最终成功信号在 load 后（`:253-261`） |

完成本页后，读者应能把一条报错定位到“进程组、模型/数据、调度、梯度完成、持久化”之一，而不是只知道重跑命令。

## Related Pages

- [[01_megatron_architecture_analysis]] —— 在最小脚本之上建立完整框架分层与训练状态机。
- [[03_megatron_parallelism_geometry_quickstart]] —— 解释 TP=2、PP=1 在两个进程上形成的 rank 与组。
- [[10_megatron_model_structure_analysis]] —— 深入 `GPTModel`、layer spec 和模型构造边界。
- [[11_megatron_dataset_analysis]] —— 从 mock data 进入真实 indexed dataset 与 tokenizer。
- [[15_megatron_pp_schedulers_analysis]] —— 解释 `get_forward_backward_func` 可能选择的调度器。
- [[16_megatron_distributed_optimizer_analysis]] —— 深入 DDP buffer、梯度完成和 ZeRO 分片。
- [[19_megatron_dist_checkpointing_analysis]] —— 深入 sharded state 的保存与加载协议。
