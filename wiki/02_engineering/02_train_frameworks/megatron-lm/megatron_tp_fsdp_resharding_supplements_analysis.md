# Megatron-LM TP·FSDP·Resharding 补遗

> 代码基准:`Megatron-LM/` 子仓库 `dev` 分支,commit `ee3f1ff`
> 核心文件:`megatron/core/distributed/fsdp/`(Megatron-FSDP,~9500 行)、`distributed/nonuniform_tp.py`(1463 行)、`resharding/`(~2000 行)
> 配套阅读:`megatron_tp_analysis.md`、`megatron_ddp_optimizer_analysis.md`、`megatron_parallelism_orchestration_analysis.md`
> 定位:"第一层补遗"第③份。补齐 3 块前面只点了名的内容。先做一处**勘误**:tier-1 清单里把它们叫"非均匀 TP / 弹性 resharding",读源码后更准确的命名是 ——

| tier-1 旧称 | 准确含义 |
|------------|---------|
| Megatron-FSDP 内部 | ZeRO-2/3 的**具体实现机器**(`megatron_ddp_optimizer_analysis.md` 只给了概念) |
| "非均匀 TP" | **Nonuniform TP(NTP)= TP 级容错**:备用 rank 顶替故障 rank |
| "弹性 resharding" | **Resharding / Refit = 跨并行配置搬运权重**,主用于 RL 的训练→推理 |

---

## 0. 总览

| # | 主题 | 代码 | 一句话 |
|---|------|------|--------|
| 1 | Megatron-FSDP 内部 | `distributed/fsdp/src/megatron_fsdp/` | ZeRO-2/3 的流水线 AG/RS、预取、临时桶分配器、DTensor 分片 |
| 2 | Nonuniform TP(NTP) | `distributed/nonuniform_tp.py` | TP 组里留备用 rank,核心 rank 故障时重分片续训 |
| 3 | Resharding / Refit | `resharding/` | 在两套并行布局间搬运权重(RL:训练模型→推理模型) |

---

## 1. Megatron-FSDP 内部实现(ZeRO-2/3 的机器)

### 1.1 动机:`megatron_ddp_optimizer_analysis.md` 只讲了概念

DDP 文档把 ZeRO-2/3 讲成"梯度也切 / 参数也切,逐层 all-gather 出参数、用完释放"。但**怎么切、怎么 gather、怎么不让通信拖慢**,实现细节全在 `megatron/core/distributed/fsdp/`。本节补齐这台"机器"。

### 1.2 `MegatronFSDP` —— FSDP 包装器

`megatron_fsdp.py:106`。包住模型,按 `data_parallel_sharding_strategy` 选 ZeRO 阶段(`no_shard`/`optim`/`optim_grads`/`optim_grads_params`,对应 ZeRO-0/1/2/3,见 `megatron_ddp_optimizer_analysis.md` §0.3)。一个 `TrainingState` 状态机(`:63`)跟踪"此刻参数/梯度处于分片还是聚合态"。

关键默认值(`:318`):
- `optim_grads` / `optim_grads_params` → **默认开启梯度 reduce-scatter 重叠**(切了梯度就必须重叠,否则严重掉速)。
- `optim_grads_params` → **默认开启参数 all-gather 重叠**。

### 1.3 核心:`ParamAndGradBuffer` + 两条流水线

`param_and_grad_buffer.py`(4712 行,FSDP 的心脏)。所有参数/梯度装进扁平 buffer,再分桶(`BucketingPolicy:232`、`Bucket:444`)。两条流水线把通信与计算重叠:

**`AllGatherPipeline`(`:3856`)—— 参数 all-gather 流水线**
逐桶 all-gather 出完整参数。关键是 **`PrefetchOrder`(`:3843`)**:`FORWARD_PASS_ORDER` 让它**在算第 `i` 层时,提前 all-gather 第 `i+1` 层的参数** —— 用计算掩盖参数聚合的延迟。这正是 ZeRO-3 "通信 ×1.5" 能落地不掉速的关键。

**`GradReducePipeline`(`:3398`)—— 梯度 reduce-scatter 流水线**
逐桶把梯度 reduce-scatter 成分片,与反向计算重叠(类比 DDP 的 `overlap_grad_reduce`,但这里 RS 后只留 `1/N` 分片)。

```
ZeRO-3 一层的前向(AllGatherPipeline):
  计算第 i 层 ───────────────►
  通信流: all-gather 第 i+1 层参数(预取)────►  ← 用第 i 层计算掩盖
  第 i 层算完即释放其完整参数(只留 1/N 分片)
```

### 1.4 临时桶分配器:聚合出来的参数放哪

参数 all-gather 出来需要一块**临时**完整缓冲(用完就释放)。反复 `malloc/free` 很慢,于是有一组 `TemporaryBucketAllocator`(`:461`)策略:
- `StorageResizeBasedBucketAllocator`(`:524`):靠 `storage().resize_()` 伸缩。
- `RotaryBucketAllocator`(`:558`):轮转复用几块缓冲。
- `FixedPoolAllocator`(`:642`):固定的**双缓冲**池(`fsdp_double_buffer`)—— 多占显存,但能注册 NCCL user buffer(`nccl_ub`),拿到 SM 高效的 NCCL 算法。

### 1.5 其他实现要点

- **DTensor 分片**:`fully_shard.py` 用 torch 的 `DeviceMesh` / `DTensor` 表达分片,`--ckpt-format fsdp_dtensor`。
- **不整除分片**:`uneven_dtensor.py`(483 行)处理参数 numel 不能被 DP 组整除的情况。
- **混合精度**:`mixed_precision.py` 的 `MixedPrecisionPolicy`,含 fp8 transpose cache。
- **HSDP**:`outer_dp_sharding_strategy`(内层节点内分片、外层跨节点复制),见 `megatron_ddp_optimizer_analysis.md` §3。
- **接入**:`mcore_fsdp_adapter.py` 把 Megatron-FSDP 缝进 Megatron-LM(`--use-megatron-fsdp`)。

### 1.6 与 DDP 文档的衔接

`megatron_ddp_optimizer_analysis.md` 的阶段③④(`optim_grads`/`optim_grads_params`)说"由 Megatron-FSDP 实现" —— 实现就是本节:`ParamAndGradBuffer` + `AllGatherPipeline`(预取重叠)+ `GradReducePipeline` + 临时桶分配器。ZeRO-3 通信 1.5× 不掉速,全靠这套预取流水线。

---

## 2. Nonuniform TP(NTP)—— TP 级容错

`distributed/nonuniform_tp.py`。模块 docstring 直说:"provides fault tolerance for tensor-parallel training by allowing a subset of TP ranks ('spares') to handle failures while 'core' ranks continue computation."

### 2.1 动机与解决的问题

大规模训练里单 GPU 故障是常态。TP 组是同步的 —— 组里**任一 rank 挂掉,整个 TP 组就停**,通常得整体重启。NTP 的动机:**让 TP 训练能扛住组内 rank 故障而不重启**。

### 2.2 机制:core rank + spare rank

一个 TP 组配置成 `tp_base` 个 rank,其中 `tp_spares` 个是**备用(spare)**:
- 健康时:`tp_base − tp_spares` 个 **core rank** 干活,spare 待命。
- 某 core rank 故障:它负责的 TP 分片**重分片(reshard)到存活的 core rank 上** —— 通过 `_ntp_all_to_all`(`:100`)做分片再分配;健康 core rank 用额外的 `side_grad` 存储(`_ntp_should_expand_param_grad`,`:193`)容纳多接手的那部分梯度。`recv_splits` 记录每个 rank 接收的分片切分。

```
正常:   TP 组 = [core0 core1 core2 | spare]    3 core 干活
core1 挂:reshard → core1 的分片经 A2A 摊到 core0/core2(+ side_grad 存储)
         训练用 reduced_tp_size = tp_base − tp_spares 继续,不重启
```

### 2.3 非侵入式设计

NTP 全部逻辑收在 `nonuniform_tp.py` 一个文件,以 Megatron 核心类的**子类**形式实现(`NonuniformTPDistributedDataParallel` 继承 `DistributedDataParallel`),不改动主代码。用法:`initialize_model_parallel()` 之后调 `initialize_nonuniform_tp_process_groups()`,并用 NTP 变体替换 DDP。`PerBufferParamLayout` / `FullParamLayout` 描述重分片后的参数布局。

### 2.4 开销与适用场景

| 维度 | 说明 |
|------|------|
| 代价 | spare rank 平时空转(算力冗余);故障后 `reduced_tp_size` 下负载略增 + side_grad 额外显存 |
| 收益 | TP 组扛单点故障,免整体重启 —— 大规模、长训练的可用性 |
| 适用 | 几千卡级长周期训练;小规模训练不必要 |

---

## 3. Resharding / Refit —— 跨并行配置的权重搬运

`resharding/`。README 首句:"Transfer model weights between different parallelism configurations (TP, PP, EP, DP) ... Used primarily in RL loops to move weights from a training model to an inference model that may use a different parallelism layout."

### 3.1 动机与解决的问题

**RLHF / RL 训练**里有两个模型实例:**训练模型**(为训练吞吐选一套并行布局,如 TP8×PP4)和**推理模型**(为生成延迟选另一套,如 TP2×PP1,可能还是 MXFP8)。每个 RL 迭代:训练模型更新权重 → 必须把新权重**搬进**推理模型,而两者并行布局不同 —— 参数的切分方式完全不一样。Resharding 解决的就是"**把权重从布局 A 搬到布局 B**"。

### 3.2 架构(四层)

```
refit.py        高层 API:swap_model_weights / prepare_swap_model_weights;计划缓存;MXFP8 自动识别
   │
planner.py      rank 0 集中构建 ReshardPlan(谁把哪个参数切片发给谁),scatter 给所有 rank
   │
execution.py    把 plan 里的 send/recv op 提交给 CopyService,处理写回
   │
copy_services/  可插拔传输后端:
   ├── nccl     GPU↔GPU,torch.distributed P2P
   ├── gloo     经 CPU 中转,Gloo 进程组
   └── nvshmem  NVSHMEM 流水线式 GPU↔GPU
```

外加 `transforms.py` 的 `MXFP8ReshardTransform` —— 搬运途中顺带做**格式转换**(BF16 → MXFP8),给推理模型用。

### 3.3 流程

```
prepare_swap_model_weights()  ── 一次性:构建并缓存 ReshardPlan(此时不传数据)
        │                         若目标是 MXFP8,先把目标权重量化成持久 MXFP8 buffer
        ▼
每个 RL 迭代:
swap_model_weights(src_model, target_model, refit_method="nccl")
        │  按缓存的 plan,src 各 rank 把参数切片 P2P 发给 target 各 rank
        ▼
推理模型拿到训练模型的最新权重(布局已转换),开始 rollout
```

### 3.4 要点与适用场景

| 维度 | 说明 |
|------|------|
| 核心 | 不是"训练并行",是**两套并行布局之间的权重迁移工具** |
| 计划集中化 | rank 0 算 plan 再 scatter,避免每个 rank 各算一遍 |
| 后端可插拔 | collocated(同卡同时持两模型)用 nccl;分离部署可用 gloo/nvshmem |
| 适用 | RLHF / RL 训练循环;训练与推理用不同并行配置的任何场景 |

它与前面所有文档不同 —— 那些讲"怎么并行训练",这一节讲"训练好的权重怎么换到另一套并行布局里去推理"。

---

## 4. 小结

| 主题 | 何时关心 | 与已有文档的关系 |
|------|---------|-----------------|
| **Megatron-FSDP 内部** | 用 ZeRO-2/3(`--use-megatron-fsdp`) | 是 `megatron_ddp_optimizer_analysis.md` 阶段③④的实现机器:预取流水线 AG/RS + 临时桶分配器 |
| **Nonuniform TP** | 千卡级长训练要扛 TP 组单点故障 | `megatron_tp_analysis.md` 的容错扩展;非侵入式子类 |
| **Resharding / Refit** | RL 训练,训练模型↔推理模型布局不同 | 独立工具,跨 `megatron_parallelism_orchestration_analysis.md` 描述的两套布局搬权重 |

至此"第一层补遗"3 份文档全部完成:① 并行编排 capstone、② PP 补遗、③ TP·FSDP·resharding 补遗。

---

*生成依据:`Megatron-LM` `dev` 分支 `ee3f1ff`。源码行号以该 commit 为准。配套文档:`megatron_tp_analysis.md`、`megatron_ddp_optimizer_analysis.md`、`megatron_parallelism_orchestration_analysis.md`、`megatron_pp_supplements_analysis.md`。*

## Related Pages

- [[megatron_tp_analysis]] · [[megatron_ddp_optimizer_analysis]] · [[megatron_parallelism_orchestration_analysis]] · [[megatron_rl_posttraining_consistency_analysis]] · [[megatron_dist_checkpointing_analysis]]
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]]
