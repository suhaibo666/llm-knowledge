---
title: "Megatron-FSDP 深度解析"
---

# Megatron-FSDP 深度解析

> **源码基线**：`NVIDIA/Megatron-LM@85902ef599ea4eb06ada7567a479c524b605767a`（`dev`，2026-09-01）
> **重定基线**：2026-09-01 由 `71092579`（2026-08-27）推进，跨 7 个提交；该增量只触及 20 个 `megatron/` 文件，本页 `path:line` 引用所涉源文件均不在其中，故无行号漂移，无需逐条重核。
> 核心文件:`megatron/core/distributed/fsdp/src/megatron_fsdp/`(16 个 `.py`、11321 行;其中 `param_and_grad_buffer.py` 5332 行、`megatron_fsdp.py` 1544 行),接入层 `megatron/core/distributed/fsdp/mcore_fsdp_adapter.py`(654 行),仓内文档 `docs/user-guide/features/megatron_fsdp.md`(619 行)
> 配套阅读：[[16_megatron_distributed_optimizer_analysis]] §11（三方对比）、[[22_megatron_memory_optimization_analysis]]、[[20_megatron_comm_overlap_analysis]]、[[19_megatron_dist_checkpointing_analysis]]。
> 适用读者:已了解 ZeRO 分级与 Megatron DDP,要读懂、调参或移植 Megatron-FSDP 这台机器的工程师。
> **叙事顺序**：本页按五拍组织——背景 → 为什么这么设计（含被否掉的替代）→ 实现思路与细节 → 约束 → 发展趋势。
> **合并来源**：2026-08-28 新建,吸收并取代 [[16_megatron_distributed_optimizer_analysis]] §11.2「MegatronFSDP 详细分析」与 §11.6「FSDP 与并行拓扑的关系」、旧 `27_megatron_tp_fsdp_resharding_supplements_analysis` §3「Megatron-FSDP 内部实现」;三套分片方案的**横向对比**仍是 16 号页 §11 的职责,本页不重复。
> **最近更新**：2026-09-03。完成旧补遗页的最终归并：补入参数 gather 粒度开关；16 号页仍以 §11.2/§11.6 保留 owner link，三方对比（DistributedOptimizer / TorchFSDP2 / MegatronFSDP）仍留在那里。

---

## 1. 背景：ZeRO-2/3 在 Megatron 里没有现成的机器可用

### 1.1 要解决的问题

`DistributedOptimizer` 只做到 ZeRO-1 —— 梯度缓冲区按 DP 度切开,每 rank 只保管一段优化器状态。**参数本身仍是每 rank 一份完整副本**。当模型大到"单 DP rank 装不下一份 bf16 权重"时(1.xT 级 MoE、DeepSeek-V3 规模),必须把参数也切开,即 ZeRO-3:算某一层之前把这层参数 all-gather 出来、算完立刻释放。

这条路上有两个现成选项,Megatron 都没有直接采用:

| 选项 | 为什么不够 |
|---|---|
| 扩展 `DistributedOptimizer` 到 ZeRO-3 | 它的分片建立在"梯度缓冲区的字节"上,没有"层"的概念,也没有 forward/backward 的 unshard/reshard 生命周期 |
| 直接用 PyTorch FSDP2(`fully_shard`) | Megatron 确实接了一条(`megatron/core/distributed/torch_fully_sharded_data_parallel.py`),但它是**逐参数 DTensor 分片**,与 Megatron 的扁平桶、TE 量化参数、MoE grouped GEMM 的连续内存要求冲突(§2.1) |

### 1.2 它从哪来:两次改名与一次搬家

这个子系统的历史很短,但两个节点都能从提交历史里读出来:

- **2025-02-26,`d165a8548`** —— commit message 即 `MCore Customized FSDP (Distopt-based FSDP)`,首次加入 `megatron/core/distributed/custom_fsdp/fully_sharded_data_parallel.py`。名字直说了它的出身:**建在 distributed optimizer 那套缓冲区之上的 FSDP**。
- **2025-08-21,`af28b5a55`** —— commit message 即 `[FSDP] Decouple Custom FSDP to make it independently installable`。该提交删掉 `custom_fsdp/fully_sharded_data_parallel.py`(835 行),把 `custom_fsdp/param_and_grad_buffer.py` 整体搬到 `fsdp/src/megatron_fsdp/`,并新建 `fsdp/src/README.md`、`fsdp/src/megatron_fsdp/__init__.py`、`fully_shard.py`、`uneven_dtensor.py`、`utils.py` 与接入层 `fsdp/mcore_fsdp_adapter.py`。**这一步是本页第 2 拍的核心事件**。

旧名字至今还留着痕迹:前向的 profiler 标签仍写作 `record_function("CustomFSDP.forward")`(`megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py:1504`);CLI 侧 `--use-megatron-fsdp` 会顺手把已弃用的 `args.use_custom_fsdp` 置 True,并在注释里写明「The flag `use_custom_fsdp` is deprecated and will be removed in future versions」(`megatron/training/arguments.py:1206-1210`)。

### 1.3 子系统边界

`megatron_fsdp/` 是一个**自带 `pyproject.toml` 与 `package_info.py` 的独立分发包**,当前版本 `0.6.0rc0`(`megatron/core/distributed/fsdp/src/megatron_fsdp/package_info.py:4-7`)、发行名 `megatron-fsdp`(`megatron/core/distributed/fsdp/src/pyproject.toml:18`)、依赖只有 `torch`/`einops`/`packaging`(`:23`)。

| 文件 | 行数 | 职责 |
|---|---|---|
| `param_and_grad_buffer.py` | 5332 | 心脏:参数分组分桶、四类扁平 buffer、四档临时分配器、AG/RS 两条流水线 |
| `megatron_fsdp.py` | 1544 | 主模块 `MegatronFSDP`:训练状态机、全部 module hook、对外的 sync API |
| `utils.py` | 869 | `FSDPDistributedIndex`(进程组/DeviceMesh 索引)、RNG tracker、版本探测 |
| `fully_shard.py` | 783 | 公开 API `fully_shard_model` / `fully_shard_optimizer` 与全部参数校验 |
| `uneven_dtensor.py` | 483 | 非均匀 DTensor 的分片元数据、DCP write item、redistribute |
| `mixed_precision.py` | 406 | `MixedPrecisionPolicy` 与 FP8 transpose cache |
| `distributed_data_parallel_config.py` | 237 | 脱离 MCore 时的 `DistributedDataParallelConfig` 备份定义 |
| `experimental/`(6 文件 + `__init__`) | 1487 | 以 `DBuffer`/`Placement`/`ParameterGroup` 为骨架的"最小实现"重写(§10) |
| `package_info.py` | 27 | 版本与发行元数据 |

> [!warning] 同名文件不是同一个东西
> `megatron/core/distributed/param_and_grad_buffer.py` 是 **DDP / DistributedOptimizer** 的缓冲区实现(见 [[16_megatron_distributed_optimizer_analysis]] §3.7);本页讲的是 `megatron/core/distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py`。两者同名、职责相近、代码不共享。引用行号时必须带全路径。

### 1.4 记号与命名约定

| 出现位置 | 名字 | 说明 |
|---|---|---|
| `fully_shard_*` 公开 API | `zero_dp_strategy` | `fully_shard.py:75`,取 `0..3` 或四个字符串 |
| `ddp_config` / Megatron CLI | `data_parallel_sharding_strategy` | 前者在 `fully_shard.py:386` 被翻译成后者;CLI 为 `--data-parallel-sharding-strategy` |
| 四个取值 | `no_shard` / `optim` / `optim_grads` / `optim_grads_params` | 对应 DDP / ZeRO-1 / ZeRO-2 / ZeRO-3(`fully_shard.py:31-48`) |
| DP 轴的两级 | `dp_shard_dim`(内层)/ `dp_outer_dim`(外层) | HSDP/HFSDP 用,`fully_shard.py:66-70` |
| FSDP unit | 最小可释放模型单元(`megatron_fsdp.py:130-133`) | Megatron 侧默认 `TransformerLayer`/`MoETransformerLayer`/`MambaLayer`(`mcore_fsdp_adapter.py:63-71`) |

---

## 2. 为什么这么设计：为了零 `COPY`,把分片切在"FSDP unit 的扁平桶"上,而不是切在参数上

朴素做法有两条:要么按 PyTorch FSDP2 的方式逐参数切 DTensor,要么像 DistributedOptimizer 那样把整段字节按 DP 度均分。Megatron-FSDP 两条都没走。仓内文档把理由写得相当直白,下面四条中前三条有源码/文档原话,第四条源码沉默、由本页重建并标为推断。

**① 被否掉的替代:FSDP2 的逐参数分片 —— 判据是"进出通信缓冲区的那次 `COPY`"。**
`docs/user-guide/features/megatron_fsdp.md:514` 明写:「While `torch.distributed.tensor.DTensor` defaults to per-parameter sharding, where Tensors are split evenly on `dim=0` across the data-parallel domain, Megatron-FSDP uses **non-uniform or un-even `DTensor` shards** of a (flattened) group of parameters associated with an FSDP unit」;配图说明把两者的代价并排写出:「FSDP2 requires `COPY` operations to move parameters and gradients in and out of communication buffers to reduce the frequency of NCCL collective calls, while Megatron-FSDP assigns sliced views of contiguous communication buffers to parameters associated with an FSDP unit」(`:520`)。收益列在 `:523-526`:**更少的 NCCL 调用**(只有 dtype 或分布式拓扑不同的参数才拆成独立 collective)、**缓冲区天然连续**(「contiguous-by-design, supporting optimized CUDA kernels that require buffers backed by contiguous memory, such as grouped GEMMs used in MoE」)。
→ 代价也写明了:同一个 `DTensor` 参数**在不同 rank 上形状可以完全不同**,整参数落在别的 rank 时本地张量为空(`:528`);因此需要一整个 `uneven_dtensor` 库来做逐参数的 unshard/reduce,否则"假设各 rank 字节数对称"的逐参数集合通信会**永久等待不会到达的字节**(`:530`)。

**② 被否掉的替代:按 DP 度直接均分字节 —— 判据是 kernel 的 locality。**
同一节 `:532` 指出连续只是一半要求,另一半是**locality**,而 FSDP 会破坏它:「block-wise quantization (scaling factor / `absmax` calculations for MXFP8, NVFP4, etc.) requires DP communication and custom max-reduce kernels if the block is sharded by FSDP」。于是 `:534` 给出替代方案:计算一个 FSDP unit 内所有参数 `p.shape[1:]` 的**最小公倍数**,把未分片缓冲区 pad 到 `DP × LCM` 的整数倍,形成"DP-LCM 网格",保证 **`dim=0` 的任何一行都不会被 FSDP 从中间劈开**;三种参数(整除 LCM / 大于但不整除 / 小于但不整除)的装填规则逐条列在 `:543-545`。代码侧就是 `_pad_if_needed`(`param_and_grad_buffer.py:284-287`)与分组第 3 步的 `math.lcm`(`:2031-2032`)。

**③ 被否掉的替代:直接写进 `megatron/core/distributed/` —— 判据是"能不能被别的框架装走"。**
提交 `af28b5a55`(2025-08-21)的标题就是结论:`Decouple Custom FSDP to make it independently installable`。落到当前源码有四处硬证据:
- `src/` 下自带 `pyproject.toml`,`name = "megatron-fsdp"`、`packages.find.include = ["megatron_fsdp", "megatron_fsdp.*"]`(`megatron/core/distributed/fsdp/src/pyproject.toml:11`、`:18`),README 给出 `pip install megatron-fsdp` 与 PyPI 链接(`megatron/core/distributed/fsdp/src/README.md:29-33`);
- 两个入口文件都写了 `try: from megatron.core… / except ImportError:` 的**双源导入**,注释直说「Megatron-LM is not installed, use Megatron-FSDP as a standalone module」(`megatron_fsdp.py:36-48`、`fully_shard.py:18-25`)—— 这正是包内为什么要留一份 `distributed_data_parallel_config.py`(237 行)与自己的 `is_submodule`;
- `experimental_api` 装饰器上挂着 `TODO(@cspades): Copied from megatron.core.utils to avoid depending on MCore for Megatron-FSDP`(`fully_shard.py:55-56`);
- 仓内文档把定位写成「"Bring Your Own Parallelism": Works seamlessly with PyTorch, Megatron-LM, Megatron-Bridge, and TransformerEngine, and can be plugged into other frameworks such as HuggingFace Transformers and TorchTitan」(`docs/user-guide/features/megatron_fsdp.md:30`)。
→ 这条路要付的价钱见 §8 与 §9:MCore 专属的知识(哪个参数是列并行/行并行)只能由接入层**按模块类名重新推断**一遍。

**④ 被否掉的替代:每次 unshard 现分配一块临时缓冲 —— 判据是分配器抖动与 NCCL 注册。**
默认路径是 `StorageResizeBasedBucketAllocator`(`param_and_grad_buffer.py:532`;不开 `fsdp_double_buffer` 时走的就是这一支,`:2710-2711`),靠 `_alloc_storage`/`_free_storage` 内的 `Tensor._typed_storage()._resize_()`(`:119`)直接伸缩存储,文档解释了为什么绕开 caching allocator:「Cache fragmentation and garbage collection can procrastinate large quantities of `cudaMalloc` and `cudaFree` operations that can block programs and spike memory」(`docs/user-guide/features/megatron_fsdp.md:401`)。但同一句紧接着给出这条路的死穴:「modifying the underlying storage of a buffer is not compatible with NCCL symmetric registration or CUDA graphability, which require a persistent state during runtime」。于是又有三档持久池,docstring 各自写死理由 —— `RotaryBucketAllocator`「implements a circular buffer recycling strategy **to minimize memory fragmentation**」(`param_and_grad_buffer.py:567-574`)、`FixedPoolAllocator`「maintains a fixed pool of pre-allocated buffers, reusing them to **reduce the overhead and fragmentation caused by frequent allocation and deallocation**」(`:652-661`,池大小默认 2、注释注明是双缓冲 `:672`)、`MaxPoolAllocator`「For every parameter group / bucket, the maximum storage required across all FSDP units is pre-computed to recycle buffers across different FSDP units」(`:916-929`)。
→ 判据是**拿显存换分配器确定性与 NCCL 能力**:`nccl_ub` 会强制打开持久池(`distributed_data_parallel_config.py:224-226`),换来的是零 `COPY`、对称 kernel 与 SHARP 卸载(`docs/user-guide/features/megatron_fsdp.md:610`、`:614-617`),以及**更少的 SM 占用** —— 主模块 docstring 的原话是「uses less number of SMs, resulting better overlapped computation performance」(`megatron_fsdp.py:153-155`),即通信本身少抢计算资源,重叠才真正兑现。

> [!note] 推断
> 源码与文档陈述的是**事实**:四档分配器各自的 docstring 理由、`FixedPoolAllocator` 要求"深度方向模型对称"而 `MaxPoolAllocator` 取各 unit 的最大值(`docs/user-guide/features/megatron_fsdp.md:412-414`)。**"四档构成一条由松到紧的取舍阶梯、默认档位选 `_resize_` 是因为绝大多数模型不开 `nccl_ub`"这层判断由本页承担 —— 源码没有这样表态**,也没有任何地方比较过四档的显存开销。要引用这条判断,请回到 `param_and_grad_buffer.py:462-470`、`:567-574`、`:652-661`、`:916-929` 与 `docs/user-guide/features/megatron_fsdp.md:401-414` 这几个 locator,不要引用本段推断。

---

## 3. Quick Start：两个入口与最小调用链

**入口 A —— 独立包(不需要 Megatron)**:`fully_shard_model` + `fully_shard_optimizer`,`megatron/core/distributed/fsdp/src/megatron_fsdp/fully_shard.py:63`。完整最小例子见 `megatron/core/distributed/fsdp/src/README.md:37-88`。注意这个 API 被 `@experimental_api` 标记(`fully_shard.py:62`),文档也明说「`fully_shard` is an _**experimental**_ API」(`docs/user-guide/features/megatron_fsdp.md:139`)。

**入口 B —— Megatron-LM 训练脚本**:三个 flag(`docs/user-guide/features/megatron_fsdp.md:147-149`)
```
--use-megatron-fsdp
--data-parallel-sharding-strategy {no_shard, optim, optim_grads, optim_grads_params}
--ckpt-format fsdp_dtensor
```
走 `megatron/core/distributed/fsdp/mcore_fsdp_adapter.py:74` 的 `FullyShardedDataParallel`,它在 `:222` 内部再构造 `MegatronFSDP`。

**从哪开始读源码**(主类 `MegatronFSDP(torch.nn.Module)` 定义在 `megatron_fsdp.py:94`):

```
fully_shard_model                                      fully_shard.py:63
  ├─ DistributedDataParallelConfig                    :384-402
  ├─ FSDPDistributedIndex(device_mesh, shard dims)    :405-426
  └─ MegatronFSDP(...) 构造调用                       :429-442
       └─ MegatronFSDP.__init__              megatron_fsdp.py:198
            ├─ _check_module_parameter_types                 :351  # 有没有 expert 参数
            ├─ _init_fsdp_param_and_grad_buffer              :365
            │    ├─ ParamAndGradBuffer       param_and_grad_buffer.py:2098
            │    │    └─ _get_parameter_groups                :1831 # 四步分组(§4)
            │    ├─ GradReducePipeline                        :3957 # RS 流水线(§6)
            │    └─ AllGatherPipeline                         :4417 # AG 流水线(§6)
            └─ _register_fsdp_hooks         megatron_fsdp.py:509    # 四类 hook(§6)
```

关键调用点在 `fully_shard_model` 的函数体，而不是类定义本身：入口先把 API 参数落成 DDP config 与 `FSDPDistributedIndex`，然后才在 `megatron/core/distributed/fsdp/src/megatron_fsdp/fully_shard.py:429-442` 实际调用 `MegatronFSDP(...)`。因此按上图可直接进入构造器，不需要再搜索谁实例化主类。

一次训练迭代里 Megatron 侧真正碰到的 FSDP API,就是接入层转发出来的那 8 个(`mcore_fsdp_adapter.py:254-261`):`zero_grad_buffer` → forward/backward(参数的 gather 与释放全靠 hook,训练循环看不见)→ `finish_grad_sync` → `optimizer.step()`。独立包路径下多一层:`fully_shard_optimizer` 把 `optimizer.step()` 换成包装器,在 base step 前后分别补 `finish_grad_sync()` 与 `install_optimized_model_weights()`(`fully_shard.py:536-556`)。

---

## 4. 机制①：从 module 到 bucket —— 四步分组与 DP-LCM 分片网格

### 4.1 FSDP unit 怎么定

`policy.fsdp_unit_modules` 里给的是**类**,不是实例。`_get_parameter_groups` 的 Step 0 遍历 `module.modules()`,把每个匹配到的子模块的参数名收成一个 unit;并且**跳过已经属于某个 unit 的嵌套模块**(`param_and_grad_buffer.py:1860-1872`)。文档把这条规则说得更完整:「if a module matches an FSDP unit class but is already a sub-module of a previously registered FSDP unit, it is skipped, so the outermost (and necessarily largest) FSDP unit class in any module sub-tree becomes the effective FSDP unit module」(`docs/user-guide/features/megatron_fsdp.md:358`)。

**没被任何 unit 覆盖的参数不会被释放**——文档写作「Parameters and sub-modules that are not members of an FSDP unit are not sharded」(`:346`),源码侧对应 `AllGatherPipeline.reset(preserve_non_fsdp_units=True)` 默认把非 unit 桶标成 `PRESERVED` 而不释放,理由写在 docstring 里:「their params may be read across module boundaries」(`param_and_grad_buffer.py:4478-4485`)。

### 4.2 四步分组算法

`_get_parameter_groups`(`param_and_grad_buffer.py:1831`)按注释分成四步:

| 步 | 位置 | 做什么 | 判据 |
|---|---|---|---|
| Step 0 | `:1856-1872` | 登记 FSDP unit 的参数名 | 外层优先,嵌套 unit 跳过 |
| Step 1 | `:1912-1948` | 按属性四元组 `(dtype, is_expert_param, requires_grad, fsdp_unit_id)` 归组 | 属性完全相同才能进同一 collective |
| Step 2 | `:1950-1997` | 按 `suggested_bucket_size`(默认 `40_000_000`,`:248`)切桶 | **只对 `fsdp_unit_id is None` 的组生效**(`:1983-1988`)——unit 内部不按大小切 |
| Step 3 | `:1999-2046` | 按 `chunk_size_factor = p.shape[1:].numel()` 再分,不同 factor 取 `math.lcm`(`:2031`) | 保证 DP 分片边界不劈开 `dim=0` 的一行(§2 ②) |
| Step 4 | `:2054-2090` | 把同一 `(fsdp_unit_id, is_expert_param)` 的桶聚成一个 **bucket group** | 注释:「reducing the number of collective calls and increasing per-collective efficiency」(`:2054-2056`) |

两条会**额外切桶**的特例:
- **共享 embedding**:`_does_param_require_new_bucket`(`:1874-1885`)在非 `no_shard` 下把 `shared_embedding` 参数单独成桶,docstring 说明理由是让首尾 PP stage 对该参数的优化器状态切法一致,「allowing the DP reduce-scatter to be before the embedding all-reduce」。
- **grouped expert 张量**(由 #5013 引入):`_should_split_from_grouped_expert_bucket`(`:1889-1899`,调用点 `:2012`)把 ≥3D、且 `chunk_size_factor` 与桶不一致的 expert 张量拆出去,docstring 写明是「to avoid LCM-inflated bucket alignment padding」。
- expert 参数的判定极其朴素:`is_expert_parameter = lambda n, p: ".experts." in n`(`:1887`),即**按参数全名里有没有 `.experts.`**。

### 4.3 桶内的索引:三层坐标

承载这套概念的两个 dataclass 薄得出奇:`BucketingPolicy`(`:233`)只有 `suggested_bucket_size` / `fsdp_unit_modules` / `data_parallel_sharding_strategy` 三个字段(`:248-250`),`Bucket`(`:445`)干脆只有 `data: torch.Tensor` 一个字段(`:459`)。**桶不是一张对象图,而是「一段扁平张量 + 一组索引」**——索引才是重头:`build_data_parallel_buffer_index`(`:257`)不真正分配显存,只算索引:每个参数在全局扁平 buffer 里的 `TensorItemIndex`、桶的 `BucketIndex`、本 rank 分片的 `ShardBucketIndex`(docstring `:266-282` 明说「the global bucket buffer is only temporarily allocated, but is abstractly tracked via indices」)。`no_shard` 下不做 pad,其余策略一律 pad 到 `dp_world_size × chunk_size_factor`(`:284-287`)。

---

## 5. 机制②：四类 buffer,以及 ZeRO 阶梯就是三个布尔量

### 5.1 ZeRO 四档 = 三个"要不要切"

`_init_each_parameter_group_buffers`(`param_and_grad_buffer.py:2455`)把四种策略翻译成三个布尔量(`:2461-2476`),这是整套实现里最干净的一处:

| `data_parallel_sharding_strategy` | model weight | main weight | grad | ≈ ZeRO |
|---|---|---|---|---|
| `no_shard` | ✗ | ✗ | ✗ | DDP |
| `optim` | ✗ | ✓ | ✗ | ZeRO-1 |
| `optim_grads` | ✗ | ✓ | ✓ | ZeRO-2 |
| `optim_grads_params` | ✓ | ✓ | ✓ | ZeRO-3 |

非法取值直接 `ValueError`(`:2477-2480`)。

主类 docstring 用另一种口径写同一张表(`megatron_fsdp.py:101-109`),并补上一处容易漏掉的细节:**`optim` 这一档除优化器状态外也切混合精度的 main weight**,`optim_grads` / `optim_grads_params` 同样切,docstring 用一句「omitted without detailed notation」带过 —— 也就是说上表 `main weight` 一列在三档里都是 ✓ 并不是笔误。

>  [!warning] `no_shard` 档有一条独立的正确性陷阱
> `no_shard` 下参数本就在各 DP rank 复制,梯度经 all-reduce 后也是复制值。因此 ① `start_param_sync` 对 `no_shard` **直接 return**、不做 all-gather(`megatron_fsdp.py:1289-1290`);② 梯度统计与范数只能在 **TP/PP(`model_parallel_group`)** 上规约,**不能**再在 DP 维度规约,否则 grad norm 虚高致不收敛 —— 实现是 `effective_intra_dist_opt_group = mp_group if data_parallel_sharding_strategy == 'no_shard' else intra_dist_opt_group`(`megatron/core/optimizer/__init__.py:1068-1075`,注释自陈:「gradients are replicated across DP ranks after all-reduce, so grad stats should only be reduced over TP/PP (model_parallel_group) to avoid inflating the norm」)。

**切了就必须重叠,不是可选项。** `MegatronFSDP.__init__` 在 `megatron_fsdp.py:323-333` 把这条写成硬规则:`optim_grads_params` 强制 `overlap_param_gather = True`、`optim_grads`/`optim_grads_params` 强制 `overlap_grad_reduce = True`,并且只要不是"延迟规约"档(`no_shard`/`optim`)就 `assert self.ddp_config.overlap_grad_reduce`。**这意味着"关掉重叠省显存"这条调优空间在 ZeRO-2/3 下已经被用掉了**,再想省只能动 bucket 与分配器(§2 ④)。

### 5.2 一个 `ParameterGroup` 上挂几种 buffer

`ParameterGroup`(`:1764-1828`)最多挂 8 个 `DataParallelBuffer`,常用的四个:

| 字段 | 何时创建 | 内容 |
|---|---|---|
| `model_weight_buffer` | `!= no_shard`(`:2784-2785`) | **计算精度**权重(FP8 时 dtype 记为 `torch.uint8`,`:2763-2764`) |
| `transpose_weight_buffer` | 参数是需要转置数据的 FP8 张量(`:2766-2775`,`:2804`) | MXFP8 的列向数据 |
| `main_weight_buffer` | `requires_grad`(`:2777-2781`,`:2826`) | 高精度 master 权重,默认 fp32 |
| `main_grad_buffer` | 同上(`:2845`) | 梯度累加 buffer |

`DataParallelBuffer` 本身有**两种工作模式**,docstring 写在 `:1243-1258`:sharded(本 rank 持久保存一段)与 unsharded(整桶持久保存,但仍能按 rank 取"虚拟分片")。这个双模是 `optim` / `optim_grads` 这类"部分切"的策略能和全切策略共用一套代码的原因。

### 5.3 AG 与 RS 走不同的通信组

`model_weight_buffer` 可以拿到一个**独立的 all-gather 进程组**(`fsdp_group_ag`),注释给了理由:「to enable overlap with gradient reduction operations (main_grad_buffer). This avoids head-of-line blocking between forward all-gather and backward reduce-scatter on the same communicator」(`param_and_grad_buffer.py:2744-2754`)。这一条是 §6 两条流水线能真正并行的前提,而不只是"用两条 CUDA stream"。

---

## 6. 机制③：hook 状态机 + 两条流水线

### 6.1 四个训练状态

`TrainingState`(`megatron_fsdp.py:51-63`)只有四个值,挂在**每个子模块**上(`:533-534`):

| 状态 | 含义 |
|---|---|
| `FORWARD` | pre-forward 到 post-forward 之间,参数需要 unshard |
| `PRE_BACKWARD` | 反向计算前,参数需要 unshard |
| `POST_BACKWARD` | 反向计算后,梯度需要 re-shard |
| `IDLE` | 无 un/sharding 活动 |

状态机真正的作用在**激活重计算**这一处:`_root_pre_backward` 把所有子模块置成 `PRE_BACKWARD`(`:944-948`),于是重计算触发的那次 forward 走到 `_pre_forward_param_unshard` 时会**取消前向预取**(`:780-782`),走到 `_post_forward` 时会**延迟释放**参数(`lazy_release = True`,`:976-981`,注释:「The corresponding backward pass may still need these parameters, and delaying avoids an unnecessary all-gather」)。这就是主模块 docstring 里那句「When recomputing a whole Transformer layer, gather parameters once for both the recomputation and backward computation」(`:116-118`)的实现。

### 6.2 四类 hook 与它们的 autograd 载体

`_register_fsdp_hooks`(`megatron_fsdp.py:509`)注册的四类 hook,以及各自的实现手段(文档 `docs/user-guide/features/megatron_fsdp.md:380-394` 与源码一一对应):

| Hook | 源码 | autograd 载体 | 做什么 |
|---|---|---|---|
| pre-forward | `:776` | `register_forward_pre_hook` | AG 本 unit + 按 `FORWARD_PASS_ORDER` 预取下一个 |
| post-forward | `:972` | `register_forward_hook` | 释放本 unit 参数(重计算中改为 lazy) |
| pre-backward | `:893` | `register_multi_grad_hook`(由 post-forward 装) | AG 本 unit + 按 `BACKWARD_PASS_ORDER` 预取 |
| post-backward | `:678` / `:701` | 注入 `RegisterFSDPBackwardFunction`(`:1510`)+ `register_post_accumulate_grad_hook` | 释放参数、`_grad_acc` 累加、按需发起 RS |
| root post-backward | `:849` | `torch.autograd.Variable._execution_engine.queue_callback`(`:969`) | 兜底处理剩余梯度、收尾 |

`_grad_acc`(`:629-674`)是"切了没切"的分叉点:梯度 buffer 已分片时,把 `param.grad` **拷进**未分片的临时桶等 RS(`:646-655`);未分片时直接 `main_grad.add_` **原地累加**,注释写明「because we only reduce once per optimization cycle」(`:660-667`)。

### 6.3 两条流水线与预取

- **`AllGatherPipeline`**(`param_and_grad_buffer.py:4417`):逐 bucket group all-gather 参数。预取由 `PrefetchOrder`(`:4404-4414`)选方向,实现就是 `next_bucket_id` 在 bucket id 上 **±1**(`:4587-4608`)—— 前向找下一个更大的 id、反向找下一个更小的 id。
- **`GradReducePipeline`**(`:3957`):逐桶把梯度 reduce-scatter 成分片。它用 `bucket_grad_ready_params` 计数(`:3979`),集齐一个桶的参数才发 collective。

默认的参数 gather 粒度是 **FSDP unit**：一次共同 unshard 该 unit 下的全部子模块。`enable_fine_grained_param_gather=True` 会改成 **per-Module** 粒度，默认值为 `False`（`megatron/core/distributed/fsdp/src/README.md:134-135`）。这个开关改变的是参数 materialize 的窗口与粒度，不改变 §5 的分片语义。

每个桶有四态 `BucketStatus`(`:3940-3954`):`EMPTY` / `PRESERVED` / `COMMUNICATING` / `READY_TO_USE`。

```
optim_grads_params 一层的前向:
  compute:  ── 算第 i 层 ────────────►  算第 i+1 层 ──►
  ag 流:    ── AG 第 i+1 层参数(预取)──►
  第 i 层算完 → release_module_parameters(bwd=False) → 只留 1/N 分片
```

> [!warning] 预取顺序 = bucket 编号顺序,不是执行顺序
> 预取只做 `bucket_id ± 1`(`:4587-4608`),而 bucket 编号来自 `module.named_parameters()` 的**注册顺序**(§4.2 Step 1 的遍历)。两者一致时预取有效;模型的实际执行顺序与参数注册顺序不一致时(例如自定义调度),预取会取错桶。源码对此没有做任何校验,这条是**本页的推断**,依据是 `param_and_grad_buffer.py:1916`(按 `named_parameters()` 顺序建组)与 `:4587-4608`(按 id 相邻预取)这两个 locator。

开启持久池时,预取还要先问一句"下一个桶塞得下吗"——`need_skip_prefetch` 在 `fsdp_double_buffer` 下检查 `_persistent_allocators_can_fit`(`:4610-4618`)。这解释了 `fsdp_buffer_count` 为什么要能大于 2:1F1B EP overlap 下同时活着的是"反向/重计算单元 + 当前前向单元 + 预取的后继",接入层因此硬要求 `>= 3`(`mcore_fsdp_adapter.py:192-199`)。

### 6.4 收口:`finish_grad_sync` 与参数指针的两副面孔

`finish_grad_sync`(`megatron_fsdp.py:1359-1389`)按顺序做四件事:等 RS 完成 → `attach_grad_to_optimizer_state` 把分片梯度挂到优化器状态 → 等参数 AG 完成 → `_replace_param_with_distributed_if_needed`。

最后这一步是这套设计里容易被忽略的关键:**同一个 `nn.Parameter` 在训练态与优化/存档态指向两个不同的张量**。`_replace_param_with_distributed_if_needed`(`:1391`)把模块参数换成 `optimizer_named_parameters` 里的分片 DTensor,`forward()` 一进来又 `_replace_param_with_raw_if_needed()` 换回去(`:1503`)。文档把这条写在 State Dictionary 一节:「When `module.state_dict()` … is invoked, Megatron-FSDP will swap all parameter references to point to sharded `DTensor` main weights」(`docs/user-guide/features/megatron_fsdp.md:396`)。

---

## 7. 机制④：与 EP / TP / HSDP·HFSDP 的叠加

### 7.1 “最后切”是状态变换顺序，不是 rank 集合互斥

文档规定：跨多维拓扑切模型状态时，FSDP **最后执行**，因为它在计算前后立即 unshard/reshard；因此 FSDP 操作的是已经被 TP/EP 切过的 strided shard（`docs/user-guide/features/megatron_fsdp.md:492-500`）。这里的“最后”描述**张量状态的变换顺序**，不意味着 FSDP、TP、EP、PP 的 process-group rank 集合两两不相交；对包含当前 rank 的正交 mesh 维度，两组在规则矩形 mesh 中通常至少以该 rank 为交点。

实现把这种正交性表达为**独立 DeviceMesh 维度**。standalone API 的 `dp_shard_dim` 可以指向纯 DP 子 mesh，也明确支持展平的 DP-CP 子 mesh，此时参数、梯度、optimizer state 同时沿 DP 与 CP ranks 分片（`megatron/core/distributed/fsdp/src/megatron_fsdp/fully_shard.py:112-129`）。Megatron adapter 的 dense mesh 则明确命名为 `["dp_cp", "tp"]`，并固定传 `dp_shard_dim="dp_cp"`、`tp_dim="tp"`（`megatron/core/distributed/fsdp/mcore_fsdp_adapter.py:472-485`）；HSDP 再增加独立的 `outer_fsdp_dp` 维（`:442-459`）。PP 不在这个单 stage 的 DeviceMesh 里。

令 dense 拓扑的纯 DP、CP、TP、PP 大小分别为 `D、C、T、P`，则两种合法口径要分开：

```
standalone 选择纯 DP shard dim:       |G_fsdp| = D = world / (T × P × C)
Megatron adapter 选择 dp_cp shard dim: |G_fsdp| = D × C = world / (T × P)

例：world=64、T=4、P=2、C=2
    D = 64 / (4 × 2 × 2) = 4
    adapter 的 dense FSDP shard group 大小 = D × C = 8
    每次 AG/RS 处理的是当前 PP stage 上、已经按 TP 切过的 local shard
```

所以准确的不变量是：FSDP 通信沿 `dp_shard_dim` 改变 rank 坐标，同时固定 TP（以及当前 PP stage）坐标；不能写成 `FSDP group ∩ TP/EP/PP group = ∅`，也不能在 adapter 路径把 CP 从 shard size 中除掉。EP 使用另一张 expert mesh，见 §7.2。

### 7.2 EP:两个 DeviceMesh,自动识别

Megatron-FSDP 维护**两个** mesh —— 稠密模块一个、MoE 专家模块一个(`docs/user-guide/features/megatron_fsdp.md:505-508`)。expert 参数的识别有两处、口径不同:

- **有没有 expert 参数**(决定是否必须提供 `expt_dp_group`):`_check_module_parameter_types` 看 `param.allreduce` 属性(`megatron_fsdp.py:351-363`),调用点在 `:296`,缺 group 就 `ValueError`(`:299-306`)。
- **某个参数是不是 expert**(决定分到哪个组):`".experts." in name`(`param_and_grad_buffer.py:1887`)。

**delayed wgrad**:开了 `overlap_dispatch_backward_with_experts_wgrad` 的 `TransformerLayer`,其 expert 参数会跳过常规 post-accumulate-grad hook,改由 MoE 层在延迟 wgrad 算完后回调 FSDP 的梯度处理函数(`setup_delayed_wgrad_acc_hook`,`megatron_fsdp.py:66-91`),把 EP 的 A2A 与 DP 的梯度 RS 重叠(另见 [[20_megatron_comm_overlap_analysis]])。

### 7.3 HSDP 与 HFSDP:外层切不切优化器状态

`outer_dp_sharding_strategy` 只有两个合法值(`fully_shard.py:334-347`):

| 组合 | 名字 | 内层(DP-Shard) | 外层(DP-Outer) |
|---|---|---|---|
| `optim_grads_params` + `no_shard` | **HSDP** | ZeRO-3 全分片 | 复制,最后一次反向 all-reduce |
| `optim_grads_params` + `optim` | **HFSDP** | ZeRO-3 全分片 | 优化器状态再切一刀(ZeRO-1) |

**默认组合就是 HSDP 的退化形式。** 不传 `ddp_config` 时 `MegatronFSDP` 自建的默认配置是 `data_parallel_sharding_strategy="optim_grads_params"` + `outer_dp_sharding_strategy="no_shard"`,重叠两项均为 `True`(`megatron_fsdp.py:250-264`);只要没指定 `dp_outer_dim`,外层就退化成一个 size-1 的组,即纯 FSDP。`fully_shard_model` 的签名默认值同口径(`zero_dp_strategy=3` / `outer_dp_sharding_strategy=0`,`fully_shard.py:75-76`),且 `dp_outer_dim` 与 `hybrid_fsdp_group` 必须**同时给或同时不给**,XOR 就 `ValueError`(`fully_shard.py:360-366`)。

HFSDP 的收益写在 `docs/user-guide/features/megatron_fsdp.md:468-472`:优化器状态按 `DP-Inner × DP-Outer` 切、梯度与权重只按 `DP-Inner` 切;因为权重与梯度每个优化周期才更新一次,**所有跨节点(DP-Outer)集合通信都能推迟到优化步**(`:481`)。代价是外层必须依赖内层:`outer_dp_sharding_strategy == "optim"` 而内层不是 `optim_grads_params` 会直接 `ValueError`,旁边挂着 `TODO(@shjwudp, @cspades): Requires various modifications to support`(`fully_shard.py:351-359`)。

---

## 8. 机制⑤：接入层 `mcore_fsdp_adapter.py` 干了什么

包本体不认识 MCore,所以"MCore 专属的知识"全在 654 行的接入层里。三件最能说明问题的事:

1. **重新推断 TP 属性。** `_MODULE_TYPE_REGISTRY`(`mcore_fsdp_adapter.py:79-104`)按**模块类名**把 `ColumnParallelLinear`/`TEColumnParallelLinear`/… 归为 `column`、`RowParallelLinear`/… 归为 `row`、各种 Norm 归为 `replicated`,`_detect_parallelism_type`(`:290-346`)再加一串回退规则(融合模块 `TELayerNormColumnParallelLinear` 的 layer-norm 权重要单独判成 `replicated`、看 `partition_dim`、看 `TELinear.parallel_mode`),最后由 `_annotate_tensor_parallelism`(`:348-358`)把结果写成参数上的 `_tensor_parallel_mode`。注释直说这套是 `forked from Megatron-Bridge`(`:79`、`:293`)。
2. **决定默认 FSDP unit。** 只有 `optim_grads_params` 才给默认 unit 列表 `[TransformerLayer, MoETransformerLayer, MambaLayer]`(+ 非 EP-overlap 时的 `MoTTransformerLayer`),其余策略给空列表(`:180-188`、`:63-71`)。
3. **建 device mesh 并转发 API。** mesh 由 `einops.rearrange` 生成,轴序写死并挂着 `# TODO: Supports configurable (dp, cp, ep, tp) order.`(`:587-590`);构造完 `MegatronFSDP` 后把 8 个方法直接绑到适配器上(`:254-261`),并补一次 TP 组内 RNG 同步(`:266`)。

**与 `DistributedOptimizer` 的关系不是"二选一",而是"套一层"。** `--use-megatron-fsdp` 会强制打开 `--use-distributed-optimizer`(`megatron/training/arguments.py:1212-1218`),但 `DistributedOptimizer.__init__` 一旦看到 `use_megatron_fsdp` 就**直接 return**、不再建自己的 buffer 与 range(`megatron/core/optimizer/distrib_optimizer.py:711-713`);`step` 里的参数同步也改成调 `model_chunk.start_param_sync()`(`:3265-3269`),状态字典直接交给内层优化器(`:958-967`)。

---

## 9. 约束

每条都能落到一个 `file:line`,越出前提就不再适用。

| # | 前提 / 代价 | 源码落点 | 破坏后的表现 |
|---|---|---|---|
| 1 | **`CUDA_DEVICE_MAX_CONNECTIONS` 不能是 `1`** | `megatron/training/arguments.py:1252-1254` 硬 assert:「FSDP requires CUDA_DEVICE_MAX_CONNECTIONS > 1 or unset」 | 直接启动失败。而 TP 的异步通信重叠**依赖**该变量为 1(`megatron/core/tensor_parallel/layers.py:761-775`,那边只是 `warnings.warn`)——文档承认这是取舍:「May slightly affect TP and CP performance though」(`docs/user-guide/features/megatron_fsdp.md:159-160`)。**不是正确性问题,是 TP/CP 侧丢重叠**(详见 [[12_megatron_tp_analysis]] §3.2) |
| 2 | 只能配 `--ckpt-format fsdp_dtensor` | `arguments.py:1256-1258` | assert 失败;与 [[19_megatron_dist_checkpointing_analysis]] 的 `torch_dist` 不通用,跨格式要走 `checkpoint_inspector.py` 转换(`docs/user-guide/features/megatron_fsdp.md:260-285`) |
| 3 | 优化器只支持 `sgd` / `adam` | `arguments.py:1233-1236` | assert 失败 |
| 4 | 不支持 `moe_single_grouped_weight` / `moe_single_grouped_bias` | `arguments.py:1221-1231`,`ValueError` 报文自陈:TE `GroupedTensor` 参数需要重映射 grouped 底层存储,「DDP has a separate GroupedTensor-aware path」 | 只能退回 DDP/DistributedOptimizer 路径 |
| 5 | HSDP 外层切分要求内层必须 `optim_grads_params` | `fully_shard.py:351-359`(旁边 `TODO(@shjwudp, @cspades)`) | `ValueError`:「outer sharding is dependent on inner sharding」 |
| 6 | `init_model_with_meta_device` 不能配 `no_shard` | `fully_shard.py:367-371`、`arguments.py:1287-1291` | `ValueError` |
| 7 | `prefetch_recompute_forward_weights` 三重前提:`optim_grads_params` + 全量重计算 + 不开 EP overlap | `fully_shard.py:372-382`、`arguments.py:1260-1272` | assert 失败 |
| 8 | TP sub-mesh 即使不用 TP 也必须给 | `fully_shard.py:411-413` 注释自陈是已知接口债 | 不是可配置项 |
| 9 | **FP8 参数只在 ZeRO-3 这一档支持** | `param_and_grad_buffer.py:3658-3662` 的 TODO 自陈:目前只覆盖 FSDP,`no_shard`/`optim`/`optim_grads` 的量化路径仍是开放问题 | ZeRO-1/2 下不要指望同等 FP8 覆盖 |
| 10 | `nccl_ub` 连带强开持久池,且与 `expandable_segments:True` 冲突 | `distributed_data_parallel_config.py:217-226`;主模块 docstring 同口径:「This flag automatically sets fsdp_double_buffer to True, which uses additional GPU memory」(`megatron_fsdp.py:153-160`) | 前者多占显存(README `:141`);后者直接 `ValueError`(torch < 2.11) |
| 11 | `fsdp_buffer_count` 只有开了持久池才允许改;开了就必须 ≥ 2 | `distributed_data_parallel_config.py:228-237` | `ValueError` |
| 12 | `FixedPoolAllocator` 要求**深度方向模型对称** | `docs/user-guide/features/megatron_fsdp.md:412`;不对称的 unit 回退到 `_resize_` 分配器 | 混合架构(Mamba+Transformer+MoE)必须改用 `MaxPoolAllocator`(`--megatron-fsdp-max-pool-double-buffer`,`:414`) |
| 13 | 1F1B EP overlap 下持久池必须 ≥ 3 且不能用 per-layer CUDA graph | `mcore_fsdp_adapter.py:192-205` | assert 失败 |
| 14 | `optim_grads_params` 下不能开 `check_weight_hash_across_dp_replicas_interval` | `arguments.py:1247-1250` | assert 失败 |
| 15 | `sync_model_each_microbatch` 配 `no_shard`/`optim` 时用户必须自己调 `zero_grad_buffer()` | README `:133` 的 WARNING | 未分片梯度会被重复规约进梯度累加 buffer |
| 16 | 非均匀 DTensor 上不能跑"逐参数、假设字节对称"的集合通信 | `docs/user-guide/features/megatron_fsdp.md:530` | **挂死**——等待永远不会到达的字节;必须走 `uneven_dtensor` 里的对应函数 |
| 17 | mesh 轴序写死 | `mcore_fsdp_adapter.py:587-590` 的 TODO | 与 [[17_megatron_parallelism_orchestration_analysis]] 的 `order` 不是同一套可配置机制 |
| 18 | `keep_fp8_transpose_cache` 在 Blackwell 上没有收益 | README `:136` | 纯亏(参数量 × 1 Byte)显存 |

**与激活重计算的协同**(自 16 号页 §18.2 并入):虽然 FSDP 不接管激活,但它对重算路径做了专门优化——主模块 docstring 的能力清单里写着「Optimized activation recompute with shard-aware communication: When recomputing a whole Transformer layer, **gather parameters once for both the recomputation and backward computation**」(`megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py:116-119`)。即整层重算时参数只 all-gather 一次,重算与反向共用,不会因为多跑一遍前向而多付一次通信。

**故意不做的事**:

- **不接管重计算与激活。** 文档划界清楚:「Activations (`fprop`) and data gradients (`dgrad`) are not sharded or distributed」(`docs/user-guide/features/megatron_fsdp.md:309`)——激活侧的手段见 [[18_megatron_recompute_analysis]] 与 [[22_megatron_memory_optimization_analysis]]。
- **不做自己的优化器。** 它只提供分片的参数与梯度,step 仍由 `DistributedOptimizer` 包着的 torch 优化器做(§8)。
- **不承担 mesh 轴序配置**(约束 17),也不感知 Megatron 的 `order` 字符串。
- **`experimental/` 不对外暴露**:`megatron_fsdp/__init__.py:34-53` 的 `__all__` 里没有它的任何符号。

---

## 10. 发展趋势

> [!note] 推断:锚点是基线 `71092579` 下的源码事实(`experimental/` 子包、TODO/FIXME、弃用函数、发布方式),方向判断由本页承担,不是源码的自陈计划。

**一、现役实现正在被一套"最小实现"重写。**
`experimental/` 子包 1487 行,三处 docstring 把意图写得很直白:`experimental/__init__.py:15`「Experimental Megatron-FSDP implementation.」、`experimental/fully_shard.py:15`「**Minimal** Megatron-FSDP fully_shard entrypoint.」、`experimental/module.py:15`「Module mixin for the **minimal** Megatron-FSDP path.」。对照现役实现,`param_and_grad_buffer.py` 第 3 行就挂着 `# TODO: Split this file into smaller files.`。新骨架是 `DBuffer` / `Placement` / `Layout` / `FsdpParameterGroup` 四件套(`experimental/placement.py:15-20` 把 placement 概念对标 DTensor 的 `Replicate`/`Partial`,并说明 `Flat` 是目前**唯一**实现的分片 placement)。**由此可推断**:§4–§6 那套 `ParamAndGradBuffer` + 两条流水线的组织方式会被替代;而 `experimental/dbuffer.py:22` 直接 `import torch.distributed._symmetric_memory as symm_mem`,说明新路径把 NCCL 对称内存当作一等公民,而不是像现役实现那样靠 `nccl_ub` 选项去注册。该子包目前**不在公开 API 里**(§9),迁移进度可以盯 `experimental/module.py:223-224` 那条与主仓 PR 绑定的待办:「After NVIDIA/Megatron-LM#5411 lands, move this sync to the optimizer post-step hook instead of running it every microbatch」。

**二、"包"的一侧会继续减薄,"接入层"的一侧会继续加厚。**
`pip install megatron-fsdp` 已经是正式发布路径(README `:29-33`),`fully_shard.py:55-56` 的 `avoid depending on MCore` 是这条线的直接证据。**由此可推断**:凡是"只有 Megatron 才知道"的知识(§8 那套按类名推断 TP 属性、默认 FSDP unit 列表、mesh 轴序)都会继续往 `mcore_fsdp_adapter.py` 沉;读这套代码时应当先分清"库本体"与"接入层"两侧,否则会把接入层的限制当成库的限制。

**三、DTensor 侧接口尚未定型。**
`uneven_dtensor.py:390-394` 的 `gather_uneven_dtensor_to_full_tensor` 只剩一句 docstring「Deprecated: use `redistribute_uneven_dtensor_to_replicated` instead.」,函数体就是转调新名;同文件 `:127-131` 还挂着一条同步优化待办(加形状一致性预检以防挂死、把 barrier 批量化)。`param_and_grad_buffer.py:5248`、`:5283` 另留两条校验缺口(「Add validation checks for the legality of DTensor」「Implement consistency check for duplicated TP parameters」)。**由此可推断**:跨版本使用 `--ckpt-format fsdp_dtensor` 时要留意接口改名与校验缺失,§9 约束 16 那条"挂死"目前**没有**运行期防护。

**四、契约性的一条:contiguity/locality 官方自陈是 work-in-progress。**
`docs/user-guide/features/megatron_fsdp.md:547` 明写「Generalized support for contiguity and locality in Megatron-FSDP is a **_work-in-progress_** and will evolve with contribution from the OSS community and PyTorch」,并点名 veScale 论文(arXiv 2509.07003)作为该问题的系统分析。**由此可推断**:§2 ② 的 DP-LCM 网格是当前答案而非终局,新的量化 recipe(block size 更大的 NVFP4 之类)可能再次改写这条分片规则。

---

## 11. 小结

| 问 | 答 | 位置 |
|---|---|---|
| 它是什么 | 一个 vendor 进 Megatron、也能 `pip install` 的独立 FSDP 库 | §1.3、§2 ③ |
| 主线 | 按 FSDP unit 的**扁平桶 + 非均匀 DTensor**,换 FSDP2 的逐参数分片,目标是零 `COPY` | §2 ① |
| ZeRO 阶梯怎么实现 | 三个布尔量(model/main weight、grad 切不切) | §5.1 |
| 通信怎么不掉速 | 两条流水线 + `bucket_id ± 1` 预取 + 独立 AG 进程组 | §6.3、§5.3 |
| 最容易踩的坑 | `CUDA_DEVICE_MAX_CONNECTIONS` 与 TP 冲突;非均匀 DTensor 上跑对称集合通信会挂死 | §9 约束 1、16 |
| 三套方案怎么选 | 见 [[16_megatron_distributed_optimizer_analysis]] §11.4 选型矩阵 | — |

---

*生成依据:`Megatron-LM` `dev` 分支 `85902ef599ea4eb06ada7567a479c524b605767a`(2026-09-01;由 `71092579` 重定基线而来,更早一次为 2026-08-28 由 `ee3f1ff` 推进)。源码行号以该 commit 为准。历史结论取自 `d165a8548`(2025-02-26)与 `af28b5a55`(2025-08-21)两个提交的 message 与文件变更。*

## 配置契约：FSDP 实现选择

| 字段 | 来源 | 默认 | 契约 | 行 |
|---|---|---|---|---|
| `use_megatron_fsdp` | `DistributedInitConfig` | `False` | 启用 Megatron-FSDP；不能与 Torch FSDP2 同开。 | `megatron/training/config/common_config.py:95-96` |
| `use_torch_fsdp2` | `DistributedInitConfig` | `False` | 启用 Torch FSDP2；当前不支持 PP，且源码声明尚非稳定发布阶段。 | `megatron/training/config/common_config.py:98-101` |

这两个字段决定模型采用哪套全分片实现，因此 owner 在本页；16 只保留三方案的横向选择，[[26_megatron_optimizer_step_internals_deepdive]] 只解释选定实现进入 step 后的状态变化。

## Related Pages

- [[16_megatron_distributed_optimizer_analysis]] — ZeRO 0-3 四阶段的概念层与三套分片方案的横向对比（§11）；本页是其中 MegatronFSDP 一栏的实现权威页。
- [[30_megatron_rl_posttraining_consistency_analysis]] — 跨并行配置的 Resharding/Refit owner；旧补遗中的权重搬运内容已归并到该页。
- [[12_megatron_tp_analysis]] — TP 的异步重叠依赖 `CUDA_DEVICE_MAX_CONNECTIONS=1`,与本页 §9 约束 1 正面冲突,配置时必须一起看。
- [[19_megatron_dist_checkpointing_analysis]] — `fsdp_dtensor` 与 `torch_dist` 两套存档格式的分工。
- [[20_megatron_comm_overlap_analysis]] — delayed wgrad / A2A overlap 把 EP 通信与本页的 AG/RS 流水线拼在一起。
- [[22_megatron_memory_optimization_analysis]] — 激活侧的显存手段,与本页只管模型态的边界互补。
- [[02_engineering/02_train_frameworks/megatron-lm/index|Megatron-LM 知识地图]] — 返回全部 35 篇内容页的主题索引。
