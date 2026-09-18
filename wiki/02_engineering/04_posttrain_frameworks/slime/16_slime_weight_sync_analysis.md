---
title: "slime Megatron→SGLang 权重同步分析"
---

# slime Megatron→SGLang 权重同步分析

> **源码基线**：`THUDM/slime@4c193f1f37509cca70f0e88807a9305b70f63f4e`（`main`，2026-09-03）
> **源码基线**：`sgl-project/sglang@0b3bb0cbe31873994c9f989fddfe2f87ca839fdd`（`v0.5.15.post1`，2026-07-13；上游源码，不含 slime 补丁）
> **源码基线**：`NVIDIA/Megatron-LM@1dcf0dafa884ad52ffb243625717a3471643e087`（`dev`，2026-02-14）
> **主题**：optimizer step 之后的参数怎样交到正在服务的 SGLang：updater 的选择，四条数据面共用的拓扑无关重组与分桶，以及 NCCL、共卡 CUDA IPC（含 MoE 定向路由与越界 engine）、整份磁盘、增量磁盘四条数据面各自的搬运方式与暂停窗口。随后是提交协议与 SGLang 依赖边界（上游 v0.5.15.post1、slime 镜像补丁与镜像构建闸门各自提供什么）、调用流程、配套检查、约束与配置契约。核心代码在 `slime/backends/megatron_utils/update_weight/`、`slime/ray/actor_group.py` 与 `docker/patch/latest/`。
> **适用范围**：权重从训练侧到 serving 侧的重组、搬运与提交；CPU tag 与 sleep/wake 归 [[14_slime_megatron_training_analysis|Megatron 训练]]，full+disk 的 actor 生命周期与控制面归 [[11_slime_ray_control_plane_analysis|Ray 控制面]]，engine 健康检测与重建归 [[18_slime_fault_tolerance_observability_analysis|容错与可观测性]]，训推数值一致性归 [[17_slime_train_inference_consistency_analysis|训推一致性]]，低精度格式归 [[22_slime_low_precision_training_rollout_analysis|低精度训练与 rollout]]。
> **最近更新**：2026-09-17。覆盖四条数据面的同一最小实例与三张原理图、SGLang 上游与镜像补丁的逐端点依赖边界及缺补丁时的失效面，以及集中的源码阅读路线。

---

## 1. 特性概览

### 1.1 问题背景

训练侧每轮结束时，新参数以 Megatron 的 TP/PP/EP 分片散落在各训练 rank 上，并被 `TensorBackuper` 备份成 CPU `actor` tag；rollout 侧是另一组 SGLang 进程，按自己的 TP/EP 拓扑持有参数、正在服务请求，还持有用旧参数算出的 KV 与 radix 前缀缓存。把前者变成后者要同时守住四条不变量：一次提交只用同一个训练边界的完整快照，否则不同 rank 拿到不同 step 的分片，传输再准确也拼出一个不存在的模型；交付给 loader 的必须是它认识的名字、形状与行序，而训练分片既不是推理分片也不是 HF 张量（§2.1.1 的 fc1 就是反例）；并发请求看不到一半新一半旧的模型，也不能在新权重上复用旧 KV；共置时训练进程与 serving 进程分时占用同一块 HBM，谁拥有哪块显存必须明确。"把参数广播出去"只解决了搬运，另外三条都没有碰到。

### 1.2 解决方法

slime 把同步做成带版本号的提交。driver 在训练与保存之后调用 `RayTrainGroup.update_weights`（这一步在一轮迭代里的位置见 [[10_slime_end_to_end_iteration_analysis|端到端迭代]]），actor 先按需恢复坏掉的 engine 并重连，再调用 updater；updater 由 `create_weight_updater` 在 actor 初始化时按 mode × transport × colocate 选定。四种 updater 共用一个前半段：从 CPU `actor` tag（或 GPU 参数）出发，经 PP/EP 广播与 TP all-gather 还原每个参数的完整逻辑张量，用 `convert_to_hf` 转成 HF 名字与布局，按 `--update-weight-buffer-size` 分桶。后半段分四条：训推分离时每个 PP 源 rank 建一个临时 NCCL group，逐桶在 Ray 锁下先发元数据 RPC 再广播；共卡时每张训练卡把完整桶拍平成 CUDA IPC 句柄，由同 engine 的 Gloo 源收齐描述符后发一次 RPC，满足条件的 MoE 专家改走 rank 间定向 P2P，越出 actor 区间的 engine 由全局源 rank 走 NCCL 补发；整份磁盘把完整 HF checkpoint 写到共享目录，由 `RayTrainGroup` 在暂停窗口里让 engine reload；增量磁盘把每个 HF 张量与上一版的 CPU 字节快照做差、zstd 压缩后发布，每台 serving host 通过镜像补丁 `sglang-pull_weights.patch` 提供的 `/pull_weights` 在本地 checkpoint 上原位应用再 reload。在线两条的搬运落在 pause/flush 与 continue 之间；磁盘两条把写盘与主机侧应用挪到 pause 之前，窗口只覆盖 reload。SGLang 一侧的端点并不都来自上游：`/post_process_weights`、`/get_weight_version` 的返回值、`/pull_weights` 与 IPC 加载返回前的设备同步都由 slime 镜像在构建时打的补丁提供（§2.2.8）。

### 1.3 收益、开销和约束

| 维度 | 直接收益 | 必付成本或边界 |
|---|---|---|
| 拓扑解耦 | 训练 TP=2 可对接推理 TP=1/2/4；转换器与 loader 之间只约定 HF 名字、形状、dtype | 每个桶都在训练 GPU 上物化完整参数；只支持 `convert_to_hf` 覆盖的模型族 |
| NCCL | 不落盘，GPU 直连广播；异构 engine 按累计偏移加入同一 group | 多个 PP 源串行持锁；每桶一次 RPC 往返；整个搬运在暂停窗口里 |
| 共卡 IPC | 跨进程只传句柄与元数据，payload 走同卡显存 | 每张卡临时持有完整桶；pause/flush 只覆盖共卡前缀；RPC 返回前拷贝是否完成取决于镜像是否打了 `sglang-deterministic.patch` |
| MoE 定向路由 | 本例 P2P 12 份专家包、每卡转换并交付 4 个专家，对比通用路径 24 份、8 个 | 准入条件苛刻，任一不满足记日志退回通用桶 |
| 整份磁盘 | 训推只经共享目录与 HTTP 耦合，支持 external、异构 GPU、`release_train` | 每次写完整 checkpoint；共享文件系统可见性要靠 hook 补齐 |
| 增量磁盘 | 只发布变化字节，未变张量不写；主机侧原位应用 | 训练侧仍全量重组、全量 D2H 与逐字节比较；每台 host 需完整本地 checkpoint；版本链必须严格串行；主机侧依赖补丁端点 |
| 提交语义 | 版本号随每次 RPC 或 reload 交给 engine；整份磁盘在 CI 下逐 engine 读回 | 其余路径没有读回核对；RPC 成功不等于数值正确（§4.2） |
| 镜像补丁 | 补齐量化前后处理、版本读回、主机侧 pull 与 IPC 返回前同步 | `ENABLE_SGLANG_PATCH=0` 或不带补丁的镜像下，compressed-tensors 在线同步、CI 读回、增量与本地目录路径失败，IPC 返回语义退回上游（§2.2.8） |

### 1.4 术语约定

| 术语 | 含义 |
|---|---|
| HF 张量 / 桶 | `convert_to_hf` 产出的 `(HF 名, 完整逻辑张量)`；桶是一次搬运的若干 HF 张量，按 `--update-weight-buffer-size` 切分 |
| PP 源 rank | NCCL 与增量路径中每个 PP stage 上 `DP(含 CP)=0、TP=0` 的 rank，唯一负责转换与发送 |
| Gloo 源 | 共卡路径里每个 engine 对应一组训练 rank（即它占用的 GPU 槽位），组内首个 rank 收集描述符并发 RPC |
| 共卡前缀 / 越界后缀 | 共卡 updater 把 `gpu_offset + gpu_count` 不超过 actor GPU 数的前若干个 engine 当前缀，其余是后缀 |
| weight_version | updater 或 `RayTrainGroup` 维护的整数计数器，以字符串随 RPC 或 reload 交给 SGLang |
| base_version / 已应用标记 | 增量版本 N 声明建立在 N−1 之上；主机本地 checkpoint 在 `.weight_sync/state.json` 里记录已应用到的版本 |
| slime 代码 / 补丁 / 上游 | 本页对 SGLang 侧行为的三种证据标签：slime 仓内 Python 源码；`docker/patch/latest/*.patch` 里只在镜像构建时应用的补丁；`sgl-project/sglang@0b3bb0cb` 上游源码中读到的契约（未运行） |

---

## 2. 权重同步详细方案

### 2.1 最小实例：一个 GLU fc1 走四条数据面

取 Qwen2 结构第 0 层的 `decoder.layers.0.mlp.linear_fc1.weight`：hidden H=2，FFN F=4，完整形状 [8,2]，行记作 gate g0–g3 与 up u0–u3。训练侧 4 张卡 TP=2、DP=2（PP=1、EP=1）；slime 的 fc1 重排规则假设 Megatron GLU 分片的布局是 TP rank t 持有 [gate_t ; up_t]，即 TP 0 持有 `g0 g1 u0 u1`、TP 1 持有 `g2 g3 u2 u3`。四条数据面各配最小拓扑：训推分离时两个 engine 分别 TP=2 与 TP=4；共卡时 rollout 6 卡、每 engine 2 卡，actor 只占槽位 0–3；MoE 变体把同样 4 张卡改成 TP=1、EP=4、8 个专家，两个共卡 engine 各 EP=2；增量路径只看 gate_proj 的前 4 个 bf16 元素。三张原理图依次是：共同前半段与分桶、两条在线数据面、两条磁盘数据面；每条数据面底部都有按源码顺序排列的时间条，浅蓝底是服务暂停窗口。

![共同前半段：fc1 的两个训练 TP 分片经 all-gather 与 GLU 重排、HF 转换得到 gate_proj 与 up_proj，SGLang 加载器按推理 tp_rank 截取，下方是三种分桶规则](assets/slime_weight_sync_common.svg)

| 步骤 | 输入 | 决定性转换 | 输出 |
|---|---|---|---|
| 重组 | TP 0 `g0 g1 u0 u1`、TP 1 `g2 g3 u2 u3` | `all_gather_param`：各片 `chunk(2)`，按 [各片 gate…, 各片 up…] 拼接 | `g0 g1 g2 g3 u0 u1 u2 u3`；若直接首尾相接，gate_proj 会拿到 `g0 g1 u0 u1` |
| 转换 | 完整 [8,2] | `convert_qwen2_to_hf` 再 `chunk(2)` | `gate_proj` 为 `g0 g1 g2 g3`，`up_proj` 为 `u0 u1 u2 u3` |
| 加载（上游） | 完整 gate/up | SGLang 列并行 loader 按本 rank 的 `tp_rank` 截取 | 推理 TP=4 的 rank 0 取 `g0 u0`；把训练 TP 0 的分片直接拷过去是错的，推理 TP=2 时两边恰好相等只是布局巧合 |
| 分桶 | 转换后 chunk 300、260、100 MiB，阈值 512 | 已有内容且会超限才换桶 | `[300] [260+100]`；单个 600 MiB 独占一桶；专家参数每个 100 MiB、EP=4 时每批只装 1 个，EP 聚合后 400 MiB |
| NCCL | 两个 engine TP=2、TP=4 | `slime-pp_0` 的 world = 2 + 4 + 1 = 7，rank_offset 1 与 3 | rank 0 逐桶持锁广播；rank 1–3 只参加 TP all-gather |
| 共卡 IPC | e0 [0,2)、e1 [2,4)、e2 [4,6)，actor 4 卡 | 前缀判定 `offset + count ≤ 4` | e0、e1 走 IPC（Gloo 组 {0,1}、{2,3}，源 rank 0、2）；e2 越界，走 NCCL group `slime`，world 3 |
| MoE 路由 | 训练 rank r 持有 e(2r)、e(2r+1)；推理 EP 分片 0 为 e0–e3、分片 1 为 e4–e7 | 目标 rank = engine 偏移 + `dp_rank × EP + ep_rank` | 分片 0 → rank 0、2，分片 1 → rank 1、3；P2P 12 份，装批 `batch0 = e0–e3`、`batch1 = e4–e7` |
| 整份磁盘 | 版本 1 | 写 `weight_v000001/` → barrier、hook、barrier → pause → flush → reload | engine 版本 `"1"`；CI 下逐 engine 读回核对（读回端点由补丁改写） |
| 增量磁盘 | 旧 `80 3F 00 3F 00 C0 80 3E`，新 `80 3F 02 3F 00 C0 81 3E` | 逐字节 xor | 变化 2/8 字节，`perf/update_weights_density` 为 25%；xor 差分 8 字节、overwrite 编码 14 字节（均为压缩前） |

#### 2.1.1 共同前半段：训练分片不是 loader 能装的形状

TP 分片是 Megatron 为自己的矩阵乘切出来的：GLU 的 fc1 在每片内部把 gate 与 up 各放一半，所以把两片首尾相接得到的是交错行序。`all_gather_param` 先把每片对半切开，把所有 gate 半片放前、up 半片放后，才得到 HF 语义下的 `[gate; up]`；`convert_qwen2_to_hf` 再对半切成两个 HF 名字。此后的切分归 loader：上游 `ColumnParallelLinear.weight_loader` 拿到完整张量后按自己的 `tp_rank` 截取（§2.2.8），推理 TP=4 的 rank 0 要的是 `g0 u0`。训练分片与推理分片的对应关系因此随两边的 TP 与层类型而变，没有一种"直接拷贝分片"能在所有组合上成立；推理 TP=2 时两边恰好都是 `g0 g1 u0 u1`，那是布局巧合而不是接口。完整逻辑张量是训练侧唯一不依赖推理拓扑的交付物，slime 用它避免为每一对训练/推理拓扑写专用的重分片协议（本页推断），代价是每个桶都要在训练 GPU 上物化一次完整参数。两条重组路径都只按名字对 `linear_fc1` 做 GLU 重排：`all_gather_param` 的 stride 断言放行 stride 为 1 的 fc1，Direct 路径（`all_gather_params_async`）连 stride 断言也没有，所以非 GLU 的 fc1 在同步时不会报错，它在导入切分与训练前向上的错配归 [[23_slime_model_architecture_extension_analysis|模型架构扩展]] §2.2。`linear_fc2.weight` 还有一条特判：`partition_dim` 为 0 时改成 1 再拼接，源码注释写明是在绕开 Megatron grouped MoE 的缺陷。

#### 2.1.2 分桶：阈值约束的是桶，不是显存

三种装桶规则都是"已有内容且会超限才换桶"，所以单个超阈值的 chunk 独占一桶，阈值不是显存上限。它们度量的对象不同：NCCL 与增量路径的非专家桶按转换后 HF chunk 的字节累计（本例 300 | 260+100，600 独占）；专家桶按 TP 聚合后的本地参数字节乘 EP world size 判断，超限就先做 EP all-gather 再转换（本例每批 1 个 100 MiB 参数，聚合后 400 MiB）；张量路径与整份磁盘在构造 `HfWeightIteratorDirect` 时由 `pack_param_info_buckets` 按"分片字节 × TP（专家用专家 TP）"预估完整参数大小装好 `ParamInfo` 桶：张量 updater 在初始化时构造一次、此后每次同步复用；整份磁盘的 `save_hf_model_to_path` 每次同步都重新构造，于是每次都重做 PP/EP 的 `all_gather_object` 与全 world 的元数据核对。按桶流式处理让峰值停在桶级而不是整模型级；在每个 rank 上物化全模型会把拓扑复杂度换成全模型 HBM 峰值（本页推断）。临时 gather 缓冲、转换副本与量化辅助张量都不计入阈值。

#### 2.1.3 四条数据面的暂停窗口

![在线数据面：NCCL 的临时 group 与逐桶锁、共卡 IPC 的前缀与越界划分、MoE 定向路由的发送与装批，以及 RPC 返回前拷贝是否完成的补丁边界和两条暂停窗口时间条](assets/slime_weight_sync_online.svg)

在线两条（NCCL、共卡 IPC）的顺序是：version+1 → rank 0 对 engine 发 `pause_generation` 与 `flush_cache`（compressed-tensors 时再发一次 restore）→ Gloo barrier → 逐桶重组、转换、搬运 → 量化后处理 → `continue_generation` → barrier，搬运整体落在暂停窗口里。整份磁盘由 actor 写完 checkpoint 并跑完 hook 才返回，`RayTrainGroup` 随后可选地 pull 到主机本地盘，再 pause → flush → reload →（CI）核对版本 → 删目录 → continue；增量磁盘由 actor 自己完成做差、写盘、hook、`pull_weights`（主机侧应用与校验），然后 pause → flush → reload 本地目录 → continue。两条磁盘路径把最重的字节搬运移到暂停之前，窗口只覆盖 reload（下方磁盘数据面的图）。

#### 2.1.4 MoE 专家为什么可以不先全量聚合

TP 切的是单个张量的维度，必须先拼回完整张量；EP 切的是专家集合，训练专家 TP 与推理 MoE-TP 都为 1 时，每个专家的 fc1/fc2 在它的训练持有者上已经完整。规划器用 Gloo `all_gather_object` 找出每个专家名的最低持有 rank，按 `expert // (专家数 / 推理 EP)` 算出推理 EP 分片，再把分片映射到每个共卡 engine、每个 MoE-DP 副本上的训练 rank（engine 偏移 + `dp_rank × EP + ep_rank`），所以训推 EP 数不同、同一分片要复制到多个 engine 都能表达。本例 e0–e3 送到 rank 0 与 2、e4–e7 送到 rank 1 与 3；持有者本身就是目标时不发送，于是 e0、e1 各发 1 份，e2–e5 各发 2 份，e6、e7 各发 1 份，共 12 份；通用路径的 EP 广播让 8 个专家各到另外 3 个 rank，共 24 份，每张卡物化 8 个专家；定向路由下每个目标 rank 只转换并交付 4 个，作为发送方的 rank 另要暂存自己发出的专家。装批按包大小降序首次适配，约束是每个参与 rank 的 staging 字节不超过阈值：阈值取 4 包时，e4 因 rank 2 会超限另开 batch1，e6、e7 在两个候选里选总量更小的 batch1，得到 `batch0 = e0–e3`、`batch1 = e4–e7`；`tests/test_expert_routing.py` 锁定了按专家边界拆批与单包超阈值抛错两条规则。

#### 2.1.5 增量：字节差分与版本链

![磁盘数据面：整份磁盘的写盘、hook、补丁端点 pull 与暂停窗口里的 reload 和 CI 读回；增量磁盘的字节差分、两种编码、补丁提供的主机侧 pull 版本链，以及首次与重启两个失败边界](assets/slime_weight_sync_disk.svg)

`_encode_delta` 把每个 HF 张量 `contiguous().view(uint8)` 展平后与上一版的 CPU 快照比较。本例 bf16 值 0.5 → 0.5078125、0.25 → 0.251953125 只改了两个低位尾数字节：xor 编码写 `new ^ old`，8 字节里 6 个为 0，zstd（固定 level 1）压得最小，但它是对合，在已应用的状态上再应用一次会还原成旧值；overwrite 编码写"变化计数（u4）+ 位置（u4）+ 新值"，本例 4 + 2×4 + 2 = 14 字节，重复应用幂等。整张量没有变化就不写入分片。这里同时存在两个 base：trainer 手里上一版的 CPU 字节快照，与每台 host 本地 checkpoint 里已应用的版本，所以版本 N 只能基于 N−1 生成和应用，不是任意两个 checkpoint 之间的无状态 diff。版本链从 v0 开始：v0 是 engine 的 `server_args.model_path`（补丁 `SchedulerWeightUpdaterManager.pull_weights` 把它作为 `base_dir` 传给 `local_checkpoint.pull`）；可更新模型的 `model_path` 缺省就是 `--hf-checkpoint`（`slime/backends/sglang_utils/sglang_config.py::ModelConfig.resolve` 在两者不同时把 `update_weights` 推断为 False），而 trainer 快照也从 `--hf-checkpoint` 读，所以前提是两处的字节相同，路径可以不同（external engine 不核对 `model_path`）；否则第一个增量在主机侧 checksum 失败（本页推断）。第一次调用只捕获快照并让每台 host `pull_weights(0)` 物化本地 base，不发布、也不递增版本；此后 v1 声明 base 0、v2 声明 base 1。新 host 执行 `pull(2)` 时从 2 往回找最近的完整版本，找到 v0 就用 model path 重置本地目录，再依次应用 v1、v2；已在 1 的 host 只应用 v2。

### 2.2 从最小实例到整个同步系统

下面各组件的"为何"是本页依据源码形态与失败路径重建的理由（标"本页推断"），源码或文档写出的理由单独注明；"怎样"与"代价"以冻结基线为准。SGLang 侧行为统一按 §1.4 的三种标签标注，逐端点的交接表在 §2.2.8。

#### 2.2.1 入口：updater 选择、恢复与连接

**职责。** `MegatronTrainRayActor.init` 在 actor（非 critic）上调用 `slime/backends/megatron_utils/update_weight/__init__.py::create_weight_updater`：`update_weight_mode == "delta"` 时断言非 colocate、transport 为 disk，选 `UpdateWeightFromDiskDelta`；否则 transport 为 disk 选 `UpdateWeightFromDisk`；否则 colocate 选 `UpdateWeightFromTensor`；剩下的必须是 full + nccl，选 `UpdateWeightFromDistributed`，其他组合以 `unsupported weight sync mode/transport` 断言失败。工厂按分支延迟 import 各 updater 模块，构造后把 `weight_version` 设为 `args.update_weight_start_version`（缺省 0）；`tests/test_update_weight_factory.py::test_create_weight_updater_selects_implementation` 用四组参数锁定了这四个分支与起始版本的回写。updater 拿到的 `weights_getter` 指向 CPU `actor` tag（只有张量 updater 实际读它，§5.1）。`update_weights` 在 `--debug-train-only` 或 `--debug-rollout-only` 下直接返回；开 `--use-fault-tolerance` 时 rank 0 请求 `RolloutManager.recover_updatable_engines`，所有 rank 过 Gloo barrier；然后取 `get_updatable_engines_and_lock` 的六元组（engines、Ray `Lock`、新建 engine 数、每 engine GPU 数、GPU 偏移、并行配置）；`offload_train ∧ use_critic ∧ ¬colocate` 时整轮 wake/sleep 并重连，其余 offload 情形只 `reload_process_groups`、结束时再销毁；没有可更新 engine 且不需要重连就记日志返回；有新 engine 或需要重连时调用 `connect_rollout_engines`、barrier，rank 0 清零新 engine 计数；最后在 `torch_memory_saver.disable()` 区间里调用 `weight_updater.update_weights()`，`keep_old_actor` 时再按 `update_weights_interval` 是否等于 1 选择 `rollout_actor → old_actor` 的轮转方式（归 [[14_slime_megatron_training_analysis|Megatron 训练]]）。

**为何。** 选择在构造时一次定死，避免运行中切换 updater 时新旧 group、快照、版本号并存（本页推断）。delta 的两条限制，源码各有原话：只配 disk，是因为工厂注释说每个 engine 的 `/pull_weights` 要把发布的增量应用到它跨越的每台 host 的本地 checkpoint，再走普通 `update_weights_from_disk`；不配 colocate，是因为参数校验说共卡传的只是 CUDA IPC 句柄，快照、做差、编码"纯属开销"。未知组合断言失败而不是回退到"最接近"的 updater。

**代价与边界。** 参数校验先于工厂抛 `ValueError`：disk 需要 `--update-weight-disk-dir`，delta 需要 disk、非 colocate、`--update-weight-local-checkpoint-dir`，`--release-train` 需要 full + disk 与 `--save`。`RolloutManager._get_updatable_server` 只返回第一个 `update_weights=True` 的模型，docstring 自承多模型权重更新尚未支持，冻结的 ref/reward 模型不进入这条路径。

#### 2.2.2 拓扑无关重组：两套迭代器

**职责。** `HfWeightIteratorDirect` 服务张量路径与整份磁盘（`slime/backends/megatron_utils/hf_checkpoint_saver.py::save_hf_model_to_path` 每次构造它，并传 `transform_ue8m0=False`）。构造时 `_get_megatron_local_param_infos` 收集本地参数名、形状、dtype 与 TP 属性，沿 PP 组 `all_gather_object` 交换（重复名取最小 `src_rank`，处理 MTP 的虚拟 PP），再沿 EP 组补上其他 EP rank 的专家，按名字排序后经 Gloo 在全 world 核对名字、形状、dtype。`get_hf_weight_chunks` 对每个桶调用 `_get_megatron_full_params`：持有者把本地分片搬上设备、其余 rank 分配空张量，PP 组内按 `src_rank` 广播，专家再沿 EP 组广播，最后 `all_gather_params_async` 先对整桶发起全部异步 TP all-gather、统一 wait、再拼接；结果是每个训练 rank 都持有该桶全部完整参数。之后张量路径在每个 rank 上 `convert_to_hf`；整份磁盘传 `should_convert_chunk=lambda _: is_writer_rank`，只有每个节点的首个 rank（writer rank）转换，其余 rank 只参加集合通信，注释说明 writer 要看到每个桶，是因为 `q_a_proj` 与 `kv_a_proj` 这类要成对输出的参数可能落在相邻桶。NCCL 与增量路径用另一套：`_iter_non_expert_chunks` 逐参数同步 `all_gather_param`，只有 PP 源转换并装桶，其余 rank（包括 DP 副本）做完 TP all-gather 就跳过；`_iter_expert_chunks` 攒批后由 `_ep_gather_and_convert` 做 EP all-gather，名字列表先 `all_gather_object` 并断言各 EP rank 数量一致。

**为何。** 在线 NCCL 只需要一个发送者，没必要让每张卡都物化全部参数；共卡 IPC 恰好相反，每张卡都要把完整桶交给同卡 SGLang rank，所以张量路径让所有 rank 都拿到完整桶（本页推断，判据是谁需要完整张量）。

**代价与边界。** 守卫只在 NCCL 与增量用的 `all_gather_param` 里：断言参数带 `tensor_model_parallel`，`partition_stride` 只接受 1 或 fc1 的 2。直接迭代器用 `getattr` 给缺失属性填默认值（非 TP、stride 1），`all_gather_params_async` 也不检查 stride，所以张量与整份磁盘路径上缺属性的分片参数会以本地分片原样交付，非 fc1 的 stride-2 参数会被普通拼接，都不报错（§5.1）；反过来，各 rank 参数名、形状、dtype 一致的核对只在直接迭代器里。buffer 只同步名字含 `expert_bias` 的一类，其余被跳过（`named_params_and_buffers` 里的 `TODO shall we handle (almost) all buffers`）。`convert_to_hf` 按模型名分派到 qwen2、qwen3moe、deepseekv3、glm4 等转换器，不认识的模型抛 `ValueError`；视觉塔（`model.visual.`）原样透传，其余先去 vocab padding 再按 `quantization_config` 量化。FP8 块量化且 rollout 运行时要求 UE8M0 scale 时，`transform_ue8m0` 决定是否把 scale 打包成运行时格式：写盘的整份路径传 False，在线两条与增量（沿用 NCCL 迭代器、最终也写盘）走缺省 True；增量写盘时 SGLang 磁盘加载器怎样处理已打包的 scale 未核（格式细节归 [[22_slime_low_precision_training_rollout_analysis|低精度训练与 rollout]]）。完整逻辑张量解耦的是下表这些维度，不是任意拓扑转换：

| 拓扑维度 | 通用路径怎样消除差异 | 仍然存在的约束 |
|---|---|---|
| 训练 DP/CP | 权重在这些维度上复制，不影响 HF 形状 | 只选一份语义一致的快照（PP 源或 `src_rank`） |
| 训练 TP → 推理 TP | TP all-gather 成完整张量，loader 再按推理 TP 截取 | 目标维度可整除，或 loader 明确支持 padding 与特殊布局 |
| 训练 PP → 推理 PP | 各 stage 的层按全局层号改名后交给 loader | 层归属、名字映射与推理 engine 的 rank 顺序一致 |
| 训练 EP → 推理 EP | 恢复逐专家张量后加载；合格时走 rank 内定向路由 | 专家数与布局、EP rank 映射；EPLB 等动态布局不支持定向路由 |
| 量化与融合格式 | 转换器产出 loader 认识的名字与张量 | 不是任意 dtype 与量化方案都能互转 |

#### 2.2.3 NCCL 分离数据面

**职责。** `connect_rollout_engines` 在每个 PP 源上建 group `slime-pp_{pp_rank}`：主机取本机 IP 与空闲端口，world = 所有 engine GPU 数之和 + 1，engine i 以 `rank_offset = 前面 engine 的 GPU 累计数 + 1` 加入（异构 TP 的 prefill 与 decode 各占不同数量的 rank），重连时先销毁旧 group。group 的后端由 `slime/utils/accelerator/__init__.py::weight_update_backend` 决定：CUDA 为 `nccl`，MUSA 为 `cpu:gloo,musa:mccl`；本页的"NCCL"指 CUDA 缺省。每个桶 `_update_bucket_weights_from_distributed`：自旋获取 `rollout_engine_lock`（0.1 秒重试），对每个 engine 发 `update_weights_from_distributed` RPC（names、dtypes、shapes、group 名、版本号字符串），同时每个张量 `dist.broadcast(src=0, async_op=True)` 并逐个 wait，`ray.get` 等 RPC 返回后清桶、释放锁。SGLang 端（上游）先分配本进程缓冲、在同一 group 上接收广播，再 `load_weights`；上游 `SchedulerWeightUpdaterManager.update_weights_from_distributed` 只调 `tp_worker`，EAGLE draft runner 不随这个入口更新（五个补丁都没有改它，transport 覆盖表见 [[21_slime_speculative_decoding_mtp_analysis|投机解码与 MTP]] §2.1.2）。

**为何。** 这个临时 group 只是"一个训练源 + 全部推理 rank"的数据面，不合并两边原有的并行组，也不参与 forward/backward；训练侧已经还原出完整张量，group 只负责运输（本页推断）。锁的理由源码写明是防止通信死锁：多个 PP 源各有自己的 group，却指向同一批 engine，交错广播会让 engine 同时卡在两个集合通信里。

**代价与边界。** PP 源之间串行；每桶一次 RPC 往返；sleep 时（有 critic 且非 colocate）`disconnect_rollout_engines` 销毁 group，下次再建。这条路径只用上游已有的端点（§2.2.8），接收缓冲属于 SGLang 进程自己，不存在共卡 IPC 那种跨进程显存生命周期问题（本页推断）。

#### 2.2.4 共卡 CUDA IPC 数据面

**职责。** `UpdateWeightFromTensor.connect_rollout_engines` 从头扫描 engine，`gpu_offset + gpu_count` 超出 actor GPU 数（`actor_num_nodes × actor_num_gpus_per_node`）就停止，前面的 engine 是共卡前缀，后面的全部是越界后缀；Gloo 组只在首次连接时按真实偏移建（组成员就是 engine 占用槽位上的训练 rank，源是组内首个 rank），偏移缺省时按紧密排列推断。每个桶 `_send_to_colocated_engine`：按 dtype 分组（`FlattenedTensorBucket` 支持多 dtype 时合成一组）拍平成一个新的设备张量加 name/shape/dtype/偏移元数据，经 `MultiprocessingSerializer` 序列化成字符串，`gather_object` 送到 Gloo 源；源对每个桶位发一次 `update_weights_from_tensor`（`load_format="flattened_bucket"`、版本号），某个 rank 缺该桶位时用空桶补齐；SGLang 的 TP rank k 取列表第 k 个描述符（上游 `BaseTpWorker.update_weights_from_tensor`）。越界后缀由 `DP=TP=PP=0` 的全局源 rank 建 NCCL group `slime`，逐桶 RPC + 广播补发，但不持 Ray 锁（只有一个源，不会交错）。每个桶在 `ray.get` 后 `del`、`ipc_collect()`、`empty_cache()`；定向路由每批 `ray.get` 之后都过一次 Gloo barrier；两遍都结束后再过一次 Gloo barrier 并再清一次。

**为何。** 共卡时两个进程在同一张卡上，句柄比字节便宜得多：Gloo 与 Ray 只搬描述符，payload 从训练进程的设备张量直接拷进 SGLang 参数。每桶新建拍平张量而不复用，slime 源码注释（`_build_flattened_tensor_data`）给了原因：SGLang 在把 GPU 拷贝排入队列后就返回 HTTP/Ray 响应，并不保证设备同步，立即覆写同一块生产者缓冲会与消费者的拷贝竞争。这条注释描述的是上游语义，镜像里的实际行为分两种：

- **上游**：`ModelRunner._update_weights_from_flattened_bucket` 用元数据重建零拷贝视图后调用 `load_weights` 就返回；`SchedulerWeightUpdaterManager.update_weights_from_tensor` 随后只在 TP 组上做 CPU barrier。RPC 返回时拷贝可能仍在 SGLang 的 CUDA 流上排队，注释所说的竞争成立。
- **补丁**：`sglang-deterministic.patch` 在同一函数的 `load_weights` 之后加了 `torch.cuda.synchronize()`（仅 `self.device == "cuda"`），补丁注释写明理由：flattened tensor 可能由训练进程持有的 CUDA IPC 内存承载，RPC 响应会释放生产者侧缓冲，下一个桶若复用这块存储就会破坏刚加载的权重。叠加上游的 TP 组 barrier，Gloo 源的 `ray.get` 返回时该 engine 每个 TP rank 的拷贝都已在设备上完成（补丁 + 上游契约的组合推断）。

所以在打了补丁的缺省镜像里，slime 的"每桶新建缓冲"是保守的第二道防线；在不打补丁的镜像里，它只避免了直接覆写同一个张量，块被 `ipc_collect` 与 `empty_cache` 释放后能否被下一个桶复用，仍取决于 PyTorch CUDA IPC 的引用计数与消费者何时关闭句柄（依赖侧，未核）。CPU `actor` tag 在这里是显存交接的稳定快照，不是 IPC 介质：一次更新的数据路径是 pinned CPU 快照 → 训练进程设备临时桶 → IPC 句柄 → SGLang 映射同一块显存 → `load_weights` 拷入本 rank 参数分片，整轮看有一次 D2H2D，但最后一段是 GPU→GPU（本页推断，依据 `weights_getter` 与 memory saver 区间）。共卡本身是分时驻留：actor 与 rollout 从槽位 0 起复用同一批 placement group 资源，colocate 缺省同时开 `offload_train` 与 `offload_rollout`（生命周期归 [[14_slime_megatron_training_analysis|Megatron 训练]]）。

**代价与边界。** slime 的 wrapper `SGLangEngine.update_weights_from_tensor` 注明 HTTP 只 post 元数据、真实权重直接从 GPU 拷贝，模型必须已在 GPU 上；`train.py` 在 `offload_rollout ∧ ¬release_train` 时先 `onload_weights` 再更新，与这条前提相合（本页推断）；`train_async.py` 断言不支持 colocate，所以这条数据面只出现在 `train.py`。每张卡在暂停窗口里临时持有完整桶；只有 Gloo 源的 `ray.get` 真正等到消费者返回，非源 rank 手里没有 ref，删除后的块能否被复用取决于上面的 IPC 引用计数（源码注释只说释放"消费者已关闭句柄"的条目）。空桶的 rank 仍须进入 `gather_object`，因为那是组内集合通信（`tests/test_empty_colocated_weight_bucket.py`）。前缀判定遇到第一个越界 engine 就结束，要求共卡 engine 排在前面。越界后缀不在 pause/flush/量化前后处理的名单里（§5.1）。

#### 2.2.5 MoE rank 内定向路由

**职责。** `configure_expert_routing` 在每次 `connect_rollout_engines` 时决定是否启用：存在越界 engine、没有共卡 engine、各 engine 的 SGLang 并行配置不一致、准入条件不满足（推理 PP=1、推理 EP>1、未开 EPLB、无冗余专家、专家初始位置 trivial、未开 elastic expert backup、训练专家 TP=1、每个 engine 的 GPU 数等于 `EP × MoE-DP`），或规划中抛 `AttributeError/TypeError/ValueError`（专家数不能整除推理 EP、元数据没覆盖每层每个专家的 fc1/fc2、单个专家包超过阈值、engine 越出训练 world）时，rank 0 记一条 "Disable rank-local expert update" 日志并返回通用桶表；没有 `ParamInfo` 桶表或桶里没有匹配的 routed expert 时静默返回。启用时稠密参数重新装桶，专家按层组织成传输组。更新时 `_update_expert_weights` 先 Gloo barrier、再 WORLD barrier（注释：子集批量 P2P 之前先初始化 WORLD），逐批 `_prepare_expert_weight_batch`：源 rank 从 CPU tag 拷进按 `(dtype, shape)` 跨层复用的 staging 缓冲并 `isend` 给每个非自身目标，目标 `irecv`，`batch_isend_irecv` 后 wait，目标 rank 本地 `convert_to_hf`，再走 §2.2.4 的 IPC 交给同卡 SGLang EP rank；每批 `ray.get` 之后都有 Gloo barrier 与 `accelerator.synchronize()`，后者同步的是训练进程自己的设备，不等待 SGLang 进程的拷贝（本页推断）。

**为何。** 见 §2.1.4：EP 切的是专家集合，满足条件时每个专家在持有者上已经完整，通用路径的"每张卡物化全部专家"是纯开销。

**代价与边界。** 正则只匹配 `module.module.decoder.layers.*.mlp.experts.linear_fc{1,2}.weight*`：router/gate、shared expert、dense 层、MTP 层与带 `language_model.` 前缀的 VLM 专家仍走通用桶。它减少的是物化与交付量，不取消 HF 命名转换，也不让 SGLang 长期引用训练进程的显存。staging 缓冲跨层复用，但跨批仍是 IPC 交接，拷贝完成语义与 §2.2.4 相同。

#### 2.2.6 整份磁盘数据面

**职责。** `UpdateWeightFromDisk.update_weights`：version+1，目录 `update_weight_disk_dir/weight_v{06d}`；rank 0 `rmtree` 清掉同名版本目录的残留后 Gloo barrier，每个 rank 自己 `mkdir`（注释：非 POSIX 共享文件系统在提交前不一定让别的 rank 看到这次 mkdir），`save_hf_model_to_path` 在 rank 0 清掉目标目录里的旧权重文件并从 `--hf-checkpoint` 拷来非权重文件（config、tokenizer 等），再用 `HfWeightIteratorDirect` 重组，按 chunk 轮流交给各节点的 writer rank 写 safetensors 分片并汇总 index，barrier，所有 rank 各自调用可选的 post-write hook（签名 `hook(args, version_dir, rollout_engines)`，注释：对象存储挂载没有跨 host 的读后写一致性），再 barrier。reload 由 `RayTrainGroup.update_weights` 接管：它按自己的计数器 +1 定出版本目录，actor RPC 返回后记下该版本，`release_train` 时杀掉训练 actor，然后 `_reload_rollout_weights_from_disk`：`offload_rollout` 时先 `onload_weights`，取可更新 engine（没有就按 keep-files 决定是否删目录后返回），设了本地目录就先 `pull_weights(version)`（补丁端点），再 pause → flush → `update_weights_from_disk(model_path, weight_version)` →（`ci_test`）逐 engine `get_weight_version`，不等就抛 `RuntimeError` → 除非 `--update-weight-disk-keep-files` 否则删目录 → continue。上游的 disk reload 走 `load_weights_and_postprocess`，并按请求缺省 `flush_cache=True` 在加载成功后再清一次缓存。

**为何。** 写盘由训练侧完成、reload 由 driver 侧的 `RayTrainGroup` 编排，注释给了原因：训练侧生命周期要能决定 reload 时 Megatron actor 是否还活着，`release_train` 正是先杀 actor 再 reload。直接把文件写进共享目录就让 engine reload，reader 可能看到半个版本；所以整份用独立版本目录加两道 barrier 与 hook，增量再加原子替换、base_version、checksum 与 host 锁（本页推断）。整份 checkpoint 让训推只经共享目录与 HTTP 耦合，external engine、不同型号或厂家的 GPU 都能接，前提是 SGLang 支持对应硬件与模型格式（官方 external engine 文档）。

**代价与边界。** actor 端 `weight_version` 与 `RayTrainGroup._disk_weight_version` 是两个计数器，`create` 时把后者写回 `update_weight_start_version`，新 actor 从同一处起算（版本归属与 actor 生命周期归 [[11_slime_ray_control_plane_analysis|Ray 控制面]]）。每次写完整权重，写放大随同步频率线性增长；共享目录必须对训练端与 engine 路径相同；`--hf-checkpoint` 必须是本地目录且不能与版本目录相同，否则 `save_hf_model_to_path` 抛 `ValueError`；`release_train` 下训练 actor 在 reload 前已被释放，reload 仍能从版本目录或主机本地副本完成，但版本目录在 reload 后被删除，除非 `--update-weight-disk-keep-files`。不设本地目录且不开 `--ci-test` 时，这条路径只用上游端点；CI 读回依赖 `sglang.patch`（§2.2.8）。

#### 2.2.7 增量磁盘数据面

**职责。** `UpdateWeightFromDiskDelta` 继承 NCCL updater 的两个 chunk 迭代器但不建 group（注释：主机侧应用由 host 级 flock 串行化，用不到 NCCL 那把锁）。首次调用 `_capture_baseline`：rank 0 清空整个 delta 目录（注释：上一次运行的版本会应用到错误的 base 上），跑 hook，异步发 `pull_weights(target_version=0)`；所有 PP 源从 `--hf-checkpoint` 的 safetensors 按名字读字节作快照，缺失的张量回退为当前 gathered 值并告警（docstring：以 hf_checkpoint 为种子，是为了在 Megatron→HF 往返裁掉 embed/lm_head 的 vocab padding 行时仍保证快照等于 engine 的 base）；rank 0 最后 `ray.get` 这批 pull。此后每次：version+1；`_encode_delta` 在 PP 源上开 `max(4, min(2×NUM_WORKERS, 32 GiB / 最大张量字节))` 个 pinned 缓冲做 D2H 暂存（memlock 不足时退回可分页 `.cpu()`），主循环把张量拷进缓冲后提交线程池，工作线程做 xor 或 overwrite、zstd level 1、新状态 checksum，在途任务上限 `2 × NUM_WORKERS` 形成反压，每个结果把新字节写回快照作为下一版的 base；`_write_delta_files` 只给有变化的 rank 编号 `model-{offset:05d}-of-{total:05d}.safetensors`，以 `.tmp` → flush → fsync → `os.replace` 原子写入，rank 0 写 index（version、base_version、delta_encoding、`compression_format: zstd`、checksum_format、weight_map）；`_reload_engines` 跑 hook，rank 0 依次 `pull_weights(version)` → pause → flush → `update_weights_from_disk(local_checkpoint_dir, version)` → continue；`_record_metrics` all-reduce 变化字节、总字节与写出字节，记 `perf/update_weights_density` 与 `perf/update_weights_wire_bytes`。主机侧的 `/pull_weights` 由补丁 `docker/patch/latest/sglang-pull_weights.patch` 实现：每个 scheduler rank 都在 `SchedulerWeightUpdaterManager.pull_weights` 里以 `base_dir=server_args.model_path` 调用 `python/sglang/srt/weight_sync/local_checkpoint.py::pull`，host 级 flock 与已应用标记让同机多 rank 只做一次；从目标版本往回找最近的完整版本（找不到就用 model path 重置本地目录，完整版本按文件大小核对拷贝），再逐版应用：base_version 不等于本地标记就抛 out-of-order，逐张量解压、在 mmap 区域原位 xor 或按位置覆写、校验 checksum，不符就抛错；成功要 TP 组内所有 host 都返回成功。仓内可运行的配方是 `examples/delta_weight_sync/run-glm4.7-30B-A3B-delta.sh`（GLM-4.7-Flash、非共卡、两节点，入口 `train.py`）。

**为何。** 官方 delta 文档把场景写成跨集群或跨数据中心的训推解耦，那里每次写整份权重是主要开销；只有一次同步改变的字节占比不高时增量才划算（本页推断）。版本目录自描述（index 有无 `delta_encoding`），同一个 `/pull_weights` 同时服务整份与增量，reload 走原生路径，权重加载代码从不接触增量格式。`pull_weights` 这个 RPC 本身只是控制面，真正的 payload 是 host 读取版本目录并在本地 base 上应用。

**代价与边界。** 训练侧的重组、D2H 与逐字节比较都是全量的，还常驻一份完整字节快照（普通 numpy 数组；参数帮助写的 "pinned-CPU snapshot" 实际只有暂存缓冲是 pinned，以源码为准）；每台 host 要有完整本地 checkpoint；没有按 density 自动回退整份的逻辑。trainer 不清理旧版本目录，一次运行内版本持续累积，新 host 要从 v0 回放整条链；官方文档说整份版本会重置链条、旧增量"可以被清理"，这是 `pull` 的能力，增量 updater 本身只发布增量版本，也不做清理。作为对照，verl 删除了同样"先 full gather 再 diff"的 plain delta，改为在每个 rank 的本地 shard 上做差的 `delta_sharded`（见 [[21_verl_weight_publication_analysis|verl 权重发布]]）；slime 的增量优化的是跨 host 的 wire 与存储，不减少训练侧 gather（对照判断）。

#### 2.2.8 提交协议与 SGLang 依赖边界

**职责。** 把上面的数据面收束成六个可以各自失败的动作：①快照：driver 在训练与保存之后才调用更新，actor 在训练末尾已 `backup("actor")`；②拓扑无关：先还原完整 HF 张量；③暂停与清缓存：`pause_generation`、`flush_cache`；④完整搬运：每遍或每批之后的 barrier、IPC 源的 `ray.get`、磁盘的版本目录与 barrier；⑤量化后处理与版本号：compressed-tensors 的 restore 与 post-process，版本号随 RPC 或 reload 交给 engine；⑥确认后才 continue。slime 侧能证明的是它发了什么：`pause_generation` 与 `continue_generation` POST 空 JSON；`flush_cache` 是 GET，非 200 时每秒重试、最多 60 次后抛 `TimeoutError`，注释说有在途请求时 flush 不会返回 200；更新 RPC 带 `weight_version` 字符串，逐桶 RPC 不传 `flush_cache`，wrapper 缺省为 False；其余 POST 端点经 `_make_request` 的 `raise_for_status` 把非 2xx 变成 `HTTPError`。

**为何。** 若不先停止生成，一次 forward 可能跨过一半新层一半旧层；只换权重不清 KV，后续 decode 会把旧参数算出的缓存与新参数混用；所以"每个桶的 RPC 成功"不等于"服务已提交"，全部完成并 flush 后的 continue 才是对外提交点（本页推断）。源码逐点写了单个组合为什么被禁止，但没有在任何一处写下"同步应组织成带版本的提交事务"；这六个动作是本页据状态转移与失败路径重建的组织方式。框架无关的 weight publish 事务阶段（prepare、transfer、install、commit、retire）见 [[01_posttraining_infra_mechanism_analysis|后训练 Infra 核心机制]] 的 weight publish 协议一节。

**依赖边界：逐端点交接。** slime 调用的每个端点，按"上游 v0.5.15.post1 有没有、哪个补丁改了它、缺补丁时哪条路径受影响"列出。缺补丁一列是按 slime 调用点与上游路由推断的失效面，未运行验证。

| slime 调用 | 上游 `0b3bb0cb` | 补丁（`docker/patch/latest/`） | 缺补丁时 |
|---|---|---|---|
| `/pause_generation`、`/continue_generation` | 有；`PauseGenerationReqInput.mode` 缺省 `abort`：`TokenizerManager.pause_generation` 中止全部请求并等 `model_update_lock` 空闲；`in_place` 保留旧 KV，`retract` 允许 flush 后重算；被中止请求在 slime 侧怎样处理归 [[13_slime_sglang_rollout_engine_analysis|SGLang rollout 引擎]] | — | 不受影响 |
| `/flush_cache` | 有；`Scheduler.flush_cache` 只在完全空闲时重置 radix 与 KV 池，否则 HTTP 400 | — | 不受影响 |
| `/destroy_weights_update_group` | 有；NCCL 与越界后缀的 group 在重连或 sleep 时经 `disconnect_rollout_engines_from_distributed` 销毁，slime 的 `SGLangEngine.destroy_weights_update_group` 吞掉请求异常（刚建的 engine 还没有 group） | — | 不受影响 |
| `/init_weights_update_group`、`/update_weights_from_distributed` | 有；更新失败时 `ModelRunner.update_weights_from_distributed` 返回"模型可能已被部分更新、应丢弃整套权重" | — | NCCL 路径不受影响 |
| `/update_weights_from_disk` | 有；两条磁盘数据面都调用；`TokenizerManager.update_weights_from_disk` 经 `ModelRunner.update_weights_from_disk` 的 `load_weights_and_postprocess` 加载，请求 `flush_cache` 缺省 True，成功后再清一次缓存 | — | 不受影响 |
| `/update_weights_from_tensor`（`flattened_bucket`） | 有；`_update_weights_from_flattened_bucket` 排入拷贝即返回，不做设备同步 | `sglang-deterministic.patch`：`load_weights` 后 `torch.cuda.synchronize()` | 共卡 IPC 仍可运行，但 RPC 返回不再意味着拷贝完成，退回 §2.2.4 的上游语义（静默，不报错） |
| `weight_version` 写入 | `TokenizerControlMixin.update_weights_from_tensor` 等在全部 worker 成功后才调用 `_update_weight_version_if_provided` | — | 不受影响 |
| `/post_process_weights` | 无此路由 | `sglang.patch` 新增路由、`ModelRunner.post_process_weights` 与 compressed-tensors 的 `restore_weights_before_loading` 分派 | 训练侧 compressed-tensors 在线量化只支持 INT4（见 [[22_slime_low_precision_training_rollout_analysis|低精度训推]] §5.4）；这类模型在 NCCL 与共卡 IPC 路径上 pause 之后的第一个 restore 请求就抛 `HTTPError`，engine 停在暂停态；磁盘路径不调用它，上游 disk reload 自带 `load_weights_and_postprocess` |
| `/get_weight_version` | `/get_weight_version` 与 `/weight_version` 共用一个处理函数，直接返回 404（已弃用，指向 `/model_info`） | `sglang.patch` 改为返回 `model_info` 里的 `weight_version` | 整份磁盘在 `--ci-test` 下读回时抛 `HTTPError`（`tests/test_full_disk_weight_update.py` 正是开 `--ci-test` 的用例）；不开 CI 不受影响 |
| `/pull_weights` 与 `--sglang-custom-pull-weights-pre-read-hook` | 无此路由，也无此 ServerArgs 字段 | `sglang-pull_weights.patch` 新增路由、`SchedulerWeightUpdaterManager.pull_weights`、`weight_sync/local_checkpoint.py` 与服务参数 | 增量模式在首次 `_capture_baseline` 的 `ray.get(pulls)` 处失败；整份磁盘设了本地目录时在 pause 前失败；hook 服务参数不存在，而 slime 两个解析阶段都容忍未知参数（`parse_known_args` 与 `ignore_unknown_args`），传入会被静默忽略（按解析注释推断） |
| `/weights_checker`（`--check-weight-update-equal`） | 有；`utils/weight_checker.py` 支持 snapshot、reset_tensors、compare | — | 不受影响 |

其余几条上游契约也在这条边界上：列并行 loader `ColumnParallelLinear.weight_loader` 按 `tp_rank` 截取完整张量，另有 `use_presharded_weights` 分支，所以"推理引擎只能加载完整张量"并非普遍原理；`FlattenedTensorBucket.reconstruct_tensors` 按元数据在拍平张量上切出零拷贝视图；未暂停的 engine 上，更新 RPC 先取 `model_update_lock` 写锁、等在途请求结束，桶与桶之间锁会释放。在暂停窗口内，版本号在第一个桶成功时就已写成新值，但此时没有请求在跑，continue 之后新请求看到的版本号与权重一致（本页推断）；越界 engine 不在暂停名单里，这一推断对它不成立（§5.1）。slime 源码与上游源码都无法证明 flush 覆盖了所有缓存层（例如 hicache 的主机侧缓存，其 release/resume 由 `sglang-release_hicache.patch` 改写，归 [[13_slime_sglang_rollout_engine_analysis|SGLang rollout 引擎]]）。

**镜像闸门。** 上表的"补丁"列只在镜像构建时生效。`docker/Dockerfile` 以 `slimerl/sglang:${SGLANG_IMAGE_TAG}`（缺省 `v0.5.15.post1-cu129`）为底，`ARG PATCH_VERSION=latest` 决定从 `docker/patch/${PATCH_VERSION}/` 复制补丁；`ARG ENABLE_SGLANG_PATCH=1` 为 1 时在 `/sgl-workspace/sglang` 里按 `sglang.patch` → `sglang-top_p.patch` → `sglang-release_hicache.patch` → `sglang-pull_weights.patch` → `sglang-deterministic.patch` 的顺序逐个 `git apply --check` 再应用：目录里不存在的补丁跳过，存在但不能干净应用就让构建失败。这个开关旁边的注释写明是为 GB200/GB300 临时跳过打补丁、要求用户自带 SGLang 版本。补丁是叠加的：`sglang-deterministic.patch` 对 `model_runner.py` 的 hunk 以 `sglang.patch` 应用后的文件为底（前者的原始 blob `159ef8fc` 正是后者的结果 blob），所以顺序不能换。基线上 `docker/patch/latest/` 与 `docker/patch/v0.5.15.post1/` 的 7 个补丁逐字节相同；`build_conda.sh` 固定 `PATCH_VERSION=v0.5.15.post1`、按同样顺序应用。更早的补丁目录（如 `v0.5.12.post1`）只有 `sglang.patch` 与 `megatron.patch`，没有 pull_weights 与 deterministic 补丁；`docker/Dockerfile.gb10` 装的是 sglang 0.5.9 wheel、不打任何 SGLang 补丁，属于上表"缺补丁"一列的情形，但端点行为要按 0.5.9 另核（未核）。v0.3.1 发布时 `latest` 与 `v0.5.15.post1` 两个目录都还没有 deterministic 补丁，那个版本构建的镜像在共卡 IPC 上是上游语义。

**代价与边界。** 版本确认的强度因路径而异：只有整份磁盘在 `ci_test` 下逐 engine 读回 `get_weight_version`，而且这个读回本身依赖补丁；在线路径与增量路径都没有读回核对，也没有逐次等值检查（§4.1）。补丁与 slime 代码的版本是分开演进的：镜像构建时 `git apply --check` 只保证补丁能应用，不保证 slime 调用的端点集合与补丁提供的一致，这层对应关系没有运行期探测（本页推断）。

### 2.3 变体：同一实例在五条选择轴上

| 选择轴 | 枚举依据 | 变体 | 本例的表现 | 压力与上限 |
|---|---|---|---|---|
| updater | `slime/backends/megatron_utils/update_weight/__init__.py::create_weight_updater` 的 `update_weight_mode` × `update_weight_transport` × `colocate` 分支 | NCCL / 张量 IPC / 整份磁盘 / 增量磁盘 | 分别见数据面 ②–⑤ | 网络与共享存储的带宽；delta 只配 disk 且非 colocate |
| 共卡前缀 | `UpdateWeightFromTensor.connect_rollout_engines` 的 `offset + count` 判定 | 全部共卡 / 前缀共卡 + 越界 NCCL | e0、e1 走 IPC，e2 走 NCCL `slime`（world 3） | 越界 engine 不被 pause/flush |
| 专家传输 | `configure_expert_routing` 的准入 | 通用桶 / rank 内定向 P2P | 定向 12 份、每卡交付 4 个专家；通用 24 份、8 个 | 准入条件；单包必须不超过阈值 |
| 增量编码 | `--update-weight-delta-encoding` 的 choices | xor / overwrite | 8 与 14 字节（压缩前） | wire 大小对幂等性 |
| 磁盘读取点 | `--update-weight-local-checkpoint-dir` 是否设置 | 直接读共享目录 / 先 pull 到主机本地盘 | 整份可选，增量必需 | 官方 external engine 文档：本地目录让共享文件系统每台 host 只读一次，而不是每个 rank 读一次；本地目录依赖补丁端点 |

同一关注点的兄弟轴：多模型只更新第一个 `update_weights=True` 的 server，冻结模型在 offload 后从 CPU 备份恢复；EPD 部署里 encoder 组与 language-only 的 prefill/regular 组属于同一个模型（`slime/backends/sglang_utils/disaggregation.py` 两阶段启动），`RolloutServer.engines` 不按 worker type 过滤，updater 也不过滤，所以 encoder engine 会进入更新名单；但 slime 用上游 `python/sglang/srt/disaggregation/encode_server.py` 启动它，该服务只暴露 `/encode`、`/send`、健康检查与 profile 等路由，没有 pause、flush 或任何权重更新端点，五个补丁也没有改它，因此视觉塔权重不会经这条路径刷新，同步大概率在第一个请求处抛 `HTTPError`（源码路径推断，未运行，仓内也没有覆盖 EPD 权重同步的测试）；external engine 由 `--rollout-external-engine-addrs` 接入，权重同步方式仍由上面的轴决定，官方文档建议不能建 NCCL group 时用 disk；`--release-train` 强制 full + disk，因为 reload 时训练 actor 已被释放（归 [[11_slime_ray_control_plane_analysis|Ray 控制面]]）；新的 rollout backend 必须实现同一组 engine 端点才能接入这些 updater（归 [[19_slime_rollout_backend_extension_analysis|rollout backend 扩展]]），其中 `/post_process_weights`、`/pull_weights` 与读回版本在 SGLang 上是补丁提供的；同步 group 的通信后端随加速器变化（CUDA `nccl`、MUSA `mccl`）；训练侧只有 Megatron 一个后端。Megatron-LM 自己的 RL 运行时走的是另一条路：它在 Megatron 训练模型与 Megatron 推理模型之间用 `megatron/core/resharding/refit.py::swap_model_weights`（Megatron-LM 仓库路径）换权重，不经过 HF 张量与 SGLang（见 [[33_megatron_rl_runtime_analysis|Megatron RL 运行时]]，该页按 `NVIDIA/Megatron-LM@85902ef5` 分析，比 slime 镜像钉的 `1dcf0daf` 新；`swap_model_weights` 在 `1dcf0daf` 已存在）；slime 仓内没有调用 `megatron.rl`。

### 2.4 整体开销

| 维度 | 来源 | 评估状态 |
|---|---|---|
| 计算 | 每次同步全量 TP/PP/EP 重组与 HF 转换；增量额外逐字节比较、zstd、checksum | 源码可见，未测量 |
| 显存 | 每个桶在训练 GPU 上物化完整参数；共卡时每张卡临时持有完整桶；专家 staging 按阈值封顶 | 源码可见 |
| 主机内存与存储 | 增量的整份字节快照与 pinned 暂存池；每台 host 的完整本地 checkpoint；增量目录版本累积 | 源码可见 |
| 网络与 I/O | NCCL 广播完整桶；整份磁盘写完整 checkpoint；增量只写变化张量的压缩差分 | `perf/update_weights_wire_bytes` 可观测 |
| 同步 | 每桶 Ray 锁与 RPC 往返；Gloo barrier；在线路径整个搬运在暂停窗口里；整份磁盘每次同步重建桶表（PP/EP `all_gather_object` 与全 world 核对）；补丁镜像里每个 IPC 桶在 SGLang 侧多一次设备同步 | 源码与补丁可见 |
| 兼容性 | 模型族受 `convert_to_hf` 限制；`/post_process_weights`、`/pull_weights`、CI 读回依赖镜像里的 SGLang 补丁；compressed-tensors 需要额外前后处理 | 源码、补丁与文档 |

`actor.update_weights` 被 timer 包裹，写成 `perf/update_weights_time`；整份磁盘的 reload 发生在 actor RPC 返回之后的 `RayTrainGroup` 里，不在这个 timer 内，要用 driver 侧外层墙钟补测。同步占比可按

$$
\rho_{\mathrm{sync}}=
\frac{T_{\mathrm{update}}}
{T_{\mathrm{rollout}}+T_{\mathrm{train}}+T_{\mathrm{update}}}
$$

估算；上线前只能做下界 $T_{\mathrm{transport}}\gtrsim D_{\mathrm{wire}}/\mathrm{BW}_{\mathrm{eff}}$：整份路径的 $D_{\mathrm{wire}}$ 至少是一份模型权重的量级，增量路径直接看 `perf/update_weights_wire_bytes`，实际时间还要加重组、barrier、flush、主机应用、reload 与量化后处理。one-stage async 下分母应换成实测的外层迭代墙钟。actor 在本轮训练末尾就 flush 了 perf 指标，driver 随后才调用更新，所以刚产生的 `update_weights_time` 通常出现在下一次 perf flush 里。同步时长与 generation overlap 怎样一起调，归 [[30_slime_rollout_optimization_analysis|rollout 优化]]。

**总体代价与运行包络。** 同步用完整逻辑张量隔开"训练如何切"与"推理如何切"，代价是每次都在训练侧做一次全量重组；四条数据面只在"搬运落在暂停窗口里还是之前"和"跨进程搬字节还是搬句柄"上取舍。增量优化的是跨 host 的 wire 与存储 I/O，不减少重组、全量扫描、快照、主机完整 checkpoint 与 reload 进 HBM 的成本；若几乎每个字节都变，压缩后的差分加元数据可能接近甚至超过整份。运行包络还有一条镜像维度：打了补丁的缺省镜像支持全部四条数据面及其量化与 CI 分支；不打补丁时只剩 NCCL（非 compressed-tensors）、整份磁盘（不设本地目录、不开 CI）完整可用，共卡 IPC 退回上游返回语义。失败边界见 §5.1。本页未运行 slime 训练或 SGLang 服务，所有耗时与体量判断都是源码推断或本例的字节计算。

---

## 3. 代码实现分析

### 3.1 对象与所有权视图

<!-- Figure spec: ownership graph; driver → RayTrainGroup → MegatronTrainRayActor → updater; updater reaches RolloutManager for engines and lock, sends RPC/NCCL/IPC handles to SGLangEngine proxies; disk paths write the shared directory, hosts pull into local checkpoints; SGLang process is a dependency node built from upstream plus image patches. -->
```mermaid
flowchart TB
    DR["train.py driver<br/>generate → train → save → update"]
    TG["RayTrainGroup<br/>_disk_weight_version、release"]
    TA["MegatronTrainRayActor<br/>weights_backuper、weight_updater"]
    UP["weight updater<br/>weight_version、NCCL 与 Gloo 组、字节快照"]
    RM["RolloutManager<br/>Lock、servers、健康监控"]
    SV["RolloutServer 与 ServerGroup<br/>engines、GPU 数、偏移、并行配置"]
    EN["SGLangEngine Ray actor<br/>HTTP 代理"]
    SG["SGLang 进程（依赖：上游 + 镜像补丁）<br/>参数、KV、radix、weight_version"]
    FS["共享目录<br/>weight_vNNNNNN"]
    LC["主机本地 checkpoint<br/>.weight_sync/state.json"]
    DR --> TG --> TA --> UP
    UP -->|get_updatable_engines_and_lock| RM --> SV --> EN --> SG
    UP -->|RPC、NCCL、IPC 句柄| EN
    UP -->|写版本目录| FS
    TG -->|pull 与 reload| EN
    FS -->|pull_weights 补丁端点| LC -->|update_weights_from_disk| SG
```

| 对象 | 所在进程 | 拥有的状态 | 生命周期 |
|---|---|---|---|
| `RayTrainGroup` | driver | actor handles、`_disk_weight_version`、整份磁盘的 reload 编排 | 训练全程 |
| `MegatronTrainRayActor` | 每 GPU 一个 Ray actor | `weights_backuper`（CPU `actor` tag）、`weight_updater` | 训练全程；`release_train` 下按轮重建 |
| updater | 同上 | `weight_version`；NCCL group 或 Gloo 组与 IPC engine 映射；定向路由计划；增量的字节快照 | 随 actor；group 在重连或 sleep 时重建 |
| `RolloutManager` | 单个 Ray actor | `rollout_engine_lock`、servers、健康监控 | 训练全程 |
| `RolloutServer` / `ServerGroup` | RolloutManager 内 | engine handles、GPU 数、偏移、并行配置、`num_new_engines`、`needs_offload` | 训练全程；死 engine 槽位置 `None` 后重建 |
| `SGLangEngine` | 每 engine 一个 Ray actor | HTTP 地址、node rank、serving 子进程 | engine 生命周期 |
| 共享目录与主机本地 checkpoint | 文件系统 | 版本目录；`.weight_sync/state.json` 已应用标记 | 跨进程，也跨运行保留 |

### 3.2 调用流程

#### 3.2.1 在线提交：NCCL 与共卡 IPC

```text
train.py::train（初始化后一次，此后每轮 save 之后一次）
`-- RayTrainGroup.update_weights → [非整份磁盘] ray.get(actor.update_weights.remote() × world)
    `-- MegatronTrainRayActor.update_weights
        |-- [use_fault_tolerance] rank 0: RolloutManager.recover_updatable_engines → RolloutServer.recover   [归容错与可观测性]
        |-- RolloutManager.get_updatable_engines_and_lock → (engines, Lock, num_new, counts, offsets, configs)
        |-- [num_new > 0 ∨ 需重连] updater.connect_rollout_engines(...)
        |   |-- NCCL：[PP 源] connect_rollout_engines_from_distributed → init_weights_update_group × engines + init_process_group(rank 0, weight_update_backend)
        |   `-- 张量：前缀/后缀划分 → [后缀] 同上建 "slime" → [首次] dist.new_group(gloo) × 前缀 → configure_expert_routing
        `-- [offload_train] torch_memory_saver.disable() 区间：updater.update_weights()
            |-- weight_version += 1；rank 0：pause_generation → flush_cache → [compressed-tensors] post_process_weights(restore)   [补丁端点]
            |-- Gloo barrier
            |-- NCCL：_send_weights
            |   |-- _iter_non_expert_chunks：all_gather_param → [PP 源] convert_to_hf → 装桶
            |   |-- _iter_expert_chunks：all_gather_param → 攒批 → _ep_gather_and_convert
            |   `-- 每桶 _update_bucket_weights_from_distributed：Lock.acquire → RPC + dist.broadcast → ray.get → Lock.release；每遍后 barrier
            |-- 张量：HfWeightIteratorDirect.get_hf_weight_chunks → _get_megatron_full_params → convert_to_hf
            |   |-- 每桶 _send_hf_params：_send_to_colocated_engine（拍平、序列化、gather_object、源发 update_weights_from_tensor）+ [后缀] update_weights_from_distributed
            |   |   `-- SGLang：_update_weights_from_flattened_bucket → load_weights → [补丁] synchronize → TP 组 CPU barrier → 返回
            |   |-- ray.get → del → ipc_collect / empty_cache
            |   `-- [定向路由] _update_expert_weights：barrier ×2 → 每批 _prepare_expert_weight_batch（batch_isend_irecv）→ _send_hf_params → barrier → accelerator.synchronize
            |-- Gloo barrier → ipc_collect
            `-- rank 0：[compressed-tensors] post_process_weights(post_process) → continue_generation → barrier
        `-- [keep_old_actor] 按 update_weights_interval == 1 与否轮转 rollout_actor → old_actor                [归 Megatron 训练]
```

完成边界是 rank 0 的 `continue_generation` 返回且最后一次 Gloo barrier 通过；此后新请求由新版本服务。IPC 桶"返回即拷贝完成"只在打了 `sglang-deterministic.patch` 的镜像里成立（§2.2.4）。

#### 3.2.2 磁盘提交：整份与增量

```text
RayTrainGroup.update_weights
|-- [整份] version = _disk_weight_version + 1
|   |-- ray.get(actor.update_weights) → UpdateWeightFromDisk.update_weights
|   |   `-- rank 0 rmtree → barrier → mkdir → save_hf_model_to_path（HfWeightIteratorDirect，writer rank 转换并写）→ barrier → post-write hook → barrier
|   |-- [release_train] release()（ray.kill，no_restart）
|   `-- _reload_rollout_weights_from_disk
|       |-- [offload_rollout] onload_weights → get_updatable_engines_and_lock
|       |-- [本地目录] engine.pull_weights(version) → 补丁 local_checkpoint.pull
|       |-- pause → flush → update_weights_from_disk(model_path, version)
|       |-- [ci_test] get_weight_version × engines（补丁改写的 /get_weight_version），不等 → RuntimeError
|       `-- [¬keep_files] rmtree → continue_generation
`-- [增量] ray.get(actor.update_weights) → UpdateWeightFromDiskDelta.update_weights
    |-- [首次] _capture_baseline：rmtree(delta_dir) → hook → pull_weights(0) ∥ 读 hf_checkpoint 字节作快照 → ray.get(pulls) → return
    `-- version += 1
        |-- _publish：_encode_delta（_iter_hf_tensors → pinned D2H → 线程池 diff/zstd/checksum → 快照前移）→ barrier → _write_delta_files（_atomic_write 分片与 index）
        |-- _reload_engines：hook → barrier → rank 0：pull_weights(version) → pause → flush → update_weights_from_disk(local_dir, version) → continue → barrier
        `-- _record_metrics：all_reduce → perf/update_weights_density、perf/update_weights_wire_bytes
主机侧（sglang-pull_weights.patch）：/pull_weights → SchedulerWeightUpdaterManager.pull_weights(base_dir=server_args.model_path) → local_checkpoint.pull
|-- [target > 0] pre-read hook
`-- flock → 读已应用标记 → 往回找完整版本 → [需要] _reset_checkpoint → 逐版 _apply_delta（base 校验、解压、原位应用、checksum）→ 写标记 → TP 组 all_gather_object 汇总成败
```

### 3.3 源码阅读路线

slime 路径相对 `THUDM/slime@4c193f1f` 仓库根；SGLang 路径相对 `sgl-project/sglang@0b3bb0cb` 仓库根；补丁里的符号以补丁新增或改写的文件路径标注。

1. 入口与选择：`slime/backends/megatron_utils/actor.py::MegatronTrainRayActor.init` / `MegatronTrainRayActor.update_weights` → `slime/backends/megatron_utils/update_weight/__init__.py::create_weight_updater` → `tests/test_update_weight_factory.py::test_create_weight_updater_selects_implementation` → `slime/ray/rollout.py::RolloutManager.get_updatable_engines_and_lock` / `RolloutManager._get_updatable_server` / `RolloutManager.clear_updatable_num_new_engines` → `slime/backends/sglang_utils/engine_group.py::RolloutServer.engine_gpu_counts` / `RolloutServer.engine_gpu_offsets` / `RolloutServer.engine_parallel_configs` → `slime/utils/arguments.py::slime_validate_args` → `train.py::train` / `train_async.py::train`。
2. 重组：`slime/backends/megatron_utils/update_weight/hf_weight_iterator_direct.py::HfWeightIteratorDirect.get_hf_weight_chunks` / `_get_megatron_full_params` / `_get_megatron_local_param_infos` / `pack_param_info_buckets` → `slime/backends/megatron_utils/update_weight/common.py::all_gather_param` / `all_gather_params_async` / `named_params_and_buffers` → `slime/backends/megatron_utils/megatron_to_hf/__init__.py::convert_to_hf` / `_convert_to_hf_core` → `slime/backends/megatron_utils/megatron_to_hf/qwen2.py::convert_qwen2_to_hf` → `slime/backends/megatron_utils/megatron_to_hf/processors/__init__.py::quantize_params`。
3. NCCL：`slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py::UpdateWeightFromDistributed.connect_rollout_engines` / `update_weights` / `_send_weights` / `_iter_non_expert_chunks` / `_iter_expert_chunks` / `_ep_gather_and_convert` / `_update_bucket_weights_from_distributed` → 同文件 `connect_rollout_engines_from_distributed` / `update_weights_from_distributed` / `post_process_weights` → `slime/utils/accelerator/__init__.py::weight_update_backend`。
4. 共卡：`slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py::UpdateWeightFromTensor.connect_rollout_engines` / `update_weights` / `_send_hf_params` / `_send_to_colocated_engine` / `_build_flattened_tensor_data` → `slime/backends/megatron_utils/sglang.py`（`FlattenedTensorBucket`、`MultiprocessingSerializer` 的导入垫片）→ `tests/test_empty_colocated_weight_bucket.py::test_empty_colocated_bucket_still_participates_in_gather` / `test_source_rank_pads_empty_colocated_bucket_entries`。
5. 定向路由：`slime/backends/megatron_utils/update_weight/expert_routing.py::configure_expert_routing` / `_can_route_experts` / `_get_expert_target_ranks` / `_build_expert_params` / `_resolve_expert_source_ranks` / `_build_expert_transfer_plan` / `_pack_expert_transfer_batches` → `UpdateWeightFromTensor._update_expert_weights` / `_prepare_expert_weight_batch` → `tests/test_expert_routing.py::test_transfer_plan_splits_same_rank_experts_at_expert_boundaries` / `test_transfer_plan_rejects_one_expert_larger_than_buffer`。
6. 整份磁盘：`slime/backends/megatron_utils/update_weight/update_weight_from_disk.py::UpdateWeightFromDisk.update_weights` → `slime/backends/megatron_utils/hf_checkpoint_saver.py::save_hf_model_to_path` / `_get_node_save_layout` / `_copy_hf_assets` → `slime/ray/actor_group.py::RayTrainGroup.update_weights` / `_reload_rollout_weights_from_disk` / `create` → `tests/test_full_disk_weight_update.py`。
7. 增量：`slime/backends/megatron_utils/update_weight/update_weight_from_disk_delta.py::UpdateWeightFromDiskDelta.update_weights` / `_capture_baseline` / `_encode_delta` / `_write_delta_files` / `_reload_engines` / `_record_metrics` / `_atomic_write` → `slime/utils/disk_delta.py::overwrite_encode` / `checksum` / `make_tensor_reader` → `slime/backends/sglang_utils/sglang_config.py::ModelConfig.resolve`（`model_path` 与 `update_weights` 推断）→ `docs/zh/advanced/delta-weight-sync.md` → `examples/delta_weight_sync/run-glm4.7-30B-A3B-delta.sh`。
8. engine 端代理：`slime/backends/sglang_utils/sglang_engine.py::SGLangEngine._make_request` / `pause_generation` / `flush_cache` / `continue_generation` / `update_weights_from_tensor` / `update_weights_from_distributed` / `init_weights_update_group` / `update_weights_from_disk` / `pull_weights` / `post_process_weights` / `get_weight_version` / `check_weights`。
9. SGLang 上游契约：`python/sglang/srt/managers/io_struct.py::PauseGenerationReqInput` / `UpdateWeightFromDiskReqInput` / `UpdateWeightsFromTensorReqInput` → `python/sglang/srt/managers/tokenizer_manager.py::TokenizerManager.pause_generation` / `TokenizerManager.update_weights_from_disk` → `python/sglang/srt/managers/scheduler.py::Scheduler.flush_cache` → `python/sglang/srt/managers/tokenizer_control_mixin.py::TokenizerControlMixin.update_weights_from_tensor` / `update_weights_from_distributed` / `_update_weight_version_if_provided` → `python/sglang/srt/managers/scheduler_components/weight_updater.py::SchedulerWeightUpdaterManager.update_weights_from_tensor` → `python/sglang/srt/managers/tp_worker.py::BaseTpWorker.update_weights_from_tensor` → `python/sglang/srt/model_executor/model_runner.py::ModelRunner.update_weights_from_tensor` / `_update_weights_from_flattened_bucket` / `update_weights_from_distributed` / `update_weights_from_disk` → `python/sglang/srt/weight_sync/tensor_bucket.py::FlattenedTensorBucket.reconstruct_tensors` → `python/sglang/srt/layers/linear.py::ColumnParallelLinear.weight_loader` → `python/sglang/srt/entrypoints/http_server.py::weight_version`（`/get_weight_version` 与 `/weight_version` 两个路由）→ `python/sglang/srt/utils/weight_checker.py`。
10. 补丁与镜像闸门：`docker/Dockerfile`（`ARG PATCH_VERSION=latest`、`ARG ENABLE_SGLANG_PATCH=1`、补丁循环）→ `docker/patch/latest/sglang.patch`（`python/sglang/srt/entrypoints/http_server.py::weight_version` 改写、`python/sglang/srt/entrypoints/http_server.py::post_process_weights`、`python/sglang/srt/model_executor/model_runner.py::ModelRunner.post_process_weights`、`python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py::CompressedTensorsLinearMethod.restore_weights_before_loading`）→ `docker/patch/latest/sglang-pull_weights.patch`（`python/sglang/srt/managers/scheduler_components/weight_updater.py::SchedulerWeightUpdaterManager.pull_weights`、`python/sglang/srt/weight_sync/local_checkpoint.py::pull` / `_reset_checkpoint` / `_apply_delta`、`python/sglang/srt/server_args.py::ServerArgs.custom_pull_weights_pre_read_hook`）→ `docker/patch/latest/sglang-deterministic.patch`（`python/sglang/srt/model_executor/model_runner.py::ModelRunner._update_weights_from_flattened_bucket` 末尾的 `torch.cuda.synchronize()`）→ `build_conda.sh` → `docker/Dockerfile.gb10`。

---

## 4. 配套机制

### 4.1 权重等值检查

`--check-weight-update-equal` 时，`slime/ray/placement_group.py::create_rollout_manager` 在 engine 启动后对全部 engine 发 `check_weights("snapshot")` 与 `check_weights("reset_tensors")`，`train.py` 与 `train_async.py` 在第一次 `actor_model.update_weights()` 之后发 `check_weights("compare")`；动作由 SGLang 上游的 `/weights_checker` 执行（`utils/weight_checker.py` 支持这三个动作）。它比较的是首次推送后的权重与 engine 启动时从 hf_checkpoint 加载的快照，回答的是"首次推送是否完整、正确地覆盖了每个张量"，只在 actor 初始权重等于 hf_checkpoint 时有意义。`RolloutManager.check_weights` 作用于所有 server 的 engine，冻结模型不接收推送；续训（`--load` 指向不同权重）或增量模式（首次调用不推送）下这项检查的前提也不成立（本页推断，未运行验证）。它不会在后续轮次或恢复后的 engine 上重做。

### 4.2 量化前后处理与静默失败

compressed-tensors 时，在线两条路径在加载前调用 `post_process_weights(restore_weights_before_load=True)`、全部加载后调用 `post_process_quantization=True`，二者都在同一个暂停窗口里；这个端点及其对 `restore_weights_before_loading` 的分派来自 `sglang.patch`（§2.2.8），上游镜像里第一个请求就会失败。量化本身在训练侧 `convert_to_hf` 的 `quantize_params` 里按 `quantization_config` 完成。调试文档（`docs/en/developer_guide/debug.md`）记录了一个静默失败：`config.json` 的 `quantization_config.ignore` 若漏掉 MoE 路由权重 `mlp.gate.weight`，训练侧会把这个非 Linear 的 2D 张量量化成 SGLang 不以该名加载的量化名，`load_weights` 时被跳过，gate 权重全零；修法是把 `re:.*mlp\\.gate\\..*` 加入 ignore list，embedding 等其他非 Linear 2D 权重同理。RPC 成功只说明搬运与加载调用返回，不能替代等值检查与首轮 rollout/logprob 对齐（归 [[17_slime_train_inference_consistency_analysis|训推一致性]] 与 [[22_slime_low_precision_training_rollout_analysis|低精度训练与 rollout]]）。

### 4.3 与异步训练的关系

| 阶段 | 能否与其他阶段重叠 | 固定基线的边界 |
|---|---|---|
| rollout N+1 与 train N | 可以 | `train_async.py::train` 先发下一轮 generate 再训练当前轮；更新前 `ray.get` 等 generation 完成，注释"防止在生成中途更新权重" |
| 磁盘路径的写盘与主机 pull | 与在途生成可以 | 两条都在 pause 之前，pause 中止的在途请求能否回队续生成见 [[13_slime_sglang_rollout_engine_analysis#2.3 变体：同一协议的三条替换轴|ABORTED 回队可达条件]]；标准的同步与 one-stage 循环在更新时并无在途生成，这份重叠要在跨越更新边界仍在生成的 rollout 函数里才兑现（如 [[13_slime_sglang_rollout_engine_analysis|SGLang rollout 引擎]] 的 fully-async，本页推断） |
| pause → 搬运或 reload → continue | 不可以 | 提交窗口 |
| `--update-weights-interval > 1` | 频率摊薄 | 只有 `train_async.py::train` 按 `release_train or (rollout_id + 1) % interval == 0` 决定是否调用更新，`--release-train` 时每轮都更新；`train.py::train` 每轮 save 之后都调用、不读这个参数；actor 侧 `MegatronTrainRayActor.init` 与 `update_weights` 只在 `--keep-old-actor` 下判断它是否等于 1，选择 `rollout_actor → old_actor` 队列式轮转还是直接备份 `old_actor`；摊薄同步次数的代价是行为策略更旧 |

优化顺序：先用 timer 拆出占比，再看桶与拓扑或 wire 与 density，最后才调大 interval，因为后者改变的是 on-policy 新鲜度，不只是性能（本页推断）。slime 的提交是单版本的：一次更新暂停全部可更新 engine、全体切到新版本，集群里同一时刻只服务一个策略版本；DORA 描述的多版本 rollout 则让不同 DP group 各跑一个版本、长尾轨迹留在旧版本里跑完，可作为这条设计轴的对照（见 [[23_dora_multi_version_rollout_analysis|DORA 多版本 rollout]]，论文描述，无公开实现）。

### 4.4 恢复后的重连

engine 被健康监控杀掉、在下一次 `update_weights` 之前由 `RolloutServer.recover` 重建后，`num_new_engines > 0` 触发 `connect_rollout_engines`：NCCL 与越界后缀重建 group，张量路径的 Gloo 组不重建（注释：划分在重连间固定），定向路由重新规划；新 engine 从 hf_checkpoint 启动，靠这次完整推送回到当前版本。增量路径下同一台 host 的本地 checkpoint 仍在，新 engine 下一次 pull 只应用新版本；换到新 host 时从 v0 回放整条链。健康检测、重建与 CI 故障注入归 [[18_slime_fault_tolerance_observability_analysis|容错与可观测性]]。

---

## 5. 约束、适用场景与趋势

### 5.1 硬约束与失败边界

| 前提 | 源码边界 | 破坏后的行为 |
|---|---|---|
| mode、transport、colocate 组合合法 | `slime/utils/arguments.py::slime_validate_args`；`slime/backends/megatron_utils/update_weight/__init__.py::create_weight_updater` | `ValueError`；绕过校验时 `AssertionError`（`unsupported weight sync mode/transport`） |
| disk 有 `--update-weight-disk-dir`；delta 有 `--update-weight-local-checkpoint-dir` | `slime_validate_args` | `ValueError` |
| `--release-train` 配 full + disk 与 `--save`，不配 critic 与 `keep_old_actor` | 同上 | `ValueError` |
| 整份磁盘的 `--hf-checkpoint` 是本地目录且不等于版本目录 | `slime/backends/megatron_utils/hf_checkpoint_saver.py::save_hf_model_to_path` | `ValueError` |
| 各 rank 参数名、形状、dtype 一致（张量与整份磁盘路径） | `slime/backends/megatron_utils/update_weight/hf_weight_iterator_direct.py::_get_megatron_local_param_infos` | `AssertionError`；NCCL 与增量路径没有这项核对 |
| 参数带 `tensor_model_parallel`，`partition_stride` 为 1（fc1 可为 2） | NCCL 与增量：`slime/backends/megatron_utils/update_weight/common.py::all_gather_param`；张量与整份磁盘：`_get_megatron_local_param_infos` 填默认值、`all_gather_params_async` 不查 stride | 前者 `AssertionError`；后者无守卫：缺属性的分片参数以本地分片交付，非 fc1 的 stride-2 参数被普通拼接 |
| 各 EP rank 的专家批次名字数相同 | `UpdateWeightFromDistributed._ep_gather_and_convert` | `AssertionError` |
| 模型族被转换器覆盖 | `slime/backends/megatron_utils/megatron_to_hf/__init__.py::_convert_to_hf_core` | `ValueError: Unsupported model` |
| flush 在 60 次重试内成功 | `SGLangEngine.flush_cache` | `TimeoutError` |
| 整份磁盘 CI 下版本一致 | `RayTrainGroup._reload_rollout_weights_from_disk` | `RuntimeError` |
| 镜像提供 slime 调用的补丁端点 | `SGLangEngine._make_request` / `get_weight_version` 的 `raise_for_status`；端点来自 `sglang.patch`、`sglang-pull_weights.patch` | `HTTPError`：compressed-tensors 在线同步停在暂停态、CI 读回失败、增量与本地目录 pull 失败（§2.2.8 表，未运行验证） |
| 增量主机侧 base 连续、checksum 一致、整份拷贝大小一致 | 补丁 `local_checkpoint._apply_delta` / `_reset_checkpoint` | `RuntimeError`，`/pull_weights` 返回失败，slime 侧 HTTP 抛错 |
| 定向路由准入与规划 | `configure_expert_routing` | 无异常：记日志退回通用桶 |
| 共卡 engine 连续排在前面 | `UpdateWeightFromTensor.connect_rollout_engines` | 无守卫：第一个越界之后全部当作越界后缀 |
| 越界 engine 被暂停与清缓存 | `UpdateWeightFromTensor.update_weights` 只对共卡前缀发 pause、flush、量化前后处理与 continue；逐桶 RPC 不传 `flush_cache`（wrapper 缺省 False） | 无守卫：它所在的 server group 被判 `needs_offload`（组起点落在 actor 区间内，本例即如此）时，offload 路径的 `release_memory_occupation` 会先 flush；单独成组且起点越界时，radix/前缀缓存跨版本保留，compressed-tensors 的前后处理也缺失。上游对未暂停 engine 的每个更新 RPC 先取 `model_update_lock` 写锁、等在途请求结束，桶与桶之间锁会释放，新请求可能跑在半新半旧的权重上，而版本号在第一桶成功后已是新值（源码路径推断，未运行验证） |
| offload 下 updater 读到的参数有效 | `slime/backends/megatron_utils/actor.py::MegatronTrainRayActor.update_weights` 只在 `offload_train ∧ use_critic ∧ ¬colocate` 时 wake；`slime/backends/megatron_utils/update_weight/common.py::named_params_and_buffers` 只遍历模型上的参数与 `expert_bias` buffer，没有 CPU 回退选项 | 无守卫：NCCL、整份磁盘、增量 updater 直接读设备上的参数，在其余 offload 组合（colocate 配 disk transport 且非 `release_train`、非 colocate 手动开 `offload_train` 而无 critic）里参数已被 `torch_memory_saver.pause()`；张量 updater 读 CPU tag 不受影响；作为对照，`save_model` 在 `save_hf_model_to_path` 之前会显式 wake（后果属依赖行为，未运行验证） |
| 主机本地 checkpoint 属于本次运行 | 补丁 `local_checkpoint.pull` 以 `.weight_sync/state.json` 的已应用版本为下限；`_capture_baseline` 只清共享 delta 目录 | 无守卫：重启后本地目录若保留上次运行的标记（如 57），版本号从 1 重新计数，`pull(1)` 什么也不做，engine reload 的仍是上次运行 v57 的内容，版本号却写成 "1"；增量模式到 v58 时 checksum 不符才报错，整份模式到 v58 整份重置后恢复（源码路径推断，未运行验证） |
| 首次推送覆盖 actor 初始权重 | `UpdateWeightFromDiskDelta.update_weights` 首次只 `_capture_baseline` | 无守卫：增量模式第一轮 rollout 用 engine 启动时的 hf_checkpoint 权重，`--load` 指向不同权重时这一轮是离策略的；开 `--check-weight-update-equal` 时 engine 启动后已被 `reset_tensors`，首次调用又不 reload，第一轮与等值比较面对的是重置后的权重（`reset_tensors` 的语义属依赖侧） |
| 快照与主机 base 字节对齐 | `_capture_baseline` 按名读 `--hf-checkpoint`；主机 base 是 engine 的 `model_path` | 同名张量字节长度与转换结果不同，xor 在工作线程里报错、收集结果时抛出；`model_path` 与 `--hf-checkpoint` 字节不同时第一个增量 checksum 失败（本页推断） |
| 增量确实划算 | 没有按 density 的回退 | 无守卫：看 `perf/update_weights_density` 与 wire bytes 决定 |
| IPC 块在消费者拷贝完成前不被复用 | 补丁 `ModelRunner._update_weights_from_flattened_bucket` 的 `synchronize`；slime 侧只有每桶新建缓冲，`_send_to_colocated_engine` 只在源 rank 返回 ref | 补丁镜像下成立；上游镜像下依赖 PyTorch CUDA IPC 引用计数，slime 没有显式等待，损坏会是静默的（依赖侧，未核） |

### 5.2 常见误读

| 误读 | 固定基线的实际行为 |
|---|---|
| 权重同步就是一次 broadcast | 广播只解决搬运；名字、行序、暂停、缓存、量化后处理与版本号各有归属 |
| 把训练分片直接拷给推理分片更快 | 对应关系随 TP 与层类型变；fc1 在推理 TP=4 下就错位，TP 相等时一致只是巧合 |
| 完整 HF 张量意味着每个 SGLang rank 都存全参数 | 那是交付形状；loader 按 `tp_rank` 截取后只写本 rank 分片（上游契约） |
| 每个参数只在一个训练 rank 上聚合 | NCCL 路径所有 rank 都做 TP all-gather，只有 PP 源转换发送；张量路径每个 rank 都物化并转换完整桶；整份磁盘每个 rank 都物化、只有 writer rank 转换 |
| 共卡是零拷贝共享参数 | 共享的是临时桶的显存，SGLang 仍拷进自己的参数；CPU 快照是交接介质不是 IPC 介质 |
| slime 调用的 SGLang 端点都是上游的 | `/post_process_weights`、`/get_weight_version` 的返回、`/pull_weights` 与 IPC 返回前的设备同步都来自镜像补丁；`ENABLE_SGLANG_PATCH=0` 的镜像没有它们 |
| IPC 的 RPC 返回说明拷贝已完成 | 只在打了 `sglang-deterministic.patch` 的镜像里成立；slime 源码注释描述的是上游语义 |
| 阈值是显存上限 | 单个超阈值 chunk 独占一桶，临时副本不计入 |
| delta 让 HBM 里只更新变化部分 | 主机应用后仍整模型 reload；省下的是跨 host 的 wire 与存储 |
| delta 首次同步就把 actor 权重推过去 | 首次只捕获快照 |
| delta 也能配 NCCL 做同机验证 | 参数校验与工厂断言都拒绝；当前 external engine 文档也写明 delta 只支持 disk（v0.3.1 文档的选型表曾列过 `delta + nccl`） |
| 所有路径都读回校验了版本 | 只有整份磁盘在 CI 下读回，且依赖补丁端点 |
| `--update-weights-interval` 在 `train.py` 同步训练里降低更新频率 | `train.py::train` 每轮都更新、不读它；按 interval 跳过更新只发生在 `train_async.py::train` |
| `--update-weights-interval` 只影响调用频率 | 开 `--keep-old-actor` 时 actor 还按它是否等于 1 选择 old_actor 的轮转方式；`train_async.py` 下 `--release-train` 每轮强制更新 |
| 定向路由覆盖全部 MoE 参数 | 只匹配 decoder 层 routed expert 的 fc1/fc2 |

### 5.3 何时使用与检查清单

| 场景 | 首选 | 为什么匹配 | 不应选择或重点失败信号 |
|---|---|---|---|
| 同集群、训推分离、GPU 网络充足 | NCCL full | 不落盘，逻辑 HF 桶直接广播；非量化模型不依赖补丁端点 | engine rank 排序或建组不稳定时不选；看建组失败、Lock 长等待、单桶 RPC 失败 |
| colocate，共享槽位可解释 | 张量 IPC | 控制面只传描述符，payload 走同卡显存；MoE 合格时定向路由 | 镜像没打 deterministic 补丁时不应依赖"返回即完成"；有越界 engine 时核对它的缓存是否被 flush；看 RPC 未返回与暂停窗口里的 HBM 峰值 |
| external 或异构 GPU，共享目录可靠 | full disk | 训推只经版本化 HF checkpoint 耦合，pull 可放在 pause 前 | 可见性弱时配 hook；看 reload 延迟与 CI 版本不一致；重启前清理本地目录 |
| 跨 host 的 wire 或存储是瓶颈，字节差分可压缩 | delta disk | 省略未变张量，只发布压缩差分 | base 不能严格串行、本地盘不足、镜像没有 pull 补丁或 density 高时不选；看 out-of-order、checksum、density 与 wire bytes；重启前清理 `.weight_sync` |

改动同步路径前逐项核对：镜像是否按 `PATCH_VERSION` 打了 slime 的 SGLang 补丁（`ENABLE_SGLANG_PATCH` 是否被关掉）；新模型是否被 `convert_to_hf` 覆盖且 buffer 不止 `expert_bias`；阈值是否让单个最大参数独占一桶也放得下；colocate 下是否存在越界 engine、它的缓存由谁清；offload 组合下 updater 读的是 CPU tag 还是设备参数；定向路由的日志是 Enabled 还是 Disable；量化模型的 ignore list 是否覆盖所有非 Linear 权重；磁盘路径的共享目录对训推两侧是否同路径可见、本地目录是否属于本次运行、engine 的 `model_path` 与 `--hf-checkpoint` 字节是否相同；验收是否同时看了 engine 版本、首批 rollout 的 `weight_versions`、首轮等值检查、flush 成功与 `perf/update_weights_time`。

### 5.4 当前演进方向

| 位置 | 注释或提交原文 | 指向什么 |
|---|---|---|
| `slime/backends/megatron_utils/update_weight/common.py::all_gather_param` / `all_gather_params_async` | `# TODO: here we did an extra copy during concat, maybe merge this with convert_to_hf is better?`；`# TODO: check only GLU is used.` | TP 拼接与 HF 转换之间多一次拷贝，是合并候选；fc1 的重排默认它是 GLU |
| 同上 | `# this is bug in megatron's grouped moe.` | `linear_fc2.weight` 的 `partition_dim` 修正在绕开上游缺陷 |
| `slime/backends/megatron_utils/update_weight/common.py::named_params_and_buffers` | `# TODO shall we handle (almost) all buffers` | 只同步 `expert_bias` 一类 buffer，依赖其他 buffer 的模型落在边界外 |
| `slime/backends/megatron_utils/megatron_to_hf/__init__.py::convert_to_hf` / `_convert_to_hf_core` | `# TODO optimize code details`；模块级缓存 `_cached_tensors` 前的 `# TODO optimize` | 转换入口与跨调用缓存的实现细节仍待整理 |
| `create_weight_updater` 与迭代器、名字遍历函数 | 提交 `[cleanup] extract create_weight_updater to make actor's init func cleaner`、`[cleanup] remove dead code and merge never visited branches` | updater 选择移出 actor，`HfWeightIteratorBase`、无调用方的 CPU 回退与非全局命名分支被删，层次在收窄 |
| `slime/utils/accelerator/` 与 `_build_flattened_tensor_data` 注释 | `weight_update_backend`；"Do not reuse the IPC-facing flattened tensor" | 搬运后端从写死 `nccl` 变成按加速器选择；IPC 完成语义在补丁里补齐，slime 侧仍保留不复用缓冲的防线 |

> [!note] 推断
> 这些标记方向一致：**完整逻辑张量这个中间表示不动，拷贝次数、选择层次与平台耦合在收窄**。合并 concat 与转换优化的是暂停窗口里的拷贝与 HBM 峰值，不改变 §2.2.8 的六个动作；buffer 覆盖扩大后，"RPC 返回即同步完整"才更接近成立；依赖侧语义目前靠镜像补丁补齐，补丁与 slime 调用之间没有运行期探测。源码只写了 TODO、提交标题与注释，没有给出替代方案或时间；这层归纳由本页承担，不代表项目路线图。

---

## 6. 配置契约

slime 域没有配置 coverage ledger；下表只列本页路径直接读取的参数，默认值取自 `slime/utils/arguments.py`，SGLang 服务参数经 `--sglang-*` 透传（由已安装 SGLang 的 `ServerArgs` 生成，补丁新增的字段只在打了补丁的镜像里存在）；其余参数与脚本的对应归 [[02_slime_quickstart_and_configuration_guide|配置指南]]。

### 选择与传输

| 参数 | 默认 | 契约 |
|---|---|---|
| `--update-weight-mode` | `full` | choices full / delta；delta 只配 disk、非 colocate，且需要本地目录 |
| `--update-weight-transport` | `nccl` | choices nccl / disk；disk 需要共享目录；nccl 的实际通信后端由加速器决定 |
| `--colocate` | False | transport 为 nccl 时选张量 IPC updater（disk 优先于 colocate）；缺省开 `offload_train` 与 `offload_rollout`（`release_train` 时只开后者）；`train_async.py` 不支持 |
| `--release-train` | False | 需 full + disk 与 `--save`；不配 critic 与 `keep_old_actor`；reload 前释放训练 actor；`train_async.py` 下每轮都更新 |

### 磁盘与增量

| 参数 | 默认 | 契约 |
|---|---|---|
| `--update-weight-disk-dir` | None | 训推共享目录；full 每次写一个 `weight_vNNNNNN`，delta 每次写一个差分目录 |
| `--update-weight-disk-keep-files` | False | 只对 full disk：保留版本目录 |
| `--update-weight-local-checkpoint-dir` | None | 主机本地完整 checkpoint；delta 必需，full 可选；跨运行保留已应用标记；依赖补丁端点 `/pull_weights` |
| `--update-weight-delta-encoding` | `xor` | xor（最小、须恰好应用一次）/ overwrite（更大、幂等） |
| `--update-weight-delta-checksum` | `xxh3-128` | xxh3-128 / blake3 / adler32；帮助说明这是摘要属性的选择而不是速度选择 |
| `--custom-update-weight-post-write-path` | None | 每个训练 rank 写完后调用，`hook(args, version_dir, rollout_engines)` |
| `--sglang-custom-pull-weights-pre-read-hook` | None | `sglang-pull_weights.patch` 新增的 SGLang 服务参数，主机侧读版本目录前调用 `hook(source_dir, target_version)` |

### 分桶、频率与检查

| 参数 | 默认 | 契约 |
|---|---|---|
| `--update-weight-buffer-size` | 512 MiB | 分桶阈值，不是显存上限；专家按 × EP 判断；也是定向路由每 rank 的 staging 上限 |
| `--update-weights-interval` | 1 | 只有 `train_async.py::train` 据此决定是否调用更新（`--release-train` 时忽略）；`train.py` 不读；actor 只在 `--keep-old-actor` 下比较它是否为 1 以选择 old_actor 轮转方式 |
| `--check-weight-update-equal` | False | 启动时 snapshot 与 reset，首次推送后 compare |
| `--use-fault-tolerance` | False | 每次更新前先恢复坏 engine（归 [[18_slime_fault_tolerance_observability_analysis|容错与可观测性]]） |
| `--sglang-ep-size` / `--sglang-moe-dp-size` / `--sglang-pp-size` / `--sglang-enable-eplb` 等 | SGLang 默认 | 决定 MoE 定向路由的准入 |

### 镜像构建参数（`docker/Dockerfile`，不是 slime CLI）

| 构建参数 | 默认 | 契约 |
|---|---|---|
| `SGLANG_IMAGE_TAG` | `v0.5.15.post1-cu129` | SGLang 底座镜像；CUDA 13 发布配方改为 `v0.5.15.post1-cu130` |
| `PATCH_VERSION` | `latest` | 选择 `docker/patch/<version>/`；基线上 `latest` 与 `v0.5.15.post1` 两个目录相同，更早目录没有 pull_weights 与 deterministic 补丁 |
| `ENABLE_SGLANG_PATCH` | `1` | 为 1 时按固定顺序检查并应用 5 个 SGLang 补丁，任一不能干净应用即构建失败；注释写明 GB200/GB300 临时关闭，关闭后 §2.2.8 表中的补丁端点与同步语义都不存在 |

## Related Pages

- [[10_slime_end_to_end_iteration_analysis]] — 权重提交在一轮 generate → train → save → update 中的位置，以及异步循环里更新前等待 generation 的时机。
- [[11_slime_ray_control_plane_analysis]] — engine 句柄、锁、GPU 偏移与 `release_train` 的 actor 生命周期由哪个对象负责。
- [[14_slime_megatron_training_analysis]] — CPU `actor` tag、sleep/wake 与 `keep_old_actor` 轮转怎样提供本页的快照。
- [[17_slime_train_inference_consistency_analysis]] — 版本、暂停与量化都对齐之后仍需要的数值一致性证据，以及同一组镜像补丁在对齐栈里的作用。
- [[21_verl_weight_publication_analysis]] — verl 把权重发布拆成语义、传输、应用三段，并用 shard-local 的 `delta_sharded` 取代先 full gather 再做差的增量，可与本页的四条数据面对照。
- [[01_posttraining_infra_mechanism_analysis]] — 框架无关的 weight publish 事务阶段与观测指标，本页的六个提交动作是它在 slime 上的落地。
- [[23_dora_multi_version_rollout_analysis]] — 多版本并存的 rollout 编排，对照 slime 暂停全部 engine、单版本整体切换的提交方式。
