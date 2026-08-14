# slime Megatron→SGLang 权重同步实现分析

> **源码基线**：slime `main@681b3adca54105d5ecd3fb822fa0dc58a427e0f9`
> **核验日期**：2026-08-14 · **系列**：[[slime/index]]
> **结论先行**：slime 的权重同步是一个带版本的服务提交协议，不是简单的 parameter broadcast。共同语义是：选定训练快照 → Megatron shard 还原并转 HF/SGLang 名称 → 暂停新生成并清 KV/cache → 完整传输或落盘 → 量化后处理/版本确认 → 恢复生成。NCCL、colocated tensor IPC、full disk、delta disk 只是载体不同；任何一条路径若省掉 pause、flush、barrier、version 或 atomic publish，都会允许半套权重或旧 KV 泄漏。

## 1. Updater 选择矩阵

actor 初始化按 `update_weight_mode × transport × colocate` 选择实现：[ `actor.py:151-182`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/actor.py#L151-L182)

| mode | transport/topology | updater | 主要载体 |
|---|---|---|---|
| full | NCCL、训推分离 | `UpdateWeightFromDistributed` | Megatron rank→SGLang engine NCCL group |
| full | colocate | `UpdateWeightFromTensor` | GPU tensor→CPU serialize→Gloo gather→Ray/CUDA IPC；可混合远端 NCCL |
| full | disk | `UpdateWeightFromDisk` | versioned HF checkpoint + SGLang disk reload |
| delta | disk、非 colocate | `UpdateWeightFromDiskDelta` | changed bytes + checksum/index + host-local base apply |

delta 被显式限制为 disk 且不支持 colocate；其他未知组合立即 assert，而不是静默回退。[`actor.py:154-174`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/actor.py#L154-L174)

## 2. 共同控制协议

```mermaid
sequenceDiagram
    participant Driver as Megatron ranks
    participant Manager as RolloutManager
    participant Engine as SGLang engines
    Driver->>Manager: get updatable engines + lock + topology
    Manager-->>Driver: handles, gpu counts/offsets/config
    Driver->>Engine: pause_generation
    Driver->>Engine: flush_cache
    Driver->>Driver: gather/convert/bucket or publish files
    Driver->>Engine: transfer/reload(version=v+1)
    Driver->>Engine: quantization postprocess if needed
    Driver->>Engine: continue_generation
```

actor 每次提交前可先恢复 crash 的 updatable engines，必要时重连 NCCL/process groups；拿到的 per-engine GPU counts/offsets/parallel config 同时服务异构 TP 和 colocated expert routing。[`actor.py:592-636`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/actor.py#L592-L636)

RolloutManager 只把第一个 `update_weights=True` model 暴露给 updater，冻结 ref/reward models 自动排除；当前多 updatable model 尚不支持。[`rollout.py:555-584`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/rollout.py#L555-L584)

## 3. Megatron shard → HF/SGLang tensor

两类在线 updater 都要解决三种 shard：

1. TP/non-expert：all-gather 参数，按模型 conversion 变成 HF 名称/shape；
2. EP/expert：先 TP gather，再按 EP group 批量 all-gather/convert；
3. PP：每个 PP source rank建立独立 `slime-pp_{pp_rank}` 更新组，负责自己层段。

NCCL path 的 non-expert/expert 分批逻辑见 [`update_weight_from_distributed.py:153-239`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py#L153-L239)。buffer size 限制 conversion 后 bucket，避免一次 materialize 全模型 HF 权重。

tensor path 用 `HfWeightIterator` 提供“像 HF named_parameters 一样”的 chunk 视图；默认 direct iterator 预计算 local param info buckets。[`hf_weight_iterator_base.py:7-31`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/update_weight/hf_weight_iterator_base.py#L7-L31)

## 4. NCCL 分离路径

### 4.1 建组

每个 PP source（DP=0、TP=0）创建一个 group，world size = 所有 SGLang engine GPU 数之和 + 训练 source rank；异构 engine TP 通过 cumulative rank offsets 加入同一 group。[`update_weight_from_distributed.py:57-100`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py#L57-L100) [`update_weight_from_distributed.py:268-314`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py#L268-L314)

### 4.2 提交

rank 0 pause/flush 后，所有 trainer ranks 过 Gloo barrier；再依次发送 non-expert、barrier、expert、barrier；全部完成后做量化 postprocess并 resume。[`update_weight_from_distributed.py:102-146`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py#L102-L146)

每 bucket 先获取 Ray engine lock，metadata 走 Ray RPC，tensor data 由 NCCL async broadcast，等待 engine RPC 完成后才清 bucket、释放 lock。锁的目的明确是防并发 broadcast deadlock。[`update_weight_from_distributed.py:240-265`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py#L240-L265) [`update_weight_from_distributed.py:326-355`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py#L326-L355)

## 5. Colocated tensor/IPC 路径

同卡路径不能让 SGLang 与 Megatron 同时长期占满 GPU。`UpdateWeightFromTensor` 根据 engine GPU offsets 判断哪些 engines 完全落在 actor GPU 范围，分别建立 colocated Gloo gather group；若还有 rollout-only 远端 engines，同一个 updater 可同时使用 NCCL 分发。[`update_weight_from_tensor.py:95-180`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py#L95-L180)

每个 HF chunk 的 colocated path 将 flattened tensor bucket 序列化，各 local ranks 用 Gloo `gather_object` 收到 engine source rank，再通过 Ray 调 `update_weights_from_tensor`;缺少某 dtype bucket 的 rank显式发送 empty bucket，保持集合形状一致。[`update_weight_from_tensor.py:359-424`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py#L359-L424)

提交循环每 chunk 等 SGLang consumer 返回后才 `ipc_collect/empty_cache`，防止 producer 提前释放 CUDA IPC backing storage；最后 barrier 后再清一次，全部完成才 resume generation。[`update_weight_from_tensor.py:276-331`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py#L276-L331)

### 5.1 Rank-local expert update

当所有 engines 都 colocated 且 Megatron/SGLang MoE topology满足条件时，expert routing planner把专家直接发送到目标 SGLang ranks，dense params继续常规 buckets；存在 distributed engines、异构/不合格 topology 时自动禁用。[`expert_routing.py:295-380`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/update_weight/expert_routing.py#L295-L380)

这优化了无谓的全量 expert 汇聚/再分发，但正确性依赖 engine GPU ordering、EP/MoE-DP topology 和 expert id mapping 全部一致。

## 6. Full disk 路径

每次 version++ 后写 `weight_vNNNNNN` 完整 HF checkpoint：rank 0 清旧目录，所有 writing ranks各自 mkdir/write，Gloo barrier 后运行可选 post-write hook，再 barrier。hook 用于对象存储型 shared filesystem 的显式 publish/read-after-write。[`update_weight_from_disk.py:17-95`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/update_weight/update_weight_from_disk.py#L17-L95)

真正的 SGLang reload 由 `RayTrainGroup` 接管：可先 pull 到 host-local NVMe（这段可与 generation overlap），然后 pause/flush/reload；CI 模式逐 engine 核对 version，成功后清临时目录并 resume。[`actor_group.py:227-269`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/actor_group.py#L227-L269)

full disk 的优点是跨环境/异构 GPU、external serving 友好，且 `release_train` 后仍有完整提交物；代价是写放大和 shared filesystem latency。

## 7. Delta disk 路径

### 7.1 baseline 如何建立

第一次调用只捕获 baseline，不发布；baseline优先来自 SGLang host materialize 的 HF checkpoint，以保证 snapshot 与 engine base 一致，缺失 tensor才回退到当前 gathered weights。[`update_weight_from_disk_delta.py:82-125`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/update_weight/update_weight_from_disk_delta.py#L82-L125)

这里有两个同步发生的“base”：trainer 持有上一版本的 CPU byte snapshot；每个 SGLang host 的 local checkpoint 也 materialize version 0。后续 version $v$ 只能基于 $v-1$ 生成和应用，所以它不是任意两个 checkpoint 之间的 stateless diff。

### 7.2 delta 权重 diff 到底怎样生成

trainer 仍先把 Megatron TP/EP shards gather/convert 成逐 tensor HF 视图，再把新 tensor contiguous 后按 `uint8` 展平。对旧 snapshot $w^{(8)}(v-1)$ 与新权重 $w^{(8)}(v)$，支持两种编码：

$$
d^{\mathrm{xor}}(v)=w^{(8)}(v)\oplus w^{(8)}(v-1).
$$

- `xor`：保存逐字节 XOR；未变化字节为 0，zstd 很容易压缩，但同一 delta 只能对正确 base 应用一次。
- `overwrite`：先找 `new != old` 的 byte positions，再编码“changed count + uint32 positions + new byte values”；体积通常更大，但重复写相同位置是幂等的。[`disk_delta.py:21-25`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/disk_delta.py#L21-L25)

GPU→CPU copy 使用 pinned buffer pool，并与 CPU thread-pool 的 diff/zstd level-1/checksum 流水；整个 tensor 未变化就不写入 delta shard。每轮完成后，新 byte array 替换 snapshot，成为下一版 base。[`update_weight_from_disk_delta.py:199-273`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/update_weight/update_weight_from_disk_delta.py#L199-L273)

之后写 canonical safetensors shards，index 记录 `version/base_version/encoding/checksum`；SGLang host 解压后在 local checkpoint 的 mmap region 原位 XOR 或 overwrite，校验新 tensor checksum，再走普通 disk reload。[`update_weight_from_disk_delta.py:127-190`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/update_weight/update_weight_from_disk_delta.py#L127-L190) [`sglang-pull_weights.patch:426-545`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/docker/patch/latest/sglang-pull_weights.patch#L426-L545)

### 7.3 为什么 push/pull 变小，哪些成本没有变小

所谓“减少 push/pull”准确地说，是减少 **trainer 发布到 shared filesystem 的 bytes** 和 **每个 serving host 拉取的 network/storage bytes**：不变 tensor 完全省略，变化 tensor 只传压缩后的 byte diff；`pull_weights(version)` 这个 Ray RPC 本身只是控制面，真正 payload 是 host 读取 version directory 并在本地 base 上 apply。[`sglang-pull_weights.patch:197-215`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/docker/patch/latest/sglang-pull_weights.patch#L197-L215)

它**不会**减少以下成本：Megatron→HF 的全 tensor gather/convert、GPU→CPU 的全量扫描、trainer 上一版完整 CPU snapshot、每个 host 的完整 local checkpoint，以及 apply 后 SGLang 把完整模型重新装入 serving HBM。换言之，delta 优化的是跨 host 的 wire/storage I/O，不是把模型在 HBM 中变成稀疏增量更新。

固定基线也没有按 density 自动回退 full checkpoint：updater 在初始化时已由 mode/transport 固定选择，delta 每轮都执行 byte diff/compress。[`actor.py:151-174`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/actor.py#L151-L174) 若更新后几乎每个 byte 都改变，zstd 后 wire bytes 可能接近甚至因 metadata 略高于 full；必须看框架已经记录的 density/wire 指标再决定是否使用。

文件先写 `.tmp`、flush+fsync，再 `os.replace`，防 reader 看见半文件。[`update_weight_from_disk_delta.py:297-303`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/update_weight/update_weight_from_disk_delta.py#L297-L303) 它还记录 changed density 与 wire bytes，让“delta 是否真的更省”可观测。[`update_weight_from_disk_delta.py:275-294`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/update_weight/update_weight_from_disk_delta.py#L275-L294)

`xor` delta 最小但必须对正确 base 恰好应用一次；`overwrite` 更大但幂等。base_version/checksum/host lock 是 delta 正确性的必要部分，不能把它理解为普通增量 checkpoint。

## 8. 同步时长、关键路径与 overlap

没有一个脱离模型大小、TP/EP、网络、shared FS 和量化方式的固定同步占比。`actor.update_weights` 被 timer 包裹，日志键为 `perf/update_weights_time`；delta 还额外上报 `perf/update_weights_density` 与 `perf/update_weights_wire_bytes`。[`actor.py:591-653`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/actor.py#L591-L653) [`train_metric_utils.py:13-50`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/train_metric_utils.py#L13-L50) 但 full-disk path 的 actor RPC 只负责 publish，RayTrainGroup 在 RPC 返回后另做 host pull/reload；这部分不在该 timer 内，必须用 driver/outer-iteration wall-clock 补测。[`actor_group.py:162-173`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/actor_group.py#L162-L173)

同步主链可先用以下 wall-clock 口径评估：

$$
\rho_{\mathrm{sync}}=
\frac{T_{\mathrm{update}}}
{T_{\mathrm{rollout}}+T_{\mathrm{train}}+T_{\mathrm{update}}}.
$$

上线前只能做带假设的下界估算：令实际跨通道字节数为 $D_{\mathrm{wire}}$、有效带宽为 $\mathrm{BW}_{\mathrm{eff}}$，则 $T_{\mathrm{transport}}\gtrsim D_{\mathrm{wire}}/\mathrm{BW}_{\mathrm{eff}}$。full path 的 $D_{\mathrm{wire}}$ 至少是一个模型权重体量的量级，delta path 则应直接使用 `perf/update_weights_wire_bytes`；实际时间还要加 shard gather/conversion、barrier、cache flush、host apply、engine reload 和量化后处理，因此这个公式只能做容量规划下界，不能替代 timer。

若启用一拍异步，则分母应改成实测 outer-iteration wall time，不能再把 rollout 与 train 简单相加。另一个日志细节是：actor 在本轮 train 末尾 flush perf metrics，而 driver 随后才调用 weight update，因此刚产生的 actor-side `update_weights_time` 通常在下一次 actor perf flush 才出现。[`actor.py:514-564`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/actor.py#L514-L564) [`train.py:53-85`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/train.py#L53-L85)

| 阶段 | 能否 overlap | 固定基线的边界 |
|---|---|---|
| rollout N+1 与 train N | 可以 | `train_async.py` 的一拍 pipeline |
| full/delta disk 的 host pull/apply | 部分可以 | 在 pause 前预拉到 local checkpoint，可与 generation overlap |
| pause→flush→online transfer/reload→resume | 不可以 | serving commit barrier；保证单请求不跨 version |
| `update_weights_interval > 1` | 属于频率摊薄，不是单次掩盖 | 减少平均同步次数，但增加 behavior-policy staleness |

同步路径的优先优化顺序应是：先用 timer 拆出占比；再看 NCCL bucket/拓扑或 disk wire/density；最后才调大 update interval，因为后者改变的是 on-policy 新鲜度，不只是性能。

## 9. 量化权重的额外阶段

compressed-tensors INT4/FP4 在加载前先 restore original weights、加载后再 quantize postprocess，因此在线 updater必须把这两个动作包在同一次 pause/resume 事务中。[`update_weight_from_distributed.py:109-134`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py#L109-L134)

量化 ignore list 错误可能让 MoE gate 等非 Linear 2D tensor被转成 SGLang不识别的名字而静默跳过；`check_weight_update_equal` 和首轮 rollout/logprob 对齐是必要门禁，而不是仅看 update RPC 成功。

## 10. 选择建议

| 场景 | 首选 | 原因 | 主要风险 |
|---|---|---|---|
| 同集群训推分离 | NCCL full | 低落盘、直接 GPU broadcast | 建组/锁/异构 TP 复杂 |
| colocate | tensor/IPC | 避免远端网络，支持混合 extra rollout GPUs | IPC lifetime、GPU ordering |
| external/异构 serving | full disk | 环境隔离、可用 shared FS | 全量写放大 |
| 大模型低密度更新/跨环境 | delta disk | 降 wire bytes，可 host-local apply | base/version/checksum 协议最严格 |

## Related Pages

- [[10_slime_end_to_end_iteration_analysis]] — 权重提交在一轮事务中的位置
- [[11_slime_ray_control_plane_analysis]] — engine handles/lock/topology 的 owner
- [[19_slime_rollout_backend_extension_analysis]] — 新 backend 必须重建的权重提交契约
- [[22_slime_low_precision_training_rollout_analysis]] — FP8/INT4 的量化提交阶段
- [[17_slime_train_inference_consistency_analysis]] — version、pause/flush 和量化一致性
- [[30_slime_rollout_optimization_analysis]] — 权重同步与 generation overlap 的性能权衡
