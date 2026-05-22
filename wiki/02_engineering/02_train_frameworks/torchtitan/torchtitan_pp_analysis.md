# 流水线并行 PP —— 机制级深度分析

> **代码基准**:torchtitan `main` @ `cf3c4312` · PyTorch `2.9.1`(`torch/distributed/pipelining/` 内核)
> **最后更新**:2026-05-22 · **系列**:torchtitan 多维并行源码级分析(见 [[torchtitan/index]])
>
> 本文按统一结构回答:**模型怎么切成 stage?stage 间怎么通信?气泡怎么消?异步怎么实现?** 重点是各调度的气泡结构与 Zero Bubble。
>
> 行号约定:torchtitan 以 `torchtitan/` 为根;PyTorch 2.9.1 以 `[pt]` 前缀,根目录 `torch/distributed/pipelining/`。

---

## 1. 功能范围与定位

**PP(流水线并行)** 把模型**按层(深度)**切成多个 stage,每个 stage 一组卡,microbatch 在 stage 间像流水线一样依次流过。它服务于"层数极多、跨大量节点"的场景——PP 在 stage 边界**只传激活/梯度**(P2P,数据量小),是最省跨节点带宽的并行方式,适合放在最慢的网络层级。

torchtitan 侧只做**模型切分 + schedule 工厂**(`torchtitan/distributed/pipeline_parallel.py`,入口 `pipeline_llm`),真正的调度执行机制全在 PyTorch `pipelining` 包。

> **版本提示**:torchtitan 给 `PipelineStage` 传了 `get_mesh=` 关键字参数(`pipeline_parallel.py:586`),但本机 PyTorch 2.9.1 的 `PipelineStage.__init__` 不接受该参数——torchtitan 主线针对更新的 PyTorch。DTensor 跨 stage 重组的机制结论不受影响。

---

## 2. 模型切分:从整模型到 PipelineStage

### 2.1 三步切分(torchtitan 侧)

`pipeline_llm`(`pipeline_parallel.py:68`):

**① 算 virtual stage 数** —— `_get_pipeline_metadata`(`pipeline_parallel.py:147`)。用 `get_schedule_class` 判断 schedule 是单段(`PipelineScheduleSingle` 子类)还是多段(`PipelineScheduleMulti`)。virtual stage 数由 `pipeline_parallel_layers_per_stage` 推出,强制 `num_virtual_stages % pp_degree == 0`。单段调度要求每 rank 1 个 stage,多段要求 ≥2 个。

**② 生成每 stage 的模块 FQN 列表** —— `_generate_llm_fqn_per_model_part`(`pipeline_parallel.py:313`)。把 `tok_embeddings`/`norm`/`lm_head` 按"等效层数"参与均分:`num_effective_layers = num_layers + input_weight + output_weight`(`input_weight`/`output_weight` 让首/末 stage 少分几层真实 transformer 层,补偿 embedding/输出投影的计算量)。产物形如:

```
[["tok_embeddings","layers.0"], ["layers.1","layers.2"], ["norm","lm_head"]]
```

**③ 物理切模型** —— `_split_module`(`pipeline_parallel.py:426`):

```python
model = copy.deepcopy(whole_model)
# 对 ModuleList/ModuleDict 删掉不属于本 stage 的层;对普通子模块不保留则 setattr(model, name, None)
```

> 这要求模型 `forward()` 必须**容忍被删层**(`None` 子模块)——是 torchtitan 模型代码的硬约束。

### 2.2 PP rank → stage 索引映射

`_get_pp_rank_to_stage_indices_mapping`(`pipeline_parallel.py:488`)两种风格:

- **`loop`**(普通 looped/interleaved):rank `r` 持有 stage `r, r+pp, r+2pp, …`。例 pp=2、4 stage → rank0={0,2}、rank1={1,3}。
- **`v`**(`ScheduleZBVZeroBubble`/`ScheduleDualPipeV`):每 rank 恰好 2 个 stage,配对 `zip(range(pp), range(num_stages-1, pp-1, -1))`。例 pp=4、8 stage → rank0={0,7}、rank1={1,6}、rank2={2,5}、rank3={3,4},形成 **V 形折叠**。

### 2.3 PipelineStage 与跨 stage 的形状推断

`PipelineStage`(`[pt] stage.py`)假设线性切分、无 skip-connection。它不需要用户给输入输出形状——`_shape_inference` 在初始化时**沿 stage 链顺序传播**:stage 0 用真实 meta 输入跑一遍,把输出 meta 用 `dist.send_object_list` 给 stage 1,以此类推(只传 meta tensor,不传数据,更快且避免在 src rank 意外激活 CUDA context)。

非首 stage 为每个 microbatch chunk 预分配接收 buffer `args_recv_info`,且 buffer 设 `requires_grad_(True)`——**收来的激活成为本 stage 新 autograd 图的叶子**。

> **DTensor 跨 stage 问题**:DTensor 不能跨 PP stage 序列化(ProcessGroup 不可序列化)。`_build_get_mesh_callback`(`pipeline_parallel.py:44`)让每个 stage 收到普通 tensor 后用本地 mesh 重新 wrap 成 DTensor。

### 2.4 microbatch 切分

调度的 `step()` 把整批输入用 `split_args_kwargs_into_chunks`(`[pt] microbatch.py:245`)沿 batch 维(`DEFAULT_CHUNK_DIM=0`)`tensor_split` 成 `n_microbatches` 份。torchtitan 侧 `n_microbatches = local_batch_size // pipeline_parallel_microbatch_size`,强制整除;`n_microbatches < num_total_stages` 时告警提示气泡(`pipeline_parallel.py:267`)。

### 2.5 切分后:每个 stage 各自再走一遍 SPMD 并行

`pipeline_llm` 切完 stage 后,**每个 model chunk 各自再走一遍完整 `parallelize_fn`**(`pipeline_parallel.py:114`)——所以每个 stage 内部仍可叠加 TP/CP/EP/FSDP。这就是 PP 与其他维度组合的方式:PP 在最外层切,每个 stage 是一个独立的 SPMD 单元。

---

## 3. 通信原语:P2P send/recv

PP **不用 collective**,全部用 **P2P**。

### 3.1 底层原语

核心是 `dist.P2POp` + `dist.batch_isend_irecv`(`[pt] distributed_c10d.py`):`P2POp` 封装 `(op, tensor, peer)`,`op` 是 `dist.isend`/`dist.irecv`(异步)。`batch_isend_irecv` 把一批 P2POp 一次发出,对 NCCL 用 `_coalescing_manager` 把多个 P2P 融成一次内核启动,返回 `Work` 句柄需 `wait()`。

`schedules.py` 三个封装:`_batch_p2p`(薄包)、`_sorted_batch_p2p`(按 peer 分组分别发,避免死锁)、`_wait_batch_p2p`(逐个 `wait`)。

### 3.2 激活与梯度的 send/recv

stage 提供四个 ops 生成函数(只**构造** P2POp,不执行):

| 函数 | 作用 |
|------|------|
| `get_fwd_recv_ops` | 收来自 stage-1 的激活 |
| `get_fwd_send_ops` | 把本 stage 输出激活发给 stage+1 |
| `get_bwd_recv_ops` | 收来自 stage+1 的"对本 stage 输出的梯度" |
| `get_bwd_send_ops` | 把本 stage 对其输入的梯度发给 stage-1 |

激活在 `forward_one_chunk` 算完后存进 `fwd_cache[chunk_id]`,`get_fwd_send_ops` 读它发送、`backward_one_chunk` 读它反传。通信不在 stage 内部触发,而是由 **schedule 的 `_step_microbatches`** 编排。

---

## 4. 调度与气泡:五种 schedule 对比

记号:`p` = PP 度数,`m` = microbatch 数,`v` = 每 rank 的 virtual stage 数。`F`=forward,`B`=full backward,`I`=backward-input(对输入求梯),`W`=backward-weight(对权重求梯),空格=气泡。

### 4.1 GPipe(fill-drain)

`_step_microbatches` 两个完全分开的循环:先把所有 `m` 个 microbatch 全部 forward,再全部 backward。

```
GPipe, p=4, m=4
rank0: F0 F1 F2 F3 . . . . . . B0 B1 B2 B3
rank1: . F0 F1 F2 F3 . . . . B0 B1 B2 B3 .
rank2: . . F0 F1 F2 F3 . . B0 B1 B2 B3 . .
rank3: . . . F0 F1 F2 F3 B0 B1 B2 B3 . . .
       └warmup┘          └─ 全 F 完才开始全 B ─┘
```

气泡占比 ≈ `(p-1)/(m+p-1)`。**峰值显存最差**:第一个 backward 前,所有 `m` 个 microbatch 的激活同时驻留。

### 4.2 1F1B(`Schedule1F1B`)

三段:warmup(只 forward,rank `r` 跑 `min(m, num_stages-r)` 个)→ **1B1F 稳定态**(先 1 个 B 再 1 个 F)→ cooldown(剩余 B)。

```
1F1B, p=4, m=8  (B 占 2 时隙)
rank0: F0 F1 F2 F3 F4 B0 F5 B1 F6 B2 F7 B3 B4 B5 B6 B7
rank3: ...... F0 B0 F1 B1 F2 B2 F3 B3 F4 B4 F5 B5 F6 B6 F7 B7
       └warmup┘└──── steady 1F1B ────┘└─ cooldown ─┘
```

**1F1B 如何稳定峰值显存**:warmup 阶段 rank `r` 只缓存 `num_stages-r` 份激活(首 stage 最多 `p` 份,而非 GPipe 的 `m` 份)。进稳定态后每做 1 个 F 就立刻做 1 个 B 释放一份激活,**in-flight 激活数恒定 ≈ `num_stages-stage_index`,与 m 无关**。这就是 1F1B 把峰值显存从 `O(m)` 压到 `O(p)` 的本质——"一前一后"让激活的产生与消费速率匹配。气泡占比与 GPipe 同阶,但显存大幅下降。

### 4.3 Interleaved 1F1B(`ScheduleInterleaved1F1B`)

多段调度,每 rank `v≥2` 个 stage(`loop` 风格分布)。每 rank 在自己的 `v` 个 stage 间交错跑。

气泡占比 ≈ `(p-1)/(v·m+p-1)` ≈ **1F1B 的 1/v**。关键收益:warmup/cooldown 绝对长度仍 `~p-1`,但稳定态被拉长 `v` 倍(每 rank 有 `v·m` 个 forward),气泡相对占比降为 1/v。代价:每个 microbatch 的 stage 间 hop 从 `p-1` 变成 `v·p-1`,P2P 通信量增加 `v` 倍。

### 4.4 Zero Bubble V(`ScheduleZBVZeroBubble`)

多段调度,**强制每 rank 恰好 2 个 stage**,`v` 风格映射。它直接生成含 `I`(BACKWARD_INPUT)和 `W`(BACKWARD_WEIGHT)的动作流——**把 backward 拆成 I 和 W 两步**,`W` 可被推迟去填气泡(原理见 §7)。

气泡占比:理论上当 `T_F ≈ T_I ≈ T_W` 时**接近 0**。真实模型三者不等,所以只是"接近"。V 形 + 早做 I 还让激活更早释放,显存优于 1F1B。

### 4.5 DualPipeV(`ScheduleDualPipeV`)

唯一直接继承 `_PipelineScheduleRuntime` 的内置 schedule(源自 DeepSeek DualPipe)。同样每 rank 2 stage、`v` 映射。两个特色:

- **`OVERLAP_F_B` 动作**:把"一个 stage 的 forward"和"另一个 stage 的 backward"打包成一个动作,让 forward 计算与 backward 计算**在同一时隙重叠执行**(两个 GEMM 同时占 SM)。
- **双向流水**:microbatch 分两半,一半从 stage 0 正向喂、另一半从 stage N-1 反向喂,V 形折叠后两个方向的流水在每个 rank 上同时存在。

气泡接近 0,计算单元利用率最高。代价:同时维护两条流水,激活显存近似翻倍。

### 4.6 气泡占比对照表

| Schedule | 类基 | stage/rank | 气泡占比 | 峰值激活显存 | backward 拆分 |
|---|---|---|---|---|---|
| GPipe | `PipelineScheduleSingle` | 1 | `(p-1)/(m+p-1)` | `O(m)` 最差 | 否 |
| 1F1B | `PipelineScheduleSingle` | 1 | `(p-1)/(m+p-1)` | `O(p)` | 否 |
| Interleaved1F1B | `PipelineScheduleMulti` | v≥2 | ≈ 1F1B 的 1/v | `O(v·p)` | 否 |
| ZBVZeroBubble | `PipelineScheduleMulti` | =2 | ≈0(T_F=T_I=T_W 时) | 优于 1F1B | 是(I+W) |
| DualPipeV | `_PipelineScheduleRuntime` | =2 | ≈0 | ≈2× | 是(I+W)+OVERLAP_F_B |

---

## 5. 通信掩盖:action-based runtime

### 5.1 单段 schedule 的掩盖手法(以 1F1B 为例)

`Schedule1F1B._step_microbatches` 三处技巧:
1. **发送不立即 wait**:warmup 里 `_batch_p2p(fwd_sends)` 后不马上 wait,先跑下一个 chunk 的 forward,下一轮开头才 wait——isend 与下一个 forward 计算重叠。
2. **双向 P2P 融合**:稳定态把"发激活"和"收梯度"融进同一个 `_batch_p2p`,一次 NCCL coalesced 调用同时承载上下行流量。
3. **recv 紧贴计算**:计算前才 wait 对应的 recv,让 irecv 在前一步计算期间已在后台传输。

### 5.2 `_PipelineScheduleRuntime` 的 action-based 编排

这是通信掩盖最彻底的实现。核心思想:**把 SEND_F / RECV_F / SEND_B / RECV_B 提升为与 FORWARD / BACKWARD 平级的一等动作**,由 runtime 按动作流顺序执行,从而把"发起通信"和"等待通信"在时间上拉开,中间塞计算。

**调度降级(lowering)**:`_prepare_schedule_with_comms` 把 compute-only 的 `pipeline_order` 转成含 SEND/RECV 的 `pipeline_order_with_comms`。`_add_send_recv` 给每个 compute 动作配上通信动作:`F → (SEND_F@stage, RECV_F@stage+1)`,`I/B → (SEND_B@stage, RECV_B@stage-1)`。

**runtime 执行**:逐 `time_step` 遍历动作流:
- `SEND_F`/`SEND_B`:立即发起 P2P,`Work` 攒进 `send_ops` 列表,**不 wait**。
- `RECV_F`/`RECV_B`:立即发起 irecv,`Work` 存进字典,**不 wait**。
- `FORWARD`/`BACKWARD`:**计算前**才 wait 对应的 recv——此时 irecv 早已在前面若干 time_step 发起,数据大概率已到。
- `BACKWARD_WEIGHT`:纯计算无通信依赖——正是 zero-bubble 把 `W` 拿来填气泡的落点。
- 所有 `send_ops` 在 `_step_microbatches` 末尾统一 `wait`。

**掩盖如何发生**:lowering 把 `RECV_F` 排在需要它的 `FORWARD` 之前好几个 time_step。runtime 执行到 `RECV_F` 时只是发起 irecv 立刻返回,接下来若干 time_step 跑别的计算,等真正轮到那个 `FORWARD` 时 `wait` 几乎瞬间返回。**通信被计算"夹住"了。**

**V-schedule 特例**:相邻 stage 在同一 rank(V 折点)时,完全跳过 send/recv,直接传 tensor 引用——零拷贝、零通信。

---

## 6. 异步实现:isend/irecv 与 wait 时机

### 6.1 异步链条

1. `get_*_ops` 只**构造** `P2POp(dist.isend/irecv, ...)`,这俩 API 本身非阻塞。
2. `_batch_p2p` → `batch_isend_irecv`,NCCL 路径用 `_coalescing_manager` 把整批 P2P 合成一次 coalesced 内核启动,返回一组 `Work`。
3. `Work.wait()` 阻塞当前流直到该通信完成。

### 6.2 wait 的调用时机

核心模式:**recv 早发起、用前才 wait;send 发起后不等、攒到最后或下一轮才 wait**。中间窗口全部被计算填满。

| 场景 | recv wait 时机 | send wait 时机 |
|---|---|---|
| GPipe | 计算前立即 wait | warmup 的 fwd send 攒到 backward 前才 wait |
| 1F1B | 计算前 wait,与反向 recv 融合 | 延迟一轮,下一个 chunk 计算前才清 |
| `_PipelineScheduleRuntime` | RECV 发起后存字典,FORWARD/BACKWARD 计算前才 pop+wait | SEND 攒进 `send_ops`,`step` 末尾统一 wait |

action-based runtime 把依赖**显式编码进动作流顺序**:`_ready_to_schedule` 保证 `FORWARD@stage` 一定排在对应 `RECV_F@stage` 之后。runtime 顺序执行 + "用前 wait" 两条规则一起,既保证正确性(数据到了才算)又实现掩盖。

`_initialize_stage` 里用 dummy tensor 与前后邻居 isend/irecv 提前建好 NCCL P2P 通信器,避免首次真实通信的握手开销混入关键路径。

---

## 7. Zero Bubble 原理:把 backward 拆成 I 和 W

### 7.1 为什么拆

一次标准 backward 同时算两类梯度:

- **dInput(I)** —— 对本 stage 输入的梯度。必须立刻算出并发给上一个 stage,**在流水线关键路径上**,上游等着它。
- **dWeight(W)** —— 对本 stage 权重的梯度。只需在 optimizer step 前 ready,**不在关键路径上**,谁也不等它。

标准 `B` 把两者绑死,意味着 `W` 的计算时间也卡在关键路径里,产生气泡。Zero Bubble 的洞见:**把 `W` 解耦出来,推迟到流水线本来空闲(气泡)的时隙去算**。关键路径上只剩 `F` 和 `I`。当 `T_F ≈ T_I ≈ T_W` 时,推迟的 `W` 恰好填满所有气泡 → 接近零气泡。

```
标准 1F1B:        F F F F B B B B          B 含 I+W,W 卡在关键路径 → 有气泡
                  ────────────────►
Zero Bubble:      F F F F I I I I          I 在关键路径
                          W W W W          W 被推迟填进原本的气泡时隙
```

### 7.2 `_backward.py` 的两步实现

**`stage_backward_input`** —— 算 dInput:`torch.autograd.grad(stage_outputs, inputs=input_values, grad_outputs=output_grads, retain_graph=True)`。注意 `inputs=` 只指定 stage 输入、**不碰权重**;`retain_graph=True` 因为 `W` 还要用图。同时在中间节点上注册 hook 把流经的梯度缓存进 `param_groups["grads"]`——这是 `W` 步骤的输入。`dinputs` 立刻通过 `get_bwd_send_ops` 发给上游。

**`stage_backward_weight`** —— 算 dWeight:从 `param_groups["grads"]`(I 步骤里 hook 缓存的中间梯度)出发,用 `GradientEdge` 作为 autograd 端点,只算到权重,把 dW 累加进 `weight.grad`。

runtime 侧:`BACKWARD_INPUT` 动作调 `backward_one_chunk(full_backward=False)`,`BACKWARD_WEIGHT` 动作调 `backward_weight_one_chunk`。schedule 生成器负责把 `I` 排在关键路径、把 `W` 排进气泡时隙。

### 7.3 DualPipeV 的进一步压榨

DualPipeV 在 ZBV(每 rank 2 stage + V 折叠 + I/W 拆分)基础上再加 **`OVERLAP_F_B`**:把一个 stage 的 forward 和另一个 stage 的 backward 打包,**计算本身重叠发起**——一个 forward 的 GEMM 和一个 backward 的 GEMM 同时占用 SM,把"forward 在算时 backward 单元闲着"的浪费也消除。配合双向流水,DualPipeV 的气泡占比是所有内置 schedule 里最低的。

---

## 8. 完整流程图

```
═══ 建模期 ═══
pipeline_llm()                                  pipeline_parallel.py:68
  ├─ _get_pipeline_metadata()        决定 virtual stage 数
  ├─ _generate_llm_fqn_per_model_part()  每 stage 的模块名
  ├─ _pipeline_module_split()
  │    ├─ _get_pp_rank_to_stage_indices_mapping()  loop / v 映射
  │    ├─ _split_module()  deepcopy + 删除非本 stage 的层
  │    └─ PipelineStage(...)  形状沿 stage 链推断
  ├─ 每个 model chunk 各自走一遍 parallelize_fn(叠加 TP/CP/EP/FSDP)
  └─ _build_pipeline_schedule()      schedule 工厂

═══ 训练期 ═══
pp_schedule.step(inputs, target)
  ├─ split_args_kwargs_into_chunks()  切 microbatch
  └─ _step_microbatches()
       GPipe   : 全 F → 全 B,峰值显存 O(m)
       1F1B    : warmup → 1B1F 稳定态 → cooldown,峰值显存 O(p)
       Runtime : 动作流 SEND_F/RECV_F/SEND_B/RECV_B/F/B/I/W
                 ├─ RECV 早发起,FORWARD/BACKWARD 计算前才 wait  ┐ 通信掩盖
                 ├─ SEND 发起不等,step 末尾统一 wait            ┘
                 └─ W(BACKWARD_WEIGHT)填进气泡时隙              ← Zero Bubble
            通信原语:get_*_send/recv_ops → batch_isend_irecv(P2P)
```

---

## 9. 小结

- **模型切分**:torchtitan `_split_module` 用 `copy.deepcopy` + 删除非本 stage 的层(模型 forward 须容忍 `None` 子模块);`PipelineStage` 沿 stage 链顺序推断形状。每个 stage 切完各自再走一遍 `parallelize_fn`,所以 PP 可与 TP/CP/EP/FSDP 组合。
- **通信原语**:PP 全用 **P2P**(`isend`/`irecv` + `batch_isend_irecv`),stage 间只传激活(forward)和梯度(backward),不用 collective。
- **调度与气泡**:GPipe(气泡大、显存 `O(m)`)→ 1F1B("一前一后"把显存压到 `O(p)`)→ Interleaved1F1B(气泡降 `1/v`)→ ZBV / DualPipeV(气泡接近 0)。
- **通信掩盖**:`_PipelineScheduleRuntime` 的 action-based 编排把 SEND/RECV 提升为一等动作,**recv 早发起、用前才 wait,send 发起不等**,中间塞计算。
- **异步实现**:`isend`/`irecv` 非阻塞返回 `Work`;NCCL 下 `batch_isend_irecv` 用 coalescing 合并;"用前 wait"模式实现通信/计算重叠。
- **Zero Bubble**:把 backward 拆成 `I`(对输入的梯度,在关键路径)和 `W`(对权重的梯度,不在关键路径),把 `W` 推迟填进气泡时隙;`T_F≈T_I≈T_W` 时气泡趋近 0。DualPipeV 再加 `OVERLAP_F_B` 让 F/B 计算重叠 + 双向流水。

## Related Pages

- [[torchtitan/index]] · [[torchtitan_parallel_dims_analysis]] —— 知识地图与并行基座
- [[torchtitan_cp_analysis]] · [[torchtitan_ep_analysis]] —— 相邻并行维度
- [[pp_schedulers_analysis]] —— Megatron-LM 流水线 5 调度器、气泡公式推导、流水线模拟图
- [[megatron_pp_parallelism_analysis.html]] —— PP 并行:1F1B/VPP/Combined 调度、P2P 通信、Bubble 分析
- [[comm_compute_overlap_analysis.html]] —— combined_1f1b vs ZBV/DualPipe、sub-layer 级调度
