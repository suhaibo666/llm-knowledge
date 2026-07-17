# ACLGraph 状态化捕获：多流依赖与 graph-safe RNG

> **Source baseline**：`torch_npu v2.7.1@b3c8a815b4bf6f8ec28b418aa9ec42815db0d91e`；其记录的 `op-plugin` gitlink 为 `6ef73e3994433d2804eaf29b1f9f45b730d49087`；PyTorch upstream 为 `main@2b460d01b8a5d2c12188b9ea8f9b59a58b9f6a09`。
> **Last updated**：2026-07-16
>
> 当前工作区的 `op-plugin` checkout 是较旧的 `b7a17be03c75...`，没有停在 torch_npu 记录的 gitlink 上。因此本文凡标为 `op-plugin@6ef73e3` 的位置，均来自本地 Git 对象 `git show 6ef73e3:<path>`，不能用当前工作树行号直接跳转。

---

## 1. 核心结论：捕获的不只是算子，还包括依赖与状态协议

普通确定性单流算子只要把 device task 记录下来即可。多流和随机数之所以需要额外适配，是因为它们各自携带了图外信息：

| 问题 | 图外信息 | ACLGraph 必须记录或间接化的内容 |
|---|---|---|
| 多流 | 不同 stream 之间的先后关系 | Event Record / Wait 形成的依赖边，以及所有参与 stream 上的 task |
| RNG | CPU generator 中持续变化的 seed / Philox offset | 地址固定的 device seed/offset tensor、图内静态 offset、每次 replay 前的状态推进 |

所以需要先纠正两个常见理解：

1. **多流捕获不是 fork 多个独立图，最后再同步合并。** `NPUGraph::capture_begin()` 只创建一个 `model_ri_`，其他 stream 通过 event 依赖加入这一次 capture；`capture_end()` 得到的仍是同一个模型，replay 也只有一次 `AclmdlRIExecuteAsync(model_ri_, stream)`（`torch_npu/csrc/core/npu/NPUGraph.cpp:151-225,227-258,267-281`）。
2. **随机数入图不是“CPU 完全不能生成随机数”。** 真正的问题是 replay 不会重新执行 capture 时的 Python/C++ 取 seed 逻辑；若把 seed/offset 作为普通 host 标量固化进图，每次 replay 就会复用同一段随机序列。torch_npu 的解法是让图中 kernel 间接读取 device tensor，而在每次 replay 前由 generator 更新这些 tensor（`torch_npu/csrc/aten/NPUGeneratorImpl.h:20-52`）。

dropout 是两种适配同时出现的代表：随机 mask 需要 graph-safe RNG，mask 又在 secondary stream 上生成，需要 event fork/join 和跨流内存保活。

---

## 2. 多流捕获：一个模型中的 fork/join DAG

### 2.1 `wait_stream()` 就是 `record_event()` + `wait_event()`

Python 实现没有额外魔法：

```python
# torch_npu/torch_npu/npu/streams.py:42-69
def wait_stream(self, stream):
    self.wait_event(stream.record_event())
```

若写成 `dst.wait_stream(src)`，等价于：

```python
event = src.record_event()  # 在 src 当前提交点之后记录 event
dst.wait_event(event)       # dst 此后的任务等待 event
```

C++ 层 `NPUEvent::record()` 向源流下发 Record Event task，`NPUEvent::block()` 向目标流下发 Wait Event task（`torch_npu/csrc/core/npu/NPUEvent.cpp:115-145`）。两个 API 都是异步下发，CPU 不会原地等待；`wait_event()` 的注释也明确“只影响未来提交到该流的工作”（`torch_npu/torch_npu/npu/streams.py:28-54`）。

它只建立一条**单向、局部**偏序：

```text
src_before -> record(E) -> wait(E) -> dst_after

没有由这次调用建立顺序：
  dst_before  ?  src_before
  src_after   ?  dst_after
```

因此 `side_stream.wait_stream(capture_stream)` 的准确含义是：调用点以前已经提交到主流的任务，必须先于 side stream 在 wait 之后提交的任务完成。它不约束主流在 event 之后的新任务与 side stream 后续任务，两者可以并行。

`wait_stream` 与显式 Event 的关系可以进一步拆开：

| 写法 | 建立的依赖 | 适合场景 |
|---|---|---|
| `dst.wait_stream(src)` | 立即在 `src` 当前点创建并记录一个新 event，再让 `dst` 等它 | 一次性的“截至此刻，dst 等 src” |
| `event.record(src)`；`event.wait(dst)` | 把“记录里程碑”和“谁来等待”分开 | 一个完成点扇出给多条流，或稍后才决定消费者 |
| 只有 `event.record(src)` | 只放置里程碑，没有任何消费者等待 | 查询进度、计时；**不会单独产生跨流顺序** |
| 只有 `event.wait(dst)` | 等待该 event 最近一次 record 所代表的完成点 | 使用已由别处记录的同步点 |

`Event.record()` 默认记录当前流，`Event.wait()` 默认阻塞当前流未来的工作（`torch_npu/torch_npu/npu/streams.py:113-160`）。因此真正决定先后关系的是 **record/wait 在各自 stream 队列中的位置**，不是 Python `with torch.npu.stream(...)` 的视觉嵌套。把 wait 提前或延后一个算子，得到的 DAG 就不同。

### 2.2 `with graph()` 下的第二条流如何进入同一次捕获

典型 fork/join 写法如下，torch_npu 的测试正是这一结构（`torch_npu/test/npu/test_aclgraph_multi_stream.py:93-116`）：

```python
with torch.npu.stream(capture_stream):
    with torch.npu.graph(g, stream=capture_stream):
        side_stream.wait_stream(capture_stream)  # fork：主流 -> side
        with torch.npu.stream(side_stream):
            branch_b = torch.matmul(x, wb)

        branch_a = torch.matmul(x, wa)           # 与 branch_b 可重叠
        capture_stream.wait_stream(side_stream)  # join：side -> 主流
        output = branch_a + branch_b
```

CPU 仍按 Python 顺序提交这些调用，但这个顺序不等于 device 执行顺序。capture 记录到的依赖图是：

```text
                         +-> branch_a --------+
prefix -> fork event ----+                     +-> join -> output
                         +-> branch_b (side) --+
```

- `fork` 之前的 `prefix` 是两条分支共同前驱。
- `branch_a` 与 `branch_b` 之间没有 event 或数据依赖，可以并行。
- `output` 读取两条分支结果，所以必须放在 join 之后。
- capture 阶段记录的是两条流的 task 与 event 节点；replay 阶段 Python 上下文和 `wait_stream()` 不会再跑，而是由一次 `AclmdlRIExecuteAsync` 重放整个多流 DAG（`NPUGraph.cpp:267-281`）。

把 capture 与 replay 两个阶段逐项对齐，会更清楚：

| 阶段 | CPU 做什么 | device/runtime 看到什么 |
|---|---|---|
| 进入 `with graph()` | 同步设备、切换到 capture stream、调用 `capture_begin()` | 主流进入一次 `model_ri_` 捕获（`torch_npu/npu/graphs.py:855-874`） |
| Python 走主流代码 | 逐个调用算子下发 API | 主流 task 被记录，不代表 Python 调用之间都成为跨流依赖 |
| `side.wait_stream(main)` | 串行调用 `record_event`、`wait_event` 后立即返回 | Record/Wait 节点把 side stream 拉进同一个 capture |
| 切到 side stream 下发 B | CPU 仍是顺序执行 API | B 记录在 side 队列；只受 fork event 和 side 自身 FIFO 约束 |
| 回主流下发 A | CPU 调用发生在 B 的下发之后 | A 与 B 没有依赖，可以由 runtime 并发调度 |
| `main.wait_stream(side)` | 再下发一对 Record/Wait | side 分支返回主流，形成 join |
| 退出上下文 | 在开始 capture 的同一主流调用 `capture_end()` | 校验所有参与流都已闭合，得到一个 `model_ri_`（`NPUGraph.cpp:227-255`） |
| `g.replay()` | 只做 replay prologue 和一次 model execute | runtime 按图中保留的 stream FIFO 与 event 边重放 A/B/join，不再执行 Python `with` 和 `wait_stream` |

> [!important] 对“两个流都在捕获吗”的精确回答
> 只有主流显式执行了一次 `AclmdlRICaptureBegin`；side stream 通过主流 Record Event、side Wait Event **加入这一次捕获**。加入后，两条流上的 task 都记录进同一个模型，但并不存在两份独立 graph，也没有 capture 后的 CPU 侧“合图”步骤。

第二条流并不是因为语法上位于 `with graph()` 代码块内就自动入图。CANN 的规则是：主流 Record Event、其他流 Wait Event 后，其他流才直接或间接加入同一模型；参与流最终还必须通过 event 直接或间接返回主流，否则 capture end 报错。返回主流后到 capture end 之间也不能再往该 side stream 下发未重新关联的 task。见 CANN 官方[跨流捕获](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/programug/acldevg/runtime_doc_dev_0031.html)。

### 2.3 更复杂的三流例子：一条主干与一条两级 side pipeline 重叠

下面的结构让 `stage2_stream` **间接**加入主流 capture，并最终逐级返回：

```python
with torch.npu.graph(g):
    main = torch.npu.current_stream()
    prefix = preprocess(x)                    # 共同前驱

    stage1_stream.wait_stream(main)           # main -> stage1
    with torch.npu.stream(stage1_stream):
        side_1 = shared_expert(prefix)

    stage2_stream.wait_stream(stage1_stream)  # stage1 -> stage2，间接加入 capture
    with torch.npu.stream(stage2_stream):
        side_2 = stage2(side_1)

    stage1_stream.wait_stream(stage2_stream)  # stage2 -> stage1
    with torch.npu.stream(stage1_stream):
        side_3 = side_projection(side_2)

    routed = routed_expert(prefix)             # 与整条 side pipeline 可重叠
    main.wait_stream(stage1_stream)            # stage1 -> main，最终闭合
    out = combine(routed, side_3)
```

它记录出的关键偏序是：

```text
                         +-> routed_expert -----------------------+
prefix -> fork(main→s1) -+                                       +-> combine
                         +-> side_1 -> s1→s2 -> side_2 -> s2→s1
                                                   -> side_3 -----+
```

CPU 会先提交 `side_1/side_2/side_3` 的 API，再提交 `routed_expert`，但 device 上 routed 分支只依赖 `prefix`，可与整条 side pipeline 重叠。真实 MoE 可以把 `stage1` 映射为 shared expert 计算、`stage2` 映射为通信或另一阶段处理；若 `stage2` 是 HCCL，还必须单独验证目标 collective 的 ACLGraph 支持，而不能仅凭这个依赖模板推定可入图。

### 2.4 忘记 `capture_stream.wait_stream(side_stream)` 会发生什么

要区分 eager 与 capture：

- **ACLGraph capture 中**：side stream 已通过 fork 进入捕获，却没有回到主流，`AclmdlRICaptureEnd` 会因捕获拓扑未闭合而报错。即使 side 的结果最终未被使用，也不能省掉闭合边。
- **普通 eager 多流中**：如果主流直接读取 side 产出的 tensor，则形成数据竞争，结果可能未完成、陈旧或不确定；allocator 还可能在 side 使用结束前复用内存。
- **在 `with graph()` 外再 `torch.npu.synchronize()` 不能补救捕获缺失的 join**：退出上下文时 `capture_end()` 已先执行并失败。同步只能等待已提交任务，不能补写模型内部缺失的依赖边。

`record_stream()` 解决的是另一个维度：event 管**执行顺序**，allocator 的 `recordStream(ptr, usage_stream)` 管**对象寿命**。HCCL 的实现先让通信流等待当前计算流，再用 `recordStream` 防止输入在通信流用完前被回收（`torch_npu/csrc/distributed/ProcessGroupHCCL.cpp:245-270,4068-4099`）。两者通常必须配合，不能互相替代。

### 2.5 实际应用场景

| 场景 | 主流 | side stream | 想隐藏的延迟 |
|---|---|---|---|
| MoE shared expert | routed expert / 主干计算 | shared expert | 两路 expert 计算互相重叠；torch_npu 已有同型测试（`test_aclgraph_multi_stream.py:16-60`） |
| 并行计算分支 | 分支 A | 分支 B | 独立 matmul、预处理或后处理并行 |
| dropout | `DropoutDoMask` 与主干 | `DropoutGenMask` | mask 生成与可并行工作重叠 |
| DDP/FSDP/TP/MoE 通信 | 计算流 | HCCL all-reduce、all-gather、reduce-scatter、all-to-all 流 | 通信与无依赖计算重叠 |
| pipeline P2P | 当前 stage 计算 | send/recv 流 | 激活或梯度传输与下一段计算重叠 |

通信流属于多流的一种。`ProcessGroupHCCL` 持有独立的 `hcclStreams_` / `hcclEvents_`（`ProcessGroupHCCL.hpp:1051-1056`），collective 入口先 `syncStreams()`，随后在 HCCL stream 下发通信，并在结束处记录事件（`ProcessGroupHCCL.cpp:3944-3971,4115-4136`）。它还在 capture 时跳过普通 async-error work enqueue（`:4153-4155`），说明实现意识到了捕获状态。

通信流的完整依赖通常也分成两半：

```text
计算流产出输入
    -> syncStreams: current stream Record / HCCL stream Wait
    -> HCCL collective 在通信流执行
    -> hcclEndEvent 在通信流 Record
    -> WorkHCCL::synchronizeInternal: 当前流 Wait
    -> 当前流安全消费通信结果
```

前半段由 `syncStreams()` 保证通信不会抢读尚未生产完的输入（`ProcessGroupHCCL.cpp:245-270`）；后半段由 `WorkHCCL::synchronizeInternal()` 让当前流等待 `hcclEndEvents_`（`:988-999`）。如果上层只发起 async collective，却没有在消费结果前形成后半段 wait，逻辑上仍缺少 `comm -> compute` 的依赖。ACLGraph capture 还要求这个返回关系在 capture end 前闭合。

但这只能证明 **HCCL 是多流且有 capture-aware 分支**，不能推出“所有 collective、所有 SoC 都能 ACLGraph 入图”。本基线没有找到覆盖 ACLGraph + HCCL 的端到端测试；实际支持还取决于 CANN/HCCL 版本、具体 collective 和拓扑，应按目标环境实测。

---

## 3. 随机数入图：把 host 标量改成 replay 时可更新的 device 状态

### 3.1 为什么直接捕获 seed/offset 会冻结随机序列

eager 随机算子通常执行：

```text
CPU generator 取 (seed, offset) -> offset += increment -> 下发随机 kernel
```

若 capture 时把 `(seed, offset)` 作为普通标量写进 kernel 参数，replay 只重放已记录的 kernel，不会再次运行“取状态并推进 offset”的 C++ 代码，于是每次得到相同随机数。正确协议需要满足：

1. 同一次图内多个随机算子不能使用重叠 counter 区间。
2. 每次 replay 必须从 generator 当前全局 offset 开始。
3. 一次 replay 后，全局 offset 要前进整张图的总消耗量。
4. 重设 seed 后，同样的 eager/graph 调用顺序应可复现。

这里 CPU、NPU 的职责应分开理解：CPU generator 并不直接生成整块随机 tensor，它维护 seed 和 Philox counter，并把某一段 counter 区间分配给 kernel；真正的随机值由 NPU kernel 生成。capture 的问题不是“CPU 在 graph 模式完全不可运行”，而是 **per-op 的 CPU 分配逻辑只在 capture 时走一遍、replay 时不会重走**。实际上每次 `g.replay()` 仍会在 CPU 进入 `NPUGraph::replay()`，运行一次整图级 `replay_prologue()`，然后才下发 `AclmdlRIExecuteAsync`（`torch_npu/csrc/core/npu/NPUGraph.cpp:267-281`）。

为什么不在 replay 时重新逐算子调用 generator？因为这会重新引入每个随机算子的 host dispatch，违背 graph replay 把一串操作压成一次提交的目的。device tensor 间接寻址只需在图外更新一次 base offset，图内所有随机 kernel 用静态 intragraph offset 自行定位各自区间。

### 3.2 `NPUGeneratorImpl` 的两层 offset

torch_npu 把一张含 RNG 的图视为一个“大 kernel”（`torch_npu/csrc/aten/NPUGeneratorImpl.h:20-52`）：

- `offset_extragraph_`：一元素 NPU tensor，保存本次 replay 的图外起始 offset。
- `offset_intragraph_`：capture 时在 host 侧累计的图内静态偏移。
- `wholegraph_increment`：capture end 得到的整图 RNG 总消耗。

假设图中两个算子分别消费 `I1`、`I2`，capture 会记录：

```text
op1 offset = *offset_extragraph + 0
op2 offset = *offset_extragraph + I1
wholegraph_increment = I1 + I2
```

第 `k` 次 replay 前若 generator 全局 offset 为 `O_k`，`replay_prologue()` 把 `O_k` 写入 `offset_extragraph_`，再把全局 offset 推进到 `O_k + I1 + I2`。图内两个 kernel 分别读到 `O_k` 和 `O_k + I1`；下一次 replay 自然从新区间开始。

一个具体例子：图中 `uniform_` 消耗 12、dropout 消耗 8，capture 得到 `wholegraph_increment=20`。若 replay 前全局 offset 为 100：

| replay | device base offset | `uniform_` 使用 | dropout 使用 | replay 后全局 offset |
|---|---:|---:|---:|---:|
| 第 1 次 | 100 | 100 | 112 | 120 |
| 第 2 次 | 120 | 120 | 132 | 140 |

所以连续 replay 不会重复；若在图外重新 `manual_seed(s)`，`set_current_seed()` 会同时把全局 Philox offset 清零（`NPUGeneratorImpl.cpp:252-258`），相同 seed 和相同调用序列又能复现。

对应源码生命周期是：

| 阶段 | 动作 | 证据 |
|---|---|---|
| 注册 | 第一个 graph 注册时分配一元素 NPU seed/offset tensor | `NPUGeneratorImpl.cpp:136-154` |
| capture begin | 默认 generator 自动注册；`capturing_=true`、图内 offset 清零、填 device tensor | `NPUGraph.cpp:176-181`；`NPUGeneratorImpl.cpp:180-186` |
| 算子 capture | `philox_npu_state(increment)` 返回 device tensor 指针 + 当前 intragraph offset，并累计 increment | `NPUGeneratorImpl.cpp:461-475` |
| capture end | 保存 `wholegraph_increment` | `NPUGraph.cpp:257-258`；`NPUGeneratorImpl.cpp:192-196` |
| replay | 先填当前 seed/global offset，再推进整图 offset，最后执行 model RI | `NPUGeneratorImpl.cpp:202-213`；`NPUGraph.cpp:267-281` |

时间顺序还有一个容易遗漏的细节：`capture_prologue()` 在 `AclmdlRICaptureBegin` **之前**运行（`NPUGraph.cpp:180-214`），`replay_prologue()` 在 `AclmdlRIExecuteAsync` **之前**运行（`:267-281`）。图内 kernel 捕获的是一元素 tensor 的固定地址，不是 capture 当时的数值；每次 replay 先在同一提交序列中刷新该地址上的值，再执行模型，所以既保持静态地址，又能改变随机状态。

`increase()` 会把 increment 向上对齐到 4；capture 时推进 `offset_intragraph_`，eager/replay 图外则推进 `philox_offset_per_thread_`（`NPUGeneratorImpl.cpp:98-130`）。increment 不能小于 kernel 实际消耗，否则会复用旧随机数；过大只会跳过 counter，不会产生重叠（`NPUGeneratorImpl.cpp:442-459`）。

默认 generator 在 `capture_begin()` 自动注册。显式传入自定义 NPU generator 时，需在 capture 前调用 `g.register_generator_state(generator)`；绑定接口见 `torch_npu/csrc/npu/Graph.cpp:772-780`。否则 `philox_npu_state()` 在 capture 中推进一个未进入 capturing 状态的 generator，会触发 `Attempt to increase offset for a NPU generator not in capture mode`。

### 3.3 单个算子 C++ 需要改什么

op-plugin 中完整的适配模式通常就在算子 C++ 的一处分支：

```cpp
auto status = c10_npu::currentStreamCaptureStatusMayInitCtx();
auto increment = calc_final_counter_offset(out);
if (status == c10_npu::CaptureStatus::None) {
    auto [seed, offset] = gen->philox_engine_inputs(increment);
    EXEC_NPU_CMD(aclnnXxx, ..., seed, offset);               // eager 标量接口
} else {
    auto state = gen->philox_npu_state(increment);
    EXEC_NPU_CMD(aclnnXxxTensor, ..., *state.seed_.ptr,
                 *state.offset_.ptr, state.offset_intragraph_); // graph Tensor 接口
}
```

这里不是只把 `if capture` 写上就结束，算子侧还要同时完成四件事：

1. **准确计算 counter increment**：由输出 numel、unroll、分片/索引方式决定。公共工具在 `op-plugin@6ef73e3:op_plugin/utils/RandomUtil.h:87-143`。
2. **准备 graph-safe ACLNN 变体**：例如 `aclnnInplaceUniformTensor`，其 seed/offset 是 tensor 输入，而不是 capture 时固化的 host 标量。
3. **保留 eager 路径**：非捕获仍走 `philox_engine_inputs()`；这个旧接口内部有 `assertNotCapturing`，可让未适配算子在 capture 时显式失败（`NPUGeneratorImpl.cpp:479-488`）。
4. **处理所有 overload、SoC 分支和 fallback**：只改主路径不代表整个公开 API 都支持。

`calc_final_counter_offset()` 本身也不是简单的 `numel`：在 `IsAclnnOnly()` 路径，它依据 block/grid 和 unroll 计算每线程需要跨过的 Philox 区间；`random_(from,to)` 的数值范围还会影响 unroll 是 2 还是 4；无法使用 32-bit indexing 的大 tensor 会递归分片并累加每片 counter。非 aclnn-only 的 A2/A3 路径当前仍直接返回 10（`op-plugin@6ef73e3:op_plugin/utils/RandomUtil.h:87-143`）。这正是测试必须覆盖大 tensor、非连续布局和不同整数范围的原因：**算子数值正确不等于 counter 消耗正确**，后者出错往往要到第二个随机算子或下一次 replay 才暴露。

所谓“通常只改一处算子 C++ capture 分支”，准确边界是：torch_npu 公共 generator/graph 框架已经存在时，某个简单随机原语的 op-plugin 主路径通常只需增加这一处分流；但完整交付仍可能涉及 ACLNN 新增 Tensor seed/offset 接口、所有 C++ overload、SoC 特化、兼容 fallback、自动生成配置和测试。它不是“整个功能只改一个文件”。

PyTorch factory API 会下沉到这些 inplace 原语：`torch.rand -> uniform_`、`torch.randn -> normal_`、`torch.randint -> random_`（`pytorch/aten/src/ATen/native/TensorFactories.cpp:1055-1056,1163,1358-1359`），所以底层一处适配可以覆盖多个公开入口。

### 3.4 当前 op-plugin 的支持边界

下表只按 `torch_npu` 所记录的 `op-plugin@6ef73e3` 源码判断“是否存在显式 capture 分支”，不是设备实测通过清单。

| 状态 | 算子/路径 | 源码证据与边界 |
|---|---|---|
| 明确有 Tensor capture 分支 | `uniform_` / `torch.rand` | `op-plugin@6ef73e3:op_plugin/ops/opapi/UniformKernelNpuOpApi.cpp:26-46` |
| 明确有 Tensor capture 分支 | `random_` / `torch.randint` | `op-plugin@6ef73e3:op_plugin/ops/opapi/RandomKernelNpuOpApi.cpp:118-156` |
| 明确有 Tensor capture 分支 | `multinomial` | `op-plugin@6ef73e3:op_plugin/ops/opapi/MultinomialKernelNpuOpApi.cpp:26-64` |
| 明确有 Tensor capture 分支 | `normal_(mean: scalar, std: scalar)` / 常见 `torch.randn` lowering | `op-plugin@6ef73e3:op_plugin/ops/opapi/NormalKernelNpuOpApi.cpp:41-67` |
| 明确有 Tensor capture 分支 | top-k/top-p 采样中的 multinomial 路径 | `op-plugin@6ef73e3:op_plugin/ops/opapi/TopKTopPSampleKernelNpuOpApi.cpp:27-50` |
| 多流 + RNG 联合适配 | 两阶段 `native_dropout` 的 mask 生成路径 | `op-plugin@6ef73e3:op_plugin/ops/opapi/NativeDropoutKernelNpuOpApi.cpp:65-119`；但 `IsAclnnOnly()` 的 `aclnnDropoutV3` 分支仍在 `:45-61` 使用标量接口 |
| 仍是旧标量接口 | `bernoulli_`、`randperm` | `op-plugin@6ef73e3:op_plugin/ops/opapi/BernoulliKernelNpuOpApi.cpp:28-53`；`op-plugin@6ef73e3:op_plugin/ops/opapi/RandpermKernelNpuOpApi.cpp:24-29` |
| SoC/分支不完整 | `exponential_` | Ascend950 直接路径在 `op-plugin@6ef73e3:op_plugin/ops/opapi/ExponentialKernelNpuOpApi.cpp:28-47` 仍取 host 标量；其他路径分解到已适配的 `uniform_` |
| overload 不完整 | functional `torch.normal` 的 Tensor/Scalar 多组重载 | `op-plugin@6ef73e3:op_plugin/ops/opapi/NormalKernelNpuOpApi.cpp:71-230` 多处只有 `if capture == None`，capture 时没有对应 Tensor 下发分支 |

因此“一个随机算子支持 ACLGraph”必须精确到 **API overload × dtype/shape × SoC × aclnn 分支**。看到同名文件里有一处 `philox_npu_state()`，不能推导所有重载都已支持。

> [!warning] 测试清单与固定 op-plugin 源码存在需要实机澄清的张力
> `torch_npu/test/test_npu.py:2433-2549` 的 `test_graph_rng_distributions` 罗列了 `bernoulli/normal/poisson/rand/randint/randn` 以及多种 inplace distribution；但固定的 `op-plugin@6ef73e3` 中，`bernoulli_` 等仍直接调用 capture 禁用的 `philox_engine_inputs()`。这可能来自测试移植、dispatch/SoC 分支或版本错位。故该测试名单只能代表**期望契约**，不能替代目标镜像上的逐项执行结果。

---

## 4. dropout：多流与 RNG 的交汇实例

`op-plugin@6ef73e3:op_plugin/ops/opapi/NativeDropoutKernelNpuOpApi.cpp:65-119` 展示了一次完整的状态化适配：

1. 保存 `original_stream`，取得框架 secondary stream（`:65-67`）。
2. capture 时调用 `philox_npu_state(10)`，取得 device seed/offset 与 intragraph offset（`:92-95`）。
3. 第一次使用 secondary stream 时，在 original stream Record Event、secondary stream Wait Event，把 side stream 纳入捕获（`:96-100`）。
4. 在 `SecondaryStreamGuard` 下用 `aclnnDropoutGenMaskV2Tensor` 生成 mask（`:103-107`）。guard 析构时在 secondary stream Record Event、original stream Wait Event，形成 join（`torch_npu/csrc/core/npu/SecondaryStreamGuard.cpp:22-26`）。
5. 将 generator 的 `secondary_stream_capture_state_` 标为 true，避免重复执行首次引流逻辑（`:108-110`；状态定义见 `NPUGeneratorImpl.h:105-126,130-138`）。
6. `recordStream(mask, original_stream)` 保护跨流 mask 的内存寿命，再在 original stream 下发 `DropoutDoMask`（`:115-119`）。

把这条路径放进 capture/replay 时间线：

```text
capture（CPU 逐项下发，只发生一次）
  philox_npu_state(10)：记录 dropout 的 intragraph offset
  original --Record/Wait--> secondary：副流加入 capture
  secondary: 记录 DropoutGenMaskV2Tensor(mask)
  secondary --Record/Wait--> original：副流返回主流
  original: 记录 DropoutDoMask(input, mask)

replay（不再进入 dropout C++ 函数）
  replay_prologue：刷新 device seed/base offset
  AclmdlRIExecuteAsync：
    secondary 按捕获的 base+intragraph 生成新 mask
    captured event 让 original 等 mask
    original 使用新 mask 计算输出
```

所以 replay 时并不是 CPU 再次调用 `DropoutGenMask` 后“通知”主流；生成 mask、event join、消费 mask 三者都已经是模型内部节点。每次变化的是图外 base seed/offset tensor 的内容。

还有两个实现边界不能被“dropout 已适配”一句话掩盖：第一，`IsAclnnOnly()` 的 `aclnnDropoutV3` 早期分支在固定版本仍调用 `philox_engine_inputs()`（`op-plugin@6ef73e3:op_plugin/ops/opapi/NativeDropoutKernelNpuOpApi.cpp:45-61`），并未进入上述 graph-safe 分支；第二，Tensor 路径受 `VERSION_BETWEEN(V2R5, VERSION_NEWEST)` 条件编译保护（同文件 `:91-111`）。因此必须按实际 SoC、ACLNN 版本和最终 dispatch 分支验证。

这说明“随机数算子通常只改一处 C++ capture 分支”是常见情况，但不是铁律。若算子内部又使用副流，还需额外处理：

- 首次将副流关联进图；
- 副流回主流的闭合依赖；
- 跨流 tensor 的 allocator 生命周期；
- 多次随机调用的 offset 不重叠；
- secondary stream 状态在每次新 capture 开始时重置。默认 generator 的 reset 发生在 `NPUGraph::capture_begin()`（`NPUGraph.cpp:176-181`）。

---

## 5. 算子侧与 torch_npu 侧各自要做什么

### 5.1 算子/op-plugin 实现侧

- 确认实际走 aclnn，而不是 capture 期禁止的 aclop；参见 [[npu_operator_graph_eligibility_guide]]。
- 为每个会被 dispatch 到的 overload 增加 capture 状态分支。
- eager 使用标量 seed/offset，capture 使用 Tensor seed/offset + intragraph offset。
- 按 kernel 实际消费量计算 counter increment，并保持 4 对齐语义。
- 若内部有 side stream，补齐 fork、join 和 `recordStream`。
- 对尚无 `aclnnXxxTensor` 接口的算子，不能靠缓存 host seed 绕过；应推动 ACLNN 提供 graph-safe 接口，或在捕获期明确报不支持。

### 5.2 torch_npu 框架侧

- `NPUGraph` 管理 generator 注册、capture prologue/epilogue 和 replay prologue。
- `NPUGeneratorImpl` 管理 device seed/offset tensor、图内/图外 offset 及 capture 断言。
- Stream/Event API 必须在 capture 中形成正确的跨流依赖节点。
- caching allocator 必须识别所有参与 capture 的 stream，并为跨流 tensor 记录使用流。
- 给用户暴露自定义 generator 注册接口，并保持与 PyTorch graph RNG 语义一致。

框架侧机制只提供协议，不能自动修复仍调用 `philox_engine_inputs()` 的算子。反过来，算子只加 Tensor ACLNN 调用也不够；若没有 generator replay prologue，每次 replay 仍会重复同一段 counter。

比较稳妥的落地顺序是：

1. 从公开 API 反查实际 ATen 原语和所有 op-plugin overload，不先假设 `torch.rand` 与 `uniform_` 之外没有旁路。
2. 对每个 SoC/format 确认最终走 aclnn，并确认存在接受 Tensor seed/offset 的 ACLNN 变体。
3. 在 op-plugin 中增加 capture 分支，复用公共 `NPUGeneratorImpl`，不要自行维护第二套 seed。
4. 核对 counter 消耗；若算子会分片、改变 unroll 或内部启动多个 kernel，increment 必须覆盖总消耗。
5. 若有副流，再补 event 拓扑和 allocator 生命周期；这一步与 RNG 分支是正交维度。
6. 先做逐 overload eager 回归，再做 ACLGraph 多次 replay、混合 RNG 序列和目标 SoC 测试。

---

## 6. 测试与验收矩阵

### 6.1 已有覆盖

- 多流：`test_aclgraph_multi_stream.py:16-116` 覆盖 shared-expert side stream 和双 matmul fork/join，并在修改输入后重复 replay 验证数值。
- RNG 功能序列：`test/test_npu.py:2349-2431` 用 dropout、rrelu 比较 eager 与“graph -> eager -> graph”的 RNG 状态推进，并检查换 seed 后输出变化。
- RNG 分布：`test/test_npu.py:2433-2549` 期望多种 factory/inplace distribution 在两次图内调用和多组 seed 下与 eager 对齐。

### 6.2 每个随机算子至少应补的用例

1. **capture 可用性**：目标 overload、dtype、shape、SoC 分支确实进入 Tensor ACLNN 路径。
2. **连续 replay 不冻结**：不重设 seed 时，两次 replay 的随机输出应不同。
3. **重设 seed 可复现**：相同 seed + 相同调用顺序得到相同结果。
4. **eager/graph 序列一致**：同一 seed 下，图内多个 RNG op 与 eager 的 counter 推进一致。
5. **图内不重叠**：至少两个 RNG op 连续入图，验证第二个使用第一个之后的 offset。
6. **自定义 generator**：capture 前注册；同时增加“未注册应明确失败”的负例。
7. **边界输入**：空 tensor、大 tensor、非连续 tensor、不同 dtype，以及会改变 counter 计算的分片路径。
8. **不支持路径明确失败**：不能出现 capture 成功但输出未写、每次重复或静默返回未初始化 tensor。

最小测试不能只做“capture 不报错”，而应同时检查状态推进与复现，例如：

```python
torch.npu.manual_seed(123)
g = torch.npu.NPUGraph()
with torch.npu.graph(g):
    out1 = rng_op(static_input)   # 同一图中至少放两个 RNG 消费者
    out2 = rng_op(out1)

g.replay()
first = out2.clone()
g.replay()
second = out2.clone()
assert not torch.equal(first, second)       # replay 确实推进整图 offset

torch.npu.manual_seed(123)
g.replay()
reset = out2.clone()
# 用相同 seed、相同前置 RNG 调用和相同算子顺序生成 eager_ref
assert torch.equal(reset, eager_ref)        # 图与 eager 的 counter 协议一致
```

比较 `reset` 与 `eager_ref` 时必须把 capture 前的 dummy/warmup RNG 消耗也对齐；现有 `test_graph_rng_distributions` 正是先跑 dummy，再分别构造 control 与 graph 序列（`torch_npu/test/test_npu.py:2453-2542`）。否则测试失败可能只是两边起始 offset 不同，而不是 graph-safe 实现错误。

### 6.3 多流还应补的用例

1. 无 join 的 capture-end 负例，锁定报错行为。
2. 三条及以上 stream 的直接/间接加入与逐级返回。
3. fork 前、join 后各有主流任务，验证只建立局部偏序。
4. 多次 replay 更新输入，验证依赖 DAG 与静态地址共同生效。
5. side stream 产生临时 tensor 后释放，验证 `recordStream` 防止提前复用。
6. 多个 dropout/RNG side-stream 调用交错，验证 `secondary_stream_capture_state_` 与 offset 累计。
7. ACLGraph + HCCL 的真实 collective 用例，按 CANN/HCCL/SoC 建立支持矩阵。

数值测试只能证明依赖正确，不能证明两条流真的重叠。`test_aclgraph_multi_stream.py` 的结果对齐能发现漏 wait、静态输入更新失败等问题，但即使 runtime 把两分支串行执行也会通过。若目标是验证性能收益，还要用 profiler/timeline 检查两个 stream 上的 kernel 时间区间是否交叠，并分别比较 capture 与 replay；这属于性能验收，不应混入纯正确性断言。

---

## 7. 排查速查

| 现象 | 优先检查 |
|---|---|
| capture end 报跨流关联错误 | side stream 是否通过 event 加入并在结束前返回主流；join 后是否又向 side 下发 task |
| eager 正常、capture 报 `philox_engine_inputs` | 该 overload 仍未改成 `philox_npu_state()` + Tensor ACLNN API |
| 每次 replay 输出完全相同 | seed/offset 是否被当 host 标量固化；replay prologue 是否执行 |
| 图内两个 RNG op 序列重叠 | counter increment 是否小于 kernel 实际消耗；intragraph offset 是否逐算子推进 |
| 多流结果偶发错误 | event 方向是否写反；消费者是否在 join 后；跨流 tensor 是否 `recordStream` |
| 自定义 generator 报 not in capture mode | 是否在 capture 前 `g.register_generator_state(generator)` |
| 测试名声称支持但目标环境失败 | 对齐 op-plugin gitlink、SoC 分支、ACLNN Tensor 接口版本和实际 dispatch overload |

---

## Related Pages

- [[aclgraph]] —— ACLGraph 基础用法与 capture/replay 生命周期
- [[aclgraph_deep_analysis]] —— ACLGraph、NpuGraphOpHandler、aclnn/aclop 门禁与社区差异
- [[npu_operator_graph_eligibility_guide]] —— 从 Dynamo、Inductor 到 ACLGraph 的算子入图判别
- [[npugraphs_memory_reuse_analysis]] —— Graph Tree、静态地址与 allocator 生命周期
- [[comparison]] —— CUDA Graphs 与 NPU Graphs 的接口和实现对照
- [[torch_npu_upstream_adaptation_analysis]] —— torch_npu out-of-tree 适配全景与源码基线
