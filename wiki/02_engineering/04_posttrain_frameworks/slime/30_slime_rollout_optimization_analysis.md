# slime Rollout 优化：先找关键路径，再调吞吐

> **文档类型**：slime 段 3 横切专题 · 性能诊断与容量规划
> **源码基线**：THUDM/slime `main@681b3adca54105d5ecd3fb822fa0dc58a427e0f9`
> **基线提交时间**：2026-08-12T16:50:12+07:00
> **核验日期**：2026-08-18
> **系列入口**：[[02_engineering/04_posttrain_frameworks/slime/index|slime]]
> **证据边界**：机制结论来自上述 slime 固定提交；PD、投机解码与低精度的适用描述来自同提交项目文档。SGLang 内部 scheduler/kernel 不在本页源码基线内，因此不推断其具体算法。
> **结论先行**：Rollout 优化不是寻找一个“最快开关”，而是先找出训练闭环的瓶颈服务与更新关键路径，再只优化该项。SGLang decode tok/s 只是局部服务率；过滤/abort 的无效工作、队列与长尾、trainer 消费速度、offload 和权重发布、策略新鲜度都可能让局部加速无法转化为单位 wall time 的有效梯度。

---

## 1. 问题背景：为什么 aggregate tok/s 会把优化方向带偏

slime 的一条训练数据不是“生成完 token 就结束”：请求还可能经过工具或 reward/verifier，按 group 被动态过滤，转换和调度后才被 trainer 消费；训练结束又要释放/恢复显存并发布新权重。同步入口把 generate、rollout offload、train、train offload、weight update 和 KV onload 串成明确的阶段序列。[`train.py:48-88`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/train.py#L48-L88)

```mermaid
flowchart LR
    A["prompt 与 partial buffer"] --> B["admission 与排队"]
    B --> C["SGLang prefill decode"]
    C --> D["tool reward verifier"]
    D --> E{"是否接收"}
    E -->|接收| F["训练批"]
    E -->|过滤后补采| B
    F --> G["Megatron 消费"]
    G --> H["offload 与权重提交"]
    H --> B
```

只看生成 token/s 至少会漏掉五件事：

1. **分子不对**：被 filter、abort 或超额生成的 token 增加了 serving tok/s，却没有进入 loss；
2. **统计单位不对**：RL 的约束常在完整 prompt group、rollout 或可训练 token，而非原始 completion token；
3. **分母不对**：trainer 等待、数据搬运、offload、权重发布都属于训练闭环 wall time；
4. **分布不对**：均值吞吐不显示 p95/p99 长尾与 batch barrier 浪费；
5. **目标不完整**：更老的 behavior policy、改变后的筛选分布或低精度误差可能换来更高吞吐，却降低单位样本价值。

因此至少同时观察 accepted groups/s、进入 loss 的 token/s、attempted/accepted 比、生成与 reward 延迟分位数、queue age、trainer wait、权重版本年龄和端到端 step time。动态 filter 已按 reason 计数 drop；SGLang metadata trace 还保留 queue time、端到端 latency、decode throughput 以及 PD 子阶段时长，可作为诊断入口。[`filter_hub/base_types.py:40-52`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/rollout/filter_hub/base_types.py#L40-L52) [`trace_utils.py:16-44`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/trace_utils.py#L16-L44)

## 2. 容量账本：先判定“谁供不上”

### 2.1 请求服务率与 admission

把 group 作为调度单位。对 group $j$：

- $Q_j$：进入生成临界区之前的等待时间；
- $S_{\mathrm{gen},j}$：prefill/decode 的主动服务时间；
- $S_{\mathrm{post},j}$：离开生成临界区后的 hook、reward/verifier 主动服务时间；tool 若在 custom generate 内执行，则计入 $S_{\mathrm{gen},j}$；
- $a_j\in\{0,1\}$：该 group 最终是否被接收。

若 admission 速率为 $\lambda_{\mathrm{in}}$，各服务的可持续 group 处理率为 $\mu_{\mathrm{gen}}$、$\mu_{\mathrm{post}}$，稳定运行首先要求：

$$
\lambda_{\mathrm{in}}
<
\min\left(\mu_{\mathrm{gen}},\mu_{\mathrm{post}}\right).
$$

这不是源码中的 scheduler 公式，而是**容量分析模型**。它提醒我们：提高生成并发只会提高 $\lambda_{\mathrm{in}}$ 或尝试填满 $\mu_{\mathrm{gen}}$；若 reward/tool 已经更慢，结果只会把等待从 SGLang 前移到下游。

固定实现正好体现了这条边界：semaphore 只包住 generate/custom-generate 段，sample hook 与 reward 位于 semaphore 之外。[`sglang_rollout.py:224-287`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/rollout/sglang_rollout.py#L224-L287) 因而 `sglang_server_concurrency` 是生成侧 admission cap，不是整个 rollout pipeline 的端到端并发上限；自定义 tool loop 是否占用该 semaphore，取决于它是否在 custom generate 内部完成。

### 2.2 accepted batch 的准备时间与长尾税

设目标 batch 需要 $B$ 个 accepted groups，$t_B$ 是第 $B$ 个有效 group 完成筛选的时刻，$T_{\mathrm{drain}}$ 是随后 abort、等待 pending task 收敛和回收 partial 的时间，则：

$$
T_{\mathrm{rollout}}
=
t_B+T_{\mathrm{drain}}.
$$

同步循环不是等待最慢候选自然完成：它用 `FIRST_COMPLETED` 逐批消费结果，凑满 $B$ 后 abort 剩余请求；但 abort 仍会等待 pending tasks 返回，决定是否回收 partial group。[`sglang_rollout.py:407-451`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/rollout/sglang_rollout.py#L407-L451) [`sglang_rollout.py:339-371`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/rollout/sglang_rollout.py#L339-L371)

因此长尾成本不只是最慢 response latency，还包括：

- 被慢 group 占住的 in-flight capacity；
- 已生成但最终未接收的 token、tool 与 RM 成本；
- batch 满后 abort/drain 的 barrier；
- partial 恢复时保留旧上下文或跨版本 token 的成本。

### 2.3 训练闭环关键路径

对同步入口，用下面的分解比“rollout tok/s”更接近真实目标：

$$
\begin{aligned}
T_{\mathrm{step}}^{\mathrm{sync}}
&=T_{\mathrm{rollout}}
 +T_{\mathrm{data}}
 +T_{\mathrm{train}} \\
&\quad
 +T_{\mathrm{offload}}
 +T_{\mathrm{publish}}.
\end{aligned}
$$

这里 $T_{\mathrm{data}}$ 包含 Sample 到 trainer 的转换与搬运；$T_{\mathrm{offload}}$ 包含 colocate 的模型/KV 时分复用；$T_{\mathrm{publish}}$ 是更新推理副本所需的提交阶段。它们的具体协议分别由 [[12_slime_sample_datasource_analysis|Sample/DataSource]]、[[14_slime_megatron_training_analysis|Megatron 训练]]和 [[16_slime_weight_sync_analysis|权重同步]]解释，本页只把它们放回同一性能账本。

再定义 $T_{\mathrm{wait,train}}$ 为“trainer 已可运行、但下一批 accepted data 尚未就绪”的空闲区间。同步入口中这段等待被阶段串行结构吸收，近似落在 $T_{\mathrm{rollout}}+T_{\mathrm{data}}$；warm queue 与 phase overlap 的直接目标才是压缩 $T_{\mathrm{wait,train}}$，而不是凭空降低请求服务时间。

若 generation N+1 与 training N 使用独立资源重叠，理想下界才接近：

$$
T_{\mathrm{cycle}}^{\mathrm{overlap}}
\gtrsim
\max\left(T_{\mathrm{rollout}}+T_{\mathrm{data}},T_{\mathrm{train}}\right)
+T_{\mathrm{fence}}
+T_{\mathrm{publish}}.
$$

这是**分析下界**而非源码计时公式。producer arm 必须包含 $T_{\mathrm{data}}$：`RolloutManager.generate()` 在取得 rollout data 后，还会完成 Sample → train dict 转换和 DP split 才返回 future；`train_async.py` 又在开始当前轮训练前等待该 future。[`rollout.py:590-604`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/ray/rollout.py#L590-L604) [`train_async.py:31-53`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/train_async.py#L31-L53)

式中的 $T_{\mathrm{fence}}$ 也不能假设为零：`train_async.py` 在发布新权重前仍等待下一轮 generation future 完成，避免生成中途换权重；该入口还直接禁止 colocate。[`train_async.py:66-70`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/train_async.py#L66-L70) [`train_async.py:9-12`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/train_async.py#L9-L12)

> **设计分析**：overlap 消掉的是“两阶段串行空洞”，不是两个阶段本身的工作量。若 rollout 与转换之和比 training 慢很多，trainer 仍等待；若 training 更慢，warm queue 只会积累更老的样本。若两边争用网络、CPU、存储或功耗预算，重叠甚至可能让 $\max(T_{\mathrm{rollout}}+T_{\mathrm{data}},T_{\mathrm{train}})$ 变大。

## 3. 服务侧旋钮：只在生成服务真是瓶颈时使用

### 3.1 请求并发：修复欠载，不是越大越好

`GenerateState` 把 semaphore 容量设为“每 engine 并发 × engine 数”，group 作为 task 提交，而同一 group 的 samples 再并发生成。[`sglang_rollout.py:83-149`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/rollout/sglang_rollout.py#L83-L149) 它适合修复低并发导致的 GPU bubble，前提是 KV cache、HTTP、tool/RM 和 host 线程仍有余量。

反例不是假设：项目 Qwen3 示例明确警告，单 server 并发超过默认 CUDA graph 并发 160 会影响推理速度，并建议限制 `sglang_server_concurrency` 或扩充 graph batch size。[`docs/zh/examples/qwen3-4B.md:281-290`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/docs/zh/examples/qwen3-4B.md#L281-L290)

### 3.2 本地 DP 计数不等于生效的 router load balance

源码里的 `dp_rank_context()` 会选择当前计数最小的 rank 并维护 in-flight count。[`sglang_rollout.py:113-129`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/rollout/sglang_rollout.py#L113-L129) 但在固定基线的默认路径中，yield 出来的 rank 被写成 `as _`，HTTP payload 仍只发到 router `/generate`，没有携带该 rank；唯一显式的 worker-affinity 信息是 consistent-hashing 时的 session header。[`sglang_rollout.py:245-262`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/rollout/sglang_rollout.py#L245-L262) [`sglang_rollout.py:152-203`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/rollout/sglang_rollout.py#L152-L203)

> [!warning] 源码事实与常见表述的边界
> 不能据此把“最少在途 rank”写成默认请求的实际负载均衡算法。slime 在这里负责 admission 与可选 session affinity；router 如何在 worker 间选择请求属于 SGLang/Model Gateway 的实现边界，需另钉 SGLang 基线后分析。

这也给出调参顺序：先看各 worker 的 queue/KV/latency 是否失衡；若是 session affinity 导致热点，调整 router policy 或 key；若所有 worker 同时欠载，再增加 slime admission。

### 3.3 PD、投机/MTP 与低精度分别消除不同资源瓶颈

| 方案 | 它真正减少的瓶颈 | 使用前提 | 新成本与反例 |
|---|---|---|---|
| PD disaggregation | prefill 与 decode 资源比例不匹配、长上下文/多轮的阶段干扰 | 两阶段确实需要不同 TP、显存或 runtime；有可观测的 PD 分段时间 | 增加 router、KV transfer 与拓扑运维；短单轮任务可能被传输/调度开销反噬 |
| speculative decode | target model 的逐 token decode 次数 | draft acceptance 足够高，验证 kernel 与 batch 形态合适 | draft 与验证也耗时；policy 漂移导致 acceptance 降低时可出现负收益 |
| 在线 MTP | RL 中 draft/target 漂移造成的 speculative 退化 | checkpoint、模型映射、训练与发布路径都支持 MTP | 多一项训练 loss 与同步状态；若 draft 更新没跟上 actor，局部功能开启也不能保证收益 |
| rollout 低精度 | 权重/KV 的显存容量或带宽压力 | 硬件、kernel、量化配置与精度门禁已验证 | 量化/转换/一致性成本；原本 compute-bound 或不兼容 kernel 时未必加速 |

项目文档把 PD 的目标定位在 long-context、multi-turn、decode long tail 及异构 prefill/decode 配置，并明确短单轮任务用 regular layout 更简单。[`docs/zh/advanced/pd-disaggregation.md:3-15`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/docs/zh/advanced/pd-disaggregation.md#L3-L15) 它还把 prefill/decode transfer 与 queue 暴露成独立 trace 段，因而应先用这些阶段指标证明 PD 值得启用，而不是凭 workload 名称猜测。[`trace_utils.py:26-44`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/trace_utils.py#L26-L44)

投机解码的项目文档明确指出：RL 使 draft/target 分布逐渐漂移时，通过验证的 draft token 减少，spec 甚至可能负收益；在线训练 MTP 是为缓解这个问题，外部 draft model 训练在该基线仍是 WIP。[`docs/zh/advanced/speculative-decoding.md:24-38`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/docs/zh/advanced/speculative-decoding.md#L24-L38) 其同步接缝与基线缺口由 [[21_slime_speculative_decoding_mtp_analysis|Speculative/MTP 专题]]负责。

低精度文档把 BF16 training + FP8 rollout 列为推荐路径，并说明 FP8 KV cache 只改变 rollout 侧容量，实际精度/性能依赖 SGLang 版本和 GPU stack。[`docs/zh/advanced/low-precision.md:3-16`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/docs/zh/advanced/low-precision.md#L3-L16) [`docs/zh/advanced/low-precision.md:44-52`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/docs/zh/advanced/low-precision.md#L44-L52) 训练、rollout、KV 和通信精度不能合并成一个开关，详见 [[22_slime_low_precision_training_rollout_analysis|低精度专题]]。

## 4. 调度侧旋钮：目标是降低 underfill 和长尾浪费

### 4.1 first-completed、oversampling 与 dynamic filter

默认循环按 `FIRST_COMPLETED` 消费 group；若过滤后 remaining candidates 少于目标，就一次补 `over_sampling_batch_size` 个 group。[`sglang_rollout.py:400-436`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/rollout/sglang_rollout.py#L400-L436) 这一组合解决的是“有效 batch 被 filter 打空”和“等待整批最慢任务”，不是减少单个 group 的服务时间。

令尝试 group 数为 $N_{\mathrm{attempt}}$，接收数为 $N_{\mathrm{accept}}$，进入 loss 的有效 token 数为 $N_{\mathrm{loss}}$。更有意义的两个效率量是：

$$
\eta_{\mathrm{accept}}
=
\frac{N_{\mathrm{accept}}}{N_{\mathrm{attempt}}},
\qquad
R_{\mathrm{productive}}
=
\frac{N_{\mathrm{loss}}}{T_{\mathrm{wall}}}.
$$

若 oversampling 提升 aggregate tok/s，却让 $\eta_{\mathrm{accept}}$ 下降且 $R_{\mathrm{productive}}$ 不升，它只是在更快地产生废弃工作。固定实现还明确注明：凑满 batch 后，并没有把所有已完成但未使用的 samples 放回 buffer。[`sglang_rollout.py:438-451`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/rollout/sglang_rollout.py#L438-L451)

> **设计分析**：first-completed 本身只改变结果消费顺序；当 batch 以“最早完成且通过 filter”为准截断时，latency 与题目难度、工具轮数或 reward 相关，就可能产生 latency selection bias。严格 filter 又会改变训练分布。两者都应把 drop reason、长度、reward、source 与 latency 做联合分桶，而不是只报保留率。

fallback filter 在候选不足时保留本来被拒绝的 group，以免再启一轮 oversampling；这是用数据准则换等待时间，而非“等价加速”。源码把该语义编码为 `keep_when_insufficient`。[`filter_hub/base_types.py:5-24`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/rollout/filter_hub/base_types.py#L5-L24)

### 4.2 partial：回收已做的工作，不会消灭版本代价

partial 在 batch 已满后回收 unfinished group，下轮只生成剩余 token budget；可选开关把旧 response mask 清零，只训练新 on-policy token。[`sglang_rollout.py:339-371`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/rollout/sglang_rollout.py#L339-L371) [`arguments.py:456-474`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/arguments.py#L456-L474)

它适合 response 很长、abort 已生成前缀成本高的场景。反例是短 response 或权重频繁发布：abort/序列化/恢复开销可能大于省下的 decode；开启 mask 后旧 token 仍占 context/KV 却不贡献梯度，不开启则要接受跨版本 behavior。token、metadata 与 buffer 的完整语义由 [[12_slime_sample_datasource_analysis|Sample/DataSource 专题]]说明。

### 4.3 fully async：消除 rollout batch boundary，不等于 train/generate overlap

`fully_async_rollout.py` 用跨调用存活的后台 worker 保持固定 in-flight pool；完成 queue 达到一个 concurrency pool 时停止补新任务，每次 trainer 只取当前 batch 所需数量，surplus 留给下一轮。[`fully_async_rollout.py:48-90`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/rollout/fully_async_rollout.py#L48-L90) [`fully_async_rollout.py:131-171`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/rollout/fully_async_rollout.py#L131-L171) [`fully_async_rollout.py:211-266`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/rollout/fully_async_rollout.py#L211-L266)

它消除的是“每轮重新灌满并发池”和“为了固定 batch 等当前最慢 in-flight group”。它没有定义最大 weight-version age；worker 明确不拥有高层 weight-update signaling，只把 ABORTED group 放回 buffer。[`fully_async_rollout.py:13-23`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/rollout/fully_async_rollout.py#L13-L23) evaluation 也在入口处被拒绝。[`fully_async_rollout.py:269-274`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/rollout/fully_async_rollout.py#L269-L274)

> **设计分析**：当 trainer 比 producer 慢时，warm queue 把“trainer 等数据”转化成“数据等 trainer”。吞吐可能不变，queue age 与 policy staleness 却上升。因此 fully async 的验收指标必须包含 queue depth/age 与版本分布，而不是只看 trainer idle。

## 5. 消费与提交侧：rollout 快了以后，瓶颈会移到哪里

### 5.1 trainer schedule 与数据 transport

rollout batch 还要被 packing、对齐并分给 DP ranks。固定 scheduler 先按 static/dynamic 规则 pack，再把 micro-batch 数对齐到 `dp_size × VPP group`，最后按估计 workload 或 round-robin 分 rank。[`dp_schedule.py:156-207`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/dp_schedule.py#L156-L207)

- trainer 因变长样本 padding/不均衡而慢：再调 dynamic batch 或 `balance_data`；
- rollout manager 到 trainer 的大 tensor 搬运慢：比较 object store 与 NIXL；该开关只替换 CPU tensor transport。[`arguments.py:557-566`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/arguments.py#L557-L566)
- 不要把 FLOPs balance 当无风险加速：参数说明明确警告它可产生超过 `max_tokens_per_gpu` 的 micro-batch 并 OOM。[`arguments.py:730-743`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/arguments.py#L730-L743)

因此 producer 增发 token 前，应先确认 trainer 能否按相同 wall time 消费它们；否则只会把队列、CPU 内存和版本年龄推高。loss 的统计单位与并行不变量归 [[15_slime_loss_parallelism_analysis|loss/并行专题]]所有。

### 5.2 offload 与权重发布

同步入口在 generate 后 offload rollout、训练后 offload/clear trainer、再 update weights 并恢复 rollout KV，说明 colocate 本质是显存的时分复用，不是 generation/training 并行。[`train.py:53-88`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/train.py#L53-L88) 若 $T_{\mathrm{offload}}+T_{\mathrm{publish}}$ 已占主导，继续优化 decode 不会显著降低 $T_{\mathrm{step}}$。

`update_weights_interval` 可以降低发布频率，chunk buffer 可限制 MoE 更新块大小；源码参数只定义频率与 buffer，并不承诺这是免费性能。[`arguments.py:526-540`](https://github.com/THUDM/slime/blob/681b3adca54105d5ecd3fb822fa0dc58a427e0f9/slime/utils/arguments.py#L526-L540)

> **设计分析**：加大发布间隔直接把同步成本换成 behavior policy age；更大的 update buffer 也可能用峰值内存换调用次数。先测 pause/flush、转换、传输、load 与 resume 各段，再选择 transport 或 interval。完整提交协议、版本不变量与失败边界见 [[16_slime_weight_sync_analysis|权重同步专题]]。

## 6. 从症状到动作：决策矩阵

| 观测症状 | 首先证明的瓶颈 | 候选动作 | 前提与反例 |
|---|---|---|---|
| SGLang GPU 欠载且 queue 很短 | admission 不足 | 增大请求并发 | RM/tool 有余量；过高并发可越过 graph/KV 舒适区而降速 |
| worker queue/KV 极不均 | router policy 或 session 热点 | 调整路由/affinity，再看 admission | slime 本地 `dp_counts` 不是默认 worker 选择证据 |
| prefill 与 decode 一边饱和一边空 | 阶段资源比例错误 | PD/异构 server group | 短任务或 KV transfer 主导时 regular 更快 |
| target decode 调用主导 | 单 token target 成本 | spec + 可同步的 MTP/draft | acceptance 低时 draft/verify 变成纯开销 |
| KV OOM 或并发受容量限制 | KV/权重显存 | FP8 KV、rollout 量化、offload | 必须做质量与一致性门禁；compute-bound 时未必提速 |
| 有效 batch 常被 filter 打空 | acceptance 率低 | 调 oversampling 粒度或 fallback | 会增加废弃成本或改变筛选准则 |
| batch 满后仍有大量长任务 | tail/abort 浪费 | partial 或 fully async | partial 有跨版本代价；warm queue 可能增加 freshness age |
| trainer 经常等 rollout | 跨阶段串行空洞 | disaggregate + `train_async.py` | colocate 不支持；共享基础设施争用可能抵消 overlap |
| rollout queue 越积越深 | trainer/transport 才是瓶颈 | dynamic packing、DP balance、NIXL | FLOPs packing 可能 OOM；继续加生成并发只会增龄 |
| 每轮 publish/offload 占比高 | 提交/显存切换 | 优化 transport/buffer，谨慎增 interval | interval 用 freshness 换时间，不能只验吞吐 |

[[25_vime_vllm_backend_support_analysis|vime/vLLM 衍生实现]]改写了 rollout engine、请求和同步契约；同名优化旋钮不能直接沿用本页的 upstream slime 假设，应先做 backend contract 对照。

## 7. 最小可归因实验

### 7.1 先建 baseline

固定 checkpoint、prompt 集、sampling 参数、reward/tool 服务版本和训练 batch 语义，记录：

- $T_{\mathrm{step}}$ 及 generate、postprocess、train、offload、publish 分段；
- attempted/accepted groups，$N_{\mathrm{loss}}$ 与 $\eta_{\mathrm{accept}}$；
- queue/e2e latency 的 p50、p95、p99，以及 abort/drain 时间；
- rollout queue depth/age、weight-version age；
- 各 SGLang worker 与 trainer rank 的 GPU、KV、网络和 host 利用率；
- filter drop reason、长度、reward、source 与 latency 的联合分布。

trace、健康检查和 debug replay 的具体能力与空白由 [[18_slime_fault_tolerance_observability_analysis|容错与观测专题]]维护。

### 7.2 每次只改变一个容量假设

1. **并发扫描**：逐级增加 admission，直到 productive throughput 不再上升或 p99/KV/RM 饱和；
2. **long-tail 消融**：分别比较同步、partial、fully async，报告废弃 token、queue age 与版本分布；
3. **服务优化消融**：PD、spec/MTP、低精度分别单独开关，不能一次全开后只报总 tok/s；
4. **consumer 消融**：固定 rollout data，比较 packing、DP balance 和 transport；
5. **overlap 消融**：比较串行和 `train_async.py`，同时检查阶段是否因资源争用各自变慢；
6. **发布消融**：拆分 offload、convert、transfer、load、resume；改变 interval 时同步报告 freshness 和质量。

### 7.3 通过标准

一次优化只有同时满足以下条件才算端到端收益：

$$
R_{\mathrm{productive}}\uparrow,
\qquad
T_{\mathrm{step}}\downarrow,
\qquad
\text{quality/freshness invariants unchanged}.
$$

最后一项不是数学等式，而是验收约束：reward/长度/source 分布、policy age、重要性比率、精度一致性和训练曲线不能因为“更快”而悄悄换了实验。稳定性控制回路与症状归因见 [[31_slime_posttraining_stability_analysis|后训练稳定性专题]]。

## 8. 常见误读

| 误读 | 固定基线下的实际边界 |
|---|---|
| SGLang tok/s 上升就等于训练更快 | 还要经过 filter、trainer、offload 与 publish；应看 productive throughput 和 step time |
| `sglang_server_concurrency` 是全 pipeline 并发 | semaphore 只覆盖 generation，reward/tool 可能成为下游瓶颈 |
| `dp_rank_context` 已实现默认 worker 负载均衡 | rank 在默认 HTTP 路径未进入 payload；实际 worker 选择交给 router |
| oversampling 只是补齐 batch，没有统计代价 | 它增加 attempted work，且完成速度与 filter 共同决定被接收分布 |
| partial 会无损消除长尾 | 它在旧 token 梯度、context 开销与 policy freshness 之间取舍 |
| fully async 等于 generation/training overlap | 前者保持跨 batch 的 rollout queue；后者由独立入口并行两个阶段 |
| PD、spec、FP8 都是通用加速 | 三者分别针对阶段失衡、target decode、显存/带宽；前提不成立会负收益 |
| 增大发布间隔只是降低通信 | 它也增加 behavior policy age，必须与质量和一致性一起验收 |

## Related Pages

- [[13_slime_sglang_rollout_engine_analysis]] — 请求状态、router/server 分层、partial 与 streaming 的机制权威页。
- [[12_slime_sample_datasource_analysis]] — accepted sample、buffer 回收、partial token 与训练 dict 的语义边界。
- [[16_slime_weight_sync_analysis]] — 把 pause、flush、版本和 transport 解释为权重提交协议。
- [[21_slime_speculative_decoding_mtp_analysis]] — 在线 MTP 为何必须跟随 actor 版本，以及固定基线的同步接缝。
- [[22_slime_low_precision_training_rollout_analysis]] — 训练、权重、通信、rollout 与 KV 精度轴的独立风险。
- [[25_vime_vllm_backend_support_analysis]] — 更换 vLLM backend 后，容量与关键路径假设如何变化。
- [[31_slime_posttraining_stability_analysis]] — 性能优化改变数据分布、策略新鲜度或数值语义时如何定位。
