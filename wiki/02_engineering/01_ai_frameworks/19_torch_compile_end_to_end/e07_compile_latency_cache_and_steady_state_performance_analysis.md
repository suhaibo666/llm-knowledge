# E07 · Compile Latency、Cache 与 Steady-State 性能

> 卷别：E · 调试、正确性与性能  
> 固定源码：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`  
> 前置：[[e06_compiled_correctness_validation_methodology_analysis]]  
> 后续：[[e08_kernel_fusion_memory_and_hardware_performance_analysis]]  
> 最后更新：2026-07-28

## 1. 为什么一个“平均耗时”没有诊断价值

`torch.compile`把成本分布到多个时点：

- wrapper创建；
-首次frame capture；
-Dynamo、AOT、Inductor编译；
-native/Triton编译与module load；
-lazy backward compile；
-runtime autotune；
-CUDAGraph warmup/record；
-稳态guard、wrapper与kernel/replay。

如果把这些轮次平均，既看不出冷启动SLO，也看不出稳态吞吐，更无法判断cache是否有效。

## 2. 五个必须分开的测量场景

### 2.1 Eager baseline

同一输入语义、相同精度和同步边界下的eager延迟/吞吐。

### 2.2 Cold compile

干净进程、目标cache层按实验定义为空、设备context状态明确。测量从首次调用到结果可用。

### 2.3 Warm process / cold specialization

进程与toolchain已初始化，但输入触发新的Dynamo specialization。用于观察真实shape流量的
增量编译成本。

### 2.4 Disk/remote cache hit

新进程复用持久化artifact。仍包含key/guard、deserialize、module load、post-compile和可能
的first-call runtime工作。

### 2.5 Steady state

guards命中、所有lazy工作完成、无意外recompile。按稳定窗口报告分位数和吞吐。

## 3. 时间分解

\[
T_{\text{cold}} =
T_{\text{wrapper}}
+T_{\text{Dynamo}}
+T_{\text{AOT}}
+T_{\text{Inductor}}
+T_{\text{native}}
+T_{\text{load}}
+T_{\text{first-runtime}}
\]

\[
T_{\text{steady}} =
T_{\text{guard}}
+T_{\text{wrapper-runtime}}
+T_{\text{kernel/replay}}
+T_{\text{sync-visible}}
\]

训练还需：

\[
T_{\text{first-step}} =
T_{\text{fw-compile/run}}
+T_{\text{lazy-bw-compile/run}}
+T_{\text{optimizer}}
\]

若只计forward，会把lazy backward compile移到“第二阶段”，形成错误的冷启动结论。

## 4. 源码提供哪些阶段指标

`CompilationMetrics`同时记录frame、backend、Inductor、codegen时间，以及runtime
CUDAGraph、Triton autotune和backward累计编译时间
（`torch/_dynamo/utils.py:1581-1605`、
`torch/_dynamo/utils.py:1613-1635`）。

记录函数把同一事件标记为forward/backward/runtime，并用 `compile_id` 关联结构化日志
（`torch/_dynamo/utils.py:2013-2037`）。

这些字段适合阶段归因，但端到端用户延迟仍应在调用边界测量，因为：

-Python wrapper与排队可能不全在单个timer内；
-设备执行异步；
-多线程/多进程等待会跨context；
-日志本身有扰动。

## 5. Cache 实验必须定义清理范围

不同“清cache”可能只清：

-Dynamo code entries；
-进程内GraphModule/module/future；
-AOTAutograd cache；
-FX graph local/remote cache；
-generated source/native binary；
-Triton/autotune cache；
-CUDAGraph runtime tree；
-allocator/device context。

因此报告不得只写“清了cache”，而要列出清理层、进程是否重启、disk/remote是否保留。

Inductor会把部分metrics delta写入FX cache并在hit时恢复；这些字段包括kernel count、
fusion前IR node和估算bytes
（`torch/_inductor/metrics.py:101-124` 与
`torch/_inductor/metrics.py:127-140`）。所以metric恢复不代表本进程重新生成了这些kernel。

## 6. 正确计时异步设备

设备kernel launch通常异步。计时应：

-在窗口前排空先前工作；
-在窗口后等待结果真正完成；
-不把每个kernel都强制同步而改变执行形态；
-区分host latency与device elapsed time；
-固定stream和并发负载策略；
-预热allocator、library handle和频率状态。

同步的位置是测量契约的一部分。只围住Python函数而不等待device，得到的可能只是launch
开销。

## 7. Break-even 分析

设eager每次成本 \(E\)，compiled稳态成本 \(C\)，一次性冷启动成本 \(K\)，要求 \(E>C\)：

\[
N_{\text{break-even}} =
\left\lceil\frac{K}{E-C}\right\rceil
\]

若有多个specialization \(s\)：

\[
K_{\text{total}}=\sum_s K_s,\qquad
N_s \text{ 必须足以摊销 }K_s
\]

生产价值应按真实shape频率加权。一个只出现一次的长尾shape即使稳态很快，也可能永远无法
摊销编译成本。

## 8. 推荐结果表

| 场景 | 进程/cache | 调用 | 应报告 |
|---|---|---|---|
| eager | 新进程 | 多轮 | p50/p95/吞吐/显存 |
| cold compile | 全冷 | 首个完整step | 总时延与各阶段 |
| new specialization | warm进程 | 首次新shape | recompile原因与增量成本 |
| persistent hit | 新进程/保留artifact | 首次调用 | key/load/post-compile/first runtime |
| steady | warm且guard hit | 稳定窗口 | p50/p95/p99/吞吐 |

训练额外报告forward、backward、optimizer和端到端step；分布式额外报告最慢rank与同步等待。

## 9. 统计设计

-编译延迟通常长尾，应报告分位数和最大值；
-冷启动用多个独立进程样本，而不是同进程重复；
-稳态剔除轮次必须基于状态证据，不是固定“前5次”；
-同时记录compile id/recompile count；
-吞吐测试固定并发、batch和队列；
-性能比较先通过正确性；
-版本、配置、频率和后台负载固定。

## 10. 成本模型

对 \(N\) 次调用、\(S\) 个specialization：

\[
T_{\text{compiled,total}}
=\sum_{s=1}^{S}K_s+\sum_{i=1}^{N}C_i
\]

空间成本：

\[
M_{\text{total}}
=M_{\text{code cache}}
+M_{\text{runtime modules}}
+M_{\text{allocator}}
+M_{\text{CUDAGraph pools}}
\]

速度收益不能脱离cache容量、编译CPU、disk IO和device reserved memory评估。

## 11. 常见误解

- **“第一次调用就是compile time。”** 还可能包含load、autotune、record和实际执行。
- **“第二次调用就是steady state。”** lazy backward或CUDAGraph可能尚未完成。
- **“disk hit等于0编译开销。”** 仍有load/post-compile/first-runtime。
- **“kernel time下降就代表端到端加速。”** guard、wrapper、通信和队列可能主导。
- **“平均值足够。”** 编译与线上延迟常是长尾分布。

## Related Pages

- [[00_torch_compile_end_to_end_index]]
- [[a05_eager_capture_compile_and_replay_cost_model_analysis]]
- [[d04_compile_cache_hierarchy_keys_and_invalidation_analysis]]
- [[d07_compiled_artifact_lifecycle_and_runtime_failures_analysis]]
- [[e06_compiled_correctness_validation_methodology_analysis]]
- [[e08_kernel_fusion_memory_and_hardware_performance_analysis]]
- [[e09_production_rollout_fallback_and_monitoring_analysis]]
