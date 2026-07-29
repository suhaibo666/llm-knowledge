# 06 · NPU Graphs / ACLGraph — 目录索引

> NPU Graphs 的编译集成、内存管理与重用、make_graphed_callables 适配、ACLGraph 深度分析、多流与随机状态捕获，以及与 CUDA Graphs 的对比。
> 知识分层:overview→quick start→deep dive(约定见 [[01_ai_frameworks/index]])。冗余文档已合并(内存两篇、torch.compile 两篇),信息无损。
> 核验基准:torch_npu **v2.7.1.post5**
> 最后更新: 2026-07-29

---

## 页面列表(按层次)

| 页面 | 层次 | 核心主题 |
|------|------|---------|
| [[aclgraph]] | **overview** | ACL Graph(昇腾计算语言)集成基础:是什么、调用流程、使用限制与注意事项、与 CUDA Graph 关系(已并入原 README) |
| [[comparison]] | **overview** | CUDA Graphs vs NPU Graphs 特性对比(API/实现/捕获时序) |
| [[npugraphs_make_graphed_callables_deep_dive]] | **quick start** | make_graphed_callables API:六阶段实现流程、内存峰值 debug 方法 |
| [[aclgraph_deep_analysis]] | deep dive | ACLGraph 深度:图捕获/重放、Super Kernel、NpuGraphOpHandler、aclop/aclnn 捕获门禁、与社区差异及演进 |
| [[aclgraph_multistream_rng_analysis]] | deep dive | ACLGraph 状态化捕获:Event fork/join、多流/通信流边界、graph-safe Philox RNG、dropout 联合路径与算子测试矩阵 |
| [[torch_compile_npugraphs_deep_dive]] | deep dive | NPU Graphs × torch.compile:Path B(backend=npugraphs)完整链路;**附录 A:Path A(mode=reduce-overhead)完整流程与双路径对比**(已并入原 reduce_overhead_vs_backend) |
| [[npugraphs_memory_reuse_analysis]] | deep dive | 内存重用 + 管理:内存池/Graph Tree、Capture-Replay、Liveness、路径切换案例;**关键代码解析**(TreeManager 生命周期、Checkpoint 数据结构、warmup/静态输入,已并入原 memory_management) |

> 代码示例:`npugraphs_usage_guide.py`;原 README 速览已并入 [[aclgraph]](使用限制)与 [[comparison]](API 对应关系),信息无损。

---

## 关联域

- [[06_graphs/cuda/index]] — CUDA Graphs(对比基准)
- [[04_inductor/npu/index]] — NPU Inductor(ACLGraph 为其路径之一)
- [[07_op_registration/npu/index]] — 入图判别第三关(aclgraph aclnn-only)
- [[01_ai_frameworks/index]] — 本域总索引
