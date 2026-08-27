---
title: "06 · NPU Graphs / ACLGraph — 目录索引"
---

# 06 · NPU Graphs / ACLGraph — 目录索引

> NPU Graphs 的编译集成、内存管理与重用、make_graphed_callables 适配、ACLGraph 深度分析、多流与随机状态捕获，以及与 CUDA Graphs 的对比。
> 知识分层:overview→quick start→deep dive(约定见 [[01_pytorch/index]])。冗余文档已合并(内存两篇、torch.compile 两篇、Graph Tree 双写两篇),信息无损。
> 核验基准:torch_npu **v2.7.1.post5**
> 最后更新: 2026-07-30

---

## 页面列表(按层次)

> **段位与阅读顺序**(kb-reorg P4 Task 9.5,2026-07-30):段 0(01-09)入门;段 1(10-19)核心机制——两条并行的核心捕获路径:reduce-overhead/cudagraphs backend 的 ACLGraph 捕获(10)、backend="npugraphs" 的 NPU Graph Tree 完整链路(11);段 2(20-29)深潜/专题——`make_graphed_callables`(内容为六阶段实现流程与内存峰值 debug,原按体裁标"quick start"但实为窄 API 的实现级深挖,按内容实质改判段 2)、多流/RNG 状态化捕获专题;段 3(30-39)对照——CUDA vs NPU 差异表收尾。

| 页面 | 层次 | 核心主题 |
|------|------|---------|
| [[01_aclgraph]] | **overview**(段 0) | ACL Graph(昇腾计算语言)集成基础:是什么、调用流程、使用限制与注意事项、与 CUDA Graph 关系(已并入原 README) |
| [[10_aclgraph_deep_analysis]] | deep dive(段 1) | ACLGraph 深度:图捕获/重放、Super Kernel、NpuGraphOpHandler、aclop/aclnn 捕获门禁、与社区差异及演进;**mode="reduce-overhead" 捕获路径权威页**(§1.5 mode 参数与两条路径触发关系、§4.4 与 backend="npugraphs" 路径对比,已并入原 reduce_overhead_vs_backend 的独有内容) |
| [[11_torch_compile_npugraphs_deepdive]] | deep dive(段 1) | NPU Graphs × torch.compile:Path B(backend=npugraphs)完整链路;§三 NPU Graph Tree 核心机制(状态机、内存池共享、TreeManagerContainer 生命周期、C++ 层数据结构、Checkpoint 恢复与内存复用三分类、Liveness/弱引用/别名检测、graph-break 案例,已并入原 memory_reuse);§四/附录 A 已收缩为结论表+摘要,详情见 [[20_npugraphs_make_graphed_callables_deepdive]] 与 [[10_aclgraph_deep_analysis]] |
| [[20_npugraphs_make_graphed_callables_deepdive]] | deep dive(专题,段 2) | make_graphed_callables API:六阶段实现流程、内存峰值 debug 方法(原体裁标 quick start,按内容实质改判段 2,见上方段位说明) |
| [[21_aclgraph_multistream_rng_analysis]] | deep dive(专题,段 2) | ACLGraph 状态化捕获:Event fork/join、多流/通信流边界、graph-safe Philox RNG、dropout 联合路径与算子测试矩阵 |
| [[30_comparison]] | overview(对照,段 3) | CUDA Graphs vs NPU Graphs 真差异表(API 映射/组件对照/捕获时序行为差异);机制原理见 [[01_aclgraph]] 等权威页 |

> 代码示例:`npugraphs_usage_guide.py`;原 README 速览已并入 [[01_aclgraph]](使用限制)与 [[30_comparison]](API 对应关系),信息无损。

---

## 关联域

- [[03_runtime_graphs/cuda/index]] — CUDA Graphs(对比基准)
- [[02_compile_stack/04_inductor/npu/index]] — NPU Inductor(ACLGraph 为其路径之一)
- [[01_eager_runtime/03_op_registration/npu/index]] — 入图判别第三关(aclgraph aclnn-only)
- [[01_pytorch/index]] — 本域总索引
