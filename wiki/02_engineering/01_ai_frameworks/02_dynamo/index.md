# 02 · TorchDynamo 图捕获 — 目录索引

> torch.compile 前端:Python 帧评估钩子、字节码符号执行、Guard 生成与守卫失败重编译。
> 知识分层:overview(本索引)→ quick start → deep dive(约定见 [[01_ai_frameworks/index]])。
> 最后更新: 2026-06-13

---

## 页面列表(按层次)

| 页面 | 层次 | 核心主题 |
|------|------|---------|
| [[dynamo_quickstart]] | **quick start** | 看捕获结果 `torch._dynamo.explain`、graph break 定位(`TORCH_LOGS=graph_breaks`)、`fullgraph=True`、guards/recompiles、`disable`/`allow_in_graph`/`reset` 逃生阀 |
| [[PyTorch_Dynamo_Technical_Analysis]] | deep dive | 帧评估 API、字节码符号执行、OutputGraph/SubgraphTracer、Guard 生成与重编译 |
| [[control_flow_capture_analysis]] | deep dive | 控制流专题:路径 A 显式 HOP(`cond`/`while_loop`/`map`/`scan`)投机子图入图,路径 B 原生 `if`/`for`/`while` 字节码特化/展开/切图 |

> 端到端流水线(Dynamo→AOTAutograd→Inductor)见 [[torch_compile_architecture]]。

---

## 关联域

- [[03_aot_autograd/index]] — 下一阶段:前/反向分解
- [[04_inductor/index]] — 编译后端
- [[01_ai_frameworks/index]] — 本域总索引
