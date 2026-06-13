# 03 · AOTAutograd — 目录索引

> 介于 Dynamo 与 Inductor 之间:functionalization、前/反向联合图(joint graph)生成与 min-cut partition。
> 知识分层:overview(本索引)→ quick start → deep dive(约定见 [[01_ai_frameworks/index]])。
> 最后更新: 2026-06-13

---

## 页面列表(按层次)

| 页面 | 层次 | 核心主题 |
|------|------|---------|
| [[aot_autograd_quickstart]] | **quick start** | 看前/反向图:`backend="aot_eager"` + `TORCH_LOGS=aot_graphs`;看联合图 `aot_joint_graph`;partitioner(min-cut vs default)与重计算;`aot_function` 最小用法;`AOT_PARTITIONER_DEBUG`/activation_memory_budget |
| [[aotautograd_analysis]] | deep dive | aot_function/aot_module、joint graph 构建、partitioner、functionalization、runtime wrappers |

> joint graph 上的优化 pass 见 [[joint_graph_passes_guide]](实现于 Inductor `fx_passes/joint_graph.py`)。

---

## 关联域

- [[02_dynamo/index]] — 上游:图捕获
- [[04_inductor/index]] — 下游:lowering 与 codegen
- [[01_ai_frameworks/index]] — 本域总索引
