# 04 · 图导出与分布式原语 — 目录索引

> 本层包含两类扩展面：`torch.fx`/`torch.export`/自定义算子接入（eager 图 IR 与图导出），以及 `torch.distributed` 原生原语（c10d/DDP/FSDP/DTensor，是训练框架应用层的底座）。

## 子目录

| 目录 | 核心主题 |
|------|---------|
| [[01_fx_export_extensibility/index]] | `torch.fx`(eager 图 IR)、`torch.export`(ExportedProgram)、`torch.library`/custom_op、functorch(vmap/grad) |
| [[02_distributed_primitives/index]] | `torch.distributed` 原生原语:c10d/ProcessGroup、DDP Reducer、FSDP/FSDP2、DTensor/DeviceMesh、TP/PP |

## Related Pages

- [[01_ai_frameworks/index]] — 本域总索引（5 层架构导航）
- [[02_train_frameworks/index]] — 训练框架：建立在 [[02_distributed_primitives/index]] 之上的并行应用层
