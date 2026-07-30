# 02 · torch.compile 编译栈 — 目录索引

> 把 eager 计算(见 [[01_eager_runtime/index]])捕获、分解、编译成 kernel 的完整链路:Dynamo 图捕获 → AOTAutograd 前/反向分解 → Graph IR/Passes → TorchInductor lowering/调度/codegen → codegen 后端 → 跨阶段编译缓存 → 调试诊断。

## 子目录

| 目录 | 核心主题 |
|------|---------|
| [[01_dynamo/index]] | torch.compile 前端:帧评估图捕获、Guard、字节码符号执行 |
| [[02_aot_autograd/index]] | 前/反向图分解、partition、functionalization(编译期 autograd) |
| [[03_graph_ir_and_passes/index]] | Graph IR 与 Passes(P4 后续任务填充) |
| [[04_inductor/index]] | 编译后端核心:lowering、调度、codegen、FX passes、动态形状;`npu/` NPU Inductor 后端 |
| [[05_codegen_backends/index]] | codegen 后端:MLIR(及 Triton 对比) |
| [[06_compile_cache/index]] | 跨阶段编译缓存:Dynamo PGO、AOTAutograd result、Inductor FX graph artifact、Triton autotune cache |
| [[07_debugging/index]] | 调试与诊断(P4 后续任务填充) |

## Related Pages

- [[01_ai_frameworks/index]] — 本域总索引(5 层架构导航)
- [[01_eager_runtime/index]] — eager 运行时地基(本层的底座)
- [[03_runtime_graphs/index]] — 运行时图捕获(与 Inductor `mode="reduce-overhead"` 集成)
