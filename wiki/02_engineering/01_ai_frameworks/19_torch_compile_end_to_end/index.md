# 19 · `torch.compile` 端到端学习域

> 层次：overview → source-level deep dive  
> 固定源码：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`  
> 当前阶段：原理解读优先；不新增 demo  
> 最后更新：2026-07-28

本域把知识库中分散的 eager 地基、Dynamo、AOTAutograd、Inductor、cache、runtime、
debug、distributed 和 extensibility 组织成一条从 `torch.compile()` 到生产运行的学习主线。

## 从哪里开始

先读 [[00_torch_compile_end_to_end_index]]。它给出 A→F 六卷顺序、每篇前置依赖、
源码入口和三条阅读路径。

```text
A 执行模型
→ B torch.compile 与 Dynamo
→ C FX/AOT/Inductor 图编译
→ D 编译产物、缓存与 runtime
→ E 调试、正确性与性能
→ F 训练、分布式、扩展与部署
```

卷 C 不在本目录复制；它继续由
[[19_torch_compile_end_to_end/00_pytorch_graph_series_index]]维护。本域只为它提供端到端
前置和后续。

## 页面地图

| 卷 | 页面 | 学习目标 |
|---|---:|---|
| A | A01–A05 | 建立 Tensor/operator/Python frame/dispatch mode/成本模型 |
| B | B01–B10 | 解释 API、eval-frame、字节码、guards、break、recompile 与 backend |
| C | C01–C21 | 复用现有 FX/AOT/Inductor 图编译系统课程 |
| D | D01–D07 | 解释编译产物、cache、wrapper、module load 与 CUDAGraph runtime |
| E | E01–E09 | 建立分阶段调试、正确性和性能验收方法 |
| F | F01–F08 | 扩展到训练、分布式、custom op/backend 与 AOTInductor |

## 证据边界

- `[S]`：固定 checkout 的源码事实；
- `[R]`：已有正式 runtime receipt；
- `[I]`：绑定已验证父结论的机制推论；
- `[M]`：只生成代码或使用 mock/no-op，native kernel 未执行；
- `[B]`：环境或能力阻塞。

当前机器没有 MSVC `cl`、CUDA 或 Triton。C++ native kernel、CUDA/Triton autotune 和
CUDAGraph replay 的真实测量不能由源码结构或 generated source 外推。

## Related Pages

- [[00_torch_compile_end_to_end_index]] — 六卷总索引
- [[19_torch_compile_end_to_end/00_pytorch_graph_series_index]] — 卷 C
- [[02_dynamo/index]] — Dynamo 领域资料
- [[17_compile_cache/index]] — 编译缓存领域资料
- [[15_distributed_primitives/index]] — 分布式原语
- [[01_ai_frameworks/index]] — 框架知识总图

