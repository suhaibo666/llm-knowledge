# `torch.compile` A–F 配套 Demo 交付报告

> 交付日期：2026-07-29
> 实现审计基线：PyTorch `e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52`
> 本机预检：Windows、Python 3.13.5、PyTorch `2.9.1+cpu`
> 正式验收主线：CUDA/Linux；本机不具备 CUDA、Triton、native compiler 或多卡验收条件

## 1. 交付结果

本轮为 A–F 六卷建立了统一可执行层，同时保持“原理正文解释机制、Demo 只证明实际运行”
的证据边界：

| 卷 | 入口 | case 数 | 覆盖主题 |
|---|---|---:|---|
| A | `labs/demo_a_execution_model.py` | 5 | Tensor/storage、dispatcher/autograd、frame/bytecode、Proxy/FakeTensor、成本 |
| B | `labs/demo_b_dynamo_capture.py` | 10 | lifecycle、cache、bytecode、guard、break、dynamic shape、backend |
| C | `labs/demo_c_graph_compiler.py` | 6 | 既有 Part I–IV Labs 与全链路 bundle 编排 |
| D | `labs/demo_d_artifact_runtime.py` | 7 | compile_fx、AOT wrapper、module load、cache、memory、CUDAGraph、failure |
| E | `labs/demo_e_diagnostics.py` | 9 | explain、recompile、故障分层、repro、正确性、性能、rollout |
| F | `labs/demo_f_advanced_topics.py` | 8 | compiled autograd、checkpoint、DDP/FSDP、custom op/backend、AOTI |

合计 45 个 case。`demo_manifest.json` 把 60 篇课程正文一一映射到卷入口和主 case；
C01–C21 还可运行 `full_bundle`，但 manifest 保留更细的主题主 case。

## 2. 统一执行合同

所有入口共享：

```powershell
python -B <demo.py> --list --json
python -B <demo.py> --case <case-id> --device cuda --output-dir <dir>
python -B <demo.py> --case all --device cuda --output-dir <dir>
```

- `0 / PASS`：case 正文执行，内置机制断言全部通过；
- `2 / FAIL`：正文执行后发生断言、编译、子进程或 runtime 失败；
- `3 / BLOCKED`：能力缺失，case 正文没有执行；
- `4`：CLI 参数、未知 case 或合同错误。

每个 case 写 `<case>/result.json`，整次运行写 `summary.json`。结果固定环境、requirements、
观察值、artifact、限制和异常；异常不能被吞掉后返回 `PASS`。

## 3. 关键实现机制

### 3.1 能力门

能力快照区分 `torch`、`cuda`、`cuda_multi_gpu`、`distributed`、`triton`、
`native_compiler` 与 `linux`。默认设备是 CUDA，所以即使 case 的机制本身设备无关，
默认运行仍要求 CUDA；开发机必须显式 `--device cpu` 才进入 CPU 预检。

### 3.2 C 卷编排隔离

C 卷不复制既有 FX/AOT/PatternMatcher/Inductor 实验。每个主题 case 在独立子进程中运行
原脚本，记录：

- 完整命令和退出码；
- stdout/stderr；
- 子脚本 artifact 相对路径；
- 已完成子进程清单。

任一子进程失败会使 case 成为 `FAIL`，同时保留此前证据。进程隔离避免私有 compiler
config、hook、cache 或全局注册污染后续实验。

### 3.3 源码事实与 runtime 事实分离

正文的实现机制仍由固定源码定位支撑。Demo 的 `PASS` 只说明当前版本、输入、dtype、shape
和设备中的断言成立。generated source、FakeTensor、codegen-only 或 CPU 结果不能升级为
CUDA kernel、显存、autotune、CUDAGraph 或多卡事实。

## 4. 本机实际预检

| 卷 | PASS | BLOCKED | 说明 |
|---|---:|---:|---|
| A | 4 | 1 | CUDA 成本计时阻断 |
| B | 10 | 0 | 全部使用设备无关 backend/捕获合同 |
| C | 6 | 0 | 五个主题编排与 full bundle 全部通过 |
| D | 3 | 4 | native compiler、CUDA/Triton 路径阻断 |
| E | 7 | 2 | CUDA timing/profiler 阻断 |
| F | 4 | 4 | Linux DDP、多卡、AOTI、CUDA freezing 阻断 |

预检过程中发现并修正两项合同问题：

1. after-Dynamo repro 生成器曾错误接收 `make_fx` 图，独立脚本出现未定义类型注解；
   改为真实 Dynamo backend 捕获的 GraphModule/example inputs 后，复现脚本独立执行通过。
2. Windows 构建报告 `distributed.is_available()`，但本机 Gloo 无可用 device；
   DDP 教学 case 明确声明 Linux runtime gate，避免把“编译进 binary”误当“可建立进程组”。

## 5. 自动验证

最终门禁：

| 门禁 | 结果 |
|---|---:|
| Labs 合同测试 | 63/63，`OK` |
| 审计工具测试 | 90/90，`OK` |
| 六入口 `--list --json` subprocess | 6/6 |
| 正文 → case manifest | 60/60 |
| 非 C 正文 Demo 回链 | 39/39 |
| C 编排主题覆盖 | C01–C21 全覆盖 |
| Markdown/链接/mermaid 结构检查 | 64 文件，0 error |
| course claim decisions | 6,483/6,483 |
| claim-ledger validation errors | 0 |
| `git diff --check` | exit 0 |

重建后证据类别为：`[S]=1,287`、`[R]=366`、`[I]=3,675`、`[M]=19`、
`[B]=43`，另有 1,093 个非断言或操作说明。

## 6. 尚未完成的目标环境验收

当前不能给出以下 `PASS`：

- CUDA/Triton kernel 真实编译与执行；
- CUDA event 稳态性能、显存峰值、fusion 收益和 autotune winner；
- CUDAGraph warmup/record/replay 与 output lifetime；
- Linux DDP 和双卡 FSDP2/DTensor；
- AOTInductor `.pt2` package 的目标环境编译、加载和 ABI。

这些不是实现缺失，而是明确的验收待办。目标 CUDA/Linux 环境应按
`labs/demo_manifest.json` 逐 case 运行，并保留 `summary.json`、`result.json` 与所有
artifact，成功后才能更新知识库中的 runtime evidence 等级。
