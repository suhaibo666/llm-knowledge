# 设备接入 Quick Start：把 out-of-tree 加速器接进 PyTorch（PrivateUse1）

> 层次：quick start（浅、实用）
> 核验基准：PyTorch 上游 `E:\97-codes\pytorch\pytorch`,官方参考后端 `torch_openreg`(in-tree PrivateUse1 测试后端+接入范例)
> 最后更新:2026-06-13

**一句话**:不改 PyTorch 源码,通过 **PrivateUse1** 这个预留 dispatch key,Python 侧重命名后端 + C++ 侧注册 DeviceGuard / Hooks / 算子 kernel,就能把第三方加速器当成 `torch.device("xxx")` 用。最小可跑路径见下,所有引用均指向 `torch_openreg` 真实文件。

参考根目录(下称 **OpenReg**):
`test/cpp_extensions/open_registration_extension/torch_openreg/`

---

## 1. 最小接入步骤清单

### 步骤 A — Python 侧:重命名后端 + 注册设备模块(3 行)

`torch_openreg/torch_openreg/__init__.py:17-19`:

```python
torch.utils.rename_privateuse1_backend("openreg")          # 把 PrivateUse1 暴露成 "openreg"
torch._register_device_module("openreg", torch_openreg.openreg)  # 绑定 torch.openreg.* 模块
torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)  # 生成 .openreg()/.is_openreg 等
```

核实(函数真实位置):
- `rename_privateuse1_backend(backend_name)` → `torch/utils/backend_registration.py:76`(每进程只能调一次)
- `generate_methods_for_privateuse1_backend(for_tensor=True, for_module=True, for_packed_sequence=True, for_storage=False, unsupported_dtype=None)` → `torch/utils/backend_registration.py:409`
- `_register_device_module(device_type, module)` → `torch/__init__.py:2842`

设备模块需提供的约定 API(`torch_openreg/openreg/__init__.py`):`is_available()`、`device_count()`、`current_device()`、`set_device()`、`device`(context manager);RNG/AMP 见步骤 D/E。

### 步骤 B — C++ 侧:注册 DeviceGuard

实现 `c10::impl::DeviceGuardImplInterface`,`static_type = PrivateUse1`:
`csrc/runtime/OpenRegGuard.h:15`

```cpp
struct OpenRegGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr DeviceType static_type = c10::DeviceType::PrivateUse1;
  DeviceType type() const override { return static_type; }
  Device exchangeDevice(Device d) const override { ... }
  // getDevice/setDevice/getStream/...
};
```

用宏注册(`csrc/runtime/OpenRegGuard.cpp:6`):

```cpp
C10_REGISTER_GUARD_IMPL(PrivateUse1, OpenRegGuardImpl);
```

### 步骤 C — C++ 侧:注册算子 kernel 到 PrivateUse1

最小算子集 + 全局 CPU fallback(`csrc/aten/OpenRegMinimal.cpp:119` 与 `:141`):

```cpp
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {        // OpenRegMinimal.cpp:119
  m.impl("empty.memory_format", wrapper_empty_memory_format);
  m.impl("empty_strided",  wrapper_empty_strided);
  m.impl("as_strided",     wrapper_as_strided);
  m.impl("_copy_from",     wrapper__copy_from);
  m.impl("_local_scalar_dense", wrapper__local_scalar_densor);
  m.impl("view",           wrapper_view);
  // resize_/_reshape_alias/_copy_from_and_resize/set_.* ...
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {           // OpenRegMinimal.cpp:141 全局兜底
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&wrapper_cpu_fallback>());
}
```

> 真实最小集(刚好够建 Tensor 与基本操作):`empty.memory_format`、`empty_strided`、`as_strided`、`resize_`、`_reshape_alias`、`_copy_from`、`_copy_from_and_resize`、`_local_scalar_dense`、`_has_compatible_shallow_copy_type`、`set_.source_Tensor`/`set_.source_Storage`/`set_.source_Storage_storage_offset`、`view`。其余算子靠全局 fallback 落到 CPU。

其它算子注册形态(`csrc/aten/OpenRegExtra.cpp`):
- STUB 形式:`REGISTER_PRIVATEUSE1_DISPATCH(abs_stub, &wrapper_abs_stub)`(`:137`)
- 自定义算子:`TORCH_LIBRARY(openreg, m)` 定 schema(`:148`)+ `TORCH_LIBRARY_IMPL(openreg, PrivateUse1, m)` 实现(`:154`)
- 自定义反向:`TORCH_LIBRARY_IMPL(openreg, AutogradPrivateUse1, m)`(`:182`)
- 单算子覆盖 fallback:`TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)` 内对 `sub.Tensor` 用 boxed CPU fallback(`OpenRegMinimal.cpp:148`)

### 步骤 D — 注册 Hooks(惰性初始化等运行时钩子)

继承 `at::PrivateUse1HooksInterface`(`csrc/runtime/OpenRegHooks.h:15`),静态初始化时注册(`csrc/runtime/OpenRegHooks.cpp:7`):

```cpp
static bool register_hook_flag [[maybe_unused]] = []() {
  at::RegisterPrivateUse1HooksInterface(new OpenRegHooksInterface());
  return true;
}();
```

### 步骤 E — 打包与 autoload

`setup.py:137` 通过 CMake 构建 `_C` 扩展,并用 entry point 实现 `import torch` 时自动加载:

```python
entry_points={"torch.backends": ["torch_openreg = torch_openreg:_autoload"]}
```

安装:`python -m pip install --no-build-isolation -e .`(`README.md`)。`_autoload` 占位函数见 `torch_openreg/__init__.py:38`;环境变量 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 可关闭自动加载(`setup.py:36`)。

---

## 2. 九个接入点速查

对齐官方加速器接入指南 `docs/source/accelerator/`(`index.md` toctree 顺序),每点配 OpenReg 代码:

| # | 接入点 | 官方文档 | OpenReg 关键位置 |
|---|--------|----------|------------------|
| 1 | device(设备管理/device-agnostic) | `accelerator/device.md` | `torch_openreg/openreg/__init__.py` |
| 2 | hooks(运行时钩子/惰性 init) | `accelerator/hooks.md` | `csrc/runtime/OpenRegHooks.{h,cpp}` |
| 3 | guard(DeviceGuard/Stream) | `accelerator/guard.md` | `csrc/runtime/OpenRegGuard.{h,cpp}` |
| 4 | autoload(entry point 自动加载) | `accelerator/autoload.md` | `setup.py` + `__init__.py:_autoload` |
| 5 | operators(算子/fallback/STUB) | `accelerator/operators.md` | `csrc/aten/OpenRegMinimal.cpp`、`OpenRegExtra.cpp` |
| 6 | amp(自动混合精度) | `accelerator/amp.md` | `csrc/amp/autocast_mode.cpp:26` `TORCH_LIBRARY_IMPL(aten, AutocastPrivateUse1, m)` + `openreg/amp/__init__.py:get_amp_supported_dtype` |
| 7 | profiler | `accelerator/profiler.md` | `csrc/profiler/stubs/openreg.cpp` |
| 8 | distributed(自定义 backend) | `accelerator/distributed.md` | `__init__.py:29` `Backend.register_backend("occl", ..., devices=["openreg"])`、`csrc/distributed/c10d/ProcessGroupOCCL.cpp` |
| 9 | ci | `accelerator/ci.md` | `tests/*.py` |

---

## 3. 排查 dispatch

真实命令(均来自 `torch/csrc/utils/python_dispatch.cpp`,通过 `torch._C.*` 暴露):

```python
import torch, torch_openreg

# 某 op 在某 DispatchKey 是否有直接注册的 kernel        (python_dispatch.cpp:558)
torch._C._dispatch_has_kernel_for_dispatch_key("aten::empty.memory_format", "PrivateUse1")  # -> True

# op 是否存在任何 kernel                                 (python_dispatch.cpp:550)
torch._C._dispatch_has_kernel("aten::add.Tensor")

# 打印 op 的注册状态 / 计算后的分发表                     (python_dispatch.cpp:520 / :529)
print(torch._C._dispatch_dump("aten::add.Tensor"))         # dumpState()
print(torch._C._dispatch_dump_table("aten::add.Tensor"))   # dumpComputedTable()

# 该 key 命中的是否是 fallthrough                         (python_dispatch.cpp:569)
torch._C._dispatch_kernel_for_dispatch_key_is_fallthrough("aten::add.Tensor", "PrivateUse1")
```

- op 名格式:`"aten::<op>.<overload>"`(如 `aten::add.Tensor`);dispatch key 用字符串(如 `"PrivateUse1"`、`"AutogradPrivateUse1"`、`"AutocastPrivateUse1"`)。
- `_dispatch_dump`/`_dispatch_dump_table` 找不到 op 时返回空串;`_dispatch_has_kernel_for_dispatch_key` 找不到 op 会 `TORCH_CHECK` 抛错。

**Python 侧拦截**(在 dispatcher 之上看每个 op 的真实调用):用 `__torch_dispatch__`。继承 `torch.utils._python_dispatch.TorchDispatchMode`,重写 `__torch_dispatch__(self, func, types, args, kwargs)` 后用 `with MyMode():` 包住代码,即可逐 op 观察分发(实现见 `torch/utils/_python_dispatch.py`)。算子级注册/伪实现用 `torch.library`(`torch.library.impl(qualname, types, func)`、`torch.library.impl_abstract`,见 `torch/library.py:763` / `:965`)。

---

## 4. 深入导航

- [[privateuse1_device_integration_analysis]] —— deep dive:9 个接入点逐一展开(做什么/为什么/怎么做)+ torch_npu 生产实现对照
- [[pytorch_dispatcher_analysis]] —— deep dive:Dispatcher 内部机制、PrivateUse1 key 如何被分发
- [[07_op_registration/index]] —— 算子供给侧:yaml→codegen→TORCH_LIBRARY_IMPL 注册链路

---

## Related Pages

- [[01_ai_frameworks/index]]
- [[privateuse1_device_integration_analysis]]
- [[pytorch_dispatcher_analysis]]
- [[07_op_registration/index]]
