# SGLang 编译 Pass 与 torch.compile 适配 — 从 vLLM 搬来 piecewise 编译管线，却几乎没搬 fusion pass

> **Source baseline**: sglang @ `d6ef68881e`（main，拉取 2026-07-20）
> **Dimension**: Deep Dive（mechanism-level，逐函数读源）
> 本页回答：SGLang `srt/compilation/` 这套 torch.compile / FX pass 体系到底做了什么、由谁驱动、门控在哪；它与 vLLM 的血缘有多深；以及一个反直觉但确凿的结论——**SGLang 出厂的真实「图重写 pass」数量是 0**。与 [[vllm_ir_and_fusion_passes_analysis]] 正好互为对照：vLLM 把重心放在 pattern-matching fusion，SGLang 只搬了 piecewise cudagraph 的管线骨架。方法论层面的定位见 [[fx_pass_optimization_methodology]]，上游基线见 [[torch_upstream_pass_deepdive]]。

---

## 1. 概览

### 主线（一条主线）

**SGLang 的 `compilation/` 是 vLLM `compilation/`（v0.10.0 时点）的一次近乎逐文件 fork，但只 fork 了「piecewise 编译 + 分段 cudagraph 捕获」这套 *plumbing*，把 vLLM 真正的价值所在——一整套 pattern-matching 的 fusion / 图重写 pass——整个略去了。** 证据：整个 `srt/compilation/` 里唯一一个「具体自定义 pass」`FixFunctionalizationPass`，其 `__call__` 只是数了一遍 `auto_functionalized` 节点、然后**一个都不改写**（`fix_functionalization.py:34-37`）；`PostGradPassManager.configure()` 也只挂这一个 no-op pass、`.add()` 全仓无人调用（`pass_manager.py:47-51`）。SGLang 把「融合」这件事外包给了 fused/flashinfer kernel + inductor 原生的 `combo_kernels`（`compilation_config.py:59-61`），而不是自己写 FX pass。

所以这层的定位是：**薄。** 它承担的是「把模型 forward 按注意力/通信 op 切成段、每段单独编译并按 batch 尺寸捕获 cudagraph」的工程管线，而非「在 FX 图上做算子融合」的优化器。这不是缺陷指控而是事实发现——**如果你来这里找 rmsnorm+quant fusion、attention+quant fusion、rope fusion，它们不在 SGLang；它们在 vLLM。**

### 两条 torch.compile 路径（不要混淆）

SGLang 里有两套独立的 torch.compile 接入点，只有第二套用到 `compilation/` 包：

```
              ServerArgs.enable_torch_compile / cuda_graph_config
                              │
        ┌─────────────────────┴──────────────────────────┐
 decode 相: backend="full"                prefill 相: backend="tc_piecewise"(默认)
 + --enable-torch-compile                 + tc_compiler ∈ {eager, inductor}
        │                                                 │
        ▼                                                 ▼
 patch_model()                            TcPiecewiseCudaGraphBackend
 torch_compile_decoration.py:43           _run_compile_pass() 逐尺寸预热
   torch.compile(model.forward,           install_torch_compiled()  compile.py:150
    mode="max-autotune-no-cudagraphs")              │
   —— 香草 inductor,无 piecewise、                  ▼
      无自定义 pass、无图切分            SGLangBackend.__call__  backend.py:404
        │                                           │
        ▼                                           ├─ split_graph() 在 @register_split_op 边界切图
 返回已编译 forward                                 │    (注意力 / all_reduce / MoE dispatch)
 (旧路径, 与 compilation/ 包无关)                    ├─ PiecewiseCompileInterpreter 逐子图编译
                                                    │    CompilerManager → InductorAdaptor / EagerAdapter
                                                    │      inductor_config["post_grad_custom_post_pass"]
                                                    │        = PostGradPassManager
                                                    │          (运行期只跑 1 个 pass:
                                                    │           fix_functionalization = NO-OP)
                                                    ├─ make_backend() → {CUDA,NPU,XPU}PiecewiseBackend
                                                    ▼
                                           每尺寸捕获/回放 cudagraph
                                           cuda_piecewise_backend.py:110
```

`patch_model` 路径（`torch_compile_decoration.py`）是旧的 decode-full 接入：直接 `torch.compile(mode="max-autotune-no-cudagraphs")`，不切图、不进 `compilation/` 包、不跑任何自定义 pass。本页主体是第二条 **tc_piecewise** 路径，`compilation/` 整个包为它服务。

### 关键概念表

| 概念 | 文件:行 | 作用 | 血缘 |
|---|---|---|---|
| `InductorPass` / `SGLangInductorPass` | `inductor_pass.py:50, 116` | pass 基类，uuid=源码 sha256（影响 inductor cache） | vLLM 逐字 fork（`VllmInductorPass`→`SGLangInductorPass`） |
| `PostGradPassManager` | `pass_manager.py:20` | post-grad pass 编排器，作为 inductor `post_grad_custom_post_pass` 挂入 | vLLM fork，但 `configure()` 被掏空 |
| `FixFunctionalizationPass` | `fix_functionalization.py:19` | **唯一**具体自定义 pass；当前为 **no-op**（只计数不改写） | vLLM fork 后删掉整条 if-elif 改写链 |
| `SGLangBackend` | `backend.py:366` | torch.compile backend：切图 + 逐段编译 + 逐段包 piecewise backend | vLLM `VllmBackend` fork |
| `split_graph` / `@register_split_op` | `backend.py:225`, `compilation_config.py:10` | 在注意力/通信 op 处切分 FX 图 | vLLM `splitting_ops` 机制重写 |
| `CompilerManager` / `InductorAdaptor` / `EagerAdapter` | `backend.py:76`, `compiler_interface.py:170,487` | 编译器抽象 + inductor cache 劫持（FxGraphCache monkey-patch） | vLLM 逐字 fork |
| `CUDAPiecewiseBackend` | `cuda_piecewise_backend.py:43` | 每段、每尺寸捕获/回放 cudagraph | vLLM `PiecewiseBackend` fork |
| `NPUPiecewiseBackend` / `XPUPiecewiseBackend` | `npu_piecewise_backend.py:16`, `xpu_piecewise_backend.py:19` | CUDA 版子类，换 `torch.npu.graph` / `torch.xpu.graph` | SGLang 自加（vLLM 无此文件） |
| `combo_kernels` | `compilation_config.py:59-61` | inductor **原生**水平融合（q_norm+k_norm），非自定义 pass | SGLang 自加 |

---

## 2. Pass 框架

### 2.1 `InductorPass` 契约与 uuid/hash

基类 `InductorPass` 继承 torch 的 `CustomGraphPass`（`inductor_pass.py:50`）。核心契约只有一条：**pass 必须提供 `uuid()`，该 uuid 会进 Inductor 的 code cache**——pass 变了，uuid 变，编译产物失效重编。默认实现把 pass 对象的**源码文本** sha256：`uuid()` → `hash_source(self)`（`inductor_pass.py:56-63`），`hash_source` 对 str 直接哈希、对 function/对象取 `inspect.getsource`（`:65-82`）。`SGLangInductorPass`（`:116`）加 `begin()/end_and_log()` 计时与 `dump_graph()`。`is_applicable_for_shape(shape)` 默认恒 True（`:93-94`），配合 `PassContext.runtime_shape` 本可按尺寸门控 pass——但既然没有真实 pass，这条支路目前是死的。

### 2.2 `PostGradPassManager` 的编排——以及它被掏空的证据

docstring 号称 pass 顺序是「构造参数 passes → 默认 passes(NoopEliminationPass, FusionPass) → config 自定义 pass → fix_functionalization」（`pass_manager.py:27-32`）。**这段 docstring 是从 vLLM 逐字抄来的，与 SGLang 实现不符**：

```python
def configure(self):                      # pass_manager.py:47-51
    self.pass_config = dict()
    self.fix_functionalization = FixFunctionalizationPass()
```

`configure()` 里**没有** NoopElimination、没有 FusionPass，只 new 了一个 fix_functionalization。`self.passes` 初始为空（`:36`），只能靠 `add()` 追加（`:53-55`），而 `add()` 在整个 `srt/` **无人调用**（grep 确认零命中）。于是运行期 `__call__`（`:38-45`）的循环 `for pass_ in self.passes` 直接空转，只剩最后那句 `self.fix_functionalization(graph)`。

**门控**：PassManager 只在 `SGLangBackend.configure_post_pass()` 里被设进 `inductor_config["post_grad_custom_post_pass"]`（`backend.py:400-402`），而该项**只有 InductorAdaptor 走 `compile_fx` 时才被消费**；若 `tc_compiler="eager"`（代码强烈默认的取值），`EagerAdapter.compile` 直接返回原图（`compiler_interface.py:490-499`），PassManager 连跑都不跑。两种情况净效果一致：**零图重写**。

### 2.3 compile 驱动与 phases

- **驱动**：`TcPiecewiseCudaGraphBackend` 在 cudagraph 捕获前调 `_run_compile_pass`，它 `install_torch_compiled(inner_model)`（`compile.py:150`）把 `model.forward` 换成 trampoline，再对 `capture_num_tokens` 里每个尺寸各跑一次 dummy forward，驱动 Dynamo/inductor 逐尺寸编译**但先不捕获 cudagraph**。
- **phase 标志**（`compile_phase.py`）：进程级状态。`_in_torch_compile_warmup` 让 piecewise backend 在 warmup 期短路掉 cudagraph 捕获；`_pcg_capture_stream` 把 runner 的捕获 stream 透给 FX backend。
- **install 机制**（`compile.py:150`）：`register_bytecode_hook` + trampoline 实现「首次真正需要时才编译」，并在编译前给运行期尺寸维打 `maybe_mark_dynamic`。

### 2.4 backend：切图 + 逐段编译

`SGLangBackend.__call__`（`backend.py:404`）拿到整张 FX 图后：
1. `split_graph(graph, split_ops)`（`:225-268`）：凡 `call_function` 且 `str(target)` 命中 `split_ops` 就单独成段；用 `split_module(..., keep_original_order=True)` 切分——注释强调 `keep_original_order` 至关重要，否则 pytorch 重排节点会破坏含 mutation 的图语义（`:243-249`）。
2. `PiecewiseCompileInterpreter`（`:274`）对每个非「切分 op」子模块用 `CompilerManager.compile(runtime_shape=None)` 编译 dynamic-shape 版本，再用 `make_backend` 把该段替换成 piecewise backend 实例。
3. `InductorAdaptor.compile`（`compiler_interface.py:203`）大量 monkey-patch inductor 内部（`FxGraphCache._get_shape_env`→`AlwaysHitShapeEnv`、放行 `_check_can_cache`）以便「在 Dynamo tracing context 之外仍命中 inductor code cache」——这段连注释里的 "vLLM" 字样都原样保留（`:300-301, 352`）。

### 2.5 split_ops：图在哪里被切

切分边界靠装饰器 `@register_split_op()`（`compilation_config.py:10-16`）在定义算子处注册进全局 `SPLIT_OPS`。已确认注册点：`inplace_all_reduce`（TP all-reduce）、`unified_attention_with_output`（注意力）、`radix_linear_attention`、`dsa_indexer`、`deepseek_v4`、`forward_mla`、`nemotron_h`；DeepEP/Mooncake 在用时追加 MoE dispatch op。语义等价于 vLLM 的 `splitting_ops`：**在注意力和集合通信处切段**，使每段「纯计算」子图能被独立 inductor 编译并独立捕获 cudagraph，把动态、不可捕获的注意力/通信留在段与段之间。

### 2.6 piecewise cudagraph backends（三平台差异）

`CUDAPiecewiseBackend.__call__`（`cuda_piecewise_backend.py:110-222`）是每段运行期状态机，`ConcreteSizeEntry` 按尺寸持有 `(need_to_compile, use_cudagraph, compiled, runnable, cudagraph, output)`：首跑走 general-shape 结果并触发编译收尾；该尺寸需专门编译则 `compile(runtime_shape=...)` 并开 `max_autotune`+`coordinate_descent_tuning`；warmup 期直接跑不捕获；每尺寸首次先 warmup 一次再在 `torch.cuda.graph(...)` 里捕获，最后一段把 output 转 `weak_ref_tensors` 省显存；之后同尺寸直接 `replay()`。**健壮性分支**：若捕获 stream 为空（Dynamo 静默重编译换了 backend）回退 eager 而非崩溃（`:162-170`）。

| Backend | graph 类型 | stream 处理 | 与 CUDA 版差异 |
|---|---|---|---|
| `CUDAPiecewiseBackend` | `torch.cuda.CUDAGraph()` | 用 `get_pcg_capture_stream()`，为空则回退 eager | 基类 |
| `XPUPiecewiseBackend` | `torch.xpu.XPUGraph()` | 同样查 stream、为空回退 | 几乎逐行照抄，仅换 `torch.xpu.*` |
| `NPUPiecewiseBackend` | `torch.npu.NPUGraph()` | **不查 stream**、`torch.npu.graph(...)` 不传 stream | 覆写整个 `__call__`，**缺少 warmup 短路与 stream-为空回退两条分支**（`npu_piecewise_backend.py:41-109`） |

`weak_ref_tensors` 按平台从 `sgl_kernel`（CUDA/HIP/MUSA/XPU）或 `torch_npu._C._weak_ref_tensor`（NPU）取底层实现。

> [inferred] NPU 子类没有 CUDA 版的 warmup 短路，也没有 stream-为空的 eager 回退——意味着 NPU 路径若遇 Dynamo 运行期重编译，行为与 CUDA/XPU 不一致。这是移植取舍/遗漏，非源码显式声明。

---

## 3. 实际的自定义 pass：`fix_functionalization` 详解——以及「是否仅此一个」

### 3.1 它本应干什么（vLLM 语义）

在 vLLM 里，inductor 会把带原地 mutation 的自定义算子（rms_norm、rotary_embedding、silu_and_mul…）包成 HOP `auto_functionalized` 以满足函数式化，代价是额外张量拷贝。`FixFunctionalizationPass` 的职责是**在 post-grad 阶段把这些节点「去函数式化」**：用 `defunctionalize()` 换回原地算子、把 `getitem` 用户重定向到被 mutate 的实参、删冗余节点，消掉拷贝。vLLM 版 `__call__` 是一条很长的 if-elif 链，对每类算子分别 `defunctionalize(...)`（见 [[vllm_ir_and_fusion_passes_analysis]] 引用的 `utility/fix_functionalization.py`）。

### 3.2 它在 SGLang 里实际干什么：**什么都不改写**

SGLang 保留了所有 helper——`defunctionalize`（`fix_functionalization.py:61`）、`replace_users_with_mutated_args`、`getitem_users`、`insert_defunctionalized`——但 **`__call__` 里那条 if-elif 改写链被整段删掉了**：

```python
def __call__(self, graph):                       # fix_functionalization.py:28-50
    ...
    count = 0
    for node in graph.nodes:
        if not is_func(node, auto_functionalized):
            continue                              # ← 唯一分支
        count += 1                                # ← 只计数,从不调 defunctionalize / _remove
    count_removed = len(self.nodes_to_remove)     # 恒为 0(nodes_to_remove 从未 append)
    for node in self.nodes_to_remove:             # 空循环
        graph.erase_node(node)
    logger.debug("De-functionalized %s nodes, removed %s nodes", count, count_removed)
```

循环体只有 `count += 1; continue`，`self.nodes_to_remove` 永远空，`count_removed` 恒 0，`defunctionalize`/`_remove` 从不被 `__call__` 触达。**这是一个「结构存在、功能为空」的 no-op pass**：遍历、计数、打日志、dump 三张调试图，但对 FX 图零改动。

雪上加霜：backend 仍把 `enable_auto_functionalized_v2` 设为 `False`（`backend.py:396`），即**强制生成 auto_functionalized(v1) 节点**（正是本应被这个 pass 清理的东西），同时 `monkey_patch_torch_compile` 把 `auto_functionalized._cacheable=True`（`utils/patch_torch.py:112-113`）。

> [inferred] 组合看：SGLang 关掉了更高效的 auto_functionalized_v2、生成 v1 节点、又不去函数式化它们，留下潜在冗余拷贝开销——除非上游 fused kernel 本身不经 auto_functionalized 路径，或该开销在 eager-compiler 默认下不实际发生。源码只陈述配置项，净效果是推断。

### 3.3 是否仅此一个？——是，而且它还是空的

跨整个 `srt/` grep `register_graph_pattern|PatternMatcherPass|register_replacement|CustomGraphPass|post_grad_custom_post_pass`，命中仅 `compilation/` 内部六个文件 + `utils/patch_torch.py`（后者仅设 `_cacheable`）。**没有任何 rmsnorm/quant/attention/rope/collective fusion、noop elimination 的 pattern-matching pass**。结论：

- SGLang 出厂**真实图重写 pass 数 = 0**（唯一的 fix_functionalization 是 no-op）；算「框架上挂着的」则 = 1，但它不改图。
- 对比 vLLM：`compilation/passes/fusion/` 下**十余个**融合 pass + IR/utility 支撑 pass 若干，`pass_manager.configure()` 按条件挂载十余个真实 pass（完整目录见 [[vllm_ir_and_fusion_passes_analysis]]）。

**SGLang 用什么替代 fusion pass？** 两条：(1) 预融合好的 kernel（flashinfer / sgl-kernel / 上游 fused op 如 `unified_attention_with_output`、fused rmsnorm）；(2) inductor **原生**能力——`combo_kernels`+`benchmark_combo_kernel`（`compilation_config.py:59-61`，仅 `compiler="inductor"` 时开）把 q_norm+k_norm 之类 sibling op 水平融合，以及专门尺寸下的 `max_autotune`。即 SGLang 把「融合」下推到 kernel 层与 inductor 层，不在 FX pass 层做。

---

## 4. 与 vLLM 的血缘 & 与上游/NPU 的定位对比

### 4.1 血缘：哪些是近拷贝

每个文件头都写着 `Adapted from https://github.com/vllm-project/vllm/blob/v0.10.0/vllm/compilation/<same-name>.py`，文件名与 vLLM 一一对应。按改动量分档：

| SGLang 文件 | 与 vLLM 关系 |
|---|---|
| `inductor_pass.py` / `fx_utils.py` / `compilation_counter.py` / `compiler_interface.py` | **近逐字**（仅改名；注释多处仍写 "vLLM"） |
| `pass_manager.py` | **fork 但掏空**：docstring 抄 vLLM（提 NoopElimination/FusionPass），`configure()` 只挂 fix_functionalization |
| `fix_functionalization.py` | **fork 后删掉核心改写链**：helper 全留、`__call__` 变 no-op |
| `backend.py` / `cuda_piecewise_backend.py` | **近逐字**（`VllmBackend`→`SGLangBackend`，`PiecewiseBackend`→`CUDAPiecewiseBackend`） |
| `compilation_config.py` | **重写**（vLLM 的 `CompilationConfig` 很大；SGLang 只留 split_ops + capture_sizes + combo_kernels，注释 "TODO: support better compile config"） |
| `compile.py` / `compile_phase.py` / `torch_compile_decoration.py` / `tc_piecewise_cuda_graph_backend.py` | **SGLang 原创**（install/trampoline、warmup 标志、runner 驱动） |
| `npu_piecewise_backend.py` / `xpu_piecewise_backend.py` | **SGLang 自加**（vLLM 无这两个文件） |

一句话：**pass 契约层与 piecewise 管线层是 vLLM 近拷贝；pass 内容层（pass_manager 挂载 + fix_functionalization 改写）被抽空；驱动层与多平台层是 SGLang 自写。**

### 4.2 定位对比：vLLM ↔ SGLang ↔ 上游/NPU

- **vLLM**：pass 体系是「编译期优化器」——用 PatternMatcher 在 FX 图上做十几种融合（rms+quant、silu+quant、attn+quant、rope+kvcache、collective+rms、sequence parallelism），fix_functionalization 是这套融合的「收尾清理」。torch.compile 是优化主战场。
- **SGLang**：`compilation/` 是「piecewise cudagraph 的编译管线」，核心价值在**按注意力/通信切图 + 逐尺寸 cudagraph 捕获/回放**，融合外包给 kernel 与 inductor 原生能力。pass 层薄、且当前 fix_functionalization 为 no-op。
- **NPU 侧**：`NPUPiecewiseBackend` 把 `torch.cuda.CUDAGraph` 换成 `torch.npu.NPUGraph`、`weak_ref_tensor` 换成 `torch_npu._C._weak_ref_tensor`；但覆写 `__call__` 时丢了 warmup 短路与 stream-为空回退。这与 [[aclgraph]] 的「捕获-回放静态图」范式相邻——只是 device API 不同，且 SGLang 这层不依赖任何 FX 融合 pass，对 NPU 移植更友好（要搬的只有捕获壳，不用搬一堆 CUDA-only fusion pattern）。

### 4.3 给读者的结论（诚实版）

如果你带着「SGLang 的 compile pass 做了哪些图优化」来，答案是：**目前一个都没做**。这层是工程管线而非优化器。它的工程价值真实存在（piecewise + 逐尺寸 cudagraph + inductor cache 劫持 + 多平台捕获），但「pass」这个词在这里名不副实——真正的算子融合发生在 kernel 与 inductor 内部。fix_functionalization 是这个判断最硬的单点证据：一个从 vLLM 抄来、helper 齐全、却在 `__call__` 里被抽成 no-op 的 pass。

---

## Related Pages

- [[vllm_ir_and_fusion_passes_analysis]] — 对照面：vLLM 的 IR/fusion pass 全家桶（本页反复引用其 `passes/fusion`、`passes/utility` 作为血缘对照）
- [[torch_upstream_pass_deepdive]] — 上游 Inductor pass 全集与机制（SGLang/vLLM pass 框架的共同基座 `CustomGraphPass`/`post_grad_custom_post_pass`）
- [[fx_pass_optimization_methodology]] — 工业界 pass 开发方法论归纳（SGLang 是「把融合下推到 kernel/inductor」这一路线的代表）
- [[aclgraph]] — NPU 侧「捕获-回放静态图」范式，与 `NPUPiecewiseBackend` 相邻
