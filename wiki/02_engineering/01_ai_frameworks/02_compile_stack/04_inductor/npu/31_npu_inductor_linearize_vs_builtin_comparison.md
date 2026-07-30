# 三方对标：GPU / torch_npu 内置 / npu_inductor Linearize（output code + 实测）

> 对比对象：**GPU（CUDA Inductor）** vs **torch_npu 内置 `_inductor`（default/Split-Tiling）** vs **实验性 `npu_inductor_2.9.0`（Linearize）** —— 三方生成 kernel 逐行对比 + 算子级实测对标。
> 数据来源：`npu_inductor_2.9.0` 仓库 `test_cases/benchmarks/torchbench/{result,result2}.xlsx`、`test_cases/{compile,compile2}.log`、京东 OneRec 客户现场实测。
> 版本基线：`npu_inductor_2.9.0` 包 + PyTorch 2.9.0；内置后端 = torch_npu v2.7.1.post5。
> 最后更新：2026-06-17

> 本页是 [[23_npu_inductor_linearize_backend_analysis]] 系列的对标分册。机制原理见主页与 [[24_npu_inductor_linearize_dynamic_shape_analysis]]；内置后端细节见 [[11_npu_inductor_splittiling_backend_analysis]]；GPU 上游基线见 [[23_inductor_gpu_kernel_dispatch_model]]。
>
> [!note] 实测数据口径
> §二实测数字均为作者在仓库内的实测记录（本知识库未独立复跑），以实际环境为准。**只对比算子加速比/设备 kernel 耗时**（直接反映 codegen 质量）；E2E 因两侧 aclgraph 开关不一致**不可比**，不纳入。

---

## 一、开篇案例：permute + add 动态 shape 的三方 output code

`x.permute(0,2,1) + y`，`dynamic=True`，把 Linearize / 索引拆轴 / 动态 numel·divisor 参数流落到可见产物上。三符号记为 `s77`=256(batch)、`s27`=128、`s53`=1024。

```python
class Model(nn.Module):
    def forward(self, x, y):
        return x.permute(0, 2, 1) + y
model = torch.compile(Model(), dynamic=True)
x = torch.randn([256, 128, 1024], device="npu")   # [s0, s1, s2]
y = torch.randn([256, 1024, 128], device="npu")    # [s0, s2, s1]
```

### 1.1 GPU（CUDA Inductor）—— 多维 grid，kernel 内无循环，运行期 mod/div 还原 permute

```python
@triton_heuristics.pointwise(size_hints={'y': 32768, 'x': 1024}, ...,
    triton_meta={'signature': {..., 'ks0': 'i64', 'ks1': 'i64', 'ynumel': 'i32', 'xnumel': 'i32', ...}},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', ...})
@triton.jit
def triton_poi_fused_add_permute_0(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ynumel, xnumel,
                                   YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]; ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]; xmask = xindex < xnumel
    x2 = xindex; y3 = yindex
    y0 = (yindex % ks1)        # ← 运行期 mod：从扁平 y 还原 permute 子维（GPU 上廉价）
    y1 = yindex // ks1         # ← 运行期 div
    tmp0 = tl.load(in_ptr0 + (x2 + ks0*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + ks1*x2 + ks0*ks1*y1), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + ks0*y3), tmp2, xmask & ymask)
```

- `Grid2DWithYZOverflow` + `program_id(0/1/2)` **多维 grid**，kernel 内**无循环**；permute 子维靠 `yindex % ks1` / `// ks1` 运行期还原；动态符号只作 `ks0/ks1` + `ynumel/xnumel`。

### 1.2 torch_npu 内置（default，Split-Tiling）—— 动态多维 GridNpu + 三层 SUB_BLOCK 嵌套循环 + kernel 内显式 permute

```python
@triton_heuristics.pointwise(size_hints={'y1': 256, 'y0': 128, 'x2': 1024},
    triton_meta={'signature': {..., 'ks0':'i32','ks1':'i32','y1_numel':'i32','y0_numel':'i32','x2_numel':'i32'},
                 'device': DeviceProperties(type='npu', multi_processor_count=40, cc='Ascend910B3', ...), 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'split_axis': [0], 'tiling_axis': [0,1,2], 'no_loop_axis': [],
                   'axis_names': ['y1','y0','x2'], 'inductor_ascend_linear_mode': 'linear', ...})
@triton.jit
def triton_poi_fused_add_0(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, y1_numel, y0_numel, x2_numel,
                           Y1BLOCK: tl.constexpr, Y1BLOCK_SUB: tl.constexpr,
                           Y0BLOCK_SUB: tl.constexpr, X2BLOCK_SUB: tl.constexpr):
    y1_offset = tl.program_id(0) * Y1BLOCK                    # split 轴 y1 → grid
    base_y1 = tl.arange(0, Y1BLOCK_SUB); loops_y1 = (Y1BLOCK + Y1BLOCK_SUB - 1) // Y1BLOCK_SUB
    base_y0 = tl.arange(0, Y0BLOCK_SUB); loops_y0 = (y0_numel + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x2 = tl.arange(0, X2BLOCK_SUB); loops_x2 = (x2_numel + X2BLOCK_SUB - 1) // X2BLOCK_SUB
    for loop_y1 in range(loops_y1):                          # tiling 轴三层 SUB_BLOCK 嵌套循环
        y1 = y1_offset + (loop_y1 * Y1BLOCK_SUB) + base_y1[:, None, None]
        y1_mask = y1 < min(Y1BLOCK + y1_offset, y1_numel)
        for loop_y0 in range(loops_y0):
            y0_2 = (loop_y0 * Y0BLOCK_SUB) + base_y0[None, None, :]
            y0   = (loop_y0 * Y0BLOCK_SUB) + base_y0[None, :, None]
            y0_mask = y0 < y0_numel; y0_2_mask = y0_2 < y0_numel
            for loop_x2 in range(loops_x2):
                x2_1 = (loop_x2 * X2BLOCK_SUB) + base_x2[None, :, None]
                x2   = (loop_x2 * X2BLOCK_SUB) + base_x2[None, None, :]
                x2_mask = x2 < x2_numel; x2_1_mask = x2_1 < x2_numel
                tmp0 = tl.load(in_ptr0 + (x2 + ks1*y0 + ks0*ks1*y1), y1_mask & y0_mask & x2_mask)
                tmp1 = tl.load(in_ptr1 + (y0_2 + ks0*x2_1 + ks0*ks1*y1), x2_1_mask & y0_2_mask & y1_mask)
                tmp2 = tmp1.permute([0, 2, 1])               # ← kernel 内显式 permute 后相加
                tmp3 = tmp0 + tmp2
                tl.store(out_ptr0 + (x2 + ks1*y0 + ks0*ks1*y1), tmp3, y1_mask & y0_mask & x2_mask)
```

- `grid_type='GridNpu'` 动态多维 grid；meta 暴露 SplitTiling 选轴 `split_axis=[0]`/`tiling_axis=[0,1,2]`；**三层 SUB_BLOCK 嵌套循环**；permute 用 kernel 内显式 `tmp1.permute([0,2,1])`（并为两输入各备一套索引 `y0/y0_2`、`x2/x2_1`）；动态符号 `ks0/ks1` + 每轴 numel，**无 divisor**。

### 1.3 本后端 npu_inductor（Linearize）—— 固定 40-CU group dispatch + 索引拆轴（无 mod/div、无 kernel 内 permute）

```python
@npu_triton_heuristics.pointwise(size_hints={'y': 32768, 'x': 1024}, ...,
    triton_meta={'signature': {..., 'ks0':'i64','ks1':'i64','ynumel':'i32',
                 'y0numel':'i32','y1numel':'i32','y1divisor':'i32','xnumel':'i32','x2numel':'i32'},
                 'device': DeviceProperties(type='npu', multi_processor_count=40, cc='Ascend910B3', ...), 'mix_mode': 'aiv',
                 'axis_hints': [{'name':'y0','divisor':1,'seed':1},{'name':'y1','divisor':128,'seed':2},{'name':'x2','divisor':1,'seed':1}], ...},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'num_load': 2, 'npu_num_x_nodes': 3, ...})
@triton.jit
def triton_unk_fused_add_permute_0(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ynumel,
                           y0numel, y1numel, y1divisor, xnumel, x2numel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    total_thread = 40
    group_id = tl.program_id(0)
    real_block_y0 = y0numel if y0numel <= (YBLOCK // 1) else (YBLOCK // 1) if (YBLOCK > 1) else 1
    real_block_y1 = y1numel if y1numel <= (YBLOCK // y1divisor) else (YBLOCK // y1divisor) if (YBLOCK > y1divisor) else 1
    y0_blocks = (y0numel + real_block_y0 - 1) // real_block_y0
    y1_blocks = (y1numel + real_block_y1 - 1) // real_block_y1
    real_block_x2 = x2numel if x2numel <= (XBLOCK // 1) else (XBLOCK // 1) if (XBLOCK > 1) else 1
    x2_blocks = (x2numel + real_block_x2 - 1) // real_block_x2
    total_blocks = y0_blocks * y1_blocks * x2_blocks
    group_size = total_blocks // total_thread; group_tail = total_blocks % total_thread
    group_base = group_id * group_size + group_tail
    if group_id < group_tail: group_size = group_size + 1
    if group_id < group_tail: group_base = group_id * group_size
    for i in range(group_size):
        x2offset = (group_base + i) // y0_blocks // y1_blocks % x2_blocks * real_block_x2
        x2index = x2offset + tl.arange(0, XBLOCK)[None, None, :]; x2 = x2index; x2mask = x2index < x2numel
        y0offset = (group_base + i) % y0_blocks * real_block_y0
        y1offset = (group_base + i) // y0_blocks % y1_blocks * real_block_y1
        y0index = y0offset + tl.arange(0, YBLOCK)[None, :, None]; y0 = y0index; y0mask = y0index < y0numel
        # y1 divisor 为符号(=s27)、divisor_hint=128>1 → inner-loop（动态 shape 情形 B/C）
        for y1inner in range(0, real_block_y1, ((YBLOCK // 128) if (YBLOCK > 128) else 1)):
            y1index = y1offset + y1inner + tl.arange(0, ((YBLOCK // 128) if (YBLOCK > 128) else 1))[:, None, None]
            y1 = y1index; y1mask = y1index < y1numel
            tmp0 = tl.load(in_ptr0 + (x2 + ks1*y0 + ks0*ks1*y1), x2mask & y0mask & y1mask, eviction_policy='evict_last')
            tmp1 = tl.load(in_ptr1 + (y0 + ks0*x2 + ks0*ks1*y1), x2mask & y0mask & y1mask, eviction_policy='evict_last')
            tmp2 = tmp0 + tmp1
            tl.store(out_ptr0 + (x2 + ks1*y0 + ks0*ks1*y1), tmp2, x2mask & y0mask & y1mask)
```

- `total_thread=40` group dispatch + 单层 `for i in range(group_size)`；permute 维保持原生子轴（`_maybe_split_fused_axes` 撤销 y 轴合并），load/store **无 mod/div、无 kernel 内 permute**；`y1` divisor 符号、`hint=128>1` 故多一层 `for y1inner`；签名多 `y1divisor`（动态 numel·divisor 参数流）。机制详见 [[24_npu_inductor_linearize_dynamic_shape_analysis]] §三、§五。

### 1.4 三方逐项差异

| 维度 | GPU（CUDA） | torch_npu 内置（Split-Tiling） | 本后端（Linearize） |
|---|---|---|---|
| grid_type | `Grid2DWithYZOverflow` | `GridNpu`（动态多维 grid） | `Grid2DWithYZOverflow`（内部 group dispatch） |
| 派发 / 循环 | `program_id(0/1/2)`，无循环 | split 轴→`program_id(0)` + 三层 SUB_BLOCK 嵌套循环 | `total_thread=40` group dispatch（`for i`）+ y1 轴一层 `for y1inner` |
| permute 处理 | `yindex % ks1` / `// ks1`（运行期 mod/div） | kernel 内显式 `tmp1.permute([0,2,1])`，两输入各备两套索引 | `_maybe_split_fused_axes` 撤销合并，退回原生子轴，无 mod/div、无 permute |
| 轴建模 | y 扁平 + x，2D | SplitTiling 选轴 `split_axis=[0]`/`tiling_axis=[0,1,2]` | 原生 3 轴 `y0/y1/x2`（`npu_num_x_nodes=3`），`tree_node_mapping` 表融合符号→子轴 |
| 签名动态量 | `ks0,ks1` + `ynumel,xnumel` | `ks0,ks1` + 每轴 `*_numel`（无 divisor） | `ks0,ks1`(i64) + 扁平 `ynumel/xnumel` + 每轴 numel **+ `y1divisor`** |
| 动态 divisor | 无（mod/div 内联） | 无（permute kernel 内显式） | `y1divisor` 进签名；`real_block_y1=YBLOCK//y1divisor`，arange 上界用 `divisor_hint=128` 估 |
| 内层循环 | 无 | `Y1/Y0/X2BLOCK_SUB` 三层 | 外层 group `for i` + `y1` 轴 `for y1inner`（y0/x2 divisor=1 无内层） |

要点：① **三种 permute 处理各异**——GPU 运行期 mod/div 还原；内置保留多维 + kernel 内显式转置；本后端撤销轴合并退回原生子轴的仿射组合，无 mod/div 无 permute。② **派发三分**——GPU 多维 grid 无循环；内置多维 grid + 三层 SUB_BLOCK；本后端固定 40-CU group + 必要时一层 inner loop。③ **动态参数流**——只有本后端额外传 `y1divisor`（permute 维保持原生子轴后带动态步长）。

---

## 二、算子级实测对标（torchbench + test_all + 京东 OneRec）

### 2.1 总体结论（torchbench 34 模型，OP = compile/eager 算子加速比）

| 指标 | npu_inductor | dvm | 说明 |
|---|---|---|---|
| **OP 几何平均** | **1.30** | 1.15 | 生成 kernel 算子级整体更快 |
| **OP 占优模型数（共 32 可比）** | **22** | 10 | 逐模型比，npu_inductor 22/32 更高 |
| **OP > 1.0 模型数** | 30 / 34 | — | 30/34 模型算子比 eager 快 |
| **精度（34 模型）** | 30 通过 / **4 失败** | 全通过 | 失败：`hf_Bart`、`hf_T5_base`、`hf_T5_large`、`soft_actor_critic` |

读数：视觉模型优势明显（`resnet152` 1.91 vs 1.00、`resnet50` 1.60 vs 1.00、`densenet121` 1.59 vs 1.01、`timm_regnet` 2.42 vs 2.03）；少数 dvm 更优（`mobilenet_v2` 2.26 vs 1.85、`hf_Albert` 1.35 vs 1.21）；4 个精度缺口是当前主要短板。

### 2.2 逐模型算子加速比（节选；OP=compile/eager，>1 更快）

| 模型 | NI OP | dvm OP | | 模型 | NI OP | dvm OP |
|---|---|---|---|---|---|---|
| BERT_pytorch | 1.72 | 1.35 | | resnet152 | 1.91 | 1.00 |
| densenet121 | 1.59 | 1.01 | | resnet50 | 1.60 | 1.00 |
| hf_Albert | 1.21 | 1.35 | | resnext50_32x4d | 1.40 | 1.00 |
| hf_Bert | 1.07 | 1.08 | | shufflenet_v2_x1_0 | 1.89 | 1.75 |
| hf_T5_base ✗ | 1.15 | 1.05 | | speech_transformer | 1.25 | 0.96 |
| hf_T5_large ✗ | 1.39 | — | | timm_regnet | 2.42 | 2.03 |
| LearningToPaint | 1.50 | 1.43 | | timm_vovnet | 1.42 | 1.44 |
| mobilenet_v2 | 1.85 | 2.26 | | torch_multimodal_clip | 1.17 | 1.07 |
| phlippe_densenet | 1.91 | 1.61 | | resnet18 | 1.69 | 1.73 |

> 动态 shape（`dynamic=True`，基于 2.7.1 旧数据，仅供趋势）：33 模型 OP 几何平均 **1.19**，视觉模型仍优（resnet152 1.85、timm_regnet 1.91、mobilenet_v2 1.70）；不与上表横比。

### 2.3 京东 OneRec 真实客户场景（生成式推荐训练吞吐，全单轴动态 shape）

四列对比 eager / dvm / 本项目 triton / **torch_npu 当前 triton（内置 default 后端）**；吞吐越大越好。

| backbone | eager | dvm | triton（本项目） | torch_npu 当前 triton | triton/eager | triton/dvm |
|---|---|---|---|---|---|---|
| T5 dense | 588.13 | 673.10 | **692.17** | Runtime Error | 1.18 | 1.03 |
| T5 fused MoE | 543.84 | 637.07 | **646.41** | Runtime Error | 1.19 | 1.01 |
| qwen3 0.6B | 80.63 | 110.43 | **114.87** | Compile Error | 1.43 | 1.04 |
| qwen3 4B | 17.44 | 21.02 | **21.53** | Runtime Error | 1.24 | 1.02 |
| **几何平均** | — | — | — | — | **1.25** | **1.03** |

要点：本项目 triton **4 个 backbone 吞吐全部第一**（vs eager 几何 1.25×，vs dvm ×1.03）；**内置 default Triton-Ascend 后端 4 个全跑不通**（3 Runtime Error + 1 Compile Error）——在这组真实客户单轴动态 shape 训练场景下，本后端是**唯一跑通**的 Triton 方案。

### 2.4 算子 case 级（test_all 单算子，设备 kernel 耗时，dvm/NI>1 = 本后端更快）

**总体 60 可比用例**：几何平均 dvm/NI = **1.27**（本后端设备耗时整体低约 27%），逐用例本后端更快 **38/60**。

- **视图重排融合优势最大**（Linearize + 索引化简 = 撤销轴合并、消除 mod/div 的主战场）：`clip_qkv_bias_grad_sum` 8.83×、`select_scatter_qkv_bw` 6.24×、`masked_fill_softmax_bw` 5.90×、`var_mean_norm` 5.42×、`row_bcast_pointwise_dyn` 4.72×、`slice_scatter_permute_pw` 4.70×、`permute_31204` 4.63×、`residual_layernorm_dyn` 4.44×、`concat_permute_pw`/`add_mul_slice_permute_cat`(+bw) 2.4–2.5×。
- **dvm 占优集中在 reduction / 归一化反向**（本后端优化方向）：`bn_backward_reduce` 0.28×、`bn_backward_reduction` 0.30×、`clip_ln_bw_sum_transpose` 0.41×、`sum_reduce0_1d` 0.47×、`view_sum_bcast_dyn` 0.55×、`bart_ln_bw_dual_sum_512` 0.56×、`softmax_dyn` 0.57×、`view_sum_dyn` 0.62×，及 `gelu`/`gelu_backward` 小幅领先。

---

## 三、逐维综合矩阵

| 维度 | GPU 上游 | torch_npu 内置 default | 本后端 `npu_inductor_2.9.0` |
|---|---|---|---|
| 后端注册 | `CUDACombinedScheduling`（Triton + CUTLASS） | `NPUCombinedScheduling`（CATLASS / Split-Tiling / 非线性） | `NPUTritonScheduling`（单一 Triton；关掉内置） |
| 形态/规模 | upstream 框架 | 独立框架 127 文件、三后端 | monkey-patch ~9 千行 / 9 文件 |
| 多维 kernel | 多维 grid + program_id | Split-Tiling + SUB_BLOCK 循环 + GridNpu | Linearize + 40-CU group dispatch（四遍折叠） |
| 动态 shape | 多维 grid 重算；`ks*` 升 i64 防溢出 | gears/`NPUShapeHandling` 分桶，编译 N 次 | 运行期 numel/divisor + 编译 1 次；降 i32（溢出风险） |
| persistent reduction | 可选（阈值） | 支持（UB 塞满单核） | **恒关**（looped + r 轴 rsplit） |
| 长 reduction 跨核 | split / cooperative（semaphore barrier） | split_reductions / cooperative | r 轴 rsplit 两-kernel（无 barrier） |
| GEMM | CUTLASS | **CATLASS / CK** + EVG epilogue | **无 codegen**，mm 走 CANN |
| 融合 | Scheduler 全套（proximity 64） | 自定义 can_fuse（proximity 收 20） | 复用上游 + `NPU_MAX_FUSED_READS` 门控 |

> 设计取舍：内置后端覆盖面全（GEMM/persistent/多后端/Ascend950），是通用框架；本后端代码轻、单一路径、动态 shape 编译一次、在 view/permute 重排类场景优势明显（§2.4）、逐个真实模型精修，且在真实客户单轴动态 shape 训练上目前唯一跑通（§2.3）。

---

## Related Pages

- [[23_npu_inductor_linearize_backend_analysis]] — 本后端机制总览（架构 + Linearize + 融合 + rsplit + 优化点）
- [[24_npu_inductor_linearize_dynamic_shape_analysis]] — 本后端动态 shape（三情形 + permute 产物机制）
- [[11_npu_inductor_splittiling_backend_analysis]] — torch_npu 内置 Triton/Split-Tiling 路径（§1.2 内置后端细节）
- [[01_npu_compile_paths_overview]] — 内置三路径全景 + §九 GPU vs NPU 动态 shape
- [[23_inductor_gpu_kernel_dispatch_model]] — GPU 上游派发模型（§1.1 GPU 侧基线）
- [[22_inductor_reduction_codegen_deep_analysis]] — 上游 reduction（§2.4 短板对应的 cooperative/split 基线）
- [[02_compile_stack/04_inductor/npu/index]] — NPU Inductor 后端目录索引
