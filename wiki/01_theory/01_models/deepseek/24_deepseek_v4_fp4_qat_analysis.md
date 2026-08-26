# DeepSeek-V4 FP4 量化感知训练（QAT）

> **核对基线**: arXiv:**2606.19348v1** (DeepSeek-AI, **2026-04-26**) **§5.2.1**「FP4 Quantization-Aware Training」（§5.2 Post-Training Infrastructures）＝ `raw/01_theory/01_models/deepseek/DeepSeek_V4.md`  
> **创建**: 2026-04-29（预发布稿）　**对正式版核对/订正**: 2026-06-25
>
> [!note] **出处订正**：原稿标注「§3.7」有误（预发布稿中 FP4 为 §3.4，正式版**移入后训练 §5.2.1**；§3.7 两版皆无）。
> FP4-QAT 是**后训练**技术。详见 [[30_deepseek_v4_audit_analysis]]。

---

## 概述

DeepSeek-V4 在**后训练阶段**引入 FP4 量化感知训练（QAT），使模型适应量化精度损失，在保持模型精度的同时实现推理加速和内存节省。这是支持 1M token 超长上下文的关键技术之一。

---

## 量化对象

> [!important] **口径订正**：正式版 §5.2.1 中，**FP4 仅作用于两个组件**——(1) MoE 专家权重、(2) CSA indexer 的 QK path。
> **第 3 项「索引分数 I」是 FP32→BF16，并非 FP4**。下文标题保留三项以便对照，但请注意第 3 项的精度是 BF16。

FP4 量化对象（**2 项 FP4 + 1 项 BF16**）：

### 1. MoE Expert Weights

MoE 专家权重是 GPU 内存占用的主要来源。

```
FP32 master weights (optimizer)
    ↓ quantize to FP4
FP4 weights
    ↓ dequantize to FP8 (lossless)
FP8 computation (forward pass)
```

**关键特性**：
- **无损 FP4→FP8 反量化**：FP8 (E4M3) 比 FP4 (E2M1) 多 2 个指数位，动态范围更大
- **细粒度信息吸收**：只要 FP4 子块（1×32 tiles）的缩放因子比率在阈值内，细粒度缩放信息可被 FP8 的扩展动态范围完全吸收
- **无需修改训练框架**：可完全复用现有 FP8 训练框架

### 2. CSA Indexer 的 QK Path

- QK activations 在缓存、加载和乘法运算中完全在 FP4 中进行
- 加速长上下文场景下的注意力分数计算
- 减少内存带宽占用

### 3. Index Scores

- 索引分数 $I_{t,s}$ 从 FP32 量化到 BF16
- **2× 速度提升**：top-k 选择器加速
- **99.7% recall rate**：KV 条目召回率

---

## 训练过程

### 前向传播

```python
# 1. FP32 主权重
master_weights = optimizer.param_groups[0]['params']  # FP32

# 2. 量化到 FP4
quantized_weights = quantize_to_fp4(master_weights)

# 3. 反量化到 FP8（无损）
fp8_weights = dequantize_to_fp8(quantized_weights)  # lossless!

# 4. FP8 计算
output = forward_pass(fp8_weights, input)
```

前向传播使用**模拟量化**（simulated quantization），量化/反量化过程可微。

### 反向传播

```python
# 梯度直接传播回 FP32 主权重 (STE)
grad_fp8 = compute_gradient(loss, fp8_weights)
grad_master = grad_fp8  # Straight-Through Estimator

# 更新 FP32 主权重
optimizer.step()
```

- 等价于 Straight-Through Estimator (STE)
- 无需重新量化转置权重，避免额外计算开销

### 量化策略对比

| 阶段 | 权重精度 | 激活精度 | 说明 |
|------|---------|---------|------|
| 训练（前向） | FP32 → FP4 → FP8 | FP8 | 模拟量化 |
| 训练（反向） | FP32 | FP8 | STE 梯度回传 |
| 推理/RL rollout | FP4 | FP4 | 实际量化部署 |

---

## 无损反量化原理

| 格式 | 指数位 | 尾数位 | 动态范围 |
|------|--------|--------|---------|
| FP4 (E2M1) | 2 | 1 | 较小 |
| FP8 (E4M3) | 4 | 3 | 较大 |

FP8 量化块为 128×128 tiles，FP4 子块为 1×32 tiles。FP8 的扩展动态范围可吸收 FP4 子块的细粒度缩放信息——只要缩放因子比率在阈值内，反量化即无损。DeepSeek-V4 实证验证了当前权重满足此条件。

---

## 效果总结

> [!note] **数据来源标注**：正式版 §5.2.1 **明确给出的只有**「索引分数 top-k 选择器 **2× 加速、99.7% KV 召回**」。
> 下表 MoE / CSA-QK 的「~2×、~75%」是按 FP4(4bit) vs FP8/BF16 的**推断/估算**，论文未直接给出，仅供量级参考。

| 组件 | 加速比 | 内存节省 | 精度保持 | 数据来源 |
|------|--------|---------|---------|---------|
| MoE Expert Weights | ~2×（估算） | ~75%（估算） | 无损 FP4→FP8 反量化 | 推断 |
| CSA QK Path | ~2×（估算） | ~75%（估算） | 保持计算精度 | 推断 |
| Index Scores (FP32→**BF16**) | **2×** | ~50%（估算） | **99.7% recall** | §5.2.1 论文明述 |

**整体效果**：FP4-QAT 在后训练阶段为部署换取推理加速与内存节省；论文确证项为 top-k 选择器 2× 加速 + 99.7% 召回，
其余为量级估算。

---

## 相关页面

- [[13_deepseek_v4_analysis]] — V4 整体架构
- [[12_deepseek_v3_analysis]] — V3 FP8 混合精度训练
- [[23_deepseek_v4_cp_analysis]] — Context Parallelism 分析
- [[25_mhc_analysis]] — 流形约束超连接
