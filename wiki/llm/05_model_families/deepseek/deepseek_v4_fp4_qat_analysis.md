# DeepSeek-V4 FP4 量化感知训练（QAT）

> **来源**: `raw/05_model_families/deepseek/DeepSeek_V4.pdf` §3.7 + `raw/05_model_families/deepseek/DeepSeek_V4_FP4_QAT.md`  
> **创建日期**: 2026-04-29

---

## 概述

DeepSeek-V4 在**后训练阶段**引入 FP4 量化感知训练（QAT），使模型适应量化精度损失，在保持模型精度的同时实现推理加速和内存节省。这是支持 1M token 超长上下文的关键技术之一。

---

## 量化对象

QAT 针对三个组件进行 FP4 量化：

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

| 组件 | 加速比 | 内存节省 | 精度保持 |
|------|--------|---------|---------|
| MoE Expert Weights | ~2× | ~75% | 无损反量化 |
| CSA QK Path | ~2× | ~75% | 保持计算精度 |
| Index Scores | 2× | ~50% | 99.7% recall |

**整体效果**：推理 2× 加速，专家权重 75% 内存减少，1M token 上下文变得可行，无损反量化和高召回率保证模型性能。

---

## 相关页面

- [[deepseek_v4_analysis]] — V4 整体架构
- [[deepseek_v3_analysis]] — V3 FP8 混合精度训练
- [[deepseek_v4_cp_analysis]] — Context Parallelism 分析
- [[mHC]] — 流形约束超连接
