# Mooncake: 以KVCache为中心的分离式LLM服务架构

> **论文信息**: Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving
> **作者**: Ruoyu Qin, Zheming Li, Weiran He, Mingxing Zhang, Yongwei Wu, Weimin Zheng, Xinran Xu
> **机构**: Moonshot AI & 清华大学
> **arXiv**: 2407.00079 (2024-07)
> **开源**: https://github.com/kvcache-ai/Mooncake

---

## 一、核心问题：KVCache瓶颈与LLM服务挑战

### 1.1 问题背景

Kimi 作为 Model as a Service (MaaS) 平台面临多约束优化问题：

**优化目标**：最大化整体有效吞吐量
**约束条件**：
- **TTFT** (Time To First Token)：$\text{TTFT}_{P90} = 10\times$
- **TBT** (Time Between Tokens)：$\text{TBT}_{P90} = 5\times$

### 1.2 Prefill vs Decoding 的计算特征差异

| 阶段 | 计算特征 | 资源瓶颈 | 优化目标 |
|------|---------|---------|---------|
| **Prefill** | 所有输入token并行 | 计算密集型 | 减少TTFT |
| **Decoding** | 自回归逐token生成 | 内存密集型 | 减少TBT |

**关键洞察**：KVCache调度是LLM服务调度的核心。提升吞吐量有两条路径：
1. 复用KVCache减少计算
2. 最大化batch size提高MFU

但两者都与延迟SLO冲突。

### 1.3 过载场景

- GPU供应受限，用户请求指数增长
- 需要**预测未来负载并提前拒绝**，避免浪费计算

---

## 二、架构设计：三池分离

```mermaid
graph TB
    Client[客户端请求] --> Gateway[API网关]
    Gateway --> Conductor[全局调度器 Conductor]

    subgraph "Prefill Pool"
        P1[Prefill Node 1]
        P2[Prefill Node 2]
    end

    subgraph "Decode Pool"
        D1[Decode Node 1]
        D2[Decode Node 2]
    end

    subgraph "KVCache Pool"
        KV1[CPU+DRAM+SSD Node 1]
        KV2[CPU+DRAM+SSD Node 2]
    end

    Conductor --> P1
    Conductor --> D1
    P1 <-->|RDMA| KV1
    D1 <-->|RDMA| KV1
    P1 -.->|KVCache流式传输| D1
```

| 组件 | 功能 |
|------|------|
| **Conductor** | 全局调度、KVCache分布管理、热点块复制 |
| **Prefill Pool** | 独立预填充节点池，弹性扩展 |
| **Decode Pool** | 独立解码节点池，连续批处理 |
| **KVCache Pool** | CPU+DRAM+SSD 分布式缓存 |
| **Messenger** | 基于(GPUDirect) RDMA的高速KVCache传输 |

---

## 三、关键技术创新

### 3.1 Chunked Pipeline Parallelism (CPP)

**问题**：长上下文(128K-1M token)prefill需要跨节点并行，传统TP跨节点每层需2次RDMA all-reduce，MFU大幅下降。

**CPP方案**：利用decoder-only transformer自回归特性

```
输入: [Chunk 1] [Chunk 2] [Chunk 3] [Chunk 4]
Node 1: ████████░░░░░░░░░░░░░░░░░░░░
Node 2: ░░░░░░░░████████░░░░░░░░░░░░
Node 3: ░░░░░░░░░░░░░░░░████████░░░░
Node 4: ░░░░░░░░░░░░░░░░░░░░░░░░████
```

**优势**：
- 仅在pipeline边界通信，可与计算重叠
- 自适应短/长上下文，无需动态调整分区

### 3.2 Layer-wise Prefill

逐层预填充，KVCache传输与计算重叠：

```
传统:     Layer 1: [Load]→[Compute]→[Store]→Layer 2: [Load]→...
Layer-wise: Layer 1: [Load]→[Compute]→[Store]
            Layer 2:          [Load]→[Compute]→[Store]
            Layer 3:                   [Load]→[Compute]→[Store]
```

**效果**：执行时间 ≈ `max(KVCache加载时间, 标准prefill时间)`

### 3.3 缓存感知的全局调度

```
Algorithm: KVCache-centric Scheduling
1. block_keys ← PrefixHash(R.prompt_tokens, B)
2. FindBestPrefixMatch(P, block_keys)
3. for instance ∈ P:
     if best_prefix_len/prefix_len < threshold:
       // 使用本地cache
       TTFT ← T_queue + T_prefill
     else:
       // 考虑传输远程cache
       TTFT ← T_transfer + T_queue + T_prefill
4. if TTFT > SLO or TBT > SLO: reject R
5. if best_prefix_len/p.prefix_len > threshold:
     TransferKVCache()  // 热点迁移
```

**前缀链式哈希**：
$$\text{Hash}(block_i) = \text{Hash}(block_i \parallel \text{Hash}(block_{i-1}))$$

### 3.4 面向过载的调度

#### Early Rejection

将decode负载评估提前到prefill之前，避免prefill完成后被拒绝浪费计算。

#### 预测性早期拒绝

**问题**：Early Rejection导致prefill/decode负载反相波动

```
Prefill: 高 ──── 低 ──── 高 ──── 低
Decode:  低 ──── 高 ──── 低 ──── 高
```

**方案**：系统级预测
- 假设decode耗时均匀为 $t_d$
- 预测时刻 $t$ 的decode负载
- 计算平均TBT比率预测负载

---

## 四、性能评估

| 场景 | 结果 |
|------|------|
| 模拟场景吞吐量 | 提升高达 **525%** |
| 真实工作负载 | Kimi处理 **75%** 更多请求 |
| 缓存命中率(50K blocks) | ~50% |
| 平均输入/输出比 | ~720:1 |

**调度策略对比**（8 prefill + 8 decode实例，23K真实请求）：

| 策略 | 平均TTFT | SLO达成率 |
|------|---------|----------|
| 随机 | 最高 | 最低 |
| 负载均衡 | 中等 | 中等 |
| 缓存感知 | 较低 | 较高 |
| **KVCache中心（本文）** | **最低** | **最高** |

---

## 五、在Moonshot AI技术路线中的地位

```
Mooncake (2407) → Kimi推理服务底座
    │
    ├── 支撑长上下文 → K2/K2.5 256K+ context
    ├── 分布式KVCache → 多轮对话/文档处理优化
    ├── 过载调度 → 大规模用户支撑
    └── 开源 → https://github.com/kvcache-ai/Mooncake
```

Mooncake是Kimi系列的**推理服务基础设施**，与后续模型能力形成协同：

| Kimi模型能力 | Mooncake支撑 |
|-------------|-------------|
| 更大参数规模 | 分布式KVCache管理 |
| 更长上下文 | CPP多节点prefill |
| 多模态 | 架构可扩展性 |
| 更多用户 | 预测性过载调度 |

---

## 六、关键技术公式

### 调度决策

$$\text{TTFT} = T_{queue} + T_{prefill} + T_{transfer}$$

### 负载均衡阈值

$$\frac{\text{best\_prefix\_len}}{\text{prefix\_len}} < \text{kvcache\_balancing\_threshold}$$

### KVCache占用成本

$$\text{Occupation Cost} = S \times T$$

---

## Related Pages

- [[llm/index]]
- [[llm/06_infra/megatron-lm/index]]
- [[torch_compile/index]]
