# verl 中 Megatron + vLLM 权重同步分析

本文档分析了 `verl` 框架中训练模型（Megatron-LM）与推理模型（vLLM）之间的权重同步过程，特别关注 **共集群（Colocation）** 场景（即 Actor 和 Rollout 共享同一组 GPU）。

## 1. 概述

在 Megatron + vLLM 组合中，权重同步遵循 **"Gather-Broadcast-Load"（聚合-广播-加载）** 模式。

由于 Megatron-LM 和 vLLM 可能使用不同的内部内存布局或并行策略，`verl` 采用了一种通用方法：**Megatron 将分片权重重构为标准的 HuggingFace 格式（完整张量），流式传输给 vLLM，然后 vLLM 根据其自身配置重新进行分片。**

**触发点：**
同步由 `verl/workers/megatron_workers.py` 中的 `ActorRolloutRefWorker` 类编排。具体来说，它发生在 `rollout_mode()` 方法内部，该方法在生成开始之前被调用。

## 2. 详细调用栈与时序

下面的时序图将高层逻辑映射到代码库中的具体类和函数。

```mermaid
sequenceDiagram
    participant Worker as ActorRolloutRefWorker<br/>(megatron_workers.py)
    participant Utils as megatron_utils<br/>(per_tensor_generator)
    participant Comm as NCCL Comm<br/>(torch.distributed)
    participant Rollout as vLLMRollout<br/>(vllm_rollout_spmd.py)
    participant vLLM_Lib as vLLM Library<br/>(LLMEngine/Model)

    Note over Worker: 入口点: generate_sequences()

    Worker->>Worker: rollout_mode()
    Note right of Worker: 上下文切换开始

    Worker->>Worker: aggressive_empty_cache()

    rect rgb(240, 248, 255)
        Note right of Worker: 1. 创建生成器 (Lazy)
        Worker->>Utils: per_tensor_generator(actor_module, ...)
        Utils-->>Worker: 返回 generator 对象
    end

    rect rgb(255, 250, 240)
        Note right of Worker: 2. 触发更新
        Worker->>Rollout: update_weights(generator)
    end

    rect rgb(230, 230, 250)
        Note right of Rollout: 3. vLLM 消费生成器
        Rollout->>vLLM_Lib: model.load_weights(generator)

        loop 在 vLLM load_weights 循环中 (迭代生成器)
            vLLM_Lib->>Utils: next(generator)
            activate Utils

            Note right of Utils: 3.1 元数据同步
            Utils->>Comm: all_gather_object (PP Group)

            Note right of Utils: 3.2 重构 (聚合 & 广播)
            Utils->>Comm: broadcast_object_list (Name)
            Utils->>Comm: broadcast (从 Owner PP 处广播 Tensor)

            alt Tensor 是 TP 分片的
                Utils->>Comm: all_gather (TP Group)
                Utils->>Utils: default_tp_concat_fn (拼接分片)
            end

            Utils-->>vLLM_Lib: yield (name, full_tensor)
            deactivate Utils

            Note right of vLLM_Lib: 3.3 加载 & 重新分片
            vLLM_Lib->>vLLM_Lib: 处理 & 拷贝到 GPU (为 TP 进行切片)
        end
    end

    Note over Worker: 阶段 4: 恢复
    Worker->>Rollout: resume(tags=['kv_cache'])
    Worker-->>Worker: 返回 generate_sequences
```

## 3. 代码调用栈分析

以下是权重同步过程的分步跟踪：

### 步骤 1: 触发同步
**位置:** `verl/workers/megatron_workers.py`
**类:** `ActorRolloutRefWorker`
**方法:** `generate_sequences` -> `rollout_mode`

当 PPO 训练循环请求 rollout 时，`generate_sequences` 被调用。它显式地切换上下文：

```python
# verl/workers/megatron_workers.py

def generate_sequences(self, prompts: DataProto):
    # ...
    if self._is_actor:
        # 在生成之前显式切换到 rollout 模式
        loop.run_until_complete(self.rollout_mode())
    # ...
```

### 步骤 2: 生成器创建
**位置:** `rollout_mode` 内部的 `verl/workers/megatron_workers.py`
**函数:** `verl.utils.megatron_utils.per_tensor_generator`

Worker 不会立即获取所有权重。相反，它创建一个生成器。这对显存效率至关重要。

```python
# verl/workers/megatron_workers.py

async def rollout_mode(self):
    # ...
    # 创建生成器。尚未发生通信。
    per_tensor_param = per_tensor_generator(
        self.actor.actor_module,
        self.actor_model_config,
        # ...
    )

    # 将生成器传递给 rollout worker
    await self.rollout.update_weights(per_tensor_param)
```

### 步骤 3: 驱动生成器 (vLLM 侧)
**位置:** `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`
**类:** `vLLMRollout` (或 `vLLMAsyncRollout`)
**方法:** `update_weights`

Rollout worker 接收生成器并将其直接传递给底层的 vLLM 引擎。

```python
# verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py

async def update_weights(self, weights: Generator, ...):
    # ...
    # vLLM 的 model.load_weights() 迭代 'weights' 生成器。
    # 这个迭代触发了执行通信的 'next()' 调用。
    model.load_weights(weights)
```

### 步骤 4: 通信循环 (生成器内部)
**位置:** `verl/utils/megatron_utils.py`
**函数:** `per_tensor_generator`

这是实际工作发生的地方，由 vLLM 的迭代惰性触发。

1.  **元数据同步:** `torch.distributed.all_gather_object` 确保所有 rank 知道参数的顺序。
2.  **PP 广播:** `broadcast_from_megatron_pp` 将张量从拥有它的流水线阶段（Pipeline Stage）移动到所有其他阶段。
3.  **TP 聚合:** `torch.distributed.all_gather` 收集张量模型并行分片（例如，切分的线性层），`default_tp_concat_fn` 将它们合并成单个 `Full Tensor`（完整张量）。

## 4. 效率与并发分析

### 效率：低到中
效率很大程度上取决于模型大小和网络带宽（NVLink vs PCIe）。

1.  **冗余通信 (双重通信):**
    *   **Megatron 侧:** `TP Gather` -> `PP Broadcast`。这意味着每个参数最终都会作为完整副本复制到**每一张 GPU** 上（逐层流式传输）。
    *   **vLLM 侧:** 接收到 Full Tensor 后，vLLM（如果启用了 TP）会丢弃大部分数据，只保留其特定的分片。
    *   **浪费:** 对于 $TP=8$，每个 GPU 接收 $8/8$ 的权重，但只保留 $1/8$。与理想的 P2P 传输相比，带宽使用量实际上膨胀了约 $TP \times PP$ 倍。

2.  **串行阻塞:**
    *   生成器按顺序 yield 项目。
    *   PyTorch 中的网络操作（All-Gather, Broadcast）默认是阻塞/同步的。
    *   在此过程中，计算（推理/训练）完全停止。

### 并发与掩盖
**结论：在权重同步期间，计算和通信之间几乎没有有效的重叠掩盖。**

1.  **Stop-the-World:** 在 `update_weights` 期间，Actor 被阻塞。
2.  **无流水线:** 代码缺乏显式的流水线逻辑（例如，在传输第 $N$ 层时预处理第 $N+1$ 层的元数据）。它遵循严格的 `Gather -> Broadcast -> Yield -> Load` 串行循环。
3.  **LoRA 优化 (特例):**
    *   如果使用 LoRA 且 `base_sync_done=True`，该过程会**跳过**这个繁重的参数同步，仅传输微小的 Adapter 权重。这是当前架构中主要的效率优化点。

## 5. 总结

当前的 Megatron + vLLM 同步实现优先考虑 **工程兼容性和正确性**，而不是原始性能。

*   **优点:** 解耦了 Megatron 的并行策略（TP/PP/DP）与 vLLM 的并行策略（TP）。它们不需要匹配的拓扑结构。
*   **缺点:** 通信开销高（Megatron 分片 -> 完整张量 -> vLLM 分片），导致全量微调时 `rollout_mode` 切换出现显著延迟。
*   **未来改进:** 如果拓扑匹配，直接 P2P 传输或共享内存方法（避免完整张量重构）将大幅提高性能。

## Related Pages

- [[llm/06_infra/megatron-lm/overview]]
- [[torch_compile/overview]]
