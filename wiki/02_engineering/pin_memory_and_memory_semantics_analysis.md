# Pin Memory 与内存语义通信在大模型社区的应用

> **Wiki 说明**：本文内容均来自经过搜索验证的公开技术资料、论文、官方文档及社区 Issue，每条关键技术论断均标注出处。最后更新：2025 年 5 月。

---

## 目录

1. [基础概念](#1-基础概念)
   - 1.1 [Pin Memory（锁页内存）](#11-pin-memory锁页内存)
   - 1.2 [内存语义通信](#12-内存语义通信)
   - 1.3 [两者的关系](#13-两者的关系)
2. [为什么传统消息语义不够用](#2-为什么传统消息语义不够用)
3. [Pin Memory 在大模型社区的应用](#3-pin-memory-在大模型社区的应用)
   - 3.1 [PyTorch：数据管道层](#31-pytorch数据管道层)
   - 3.2 [DeepSpeed：Offload 层](#32-deepspeedoffload-层)
   - 3.3 [vLLM：KV Cache Offloading](#33-vllmkv-cache-offloading)
4. [内存语义通信在大模型社区的应用](#4-内存语义通信在大模型社区的应用)
   - 4.1 [vLLM：Disaggregated Prefill/Decode](#41-vllmdisaggregated-prefilldecodle)
   - 4.2 [Mooncake TransferEngine](#42-mooncake-transferengine)
   - 4.3 [DeepSeek DeepEP：MoE Expert Parallelism](#43-deepseek-deepepmoe-expert-parallelism)
   - 4.4 [DeepSeek 3FS：RDMA Read 驱动的分布式存储](#44-deepseek-3fsrdma-read-驱动的分布式存储)
   - 4.5 [RLHF 框架：训练-推理权重同步](#45-rlhf-框架训练推理权重同步)
   - 4.6 [通信库演进：走向标准化内存语义 API](#46-通信库演进走向标准化内存语义-api)
5. [两种 Pin Memory 层次的区分](#5-两种-pin-memory-层次的区分)
6. [社区应用全景总结](#6-社区应用全景总结)
7. [参考资料](#7-参考资料)

---

## 1. 基础概念

### 1.1 Pin Memory（锁页内存）

正常情况下，操作系统可以将任意内存页换出（swap）到磁盘，物理地址随时可能变化。**Pin Memory**（也称锁页内存、固定内存、page-locked memory）是将某块内存锁定在物理 RAM 中，告诉操作系统这些页面不许换出，物理地址固定不变。

**为什么需要 Pin Memory？**

硬件设备（GPU、网卡、NVMe 控制器）在做 DMA（直接内存访问）时，直接操作物理地址，绕过 CPU。如果传输过程中物理地址发生变化，硬件就会读写到错误的位置，导致数据损坏甚至系统崩溃。Pin Memory 正是为了满足这一 DMA 操作的基本前提。

```
普通内存（Pageable）：
  虚拟地址 → [OS 随时可换出] → 物理地址（不稳定）

Pin Memory（Page-locked）：
  虚拟地址 → 物理地址（固定，OS 不能挪动）
                  ↑
        DMA 引擎可以安全操作
```

**代价**：
- 锁定的页面无法被 OS 回收，占用宝贵物理内存
- Pin/Unpin 本身有开销，需要通知 OS 做页表操作
- 过度使用会造成系统内存压力

---

### 1.2 内存语义通信

传统网络通信是**消息语义（Message Semantics）**：发送方调用 send()，数据经内核协议栈到达网卡；接收方调用 recv()，数据从网卡经内核协议栈到达应用。双方 CPU 都深度参与。

**内存语义通信（Memory Semantics）** 则允许一台机器像访问本地内存一样直接读写远端机器的内存，远端 CPU 不需要参与。其核心原语是：

| 原语 | 语义 |
|------|------|
| **RDMA Write** | 本地机器直接写入远端机器的内存 |
| **RDMA Read** | 本地机器直接读取远端机器的内存 |
| **RDMA Atomic** | 对远端内存执行原子操作（CAS、Fetch&Add） |

底层实现技术主要包括：InfiniBand、RoCE（RDMA over Converged Ethernet）、iWARP，以及 NVLink（节点内 GPU 间）。

---

### 1.3 两者的关系

**Pin Memory 是内存语义通信的必要前提**。在 RDMA 传输开始之前，通信双方必须向 RDMA 网卡注册（register）目标内存区域，这一注册动作的本质就是 Pin Memory——将目标内存的物理地址告知 NIC，NIC 才能在 CPU 不参与的情况下直接 DMA 操作。

```
内存语义通信（RDMA）的工作流程：

① malloc / cudaMalloc 分配内存
          ↓
② 调用 ibv_reg_mr() / nixl_wrapper.register_memory() 注册内存区域
   → 触发 Pin Memory，物理地址固定
   → NIC 获得这块内存的"访问凭证"（lkey/rkey）
          ↓
③ 远端机器用 rkey 发起 RDMA Write/Read，NIC 直接 DMA 操作
          ↓
④ 全程零拷贝，双方 CPU 不参与数据路径
```

---

## 2. 为什么传统消息语义不够用

理解这一问题是理解内存语义通信在大模型中兴起的核心。

**NCCL（集合通信）的根本局限**：NCCL 的集合通信在静态模式（数据并行、张量并行）下表现优秀，但对新兴 LLM 工作负载存在根本性约束：固定成员（fixed membership）阻碍了动态扩缩容；同步初始化增加了协调开销；统一的 buffer 尺寸要求即便在稀疏通信场景（如 MoE）下也强制走密集通信路径。高性能计算领域早已用基于 RDMA 的 SEND/RECV/WRITE 原语实现灵活的低延迟高带宽传输，但这类原语在 LLM 框架中几乎不可用。[^transferengine]

**NCCL P2P 的内存拷贝开销**：在生产环境实测中，NCCL P2P 通信中从应用 buffer 到 chunk buffer 的内存拷贝——这部分与实际数据传输无关——占整个 P2P 过程的近 25%，让内存拷贝开销与真正的数据传输耗时相当，严重拖慢了 LLM 训练中频繁的点对点通信。[^iccl]

**NIC 厂商碎片化**：NVIDIA ConnectX NIC 使用传统的可靠连接（RC）传输（保序），而 AWS EFA 实现了私有的 SRD 协议（乱序交付）。现有方案存在严重的厂商锁定：DeepEP 需要 IBGDA（仅 ConnectX 支持），NVSHMEM 在 EFA 上性能严重降级。这直接催生了统一内存语义抽象层的需求。[^transferengine]

---

## 3. Pin Memory 在大模型社区的应用

### 3.1 PyTorch：数据管道层

#### 3.1.1 DataLoader `pin_memory=True`

这是大模型训练中最普遍的 Pin Memory 使用点。

**原理**：`pin_memory=True` 使 DataLoader 对每个 batch 调用 `pin_memory()`，将 CPU 侧数据锁定在物理页。结合 `non_blocking=True` 使用时，CPU 将当前 batch 向 GPU 发起异步 DMA 传输的同时，可以并行准备下一个 batch，从而避免 GPU 空等。[^pytorch_pinmem]

**内存占用估算**：

```
pinned_memory ≈ num_workers × prefetch_factor × batch_memory_size
示例：4 workers × 2 prefetch × 50MB batch = 400MB pinned memory
```

**NUMA 拓扑注意事项**：在多 socket 服务器上，pinned memory 分配在调用线程所在的 NUMA 节点。若 GPU 与调用线程不在同一 NUMA 节点，DMA 会跨 socket 链路，引入额外延迟。[^pytorch_pinmem]

**重要的反直觉警告**：PyTorch 官方教程（2024 年更新）指出，`tensor.pin_memory().to(device, non_blocking=True)` 有时比直接 `tensor.to(device)` 慢 2 倍。正确的姿势是在 DataLoader 层面预分配 pinned buffer，而非在训练循环中临时 pin。[^pytorch_official]

**何时效果不明显**：当 dataset 很小或 tensor 很小时，pin_memory 效果微乎其微，因为 CPU→GPU 传输本身耗时就不是瓶颈。[^pytorch_mem_pinning]

#### 3.1.2 GNN 训练中的 PyTorch-Direct

针对 GNN 训练的研究提出了 PyTorch-Direct 方案，通过绕过 CPU 侧的内存中转，让 GPU 直接从主机内存中 DMA 取样本，可将整体训练时间减少最高 38.2%；在 LLM 训练中，SSDTrain 框架则通过直接的 GPU-SSD 数据路径将 activation 卸载到 NVMe，将 activation 峰值内存占用减少了最高 47%。两条路径都依赖 Pin Memory 作为 DMA 操作的前提。[^ssdtrain]

---

### 3.2 DeepSpeed：Offload 层

DeepSpeed 是 Pin Memory 在大模型训练中使用最密集、最系统化的框架，体现在 ZeRO 系列的 CPU Offload 机制中。

#### 3.2.1 ZeRO-Offload（ZeRO Stage 2）

ZeRO-Offload 将 optimizer 的内存和计算从 GPU 卸载到 CPU，训练过程中梯度和优化器状态需要在 CPU-GPU 间频繁双向传输。[^deepspeed_zerooffload]

配置示例：
```json
"offload_optimizer": {
    "device": "cpu",
    "pin_memory": true
}
```

**为什么必须 pin_memory**：若使用 pageable 内存，每次 CPU→GPU 传输都会触发临时 staging buffer 的分配与内存拷贝，这不仅增加一次多余的内存复制，更阻塞了计算与传输的异步重叠（overlap），大幅降低效率。

#### 3.2.2 ZeRO-Infinity（ZeRO Stage 3）

ZeRO-Infinity 能将全部模型状态（参数、梯度、优化器状态）卸载到 CPU 或 NVMe，相比 ZeRO-Offload，它有更高效的带宽利用和计算通信重叠能力。[^deepspeed_zero3]

DeepSpeed 配置文件同时暴露了 `offload_param` 和 `offload_optimizer` 两个独立的 `pin_memory` 选项，以及 `non_blocking` 选项控制异步传输。[^deepspeed_config_json]

**ZeRO 各阶段 Pin Memory 使用频率**：

| ZeRO 阶段 | Offload 内容 | Pin Memory 使用频率 |
|----------|------------|------------------|
| Stage 1 | Optimizer states | 中 |
| Stage 2（ZeRO-Offload） | + Gradients | 高 |
| Stage 3（ZeRO-Infinity） | + Parameters | 极高（每个 forward/backward 都触发） |

---

### 3.3 vLLM：KV Cache Offloading

vLLM v1 在推进 KV cache 卸载到 CPU 的过程中，Pin Memory 是 GPU→CPU 异步传输路径的核心前提。

社区 RFC 描述了具体需求：支持异步保存新 KV 数据从 GPU 到 CPU cache，以及异步从 CPU cache 加载 KV 数据回 GPU。[^vllm_kv_offload_rfc] KV cache offloading 将大型 KV cache 从 GPU 显存迁移到 CPU 或磁盘，使得更多 KV cache 命中成为可能。[^vllm_kv_offload_tutorial] CPU 侧的接收/发送 buffer 必须是 pinned memory，才能让 GPU 侧的异步 DMA 传输正确且高效地完成。

---

## 4. 内存语义通信在大模型社区的应用

### 4.1 vLLM：Disaggregated Prefill/Decode

P/D 分离（prefill 和 decode 运行在不同 vLLM 实例上）是 vLLM 社区 2024-2025 年最重要的架构演进，也是内存语义通信的主战场。

**架构概述**：prefiller（producer）将 prompt 跑完模型得到 KV cache block，然后通过网络将这些 block 传输到 decoder（consumer），后者直接从 token 生成阶段开始，无需重新计算 prefill。[^vllm_disagg_deepwiki]

**为什么不用 NCCL**：早期用 cuPy（NCCL 的 Python 绑定）做多节点 P/D 分离时，性能很差且故障域很大，根本原因在于 NCCL 的集合通信模式本就不适合点对点 KV 传输。[^mooncake_acm]

**NixlConnector**：vLLM 当前主要的 KV 传输 connector。KV block 通过 RDMA 在 GPU VRAM 之间直接传输，尽可能绕过 CPU；decode worker 在传输开始前调用 `nixl_wrapper.register_memory()` 注册 KV cache 所在内存区域，并生成握手元数据用于与远端 worker 协商传输。[^vllm_disagg_deepwiki]

**部署前提**：`IPC_LOCK` 权限是 RDMA pinned memory 的必要条件；缺少该权限会导致 NIXL 传输直接失败。[^llmd_pd]

**vLLM 目前支持的 KV Connector 对比**：

| Connector | 传输机制 | 主要场景 |
|-----------|---------|---------|
| NixlConnector | RDMA/UCX（GPU→GPU） | 跨节点 P/D 分离 |
| MooncakeConnector | RDMA/NVLink/TCP 统一 API | P/D 分离 + 多级 KV 缓存 |
| LMCacheConnector | NIXL 作为传输层 | 分布式 KV cache 管理 |
| OffloadingConnector | GPU→CPU pinned memory | 单机 KV offload |

---

### 4.2 Mooncake TransferEngine

Mooncake 是 Moonshot AI（Kimi）的 LLM 服务平台，其核心技术贡献 Transfer Engine 已成为业界广泛使用的内存语义通信库。

**发展历程**：
- 2024 年 6 月：Mooncake 技术报告发布
- 2024 年 11 月：Transfer Engine 开源
- 2024 年 12 月：vLLM 官方支持 Mooncake Transfer Engine 用于 disaggregated prefilling
- 2025 年 2 月：Mooncake 获 FAST 2025 最佳论文奖
- 2025 年 12 月：Transfer Engine 直接集成进 vLLM v1 和 TensorRT-LLM [^mooncake_readme]

**技术核心**：Transfer Engine 的核心价值在于将 RDMA、NVLink、TCP、NVMe-oF 等协议统一到一个传输语义 API 之下，屏蔽底层协议差异。[^mooncake_acm] 这正是内存语义通信"像访问本地内存一样操作远端内存"理念的工程实现。

**集成广度**（截至 2025 年）：vLLM、SGLang、TensorRT-LLM、Huawei Ascend vLLM 均已集成。[^mooncake_docs]

---

### 4.3 DeepSeek DeepEP：MoE Expert Parallelism

DeepEP 是 DeepSeek 在 2025 年 2 月开源周发布的第二个项目，是业界首个专为 MoE 模型训练和推理设计的 Expert Parallelism 通信库。

**定位**：DeepEP（DeepEveryParallel）面向现代机器学习训练和推理，聚焦于 expert parallelism（EP），提供高吞吐、低延迟的全对全 GPU 通信内核（MoE dispatch 和 combine），支持包括 FP8 在内的低精度，同时提供 pipeline parallelism、context parallelism 以及远程内存访问（Engram）的实验性原语，所有内核设计目标均为零或极少 SM 占用。[^deepep_github]

**两种内核模式**：

| 内核类型 | 通信机制 | 适用阶段 | 特点 |
|---------|---------|---------|------|
| High-Throughput (HT) 内核 | NVLink + RDMA 混合 | 训练、推理 prefill | 最大化带宽 |
| Low-Latency (LL) 内核 | 纯 RDMA | 推理 decode | 微秒级延迟，不占 SM |

**内存语义的体现**：DeepEP 的低延迟内核基于纯 RDMA 模式，通过独特的 hook-based 通信-计算重叠方法实现出色的并行效率，不占用 SM 资源。[^deepep_org] 在底层，DeepEP 是 GPU-Initiated Networking（GIN）的典型实现，通过 NVSHMEM 配合 InfiniBand GPU Direct Async（IBGDA），将 CUDA kernel 与 GPUDirect 通信融合，CUDA kernel 可以直接发起 put/get 操作（内存语义的 RDMA Write/Read），无需将控制权交还给 host CPU。[^nvshmem_blog]

NVSHMEM 的核心正是内存语义通信在 GPU 侧的实现：在 NVLink、PCIe 和 InfiniBand 网络上实现了 symmetric-heap 上的单侧 GPU-initiated RMA（远程内存访问）操作，通过系统全局可见的 "signal" 计数器完成设备侧同步。[^gin_emergentmind]

**云环境局限**：DeepEP 的 GPU-driven 通信方案与特定 NIC 紧耦合，无法直接在公有云的异构平台上运行。这促使了 UCCL-EP 的出现——它与 DeepEP 接口和功能完全相同，但能在 AWS EFA 等公有云环境运行。[^uccl_ep]

---

### 4.4 DeepSeek 3FS：RDMA Read 驱动的分布式存储

3FS（Fire-Flyer File System）是 DeepSeek 在 2025 年 2 月开源周发布的第五个项目，是专为 AI 训练和推理设计的分布式文件系统。

**设计理念**：3FS 采用 Direct I/O 和 RDMA Read，与标准的 Buffered I/O 形成鲜明对比，摒弃了传统文件系统的 page cache 模式。[^deepseek_wikipedia]

**架构**：3FS 的四个组件（cluster manager、metadata service、storage service、client）全部通过 RDMA 网络（InfiniBand 或 RoCE）连接。客户端直接发起 RDMA Read 到存储节点，存储节点的内存被直接读取，整个集群形成一个大型内存语义存储池。[^3fs_design_notes]

**性能数据**：在 180 个存储节点的压测中，3FS 集群的聚合读吞吐达到 6.6 TiB/s，单节点 KVCache 查找峰值超过 40 GiB/s。[^3fs_aibase]

**在 LLM 中的应用**：
- 训练阶段：从多个存储节点并行 RDMA Read 训练数据，持续喂给计算节点
- 推理阶段：高效加载模型权重、服务 KV cache 的分布式存储

---

### 4.5 RLHF 框架：训练-推理权重同步

这是 2025 年内存语义通信在大模型社区增长最快的新兴场景。

**问题背景**：RLHF 的典型流程中，训练集群更新完权重后，需要把新权重同步给推理集群（rollout engine），让其生成新的样本。当模型规模到达千亿参数量级时，权重同步的带宽需求极高，同步延迟直接卡住整个训练迭代。

**使用内存语义的实践**：TransferEngine 将 RL 权重更新实现为：每个训练 GPU 通过单侧 RDMA Write 直接写到推理 GPU，利用整个集群的全部带宽；流水线化执行将 H2D memcpy、权重准备和 RDMA 传输三者重叠，对万亿参数模型仅需 1.3 秒完成权重同步，比已有 RL 框架快 100 倍以上。[^transferengine_rl]

**veRL / OpenRLHF 的现状**：在 veRL 和 OpenRLHF 等框架中，同节点内训练 actor 与 rollout 推理引擎的权重同步通过 NCCL 或 CUDA IPC 实现（消息语义）；当采用跨节点 disaggregated 模式时，走网络传输，RDMA 的带宽直接影响同步延迟。[^verl_rl_frameworks]

**异步 RL 的量化对比**：在异步 RL 训练实验中，通过 Mooncake 进行权重传输时，RDMA（400 Gbps InfiniBand）相比 TCP（200 Gbps 以太网）提供了明显更高的带宽和更低的通信开销；在同步 RL 训练模式下，权重传输延迟直接卡住 rollout 阶段，低带宽链路会大幅增加端到端训练时间。[^rollart]

---

### 4.6 通信库演进：走向标准化内存语义 API

大模型社区正在推动内存语义通信从"特定硬件的黑魔法"走向"可移植的标准化 API 层"。

#### NCCL 的演进

NCCL 2.27 引入了 symmetric memory 支持，允许在所有 GPU 上虚拟地址相同的 buffer 使用优化内核，对小消息尺寸可将延迟降低最高 7.6 倍；同期引入的 Direct NIC 支持，使 GPU scale-out 通信可以绕过 CPU 瓶颈获得完整网络带宽，对高吞吐推理和训练工作负载尤为重要。[^nccl_227]

Meta 的 NCCLX 在 NCCL 基础上进一步提供了三类 API：Host-initiated（传统集合通信）、Host-initiated with GPU-resident metadata，以及 Device-initiated API（内存语义方向）。[^ncclx_meta] Device-initiated API 意味着 GPU kernel 可以直接发起通信操作，这正是内存语义通信的核心特征。

#### SHIFT：内存语义通信的容错保障

SHIFT 是一个用户态 RDMA 层，它观察到主流训练框架（如 NCCL）依赖的是幂等批量传输，在保证通知顺序的前提下可以容忍宽松的内存顺序；基于此，SHIFT 在 rdma-core 中实现了跨 NIC 的容错机制，在 PyTorch 分布式训练评测中显示出几乎零开销，并能成功屏蔽致命的 NIC 失效和链路异常。[^shift]

#### TransferEngine / NIXL 的定位

面对 NVIDIA ConnectX（保序 RC 传输）和 AWS EFA（乱序 SRD 协议）的碎片化现实，TransferEngine（来自 Cloudflare Research）通过桥接常见 NIC 功能暴露统一接口，使用单侧 WriteImm 操作配合 ImmCounter 原语进行完成通知，无需假设网络传输的顺序，透明管理每个 GPU 的多个 NIC，在 NVIDIA ConnectX-7 和 AWS EFA 上均实现了 400 Gbps 峰值吞吐。[^transferengine]

---

## 5. 两种 Pin Memory 层次的区分

大模型场景中存在两个层次的 Pin Memory，概念上容易混淆：

```
层次 1：CPU 侧 Pin Memory（CPU DRAM 锁页）
  ┌─────────────────────────────────────────┐
  │  API：cudaHostAlloc() / mlock()         │
  │  作用：锁定 CPU 物理页，让 GPU 的        │
  │       PCIe DMA 引擎可以直接操作         │
  │  典型场景：DataLoader、ZeRO Offload     │
  └─────────────────────────────────────────┘

层次 2：GPU 显存 Registration（GPU HBM 注册）
  ┌─────────────────────────────────────────┐
  │  API：ibv_reg_mr() /                    │
  │       nixl_wrapper.register_memory()    │
  │  作用：向 RDMA NIC 注册 GPU 显存的      │
  │       BAR 物理地址，NIC 可直接 DMA 操作  │
  │  典型场景：NIXL KV Transfer、DeepEP、   │
  │           NCCL 跨节点通信               │
  └─────────────────────────────────────────┘
```

两者在概念上都是"固定物理地址以供硬件 DMA"，但一个作用在 CPU DRAM，另一个作用在 GPU HBM。在实际工程中，一个常见的折中是：为通信保留少量 pinned buffer，数据在发送/接收前做一次 CPU 侧拷贝，以避免大量内存 pin/unpin 的开销。[^np_rdma]

---

## 6. 社区应用全景总结

### Pin Memory 应用总结

| 框架 | 具体场景 | Pin Memory 类型 | 配置方式 |
|------|---------|--------------|---------|
| PyTorch DataLoader | 训练数据 CPU→GPU | CPU pageable→pinned | `pin_memory=True` |
| PyTorch 手动传输 | 临时异步传输 | CPU pinned buffer | `tensor.pin_memory()` |
| DeepSpeed ZeRO-2 | Optimizer state offload | CPU pinned（双向） | JSON `"pin_memory": true` |
| DeepSpeed ZeRO-3 | 参数 prefetch/swap | CPU pinned（双向） | JSON `"pin_memory": true` |
| vLLM KV Offload | KV cache CPU offload | CPU pinned | OffloadingConnector |
| vLLM NixlConnector | KV 传输缓冲区 | GPU memory registration | `IPC_LOCK` 权限 |

### 内存语义通信应用总结

| 框架/项目 | 场景 | 使用的内存语义机制 | 关键性能数据 |
|---------|------|-----------------|----------|
| vLLM + NIXL | P/D 分离 KV 传输 | RDMA Write（GPU→GPU） | 400 Gbps（ConnectX-7）[^transferengine] |
| Mooncake TransferEngine | KV cache 传输 + RL 权重同步 | RDMA/NVLink 统一 API | FAST 2025 最佳论文 [^mooncake_readme] |
| DeepSeek DeepEP | MoE Expert dispatch/combine | GPU-Initiated RDMA（NVSHMEM+IBGDA） | 微秒级延迟（LL kernel）[^deepep_org] |
| DeepSeek 3FS | 训练数据/KV 分布式存储 | RDMA Read（Direct I/O） | 6.6 TiB/s（180 节点）[^3fs_aibase] |
| TransferEngine（RL 场景） | RLHF 训练→推理权重同步 | 单侧 RDMA Write | 1.3s 完成万亿参数同步 [^transferengine_rl] |
| NCCL 2.27 | 集合通信底层 | Symmetric Memory + Direct NIC | 小消息延迟降低 7.6x [^nccl_227] |
| NCCLX（Meta） | 集合通信 + Device API | Host/Device-initiated 混合 | 向内存语义方向演进 [^ncclx_meta] |
| SHIFT | 容错 RDMA 传输 | RDMA 用户态层 | 几乎零开销，可屏蔽 NIC 故障 [^shift] |

### 核心趋势

新兴的 LLM 系统模式——disaggregated inference、MoE routing、异步强化学习微调——需要灵活的点对点通信，已经超出了简单集合通信的范畴。现有实现与特定 NIC 绑定，阻碍了集成到推理引擎和跨硬件厂商的可移植性；统一接口的内存语义通信库（TransferEngine、NIXL、Mooncake）的出现，正是对这一需求的直接回应。[^transferengine]

---

## 7. 参考资料

[^pytorch_pinmem]: Sarkar, A. "Pinned Memory and DMA Transfers in PyTorch." *abhik.ai*, December 2024. https://www.abhik.ai/concepts/pytorch/pin-memory

[^pytorch_official]: PyTorch Documentation. "A guide on good usage of non_blocking and pin_memory() in PyTorch." *docs.pytorch.org*, Updated April 2026. https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html

[^pytorch_mem_pinning]: Chawla, A. "Memory Pinning to Accelerate Model Training." *Daily Dose of Data Science*, September 2024. https://blog.dailydoseofds.com/p/memory-pinning-to-accelerate-model

[^ssdtrain]: Doctoral Dissertation. "Code generation and runtime techniques for enabling data-efficient deep learning training on GPUs." *arxiv.org*, December 2024. https://arxiv.org/pdf/2412.04747

[^deepspeed_zerooffload]: DeepSpeed Team. "ZeRO-Offload." *deepspeed.ai*. https://www.deepspeed.ai/tutorials/zero-offload/

[^deepspeed_zero3]: DeepSpeed Documentation. "ZeRO — DeepSpeed 0.19.1 documentation." *deepspeed.readthedocs.io*. https://deepspeed.readthedocs.io/en/latest/zero3.html

[^deepspeed_config_json]: DeepSpeed Documentation. "DeepSpeed Configuration JSON." *deepspeed.ai*. https://www.deepspeed.ai/docs/config-json/

[^vllm_kv_offload_rfc]: vLLM Community. "[RFC]: KV cache offloading · Issue #19854 · vllm-project/vllm." *github.com*, June 2025. https://github.com/vllm-project/vllm/issues/19854

[^vllm_kv_offload_tutorial]: vLLM Documentation. "KV Cache Offloading." *docs.vllm.ai*. https://docs.vllm.ai/projects/production-stack/en/vllm-stack-0.1.2/tutorials/kv_cache.html

[^vllm_disagg_deepwiki]: DeepWiki. "KV Cache Transfer and Disaggregated Serving | vllm-project/vllm." *deepwiki.com*, 2025. https://deepwiki.com/vllm-project/vllm/9.4-kv-cache-transfer-and-disaggregated-serving

[^llmd_pd]: llm-d Documentation. "Prefill/Decode Disaggregation." *llm-d.ai*. https://llm-d.ai/docs/guide/Installation/pd-disaggregation

[^mooncake_readme]: Moonshot AI. "Mooncake README." *github.com/kvcache-ai/Mooncake*, March 2025. https://github.com/kvcache-ai/Mooncake/blob/main/README.md

[^mooncake_docs]: Moonshot AI. "Welcome to Mooncake." *kvcache-ai.github.io*. https://kvcache-ai.github.io/Mooncake/

[^mooncake_acm]: Qin, R. et al. "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving." *ACM Transactions on Storage*, 2025. https://dl.acm.org/doi/10.1145/3773772

[^deepep_github]: DeepSeek AI. "DeepEP: an efficient expert-parallel communication library." *github.com/deepseek-ai/DeepEP*, 2025. https://github.com/deepseek-ai/DeepEP

[^deepep_org]: DeepEP Project. "DeepEP." *deepep.org*. https://www.deepep.org/

[^nvshmem_blog]: cppcheatsheet. "Building NVSHMEM from Scratch: GPU-Initiated Networking." *cppcheatsheet.com*, May 2025. https://cppcheatsheet.com/notes/blog/nvshmem.html

[^gin_emergentmind]: emergentmind. "GPU-Initiated Networking (GIN)." *emergentmind.com*, November 2025. https://www.emergentmind.com/topics/gpu-initiated-networking-gin

[^uccl_ep]: Mao, Z. et al. "Previewing UCCL-EP: Flexible and Efficient Expert Parallelism for Cloud and Beyond." *uccl-project.github.io*, October 2025. https://uccl-project.github.io/posts/uccl-ep/

[^deepseek_wikipedia]: Wikipedia. "DeepSeek." *en.wikipedia.org*. https://en.wikipedia.org/wiki/DeepSeek

[^3fs_design_notes]: DeepSeek AI. "3FS Design Notes." *github.com/deepseek-ai/3FS*, 2025. https://github.com/deepseek-ai/3FS/blob/main/docs/design_notes.md

[^3fs_aibase]: aibase. "DeepSeek Open Source Week Day Five: 6.6 TiB/s Blowout!" *aibase.com*, 2025. https://www.aibase.com/news/15819

[^transferengine]: Licker, N., Hu, K., Zaytsev, V., Chen, L. "RDMA Point-to-Point Communication for LLM Systems (TransferEngine/fabric-lib)." *arxiv.org*, October 2025. https://arxiv.org/abs/2510.27656

[^transferengine_rl]: Licker, N. et al. "RDMA Point-to-Point Communication for LLM Systems." *arxiv.org*, October 2025. https://arxiv.org/html/2510.27656v1

[^verl_rl_frameworks]: Leoputera, H. "Anatomy of RL Frameworks." *hanifleo.com*, October 2025. https://www.hanifleo.com/anatomy-of-rl-frameworks/

[^rollart]: "ROLLART: Scaling Agentic RL Training via Disaggregated Infrastructure." *arxiv.org*. https://arxiv.org/pdf/2512.22560

[^nccl_227]: NVIDIA Technical Blog. "Enabling Fast Inference and Resilient Training with NCCL 2.27." *developer.nvidia.com*, October 2025. https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/

[^ncclx_meta]: Meta et al. "Collective Communication for 100k+ GPUs (NCCLX)." *arxiv.org*, October 2025. https://arxiv.org/html/2510.20171v1

[^shift]: SHIFT Authors. "SHIFT: Exploring the Boundary of RDMA Network Fault Tolerance." *arxiv.org*, February 2026. https://arxiv.org/html/2512.11094v2

[^iccl]: ICCL Authors. "An Efficient, Reliable and Observable Collective Communication Library in Large-scale GPU Training Clusters." *arxiv.org*, October 2025. https://arxiv.org/pdf/2510.00991

[^np_rdma]: NP-RDMA Authors. "NP-RDMA: Using Commodity RDMA without Pinning Memory." *arxiv.org*. https://arxiv.org/pdf/2310.11062

---

## Related Pages

- [[02_engineering/02_train_frameworks/comm_compute_fusion_guide]] — 通算融合（含 DeepEP/HybridEP 通信融合分析）
- [[03_infer_frameworks/mooncake_analysis]] — Mooncake 分离式推理架构与 RDMA KV Cache 传输
- [[02_train_frameworks/distributed_optimizer_deep_dive]] — 分布式优化器（含 ZeRO/DeepSpeed Offload 机制）
- [[02_engineering/02_train_frameworks/megatron-lm/megatron_comm_overlap_analysis]] — Megatron-LM 计算通信重叠（含 NCCL/RDMA 通信优化）
- [[../01_theory/01_models/deepseek/13_deepseek_v4_analysis]] — DeepSeek-V4（含 DeepEP、3FS 架构背景）
- [[index]] — 工程实现领域索引
- [[../changelog]] — 变更日志
