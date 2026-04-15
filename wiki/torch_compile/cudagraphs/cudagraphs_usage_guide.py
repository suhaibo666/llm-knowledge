"""
PyTorch CUDA Graphs 使用指南
包含所有使用方式的完整示例代码和 Mermaid 时序图
"""

import torch
import torch.nn as nn
import time
from typing import Optional, List, Dict, Any

print("=" * 80)
print("PyTorch CUDA Graphs 使用方式详解")
print("=" * 80)

# ============================================================================
# 方式1: 使用 torch.compile(model_fn, backend="cudagraphs")
# ============================================================================

print("\n" + "=" * 80)
print("方式1: torch.compile(model_fn, backend='cudagraphs')")
print("=" * 80)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1024, 2048)
        self.linear2 = nn.Linear(2048, 1024)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

def example_1_cudagraphs_backend():
    """
    方式1示例: 使用 torch.compile(model_fn, backend="cudagraphs")
    
    实现原理:
    1. cudagraphs backend 是 PyTorch 提供的专用后端，专门用于 CUDA Graphs 优化
    2. 它会自动捕获模型的计算图，并将其转换为 CUDA Graph
    3. CUDA Graph 是一个预先记录的 GPU 操作序列，可以显著减少 CPU-GPU 交互开销
    4. 该后端适用于形状固定的静态图，不支持动态形状
    
    优化机制:
    - 消除每次前向传播时的 CPU 开销（kernel launch、同步等）
    - 预先分配所有需要的内存，避免运行时内存分配
    - 将所有 GPU 操作记录为一个图，可以一次性提交执行
    
    时序图:
    1. 首次调用: warmup -> 捕获图 -> 记录所有操作 -> 创建 CUDA Graph
    2. 后续调用: 复制输入数据 -> replay CUDA Graph -> 获取输出
    
    限制:
    - 输入形状必须固定
    - 不支持控制流（if/else、循环等）
    - 不支持某些动态操作
    """
    print("\n--- 示例1: cudagraphs backend ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("警告: CUDA 不可用，使用 CPU 模拟")
    
    model = SimpleModel().to(device)
    model.eval()
    
    # 准备输入数据
    batch_size = 32
    input_size = 1024
    x = torch.randn(batch_size, input_size, device=device)
    
    # 使用 cudagraphs backend 编译模型
    compiled_model = torch.compile(model, backend="cudagraphs")
    
    # Warmup: 首次运行会捕获 CUDA Graph
    print("首次运行（捕获 CUDA Graph）...")
    with torch.no_grad():
        output = compiled_model(x)
    
    # 计时运行
    print("执行性能测试...")
    num_iterations = 100
    
    # 测试未编译的模型
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(x)
    uncompiled_time = time.time() - start_time
    
    # 测试编译后的模型
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = compiled_model(x)
    compiled_time = time.time() - start_time
    
    print(f"未编译模型: {uncompiled_time:.4f}s ({num_iterations} 次迭代)")
    print(f"编译模型 (cudagraphs): {compiled_time:.4f}s ({num_iterations} 次迭代)")
    print(f"加速比: {uncompiled_time / compiled_time:.2f}x")
    
    return compiled_model

# Mermaid 时序图: 方式1
TimeSequenceDiagramCudagraphsBackend = """
```mermaid
sequenceDiagram
    participant CPU as CPU (Python)
    participant Backend as CUDA Graph Backend
    participant GPU as GPU (CUDA)
    
    Note over CPU,GPU: 首次调用 (Warmup 阶段)
    
    CPU->>Backend: 调用 compiled_model(x)
    Backend->>Backend: 检查 graphExec 是否存在
    Backend->>Backend: graphExec = None (首次调用)
    
    Note over Backend: 开始 Warmup 阶段
    
    Backend->>GPU: Warmup 执行 (验证操作序列)
    activate GPU
    GPU->>GPU: cuLaunchKernel (linear1)
    GPU->>GPU: cuLaunchKernel (relu)
    GPU->>GPU: cuLaunchKernel (linear2)
    deactivate GPU
    GPU-->>Backend: 操作完成
    
    Note over Backend: 开始捕获阶段
    
    Backend->>GPU: cudaGraphCaptureBegin()
    activate GPU
    GPU->>GPU: 进入捕获模式 (记录所有操作)
    deactivate GPU
    GPU-->>Backend: 模式已激活
    
    Backend->>GPU: 重放模型执行 (记录到图)
    activate GPU
    GPU->>GPU: cuLaunchKernel (linear1) - 记录到图 (不实际执行)
    GPU->>GPU: cuLaunchKernel (relu) - 记录到图
    GPU->>GPU: cuLaunchKernel (linear2) - 记录到图
    deactivate GPU
    GPU-->>Backend: 已记录
    
    Backend->>GPU: cudaGraphCaptureEnd()
    activate GPU
    GPU->>GPU: 退出捕获模式，返回 graph 对象
    deactivate GPU
    GPU-->>Backend: graph 对象
    
    Note over Backend: graph 对象包含节点列表、内存依赖、执行顺序
    
    Backend->>GPU: cudaGraphInstantiate()
    activate GPU
    GPU->>GPU: 分配资源 (内存池、执行队列、同步对象)
    deactivate GPU
    GPU-->>Backend: graphExec 对象
    
    Note over Backend: graphExec 对象包含可执行实例、静态内存指针
    
    Backend->>Backend: 保存 graphExec
    
    Note over CPU,GPU: 后续调用 (执行阶段)
    
    CPU->>Backend: 调用 compiled_model(x)
    Backend->>Backend: 检查 graphExec
    Backend->>Backend: graphExec 存在，直接执行
    
    Backend->>GPU: cudaMemcpyAsync (复制输入到静态内存)
    activate GPU
    GPU->>GPU: 复制数据到静态输入缓冲区
    deactivate GPU
    GPU-->>Backend: 复制完成
    
    Backend->>GPU: cudaGraphLaunch()
    activate GPU
    GPU->>GPU: 提交整个图执行
    GPU->>GPU: 执行 linear1 (使用静态内存)
    GPU->>GPU: 执行 relu
    GPU->>GPU: 执行 linear2
    deactivate GPU
    GPU-->>Backend: 执行完成
    
    Backend->>GPU: cudaMemcpyAsync (从静态内存复制输出)
    activate GPU
    deactivate GPU
    GPU-->>Backend: 复制完成
    
    Backend-->>CPU: 返回结果
    
    Note over CPU,GPU: 性能优势: 消除多次 CPU-GPU 交互、kernel launch 开销
```
"""

print("\n" + "=" * 80)
print("时序图: torch.compile(backend='cudagraphs')")
print("=" * 80)
print(TimeSequenceDiagramCudagraphsBackend)

# ============================================================================
# 方式2: 使用 torch.compile(model_fn, backend="inductor", mode="reduce-overhead")
# ============================================================================

print("\n" + "=" * 80)
print("方式2: torch.compile(model_fn, backend='inductor', mode='reduce-overhead')")
print("=" * 80)

def example_2_inductor_reduce_overhead():
    """
    方式2示例: 使用 torch.compile(model_fn, backend="inductor", mode="reduce-overhead")
    
    实现原理:
    1. Inductor 是 PyTorch 2.0 的默认编译后端，基于 TorchScript 和 AOTAutograd
    2. "reduce-overhead" 模式专门优化推理性能，会自动使用 CUDA Graphs
    3. 该模式会：
       - 生成优化的 Triton kernels
       - 自动使用 CUDA Graphs（当条件满足时）
       - 优化内存分配和 kernel launch 开销
    
    优化机制:
    - AOT 编译: 提前编译所有算子
    - Kernel fusion: 将多个操作融合为一个 kernel
    - CUDA Graphs: 对静态形状的子图使用 CUDA Graphs
    - 内存规划: 预先分配内存，减少运行时分配
    
    时序图:
    1. 编译阶段: 导出 FX 图 -> lowering -> 代码生成 -> 编译 kernels
    2. 运行阶段: 检查缓存 -> 使用 CUDA Graphs 或常规执行
    
    适用场景:
    - 推理场景（inference）
    - 批量大小固定的场景
    - 需要最大性能的场景
    """
    print("\n--- 示例2: inductor backend with reduce-overhead mode ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SimpleModel().to(device)
    model.eval()
    
    # 准备输入数据
    batch_size = 32
    input_size = 1024
    x = torch.randn(batch_size, input_size, device=device)
    
    # 使用 inductor backend 编译模型，启用 reduce-overhead 模式
    # 这个模式会自动使用 CUDA Graphs（当条件满足时）
    compiled_model = torch.compile(model, backend="inductor", mode="reduce-overhead")
    
    # Warmup: 首次运行会触发编译
    print("首次运行（触发编译）...")
    with torch.no_grad():
        output = compiled_model(x)
    
    # 计时运行
    print("执行性能测试...")
    num_iterations = 100
    
    # 测试未编译的模型
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(x)
    uncompiled_time = time.time() - start_time
    

    
    # 测试编译后的模型
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = compiled_model(x)
    compiled_time = time.time() - start_time
    
    print(f"未编译模型: {uncompiled_time:.4f}s ({num_iterations} 次迭代)")
    print(f"编译模型 (inductor+reduce-overhead): {compiled_time:.4f}s ({num_iterations} 次迭代)")
    print(f"加速比: {uncompiled_time / compiled_time:.2f}x")
    
    return compiled_model

# Mermaid 时序图: 方式2
TimeSequenceDiagramInductor = """
```mermaid
sequenceDiagram
    participant CPU as CPU (Python)
    participant Inductor as Inductor Backend
    participant GPU as GPU (CUDA)
    
    Note over CPU,GPU: 首次调用 (编译阶段)
    
    CPU->>Inductor: 调用 compiled_model(x)
    Inductor->>Inductor: 检查编译缓存
    Inductor->>Inductor: 缓存未命中，开始编译流程
    
    Note over Inductor: 步骤1: 导出 FX Graph
    
    Inductor->>Inductor: torch.fx.symbolic_trace
    Inductor->>Inductor: 创建 FX Graph (输入节点、linear1、relu、linear2、输出节点)
    
    Note over Inductor: 步骤2: Lowering
    
    Inductor->>Inductor: 转换为中间表示 (IR)
    Inductor->>Inductor: IR 包含 Buffer 管理、Kernel 调用、依赖关系
    

    
    Note over Inductor: 步骤3: 代码生成
    
    Inductor->>Inductor: 生成 Triton/CUDA 代码
    Inductor->>Inductor: Triton kernels: fused_linear, fused_relu
    
    Note over Inductor: 步骤4: 分析子图
    
    Inductor->>Inductor: 识别静态子图
    Inductor->>Inductor: 子图1 (静态): linear1, relu, linear2 - 可使用 CUDA Graphs
    
    Note over Inductor: 步骤5: 编译 Kernels
    
    Inductor->>GPU: NVRTC/Triton 编译
    activate GPU
    GPU->>GPU: 编译 Triton kernels (优化 tiling、向量化)
    deactivate GPU
    GPU-->>Inductor: 编译完成
    
    Note over Inductor: 步骤6: 创建 CUDA Graphs (如果适用)
    
    Inductor->>GPU: cuStreamBeginCapture
    activate GPU
    GPU->>GPU: 捕获子图操作
    deactivate GPU
    GPU-->>Inductor: 已捕获
    
    Inductor->>GPU: cuStreamEndCapture
    activate GPU
    deactivate GPU
    GPU-->>Inductor: graph 对象
    
    Inductor->>GPU: cuGraphInstantiate
    activate GPU
    GPU->>GPU: 分配资源 (内存池、执行队列)
    deactivate GPU
    GPU-->>Inductor: graphExec 创建完成
    
    Note over Inductor: 步骤7: 生成执行函数
    
    Inductor->>Inductor: 创建 compiled_fn (Triton kernels + CUDA Graphs + 内存管理)
    
    Inductor->>Inductor: 保存到编译缓存
    
    Note over CPU,GPU: 后续调用 (执行阶段)
    
    CPU->>Inductor: 调用 compiled_model(x)
    Inductor->>Inductor: 使用编译结果
    
    Note over Inductor: 执行优化代码
    
    Inductor->>GPU: cudaMemcpyAsync (复制输入到内存池)
    activate GPU
    deactivate GPU
    GPU-->>Inductor: 完成
    
    Inductor->>GPU: cuLaunchKernel (fused_linear)
    activate GPU
    deactivate GPU
    GPU-->>Inductor: 完成
    
    Inductor->>GPU: cuGraphLaunch (执行子图)
    activate GPU
    GPU->>GPU: 执行整个子图 (linear1, relu, linear2 连续执行)
    deactivate GPU
    GPU-->>Inductor: 完成
    
    Inductor->>GPU: cudaMemcpyAsync (复制输出)
    activate GPU
    deactivate GPU
    GPU-->>Inductor: 完成
    
    Inductor-->>CPU: 返回结果
    
    Note over CPU,GPU: 优化层级: Kernel Fusion -> Triton Kernels -> CUDA Graphs -> 内存优化
```
"""

print("\n" + "=" * 80)
print("时序图: torch.compile(backend='inductor', mode='reduce-overhead')")
print("=" * 80)
print(TimeSequenceDiagramInductor)

# ============================================================================
# 方式3: 使用 with torch.cuda.graph() 上下文管理器
# ============================================================================

print("\n" + "=" * 80)
print("方式3: with torch.cuda.graph() 上下文管理器")
print("=" * 80)

def example_3_cuda_graph_context():
    """
    方式3示例: 使用 torch.cuda.graph() 上下文管理器
    
    实现原理:
    1. 这是最低级别的 CUDA Graphs API，提供最大的灵活性
    2. 需要手动管理输入/输出的静态内存池
    3. 流程：
       a. 创建静态内存池（用于输入、输出、中间结果）
       b. 在 graph() 上下文中执行一次，记录所有操作
       c. 后续调用时，只需复制输入到静态内存，然后 replay 图
    
    优化机制:
    - 完全消除 kernel launch 开销
    - 预先分配所有内存
    - 一次性提交整个计算图
    
    时序图:
    1. 初始化: 分配静态内存池
    2. 捕获: 在 graph() 上下文中执行一次
    3. 运行: 复制输入 -> replay -> 复制输出
    
    适用场景:
    - 需要精细控制内存的场景
    - 自定义推理 pipeline
    - 需要与其他 CUDA 操作同步的场景
    
    注意事项:
    - 必须使用静态内存（不能在图内分配新内存）
    - 输入形状必须固定
    - 需要手动管理内存同步
    """
    print("\n--- 示例3: torch.cuda.graph() 上下文管理器 ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("警告: CUDA 不可用，使用 CPU 模拟")
        return None
    
    model = SimpleModel().to(device)
    model.eval()
    
    # 准备输入数据
    batch_size = 32
    input_size = 1024
    output_size = 1024
    
    # 创建静态内存池
    # 注意: 在 CUDA Graph 中，所有内存必须预先分配
    static_input = torch.randn(batch_size, input_size, device=device)
    static_output = torch.randn(batch_size, output_size, device=device)
    
    # 创建 CUDA Graph
    print("捕获 CUDA Graph...")
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    
    with torch.cuda.graph(graph, stream=stream):
        # 在这个上下文中，所有操作都会被记录到图中
        # 注意: 必须使用静态内存，不能创建新的张量
        temp = static_input
        temp = model.linear1(temp)
        temp = model.relu(temp)
        temp = model.linear2(temp)
        static_output.copy_(temp)
    
    print("CUDA Graph 捕获完成")
    
    # 测试性能
    print("执行性能测试...")
    num_iterations = 100
    
    # 测试未使用 CUDA Graph 的模型
    test_input = torch.randn(batch_size, input_size, device=device)
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(test_input)
    uncompiled_time = time.time() - start_time
    
    # 测试使用 CUDA Graph
    start_time = time.time()
    for _ in range(num_iterations):
        # 复制输入到静态内存
        static_input.copy_(test_input)
        # Replay CUDA Graph
        graph.replay()
        # 从静态内存复制输出
        result = static_output.clone()
    compiled_time = time.time() - start_time
    
    print(f"未使用 CUDA Graph: {uncompiled_time:.4f}s ({num_iterations} 次迭代)")
    print(f"使用 CUDA Graph: {compiled_time:.4f}s ({num_iterations} 次迭代)")
    print(f"加速比: {uncompiled_time / compiled_time:.2f}x")
    
    return graph

# Mermaid 时序图: 方式3
TimeSequenceDiagramContextManager = """
```mermaid
sequenceDiagram
    participant CPU as CPU (Python)
    participant API as CUDA Graph API
    participant GPU as GPU (CUDA)
    
    Note over CPU,GPU: 初始化阶段
    
    CPU->>CPU: 创建静态内存池
    CPU->>CPU: static_input = torch.empty_like(x)
    CPU->>CPU: static_output = torch.empty_like(y)
    
    Note over CPU: 静态内存池布局: 输入缓冲区(固定形状) + 输出缓冲区(固定形状)
    
    CPU->>API: stream = cuda.Stream()
    API->>GPU: cuStreamCreate
    activate GPU
    GPU-->>API: Stream 创建完成
    deactivate GPU
    
    CPU->>CPU: graph = CUDAGraph()
    
    Note over CPU,GPU: 捕获阶段
    
    CPU->>API: 进入 graph 上下文 (with torch.cuda.graph)
    API->>API: __enter__()
    API->>API: 保存当前 stream
    API->>API: 切换到目标 stream
    API->>GPU: cuStreamSetCurrent
    activate GPU
    GPU-->>API: Stream 已切换
    deactivate GPU
    
    API->>API: 开始捕获
    API->>GPU: cuStreamBeginCapture
    activate GPU
    GPU->>GPU: 进入捕获模式 (mode=GLOBAL, 记录所有操作)
    deactivate GPU
    GPU-->>API: 捕获模式已激活
    
    Note over CPU,GPU: 执行模型操作 (使用静态内存)
    
    CPU->>API: temp = static_input
    API->>GPU: 记录 buffer 引用
    deactivate GPU
    
    CPU->>API: temp = model.linear1(temp)
    API->>GPU: cuLaunchKernel (linear1)
    activate GPU
    GPU->>GPU: 记录到图: kernel 参数、buffer 指针、执行配置 (不实际执行)
    deactivate GPU
    GPU-->>API: 已记录
    
    CPU->>API: temp = model.relu(temp)
    API->>GPU: cuLaunchKernel (relu)
    activate GPU
    GPU->>GPU: 记录到图
    deactivate GPU
    GPU-->>API: 已记录
    
    CPU->>API: temp = model.linear2(temp)
    API->>GPU: cuLaunchKernel (linear2)
    activate GPU
    GPU->>GPU: 记录到图
    deactivate GPU
    GPU-->>API: 已记录
    
    CPU->>API: static_output.copy_(temp)
    API->>GPU: cudaMemcpyAsync (Device -> Device)
    activate GPU
    GPU->>GPU: 记录内存操作
    deactivate GPU
    GPU-->>API: 已记录
    
    CPU->>API: 退出 graph 上下文
    API->>API: __exit__()
    API->>API: 结束捕获
    API->>GPU: cuStreamEndCapture
    activate GPU
    GPU->>GPU: 退出捕获模式，验证图完整性，返回 graph 对象
    deactivate GPU
    GPU-->>API: graph 对象
    
    Note over API: graph 对象包含节点列表、边列表、内存依赖、执行顺序
    
    API->>API: 实例化图
    API->>GPU: cuGraphInstantiate
    activate GPU
    GPU->>GPU: 分配资源: 验证节点、分配内存池、创建执行队列、设置同步对象
    deactivate GPU
    GPU-->>API: graphExec 对象
    
    Note over API: graphExec 对象包含可执行实例、静态内存映射、准备好执行
    
    API->>API: 恢复 stream
    API->>GPU: cuStreamSetCurrent
    activate GPU
    GPU-->>API: Stream 已恢复
    deactivate GPU
    
    API-->>CPU: 上下文退出
    
    Note over CPU,GPU: 执行阶段
    
    CPU->>CPU: 后续调用: 复制输入到静态内存
    CPU->>CPU: static_input.copy_(x)
    CPU->>API: cudaMemcpyAsync()
    API->>GPU: cudaMemcpyAsync (Host -> Device)
    activate GPU
    deactivate GPU
    GPU-->>API: 复制完成
    
    CPU->>CPU: Replay 图
    CPU->>API: graph.replay()
    API->>GPU: cuGraphLaunch()
    activate GPU
    GPU->>GPU: 提交整个图执行 (一次性提交)
    GPU->>GPU: 执行 linear1 (使用静态内存)
    GPU->>GPU: 执行 relu
    GPU->>GPU: 执行 linear2
    GPU->>GPU: copy output
    deactivate GPU
    GPU-->>API: 执行完成
    
    CPU->>CPU: 复制输出
    CPU->>CPU: result = static_output.clone()
    CPU->>API: cudaMemcpyAsync()
    API->>GPU: cudaMemcpyAsync (Device -> Host)
    activate GPU
    deactivate GPU
    GPU-->>API: 复制完成
    
    CPU->>CPU: 返回结果
    
    Note over CPU,GPU: 关键特性: 静态内存管理、完全控制、高性能
```
"""

print("\n" + "=" * 80)
print("时序图: torch.cuda.graph() 上下文管理器")
print("=" * 80)
print(TimeSequenceDiagramContextManager)

# ============================================================================
# 方式4: 使用 torch.cuda.make_graphed_callables (高级用法)
# ============================================================================

print("\n" + "=" * 80)
print("方式4: torch.cuda.make_graphed_callables (高级用法)")
print("=" * 80)

def example_4_make_graphed_callables():
    """
    方式4示例: 使用 torch.cuda.make_graphed_callables
    
    实现原理:
    1. 这是 PyTorch 提供的高级 API，用于将多个 callable 包装为 CUDA Graphs
    2. 可以同时处理多个模型或函数
    3. 自动管理内存池和输入/输出
    
    优化机制:
    - 自动创建和管理静态内存池
    - 支持多个独立的 callable
    - 自动处理输入/输出的复制
    
    时序图:
    1. 创建: 准备样本输入 -> make_graphed_callables
    2. 运行: 直接调用包装后的函数
    
    适用场景:
    - 需要同时优化多个函数
    - 不想手动管理内存
    - 需要更高级的抽象
    
    注意事项:
    - 需要 PyTorch 1.10+
    - 输入形状必须固定
    """
    print("\n--- 示例4: torch.cuda.make_graphed_callables ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("警告: CUDA 不可用，使用 CPU 模拟")
        return None
    
    model = SimpleModel().to(device)
    model.eval()
    
    # 准备样本输入（用于捕获图）
    batch_size = 32
    input_size = 1024
    sample_input = torch.randn(batch_size, input_size, device=device)
    
    # 使用 make_graphed_callables 包装模型
    # 这会自动捕获 CUDA Graph 并管理内存
    print("创建 graphed callable...")
    graphed_model = torch.cuda.make_graphed_callables(model, [sample_input])
    
    print("CUDA Graph 创建完成")
    
    # 测试性能
    print("执行性能测试...")
    num_iterations = 100
    
    # 测试未使用 CUDA Graph 的模型
    test_input = torch.randn(batch_size, input_size, device=device)
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(test_input)
    uncompiled_time = time.time() - start_time
    
    # 测试使用 CUDA Graph
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = graphed_model(test_input)
    compiled_time = time.time() - start_time
    
    print(f"未使用 CUDA Graph: {uncompiled_time:.4f}s ({num_iterations} 次迭代)")
    print(f"使用 make_graphed_callables: {compiled_time:.4f}s ({num_iterations} 次迭代)")
    print(f"加速比: {uncompiled_time / compiled_time:.2f}x")
    
    return graphed_model

# Mermaid 时序图: 方式4
TimeSequenceDiagramMakeGraphed = """
```mermaid
sequenceDiagram
    participant CPU as CPU (Python)
    participant API as make_graphed_callables
    participant GPU as GPU (CUDA)
    
    Note over CPU,GPU: 准备阶段
    
    CPU->>CPU: 提供 callables 和 sample_inputs
    CPU->>CPU: [model1, model2], [input1, input2]
    
    CPU->>API: 接收参数
    API->>API: 验证参数 (callables 列表、sample_inputs 列表、长度匹配)
    
    Note over CPU,GPU: 处理每个 callable
    
    par 处理 callable 1
        API->>API: 步骤1: 分析输入形状
        API->>API: input_shape = sample_input.shape (32, 512)
        
        API->>API: 步骤2: 创建静态内存池
        API->>GPU: 执行一次以追踪内存
        activate GPU
        GPU-->>API: 返回输出形状
        deactivate GPU
        
        Note over API: 静态内存池: input_buffer, output_buffer, intermediate buffers
        
        API->>API: 步骤3: Warmup 执行
        API->>GPU: cuLaunchKernel
        activate GPU
        GPU-->>API: 完成
        deactivate GPU
        
        API->>API: 步骤4: 捕获 CUDA Graph
        API->>GPU: cuStreamBeginCapture
        activate GPU
        GPU->>GPU: 进入捕获模式
        deactivate GPU
        GPU-->>API: 模式已激活
        
        API->>GPU: 记录所有操作 (不实际执行)
        activate GPU
        deactivate GPU
        GPU-->>API: 已记录
        
        API->>GPU: cuStreamEndCapture
        activate GPU
        deactivate GPU
        GPU-->>API: graph 对象
        
        API->>GPU: cuGraphInstantiate
        activate GPU
        GPU->>GPU: 分配资源
        deactivate GPU
        GPU-->>API: graphExec 对象
        
        API->>API: 步骤5: 创建包装函数
        Note over API: graphed_callable: graphExec, static_pool, auto I/O
    end
    
    par 处理 callable 2
        API->>API: 重复步骤1-5
    end
    
    API-->>CPU: 返回 graphed callables
    
    Note over CPU,GPU: 执行阶段
    
    CPU->>API: 调用 graphed_model(x)
    API->>API: graphed_callable 调用
    
    Note over API: 自动处理 I/O
    
    API->>API: 1. 复制输入到静态内存
    API->>GPU: cudaMemcpyAsync (Host -> Device)
    activate GPU
    deactivate GPU
    GPU-->>API: 复制完成
    
    API->>API: 2. cudaGraphLaunch()
    API->>GPU: cuGraphLaunch
    activate GPU
    GPU->>GPU: Replay 整个图
    deactivate GPU
    GPU-->>API: 执行完成
    
    API->>API: 3. 复制输出
    API->>GPU: cudaMemcpyAsync (Device -> Host)
    activate GPU
    deactivate GPU
    GPU-->>API: 复制完成
    
    API-->>CPU: 返回结果
    
    Note over CPU,GPU: 优势: 自动化、多函数支持、简单易用
```
"""

print("\n" + "=" * 80)
print("时序图: torch.cuda.make_graphed_callables")
print("=" * 80)
print(TimeSequenceDiagramMakeGraphed)

# ============================================================================
# 方式5: 使用 torch.compile 的 experimental 参数 (实验性功能)
# ============================================================================

print("\n" + "=" * 80)
print("方式5: torch.compile 的 experimental 参数 (实验性功能)")
print("=" * 80)

def example_5_experimental_cudagraphs():
    """
    方式5示例: 使用 torch.compile 的 experimental 参数
    
    实现原理:
    1. PyTorch 2.1+ 提供了 experimental 参数来提供更细粒度的控制
    2. 可以通过 enable_cuda_graph=True 显式启用 CUDA Graphs
    3. 可以配置 CUDA Graphs 的捕获策略
    
    优化机制:
    - 与 inductor backend 集成
    - 提供更细粒度的控制
    - 支持动态形状的部分优化
    
    时序图:
    类似于 inductor backend，但增加了 experimental 层的控制
    
    适用场景:
    - 需要精细控制 CUDA Graphs 行为
    - 实验性功能测试
    - 高级性能调优
    
    注意事项:
    - 实验性 API，可能不稳定
    - 需要 PyTorch 2.1+
    """
    print("\n--- 示例5: torch.compile experimental 参数 ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SimpleModel().to(device)
    model.eval()
    
    # 准备输入数据
    batch_size = 32
    input_size = 1024
    x = torch.randn(batch_size, input_size, device=device)
    
    try:
        # 使用 experimental 参数启用 CUDA Graphs
        compiled_model = torch.compile(
            model,
            backend="inductor",
            mode="reduce-overhead",
            experimental={"enable_cuda_graph": True}
        )
        
        # Warmup
        print("首次运行（触发编译）...")
        with torch.no_grad():
            output = compiled_model(x)
        
        # 测试性能
        print("执行性能测试...")
        num_iterations = 100
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(x)
        uncompiled_time = time.time() - start_time
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = compiled_model(x)
        compiled_time = time.time() - start_time
        
        print(f"未编译模型: {uncompiled_time:.4f}s ({num_iterations} 次迭代)")
        print(f"编译模型 (experimental CUDA Graphs): {compiled_time:.4f}s ({num_iterations} 次迭代)")
        print(f"加速比: {uncompiled_time / compiled_time:.2f}x")
        
        return compiled_model
    except Exception as e:
        print(f"experimental 参数可能不可用: {e}")
        return None

# Mermaid 时序图: 方式5
TimeSequenceDiagramExperimental = """
```mermaid
sequenceDiagram
    participant CPU as CPU (Python)
    participant Inductor as Inductor + Experimental
    participant GPU as GPU (CUDA)
    
    Note over CPU,GPU: 首次调用 (编译阶段)
    
    CPU->>Inductor: 调用 compiled_model(x)
    Inductor->>Inductor: 检查编译缓存
    Inductor->>Inductor: 缓存未命中，开始编译流程
    
    Note over Inductor: 步骤1: 导出 FX Graph
    
    Inductor->>Inductor: torch.fx.symbolic_trace
    Inductor->>Inductor: 创建 FX Graph
    
    Note over Inductor: 步骤2: 分析 experimental 参数
    
    Inductor->>Inductor: 解析 experimental 配置
    Note over Inductor: experimental: {enable_cuda_graph: True, cuda_graph_capture_steps: 3, cuda_graph_min_size: 10, cuda_graph_max_size: 100}
    
    Note over Inductor: 步骤3: 分析子图
    
    Inductor->>Inductor: 遍历 FX Graph 节点
    Inductor->>Inductor: 节点分析: 节点类型、输入形状、输出形状、依赖关系
    
    Inductor->>Inductor: 识别静态子图
    Note over Inductor: 静态子图1: 节点数=15, 形状固定, 无控制流 -> 可捕获
    Note over Inductor: 动态子图1: 节点数=3, 形状不固定 -> 不可捕获
    
    Note over Inductor: 步骤4: 捕获静态子图
    
    par 对于每个可捕获子图
        Inductor->>Inductor: 创建静态内存池
        
        Inductor->>GPU: cuStreamBeginCapture
        activate GPU
        GPU->>GPU: 捕获子图操作
        deactivate GPU
        GPU-->>Inductor: 已捕获
        
        Inductor->>GPU: cuStreamEndCapture
        activate GPU
        deactivate GPU
        GPU-->>Inductor: graph 对象
        
        Inductor->>GPU: cuGraphInstantiate
        activate GPU
        GPU->>GPU: 分配资源
        deactivate GPU
        GPU-->>Inductor: graphExec 对象
        
        Inductor->>Inductor: 保存 graph 和静态池
    end
    
    Note over Inductor: 步骤5: 编译动态部分
    
    Inductor->>Inductor: Lowering 到 Triton
    Inductor->>GPU: NVRTC/Triton 编译
    activate GPU
    deactivate GPU
    GPU-->>Inductor: 编译完成
    
    Note over Inductor: 步骤6: 创建混合执行函数
    
    Inductor->>Inductor: compiled_fn: 静态子图(CUDA Graphs) + 动态子图(Triton) + 内存管理(自动)
    
    Inductor->>Inductor: 保存到编译缓存
    
    Note over CPU,GPU: 后续调用 (执行阶段)
    
    CPU->>Inductor: 调用 compiled_model(x)
    Inductor->>Inductor: 使用编译结果
    
    Note over Inductor: 执行动态子图
    
    Inductor->>GPU: cuLaunchKernel (Triton kernel)
    activate GPU
    deactivate GPU
    GPU-->>Inductor: 完成
    
    Note over Inductor: 执行静态子图 (使用 CUDA Graphs)
    
    par 对于每个 graphed 子图
        Inductor->>Inductor: 复制输入到静态内存
        Inductor->>GPU: cudaMemcpyAsync
        activate GPU
        deactivate GPU
        GPU-->>Inductor: 复制完成
        
        Inductor->>GPU: cuGraphLaunch
        activate GPU
        GPU->>GPU: Replay 子图
        deactivate GPU
        GPU-->>Inductor: 执行完成
        
        Inductor->>Inductor: 复制输出
        Inductor->>GPU: cudaMemcpyAsync
        activate GPU
        deactivate GPU
        GPU-->>Inductor: 复制完成
    end
    
    Inductor-->>CPU: 返回结果
    
    Note over CPU,GPU: 优势: 智能捕获、混合执行、细粒度控制、实验性功能
```
"""

print("\n" + "=" * 80)
print("时序图: torch.compile(experimental={'enable_cuda_graph': True})")
print("=" * 80)
print(TimeSequenceDiagramExperimental)

# ============================================================================
# 综合比较和总结
# ============================================================================

print("\n" + "=" * 80)
print("综合比较和总结")
print("=" * 80)

ComparisonTable = """
+------------------------+------------------+------------------+------------------+
| 使用方式               | 易用性           | 灵活性           | 性能优化         |
+------------------------+------------------+------------------+------------------+
| backend="cudagraphs"   | 高               | 中               | 高               |
|                        | (自动捕获)       | (固定形状)       | (纯 CUDA Graphs) |
+------------------------+------------------+------------------+------------------+
| backend="inductor"     | 高               | 高               | 很高             |
| mode="reduce-overhead" | (自动优化)       | (支持动态形状)   | (多级优化)       |
+------------------------+------------------+------------------+------------------+
| torch.cuda.graph()     | 低               | 很高             | 很高             |
| 上下文管理器           | (手动管理)       | (完全控制)       | (精细控制)       |
+------------------------+------------------+------------------+------------------+
| make_graphed_callables | 中               | 中               | 高               |
|                        | (半自动)         | (多函数支持)     | (自动管理)       |
+------------------------+------------------+------------------+------------------+
| experimental 参数      | 中               | 高               | 很高             |
|                        | (高级配置)       | (细粒度控制)     | (实验性优化)     |
+------------------------+------------------+------------------+------------------+
"""

print(ComparisonTable)

print("\n" + "=" * 80)
print("使用建议")
print("=" * 80)

Recommendations = """
1. **初学者/快速原型**: 使用 backend="cudagraphs"
   - 最简单，自动处理大部分细节
   - 适合固定形状的推理场景

2. **生产环境/推理优化**: 使用 backend="inductor" + mode="reduce-overhead"
   - PyTorch 2.0 推荐
   - 自动选择最优策略
   - 支持更多场景

3. **高级用户/精细控制**: 使用 torch.cuda.graph()
   - 完全控制内存和执行
   - 适合自定义 pipeline
   - 需要深入理解 CUDA

4. **多函数优化**: 使用 make_graphed_callables
   - 同时优化多个函数
   - 自动内存管理
   - 适合复杂场景

5. **实验性功能**: 使用 experimental 参数
   - 探索最新优化
   - 细粒度控制
   - 注意 API 稳定性
"""

print(Recommendations)

print("\n" + "=" * 80)
print("运行所有示例")
print("=" * 80)

if __name__ == "__main__":
    # 运行所有示例
    example_1_cudagraphs_backend()
    example_2_inductor_reduce_overhead()
    example_3_cuda_graph_context()
    example_4_make_graphed_callables()
    example_5_experimental_cudagraphs()
    
    print("\n" + "=" * 80)
    print("所有示例运行完成!")
    print("=" * 80)
