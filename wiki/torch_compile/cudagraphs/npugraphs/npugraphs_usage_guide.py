"""
NPU Graphs (npugraphs) 使用指南
对比 CUDA Graphs 和 NPU Graphs 的实现
"""

import torch
import torch.nn as nn
import time

print("=" * 80)
print("NPU Graphs vs CUDA Graphs 对比指南")
print("=" * 80)

# 检查设备可用性
cuda_available = torch.cuda.is_available()
npu_available = hasattr(torch, 'npu') and hasattr(torch.npu, 'is_available') and torch.npu.is_available()

print(f"\nCUDA 可用: {cuda_available}")
print(f"NPU 可用: {npu_available}")

if not cuda_available and not npu_available:
    print("\n错误: CUDA 和 NPU 都不可用")
    exit(1)

# ============================================================================
# 简单模型定义
# ============================================================================

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

# ============================================================================
# CUDA Graphs 示例
# ============================================================================

print("\n" + "=" * 80)
print("CUDA Graphs 示例")
print("=" * 80)

def example_cuda_graphs():
    """
    CUDA Graphs 使用示例
    
    使用 torch.cuda.graph() 上下文管理器
    """
    if not cuda_available:
        print("跳过: CUDA 不可用")
        return None
    
    print("\n--- CUDA Graphs 示例 ---")
    
    device = torch.device("cuda")
    model = SimpleModel().to(device)
    model.eval()
    
    batch_size = 32
    input_size = 1024
    output_size = 1024
    
    # 创建静态内存池
    static_input = torch.randn(batch_size, input_size, device=device)
    static_output = torch.randn(batch_size, output_size, device=device)
    
    # 创建 CUDA Graph
    print("捕获 CUDA Graph...")
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    
    with torch.cuda.graph(graph, stream=stream):
        temp = static_input
        temp = model.linear1(temp)
        temp = model.relu(temp)
        temp = model.linear2(temp)
        static_output.copy_(temp)
    
    print("CUDA Graph 捕获完成")
    
    # 性能测试
    num_iterations = 50
    test_input = torch.randn(batch_size, input_size, device=device)
    
    # 未使用 CUDA Graph
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(test_input)
    uncompiled_time = time.time() - start_time
    
    # 使用 CUDA Graph
    start_time = time.time()
    for _ in range(num_iterations):
        static_input.copy_(test_input)
        graph.replay()
        result = static_output.clone()
    compiled_time = time.time() - start_time
    
    print(f"未使用 CUDA Graph: {uncompiled_time:.4f}s")
    print(f"使用 CUDA Graph: {compiled_time:.4f}s")
    print(f"加速比: {uncompiled_time / compiled_time:.2f}x")
    
    return graph

cuda_graph = example_cuda_graphs()

# ============================================================================
# NPU Graphs 示例
# ============================================================================

print("\n" + "=" * 80)
print("NPU Graphs 示例")
print("=" * 80)

def example_npu_graphs():
    """
    NPU Graphs 使用示例
    
    使用 torch_npu.npu.graph() 上下文管理器
    与 CUDA Graphs API 类似
    """
    if not npu_available:
        print("跳过: NPU 不可用")
        return None
    
    print("\n--- NPU Graphs 示例 ---")
    
    import torch_npu
    
    device = torch.device("npu")
    model = SimpleModel().to(device)
    model.eval()
    
    batch_size = 32
    input_size = 1024
    output_size = 1024
    
    # 创建静态内存池
    static_input = torch.randn(batch_size, input_size, device=device)
    static_output = torch.randn(batch_size, output_size, device=device)
    
    # 创建 NPU Graph
    print("捕获 NPU Graph...")
    stream = torch_npu.npu.Stream()
    graph = torch_npu.npu.NPUGraph()
    
    with torch_npu.npu.graph(graph, stream=stream):
        temp = static_input
        temp = model.linear1(temp)
        temp = model.relu(temp)
        temp = model.linear2(temp)
        static_output.copy_(temp)
    
    print("NPU Graph 捕获完成")
    
    # 性能测试
    num_iterations = 50
    test_input = torch.randn(batch_size, input_size, device=device)
    
    # 未使用 NPU Graph
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(test_input)
    uncompiled_time = time.time() - start_time
    
    # 使用 NPU Graph
    start_time = time.time()
    for _ in range(num_iterations):
        static_input.copy_(test_input)
        graph.replay()
        result = static_output.clone()
    compiled_time = time.time() - start_time
    
    print(f"未使用 NPU Graph: {uncompiled_time:.4f}s")
    print(f"使用 NPU Graph: {compiled_time:.4f}s")
    print(f"加速比: {uncompiled_time / compiled_time:.2f}x")
    
    return graph

npu_graph = example_npu_graphs()

# ============================================================================
# make_graphed_callables 对比
# ============================================================================

print("\n" + "=" * 80)
print("make_graphed_callables 对比")
print("=" * 80)

def example_make_graphed_callables_cuda():
    """
    CUDA make_graphed_callables 示例
    """
    if not cuda_available:
        print("跳过: CUDA 不可用")
        return None
    
    print("\n--- CUDA make_graphed_callables ---")
    
    device = torch.device("cuda")
    model = SimpleModel().to(device)
    model.eval()
    
    batch_size = 32
    input_size = 1024
    sample_input = torch.randn(batch_size, input_size, device=device)
    
    print("创建 graphed callable...")
    graphed_model = torch.cuda.make_graphed_callables(model, [sample_input])
    
    print("CUDA Graph 创建完成")
    
    # 性能测试
    num_iterations = 50
    test_input = torch.randn(batch_size, input_size, device=device)
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(test_input)
    uncompiled_time = time.time() - start_time
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = graphed_model(test_input)
    compiled_time = time.time() - start_time
    
    print(f"未使用: {uncompiled_time:.4f}s")
    print(f"使用 make_graphed_callables: {compiled_time:.4f}s")
    print(f"加速比: {uncompiled_time / compiled_time:.2f}x")
    
    return graphed_model

def example_make_graphed_callables_npu():
    """
    NPU make_graphed_callables 示例
    """
    if not npu_available:
        print("跳过: NPU 不可用")
        return None
    
    print("\n--- NPU make_graphed_callables ---")
    
    import torch_npu
    
    device = torch.device("npu")
    model = SimpleModel().to(device)
    model.eval()
    
    batch_size = 32
    input_size = 1024
    sample_input = torch.randn(batch_size, input_size, device=device)
    
    print("创建 graphed callable...")
    graphed_model = torch_npu.npu.make_graphed_callables(model, [sample_input])
    
    print("NPU Graph 创建完成")
    
    # 性能测试
    num_iterations = 50
    test_input = torch.randn(batch_size, input_size, device=device)
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(test_input)
    uncompiled_time = time.time() - start_time
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = graphed_model(test_input)
    compiled_time = time.time() - start_time
    
    print(f"未使用: {uncompiled_time:.4f}s")
    print(f"使用 make_graphed_callables: {compiled_time:.4f}s")
    print(f"加速比: {uncompiled_time / compiled_time:.2f}x")
    
    return graphed_model

cuda_graphed = example_make_graphed_callables_cuda()
npu_graphed = example_make_graphed_callables_npu()

# ============================================================================
# torch.compile 对比
# ============================================================================

print("\n" + "=" * 80)
print("torch.compile 对比")
print("=" * 80)

def example_torch_compile_cuda():
    """
    CUDA torch.compile 示例
    """
    if not cuda_available:
        print("跳过: CUDA 不可用")
        return None
    
    print("\n--- CUDA torch.compile ---")
    
    device = torch.device("cuda")
    model = SimpleModel().to(device)
    model.eval()
    
    batch_size = 32
    input_size = 1024
    x = torch.randn(batch_size, input_size, device=device)
    
    try:
        print("编译模型...")
        compiled_model = torch.compile(model, backend="cudagraphs")
        
        print("Warmup...")
        with torch.no_grad():
            output = compiled_model(x)
        
        # 性能测试
        num_iterations = 50
        
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
        
        print(f"未编译: {uncompiled_time:.4f}s")
        print(f"编译后: {compiled_time:.4f}s")
        print(f"加速比: {uncompiled_time / compiled_time:.2f}x")
        
        return compiled_model
    except Exception as e:
        print(f"错误: {e}")
        return None

def example_torch_compile_npu():
    """
    NPU torch.compile 示例
    """
    if not npu_available:
        print("跳过: NPU 不可用")
        return None
    
    print("\n--- NPU torch.compile ---")
    
    import torch_npu
    
    device = torch.device("npu")
    model = SimpleModel().to(device)
    model.eval()
    
    batch_size = 32
    input_size = 1024
    x = torch.randn(batch_size, input_size, device=device)
    
    try:
        print("编译模型...")
        compiled_model = torch.compile(model, backend="npugraphs")
        
        print("Warmup...")
        with torch.no_grad():
            output = compiled_model(x)
        
        # 性能测试
        num_iterations = 50
        
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
        
        print(f"未编译: {uncompiled_time:.4f}s")
        print(f"编译后: {compiled_time:.4f}s")
        print(f"加速比: {uncompiled_time / compiled_time:.2f}x")
        
        return compiled_model
    except Exception as e:
        print(f"错误: {e}")
        return None

cuda_compiled = example_torch_compile_cuda()
npu_compiled = example_torch_compile_npu()

# ============================================================================
# API 对比总结
# ============================================================================

print("\n" + "=" * 80)
print("API 对比总结")
print("=" * 80)

ComparisonTable = """
+----------------------------+--------------------------+--------------------------+
| 功能                        | CUDA Graphs          | NPU Graphs           |
+----------------------------+--------------------------+--------------------------+
| 上下文管理器                | torch.cuda.graph()    | torch_npu.npu.graph() |
| 图对象                      | torch.cuda.CUDAGraph()| torch_npu.npu.NPUGraph()|
| 高级 API                    | make_graphed_callables | make_graphed_callables |
| 编译后端                    | backend="cudagraphs"  | backend="npugraphs"   |
| 捕获开始                    | cudaGraphCaptureBegin | aclmdlRICaptureBegin   |
| 捕获结束                    | cudaGraphCaptureEnd   | aclmdlRICaptureEnd     |
| 执行图                      | cudaGraphLaunch       | aclmdlRIExecuteAsync   |
+----------------------------+--------------------------+--------------------------+
"""

print(ComparisonTable)

# ============================================================================
# 实现原理对比
# ============================================================================

print("\n" + "=" * 80)
print("实现原理对比")
print("=" * 80)

ImplementationComparison = """
CUDA Graphs 实现:
├── CUDA Runtime API
│   ├── cudaGraphCaptureBegin()
│   ├── cudaGraphCaptureEnd()
│   ├── cudaGraphInstantiate()
│   └── cudaGraphLaunch()
├── 内存管理
│   └── 静态内存池
└── 优化机制
    ├── 消除 kernel launch 开销
    ├── 减少 CPU-GPU 交互
    └── 预先分配内存

NPU Graphs 实现:
├── ACL (Ascend Computing Language) API
│   ├── aclmdlRICaptureBegin()
│   ├── aclmdlRICaptureEnd()
│   ├── aclmdlRIExecuteAsync()
│   └── aclmdlRIInstantiate()
├── 内存管理
│   └── NPUCachingAllocator
└── 优化机制
    ├── 消除 kernel launch 开销
    ├── 减少 CPU-NPU 交互
    └── 预先分配内存
"""

print(ImplementationComparison)

# ============================================================================
# Mermaid 时序图对比
# ============================================================================

print("\n" + "=" * 80)
print("Mermaid 时序图对比")
print("=" * 80)

# CUDA Graphs 时序图
CUDAGraphsMermaid = """
```mermaid
sequenceDiagram
    participant CPU as CPU (Python)
    participant CUDA as CUDA Runtime
    participant GPU as GPU (CUDA)
    
    Note over CPU,GPU: CUDA Graphs 捕获阶段
    
    CPU->>CUDA: cudaGraphCaptureBegin()
    activate GPU
    GPU->>GPU: 进入捕获模式
    deactivate GPU
    GPU-->>CUDA: 模式已激活
    
    CPU->>CUDA: 记录操作
    activate GPU
    GPU->>GPU: cuLaunchKernel (记录到图)
    deactivate GPU
    GPU-->>CUDA: 已记录
    
    CPU->>CUDA: cudaGraphCaptureEnd()
    activate GPU
    GPU->>GPU: 退出捕获模式
    deactivate GPU
    GPU-->>CUDA: graph 对象
    
    CPU->>CUDA: cudaGraphInstantiate()
    activate GPU
    GPU->>GPU: 分配资源
    deactivate GPU
    GPU-->>CUDA: graphExec 对象
    
    Note over CPU,GPU: CUDA Graphs 执行阶段
    
    CPU->>CUDA: cudaGraphLaunch()
    activate GPU
    GPU->>GPU: 执行整个图
    deactivate GPU
    GPU-->>CUDA: 执行完成
```
"""

# NPU Graphs 时序图
NPUGraphsMermaid = """
```mermaid
sequenceDiagram
    participant CPU as CPU (Python)
    participant ACL as ACL Runtime
    participant NPU as NPU Device
    
    Note over CPU,NPU: NPU Graphs 捕获阶段
    
    CPU->>ACL: aclmdlRICaptureBegin()
    activate NPU
    NPU->>NPU: 进入捕获模式
    deactivate NPU
    NPU-->>ACL: 模式已激活
    
    CPU->>ACL: 记录操作
    activate NPU
    NPU->>NPU: aclopExecute (记录到图)
    deactivate NPU
    NPU-->>ACL: 已记录
    
    CPU->>ACL: aclmdlRICaptureEnd()
    activate NPU
    NPU->>NPU: 退出捕获模式
    deactivate NPU
    NPU-->>ACL: graph 对象
    
    CPU->>ACL: aclmdlRIInstantiate()
    activate NPU
    NPU->>NPU: 分配资源
    deactivate NPU
    NPU-->>ACL: graphExec 对象
    
    Note over CPU,NPU: NPU Graphs 执行阶段
    
    CPU->>ACL: aclmdlRIExecuteAsync()
    activate NPU
    NPU->>NPU: 执行整个图
    deactivate NPU
    NPU-->>ACL: 执行完成
```
"""

print("\nCUDA Graphs 时序图:")
print(CUDAGraphsMermaid)

print("\nNPU Graphs 时序图:")
print(NPUGraphsMermaid)

# ============================================================================
# 总结
# ============================================================================

print("\n" + "=" * 80)
print("总结")
print("=" * 80)

Summary = """
CUDA Graphs 和 NPU Graphs 的主要区别:

1. **硬件平台**
   - CUDA Graphs: NVIDIA GPU
   - NPU Graphs: 华为昇腾 NPU

2. **底层 API**
   - CUDA Graphs: CUDA Runtime API
   - NPU Graphs: ACL (Ascend Computing Language) API

3. **Python 库**
   - CUDA Graphs: PyTorch (torch.cuda)
   - NPU Graphs: torch_npu (torch_npu.npu)

4. **API 设计**
   - 两者 API 设计高度相似
   - 都提供上下文管理器、图对象、高级 API
   - 都支持静态内存池和图重放

5. **性能优化**
   - 两者都通过预先记录计算图来优化性能
   - 都消除 kernel launch 开销
   - 都减少 CPU-GPU/NPU 交互

使用建议:
- 在 NVIDIA GPU 上使用 CUDA Graphs
- 在华为昽腾 NPU 上使用 NPU Graphs
- API 设计相似，代码可以复用
"""

print(Summary)

print("\n" + "=" * 80)
print("所有示例运行完成!")
print("=" * 80)
