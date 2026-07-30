"""
PyTorch CUDA Graphs 完整示例脚本
包含所有5种使用方式的可运行代码
"""

import torch
import torch.nn as nn
import time
import sys

print("=" * 100)
print("PyTorch CUDA Graphs 完整示例")
print("=" * 100)
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
print("=" * 100)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cpu":
    print("\n警告: CUDA 不可用，将在 CPU 上运行（CUDA Graphs 需要 CUDA）")
    print("部分示例可能无法运行\n")


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


def benchmark_model(model_fn, input_tensor, num_iterations=50, warmup=5):
    """基准测试函数"""
    model_fn.eval()
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    for _ in range(warmup):
        with torch.no_grad():
            _ = model_fn(input_tensor)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model_fn(input_tensor)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    return time.time() - start_time


print("\n" + "=" * 100)
print("方式1: torch.compile(backend='cudagraphs')")
print("=" * 100)

def example_1_cudagraphs_backend():
    """
    方式1: 使用 torch.compile(model_fn, backend="cudagraphs")
    
    实现原理:
    - cudagraphs backend 是 PyTorch 提供的专用后端
    - 自动捕获模型的计算图并转换为 CUDA Graph
    - 消除 CPU-GPU 交互开销
    - 预先分配所有需要的内存
    
    适用场景:
    - 形状固定的静态图
    - 推理场景
    - 简单模型优化
    """
    if device.type == "cpu":
        print("跳过: 需要 CUDA")
        return None
    
    print("\n--- 示例1: cudagraphs backend ---")
    
    model = SimpleModel().to(device)
    model.eval()
    
    batch_size = 32
    input_size = 1024
    x = torch.randn(batch_size, input_size, device=device)
    
    try:
        print("编译模型 (backend='cudagraphs')...")
        compiled_model = torch.compile(model, backend="cudagraphs")
        
        print("Warmup (捕获 CUDA Graph)...")
        with torch.no_grad():
            output = compiled_model(x)
        
        num_iterations = 50
        
        print("基准测试...")
        uncompiled_time = benchmark_model(model, x, num_iterations)
        compiled_time = benchmark_model(compiled_model, x, num_iterations)
        
        print(f"\n结果:")
        print(f"  未编译模型: {uncompiled_time:.4f}s ({num_iterations} 次迭代)")
        print(f"  编译模型 (cudagraphs): {compiled_time:.4f}s ({num_iterations} 次迭代)")
        print(f"  加速比: {uncompiled_time / compiled_time:.2f}x")
        
        return compiled_model
        
    except Exception as e:
        print(f"错误: {e}")
        print("可能原因: cudagraphs backend 在当前 PyTorch 版本中不可用")
        return None


example_1_result = example_1_cudagraphs_backend()


print("\n" + "=" * 100)
print("方式2: torch.compile(backend='inductor', mode='reduce-overhead')")
print("=" * 100)

def example_2_inductor_reduce_overhead():
    """
    方式2: 使用 torch.compile(model_fn, backend="inductor", mode="reduce-overhead")
    
    实现原理:
    - Inductor 是 PyTorch 2.0 的默认编译后端
    - "reduce-overhead" 模式专门优化推理性能
    - 自动使用 CUDA Graphs（当条件满足时）
    - 生成优化的 Triton kernels
    
    优化机制:
    - AOT 编译: 提前编译所有算子
    - Kernel fusion: 融合多个操作
    - CUDA Graphs: 对静态形状子图使用 CUDA Graphs
    - 内存规划: 预先分配内存
    
    适用场景:
    - 推理场景（inference）
    - 批量大小固定
    - 需要最大性能
    """
    print("\n--- 示例2: inductor backend with reduce-overhead mode ---")
    
    model = SimpleModel().to(device)
    model.eval()
    
    batch_size = 32
    input_size = 1024
    x = torch.randn(batch_size, input_size, device=device)
    
    try:
        print("编译模型 (backend='inductor', mode='reduce-overhead')...")
        compiled_model = torch.compile(
            model,
            backend="inductor",
            mode="reduce-overhead"
        )
        
        print("Warmup (触发编译)...")
        with torch.no_grad():
            output = compiled_model(x)
        
        num_iterations = 50
        
        print("基准测试...")
        uncompiled_time = benchmark_model(model, x, num_iterations)
        compiled_time = benchmark_model(compiled_model, x, num_iterations)
        
        print(f"\n结果:")
        print(f"  未编译模型: {uncompiled_time:.4f}s ({num_iterations} 次迭代)")
        print(f"  编译模型 (inductor+reduce-overhead): {compiled_time:.4f}s ({num_iterations} 次迭代)")
        print(f"  加速比: {uncompiled_time / compiled_time:.2f}x")
        
        return compiled_model
        
    except Exception as e:
        print(f"错误: {e}")
        print("可能原因: inductor backend 在当前 PyTorch 版本中不可用")
        return None


example_2_result = example_2_inductor_reduce_overhead()


print("\n" + "=" * 100)
print("方式3: torch.cuda.graph() 上下文管理器")
print("=" * 100)

def example_3_cuda_graph_context():
    """
    方式3: 使用 torch.cuda.graph() 上下文管理器
    
    实现原理:
    - 最低级别的 CUDA Graphs API
    - 需要手动管理输入/输出的静态内存池
    - 流程:
      1. 创建静态内存池
      2. 在 graph() 上下文中执行一次，记录所有操作
      3. 后续调用时，复制输入 -> replay -> 复制输出
    
    优化机制:
    - 完全消除 kernel launch 开销
    - 预先分配所有内存
    - 一次性提交整个计算图
    
    适用场景:
    - 需要精细控制内存
    - 自定义推理 pipeline
    - 需要与其他 CUDA 操作同步
    
    注意事项:
    - 必须使用静态内存
    - 输入形状必须固定
    - 需要手动管理内存同步
    """
    if device.type == "cpu":
        print("跳过: 需要 CUDA")
        return None
    
    print("\n--- 示例3: torch.cuda.graph() 上下文管理器 ---")
    
    model = SimpleModel().to(device)
    model.eval()
    
    batch_size = 32
    input_size = 1024
    output_size = 1024
    
    try:
        print("创建静态内存池...")
        static_input = torch.randn(batch_size, input_size, device=device)
        static_output = torch.randn(batch_size, output_size, device=device)
        
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
        
        num_iterations = 50
        test_input = torch.randn(batch_size, input_size, device=device)
        
        print("基准测试...")
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(test_input)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        uncompiled_time = time.time() - start_time
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        start_time = time.time()
        for _ in range(num_iterations):
            static_input.copy_(test_input)
            graph.replay()
            result = static_output.clone()
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        compiled_time = time.time() - start_time
        
        print(f"\n结果:")
        print(f"  未使用 CUDA Graph: {uncompiled_time:.4f}s ({num_iterations} 次迭代)")
        print(f"  使用 CUDA Graph: {compiled_time:.4f}s ({num_iterations} 次迭代)")
        print(f"  加速比: {uncompiled_time / compiled_time:.2f}x")
        
        return graph
        
    except Exception as e:
        print(f"错误: {e}")
        print("可能原因: torch.cuda.graph() 需要 CUDA 10+")
        return None


example_3_result = example_3_cuda_graph_context()


print("\n" + "=" * 100)
print("方式4: torch.cuda.make_graphed_callables")
print("=" * 100)

def example_4_make_graphed_callables():
    """
    方式4: 使用 torch.cuda.make_graphed_callables
    
    实现原理:
    - PyTorch 提供的高级 API
    - 将多个 callable 包装为 CUDA Graphs
    - 自动管理内存池和输入/输出
    
    优化机制:
    - 自动创建和管理静态内存池
    - 支持多个独立的 callable
    - 自动处理输入/输出的复制
    
    适用场景:
    - 需要同时优化多个函数
    - 不想手动管理内存
    - 需要更高级的抽象
    
    注意事项:
    - 需要 PyTorch 1.10+
    - 输入形状必须固定
    """
    if device.type == "cpu":
        print("跳过: 需要 CUDA")
        return None
    
    print("\n--- 示例4: torch.cuda.make_graphed_callables ---")
    
    model = SimpleModel().to(device)
    model.eval()
    
    batch_size = 32
    input_size = 1024
    sample_input = torch.randn(batch_size, input_size, device=device)
    
    try:
        print("创建 graphed callable...")
        graphed_model = torch.cuda.make_graphed_callables(model, [sample_input])
        
        print("CUDA Graph 创建完成")
        
        num_iterations = 50
        test_input = torch.randn(batch_size, input_size, device=device)
        
        print("基准测试...")
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(test_input)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        uncompiled_time = time.time() - start_time
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = graphed_model(test_input)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        compiled_time = time.time() - start_time
        
        print(f"\n结果:")
        print(f"  未使用 CUDA Graph: {uncompiled_time:.4f}s ({num_iterations} 次迭代)")
        print(f"  使用 make_graphed_callables: {compiled_time:.4f}s ({num_iterations} 次迭代)")
        print(f"  加速比: {uncompiled_time / compiled_time:.2f}x")
        
        return graphed_model
        
    except Exception as e:
        print(f"错误: {e}")
        print("可能原因: make_graphed_callables 需要 PyTorch 1.10+ 和 CUDA")
        return None


example_4_result = example_4_make_graphed_callables()


print("\n" + "=" * 100)
print("方式5: experimental 参数")
print("=" * 100)

def example_5_experimental_cudagraphs():
    """
    方式5: 使用 torch.compile 的 experimental 参数
    
    实现原理:
    - PyTorch 2.1+ 提供的 experimental 参数
    - 通过 enable_cuda_graph=True 显式启用 CUDA Graphs
    - 可以配置 CUDA Graphs 的捕获策略
    
    优化机制:
    - 与 inductor backend 集成
    - 提供更细粒度的控制
    - 支持动态形状的部分优化
    
    适用场景:
    - 需要精细控制 CUDA Graphs 行为
    - 实验性功能测试
    - 高级性能调优
    
    注意事项:
    - 实验性 API，可能不稳定
    - 需要 PyTorch 2.1+
    """
    print("\n--- 示例5: torch.compile experimental 参数 ---")
    
    model = SimpleModel().to(device)
    model.eval()
    
    batch_size = 32
    input_size = 1024
    x = torch.randn(batch_size, input_size, device=device)
    
    try:
        print("编译模型 (experimental={'enable_cuda_graph': True})...")
        compiled_model = torch.compile(
            model,
            backend="inductor",
            mode="reduce-overhead",
            experimental={"enable_cuda_graph": True}
        )
        
        print("Warmup (触发编译)...")
        with torch.no_grad():
            output = compiled_model(x)
        
        num_iterations = 50
        
        print("基准测试...")
        uncompiled_time = benchmark_model(model, x, num_iterations)
        compiled_time = benchmark_model(compiled_model, x, num_iterations)
        
        print(f"\n结果:")
        print(f"  未编译模型: {uncompiled_time:.4f}s ({num_iterations} 次迭代)")
        print(f"  编译模型 (experimental CUDA Graphs): {compiled_time:.4f}s ({num_iterations} 次迭代)")
        print(f"  加速比: {uncompiled_time / compiled_time:.2f}x")
        
        return compiled_model
        
    except Exception as e:
        print(f"错误: {e}")
        print("可能原因: experimental 参数需要 PyTorch 2.1+")
        return None


example_5_result = example_5_experimental_cudagraphs()


print("\n" + "=" * 100)
print("总结")
print("=" * 100)

Summary = """
方式对比:

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃ 使用方式                      ┃ 易用性 ┃ 灵活性 ┃ 性能   ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━╋━━━━━━━━╋━━━━━━━━┫
┃ backend="cudagraphs"          ┃   ⭐⭐⭐⭐⭐ ┃  ⭐⭐⭐ ┃  ⭐⭐⭐⭐ ┃
┃                                ┃(自动) ┃(固定) ┃(高)   ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━╋━━━━━━━━╋━━━━━━━━┫
┃ backend="inductor"             ┃   ⭐⭐⭐⭐⭐ ┃ ⭐⭐⭐⭐⭐ ┃ ⭐⭐⭐⭐⭐ ┃
┃ mode="reduce-overhead"        ┃(自动) ┃(灵活) ┃(很高) ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━╋━━━━━━━━╋━━━━━━━━┫
┃ torch.cuda.graph()            ┃   ⭐⭐ ┃ ⭐⭐⭐⭐⭐ ┃ ⭐⭐⭐⭐⭐ ┃
┃ 上下文管理器                  ┃(手动) ┃(完全) ┃(很高) ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━╋━━━━━━━━╋━━━━━━━━┫
┃ make_graphed_callables         ┃  ⭐⭐⭐⭐ ┃  ⭐⭐⭐⭐ ┃  ⭐⭐⭐⭐ ┃
┃                                ┃(半自动)┃(多函数)┃(高)   ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━╋━━━━━━━━╋━━━━━━━━┫
┃ experimental 参数              ┃  ⭐⭐⭐ ┃ ⭐⭐⭐⭐⭐ ┃ ⭐⭐⭐⭐⭐ ┃
┃                                ┃(高级) ┃(细粒度)┃(很高) ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━┻━━━━━━━━┻━━━━━━━━┛

使用建议:

1. 初学者/快速原型:
   → 使用 backend="cudagraphs"
   最简单，自动处理大部分细节

2. 生产环境/推理优化:
   → 使用 backend="inductor" + mode="reduce-overhead"
   PyTorch 2.0 推荐，自动选择最优策略

3. 高级用户/精细控制:
   → 使用 torch.cuda.graph()
   完全控制内存和执行

4. 多函数优化:
   → 使用 make_graphed_callables
   同时优化多个函数，自动内存管理

5. 实验性功能:
   → 使用 experimental 参数
   探索最新优化，细粒度控制

注意事项:

- CUDA Graphs 需要 CUDA 10+ 和支持 CUDA Graphs 的 GPU
- 输入形状必须固定（部分方式支持动态形状）
- 不支持动态控制流（if/else、循环等）
- 首次运行会有额外的编译/捕获开销
- 适合推理场景，训练场景使用较少
"""

print(Summary)

print("\n" + "=" * 100)
print("所有示例运行完成!")
print("=" * 100)
