# PyTorch CUDA Graphs 代码调用流程时序图汇总

本文档包含所有5种 CUDA Graphs 使用方式的详细 Mermaid 时序图。

---

## 目录

1. [方式1: torch.compile(backend="cudagraphs")](#方式1-torchcompilebackendcudagraphs)
2. [方式2: torch.compile(backend="inductor", mode="reduce-overhead")](#方式2-torchcompilebackendinductor-modereduce-overhead)
3. [方式3: torch.cuda.graph() 上下文管理器](#方式3-torchcudagraph-上下文管理器)
4. [方式4: torch.cuda.make_graphed_callables](#方式4-torchcudamake_graphed_callables)
5. [方式5: experimental 参数](#方式5-experimental-参数)

---

## 方式1: torch.compile(backend="cudagraphs")

### 完整时序图

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

### 性能优势总结

**传统执行 (每次调用):**
```
CPU → Launch Kernel 1 → GPU → CPU → Launch Kernel 2 → GPU → CPU → Launch Kernel 3 → GPU
     (开销)              (等待)    (开销)              (等待)    (开销)              (等待)
```

**CUDA Graphs 执行 (后续调用):**
```
CPU → Replay CUDA Graph → GPU (执行所有操作)
     (一次开销)           (连续执行)
```

**优化效果:**
- ✓ 消除多次 CPU-GPU 交互
- ✓ 消除多次 kernel launch 开销
- ✓ 消除多次驱动程序调用
- ✓ 提高 GPU 利用率
- ✓ 减少总执行时间 30-70%

---

## 方式2: torch.compile(backend="inductor", mode="reduce-overhead")

### 完整时序图

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

### 优化层级总结

**Level 1: Kernel Fusion**
```
matmul + bias + relu → fused_kernel
✓ 减少内存访问
✓ 减少 kernel launch
```

**Level 2: Triton Kernels**
```
自动优化 tiling、向量化
✓ 适应不同硬件
✓ 高度优化
```

**Level 3: CUDA Graphs**
```
对静态子图使用 CUDA Graphs
✓ 消除 kernel launch 开销
✓ 连续执行
```

**Level 4: 内存优化**
```
预先分配内存池
✓ 减少运行时分配
✓ 提高缓存命中率
```

**总优化效果: 2-5x 加速**

---

## 方式3: torch.cuda.graph() 上下文管理器

### 完整时序图

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

### 关键特性总结

**静态内存管理:**
- ✓ 所有内存预先分配
- ✓ 避免运行时内存分配
- ✓ 提高内存访问效率

**完全控制:**
- ✓ 手动管理输入/输出
- ✓ 精确控制执行流程
- ✓ 支持自定义同步

**高性能:**
- ✓ 一次性提交整个图
- ✓ 消除所有 kernel launch 开销
- ✓ 连续执行所有操作

**适用场景:**
- ✓ 需要精细控制的场景
- ✓ 自定义推理 pipeline
- ✓ 与其他 CUDA 操作同步

---

## 方式4: torch.cuda.make_graphed_callables

### 完整时序图

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

### 优势总结

**自动化:**
- ✓ 自动创建静态内存池
- ✓ 自动管理输入/输出
- ✓ 无需手动内存管理

**多函数支持:**
- ✓ 同时优化多个函数
- ✓ 每个函数独立的 graph
- ✓ 可以组合使用

**简单易用:**
- ✓ 比 torch.cuda.graph() 更简单
- ✓ 类似于普通函数调用
- ✓ 隐藏底层细节

**适用场景:**
- ✓ 需要优化多个函数
- ✓ 不想手动管理内存
- ✓ 需要高级抽象

---

## 方式5: experimental 参数

### 完整时序图

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

### 优势总结

**智能捕获:**
- ✓✓ 自动识别可捕获子图
- ✓✓ 分析子图大小和形状
- ✓✓ 智能决策是否捕获

**混合执行:**
- ✓✓ 静态部分使用 CUDA Graphs
- ✓✓ 动态部分使用常规执行
- ✓✓ 最佳性能平衡

**细粒度控制:**
- ✓✓ 配置捕获步数
- ✓✓ 配置最小/最大图大小
- ✓✓ 调试选项

**实验性功能:**
- ✓✓ 访问最新优化技术
- ✓✓ 探索性功能
- ⚠️ API 可能不稳定

**适用场景:**
- ✓✓ 需要精细控制
- ✓✓ 混合静态/动态模型
- ✓✓ 实验性功能测试

---

## 总结

### 时序图对比

| 方式 | 捕获复杂度 | 执行复杂度 | 灵活性 | 性能 |
|------|-----------|-----------|--------|------|
| backend="cudagraphs" | 低 | 低 | 中 | 高 |
| backend="inductor" + reduce-overhead | 中 | 低 | 高 | 很高 |
| torch.cuda.graph() | 高 | 中 | 很高 | 很高 |
| make_graphed_callables | 中 | 低 | 中 | 高。 |
| experimental 参数 | 中 | 低 | 很高 | 很高 |

### 关键执行流程对比

**方式1 (cudagraphs):**
```
首次: warmup → 捕获 → 实例化
后续: 复制输入 → replay → 复制输出
```

**方式2 (inductor):**
```
首次: 导出FX → lowering → 编译 → 捕获子图
后续: 执行Triton → replay子图
```

**方式3 (context manager):**
```
首次: 创建静态池 → 捕获 → 实例化
后续: 复制输入 → replay → 复制输出
```

**方式4 (make_graphed):**
```
首次: 分析 → 创建静态池 → 捕获
后续: 自动I/O → replay
```

**方式5 (experimental):**
```
首次: 分析子图 →) 智能捕获 → 混合编译
后续: 动态部分 → 静态部分
```

### 性能优化层级

```
Level 0: 原始 PyTorch
  └─ 基础执行

Level 1: Kernel Fusion
  └─ 融合多个操作

Level 2: Triton Kernels
  └─ 自动优化

Level 3: CUDA Graphs
  └─ 消除 kernel launch

Level 4: 内存优化
  └─ 预先分配内存

Level 5: 混合优化
  └─ 静态+动态结合
```

### 使用建议

1. **初学者**: 使用 `backend="cudagraphs"`
2. **生产环境**: 使用 `backend="inductor" + mode="reduce-overhead"`
3. **高级用户**: 使用 `torch.cuda.graph()`
4. **多函数**: 使用 `make_graphed_callables`
5. **实验性**: 使用 `experimental` 参数

## Related Pages

- [[02_engineering/01_ai_frameworks/index]]
- [[PyTorch_CUDA_Graphs_Complete_Guide]]
- [[SUMMARY]]
