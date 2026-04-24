# NPUGraphs 内存管理与复用机制深度解析

## 目录
1. [概述](#概述)
2. [核心架构组件](#核心架构组件)
3. [内存池管理](#内存池管理)
4. [Graph Tree 机制](#graph-tree-机制)
5. [Capture 与 Replay 流程](#capture-与-replay-流程)
6. [内存复用策略](#内存复用策略)
7. [关键代码解析](#关键代码解析)

---

## 概述

NPUGraphs 是 PyTorch NPU 后端实现的一套图捕获与重放机制，类似于 CUDA Graphs。其核心目标是通过捕获 NPU 操作序列并在后续迭代中重放，从而消除 CPU 开销，提升推理性能。

**关键特性：**
- **内存池共享**：多个 Graph 可以共享同一个内存池，减少内存碎片
- **Graph Tree 结构**：支持任意树状结构的 Graph 执行路径，而非严格的线性顺序
- **动态重录**：支持在不同路径间切换时自动重新录制 Graph
- **内存复用**：通过精细的 liveness tracking 实现内存高效复用

---

## 核心架构组件

### 1. NPUGraph (C++ 层)

```cpp
// torch_npu/csrc/core/npu/NPUGraph.h
struct TORCH_NPU_API NPUGraph {
    // 核心状态
    aclmdlRI model_ri_ = nullptr;           // 昇腾模型实例
    bool has_graph_exec_ = false;           // 是否成功捕获
    MempoolId_t mempool_id_;                // 内存池标识
    NPUStream capture_stream_;              // 捕获使用的流
    int capture_dev_;                       // 捕获设备
    
    // 主要接口
    void capture_begin(MempoolId_t pool, aclmdlRICaptureMode mode, bool report_shape);
    void capture_end();
    void replay();
    void reset();
};
```

**职责**：
- 管理单个 Graph 的生命周期（捕获 → 重放 → 销毁）
- 维护与昇腾底层（ACL）的交互（`aclmdlRICaptureBegin/End`）
- 管理 Graph 私有的内存池

### 2. NPUCachingAllocator (C++ 层)

```cpp
// torch_npu/csrc/core/npu/NPUCachingAllocator.cpp
struct PrivatePool {
    int use_count{ 1 };           // 活跃 Graph 引用计数
    int npuMalloc_count{ 0 };     // 未释放的分配计数
    BlockPool large_blocks;       // 大块内存池 (>1MB)
    BlockPool small_blocks;       // 小块内存池 (≤1MB)
};
```

**核心机制**：
- **Block Pool**：按大小分别管理 large/small 内存块
- **Private Pool**：每个 Graph 或共享池有独立的内存管理空间
- **ExpandableSegment**：支持动态扩展的内存段（类似 mmap）

### 3. NPUGraphTreeManager (Python 层)

```python
# torch_npu/npu/_graph_tree.py
class NPUGraphTreeManager:
    """
    管理整个 Graph Tree 的生命周期
    - 维护从根到当前节点的路径
    - 管理不同路径间的状态切换
    - 处理内存池的 checkpoint 和恢复
    """
    
    def __init__(self, device_index: int):
        self.device_index = device_index
        self.current_node: Optional[NPUGraphNode] = None
        self.current_generation = 0
        self.checkpointed_caching_state: Optional[AllocatorState] = None
```

### 4. NPUGraphNode (Python 层)

```python
# torch_npu/npu/_graph_tree.py
class NPUGraphNode:
    """
    Graph Tree 中的单个节点
    封装了一个 NPU Graph 的录制和重放
    """
    
    def __init__(
        self,
        wrapped_function: WrappedFunction,
        graph_id: GraphID,
        parent: Optional[NPUGraphNode],
        inputs: List[Tensor],
        npu_graphs_pool: Tuple[int, int],
        device_index: int,
        stack_traces: Optional[StackTraces],
        stream: torch.npu.Stream,
    ):
        # 核心属性
        self.wrapped_function = wrapped_function      # 被封装的函数
        self.id = graph_id                            # 唯一标识
        self.parent = parent                            # 父节点（路径前缀）
        self.children: Dict[FunctionID, List[NPUGraphNode]] = defaultdict(list)  # 子节点
        
        # 内存相关
        self.npu_graphs_pool = npu_graphs_pool        # 共享内存池ID
        self.checkpointed_caching_state: Optional[AllocatorState] = None  # 分配器状态检查点
        
        # 输出跟踪
        self.outputs_weakrefs: List[Optional[StorageWeakRefWrapper]] = []  # 输出的弱引用
        self.tensor_weakrefs: List[Optional[TensorWeakRef]] = []         # Tensor弱引用
        
        # 静态输入追踪
        self.static_input_idxs: List[int] = []        # 静态输入索引
        self.static_input_data_ptrs: List[Optional[int]] = []  # 静态输入的数据指针
        
        # 别名追踪
        self.output_storage_alias: List[Optional[OutputAliasInfo]] = []  # 输出存储别名信息
        self.unaliased_in_all_paths: List[bool] = []                     # 在所有路径中是否未别名
        self.cached_tensor_outputs: List[Optional[Tensor]] = []          # 缓存的输出Tensor
```

---

## 内存池管理

### 内存池标识 (MempoolId_t)

```cpp
// MempoolId_t 是一个 pair<uint64_t, uint64_t>
using MempoolId_t = std::pair<uint64_t, uint64_t>;

// 创建新的内存池
MempoolId_t graph_pool_handle() {
    // 只设置第二个值，用于区分 capture_begin 创建的 id
    auto new_pool = c10_npu::MemPool();
    return new_pool.id();  // 返回 {0, unique_id}
}
```

**创建方式**：
1. **Graph 内部创建**：`capture_begin` 时如果不传 pool 参数，自动创建新的内存池
2. **显式创建**：调用 `graph_pool_handle()` 获取共享池标识

### 内存池生命周期

```cpp
// 在 capture_begin 中关联分配器与内存池
void NPUGraph::capture_begin(MempoolId_t pool, ...) {
    // ... 前置检查 ...
    
    // 确定内存池ID
    if (pool.first != 0 || pool.second != 0) {
        // 用户提供了共享池
        TORCH_INTERNAL_ASSERT(!(pool.first && pool.second));  // 只有一个非零
        mempool_id_ = pool;
    } else {
        // 创建新的私有池
        auto mempool = c10_npu::MemPool({}, false);
        mempool_id_ = mempool.id();
        TORCH_INTERNAL_ASSERT(mempool_id_.first > 0);
    }
    
    // 注册到分配器：开始将该设备的所有分配导向此内存池
    c10_npu::NPUCachingAllocator::beginAllocateToPool(
        capture_dev_, 
        mempool_id_, 
        [this](aclrtStream stream) {
            // 回调函数：检查该流是否处于活跃捕获状态
            aclmdlRICaptureStatus status;
            aclmdlRI model_ri;
            NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureGetInfo(stream, &status, &model_ri));
            return status == aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE 
                   && model_ri == model_ri_;
        }
    );
    
    // 开始捕获
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureBegin(capture_stream_, capture_mode));
    // ...
}

// 在 capture_end 中结束关联
void NPUGraph::capture_end() {
    // ... 结束捕获 ...
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureEnd(capture_stream_, &model_ri));
    
    // 结束分配器的内存池导向
    c10_npu::NPUCachingAllocator::endAllocateToPool(capture_dev_, mempool_id_);
    
    has_graph_exec_ = true;
    // ...
}
```

### 私有内存池 (PrivatePool)

```cpp
struct PrivatePool {
    int use_count{ 1 };              // 引用计数：活跃 Graph 数量
    int npuMalloc_count{ 0 };        // 未释放的分配计数
    BlockPool large_blocks;          // 大块池 (>1MB)
    BlockPool small_blocks;          // 小块池 (≤1MB)
    
    PrivatePool() : large_blocks(false, this), small_blocks(true, this) {}
};
```

**生命周期管理**：
- `use_count == 0` 且 `npuMalloc_count == 0` 时可安全销毁
- Graph 的 `reset()` 会减少引用并触发清理

---

## Graph Tree 机制

### 核心概念

NPU Graph Tree 是对传统线性 Graph 执行模式的重大改进，允许构建任意树状结构的 Graph 执行路径。

**传统模式的局限**：
```
A → B → C → D  (只能顺序执行，不能回退或分支)
```

**Graph Tree 的优势**：
```
       [Root]
       /    \
    [A]      [B]
    / \        \
 [C]   [D]     [E]
  |
 [F]
```
- 支持从任意节点分叉创建新路径
- 支持在不同分支间动态切换
- 内存占用最优：max(mem(A→C→F), mem(A→D), mem(B→E))

### 树节点结构

```python
class NPUGraphNode:
    def __init__(self, ...):
        self.wrapped_function: WrappedFunction  # 封装的计算函数
        self.id: GraphID                          # 唯一标识
        self.parent: Optional[NPUGraphNode]         # 父节点（路径前缀）
        self.children: Dict[FunctionID, List[NPUGraphNode]]  # 子节点分组
        
        self.npu_graphs_pool: Tuple[int, int]       # 共享内存池ID
        self.checkpointed_caching_state: AllocatorState  # 分配器状态检查点
```

### 路径管理与状态切换

```python
class NPUGraphTreeManager:
    def __init__(self, device_index: int):
        self.device_index = device_index
        self.current_node: Optional[NPUGraphNode] = None
        self.current_generation = 0
        
    def add_function(
        self,
        model: ModelType,
        inputs: List[InputType],
        static_input_idxs: Sequence[int],
        ...
    ) -> Tuple[ModelType, OutputType]:
        """添加新函数到 Graph Tree"""
        
        # 1. 确定执行路径：新节点 vs 已有节点
        existing_node = self._get_existing_node(inputs)
        
        if existing_node is not None:
            # 路径已存在，直接重放
            return self._replay_existing(existing_node, inputs)
        
        # 2. 需要创建新节点
        # 检查是否需要从当前路径分叉
        if self.current_node is not None:
            # 保存当前路径状态（checkpoint）
            self._checkpoint_current_state()
        
        # 3. 创建新节点
        new_node = self._create_node(
            model, inputs, static_input_idxs, parent=self.current_node
        )
        
        # 4. 如果是分叉路径，需要恢复或重建分配器状态
        if self.current_node is not None:
            self._restore_or_rebuild_state(new_node)
        
        # 5. 录制并执行
        outputs = self._record_and_run(new_node, inputs)
        
        # 6. 更新当前节点指针
        self.current_node = new_node
        self.current_generation += 1
        
        return outputs
```

### 内存池状态检查点 (Checkpoint)

```python
class NPUGraphNode:
    def _record(self, model, inputs):
        """录制 Graph 并保存分配器状态"""
        
        # 1. 创建 NPU Graph 对象
        self.graph = torch.npu.NPUGraph()
        
        # 2. 保存当前分配器状态（checkpoint）
        # 这是实现路径切换的关键！
        self.checkpointed_caching_state = self._get_allocator_state()
        
        # 3. 开始录制
        self.graph.capture_begin(pool=self.npu_graphs_pool)
        
        # 4. 执行函数（真正的 NPU 操作会被记录）
        outputs = model(inputs)
        
        # 5. 结束录制
        self.graph.capture_end()
        
        # 6. 记录输出元数据（用于重建 tensor）
        self.outputs_metadata = [self._tensor_metadata(o) for o in outputs]
        
        return outputs
    
    def _get_allocator_state(self) -> AllocatorState:
        """获取当前分配器的状态快照"""
        # 遍历所有 block，记录它们的分配状态
        # 包括：已分配 block、空闲 block、segment 结构等
        return torch_npu._C._npu_getAllocatorState()
    
    def restore_allocator_state(self, state: AllocatorState):
        """恢复分配器到之前保存的状态"""
        # 这在从其他分支切换回来时至关重要
        # 确保 allocator 的 book-keeping 与捕获时一致
        torch_npu._C._npu_restoreAllocatorState(state)
```

---

## Capture 与 Replay 流程

### Capture 流程详解

```cpp
// NPUGraph::capture_begin - C++ 层实现
void NPUGraph::capture_begin(MempoolId_t pool, aclmdlRICaptureMode capture_mode, bool report_shape) {
    // 1. 前置检查
    static const auto _task_queue_enable = c10_npu::option::OptionsManager::GetTaskQueueEnable();
    TORCH_CHECK(_task_queue_enable != 2, "Do not support TASK_QUEUE_ENABLE = 2 during NPU graph capture");
    
    TORCH_CHECK(!has_graph_exec_, "This NPUGraph instance already owns a captured graph");
    
    auto stream = c10_npu::getCurrentNPUStream();
    TORCH_CHECK(stream != c10_npu::getDefaultNPUStream(), 
                "NPU graphs must be captured on a non-default stream");
    
    // 2. 设置流属性（如 shape 报告）
    apply_cache_op_info(stream, report_shape);
    
    // 3. 注册随机数生成器状态
    auto* gen = at::get_generator_or_default<at_npu::NPUGeneratorImpl>(
        c10::nullopt, at_npu::detail::getDefaultNPUGenerator());
    gen->register_graph(this);
    
    // 4. 确定/创建内存池
    if (pool.first != 0 || pool.second != 0) {
        // 使用用户提供的共享池
        TORCH_INTERNAL_ASSERT(!(pool.first && pool.second));
        mempool_id_ = pool;
    } else {
        // 创建新的私有池
        auto mempool = c10_npu::MemPool({}, false);
        mempool_id_ = mempool.id();
        TORCH_INTERNAL_ASSERT(mempool_id_.first > 0);
    }
    
    // 5. 注册到分配器：将此设备的所有分配导向此内存池
    c10_npu::NPUCachingAllocator::beginAllocateToPool(
        capture_dev_, 
        mempool_id_, 
        [this](aclrtStream stream) {
            // 回调：检查流是否处于活跃捕获状态
            aclmdlRICaptureStatus status;
            aclmdlRI model_ri;
            NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureGetInfo(stream, &status, &model_ri));
            return status == aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE 
                   && model_ri == model_ri_;
        }
    );
    
    // 6. 开始 ACL 捕获
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureBegin(capture_stream_, capture_mode));
    
    // 7. 获取捕获信息
    aclmdlRICaptureStatus status;
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureGetInfo(stream, &status, &model_ri_));
    TORCH_INTERNAL_ASSERT(status == aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE);
}
```

### Replay 流程详解

```cpp
void NPUGraph::replay() {
    // 1. 前置检查
    TORCH_CHECK(has_graph_exec_, "Called NPUGraph::replay without a preceding successful capture");
    
    // 2. 设置设备上下文
    c10::OptionalDeviceGuard device_guard{capture_stream_.device()};
    
    // 3. 更新随机数生成器状态
    for (auto& [generator_state, wholegraph_increments] : captured_generator_states_) {
        generator_state->replay_prologue(wholegraph_increments);
    }
    
    // 4. 执行 ACL 重放（核心操作）
    // model_ri_ 可以在任意流上重放
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRIExecuteAsync(model_ri_, c10_npu::getCurrentNPUStream()));
}
```

### 关键差异：Capture vs Replay

| 方面 | Capture | Replay |
|------|---------|--------|
| **CPU 开销** | 高（首次执行所有操作） | 极低（直接提交预编译图） |
| **内存分配** | 实际执行分配 | 不复用分配逻辑，直接使用预分配内存 |
| **Kernel 启动** | 逐个启动 | 批量提交 |
| **适用场景** | 初始化/编译阶段 | 推理/训练迭代 |

---

## 内存复用策略

### 1. Liveness Tracking（活性追踪）

```python
# torch_npu/npu/_graph_tree.py

class NPUGraphNode:
    def __init__(self, ...):
        # ...
        
        # 记录录制前/后的活性状态
        # 每个节点维护其路径上所有节点的输出活性
        self.recorded_liveness_before_graph: LevelList[OutputList[bool]] = []
        self.recorded_liveness_after_graph: LevelList[OutputList[bool]] = []
        
        # 预期在录制前/后应该死亡的索引
        self.expected_dead_indices_before_graph: List[PathOutputIndex] = []
        self.expected_dead_indices_after_graph: List[PathOutputIndex] = []
        
        # 录制后仍存活的索引
        self.live_indices_after_graph: List[PathOutputIndex] = []
```

**活性追踪的工作流程**：

```python
def _record(self, model, inputs):
    # 1. 记录录制前的活性状态
    if self.parent is not None:
        previous_liveness = self.parent.recorded_liveness_after_graph
        curr_liveness = self._get_liveness(self.path_weakrefs)
        
        # 找出活性发生变化的索引
        different_indices = self._get_different_indices(previous_liveness, curr_liveness)
        
        self.recorded_liveness_before_graph = curr_liveness
        self.expected_dead_indices_before_graph = different_indices
    
    # 2. 执行录制
    self.graph.capture_begin(pool=self.npu_graphs_pool)
    outputs = model(inputs)
    self.graph.capture_end()
    
    # 3. 记录录制后的活性状态
    self.recorded_liveness_after_graph = self._get_liveness(self.path_weakrefs)
    self.expected_dead_indices_after_graph = ...
    
    return outputs
```

### 2. Storage Weak Reference（存储弱引用）

```python
class StorageWeakRefWrapper:
    """
    对 Storage 的弱引用包装
    用于追踪内存是否仍被使用，而不阻止垃圾回收
    """
    
    def __init__(self, inp: Union[Tensor, UntypedStorage]):
        if isinstance(inp, Tensor):
            stor = inp.untyped_storage()
        else:
            stor = inp
        
        # 创建弱引用
        self.ref = StorageWeakRef(stor)
        self._data_ptr = stor.data_ptr()
        self.extra_ref_check = None
    
    def expired(self) -> bool:
        """检查引用的 Storage 是否已被释放"""
        if self.extra_ref_check is not None and not self.extra_ref_check():
            return False
        
        # 检查 Storage 的引用计数
        stor_count = torch_npu._C._storage_Use_Count(self.ref.cdata)
        return (stor_count - (self.extra_ref_check is not None)) == 0
    
    def data_ptr(self) -> int:
        """返回数据指针（即使 Storage 已过期）"""
        return self._data_ptr
```

### 3. Alias Detection（别名检测）

```python
class OutputAliasInfo:
    """标记输出存储的别名关系"""
    pass

class _UnaliasedStorage(OutputAliasInfo):
    """未别名：输出创建了新存储或为 None"""
    pass

class AliasesPriorGraphOutput(OutputAliasInfo):
    """别名：输出是之前 Graph 输出的别名"""
    __slots__ = ["index"]
    index: PathOutputIndex  # (depth, offset) 路径索引

class AliasesNewOutput(OutputAliasInfo):
    """别名：输出是当前新输出中的某个索引的别名"""
    __slots__ = ["index"]
    index: int
```

**别名检测的重要性**：
- 决定输出是否可以被缓存
- 影响内存复用策略
- 防止过早释放被别名的内存

---

## 关键代码解析

### 1. TreeManager 生命周期管理

```python
class TreeManagerContainer:
    """
    管理 TreeManager 的生命周期
    确保只要有任何 Graph 或 Tensor 输出存活，Tree 就保持存活
    """
    
    def __init__(self, device_index: int):
        self.tree_manager: Optional[NPUGraphTreeManager] = None
        self.live_npugraphify_fns = 0        # 活跃的 graph 函数数量
        self.device_index = device_index
        self.live_storages_count = 0          # 存活的 storage 数量（Tensor 输出）
        self.graph: Optional[torch.npu.NPUGraph] = None  # 用于保持 pool 存活
        self.lock = threading.Lock()
    
    def add_strong_reference(self, fn: Callable[..., Any]) -> None:
        """增加引用计数，当 fn 被垃圾回收时自动减少"""
        with self.lock:
            self.live_npugraphify_fns += 1
        
        # 注册终结器：当 fn 被释放时自动调用 finalize_npugraphify_fn
        weakref.finalize(fn, self.finalize_npugraphify_fn)
    
    def finalize_npugraphify_fn(self) -> None:
        """Graph 函数被释放时的回调"""
        with self.lock:
            self.live_npugraphify_fns -= 1
            if self.live_npugraphify_fns == 0 and self.live_storages_count == 0:
                # 没有活跃引用，可以安全释放 tree_manager
                self.tree_manager = None
    
    def _finalize_tensor(self) -> None:
        """Tensor 输出被释放时的回调"""
        with self.lock:
            self.live_storages_count -= 1
            if self.live_storages_count == 0:
                self.graph = None  # 释放对 pool 的引用
                if self.live_npugraphify_fns == 0:
                    self.tree_manager = None
```

### 2. 分配器状态检查点

```cpp
// BlockState: 单个 block 的状态
struct BlockState {
  c10::DeviceIndex device = 0;
  aclrtStream stream = nullptr;
  stream_set stream_uses = {};
  size_t size = 0;
  void* ptr = nullptr;
  bool allocated = false;
  int64_t gc_count_base = 0;
  
  explicit BlockState(Block* block);
};

// SegmentState: 整个 segment（连续的 block 链）的状态
struct SegmentState {
  std::vector<BlockState> blocks;
  bool is_small = false;
  
  explicit SegmentState(Block* head);
};

// PrivatePoolState: 私有内存池的完整状态
struct PrivatePoolState : AllocatorState {
  MempoolId_t owner_id = {0, 0};
  std::vector<SegmentState> segments;
  
  PrivatePoolState(
      MempoolId_t pool_id,
      const std::vector<Block*>& private_pool_head_blocks);
};
```

**检查点的作用**：
- 在分叉路径时保存当前分配器状态
- 允许在分支间切换时恢复正确的内存布局
- 避免重复分配，实现内存复用

### 3. Warmup 与静态输入处理

```python
class NPUWarmupNode:
    """
    Warmup 节点的特殊处理
    - 在正式录制 Graph 前执行，确保所有 NPU 操作都已完成编译和初始化
    - 不录制到 Graph 中，仅用于准备
    """
    
    def run(self, new_inputs: List[InputType]) -> OutputType:
        # 1. 收集当前路径上所有存活的 storage
        existing_path_data_ptrs = {
            t.data_ptr()
            for t in self.path_live_weakrefs()
            if t()
        }
        
        # 2. 找出非 Graph 管理的输入（需要拷贝到 pool）
        non_npugraph_inps = []
        for t in itertools.chain(new_inputs, self.wrapped_function.constants):
            if (isinstance(t, torch.Tensor) 
                and t.untyped_storage().data_ptr() not in existing_path_data_ptrs):
                non_npugraph_inps.append(weakref.ref(t.untyped_storage()))
        
        # 3. 使用内存池执行函数
        with _use_npu_memory_pool_manager(
            self.device_index, self.npu_graphs_pool, self.stream
        ):
            out = self.wrapped_function.model(new_inputs)
        
        # 4. 追踪输出：创建弱引用但不阻止 GC
        self.outputs_weakrefs.extend([
            map_to_ref(out_) if self._should_track_output(out_) else None
            for out_ in out
        ])
        
        return out
```

### 4. 静态输入优化

```python
class NPUGraphNode:
    def __init__(self, ...):
        # ...
        
        # 识别静态输入：来自之前 Graph 输出的 Tensor
        self.npugraph_managed_idxs: List[int] = [
            idx
            for idx, t in enumerate(inputs)
            if isinstance(t, torch.Tensor) and self._is_npu_graph_recorded_tensor(t)
        ]
        
        # 识别活性的别名引用
        self.live_npugraph_managed_path_refs: List[Optional[PathOutputIndex]] = [
            self._is_alias_of_live_recorded_tensor(t) if isinstance(t, torch.Tensor) else None
            for t in inputs
        ]
        
        # 合并静态输入索引
        self.static_input_idxs: List[int] = list(
            set(wrapped_function.static_input_idxs) | set(self.npugraph_managed_idxs)
        )
        
        # 记录静态输入的数据指针（用于后续验证）
        self.static_input_data_ptrs: List[Optional[int]] = [
            self._get_static_data_ptr(i, inputs, self.static_input_idxs)
            for i in range(len(inputs))
        ]
    
    def _is_npu_graph_recorded_tensor(self, t: torch.Tensor) -> bool:
        """检查 Tensor 是否来自之前 Graph 的输出"""
        for storage_weak_ref in self.path_live_weakrefs():
            if t.untyped_storage().data_ptr() == storage_weak_ref.data_ptr():
                return True
        return False
    
    def _is_alias_of_live_recorded_tensor(self, t: torch.Tensor) -> Optional[PathOutputIndex]:
        """检查 Tensor 是否是某个活性输出的别名"""
        for depth, node_outputs in enumerate(self.path_weakrefs):
            for offset, weak_ref in enumerate(node_outputs):
                if weak_ref is None:
                    continue
                storage_ptr = weak_ref.data_ptr()
                if t.untyped_storage().data_ptr() == storage_ptr:
                    return (depth, offset)
        return None
```

---

## 总结

### 核心设计思想

1. **内存池共享**：通过 `MempoolId_t` 实现多 Graph 间的内存复用，减少碎片
2. **惰性分配**：仅在需要时才分配实际物理内存（通过 ExpandableSegment）
3. **弱引用追踪**：使用 `StorageWeakRefWrapper` 追踪 Tensor 生命周期，不影响 GC
4. **状态检查点**：通过 `AllocatorState` 保存/恢复分配器状态，支持路径切换

### 性能优化要点

| 优化点 | 实现方式 | 效果 |
|--------|----------|------|
| 消除 CPU 开销 | Graph Replay 替代逐操作执行 | 显著降低延迟 |
| 内存复用 | 共享 Mempool + Liveness Tracking | 减少内存占用 |
| 避免重新编译 | 一次捕获多次重放 | 预热后零编译开销 |
| 静态输入优化 | 识别并复用稳定地址的 Tensor | 减少数据拷贝 |

### 使用建议

1. **内存池共享**：当多个 Graph 顺序执行时，使用 `pool=graph.pool()` 共享内存池
2. **Warmup 重要性**：确保在 capture 前充分 warmup，避免延迟初始化进入 Graph
3. **避免动态性**：Graph 内避免动态控制流、动态形状、动态内存分配
4. **监控内存**：使用 `torch.npu.memory_summary()` 监控内存池使用情况

---

## 参考链接

- [PyTorch CUDA Graphs Documentation](https://pytorch.org/docs/stable/cuda.html#cuda-graphs)
- [ACL (Ascend Computing Language) Documentation](https://www.hiascend.com/document)
- [torch_npu Source Code](https://gitee.com/ascend/pytorch)

## Related Pages

- [[llm/02_training/torch_compile/overview]]
- [[npugraphs_memory_reuse_analysis]]
- [[torch_compile_npugraphs_deep_dive]]
