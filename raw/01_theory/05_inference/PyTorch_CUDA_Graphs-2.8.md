# PyTorch CUDA semantics（2.8）

- **来源**：PyTorch 官方版本化文档 [CUDA semantics](https://docs.pytorch.org/docs/2.8/notes/cuda.html)；阅读快照 2026-09-17。
- **版本与定位**：文档路径 `docs/2.8`；“CUDA streams”“CUDA Graphs / Why CUDA Graphs?”“PyTorch API”“Constraints”“Graph memory management”。
- **用途**：核对不同流的并发与显式同步、侧流内存生命周期、图重放的固定参数和虚拟地址、输入拷贝顺序、CPU 工作不被捕获、输出读取与图私有池成本。
- **边界**：这些是 PyTorch 2.8 的 CUDA 语义与 API 约束，不承诺某个推理引擎会按图示时间调度，也不把流的“可并发”误写成硬件上的必然重叠。
