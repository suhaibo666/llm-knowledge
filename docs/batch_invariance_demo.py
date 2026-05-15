"""
DeepSeek V4 批次不变性与确定性算子的代码示例
=============================================
基于 DeepSeek_V4.pdf §3.3 和 DeepGEMM 源码的实际实现逻辑。
所有 CUDA kernel 以伪代码形式展示，侧重说明累加路径设计。
"""

import numpy as np
import math

# ============================================================================
# 第一部分：问题根源 — 浮点非结合性
# ============================================================================

def demo_fp_non_associativity():
    """演示为什么累加顺序不同会导致 bitwise 不同的结果"""
    # 使用 float32 演示非结合性，选择数量级差距大的值
    a = np.float32(1e7)
    b = np.float32(1e-7)
    c = np.float32(-1e7)
    r1 = np.float32(np.float32(a + b) + c)   # (1e7 + 1e-7) - 1e7 = 0 (1e-7 被吞掉)
    r2 = np.float32(a + np.float32(b + c))   # 1e7 + (1e-7 - 1e7) = 1e-7 (b 被保留)
    print(f"  float32:  a=1e7, b=1e-7, c=-1e7")
    print(f"  (a+b)+c = {r1:.15f}")
    print(f"  a+(b+c) = {r2:.15f}")
    print(f"  bitwise equal: {r1 == r2}  ← 浮点加法不满足结合律!")

    # 模拟 split-KV 中 4 个 SM 的 partial sum 问题
    # 选择数值范围有差距的 partials，展示顺序影响
    partials = np.array([1e7, -1e7, 1e-7, 2e-7], dtype=np.float32)
    np.random.seed(42)
    print(f"\n  模拟 split-KV: 4 个 SM 各算一段 partial")
    print(f"  partial 值: {partials}")
    print(f"  正确数学结果 (无限精度): {np.sum(partials.astype(np.float64)):.15f}")
    for trial in range(3):
        order = np.random.permutation(4)
        result = np.float32(0.0)
        for idx in order:
            result = np.float32(result + partials[idx])
        print(f"  trial {trial} atomicAdd 顺序 {order} → {result:.15f}")


# ============================================================================
# 第二部分：Attention 双内核策略（核心）
# ============================================================================

# ---------------------------------------------------------------------------
# 2.1 传统 split-KV — 破坏批次不变性的根源
# ---------------------------------------------------------------------------

def traditional_split_kv_attention_pseudocode():
    """
    传统 FlashAttention split-KV 的逻辑（破坏批次不变性）。
    这不是 DeepSeek V4 的做法，而是它要避免的做法。
    """
    print("""
    // ===== split-KV Kernel（破坏不变性）=====
    // 单条序列被拆到 4 个 SM 上

    __global__ void split_kv_kernel(
        float* Q, float* K, float* V, float* O,  // O 为全局输出
        int seq_len
    ) {
        int sm_id = blockIdx.x;       // SM 编号 (0~3)
        int chunk_size = seq_len / 4;
        int start = sm_id * chunk_size;
        int end = start + chunk_size;

        // 每个 SM 只算自己那一段 KV 对 Q 的贡献
        float partial_o[MAX_TOKENS][HEAD_DIM] = {0};
        for (int t = 0; t < num_queries; t++) {
            // 对每段 KV 做局部 softmax + 累加
            float local_max = -INFINITY;
            float local_sum = 0.0f;

            // ... softmax 的 online 计算 ...

            for (int k = start; k < end; k++) {
                float score = dot(Q[t], K[k]) / sqrt(d_k);
                // scaled dot-product + local accumulation
                for (int d = 0; d < HEAD_DIM; d++) {
                    partial_o[t][d] += exp_score * V[k][d];
                }
            }
        }

        // ★★★ 问题所在 ★★★
        // atomicAdd 写入全局 O——四个 SM 的到达顺序不确定
        for (int t = 0; t < num_queries; t++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                atomicAdd(&O[t * HEAD_DIM + d], partial_o[t][d]);
                // ^^^^ SM0 先到还是 SM1 先到取决于硬件调度
                // 不同 batch 下调度不同 → 大 O[t] 的 bitwise 值不同
                // → 批次不变性被破坏！
            }
        }
    }
    """)

# ---------------------------------------------------------------------------
# 2.2 双内核策略 — Kernel 1：单 SM 一条序列（满波高效）
# ---------------------------------------------------------------------------

def kernel1_single_sm_attention_pseudocode():
    """
    Kernel 1: 一条完整序列在单个 SM 内完成 Attention。
    累加路径天然确定（SM 内 sequential）。
    适用于满波场景。
    """
    print("""
    // ===== Kernel 1：单 SM 一条序列（主 Kernel）=====
    // 每个 thread block = 一个 SM，处理一整条序列
    // 累加顺序：在 SM 内 sequential → 天然确定

    __global__ void kernel1_single_sm_attention(
        float* Q,          // [num_seqs, seq_len, head_dim]
        float* K,
        float* V,
        float* O,          // [num_seqs, seq_len, head_dim]
        int seq_len
    ) {
        // 每个 block 处理一条完整序列
        int seq_id = blockIdx.x;
        int head_id = blockIdx.y;

        // 计算该序列 + 该 head 的偏移
        Q += seq_id * seq_len * HEAD_DIM + head_id * HEAD_DIM;
        K += seq_id * seq_len * HEAD_DIM + head_id * HEAD_DIM;
        V += seq_id * seq_len * HEAD_DIM + head_id * HEAD_DIM;
        O += seq_id * seq_len * HEAD_DIM + head_id * HEAD_DIM;

        // SM 内 threads 协作完成 softmax attention
        // 关键：所有累加都在 SM 的 shared memory + register 内顺序完成
        // 不走 global atomicAdd
        __shared__ float smem[BLOCK_SIZE][HEAD_DIM];

        for (int t = 0; t < seq_len; t++) {
            float output[HEAD_DIM] = {0};

            // Online softmax：单线程或 warp 内顺序累加
            float max_val = -INFINITY;
            float sum_exp = 0.0f;

            // ★ 累加路径确定：j 从 0 到 seq_len-1 顺序遍历
            for (int j = 0; j < seq_len; j++) {
                float score = 0.0f;
                for (int d = 0; d < HEAD_DIM; d++) {
                    score += Q[t * HEAD_DIM + d] * K[j * HEAD_DIM + d];
                }
                score *= rsqrt(HEAD_DIM);

                // 更新 online softmax 状态
                float new_max = fmaxf(max_val, score);
                float exp_diff = expf(max_val - new_max);
                sum_exp = sum_exp * exp_diff + expf(score - new_max);
                max_val = new_max;

                // 累加到 output（顺序确定！）
                float weight = expf(score - max_val);
                for (int d = 0; d < HEAD_DIM; d++) {
                    output[d] += weight * V[j * HEAD_DIM + d];
                }
                // output[d] 的有效累加顺序：j=0,1,2,...,seq_len-1
                // 无论 batch 里有哪些其他序列，本序列的累加顺序永远相同
            }

            // 归一化并写入全局输出
            for (int d = 0; d < HEAD_DIM; d++) {
                O[t * HEAD_DIM + d] = output[d] / sum_exp;
            }
        }
        // ★ 没有 atomicAdd，没有跨 SM 竞争。输出纯由本 SM 的累加路径决定。
        // 只要 Q,K,V 输入相同，输出就 bitwise 相同 → 批次不变性成立。
    }
    """)

# ---------------------------------------------------------------------------
# 2.3 双内核策略 — Kernel 2：多 SM 协作一条序列（尾部低延迟）
# ---------------------------------------------------------------------------

def kernel2_multi_sm_attention_pseudocode():
    """
    Kernel 2: 多个 SM 协作处理一条序列。
    必须在多个 SM 分片计算后，用固定顺序做归约，保证与 Kernel 1 bitwise 一致。
    适用于部分填充的波。
    """
    print("""
    // ===== Kernel 2：多 SM 协作一条序列（尾部 Kernel）=====
    // 序列被切成 N 段，每段一个 SM 计算 partial output
    // 然后用 distributed shared memory 按固定顺序归约
    //
    // 关键约束：有效累加路径必须与 Kernel 1 一致！
    //   Kernel 1: O = (...((v0*w0 + v1*w1) + v2*w2) + ... + vN*wN)
    //   Kernel 2: O = reduce_fixed_order(partial_0, partial_1, ..., partial_K)
    //              = (...((partial_0 + partial_1) + partial_2) + ... + partial_K)
    //   两者在数学上等价 → bitwise 相同

    __global__ void kernel2_multi_sm_attention(
        float* Q,
        float* K,
        float* V,
        float* O,
        int seq_len
    ) {
        // 每个 block 处理序列的一段
        int num_blocks = gridDim.x;     // 分配给本序列的 block 数
        int block_id = blockIdx.x;       // 本 block 的编号 (0 ~ num_blocks-1)
        int chunk_start = block_id * (seq_len / num_blocks);
        int chunk_end = (block_id + 1) * (seq_len / num_blocks);

        // ★ 独立的局部累加缓冲区（在 shared memory 中）
        __shared__ float partial_output[MAX_TOKENS][HEAD_DIM];
        // 初始化为 0，不使用 atomicAdd 到全局

        // ... 计算本段的 partial output（与 Kernel 1 相同的方式）...

        __syncthreads();  // 确保所有 partial 计算完成

        // ============================================================
        // ★ 核心：固定顺序的跨 SM 归约（替代 atomicAdd）
        // ============================================================
        // 使用 distributed shared memory（SM90+ thread-block cluster 特性）
        // 让相邻 SM 的 shared memory 可以直接互相访问

        // 归约顺序：block 0 → block 1 → block 2 → ...（与 Kernel 1 的 token 顺序一致）
        if (block_id == 0) {
            // Block 0 作为最终累加器，按固定顺序合并所有其他 block 的 partial
            for (int src_block = 1; src_block < num_blocks; src_block++) {
                // 通过 distributed shared memory 读取 src_block 的 shared memory
                // cluster.sync() 确保 src_block 已写完
                cluster_arrive_and_wait();

                // 读取其他 SM 的 partial output
                float* src_partial = cluster_map_shared_memory(src_block, partial_output);

                // ★ 按固定顺序累加（不是 atomicAdd！）
                for (int t = 0; t < chunk_queries; t++) {
                    for (int d = 0; d < HEAD_DIM; d++) {
                        partial_output[t][d] += src_partial[t][d];
                        // 累加顺序确定：
                        //   partial_output += src_block1_partial
                        //   partial_output += src_block2_partial
                        //   ...
                        // 永远是这个顺序，不依赖硬件调度
                    }
                }
            }

            // Block 0 将最终结果写入全局内存（仅一次，无竞争）
            for (int t = 0; t < num_queries; t++) {
                for (int d = 0; d < HEAD_DIM; d++) {
                    O[t * HEAD_DIM + d] = partial_output[t][d];
                }
            }
        }
        // 其他 block 只提供 partial，不写入全局 O
    }

    // 对比传统做法（破坏不变性）：
    //   每个 SM 直接 atomicAdd(&O[...], partial)
    //   → O 的最终值 = partial_? + partial_? + ... 顺序不确定
    //
    // 本方案：
    //   Block 0 显式按 block_id 顺序累加
    //   → O 的最终值 = (...((p0 + p1) + p2) + p3)
    //   → 与 Kernel 1 的 token 级累加路径完全等价
    """)

# ---------------------------------------------------------------------------
# 2.4 双内核调度器（Python 伪代码）
# ---------------------------------------------------------------------------

class DualKernelScheduler:
    """
    决定用 Kernel 1 还是 Kernel 2 处理当前 batch。
    """
    def __init__(self, num_sms: int = 132):  # H100
        self.num_sms = num_sms
        self._kernel1 = None  # CUDA kernel handle
        self._kernel2 = None

    def dispatch(self, batch_size: int) -> str:
        """
        返回应该使用的 kernel 名称。
        - 满波：全用 Kernel 1（每个 SM 一条序列）
        - 最后的部分波：用 Kernel 2（多条序列共享所有 SM）
        """
        num_full_waves = batch_size // self.num_sms
        remainder = batch_size % self.num_sms

        if remainder == 0:
            # 完美对齐，全部用 Kernel 1
            return "kernel1 × all"
        else:
            # 前 num_full_waves 波用 Kernel 1
            # 最后 remainder 条序列用 Kernel 2（空闲 SM 也来帮忙）
            return f"kernel1 × {num_full_waves} waves, kernel2 for last {remainder} seqs"

    def launch(self, Q, K, V, num_seqs: int):
        """
        伪代码：启动 attention 内核
        """
        full_waves = num_seqs // self.num_sms
        remainder = num_seqs % self.num_sms

        # 满波：每条序列一个 SM → Kernel 1
        for wave in range(full_waves):
            start = wave * self.num_sms
            end = start + self.num_sms
            # kernel1_single_sm_attention<<<self.num_sms, THREADS_PER_BLOCK>>>(
            #     Q[start:end], K[start:end], V[start:end], O[start:end]
            # )

        # 尾部部分波：用 Kernel 2
        if remainder > 0:
            start = full_waves * self.num_sms
            # 每条尾部序列分配 self.num_sms // remainder 个 SM
            # sms_per_seq = self.num_sms // remainder
            # kernel2_multi_sm_attention<<<sms_per_seq * remainder, THREADS_PER_BLOCK>>>(
            #     Q[start:], K[start:], V[start:], O[start:]
            # )


# ============================================================================
# 第三部分：矩阵乘法 — DeepGEMM 的确定性方法
# ============================================================================

# ---------------------------------------------------------------------------
# 3.1 传统 split-k（破坏不变性）
# ---------------------------------------------------------------------------

def traditional_split_k_matmul():
    """
    传统 split-k 矩阵乘法：C = A @ B
    K 维度拆分到多个 thread block，每个算 partial，最后 atomicAdd 合并。
    破坏批次不变性。
    """
    print("""
    // ===== 传统 split-k GEMM（破坏不变性）=====
    // C[m][n] = sum over k of A[m][k] * B[k][n]
    // K 维度被拆分到 N 个 block

    __global__ void split_k_gemm(
        float* A, float* B, float* C, int M, int N, int K
    ) {
        int m = blockIdx.x;
        int n = blockIdx.y;
        int k_block = blockIdx.z;       // K 分片编号
        int k_chunk_size = K / gridDim.z;

        // 本 block 只算一段 K 的 partial dot product
        float partial_acc = 0.0f;
        for (int k = k_block * k_chunk_size; k < (k_block+1) * k_chunk_size; k++) {
            partial_acc += A[m * K + k] * B[k * N + n];
        }

        // ★ 问题：atomicAdd 合并各 block 的 partial
        atomicAdd(&C[m * N + n], partial_acc);
        //   4 个 block 的 partial 值分别是 [0.1, 0.2, 0.3, 1e-8]
        //   到达顺序不同 → 中间舍入不同 → 最终 C[m][n] 不同
    }

    // C[m][n] = partial_block0 + partial_block1 + partial_block2 + partial_block3
    //         = (0.1 + 0.2) + 0.3 + 1e-8  ← 可能是这样
    //         = 0.1 + (0.2 + 0.3) + 1e-8  ← 也可能是这样（硬件调度不同）
    //   → 虽然数学上近似，但 bitwise 不同
    """)

# ---------------------------------------------------------------------------
# 3.2 DeepGEMM 1D1D 布局 — 避免 split-k
# ---------------------------------------------------------------------------

def deepgemm_1d1d_matmul():
    """
    DeepGEMM 的核心设计：1D1D 布局。
    每个输出 tile 恰好由一个 CTA（thread block）计算。
    不做 K 维度拆分 → 没有跨 block 累加 → 天然确定。
    """
    print("""
    // ===== DeepGEMM 1D1D GEMM（确定性）=====
    //
    // 关键理念：每个 C[m_block][n_block] tile 由且仅由一个 CTA 计算
    // 不拆分 K 维度 → 没有跨 block 累加 → 天然 bitwise 确定
    //
    // DeepGEMM 源码位置: deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh
    //
    // 实际源码结构（简化）：

    template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int kNumStages>
    __global__ void sm90_fp8_gemm_1d1d_impl(
        __nv_fp8_e4m3* gmem_a,
        __nv_fp8_e4m3* gmem_b,
        // ... TMA descriptors ...
        uint32_t shape_m, uint32_t shape_n, uint32_t shape_k
    ) {
        // ★ 每个 CTA 有自己的 accumulator（在寄存器中）
        float accum[WGMMA::kNumAccum];           // 当前 tile 的 partial
        float final_accum[WGMMA::kNumAccum];     // 跨 K 的累加结果

        // ★ Persistent Scheduler：每个 CTA 独立领取一个 (m_block, n_block) tile
        // 同一个 tile 不会分配给两个 CTA → 没有竞争
        Scheduler scheduler(shape_m, shape_n);
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // 清零累加器
            for (int i = 0; i < WGMMA::kNumAccum; i++)
                final_accum[i] = 0.0f;

            // 遍历所有 K blocks（本 CTA 独立完成）
            for (int k_block = 0; k_block < shape_k / BLOCK_K; k_block++) {
                // TMA 异步加载 A/B tile → shared memory
                pipeline_copy(gmem_a, gmem_b, smem_a, smem_b, k_block);

                // WGMMA 指令：A[BLOCK_M][BLOCK_K] × B[BLOCK_K][BLOCK_N]
                wgmma(accum, smem_a, smem_b);

                // ★ 本 CTA 内部顺序累加（没有跨 CTA 的 atomicAdd！）
                for (int i = 0; i < WGMMA::kNumAccum; i++)
                    final_accum[i] += accum[i];
                    // 累加路径：K=0 的 partial, 然后 K=1 的 partial, ...
                    // 完全由本 CTA 的循环顺序决定 → bitwise 确定
            }

            // 直接写回全局内存（本 tile 仅此一个 writer）
            store_to_gmem(gmem_c, final_accum, m_block_idx, n_block_idx);
            // ★ 没有 atomicAdd，没有跨 CTA 协调，输出纯由本 CTA 决定
        }
    }
    """)

# ---------------------------------------------------------------------------
# 3.3 小 batch 场景：放弃 split-k + 其他优化补偿
# ---------------------------------------------------------------------------

def deepgemm_small_batch_optimization():
    """
    小 batch 下传统做法依赖 split-k 来填满 SM。
    DeepGEMM 放弃 split-k，用以下优化补偿性能。
    """
    print("""
    // ===== DeepGEMM 小 batch 优化（替代 split-k）=====
    //
    // 小 batch → 输出 tile 数量少 → SM 利用率低
    // 传统方案：split-k → 多个 CTA 算同一 tile → SM 填满了但破坏不变性
    // DeepGEMM 方案：不做 split-k，改做以下优化：

    // 1. 更细粒度的 tiling — 把输出 tile 切得更小
    //    BLOCK_M=64, BLOCK_N=64（而非 128×128）
    //    → 相同输出总面积 = 更多 tile = 更多 CTA → SM 利用率提升

    // 2. Warp specialization — TMA 线程 + 数学线程分工
    //    一部分 warp 专门负责数据搬运（TMA），另一部分只做计算
    //    → 隐藏访存延迟，单 CTA 的吞吐更高

    // 3. Multi-stage pipeline — 多个 shared memory buffer 流水线
    //    stage 0: 加载 K block → 计算
    //    stage 1: 加载下一个 K block（与当前计算重叠）
    //    → TMA 延迟被完全隐藏

    // 4. Persistent kernel — CTA 处理完一个 tile 后领取下一个
    //    而非一个 CTA 只处理一个 tile
    //    → 避免 kernel launch overhead

    // DeepGEMM scheduler 源码（gemm.cuh）逻辑：
    struct Scheduler {
        int next_m_block = 0;
        int next_n_block = 0;
        int total_m_blocks, total_n_blocks;

        bool get_next_block(int& m, int& n) {
            // 原子领取下一个 tile（tile 级别，不是 K 分片级别！）
            int idx = atomicAdd(&global_tile_counter, 1);
            if (idx >= total_m_blocks * total_n_blocks) return false;
            m = idx / total_n_blocks;
            n = idx % total_n_blocks;
            return true;
            // ★ 每个 (m,n) tile 只分配给一个 CTA → 不需要跨 CTA 累加
        }
    };
    """)

# ---------------------------------------------------------------------------
# 3.4 DeepGEMM 的分组 GEMM 场景 — TMA_REDUCE_ADD
# ---------------------------------------------------------------------------

def deepgemm_grouped_gemm_reduction():
    """
    当分组 GEMM 中多个组写入同一输出区域时，
    DeepGEMM 用 SM90 的硬件 TMA_REDUCE_ADD 做确定性归约。
    """
    print("""
    // ===== DeepGEMM 分组 GEMM 的确定性归约 =====
    //
    // 场景：Grouped GEMM（如 MoE），多个 expert group 输出到同一缓冲区
    // 需要把多个 CTA 的结果合并
    //
    // 传统方案：atomicAdd → 非确定性
    // DeepGEMM 方案：SM90_TMA_REDUCE_ADD → 硬件管理归约

    // 1. 每个 group 的 CTA 独立计算自己的 tile（不跨 group 冲突）
    // 2. 写入时使用 TMA 的 reduce-add 语义

    // 源码片段（简化自 sm90_fp8_gemm_1d1d.cuh）：

    // 从 shared memory 写出 + 归约到 global memory
    cute::SM90_TMA_REDUCE_ADD_2D::copy(
        &tensor_map_cd,           // 目标 global memory 的 TMA descriptor
        smem_output,              // 源 shared memory
        n_block_idx * BLOCK_N,    // 目标 N 坐标
        group_idx * shape_m + m_block_idx * BLOCK_M  // 目标 M 坐标
    );
    // SM90_TMA_REDUCE_ADD_2D:
    //   - 硬件级别的 read-modify-write
    //   - 归约结果由 L2 cache 的内存控制器保证一致性
    //   - 不依赖 software atomicAdd 的到达顺序
    //   - 在固定 group 顺序下行为可预测

    // 注意：TMA_REDUCE_ADD 本身在跨 CTA 时仍有硬件调度影响，
    // 但在 DeepGEMM 的 1D1D 布局下，同一 tile 只有一个 CTA 写入，
    // TMA_REDUCE_ADD 主要用于分步写入（如 epilogue 拆分为多个写操作）。
    """)

# ============================================================================
# 第四部分：MoE 反向传播的确定性累加
# ============================================================================

def moe_backward_deterministic():
    """
    MoE 反向传播确定性方案：
    1. Per-SM 独立缓冲区
    2. Token 顺序预处理
    3. 确定性全局求和
    """
    print("""
    // ===== MoE Backward：确定性梯度累加 =====
    //
    // 问题：多个 rank 的多个 SM 向同一 expert 的梯度缓冲区写入
    // atomicAdd 顺序不确定 → 梯度在最后几个 bit 上可能不同
    // → 反向传播非确定性

    // ------------------------------------------------------------------
    // Step 1: Per-SM 独立累加缓冲区
    // ------------------------------------------------------------------
    // 每个 SM 有自己的 local gradient buffer，不直接写全局

    __global__ void moe_backward_collect(
        float* token_grads,        // 输入 token 的梯度
        int* expert_assignments,   // 各 token 路由到哪个 expert
        float* per_sm_buffers,     // [NUM_SMS][NUM_EXPERTS][HIDDEN_DIM]
        int num_tokens
    ) {
        int sm_id = blockIdx.x;
        int token_id = blockIdx.x * blockDim.x + threadIdx.x;

        if (token_id < num_tokens) {
            int expert_id = expert_assignments[token_id];

            // ★ 写到本 SM 独占的 buffer 区域，不使用 atomicAdd
            //   per_sm_buffers[sm_id][expert_id][:] += token_grads[token_id][:]
            int offset = sm_id * NUM_EXPERTS * HIDDEN_DIM
                       + expert_id * HIDDEN_DIM;
            for (int d = 0; d < HIDDEN_DIM; d++) {
                per_sm_buffers[offset + d] += token_grads[token_id * HIDDEN_DIM + d];
                // 单 SM 内 sequential 累加 → 确定
            }
        }
    }

    // ------------------------------------------------------------------
    // Step 2: 确定性全局求和（替代 atomicAdd）
    // ------------------------------------------------------------------

    __global__ void deterministic_global_sum(
        float* per_sm_buffers,     // [NUM_SMS][NUM_EXPERTS][HIDDEN_DIM]
        float* global_grads,       // [NUM_EXPERTS][HIDDEN_DIM]
        int num_experts
    ) {
        int expert_id = blockIdx.x;
        int dim_offset = threadIdx.x;

        // ★ 固定 SM 顺序累加：SM 0 → SM 1 → ... → SM N-1
        float accum = 0.0f;
        for (int sm = 0; sm < NUM_SMS; sm++) {
            int offset = sm * NUM_EXPERTS * HIDDEN_DIM
                       + expert_id * HIDDEN_DIM
                       + dim_offset;
            accum += per_sm_buffers[offset];
            // 累加顺序确定：先加 SM 0 的，再加 SM 1 的，...
            // 不管硬件调度如何，这个 for 循环的迭代顺序不变
        }

        // 写入全局
        int global_offset = expert_id * HIDDEN_DIM + dim_offset;
        global_grads[global_offset] = accum;
    }

    // ------------------------------------------------------------------
    // Step 3: Token 顺序预处理（跨 rank 场景）
    // ------------------------------------------------------------------
    // 每个 rank 在发送 expert 梯度前，按 token 顺序预处理
    // 配合跨 rank 的 buffer 隔离，确保接收端的累加顺序确定

    void preprocess_token_order(
        float* token_grads,
        int* token_ids,
        int* expert_assignments,
        int num_tokens
    ) {
        // 1. 按 (expert_id, token_id) 排序
        //    确保同一 expert 的 token 以确定顺序处理
        sort_by_expert_then_token(token_grads, token_ids, expert_assignments, num_tokens);

        // 2. 每个 expert 内部，token 以固定顺序累加
        //    不依赖 token 到达的时序

        // 3. 每个 rank 的输出 buffer 隔离
        //    rank_i 只能写自己的 buffer 区域
        //    不与 rank_j 的区域重叠 → 消除跨 rank 竞争
    }
    """)


# ============================================================================
# 第五部分：端到端验证 demo
# ============================================================================

def batch_invariance_verification_demo():
    """
    模拟验证批次不变性：
    同一条输入在不同 batch 中的输出是否 bitwise 一致。
    """
    np.random.seed(42)

    def softmax(x, axis=-1):
        """稳定的 softmax 实现"""
        x_max = np.max(x, axis=axis, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def scaled_dot_product_attention(q, k, v):
        """标准 scaled dot-product attention"""
        d_k = k.shape[-1]
        scores = q @ k.T / math.sqrt(d_k)
        attn = softmax(scores, axis=-1)
        return attn @ v

    def simulate_atomic_add_non_deterministic(partials_list):
        """
        模拟 atomicAdd 的非确定性：
        多个 partial sum 以随机顺序累加 → 不同结果
        """
        # 每轮用不同顺序累加
        orders = [
            [0, 1, 2, 3],
            [3, 1, 0, 2],
            [1, 3, 2, 0],
        ]
        results = []
        for order in orders:
            accum = np.float32(0.0)
            for idx in order:
                accum = np.float32(accum + partials_list[idx])
            results.append(accum)
        return results

    # ============================================
    # Demo 1: 同一函数同一输入 → 永远 bitwise 相同
    # （模拟 Kernel 1 的效果：单 SM 内顺序确定）
    # ============================================
    seq_a = np.random.randn(4, 64).astype(np.float32)
    seq_b = np.random.randn(4, 64).astype(np.float32)

    out1 = scaled_dot_product_attention(seq_a, seq_a, seq_a)
    out2 = scaled_dot_product_attention(seq_a, seq_a, seq_a)

    print("=" * 60)
    print("Demo 1: 确定性 Attention（单 SM 风格）")
    print("=" * 60)
    print(f"  输入 seq_a 的形状: {seq_a.shape}")
    print(f"  运行两次，输出 bitwise 相同吗？ {np.array_equal(out1, out2)}")
    print(f"  out1[0, :4]: {out1[0, :4]}")
    print(f"  out2[0, :4]: {out2[0, :4]}")
    print()
    print("  → Kernel 1 的效果：单 SM 内顺序累加，每次结果都一样")
    print("  → 不管 batch 里有 seq_b 还是 seq_c，seq_a 的结果始终不变")
    print()

    # ============================================
    # Demo 2: 模拟 atomicAdd 的非确定性
    # ============================================
    print("=" * 60)
    print("Demo 2: 模拟 split-KV / split-k 的 non-deterministic 累加")
    print("=" * 60)
    # 4 个 partial sum（模拟 4 个 SM 各算一段，有数量级差异）
    partials = np.array([1e7, -1e7, 1e-7, 2e-7], dtype=np.float32)
    results = simulate_atomic_add_non_deterministic(partials)
    for i, (order, r) in enumerate(zip(
        [[0,1,2,3], [3,1,0,2], [1,3,2,0]], results
    )):
        print(f"  atomicAdd 顺序 {order}: {r:.7f}")

    expected = np.float64(0)
    for p in partials.astype(np.float64):
        expected += p
    print(f"  真实数学结果 (float64): {expected:.7f}")
    print(f"  所有结果相同吗？ {all_same}")
    print(f"  所有结果都等于真实值吗？ {all(r == expected for r in results)}")
    print()
    print("  → 浮点加法 (a+b)+c ≠ a+(b+c)，累加顺序不同 → bitwise 不同")
    print("  → 这就是 split-KV 和 split-k 被放弃的原因")
    print()

    # ============================================
    # Demo 3: DeepGEMM 风格的固定顺序累加 = 确定性
    # ============================================
    print("=" * 60)
    print("Demo 3: DeepGEMM/Kernel 2 风格的固定顺序归约")
    print("=" * 60)
    # 固定顺序累加：总是 partial[0] → partial[1] → partial[2] → partial[3]
    fixed_results = []
    for _ in range(3):
        accum = np.float32(0.0)
        for i in range(4):  # 固定顺序 0,1,2,3
            accum = np.float32(accum + partials[i])
        fixed_results.append(accum)

    for i, r in enumerate(fixed_results):
        print(f"  第 {i+1} 次固定顺序累加: {r:.15f}")

    all_same_fixed = all(r == fixed_results[0] for r in fixed_results)
    print(f"  所有结果相同吗？ {all_same_fixed}")
    print()
    print("  → 固定累加顺序 → 每次结果 bitwise 相同 → 确定性/批次不变性")
    print()

    # ============================================
    # Demo 4: 批次不变性 vs 批次依赖性
    # ============================================
    print("=" * 60)
    print("Demo 4: 批次不变性的含义")
    print("=" * 60)
    print("""
  批次不变性 (Batch Invariance):
    对于输入序列 X，无论它在 batch 中的位置如何，输出 O(X) 始终 bitwise 相同。

    batch1 = [A, B, C]  →  A 的输出 = O_A
    batch2 = [B, A, C]  →  A 的输出 = O_A  (相同!)
    batch3 = [C, X, A]  →  A 的输出 = O_A  (相同!)

  这在 RL 后训练中至关重要:
    - GRPO 对同一 prompt 生成 G 个 response，在组内比较 advantage
    - 如果同一 response 的 logit 因 batch 位置不同而变化
      → advantage 比较失效 → 训练不稳定 → loss spike
    """)


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Part 0: 浮点非结合性演示")
    print("=" * 60)
    demo_fp_non_associativity()

    print("\n" + "=" * 60)
    print("Part 1: 传统 split-KV（破坏不变性）")
    print("=" * 60)
    traditional_split_kv_attention_pseudocode()

    print("\n" + "=" * 60)
    print("Part 2: Kernel 1 — 单 SM 一条序列")
    print("=" * 60)
    kernel1_single_sm_attention_pseudocode()

    print("\n" + "=" * 60)
    print("Part 3: Kernel 2 — 多 SM 协作 + 固定顺序归约")
    print("=" * 60)
    kernel2_multi_sm_attention_pseudocode()

    print("\n" + "=" * 60)
    print("Part 4: DeepGEMM 1D1D 确定性 MatMul")
    print("=" * 60)
    traditional_split_k_matmul()
    deepgemm_1d1d_matmul()
    deepgemm_small_batch_optimization()

    print("\n" + "=" * 60)
    print("Part 5: MoE 反向传播确定性累加")
    print("=" * 60)
    moe_backward_deterministic()

    print("\n" + "=" * 60)
    print("Part 6: 验证 Demo")
    print("=" * 60)
    batch_invariance_verification_demo()
