// 35_deepseek_v4_context_parallel_analysis.md 的三张图。
//
// 图 1：**布局图** —— 两个 CP rank 各自拿到什么、缺什么、从哪补。全局 THD 行按 contiguous
//       切到 rank0 / rank1，rank1 的左边界行来自 rank0 尾部；每个 rank 的 compressor 输入
//       如何被压紧成定容槽位、哪些槽是重复计算、哪些是容量补齐；AllGather 之后的 rank-major
//       压缩行编号与 seq_to_rank_row；最后把三条本地 query 的 window 行与 top-k 压缩块降到
//       kv_full 的物理下标。全部由复刻自冻结基线的
//       csa_utils/cp_utils.py::exchange_cp_boundary_hidden / prepare_cp_compressor_input /
//       _build_cp_indexer_layout、csa_utils/cp_layout_kernels.py 的
//       _compressor_input_compact_fwd_kernel / _build_attention_indices_kernel 语义算出。
// 图 2：**时序图** —— 前向 Stage 2 两个异步 AllGather 与本地投影的重叠顺序（有 indexer /
//       无 indexer 两条 lane），以及反向两个延迟 reduce-scatter 的发起顺序与分支内 wait 位置。
//       前向顺序按 csa.py::CompressedSparseAttention._forward_thd_cp 的语句序编码；反向顺序
//       由一个「按 autograd 序号从大到小调度」的最小引擎复刻算出，并锁定到
//       test_csa_fused_sparse_attention.py::TestCPCommunicationOverlap 的两组事件序列。
// 图 3：**代价图** —— 用真实配方 mxfp8_THD64K_128GPU_TP1PP2EP64CP16.yaml 的形状算每层每 rank
//       的边界 P2P 字节、Indexer-K gather 字节、KV gather 字节，与「朴素全量 KV AllGather」
//       对比；再把 CP 从 2 扫到 128，看 gather 接收量与本地行数如何反向变化。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
//
// 图 1 要回答「为什么只搬 8 行边界和压缩行就够了，以及搬来的行在 kv_full 里放在哪」。
//   共用算例：CP=2，全局 32 个 token 打包成两条序列 18+14（都不是 4 的倍数，尾巴各丢 2 个），
//   csa_window_size=4、ratio=4（d_comp=8、d_window=8）、indexer top-k=2。
//   面板 A：32 个 token 格，上方序列括号、下方 4 个一组的压缩组编号（尾巴画 ghost ×），
//     再下一行是 contiguous 切分（rank0 取 0..15，rank1 取 16..31），rank1 的左边界 8 行
//     （全局 8..15）用 acc1 描边并连到面板 B。
//   面板 B：每个 rank 一个子面板：左边「boundary 8 | local 16」两段条，右边 c_cap=8 个
//     compressor 槽位，每槽写它装的压缩组（seq·g）；本 rank 拥有的槽 neutral，重复计算
//     但不拥有的槽 acc2 描边，容量补齐的槽 ghost。
//   面板 C：AllGather 后的 rank-major 缓冲 16 个槽（rank0 s0..s7 | rank1 s0..s7），下面一行
//     seq_to_rank_row：逻辑压缩行 0..7 各指向哪个物理槽（-1 为不存在的行）。
//   面板 D：三条 rank1 query 的索引降级表：window 行 → kv_full 下标（边界行 < 8、本地行 8..23）、
//     可见压缩块数、top-2 逻辑块 → seq-major 行 → rank-major 行 → kv_full 下标（= 24 + 行号）、
//     topk_length。
//   底部三个盒子：规则（d_comp / d_window / c_cap / 所有权）、两阶段各搬多少行与朴素方案对比、
//     本图复刻了什么 / 简化了什么。
//
// 图 2 要回答「两个 AllGather 分别藏在哪段本地计算后面，反向为什么 KV 的 reduce-scatter 先发」。
//   三个面板纵向排列，每个面板两条泳道（计算流 / 通信流），块按顺序等宽排列、不写时长：
//   ① 前向·有 indexer：P2P 边界（阻塞）→ KV/Q 投影 → 压紧 → Indexer-K compressor ⟶ AG-K 发起
//     → attention-KV compressor ⟶ AG-KV 发起 → Indexer Q 投影 + CP-aware RoPE + weights 投影
//     → wait AG-K → seq-major 重排 + top-k → wait AG-KV → cat + 索引降级 → 融合稀疏注意力。
//   ② 前向·无 indexer：同样的前缀，但 KV compressor 之后是同步 AllGather（通信流上一个阻塞块）。
//   ③ 反向·融合 indexer-loss：稀疏注意力反向 → 发起 RS-KV → 发起 RS-K → 本地 indexer 梯度 →
//     Q/weights 投影反向 → wait RS-KV → attention-KV compressor 反向 → wait RS-K →
//     Indexer-K compressor 反向 → 压紧反向（散射到 local + boundary）→ KV 投影反向 →
//     边界梯度 P2P（阻塞）。acc1 = 被掩盖的通信段，acc2 = 阻塞等待。
//   右侧两个盒子：顺序不变量（每个 rank 都先 K 后 KV；反向先 KV 后 K；wait 只在消费分支）
//     与锁定这些顺序的测试名。
//
// 图 3 要回答「压缩态 gather 到底省了多少，以及这笔账随 CP 怎么变」。
//   左：两组柱（ratio-4 层 / ratio-128 层），每组四根：边界 P2P 接收、Indexer-K AG 接收、
//     KV AG 接收、朴素全量 KV AG 接收，单位 MiB，柱顶写数值。
//   右：CP ∈ {2,4,8,16,32,64,128}，两条折线：每 rank 压缩态 gather 接收字节（K+KV，ratio-4 层）
//     与每 rank 本地行数；接收量随 (CP−1)/CP 饱和而本地行数按 1/CP 缩，是重叠窗口随 CP 收窄的来源。
//   底部：形状与 dtype 假设（bf16、hidden 4096、v_head_dim 512、indexer head_dim 128、
//     window 128、l_local = 65536/16 = 4096）。
//
// 用法：node tools/figs/svg/megatron_dsv4_cp_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// ============================================================================
// 共用算例
// ============================================================================

const CFG = Object.freeze({
  cpSize: 2,
  seqLens: Object.freeze([18, 14]), // 两条打包序列，都不是 ratio 的倍数
  windowSize: 4,
  ratio: 4,
  topk: 2,
  // 图 1 面板 D 要展开的三条 rank1 query（全局行号）
  demoQueries: Object.freeze([16, 21, 29]),
});

// 真实配方 examples/moe_recipes/deepseek_v4_flash/gb200/mxfp8_THD64K_128GPU_TP1PP2EP64CP16.yaml
const RECIPE = Object.freeze({
  seqLength: 65536,
  cpSize: 16,
  hiddenSize: 4096,
  vHeadDim: 512, // linear_kv_proj 输出与 attention compressor 的 head_dim
  indexerHeadDim: 128, // dsa_indexer_head_dim，indexer compressor 的 head_dim
  windowSize: 128, // csa_window_size
  ratios: Object.freeze([4, 128]), // csa_compress_ratios 里出现的两种 >1 的比
  dtypeBytes: 2, // bf16：hidden / 压缩行 / KV 行的驻留类型（见页面 §2.9 的假设说明）
  cpSweep: Object.freeze([2, 4, 8, 16, 32, 64, 128]),
});

// ============================================================================
// 复刻 cp_utils.py 的整数规则
// ============================================================================

const gcd = (a, b) => (b === 0 ? a : gcd(b, a % b));

// exchange_cp_boundary_hidden / prepare_cp_compressor_input 的 d_comp 规则
function dComp(ratio) {
  if (ratio === 4) return 8;
  return ratio > 1 ? ratio : 0;
}

// exchange_cp_boundary_hidden：d_window = max(csa_window_size, d_comp)
function dWindow(windowSize, ratio) {
  return Math.max(windowSize, dComp(ratio));
}

// prepare_cp_compressor_input：c_cap = align(max(1, (l_local + d_comp) // ratio), 32 // gcd(32, ratio))
function compactCapacity(lLocal, ratio) {
  const alignment = 32 / gcd(32, ratio);
  let cCap = Math.max(1, Math.floor((lLocal + dComp(ratio)) / ratio));
  cCap = Math.ceil(cCap / alignment) * alignment;
  return cCap;
}

// torch.bucketize(x, boundaries, right=True)：boundaries 中 <= x 的个数
function bucketizeRight(x, boundaries) {
  let n = 0;
  for (const b of boundaries) if (b <= x) n += 1;
  return n;
}

function cumsum(lens) {
  const out = [0];
  for (const l of lens) out.push(out[out.length - 1] + l);
  return out;
}

// _thd_cp_position_ids：连续 CP 行区间 → 打包序列内的位置；越界行（边界为负、超出末序列）映射到 0
function thdCpPositionIds(cu, globalStart, localRows) {
  const nSeq = cu.length - 1;
  const out = [];
  for (let i = 0; i < localRows; i += 1) {
    const row = globalStart + i;
    const seq = Math.min(bucketizeRight(row, cu.slice(1)), nSeq - 1);
    const valid = row >= cu[seq] && row < cu[seq + 1];
    out.push(valid ? row - cu[seq] : 0);
  }
  return out;
}

// _build_cp_indexer_layout：本地 Q 段 × 全量压缩 K 段 + 因果偏移 + 末尾合成 padding 段
function buildCpIndexerLayout(cuQ, cuComp, globalStart, localRows) {
  const globalEnd = globalStart + localRows;
  const nSeq = cuQ.length - 1;
  const qLens = [];
  const offsets = [];
  for (let s = 0; s < nSeq; s += 1) {
    const localStart = Math.max(cuQ[s], globalStart);
    const localEnd = Math.min(cuQ[s + 1], globalEnd);
    const len = Math.max(localEnd - localStart, 0);
    qLens.push(len);
    offsets.push(len > 0 ? localStart - cuQ[s] : 0);
  }
  const prefix = cumsum(qLens);
  const paddingQ = Math.max(globalEnd - Math.max(cuQ[nSeq], globalStart), 0);
  const cuQTopk = [...prefix, prefix[prefix.length - 1] + paddingQ];
  const cuKTopk = [...cuComp, cuComp[cuComp.length - 1]];
  return { cuQTopk, cuKTopk, qCausalOffsets: [...offsets, 0] };
}

// ============================================================================
// 复刻 cp_layout_kernels.py::_compressor_input_compact_fwd_kernel 的组枚举
// 与 cp_utils.py::prepare_cp_compressor_input 的 rank-row 映射
// ============================================================================

// 一个 rank 的 compressor 输入：按序列枚举 [global_start - d_comp, global_start + l_local) 内
// 可见的完整压缩组，把每组的 ratio 个 token 从 boundary / local 拷进定容的 hidden_compact
function compactForward({ cu, globalStart, lLocal, ratio, dComp: dc, dWindow: dw, cCap }) {
  const rangeStart = globalStart;
  const rangeEnd = globalStart + lLocal;
  const firstRangeGroupStart = rangeStart - dc;
  const rows = []; // 每个压紧行：{src, comp, seq, from}
  for (let seq = 0; seq < cu.length - 1; seq += 1) {
    const seqStart = cu[seq];
    const seqEnd = cu[seq + 1];
    const localSeqEnd = Math.min(seqEnd, rangeEnd);
    if (seqStart < localSeqEnd && rangeStart < localSeqEnd) {
      const firstVisibleNumer = Math.max(firstRangeGroupStart - seqStart, 0);
      const firstVisibleGroup = Math.ceil(firstVisibleNumer / ratio);
      const stopVisibleGroup = Math.floor((localSeqEnd - seqStart) / ratio);
      for (let g = firstVisibleGroup; g < stopVisibleGroup; g += 1) {
        for (let t = 0; t < ratio; t += 1) {
          const src = seqStart + g * ratio + t;
          rows.push({
            src,
            comp: g,
            seq,
            from: src < rangeStart ? 'boundary' : 'local',
            srcRow: src < rangeStart ? src - (rangeStart - dw) : src - rangeStart,
          });
        }
      }
    }
  }
  const compactLen = cCap * ratio;
  if (rows.length > compactLen) throw new Error(`压紧行 ${rows.length} 超过容量 ${compactLen}`);
  const slots = [];
  for (let s = 0; s < cCap; s += 1) {
    const head = rows[s * ratio];
    slots.push(
      head
        ? { comp: head.comp, seq: head.seq, tokens: rows.slice(s * ratio, (s + 1) * ratio) }
        : { comp: -1, seq: -1, tokens: [] },
    );
  }
  return { rows, slots, compactLen };
}

// prepare_cp_compressor_input 的第二段：逻辑（sequence-major）压缩行 → rank-major 物理行
function seqToRankRow({ cu, cuComp, lLocal, cpSize, ratio, cCap }) {
  const dc = dComp(ratio);
  const nSeq = cu.length - 1;
  const seqMajorRows = Math.floor((lLocal * cpSize) / ratio);
  const rankStarts = Array.from({ length: cpSize }, (_, r) => r * lLocal);
  const firstLogicalRows = rankStarts.map((rs) => {
    const firstSeq = Math.min(bucketizeRight(rs, cu.slice(1)), nSeq - 1);
    const firstComp = Math.floor((Math.max(rs - dc - cu[firstSeq], 0) + ratio - 1) / ratio);
    return cuComp[firstSeq] + firstComp;
  });
  const map = [];
  for (let row = 0; row < seqMajorRows; row += 1) {
    const seq = Math.min(bucketizeRight(row, cuComp.slice(1)), nSeq - 1);
    const comp = row - cuComp[seq];
    const groupLastRow = cu[seq] + (comp + 1) * ratio - 1;
    const owner = Math.min(Math.max(Math.floor(groupLastRow / lLocal), 0), cpSize - 1);
    const slot = row - firstLogicalRows[owner];
    const rankRow = owner * cCap + slot;
    map.push(row < cuComp[cuComp.length - 1] ? rankRow : -1);
  }
  return { map, firstLogicalRows, seqMajorRows };
}

// ============================================================================
// 复刻 cp_layout_kernels.py::_build_attention_indices_kernel
//   index_mode 0：选中的 top-k（window 在前，压缩块在后，返回 topk_length）
//   index_mode 1：全部可见压缩块（无 indexer 的层）
//   index_mode 2：indexer-loss 布局（压缩块在前、window 在后，另返回 rank-major 行）
// ============================================================================

function buildAttentionIndices({
  cu, cuComp, globalStart, lLocal, dWindow: dw, windowSize, ratio, compressedWidth,
  compressedTopk, map, mode,
}) {
  const nSeq = cu.length - 1;
  const compressedBase = dw + lLocal;
  const totalWidth = windowSize + compressedWidth;
  const seqMajorRows = map.length;
  const rowsOut = [];
  for (let row = 0; row < lLocal; row += 1) {
    const globalQ = globalStart + row;
    let seqStartFound = -1;
    let seqCompStart = 0;
    let seqCompLen = 0;
    for (let seq = 0; seq < nSeq; seq += 1) {
      if (globalQ >= cu[seq] && globalQ < cu[seq + 1]) {
        seqStartFound = cu[seq];
        if (ratio > 1 && compressedWidth > 0) {
          seqCompStart = cuComp[seq];
          seqCompLen = cuComp[seq + 1] - seqCompStart;
        }
      }
    }
    const idx = new Array(totalWidth).fill(-1);
    const rankMajor = new Array(compressedWidth).fill(-1);
    let length = 0;
    const lower = (pos) => (pos < globalStart ? pos - (globalStart - dw) : dw + pos - globalStart);
    if (seqStartFound >= 0) {
      const windowStart = Math.max(globalQ - windowSize + 1, seqStartFound);
      const windowCount = globalQ - windowStart + 1;
      if (mode === 0) {
        let write = 0;
        for (let c = 0; c < windowCount; c += 1) idx[write++] = lower(windowStart + c);
        if (ratio > 1 && compressedWidth > 0) {
          for (let c = 0; c < compressedWidth; c += 1) {
            const comp = compressedTopk[row][c];
            if (comp >= 0 && comp < seqCompLen) {
              const seqMajorId = seqCompStart + comp;
              if (seqMajorId < seqMajorRows && map[seqMajorId] >= 0) {
                idx[write++] = compressedBase + map[seqMajorId];
              }
            }
          }
        }
        length = write;
      } else if (mode === 1) {
        let compCount = 0;
        if (ratio > 1 && compressedWidth > 0) {
          compCount = Math.min(
            Math.floor((globalQ - seqStartFound + 1) / ratio), compressedWidth, seqCompLen,
          );
        }
        length = windowCount + compCount;
        for (let col = 0; col < totalWidth; col += 1) {
          if (col < windowCount) idx[col] = lower(windowStart + col);
          else if (col < length) {
            const seqMajorId = seqCompStart + col - windowCount;
            if (seqMajorId < seqMajorRows && map[seqMajorId] >= 0) {
              idx[col] = compressedBase + map[seqMajorId];
            }
          }
        }
      } else {
        for (let col = 0; col < totalWidth; col += 1) {
          if (col < compressedWidth) {
            const comp = compressedTopk[row][col];
            if (comp >= 0 && comp < seqCompLen) {
              const seqMajorId = seqCompStart + comp;
              if (seqMajorId < seqMajorRows && map[seqMajorId] >= 0) {
                rankMajor[col] = map[seqMajorId];
                idx[col] = compressedBase + map[seqMajorId];
              }
            }
          } else {
            const windowCol = col - compressedWidth;
            if (windowCol < windowCount) idx[col] = lower(windowStart + windowCol);
          }
        }
      }
    } else if (totalWidth > 0 && mode !== 2) {
      idx[0] = 0;
      length = 1;
    }
    rowsOut.push({ idx, length, rankMajor, windowCount: seqStartFound >= 0 ? Math.min(windowSize, globalQ - seqStartFound + 1) : 0 });
  }
  return { rows: rowsOut, compressedBase };
}

// compute_cp_indexer_topk 的可见块数：min((pos + 1) // ratio, seq 的压缩行数)
function visibleCompressed(cu, cuComp, globalQ, ratio) {
  const nSeq = cu.length - 1;
  const seq = Math.min(bucketizeRight(globalQ, cu.slice(1)), nSeq - 1);
  const valid = globalQ >= cu[seq] && globalQ < cu[seq + 1];
  if (!valid) return { seq, pos: 0, visible: 0 };
  const pos = globalQ - cu[seq];
  return { seq, pos, visible: Math.min(Math.floor((pos + 1) / ratio), cuComp[seq + 1] - cuComp[seq]) };
}

// 图上用「最近 k 个可见压缩块」代替 indexer 打分；输出仍是 compute_cp_indexer_topk 的契约：
// 序列内逻辑块 id，不足补 -1
function demoTopk(cu, cuComp, globalStart, lLocal, ratio, topk) {
  const out = [];
  for (let row = 0; row < lLocal; row += 1) {
    const { visible } = visibleCompressed(cu, cuComp, globalStart + row, ratio);
    const sel = [];
    for (let c = visible - 1; c >= 0 && sel.length < topk; c -= 1) sel.push(c);
    while (sel.length < topk) sel.push(-1);
    out.push(sel);
  }
  return out;
}

// ============================================================================
// 图 1 的算例
// ============================================================================

function runLayout(cfg) {
  const cu = cumsum(cfg.seqLens);
  const total = cu[cu.length - 1];
  if (total % cfg.cpSize !== 0) throw new Error('contiguous 切分要求 total_tokens 能被 cp_size 整除');
  const lLocal = total / cfg.cpSize;
  const cuComp = cumsum(cfg.seqLens.map((l) => Math.floor(l / cfg.ratio)));
  const dc = dComp(cfg.ratio);
  const dw = dWindow(cfg.windowSize, cfg.ratio);
  if (lLocal < dw) throw new Error('local_rows < d_window：源码会在边界交换前 RuntimeError');
  const cCap = compactCapacity(lLocal, cfg.ratio);
  const rankRow = seqToRankRow({ cu, cuComp, lLocal, cpSize: cfg.cpSize, ratio: cfg.ratio, cCap });
  const ranks = [];
  for (let r = 0; r < cfg.cpSize; r += 1) {
    const globalStart = r * lLocal;
    const boundaryRows = Array.from({ length: dw }, (_, i) => globalStart - dw + i); // <0 为无效行
    const compact = compactForward({ cu, globalStart, lLocal, ratio: cfg.ratio, dComp: dc, dWindow: dw, cCap });
    // 该槽的压缩组是否由本 rank 拥有：所有权 = 组末 token 所在 rank
    const owned = compact.slots.map((s) => {
      if (s.comp < 0) return 'pad';
      const lastTok = cu[s.seq] + (s.comp + 1) * cfg.ratio - 1;
      return Math.floor(lastTok / lLocal) === r ? 'owned' : 'dup';
    });
    const topk = demoTopk(cu, cuComp, globalStart, lLocal, cfg.ratio, cfg.topk);
    const positions = thdCpPositionIds(cu, globalStart, lLocal);
    const boundaryPositions = thdCpPositionIds(cu, globalStart - dw, dw);
    const selected = buildAttentionIndices({
      cu, cuComp, globalStart, lLocal, dWindow: dw, windowSize: cfg.windowSize, ratio: cfg.ratio,
      compressedWidth: cfg.topk, compressedTopk: topk, map: rankRow.map, mode: 0,
    });
    const allVisible = buildAttentionIndices({
      cu, cuComp, globalStart, lLocal, dWindow: dw, windowSize: cfg.windowSize, ratio: cfg.ratio,
      compressedWidth: Math.floor(Math.max(...cfg.seqLens) / cfg.ratio), compressedTopk: null, map: rankRow.map, mode: 1,
    });
    const lossLayout = buildAttentionIndices({
      cu, cuComp, globalStart, lLocal, dWindow: dw, windowSize: cfg.windowSize, ratio: cfg.ratio,
      compressedWidth: cfg.topk, compressedTopk: topk, map: rankRow.map, mode: 2,
    });
    const indexerLayout = buildCpIndexerLayout(cu, cuComp, globalStart, lLocal);
    ranks.push({
      rank: r, globalStart, boundaryRows, compact, owned, topk, positions, boundaryPositions,
      selected, allVisible, lossLayout, indexerLayout,
      counts: {
        owned: owned.filter((o) => o === 'owned').length,
        dup: owned.filter((o) => o === 'dup').length,
        pad: owned.filter((o) => o === 'pad').length,
      },
    });
  }
  // 两阶段各搬多少行（以「行」计，不带宽度；宽度对比在图 3）
  const stage1RowsRecv = ranks.map((r) => (r.rank > 0 ? dw : 0));
  const gatheredRows = cfg.cpSize * cCap; // 每个 AllGather 的缓冲行数
  const realCompressedRows = cuComp[cuComp.length - 1];
  const droppedTail = cfg.seqLens.reduce((s, l) => s + (l % cfg.ratio), 0);
  const naiveRows = total; // 朴素：全量 KV AllGather 的行数
  return {
    cu, cuComp, total, lLocal, dc, dw, cCap, rankRow, ranks,
    stage1RowsRecv, gatheredRows, realCompressedRows, droppedTail, naiveRows,
    dupRows: ranks.reduce((s, r) => s + r.counts.dup, 0),
    padRows: ranks.reduce((s, r) => s + r.counts.pad, 0),
  };
}

const LAYOUT = runLayout(CFG);

// 图 1 面板 D：三条 rank1 query 的降级表
function demoQueryRows(layout, queries) {
  const r1 = layout.ranks[layout.ranks.length - 1];
  return queries.map((q) => {
    const row = q - r1.globalStart;
    const { seq, pos, visible } = visibleCompressed(layout.cu, layout.cuComp, q, CFG.ratio);
    const sel = r1.selected.rows[row];
    const windowIdx = sel.idx.slice(0, sel.windowCount);
    const windowGlobal = windowIdx.map((i) => (i < layout.dw ? r1.globalStart - layout.dw + i : r1.globalStart + i - layout.dw));
    const topk = r1.topk[row].filter((c) => c >= 0);
    const seqMajor = topk.map((c) => layout.cuComp[seq] + c);
    const rankMajor = seqMajor.map((s) => layout.rankRow.map[s]);
    const compIdx = rankMajor.map((rm) => r1.selected.compressedBase + rm);
    return { q, row, seq, pos, visible, windowGlobal, windowIdx, topk, seqMajor, rankMajor, compIdx, length: sel.length, idx: sel.idx };
  });
}

const DEMO = demoQueryRows(LAYOUT, CFG.demoQueries);

// ============================================================================
// 图 2：前向语句序 + 反向 autograd 顺序复刻
// ============================================================================

// 前向（csa.py::_forward_thd_cp 的语句序；Stage 1 来自 deepseek_v4_hybrid_attention.py::forward）
const FORWARD_INDEXER = Object.freeze([
  { lane: 'comm', label: 'P2P 左边界 hidden', kind: 'block', sym: '_LeftBoundaryExchange.forward' },
  { lane: 'compute', label: 'Q / KV 投影 + CP-aware RoPE（含 boundary KV）', kind: 'work', sym: 'qkv_up_proj_and_rope_apply' },
  { lane: 'compute', label: '压紧 hidden_compact', kind: 'work', sym: 'CompressorInputCompact.forward' },
  { lane: 'compute', label: 'Indexer-K compressor', kind: 'work', sym: 'indexer.compressor._forward_thd' },
  { lane: 'comm', label: 'AG Indexer-K', bar: 'AG-K', kind: 'async', sym: 'async_gather_from_sequence_parallel_region', launchAfter: 3, waitBefore: 8 },
  { lane: 'compute', label: 'attention-KV compressor', kind: 'work', sym: 'self.compressor._forward_thd' },
  { lane: 'comm', label: 'AG 压缩 KV', bar: 'AG-KV', kind: 'async', sym: 'async_gather_from_sequence_parallel_region', launchAfter: 5, waitBefore: 10 },
  { lane: 'compute', label: 'Indexer Q 投影 + RoPE + weights 投影', kind: 'work', sym: 'apply_thd_cp_local_rope_*' },
  { lane: 'compute', label: 'wait K', kind: 'wait', sym: 'k_indexer_gather.wait' },
  { lane: 'compute', label: 'seq-major 重排 + top-k', kind: 'work', sym: 'compute_cp_indexer_topk' },
  { lane: 'compute', label: 'wait KV', kind: 'wait', sym: 'compressed_kv_gather.wait' },
  { lane: 'compute', label: 'cat + build_attention_indices', kind: 'work', sym: 'build_attention_indices' },
  { lane: 'compute', label: '融合稀疏注意力 + indexer loss', kind: 'work', sym: 'FusedCSAIndexerSparseAttnFromTopkFunc' },
]);

const FORWARD_NO_INDEXER = Object.freeze([
  { lane: 'comm', label: 'P2P 左边界 hidden', kind: 'block', sym: '_LeftBoundaryExchange.forward' },
  { lane: 'compute', label: 'Q / KV 投影 + CP-aware RoPE', kind: 'work', sym: 'qkv_up_proj_and_rope_apply' },
  { lane: 'compute', label: '压紧 hidden_compact', kind: 'work', sym: 'CompressorInputCompact.forward' },
  { lane: 'compute', label: 'attention-KV compressor', kind: 'work', sym: 'self.compressor._forward_thd' },
  { lane: 'comm', label: '同步 AG 压缩 KV', kind: 'block', sym: 'gather_from_sequence_parallel_region' },
  { lane: 'compute', label: 'build_attention_indices（全部可见块）', kind: 'work', sym: 'build_attention_indices' },
  { lane: 'compute', label: '稀疏注意力', kind: 'work', sym: 'csa_sparse_attn' },
]);

// 反向：一个最小 autograd 引擎——节点按前向创建顺序编号，就绪节点里序号最大的先执行
// （PyTorch 引擎的 sequence_nr 优先级）。节点的 backward 可以「发起」或「等待」集合通信。
function autogradOrder(nodes) {
  // nodes: [{ name, inputs: [names], run: (events) => void }]，按创建顺序给出
  const byName = new Map(nodes.map((n, i) => [n.name, { ...n, seq: i }]));
  const pending = new Map(); // name -> 尚未到达的梯度数
  for (const n of byName.values()) pending.set(n.name, 0);
  for (const n of byName.values()) for (const inp of n.inputs) pending.set(inp, pending.get(inp) + 1);
  const ready = [byName.get(nodes[nodes.length - 1].name)];
  const events = [];
  const executed = [];
  while (ready.length) {
    ready.sort((a, b) => b.seq - a.seq);
    const node = ready.shift();
    if (node.run) node.run(events);
    executed.push(node.name);
    for (const inp of node.inputs) {
      pending.set(inp, pending.get(inp) - 1);
      if (pending.get(inp) === 0) ready.push(byName.get(inp));
    }
  }
  return { events, executed };
}

// (a) 锁定 TestCPCommunicationOverlap::test_deferred_reduce_scatter_waits_follow_consumer_branch_order
function replayBranchOrderTest() {
  const nodes = [
    { name: 'indexer_compressor', inputs: [], run: (e) => e.push('indexer_compressor') },
    { name: 'indexer_edge', inputs: ['indexer_compressor'], run: (e) => e.push('wait_indexer') },
    { name: 'attention_kv_compressor', inputs: [], run: (e) => e.push('attention_kv_compressor') },
    { name: 'compressed_kv_edge', inputs: ['attention_kv_compressor'], run: (e) => e.push('wait_compressed_kv') },
    { name: 'q_weight_branch', inputs: [], run: (e) => e.push('q_weight_branch') },
    { name: 'fused', inputs: ['indexer_edge', 'compressed_kv_edge', 'q_weight_branch'], run: (e) => e.push('fused') },
  ];
  return autogradOrder(nodes).events;
}

// (b) FusedCSAIndexerSparseAttnFromTopkFunc.backward 内部的发起顺序
//     （test_cp_backward_launches_collectives_in_dependency_order）
const FUSED_BACKWARD_INTERNAL = Object.freeze([
  'sparse_attention_backward',
  'launch_compressed_kv',
  'launch_indexer',
  'local_indexer_grads',
]);

// (c) 整层反向：按 _forward_thd_cp / DSv4HybridAttention.forward 的创建顺序建图
function replayLayerBackward() {
  const nodes = [
    { name: 'p2p_boundary', inputs: [], run: (e) => e.push({ lane: 'comm', label: 'P2P 边界梯度回送（阻塞）', kind: 'block', sym: '_LeftBoundaryExchange.backward' }) },
    { name: 'kv_proj', inputs: ['p2p_boundary'], run: (e) => e.push({ lane: 'compute', label: 'KV 投影反向（boundary KV → boundary hidden）', kind: 'work', sym: 'linear_kv_proj' }) },
    { name: 'q_proj', inputs: [], run: (e) => e.push({ lane: 'compute', label: 'Q 投影反向', kind: 'work', sym: 'linear_q_up_proj' }) },
    { name: 'compact', inputs: ['p2p_boundary'], run: (e) => e.push({ lane: 'compute', label: '压紧反向：散射到 local + boundary', kind: 'work', sym: 'CompressorInputCompact.backward' }) },
    { name: 'indexer_k_compressor', inputs: ['compact'], run: (e) => e.push({ lane: 'compute', label: 'Indexer-K compressor 反向', kind: 'work', sym: 'indexer.compressor' }) },
    { name: 'indexer_k_edge', inputs: ['indexer_k_compressor'], run: (e) => e.push({ lane: 'compute', label: 'wait RS-K', kind: 'wait', sym: '_WaitForDeferredReduceScatter' }) },
    { name: 'attention_kv_compressor', inputs: ['compact'], run: (e) => e.push({ lane: 'compute', label: 'attention-KV compressor 反向', kind: 'work', sym: 'self.compressor' }) },
    { name: 'compressed_kv_edge', inputs: ['attention_kv_compressor'], run: (e) => e.push({ lane: 'compute', label: 'wait RS-KV', kind: 'wait', sym: '_WaitForDeferredReduceScatter' }) },
    { name: 'indexer_q_weights', inputs: [], run: (e) => e.push({ lane: 'compute', label: 'Indexer Q / weights 投影反向', kind: 'work', sym: 'linear_wq_b / linear_weights_proj' }) },
    {
      name: 'fused',
      inputs: ['q_proj', 'kv_proj', 'indexer_k_edge', 'compressed_kv_edge', 'indexer_q_weights'],
      run: (e) => {
        e.push({ lane: 'compute', label: '稀疏注意力反向', kind: 'work', sym: 'sparse_attention_backward_wrapper' });
        e.push({ lane: 'comm', label: 'RS 压缩 KV 发起', bar: 'RS-KV', kind: 'async', sym: 'async_reduce_scatter_along_first_dim' });
        e.push({ lane: 'comm', label: 'RS Indexer-K 发起', bar: 'RS-K', kind: 'async', sym: 'async_reduce_scatter_along_first_dim' });
        e.push({ lane: 'compute', label: '本地 indexer 梯度缩放', kind: 'work', sym: 'saved_grad_q_indexer * grad_loss' });
      },
    },
  ];
  return autogradOrder(nodes);
}

const BRANCH_ORDER = replayBranchOrderTest();
const LAYER_BACKWARD = replayLayerBackward();

// 图 2 的立论前提：wait 只出现在消费分支上，且 RS-KV 的 wait 先于 RS-K 的 wait
{
  const labels = LAYER_BACKWARD.events.map((e) => e.label);
  const iKV = labels.indexOf('wait RS-KV');
  const iK = labels.indexOf('wait RS-K');
  const iQ = labels.indexOf('Indexer Q / weights 投影反向');
  if (!(iQ < iKV && iKV < iK)) throw new Error('反向顺序复刻与源码测试断言不一致');
}

// ============================================================================
// 图 3：真实配方的通信量
// ============================================================================

const MiB = 1024 * 1024;

function recipeCost(rec, cpSize) {
  const lLocal = rec.seqLength / cpSize;
  const dw = dWindow(rec.windowSize, 4);
  const boundaryBytes = dw * rec.hiddenSize * rec.dtypeBytes; // 每个内部 rank 收 1 份、发 1 份
  const perRatio = {};
  for (const ratio of rec.ratios) {
    const cCap = compactCapacity(lLocal, ratio);
    const gatheredRows = cpSize * cCap;
    const realRows = Math.floor(rec.seqLength / ratio);
    const kvBuffer = gatheredRows * rec.vHeadDim * rec.dtypeBytes;
    const kvRecv = (cpSize - 1) * cCap * rec.vHeadDim * rec.dtypeBytes;
    const hasIndexer = ratio === 4;
    const kBuffer = hasIndexer ? gatheredRows * rec.indexerHeadDim * rec.dtypeBytes : 0;
    const kRecv = hasIndexer ? (cpSize - 1) * cCap * rec.indexerHeadDim * rec.dtypeBytes : 0;
    perRatio[ratio] = { ratio, cCap, gatheredRows, realRows, padRows: gatheredRows - realRows, kvBuffer, kvRecv, kBuffer, kRecv, hasIndexer };
  }
  const naiveBuffer = rec.seqLength * rec.vHeadDim * rec.dtypeBytes;
  const naiveRecv = (cpSize - 1) * lLocal * rec.vHeadDim * rec.dtypeBytes;
  return { cpSize, lLocal, dw, boundaryBytes, perRatio, naiveBuffer, naiveRecv };
}

const COST16 = recipeCost(RECIPE, RECIPE.cpSize);
const SWEEP = RECIPE.cpSweep.map((cp) => {
  const c = recipeCost(RECIPE, cp);
  const r4 = c.perRatio[4];
  return { cp, lLocal: c.lLocal, recvBytes: r4.kvRecv + r4.kRecv, gatheredRows: r4.gatheredRows, padRows: r4.padRows, boundaryBytes: c.boundaryBytes };
});
const CP_LIMIT = RECIPE.seqLength / dWindow(RECIPE.windowSize, 4); // local_rows >= d_window 的上限

const fmtMiB = (bytes) => (bytes / MiB).toFixed(2);

if (process.env.DSV4_FIG_DEBUG) {
  console.log('LAYOUT', JSON.stringify({ cu: LAYOUT.cu, cuComp: LAYOUT.cuComp, lLocal: LAYOUT.lLocal, dc: LAYOUT.dc, dw: LAYOUT.dw, cCap: LAYOUT.cCap, map: LAYOUT.rankRow.map, first: LAYOUT.rankRow.firstLogicalRows }));
  for (const r of LAYOUT.ranks) console.log(`rank${r.rank}`, JSON.stringify({ slots: r.compact.slots.map((s) => `${s.seq}·g${s.comp}`), owned: r.owned, counts: r.counts, layout: r.indexerLayout }));
  console.log('DEMO', JSON.stringify(DEMO));
  console.log('BRANCH', JSON.stringify(BRANCH_ORDER));
  console.log('LAYER', JSON.stringify(LAYER_BACKWARD.events.map((e) => e.label)));
  console.log('COST16', JSON.stringify(COST16));
  console.log('SWEEP', JSON.stringify(SWEEP));
}

// ============================================================================
// SVG 基础设施（与 tools/figs/svg/megatron_inference_engine_figures.mjs 同一套 token）
// ============================================================================

const esc = (value) =>
  String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

function textWidth(value, fontSize) {
  let units = 0;
  for (const ch of String(value)) units += ch.charCodeAt(0) < 0x7f ? 0.56 : 1;
  return units * fontSize;
}

function guard(value, fontSize, limit, where) {
  const width = textWidth(value, fontSize);
  if (width > limit) {
    throw new Error(`${where}: "${value}" 需要 ${width.toFixed(1)}px，超出 ${limit}px`);
  }
  return value;
}

const sharedStyle = `
  text{font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif}
  .card{fill:#fff;stroke:#E4E7EC;stroke-width:1.2}
  .panel{fill:#FBFCFE;stroke:#D9DEE7;stroke-width:1.2}
  .neutral{fill:#fff;stroke:#AEB6C2;stroke-width:1.2}
  .ghost{fill:#F5F7FA;stroke:#D9DEE7;stroke-width:1.1}
  .acc1{fill:#EAF1FD;stroke:#2563EB;stroke-width:1.5}
  .acc2{fill:#FCF1E6;stroke:#C3651F;stroke-width:1.5}
  .acc1fill{fill:#2563EB;stroke:#2563EB;stroke-width:1}
  .acc2fill{fill:#C3651F;stroke:#C3651F;stroke-width:1}
  .s0{fill:#EAF1FD;stroke:#2563EB;stroke-width:1.1}
  .s1{fill:#FCF1E6;stroke:#C3651F;stroke-width:1.1}
  .bar{fill:#F4C9A3;stroke:#C3651F;stroke-width:.8}
  .bar1{fill:#CFE0FA;stroke:#2563EB;stroke-width:.8}
  .main{fill:none;stroke:#2563EB;stroke-width:2;marker-end:url(#arrowMain)}
  .aux{fill:none;stroke:#AEB6C2;stroke-width:1.4;stroke-dasharray:5 4;marker-end:url(#arrowAux)}
  .cost{fill:none;stroke:#C3651F;stroke-width:2;marker-end:url(#arrowCost)}
  .edge{fill:none;stroke:#AEB6C2;stroke-width:1.4;stroke-dasharray:4 4}
  .line1{fill:none;stroke:#2563EB;stroke-width:2}
  .line2{fill:none;stroke:#C3651F;stroke-width:2;stroke-dasharray:6 4}
  .ti{font-size:18px;font-weight:700;fill:#1F2430}
  .su{font-size:11.5px;fill:#747C88}
  .pt{font-size:14px;font-weight:700;fill:#2A313B}
  .tx{font-size:12px;fill:#38414D}
  .sm{font-size:10.5px;fill:#68717D}
  .dim{font-size:10.5px;font-weight:600;fill:#173F87}
  .costtx{font-size:10.5px;font-weight:600;fill:#8A4A11}
  .rank{font-size:11px;font-weight:700;fill:#5B6470}
  .cap{font-size:11px;fill:#747C88}
  .gl{fill:none;stroke:#C8CFDA;stroke-width:.9}
  .sep{fill:none;stroke:#5B6470;stroke-width:1.6}
`;

const defs = `
  <defs>
    <marker id="arrowMain" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#2563EB"/></marker>
    <marker id="arrowAux" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#AEB6C2"/></marker>
    <marker id="arrowCost" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#C3651F"/></marker>
  </defs>`;

function rect(x, y, w, h, cls = 'neutral', radius = 8) {
  return `<rect class="${cls}" x="${x}" y="${y}" width="${w}" height="${h}" rx="${radius}"/>`;
}
function text(x, y, value, cls = 'tx', anchor = 'start') {
  return `<text class="${cls}" x="${x}" y="${y}" text-anchor="${anchor}">${esc(value)}</text>`;
}
function line(x1, y1, x2, y2, cls = 'edge') {
  return `<path class="${cls}" d="M ${x1} ${y1} L ${x2} ${y2}"/>`;
}
function polyline(points, cls) {
  return `<path class="${cls}" d="M ${points.map(([x, y]) => `${x.toFixed(1)} ${y.toFixed(1)}`).join(' L ')}"/>`;
}
function infoBox(x, y, w, h, title, lines, cls = 'neutral', where = 'box') {
  const out = [rect(x, y, w, h, cls)];
  const inner = w - 24;
  out.push(text(x + 12, y + 21, guard(title, 12, inner, `${where}/title`), 'tx'));
  out.push(line(x + 12, y + 29, x + w - 12, y + 29));
  lines.forEach((entry, index) => {
    const value = typeof entry === 'string' ? entry : entry.text;
    const lineCls = typeof entry === 'string' ? 'sm' : entry.cls;
    out.push(
      text(x + 12, y + 47 + index * 16, guard(value, 10.5, inner, `${where}/L${index}`), lineCls),
    );
  });
  return out.join('\n');
}
function header(w, title, subtitle) {
  return [
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} __H__" width="${w}" height="__H__" role="img">`,
    defs,
    `<style>${sharedStyle}</style>`,
    text(28, 32, title, 'ti'),
    text(28, 52, subtitle, 'su'),
  ];
}

const FONT = Object.freeze({
  ti: 18, su: 11.5, pt: 14, tx: 12, sm: 10.5, dim: 10.5,
  costtx: 10.5, rank: 11, cap: 11,
});

function assertNoTextOverlap(svg, name) {
  const { w: canvasW, h: canvasH } = (() => {
    const m = svg.match(/viewBox="0 0 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)"/);
    return { w: Number(m[1]), h: Number(m[2]) };
  })();
  const boxes = [];
  for (const m of svg.matchAll(
    /<text class="([a-z0-9]+)" x="(-?[\d.]+)" y="(-?[\d.]+)" text-anchor="(\w+)">([^<]*)<\/text>/g,
  )) {
    const [, cls, xs, ys, anchor, raw] = m;
    const size = FONT[cls];
    if (!size || raw.trim() === '') continue;
    const w = textWidth(raw.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>'), size);
    const x = anchor === 'middle' ? Number(xs) - w / 2 : anchor === 'end' ? Number(xs) - w : Number(xs);
    boxes.push({ x, y: Number(ys) - size * 0.78, w, h: size * 1.06, raw });
  }
  for (const b of boxes) {
    if (b.x < -1 || b.y < -1 || b.x + b.w > canvasW + 1 || b.y + b.h > canvasH + 1) {
      throw new Error(
        `${name}: 文字盒出画布 "${b.raw}"（${b.x.toFixed(1)}..${(b.x + b.w).toFixed(1)} × ` +
          `${b.y.toFixed(1)}..${(b.y + b.h).toFixed(1)}，画布 ${canvasW}×${canvasH}）`,
      );
    }
  }
  for (let i = 0; i < boxes.length; i += 1) {
    for (let j = i + 1; j < boxes.length; j += 1) {
      const a = boxes[i];
      const b = boxes[j];
      const dx = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
      const dy = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
      if (dx > 1 && dy > 1) {
        throw new Error(`${name}: 文字重叠 "${a.raw}" × "${b.raw}"（重合 ${dx.toFixed(1)}×${dy.toFixed(1)}px）`);
      }
    }
  }
  return svg;
}

const seal = (parts, w, h, name) =>
  assertNoTextOverlap(parts.join('\n').replace(/__H__/g, String(h)) + '\n</svg>', name);

// ============================================================================
// 图 1：布局
// ============================================================================

function renderLayout() {
  const W = 1272;
  const X0 = 28;
  const L = LAYOUT;
  const parts = header(
    W,
    '图 1　DSv4 THD CP 的布局：contiguous 切分、左边界 8 行、定容压紧槽位与 rank-major 压缩行',
    `算例：CP=${CFG.cpSize}，全局 ${L.total} 个 token 打包成 ${CFG.seqLens.join('+')} 两条序列；csa_window_size=${CFG.windowSize}，ratio=${CFG.ratio}（d_comp=${L.dc}，d_window=${L.dw}），indexer top-k=${CFG.topk}；每 rank ${L.lLocal} 行，c_cap=${L.cCap}`,
  );

  // ---- 面板 A：全局 token 行 ----
  const CELL = 27;
  const AX = 120;
  const AY = 96;
  parts.push(text(X0, AY - 10, 'A　全局 THD 行与 contiguous 切分', 'pt'));
  // 序列括号
  for (let s = 0; s < L.cu.length - 1; s += 1) {
    const x1 = AX + L.cu[s] * CELL;
    const x2 = AX + L.cu[s + 1] * CELL - 3;
    parts.push(line(x1, AY + 4, x2, AY + 4, 'sep'));
    parts.push(text((x1 + x2) / 2, AY, `seq${s}（${CFG.seqLens[s]} 个 token，${Math.floor(CFG.seqLens[s] / CFG.ratio)} 个压缩组）`, 'sm', 'middle'));
  }
  const r1 = L.ranks[1];
  const boundarySet = new Set(r1.boundaryRows);
  for (let t = 0; t < L.total; t += 1) {
    const x = AX + t * CELL;
    const y = AY + 10;
    const cls = boundarySet.has(t) ? 'acc1' : 'neutral';
    parts.push(rect(x, y, CELL - 3, 20, cls, 3));
    parts.push(text(x + (CELL - 3) / 2, y + 14, String(t), boundarySet.has(t) ? 'dim' : 'sm', 'middle'));
  }
  parts.push(text(AX - 6, AY + 24, 'token', 'sm', 'end'));
  // 压缩组行
  const GY = AY + 36;
  parts.push(text(AX - 6, GY + 14, '压缩组', 'sm', 'end'));
  for (let s = 0; s < L.cu.length - 1; s += 1) {
    const nGroups = Math.floor(CFG.seqLens[s] / CFG.ratio);
    for (let g = 0; g < nGroups; g += 1) {
      const x = AX + (L.cu[s] + g * CFG.ratio) * CELL;
      parts.push(rect(x, GY, CELL * CFG.ratio - 3, 20, 'neutral', 3));
      parts.push(text(x + (CELL * CFG.ratio - 3) / 2, GY + 14, `seq${s}·g${g}`, 'sm', 'middle'));
    }
    const tail = CFG.seqLens[s] % CFG.ratio;
    if (tail > 0) {
      const x = AX + (L.cu[s] + nGroups * CFG.ratio) * CELL;
      parts.push(rect(x, GY, CELL * tail - 3, 20, 'ghost', 3));
      parts.push(text(x + (CELL * tail - 3) / 2, GY + 14, '×', 'costtx', 'middle'));
    }
  }
  // rank 行
  const RY = GY + 28;
  parts.push(text(AX - 6, RY + 14, 'rank', 'sm', 'end'));
  for (const r of L.ranks) {
    const x = AX + r.globalStart * CELL;
    parts.push(rect(x, RY, CELL * L.lLocal - 3, 20, r.rank === 0 ? 's0' : 's1', 3));
    parts.push(text(x + (CELL * L.lLocal - 3) / 2, RY + 14, `rank${r.rank}：全局行 [${r.globalStart}, ${r.globalStart + L.lLocal})，global_start=${r.globalStart}`, 'sm', 'middle'));
  }
  const legendY = RY + 36;
  parts.push(text(X0, legendY, guard(`蓝框 = rank1 的左边界 d_window=${L.dw} 行（全局 ${r1.boundaryRows[0]}..${r1.boundaryRows[L.dw - 1]}，rank0 尾部经 P2P 送达）；× = 不足一个 ratio 组的尾巴，不生成压缩行（共 ${L.droppedTail} 个）；rank0 的边界行全为无效行`, 11, W - 2 * X0, 'fig1/capA'), 'cap'));

  // ---- 面板 B：每个 rank 的 compressor 输入 ----
  const BY = legendY + 30;
  parts.push(text(X0, BY, 'B　每个 rank 的 compressor 输入：boundary + local → 定容槽位（每槽 ratio=4 个 token）', 'pt'));
  const SLOT = 84;
  L.ranks.forEach((r, k) => {
    const y = BY + 14 + k * 62;
    parts.push(text(X0, y + 14, `rank${r.rank}`, 'rank'));
    // boundary | local 条
    const bx = X0 + 44;
    parts.push(rect(bx, y, 84, 22, r.rank === 0 ? 'ghost' : 'acc1', 3));
    parts.push(text(bx + 42, y + 15, r.rank === 0 ? `boundary ${L.dw}（无效）` : `boundary ${L.dw}`, r.rank === 0 ? 'sm' : 'dim', 'middle'));
    parts.push(rect(bx + 88, y, 118, 22, r.rank === 0 ? 's0' : 's1', 3));
    parts.push(text(bx + 88 + 59, y + 15, `local ${L.lLocal}（${r.globalStart}..${r.globalStart + L.lLocal - 1}）`, 'sm', 'middle'));
    parts.push(line(bx + 214, y + 11, bx + 240, y + 11, 'main'));
    // 槽位
    const sx = bx + 246;
    r.compact.slots.forEach((s, i) => {
      const cls = r.owned[i] === 'owned' ? 'neutral' : r.owned[i] === 'dup' ? 'acc2' : 'ghost';
      parts.push(rect(sx + i * SLOT, y, SLOT - 4, 22, cls, 3));
      const label = s.comp < 0 ? 'pad' : `seq${s.seq}·g${s.comp}`;
      parts.push(text(sx + i * SLOT + (SLOT - 4) / 2, y + 15, label, r.owned[i] === 'dup' ? 'costtx' : 'sm', 'middle'));
      parts.push(text(sx + i * SLOT + (SLOT - 4) / 2, y + 34, `s${i}`, 'sm', 'middle'));
      if (s.comp >= 0) {
        const src = s.tokens.map((t) => t.src);
        parts.push(text(sx + i * SLOT + (SLOT - 4) / 2, y + 47, `${src[0]}..${src[src.length - 1]}${s.tokens.some((t) => t.from === 'boundary') ? '·含边界' : ''}`, 'sm', 'middle'));
      }
    });
  });
  const BY2 = BY + 14 + L.ranks.length * 62 + 4;
  parts.push(text(X0, BY2, guard(`橙框 = 重算但不拥有的组（所有权 = 组末 token 所在 rank；rank1 为 overlap 重算 seq0·g2/g3），共 ${L.dupRows} 槽；ghost = 容量补齐（c_cap 对齐到 ${32 / gcd(32, CFG.ratio)} 的倍数），共 ${L.padRows} 槽；comp_id × ratio 是压缩行的 RoPE 位置`, 11, W - 2 * X0, 'fig1/capB'), 'cap'));

  // ---- 面板 C：rank-major 缓冲与 seq_to_rank_row ----
  const CY = BY2 + 30;
  parts.push(text(X0, CY, 'C　AllGather 之后的 rank-major 缓冲（两个 gather 共用这张行号表）与 seq_to_rank_row', 'pt'));
  const CX = X0 + 60;
  const CS = 64;
  L.ranks.forEach((r, k) => {
    r.compact.slots.forEach((s, i) => {
      const x = CX + (k * L.cCap + i) * CS;
      const cls = r.owned[i] === 'owned' ? (k === 0 ? 's0' : 's1') : r.owned[i] === 'dup' ? 'acc2' : 'ghost';
      parts.push(rect(x, CY + 12, CS - 4, 22, cls, 3));
      parts.push(text(x + (CS - 4) / 2, CY + 27, s.comp < 0 ? 'pad' : `seq${s.seq}·g${s.comp}`, r.owned[i] === 'dup' ? 'costtx' : 'sm', 'middle'));
      parts.push(text(x + (CS - 4) / 2, CY + 46, `行 ${k * L.cCap + i}`, 'sm', 'middle'));
    });
  });
  parts.push(text(CX - 6, CY + 27, 'rank-major', 'sm', 'end'));
  // 逻辑行 → 物理行
  const MY = CY + 62;
  parts.push(text(CX - 6, MY + 15, '逻辑行', 'sm', 'end'));
  L.rankRow.map.forEach((phys, logical) => {
    const x = CX + logical * CS;
    parts.push(rect(x, MY, CS - 4, 22, phys >= 0 ? 'acc1' : 'ghost', 3));
    parts.push(text(x + (CS - 4) / 2, MY + 15, phys >= 0 ? `${logical} → 行 ${phys}` : `${logical} → −1`, phys >= 0 ? 'dim' : 'sm', 'middle'));
    if (phys >= 0) {
      const tx = CX + phys * CS + (CS - 4) / 2;
      parts.push(line(x + (CS - 4) / 2, MY, tx, CY + 36, 'aux'));
    }
  });
  parts.push(text(X0, MY + 40, guard(`seq_to_rank_row = [${L.rankRow.map.join(', ')}]：sequence-major 的 ${L.realCompressedRows} 条真实压缩行落在 ${L.gatheredRows} 行缓冲里；−1 = 超出 cu_seqlens_compressed[-1] 的容量尾巴；重算槽与 pad 槽永远不被引用`, 11, W - 2 * X0, 'fig1/capC'), 'cap'));

  // ---- 面板 D：三条 query 的索引降级 ----
  const DY = MY + 66;
  parts.push(text(X0, DY, `D　rank1 三条 query 的索引降级（build_attention_indices，index_mode=0）：window 在前、top-k 压缩块在后，compressed_base = d_window + l_local = ${r1.selected.compressedBase}`, 'pt'));
  const cols = [X0, X0 + 150, X0 + 320, X0 + 480, X0 + 570, X0 + 690, X0 + 810, X0 + 940, X0 + 1090];
  const headers = ['query（全局 / seq·pos）', 'window 全局行', '→ kv_full 下标', '可见压缩块', 'top-2 逻辑块', '→ seq-major 行', '→ rank-major 行', '→ kv_full 下标', 'topk_length'];
  const TY = DY + 12;
  parts.push(rect(X0 - 4, TY, W - 2 * X0 + 8, 24 * (DEMO.length + 1) + 10, 'panel'));
  headers.forEach((h, i) => parts.push(text(cols[i], TY + 18, h, 'rank')));
  parts.push(line(X0, TY + 24, W - X0, TY + 24));
  DEMO.forEach((d, k) => {
    const y = TY + 42 + k * 24;
    const vals = [
      `${d.q}（seq${d.seq}·pos ${d.pos}）`,
      `[${d.windowGlobal.join(', ')}]`,
      `[${d.windowIdx.join(', ')}]`,
      String(d.visible),
      d.topk.length ? `[g${d.topk.join(', g')}]` : '[]',
      d.seqMajor.length ? `[${d.seqMajor.join(', ')}]` : '[]',
      d.rankMajor.length ? `[${d.rankMajor.join(', ')}]` : '[]',
      d.compIdx.length ? `[${d.compIdx.join(', ')}]` : '[]',
      String(d.length),
    ];
    vals.forEach((v, i) => parts.push(text(cols[i], y, guard(v, 10.5, (cols[i + 1] ?? W - X0) - cols[i] - 6, 'fig1/D'), i === 2 || i === 7 ? 'dim' : 'sm')));
  });
  const DY2 = TY + 24 * (DEMO.length + 1) + 22;
  parts.push(text(X0, DY2, guard(`下标 < ${L.dw} 读 boundary_kv，[${L.dw}, ${L.dw + L.lLocal}) 读本地 KV，≥ ${r1.selected.compressedBase} 读 gather 来的压缩 KV；kv_full_thd = cat(boundary_kv, kv_local, compressed_kv_rank_major) 共 ${L.dw + L.lLocal + L.gatheredRows} 行`, 11, W - 2 * X0, 'fig1/capD'), 'cap'));

  // ---- 底部盒子：两个并排 + 一个通栏 ----
  const OY = DY2 + 22;
  const OW = 600;
  const gap = W - 2 * X0 - 2 * OW;
  parts.push(
    infoBox(X0, OY, OW, 130, '本图算出的规则', [
      `d_comp = ${L.dc}（ratio 4 → 8，其余 ratio>1 → ratio）；d_window = max(window, d_comp) = ${L.dw}`,
      `c_cap = align((l_local + d_comp) // ratio, 32 // gcd(32, ratio)) = align(${Math.floor((L.lLocal + L.dc) / CFG.ratio)}, ${32 / gcd(32, CFG.ratio)}) = ${L.cCap}`,
      '压缩组所有权 = 组末 token 所在 rank；每个 rank 的首个可见逻辑行 =',
      `  cu_comp[seq] + ceil((rank_start − d_comp − seq_start)⁺ / ratio) → [${L.rankRow.firstLogicalRows.join(', ')}]`,
      '压缩行 RoPE 位置 = comp_id × ratio；本地行位置 = 全局行 − seq_start（越界行 → 0）',
    ], 'neutral', 'fig1/rules'),
  );
  parts.push(
    infoBox(X0 + OW + gap, OY, OW, 130, '两阶段各搬多少行（本算例）', [
      `Stage 1：rank1 收 ${L.stage1RowsRecv[1]} 行 hidden；rank0 收 ${L.stage1RowsRecv[0]} 行（首 rank 不收，只发）`,
      `Stage 2：两个 AllGather 各 ${L.gatheredRows} 行缓冲（每 rank 贡献 c_cap=${L.cCap} 行）`,
      `  其中真实压缩行 ${L.realCompressedRows}、重算槽 ${L.dupRows}、容量补齐 ${L.padRows}`,
      { text: `朴素全量 KV AllGather：${L.naiveRows} 行。玩具规模看不出省，宽度与规模的账见图 3`, cls: 'costtx' },
      `每条本地 query 只读 ≤ ${CFG.windowSize} 行 window（可含边界行）+ ≤ ${CFG.topk} 行被选中的压缩行`,
    ], 'acc1', 'fig1/rows'),
  );
  parts.push(
    infoBox(X0, OY + 142, W - 2 * X0, 82, '本图复刻了什么、简化了什么', [
      '精确：exchange_cp_boundary_hidden 的 d_window；compact 内核的组枚举；prepare_cp_compressor_input 的 c_cap 与 seq_to_rank_row；build_attention_indices 三种 index_mode',
      { text: '简化：top-k 用「最近 k 个可见块」代替 indexer 打分，只为把索引映射画出来；不画 RoPE 数值', cls: 'costtx' },
      '不画：MTP、TP、padding mask（q_padding_mask 见正文 §2.7）；Stage 2 的重叠顺序见图 2',
    ], 'neutral', 'fig1/scope'),
  );
  const H = OY + 142 + 82 + 20;
  return seal(parts, W, H, 'fig1-layout');
}

// ============================================================================
// 图 2：时序
// ============================================================================

function timelinePanel(parts, { x0, y0, w, title, sub, steps, where }) {
  parts.push(text(x0, y0, title, 'pt'));
  parts.push(text(x0, y0 + 16, sub, 'sm'));
  const LH = 26;
  const asyncs = steps.filter((s) => s.kind === 'async');
  const commH = asyncs.length > 1 ? 16 * asyncs.length + 4 : LH;
  const laneY = { compute: y0 + 30, comm: y0 + 30 + LH + 10 };
  parts.push(text(x0 + 56, laneY.compute + 17, '计算流', 'rank', 'end'));
  parts.push(text(x0 + 56, laneY.comm + commH / 2 + 4, '通信流', 'rank', 'end'));
  parts.push(line(x0 + 62, laneY.compute + LH / 2, x0 + w, laneY.compute + LH / 2, 'gl'));
  parts.push(line(x0 + 62, laneY.comm + commH / 2, x0 + w, laneY.comm + commH / 2, 'gl'));
  // 每个非异步步骤占一列；异步集合通信块横跨「发起后」到「wait 前」的列
  const slots = [];
  steps.forEach((s, i) => {
    if (s.kind !== 'async') slots.push(i);
  });
  const colW = (w - 70) / slots.length;
  const colX = (i) => x0 + 66 + slots.indexOf(i) * colW;
  const numberOf = (i) => slots.indexOf(i) + 1;
  const drawn = [];
  steps.forEach((s, i) => {
    if (s.kind === 'async') return;
    const x = colX(i);
    const y = s.lane === 'compute' ? laneY.compute : laneY.comm + (commH - LH) / 2;
    const cls = s.kind === 'block' || s.kind === 'wait' ? 'acc2' : 'neutral';
    parts.push(rect(x + 2, y, colW - 4, LH, cls, 4));
    parts.push(text(x + 10, y + 17, String(numberOf(i)), s.kind === 'work' ? 'sm' : 'costtx'));
    drawn.push({ i, label: s.label, kind: s.kind });
  });
  const asyncNotes = [];
  asyncs.forEach((s, k) => {
    const from = colX(s.launchAfter) + colW;
    const to = colX(s.waitBefore);
    if (!(to > from)) throw new Error(`${where}: 异步块 ${s.bar} 的 wait 不在发起之后`);
    const bh = asyncs.length > 1 ? 14 : LH;
    const y = laneY.comm + (asyncs.length > 1 ? 2 + k * 16 : 0);
    parts.push(rect(from + 2, y, to - from - 4, bh, 'acc1', 3));
    parts.push(text(from + 8, y + bh - 3.5, guard(`${s.bar}·在飞`, 10.5, to - from - 12, `${where}/async`), 'dim'));
    asyncNotes.push(`${s.bar}：${s.label}，发起于 ${numberOf(s.launchAfter)} 之后、wait 于 ${numberOf(s.waitBefore)} 之前`);
  });
  const notes = drawn.map((d, k) => `${k + 1} ${d.label}`);
  const ny = laneY.comm + commH + 18;
  const cols = 3;
  const colWidth = w / cols;
  notes.forEach((n, k) => {
    const cx = x0 + (k % cols) * colWidth;
    const cy = ny + Math.floor(k / cols) * 14;
    parts.push(text(cx, cy, guard(n, 10.5, colWidth - 8, `${where}/note`), 'sm'));
  });
  let y = ny + Math.ceil(notes.length / cols) * 14;
  asyncNotes.forEach((n) => {
    parts.push(text(x0, y, guard(n, 10.5, w, `${where}/asyncnote`), 'dim'));
    y += 14;
  });
  return y + 4;
}

function renderTiming() {
  const W = 1272;
  const X0 = 28;
  const PW = W - 2 * X0;
  const parts = header(
    W,
    '图 2　Stage 2 的重叠：前向两个异步 AllGather 藏在本地投影后面，反向两个 reduce-scatter 只在消费分支等待',
    '块按源码语句序等宽排列，不表示时长；蓝 = 在飞的异步集合通信，橙 = 阻塞点（同步 P2P / 同步 gather / wait）',
  );
  let y = 84;
  y = timelinePanel(parts, {
    x0: X0, y0: y, w: PW,
    title: '① 前向·有 indexer 的层（ratio 4）：_forward_thd_cp',
    sub: '每个 rank 都先发 Indexer-K 再发压缩 KV，所以集合通信入队顺序全组一致；两次 wait 各自紧贴第一个消费者',
    steps: FORWARD_INDEXER, where: 'fig2/fwd',
  });
  y = timelinePanel(parts, {
    x0: X0, y0: y + 14, w: PW,
    title: '② 前向·无 indexer 的层（ratio 128）：同一入口，但 gather 是同步的',
    sub: '没有可与之重叠的本地投影，压缩 KV 走 gather_from_sequence_parallel_region，返回前阻塞',
    steps: FORWARD_NO_INDEXER, where: 'fig2/fwd-noidx',
  });
  const bwdSteps = LAYER_BACKWARD.events.map((e) => {
    if (e.kind === 'async') {
      // 两个 RS 都在稀疏注意力反向之后发起；wait 点 = 各自的延迟 wait 块
      const waitLabel = e.bar === 'RS-KV' ? 'wait RS-KV' : 'wait RS-K';
      const launchIdx = LAYER_BACKWARD.events.findIndex((x) => x.label === '稀疏注意力反向');
      const waitIdx = LAYER_BACKWARD.events.findIndex((x) => x.label === waitLabel);
      return { ...e, launchAfter: launchIdx, waitBefore: waitIdx };
    }
    return e;
  });
  y = timelinePanel(parts, {
    x0: X0, y0: y + 14, w: PW,
    title: '③ 反向·融合 indexer-loss 路径：FusedCSAIndexerSparseAttnFromTopkFunc.backward + 两个延迟 wait',
    sub: '两个 RS 都在稀疏注意力反向之后发起，KV 先于 K；wait 只在各自 compressor 的反向分支上，由 autograd 序号从大到小的调度自然错开',
    steps: bwdSteps, where: 'fig2/bwd',
  });

  const BW = 600;
  const gap = PW - 2 * BW;
  const BY = y + 16;
  parts.push(
    infoBox(X0, BY, BW, 182, '顺序不变量（脚本算出）', [
      { text: '前向：AG-K 发起 → AG-KV 发起 → 本地 Indexer Q / weights 投影 → wait K → top-k → wait KV', cls: 'dim' },
      `反向 FromTopkFunc 内：${FUSED_BACKWARD_INTERNAL.slice(0, 3).join(' → ')}`,
      '反向 autograd（就绪节点里序号大者先执行）：',
      `  ${BRANCH_ORDER.slice(0, 3).join(' → ')}`,
      `  → ${BRANCH_ORDER.slice(3).join(' → ')}`,
      '每个 wait 只被消费它的分支触发；unfused / 无 loss 的分支不走这条路，退回同步 RS',
      '  （_GatherFromSequenceParallelRegionAsync.backward 保留同步 gather 的 RS 语义）',
    ], 'neutral', 'fig2/inv'),
  );
  parts.push(
    infoBox(X0 + BW + gap, BY, BW, 182, '锁定这些顺序的测试（冻结基线）', [
      'test_csa_fused_sparse_attention.py::TestCPCommunicationOverlap::',
      '  test_cp_backward_launches_collectives_in_dependency_order（发起顺序、梯度槽位）',
      '  test_deferred_reduce_scatter_waits_follow_consumer_branch_order（wait 顺序）',
      '  test_deferred_reduce_scatter_wait_requires_published_handle（未发起即消费 → RuntimeError）',
      '  test_deferred_reduce_scatter_wait_is_branch_local（wait 后 handle 置空）',
      'tests/unit_tests/tensor_parallel/test_mappings.py：async gather / reduce-scatter 与同步版等价',
      { text: '本图不写时长：真实重叠比例取决于 NCCL 带宽与本地投影的 GEMM 规模', cls: 'costtx' },
    ], 'neutral', 'fig2/tests'),
  );
  const H = BY + 182 + 20;
  return seal(parts, W, H, 'fig2-timing');
}

// ============================================================================
// 图 3：代价
// ============================================================================

function renderCost() {
  const W = 1272;
  const X0 = 28;
  const c = COST16;
  const r4 = c.perRatio[4];
  const r128 = c.perRatio[128];
  const parts = header(
    W,
    '图 3　真实配方 THD64K / CP16 下每层每 rank 的通信量：压缩态 gather 对比朴素全量 KV AllGather',
    `mxfp8_THD64K_128GPU_TP1PP2EP64CP16.yaml：seq ${RECIPE.seqLength}，CP ${RECIPE.cpSize} → l_local ${c.lLocal}；hidden ${RECIPE.hiddenSize}，v_head_dim ${RECIPE.vHeadDim}，indexer head_dim ${RECIPE.indexerHeadDim}，window ${RECIPE.windowSize}，bf16`,
  );

  // ---- 左：柱状 ----
  const LX = X0;
  const LY = 92;
  const LW = 640;
  parts.push(text(LX, LY, '每 rank 接收字节（MiB，log 刻度）：ratio-4 层与 ratio-128 层', 'pt'));
  const groups = [
    { name: `ratio-4 层（c_cap=${r4.cCap}，gather ${r4.gatheredRows} 行）`, bars: [
      ['边界 P2P', c.boundaryBytes, 'bar1'],
      ['Indexer-K AG', r4.kRecv, 'bar1'],
      ['压缩 KV AG', r4.kvRecv, 'bar1'],
      ['朴素 KV AG', c.naiveRecv, 'bar'],
    ] },
    { name: `ratio-128 层（c_cap=${r128.cCap}，gather ${r128.gatheredRows} 行）`, bars: [
      ['边界 P2P', c.boundaryBytes, 'bar1'],
      ['Indexer-K AG', 0, 'bar1'],
      ['压缩 KV AG', r128.kvRecv, 'bar1'],
      ['朴素 KV AG', c.naiveRecv, 'bar'],
    ] },
  ];
  const AXY = LY + 200;
  const AXH = 150;
  const maxVal = c.naiveRecv;
  const minVal = 0.1 * MiB;
  const scaleY = (v) => {
    if (v <= 0) return 0;
    const lv = Math.log10(Math.max(v, minVal) / minVal);
    const lmax = Math.log10(maxVal / minVal);
    return (lv / lmax) * AXH;
  };
  parts.push(line(LX + 40, AXY, LX + LW, AXY, 'sep'));
  for (const tick of [0.1, 1, 10, 60]) {
    const yy = AXY - scaleY(tick * MiB);
    parts.push(line(LX + 40, yy, LX + LW, yy, 'gl'));
    parts.push(text(LX + 36, yy + 4, `${tick} MiB`, 'sm', 'end'));
  }
  const gw = (LW - 60) / groups.length;
  groups.forEach((g, gi) => {
    const gx = LX + 50 + gi * gw;
    const bw = (gw - 40) / g.bars.length;
    g.bars.forEach(([name, v, cls], bi) => {
      const x = gx + bi * bw;
      const h = scaleY(v);
      if (h > 0) parts.push(`<rect class="${cls}" x="${(x + 4).toFixed(1)}" y="${(AXY - h).toFixed(1)}" width="${(bw - 8).toFixed(1)}" height="${h.toFixed(1)}"/>`);
      parts.push(text(x + bw / 2, AXY - h - 6, v > 0 ? fmtMiB(v) : '无', cls === 'bar' ? 'costtx' : 'dim', 'middle'));
      parts.push(text(x + bw / 2, AXY + 14, name, 'sm', 'middle'));
    });
    parts.push(text(gx + (gw - 40) / 2, AXY + 30, g.name, 'rank', 'middle'));
  });
  parts.push(
    infoBox(LX, AXY + 44, LW, 116, '怎么算的', [
      `边界 P2P = d_window × hidden × 2 B = ${c.dw} × ${RECIPE.hiddenSize} × 2 = ${fmtMiB(c.boundaryBytes)} MiB（内部 rank 收 1 发 1，反向再各 1）`,
      `AG 接收 = (CP − 1) × c_cap × 行宽 × 2 B；KV 行宽 ${RECIPE.vHeadDim}、Indexer-K 行宽 ${RECIPE.indexerHeadDim}；缓冲 CP × c_cap 行`,
      `ratio-4：真实压缩行 ${r4.realRows}，缓冲 ${r4.gatheredRows} 行，容量补齐 ${r4.padRows} 行（${((r4.padRows / r4.gatheredRows) * 100).toFixed(1)}%）`,
      { text: `朴素 = (CP − 1) × l_local × ${RECIPE.vHeadDim} × 2 B = ${fmtMiB(c.naiveRecv)} MiB；ratio-4 层压缩态 K+KV 合计 ${fmtMiB(r4.kRecv + r4.kvRecv)} MiB`, cls: 'costtx' },
    ], 'neutral', 'fig3/how'),
  );

  // ---- 右：随 CP 变化 ----
  const RX = X0 + LW + 40;
  const RW = W - RX - X0;
  parts.push(text(RX, LY, '同一条 64K 序列，CP 从 2 扫到 128（ratio-4 层）', 'pt'));
  const PX = RX + 50;
  const PY = LY + 30;
  const PH = 170;
  const PWW = RW - 60;
  const n = SWEEP.length;
  const xAt = (i) => PX + (i / (n - 1)) * PWW;
  const maxRecv = Math.max(...SWEEP.map((s) => s.recvBytes));
  const maxRows = Math.max(...SWEEP.map((s) => s.lLocal));
  parts.push(line(PX, PY + PH, PX + PWW, PY + PH, 'sep'));
  parts.push(line(PX, PY, PX, PY + PH, 'gl'));
  const recvPts = SWEEP.map((s, i) => [xAt(i), PY + PH - (s.recvBytes / maxRecv) * PH]);
  const rowPts = SWEEP.map((s, i) => [xAt(i), PY + PH - (s.lLocal / maxRows) * PH]);
  parts.push(polyline(recvPts, 'line1'));
  parts.push(polyline(rowPts, 'line2'));
  SWEEP.forEach((s, i) => {
    parts.push(text(xAt(i), PY + PH + 14, `CP ${s.cp}`, 'sm', 'middle'));
    parts.push(text(xAt(i), recvPts[i][1] - 6, `${fmtMiB(s.recvBytes)}`, 'dim', 'middle'));
    parts.push(text(xAt(i), rowPts[i][1] + (rowPts[i][1] > PY + PH * 0.55 || i === 0 ? -6 : 14), `${s.lLocal}`, 'costtx', 'middle'));
  });
  parts.push(text(PX - 4, PY + 8, '接收 MiB', 'dim', 'end'));
  parts.push(text(PX - 4, PY + 24, '本地行', 'costtx', 'end'));
  parts.push(
    infoBox(RX, PY + PH + 30, RW, 116, '这笔账随 CP 怎么变（脚本算出）', [
      { text: `gather 接收量随 (CP−1)/CP 饱和：CP2 ${fmtMiB(SWEEP[0].recvBytes)} → CP16 ${fmtMiB(SWEEP[3].recvBytes)} → CP128 ${fmtMiB(SWEEP[6].recvBytes)} MiB`, cls: 'dim' },
      { text: `本地行数 65536/CP：CP2 ${SWEEP[0].lLocal} → CP16 ${SWEEP[3].lLocal} → CP128 ${SWEEP[6].lLocal}，可重叠的本地投影随之收窄`, cls: 'costtx' },
      `容量补齐随 CP 增长：CP2 ${SWEEP[0].padRows} 行 → CP128 ${SWEEP[6].padRows} 行（每 rank 对齐到 8）`,
      `边界 P2P 固定 ${fmtMiB(c.boundaryBytes)} MiB / rank；local_rows ≥ d_window 给出 CP ≤ ${CP_LIMIT} 的硬上限`,
    ], 'neutral', 'fig3/sweep'),
  );
  const H = AXY + 44 + 116 + 24;
  return seal(parts, W, H, 'fig3-cost');
}

// ============================================================================

const outputs = new Map([
  ['megatron_dsv4_cp_layout.svg', renderLayout()],
  ['megatron_dsv4_cp_timing.svg', renderTiming()],
  ['megatron_dsv4_cp_cost.svg', renderCost()],
]);

export {
  CFG, RECIPE, LAYOUT, DEMO, COST16, SWEEP, CP_LIMIT, MiB, fmtMiB,
  FORWARD_INDEXER, FORWARD_NO_INDEXER, FUSED_BACKWARD_INTERNAL, BRANCH_ORDER, LAYER_BACKWARD,
  dComp, dWindow, compactCapacity, thdCpPositionIds, buildCpIndexerLayout, compactForward,
  seqToRankRow, buildAttentionIndices, visibleCompressed, demoTopk, autogradOrder,
  replayBranchOrderTest, recipeCost, cumsum, outputs,
};

const invokedDirectly = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (invokedDirectly) {
  const here = dirname(fileURLToPath(import.meta.url));
  const outDir =
    process.argv[2] ??
    join(here, '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks', 'megatron-lm', 'assets');
  mkdirSync(outDir, { recursive: true });
  for (const [name, svg] of outputs) {
    writeFileSync(join(outDir, name), svg, 'utf8');
    console.log(`${name}  ${svg.length} bytes`);
  }
  console.log('\n图 1：', JSON.stringify({ lLocal: LAYOUT.lLocal, dc: LAYOUT.dc, dw: LAYOUT.dw, cCap: LAYOUT.cCap, map: LAYOUT.rankRow.map, gathered: LAYOUT.gatheredRows, real: LAYOUT.realCompressedRows, dup: LAYOUT.dupRows, pad: LAYOUT.padRows, dropped: LAYOUT.droppedTail }));
  for (const d of DEMO) console.log('  query', d.q, JSON.stringify({ window: d.windowIdx, topk: d.topk, comp: d.compIdx, length: d.length }));
  console.log('图 2：', JSON.stringify(BRANCH_ORDER), JSON.stringify(LAYER_BACKWARD.events.map((e) => e.label)));
  console.log('图 3：', JSON.stringify({ boundary: fmtMiB(COST16.boundaryBytes), r4: { cCap: COST16.perRatio[4].cCap, k: fmtMiB(COST16.perRatio[4].kRecv), kv: fmtMiB(COST16.perRatio[4].kvRecv), pad: COST16.perRatio[4].padRows }, r128: { cCap: COST16.perRatio[128].cCap, kv: fmtMiB(COST16.perRatio[128].kvRecv) }, naive: fmtMiB(COST16.naiveRecv) }));
  console.log('      sweep', JSON.stringify(SWEEP.map((s) => [s.cp, fmtMiB(s.recvBytes), s.lLocal, s.padRows])));
}
