// 34_deepseek_v4_tensor_parallel_analysis.md 的三张图。
//
// 图 1：**原理图** —— CSA 一个 query 看哪些行。token 行 + 压缩行拼成 kv_full，滑窗索引、
//       压缩行的因果可见集、indexer top-k 选中集与 -1 无效位，全部由复刻自冻结基线
//       csa.py::_get_window_topk_idxs_cached / _get_compress_topk_idxs_cached /
//       Compressor._forward_sbhd（sq < ratio → None，cutoff = (sq // ratio) * ratio）/
//       CompressedSparseAttention._forward_unfused_csa（causal mask → top-k → 校验 → +offset）
//       与 dsa.py::fused_qk_topk_naive（-inf → -1）的规则算出；THD 拼接顺序由复刻的
//       csa.py::build_cu_seqlens_kv_full / cat_per_segment / get_compress_topk_idxs_thd 算出。
// 图 2：**布局图** —— 参数所有权与精度账本。按冻结源码里每个投影的构造形状
//       （deepseek_v4_hybrid_attention.py::DSv4HybridSelfAttention.__init__ /
//       DSv4HybridAttention.__init__、csa.py::Compressor.__init__ / CSAIndexer.__init__ /
//       CompressedSparseAttention.__init__）把共用算例的参数量逐个乘出来，并标出所有权
//       （duplicated / Column 接口 / Row 接口 / nn.Parameter）与驻留精度
//       （随外层 FP8 上下文 / FP8 上下文外 BF16 / mark_keep_in_fp32 / params_dtype）。
// 图 3：**分派图 + 数值图** —— CompressedSparseAttention.forward 的 fused/unfused × 训练/推理
//       × 层型分派矩阵（复刻 dsa_kernels.py::use_fused_dsa_kernels 与 forward 的分支谓词），
//       以及 mHC 的 Sinkhorn 双随机映射（复刻 hyper_connection.py::_sinkhorn_iterations）
//       在 1/2/3/5/20 次迭代后的行列和偏差。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
//
// 图 1 要回答「一个 query 的 KV 集合怎么拼出来、为什么 ratio 128 层可以不要 indexer」。
//   小算例：S=16、window=4、ratio=4、top-k=2；indexer 分数用一张确定性表（不是学到的分数）。
//   左：16 行（query i）× 20 列（kv_full：t0..t15 | c0..c3）的格子；滑窗命中格 acc1、
//   top-k 选中的压缩格 acc1 并写 k、因果可见但没选中的压缩格 neutral、未来位 ghost；
//   第 16 列前画一条 offset=sq 分隔线。格子下方写出两行 topk_idxs 的原样数组，-1 用 acc2。
//   右栏四个盒子：索引规则（复刻的四条公式与拼接顺序）、规模面板（S=4096 / 65536 时
//   W / C / C dense / H / dense 每 query 最多访问的 KV 行数，全部算出）、THD 拼接
//   （两段 [8, 6] 的 cu_seqlens_kv_full 与 compressed 局部索引偏移）、复刻了什么 / 简化了什么。
//
// 图 2 要回答「哪些参数只是接口像 TP、哪些参数从构造起就被复制、哪些参数留在 FP8 之外」。
//   共用算例：tests/functional_tests/test_cases/gpt/gpt3_mcore_te_tp1_pp2_dsv4_hybrid_fused
//   的 model_config.yaml（hidden 512、8 heads、q_lora 192、v_head_dim 16、qk_pos_emb 8、
//   o_groups 8、o_lora_rank 1024（默认）、indexer 64 × 128、top-k 512、window 128）。
//   左：ratio-4（C）层的每个参数张量一行：模块路径、形状、参数量、所有权、驻留精度；
//   所有权与精度用两种芯片：acc1 = 「Column/Row 接口，但 TP group 只有一个 rank」，
//   acc2 = 「精度例外：FP8 上下文外 BF16 或 mark_keep_in_fp32」。右栏：TP group（size 1）
//   的三条源码事实、若 TP>1 要证明的四项（分析重建）、账本合计（C/H/W 层参数量、
//   linear_proj 占比、fp32 保持与 FP8 之外的参数个数），以及 DSv4-Flash 配方 C 层的同一账本。
//
// 图 3 要回答「同一层在哪些条件下走哪条 kernel 路径、mHC 的映射几步就近似双随机」。
//   左：行 = 层型（W ratio 0 / C ratio 4 / C + csa_dense_mode / H ratio 128），
//   列 = unfused·train / unfused·eval / cudnn·train / cudnn·eval，格子写路径名与 kernel 家族；
//   路径由复刻的 use_fused_dsa_kernels 与 forward 谓词逐格算出。
//   右：n=4 的 Sinkhorn：一张确定性 logits 表经 1/2/3/5/20 次迭代后 max|行和−1| 与
//   max|列和−1|，以及 20 次后的 4×4 矩阵；mapping_proj 的形状与参数量按 n=4、hidden 512 算出。
//
// 用法：node tools/figs/svg/megatron_dsv4_tp_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// ============================================================================
// 共用算例（gpt3_mcore_te_tp1_pp2_dsv4_hybrid_fused/model_config.yaml + 基线默认值）
// ============================================================================

const CFG = Object.freeze({
  hidden: 512,
  heads: 8,
  qLoraRank: 192,
  vHeadDim: 16,
  qkPosEmbHeadDim: 8,
  oGroups: 8, // MLATransformerConfig 默认
  oLoraRank: 1024, // MLATransformerConfig 默认
  indexerHeads: 64,
  indexerHeadDim: 128,
  indexerTopk: 512,
  windowSize: 128,
  compressRatios: Object.freeze([0, 4, 128, 4, 128, 4]),
  numResidualStreams: 4, // TransformerConfig.num_residual_streams 默认
  sinkhornIterations: 20, // 配方 mhc_sinkhorn_iterations
  // 图 1 的小算例
  fig1: Object.freeze({ seqlen: 16, windowSize: 4, ratio: 4, topk: 2 }),
  // 图 1 规模面板的序列长度（配方 SL4K / THD64K）
  scaleSeqlens: Object.freeze([4096, 65536]),
  // DSv4-Flash gb200 配方（examples/moe_recipes/deepseek_v4_flash/gb200/*.yaml）
  flash: Object.freeze({
    hidden: 4096, heads: 64, qLoraRank: 1024, vHeadDim: 512, qkPosEmbHeadDim: 64,
    oGroups: 8, oLoraRank: 1024, indexerHeads: 64, indexerHeadDim: 128, indexerTopk: 512,
  }),
});

// ============================================================================
// 复刻 csa.py 的索引规则（SBHD）
// ============================================================================

// _get_window_topk_idxs_cached：matrix = clamp(i - W + 1, 0) + j；matrix > i → -1
function windowTopkIdxs(windowSize, seqlen) {
  const rows = [];
  for (let i = 0; i < seqlen; i += 1) {
    const base = Math.max(i - windowSize + 1, 0);
    const row = [];
    for (let j = 0; j < windowSize; j += 1) {
      const v = base + j;
      row.push(v > i ? -1 : v);
    }
    rows.push(row);
  }
  return rows;
}

// _get_compress_topk_idxs_cached：n_comp = seqlen // ratio；c >= (i+1) // ratio → -1，否则 c + offset
function compressTopkIdxs(ratio, seqlen, offset) {
  const nComp = Math.floor(seqlen / ratio);
  const rows = [];
  for (let i = 0; i < seqlen; i += 1) {
    const visible = Math.floor((i + 1) / ratio);
    const row = [];
    for (let c = 0; c < nComp; c += 1) row.push(c >= visible ? -1 : c + offset);
    rows.push(row);
  }
  return rows;
}

// Compressor._forward_sbhd：sq < ratio → None；否则 cutoff = (sq // ratio) * ratio，n_compressed = cutoff // ratio
function compressorRows(seqlen, ratio) {
  if (ratio <= 1) return 0;
  if (seqlen < ratio) return 0;
  return Math.floor(seqlen / ratio);
}

// 确定性的 indexer 分数表（代替学到的分数；只用于演示 top-k 的选择与校验）
function demoIndexerScores(seqlen, nComp) {
  const rows = [];
  for (let i = 0; i < seqlen; i += 1) {
    const row = [];
    for (let c = 0; c < nComp; c += 1) row.push(((i * 7 + c * 3) % 5) + c * 0.1);
    rows.push(row);
  }
  return rows;
}

// _forward_unfused_csa 的 causal_mask（c >= (i+1)//ratio → -inf）+ fused_qk_topk_naive 的
// topk（-inf → -1）+ 校验 valid = (idx >= 0) & (idx < (i+1)//ratio) → idx + offset，否则 -1
function indexerTopkWithOffset(scores, ratio, topk, offset) {
  const rows = [];
  scores.forEach((row, i) => {
    const visible = Math.floor((i + 1) / ratio);
    const masked = row.map((s, c) => (c >= visible ? -Infinity : s));
    const k = Math.min(topk, row.length);
    const order = masked
      .map((s, c) => ({ s, c }))
      .sort((a, b) => b.s - a.s || a.c - b.c)
      .slice(0, k);
    const picked = order.map((o) => (o.s === -Infinity ? -1 : o.c));
    const checked = picked.map((idx) => (idx >= 0 && idx < visible ? idx + offset : -1));
    rows.push(checked);
  });
  return rows;
}

// 一个 query 最多访问多少 KV 行（最后一个 query，i = S-1）
function rowsPerQuery(kind, seqlen, { windowSize, indexerTopk: topk }) {
  const win = Math.min(windowSize, seqlen);
  if (kind === 'dense') return seqlen;
  if (kind === 'W') return win;
  if (kind === 'C') return win + Math.min(topk, compressorRows(seqlen, 4));
  if (kind === 'Cdense') return win + compressorRows(seqlen, 4);
  if (kind === 'H') return win + compressorRows(seqlen, 128);
  throw new Error(`unknown kind ${kind}`);
}

// ---- THD ----
// build_cu_seqlens_kv_full：full_lens = kv_lens + comp_lens；cumsum 前置 0
function buildCuSeqlensKvFull(cuKv, cuComp) {
  const out = [0];
  for (let b = 0; b + 1 < cuKv.length; b += 1) {
    out.push(out[b] + (cuKv[b + 1] - cuKv[b]) + (cuComp[b + 1] - cuComp[b]));
  }
  return out;
}

// cat_per_segment 的目的下标：kv 行 → cu_full[b] + (i - cu_kv[b])；压缩行 → cu_full[b] + kv_len[b] + (j - cu_comp[b])
function catPerSegmentOrder(cuKv, cuComp) {
  const cuFull = buildCuSeqlensKvFull(cuKv, cuComp);
  const total = cuFull[cuFull.length - 1];
  const order = new Array(total).fill(null);
  for (let b = 0; b + 1 < cuKv.length; b += 1) {
    const kvLen = cuKv[b + 1] - cuKv[b];
    for (let i = cuKv[b]; i < cuKv[b + 1]; i += 1) order[cuFull[b] + (i - cuKv[b])] = `t${i}`;
    for (let j = cuComp[b]; j < cuComp[b + 1]; j += 1) order[cuFull[b] + kvLen + (j - cuComp[b])] = `c${j}`;
  }
  return { cuFull, order };
}

// get_compress_topk_idxs_thd：局部索引 = col + seqlen_kv[b]（col < min((pos+1)//ratio, comp_len[b])）
function compressTopkIdxsThdRow(pos, b, ratio, cuKv, cuComp, maxN) {
  const kvLen = cuKv[b + 1] - cuKv[b];
  const compLen = cuComp[b + 1] - cuComp[b];
  const nValid = Math.min(Math.floor((pos + 1) / ratio), compLen);
  const row = [];
  for (let col = 0; col < maxN; col += 1) row.push(col < nValid ? col + kvLen : -1);
  return row;
}

// ============================================================================
// 图 1 的算例
// ============================================================================

const FIG1 = (() => {
  const { seqlen, windowSize, ratio, topk } = CFG.fig1;
  const nComp = compressorRows(seqlen, ratio);
  const offset = seqlen;
  const win = windowTopkIdxs(windowSize, seqlen);
  const allComp = compressTopkIdxs(ratio, seqlen, offset);
  const scores = demoIndexerScores(seqlen, nComp);
  const sel = indexerTopkWithOffset(scores, ratio, topk, offset);
  const topkIdxs = win.map((w, i) => [...w, ...sel[i]]);
  const kvFullRows = seqlen + nComp;
  // 每行统计
  const perRow = topkIdxs.map((row, i) => ({
    i,
    valid: row.filter((v) => v >= 0).length,
    invalid: row.filter((v) => v < 0).length,
    visibleComp: Math.floor((i + 1) / ratio),
    selected: sel[i].filter((v) => v >= 0).length,
  }));
  const invalidTotal = perRow.reduce((s, r) => s + r.invalid, 0);
  const firstFullWindow = perRow.findIndex((r) => r.i >= windowSize - 1);
  const firstCompVisible = perRow.findIndex((r) => r.visibleComp > 0);
  const firstTopkSaturated = perRow.findIndex((r) => r.visibleComp > topk);
  // 规模面板
  const scale = CFG.scaleSeqlens.map((S) => ({
    S,
    dense: rowsPerQuery('dense', S, CFG),
    W: rowsPerQuery('W', S, CFG),
    C: rowsPerQuery('C', S, CFG),
    Cdense: rowsPerQuery('Cdense', S, CFG),
    H: rowsPerQuery('H', S, CFG),
    comp4: compressorRows(S, 4),
    comp128: compressorRows(S, 128),
  }));
  // THD 拼接示例：两段 [8, 6]
  const segLens = [8, 6];
  const cuKv = [0];
  const cuComp = [0];
  segLens.forEach((l, b) => {
    cuKv.push(cuKv[b] + l);
    cuComp.push(cuComp[b] + compressorRows(l, ratio));
  });
  const thd = catPerSegmentOrder(cuKv, cuComp);
  const maxN = Math.max(...segLens.map((l) => compressorRows(l, ratio)));
  const thdRows = [
    { b: 0, pos: 7, row: compressTopkIdxsThdRow(7, 0, ratio, cuKv, cuComp, maxN) },
    { b: 1, pos: 5, row: compressTopkIdxsThdRow(5, 1, ratio, cuKv, cuComp, maxN) },
  ];
  return {
    seqlen, windowSize, ratio, topk, nComp, offset, win, allComp, scores, sel, topkIdxs, kvFullRows,
    perRow, invalidTotal, firstFullWindow, firstCompVisible, firstTopkSaturated, scale,
    thd: { segLens, cuKv, cuComp, cuFull: thd.cuFull, order: thd.order, rows: thdRows, maxN },
  };
})();

if (FIG1.firstTopkSaturated < 0) {
  throw new Error('图 1 的小算例里 top-k 从未饱和：可见压缩行数应在某个 query 超过 top-k');
}

// ============================================================================
// 图 2：参数账本（形状来自冻结源码的构造调用）
// ============================================================================

function ledgerFor(cfg, ratio, dense = false) {
  const qHeadDim = cfg.vHeadDim; // DSv4HybridAttention.__init__：q_head_dim = v_head_dim
  const queryProjectionSize = cfg.vHeadDim * cfg.heads;
  if (queryProjectionSize % cfg.oGroups !== 0) throw new Error('assert query_projection_size % o_groups == 0');
  const groupIn = queryProjectionSize / cfg.oGroups;
  const groupOut = cfg.oGroups * cfg.oLoraRank;
  const rows = [];
  const add = (path, shape, own, dtype, note = '') => {
    const n = shape.reduce((a, b) => a * b, 1);
    rows.push({ path, shape, n, own, dtype, note });
  };
  add('linear_q_down_proj', [cfg.qLoraRank, cfg.hidden], 'duplicated', 'fp8ctx', 'TELinear，tp_group=None');
  add('q_layernorm', [cfg.qLoraRank], 'param', 'pdtype');
  add('linear_q_up_proj', [cfg.heads * qHeadDim, cfg.qLoraRank], 'column', 'fp8ctx', 'gather_output=False');
  add('linear_kv_proj', [cfg.vHeadDim, cfg.hidden], 'column', 'fp8ctx', 'gather_output=False');
  add('kv_layernorm', [cfg.vHeadDim], 'param', 'pdtype');
  add('linear_o_group_proj', [groupOut, groupIn], 'param', 'pdtype', 'torch.nn.Parameter');
  add('linear_proj', [cfg.hidden, groupOut], 'row', 'fp8ctx', 'input_is_parallel=True');
  add('core.attn_sink', [cfg.heads], 'param', 'fp32', 'mark_keep_in_fp32');
  if (ratio > 1) {
    const coff = ratio === 4 ? 2 : 1;
    const hd = cfg.vHeadDim;
    add('core.compressor.linear_wkv', [coff * hd, cfg.hidden], 'duplicated', 'bf16out');
    add('core.compressor.linear_wgate', [coff * hd, cfg.hidden], 'duplicated', 'bf16out');
    add('core.compressor.ape', [ratio, coff * hd], 'param', 'fp32', 'mark_keep_in_fp32');
    add('core.compressor.norm', [hd], 'param', 'pdtype');
  }
  if (ratio === 4 && !dense) {
    const ihd = cfg.indexerHeadDim;
    const coff = 2;
    add('core.indexer.linear_wq_b', [cfg.indexerHeads * ihd, cfg.qLoraRank], 'duplicated', 'fp8ctx');
    add('core.indexer.linear_weights_proj', [cfg.indexerHeads, cfg.hidden], 'duplicated', 'bf16out');
    add('core.indexer.compressor.linear_wkv', [coff * ihd, cfg.hidden], 'duplicated', 'bf16out');
    add('core.indexer.compressor.linear_wgate', [coff * ihd, cfg.hidden], 'duplicated', 'bf16out');
    add('core.indexer.compressor.ape', [ratio, coff * ihd], 'param', 'fp32', 'mark_keep_in_fp32');
    add('core.indexer.compressor.norm', [ihd], 'param', 'pdtype');
  }
  const total = rows.reduce((s, r) => s + r.n, 0);
  const by = (pred) => rows.filter(pred).reduce((s, r) => s + r.n, 0);
  return {
    rows,
    total,
    proj: rows.find((r) => r.path === 'linear_proj').n,
    fp32Params: by((r) => r.dtype === 'fp32'),
    fp32Tensors: rows.filter((r) => r.dtype === 'fp32').length,
    bf16OutParams: by((r) => r.dtype === 'bf16out'),
    bf16OutTensors: rows.filter((r) => r.dtype === 'bf16out').length,
    duplicated: by((r) => r.own === 'duplicated'),
    interface: by((r) => r.own === 'column' || r.own === 'row'),
    duplicatedTensors: rows.filter((r) => r.own === 'duplicated').length,
    interfaceTensors: rows.filter((r) => r.own === 'column' || r.own === 'row').length,
  };
}

const FIG2 = Object.freeze({
  C: ledgerFor(CFG, 4),
  Cdense: ledgerFor(CFG, 4, true),
  H: ledgerFor(CFG, 128),
  W: ledgerFor(CFG, 0),
  flashC: ledgerFor(CFG.flash, 4),
  flashH: ledgerFor(CFG.flash, 128),
  flashW: ledgerFor(CFG.flash, 0),
  headsPerPartition: CFG.heads / 1,
  derivedQkHeadDim: CFG.vHeadDim - CFG.qkPosEmbHeadDim, // __post_init__：qk_head_dim = kv_lora_rank = v_head_dim - qk_pos_emb_head_dim
});

const pct = (a, b) => Math.round((a / b) * 100);
const fmt = (n) => n.toLocaleString('en-US');

// mHC mapping_proj：nn.Linear(n*C → n^2 + 2n, bias=False)，alpha ×3（各 1 个），bias n^2 + 2n
function mhcLedger(hidden, n) {
  const out = n * n + 2 * n;
  return { n, out, mappingProj: [out, n * hidden], mappingParams: out * n * hidden, alpha: 3, bias: out };
}
const MHC = Object.freeze(mhcLedger(CFG.hidden, CFG.numResidualStreams));

// ============================================================================
// 图 3：分派矩阵（复刻 dsa_kernels.py::use_fused_dsa_kernels 与 forward 的谓词）与 Sinkhorn
// ============================================================================

function useFusedDsaKernels(attentionBackend, dsaKernelBackend) {
  if (attentionBackend === 'unfused') return false;
  if (dsaKernelBackend !== 'none' && !['tilelang', 'cudnn'].includes(dsaKernelBackend)) {
    throw new Error('dsa_kernel_backend must be one of: none, tilelang, cudnn');
  }
  return dsaKernelBackend !== 'none';
}

// CompressedSparseAttention.__init__ 的条件构造 + forward 的分派
function dispatch({ ratio, dense, attentionBackend, dsaKernelBackend, training, seqlen }) {
  const compressorBuilt = ratio > 1;
  const indexerBuilt = ratio === 4 && !dense;
  const nCompressed = compressorBuilt ? compressorRows(seqlen, ratio) : 0;
  const useFused = useFusedDsaKernels(attentionBackend, dsaKernelBackend);
  const hasIndexerCompressed = ratio > 1 && nCompressed > 0 && indexerBuilt;
  if (!useFused) {
    if (hasIndexerCompressed && training) return { path: 'unfused·B', kernels: 'DSAIndexerLoss + torch', loss: true };
    if (hasIndexerCompressed) return { path: 'unfused·C', kernels: 'qk_topk_naive + torch', loss: false };
    if (nCompressed > 0) return { path: 'unfused·A', kernels: 'all compressed + torch', loss: false };
    return { path: 'unfused·A', kernels: 'window + torch', loss: false };
  }
  if (hasIndexerCompressed && training) return { path: 'B', kernels: 'fused indexer + attn', loss: true };
  if (hasIndexerCompressed) return { path: 'C', kernels: 'topk → csa_sparse_attn', loss: false };
  if (nCompressed > 0) return { path: 'A', kernels: 'all → csa_sparse_attn', loss: false };
  return { path: 'A', kernels: 'window → csa_sparse_attn', loss: false };
}

const FIG3_LAYERS = Object.freeze([
  { key: 'W', label: 'W（ratio 0）', ratio: 0, dense: false },
  { key: 'C', label: 'C（ratio 4）', ratio: 4, dense: false },
  { key: 'Cd', label: 'C + csa_dense_mode', ratio: 4, dense: true },
  { key: 'H', label: 'H（ratio 128）', ratio: 128, dense: false },
]);
const FIG3_COLS = Object.freeze([
  { key: 'u-train', label: 'unfused · train', attentionBackend: 'fused', dsaKernelBackend: 'none', training: true },
  { key: 'u-eval', label: 'unfused · eval', attentionBackend: 'fused', dsaKernelBackend: 'none', training: false },
  { key: 'c-train', label: 'cudnn · train', attentionBackend: 'fused', dsaKernelBackend: 'cudnn', training: true },
  { key: 'c-eval', label: 'cudnn · eval', attentionBackend: 'fused', dsaKernelBackend: 'cudnn', training: false },
]);

const FIG3 = (() => {
  const seqlen = CFG.scaleSeqlens[0];
  const cells = FIG3_LAYERS.map((L) => FIG3_COLS.map((C) => dispatch({ ...L, ...C, seqlen })));
  const distinct = [...new Set(cells.flat().map((c) => c.path))];
  const lossCells = cells.flat().filter((c) => c.loss).length;
  // attention_backend=unfused 把一切 fused 关掉（功能测试配置的重复键落到 unfused）
  const unfusedOverride = dispatch({ ratio: 4, dense: false, attentionBackend: 'unfused', dsaKernelBackend: 'cudnn', training: true, seqlen });
  // HCA 在 S < 128 时退化：compressor 返回 None
  const hcaShort = dispatch({ ratio: 128, dense: false, attentionBackend: 'fused', dsaKernelBackend: 'cudnn', training: true, seqlen: 64 });
  return { seqlen, cells, distinct, lossCells, unfusedOverride, hcaShort };
})();

// 复刻 hyper_connection.py::_sinkhorn_iterations
function sinkhornIterations(logits, numIterations, eps = 1e-6) {
  const n = logits.length;
  let M = logits.map((row) => {
    const mx = Math.max(...row);
    const e = row.map((v) => Math.exp(v - mx));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s + eps);
  });
  const colNorm = (A) => {
    const cs = Array.from({ length: n }, (_, j) => A.reduce((s, r) => s + r[j], 0));
    return A.map((row) => row.map((v, j) => v / (cs[j] + eps)));
  };
  const rowNorm = (A) => A.map((row) => {
    const rs = row.reduce((a, b) => a + b, 0);
    return row.map((v) => v / (rs + eps));
  });
  M = colNorm(M);
  for (let k = 0; k < numIterations - 1; k += 1) {
    M = rowNorm(M);
    M = colNorm(M);
  }
  return M;
}

function sinkhornDeviation(M) {
  const n = M.length;
  const rowDev = Math.max(...M.map((r) => Math.abs(r.reduce((a, b) => a + b, 0) - 1)));
  const colDev = Math.max(...Array.from({ length: n }, (_, j) => Math.abs(M.reduce((s, r) => s + r[j], 0) - 1)));
  return { rowDev, colDev };
}

const SINKHORN = (() => {
  const n = CFG.numResidualStreams;
  // 确定性 logits（不是训练得到的映射）
  const logits = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => ((i * 3 + j * 5) % 7) * 0.6 - 1.2));
  const iters = [1, 2, 3, 5, CFG.sinkhornIterations];
  const trace = iters.map((k) => ({ k, ...sinkhornDeviation(sinkhornIterations(logits, k)) }));
  const final = sinkhornIterations(logits, CFG.sinkhornIterations);
  return { n, logits, iters, trace, final };
})();

if (SINKHORN.trace[SINKHORN.trace.length - 1].rowDev > 1e-4 || SINKHORN.trace[SINKHORN.trace.length - 1].colDev > 1e-4) {
  throw new Error('图 3 的立论前提被推翻：20 次 Sinkhorn 迭代后行列和应已接近 1');
}

if (process.env.DSV4_FIG_DEBUG) {
  console.log('FIG1 topk_idxs:');
  FIG1.topkIdxs.forEach((r, i) => console.log(`  i=${i} [${r.join(',')}] visible=${FIG1.perRow[i].visibleComp}`));
  console.log('FIG1 scale', JSON.stringify(FIG1.scale));
  console.log('FIG1 thd', JSON.stringify(FIG1.thd));
  for (const k of ['C', 'Cdense', 'H', 'W', 'flashC', 'flashH', 'flashW']) {
    const l = FIG2[k];
    console.log(`FIG2 ${k}: total=${l.total} proj=${l.proj} fp32=${l.fp32Params}/${l.fp32Tensors} bf16out=${l.bf16OutParams}/${l.bf16OutTensors} dup=${l.duplicated} iface=${l.interface}`);
  }
  console.log('MHC', JSON.stringify(MHC));
  FIG3.cells.forEach((row, i) => console.log(`FIG3 ${FIG3_LAYERS[i].key}: ${row.map((c) => c.path).join(' | ')}`));
  console.log('FIG3 override', JSON.stringify(FIG3.unfusedOverride), 'hcaShort', JSON.stringify(FIG3.hcaShort));
  console.log('SINKHORN', JSON.stringify(SINKHORN.trace));
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

const GUARD_LENIENT = Boolean(process.env.DSV4_FIG_LENIENT);
function guard(value, fontSize, limit, where) {
  const width = textWidth(value, fontSize);
  if (width > limit) {
    const msg = `${where}: "${value}" 需要 ${width.toFixed(1)}px，超出 ${limit}px`;
    if (GUARD_LENIENT) console.error('GUARD ' + msg);
    else throw new Error(msg);
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
  .main{fill:none;stroke:#2563EB;stroke-width:2;marker-end:url(#arrowMain)}
  .aux{fill:none;stroke:#AEB6C2;stroke-width:1.4;stroke-dasharray:5 4;marker-end:url(#arrowAux)}
  .edge{fill:none;stroke:#AEB6C2;stroke-width:1.4;stroke-dasharray:4 4}
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
  ti: 18, su: 11.5, pt: 14, tx: 12, sm: 10.5, dim: 10.5, costtx: 10.5, rank: 11, cap: 11,
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
        const msg = `${name}: 文字重叠 "${a.raw}" × "${b.raw}"（重合 ${dx.toFixed(1)}×${dy.toFixed(1)}px）`;
        if (GUARD_LENIENT) console.error('OVERLAP ' + msg);
        else throw new Error(msg);
      }
    }
  }
  return svg;
}

const seal = (parts, w, h, name) =>
  assertNoTextOverlap(parts.join('\n').replace(/__H__/g, String(h)) + '\n</svg>', name);

// ============================================================================
// 图 1：CSA 一个 query 看哪些行
// ============================================================================

function renderCsaIndex() {
  const W = 1272;
  const f = FIG1;
  const parts = header(
    W,
    '图 1　CSA 一个 query 看哪些行：滑窗行 ∪ indexer 选中的压缩行，拼在 kv_full 的两段上',
    `小算例：S=${f.seqlen}、window=${f.windowSize}、ratio=${f.ratio}、top-k=${f.topk}；kv_full = [t0..t${f.seqlen - 1} | c0..c${f.nComp - 1}]，压缩行下标 = c + offset，offset = sq = ${f.offset}；indexer 分数用确定性表`,
  );
  const X0 = 28;
  const COL = 26;
  const ROW = 20;
  const gx = X0 + 56;
  const gy = 108;
  // 列头
  for (let c = 0; c < f.kvFullRows; c += 1) {
    const label = c < f.seqlen ? `t${c}` : `c${c - f.seqlen}`;
    parts.push(text(gx + c * COL + COL / 2, gy - 6, label, c < f.seqlen ? 'sm' : 'dim', 'middle'));
  }
  parts.push(text(gx, gy - 24, 'kv_full 列 →', 'sm'));
  // 行
  for (let i = 0; i < f.seqlen; i += 1) {
    const y = gy + i * ROW;
    parts.push(text(gx - 8, y + 14, `q${i}`, 'rank', 'end'));
    const winSet = new Set(f.win[i].filter((v) => v >= 0));
    const selSet = new Set(f.sel[i].filter((v) => v >= 0));
    for (let c = 0; c < f.kvFullRows; c += 1) {
      const x = gx + c * COL;
      let cls = 'ghost';
      let label = '';
      let lcls = 'sm';
      if (c < f.seqlen) {
        if (winSet.has(c)) {
          cls = 'acc1';
          label = 'w';
          lcls = 'dim';
        } else if (c <= i) {
          cls = 'ghost';
        }
      } else {
        const compIdx = c - f.seqlen;
        const visible = compIdx < f.perRow[i].visibleComp;
        if (selSet.has(c)) {
          cls = 'acc1';
          label = 'k';
          lcls = 'dim';
        } else if (visible) {
          cls = 'neutral';
          label = '·';
        }
      }
      parts.push(rect(x + 1, y + 1, COL - 2, ROW - 2, cls, 3));
      if (label) parts.push(text(x + COL / 2, y + 14, label, lcls, 'middle'));
    }
  }
  // offset 分隔线
  const sepX = gx + f.seqlen * COL;
  parts.push(line(sepX, gy - 30, sepX, gy + f.seqlen * ROW + 4, 'sep'));
  parts.push(text(sepX + 4, gy - 24, `offset = sq = ${f.offset}`, 'costtx'));
  // 图例
  const ly = gy + f.seqlen * ROW + 16;
  parts.push(rect(gx, ly - 10, 14, 12, 'acc1', 2));
  parts.push(text(gx + 18, ly, 'w 滑窗行　k top-k 选中的压缩行', 'sm'));
  parts.push(rect(gx + 226, ly - 10, 14, 12, 'neutral', 2));
  parts.push(text(gx + 244, ly, '· 因果可见但未选中', 'sm'));
  parts.push(rect(gx + 378, ly - 10, 14, 12, 'ghost', 2));
  parts.push(text(gx + 396, ly, '未来位 / 不在集合里', 'sm'));
  // 两行 topk_idxs 原样数组
  const ay = ly + 30;
  const showRow = (i, yy) => {
    parts.push(text(X0, yy, `topk_idxs[q${i}] =`, 'tx'));
    let x = X0 + 118;
    parts.push(text(x, yy, '[', 'tx'));
    x += 10;
    f.topkIdxs[i].forEach((v, k) => {
      const s = String(v);
      const wdt = textWidth(s, 10.5) + 10;
      parts.push(rect(x, yy - 12, wdt, 16, v < 0 ? 'acc2' : k < f.windowSize ? 'acc1' : 'acc1', 3));
      parts.push(text(x + wdt / 2, yy, s, v < 0 ? 'costtx' : 'dim', 'middle'));
      x += wdt + 4;
      if (k === f.windowSize - 1) {
        parts.push(text(x + 2, yy, '|', 'tx'));
        x += 12;
      }
    });
    parts.push(text(x, yy, ']', 'tx'));
    parts.push(text(x + 14, yy, `可见压缩行 ${f.perRow[i].visibleComp}，有效 ${f.perRow[i].valid}，−1 共 ${f.perRow[i].invalid}`, 'sm'));
  };
  showRow(1, ay);
  showRow(5, ay + 22);
  showRow(f.seqlen - 1, ay + 44);
  parts.push(text(X0, ay + 70, guard(`−1 在 kernel 里被 mask 成 −inf；全图 ${f.seqlen} 个 query 共 ${f.invalidTotal} 个 −1 槽位`, 11, 600, 'fig1/cap0'), 'cap'));
  parts.push(text(X0, ay + 86, guard(`q${f.firstFullWindow} 起滑窗填满，q${f.firstCompVisible} 起看得见压缩行，q${f.firstTopkSaturated} 起可见压缩行 > top-k、开始真正做选择`, 11, 600, 'fig1/cap1'), 'cap'));

  // 右栏
  const RX = 656;
  const RW = 588;
  parts.push(
    infoBox(
      RX,
      96,
      RW,
      150,
      '索引规则（复刻 csa.py 的 SBHD 助手）',
      [
        `window：clamp(i − W + 1, 0) + j，超过 i 的位置写 −1（_get_window_topk_idxs_cached）`,
        `compressed：c + offset，仅当 c < (i+1)//ratio；n_comp = S//ratio = ${f.nComp}（S < ratio → None）`,
        `indexer：causal mask → top-k（−inf → −1）→ 校验 (idx ≥ 0) ∧ (idx < (i+1)//ratio) → +offset`,
        { text: `拼接顺序固定 [window | compressed]；fused 路径再 local_to_global_flat 成扁平全局 id`, cls: 'dim' },
        `ratio 128 与 csa_dense_mode 不建 indexer：压缩行全部进集合（get_compress_topk_idxs）`,
        `top-k 只在「可见压缩行 > top-k」时才真的丢行：本例从 q${f.firstTopkSaturated} 起`,
      ],
      'neutral',
      'fig1/rules',
    ),
  );
  const s0 = f.scale[0];
  const s1 = f.scale[1];
  parts.push(
    infoBox(
      RX,
      262,
      RW,
      166,
      `规模面板：最后一个 query 最多访问的 KV 行数（window=${CFG.windowSize}，top-k=${CFG.indexerTopk}）`,
      [
        `S=${fmt(s0.S)}：dense ${fmt(s0.dense)}｜W ${s0.W}｜C ${s0.C}（压缩行 ${fmt(s0.comp4)} > top-k）｜C dense ${fmt(s0.Cdense)}｜H ${s0.H}`,
        `S=${fmt(s1.S)}：dense ${fmt(s1.dense)}｜W ${s1.W}｜C ${s1.C}（压缩行 ${fmt(s1.comp4)}）｜C dense ${fmt(s1.Cdense)}｜H ${s1.H}`,
        { text: `C 层被 window + top-k 钉死在 ${s1.C} 行、与 S 无关；dense 是 ${fmt(s1.dense)} 行`, cls: 'dim' },
        `H 层不用 indexer：S/128 行本来就少（${s1.S / 128} 行 @ ${fmt(s1.S)}），全部看完也比 top-k 便宜`,
        { text: `C dense 让访问量随 S/4 增长（${fmt(s1.Cdense)} 行 @ ${fmt(s1.S)}）：csa_dense_mode 的代价`, cls: 'costtx' },
        `S < ratio 的段没有压缩行：Compressor 返回 None，H 层在 S < 128 时退化成纯滑窗`,
        `行数与压缩 KV 的内容都由 Compressor 决定；本图不画门控池化本身`,
      ],
      'acc1',
      'fig1/scale',
    ),
  );
  const t = f.thd;
  parts.push(
    infoBox(
      RX,
      444,
      RW,
      134,
      'THD：按段拼接（build_cu_seqlens_kv_full / cat_per_segment）',
      [
        `两段 [${t.segLens.join(', ')}]、ratio=${f.ratio} → 压缩行 [${t.segLens.map((l) => compressorRows(l, f.ratio)).join(', ')}]；cu_kv=[${t.cuKv.join(', ')}]，cu_comp=[${t.cuComp.join(', ')}]`,
        { text: `cu_kv_full=[${t.cuFull.join(', ')}]：行序 ${t.order.slice(0, t.cuFull[1]).join(' ')} | ${t.order.slice(t.cuFull[1]).join(' ')}`, cls: 'dim' },
        `压缩行的局部下标偏移 = 该段 seqlen_kv：段 0 偏移 ${t.segLens[0]}、段 1 偏移 ${t.segLens[1]}（SBHD 的 offset = sq）`,
        `段 0 末 query（pos ${t.rows[0].pos}）压缩下标 [${t.rows[0].row.join(', ')}]；段 1（pos ${t.rows[1].pos}）[${t.rows[1].row.join(', ')}]`,
        `窗口不跨段：段首之前写 −1（get_window_topk_idxs_thd）；CP>1 的边界交换归 35 页`,
      ],
      'neutral',
      'fig1/thd',
    ),
  );
  parts.push(
    infoBox(
      RX,
      594,
      RW,
      102,
      '本图复刻了什么、简化了什么',
      [
        '精确：window / compressed 索引、因果可见集、top-k 后的校验与 +offset、THD 的按段拼接',
        '简化：indexer 分数是确定性表，不是 wq_b / weights_proj 算出来的；不画 4x 重叠池化与 RoPE',
        'fused 路径的集合与本图相同，只是索引先扁平化、再按 SM 对齐 pad 到 64 / 128 的倍数',
      ],
      'neutral',
      'fig1/scope',
    ),
  );
  const H = 716;
  return seal(parts, W, H, 'fig1-csa-index');
}

// ============================================================================
// 图 2：参数所有权与精度账本
// ============================================================================

const OWN_LABEL = { duplicated: 'duplicated', column: 'Column 接口', row: 'Row 接口', param: 'nn.Parameter' };
const DTYPE_LABEL = { fp8ctx: '随外层 FP8 上下文', bf16out: 'FP8 上下文外 · BF16', fp32: 'mark_keep_in_fp32', pdtype: 'params_dtype' };

function renderParamLedger() {
  const W = 1272;
  const L = FIG2.C;
  const parts = header(
    W,
    '图 2　参数所有权与精度账本：接口像 TP 的只有三个投影，TP group 里却只有一个 rank',
    `算例：hidden ${CFG.hidden}、${CFG.heads} heads、q_lora_rank ${CFG.qLoraRank}、v_head_dim ${CFG.vHeadDim}、qk_pos_emb ${CFG.qkPosEmbHeadDim}、o_groups ${CFG.oGroups} × o_lora_rank ${CFG.oLoraRank}、indexer ${CFG.indexerHeads} × ${CFG.indexerHeadDim}；一个 ratio-4（C）层，参数量 = 形状连乘`,
  );
  const X0 = 28;
  const colX = [X0, X0 + 228, X0 + 338, X0 + 426, X0 + 536];
  const TY = 92;
  parts.push(rect(X0 - 8, TY - 18, 730, 26 + L.rows.length * 22 + 8, 'panel'));
  ['参数张量', '形状', '参数量', '所有权', '驻留精度'].forEach((h, k) => parts.push(text(colX[k], TY, h, 'rank')));
  parts.push(line(X0, TY + 7, X0 + 714, TY + 7));
  L.rows.forEach((r, i) => {
    const y = TY + 26 + i * 22;
    parts.push(text(colX[0], y, guard(r.path, 10.5, 222, 'fig2/path'), 'sm'));
    parts.push(text(colX[1], y, `[${r.shape.join(', ')}]`, 'sm'));
    parts.push(text(colX[2], y, fmt(r.n), 'sm'));
    const ownCls = r.own === 'column' || r.own === 'row' ? 'acc1' : r.own === 'duplicated' ? 'ghost' : 'neutral';
    const ownTxt = OWN_LABEL[r.own];
    const ow = textWidth(ownTxt, 10.5) + 12;
    parts.push(rect(colX[3], y - 12, ow, 16, ownCls, 3));
    parts.push(text(colX[3] + ow / 2, y, ownTxt, ownCls === 'acc1' ? 'dim' : 'sm', 'middle'));
    const dtCls = r.dtype === 'bf16out' || r.dtype === 'fp32' ? 'acc2' : 'neutral';
    const dtTxt = DTYPE_LABEL[r.dtype];
    const dw = textWidth(dtTxt, 10.5) + 12;
    parts.push(rect(colX[4], y - 12, dw, 16, dtCls, 3));
    parts.push(text(colX[4] + dw / 2, y, dtTxt, dtCls === 'acc2' ? 'costtx' : 'sm', 'middle'));
  });
  const bottomTable = TY + 26 + L.rows.length * 22;
  const caps = [
    '芯片：acc1 = Column/Row 接口（gather_output=False / input_is_parallel=True，tp_group 只有 1 个 rank）',
    'acc2 = 精度例外：FP8 上下文外 BF16、mark_keep_in_fp32；其余随 params_dtype 或外层 FP8 上下文',
    'q_layernorm / kv_layernorm 用 attention_latent_norm_epsilon（默认继承 layernorm_epsilon）',
    `__post_init__ 把 qk_head_dim 与 kv_lora_rank 改写成 v_head_dim − qk_pos_emb_head_dim = ${FIG2.derivedQkHeadDim}，配置写的 16 / 64 不生效`,
  ];
  caps.forEach((c, k) => parts.push(text(X0, bottomTable + 14 + k * 16, guard(c, 11, 700, `fig2/cap${k}`), 'cap')));

  // 右栏
  const RX = 766;
  const RW = 478;
  const C = FIG2.C;
  const Hh = FIG2.H;
  const Ww = FIG2.W;
  const fc = FIG2.flashC;
  const wqb = C.rows.find((r) => r.path === 'core.indexer.linear_wq_b').n;
  const fwqb = fc.rows.find((r) => r.path === 'core.indexer.linear_wq_b').n;
  parts.push(
    infoBox(
      RX,
      74,
      RW,
      118,
      'TP group（size 1）：三条源码事实',
      [
        'DSv4HybridAttention.__init__ 断言 get_pg_size(pg_collection.tp) == 1',
        `Attention.__init__：heads_per_partition = ${CFG.heads}/1 = ${FIG2.headsPerPartition}；CSA n_local_heads 不除 TP`,
        'TELinear(parallel_mode="duplicated") 断言 tp_group is None',
        { text: '接口保留 Column/Row 形状，但 group 只有一个 rank、没有 AG/RS', cls: 'dim' },
      ],
      'neutral',
      'fig2/tp',
    ),
  );
  parts.push(
    infoBox(
      RX,
      206,
      RW,
      118,
      '若 TP>1 要证明的四项（分析重建，源码沉默）',
      [
        '① 三个接口投影沿哪一维分片，attn_sink 与 indexer 的 head 和怎样跟着分',
        '② 复制的 compressor / indexer 各 rank 算出同一组 top-k 还是各算各的',
        '③ compressed K / KV 的 collective 在 TP 与 CP 两个 group 上按什么顺序发生',
        { text: '④ duplicated 参数的梯度在哪个 group 归并；fused kernel 只吃扁平 (rows, H, D)', cls: 'costtx' },
      ],
      'acc2',
      'fig2/proof',
    ),
  );
  parts.push(
    infoBox(
      RX,
      338,
      RW,
      230,
      '账本合计',
      [
        { text: `C 层 ${fmt(C.total)} 参数，linear_proj ${fmt(C.proj)}（${pct(C.proj, C.total)}%）`, cls: 'dim' },
        `indexer.linear_wq_b ${fmt(wqb)}（${pct(wqb, C.total)}%）；duplicated 共 ${C.duplicatedTensors} 个张量 ${fmt(C.duplicated)}`,
        `H 层 ${fmt(Hh.total)}：compressor coff=1，wkv/wgate [${Hh.rows.find((r) => r.path === 'core.compressor.linear_wkv').shape.join(', ')}]、ape [${Hh.rows.find((r) => r.path === 'core.compressor.ape').shape.join(', ')}]`,
        `W 层 ${fmt(Ww.total)}；csa_dense_mode 的 C 层 ${fmt(FIG2.Cdense.total)}（去掉整个 indexer）`,
        `接口投影 ${C.interfaceTensors} 个张量 ${fmt(C.interface)} 参数，全部在 size-1 的 TP group 里`,
        { text: `fp32 保持 ${C.fp32Tensors} 个张量 ${fmt(C.fp32Params)} 参数；FP8 上下文外 BF16 ${C.bf16OutTensors} 个张量 ${fmt(C.bf16OutParams)} 参数`, cls: 'costtx' },
        `DSv4-Flash 配方 C 层（hidden ${fmt(CFG.flash.hidden)}、${CFG.flash.heads} heads、v_head_dim ${CFG.flash.vHeadDim}）：${fmt(fc.total)}`,
        `其中 linear_proj ${fmt(fc.proj)}（${pct(fc.proj, fc.total)}%），wq_b ${fmt(fwqb)}，FP8 外 BF16 ${fmt(fc.bf16OutParams)}`,
        `mHC（n=${MHC.n}）：mapping_proj [${MHC.mappingProj.join(', ')}] = ${fmt(MHC.mappingParams)} 参数 + ${MHC.alpha} alpha + ${MHC.bias} bias，fp32`,
        'linear_proj 在 fp8/fp4 下 set_save_original_input：不存量化副本',
        'delay_wgrad_compute 下 backward_dw 逐个冲刷 TE 线性层（含 compressor/indexer）',
      ],
      'neutral',
      'fig2/totals',
    ),
  );
  const H = Math.max(bottomTable + 14 + caps.length * 16 + 12, 338 + 230 + 24);
  return seal(parts, W, H, 'fig2-param-ledger');
}

// ============================================================================
// 图 3：分派矩阵 + Sinkhorn
// ============================================================================

function renderDispatch() {
  const W = 1272;
  const parts = header(
    W,
    '图 3　同一层走哪条 kernel 路径，以及 mHC 映射几步就近似双随机',
    `左：CompressedSparseAttention.forward 的分派（S=${fmt(FIG3.seqlen)}，attention_backend=fused）；右：hyper_connection.py::_sinkhorn_iterations 在 n=${SINKHORN.n} 上的收敛（确定性 logits）`,
  );
  const X0 = 28;
  const TY = 92;
  const colW = 158;
  const rowH = 44;
  const labelW = 150;
  parts.push(rect(X0 - 8, TY - 18, labelW + colW * FIG3_COLS.length + 16, 26 + rowH * FIG3_LAYERS.length + 8, 'panel'));
  FIG3_COLS.forEach((c, k) => parts.push(text(X0 + labelW + k * colW + colW / 2, TY, c.label, 'rank', 'middle')));
  parts.push(line(X0, TY + 7, X0 + labelW + colW * FIG3_COLS.length, TY + 7));
  FIG3_LAYERS.forEach((Lr, i) => {
    const y = TY + 18 + i * rowH;
    parts.push(text(X0, y + 20, Lr.label, 'tx'));
    FIG3.cells[i].forEach((cell, k) => {
      const x = X0 + labelW + k * colW;
      const cls = cell.loss ? 'acc2' : cell.path.startsWith('unfused') ? 'ghost' : 'acc1';
      parts.push(rect(x + 3, y + 2, colW - 6, rowH - 6, cls, 5));
      parts.push(text(x + colW / 2, y + 16, cell.path, cell.loss ? 'costtx' : cls === 'acc1' ? 'dim' : 'sm', 'middle'));
      parts.push(text(x + colW / 2, y + 31, guard(cell.kernels, 10.5, colW - 10, 'fig3/kernels'), 'sm', 'middle'));
    });
  });
  const by = TY + 18 + rowH * FIG3_LAYERS.length + 10;
  parts.push(
    infoBox(
      X0 - 8,
      by + 8,
      labelW + colW * FIG3_COLS.length + 16,
      166,
      '读法',
      [
        `谓词：use_fused = attention_backend ≠ unfused ∧ dsa_kernel_backend ≠ none；has_indexer = ratio > 1 ∧ n_comp > 0 ∧ indexer 已建`,
        `indexer 只在 ratio == 4 ∧ ¬csa_dense_mode 时构建；compressor 只在 ratio > 1 时构建（W 层两者都没有）`,
        { text: `${FIG3.distinct.length} 种路径：${FIG3.distinct.join('、')}；带 indexer loss 的格子 ${FIG3.lossCells} 个（acc2）`, cls: 'dim' },
        `Path B / C / A 的 kernel：FlashMLA 前向 + cuDNN DSA 反向；B 把 indexer 的 KL 反向提前到前向里算好`,
        { text: `attention_backend=unfused 时即使 dsa_kernel_backend=cudnn 也走 ${FIG3.unfusedOverride.path}（${FIG3.unfusedOverride.kernels}）`, cls: 'costtx' },
        `H 层在 S < 128 时 n_compressed = 0：S=64 的 cudnn·train 走 ${FIG3.hcaShort.path}（${FIG3.hcaShort.kernels}）`,
        `THD 与 SBHD 的分派谓词相同，CP>1 另走 _forward_thd_cp；fused compressor 的 gate 另见正文`,
      ],
      'neutral',
      'fig3/read',
    ),
  );

  // Sinkhorn
  const RX = 836;
  const RW = 408;
  const sy = TY - 18;
  parts.push(rect(RX, sy, RW, 178, 'panel'));
  parts.push(text(RX + 12, sy + 20, 'Sinkhorn：softmax(行)+ε → 列归一 → (行归一, 列归一)×(k−1)', 'tx'));
  parts.push(line(RX + 12, sy + 28, RX + RW - 12, sy + 28));
  parts.push(text(RX + 12, sy + 46, '迭代 k', 'rank'));
  parts.push(text(RX + 120, sy + 46, 'max|行和 − 1|', 'rank'));
  parts.push(text(RX + 260, sy + 46, 'max|列和 − 1|', 'rank'));
  SINKHORN.trace.forEach((t, i) => {
    const y = sy + 66 + i * 20;
    const last = i === SINKHORN.trace.length - 1;
    parts.push(text(RX + 12, y, String(t.k), last ? 'dim' : 'sm'));
    parts.push(text(RX + 120, y, t.rowDev.toExponential(2), last ? 'dim' : 'sm'));
    parts.push(text(RX + 260, y, t.colDev.toExponential(2), last ? 'dim' : 'sm'));
  });
  const my = sy + 194;
  parts.push(rect(RX, my, RW, 150, 'panel'));
  parts.push(text(RX + 12, my + 20, `k=${CFG.sinkhornIterations}（配方 mhc_sinkhorn_iterations）后的 H_res（${SINKHORN.n}×${SINKHORN.n}）`, 'tx'));
  parts.push(line(RX + 12, my + 28, RX + RW - 12, my + 28));
  const cell = 46;
  SINKHORN.final.forEach((row, i) => {
    row.forEach((v, j) => {
      const x = RX + 12 + j * cell;
      const y = my + 40 + i * 22;
      parts.push(rect(x, y - 12, cell - 4, 18, v === Math.max(...row) ? 'acc1' : 'ghost', 3));
      parts.push(text(x + (cell - 4) / 2, y + 1, v.toFixed(2), v === Math.max(...row) ? 'dim' : 'sm', 'middle'));
    });
    const rs = row.reduce((a, b) => a + b, 0);
    parts.push(text(RX + 12 + SINKHORN.n * cell + 8, my + 41 + i * 22, `Σ行 = ${rs.toFixed(4)}`, 'sm'));
  });
  const cs = Array.from({ length: SINKHORN.n }, (_, j) => SINKHORN.final.reduce((s, r) => s + r[j], 0));
  parts.push(text(RX + 12, my + 40 + SINKHORN.n * 22 + 4, `Σ列 = [${cs.map((v) => v.toFixed(4)).join(', ')}]`, 'sm'));
  parts.push(
    infoBox(
      RX,
      my + 166,
      RW,
      102,
      'mHC 参数怎么算、怎么同步',
      [
        `mapping_proj = nn.Linear(n·C → n²+2n)：n=${MHC.n}、C=${CFG.hidden} → [${MHC.mappingProj.join(', ')}]`,
        '映射在 fp32 里算（权重 mark_keep_in_fp32），H 再转回激活 dtype',
        { text: 'sequence_parallel 属性 → finalize_model_grads 按 TP group 求和', cls: 'dim' },
        'Sinkhorn 后端：Triton → cuTile → native；MHC_FORCE_BACKEND 强制',
      ],
      'neutral',
      'fig3/mhc',
    ),
  );
  const H = Math.max(by + 8 + 166 + 30, my + 166 + 102 + 24);
  return seal(parts, W, H, 'fig3-dispatch');
}

// ============================================================================

const outputs = new Map([
  ['megatron_dsv4_tp_csa_index.svg', renderCsaIndex()],
  ['megatron_dsv4_tp_param_ledger.svg', renderParamLedger()],
  ['megatron_dsv4_tp_dispatch.svg', renderDispatch()],
]);

export {
  CFG, FIG1, FIG2, FIG3, FIG3_LAYERS, FIG3_COLS, MHC, SINKHORN, outputs,
  windowTopkIdxs, compressTopkIdxs, compressorRows, indexerTopkWithOffset, rowsPerQuery,
  buildCuSeqlensKvFull, catPerSegmentOrder, compressTopkIdxsThdRow, ledgerFor, mhcLedger,
  useFusedDsaKernels, dispatch, sinkhornIterations, sinkhornDeviation, fmt, pct,
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
  console.log('\n图 1：', JSON.stringify({ nComp: FIG1.nComp, invalidTotal: FIG1.invalidTotal, firstTopkSaturated: FIG1.firstTopkSaturated, scale: FIG1.scale, thd: { cuFull: FIG1.thd.cuFull, order: FIG1.thd.order } }));
  for (const k of ['C', 'Cdense', 'H', 'W', 'flashC']) {
    const l = FIG2[k];
    console.log(`图 2 ${k}：total=${l.total} proj=${l.proj}(${pct(l.proj, l.total)}%) fp32=${l.fp32Params}/${l.fp32Tensors} bf16out=${l.bf16OutParams}/${l.bf16OutTensors} dup=${l.duplicated}/${l.duplicatedTensors} iface=${l.interface}/${l.interfaceTensors}`);
  }
  console.log('图 2 mHC：', JSON.stringify(MHC));
  FIG3.cells.forEach((row, i) => console.log(`图 3 ${FIG3_LAYERS[i].key}：${row.map((c) => c.path).join(' | ')}`));
  console.log('图 3 Sinkhorn：', JSON.stringify(SINKHORN.trace));
}
