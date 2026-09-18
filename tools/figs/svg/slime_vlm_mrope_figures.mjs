// 图：slime Qwen3.5-VL 原生训练路径上两个决定性变换的最小重放。
// ① packed MRoPE：build_packed_mrope_position_ids 按 cu_seqlens 逐样本从 0 计数，视觉块之后文本位置只推进
//    max(h, w) // spatial_merge_size，所以样例里视觉块后的文本从 3 而不是 5 继续；与 SGLang 上游
//    get_rope_index 的“上一段 max() + 1”在 grid_t = 1 时相同，decode 续算也落在同一位置。
// ② CP=2 的视觉注入：_inject_vision_embeddings 在完整序列上把视觉 token 映射到 feature 行，
//    再按 get_packed_cp_local_indices 取本 rank 的首尾两段；每个 CP rank 都对全部图像跑 ViT。
//    CP=1 时同一个整除检查会拒绝奇数长度的 packed 序列。
// 源码基线：THUDM/slime@4c193f1f37509cca70f0e88807a9305b70f63f4e；
// 依赖侧对照：sgl-project/sglang@0b3bb0cbe318（v0.5.15.post1）mrope_rope_index.py::get_rope_index、
// NVIDIA/Megatron-LM@1dcf0dafa884 rope_utils.py::_apply_rotary_pos_emb_thd（经 slime megatron.patch 的 packed_seq 开关）。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
// 图 1 要讲清楚：两条 7-token 样本打包后，T/H/W 三轴坐标逐样本归零；视觉块在 H、W 轴上占 start..start+max(h',w')−1，
// 文本续接 start + max(h',w')（样例 3，非方图 1×4×6 为 4），被拒绝的“按 token 数推进”会得到 5 与 7；
// SGLang 的 max()+1 与 decode 的 delta 续算给出同一位置。acc1 标视觉 token 与成立的规则，acc2 标样本边界与被拒绝的规则，
// 依赖侧用虚线框。
// 图 2 要讲清楚：get_batch 把每条样本 pad 到 2·CP 的倍数后按 zigzag 切两段，cu_seqlens × CP 回到完整长度；
// 完整 ids 上 feature_indices 把 8 个视觉 token 编号 0..7；rank 0 取 [0,1,6,7,8,9,14,15] 只用行 0、4，
// rank 1 取其余位置用 6 行；每个 rank 都算完 8 行（代价，acc2）；CP=1 奇数长度直接 ValueError（失败边界，acc2）。
//
// 用法：node tools/figs/svg/slime_vlm_mrope_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// ---------------- 冻结的示例输入 ----------------
// tests/test_qwen3_5_vl_native.py::test_packed_mrope_resets_positions_for_each_sample 的输入
export const CFG = Object.freeze({
  seqA: [99, 10, 10, 10, 10, 98, 7],
  seqB: [99, 10, 10, 10, 10, 98, 8],
  imageGrid: [1, 4, 4],
  nonSquareGrid: [1, 4, 6], // 追加的非方图对照：同一条规则下推进 max 边而不是面积
  ids: { image: 10, video: 20, visionStart: 99 },
  merge: 2,
  cp: { size: 2, tp: 1, padMultiplier: 4, padToken: 0 }, // TP=1、--data-pad-size-multiplier 4 让尾部对齐 pad 为 0
  // slime_plugins/models/qwen3_5_vl_utils.py::build_packed_mrope_position_ids 的推进语句（原文）
  sourceRule: 'current_position += max(int(grid[1]), int(grid[2])) // spatial_merge_size',
  // geo3k 单轮脚本：64 prompts × 8 samples、TP2、CP1，每轮 1 个训练 step 且满足 logprob 复用条件
  geo3k: { prompts: 64, samplesPerPrompt: 8, tp: 2, cp: 1, forwardPasses: 1, padMultiplier: 128 }, // --data-pad-size-multiplier 默认 128
});

const range = (n) => Array.from({ length: n }, (_, i) => i);
const sum = (xs) => xs.reduce((a, x) => a + x, 0);

// ---------------- 对源码算法的最小复现 ----------------

// slime_plugins/models/qwen3_5_vl_utils.py::_vision_positions
export function visionPositions(start, grid, merge) {
  const [t, h0, w0] = grid;
  const h = Math.floor(h0 / merge);
  const w = Math.floor(w0 / merge);
  const T = [];
  const H = [];
  const W = [];
  for (let ti = 0; ti < t; ti += 1) {
    for (let hi = 0; hi < h; hi += 1) {
      for (let wi = 0; wi < w; wi += 1) {
        T.push(start + ti);
        H.push(start + hi);
        W.push(start + wi);
      }
    }
  }
  return [T, H, W];
}

// slime_plugins/models/qwen3_5_vl_utils.py::build_packed_mrope_position_ids（仅图像；视频 grid 先按帧拆成 t=1）
// advance 可替换成被拒绝的“按 token 数推进”以做对照。
export function buildPackedMropePositions(ids, cu, grids, { image, video, visionStart }, merge, advance = 'slime') {
  const pos = [[], [], []];
  const pieces = [];
  const gridIter = grids[Symbol.iterator]();
  for (let s = 0; s + 1 < cu.length; s += 1) {
    const tokens = ids.slice(cu[s], cu[s + 1]);
    let cursor = 0;
    let current = 0;
    const seqPieces = [];
    const starts = tokens.map((t, i) => (t === visionStart ? i : -1)).filter((i) => i >= 0);
    for (const vs of starts) {
      const modality = tokens[vs + 1];
      if (modality !== image && modality !== video) continue;
      const first = tokens.indexOf(modality, cursor);
      const next = gridIter.next();
      if (first < 0 || next.done) throw new Error('Qwen3.5-VL tokens and vision grids do not match');
      const grid = next.value;
      const textLength = first - cursor;
      if (textLength) {
        seqPieces.push({ kind: 'text', from: cursor, len: textLength, start: current });
        for (let k = 0; k < 3; k += 1) range(textLength).forEach((i) => pos[k].push(current + i));
        current += textLength;
      }
      const vp = visionPositions(current, grid, merge);
      const n = vp[0].length;
      seqPieces.push({ kind: 'vision', from: first, len: n, start: current, grid });
      for (let k = 0; k < 3; k += 1) pos[k].push(...vp[k]);
      const step = advance === 'slime' ? Math.floor(Math.max(grid[1], grid[2]) / merge) : n;
      seqPieces.push({ kind: 'advance', by: step, from: current, to: current + step });
      current += step;
      cursor = first + n;
    }
    if (cursor < tokens.length) {
      const textLength = tokens.length - cursor;
      seqPieces.push({ kind: 'text', from: cursor, len: textLength, start: current });
      for (let k = 0; k < 3; k += 1) range(textLength).forEach((i) => pos[k].push(current + i));
    }
    if (pos[0].length !== cu[s + 1]) throw new Error('Qwen3.5-VL vision token count does not match its grid');
    pieces.push(seqPieces);
  }
  if (!gridIter.next().done) throw new Error('Unused Qwen3.5-VL image grids');
  return { pos, pieces };
}

// sglang@0b3bb0cbe318 python/sglang/srt/layers/rotary_embedding/mrope_rope_index.py::get_rope_index
// （单条序列、仅图像、qwen3_5 分支）：每段起点 st_idx = 上一段 max() + 1；并返回 mrope_position_delta = max + 1 − len。
export function sglangRopeIndex(tokens, grids, { image, visionStart }, merge) {
  const list = [];
  let st = 0;
  let gi = 0;
  const nImages = tokens.filter((t, i) => i > 0 && tokens[i - 1] === visionStart && t === image).length;
  const lastMax = () => Math.max(...list[list.length - 1].flat());
  for (let k = 0; k < nImages; k += 1) {
    const ed = tokens.indexOf(image, st);
    const [t, h0, w0] = grids[gi];
    gi += 1;
    const [gh, gw] = [Math.floor(h0 / merge), Math.floor(w0 / merge)];
    const textLen = ed - st;
    const stIdx = list.length ? lastMax() + 1 : 0;
    list.push(range(3).map(() => range(textLen).map((i) => stIdx + i)));
    const T = [];
    const H = [];
    const W = [];
    for (let ti = 0; ti < t; ti += 1) for (let hi = 0; hi < gh; hi += 1) for (let wi = 0; wi < gw; wi += 1) {
      T.push(ti + textLen + stIdx);
      H.push(hi + textLen + stIdx);
      W.push(wi + textLen + stIdx);
    }
    list.push([T, H, W]);
    st = ed + t * gh * gw;
  }
  if (st < tokens.length) {
    const stIdx = list.length ? lastMax() + 1 : 0;
    list.push(range(3).map(() => range(tokens.length - st).map((i) => stIdx + i)));
  }
  const pos = range(3).map((k) => list.flatMap((piece) => piece[k]));
  const max = Math.max(...pos.flat());
  return { pos, delta: max + 1 - tokens.length };
}

// slime/backends/megatron_utils/cp_utils.py::slice_with_cp
export function sliceWithCp(tokens, cpSize, cpRank, padValue) {
  if (cpSize === 1) return tokens.slice();
  const chunk = Math.ceil(tokens.length / (2 * cpSize));
  const padded = tokens.concat(Array(2 * cpSize * chunk - tokens.length).fill(padValue));
  return padded
    .slice(chunk * cpRank, chunk * (cpRank + 1))
    .concat(padded.slice(chunk * (2 * cpSize - cpRank - 1), chunk * (2 * cpSize - cpRank)));
}

// slime/backends/megatron_utils/data.py::get_batch（非 allgather_cp 分支）：逐条切片后拼接、按 TP·multiplier 对齐，
// cu_seqlens 由本地长度乘 CP 得到完整长度边界。
export function getBatchLayout(seqs, { size, tp, padMultiplier, padToken }) {
  const padSize = tp * padMultiplier;
  const ranks = range(size).map((r) => {
    const parts = seqs.map((s) => sliceWithCp(s, size, r, padToken));
    const localCu = [0];
    parts.forEach((p) => localCu.push(localCu[localCu.length - 1] + p.length));
    let tokens = parts.flat();
    const pad = (padSize - (tokens.length % padSize)) % padSize;
    if (pad) {
      tokens = tokens.concat(Array(pad).fill(padToken));
      localCu.push(localCu[localCu.length - 1] + pad);
    }
    return { rank: r, tokens, pad, cu: localCu.map((x) => x * size) };
  });
  return { padSize, ranks, cu: ranks[0].cu };
}

// slime_plugins/models/qwen3_5_vl_utils.py::get_packed_cp_local_indices
export function getPackedCpLocalIndices(cu, cpSize, cpRank) {
  const out = [];
  for (let s = 0; s + 1 < cu.length; s += 1) {
    const len = cu[s + 1] - cu[s];
    if (len % (2 * cpSize) !== 0) {
      throw new Error(`Packed sequence length ${len} must be divisible by 2 * CP size ${cpSize}`);
    }
    const chunk = len / (2 * cpSize);
    const first = cu[s] + cpRank * chunk;
    const second = cu[s] + (2 * cpSize - cpRank - 1) * chunk;
    out.push(...range(chunk).map((i) => first + i), ...range(chunk).map((i) => second + i));
  }
  return out;
}

// slime_plugins/models/qwen3_5_vl_utils.py::gather_packed_input_ids
export function gatherPackedInputIds(layout, cpSize) {
  const full = Array(layout.cu[layout.cu.length - 1]).fill(null);
  layout.ranks.forEach((r) => {
    const idx = getPackedCpLocalIndices(layout.cu, cpSize, r.rank);
    if (idx.length !== r.tokens.length) throw new Error(`CP rank ${r.rank} has ${r.tokens.length} tokens, expected ${idx.length}`);
    idx.forEach((p, i) => {
      full[p] = r.tokens[i];
    });
  });
  return full;
}

// slime_plugins/models/qwen3_5_vl.py::Qwen3_5VLModel._inject_vision_embeddings（图像分支的索引部分）
export function injectVision(fullIds, cu, cpSize, rank, localIds, tokenId, numFeatures) {
  const fullPositions = fullIds.map((t, i) => (t === tokenId ? i : -1)).filter((i) => i >= 0);
  if (fullPositions.length !== numFeatures) {
    throw new Error(`Qwen3.5-VL token/features mismatch: ${fullPositions.length} tokens, ${numFeatures} features`);
  }
  const featureIndices = Array(fullIds.length).fill(-1);
  fullPositions.forEach((p, k) => {
    featureIndices[p] = k;
  });
  const localIndices = getPackedCpLocalIndices(cu, cpSize, rank);
  const localFeature = localIndices.map((p) => featureIndices[p]);
  const mask = localFeature.map((f) => f >= 0);
  const expected = localIds.map((t) => t === tokenId);
  if (mask.some((m, i) => m !== expected[i])) throw new Error('Qwen3.5-VL CP token layout does not match its full packed sequence');
  return { fullPositions, featureIndices, localIndices, localFeature, rowsUsed: localFeature.filter((f) => f >= 0) };
}

export function model(cfg = CFG) {
  const { seqA, seqB, imageGrid, nonSquareGrid, ids, merge } = cfg;
  const packed = seqA.concat(seqB);
  const cu = [0, seqA.length, packed.length];
  const slime = buildPackedMropePositions(packed, cu, [imageGrid, imageGrid], ids, merge);
  const byTokens = buildPackedMropePositions(packed, cu, [imageGrid, imageGrid], ids, merge, 'tokens');
  const merged = (g) => g[0] * Math.floor(g[1] / merge) * Math.floor(g[2] / merge);
  const nsTokens = [ids.visionStart, ...Array(merged(nonSquareGrid)).fill(ids.image), 98, 7];
  const nonSquare = buildPackedMropePositions(nsTokens, [0, nsTokens.length], [nonSquareGrid], ids, merge);
  const nonSquareByTokens = buildPackedMropePositions(nsTokens, [0, nsTokens.length], [nonSquareGrid], ids, merge, 'tokens');
  const sglA = sglangRopeIndex(seqA, [imageGrid], ids, merge);
  const sglNs = sglangRopeIndex(nsTokens, [nonSquareGrid], ids, merge);

  const posA = slime.pos.map((axis) => axis.slice(0, seqA.length));
  const posB = slime.pos.map((axis) => axis.slice(seqA.length));
  const visionStartPos = slime.pieces[0].find((p) => p.kind === 'vision').start;
  const resume = slime.pieces[0].find((p) => p.kind === 'advance').to;
  const resumeByTokens = byTokens.pieces[0].find((p) => p.kind === 'advance').to;
  const nsResume = nonSquare.pieces[0].find((p) => p.kind === 'advance').to;
  const nsResumeByTokens = nonSquareByTokens.pieces[0].find((p) => p.kind === 'advance').to;
  const nextIndex = seqA.length; // 第一个生成 token 的下标
  const tailStartIndex = slime.pieces[0].filter((p) => p.kind === 'text').pop().from;
  const decode = {
    index: nextIndex,
    trainSide: resume + (nextIndex - tailStartIndex),
    sglangSide: nextIndex + sglA.delta,
    delta: sglA.delta,
    tailStartIndex,
  };

  // ---- CP=2 注入 ----
  const layout = getBatchLayout([seqA, seqB], cfg.cp);
  const full = gatherPackedInputIds(layout, cfg.cp.size);
  const featuresPerImage = merged(imageGrid);
  const numFeatures = 2 * featuresPerImage;
  const fullPos = buildPackedMropePositions(full, layout.cu, [imageGrid, imageGrid], ids, merge).pos;
  const ranks = layout.ranks.map((r) => {
    const inj = injectVision(full, layout.cu, cfg.cp.size, r.rank, r.tokens, ids.image, numFeatures);
    return { ...r, ...inj, localT: inj.localIndices.map((p) => fullPos[0][p]) };
  });
  // CP=1：slice_with_cp 不补齐，cu_seqlens 是原始长度加尾部对齐 pad 段
  const cp1Layout = getBatchLayout([seqA, seqB], { ...cfg.cp, size: 1 });
  const geo3kCp1Layout = getBatchLayout([seqA, seqB], { ...cfg.cp, size: 1, tp: cfg.geo3k.tp, padMultiplier: cfg.geo3k.padMultiplier });
  let cp1Error = null;
  try {
    getPackedCpLocalIndices(cp1Layout.cu, 1, 0);
  } catch (e) {
    cp1Error = e.message;
  }

  const g = cfg.geo3k;
  const samples = g.prompts * g.samplesPerPrompt;
  const vitPasses = samples * g.forwardPasses * g.tp * g.cp;

  return {
    packed,
    cu,
    posA,
    posB,
    visionStartPos,
    resume,
    resumeByTokens,
    nonSquare: { tokens: nsTokens, pos: nonSquare.pos, resume: nsResume, resumeByTokens: nsResumeByTokens, merged: merged(nonSquareGrid) },
    sglang: { a: sglA, nonSquare: sglNs },
    decode,
    cp: {
      layout,
      full,
      fullPos,
      featuresPerImage,
      numFeatures,
      ranks,
      computedRows: cfg.cp.size * numFeatures,
      injectedRows: sum(ranks.map((r) => r.rowsUsed.length)),
      cp1Error,
      cp1Cu: cp1Layout.cu,
      cp1Pad: cp1Layout.ranks[0].pad,
      geo3kCp1Cu: geo3kCp1Layout.cu,
    },
    geo3k: { samples, vitPasses, distinctImages: g.prompts, redundancy: vitPasses / g.prompts },
  };
}

// ---------------- 渲染 ----------------
const esc = (v) => String(v).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
export function textWidth(s, size) {
  // CJK 统一表意字、全角标点与符号按字号计宽，其余字符按 0.58 倍估算
  let w = 0;
  for (const ch of String(s)) w += /[\u2E80-\u9FFF\uF900-\uFAFF\uFE30-\uFE4F\uFF00-\uFFEF]/.test(ch) ? size : size * 0.58;
  return w;
}
export const FONT_SIZE = { ti: 19, su: 12, pt: 14, tx: 12, sm: 10.5, cap: 11.5, cell: 12, idx: 9.5 };
const STYLE = `
  text{font-family:"Segoe UI","Microsoft YaHei","PingFang SC",system-ui,sans-serif;fill:#2A313B}
  .ti{font-size:19px;font-weight:700;fill:#1F2430}
  .su{font-size:12px;fill:#747C88}
  .pt{font-size:14px;font-weight:700}
  .tx{font-size:12px;fill:#38414D}
  .sm{font-size:10.5px;fill:#5B6470}
  .cap{font-size:11.5px;fill:#5B6470}
  .cell{font-size:12px;fill:#2A313B;font-weight:600}
  .idx{font-size:9.5px;fill:#8A919C}
  .panel{fill:#FBFCFE;stroke:#D9DEE7;stroke-width:1.2}
  .neutral{fill:#fff;stroke:#AEB6C2;stroke-width:1.2}
  .ghost{fill:#F5F7FA;stroke:#D9DEE7;stroke-width:1.1}
  .dep{fill:#F5F7FA;stroke:#AEB6C2;stroke-width:1.2;stroke-dasharray:5 4}
  .acc1{fill:#EAF1FD;stroke:#2563EB;stroke-width:1.5}
  .acc2{fill:#FCF1E6;stroke:#C3651F;stroke-width:1.5}
  .cut{fill:none;stroke:#C3651F;stroke-width:2;stroke-dasharray:6 4}
  .main{fill:none;stroke:#2563EB;stroke-width:2;marker-end:url(#arrowMain)}
  .aux{fill:none;stroke:#AEB6C2;stroke-width:1.3;stroke-dasharray:5 4}
`;

function canvas(W, H, title, desc) {
  const o = [];
  o.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="title desc">`);
  o.push(`<title id="title">${esc(title)}</title>`);
  o.push(`<desc id="desc">${esc(desc)}</desc>`);
  o.push('<defs><marker id="arrowMain" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#2563EB"/></marker></defs>');
  o.push(`<style>${STYLE}</style><rect width="${W}" height="${H}" fill="white"/>`);
  const rect = (x, y, w, h, cls = 'neutral', r = 6) =>
    o.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" class="${cls}"/>`);
  const text = (x, y, s, cls = 'tx', anchor = 'start') =>
    o.push(`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${esc(s)}</text>`);
  const line = (x1, y1, x2, y2, cls = 'aux') => o.push(`<path d="M${x1} ${y1} L${x2} ${y2}" class="${cls}"/>`);
  return { o, rect, text, line };
}

function renderPositions(m, cfg = CFG) {
  const W = 1180;
  const H = 700;
  const { o, rect, text, line } = canvas(
    W,
    H,
    'slime packed MRoPE：逐样本归零，视觉块之后文本只推进 max(h, w) // merge',
    '上半部分按 build_packed_mrope_position_ids 重放两条 7-token 样本的 T/H/W 三轴坐标，样本边界处 current_position 归零，视觉块后的文本从 3 继续。下半部分解释推进量：视觉块在 H、W 轴上只占 max(h\', w\') 个坐标，方图与 1×4×6 非方图分别续接 3 与 4；被拒绝的按 token 数推进会得到 5 与 7；SGLang 上游 get_rope_index 的上一段最大值加一与 decode 的 delta 续算给出同一位置。',
  );

  text(24, 34, 'packed MRoPE：每条样本从 0 计数，视觉块之后文本只推进 max(h, w) // merge', 'ti');
  text(24, 56, `输入取自 test_packed_mrope_resets_positions_for_each_sample：两条 7-token 样本各含 1 张 ${cfg.imageGrid.join('×')} 的图，merge ${cfg.merge}；99 = vision_start，10 = image token，其余按文本处理`, 'su');

  // ---------- ① 三轴坐标 ----------
  const py = 72;
  rect(24, py, 1132, 300, 'panel', 10);
  text(40, py + 24, `① build_packed_mrope_position_ids：cu_seqlens = [${m.cu.join(', ')}]，每段 current_position 从 0 开始`, 'pt');
  const x0 = 132;
  const cw = 52;
  const ch = 30;
  const rowY = { idx: py + 50, ids: py + 58, T: py + 104, H: py + 140, W: py + 176 };
  text(40, rowY.ids + 20, 'input_ids', 'tx');
  const posAll = m.posA.map((axis, k) => axis.concat(m.posB[k]));
  m.packed.forEach((tok, i) => {
    const x = x0 + i * cw;
    text(x + cw / 2 - 2, rowY.idx, `#${i}`, 'idx', 'middle');
    rect(x, rowY.ids, cw - 4, ch, tok === cfg.ids.image ? 'acc1' : 'neutral', 4);
    text(x + cw / 2 - 2, rowY.ids + 20, tok, 'cell', 'middle');
  });
  ['T', 'H', 'W'].forEach((axis, k) => {
    const y = rowY[axis];
    text(40, y + 20, `${axis}（${['时间', '高', '宽'][k]}）`, 'tx');
    posAll[k].forEach((v, i) => {
      const x = x0 + i * cw;
      rect(x, y, cw - 4, ch, m.packed[i] === cfg.ids.image ? 'acc1' : 'ghost', 4);
      text(x + cw / 2 - 2, y + 20, v, 'cell', 'middle');
    });
  });
  const cutX = x0 + m.cu[1] * cw - 2;
  line(cutX, rowY.ids - 4, cutX, rowY.W + ch + 34, 'cut');

  // 分段标注（样本 A）
  const segY = rowY.W + ch + 12;
  const pa = m.posA;
  const vis = m.visionStartPos;
  const bracket = (from, len, label, cls) => {
    const x = x0 + from * cw;
    const w = len * cw - 4;
    rect(x, segY, w, 22, cls, 4);
    text(x + w / 2, segY + 15, label, 'sm', 'middle');
  };
  bracket(0, 1, '文本', 'ghost');
  bracket(1, 4, `视觉块 start = ${vis}`, 'acc1');
  bracket(5, 2, `文本续接 ${m.resume}`, 'acc1');
  bracket(7, 7, '样本 B：边界处归零，三轴与 A 相同', 'ghost');
  text(40, segY + 15, '样本 A 分段', 'sm');

  // 右栏：样本 A 的逐段状态
  const bx = x0 + m.packed.length * cw + 12;
  const bw = 1140 - bx;
  rect(bx, rowY.ids, bw, rowY.W + ch - rowY.ids + 34, 'neutral', 6);
  [
    ['样本 A 逐段（cursor / current_position）', 'tx'],
    [`#0 文本 99 → 位置 0，current = ${vis}`, 'sm'],
    [`#1–#4 视觉 4 个 token，start = ${vis}`, 'sm'],
    [`T ${pa[0].slice(1, 5).join(' ')}｜H ${pa[1].slice(1, 5).join(' ')}｜W ${pa[2].slice(1, 5).join(' ')}`, 'sm'],
    [`推进 max(${cfg.imageGrid[1]}, ${cfg.imageGrid[2]}) // ${cfg.merge} = ${m.resume - vis} → current = ${m.resume}`, 'sm'],
    [`#5–#6 文本 98、7 → 位置 ${m.resume}、${m.resume + 1}`, 'sm'],
    ['#7 cu_seqlens 边界 → current = 0', 'sm'],
  ].forEach(([s, cls], i) => text(bx + 12, rowY.ids + 22 + i * 25, s, cls));

  const ey = segY + 50;
  text(40, ey, `推进语句：${cfg.sourceRule}`, 'tx');
  text(40, ey + 22, `样本 A：${vis} + max(${cfg.imageGrid[1]}, ${cfg.imageGrid[2]}) // ${cfg.merge} = ${m.resume}，视觉块后的 98 从 ${m.resume} 继续；若按视觉 token 数推进则是 ${vis} + 4 = ${m.resumeByTokens}`, 'tx');

  // ---------- ② 推进量从哪来 ----------
  const qy = 386;
  rect(24, qy, 560, 276, 'panel', 10);
  text(40, qy + 24, "② 视觉块在 H、W 轴上只占 max(h', w') 个坐标", 'pt');
  const drawGrid = (gx, gy, grid, start, resume, label) => {
    const gh = grid[1] / cfg.merge;
    const gw = grid[2] / cfg.merge;
    const cs = 40;
    text(gx, gy - 8, label, 'tx');
    for (let h = 0; h < gh; h += 1) {
      for (let w = 0; w < gw; w += 1) {
        rect(gx + 34 + w * cs, gy + h * cs, cs - 4, cs - 4, 'acc1', 3);
        text(gx + 34 + w * cs + (cs - 4) / 2, gy + h * cs + 22, `${start + h},${start + w}`, 'sm', 'middle');
      }
    }
    range(gh).forEach((h) => text(gx + 26, gy + h * cs + 22, `H ${start + h}`, 'idx', 'end'));
    range(gw).forEach((w) => text(gx + 34 + w * cs + (cs - 4) / 2, gy + gh * cs + 12, `W ${start + w}`, 'idx', 'middle'));
    const by = gy + gh * cs + 30;
    text(gx, by, `${gh * gw} 个 token，最大坐标 ${start + Math.max(gh, gw) - 1}`, 'sm');
    text(gx, by + 16, `文本续接 ${start} + ${Math.max(gh, gw)} = ${resume}`, 'sm');
  };
  drawGrid(48, qy + 64, cfg.imageGrid, m.visionStartPos, m.resume, `方图 grid ${cfg.imageGrid.join('×')}`);
  drawGrid(292, qy + 64, cfg.nonSquareGrid, m.visionStartPos, m.nonSquare.resume, `非方图 grid ${cfg.nonSquareGrid.join('×')}`);
  text(40, qy + 228, '规则等价于“文本从视觉块用过的最大坐标 + 1 开始”：推进量随图的长边增长，', 'sm');
  text(40, qy + 246, `不随面积增长；grid_t = 1（图像，视频先按帧拆成 t = 1）时 T 轴不影响最大值。`, 'sm');

  // ---------- ③ 规则对照 ----------
  const rx = 596;
  rect(rx, qy, 560, 276, 'panel', 10);
  text(rx + 16, qy + 24, '③ 同一输入下三种续接位置', 'pt');
  const rows = [
    ['slime：+ max(h, w) // merge', `方图 ${m.resume}，非方图 ${m.nonSquare.resume}`, 'acc1'],
    ['按 token 数推进（被拒绝）', `方图 ${m.resumeByTokens}，非方图 ${m.nonSquare.resumeByTokens}：空出坐标，与推理侧不一致`, 'acc2'],
    ['SGLang 纯图像分支 / 回退 get_rope_index（依赖侧源码）', `next_pos += max(t, h', w')，或上一段 max() + 1：方图 ${m.sglang.a.pos[0][5]}，非方图 ${m.sglang.nonSquare.pos[0][m.nonSquare.merged + 1]}`, 'dep'],
  ];
  rows.forEach(([a, b, cls], i) => {
    const y = qy + 42 + i * 58;
    rect(rx + 16, y, 528, 48, cls, 5);
    text(rx + 28, y + 19, a, 'tx');
    text(rx + 28, y + 38, b, 'sm');
  });
  const dy = qy + 42 + 3 * 58;
  rect(rx + 16, dy, 528, 48, 'dep', 5);
  text(rx + 28, dy + 19, `decode 续算（依赖侧）：delta = max + 1 − len = ${m.sglang.a.delta}`, 'tx');
  text(rx + 28, dy + 38, `第一个生成 token 下标 ${m.decode.index}：训练侧 ${m.resume} + (${m.decode.index} − ${m.decode.tailStartIndex}) = ${m.decode.trainSide}，SGLang ${m.decode.index} + (${m.decode.delta}) = ${m.decode.sglangSide}`, 'sm');

  text(24, 688, '阅读顺序：① 逐样本归零与三轴坐标 → ② 推进量为什么是长边 → ③ 与被拒绝规则、SGLang 上游和 decode 续算对账。', 'cap');
  o.push('</svg>');
  return o.join('\n');
}

function renderCpInjection(m, cfg = CFG) {
  const W = 1180;
  const H = 720;
  const c = m.cp;
  const { o, rect, text, line } = canvas(
    W,
    H,
    'slime Qwen3.5-VL 在 CP=2 下的视觉注入：完整位置 → feature 行 → 本 rank 局部索引',
    '上半部分给出 get_batch 之后由 all_gather 重建的完整 16 个 token、zigzag 两段归属与 feature_indices：8 个视觉 token 编号 0 到 7。中间两条泳道分别给出 rank 0 与 rank 1 的 local_indices、本地 ids 与 local_feature_indices：rank 0 只注入行 0 与 4，rank 1 注入其余 6 行。下半部分给出代价与失败边界：每个 rank 都对两张图跑完整 ViT，本例共算 16 行、注入 8 行；CP=1 时 cu_seqlens [0, 7, 14] 的奇数长度被整除检查拒绝。',
  );

  text(24, 34, 'CP=2 的视觉注入：完整序列位置 → feature 行 → 本 rank 局部索引', 'ti');
  text(24, 56, `同一对样本经 get_batch：每条 pad 到 2·CP 的倍数（7 → ${c.layout.cu[1]}），TP ${cfg.cp.tp}、--data-pad-size-multiplier ${cfg.cp.padMultiplier} 时尾部对齐 pad = ${c.layout.ranks[0].pad}；cu_seqlens = 本地长度 × CP = [${c.layout.cu.join(', ')}]`, 'su');

  const x0 = 200;
  const cw = 54;
  const ch = 28;

  // ---------- ① 完整序列 ----------
  const py = 72;
  rect(24, py, 1132, 206, 'panel', 10);
  text(40, py + 24, '① 完整序列：gather_packed_input_ids 按各 rank 的 local_indices 写回，再在完整 ids 上编号视觉 token', 'pt');
  const owner = Array(c.full.length).fill(null);
  c.ranks.forEach((r) => r.localIndices.forEach((p) => {
    owner[p] = r.rank;
  }));
  const yIdx = py + 48;
  const yIds = py + 56;
  const yOwn = py + 92;
  const yFeat = py + 128;
  const yPos = py + 164;
  text(40, yIds + 19, '完整 ids', 'tx');
  text(40, yOwn + 19, 'zigzag 归属', 'tx');
  text(40, yFeat + 19, 'feature_indices', 'tx');
  text(40, yPos + 19, 'MRoPE T 轴', 'tx');
  c.full.forEach((tok, i) => {
    const x = x0 + i * cw;
    text(x + cw / 2 - 2, yIdx, `#${i}`, 'idx', 'middle');
    const isVis = tok === cfg.ids.image;
    rect(x, yIds, cw - 4, ch, isVis ? 'acc1' : tok === cfg.cp.padToken ? 'ghost' : 'neutral', 4);
    text(x + cw / 2 - 2, yIds + 19, tok, 'cell', 'middle');
    rect(x, yOwn, cw - 4, ch, 'ghost', 4);
    text(x + cw / 2 - 2, yOwn + 19, `r${owner[i]}`, 'cell', 'middle');
    const f = c.ranks[0].featureIndices[i];
    rect(x, yFeat, cw - 4, ch, f >= 0 ? 'acc1' : 'ghost', 4);
    text(x + cw / 2 - 2, yFeat + 19, f, 'cell', 'middle');
    rect(x, yPos, cw - 4, ch, 'ghost', 4);
    text(x + cw / 2 - 2, yPos + 19, c.fullPos[0][i], 'cell', 'middle');
  });
  const cutX = x0 + c.layout.cu[1] * cw - 2;
  line(cutX, yIds - 12, cutX, yPos + ch + 6, 'cut');

  // ---------- ② 两条 rank 泳道 ----------
  const ly = 292;
  rect(24, ly, 1132, 250, 'panel', 10);
  text(40, ly + 24, '② 每个 rank：local_indices 取首尾两段 → 本地 ids 校验视觉 mask → 按 local_feature_indices 取行', 'pt');
  const lcw = 54;
  const lx0 = 200;
  c.ranks.forEach((r, k) => {
    const by = ly + 42 + k * 102;
    text(40, by + 14, `rank ${r.rank}`, 'pt');
    const rowsDef = [
      ['local_indices', r.localIndices, () => 'ghost'],
      ['本地 ids', r.tokens, (v) => (v === cfg.ids.image ? 'acc1' : v === cfg.cp.padToken ? 'ghost' : 'neutral')],
      ['local_feature', r.localFeature, (v) => (v >= 0 ? 'acc1' : 'ghost')],
    ];
    rowsDef.forEach(([label, vals, cls], j) => {
      const y = by + j * 31;
      text(100, y + 19, label, 'sm');
      vals.forEach((v, i) => {
        const x = lx0 + i * lcw;
        rect(x, y, lcw - 4, 26, cls(v), 4);
        text(x + lcw / 2 - 2, y + 18, v, 'cell', 'middle');
      });
    });
    const nx = lx0 + 8 * lcw + 16;
    rect(nx, by, 1140 - nx, 92, r.rowsUsed.length === c.numFeatures ? 'neutral' : 'acc1', 6);
    text(nx + 12, by + 22, 'mask 与 本地 ids == 10 逐位相等', 'tx');
    text(nx + 12, by + 44, `注入 feature 行 {${r.rowsUsed.join(', ')}}`, 'tx');
    text(nx + 12, by + 64, `本 rank ViT 算出 ${c.numFeatures} 行，只用 ${r.rowsUsed.length} 行`, 'sm');
    text(nx + 12, by + 82, `本 rank 的 T 位置 ${r.localT.join(' ')}（Megatron 按同两段取，依赖侧）`, 'sm');
  });

  // ---------- ③ 代价与失败边界 ----------
  const cy = 556;
  rect(24, cy, 560, 138, 'panel', 10);
  text(40, cy + 24, '③ 代价：每个 CP rank 都对全部图像跑 ViT', 'pt');
  rect(40, cy + 38, 528, 88, 'acc2', 6);
  text(52, cy + 60, `视觉塔输出 ${c.numFeatures} 行（每图 1·4·4 个 patch，merge 2² 后 ${c.featuresPerImage} 行；数量由断言核对）`, 'tx');
  text(52, cy + 82, `CP ${cfg.cp.size} 个 rank 共算 ${c.computedRows} 行、注入 ${c.injectedRows} 行；ViT 可训练时反向也各走一遍`, 'tx');
  text(52, cy + 104, `另加 all_gather：每 rank 发送 ${c.layout.ranks[0].tokens.length} 个 token id，位置 ids 在完整序列上算`, 'sm');

  rect(596, cy, 560, 138, 'panel', 10);
  text(612, cy + 24, '④ 失败边界：CP=1 走同一个整除检查', 'pt');
  rect(612, cy + 38, 528, 88, 'acc2', 6);
  text(624, cy + 60, `cu_seqlens = [${c.cp1Cu.join(', ')}]：7、7 不补齐，尾部 pad 段 ${c.cp1Pad}`, 'tx');
  text(624, cy + 82, `第一段 7 % (2 × 1) ≠ 0 → ValueError（geo3k TP2/128：[${c.geo3kCp1Cu.join(', ')}]）`, 'tx');
  text(624, cy + 104, `“${c.cp1Error}”`, 'sm');

  text(24, 712, '阅读顺序：① 完整位置上的 feature 编号 → ② 各 rank 只注入自己两段里的视觉行 → ③ 重复的 ViT 计算 → ④ CP=1 的奇数长度被拒绝。', 'cap');
  o.push('</svg>');
  return o.join('\n');
}

export const OUTPUTS = Object.freeze({
  positions: 'slime_vlm_mrope_positions.svg',
  cpInjection: 'slime_vlm_mrope_cp_injection.svg',
});

export function renderAll(m = model()) {
  return { [OUTPUTS.positions]: renderPositions(m), [OUTPUTS.cpInjection]: renderCpInjection(m) };
}

const here = dirname(fileURLToPath(import.meta.url));
const defaultOutput = join(here, '..', '..', '..', 'wiki', '02_engineering', '04_posttrain_frameworks', 'slime', 'assets');
const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  const outputDir = process.argv[2] ? process.argv[2] : defaultOutput;
  mkdirSync(outputDir, { recursive: true });
  for (const [name, svg] of Object.entries(renderAll())) writeFileSync(join(outputDir, name), `${svg}\n`, 'utf8');
}
