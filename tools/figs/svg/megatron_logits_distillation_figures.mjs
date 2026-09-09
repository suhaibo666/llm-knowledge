// 38_megatron_logits_distillation_analysis.md 的两张图。
//
// 图 1：**布局图** —— 教师与学生 DP 度不同时，学生 rank 从哪几份存档、按什么步长取哪些 microbatch，
//       以及重建出的顺序为什么等于学生 DP 下的轮转序。逐句复刻冻结基线
//       megatron/training/distillation/cached_logits_loss.py::_compute_dp_remapping（相等 / 升配 /
//       降配三支与两条 ValueError）、TeacherTarDataset._slice_microbatches（`[sub_rank::dp_ratio]`
//       与 num_mb 整除检查）、TeacherTarDataset._interleave_microbatches（按 mb 序、源序交错），
//       以及 utils_logits.py::detect_saved_dp_size（文件名里 max(dp)+1）。
// 图 2：**变换图** —— 一个 token 上 producer 与 consumer 各自的数值变换。producer 复刻
//       logits_saver.py::LogitsSaverHooks._process_single_microbatch（fp32、局部 top-K、
//       logsumexp 的 MAX / SUM 两次 all_reduce、logprob = logit − global_lse）、
//       ::_compute_global_topk（gather 到 tp0、按 fp32 logit 排序取全局 top-K）、
//       ::_apply_topp_truncation（cumprobs − probs < p 的 nucleus 规则 + min_k）与
//       utils_logits.py::pack_indices / unpack_indices（uint16 低 16 位 + bool 第 17 位）；
//       consumer 复刻 cached_logits_loss.py::topk_kl_div（MAX / SUM 两次 all_reduce、offset 映射、
//       mask、ghost 残差跨 TP 求和且只在 tp0 计入、各 rank 局部贡献）与
//       LossFuncCallable.__call__ 的 TP all_reduce 求和 + α 加权；再用同一数据算完整词表 KL 与
//       不带 ghost 的稀疏和作对照；最后按 V = 2^17、K = 64 算每 token 的原始 payload 字节账。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
//
// 图 1 要回答「落盘顺序是确定的轮转，那么 DP 变了以后每个学生 rank 到底读什么、怎么拼回来」。
//   算例 A：G = 8 个全局 microbatch（global_batch_size / micro_batch_size 两次运行相同），
//   全局 microbatch g 落到 saved rank g mod dp_saved。
//   面板 A：教师 dp_saved = 2 的轮转落盘——8 格按 saved rank 着色（r0 浅 / r1 深），两条 tar 行
//     cp0_dp0__I.tar = [0,2,4,6]、cp0_dp1__I.tar = [1,3,5,7]，一个 iteration 一个 member。
//   面板 B：升配 dp = 4（dp_ratio = 2）——每个学生 rank 一行：源存档 dp_rank mod 2 的四格，
//     被 [sub_rank::2] 跨取到的两格用 acc1，其余 ghost；主箭头指向重建的两格；右侧标「✓ 轮转」。
//   面板 C：降配 dp = 2——每个学生 rank 读 dp_rank 与 dp_rank+2 两份存档（各两格），
//     _interleave 按 mb 序交错成四格（acc1 描边）；右侧标「✓ 轮转」。
//   面板 D：整除被破坏的两个例子（acc2）：G = 6 → num_mb = 3、dp_ratio = 2 → _slice_microbatches
//     ValueError；dp_saved = 2 → dp = 3 → _compute_dp_remapping ValueError；以及找不到 tar 时
//     的恒等回退。
//   底部盒子：两条整除条件 + 不变量（重建序 = 学生 DP 的轮转序；哈希校验查不出这一层）。
//
// 图 2 要回答「存下来的是什么、学生怎么用它算出一个 KL，以及 ghost token 保住了什么、丢了什么」。
//   算例 B：V = 16、TP = 2（每 shard 8 个词）、K = 4、一个 token；教师 logits 16 个数使全局 top-4
//   的索引依次为 1、9、3、14；学生 logits 16 个数脚本写死。
//   面板 A（producer）：16 格教师 logits（两 shard，top-4 格 acc1）→ 各 shard 局部 top-4 与局部
//     lse → global_lse → gather 到 tp0 的全局 top-4（idx / logprob）→ 两条小带：indices_low
//     （uint16）与 bit_17（bool），并算一个 idx = 100000 的拆分示例；右侧写 3 B/index 对 int32 的节省。
//   面板 B（consumer）：tp0 / tp1 两行学生 logits（命中教师 top-K 的格 acc1），每行三列：
//     局部 max / Σexp → TP MAX / SUM；offset 映射与命中集合；ghost 残差与本 rank 贡献。
//     底部三格：wrapper all_reduce 求和得总 KL（acc1）；(1−α)·LM + α·KD；对照——完整词表 KL 与
//     无 ghost 的稀疏和（acc2）。
//   面板 C（字节账）：V = 131072、K = 64、值 2 B：稠密 V×2、top-K + int32、top-K + 17 位；
//     压缩比、索引节省与合计节省。
//
// 用法：node tools/figs/svg/megatron_logits_distillation_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// ============================================================================
// 共用算例
// ============================================================================

// 算例 A：DP 重映射
const REMAP_UP = Object.freeze({ G: 8, dpSaved: 2, dp: 4 });
const REMAP_DOWN = Object.freeze({ G: 8, dpSaved: 4, dp: 2 });
const REMAP_BAD_MB = Object.freeze({ G: 6, dpSaved: 2, dp: 4 });
const REMAP_BAD_WORLD = Object.freeze({ G: 8, dpSaved: 2, dp: 3 });

// 算例 B：稀疏 KL
const KL_CASE = Object.freeze({
  V: 16,
  tp: 2,
  K: 4,
  topP: 0.7,
  minK: 1,
  // 教师 logits：全局 top-4 依次是 idx 1（4.0）、9（3.5）、3（3.0）、14（2.5）
  teacher: Object.freeze([0.2, 4.0, 0.5, 3.0, 0.1, 1.0, 0.3, 0.8, 0.4, 3.5, 0.6, 0.9, 0.2, 1.2, 2.5, 0.7]),
  // 学生 logits：脚本写死；idx 15 是学生独有的高分尾部
  student: Object.freeze([0.5, 3.0, 0.2, 2.0, 0.4, 1.5, 0.1, 0.3, 0.9, 3.8, 0.2, 0.6, 1.1, 0.3, 1.0, 2.2]),
  packDemoIndex: 100000, // 17 位拆分示例（> 65535 才能看见 bit_17）
});

// 字节账：V = 2^17（_MAX_VOCAB_SIZE），K = 64，值 2 B（save_dtype 默认 fp16 / bf16 同宽）
const BYTES_CASE = Object.freeze({ V: 2 ** 17, K: 64, valueBytes: 2 });

// 源码常量
const CACHED_LOGITS_LOGPROB_SENTINEL = -1e3;
const CACHED_LOGITS_INDEX_SENTINEL = -1;
const GHOST_EPS = 1e-8;

// ============================================================================
// 复刻 cached_logits_loss.py::_compute_dp_remapping / _slice_microbatches / _interleave_microbatches
// ============================================================================

// utils_logits.py::detect_saved_dp_size：文件名里出现过的 dp 的 max + 1；没有 tar 时 None
function detectSavedDpSize(dpRanksFound) {
  if (dpRanksFound.length === 0) return null;
  return Math.max(...dpRanksFound) + 1;
}

// 返回值四元组 (source_dp_ranks, sub_rank, dp_ratio, dp_size_saved)，分支顺序与源码一致
function computeDpRemapping(dpSizeSaved, dpRank, dpSize) {
  if (dpSizeSaved === null) return { sourceDpRanks: [dpRank], subRank: 0, dpRatio: 1, dpSizeSaved: dpSize };
  if (dpSizeSaved === dpSize) return { sourceDpRanks: [dpRank], subRank: 0, dpRatio: 1, dpSizeSaved };
  if (dpSizeSaved < dpSize) {
    if (dpSize % dpSizeSaved !== 0) {
      throw new Error(`ValueError: Current DP size (${dpSize}) is not an exact multiple of saved DP size (${dpSizeSaved}).`);
    }
    const dpRatio = Math.floor(dpSize / dpSizeSaved);
    const mappedDpRank = dpRank % dpSizeSaved;
    const subRank = Math.floor(dpRank / dpSizeSaved);
    return { sourceDpRanks: [mappedDpRank], subRank, dpRatio, dpSizeSaved };
  }
  if (dpSizeSaved % dpSize !== 0) {
    throw new Error(`ValueError: Saved DP size (${dpSizeSaved}) is not an exact multiple of current DP size (${dpSize}).`);
  }
  const numSources = Math.floor(dpSizeSaved / dpSize);
  const sourceDpRanks = Array.from({ length: numSources }, (_, i) => dpRank + i * dpSize);
  return { sourceDpRanks, subRank: 0, dpRatio: dpSize / dpSizeSaved, dpSizeSaved };
}

// TeacherTarDataset._slice_microbatches：dp_ratio ≤ 1 原样返回；否则 num_mb 必须整除 dp_ratio
function sliceMicrobatches(list, subRank, dpRatio) {
  if (dpRatio <= 1) return list.slice();
  const numMb = list.length;
  if (numMb % dpRatio !== 0) {
    throw new Error(`ValueError: Saved microbatch count (${numMb}) is not divisible by DP ratio (${dpRatio}).`);
  }
  const out = [];
  for (let i = subRank; i < numMb; i += dpRatio) out.push(list[i]);
  return out;
}

// TeacherTarDataset._interleave_microbatches：src0_mb0, src1_mb0, …, src0_mb1, src1_mb1, …
function interleaveMicrobatches(allLists) {
  const numSources = allLists.length;
  const numMb = allLists[0].length;
  const merged = [];
  for (let mb = 0; mb < numMb; mb += 1) {
    for (let src = 0; src < numSources; src += 1) merged.push(allLists[src][mb]);
  }
  return merged;
}

// docstring 里的落盘模型：saved rank d 持有 d, d + dp_saved, d + 2·dp_saved, …
function roundRobin(G, dp) {
  return Array.from({ length: dp }, (_, d) => {
    const held = [];
    for (let g = d; g < G; g += dp) held.push(g);
    return held;
  });
}

function replayRemap({ G, dpSaved, dp }) {
  const saved = roundRobin(G, dpSaved);
  const expected = roundRobin(G, dp);
  const dpSizeSaved = detectSavedDpSize(saved.map((_, d) => d));
  const ranks = [];
  for (let r = 0; r < dp; r += 1) {
    const map = computeDpRemapping(dpSizeSaved, r, dp);
    let reconstructed;
    if (map.sourceDpRanks.length > 1) {
      reconstructed = interleaveMicrobatches(map.sourceDpRanks.map((s) => saved[s]));
    } else {
      reconstructed = sliceMicrobatches(saved[map.sourceDpRanks[0]], map.subRank, map.dpRatio);
    }
    ranks.push({ rank: r, ...map, reconstructed, expected: expected[r], ok: reconstructed.join(',') === expected[r].join(',') });
  }
  return { G, dpSaved, dp, saved, expected, dpSizeSaved, ranks, numMbSaved: saved[0].length, numMbNew: expected[0].length };
}

function tryRemap(spec) {
  try {
    return { ok: true, result: replayRemap(spec) };
  } catch (err) {
    return { ok: false, error: err.message };
  }
}

const REMAP = Object.freeze({
  up: replayRemap(REMAP_UP),
  down: replayRemap(REMAP_DOWN),
  badMb: tryRemap(REMAP_BAD_MB),
  badWorld: tryRemap(REMAP_BAD_WORLD),
  identity: computeDpRemapping(null, 3, 4),
});

if (!REMAP.up.ranks.every((r) => r.ok) || !REMAP.down.ranks.every((r) => r.ok)) {
  throw new Error('重映射不变量被破坏：重建序 ≠ 学生 DP 的轮转序');
}
if (REMAP.badMb.ok || REMAP.badWorld.ok) throw new Error('整除破坏的例子应当抛 ValueError');

// ============================================================================
// 复刻 logits_saver.py 的 producer 与 cached_logits_loss.py::topk_kl_div
// ============================================================================

const logsumexp = (arr) => {
  const m = Math.max(...arr);
  return m + Math.log(arr.reduce((acc, v) => acc + Math.exp(v - m), 0));
};
const shardOf = (arr, tp, r) => {
  const local = arr.length / tp;
  return arr.slice(r * local, (r + 1) * local);
};
// torch.topk：按值降序（本算例无并列）
const topk = (values, k) =>
  values.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, k);

// LogitsSaverHooks._process_single_microbatch + _compute_global_topk（一个 token）
function teacherProduce(logits, tp, K) {
  const V = logits.length;
  const localV = V / tp;
  if (V > 2 ** 17) throw new Error('AssertionError: Global vocab size exceeds maximum supported (17 bits)');
  const effectiveK = Math.min(K, V);
  const localK = Math.min(effectiveK, localV);
  const perRank = [];
  for (let r = 0; r < tp; r += 1) {
    const shard = shardOf(logits, tp, r);
    const local = topk(shard, localK); // [logit, local idx]
    perRank.push({ rank: r, shard, localTop: local, localLse: logsumexp(shard) });
  }
  // global_lse = max_lse + log(sum_i exp(lse_i − max_lse))：MAX all_reduce 再 SUM all_reduce
  const maxLse = Math.max(...perRank.map((p) => p.localLse));
  const sumExpLse = perRank.reduce((acc, p) => acc + Math.exp(p.localLse - maxLse), 0);
  const globalLse = tp > 1 ? maxLse + Math.log(sumExpLse) : perRank[0].localLse;
  // 每 rank：local_logprob = local_logit − global_lse；global_indices = local + tp_rank × local_vocab
  const candidates = [];
  perRank.forEach((p) => {
    p.candidates = p.localTop.map(([logit, li]) => ({ logit, logprob: logit - globalLse, index: li + p.rank * localV }));
    candidates.push(...p.candidates);
  });
  // gather 到 tp0 后按 fp32 logit 排序取全局 top-K
  const chosen = tp > 1 ? topk(candidates.map((c) => c.logit), effectiveK).map(([, pos]) => candidates[pos]) : perRank[0].candidates;
  return {
    V, localV, effectiveK, localK, perRank, maxLse, sumExpLse, globalLse,
    values: chosen.map((c) => c.logprob),
    indices: chosen.map((c) => c.index),
  };
}

// LogitsSaverHooks._apply_topp_truncation：keep_i ⇔ (cumprobs − probs)_i < p，再 OR 上 arange < min_k
function applyToppTruncation(values, indices, p, minK) {
  const probs = values.map((v) => Math.exp(v));
  let cum = 0;
  const keep = probs.map((pr, i) => {
    const before = cum;
    cum += pr;
    return before < p || i < Math.min(minK, values.length);
  });
  const kept = keep.filter(Boolean).length;
  const maxKept = kept; // 单 token：max 就是它自己
  return {
    keptPerToken: kept,
    values: values.slice(0, maxKept).map((v, i) => (keep[i] ? v : CACHED_LOGITS_LOGPROB_SENTINEL)),
    indices: indices.slice(0, maxKept).map((v, i) => (keep[i] ? v : CACHED_LOGITS_INDEX_SENTINEL)),
    cumprobs: probs.map((_, i) => probs.slice(0, i + 1).reduce((a, b) => a + b, 0)),
  };
}

// utils_logits.py::pack_indices / unpack_indices
const packIndices = (indices) => ({
  low: indices.map((i) => i & 0xffff),
  bit17: indices.map((i) => (i >> 16) !== 0),
});
const unpackIndices = (low, bit17) => low.map((l, i) => ((bit17[i] ? 1 : 0) << 16) | l);

// cached_logits_loss.py::topk_kl_div（一个 token，逐 rank 返回局部贡献与中间量）
function topkKlDiv(studentLogits, teacherVals, teacherIdx, tp, addGhost) {
  const localV = studentLogits.length / tp;
  const ranks = [];
  for (let r = 0; r < tp; r += 1) {
    const shard = shardOf(studentLogits, tp, r);
    ranks.push({ rank: r, shard, localMax: Math.max(...shard) });
  }
  const logitsMax = Math.max(...ranks.map((x) => x.localMax)); // TP MAX all_reduce
  ranks.forEach((x) => {
    x.shifted = x.shard.map((v) => v - logitsMax);
    x.localSumExp = x.shifted.reduce((acc, v) => acc + Math.exp(v), 0);
  });
  const sumExp = ranks.reduce((acc, x) => acc + x.localSumExp, 0); // TP SUM all_reduce（dist_nn，可微）
  ranks.forEach((x) => {
    x.logprobs = x.shifted.map((v) => v - Math.log(sumExp));
    const offset = localV * x.rank;
    x.offset = offset;
    x.mask = teacherIdx.map((i, k) => i >= offset && i < offset + localV && teacherVals[k] !== CACHED_LOGITS_LOGPROB_SENTINEL);
    x.localIdx = teacherIdx.map((i) => Math.min(Math.max(i - offset, 0), localV - 1));
    x.gathered = x.localIdx.map((li) => x.logprobs[li]);
    x.hits = teacherIdx.filter((_, k) => x.mask[k]);
    x.hitExpSum = x.gathered.reduce((acc, g, k) => acc + (x.mask[k] ? Math.exp(g) : 0), 0);
  });
  let studentResidual = null;
  let teacherResidual = null;
  if (addGhost) {
    const studentTopkExpSum = ranks.reduce((acc, x) => acc + x.hitExpSum, 0); // TP SUM all_reduce
    studentResidual = Math.log(Math.max(1 - studentTopkExpSum, GHOST_EPS));
    teacherResidual = Math.log(Math.max(1 - teacherVals.reduce((acc, v) => acc + Math.exp(v), 0), GHOST_EPS));
  }
  ranks.forEach((x) => {
    const t = addGhost ? [...teacherVals, teacherResidual] : teacherVals;
    const s = addGhost ? [...x.gathered, studentResidual] : x.gathered;
    const m = addGhost ? [...x.mask, x.rank === 0] : x.mask;
    x.klTerms = t.map((tv, k) => (m[k] ? Math.exp(tv) * (tv - s[k]) : 0));
    x.contribution = x.klTerms.reduce((a, b) => a + b, 0);
  });
  return {
    ranks, logitsMax, sumExp, studentResidual, teacherResidual,
    total: ranks.reduce((acc, x) => acc + x.contribution, 0), // LossFuncCallable：dist.all_reduce(SUM, TP)
  };
}

// 对照：完整词表 KL(p_T ‖ p_S)
function fullKl(teacherLogits, studentLogits) {
  const tl = logsumexp(teacherLogits);
  const sl = logsumexp(studentLogits);
  return teacherLogits.reduce((acc, t, i) => {
    const lt = t - tl;
    return acc + Math.exp(lt) * (lt - (studentLogits[i] - sl));
  }, 0);
}

function runKl(caseSpec) {
  const producer = teacherProduce(caseSpec.teacher, caseSpec.tp, caseSpec.K);
  const packed = packIndices(producer.indices);
  const demoPacked = packIndices([caseSpec.packDemoIndex]);
  const topp = applyToppTruncation(producer.values, producer.indices, caseSpec.topP, caseSpec.minK);
  const ghost = topkKlDiv(caseSpec.student, producer.values, producer.indices, caseSpec.tp, true);
  const noGhost = topkKlDiv(caseSpec.student, producer.values, producer.indices, caseSpec.tp, false);
  const full = fullKl(caseSpec.teacher, caseSpec.student);
  const teacherTopMass = producer.values.reduce((acc, v) => acc + Math.exp(v), 0);
  return {
    producer, packed, demoPacked, demoUnpacked: unpackIndices(demoPacked.low, demoPacked.bit17)[0], topp,
    ghost, noGhost, full, teacherTopMass, teacherTailMass: 1 - teacherTopMass,
    lostTail: full - ghost.total,
  };
}

const KL = runKl(KL_CASE);
if (KL.demoUnpacked !== KL_CASE.packDemoIndex) throw new Error('pack/unpack 往返失败');
if (!(KL.ghost.total <= KL.full + 1e-12)) throw new Error('稀疏 + ghost 的 KL 必须 ≤ 完整词表 KL（粗化不增加散度）');

// 字节账：每 token 原始 payload
function byteLedger({ V, K, valueBytes }) {
  const dense = V * valueBytes;
  const topkInt32 = K * (valueBytes + 4);
  const topk17 = K * (valueBytes + 2 + 1);
  return {
    dense, topkInt32, topk17,
    ratio: dense / topk17,
    indexSaving: 1 - 3 / 4,
    totalSaving: 1 - topk17 / topkInt32,
  };
}
const BYTES = byteLedger(BYTES_CASE);

const f4 = (x) => x.toFixed(4);
const f2 = (x) => x.toFixed(2);
const pct1 = (x) => `${(x * 100).toFixed(1)}%`;
const pct0 = (x) => `${Math.round(x * 100)}%`;
const intl = (x) => x.toLocaleString('en-US');

// ============================================================================
// SVG 基础设施（与 tools/figs/svg/megatron_fsdp_figures.mjs 同一套 token）
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
    layoutFail(`${where}: "${value}" 需要 ${width.toFixed(1)}px，超出 ${limit}px`);
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
  .outline1{fill:none;stroke:#2563EB;stroke-width:1.8}
  .outline2{fill:none;stroke:#C3651F;stroke-width:1.8}
  .r0{fill:#EEF1F5;stroke:#AEB6C2;stroke-width:1}
  .r1{fill:#D8DEE8;stroke:#AEB6C2;stroke-width:1}
  .r2{fill:#C2CBDA;stroke:#AEB6C2;stroke-width:1}
  .r3{fill:#ACB8CB;stroke:#AEB6C2;stroke-width:1}
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

const REPORT_ALL = Boolean(process.env.FIG_REPORT_ALL);
const layoutFailures = [];
function layoutFail(message) {
  if (REPORT_ALL) layoutFailures.push(message);
  else throw new Error(message);
}

function assertNoTextOverlap(svg, name) {
  const m = svg.match(/viewBox="0 0 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)"/);
  const canvasW = Number(m[1]);
  const canvasH = Number(m[2]);
  const boxes = [];
  for (const hit of svg.matchAll(
    /<text class="([a-z0-9]+)" x="(-?[\d.]+)" y="(-?[\d.]+)" text-anchor="(\w+)">([^<]*)<\/text>/g,
  )) {
    const [, cls, xs, ys, anchor, raw] = hit;
    const size = FONT[cls];
    if (!size || raw.trim() === '') continue;
    const w = textWidth(raw.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>'), size);
    const x = anchor === 'middle' ? Number(xs) - w / 2 : anchor === 'end' ? Number(xs) - w : Number(xs);
    boxes.push({ x, y: Number(ys) - size * 0.78, w, h: size * 1.06, raw });
  }
  for (const b of boxes) {
    if (b.x < -1 || b.y < -1 || b.x + b.w > canvasW + 1 || b.y + b.h > canvasH + 1) {
      layoutFail(
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
        layoutFail(`${name}: 文字重叠 "${a.raw}" × "${b.raw}"（重合 ${dx.toFixed(1)}×${dy.toFixed(1)}px）`);
      }
    }
  }
  return svg;
}

const seal = (parts, w, h, name) =>
  assertNoTextOverlap(parts.join('\n').replace(/__H__/g, String(h)) + '\n</svg>', name);

// 一排小格：items = [{label, cls, sub?}]
function cellRow(parts, x, y, items, { w = 40, h = 26, gap = 6 } = {}) {
  items.forEach((it, i) => {
    const cx = x + i * (w + gap);
    parts.push(rect(cx, y, w, h, it.cls, 4));
    if (it.sub !== undefined) {
      parts.push(text(cx + w / 2, y + 11, it.sub, 'sm', 'middle'));
      parts.push(text(cx + w / 2, y + h - 6, it.label, 'tx', 'middle'));
    } else {
      parts.push(text(cx + w / 2, y + h / 2 + 4, it.label, 'tx', 'middle'));
    }
  });
  return x + items.length * (w + gap) - gap;
}

const listStr = (arr) => `[${arr.join(', ')}]`;

// ============================================================================
// 图 1：DP 重映射
// ============================================================================

function renderRemap() {
  const W = 1272;
  const H = 690;
  const parts = header(
    W,
    '图 1 · 落盘是确定的轮转，DP 变了就按步长跨取或交错重建',
    `算例 A：G = ${REMAP_UP.G} 个全局 microbatch；教师 dp_saved = ${REMAP_UP.dpSaved} → 学生 dp = ${REMAP_UP.dp}（升配）、dp_saved = ${REMAP_DOWN.dpSaved} → dp = ${REMAP_DOWN.dp}（降配）；数字由复刻的 _compute_dp_remapping 等三个函数算出`,
  );
  const rankCls = (d) => `r${d}`;

  // ---- 面板 A：教师 dp_saved=2 轮转落盘 ----
  {
    const px = 28;
    const py = 68;
    const pw = 600;
    const ph = 250;
    const up = REMAP.up;
    parts.push(rect(px, py, pw, ph, 'panel', 10));
    parts.push(text(px + 12, py + 22, `A · 教师 dp_saved = ${up.dpSaved}：轮转落盘（G = ${up.G}）`, 'pt'));
    parts.push(text(px + 12, py + 40, `全局 microbatch g 落到 saved rank g mod ${up.dpSaved}；每 rank 每 iteration num_mb = G / dp_saved = ${up.numMbSaved}`, 'sm'));
    parts.push(text(px + 12, py + 74, '全局 g', 'rank'));
    cellRow(parts, px + 80, py + 57, Array.from({ length: up.G }, (_, g) => ({ label: String(g), cls: rankCls(g % up.dpSaved) })), { w: 44, h: 28 });
    parts.push(text(px + 500, py + 74, `浅 = dp0，深 = dp1`, 'sm'));
    up.saved.forEach((held, d) => {
      const y = py + 108 + d * 44;
      parts.push(text(px + 12, y + 18, `cp0_dp${d}__I.tar`, 'dim'));
      const end = cellRow(parts, px + 150, y, held.map((g) => ({ label: String(g), cls: rankCls(d) })), { w: 44, h: 28 });
      parts.push(text(end + 14, y + 18, `→ member I.pt.zst（${held.length} 个 mb）`, 'sm'));
    });
    parts.push(text(px + 12, py + 210, 'tar 名 cp{C}_dp{D}__{I}.tar 只编码 cp / dp / 末 iteration；一个 tar 攒多个 iteration', 'sm'));
    parts.push(text(px + 12, py + 228, `detect_saved_dp_size = max(dp) + 1 = ${up.dpSizeSaved}；同一 cp-dp 的 tar 按末 iteration 数值排序`, 'sm'));
  }

  // ---- 面板 B：升配 ----
  {
    const px = 644;
    const py = 68;
    const pw = 600;
    const ph = 250;
    const up = REMAP.up;
    parts.push(rect(px, py, pw, ph, 'panel', 10));
    parts.push(text(px + 12, py + 22, `B · 升配 dp = ${up.dp}（dp_ratio = ${up.ranks[0].dpRatio}）：一份存档被 ${up.ranks[0].dpRatio} 个 rank 按步长跨取`, 'pt'));
    parts.push(text(px + 12, py + 40, `_compute_dp_remapping → ([dp_rank mod ${up.dpSaved}], sub_rank = dp_rank div ${up.dpSaved}, ${up.ranks[0].dpRatio}, ${up.dpSaved})；切片 [sub_rank::${up.ranks[0].dpRatio}]`, 'sm'));
    up.ranks.forEach((rk) => {
      const y = py + 58 + rk.rank * 42;
      const src = rk.sourceDpRanks[0];
      parts.push(text(px + 12, y + 18, `student dp${rk.rank}`, 'rank'));
      parts.push(text(px + 100, y + 18, `← dp${src}`, 'sm'));
      const held = up.saved[src];
      const end = cellRow(parts, px + 150, y, held.map((g, i) => ({ label: String(g), cls: i % rk.dpRatio === rk.subRank ? 'acc1' : 'ghost' })), { w: 38, h: 28, gap: 4 });
      parts.push(line(end + 8, y + 14, end + 48, y + 14, 'main'));
      parts.push(text(end + 28, y + 9, `[${rk.subRank}::${rk.dpRatio}]`, 'dim', 'middle'));
      const end2 = cellRow(parts, end + 56, y, rk.reconstructed.map((g) => ({ label: String(g), cls: rankCls(rk.rank) })), { w: 38, h: 28, gap: 4 });
      parts.push(text(end2 + 12, y + 18, `${rk.ok ? '✓' : '✗'} 轮转 dp${rk.rank} = ${listStr(rk.expected)}`, 'sm'));
    });
    parts.push(text(px + 12, py + 238, `每 iteration 每 rank：${up.numMbSaved} → ${up.numMbNew} 个 mb；dp_ratio 既是共享数也是 stride`, 'sm'));
  }

  // ---- 面板 C：降配 ----
  {
    const px = 28;
    const py = 334;
    const pw = 600;
    const ph = 230;
    const down = REMAP.down;
    parts.push(rect(px, py, pw, ph, 'panel', 10));
    parts.push(text(px + 12, py + 22, `C · 降配 dp = ${down.dp}：每 rank 读 ${down.ranks[0].sourceDpRanks.length} 份存档，按 mb 序交错重建轮转`, 'pt'));
    parts.push(text(px + 12, py + 40, `_compute_dp_remapping → ([dp_rank, dp_rank + ${down.dp}], 0, ${down.ranks[0].dpRatio}, ${down.dpSaved})；dp_ratio < 1 只作信息`, 'sm'));
    down.ranks.forEach((rk) => {
      const y0 = py + 60 + rk.rank * 76;
      parts.push(text(px + 12, y0 + 34, `student dp${rk.rank}`, 'rank'));
      rk.sourceDpRanks.forEach((src, si) => {
        const y = y0 + si * 34;
        parts.push(text(px + 110, y + 18, `dp${src}`, 'sm'));
        cellRow(parts, px + 150, y, down.saved[src].map((g) => ({ label: String(g), cls: rankCls(src) })), { w: 38, h: 28, gap: 4 });
      });
      parts.push(line(px + 240, y0 + 30, px + 292, y0 + 30, 'main'));
      parts.push(text(px + 266, y0 + 22, '_interleave', 'dim', 'middle'));
      const end = cellRow(parts, px + 300, y0 + 16, rk.reconstructed.map((g) => ({ label: String(g), cls: 'acc1' })), { w: 38, h: 28, gap: 4 });
      parts.push(text(end + 12, y0 + 34, `${rk.ok ? '✓' : '✗'} 轮转 dp${rk.rank}`, 'sm'));
    });
    parts.push(text(px + 12, py + 216, '同一 group 内各源的 iteration 必须相同，否则 _interleave_decoded_group RuntimeError', 'sm'));
  }

  // ---- 面板 D：整除破坏 ----
  {
    const px = 644;
    const py = 334;
    const pw = 600;
    const ph = 230;
    parts.push(rect(px, py, pw, ph, 'panel', 10));
    parts.push(text(px + 12, py + 22, 'D · 两层整除条件被破坏时', 'pt'));
    const bad = REMAP_BAD_MB;
    const numMb = bad.G / bad.dpSaved;
    const ratio = bad.dp / bad.dpSaved;
    parts.push(rect(px + 12, py + 36, pw - 24, 76, 'acc2', 6));
    parts.push(text(px + 24, py + 56, `G = ${bad.G}，dp_saved = ${bad.dpSaved} → num_mb = ${numMb}；学生 dp = ${bad.dp} → dp_ratio = ${ratio}`, 'tx'));
    cellRow(parts, px + 24, py + 66, Array.from({ length: numMb }, (_, i) => ({ label: String(i * bad.dpSaved), cls: 'acc2' })), { w: 38, h: 28, gap: 4 });
    parts.push(text(px + 160, py + 84, `${numMb} mod ${ratio} ≠ 0 → _slice_microbatches ValueError（切片前）`, 'costtx'));
    parts.push(text(px + 24, py + 106, guard(REMAP.badMb.error.replace('ValueError: ', ''), 10.5, pw - 48, 'D/badMb'), 'sm'));
    const badW = REMAP_BAD_WORLD;
    parts.push(rect(px + 12, py + 122, pw - 24, 50, 'acc2', 6));
    parts.push(text(px + 24, py + 142, `dp_saved = ${badW.dpSaved} → 学生 dp = ${badW.dp}：${badW.dp} mod ${badW.dpSaved} ≠ 0 → _compute_dp_remapping ValueError（构造时）`, 'tx'));
    parts.push(text(px + 24, py + 162, guard(REMAP.badWorld.error.replace('ValueError: ', ''), 10.5, pw - 48, 'D/badWorld'), 'sm'));
    parts.push(text(px + 12, py + 196, `目录里没有 tar：detect_saved_dp_size = None → 恒等 ([dp_rank], 0, 1, dp_size)，例 dp_rank 3 → ${listStr(REMAP.identity.sourceDpRanks)}`, 'sm'));
    parts.push(text(px + 12, py + 214, '随后 _shard_groups 找不到任何 shard → FileNotFoundError', 'sm'));
  }

  // ---- 底部规则盒 ----
  parts.push(infoBox(28, 580, 1216, 96, '整除条件（两条）与本图复刻的不变量', [
    `① DP 世界大小整除：升配要求 dp mod dp_saved = 0，降配要求 dp_saved mod dp = 0（_compute_dp_remapping，两条 ValueError）`,
    `② 升配时每个存档 iteration 内的 num_mb mod dp_ratio = 0（_slice_microbatches ValueError）；降配走交错路径，不查这一条`,
    { text: `不变量：重建后的 microbatch 序 = 学生 DP 下的轮转序（本图 ${REMAP.up.dp + REMAP.down.dp} 个 rank 逐个核对 ✓）；_meta.json 的哈希只管样本流身份，查不出这一层`, cls: 'dim' },
  ], 'neutral', 'rules'));

  return seal(parts, W, H, 'megatron_logits_distillation_remap.svg');
}

// ============================================================================
// 图 2：稀疏 KL
// ============================================================================

function renderKl() {
  const W = 1272;
  const H = 786;
  const c = KL_CASE;
  const P = KL.producer;
  const parts = header(
    W,
    '图 2 · 存的是跨 TP 的全局 top-K，算的是 TP 感知的稀疏 KL',
    `算例 B：V = ${c.V}，TP = ${c.tp}（每 shard ${P.localV} 个词），K = ${c.K}，一个 token；教师 / 学生 logits 脚本写死，其余数字由复刻的 _process_single_microbatch、pack_indices、topk_kl_div 算出`,
  );
  const topSet = new Set(P.indices);

  // ---- 面板 A：producer ----
  {
    const px = 28;
    const py = 68;
    const pw = 1216;
    const ph = 262;
    parts.push(rect(px, py, pw, ph, 'panel', 10));
    parts.push(text(px + 12, py + 22, 'A · producer（教师）：局部 top-K → 跨 TP 的 logsumexp → gather 到 tp0 取全局 top-K → 17 位打包', 'pt'));
    parts.push(text(px + 12, py + 40, `局部 top-K 在 fp32 logit 上选；global_lse = max_lse + log Σ exp(lse_r − max_lse)（MAX 再 SUM 两次 all_reduce）；logprob = logit − global_lse，只算 top-K 位置`, 'sm'));
    const cw = 52;
    const cg = 4;
    const x0 = px + 110;
    parts.push(text(px + 12, py + 82, '教师 logits', 'rank'));
    for (let r = 0; r < c.tp; r += 1) {
      parts.push(text(x0 + (r * P.localV + P.localV / 2) * (cw + cg) - cg / 2, py + 60, `tp${r} shard [${r * P.localV}, ${(r + 1) * P.localV})`, 'dim', 'middle'));
    }
    cellRow(parts, x0, py + 66, c.teacher.map((v, i) => ({ label: v.toFixed(1), sub: String(i), cls: topSet.has(i) ? 'acc1' : 'neutral' })), { w: cw, h: 34, gap: cg });
    const sepX = x0 + P.localV * (cw + cg) - cg / 2;
    parts.push(line(sepX, py + 62, sepX, py + 104, 'sep'));
    // 局部 top-K
    parts.push(text(px + 12, py + 140, `局部 top-${P.localK}`, 'rank'));
    P.perRank.forEach((pr) => {
      const x = x0 + pr.rank * P.localV * (cw + cg);
      const end = cellRow(parts, x, py + 122, pr.candidates.map((cd) => ({ label: cd.logprob.toFixed(3), sub: `idx ${cd.index}`, cls: 'neutral' })), { w: cw, h: 34, gap: cg });
      parts.push(text(end + 10, py + 134, `lse${pr.rank} = ${f4(pr.localLse)}`, 'sm'));
      parts.push(text(end + 10, py + 150, `idx = local + ${pr.rank}·${P.localV}`, 'sm'));
    });
    parts.push(text(px + 1010, py + 134, `max_lse = ${f4(P.maxLse)}`, 'sm'));
    parts.push(text(px + 1010, py + 150, `global_lse = ${f4(P.globalLse)}`, 'dim'));
    // 全局 top-K
    parts.push(text(px + 12, py + 196, `全局 top-${P.effectiveK} → tp0`, 'rank'));
    const endG = cellRow(parts, x0, py + 178, P.indices.map((idx, k) => ({ label: P.values[k].toFixed(3), sub: `idx ${idx}`, cls: 'acc1' })), { w: cw, h: 34, gap: cg });
    parts.push(text(endG + 10, py + 190, `gather 到 tp0 后按 fp32 logit 排序；Σ p_T(top-K) = ${f4(KL.teacherTopMass)}`, 'sm'));
    parts.push(text(endG + 10, py + 206, `top-P = ${c.topP}、min_k = ${c.minK} 时 keep ⇔ cumprobs − probs < p → 保留 ${KL.topp.keptPerToken} 项`, 'sm'));
    // 17 位打包两条带
    const bx = px + 700;
    parts.push(text(bx, py + 190, 'indices_low (uint16)', 'dim'));
    const endL = cellRow(parts, bx + 130, py + 178, KL.packed.low.map((v) => ({ label: String(v), cls: 'neutral' })), { w: 44, h: 22, gap: 4 });
    parts.push(text(bx, py + 226, 'bit_17 (bool)', 'dim'));
    cellRow(parts, bx + 130, py + 214, KL.packed.bit17.map((v) => ({ label: v ? '1' : '0', cls: 'ghost' })), { w: 44, h: 22, gap: 4 });
    parts.push(text(endL + 12, py + 194, `idx ${intl(c.packDemoIndex)} → low ${intl(KL.demoPacked.low[0])}，bit17 ${KL.demoPacked.bit17[0] ? 1 : 0}`, 'sm'));
    parts.push(text(endL + 12, py + 230, `2 + 1 = 3 B/index，对 int32 省 ${pct0(BYTES.indexSaving)}`, 'costtx'));
    parts.push(text(px + 12, py + 250, `values 转 save_dtype（默认 fp16）后连同两条带经 torch.save 进 _pending_writes[iteration]；V ≤ 2^17 由 assert 守住`, 'sm'));
  }

  // ---- 面板 B：consumer ----
  {
    const px = 28;
    const py = 346;
    const pw = 1216;
    const ph = 302;
    const G = KL.ghost;
    parts.push(rect(px, py, pw, ph, 'panel', 10));
    parts.push(text(px + 12, py + 22, 'B · consumer（学生）：核内跨 TP 归一化 + 本 shard 的稀疏贡献，wrapper 做 TP 求和与 α 加权', 'pt'));
    parts.push(text(px + 12, py + 40, `两次 all_reduce：logits_max 取 MAX（detach，只为数值稳定）、sum_exp 取 SUM（dist_nn，可微）；ghost 残差的 Σ p_S(hit) 再 SUM 一次`, 'sm'));
    const cw = 40;
    const cg = 4;
    G.ranks.forEach((rk) => {
      const y = py + 62 + rk.rank * 84;
      parts.push(text(px + 12, y + 20, `tp${rk.rank}`, 'rank'));
      parts.push(text(px + 12, y + 36, `学生 logits`, 'sm'));
      const end = cellRow(parts, px + 90, y, rk.shard.map((v, i) => ({ label: v.toFixed(1), sub: String(rk.offset + i), cls: topSet.has(rk.offset + i) ? 'acc1' : 'neutral' })), { w: cw, h: 34, gap: cg });
      const colA = end + 16;
      parts.push(text(colA, y + 12, `local max = ${f2(rk.localMax)} → TP MAX = ${f2(G.logitsMax)}`, 'sm'));
      parts.push(text(colA, y + 28, `local Σexp = ${f4(rk.localSumExp)} → TP SUM = ${f4(G.sumExp)}`, 'sm'));
      parts.push(text(colA, y + 44, `log p_S = logit − max − log SUM`, 'sm'));
      const colB = colA + 250;
      parts.push(text(colB, y + 12, `offset = ${P.localV}·${rk.rank} = ${rk.offset}；命中 idx ${listStr(rk.hits)}`, 'sm'));
      parts.push(text(colB, y + 28, `gather 命中的 log p_S：${rk.gathered.filter((_, k) => rk.mask[k]).map((v) => v.toFixed(3)).join(', ')}`, 'sm'));
      parts.push(text(colB, y + 44, `未命中 → clamp 后 gather，再被 mask 置 0`, 'sm'));
      const colC = colB + 250;
      parts.push(text(colC, y + 12, `Σ p_S(hit) = ${f4(rk.hitExpSum)} → TP SUM ${f4(G.ranks.reduce((a, x) => a + x.hitExpSum, 0))}`, 'sm'));
      parts.push(text(colC, y + 28, rk.rank === 0 ? `ghost 项 = ${f4(G.ranks[0].klTerms[G.ranks[0].klTerms.length - 1])}（只 tp0 计入）` : 'ghost 项 = 0（mask 末位 = tp_rank == 0）', 'sm'));
      parts.push(text(colC, y + 44, `贡献_${rk.rank} = ${f4(rk.contribution)}`, 'dim'));
    });
    const yb = py + 236;
    parts.push(rect(px + 12, yb, 400, 52, 'acc1', 6));
    parts.push(text(px + 24, yb + 20, `wrapper：dist.all_reduce(SUM, TP)`, 'tx'));
    parts.push(text(px + 24, yb + 40, `KL = ${f4(G.ranks[0].contribution)} + (${f4(G.ranks[1].contribution)}) = ${f4(G.total)}`, 'dim'));
    parts.push(rect(px + 428, yb, 360, 52, 'neutral', 6));
    parts.push(text(px + 440, yb + 20, `loss = (1 − α)·LM + α·KD；CLI 默认 α = 1.0`, 'tx'));
    parts.push(text(px + 440, yb + 40, `教师残差 = log(1 − Σ p_T) = ${f4(G.teacherResidual)}，学生残差 = ${f4(G.studentResidual)}`, 'sm'));
    parts.push(rect(px + 804, yb, 400, 52, 'acc2', 6));
    parts.push(text(px + 816, yb + 20, `对照：完整词表 KL = ${f4(KL.full)}，无 ghost 的稀疏和 = ${f4(KL.noGhost.total)}`, 'tx'));
    parts.push(text(px + 816, yb + 40, `ghost 保住尾部总质量 ${f4(KL.teacherTailMass)}，丢掉尾部形状：差 ${f4(KL.lostTail)}`, 'costtx'));
  }

  // ---- 面板 C：字节账 ----
  {
    const px = 28;
    const py = 664;
    const pw = 1216;
    const ph = 110;
    const B = BYTES_CASE;
    parts.push(rect(px, py, pw, ph, 'panel', 10));
    parts.push(text(px + 12, py + 22, `C · 字节账（每 token 原始 payload，不计 tar / zstd）：V = ${intl(B.V)}（2^17 上限），K = ${B.K}，值 ${B.valueBytes} B`, 'pt'));
    parts.push(infoBox(px + 12, py + 34, 300, 64, '被否掉：稠密 logits', [`V × ${B.valueBytes} B = ${intl(BYTES.dense)} B / token`], 'ghost', 'C/dense'));
    parts.push(infoBox(px + 324, py + 34, 320, 64, '被否掉：top-K + int32 索引', [`K × (${B.valueBytes} + 4) = ${intl(BYTES.topkInt32)} B / token`], 'ghost', 'C/int32'));
    parts.push(infoBox(px + 656, py + 34, 548, 64, '现行：top-K + uint16 低位 + bool 第 17 位', [
      { text: `K × (${B.valueBytes} + 2 + 1) = ${intl(BYTES.topk17)} B / token；对稠密 ${BYTES.ratio.toFixed(1)}×，对 int32 合计省 ${pct1(BYTES.totalSaving)}`, cls: 'dim' },
    ], 'acc1', 'C/17bit'));
  }

  return seal(parts, W, H, 'megatron_logits_distillation_kl.svg');
}

// ============================================================================

const outputs = new Map([
  ['megatron_logits_distillation_remap.svg', renderRemap()],
  ['megatron_logits_distillation_kl.svg', renderKl()],
]);

export {
  REMAP_UP, REMAP_DOWN, REMAP_BAD_MB, REMAP_BAD_WORLD, KL_CASE, BYTES_CASE,
  CACHED_LOGITS_LOGPROB_SENTINEL, CACHED_LOGITS_INDEX_SENTINEL,
  REMAP, KL, BYTES,
  detectSavedDpSize, computeDpRemapping, sliceMicrobatches, interleaveMicrobatches, roundRobin, replayRemap,
  logsumexp, teacherProduce, applyToppTruncation, packIndices, unpackIndices, topkKlDiv, fullKl, runKl, byteLedger,
  f4, f2, pct1, pct0, intl, outputs,
};

const invokedDirectly = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (invokedDirectly) {
  if (layoutFailures.length) {
    console.error(layoutFailures.join('\n'));
    process.exit(1);
  }
  const here = dirname(fileURLToPath(import.meta.url));
  const outDir =
    process.argv[2] ??
    join(here, '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks', 'megatron-lm', 'assets');
  mkdirSync(outDir, { recursive: true });
  for (const [name, svg] of outputs) {
    writeFileSync(join(outDir, name), svg, 'utf8');
    console.log(`${name}  ${svg.length} bytes`);
  }
  console.log('\n图 1：', JSON.stringify({
    up: REMAP.up.ranks.map((r) => [r.rank, r.sourceDpRanks, r.subRank, r.dpRatio, r.reconstructed]),
    down: REMAP.down.ranks.map((r) => [r.rank, r.sourceDpRanks, r.reconstructed]),
    badMb: REMAP.badMb.error, badWorld: REMAP.badWorld.error,
  }));
  console.log('图 2：', JSON.stringify({
    indices: KL.producer.indices, values: KL.producer.values.map(f4), globalLse: f4(KL.producer.globalLse),
    packed: KL.packed, demo: KL.demoPacked, topp: KL.topp.keptPerToken,
    contributions: KL.ghost.ranks.map((r) => f4(r.contribution)), total: f4(KL.ghost.total),
    full: f4(KL.full), noGhost: f4(KL.noGhost.total), tail: f4(KL.teacherTailMass), lost: f4(KL.lostTail),
    bytes: BYTES,
  }));
}
