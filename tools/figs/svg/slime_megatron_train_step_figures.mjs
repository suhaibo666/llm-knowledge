// 图：slime Megatron 训练后端怎样把一份 rollout 训练字典变成一次 optimizer step：
// build_dp_schedule 先按 rollout id 组步、first-fit 打包、拆 bin 对齐、再分给 DP rank；
// get_batch 把一个 micro-batch 切成 zigzag CP 片、拼成 THD 流并对齐 next-token mask；
// actor 一轮内按 CPU tag 切换 ref / old_actor / actor 只做前向；train_one_step 跑完全部
// micro-batch 的前反向后做一次 optimizer.step 并以逻辑 rollout 数推进 LR scheduler。
// 源码基线：THUDM/slime@681b3adca54105d5ecd3fb822fa0dc58a427e0f9。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
// 要讲清楚：
//  1. 调度单位是逻辑 rollout：同一 rollout_id 的片段留在同一步；步数 = distinct rollout // global_batch_size。
//  2. 动态打包是 first-fit + 拆最大多样本 bin 对齐到 dp_size × mb_group；静态路径不对齐就直接断言。
//  3. CP 切片是 zigzag 两段：每条样本各自补齐到 2·cp·chunk，rank r 拿第 r 段与第 2cp−1−r 段；
//     mask 左补 prompt_len−1、右补 1 后再同样切片，所以 mask 位置对应"预测下一个 token 的 logit"。
//  4. 一轮训练里 ref/teacher/old_actor 都是同一份 GPU 模型换入 CPU tag 后的前向；
//     单步且无 KL 等条件满足时跳过独立的 old-policy 前向，训练前向的 detached logprob 充当 old logprob。
//  5. optimizer 每个训练步只推进一次，LR scheduler 的 increment 是该步的逻辑 rollout 数。
//
// 布局：四条泳道。A 调度（5 条样本 → 4 个 bin → 2 rank × 2 mb，右侧静态路径的断言）；
// B get_batch（rank0 mb0 两个 CP rank 的 token 流、cu_seqlens、mask 对齐）；
// C 一轮 actor 的 tag 切换与前向计数；D train_one_step 的闭环与缩放入口。
// acc1 标决定性转换与承重结果，acc2 标断言、代价与被拒路径。
//
// 用法：node tools/figs/svg/slime_megatron_train_step_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// ---------------- 冻结的示例输入 ----------------
export const CFG = Object.freeze({
  dpSize: 2,
  cpSize: 2,
  tpSize: 1,
  vppSize: 1,
  mbGroup: 1, // microbatch_group_size_per_vp_stage；vpp_size == 1 时 align_to 不乘它
  useDynamicBatchSize: true,
  maxTokensPerGpu: 9,
  microBatchSize: 2, // 仅静态路径使用
  dataPadSizeMultiplier: 8,
  globalBatchSize: 4, // 单位：逻辑 rollout
  rolloutBatchSize: 2,
  nSamplesPerPrompt: 2,
  promptLen: 4,
  numRollout: 100, // 只用于 LR scheduler 的 train_iters 估算
  klCoef: 0,
  useKlLoss: true, // 让 ref tag 存在但 kl_coef == 0
});

// 五条训练样本：rollout 2 是 compact 扇出的两个片段；s0 第 3 个 response token 是工具 token（mask 0）。
export const SAMPLES = Object.freeze(
  [
    { name: 's0', rolloutId: 0, responseLen: 8, maskedResponseIdx: [3] },
    { name: 's1', rolloutId: 1, responseLen: 2, maskedResponseIdx: [] },
    { name: 's2a', rolloutId: 2, responseLen: 4, maskedResponseIdx: [] },
    { name: 's2b', rolloutId: 2, responseLen: 6, maskedResponseIdx: [] },
    { name: 's3', rolloutId: 3, responseLen: 2, maskedResponseIdx: [] },
  ].map((s) => Object.freeze({
    ...s,
    totalLen: CFG.promptLen + s.responseLen,
    lossMask: Array.from({ length: s.responseLen }, (_, i) => (s.maskedResponseIdx.includes(i) ? 0 : 1)),
  })),
);

// ---------------- slime/utils/seqlen_balancing.py ----------------
export function firstFitPack(totalLengths, maxTokensPerBin) {
  const bins = []; const sums = [];
  totalLengths.forEach((length, idx) => {
    const j = sums.findIndex((s) => s + length <= maxTokensPerBin);
    if (j >= 0) { bins[j].push(idx); sums[j] += length; } else { bins.push([idx]); sums.push(length); }
  });
  return bins;
}

export function splitBinByTokens(binIndices, lengths) {
  const halves = [[], []]; const sums = [0, 0];
  for (const idx of [...binIndices].sort((a, b) => lengths[b] - lengths[a])) {
    const h = sums[0] <= sums[1] ? 0 : 1;
    halves[h].push(idx); sums[h] += lengths[idx];
  }
  return halves;
}

export function expandBinsBySplitting(bins, targetCount, lengths, trace = []) {
  while (bins.length < targetCount) {
    const candidates = bins.map((b, idx) => [b.reduce((a, i) => a + lengths[i], 0), idx]).filter(([, idx]) => bins[idx].length > 1);
    if (!candidates.length) break;
    // Python 的 max 对 (sum, idx) 元组比较：和相同时取 idx 大者
    const [, idx] = candidates.reduce((best, c) => (c[0] > best[0] || (c[0] === best[0] && c[1] > best[1]) ? c : best));
    const [left, right] = splitBinByTokens(bins[idx], lengths);
    trace.push({ splitBin: idx, before: [...bins[idx]], left, right });
    bins[idx] = left; bins.push(right);
  }
}

// ---------------- slime/utils/dp_schedule.py::build_dp_schedule ----------------
export function buildDpSchedule({ cfg = CFG, totalLengths, rolloutIndices, useDynamicBatchSize = cfg.useDynamicBatchSize, balanceByFlops = false, balanceData = balanceByFlops, proxy = null }) {
  // 工作量代理取 calculate_fwd_flops 的形状 aL + bL²；balance_by_flops 在参数校验里会强制 balance_data=True
  const wl = (L) => proxy.a * L + proxy.b * L * L;
  const alignTo = cfg.dpSize * (cfg.vppSize > 1 ? cfg.mbGroup : 1);
  const maxPerBin = useDynamicBatchSize ? cfg.maxTokensPerGpu * cfg.cpSize : null;
  const byRollout = new Map();
  rolloutIndices.forEach((rid, pos) => { if (!byRollout.has(rid)) byRollout.set(rid, []); byRollout.get(rid).push(pos); });
  const rolloutIds = [...byRollout.keys()];
  const numSteps = Math.floor(rolloutIds.length / cfg.globalBatchSize);
  if (numSteps < 1) throw new Error(`AssertionError: num_rollouts (${rolloutIds.length}) < global_batch_size (${cfg.globalBatchSize})`);
  const partitions = Array.from({ length: cfg.dpSize }, () => []);
  const microBatchIndices = Array.from({ length: cfg.dpSize }, () => []);
  const numMicrobatches = []; const globalBatchSizes = []; const steps = [];
  for (let stepI = 0; stepI < numSteps; stepI += 1) {
    const stepRollouts = rolloutIds.slice(stepI * cfg.globalBatchSize, (stepI + 1) * cfg.globalBatchSize);
    const sampleIndices = stepRollouts.flatMap((rid) => byRollout.get(rid));
    const stepLengths = sampleIndices.map((i) => totalLengths[i]);
    globalBatchSizes.push(cfg.globalBatchSize);
    if (sampleIndices.length < cfg.dpSize) throw new Error(`AssertionError: step ${stepI}: ${sampleIndices.length} samples < dp_size ${cfg.dpSize}`);
    let stepMbs;
    if (useDynamicBatchSize && balanceByFlops) {
      const total = stepLengths.reduce((a, b) => a + b, 0);
      const numMbs = Math.max(1, Math.ceil(total / maxPerBin));
      stepMbs = numMbs >= stepLengths.length ? stepLengths.map((_, i) => [i]) : getSeqlenBalancedPartitions(stepLengths.map(wl), numMbs, false);
    } else if (useDynamicBatchSize) stepMbs = firstFitPack(stepLengths, maxPerBin);
    else {
      stepMbs = [];
      for (let i = 0; i < stepLengths.length; i += cfg.microBatchSize) stepMbs.push(Array.from({ length: Math.min(cfg.microBatchSize, stepLengths.length - i) }, (_, k) => i + k));
    }
    const packed = stepMbs.map((b) => [...b]);
    const targetK = Math.max(Math.ceil(stepMbs.length / alignTo) * alignTo, alignTo);
    const splitTrace = [];
    if (targetK !== stepMbs.length) {
      if (useDynamicBatchSize) {
        expandBinsBySplitting(stepMbs, targetK, stepLengths, splitTrace);
        if (stepMbs.length !== targetK) throw new Error(`AssertionError: dynamic path: could only produce ${stepMbs.length} mbs; need ${targetK}`);
      } else {
        throw new Error(`AssertionError: static path: num_mbs (${stepMbs.length}) is not a multiple of dp_size * mb_group (${alignTo}); step_size=${sampleIndices.length}, micro_batch_size=${cfg.microBatchSize}`);
      }
    }
    const K = stepMbs.length; const perRank = K / cfg.dpSize;
    numMicrobatches.push(perRank);
    const rankMbsIdx = balanceData
      ? getSeqlenBalancedPartitions(stepMbs.map((bin) => bin.reduce((acc, i) => acc + wl(stepLengths[i]), 0)), cfg.dpSize, true)
      : Array.from({ length: cfg.dpSize }, (_, r) => Array.from({ length: perRank }, (_, k) => r + k * cfg.dpSize)); // balance_data=False 的跨步轮询
    rankMbsIdx.forEach((mbsList, r) => {
      mbsList.forEach((mbsIdx) => {
        const locals = stepMbs[mbsIdx];
        const localStart = partitions[r].length;
        partitions[r].push(...locals.map((i) => sampleIndices[i]));
        microBatchIndices[r].push(Array.from({ length: locals.length }, (_, k) => localStart + k));
      });
    });
    steps.push({ stepRollouts, sampleIndices, stepLengths, packed, alignTo, targetK, splitTrace, bins: stepMbs.map((b) => b.map((i) => sampleIndices[i])), rankMbsIdx, maxPerBin });
  }
  return { partitions, microBatchIndices, numMicrobatches, globalBatchSizes, steps };
}

// ---------------- slime/backends/megatron_utils/cp_utils.py ----------------
export function sliceWithCp(items, padValue, cpRank, cpSize) {
  if (cpSize === 1) return [...items];
  const chunk = Math.ceil(items.length / (2 * cpSize));
  const padded = [...items, ...Array(2 * cpSize * chunk - items.length).fill(padValue)];
  return [...padded.slice(chunk * cpRank, chunk * (cpRank + 1)), ...padded.slice(chunk * (2 * cpSize - cpRank - 1), chunk * (2 * cpSize - cpRank))];
}

export function logitsTokensOffsetWithCp(totalLength, responseLength, cpRank, cpSize) {
  const promptLength = totalLength - responseLength;
  const chunk = Math.ceil(totalLength / (2 * cpSize));
  const chunk0 = [cpRank * chunk, (cpRank + 1) * chunk];
  const chunk1 = [(2 * cpSize - cpRank - 1) * chunk, (2 * cpSize - cpRank) * chunk];
  let logits0 = [Math.max(chunk0[0], promptLength - 1), Math.min(chunk0[1], totalLength - 1)];
  let logits1 = [Math.max(chunk1[0], promptLength - 1), Math.min(chunk1[1], totalLength - 1)];
  let token0; let token1;
  if (logits0[0] < logits0[1]) token0 = [logits0[0] + 1, logits0[1] + 1]; else { logits0 = [0, 0]; token0 = [0, 0]; }
  if (logits1[0] < logits1[1]) token1 = [logits1[0] + 1, logits1[1] + 1]; else { logits1 = [0, 0]; token1 = [0, 0]; }
  return { chunk, chunks: [chunk0, chunk1], logits: [logits0, logits1], tokens: [token0, token1] };
}

// 本 CP rank 负责的 response 下标（按 tokens_offset 换算到 response 空间）
export function ownedResponseIdx(sample, cpRank, cpSize) {
  if (cpSize === 1) return Array.from({ length: sample.responseLen }, (_, i) => i);
  const { tokens } = logitsTokensOffsetWithCp(sample.totalLen, sample.responseLen, cpRank, cpSize);
  const p = sample.totalLen - sample.responseLen;
  return tokens.flatMap(([a, b]) => (b > a ? Array.from({ length: b - a }, (_, i) => a + i - p) : []));
}

// ---------------- slime/backends/megatron_utils/data.py::get_batch（zigzag 分支） ----------------
export function getBatch(samples, cpRank, cfg = CFG) {
  const padSize = cfg.tpSize * cfg.dataPadSizeMultiplier;
  const PAD = 'P';
  const tokenStreams = samples.map((s) => sliceWithCp(Array.from({ length: s.totalLen }, (_, i) => `${s.name}:${i}`), PAD, cpRank, cfg.cpSize));
  const cuLocal = [0]; tokenStreams.forEach((t) => cuLocal.push(cuLocal.at(-1) + t.length));
  let tokens = tokenStreams.flat();
  const pad = (padSize - (tokens.length % padSize)) % padSize;
  if (pad) { tokens = [...tokens, ...Array(pad).fill(PAD)]; cuLocal.push(cuLocal.at(-1) + pad); }
  const cuSeqlens = cuLocal.map((c) => c * cfg.cpSize);
  const maskStreams = samples.map((s) => {
    const promptLength = s.totalLen - s.responseLen;
    const aligned = [...Array(promptLength - 1).fill(0), ...s.lossMask, 0];
    return sliceWithCp(aligned, 0, cpRank, cfg.cpSize);
  });
  let fullLossMasks = maskStreams.flat();
  if (pad) fullLossMasks = [...fullLossMasks, ...Array(pad).fill(0)];
  if (fullLossMasks.length !== tokens.length) throw new Error('AssertionError: loss_masks.shape != tokens.shape');
  const maxSeqlen = Math.max(...cuSeqlens.slice(1).map((c, i) => c - cuSeqlens[i]));
  const owned = samples.map((s) => ownedResponseIdx(s, cpRank, cfg.cpSize));
  return { tokens, tokenStreams, cuLocal, cuSeqlens, maxSeqlen, pad, padSize, maskStreams, fullLossMasks, owned, maskedValid: fullLossMasks.reduce((a, b) => a + b, 0) };
}

// ---------------- data.py::get_batch（allgather_cp 分支：先整体拼接、补齐，再按 rank 连续等分） ----------------
export function getBatchAllgather(samples, cpRank, cfg = CFG) {
  const padSize = cfg.tpSize * cfg.dataPadSizeMultiplier;
  const PAD = 'P';
  const cu = [0]; samples.forEach((s) => cu.push(cu.at(-1) + s.totalLen));
  let tokens = samples.flatMap((s) => Array.from({ length: s.totalLen }, (_, i) => `${s.name}:${i}`));
  const globalPad = cfg.cpSize * padSize;
  const pad = (globalPad - (tokens.length % globalPad)) % globalPad;
  if (pad) { tokens = [...tokens, ...Array(pad).fill(PAD)]; cu.push(cu.at(-1) + pad); }
  const chunk = tokens.length / cfg.cpSize; // 已补到 cp 的整数倍，torch.chunk 等分
  let masks = samples.flatMap((s) => [...Array(s.totalLen - s.responseLen - 1).fill(0), ...s.lossMask, 0]);
  if (pad) masks = [...masks, ...Array(pad).fill(0)];
  const localTokens = tokens.slice(chunk * cpRank, chunk * (cpRank + 1));
  const localMasks = masks.slice(chunk * cpRank, chunk * (cpRank + 1));
  const maxSeqlen = Math.max(...cu.slice(1).map((c, i) => c - cu[i]));
  return { tokens: localTokens, cuSeqlens: cu, pad, maxSeqlen, fullLossMasks: localMasks, maskedValid: localMasks.reduce((a, b) => a + b, 0), realTokens: localTokens.filter((t) => t !== PAD).length };
}

// ---------------- seqlen_balancing.py::karmarkar_karp / get_seqlen_balanced_partitions（逐行复现 heapq 语义） ----------------
function heappush(h, x, lt) { h.push(x); let i = h.length - 1; while (i > 0) { const p = (i - 1) >> 1; if (lt(h[i], h[p])) { [h[i], h[p]] = [h[p], h[i]]; i = p; } else break; } }
function heappop(h, lt) {
  const last = h.pop(); if (!h.length) return last;
  const top = h[0]; h[0] = last;
  // CPython heapq._siftup：先把空位沿较小子节点下沉到叶，再 _siftdown 回位
  const n = h.length; let pos = 0; const item = h[0]; let child = 1;
  while (child < n) { const right = child + 1; if (right < n && !lt(h[child], h[right])) child = right; h[pos] = h[child]; pos = child; child = 2 * pos + 1; }
  h[pos] = item; let i = pos; while (i > 0) { const p = (i - 1) >> 1; if (lt(h[i], h[p])) { [h[i], h[p]] = [h[p], h[i]]; i = p; } else break; }
  return top;
}
export function karmarkarKarp(seqlenList, k, equalSize) {
  const cmpItems = (a, b) => { for (let i = 0; i < Math.min(a.length, b.length); i += 1) { if (a[i][0] !== b[i][0]) return a[i][0] - b[i][0]; if (a[i][1] !== b[i][1]) return a[i][1] - b[i][1]; } return a.length - b.length; };
  const setLt = (a, b) => (a.sum !== b.sum ? a.sum < b.sum : a.items.length !== b.items.length ? a.items.length < b.items.length : cmpItems(a.items, b.items) < 0);
  const sortDesc = (sets) => sets.sort((x, y) => (setLt(y, x) ? -1 : setLt(x, y) ? 1 : 0));
  const newSet = () => ({ sum: 0, items: [] });
  const makeState = (items) => { const sets = Array.from({ length: k }, newSet); items.forEach(([idx, v], i) => { sets[i].items.push([idx, v]); sets[i].sum += v; }); return { sets: sortDesc(sets) }; };
  const spread = (st) => st.sets[0].sum - st.sets[k - 1].sum;
  const stateLt = (a, b) => (spread(a) !== spread(b) ? spread(a) > spread(b) : setLt(b.sets[0], a.sets[0]));
  const sorted = seqlenList.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const pq = [];
  if (equalSize) {
    if (seqlenList.length % k !== 0) throw new Error('AssertionError: len % k != 0');
    for (let off = 0; off < sorted.length; off += k) heappush(pq, makeState(sorted.slice(off, off + k).map(([v, i]) => [i, v])), stateLt);
  } else sorted.forEach(([v, i]) => heappush(pq, makeState([[i, v]]), stateLt));
  while (pq.length > 1) {
    const s0 = heappop(pq, stateLt); const s1 = heappop(pq, stateLt);
    for (let i = 0; i < k; i += 1) { for (const it of s1.sets[k - 1 - i].items) { s0.sets[i].items.push(it); s0.sets[i].sum += it[1]; } }
    sortDesc(s0.sets); heappush(pq, s0, stateLt);
  }
  return pq[0].sets.map((st) => st.items.map(([idx]) => idx));
}
export function getSeqlenBalancedPartitions(seqlenList, k, equalSize) {
  if (seqlenList.length < k) throw new Error('AssertionError: number of items < k_partitions');
  return karmarkarKarp(seqlenList, k, equalSize).map((part) => [...part].sort((a, b) => a - b));
}
// balance_data=True 的分发：mb 工作量 = Σ 样本 f(L)，f 取 calculate_fwd_flops 的形状 aL + bL²
export function kkDistribution(step, { a, b }, dpSize = CFG.dpSize) {
  const f = (L) => a * L + b * L * L;
  const weights = step.bins.map((bin) => bin.reduce((acc, g) => acc + f(SAMPLES[g].totalLen), 0));
  return getSeqlenBalancedPartitions(weights, dpSize, true);
}

// ---------------- actor 一轮的 tag 切换（slime/backends/megatron_utils/actor.py::train_actor） ----------------
export function roundPlan({ cfg = CFG, numSteps, withRef = cfg.useKlLoss || cfg.klCoef !== 0, withTeacher = false, keepOldActor = false, useCritic = false, useRolloutLogprobs = false, getMismatchMetrics = false, lossType = 'policy_loss', advantageEstimator = 'grpo', useRoutingReplay = false, useRolloutRoutingReplay = false, useOpd = false }) {
  const canReuse = numSteps === 1 && lossType === 'policy_loss' && cfg.klCoef === 0 && !useRolloutLogprobs && !getMismatchMetrics && !useCritic && !keepOldActor && !useOpd && (!useRoutingReplay || useRolloutRoutingReplay) && advantageEstimator !== 'gspo';
  const needOldForward = (!useRolloutLogprobs || getMismatchMetrics) && !canReuse;
  const phases = [];
  if (withRef) phases.push({ tag: 'ref', kind: 'forward_only', store: 'ref_log_probs' });
  if (withTeacher) phases.push({ tag: 'teacher', kind: 'forward_only', store: 'teacher_log_probs' });
  phases.push({ tag: keepOldActor ? 'old_actor' : 'actor', kind: needOldForward ? 'forward_only' : 'switch_only', store: needOldForward ? 'log_probs' : '跳过前向；detach 充当 old' });
  if (useCritic) phases.push({ tag: '(critic actor)', kind: 'external_data', store: 'values' });
  phases.push({ tag: 'actor', kind: 'advantages', store: 'advantages, returns' });
  phases.push({ tag: 'actor', kind: 'train', store: `${numSteps} × train_one_step` });
  phases.push({ tag: 'actor', kind: 'backup', store: 'CPU tag actor ← GPU' });
  const forwardPasses = phases.filter((p) => p.kind === 'forward_only').length;
  // forward_only 与 train 都按训练步各调一次 forward_backward_func，所以两种计数单位不同：
  // 全批前向 = 每个前向阶段覆盖整批一次 + 训练前向覆盖整批一次；流水线调用 = (前向阶段 + 训练) × 步数
  return { canReuse, needOldForward, phases, forwardPasses, fullBatchForwards: forwardPasses + 1, pipelineCalls: (forwardPasses + 1) * numSteps };
}

// ---------------- LR scheduler 的样本计数（model.py::get_optimizer_param_scheduler / train_one_step） ----------------
export function schedulerCounters(cfg = CFG) {
  const trainIters = Math.floor((cfg.numRollout * cfg.rolloutBatchSize * cfg.nSamplesPerPrompt) / cfg.globalBatchSize);
  return { trainIters, lrDecaySteps: trainIters * cfg.globalBatchSize, incrementPerStep: cfg.globalBatchSize };
}

export function model(cfg = CFG) {
  const totalLengths = SAMPLES.map((s) => s.totalLen);
  const rolloutIndices = SAMPLES.map((s) => s.rolloutId);
  const dynamic = buildDpSchedule({ cfg, totalLengths, rolloutIndices });
  let staticError = null;
  try { buildDpSchedule({ cfg, totalLengths, rolloutIndices, useDynamicBatchSize: false }); } catch (e) { staticError = e.message; }
  const rank0Mb0 = dynamic.partitions[0].filter((_, j) => dynamic.microBatchIndices[0][0].includes(j)).map((g) => SAMPLES[g]);
  const batches = Array.from({ length: cfg.cpSize }, (_, cpRank) => getBatch(rank0Mb0, cpRank, cfg));
  const allgather = Array.from({ length: cfg.cpSize }, (_, cpRank) => getBatchAllgather(rank0Mb0, cpRank, cfg));
  const kkProxies = [{ a: 1, b: 0 }, { a: 0, b: 1 }, { a: 1, b: 1 }, { a: 24576, b: 8 }];
  const kk = kkProxies.map((proxy) => ({ ...proxy, partitions: kkDistribution(dynamic.steps[0], proxy, cfg.dpSize) }));
  // balance_by_flops：KK 先按估算 FLOPs 分 ceil(total/cap) 组，再拆 bin 对齐，随后强制的 balance_data 再做一次等大小 KK 分给 rank
  const nm = (b) => `[${b.map((g) => SAMPLES[g].name).join(' ')}]`;
  const flops = kkProxies.map((proxy) => {
    const sch = buildDpSchedule({ cfg, totalLengths, rolloutIndices, balanceByFlops: true, proxy });
    const st = sch.steps[0];
    return {
      ...proxy,
      groups: st.packed.map((b) => b.map((i) => SAMPLES[st.sampleIndices[i]].name).sort().join(',')).sort(),
      numGroups: st.packed.length,
      split: st.splitTrace.map((t) => ({ before: t.before.map((i) => SAMPLES[st.sampleIndices[i]].name), left: t.left.map((i) => SAMPLES[st.sampleIndices[i]].name), right: t.right.map((i) => SAMPLES[st.sampleIndices[i]].name) })),
      rankBins: st.rankMbsIdx.map((idxs) => idxs.map((k) => nm(st.bins[k]))),
      maxBinTokens: Math.max(...st.bins.map((b) => b.reduce((acc, g) => acc + SAMPLES[g].totalLen, 0))),
      regime: proxy.a > 22 * proxy.b ? 'a>22b' : 'a<=22b',
    };
  });
  const round = roundPlan({ cfg, numSteps: dynamic.numMicrobatches.length });
  const roundTwoSteps = roundPlan({ cfg, numSteps: 2 });
  const rankView = dynamic.partitions.map((partition, r) => ({
    rank: r,
    partition,
    mbs: dynamic.microBatchIndices[r].map((locals) => locals.map((j) => SAMPLES[partition[j]].name)),
  }));
  const rolloutMaskSums = (() => {
    const totals = new Map();
    SAMPLES.forEach((s) => totals.set(s.rolloutId, (totals.get(s.rolloutId) ?? 0) + s.lossMask.reduce((a, b) => a + b, 0)));
    return SAMPLES.map((s) => totals.get(s.rolloutId));
  })();
  return { cfg, samples: SAMPLES, dynamic, staticError, rankView, rank0Mb0, batches, allgather, kk, flops, round, roundTwoSteps, scheduler: schedulerCounters(cfg), rolloutMaskSums };
}

// ---------------- 渲染 ----------------
const esc = (v) => String(v).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const STYLE = `
  text{font-family:"Segoe UI","Microsoft YaHei","PingFang SC",system-ui,sans-serif;fill:#2A313B}
  .ti{font-size:19px;font-weight:700;fill:#1F2430}
  .su{font-size:12px;fill:#747C88}
  .pt{font-size:14px;font-weight:700}
  .tx{font-size:12px;fill:#38414D}
  .sm{font-size:10.5px;fill:#5B6470}
  .cap{font-size:11.5px;fill:#5B6470}
  .mono{font-family:"Cascadia Mono",Consolas,"Courier New",monospace;font-size:11px;fill:#38414D}
  .mono2{font-family:"Cascadia Mono",Consolas,"Courier New",monospace;font-size:9.5px;fill:#38414D}
  .panel{fill:#FBFCFE;stroke:#D9DEE7;stroke-width:1.2}
  .neutral{fill:#fff;stroke:#AEB6C2;stroke-width:1.2}
  .ghost{fill:#F5F7FA;stroke:#D9DEE7;stroke-width:1.1}
  .acc1{fill:#EAF1FD;stroke:#2563EB;stroke-width:1.5}
  .acc2{fill:#FCF1E6;stroke:#C3651F;stroke-width:1.5}
  .cell{fill:#fff;stroke:#AEB6C2;stroke-width:1.1}
  .h1{fill:#DCE7FB;stroke:#AEB6C2;stroke-width:1.1}
  .x{fill:#EEF1F5;stroke:#D9DEE7;stroke-width:1}
  .main{fill:none;stroke:#2563EB;stroke-width:2;marker-end:url(#arrowMain)}
  .aux{fill:none;stroke:#AEB6C2;stroke-width:1.3;stroke-dasharray:5 4;marker-end:url(#arrowAux)}
`;

function render(m) {
  const W = 1180; const H = 1420;
  const o = [];
  const rect = (x, y, w, h, cls = 'neutral', r = 7) => o.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" class="${cls}"/>`);
  const text = (x, y, s, cls = 'tx', anchor = 'start') => o.push(`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${esc(s)}</text>`);
  const arrow = (x1, y1, x2, y2, cls = 'main') => o.push(`<path d="M${x1} ${y1} L${x2} ${y2}" class="${cls}"/>`);
  const c = m.cfg;
  const step = m.dynamic.steps[0];

  o.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="title desc">`);
  o.push('<title id="title">slime Megatron 训练后端：从训练字典到一次 optimizer step 的四次决定性转换</title>');
  o.push('<desc id="desc">泳道 A 复现 build_dp_schedule 的按 rollout 组步、first-fit 打包、拆 bin 对齐与轮询分发；泳道 B 复现 get_batch 对一个 micro-batch 的 zigzag CP 切片、THD 拼接、cu_seqlens 与 next-token mask 对齐，并用同一 micro-batch 对照 allgather_cp 的整体拼接与连续等分；泳道 C 复现 actor 一轮内 ref、old_actor、actor 三个 CPU tag 的切换，以及全批前向次数与 forward_backward_func 调用次数；泳道 D 复现 train_one_step 的清梯度、全部 micro-batch 前反向、一次 optimizer step 与按逻辑 rollout 数推进 LR scheduler。</desc>');
  o.push(`<defs><marker id="arrowMain" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#2563EB"/></marker><marker id="arrowAux" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#AEB6C2"/></marker></defs>`);
  o.push(`<style>${STYLE}</style><rect width="${W}" height="${H}" fill="white"/>`);

  text(24, 34, '一份训练字典 → 一次 optimizer step：调度、打包、角色切换、训练步', 'ti');
  text(24, 56, `dp_size=${c.dpSize} · cp_size=${c.cpSize} · tp=${c.tpSize} · global_batch_size=${c.globalBatchSize} 个逻辑 rollout · max_tokens_per_gpu=${c.maxTokensPerGpu} → cap ${step.maxPerBin} · data_pad_size_multiplier=${c.dataPadSizeMultiplier} · prompt 长 ${c.promptLen}`, 'su');

  // ---- 泳道 A：调度 ----
  const yA = 74; const hA = 336;
  rect(24, yA, 1132, hA, 'panel', 10);
  text(40, yA + 24, 'A  build_dp_schedule：先按 rollout id 组步，再 first-fit 打包，拆 bin 对齐到 dp_size，最后轮询分给 rank', 'pt');
  const sx = 40; const sy = yA + 46;
  text(sx, sy + 12, '样本（total = prompt + response，mask 和）', 'sm');
  m.samples.forEach((s, i) => {
    const x = sx + i * 118;
    rect(x, sy + 20, 110, 44, s.rolloutId === 2 ? 'acc1' : 'cell', 5);
    text(x + 55, sy + 37, `${s.name} · r${s.rolloutId}`, 'mono', 'middle');
    text(x + 55, sy + 54, `${c.promptLen}+${s.responseLen}=${s.totalLen}，mask ${s.lossMask.reduce((a, b) => a + b, 0)}`, 'mono2', 'middle');
  });
  text(sx, sy + 82, `① 按 rollout_id 首次出现顺序分组：r0 r1 r2 r3 → ${step.stepRollouts.length} // ${c.globalBatchSize} = ${m.dynamic.numMicrobatches.length} 步；r2 的两个片段同步`, 'tx');
  text(sx, sy + 100, `② first-fit（cap ${step.maxPerBin}）：${step.packed.map((b) => `[${b.map((i) => m.samples[step.sampleIndices[i]].name).join(' ')}]=${b.reduce((a, i) => a + step.stepLengths[i], 0)}`).join('  ')}  → K=${step.packed.length}`, 'tx');
  const sp = step.splitTrace[0];
  text(sx, sy + 118, `③ 对齐：target_K = ceil(${step.packed.length}/${step.alignTo})×${step.alignTo} = ${step.targetK}；拆最大的多样本 bin（和相同取下标大者）bin${sp.splitBin} [${sp.before.map((i) => m.samples[step.sampleIndices[i]].name).join(' ')}] → [${sp.left.map((i) => m.samples[step.sampleIndices[i]].name).join(' ')}] + [${sp.right.map((i) => m.samples[step.sampleIndices[i]].name).join(' ')}]`, 'tx');
  const by = sy + 132;
  step.bins.forEach((b, k) => {
    const x = sx + k * 150;
    rect(x, by, 140, 30, k === sp.splitBin || k === step.bins.length - 1 ? 'acc1' : 'cell', 5);
    text(x + 70, by + 19, `bin${k} [${b.map((g) => m.samples[g].name).join(' ')}] = ${b.reduce((a, g) => a + m.samples[g].totalLen, 0)}`, 'mono', 'middle');
  });
  const kkSets = m.kk[0].partitions.map((p) => `{${p.map((i) => `bin${i}`).join(',')}}`).join(' / ');
  text(sx, by + 58, `④ 分发：balance_data=False 时 rank r 取 bin r, r+${c.dpSize}, …（各 ${m.dynamic.numMicrobatches[0]} 个 mb）；=True 时 KK 按估算 FLOPs 配对，本例对任意 aL+bL² 也得 ${kkSets}`, 'tx');
  m.rankView.forEach((rv, r) => {
    const x = sx + r * 560;
    rect(x, by + 66, 540, 48, 'neutral', 6);
    text(x + 10, by + 84, `DP rank ${r}: partition=[${rv.partition.join(',')}]  mbs = ${rv.mbs.map((mb) => `[${mb.join(' ')}]`).join(' ')}`, 'mono');
    text(x + 10, by + 102, `micro_batch_indices=${JSON.stringify(m.dynamic.microBatchIndices[r])}  num_microbatches=[${m.dynamic.numMicrobatches.join(',')}]  global_batch_sizes=[${m.dynamic.globalBatchSizes.join(',')}]`, 'mono2');
  });
  // 静态路径
  const rx = 800; const ry = sy + 78;
  rect(rx, ry, 340, 96, 'acc2', 8);
  text(rx + 12, ry + 20, `静态路径（micro_batch_size=${c.microBatchSize}）：固定步长切块`, 'pt');
  text(rx + 12, ry + 40, '[s0 s1] [s2a s2b] [s3] → K=3，不是 dp_size 的倍数', 'tx');
  text(rx + 12, ry + 58, '不拆块，直接 AssertionError（拆会破坏定长不变量）', 'tx');
  text(rx + 12, ry + 76, '→ 调 step_size / micro_batch_size / DP·VPP 使其整除', 'tx');
  text(rx + 12, ry + 92, `尾部凑不满一整步的 rollout 连同片段被丢出 schedule`, 'cap');
  const fA = m.flops.find((f) => f.regime === 'a>22b'); const fB = m.flops.find((f) => f.regime === 'a<=22b');
  text(sx, yA + hA - 28, `balance_by_flops：KK 先分 ${fA.numGroups} 组 ${fA.groups.map((g) => `{${g}}`).join(' ')}（与系数无关）→ 拆 [${fA.split[0].before.join(' ')}] 为 [${fA.split[0].left.join(' ')}]+[${fA.split[0].right.join(' ')}]；强制 balance_data 时 a > 22b 给 rank0 ${fA.rankBins[0].join(' ')}，a ≤ 22b 给 rank0 ${fB.rankBins[0].join(' ')}`, 'cap');
  text(sx, yA + hA - 10, `每条样本还带 rollout_mask_sums=[${m.rolloutMaskSums.join(',')}]（同一 rollout 的片段共用；见 12 页）→ 切分前算好，切分后 s2a、s2b 已在不同 mb`, 'cap');

  // ---- 泳道 B：get_batch ----
  const yB = yA + hA + 12; const hB = 484;
  rect(24, yB, 1132, hB, 'panel', 10);
  text(40, yB + 24, `B  get_batch：rank 0 的 mb0 [${m.rank0Mb0.map((s) => s.name).join(' ')}] 在 cp_size=${c.cpSize} 下的 zigzag 切片、THD 拼接与 next-token mask 对齐`, 'pt');
  const cellW = 34; const cellH = 22; const bx0 = 150;
  const drawStream = (y, label, items, clsFn, textFn) => {
    text(bx0 - 8, y + 15, label, 'mono', 'end');
    items.forEach((it, i) => {
      rect(bx0 + i * cellW, y, cellW - 2, cellH, clsFn(it, i), 3);
      text(bx0 + i * cellW + (cellW - 2) / 2, y + 15, textFn(it, i), 'mono2', 'middle');
    });
  };
  // 原始 token 流（两条样本）
  let yy = yB + 40;
  const s0 = m.rank0Mb0[0]; const s1 = m.rank0Mb0[1];
  const srcItems = [...Array.from({ length: s0.totalLen }, (_, i) => ({ s: s0, i })), ...Array.from({ length: s1.totalLen }, (_, i) => ({ s: s1, i }))];
  drawStream(yy, '原 tokens', srcItems, (it) => (it.i >= c.promptLen ? (it.s.lossMask[it.i - c.promptLen] ? 'h1' : 'x') : 'cell'), (it) => `${it.s.name}:${it.i}`);
  text(bx0 + srcItems.length * cellW + 8, yy + 15, `蓝 = 可训练 response，灰 = prompt / 工具 token（${s0.name}:${c.promptLen + s0.maskedResponseIdx[0]}）`, 'sm');
  yy += 30;
  text(bx0, yy + 10, `每条样本各自 chunk = ceil(total / (2·cp))：${s0.name} chunk=${Math.ceil(s0.totalLen / (2 * c.cpSize))} 无补齐；${s1.name} chunk=${Math.ceil(s1.totalLen / (2 * c.cpSize))} 补 ${2 * c.cpSize * Math.ceil(s1.totalLen / (2 * c.cpSize)) - s1.totalLen} 个 pad → rank r 取第 r 段与第 ${2 * c.cpSize - 1}−r 段`, 'sm');
  yy += 22;
  m.batches.forEach((b, cpRank) => {
    drawStream(yy, `cp${cpRank} tokens`, b.tokens, (t) => (t === 'P' ? 'x' : 'cell'), (t) => (t === 'P' ? 'pad' : t));
    text(bx0 + b.tokens.length * cellW + 8, yy + 15, `cu_seqlens=[${b.cuSeqlens.join(',')}]（局部 [${b.cuLocal.join(',')}] × cp）`, 'mono2');
    yy += cellH + 4;
    drawStream(yy, `cp${cpRank} mask`, b.fullLossMasks, (v, i) => (v ? 'acc1' : b.tokens[i] === 'P' ? 'x' : 'cell'), (v) => v);
    text(bx0 + b.fullLossMasks.length * cellW + 8, yy + 15, `有效 ${b.maskedValid}；负责 response 下标 ${b.owned.map((own, k) => `${m.rank0Mb0[k].name}{${own.join(',')}}`).join(' ')}`, 'mono2');
    yy += cellH + 12;
  });
  const ag = m.allgather;
  text(40, yy + 12, `allgather_cp（DSA 模式）对照：先把整个 mb 拼接、补到 cp × pad_size = ${c.cpSize * m.batches[0].padSize} 的倍数（这里补 ${ag[0].pad}），再按 rank 连续等分`, 'sm');
  yy += 20;
  ag.forEach((b, cpRank) => {
    drawStream(yy, `ag cp${cpRank} tokens`, b.tokens, (t) => (t === 'P' ? 'x' : 'cell'), (t) => (t === 'P' ? 'pad' : t));
    text(bx0 + b.tokens.length * cellW + 8, yy + 15, `cu_seqlens=[${b.cuSeqlens.join(',')}]（不乘 cp）；真实 token ${b.realTokens}`, 'mono2');
    yy += cellH + 4;
    drawStream(yy, `ag cp${cpRank} mask`, b.fullLossMasks, (v, i) => (v ? 'acc2' : b.tokens[i] === 'P' ? 'x' : 'cell'), (v) => v);
    text(bx0 + b.fullLossMasks.length * cellW + 8, yy + 15, `有效 ${b.maskedValid}`, 'mono2');
    yy += cellH + 12;
  });
  const cu0 = m.batches[0].cuSeqlens;
  const paddedLens = m.rank0Mb0.map((smp, i) => `${smp.name} ${cu0[i + 1] - cu0[i]}`).join('、');
  rect(40, yy, 1100, 114, 'ghost', 8);
  text(52, yy + 20, `mask 先按样本左补 prompt_len−1=${c.promptLen - 1}、右补 1，再与 tokens 同样切片：位置 p 的 mask 属于"预测 token p+1 的 logit"，`, 'tx');
  text(52, yy + 38, `所以 response 第 r 个 token 的 mask 落在 p = prompt_len + r − 1。zigzag 拼接后补到 pad_size = tp × ${c.dataPadSizeMultiplier} = ${m.batches[0].padSize} 的倍数（这里补 ${m.batches[0].pad}），`, 'tx');
  text(52, yy + 56, `局部 cu_seqlens 乘 cp_size 得到每条样本补齐后的全局长度（${paddedLens}，${s1.name} 原长 ${s1.totalLen}）。两个 CP rank 的 mask 有效数 ${m.batches.map((b) => b.maskedValid).join(' + ')} = ${m.batches.reduce((a, b) => a + b.maskedValid, 0)} = 两条样本的 mask 和。`, 'tx');
  text(52, yy + 76, `${s1.name} 在 cp0 上一个 response 位置都没有：它分到的两段是 prompt 头与补齐尾。这就是 loss 侧"空 rank 仍参加集合通信"的来源（15 页）。`, 'tx');
  text(52, yy + 96, `allgather 下 cp1 只有 ${ag[1].realTokens} 个真实 token、${ag[1].maskedValid} 个有效位置；logprob 先按连续片算，再经 _allgather_cp_redistribute 一次可微 all-reduce 切回 zigzag。`, 'tx');

  // ---- 泳道 C：一轮 actor 的 tag 切换 ----
  const yC = yB + hB + 12; const hC = 250;
  rect(24, yC, 1132, hC, 'panel', 10);
  text(40, yC + 24, `C  train_actor 一轮：同一份 GPU 模型按 CPU tag 换入换出，只有 actor 走反向；本例 kl_coef=0、use_kl_loss 让 ref 存在`, 'pt');
  const px = 40; const py = yC + 44; const pw = 200;
  m.round.phases.forEach((p, i) => {
    const x = px + i * (pw + 8);
    const cls = p.kind === 'train' ? 'acc1' : p.kind === 'switch_only' ? 'acc2' : p.kind === 'forward_only' ? 'neutral' : 'ghost';
    rect(x, py, pw, 62, cls, 6);
    text(x + pw / 2, py + 18, `${i + 1}. tag ${p.tag}`, 'mono', 'middle');
    text(x + pw / 2, py + 35, p.kind === 'forward_only' ? 'forward_only' : p.kind === 'switch_only' ? '_switch_model 无前向' : p.kind, 'sm', 'middle');
    text(x + pw / 2, py + 52, p.store, 'mono2', 'middle');
    if (i < m.round.phases.length - 1) arrow(x + pw, py + 31, x + pw + 8, py + 31, 'aux');
  });
  text(px, py + 88, `can_reuse_log_probs_in_loss = 单步(${m.dynamic.numMicrobatches.length}) ∧ policy_loss ∧ kl_coef=0 ∧ ¬rollout_logprobs ∧ ¬mismatch ∧ ¬critic ∧ ¬old_actor ∧ ¬OPD ∧ (¬routing_replay ∨ R3) ∧ ≠gspo → ${m.round.canReuse}`, 'tx');
  text(px, py + 106, `全批前向 = ref ${m.round.forwardPasses} + 训练 1 = ${m.round.fullBatchForwards}（forward_backward_func 调用 ${m.round.pipelineCalls} 次）；num_steps_per_rollout=2（G 变为 2）时不可复用，多一次 old-policy 全批前向 = ${m.roundTwoSteps.fullBatchForwards}，调用 ${m.roundTwoSteps.pipelineCalls} 次`, 'tx');
  text(px, py + 124, '每次 _switch_model 是 CPU pinned 张量 → GPU 参数的同名整份拷贝，之后 cuda.synchronize；未知 tag 抛 ValueError。', 'tx');
  text(px, py + 142, '训练结束 backup("actor")；(rollout_id+1) % ref_update_interval == 0 且有 ref tag 时 backup("ref")。critic 是独立 RayTrainGroup，不走 tag。', 'tx');
  text(px, py + 160, 'compute_advantages_and_returns 只在 PP last stage 有 log_probs/values 时计算；normalize_advantages 在 DP×CP 组上 all-reduce 带 mask 统计。', 'cap');
  text(px, py + 178, 'forward_only 用同一个 get_forward_backward_func(forward_only=True) 按步调用；按 micro_batch_indices 还原顺序，冻结基线下是恒等映射（源码 TODO）。', 'cap');

  // ---- 泳道 D：train_one_step ----
  const yD = yC + hC + 12; const hD = H - yD - 34;
  rect(24, yD, 1132, hD, 'panel', 10);
  text(40, yD + 24, `D  train_one_step（每个训练步一次）：K=${m.dynamic.numMicrobatches[0]} 个 micro-batch 的前反向累积，一次 optimizer.step，scheduler.step(increment=${c.globalBatchSize})`, 'pt');
  const dx = 40; const dy = yD + 44; const dw = 176;
  const boxes = [
    ['zero_grad_buffer / zero_grad', 'before-train-step hook', '（可选自定义钩子）', 'neutral'],
    [`forward_backward_func(K=${m.dynamic.numMicrobatches[0]})`, 'closure: get_batch → model', '→ loss_function 回调', 'acc1'],
    ['loss × M / G × world', `M=${m.dynamic.numMicrobatches[0]} G=${c.globalBatchSize} world=dp·cp=${c.dpSize * c.cpSize}`, 'Megatron 再 ÷ M', 'neutral'],
    ['valid_step → optimizer.step()', 'assert update_successful', '仅 NaN 检查关时预检跳过', 'acc1'],
    [`scheduler.step(+${c.globalBatchSize})`, '以逻辑 rollout 数', '计 samples-seen', 'acc1'],
    ['reduce_train_step_metrics', 'PP last stage: Σ_mb，', 'all-reduce dp·cp，再 / G', 'ghost'],
  ];
  boxes.forEach((b, i) => {
    const x = dx + i * (dw + 8);
    rect(x, dy, dw, 66, b[3], 6);
    text(x + dw / 2, dy + 20, b[0], 'mono2', 'middle');
    text(x + dw / 2, dy + 38, b[1], 'sm', 'middle');
    text(x + dw / 2, dy + 54, b[2], 'sm', 'middle');
    if (i < boxes.length - 1) arrow(x + dw, dy + 33, x + dw + 8, dy + 33, i === 1 ? 'main' : 'aux');
  });
  text(dx, dy + 92, `三种边界不能混：rollout round 是数据版本边界；global_batch_sizes 的一个元素（${c.globalBatchSize} 个 rollout）是 optimizer 边界；micro-batch 只是流水线/梯度累积单元。`, 'tx');
  text(dx, dy + 110, `LR scheduler 的总量估算 train_iters = num_rollout × rollout_batch_size × n // G = ${c.numRollout}×${c.rolloutBatchSize}×${c.nSamplesPerPrompt} // ${c.globalBatchSize} = ${m.scheduler.trainIters}，lr_decay_steps = train_iters × G = ${m.scheduler.lrDecaySteps}；实际进度靠每步 increment 累加。`, 'tx');
  text(dx, dy + 128, 'loss_function 内的缩放只是链条第一环：Megatron 再除 M，DDP 在 dp·cp 组平均，最终留下 Σ_g L_g / G；数值账本见 15 页。', 'cap');

  text(24, H - 12, '源码基线：THUDM/slime@681b3adca541 · 复现 build_dp_schedule / expand_bins_by_splitting / slice_with_cp / get_batch / train_actor / train_one_step', 'su');
  o.push('</svg>');
  return o.join('\n');
}

const here = dirname(fileURLToPath(import.meta.url));
const defaultOutput = join(here, '..', '..', '..', 'wiki', '02_engineering', '04_posttrain_frameworks', 'slime', 'assets');
const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  const outputDir = process.argv[2] ? process.argv[2] : defaultOutput;
  mkdirSync(outputDir, { recursive: true });
  writeFileSync(join(outputDir, 'slime_megatron_train_step.svg'), `${render(model())}\n`, 'utf8');
}
