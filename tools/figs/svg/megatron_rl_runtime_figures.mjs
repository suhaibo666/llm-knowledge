// 33_megatron_rl_runtime_analysis.md 的两张图。
//
// 图 1：**时序图** —— rollout 三阶段流水线（prepare / infer / assemble / consume）与提交闸门
//       `_SubmissionGate` 在四种配置下的离散事件仿真。闸门容量按冻结基线
//       megatron/rl/rollout_granularity.py::get_rl_parallel_generation_tasks 算出，归还状态按
//       RELEASE_STATE_BY_SUBMISSION 定，acquire / release 的发生点逐句照抄
//       megatron/rl/agent/api.py::_RolloutPipeline 的 stage_prepare / _infer_one / stage_assemble /
//       stage_consume；训练相与推理相交替、事件循环只在 rank 0 驱动时前进、`consumed` 归还延后到下一次
//       anext，这三条来自 megatron/rl/rl_utils.py::get_environment_rollouts 与 megatron_rl_inference_mode。
//       输出各 lane 的 makespan、trainer 等待、峰值在途 rollout、闸门峰值占用与每批训练时的
//       policy-version staleness（最旧 / 最新 token，口径同 rl_utils.py::prep_wandb_metrics）。
// 图 2：**布局图** —— 复刻 megatron/rl/sequence_packing_utils.py::SequencePacker.pack_sequences
//       （按长度降序、放不下就开新箱、pad 到 bin_size）、distribute_packed_bins 的 fifo / round-robin
//       与空箱补齐、update_microbatch_calculator 的 bins 口径微批数、get_packing_efficiency 与
//       log_packing_efficiency 两种效率口径，以及 rl_utils.py::calculate_grpo_loss 在 packed 布局下
//       按 seq_starts / seq_lengths 铺开优势的规则；另跑一条 first-fit-decreasing 对照 lane（docstring
//       自称 first-fit，代码是 next-fit——对照是本图的分析重建）。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
//
// 图 1 要回答「`rl_generation_lag` 与 R/G/B 提交粒度各自决定什么：闸门容量、归还时机、
//   trainer 能领先几批、policy staleness 到底是几」。算例：grpo_prompts_per_step=2、grpo_group_size=2、
//   4 个 trainer batch = 16 条 rollout，推理时长 [3,5,2,4, 6,1,3,2, 4,4,2,5, 3,2,6,1]（示意单位），
//   装配 a=1，训练 T=6。四条 lane：① lag=0 / B（非 streaming 同步基线）；② lag=1 / B；③ lag=1 / G；
//   ④ lag=1 / R。每条 lane：上方按 slot 画闸门占用条（acc1），中间每条 rollout 一行的三阶段阶梯
//   （排队 ghost → 推理 tick 按「训练时的 staleness」着色：0 浅蓝、1 中蓝、≥2 橙 → 装配 neutral），
//   归还点用 acc1 圆点标在该提交单元第一行；下方 trainer 条与 `broadcast_object_list` 可见点（acc2）。
//   底部两个盒子：R/G/B × 归还状态 × capacity 表；仿真结论（B 的最旧 token staleness ≤ lag；G/R 的归还
//   不再与训练消费挂钩，staleness 由推理时长与 capacity 决定，本算例 G 到 2、R 到 3；
//   `prevent_dataset_reorder == (consumption == "B")`；slot 归还 ≠ 全 rank 可见）。
//
// 图 2 要回答「变长 rollout 怎样被装进 THD 箱、箱怎样分到 DP rank、为什么微批数会变、优势为什么
//   不能靠广播」。算例：10 条序列长度 {7,3,9,2,5,6,1,8,4,12}，bin_size=16，DP=2，
//   max_sequences_per_bin=50（CLI 默认），samples_ratio_per_step=1，micro_batch_size=1。
//   面板 A：原始变长序列条；面板 B：next-fit 装箱结果（每格标序列 id，pad 用 ghost）与
//   first-fit-decreasing 对照（acc2 标少掉的箱）；面板 C：fifo / round-robin 下各 rank 的 bins、
//   空箱补齐与微批数（unpacked 5 → packed 3；换一批更长的 rollout → 5：可增可减）；
//   面板 D：一个箱里两条序列的优势按 seq_starts / seq_lengths 逐条铺开 vs 广播（acc2 标串到隔壁）。
//
// 用法：node tools/figs/svg/megatron_rl_runtime_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// ============================================================================
// 共用算例
// ============================================================================

// 图 1：流水线仿真
const PIPE_CASE = Object.freeze({
  nPrompts: 2, // --grpo-prompts-per-step
  groupSize: 2, // --grpo-group-size
  batches: 4, // 仿真跑几个 trainer batch
  durations: Object.freeze([3, 5, 2, 4, 6, 1, 3, 2, 4, 4, 2, 5, 3, 2, 6, 1]), // 每条 rollout 的推理时长（示意单位）
  assemble: 1, // build_rollout（reward）每组 1 单位
  train: 6, // 一次训练 step 的时长
});

// 图 2：打包
const PACK_CASE = Object.freeze({
  lengths: Object.freeze([7, 3, 9, 2, 5, 6, 1, 8, 4, 12]),
  binSize: 16, // = seq_length（pack_all_trajectories 传 args.seq_length）
  dp: 2,
  maxSequencesPerBin: 50, // --rl-sequence-packing-max-sequences-per-bin 的 CLI 默认
  microBatchSize: 1, // validate_args：打包时 micro_batch_size 必须为 1
  samplesRatioPerStep: 1, // global_batch_size / (prompts × group_size)
});

// 图 2 面板 C 的第二算例：更长的一批 rollout，用来证明微批数也会增
const PACK_CASE_LONG = Object.freeze({
  ...PACK_CASE,
  lengths: Object.freeze([15, 14, 13, 12, 11, 10, 9, 9, 8, 8]),
});

// ============================================================================
// 复刻 rollout_granularity.py 与 _GranularityConfig / validate_args 的守卫
// ============================================================================

const RELEASE_STATE_BY_SUBMISSION = Object.freeze({ R: 'inferred', G: 'assembled', B: 'consumed' });

function getRlParallelGenerationTasks(args) {
  // rollout_granularity.py::get_rl_parallel_generation_tasks 逐句
  let parallelGenerationTasks = args.rl_generation_lag + 1;
  if (args.rl_submission_granularity !== 'B') parallelGenerationTasks *= args.grpo_prompts_per_step;
  if (args.rl_submission_granularity === 'R') parallelGenerationTasks *= args.grpo_group_size;
  return parallelGenerationTasks;
}

function validateGranularity(req) {
  // arguments.py::validate_args 的四条 assert + api.py::_GranularityConfig._validate 的两条
  if (req.lag > 0 && !req.partialRollouts) throw new Error('--rl-generation-lag requires --rl-partial-rollouts.');
  if (req.submission === 'R' && !req.partialRollouts) throw new Error('Rollout submission granularity requires streaming grouped rollouts.');
  if (req.consumption === 'R') throw new Error('--rl-consumption-granularity R is not currently supported.');
  if (req.submission === 'B' && req.consumption === 'G') throw new Error('Batch submission with group consumption is not supported.');
  if (req.filterGroupsWithSameReward) throw new Error('filter_groups_with_same_reward is not currently supported');
  return { submission: req.submission, consumption: req.consumption, preventDatasetReorder: req.consumption === 'B' };
}

function rolloutsPerSubmissionUnit(submission, n, g) {
  return { R: 1, G: g, B: n * g }[submission];
}

function numInferWorkers({ capacity, submission, n, g, streaming }) {
  // _RolloutPipeline.__init__
  let workers = capacity * rolloutsPerSubmissionUnit(submission, n, g);
  if (!streaming) workers = Math.min(workers, n * g);
  return workers;
}

// ============================================================================
// 图 1：离散事件仿真
//   时间为整数 tick；推理只在「推理相」前进（事件循环只在 rank 0 的 run_until_complete 里跑，
//   训练相里引擎 suspend）；一条 rollout 每 tick 前进 1 单位并盖上当时的 policy version；
//   prepare 在闸门有空位时立刻提交；被提交但当 tick 结束即为相末的项不前进（下一相才拿到 token）；
//   B-consume：批内 n 组齐了才整批 yield，rank 0 拿满 n 组即离开推理相；
//   `consumed` 归还在下一次 anext（下一相开头）才执行；非 streaming 时每相新建流水线并在相末 drain。
// ============================================================================

function simulatePipeline(cfg) {
  const { lag, submission } = cfg;
  const n = cfg.nPrompts;
  const g = cfg.groupSize;
  const K = cfg.batches;
  const a = cfg.assemble;
  const T = cfg.train;
  const streaming = cfg.streaming ?? true;
  const consumption = cfg.consumption ?? 'B';
  validateGranularity({ lag, submission, consumption, partialRollouts: streaming, filterGroupsWithSameReward: false });
  const args = {
    rl_generation_lag: lag,
    rl_submission_granularity: submission,
    grpo_prompts_per_step: n,
    grpo_group_size: g,
  };
  const capacity = getRlParallelGenerationTasks(args);
  const releaseOn = RELEASE_STATE_BY_SUBMISSION[submission];
  const workers = numInferWorkers({ capacity, submission, n, g, streaming });
  const total = K * n * g;
  if (cfg.durations.length !== total) throw new Error('durations 长度必须等于 batches × n × g');

  const rollouts = cfg.durations.map((dur, id) => ({
    id,
    group: Math.floor(id / g),
    batch: Math.floor(id / (n * g)),
    dur,
    progress: 0,
    submittedAt: null,
    startedAt: null,
    inferredAt: null,
    versions: [],
  }));
  const groups = Array.from({ length: K * n }, (_, gi) => ({
    id: gi,
    batch: Math.floor(gi / n),
    inferredAt: null,
    assembleLeft: a,
    assembledAt: null,
  }));
  const batches = Array.from({ length: K }, (_, b) => ({ id: b, phaseStart: null, yieldedAt: null, trainStart: null, trainEnd: null, releaseAt: null, version: null }));

  // ---- 闸门（asyncio.Semaphore(capacity)）；slot 编号只为画图 ----
  let held = 0;
  let peakHeld = 0;
  const slotFree = new Array(capacity).fill(true);
  const holders = new Map();
  const slotIntervals = [];
  const acquire = (key, t) => {
    if (held >= capacity) return false;
    const s = slotFree.indexOf(true);
    slotFree[s] = false;
    held += 1;
    peakHeld = Math.max(peakHeld, held);
    holders.set(key, { slot: s, start: t });
    return true;
  };
  const release = (key, t) => {
    const h = holders.get(key);
    if (!h) throw new Error(`release of unheld ${key}`);
    slotFree[h.slot] = true;
    held -= 1;
    holders.delete(key);
    slotIntervals.push({ slot: h.slot, start: h.start, end: t, key });
  };
  const resetGate = () => {
    if (holders.size !== 0) throw new Error('non-streaming 相末 drain 后闸门应为空');
    slotFree.fill(true);
    held = 0;
  };

  // ---- stage_prepare：顺序提交，按提交粒度 acquire ----
  let next = 0;
  const submitQueue = []; // 已提交、未推理完（FIFO 给 infer worker）
  const unitKey = (r) => (submission === 'B' ? `B${r.batch}` : submission === 'G' ? `G${r.group}` : `R${r.id}`);
  const isBoundary = (r) => (submission === 'B' ? r.id % (n * g) === 0 : submission === 'G' ? r.id % g === 0 : true);
  const prepare = (t, limit) => {
    while (next < limit) {
      const r = rollouts[next];
      if (isBoundary(r) && !acquire(unitKey(r), t)) return;
      r.submittedAt = t;
      submitQueue.push(r);
      next += 1;
    }
  };

  let t = 0;
  let v = 0;
  let engineIdle = 0;
  let peakInflight = 0;
  const trace = [];
  const phases = [];
  const allAssembled = (b) => groups.filter((x) => x.batch === b).every((x) => x.assembledAt !== null);

  for (let k = 0; k < K; k += 1) {
    const limit = streaming ? total : (k + 1) * n * g;
    if (!streaming) {
      resetGate();
      next = k * n * g;
    }
    batches[k].phaseStart = t;
    batches[k].version = v;
    phases.push({ batch: k, start: t, version: v });
    // stage_consume 在上一次 yield 处挂起；这次 anext 先执行 release_after("consumed")
    if (streaming && k > 0 && releaseOn === 'consumed') {
      release(`B${k - 1}`, t);
      batches[k - 1].releaseAt = t;
    }
    for (;;) {
      if (allAssembled(k)) {
        batches[k].yieldedAt = t;
        break;
      }
      prepare(t, limit);
      const active = submitQueue.slice(0, workers);
      if (active.length === 0) engineIdle += 1;
      peakInflight = Math.max(peakInflight, submitQueue.length);
      trace.push({ t, inflight: submitQueue.length, held });
      for (const r of active) {
        if (r.startedAt === null) r.startedAt = t;
        r.progress += 1;
        r.versions.push(v);
      }
      for (const grp of groups) {
        if (grp.inferredAt !== null && grp.assembledAt === null) grp.assembleLeft -= 1;
      }
      t += 1;
      for (const r of active) {
        if (r.progress === r.dur && r.inferredAt === null) {
          r.inferredAt = t;
          submitQueue.splice(submitQueue.indexOf(r), 1);
          if (releaseOn === 'inferred') release(unitKey(r), t);
          const grp = groups[r.group];
          if (rollouts.filter((x) => x.group === grp.id).every((x) => x.inferredAt !== null)) grp.inferredAt = t;
        }
      }
      for (const grp of groups) {
        if (grp.inferredAt !== null && grp.assembledAt === null && grp.assembleLeft === 0) {
          grp.assembledAt = t;
          if (releaseOn === 'assembled') release(`G${grp.id}`, t);
        }
      }
    }
    phases[k].end = t;
    if (!streaming) {
      // 非 streaming：rank 0 把 generator 耗尽 —— consumed 归还与 shutdown 都在本相相末
      if (releaseOn === 'consumed') {
        release(`B${k}`, t);
        batches[k].releaseAt = t;
      }
      if (submitQueue.length !== 0) throw new Error('non-streaming 相末不应有在途 rollout');
    }
    batches[k].trainStart = t;
    batches[k].trainEnd = t + T;
    t += T;
    v += 1;
  }
  const makespan = t;
  // streaming 下最后一批的 consumed 归还永远不会执行（没有下一次 anext）；画图时延到 makespan
  for (const [key, h] of holders) slotIntervals.push({ slot: h.slot, start: h.start, end: makespan, key, open: true });

  const staleness = batches.map((b) => {
    const rs = rollouts.filter((r) => r.batch === b.id);
    const oldest = Math.max(...rs.map((r) => b.version - r.versions[0]));
    const newest = Math.max(...rs.map((r) => b.version - r.versions[r.versions.length - 1]));
    return { batch: b.id, oldest, newest };
  });
  const trainerWait = batches.reduce((acc, b) => acc + (b.yieldedAt - b.phaseStart), 0);
  return {
    name: `lag=${lag}/${submission}${streaming ? '' : '（非 streaming）'}`,
    lag, submission, streaming, capacity, releaseOn, workers,
    rollouts, groups, batches, phases, slotIntervals, trace,
    makespan, trainerWait, engineIdle, peakInflight, peakHeld,
    staleness,
    maxOldest: Math.max(...staleness.map((s) => s.oldest)),
    maxNewest: Math.max(...staleness.map((s) => s.newest)),
  };
}

const LANE_SYNC = simulatePipeline({ ...PIPE_CASE, lag: 0, submission: 'B', streaming: false });
const LANE_B0_STREAM = simulatePipeline({ ...PIPE_CASE, lag: 0, submission: 'B', streaming: true });
const LANE_B = simulatePipeline({ ...PIPE_CASE, lag: 1, submission: 'B' });
const LANE_G = simulatePipeline({ ...PIPE_CASE, lag: 1, submission: 'G' });
const LANE_R = simulatePipeline({ ...PIPE_CASE, lag: 1, submission: 'R' });
const LANE_G0 = simulatePipeline({ ...PIPE_CASE, lag: 0, submission: 'G' });
const LANE_R0 = simulatePipeline({ ...PIPE_CASE, lag: 0, submission: 'R' });
const LANES = Object.freeze([LANE_SYNC, LANE_B, LANE_G, LANE_R]);

// capacity 表：算例 (n=2, g=2) 与 tests/unit_tests/rl/test_rl_utils.py 锁定的 (n=8, g=4)
function capacityTable(n, g, lags) {
  const rows = [];
  for (const submission of ['B', 'G', 'R']) {
    rows.push({
      submission,
      releaseOn: RELEASE_STATE_BY_SUBMISSION[submission],
      perUnit: rolloutsPerSubmissionUnit(submission, n, g),
      capacity: lags.map((lag) => getRlParallelGenerationTasks({ rl_generation_lag: lag, rl_submission_granularity: submission, grpo_prompts_per_step: n, grpo_group_size: g })),
    });
  }
  return rows;
}
const CAPACITY = capacityTable(PIPE_CASE.nPrompts, PIPE_CASE.groupSize, [0, 1]);
const CAPACITY_TEST = capacityTable(8, 4, [0, 2]);

// 立论前提自检
{
  if (LANE_SYNC.makespan !== LANE_B0_STREAM.makespan) throw new Error('lag=0/B 的 streaming 与非 streaming 应给出同一条时间线');
  if (LANE_B.maxOldest > LANE_B.lag) throw new Error('B 提交下 staleness 应 ≤ lag');
  if (LANE_G.maxOldest <= LANE_G.lag) throw new Error('本算例 G 提交应出现 staleness > lag，否则图的结论不成立');
  for (const lane of LANES) if (lane.peakHeld > lane.capacity) throw new Error('闸门占用不应超过 capacity');
  if (LANE_R.peakInflight > LANE_R.capacity) throw new Error('R 提交下在途 rollout 应 ≤ capacity（test_rollout_submission_granularity_limits_inference_concurrency）');
}

// ============================================================================
// 图 2：复刻 sequence_packing_utils.py
// ============================================================================

function packSequences(lengths, binSize, maxSequencesPerBin) {
  // SequencePacker.pack_sequences：sorted(reverse=True) 后逐条放；放不下（或箱满）就开新箱，
  // 从不回头看旧箱 —— docstring 写 first-fit，代码是 next-fit
  const sortedIndices = [...lengths.keys()].sort((i, j) => lengths[j] - lengths[i]);
  const bins = [];
  let current = [];
  let currentLen = 0;
  for (const idx of sortedIndices) {
    const len = lengths[idx];
    if (currentLen + len <= binSize && current.length < maxSequencesPerBin) {
      current.push(idx);
      currentLen += len;
    } else {
      if (current.length) bins.push(current);
      current = [idx];
      currentLen = len;
    }
  }
  if (current.length) bins.push(current);
  const seqStarts = bins.map((bin) => {
    const starts = [];
    let pos = 0;
    for (const idx of bin) {
      starts.push(pos);
      pos += lengths[idx];
    }
    starts.push(pos); // 源码最后再 append 一次 current_pos
    return starts;
  });
  const seqToBin = new Array(lengths.length).fill(null);
  bins.forEach((bin, b) => bin.forEach((idx) => { seqToBin[idx] = b; }));
  return { bins, seqStarts, seqLengths: [...lengths], seqToBin, sortedIndices };
}

function firstFitDecreasing(lengths, binSize, maxSequencesPerBin) {
  // 对照：docstring 所说的 first-fit（回头找第一个装得下的箱）—— 本图的分析重建，不是源码
  const sortedIndices = [...lengths.keys()].sort((i, j) => lengths[j] - lengths[i]);
  const bins = [];
  const used = [];
  for (const idx of sortedIndices) {
    const len = lengths[idx];
    let placed = false;
    for (let b = 0; b < bins.length; b += 1) {
      if (used[b] + len <= binSize && bins[b].length < maxSequencesPerBin) {
        bins[b].push(idx);
        used[b] += len;
        placed = true;
        break;
      }
    }
    if (!placed) {
      bins.push([idx]);
      used.push(len);
    }
  }
  return { bins };
}

function distributePackedBins(numBins, dp, algo) {
  // distribute_packed_bins：round-robin 取 rank, rank+world, …；fifo 连续切、多出的箱给前面的 rank；
  // 不足 max_bins_per_rank 的 rank 补空箱
  const maxBinsPerRank = Math.ceil(numBins / dp);
  const ranks = [];
  for (let rank = 0; rank < dp; rank += 1) {
    let mine;
    if (algo === 'round-robin') {
      mine = [];
      for (let b = rank; b < numBins; b += dp) mine.push(b);
    } else {
      const binsPerRank = Math.floor(numBins / dp);
      const extra = numBins % dp;
      let start;
      let end;
      if (rank < extra) {
        start = rank * (binsPerRank + 1);
        end = start + binsPerRank + 1;
      } else {
        start = rank * binsPerRank + extra;
        end = start + binsPerRank;
      }
      mine = [];
      for (let b = start; b < end; b += 1) mine.push(b);
    }
    ranks.push({ rank, bins: mine, emptyBins: maxBinsPerRank - mine.length });
  }
  return { algo, maxBinsPerRank, ranks };
}

function updateMicrobatchCalculator({ samplesRatioPerStep, numBinsThisRank, dp, microBatchSize }) {
  // update_microbatch_calculator + ConstantNumMicroBatchesCalculator（decrease_batch_size_if_needed=False）
  const localBinsPerStep = Math.ceil(samplesRatioPerStep * numBinsThisRank);
  const binsBs = localBinsPerStep * dp;
  const mbTimesDp = microBatchSize * dp;
  if (binsBs % mbTimesDp !== 0) throw new Error('bins_bs 必须能被 micro_batch_size × dp 整除');
  return { localBinsPerStep, binsBs, numMicrobatches: binsBs / mbTimesDp };
}

function unpackedMicrobatches({ samplesRatioPerStep, totalTurns, dp, microBatchSize }) {
  // prepare_data_for_update 非打包分支：global_batch_size = ceil(ratio × total_turns_sampled)
  const gbs = Math.ceil(samplesRatioPerStep * totalTurns);
  const mbTimesDp = microBatchSize * dp;
  if (gbs % mbTimesDp !== 0) throw new Error('unpacked global batch 必须能被 micro_batch_size × dp 整除');
  return { globalBatchSize: gbs, numMicrobatches: gbs / mbTimesDp };
}

function packingEfficiency(seqLengths, binsPerRank, binSize, dp) {
  // get_packing_efficiency：全体真实 token / (每 rank 箱数（含空箱） × bin_size × dp)
  const total = seqLengths.reduce((acc, x) => acc + x, 0);
  return total / (binsPerRank * binSize * dp);
}

function perRankEfficiency(rankBins, packing, binSize, emptyBins) {
  // log_packing_efficiency：本 rank 真实 token / (本 rank packed_trajs.shape[0] × bin_size)，shape[0] 含空箱
  const myTokens = rankBins.reduce((acc, b) => acc + packing.bins[b].reduce((s, idx) => s + packing.seqLengths[idx], 0), 0);
  const capacity = (rankBins.length + emptyBins) * binSize;
  return { myTokens, capacity, efficiency: myTokens / capacity };
}

function packedAdvantages(bin, packing, advantages, binSize) {
  // calculate_grpo_loss 的 packed 分支：logprobs 比 token 短 1，end = min(start + len − 1, bin_size)
  const out = new Array(binSize).fill(0);
  const owner = new Array(binSize).fill(null);
  bin.forEach((idx, k) => {
    const start = packing.seqStarts[packing.seqToBin[idx]][k];
    const end = Math.min(start + packing.seqLengths[idx] - 1, binSize);
    for (let p = start; p < end; p += 1) {
      out[p] = advantages[idx];
      owner[p] = idx;
    }
  });
  return { values: out, owner };
}

function runPacking(caseSpec) {
  const packing = packSequences(caseSpec.lengths, caseSpec.binSize, caseSpec.maxSequencesPerBin);
  const ffd = firstFitDecreasing(caseSpec.lengths, caseSpec.binSize, caseSpec.maxSequencesPerBin);
  const numBins = packing.bins.length;
  const fifo = distributePackedBins(numBins, caseSpec.dp, 'fifo');
  const rr = distributePackedBins(numBins, caseSpec.dp, 'round-robin');
  const micro = updateMicrobatchCalculator({
    samplesRatioPerStep: caseSpec.samplesRatioPerStep,
    numBinsThisRank: fifo.maxBinsPerRank, // 补空箱后每 rank 箱数相同
    dp: caseSpec.dp,
    microBatchSize: caseSpec.microBatchSize,
  });
  const unpacked = unpackedMicrobatches({
    samplesRatioPerStep: caseSpec.samplesRatioPerStep,
    totalTurns: caseSpec.lengths.length,
    dp: caseSpec.dp,
    microBatchSize: caseSpec.microBatchSize,
  });
  const totalTokens = caseSpec.lengths.reduce((acc, x) => acc + x, 0);
  const binFill = packing.bins.map((bin) => bin.reduce((s, idx) => s + caseSpec.lengths[idx], 0));
  return {
    ...caseSpec,
    packing,
    ffd,
    numBins,
    binFill,
    totalTokens,
    fifo,
    rr,
    micro,
    unpacked,
    efficiency: packingEfficiency(caseSpec.lengths, fifo.maxBinsPerRank, caseSpec.binSize, caseSpec.dp),
    rawEfficiency: totalTokens / (numBins * caseSpec.binSize), // 不算空箱的朴素口径（本图对照）
    ffdEfficiency: totalTokens / (ffd.bins.length * caseSpec.binSize),
    perRank: {
      fifo: fifo.ranks.map((r) => perRankEfficiency(r.bins, packing, caseSpec.binSize, r.emptyBins)),
      rr: rr.ranks.map((r) => perRankEfficiency(r.bins, packing, caseSpec.binSize, r.emptyBins)),
    },
  };
}

const PACK = runPacking(PACK_CASE);
const PACK_LONG = runPacking(PACK_CASE_LONG);

// 面板 D：挑一个装了两条序列的箱
const ADV_BIN_INDEX = PACK.packing.bins.findIndex((bin) => bin.length === 2);
if (ADV_BIN_INDEX < 0) throw new Error('算例里应有一个恰好装两条序列的箱');
const ADV_SYMBOLS = Object.fromEntries(PACK.lengths.map((_, idx) => [idx, `A${idx}`]));
const ADV = packedAdvantages(PACK.packing.bins[ADV_BIN_INDEX], PACK.packing, ADV_SYMBOLS, PACK.binSize);

// 立论前提自检
{
  if (PACK.numBins <= PACK.ffd.bins.length) throw new Error('本算例 next-fit 应比 first-fit 多用箱，否则对照无意义');
  if (PACK.micro.numMicrobatches >= PACK.unpacked.numMicrobatches) throw new Error('本算例打包后微批数应减少');
  if (PACK_LONG.micro.numMicrobatches <= PACK.micro.numMicrobatches) throw new Error('第二算例微批数应增加');
  if (PACK.fifo.ranks.every((r) => r.emptyBins === 0)) throw new Error('本算例应出现空箱补齐');
  // tests/unit_tests/rl/test_sequence_packing_utils.py::test_sequence_packing_integration 的算例
  const t = packSequences([4, 3, 5], 16, 16);
  if (t.bins.length !== 1 || t.bins[0].join(',') !== '2,0,1') throw new Error('test_sequence_packing_integration 复刻失败');
}

if (process.env.MRL_FIG_DEBUG) {
  for (const lane of [LANE_SYNC, LANE_B0_STREAM, LANE_B, LANE_G, LANE_R, LANE_G0, LANE_R0]) {
    console.log(lane.name, JSON.stringify({ cap: lane.capacity, workers: lane.workers, makespan: lane.makespan, wait: lane.trainerWait, idle: lane.engineIdle, peak: lane.peakInflight, held: lane.peakHeld, stal: lane.staleness }));
    console.log('  phases', JSON.stringify(lane.phases));
    console.log('  slots', JSON.stringify(lane.slotIntervals));
  }
  console.log('PACK', JSON.stringify({ bins: PACK.packing.bins, fill: PACK.binFill, ffd: PACK.ffd.bins, fifo: PACK.fifo, rr: PACK.rr, micro: PACK.micro, unpacked: PACK.unpacked, eff: PACK.efficiency, perRank: PACK.perRank }));
  console.log('PACK_LONG', JSON.stringify({ bins: PACK_LONG.packing.bins, micro: PACK_LONG.micro }));
  console.log('ADV', JSON.stringify(ADV));
}

// ============================================================================
// SVG 基础设施（与 tools/figs/svg/megatron_fsdp_figures.mjs 同一套 token）
// ============================================================================

const esc = (value) => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

function textWidth(value, fontSize) {
  let units = 0;
  // ASCII 按 0.6em 估（WebKit/system-ui 实测比 0.56 宽），其余字符按 1em
  for (const ch of String(value)) units += ch.charCodeAt(0) < 0x7f ? 0.6 : 1;
  return units * fontSize;
}

// 从候选标签里挑第一个放得下的；都放不下返回 null（调用方只画格子不写字）
function fitLabel(candidates, fontSize, limit) {
  return candidates.find((c) => textWidth(c, fontSize) <= limit) ?? null;
}

function guard(value, fontSize, limit, where) {
  const width = textWidth(value, fontSize);
  if (width > limit) throw new Error(`${where}: "${value}" 需要 ${width.toFixed(1)}px，超出 ${limit}px`);
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
  .outline1{fill:none;stroke:#2563EB;stroke-width:1.8}
  .outline2{fill:none;stroke:#C3651F;stroke-width:1.8}
  .r0{fill:#EEF1F5;stroke:#AEB6C2;stroke-width:1}
  .r1{fill:#D8DEE8;stroke:#AEB6C2;stroke-width:1}
  .bar{fill:#F4C9A3;stroke:#C3651F;stroke-width:.8}
  .bar1{fill:#CFE0FA;stroke:#2563EB;stroke-width:.8}
  .bar2{fill:#9CC2F3;stroke:#2563EB;stroke-width:.8}
  .bar3{fill:#5E97E4;stroke:#2563EB;stroke-width:.8}
  .bar4{fill:#E7EAEE;stroke:#AEB6C2;stroke-width:.8}
  .train{fill:#DDE3EA;stroke:#5B6470;stroke-width:1}
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
  .sep2{fill:none;stroke:#C3651F;stroke-width:1.4;stroke-dasharray:3 3}
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
function circle(cx, cy, r, cls) {
  return `<circle class="${cls}" cx="${cx}" cy="${cy}" r="${r}"/>`;
}
function infoBox(x, y, w, h, title, lines, cls = 'neutral', where = 'box') {
  const out = [rect(x, y, w, h, cls)];
  const inner = w - 24;
  out.push(text(x + 12, y + 21, guard(title, 12, inner, `${where}/title`), 'tx'));
  out.push(line(x + 12, y + 29, x + w - 12, y + 29));
  lines.forEach((entry, index) => {
    const value = typeof entry === 'string' ? entry : entry.text;
    const lineCls = typeof entry === 'string' ? 'sm' : entry.cls;
    out.push(text(x + 12, y + 47 + index * 16, guard(value, 10.5, inner, `${where}/L${index}`), lineCls));
  });
  return out.join('\n');
}
function header(w, title, subtitle) {
  // subtitle 可以是字符串或多行数组；每行都过宽度守卫
  const lines = Array.isArray(subtitle) ? subtitle : [subtitle];
  return [
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} __H__" width="${w}" height="__H__" role="img">`,
    defs,
    `<style>${sharedStyle}</style>`,
    text(28, 32, guard(title, 18, w - 56, 'header/title'), 'ti'),
    ...lines.map((ln, i) => text(28, 52 + i * 14, guard(ln, 11.5, w - 56, `header/sub${i}`), 'su')),
  ];
}

const FONT = Object.freeze({ ti: 18, su: 11.5, pt: 14, tx: 12, sm: 10.5, dim: 10.5, costtx: 10.5, rank: 11, cap: 11 });

function assertNoTextOverlap(svg, name) {
  const m = svg.match(/viewBox="0 0 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)"/);
  const canvasW = Number(m[1]);
  const canvasH = Number(m[2]);
  const boxes = [];
  for (const hit of svg.matchAll(/<text class="([a-z0-9]+)" x="(-?[\d.]+)" y="(-?[\d.]+)" text-anchor="(\w+)">([^<]*)<\/text>/g)) {
    const [, cls, xs, ys, anchor, raw] = hit;
    const size = FONT[cls];
    if (!size || raw.trim() === '') continue;
    const w = textWidth(raw.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>'), size);
    const x = anchor === 'middle' ? Number(xs) - w / 2 : anchor === 'end' ? Number(xs) - w : Number(xs);
    boxes.push({ x, y: Number(ys) - size * 0.78, w, h: size * 1.06, raw });
  }
  for (const b of boxes) {
    if (b.x < -1 || b.y < -1 || b.x + b.w > canvasW + 1 || b.y + b.h > canvasH + 1) {
      throw new Error(`${name}: 文字盒出画布 "${b.raw}"（${b.x.toFixed(1)}..${(b.x + b.w).toFixed(1)} × ${b.y.toFixed(1)}..${(b.y + b.h).toFixed(1)}，画布 ${canvasW}×${canvasH}）`);
    }
  }
  for (let i = 0; i < boxes.length; i += 1) {
    for (let j = i + 1; j < boxes.length; j += 1) {
      const a = boxes[i];
      const b = boxes[j];
      const dx = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
      const dy = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
      if (dx > 1 && dy > 1) throw new Error(`${name}: 文字重叠 "${a.raw}" × "${b.raw}"（重合 ${dx.toFixed(1)}×${dy.toFixed(1)}px）`);
    }
  }
  return svg;
}

const seal = (parts, w, h, name) => assertNoTextOverlap(parts.join('\n').replace(/__H__/g, String(h)) + '\n</svg>', name);

const pct = (x) => `${(x * 100).toFixed(1)}%`;

// ============================================================================
// 图 1：时序
// ============================================================================

const LANE_TITLES = {
  [LANE_SYNC.name]: '① lag=0 / B（非 streaming 同步基线）',
  [LANE_B.name]: '② lag=1 / B',
  [LANE_G.name]: '③ lag=1 / G',
  [LANE_R.name]: '④ lag=1 / R',
};

function renderLane(parts, lane, { x0, y0, labelW, timeW, tMax, where }) {
  const px = timeW / tMax;
  const tx = (t) => x0 + labelW + t * px;
  const n = PIPE_CASE.nPrompts;
  const g = PIPE_CASE.groupSize;
  const title = `${LANE_TITLES[lane.name]}：capacity ${lane.capacity} × ${lane.submission === 'B' ? 'batch' : lane.submission === 'G' ? 'group' : 'rollout'} slot，归还于 ${lane.releaseOn}，infer workers ${lane.workers}`;
  parts.push(text(x0, y0, guard(title, 14, labelW + timeW, `${where}/title`), 'pt'));
  const stats = `makespan ${lane.makespan} · trainer 等待 ${lane.trainerWait} · 引擎空闲 ${lane.engineIdle} · 峰值在途 ${lane.peakInflight} · 闸门峰值 ${lane.peakHeld}/${lane.capacity} · staleness 最旧 ≤ ${lane.maxOldest}、最新 ≤ ${lane.maxNewest}`;
  parts.push(text(x0, y0 + 16, guard(stats, 10.5, labelW + timeW, `${where}/stats`), 'costtx'));
  let y = y0 + 24;
  // 推理相 / 训练相带
  const bandY = y + 2;
  lane.phases.forEach((ph) => {
    parts.push(rect(tx(ph.start), bandY, Math.max((ph.end - ph.start) * px, 1), 8, 'bar1', 1));
    parts.push(text(tx(ph.start) + 2, bandY + 20, `v${ph.version}`, 'dim'));
  });
  lane.batches.forEach((b) => parts.push(rect(tx(b.trainStart), bandY, (b.trainEnd - b.trainStart) * px, 8, 'train', 1)));
  parts.push(text(x0 + labelW - 8, bandY + 8, '相', 'rank', 'end'));
  y = bandY + 26;
  // 闸门占用（按 slot）
  const slotH = lane.capacity > 4 ? 7 : 11;
  parts.push(text(x0 + labelW - 8, y + Math.min(lane.capacity * slotH, 22) / 2 + 4, `gate ×${lane.capacity}`, 'rank', 'end'));
  for (let s = 0; s < lane.capacity; s += 1) {
    const sy = y + s * slotH;
    parts.push(line(tx(0), sy + slotH / 2, tx(tMax), sy + slotH / 2, 'gl'));
  }
  lane.slotIntervals.forEach((iv) => {
    const sy = y + iv.slot * slotH;
    const w = Math.max((iv.end - iv.start) * px, 2);
    parts.push(rect(tx(iv.start), sy + 1, w, slotH - 2, iv.open ? 'ghost' : 'acc1', 1));
    if (slotH >= 11 && w >= textWidth(iv.key, 10.5) + 6) parts.push(text(tx(iv.start) + 3, sy + slotH - 2, iv.key, 'dim'));
  });
  y += lane.capacity * slotH + 6;
  // rollout 三阶段
  const rowH = 7;
  const groupPitch = rowH * g;
  lane.rollouts.forEach((r) => {
    const ry = y + r.id * rowH;
    parts.push(line(tx(0), ry + rowH / 2, tx(tMax), ry + rowH / 2, 'gl'));
    // 排队：提交到首个 tick
    if (r.startedAt !== null && r.startedAt > r.submittedAt) parts.push(rect(tx(r.submittedAt), ry + 1, (r.startedAt - r.submittedAt) * px, rowH - 2, 'ghost', 1));
    // 推理 tick：按训练时的 staleness 着色
    const trainVersion = lane.batches[r.batch].version;
    let tick = 0;
    let cursor = r.startedAt;
    // 逐相重建 tick 时间：tick 只在推理相里发生
    for (const ph of lane.phases) {
      if (cursor === null) break;
      let t = Math.max(cursor, ph.start);
      while (t < ph.end && tick < r.versions.length) {
        const stale = trainVersion - r.versions[tick];
        const cls = stale <= 0 ? 'bar1' : stale === 1 ? 'bar2' : 'bar';
        parts.push(rect(tx(t), ry + 1, px, rowH - 2, cls, 0));
        t += 1;
        tick += 1;
      }
      cursor = t;
    }
    if (tick !== r.versions.length) throw new Error(`rollout ${r.id} 的 tick 重建与仿真不一致`);
    // 装配：组内全部推理完到 assembled
    const grp = lane.groups[r.group];
    if (grp.assembledAt > grp.inferredAt) parts.push(rect(tx(grp.inferredAt), ry + 1, (grp.assembledAt - grp.inferredAt) * px, rowH - 2, 'neutral', 1));
    // 归还点
    if (lane.releaseOn === 'inferred') parts.push(circle(tx(r.inferredAt), ry + rowH / 2, 2.4, 'acc1fill'));
    if (lane.releaseOn === 'assembled' && r.id % g === 0) parts.push(circle(tx(grp.assembledAt), ry + rowH / 2, 2.4, 'acc1fill'));
    if (lane.releaseOn === 'consumed' && r.id % (n * g) === 0 && lane.batches[r.batch].releaseAt !== null) parts.push(circle(tx(lane.batches[r.batch].releaseAt), ry + rowH / 2, 2.4, 'acc1fill'));
  });
  lane.groups.forEach((grp) => {
    const gy = y + grp.id * groupPitch + groupPitch / 2 + 4;
    parts.push(text(x0 + labelW - 8, gy, `b${grp.batch} g${grp.id}`, 'sm', 'end'));
  });
  y += lane.rollouts.length * rowH + 4;
  // trainer 行
  parts.push(text(x0 + labelW - 8, y + 12, 'trainer', 'rank', 'end'));
  lane.batches.forEach((b) => {
    parts.push(rect(tx(b.trainStart), y + 1, (b.trainEnd - b.trainStart) * px, 14, 'train', 2));
    parts.push(text(tx(b.trainStart) + (b.trainEnd - b.trainStart) * px / 2, y + 11, `train b${b.id}`, 'sm', 'middle'));
    parts.push(line(tx(b.yieldedAt), y - 2, tx(b.yieldedAt), y + 16, 'sep2'));
  });
  y += 20;
  // 时间刻度
  for (let t = 0; t <= tMax; t += 5) {
    parts.push(line(tx(t), y, tx(t), y + 3, 'sep'));
    parts.push(text(tx(t), y + 13, String(t), 'sm', 'middle'));
  }
  return y + 22;
}

function renderPipeline() {
  const W = 1272;
  const X0 = 28;
  const labelW = 74;
  const timeW = W - 2 * X0 - labelW;
  const tMax = Math.max(...LANES.map((l) => l.makespan));
  const parts = header(
    W,
    '图 1　rollout 流水线与提交闸门：同一个 rl_generation_lag 在 B / G / R 三种提交粒度下各买到什么',
    [
      `算例：grpo_prompts_per_step=${PIPE_CASE.nPrompts}，grpo_group_size=${PIPE_CASE.groupSize}，${PIPE_CASE.batches} 个 trainer batch = ${PIPE_CASE.durations.length} 条 rollout，推理时长 [${PIPE_CASE.durations.join(',')}]，装配 a=${PIPE_CASE.assemble}，训练 T=${PIPE_CASE.train}（示意单位）`,
      '浅蓝带 = 推理相（引擎跑，事件循环只由 rank 0 驱动），灰带 = 训练相（引擎 suspend，policy version +1）；gate 条 = 每个 slot 被哪个提交单元占着（灰 = 最后一批等不到下一次 anext）',
      '每行一条 rollout：灰 = 提交后排队，tick 颜色 = 该 token 在训练时的 staleness（浅蓝 0、中蓝 1、橙 ≥ 2），白 = 装配；蓝点 = 闸门归还（画在提交单元第一行）',
      '橙虚线 = rank 0 取满 n 组的时刻：之后才 suspend 引擎、broadcast_object_list 到全 rank —— slot 归还与全 rank 可见是两个不同的完成边界',
    ],
  );
  let y = 126;
  LANES.forEach((lane, i) => {
    y = renderLane(parts, lane, { x0: X0, y0: y, labelW, timeW, tMax, where: `fig1/lane${i}` });
    y += 6;
  });
  // 底部盒子：两个满幅盒子上下叠放，每行短标签
  const BW = W - 2 * X0;
  const BY = y + 4;
  const capRows = CAPACITY.map((row) => `${row.submission}：归还于 ${row.releaseOn}；每 slot ${row.perUnit} 条 rollout；capacity = ${row.capacity[0]}（lag=0）/ ${row.capacity[1]}（lag=1）；infer workers = capacity × 每 slot 条数 = ${row.capacity[1] * row.perUnit}（lag=1，streaming）`);
  const tableH = 47 + 16 * 7 + 6;
  parts.push(
    infoBox(X0, BY, BW, tableH, 'R / G / B × 归还状态 × capacity（rollout_granularity.py::get_rl_parallel_generation_tasks · RELEASE_STATE_BY_SUBMISSION · _RolloutPipeline.__init__）', [
      { text: `capacity = (lag + 1) × [submission ≠ B: grpo_prompts_per_step] × [submission = R: grpo_group_size]；算例 n=${PIPE_CASE.nPrompts}, g=${PIPE_CASE.groupSize}`, cls: 'dim' },
      ...capRows,
      `test_rl_utils.py::test_get_rl_parallel_generation_tasks 锁定的 (n=8, g=4)：B ${CAPACITY_TEST[0].capacity.join(' / ')}，G ${CAPACITY_TEST[1].capacity.join(' / ')}，R ${CAPACITY_TEST[2].capacity.join(' / ')}（lag=0 / lag=2）`,
      `prevent_dataset_reorder == (consumption == "B")：四条 lane 都是 B-consume，stage_consume 攒齐 num_groups_per_batch 组、按 index_in_batch 排序后整批 yield`,
      '被拒绝的组合：B-submit + G-consume（_GranularityConfig._validate）、consume R（validate_args）、lag>0 或 R-submit 而无 --rl-partial-rollouts（validate_args）',
    ], 'neutral', 'fig1/table'),
  );
  const CY2 = BY + tableH + 10;
  const conclH = 47 + 16 * 5 + 6;
  parts.push(
    infoBox(X0, CY2, BW, conclH, '仿真结论（脚本算出；时长是示意单位，发起点与归还点来自冻结源码）', [
      { text: `同步基线 makespan ${LANE_SYNC.makespan}、trainer 等待 ${LANE_SYNC.trainerWait}；lag=1 的 B / G / R：makespan ${LANE_B.makespan} / ${LANE_G.makespan} / ${LANE_R.makespan}，等待 ${LANE_B.trainerWait} / ${LANE_G.trainerWait} / ${LANE_R.trainerWait}，峰值在途 ${LANE_B.peakInflight} / ${LANE_G.peakInflight} / ${LANE_R.peakInflight}（同步基线 ${LANE_SYNC.peakInflight}）`, cls: 'dim' },
      { text: `B：slot 到 consumed 才还，b(k+2) 只能在相 k+1 开头进闸 → 最旧 token staleness ≤ lag（本例 ${LANE_B.maxOldest}）`, cls: 'dim' },
      { text: `G / R：slot 在 assembled / inferred 就还，归还不再等训练消费，后面的批在相 k 里就进闸 → 最旧 token staleness 由推理时长与 capacity 决定，本例 G ${LANE_G.maxOldest}、R ${LANE_R.maxOldest}（lag=1）；lag=0 时 G / R 也到 ${LANE_G0.maxOldest} / ${LANE_R0.maxOldest}`, cls: 'costtx' },
      { text: 'capacity 只限制「未到归还状态的提交单元」数：lag 是 B 提交下的精确版本差上限，G / R 只把 lag 换算成更细的槽位，并没有把上限一起换算过去', cls: 'costtx' },
      'slot 归还 ≠ 全 rank 可见：rank 0 取满 n 组（橙虚线）→ 退出推理相（suspend）→ broadcast_object_list 才让其它 rank 拿到这批',
    ], 'neutral', 'fig1/conclusion'),
  );
  const H = CY2 + conclH + 20;
  return seal(parts, W, H, 'fig1-pipeline');
}

// ============================================================================
// 图 2：打包
// ============================================================================

function renderPacking() {
  const W = 1272;
  const X0 = 28;
  const P = PACK;
  const parts = header(
    W,
    '图 2　RL 序列打包：变长 rollout 怎样进 THD 箱、箱怎样分到 rank、微批数为什么会变、优势为什么不能广播',
    [
      `算例：${P.lengths.length} 条序列长度 {${P.lengths.join(',')}}（共 ${P.totalTokens} token），bin_size = seq_length = ${P.binSize}，DP=${P.dp}，max_sequences_per_bin=${P.maxSequencesPerBin}（CLI 默认），micro_batch_size=${P.microBatchSize}`,
      `samples_ratio_per_step=${P.samplesRatioPerStep}（global_batch_size = prompts × group_size）；每个数字由复刻自冻结基线 sequence_packing_utils.py 与 rl_utils.py 规则的脚本算出`,
    ],
  );
  // ---- 面板 A ----
  const AY = 106;
  parts.push(text(X0, AY, 'A　prepare_trajectories 之后：每条序列各自 pad 到 seq_length，训练矩阵是 10 × 16', 'pt'));
  const CELL = 6; // 面板 A 按 1/2 比例缩画：每 token 6px，一行 16 token = 96px，10 行并排放得下
  let ax = X0;
  P.lengths.forEach((len, idx) => {
    parts.push(rect(ax, AY + 8, len * CELL, 16, 'r1', 2));
    parts.push(rect(ax + len * CELL, AY + 8, (P.binSize - len) * CELL, 16, 'ghost', 2));
    parts.push(text(ax + 3, AY + 20, guard(`s${idx}=${len}`, 10.5, P.binSize * CELL - 4, `fig2/A/s${idx}`), 'sm'));
    ax += P.binSize * CELL + 22;
  });
  const padA = P.lengths.length * P.binSize - P.totalTokens;
  parts.push(text(X0, AY + 42, guard(`每条独占一行：${P.lengths.length} × ${P.binSize} = ${P.lengths.length * P.binSize} 格里 ${P.totalTokens} 格是真 token，pad ${padA} 格（${pct(P.totalTokens / (P.lengths.length * P.binSize))} 有效）；条形按 1/2 比例缩画，灰 = pad`, 11, W - 2 * X0, 'fig2/capA'), 'cap'));

  // ---- 面板 B ----
  const BY = AY + 66;
  parts.push(text(X0, BY, `B　SequencePacker.pack_sequences：按长度降序，放不下就开新箱（next-fit）→ ${P.numBins} 个箱`, 'pt'));
  const BC = 22;
  const binRow = (bins, fill, x, y, cls, where) => {
    bins.forEach((bin, b) => {
      const by = y + b * 26;
      parts.push(text(x - 6, by + 15, `bin ${b}`, 'rank', 'end'));
      let cx = x;
      bin.forEach((idx) => {
        const len = P.lengths[idx];
        parts.push(rect(cx, by, len * BC, 22, cls, 2));
        const label = fitLabel([`s${idx}·${len}`, `s${idx}`], 10.5, len * BC - 2);
        if (label) parts.push(text(cx + len * BC / 2, by + 15, label, 'sm', 'middle'));
        cx += len * BC;
      });
      const padCells = P.binSize - fill[b];
      if (padCells > 0) {
        parts.push(rect(cx, by, padCells * BC, 22, 'ghost', 2));
        const label = fitLabel([`pad ${padCells}`, `${padCells}`], 10.5, padCells * BC - 2);
        if (label) parts.push(text(cx + padCells * BC / 2, by + 15, label, 'sm', 'middle'));
      }
    });
  };
  const BX = X0 + 44;
  const BROW = BY + 30;
  binRow(P.packing.bins, P.binFill, BX, BROW, 'r1', 'fig2/nextfit');
  const nextFitH = P.numBins * 26;
  // first-fit 对照
  const FX = BX + P.binSize * BC + 80;
  parts.push(text(FX - 6 - 30, BROW - 8, '对照：docstring 所说的 first-fit（分析重建）', 'costtx'));
  const ffdFill = P.ffd.bins.map((bin) => bin.reduce((s, idx) => s + P.lengths[idx], 0));
  binRow(P.ffd.bins, ffdFill, FX, BROW, 'bar', 'fig2/ffd');
  parts.push(text(FX, BROW + P.ffd.bins.length * 26 + 14, guard(`${P.ffd.bins.length} 个箱，${pct(P.ffdEfficiency)} 有效；代码没有回头看旧箱，所以 s${P.packing.bins[1][0]} 之后的 s${P.packing.sortedIndices[2]} 不会填进 bin 0`, 11, W - X0 - FX, 'fig2/capFFD'), 'cap'));
  const capB = `装填顺序 = sorted(lengths, reverse=True) = [${P.packing.sortedIndices.map((i) => `s${i}`).join(',')}]；bin ${ADV_BIN_INDEX} 的 seq_starts = [${P.packing.seqStarts[ADV_BIN_INDEX].join(',')}]（末位是终点）；不算空箱的朴素效率 ${P.totalTokens} / (${P.numBins} × ${P.binSize}) = ${pct(P.rawEfficiency)}`;
  parts.push(text(X0, BROW + nextFitH + 14, guard(capB, 11, W - 2 * X0, 'fig2/capB'), 'cap'));

  // ---- 面板 C ----
  const CY = BROW + nextFitH + 40;
  parts.push(text(X0, CY, `C　distribute_packed_bins → update_microbatch_calculator：${P.numBins} 个箱分到 ${P.dp} 个 rank，补空箱到 max_bins_per_rank = ${P.fifo.maxBinsPerRank}`, 'pt'));
  const cols = [X0, X0 + 135, X0 + 350, X0 + 565, X0 + 1040];
  const colW = (i) => (i + 1 < cols.length ? cols[i + 1] - cols[i] - 10 : W - X0 - 4 - cols[i]);
  const cell = (i, y, value, cls) => parts.push(text(cols[i], y, guard(value, cls === 'rank' ? 11 : 10.5, colW(i), `fig2/C/col${i}`), cls));
  const TY = CY + 12;
  parts.push(rect(X0 - 4, TY, W - 2 * X0 + 8, 22 * 5 + 10, 'panel'));
  ['分发算法', 'rank0 的箱', 'rank1 的箱', '每 rank 效率（log_packing_efficiency 口径）', 'bins_bs → 微批数'].forEach((h, i) => cell(i, TY + 17, h, 'rank'));
  parts.push(line(X0, TY + 23, W - X0 - 4, TY + 23));
  const rankStr = (r) => `[${r.bins.join(',')}]${r.emptyBins ? ` + ${r.emptyBins} 空箱` : ''}`;
  [['fifo（默认）', P.fifo, P.perRank.fifo], ['round-robin', P.rr, P.perRank.rr]].forEach(([name, dist, eff], k) => {
    const y = TY + 40 + k * 22;
    cell(0, y, name, 'sm');
    cell(1, y, rankStr(dist.ranks[0]), dist.ranks[0].emptyBins ? 'costtx' : 'sm');
    cell(2, y, rankStr(dist.ranks[1]), dist.ranks[1].emptyBins ? 'costtx' : 'sm');
    cell(3, y, `${pct(eff[0].efficiency)} / ${pct(eff[1].efficiency)}（${eff[0].myTokens} / ${eff[1].myTokens} token ÷ ${eff[0].capacity}）`, 'sm');
    cell(4, y, `${P.micro.binsBs} → ${P.micro.numMicrobatches}`, 'dim');
  });
  const y3 = TY + 40 + 2 * 22;
  cell(0, y3, 'unpacked（对照）', 'sm');
  cell(1, y3, `${P.lengths.length / P.dp} 条序列 / rank，各自一行`, 'sm');
  cell(3, y3, `get_packing_efficiency 口径（含空箱、全 DP）：${P.totalTokens} / (${P.fifo.maxBinsPerRank} × ${P.binSize} × ${P.dp}) = ${pct(P.efficiency)}`, 'sm');
  cell(4, y3, `${P.unpacked.globalBatchSize} → ${P.unpacked.numMicrobatches}`, 'costtx');
  const y4 = y3 + 22;
  cell(0, y4, '第二算例（更长的一批）', 'sm');
  // 第二算例的长度列表横跨 rank0 / rank1 两列（该行这两列没有别的内容）
  parts.push(text(cols[1], y4, guard(`{${PACK_LONG.lengths.join(',')}} → ${PACK_LONG.numBins} 个箱 → 每 rank ${PACK_LONG.fifo.maxBinsPerRank}`, 10.5, cols[3] - cols[1] - 10, 'fig2/C/row4'), 'sm'));
  cell(3, y4, `微批数 ${P.micro.numMicrobatches} → ${PACK_LONG.micro.numMicrobatches} → 回到短的一批又是 ${P.micro.numMicrobatches}：training.py「只增不减」断言为此让路`, 'costtx');
  cell(4, y4, `${PACK_LONG.micro.binsBs} → ${PACK_LONG.micro.numMicrobatches}`, 'dim');
  const capC = `微批数 = bins_bs ÷ (micro_batch_size × DP)，bins_bs = ceil(ratio × 本 rank 箱数) × DP；unpacked 用 ceil(ratio × 序列数) 除同一分母；空箱 loss_mask 全 0，只为让各 rank 迭代次数一致`;
  parts.push(text(X0, TY + 22 * 5 + 30, guard(capC, 11, W - 2 * X0, 'fig2/capC'), 'cap'));

  // ---- 面板 D ----
  const DY = TY + 22 * 5 + 56;
  const bin = P.packing.bins[ADV_BIN_INDEX];
  parts.push(text(X0, DY, `D　calculate_grpo_loss 在 packed 布局下：bin ${ADV_BIN_INDEX} 装了 s${bin[0]}（${P.lengths[bin[0]]}）与 s${bin[1]}（${P.lengths[bin[1]]}），优势按 seq_starts / seq_lengths 逐条铺开`, 'pt'));
  const DC = 24;
  const DX = X0 + 130;
  const rowD = (label, y, cellsFn, cls) => {
    parts.push(text(DX - 8, y + 15, label, 'rank', 'end'));
    for (let p = 0; p < P.binSize - 1; p += 1) {
      const { txt, cls: c } = cellsFn(p);
      parts.push(rect(DX + p * DC, y, DC - 1, 22, c ?? cls, 2));
      parts.push(text(DX + p * DC + (DC - 1) / 2, y + 15, txt, 'sm', 'middle'));
    }
  };
  const starts = P.packing.seqStarts[ADV_BIN_INDEX];
  const tokenOwner = (p) => (p < starts[1] ? bin[0] : p < starts[2] ? bin[1] : null);
  rowD('token 位置', DY + 10, (p) => {
    const o = tokenOwner(p);
    return { txt: o === null ? '·' : `s${o}`, cls: o === null ? 'ghost' : o === bin[0] ? 'r0' : 'r1' };
  });
  rowD('packed 优势', DY + 38, (p) => {
    const v = ADV.values[p];
    return { txt: v === 0 ? '0' : v, cls: ADV.owner[p] === null ? 'ghost' : ADV.owner[p] === bin[0] ? 'bar1' : 'bar2' };
  });
  rowD('广播（被否）', DY + 66, (p) => {
    const o = tokenOwner(p);
    const leaked = o === bin[1];
    return { txt: `A${bin[0]}`, cls: leaked ? 'bar' : o === null ? 'ghost' : 'bar1' };
  });
  parts.push(text(DX + (P.binSize - 1) * DC + 10, DY + 25, `logprob 位置 0..${P.binSize - 2}（比 token 少 1）`, 'sm'));
  parts.push(text(DX + (P.binSize - 1) * DC + 10, DY + 53, `end = min(start + len − 1, bin_size)`, 'dim'));
  parts.push(text(DX + (P.binSize - 1) * DC + 10, DY + 81, `advantages.view(-1, 1) 会把 A${bin[0]} 串给 s${bin[1]}`, 'costtx'));
  const capD = `packed 时 advantages 形状是 [该箱序列数]，unpacked 时是 [batch]；位置 ${starts[1] - 1}（s${bin[0]} 末 token 预测 s${bin[1]} 首 token）与 ${starts[2] - 1} 留 0，本来就被 loss_mask 掩掉；pack_inference_logprobs 用同一规则对齐 IS 权重`;
  parts.push(text(X0, DY + 108, guard(capD, 11, W - 2 * X0, 'fig2/capD'), 'cap'));

  // ---- 底部盒子：两个满幅盒子上下叠放 ----
  const OY = DY + 126;
  const OW = W - 2 * X0;
  const rulesH = 47 + 16 * 5 + 6;
  parts.push(
    infoBox(X0, OY, OW, rulesH, '本图复刻的规则（sequence_packing_utils.py · rl_utils.py）', [
      'pack_sequences：get_actual_sequence_lengths 按 pad 反推真实长度 → 按长度降序 → 当前箱装得下且未到 max_sequences_per_bin 就放，否则开新箱（从不回头看旧箱）',
      '每箱 pad 到 bin_size = seq_length；position_ids 逐条从 0 起；loss_mask = 1 再乘 generation_mask；PackedSeqParams 的 cu_seqlens 补到 max_sequences_per_bin + 2 项以固定形状',
      'distribute_packed_bins：fifo 连续切（多出的箱给前面的 rank）/ round-robin 跨步取；不足 max_bins_per_rank 的 rank 补 pad 箱（loss_mask 全 0）',
      'update_microbatch_calculator：reconfigure_num_microbatches_calculator(global_batch_size = bins_bs)，训练主循环因此看到微批数可增可减',
      'calculate_grpo_loss packed 分支：优势按 seq_starts 定位、len − 1 截尾逐条铺开；unpacked 分支走 advantages.view(-1, 1) 广播；pack_inference_logprobs 用同一定位规则对齐 IS 权重',
    ], 'neutral', 'fig2/rules'),
  );
  const SY = OY + rulesH + 10;
  const scopeH = 47 + 16 * 4 + 6;
  parts.push(
    infoBox(X0, SY, OW, scopeH, '简化与对照', [
      { text: `first-fit 对照是分析重建：源码 docstring 写 "greedy first-fit"，代码从不回看旧箱（next-fit）；本算例 next-fit ${P.numBins} 箱 vs first-fit-decreasing ${P.ffd.bins.length} 箱`, cls: 'costtx' },
      '不画 pack_all_trajectories 里先 all_gather 把各 rank 的 trajs 拼成全局再打包的那一步，也不画 generation_mask 与 loss_mask 的相乘、old / ref logprob 的 [num_bins, bin_size − 1] 张量',
      `效率两种口径：get_packing_efficiency（全 DP 真 token ÷ 含空箱的总容量）${pct(P.efficiency)}；log_packing_efficiency（本 rank 真 token ÷ 本 rank 含空箱容量）见面板 C`,
      '第二算例只用来证明微批数可增可减，长度是随意选的',
    ], 'neutral', 'fig2/scope'),
  );
  const H = SY + scopeH + 20;
  return seal(parts, W, H, 'fig2-packing');
}

// ============================================================================

const outputs = new Map([
  ['megatron_rl_runtime_pipeline.svg', renderPipeline()],
  ['megatron_rl_runtime_packing.svg', renderPacking()],
]);

export {
  PIPE_CASE, PACK_CASE, PACK_CASE_LONG,
  RELEASE_STATE_BY_SUBMISSION, getRlParallelGenerationTasks, validateGranularity, rolloutsPerSubmissionUnit, numInferWorkers,
  simulatePipeline, LANE_SYNC, LANE_B0_STREAM, LANE_B, LANE_G, LANE_R, LANE_G0, LANE_R0, LANES, CAPACITY, CAPACITY_TEST, capacityTable,
  packSequences, firstFitDecreasing, distributePackedBins, updateMicrobatchCalculator, unpackedMicrobatches,
  packingEfficiency, perRankEfficiency, packedAdvantages, runPacking, PACK, PACK_LONG, ADV, ADV_BIN_INDEX,
  pct, outputs,
};

const invokedDirectly = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (invokedDirectly) {
  const here = dirname(fileURLToPath(import.meta.url));
  const outDir = process.argv[2] ?? join(here, '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks', 'megatron-lm', 'assets');
  mkdirSync(outDir, { recursive: true });
  for (const [name, svg] of outputs) {
    writeFileSync(join(outDir, name), svg, 'utf8');
    console.log(`${name}  ${svg.length} bytes`);
  }
  for (const lane of [LANE_SYNC, LANE_B, LANE_G, LANE_R, LANE_G0, LANE_R0]) {
    console.log('图 1：', lane.name, JSON.stringify({ capacity: lane.capacity, workers: lane.workers, makespan: lane.makespan, trainerWait: lane.trainerWait, engineIdle: lane.engineIdle, peakInflight: lane.peakInflight, peakHeld: lane.peakHeld, staleness: lane.staleness }));
  }
  console.log('图 2：', JSON.stringify({ bins: PACK.packing.bins, fill: PACK.binFill, ffd: PACK.ffd.bins.length, fifo: PACK.fifo.ranks, rr: PACK.rr.ranks, micro: PACK.micro, unpacked: PACK.unpacked, efficiency: PACK.efficiency, long: PACK_LONG.micro }));
}
