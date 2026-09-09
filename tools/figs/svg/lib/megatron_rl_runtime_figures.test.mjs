// 锁住 Megatron RL 运行时图示的可执行契约：闸门 capacity 换算、三阶段流水线仿真的 makespan /
// 峰值 / staleness、序列打包在算例上的逐格结果、两种分发与微批数、packed 优势铺开，全部由复刻自
// 冻结基线（NVIDIA/Megatron-LM@85902ef）的规则算出，并与 33_megatron_rl_runtime_analysis.md 正文
// 引用的数值逐个对齐 —— 图和正文不许各写各的。
//
// 运行：node --test tools/figs/svg/lib/megatron_rl_runtime_figures.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

import {
  PIPE_CASE, PACK_CASE, PACK_CASE_LONG,
  RELEASE_STATE_BY_SUBMISSION, getRlParallelGenerationTasks, validateGranularity, rolloutsPerSubmissionUnit, numInferWorkers,
  simulatePipeline, LANE_SYNC, LANE_B0_STREAM, LANE_B, LANE_G, LANE_R, LANE_G0, LANE_R0, LANES, CAPACITY, CAPACITY_TEST,
  packSequences, firstFitDecreasing, distributePackedBins, updateMicrobatchCalculator, unpackedMicrobatches,
  packedAdvantages, PACK, PACK_LONG, ADV, ADV_BIN_INDEX, pct,
} from '../megatron_rl_runtime_figures.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const generator = join(here, '..', 'megatron_rl_runtime_figures.mjs');
const assetDir = join(
  here, '..', '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks',
  'megatron-lm', 'assets',
);
const pagePath = join(
  here, '..', '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks',
  'megatron-lm', '33_megatron_rl_runtime_analysis.md',
);

const NAMES = ['megatron_rl_runtime_pipeline.svg', 'megatron_rl_runtime_packing.svg'];

const FONT = { ti: 18, su: 11.5, pt: 14, tx: 12, sm: 10.5, dim: 10.5, costtx: 10.5, rank: 11, cap: 11 };

function textWidth(value, fontSize) {
  let units = 0;
  for (const ch of String(value)) units += ch.charCodeAt(0) < 0x7f ? 0.6 : 1;
  return units * fontSize;
}

function parseCanvas(svg, name) {
  const m = svg.match(/viewBox="0 0 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)"/);
  assert.ok(m, `${name}: SVG 必须声明 viewBox`);
  return { w: Number(m[1]), h: Number(m[2]) };
}

function assertInsideCanvas(svg, name) {
  const { w, h } = parseCanvas(svg, name);
  assert.equal(w, 1272, `${name}: viewBox 宽度须为 1272`);
  for (const [, x, y, rw, rh] of svg.matchAll(
    /<rect[^>]*?x="(-?\d+(?:\.\d+)?)"[^>]*?y="(-?\d+(?:\.\d+)?)"[^>]*?width="(\d+(?:\.\d+)?)"[^>]*?height="(\d+(?:\.\d+)?)"/g,
  )) {
    assert.ok(Number(x) >= -2, `${name}: rect 左边越界 x=${x}`);
    assert.ok(Number(y) >= -2, `${name}: rect 上边越界 y=${y}`);
    assert.ok(Number(x) + Number(rw) <= w + 2, `${name}: rect 右边越界 ${x}+${rw} > ${w}`);
    assert.ok(Number(y) + Number(rh) <= h + 2, `${name}: rect 下边越界 ${y}+${rh} > ${h}`);
  }
  for (const [, cx, cy] of svg.matchAll(/<circle[^>]*?cx="(-?\d+(?:\.\d+)?)" cy="(-?\d+(?:\.\d+)?)"/g)) {
    assert.ok(Number(cx) >= 0 && Number(cx) <= w && Number(cy) >= 0 && Number(cy) <= h, `${name}: circle 越界 (${cx}, ${cy})`);
  }
  for (const [, x, y] of svg.matchAll(/<text[^>]*?x="(-?\d+(?:\.\d+)?)" y="(-?\d+(?:\.\d+)?)"/g)) {
    assert.ok(Number(x) >= 0 && Number(x) <= w, `${name}: text x=${x} 越界`);
    assert.ok(Number(y) >= 0 && Number(y) <= h, `${name}: text y=${y} 越界`);
  }
}

// 文字盒之间不重叠、文字盒不出画布（越界不等于不重叠，两条都要）
function assertNoTextOverlap(svg, name) {
  const { w: canvasW, h: canvasH } = parseCanvas(svg, name);
  const boxes = [];
  for (const m of svg.matchAll(
    /<text class="([a-z0-9]+)" x="(-?[\d.]+)" y="(-?[\d.]+)" text-anchor="(\w+)">([^<]*)<\/text>/g,
  )) {
    const [, cls, xs, ys, anchor, raw] = m;
    const size = FONT[cls];
    if (!size || raw.trim() === '') continue;
    const tw = textWidth(raw.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>'), size);
    const x = anchor === 'middle' ? Number(xs) - tw / 2 : anchor === 'end' ? Number(xs) - tw : Number(xs);
    boxes.push({ x, y: Number(ys) - size * 0.78, w: tw, h: size * 1.06, raw });
  }
  for (const b of boxes) {
    assert.ok(
      b.x >= -1 && b.y >= -1 && b.x + b.w <= canvasW + 1 && b.y + b.h <= canvasH + 1,
      `${name}: 文字盒出画布 "${b.raw}"`,
    );
  }
  for (let i = 0; i < boxes.length; i += 1) {
    for (let j = i + 1; j < boxes.length; j += 1) {
      const a = boxes[i];
      const b = boxes[j];
      const dx = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
      const dy = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
      assert.ok(!(dx > 1 && dy > 1), `${name}: 文字重叠 "${a.raw}" × "${b.raw}"`);
    }
  }
}

// ============================================================================
// (a) 复刻器对源码语义的锁定
// ============================================================================

test('get_rl_parallel_generation_tasks：算例 (n=2, g=2) 与 test_rl_utils.py 锁定的 (n=8, g=4)', () => {
  // tests/unit_tests/rl/test_rl_utils.py::test_get_rl_parallel_generation_tasks
  const cases = [['B', 0, 1], ['B', 2, 3], ['G', 0, 8], ['G', 2, 24], ['R', 0, 32], ['R', 2, 96]];
  for (const [sub, lag, expected] of cases) {
    assert.equal(
      getRlParallelGenerationTasks({ rl_submission_granularity: sub, rl_generation_lag: lag, grpo_prompts_per_step: 8, grpo_group_size: 4 }),
      expected,
      `${sub} lag=${lag}`,
    );
  }
  assert.deepEqual(CAPACITY_TEST.map((r) => r.capacity), [[1, 3], [8, 24], [32, 96]]);
  // 本页算例
  assert.deepEqual(CAPACITY.map((r) => [r.submission, r.releaseOn, r.perUnit, ...r.capacity]), [
    ['B', 'consumed', 4, 1, 2],
    ['G', 'assembled', 2, 2, 4],
    ['R', 'inferred', 1, 4, 8],
  ]);
  assert.deepEqual(RELEASE_STATE_BY_SUBMISSION, { R: 'inferred', G: 'assembled', B: 'consumed' });
  // _RolloutPipeline.__init__：workers = capacity × 每提交单元条数；非 streaming 再 min 到 num_groups × rollouts_per_group
  assert.equal(numInferWorkers({ capacity: 2, submission: 'B', n: 2, g: 2, streaming: true }), 8);
  assert.equal(numInferWorkers({ capacity: 1, submission: 'B', n: 2, g: 2, streaming: false }), 4);
  assert.equal(numInferWorkers({ capacity: 8, submission: 'R', n: 2, g: 2, streaming: true }), 8);
  assert.equal(rolloutsPerSubmissionUnit('G', 2, 2), 2);
});

test('validate_args 与 _GranularityConfig._validate 的拒绝面', () => {
  // tests/unit_tests/rl/test_rl_utils.py::test_rl_granularity_validation_rejects_unsupported_modes
  assert.throws(() => validateGranularity({ lag: 1, submission: 'B', consumption: 'B', partialRollouts: false }), /requires --rl-partial-rollouts/);
  assert.throws(() => validateGranularity({ lag: 0, submission: 'R', consumption: 'B', partialRollouts: false }), /requires streaming grouped rollouts/);
  assert.throws(() => validateGranularity({ lag: 0, submission: 'B', consumption: 'R', partialRollouts: true }), /not currently supported/);
  assert.throws(() => validateGranularity({ lag: 0, submission: 'B', consumption: 'G', partialRollouts: true }), /Batch submission with group consumption/);
  // tests/unit_tests/rl/test_grouped_rollouts.py::test_filter_groups_with_same_reward_rejected
  assert.throws(() => validateGranularity({ lag: 0, submission: 'G', consumption: 'G', partialRollouts: true, filterGroupsWithSameReward: true }), /filter_groups_with_same_reward/);
  // prevent_dataset_reorder 精确等于 consumption == "B"
  assert.equal(validateGranularity({ lag: 0, submission: 'G', consumption: 'B', partialRollouts: true }).preventDatasetReorder, true);
  assert.equal(validateGranularity({ lag: 0, submission: 'G', consumption: 'G', partialRollouts: true }).preventDatasetReorder, false);
});

test('流水线仿真：同步基线、lag=1 的 B / G / R 与 lag=0 的 G / R', () => {
  assert.equal(PIPE_CASE.durations.length, PIPE_CASE.batches * PIPE_CASE.nPrompts * PIPE_CASE.groupSize);
  // 同步基线：非 streaming 与 streaming 的 lag=0/B 给出同一条时间线
  assert.equal(LANE_SYNC.makespan, 50);
  assert.equal(LANE_SYNC.trainerWait, 26);
  assert.equal(LANE_SYNC.peakInflight, 4);
  assert.equal(LANE_SYNC.maxOldest, 0);
  assert.equal(LANE_B0_STREAM.makespan, LANE_SYNC.makespan);
  // lag=1
  assert.deepEqual([LANE_B.capacity, LANE_G.capacity, LANE_R.capacity], [2, 4, 8]);
  assert.deepEqual([LANE_B.workers, LANE_G.workers, LANE_R.workers], [8, 8, 8]);
  assert.deepEqual([LANE_B.makespan, LANE_G.makespan, LANE_R.makespan], [38, 38, 35]);
  assert.deepEqual([LANE_B.trainerWait, LANE_G.trainerWait, LANE_R.trainerWait], [14, 14, 11]);
  assert.deepEqual([LANE_B.peakInflight, LANE_G.peakInflight, LANE_R.peakInflight], [8, 8, 8]);
  assert.deepEqual([LANE_B.peakHeld, LANE_G.peakHeld, LANE_R.peakHeld], [2, 4, 8]);
  // B：最旧 token 的 staleness ≤ lag；G / R：归还不等训练消费，staleness 越过 lag
  assert.equal(LANE_B.maxOldest, 1);
  assert.equal(LANE_G.maxOldest, 2);
  assert.equal(LANE_R.maxOldest, 3);
  assert.deepEqual(LANE_B.staleness.map((s) => s.oldest), [0, 1, 1, 1]);
  assert.deepEqual(LANE_G.staleness.map((s) => s.oldest), [0, 1, 2, 2]);
  assert.deepEqual(LANE_R.staleness.map((s) => s.oldest), [0, 1, 2, 3]);
  // lag=0 的 G / R 也超过 0
  assert.equal(LANE_G0.maxOldest, 1);
  assert.equal(LANE_R0.maxOldest, 2);
  for (const lane of LANES) assert.ok(lane.peakHeld <= lane.capacity);
  // test_grouped_rollouts.py::test_rollout_submission_granularity_limits_inference_concurrency：R 提交下在途 ≤ capacity
  const r2 = simulatePipeline({ ...PIPE_CASE, lag: 0, submission: 'R' });
  assert.ok(r2.peakInflight <= r2.capacity);
  // B 归还发生在下一相开头（下一次 anext），slot 区间终点等于下一相的 phaseStart
  const b0 = LANE_B.slotIntervals.find((iv) => iv.key === 'B0');
  assert.equal(b0.end, LANE_B.phases[1].start);
});

test('SequencePacker.pack_sequences 复刻：降序 next-fit（test_sequence_packing_integration 与本页算例）', () => {
  // tests/unit_tests/rl/test_sequence_packing_utils.py::test_sequence_packing_integration：[4,3,5] 装进 1 个箱，顺序 [2,0,1]
  const t = packSequences([4, 3, 5], 16, 16);
  assert.deepEqual(t.bins, [[2, 0, 1]]);
  assert.deepEqual(t.seqStarts[0], [0, 5, 9, 12]);
  // 本页算例
  assert.deepEqual(PACK_CASE.lengths, [7, 3, 9, 2, 5, 6, 1, 8, 4, 12]);
  assert.deepEqual(PACK.packing.bins, [[9], [2], [7, 0], [5, 4, 8], [1, 3, 6]]);
  assert.deepEqual(PACK.binFill, [12, 9, 15, 15, 6]);
  assert.equal(PACK.numBins, 5);
  assert.equal(PACK.totalTokens, 57);
  assert.deepEqual(PACK.packing.seqStarts[2], [0, 8, 15]);
  assert.deepEqual(PACK.packing.seqToBin, [2, 4, 1, 4, 3, 3, 4, 2, 3, 0]);
  // max_sequences_per_bin 生效：上限 1 时每条各占一箱
  assert.equal(packSequences(PACK_CASE.lengths, 16, 1).bins.length, 10);
  // 对照：first-fit-decreasing 少用一个箱（分析重建）
  assert.equal(PACK.ffd.bins.length, 4);
  assert.deepEqual(firstFitDecreasing([12, 9, 8, 7, 6, 5, 4, 3, 3, 1], 16, 50).bins, [[0, 6], [1, 3], [2, 4, 9], [5, 7, 8]]);
});

test('distribute_packed_bins 与 update_microbatch_calculator 复刻', () => {
  assert.deepEqual(PACK.fifo.ranks.map((r) => [r.bins, r.emptyBins]), [[[0, 1, 2], 0], [[3, 4], 1]]);
  assert.deepEqual(PACK.rr.ranks.map((r) => [r.bins, r.emptyBins]), [[[0, 2, 4], 0], [[1, 3], 1]]);
  assert.equal(PACK.fifo.maxBinsPerRank, 3);
  // fifo：多出的箱给前面的 rank
  assert.deepEqual(distributePackedBins(7, 3, 'fifo').ranks.map((r) => r.bins), [[0, 1, 2], [3, 4], [5, 6]]);
  assert.deepEqual(distributePackedBins(7, 3, 'round-robin').ranks.map((r) => r.bins), [[0, 3, 6], [1, 4], [2, 5]]);
  // tests/unit_tests/rl/test_sequence_packing_utils.py::test_get_bins_bs_and_steps
  assert.equal(updateMicrobatchCalculator({ samplesRatioPerStep: 1.0, numBinsThisRank: 42, dp: 8, microBatchSize: 1 }).binsBs, 42 * 8);
  assert.equal(updateMicrobatchCalculator({ samplesRatioPerStep: 0.5, numBinsThisRank: 1, dp: 8, microBatchSize: 1 }).binsBs, 8);
  assert.equal(updateMicrobatchCalculator({ samplesRatioPerStep: 1 / 3, numBinsThisRank: 4, dp: 8, microBatchSize: 1 }).binsBs, 16);
  // 本页算例：微批数 5（unpacked）→ 3（packed）→ 5（更长的一批）
  assert.deepEqual([PACK.micro.binsBs, PACK.micro.numMicrobatches], [6, 3]);
  assert.deepEqual([PACK.unpacked.globalBatchSize, PACK.unpacked.numMicrobatches], [10, 5]);
  assert.deepEqual([PACK_LONG.numBins, PACK_LONG.fifo.maxBinsPerRank, PACK_LONG.micro.binsBs, PACK_LONG.micro.numMicrobatches], [9, 5, 10, 5]);
  assert.throws(() => unpackedMicrobatches({ samplesRatioPerStep: 1, totalTurns: 7, dp: 2, microBatchSize: 1 }), /整除/);
  // 两种效率口径
  assert.equal(pct(PACK.efficiency), '59.4%');
  assert.equal(pct(PACK.rawEfficiency), '71.3%');
  assert.equal(pct(PACK.ffdEfficiency), '89.1%');
  assert.deepEqual(PACK.perRank.fifo.map((r) => pct(r.efficiency)), ['75.0%', '43.8%']);
  assert.deepEqual(PACK.perRank.rr.map((r) => pct(r.efficiency)), ['68.8%', '50.0%']);
  assert.deepEqual(PACK.perRank.fifo.map((r) => [r.myTokens, r.capacity]), [[36, 48], [21, 48]]);
});

test('calculate_grpo_loss packed 分支：优势按 seq_starts / seq_lengths 铺开，len − 1 截尾', () => {
  assert.equal(ADV_BIN_INDEX, 2);
  const bin = PACK.packing.bins[ADV_BIN_INDEX];
  assert.deepEqual(bin, [7, 0]);
  // s7 占 0..6（8 − 1 = 7 个 logprob 位置），位置 7 留 0；s0 占 8..13，位置 14 留 0；15 之后在 bin_size − 1 之外
  assert.deepEqual(ADV.owner.slice(0, 16), [7, 7, 7, 7, 7, 7, 7, null, 0, 0, 0, 0, 0, 0, null, null]);
  assert.deepEqual(ADV.values.slice(0, 16), ['A7', 'A7', 'A7', 'A7', 'A7', 'A7', 'A7', 0, 'A0', 'A0', 'A0', 'A0', 'A0', 'A0', 0, 0]);
  // 单条恰好填满整箱时 end 被 bin_size 截住
  const full = packSequences([16], 16, 50);
  const advFull = packedAdvantages(full.bins[0], full, { 0: 'A' }, 16);
  assert.equal(advFull.owner.filter((o) => o === 0).length, 15);
});

// ============================================================================
// (b) 生成器同步产出两张图；regen 必须等于已跟踪的资产；图元不出画布、文字不重叠
// ============================================================================

test('生成器同步产出两张图，且图上的关键量与算例一致', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'megatron-rl-runtime-figures-'));
  const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
  assert.equal(run.status, 0, run.stderr);
  const [pipeline, packing] = await Promise.all(NAMES.map((name) => readFile(join(outputDir, name), 'utf8')));

  // ---- 图 1 ----
  assert.ok(pipeline.includes(`makespan ${LANE_SYNC.makespan} · trainer 等待 ${LANE_SYNC.trainerWait}`));
  assert.ok(pipeline.includes(`makespan ${LANE_B.makespan} / ${LANE_G.makespan} / ${LANE_R.makespan}`));
  assert.ok(pipeline.includes(`staleness 最旧 ≤ ${LANE_R.maxOldest}、最新 ≤ ${LANE_R.maxNewest}`));
  for (const row of CAPACITY) {
    assert.ok(pipeline.includes(`${row.submission}：归还于 ${row.releaseOn}；每 slot ${row.perUnit} 条 rollout；capacity = ${row.capacity[0]}（lag=0）/ ${row.capacity[1]}（lag=1）`), `图 1 缺 ${row.submission} 行`);
  }
  assert.ok(pipeline.includes('B 1 / 3，G 8 / 24，R 32 / 96'));
  for (const s of ['broadcast_object_list', 'prevent_dataset_reorder', 'get_rl_parallel_generation_tasks', 'RELEASE_STATE_BY_SUBMISSION', 'gate ×8']) {
    assert.ok(pipeline.includes(s), `图 1 缺 ${s}`);
  }
  // 每条 lane 的 rollout 行数 = 16，四条 lane
  assert.equal((pipeline.match(/>b3 g7</g) ?? []).length, 4);

  // ---- 图 2 ----
  assert.ok(packing.includes(`→ ${PACK.numBins} 个箱`));
  assert.ok(packing.includes(`${PACK.ffd.bins.length} 个箱，${pct(PACK.ffdEfficiency)} 有效`));
  assert.ok(packing.includes(`${PACK.totalTokens} / (${PACK.fifo.maxBinsPerRank} × ${PACK.binSize} × ${PACK.dp}) = ${pct(PACK.efficiency)}`));
  assert.ok(packing.includes(`${PACK.micro.binsBs} → ${PACK.micro.numMicrobatches}`) && packing.includes(`${PACK.unpacked.globalBatchSize} → ${PACK.unpacked.numMicrobatches}`));
  assert.ok(packing.includes(`bin ${ADV_BIN_INDEX} 装了 s7（8）与 s0（7）`));
  assert.ok(packing.includes('advantages.view(-1, 1) 会把 A7 串给 s0'));
  assert.equal((packing.match(/>pad \d+</g) ?? []).length, 4, '面板 B：next-fit 4 个可标注的 pad 段 + first-fit 1 个');
  for (const s of ['seq_starts', 'distribute_packed_bins', 'update_microbatch_calculator', 'round-robin', 'log_packing_efficiency']) {
    assert.ok(packing.includes(s), `图 2 缺 ${s}`);
  }

  const all = [pipeline, packing];
  for (const svg of all) assert.ok(!svg.includes('[['), '图上不允许漏出 wikilink 语法');
  NAMES.forEach((name, i) => {
    assertInsideCanvas(all[i], name);
    assertNoTextOverlap(all[i], name);
  });

  const tracked = await Promise.all(NAMES.map((name) => readFile(join(assetDir, name), 'utf8')));
  NAMES.forEach((name, i) => {
    assert.equal(
      tracked[i],
      all[i],
      `${name} 与生成器输出不一致：重跑 node tools/figs/svg/megatron_rl_runtime_figures.mjs`,
    );
  });
});

// ============================================================================
// (c) 页面正文引用的数值与图上一致
// ============================================================================

test('页面正文引用的数值与图上一致', async () => {
  // 只改正文、不改图，这个用例必须红。
  const page = await readFile(pagePath, 'utf8');

  // 共用算例
  assert.ok(page.includes(`grpo_prompts_per_step = ${PIPE_CASE.nPrompts}`) && page.includes(`grpo_group_size = ${PIPE_CASE.groupSize}`));
  assert.ok(page.includes(`[${PIPE_CASE.durations.join(', ')}]`), '正文缺推理时长序列');
  assert.ok(page.includes(`a = ${PIPE_CASE.assemble}`) && page.includes(`T = ${PIPE_CASE.train}`));
  assert.ok(page.includes(`{${PACK_CASE.lengths.join(', ')}}`) && page.includes(`bin_size = ${PACK_CASE.binSize}`) && page.includes(`DP = ${PACK_CASE.dp}`));

  // capacity 换算
  const cap = (i) => CAPACITY[i].capacity;
  assert.ok(page.includes(`B 提交 capacity 为 ${cap(0)[0]}（lag=0）/ ${cap(0)[1]}（lag=1），G 为 ${cap(1)[0]} / ${cap(1)[1]}，R 为 ${cap(2)[0]} / ${cap(2)[1]}`), '正文缺 capacity 表');
  assert.ok(page.includes(`${CAPACITY_TEST[0].capacity.join(' / ')}、${CAPACITY_TEST[1].capacity.join(' / ')}、${CAPACITY_TEST[2].capacity.join(' / ')}`), '正文缺 (8, 4) 的测试锁定值');

  // 仿真结论
  assert.ok(page.includes(`makespan ${LANE_SYNC.makespan}`) && page.includes(`trainer 等待 ${LANE_SYNC.trainerWait}`), '同步基线');
  assert.ok(page.includes(`makespan ${LANE_B.makespan} / ${LANE_G.makespan} / ${LANE_R.makespan}`), 'lag=1 三条 lane 的 makespan');
  assert.ok(page.includes(`等待 ${LANE_B.trainerWait} / ${LANE_G.trainerWait} / ${LANE_R.trainerWait}`), 'lag=1 三条 lane 的等待');
  assert.ok(page.includes(`峰值在途 rollout 从 ${LANE_SYNC.peakInflight} 变成 ${LANE_B.peakInflight}`), '峰值在途');
  assert.ok(page.includes(`最旧 token 的 staleness 最大为 ${LANE_B.maxOldest}`), 'B 的 staleness');
  assert.ok(page.includes(`G 到 ${LANE_G.maxOldest}、R 到 ${LANE_R.maxOldest}`), 'G / R 的 staleness');
  assert.ok(page.includes(`lag=0 时 G / R 也到 ${LANE_G0.maxOldest} / ${LANE_R0.maxOldest}`), 'lag=0 的 G / R');

  // 打包
  assert.ok(page.includes(`${PACK.numBins} 个箱`) && page.includes(`first-fit-decreasing 只需 ${PACK.ffd.bins.length} 个`));
  assert.ok(page.includes('bin 0 = {s9}，bin 1 = {s2}，bin 2 = {s7, s0}，bin 3 = {s5, s4, s8}，bin 4 = {s1, s3, s6}'), '正文缺逐箱结果');
  assert.ok(page.includes(`装填量 ${PACK.binFill.join(' / ')}`), '正文缺逐箱装填量');
  assert.ok(page.includes('rank0 拿 [0, 1, 2]，rank1 拿 [3, 4] 再补 1 个空箱') && page.includes('rank0 拿 [0, 2, 4]，rank1 拿 [1, 3] 再补 1 个空箱'));
  assert.ok(page.includes(`bins_bs = ${PACK.micro.binsBs}`) && page.includes(`${PACK.micro.numMicrobatches} 个微批`));
  assert.ok(page.includes(`unpacked 时是 ${PACK.unpacked.globalBatchSize} → ${PACK.unpacked.numMicrobatches}`));
  assert.ok(page.includes(`{${PACK_CASE_LONG.lengths.join(', ')}}`) && page.includes(`${PACK_LONG.micro.binsBs} → ${PACK_LONG.micro.numMicrobatches}`));
  assert.ok(page.includes(pct(PACK.efficiency)) && page.includes(pct(PACK.rawEfficiency)) && page.includes(pct(PACK.ffdEfficiency)));
  assert.ok(page.includes(`${pct(PACK.perRank.fifo[0].efficiency)} / ${pct(PACK.perRank.fifo[1].efficiency)}`) && page.includes(`${pct(PACK.perRank.rr[0].efficiency)} / ${pct(PACK.perRank.rr[1].efficiency)}`));
  assert.ok(page.includes(`bin ${ADV_BIN_INDEX} 装了 s7（8）与 s0（7）`) && page.includes('seq_starts = [0, 8, 15]') && page.includes('位置 7 与 14'));

  for (const name of NAMES) {
    assert.ok(page.includes(`assets/${name}`), `正文没有引用 ${name}`);
  }
  assert.ok(!/[A-Za-z_/]+\.py:\d+/.test(page), '正文不得含 path:line 引用');
  assert.ok(!page.includes('../'), '正文不得含 ../ 相对链接');
});
