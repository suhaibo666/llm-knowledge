// 锁住 slime 数据契约原理图的可执行契约：图上每个数字都由同一份 CFG 经源码算法的复现推导，
// 并且必须与 12_slime_sample_datasource_analysis.md 正文引用的数值一致。
//
// 运行：node --test tools/figs/svg/lib/slime_sample_data_contract_figures.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

import { CFG, appendResponseTokens, convert, model, newSample, recycledEntry, reducerDenominator } from '../slime_sample_data_contract_figures.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const generator = join(here, '..', 'slime_sample_data_contract_figures.mjs');
const slimeDir = join(here, '..', '..', '..', '..', 'wiki', '02_engineering', '04_posttrain_frameworks', 'slime');
const pagePath = join(slimeDir, '12_slime_sample_datasource_analysis.md');
const trackedSvg = join(slimeDir, 'assets', 'slime_sample_data_contract.svg');

function viewBox(svg) {
  const match = svg.match(/viewBox="0 0 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)"/);
  assert.ok(match, 'SVG 必须声明 viewBox');
  return { w: Number(match[1]), h: Number(match[2]) };
}

function assertInsideCanvas(svg) {
  const { w, h } = viewBox(svg);
  for (const [, x, y, rw, rh] of svg.matchAll(
    /<rect[^>]*?x="(-?\d+(?:\.\d+)?)"[^>]*?y="(-?\d+(?:\.\d+)?)"[^>]*?width="(\d+(?:\.\d+)?)"[^>]*?height="(\d+(?:\.\d+)?)"/g,
  )) {
    assert.ok(Number(x) >= 0 && Number(y) >= 0, `rect 左上越界 ${x},${y}`);
    assert.ok(Number(x) + Number(rw) <= w && Number(y) + Number(rh) <= h, `rect 右下越界 ${x}+${rw},${y}+${rh}`);
  }
  for (const [, x, y] of svg.matchAll(/<text[^>]*?x="(-?\d+(?:\.\d+)?)" y="(-?\d+(?:\.\d+)?)"/g)) {
    assert.ok(Number(x) >= 0 && Number(x) <= w && Number(y) >= 0 && Number(y) <= h, `text 越界 ${x},${y}`);
  }
}

test('append_response_tokens 的守卫与对齐规则', () => {
  const s = newSample({ groupIndex: 0, index: 0 });
  assert.throws(() => appendResponseTokens(s, { tokens: [1], logProbs: null, trainable: true }), /require rollout log probabilities/);
  assert.throws(() => appendResponseTokens(s, { tokens: [1], logProbs: [-0.1], trainable: false }), /should not pass rollout log probabilities/);
  appendResponseTokens(s, { tokens: [1, 2], logProbs: [-0.1, -0.2], finishReason: 'abort', weightVersion: 'v1' });
  appendResponseTokens(s, { tokens: [9], trainable: false }); // 工具 token：mask 0，logprob 填 0
  appendResponseTokens(s, { tokens: [3], logProbs: [-0.3], finishReason: 'stop', weightVersion: 'v2' });
  assert.deepEqual(s.lossMask, [1, 1, 0, 1]);
  assert.deepEqual(s.rolloutLogProbs, [-0.1, -0.2, 0, -0.3]);
  assert.deepEqual(s.weightVersions, ['v1', 'v2']);
  assert.equal(s.status, 'COMPLETED');
  assert.equal(s.responseLength, 4);
  // 已有 response 却无 rollout_log_probs 时不能再追加可训练 logprob
  const t = { ...newSample({ groupIndex: 0, index: 1 }), tokens: [11, 12, 5], responseLength: 1, lossMask: [1] };
  assert.throws(() => appendResponseTokens(t, { tokens: [6], logProbs: [-0.1] }), /no existing rollout_log_probs/);
  // generate_and_rm 入口：mask-offpolicy 清零发生在早退之前，已完成成员也被清 0
  const done = { ...newSample({ groupIndex: 0, index: 2 }), responseLength: 2, lossMask: [1, 1], status: 'COMPLETED' };
  const e = recycledEntry(done, { partialRollout: true, maskOffpolicy: true });
  assert.deepEqual([e.skip, e.sample.lossMask], [true, [0, 0]]);
  assert.equal(reducerDenominator(0), 1);
});

test('运行 ①（partial）复现源码算法', () => {
  const P = model().partial;
  assert.deepEqual(P.round3.groups, [[0, 1], [2, 3], [4, 5]]);
  assert.equal(P.round3.cursorAfter, 3);
  assert.equal(P.round3.bufferAfter, 1);
  assert.deepEqual(P.round4.groups, [[0, 1], [6, 7], [8, 9]]);
  assert.equal(P.round4.cursorAfter, 5);
  assert.equal(P.round4.fromBuffer, true);
  assert.equal(P.round4.budget, CFG.rolloutMaxResponseLen - P.s1AfterAbort.responseLength);
  assert.equal(P.s1AfterAbort.status, 'ABORTED');
  assert.equal(P.s0StartRolloutId, CFG.abortRound);
  assert.deepEqual(P.defaultMode.s1.tokens, [11, 12, 21, 22, 23, 24]);
  assert.deepEqual(P.defaultMode.s1.lossMask, [1, 1, 1, 1]);
  assert.deepEqual(P.defaultMode.s1.weightVersions, ['v3', 'v4']);
  assert.deepEqual(P.maskedMode.s1.lossMask, [0, 0, 1, 1]);
  assert.deepEqual(P.maskedMode.s0.lossMask, [0, 0]);
  assert.deepEqual(P.defaultMode.converted.rolloutIds, [0, 1, 2, 3]);
  assert.deepEqual(P.defaultMode.converted.maskSums, [2, 4, 3, 3]);
  assert.deepEqual(P.defaultMode.converted.rolloutMaskSums, [2, 4, 3, 3]);
  assert.deepEqual(P.maskedMode.converted.maskSums, [0, 2, 3, 3]);
  assert.deepEqual(P.maskedMode.converted.rolloutMaskSums, [0, 2, 3, 3]);
  assert.deepEqual(P.zeroDenominatorSamples, [0]);
  assert.equal(P.clampedDenominator, 1);
});

test('运行 ②（compact 扇出）复现源码算法', () => {
  const F = model().fanout;
  assert.deepEqual(F.converted.sampleIndices, [0, 1, 2, 2, 3]);
  assert.deepEqual(F.converted.rolloutIds, [0, 1, 2, 2, 3]);
  assert.deepEqual(F.converted.maskSums, [2, 4, 1, 3, 3]);
  assert.deepEqual(F.converted.rolloutMaskSums, [2, 4, 4, 4, 3]);
  assert.equal(F.globalBatchSize, 4);
  assert.equal(F.schedule.numSteps, 1);
  assert.deepEqual(F.schedule.steps[0].rollouts, [0, 1, 2, 3]);
  assert.equal(F.schedule.steps[0].mbs.length, 5);
  assert.deepEqual(F.fragmentMbs, [2, 3]);
  assert.equal(F.denominator, 4);
  assert.equal(F.sharedDenominatorTotal, 4);
  assert.equal(F.localDenominatorTotal, 12);
  // 兜底 id 跳过已存在的值
  const c = convert([{ ...newSample({ groupIndex: 0, index: 0 }), rolloutId: 0, responseLength: 1, lossMask: [1] }, { ...newSample({ groupIndex: 0, index: 1 }), responseLength: 1, lossMask: [1] }]);
  assert.deepEqual(c.rolloutIds, [0, 1]);
});

test('生成器产出原理图，且与已跟踪的 SVG 一致', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'slime-data-contract-'));
  try {
    const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
    assert.equal(run.status, 0, run.stderr || run.stdout);
    const svg = await readFile(join(outputDir, 'slime_sample_data_contract.svg'), 'utf8');
    assertInsideCanvas(svg);
    assert.match(svg, /start_rollout_id=3/);
    assert.match(svg, /max_new_tokens = 4 − 2 = 2/);
    assert.match(svg, /rollout_id 兜底：已存在 \{2\}/);
    assert.match(svg, /10\/4 \+ 6\/4 = 4/);
    assert.match(svg, /10\/1 \+ 6\/3 = 12/);
    assert.match(svg, /micro-batch 2 与 3/);
    assert.match(svg, /clamp_min\(denom, 1\)/);
    assert.doesNotMatch(svg, /\[\[/, 'SVG 不得泄漏 wikilink 标记');
    const tracked = await readFile(trackedSvg, 'utf8');
    assert.equal(tracked, svg, '已跟踪的 SVG 必须由当前生成器重新生成');
  } finally {
    await rm(outputDir, { recursive: true, force: true });
  }
});

test('正文引用的数值与模型一致', async () => {
  const page = await readFile(pagePath, 'utf8');
  const { partial: P, fanout: F } = model();
  const fmt = (arr) => `[${arr.join(',')}]`;
  for (const needle of [
    `\`rollout_ids=${fmt(P.defaultMode.converted.rolloutIds)}\``,
    `默认 \`${fmt(P.defaultMode.converted.rolloutMaskSums)}\`，mask-offpolicy \`${fmt(P.maskedMode.converted.rolloutMaskSums)}\``,
    `mask 和默认 \`${fmt(P.defaultMode.converted.maskSums)}\`、mask-offpolicy \`${fmt(P.maskedMode.converted.maskSums)}\``,
    `start_rollout_id=${CFG.abortRound}`,
    `max_new_tokens = ${CFG.rolloutMaxResponseLen} − ${P.s1AfterAbort.responseLength} = ${P.round4.budget}`,
    `样本 ${P.round3.groups[0][0]}–${P.round3.groups.at(-1).at(-1)}`,
    `样本 ${P.round4.groups[1][0]}–${P.round4.groups.at(-1).at(-1)}`,
    `\`rollout_ids=${fmt(F.converted.rolloutIds)}\``,
    `\`rollout_mask_sums=${fmt(F.converted.rolloutMaskSums)}\``,
    `逐样本 mask 和 \`${fmt(F.converted.maskSums)}\``,
    `${F.numerators[0]}/${F.denominator} + ${F.numerators[1]}/${F.denominator} = ${F.sharedDenominatorTotal}`,
    `${F.numerators[0]}/${F.converted.maskSums[F.fragmentPositions[0]]} + ${F.numerators[1]}/${F.converted.maskSums[F.fragmentPositions[1]]} = ${F.localDenominatorTotal}`,
    `global_batch_size = ${F.globalBatchSize}`,
    `micro-batch ${F.fragmentMbs[0]} 与 ${F.fragmentMbs[1]}`,
    `${F.schedule.steps[0].rollouts.length} // ${F.globalBatchSize} = ${F.schedule.numSteps}`,
    `clamp_min(denom, ${P.clampedDenominator})`,
  ]) {
    assert.ok(page.includes(needle), `正文必须出现 ${needle}`);
  }
  assert.ok(page.includes('assets/slime_sample_data_contract.svg'), '正文必须引用原理图');
});
