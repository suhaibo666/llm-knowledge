// 锁住 slime Megatron 训练后端原理图的可执行契约：图上每个数字都由同一份 CFG/SAMPLES 经源码算法的复现推导，
// 并且必须与 14_slime_megatron_training_analysis.md 正文引用的数值一致。
//
// 运行：node --test tools/figs/svg/lib/slime_megatron_train_step_figures.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

import {
  CFG, SAMPLES, buildDpSchedule, expandBinsBySplitting, firstFitPack, getBatch, getSeqlenBalancedPartitions, logitsTokensOffsetWithCp, model, ownedResponseIdx, roundPlan, sliceWithCp, splitBinByTokens,
} from '../slime_megatron_train_step_figures.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const generator = join(here, '..', 'slime_megatron_train_step_figures.mjs');
const slimeDir = join(here, '..', '..', '..', '..', 'wiki', '02_engineering', '04_posttrain_frameworks', 'slime');
const pagePath = join(slimeDir, '14_slime_megatron_training_analysis.md');
const trackedSvg = join(slimeDir, 'assets', 'slime_megatron_train_step.svg');

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

test('first-fit、拆 bin 与 zigzag 切片复现源码规则', () => {
  // tests/test_dp_schedule.py 记录的 help 例子：100/200/300 在 cap 300 下打成 [100,200],[300]
  assert.deepEqual(firstFitPack([100, 200, 300], 300), [[0, 1], [2]]);
  // 单条超 cap 样本独占一个 bin
  assert.deepEqual(firstFitPack([500, 100], 300), [[0], [1]]);
  // 按长度降序逐个放进当前较轻的一半（相等时放左半）；两半都是原 bin 的真子集
  assert.deepEqual(splitBinByTokens([0, 1, 2], [4, 10, 6]), [[1], [2, 0]]);
  const bins = [[0, 1], [2]];
  expandBinsBySplitting(bins, 4, [4, 10, 6]);
  assert.deepEqual(bins, [[1], [2], [0]], '所有 bin 都成单样本后停止，不足 target 不再拆');
  // zigzag：cp=2，长度 6 补到 8，rank0 取段 0 与 3
  assert.deepEqual(sliceWithCp([0, 1, 2, 3, 4, 5], 'P', 0, 2), [0, 1, 'P', 'P']);
  assert.deepEqual(sliceWithCp([0, 1, 2, 3, 4, 5], 'P', 1, 2), [2, 3, 4, 5]);
  // total 12 / response 8：rank0 只拿 response 末两位，rank1 拿前六位
  const off0 = logitsTokensOffsetWithCp(12, 8, 0, 2);
  assert.deepEqual(off0.logits, [[0, 0], [9, 11]]);
  assert.deepEqual(off0.tokens, [[0, 0], [10, 12]]);
  assert.deepEqual(ownedResponseIdx(SAMPLES[0], 0, 2), [6, 7]);
  assert.deepEqual(ownedResponseIdx(SAMPLES[0], 1, 2), [0, 1, 2, 3, 4, 5]);
  assert.deepEqual(ownedResponseIdx(SAMPLES[1], 0, 2), []);
});

test('build_dp_schedule 的组步、对齐、分发与静态断言', () => {
  const m = model();
  const d = m.dynamic;
  assert.deepEqual(d.steps[0].packed, [[0, 1], [2, 3], [4]]);
  assert.equal(d.steps[0].targetK, 4);
  assert.deepEqual(d.steps[0].splitTrace, [{ splitBin: 1, before: [2, 3], left: [3], right: [2] }]);
  assert.deepEqual(d.steps[0].bins, [[0, 1], [3], [4], [2]]);
  assert.deepEqual(d.partitions, [[0, 1, 4], [3, 2]]);
  assert.deepEqual(d.microBatchIndices, [[[0, 1], [2]], [[0], [1]]]);
  assert.deepEqual(d.numMicrobatches, [2]);
  assert.deepEqual(d.globalBatchSizes, [4]);
  assert.match(m.staticError, /static path: num_mbs \(3\) is not a multiple of dp_size \* mb_group \(2\)/);
  assert.deepEqual(m.rolloutMaskSums, [7, 2, 10, 10, 2]);
  // 不变量：每 rank 的 micro_batch_indices 展平恰为 range(n)，样本并集恰为全部样本
  d.microBatchIndices.forEach((mbi, r) => assert.deepEqual(mbi.flat(), [...Array(d.partitions[r].length).keys()]));
  assert.deepEqual([...d.partitions.flat()].sort(), [0, 1, 2, 3, 4]);
  // 尾部裁剪与不足一步的断言
  assert.throws(() => buildDpSchedule({ totalLengths: [3, 3, 3, 3, 3, 3], rolloutIndices: [0, 0, 1, 1, 2, 2] }), /num_rollouts \(3\) < global_batch_size \(4\)/);
  const trimmed = buildDpSchedule({ cfg: { ...CFG, globalBatchSize: 2, dpSize: 1 }, totalLengths: [3, 3, 3, 3, 3, 3, 3, 3], rolloutIndices: [0, 0, 1, 2, 2, 3, 4, 4] });
  assert.deepEqual(trimmed.globalBatchSizes, [2, 2]);
  assert.deepEqual([...trimmed.partitions[0]].sort(), [0, 1, 2, 3, 4, 5], '第 5 个 rollout 的样本 6、7 被丢出 schedule');
});

test('get_batch 的 THD 流、cu_seqlens 与 mask 对齐', () => {
  const m = model();
  const [cp0, cp1] = m.batches;
  assert.deepEqual(cp0.tokens.slice(0, 8), ['s0:0', 's0:1', 's0:2', 's0:9', 's0:10', 's0:11', 's1:0', 's1:1']);
  assert.equal(cp0.tokens.filter((t) => t === 'P').length, 8);
  assert.deepEqual(cp0.cuLocal, [0, 6, 10, 16]);
  assert.deepEqual(cp0.cuSeqlens, [0, 12, 20, 32]);
  assert.equal(cp0.pad, 6);
  assert.equal(cp0.maxSeqlen, 12);
  assert.deepEqual(cp0.fullLossMasks.slice(0, 10), [0, 0, 0, 1, 1, 0, 0, 0, 0, 0]);
  assert.deepEqual(cp1.fullLossMasks.slice(0, 10), [1, 1, 1, 0, 1, 1, 0, 1, 1, 0]);
  assert.deepEqual([cp0.maskedValid, cp1.maskedValid], [2, 7]);
  assert.deepEqual(cp0.owned, [[6, 7], []]);
  assert.deepEqual(cp1.owned, [[0, 1, 2, 3, 4, 5], [0, 1]]);
  assert.equal(cp0.maskedValid + cp1.maskedValid, SAMPLES[0].lossMask.reduce((a, b) => a + b, 0) + SAMPLES[1].lossMask.reduce((a, b) => a + b, 0));
  assert.throws(() => getBatch([{ ...SAMPLES[1], lossMask: [1, 1, 1, 1, 1, 1] }], 0), /loss_masks.shape != tokens.shape/, 'mask 长度与 response 不符时形状断言失败');
});

test('allgather_cp 分支与 Karmarkar-Karp 分发', () => {
  const m = model();
  const [a0, a1] = m.allgather;
  assert.deepEqual(a0.cuSeqlens, [0, 12, 18, 32], 'allgather 的 cu_seqlens 是全局前缀和，不乘 cp');
  assert.equal(a0.pad, 14);
  assert.equal(a0.maxSeqlen, 14);
  assert.deepEqual(a0.tokens, [...Array.from({ length: 12 }, (_, i) => `s0:${i}`), 's1:0', 's1:1', 's1:2', 's1:3']);
  assert.deepEqual(a1.tokens.slice(0, 3), ['s1:4', 's1:5', 'P']);
  assert.deepEqual([a0.realTokens, a1.realTokens], [16, 2]);
  assert.deepEqual([a0.maskedValid, a1.maskedValid], [8, 1]);
  assert.equal(a0.maskedValid + a1.maskedValid, m.batches[0].maskedValid + m.batches[1].maskedValid, '两种布局的有效 mask 总数相同');
  // 等大小 KK：mb 工作量对任意非负 aL+bL² 的排序与差值关系相同，本例配对结果都与轮询一致
  for (const k of m.kk) assert.deepEqual(k.partitions, [[0, 2], [1, 3]], `a=${k.a} b=${k.b}`);
  assert.deepEqual(m.kk[0].partitions, m.dynamic.steps[0].rankMbsIdx);
  // 手算的等大小 KK：[1,2,3,4] 分两份 → {0,3} 与 {1,2}，和 5 / 5
  assert.deepEqual(getSeqlenBalancedPartitions([1, 2, 3, 4], 2, true), [[0, 3], [1, 2]]);
  // balance_by_flops：KK 分组与系数无关；只有随后强制的 balance_data 分 rank 时以 a = 22b 为界
  for (const f of m.flops) {
    assert.deepEqual(f.groups, ['s0', 's1,s2b', 's2a,s3'], `a=${f.a} b=${f.b}`);
    assert.deepEqual(f.split, [{ before: ['s1', 's2b'], left: ['s2b'], right: ['s1'] }]);
    assert.equal(f.maxBinTokens, 14);
    assert.deepEqual(f.rankBins, f.regime === 'a>22b' ? [['[s2b]', '[s0]'], ['[s2a s3]', '[s1]']] : [['[s2b]', '[s2a s3]'], ['[s0]', '[s1]']]);
  }
  assert.deepEqual(m.flops.map((f) => f.regime), ['a>22b', 'a<=22b', 'a<=22b', 'a>22b']);
  const rb = (proxy) => {
    const st = buildDpSchedule({ totalLengths: SAMPLES.map((x) => x.totalLen), rolloutIndices: SAMPLES.map((x) => x.rolloutId), balanceByFlops: true, proxy }).steps[0];
    return st.rankMbsIdx.map((idxs) => idxs.map((k) => st.bins[k].map((g) => SAMPLES[g].name).join(' ')));
  };
  assert.deepEqual(rb({ a: 23, b: 1 }), [['s2b', 's0'], ['s2a s3', 's1']], '边界一侧：a > 22b');
  assert.deepEqual(rb({ a: 21, b: 1 }), [['s2b', 's2a s3'], ['s0', 's1']], '边界另一侧：a < 22b');
});

test('一轮 actor 的前向计数与 scheduler 计数', () => {
  const m = model();
  assert.equal(m.round.canReuse, true);
  assert.deepEqual(m.round.phases.map((p) => p.tag), ['ref', 'actor', 'actor', 'actor', 'actor']);
  assert.deepEqual([m.round.fullBatchForwards, m.round.pipelineCalls], [2, 2]);
  assert.equal(m.roundTwoSteps.canReuse, false);
  assert.deepEqual([m.roundTwoSteps.fullBatchForwards, m.roundTwoSteps.pipelineCalls], [3, 6], 'forward_only 也按步调用 forward_backward_func');
  assert.equal(roundPlan({ numSteps: 1, useCritic: true }).canReuse, false);
  assert.equal(roundPlan({ numSteps: 1, advantageEstimator: 'gspo' }).canReuse, false);
  assert.equal(roundPlan({ numSteps: 1, useRolloutLogprobs: true }).needOldForward, false, '用 rollout logprob 时不做 old 前向');
  assert.equal(roundPlan({ numSteps: 1, useRolloutLogprobs: true, getMismatchMetrics: true }).needOldForward, true, 'mismatch 指标仍多一次前向');
  assert.deepEqual(m.scheduler, { trainIters: 100, lrDecaySteps: 400, incrementPerStep: 4 });
});

test('生成器产出原理图，且与已跟踪的 SVG 一致', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'slime-train-step-'));
  try {
    const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
    assert.equal(run.status, 0, run.stderr || run.stdout);
    const svg = await readFile(join(outputDir, 'slime_megatron_train_step.svg'), 'utf8');
    assertInsideCanvas(svg);
    assert.match(svg, /\[s0 s1\]=18 {2}\[s2a s2b\]=18 {2}\[s3\]=6 {2}→ K=3/);
    assert.match(svg, /target_K = ceil\(3\/2\)×2 = 4/);
    assert.match(svg, /partition=\[0,1,4\]/);
    assert.match(svg, /cu_seqlens=\[0,12,20,32\]/);
    assert.match(svg, /全批前向 = ref 1 \+ 训练 1 = 2（forward_backward_func 调用 2 次）/);
    assert.match(svg, /cu_seqlens=\[0,12,18,32\]（不乘 cp）/);
    assert.match(svg, /balance_by_flops：KK 先分 3 组 \{s0\} \{s1,s2b\} \{s2a,s3\}/);
    assert.match(svg, /本例对任意 aL\+bL² 也得 \{bin0,bin2\} \/ \{bin1,bin3\}/);
    assert.match(svg, /scheduler\.step\(\+4\)/);
    assert.doesNotMatch(svg, /\[\[\d+_|\[\[[A-Za-z\u4e00-\u9fff]/, 'SVG 不得泄漏 wikilink 标记');
    const tracked = await readFile(trackedSvg, 'utf8');
    assert.equal(tracked, svg, '已跟踪的 SVG 必须由当前生成器重新生成');
  } finally {
    await rm(outputDir, { recursive: true, force: true });
  }
});

test('正文引用的数值与模型一致', async () => {
  const page = await readFile(pagePath, 'utf8');
  const m = model();
  const step = m.dynamic.steps[0];
  const name = (g) => SAMPLES[g].name;
  const binStr = (b, lens) => `[${b.map((i) => name(step.sampleIndices[i])).join(' ')}]=${b.reduce((a, i) => a + lens[i], 0)}`;
  const [cp0, cp1] = m.batches;
  const [a0, a1] = m.allgather;
  const fA = m.flops.find((f) => f.regime === 'a>22b'); const fB = m.flops.find((f) => f.regime === 'a<=22b');
  for (const needle of [
    `cap = ${CFG.maxTokensPerGpu} × ${CFG.cpSize} = ${step.maxPerBin}`,
    `\`${step.packed.map((b) => binStr(b, step.stepLengths)).join(' ')} → K=${step.packed.length}\``,
    `\`target_K = ceil(${step.packed.length}/${step.alignTo})×${step.alignTo} = ${step.targetK}\``,
    `bins \`${step.bins.map((b) => `[${b.map(name).join(' ')}]`).join(' ')}\``,
    `rank 0 \`partition=[${m.dynamic.partitions[0].join(',')}]\`、\`micro_batch_indices=${JSON.stringify(m.dynamic.microBatchIndices[0])}\``,
    `rank 1 \`partition=[${m.dynamic.partitions[1].join(',')}]\`、\`${JSON.stringify(m.dynamic.microBatchIndices[1])}\``,
    `\`num_microbatches=[${m.dynamic.numMicrobatches.join(',')}]\`、\`global_batch_sizes=[${m.dynamic.globalBatchSizes.join(',')}]\``,
    `cp0 tokens \`${cp0.tokens.filter((t) => t !== 'P').join(' ')}\` + ${cp0.tokens.filter((t) => t === 'P').length} pad`,
    `\`cu_seqlens=[${cp0.cuSeqlens.join(',')}]\``,
    `cp0 有效 ${cp0.maskedValid}（s0 的 response 下标 \`{${cp0.owned[0].join(',')}}\`，s1 无），cp1 有效 ${cp1.maskedValid}`,
    `全批前向 ${m.round.fullBatchForwards} 次（ref ${m.round.forwardPasses} + 训练 1）；若 \`num_steps_per_rollout=2\`（此时 G 变为 2、共 2 步），可复用条件失效，多一次 old-policy 全批前向共 ${m.roundTwoSteps.fullBatchForwards} 次，按 \`forward_backward_func\` 调用计则从 ${m.round.pipelineCalls} 次变为 ${m.roundTwoSteps.pipelineCalls} 次`,
    `\`cu_seqlens=[${a0.cuSeqlens.join(',')}]\` 不乘 cp`,
    `cp0 拿 \`s0:0–11\` 与 \`s1:0–3\`（有效 mask ${a0.maskedValid}），cp1 拿 \`s1:4–5\` 与 ${a1.tokens.length - a1.realTokens} 个 pad（有效 mask ${a1.maskedValid}）`,
    `rank 0 \`{bin${m.kk[0].partitions[0].join(',bin')}}\`、rank 1 \`{bin${m.kk[0].partitions[1].join(',bin')}}\``,
    `\`train_iters = ${CFG.numRollout}×${CFG.rolloutBatchSize}×${CFG.nSamplesPerPrompt} // ${CFG.globalBatchSize} = ${m.scheduler.trainIters}\`，\`lr_decay_steps = ${m.scheduler.lrDecaySteps}\``,
    `\`rollout_mask_sums=[${m.rolloutMaskSums.join(',')}]\``,
    `本例复用，一轮 ${m.round.fullBatchForwards} 次全批前向（${m.round.pipelineCalls} 次 \`forward_backward_func\` 调用）；两步时 ${m.roundTwoSteps.fullBatchForwards} 次全批前向、${m.roundTwoSteps.pipelineCalls} 次调用`,
    `都分出 \`{${fA.groups.join('}`、`{')}}\``,
    `再把 \`{${fA.split[0].before.join(',')}}\` 拆成 \`[${fA.split[0].left.join(' ')}]\` 与 \`[${fA.split[0].right.join(' ')}]\``,
    `a > 22b 时给 rank 0 \`${fA.rankBins[0].join(' ')}\`、rank 1 \`${fA.rankBins[1].join(' ')}\``,
    `a ≤ 22b 时给 rank 0 \`${fB.rankBins[0].join(' ')}\`、rank 1 \`${fB.rankBins[1].join(' ')}\``,
    `本例最大 bin ${fA.maxBinTokens} 个 token`,
    `K=${step.packed.length} 不是 ${step.alignTo} 的倍数`,
  ]) {
    assert.ok(page.includes(needle), `正文必须出现 ${needle}`);
  }
  assert.ok(page.includes('assets/slime_megatron_train_step.svg'), '正文必须引用原理图');
});
