// 锁住 slime loss 归约账本原理图的可执行契约：图上每个数字都由 14 页同一份 CFG/SAMPLES 与本图的 LOSS
// 经源码算法的复现推导，并且必须与 15_slime_loss_parallelism_analysis.md 正文引用的数值一致。
//
// 运行：node --test tools/figs/svg/lib/slime_loss_reducer_figures.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

import { GSPO_DIFF, LOSS, RAW_REWARDS, ROLLOUT_MASK_SUMS, gspoReplay, ledger, localNumerator, model, ppoRewardSlot, rewardPostProcess, sumOfSampleMean, threeMeans } from '../slime_loss_reducer_figures.mjs';
import { SAMPLES } from '../slime_megatron_train_step_figures.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const generator = join(here, '..', 'slime_loss_reducer_figures.mjs');
const slimeDir = join(here, '..', '..', '..', '..', 'wiki', '02_engineering', '04_posttrain_frameworks', 'slime');
const pagePath = join(slimeDir, '15_slime_loss_parallelism_analysis.md');
const trackedSvg = join(slimeDir, 'assets', 'slime_loss_reducer_ledger.svg');

const approx = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) < eps, `${a} ≠ ${b}`);

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

test('归约器复现 tests/test_cp_utils.py 的四个契约', () => {
  // sample_denoms=None ≡ per-sample mean：三条 3-token 样本 → 2 + 5 + 8 = 15
  const mk = (lens) => lens.map((r, k) => ({ name: `t${k}`, rolloutId: k, responseLen: r, totalLen: r + 4, lossMask: Array(r).fill(1) }));
  const three = mk([3, 3, 3]);
  const x = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
  const withLoss = (samples, denoms, cp, cps, xs) => samples.reduce((a, s, k) => a + localNumerator(s, cp, cps, s.lossMask, xs[k]) / Math.max(denoms[k], 1), 0);
  approx(withLoss(three, [3, 3, 3], 0, 1, x), 15);
  // 整 rollout 分母把兄弟样本并成一票：R0 = (1+…+9)/9 = 5，R1 = 33/3 = 11 → 16
  const four = mk([3, 3, 3, 3]);
  const x4 = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]];
  approx(withLoss(four, [9, 9, 9, 3], 0, 1, x4), 16);
  // 跨 mb 拆分后分段之和等于整体；局部分母则不等
  approx(withLoss(four.slice(0, 2), [9, 9], 0, 1, x4.slice(0, 2)) + withLoss(four.slice(2), [9, 3], 0, 1, x4.slice(2)), 16);
  assert.notEqual(withLoss(four.slice(0, 2), [6, 6], 0, 1, x4.slice(0, 2)) + withLoss(four.slice(2), [3, 3], 0, 1, x4.slice(2)), 16);
  // CP 两片之和等于 cp=1（本图样本 s0 与 s2b）
  for (const s of [SAMPLES[0], SAMPLES[3]]) {
    approx(localNumerator(s, 0, 2) + localNumerator(s, 1, 2), localNumerator(s, 0, 1));
  }
  assert.deepEqual([...ROLLOUT_MASK_SUMS], [7, 2, 10, 10, 2]);
  assert.deepEqual(LOSS.s0.length, SAMPLES[0].responseLen);
});

test('三种估计量、reward 分组回落与账本数值', () => {
  const T = threeMeans();
  assert.deepEqual(T.N, [14, 8, 6, 14, 8]);
  assert.deepEqual(T.D, [7, 2, 4, 6, 2]);
  assert.deepEqual([T.sumN, T.sumD, T.G, T.I], [50, 21, 4, 5]);
  approx(T.token, 50 / 21);
  approx(T.sample, (2 + 4 + 1.5 + 14 / 6 + 4) / 5);
  assert.deepEqual(T.rolloutMeans, [2, 4, 2, 4]);
  approx(T.rollout, 3);
  const R = rewardPostProcess();
  assert.equal(R.fallback, true);
  assert.equal(R.groupSize, RAW_REWARDS.length);
  approx(R.rewards[0], 0.4 / (Math.sqrt(0.3) + 1e-6));
  approx(R.rewards[1], -0.6 / (Math.sqrt(0.3) + 1e-6));
  const RP = rewardPostProcess([1, 0, 1, 0]);
  assert.equal(RP.fallback, false);
  approx(RP.rewards[0], 0.5 / (Math.sqrt(0.5) + 1e-6));
  const L = ledger();
  assert.deepEqual([L.G, L.M, L.world], [4, 2, 4]);
  assert.deepEqual(L.ranks.map((r) => [r.dp, r.cp, r.rankPartial]), [[0, 0, 1], [0, 1, 9], [1, 0, 0.2], [1, 1, 1.7999999999999998]]);
  assert.deepEqual(L.ranks[2].cells.map((c) => c.parts.map((p) => [p.name, p.owned, p.numerator, p.denom])), [[['s2b', [], 0, 10]], [['s2a', [3], 2, 10]]]);
  approx(L.total, 12);
  approx(L.scale, 2);
  approx(L.perMbAfterMegatron, 1);
  approx(L.finalGrad, 3);
  approx(L.totalLocalDenom, 13.833333333333334);
  approx(L.finalGradLocalDenom, 13.833333333333334 / 4);
  approx(L.reportRollout, 3);
  assert.deepEqual([L.allReducedNumTokens, L.allReducedTokenSum], [42, 50]);
  approx(L.reportToken, 50 / 21);
  approx(L.perTokenGrad, (2 * 50) / 42);
  const rj = model().rejection;
  assert.deepEqual([rj.denom, rj.original, rj.modifiedNumerator, rj.kept, rj.naive], [2, 4, 6, 3, 6]);
  // 空 rank：cp0 上 s2b 没有 response 位置但 reducer 仍返回 0 而非报错
  approx(sumOfSampleMean([SAMPLES[3]], [10], 0, 2), 0);
});

test('序列统计量的 CP 重建与 PPO reward 落点', () => {
  const g = gspoReplay();
  assert.deepEqual(g.ranks.map((r) => r.own), [[6, 7], [0, 1, 2, 3, 4, 5]]);
  g.reduced.forEach((v, i) => approx(v, GSPO_DIFF.s0[i]));
  approx(g.num, 1.4);
  assert.equal(g.den, 7);
  approx(g.seqKl, 0.2);
  approx(g.ranks[0].localMean, 0.35);
  approx(g.ranks[1].localMean, 0.14);
  assert.deepEqual(g.expandedCounts, [2, 6]);
  assert.deepEqual(g.emptyOwned, [], 's2b 在 cp0 本地为空');
  const slots = Object.fromEntries(model().ppoSlots.map((p) => [p.name, p]));
  assert.deepEqual([slots.s0.outcome, slots.s0.slot], ['ok', 7]);
  assert.deepEqual([slots.s2a.outcome, slots.s2a.slot], ['ok', 3]);
  for (const n of ['s1', 's2b', 's3']) assert.equal(slots[n].outcome, 'IndexError', n);
  assert.deepEqual([slots['T10/R8'].outcome, slots['T10/R8'].own, slots['T10/R8'].slot], ['misplaced', [0, 1], 1]);
  // 推导的判据：cp0 尾段覆盖最后一个 response logit ⇔ chunk ≥ pad + 2；T ≥ 3 时与逐位复现一致（T=2 时末 logit 落在 cp0 头段）
  for (const cp of [2, 4]) {
    for (let T = 3; T <= 48; T += 1) {
      for (let R = 1; R < T; R += 1) {
        const p = ppoRewardSlot(T, R, cp);
        assert.equal(p.outcome === 'ok', p.tailCovers, `cp=${cp} T=${T} R=${R}`);
      }
    }
  }
});

test('生成器产出原理图，且与已跟踪的 SVG 一致', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'slime-loss-ledger-'));
  try {
    const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
    assert.equal(run.status, 0, run.stderr || run.stdout);
    const svg = await readFile(join(outputDir, 'slime_loss_reducer_ledger.svg'), 'utf8');
    assertInsideCanvas(svg);
    assert.match(svg, /L_token = ΣN \/ Σmax\(D_i,1\) = 50 \/ 21 = 2\.381/);
    assert.match(svg, /= \(2 \+ 4 \+ 2 \+ 4\) \/ 4 = 3/);
    assert.match(svg, /s2b\{\}: 0 \/ 10 = 0/);
    assert.match(svg, /总和 13\.833，最终 3\.458 ≠ 3/);
    assert.match(svg, /× cp_size 2 \/ 42 = 2\.381 = L_token/);
    assert.match(svg, /6 \/ 1 = 6：拒得越多权重越大/);
    assert.match(svg, /masked 均值 1\.4 \/ 7 = 0\.2/);
    assert.match(svg, /错位 → 1（应为 7）/);
    assert.match(svg, /→ 2 × 50 \/ 42 = 2\.381/);
    assert.doesNotMatch(svg, /\[\[\d+_|\[\[[A-Za-z一-鿿]/, 'SVG 不得泄漏 wikilink 标记');
    const tracked = await readFile(trackedSvg, 'utf8');
    assert.equal(tracked, svg, '已跟踪的 SVG 必须由当前生成器重新生成');
  } finally {
    await rm(outputDir, { recursive: true, force: true });
  }
});

test('正文引用的数值与模型一致', async () => {
  const page = await readFile(pagePath, 'utf8');
  const m = model();
  const T = m.means; const L = m.ledger; const rj = m.rejection; const g = m.gspo;
  const slots = Object.fromEntries(m.ppoSlots.map((p) => [p.name, p]));
  const okNames = m.ppoSlots.filter((p) => p.outcome === 'ok' && p.name !== 'T10/R8').map((p) => p.name);
  const errNames = m.ppoSlots.filter((p) => p.outcome === 'IndexError').map((p) => p.name);
  const ptg = `${m.cfg.cpSize} × ${L.allReducedTokenSum} / ${L.allReducedNumTokens} ≈ ${L.perTokenGrad.toFixed(3)}`;
  const f2 = (v) => (v < 0 ? '-' : '') + Math.abs(v).toFixed(2);
  const cell = (dp, cp, mb, k) => L.ranks.find((r) => r.dp === dp && r.cp === cp).cells[mb].parts[k];
  const frac = (p) => `${p.name}: ${p.numerator}/${p.denom} = ${Number((p.numerator / Math.max(p.denom, 1)).toFixed(3))}`;
  for (const needle of [
    `\`L_token = ${T.sumN}/${T.sumD} ≈ ${T.token.toFixed(3)}\``,
    `\`L_sample ≈ ${T.sample.toFixed(3)}\``,
    `\`L_rollout = (${T.rolloutMeans.join(' + ')})/${T.G} = ${T.rollout}\``,
    `\`[${m.rewards.rewards.map(f2).join(', ')}]\``,
    `\`[${m.rewardsPerPrompt.rewards.map(f2).join(', ')}]\``,
    `\`${frac(cell(0, 0, 0, 0))}\``,
    `\`${frac(cell(0, 0, 0, 1))}\``,
    `\`${frac(cell(1, 1, 0, 0))}\``,
    `\`${frac(cell(1, 1, 1, 0))}\``,
    `\`${cell(1, 0, 1, 0).numerator}/${cell(1, 0, 1, 0).denom} = ${cell(1, 0, 1, 0).numerator / cell(1, 0, 1, 0).denom}\``,
    `\`[${L.ranks.map((r) => Number(r.rankPartial.toFixed(3))).join(', ')}]\`，总和 ${L.total}`,
    `\`${L.M}/${L.G} × ${L.world} = ${L.scale}\``,
    `\`${L.total} / ${L.G} = ${L.finalGrad} = L_rollout\``,
    `总和 ${L.totalLocalDenom.toFixed(3)}，最终 \`${L.finalGradLocalDenom.toFixed(3)} ≠ ${L.finalGrad}\``,
    `\`${L.allReducedTokenSum} × ${m.cfg.cpSize} / ${L.allReducedNumTokens} ≈ ${L.reportToken.toFixed(3)} = L_token\``,
    `\`${rj.modifiedNumerator}/${rj.denom} = ${rj.kept}\``,
    `\`${rj.modifiedNumerator}/${rj.survivorTokens} = ${rj.naive}\``,
    `\`rollout_mask_sums=[${ROLLOUT_MASK_SUMS.join(',')}]\``,
    `${L.totalLocalDenom.toFixed(3)} 对 ${L.total}`,
    `梯度 \`${ptg}\``,
    `\`${ptg} = L_token\``,
    `\`${Number(g.num.toFixed(3))}/${g.den} = ${Number(g.seqKl.toFixed(3))}\``,
    `\`${g.ranks.map((r) => Number(r.localMean.toFixed(3))).join('` 与 `')}\``,
    '`chunk ≥ pad + 2`',
    `本例 ${okNames.join(' 与 ')} 满足；${errNames.join('、')} 在 cp0 本地为空`,
    `reward 落在下标 ${slots['T10/R8'].slot} 而非 ${slots['T10/R8'].responseLen - 1}`,
  ]) {
    assert.ok(page.includes(needle), `正文必须出现 ${needle}`);
  }
  assert.ok(page.includes('assets/slime_loss_reducer_ledger.svg'), '正文必须引用原理图');
});
