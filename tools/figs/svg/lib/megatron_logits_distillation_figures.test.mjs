// 锁住 38_megatron_logits_distillation_analysis.md 两张图的可执行契约：DP 重映射三种情况逐 rank 的
// 读取与重建序、两条 ValueError、producer 的全局 top-K / top-P / 17 位打包、consumer 的 TP 感知稀疏 KL
// （两 rank 贡献、总和、与完整词表 KL 的差）、每 token 字节账——全部由复刻自冻结基线
// （NVIDIA/Megatron-LM@85902ef）的规则算出，并与页面正文引用的数值逐个对齐：图和正文不许各写各的。
//
// 运行：node --test tools/figs/svg/lib/megatron_logits_distillation_figures.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

import {
  REMAP_UP, REMAP_DOWN, REMAP_BAD_MB, REMAP_BAD_WORLD, KL_CASE, BYTES_CASE,
  CACHED_LOGITS_LOGPROB_SENTINEL, CACHED_LOGITS_INDEX_SENTINEL,
  REMAP, KL, BYTES,
  detectSavedDpSize, computeDpRemapping, sliceMicrobatches, interleaveMicrobatches, roundRobin, replayRemap,
  teacherProduce, applyToppTruncation, packIndices, unpackIndices, topkKlDiv, fullKl, byteLedger,
  f4, f2, pct1, pct0, intl,
} from '../megatron_logits_distillation_figures.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const generator = join(here, '..', 'megatron_logits_distillation_figures.mjs');
const assetDir = join(
  here, '..', '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks',
  'megatron-lm', 'assets',
);
const pagePath = join(
  here, '..', '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks',
  'megatron-lm', '38_megatron_logits_distillation_analysis.md',
);

const NAMES = [
  'megatron_logits_distillation_remap.svg',
  'megatron_logits_distillation_kl.svg',
];

const FONT = { ti: 18, su: 11.5, pt: 14, tx: 12, sm: 10.5, dim: 10.5, costtx: 10.5, rank: 11, cap: 11 };

function textWidth(value, fontSize) {
  let units = 0;
  for (const ch of String(value)) units += ch.charCodeAt(0) < 0x7f ? 0.56 : 1;
  return units * fontSize;
}

function parseCanvas(svg, name) {
  const m = svg.match(/viewBox="0 0 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)"/);
  assert.ok(m, `${name}: SVG 必须声明 viewBox`);
  return { w: Number(m[1]), h: Number(m[2]) };
}

// (d) 图元不出画布
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
  for (const [, x, y] of svg.matchAll(/<text[^>]*?x="(-?\d+(?:\.\d+)?)" y="(-?\d+(?:\.\d+)?)"/g)) {
    assert.ok(Number(x) >= 0 && Number(x) <= w, `${name}: text x=${x} 越界`);
    assert.ok(Number(y) >= 0 && Number(y) <= h, `${name}: text y=${y} 越界`);
  }
  for (const [, d] of svg.matchAll(/<path class="(?:main|aux|edge|sep)" d="([^"]+)"/g)) {
    for (const [, x, y] of d.matchAll(/(-?\d+(?:\.\d+)?) (-?\d+(?:\.\d+)?)/g)) {
      assert.ok(Number(x) >= 0 && Number(x) <= w && Number(y) >= 0 && Number(y) <= h, `${name}: path 顶点越界 ${x},${y}`);
    }
  }
}

// (d) 文字盒之间不重叠、文字盒不出画布（越界不等于不重叠，两条都要）
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
// (a) 复刻器对源码语义的锁定 —— 算例 A：DP 重映射
// ============================================================================

test('_compute_dp_remapping：相等 / 升配 / 降配三支与两条 ValueError 的返回值', () => {
  assert.deepEqual(computeDpRemapping(null, 3, 4), { sourceDpRanks: [3], subRank: 0, dpRatio: 1, dpSizeSaved: 4 }, '找不到 tar → 恒等 ([dp_rank], 0, 1, dp_size)');
  assert.deepEqual(computeDpRemapping(4, 2, 4), { sourceDpRanks: [2], subRank: 0, dpRatio: 1, dpSizeSaved: 4 }, '相等 → ([dp_rank], 0, 1, dp_size_saved)');
  assert.deepEqual(computeDpRemapping(2, 3, 4), { sourceDpRanks: [1], subRank: 1, dpRatio: 2, dpSizeSaved: 2 }, '升配：dp_rank 3 → 存档 dp1、sub_rank 1、stride 2');
  assert.deepEqual(computeDpRemapping(4, 1, 2), { sourceDpRanks: [1, 3], subRank: 0, dpRatio: 0.5, dpSizeSaved: 4 }, '降配：dp_rank 1 读 dp1、dp3，dp_ratio 0.5 只作信息');
  assert.throws(() => computeDpRemapping(2, 0, 3), /Current DP size \(3\) is not an exact multiple of saved DP size \(2\)/);
  assert.throws(() => computeDpRemapping(3, 0, 2), /Saved DP size \(3\) is not an exact multiple of current DP size \(2\)/);
  assert.equal(detectSavedDpSize([0, 1]), 2);
  assert.equal(detectSavedDpSize([0, 3]), 4, 'max(dp) + 1，缺文件的 rank 不会被发现');
  assert.equal(detectSavedDpSize([]), null);
});

test('_slice_microbatches 与 _interleave_microbatches 在算例上的逐格结果', () => {
  assert.deepEqual(sliceMicrobatches([0, 2, 4, 6], 0, 2), [0, 4]);
  assert.deepEqual(sliceMicrobatches([0, 2, 4, 6], 1, 2), [2, 6]);
  assert.deepEqual(sliceMicrobatches([0, 2, 4, 6], 0, 1), [0, 2, 4, 6], 'dp_ratio ≤ 1 原样返回');
  assert.throws(() => sliceMicrobatches([0, 2, 4], 0, 2), /Saved microbatch count \(3\) is not divisible by DP ratio \(2\)/);
  assert.deepEqual(interleaveMicrobatches([[0, 4], [2, 6]]), [0, 2, 4, 6], 'src0_mb0, src1_mb0, src0_mb1, src1_mb1');
  assert.deepEqual(roundRobin(8, 2), [[0, 2, 4, 6], [1, 3, 5, 7]]);
});

test('算例 A：升配 dp 2 → 4 与降配 dp 4 → 2 逐 rank 重建出学生 DP 的轮转序', () => {
  assert.deepEqual(REMAP_UP, { G: 8, dpSaved: 2, dp: 4 });
  assert.deepEqual(REMAP_DOWN, { G: 8, dpSaved: 4, dp: 2 });
  const up = REMAP.up;
  assert.equal(up.dpSizeSaved, 2);
  assert.deepEqual(up.ranks.map((r) => [r.rank, r.sourceDpRanks[0], r.subRank, r.dpRatio, r.reconstructed]), [
    [0, 0, 0, 2, [0, 4]],
    [1, 1, 0, 2, [1, 5]],
    [2, 0, 1, 2, [2, 6]],
    [3, 1, 1, 2, [3, 7]],
  ]);
  assert.ok(up.ranks.every((r) => r.ok));
  assert.equal(up.numMbSaved, 4);
  assert.equal(up.numMbNew, 2);
  const down = REMAP.down;
  assert.deepEqual(down.ranks.map((r) => [r.rank, r.sourceDpRanks, r.reconstructed]), [
    [0, [0, 2], [0, 2, 4, 6]],
    [1, [1, 3], [1, 3, 5, 7]],
  ]);
  assert.ok(down.ranks.every((r) => r.ok));
  // 整除破坏
  assert.deepEqual(REMAP_BAD_MB, { G: 6, dpSaved: 2, dp: 4 });
  assert.equal(REMAP.badMb.ok, false);
  assert.match(REMAP.badMb.error, /Saved microbatch count \(3\) is not divisible by DP ratio \(2\)/);
  assert.deepEqual(REMAP_BAD_WORLD, { G: 8, dpSaved: 2, dp: 3 });
  assert.equal(REMAP.badWorld.ok, false);
  assert.match(REMAP.badWorld.error, /Current DP size \(3\) is not an exact multiple of saved DP size \(2\)/);
  // 不变量在另一组参数上也成立：dp 4 → 8 与 dp 8 → 2
  assert.ok(replayRemap({ G: 16, dpSaved: 4, dp: 8 }).ranks.every((r) => r.ok));
  assert.ok(replayRemap({ G: 16, dpSaved: 8, dp: 2 }).ranks.every((r) => r.ok));
});

// ============================================================================
// (a) 复刻器对源码语义的锁定 —— 算例 B：producer 与 consumer
// ============================================================================

test('producer：局部 top-K → 跨 TP logsumexp → gather 后按 fp32 logit 取全局 top-K', () => {
  assert.equal(KL_CASE.V, 16);
  assert.equal(KL_CASE.tp, 2);
  assert.equal(KL_CASE.K, 4);
  const P = KL.producer;
  assert.equal(P.localV, 8);
  assert.equal(P.localK, 4);
  assert.deepEqual(P.indices, [1, 9, 3, 14]);
  assert.deepEqual(P.values.map(f4), ['-0.9599', '-1.4599', '-1.9599', '-2.4599']);
  assert.equal(f4(P.perRank[0].localLse), '4.4421');
  assert.equal(f4(P.perRank[1].localLse), '4.0540');
  assert.equal(f4(P.globalLse), '4.9599');
  // 全局 lse 等于对完整词表直接 logsumexp（MAX / SUM 两次 all_reduce 只是数值稳定的分解）
  const direct = Math.log(KL_CASE.teacher.reduce((acc, v) => acc + Math.exp(v), 0));
  assert.ok(Math.abs(direct - P.globalLse) < 1e-12);
  assert.equal(f4(KL.teacherTopMass), '0.8415');
  assert.equal(f4(KL.teacherTailMass), '0.1585');
  // TP = 1 时无 gather：结果与 TP = 2 的全局 top-K 一致（同一 fp32 logit 排序）
  const single = teacherProduce(KL_CASE.teacher, 1, KL_CASE.K);
  assert.deepEqual(single.indices, P.indices);
  assert.deepEqual(single.values.map(f4), P.values.map(f4));
  // V > 2^17 由 assert 守住
  assert.throws(() => teacherProduce(new Array(2 ** 17 + 2).fill(0), 2, 4), /17 bits/);
});

test('_apply_topp_truncation：keep ⇔ cumprobs − probs < p，OR 上 min_k，截到 max_kept，尾部填 sentinel', () => {
  const P = KL.producer;
  assert.equal(KL_CASE.topP, 0.7);
  assert.equal(KL.topp.keptPerToken, 3, '前三项累计 0.756 才越过 0.7：第 4 项被截');
  assert.deepEqual(KL.topp.indices, [1, 9, 3]);
  assert.equal(KL.topp.values.length, 3, '单 token 的 max_kept 就是 kept');
  // p 很小 → 只剩 min_k 条
  const tiny = applyToppTruncation(P.values, P.indices, 0.01, 2);
  assert.equal(tiny.keptPerToken, 2);
  // 多 token 语义：kept 少于 max_kept 的位置用 sentinel（单 token 模拟：把 keep 全关但保留 K 维）
  const forced = applyToppTruncation(P.values, P.indices, 1.0, 1);
  assert.equal(forced.keptPerToken, 4, 'p = 1 时全部保留');
  assert.equal(CACHED_LOGITS_LOGPROB_SENTINEL, -1e3);
  assert.equal(CACHED_LOGITS_INDEX_SENTINEL, -1);
});

test('pack_indices / unpack_indices：uint16 低 16 位 + bool 第 17 位', () => {
  assert.deepEqual(KL.packed, { low: [1, 9, 3, 14], bit17: [false, false, false, false] });
  assert.equal(KL_CASE.packDemoIndex, 100000);
  assert.deepEqual(KL.demoPacked, { low: [34464], bit17: [true] });
  assert.equal(KL.demoUnpacked, 100000);
  const round = (i) => unpackIndices(...Object.values(packIndices([i])))[0];
  for (const i of [0, 65535, 65536, 131071]) assert.equal(round(i), i);
});

test('topk_kl_div：两次 all_reduce、offset 映射与 mask、ghost 残差只在 tp0 计入、各 rank 局部贡献', () => {
  const G = KL.ghost;
  assert.equal(G.logitsMax, 3.8);
  assert.equal(f4(G.sumExp), '2.3506');
  assert.deepEqual(G.ranks.map((r) => f4(r.localSumExp)), ['0.8674', '1.4832']);
  assert.deepEqual(G.ranks.map((r) => r.offset), [0, 8]);
  assert.deepEqual(G.ranks.map((r) => r.hits), [[1, 3], [9, 14]]);
  assert.deepEqual(G.ranks[0].mask, [true, false, true, false]);
  assert.deepEqual(G.ranks[1].mask, [false, true, false, true]);
  assert.deepEqual(G.ranks[0].gathered.filter((_, k) => G.ranks[0].mask[k]).map(f4), ['-1.6547', '-2.6547']);
  assert.deepEqual(G.ranks[1].gathered.filter((_, k) => G.ranks[1].mask[k]).map(f4), ['-0.8547', '-3.6547']);
  assert.deepEqual(G.ranks.map((r) => f4(r.hitExpSum)), ['0.2615', '0.4513']);
  assert.equal(f4(G.ranks[0].hitExpSum + G.ranks[1].hitExpSum), '0.7128');
  assert.equal(f4(G.studentResidual), '-1.2475');
  assert.equal(f4(G.teacherResidual), '-1.8420');
  assert.equal(f4(G.ranks[0].klTerms[4]), '-0.0942', 'ghost 项只在 tp0');
  assert.equal(G.ranks[1].klTerms[4], 0);
  assert.deepEqual(G.ranks.map((r) => f4(r.contribution)), ['0.2697', '-0.0385']);
  assert.equal(f4(G.total), '0.2312');
  // 对照
  assert.equal(f4(KL.full), '0.2729');
  assert.equal(f4(KL.noGhost.total), '0.3254');
  assert.equal(f4(KL.lostTail), '0.0417');
  assert.ok(G.total <= KL.full + 1e-12, '粗化不增加散度');
  assert.ok(KL.full >= 0);
  // 学生 = 教师时 KL 为 0（含 ghost）
  const same = topkKlDiv(KL_CASE.teacher, KL.producer.values, KL.producer.indices, 2, true);
  assert.ok(Math.abs(same.total) < 1e-12);
  assert.ok(Math.abs(fullKl(KL_CASE.teacher, KL_CASE.teacher)) < 1e-12);
  // sentinel 项被 mask：把第 4 项换成 sentinel 后总和只少这一项的贡献
  const vals = [...KL.producer.values.slice(0, 3), CACHED_LOGITS_LOGPROB_SENTINEL];
  const idx = [...KL.producer.indices.slice(0, 3), CACHED_LOGITS_INDEX_SENTINEL];
  const truncated = topkKlDiv(KL_CASE.student, vals, idx, 2, true);
  assert.ok(truncated.ranks[1].mask[3] === false && truncated.ranks[0].mask[3] === false);
  // TP = 1 与 TP = 2 的总 KL 一致
  const tp1 = topkKlDiv(KL_CASE.student, KL.producer.values, KL.producer.indices, 1, true);
  assert.ok(Math.abs(tp1.total - G.total) < 1e-12);
});

test('字节账：V = 2^17、K = 64、值 2 B 的每 token 原始 payload', () => {
  assert.deepEqual(BYTES_CASE, { V: 131072, K: 64, valueBytes: 2 });
  assert.equal(BYTES.dense, 262144);
  assert.equal(BYTES.topkInt32, 384);
  assert.equal(BYTES.topk17, 320);
  assert.equal(BYTES.ratio, 819.2);
  assert.equal(pct0(BYTES.indexSaving), '25%');
  assert.equal(pct1(BYTES.totalSaving), '16.7%');
  assert.equal(byteLedger({ V: 131072, K: 64, valueBytes: 4 }).topk17, 448, 'fp32 值时 K × 7');
});

// ============================================================================
// (b) 生成器产出 = 已跟踪资产；(d) 图元与文字盒
// ============================================================================

test('生成器同步产出两张图，图上的关键量与算例一致，且与 assets 里的文件逐字节相同', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'megatron-logits-distillation-figures-'));
  const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
  assert.equal(run.status, 0, run.stderr || run.stdout);
  const [remap, kl] = await Promise.all(NAMES.map((name) => readFile(join(outputDir, name), 'utf8')));

  // 图 1
  for (const r of REMAP.up.ranks) assert.ok(remap.includes(`✓ 轮转 dp${r.rank} = [${r.expected.join(', ')}]`), `图 1 缺 dp${r.rank} 的重建序`);
  assert.ok(remap.includes(`[0::2]`) && remap.includes(`[1::2]`));
  assert.ok(remap.includes(`detect_saved_dp_size = max(dp) + 1 = ${REMAP.up.dpSizeSaved}`));
  assert.ok(remap.includes(REMAP.badMb.error.replace('ValueError: ', '')));
  assert.ok(remap.includes(REMAP.badWorld.error.replace('ValueError: ', '')));
  assert.ok(remap.includes(`本图 ${REMAP.up.dp + REMAP.down.dp} 个 rank 逐个核对`));
  // 图 2
  assert.ok(kl.includes(`global_lse = ${f4(KL.producer.globalLse)}`));
  assert.ok(kl.includes(`Σ p_T(top-K) = ${f4(KL.teacherTopMass)}`));
  assert.ok(kl.includes(`保留 ${KL.topp.keptPerToken} 项`));
  assert.ok(kl.includes(`idx ${intl(KL_CASE.packDemoIndex)} → low ${intl(KL.demoPacked.low[0])}，bit17 1`));
  assert.ok(kl.includes(`TP MAX = ${f2(KL.ghost.logitsMax)}`) && kl.includes(`TP SUM = ${f4(KL.ghost.sumExp)}`));
  assert.ok(kl.includes(`KL = ${f4(KL.ghost.ranks[0].contribution)} + (${f4(KL.ghost.ranks[1].contribution)}) = ${f4(KL.ghost.total)}`));
  assert.ok(kl.includes(`完整词表 KL = ${f4(KL.full)}`) && kl.includes(`无 ghost 的稀疏和 = ${f4(KL.noGhost.total)}`));
  assert.ok(kl.includes(`差 ${f4(KL.lostTail)}`));
  assert.ok(kl.includes(`${intl(BYTES.dense)} B / token`) && kl.includes(`${intl(BYTES.topkInt32)} B / token`) && kl.includes(`${intl(BYTES.topk17)} B / token`));
  assert.ok(kl.includes(`${BYTES.ratio.toFixed(1)}×`) && kl.includes(pct1(BYTES.totalSaving)));
  for (const s of [remap, kl]) assert.ok(!s.includes('[['), '图上不允许漏出 wikilink 语法');

  const all = [remap, kl];
  NAMES.forEach((name, i) => {
    assertInsideCanvas(all[i], name);
    assertNoTextOverlap(all[i], name);
  });

  const tracked = await Promise.all(NAMES.map((name) => readFile(join(assetDir, name), 'utf8')));
  NAMES.forEach((name, i) => {
    assert.equal(tracked[i], all[i], `${name} 与生成器输出不一致：重跑 node tools/figs/svg/megatron_logits_distillation_figures.mjs`);
  });
});

// ============================================================================
// (c) 页面正文引用的数值与图上一致 —— 只改正文、不改图，这个用例必须红
// ============================================================================

test('页面正文引用的数值与图上一致', async () => {
  const page = await readFile(pagePath, 'utf8');

  // 算例 A
  assert.ok(page.includes(`G = ${REMAP_UP.G}`) && page.includes(`dp_saved = ${REMAP_UP.dpSaved}`) && page.includes(`dp = ${REMAP_UP.dp}`));
  assert.ok(page.includes(`dp_ratio = ${REMAP.up.ranks[0].dpRatio}`));
  for (const r of REMAP.up.ranks) {
    assert.ok(page.includes(`[${r.reconstructed.join(', ')}]`), `正文缺升配 dp${r.rank} 的重建序`);
  }
  assert.ok(page.includes(`[${REMAP.up.subRank ?? 0}::2]`) || page.includes('[sub_rank::dp_ratio]'));
  for (const r of REMAP.down.ranks) {
    assert.ok(page.includes(`[${r.reconstructed.join(', ')}]`), `正文缺降配 dp${r.rank} 的重建序`);
    assert.ok(page.includes(`dp${r.sourceDpRanks[0]}`) && page.includes(`dp${r.sourceDpRanks[1]}`));
  }
  assert.ok(page.includes(`num_mb = ${REMAP_BAD_MB.G / REMAP_BAD_MB.dpSaved}`), '正文缺整除破坏例的 num_mb');
  assert.ok(page.includes('not divisible by DP ratio') && page.includes('not an exact multiple'));
  assert.ok(page.includes(`dp = ${REMAP_BAD_WORLD.dp}`));

  // 算例 B：producer
  const P = KL.producer;
  assert.ok(page.includes(`V = ${KL_CASE.V}`) && page.includes(`TP = ${KL_CASE.tp}`) && page.includes(`K = ${KL_CASE.K}`));
  assert.ok(page.includes(`{${P.indices.join(', ')}}`) || page.includes(P.indices.join('、')), '正文缺全局 top-K 索引');
  assert.ok(page.includes(f4(P.perRank[0].localLse)) && page.includes(f4(P.perRank[1].localLse)), '正文缺局部 lse');
  assert.ok(page.includes(f4(P.globalLse)), '正文缺 global_lse');
  for (const v of P.values) assert.ok(page.includes(f4(v)), `正文缺 logprob ${f4(v)}`);
  assert.ok(page.includes(f4(KL.teacherTopMass)) && page.includes(f4(KL.teacherTailMass)));
  assert.ok(page.includes(`top-P = ${KL_CASE.topP}`) && page.includes(`保留 ${KL.topp.keptPerToken} 项`) || page.includes(`只剩 ${KL.topp.keptPerToken} 项`));
  assert.ok(page.includes(intl(KL_CASE.packDemoIndex)) && page.includes(intl(KL.demoPacked.low[0])));

  // 算例 B：consumer
  const G = KL.ghost;
  assert.ok(page.includes(f2(G.logitsMax)) || page.includes(String(G.logitsMax)));
  assert.ok(page.includes(f4(G.sumExp)));
  for (const r of G.ranks) {
    assert.ok(page.includes(f4(r.localSumExp)), `正文缺 tp${r.rank} 的局部 Σexp`);
    assert.ok(page.includes(f4(r.hitExpSum)), `正文缺 tp${r.rank} 的 Σ p_S(hit)`);
    assert.ok(page.includes(f4(r.contribution)), `正文缺 tp${r.rank} 的贡献`);
    for (const k of r.hits.map((h) => r.gathered[P.indices.indexOf(h)])) assert.ok(page.includes(f4(k)), `正文缺命中的 log p_S ${f4(k)}`);
  }
  assert.ok(page.includes(f4(G.ranks[0].hitExpSum + G.ranks[1].hitExpSum)));
  assert.ok(page.includes(f4(G.studentResidual)) && page.includes(f4(G.teacherResidual)));
  assert.ok(page.includes(f4(G.ranks[0].klTerms[4])), '正文缺 ghost 项');
  assert.ok(page.includes(f4(G.total)) && page.includes(f4(KL.full)) && page.includes(f4(KL.noGhost.total)) && page.includes(f4(KL.lostTail)));

  // 字节账
  assert.ok(page.includes(intl(BYTES_CASE.V)) && page.includes(`K = ${BYTES_CASE.K}`));
  assert.ok(page.includes(`${intl(BYTES.dense)} B`) && page.includes(`${intl(BYTES.topkInt32)} B`) && page.includes(`${intl(BYTES.topk17)} B`));
  assert.ok(page.includes(`${BYTES.ratio.toFixed(1)}`) && page.includes(pct0(BYTES.indexSaving)) && page.includes(pct1(BYTES.totalSaving)));

  for (const name of NAMES) assert.ok(page.includes(`assets/${name}`), `正文没有引用 ${name}`);
  assert.ok(!/[A-Za-z_/]+\.py:\d+/.test(page), '正文不得含 path:line 引用');
  assert.ok(!page.includes('../'), '正文不得含 ../ 相对链接');
});
