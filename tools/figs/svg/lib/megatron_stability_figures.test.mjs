// 锁住稳定性图示的可执行契约：尖峰判据与 aux-loss 归一都由复刻的源码算法算出，
// 且与 28_megatron_training_stability_observability_analysis.md 正文引用的数值逐个对齐。
//
// 运行：node --test tools/figs/svg/lib/megatron_stability_figures.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

import {
  CFG, BASE, POLLUTED, LATE_RATIO, BASE_FLAGS, POLLUTED_FLAGS,
  AUX_SKEWED, AUX_UNIFORM, spikeDetector, auxScale,
} from '../megatron_stability_figures.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const generator = join(here, '..', 'megatron_stability_figures.mjs');
const assetDir = join(
  here, '..', '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks',
  'megatron-lm', 'assets',
);
const pagePath = join(
  here, '..', '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks',
  'megatron-lm', '28_megatron_training_stability_observability_analysis.md',
);

const NAMES = [
  'megatron_stability_spike_detector.svg',
  'megatron_stability_grad_gates.svg',
  'megatron_stability_auxloss_scale.svg',
];

function assertInsideCanvas(svg, name) {
  const m = svg.match(/viewBox="0 0 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)"/);
  assert.ok(m, `${name}: SVG 必须声明 viewBox`);
  const w = Number(m[1]);
  const h = Number(m[2]);
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
}

// ============================================================================

test('尖峰判据：窗口只估 max，armed 后才判；两条失效路径都成立', () => {
  assert.equal(BASE.maxValue, 2.1, '窗口内最大值');
  assert.equal(BASE.trigger, 21, 'max × threshold');
  assert.equal(BASE_FLAGS, 1, '基准序列恰好判出一次尖峰');
  // 窗口内的样本一律不判，哪怕它比后面的值大
  assert.equal(BASE.trace.slice(0, CFG.numSamples).every((t) => !t.flagged && t.phase === 'sampling'), true);
  // 3.9 大于其余正常值，但够不到触发线
  const near = BASE.trace.find((t) => t.value === 3.9);
  assert.equal(near.flagged, false, '3.9 < 21，不该判出');
  const spike = BASE.trace.find((t) => t.value === 22.5);
  assert.equal(spike.flagged, true);

  // 失效路径 (a)：窗口被污染，判据整体失灵
  assert.equal(POLLUTED.maxValue, CFG.pollutedValue);
  assert.equal(POLLUTED.trigger, CFG.pollutedValue * CFG.threshold);
  assert.equal(POLLUTED_FLAGS, 0, '污染后同一次真尖峰不再被判出');

  // 失效路径 (b)：触发线不随训练下移
  assert.equal(LATE_RATIO, BASE.trigger / CFG.lateLoss);
  assert.equal(Math.round(LATE_RATIO), 42);

  // NaN / Inf 走另一条路：不被判、也不进入 max 估计
  const withNan = spikeDetector([NaN, Infinity, ...CFG.series], CFG);
  assert.equal(withNan.maxValue, BASE.maxValue, '非有限值不得污染 max 估计');
  assert.equal(withNan.trace[0].flagged, false);
  assert.equal(withNan.trace[1].flagged, false);

  // resample 语义：换一个更大的窗口会得到不同的触发线，说明判据确实由数据决定
  const wide = spikeDetector(CFG.series, { threshold: CFG.threshold, numSamples: 12 });
  assert.ok(wide.maxValue > BASE.maxValue, '把尖峰纳入窗口后 max 必然变大');
});

test('aux-loss 归一：闭式因子只在等长时与实测一致', () => {
  assert.equal(AUX_SKEWED.measured, 1600);
  assert.deepEqual(AUX_SKEWED.closed, [2048, 1536, 1792, 1024]);
  assert.deepEqual(AUX_SKEWED.ratio.map((r) => Number(r.toFixed(2))), [1.28, 0.96, 1.12, 0.64]);
  assert.equal(Math.max(...AUX_SKEWED.ratio).toFixed(2), '1.28');
  assert.equal(Math.min(...AUX_SKEWED.ratio).toFixed(2), '0.64');

  assert.equal(AUX_UNIFORM.measured, 2048);
  assert.deepEqual(AUX_UNIFORM.ratio, [1, 1, 1, 1], '等长时闭式与实测完全一致');

  // 偏差只取决于「本 rank 的 token 数 / 组内均值」：把组翻倍但保持分布，比值一模一样
  const doubled = auxScale([...CFG.validTokens, ...CFG.validTokens]);
  assert.deepEqual(
    doubled.ratio.slice(0, 4).map((r) => Number(r.toFixed(4))),
    AUX_SKEWED.ratio.map((r) => Number(r.toFixed(4))),
    '闭式因子的偏差与组大小无关，只与分布不均有关',
  );
  // 分布越不均，偏差越大 —— 所以它不是一个能靠调常数补救的量
  const skewer = auxScale([1024, 128, 128, 128]);
  assert.ok(Math.max(...skewer.ratio) > Math.max(...AUX_SKEWED.ratio));
});

test('生成器同步产出三张图，且图上的关键量与算例一致', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'megatron-stability-figures-'));
  const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
  assert.equal(run.status, 0, run.stderr || run.stdout);

  const [spike, gates, aux] = await Promise.all(
    NAMES.map((name) => readFile(join(outputDir, name), 'utf8')),
  );

  // ---- 图 1：原理图 ----
  assert.ok(spike.includes(`is_unexpectedly_large(threshold=${CFG.threshold})`));
  assert.ok(spike.includes(`num_samples=${CFG.numSamplesDefault}`), '必须写明源码默认窗口');
  assert.ok(spike.includes(`max=${BASE.maxValue}`));
  assert.ok(spike.includes('触发 21'));
  assert.ok(spike.includes('22.5'), '被判出的那一步必须标出数值');
  assert.ok(spike.includes(`触发线由 21 抬到 ${POLLUTED.trigger}`));
  assert.ok(spike.includes(`命中数从 ${BASE_FLAGS} 变成 ${POLLUTED_FLAGS}`));
  assert.ok(spike.includes(`${Math.round(LATE_RATIO)}× 于当前 loss`));
  assert.ok(spike.includes('They should be checked separately'));
  assert.ok(!spike.includes('[['), '图上不允许漏出 wikilink 语法');

  // ---- 图 2：三道闸 ----
  for (const gate of ['闸①', '闸②', '闸③']) {
    assert.ok(gates.includes(gate), `图 2 缺 ${gate}`);
  }
  assert.ok(gates.includes(`clip_grad=${CFG.clipGrad}`));
  assert.ok(gates.includes(`skip=${CFG.gradNormSkipThreshold}`));
  assert.ok(gates.includes(`主组 ‖g‖=${CFG.mainNorm}`));
  assert.ok(gates.includes(`‖g_mtp‖=${CFG.mtpNorm}`));
  assert.ok(gates.includes('整步不会被丢'), 'mtp 组不参与超阈跳步这一点必须画出来');
  assert.ok(gates.includes("float('inf')"));

  // ---- 图 3：aux-loss ----
  for (const t of CFG.validTokens) assert.ok(aux.includes(`>${t}<`), `缺 token 数 ${t}`);
  assert.ok(aux.includes(`all_reduce(SUM) = ${AUX_SKEWED.measured}`));
  assert.ok(aux.includes(`all_reduce(SUM) = ${AUX_UNIFORM.measured}`));
  AUX_SKEWED.closed.forEach((c, i) => {
    assert.ok(
      aux.includes(`${c} / ${AUX_SKEWED.measured} = ${Number(AUX_SKEWED.ratio[i].toFixed(2))}×`),
      `缺 rank ${i} 的比值`,
    );
  });
  assert.ok(aux.includes('valid token counts can differ by rank/group'));

  NAMES.forEach((name, i) => assertInsideCanvas([spike, gates, aux][i], name));

  const tracked = await Promise.all(NAMES.map((name) => readFile(join(assetDir, name), 'utf8')));
  NAMES.forEach((name, i) => {
    assert.equal(
      tracked[i],
      [spike, gates, aux][i],
      `${name} 与生成器输出不一致：重跑 node tools/figs/svg/megatron_stability_figures.mjs`,
    );
  });
});

test('页面正文引用的数值与图上一致', async () => {
  // 只改正文、不改图，这个用例必须红。
  const page = await readFile(pagePath, 'utf8');

  assert.ok(page.includes(`num_samples=${CFG.numSamplesDefault}`), '正文必须给出源码默认窗口');
  assert.ok(page.includes(`窗口内最大值 ${BASE.maxValue}`));
  assert.ok(page.includes(`threshold=${CFG.threshold}`));
  assert.ok(page.includes(`触发线 ${BASE.trigger}`), '触发线必须与图一致');
  assert.ok(page.includes(`第 ${CFG.pollutedIndex} 个样本换成 ${CFG.pollutedValue}`));
  assert.ok(
    page.includes(`从 ${BASE.maxValue} 抬到 ${POLLUTED.maxValue}`),
    '污染后的 max 必须与图一致',
  );
  assert.ok(
    page.includes(`从 ${BASE.trigger} 抬到 ${POLLUTED.trigger}`),
    '污染后的触发线必须与图一致',
  );
  assert.ok(page.includes(`从 ${BASE_FLAGS} 变成 ${POLLUTED_FLAGS}`));
  assert.ok(page.includes(`loss 降到 ${CFG.lateLoss}`));
  assert.ok(page.includes(`${Math.round(LATE_RATIO)}× 于当前 loss`));

  assert.ok(page.includes(`[${CFG.validTokens.join(', ')}]`), '正文必须列出不等长 token 数');
  assert.ok(page.includes(`all_reduce(SUM)\` = ${AUX_SKEWED.measured}`) || page.includes(`= ${AUX_SKEWED.measured}`));
  assert.ok(page.includes(AUX_SKEWED.closed.join(' / ')), '正文必须给出逐 rank 的闭式值');
  assert.ok(page.includes(`高估 ${Number(Math.max(...AUX_SKEWED.ratio).toFixed(2))}×`));
  assert.ok(page.includes(`低估到 ${Number(Math.min(...AUX_SKEWED.ratio).toFixed(2))}×`));

  assert.ok(page.includes(`clip_grad\`（默认 ${CFG.clipGrad}）`) || page.includes(`默认 ${CFG.clipGrad}`));

  for (const name of NAMES) {
    assert.ok(page.includes(`assets/${name}`), `正文没有引用 ${name}`);
  }
});
