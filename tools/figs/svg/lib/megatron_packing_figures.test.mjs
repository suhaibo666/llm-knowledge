// 锁住打包 / 动态 CP 图示的可执行契约：两套分组算法都复刻自源码，逐 rank 工作量、
// 关键路径与不均比全部算出来，并与 29_megatron_packed_dataset_dynamic_cp_analysis.md 正文对齐。
//
// 运行：node --test tools/figs/svg/lib/megatron_packing_figures.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

import {
  CFG, TOTAL_GPUS, DPB, DPB_WORK, DCP, DCP_WORK, ALT,
  DPB_CRIT, DCP_CRIT, DPB_MAX_SPREAD, DCP_MAX_SPREAD,
  CONN_ALL2ALL, CONN_GATHER,
  dcpGpusNeeded, workload, dpBalancedGroups, dynamicCpGroups, critical,
} from '../megatron_packing_figures.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const generator = join(here, '..', 'megatron_packing_figures.mjs');
const assetDir = join(
  here, '..', '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks',
  'megatron-lm', 'assets',
);
const pagePath = join(
  here, '..', '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks',
  'megatron-lm', '29_megatron_packed_dataset_dynamic_cp_analysis.md',
);

const NAMES = [
  'megatron_packing_scheduler_replay.svg',
  'megatron_packing_pipeline.svg',
  'megatron_packing_reroute.svg',
];

const mu = (v) => `${(v / 1e6).toFixed(2)}M`;
const fx = (v, d = 2) => v.toFixed(d).replace(/\.?0+$/, '');

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

test('dcp_gpus_needed：向上取 2 的幂，并受 min_cp_size 抬底', () => {
  const m = CFG.maxSeqLenPerRank;
  assert.equal(dcpGpusNeeded(m, m, 1), 1, '刚好一个 rank 装得下');
  assert.equal(dcpGpusNeeded(m / 2, m, 1), 1, '短于一个 rank 也只要 1');
  assert.equal(dcpGpusNeeded(m * 2, m, 1), 2);
  assert.equal(dcpGpusNeeded(m * 2 + 1, m, 1), 4, '略超 2 倍就跳到 4，取 2 的幂');
  assert.equal(dcpGpusNeeded(m, m, 4), 4, 'min_cp_size 抬底');
  // 工作量代理：CP 翻倍，单 rank 工作量减半
  assert.equal(workload(4096, 2), workload(4096, 1) / 2);
});

test('固定 CP 的 first-fit 对样本顺序敏感', () => {
  // 本页算例：4096 排在第二位，于是它独占一箱
  assert.deepEqual(DPB.packed.map((g) => g.map((i) => CFG.seqlens[i])), [
    [1024], [4096], [2048, 1024], [2048, 1024], [2048], [1024],
  ]);
  assert.equal(DPB_WORK.length, 3, '六箱摊成 3 个 microbatch');
  // 同一批样本换个顺序，装箱结果就不同 —— 这正是 16× 那一格的成因
  const sorted = [...CFG.seqlens].sort((a, b) => b - a);
  const other = dpBalancedGroups(sorted, CFG);
  assert.notDeepEqual(
    other.packed.map((g) => g.map((i) => sorted[i])),
    DPB.packed.map((g) => g.map((i) => CFG.seqlens[i])),
    'first-fit 的结果必须随顺序改变',
  );
});

test('两条数据面的逐 rank 工作量、关键路径与不均比', () => {
  // 固定 CP：mb0 里 4096 那一箱 8.39M，对面只有 1024 的 0.52M
  assert.deepEqual(DPB_WORK[0].map(mu), ['0.52M', '0.52M', '8.39M', '8.39M']);
  assert.equal(DPB_MAX_SPREAD, 16);
  assert.equal(mu(DPB_CRIT), '13.11M');

  // 动态 CP：4096 用 CP=2 摊到 r0/r1，两条 2048 各占一个 rank
  assert.deepEqual(DCP_WORK[0].map(mu), ['8.39M', '8.39M', '4.19M', '4.19M']);
  assert.equal(DCP_MAX_SPREAD, 2);
  assert.equal(mu(DCP_CRIT), '10.49M');
  assert.equal(DCP_WORK.length, 2, '动态 CP 用两轮消化完 8 条样本');

  // 立论：动态 CP 的最坏一格更均衡，且关键路径更短
  assert.ok(DCP_MAX_SPREAD < DPB_MAX_SPREAD);
  assert.ok(DCP_CRIT < DPB_CRIT);

  // 第一轮确实留了 leftovers，且下一轮把它们消化完
  assert.ok(DCP[0].leftovers.length > 0, '第一轮必须有装不下的样本');
  assert.equal(DCP[DCP.length - 1].leftovers.length, 0, '最后一轮必须清空');
  const scheduled = DCP.flatMap((r) => r.sampleIdsPerGpu.flat());
  assert.equal(new Set(scheduled).size, CFG.seqlens.length, '每条样本都被排到过');
});

test('动态 CP 优化的是关键路径，不是每格的不均比', () => {
  // 换一批长度分布（同样含超长样本）：docstring 说的判据是 critical-path rank workload，
  // 所以关键路径不该更差 —— 但每个 microbatch 内部的不均比可能反而更大。
  assert.ok(ALT.dcpCrit <= ALT.dpbCrit, '关键路径不该更差');
  assert.ok(ALT.dcpSpread > ALT.dpbSpread, '这一组恰好展示了不均比可能更差');
  // 主算例上两个指标同向，所以正文必须把「同向」限定在算例上，而不是当成普适结论
  assert.ok(DCP_CRIT < DPB_CRIT && DCP_MAX_SPREAD < DPB_MAX_SPREAD);
});

test('分组结果确实由数据与配置决定', () => {
  // min_cp_size 是活旋钮：把它顶到整组，最高那条被摊到全部 rank，关键路径反而更短。
  // 注意 min_cp_size=2 在本算例上得到与 1 完全相同的 exec —— 落位不同但工作量巧合相等，
  // 所以这里用 4 才能可靠地证明旋钮生效。
  const raised = dynamicCpGroups(CFG.seqlens, {
    totalGpus: TOTAL_GPUS, maxSeqLenPerRank: CFG.maxSeqLenPerRank, minCpSize: TOTAL_GPUS,
  });
  assert.notDeepEqual(
    raised.map((r) => r.execTimes),
    DCP_WORK,
    'min_cp_size 顶到整组必须改变分组',
  );
  assert.ok(
    critical(raised.map((r) => r.execTimes)) < DCP_CRIT,
    '本算例下顶到整组反而缩短关键路径 —— 说明默认的最小 CP 并不总是最优',
  );
  // 源码自陈的不变量：填充之后没有空 rank
  for (const round of DCP) {
    assert.ok(
      round.microBatches.every((mb) => mb.length > 0),
      '每个 microbatch 都不该留下空 DPxCP rank',
    );
  }
});

test('生成器同步产出三张图，且图上的关键量与算例一致', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'megatron-packing-figures-'));
  const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
  assert.equal(run.status, 0, run.stderr || run.stdout);

  const [replay, pipeline, reroute] = await Promise.all(
    NAMES.map((name) => readFile(join(outputDir, name), 'utf8')),
  );

  // ---- 图 1：原理图 ----
  assert.ok(replay.includes(`[${CFG.seqlens.join(', ')}]`), '必须列出原始顺序的样本长度');
  assert.ok(replay.includes(`关键路径合计 ${mu(DPB_CRIT)}`));
  assert.ok(replay.includes(`关键路径合计 ${mu(DCP_CRIT)}`));
  assert.ok(replay.includes(`最坏一格不均 ${fx(DPB_MAX_SPREAD)}×`));
  assert.ok(replay.includes(`最坏一格不均 ${fx(DCP_MAX_SPREAD)}×`));
  // 每个 microbatch 的不均比都要标出来
  const spread = (row) => Math.max(...row) / Math.max(Math.min(...row), 1e-9);
  [...DPB_WORK, ...DCP_WORK].forEach((row) => {
    assert.ok(replay.includes(`max/min = ${fx(spread(row))}×`), `缺不均比 ${fx(spread(row))}`);
  });
  assert.ok(replay.includes(`剩 ${DCP[0].leftovers.length} 条留给下一轮`));

  // ---- 图 2：九步 ----
  for (const n of ['①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨']) {
    assert.ok(pipeline.includes(n), `图 2 缺第 ${n} 步`);
  }
  assert.ok(pipeline.includes('get_groups_and_subsamples'));
  assert.ok(pipeline.includes('唯一分叉'));
  assert.ok(pipeline.includes('没有 PP 组广播'));
  assert.ok(pipeline.includes('DP 组 all-gather，逐 key'));
  assert.ok(!pipeline.includes('[['), '图上不允许漏出 wikilink 语法');

  // ---- 图 3：reroute ----
  assert.ok(reroute.includes(`NCCL all-to-all（${CONN_ALL2ALL} 对连接）`));
  assert.ok(reroute.includes(`${CFG.cpSize} × ${CFG.dpSize}² = ${CONN_GATHER} 对`));
  for (let rank = 0; rank < TOTAL_GPUS; rank += 1) {
    assert.ok(reroute.includes(`DCP rank ${rank}`), `缺 rank ${rank}`);
    assert.ok(
      reroute.includes(`Σ seq²/cp = ${mu(DCP[0].execTimes[rank])}`),
      `缺 rank ${rank} 的工作量`,
    );
  }
  assert.ok(reroute.includes('逐字节相同'));

  NAMES.forEach((name, i) => assertInsideCanvas([replay, pipeline, reroute][i], name));

  const tracked = await Promise.all(NAMES.map((name) => readFile(join(assetDir, name), 'utf8')));
  NAMES.forEach((name, i) => {
    assert.equal(
      tracked[i],
      [replay, pipeline, reroute][i],
      `${name} 与生成器输出不一致：重跑 node tools/figs/svg/megatron_packing_figures.mjs`,
    );
  });
});

test('页面正文引用的数值与图上一致', async () => {
  // 只改正文、不改图，这个用例必须红。
  const page = await readFile(pagePath, 'utf8');

  assert.ok(page.includes(`[${CFG.seqlens.join(', ')}]`), '正文必须列出算例样本长度');
  assert.ok(page.includes(`DP=${CFG.dpSize}、CP=${CFG.cpSize}`), '正文必须点明并行度');
  assert.ok(page.includes(String(CFG.maxSeqLenPerRank)), '正文必须给出每 rank 容量');
  assert.ok(
    page.includes(`${CFG.maxSeqLenPerRank}\\times2=${CFG.maxSeqLenPerRank * CFG.cpSize}`) ||
      page.includes(`${CFG.maxSeqLenPerRank}\\times${CFG.cpSize}=${CFG.maxSeqLenPerRank * CFG.cpSize}`),
    '正文必须给出固定 CP 的箱容量',
  );
  assert.ok(page.includes(`**${fx(DPB_MAX_SPREAD)}×**`), '固定 CP 的最坏不均必须与图一致');
  assert.ok(page.includes(`**${fx(DCP_MAX_SPREAD)}×**`), '动态 CP 的最坏不均必须与图一致');
  assert.ok(page.includes(mu(DPB_CRIT)), '固定 CP 的关键路径必须与图一致');
  assert.ok(page.includes(mu(DCP_CRIT)), '动态 CP 的关键路径必须与图一致');
  assert.ok(page.includes(`| ${DPB_WORK.length} | ${DCP_WORK.length} |`), 'microbatch 数必须与图一致');
  assert.ok(page.includes(`${CFG.cpSize}\\times${CFG.dpSize}^2=${CONN_GATHER}`), 'gather 连接数必须与图一致');
  assert.ok(page.includes(`${CONN_ALL2ALL} 对全连接 P2P`), 'all-to-all 连接数必须与图一致');
  assert.ok(page.includes(`(1+0.05)`) || page.includes('1+0.05'), '工作量上限的 delta 必须出现在正文');

  // 两条限定结论也必须与复刻结果同源
  assert.ok(page.includes(`[${CFG.altSeqlens.join(', ')}]`), '第二组算例的样本长度必须出现在正文');
  assert.ok(page.includes(`从 ${mu(ALT.dpbCrit)} 降到 ${mu(ALT.dcpCrit)}`));
  assert.ok(page.includes(`从 ${fx(ALT.dpbSpread)}× **升到** ${fx(ALT.dcpSpread)}×`));
  const raisedCrit = critical(
    dynamicCpGroups(CFG.seqlens, {
      totalGpus: TOTAL_GPUS, maxSeqLenPerRank: CFG.maxSeqLenPerRank, minCpSize: TOTAL_GPUS,
    }).map((r) => r.execTimes),
  );
  assert.ok(
    page.includes(`从 ${mu(DCP_CRIT)} 降到 ${mu(raisedCrit)}`),
    'min_cp 顶到整组后的关键路径必须与复刻结果一致',
  );

  for (const name of NAMES) {
    assert.ok(page.includes(`assets/${name}`), `正文没有引用 ${name}`);
  }
});
