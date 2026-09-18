// 锁住 slime 训练闭环关键路径原理图的可执行契约：同步与 one-stage async 的周期、driver 等待（保存轮在 rollout_manager.save、非保存轮在 fence）、
// actor Timer 的 train_wait 拆段都由同一份 CFG 经 train.py / train_async.py 主循环的复现推导，
// 并且必须与 30_slime_rollout_optimization_analysis.md 正文引用的数值一致。
//
// 运行：node --test tools/figs/svg/lib/slime_rollout_cycle_figures.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

import { CFG, GEOM, closedForms, model, renderSvg, simulateAsync, simulateSync } from '../slime_rollout_cycle_figures.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const generator = join(here, '..', 'slime_rollout_cycle_figures.mjs');
const slimeDir = join(here, '..', '..', '..', '..', 'wiki', '02_engineering', '04_posttrain_frameworks', 'slime');
const pagePath = join(slimeDir, '30_slime_rollout_optimization_analysis.md');
const trackedSvg = join(slimeDir, 'assets', 'slime_rollout_cycle_timeline.svg');

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

test('闭式与主循环复现一致：同步相加，异步取两臂较大者再加发布', () => {
  const c = closedForms();
  assert.deepEqual(c, { genArm: 7, trainArm: 5, sync: 13, async: 8, driverWait: 2, oldBound: 10 });
  const s = simulateSync(CFG, 4);
  assert.deepEqual(s.timer.slice(1).map((r) => r.stepTime), [c.sync, c.sync, c.sync]);
  const a = simulateAsync(CFG, 5, 1);
  assert.deepEqual(a.tops, [7, 15, 23, 31, 39]);
  assert.deepEqual(a.cycles, [c.async, c.async, c.async, c.async]);
  assert.ok(c.oldBound > c.async, '在 max 外再加一次 driver 等待会高估周期');
  // driver 等待只出现在训练臂更短时；训练臂更长时等待为 0，周期仍是 max + 发布
  const slowTrain = { ...CFG, T: { ...CFG.T, train: 8 } };
  const cs = closedForms(slowTrain);
  assert.equal(cs.driverWait, 0);
  assert.deepEqual(simulateAsync(slowTrain, 4, 1).cycles, [cs.async, cs.async, cs.async]);
  assert.equal(simulateAsync(slowTrain, 4, 1).lanes.actor.filter((x) => x.kind === 'fence' || x.kind === 'rmsave').length, 0);
  // 保存轮：等待落在 ray.get(rollout_manager.save)，fence 随即返回
  const a1 = simulateAsync(CFG, 3, 1);
  assert.deepEqual(a1.lanes.actor.filter((x) => x.kind === 'fence' || x.kind === 'rmsave').map((x) => x.kind), ['rmsave', 'rmsave']);
  // 关掉 global dataset（不调 rollout_manager.save）时，同一段等待回到 fence
  const noDs = simulateAsync({ ...CFG, rolloutGlobalDataset: false }, 3, 1);
  assert.deepEqual(noDs.lanes.actor.filter((x) => x.kind === 'fence' || x.kind === 'rmsave').map((x) => [x.kind, x.end - x.start]), [['fence', 2], ['fence', 2]]);
  assert.deepEqual(noDs.cycles, a1.cycles);
});

test('--update-weights-interval 只改变 train_async.py：周期与版本滞后', () => {
  const m = model();
  assert.deepEqual(m.interval.syncSteps, [13, 13, 13, 13, 13]);
  assert.deepEqual(m.interval.syncLags, [0, 0, 0, 0, 0, 0]);
  assert.deepEqual(m.interval.async1Cycles, [8, 8, 8, 8, 8]);
  assert.deepEqual(m.interval.async1Lags, [0, 1, 1, 1, 1, 1]);
  assert.deepEqual(m.interval.async2Cycles, [7, 8, 7, 8, 7]);
  assert.deepEqual(m.interval.async2Lags, [0, 1, 2, 1, 2, 1]);
  // --save-interval 2：不保存的轮次在 fence 上等 3，保存轮在 rollout_manager.save 上等 2，周期不变
  assert.deepEqual(m.interval.altSaveWaits, [
    { round: 0, kind: 'fence', len: 3 },
    { round: 1, kind: 'rmsave', len: 2 },
    { round: 2, kind: 'fence', len: 3 },
  ]);
  assert.deepEqual(m.interval.altSaveCycles, [8, 8, 8]);
});

test('actor Timer：train_wait 窗口拆段与残差', () => {
  const m = model();
  const s = m.row.sync;
  assert.deepEqual([s.trainWait, s.train, s.stepTime, s.ratio], [10, 3, 13, 0.769]);
  assert.deepEqual(s.keyed, { pre: 1, save: 1, publish: 1 });
  assert.equal(s.residual, 7);
  assert.deepEqual(s.residualParts, { rollout: 6, data: 1, rmsave: 0, fence: 0 });
  assert.deepEqual(s.pieces.map((p) => p.kind), ['save', 'publish', 'rollout', 'data', 'pre']);
  const a = m.row.async;
  assert.deepEqual([a.trainWait, a.train, a.stepTime, a.ratio], [5, 3, 8, 0.625]);
  assert.deepEqual(a.keyed, { pre: 1, save: 1, publish: 1 });
  assert.equal(a.residual, 2);
  assert.deepEqual(a.residualParts, { rollout: 0, data: 0, rmsave: 2, fence: 0 });
  assert.deepEqual(a.pieces.map((p) => p.kind), ['save', 'rmsave', 'publish', 'pre']);
  assert.deepEqual(m.genIdle, { sync: 6, async: 1 });
  // 每个窗口：有键段 + 残差 = train_wait；train_wait + train = step_time
  for (const row of [...m.sync.timer, ...m.async.timer]) {
    assert.equal(row.keyed.pre + row.keyed.save + row.keyed.publish + row.residual, row.trainWait);
    assert.equal(row.trainWait + row.train, row.stepTime);
  }
});

test('生成器产出原理图，且与已跟踪的 SVG 一致', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'slime-cycle-'));
  try {
    const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
    assert.equal(run.status, 0, run.stderr || run.stdout);
    const svg = await readFile(join(outputDir, 'slime_rollout_cycle_timeline.svg'), 'utf8');
    assertInsideCanvas(svg);
    const { labels } = renderSvg();
    for (const l of labels) assert.ok(l.need + 4 <= l.width, `条内标签放不下：${l.label}（${l.need} > ${l.width}）`);
    const { W } = GEOM;
    assert.ok(GEOM.x0 + 32 * GEOM.unit <= W - 24, '时间轴超出面板');
    for (const needle of [
      '6 + 1 + 1 + 3 + 1 + 1 = 13',
      'max(6 + 1, 1 + 3 + 1) + 1 = 8',
      'driver 等待 2 已在 max 内',
      '排队 2',
      '不保存的轮次改在 fence 上等 3',
      'wait_time_ratio = 10/13 = 0.769',
      'wait_time_ratio = 5/8 = 0.625',
      'max(7, 4) + 2 + 1 = 10 把等待在 max 外又算一次',
      '轮首间隔 7、8 交替，滞后 1、2 交替',
      '残差 7（rollout_time 6 + 转换 1）',
      '残差 2（driver 等待：本轮在 rollout_manager.save 排队）',
      'perf/update_weights_time',
    ]) {
      assert.ok(svg.includes(needle), `SVG 必须出现 ${needle}`);
    }
    assert.doesNotMatch(svg, /\[\[/, 'SVG 不得泄漏 wikilink 标记');
    const tracked = await readFile(trackedSvg, 'utf8');
    assert.equal(tracked, svg, '已跟踪的 SVG 必须由当前生成器重新生成');
  } finally {
    await rm(outputDir, { recursive: true, force: true });
  }
});

test('正文引用的数值与模型一致', async () => {
  const page = await readFile(pagePath, 'utf8');
  const m = model();
  const c = m.closed;
  const T = CFG.T;
  const s = m.row.sync;
  const a = m.row.async;
  for (const needle of [
    `${T.rollout} + ${T.data} + ${T.pre} + ${T.train} + ${T.save} + ${T.publish} = ${c.sync}`,
    `\\max(${T.rollout}+${T.data},\\,${T.pre}+${T.train}+${T.save})+${T.publish}=${c.async}`,
    `driver 等待 ${c.driverWait}`,
    `在 fence 上等 ${m.interval.altSaveWaits[0].len}`,
    'ray.get(rollout_manager.save.remote(i))',
    '[[10_slime_end_to_end_iteration_analysis#5. 保存与恢复切口：只有同步入口对齐',
    `会得到 ${c.oldBound}`,
    `\`perf/train_wait_time\` 为 ${s.trainWait}`,
    `${s.trainWait}/${s.stepTime} = ${s.ratio}`,
    `\`perf/train_wait_time\` 为 ${a.trainWait}`,
    `${a.trainWait}/${a.stepTime} = ${a.ratio}`,
    `残差 ${s.residual}`,
    `残差 ${a.residual}`,
    `轮首间隔 ${m.interval.async2Cycles.slice(0, 2).join('、')} 交替`,
    `版本滞后 ${m.interval.async2Lags.slice(1, 3).join('、')} 交替`,
    `推理侧每轮空闲 ${m.genIdle.sync}`,
    'assets/slime_rollout_cycle_timeline.svg',
  ]) {
    assert.ok(page.includes(needle), `正文必须出现 ${needle}`);
  }
  assert.doesNotMatch(page, /github\.com\/THUDM\/slime\/blob\/[0-9a-f]+\/[^)]*#L\d+/, '正文不再保留 path:line 链接');
  assert.ok(page.includes('THUDM/slime@4c193f1f37509cca70f0e88807a9305b70f63f4e'), '页头必须钉住当前基线');
  assert.doesNotMatch(page, /旧版本页|旧式/, '正文不叙述页面历史');
});
