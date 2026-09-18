// 锁住 slime 容错原理图与指标 x 轴图的可执行契约：图上每个数字都由同一份 CFG / AXES 经源码规则的复现推导，
// 并且必须与 18_slime_fault_tolerance_observability_analysis.md 正文引用的数值一致。
//
// 运行：node --test tools/figs/svg/lib/slime_fault_recovery_figures.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

import { AXES, CFG, axesModel, driverRun, model, restart, shouldRunPeriodic } from '../slime_fault_recovery_figures.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const generator = join(here, '..', 'slime_fault_recovery_figures.mjs');
const slimeDir = join(here, '..', '..', '..', '..', 'wiki', '02_engineering', '04_posttrain_frameworks', 'slime');
const pagePath = join(slimeDir, '18_slime_fault_tolerance_observability_analysis.md');
const trackedSvg = join(slimeDir, 'assets', 'slime_fault_recovery_timeline.svg');
const trackedAxesSvg = join(slimeDir, 'assets', 'slime_metric_step_axes.svg');

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

test('检测上界与 CI 等待复现 health_monitor 与 _try_ci_fault_injection', () => {
  const b = Object.fromEntries(model().bounds.map((x) => [x.key, x]));
  assert.deepEqual([b.ft.detect, b.ft.general, b.ft.wait], [15, 45, 20]);
  assert.deepEqual([b.defaults.detect, b.defaults.wait], [60, 65]);
  assert.deepEqual([b.doc.interval, b.doc.timeout, b.doc.firstWait], [10, 5, 300]);
});

test('故障注入时间线复现 generate 注入、降容与更新边界重建', () => {
  const run = model().run;
  assert.deepEqual(run.rounds.map((r) => r.liveDuringRollout), [4, 4, 3]);
  assert.deepEqual(run.rounds.map((r) => r.injected), [false, false, true]);
  assert.deepEqual(run.rounds.map((r) => r.rebuiltAtUpdate), [0, 0, 1]);
  assert.equal(run.roundsServedAfterRebuild, 0, '注入落在最后一轮，重建之后没有 rollout');
});

test('保存点与续训复现 should_run_periodic_action 与 start = iteration + 1', () => {
  const c = model().ckpt;
  assert.deepEqual(c.saves, [1, 3, 5]);
  assert.deepEqual(c.normal, { start: 4, loadId: 3, cursor: 'restored' });
  assert.deepEqual(c.torn, { start: 4, loadId: 3, cursor: 'reset' });
  assert.equal(c.common, 1);
  assert.deepEqual(c.fallback, { start: 2, loadId: 1, cursor: 'restored' });
  assert.deepEqual(c.asyncCase, { start: 2, loadId: 1, cursor: 'restored' });
  // 最后一轮总会保存，即使不整除
  assert.equal(shouldRunPeriodic(4, 2, 5), true);
  assert.equal(shouldRunPeriodic(0, null, 5), false);
  assert.deepEqual(restart({ trainerLatest: 0, dsIds: [] }), { start: 1, loadId: 0, cursor: 'reset' });
});

test('生成器产出原理图，且与已跟踪的 SVG 一致', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'slime-fault-'));
  try {
    const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
    assert.equal(run.status, 0, run.stderr || run.stdout);
    const svg = await readFile(join(outputDir, 'slime_fault_recovery_timeline.svg'), 'utf8');
    assertInsideCanvas(svg);
    for (const needle of ['5 + 10 = 15 s', '5 + 10 + 5 = 20 s', '5 + 4 × 10 = 45 s', '3 个 engine 生成', 'onload_weights', '服务过的轮数 = 0', 'rollout 1、3、5', 'start_rollout_id = 4', '游标从 0 重来', '--ckpt-step 1', 'start_rollout_id = 2']) {
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
  const b = Object.fromEntries(m.bounds.map((x) => [x.key, x]));
  for (const needle of [
    `${CFG.ft.interval} + ${CFG.ft.timeout} + 5 = ${b.ft.wait} 秒`,
    `检测上界 ${b.ft.detect} 秒`,
    `检测上界 ${b.defaults.detect} 秒、CI 等待 ${b.defaults.wait} 秒`,
    `first-wait ${b.doc.firstWait}、interval ${b.doc.interval}、timeout ${b.doc.timeout}`,
    `${CFG.ft.interval} + ${CFG.engines} × ${CFG.ft.timeout} = ${b.ft.general} 秒`,
    'ExternalRolloutServer.recover',
    '`range(2, 2)` 一轮也不跑',
    `rollout ${m.ckpt.saves.join('、')} 保存`,
    `start_rollout_id = 3 + 1 = ${m.ckpt.torn.start}`,
    `\`load(${m.ckpt.torn.loadId})\``,
    `--ckpt-step ${m.ckpt.common}`,
    `start_rollout_id = ${m.ckpt.fallback.start}`,
    `服务过的轮数为 ${m.run.roundsServedAfterRebuild}`,
    '剩余 3 个 engine',
  ]) {
    assert.ok(page.includes(needle), `正文必须出现 ${needle}`);
  }
  assert.ok(page.includes('assets/slime_fault_recovery_timeline.svg'), '正文必须引用原理图');
  assert.ok(page.includes('#### 2.2.2 推理引擎的局部恢复：检测、整组标死、在更新边界重建'), '13、19、31 页入链标签引用的 §2.2.2 必须存在');
  assert.doesNotMatch(page, /github\.com\/THUDM\/slime\/blob\/[0-9a-f]+\/[^)]*#L\d+/, '正文不再保留 path:line 链接');
});

test('x 轴折算复现 compute_rollout_step 与 build_dp_schedule 的步数', () => {
  const m = axesModel();
  assert.equal(AXES.rolloutBatchSize * AXES.nSamplesPerPrompt, 32);
  assert.deepEqual([m.stepsEven, m.stepsUneven, m.droppedUneven], [2, 1, 8]);
  assert.deepEqual(m.trainEven, [[0, 1], [2, 3], [4, 5], [6, 7]]);
  assert.deepEqual(m.rolloutDefault, [0, 1, 2, 3]);
  assert.deepEqual(m.rolloutEven, [0, 2, 4, 6], '整除时折算值等于该批第一个 train/step');
  assert.deepEqual(m.rolloutEven, m.trainEven.map((s) => s[0]));
  assert.deepEqual(m.trainUneven.map((s) => s[0]), [0, 1, 2, 3]);
  assert.deepEqual(m.rolloutUneven, [0, 1, 2, 4], '不整除时 rollout 3 错位');
});

test('sync 与 async 入口复现生成权重的已训批次与计时归属', () => {
  const m = axesModel();
  assert.deepEqual(m.sync.trainedAtGenerate, [0, 1, 2, 3]);
  assert.deepEqual(m.async1.trainedAtGenerate, [0, 0, 1, 2]);
  assert.deepEqual(m.async1.parallelTrain, [null, 0, 1, 2]);
  assert.deepEqual(m.sync.parallelTrain, [null, null, null, null]);
  assert.deepEqual(m.sync.perfUpdateFrom, [['init'], [0], [1], [2]]);
  assert.deepEqual(m.async1.perfUpdateFrom, m.sync.perfUpdateFrom);
  // interval 2：两次训练才推送一次，async 生成用的权重更旧
  assert.deepEqual(driverRun('async', 4, 2).trainedAtGenerate, [0, 0, 0, 2]);
});

test('生成器产出 x 轴图，且与已跟踪的 SVG 一致', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'slime-axes-'));
  try {
    const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
    assert.equal(run.status, 0, run.stderr || run.stdout);
    const svg = await readFile(join(outputDir, 'slime_metric_step_axes.svg'), 'utf8');
    assertInsideCanvas(svg);
    for (const needle of ['0、1', '6、7', '无（初始权重）', '批 0–1', 'train 2 之后', '初始推送', '尾部 8 条丢弃', 'rollout 3 折算成 4']) {
      assert.ok(svg.includes(needle), `SVG 必须出现 ${needle}`);
    }
    assert.doesNotMatch(svg, /\[\[/, 'SVG 不得泄漏 wikilink 标记');
    const tracked = await readFile(trackedAxesSvg, 'utf8');
    assert.equal(tracked, svg, '已跟踪的 x 轴 SVG 必须由当前生成器重新生成');
  } finally {
    await rm(outputDir, { recursive: true, force: true });
  }
});

test('指标落点小节引用的数值、开关与模型一致', async () => {
  const page = await readFile(pagePath, 'utf8');
  const m = axesModel();
  assert.ok(page.includes('### 4.2 指标落点与 x 轴'), '配置指南页入链的指标落点小节必须存在');
  assert.ok(page.includes('assets/slime_metric_step_axes.svg'), '正文必须引用 x 轴图');
  for (const needle of [
    `每批 ${AXES.rolloutBatchSize * AXES.nSamplesPerPrompt} 条 rollout`,
    m.trainEven.map((s) => s.join('、')).join(' / '),
    `折算成 ${m.rolloutEven.join('、')}`,
    `尾部 ${m.droppedUneven} 条 rollout 被丢弃`,
    `\`train/step\` 是 ${m.trainUneven.map((s) => s[0]).join('、')}`,
    `折算出的 \`rollout/step\` 却是 ${m.rolloutUneven.join('、')}`,
    'rollout 1 与 rollout 0 一样用初始权重',
    'rollout 3 用训过批 0–1 的权重',
    'train k−1 之后那次推送',
  ]) {
    assert.ok(page.includes(needle), `正文必须出现 ${needle}`);
  }
  for (const flag of [
    '--use-wandb', '--wandb-mode', '--wandb-dir', '--wandb-key', '--wandb-host', '--wandb-team', '--wandb-group',
    '--wandb-run-id', '--disable-wandb-random-suffix', '--wandb-always-use-train-step', '--use-tensorboard',
    '--tb-project-name', '--tb-experiment-name', '--log-multi-turn', '--custom-rollout-log-function-path',
    '--custom-eval-rollout-log-function-path', '--memory-snapshot-dir',
  ]) {
    assert.ok(page.includes(`| \`${flag}\` |`), `落点开关表必须有 ${flag} 一行`);
  }
  assert.doesNotMatch(page, /--profile-target/, '4c193f1f 已删除 --profile-target');
});
