// 锁住 slime 迭代时序原理图的可执行契约：图上每个数字都由同一份 CFG 经 train.py / train_async.py /
// fully_async_rollout.py 控制流的复现推导，并且必须与 10_slime_end_to_end_iteration_analysis.md 正文引用的数值一致。
//
// 运行：node --test tools/figs/svg/lib/slime_iteration_timeline_figures.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

import { CFG, LANES, fullyCapacity, model, shouldRunPeriodic, simulate, textWidth } from '../slime_iteration_timeline_figures.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const generator = join(here, '..', 'slime_iteration_timeline_figures.mjs');
const slimeDir = join(here, '..', '..', '..', '..', 'wiki', '02_engineering', '04_posttrain_frameworks', 'slime');
const pagePath = join(slimeDir, '10_slime_end_to_end_iteration_analysis.md');
const trackedSvg = join(slimeDir, 'assets', 'slime_iteration_timeline.svg');

const FONT = { ti: 19, su: 12, pt: 14, tx: 12, sm: 10.5, xs: 9.5, cap: 11.5, inv: 10.5, ab: 11 };
const unescape = (s) => s.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');

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
  for (const [, x, y, cls, anchor, body] of svg.matchAll(
    /<text x="(-?\d+(?:\.\d+)?)" y="(-?\d+(?:\.\d+)?)" class="(\w+)" text-anchor="(\w+)">([^<]*)<\/text>/g,
  )) {
    const size = FONT[cls];
    assert.ok(size, `未知文字样式 ${cls}`);
    const width = textWidth(unescape(body), size);
    const left = anchor === 'start' ? Number(x) : anchor === 'middle' ? Number(x) - width / 2 : Number(x) - width;
    assert.ok(left >= 0 && left + width <= w, `文字横向越界 "${body}" [${left.toFixed(1)}, ${(left + width).toFixed(1)}]`);
    assert.ok(Number(y) - size >= 0 && Number(y) <= h, `文字纵向越界 "${body}"`);
  }
}

test('周期保存规则复现 should_run_periodic_action', () => {
  assert.deepEqual([0, 1, 2, 3].filter((id) => shouldRunPeriodic(id, CFG.saveInterval, CFG.numRollout)), [1, 3]);
  assert.equal(shouldRunPeriodic(2, 2, 3), true, '最后一轮总会保存');
  assert.equal(shouldRunPeriodic(0, null, 4), false);
});

test('三个 driver 的版本年龄、发布次数与保存切口', () => {
  const m = model();
  assert.deepEqual(m.ages.sync, [0, 0, 0, 0]);
  assert.deepEqual(m.ages.async1, [0, 1, 1, 1]);
  assert.deepEqual(m.ages.async2, [0, 1, 2, 1]);
  assert.deepEqual(m.publishCounts, { sync: 5, async1: 5, async2: 3, fully: 5 });
  // train.py：save(1) 时 RolloutManager 空闲，游标与已训练组数一致
  assert.deepEqual([m.save1.sync.cursor, m.save1.sync.trainedGroups, m.save1.sync.lead], [4, 4, 0]);
  // train_async.py：save(1) 排在 generate(2) 之后执行，游标越过一批
  for (const [key, submit, exec] of [['async1', 10, 11], ['async2', 9, 10]]) {
    const s = m.save1[key];
    assert.deepEqual([s.submitAt, s.execAt, s.cursor, s.lead], [submit, exec, 6, CFG.rolloutBatchSize], key);
    const gen2 = m.lanes[key].generates.find((g) => g.id === 2);
    assert.ok(s.execAt >= gen2.endAt, `${key}: save(1) 必须在 generate(2) 完成之后执行`);
  }
  // 最后一轮不提交下一次生成，保存对齐
  for (const key of ['sync', 'async1', 'async2']) assert.equal(m.lanes[key].saves.find((s) => s.id === 3).lead, 0, key);
  // 周期：同步相加；async 发布轮取 max(生成臂, 训练臂 + 保存) + 发布（与 Rollout 优化页 §3.3 的公式一致）
  const starts = (key) => m.lanes[key].publishes.map((p) => p.startAt);
  assert.deepEqual(starts('sync').slice(1).map((x, i, a) => (i ? x - a[i - 1] : null)).slice(1), [6, 6, 6]);
  assert.deepEqual(starts('async1').slice(1, 4).map((x, i, a) => (i ? x - a[i - 1] : null)).slice(1), [Math.max(CFG.ticks.generate, CFG.ticks.train) + CFG.ticks.publish, 4]);
  // 默认 rollout 的异步入口里，发布窗口不与任何生成重叠（发布前等 future）
  for (const key of ['async1', 'async2']) {
    const lane = m.lanes[key];
    for (const p of lane.publishes) {
      for (const g of lane.generates) assert.ok(p.startAt >= g.endAt || p.endAt <= g.startAt, `${key}: 发布 ${p.tag} 与生成 ${g.id} 重叠`);
    }
  }
});

test('fully-async 的池、发布中止回队、积压与保存游标', () => {
  const m = model();
  assert.deepEqual(fullyCapacity(), { pool: 4, semaphore: 4, generatingGroups: 2 });
  const f = m.lanes.fully;
  assert.deepEqual(m.ages.fully, [0, 1, 2, 2]);
  assert.deepEqual([f.generates.find((g) => g.id === 2).startAt, f.generates.find((g) => g.id === 2).endAt], [7, 12]);
  // 默认 mode=abort：每次发布开始时在途组全部中止、回 buffer
  assert.deepEqual(m.fullyAbortedAtPublish, [0, 1, 0, 1, 2]);
  assert.deepEqual(f.publishes.map((p) => p.aborted), [[], [4], [], [7], [9, 10]]);
  const g4 = f.groups[4];
  assert.deepEqual(g4.aborts, [6]);
  assert.deepEqual(g4.pulls, [3, 6], '回队组在中止的同一时刻被补位取回');
  assert.deepEqual(g4.runs.map((r) => [r.startAt, r.endAt, r.versions]), [[5, 6, [0]], [11, 12, [1]]]);
  assert.deepEqual(m.fullyResumedConsumed, [{ id: 2, gid: 4, versions: [0, 1] }]);
  // 取回不推进游标：12 个组、游标 12，回队不新增
  assert.equal(f.cursor, f.groups.length);
  const tr2 = f.trains.find((t) => t.id === 2);
  const tr3 = f.trains.find((t) => t.id === 3);
  assert.deepEqual(tr2.gids, [6, 4]);
  assert.deepEqual([tr2.gids[tr2.ages.indexOf(2)], tr3.gids[tr3.ages.indexOf(2)]], [4, 5]);
  const g5 = f.groups[5];
  const g6 = f.groups[6];
  assert.deepEqual([g5.doneAt, g5.consumedBy, g5.versions], [12, 3, [1]]);
  assert.deepEqual([g6.doneAt, g6.consumedBy, g6.versions], [11, 2, [1]]);
  const s1 = m.save1.fully;
  assert.deepEqual([s1.submitAt, s1.execAt, s1.cursor, s1.trainedGroups], [9, 12, 9, 4]);
  assert.deepEqual([m.fullyLead.nextBatch, m.fullyLead.queued, m.fullyLead.pool, m.fullyLead.buffered], [2, 1, 2, 0]);
  assert.equal(s1.lead, m.fullyLead.nextBatch + m.fullyLead.queued + m.fullyLead.pool + m.fullyLead.buffered, '游标领先 = 下一批 + 完成队列 + 池 + buffer');
  for (const gid of s1.pool) assert.ok(f.groups[gid].runs.every((r) => r.startAt > s1.execAt), '池内两组保存时仍在等信号量');
  const s3 = f.saves.find((s) => s.id === 3);
  assert.deepEqual([s3.queued.length, s3.pool.length, s3.lead], [0, 4, 4]);
  // 消费语义：有 generate 在执行时每个时刻都 drain；每批恰好 rollout_batch_size 组，没有组被消费两次
  const consumed = f.trains.flatMap((t) => t.gids);
  assert.equal(new Set(consumed).size, consumed.length);
  for (const t of f.trains) assert.equal(t.gids.length, CFG.rolloutBatchSize);
  for (const g of f.groups) {
    if (g.doneAt === null || g.consumedAt === null) continue;
    const active = f.generates.find((x) => x.startAt <= g.doneAt && x.endAt >= g.doneAt);
    if (active) assert.ok(g.consumedAt === g.doneAt || active.endAt === g.doneAt, `组 ${g.gid} 在 generate ${active.id} 执行中完成却没被立即取走`);
  }
  // 生成中的组不超过整组放行上限，暂停期间没有组在生成
  for (let t = 0; t <= f.endAt; t += 1) {
    const gen = f.groups.filter((g) => g.runs.some((r) => r.startAt <= t && (r.endAt === null || r.endAt > t))).length;
    assert.ok(gen <= 2, `t=${t} 生成中 ${gen} 组`);
    if (f.publishes.some((p) => p.startAt <= t && t < p.endAt)) assert.equal(gen, 0, `t=${t} 暂停期间不应有组在生成`);
  }
  // 改变 CFG 会重新计算：组时长全为 6 时年龄与游标随之改变
  const uniform = simulate(LANES[3], { ...CFG, fully: { ...CFG.fully, groupTicks: Array(16).fill(6) } });
  assert.notDeepEqual(uniform.trains.map((t) => t.maxAge), [0, 1, 2, 2]);
});

test('生成器产出原理图，且与已跟踪的 SVG 一致', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'slime-iteration-'));
  try {
    const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
    assert.equal(run.status, 0, run.stderr || run.stdout);
    const svg = await readFile(join(outputDir, 'slime_iteration_timeline.svg'), 'utf8');
    assertInsideCanvas(svg);
    for (const needle of [
      '年龄 0、0、0、0 · 发布 5 次',
      '年龄 0、1、1、1 · 发布 5 次',
      '年龄 0、1、2、1 · 发布 3 次',
      '游标 6 &gt; 已训 4 组，从 1 恢复跳过 2 组',
      '游标 9 = 已训 4 + 取2 的 2 + 完成队列 1 + 池内 2',
      '从 1 恢复跳过 5 组',
      '同时生成 2 组',
      '续v1',
      '回队后接前缀续生成',
      '+ buffer 0',
      '训3 龄2',
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
  const s1 = m.save1;
  const stepsPerRound = (32 * 8) / 128;
  const f4 = m.lanes.fully.groups[4];
  for (const needle of [
    `年龄依次为 ${m.ages.sync.join('、')}，发布 ${m.publishCounts.sync} 次`,
    `同步一轮是 ${CFG.ticks.generate} + ${CFG.ticks.train} + ${CFG.ticks.publish} = ${m.lanes.sync.publishes[2].startAt - m.lanes.sync.publishes[1].startAt} 个时刻`,
    `interval 1 则每 ${m.lanes.async1.publishes[2].startAt - m.lanes.async1.publishes[1].startAt} 个时刻发布一次`,
    `年龄依次为 ${m.ages.async1.join('、')}，发布仍是 ${m.publishCounts.async1} 次`,
    `年龄依次为 ${m.ages.async2.join('、')}，发布只有 ${m.publishCounts.async2} 次`,
    `最大年龄依次为 ${m.ages.fully.join('、')}`,
    `游标 ${s1.sync.cursor} = 已训 ${s1.sync.trainedGroups} 组`,
    `interval 1 的保存在 t=${s1.async1.submitAt} 提交、t=${s1.async1.execAt} 执行，interval 2 在 t=${s1.async2.submitAt} 提交、t=${s1.async2.execAt} 执行，两者保存的都是游标 ${s1.async1.cursor}`,
    `跳过 ${s1.async1.lead} 组`,
    `从 t=${s1.async2.submitAt} 等到 t=${s1.async2.execAt}`,
    `在 t=${s1.fully.submitAt} 提交、t=${s1.fully.execAt} 才执行，driver 被挡了 ${s1.fully.execAt - s1.fully.submitAt} 个时刻`,
    `游标 ${s1.fully.cursor} = 已训 ${s1.fully.trainedGroups} + 下一批 ${m.fullyLead.nextBatch} + 完成队列 ${m.fullyLead.queued} + 池内 ${m.fullyLead.pool}`,
    '两个入口（fully-async 沿用 `train_async.py`）',
    `从 rollout 1 恢复会跳过 ${s1.fully.lead} 组`,
    `池内仍有 ${m.lanes.fully.saves.find((s) => s.id === 3).pool.length} 组`,
    `池 ${m.capacity.pool} 组、信号量 ${m.capacity.semaphore} 条请求，每组 ${CFG.fully.nSamplesPerPrompt} 条样本`,
    `模型简化为整组放行，所以同时生成 ${m.capacity.generatingGroups} 组`,
    `取 2 从 t=${m.lanes.fully.generates.find((g) => g.id === 2).startAt} 等到 t=${m.lanes.fully.generates.find((g) => g.id === 2).endAt}`,
    `$v_4$ 发布开始时中止 ${m.fullyAbortedAtPublish[4]} 组`,
    `t=${f4.runs[1].startAt} 才接着 $v_0$ 前缀用 $v_1$ 续生成`,
    `组 6 在 t=${m.lanes.fully.groups[6].doneAt}、组 4 与组 5 在 t=${m.lanes.fully.groups[4].doneAt} 完成`,
    `训练 3 里的组 5 因为在队列里多等一轮而年龄 2`,
    `完成队列 ${m.fullyLead.queued} + 池内 ${m.fullyLead.pool} + buffer ${m.fullyLead.buffered}`,
    '13_slime_sglang_rollout_engine_analysis#2.3 变体：同一协议的三条替换轴',
    `每轮 256 条响应切成 ${stepsPerRound} 个训练 step，${CFG.numRollout} 轮共 ${stepsPerRound * CFG.numRollout} 个 step`,
    `\`num_rollout=${CFG.numRollout}\`、\`rollout_batch_size=${CFG.rolloutBatchSize}\` 组、\`save_interval=${CFG.saveInterval}\``,
  ]) {
    assert.ok(page.includes(needle), `正文必须出现 ${needle}`);
  }
  assert.ok(page.includes('assets/slime_iteration_timeline.svg'), '正文必须引用原理图');
  assert.ok(page.includes('> **源码基线**：`THUDM/slime@4c193f1f37509cca70f0e88807a9305b70f63f4e`'), '页头基线必须与生成器一致');
  assert.doesNotMatch(page, /依赖侧/, '发布中止语义已由 13 页定论，不再保留依赖侧保留语');
  assert.doesNotMatch(page, /github\.com\/THUDM\/slime\/blob\/[0-9a-f]+\/[^)]*#L\d+/, '正文不再保留 path:line 链接');
});
