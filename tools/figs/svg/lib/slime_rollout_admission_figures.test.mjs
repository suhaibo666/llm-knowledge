// 锁住 slime rollout 接收循环原理图的可执行契约：图上每个数字都由同一份 CFG 经源码算法的复现推导，
// 并且必须与 13_slime_sglang_rollout_engine_analysis.md 正文引用的数值一致。
//
// 运行：node --test tools/figs/svg/lib/slime_rollout_admission_figures.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

import { CFG, checkRewardNonzeroStd, checkRewardNonzeroStdWithFallback, model, shouldDrop } from '../slime_rollout_admission_figures.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const generator = join(here, '..', 'slime_rollout_admission_figures.mjs');
const slimeDir = join(here, '..', '..', '..', '..', 'wiki', '02_engineering', '04_posttrain_frameworks', 'slime');
const pagePath = join(slimeDir, '13_slime_sglang_rollout_engine_analysis.md');
const trackedSvg = join(slimeDir, 'assets', 'slime_rollout_admission_timeline.svg');

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

test('动态过滤与丢弃规则复现源码', () => {
  assert.equal(checkRewardNonzeroStd([1, 1, 1, 1]).keep, false);
  assert.equal(checkRewardNonzeroStd([1, 1, 1, 1]).reason, 'zero_std_1');
  assert.equal(checkRewardNonzeroStd([1, 0, 0, 1]).keep, true);
  const zero = checkRewardNonzeroStdWithFallback([0, 0, 0, 0]);
  assert.equal(shouldDrop(zero, { remainingBatchSize: 3, targetDataSize: 2 }), true);
  assert.equal(shouldDrop(zero, { remainingBatchSize: 2, targetDataSize: 2 }), false);
  assert.equal(shouldDrop(checkRewardNonzeroStd([0, 0, 0, 0]), { remainingBatchSize: 2, targetDataSize: 2 }), true);
});

test('接收循环与 fully-async 模型复现源码算法', () => {
  const m = model();
  assert.deepEqual(m.semaphore, { engines: 2, capacity: 1024 });
  assert.deepEqual(m.main.all, ['B', 'A', 'C']);
  assert.deepEqual(m.main.data, ['A', 'C']);
  assert.deepEqual(m.main.inFlight, ['D']);
  assert.deepEqual(m.main.recycled, ['D']);
  assert.deepEqual(m.main.log.map((l) => l.remaining), [4, 3, 3, 3, 3]);
  assert.deepEqual(m.main.log.map((l) => l.accepted), [0, 0, 1, 2, 2]);
  assert.deepEqual(m.main.log.map((l) => l.pendings), [4, 3, 2, 1, 0]);
  assert.deepEqual(m.groupsMain.D.indices, [12, 13, 14, 15]);
  // 连续拒绝：remaining 4 → 3 → 2 → 1，再补一整波 → 5
  assert.deepEqual(m.cascade.log.slice(0, 5).map((l) => l.remaining), [4, 3, 2, 1, 5]);
  assert.deepEqual(m.cascade.waves.map((w) => w.wave), [['A', 'B', 'C', 'D'], ['E', 'F', 'G', 'H']]);
  assert.deepEqual(m.cascade.data, ['D', 'E']);
  assert.equal(m.cascade.inFlight.length, 3);
  // fallback：remaining 2 ≤ 2 时保留零方差的 C
  assert.equal(m.fallback.log[2].remaining, 2);
  assert.equal(m.fallback.log[3].action, 'accept (keep_when_insufficient)');
  assert.deepEqual(m.fallback.data, ['C', 'D']);
  assert.equal(m.fallback.waves.length, 1);
  // fully-async：闸门在 t2 关闭，C 仍完成入队，drain 只取 2
  const fa = m.fullyAsync;
  assert.equal(fa.cap, 2);
  assert.deepEqual(fa.ticks[1].queue, ['A']);
  assert.deepEqual(fa.ticks[1].topped, ['C']);
  assert.deepEqual(fa.ticks[2].queue, ['A', 'B']);
  assert.equal(fa.ticks[2].gateClosed, true);
  assert.deepEqual(fa.ticks[2].topped, []);
  assert.deepEqual(fa.drained, ['A', 'B']);
  assert.deepEqual(fa.queueLeft, ['C']);
});

test('生成器产出原理图，且与已跟踪的 SVG 一致', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'slime-admission-'));
  try {
    const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
    assert.equal(run.status, 0, run.stderr || run.stdout);
    const svg = await readFile(join(outputDir, 'slime_rollout_admission_timeline.svg'), 'utf8');
    assertInsideCanvas(svg);
    assert.match(svg, /512 × 2 engines = 1024/);
    assert.match(svg, /B 被拒：remaining 4 → 3/);
    assert.match(svg, /remaining 3 → 2 → 1/);
    assert.match(svg, /remaining=2 ≤ 2/);
    assert.match(svg, /→ A B/);
    assert.match(svg, /队列留下 \[C\]/);
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
  const c = m.semaphore;
  for (const needle of [
    `${CFG.sglangServerConcurrency} × ${c.engines} = ${c.capacity}`,
    `${CFG.overSamplingBatchSize * CFG.nSamplesPerPrompt} 个请求`,
    `、${m.groupsMain.D.indices[0]}–${m.groupsMain.D.indices.at(-1)}）`,
    `| t1 | B 完成，reward \`${m.groupsMain.B.rewards.join(' ')}\` 零方差 → 拒绝 | ${m.main.log[0].remaining} → ${m.main.log[1].remaining} |`,
    `| t3 | C 完成 → 入选，达到目标 | ${m.main.log[3].remaining} | ${m.main.log[3].accepted} | ${m.main.log[3].pendings} |`,
    `入选的 ${m.main.data.join('、')}`,
    `从 ${m.cascade.log[0].remaining} 降到 ${m.cascade.log[1].remaining}、${m.cascade.log[2].remaining}、${m.cascade.log[3].remaining} 后，一次再补 ${CFG.overSamplingBatchSize} 组`,
    `随后 remaining 为 ${m.cascade.log[4].remaining}`,
    `池容量 ${m.fullyAsync.cap}`,
    `队列 \`[${m.fullyAsync.ticks[2].queue.join(' ')}]\``,
    `\`[${m.fullyAsync.queueLeft.join(' ')}]\` 留待下轮`,
  ]) {
    assert.ok(page.includes(needle), `正文必须出现 ${needle}`);
  }
  assert.ok(page.includes('assets/slime_rollout_admission_timeline.svg'), '正文必须引用原理图');
});
