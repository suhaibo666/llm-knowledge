// 锁住 slime 训推一致性原理图的可执行契约：图上每个数字都由同一份 CFG 经源码算法的复现推导，
// 并且必须与 17_slime_train_inference_consistency_analysis.md 正文引用的数值一致。
//
// 运行：node --test tools/figs/svg/lib/slime_train_infer_consistency_figures.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

import { CFG, model, orderedBf16Sum, replayCursors, vocabParallelLogProb } from '../slime_train_infer_consistency_figures.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const generator = join(here, '..', 'slime_train_infer_consistency_figures.mjs');
const slimeDir = join(here, '..', '..', '..', '..', 'wiki', '02_engineering', '04_posttrain_frameworks', 'slime');
const pagePath = join(slimeDir, '17_slime_train_inference_consistency_analysis.md');
const trackedSvg = join(slimeDir, 'assets', 'slime_train_infer_replay.svg');

const f3 = (x) => (Object.is(x, -0) ? 0 : x).toFixed(3);

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

test('top-p 重放复现 keep mask 与词表并行 logprob', () => {
  const t = model().topP;
  assert.equal(f3(t.full), '-1.440');
  assert.equal(f3(t.nucleus), '-1.313');
  assert.equal(f3(t.phantomDelta), '0.127');
  assert.ok(Math.abs(t.tpLogProb - t.nucleus) < 1e-12, 'TP 分片后结果不变');
  assert.deepEqual(t.tpShards.map((s) => s.keptLocally), [true, false], 'rank 1 本地整行被屏蔽');
  assert.equal(t.toolLogProb, 0, '空 nucleus 写回目标后 logprob 为 0');
  // 目标 logit 写回：即使目标不在 nucleus 里也保留
  const outside = vocabParallelLogProb(CFG.logits, 2, [0, 1], 2);
  const expected = CFG.logits[2] - Math.log(Math.exp(2) + Math.exp(1) + Math.exp(0));
  assert.ok(Math.abs(outside.logProb - expected) < 1e-12);
});

test('路由重放 cursor 复现 RoutingReplay 与 stage 顺序', () => {
  const m = model();
  assert.deepEqual(m.r3.final, { recorded: 2, forward: 2, backward: 2 });
  assert.deepEqual(m.r3NoRecompute.final, { recorded: 2, forward: 2, backward: 0 });
  assert.deepEqual(m.r2.final, { recorded: 2, forward: 2, backward: 2 });
  assert.equal(m.r3.log.find((e) => e.what === 'clear_all_forward').forward, 0);
  assert.deepEqual(replayCursors({ microBatches: 3, r3: true, recompute: true }).final, { recorded: 3, forward: 3, backward: 3 });
});

test('route 梯度按列顺序逐次 BF16 舍入', () => {
  const m = model();
  assert.equal(m.accum.columnOrder, 1);
  assert.equal(m.accum.reversed, 1.0078125);
  assert.equal(m.accum.fp32Once, 1.0078125);
  assert.equal(orderedBf16Sum([1, 2 ** -7]), 1.0078125, '2⁻⁷ 是 1 附近的一个 ULP，可以表示');
});

test('校正函数复现 vanilla / ICEPOP / MIS', () => {
  const c = model().corr;
  assert.deepEqual(c.vanilla, { w: [0.5, 2], mask: [1, 1] });
  assert.deepEqual(c.icepop, { w: [0.5, 0], mask: [1, 1] });
  assert.deepEqual(c.truncate.w, [0.5, 2]);
  assert.deepEqual(c.clip.w, [0.5, 2]);
  assert.deepEqual(c.maskMode, { w: [0.5, 4], mask: [1, 0] });
  assert.ok(c.sequence.w.every((w) => Math.abs(w - 2) < 1e-12));
  assert.ok(c.geometric.w.every((w) => Math.abs(w - Math.SQRT2) < 1e-12));
  assert.equal(c.yaml.batchMean, 1.25);
  assert.deepEqual(c.yaml.w, [0.4, 1.6]);
  assert.deepEqual(c.yaml.mask, [1, 0]);
});

test('生成器产出原理图，且与已跟踪的 SVG 一致', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'slime-consistency-'));
  try {
    const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
    assert.equal(run.status, 0, run.stderr || run.stdout);
    const svg = await readFile(join(outputDir, 'slime_train_infer_replay.svg'), 'utf8');
    assertInsideCanvas(svg);
    for (const needle of ['-1.440', '-1.313', '0.127', '训练 logprob = 0.000', '记录 2 · 前 2 · 后 2', '记录 2 · 前 2 · 后 0', '1.0078125', '[0.5, 0]', '[0.4, 1.6]', '[1, 0]']) {
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
  for (const needle of [
    `−${f3(-m.topP.full)}`,
    `−${f3(-m.topP.nucleus)}`,
    `差 ${f3(m.topP.phantomDelta)}`,
    '训练 0.000',
    `前 ${m.r3.final.forward} / 后 ${m.r3.final.backward} / 记录 ${m.r3.final.recorded}`,
    `后向 cursor 停在 ${m.r3NoRecompute.final.backward}`,
    `${m.accum.reversed}`,
    `[${m.corr.vanilla.w.join(', ')}]`,
    `[${m.corr.icepop.w.join(', ')}]`,
    `[${m.corr.yaml.w.join(', ')}]`,
    `mask [${m.corr.yaml.mask.join(', ')}]`,
    `均值 ${m.corr.yaml.batchMean}`,
    '1.414',
    'S ∪ {y}',
  ]) {
    assert.ok(page.includes(needle), `正文必须出现 ${needle}`);
  }
  assert.ok(page.includes('assets/slime_train_infer_replay.svg'), '正文必须引用原理图');
  assert.ok(page.includes('#### 2.2.9 TIS/MIS 校正机制'), '10、15 页入链标签引用的 §2.2.9 必须存在');
  assert.doesNotMatch(page, /github\.com\/THUDM\/slime\/blob\/[0-9a-f]+\/[^)]*#L\d+/, '正文不再保留 path:line 链接');
});
