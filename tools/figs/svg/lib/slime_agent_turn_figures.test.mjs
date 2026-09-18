// 锁住 slime agent 两层判断原理图的可执行契约：图上每个数字都由同一份 CFG 经 slime/agent/trajectory.py 的复现推导；
// 复现必须逐字重现 slime 分支测试的 golden 字符串，且与 24_slime_agent_workflow_examples_analysis.md 正文引用的数值一致。
//
// 运行：node --test tools/figs/svg/lib/slime_agent_turn_figures.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

import { CFG, commonPrefixLen, golden, model, runSession, trainedSpans } from '../slime_agent_turn_figures.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const generator = join(here, '..', 'slime_agent_turn_figures.mjs');
const slimeDir = join(here, '..', '..', '..', '..', 'wiki', '02_engineering', '04_posttrain_frameworks', 'slime');
const pagePath = join(slimeDir, '24_slime_agent_workflow_examples_analysis.md');
const FIGS = ['slime_agent_message_layer.svg', 'slime_agent_token_layer.svg'];

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

// token 格子里的文字不能比格子宽（等宽字体按 0.62em 估算）
function assertTokensFitCells(svg) {
  const cells = [...svg.matchAll(/<rect x="(\d+(?:\.\d+)?)" y="(\d+(?:\.\d+)?)" width="(56)" height="26"/g)].map((m) => ({ x: Number(m[1]), y: Number(m[2]), w: Number(m[3]) }));
  const toks = [...svg.matchAll(/<text x="(\d+(?:\.\d+)?)" y="(\d+(?:\.\d+)?)" class="tk" text-anchor="middle">([^<]*)<\/text>/g)];
  assert.ok(cells.length > 0 && toks.length > 0);
  for (const [, x, y, raw] of toks) {
    const s = raw.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');
    const cell = cells.find((c) => Number(x) > c.x && Number(x) < c.x + c.w && Number(y) > c.y && Number(y) < c.y + 26);
    if (!cell) continue;
    assert.ok([...s].length * 10.5 * 0.62 <= cell.w, `token 文字超出格子：${s}`);
  }
}

test('复现逐字重现 slime 分支测试的 golden（2.2 / 2.3 / 2.4 / 2.5 / 3.1 / 3.2）', () => {
  // test_2_2_clean_multiturn_linearize：tool 标签为 "4"
  assert.deepEqual(runSession({ toolLabel: '4' }).samples.map(golden), [
    '<sys> system:S </sys> <usr> user:u </usr> <gen> [r:call] [</ast>] <tul> tool:4 </tul> <gen> [r:done] [</ast>]',
  ]);
  const p1Len = 7;
  // test_2_3_drift_case_A_forks：在位置 len(p1) - 1 插入漂移
  assert.deepEqual(runSession({ drift: { mode: 'insert', at: p1Len - 1 } }).samples.map(golden), [
    '<sys> system:S </sys> <usr> user:u </usr> <gen> [r:call] [</ast>]',
    '<sys> system:S </sys> <usr> user:u </usr> <DRIFT> <gen> r:call </ast> <tul> tool:t </tul> <gen> [r:done] [</ast>]',
  ]);
  // test_2_4_drift_case_B1_short_replaces：替换 A 回放的最后一个 token
  const b1 = runSession({ drift: { mode: 'replace', at: p1Len + 2 - 1 } });
  assert.deepEqual(b1.samples.map(golden), [
    '<sys> system:S </sys> <usr> user:u </usr> <gen> r:call <DRIFT> <tul> tool:t </tul> <gen> [r:done] [</ast>]',
  ]);
  assert.deepEqual(b1.samples[0].rolloutLogProbs, [...Array(b1.p2.length - b1.p1.length).fill(0), -0.4, -0.4]);
  assert.equal(commonPrefixLen([...b1.p1, ...b1.r1], b1.p2), p1Len + 1);
  // test_2_5_drift_case_B1_long_forks
  assert.deepEqual(runSession({ drift: { mode: 'replace', at: p1Len + 1 }, threshold: 1 }).samples.map(golden), [
    '<sys> system:S </sys> <usr> user:u </usr> <gen> [r:call] [</ast>]',
    '<sys> system:S </sys> <usr> user:u </usr> <gen> r:call <DRIFT> <tul> tool:t </tul> <gen> [r:done] [</ast>]',
  ]);
  // test_2_6_drift_case_B1_threshold_zero_forks
  assert.equal(runSession({ drift: { mode: 'replace', at: p1Len + 1 }, threshold: 0 }).samples.length, 2);
  // test_3_1_rewrite_merge_absorbs_short：标签 ok → "ok "
  const merged = runSession({ firstLabel: 'ok', echoLabel: 'ok ' });
  assert.deepEqual(merged.samples.map(golden), [
    '<sys> system:S </sys> <usr> user:u </usr> <gen> r:ok␣ </ast> <tul> tool:t </tul> <gen> [r:done] [</ast>]',
  ]);
  assert.deepEqual(merged.tree.find((t) => t.merged).merged, { abandoned_turn_index: 1, abandoned_response_tokens: 2 });
  // test_3_2_rewrite_merge_long_forks：阈值 1
  assert.deepEqual(runSession({ firstLabel: 'ok', echoLabel: 'ok2 ', threshold: 1 }).samples.map(golden), [
    '<sys> system:S </sys> <usr> user:u </usr> <gen> [r:ok] [</ast>]',
    '<sys> system:S </sys> <usr> user:u </usr> <gen> r:ok2␣ </ast> <tul> tool:t </tul> <gen> [r:done] [</ast>]',
  ]);
  // 阈值为 0：合并关闭，改写一律分叉（test_3_3）
  assert.equal(runSession({ firstLabel: 'ok', echoLabel: 'ok3 ', threshold: 0 }).samples.length, 2);
  // 每个片段完整 reward（test_2_8）
  for (const s of runSession({ drift: { mode: 'insert', at: p1Len - 1 } }).samples) assert.equal(s.reward, 1);
});

test('图中的决策量与结果', () => {
  const m = model();
  assert.deepEqual([m.p1Len, m.aLen, m.driftAt, m.earlyAt], [7, 2, 8, 6]);
  const d = m.token.realign.decisions[0];
  assert.deepEqual(d, { held: 9, realignAt: 8, drift: 1, start: 7, outputLen: 2, threshold: CFG.defaultForkThreshold, kind: 'REALIGN' });
  assert.equal(m.token.fork.decisions[0].kind, 'FORK');
  assert.deepEqual(m.token.early.decisions[0].kind, 'FORK');
  assert.deepEqual(m.token.clean.samples[0].lossMask, [1, 1, 0, 0, 0, 0, 1, 1]);
  assert.deepEqual(m.token.realign.samples[0].lossMask, [0, 0, 0, 0, 0, 0, 1, 1]);
  assert.equal(m.token.realign.samples[0].responseLength, 8);
  assert.deepEqual([m.token.clean.trained, m.token.realign.trained, m.token.fork.trained], [4, 2, 4]);
  const f2 = m.token.fork.samples[1];
  assert.equal(f2.tokens.length - f2.responseLength, 13);
  assert.deepEqual(m.message.equal.samples.map((s) => trainedSpans(s)), [['A', 'C']]);
  assert.deepEqual(m.message.merge.samples.map((s) => trainedSpans(s)), [['C']]);
  assert.deepEqual(m.message.noMerge.samples.map((s) => trainedSpans(s)), [['A'], ['C']]);
  assert.deepEqual(m.message.merge.tree.map((t) => t.kind), ['route', 'route', 'demoted', 'route', 'generated']);
});

test('生成器产出原理图，且与已跟踪的 SVG 一致', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'slime-agent-turn-'));
  try {
    const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
    assert.equal(run.status, 0, run.stderr || run.stdout);
    for (const name of FIGS) {
      const svg = await readFile(join(outputDir, name), 'utf8');
      assertInsideCanvas(svg);
      assert.doesNotMatch(svg, /\[\[/, 'SVG 不得泄漏 wikilink 标记');
      const tracked = await readFile(join(slimeDir, 'assets', name), 'utf8');
      assert.equal(tracked, svg, `已跟踪的 ${name} 必须由当前生成器重新生成`);
    }
    const msg = await readFile(join(outputDir, FIGS[0]), 'utf8');
    for (const needle of ['旧叶输出 2 &lt; 1024 → 改写合并', 'abandoned_turn_index=1', '导出 2 条 Sample', '训练 token 共 2', '输出 2 ≥ 1 → 不合并']) {
      assert.ok(msg.includes(needle), `消息层图必须出现 ${needle}`);
    }
    const tok = await readFile(join(outputDir, FIGS[1]), 'utf8');
    assertTokensFitCells(tok);
    for (const needle of ['公共前缀 8', 'drift = 9 − 8 = 1', '2 &lt; 1024 → REALIGN', '2 ≥ 1 → FORK', '公共前缀 6 &lt; 7 → FORK', '训练 2 个 token；A 清零', 'prompt 13 个 token 重算']) {
      assert.ok(tok.includes(needle), `token 层图必须出现 ${needle}`);
    }
  } finally {
    await rm(outputDir, { recursive: true, force: true });
  }
});

test('正文引用的数值与模型一致', async () => {
  const page = await readFile(pagePath, 'utf8');
  const m = model();
  const d = m.token.realign.decisions[0];
  const de = m.token.early.decisions[0];
  const f2 = m.token.fork.samples[1];
  const merged = m.message.merge.tree.find((t) => t.merged).merged;
  const fmt = (a) => `[${a.join(',')}]`;
  for (const needle of [
    `渲染成 ${m.p1Len} 个 prompt token`,
    `共 ${m.aLen} 个 token`,
    `在位置 ${m.driftAt}（A 回放的 \`</ast>\`）`,
    `已持有 ${d.held} 个 token，公共前缀 ${d.realignAt}，drift = ${d.held} − ${d.realignAt} = ${d.drift}`,
    `最近 response 起点 ${d.start}，${d.realignAt} ≥ ${d.start}`,
    `新 output 长度 ${d.outputLen} < ${d.threshold}，于是走 REALIGN`,
    `\`response_length=${m.token.realign.samples[0].responseLength}\`、\`loss_mask=${fmt(m.token.realign.samples[0].lossMask)}\``,
    `\`loss_mask=${fmt(m.token.clean.samples[0].lossMask)}\`，训练 token 从 ${m.token.clean.trained} 个降到 ${m.token.realign.trained} 个`,
    `全部 ${f2.tokens.length - f2.responseLength} 个 prompt token`,
    `插在位置 ${m.earlyAt}（A 起点之前），公共前缀 ${de.realignAt} < ${de.start}`,
    `\`abandoned_turn_index=${merged.abandoned_turn_index}\`、\`abandoned_response_tokens=${merged.abandoned_response_tokens}\``,
    `取默认 ${CFG.defaultForkThreshold}`,
    `阈值取 ${CFG.smallThreshold} 使 ${m.aLen} ≥ ${CFG.smallThreshold}`,
  ]) {
    assert.ok(page.includes(needle), `正文必须出现 ${needle}`);
  }
  for (const name of FIGS) assert.ok(page.includes(`assets/${name}`), `正文必须引用 ${name}`);
});
