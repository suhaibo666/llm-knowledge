// 锁住 slime 模型架构扩展原理图的可执行契约：两张图上的每个布局与数字都由同一份 CFG 经源码规则的复现推导，
// 并且必须与 23_slime_model_architecture_extension_analysis.md 正文引用的数值一致。
//
// 运行：node --test tools/figs/svg/lib/slime_model_layout_figures.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

import { CFG, FONT, OUTPUTS, chunk, gatherTp, model, textWidth, tpShard } from '../slime_model_layout_figures.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const generator = join(here, '..', 'slime_model_layout_figures.mjs');
const slimeDir = join(here, '..', '..', '..', '..', 'wiki', '02_engineering', '04_posttrain_frameworks', 'slime');
const pagePath = join(slimeDir, '23_slime_model_architecture_extension_analysis.md');
const bracket = (xs) => `[${xs.join(' ')}]`;

function viewBox(svg) {
  const match = svg.match(/viewBox="0 0 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)"/);
  assert.ok(match, 'SVG 必须声明 viewBox');
  return { w: Number(match[1]), h: Number(match[2]) };
}

const unescape = (s) => s.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');

function assertInsideCanvas(svg) {
  const { w, h } = viewBox(svg);
  for (const [, x, y, rw, rh] of svg.matchAll(
    /<rect[^>]*?x="(-?\d+(?:\.\d+)?)"[^>]*?y="(-?\d+(?:\.\d+)?)"[^>]*?width="(\d+(?:\.\d+)?)"[^>]*?height="(\d+(?:\.\d+)?)"/g,
  )) {
    assert.ok(Number(x) >= 0 && Number(y) >= 0, `rect 左上越界 ${x},${y}`);
    assert.ok(Number(x) + Number(rw) <= w && Number(y) + Number(rh) <= h, `rect 右下越界 ${x}+${rw},${y}+${rh}`);
  }
  // 文字锚点在画布内，且按字宽估算，左对齐文字的右端不越过画布
  for (const [, x, y, cls, anchor, body] of svg.matchAll(
    /<text x="(-?\d+(?:\.\d+)?)" y="(-?\d+(?:\.\d+)?)" class="(\w+)" text-anchor="(\w+)">([^<]*)<\/text>/g,
  )) {
    assert.ok(Number(x) >= 0 && Number(x) <= w && Number(y) >= 0 && Number(y) <= h, `text 越界 ${x},${y}`);
    if (anchor === 'start') {
      const right = Number(x) + textWidth(unescape(body), FONT[cls]);
      assert.ok(right <= w, `text 右端估计越界 ${Math.round(right)} > ${w}：${body.slice(0, 30)}`);
    }
  }
}

test('GLU 与非 GLU fc1 复现 _tensor_parallel_shard / all_gather_params_async 的名字规则', () => {
  const m = model();
  assert.deepEqual(m.glu.shards, [['g0', 'g1', 'u0', 'u1'], ['g2', 'g3', 'u2', 'u3']]);
  assert.deepEqual(m.glu.naiveGate, ['g0', 'g1', 'u0', 'u1'], '朴素按 rank 拼接时 gate_proj 混进 up 行');
  assert.deepEqual(m.glu.gathered, ['g0', 'g1', 'g2', 'g3', 'u0', 'u1', 'u2', 'u3']);
  assert.equal(m.glu.roundTrip, true);
  assert.equal(m.glu.naiveBroken, true);
  // 与 tests/test_qwen3_5_vl_native.py::test_raw_loader_shards_swiglu_and_grouped_moe_fc2 一致：8 行 fc1，rank1 取 2–3、6–7 行
  const rows = Array.from({ length: 8 }, (_, i) => i);
  assert.deepEqual(tpShard('decoder.layers.0.mlp.linear_fc1.weight', rows, 2, 1, 1), [2, 3, 6, 7]);
  assert.deepEqual(tpShard('decoder.layers.0.mlp.linear_fc2.weight', [0, 1, 2, 3], 2, 1), [2, 3]);
  // 通用分支的 stride：chunk(size·stride) 后取 [rank::size]
  assert.deepEqual(tpShard('x.weight', rows, 2, 0, 2), [0, 1, 4, 5]);
  assert.deepEqual(chunk([1, 2, 3, 4, 5], 2), [[1, 2, 3], [4, 5]], 'torch.chunk 语义');

  assert.deepEqual(m.nonGlu.fc1Shards, [['a0', 'a2'], ['a1', 'a3']]);
  assert.deepEqual(m.nonGlu.fc2Shards, [['c0', 'c1'], ['c2', 'c3']]);
  assert.equal(m.nonGlu.mismatches, 2);
  assert.equal(m.nonGlu.roundTrip, true, '往返恒等，映射单测看不出训练错配');
  assert.deepEqual(gatherTp('decoder.layers.0.mlp.linear_qkv.weight', [['x'], ['y']]), ['x', 'y']);
});

test('Qwen3-Next 门控 QKV 复现 reshape(groups, qpg, 2, hd).transpose(1, 2) 与逆变换', () => {
  const q = model().qkv;
  assert.equal(q.hfQRows, 16);
  assert.equal(q.rows, 24, 'test_hf_and_megatron_mappings_round_trip 的 qwen3_next 形状 (24, 8)');
  assert.deepEqual(q.groupsLayout, [
    ['q0', 'q1', 'z0', 'z1', 'k0', 'v0'],
    ['q2', 'q3', 'z2', 'z3', 'k1', 'v1'],
  ]);
  assert.deepEqual(q.shards, q.groupsLayout, 'TP=2 按行均分时每个 rank 恰好一整组');
  assert.deepEqual(q.shardRows, [12, 12]);
  assert.deepEqual(q.megatronSplit, { query: 4, gate: 4, key: 2, value: 2 });
  assert.deepEqual(q.exportQ, ['q0', 'z0', 'q1', 'z1', 'q2', 'z2', 'q3', 'z3']);
  assert.equal(q.roundTrip, true);
  assert.equal(q.genericExpectedQRows, 8);
});

test('PP/EP 命名、CP 两段重组与反向梯度账', () => {
  const m = model();
  assert.equal(m.naming.union.length, CFG.layers * CFG.experts);
  assert.equal(m.naming.lastGlobal, 'decoder.layers.3.mlp.experts.linear_fc1.weight3');
  assert.equal(m.naming.hfExample, 'model.layers.3.mlp.experts.3');
  assert.match(m.naming.mtp.global, /^mtp\.layers\.0\..*weight3$/, 'MTP 层号不加 PP offset，expert 加 EP offset');
  assert.deepEqual(
    m.naming.ranks.map((r) => [r.layerOffset, r.expertOffset]),
    [
      [0, 0],
      [0, 2],
      [2, 0],
      [2, 2],
    ],
  );

  const cp = m.cp;
  assert.deepEqual(cp.local, [
    ['a0', 'a1', 'a6', 'a7', 'b0', 'b3'],
    ['a2', 'a3', 'a4', 'a5', 'b1', 'b2'],
  ]);
  assert.equal(cp.pad, 122);
  assert.deepEqual(cp.cuSeqlens, [0, 8, 12, 256]);
  assert.deepEqual(cp.segments.map((s) => s.chunkSize), [2, 1]);
  assert.equal(cp.restoresOrder, true, '前段按 rank 顺序、后段按 rank 逆序恰好还原原序');
  assert.equal(cp.outputsMatchInput, true, 'chunk(2·CP) 取回的布局与输入相同');

  const g = m.grads;
  assert.deepEqual(g.truth, [8, 7, 6, 5, 4, 3, 2, 1]);
  assert.deepEqual(g.spLocal, g.truth, 'SP：scatter 反向 all-gather，取本片正确');
  assert.deepEqual(g.cpLocal, [4, 3, 4, 3, 2, 1, 2, 1]);
  assert.deepEqual(g.cpReduce, g.truth, 'CP：reduce-scatter 求和才等于真实梯度');
  assert.deepEqual(g.totals, { truth: 36, spLocal: 36, spReduce: 72, cpLocal: 20, cpReduce: 36 });
});

test('生成器产出两张原理图，且与已跟踪的 SVG 一致', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'slime-model-layout-'));
  try {
    const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
    assert.equal(run.status, 0, run.stderr || run.stdout);
    const needles = {
      tpFusion: ['[g0 g1 u0 u1]，混入 up 行', '[g0 g1 g2 g3]，逐行还原', '2/4 个神经元错配', '= HF [a0 a1 a2 a3]', 'linear_qkv（24×8）', 'rank0：12 行', 'query 4 行', 'q_proj = [q0 z0 q1 z1 q2 z2 q3 z3]', '4 × 2 = 8 行，实得 16 行', '39 个 scripts/models preset'],
      epCp: ['全局 layers.2–3 · weight2–3', '16 个 expert fc1 名', 'layers.3.mlp.experts.linear_fc1.weight3', 'weight1 → weight3', 'pad 本地 122', 'pad 全局 244', 'cu_seqlens = [0, 8, 12, 256]', '合计', '>72<', '>20<'],
    };
    for (const [key, file] of Object.entries(OUTPUTS)) {
      const svg = await readFile(join(outputDir, file), 'utf8');
      assertInsideCanvas(svg);
      for (const needle of needles[key]) assert.ok(svg.includes(needle), `${file} 必须出现 ${needle}`);
      assert.doesNotMatch(svg, /\[\[/, 'SVG 不得泄漏 wikilink 标记');
      const tracked = await readFile(join(slimeDir, 'assets', file), 'utf8');
      assert.equal(tracked, svg, `已跟踪的 ${file} 必须由当前生成器重新生成`);
    }
  } finally {
    await rm(outputDir, { recursive: true, force: true });
  }
});

test('正文引用的布局与数值与模型一致', async () => {
  const page = await readFile(pagePath, 'utf8');
  const m = model();
  const { glu, nonGlu, qkv, naming, cp, grads } = m;
  for (const needle of [
    `rank0 拿 \`${bracket(glu.shards[0])}\`，rank1 拿 \`${bracket(glu.shards[1])}\``,
    `得到 \`${bracket(glu.naive)}\``,
    `\`gate_proj\` 就成了 \`${bracket(glu.naiveGate)}\``,
    `得到 \`${bracket(glu.gathered)}\``,
    `rank0 拿 \`${bracket(nonGlu.fc1Shards[0])}\`、rank1 拿 \`${bracket(nonGlu.fc1Shards[1])}\``,
    `rank0 拿 \`${bracket(nonGlu.fc2Shards[0])}\`、rank1 拿 \`${bracket(nonGlu.fc2Shards[1])}\``,
    `${CFG.ffn} 个神经元里有 ${nonGlu.mismatches} 个错配`,
    `拼回 \`${bracket(nonGlu.exported)}\``,
    `${CFG.presetCount} 个 preset 全部直接或经 \`source\` 带 \`--swiglu\``,
    `是 ${qkv.hfQRows} 行`,
    `得到 ${qkv.rows}×${CFG.hidden} 的 \`linear_qkv\``,
    `group 0 为 \`${bracket(qkv.groupsLayout[0])}\`，group 1 为 \`${bracket(qkv.groupsLayout[1])}\``,
    `query ${qkv.megatronSplit.query} 行、gate ${qkv.megatronSplit.gate} 行、key ${qkv.megatronSplit.key} 行、value ${qkv.megatronSplit.value} 行`,
    `每个 rank 恰好 ${qkv.shardRows[0]} 行`,
    `回到 \`${bracket(qkv.exportQ)}\``,
    `期望 q 为 ${CFG.heads} × ${CFG.headDim} = ${qkv.genericExpectedQRows} 行，实得 ${qkv.hfQRows} 行`,
    `变成 \`${naming.lastGlobal}\``,
    `\`${naming.hfExample}.gate_proj\``,
    `同一份 ${naming.union.length} 个 expert fc1 名`,
    `rank0 本地为 \`${bracket(cp.local[0])}\`，rank1 为 \`${bracket(cp.local[1])}\``,
    `本地补 ${cp.pad} 个 pad`,
    `\`[${cp.cuSeqlens.join(', ')}]\``,
    `（a 为 ${cp.segments[0].chunkSize}、b 为 ${cp.segments[1].chunkSize}）`,
    `真实梯度是 \`${bracket(grads.truth)}\`，合计 ${grads.totals.truth}`,
    `会翻倍成 ${grads.totals.spReduce}`,
    `取本片得到 \`${bracket(grads.cpLocal)}\`，合计只有 ${grads.totals.cpLocal}`,
    `才回到 ${grads.totals.cpReduce}`,
    `梯度合计 ${grads.totals.cpLocal} 对 ${grads.totals.truth}`,
  ]) {
    assert.ok(page.includes(needle), `正文必须出现 ${needle}`);
  }
  for (const file of Object.values(OUTPUTS)) assert.ok(page.includes(`assets/${file}`), `正文必须引用 ${file}`);
  assert.doesNotMatch(page, /github\.com\/THUDM\/slime\/blob\/[0-9a-f]+\/[^)]*#L\d+/, '正文不再保留 path:line 链接');
  assert.doesNotMatch(page, /681b3adc/, '正文不叙述旧基线');
});
