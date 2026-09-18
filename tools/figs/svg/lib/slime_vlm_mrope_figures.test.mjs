// 锁住 slime 多模态 MRoPE 与 CP 视觉注入原理图的可执行契约：图上每个数字都由同一份 CFG 经源码规则的复现推导，
// 与 tests/test_qwen3_5_vl_native.py 的断言一致，并且必须与 26_slime_multimodal_vlm_path_analysis.md 正文引用的数值一致。
//
// 运行：node --test tools/figs/svg/lib/slime_vlm_mrope_figures.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

import {
  CFG,
  FONT_SIZE,
  OUTPUTS,
  buildPackedMropePositions,
  getPackedCpLocalIndices,
  model,
  sglangRopeIndex,
  sliceWithCp,
  textWidth,
} from '../slime_vlm_mrope_figures.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const generator = join(here, '..', 'slime_vlm_mrope_figures.mjs');
const slimeDir = join(here, '..', '..', '..', '..', 'wiki', '02_engineering', '04_posttrain_frameworks', 'slime');
const pagePath = join(slimeDir, '26_slime_multimodal_vlm_path_analysis.md');

function viewBox(svg) {
  const match = svg.match(/viewBox="0 0 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)"/);
  assert.ok(match, 'SVG 必须声明 viewBox');
  return { w: Number(match[1]), h: Number(match[2]) };
}

const unescape = (s) => s.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');

// 估算每个 text 的包围盒：rect 与 text 都不得越出画布，text 之间不得互相压叠。
function assertGeometry(svg) {
  const { w, h } = viewBox(svg);
  for (const [, x, y, rw, rh] of svg.matchAll(
    /<rect[^>]*?x="(-?\d+(?:\.\d+)?)"[^>]*?y="(-?\d+(?:\.\d+)?)"[^>]*?width="(\d+(?:\.\d+)?)"[^>]*?height="(\d+(?:\.\d+)?)"/g,
  )) {
    assert.ok(Number(x) >= 0 && Number(y) >= 0, `rect 左上越界 ${x},${y}`);
    assert.ok(Number(x) + Number(rw) <= w && Number(y) + Number(rh) <= h, `rect 右下越界 ${x}+${rw},${y}+${rh}`);
  }
  const boxes = [];
  for (const [, x, y, cls, anchor, body] of svg.matchAll(
    /<text x="(-?\d+(?:\.\d+)?)" y="(-?\d+(?:\.\d+)?)" class="(\w+)" text-anchor="(\w+)">([^<]*)<\/text>/g,
  )) {
    const size = FONT_SIZE[cls];
    assert.ok(size, `未知文字样式 ${cls}`);
    const tw = textWidth(unescape(body), size);
    const left = anchor === 'middle' ? Number(x) - tw / 2 : anchor === 'end' ? Number(x) - tw : Number(x);
    const box = { left, right: left + tw, top: Number(y) - size * 0.8, bottom: Number(y) + size * 0.2, body };
    assert.ok(box.left >= 0 && box.right <= w && box.top >= 0 && box.bottom <= h, `text 越出画布：${body}`);
    boxes.push(box);
  }
  assert.ok(boxes.length > 0, 'SVG 必须包含文字');
  for (let i = 0; i < boxes.length; i += 1) {
    for (let j = i + 1; j < boxes.length; j += 1) {
      const a = boxes[i];
      const b = boxes[j];
      const overlap = a.left < b.right - 1 && b.left < a.right - 1 && a.top < b.bottom - 1 && b.top < a.bottom - 1;
      assert.ok(!overlap, `文字互相压叠：“${a.body}” 与 “${b.body}”`);
    }
  }
}

test('packed MRoPE 复现 build_packed_mrope_position_ids 与仓内测试的期望值', () => {
  const m = model();
  const expected = [
    [0, 1, 1, 1, 1, 3, 4],
    [0, 1, 1, 2, 2, 3, 4],
    [0, 1, 2, 1, 2, 3, 4],
  ];
  assert.deepEqual(m.posA, expected, '与 test_packed_mrope_resets_positions_for_each_sample 的 expected 相同');
  assert.deepEqual(m.posB, expected, '第二条样本在 cu_seqlens 边界归零');
  assert.equal(m.visionStartPos, 1);
  assert.equal(m.resume, 3, '视觉块后文本从 3 继续');
  assert.equal(m.resumeByTokens, 5, '按 token 数推进会得到 5');
  assert.equal(m.nonSquare.merged, 6);
  assert.equal(m.nonSquare.resume, 4);
  assert.equal(m.nonSquare.resumeByTokens, 7);
  assert.deepEqual(m.nonSquare.pos, [
    [0, 1, 1, 1, 1, 1, 1, 4, 5],
    [0, 1, 1, 1, 2, 2, 2, 4, 5],
    [0, 1, 2, 3, 1, 2, 3, 4, 5],
  ]);
  // 失败分支与源码的报错一致
  assert.throws(
    () => buildPackedMropePositions([1, 2, 3], [0, 3], [[1, 4, 4]], CFG.ids, CFG.merge),
    /Unused Qwen3\.5-VL image grids/,
    '与 test_packed_mrope_rejects_unused_grids 一致',
  );
  assert.throws(() => buildPackedMropePositions([99, 10, 10, 7], [0, 4], [[1, 4, 4]], CFG.ids, CFG.merge), /does not match/);
});

test('SGLang get_rope_index 的上一段 max()+1 与 slime 规则在 grid_t = 1 时逐位相同，decode 续算落在同一位置', () => {
  const m = model();
  assert.deepEqual(m.sglang.a.pos, m.posA);
  assert.deepEqual(m.sglang.nonSquare.pos, m.nonSquare.pos);
  assert.equal(m.sglang.a.delta, -2);
  assert.deepEqual(m.decode, { index: 7, trainSide: 5, sglangSide: 5, delta: -2, tailStartIndex: 5 });
  // 多图：两段视觉块之间的文本同样对齐
  const tokens = [99, 10, 10, 10, 10, 5, 99, 10, 10, 10, 10, 10, 10, 6];
  const grids = [[1, 4, 4], [1, 4, 6]];
  const slime = buildPackedMropePositions(tokens, [0, tokens.length], grids, CFG.ids, CFG.merge).pos;
  assert.deepEqual(sglangRopeIndex(tokens, grids, CFG.ids, CFG.merge).pos, slime);
});

test('CP=2 注入复现 get_batch、get_packed_cp_local_indices 与 _inject_vision_embeddings', () => {
  const c = model().cp;
  assert.deepEqual(sliceWithCp(CFG.seqA, 2, 0, 0), [99, 10, 7, 0]);
  assert.deepEqual(sliceWithCp(CFG.seqA, 2, 1, 0), [10, 10, 10, 98]);
  assert.deepEqual(c.layout.cu, [0, 8, 16]);
  assert.equal(c.layout.ranks[0].pad, 0);
  assert.deepEqual(c.full, [99, 10, 10, 10, 10, 98, 7, 0, 99, 10, 10, 10, 10, 98, 8, 0]);
  assert.deepEqual(c.ranks[0].featureIndices, [-1, 0, 1, 2, 3, -1, -1, -1, -1, 4, 5, 6, 7, -1, -1, -1]);
  // 与 test_thd_cp_indices_select_two_chunks_per_packed_sequence 的断言相同
  assert.deepEqual(c.ranks[0].localIndices, [0, 1, 6, 7, 8, 9, 14, 15]);
  assert.deepEqual(c.ranks[1].localIndices, [2, 3, 4, 5, 10, 11, 12, 13]);
  assert.deepEqual(c.ranks[0].tokens, [99, 10, 7, 0, 99, 10, 8, 0]);
  assert.deepEqual(c.ranks[0].rowsUsed, [0, 4]);
  assert.deepEqual(c.ranks[1].rowsUsed, [1, 2, 3, 5, 6, 7]);
  assert.deepEqual(c.ranks[0].localT, [0, 1, 4, 5, 0, 1, 4, 5]);
  assert.equal(c.featuresPerImage, 4);
  assert.equal(c.computedRows, 16);
  assert.equal(c.injectedRows, 8);
  assert.equal(c.cp1Error, 'Packed sequence length 7 must be divisible by 2 * CP size 1');
  assert.deepEqual(c.cp1Cu, [0, 7, 14, 16], 'CP=1 不补齐，TP1/multiplier 4 的尾部 pad 段为 2');
  assert.equal(c.cp1Pad, 2);
  assert.deepEqual(c.geo3kCp1Cu, [0, 7, 14, 256], 'geo3k TP2、multiplier 128');
  assert.deepEqual(getPackedCpLocalIndices([0, 8, 16], 1, 0), Array.from({ length: 16 }, (_, i) => i), 'CP=1 偶数长度可通过');
});

test('geo3k 单轮脚本的训练侧视觉塔前向次数', () => {
  const g = model().geo3k;
  assert.deepEqual(g, { samples: 512, vitPasses: 1024, distinctImages: 64, redundancy: 16 });
});

test('生成器产出两张原理图，几何无越界与压叠，且与已跟踪的 SVG 一致', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'slime-vlm-mrope-'));
  try {
    const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
    assert.equal(run.status, 0, run.stderr || run.stdout);
    const needles = {
      [OUTPUTS.positions]: [
        'cu_seqlens = [0, 7, 14]',
        CFG.sourceRule,
        '1 + max(4, 4) // 2 = 3',
        '1 + 4 = 5',
        '文本续接 1 + 3 = 4',
        '方图 5，非方图 7',
        "next_pos += max(t, h', w')，或上一段 max() + 1：方图 3，非方图 4",
        'delta = max + 1 − len = -2',
        '训练侧 3 + (7 − 5) = 5，SGLang 7 + (-2) = 5',
      ],
      [OUTPUTS.cpInjection]: [
        '[0, 8, 16]',
        '注入 feature 行 {0, 4}',
        '注入 feature 行 {1, 2, 3, 5, 6, 7}',
        '本 rank 的 T 位置 0 1 4 5 0 1 4 5',
        '共算 16 行、注入 8 行',
        '视觉塔输出 8 行',
        'cu_seqlens = [0, 7, 14, 16]',
        '[0, 7, 14, 256]',
        'Packed sequence length 7 must be divisible by 2 * CP size 1',
      ],
    };
    for (const name of Object.values(OUTPUTS)) {
      const svg = await readFile(join(outputDir, name), 'utf8');
      assertGeometry(svg);
      for (const needle of needles[name]) assert.ok(unescape(svg).includes(needle), `${name} 必须出现 ${needle}`);
      assert.doesNotMatch(svg, /\[\[/, 'SVG 不得泄漏 wikilink 标记');
      const tracked = await readFile(join(slimeDir, 'assets', name), 'utf8');
      assert.equal(tracked, svg, `已跟踪的 ${name} 必须由当前生成器重新生成`);
    }
  } finally {
    await rm(outputDir, { recursive: true, force: true });
  }
});

test('正文引用的规则与数值和模型一致', async () => {
  const page = await readFile(pagePath, 'utf8');
  const m = model();
  const c = m.cp;
  for (const needle of [
    CFG.sourceRule,
    `视觉块后的文本从 ${m.resume} 而不是 ${m.resumeByTokens} 继续`,
    `$1+4=${m.resumeByTokens}$`,
    `$1+3=${m.nonSquare.resume}$`,
    `则会得到 ${m.nonSquare.resumeByTokens}`,
    `max(4, 4) // 2 = 2`,
    `本例 delta 为 ${String(m.sglang.a.delta).replace('-', '−')}`,
    `$7-2=${m.decode.sglangSide}$`,
    `$3+(7-5)=${m.decode.trainSide}$`,
    `\`[${c.layout.cu.join(', ')}]\``,
    `\`[${c.full.join(', ')}]\``,
    `\`[${c.ranks[0].localIndices.join(', ')}]\``,
    `\`[${c.ranks[1].localIndices.join(', ')}]\``,
    `\`[${c.ranks[0].tokens.join(', ')}]\``,
    `只注入 feature 行 ${c.ranks[0].rowsUsed.join(' 与 ')}`,
    `注入行 ${c.ranks[1].rowsUsed.join('、')}`,
    `共算 ${c.computedRows} 行、注入 ${c.injectedRows} 行`,
    `T 位置是 ${c.ranks[0].localT.join(' ')}`,
    c.cp1Error,
    `$512\\cdot1\\cdot2\\cdot1=${m.geo3k.vitPasses}$`,
    `重复 ${m.geo3k.redundancy} 倍`,
    'budget 不是“累计 4096”',
    `\`cu_seqlens = [${c.cp1Cu.join(', ')}]\``,
    `\`[${c.geo3kCp1Cu.join(', ')}]\``,
    '_compute_image_only_mrope_positions_from_offsets',
    '`next_pos += max(llm_grid_t, llm_grid_h, llm_grid_w)`',
    'compute_spec_mrope_positions',
  ]) {
    assert.ok(page.includes(needle), `正文必须出现 ${needle}`);
  }
  for (const name of Object.values(OUTPUTS)) assert.ok(page.includes(`assets/${name}`), `正文必须引用 ${name}`);
  assert.ok(page.includes('`THUDM/slime@4c193f1f37509cca70f0e88807a9305b70f63f4e`'), '页头基线必须与生成器一致');
  assert.doesNotMatch(page, /奇数样本对应的 pad 段也是奇数/, 'CP=1 尾部 pad 段的奇偶取决于总长，不随样本奇偶');
  assert.doesNotMatch(page, /github\.com\/THUDM\/slime\/blob\/[0-9a-f]+\/[^)]*#L\d+/, '正文不保留 path:line 链接');
});
