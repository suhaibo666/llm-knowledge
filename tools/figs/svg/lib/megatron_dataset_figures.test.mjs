// 锁住数据入口两张图的可执行契约：图上的每个数字都必须由生成器从同一组配置算出来，
// 不能把正文引用的 (2,300)→(5,24)、440/260/300/25、g=64、384/2560=15.0% 手写成
// 彼此可能漂移的常量。基线推进导致 build_sample_idx / _calculate_padding_divisor /
// get_groups_and_subsamples / pad_sequence_for_thd 的行为变化时，这个测试先红。
//
// 运行：node --test tools/figs/svg/lib/megatron_dataset_figures.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const here = dirname(fileURLToPath(import.meta.url));
const generator = join(here, '..', 'megatron_dataset_figures.mjs');
const trackedDir = join(
  here, '..', '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks',
  'megatron-lm', 'assets',
);
const pagePath = join(here, '..', '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks',
  'megatron-lm', '11_megatron_dataset_analysis.md');

// —— 以下两块由 megatron_cp_figures.test.mjs 移植：画布越界与图/正文漂移的机械判定 ——

function viewBox(svg) {
  const match = svg.match(/viewBox="0 0 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)"/);
  assert.ok(match, 'SVG 必须声明 viewBox');
  return { w: Number(match[1]), h: Number(match[2]) };
}

// 没有任何图元允许溢出画布 —— 溢出等于裁字，肉眼过一遍不可靠，这里机械判定
function assertInsideCanvas(svg, name) {
  const { w, h } = viewBox(svg);
  const rects = svg.matchAll(
    /<rect[^>]*?x="(-?\d+(?:\.\d+)?)"[^>]*?y="(-?\d+(?:\.\d+)?)"[^>]*?width="(\d+(?:\.\d+)?)"[^>]*?height="(\d+(?:\.\d+)?)"/g,
  );
  for (const [, x, y, rw, rh] of rects) {
    assert.ok(Number(x) >= -1, `${name}: rect 左边越界 x=${x}`);
    assert.ok(Number(y) >= -1, `${name}: rect 上边越界 y=${y}`);
    assert.ok(Number(x) + Number(rw) <= w + 1, `${name}: rect 右边越界 ${x}+${rw} > ${w}`);
    assert.ok(Number(y) + Number(rh) <= h + 1, `${name}: rect 下边越界 ${y}+${rh} > ${h}`);
  }
  for (const [, x, y] of svg.matchAll(/<text[^>]*?x="(-?\d+(?:\.\d+)?)" y="(-?\d+(?:\.\d+)?)"/g)) {
    assert.ok(Number(x) >= 0 && Number(x) <= w, `${name}: text x=${x} 越界`);
    assert.ok(Number(y) >= 0 && Number(y) <= h, `${name}: text y=${y} 越界`);
  }
}


test('生成器同步产出三级索引图与两条打包路径对照图', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'megatron-dataset-figures-'));
  try {
    const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
    assert.equal(run.status, 0, run.stderr || run.stdout);

    const index = await readFile(join(outputDir, 'megatron_dataset_sample_index.svg'), 'utf8');
    const packing = await readFile(join(outputDir, 'megatron_dataset_packing_paths.svg'), 'utf8');

    // --- 图 1：复算 helpers.cpp::build_sample_idx 的取样窗口 ---
    assert.match(index, /S=1024 · ε=1 · sample_index\[j\]=\(2, 300\) · sample_index\[j\+1\]=\(5, 24\)/);
    assert.match(index, /文档长度 740 \/ 260 \/ 300 \/ 880/);
    // 四段的区间与长度：440 + 260 + 300 + (24+ε) = 1025
    assert.match(index, /document_index\[2\]\[300:740\)/);
    assert.match(index, /document_index\[3\]\[0:260\)/);
    assert.match(index, /document_index\[4\]\[0:300\)/);
    assert.match(index, /document_index\[5\]\[0:25\)/);
    assert.match(index, /440 token/);
    assert.match(index, /24\+ε = 25 token/);
    assert.match(index, /Σ = 1025 = S\+ε/);
    // 重叠视图：一块缓冲区、两个错开一格的切片
    assert.match(index, /tokens = text\[:-1\]   \(1024,\)/);
    assert.match(index, /labels = text\[1:\]   \(1024,\)/);
    assert.match(index, /重叠区间 \[1, 1024\)：1023 个 token 共享同一段内存/);
    assert.match(index, /仅 tokens/);
    assert.match(index, /仅 labels（ε）/);
    assert.match(index, /下标 1 … 1023/);
    // 不变量：步长 S、窗口 S+ε、重叠恰好 ε
    assert.match(index, /以 S=1024 为步长、S\+ε=1025 为窗口，重叠恰好 ε=1 个 token/);
    assert.match(index, /重叠恰好 1 个 token/);

    // --- 图 2：两条数据面用同一组文档回放 ---
    assert.match(packing, /cp=4 · sp\(tp\)=8 → g=64 · C=512 · 尾部对齐 64/);
    assert.match(packing, /同一比例尺 0\.6 px\/token/);
    assert.match(packing, /Σ = 2180 token/);
    // 隐式：EOD 位置、逐段 position_ids、被切断的文档数、样本内零 padding
    assert.match(packing, /EOD@439/);
    assert.match(packing, /EOD@699/);
    assert.match(packing, /EOD@999/);
    assert.match(packing, /0 … 439/);
    assert.match(packing, /0 … 23/);
    assert.match(packing, /代价：2 篇文档被切断/);
    assert.match(packing, /document_index\[2\] 的前 300 个 token 在样本 j−1/);
    assert.match(packing, /document_index\[5\] 的后 856 个在样本 j\+1/);
    // 被切走的两半在 lane 上也必须标出来（橙 = 代价）
    assert.match(packing, /class="costtx"[^>]*>document_index\[2\] 的前 300 个 → 样本 j−1</);
    assert.match(packing, /class="costtx"[^>]*>document_index\[5\] 的后 856 个 → 样本 j\+1</);
    // EOD 是隔离机制，用蓝色标注，不占用代价色
    assert.match(packing, /class="dim"[^>]*>EOD@439</);
    assert.match(packing, /收益：样本内 padding = 0/);
    assert.match(packing, /1024 \/ 1024 行都是真实 token/);
    assert.match(packing, /attention_mask \[1,1024,1024\]/);
    // 显式：两级 padding 的逐项加法与两套边界数组
    assert.match(packing, /document_index\[2\]：739 → 768（\+29）/);
    assert.match(packing, /document_index\[3\]：259 → 320（\+61）/);
    assert.match(packing, /document_index\[4\]：299 → 320（\+21）/);
    assert.match(packing, /document_index\[5\]：879 → 896（\+17）/);
    assert.match(packing, /合计 \+128 行/);
    assert.match(packing, /mb 0：CP 本地 352 → 384，全局 1408 → 1536（\+128）/);
    assert.match(packing, /mb 1：CP 本地 224 → 256，全局 896 → 1024（\+128）/);
    assert.match(packing, /合计 \+256 行/);
    assert.match(packing, /cu_seqlens = \[0, 739, 998, 1297, 1425\]/);
    assert.match(packing, /cu_seqlens_padded = \[0, 768, 1088, 1408, 1536\] · max_seqlen = 768/);
    assert.match(packing, /cu_seqlens = \[0, 879, 1007\]/);
    assert.match(packing, /cu_seqlens_padded = \[0, 896, 1024\] · max_seqlen = 896/);
    assert.match(packing, /贪心装桶：上限 C×cp = 2048/);
    assert.match(packing, /768 \+ 320 \+ 320 = 1408 ≤ 2048/);
    assert.match(packing, /再加 896 → 2304 &gt; 2048，另起 microbatch/);
    assert.match(packing, /代价：死槽 384 \/ 2560 = 15\.0%/);
    assert.match(packing, /逐样本 128 行 \+ 逐 microbatch 尾部 256 行/);
    assert.match(packing, /收益：0 篇文档被切断/);
    assert.match(packing, /cu_seqlens 末项 1425 把尾部 dummy 记成"有效"/);
    // 这张图要证的那笔交易
    assert.match(packing, /同一组 2180 token 的语料，隐式路径切断 2 篇文档换来样本内 0 padding；显式路径一篇不切，换来 384\/2560 = 15\.0% 的死槽/);
    assert.match(packing, /g=64 由 cp_pad=8 乘 tp_pad=8 得出；装桶上限 2048 = C×cp；尾部 dummy 长度必须能被 2×cp=8 整除/);

    const trackedIndex = await readFile(
      join(trackedDir, 'megatron_dataset_sample_index.svg'), 'utf8',
    );
    const trackedPacking = await readFile(
      join(trackedDir, 'megatron_dataset_packing_paths.svg'), 'utf8',
    );
    assert.equal(index, trackedIndex, '已跟踪的三级索引图必须与生成器同步');
    assert.equal(packing, trackedPacking, '已跟踪的打包路径图必须与生成器同步');

    for (const svg of [index, packing]) {
      assert.match(svg, /^<svg[\s\S]*<\/svg>\s*$/);
      assert.doesNotMatch(svg, /undefined|NaN/);
      assert.doesNotMatch(svg, /\[\[/, '标注里不得漏进 wikilink 方括号');
    }
  } finally {
    await rm(outputDir, { recursive: true, force: true });
  }
});

test('两条 lane 的行数账必须自洽：真实 token 加死槽等于物理行数', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'megatron-dataset-ledger-'));
  try {
    const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
    assert.equal(run.status, 0, run.stderr || run.stdout);
    const packing = await readFile(join(outputDir, 'megatron_dataset_packing_paths.svg'), 'utf8');

    // 从图上读回死槽与物理行数，再用两级 padding 的逐项加法独立复核
    const ledger = packing.match(/死槽 (\d+) \/ (\d+) = ([\d.]+)%/);
    assert.ok(ledger, '代价卡必须给出死槽 / 物理行数 / 比例');
    const dead = Number(ledger[1]), rows = Number(ledger[2]), ratio = Number(ledger[3]);
    const perSample = Number(packing.match(/逐样本 (\d+) 行/)[1]);
    const tail = Number(packing.match(/逐 microbatch 尾部 (\d+) 行/)[1]);
    assert.equal(perSample + tail, dead, '两级 padding 之和必须等于死槽总数');
    assert.equal(rows, 1536 + 1024, '物理行数必须等于两个 microbatch 的对齐后长度之和');
    assert.equal(rows - dead, 739 + 259 + 299 + 879, '有效行数必须等于四条样本的 original_seq_len 之和');
    assert.equal(Number(((100 * dead) / rows).toFixed(1)), ratio);
  } finally {
    await rm(outputDir, { recursive: true, force: true });
  }
});

test('页面正文引用图，且图上的数值在正文中出现', async () => {
  const page = await readFile(pagePath, 'utf8');

  // 两张图都必须以外链 svg 的方式被正文引用，且不许内联 <svg>（内联会被 docs-site 打坏）
  assert.match(page, /!\[[^\]]+\]\(assets\/megatron_dataset_sample_index\.svg\)/);
  assert.match(page, /!\[[^\]]+\]\(assets\/megatron_dataset_packing_paths\.svg\)/);
  assert.doesNotMatch(page, /<svg/);

  // 图上算出来的关键量必须在正文里出现，防止图与正文各自漂移
  for (const needle of [
    '1025',
    '440',
    '260',
    '300',
    '15.0',
    '2560',
    '384',
    'cu_seqlens',
  ]) {
    assert.ok(page.includes(needle), `正文必须引用图上的 ${needle}`);
  }
});

test('两张图都不越界', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'fig-bounds-'));
  try {
    const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
    assert.equal(run.status, 0, run.stderr || run.stdout);
    for (const name of ['megatron_dataset_sample_index', 'megatron_dataset_packing_paths']) {
      const svg = await readFile(join(outputDir, `${name}.svg`), 'utf8');
      assertInsideCanvas(svg, name);
      assert.doesNotMatch(svg, /\[\[[^\]]{3,}\]\]/, `${name}: wikilink 不许漏进标注`);
    }
  } finally {
    await rm(outputDir, { recursive: true, force: true });
  }
});
