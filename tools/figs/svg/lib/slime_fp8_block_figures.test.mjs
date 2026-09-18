// 锁住 slime 低精度原理图的可执行契约：图上的块、scale、q、int32 字与字节数都由生成器复现源码算法算出，
// 并且必须与 22_slime_low_precision_training_rollout_analysis.md 正文引用的数值一致。
// 数值曾用 torch 2.14 对复制出的 block_fp8 / tensor_fp8 / per_block_cast_to_fp8 / _FakeInt4QuantizationSTE 函数体交叉核对。
//
// 运行：node --test tools/figs/svg/lib/slime_fp8_block_figures.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

import { CFG, OUTPUTS, castE4m3, ceilUe8m0, model, pow2, roundHalfEven, sci, toBf16 } from '../slime_fp8_block_figures.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const generator = join(here, '..', 'slime_fp8_block_figures.mjs');
const slimeDir = join(here, '..', '..', '..', '..', 'wiki', '02_engineering', '04_posttrain_frameworks', 'slime');
const pagePath = join(slimeDir, '22_slime_low_precision_training_rollout_analysis.md');

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

test('数值格式原语与 torch 行为一致', () => {
  assert.equal(toBf16(Math.fround(1e-12)), 1.0018652574217413e-12);
  assert.equal(toBf16(Math.fround(1 / 7)), 0.142578125);
  assert.equal(toBf16(Math.fround(1e-5)), 1.0013580322265625e-5);
  // E4M3：最大 448，最小正次正规 2⁻⁹，中点取偶数码（torch 实测 3.5 个单位 → 4，2.5 → 2）
  assert.equal(castE4m3(448), 448);
  assert.equal(castE4m3(1.8), 1.75);
  assert.equal(castE4m3(3.5 * 2 ** -9), 4 * 2 ** -9);
  assert.equal(castE4m3(2.5 * 2 ** -9), 2 * 2 ** -9);
  assert.equal(castE4m3(2 ** -10), 0);
  assert.ok(Number.isNaN(castE4m3(NaN)));
  assert.equal(ceilUe8m0(Math.fround(1 / 448)), 2 ** -8);
  assert.equal(ceilUe8m0(Math.fround(1e-4 / 448)), 2 ** -22);
  assert.equal(ceilUe8m0(2 ** -8), 2 ** -8);
  assert.deepEqual([roundHalfEven(0.5), roundHalfEven(1.5), roundHalfEven(2.5), roundHalfEven(6.9999998)], [0, 2, 2, 7]);
});

test('FP8 分块复现 tests/test_block_fp8_zero_block.py 的断言，并分开 blockwise / per-tensor / UE8M0', () => {
  const m = model();
  const T = m.fp8.test;
  const E = m.fp8.ext;
  assert.deepEqual(CFG.shape, [256, 256]);
  assert.deepEqual(CFG.block, [128, 128]);
  assert.deepEqual(m.fp8.scaleGrid, [2, 2]);
  assert.equal(T.scale00, 0.0022321429569274187);
  assert.equal(T.q00, 448);
  assert.equal(T.zeroScale, 2.2363063176711762e-15);
  assert.equal(T.zeroQ, 0);
  assert.equal(T.allScalesPositive, true, 'assert (scale > 0).all()');
  assert.equal(T.zeroTiles, 3);
  assert.equal(T.unclampedZeroScale, 0);
  assert.equal(T.nanCount, 49152, '去掉 clamp 后三块零块全部 0/0');
  assert.equal(T.tritonZeroScale, 2.2321428288629014e-13);
  assert.equal(T.tensorAgrees, true, '单个非零元素时 blockwise 与 per-tensor 的 Q 相同');
  assert.equal(E.value, 2 ** -20);
  assert.equal(E.blockScale, 2.1287374085687816e-9);
  assert.deepEqual([E.blockQ, E.tritonQ], [448, 448]);
  assert.equal(E.blockDeq, E.value);
  assert.ok(E.tensorRatio <= E.zeroThreshold);
  assert.equal(E.tensorQ, 0);
  assert.deepEqual([E.ueSf00, E.ueQ00, E.ueSfZero, E.ueSfX, E.ueQX, E.ueDeqX], [2 ** -8, 256, 2 ** -22, 2 ** -22, 4, 2 ** -20]);
  assert.deepEqual([m.fp8.bf16Bytes, m.fp8.weightBytes, m.fp8.blockScaleBytes, m.fp8.tensorScaleBytes], [131072, 65536, 16, 4]);
});

test('INT4 fake-QAT 与 rollout 打包的分组、q、scale 存储与 int32 字', () => {
  const I = model().int4;
  assert.deepEqual(I.scaleShape, [256, 2]);
  assert.deepEqual([I.row0.q, I.row0.out], [7, 1]);
  assert.equal(I.zero.q, 0);
  assert.ok(I.x.raw < CFG.int4.scaleFloor, '2⁻²⁰/7 低于 scale 下限');
  assert.equal(I.x.scale, Math.fround(1e-5));
  assert.equal(Number(I.x.ratio.toFixed(4)), 0.0954);
  assert.deepEqual([I.x.q, I.x.out], [0, 0]);
  assert.deepEqual([I.packQ00, I.packQX], [I.row0.q, I.x.q], '打包与 STE 对同一输入给出同一个 q');
  assert.equal(I.storedScale00, 0.142578125);
  assert.deepEqual([I.nibble00, I.nibbleZero], [15, 8]);
  assert.equal(I.word0, '0x8888888F');
  assert.equal(I.wordsPerRow, 32);
  assert.deepEqual(I.bytes, { packed: 32768, scale: 1024, shape: 8 });
  assert.equal(I.total, 33800);
});

test('生成器产出两张原理图，且与已跟踪的 SVG 一致', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'slime-fp8-block-'));
  try {
    const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
    assert.equal(run.status, 0, run.stderr || run.stdout);
    const fp8 = await readFile(join(outputDir, OUTPUTS.fp8), 'utf8');
    const int4 = await readFile(join(outputDir, OUTPUTS.int4), 'utf8');
    for (const svg of [fp8, int4]) {
      assertInsideCanvas(svg);
      assert.doesNotMatch(svg, /\[\[\d+_|\[\[[A-Za-z一-鿿]/, 'SVG 不得泄漏 wikilink 标记');
    }
    assert.match(fp8, /s = 1\/448/);
    assert.match(fp8, /= 2\.2321e-3/);
    assert.match(fp8, /BF16\(1e-12\) = 1\.0019e-12/);
    assert.match(fp8, /s = 2\.2363e-15/);
    assert.match(fp8, /49152 个 NaN/);
    assert.match(fp8, /1e-10\/448 = 2\.2321e-13/);
    assert.match(fp8, /= 2\.129e-9/);
    assert.match(fp8, /W\/s = 4\.2725e-4 ≤ 2⁻¹⁰/);
    assert.match(fp8, /Q\[0,0\] = 256（不是 448）/);
    assert.match(fp8, /T11：Q = 4 → Q·sf = 9\.5367e-7/);
    assert.match(fp8, /sf = 2⁻²²/);
    assert.match(int4, /s = 1\/7，q = round\(7\) = 7/);
    assert.match(int4, /W\/s = 0\.0954/);
    assert.match(int4, /BF16\(1\/7\) = 0\.142578125/);
    assert.match(int4, /第 1 个字位模式 0x8888888F/);
    assert.match(int4, /共 33800 字节/);
    for (const [name, svg] of [[OUTPUTS.fp8, fp8], [OUTPUTS.int4, int4]]) {
      const tracked = await readFile(join(slimeDir, 'assets', name), 'utf8');
      assert.equal(tracked, svg, `已跟踪的 ${name} 必须由当前生成器重新生成`);
    }
  } finally {
    await rm(outputDir, { recursive: true, force: true });
  }
});

test('正文引用的数值与模型一致', async () => {
  const page = await readFile(pagePath, 'utf8');
  const m = model();
  const T = m.fp8.test;
  const E = m.fp8.ext;
  const I = m.int4;
  for (const needle of [
    `\`s = 1/${CFG.fp8Max} ≈ ${sci(T.scale00)}\``,
    `\`Q[0,0] = ${T.q00}\``,
    `\`${sci(m.fp8.toolFloor)}\``,
    `\`s = ${sci(T.zeroScale)}\``,
    `\`${m.fp8.weightBytes} + ${m.fp8.blockScaleBytes}\``,
    `为 ${m.fp8.bf16Bytes} 字节`,
    `\`${T.nanCount}\` 个 NaN`,
    `\`1e-10/448 = ${sci(T.tritonZeroScale)}\``,
    `\`W[128,128] = 2^${Math.log2(E.value)} ≈ ${sci(E.value)}\``,
    `\`s = 2^${Math.log2(E.value)}/448 ≈ ${sci(E.blockScale, 3)}\``,
    `\`Q = ${E.blockQ}\``,
    `\`W/s = ${sci(E.tensorRatio)}\``,
    `\`Q = ${E.tensorQ}\``,
    `\`sf = 2^${Math.log2(E.ueSf00)}\``,
    `\`Q[0,0] = ${E.ueQ00}\``,
    `\`sf = 2^${Math.log2(E.ueSfX)}\``,
    `\`Q = ${E.ueQX}\``,
    `\`${m.fp8.weightBytes} + ${m.fp8.tensorScaleBytes}\``,
    `\`${I.bytes.packed} + ${I.bytes.scale} + ${I.bytes.shape} = ${I.total}\``,
    `\`s = 1/7\`，\`q = ${I.row0.q}\``,
    `\`W/s = ${I.x.ratio.toFixed(4)}\`、\`q = ${I.x.q}\``,
    `\`${I.storedScale00}\``,
    `\`[${I.nibble00}, ${Array(7).fill(I.nibbleZero).join(', ')}]\``,
    `\`${I.word0}\``,
    `每行 ${I.wordsPerRow} 个字`,
    `scale 形状 ${I.scaleShape.join('×')}`,
    `${m.fp8.weightBytes + m.fp8.blockScaleBytes} 与 ${I.total} 对 ${m.fp8.bf16Bytes} 字节`,
  ]) {
    assert.ok(page.includes(needle), `正文必须出现 ${needle}`);
  }
  assert.equal(pow2(E.ueSfX), '2⁻²²');
  for (const name of Object.values(OUTPUTS)) assert.ok(page.includes(`assets/${name}`), `正文必须引用 ${name}`);
});
