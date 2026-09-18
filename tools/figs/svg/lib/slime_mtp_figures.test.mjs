// 锁住 slime 在线 MTP 三张原理图的可执行契约：图上每个数字都由同一份 CFG 经源码规则的复现推导，
// 并且必须与 21_slime_speculative_decoding_mtp_analysis.md 正文引用的数值一致。
//
// 运行：node --test tools/figs/svg/lib/slime_mtp_figures.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

import { CFG, OUTPUTS, alignment, fmt, fullLossMask, globalName, hfNames, model, roll, specInfo, textWidth } from '../slime_mtp_figures.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const generator = join(here, '..', 'slime_mtp_figures.mjs');
const slimeDir = join(here, '..', '..', '..', '..', 'wiki', '02_engineering', '04_posttrain_frameworks', 'slime');
const pagePath = join(slimeDir, '21_slime_speculative_decoding_mtp_analysis.md');

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

// 文字必须留在包含它的最小矩形内（左右各留 2px），用生成器同一套字宽估计；背景 rect 没有 x/y，不参与。
const FONT = { ti: 19, su: 12, pt: 14, tx: 12, sm: 10.5, mo: 10.5, cap: 11.5 };
const unescape = (t) => t.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');
function assertTextInsideRects(svg, name) {
  const rects = [...svg.matchAll(/<rect x="(-?[\d.]+)" y="(-?[\d.]+)" width="([\d.]+)" height="([\d.]+)"/g)].map((m) => m.slice(1).map(Number));
  for (const [, x, y, cls, anchor, body] of svg.matchAll(/<text x="(-?[\d.]+)" y="(-?[\d.]+)" class="(\w+)" text-anchor="(\w+)">([^<]*)<\/text>/g)) {
    const tx = Number(x);
    const ty = Number(y) - 4;
    const holders = rects.filter(([rx, ry, rw, rh]) => tx >= rx && tx <= rx + rw && ty >= ry && ty <= ry + rh);
    if (holders.length === 0) continue;
    const [rx, , rw] = holders.reduce((a, b) => (a[2] * a[3] <= b[2] * b[3] ? a : b));
    const w = textWidth(unescape(body), FONT[cls], cls === 'mo');
    const x0 = anchor === 'middle' ? tx - w / 2 : anchor === 'end' ? tx - w : tx;
    assert.ok(x0 >= rx + 2 && x0 + w <= rx + rw - 2, `${name}：文字「${unescape(body)}」超出或贴住矩形 [${rx}, ${rx + rw}]（估计 ${x0.toFixed(0)}–${(x0 + w).toFixed(0)}）`);
  }
}

test('roll 与 full_loss_masks 复现 roll_tensor 与 get_batch 的 F.pad', () => {
  assert.deepEqual(roll([0, 1, 2]), [1, 2, null]);
  assert.deepEqual(fullLossMask(3, 4), [0, 0, 1, 1, 1, 1, 0]);
  assert.deepEqual(fullLossMask(2, 3, [1, 0, 1]), [0, 1, 0, 1, 0]);
});

test('补丁的两次 roll：标签对齐到 t[i+2]，mask 早一位', () => {
  const a = model().align;
  assert.deepEqual(a.mtpLabel, [2, 3, 4, 5, 6, null, null]);
  assert.deepEqual(a.maskPatched, [1, 1, 1, 1, 0, 0, 0]);
  assert.equal(a.numTokens, 4);
  assert.deepEqual(a.counted, [2, 3, 4, 5]);
  assert.deepEqual(a.countedLoopOnly, [3, 4, 5, 6]);
  assert.deepEqual(a.response, a.countedLoopOnly, '只随循环 roll 一次时正好是 response 区间');
  assert.deepEqual(a.extraPrompt, [2]);
  assert.deepEqual(a.missedResponse, [6]);
  // prompt 只有 2 个 token 时，预 roll 把第一个 mask 位移出序列：正确计数应为 R，补丁只剩 R − 1
  const p2 = alignment({ ...CFG, sample: { P: 2, R: 4 } });
  assert.equal(p2.numTokensLoopOnly, 4);
  assert.equal(p2.numTokens, 3);
  const p3 = alignment({ ...CFG, sample: { P: 3, R: 4 } });
  assert.equal(p3.numTokens, p3.numTokensLoopOnly);
});

test('改名规则：decoder 加层偏移，MTP 层号从 0 开始，expert 下标加 EP 偏移', () => {
  const r = model().rename;
  assert.deepEqual(r.ranks.map((k) => [k.layerOffset, k.expertOffset, k.hasMtp]), [
    [0, 0, false],
    [0, 4, false],
    [2, 0, true],
    [2, 4, true],
  ]);
  const t = Object.fromEntries(r.traces.map((x) => [x.key, x]));
  assert.equal(t.decoderExpert.global, 'decoder.layers.3.mlp.experts.linear_fc1.weight7');
  assert.deepEqual(t.decoderExpert.hf, ['model.layers.3.mlp.experts.7.gate_proj.weight', 'model.layers.3.mlp.experts.7.up_proj.weight']);
  assert.equal(t.mtpExpert.global, 'mtp.layers.0.transformer_layer.mlp.experts.linear_fc1.weight7');
  assert.equal(t.mtpExpert.hf[0], 'model.layers.4.mlp.experts.7.gate_proj.weight');
  assert.equal(t.mtpProj.global, 'mtp.layers.0.eh_proj.weight');
  assert.deepEqual(t.mtpProj.hf, ['model.layers.4.eh_proj.weight']);
  assert.equal(r.wrongHf, 'model.layers.6.eh_proj.weight');
  assert.ok(Number(r.wrongHf.match(/layers\.(\d+)/)[1]) >= r.hfLayerCount, '错加偏移后的 HF 层号超出 checkpoint');
  assert.deepEqual(r.ppGroup, [1, 3]);
  assert.deepEqual(r.embeddingHolders, [1, 3]);
  assert.equal(r.embeddingSrc, 1);
  // 非 MTP、非 decoder 名字原样保留；MTP 非 expert 名字不加偏移
  assert.equal(globalName('output_layer.weight', { layerOffset: 2, expertOffset: 4 }), 'output_layer.weight');
  assert.equal(globalName('mtp.layers.0.enorm.weight', { layerOffset: 2, expertOffset: 4 }), 'mtp.layers.0.enorm.weight');
  assert.deepEqual(hfNames('mtp.layers.0.final_layernorm.weight', 4), ['model.layers.4.shared_head.norm.weight']);
});

test('SpecInfo：锁定镜像读不到分子分母，键兼容时按样本等权平均', () => {
  const s = model().accept;
  assert.equal(s.proposedPerVerify, CFG.spec.numDraftTokens - 1);
  assert.deepEqual(s.pinned.map((x) => [x.accept, x.draft, x.rate]), [
    [0, 0, 0],
    [0, 0, 0],
  ]);
  assert.equal(s.pinnedRate, 0);
  assert.equal(fmt(s.pinnedLength, 3), '2.850');
  const [A, B] = s.compatible;
  assert.deepEqual([A.accept, A.draft, A.verify, A.completion], [5, 6, 2, 8]);
  assert.deepEqual([B.accept, B.draft, B.verify, B.completion], [12, 60, 20, 34]);
  assert.equal(fmt(A.rate), '0.833');
  assert.equal(fmt(B.rate), '0.200');
  assert.equal(fmt(s.meanRate), '0.517');
  assert.equal(fmt(s.pooledRate), '0.258');
  assert.equal(fmt(s.meanLength, 3), '2.850');
  assert.equal(fmt(s.pooledLength, 3), '1.909');
  assert.ok(s.meanRate > s.pooledRate, '短样本高接受率被等权放大');
  assert.deepEqual(specInfo([]), { accept: 0, draft: 0, verify: 0, completion: 0, rate: 0, length: 0 });
});

test('生成器产出三张原理图，且与已跟踪的 SVG 一致', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'slime-mtp-'));
  try {
    const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
    assert.equal(run.status, 0, run.stderr || run.stdout);
    const needles = {
      [OUTPUTS.training]: ['mask 却因此早一位', 'P = 2 时只剩 R − 1', 't2–t5：多 t2、漏 t6', 't3–t6 = response', 'num_tokens = 4', '✕ decoder_input', 'keep_graph=False)', 'mtp_output_weight', 'check_mtp_only_grad', 't6 未计'],
      [OUTPUTS.publication]: ['decoder.layers.3.mlp.experts.linear_fc1.weight7', 'mtp.layers.0.transformer_layer.mlp.experts.linear_fc1.weight7', 'model.layers.4.eh_proj.weight', 'mtp.layers.2 → model.layers.6', 'src_rank = 1', 'target ✓，draft 无调用 ✕', 'draft_runner ✓ → target ✓', 'layer_offset = 2，expert_offset = 4'],
      [OUTPUTS.accept]: ['5/6 = 0.833', '12/60 = 0.200', '(0.833 + 0.200) / 2 = 0.517', '(5 + 12) / (6 + 60) = 0.258', '(8 + 34) / (2 + 20) = 1.909', '无此键 → 0', '日志 rate = 0.0'],
    };
    for (const name of Object.values(OUTPUTS)) {
      const svg = await readFile(join(outputDir, name), 'utf8');
      assertInsideCanvas(svg);
      assertTextInsideRects(svg, name);
      for (const needle of needles[name]) assert.ok(svg.includes(needle), `${name} 必须出现 ${needle}`);
      assert.doesNotMatch(svg, /\[\[/, 'SVG 不得泄漏 wikilink 标记');
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
  const a = m.align;
  const s = m.accept;
  const t = Object.fromEntries(m.rename.traces.map((x) => [x.key, x]));
  for (const needle of [
    `\`full_loss_masks = [${a.full.join(', ')}]\``,
    `\`full[i+2] = [${a.maskPatched.join(', ')}]\``,
    `\`num_tokens = ${a.numTokens}\``,
    `计入的目标是 t${a.counted[0]}–t${a.counted[a.counted.length - 1]}，而 response 区间是 t${a.response[0]}–t${a.response[a.response.length - 1]}`,
    `\`${t.decoderExpert.global}\``,
    `\`${t.mtpExpert.global}\``,
    `\`${t.mtpProj.hf[0]}\``,
    `\`mtp.layers.${m.rename.wrongLayer}\` 会被转成 \`${m.rename.wrongHf.replace('.eh_proj.weight', '')}\``,
    `\`model.layers.0–${m.rename.hfLayerCount - 1}\``,
    `src_rank = ${m.rename.embeddingSrc}`,
    `PP group {${m.rename.ppGroup.join(', ')}}`,
    `\`expert_offset = 8 × 1 // 2 = ${m.rename.ranks[3].expertOffset}\``,
    `5/6 = ${fmt(s.compatible[0].rate)}`,
    `12/60 = ${fmt(s.compatible[1].rate)}`,
    `(0.833 + 0.200) / 2 = ${fmt(s.meanRate)}`,
    `(4.00 + 1.70) / 2 = ${fmt(s.meanLength, 3)}`,
    `(5 + 12) / (6 + 60) = ${fmt(s.pooledRate)}`,
    `(8 + 34) / (2 + 20) = ${fmt(s.pooledLength, 3)}`,
    `本例的 length 仍是 ${fmt(s.pinnedLength, 3)}`,
    `本例每次 verify 提议 ${s.proposedPerVerify} 个`,
  ]) {
    assert.ok(page.includes(needle), `正文必须出现 ${needle}`);
  }
  for (const name of Object.values(OUTPUTS)) assert.ok(page.includes(`assets/${name}`), `正文必须引用 ${name}`);
  assert.ok(page.includes('> **源码基线**：`THUDM/slime@4c193f1f37509cca70f0e88807a9305b70f63f4e`'), '页头基线必须是 4c193f1f');
  assert.doesNotMatch(page, /github\.com\/[^)\s]*\/blob\/[0-9a-f]+\/[^)]*#L\d+/, '正文不再保留逐句行号永久链接');
  assert.doesNotMatch(page, /_named_params_and_buffers_(global|vanilla)\b(?!` 改为)/, '已删除的函数名只能出现在漂移说明里');
});
