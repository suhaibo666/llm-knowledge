import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {model, draw, out} from './vllm_w2_20_marlin_warp_tile.mjs';

const page = readFileSync(
  new URL('../../../wiki/02_engineering/03_infer_frameworks/vllm/20_vllm_fused_ops_and_kernels_analysis.md', import.meta.url),
  'utf8',
);

test('the lane map tiles one warp block exactly once', () => {
  const m = model();
  // 4 warps x 16 columns = tile_n_size; tile_size uint32 per tile.
  assert.equal(m.warps * 16, m.tileN);
  assert.equal(m.tileN, m.tileK * 4);
  assert.equal(m.tileSize, (m.tileK * m.tileN) / m.packFactor);
  assert.equal(m.tileSize, 128);
  // Every (k, n) in warp 0's 16x16 sub-block has exactly one owner.
  const count = new Map();
  for (let k = 0; k < m.tileK; k++) {
    for (let n = 0; n < 16; n++) {
      const lane = m.ownerOf[k][n];
      assert.ok(lane >= 0 && lane < m.lanes, `no owner for (k=${k}, n=${n})`);
      count.set(lane, (count.get(lane) ?? 0) + 1);
    }
  }
  assert.equal(count.size, m.lanes);
  for (const [lane, c] of count) assert.equal(c, 8, `lane ${lane} owns ${c} cells, expected 8`);
});

test('the worked lane reproduces the repack kernel arithmetic', () => {
  const m = model();
  // th_id = 5 -> tc_col = 5/4 = 1, tc_row = (5%4)*2 = 2, cur_n = 0*16 + 1.
  assert.deepEqual([m.worked.tcCol, m.worked.tcRow, m.worked.curN], [1, 2, 1]);
  assert.deepEqual(m.worked.cells.map(c => c.k), [2, 3, 10, 11, 2, 3, 10, 11]);
  assert.deepEqual(m.worked.cells.map(c => c.n), [1, 1, 1, 1, 9, 9, 9, 9]);
  // out_ptr[out_offset + th_id*4 + warp_id]
  assert.equal(m.outIndex, m.wLane * 4 + m.wWarp);
  assert.equal(m.outIndex, 20);
  // Every out index in a tile is hit exactly once by (lane, warp).
  const hit = new Set();
  for (let w = 0; w < m.warps; w++) for (let l = 0; l < m.lanes; l++) hit.add(l * 4 + w);
  assert.equal(hit.size, m.tileSize);
});

test('get_scale_perms is reproduced with both lengths', () => {
  const m = model();
  assert.equal(m.scalePerm.length, 64);
  assert.equal(m.scalePermSingle.length, 32);
  assert.deepEqual(m.scalePerm.slice(0, 10), [0, 8, 16, 24, 32, 40, 48, 56, 1, 9]);
  assert.deepEqual(m.scalePermSingle.slice(0, 10), [0, 1, 8, 9, 16, 17, 24, 25, 2, 3]);
  // 64-entry permutation is an 8x8 transpose: output 8i+j takes source i+8j.
  assert.deepEqual([...m.scalePerm].sort((a, b) => a - b), Array.from({length: 64}, (_, i) => i));
});

test('page prose uses the numbers the figure is drawn from', () => {
  const m = model();
  assert.ok(page.includes('assets/vllm_w2_20_marlin_warp_tile.svg'), 'page does not embed the figure');
  assert.ok(page.includes(`tile_n_size = tile_k_size * 4 = ${m.tileN}`), 'tile_n_size drifted');
  assert.ok(page.includes(`\`tile_size = ${m.tileK} * ${m.tileN} / ${m.packFactor} = ${m.tileSize}\``), 'tile_size arithmetic drifted');
  assert.ok(page.includes(`${m.tileK}(K)×${m.tileN}(N) = ${m.tileK * m.tileN} 个 4-bit 值`), 'tile cell count drifted');
  assert.ok(page.includes(`tc_offsets = {${m.tcOffsets.join(', ')}}`), 'tc_offsets drifted');
  assert.ok(page.includes(`pack_idx = {${m.packIdx.join(',')}}`), 'pack_idx drifted');
  assert.ok(
    page.includes(`\`th_id = ${m.wLane}\`、\`warp_id = ${m.wWarp}\` → \`tc_col = ${m.worked.tcCol}\`、\`tc_row = ${m.worked.tcRow}\`、\`cur_n = ${m.worked.curN}\``),
    'worked lane header drifted',
  );
  const nibbleText = m.nibbles.map(c => `\`(k=${c.k},n=${c.n})\``).join('、');
  assert.ok(page.includes(nibbleText), 'nibble order drifted');
  assert.ok(page.includes(`out_offset + ${m.outIndex}`), 'output index drifted');
  assert.ok(page.includes(`* ${m.tileSize}\``), 'out_offset formula drifted');
  assert.ok(page.includes(`${m.scalePerm.length} 项的 \`scale_perm\``), 'scale_perm length drifted');
  assert.ok(page.includes(`${m.scalePermSingle.length} 项的 \`scale_perm_single\``), 'scale_perm_single length drifted');
});

// ---- Layout geometry: the numeric tests above cannot see clipping or overlap. ----
const FONT = {title: 20, body: 15, small: 13, tiny: 10};
const attr = (tag, name) => {
  const m = tag.match(new RegExp(`\\s${name}="([^"]*)"`));
  return m ? m[1] : undefined;
};
const unescape = s => s.replaceAll('&lt;', '<').replaceAll('&gt;', '>').replaceAll('&amp;', '&');
// Conservative advance-width estimate: CJK and full-width glyphs ~1em, others ~0.6em.
const textWidth = (str, fs) =>
  [...str].reduce((w, ch) => w + (ch.codePointAt(0) >= 0x2e80 ? fs : 0.6 * fs), 0);

function layout(svg) {
  const vb = svg.match(/viewBox="0 0 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)"/);
  const W = Number(vb[1]), H = Number(vb[2]);
  const rects = [...svg.matchAll(/<rect\b[^>]*>/g)].map(([tag]) => ({
    x: Number(attr(tag, 'x') ?? 0), y: Number(attr(tag, 'y') ?? 0),
    w: Number(attr(tag, 'width')), h: Number(attr(tag, 'height')),
  }));
  const texts = [...svg.matchAll(/<text\b([^>]*)>([^<]*)<\/text>/g)].map(([, a, raw]) => {
    const tag = ` ${a}`;
    const fs = FONT[attr(tag, 'class')];
    const str = unescape(raw);
    const w = textWidth(str, fs);
    const x0 = Number(attr(tag, 'x')), base = Number(attr(tag, 'y'));
    const anchor = attr(tag, 'text-anchor');
    const left = anchor === 'middle' ? x0 - w / 2 : anchor === 'end' ? x0 - w : x0;
    return {str, fs, base, left, right: left + w, top: base - 0.8 * fs, bottom: base + 0.2 * fs};
  });
  return {W, H, rects, texts};
}

test('every rect and text stays inside the viewBox', () => {
  const {W, H, rects, texts} = layout(draw());
  for (const r of rects) {
    assert.ok(r.x >= 0 && r.y >= 0, `rect starts outside viewBox at (${r.x}, ${r.y})`);
    assert.ok(r.x + r.w <= W, `rect right edge ${r.x + r.w} exceeds width ${W}`);
    assert.ok(r.y + r.h <= H, `rect bottom ${r.y + r.h} exceeds height ${H}`);
  }
  for (const t of texts) {
    assert.ok(t.base <= H && t.bottom <= H, `text "${t.str.slice(0, 30)}" baseline ${t.base} exceeds height ${H}`);
    assert.ok(t.top >= 0, `text "${t.str.slice(0, 30)}" top ${t.top} is above the viewBox`);
    assert.ok(t.left >= 0 && t.right <= W, `text "${t.str.slice(0, 30)}" spans ${t.left.toFixed(1)}..${t.right.toFixed(1)}, outside width ${W}`);
  }
});

test('no two text boxes overlap', () => {
  const {texts} = layout(draw());
  for (let i = 0; i < texts.length; i++) {
    for (let j = i + 1; j < texts.length; j++) {
      const a = texts[i], b = texts[j];
      const overlap = a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom;
      assert.ok(!overlap, `text "${a.str.slice(0, 24)}" overlaps "${b.str.slice(0, 24)}"`);
    }
  }
});

test('figure states its has_perm=false premise and explains both color uses', () => {
  const svg = draw();
  assert.ok(svg.includes('has_perm=false'), 'premise has_perm=false missing');
  assert.ok(svg.includes('ℓ%4 = 0 的 lane'), 'grid blue legend missing');
  assert.ok(svg.includes('低 4 个 nibble'), 'nibble-row color legend missing');
});

test('checked-in SVG is current and contains real text', () => {
  const svg = draw();
  assert.equal(readFileSync(out, 'utf8').trim(), svg.trim());
  assert.ok(svg.includes('<text') && !svg.includes('<foreignObject'));
});
