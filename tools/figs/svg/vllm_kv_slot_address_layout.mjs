// Source baseline: vllm@199cb9b964822e59ab9b58d88e7be31eb419a2ae.
// Conceptual layout, not an observed GPU allocation. Figure specification:
// Top: two layers x kernel blocks 12/13/14. Layer-major (LBNHC) rows order whole
// layers; block-major (BLNHC) rows order each block's layer pages. Block 13 (A's
// logical block 1) is blue in both. Bottom zoom: layer1 / kernel block 13 /
// offset 3 = A19, slot 211. LBNHC stores one token row as head0 K|V then head1
// K|V (D=64 each, H=2 local KV heads, TP=1). Orange ticks mark element d=5 of
// head0 K (content 5) and V (content 69). Caption: dense LBNHC byte offsets,
// computed from the same parameters as the page (see model()).
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
export const out = path.join(root, 'wiki/02_engineering/03_infer_frameworks/vllm/assets/vllm_kv_slot_address_layout.svg');

// Running example shared with pages 10/11/12: order [B, A], kernel block 16.
export function model() {
  const H = 2, D = 64, e = 2, Bk = 16, h = 0, d = 5;
  const tables = {B: [28], A: [12, 13]};
  const rows = [['B', 5], ['A', 18], ['A', 19]];
  const slots = rows.map(([r, pos]) => tables[r][Math.floor(pos / Bk)] * Bk + (pos % Bk));
  const block = tables.A[Math.floor(19 / Bk)], offset = 19 % Bk, slot = slots[2];
  // Byte strides of one LBNHC layer view [b, h, o, c] without padding
  // (port of compute_layout_strides: physical order b > o > h > c).
  const sC = e, sH = 2 * D * e, sO = H * 2 * D * e, sB = Bk * H * 2 * D * e;
  const k = block * sB + h * sH + offset * sO + d * sC;
  const v = k + D * e;
  const kFormula = (slot * H * 2 * D + h * 2 * D + d) * e;
  // LBHNC (physical b > h > o > c): same logical index, different byte strides.
  const lbhnc = {sB: H * Bk * 2 * D * e, sH: Bk * 2 * D * e, sO: 2 * D * e};
  const kLBHNC = block * lbhnc.sB + h * lbhnc.sH + offset * lbhnc.sO + d * sC;
  const vLBHNC = kLBHNC + D * e;
  // Manager 64 -> kernel 16: manager id m expands to m*4 .. m*4+3.
  const ratio = 64 / Bk, expand = m => Array.from({length: ratio}, (_, i) => m * ratio + i);
  return {H, D, e, Bk, h, d, slots, block, offset, slot, strides: {sB, sH, sO, sC}, k, v, kFormula,
    lbhnc, kLBHNC, vLBHNC, kIndex: d, vIndex: D + d, expandB: expand(7), expandA: expand(3)};
}

const esc = s => String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');

export function draw() {
  const m = model();
  const W = 960, Ht = 590;
  const s = [`<svg id="kv-address-svg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${Ht}" role="img" aria-labelledby="kv-title kv-desc"><title id="kv-title">同一 kernel block 的跨层视图与 A19 的 K/V 元素地址</title><desc id="kv-desc">两种共享 backing 排列的局部切片。普通 FlashAttention 每层 view 为块、head、state、内容；LBNHC 块内按 token、head、内容排列。K/V 在同一 head 的内容轴相邻，层间连续性由布局决定。</desc><style>#kv-address-svg text{font-family:Arial,'PingFang SC','Microsoft YaHei',sans-serif;fill:#0f172a}#kv-address-svg .title{font-size:20px;font-weight:600}#kv-address-svg .body{font-size:16px}#kv-address-svg .small{font-size:14px;fill:#475569}#kv-address-svg .neutral{fill:#fff;stroke:#64748b}#kv-address-svg .ghost{fill:#f8fafc;stroke:#94a3b8}#kv-address-svg .acc1{fill:#dbeafe;stroke:#2563eb}#kv-address-svg .acc2{fill:#ffedd5;stroke:#ea580c}</style><rect width="${W}" height="${Ht}" fill="white"/>`];
  const t = (x, y, str, c = 'body') => s.push(`<text x="${x}" y="${y}" class="${c}">${esc(str)}</text>`);
  const box = (x, y, w, h, c) => s.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" class="${c}"/>`);
  t(24, 32, '同一个编号 ≠ 同一种物理排列', 'title');
  t(24, 58, '共享 backing 的局部切片：每格是一层的一页；省略其他块与层。', 'small');
  const blocks = [m.block - 1, m.block, m.block + 1];
  const lane = (items, y) => items.forEach(([l, b], i) => {
    const x = 60 + i * 150;
    box(x, y, 138, 64, b === m.block ? 'acc1' : 'neutral');
    t(x + 14, y + 25, `layer ${l}`);
    t(x + 14, y + 49, `block ${b}`);
    if (i < items.length - 1) t(x + 140, y + 37, '›', 'small');
  });
  t(24, 92, 'LBNHC：按层排，再排块；地址向右增加 →');
  lane([0, 1].flatMap(l => blocks.map(b => [l, b])), 106);
  t(24, 196, `仅展示 layer 0/1 与 kernel block ${blocks.join('/')}；省略范围不表示地址相邻。`, 'small');
  t(24, 226, 'BLNHC：按块排，再排层；地址向右增加 →');
  lane(blocks.flatMap(b => [0, 1].map(l => [l, b])), 240);
  t(24, 340, `放大：layer 1 / kernel block ${m.block} / offset ${m.offset}（A19）；slot = ${m.block}×${m.Bk}+${m.offset} = ${m.slot}`);
  t(24, 365, `LBNHC 块内按 token → head → content：一个 token 行依次放 head 0、head 1，各为 K(${m.D}) 接 V(${m.D})。`, 'small');
  const segW = 220, x0 = 60, y0 = 386;
  const segs = [[0, 'K', 0, 'acc1'], [0, 'V', m.D, 'neutral'], [1, 'K', 0, 'ghost'], [1, 'V', m.D, 'ghost']];
  segs.forEach(([head, kv, c0, cls], j) => {
    const x = x0 + j * segW;
    box(x, y0, segW, 66, cls);
    // Labels start right of the d=5 tick (x + 5·cell ≈ x + 17..21) so the tick never covers text.
    t(x + 28, y0 + 26, `head ${head} · ${kv}：c ${c0}–${c0 + m.D - 1}`);
    t(x + 28, y0 + 52, `${m.D} 元素 × ${m.e} B = ${m.D * m.e} B`, 'small');
  });
  const cell = segW / m.D;
  box(+(x0 + m.d * cell).toFixed(2), y0, +cell.toFixed(2), 66, 'acc2');
  box(+(x0 + segW + m.d * cell).toFixed(2), y0, +cell.toFixed(2), 66, 'acc2');
  box(x0, 462, 210, 29, 'acc2');
  t(x0 + 10, 482, `K 的 d=${m.d} → cache[${m.block},0,${m.offset},${m.kIndex}]`, 'small');
  box(x0 + segW, 462, 210, 29, 'acc2');
  t(x0 + segW + 10, 482, `V 的 d=${m.d} → cache[${m.block},0,${m.offset},${m.vIndex}]`, 'small');
  t(x0 + 2 * segW + 10, 482, `head 1 → cache[${m.block},1,${m.offset},0:${2 * m.D}]`, 'small');
  t(24, 526, `无 padding、LBNHC、H=${m.H}、D=${m.D}、B_k=${m.Bk}、BF16：K 偏移 (${m.slot}·${m.H * 2 * m.D}+${m.d})·${m.e} = ${m.k} B；V 偏移 ${m.v} B。`, 'small');
  t(24, 550, `两者只差 D×e = ${m.D * m.e} B；b·B_k+o = ${m.slot} 正是 slot_mapping 的值。偏移从该层 view 起点算起。`, 'small');
  t(24, 574, '跨层 / 跨 head 的连续性仍要读真实 stride；换布局或 padding 后致密简式失效。', 'small');
  s.push('</svg>');
  return s.join('\n');
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  fs.mkdirSync(path.dirname(out), {recursive: true});
  fs.writeFileSync(out, draw() + '\n');
  console.log(out);
}
