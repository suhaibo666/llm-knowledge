// Source baseline: vllm@199cb9b964822e59ab9b58d88e7be31eb419a2ae.
// Figure specification: two-axis grid. Axis 1 = k (0..15 inside one repack
// k-tile), axis 2 = n (0..63 inside one repack n-tile). Every (k, n) cell
// carries a per-cell decision: which lane of which warp holds that 4-bit
// weight, and which nibble of which output uint32 it lands in. Left block
// draws warp 0's full 16x16 sub-block with the owning lane id in every cell;
// the three remaining warp bands are collapsed because their lane map is the
// same shifted by 16 columns. The panel below the grid replays one lane
// (warp 0, lane 5) from its four k offsets {tc_row+0,+1,+8,+9} and two n
// columns {cur_n, cur_n+8} into the eight nibbles of out_ptr[out_offset +
// 4*lane + warp]. Bottom strip shows the two scale permutations. Every cell
// shows only the owning lane id; nibble slot and output index are expanded for
// lane 5 only. Assumes num_bits=4, is_a_8bit=false, has_perm=false (no
// act-order): with act-order the value read at cell k comes from perm[k].
// Layout uses a y cursor and the viewBox height is derived from it; the test
// checks every element stays inside the viewBox and no two text boxes collide. Ported from
// csrc/libtorch_stable/quantization/marlin/gptq_marlin_repack.cu
// ::gptq_marlin_repack_kernel and marlin_utils.py::get_scale_perms.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
export const out = path.join(
  root,
  'wiki/02_engineering/03_infer_frameworks/vllm/assets/vllm_w2_20_marlin_warp_tile.svg',
);

// Port of the repack kernel's index arithmetic for num_bits=4, is_a_8bit=false.
export function model() {
  const numBits = 4;
  const packFactor = 32 / numBits;            // 8 values per uint32
  const tileK = 16;                           // tile_k_size
  const tileN = 64;                           // tile_n_size = tile_k_size * 4
  const warps = 4;                            // warp_id >= 4 returns
  const lanes = 32;
  const tileInts = tileK / packFactor;        // 2
  const tileSize = (tileK * tileN) / packFactor;  // 128 uint32 per tile
  const tcOffsets = [0, 1, 8, 9];
  const packIdx = [0, 2, 4, 6, 1, 3, 5, 7];

  // lane -> the eight (k, n) cells it owns inside its warp's 16 columns.
  const laneCells = (warp, lane) => {
    const tcCol = Math.floor(lane / 4);
    const tcRow = (lane % 4) * 2;
    const curN = warp * 16 + tcCol;
    const cells = [];
    for (let i = 0; i < 4; i++) cells.push({k: tcRow + tcOffsets[i], n: curN, val: i});
    for (let i = 0; i < 4; i++) cells.push({k: tcRow + tcOffsets[i], n: curN + 8, val: 4 + i});
    return {tcCol, tcRow, curN, cells};
  };

  // Owner map for warp 0's 16x16 sub-block: ownerOf[k][n] = lane id.
  const ownerOf = Array.from({length: tileK}, () => new Array(16).fill(-1));
  for (let lane = 0; lane < lanes; lane++) {
    for (const c of laneCells(0, lane).cells) ownerOf[c.k][c.n] = lane;
  }

  // Worked lane: warp 0, lane 5.
  const wLane = 5, wWarp = 0;
  const worked = laneCells(wWarp, wLane);
  const outIndex = wLane * 4 + wWarp;         // out_ptr[out_offset + th_id*4 + warp_id]
  // nibble slot i receives vals[packIdx[i]]
  const nibbles = packIdx.map(v => worked.cells[v]);

  // get_scale_perms(): 64-entry grouped permutation, 32-entry column-wise one.
  const scalePerm = [];
  for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) scalePerm.push(i + 8 * j);
  const scalePermSingle = [];
  for (let i = 0; i < 4; i++) for (const j of [0, 1, 8, 9, 16, 17, 24, 25]) scalePermSingle.push(2 * i + j);

  return {
    numBits, packFactor, tileK, tileN, warps, lanes, tileInts, tileSize,
    tcOffsets, packIdx, ownerOf, wLane, wWarp, worked, outIndex, nibbles,
    scalePerm, scalePermSingle, laneCells,
  };
}

const esc = s => String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');

export function draw() {
  const m = model();
  const W = 1080;
  const body = [];
  const t = (x, y, str, c = 'body', anchor) =>
    body.push(`<text x="${x}" y="${y}" class="${c}"${anchor ? ` text-anchor="${anchor}"` : ''}>${esc(str)}</text>`);
  const box = (x, y, w, h, c) => body.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" class="${c}"/>`);

  let y = 32;
  t(24, y, `一个 ${m.tileK}×${m.tileN} repack tile：每格 (k, n) 归哪条 lane，落到哪个 nibble`, 'title');
  y += 24;
  t(24, y, `前提：num_bits=${m.numBits}、pack_factor=${m.packFactor}、is_a_8bit=false、has_perm=false（无 act-order）；tile 内 ${m.tileK * m.tileN} 个 4 bit 权重压成 ${m.tileSize} 个 uint32。`, 'small');
  y += 34;

  // ---- Panel B: warp 0's 16x16 lane-owner grid ----
  t(24, y, 'warp 0 的 16 列（n = 0..15）：格内数字是持有该权重的 lane 号', 'body');
  y += 26;
  const gx = 70, cw = 27, ch = 22;
  t(gx + 8 * cw, y, 'n（tile 内输出通道）', 'small', 'middle');
  const bx = gx + 16 * cw + 26;
  t(bx, y, '其余三个 warp', 'small');
  y += 20;
  for (let n = 0; n < 16; n++) t(gx + n * cw + cw / 2, y, String(n), 'tiny', 'middle');
  y += 6;
  const gy = y;
  for (let k = 0; k < m.tileK; k++) {
    t(gx - 8, gy + k * ch + 16, `k=${k}`, 'tiny', 'end');
    for (let n = 0; n < 16; n++) {
      const lane = m.ownerOf[k][n];
      const cls = lane === m.wLane ? 'acc2' : (lane % 4 === 0 ? 'acc1' : 'neutral');
      box(gx + n * cw, gy + k * ch, cw, ch, cls);
      t(gx + n * cw + cw / 2, gy + k * ch + 16, String(lane), 'tiny', 'middle');
    }
  }
  const gh = m.tileK * ch;
  // ---- Collapsed warp bands ----
  for (let w = 1; w < m.warps; w++) {
    const by = gy + (w - 1) * (gh / 3);
    box(bx, +by.toFixed(2), 196, +(gh / 3 - 8).toFixed(2), 'ghost');
    t(bx + 10, +(by + 24).toFixed(2), `warp ${w}`, 'body');
    t(bx + 10, +(by + 46).toFixed(2), `n = ${w * 16}..${w * 16 + 15}`, 'small');
    t(bx + 10, +(by + 66).toFixed(2), `cur_n = ${w}×16 + tc_col`, 'small');
    t(bx + 10, +(by + 86).toFixed(2), 'lane 映射与左图相同', 'small');
  }
  // ---- Grid legend ----
  const lx = bx + 216;
  box(lx, gy, 14, 14, 'acc2');
  t(lx + 22, gy + 12, `橙：lane ${m.wLane}，下方逐 nibble 复演`, 'small');
  box(lx, gy + 26, 14, 14, 'acc1');
  t(lx + 22, gy + 38, 'ℓ%4 = 0 的 lane（tc_row = 0，', 'small');
  t(lx + 22, gy + 56, '持有 k = 0,1,8,9），只为看清条纹', 'small');
  box(lx, gy + 70, 14, 14, 'neutral');
  t(lx + 22, gy + 82, '其余 lane', 'small');
  y = gy + gh + 24;
  t(24, y, `lane ℓ 持有 k ∈ {tc_row+${m.tcOffsets.join(', tc_row+')}}（tc_row=(ℓ%4)×2）与 n ∈ {tc_col, tc_col+8}（tc_col=ℓ/4）：4×2 = 8 格。`, 'small');
  y += 20;
  t(24, y, `32 条 lane × 8 格 = 256 格，恰好铺满一个 warp 的 16×16；4 个 warp 拼出 ${m.tileN} 列。`, 'small');
  y += 40;

  // ---- Panel C: worked lane ----
  const px = 24;
  t(px, y, `复演一条 lane：warp ${m.wWarp}、lane ${m.wLane} → tc_col=${m.worked.tcCol}、tc_row=${m.worked.tcRow}、cur_n=${m.worked.curN}`, 'body');
  y += 14;
  const nw = 118, nh = 46;
  m.nibbles.forEach((c, i) => {
    const x = px + i * (nw + 6);
    box(x, y, nw, nh, i < 4 ? 'acc1' : 'acc2');
    t(x + 8, y + 18, `nibble ${i}`, 'tiny');
    t(x + 8, y + 36, `(k=${c.k}, n=${c.n})`, 'small');
  });
  y += nh + 22;
  t(px, y, '这一行的颜色另有含义：蓝 = 低 4 个 nibble（k 偏移 +0 / +8），橙 = 高 4 个 nibble（k 偏移 +1 / +9）；与上方网格的配色无关。', 'small');
  y += 20;
  t(px, y, `nibble i 装 vals[pack_idx[i]]，pack_idx = [${m.packIdx.join(', ')}]（源码注释指向 FasterTransformer 的 interleaved numeric conversion）。`, 'small');
  y += 20;
  t(px, y, `这 8 个 nibble 组成一个 uint32，写到 out_ptr[out_offset + th_id×4 + warp_id] = out_offset + ${m.wLane}×4 + ${m.wWarp} = out_offset + ${m.outIndex}。`, 'small');
  y += 20;
  t(px, y, `out_offset = (k_tile_id × n_tiles + n_tile_id) × ${m.tileSize}：tile 之间按 K 外、N 内连续排列，tile 内按 lane 再 warp 排列。`, 'small');
  y += 22;

  // ---- Panel D: scale permutations ----
  const sy = y;
  box(px, sy, W - 2 * px, 80, 'ghost');
  t(px + 12, sy + 22, 'scale 有两组置换，选择条件是 group_size < size_k 且 group_size != -1 且非 8 bit activation：', 'small');
  t(px + 12, sy + 42, `分组：${m.scalePerm.length} 项 8×8 转置，前 10 项 = [${m.scalePerm.slice(0, 10).join(', ')}, …]；列向：${m.scalePermSingle.length} 项，前 10 项 = [${m.scalePermSingle.slice(0, 10).join(', ')}, …]。`, 'small');
  t(px + 12, sy + 62, 'kernel 侧注释给的理由：分组量化按 column-major 缩放一个 half2 tile，列向量化按 row-major，s_sh_rd 因此分支取不同公式。', 'small');
  y = sy + 80 + 26;

  t(px, y, '本图是从 repack kernel 的索引算术重建的静态布局，不是 GPU 上观测到的寄存器分配。', 'small');
  y += 20;
  t(px, y, 'GEMM kernel 自身的 thread_n_blocks 分块与本图的 4 warp / 64 列 repack tile 不是同一层。', 'small');
  const H = y + 18;

  const head =
    `<svg id="marlin-warp-svg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="mw-title mw-desc">` +
    `<title id="mw-title">Marlin repack 之后一个 16×64 tile 的 warp/lane 归属与 nibble 落位</title>` +
    `<desc id="mw-desc">两根轴联合决定逐格归属：横轴是 tile 内的输出通道 n，纵轴是 tile 内的归约维 k。每格只标持有该 4 bit 权重的 lane 号；网格下方以 warp 0 的 lane 5 为例展开它的八个 (k, n)、输出 uint32 的八个 nibble 槽位与输出下标。前提为 num_bits=4、is_a_8bit=false、has_perm=false。</desc>` +
    `<style>#marlin-warp-svg text{font-family:Arial,'PingFang SC','Microsoft YaHei',sans-serif;fill:#0f172a}` +
    `#marlin-warp-svg .title{font-size:20px;font-weight:600}` +
    `#marlin-warp-svg .body{font-size:15px}` +
    `#marlin-warp-svg .small{font-size:13px;fill:#475569}` +
    `#marlin-warp-svg .tiny{font-size:10px;fill:#334155}` +
    `#marlin-warp-svg .neutral{fill:#fff;stroke:#64748b}` +
    `#marlin-warp-svg .ghost{fill:#f8fafc;stroke:#94a3b8}` +
    `#marlin-warp-svg .acc1{fill:#dbeafe;stroke:#2563eb}` +
    `#marlin-warp-svg .acc2{fill:#ffedd5;stroke:#ea580c}</style>` +
    `<rect width="${W}" height="${H}" fill="white"/>`;
  return [head, ...body, '</svg>'].join('\n');
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  fs.mkdirSync(path.dirname(out), {recursive: true});
  fs.writeFileSync(out, draw() + '\n');
  console.log(out);
}
