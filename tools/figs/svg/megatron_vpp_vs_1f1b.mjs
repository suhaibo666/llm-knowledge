// 图：标准 1F1B vs 交错 1F1B(VPP) —— 气泡率 ÷ vp，代价是 P2P ×vp
// 用于 wiki/02_engineering/02_train_frameworks/megatron-lm/15_megatron_pp_schedulers_analysis.md §③.3
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
// 要讲清楚：VPP 把每卡的连续层段拆成 vp 个不连续 chunk，单个「设备-块」的耗时降为 t_f/vp，
// 填充/排空随之缩短 vp 倍 → 气泡率 (pp-1)/m → (pp-1)/(m·vp)。代价：一个 microbatch 要在
// 设备间往返 vp 趟 → P2P 次数 ×vp。
//
// 布局（上下两面板，共用一条按真实耗时刻度的时间轴，这是全图的立论所在）：
//   上 标准 1F1B：4 行 × 22 格，每格宽 2 单位（一格 = t_f）
//   下 VPP      ：4 行 × 38 格，每格宽 1 单位（一格 = t_f/vp）
//   轴总长 44 单位；1F1B 铺满 44，VPP 只到 38 —— 「更快」是量出来的，不是写出来的
//   每面板下方一条 makespan 尺规，标 22 t_f / 19 t_f
// 配色：前向=蓝族、反向=绿族；chunk0 浅档、chunk1 深档（1F1B 无 chunk 概念，用深档）；气泡=中性斜纹
// 强调（每图至多两个）：1F1B 右侧 acc2（代价色）标空泡比 6/16；VPP 右侧 acc1（收益色）标 6/32
// 额外一处：VPP 面板里把 mb0 的 8 个前向格描 acc1 边框 + 旁注「往返 vp=2 趟 → P2P ×vp」
//   —— 同一处标注同时解释收益与代价，避免再开一张图
//
// 数据来源：lib/megatron_pp_sim.mjs 的离散事件仿真（算法照 Megatron@71092579 的 schedules.py），
// 已由 lib/megatron_pp_sim.test.mjs 锁定与页面 ASCII 表逐格一致。图上每个数字都是算出来的。
//
// 用法: node tools/figs/svg/megatron_vpp_vs_1f1b.mjs > <页面目录>/assets/megatron_pp_vpp_vs_1f1b.svg

import { seq1f1b, seqVpp, simulate } from './lib/megatron_pp_sim.mjs';

const PP = 4, M = 8, VP = 2, N = 4;
const A = simulate(seq1f1b({ pp: PP, m: M }), { pp: PP, vp: 1 });          // 标准 1F1B
const B = simulate(seqVpp({ pp: PP, m: M, vp: VP, N }), { pp: PP, vp: VP }); // VPP

/* ---------- 几何：U = 一个 chunk-op 的宽度，1F1B 一格 = VP×U ---------- */
const U = 20, CH = 22, ROWG = 3, PAD = 20, LAB = 64, NOTE = 172, INSET = 1.5;
const axisU = Math.max(A.span * VP, B.span);            // 44 vs 38 → 44
const gridW = axisU * U;
const W = PAD + LAB + gridW + 14 + NOTE + PAD;
const panelH = PP * (CH + ROWG) - ROWG;

const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const body = [];
const put = (s) => body.push(s);

/** 一个面板。cellU = 每格占多少个 U（1F1B=VP，VPP=1）。 */
function panel(y, title, sub, sim, cellU, { tone, note, highlight }) {
  put(`  <text class="pt" x="${PAD}" y="${y}">${esc(title)}</text>`);
  put(`  <text class="ps" x="${PAD}" y="${y + 17}">${esc(sub)}</text>`);
  const top = y + 30, cw = cellU * U;
  sim.rows.forEach((row, r) => {
    const ry = top + r * (CH + ROWG);
    put(`  <text class="lab" x="${PAD + LAB - 10}" y="${ry + CH / 2 + 4}">Dev${r}</text>`);
    row.forEach((op, t) => {
      const x = PAD + LAB + t * cw;
      if (op === null) {
        put(`  <rect class="bub" x="${x + INSET}" y="${ry}" width="${cw - 2 * INSET}" height="${CH}" rx="3"/>`);
        return;
      }
      const kind = (op.f ? 'f' : 'b') + (op.c === 0 ? '0' : '1');
      put(`  <rect class="${kind}" x="${x + INSET}" y="${ry}" width="${cw - 2 * INSET}" height="${CH}" rx="3"/>`);
      const tag = (op.c === 0 ? (op.f ? 'f' : 'b') : (op.f ? 'F' : 'B')) + op.mb;
      put(`  <text class="cl" x="${x + cw / 2}" y="${ry + CH / 2 + 3.5}">${tag}</text>`);
    });
  });
  // 高亮 mb0 的前向路径：chunk0 走完 pp 个 stage，再回到 Dev0 起 chunk1
  if (highlight) {
    sim.rows.forEach((row, r) => {
      const ry = top + r * (CH + ROWG);
      row.forEach((op, t) => {
        if (op && op.f && op.mb === 0) {
          put(`  <rect class="hl" x="${PAD + LAB + t * cw + INSET}" y="${ry}" width="${cw - 2 * INSET}" height="${CH}" rx="3"/>`);
        }
      });
    });
  }
  // makespan 尺规：长度按真实耗时刻度，两条一比就是结论
  const my = top + panelH + 10, len = sim.span * cw;
  put(`  <line class="rule" x1="${PAD + LAB}" y1="${my}" x2="${PAD + LAB + len}" y2="${my}"/>`);
  put(`  <line class="tick" x1="${PAD + LAB + len}" y1="${my - 5}" x2="${PAD + LAB + len}" y2="${my + 5}"/>`);
  put(`  <text class="mk" x="${PAD + LAB + len / 2}" y="${my + 14}">makespan = ${sim.span * cellU / VP} t_f</text>`);
  // 右侧唯一的强调框
  const bx = PAD + LAB + gridW + 14, by = top + panelH / 2 - 26;
  put(`  <rect class="box-${tone}" x="${bx}" y="${by}" width="${NOTE}" height="52" rx="8"/>`);
  note.forEach((ln, i) => put(`  <text class="txt-${tone}" x="${bx + 11}" y="${by + 18 + i * 15}">${esc(ln)}</text>`));
  return my + 26;
}

let y = PAD + 60;
y = panel(y, '① 标准 1F1B（调度器②）—— 每卡一段连续层',
  `一格 = t_f（一个 microbatch 走完本 stage 整段层）`, A, VP,
  { tone: 'cost', note: ['空泡 / 计算 = 6 / 16', `= (pp−1)/m = 3/8 = 37.5%`] });
y += 16;
y = panel(y, '② 交错 1F1B / VPP（调度器③）—— 每卡 vp=2 个不连续 chunk',
  `一格 = t_f/vp（只走 1 个 chunk）· f/b = chunk0，F/B = chunk1`, B, 1,
  { tone: 'gain', highlight: true, note: ['空泡 / 计算 = 6 / 32', `= (pp−1)/(m·vp) = 3/16 = 18.75%`] });

const H = y + 62;

const head = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="100%" role="img"
     aria-label="标准 1F1B 与 VPP 交错调度的同轴对照：VPP 把每格耗时降为 t_f/vp，空泡比从 3/8 降到 3/16，makespan 从 22 t_f 降到 19 t_f，代价是每个 microbatch 往返 vp 趟、P2P 次数乘以 vp">
  <style>
    text{font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif}
    .card{fill:#fff;stroke:#E4E7EC}
    .ti{font-size:15px;font-weight:700;fill:#1F2430}
    .su{font-size:11.5px;fill:#8A8F98}
    .pt{font-size:13px;font-weight:700;fill:#2A313B}
    .ps{font-size:11px;fill:#8A8F98}
    .lab{font-size:11px;fill:#5B6470;text-anchor:end}
    .cl{font-size:9.5px;font-weight:700;fill:#fff;text-anchor:middle}
    .f0{fill:#9CC2F3}   /* 前向 chunk0：蓝族浅档 */
    .f1{fill:#3E7BD0}   /* 前向 chunk1：蓝族深档 */
    .b0{fill:#A5D6BC}   /* 反向 chunk0：绿族浅档 */
    .b1{fill:#3E9970}   /* 反向 chunk1：绿族深档 */
    .bub{fill:url(#hatch);stroke:#E0DCD2}
    .hl{fill:none;stroke:#2563EB;stroke-width:2}
    .hlt{font-size:10.5px;font-weight:600;fill:#173F87}
    .rule{stroke:#C7CCD3;stroke-dasharray:3 3}
    .tick{stroke:#9AA1AC;stroke-width:1.5}
    .mk{font-size:10px;fill:#9AA1AC;text-anchor:middle}
    .box-cost{fill:#FCF1E6;stroke:#C3651F;stroke-width:1.4}
    .txt-cost{font-size:10.5px;font-weight:600;fill:#8A4A11}
    .box-gain{fill:#EAF1FD;stroke:#2563EB;stroke-width:1.4}
    .txt-gain{font-size:10.5px;font-weight:600;fill:#173F87}
    .cap{font-size:11px;fill:#7A808A}
    .lg{font-size:10.5px;fill:#7A808A}
  </style>
  <defs>
    <pattern id="hatch" width="6" height="6" patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
      <rect width="6" height="6" fill="#F3F1EB"/>
      <rect width="3" height="6" fill="#E4E0D6"/>
    </pattern>
  </defs>
  <rect class="card" x="0.5" y="0.5" width="${W - 1}" height="${H - 1}" rx="14"/>
  <text class="ti" x="${PAD}" y="${PAD + 14}">交错 1F1B(VPP)：气泡率 ÷ vp，代价是 P2P ×vp</text>
  <text class="su" x="${PAD}" y="${PAD + 32}">pp=4, m=8, vp=2, N=4 · 上下两面板共用同一条按真实耗时刻度的时间轴（每格宽度 ∝ 该格耗时）· 斜纹 = 空泡</text>
  <text class="su" x="${PAD}" y="${PAD + 48}">蓝 = 前向，绿 = 反向；浅档 = chunk0，深档 = chunk1</text>`;

const tail = `  <text class="cap" x="${PAD}" y="${y + 8}">同一条轴上量：makespan 从 ${A.span} t_f 降到 ${B.span / VP} t_f；空泡/计算从 ${A.bubble}/${A.ops} = (pp−1)/m 降到 ${B.bubble}/${B.ops} = (pp−1)/(m·vp)，正好 ÷vp。</text>
  <text class="cap" x="${PAD}" y="${y + 24}"><tspan fill="#173F87" font-weight="700">蓝框</tspan> = mb0 的前向路径：chunk0 走完 4 个 stage 后回到 Dev0 才起 chunk1 —— 一个 microbatch 往返 vp=2 趟，这正是收益的来源，也是代价的来源。</text>
  <text class="cap" x="${PAD}" y="${y + 40}">代价：P2P 次数 ×vp；峰值激活 ≈ (1+1/vp)·pp·A（略高于 1F1B）。跨机时须配 overlap_p2p_comm（调度器④）把这 vp× 通信藏起来。</text>
</svg>`;

console.log([head, ...body, tail].join('\n'));
