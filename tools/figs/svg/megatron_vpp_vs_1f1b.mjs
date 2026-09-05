// P=4,m=8：no-pipeline、non-interleaved 1F1B 与 VPP/interleaved 1F1B。
// 时间格与数字全部由 lib/megatron_pp_sim.mjs 计算。
// Usage: node tools/figs/svg/megatron_vpp_vs_1f1b.mjs > .../assets/megatron_pp_vpp_vs_1f1b.svg

import { seq1f1b, seqVpp, simulate } from './lib/megatron_pp_sim.mjs';

const P = 4, M = 8, V = 2, N = 4;
const standard = simulate(seq1f1b({ pp: P, m: M }), { pp: P, vp: 1 });
const vpp = simulate(seqVpp({ pp: P, m: M, vp: V, N }), { pp: P, vp: V });

function peakLive(sim) {
  return sim.rows.map((row) => {
    let live = 0, peak = 0;
    for (const op of row) {
      if (!op) continue;
      live += op.f ? 1 : -1;
      peak = Math.max(peak, live);
    }
    return peak;
  });
}

const standardLive = peakLive(standard);
const vppLive = peakLive(vpp);
const standardTransfers = 2 * M * (P - 1);
const vppTransfers = 2 * M * (P * V - 1);
const transferRatio = `${P * V - 1}/${P - 1}`;

const W = 1480, H = 850, X0 = 112, U = 21, ROW_H = 24, ROW_GAP = 4;
const axisUnits = standard.span * V; // 44 chunk-time units; standard cell is V units.
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const out = [];
const put = (s) => out.push(s);
const label = (op) => `${op.c ? (op.f ? 'F' : 'B') : (op.f ? 'f' : 'b')}${op.mb}`;

function multiline(x, y, lines, cls = 'sm', gap = 16, anchor = 'start') {
  put(`<text class="${cls}" x="${x}" y="${y}" text-anchor="${anchor}">${lines.map((line, i) => `<tspan x="${x}" dy="${i ? gap : 0}">${esc(line)}</tspan>`).join('')}</text>`);
}

function schedulePanel(y, title, subtitle, sim, cellUnits, live, metrics) {
  put(`<text class="pt" x="28" y="${y}">${esc(title)}</text>`);
  put(`<text class="sub" x="28" y="${y + 20}">${esc(subtitle)}</text>`);
  const top = y + 36;
  sim.rows.forEach((row, rank) => {
    const ry = top + rank * (ROW_H + ROW_GAP);
    put(`<text class="lane" x="${X0 - 12}" y="${ry + 16}">rank ${rank}</text>`);
    row.forEach((op, t) => {
      const x = X0 + t * cellUnits * U;
      const width = cellUnits * U - 2;
      if (!op) {
        put(`<rect class="bubble" x="${x + 1}" y="${ry}" width="${width}" height="${ROW_H}" rx="3"/>`);
      } else {
        const cls = op.f ? (op.c ? 'f-light' : 'f') : (op.c ? 'b-light' : 'b');
        const focus = op.mb === 0 ? ' focus' : '';
        put(`<rect class="${cls}${focus}" x="${x + 1}" y="${ry}" width="${width}" height="${ROW_H}" rx="3"/>`);
        put(`<text class="cell" x="${x + cellUnits * U / 2}" y="${ry + 16}">${label(op)}</text>`);
      }
    });
  });
  const rulerY = top + 4 * (ROW_H + ROW_GAP) + 4;
  const len = sim.span * cellUnits * U;
  put(`<path class="ruler" d="M ${X0} ${rulerY} H ${X0 + len} M ${X0} ${rulerY - 5} V ${rulerY + 5} M ${X0 + len} ${rulerY - 5} V ${rulerY + 5}"/>`);
  put(`<text class="ruler-t" x="${X0 + len / 2}" y="${rulerY + 18}">${esc(metrics)}</text>`);
  const bx = 1080;
  put(`<rect class="metric" x="${bx}" y="${top - 4}" width="370" height="112" rx="10"/>`);
  multiline(bx + 16, top + 20, [
    `峰值 live forward record / rank = [${live.join(', ')}]`,
    `边界 data messages = ${cellUnits === V ? standardTransfers : vppTransfers}`,
    cellUnits === V ? '每条 microbatch：F → loss@rank3 → B → grad@rank0' : `相对标准 = ${transferRatio} = ${(vppTransfers / standardTransfers).toFixed(2)}×（不是恰好 ${V}×）`,
    '斜纹格 = 因依赖未就绪产生的 bubble',
  ], 'metric-t', 22);
  return rulerY + 36;
}

put(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="100%" role="img"
  data-contract="P=4;m=8;v=2;N=4;standard_span=22;vpp_span=38;standard_bubble=6;vpp_bubble=6;standard_messages=48;vpp_messages=112;standard_live=4,3,2,1;vpp_live=11,9,7,5"
  aria-label="P=4 m=8 的无流水、普通 1F1B 与 VPP 调度。每个 rank 都可沿 microbatch 0 从 forward、末级 loss、backward 追到首级 gradient；图中同时给出气泡、激活记录和边界消息增量。">`);
put(`<style>
 text{font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif}.card{fill:#fff;stroke:#dfe3e8}.panel{fill:#fafbfc;stroke:#dfe3e8}.title{font-size:19px;font-weight:700;fill:#202630}.sub{font-size:11px;fill:#737b87}.pt{font-size:14px;font-weight:700;fill:#303844}.lane{font-size:10px;fill:#596270;text-anchor:end}.cell{font-size:9px;font-weight:700;fill:#fff;text-anchor:middle}.f{fill:#2563eb}.f-light{fill:#7ca8ef}.b{fill:#c3651f}.b-light{fill:#df9c65}.focus{stroke:#172033;stroke-width:1.7}.bubble{fill:url(#hatch);stroke:#d6d9de}.ruler{fill:none;stroke:#9ba3ad}.ruler-t{font-size:10px;fill:#68717d;text-anchor:middle}.metric{fill:#fff;stroke:#b7bec8}.metric-t{font-size:11px;fill:#47515e}.tag{font-size:11px;font-weight:700;fill:#173f87}.footer{font-size:11px;fill:#606a76}
</style>`);
put(`<defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" fill="#f4f5f6"/><rect width="2" height="6" fill="#dfe2e6"/></pattern></defs>`);
put(`<rect class="card" x=".5" y=".5" width="${W - 1}" height="${H - 1}" rx="14"/>`);
put(`<text class="title" x="28" y="38">同一训练闭环：no-pipeline、non-interleaved 1F1B、VPP/interleaved</text>`);
put(`<text class="sub" x="28" y="60">蓝 = forward，橙 = backward；小写 = chunk 0，大写 = chunk 1；黑边沿 microbatch 0。下两图共用真实时间比例：标准一格 = t_f，VPP 一格 = t_f/2。</text>`);

put(`<rect class="panel" x="20" y="80" width="1440" height="108" rx="11"/>`);
put(`<text class="pt" x="38" y="108">⓪ PP=1 / no-pipeline 控制流（m=8）</text>`);
put(`<text class="sub" x="38" y="130">控制流参考，不与下方 t_f 共用壁钟：F0 → B0 → F1 → B1 → …；每个 F 在整模型末端产生 loss，每个 B 再产生首端 gradient；stage P2P = 0。</text>`);
for (let mb = 0; mb < M; mb++) {
  const x = 38 + mb * 128;
  put(`<rect class="f" x="${x}" y="146" width="51" height="24" rx="4"/><text class="cell" x="${x + 25.5}" y="162">F${mb}</text>`);
  put(`<rect class="b" x="${x + 55}" y="146" width="51" height="24" rx="4"/><text class="cell" x="${x + 80.5}" y="162">B${mb}</text>`);
}
put(`<text class="tag" x="1080" y="162">峰值：当前 microbatch 1 份 forward state</text>`);

put(`<rect class="panel" x="20" y="204" width="1440" height="248" rx="11"/>`);
schedulePanel(232, '① 普通 non-interleaved 1F1B', 'P=4,m=8；一格 = 一个 physical stage 的 t_f；warmup / steady / cooldown 来自离散依赖解算。', standard, V, standardLive,
  `makespan = ${standard.span} t_f；bubble/compute = ${standard.bubble}/${standard.ops} = ${(100 * standard.bubble / standard.ops).toFixed(1)}%`);

put(`<rect class="panel" x="20" y="470" width="1440" height="248" rx="11"/>`);
schedulePanel(498, '② VPP / interleaved 1F1B', 'P=4,m=8,v=2,N=4；f/b=chunk0，F/B=chunk1；一格 = t_f/v。', vpp, 1, vppLive,
  `makespan = ${vpp.span}/${V} = ${vpp.span / V} t_f；bubble/compute = ${vpp.bubble}/${vpp.ops} = ${(100 * vpp.bubble / vpp.ops).toFixed(2)}%`);

put(`<text class="footer" x="28" y="754">量化结论：等时 chunk 模型中 22t_f → 19t_f；bubble/compute 从 3/8 → 3/16。消息按 logical stage boundary 计，VPP 是 112/48=7/3×；真实壁钟还取决于层异构、网络和 overlap。</text>`);
put(`<text class="footer" x="28" y="776">激活数字是“已 forward、尚未 backward”的记录数，不等价于字节；chunk 大小、重计算、offload 和输出伪释放会改变字节账本。</text>`);
put(`<text class="footer" x="28" y="812">决策线：显存/通信受限且 m 足够时先用普通 1F1B；bubble 主导且能承担 7/3× 边界消息与更多 live record 时再用 VPP。</text>`);
put(`</svg>`);

console.log(out.join('\n'));
