// P=4,m=8,VPP=2 的 combined-1F1B 宿主时序，以及层内 compute/EP A2A 共调度。
// Usage: node tools/figs/svg/megatron_combined_1f1b.mjs > .../assets/megatron_pp_combined_1f1b.svg

import { seqVpp, simulate } from './lib/megatron_pp_sim.mjs';

const P = 4, M = 8, V = 2, N = 4, EXTRA = 1;
const sim = simulate(seqVpp({ pp: P, m: M, vp: V, N, extraWarmup: EXTRA }), { pp: P, vp: V });
function peakLive(row) {
  let live = 0, peak = 0;
  for (const op of row) { if (op) { live += op.f ? 1 : -1; peak = Math.max(peak, live); } }
  return peak;
}
const live = sim.rows.map(peakLive);
const messages = 2 * M * (P * V - 1);
const W = 1560, H = 1030, out = [];
const put = (s) => out.push(s);
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
function lines(x, y, xs, cls = 'sm', gap = 16, anchor = 'start') {
  put(`<text class="${cls}" x="${x}" y="${y}" text-anchor="${anchor}">${xs.map((v, i) => `<tspan x="${x}" dy="${i ? gap : 0}">${esc(v)}</tspan>`).join('')}</text>`);
}
function box(x, y, w, h, title, body = [], cls = 'neutral') {
  put(`<rect class="${cls}" x="${x}" y="${y}" width="${w}" height="${h}" rx="9"/>`);
  put(`<text class="bt" x="${x + 13}" y="${y + 23}">${esc(title)}</text>`);
  lines(x + 13, y + 44, body, 'sm', 16);
}
function arrow(x1, y1, x2, y2, cls = 'arr') { put(`<path class="${cls}" d="M ${x1} ${y1} H ${x2}"/>`); }

put(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="100%" role="img"
 data-contract="P=4;m=8;v=2;N=4;extra_warmup=1;span=38;bubble=6;messages=112;live=12,10,8,6"
 aria-label="combined-1F1B 在 P=4 m=8 VPP=2 上额外 warmup 一格，峰值 live record 增加到 12 10 8 6；层内将相邻 microbatch 的 backward 和 forward 拆成 compute 与 EP A2A 节点，但 payload 不消失。">`);
put(`<style>text{font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif}.card{fill:#fff;stroke:#dfe3e8}.panel{fill:#fafbfc;stroke:#dfe3e8}.title{font-size:19px;font-weight:700;fill:#202630}.sub{font-size:11px;fill:#737b87}.pt{font-size:14px;font-weight:700;fill:#303844}.neutral{fill:#fff;stroke:#b6bdc7}.blue{fill:#eaf1fd;stroke:#2563eb;stroke-width:1.5}.orange{fill:#fcf0e6;stroke:#c3651f;stroke-width:1.5}.f{fill:#2563eb}.fl{fill:#7ca8ef}.b{fill:#c3651f}.bl{fill:#df9c65}.bubble{fill:url(#hatch);stroke:#d5d9de}.cell{font-size:8.5px;font-weight:700;fill:#fff;text-anchor:middle}.bt{font-size:12px;font-weight:700;fill:#303844}.sm{font-size:10.5px;fill:#5f6976}.lane{font-size:10.5px;fill:#596270;text-anchor:end}.arr{fill:none;stroke:#2563eb;stroke-width:1.8;marker-end:url(#ab)}.dep{fill:none;stroke:#8c949f;stroke-width:1.4;stroke-dasharray:5 4;marker-end:url(#ag)}.cost{font-size:10.5px;font-weight:700;fill:#8a4a11}.foot{font-size:10.5px;fill:#5f6976}</style>`);
put(`<defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" fill="#f4f5f6"/><rect width="2" height="6" fill="#dfe2e6"/></pattern><marker id="ab" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="#2563eb"/></marker><marker id="ag" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="#8c949f"/></marker></defs>`);
put(`<rect class="card" x=".5" y=".5" width="${W - 1}" height="${H - 1}" rx="14"/>`);
put(`<text class="title" x="28" y="38">combined-1F1B：改变 EP A2A 的暴露窗口，不改变 PP 拓扑或 payload</text>`);
put(`<text class="sub" x="28" y="60">统一示例 P=4,m=8,v=2,N=4；蓝=forward/compute，橙=backward/EP communication。上图是可计数 host schedule，下图是源码 node 顺序。</text>`);

put(`<rect class="panel" x="20" y="80" width="1520" height="286" rx="11"/>`);
put(`<text class="pt" x="42" y="110">① VPP host schedule：combined 令每 rank extra warmup = 1</text>`);
const X = 104, U = 20, ROW_H = 25;
sim.rows.forEach((row, rank) => {
  const y = 132 + rank * 31;
  put(`<text class="lane" x="${X - 12}" y="${y + 17}">rank ${rank}</text>`);
  row.forEach((op, t) => {
    const x = X + t * U;
    if (!op) put(`<rect class="bubble" x="${x + 1}" y="${y}" width="18" height="${ROW_H}" rx="3"/>`);
    else {
      const cls = op.f ? (op.c ? 'fl' : 'f') : (op.c ? 'bl' : 'b');
      const tag = `${op.c ? (op.f ? 'F' : 'B') : (op.f ? 'f' : 'b')}${op.mb}`;
      put(`<rect class="${cls}" x="${x + 1}" y="${y}" width="18" height="${ROW_H}" rx="3"/><text class="cell" x="${x + 10}" y="${y + 17}">${tag}</text>`);
    }
  });
});
box(900, 128, 610, 128, '由模拟计算的账本', [
  `warmup / rank = [${seqVpp({ pp: P, m: M, vp: V, N, extraWarmup: EXTRA }).map((r) => r.warmup).join(', ')}]（普通 VPP 为 [10, 8, 6, 4]）`,
  `makespan = ${sim.span} chunk-op = ${sim.span / V}t_f；bubble = ${sim.bubble}/${sim.ops}，与普通 VPP 相同`,
  `峰值 live forward record = [${live.join(', ')}]，比普通 VPP 每 rank +1`,
  `PP boundary data messages = ${messages}，combined 不会消除这些字节`,
], 'orange');
put(`<text class="foot" x="42" y="282">外层配对（m=8）：Phase 0：F(m0) → Phase 1：B(m0) ∥ F(m1) → Phase 2 … → Phase 4：B(m3) ∥ F(m4) → … → Phase 8：B(m7)。</text>`);
put(`<text class="foot" x="42" y="304">每一对仍沿 PP lane 完成：new F → sink loss；old B 从 loss-node gradient 起步 → 向 prev stage handoff input gradient。</text>`);
put(`<text class="cost" x="42" y="340">额外 warmup = 1 的付款点是另一份可配对 forward state；它只移动时序，不是固定加速比。</text>`);

put(`<rect class="panel" x="20" y="384" width="1520" height="376" rx="11"/>`);
put(`<text class="pt" x="42" y="414">② TransformerLayerSchedulePlan.run：同一对 B(m0) ∥ F(m1) 的层内双 stream</text>`);
put(`<text class="sub" x="42" y="434">条宽只表示节点窗，不表示实测时长；依赖 event 决定某段 A2A 能否与不消费其结果的 compute 同时推进。</text>`);
const SX = 172, SLOT = 246, GAP = 12;
const slots = [
  { c: ['combine_bwd(m0)', 'EP A2A'], f: ['attn_fwd(m1)', 'attention/MLP compute'] },
  { c: ['dispatch_fwd(m1)', 'EP A2A'], f: ['mlp_bwd(m0)', 'attention/MLP compute'] },
  { c: ['dispatch_bwd(m0)', 'EP A2A'], f: ['mlp_bwd_dw(m0)', 'wgrad compute'] },
  { c: ['event handoff', 'dependency only'], f: ['mlp_fwd(m1)', 'attention/MLP compute'] },
  { c: ['combine_fwd(m1)', 'EP A2A'], f: ['attn_bwd(m0)', 'attention/MLP compute'] },
];
put(`<text class="lane" x="${SX - 14}" y="500">comm stream</text><text class="lane" x="${SX - 14}" y="582">compute stream</text>`);
slots.forEach((slot, i) => {
  const x = SX + i * (SLOT + GAP);
  box(x, 464, SLOT, 58, slot.c[0], [slot.c[1]], i === 3 ? 'neutral' : 'orange');
  box(x, 546, SLOT, 58, slot.f[0], [slot.f[1]], 'blue');
  if (i < slots.length - 1) { arrow(x + SLOT, 493, x + SLOT + GAP, 493, 'dep'); arrow(x + SLOT, 575, x + SLOT + GAP, 575, 'dep'); }
});
box(42, 632, 450, 94, 'forward 交接', ['dispatch_fwd → expert MLP → combine_fwd', '最终得到 m1 layer output，再沿 PP send_forward'], 'blue');
box(554, 632, 450, 94, 'backward 交接', ['combine_bwd → expert dgrad/wgrad → dispatch_bwd', '最终得到 m0 layer input-grad，再沿 PP send_backward'], 'orange');
box(1066, 632, 452, 94, '没有消失的成本', ['A2A payload、event、两个 plan state 仍存在', 'exposed = max(0, A2A − independent compute)'], 'neutral');
put(`<text class="foot" x="42" y="746">` + esc('delay_wgrad_compute 可推迟 wgrad node；ep_overlap_early_attn_memory_release 提前 attention backward，却可能重新暴露 combine_fwd / dispatch_bwd。') + `</text>`);

put(`<rect class="panel" x="20" y="778" width="1520" height="214" rx="11"/>`);
put(`<text class="pt" x="42" y="808">③ guards 与选择线</text>`);
box(42, 828, 350, 122, '入口', ['no-pipeline 或 VPP/interleaved 宿主', 'forward_only 不进入 combined', '最终 unwrap 必须是 GPTModel'], 'blue');
box(414, 828, 350, 122, '计划契约', ['forward 返回 AbstractSchedulePlan', 'output 暂存 plan + loss_func', '对应 backward 必须消费同一 plan'], 'neutral');
box(786, 828, 350, 122, '拒绝组合', ['checkpoint_activations_microbatch = None', 'VPP + Megatron-FSDP 未支持', '部分 full-recompute/FSDP hooks 未支持'], 'orange');
box(1158, 828, 360, 122, '决策', ['只有 EP A2A 暴露且存在独立 compute 才尝试', 'profiler 验证剩余 wait、SM/链路竞争与显存', 'MoE 数学与 A2A ownership 在 EP 页'], 'neutral');
put(`<text class="foot" x="42" y="1010">陌生读者线：先看上图确认额外存活记录与 PP 消息没少，再沿 comm/compute 两 lane 找真正无依赖的覆盖窗；若 A2A 长于可独立计算，尾部仍在关键路径。</text>`);
put(`</svg>`);
console.log(out.join('\n'));
