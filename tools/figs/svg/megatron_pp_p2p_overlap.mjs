// VPP steady 的同步 P2P 与 independent isend/irecv overlap；时间为显式示意参数。
// Usage: node tools/figs/svg/megatron_pp_p2p_overlap.mjs > .../assets/megatron_pp_p2p_overlap.svg

const TF = 2, TB = 2, TP = 0.75, STEPS = 2;
const compute = STEPS * (TF + TB);
const syncSpan = compute + STEPS * 2 * TP;
const asyncCompletion = compute + TP; // 最后一次 send/recv 仍需 drain。
const saved = syncSpan - asyncCompletion;
const W = 1440, H = 820, X = 160, U = 92;
const out = [];
const put = (s) => out.push(s);
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

function bar(x, y, duration, cls, label) {
  put(`<rect class="${cls}" x="${x}" y="${y}" width="${duration * U - 3}" height="34" rx="5"/>`);
  put(`<text class="bar" x="${x + duration * U / 2}" y="${y + 22}">${esc(label)}</text>`);
}

function ruler(y, span, label) {
  put(`<path class="rule" d="M ${X} ${y} H ${X + span * U} M ${X} ${y - 5} V ${y + 5} M ${X + span * U} ${y - 5} V ${y + 5}"/>`);
  put(`<text class="small" x="${X + span * U / 2}" y="${y + 18}" text-anchor="middle">${esc(label)}</text>`);
}

put(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="100%" role="img"
 data-contract="steps=2;tf=2;tb=2;tp2p=0.75;sync_span=11;async_compute=8;async_completion=8.75;saved=2.25;hyper_hidden=hidden_size*num_residual_streams;hyper_scope=intermediate_pp_boundary;hyper_recv_gate=physical_pp_rank_gt_0;hyper_send_gate=physical_pp_rank_lt_last;hyper_endpoints=hidden_size;hyper_flexible_vpp=TODO"
 aria-label="两个 VPP steady step 的同步与异步 P2P 时间线，以及 Hyper Connections 在固定 shape 中间流水边界扩展 hidden 维、端点保持 hidden_size 和 flexible VPP TODO。异步只缩短暴露通信，最后请求仍在 8.75t 才完成。">`);
put(`<style>text{font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif}.card{fill:#fff;stroke:#dfe3e8}.panel{fill:#fafbfc;stroke:#dfe3e8}.title{font-size:19px;font-weight:700;fill:#202630}.sub{font-size:11px;fill:#737b87}.pt{font-size:14px;font-weight:700;fill:#303844}.lane{font-size:11px;fill:#596270;text-anchor:end}.f{fill:#2563eb}.b{fill:#c3651f}.comm{fill:#e9f0fd;stroke:#2563eb;stroke-width:1.4}.stall{fill:url(#hatch);stroke:#c3651f}.bar{font-size:10px;font-weight:700;fill:#fff;text-anchor:middle}.comm+.bar,.stall+.bar{fill:#46505d}.rule{fill:none;stroke:#9ba3ad}.small{font-size:10.5px;fill:#616b78}.call{font-size:10.5px;fill:#173f87}.cost{font-size:10.5px;fill:#8a4a11}.note{fill:#fff;stroke:#b7bec8}.note-t{font-size:11px;fill:#47515e}.shape-box{fill:#fff;stroke:#9ba3ad}.shape-mid{fill:#e9f0fd;stroke:#2563eb;stroke-width:1.4}.shape-title{font-size:11px;font-weight:700;fill:#303844;text-anchor:middle}.shape-value{font-size:11px;fill:#173f87;text-anchor:middle}.shape-gate{font-size:10.5px;fill:#47515e;text-anchor:middle}.arrow{font-size:22px;font-weight:700;fill:#c3651f;text-anchor:middle}.todo{font-size:10.5px;fill:#8a4a11}</style>`);
put(`<defs><pattern id="hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="7" height="7" fill="#fbf0e7"/><rect width="2" height="7" fill="#e7bf9e"/></pattern></defs>`);
put(`<rect class="card" x=".5" y=".5" width="${W - 1}" height="${H - 1}" rx="14"/>`);
put(`<text class="title" x="28" y="38">overlap_p2p_comm：post request 之后仍要在数据/存储边界 wait</text>`);
put(`<text class="sub" x="28" y="60">同一 2 个 steady step；参数 t_f=t_b=${TF}t、t_p2p=${TP}t 仅作图，不是实测。蓝/橙为 F/B，浅蓝为 in-flight P2P，斜纹为暴露等待。</text>`);

put(`<rect class="panel" x="20" y="82" width="1400" height="190" rx="11"/>`);
put(`<text class="pt" x="40" y="112">① batch_isend_irecv（wait_on_reqs 强制为 true）：compute 与 P2P 串行</text>`);
let t = 0;
put(`<text class="lane" x="${X - 14}" y="166">compute / wait</text>`);
for (let i = 0; i < STEPS; i++) {
  bar(X + t * U, 142, TF, 'f', `F${i}`); t += TF;
  bar(X + t * U, 142, TP, 'stall', 'wait A'); t += TP;
  bar(X + t * U, 142, TB, 'b', `B${i}`); t += TB;
  bar(X + t * U, 142, TP, 'stall', 'wait G'); t += TP;
}
ruler(218, syncSpan, `同步完成 = ${syncSpan.toFixed(2)}t`);
put(`<text class="cost" x="1210" y="166">4 次 × ${TP}t 暴露</text>`);

put(`<rect class="panel" x="20" y="290" width="1400" height="250" rx="11"/>`);
put(`<text class="pt" x="40" y="320">② independent isend/irecv（wait_on_reqs=false）：请求与下一个可独立 chunk 计算重叠</text>`);
put(`<text class="lane" x="${X - 14}" y="376">compute</text>`);
t = 0;
for (let i = 0; i < STEPS; i++) {
  bar(X + t * U, 352, TF, 'f', `F${i}`); t += TF;
  bar(X + t * U, 352, TB, 'b', `B${i}`); t += TB;
}
put(`<text class="lane" x="${X - 14}" y="428">comm</text>`);
for (let i = 0; i < STEPS; i++) {
  const base = i * (TF + TB);
  bar(X + (base + TF) * U, 404, TP, 'comm', 'A req');
  bar(X + (base + TF + TB) * U, 404, TP, 'comm', 'G req');
}
put(`<path class="rule" d="M ${X + TF * U} 393 V 447 M ${X + (TF + TB) * U} 393 V 447"/>`);
put(`<text class="call" x="${X + TF * U + 5}" y="460">post F：isend activation / prefetch recv</text>`);
put(`<text class="call" x="${X + (TF + TB) * U + 5}" y="480">post B：isend gradient / prefetch recv</text>`);
ruler(506, asyncCompletion, `compute 区间 = ${compute.toFixed(2)}t；最后 drain 后完成 = ${asyncCompletion.toFixed(2)}t`);
put(`<text class="call" x="1010" y="376">recv：pre-forward / pre-backward 消费前 wait</text>`);
put(`<text class="call" x="1010" y="398">send：源 storage 复用/伪释放前 wait</text>`);
put(`<text class="cost" x="1010" y="420">cleanup：handle 队列必须为空</text>`);

put(`<rect class="note" x="20" y="560" width="1400" height="62" rx="10"/>`);
put(`<text class="note-t" x="40" y="585">本示例省 ${saved.toFixed(2)}t（${(100 * saved / syncSpan).toFixed(1)}%），不是从 ${syncSpan}t 直接到 ${compute}t：最后一个 ${TP}t request 仍要完成。没有独立 chunk、链路争用或 recv 提前被消费时，wait 会重新暴露。</text>`);
put(`<text class="note-t" x="40" y="606">overlap_p2p_comm_warmup_flush 只把同一生命周期延伸到 warmup/flush；要求 overlap_p2p_comm=true 且 batch_p2p_comm=false。</text>`);

put(`<rect class="panel" x="20" y="640" width="1400" height="158" rx="11"/>`);
put(`<text class="pt" x="40" y="669">③ get_tensor_shapes：Hyper Connections 的 fixed-shape PP boundary replay</text>`);
put(`<rect class="shape-box" x="45" y="686" width="270" height="62" rx="8"/>`);
put(`<text class="shape-title" x="180" y="708">first-stage input</text>`);
put(`<text class="shape-value" x="180" y="731">[S, B, hidden_size]</text>`);
put(`<text class="arrow" x="340" y="726">→</text>`);
put(`<rect class="shape-mid" x="365" y="686" width="710" height="62" rx="8"/>`);
put(`<text class="shape-title" x="720" y="705">intermediate PP send / recv</text>`);
put(`<text class="shape-value" x="720" y="725">[S, B, hidden_size * num_residual_streams]</text>`);
put(`<text class="shape-gate" x="720" y="743">${esc('recv：physical PP rank > 0 · send：physical PP rank < PP last')}</text>`);
put(`<text class="arrow" x="1100" y="726">→</text>`);
put(`<rect class="shape-box" x="1125" y="686" width="270" height="62" rx="8"/>`);
put(`<text class="shape-title" x="1260" y="708">last-stage output</text>`);
put(`<text class="shape-value" x="1260" y="731">[S, B, hidden_size]</text>`);
put(`<text class="todo" x="45" y="779">flexible VPP TODO：两类现有 layout 有 send/recv 一致性测试；helper 尚未承诺任意 flexible-VPP 排布。</text>`);
put(`</svg>`);
console.log(out.join('\n'));
