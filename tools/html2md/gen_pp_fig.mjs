// Generator for the PP schedule figure (dp_pipeline_parallel_fig1).
// Emits a P×time grid whose cells come directly from a discrete-event sim of
// GPipe and 1F1B, so the drawing cannot disagree with the schedule.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const P = 4, m = 8;

function orders1f1b() {
  const S = [];
  for (let s = 0; s < P; s++) {
    const w = P - 1 - s; const seq = [];
    for (let i = 1; i <= w; i++) seq.push(['F', i]);
    let f = w + 1, b = 1;
    for (let k = 0; k < m - w; k++) { seq.push(['F', f++]); seq.push(['B', b++]); }
    while (b <= m) seq.push(['B', b++]);
    S.push(seq);
  }
  return S;
}
function ordersGpipe() {
  const S = [];
  for (let s = 0; s < P; s++) { const seq = [];
    for (let i = 1; i <= m; i++) seq.push(['F', i]);
    for (let i = 1; i <= m; i++) seq.push(['B', i]);
    S.push(seq); }
  return S;
}
function sim(S) {
  const end = {}, start = {}; const K = (t, i, s) => t + i + '@' + s;
  const ptr = Array(P).fill(0), free = Array(P).fill(0); let rem = P * 2 * m, g = 0;
  while (rem > 0 && g++ < 100000) { let prog = false;
    for (let s = 0; s < P; s++) { const p = ptr[s]; if (p >= S[s].length) continue;
      const [t, i] = S[s][p]; let dep = free[s];
      if (t === 'F') { if (s > 0) { const k = K('F', i, s - 1); if (!(k in end)) continue; dep = Math.max(dep, end[k]); } }
      else { if (s < P - 1) { const k = K('B', i, s + 1); if (!(k in end)) continue; dep = Math.max(dep, end[k]); }
             else { const k = K('F', i, s); if (!(k in end)) continue; dep = Math.max(dep, end[k]); } }
      start[K(t, i, s)] = dep; end[K(t, i, s)] = dep + 1; free[s] = dep + 1; ptr[s]++; rem--; prog = true; }
    if (!prog) break; }
  return { start, end };
}
function cellsFor(S, res) {
  const ms = Math.max(...Object.values(res.end));
  const rows = [];
  for (let s = 0; s < P; s++) {
    const c = Array(ms).fill(null);
    for (const [t, i] of S[s]) c[res.start[t + i + '@' + s]] = t + i;
    rows.push(c);
  }
  return { ms, rows };
}

const g = cellsFor(ordersGpipe(), sim(ordersGpipe()));
const f = cellsFor(orders1f1b(), sim(orders1f1b()));
const ms = g.ms; // 22
const ratio = `${P - 1}/${m + P - 1}`;

const cellHtml = (v) => {
  if (v == null) return '<div class="pc pci"></div>';
  const cls = v[0] === 'F' ? 'pcf' : 'pcb';
  return `<div class="pc ${cls}">${v}</div>`;
};
const rowHtml = (label, cells) =>
  `      <div class="ppr"><div class="ppl">${label}</div><div class="ppc">${cells.map(cellHtml).join('')}</div></div>`;
const gridHtml = (rows) => rows.map((c, s) => rowHtml('Stage' + s, c)).join('\n');
const axis = Array.from({ length: ms }, (_, t) => `<div class="tt">${t}</div>`).join('');

const html = `<!doctype html><html lang="zh"><head><meta charset="utf-8">
<link rel="stylesheet" href="figstyle.css">
<style>
  .ppg{display:flex;flex-direction:column;gap:3px;margin:4px 0 2px}
  .ppr{display:flex;gap:6px;align-items:center}
  .ppl{width:52px;flex:0 0 auto;font-size:12px;font-weight:600;color:#3A434F;text-align:right}
  .ppc{display:flex;gap:2px;flex:1}
  .pc{flex:1;height:24px;border-radius:3px;display:flex;align-items:center;justify-content:center;
    font-size:10px;font-weight:700}
  .pcf{background:#6296D4;color:#fff}
  .pcb{background:#5BAE80;color:#fff}
  .pci{background:repeating-linear-gradient(45deg,#E4E0D6,#E4E0D6 4px,#F3F1EB 4px,#F3F1EB 8px);border:1px solid #EAE6DD}
  .panelttl{font-size:14.5px;font-weight:700;color:#1F2430;margin:14px 0 2px}
  .panelsub{font-size:12px;color:#8A8F98;margin-bottom:6px}
  .taxis{display:flex;gap:6px;align-items:center;margin-top:3px}
  .taxis .ppl{color:#A7AEB8;font-weight:400}
  .taxis .tt{flex:1;text-align:center;font-size:9px;color:#B0A597}
  .memnote{display:inline-block;margin-top:8px;font-size:12.5px;font-weight:600;padding:5px 12px;border-radius:8px}
  .mem-hi{background:#FBE9EC;color:#8A2A3C}
  .mem-lo{background:#E6F4EC;color:#1F6B41}
</style></head><body>

<div class="diagram" data-name="dp_pipeline_parallel_fig1">
  <div class="dg-title">流水线调度：同样的气泡，不同的显存（P=4, m=8）</div>
  <div class="dg-sub">蓝＝前向 Fᵢ，绿＝反向 Bᵢ，斜纹＝气泡（stage 空转）· 每格 1 时间单位 · 两图<strong>同宽（同 makespan=${ms}）→ 气泡率相同</strong> = (P−1)/(m+P−1) = ${ratio} ≈ ${(( P - 1) / (m + P - 1)).toFixed(2)}</div>

  <div style="width:1200px">
    <!-- ===== GPipe ===== -->
    <div class="panelttl">① GPipe：先全前向（F1…F8），再全反向（B1…B8）</div>
    <div class="panelsub">前向阶梯填充 + 反向阶梯排空；气泡集中在<strong>左上 / 右下两个三角</strong>。反向前须缓存<strong>全部 m=8 份</strong>前向激活</div>
    <div class="ppg">
${gridHtml(g.rows)}
    </div>
    <div><span class="memnote mem-hi">峰值激活显存 ∝ m（反向前攒满全部 8 份 micro 的激活）</span></div>

    <!-- ===== 1F1B ===== -->
    <div class="panelttl">② 1F1B：预热后进入一前一后稳态</div>
    <div class="panelsub">中段<strong>满负荷 F/B 交替、无空转</strong>（m≫P 时稳态很长）；每份 micro 反向一到就<strong>释放其激活</strong>。气泡＝左上预热阶梯 + 右下排空（含首级反向等下游回传的空档）</div>
    <div class="ppg">
${gridHtml(f.rows)}
    </div>
    <div class="taxis"><div class="ppl">时间 →</div><div class="ppc">${axis}</div></div>
    <div><span class="memnote mem-lo">峰值激活显存 ∝ P（在途仅 ~P=4 份，反向早释放）——同样气泡，显存降到约一半</span></div>

    <div class="legend" style="margin-top:16px">
      <span><span class="dot" style="background:#6296D4;border-color:#6296D4"></span>前向 Fᵢ</span>
      <span><span class="dot" style="background:#5BAE80;border-color:#5BAE80"></span>反向 Bᵢ</span>
      <span><span class="dot" style="background:repeating-linear-gradient(45deg,#E4E0D6,#E4E0D6 3px,#F3F1EB 3px,#F3F1EB 6px);border-color:#E0DDD6"></span>气泡（空转）</span>
    </div>
    <div class="note"><b>关键对照</b>：两图<b>同宽（makespan=${ms}）</b> ⇒ 气泡率相同 = (P−1)/(m+P−1) = ${ratio}，GPipe→1F1B <b>省的是显存而非气泡</b>——GPipe 攒满 m=8 份激活（∝m），1F1B 反向早、在途仅 ~P=4 份（∝P），显存约减半。<b>1F1B 气泡分布</b>：中段稳态满载 F/B 交替（m 越大这段越长、占比越高），空转集中在<b>预热（左上阶梯）</b>与<b>排空（右下）</b>；越靠前的 stage 排空时越要等下游把反向回传，故 Stage0 尾部 B6/B7/B8 被拉开。<b>Interleaved 1F1B</b>（每卡持 v 个虚拟 stage → 气泡 (P−1)/(v·m+P−1)）与 <b>Zero-Bubble/DualPipe</b>（用不在关键链上的 wgrad 填空转）才真正<b>缩小气泡本身</b>。</div>
  </div>
</div>

</body></html>
`;

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const out = path.resolve(__dirname, '..', 'figs', 'dp_pipeline_parallel.html');
fs.writeFileSync(out, html);
console.log('wrote', out, '| makespan', ms, '| GPipe rows', g.rows.length, '| 1F1B rows', f.rows.length);
