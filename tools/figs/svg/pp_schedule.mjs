// 生成「GPipe vs 1F1B：同样的气泡，不同的显存」配图（SVG）。
//
// 与 .html2md/gen_pp_fig.mjs 同源：格子来自 GPipe / 1F1B 的离散事件仿真，
// 所以**图不可能和调度本身对不上**——这是「图也要 source-faithful」的做法：
// 图上的 makespan、在途激活份数都是算出来的，不是手写的。
//
// 产物是**外部 .svg 文件**，页面用标准图片语法引用：
//     ![流水线调度：同样的气泡，不同的显存](assets/pp_schedule.svg)
// Obsidian 的 ![[...]] 嵌入语法同样可用。两种写法都已在 docs-site(Quartz v5) 实测通过：
// .svg 被原样拷进产物(MD5 一致)、渲染为 <img>；Obsidian 按普通图片显示，
// 编辑态和阅读态都看得见图，md 源码只占一行。
//
// 为什么不内联进 .md：实测(2026-08-29)两头都不合适。
//   - docs-site：Quartz 的 remark→hast-util-to-jsx-runtime 会打坏 SVG——<svg> 被立刻
//     自闭合、驼峰属性被小写化(patternTransform→patterntransform，而 SVG 属性大小写敏感)、
//     自闭合标签被当成嵌套(<rect/><rect/> → <rect><rect></rect></rect>)。
//   - Obsidian：阅读态能渲，但编辑态是一堵几十 KB 的文本墙，「在 md 里看图」反而变差。
// 走外链则由浏览器把 .svg 当独立 XML 文档解析：驼峰属性、<defs>/<pattern>、<style>
// 全部正常，且 <style> 天然隔离在该文档内，不会泄漏成全站 CSS。
//
// 视觉规范见 skills/drawing-wiki-figures/SKILL.md：默认中性，强调色只标核心贡献与代价。
//
// 用法: node tools/figs/svg/pp_schedule.mjs [P] [m] > wiki/<域>/assets/pp_schedule.svg

const P = Number(process.argv[2] || 4);
const m = Number(process.argv[3] || 8);

/* ---------- 调度序列 ---------- */
function orders1f1b() {
  const S = [];
  for (let s = 0; s < P; s++) {
    const w = P - 1 - s, seq = [];
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
  for (let s = 0; s < P; s++) {
    const seq = [];
    for (let i = 1; i <= m; i++) seq.push(['F', i]);
    for (let i = 1; i <= m; i++) seq.push(['B', i]);
    S.push(seq);
  }
  return S;
}

/* ---------- 离散事件仿真（每个 F/B 占 1 个时间单位） ---------- */
function sim(S) {
  const end = {}, start = {}, K = (t, i, s) => `${t}${i}@${s}`;
  const ptr = Array(P).fill(0), free = Array(P).fill(0);
  let rem = P * 2 * m, guard = 0;
  while (rem > 0 && guard++ < 100000) {
    let progressed = false;
    for (let s = 0; s < P; s++) {
      const p = ptr[s];
      if (p >= S[s].length) continue;
      const [t, i] = S[s][p];
      let dep = free[s];
      if (t === 'F') {
        if (s > 0) { const k = K('F', i, s - 1); if (!(k in end)) continue; dep = Math.max(dep, end[k]); }
      } else if (s < P - 1) {
        const k = K('B', i, s + 1); if (!(k in end)) continue; dep = Math.max(dep, end[k]);
      } else {
        const k = K('F', i, s); if (!(k in end)) continue; dep = Math.max(dep, end[k]);
      }
      start[K(t, i, s)] = dep; end[K(t, i, s)] = dep + 1;
      free[s] = dep + 1; ptr[s]++; rem--; progressed = true;
    }
    if (!progressed) break;
  }
  return { start, end };
}
function grid(S) {
  const res = sim(S), span = Math.max(...Object.values(res.end)), rows = [];
  for (let s = 0; s < P; s++) {
    const row = Array(span).fill(null);
    for (const [t, i] of S[s]) row[res.start[`${t}${i}@${s}`]] = t + i;
    rows.push(row);
  }
  return { span, rows };
}
/* 在途激活份数的峰值：GPipe ∝ m、1F1B ∝ P。第 i 份 micro 的激活从它在
   stage0 前向算完起、到它在 stage0 反向算完为止都必须留着。 */
function peakLive(S) {
  const res = sim(S), span = Math.max(...Object.values(res.end));
  let peak = 0;
  for (let t = 0; t < span; t++) {
    let live = 0;
    for (let i = 1; i <= m; i++) {
      const a = res.end[`F${i}@0`], b = res.end[`B${i}@0`];
      if (a !== undefined && b !== undefined && t >= a - 1 && t < b) live++;
    }
    peak = Math.max(peak, live);
  }
  return peak;
}

const G = grid(ordersGpipe()), F = grid(orders1f1b());
const span = Math.max(G.span, F.span);
const ratio = `${P - 1}/${m + P - 1}`;
const ratioVal = ((P - 1) / (m + P - 1)).toFixed(2);
const peakG = peakLive(ordersGpipe()), peakF = peakLive(orders1f1b());

/* ---------- 尺寸 ---------- */
const PAD = 20, LAB = 62, CW = 30, CH = 24, GAP = 2, ROWG = 3;
const gridW = span * (CW + GAP) - GAP;
const W = PAD * 2 + LAB + gridW + 168;
const panelH = P * (CH + ROWG) - ROWG;

const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const out = [];
const put = (s) => out.push(s);

function panel(y, title, sub, g, noteLines, tone) {
  put(`  <text class="pt" x="${PAD}" y="${y}">${esc(title)}</text>`);
  put(`  <text class="ps" x="${PAD}" y="${y + 17}">${esc(sub)}</text>`);
  const top = y + 30;
  g.rows.forEach((row, s) => {
    const ry = top + s * (CH + ROWG);
    put(`  <text class="lab" x="${PAD + LAB - 8}" y="${ry + CH / 2 + 4}">stage ${s}</text>`);
    row.forEach((v, t) => {
      const x = PAD + LAB + t * (CW + GAP);
      if (v == null) {
        put(`  <rect class="bub" x="${x}" y="${ry}" width="${CW}" height="${CH}" rx="3"/>`);
      } else {
        put(`  <rect class="${v[0] === 'F' ? 'fwd' : 'bwd'}" x="${x}" y="${ry}" width="${CW}" height="${CH}" rx="3"/>`);
        put(`  <text class="cl" x="${x + CW / 2}" y="${ry + CH / 2 + 3.5}">${esc(v)}</text>`);
      }
    });
  });
  // makespan 尺规：两个 panel 等宽 ⇒ 气泡率相同，这正是本图要证的事
  const my = top + panelH + 9;
  put(`  <line class="rule" x1="${PAD + LAB}" y1="${my}" x2="${PAD + LAB + gridW}" y2="${my}"/>`);
  put(`  <text class="mk" x="${PAD + LAB + gridW / 2}" y="${my + 13}">makespan = ${g.span}</text>`);
  // 右侧代价 / 收益标注 —— 每张图只有这两个强调对象
  const bx = PAD + LAB + gridW + 14, by = top + panelH / 2 - 20;
  const k = tone === 'cost' ? 'cost' : 'gain';
  put(`  <rect class="box-${k}" x="${bx}" y="${by}" width="140" height="40" rx="8"/>`);
  noteLines.forEach((ln, i) =>
    put(`  <text class="txt-${k}" x="${bx + 10}" y="${by + 17 + i * 15}">${esc(ln)}</text>`));
  return my + 26;
}

let y = PAD + 58;
const body = [];
{
  const mark = out.length;
  y = panel(y, '① GPipe — 先全前向，再全反向',
    '气泡集中在左上填充 / 右下排空两个三角', G,
    ['峰值激活 ∝ m', `在途 ${peakG} 份`], 'cost');
  y += 10;
  y = panel(y, '② 1F1B — 预热后一前一后稳态',
    'micro 反向一到就释放其激活，中段满负荷 F/B 交替', F,
    ['峰值激活 ∝ P', `在途 ${peakF} 份`], 'gain');
  body.push(...out.splice(mark));
}
const H = y + 44;

const head = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="100%" role="img"
     aria-label="GPipe 与 1F1B 流水线调度对照：两条时间轴等长故气泡率相同，峰值激活显存 GPipe 正比 m、1F1B 正比 P">
  <style>
    text{font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif}
    .card{fill:#fff;stroke:#E4E7EC}
    .ti{font-size:15px;font-weight:700;fill:#1F2430}
    .su{font-size:11.5px;fill:#8A8F98}
    .pt{font-size:13px;font-weight:700;fill:#2A313B}
    .ps{font-size:11px;fill:#8A8F98}
    .lab{font-size:11px;fill:#5B6470;text-anchor:end}
    .cl{font-size:9.5px;font-weight:700;fill:#fff;text-anchor:middle}
    .fwd{fill:#5E97E4}                                    /* 前向：同族深档 */
    .bwd{fill:#9CC2F3}                                    /* 反向：同族浅档 */
    .bub{fill:url(#hatch);stroke:#E0DCD2}                 /* 气泡：中性斜纹 */
    .rule{stroke:#C7CCD3;stroke-dasharray:3 3}
    .mk{font-size:10px;fill:#9AA1AC;text-anchor:middle}
    .box-cost{fill:#FCF1E6;stroke:#C3651F;stroke-width:1.4}   /* acc2 = 代价 */
    .txt-cost{font-size:10.5px;font-weight:600;fill:#8A4A11}
    .box-gain{fill:#EAF1FD;stroke:#2563EB;stroke-width:1.4}   /* acc1 = 收益 */
    .txt-gain{font-size:10.5px;font-weight:600;fill:#173F87}
    .cap{font-size:11px;fill:#7A808A}
  </style>
  <defs>
    <pattern id="hatch" width="6" height="6" patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
      <rect width="6" height="6" fill="#F3F1EB"/>
      <rect width="3" height="6" fill="#E4E0D6"/>
    </pattern>
  </defs>
  <rect class="card" x="0.5" y="0.5" width="${W - 1}" height="${H - 1}" rx="14"/>
  <text class="ti" x="${PAD}" y="${PAD + 14}">流水线调度：同样的气泡，不同的显存</text>
  <text class="su" x="${PAD}" y="${PAD + 32}">P=${P} stage，m=${m} micro-batch，每格 1 时间单位 · 深蓝 Fᵢ 前向 · 浅蓝 Bᵢ 反向 · 斜纹 气泡</text>`;

const tail = `  <text class="cap" x="${PAD}" y="${y + 6}">两条时间轴等长（makespan=${span}）⇒ 气泡率相同 = (P−1)/(m+P−1) = ${ratio} ≈ ${ratioVal}。</text>
  <text class="cap" x="${PAD}" y="${y + 22}">GPipe→1F1B 省的是显存不是气泡：在途激活从 ${peakG} 份降到 ${peakF} 份。真正缩小气泡的是 Interleaved（(P−1)/(v·m+P−1)）与 Zero-Bubble。</text>
</svg>`;

console.log([head, ...body, tail].join('\n'));
