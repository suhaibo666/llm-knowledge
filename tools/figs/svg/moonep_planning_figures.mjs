// 教学数据；对应 MoonshotAI/MoonEP@33327eb9 的 tests/planning_reference.py::launch_planning_torch_reference
// （与 moonep/planning.py::PlanningKernel 同语义，由 tests/test_planning.py 对拍）。
// 不运行 MoonEP：这里逐步移植参考实现的 Step 1–5，argmax/argmin 取首个极值（= torch 与 kernel 的 *_min_idx）。
// planes() 在同一算例上推出 token 数据面（dispatch 写入、去重、combine 回拉）与权重面（预取推送、梯度回收），
// 规则分别对应 moonep/dispatch.py、dispatch_epilogue.py、combine_prologue.py、combine.py、prefetch.py、grad_reduce.py。
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

export const R = 4, EPN = 2, E = R * EPN, S = 4, K = 2, CAP = S * K;
// 每个 source rank 的 topk（S 个 token × K 个条目，token-major 展平）。
export const TOPK = [
  [0, 1, 0, 2, 0, 1, 2, 3],
  [0, 2, 0, 4, 1, 5, 2, 6],
  [0, 2, 0, 3, 1, 4, 2, 7],
  [0, 1, 0, 2, 3, 4, 5, 6],
];

const argmax = v => v.reduce((b, x, i) => (x > v[b] ? i : b), 0);
const argmin = v => v.reduce((b, x, i) => (x < v[b] ? i : b), 0);

export function replay(tokenPadding = 1) {
  const NvS = Math.ceil((CAP + (tokenPadding - 1) * 2 * EPN) / tokenPadding) * tokenPadding;
  const tpe = TOPK.map(row => Array.from({ length: E }, (_, e) => row.filter(x => x === e).length));
  // Step 1：全局直方图、home group 负载、balance
  const tpeCumsum = tpe.map((_, r) => Array.from({ length: E }, (_, e) => tpe.slice(0, r + 1).reduce((s, row) => s + row[e], 0)));
  const expertCount = tpeCumsum[R - 1];
  const groupTokens = Array.from({ length: R }, (_, h) => expertCount.slice(h * EPN, (h + 1) * EPN).reduce((a, b) => a + b, 0));
  const balance0 = groupTokens.map(g => g - CAP);
  // Step 2：贪心配对
  const balance = balance0.slice();
  const z = Array.from({ length: R }, () => Array(R).fill(0));
  const rounds = [];
  for (;;) {
    const h = argmax(balance), u = argmin(balance);
    if (balance[h] <= 0) break;
    const move = -balance[u];
    z[h][u] = move; balance[h] -= move; balance[u] = 0;
    rounds.push({ h, u, move, after: balance.slice() });
  }
  // Step 3：配额摊到具体专家
  const alloc = Array.from({ length: E }, (_, e) => Array.from({ length: R }, (_, d) => (d === Math.floor(e / EPN) ? expertCount[e] : 0)));
  const takes = [];
  for (let h = 0; h < R; h++) {
    const remaining = expertCount.slice(h * EPN, (h + 1) * EPN), quotas = z[h].slice();
    for (;;) {
      const d = argmax(quotas);
      if (quotas[d] <= 0) break;
      const le = argmax(remaining), e = h * EPN + le, take = Math.min(remaining[le], quotas[d]);
      alloc[e][d] += take; alloc[e][h] -= take; remaining[le] -= take; quotas[d] -= take;
      takes.push({ h, d, e, take });
    }
  }
  // Step 4：每个 dest rank 的 2·epn 个 VM 组
  const allocCumsum = alloc.map(row => row.map((_, d) => row.slice(0, d + 1).reduce((a, b) => a + b, 0)));
  const expertOff = Array.from({ length: R }, () => Array(E).fill(0));
  const ranks = [];
  for (let d = 0; d < R; d++) {
    const ls = d * EPN;
    const remote = [];
    for (let e = 0; e < E; e++) if (alloc[e][d] > 0 && !(e >= ls && e < ls + EPN)) remote.push(e);
    if (remote.length > EPN) throw new Error('remote experts exceed epn');
    const etc = Array.from({ length: EPN }, (_, b) => remote[b] ?? -1);
    const groups = [];
    let start = 0;
    for (let g = 0; g < 2 * EPN; g++) {
      const eid = g < EPN ? ls + g : etc[g - EPN];
      const cnt = eid >= 0 ? alloc[eid][d] : 0;
      const padded = cnt > 0 ? Math.ceil(cnt / tokenPadding) * tokenPadding : 0;
      if (cnt > 0) expertOff[d][eid] = start;
      groups.push({ g, eid, cnt, start, end: start + padded, slot: g >= EPN });
      start += padded;
    }
    ranks.push({ d, etc, groups, cu: groups.map(x => x.end), total: groups.reduce((s, x) => s + x.cnt, 0) });
  }
  // Step 5：每个条目的落点 dst，以及同 token 同 dest 的去重编码
  const dst = TOPK.map((row, src) => {
    const counter = Array(E).fill(0);
    return row.map(e => {
      const g = (src ? tpeCumsum[src - 1][e] : 0) + counter[e]++;
      const dest = allocCumsum[e].findIndex(c => c > g);
      const segPos = g - (dest ? allocCumsum[e][dest - 1] : 0);
      return { e, g, dest, row: expertOff[dest][e] + segPos, raw: dest * NvS + expertOff[dest][e] + segPos };
    });
  });
  const canon = dst.map(row => row.map((x, i) => {
    const tok = Math.floor(i / K) * K;
    const seen = row.slice(tok, i).some(y => y.dest === x.dest);
    return seen ? -x.raw - 1 : x.raw;
  }));
  const moved = rounds.reduce((s, r) => s + r.move, 0);
  const lowerBound = balance0.filter(b => b > 0).reduce((a, b) => a + b, 0);
  return { NvS, tpe, expertCount, groupTokens, balance0, rounds, z, takes, alloc, allocCumsum, ranks, dst, canon, moved, lowerBound };
}

export const fmt = a => '[' + a.join(', ') + ']';
const esc = s => String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;');

export function draw() {
  const r = replay(), W = 1120, H = 900;
  const out = [`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}"><title>MoonEP planning 五步算例</title><desc>R=4、epn=2、每 rank 容量 S·K=8：从全局直方图到每 rank 恰好 8 个 token 的 VM 布局与条目落点。</desc><style>text{font-family:Arial,'PingFang SC','Microsoft YaHei',sans-serif;fill:#2A313B;font-size:18px}.neutral{fill:#fff;stroke:#C7CCD3}.ghost{fill:#F7F6F3;stroke:#DDD9D2}.acc1{fill:#EAF1FD;stroke:#2563EB}.acc2{fill:#FCF1E6;stroke:#C3651F}.hot{fill:none;stroke:#C3651F;stroke-width:3.5}.sep{stroke:#2A313B;stroke-width:2.5}.cap{font-size:17px;fill:#6B7280}.heading{font-size:22px;font-weight:600}.title{font-size:25px;font-weight:600}.t1{fill:#2563EB;font-weight:600}.t2{fill:#C3651F;font-weight:600}.mono{font-family:Menlo,Consolas,monospace;font-size:17px}.small{font-size:12px}</style><rect width="${W}" height="${H}" fill="white"/>`];
  const text = (x, y, s, cls = '', anchor = 'start') => out.push(`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${esc(s)}</text>`);
  const rect = (x, y, w, h, cls, extra = '') => out.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" class="${cls}"${extra}/>`);
  text(30, 38, `MoonEP planning 算例：R=${R}、epn=${EPN}、E=${E}，每 rank 容量 CAP = S·K = ${CAP}`, 'title');

  // ① Step 1：home group 负载与 CAP
  text(30, 84, '① Step 1  home group 负载 vs CAP', 'heading');
  const unit = 15, base = 360, capY = base - CAP * unit;
  const labels = [];
  r.groupTokens.forEach((gt, h) => {
    const x = 120 + h * 108, w = 72;
    // 超出 CAP 的盈余段垫在底层（橙），专家分段只描边，标签最后画，避免被遮住
    if (gt > CAP) rect(x, base - gt * unit, w, (gt - CAP) * unit, 'acc2');
    else rect(x, capY, w, (CAP - gt) * unit, 'ghost', ' stroke-dasharray="5 4"');
    let top = base;
    for (let j = 0; j < EPN; j++) {
      const e = h * EPN + j, c = r.expertCount[e], y = top - c * unit;
      rect(x, y, w, c * unit, 'neutral', ' fill-opacity="0"');
      // 跨 CAP 线的分段把标签放到较大的一侧，避免与虚线重叠
      let [lo, hi] = [y, y + c * unit];
      if (lo < capY && capY < hi) [lo, hi] = capY - lo >= hi - capY ? [lo, capY] : [capY, hi];
      if (c * unit >= 20) labels.push([x + w / 2, (lo + hi) / 2 + 6, `e${e}·${c}`]);
      else labels.push([x + w / 2, (lo + hi) / 2 + 4, `e${e}·${c}`, 'small']);
      top = y;
    }
    const b = r.balance0[h];
    labels.push([x + w / 2, Math.min(top, capY) - 10, b > 0 ? `+${b}` : `${b}`, b > 0 ? 't2' : 't1']);
    labels.push([x + w / 2, base + 26, `rank ${h}`]);
    labels.push([x + w / 2, base + 48, `${gt}`, 'cap']);
  });
  out.push(`<line x1="30" y1="${capY}" x2="540" y2="${capY}" stroke="#2563EB" stroke-width="2" stroke-dasharray="8 5"/>`);
  labels.forEach(([x, y, s, cls = '']) => text(x, y, s, cls, 'middle'));
  text(30, capY - 8, `CAP=${CAP}`, 't1');
  text(30, 434, `Σ balance = ${r.balance0.reduce((a, b) => a + b, 0)}：全局条目数恰为 R·CAP`, 'cap');

  // ② Step 2：贪心轮次
  const tx = 580;
  text(tx, 84, '② Step 2  每轮把最空的 rank 一次填满', 'heading');
  const cols = [['轮', 50], ['h 最满', 90], ['u 最空', 90], ['move', 70], ['balance', 210]];
  let cx = tx;
  cols.forEach(([label, w]) => { rect(cx, 102, w, 38, 'ghost'); text(cx + 10, 128, label); cx += w; });
  const rows = [['初始', '', '', '', fmt(r.balance0)], ...r.rounds.map((x, i) => [i + 1, x.h, x.u, x.move, fmt(x.after)])];
  rows.forEach((vals, i) => {
    const y = 140 + i * 40;
    let xx = tx;
    // 发送量超过自身盈余、被翻成接收方的一轮
    const flips = i > 0 && r.rounds[i - 1].after[r.rounds[i - 1].h] < 0;
    vals.forEach((v, c) => { rect(xx, y, cols[c][1], 40, flips && c === 4 ? 'acc2' : 'neutral'); text(xx + 10, y + 27, v, c === 4 ? 'mono' : ''); xx += cols[c][1]; });
  });
  const flip = r.rounds.find(x => x.after[x.h] < 0);
  text(tx, 330, `第 ${r.rounds.indexOf(flip) + 1} 轮 rank ${flip.h} 发出 ${flip.move} > 盈余 ${r.balance0[flip.h]}：翻为 ${flip.after[flip.h]}，`, 't2');
  text(tx, 354, '下一轮改作接收方（发送方也可能接收）', 't2');
  text(tx, 388, 'u 被置 0 后不再变化 ⇒ 每个 rank 至多被选作', 't1');
  text(tx, 412, '接收方一次，只从一个 home group 收 token', 't1');
  text(tx, 434, `迁移 ${r.moved} 个，下界 Σ正 balance = ${r.lowerBound}`, 'cap');

  // ③④⑤：配额摊到专家、VM 布局、条目落点
  text(30, 490, '③ Step 3  配额摊到本地剩余最多的专家', 'heading');
  text(580, 490, '④ Step 4  每 rank 的 2·epn 个 VM 组', 'heading');
  text(30, 520, r.takes.map(t => `z[${t.h},${t.d}]=${t.take} → e${t.e}`).join('；'), 'mono');
  const gx = 120, cw = 58, ch = 44, gy0 = 566, rowGap = 60;
  text(gx, gy0 - 12, '← 8 个 token 槽（物理行号 0…7）→', 'cap');
  text(640, gy0 - 12, 'experts_to_copy', 'cap');
  text(840, gy0 - 12, 'cu_seqlens', 'cap');
  const hotCells = new Set(r.dst[1].filter(x => x.e === 0).map(x => `${x.dest}:${x.row}`));
  r.ranks.forEach((rk, d) => {
    const y = gy0 + d * rowGap;
    text(30, y + 29, `rank ${d}`);
    rk.groups.forEach(g => {
      for (let i = g.start; i < g.start + g.cnt; i++) {
        rect(gx + i * cw, y, cw, ch, g.slot ? 'acc1' : 'neutral');
        text(gx + i * cw + cw / 2, y + 29, `e${g.eid}`, '', 'middle');
      }
      if (g.cnt > 0 && g.start > 0) out.push(`<line x1="${gx + g.start * cw}" y1="${y - 4}" x2="${gx + g.start * cw}" y2="${y + ch + 4}" class="sep"/>`);
    });
    for (const key of hotCells) {
      const [dd, row] = key.split(':').map(Number);
      if (dd === d) rect(gx + row * cw + 2, y + 2, cw - 4, ch - 4, 'hot');
    }
    text(640, y + 29, fmt(rk.etc), 'mono');
    text(840, y + 29, fmt(rk.cu), 'mono');
  });
  rect(640, gy0 + 4 * rowGap - 2, 22, 18, 'neutral');
  text(670, gy0 + 4 * rowGap + 13, '本地专家组 g0、g1', 'cap');
  rect(850, gy0 + 4 * rowGap - 2, 22, 18, 'acc1');
  text(880, gy0 + 4 * rowGap + 13, '预取槽 g2、g3', 'cap');

  const [a, b] = r.dst[1].filter(x => x.e === 0);
  text(30, 830, `⑤ src1 的两个 e0 条目全局序号 ${a.g}、${b.g}，alloc_cumsum[e0]=${fmt(r.allocCumsum[0])}：`, 't2');
  text(30, 856, `   → rank ${a.dest} 行 ${a.row}，rank ${b.dest} 行 ${b.row}（dst = ${b.dest}·NvS + ${b.row} = ${b.raw}，NvS=${r.NvS}）；橙框即这两个落点`, 't2');
  text(30, 886, `每行恰好 ${CAP} 个 token；每行的远端专家 ≤ epn=${EPN} 且来自同一 home group，所以 epn 个预取槽就够`, 't1');
  out.push('</svg>');
  return out.join('\n');
}

// ---------------- 数据面与权重面 ----------------
export function planes() {
  const r = replay();
  const ents = Array.from({ length: R }, () => Array(R).fill(0));   // 条目数（含重复）
  const rows = Array.from({ length: R }, () => Array(R).fill(0));   // dispatch 真正写出的 hidden 行（dst ≥ 0）
  r.dst.forEach((arr, src) => arr.forEach((x, i) => {
    ents[src][x.dest]++;
    if (r.canon[src][i] >= 0) rows[src][x.dest]++;
  }));
  const colSum = m => Array.from({ length: R }, (_, d) => m.reduce((s, row) => s + row[d], 0));
  const written = colSum(rows);
  const viaNvlink = written.map((w, d) => w - rows[d][d]);           // src == dest 的行是本地写
  // rank 0 的 8 行：来源与到达方式
  const landed = [];
  r.dst.forEach((arr, src) => arr.forEach((x, i) => {
    if (x.dest !== 0) return;
    const tok = Math.floor(i / K), dup = r.canon[src][i] < 0;
    const primary = dup ? arr.find((y, j) => Math.floor(j / K) === tok && y.dest === 0 && r.canon[src][j] >= 0) : null;
    landed.push({ row: x.row, e: x.e, src, tok, k: i % K, dup, primaryRow: primary ? primary.row : null,
      kind: dup ? 'dup' : src === 0 ? 'local' : 'nvlink' });
  }));
  landed.sort((a, b) => a.row - b.row);
  // combine 回拉：每个代表行由它的 source rank 读回
  const pulls = [];
  for (const x of landed.filter(y => !y.dup)) {
    let p = pulls.find(q => q.src === x.src);
    if (!p) pulls.push(p = { src: x.src, rows: [] });
    p.rows.push(x.row);
  }
  // 权重面：experts_to_copy[d][b] = e 表示所有者 e/epn 把 e 推到 rank d 的槽 b
  const pushes = [];
  r.ranks.forEach(rk => rk.etc.forEach((e, b) => { if (e >= 0) pushes.push({ owner: Math.floor(e / EPN), e, local: e % EPN, dest: rk.d, slot: b }); }));
  return { r, ents, rows, written, viaNvlink, landed, pulls, pushes };
}

export function drawPlanes() {
  const p = planes(), r = p.r, W = 1120, H = 960;
  const out = [`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}"><title>MoonEP 数据面与权重面算例</title><desc>沿用 planning 算例：每对 rank 的条目数、rank 0 八行的来源与到达方式，以及专家权重的前向推送与梯度回收。</desc><style>text{font-family:Arial,'PingFang SC','Microsoft YaHei',sans-serif;fill:#2A313B;font-size:18px}.neutral{fill:#fff;stroke:#C7CCD3}.ghost{fill:#F7F6F3;stroke:#DDD9D2}.acc1{fill:#EAF1FD;stroke:#2563EB}.acc2{fill:#FCF1E6;stroke:#C3651F}.cap{font-size:17px;fill:#6B7280}.heading{font-size:22px;font-weight:600}.title{font-size:25px;font-weight:600}.t1{fill:#2563EB;font-weight:600}.t2{fill:#C3651F;font-weight:600}.mono{font-family:Menlo,Consolas,monospace;font-size:17px}.fwd{stroke:#2563EB;stroke-width:2.5;fill:none}.bwd{stroke:#C3651F;stroke-width:2;fill:none;stroke-dasharray:6 4}</style><defs><marker id="a1" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0 0L9 4.5L0 9" fill="#2563EB"/></marker><marker id="a2" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0 0L9 4.5L0 9" fill="#C3651F"/></marker></defs><rect width="${W}" height="${H}" fill="white"/>`];
  const text = (x, y, s, cls = '', anchor = 'start') => out.push(`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${esc(s)}</text>`);
  const rect = (x, y, w, h, cls, extra = '') => out.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" class="${cls}"${extra}/>`);
  text(30, 38, '同一算例的数据面：token 行与专家权重怎样在 4 个 rank 之间移动', 'title');

  // ① 每对 rank 的条目数
  text(30, 84, '① 每对 rank 的条目数（含重复）', 'heading');
  const mx = 110, my = 104, cw = 78, ch = 42;
  for (let d = 0; d < R; d++) { rect(mx + d * cw, my, cw, ch, 'ghost'); text(mx + d * cw + cw / 2, my + 28, `→ r${d}`, '', 'middle'); }
  rect(mx + R * cw, my, cw, ch, 'ghost'); text(mx + R * cw + cw / 2, my + 28, '发出', '', 'middle');
  p.ents.forEach((row, s) => {
    const y = my + ch * (s + 1);
    text(30, y + 28, `src${s}`);
    row.forEach((v, d) => { rect(mx + d * cw, y, cw, ch, s === d ? 'ghost' : 'neutral'); text(mx + d * cw + cw / 2, y + 28, v, '', 'middle'); });
    rect(mx + R * cw, y, cw, ch, 'neutral'); text(mx + R * cw + cw / 2, y + 28, row.reduce((a, b) => a + b, 0), '', 'middle');
  });
  const sy = my + ch * (R + 1);
  text(30, sy + 28, '接收');
  for (let d = 0; d < R; d++) { rect(mx + d * cw, sy, cw, ch, 'acc1'); text(mx + d * cw + cw / 2, sy + 28, p.ents.reduce((s, row) => s + row[d], 0), 't1', 'middle'); }
  text(30, sy + 78, `接收合计恒为 S·K=${CAP}：接收 buffer 按它的上界一次分配`, 't1');
  text(30, sy + 104, '单元格随路由变化：只存在于 GPU 上的 dst 里，host 不需要', 't2');
  text(30, sy + 140, `dispatch 写出的 hidden 行 ${p.written.join(' / ')}`, 'mono');
  text(30, sy + 166, `其中跨 NVLink 到达   ${p.viaNvlink.join(' / ')}`, 'mono');
  text(30, sy + 192, '灰格是 src = dest 的本地写', 'cap');

  // ② rank 0 的 8 行
  const rx = 580;
  text(rx, 84, '② rank 0 的 8 行：谁写、怎样到达', 'heading');
  const tag = { local: ['neutral', '本地写'], nvlink: ['acc1', 'NVLink 直写'], dup: ['acc2', ''] };
  p.landed.forEach((x, i) => {
    const y = 104 + i * 42;
    rect(rx, y, 110, 38, 'ghost'); text(rx + 10, y + 26, `行 ${x.row} · e${x.e}`);
    rect(rx + 110, y, 200, 38, 'neutral'); text(rx + 120, y + 26, `src${x.src} token ${x.tok}（k=${x.k}）`);
    const [cls, label] = tag[x.kind];
    rect(rx + 310, y, 190, 38, cls); text(rx + 320, y + 26, x.dup ? `epilogue 复制行 ${x.primaryRow}` : label, x.kind === 'dup' ? 't2' : x.kind === 'nvlink' ? 't1' : '');
  });
  const by = 104 + p.landed.length * 42 + 30;
  const dups = p.landed.filter(x => x.dup);
  text(rx, by, `combine 反向：prologue 把${dups.map(x => `行 ${x.row} 加进行 ${x.primaryRow}`).join('、')}，`, 't2');
  text(rx, by + 26, '再由各 source rank 只读回代表行：', 't2');
  p.pulls.forEach((q, i) => text(rx, by + 52 + i * 24, `src${q.src} ${q.src === 0 ? '本地' : '远端'}读行 ${q.rows.join('、')}`, 'mono'));

  // ③ 权重面
  const wy = 624;
  text(30, wy - 20, '③ 权重面：每 rank 的计算视图是 2·epn = 4 行（本地 g0、g1 + 预取槽 g2、g3）', 'heading');
  const gx = 110, gw = 92, gh = 40;
  ['g0', 'g1', 'g2 槽0', 'g3 槽1'].forEach((h, g) => text(gx + g * gw + gw / 2, wy + 20, h, 'cap', 'middle'));
  r.ranks.forEach((rk, d) => {
    const y = wy + 32 + d * 50;
    text(30, y + 27, `rank ${d}`);
    for (let g = 0; g < 2 * EPN; g++) {
      const e = g < EPN ? d * EPN + g : rk.etc[g - EPN];
      rect(gx + g * gw, y, gw, gh, g < EPN ? 'neutral' : e >= 0 ? 'acc1' : 'ghost');
      text(gx + g * gw + gw / 2, y + 27, e >= 0 ? `e${e}` : '空', e >= 0 ? '' : 'cap', 'middle');
    }
  });
  const lx = 520;
  p.pushes.forEach((q, i) => {
    const y = wy + 36 + i * 64;
    rect(lx, y, 190, 40, 'neutral'); text(lx + 10, y + 27, `rank ${q.owner} 本地 e${q.e}（g${q.local}）`);
    rect(lx + 380, y, 180, 40, 'acc1'); text(lx + 390, y + 27, `rank ${q.dest} 槽 ${q.slot}（g${EPN + q.slot}）`);
    out.push(`<path d="M${lx + 196} ${y + 14}H${lx + 372}" class="fwd" marker-end="url(#a1)"/>`);
    out.push(`<path d="M${lx + 374} ${y + 30}H${lx + 198}" class="bwd" marker-end="url(#a2)"/>`);
  });
  out.push(`<path d="M${lx} ${wy + 14}H${lx + 40}" class="fwd" marker-end="url(#a1)"/>`);
  text(lx + 48, wy + 20, '前向：读 1 次、推送', 't1');
  out.push(`<path d="M${lx + 330} ${wy + 14}H${lx + 290}" class="bwd" marker-end="url(#a2)"/>`);
  text(lx + 338, wy + 20, '反向：远端读回槽梯度', 't2');
  text(30, wy + 262, `前向：所有者按 experts_to_copy 把本地专家推到目标槽，一个 tile 读 1 次、扇出到全部目标；共 ${p.pushes.length} 份`, 't1');
  text(30, wy + 290, '反向：所有者远端读各槽梯度、加进本地参数梯度；barrier 后每个目标 rank 在本地清零自己被读过的槽', 't2');
  text(30, wy + 322, `每 rank 恰算 ${CAP} 个 token；不变的是 buffer 与视图形状，变化的只有 dst、cu_seqlens 与 experts_to_copy 的取值`, 'cap');
  out.push('</svg>');
  return out.join('\n');
}

export const FIGURES = {
  'moonep_planning_example.svg': draw,
  'moonep_dataplane_example.svg': drawPlanes,
};

const here = dirname(fileURLToPath(import.meta.url));
const defaultOutput = join(here, '..', '..', '..', 'wiki', '01_theory', '01_models', 'moonshot_kimi', 'assets');
if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const outputDir = process.argv[2] ? process.argv[2] : defaultOutput;
  mkdirSync(outputDir, { recursive: true });
  for (const [name, fn] of Object.entries(FIGURES)) writeFileSync(join(outputDir, name), `${fn()}\n`, 'utf8');
}
