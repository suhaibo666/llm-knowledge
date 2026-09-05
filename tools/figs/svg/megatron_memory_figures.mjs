// 图 1：细粒度激活 offload 在三条流上的时序 —— 组提交触发 D2H、反向组起点提前一组预取
//       H2D、warmup 后保留的 margin 组让最后两组根本不出 GPU；常驻字节随之变化。
// 图 2：paged stash 的分页缓冲、复制 kernel 的三路判定（CUDA 页 / 宿主页溢写 / overflow）、
//       以及 runner 的整步重跑。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
//
// 图 1 要讲清楚：22 号页 §2.3 的"子模块级换出"为什么能把 PCIe 拷贝藏在计算后面，
// 以及 post_warmup_callback 保留 margin 组的意义。算例：PP=1、一个 microbatch、3 层，
// 每层两个 offload 组 core_attn(96 MB)/expert_fc1(128 MB)，前向每组 10 单位、反向每组 20 单位，
// PCIe 16 MB/单位。三条泳道：compute / d2h_stream / h2d_stream，横轴时间。
//   - 前向：组提交（group_offload）时 d2h 等主流，再串行拷贝；
//   - warmup 后 margin = 组名去重数 = 2，最后两组（L3 的两组）不 offload；
//   - 反向：组 j 的 commit 反向先等它自己的 reload 事件；组 j 的 start 反向再预取 LIFO 的下一组
//     —— 预取距离正好等于 margin，所以本例零阻塞；
//   - 下方常驻字节条：无 offload 峰值 vs 有 offload 峰值；
//   - 右侧对照：margin=0 时最后两组在 tensor_pop 里同步拷回，暴露 14 单位。
// 强调色：acc1 = 被藏住的拷贝 / margin 组（收益），acc2 = 暴露的等待（代价）。
//
// 图 2 要讲清楚：22 号页 §2.4 的"留在 GPU 上的分页暂存"如何用 capture 迭代的平均 token 数
// 预定页数、复制 kernel 怎样按 free list 余量决定去向、以及 overflow 如何变成整步重跑。
// 算例：3 个 MoE 层、page=64、rank 容量 256 行、容量因子 1.6 → avg=160；三层同时在存 → 480；
// cuda factor 1.10 → 528 行 → 9 页；步 A 实际 [150,190,120] → 8 页够；
// 步 B 偏斜 [150,250,220] → 11 页：第三层缺页 → cpu factor 0 则 overflow → 重跑一次；
// cpu factor 0.5 → 240 行 → 4 宿主页 → 溢写成功、只记 host_spill。
// 强调色：acc1 = 命中 CUDA 页，acc2 = 溢写 / overflow / 重跑。
//
// 硬规矩：图上每个数字来自 simulateOffload() / solveStash()；每行文字过宽度守卫；
// 渲染后做图元级文字重叠断言。
//
// 用法：node tools/figs/svg/megatron_memory_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// ============================================================================
// 图 1 的离散事件仿真：复刻 fine_grained_activation_offload.py 的组提交 / 预取规则
// ============================================================================

export const OFFLOAD_CFG = Object.freeze({
  layers: 3,
  names: Object.freeze(['core_attn', 'expert_fc1']),
  bytesMB: Object.freeze({ core_attn: 96, expert_fc1: 128 }),
  fwdCompute: 10, // 每组前向计算，单位时间
  bwdCompute: 20, // 每组反向计算
  pcieMBPerUnit: 16, // PCIe 带宽
});

export function simulateOffload(cfg = OFFLOAD_CFG) {
  const groups = [];
  for (let l = 1; l <= cfg.layers; l += 1) {
    for (const name of cfg.names) groups.push({ id: `L${l}.${name}`, layer: l, name, mb: cfg.bytesMB[name] });
  }
  // post_warmup_callback：margin = get_max_deduplicated_groups() = 组名去重数；
  // 对每个名字，前向顺序里最后一个同名组 offload=False。
  const margin = new Set(groups.map((g) => g.name)).size;
  const lastByName = {};
  for (const g of groups) lastByName[g.name] = g;
  let left = margin;
  for (const name of Object.keys(lastByName)) {
    if (left > 0) {
      lastByName[name].offload = false;
      left -= 1;
    }
  }
  for (const g of groups) if (g.offload === undefined) g.offload = true;

  // 前向：compute 串行；提交时 d2h 等主流，再 FIFO 串行
  let t = 0;
  let d2hFree = 0;
  for (const g of groups) {
    g.fwd = [t, t + cfg.fwdCompute];
    t += cfg.fwdCompute;
    if (g.offload) {
      const start = Math.max(t, d2hFree);
      const dur = g.mb / cfg.pcieMBPerUnit;
      g.d2h = [start, start + dur];
      d2hFree = start + dur;
    }
  }
  const fwdEnd = t;

  // 反向：逆序。commit-bwd 等自己的 reload；start-bwd 预取 LIFO 下一组
  const toReload = groups.filter((g) => g.offload); // _groups_to_reload（前向顺序，LIFO 出队）
  let h2dFree = 0;
  let stall = 0;
  for (const g of [...groups].reverse()) {
    let begin = t;
    if (g.offload) {
      // FineGrainedOffloadingGroupCommitFunction.backward → wait_reload_event
      const ready = g.h2d ? g.h2d[1] : Infinity;
      if (ready === Infinity) throw new Error(`${g.id} 没有被预取就进入反向`);
      if (ready > begin) {
        stall += ready - begin;
        begin = ready;
      }
    }
    g.bwd = [begin, begin + cfg.bwdCompute];
    t = begin + cfg.bwdCompute;
    // FineGrainedOffloadingGroupStartFunction.backward → on_group_start_backward → bulk_reload
    const next = toReload.pop();
    if (next) {
      const start = Math.max(t, h2dFree);
      const dur = next.mb / cfg.pcieMBPerUnit;
      next.h2d = [start, start + dur];
      h2dFree = start + dur;
    }
  }
  const bwdEnd = t;

  // 对照：margin=0 时最后 margin 个组没有任何预取者，tensor_pop 同步拷回
  const exposedIfNoMargin = groups
    .filter((g) => !g.offload)
    .reduce((acc, g) => acc + g.mb / cfg.pcieMBPerUnit, 0);

  // 常驻字节：组产出后常驻，直到 D2H 完成（record_stream 后释放）；不 offload 的组常驻到反向用完
  const events = [];
  for (const g of groups) {
    events.push([g.fwd[1], g.mb]);
    events.push([g.offload ? g.d2h[1] : g.bwd[1], -g.mb]);
  }
  events.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  let cur = 0;
  let peak = 0;
  const curve = [];
  for (const [time, delta] of events) {
    cur += delta;
    peak = Math.max(peak, cur);
    curve.push([time, cur]);
  }
  const peakNoOffload = groups.reduce((acc, g) => acc + g.mb, 0);
  return { groups, margin, fwdEnd, bwdEnd, stall, exposedIfNoMargin, peakMB: peak, peakNoOffloadMB: peakNoOffload, curve };
}

// ============================================================================
// 图 2 的解算：复刻 paged_stash.py 的缓冲预定与 ops/paged_stash.py 的复制判定
// ============================================================================

export const STASH_CFG = Object.freeze({
  layers: 3,
  pageSize: 64, // moe_paged_stash_page_size
  maxNumTokens: 256, // 每层 permuted 缓冲行数（rank capacity 已 pad）
  capacityFactor: 1.6, // moe_expert_rank_capacity_factor
  cudaFactor: 1.1, // moe_paged_stash_buffer_size_factor_cuda
  cpuFactor: 0.5, // 对照分支使用的 moe_paged_stash_buffer_size_factor_cpu
  stepA: Object.freeze([150, 190, 120]),
  stepB: Object.freeze([150, 250, 220]),
});

const ceilDiv = (a, b) => Math.ceil(a / b);

// 复刻 paged_stash_copy_kernel 的去向判定：CUDA 余量够 → CUDA；否则宿主余量够 → 溢写；否则 overflow
function stashStep(tokensPerLayer, cudaPages, hostPages, pageSize) {
  let cudaHead = 0;
  let hostHead = 0;
  const rows = [];
  let overflow = 0;
  let hostSpill = 0;
  for (let i = 0; i < tokensPerLayer.length; i += 1) {
    const need = ceilDiv(tokensPerLayer[i], pageSize);
    const row = { layer: i + 1, tokens: tokensPerLayer[i], need };
    if (overflow) {
      row.where = 'skipped';
    } else if (cudaPages - cudaHead >= need) {
      row.where = 'cuda';
      row.pages = [cudaHead, cudaHead + need];
      cudaHead += need;
    } else if (hostPages > 0 && hostPages - hostHead >= need) {
      row.where = 'host';
      row.pages = [hostHead, hostHead + need];
      hostHead += need;
      hostSpill = 1;
    } else {
      row.where = 'overflow';
      overflow = 1;
    }
    rows.push(row);
  }
  return { rows, overflow, hostSpill, cudaUsed: cudaHead, hostUsed: hostHead, totalNeed: rows.reduce((a, r) => a + r.need, 0) };
}

export function solveStash(cfg = STASH_CFG) {
  // TEGroupedMLP._fused_forward：avg_num_tokens = max_num_tokens // cap_factor
  const avgTokens = Math.floor(cfg.maxNumTokens / cfg.capacityFactor);
  // capture 迭代：三层的保存张量同时在存 → 峰值 = 3 × avg
  const peakAvg = avgTokens * cfg.layers;
  // allocate_stash_buffers：num_tokens = int(max_tokens_dict × scale)，页数向上取整
  const cudaTokens = Math.floor(peakAvg * cfg.cudaFactor);
  const cudaPages = ceilDiv(cudaTokens, cfg.pageSize);
  const hostTokens = Math.floor(peakAvg * cfg.cpuFactor);
  const hostPages = ceilDiv(hostTokens, cfg.pageSize);
  const stepA = stashStep(cfg.stepA, cudaPages, 0, cfg.pageSize);
  const stepBNoHost = stashStep(cfg.stepB, cudaPages, 0, cfg.pageSize);
  const stepBHost = stashStep(cfg.stepB, cudaPages, hostPages, cfg.pageSize);
  return {
    avgTokens,
    peakAvg,
    cudaTokens,
    cudaPages,
    cudaPagedTokens: cudaPages * cfg.pageSize,
    hostTokens,
    hostPages,
    stepA,
    stepBNoHost,
    stepBHost,
    maxTries: 2, // PagedStashRunner.__call__：assert num_tries < 2
  };
}

// ============================================================================
// SVG 基础设施（与 tools/figs/svg/megatron_comm_overlap_figures.mjs 同一套 token）
// ============================================================================

const esc = (value) => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

function textWidth(value, fontSize) {
  let units = 0;
  for (const ch of String(value)) units += ch.charCodeAt(0) < 0x7f ? 0.56 : 1;
  return units * fontSize;
}
function guard(value, fontSize, limit, where) {
  const width = textWidth(value, fontSize);
  if (width > limit) throw new Error(`${where}: "${value}" 需要 ${width.toFixed(1)}px，超出 ${limit}px`);
  return value;
}

const sharedStyle = `
  text{font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif}
  .panel{fill:#FBFCFE;stroke:#D9DEE7;stroke-width:1.2}
  .neutral{fill:#fff;stroke:#AEB6C2;stroke-width:1.2}
  .ghost{fill:#F5F7FA;stroke:#D9DEE7;stroke-width:1.1}
  .acc1{fill:#EAF1FD;stroke:#2563EB;stroke-width:1.5}
  .acc2{fill:#FCF1E6;stroke:#C3651F;stroke-width:1.5}
  .main{fill:none;stroke:#2563EB;stroke-width:2;marker-end:url(#arrowMain)}
  .aux{fill:none;stroke:#AEB6C2;stroke-width:1.4;stroke-dasharray:5 4;marker-end:url(#arrowAux)}
  .cost{fill:none;stroke:#C3651F;stroke-width:2;marker-end:url(#arrowCost)}
  .edge{fill:none;stroke:#AEB6C2;stroke-width:1.4;stroke-dasharray:4 4}
  .ti{font-size:18px;font-weight:700;fill:#1F2430}
  .su{font-size:11.5px;fill:#747C88}
  .pt{font-size:14px;font-weight:700;fill:#2A313B}
  .tx{font-size:12px;fill:#38414D}
  .sm{font-size:10.5px;fill:#68717D}
  .dim{font-size:10.5px;font-weight:600;fill:#173F87}
  .costtx{font-size:10.5px;font-weight:600;fill:#8A4A11}
  .rank{font-size:11px;font-weight:700;fill:#5B6470}
  .cap{font-size:11px;fill:#747C88}
  .gl{fill:none;stroke:#C8CFDA;stroke-width:.9}
  .fcell{fill:#DCE9FB;stroke:#fff;stroke-width:.8}
  .bcell{fill:#F6DFC8;stroke:#fff;stroke-width:.8}
  .copy{fill:#9CC2F3;stroke:#fff;stroke-width:.8}
  .keep{fill:#EAF1FD;stroke:#2563EB;stroke-width:1.2}
  .bub{fill:#F1F3F7;stroke:#fff;stroke-width:.8}
  .page{fill:#fff;stroke:#AEB6C2;stroke-width:1}
  .pagec{fill:#DCE9FB;stroke:#2563EB;stroke-width:1}
  .pageh{fill:#F6DFC8;stroke:#C3651F;stroke-width:1}
  .pagex{fill:#F1F3F7;stroke:#C3651F;stroke-width:1.2;stroke-dasharray:3 2}
`;
const defs = `
  <defs>
    <marker id="arrowMain" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#2563EB"/></marker>
    <marker id="arrowAux" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#AEB6C2"/></marker>
    <marker id="arrowCost" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#C3651F"/></marker>
  </defs>`;

function rect(x, y, w, h, cls = 'neutral', radius = 8) {
  return `<rect class="${cls}" x="${x}" y="${y}" width="${w}" height="${h}" rx="${radius}"/>`;
}
function text(x, y, value, cls = 'tx', anchor = 'start') {
  return `<text class="${cls}" x="${x}" y="${y}" text-anchor="${anchor}">${esc(value)}</text>`;
}
function line(x1, y1, x2, y2, cls = 'edge') {
  return `<path class="${cls}" d="M ${x1} ${y1} L ${x2} ${y2}"/>`;
}
function arrow(x1, y1, x2, y2, cls = 'main') {
  return `<path class="${cls}" d="M ${x1} ${y1} L ${x2} ${y2}"/>`;
}
function box(x, y, w, h, title, lines, cls = 'neutral', where = 'box') {
  const out = [rect(x, y, w, h, cls)];
  const inner = w - 20;
  out.push(text(x + 10, y + 19, guard(title, 12, inner, `${where}/title`), 'tx'));
  lines.forEach((entry, index) => {
    const value = typeof entry === 'string' ? entry : entry.text;
    const lineCls = typeof entry === 'string' ? 'sm' : entry.cls;
    out.push(text(x + 10, y + 36 + index * 15, guard(value, 10.5, inner, `${where}/L${index}`), lineCls));
  });
  return out.join('\n');
}
function header(w, title, subtitle) {
  return [
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} __H__" width="${w}" height="__H__" role="img">`,
    defs,
    `<style>${sharedStyle}</style>`,
    text(28, 32, title, 'ti'),
    text(28, 52, subtitle, 'su'),
  ];
}
const FONT = Object.freeze({ ti: 18, su: 11.5, pt: 14, tx: 12, sm: 10.5, dim: 10.5, costtx: 10.5, rank: 11, cap: 11 });

function assertNoTextOverlap(svg, name) {
  const boxes = [];
  for (const m of svg.matchAll(/<text class="([a-z0-9]+)" x="(-?[\d.]+)" y="(-?[\d.]+)" text-anchor="(\w+)">([^<]*)<\/text>/g)) {
    const [, cls, xs, ys, anchor, raw] = m;
    const size = FONT[cls];
    if (!size || raw.trim() === '') continue;
    const w = textWidth(raw.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>'), size);
    const x = anchor === 'middle' ? Number(xs) - w / 2 : anchor === 'end' ? Number(xs) - w : Number(xs);
    boxes.push({ x, y: Number(ys) - size * 0.78, w, h: size * 1.06, raw });
  }
  for (let i = 0; i < boxes.length; i += 1) {
    for (let j = i + 1; j < boxes.length; j += 1) {
      const a = boxes[i];
      const b = boxes[j];
      const dx = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
      const dy = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
      if (dx > 1 && dy > 1) throw new Error(`${name}: 文字重叠 "${a.raw}" × "${b.raw}"（重合 ${dx.toFixed(1)}×${dy.toFixed(1)}px）`);
    }
  }
  return svg;
}
const seal = (parts, w, h, name) => assertNoTextOverlap(parts.join('\n').replace(/__H__/g, String(h)) + '\n</svg>', name);

// ============================== 图 1 ==============================

export function renderOffloadTimeline() {
  const s = simulateOffload();
  const W = 1240;
  const p = header(
    W,
    '图 1　子模块级 offload：组提交触发 D2H，反向起点提前一组预取 H2D，margin 组不出 GPU',
    `PP=1、1 个 microbatch、${OFFLOAD_CFG.layers} 层 × 2 组；前向每组 ${OFFLOAD_CFG.fwdCompute}、反向每组 ${OFFLOAD_CFG.bwdCompute} 单位；PCIe ${OFFLOAD_CFG.pcieMBPerUnit} MB/单位。所有格子由仿真算出。`,
  );
  const x0 = 150;
  const x1 = W - 250;
  const scale = (x1 - x0) / s.bwdEnd;
  const X = (time) => x0 + time * scale;
  const laneY = { compute: 100, d2h: 150, h2d: 200 };
  const laneH = 34;
  const lanes = [
    ['compute', '主流（计算）'],
    ['d2h', 'd2h_stream'],
    ['h2d', 'h2d_stream'],
  ];
  for (const [key, label] of lanes) {
    p.push(text(x0 - 12, laneY[key] + laneH / 2 + 4, label, 'rank', 'end'));
    p.push(rect(x0, laneY[key], x1 - x0, laneH, 'ghost', 4));
  }
  // 时间刻度
  for (let tick = 0; tick <= s.bwdEnd; tick += 20) {
    p.push(line(X(tick), laneY.compute - 6, X(tick), laneY.h2d + laneH + 4, 'gl'));
    p.push(text(X(tick), laneY.compute - 10, String(tick), 'sm', 'middle'));
  }
  // 计算格子
  for (const g of s.groups) {
    const [a, b] = g.fwd;
    p.push(rect(X(a) + 1, laneY.compute + 3, X(b) - X(a) - 2, laneH - 6, 'fcell', 3));
    p.push(text((X(a) + X(b)) / 2, laneY.compute + laneH / 2 + 4, guard(`F${g.layer}${g.name === 'core_attn' ? 'a' : 'f'}`, 10.5, X(b) - X(a) - 4, 'fcell'), 'sm', 'middle'));
    const [c, d] = g.bwd;
    p.push(rect(X(c) + 1, laneY.compute + 3, X(d) - X(c) - 2, laneH - 6, g.offload ? 'bcell' : 'keep', 3));
    p.push(text((X(c) + X(d)) / 2, laneY.compute + laneH / 2 + 4, guard(`B${g.layer}${g.name === 'core_attn' ? 'a' : 'f'}`, 10.5, X(d) - X(c) - 4, 'bcell'), 'sm', 'middle'));
    if (g.d2h) {
      const [e, f] = g.d2h;
      p.push(rect(X(e) + 1, laneY.d2h + 3, X(f) - X(e) - 2, laneH - 6, 'copy', 3));
      p.push(text((X(e) + X(f)) / 2, laneY.d2h + laneH / 2 + 4, guard(`${g.mb}`, 10.5, X(f) - X(e) - 4, 'd2h'), 'sm', 'middle'));
      p.push(arrow(X(g.fwd[1]), laneY.compute + laneH, X(e) + 1, laneY.d2h, 'aux'));
    }
    if (g.h2d) {
      const [e, f] = g.h2d;
      p.push(rect(X(e) + 1, laneY.h2d + 3, X(f) - X(e) - 2, laneH - 6, 'copy', 3));
      p.push(text((X(e) + X(f)) / 2, laneY.h2d + laneH / 2 + 4, guard(`${g.mb}`, 10.5, X(f) - X(e) - 4, 'h2d'), 'sm', 'middle'));
      // 预取由后一组的 start 反向触发；命中的等待点是本组 commit 反向
      p.push(arrow(X(f) - 1, laneY.h2d, X(g.bwd[0]) + 1, laneY.compute + laneH, 'main'));
    }
  }
  // 前向 / 反向分界
  p.push(line(X(s.fwdEnd), laneY.compute - 6, X(s.fwdEnd), laneY.h2d + laneH + 4, 'cost'));
  p.push(text(X(s.fwdEnd) + 4, laneY.h2d + laneH + 16, '前向结束 → 反向开始', 'costtx'));

  // 常驻字节条
  const barY = 262;
  p.push(text(x0 - 12, barY + 12, 'GPU 常驻激活', 'rank', 'end'));
  const barH = 40;
  const maxMB = s.peakMB;
  p.push(rect(x0, barY, x1 - x0, barH, 'ghost', 4));
  let prev = 0;
  let prevMB = 0;
  const curve = [...s.curve, [s.bwdEnd, 0]];
  for (const [time, mb] of curve) {
    if (time > prev) {
      const h = (prevMB / maxMB) * (barH - 4);
      if (h > 0) p.push(`<rect class="copy" x="${X(prev)}" y="${barY + barH - 2 - h}" width="${Math.max(X(time) - X(prev), 0.5)}" height="${h}"/>`);
    }
    prev = time;
    prevMB = mb;
  }
  p.push(text(x1 + 8, barY + 14, `峰值 ${s.peakMB} MB（有 offload）`, 'costtx'));
  p.push(text(x1 + 8, barY + 30, `对照 ${s.peakNoOffloadMB} MB（全留 GPU，条高按 ${s.peakMB} MB 满格）`, 'sm'));

  // 右侧注解
  const noteX = x1 + 8;
  p.push(box(noteX, laneY.compute - 2, W - noteX - 28, 62, `margin = ${s.margin} 组`, ['组名去重数；L3 两组不 offload', '反向前两组因此无需等待'], 'acc1', 'note/margin'));
  p.push(box(noteX, laneY.h2d - 12, W - noteX - 28, 62, `本例阻塞 ${s.stall} 单位`, [`margin=0 时暴露 ${s.exposedIfNoMargin} 单位`, '最后两组只能在 tensor_pop 同步拷回'], 'acc2', 'note/stall'));

  const capY = barY + barH + 26;
  p.push(text(28, capY, 'F/B = 前向/反向，1a = 第 1 层 core_attn，1f = 第 1 层 expert_fc1；橙=须先等 reload 事件的反向，蓝边=margin 组的反向；拷贝条上的数字是组的 MB 数。', 'cap'));
  p.push(text(28, capY + 16, 'D2H 在 group_offload 之后由 d2h_stream 串行发起；H2D 由前一个反向组的 start 函数触发，因此预取距离 = margin。', 'cap'));
  return seal(p, W, capY + 30, 'megatron_memory_offload_timeline.svg');
}

// ============================== 图 2 ==============================

export function renderPagedStash() {
  const s = solveStash();
  const c = STASH_CFG;
  const W = 1240;
  const p = header(
    W,
    '图 2　paged stash：按 capture 迭代的平均 token 数预定页数，复制 kernel 按 free list 余量三选一',
    `${c.layers} 个 MoE 层、page=${c.pageSize}、每层 permuted 缓冲 ${c.maxNumTokens} 行、容量因子 ${c.capacityFactor} → avg=${s.avgTokens}；三层同时在存 → ${s.peakAvg} 行；cuda factor ${c.cudaFactor} → ${s.cudaTokens} 行 → ${s.cudaPages} 页。`,
  );
  // 左：页缓冲
  const px = 28;
  const py = 92;
  p.push(rect(px, py, 400, 236, 'panel', 10));
  p.push(text(px + 12, py + 22, 'PagedStashBuffer[dtype][hidden]', 'pt'));
  p.push(text(px + 12, py + 40, `cuda_buffer [${s.cudaPages} 页 × ${c.pageSize}, H]，free list 环形：head/tail`, 'sm'));
  const cellW = 36;
  const cellH = 28;
  for (let i = 0; i < s.cudaPages; i += 1) {
    const x = px + 12 + i * (cellW + 4);
    p.push(rect(x, py + 50, cellW, cellH, 'page', 3));
    p.push(text(x + cellW / 2, py + 50 + 18, `p${i}`, 'sm', 'middle'));
  }
  p.push(text(px + 12, py + 100, `host_buffer（cpu factor ${c.cpuFactor} 时）[${s.hostPages} 页]，factor 0 时不分配`, 'sm'));
  for (let i = 0; i < s.hostPages; i += 1) {
    const x = px + 12 + i * (cellW + 4);
    p.push(rect(x, py + 110, cellW, cellH, 'page', 3));
    p.push(text(x + cellW / 2, py + 110 + 18, `h${i}`, 'sm', 'middle'));
  }
  p.push(text(px + 12, py + 162, 'free_list_head/tail/capacity: shape (2,) = [cuda, host]，全在 device', 'sm'));
  p.push(text(px + 12, py + 178, '每个 PagedTensor 只留 page_record[max_pages] 与 spilled_to_host 标志', 'sm'));
  p.push(text(px + 12, py + 194, '状态推进：begin → capture（记 max tokens，不用 graph）→ captured', 'sm'));
  p.push(text(px + 12, py + 210, `预定：int(${s.peakAvg} × ${c.cudaFactor}) = ${s.cudaTokens} 行 → ceil/${c.pageSize} = ${s.cudaPages} 页 = ${s.cudaPagedTokens} 行`, 'dim'));
  p.push(text(px + 12, py + 226, '页是定长的：碎片只发生在每层最后一页内', 'sm'));

  // 中：两步的页分配
  const mx = 450;
  const colW = 392;
  const rowH = 22;
  function stepPanel(y, title, step, rowsCls) {
    const h = 62 + step.rows.length * rowH;
    p.push(rect(mx, y, colW, h, 'panel', 10));
    p.push(text(mx + 12, y + 22, title, 'pt'));
    p.push(text(mx + 12, y + 40, `需求 ${step.totalNeed} 页 vs CUDA ${s.cudaPages} 页`, 'sm'));
    step.rows.forEach((r, i) => {
      const ry = y + 52 + i * rowH;
      let cls = 'pagec';
      let where = `CUDA 页 ${r.pages ? `p${r.pages[0]}–p${r.pages[1] - 1}` : ''}`;
      if (r.where === 'host') {
        cls = 'pageh';
        where = `溢写宿主页 h${r.pages[0]}–h${r.pages[1] - 1}，host_spill=1`;
      } else if (r.where === 'overflow') {
        cls = 'pagex';
        where = 'CUDA 不够且无宿主页 → overflow=1';
      } else if (r.where === 'skipped') {
        cls = 'pagex';
        where = 'overflow 已置位，kernel 直接返回';
      }
      p.push(rect(mx + 12, ry, 26, rowH - 4, cls, 3));
      p.push(text(mx + 46, ry + 14, guard(`L${r.layer}: ${r.tokens} tokens → ${r.need} 页 → ${where}`, 10.5, colW - 50, rowsCls), 'sm'));
    });
    return h;
  }
  let my = py;
  my += stepPanel(my, `步 A 实际 [${c.stepA.join(', ')}]：全部命中 CUDA 页`, s.stepA, 'stepA') + 10;
  my += stepPanel(my, `步 B 偏斜 [${c.stepB.join(', ')}]，cpu factor 0`, s.stepBNoHost, 'stepB0') + 10;
  const stepBHostH = stepPanel(my, `步 B 偏斜，cpu factor ${c.cpuFactor}（${s.hostPages} 宿主页）`, s.stepBHost, 'stepBh');

  // 右：runner
  const rx = 852;
  const rw = W - rx - 28;
  p.push(box(rx, py, rw, 96, 'PagedStashRunner.__call__', ['每次尝试：跑完 forward-backward', 'stack(stash_overflow, over_budget, host_spill)', '一次 all_reduce(SUM) 跨全 rank 汇总', `assert num_tries < ${s.maxTries}`], 'neutral', 'runner'));
  p.push(arrow(rx + rw / 2, py + 96, rx + rw / 2, py + 112, 'main'));
  p.push(box(rx, py + 114, rw, 78, '仅 host_spill > 0', ['成功的回退：只打日志', '建议调大 factor_cuda', '不重跑'], 'acc1', 'runner/spill'));
  p.push(arrow(rx + rw / 2, py + 192, rx + rw / 2, py + 208, 'cost'));
  p.push(box(rx, py + 210, rw, 96, 'stash_overflow 或 over_budget > 0', ['prepare_for_rerun：清容量因子、关 paged stash', 'zero_grad、重置 full CUDA graph、释放页缓冲', '同一批 microbatch 整步重跑一次'], 'acc2', 'runner/rerun'));
  p.push(arrow(rx + rw / 2, py + 306, rx + rw / 2, py + 322, 'cost'));
  p.push(box(rx, py + 324, rw, 62, 'TE whole-MoE graph 已捕获', ['不允许动态回退 → RuntimeError', '调大容量因子后重启'], 'acc2', 'runner/graph'));

  const capY = Math.max(my + stepBHostH, py + 386) + 28;
  p.push(text(28, capY, `蓝=命中 CUDA 页，橙=溢写宿主页或 overflow。同一步 B，宿主页把"重跑一次"变成"只记 host_spill"；重跑上限 ${s.maxTries - 1} 次。`, 'cap'));
  return seal(p, W, capY + 16, 'megatron_memory_paged_stash.svg');
}

export function buildFigures() {
  return {
    'megatron_memory_offload_timeline.svg': renderOffloadTimeline(),
    'megatron_memory_paged_stash.svg': renderPagedStash(),
  };
}

const here = dirname(fileURLToPath(import.meta.url));
const invokedDirectly = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (invokedDirectly) {
  const outDir = process.argv[2] || join(here, '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks', 'megatron-lm', 'assets');
  mkdirSync(outDir, { recursive: true });
  for (const [name, svg] of Object.entries(buildFigures())) {
    writeFileSync(join(outDir, name), svg, 'utf8');
    console.log(`wrote ${join(outDir, name)}`);
  }
}
