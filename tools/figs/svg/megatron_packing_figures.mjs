// 29_megatron_packed_dataset_dynamic_cp_analysis.md 的三张图。
//
// 图 1：**原理图** —— 同一批变长样本喂给两个调度器，逐 microbatch、逐 DCP rank 算出工作量。
//       两套分组算法都复刻自源码（贪心 first-fit 与 packing-aware 的 CP 组选择），
//       条形高度、关键路径、剩余样本全部算出来，不手写。
// 图 2：共享的 run() 九步流水线 —— 每一步做什么、走哪种集合通信、谁是唯一分叉点。
// 图 3：reroute 为什么从 all-to-all 换成 DP 组 all-gather：谁 load 了样本、谁要算它，
//       以及两种通信形态的连接数对照。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
//
// 图 1 是本页的 principle figure，要回答「同一批样本，两个调度器到底把它排成了什么样」。
// 左右两栏：左 = DpBalancedScheduler（固定 CP，按原顺序贪心 first-fit），
// 右 = DefaultDynamicCPScheduler（按长度定 CP，工作量均衡）。
// 每栏画若干 microbatch，每个 microbatch 四条 rank 行，条长 = 该 rank 的 seq²/cp 工作量，
// 每个 microbatch 右侧标 max/min 比值（这一格的等待就由它决定）。
// 栏底给关键路径合计 = Σ 各 microbatch 的 max —— 这是 DP/PP 真正要等的量。
// 强调色只给两处：各 microbatch 的关键路径 rank（acc2），以及动态 CP 留到下一轮的剩余样本（ghost）。
//
// 图 2 布局：九个步骤竖排，每步三列：做什么 / 集合通信 / 归谁；第 ④ 步用 acc1 单独标为分叉点，
// 右侧一个盒子写「没有 PP 组广播」这条与直觉相反的设计及其补偿。
//
// 图 3 布局：上半 4 个 DCP rank 两行（load 侧 / compute 侧），画出错位；
// 下半两个盒子对照 all-to-all 与 DP 组 all-gather 的连接数与源码给的两条判据。
//
// 用法：node tools/figs/svg/megatron_packing_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const CFG = Object.freeze({
  dpSize: 2,
  cpSize: 2, // DpBalancedScheduler 的固定 CP 度
  maxSeqLenPerRank: 2048, // --max-seqlen-per-dp-cp-rank
  minCpSize: 1, // --min-dynamic-context-parallel-size
  // 一个 global batch 的样本长度，按数据集给出的**原始顺序**（first-fit 对顺序敏感）
  seqlens: Object.freeze([1024, 4096, 2048, 1024, 2048, 1024, 2048, 1024]),
  // data_schedule_utils.py:12
  workloadCapDelta: 0.05,
  // 第二组算例：只用来检验「均衡是不是算例凑出来的」，不进图，只进正文与测试
  altSeqlens: Object.freeze([8192, 1024, 1024, 2048, 1024, 2048, 1024, 1024]),
  altMaxSeqLenPerRank: 4096,
});

const TOTAL_GPUS = CFG.dpSize * CFG.cpSize;

// ============================================================================
// 复刻 data_schedule_utils.py::dcp_gpus_needed
// ============================================================================

const dcpGpusNeeded = (seqLen, maxSeqLenPerRank, minCpSize = 1) =>
  Math.max(minCpSize, Math.max(1, 2 ** Math.ceil(Math.log2(seqLen / maxSeqLenPerRank))));

const workload = (seqLen, cpSize) => (seqLen * seqLen) / cpSize;

// ============================================================================
// 复刻 data_schedule.py::DpBalancedScheduler.get_groups_and_subsamples
// ============================================================================

function dpBalancedGroups(seqlens, { dpSize, cpSize, maxSeqLenPerRank }) {
  const maxSeqLenAllRanks = maxSeqLenPerRank * cpSize;
  const packed = [];
  let single = [];
  let sum = 0;
  for (let i = 0; i < seqlens.length; i += 1) {
    if (sum + seqlens[i] <= maxSeqLenAllRanks) {
      single.push(i);
      sum += seqlens[i];
    } else {
      packed.push(single);
      single = [i];
      sum = seqlens[i];
    }
  }
  if (single.length > 0) packed.push(single);

  // 补齐到 dp_size 的整数倍：从后往前拆出单样本组
  const multiple = dpSize;
  if (packed.length % multiple !== 0) {
    let toMove = multiple - (packed.length % multiple);
    let i = packed.length - 1;
    while (toMove > 0) {
      if (i < 0) throw new Error('Not enough samples to move —— 复刻的断言被触发');
      if (packed[i].length > 1) {
        packed.push([packed[i].pop()]);
        toMove -= 1;
      } else {
        i -= 1;
      }
    }
  }

  // 按 seq_id = i*dp_size + floor(j/cp_size) 摊到 cp_size*dp_size 个 rank 上
  const microbatches = [];
  const count = packed.length / dpSize;
  for (let i = 0; i < count; i += 1) {
    const row = [];
    for (let j = 0; j < cpSize * dpSize; j += 1) {
      row.push(packed[Math.trunc(i * dpSize + j / cpSize)]);
    }
    microbatches.push(row);
  }
  return { packed, microbatches };
}

// ============================================================================
// 复刻 data_schedule_utils.py::next_hdp_group_packing_aware
// ============================================================================

function nextHdpGroupPackingAware(sampleSeqlens, totalGpus, maxSeqLenPerRank, minCpSize) {
  if (sampleSeqlens.length === 0) {
    return {
      microBatches: Array.from({ length: totalGpus }, () => []),
      leftovers: [],
      execTimes: Array.from({ length: totalGpus }, () => 0),
      sampleIdsPerGpu: Array.from({ length: totalGpus }, () => []),
    };
  }
  const cpMin = (s) => dcpGpusNeeded(s, maxSeqLenPerRank, minCpSize);

  const sorted = [...sampleSeqlens].sort((a, b) => b[1] - a[1]);
  const localTall = sorted[0][1];
  const cap = localTall * maxSeqLenPerRank * (1 + CFG.workloadCapDelta);

  let microBatches = Array.from({ length: totalGpus }, () => []);
  let execTimes = Array.from({ length: totalGpus }, () => 0);
  let sampleIdsPerGpu = Array.from({ length: totalGpus }, () => []);
  const packingLen = new Map();
  const gpuGroupId = Array.from({ length: totalGpus }, () => null);
  const groupMembers = new Map();
  const groupSize = new Map();
  let nextGid = 0;

  // 最高的那条按它的**最小** CP 数开一个组 —— 它不参与后面的搜索
  {
    const [sampleId, seqLen] = sorted[0];
    const cp = cpMin(seqLen);
    if (cp > totalGpus) {
      throw new Error(`序列 ${seqLen} 需要 CP=${cp}，但只有 ${totalGpus} 个 DCP rank`);
    }
    const gid = nextGid;
    nextGid += 1;
    const members = Array.from({ length: cp }, (_, i) => i);
    groupMembers.set(gid, members);
    groupSize.set(gid, cp);
    packingLen.set(gid, seqLen / cp);
    const cost = workload(seqLen, cp);
    for (const rank of members) {
      gpuGroupId[rank] = gid;
      microBatches[rank].push(seqLen);
      execTimes[rank] += cost;
      sampleIdsPerGpu[rank].push(sampleId);
    }
  }

  let leftovers = [];
  for (const [sampleId, seqLen] of sorted.slice(1)) {
    const minNeeded = cpMin(seqLen);
    let best = null;
    for (let cp = minNeeded; cp <= totalGpus; cp *= 2) {
      const cost = workload(seqLen, cp);
      for (const [gid, size] of [...groupSize.entries()]) {
        if (size !== cp) continue;
        if ((packingLen.get(gid) ?? 0) + seqLen / cp > maxSeqLenPerRank) continue;
        const memberSet = new Set(groupMembers.get(gid));
        const projected = Math.max(
          ...execTimes.map((t, rank) => (memberSet.has(rank) ? t + cost : t)),
        );
        if (projected <= cap && (best === null || projected < best.projected)) {
          best = { projected, cp, action: 'add', gid, members: null };
        }
      }
      const free = gpuGroupId.map((g, rank) => (g === null ? rank : -1)).filter((r) => r >= 0);
      if (free.length >= cp) {
        const chosen = [...free].sort((a, b) => execTimes[a] - execTimes[b]).slice(0, cp);
        const chosenSet = new Set(chosen);
        const projected = Math.max(
          ...execTimes.map((t, rank) => (chosenSet.has(rank) ? t + cost : t)),
        );
        if (projected <= cap && (best === null || projected < best.projected)) {
          best = { projected, cp, action: 'new', gid: null, members: chosen };
        }
      }
    }
    if (best === null) {
      leftovers.push([sampleId, seqLen]);
      continue;
    }
    const cost = workload(seqLen, best.cp);
    if (best.action === 'add') {
      packingLen.set(best.gid, packingLen.get(best.gid) + seqLen / best.cp);
      for (const rank of groupMembers.get(best.gid)) {
        microBatches[rank].push(seqLen);
        execTimes[rank] += cost;
        sampleIdsPerGpu[rank].push(sampleId);
      }
    } else {
      const gid = nextGid;
      nextGid += 1;
      groupMembers.set(gid, best.members);
      groupSize.set(gid, best.cp);
      packingLen.set(gid, seqLen / best.cp);
      for (const rank of best.members) {
        gpuGroupId[rank] = gid;
        microBatches[rank].push(seqLen);
        execTimes[rank] += cost;
        sampleIdsPerGpu[rank].push(sampleId);
      }
    }
  }

  // 兜底：DPxCP 不是 2 的幂、或扩不满时，整组 CP 收一批
  function fillWithFullGroup() {
    const selected = [];
    const next = [];
    let packed = 0;
    for (const [sampleId, seqLen] of sorted) {
      if (packed + seqLen / totalGpus <= maxSeqLenPerRank) {
        selected.push([sampleId, seqLen]);
        packed += seqLen / totalGpus;
      } else {
        next.push([sampleId, seqLen]);
      }
    }
    if (selected.length === 0) {
      throw new Error("至少要有一条能装进整组 CP —— 调大 max-seqlen-per-dp-cp-rank");
    }
    const perRank = selected.reduce((s, [, len]) => s + workload(len, totalGpus), 0);
    microBatches = Array.from({ length: totalGpus }, () => selected.map(([, len]) => len));
    execTimes = Array.from({ length: totalGpus }, () => perRank);
    sampleIdsPerGpu = Array.from({ length: totalGpus }, () => selected.map(([id]) => id));
    leftovers = next;
  }

  if (microBatches.some((mb) => mb.length === 0)) fillWithFullGroup();

  return { microBatches, leftovers, execTimes, sampleIdsPerGpu, groupSize, groupMembers };
}

// DefaultDynamicCPScheduler.get_groups_and_subsamples 的外层循环
function dynamicCpGroups(seqlens, { totalGpus, maxSeqLenPerRank, minCpSize }) {
  let pending = seqlens.map((len, id) => [id, len]);
  const rounds = [];
  let guard = 0;
  while (pending.length > 0) {
    guard += 1;
    if (guard > 32) throw new Error('动态 CP 分组没有收敛');
    const out = nextHdpGroupPackingAware(pending, totalGpus, maxSeqLenPerRank, minCpSize);
    rounds.push(out);
    if (out.leftovers.length === pending.length) {
      throw new Error('动态 CP 分组停滞：一轮没有消化任何样本');
    }
    pending = out.leftovers;
  }
  return rounds;
}

// ============================================================================
// 两条数据面的结算
// ============================================================================

const DPB = dpBalancedGroups(CFG.seqlens, CFG);
const DPB_WORK = DPB.microbatches.map((row) =>
  row.map((group) => group.reduce((s, id) => s + workload(CFG.seqlens[id], CFG.cpSize), 0)),
);
const DCP = dynamicCpGroups(CFG.seqlens, {
  totalGpus: TOTAL_GPUS,
  maxSeqLenPerRank: CFG.maxSeqLenPerRank,
  minCpSize: CFG.minCpSize,
});
const DCP_WORK = DCP.map((r) => r.execTimes);

const critical = (rows) => rows.reduce((s, row) => s + Math.max(...row), 0);
const spread = (row) => Math.max(...row) / Math.max(Math.min(...row), 1e-9);

const DPB_CRIT = critical(DPB_WORK);
const DCP_CRIT = critical(DCP_WORK);
const DPB_MAX_SPREAD = Math.max(...DPB_WORK.map(spread));
const DCP_MAX_SPREAD = Math.max(...DCP_WORK.map(spread));

if (DPB_WORK.length === 0 || DCP_WORK.length === 0) {
  throw new Error('图 1 的算例没有产出任何 microbatch');
}
if (!(DCP_MAX_SPREAD < DPB_MAX_SPREAD)) {
  throw new Error(
    `图 1 的立论前提被推翻：动态 CP 的最坏不均 ${DCP_MAX_SPREAD.toFixed(2)} 未低于固定 CP 的 ${DPB_MAX_SPREAD.toFixed(2)}`,
  );
}

// 第二组算例：同一对算法换一批长度分布。它要回答的是「动态 CP 到底在优化什么」——
// 源码 docstring 说的是 critical-path rank workload，不是每个 microbatch 内部的不均比。
const ALT = (() => {
  const opts = { ...CFG, maxSeqLenPerRank: CFG.altMaxSeqLenPerRank };
  const dpb = dpBalancedGroups(CFG.altSeqlens, opts);
  const dpbWork = dpb.microbatches.map((row) =>
    row.map((g) => g.reduce((sum, i) => sum + workload(CFG.altSeqlens[i], CFG.cpSize), 0)),
  );
  const dcp = dynamicCpGroups(CFG.altSeqlens, {
    totalGpus: TOTAL_GPUS,
    maxSeqLenPerRank: CFG.altMaxSeqLenPerRank,
    minCpSize: CFG.minCpSize,
  });
  const dcpWork = dcp.map((r) => r.execTimes);
  return {
    dpbCrit: critical(dpbWork),
    dcpCrit: critical(dcpWork),
    dpbSpread: Math.max(...dpbWork.map(spread)),
    dcpSpread: Math.max(...dcpWork.map(spread)),
  };
})();

// 立论只到源码自陈的那一条：关键路径不会更差。每格不均比则可能反过来。
if (!(ALT.dcpCrit <= ALT.dpbCrit)) {
  throw new Error('第二组算例上动态 CP 的关键路径反而更长，结论需要重写');
}
if (!(ALT.dcpSpread > ALT.dpbSpread)) {
  throw new Error('第二组算例本该展示「每格不均比可能更差」，现在没展示出来');
}

// reroute 的连接数：all-to-all 在 DPxCP 全域两两相连；all-gather 只在各 CP lane 的 dp_group 内
const CONN_ALL2ALL = TOTAL_GPUS * TOTAL_GPUS;
const CONN_GATHER = CFG.cpSize * CFG.dpSize * CFG.dpSize;

// ============================================================================
// SVG 基础设施（与 tools/figs/svg/megatron_comm_overlap_figures.mjs 同一套 token）
// ============================================================================

const esc = (value) =>
  String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

function textWidth(value, fontSize) {
  let units = 0;
  for (const ch of String(value)) units += ch.charCodeAt(0) < 0x7f ? 0.56 : 1;
  return units * fontSize;
}

function guard(value, fontSize, limit, where) {
  const width = textWidth(value, fontSize);
  if (width > limit) {
    throw new Error(`${where}: "${value}" 需要 ${width.toFixed(1)}px，超出 ${limit}px`);
  }
  return value;
}

const sharedStyle = `
  text{font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif}
  .card{fill:#fff;stroke:#E4E7EC;stroke-width:1.2}
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
  .refline{fill:none;stroke:#C8CFDA;stroke-width:1;stroke-dasharray:3 4}
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
function infoBox(x, y, w, h, title, lines, cls = 'neutral', where = 'box') {
  const out = [rect(x, y, w, h, cls)];
  const inner = w - 24;
  out.push(text(x + 12, y + 21, guard(title, 12, inner, `${where}/title`), 'tx'));
  out.push(line(x + 12, y + 29, x + w - 12, y + 29));
  lines.forEach((entry, index) => {
    const value = typeof entry === 'string' ? entry : entry.text;
    const lineCls = typeof entry === 'string' ? 'sm' : entry.cls;
    out.push(
      text(x + 12, y + 47 + index * 16, guard(value, 10.5, inner, `${where}/L${index}`), lineCls),
    );
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

const FONT = Object.freeze({
  ti: 18, su: 11.5, pt: 14, tx: 12, sm: 10.5, dim: 10.5,
  costtx: 10.5, rank: 11, cap: 11,
});

function assertNoTextOverlap(svg, name) {
  const { w: canvasW, h: canvasH } = (() => {
    const m = svg.match(/viewBox="0 0 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)"/);
    return { w: Number(m[1]), h: Number(m[2]) };
  })();
  const boxes = [];
  for (const m of svg.matchAll(
    /<text class="([a-z0-9]+)" x="(-?[\d.]+)" y="(-?[\d.]+)" text-anchor="(\w+)">([^<]*)<\/text>/g,
  )) {
    const [, cls, xs, ys, anchor, raw] = m;
    const size = FONT[cls];
    if (!size || raw.trim() === '') continue;
    const w = textWidth(raw.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>'), size);
    const x = anchor === 'middle' ? Number(xs) - w / 2 : anchor === 'end' ? Number(xs) - w : Number(xs);
    boxes.push({ x, y: Number(ys) - size * 0.78, w, h: size * 1.06, raw });
  }
  for (const b of boxes) {
    if (b.x < -1 || b.y < -1 || b.x + b.w > canvasW + 1 || b.y + b.h > canvasH + 1) {
      throw new Error(
        `${name}: 文字盒出画布 "${b.raw}"（${b.x.toFixed(1)}..${(b.x + b.w).toFixed(1)} × ` +
          `${b.y.toFixed(1)}..${(b.y + b.h).toFixed(1)}，画布 ${canvasW}×${canvasH}）`,
      );
    }
  }
  for (let i = 0; i < boxes.length; i += 1) {
    for (let j = i + 1; j < boxes.length; j += 1) {
      const a = boxes[i];
      const b = boxes[j];
      const dx = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
      const dy = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
      if (dx > 1 && dy > 1) {
        throw new Error(`${name}: 文字重叠 "${a.raw}" × "${b.raw}"（重合 ${dx.toFixed(1)}×${dy.toFixed(1)}px）`);
      }
    }
  }
  return svg;
}

const seal = (parts, w, h, name) =>
  assertNoTextOverlap(parts.join('\n').replace(/__H__/g, String(h)) + '\n</svg>', name);

const mu = (v) => `${(v / 1e6).toFixed(2)}M`;
const fx = (v, d = 2) => v.toFixed(d).replace(/\.?0+$/, '');

// ============================================================================
// 图 1：原理图 —— 同一批样本，两个调度器
// ============================================================================

function schedulerColumn(parts, { x, w, title, sub, rows, labels, where }) {
  const ROW_H = 22;
  const MB_GAP = 20;
  const top = 128;
  const bx = x + 62;
  const bw = w - 250;
  const peak = Math.max(...rows.flat());
  const sw = (v) => (v / peak) * bw;

  parts.push(text(x, top - 46, guard(title, 12, w, `${where}/t`), 'tx'));
  parts.push(text(x, top - 28, guard(sub, 10.5, w, `${where}/s`), 'sm'));

  let y = top;
  rows.forEach((row, m) => {
    const rowMax = Math.max(...row);
    parts.push(text(x, y + 14, guard(`mb ${m}`, 10.5, 56, `${where}/mb${m}`), 'rank'));
    row.forEach((v, rank) => {
      const yy = y + rank * ROW_H;
      const isCrit = v === rowMax;
      parts.push(rect(bx, yy + 2, Math.max(sw(v), 2), ROW_H - 7, isCrit ? 'acc2' : 'ghost', 3));
      parts.push(text(bx - 6, yy + 15, `r${rank}`, 'sm', 'end'));
      parts.push(
        text(bx + sw(v) + 6, yy + 15, guard(mu(v), 10.5, 60, `${where}/v${m}${rank}`), isCrit ? 'costtx' : 'sm'),
      );
    });
    parts.push(
      text(
        x + w - 8,
        y + 15,
        guard(`max/min = ${fx(spread(row))}×`, 10.5, 100, `${where}/sp${m}`),
        'dim',
        'end',
      ),
    );
    if (labels[m]) {
      parts.push(text(x + w - 8, y + 33, guard(labels[m], 10.5, 100, `${where}/lb${m}`), 'sm', 'end'));
    }
    y += row.length * ROW_H + MB_GAP;
  });
  return y;
}

function renderReplay() {
  const W = 1272;
  const H = 560;

  const parts = header(
    W,
    '图 1　同一批变长样本，两个调度器排出来的样子',
    `DP=${CFG.dpSize} × CP=${CFG.cpSize}（${TOTAL_GPUS} 个 DCP rank），max_seqlen_per_dp_cp_rank=${CFG.maxSeqLenPerRank}；` +
      `样本长度按数据集原始顺序 [${CFG.seqlens.join(', ')}]；条长 = 该 rank 的 Σ seq²/cp`,
  );

  const leftBottom = schedulerColumn(parts, {
    x: 28,
    w: 596,
    title: `DpBalancedScheduler：固定 CP=${CFG.cpSize}，按原顺序贪心 first-fit`,
    sub: `一个打包 microbatch 的容量 = ${CFG.maxSeqLenPerRank} × ${CFG.cpSize} = ${CFG.maxSeqLenPerRank * CFG.cpSize}`,
    rows: DPB_WORK,
    labels: DPB_WORK.map(() => ''),
    where: 'fig1/L',
  });
  const rightBottom = schedulerColumn(parts, {
    x: 652,
    w: 592,
    title: 'DefaultDynamicCPScheduler：按长度定 CP，工作量均衡',
    sub: `dcp_gpus_needed 向上取 2 的幂；候选上限 tall × ${CFG.maxSeqLenPerRank} × ${1 + CFG.workloadCapDelta}`,
    rows: DCP_WORK,
    labels: DCP.map((r) => (r.leftovers.length ? `剩 ${r.leftovers.length} 条留给下一轮` : '')),
    where: 'fig1/R',
  });

  const y = Math.max(leftBottom, rightBottom) + 4;
  parts.push(
    infoBox(
      28,
      y,
      596,
      104,
      `关键路径合计 ${mu(DPB_CRIT)}，最坏一格不均 ${fx(DPB_MAX_SPREAD)}×`,
      [
        'first-fit 按数据集给出的原始顺序装箱，不重排 —— 于是同一批样本换个顺序',
        '就会得到完全不同的分组。一个长样本落在哪个 microbatch 全看它排第几。',
        `本例最差的一格里，最忙 rank 是最闲 rank 的 ${fx(DPB_MAX_SPREAD)} 倍；`,
        '同一 microbatch 内所有 rank 必须等最忙的那个。',
      ],
      'neutral',
      'fig1/left',
    ),
  );
  parts.push(
    infoBox(
      652,
      y,
      592,
      104,
      `关键路径合计 ${mu(DCP_CRIT)}，最坏一格不均 ${fx(DCP_MAX_SPREAD)}×`,
      [
        '先按长度降序，最高的那条按它的最小 CP 数开组；其余样本在',
        '「加入已有同 CP 组 / 用空闲 rank 开新组」之间选让关键路径最低的那个，',
        `CP 候选按 2 的幂逐级放大 —— 短样本也可能用更大的 CP 组。`,
        '装不进本轮的留到下一轮，所以 microbatch 数可能比固定 CP 多。',
      ],
      'acc1',
      'fig1/right',
    ),
  );

  return seal(parts, W, H, 'fig1-replay');
}

// ============================================================================
// 图 2：共享的九步流水线
// ============================================================================

const STEPS = [
  ['①', 'get_batch_and_global_seqlens', '跨 DP all-gather 全局长度', '分组是一次全局决策'],
  ['②', '校验 required sample keys', '无', '缺一个 key 即 assert'],
  ['③', '按本 PP stage 裁掉用不到的字段', '无', '减少后面 reroute 的搬运量'],
  ['④', 'get_groups_and_subsamples', '无', '★ 两个调度器唯一分叉的地方'],
  ['⑤', 'reroute_samples_to_dcp_ranks', 'DP 组 all-gather，逐 key', '把样本搬到「该算它的 rank」'],
  ['⑥', 'build_packed_microbatches', '无', '拼 THD buffer，产出 PackedSeqParams'],
  ['⑦', '算 FLOPs 信息 Σseqlen、Σseqlen²', '无', '喂给吞吐统计'],
  ['⑧', 'broadcast_scalars', '跨 TP 组广播标量', '非 TP-0 rank 拿到标量'],
  ['⑨', 'create_data_iterator', '无', 'VPP 时按 vpp_needs_data 出列表'],
];

function renderPipeline() {
  const W = 1272;
  const H = 520;
  const X0 = 28;
  const RW = 432;
  const LW = 760;
  const ROW = 40;
  const Y0 = 92;

  const parts = header(
    W,
    '图 2　两个调度器共享的 run() 九步；只有第 ④ 步不同',
    'BasePackingScheduler → DpBalancedScheduler → DefaultDynamicCPScheduler：子类只重写第 ④ 步与 __init__',
  );

  parts.push(text(X0 + 44, Y0 - 12, '做什么', 'rank'));
  parts.push(text(X0 + 356, Y0 - 12, '集合通信', 'rank'));
  parts.push(text(X0 + 540, Y0 - 12, '这一步保证了什么', 'rank'));

  STEPS.forEach(([n, what, comm, why], i) => {
    const y = Y0 + i * ROW;
    const fork = n === '④';
    parts.push(rect(X0, y, LW, ROW - 6, fork ? 'acc1' : 'neutral', 5));
    parts.push(text(X0 + 12, y + 22, n, 'rank'));
    parts.push(text(X0 + 44, y + 22, guard(what, 10.5, 300, `fig2/w${i}`), fork ? 'dim' : 'sm'));
    parts.push(text(X0 + 356, y + 22, guard(comm, 10.5, 170, `fig2/c${i}`), 'sm'));
    parts.push(text(X0 + 540, y + 22, guard(why, 10.5, 214, `fig2/y${i}`), 'sm'));
    if (i < STEPS.length - 1) {
      parts.push(arrow(X0 + 6, y + ROW - 6, X0 + 6, y + ROW - 1, 'aux'));
    }
  });

  parts.push(
    infoBox(
      X0 + LW + 24,
      Y0,
      RW,
      164,
      '第 ④ 步的两种实现',
      [
        'DpBalancedScheduler：按原顺序贪心 first-fit，全部样本共用一个 CP；',
        '容量 = max_seqlen_per_dp_cp_rank × cp_size。',
        'DefaultDynamicCPScheduler：dcp_gpus_needed 按长度定 CP，取 2 的幂，',
        'next_hdp_group_packing_aware 按 seq²/cp 均衡到各 DCP rank。',
        '继承关系即结论：动态 CP 是打包调度器的子类，不是它的搭档。',
      ],
      'acc1',
      'fig2/fork',
    ),
  );
  parts.push(
    infoBox(
      X0 + LW + 24,
      Y0 + 180,
      RW,
      164,
      '一处与直觉相反：没有 PP 组广播',
      [
        '打包模式下 is_dataset_built_on_rank 对每个 stage 的 TP-0 都返回 True，',
        '所以每个 stage 各自取数、各自算全局 seqlen 统计，不从首/末 stage 广播。',
        '理由写在入口判据处：THD 打包与 SBHD 校验都要每个 stage 有 padding',
        '元数据，好让每个 MoE 层把物理 padding 排除掉。',
        '补偿是第 ③ 步：先裁掉本 stage 用不到的字段，减少搬运量。',
      ],
      'neutral',
      'fig2/nopp',
    ),
  );

  return seal(parts, W, H, 'fig2-pipeline');
}

// ============================================================================
// 图 3：reroute 的两种通信形态
// ============================================================================

function renderReroute() {
  const W = 1272;
  const H = 440;
  const X0 = 28;
  const CELL = 132;
  const GAP = 22;
  const Y_LOAD = 116;
  const Y_COMP = 236;

  const parts = header(
    W,
    '图 3　reroute：load 它的 rank 与算它的 rank 通常不是同一个',
    `DP=${CFG.dpSize} × CP=${CFG.cpSize}；CP 兄弟 rank 持有完全相同的输入，这是换成 DP 组 all-gather 的前提`,
  );

  const cellX = (i) => X0 + 96 + i * (CELL + GAP);

  parts.push(text(X0, Y_LOAD + 24, 'loader 给谁', 'rank'));
  parts.push(text(X0, Y_LOAD + 40, '（按 DP 分）', 'sm'));
  parts.push(text(X0, Y_COMP + 24, '调度说谁算', 'rank'));
  parts.push(text(X0, Y_COMP + 40, '（第 ④ 步的结果）', 'sm'));

  // loader 侧：dp rank 决定拿哪些样本；同一 dp 组内的 CP 兄弟拿到一模一样的内容
  for (let rank = 0; rank < TOTAL_GPUS; rank += 1) {
    const dp = Math.trunc(rank / CFG.cpSize);
    const cp = rank % CFG.cpSize;
    parts.push(rect(cellX(rank), Y_LOAD, CELL, 62, 'ghost', 6));
    parts.push(text(cellX(rank) + CELL / 2, Y_LOAD + 20, `DCP rank ${rank}`, 'rank', 'middle'));
    parts.push(
      text(cellX(rank) + CELL / 2, Y_LOAD + 38, `dp=${dp} cp=${cp}`, 'sm', 'middle'),
    );
    parts.push(
      text(cellX(rank) + CELL / 2, Y_LOAD + 54, `样本组 D${dp}（与兄弟同）`, 'sm', 'middle'),
    );
  }

  // compute 侧：取图 1 动态 CP 第一轮的实际分配
  const first = DCP[0];
  for (let rank = 0; rank < TOTAL_GPUS; rank += 1) {
    const ids = first.sampleIdsPerGpu[rank];
    parts.push(rect(cellX(rank), Y_COMP, CELL, 62, 'acc1', 6));
    parts.push(text(cellX(rank) + CELL / 2, Y_COMP + 20, `DCP rank ${rank}`, 'rank', 'middle'));
    parts.push(
      text(
        cellX(rank) + CELL / 2,
        Y_COMP + 38,
        guard(`样本 ${ids.length ? ids.join(', ') : '—'}`, 10.5, CELL - 12, `fig3/id${rank}`),
        'dim',
        'middle',
      ),
    );
    parts.push(
      text(
        cellX(rank) + CELL / 2,
        Y_COMP + 54,
        guard(`Σ seq²/cp = ${mu(first.execTimes[rank])}`, 10.5, CELL - 12, `fig3/w${rank}`),
        'sm',
        'middle',
      ),
    );
    parts.push(arrow(cellX(rank) + CELL / 2, Y_LOAD + 64, cellX(rank) + CELL / 2, Y_COMP - 2));
  }
  parts.push(
    text(
      cellX(0) + CELL / 2 + 8,
      (Y_LOAD + 64 + Y_COMP) / 2 + 4,
      guard('reroute：在各 CP lane 的 dp_group 内 all-gather，再只留本 rank 该算的那几条', 10.5, 620, 'fig3/mid'),
      'dim',
    ),
  );

  const bottom = Y_COMP + 84;
  parts.push(
    infoBox(
      X0,
      bottom,
      600,
      108,
      `被否掉的替代：NCCL all-to-all（${CONN_ALL2ALL} 对连接）`,
      [
        `all-to-all 在 ${TOTAL_GPUS} 个 DCP rank 之间建全连接 P2P 传输，本例 ${CONN_ALL2ALL} 对；`,
        `按 CP lane 在 dp_group 内 gather 只需 ${CFG.cpSize} × ${CFG.dpSize}² = ${CONN_GATHER} 对。`,
        '更关键的是：CP 兄弟持有的输入逐字节相同，让每个 CP rank 各收一份纯属重复。',
        'docstring 两条理由都写出来了：避免重复收件，避免全连接 P2P。',
      ],
      'acc2',
      'fig3/a2a',
    ),
  );
  parts.push(
    infoBox(
      X0 + 620,
      bottom,
      624,
      108,
      '为什么逐 key 发起，而不是一次收齐',
      [
        'gather 按 data key 一个一个发：每个 key 付一次固定的集合通信延迟，',
        '换来「同一时刻只有一个全局字段驻留」的临时显存上界；',
        '选出的切片先 clone 再进下一个 key，好让整块 gather 缓冲被释放。',
        '这是源码自陈的取舍（延迟换显存），不是实现疏漏。',
      ],
      'neutral',
      'fig3/key',
    ),
  );

  return seal(parts, W, H, 'fig3-reroute');
}

// ============================================================================

const outputs = new Map([
  ['megatron_packing_scheduler_replay.svg', renderReplay()],
  ['megatron_packing_pipeline.svg', renderPipeline()],
  ['megatron_packing_reroute.svg', renderReroute()],
]);

export {
  CFG, TOTAL_GPUS, DPB, DPB_WORK, DCP, DCP_WORK, ALT,
  DPB_CRIT, DCP_CRIT, DPB_MAX_SPREAD, DCP_MAX_SPREAD, critical, spread,
  CONN_ALL2ALL, CONN_GATHER,
  dcpGpusNeeded, workload, dpBalancedGroups, nextHdpGroupPackingAware, dynamicCpGroups,
  outputs,
};

const invokedDirectly = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (invokedDirectly) {
  const here = dirname(fileURLToPath(import.meta.url));
  const outDir =
    process.argv[2] ??
    join(here, '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks', 'megatron-lm', 'assets');
  mkdirSync(outDir, { recursive: true });
  for (const [name, svg] of outputs) {
    writeFileSync(join(outDir, name), svg, 'utf8');
    console.log(`${name}  ${svg.length} bytes`);
  }
  console.log(
    `\n固定 CP：${DPB_WORK.length} 个 microbatch，关键路径 ${mu(DPB_CRIT)}，最坏不均 ${fx(DPB_MAX_SPREAD)}×`,
  );
  console.log(
    `动态 CP：${DCP_WORK.length} 个 microbatch，关键路径 ${mu(DCP_CRIT)}，最坏不均 ${fx(DCP_MAX_SPREAD)}×`,
  );
  console.log(`reroute 连接数：all-to-all ${CONN_ALL2ALL} 对 vs DP 组 all-gather ${CONN_GATHER} 对`);
  console.log(
    `第二组算例：关键路径 ${mu(ALT.dpbCrit)} → ${mu(ALT.dcpCrit)}；` +
      `最坏一格不均 ${fx(ALT.dpbSpread)}× → ${fx(ALT.dcpSpread)}×（反而更差）`,
  );
}
