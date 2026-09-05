// 图 1：谁掩盖谁 —— 六条轴各一个两泳道甘特，通信一条管线、掩盖它的计算另一条。
// 图 2：TP 的掩盖对按层类型分成两套（列并行 / 行并行），外加三条静默降级路径。
// 图 3：同一个 overlap_p2p_comm，前向 send 被同步化、反向 send 仍异步。
// 图 4：align_grad_reduce 把 DP 梯度同步钉在 PP 时间表上 —— 以及它漏掉的那一块。
// 图 5：CUDA_DEVICE_MAX_CONNECTIONS —— 一个全局标量，三方要不同的值。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
//
// 图 1 要回答的就是本页的正题：**具体是谁掩盖谁**。六条轴（TP/CP/EP/PP/DP/FSDP）各一个面板，
// 每个面板两条泳道 —— 上「通信」、下「计算」，横轴是相对时间。块的位置表达交叠区间，
// 竖虚线是第一个真正阻塞的收口点，右栏三行给出：掩盖对是什么、靠什么保证、什么条件下不成立。
// 块上写真实符号名（AG(input)/dgrad、eager_attn_fwd、combine_bwd、irecv、AG(bucket i+1)…），
// 不写抽象的"通信"二字。EP 那条直接实例化源码 docstring 里自带的两条 stream 甘特。
// 图注给出本图要证的结论：五条轴的掩盖物都是自己流水线上的相邻工作单元，只有 EP 让
// forward 与 backward 同时在场 —— 这也是它单独需要一个显存旋钮的原因。
//
// 图 2 要讲清楚：tp_comm_overlap 是一个总闸，但它在列并行与行并行上激活的不是同一组机制。
// 两个面板同构于图 1（前向 + 反向各一段，中间一条"反向"分隔线）：
//   列并行 qkv/fc1（TELayerNormColumnParallelLinear）：前向 AG↔GEMM 流水；
//     反向 bulk AG↔dgrad、bulk RS↔wgrad，另有 pipelined 的 ub_overlap_rs_dgrad。
//   行并行 proj/fc2（TELinear）：前向 RS↔GEMM；反向 AG↔dgrad。没有 bulk 开关。
// 下方三个框：依赖边界（ghost + TE 芯片，声明能证明什么/不能证明什么）、
// 两个 bulk 开关名字与 docstring 互换的矛盾、三条静默降级路径。
//
// 图 3 要讲清楚一个反直觉事实：overlap_p2p_comm=True 时前向与反向掩盖的东西不一样多。
// 两个面板结构相同、只差一句 wait 的位置：前向 pp_post_forward 发完 isend 立刻 wait
// （因为 deallocate_pipeline_outputs 要释放源存储），反向 pp_post_backward 不 wait。
// 强调色：acc2 = 被同步化的那段（代价），acc1 = 真正保持异步的那段。
//
// 图 4 要讲清楚：`align_grad_reduce` 的"对齐"是把 grad_sync 的触发条件改写成
// vmb − pp_rank，让四个 stage 落在同一个墙钟列上；以及这条规则在 vp=2 时漏掉了什么。
// 布局：上半是真实调度网格（pp=4, vp=2, m=8，由 lib/megatron_pp_sim.mjs 解算），
// 在每个 rank 行上标出 grad_sync 实际触发的那一格（acc1）与被漏下、只能进 cooldown
// 收尾循环的那个 chunk（acc2）。下半是 param_sync 的触发条件表：
// `1 < chunk_of(vmb)+1 < num_model_chunks` 在 vp=2 / 3 / 4 下分别命中哪些 chunk ——
// 由同一段复刻代码算出，结论 vp=2 时**一次都不触发**。
//
// 图 5 要讲清楚：本页 §1 的"资源上限"在源码里有一个最锋利的证据 —— 一个进程级标量，
// TP/CP 要它等于 1、FSDP 与 EP overlap 要它大于 1，由 GPU 架构仲裁。
//
// 硬规矩：图 4 的每一格、每个触发列、每个命中 chunk 都由复刻的调度代码算出，不手写；
// 每行文字过一遍宽度守卫；渲染后做图元级文字重叠断言，越界与压字都在生成时报错。
//
// 用法：node tools/figs/svg/megatron_comm_overlap_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { scheduleTable, seqVpp, simulate } from './lib/megatron_pp_sim.mjs';

const CFG = Object.freeze({
  pp: 4, // pipeline_model_parallel_size
  vp: 2, // virtual_pipeline_model_parallel_size
  m: 8, // num_microbatches
  N: 4, // microbatch_group_size_per_vp_stage
  vpSweep: Object.freeze([2, 3, 4]), // 图 2 下半的 param_sync 命中表
});

const range = (start, count) => Array.from({ length: count }, (_, i) => start + i);

// ============================================================================
// 复刻 schedules.py::forward_backward_pipelining_with_interleaving 的两处触发条件
// ============================================================================

// schedules.py:1344 —— microbatch_id_table / model_chunk_id_table 由 get_schedule_table 解耦而来
function tables(m, vp, N) {
  const t = scheduleTable(m, vp, N);
  return { microbatchId: t.map((x) => x[0]), modelChunkId: t.map((x) => x[1]), total: t.length };
}

// schedules.py::forward_backward_pipelining_with_interleaving.get_model_chunk_id
const modelChunkId = (tb, vmb, forward, vp) => {
  const c = tb.modelChunkId[vmb % tb.total];
  return forward ? c : vp - c - 1;
};
// schedules.py:1380 is_last_microbatch_for_model_chunk
const isLastForChunk = (tb, vmb, m) => vmb < tb.total && tb.microbatchId[vmb] === m - 1;
// schedules.py:1373 is_first_microbatch_for_model_chunk
const isFirstForChunk = (tb, vmb) => vmb < tb.total && tb.microbatchId[vmb] === 0;

// schedules.py::backward_step_helper_postprocess —— grad_sync 的触发点
// grad_sync_virtual_microbatch_id = virtual_microbatch_id - pipeline_parallel_rank
function gradSyncFirings(tb, { pp, vp, m }) {
  return range(0, pp).map((rank) => {
    const hits = [];
    for (let vmb = 0; vmb < tb.total; vmb += 1) {
      const g = vmb - rank;
      if (g >= 0 && isLastForChunk(tb, g, m)) {
        hits.push({ vmb, gradSyncVmb: g, chunk: modelChunkId(tb, g, false, vp) });
      }
    }
    return hits;
  });
}

// schedules.py::forward_step_helper_preprocess —— param_sync 的触发点
// 条件链：vmb+rank < total 且 is_first_microbatch_for_model_chunk，再 1 < chunk+1 < num_chunks
function paramSyncChunks(vp, { pp, m, N }) {
  const tb = tables(m, vp, N);
  const hit = new Set();
  for (let rank = 0; rank < pp; rank += 1) {
    for (let vmb = 0; vmb < tb.total; vmb += 1) {
      const p = vmb + rank;
      if (p < tb.total && isFirstForChunk(tb, p)) {
        const chunk = modelChunkId(tb, p, true, vp) + 1;
        if (1 < chunk && chunk < vp) hit.add(chunk);
      }
    }
  }
  return [...hit].sort((a, b) => a - b);
}

const TB = tables(CFG.m, CFG.vp, CFG.N);
const SIM = simulate(seqVpp(CFG), CFG);
const GRAD = gradSyncFirings(TB, CFG);
// 被调度器漏掉、只能进 cooldown「Launch any remaining grad reductions.」的 chunk
const MISSED = GRAD.map((hits) => {
  const covered = new Set(hits.map((h) => h.chunk));
  return range(0, CFG.vp).filter((c) => !covered.has(c));
});
const PARAM = Object.fromEntries(CFG.vpSweep.map((vp) => [vp, paramSyncChunks(vp, CFG)]));

// ============================================================================
// SVG 基础设施（与 tools/figs/svg/megatron_cp_figures.mjs 同一套 token）
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
  .fcell{fill:#DCE9FB;stroke:#fff;stroke-width:.8}
  .fcell2{fill:#9CC2F3;stroke:#fff;stroke-width:.8}
  .bcell{fill:#F6DFC8;stroke:#fff;stroke-width:.8}
  .bcell2{fill:#E8B887;stroke:#fff;stroke-width:.8}
  .bub{fill:#F1F3F7;stroke:#fff;stroke-width:.8}
  .mark{fill:none;stroke:#2563EB;stroke-width:2.2}
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
function chip(x, y, w, h, label, cls = 'ghost') {
  return [
    rect(x, y, w, h, cls, 6),
    text(x + w / 2, y + h / 2 + 4, guard(label, 10.5, w - 10, 'chip'), 'sm', 'middle'),
  ].join('\n');
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

// 「越界不等于不重叠」：viewBox 断言挡不住"标注压在邻列文字上"。把每条 <text> 还原成
// 包围盒两两求交，版面错误在生成时就红。
//
// 反过来也一样：**锚点在画布内不等于字形在画布内**。测试里的 assertInsideCanvas 只看
// `x`/`y` 这两个锚点坐标，一条右端超出画布的长标注照样通过——本文件的图 2 就真的这样
// 漏出去过。所以这里同时按还原出的盒子校验右/下边界。
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

// ============================================================================
// 两泳道甘特：掩盖的双方各占一条管线
// ============================================================================

const GX = 192; // 甘特区左边界
const GW = 596; // 甘特区宽度
const RX = 800; // 右栏左边界
const RW = 472;
const LANE_H = 26;

const tScale = (t) => GX + (t / 100) * GW;

// 一段工作：[t0, t1, 标签, class]。t 是 0..100 的相对刻度，只表达先后与交叠，不表达绝对时长。
function ganttBlock([t0, t1, label, cls], y, where) {
  const x = tScale(t0);
  const w = tScale(t1) - tScale(t0);
  return [
    rect(x, y, w, LANE_H, cls, 5),
    text(x + w / 2, y + 17, guard(label, 10.5, w - 10, where), 'sm', 'middle'),
  ].join('\n');
}

// 同一条泳道上的两段工作若在时间上交叠，后画的会把先画的整块盖住——图 3 的
// isend 就这样消失过一次。这类缺陷全在画布内、也不构成文字重叠，两道既有断言都挡不住，
// 所以在这里按时间轴单独校验：**一条泳道表达的是一个串行资源，交叠必须拆成两条**。
function assertLaneSerial(blocks, where) {
  const sorted = [...blocks].sort((a, b) => a[0] - b[0]);
  for (let i = 1; i < sorted.length; i += 1) {
    if (sorted[i][0] < sorted[i - 1][1]) {
      throw new Error(
        `${where}: 同一泳道上 "${sorted[i - 1][2]}" 与 "${sorted[i][2]}" 时间交叠 ` +
          `（${sorted[i - 1][0]}..${sorted[i - 1][1]} × ${sorted[i][0]}..${sorted[i][1]}）——拆成两条泳道`,
      );
    }
  }
}

// 一个轴的面板：左侧轴名、若干条泳道、收口标记行、右侧掩盖对说明。
// 泳道用 panel.lanes 显式给出；comm/comp 是「一条通信 + 一条计算」的简写。
function ganttPanel(panel, y, where) {
  const out = [];
  const lanes = panel.lanes ?? [
    { tag: '通信', blocks: panel.comm },
    { tag: '计算', blocks: panel.comp },
  ];
  const laneY = (i) => y + 4 + i * 30;
  const lastBottom = laneY(lanes.length - 1) + LANE_H;

  out.push(text(28, y + 16, panel.axis, 'pt'));
  panel.sub.forEach((s, i) => out.push(text(28, y + 34 + i * 15, guard(s, 10.5, 128, `${where}/sub`), 'sm')));

  out.push(line(GX, laneY(0) - 3, GX + GW, laneY(0) - 3, 'gl'));
  out.push(line(GX, lastBottom + 3, GX + GW, lastBottom + 3, 'gl'));

  if (panel.divider) {
    const [dt, dlabel] = panel.divider;
    out.push(line(tScale(dt), laneY(0) - 3, tScale(dt), lastBottom + 3, 'edge'));
    out.push(text(tScale(dt), laneY(0) - 8, dlabel, 'sm', 'middle'));
  }

  lanes.forEach((lane, li) => {
    out.push(text(184, laneY(li) + 17, lane.tag, 'rank', 'end'));
    assertLaneSerial(lane.blocks, `${where}/lane${li}(${lane.tag})`);
    lane.blocks.forEach((b, i) => out.push(ganttBlock(b, laneY(li), `${where}/l${li}b${i}`)));
  });

  // 收口标记：一条竖虚线穿过全部泳道，标签落在专用的一行上。
  (panel.waits ?? []).forEach(([t, label], i) => {
    const x = tScale(t);
    out.push(line(x, laneY(0) - 3, x, lastBottom + 8, 'edge'));
    out.push(text(x, lastBottom + 22, guard(label, 10.5, 300, `${where}/wait${i}`), 'costtx', 'middle'));
  });
  if (panel.note) {
    out.push(text(GX, lastBottom + 22, guard(panel.note, 10.5, 560, `${where}/note`), 'costtx'));
  }

  out.push(infoBox(RX, y + 2, RW, 84, panel.right.title, panel.right.lines, 'acc1', `${where}/right`));
  return out.join('\n');
}

// ============================== 图 1 ==============================

function renderPairsFigure() {
  const W = 1300;
  const PANEL_H = 100;
  const panels = [
    {
      axis: 'TP',
      sub: ['粒度：整块', 'sequence_parallel 反向'],
      comm: [
        [0, 30, 'AG(input) → total_input', 'acc1'],
        [42, 78, 'RS(grad_input) / AR', 'acc1'],
      ],
      comp: [
        [4, 36, 'dgrad = grad_output @ weight', 'neutral'],
        [46, 84, 'wgrad GEMM ＋ grad_bias', 'neutral'],
      ],
      waits: [[37, 'handle.wait()'], [88, 'handle.wait() → return']],
      right: {
        title: '掩盖对　AG ↔ dgrad；RS/AR ↔ wgrad',
        lines: [
          '两次异步都由 backward 自己发起、自己收口，掩盖物是它本身的下一段 GEMM',
          '前提：注释三处都写 rely on CUDA_DEVICE_MAX_CONNECTIONS=1 保证下发顺序',
          '不成立：wgrad_deferral_limit 命中时不算 wgrad，AG 压根不发',
        ],
      },
    },
    {
      axis: 'CP',
      sub: ['粒度：一个 KV head', '原生 eager fallback'],
      comm: [
        [0, 16, 'AG(K,V head 0)', 'acc1'],
        [20, 38, 'AG(head 1)', 'acc1'],
        [48, 66, 'AG(head 2)', 'acc1'],
      ],
      comp: [
        [20, 46, 'attn_fwd(head 0)', 'neutral'],
        [48, 74, 'attn_fwd(head 1)', 'neutral'],
        [76, 96, 'attn_fwd(head 2)', 'neutral'],
      ],
      waits: [[18, 'comm.wait()'], [46, 'comm.wait()'], [74, 'comm.wait()']],
      right: {
        title: '掩盖对　AG(KV head i+1) ↔ 第 i 个 head 的 attention',
        lines: [
          '循环体固定三步：comm.wait() → swap 双缓冲 → 发下一 head 的 AG → 算当前 head',
          '窗口不可调：heads_k_stride 写死为 1，旁边留着 TODO make it configurable',
          '不成立：最后一个 head 没有下一轮可预取；prologue 那次 AG 也全暴露',
        ],
      },
    },
    {
      axis: 'EP',
      sub: ['粒度：一对 microbatch', 'combined-1F1B'],
      comm: [
        [0, 18, 'combine_bwd', 'acc1'],
        [22, 60, 'dispatch_fwd → dispatch_bwd', 'acc1'],
        [64, 84, 'combine_fwd', 'acc1'],
      ],
      comp: [
        [0, 18, 'attn_fwd', 'neutral'],
        [22, 60, 'mlp_bwd → mlp_bwd_dw → mlp_fwd', 'neutral'],
        [64, 86, 'attn_bwd', 'neutral'],
      ],
      note: '无 handle：comm_stream 与 comp_stream 靠每个 microbatch 一个 torch.cuda.Event 定序',
      right: {
        title: '掩盖对　六个节点两两配对，源码 docstring 自己画了这张甘特',
        lines: [
          'comm_stream: combine_bwd | dispatch_fwd->dispatch_bwd | combine_fwd',
          'comp_stream: attn_fwd    | mlp_bwd->mlp_bwd_dw->mlp_fwd | attn_bwd',
          '唯一用「另一个 microbatch」的计算掩盖本 microbatch 通信的轴；首尾各一次无对手',
        ],
      },
    },
    {
      axis: 'PP',
      sub: ['粒度：一个调度槽', 'interleaved / VPP'],
      comm: [[8, 58, 'irecv(下一槽要吃的张量)', 'acc1']],
      comp: [
        [0, 30, '第 k 槽的 forward / backward', 'neutral'],
        [60, 92, '第 k+1 槽的 forward / backward', 'neutral'],
      ],
      waits: [[58, 'recv_prev_wait_handles.pop(0).wait()']],
      right: {
        title: '掩盖对　irecv(下一槽输入) ↔ 当前槽的计算',
        lines: [
          'handle 不当场等：塞进 FIFO，到真正消费它的那个槽才 pop 出来 wait',
          'send 侧并不对称——前向 send 被同步化，反向 send 才真异步，见图 3',
          '不成立：warmup 与 cooldown 的槽没有对手，两端阶梯全暴露',
        ],
      },
    },
    {
      axis: 'DP',
      sub: ['粒度：一个 bucket', 'DDP 的 bucket 链'],
      comm: [
        [0, 7, '…', 'ghost'],
        [8, 34, 'AG(bucket i+1)', 'acc1'],
        [38, 60, 'AG(bucket i+2)', 'acc1'],
        [74, 96, 'RS(bucket j 的梯度)', 'acc1'],
      ],
      comp: [
        [8, 34, 'fwd(bucket i)', 'neutral'],
        [38, 60, 'fwd(bucket i+1)', 'neutral'],
        [70, 96, 'bwd(其余层)', 'neutral'],
      ],
      divider: [65, 'loss'],
      waits: [[7, 'wait AG(i) → 立刻发 AG(i+1)'], [37, 'wait AG(i+1) → 发 AG(i+2)']],
      right: {
        title: '掩盖对　AG(bucket i+1) ↔ fwd(bucket i)；RS(bucket j) ↔ 其余层反向',
        lines: [
          'finish_param_sync 一句话做两件事：等当前桶，然后立刻派发下一桶',
          '注册序 ≠ 前向序时下一桶已被别人派发，源码打 warning 说这会伤害 overlap',
          '不成立：最后一个 bucket 的 RS 没有后继反向，finish_grad_sync 处全暴露',
        ],
      },
    },
    {
      axis: 'FSDP',
      sub: ['粒度：一个 unit', 'Megatron-FSDP'],
      comm: [
        [6, 30, 'AG(unit i+1)', 'acc1'],
        [32, 54, 'AG(unit i+2)', 'acc1'],
        [56, 88, 'AG(unit i+3) … 直到额度用尽', 'acc1'],
      ],
      comp: [
        [6, 32, 'fwd(unit i)', 'neutral'],
        [34, 58, 'fwd(unit i+1)', 'neutral'],
        [60, 86, 'fwd(unit i+2)', 'neutral'],
      ],
      waits: [[5, 'wait_bucket_ready(i)'], [33, '…(i+1)'], [59, '…(i+2)']],
      right: {
        title: '掩盖对　AG(后续若干 unit) ↔ 当前 unit 的 forward',
        lines: [
          '与 DP 的差别：一次预取到额度上限，不是只预取一个桶',
          '额度 suggested_AG_prefetch_size = suggested_communication_unit_size // 2',
          '只 wait 当前 bucket，后面几个继续在飞——在飞份数就是这里的显存代价',
        ],
      },
    },
  ];

  const H = 82 + panels.length * PANEL_H + 54;
  const p = header(
    W,
    '图 1　谁掩盖谁：六条轴的掩盖对',
    '每条轴两条管线——上面是被掩盖的通信，下面是掩盖它的计算。竖虚线是第一个真正阻塞的收口点；块的长短只表达先后与交叠，不表达绝对时长。',
  );
  panels.forEach((panel, i) => p.push(ganttPanel(panel, 82 + i * PANEL_H, `fig1/${panel.axis}`)));

  p.push(
    text(28, H - 34,
      '六条轴的掩盖物都是「同一条计算流上的相邻工作单元」——下一个 GEMM、下一个 head、下一个槽、下一个 bucket。只有 EP 例外：它让 forward 与 backward 同时在场，用另一个 microbatch 的计算去掩盖本 microbatch 的 A2A。',
      'cap'),
  );
  p.push(
    text(28, H - 14,
      '推论（本页归纳，源码未自述）：掩盖窗口的大小由各自 owner 的粒度决定，跨轴开关不叠加；EP 那条例外的代价是两个 microbatch 的激活必须同时活着，这正是 §5.3 那个显存旋钮存在的原因。',
      'cap'),
  );
  return seal(p, W, H, 'fig1');
}

// ============================== 图 2 ==============================

function renderTpFigure() {
  const W = 1300;
  const PANEL_H = 100;
  const panels = [
    {
      axis: '列并行',
      sub: ['qkv / fc1', 'TELayerNorm', 'ColumnParallelLinear'],
      comm: [
        [0, 34, 'AG(SP 输入)　ub_overlap_ag', 'acc1'],
        [46, 72, 'bulk AG(输入)', 'acc1'],
        [76, 98, 'bulk RS(dgrad 输出)', 'acc2'],
      ],
      comp: [
        [6, 40, 'fprop GEMM 的其余 chunk', 'neutral'],
        [50, 76, 'dgrad GEMM', 'neutral'],
        [80, 98, 'wgrad GEMM', 'neutral'],
      ],
      divider: [43, '反向'],
      right: {
        title: '前向 AG 与 GEMM 分块流水；反向两次 bulk 各配一段 GEMM',
        lines: [
          '前向输出不做 RS——列并行的输出本来就按列切，所以这里没有 ub_overlap_rs',
          '反向可改走 pipelined：ub_overlap_rs_dgrad 把 RS 与 dgrad 切块流水（默认关）',
          'tp_comm_overlap_disable_qkv / _disable_fc1 可单独关掉这一层的前向 AG',
        ],
      },
    },
    {
      axis: '行并行',
      sub: ['proj / fc2', 'TELinear'],
      comm: [
        [0, 34, 'RS(输出)　ub_overlap_rs', 'acc1'],
        [52, 92, 'AG(grad_output)　ub_overlap_ag', 'acc1'],
      ],
      comp: [
        [6, 40, 'fprop GEMM 的其余 chunk', 'neutral'],
        [56, 96, 'dgrad GEMM', 'neutral'],
      ],
      divider: [46, '反向'],
      right: {
        title: '行并行拿到的是完整输入、产出要 RS，方向与列并行正好相反',
        lines: [
          '这条路径上没有 bulk 开关：TELinear 根本不接收 ub_bulk_wgrad / ub_bulk_dgrad',
          '所以「TP overlap 开了」在两种层上激活的不是同一组机制',
          'is_expert=True 时四个 ub_* 全被写成 False —— 无 warning，静默降级',
        ],
      },
    },
  ];

  const H = 82 + panels.length * PANEL_H + 250;
  const p = header(
    W,
    '图 2　TP 的掩盖对按层类型分成两套，不是一个开关',
    'tp_comm_overlap 是总闸，但列并行与行并行走的是不同的 collective、不同的掩盖物、甚至不同的 TE 类。所有这些交叠都发生在 TE 内部——Megatron 树里只有 kwarg。',
  );
  panels.forEach((panel, i) => p.push(ganttPanel(panel, 82 + i * PANEL_H, `fig2/${i}`)));

  const boxY = 82 + panels.length * PANEL_H + 10;
  p.push(
    infoBox(28, boxY, 400, 180, '依赖边界：本页能证明什么', [
      'Megatron 侧只做两件事：initialize_ub() 注册缓冲、',
      '给 te.Linear 传 ub_* kwarg。',
      'megatron/ 全树在 UB 路径上没有 handle、',
      '没有 event、没有 .wait()。',
      '',
      '能证明：形状契约、开关条件、哪个开关挂哪个类。',
      '不能证明：交叠确实发生、省下了多少墙钟时间。',
      '',
      'TE 内部如何切块、切几块，本页不做陈述。',
    ], 'ghost', 'fig2/boundary'),
  );
  p.push(chip(28 + 400 - 136, boxY - 11, 132, 22, 'TransformerEngine', 'acc2'));

  p.push(
    infoBox(440, boxY, 400, 180, '矛盾：两个 bulk 开关的名字与自述互换', [
      'tp_comm_bulk_wgrad 的 docstring：',
      '  “All-Gather overlap with Bprop activation',
      '   gradient GEMM” —— 即 AG ↔ dgrad。',
      'tp_comm_bulk_dgrad 的 docstring：',
      '  “Reduce-Scatter overlap with Bprop weight',
      '   gradient GEMM” —— 即 RS ↔ wgrad。',
      '',
      'dgrad 是激活梯度、wgrad 是权重梯度：',
      '名字与描述正好对调，两者必有一处是错的。',
    ], 'acc2', 'fig2/swap'),
  );

  p.push(
    infoBox(852, boxY, 420, 180, '静默降级的三条路', [
      '① buffer name 不在 {qkv, proj, fc1, fc2} →',
      '   整层 tp_comm_overlap 置 False，有 warning。',
      '② is_expert=True → ub_overlap_ag 与',
      '   ub_overlap_rs 置 False，无 warning。',
      '③ parallel_mode == "duplicated" →',
      '   整段 ub_* 注入被跳过。',
      '',
      '②③ 都不打印任何东西：trace 里看不到 UB 并发，',
      '不代表开关没生效。',
    ], 'acc2', 'fig2/silent'),
  );

  p.push(
    text(28, H - 34,
      '所以「TP overlap 收益不及预期」的第一个检查点不是网络，而是这一层到底走了哪条路径：列并行还是行并行、是不是 expert、buffer name 有没有落在支持集合里。',
      'cap'),
  );
  p.push(
    text(28, H - 14,
      '依赖边界声明：本图右侧三条都取自 megatron/ 树内的分支与 docstring；TE 内部如何切块、切几块，本页不做陈述。',
      'cap'),
  );
  return seal(p, W, H, 'fig2');
}

// ============================== 图 3 ==============================

function renderPpSendFigure() {
  const W = 1300;
  const PANEL_H = 130;
  const panels = [
    {
      axis: '前向槽',
      sub: ['pp_post_forward', '发完就等'],
      lanes: [
        { tag: 'send', blocks: [[6, 26, 'isend(激活)', 'acc2']] },
        { tag: 'recv', blocks: [[8, 60, 'irecv(下一槽输入)', 'acc1']] },
        { tag: '计算', blocks: [[32, 92, '第 k+1 槽的计算', 'neutral']] },
      ],
      waits: [[28, 'send_next_wait_handle.wait() ← 当场']],
      right: {
        title: '前向 send 被同步化：能掩盖的只剩 irecv',
        lines: [
          '原因写在注释里：isend 是异步拷贝，源存储要被 deallocate_output_tensor 释放',
          '触发条件 deallocate_pipeline_outputs：core 默认 False，训练入口无条件设 True',
          '代价：省下前向激活的显存，换来 send 这一段重新暴露在关键路径上',
        ],
      },
    },
    {
      axis: '反向槽',
      sub: ['pp_post_backward', '留到下一槽'],
      lanes: [
        { tag: 'send', blocks: [[6, 30, 'isend(梯度)', 'acc1']] },
        { tag: 'recv', blocks: [[8, 60, 'irecv(下一槽梯度)', 'acc1']] },
        { tag: '计算', blocks: [[32, 92, '第 k+1 槽的计算', 'neutral']] },
      ],
      waits: [[94, '本槽不收口']],
      right: {
        title: '反向 send 保持异步：两条都被下一槽的计算掩盖',
        lines: [
          '同一个函数结构，唯独少了那一句「当场 wait」——因为梯度张量不走 deallocate',
          'send_prev_wait_handle 一直活到下一次 post_backward 才收口',
          '结论：overlap_p2p_comm=True 时，前向与反向掩盖的东西并不一样多',
        ],
      },
    },
  ];

  const H = 82 + panels.length * PANEL_H + 60;
  const p = header(
    W,
    '图 3　同一个 overlap_p2p_comm，前向与反向掩盖的东西不一样',
    '两个槽的代码结构几乎相同，差别只在一句 wait 的位置——而它由一个显存开关决定，不由 overlap 开关决定。',
  );
  panels.forEach((panel, i) => p.push(ganttPanel(panel, 82 + i * PANEL_H, `fig3/${i}`)));

  p.push(
    text(28, H - 40,
      '这解释了一个常见困惑：把 overlap_p2p_comm 打开、trace 里却仍看到前向 send 挡在关键路径上。它不是开关没生效，而是 deallocate_pipeline_outputs 主动把它换回了同步。',
      'cap'),
  );
  p.push(
    text(28, H - 20,
      '两个开关的 owner 不同：overlap_p2p_comm 归 PP 调度器，deallocate_pipeline_outputs 归显存；它们在这一句 wait 上相交，而源码没有任何一处把这层耦合写出来（本页归纳）。',
      'cap'),
  );
  return seal(p, W, H, 'fig3');
}

// ============================== 图 2 ==============================

function renderAlignFigure() {
  const W = 1240;
  const H = 640;
  const cell = 22;
  const gridX = 132;
  const gridY = 116;

  const p = header(
    W,
    '图 2　align_grad_reduce：把 DP 梯度同步钉在 PP 时间表上',
    `pp=${CFG.pp}, vp=${CFG.vp}, m=${CFG.m}, N=${CFG.N}；网格由 lib/megatron_pp_sim.mjs 解算，makespan=${SIM.span}，每 rank ${SIM.ops} 个 op、${SIM.bubble} 个空泡。`,
  );

  p.push(rect(20, 70, W - 40, 300, 'panel'));
  p.push(text(36, 94, '① 触发条件 grad_sync_vmb = virtual_microbatch_id − pipeline_parallel_rank，命中即发', 'pt'));
  p.push(
    text(
      36,
      110,
      '格子＝一个 forward(F)/backward(B) 槽，标注是 (microbatch, chunk)；■ 是该 rank 真正调用 grad_sync_func 的那一次反向。',
      'sm',
    ),
  );

  // 真实调度网格
  SIM.rows.forEach((row, r) => {
    const y = gridY + r * (cell + 6);
    p.push(text(gridX - 10, y + 17, `rank ${r}`, 'rank', 'end'));
    // 该 rank 上 grad_sync 命中的 virtual_microbatch_id → 在网格里定位第几个反向
    const hits = new Map(GRAD[r].map((h) => [h.vmb, h]));
    let bwdSeen = -1;
    row.forEach((op, t) => {
      const x = gridX + t * cell;
      if (op === null) {
        p.push(`<rect class="bub" x="${x}" y="${y}" width="${cell - 2}" height="${cell}"/>`);
        return;
      }
      if (!op.f) bwdSeen += 1;
      const hit = !op.f && hits.has(bwdSeen);
      const cls = op.f ? (op.c === 0 ? 'fcell' : 'fcell2') : op.c === 0 ? 'bcell' : 'bcell2';
      p.push(`<rect class="${cls}" x="${x}" y="${y}" width="${cell - 2}" height="${cell}"/>`);
      if (hit) p.push(`<rect class="mark" x="${x - 1.5}" y="${y - 1.5}" width="${cell + 1}" height="${cell + 3}" rx="3"/>`);
      p.push(text(x + (cell - 2) / 2, y + 17, `${op.f ? 'F' : 'B'}${op.mb}`, hit ? 'dim' : 'sm', 'middle'));
    });
    const fired = GRAD[r].map((h) => `chunk ${h.chunk}`).join('、');
    p.push(
      text(
        gridX + SIM.span * cell + 12,
        y + 17,
        guard(
          `发出：${fired || '无'}`,
          10.5,
          W - 28 - (gridX + SIM.span * cell + 12),
          `fig2/fired/r${r}`,
        ),
        'dim',
      ),
    );
  });

  p.push(
    text(
      36,
      gridY + CFG.pp * (cell + 6) + 22,
      `四个 rank 的命中列各自后移一格 —— 这正是 −pipeline_parallel_rank 的作用：把 DP reduce 摊在相邻的墙钟槽上，而不是四个 stage 同时挤进同一条链路。`,
      'sm',
    ),
  );
  p.push(
    text(
      36,
      gridY + CFG.pp * (cell + 6) + 40,
      `参数侧对称地取 +pipeline_parallel_rank（forward_step_helper_preprocess），所以预取比消费更早、reduce 比产出更晚。`,
      'sm',
    ),
  );

  // ---------- 漏网 ----------
  p.push(rect(20, 384, 596, 148, 'panel'));
  p.push(text(36, 408, '② 这条规则漏掉了什么', 'pt'));
  MISSED.forEach((chunks, r) => {
    const y = 426 + r * 22;
    const bad = chunks.length > 0;
    p.push(text(36, y + 12, `rank ${r}`, 'rank'));
    p.push(
      text(
        96,
        y + 12,
        bad
          ? `chunk ${chunks.join('、')} 未被调度器同步 → 进 cooldown 收尾循环`
          : '全部 chunk 都在对齐槽发出',
        bad ? 'costtx' : 'dim',
      ),
    );
  });
  p.push(
    text(
      36,
      518,
      `因为 vmb−rank 必须落在 [0, ${TB.total}) 内：rank r 能命中的最大 grad_sync_vmb 是 ${TB.total - 1}−r。`,
      'sm',
    ),
  );

  // ---------- param_sync 命中表 ----------
  p.push(rect(632, 384, W - 652, 148, 'panel'));
  p.push(text(648, 408, '③ 参数侧的门比梯度侧窄一格：1 < chunk+1 < num_model_chunks', 'pt'));
  p.push(text(648, 426, 'chunk 0 与 chunk 1 永远不会被调度器预取；它们只能走 finish_param_sync 的按需链。', 'sm'));
  CFG.vpSweep.forEach((vp, i) => {
    const x = 648 + i * 190;
    const hit = PARAM[vp];
    p.push(rect(x, 440, 176, 74, hit.length === 0 ? 'acc2' : 'acc1'));
    p.push(text(x + 12, 460, `vp = ${vp}`, 'rank'));
    p.push(
      text(
        x + 12,
        480,
        hit.length === 0 ? '调度器一次都不预取' : `预取 chunk ${hit.join('、')}`,
        hit.length === 0 ? 'costtx' : 'dim',
      ),
    );
    p.push(text(x + 12, 500, hit.length === 0 ? '窗口 1<c<2 为空' : `窗口 1<c<${vp}`, 'sm'));
  });

  p.push(
    text(
      28,
      H - 62,
      `所以"开了 align_param_gather 但 trace 里看不到预取"在 vp=2 下是**预期行为**，不是配置没生效 —— 排查应先看 vp，再看 bucket。`,
      'cap',
    ),
  );
  p.push(
    text(
      28,
      H - 42,
      `cooldown 末尾的「Launch any remaining grad reductions.」把 ② 里漏下的 chunk 逐个补发；这一段没有计算可掩盖，是本页 §1 "头尾上限"的具体落点。`,
      'cap',
    ),
  );
  p.push(
    text(
      28,
      H - 22,
      `字段 docstring 与执行接线方向相反（见正文的 contradiction 块）：只有 align_grad_reduce=True 才注入 grad_sync_func，本图画的正是 True 的那一支。`,
      'cap',
    ),
  );
  return seal(p, W, H, 'fig2');
}

// ============================== 图 3 ==============================

function renderConnectionFigure() {
  const W = 1240;
  const H = 660;
  const p = header(
    W,
    '图 3　CUDA_DEVICE_MAX_CONNECTIONS：一个进程级标量，三方要不同的值',
    '这是源码里最锋利的一处跨轴资源冲突：它不是显存、不是带宽，而是一个必须在启动前定死、且全进程共享的整数。',
  );

  p.push(
    infoBox(
      28,
      76,
      580,
      196,
      '设成 1 买到什么：kernel 按调用顺序下发',
      [
        'linear_with_grad_accumulation_and_async_allreduce 的 docstring 说得最完整：',
        '"…that should be scheduled before compute kernels to overlap the communication',
        ' with the computation, which is necessary for a speedup but not for correctness',
        ' so that ordering isn\'t imposed by the scheduler."',
        '',
        '三处依赖这条顺序的注释（都在 LinearWithGradAccumulationAndAsyncCommunication）：',
        '· all-gather 排在 input gradient 计算之前',
        '· all-reduce 排在 weight gradient 计算之前',
        '· reduce-scatter 排在 weight gradient 计算之前',
      ],
      'acc1',
      'fig3/one',
    ),
  );
  p.push(
    infoBox(
      632,
      76,
      580,
      196,
      '设成 1 的代价：连续通信 kernel 会挤掉计算',
      [
        'partition_buckets 的 docstring 给出反向理由：',
        '"…which doubles the number of communication kernels, and because of the use of',
        ' CUDA_DEVICE_MAX_CONNECTIONS=1, having multiple back-to-back communications will',
        ' prevent the overlap of communication kernels with computation kernels."',
        '',
        '于是 DDP 反过来合并连续的小 bucket —— 同一个标量，在 TP 侧是收益来源，',
        '在 DP 侧变成必须绕开的约束。EP dispatcher 里另有四处注释按它的取值',
        '重排 shared-expert GEMM 与 A2A 的下发顺序。',
      ],
      'acc2',
      'fig3/cost',
    ),
  );

  // ---------- 仲裁树 ----------
  p.push(rect(20, 292, W - 40, 268, 'panel'));
  p.push(text(36, 316, '仲裁：validate_args 里的一段三分支', 'pt'));
  p.push(text(36, 334, '入口条件：(tensor_model_parallel_size > 1 or context_parallel_size > 1) and get_device_arch_version() < 10', 'sm'));

  p.push(rect(36, 348, 250, 60, 'neutral'));
  p.push(text(48, 370, 'arch ≥ 10（Blackwell）', 'rank'));
  p.push(text(48, 390, '整段跳过：注释说要求已消失', 'dim'));

  p.push(rect(36, 418, 250, 124, 'neutral'));
  p.push(text(48, 440, 'arch < 10 且 TP>1 或 CP>1', 'rank'));
  p.push(text(48, 460, '按下面三支依次判定，', 'sm'));
  p.push(text(48, 476, '三支互斥，先命中先返回。', 'sm'));
  p.push(text(48, 500, '注意：这里判的是 TP/CP，', 'sm'));
  p.push(text(48, 516, '不是"谁真的需要 overlap"。', 'sm'));

  const branches = [
    {
      title: '① 同时开了 FSDP',
      lines: [
        'warn_rank_0：TP/CP 要 =1，',
        'Torch-FSDP2 / Megatron-FSDP 要',
        '"not setting CUDA_DEVICE_MAX_',
        'CONNECTIONS=1 for better',
        'parallelization"。',
        '只是警告 —— 环境变量维持原样。',
      ],
      cls: 'acc2',
    },
    {
      title: '② 同时开了 EP combined overlap',
      lines: [
        'warn_rank_0，并直接给出取舍：',
        '"you can set CUDA_DEVICE_MAX_',
        'CONNECTIONS to 1 or 32, which',
        'depends on which parallelization',
        'you want to prioritize."',
        '同样只是警告。',
      ],
      cls: 'acc2',
    },
    {
      title: '③ 都没开',
      lines: [
        'assert os.environ.get(...) == "1"',
        '',
        '硬失败，进程起不来。',
        '',
        'yaml_arguments 走另一条路：',
        'SP 未设 1 直接 RuntimeError。',
      ],
      cls: 'acc1',
    },
  ];
  branches.forEach((b, i) => {
    const x = 330 + i * 296;
    p.push(infoBox(x, 348, 276, 152, b.title, b.lines, b.cls, `fig3/br${i}`));
    p.push(arrow(292, 478, x - 6, i === 0 ? 400 : i === 1 ? 424 : 448, 'aux'));
  });

  p.push(text(330, 522, '反方向的两条硬门（与上面的 assert 直接对立）：', 'rank'));
  p.push(
    text(
      330,
      540,
      'use_torch_fsdp2 → "FSDP always requires CUDA_DEVICE_MAX_CONNECTIONS value large than one"（原文如此）',
      'sm',
    ),
  );
  p.push(
    text(
      330,
      556,
      'Megatron-FSDP → "requires CUDA_DEVICE_MAX_CONNECTIONS > 1 or unset"；走 YAML 配置那条路更严：SP 未设 1 直接 RuntimeError。',
      'sm',
    ),
  );

  p.push(
    text(
      28,
      H - 62,
      '本图要证的结论：进程组不同、CUDA stream 不同，都不代表两条通信互不干扰 —— 它们连"能同时下发几条"这个额度都共用一个进程级标量。',
      'cap',
    ),
  );
  p.push(
    text(
      28,
      H - 42,
      '诊断次序因此固定：先确认这个标量的取值与本作业开启的轴是否自洽，再去看 trace。取值错了，任何 stream priority 或 bucket 调参都只是换个等待位置。',
      'cap',
    ),
  );
  p.push(
    text(
      28,
      H - 22,
      '源码只陈述了各方的诉求与仲裁分支，没有给出任何拓扑下的最优值；"1 还是 32"必须在目标机器上量。',
      'cap',
    ),
  );
  return seal(p, W, H, 'fig3');
}

// ============================================================================

const here = dirname(fileURLToPath(import.meta.url));
const defaultOutput = join(
  here, '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks', 'megatron-lm', 'assets',
);
const outputDir = process.argv[2] ? process.argv[2] : defaultOutput;
mkdirSync(outputDir, { recursive: true });

const outputs = new Map([
  ['megatron_overlap_pairs.svg', renderPairsFigure()],
  ['megatron_overlap_tp_paths.svg', renderTpFigure()],
  ['megatron_overlap_pp_send.svg', renderPpSendFigure()],
  ['megatron_overlap_pp_dp_align.svg', renderAlignFigure()],
  ['megatron_overlap_connection_budget.svg', renderConnectionFigure()],
]);

for (const [name, svg] of outputs) writeFileSync(join(outputDir, name), `${svg}\n`, 'utf8');
console.log([...outputs.keys()].join('\n'));

export { CFG, GRAD, MISSED, PARAM, SIM, TB, gradSyncFirings, paramSyncChunks };
