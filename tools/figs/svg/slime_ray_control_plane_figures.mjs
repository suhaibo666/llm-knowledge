// 图：同一份输入（actor 1 节点 × 4 卡，rollout 4 卡，每 engine 2 卡）在 colocate 与
// disaggregate 两种布局下，怎样从 `_get_placement_group_layout` 的 (GPU 总数, rollout offset)
// 走到排序后的 bundle、trainer rank / engine 的绑定、needs_offload 判定和端口分配。
// 源码基线：THUDM/slime@681b3adca54105d5ecd3fb822fa0dc58a427e0f9。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
// 要讲清楚：placement group 只产出"总量 + 偏移"，逻辑序号由 (node ip, gpu id) 排序得到，
// Ray 原生 bundle 序号只是被映射的对象；trainer 按 rank 绑逻辑序号，engine 按
// offset + i × gpus_per_engine 绑逻辑序号并取该槽位的 physical GPU id 作 base_gpu_id；
// colocate 与 disaggregate 的差别只在总量、偏移与 needs_offload，绑定规则完全相同。
// 端口在同一节点上按 server/nccl 全部先分、dist_init_addr 再分，每个 dist 端口预留 30 + dp_size 个。
//
// 布局：上下两条泳道（colocate / disaggregate），每条自左向右：输入 → 布局函数结果 →
// 排序后的 bundle 条（逻辑序号 / 节点·GPU / 原 bundle 序号）→ 绑定行（trainer 0.4、engine 0.2）；
// 右侧一列共用：needs_offload 判定与端口表。acc1 标 engine 的 base 槽位与决定性映射，
// acc2 标 colocate 的重叠区间（needs_offload=True 的代价来源）。
//
// 用法：node tools/figs/svg/slime_ray_control_plane_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// ---------------- 冻结的示例输入 ----------------
export const CFG = Object.freeze({
  actorNumNodes: 1,
  actorNumGpusPerNode: 4,
  rolloutNumGpus: 4,
  rolloutNumGpusPerEngine: 2,
  numGpusPerNode: 4,
  sglangDpSize: 1,
  basePort: 15000,
  trainerGpuDecl: 0.4,
  engineGpuDecl: 0.2,
});

// Ray 交回的原始 bundle 顺序（故意打乱），slime 会按 (node ip, gpu id) 重排。
const RAW_BUNDLES = Object.freeze({
  colocate: [['10.0.0.1', 2], ['10.0.0.1', 0], ['10.0.0.1', 3], ['10.0.0.1', 1]],
  disaggregate: [
    ['10.0.0.2', 0], ['10.0.0.1', 2], ['10.0.0.1', 0], ['10.0.0.2', 3],
    ['10.0.0.1', 1], ['10.0.0.2', 1], ['10.0.0.1', 3], ['10.0.0.2', 2],
  ],
});

// ---------------- 对源码算法的最小复现 ----------------
// slime/ray/placement_group.py::_get_placement_group_layout（只复现本图用到的两个分支）
export function layout(mode, cfg = CFG) {
  const actorGpus = cfg.actorNumNodes * cfg.actorNumGpusPerNode;
  if (mode === 'colocate') return { pgGpus: Math.max(actorGpus, cfg.rolloutNumGpus), rolloutOffset: 0 };
  if (mode === 'disaggregate') return { pgGpus: actorGpus + cfg.rolloutNumGpus, rolloutOffset: actorGpus };
  throw new Error(`unknown mode ${mode}`);
}

// slime/ray/placement_group.py::sort_key + _create_placement_group 的重排
export function reorder(rawBundles) {
  const infos = rawBundles.map(([ip, gpu], index) => ({ index, ip, gpu }));
  const ipKey = (ip) => ip.split('.').map(Number);
  const cmp = (a, b) => {
    const ka = ipKey(a.ip); const kb = ipKey(b.ip);
    for (let i = 0; i < 4; i += 1) if (ka[i] !== kb[i]) return ka[i] - kb[i];
    return a.gpu - b.gpu;
  };
  return [...infos].sort(cmp); // 逻辑序号 = 排序后位置；.index 是 Ray 原生 bundle 序号
}

// slime/ray/actor_group.py::RayTrainGroup._allocate_gpus_for_actor：rank r → 逻辑 r
export function trainerBindings(sorted, cfg = CFG) {
  const worldSize = cfg.actorNumNodes * cfg.actorNumGpusPerNode;
  return Array.from({ length: worldSize }, (_, rank) => ({ rank, logical: rank, bundle: sorted[rank].index }));
}

// slime/ray/rollout.py::ServerGroup.start_engines：gpu_index = gpu_offset + i × per-engine-on-node
export function engineBindings(sorted, rolloutOffset, cfg = CFG) {
  const perEngineOnNode = Math.min(cfg.rolloutNumGpusPerEngine, cfg.numGpusPerNode);
  const numEngines = Math.floor(cfg.rolloutNumGpus / perEngineOnNode);
  const slice = sorted.slice(rolloutOffset);
  return Array.from({ length: numEngines }, (_, i) => {
    const gpuIndex = i * perEngineOnNode; // group gpu_offset = 0（单 regular group）
    return {
      engine: i,
      logical: rolloutOffset + gpuIndex,
      bundle: slice[gpuIndex].index,
      baseGpuId: slice[gpuIndex].gpu,
      node: slice[gpuIndex].ip,
    };
  });
}

// slime/ray/rollout.py::start_rollout_servers 里的 needs_offload 判定
export function needsOffload(mode, offloadRollout, cfg = CFG) {
  const megatronGpus = cfg.actorNumNodes * cfg.actorNumGpusPerNode;
  const rolloutPgOffset = mode === 'colocate' ? 0 : megatronGpus;
  const groupAbsStart = rolloutPgOffset + 0;
  return offloadRollout && groupAbsStart < megatronGpus;
}

// slime/ray/rollout.py::_allocate_rollout_engine_addr_and_ports_normal，假设 basePort 起全部空闲
export function ports(cfg = CFG) {
  const perEngine = cfg.rolloutNumGpusPerEngine;
  const enginesPerNode = Math.max(1, Math.floor(cfg.numGpusPerNode / perEngine));
  const numEngines = Math.floor(cfg.rolloutNumGpus / Math.min(perEngine, cfg.numGpusPerNode));
  if (numEngines > enginesPerNode) throw new Error('示例限定所有 engine 落在同一节点');
  let cursor = cfg.basePort;
  const take = (consecutive = 1) => { const p = cursor; cursor += consecutive; return p; };
  const rows = Array.from({ length: enginesPerNode }, (_, i) => ({ engine: i }));
  for (const row of rows) { row.server = take(); row.nccl = take(); }
  const reserve = 30 + cfg.sglangDpSize;
  for (const row of rows) { row.dist = take(reserve); row.distEnd = row.dist + reserve - 1; }
  return { rows: rows.slice(0, numEngines), allocatedRows: rows.length, reserve, cursor };
}

export function model() {
  const out = {};
  for (const mode of ['colocate', 'disaggregate']) {
    const lay = layout(mode);
    const sorted = reorder(RAW_BUNDLES[mode]);
    if (sorted.length !== lay.pgGpus) throw new Error(`${mode}: raw bundles ${sorted.length} != pg ${lay.pgGpus}`);
    out[mode] = {
      ...lay,
      sorted,
      trainers: trainerBindings(sorted),
      engines: engineBindings(sorted, lay.rolloutOffset),
      // colocate 归一化后 offload_rollout=True；disaggregate 默认 False
      offloadRollout: mode === 'colocate',
      needsOffload: needsOffload(mode, mode === 'colocate'),
    };
  }
  out.ports = ports();
  return out;
}

// ---------------- 渲染 ----------------
const esc = (v) => String(v).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const STYLE = `
  text{font-family:"Segoe UI","Microsoft YaHei","PingFang SC",system-ui,sans-serif;fill:#2A313B}
  .ti{font-size:19px;font-weight:700;fill:#1F2430}
  .su{font-size:12px;fill:#747C88}
  .pt{font-size:14px;font-weight:700}
  .tx{font-size:12px;fill:#38414D}
  .sm{font-size:10.5px;fill:#5B6470}
  .cap{font-size:11.5px;fill:#5B6470}
  .panel{fill:#FBFCFE;stroke:#D9DEE7;stroke-width:1.2}
  .neutral{fill:#fff;stroke:#AEB6C2;stroke-width:1.2}
  .ghost{fill:#F5F7FA;stroke:#D9DEE7;stroke-width:1.1}
  .acc1{fill:#EAF1FD;stroke:#2563EB;stroke-width:1.5}
  .acc2{fill:#FCF1E6;stroke:#C3651F;stroke-width:1.5}
  .cell{fill:#fff;stroke:#AEB6C2;stroke-width:1.1}
  .main{fill:none;stroke:#2563EB;stroke-width:2;marker-end:url(#arrowMain)}
  .aux{fill:none;stroke:#AEB6C2;stroke-width:1.3;stroke-dasharray:5 4;marker-end:url(#arrowAux)}
`;

function render(m, cfg = CFG) {
  const W = 1180; const H = 840;
  const o = [];
  const rect = (x, y, w, h, cls = 'neutral', r = 7) => o.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" class="${cls}"/>`);
  const text = (x, y, s, cls = 'tx', anchor = 'start') => o.push(`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${esc(s)}</text>`);
  const arrow = (x1, y1, x2, y2, cls = 'main') => o.push(`<path d="M${x1} ${y1} L${x2} ${y2}" class="${cls}"/>`);

  o.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="title desc">`);
  o.push('<title id="title">slime Ray 控制面：同一输入在 colocate 与 disaggregate 下的资源布局与绑定</title>');
  o.push('<desc id="desc">上下两条泳道展示 placement group 的 GPU 总数与 rollout 偏移、按节点与 GPU 排序后的逻辑序号、trainer 与 engine 的绑定，以及 needs_offload 与端口分配。</desc>');
  o.push(`<defs><marker id="arrowMain" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#2563EB"/></marker><marker id="arrowAux" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#AEB6C2"/></marker></defs>`);
  o.push(`<style>${STYLE}</style><rect width="${W}" height="${H}" fill="white"/>`);

  const actorGpus = cfg.actorNumNodes * cfg.actorNumGpusPerNode;
  text(24, 34, '同一份输入的两种资源布局：总量与偏移不同，绑定规则相同', 'ti');
  text(24, 56, `actor ${cfg.actorNumNodes} 节点 × ${cfg.actorNumGpusPerNode} 卡 · rollout ${cfg.rolloutNumGpus} 卡 · 每 engine ${cfg.rolloutNumGpusPerEngine} 卡 · 每节点 ${cfg.numGpusPerNode} 卡 · sglang_dp_size=${cfg.sglangDpSize}`, 'su');

  const lanes = [
    { key: 'colocate', y: 80, title: 'colocate', fn: `max(${actorGpus}, ${cfg.rolloutNumGpus}) = ${m.colocate.pgGpus}`, off: `rollout offset = ${m.colocate.rolloutOffset}` },
    { key: 'disaggregate', y: 430, title: 'disaggregate', fn: `${actorGpus} + ${cfg.rolloutNumGpus} = ${m.disaggregate.pgGpus}`, off: `rollout offset = ${m.disaggregate.rolloutOffset}` },
  ];
  const laneW = 860; const laneH = 330;
  for (const lane of lanes) {
    const d = m[lane.key];
    rect(24, lane.y, laneW, laneH, 'panel', 10);
    text(40, lane.y + 24, `${lane.title}`, 'pt');

    // 输入 → 布局函数
    rect(40, lane.y + 40, 150, 78, 'ghost');
    text(52, lane.y + 60, '输入 args', 'sm');
    text(52, lane.y + 78, `actor GPUs = ${actorGpus}`, 'tx');
    text(52, lane.y + 96, `rollout GPUs = ${cfg.rolloutNumGpus}`, 'tx');
    text(52, lane.y + 112, lane.key === 'colocate' ? '--colocate' : '默认（不共卡）', 'tx');
    arrow(190, lane.y + 79, 222, lane.y + 79);
    rect(224, lane.y + 40, 210, 78, 'acc1');
    text(236, lane.y + 60, '_get_placement_group_layout', 'sm');
    text(236, lane.y + 80, `PG GPU 数 = ${lane.fn}`, 'tx');
    text(236, lane.y + 100, lane.off, 'tx');
    arrow(434, lane.y + 79, 466, lane.y + 79);
    rect(468, lane.y + 40, 400, 78, 'neutral');
    text(480, lane.y + 58, '_create_placement_group', 'sm');
    text(480, lane.y + 76, `${d.pgGpus} 个 {GPU:1, CPU:1} bundle，PACK 策略`, 'tx');
    text(480, lane.y + 93, 'InfoActor 逐 bundle 读 (node ip, gpu id)', 'tx');
    text(480, lane.y + 110, 'sorted(key=(ip 数值段, gpu id)) → 逻辑序号', 'tx');

    // bundle 条
    const stripY = lane.y + 140; const cellW = 96; const cellH = 58; const x0 = 40;
    text(x0, stripY - 8, '排序后的逻辑序号 → (节点, physical GPU) → Ray 原 bundle 序号', 'sm');
    d.sorted.forEach((b, i) => {
      const x = x0 + i * (cellW + 6);
      const isRollout = i >= d.rolloutOffset;
      const isTrain = i < actorGpus;
      const overlap = isRollout && isTrain;
      const cls = overlap ? 'acc2' : 'cell';
      rect(x, stripY, cellW, cellH, cls, 5);
      text(x + cellW / 2, stripY + 18, `逻辑 ${i}`, 'pt', 'middle');
      text(x + cellW / 2, stripY + 35, `${b.ip.split('.').pop() === '1' ? '节点 A' : '节点 B'} · GPU ${b.gpu}`, 'sm', 'middle');
      text(x + cellW / 2, stripY + 50, `bundle ${b.index}`, 'sm', 'middle');
    });

    // 绑定行
    const bindY = stripY + cellH + 22;
    text(x0, bindY, `trainer rank r → 逻辑 r，每个声明 ${cfg.trainerGpuDecl} GPU / ${cfg.trainerGpuDecl} CPU；rank 0 提供 master addr/port`, 'tx');
    d.trainers.forEach((t) => {
      const x = x0 + t.logical * (cellW + 6);
      rect(x + 6, bindY + 8, cellW - 12, 22, 'neutral', 4);
      text(x + cellW / 2, bindY + 23, `rank ${t.rank} · ${cfg.trainerGpuDecl}`, 'sm', 'middle');
    });
    const engY = bindY + 40;
    text(x0, engY, `engine i → 逻辑 offset + i × ${cfg.rolloutNumGpusPerEngine}，声明 ${cfg.engineGpuDecl} GPU；base_gpu_id 取该槽位的 physical GPU id`, 'tx');
    d.engines.forEach((e) => {
      const x = x0 + e.logical * (cellW + 6);
      rect(x + 6, engY + 8, 2 * cellW - 6, 22, 'acc1', 4);
      text(x + cellW, engY + 23, `engine ${e.engine} · ${cfg.engineGpuDecl} · base_gpu_id=${e.baseGpuId}`, 'sm', 'middle');
    });
    text(x0, engY + 48, d.needsOffload
      ? `重叠区间（橙）：rollout offset ${d.rolloutOffset} < Megatron GPU 数 ${actorGpus} 且 offload_rollout=True → needs_offload=True，engine 必须支持显存让渡`
      : `无重叠：group 起点 ${d.rolloutOffset} ≥ Megatron GPU 数 ${actorGpus} → needs_offload=False；若全局开了 offload_rollout，则 setdefault enable_memory_saver=False`, 'cap');
  }

  // 右列：端口
  const px = 900; const py = 80;
  rect(px, py, 256, 560, 'panel', 10);
  text(px + 14, py + 24, '端口分配（两条泳道相同）', 'pt');
  text(px + 14, py + 44, `同一节点 ${m.ports.allocatedRows} 个 engine 槽位，从 ${cfg.basePort} 起全部空闲`, 'sm');
  text(px + 14, py + 60, '先给所有 engine 分 server、nccl，', 'sm');
  text(px + 14, py + 75, `再逐 engine 分 dist_init_addr，各预留 30 + dp = ${m.ports.reserve} 个`, 'sm');
  let ry = py + 96;
  for (const r of m.ports.rows) {
    rect(px + 14, ry, 228, 88, 'neutral');
    text(px + 26, ry + 20, `engine ${r.engine}`, 'pt');
    text(px + 26, ry + 40, `server ${r.server} · nccl ${r.nccl}`, 'tx');
    text(px + 26, ry + 58, `dist_init_addr 端口 ${r.dist}`, 'tx');
    text(px + 26, ry + 76, `预留 ${r.dist}–${r.distEnd}`, 'sm');
    ry += 100;
  }
  rect(px + 14, ry, 228, 48, 'acc2');
  text(px + 26, ry + 20, `节点 cursor 终值 ${m.ports.cursor}`, 'tx');
  text(px + 26, ry + 38, '被占用的端口让 get_free_port 向后搜索', 'sm');
  ry += 64;
  text(px + 14, ry, 'router 单独取 3000–4000 间空闲端口，', 'sm');
  text(px + 14, ry + 16, '不在此区间；每个模型一个 router。', 'sm');
  text(px + 14, ry + 40, 'trainer rendezvous：rank 0 在', 'sm');
  text(px + 14, ry + 56, '20000–21000 间取空闲端口作 MASTER_PORT。', 'sm');
  text(px + 14, ry + 84, '橙：共卡重叠区间，代价是显存让渡；', 'sm');
  text(px + 14, ry + 100, '蓝：决定性映射与 engine 的 base 槽位。', 'sm');

  text(24, 786, `阅读顺序：输入 → 布局函数（总量 + 偏移）→ 排序后的逻辑序号 → 绑定（rank 按序号，engine 按偏移 + 步长）→ 让渡判定与端口。`, 'cap');
  text(24, 804, `两条泳道的绑定规则一字不差，差别只在 PG 总量、rollout 偏移和是否重叠；这正是 slime 把"布局"与"对象"分开的收益。`, 'cap');
  text(24, 828, '源码基线：THUDM/slime@681b3adca541 · 复现 _get_placement_group_layout / sort_key / start_engines / _allocate_rollout_engine_addr_and_ports_normal', 'su');
  o.push('</svg>');
  return o.join('\n');
}

const here = dirname(fileURLToPath(import.meta.url));
const defaultOutput = join(here, '..', '..', '..', 'wiki', '02_engineering', '04_posttrain_frameworks', 'slime', 'assets');
const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  const outputDir = process.argv[2] ? process.argv[2] : defaultOutput;
  mkdirSync(outputDir, { recursive: true });
  writeFileSync(join(outputDir, 'slime_ray_control_plane_layout.svg'), `${render(model())}\n`, 'utf8');
}
