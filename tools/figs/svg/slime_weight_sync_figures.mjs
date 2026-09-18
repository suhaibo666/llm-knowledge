// 图：同一个最小实例（Qwen2 第 0 层 GLU linear_fc1.weight，H=2、F=4；训练 4 卡 TP=2、DP=2）
// 怎样先还原成与拓扑无关的 HF 张量，再分别经 NCCL、共卡 CUDA IPC（含 MoE 定向路由与越界 engine）、
// 整份磁盘、增量磁盘四条数据面搬运，并在各自的暂停窗口里提交。拆成三张图：
//   slime_weight_sync_common.svg  ① 共同前半段与三种分桶规则
//   slime_weight_sync_online.svg  ② NCCL、③ 共卡 IPC + MoE 定向路由（搬运在暂停窗口里）
//   slime_weight_sync_disk.svg    ④ 整份磁盘、⑤ 增量磁盘（写盘与主机 pull 在暂停之前）
// 源码基线：THUDM/slime@4c193f1f37509cca70f0e88807a9305b70f63f4e；
// SGLang 上游 sgl-project/sglang@0b3bb0cbe31873994c9f989fddfe2f87ca839fdd（v0.5.15.post1），
// slime 镜像补丁 docker/patch/latest/*.patch。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
// 要讲清楚：训练 TP 分片不是推理侧能直接装的形状；all_gather_param 先把各片 chunk(2) 再按
// [各片 gate…, 各片 up…] 拼接，convert_qwen2_to_hf 再 chunk(2) 成 gate_proj / up_proj，SGLang 加载器
// 按自己的 tp_rank 切片（上游契约）。四条数据面共用这一步，差别在搬运载体、谁发送、以及搬运落在
// pause 窗口之内（NCCL、IPC）还是之前（两条磁盘路径）。依赖侧的端点与语义在图上标出来源：
// IPC 返回前的设备同步来自 sglang-deterministic.patch，/pull_weights 来自 sglang-pull_weights.patch，
// CI 读回的 /get_weight_version 由 sglang.patch 改写；虚线框 = 依赖侧，acc1 = 决定性映射与提交点，
// acc2 = 错误做法与失败边界。每张图只放短标签，完整解释在正文。
//
// 用法：node tools/figs/svg/slime_weight_sync_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// ---------------- 冻结的示例输入 ----------------
export const CFG = Object.freeze({
  hidden: 2,
  ffn: 4,
  trainTp: 2,
  trainDp: 2,
  actorGpus: 4,
  inferTps: [1, 2, 4],
  bufferMiB: 512, // --update-weight-buffer-size 默认 512 * 1024**2
  nonExpertChunksMiB: [300, 260, 100],
  bigChunkMiB: 600,
  expertParamsMiB: [100, 100, 100, 100],
  trainEp: 4,
  // 训推分离：两个异构 engine（如 prefill TP=2、decode TP=4）
  ncclEngineGpuCounts: [2, 4],
  // colocate：rollout 6 卡、每 engine 2 卡；actor 只占槽位 0–3
  colocEngines: [
    { offset: 0, count: 2 },
    { offset: 2, count: 2 },
    { offset: 4, count: 2 },
  ],
  // MoE 变体：同样 4 张训练卡改成 TP=1、EP=4、8 个专家；两个共卡 engine 各 EP=2、MoE-DP=1
  moe: {
    numExperts: 8,
    trainEp: 4,
    inferEp: 2,
    moeDp: 1,
    engines: [
      { offset: 0, count: 2 },
      { offset: 2, count: 2 },
    ],
    bundle: 1, // 一个专家的 fc1 + fc2，单位"包"
    bufferBundles: 4,
  },
  // gate_proj 前 4 个 bf16 元素：一次更新改动了两个低位尾数字节
  delta: {
    old: [1.0, 0.5, -2.0, 0.25],
    neu: [1.0, 0.5078125, -2.0, 0.251953125],
  },
});

const range = (n) => Array.from({ length: n }, (_, i) => i);
const sum = (xs) => xs.reduce((a, x) => a + x, 0);

// ---------------- 对源码算法的最小复现 ----------------

// slime/backends/megatron_utils/update_weight/common.py::all_gather_param / all_gather_params_async 的
// linear_fc1 分支，与 slime/backends/megatron_utils/megatron_to_hf/qwen2.py::convert_qwen2_to_hf 的
// mlp.linear_fc1.weight 分支。SGLang 端按 tp_rank 切片是上游契约
// （python/sglang/srt/layers/linear.py::ColumnParallelLinear.weight_loader @ 0b3bb0cb）。
export function fc1Replay(cfg = CFG) {
  const F = cfg.ffn;
  const T = cfg.trainTp;
  const per = F / T;
  const shards = range(T).map((t) => [
    ...range(per).map((k) => `g${t * per + k}`),
    ...range(per).map((k) => `u${t * per + k}`),
  ]);
  const halves = shards.map((p) => [p.slice(0, p.length / 2), p.slice(p.length / 2)]);
  const gathered = [...halves.flatMap((h) => h[0]), ...halves.flatMap((h) => h[1])];
  const naiveCat = shards.flat();
  const gate = gathered.slice(0, F);
  const up = gathered.slice(F);
  const naiveGate = naiveCat.slice(0, F);
  const infer = {};
  for (const tp of cfg.inferTps) {
    const w = F / tp;
    infer[tp] = range(tp).map((r) => [...gate.slice(r * w, (r + 1) * w), ...up.slice(r * w, (r + 1) * w)]);
  }
  const directCopyOk = Object.fromEntries(
    cfg.inferTps.map((tp) => [tp, JSON.stringify(shards[0]) === JSON.stringify(infer[tp][0])]),
  );
  return { shards, gathered, naiveCat, gate, up, naiveGate, infer, directCopyOk };
}

// slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py::
// UpdateWeightFromDistributed._iter_non_expert_chunks 与
// slime/backends/megatron_utils/update_weight/hf_weight_iterator_direct.py::pack_param_info_buckets
// 共用的顺序装桶：已有内容且会超限才换桶。
export function bucketSequential(sizes, limit) {
  const out = [];
  let cur = [];
  let acc = 0;
  for (const s of sizes) {
    if (cur.length && acc + s > limit) {
      out.push(cur);
      cur = [];
      acc = 0;
    }
    cur.push(s);
    acc += s;
  }
  if (cur.length) out.push(cur);
  return out;
}

// UpdateWeightFromDistributed._iter_expert_chunks：阈值按 EP 聚合后的体量判断，
// 超限时先把已攒的批次 EP 聚合（空批次不产出）。
export function bucketExpert(sizes, ep, limit) {
  const out = [];
  let batch = [];
  let acc = 0;
  for (const s of sizes) {
    if ((acc + s) * ep > limit) {
      if (batch.length) out.push(batch);
      batch = [];
      acc = 0;
    }
    batch.push(s);
    acc += s;
  }
  if (batch.length) out.push(batch);
  return out.map((b) => ({ local: b, gathered: sum(b) * ep }));
}

// update_weight_from_distributed.py::connect_rollout_engines_from_distributed
export function ncclGroup(counts) {
  const offsets = [];
  let c = 0;
  for (const n of counts) {
    offsets.push(c + 1);
    c += n;
  }
  return { world: c + 1, rankOffsets: offsets };
}

// update_weight_from_tensor.py::UpdateWeightFromTensor.connect_rollout_engines
export function colocatePartition(actorGpus, engines) {
  let n = 0;
  for (const e of engines) {
    if (e.offset + e.count > actorGpus) break;
    n += 1;
  }
  const prefix = engines.slice(0, n);
  const suffix = engines.slice(n);
  return {
    prefixCount: n,
    gatherGroups: prefix.map((e) => ({ ranks: range(e.count).map((i) => e.offset + i), src: e.offset })),
    suffix: suffix.length ? ncclGroup(suffix.map((e) => e.count)) : null,
    suffixEngines: suffix.map((_, i) => n + i),
    pauseTargets: range(n),
  };
}

const cmpTuple = (a, b) => {
  for (let i = 0; i < Math.min(a.length, b.length); i += 1) if (a[i] !== b[i]) return a[i] - b[i];
  return a.length - b.length;
};

// update_weight/expert_routing.py::_get_expert_target_ranks / _build_expert_params /
// _build_expert_transfer_plan / _pack_expert_transfer_batches，单层、TP=1、expert-DP=1（每个专家只有一个物理持有者）。
export function expertRouting(m = CFG.moe) {
  const expected = m.inferEp * m.moeDp;
  const targets = range(m.inferEp).map(() => []);
  for (const e of m.engines) {
    if (e.count !== expected) throw new Error('SGLang MoE TP must be 1');
    for (const dp of range(m.moeDp)) for (const ep of range(m.inferEp)) targets[ep].push(e.offset + dp * m.inferEp + ep);
  }
  if (m.numExperts % m.inferEp) throw new Error('num_experts must be divisible by SGLang EP');
  const perInfer = m.numExperts / m.inferEp;
  const perTrain = m.numExperts / m.trainEp;
  let transfers = range(m.numExperts).map((expert) => ({
    expert,
    src: Math.floor(expert / perTrain),
    targets: targets[Math.floor(expert / perInfer)],
    size: m.bundle,
  }));
  transfers = [...transfers].sort((a, b) => cmpTuple(a.targets, b.targets) || a.src - b.src);
  const sized = [...transfers].sort((a, b) => b.size - a.size || cmpTuple(a.targets, b.targets) || a.src - b.src);
  if (m.bufferBundles < sized[0].size) throw new Error('bundle exceeds update_weight_buffer_size');
  const batches = [];
  const costs = [];
  for (const t of sized) {
    const parts = [...new Set([...t.targets, t.src])];
    const cands = costs
      .map((_, i) => i)
      .filter((i) => parts.every((r) => (costs[i].get(r) || 0) + t.size <= m.bufferBundles));
    let bi;
    if (cands.length) {
      bi = cands.reduce((best, i) => {
        const si = sum([...costs[i].values()]);
        const sb = sum([...costs[best].values()]);
        return si < sb || (si === sb && i < best) ? i : best;
      }, cands[0]);
    } else {
      bi = batches.length;
      batches.push([]);
      costs.push(new Map());
    }
    batches[bi].push(t);
    for (const r of parts) costs[bi].set(r, (costs[bi].get(r) || 0) + t.size);
  }
  const sends = sum(transfers.map((t) => t.targets.filter((r) => r !== t.src).length));
  const genericDeliveries = m.numExperts * (m.trainEp - 1);
  const routedPerRank = range(m.trainEp).map((r) => transfers.filter((t) => t.targets.includes(r)).length);
  return {
    targets,
    owners: range(m.trainEp).map((r) => transfers.filter((t) => t.src === r).map((t) => t.expert)),
    transfers,
    batches: batches.map((b) => b.map((t) => t.expert)),
    sends,
    genericDeliveries,
    routedPerRank,
    genericPerRank: m.numExperts,
  };
}

function bf16Bytes(x) {
  const u = new Uint32Array(new Float32Array([x]).buffer)[0];
  const r = (u + 0x7fff + ((u >>> 16) & 1)) >>> 16;
  return [r & 0xff, (r >>> 8) & 0xff]; // little-endian
}

// update_weight_from_disk_delta.py::UpdateWeightFromDiskDelta._encode_delta 与 slime/utils/disk_delta.py::overwrite_encode
export function deltaReplay(d = CFG.delta) {
  const oldB = d.old.flatMap(bf16Bytes);
  const newB = d.neu.flatMap(bf16Bytes);
  const xor = oldB.map((b, i) => b ^ newB[i]);
  const changedPos = xor.map((b, i) => (b ? i : -1)).filter((i) => i >= 0);
  const changed = changedPos.length;
  const overwriteLen = 4 + 4 * changed + changed;
  const xorTwice = xor.map((b, i) => b ^ newB[i]); // 在已应用的状态上再异或一次
  const overwriteTwice = newB.map((b, i) => (changedPos.includes(i) ? newB[i] : b));
  return {
    oldB,
    newB,
    xor,
    changed,
    total: oldB.length,
    density: changed / oldB.length,
    xorLen: xor.length,
    overwriteLen,
    xorTwiceRevertsToOld: JSON.stringify(xorTwice) === JSON.stringify(oldB),
    overwriteTwiceStaysNew: JSON.stringify(overwriteTwice) === JSON.stringify(newB),
  };
}

// docker/patch/latest/sglang-pull_weights.patch 里 python/sglang/srt/weight_sync/local_checkpoint.py::pull 的版本逻辑
export function pullReplay({ applied, target, isDelta }) {
  const ops = [];
  const floor = applied ?? 0;
  let start = target;
  while (start > floor && isDelta(start)) start -= 1;
  if (applied === null || start > applied) ops.push(start === 0 ? 'reset base' : `reset v${start}`);
  else start = applied;
  for (let v = start + 1; v <= target; v += 1) ops.push(`apply v${v}`);
  return { ops, localAfter: Math.max(start, target) };
}

// 各数据面在一次 update 里的源码顺序；pause=true 表示处于服务暂停窗口
export const PLANES = Object.freeze({
  nccl: [
    ['version+1', false],
    ['pause', true],
    ['flush', true],
    ['量化 restore*', true],
    ['非专家：gather→转换→广播', true],
    ['专家：gather→EP 聚合→广播', true],
    ['量化后处理*', true],
    ['continue', false],
  ],
  ipc: [
    ['version+1', false],
    ['pause 前缀', true],
    ['flush 前缀', true],
    ['量化 restore*', true],
    ['CPU→GPU、重组、转换→IPC', true],
    ['专家定向 P2P→IPC*', true],
    ['量化后处理*', true],
    ['continue 前缀', false],
  ],
  disk: [
    ['version+1', false],
    ['rmtree、mkdir', false],
    ['重组→写 HF 分片', false],
    ['post-write hook', false],
    ['pull 到本地*', false],
    ['pause', true],
    ['flush', true],
    ['reload', true],
    ['CI 版本核对*', true],
    ['rmtree', true],
    ['continue', false],
  ],
  delta: [
    ['version+1', false],
    ['重组→按字节做差→zstd', false],
    ['写分片与 index', false],
    ['post-write hook', false],
    ['pull：应用+校验', false],
    ['pause', true],
    ['flush', true],
    ['reload 本地', true],
    ['continue', false],
  ],
});

// 图上标出的依赖侧来源（正文 §2.2.8 的依赖边界表必须出现同样的补丁名）
export const PATCH_LABELS = Object.freeze({
  ipcSync: 'sglang-deterministic.patch',
  pull: 'sglang-pull_weights.patch',
  versionReadback: 'sglang.patch',
});

export function model(cfg = CFG) {
  return {
    fc1: fc1Replay(cfg),
    buckets: {
      nonExpert: bucketSequential(cfg.nonExpertChunksMiB, cfg.bufferMiB),
      big: bucketSequential([cfg.bigChunkMiB], cfg.bufferMiB),
      expert: bucketExpert(cfg.expertParamsMiB, cfg.trainEp, cfg.bufferMiB),
    },
    nccl: ncclGroup(cfg.ncclEngineGpuCounts),
    coloc: colocatePartition(cfg.actorGpus, cfg.colocEngines),
    moe: expertRouting(cfg.moe),
    delta: deltaReplay(cfg.delta),
    pulls: {
      fresh: pullReplay({ applied: null, target: 2, isDelta: () => true }),
      atOne: pullReplay({ applied: 1, target: 2, isDelta: () => true }),
      staleDelta: pullReplay({ applied: 57, target: 1, isDelta: () => true }),
      staleFull: pullReplay({ applied: 57, target: 1, isDelta: () => false }),
      staleFullCatchUp: pullReplay({ applied: 57, target: 58, isDelta: () => false }),
    },
  };
}

// ---------------- 渲染 ----------------
const esc = (v) => String(v).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const hex = (b) => b.toString(16).toUpperCase().padStart(2, '0');
// 估算文字宽度：CJK 与全角按字号计，其余按 0.58 字号计
export function textWidth(s, size) {
  let w = 0;
  for (const ch of String(s)) w += /[ -⯿⺀-￯]/.test(ch) ? size : size * 0.58;
  return w;
}
// 与 STYLE 中字号一致，测试用来估算文字盒
export const FONT = Object.freeze({ ti: 19, su: 12, pt: 14, tx: 12, sm: 10.5, mono: 11, cap: 11.5 });
const STYLE = `
  text{font-family:"Segoe UI","Microsoft YaHei","PingFang SC",system-ui,sans-serif;fill:#2A313B}
  .ti{font-size:19px;font-weight:700;fill:#1F2430}
  .su{font-size:12px;fill:#747C88}
  .pt{font-size:14px;font-weight:700}
  .tx{font-size:12px;fill:#38414D}
  .sm{font-size:10.5px;fill:#5B6470}
  .mono{font-size:11px;font-family:"SFMono-Regular",Menlo,Consolas,monospace;fill:#38414D}
  .cap{font-size:11.5px;fill:#5B6470}
  .panel{fill:#FBFCFE;stroke:#D9DEE7;stroke-width:1.2}
  .neutral{fill:#fff;stroke:#AEB6C2;stroke-width:1.2}
  .ghost{fill:#F5F7FA;stroke:#D9DEE7;stroke-width:1.1}
  .dep{fill:#F5F7FA;stroke:#AEB6C2;stroke-width:1.2;stroke-dasharray:5 4}
  .acc1{fill:#EAF1FD;stroke:#2563EB;stroke-width:1.5}
  .acc2{fill:#FCF1E6;stroke:#C3651F;stroke-width:1.5}
  .cell{fill:#fff;stroke:#AEB6C2;stroke-width:1.1}
  .cellu{fill:#EEF1F5;stroke:#AEB6C2;stroke-width:1.1}
  .win{fill:#EAF1FD;stroke:none}
  .main{fill:none;stroke:#2563EB;stroke-width:2;marker-end:url(#arrowMain)}
  .aux{fill:none;stroke:#AEB6C2;stroke-width:1.3;stroke-dasharray:5 4;marker-end:url(#arrowAux)}
`;
const W = 1180;
const BASELINE = '源码基线：THUDM/slime@4c193f1f3750 · 上游 sgl-project/sglang@0b3bb0cbe318（v0.5.15.post1）+ slime 的 docker/patch/latest/*.patch';

function canvas(H, title, desc) {
  const o = [];
  const api = {
    o,
    rect: (x, y, w, h, cls = 'neutral', r = 6) =>
      o.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" class="${cls}"/>`),
    text: (x, y, s, cls = 'tx', anchor = 'start') =>
      o.push(`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${esc(s)}</text>`),
    arrow: (x1, y1, x2, y2, cls = 'main') => o.push(`<path d="M${x1} ${y1} L${x2} ${y2}" class="${cls}"/>`),
  };
  api.cells = (x, y, labels, cw = 30, ch = 20) => {
    labels.forEach((lab, i) => {
      api.rect(x + i * (cw + 3), y, cw, ch, lab.startsWith('u') ? 'cellu' : 'cell', 3);
      api.text(x + i * (cw + 3) + cw / 2, y + 14, lab, 'mono', 'middle');
    });
    return x + labels.length * (cw + 3);
  };
  api.strip = (x, y, maxW, phases) => {
    const size = FONT.sm;
    const natural = phases.map(([lab]) => textWidth(lab, size));
    const spare = maxW - sum(natural) - 4 * (phases.length - 1);
    if (spare < 8 * phases.length) throw new Error(`timeline too long: ${phases.map(([l]) => l).join('|')}`);
    const ws = natural.map((w) => w + spare / phases.length);
    let cx = x;
    const xs = ws.map((w) => {
      const r = cx;
      cx += w + 4;
      return r;
    });
    const p0 = phases.findIndex(([, p]) => p);
    const p1 = phases.length - 1 - [...phases].reverse().findIndex(([, p]) => p);
    api.rect(xs[p0] - 3, y - 16, xs[p1] + ws[p1] - xs[p0] + 6, 44, 'win', 4);
    api.text(xs[p0], y - 5, '服务暂停窗口', 'sm');
    phases.forEach(([lab], i) => {
      api.rect(xs[i], y, ws[i], 22, 'neutral', 4);
      api.text(xs[i] + ws[i] / 2, y + 15, lab, 'sm', 'middle');
    });
  };
  o.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="title desc">`);
  o.push(`<title id="title">${esc(title)}</title>`);
  o.push(`<desc id="desc">${esc(desc)}</desc>`);
  o.push('<defs><marker id="arrowMain" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#2563EB"/></marker><marker id="arrowAux" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#AEB6C2"/></marker></defs>');
  o.push(`<style>${STYLE}</style><rect width="${W}" height="${H}" fill="white"/>`);
  api.done = () => {
    o.push('</svg>');
    return o.join('\n');
  };
  return api;
}

// ---------- 图 1：共同前半段 ----------
function renderCommon(m, cfg = CFG) {
  const H = 416;
  const c = canvas(
    H,
    'slime 权重同步：共同前半段把训练 TP 分片还原成 HF 张量并分桶',
    'GLU linear_fc1 的两个训练 TP 分片经 all-gather 与 GLU 重排拼成完整张量，再由 convert_qwen2_to_hf 拆成 gate_proj 与 up_proj；SGLang 加载器按推理 tp_rank 截取（上游契约）；下方是三种分桶规则。',
  );
  const f = m.fc1;
  c.text(24, 34, '共同前半段：训练 TP 分片先还原成与拓扑无关的 HF 张量，再按阈值分桶', 'ti');
  c.text(24, 56, `Qwen2 第 0 层 linear_fc1.weight：H=${cfg.hidden}、F=${cfg.ffn}、完整 [${2 * cfg.ffn},${cfg.hidden}]，行记 g0–g3 / u0–u3 · 训练 ${cfg.actorGpus} 卡 TP=${cfg.trainTp}、DP=${cfg.trainDp} · 阈值 ${cfg.bufferMiB} MiB`, 'su');

  c.rect(24, 70, 1132, 292, 'panel', 10);
  c.text(40, 94, '① 重组与转换（四条数据面都先走这一步）', 'pt');
  c.text(40, 116, '训练分片 = [gate_t ; up_t]', 'sm');
  c.text(40, 138, 'TP 0', 'sm');
  c.cells(72, 124, f.shards[0]);
  c.text(40, 168, 'TP 1', 'sm');
  c.cells(72, 154, f.shards[1]);
  c.arrow(210, 150, 246, 150);
  c.text(252, 128, 'all_gather_param：chunk(2)，gate 半片在前', 'sm');
  c.cells(252, 140, f.gathered);
  c.rect(252, 176, 262, 38, 'acc2', 5);
  c.text(262, 192, `直接 cat：${f.naiveCat.join(' ')}`, 'sm');
  c.text(262, 207, `→ gate_proj 得 ${f.naiveGate.join(' ')} ✗`, 'sm');
  c.arrow(520, 150, 556, 150);
  c.text(562, 118, 'gate_proj [4,2]', 'sm');
  c.cells(562, 124, f.gate);
  c.cells(562, 154, f.up);
  c.text(562, 190, 'up_proj [4,2]', 'sm');
  c.text(562, 210, 'convert_qwen2_to_hf：chunk(2)', 'sm');
  c.arrow(700, 150, 740, 150, 'aux');
  c.rect(746, 90, 396, 124, 'dep', 8);
  c.text(758, 108, 'SGLang 列并行 loader（上游契约）', 'sm');
  c.text(758, 128, '推理 TP=4：rank r 取 [g_r ; u_r]', 'tx');
  f.infer[4].forEach((rows, r) => c.cells(758 + r * 96, 136, rows, 26, 20));
  c.text(758, 176, '推理 TP=2 rank 0 取 [g0 g1 ; u0 u1]；TP=1 取全部', 'tx');
  c.text(758, 198, 'slime 只交付完整张量的名字、形状与 dtype', 'sm');
  c.rect(746, 222, 396, 38, 'acc2', 5);
  c.text(758, 238, `训练 TP 0 的 [${f.shards[0].join(' ')}] 直拷给推理 TP=4 rank 0 ✗`, 'sm');
  c.text(758, 253, `它要 [${f.infer[4][0].join(' ')}]；推理 TP=2 时相等只是巧合`, 'sm');

  const by = 296;
  const px = 0.36;
  c.text(40, by - 8, '非专家桶（NCCL / 增量）：按转换后 chunk 字节', 'sm');
  let cx = 40;
  m.buckets.nonExpert.forEach((b) => {
    const w = sum(b) * px;
    c.rect(cx, by, w, 22, 'neutral', 3);
    c.text(cx + w / 2, by + 15, b.join('+'), 'sm', 'middle');
    cx += w + 6;
  });
  const bigW = cfg.bigChunkMiB * px;
  c.rect(cx + 8, by, bigW, 22, 'ghost', 3);
  c.text(cx + 8 + bigW / 2, by + 15, `${cfg.bigChunkMiB} 独占一桶`, 'sm', 'middle');
  c.text(40, by + 42, '已有内容且会超限才换桶，阈值不是显存上限', 'sm');
  const ex = 620;
  c.text(ex, by - 8, `专家桶（NCCL / 增量）：(已攒 + 新参数) × EP=${cfg.trainEp} > ${cfg.bufferMiB} 先 EP 聚合`, 'sm');
  let ecx = ex;
  m.buckets.expert.forEach((b) => {
    const w = b.gathered * px * 0.5;
    c.rect(ecx, by, w, 22, 'neutral', 3);
    c.text(ecx + w / 2, by + 15, `${b.local.join('+')}→${b.gathered}`, 'sm', 'middle');
    ecx += w + 5;
  });
  c.text(ex, by + 42, `每批 ${m.buckets.expert[0].local.length} 个本地参数，EP 聚合后 ${m.buckets.expert[0].gathered} MiB`, 'sm');
  c.text(ex, by + 57, '张量 IPC 与整份磁盘：按 分片 × TP 预估装 ParamInfo 桶', 'sm');

  c.text(24, 386, '复现 all_gather_param / convert_qwen2_to_hf / pack_param_info_buckets / _iter_*_chunks。', 'cap');
  c.text(24, 406, BASELINE, 'su');
  return c.done();
}

// ---------- 图 2：在线数据面 ----------
function renderOnline(m, cfg = CFG) {
  const H = 700;
  const c = canvas(
    H,
    'slime 权重同步：NCCL 与共卡 CUDA IPC 两条在线数据面',
    '上半部分是训推分离的 NCCL 数据面：PP 源 rank 建临时 group、逐桶持锁发 RPC 并广播；下半部分是共卡 CUDA IPC：前缀 engine 走句柄、越界 engine 走 NCCL 补发、MoE 专家走 rank 间定向 P2P，并标出 RPC 返回前拷贝是否完成取决于镜像补丁。两条时间条都标出服务暂停窗口。',
  );
  c.text(24, 34, '在线数据面：NCCL 与共卡 CUDA IPC 都在服务暂停窗口里搬运', 'ti');
  c.text(24, 56, `同一 4 卡训练（TP=${cfg.trainTp}、DP=${cfg.trainDp}）· ② 训推分离两个 engine（TP=${cfg.ncclEngineGpuCounts.join('、TP=')}）· ③ rollout ${cfg.colocEngines.length * 2} 卡、每 engine 2 卡、actor 占槽位 0–${cfg.actorGpus - 1}`, 'su');

  // ② NCCL
  const ay = 70;
  c.rect(24, ay, 1132, 192, 'panel', 10);
  c.text(40, ay + 24, '② NCCL（训推分离）：每个 PP 源 rank 建临时 group，逐桶 lock → RPC → broadcast', 'pt');
  const ranks = [
    ['rank 0 · PP 源（DP=0、TP=0）', 'acc1'],
    ['rank 1 · 只参加 TP all-gather', 'neutral'],
    ['rank 2 · DP 副本，gather 后跳过', 'ghost'],
    ['rank 3 · DP 副本，gather 后跳过', 'ghost'],
  ];
  ranks.forEach(([label, cls], i) => {
    c.rect(40, ay + 36 + i * 26, 206, 22, cls, 4);
    c.text(48, ay + 51 + i * 26, label, 'sm');
  });
  c.arrow(248, ay + 47, 300, ay + 47);
  c.rect(304, ay + 34, 300, 100, 'acc1', 6);
  c.text(316, ay + 54, `group "slime-pp_0"：world = ${cfg.ncclEngineGpuCounts.join(' + ')} + 1 = ${m.nccl.world}`, 'tx');
  c.text(316, ay + 74, `engine A（TP=${cfg.ncclEngineGpuCounts[0]}）rank_offset = ${m.nccl.rankOffsets[0]}`, 'tx');
  c.text(316, ay + 92, `engine B（TP=${cfg.ncclEngineGpuCounts[1]}）rank_offset = ${m.nccl.rankOffsets[1]}`, 'tx');
  c.text(316, ay + 118, '训练侧只占 rank 0；两边原有并行组不合并', 'sm');
  c.rect(620, ay + 34, 522, 100, 'neutral', 6);
  c.text(632, ay + 54, '每个桶（持 RolloutManager 的 Ray 锁，防多个 PP 源交错广播）', 'tx');
  c.text(632, ay + 76, '① Lock.acquire　② RPC：names / dtypes / shapes / weight_version', 'sm');
  c.text(632, ay + 94, '③ dist.broadcast(src=0) → wait　④ ray.get → 清桶 → Lock.release', 'sm');
  c.text(632, ay + 118, '非专家、专家两遍之后各一次 Gloo barrier；SGLang 端 NCCL 收入自己的缓冲', 'sm');
  c.strip(40, ay + 158, 1100, PLANES.nccl);

  // ③ 共卡 IPC + MoE
  const by3 = 274;
  c.rect(24, by3, 1132, 372, 'panel', 10);
  c.text(40, by3 + 24, '③ 共卡 CUDA IPC：每张卡持有完整桶，只把句柄交给同卡 SGLang rank；越界 engine 走 NCCL 补发', 'pt');
  const slotW = 52;
  range(6).forEach((s) => {
    const inActor = s < cfg.actorGpus;
    c.rect(40 + s * (slotW + 4), by3 + 40, slotW, 26, inActor ? 'cell' : 'ghost', 4);
    c.text(40 + s * (slotW + 4) + slotW / 2, by3 + 57, `槽位 ${s}`, 'sm', 'middle');
  });
  cfg.colocEngines.forEach((e, i) => {
    const x = 40 + e.offset * (slotW + 4);
    const w = e.count * slotW + (e.count - 1) * 4;
    const isPrefix = i < m.coloc.prefixCount;
    c.rect(x, by3 + 72, w, 22, isPrefix ? 'acc1' : 'acc2', 4);
    c.text(x + w / 2, by3 + 87, `e${i} [${e.offset},${e.offset + e.count})${isPrefix ? ' 前缀' : ' 越界'}`, 'sm', 'middle');
  });
  m.coloc.gatherGroups.forEach((g, i) => {
    c.text(40, by3 + 116 + i * 18, `e${i}：Gloo 组 {${g.ranks.join(',')}}，源 rank ${g.src} 收 ${g.ranks.length} 个描述符 → 1 次 RPC`, 'sm');
  });
  const s = m.coloc.suffix;
  const e2 = cfg.colocEngines[m.coloc.prefixCount];
  c.text(40, by3 + 152, `e2：${e2.offset} + ${e2.count} > ${cfg.actorGpus} → 越界后缀，NCCL group "slime"，world ${s.world}`, 'sm');
  c.rect(40, by3 + 162, 490, 24, 'acc2', 5);
  c.text(50, by3 + 178, `pause / flush / 量化前后处理只发给 e${m.coloc.pauseTargets.join('、e')}；越界 e2 不在名单里`, 'sm');
  c.rect(40, by3 + 196, 490, 100, 'dep', 8);
  c.text(52, by3 + 214, 'RPC 返回时 SGLang 的 GPU 拷贝是否已完成（依赖侧）', 'sm');
  c.text(52, by3 + 234, `补丁镜像：${PATCH_LABELS.ipcSync} 在 load 后 synchronize → 已完成`, 'sm');
  c.text(52, by3 + 252, '上游 v0.5.15.post1：load_weights 排入拷贝即返回 → 不保证', 'sm');
  c.text(52, by3 + 270, 'slime：每桶新建 flattened tensor，del 后 ipc_collect / empty_cache', 'sm');
  c.text(52, by3 + 288, '块何时被复用取决于 PyTorch CUDA IPC 引用计数（未核）', 'sm');

  const mx = 560;
  const r = m.moe;
  c.rect(mx, by3 + 40, 582, 256, 'neutral', 8);
  c.text(mx + 12, by3 + 60, `MoE 定向路由：同 4 卡改成 TP=1、EP=${cfg.moe.trainEp}、${cfg.moe.numExperts} 专家；两个共卡 engine 各 EP=${cfg.moe.inferEp}`, 'tx');
  r.owners.forEach((owned, i) => {
    c.rect(mx + 12 + i * 140, by3 + 72, 130, 22, 'cell', 4);
    c.text(mx + 12 + i * 140 + 65, by3 + 87, `训练 rank ${i}：e${owned.join(' e')}`, 'sm', 'middle');
  });
  r.targets.forEach((t, i) => {
    c.rect(mx + 12 + i * 282, by3 + 128, 272, 22, 'acc1', 4);
    const lo = i * (cfg.moe.numExperts / cfg.moe.inferEp);
    c.text(mx + 12 + i * 282 + 136, by3 + 143, `推理 EP 分片 ${i}（e${lo}–e${lo + 3}）→ rank ${t.join('、')}`, 'sm', 'middle');
  });
  for (const t of r.transfers) {
    for (const tr of t.targets) {
      if (tr === t.src) continue;
      const x1 = mx + 12 + t.src * 140 + 65;
      const tIdx = r.targets.findIndex((tt) => tt.includes(tr));
      const x2 = mx + 12 + tIdx * 282 + 136 + (tr > 1 ? 40 : -40);
      c.o.push(`<path d="M${x1} ${by3 + 95} L${x2} ${by3 + 127}" class="aux"/>`);
    }
  }
  c.text(mx + 12, by3 + 174, `P2P 发送 ${r.sends} 份专家包（目标即持有者时不发）；通用路径 EP 广播 ${r.genericDeliveries} 份`, 'sm');
  c.text(mx + 12, by3 + 192, `每卡转换并交付：定向 ${r.routedPerRank[0]} 个专家，通用路径 ${r.genericPerRank} 个`, 'sm');
  c.text(mx + 12, by3 + 210, `装批（每 rank staging ≤ ${cfg.moe.bufferBundles} 包）：${r.batches.map((b, i) => `batch${i} = e${b[0]}–e${b[b.length - 1]}`).join('，')}`, 'sm');
  c.text(mx + 12, by3 + 236, '准入：推理 PP=1、EP>1、两侧专家 TP=1、无 EPLB 与冗余专家、全部共卡', 'sm');
  c.text(mx + 12, by3 + 254, '任一不满足 → 记日志退回通用桶', 'sm');
  c.text(mx + 12, by3 + 272, '只匹配 decoder 层 routed expert 的 linear_fc1 / linear_fc2', 'sm');
  c.strip(40, by3 + 334, 1100, PLANES.ipc);

  c.text(24, 668, '* 为条件步骤（compressed-tensors 量化、定向路由）。浅蓝底 = 服务暂停窗口；虚线框 = 依赖侧。复现 connect_rollout_engines / expert_routing。', 'cap');
  c.text(24, 688, BASELINE, 'su');
  return c.done();
}

// ---------- 图 3：磁盘数据面 ----------
function renderDisk(m, cfg = CFG) {
  const H = 660;
  const c = canvas(
    H,
    'slime 权重同步：整份磁盘与增量磁盘两条数据面',
    '上半部分是整份磁盘：actor 写版本目录与 hook，RayTrainGroup 可选地让 engine 经补丁端点 pull 到本地，再在暂停窗口里 reload 并在 CI 下读回版本；下半部分是增量磁盘：同一段 bf16 字节的 xor 与 overwrite 编码、主机侧 pull 的版本链与首次、重启两个失败边界。',
  );
  const d = m.delta;
  c.text(24, 34, '磁盘数据面：写盘与主机侧 pull 在暂停之前，暂停窗口只覆盖 reload', 'ti');
  c.text(24, 56, `④ 整份：版本 1 写 weight_v000001/ · ⑤ 增量：gate_proj 前 4 个 bf16 元素 ${cfg.delta.old.join('、')} → ${cfg.delta.neu.join('、')}`, 'su');

  const cy = 70;
  c.rect(24, cy, 1132, 192, 'panel', 10);
  c.text(40, cy + 24, '④ 整份磁盘：actor 写完整 HF checkpoint，RayTrainGroup 在暂停窗口里 reload', 'pt');
  const diskBoxes = [
    ['weight_v000001/', 'rank 0 先 rmtree', '各 rank 自己 mkdir'],
    ['HfWeightIteratorDirect', '每节点 writer rank', '转换并写 safetensors'],
    ['Gloo barrier ×2', '两次之间跑', 'post-write hook'],
    ['pull_weights(1)*', '拉到主机本地盘', PATCH_LABELS.pull],
    ['pause→flush→reload', 'update_weights_from_disk', '(path, "1")'],
    ['CI：读回须为 "1"', '/get_weight_version', `由 ${PATCH_LABELS.versionReadback} 改写`],
  ];
  diskBoxes.forEach(([a, b, t], i) => {
    const x = 40 + i * 186;
    const cls = i === 4 ? 'acc1' : i === 3 || i === 5 ? 'dep' : 'neutral';
    c.rect(x, cy + 38, 176, 64, cls, 5);
    c.text(x + 8, cy + 56, a, 'tx');
    c.text(x + 8, cy + 74, b, 'sm');
    c.text(x + 8, cy + 90, t, 'sm');
    if (i < diskBoxes.length - 1) c.arrow(x + 176, cy + 70, x + 186, cy + 70);
  });
  c.text(40, cy + 124, 'actor 的 weight_version 与 RayTrainGroup._disk_weight_version 是两个计数器，create 时回写 update_weight_start_version', 'sm');
  c.strip(40, cy + 158, 1100, PLANES.disk);

  const dy = 274;
  c.rect(24, dy, 1132, 332, 'panel', 10);
  c.text(40, dy + 24, '⑤ 增量磁盘：与上一版字节做差，发布压缩差分，主机在本地 checkpoint 上原位应用', 'pt');
  c.text(40, dy + 46, '按小端字节：', 'sm');
  const rowsB = [
    ['旧', d.oldB, 'cell'],
    ['新', d.newB, 'cell'],
    ['xor', d.xor, 'ghost'],
  ];
  rowsB.forEach(([lab, bytes, cls], i) => {
    c.text(40, dy + 70 + i * 26, lab, 'sm');
    bytes.forEach((b, j) => {
      const hot = d.xor[j] !== 0;
      c.rect(72 + j * 36, dy + 56 + i * 26, 32, 20, hot ? 'acc1' : cls, 3);
      c.text(72 + j * 36 + 16, dy + 70 + i * 26, hex(b), 'mono', 'middle');
    });
  });
  c.text(400, dy + 70, `变化 ${d.changed}/${d.total} 字节`, 'tx');
  c.text(400, dy + 90, `perf/update_weights_density = ${(d.density * 100).toFixed(0)}%`, 'tx');
  c.text(400, dy + 110, '整张量没有变化就不写入分片', 'sm');
  c.rect(40, dy + 144, 370, 40, 'neutral', 5);
  c.text(50, dy + 160, `xor：差分 ${d.xorLen} 字节（压缩前），未变字节为 0`, 'sm');
  c.text(50, dy + 176, '对合：在已应用状态上再应用一次会还原成旧值', 'sm');
  c.rect(420, dy + 144, 370, 40, 'neutral', 5);
  c.text(430, dy + 160, `overwrite：4 + ${d.changed}×4 + ${d.changed} = ${d.overwriteLen} 字节`, 'sm');
  c.text(430, dy + 176, '计数、位置、新值；更大，但重复应用幂等', 'sm');
  c.rect(800, dy + 40, 342, 180, 'dep', 8);
  c.text(812, dy + 58, `主机侧 pull（${PATCH_LABELS.pull}）`, 'sm');
  c.text(812, dy + 78, 'v0 = model_path（缺省即 --hf-checkpoint）', 'tx');
  c.text(812, dy + 96, 'v1 base 0；v2 base 1', 'tx');
  c.text(812, dy + 116, `新主机 pull(2)：${m.pulls.fresh.ops.join(' → ')}`, 'sm');
  c.text(812, dy + 134, `本地已在 1：${m.pulls.atOne.ops.join(' → ')}`, 'sm');
  c.text(812, dy + 152, '逐张量解压 → 原位 xor 或覆写 → 校验 checksum', 'sm');
  c.text(812, dy + 170, 'base 不符抛 out-of-order；checksum 不符抛错', 'sm');
  c.text(812, dy + 188, 'host flock 合并同机多 rank', 'sm');
  c.text(812, dy + 206, 'TP 组内所有 host 都成功才返回成功', 'sm');
  c.rect(40, dy + 230, 1102, 40, 'acc2', 5);
  c.text(50, dy + 246, '首次 update 只捕获快照并 pull_weights(0)、不发布：第一轮 rollout 用的是 --hf-checkpoint，而不是 --load 装入的 actor', 'sm');
  c.text(50, dy + 262, `重启复用本地目录：残留标记 57 让 pull(1) ${m.pulls.staleDelta.ops.length ? '执行' : '什么也不做'}，reload 的仍是上次运行的 v57，版本号却写成 "1"`, 'sm');
  c.strip(40, dy + 296, 1100, PLANES.delta);

  c.text(24, 628, '* 为条件步骤（设了本地目录、开 --ci-test）。浅蓝底 = 服务暂停窗口；虚线框 = 依赖侧（端点来自镜像补丁）。复现 _encode_delta / overwrite_encode / local_checkpoint.pull。', 'cap');
  c.text(24, 648, BASELINE, 'su');
  return c.done();
}

export const FIGURES = Object.freeze({
  'slime_weight_sync_common.svg': renderCommon,
  'slime_weight_sync_online.svg': renderOnline,
  'slime_weight_sync_disk.svg': renderDisk,
});

export function renderAll() {
  const m = model();
  return Object.fromEntries(Object.entries(FIGURES).map(([name, fn]) => [name, `${fn(m)}\n`]));
}

const here = dirname(fileURLToPath(import.meta.url));
const defaultOutput = join(here, '..', '..', '..', 'wiki', '02_engineering', '04_posttrain_frameworks', 'slime', 'assets');
const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  const outputDir = process.argv[2] ? process.argv[2] : defaultOutput;
  mkdirSync(outputDir, { recursive: true });
  for (const [name, svg] of Object.entries(renderAll())) writeFileSync(join(outputDir, name), svg, 'utf8');
}
