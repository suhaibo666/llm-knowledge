// 图：同一个最小实例（Qwen2 第 0 层 GLU linear_fc1.weight，H=2、F=4；训练 4 卡 TP=2、DP=2）
// 怎样先还原成与拓扑无关的 HF 张量，再分别经 NCCL、共卡 CUDA IPC（含 MoE 定向路由与越界 engine）、
// 整份磁盘、增量磁盘四条数据面搬运，并在各自的 pause 窗口里提交。
// 源码基线：THUDM/slime@681b3adca54105d5ecd3fb822fa0dc58a427e0f9。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
// 要讲清楚：训练 TP 分片不是推理侧能直接装的形状；all_gather_param 先把各片 chunk(2) 再按
// [各片 gate…, 各片 up…] 拼接，convert_qwen2_to_hf 再 chunk(2) 成 gate_proj / up_proj，SGLang 加载器
// 按自己的 tp_rank 切片（依赖侧契约）。四条数据面共用这一步，差别在搬运载体、谁发送、以及搬运落在
// pause 窗口之内（NCCL、IPC）还是之前（两条磁盘路径）。
//
// 布局：顶部横带 ① 画共同前半段与三种分桶规则；下方四条泳道 ②–⑤ 各自一张示意 + 一条按源码顺序的
// 时间条，时间条上用浅蓝底标出服务暂停窗口。acc1 标决定性映射与暂停窗口，acc2 标错误做法与失败边界。
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

// update_weight/common.py::all_gather_param / all_gather_params_async 的 linear_fc1 分支，
// 与 megatron_to_hf/qwen2.py::convert_qwen2_to_hf 的 mlp.linear_fc1.weight 分支。
// SGLang 端的按 tp_rank 切片是依赖侧契约（sgl-project/sglang@0b3bb0cb 的 linear.py），本机未复核。
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

// update_weight_from_distributed.py::_iter_non_expert_chunks 与
// hf_weight_iterator_direct.py::pack_param_info_buckets 共用的顺序装桶：已有内容且会超限才换桶。
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

// update_weight_from_distributed.py::_iter_expert_chunks：阈值按 EP 聚合后的体量判断，
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

// expert_routing.py::_get_expert_target_ranks / _build_expert_params / _build_expert_transfer_plan /
// _pack_expert_transfer_batches，单层、TP=1、expert-DP=1（每个专家只有一个物理持有者）。
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

// update_weight_from_disk_delta.py::_encode_delta 与 slime/utils/disk_delta.py::overwrite_encode
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

// docker/patch/latest/sglang-pull_weights.patch 里 weight_sync/local_checkpoint.py::pull 的版本逻辑
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
// 估算文字宽度：CJK 与全角按字号计，其余按 0.56 字号计
export function textWidth(s, size) {
  let w = 0;
  for (const ch of String(s)) w += /[ -⯿⺀-￯]/.test(ch) ? size : size * 0.58;
  return w;
}
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

function render(m, cfg = CFG) {
  const W = 1180;
  const H = 1484;
  const o = [];
  const rect = (x, y, w, h, cls = 'neutral', r = 6) =>
    o.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" class="${cls}"/>`);
  const text = (x, y, s, cls = 'tx', anchor = 'start') =>
    o.push(`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${esc(s)}</text>`);
  const arrow = (x1, y1, x2, y2, cls = 'main') => o.push(`<path d="M${x1} ${y1} L${x2} ${y2}" class="${cls}"/>`);
  const cells = (x, y, labels, cw = 30, ch = 20) => {
    labels.forEach((lab, i) => {
      rect(x + i * (cw + 3), y, cw, ch, lab.startsWith('u') ? 'cellu' : 'cell', 3);
      text(x + i * (cw + 3) + cw / 2, y + 14, lab, 'mono', 'middle');
    });
    return x + labels.length * (cw + 3);
  };
  const strip = (x, y, maxW, phases) => {
    const size = 10.5;
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
    rect(xs[p0] - 3, y - 16, xs[p1] + ws[p1] - xs[p0] + 6, 44, 'win', 4);
    text(xs[p0], y - 5, '服务暂停窗口', 'sm');
    phases.forEach(([lab], i) => {
      rect(xs[i], y, ws[i], 22, 'neutral', 4);
      text(xs[i] + ws[i] / 2, y + 15, lab, 'sm', 'middle');
    });
  };

  o.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="title desc">`);
  o.push('<title id="title">slime 权重同步：同一最小实例在四条数据面上的重组、搬运与提交窗口</title>');
  o.push('<desc id="desc">顶部横带展示 GLU fc1 从训练 TP 分片经 all-gather、GLU 重排与 HF 转换得到 gate_proj 与 up_proj，以及三种分桶规则；下方四条泳道分别展示 NCCL、共卡 CUDA IPC 与 MoE 定向路由、整份磁盘、增量磁盘的搬运方式和服务暂停窗口的位置。</desc>');
  o.push('<defs><marker id="arrowMain" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#2563EB"/></marker><marker id="arrowAux" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#AEB6C2"/></marker></defs>');
  o.push(`<style>${STYLE}</style><rect width="${W}" height="${H}" fill="white"/>`);

  text(24, 34, '同一个最小实例走四条数据面：先还原成 HF 张量，再按载体搬运，在各自的暂停窗口里提交', 'ti');
  text(24, 56, `Qwen2 第 0 层 GLU linear_fc1.weight：H=${cfg.hidden}、F=${cfg.ffn}、完整 [${2 * cfg.ffn},${cfg.hidden}]，行记 g0–g3 / u0–u3 · 训练 ${cfg.actorGpus} 卡 TP=${cfg.trainTp}、DP=${cfg.trainDp} · --update-weight-buffer-size ${cfg.bufferMiB} MiB`, 'su');

  // ---------- ① 共同前半段 ----------
  const f = m.fc1;
  rect(24, 70, 1132, 282, 'panel', 10);
  text(40, 94, '① 共同前半段：TP 分片 → all-gather + GLU 重排 → HF 命名张量（四条数据面都先走这一步）', 'pt');
  text(40, 116, '训练侧：每片按 Megatron GLU 布局 = [gate_t ; up_t]', 'sm');
  text(40, 138, 'TP 0', 'sm');
  cells(72, 124, f.shards[0]);
  text(40, 168, 'TP 1', 'sm');
  cells(72, 154, f.shards[1]);
  arrow(210, 150, 246, 150);
  text(214, 196, 'all_gather：各片 chunk(2)，', 'sm');
  text(214, 210, '按 [各片 gate…, 各片 up…] 拼接', 'sm');
  cells(252, 140, f.gathered);
  rect(252, 222, 262, 38, 'acc2', 5);
  text(262, 238, `若不重排直接 cat：${f.naiveCat.join(' ')}`, 'sm');
  text(262, 253, `→ gate_proj 会拿到 ${f.naiveGate.join(' ')}，混进 up 行 ✗`, 'sm');
  arrow(520, 150, 556, 150);
  text(562, 214, 'convert_qwen2_to_hf：', 'sm');
  text(562, 228, 'chunk(2) 成两个 HF 名', 'sm');
  text(562, 118, 'gate_proj [4,2]', 'sm');
  cells(562, 124, f.gate);
  cells(562, 154, f.up);
  text(562, 190, 'up_proj [4,2]', 'sm');
  arrow(700, 150, 740, 150, 'aux');
  rect(746, 104, 396, 126, 'dep', 8);
  text(758, 122, 'SGLang 加载器（依赖侧契约，本机无该 checkout）', 'sm');
  text(758, 140, '推理 TP=4：rank r 取 [g_r ; u_r]', 'tx');
  f.infer[4].forEach((rows, r) => cells(758 + r * 96, 148, rows, 26, 20));
  text(758, 188, '推理 TP=2 rank 0 取 [g0 g1 ; u0 u1]；TP=1 取全部', 'tx');
  text(758, 206, '四条数据面交付的都是完整张量，切片发生在加载器里', 'sm');
  text(758, 222, '（slime 源码只证明交付的名字、形状与 dtype）', 'sm');
  rect(746, 236, 396, 40, 'acc2', 5);
  text(758, 252, `把训练 TP 0 的 [${f.shards[0].join(' ')}] 直接拷给推理 TP=4 rank 0 ✗`, 'sm');
  text(758, 267, `它要的是 [${f.infer[4][0].join(' ')}]；推理 TP=2 时恰好相等只是布局巧合`, 'sm');

  // 分桶
  const bx = 40;
  const by = 292;
  const px = 0.36;
  text(bx, by - 6, `分桶（阈值 ${cfg.bufferMiB} MiB，不是显存上限）`, 'sm');
  let cx = bx;
  m.buckets.nonExpert.forEach((b, i) => {
    const w = sum(b) * px;
    rect(cx, by, w, 22, 'neutral', 3);
    text(cx + w / 2, by + 15, b.join('+'), 'sm', 'middle');
    cx += w + 6;
  });
  const bigW = cfg.bigChunkMiB * px;
  rect(cx + 8, by, bigW, 22, 'ghost', 3);
  text(cx + 8 + bigW / 2, by + 15, `${cfg.bigChunkMiB} 独占一桶`, 'sm', 'middle');
  text(bx, by + 40, 'NCCL 非专家：按转换后 chunk 字节顺序装，已有内容且会超限才换桶', 'sm');
  const ex = 620;
  text(ex, by - 6, `NCCL 专家：(已攒 + 新参数) × EP=${cfg.trainEp} > ${cfg.bufferMiB} 就先 EP 聚合`, 'sm');
  let ecx = ex;
  m.buckets.expert.forEach((b) => {
    const w = b.gathered * px * 0.5;
    rect(ecx, by, w, 22, 'neutral', 3);
    text(ecx + w / 2, by + 15, `${b.local.join('+')}→${b.gathered}`, 'sm', 'middle');
    ecx += w + 5;
  });
  text(ex, by + 40, `每批只装 ${m.buckets.expert[0].local.length} 个本地参数，EP 聚合后 ${m.buckets.expert[0].gathered} MiB`, 'sm');
  text(ex, by + 54, '张量路径初始化时、整份磁盘每次同步时按 分片×TP 预估装桶', 'sm');

  // ---------- ② NCCL ----------
  const ay = 364;
  rect(24, ay, 1132, 190, 'panel', 10);
  text(40, ay + 24, '② NCCL（训推分离）：每个 PP 源 rank 建一个临时 group，逐桶 lock → RPC → broadcast', 'pt');
  const ranks = [
    ['rank 0', 'PP 源：DP=0、TP=0', 'acc1'],
    ['rank 1', '只参加 TP all-gather', 'neutral'],
    ['rank 2', 'DP 副本：gather 后跳过', 'ghost'],
    ['rank 3', 'DP 副本：gather 后跳过', 'ghost'],
  ];
  ranks.forEach(([a, b, cls], i) => {
    rect(40, ay + 36 + i * 26, 206, 22, cls, 4);
    text(48, ay + 51 + i * 26, `${a} · ${b}`, 'sm');
  });
  arrow(248, ay + 47, 300, ay + 47);
  rect(304, ay + 34, 300, 100, 'acc1', 6);
  text(316, ay + 54, `group "slime-pp_0"：world = ${cfg.ncclEngineGpuCounts.join(' + ')} + 1 = ${m.nccl.world}`, 'tx');
  text(316, ay + 74, `engine A（TP=${cfg.ncclEngineGpuCounts[0]}）rank_offset = ${m.nccl.rankOffsets[0]}`, 'tx');
  text(316, ay + 92, `engine B（TP=${cfg.ncclEngineGpuCounts[1]}）rank_offset = ${m.nccl.rankOffsets[1]}`, 'tx');
  text(316, ay + 112, '训练侧只占 rank 0；两边原有并行组都不合并', 'sm');
  text(316, ay + 126, 'sleep 或重连时销毁、下次再建', 'sm');
  rect(620, ay + 34, 522, 100, 'neutral', 6);
  text(632, ay + 54, '每个桶：', 'tx');
  text(632, ay + 72, '① Lock.acquire（RolloutManager 的 Ray 锁，防多个 PP 源交错广播死锁）', 'sm');
  text(632, ay + 88, '② Ray RPC 发 names / dtypes / shapes / weight_version', 'sm');
  text(632, ay + 104, '③ 每个张量 dist.broadcast(src=0, async) → wait', 'sm');
  text(632, ay + 120, '④ ray.get(RPC) → 清桶 → Lock.release；非专家、专家两遍之后各一次 Gloo barrier', 'sm');
  strip(40, ay + 158, 1100, PLANES.nccl);

  // ---------- ③ 共卡 IPC + MoE ----------
  const by3 = 566;
  rect(24, by3, 1132, 344, 'panel', 10);
  text(40, by3 + 24, '③ 共卡 CUDA IPC：每张卡持有完整桶，只把句柄交给同卡 SGLang rank；越界 engine 走 NCCL 补发', 'pt');
  text(40, by3 + 46, `colocate、rollout ${cfg.colocEngines.length * 2} 卡、每 engine 2 卡；actor 只占槽位 0–${cfg.actorGpus - 1}`, 'sm');
  const slotW = 52;
  range(6).forEach((s) => {
    const inActor = s < cfg.actorGpus;
    rect(40 + s * (slotW + 4), by3 + 56, slotW, 26, inActor ? 'cell' : 'ghost', 4);
    text(40 + s * (slotW + 4) + slotW / 2, by3 + 73, `槽位 ${s}`, 'sm', 'middle');
  });
  cfg.colocEngines.forEach((e, i) => {
    const x = 40 + e.offset * (slotW + 4);
    const w = e.count * slotW + (e.count - 1) * 4;
    const isPrefix = i < m.coloc.prefixCount;
    rect(x, by3 + 88, w, 22, isPrefix ? 'acc1' : 'acc2', 4);
    text(x + w / 2, by3 + 103, `e${i} [${e.offset},${e.offset + e.count})${isPrefix ? ' 前缀' : ' 越界'}`, 'sm', 'middle');
  });
  m.coloc.gatherGroups.forEach((g, i) => {
    text(40, by3 + 132 + i * 18, `e${i}：Gloo 组 {${g.ranks.join(',')}}，源 rank ${g.src} 收 ${g.ranks.length} 个描述符 → 1 次 RPC，tp_rank k 取第 k 个`, 'sm');
  });
  const s = m.coloc.suffix;
  text(40, by3 + 170, `e2：offset 4 + 2 > ${cfg.actorGpus} → 越界后缀，NCCL group "slime"，源 = DP=TP=PP=0 的 rank 0，world ${s.world}`, 'sm');
  rect(40, by3 + 180, 330, 36, 'acc2', 5);
  text(50, by3 + 195, `pause / flush / 量化前后处理只发给 e${m.coloc.pauseTargets.join('、e')}`, 'sm');
  text(50, by3 + 209, '越界 e2 靠组级 offload 的 flush 兜底，否则无人清缓存', 'sm');
  text(40, by3 + 236, '每桶新建 flattened bucket（不复用：SGLang 返回时拷贝可能仍在 GPU 上排队）', 'sm');
  text(40, by3 + 252, '源 rank ray.get 后 del → ipc_collect / empty_cache；非源 rank 没有 ref，', 'sm');
  text(40, by3 + 268, '块何时可复用取决于 PyTorch IPC 引用计数（依赖侧）；两遍结束后 Gloo barrier 再清一次', 'sm');

  // MoE 子面板
  const mx = 560;
  const r = m.moe;
  rect(mx, by3 + 40, 582, 238, 'neutral', 8);
  text(mx + 12, by3 + 60, `MoE 定向路由：同 4 卡改成 TP=1、EP=${cfg.moe.trainEp}、${cfg.moe.numExperts} 专家；两个共卡 engine 各 EP=${cfg.moe.inferEp}`, 'tx');
  r.owners.forEach((ex2, i) => {
    rect(mx + 12 + i * 140, by3 + 72, 130, 22, 'cell', 4);
    text(mx + 12 + i * 140 + 65, by3 + 87, `训练 rank ${i}：e${ex2.join(' e')}`, 'sm', 'middle');
  });
  r.targets.forEach((t, i) => {
    rect(mx + 12 + i * 282, by3 + 128, 272, 22, 'acc1', 4);
    const lo = i * (cfg.moe.numExperts / cfg.moe.inferEp);
    text(mx + 12 + i * 282 + 136, by3 + 143, `推理 EP 分片 ${i}（e${lo}–e${lo + 3}）→ rank ${t.join('、')}`, 'sm', 'middle');
  });
  for (const t of r.transfers) {
    for (const tr of t.targets) {
      if (tr === t.src) continue;
      const x1 = mx + 12 + t.src * 140 + 65;
      const tIdx = r.targets.findIndex((tt) => tt.includes(tr));
      const x2 = mx + 12 + tIdx * 282 + 136 + (tr > 1 ? 40 : -40);
      o.push(`<path d="M${x1} ${by3 + 95} L${x2} ${by3 + 127}" class="aux"/>`);
    }
  }
  text(mx + 12, by3 + 172, `P2P 发送 ${r.sends} 份专家包（目标即持有者时不发）；通用路径 EP 广播 ${r.genericDeliveries} 份`, 'sm');
  text(mx + 12, by3 + 188, `每卡转换并交付：定向 ${r.routedPerRank[0]} 个专家，通用路径 ${r.genericPerRank} 个（发送方另暂存自己发出的专家）`, 'sm');
  text(mx + 12, by3 + 204, `装批（每 rank staging ≤ ${cfg.moe.bufferBundles} 包）：${r.batches.map((b, i) => `batch${i} = e${b[0]}–e${b[b.length - 1]}`).join('，')}`, 'sm');
  text(mx + 12, by3 + 220, '准入：推理 PP=1、EP>1、专家 TP 两侧都为 1、无 EPLB/冗余专家，', 'sm');
  text(mx + 12, by3 + 236, '全部 engine 共卡且拓扑一致；任一不满足就记日志退回通用桶', 'sm');
  text(mx + 12, by3 + 252, '只匹配 decoder.layers.*.mlp.experts.linear_fc{1,2}.weight*', 'sm');
  text(mx + 12, by3 + 268, '（router、shared expert、MTP 与 language_model 前缀仍走通用桶）', 'sm');
  strip(40, by3 + 312, 1100, PLANES.ipc);

  // ---------- ④ 整份磁盘 ----------
  const cy = 922;
  rect(24, cy, 1132, 184, 'panel', 10);
  text(40, cy + 24, '④ 整份磁盘：写完整 HF checkpoint 在 pause 之前完成，RayTrainGroup 只在暂停窗口里 reload', 'pt');
  const diskBoxes = [
    ['weight_v000001/', 'rank 0 先 rmtree', '每个写 rank 自己 mkdir'],
    ['HfWeightIteratorDirect', '重组后由节点 writer', '写 safetensors 分片'],
    ['Gloo barrier ×2', '两次 barrier 之间跑', 'post-write hook'],
    ['pull_weights(1)*', '拉到主机本地盘', '按 .weight_sync 标记'],
    ['pause→flush→reload', 'update_weights_from_disk', '(path, "1")'],
    ['CI：get_weight_version', '逐 engine 须等于 "1"', '否则 RuntimeError'],
  ];
  diskBoxes.forEach(([a, b, c], i) => {
    const x = 40 + i * 186;
    rect(x, cy + 38, 176, 64, i === 4 ? 'acc1' : 'neutral', 5);
    text(x + 8, cy + 56, a, 'tx');
    text(x + 8, cy + 74, b, 'sm');
    text(x + 8, cy + 90, c, 'sm');
    if (i < diskBoxes.length - 1) arrow(x + 176, cy + 70, x + 186, cy + 70);
  });
  text(40, cy + 122, 'actor 端 weight_version 与 RayTrainGroup._disk_weight_version 是两个计数器，靠 create 时回写 update_weight_start_version 保持同步', 'sm');
  strip(40, cy + 152, 1100, PLANES.disk);

  // ---------- ⑤ 增量磁盘 ----------
  const dy = 1118;
  const d = m.delta;
  rect(24, dy, 1132, 318, 'panel', 10);
  text(40, dy + 24, '⑤ 增量磁盘：与上一版的字节做差，发布压缩差分，主机在本地 checkpoint 上原位应用', 'pt');
  text(40, dy + 46, `gate_proj 前 4 个 bf16（${cfg.delta.old.join('、')} → ${cfg.delta.neu.join('、')}），按小端字节：`, 'sm');
  const rowsB = [
    ['旧', d.oldB, 'cell'],
    ['新', d.newB, 'cell'],
    ['xor', d.xor, 'ghost'],
  ];
  rowsB.forEach(([lab, bytes, cls], i) => {
    text(40, dy + 70 + i * 26, lab, 'sm');
    bytes.forEach((b, j) => {
      const hot = d.xor[j] !== 0;
      rect(72 + j * 36, dy + 56 + i * 26, 32, 20, hot ? 'acc1' : cls, 3);
      text(72 + j * 36 + 16, dy + 70 + i * 26, hex(b), 'mono', 'middle');
    });
  });
  text(40, dy + 150, `变化 ${d.changed}/${d.total} 字节 → perf/update_weights_density = ${(d.density * 100).toFixed(0)}%；整张量不变则整张量不写`, 'sm');
  rect(40, dy + 160, 370, 52, 'neutral', 5);
  text(50, dy + 178, `xor：差分 ${d.xorLen} 字节（压缩前），未变字节为 0，zstd 压得最小`, 'sm');
  text(50, dy + 194, '对合：在已应用状态上再应用一次会还原成旧值', 'sm');
  text(50, dy + 208, '', 'sm');
  rect(420, dy + 160, 370, 52, 'neutral', 5);
  text(430, dy + 178, `overwrite：4 + ${d.changed}×4 + ${d.changed} = ${d.overwriteLen} 字节（计数、位置、新值）`, 'sm');
  text(430, dy + 194, '更大，但重复应用幂等', 'sm');
  rect(800, dy + 40, 342, 172, 'dep', 8);
  text(812, dy + 58, '主机侧 pull（slime 随镜像提供的 SGLang 补丁）', 'sm');
  text(812, dy + 78, 'v0 = hf_checkpoint；v1 base 0；v2 base 1', 'tx');
  text(812, dy + 98, `新主机 pull(2)：${m.pulls.fresh.ops.join(' → ')}`, 'sm');
  text(812, dy + 116, `本地已在 1：${m.pulls.atOne.ops.join(' → ')}`, 'sm');
  text(812, dy + 134, '每张量解压→原位异或/覆写→校验 checksum', 'sm');
  text(812, dy + 150, 'base_version 不符抛 out-of-order；校验不符抛错', 'sm');
  text(812, dy + 168, 'host 级 flock 把同机多 rank 合成一次', 'sm');
  text(812, dy + 186, '成功要 TP 组内所有 host 都返回成功', 'sm');
  text(812, dy + 202, '（执行的是镜像里的补丁，本机未运行）', 'sm');
  rect(40, dy + 222, 1102, 38, 'acc2', 5);
  text(50, dy + 238, `首次 update 只捕获 snapshot 并 pull_weights(0)、不发布：第一轮 rollout 用的是 hf_checkpoint，而不是 --load 装入的 actor。`, 'sm');
  text(50, dy + 253, `重启复用本地目录时，残留标记 57 让 pull(1) ${m.pulls.staleDelta.ops.length ? '执行' : '什么也不做'}，reload 的仍是上次运行的 v57，版本号却写成 "1"。`, 'sm');
  strip(40, dy + 290, 1100, PLANES.delta);

  text(24, 1454, '阅读顺序：① 共同重组（四条数据面相同）→ ②③ 在线路径：搬运落在暂停窗口里 → ④⑤ 磁盘路径：写盘与主机侧 pull 在暂停之前，窗口只覆盖 reload。* 为条件步骤。', 'cap');
  text(24, 1474, '源码基线：THUDM/slime@681b3adca541 · 复现 all_gather_param / convert_qwen2_to_hf / _iter_*_chunks / connect_rollout_engines / expert_routing / _encode_delta / local_checkpoint.pull', 'su');
  o.push('</svg>');
  return o.join('\n');
}

const here = dirname(fileURLToPath(import.meta.url));
const defaultOutput = join(here, '..', '..', '..', 'wiki', '02_engineering', '04_posttrain_frameworks', 'slime', 'assets');
const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  const outputDir = process.argv[2] ? process.argv[2] : defaultOutput;
  mkdirSync(outputDir, { recursive: true });
  writeFileSync(join(outputDir, 'slime_weight_sync_planes.svg'), `${render(model())}\n`, 'utf8');
}
