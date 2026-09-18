// 图：一个已生成 token 的 logprob 怎样在训推两侧分叉、又被 slime 的重放与校正逐层收拢：
// ① L2 支持集重放：补丁的联合截断（rank < top_k 且前缀和 ≤ top_p）让返回的保留集已含 top-k，
//    训练侧 keep mask 在同一集合上重算（含 TP 词表分片与工具 token 的空 nucleus）；
//    top_p=1、top_k≠-1 时不请求 ids，两侧同在全词表域、没有假差异，但用作 old policy/TIS 的不是行为概率；
// ② L3 MoE 路由重放的 stage 与 cursor（old logprob 前向是否单独运行是前提），
//    以及同一组 route 梯度按列顺序逐次 BF16 累加的差别；
// ③ 同一组 train-old/rollout 比值走 vanilla TIS、ICEPOP 与示例 MIS 的权重与 mask。
// 源码基线：THUDM/slime@4c193f1f37509cca70f0e88807a9305b70f63f4e；
// rollout 侧保留集与重归一来自 docker/patch/latest/sglang-top_p.patch（依赖侧补丁），
// 上游对照 sgl-project/sglang@0b3bb0cbe318（v0.5.15.post1）。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
// 要讲清楚：权重相同也会因归一化域不同产生 δ；rollout 返回的保留集由补丁按 top-k 与 top-p 联合截断，
// keep mask 把集合外置 −inf 并写回目标 logit，训练侧因此在 S ∪ {y} 上重算；top_p=1 时差异不进 δ
// 而进 old policy/TIS；R3 的记录被 old-logprob 前向（若单独运行）与训练前向消费，
// backward cursor 只在重算时前进；route 梯度必须按列顺序逐次舍入；校正函数改的是权重或 mask，
// 不是证据。acc1 标重放后的一致结果，acc2 标假差异、被掩盖的差异与被拒绝的 token；
// 依赖侧（补丁、上游 SGLang、DeepEP fork）用虚线框或文字标出。
//
// 用法：node tools/figs/svg/slime_train_infer_consistency_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// ---------------- 冻结的示例输入 ----------------
export const CFG = Object.freeze({
  logits: [2.0, 1.0, 0.0, -1.0], // 同一份权重下该位置的 logits（4 个 token 的词表，已按降序排列）
  temperature: 1.0, // --rollout-temperature（解析期要求 > 0）
  topP: 0.9, // --rollout-top-p（情形 A）
  topK: 2, // --rollout-top-k（-1 表示不截断）
  sampled: 1,
  tpSize: 2, // 训练侧词表 TP 分片
  toolTarget: 3, // 工具 token：loss_mask=0，nucleus 为空
  microBatches: 2,
  routeGrads: [1.0, 2 ** -8, 2 ** -8], // 一个 token 的 3 个 route 梯度，按 top-k 列顺序
  ratios: [0.5, 4.0], // 同一序列两个 token 的 exp(train_old − rollout)
  vanilla: { low: 0, high: 2 }, // --tis-clip-low / --tis-clip 默认值
  icepop: { low: 0.5, high: 2 }, // GLM-5 门禁传入的值
  mis: { low: 0.5, high: 2, veto: 1e-4 }, // examples/train_infer_mismatch_helper/mis.yaml
});

const range = (n) => Array.from({ length: n }, (_, i) => i);
const sum = (xs) => xs.reduce((a, x) => a + x, 0);
const logsumexp = (xs) => {
  const m = Math.max(...xs);
  return m + Math.log(sum(xs.map((x) => Math.exp(x - m))));
};
const softmax = (xs) => {
  const l = logsumexp(xs);
  return xs.map((x) => Math.exp(x - l));
};

// ---------------- 对源码算法的最小复现 ----------------

// docker/patch/latest/sglang-top_p.patch::_top_p_keep_mask_sorted：按概率降序排序，
// keep = rank < top_k，且（need_top_p_sampling 时）前缀和（不含自身）≤ top_p；两项在未截断分布上联合判断，
// 与上游 sampler.py::top_k_top_p_min_p_sampling_from_probs_torch 的截断顺序一致。min_p 在 slime 默认请求里不设。
export function joinTopKTopPKeep(probs, topK, topP) {
  const order = range(probs.length).sort((a, b) => probs[b] - probs[a]);
  const kEff = topK === -1 ? Number.POSITIVE_INFINITY : topK;
  let prefix = 0;
  const rows = order.map((v, rank) => {
    const prefixBefore = prefix;
    prefix += probs[v];
    const rankKeep = rank < kEff;
    const topPKeep = prefixBefore <= topP;
    return { v, rank, p: probs[v], prefixBefore, rankKeep, topPKeep, keep: rankKeep && topPKeep };
  });
  return { rows, keep: rows.filter((r) => r.keep).map((r) => r.v).sort((a, b) => a - b) };
}

// slime/backends/megatron_utils/loss.py::get_log_probs_and_entropy（温度缩放）+ _build_topp_keep_mask +
// slime/utils/ppo_utils.py::_VocabParallelLogProbEntropy.forward（keep mask 置 −inf、目标 logit 写回、
// 词表分片上的 max / sum all-reduce）。keep === null 表示没有 ids（全词表）。
export function vocabParallelLogProb(logits, target, keep, tpSize, temperature = 1.0) {
  const z = logits.map((x) => x / temperature);
  const per = z.length / tpSize;
  const shards = range(tpSize).map((r) => {
    const ids = range(per).map((k) => r * per + k);
    const masked = ids.map((v) => (keep === null || keep.includes(v) || v === target ? z[v] : -Infinity));
    return { ids, masked, keptLocally: masked.some((x) => x > -Infinity) };
  });
  const gmax = Math.max(...shards.flatMap((s) => s.masked));
  const gsum = sum(shards.map((s) => sum(s.masked.map((x) => Math.exp(x - gmax)))));
  return { logProb: z[target] - gmax - Math.log(gsum), shards };
}

// sglang-top_p.patch::renorm_logprob_over_top_p：在 keep ∪ {采到的 token} 上重归一（force-keep）。
const renormLogProb = (z, keep, y) => z[y] - logsumexp([...new Set([...keep, y])].map((v) => z[v]));

export function topPReplay(cfg = CFG) {
  const z = cfg.logits.map((x) => x / cfg.temperature);
  const probs = softmax(z);
  const y = cfg.sampled;
  const full = z[y] - logsumexp(z);

  // 情形 A：rollout_top_p ≠ 1 → GenerateState 请求 return_top_p_token_ids
  const caseA = joinTopKTopPKeep(probs, cfg.topK, cfg.topP);
  const nucleus = renormLogProb(z, caseA.keep, y);
  const topPOnly = joinTopKTopPKeep(probs, -1, cfg.topP);
  const topPOnlyLogProb = renormLogProb(z, topPOnly.keep, y);
  const tp = vocabParallelLogProb(cfg.logits, y, caseA.keep, cfg.tpSize, cfg.temperature);
  const tool = vocabParallelLogProb(cfg.logits, cfg.toolTarget, [], cfg.tpSize, cfg.temperature);

  // 情形 B：rollout_top_p = 1、top_k ≠ -1 → 不请求 ids；上游 sampler 返回 log(probs)（全词表），
  // get_rollout_top_p_logprob_kwargs 返回空、训练侧也在全词表上重算；采样本身仍按 top-k 截断归一。
  const caseB = joinTopKTopPKeep(probs, cfg.topK, 1.0);
  const behaviorB = renormLogProb(z, caseB.keep, y);
  const rolloutB = full;
  const trainB = vocabParallelLogProb(cfg.logits, y, null, cfg.tpSize, cfg.temperature).logProb;

  return {
    probs,
    grid: caseA.rows,
    keep: caseA.keep,
    topPOnlyKeep: topPOnly.keep,
    full,
    nucleus,
    phantomDelta: Math.abs(full - nucleus),
    topPOnlyLogProb,
    topPOnlyDelta: Math.abs(topPOnlyLogProb - nucleus),
    tpLogProb: tp.logProb,
    tpShards: tp.shards.map((s) => ({ ids: s.ids, keptLocally: s.keptLocally })),
    toolLogProb: tool.logProb,
    caseB: {
      keep: caseB.keep,
      rollout: rolloutB,
      train: trainB,
      delta: Math.abs(rolloutB - trainB),
      behavior: behaviorB,
      hiddenGap: Math.abs(behaviorB - rolloutB),
    },
  };
}

// slime/utils/routing_replay.py::RoutingReplay 的 record / pop_forward / pop_backward / clear_forward，
// 按 actor.py::MegatronTrainRayActor.train_actor 与 model.py::train_one_step 设置 ROUTING_REPLAY_STAGE 的顺序推进。
// reuse=true 对应 train_actor 的 can_reuse_log_probs_in_loss 为真：old logprob 前向与 clear_all_forward 都不运行。
export function replayCursors({ microBatches, r3, recompute, reuse = false }) {
  if (reuse && !r3) throw new Error('R2（只开 --use-routing-replay）时 can_reuse_log_probs_in_loss 恒为假');
  const st = { recorded: 0, forward: 0, backward: 0 };
  const log = [];
  const snap = (what) => log.push({ what, ...st });
  if (r3) {
    for (let i = 0; i < microBatches; i += 1) st.recorded += 1;
    snap('fill_routing_replay');
  }
  snap('ref 前向 fallthrough');
  if (r3 && reuse) {
    snap('old 前向跳过（复用）');
  } else if (r3) {
    st.forward += microBatches;
    snap('old 前向 replay_forward');
    st.forward = 0;
    snap('clear_all_forward');
  } else {
    st.recorded += microBatches;
    snap('old 前向 record');
  }
  for (let i = 0; i < microBatches; i += 1) {
    st.forward += 1;
    if (recompute) st.backward += 1;
  }
  snap(recompute ? '训练前向 + 重算' : '训练前向（无重算）');
  return { final: { ...st }, log };
}

function toBf16(x) {
  const u = new Uint32Array(new Float32Array([x]).buffer)[0];
  const r = ((u + 0x7fff + ((u >>> 16) & 1)) >>> 16) << 16;
  return new Float32Array(new Uint32Array([r >>> 0]).buffer)[0];
}

// slime/backends/megatron_utils/alignment/deterministic_route_kernels.py::_scatter_routes_backward_kernel：
// FP32 累加器每加一个 slot 就舍入到 BF16
export function orderedBf16Sum(values) {
  let acc = 0;
  for (const v of values) acc = toBf16(Math.fround(acc + v));
  return acc;
}
export function fp32SumThenBf16(values) {
  return toBf16(values.reduce((a, v) => Math.fround(a + v), 0));
}

// loss.py::vanilla_tis_function / icepop_function；examples/train_infer_mismatch_helper/mis.py::compute_mis_weights
export function corrections(cfg = CFG) {
  const rho = cfg.ratios;
  const ones = rho.map(() => 1);
  const vanilla = { w: rho.map((r) => Math.min(Math.max(r, cfg.vanilla.low), cfg.vanilla.high)), mask: ones };
  const icepop = { w: rho.map((r) => (r >= cfg.icepop.low && r <= cfg.icepop.high ? r : 0)), mask: ones };
  const { low, high } = cfg.mis;
  const truncate = { w: rho.map((r) => Math.min(r, high)), mask: ones };
  const clip = { w: rho.map((r) => Math.min(Math.max(r, low), high)), mask: ones };
  const maskMode = { w: [...rho], mask: rho.map((r) => (r >= low && r <= high ? 1 : 0)) };
  const logSum = sum(rho.map(Math.log));
  const sequence = { w: rho.map(() => Math.min(Math.exp(logSum), high)), mask: ones };
  const geometric = { w: rho.map(() => Math.exp(logSum / rho.length)), mask: ones };
  // mis.yaml：truncate → RS token（同一 log-ratio，界 [low, high]）→ veto → token 级 batch 归一
  const rsMask = rho.map((r) => (r >= low && r <= high ? 1 : 0));
  const veto = rho.some((r) => r < cfg.mis.veto) ? 0 : 1;
  const mean = sum(truncate.w) / truncate.w.length;
  const yaml = { w: truncate.w.map((w) => w / mean), mask: rsMask.map((m) => m * veto), batchMean: mean };
  return { vanilla, icepop, truncate, clip, maskMode, sequence, geometric, yaml };
}

export function model(cfg = CFG) {
  return {
    topP: topPReplay(cfg),
    r3: replayCursors({ microBatches: cfg.microBatches, r3: true, recompute: true }),
    r3NoRecompute: replayCursors({ microBatches: cfg.microBatches, r3: true, recompute: false }),
    r3Reuse: replayCursors({ microBatches: cfg.microBatches, r3: true, recompute: true, reuse: true }),
    r2: replayCursors({ microBatches: cfg.microBatches, r3: false, recompute: true }),
    accum: {
      columnOrder: orderedBf16Sum(cfg.routeGrads),
      reversed: orderedBf16Sum([...cfg.routeGrads].reverse()),
      fp32Once: fp32SumThenBf16(cfg.routeGrads),
    },
    corr: corrections(cfg),
  };
}

// ---------------- 渲染 ----------------
const esc = (v) => String(v).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const f3 = (x) => (Object.is(x, -0) ? 0 : x).toFixed(3);
const fw = (x) => (Number.isInteger(x) ? String(x) : String(Number(x.toFixed(4))));
const set = (xs) => `{${xs.join(', ')}}`;
export const SVG_BASELINE = 'THUDM/slime@4c193f1f3750';
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
  .main{fill:none;stroke:#2563EB;stroke-width:2;marker-end:url(#arrowMain)}
`;

function render(m, cfg = CFG) {
  const W = 1180;
  const H = 1256;
  const o = [];
  const rect = (x, y, w, h, cls = 'neutral', r = 6) =>
    o.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" class="${cls}"/>`);
  const text = (x, y, s, cls = 'tx', anchor = 'start') =>
    o.push(`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${esc(s)}</text>`);
  const arrow = (x1, y1, x2, y2) => o.push(`<path d="M${x1} ${y1} L${x2} ${y2}" class="main"/>`);

  o.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="title desc">`);
  o.push('<title id="title">slime 训推一致性：一个 token 的 logprob 在支持集、路由与校正三处的重放</title>');
  o.push('<desc id="desc">三个面板：补丁按 top-k 与 top-p 联合截断得到保留集，训练侧 keep mask 在同一集合上重算，TP 分片与工具 token 的空 nucleus 仍得到一致结果；top_p=1、top_k≠-1 时两侧同在全词表、差异转入 old policy 与 TIS；路由重放的记录与两个 cursor 如何被 old-logprob 前向（若单独运行）、训练前向与重算消费，以及 route 梯度按列顺序逐次 BF16 累加的差别；同一组 train-old/rollout 比值在 vanilla TIS、ICEPOP 与示例 MIS 下得到的权重与 mask。</desc>');
  o.push('<defs><marker id="arrowMain" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#2563EB"/></marker></defs>');
  o.push(`<style>${STYLE}</style><rect width="${W}" height="${H}" fill="white"/>`);

  text(24, 34, '同一份权重下，一个 token 的 logprob 在支持集、路由与校正三处怎样分叉、又怎样被重放收拢', 'ti');
  text(24, 56, `词表 4、logits [${cfg.logits.join(', ')}]、温度 ${cfg.temperature}、采到 y=${cfg.sampled} · 训练 TP=${cfg.tpSize} · 一个 router、${cfg.microBatches} 个 micro-batch · 比值 ρ=[${cfg.ratios.join(', ')}]`, 'su');

  // ---------- ① top-p / top-k ----------
  const t = m.topP;
  const py = 72;
  rect(24, py, 1132, 420, 'panel', 10);
  text(40, py + 24, '① L2 采样支持集：rollout 返回的保留集已含 top-k 截断，训练侧在同一集合上重算', 'pt');

  // 情形 A 的联合截断网格
  text(40, py + 48, `情形 A：top_p=${cfg.topP}、top_k=${cfg.topK}，请求 nucleus ids`, 'tx');
  text(40, py + 64, '补丁 _top_p_keep_mask_sorted 的联合截断（依赖侧补丁）', 'sm');
  const gx = 176;
  const gy = py + 76;
  t.grid.forEach((r, j) => text(gx + j * 50 + 23, gy + 10, `v=${r.v}`, 'sm', 'middle'));
  const gridRows = [
    ['p（按降序）', (r) => r.p.toFixed(2), () => 'cell'],
    [`rank < top_k=${cfg.topK}`, (r) => (r.rankKeep ? '✓' : '×'), (r) => (r.rankKeep ? 'cell' : 'ghost')],
    [`前缀和 ≤ ${cfg.topP}`, (r) => `${r.prefixBefore.toFixed(2)} ${r.topPKeep ? '✓' : '×'}`, (r) => (r.topPKeep ? 'cell' : 'ghost')],
    ['保留（两者同时）', (r) => (r.keep ? '保留' : '—'), (r) => (r.keep ? 'acc1' : 'ghost')],
  ];
  gridRows.forEach(([label, val, cls], i) => {
    const y = gy + 16 + i * 28;
    text(40, y + 16, label, 'sm');
    t.grid.forEach((r, j) => {
      rect(gx + j * 50, y, 46, 24, cls(r), 3);
      text(gx + j * 50 + 23, y + 16, val(r), 'sm', 'middle');
    });
  });
  text(40, gy + 144, `只看 top-p 会保留 ${set(t.topPOnlyKeep)}；返回的 ids 已含 top-k：S = ${set(t.keep)}`, 'sm');
  arrow(380, gy + 100, 404, gy + 100);

  // 成对柱：全词表 vs 保留集
  text(412, py + 48, '灰：全词表 softmax（训练侧不重放）', 'sm');
  text(412, py + 64, '蓝：在 S ∪ {y} 上归一（rollout 重归一 = 训练重放）', 'sm');
  const probsKept = t.probs.map((p, v) => (t.keep.includes(v) ? p / sum(t.keep.map((k) => t.probs[k])) : 0));
  const base = py + 212;
  t.probs.forEach((p, v) => {
    const x = 416 + v * 80;
    const h1 = p * 150;
    const h2 = probsKept[v] * 150;
    rect(x, base - h1, 28, h1, v === cfg.sampled ? 'acc2' : 'ghost', 2);
    rect(x + 32, base - h2, 28, Math.max(h2, 0.01), v === cfg.sampled ? 'acc1' : t.keep.includes(v) ? 'neutral' : 'ghost', 2);
    text(x + 14, base - h1 - 5, p.toFixed(2), 'sm', 'middle');
    text(x + 46, base - Math.max(h2, 0) - 5, probsKept[v].toFixed(2), 'sm', 'middle');
    text(x + 30, base + 15, `v=${v}`, 'sm', 'middle');
  });

  // 情形 A 结论
  text(40, py + 250, `训练不重放：log p(y=${cfg.sampled}) = ${f3(t.full)}（全词表）`, 'tx');
  text(40, py + 270, `rollout 返回 ${f3(t.nucleus)}（补丁重归一，依赖侧）`, 'tx');
  rect(40, py + 282, 340, 24, 'acc2', 4);
  text(50, py + 298, `假差异 δ = ${f3(t.phantomDelta)}：来自归一化域，与权重、kernel 无关`, 'sm');
  rect(40, py + 314, 340, 24, 'acc1', 4);
  text(50, py + 330, `keep mask 重放：训练侧也得 ${f3(t.nucleus)}，δ = 0`, 'sm');
  text(40, py + 358, `若只按 top-p 的 ${set(t.topPOnlyKeep)} 归一得 ${f3(t.topPOnlyLogProb)}，仍差 ${f3(t.topPOnlyDelta)}`, 'sm');

  // 情形 B
  const b = t.caseB;
  rect(400, py + 240, 340, 166, 'neutral', 8);
  text(412, py + 260, `情形 B：top_p=1、top_k=${cfg.topK}，不请求 ids`, 'tx');
  text(412, py + 280, `rollout 返回上游 log(probs) = ${f3(b.rollout)}（全词表，依赖侧）`, 'sm');
  text(412, py + 296, `训练侧 top-p kwargs 为空，全词表重算 = ${f3(b.train)}`, 'sm');
  rect(412, py + 306, 316, 24, 'acc1', 4);
  text(422, py + 322, `两侧同域：δ = ${fw(b.delta)}，没有假差异`, 'sm');
  rect(412, py + 338, 316, 58, 'acc2', 4);
  text(422, py + 356, `但采样在 top-k 集 ${set(b.keep)} 上归一：行为 log q = ${f3(b.behavior)}`, 'sm');
  text(422, py + 372, `old policy / TIS 用的 ${f3(b.rollout)} 与行为概率差 ${f3(b.hiddenGap)}`, 'sm');
  text(422, py + 388, '差异不进 δ，而进估计量（本页推断）', 'sm');

  // 右列：TP 分片、工具 token、依赖前提
  const rx = 760;
  rect(rx, py + 40, 382, 112, 'neutral', 8);
  text(rx + 12, py + 60, `训练 TP=${cfg.tpSize}：每个 rank 只持有一段词表`, 'tx');
  t.tpShards.forEach((s, r) => {
    const y = py + 70 + r * 30;
    rect(rx + 12, y, 358, 24, s.keptLocally ? 'cell' : 'ghost', 4);
    text(rx + 22, y + 16, `rank ${r} 持有 ${set(s.ids)}：${s.keptLocally ? '本地有保留项' : '本地整行 −inf（目标不在本段）'}`, 'sm');
  });
  text(rx + 12, py + 144, `max 与 sum 在 TP 组 all-reduce 后仍得 ${f3(t.tpLogProb)}`, 'sm');
  rect(rx, py + 162, 382, 104, 'neutral', 8);
  text(rx + 12, py + 182, `工具 token（loss_mask=0，nucleus 为空，目标 v=${cfg.toolTarget}）`, 'tx');
  text(rx + 12, py + 202, '空 span 让整行 keep=False；masked_fill 之后写回目标 logit，', 'sm');
  text(rx + 12, py + 218, `支持集只剩目标：训练 logprob = ${f3(t.toolLogProb)}，rollout 占位 0.0`, 'sm');
  text(rx + 12, py + 234, '实际重放的支持集是 S ∪ {y}；entropy 用未屏蔽的 logits', 'sm');
  text(rx + 12, py + 250, 'min_p 在 slime 默认请求里不设，不进入保留集', 'sm');
  rect(rx, py + 276, 382, 130, 'dep', 8);
  text(rx + 12, py + 296, '依赖侧前提（镜像补丁与请求方式）', 'tx');
  text(rx + 12, py + 316, '未打 sglang-top_p.patch：不返回 ids，top_p≠1 时 converter 断言', 'sm');
  text(rx + 12, py + 332, '设 SGLANG_RETURN_ORIGINAL_LOGPROB：照返 ids，但返回未截断、', 'sm');
  text(rx + 12, py + 348, '未除温度的 logprob，重放反而制造差异', 'sm');
  text(rx + 12, py + 364, '流式请求级 abort：ids 与路由随终止 chunk 下发，', 'sm');
  text(rx + 12, py + 380, '断开前没收到就缺失（sglang_streaming_rollout 模块 docstring）', 'sm');

  // ---------- ② routing replay ----------
  const ry = 506;
  rect(24, ry, 1132, 390, 'panel', 10);
  text(40, ry + 24, '② L3 路由：记录被 old-logprob 前向（若单独运行）与训练前向消费，后向 cursor 靠重算推进（本页推断）', 'pt');
  const rows = [
    ['R3 开重算；old logprob 需单独前向（如 gspo、多步、kl_coef≠0）', m.r3],
    ['R3 不开重算（同样需要 old 前向）', m.r3NoRecompute],
    ['R3 开重算；can_reuse_log_probs_in_loss 为真（单步、kl_coef=0、非 gspo 等）', m.r3Reuse],
    ['R2（只开 --use-routing-replay，不可复用）', m.r2],
  ];
  rows.forEach(([lab, res], i) => {
    const y = ry + 38 + i * 72;
    text(40, y + 14, lab, 'tx');
    res.log.forEach((e, j) => {
      const x = 40 + j * 150;
      const skipped = e.what.includes('跳过');
      rect(x, y + 22, 140, 38, j === res.log.length - 1 ? 'acc1' : skipped ? 'ghost' : 'neutral', 5);
      text(x + 70, y + 37, e.what, 'sm', 'middle');
      text(x + 70, y + 53, `记录 ${e.recorded} · 前 ${e.forward} · 后 ${e.backward}`, 'mono', 'middle');
    });
  });
  text(40, ry + 338, `结束：R3 开重算为 前 ${m.r3.final.forward}/后 ${m.r3.final.backward}/记录 ${m.r3.final.recorded}，跳过 old 前向时终态相同；不开重算时后 cursor 停在 ${m.r3NoRecompute.final.backward}`, 'sm');
  text(40, ry + 354, 'assert_all_consumed 要求两者都等于记录数，但基线没有调用它；训练后 clear_all 清空记录', 'sm');
  text(40, ry + 370, '强制 id 仍从当前 scores gather 概率：离散路径固定，router 概率与梯度照常计算', 'sm');

  const ax = 800;
  rect(ax, ry + 38, 342, 340, 'neutral', 8);
  text(ax + 12, ry + 58, '反向平面：训练侧确定性 kernel 逐 slot 累加', 'tx');
  text(ax + 12, ry + 78, 'FP32 累加器每加一个 slot 舍入到 BF16（对照树内 CPU 参考）', 'sm');
  const g = cfg.routeGrads;
  const lab = (xs) => xs.map((x) => (x === 1 ? '1' : '2⁻⁸')).join(', ');
  rect(ax + 12, ry + 92, 318, 40, 'acc1', 5);
  text(ax + 22, ry + 108, `列顺序 [${lab(g)}]`, 'sm');
  text(ax + 22, ry + 124, `1 + 2⁻⁸ 恰为半个 ULP，舍回 1 → 结果 ${String(m.accum.columnOrder)}`, 'sm');
  rect(ax + 12, ry + 142, 318, 40, 'acc2', 5);
  text(ax + 22, ry + 158, `反序 [${lab([...g].reverse())}]`, 'sm');
  text(ax + 22, ry + 174, `2⁻⁸ + 2⁻⁸ = 2⁻⁷ 先成形 → 结果 ${String(m.accum.reversed)}`, 'sm');
  rect(ax + 12, ry + 192, 318, 40, 'acc2', 5);
  text(ax + 22, ry + 208, '一次 FP32 求和再转 BF16', 'sm');
  text(ax + 22, ry + 224, `结果 ${String(m.accum.fp32Once)}：不能替代逐 slot 舍入`, 'sm');
  rect(ax + 12, ry + 244, 318, 122, 'dep', 5);
  text(ax + 22, ry + 264, '前向平面（依赖侧）', 'tx');
  text(ax + 22, ry + 284, '对齐桥让 Megatron 沿用 SGLang 无序 top-k 的列顺序', 'sm');
  text(ax + 22, ry + 300, '训练侧 owner 归约调用上游 SGLang ep_gather：', 'sm');
  text(ax + 22, ry + 316, '按 slot 在 FP32 累加、末尾一次转 BF16（上游源码）', 'sm');
  text(ax + 22, ry + 332, 'rollout 侧由 DeepEP 低延迟 combine 完成', 'sm');
  text(ax + 22, ry + 348, '（zhuzilin/DeepEP fork，本机不可读）', 'sm');

  // ---------- ③ corrections ----------
  const cy = 910;
  const c = m.corr;
  rect(24, cy, 1132, 282, 'panel', 10);
  text(40, cy + 24, `③ 校正：同一组 ρ=[${cfg.ratios.join(', ')}] 走不同函数，改的是权重或 mask，不是差异本身`, 'pt');
  const crow = [
    ['vanilla TIS（默认 hook）', `clamp 到 [${cfg.vanilla.low}, ${cfg.vanilla.high}]`, c.vanilla],
    ['ICEPOP（GLM-5 门禁）', `区间 [${cfg.icepop.low}, ${cfg.icepop.high}] 外置 0`, c.icepop],
    ['MIS truncate', `只截上界 ${cfg.mis.high}`, c.truncate],
    ['MIS clip', `截到 [${cfg.mis.low}, ${cfg.mis.high}]`, c.clip],
    ['MIS mask', '权重不变，区间外 mask 置 0', c.maskMode],
    ['MIS sequence', 'log 比之和 ln 2 → 整序列同权', c.sequence],
    ['MIS geometric', 'log 比均值 → 整序列同权', c.geometric],
    ['mis.yaml 默认', `truncate → RS [${cfg.mis.low}, ${cfg.mis.high}] → veto ${cfg.mis.veto} → 归一（均值 ${fw(c.yaml.batchMean)}）`, c.yaml],
  ];
  text(40, cy + 48, '函数', 'sm');
  text(260, cy + 48, '规则', 'sm');
  text(740, cy + 48, '权重', 'sm');
  text(900, cy + 48, 'mask', 'sm');
  crow.forEach(([a, bb, r], i) => {
    const y = cy + 56 + i * 24;
    const rejected = r.mask.some((x) => x === 0);
    rect(36, y, 1106, 22, rejected ? 'acc2' : i % 2 ? 'ghost' : 'cell', 3);
    text(44, y + 15, a, 'sm');
    text(260, y + 15, bb, 'sm');
    text(740, y + 15, `[${r.w.map(fw).join(', ')}]`, 'mono');
    text(900, y + 15, `[${r.mask.join(', ')}]`, 'mono');
  });
  text(40, cy + 270, '被拒绝的 token 只从分子删除，分母仍是原始 rollout_mask_sums（归约口径见 Loss 归约页）；use_tis 关闭时 MIS 只算指标、不产生权重', 'sm');

  text(24, 1222, '阅读顺序：① 先对齐归一化域 → ② 再固定离散路由与累加顺序 → ③ 前面都定位清楚之后，才用校正函数改估计量。', 'cap');
  text(24, 1244, `源码基线：${SVG_BASELINE} · 补丁 docker/patch/latest/sglang-top_p.patch · 复现 _top_p_keep_mask_sorted、keep mask、RoutingReplay、route kernel 与 TIS/MIS`, 'su');
  o.push('</svg>');
  return o.join('\n');
}

const here = dirname(fileURLToPath(import.meta.url));
const defaultOutput = join(here, '..', '..', '..', 'wiki', '02_engineering', '04_posttrain_frameworks', 'slime', 'assets');
const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  const outputDir = process.argv[2] ? process.argv[2] : defaultOutput;
  mkdirSync(outputDir, { recursive: true });
  writeFileSync(join(outputDir, 'slime_train_infer_replay.svg'), `${render(model())}\n`, 'utf8');
}
