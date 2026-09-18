// 图：slime on-policy 蒸馏的逐 token 信号，用同一条 Sample（prompt 2 个 token、response 3 个 token）回放三步：
// ① 对齐：SGLang teacher 返回的 input_token_logprobs 经 post_process_rewards 丢首项、尾部裁剪，
//    Megatron 前向按 logits[P−1 : T−1] 取 response 位置，两边落到同一组 token；外加温度边界（依赖侧）。
// ② 注入：d̂ = 学生项 − teacher 项，apply_opd_kl_to_advantages 逐 token 写成 Â = A − λ·d̂（纯蒸馏与 RL+OPD 各一行）。
// ③ 尺度：--normalize-advantages 的 masked 白化让纯蒸馏里的 |λ| 整体消去；compute_policy_loss 的 clip
//    让 λ 只放大信任域内的梯度。
// 源码基线：THUDM/slime@4c193f1f37509cca70f0e88807a9305b70f63f4e；
// SGLang 输入 logprob 的位置与温度语义取自上游 sgl-project/sglang@0b3bb0cbe318（v0.5.15.post1）源码，
// slime 的 docker/patch/latest/sglang*.patch 不改这两处（依赖侧，源码阅读，未运行）。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
// 要讲清楚：对齐由尾部裁剪 [-R:] 决定，丢首项去的是无前驱占位 None；actor 侧只有长度断言；
// d̂ 是采样 token 上的单点差，可正可负，Â 在前向中是常量；白化后纯蒸馏的 Â 与 |λ| 无关，
// clip 让 ρ 越界的 token 梯度为 0，λ 不能线性放大更新。acc1 标对齐后的结果与被消去的尺度，
// acc2 标正的 d̂（被压低的 token）与被 clip 截断的梯度；依赖侧用虚线框。
//
// 用法：node tools/figs/svg/slime_opd_signal_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

export const SVG_BASELINE = 'THUDM/slime@4c193f1f3750';

// ---------------- 冻结的示例输入（解释用数值，不是测量） ----------------
export const CFG = Object.freeze({
  promptIds: [11, 12],
  responseIds: [21, 22, 23],
  // SGLang teacher 对完整 sample.tokens 的 input_token_logprobs；第 k 项是 log p_T(x_k | x_<k)，k=0 无前驱
  teacherInputLogProbs: [null, -1.2, -0.4, -2.1, -0.3],
  // 学生项（默认 actor tag 重算）在 response 位置上的 log_probs
  studentLogProbs: [-0.9, -1.3, -0.2],
  lambda: 1.0, // --opd-kl-coef 默认值
  lambdaAlt: 2.0,
  mixedAdvantage: 0.5, // RL+OPD：自定义 postprocess 保留任务 reward 时，GRPO 给该序列的常数 advantage
  epsClip: 0.2, // --eps-clip 默认；--eps-clip-high 缺省时等于它
  ratios: [1.0, 0.75], // 同一 token 在一次 rollout 内多个 optimizer step 中可能到达的 ρ
  clipToken: 1, // response 第 2 个 token（id 22）
  whitenEpsilon: 1e-8, // slime/utils/distributed_utils.py::distributed_masked_whiten
  temperature: { logits: [2.0, 1.0, 0.0], target: 0, tau: 0.8 },
});

const sum = (xs) => xs.reduce((a, x) => a + x, 0);
const logsumexp = (xs) => {
  const m = Math.max(...xs);
  return m + Math.log(sum(xs.map((x) => Math.exp(x - m))));
};

// ---------------- 对源码算法的最小复现 ----------------

// slime/rollout/on_policy_distillation.py::post_process_rewards：[1:] 去掉无前驱项，再按 response_length 取尾部。
export function alignSglangTeacher(inputLogProbs, responseLength) {
  const dropped = inputLogProbs.slice(1);
  const cropped = dropped.slice(-responseLength);
  // 对照：不丢首项、只做尾部裁剪，得到的对齐结果相同（只要 prompt 至少 1 个 token）
  const cropOnly = inputLogProbs.slice(-responseLength);
  return { dropped, cropped, cropOnly, tensorizable: !dropped.includes(null) };
}

// slime/backends/megatron_utils/loss.py::_build_shifted_tokens + _extract_per_sample（cp=1）：
// full_tokens[k] = tokens[k+1]，response 段取 logits 位置 [P−1, T−1)。
export function megatronResponsePositions(promptLength, totalLength) {
  const logitPositions = [];
  for (let k = promptLength - 1; k < totalLength - 1; k += 1) logitPositions.push(k);
  return logitPositions.map((k) => ({ logit: k, target: k + 1 }));
}

// slime/backends/megatron_utils/cp_utils.py::slice_log_prob_with_cp 的长度断言（cp=1 时只断言、不切片）
export function sliceLogProbWithCp(logProb, responseLength) {
  if (logProb.length !== responseLength) throw new Error(`log_prob length mismatch: ${logProb.length} != ${responseLength}`);
  return logProb;
}

// slime/backends/megatron_utils/loss.py::apply_opd_kl_to_advantages
export function applyOpd(advantages, student, teacher, lambda) {
  if (student === null) return { advantages, reverseKl: null }; // 学生项缺失时静默返回
  if (teacher === null) throw new Error('OPD requires teacher_log_probs');
  const reverseKl = student.map((s, i) => s - teacher[i]);
  return { advantages: advantages.map((a, i) => a - lambda * reverseKl[i]), reverseKl };
}

// slime/utils/distributed_utils.py::distributed_masked_whiten（单进程、mask 全 1、shift_mean=True）
export function maskedWhiten(values, mask, epsilon) {
  const n = sum(mask);
  const mean = sum(values.map((v, i) => v * mask[i])) / n;
  const meanSq = sum(values.map((v, i) => v * v * mask[i])) / n;
  let variance = meanSq - mean * mean;
  if (n >= 2) variance *= n / (n - 1);
  const invStd = 1 / Math.sqrt(variance + epsilon);
  return values.map((v) => (v - mean) * invStd);
}

// slime/utils/ppo_utils.py::compute_policy_loss：ℓ = max(−ρÂ, −clip(ρ, 1−ε, 1+ε_high)·Â)，
// 返回 g = ∂ℓ/∂log π（ρ = exp(log π − log π_old)，所以 ∂ρ/∂log π = ρ）。
export function policyLossGrad(ratio, advantage, epsClip, epsClipHigh = epsClip) {
  const clipped = Math.min(Math.max(ratio, 1 - epsClip), 1 + epsClipHigh);
  const loss1 = -ratio * advantage;
  const loss2 = -clipped * advantage;
  if (loss2 > loss1) return 0;
  return -ratio * advantage;
}

const logSoftmaxAt = (logits, target, tau) => {
  const z = logits.map((x) => x / tau);
  return z[target] - logsumexp(z);
};

export function model(cfg = CFG) {
  const tokens = [...cfg.promptIds, ...cfg.responseIds];
  const P = cfg.promptIds.length;
  const R = cfg.responseIds.length;
  const T = tokens.length;
  const sglang = alignSglangTeacher(cfg.teacherInputLogProbs, R);
  const teacher = sliceLogProbWithCp(sglang.cropped, R);
  const student = sliceLogProbWithCp(cfg.studentLogProbs, R);
  const positions = megatronResponsePositions(P, T);

  const zeros = cfg.responseIds.map(() => 0);
  const pure = applyOpd(zeros, student, teacher, cfg.lambda);
  const pureAlt = applyOpd(zeros, student, teacher, cfg.lambdaAlt);
  const mixed = applyOpd(cfg.responseIds.map(() => cfg.mixedAdvantage), student, teacher, cfg.lambda);
  const mask = cfg.responseIds.map(() => 1);
  const whitened = maskedWhiten(pure.advantages, mask, cfg.whitenEpsilon);
  const whitenedAlt = maskedWhiten(pureAlt.advantages, mask, cfg.whitenEpsilon);

  const clipRows = [cfg.lambda, cfg.lambdaAlt].map((lambda) => {
    const adv = -lambda * pure.reverseKl[cfg.clipToken];
    return { lambda, advantage: adv, grads: cfg.ratios.map((r) => policyLossGrad(r, adv, cfg.epsClip)) };
  });

  const t = cfg.temperature;
  const studentTau = logSoftmaxAt(t.logits, t.target, t.tau);
  const teacherUnscaled = logSoftmaxAt(t.logits, t.target, 1.0);

  return {
    tokens,
    P,
    R,
    T,
    sglang,
    teacher,
    student,
    positions,
    reverseKl: pure.reverseKl,
    pure: pure.advantages,
    pureAlt: pureAlt.advantages,
    mixed: mixed.advantages,
    whitened,
    whitenedAlt,
    clipRows,
    clipLow: 1 - cfg.epsClip,
    temperature: { studentTau, teacherUnscaled, gap: studentTau - teacherUnscaled },
  };
}

// ---------------- 数值格式（页面与 SVG 共用） ----------------
const clean = (x, digits) => {
  const s = x.toFixed(digits);
  return Number(s) === 0 ? (0).toFixed(digits) : s;
};
// 带符号：正数加 +，负数用 U+2212
export const fsig = (x, digits = 2) => {
  const s = clean(x, digits);
  if (s.startsWith('-')) return `−${s.slice(1)}`;
  return Number(s) === 0 ? s : `+${s}`;
};
// 只把负号换成 U+2212
export const fneg = (x, digits = 2) => {
  const s = clean(x, digits);
  return s.startsWith('-') ? `−${s.slice(1)}` : s;
};
export const vec = (xs, digits = 2, f = fsig) => `[${xs.map((x) => f(x, digits)).join(', ')}]`;

// ---------------- 渲染 ----------------
const esc = (v) => String(v).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
// d̂（d + U+0302）在部分字体里帽子脱离字母；改成 d 加一个上移、缩小的 ^，再把基线与水平位置拉回。
const HAT = '\u0302';
function withHat(s, px) {
  const parts = String(s).split(`d${HAT}`);
  if (parts.length === 1) return esc(s);
  const r = (x) => Number(x.toFixed(2));
  const up = r(0.42 * px);
  const hat = `<tspan dx="${r(-0.47 * px)}" dy="${-up}" font-size="${r(0.85 * px)}">^</tspan>`;
  let out = esc(parts[0]);
  for (let i = 1; i < parts.length; i += 1) {
    out += `d${hat}<tspan dx="${r(0.07 * px)}" dy="${up}">${esc(parts[i])}</tspan>`;
  }
  return out;
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
  .dep{fill:none;stroke:#8A93A0;stroke-width:1.2;stroke-dasharray:5 4}
  .depbox{fill:#F5F7FA;stroke:#8A93A0;stroke-width:1.2;stroke-dasharray:5 4}
  .acc1{fill:#EAF1FD;stroke:#2563EB;stroke-width:1.5}
  .acc2{fill:#FCF1E6;stroke:#C3651F;stroke-width:1.5}
  .main{fill:none;stroke:#2563EB;stroke-width:1.6;marker-end:url(#arrowMain)}
`;

const FONT_PX = { ti: 19, su: 12, pt: 14, tx: 12, sm: 10.5, mono: 11, cap: 11.5 };
// 估算文字宽度：CJK 与全角符号按 1em，其余按 0.6em（mono 0.62em）
export function estimateTextWidth(s, cls) {
  const px = FONT_PX[cls] ?? 12;
  let w = 0;
  for (const ch of String(s)) {
    const code = ch.codePointAt(0);
    if (code === 0x0302) continue; // 组合帽子不占宽度
    const wide = code >= 0x2e80 || (code >= 0xff00 && code <= 0xffef) || ch === '−' || ch === '→' || ch === '×' || ch === '·';
    w += wide ? (ch === '·' ? 0.6 : 1.0) : cls === 'mono' ? 0.62 : 0.6;
  }
  return w * px;
}

export function render(m, cfg = CFG) {
  const W = 1180;
  const H = 1030;
  const o = [];
  const texts = [];
  const rects = [];
  const rect = (x, y, w, h, cls = 'neutral', r = 6) => {
    rects.push({ x, y, w, h, cls });
    o.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" class="${cls}"/>`);
  };
  const text = (x, y, s, cls = 'tx', anchor = 'start') => {
    const w = estimateTextWidth(s, cls);
    const x0 = anchor === 'middle' ? x - w / 2 : anchor === 'end' ? x - w : x;
    texts.push({ x0, x1: x0 + w, y, s: String(s), cls, size: FONT_PX[cls] ?? 12 });
    o.push(`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${withHat(s, FONT_PX[cls] ?? 12)}</text>`);
  };
  const arrow = (x1, y1, x2, y2) => o.push(`<path d="M${x1} ${y1} L${x2} ${y2}" class="main"/>`);

  o.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="title desc">`);
  o.push('<title id="title">slime OPD：一条 Sample 的 teacher 与学生 logprob 怎样对齐、注入 advantage 并被缩放</title>');
  o.push('<desc id="desc">三个面板：SGLang teacher 的输入 logprob 丢首项并尾部裁剪、Megatron 前向按 logits 位移取 response 位置，两边落到同一组 token；逐 token 的 d̂ 写进 advantage；白化消去纯蒸馏里的 λ 尺度，clip 让越界 token 的梯度为 0。</desc>');
  o.push('<defs><marker id="arrowMain" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#2563EB"/></marker></defs>');
  o.push(`<style>${STYLE}</style><rect width="${W}" height="${H}" fill="white"/>`);

  text(24, 34, '一条 Sample 的 OPD 信号：先对齐到同一 response 位置，再逐 token 写进 advantage，最后被白化与 clip 重新定尺度', 'ti');
  text(24, 56, `tokens [${m.tokens.join(', ')}]：prompt P=${m.P}、response R=${m.R}、T=${m.T} · λ=--opd-kl-coef=${cfg.lambda} · ε=--eps-clip=${cfg.epsClip} · 数值是解释用输入，不是测量`, 'su');

  // ---------- ① 对齐 ----------
  const y1 = 72;
  rect(24, y1, 1132, 372, 'panel', 10);
  text(40, y1 + 24, '① 对齐：两条路径的 logprob 都要落到 response 的同一组 token 上', 'pt');
  const cx = (k) => 262 + k * 100;
  const cw = 88;
  const ch = 28;
  // token 行
  const rTok = y1 + 56;
  text(40, rTok + 18, `sample.tokens（位置 k）`, 'tx');
  m.tokens.forEach((id, k) => {
    text(cx(k) + cw / 2, rTok - 6, `k=${k}`, 'sm', 'middle');
    rect(cx(k), rTok, cw, ch, k < m.P ? 'ghost' : 'neutral', 4);
    text(cx(k) + cw / 2, rTok + 18, String(id), 'mono', 'middle');
  });
  text(cx(0) + cw, rTok + 46, 'prompt', 'sm', 'middle');
  text(cx(m.P + 1) + cw / 2, rTok + 46, 'response', 'sm', 'middle');

  // SGLang lane（依赖侧虚线框）
  const sgTop = y1 + 118;
  rect(34, sgTop - 8, 766, 132, 'dep', 8);
  text(44, sgTop + 8, '依赖侧 SGLang /generate（上游 v0.5.15.post1 契约）', 'sm');
  const rIn = sgTop + 16;
  text(44, rIn + 18, 'input_token_logprobs', 'mono');
  cfg.teacherInputLogProbs.forEach((v, k) => {
    rect(cx(k), rIn, cw, ch, v === null ? 'ghost' : 'neutral', 4);
    text(cx(k) + cw / 2, rIn + 18, v === null ? 'None' : fneg(v), 'mono', 'middle');
  });
  const rDrop = rIn + 36;
  text(44, rDrop + 18, 'post_process_rewards：[1:]', 'sm');
  m.sglang.dropped.forEach((v, j) => {
    const k = j + 1;
    rect(cx(k), rDrop, cw, ch, 'neutral', 4);
    text(cx(k) + cw / 2, rDrop + 18, fneg(v), 'mono', 'middle');
  });
  const rCrop = rDrop + 36;
  text(44, rCrop + 18, `[-R:] → teacher_log_probs`, 'sm');
  m.teacher.forEach((v, j) => {
    const k = m.P + j;
    rect(cx(k), rCrop, cw, ch, 'acc1', 4);
    text(cx(k) + cw / 2, rCrop + 18, fneg(v), 'mono', 'middle');
  });

  // Megatron lane
  const rLogit = y1 + 272;
  text(40, rLogit + 18, 'Megatron 前向：logits 位置', 'tx');
  m.positions.forEach(({ logit, target }) => {
    rect(cx(logit), rLogit, cw, ch, 'neutral', 4);
    text(cx(logit) + cw / 2, rLogit + 18, `预测 k=${target}`, 'sm', 'middle');
    arrow(cx(logit) + cw - 6, rLogit + ch, cx(target) + 10, rLogit + 52);
  });
  const rStu = rLogit + 54;
  text(40, rStu + 18, 'log_probs（学生或 teacher tag）', 'tx');
  m.student.forEach((v, j) => {
    const k = m.P + j;
    rect(cx(k), rStu, cw, ch, 'acc1', 4);
    text(cx(k) + cw / 2, rStu + 18, fneg(v), 'mono', 'middle');
  });

  // 右列：守卫与温度边界
  const bx = 818;
  const bw = 324;
  rect(bx, y1 + 44, bw, 124, 'neutral', 8);
  text(bx + 12, y1 + 64, '对齐由尾部裁剪决定', 'tx');
  text(bx + 12, y1 + 84, `不丢首项只做 [-R:] 也得到 ${vec(m.sglang.cropOnly, 2, fneg)}`, 'sm');
  text(bx + 12, y1 + 102, '[1:] 去掉的是无前驱占位 None，否则无法张量化', 'sm');
  text(bx + 12, y1 + 124, 'actor 侧 slice_log_prob_with_cp 只断言', 'sm');
  text(bx + 12, y1 + 142, `长度 = R=${m.R}，不比对 token id`, 'sm');
  text(bx + 12, y1 + 160, 'prompt 为空时尾部只剩 T−1 项，断言失败', 'sm');

  const tb = y1 + 184;
  rect(bx, tb, bw, 172, 'depbox', 8);
  const tc = cfg.temperature;
  text(bx + 12, tb + 20, '温度边界（依赖侧源码阅读，未运行）', 'tx');
  text(bx + 12, tb + 40, '学生项与 Megatron teacher：logits ÷ τ', 'sm');
  text(bx + 12, tb + 58, 'SGLang teacher：请求带 τ，但上游', 'sm');
  text(bx + 12, tb + 76, 'process_input_logprobs 不读 τ', 'sm');
  text(bx + 12, tb + 100, `同一 logits [${tc.logits.join(', ')}]、y=${tc.target}、τ=${tc.tau}：`, 'sm');
  text(bx + 12, tb + 118, `学生 ${fneg(m.temperature.studentTau, 3)}，SGLang teacher ${fneg(m.temperature.teacherUnscaled, 3)}`, 'sm');
  rect(bx + 12, tb + 128, bw - 24, 32, 'acc2', 5);
  text(bx + 22, tb + 148, `权重相同也有 d̂ = ${fsig(m.temperature.gap, 3)}`, 'tx');

  // ---------- ② 注入 ----------
  const y2 = 458;
  rect(24, y2, 1132, 262, 'panel', 10);
  text(40, y2 + 24, '② 注入：d̂ = 学生项 − teacher 项，逐 token 从基础 advantage 里减去 λ·d̂', 'pt');
  const colX = (j) => 330 + j * 104;
  const colW = 94;
  const rowH = 30;
  const rows = [
    { label: 'response token', vals: cfg.responseIds.map(String), cls: () => 'ghost', font: 'mono' },
    { label: '学生项 s（old_log_probs）', vals: m.student.map((v) => fneg(v)), cls: () => 'neutral', font: 'mono' },
    { label: 'teacher 项 t', vals: m.teacher.map((v) => fneg(v)), cls: () => 'neutral', font: 'mono' },
    { label: 'd̂ = s − t（opd_reverse_kl）', vals: m.reverseKl.map((v) => fsig(v)), cls: (j) => (m.reverseKl[j] > 0 ? 'acc2' : 'acc1'), font: 'mono' },
    { label: `纯蒸馏：A=0、λ=${cfg.lambda} → Â`, vals: m.pure.map((v) => fsig(v)), cls: () => 'neutral', font: 'mono' },
    { label: `RL+OPD：A=${cfg.mixedAdvantage}、λ=${cfg.lambda} → Â`, vals: m.mixed.map((v) => fsig(v)), cls: () => 'neutral', font: 'mono' },
  ];
  rows.forEach((row, i) => {
    const y = y2 + 42 + i * rowH;
    text(40, y + 17, row.label, 'tx');
    row.vals.forEach((v, j) => {
      rect(colX(j), y, colW, rowH - 4, row.cls(j), 4);
      text(colX(j) + colW / 2, y + 17, v, row.font, 'middle');
    });
  });
  text(40, y2 + 238, 'd̂>0：学生比 teacher 更偏爱该 token → Â 变小、被压低；d̂<0 → 被抬高。单个 d̂ 可负，期望才是非负的 KL', 'sm');

  const sx = 660;
  const sw = 482;
  rect(sx, y2 + 42, sw, 176, 'neutral', 8);
  text(sx + 12, y2 + 62, '学生项就是 PPO ratio 的分母 old_log_probs，来源有三条', 'tx');
  text(sx + 12, y2 + 84, '默认：切回 actor tag 重算（OPD 关掉 logprob 复用，多一次前向）', 'sm');
  text(sx + 12, y2 + 104, '--keep-old-actor：切到 old_actor tag 重算', 'sm');
  text(sx + 12, y2 + 124, '--use-rollout-logprobs：直接用 rollout 引擎的 logprob，不重算', 'sm');
  rect(sx + 12, y2 + 136, sw - 24, 70, 'acc1', 5);
  text(sx + 22, y2 + 156, 'Â 在 forward_only 与 advantage 计算里没有梯度，是常量', 'sm');
  text(sx + 22, y2 + 174, '训练前向只经 ρ = exp(log π_θ − old_log_probs) 求导：', 'sm');
  text(sx + 22, y2 + 192, 'ρ=1 处 ∇ℓ = λ·d̂·∇log π_θ，即 reverse-KL 的 score-function 梯度', 'sm');

  // ---------- ③ 尺度 ----------
  const y3 = 734;
  rect(24, y3, 1132, 244, 'panel', 10);
  text(40, y3 + 24, '③ 尺度：白化与 clip 之后，λ 不再是绝对系数', 'pt');
  rect(40, y3 + 40, 560, 190, 'neutral', 8);
  text(52, y3 + 60, '--normalize-advantages：distributed_masked_whiten（DP×CP 组）', 'tx');
  text(52, y3 + 84, `λ=${cfg.lambda}：Â = ${vec(m.pure)}`, 'mono');
  text(52, y3 + 102, `白化后 ${vec(m.whitened, 3)}`, 'mono');
  text(52, y3 + 128, `λ=${cfg.lambdaAlt}：Â = ${vec(m.pureAlt)}`, 'mono');
  text(52, y3 + 146, `白化后 ${vec(m.whitenedAlt, 3)}`, 'mono');
  rect(52, y3 + 160, 536, 58, 'acc1', 5);
  text(62, y3 + 180, '纯蒸馏 A=0：均值与标准差都按 |λ| 缩放，白化把它整体消去，只剩符号', 'sm');
  text(62, y3 + 200, 'RL+OPD：λ 只决定任务 advantage 与 teacher 信号的相对比例', 'sm');

  const kx = 616;
  const kw = 526;
  rect(kx, y3 + 40, kw, 190, 'neutral', 8);
  const tokenId = cfg.responseIds[cfg.clipToken];
  text(kx + 12, y3 + 60, `compute_policy_loss 的 clip：token ${tokenId}，d̂ = ${fsig(m.reverseKl[cfg.clipToken])}，Â = −λ·d̂`, 'tx');
  text(kx + 12, y3 + 80, 'g = ∂ℓ/∂log π_θ（梯度下降沿 −g 更新）', 'sm');
  const gx = (j) => kx + 200 + j * 150;
  cfg.ratios.forEach((r, j) => text(gx(j) + 64, y3 + 104, `ρ = ${r.toFixed(2)}`, 'mono', 'middle'));
  m.clipRows.forEach((row, i) => {
    const y = y3 + 114 + i * 34;
    text(kx + 12, y + 19, `λ=${row.lambda}（Â = ${fsig(row.advantage)}）`, 'mono');
    row.grads.forEach((g, j) => {
      rect(gx(j), y, 128, 28, g === 0 ? 'acc2' : 'neutral', 4);
      text(gx(j) + 64, y + 19, `g = ${fneg(g)}`, 'mono', 'middle');
    });
  });
  text(kx + 12, y3 + 200, `Â<0 时 ρ < 1−ε = ${m.clipLow.toFixed(2)} 走截断分支，梯度为 0；`, 'sm');
  text(kx + 12, y3 + 218, 'λ 只放大信任域内的梯度，不能把一次 rollout 的更新线性放大', 'sm');

  text(24, 1002, '阅读顺序：① 对齐 → ② 注入 → ③ 缩放。Megatron teacher 与学生走同一个 get_log_probs_and_entropy，位置规则相同。', 'cap');
  text(24, 1022, `源码基线：${SVG_BASELINE} · 复现 post_process_rewards / _build_shifted_tokens / apply_opd_kl_to_advantages / masked whiten / clip`, 'su');
  o.push('</svg>');
  return { svg: o.join('\n'), texts, rects, W, H };
}

const here = dirname(fileURLToPath(import.meta.url));
const defaultOutput = join(here, '..', '..', '..', 'wiki', '02_engineering', '04_posttrain_frameworks', 'slime', 'assets');
const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  const outputDir = process.argv[2] ? process.argv[2] : defaultOutput;
  mkdirSync(outputDir, { recursive: true });
  writeFileSync(join(outputDir, 'slime_opd_signal.svg'), `${render(model()).svg}\n`, 'utf8');
}
