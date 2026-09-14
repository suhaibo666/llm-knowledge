// 图：slime loss 归约器怎样让"估计什么"不随 DP / CP / micro-batch 的物理切分改写。
// 复用 slime_megatron_train_step_figures.mjs 的同一批样本与调度：4 个逻辑 rollout、5 条训练样本
// （rollout 2 是 compact 扇出的两个片段，s0 含一个工具 token）。
// 源码基线：THUDM/slime@681b3adca54105d5ecd3fb822fa0dc58a427e0f9。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
// 要讲清楚：
//  1. token 均值、sample 均值、rollout 均值是三个不同的估计量，同一批数据给出三个不同的数。
//  2. prompt 分组只定义 reward 的相对基线；样本数 ≠ n × rollout_batch_size 时回落成一个大组。
//  3. 每个 micro-batch、每个 CP rank 只算自己的分子，分母是切分前算好的整 rollout mask 和；
//     loss × M/G × world → Megatron ÷ M → DDP 在 dp·cp 组平均，最终恰好是 Σ_g L_g / G。
//     若用局部分母，程序照常运行但数值变成另一个目标。
//  4. rejection 只改分子的 mask，分母仍是原始 rollout_mask_sums；报告指标用改后 reducer，
//     mismatch 指标用原 reducer。
//
// 布局：四个面板。P1 三种估计量；P2 reward → advantage 的分组回落；P3 DP×CP×mb 归约账本与缩放链；
// P4 rejection 的分子/分母分离与 per-token 模式的报告分母。acc1 标承重结果，acc2 标错误口径与代价。
//
// 用法：node tools/figs/svg/slime_loss_reducer_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { CFG, SAMPLES, model as trainModel, ownedResponseIdx } from './slime_megatron_train_step_figures.mjs';

// ---------------- 冻结的示例输入 ----------------
// 逐 token 目标项 ℓ_it（policy surrogate 已算好的值，只看归约）。s0 下标 3 是工具 token，值 9 不进入分子。
export const LOSS = Object.freeze({
  s0: [1, 2, 1, 9, 2, 1, 3, 4],
  s1: [3, 5],
  s2a: [1, 1, 2, 2],
  s2b: [2, 2, 2, 2, 3, 3],
  s3: [6, 2],
});
export const RAW_REWARDS = Object.freeze([1, 0, 1, 1, 0]); // s0 s1 s2a s2b s3
// rejection 例：自定义 TIS/RS 函数把 s3 的第二个 token 拒掉
export const REJECTED = Object.freeze({ s3: [1, 0] });

const maskSum = (s) => s.lossMask.reduce((a, b) => a + b, 0);
export const ROLLOUT_MASK_SUMS = (() => {
  const totals = new Map();
  SAMPLES.forEach((s) => totals.set(s.rolloutId, (totals.get(s.rolloutId) ?? 0) + maskSum(s)));
  return Object.freeze(SAMPLES.map((s) => totals.get(s.rolloutId)));
})();

// ---------------- cp_utils.py::get_sum_of_sample_mean 的两个分支 ----------------
// 本 CP rank 上一条样本的分子：只对自己负责的 response 下标求 Σ ℓ·m
export function localNumerator(sample, cpRank, cpSize, mask = sample.lossMask, loss = LOSS[sample.name]) {
  return ownedResponseIdx(sample, cpRank, cpSize).reduce((a, i) => a + loss[i] * mask[i], 0);
}
export function sumOfSampleMean(samples, denoms, cpRank, cpSize, masks = null) {
  return samples.reduce((a, s, k) => a + localNumerator(s, cpRank, cpSize, masks ? masks[k] : s.lossMask) / Math.max(denoms[k], 1), 0);
}
export function sumOfToken(samples, cpRank, cpSize) {
  return samples.reduce((a, s) => a + localNumerator(s, cpRank, cpSize), 0);
}
// loss_function 开头的 num_tokens = Σ clamp(mask.sum, 1)（完整 mask，每个 CP rank 都算一遍）
export const numTokens = (samples) => samples.reduce((a, s) => a + Math.max(maskSum(s), 1), 0);

// ---------------- 三种估计量（cp=1 的全批口径） ----------------
export function threeMeans() {
  const N = SAMPLES.map((s) => localNumerator(s, 0, 1));
  const D = SAMPLES.map(maskSum);
  const byRollout = new Map();
  SAMPLES.forEach((s, i) => { const g = byRollout.get(s.rolloutId) ?? { N: 0, D: 0 }; g.N += N[i]; g.D += D[i]; byRollout.set(s.rolloutId, g); });
  const G = byRollout.size;
  const token = N.reduce((a, b) => a + b, 0) / D.reduce((a, d) => a + Math.max(d, 1), 0);
  const sample = N.reduce((a, n, i) => a + n / Math.max(D[i], 1), 0) / SAMPLES.length;
  const rolloutMeans = [...byRollout.values()].map((g) => g.N / Math.max(g.D, 1));
  const rollout = rolloutMeans.reduce((a, b) => a + b, 0) / G;
  return { N, D, G, I: SAMPLES.length, sumN: N.reduce((a, b) => a + b, 0), sumD: D.reduce((a, d) => a + Math.max(d, 1), 0), token, sample, rolloutMeans, rollout };
}

// ---------------- rollout.py::_post_process_rewards（grpo，rewards_normalization 开） ----------------
export function rewardPostProcess(raw = RAW_REWARDS, { n = CFG.nSamplesPerPrompt, rolloutBatchSize = CFG.rolloutBatchSize, std = true } = {}) {
  const groupSize = raw.length === n * rolloutBatchSize ? n : raw.length;
  const groups = [];
  for (let i = 0; i < raw.length; i += groupSize) groups.push(raw.slice(i, i + groupSize));
  const out = groups.flatMap((g) => {
    const mean = g.reduce((a, b) => a + b, 0) / g.length;
    const centered = g.map((r) => r - mean);
    if (!std) return centered;
    const variance = centered.reduce((a, r) => a + r * r, 0) / (g.length - 1); // torch.std 无偏
    return centered.map((r) => r / (Math.sqrt(variance) + 1e-6));
  });
  return { groupSize, fallback: groupSize !== n, groups, rewards: out };
}

// ---------------- 账本：DP × CP × micro-batch 的分子与缩放链 ----------------
export function ledger(cfg = CFG) {
  const tm = trainModel(cfg);
  const G = cfg.globalBatchSize; const M = tm.dynamic.numMicrobatches[0]; const world = cfg.dpSize * cfg.cpSize;
  const ranks = [];
  for (let dp = 0; dp < cfg.dpSize; dp += 1) {
    for (let cp = 0; cp < cfg.cpSize; cp += 1) {
      const mbs = tm.dynamic.microBatchIndices[dp].map((locals) => locals.map((j) => SAMPLES[tm.dynamic.partitions[dp][j]]));
      const cells = mbs.map((samples) => {
        const denoms = samples.map((s) => ROLLOUT_MASK_SUMS[SAMPLES.indexOf(s)]);
        const localDenoms = samples.map((s) => maskSum(s)); // 错误口径：只看本 mb 自己的 mask
        const parts = samples.map((s, k) => ({ name: s.name, numerator: localNumerator(s, cp, cfg.cpSize), denom: denoms[k], localDenom: localDenoms[k], owned: ownedResponseIdx(s, cp, cfg.cpSize) }));
        return {
          samples: samples.map((s) => s.name), parts,
          partial: sumOfSampleMean(samples, denoms, cp, cfg.cpSize),
          partialLocalDenom: sumOfSampleMean(samples, localDenoms, cp, cfg.cpSize),
          tokenSum: sumOfToken(samples, cp, cfg.cpSize),
          numTokens: numTokens(samples),
        };
      });
      const sum = (key) => cells.reduce((a, c) => a + c[key], 0);
      ranks.push({ dp, cp, cells, rankPartial: sum('partial'), rankPartialLocalDenom: sum('partialLocalDenom'), rankTokenSum: sum('tokenSum'), rankNumTokens: sum('numTokens') });
    }
  }
  const total = ranks.reduce((a, r) => a + r.rankPartial, 0);
  const totalLocalDenom = ranks.reduce((a, r) => a + r.rankPartialLocalDenom, 0);
  // 缩放链：每个 mb 的 loss × M/G × world；Megatron ÷ M；反向累加后 DDP 在 dp·cp 组平均（÷ world）
  const scale = (M / G) * world;
  const perMbAfterMegatron = scale / M; // = world / G
  const finalGrad = (total * perMbAfterMegatron) / world; // = total / G
  const finalGradLocalDenom = (totalLocalDenom * perMbAfterMegatron) / world;
  // 报告：reduce_train_step_metrics，per-rollout 分母是常量 G，cp_factor 1
  const reportRollout = total / G;
  // per-token：values[0] = Σ_rank Σ_mb num_tokens（all-reduce 覆盖 dp·cp，每个 CP rank 用完整 mask 各算一遍）
  const allReducedNumTokens = ranks.reduce((a, r) => a + r.rankNumTokens, 0);
  const allReducedTokenSum = ranks.reduce((a, r) => a + r.rankTokenSum, 0);
  const reportToken = (allReducedTokenSum * cfg.cpSize) / allReducedNumTokens;
  // per-token 梯度（Megatron 依赖契约）：每 rank loss × cp；流水线不逐 mb 相除，DDP 不缩放（梯度求和），
  // finalize_model_grads 在 dp·cp 上汇总 num_tokens 后整体 × 1/Σnum_tokens
  const perTokenGrad = (cfg.cpSize * allReducedTokenSum) / allReducedNumTokens;
  return { G, M, world, ranks, total, totalLocalDenom, scale, perMbAfterMegatron, finalGrad, finalGradLocalDenom, reportRollout, allReducedNumTokens, allReducedTokenSum, reportToken, perTokenGrad, schedule: tm.rankView };
}

// ---------------- rejection：分子 mask 改、分母不改 ----------------
export function rejection() {
  const s3 = SAMPLES.find((s) => s.name === 's3');
  const denom = ROLLOUT_MASK_SUMS[SAMPLES.indexOf(s3)];
  const original = localNumerator(s3, 0, 1) / denom;
  const modifiedNumerator = localNumerator(s3, 0, 1, REJECTED.s3);
  const kept = modifiedNumerator / denom;
  const naive = modifiedNumerator / Math.max(REJECTED.s3.reduce((a, b) => a + b, 0), 1);
  return { denom, original, modifiedNumerator, kept, naive, survivorTokens: REJECTED.s3.reduce((a, b) => a + b, 0) };
}

// ---------------- 序列统计量的 CP 重建：cp_utils.py::all_gather_with_cp + ppo_utils.py::compute_gspo_kl ----------------
// s0 的逐 token old−new logprob 差（示意输入，下标 3 被 mask）；GSPO 先在 CP 上重建完整 response，求 masked 均值，再展开回本地 token
export const GSPO_DIFF = Object.freeze({ s0: [0.1, 0.2, 0.1, 0.9, 0.2, 0.1, 0.3, 0.4] });
export function gspoReplay(cfg = CFG) {
  const s0 = SAMPLES[0]; const d = GSPO_DIFF.s0; const mask = s0.lossMask;
  const ranks = Array.from({ length: cfg.cpSize }, (_, cp) => {
    const own = ownedResponseIdx(s0, cp, cfg.cpSize);
    const localDen = own.reduce((a, i) => a + mask[i], 0);
    const localNum = own.reduce((a, i) => a + d[i] * mask[i], 0);
    // all_gather_with_cp：本地片放回完整 response 的位置，其余位置补零
    return { cp, own, local: own.map((i) => d[i]), padded: d.map((v, i) => (own.includes(i) ? v : 0)), localMean: localNum / Math.max(localDen, 1) };
  });
  const reduced = d.map((_, i) => ranks.reduce((a, r) => a + r.padded[i], 0)); // dist.nn.all_reduce（求和）
  const num = reduced.reduce((a, v, i) => a + v * mask[i], 0);
  const den = mask.reduce((a, b) => a + b, 0);
  const seqKl = num / Math.max(den, 1);
  return { ranks, reduced, num, den, seqKl, expandedCounts: ranks.map((r) => r.own.length), emptyOwned: ownedResponseIdx(SAMPLES[3], 0, cfg.cpSize) };
}

// ---------------- PPO 的 reward 注入：loss.py::compute_advantages_and_returns 在 gather 前对 cp0 的本地 KL 张量 k[-1] += reward ----------------
export function ppoRewardSlot(totalLen, responseLen, cpSize = CFG.cpSize) {
  const own = ownedResponseIdx({ totalLen, responseLen }, 0, cpSize);
  const chunk = Math.ceil(totalLen / (2 * cpSize)); const pad = 2 * cpSize * chunk - totalLen;
  const slot = own.length ? own.at(-1) : null;
  return { totalLen, responseLen, chunk, pad, own, slot, outcome: slot === null ? 'IndexError' : slot === responseLen - 1 ? 'ok' : 'misplaced', tailCovers: chunk >= pad + 2 };
}

export function model(cfg = CFG) {
  const ppoSlots = [...SAMPLES.map((smp) => ({ name: smp.name, ...ppoRewardSlot(smp.totalLen, smp.responseLen, cfg.cpSize) })), { name: 'T10/R8', ...ppoRewardSlot(10, 8, cfg.cpSize) }];
  return { cfg, samples: SAMPLES, loss: LOSS, rolloutMaskSums: ROLLOUT_MASK_SUMS, means: threeMeans(), rewards: rewardPostProcess(), rewardsPerPrompt: rewardPostProcess([1, 0, 1, 0]), ledger: ledger(cfg), rejection: rejection(), gspo: gspoReplay(cfg), ppoSlots };
}

// ---------------- 渲染 ----------------
const esc = (v) => String(v).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const f = (x, d = 3) => Number.isInteger(x) ? String(x) : String(Number(x.toFixed(d)));
const STYLE = `
  text{font-family:"Segoe UI","Microsoft YaHei","PingFang SC",system-ui,sans-serif;fill:#2A313B}
  .ti{font-size:19px;font-weight:700;fill:#1F2430}
  .su{font-size:12px;fill:#747C88}
  .pt{font-size:14px;font-weight:700}
  .tx{font-size:12px;fill:#38414D}
  .sm{font-size:10.5px;fill:#5B6470}
  .cap{font-size:11.5px;fill:#5B6470}
  .mono{font-family:"Cascadia Mono",Consolas,"Courier New",monospace;font-size:11px;fill:#38414D}
  .mono2{font-family:"Cascadia Mono",Consolas,"Courier New",monospace;font-size:9.5px;fill:#38414D}
  .panel{fill:#FBFCFE;stroke:#D9DEE7;stroke-width:1.2}
  .neutral{fill:#fff;stroke:#AEB6C2;stroke-width:1.2}
  .ghost{fill:#F5F7FA;stroke:#D9DEE7;stroke-width:1.1}
  .acc1{fill:#EAF1FD;stroke:#2563EB;stroke-width:1.5}
  .acc2{fill:#FCF1E6;stroke:#C3651F;stroke-width:1.5}
  .cell{fill:#fff;stroke:#AEB6C2;stroke-width:1.1}
  .h1{fill:#DCE7FB;stroke:#AEB6C2;stroke-width:1.1}
  .x{fill:#EEF1F5;stroke:#D9DEE7;stroke-width:1}
  .main{fill:none;stroke:#2563EB;stroke-width:2;marker-end:url(#arrowMain)}
  .aux{fill:none;stroke:#AEB6C2;stroke-width:1.3;stroke-dasharray:5 4;marker-end:url(#arrowAux)}
`;

function render(m) {
  const W = 1180; const H = 1280;
  const o = [];
  const rect = (x, y, w, h, cls = 'neutral', r = 7) => o.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" class="${cls}"/>`);
  const text = (x, y, s, cls = 'tx', anchor = 'start') => o.push(`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${esc(s)}</text>`);
  const arrow = (x1, y1, x2, y2, cls = 'main') => o.push(`<path d="M${x1} ${y1} L${x2} ${y2}" class="${cls}"/>`);
  const c = m.cfg; const L = m.ledger; const T = m.means;

  o.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="title desc">`);
  o.push('<title id="title">slime loss 归约账本：同一批样本在 token、sample、rollout 三种口径下的值，以及 DP×CP×micro-batch 切分后如何还原 rollout 均值</title>');
  o.push('<desc id="desc">面板 1 用五条样本算出 token 均值、sample 均值与 rollout 均值三个不同的数；面板 2 展示 reward 分组归一化在样本数不等于 n × rollout_batch_size 时回落成一个大组；面板 3 按 DP rank、CP rank、micro-batch 列出每个局部分子与整 rollout 分母，串起 loss 预缩放、Megatron 除 micro-batch 数与 DDP 平均，最终还原 Σ_g L_g / G，并对照局部分母的错误值与 per-token 报告分母；面板 4 展示 rejection 只改分子 mask 而分母不变；面板 5 回放 s0 的 GSPO 序列 ratio 如何先经 CP all-reduce 重建再展开回本地 token，并列出 PPO 在 gather 之前把 reward 加到 cp0 本地末位时各样本的落点。</desc>');
  o.push(`<defs><marker id="arrowMain" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#2563EB"/></marker><marker id="arrowAux" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#AEB6C2"/></marker></defs>`);
  o.push(`<style>${STYLE}</style><rect width="${W}" height="${H}" fill="white"/>`);

  text(24, 34, '同一批样本、三种估计量、一条不随切分改写的归约链', 'ti');
  text(24, 56, `与训练后端页同一实例：${T.G} 个逻辑 rollout、${T.I} 条样本（r2 = s2a + s2b 扇出，s0 下标 3 为工具 token）· dp=${c.dpSize} cp=${c.cpSize} · 每 rank M=${L.M} 个 micro-batch · G=${L.G}`, 'su');

  // ---- P1：三种估计量 ----
  const y1 = 74; const h1 = 264;
  rect(24, y1, 700, h1, 'panel', 10);
  text(40, y1 + 24, '1  三种均值是三个估计量：token、sample、rollout', 'pt');
  const tx0 = 40; const ty = y1 + 40; const cols = [64, 250, 60, 60, 60, 80];
  const heads = ['样本', '逐 token ℓ（灰=mask 0）', 'N_i', 'D_i', 'D_g', 'N_i / D_g'];
  let cx = tx0;
  heads.forEach((h, i) => { text(cx + cols[i] / 2, ty + 12, h, 'sm', 'middle'); cx += cols[i]; });
  m.samples.forEach((s, r) => {
    const yy = ty + 20 + r * 24; let x = tx0;
    const cells = [s.name, null, T.N[r], T.D[r], m.rolloutMaskSums[r], f(T.N[r] / m.rolloutMaskSums[r])];
    cells.forEach((v, i) => {
      if (i === 1) {
        LOSS[s.name].forEach((l, k) => { rect(x + k * 26, yy, 24, 20, s.lossMask[k] ? 'h1' : 'x', 3); text(x + k * 26 + 12, yy + 14, l, 'mono2', 'middle'); });
      } else {
        rect(x, yy, cols[i] - 4, 20, i === 4 ? 'acc1' : 'cell', 3);
        text(x + (cols[i] - 4) / 2, yy + 14, v, 'mono', 'middle');
      }
      x += cols[i];
    });
  });
  const fy = ty + 20 + m.samples.length * 24 + 12;
  rect(40, fy, 670, 60, 'ghost', 8);
  text(52, fy + 18, `L_token = ΣN / Σmax(D_i,1) = ${T.sumN} / ${T.sumD} = ${f(T.token)}     L_sample = (1/I) Σ N_i/max(D_i,1) = ${f(T.sample)}`, 'mono');
  text(52, fy + 36, `L_rollout = (1/G) Σ_g ΣN_i / max(D_g,1) = (${T.rolloutMeans.map((v) => f(v)).join(' + ')}) / ${T.G} = ${f(T.rollout)}  ← live 默认`, 'mono');
  text(52, fy + 52, 'sample 均值给 r2 两票；token 均值让长 response 权重大；rollout 均值先在 r2 内按 token 加权再等权。', 'cap');

  // ---- P2：reward 分组 ----
  const x2 = 736; const w2 = 1156 - x2;
  rect(x2, y1, w2, h1, 'panel', 10);
  text(x2 + 16, y1 + 24, '2  prompt 分组只定义 reward 基线', 'pt');
  const R = m.rewards; const RP = m.rewardsPerPrompt;
  text(x2 + 16, y1 + 46, `raw_reward = [${RAW_REWARDS.join(', ')}]（${RAW_REWARDS.length} 条）`, 'mono');
  text(x2 + 16, y1 + 64, `${RAW_REWARDS.length} ≠ n × rollout_batch_size = ${c.nSamplesPerPrompt} × ${c.rolloutBatchSize} → view(-1, ${R.groupSize})：一个大组`, 'tx');
  rect(x2 + 16, y1 + 74, w2 - 32, 44, 'acc2', 6);
  text(x2 + 26, y1 + 92, `减组均值、除无偏 std+1e-6 → [${R.rewards.map((v) => f(v, 2)).join(', ')}]`, 'mono2');
  text(x2 + 26, y1 + 110, '扇出让 P1 的两个 rollout 和 P0 混成一组，基线不再是同一 prompt', 'sm');
  text(x2 + 16, y1 + 138, `对照：4 条无扇出 [1,0,1,0] → 按 n=${c.nSamplesPerPrompt} 分两组 → [${RP.rewards.map((v) => f(v, 2)).join(', ')}]`, 'tx');
  text(x2 + 16, y1 + 158, 'grpo/gspo/cispo：returns = ones_like(kl) × reward，逐 token 广播；', 'tx');
  text(x2 + 16, y1 + 176, 'kl_coef=0 时 kl 全零。ppo 在 cp0 本地末位加 reward 再做 GAE，', 'tx');
  text(x2 + 16, y1 + 194, '末 token 不在 cp0 本地时会抛错或错位（面板 5）。', 'tx');
  text(x2 + 16, y1 + 218, '不规则扇出要保持按 prompt 分组，须用自定义 reward 后处理', 'cap');
  text(x2 + 16, y1 + 234, '或自定义 converter 显式恢复；reducer 不会事后修正基线。', 'cap');

  // ---- P3：账本 ----
  const y3 = y1 + h1 + 12; const h3 = 490;
  rect(24, y3, 1132, h3, 'panel', 10);
  text(40, y3 + 24, `3  DP × CP × micro-batch 账本：每格只算本地分子 ÷ 整 rollout 分母，缩放链还原 Σ_g L_g / G = ${f(L.finalGrad)}`, 'pt');
  const gx = 40; const gy = y3 + 44; const colMb = 330; const rowH = 58;
  text(gx + 90, gy + 12, 'rank (dp, cp)', 'sm', 'middle');
  for (let k = 0; k < L.M; k += 1) text(gx + 180 + k * colMb + colMb / 2, gy + 12, `micro-batch ${k}：样本、本 rank 负责的 response 下标、分子 / D_g`, 'sm', 'middle');
  text(gx + 180 + L.M * colMb + 60, gy + 12, 'rank 小计', 'sm', 'middle');
  L.ranks.forEach((r, i) => {
    const yy = gy + 20 + i * rowH;
    rect(gx, yy, 172, rowH - 6, 'cell', 5);
    text(gx + 86, yy + 22, `dp${r.dp} · cp${r.cp}`, 'mono', 'middle');
    text(gx + 86, yy + 40, `mbs ${L.schedule[r.dp].mbs.map((mb) => `[${mb.join(' ')}]`).join(' ')}`, 'mono2', 'middle');
    r.cells.forEach((cell, k) => {
      const x = gx + 180 + k * colMb;
      const empty = cell.parts.every((p) => p.owned.length === 0);
      rect(x, yy, colMb - 8, rowH - 6, empty ? 'acc2' : 'neutral', 5);
      cell.parts.forEach((p, j) => {
        text(x + 10, yy + 18 + j * 15, `${p.name}{${p.owned.join(',')}}: ${p.numerator} / ${p.denom} = ${f(p.numerator / Math.max(p.denom, 1))}`, 'mono2');
      });
      text(x + colMb - 16, yy + rowH - 14, `Σ = ${f(cell.partial)}`, 'mono', 'end');
      if (empty) text(x + 10, yy + rowH - 14, '空 rank：分子 0，仍走 collective', 'sm');
    });
    const sx = gx + 180 + L.M * colMb;
    rect(sx, yy, 120, rowH - 6, 'acc1', 5);
    text(sx + 60, yy + 30, f(r.rankPartial), 'mono', 'middle');
  });
  const chainY = gy + 20 + L.ranks.length * rowH + 8;
  const chain = [
    [`每 mb loss × M/G × world`, `= × ${L.M}/${L.G} × ${L.world} = × ${f(L.scale)}`, 'neutral'],
    ['Megatron ÷ M', `每 mb 净系数 ${f(L.perMbAfterMegatron)} = world / G`, 'neutral'],
    ['反向累加，DDP ÷ world', `Σ_rank ${f(L.total)} × ${f(L.perMbAfterMegatron)} / ${L.world}`, 'neutral'],
    [`= ${f(L.finalGrad)} = L_rollout`, 'cp 不出现；dp·cp 任意分解结果相同', 'acc1'],
  ];
  chain.forEach((b, i) => {
    const x = gx + i * 280;
    rect(x, chainY, 268, 50, b[2], 6);
    text(x + 134, chainY + 20, b[0], 'mono', 'middle');
    text(x + 134, chainY + 38, b[1], 'sm', 'middle');
    if (i < chain.length - 1) arrow(x + 268, chainY + 25, x + 280, chainY + 25, 'main');
  });
  const wy = chainY + 62;
  rect(gx, wy, 548, 82, 'acc2', 6);
  text(gx + 12, wy + 20, `错误口径：分母改成本 mb 自己的 mask 和 → 总和 ${f(L.totalLocalDenom)}，最终 ${f(L.finalGradLocalDenom)} ≠ ${f(L.finalGrad)}`, 'tx');
  text(gx + 12, wy + 38, 's2a、s2b 各用 4 与 6 作分母：同一次执行被投两票；程序不报错，目标函数已变。', 'sm');
  text(gx + 12, wy + 54, 'tests/test_cp_utils.py::test_split_with_per_mb_denom_would_be_wrong 锁定此差异。', 'sm');
  rect(gx + 560, wy, 552, 82, 'ghost', 6);
  text(gx + 572, wy + 20, `报告：reduce_train_step_metrics 把 Σ_mb 在 dp·cp 上 all-reduce = ${f(L.total)}，/ G = ${f(L.reportRollout)}（cp_factor 1）`, 'tx');
  text(gx + 572, wy + 38, `per-token 模式：values[0] = Σ_rank num_tokens = ${L.allReducedNumTokens}（每个 CP rank 用完整 mask 各算一遍），`, 'sm');
  text(gx + 572, wy + 54, `分子 Σ = ${L.allReducedTokenSum}，× cp_size ${c.cpSize} / ${L.allReducedNumTokens} = ${f(L.reportToken)} = L_token（报告）。`, 'sm');
  text(gx + 572, wy + 72, `per-token 梯度（Megatron 契约）：不逐 mb 除、DDP 不缩放，finalize 除 Σnum_tokens → ${c.cpSize} × ${L.allReducedTokenSum} / ${L.allReducedNumTokens} = ${f(L.perTokenGrad)}`, 'sm');
  text(gx, wy + 102, 'CP 分子可加：两 rank 分子之和等于 cp=1 的分子。空 rank 的本地 logprob 为空，policy_loss_function 加 0·logits.sum() 保持反向连通，它照常参加 DDP 与白化的集合通信。', 'cap');

  // ---- P4：rejection ----
  const y4 = y3 + h3 + 12; const h4 = 170;
  rect(24, y4, 1132, h4, 'panel', 10);
  const rj = m.rejection;
  text(40, y4 + 24, '4  rejection 改分子的 mask，分母仍是原始 rollout_mask_sums；mismatch 指标用改前的 reducer', 'pt');
  const bx = 40; const byy = y4 + 40;
  const boxes = [
    [`原始：s3 ℓ=[${LOSS.s3.join(',')}] mask=[${SAMPLES.find((s) => s.name === 's3').lossMask.join(',')}]`, `${LOSS.s3.reduce((a, b) => a + b, 0)} / ${rj.denom} = ${f(rj.original)}`, 'neutral'],
    [`自定义 TIS hook 返回 mask=[${REJECTED.s3.join(',')}]（IS 权重取 1）`, `分子 ${rj.modifiedNumerator} / 原分母 ${rj.denom} = ${f(rj.kept)}  ← 源码`, 'acc1'],
    [`若按 survivor 重算分母（${rj.survivorTokens} 个）`, `${rj.modifiedNumerator} / ${rj.survivorTokens} = ${f(rj.naive)}：拒得越多权重越大`, 'acc2'],
  ];
  boxes.forEach((b, i) => {
    const x = bx + i * 372;
    rect(x, byy, 360, 50, b[2], 6);
    text(x + 180, byy + 20, b[0], 'mono2', 'middle');
    text(x + 180, byy + 38, b[1], 'mono', 'middle');
    if (i < boxes.length - 1) arrow(x + 360, byy + 25, x + 372, byy + 25, i === 0 ? 'main' : 'aux');
  });
  text(bx, byy + 74, 'policy_loss_function 用 modified masks + batch["rollout_mask_sums"] 重建 reducer：pg_loss、pg_clipfrac、ppo_kl、entropy、kl_loss 都用它；', 'tx');
  text(bx, byy + 92, 'ois 与 tis_* 指标用改前的 reducer，否则被拒 token 从分母消失、truncate 类指标被推向 0。per-token 的 num_tokens 也在调用 hook 前按原始 mask 算好。', 'tx');
  text(bx, byy + 112, 'custom_pg_loss_reducer 只收到 lengths、masks 与 per-token 开关，看不到 rollout_ids / rollout_mask_sums，因此无法自行重建 rollout 均值。', 'cap');

  // ---- P5：序列统计量的 CP 重建与 PPO reward 落点 ----
  const y5 = y4 + h4 + 12; const h5 = 212;
  rect(24, y5, 1132, h5, 'panel', 10);
  const G5 = m.gspo; const d = GSPO_DIFF.s0; const s0m = SAMPLES[0].lossMask;
  text(40, y5 + 24, '5  序列统计量先在 CP 上重建再展开；PPO 的 reward 在重建之前加到 cp0 本地末位', 'pt');
  const cx0 = 190; const cw = 40;
  d.forEach((_, i) => text(cx0 + i * cw + (cw - 4) / 2, y5 + 44, i, 'sm', 'middle'));
  text(cx0 - 8, y5 + 44, 'response 下标', 'sm', 'end');
  const row5 = (yy, label, vals, clsFn, note) => {
    text(cx0 - 8, yy + 14, label, 'mono2', 'end');
    vals.forEach((v, i) => { rect(cx0 + i * cw, yy, cw - 4, 20, clsFn(v, i), 3); text(cx0 + i * cw + (cw - 4) / 2, yy + 14, v === 0 ? '0' : String(Number(v.toFixed(3))), 'mono2', 'middle'); });
    if (note) text(cx0 + vals.length * cw + 6, yy + 14, note, 'mono2');
  };
  row5(y5 + 52, 's0 old−new', d, (v, i) => (s0m[i] ? 'cell' : 'x'));
  G5.ranks.forEach((r, k) => row5(y5 + 78 + k * 26, `cp${r.cp} 本地 {${r.own.length > 2 ? `${r.own[0]}..${r.own.at(-1)}` : r.own.join(',')}} 补零`, r.padded, (v) => (v === 0 ? 'x' : 'h1'), `本地均值 ${f(r.localMean)}`));
  row5(y5 + 78 + G5.ranks.length * 26, 'all-reduce 求和', G5.reduced, (v, i) => (s0m[i] ? 'acc1' : 'x'), `masked 均值 ${f(G5.num)} / ${G5.den} = ${f(G5.seqKl)}`);
  text(40, y5 + 176, `GSPO 用 ${f(G5.seqKl)} 展开回 cp0 的 ${G5.expandedCounts[0]} 个、cp1 的 ${G5.expandedCounts[1]} 个本地 token；各 rank 只看本地片会得到 ${G5.ranks.map((r) => f(r.localMean)).join(' 与 ')} 两个不同的 log-ratio。`, 'cap');
  text(40, y5 + 194, 's2b 在 cp0 本地为空，仍以全零向量参加同一次 all-reduce；反向时可微 all-reduce 的梯度再在 CP 组上 all-reduce 一次。', 'cap');
  const tx5 = 720; const cws5 = [56, 56, 70, 110, 120];
  const heads5 = ['样本', 'T/R', 'chunk·pad', 'cp0 本地下标', 'k[-1] 落点'];
  let hx = tx5;
  heads5.forEach((h, i) => { text(hx + cws5[i] / 2, y5 + 44, h, 'sm', 'middle'); hx += cws5[i]; });
  m.ppoSlots.forEach((p, r) => {
    const yy = y5 + 52 + r * 22; let x = tx5;
    const outcome = p.outcome === 'ok' ? `ok → ${p.slot}` : p.outcome === 'IndexError' ? 'IndexError' : `错位 → ${p.slot}（应为 ${p.responseLen - 1}）`;
    [p.name, `${p.totalLen}/${p.responseLen}`, `c${p.chunk} p${p.pad}`, `{${p.own.join(',')}}`, outcome].forEach((v, i) => {
      rect(x, yy, cws5[i] - 4, 20, i === 4 ? (p.outcome === 'ok' ? 'acc1' : 'acc2') : 'cell', 3);
      text(x + (cws5[i] - 4) / 2, yy + 14, v, 'mono2', 'middle');
      x += cws5[i];
    });
  });
  text(tx5, y5 + 194, '尾段覆盖最后一个 response logit ⇔ chunk ≥ pad + 2（total ≥ 3）', 'cap');

  text(24, H - 12, '源码基线：THUDM/slime@681b3adca541 · 复现 get_sum_of_sample_mean / loss_function 缩放 / reduce_train_step_metrics / _post_process_rewards / policy_loss_function 的 mask 重建', 'su');
  o.push('</svg>');
  return o.join('\n');
}

const here = dirname(fileURLToPath(import.meta.url));
const defaultOutput = join(here, '..', '..', '..', 'wiki', '02_engineering', '04_posttrain_frameworks', 'slime', 'assets');
const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  const outputDir = process.argv[2] ? process.argv[2] : defaultOutput;
  mkdirSync(outputDir, { recursive: true });
  writeFileSync(join(outputDir, 'slime_loss_reducer_ledger.svg'), `${render(model())}\n`, 'utf8');
}
