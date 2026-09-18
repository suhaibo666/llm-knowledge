// 图：两条各自可在冻结基线上执行的运行，展示 slime 数据层的三次决定性转换。
//  运行 ①（partial，默认 generate）：一条被中断的 Sample 怎样经 DataSource 的 buffer 回收、
//    `Sample.append_response_tokens` 的对齐追加进入下一轮，以及 mask-offpolicy 对整个回收组的影响。
//  运行 ②（无 partial，返回 list[Sample] 的自定义生成函数）：一次 compact 扇出怎样经
//    `RolloutManager._convert_samples_to_train_data` 的 rollout_id 兜底与 `rollout_mask_sums`，
//    再被 `build_dp_schedule` 按 rollout 分步、切成 micro-batch。
//  两条运行都另算一行"reward 归一化开启"：`RolloutManager._post_process_rewards` 只按 reward 总数是否等于
//    n_samples_per_prompt × rollout_batch_size 决定按组 reshape 还是整批当一组；运行 ② 同时给出
//    tests/fanout_test_helpers.py::grpo_normalize_by_group_index 的按 group_index 分组结果作对照。
// 两条运行必须分开：`sglang_rollout.py::abort` 对在途组每个成员取 `.status`/`.response`，`_get_rollout_data`
// 展平只看 data[0]；partial 回收组中已完成成员的早退返回 plain Sample，与返回 list 的自定义生成函数不能共存于一轮。
// 源码基线：THUDM/slime@4c193f1f37509cca70f0e88807a9305b70f63f4e。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
// 要讲清楚四件事：
//  1. Sample 层只有一个写入口：新 token、mask、logprob、版本、状态按同一新增区间追加；
//     中断续生成不覆盖旧前缀，`mask_offpolicy_in_partial_rollout` 在 `generate_and_rm` 入口就把
//     整个回收组里所有已有 response 的成员（含已完成者）的旧区间 mask 清 0。
//  2. DataSource 层只管"下一组 prompt 从哪来、中断组放回哪"：游标顺序取数，buffer 整组 FIFO 回收；
//     abort 只回收至少有一个成员 ABORTED 且已生成 token 的组。
//  3. converter 在展平后、切分前一次性算出每个逻辑 rollout 的完整 mask 分母；
//     扇出片段即使落到不同 micro-batch，仍除以同一个分母，一次逻辑执行只占一份权重。
//  4. 默认 reward 归一化在"每组条数整齐"时按位置分组，扇出使条数变多后退化成整批一组。
//
// 布局：四条横向泳道。①-A：s1 的 token/mask/logprob 时间线；①-B：DataSource 的两轮 get_samples 与 buffer 交接；
// ①-C：运行 ① 的 converter 结果（两种 mask 模式，含分母为 0 的 s0，及归一化开的 rewards）；
// ②：运行 ② 的 converter 字段表（含归一化开的整批一组与按 group_index 分组两行）与 step/micro-batch 条。
// acc1 标决定性映射（rollout_id 兜底、分母广播、片段所在 mb、按组居中），acc2 标回收、中断、被清零与退化分组（代价来源）。
//
// 用法：node tools/figs/svg/slime_sample_data_contract_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// ---------------- 冻结的示例输入 ----------------
export const CFG = Object.freeze({
  nSamplesPerPrompt: 2,
  rolloutBatchSize: 2,
  overSamplingBatchSize: 3,
  numStepsPerRollout: 1,
  dpSize: 1,
  microBatchSize: 1,
  rolloutMaxResponseLen: 4,
  promptIds: [11, 12],
  abortRound: 3,
  resumeRound: 4,
  // --advantage-estimator 默认 grpo；--disable-grpo-std-normalization 未给时 grpo_std_normalization=True
  grpoStdNormalization: true,
});

// ---------------- 对源码算法的最小复现 ----------------

// slime/utils/types.py::Sample.append_response_tokens（复现 tokens / response / loss_mask / logprob / 版本 / 状态与其守卫）
export function appendResponseTokens(sample, { tokens, logProbs = null, trainable = true, finishReason = null, weightVersion = null, text = null }) {
  if (logProbs !== null && logProbs.length !== tokens.length) throw new Error('log_probs length != tokens length');
  if (tokens.length && trainable && logProbs === null) throw new Error('trainable response tokens require rollout log probabilities.');
  if (tokens.length && !trainable) {
    if (logProbs !== null) throw new Error('non-trainable response tokens should not pass rollout log probabilities.');
    logProbs = tokens.map(() => 0.0);
  }
  if (text !== null) sample.response += text;
  const previous = sample.responseLength;
  if (tokens.length) {
    sample.tokens = [...sample.tokens, ...tokens];
    sample.responseLength += tokens.length;
    if (sample.lossMask === null) sample.lossMask = Array.from({ length: previous }, () => 1);
    sample.lossMask = [...sample.lossMask, ...tokens.map(() => (trainable ? 1 : 0))];
  }
  if (logProbs !== null) {
    if (sample.rolloutLogProbs === null) {
      if (trainable && previous) throw new Error('Cannot append trainable rollout log probabilities to a sample with existing response tokens but no existing rollout_log_probs.');
      sample.rolloutLogProbs = Array.from({ length: previous }, () => 0.0);
    }
    sample.rolloutLogProbs = [...sample.rolloutLogProbs, ...logProbs];
  }
  if (finishReason !== null) {
    if (weightVersion !== null) sample.weightVersions = [...sample.weightVersions, weightVersion];
    sample.status = { length: 'TRUNCATED', abort: 'ABORTED', stop: 'COMPLETED' }[finishReason];
  }
  if (sample.lossMask !== null && sample.lossMask.length !== sample.responseLength) throw new Error('loss_mask length != response_length');
  if (sample.rolloutLogProbs !== null && sample.rolloutLogProbs.length !== sample.responseLength) throw new Error('rollout_log_probs length != response_length');
  return sample;
}

export function newSample({ groupIndex, index, promptIds = CFG.promptIds }) {
  return {
    groupIndex, index, rolloutId: null, tokens: [...promptIds], response: '', responseLength: 0,
    lossMask: null, rolloutLogProbs: null, weightVersions: [], status: 'PENDING', reward: null, metadata: {},
  };
}

const clone = (s) => JSON.parse(JSON.stringify(s));

// slime/rollout/sglang_rollout.py::generate_and_rm 的入口两步：先按开关清零已有 mask，再让已完成成员早退
export function recycledEntry(sample, { partialRollout, maskOffpolicy }) {
  if (partialRollout && maskOffpolicy && sample.responseLength > 0) sample.lossMask = Array.from({ length: sample.responseLength }, () => 0);
  const skip = sample.status === 'COMPLETED' || sample.status === 'TRUNCATED';
  return { sample, skip };
}

// slime/rollout/sglang_rollout.py::abort 的 partial 回收：组内至少一个成员 ABORTED 且已生成 token 才回收；
// 回收时给有 response 文本且尚无 start_rollout_id 的成员写入本轮 id。返回是否回收。
export function collectAbortedGroup(group, rolloutId) {
  if (!group.some((s) => s.status === 'ABORTED' && s.responseLength > 0)) return false;
  for (const s of group) if (s.response && !('start_rollout_id' in s.metadata)) s.metadata.start_rollout_id = rolloutId;
  return true;
}

// slime/rollout/data_source.py::RolloutDataSourceWithBuffer.get_samples / add_samples / pop_first
export function makeDataSource(cfg = CFG) {
  const state = { sampleOffset: 0, sampleGroupIndex: 0, sampleIndex: 0, buffer: [] };
  const fromDataset = (numPrompts) => {
    const groups = [];
    for (let p = 0; p < numPrompts; p += 1) {
      const group = [];
      for (let k = 0; k < cfg.nSamplesPerPrompt; k += 1) {
        group.push(newSample({ groupIndex: state.sampleGroupIndex, index: state.sampleIndex }));
        state.sampleIndex += 1;
      }
      state.sampleGroupIndex += 1;
      groups.push(group);
    }
    state.sampleOffset += numPrompts;
    return groups;
  };
  return {
    state,
    getSamples(numSamples) {
      let fromBuffer = [];
      if (state.buffer.length && numSamples) {
        const n = Math.min(state.buffer.length, numSamples); // pop_first
        fromBuffer = state.buffer.slice(0, n);
        state.buffer.splice(0, n);
      }
      const remaining = numSamples - fromBuffer.length;
      return remaining === 0 ? fromBuffer : [...fromBuffer, ...fromDataset(remaining)];
    },
    addSamples(groups) {
      for (const g of groups) {
        if (g.length !== cfg.nSamplesPerPrompt) throw new Error('the length of the elements of samples must be equal to n_samples_per_prompt');
        state.buffer.push(g);
      }
    },
  };
}

// slime/ray/rollout.py::RolloutManager._convert_samples_to_train_data（rollout_id 兜底、loss_masks、rollout_mask_sums）
export function convert(samples) {
  const rolloutIds = samples.map((s) => s.rolloutId);
  const existed = new Set(rolloutIds.filter((r) => r !== null));
  let tmp = 0;
  for (let i = 0; i < rolloutIds.length; i += 1) {
    if (rolloutIds[i] === null) {
      while (existed.has(tmp)) tmp += 1;
      rolloutIds[i] = tmp;
      existed.add(tmp);
    }
  }
  const lossMasks = samples.map((s) => (s.lossMask === null ? Array.from({ length: s.responseLength }, () => 1) : s.lossMask));
  const maskSums = lossMasks.map((m) => m.reduce((a, b) => a + b, 0));
  const totals = new Map();
  rolloutIds.forEach((rid, i) => totals.set(rid, (totals.get(rid) ?? 0) + maskSums[i]));
  return {
    tokens: samples.map((s) => s.tokens),
    totalLengths: samples.map((s) => s.tokens.length),
    responseLengths: samples.map((s) => s.responseLength),
    rewards: samples.map((s) => s.reward),
    truncated: samples.map((s) => (s.status === 'TRUNCATED' ? 1 : 0)),
    sampleIndices: samples.map((s) => s.index),
    rolloutIds,
    lossMasks,
    maskSums,
    rolloutMaskSums: rolloutIds.map((rid) => totals.get(rid)),
  };
}

// torch.Tensor.mean / std（std 为无偏估计，除以 n−1）
const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
const stdUnbiased = (xs) => {
  const m = mean(xs);
  return Math.sqrt(xs.reduce((a, x) => a + (x - m) ** 2, 0) / (xs.length - 1));
};
function centerAndScale(xs, useStd) {
  const m = mean(xs);
  let ys = xs.map((x) => x - m);
  if (useStd) {
    const s = stdUnbiased(ys);
    ys = ys.map((y) => y / (s + 1e-6));
  }
  return ys;
}

// slime/ray/rollout.py::RolloutManager._post_process_rewards 的内置分支
// （advantage_estimator ∈ grpo/gspo/cispo/reinforce_plus_plus_baseline 且 rewards_normalization）：
// reward 数等于 n_samples_per_prompt × rollout_batch_size 时按位置每 n 条一组，否则 view(-1, 总数) 整批一组。
export function postProcessRewards(rawRewards, cfg = CFG) {
  const groupSize = rawRewards.length === cfg.nSamplesPerPrompt * cfg.rolloutBatchSize ? cfg.nSamplesPerPrompt : rawRewards.length;
  const rewards = [];
  for (let i = 0; i < rawRewards.length; i += groupSize) rewards.push(...centerAndScale(rawRewards.slice(i, i + groupSize), cfg.grpoStdNormalization));
  return { groupSize, numGroups: rawRewards.length / groupSize, rewards };
}

// tests/fanout_test_helpers.py::grpo_normalize_by_group_index：按 Sample.group_index 分组，同样居中并按 std 缩放
export function normalizeByGroupIndex(samples, cfg = CFG) {
  const groups = new Map();
  samples.forEach((s, i) => { if (!groups.has(s.groupIndex)) groups.set(s.groupIndex, []); groups.get(s.groupIndex).push(i); });
  const out = samples.map(() => 0);
  for (const positions of groups.values()) {
    const scaled = centerAndScale(positions.map((p) => samples[p].reward), cfg.grpoStdNormalization);
    positions.forEach((p, k) => { out[p] = scaled[k]; });
  }
  return out;
}

// slime/backends/megatron_utils/cp_utils.py::get_sum_of_sample_mean 的分母处理：clamp_min(denom, 1)
export const reducerDenominator = (denom) => Math.max(denom, 1);

// slime/utils/dp_schedule.py::build_dp_schedule 的前两步（按 rollout 分步 + 静态 micro-batch 切块 + 轮询分发）
export function stepSplit(rolloutIds, { globalBatchSize, dpSize, microBatchSize }) {
  const byRollout = new Map();
  rolloutIds.forEach((rid, pos) => { if (!byRollout.has(rid)) byRollout.set(rid, []); byRollout.get(rid).push(pos); });
  const ids = [...byRollout.keys()];
  const numSteps = Math.floor(ids.length / globalBatchSize);
  if (numSteps < 1) throw new Error('num_rollouts < global_batch_size');
  const steps = [];
  for (let s = 0; s < numSteps; s += 1) {
    const stepRollouts = ids.slice(s * globalBatchSize, (s + 1) * globalBatchSize);
    const sampleIndices = stepRollouts.flatMap((rid) => byRollout.get(rid));
    const mbs = [];
    for (let i = 0; i < sampleIndices.length; i += microBatchSize) mbs.push(sampleIndices.slice(i, i + microBatchSize));
    const alignTo = dpSize;
    if (mbs.length % alignTo !== 0) throw new Error('static path: num_mbs is not a multiple of dp_size * mb_group');
    const perRank = Array.from({ length: dpSize }, (_, r) => mbs.filter((_, k) => k % dpSize === r));
    steps.push({ rollouts: stepRollouts, sampleIndices, mbs, perRank, numMicrobatches: mbs.length / dpSize });
  }
  return { numSteps, steps };
}

// 运行 ①：partial 开、默认 generate（每个 sample task 返回 plain Sample，一轮内深度一致）
export function runPartial(cfg = CFG) {
  const ds = makeDataSource(cfg);
  const round3 = ds.getSamples(cfg.overSamplingBatchSize); // P0 P1 P2，样本 0–5
  const cursorAfterRound3 = ds.state.sampleOffset;
  const [p0] = round3;
  const [s0, s1] = p0;
  appendResponseTokens(s0, { tokens: [31, 32], logProbs: [-0.3, -0.4], finishReason: 'stop', weightVersion: 'v3', text: 'ok' });
  s0.reward = 1;
  appendResponseTokens(s1, { tokens: [21, 22], logProbs: [-0.1, -0.2], finishReason: 'abort', weightVersion: 'v3', text: 'A B' });
  const s1AfterAbort = clone(s1);
  const recycled = collectAbortedGroup(p0, cfg.abortRound);
  if (!recycled) throw new Error('P0 含 ABORTED 且已生成 token 的成员，应被回收');
  ds.addSamples([p0]);
  const bufferAfterRound3 = ds.state.buffer.length;

  const round4 = ds.getSamples(cfg.overSamplingBatchSize); // buffer 先给 P0，dataset 补 P3 P4（样本 6–9）
  const cursorAfterRound4 = ds.state.sampleOffset;
  const fromBuffer = round4[0] === p0;
  const budget = cfg.rolloutMaxResponseLen - s1.responseLength; // generate: max_new_tokens -= response_length

  const finish = (variant) => {
    const a0 = clone(s0); const a1 = clone(s1);
    const e0 = recycledEntry(a0, { partialRollout: true, maskOffpolicy: variant === 'masked' });
    const e1 = recycledEntry(a1, { partialRollout: true, maskOffpolicy: variant === 'masked' });
    if (!e0.skip) throw new Error('s0 已完成，应早退');
    if (e1.skip) throw new Error('s1 应续生成');
    // 实例设 token 24 为 EOS：用满剩余预算的请求以 stop 结束；若不是 EOS，SGLang 以 length 结束 → TRUNCATED
    appendResponseTokens(a1, { tokens: [23, 24], logProbs: [-0.5, -0.6], finishReason: 'stop', weightVersion: 'v4', text: ' C D' });
    a1.reward = 0;
    const [s6, s7] = round4[1].map(clone);
    appendResponseTokens(s6, { tokens: [41, 42, 43], logProbs: [-0.1, -0.1, -0.1], finishReason: 'stop', weightVersion: 'v4' });
    appendResponseTokens(s7, { tokens: [51, 52, 53], logProbs: [-0.2, -0.2, -0.2], finishReason: 'stop', weightVersion: 'v4' });
    s6.reward = 1; s7.reward = 0;
    const flat = [a0, a1, s6, s7];
    return { flat, converted: convert(flat), s0: a0, s1: a1 };
  };
  const defaultMode = finish('default');
  const maskedMode = finish('masked');
  const rawRewards = defaultMode.converted.rewards;
  return {
    round3: { groups: round3.map((g) => g.map((s) => s.index)), cursorAfter: cursorAfterRound3, bufferAfter: bufferAfterRound3 },
    round4: { groups: round4.map((g) => g.map((s) => s.index)), cursorAfter: cursorAfterRound4, fromBuffer, budget },
    s1AfterAbort,
    s0StartRolloutId: p0[0].metadata.start_rollout_id,
    defaultMode,
    maskedMode,
    zeroDenominatorSamples: maskedMode.converted.rolloutMaskSums.map((d, i) => (d === 0 ? i : -1)).filter((i) => i >= 0),
    clampedDenominator: reducerDenominator(0),
    rawRewards,
    normalized: postProcessRewards(rawRewards, cfg),
  };
}

// 运行 ②：partial 关、自定义生成函数对每条样本返回 list[Sample]（s2 扇出为两个片段，其余单元素列表）
export function runFanout(cfg = CFG) {
  const ds = makeDataSource(cfg);
  const [p0, p1] = ds.getSamples(cfg.overSamplingBatchSize); // 提交 P0 P1 P2；不设过滤器时先完成的 P0 P1 入选，P2 在 abort 时放弃（partial 关）
  const [s0, s1] = p0; const [s2, s3] = p1;
  appendResponseTokens(s0, { tokens: [31, 32], logProbs: [-0.3, -0.4], finishReason: 'stop', weightVersion: 'v5' });
  appendResponseTokens(s1, { tokens: [21, 22, 23, 24], logProbs: [-0.1, -0.2, -0.5, -0.6], finishReason: 'stop', weightVersion: 'v5' });
  const s2a = { ...clone(s2), rolloutId: s2.index };
  const s2b = { ...clone(s2), rolloutId: s2.index };
  appendResponseTokens(s2a, { tokens: [61], logProbs: [-0.2], finishReason: 'stop', weightVersion: 'v5' });
  appendResponseTokens(s2b, { tokens: [62, 63, 64], logProbs: [-0.2, -0.2, -0.2], finishReason: 'stop', weightVersion: 'v5' });
  appendResponseTokens(s3, { tokens: [41, 42, 43], logProbs: [-0.1, -0.1, -0.1], finishReason: 'stop', weightVersion: 'v5' });
  s0.reward = 1; s1.reward = 0; s2a.reward = 0.5; s2b.reward = 0.5; s3.reward = 1; // 扇出 reward 按 reward / K 分配
  // 嵌套输出：prompt × rollout × 片段（每个 sample task 返回 list）
  const nested = [[[s0], [s1]], [[s2a, s2b], [s3]]];
  // validate_rollout_id_annotated：深度 ≥ 2 且多于一条的叶子必须共享非空 rollout_id
  for (const group of nested) for (const leaf of group) if (leaf.length > 1) {
    const rids = leaf.map((s) => s.rolloutId);
    if (rids.some((r) => r === null) || new Set(rids).size !== 1) throw new Error('Sibling samples from one compact rollout must share rollout_id');
  }
  let data = nested;
  while (Array.isArray(data[0])) data = data.flat(); // while isinstance(data[0], list)
  const converted = convert(data);
  const globalBatchSize = (cfg.rolloutBatchSize * cfg.nSamplesPerPrompt) / cfg.numStepsPerRollout;
  const schedule = stepSplit(converted.rolloutIds, { globalBatchSize, dpSize: cfg.dpSize, microBatchSize: cfg.microBatchSize });
  const fragmentPositions = converted.rolloutIds.map((rid, i) => (rid === s2.index ? i : -1)).filter((i) => i >= 0);
  const fragmentMbs = fragmentPositions.map((pos) => schedule.steps[0].mbs.findIndex((mb) => mb.includes(pos)));
  const denominator = converted.rolloutMaskSums[fragmentPositions[0]];
  const numerators = [10, 6];
  return {
    names: ['s0', 's1', 's2a', 's2b', 's3'],
    flat: data,
    converted,
    globalBatchSize,
    schedule,
    fragmentPositions,
    fragmentMbs,
    denominator,
    numerators,
    sharedDenominatorTotal: (numerators[0] + numerators[1]) / denominator,
    localDenominatorTotal: numerators[0] / converted.maskSums[fragmentPositions[0]] + numerators[1] / converted.maskSums[fragmentPositions[1]],
    rawRewards: converted.rewards,
    expectedRewardCount: cfg.nSamplesPerPrompt * cfg.rolloutBatchSize,
    normalized: postProcessRewards(converted.rewards, cfg),
    byGroupIndex: normalizeByGroupIndex(data, cfg),
  };
}

export function model(cfg = CFG) {
  return { cfg, partial: runPartial(cfg), fanout: runFanout(cfg) };
}

// 三位小数的展示格式（避免 -0.000）；页面与测试共用
export const fmt3 = (v) => (Math.abs(v) < 5e-4 ? 0 : v).toFixed(3);
export const fmtArr = (arr) => `[${arr.join(',')}]`;
export const fmtArr3 = (arr) => `[${arr.map(fmt3).join(',')}]`;

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
  .mono{font-family:"Cascadia Mono",Consolas,"Courier New",monospace;font-size:11px;fill:#38414D}
  .panel{fill:#FBFCFE;stroke:#D9DEE7;stroke-width:1.2}
  .neutral{fill:#fff;stroke:#AEB6C2;stroke-width:1.2}
  .ghost{fill:#F5F7FA;stroke:#D9DEE7;stroke-width:1.1}
  .acc1{fill:#EAF1FD;stroke:#2563EB;stroke-width:1.5}
  .acc2{fill:#FCF1E6;stroke:#C3651F;stroke-width:1.5}
  .cell{fill:#fff;stroke:#AEB6C2;stroke-width:1.1}
  .main{fill:none;stroke:#2563EB;stroke-width:2;marker-end:url(#arrowMain)}
  .aux{fill:none;stroke:#AEB6C2;stroke-width:1.3;stroke-dasharray:5 4;marker-end:url(#arrowAux)}
`;

function render(m) {
  const W = 1180;
  const o = [];
  const rect = (x, y, w, h, cls = 'neutral', r = 7) => o.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" class="${cls}"/>`);
  const text = (x, y, s, cls = 'tx', anchor = 'start') => o.push(`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${esc(s)}</text>`);
  const arrow = (x1, y1, x2, y2, cls = 'main') => o.push(`<path d="M${x1} ${y1} L${x2} ${y2}" class="${cls}"/>`);
  const sum = (arr) => arr.reduce((a, b) => a + b, 0);

  const c = m.cfg; const P = m.partial; const F = m.fanout;
  const laneW = 1132; const right = 24 + laneW;
  const title = () => {
    text(24, 34, '两条运行、三次决定性转换：Sample 追加、DataSource 回收、converter 分母', 'ti');
    text(24, 56, `rollout_batch_size=${c.rolloutBatchSize} · over_sampling_batch_size=${c.overSamplingBatchSize} · n_samples_per_prompt=${c.nSamplesPerPrompt} · rollout_max_response_len=${c.rolloutMaxResponseLen} · global_batch_size=${F.globalBatchSize} · dp_size=${c.dpSize} · micro_batch_size=${c.microBatchSize} · 归一化默认关`, 'su');
  };
  title();

  // ---- 泳道 ①-A：Sample 层 ----
  const y1 = 76;
  rect(24, y1, laneW, 256, 'panel', 10);
  text(40, y1 + 24, '运行 ①（partial 开，默认 generate）· Sample 层：s1 的 append_response_tokens 时间线（唯一写入口，按新增区间追加）', 'pt');
  const cellW = 54; const cellH = 26; const tx0 = 170;
  const rowLabel = (y, label) => text(tx0 - 10, y + 17, label, 'sm', 'end');
  const tokenRow = (y, tokens, classes) => tokens.forEach((t, i) => {
    rect(tx0 + i * (cellW + 4), y, cellW, cellH, classes[i], 4);
    text(tx0 + i * (cellW + 4) + cellW / 2, y + 17, t, 'mono', 'middle');
  });
  const s1 = P.defaultMode.s1; const s1a = P.s1AfterAbort; const s1m = P.maskedMode.s1;
  const promptLen = c.promptIds.length;
  const cls = s1.tokens.map((_, i) => (i < promptLen ? 'ghost' : i < promptLen + s1a.responseLength ? 'acc2' : 'acc1'));
  let ry = y1 + 44;
  rowLabel(ry, 'tokens'); tokenRow(ry, s1.tokens, cls);
  ry += cellH + 6;
  rowLabel(ry, 'rollout_log_probs'); tokenRow(ry, [...c.promptIds.map(() => '—'), ...s1.rolloutLogProbs.map((v) => v.toFixed(1))], cls.map((k) => (k === 'ghost' ? 'ghost' : 'cell')));
  ry += cellH + 6;
  rowLabel(ry, 'loss_mask 默认'); tokenRow(ry, [...c.promptIds.map(() => '—'), ...s1.lossMask], cls.map((k) => (k === 'ghost' ? 'ghost' : 'cell')));
  ry += cellH + 6;
  rowLabel(ry, 'mask-offpolicy 开'); tokenRow(ry, [...c.promptIds.map(() => '—'), ...s1m.lossMask], cls.map((k) => (k === 'ghost' ? 'ghost' : k === 'acc2' ? 'acc2' : 'cell')));
  ry += cellH + 6;
  rowLabel(ry, '来源'); tokenRow(ry, [...c.promptIds.map(() => 'prompt'), ...s1a.tokens.slice(promptLen).map(() => `round ${c.abortRound}`), ...s1.tokens.slice(promptLen + s1a.responseLength).map(() => `round ${c.resumeRound}`)], cls.map((k) => (k === 'ghost' ? 'ghost' : 'cell')));
  const rx = 560;
  rect(rx, y1 + 44, right - rx, 204, 'ghost', 8);
  const noteA = [
    [`round ${c.abortRound}：返回 ${s1a.responseLength} 个 token，finish_reason=abort → ABORTED，weight_versions=${fmtArr(s1a.weightVersions)}`, 'tx'],
    ['P0 有 ABORTED 且已生成 token 的成员 → abort() 整组回收，', 'tx'],
    [`并给有 response 的 s0、s1 写 metadata.start_rollout_id=${P.s0StartRolloutId}`, 'tx'],
    [`round ${c.resumeRound}：复用 sample.tokens；max_new_tokens = ${c.rolloutMaxResponseLen} − ${s1a.responseLength} = ${P.round4.budget}`, 'tx'],
    ['SGLang 只返回新 token；append 把 mask、logprob 与 token 等长追加', 'tx'],
    [`24 为 EOS → stop → COMPLETED，weight_versions=${fmtArr(s1.weightVersions)}；旧 token 不被覆盖`, 'tx'],
    [`默认：旧区间 mask 保持 1，新旧 token 都训练（s1 mask 和 ${sum(s1.lossMask)}）`, 'tx'],
    [`开 mask-offpolicy：入口把旧区间 mask 清 0（s1 mask 和 ${sum(s1m.lossMask)}）`, 'tx'],
  ];
  noteA.forEach(([s, k], i) => text(rx + 14, y1 + 62 + i * 19, s, k));
  text(rx + 14, y1 + 62 + noteA.length * 19 + 3, '两种模式都保留旧 token 与旧 logprob；改写的只有旧区间的 loss 权重。', 'cap');
  text(rx + 14, y1 + 62 + noteA.length * 19 + 21, '工具/环境 token 走 trainable=False：mask 0，logprob 填 0 只为等长。', 'cap');

  // ---- 泳道 ①-B：DataSource 层 ----
  const y2 = y1 + 268;
  rect(24, y2, laneW, 190, 'panel', 10);
  text(40, y2 + 24, '运行 ① · DataSource 层：游标顺序取 prompt group，中断组整组 FIFO 回收（RolloutDataSourceWithBuffer）', 'pt');
  const gx = 40; const gw = 118; const gh = 58;
  const groupBox = (x, y, label, idx, k) => {
    rect(x, y, gw, gh, k, 6);
    text(x + gw / 2, y + 20, label, 'pt', 'middle');
    text(x + gw / 2, y + 38, `样本 ${idx[0]}–${idx[idx.length - 1]}`, 'sm', 'middle');
    text(x + gw / 2, y + 52, `group_index ${label.slice(1)}`, 'sm', 'middle');
  };
  text(gx, y2 + 48, `round ${c.abortRound}：get_samples(${c.overSamplingBatchSize}) 全部来自 dataset，游标 0 → ${P.round3.cursorAfter}`, 'tx');
  P.round3.groups.forEach((idx, i) => groupBox(gx + i * (gw + 10), y2 + 58, `P${i}`, idx, i === 0 ? 'acc2' : 'cell'));
  text(gx, y2 + 138, 'P1、P2 完成入选；P0 未完成 → abort → add_samples([P0])', 'tx');
  text(gx, y2 + 156, `断言每组长度 = n_samples_per_prompt = ${c.nSamplesPerPrompt} → buffer 长度 ${P.round3.bufferAfter}`, 'tx');
  text(gx, y2 + 176, '不带 buffer 的 RolloutDataSource.add_samples 直接抛 RuntimeError', 'cap');
  const bx = 470;
  arrow(gx + 3 * gw + 20, y2 + 87, bx - 6, y2 + 87, 'aux');
  rect(bx, y2 + 58, 150, gh, 'acc2', 6);
  text(bx + 75, y2 + 80, 'buffer', 'pt', 'middle');
  text(bx + 75, y2 + 100, `[P0] · start_rollout_id=${c.abortRound}`, 'sm', 'middle');
  const gx2 = 660;
  text(gx2, y2 + 48, `round ${c.resumeRound}：get_samples(${c.overSamplingBatchSize})：buffer pop_first ${P.round3.bufferAfter} 组 + dataset ${c.overSamplingBatchSize - P.round3.bufferAfter} 组，游标 ${P.round3.cursorAfter} → ${P.round4.cursorAfter}`, 'tx');
  arrow(bx + 150, y2 + 87, gx2 - 6, y2 + 87);
  P.round4.groups.forEach((idx, i) => groupBox(gx2 + i * (gw + 10), y2 + 58, i === 0 ? 'P0' : `P${i + 2}`, idx, i === 0 ? 'acc2' : i === 1 ? 'acc1' : 'cell'));
  text(gx2, y2 + 138, 'P0 与 P3 入选（按 group[0].index 排序）；P4 若在途且已吐出 token，再被回收', 'tx');
  text(gx2, y2 + 156, 's0 已 COMPLETED 且有 reward → generate_and_rm 早退，只有 s1 续生成', 'tx');
  text(gx2, y2 + 176, 'save/load 只存游标、epoch、两个计数器与 metadata；buffer 不进 checkpoint', 'cap');

  // 表格行：label 右对齐在 labelW 内，值格从 x + labelW 起
  const table = (x, y, rows, { labelW, colW, rowH, cellClass }) => {
    rows.forEach(([label, values], r) => {
      const yy = y + r * rowH;
      text(x + labelW - 8, yy + 15, label, r === 0 ? 'sm' : 'tx', 'end');
      values.forEach((v, i) => {
        const xx = x + labelW + i * (colW + 4);
        rect(xx, yy, colW, rowH - 3, cellClass(r, i), 3);
        text(xx + colW / 2, yy + 14, v, 'mono', 'middle');
      });
    });
    return x + labelW + rows[0][1].length * (colW + 4);
  };

  // ---- 泳道 ①-C：运行 ① 的 converter ----
  const y3 = y2 + 202;
  rect(24, y3, laneW, 226, 'panel', 10);
  text(40, y3 + 24, `运行 ① · converter：round ${c.resumeRound} 接收 P0 + P3，${P.defaultMode.flat.length} 条样本，无自定义 rollout_id → 兜底为 ${fmtArr(P.defaultMode.converted.rolloutIds)}`, 'pt');
  const rows1 = [
    ['样本', ['s0', 's1', 's6', 's7']],
    ['loss_mask 和（默认）', P.defaultMode.converted.maskSums],
    ['rollout_mask_sums（默认）', P.defaultMode.converted.rolloutMaskSums],
    ['loss_mask 和（offpolicy 开）', P.maskedMode.converted.maskSums],
    ['rollout_mask_sums（offpolicy 开）', P.maskedMode.converted.rolloutMaskSums],
    ['raw_reward', P.rawRewards],
    [`rewards（归一化开：${P.rawRewards.length} = ${c.nSamplesPerPrompt}×${c.rolloutBatchSize}）`, P.normalized.rewards.map(fmt3)],
  ];
  const t1y = y3 + 40; const rowH1 = 22;
  const t1r = table(40, t1y, rows1, {
    labelW: 250, colW: 88, rowH: rowH1,
    cellClass: (r, i) => ((r === 3 || r === 4) && P.zeroDenominatorSamples.includes(i) ? 'acc2' : 'cell'),
  });
  const n1x = t1r + 12;
  rect(n1x, t1y, right - n1x, rows1.length * rowH1 - 3, 'ghost', 8);
  const noteC = [
    [18, 'offpolicy 清零在 generate_and_rm 入口、早退之前：'],
    [36, `已完成的 s0（response_length=${P.maskedMode.s0.responseLength}）也被清 0，分母 ${P.maskedMode.converted.rolloutMaskSums[0]}；`],
    [54, `训练侧 clamp_min(denom, 1) → 分母取 ${P.clampedDenominator}，s0 贡献 0，`],
    [72, '不报错，但白占一份 rollout 名额。'],
    [98, `归一化开：reward 数 ${P.rawRewards.length} = ${c.nSamplesPerPrompt}×${c.rolloutBatchSize} → 每 ${P.normalized.groupSize} 条 reshape 成一组，`],
    [116, `共 ${P.normalized.numGroups} 组，恰是 P0、P3，各自居中后除以组内 std。`],
  ];
  noteC.forEach(([dy, s]) => text(n1x + 12, t1y + dy, s, 'tx'));
  text(40, t1y + rows1.length * rowH1 + 20, `运行 ① 里 P3 的 s6、s7 是普通单元素 rollout；rollout_id 全为 None 时兜底 id 就是 ${P.defaultMode.converted.rolloutIds.join('、')} 的顺序编号。`, 'cap');

  // ---- 泳道 ②：运行 ② 的 converter 与切分 ----
  const y4 = y3 + 238;
  const ty = y4 + 40; const rowH = 22;
  const rows2 = [
    ['样本', F.names],
    ['sample_indices', F.converted.sampleIndices],
    ['Sample.rollout_id', F.flat.map((s) => (s.rolloutId === null ? 'None' : s.rolloutId))],
    ['rollout_ids 兜底后', F.converted.rolloutIds],
    ['response_lengths', F.converted.responseLengths],
    ['loss_mask 和', F.converted.maskSums],
    ['rollout_mask_sums', F.converted.rolloutMaskSums],
    ['raw_reward（reward / K 拆分）', F.rawRewards],
    [`rewards 归一化开（${F.rawRewards.length} ≠ ${F.expectedRewardCount} → 一组）`, F.normalized.rewards.map(fmt3)],
    ['按 group_index 分组（helper）', F.byGroupIndex.map(fmt3)],
  ];
  const sy = ty + rows2.length * rowH + 18;
  const panel2H = sy + 156 - y4;
  rect(24, y4, laneW, panel2H, 'panel', 10);
  text(40, y4 + 24, `运行 ②（partial 关，自定义 generate 返回 list[Sample]）· converter 与切分：入选 P0 + P1 展平为 ${F.flat.length} 条，s2 扇出为两片段，共享 rollout_id=${F.flat[2].rolloutId}`, 'pt');
  const t2r = table(40, ty, rows2, {
    labelW: 250, colW: 88, rowH,
    cellClass: (r, i) => {
      if (r === 3 && F.flat[i].rolloutId === null) return 'acc1';
      if (r === 6 && F.fragmentPositions.includes(i)) return 'acc1';
      if (r === 8) return 'acc2';
      if (r === 9) return 'acc1';
      return 'cell';
    },
  });
  const noteX = t2r + 12;
  rect(noteX, ty, right - noteX, rows2.length * rowH - 3, 'ghost', 8);
  const noteD = [
    [18, `rollout_id 兜底：已存在 {${F.flat[2].rolloutId}}，`],
    [36, `None 依次取不冲突的 ${F.converted.rolloutIds.filter((_, i) => F.flat[i].rolloutId === null).join('、')}（跳过 ${F.flat[2].rolloutId}）。`],
    [60, '分母在切分前算：同一 rollout_id'],
    [78, `的 mask 和相加（${F.converted.maskSums[F.fragmentPositions[0]]}+${F.converted.maskSums[F.fragmentPositions[1]]}=${F.denominator}），再广播回每个片段。`],
    [102, '嵌套输出 prompt × rollout × 片段：'],
    [120, '深度 ≥ 2 且多于一条的叶子才校验 rollout_id；'],
    [138, '单元素列表跳过校验，展平两次得到平面列表。'],
    [162, `归一化开：reward 数 ${F.rawRewards.length} ≠ ${c.nSamplesPerPrompt}×${c.rolloutBatchSize}，`],
    [180, '退化为整批一组，跨 prompt 居中；'],
    [198, 'helper 按 group_index 分组才按 prompt 居中。'],
  ];
  noteD.forEach(([dy, s]) => text(noteX + 12, ty + dy, s, 'tx'));
  const st = F.schedule.steps[0];
  text(40, sy + 14, `build_dp_schedule：${st.rollouts.length} 个逻辑 rollout = global_batch_size → ${F.schedule.numSteps} 个训练 step；静态 micro_batch_size=${c.microBatchSize} → ${st.mbs.length} 个 micro-batch，dp_size=${c.dpSize} 全归 rank 0`, 'tx');
  const mbW = 140; const mbH = 60; const mx0 = 40;
  st.mbs.forEach((mb, k) => {
    const x = mx0 + k * (mbW + 10);
    const isFrag = F.fragmentMbs.includes(k);
    rect(x, sy + 24, mbW, mbH, isFrag ? 'acc1' : 'cell', 6);
    text(x + mbW / 2, sy + 44, `micro-batch ${k}`, 'pt', 'middle');
    const pos = mb[0];
    text(x + mbW / 2, sy + 62, `${F.names[pos]} · rollout ${F.converted.rolloutIds[pos]}`, 'sm', 'middle');
    text(x + mbW / 2, sy + 77, `分子 / ${F.converted.rolloutMaskSums[pos]}`, 'sm', 'middle');
  });
  const ex = mx0 + st.mbs.length * (mbW + 10);
  rect(ex, sy + 24, right - ex, mbH, 'acc2', 6);
  text(ex + 12, sy + 44, `s2a、s2b 落在 micro-batch ${F.fragmentMbs.join(' 与 ')}，`, 'tx');
  text(ex + 12, sy + 62, `分子 ${F.numerators[0]}、${F.numerators[1]}：共用分母 → ${F.numerators[0]}/${F.denominator} + ${F.numerators[1]}/${F.denominator} = ${F.sharedDenominatorTotal}；`, 'tx');
  text(ex + 12, sy + 78, `各用局部长度 → ${F.numerators[0]}/${F.converted.maskSums[F.fragmentPositions[0]]} + ${F.numerators[1]}/${F.converted.maskSums[F.fragmentPositions[1]]} = ${F.localDenominatorTotal}，多投一票`, 'tx');
  text(40, sy + 108, '切分只产出 partition 与 micro_batch_indices；per-sample 字段按 partition 取子集并 tensorize 到 CPU，raw_reward 与 total_lengths 整批发给每个 rank。', 'cap');
  text(40, sy + 126, '两条运行分开：开 partial 时 abort 对在途组每个成员取 .status，返回 list 的自定义 generate 会抛 AttributeError；回收组里已完成成员早退返回 plain Sample，', 'cap');
  text(40, sy + 144, '与 list 混用时展平停在第一层，逐样本属性访问同样抛错。橙：中断、回收、被清零与退化分组；蓝：决定性映射（兜底 id、分母广播、片段所在 mb、按组居中）。', 'cap');

  const footerY = y4 + panel2H + 22;
  text(24, footerY, '源码基线：THUDM/slime@4c193f1f3750 · 复现 append_response_tokens / generate_and_rm / abort / DataSourceWithBuffer / _post_process_rewards / converter / build_dp_schedule', 'su');
  const H = footerY + 14;

  const head = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="title desc">`,
    '<title id="title">slime 数据契约：中断续生成与 compact 扇出如何穿过 Sample、DataSource 与 converter</title>',
    '<desc id="desc">四条泳道分别展示运行①的 Sample 对齐追加、DataSource 游标与回收 buffer、两种 mask 模式与 reward 归一化开启时的 converter 结果，以及运行②的 rollout_id 兜底、rollout_mask_sums 分母、归一化退化为整批一组与按 rollout 分步后的 micro-batch 归属。</desc>',
    '<defs><marker id="arrowMain" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#2563EB"/></marker><marker id="arrowAux" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#AEB6C2"/></marker></defs>',
    `<style>${STYLE}</style><rect width="${W}" height="${H}" fill="white"/>`,
  ];
  return [...head, ...o, '</svg>'].join('\n');
}

const here = dirname(fileURLToPath(import.meta.url));
const defaultOutput = join(here, '..', '..', '..', 'wiki', '02_engineering', '04_posttrain_frameworks', 'slime', 'assets');
const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  const outputDir = process.argv[2] ? process.argv[2] : defaultOutput;
  mkdirSync(outputDir, { recursive: true });
  writeFileSync(join(outputDir, 'slime_sample_data_contract.svg'), `${render(model())}\n`, 'utf8');
}
