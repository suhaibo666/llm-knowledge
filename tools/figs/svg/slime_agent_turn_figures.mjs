// 图：slime agent 轨迹的两层判断，用同一个两轮最小例子重放。
//  图 ①（消息层，TrajectoryManager.record_turn）：回放的 assistant 消息与记录的叶节点字典相等时下探；
//    不相等时 `_try_merge_assistant_rewrite` 在"唯一 assistant 子节点、是叶、已生成、输出短于阈值"时把旧叶降级为仅路由，
//    否则在消息层分叉。三条 lane 的树形与导出的 Sample 都由下方的算法复现计算。
//  图 ②（token 层，TrajectoryManager.get_trajectory → _SampleBuilder）：消息相等、prompt_ids 却在 A 内漂移时，
//    `classify_token_drift` 按公共前缀、最近 response 起点与新 output 长度在 CLEAN / REALIGN / FORK 之间选择；
//    画出每种结果的 token 行与 loss_mask，标出 REALIGN 被清零的 A 与 FORK 多出的上下文行。
// 源码基线：THUDM/slime@4c193f1f37509cca70f0e88807a9305b70f63f4e（slime/agent/trajectory.py 自 681b3adc 起未改）。
// token 命名沿用 tests/test_agent/test_trajectory_manager_branching.py 的语义 token（<gen> 与 assistant 起始 token 同 id）。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
// 要讲清楚：
//  1. 消息层只看 role + 字典相等决定树形；token 层只在 linearize 时看 token 前缀，两层互不替代。
//  2. 改写合并与 REALIGN 都会让 A 失去训练信号，但发生在不同层、由不同条件触发；阈值都按"输出长度 < fork_threshold"比较，
//     合并看被放弃的旧叶输出，REALIGN 看新一轮输出。
//  3. FORK 保住 A 的训练信号，代价是多一条 Sample，且新 Sample 把整段 prompt 作为不计 loss 的上下文重算。
// 布局：图 ① 三条横向 lane：条件 → 树 → 导出 Sample；图 ② token 网格：held、turn 2 prompt_ids、决策行，
//  再是 CLEAN（对照）、REALIGN、FORK 两行。acc1 = 训练 token / 生成节点，acc2 = 漂移 token、被清零或降级的 A。
//
// 用法：node tools/figs/svg/slime_agent_turn_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// ---------------- 冻结的示例输入 ----------------
export const CFG = Object.freeze({
  defaultForkThreshold: 1024, // TrajectoryManager.__init__：fork_threshold_tokens 为 None 时取 1024
  smallThreshold: 1, // 测试 2.5 / 3.2 用的阈值：2 个输出 token ≥ 1
  firstLabel: 'call', // turn 1 生成的 A
  rewriteLabel: 'call ', // 客户端回放的 A′（多一个尾随空格）
  secondLabel: 'done', // turn 2 生成的 C
  toolLabel: 't',
  lp1: -0.5,
  lp2: -0.4,
});

// ---------------- 语义 token 渲染（镜像 test_trajectory_manager_branching.py 的 MsgTok）----------------
const vis = (s) => String(s).replace(/ /g, '␣');
const START = { system: '<sys>', user: '<usr>', assistant: '<gen>', tool: '<tul>' };
const END = { system: '</sys>', user: '</usr>', assistant: '</ast>', tool: '</tul>' };
const BODY = { system: 'system:', user: 'user:', tool: 'tool:' };

export const msg = (role, label) => ({ role, content: label });
export function renderMessage(m) {
  const body = m.role === 'assistant' ? `r:${vis(m.content)}` : `${BODY[m.role]}${vis(m.content)}`;
  return [START[m.role], body, END[m.role]];
}
export const renderPrompt = (msgs) => [...msgs.flatMap(renderMessage), '<gen>'];
export const renderResponse = (label) => [`r:${vis(label)}`, '</ast>'];
const msgEq = (a, b) => a !== null && b !== null && a.role === b.role && a.content === b.content;

// ---------------- 对 slime/agent/trajectory.py 的最小复现 ----------------
class MessageNode {
  constructor({ role = null, message = null, parent = null } = {}) {
    this.role = role;
    this.message = message;
    this.parent = parent;
    this.children = [];
    this.turn = null; // generated TurnRecord，否则仅路由
    this.turnIndex = null;
    this.responseTrained = false;
    this.metadata = {};
  }
  get isRoot() {
    return this.parent === null;
  }
  addChild(child) {
    child.parent = this;
    this.children.push(child);
    return child;
  }
  pathFromRoot() {
    const chain = [];
    for (let n = this; n && !n.isRoot; n = n.parent) chain.push(n);
    return chain.reverse();
  }
  *leaves() {
    if (this.children.length === 0) {
      yield this;
      return;
    }
    for (const c of this.children) yield* c.leaves();
  }
}

export function commonPrefixLen(a, b) {
  const limit = Math.min(a.length, b.length);
  let i = 0;
  while (i < limit && a[i] === b[i]) i += 1;
  return i;
}

// _SampleBuilder
class SampleBuilder {
  constructor(forkThreshold) {
    this.forkThreshold = forkThreshold;
    this.tokens = [];
    this.lossMask = [];
    this.logprobs = [];
    this.lastResponseStartIdx = null;
    this.leadingPromptLen = 0;
    this.lastDecision = null;
  }
  classifyTokenDrift(turn) {
    const realignAt = commonPrefixLen(this.tokens, turn.promptIds);
    const drift = this.tokens.length - realignAt;
    let kind = 'FORK';
    if (drift === 0) kind = 'CLEAN';
    else if (this.lastResponseStartIdx !== null && realignAt >= this.lastResponseStartIdx && turn.outputIds.length < this.forkThreshold) kind = 'REALIGN';
    this.lastDecision = { held: this.tokens.length, realignAt, drift, start: this.lastResponseStartIdx, outputLen: turn.outputIds.length, threshold: this.forkThreshold, kind };
    return kind;
  }
  appendTurn(turn, kind, trained = true) {
    const isFirst = this.lastResponseStartIdx === null;
    if (kind === 'REALIGN') {
      const s = this.lastResponseStartIdx;
      const tail = turn.promptIds.slice(s);
      this.tokens.splice(s, this.tokens.length - s, ...tail);
      this.lossMask.splice(s, this.lossMask.length - s, ...tail.map(() => 0));
      this.logprobs.splice(s, this.logprobs.length - s, ...tail.map(() => 0));
    } else {
      const tail = turn.promptIds.slice(this.tokens.length);
      this.#append(tail, 0, null);
    }
    this.lastResponseStartIdx = this.tokens.length;
    this.#append(turn.outputIds, trained ? 1 : 0, trained ? turn.logprobs : null);
    if (isFirst) this.leadingPromptLen = turn.promptIds.length;
  }
  #append(ids, mask, lps) {
    this.tokens.push(...ids);
    this.lossMask.push(...ids.map(() => mask));
    this.logprobs.push(...(lps && lps.length ? lps : ids.map(() => 0)));
  }
  hasTrainedResponse() {
    return this.lossMask.slice(this.leadingPromptLen).some((m) => m);
  }
  toSample(maxSampleTokens = 0) {
    const start = this.leadingPromptLen;
    let { tokens, lossMask, logprobs } = this;
    if (maxSampleTokens && tokens.length > maxSampleTokens) {
      tokens = tokens.slice(0, maxSampleTokens);
      lossMask = lossMask.slice(0, maxSampleTokens);
      logprobs = logprobs.slice(0, maxSampleTokens);
    }
    return { tokens: [...tokens], responseLength: lossMask.length - start, lossMask: lossMask.slice(start), rolloutLogProbs: logprobs.slice(start), status: 'COMPLETED' };
  }
}

export class TrajectoryManager {
  constructor({ forkThresholdTokens = null } = {}) {
    this.forkThreshold = forkThresholdTokens === null ? CFG.defaultForkThreshold : forkThresholdTokens;
    this.trees = new Map();
    this.turnCount = new Map();
    this.decisions = [];
  }
  recordTurn(sid, { turn, promptMessages, responseMessage }) {
    if (!promptMessages.length) return;
    if (!this.trees.has(sid)) this.trees.set(sid, new MessageNode());
    const root = this.trees.get(sid);
    let [node, depth] = this.#findMountPoint(root, promptMessages);
    [node, depth] = this.#tryMergeAssistantRewrite(sid, node, promptMessages, depth);
    for (const m of promptMessages.slice(depth)) node = node.addChild(new MessageNode({ role: m.role, message: m }));
    const asst = new MessageNode({ role: 'assistant', message: responseMessage });
    asst.turn = turn;
    asst.turnIndex = (this.turnCount.get(sid) || 0) + 1;
    node.addChild(asst);
    this.turnCount.set(sid, asst.turnIndex);
  }
  #findMountPoint(root, messages) {
    let node = root;
    let depth = 0;
    while (depth < messages.length) {
      const next = node.children.find((c) => c.role === messages[depth].role && msgEq(c.message, messages[depth]));
      if (!next) break;
      node = next;
      depth += 1;
    }
    return [node, depth];
  }
  #tryMergeAssistantRewrite(sid, node, prompt, depth) {
    if (this.forkThreshold <= 0) return [node, depth];
    if (depth >= prompt.length || prompt[depth].role !== 'assistant') return [node, depth];
    const asst = node.children.filter((c) => c.role === 'assistant');
    if (asst.length !== 1) return [node, depth];
    const rw = asst[0];
    if (rw.children.length || rw.turn === null || rw.turn.outputIds.length >= this.forkThreshold) return [node, depth];
    rw.metadata.merged_rewrite = { abandoned_turn_index: rw.turnIndex, abandoned_response_tokens: rw.turn.outputIds.length };
    rw.turn = null;
    rw.turnIndex = null;
    rw.message = prompt[depth];
    return [rw, depth + 1];
  }
  #splitChainIntoBuilders(chain) {
    const builders = [];
    for (const n of chain.filter((x) => x.role === 'assistant' && x.turn !== null)) {
      const trained = !n.responseTrained;
      n.responseTrained = true;
      let kind = null;
      if (builders.length) {
        kind = builders.at(-1).classifyTokenDrift(n.turn);
        this.decisions.push(builders.at(-1).lastDecision);
      }
      if (!builders.length || kind === 'FORK') {
        builders.push(new SampleBuilder(this.forkThreshold));
        builders.at(-1).appendTurn(n.turn, 'CLEAN', trained);
      } else {
        builders.at(-1).appendTurn(n.turn, kind, trained);
      }
    }
    return builders;
  }
  getTrajectory(sid, { reward = 0, maxSampleTokens = 0 } = {}) {
    const root = this.trees.get(sid);
    if (!root) return [];
    const samples = [];
    for (const leaf of root.leaves()) {
      if (leaf.isRoot) continue;
      for (const b of this.#splitChainIntoBuilders(leaf.pathFromRoot())) if (b.hasTrainedResponse()) samples.push(b.toSample(maxSampleTokens));
    }
    for (const s of samples) s.reward = reward;
    this.trees.delete(sid);
    this.turnCount.delete(sid);
    return samples;
  }
}

// 把 Sample 渲染成 test_trajectory_manager_branching.py::golden 的字符串：训练 token 用 [..] 包住
export function golden(sample) {
  const respStart = sample.tokens.length - sample.responseLength;
  return sample.tokens.map((t, i) => (i >= respStart && sample.lossMask[i - respStart] === 1 ? `[${t}]` : t)).join(' ');
}

// 描述树：每个非根节点 → {label, kind}
function describeTree(root) {
  const lines = [];
  const walk = (n, depth) => {
    for (const c of n.children) {
      const label = c.role === 'assistant' ? `r:${vis(c.message.content)}` : `${BODY[c.role]}${vis(c.message.content)}`;
      const kind = c.role !== 'assistant' ? 'route' : c.turn !== null ? 'generated' : c.metadata.merged_rewrite ? 'demoted' : 'route';
      lines.push({ depth, label, kind, merged: c.metadata.merged_rewrite || null });
      walk(c, depth + 1);
    }
  };
  walk(root, 0);
  return lines;
}

// 一次两轮会话：turn 1 = [S, u] → A；turn 2 = [S, u, A 回放, t] → C
export function runSession({ threshold = null, firstLabel = CFG.firstLabel, echoLabel = CFG.firstLabel, secondLabel = CFG.secondLabel, toolLabel = CFG.toolLabel, promptIds2 = null, drift = null } = {}) {
  const mgr = new TrajectoryManager({ forkThresholdTokens: threshold });
  const S = msg('system', 'S');
  const U = msg('user', 'u');
  const T = msg('tool', toolLabel);
  const p1 = renderPrompt([S, U]);
  const r1 = renderResponse(firstLabel);
  mgr.recordTurn('sid', { turn: { promptIds: p1, outputIds: r1, logprobs: r1.map(() => CFG.lp1) }, promptMessages: [S, U], responseMessage: msg('assistant', firstLabel) });
  const echo = msg('assistant', echoLabel);
  let p2 = promptIds2 ?? renderPrompt([S, U, echo, T]);
  if (drift) {
    p2 = [...p2];
    if (drift.mode === 'replace') p2[drift.at] = '<DRIFT>';
    else p2.splice(drift.at, 0, '<DRIFT>');
  }
  const r2 = renderResponse(secondLabel);
  mgr.recordTurn('sid', { turn: { promptIds: p2, outputIds: r2, logprobs: r2.map(() => CFG.lp2) }, promptMessages: [S, U, echo, T], responseMessage: msg('assistant', secondLabel) });
  const tree = describeTree(mgr.trees.get('sid'));
  const samples = mgr.getTrajectory('sid', { reward: 1 });
  return { p1, r1, p2, r2, tree, samples, decisions: mgr.decisions, threshold: mgr.forkThreshold, trained: samples.reduce((a, s) => a + s.lossMask.reduce((x, y) => x + y, 0), 0) };
}

export function model(cfg = CFG) {
  const p1Len = renderPrompt([msg('system', 'S'), msg('user', 'u')]).length;
  const aLen = renderResponse(cfg.firstLabel).length;
  const driftAt = p1Len + aLen - 1; // A 回放的最后一个 token（</ast>）
  const earlyAt = p1Len - 1; // A 起点之前（turn 1 的 <gen>）
  return {
    p1Len,
    aLen,
    driftAt,
    earlyAt,
    message: {
      equal: runSession({}),
      merge: runSession({ echoLabel: cfg.rewriteLabel }),
      noMerge: runSession({ echoLabel: cfg.rewriteLabel, threshold: cfg.smallThreshold }),
    },
    token: {
      clean: runSession({}),
      realign: runSession({ drift: { mode: 'replace', at: driftAt } }),
      fork: runSession({ drift: { mode: 'replace', at: driftAt }, threshold: cfg.smallThreshold }),
      early: runSession({ drift: { mode: 'insert', at: earlyAt } }),
    },
  };
}

// ---------------- 渲染 ----------------
const esc = (v) => String(v).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
export function textWidth(s, size) {
  let w = 0;
  for (const ch of String(s)) w += /[ -⯿⺀-￯]/.test(ch) ? size : size * 0.58;
  return w;
}
const STYLE = `
  text{font-family:"Segoe UI","Microsoft YaHei","PingFang SC",system-ui,sans-serif;fill:#2A313B}
  .ti{font-size:19px;font-weight:700;fill:#1F2430}
  .su{font-size:12px;fill:#747C88}
  .pt{font-size:14px;font-weight:700}
  .tx{font-size:12px;fill:#38414D}
  .sm{font-size:10.5px;fill:#5B6470}
  .tk{font-size:10.5px;fill:#2A313B;font-family:"SFMono-Regular",Menlo,Consolas,monospace}
  .cap{font-size:11.5px;fill:#5B6470}
  .panel{fill:#FBFCFE;stroke:#D9DEE7;stroke-width:1.2}
  .neutral{fill:#fff;stroke:#AEB6C2;stroke-width:1.2}
  .ghost{fill:#F5F7FA;stroke:#D9DEE7;stroke-width:1.1}
  .acc1{fill:#EAF1FD;stroke:#2563EB;stroke-width:1.5}
  .acc2{fill:#FCF1E6;stroke:#C3651F;stroke-width:1.5}
  .main{fill:none;stroke:#2563EB;stroke-width:2;marker-end:url(#arrowMain)}
  .aux{fill:none;stroke:#AEB6C2;stroke-width:1.3;marker-end:url(#arrowAux)}
  .brk{fill:none;stroke:#8A93A0;stroke-width:1.1}
`;
const DEFS = '<defs><marker id="arrowMain" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#2563EB"/></marker><marker id="arrowAux" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#AEB6C2"/></marker></defs>';

function canvas(W, H, title, desc) {
  const o = [];
  o.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="title desc">`);
  o.push(`<title id="title">${esc(title)}</title>`);
  o.push(`<desc id="desc">${esc(desc)}</desc>`);
  o.push(DEFS);
  o.push(`<style>${STYLE}</style><rect width="${W}" height="${H}" fill="white"/>`);
  const api = {
    o,
    rect: (x, y, w, h, cls = 'neutral', r = 6) => o.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" class="${cls}"/>`),
    text: (x, y, s, cls = 'tx', anchor = 'start') => o.push(`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${esc(s)}</text>`),
    arrow: (x1, y1, x2, y2, cls = 'main') => o.push(`<path d="M${x1} ${y1} L${x2} ${y2}" class="${cls}"/>`),
    path: (d, cls = 'aux') => o.push(`<path d="${d}" class="${cls}"/>`),
    done: () => {
      o.push('</svg>');
      return o.join('\n');
    },
  };
  return api;
}

// 一条 Sample 训练了哪些生成段（按 response token 名映射回 A / C）
export function trainedSpans(sample, cfg = CFG) {
  const names = { [`r:${vis(cfg.firstLabel)}`]: 'A', [`r:${vis(cfg.secondLabel)}`]: 'C' };
  const respStart = sample.tokens.length - sample.responseLength;
  const out = [];
  sample.tokens.forEach((t, i) => {
    if (i >= respStart && sample.lossMask[i - respStart] === 1 && names[t]) out.push(names[t]);
  });
  return out;
}

const KIND_CLASS = { generated: 'acc1', demoted: 'acc2', route: 'neutral' };
function nodeNote(nd, cfg = CFG) {
  const name = { [`r:${vis(cfg.firstLabel)}`]: 'A', [`r:${vis(cfg.rewriteLabel)}`]: 'A′', [`r:${vis(cfg.secondLabel)}`]: 'C' }[nd.label];
  if (nd.kind === 'generated') return `生成 ${name}`;
  if (nd.kind === 'demoted') return `A 降级为 A′`;
  return name ? `仅路由 ${name}` : '仅路由';
}

export function renderMessageLayer(m = model(), cfg = CFG) {
  const W = 1180;
  const H = 520;
  const L = m.message;
  const mergedInfo = L.merge.tree.find((t) => t.merged).merged;
  const c = canvas(
    W,
    H,
    'slime agent 消息层：回放的 assistant 消息决定挂载、改写合并还是分叉',
    `同一两轮会话的三种回放：A 与记录的叶节点字典相等时下探，导出 ${L.equal.samples.length} 条 Sample；回放成 A′ 且旧叶唯一、已生成、输出 ${mergedInfo.abandoned_response_tokens} 个 token 短于阈值 ${L.merge.threshold} 时，旧叶降级为仅路由，导出 ${L.merge.samples.length} 条只训练 C 的 Sample；阈值取 ${L.noMerge.threshold} 时不合并，消息层分叉，A 与 C 各导出一条，共 ${L.noMerge.samples.length} 条。`,
  );
  const { rect, text, arrow, path } = c;
  text(24, 34, '消息层（record_turn）：先按 role + 字典相等挂载，assistant 改写再看能否合并', 'ti');
  text(24, 56, `turn 1：messages [S, u] → 生成 A = r:${cfg.firstLabel} </ast>（${m.aLen} token）；turn 2：messages [S, u, A 的回放, t] → 生成 C = r:${cfg.secondLabel} </ast>；fork_threshold 默认 ${cfg.defaultForkThreshold}`, 'su');

  const lanes = [
    { key: 'equal', title: '① 回放 A 与记录的 A 字典相等', lines: ['_find_mount_point 下探到生成节点 A', '剩余消息 t 挂在 A 下，C 挂在 t 下'], run: L.equal, verdict: '同一条链 → 交给 token 层（图 ②）' },
    { key: 'merge', title: `② 回放成 A′（"${vis(cfg.rewriteLabel)}"）≠ A`, lines: ['唯一 assistant 子节点，是叶且已生成', `旧叶输出 ${mergedInfo.abandoned_response_tokens} < ${L.merge.threshold} → 改写合并`], run: L.merge, verdict: `merged_rewrite：abandoned_turn_index=${mergedInfo.abandoned_turn_index}` },
    { key: 'noMerge', title: `③ 同样回放 A′，但 fork_threshold = ${cfg.smallThreshold}`, lines: [`输出 ${m.aLen} ≥ ${cfg.smallThreshold} → 不合并`, 'A′ 作为仅路由兄弟节点挂出，消息层分叉'], run: L.noMerge, verdict: '两个叶 → 两条 Sample' },
  ];

  const top = 72;
  const laneH = 128;
  lanes.forEach((ln, i) => {
    const y = top + i * (laneH + 10);
    rect(24, y, 1132, laneH, 'panel', 10);
    // 条件
    rect(40, y + 12, 300, laneH - 24, 'neutral', 8);
    text(52, y + 32, ln.title, 'tx');
    ln.lines.forEach((s, k) => text(52, y + 52 + k * 18, s, 'sm'));
    text(52, y + laneH - 22, ln.verdict, 'sm');
    arrow(342, y + laneH / 2, 372, y + laneH / 2);
    // 树：depth 0..n 横向，分叉的第二个分支画在下一行
    const tree = ln.run.tree;
    const nodeW = 96;
    const gap = 14;
    const tx0 = 380;
    const rows = [];
    let row = -1;
    let lastDepth = -1;
    for (const nd of tree) {
      if (nd.depth <= lastDepth || row < 0) row += 1;
      lastDepth = nd.depth;
      rows.push({ ...nd, row });
    }
    const rowY = (r) => y + 16 + r * 50;
    rows.forEach((nd) => {
      const x = tx0 + nd.depth * (nodeW + gap);
      rect(x, rowY(nd.row), nodeW, 34, KIND_CLASS[nd.kind], 5);
      text(x + nodeW / 2, rowY(nd.row) + 15, nd.label, 'tk', 'middle');
      text(x + nodeW / 2, rowY(nd.row) + 28, nodeNote(nd, cfg), 'sm', 'middle');
    });
    rows.forEach((nd, k) => {
      if (nd.depth === 0) return;
      const parent = rows.slice(0, k).reverse().find((p) => p.depth === nd.depth - 1);
      const px = tx0 + parent.depth * (nodeW + gap) + nodeW;
      const cx = tx0 + nd.depth * (nodeW + gap);
      if (parent.row === nd.row) arrow(px, rowY(nd.row) + 17, cx, rowY(nd.row) + 17, 'aux');
      else path(`M${px - nodeW / 2} ${rowY(parent.row) + 34} L${px - nodeW / 2} ${rowY(nd.row) + 17} L${cx} ${rowY(nd.row) + 17}`, 'aux');
    });
    // 导出的 Sample
    const bx = 940;
    arrow(tx0 + 5 * (nodeW + gap) - gap + 4, y + laneH / 2, bx - 4, y + laneH / 2);
    const samples = ln.run.samples;
    rect(bx, y + 12, 206, laneH - 24, 'neutral', 8);
    text(bx + 10, y + 32, `get_trajectory 导出 ${samples.length} 条 Sample`, 'tx');
    samples.forEach((s, k) => {
      text(bx + 10, y + 52 + k * 20, `#${k + 1}：${s.tokens.length} token，response ${s.responseLength}，训练 ${trainedSpans(s, cfg).join('、')}`, 'sm');
    });
    text(bx + 10, y + laneH - 22, `训练 token 共 ${ln.run.trained}`, 'sm');
  });
  text(24, H - 16, '阅读顺序：消息层只看消息字典，不看 token；合并与分叉都在 record_turn 当场决定，linearize 时 A′ 只作 prompt 上下文。', 'cap');
  return c.done();
}

export function renderTokenLayer(m = model(), cfg = CFG) {
  const W = 1180;
  const H = 630;
  const T = m.token;
  const d = T.realign.decisions[0];
  const df = T.fork.decisions[0];
  const de = T.early.decisions[0];
  const f2 = T.fork.samples[1];
  const f2Prompt = f2.tokens.length - f2.responseLength;
  const c = canvas(
    W,
    H,
    'slime agent token 层：消息相等但 prompt_ids 漂移时的 CLEAN、REALIGN 与 FORK',
    `同一两轮会话在 linearize 时的 token 行与 loss_mask：turn 1 后 builder 持有 ${d.held} 个 token，turn 2 的 prompt_ids 在位置 ${m.driftAt} 漂移；公共前缀 ${d.realignAt}、最近 response 起点 ${d.start}，新 output ${d.outputLen} 个 token 短于默认阈值 ${d.threshold}，走 ${d.kind}，A 整段改为 mask 0，只训练 ${T.realign.trained} 个 token；阈值取 ${df.threshold} 时走 ${df.kind}，两条 Sample 共训练 ${T.fork.trained} 个 token，第二条带 ${f2Prompt} 个 prompt token 作为上下文。`,
  );
  const { rect, text, path } = c;
  text(24, 34, 'token 层（get_trajectory）：消息已挂在同一条链上，prompt_ids 却与已持有的 token 不一致', 'ti');
  text(24, 56, `已持有 = turn 1 的 prompt ${m.p1Len} + A ${m.aLen}；turn 2 的 prompt_ids 在位置 ${m.driftAt}（A 回放的 </ast>）被替换成 <DRIFT>；C = ${T.clean.r2.join(' ')}`, 'su');

  const labelW = 206;
  const cw = 60;
  const x0 = 24 + labelW;
  const cellH = 26;
  const cellX = (i) => x0 + i * cw;
  const tokRow = (y, toks, classOf, maskOf, label, sub) => {
    text(32, y + 17, label, 'tx');
    if (sub) text(32, y + 33, sub, 'sm');
    toks.forEach((t, i) => {
      rect(cellX(i) + 2, y, cw - 4, cellH, classOf(i, t), 3);
      text(cellX(i) + cw / 2, y + 17, t, 'tk', 'middle');
      const mk = maskOf(i);
      if (mk !== null) text(cellX(i) + cw / 2, y + cellH + 13, mk, 'sm', 'middle');
    });
  };
  const bracket = (from, to, y, label) => {
    const xa = cellX(from) + 3;
    const xb = cellX(to) - 3;
    path(`M${xa} ${y} L${xa} ${y + 5} L${xb} ${y + 5} L${xb} ${y}`, 'brk');
    text((xa + xb) / 2, y + 18, label, 'sm', 'middle');
  };

  // ① 输入与决策
  let y = 72;
  rect(24, y, 1132, 196, 'panel', 10);
  text(32, y + 20, '位置', 'sm');
  for (let i = 0; i < T.clean.samples[0].tokens.length; i += 1) text(cellX(i) + cw / 2, y + 20, String(i), 'sm', 'middle');
  const held = [...T.realign.p1, ...T.realign.r1];
  tokRow(y + 30, held, (i) => (i < m.p1Len ? 'ghost' : 'acc1'), (i) => (i < m.p1Len ? '·' : '1'), 'turn 1 后 builder 持有', `A 从位置 ${d.start} 起（最近 response 起点）`);
  tokRow(y + 86, T.realign.p2, (i, t) => (t === '<DRIFT>' ? 'acc2' : i < d.realignAt ? 'ghost' : 'neutral'), () => null, 'turn 2 prompt_ids', `共 ${T.realign.p2.length} 个 token`);
  bracket(0, d.realignAt, y + 86 + cellH + 3, `公共前缀 ${d.realignAt}`);
  text(32, y + 160, `drift = ${d.held} − ${d.realignAt} = ${d.drift}；${d.realignAt} ≥ ${d.start}，漂移在 A 内；新 output 长度 ${d.outputLen}：默认阈值 ${d.outputLen} < ${d.threshold} → ${d.kind}，阈值 ${df.threshold} 时 ${df.outputLen} ≥ ${df.threshold} → ${df.kind}`, 'tx');
  text(32, y + 180, `对照：漂移若插在位置 ${m.earlyAt}（A 起点之前），公共前缀 ${de.realignAt} < ${de.start} → ${de.kind}，与阈值无关`, 'tx');

  // ② 结果
  y = 282;
  rect(24, y, 1132, 308, 'panel', 10);
  const outcome = (yy, s, label, sub, lostIdx = []) => {
    const respStart = s.tokens.length - s.responseLength;
    tokRow(
      yy,
      s.tokens,
      (i, t) => {
        if (t === '<DRIFT>') return 'acc2';
        if (i < respStart) return 'ghost';
        if (lostIdx.includes(i)) return 'acc2';
        return s.lossMask[i - respStart] === 1 ? 'acc1' : 'neutral';
      },
      (i) => (i < respStart ? '·' : String(s.lossMask[i - respStart])),
      label,
      sub,
    );
  };
  const realS = T.realign.samples[0];
  const realStart = realS.tokens.length - realS.responseLength;
  const lost = [];
  for (let i = m.p1Len; i < m.p1Len + m.aLen; i += 1) if (realS.lossMask[i - realStart] === 0) lost.push(i);
  outcome(y + 14, T.clean.samples[0], '对照：无漂移 CLEAN', `1 条，训练 ${T.clean.trained} 个 token`);
  outcome(y + 70, realS, `REALIGN（阈值 ${d.threshold}）`, `1 条，训练 ${T.realign.trained} 个 token；A 清零`, lost);
  outcome(y + 126, T.fork.samples[0], `FORK（阈值 ${df.threshold}）Sample 1`, `response ${T.fork.samples[0].responseLength}，保住 A`);
  outcome(y + 182, f2, 'FORK Sample 2', `prompt ${f2Prompt} 个 token 重算`);
  text(32, y + 250, `REALIGN：位置 ${d.start} 起整段改为 mask 0，未漂移的 r:${cfg.firstLabel} 也清零`, 'sm');
  text(32, y + 270, `FORK：保住 ${T.fork.trained} 个训练 token，代价是多 1 条 Sample、${f2Prompt} 个 prompt token 重算`, 'sm');
  text(32, y + 290, '图例：灰 = 首轮 prompt 或公共前缀，白 = mask 0，蓝 = mask 1，橙 = 漂移或失去信号', 'sm');
  text(24, H - 16, '阅读顺序：先看决策行的三个量（公共前缀、最近 response 起点、新 output 长度），再比较 REALIGN 与 FORK 在"保住 A"和"多一条 Sample"之间的取舍。', 'cap');
  return c.done();
}

const here = dirname(fileURLToPath(import.meta.url));
const defaultOutput = join(here, '..', '..', '..', 'wiki', '02_engineering', '04_posttrain_frameworks', 'slime', 'assets');
const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  const outputDir = process.argv[2] ? process.argv[2] : defaultOutput;
  mkdirSync(outputDir, { recursive: true });
  const m = model();
  writeFileSync(join(outputDir, 'slime_agent_message_layer.svg'), `${renderMessageLayer(m)}\n`, 'utf8');
  writeFileSync(join(outputDir, 'slime_agent_token_layer.svg'), `${renderTokenLayer(m)}\n`, 'utf8');
}
