// 图 1（slime_sft_loss_mask_replay.svg）：一段真实 Qwen3 模板渲染的两轮对话，走 slime 默认 `--loss-mask-type qwen`
//   得到 54 个 token 的完整 mask；从首个 1 截出 Sample.loss_mask，再经 megatron_utils/data.py::get_batch 的
//   F.pad(loss_mask, (prompt_length-1, 1)) 平移到"当前位置预测下一 token"的坐标。
// 图 2（slime_sft_loss_mask_lanes.svg）：同一段对话在三条多轮 lane（qwen / qwen3 / qwen3_5）下各训练哪些文本；
//   以及 qwen3_5 的字符 mask → offsets → token mask 投影，含跨越 `<think>\n` 前缀边界的换行 token。
//
// 证据与依赖边界：
//  - mask 规则：THUDM/slime@4c193f1f37509cca70f0e88807a9305b70f63f4e slime/utils/mask_utils.py（本文件逐条复现）。
//  - 模板：vllm-project/vllm@199cb9b964822e59ab9b58d88e7be31eb419a2ae rust/src/chat/tests/templates/qwen3.jinja
//    （vLLM 渲染器测试里的 Qwen/Qwen3-0.6B 模板夹具）。renderQwen3 只复现本例用到的子集（无 tools 参数、无 tool 角色），
//    输出已与 jinja2 按 HF apply_chat_template 的环境（trim_blocks、lstrip_blocks、tojson）渲染结果逐字比对，见测试。
//  - token id：sgl-project/sglang@0b3bb0cbe31873994c9f989fddfe2f87ca839fdd
//    experimental/sgl-router/tests/fixtures/tokenizer_parity/qwen3-30b/{short,special_token_heavy,multi_turn_with_tools}.json
//    （Qwen/Qwen3-30B-A3B 的 tokenizer 输出）；`<think>`/`</think>` 取 python/sglang/srt/sampling/custom_logit_processor.py::
//    Qwen3ThinkingBudgetLogitProcessor 的常量。片段切分按 transformers models/qwen2/tokenization_qwen2.py::PRETOKENIZE_REGEX
//    预分词，三个夹具的片段与 id 一一对应（测试断言）。`\n\n` 不在夹具里：预分词得到一个片段，是否为单一词表项未核验，
//    图 2 同时画出拆成两个 `\n` 的情形；图 1 的 qwen lane 不含该片段，全部 token 都有夹具 id。
//
// ---- spec（先写 spec 再画，见 skills/drawing-wiki-figures/SKILL.md §4）----
// 图 1 讲清楚：assistant 模板头不训练、答案与 `<|im_end|>` 及其后的 `\n` 训练；response 从首个 1 开始，
//   中间的 user 轮与第二个模板头留在 response 内但 mask=0；平移后位置 p 的 1 表示 token p+1 是目标，最后一位右补 0。
//   布局：每条消息一块，块内五行（token / 位置 / 完整 mask / Sample.loss_mask / 平移后），token 宽度随文本；
//   底部两个等宽放大条画平移箭头（位置 15–22 与 49–53），右下角结算 T、P、R、目标数与 NLL 窗口。
// 图 2 讲清楚：qwen 逐条渲染 → 没有 think 块；qwen3 在虚拟 user 前缀后渲染 → 每轮都插入并训练空 think 块；
//   qwen3_5 整段渲染 → 只有最后一个 user 之后的轮次有 think 块，字符 mask 从 `<think>\n` 之后开始；
//   token 与字符 span 有交集即为 1。布局：上表三条 lane；下方两条字符级放大（m2 尾部的结束规则、m4 头部的前缀规则）。
// acc1 标被训练的 token/字符，acc2 标与整段模板渲染不一致或依赖 tokenizer 词表的位置。
//
// 用法：node tools/figs/svg/slime_sft_loss_mask_figures.mjs [output-directory]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// ---------------- 冻结的示例输入 ----------------

// 两轮对话：问候，再问天气并以 tool call 作答（与 SGLang 夹具 multi_turn_with_tools 的 system/user/assistant 片段同源）
export const MESSAGES = Object.freeze([
  { role: 'system', content: 'You have tools.' },
  { role: 'user', content: 'Hi' },
  { role: 'assistant', content: 'Hello' },
  { role: 'user', content: 'Weather in Paris?' },
  { role: 'assistant', content: '', tool_calls: [{ function: { name: 'get_weather', arguments: { city: 'Paris' } } }] },
]);

// SGLang tokenizer-parity 夹具（Qwen/Qwen3-30B-A3B），原文与期望 id 逐字抄录
export const FIXTURES = Object.freeze([
  { name: 'short', text: 'Hello, world!', ids: [9707, 11, 1879, 0] },
  {
    name: 'special_token_heavy',
    text: '<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello<|im_end|>\n<|endoftext|>',
    ids: [151644, 8948, 198, 2610, 525, 10950, 13, 151645, 198, 151644, 872, 198, 13048, 151645, 198, 151644, 77091, 198, 9707, 151645, 198, 151643],
  },
  {
    name: 'multi_turn_with_tools',
    text: '<|im_start|>system\nYou have tools.<|im_end|>\n<|im_start|>user\nWeather in Paris?<|im_end|>\n<|im_start|>assistant\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call><|im_end|>\n',
    ids: [151644, 8948, 198, 2610, 614, 7375, 13, 151645, 198, 151644, 872, 198, 28981, 304, 12095, 30, 151645, 198, 151644, 77091, 198, 151657, 198, 4913, 606, 788, 330, 455, 69364, 497, 330, 16370, 788, 5212, 8926, 788, 330, 59604, 95642, 151658, 151645, 198],
  },
]);

// Qwen3ThinkingBudgetLogitProcessor.THINKING_START_TOKEN_ID / THINKING_END_TOKEN_ID
export const THINK_IDS = Object.freeze({ '<think>': 151667, '</think>': 151668 });

// 预分词前先按 added special token 整段切开（tokenizers 的 added-token 契约）
export const SPECIALS = Object.freeze(['<|endoftext|>', '<|im_start|>', '<|im_end|>', '<tool_call>', '</tool_call>', '<think>', '</think>']);

// transformers/models/qwen2/tokenization_qwen2.py::PRETOKENIZE_REGEX 的 JS 写法（(?i:…) 只作用于缩写分支）
const PRETOKENIZE = /'(?:[sS]|[tT]|[rR][eE]|[vV][eE]|[mM]|[lL][lL]|[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;

const range = (n) => Array.from({ length: n }, (_, i) => i);
const sum = (a) => a.reduce((x, y) => x + y, 0);

// ---------------- tokenizer：夹具 id + 预分词片段 ----------------

export function pretokenize(text, { splitNewlineRuns = false } = {}) {
  const out = [];
  let pos = 0;
  const specialRe = new RegExp(SPECIALS.map((s) => s.replace(/[|<>/]/g, (c) => `\\${c}`)).join('|'), 'g');
  const pushPlain = (seg, base) => {
    for (const m of seg.matchAll(PRETOKENIZE)) {
      const piece = m[0];
      if (splitNewlineRuns && /^\n{2,}$/.test(piece)) {
        for (let k = 0; k < piece.length; k += 1) out.push({ piece: '\n', start: base + m.index + k, end: base + m.index + k + 1, special: false });
      } else {
        out.push({ piece, start: base + m.index, end: base + m.index + piece.length, special: false });
      }
    }
  };
  for (const m of text.matchAll(specialRe)) {
    if (m.index > pos) pushPlain(text.slice(pos, m.index), pos);
    out.push({ piece: m[0], start: m.index, end: m.index + m[0].length, special: true });
    pos = m.index + m[0].length;
  }
  if (pos < text.length) pushPlain(text.slice(pos), pos);
  return out;
}

export function buildVocab() {
  const vocab = new Map(Object.entries(THINK_IDS));
  for (const f of FIXTURES) {
    const pieces = pretokenize(f.text);
    if (pieces.length !== f.ids.length) throw new Error(`fixture ${f.name}: ${pieces.length} pieces != ${f.ids.length} ids`);
    pieces.forEach((p, i) => {
      if (vocab.has(p.piece) && vocab.get(p.piece) !== f.ids[i]) throw new Error(`fixture ${f.name}: piece ${JSON.stringify(p.piece)} maps to two ids`);
      vocab.set(p.piece, f.ids[i]);
    });
  }
  return vocab;
}

export const VOCAB = buildVocab();

export function tokenize(text, opts = {}) {
  return pretokenize(text, opts).map((t) => ({ ...t, id: VOCAB.has(t.piece) ? VOCAB.get(t.piece) : null }));
}

// ---------------- 模板：qwen3.jinja 的本例子集 ----------------

const pyJsonFlat = (obj) => `{${Object.entries(obj).map(([k, v]) => {
  if (typeof v !== 'string') throw new Error('pyJsonFlat only supports flat string values');
  return `${JSON.stringify(k)}: ${JSON.stringify(v)}`;
}).join(', ')}}`;
const lstripNL = (s) => s.replace(/^\n+/, '');
const rstripNL = (s) => s.replace(/\n+$/, '');

export function renderQwen3(messages, { addGenerationPrompt = false } = {}) {
  let out = '';
  if (messages[0].role === 'system') out += `<|im_start|>system\n${messages[0].content}<|im_end|>\n`;
  let multiStepTool = true;
  let lastQueryIndex = messages.length - 1;
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const m = messages[index];
    if (multiStepTool && m.role === 'user' && typeof m.content === 'string' && !(m.content.startsWith('<tool_response>') && m.content.endsWith('</tool_response>'))) {
      multiStepTool = false;
      lastQueryIndex = index;
    }
  }
  messages.forEach((m, i) => {
    let content = typeof m.content === 'string' ? m.content : '';
    const last = i === messages.length - 1;
    if (m.role === 'user' || (m.role === 'system' && i !== 0)) {
      out += `<|im_start|>${m.role}\n${content}<|im_end|>\n`;
    } else if (m.role === 'assistant') {
      let reasoning = '';
      if (typeof m.reasoning_content === 'string') {
        reasoning = m.reasoning_content;
      } else if (content.includes('</think>')) {
        reasoning = lstripNL(rstripNL(content.split('</think>')[0]).split('<think>').at(-1));
        content = lstripNL(content.split('</think>').at(-1));
      }
      if (i > lastQueryIndex && (last || reasoning)) {
        out += `<|im_start|>assistant\n<think>\n${rstripNL(lstripNL(reasoning))}\n</think>\n\n${lstripNL(content)}`;
      } else {
        out += `<|im_start|>assistant\n${content}`;
      }
      (m.tool_calls || []).forEach((tc, k) => {
        if ((k === 0 && content) || k > 0) out += '\n';
        const fn = tc.function ?? tc;
        const args = typeof fn.arguments === 'string' ? fn.arguments : pyJsonFlat(fn.arguments);
        out += `<tool_call>\n{"name": "${fn.name}", "arguments": ${args}}\n</tool_call>`;
      });
      out += '<|im_end|>\n';
    } else if (m.role !== 'system') {
      throw new Error(`renderQwen3 subset does not cover role ${m.role}`);
    }
  });
  if (addGenerationPrompt) out += '<|im_start|>assistant\n';
  return out;
}

// ---------------- slime/utils/mask_utils.py 的复现 ----------------

// get_response_lengths：从首个 1 到末尾的长度，没有 1 时为 0
export const responseLength = (mask) => (mask.includes(1) ? mask.length - mask.indexOf(1) : 0);
// sft_rollout.generate_rollout：loss_mask[-response_length:]（Python 的 [-0:] 保留整段）
export const pyTail = (arr, n) => (n === 0 ? arr.slice() : arr.slice(arr.length - n));

const pieceIds = (toks) => toks.map((t) => t.piece);

// MultiTurnLossMaskGenerator.get_system_message_length
export function systemMessageLength() {
  const testString = 'FOR TESTING ONLY';
  const testMessages = [{ role: 'user', content: testString }, { role: 'user', content: testString }];
  const raw = pieceIds(tokenize(testString));
  const chat = pieceIds(tokenize(renderQwen3(testMessages)));
  const idx = [];
  for (let i = 0; i + raw.length <= chat.length; i += 1) if (raw.every((p, k) => chat[i + k] === p)) idx.push(i);
  if (idx.length !== 2) throw new Error(`expected 2 occurrences, got ${idx.length}`);
  const [idx1, idx2] = idx;
  const endInterval = chat.length - raw.length - idx2;
  const genTokenLength = tokenize(renderQwen3(testMessages, { addGenerationPrompt: true })).length - chat.length;
  return { systemMessageLength: idx1 - ((idx2 - idx1) - endInterval - raw.length), genTokenLength };
}

// gen_multi_turn_loss_mask_qwen：逐条消息单独套模板
export function laneQwen(messages) {
  const { systemMessageLength: S, genTokenLength: G } = systemMessageLength();
  const tokens = [];
  const mask = [];
  const owner = [];
  messages.forEach((m, i) => {
    let ids = tokenize(renderQwen3([m]));
    if (m.role !== 'system' && i > 0) ids = ids.slice(S);
    let lm = m.role === 'assistant' ? [...range(G).map(() => 0), ...range(ids.length - G).map(() => 1)] : ids.map(() => 0);
    if ((m.step_loss_mask ?? 1) !== 1) lm = ids.map(() => 0);
    ids.forEach((t, k) => { tokens.push(t); mask.push(lm[k]); owner.push(i); });
  });
  return { tokens, mask, owner, S, G };
}

// gen_multi_turn_loss_mask_qwen3：虚拟 user 前缀固定模板上下文（本例没有 tool 消息，分组退化为逐条）
export function laneQwen3(messages) {
  const { systemMessageLength: S, genTokenLength: G } = systemMessageLength();
  const prefix = { role: 'user', content: 'FOR CALCULATING LOSS MASK ONLY' };
  const prefixLen = tokenize(renderQwen3([prefix])).length;
  const tokens = [];
  const mask = [];
  const owner = [];
  messages.forEach((m, i) => {
    if (m.role === 'tool') throw new Error('tool grouping not covered by this example');
    let ids;
    if (i === 0) {
      const tailed = tokenize(renderQwen3([m, prefix]));
      ids = tailed.slice(0, tailed.length - prefixLen);
    } else {
      ids = tokenize(renderQwen3([prefix, m])).slice(prefixLen);
    }
    if (m.role !== 'system' && i > 0) ids = ids.slice(S);
    let lm = m.role === 'assistant' ? [...range(G).map(() => 0), ...range(ids.length - G).map(() => 1)] : ids.map(() => 0);
    if ((m.step_loss_mask ?? 1) !== 1) lm = ids.map(() => 0);
    ids.forEach((t, k) => { tokens.push(t); mask.push(lm[k]); owner.push(i); });
  });
  return { tokens, mask, owner, S, G };
}

export const ASSISTANT_HEADER = '<|im_start|>assistant\n';
export const THINK_PREFIX = '<think>\n';
export const END_MARKER = '<|im_end|>';

// gen_multi_turn_loss_mask_qwen3_5：整段渲染 → 字符 mask → offsets 上"有交集即 1"
export function laneQwen35(messages, tokOpts = {}) {
  const text = renderQwen3(messages);
  const tokens = tokenize(text, tokOpts);
  const charMask = range(text.length).map(() => 0);
  const spans = [];
  let cursor = 0;
  messages.forEach((m, i) => {
    if (m.role !== 'assistant') return;
    const headerPos = text.indexOf(ASSISTANT_HEADER, cursor);
    if (headerPos < 0) throw new Error('Failed to locate assistant message');
    const contentStart = headerPos + ASSISTANT_HEADER.length;
    const endPos = text.indexOf(END_MARKER, contentStart);
    if (endPos < 0) throw new Error('Failed to locate <|im_end|>');
    let spanEnd = endPos + END_MARKER.length;
    if (spanEnd < text.length && text[spanEnd] === '\n') spanEnd += 1;
    cursor = spanEnd;
    if ((m.step_loss_mask ?? 1) !== 1) return;
    const maskStart = text.slice(contentStart, contentStart + THINK_PREFIX.length) === THINK_PREFIX ? contentStart + THINK_PREFIX.length : contentStart;
    for (let p = maskStart; p < spanEnd; p += 1) charMask[p] = 1;
    spans.push({ message: i, headerPos, contentStart, maskStart, endPos, spanEnd });
  });
  const prefix = [0];
  for (const v of charMask) prefix.push(prefix.at(-1) + v);
  const mask = tokens.map((t) => (t.end <= t.start ? 0 : prefix[t.end] - prefix[t.start] > 0 ? 1 : 0));
  return { text, tokens, charMask, prefix, mask, spans };
}

// gen_multi_turn_loss_mask_distill_qwen：首条消息 + 生成提示为 prompt，最后一条 content 为 response
export function laneDistill(messages) {
  const prompt = tokenize(renderQwen3(messages.slice(0, 1), { addGenerationPrompt: true }));
  const response = tokenize(messages.at(-1).content);
  const tokens = [...prompt, ...response];
  let mask = [...prompt.map(() => 0), ...response.map(() => 1)];
  if ((messages.at(-1).step_loss_mask ?? 1) !== 1) mask = tokens.map(() => 0);
  return { tokens, mask, promptLen: prompt.length, responseTokens: response.length };
}

// get_text_from_loss_mask：连续被选中的 token 拼成文本段
export function selectedTexts(tokens, mask) {
  const out = [];
  let cur = '';
  mask.forEach((b, i) => {
    if (b === 1) cur += tokens[i].piece;
    else if (cur) { out.push(cur); cur = ''; }
  });
  if (cur) out.push(cur);
  return out;
}

// 字符级被选中的文本段
export function selectedCharTexts(text, charMask) {
  const out = [];
  let cur = '';
  charMask.forEach((b, i) => {
    if (b === 1) cur += text[i];
    else if (cur) { out.push(cur); cur = ''; }
  });
  if (cur) out.push(cur);
  return out;
}

// torch.nn.functional.pad 的一维常量填充（负值表示裁剪）
export function fPad(arr, left, right) {
  let a = arr.slice();
  a = left >= 0 ? [...range(left).map(() => 0), ...a] : a.slice(-left);
  a = right >= 0 ? [...a, ...range(right).map(() => 0)] : a.slice(0, a.length + right);
  return a;
}

// ---------------- 模型：图与正文共用的全部数值 ----------------

export function model(messages = MESSAGES) {
  const qwen = laneQwen(messages);
  const T = qwen.tokens.length;
  const R = responseLength(qwen.mask);
  const P = T - R;
  const lossMask = pyTail(qwen.mask, R);
  const shifted = fPad(lossMask, P - 1, 1);
  const targets = sum(lossMask);
  const firstOne = qwen.mask.indexOf(1);
  const nllWindow = [P - 1, T - 1];
  const shiftedOnes = shifted.map((b, p) => (b ? p : -1)).filter((p) => p >= 0);
  const trainedPositions = qwen.mask.map((b, p) => (b ? p : -1)).filter((p) => p >= 0);
  const msgRanges = messages.map((_, i) => {
    const idx = qwen.owner.map((o, p) => (o === i ? p : -1)).filter((p) => p >= 0);
    return [idx[0], idx.at(-1)];
  });
  const unknownInQwen = qwen.tokens.filter((t) => t.id === null).map((t) => t.piece);
  // step_loss_mask: 0 只排除 m2 时，首个 1 后移、response 缩短
  const skipM2 = laneQwen(messages.map((msg, i) => (i === 2 ? { ...msg, step_loss_mask: 0 } : msg)));
  const skipFirstOne = skipM2.mask.indexOf(1);
  const skipResponse = { firstOne: skipFirstOne, R: responseLength(skipM2.mask), targets: sum(pyTail(skipM2.mask, responseLength(skipM2.mask))) };

  const fullText = renderQwen3(messages);
  const full = tokenize(fullText);
  const qwen3 = laneQwen3(messages);
  const q35 = laneQwen35(messages);
  const q35split = laneQwen35(messages, { splitNewlineRuns: true });
  const distill = laneDistill(messages);
  const distillR = responseLength(distill.mask);
  const distillTail = pyTail(distill.mask, distillR);
  const samePieces = (a, b) => a.length === b.length && a.every((t, i) => t.piece === b[i].piece);

  // qwen3_5 投影窗口：m2 尾部（结束规则）与 m4 头部（前缀规则）
  const [s2, s4] = q35.spans;
  const tailWin = [s2.endPos - messages[s2.message].content.length, s2.spanEnd + '<|im_start|>user'.length];
  const headWin = [s4.headerPos, fullText.indexOf('<tool_call>', s4.headerPos) + '<tool_call>'.length];
  const windowTokens = (lane, [a, b]) => lane.tokens.map((t, i) => ({ ...t, bit: lane.mask[i], index: i })).filter((t) => t.start >= a && t.end <= b);
  const straddle = q35.tokens.map((t, i) => ({ ...t, bit: q35.mask[i], index: i })).find((t) => t.start < s4.maskStart && t.end > s4.maskStart);

  return {
    messages,
    qwen: { ...qwen, T, R, P, lossMask, shifted, targets, firstOne, nllWindow, shiftedOnes, trainedPositions, msgRanges, unknownInQwen, contextInResponse: R - targets, skipResponse },
    lanes: {
      qwen: { T, R, targets, sameAsFull: samePieces(qwen.tokens, full), texts: selectedTexts(qwen.tokens, qwen.mask), thinkBlocks: qwen.tokens.filter((t) => t.piece === '<think>').length },
      qwen3: { T: qwen3.tokens.length, R: responseLength(qwen3.mask), targets: sum(qwen3.mask), sameAsFull: samePieces(qwen3.tokens, full), texts: selectedTexts(qwen3.tokens, qwen3.mask), thinkBlocks: qwen3.tokens.filter((t) => t.piece === '<think>').length, S: qwen3.S, G: qwen3.G },
      qwen35: { T: q35.tokens.length, R: responseLength(q35.mask), targets: sum(q35.mask), sameAsFull: samePieces(q35.tokens, full), charTexts: selectedCharTexts(q35.text, q35.charMask), thinkBlocks: q35.tokens.filter((t) => t.piece === '<think>').length, charTargets: sum(q35.charMask) },
      qwen35split: { T: q35split.tokens.length, targets: sum(q35split.mask) },
      distill: { T: distill.tokens.length, promptLen: distill.promptLen, responseTokens: distill.responseTokens, R: distillR, tailLen: distillTail.length, converterAssertFails: distillTail.length !== distillR },
    },
    q35: { text: q35.text, charMask: q35.charMask, prefix: q35.prefix, spans: q35.spans, tailWin, headWin, tailTokens: windowTokens(q35, tailWin), headTokens: windowTokens(q35, headWin), headTokensSplit: windowTokens(q35split, headWin), straddle },
  };
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
  .mono{font-family:"Cascadia Mono",Consolas,"Courier New",monospace;font-size:10px;fill:#38414D}
  .bit{font-family:"Cascadia Mono",Consolas,"Courier New",monospace;font-size:10px;fill:#38414D;font-weight:700}
  .pos{font-family:"Cascadia Mono",Consolas,"Courier New",monospace;font-size:8.5px;fill:#8A919C}
  .panel{fill:#FBFCFE;stroke:#D9DEE7;stroke-width:1.2}
  .neutral{fill:#fff;stroke:#AEB6C2;stroke-width:1.1}
  .ghost{fill:#F5F7FA;stroke:#D9DEE7;stroke-width:1}
  .acc1{fill:#EAF1FD;stroke:#2563EB;stroke-width:1.3}
  .acc2{fill:#FCF1E6;stroke:#C3651F;stroke-width:1.3}
  .acc2d{fill:#FCF1E6;stroke:#C3651F;stroke-width:1.3;stroke-dasharray:3 2}
  .zoom{fill:#fff;stroke:#747C88;stroke-width:1;stroke-dasharray:4 3}
  .main{fill:none;stroke:#2563EB;stroke-width:1.4;marker-end:url(#arrowMain)}
  .aux{fill:none;stroke:#AEB6C2;stroke-width:1.2;stroke-dasharray:4 3}
`;
const DEFS = '<defs><marker id="arrowMain" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#2563EB"/></marker></defs>';

// 估算文本宽度：CJK 与全角记 1em，其余记 0.6em（等宽 0.6em）
export function textWidth(s, fontSize) {
  let w = 0;
  for (const ch of String(s)) w += /[⺀-鿿＀-￯　-〿]/.test(ch) ? fontSize : fontSize * 0.6;
  return w;
}
const FONT = { ti: 19, su: 12, pt: 14, tx: 12, sm: 10.5, cap: 11.5, mono: 10, bit: 10, pos: 8.5 };

export const glyph = (piece) => piece.replace(/\n/g, '↵').replace(/ /g, '␣');

function canvas(W, H, title, desc) {
  const o = [];
  const boxes = [];
  const rect = (x, y, w, h, cls = 'neutral', r = 4) => { o.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" class="${cls}"/>`); };
  const text = (x, y, s, cls = 'tx', anchor = 'start') => {
    const w = textWidth(s, FONT[cls] ?? 12);
    const x0 = anchor === 'middle' ? x - w / 2 : anchor === 'end' ? x - w : x;
    boxes.push({ x0, x1: x0 + w, y, s });
    o.push(`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${esc(s)}</text>`);
  };
  const line = (x1, y1, x2, y2, cls = 'main') => o.push(`<path d="M${x1} ${y1} L${x2} ${y2}" class="${cls}"/>`);
  o.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="title desc">`);
  o.push(`<title id="title">${esc(title)}</title>`);
  o.push(`<desc id="desc">${esc(desc)}</desc>`);
  o.push(DEFS);
  o.push(`<style>${STYLE}</style><rect width="${W}" height="${H}" fill="white"/>`);
  return { o, boxes, rect, text, line, finish: () => { o.push('</svg>'); return { svg: o.join('\n'), boxes, W, H }; } };
}

export function renderReplay(m = model()) {
  const q = m.qwen;
  const W = 1200;
  const labelX = 150;
  const x0 = 160;
  const cellW = (piece) => Math.max(22, Math.ceil(textWidth(glyph(piece), FONT.mono)) + 8);
  const gap = 3;
  const blockH = 126;
  const top = 72;
  const ROWS = [[0, 1], [2, 3], [4]];
  const zoomTop = top + ROWS.length * blockH + 4;
  const H = zoomTop + 244;
  const c = canvas(W, H, 'slime SFT loss mask 回放：qwen lane × Qwen3 模板', `一段两轮对话经 Qwen3 模板逐条渲染成 ${q.T} 个 token，完整 mask 首个 1 在位置 ${q.firstOne}，response_length=${q.R}，目标 token ${q.targets} 个；get_batch 用 F.pad(loss_mask, (${q.P - 1}, 1)) 把 mask 平移到预测位置。`);
  const { rect, text, line } = c;

  text(24, 30, `SFT loss mask 回放：默认 --loss-mask-type qwen × 真实 Qwen3 模板，一段两轮对话 = ${q.T} 个 token`, 'ti');
  text(24, 52, `逐条消息套模板（system_message_length=${q.S}，gen_token_length=${q.G}）· token id 取自 SGLang Qwen3 tokenizer 夹具 · ↵ 为换行，␣ 为空格 · 蓝：mask=1 · 白：response 内的 0`, 'su');

  const NOTES = {
    0: 'prompt 段',
    1: 'prompt 段',
    2: `模板头为 0，首个 1 在 ${q.firstOne}`,
    3: 'response 内的上下文，全 0',
    4: '模板头为 0；tool call、</tool_call>、<|im_end|>、↵ 全为 1',
  };
  ROWS.forEach((row, r) => {
    const by = top + r * blockH;
    rect(20, by, W - 40, blockH - 8, 'panel', 8); // 末行在 by+110 结束，距面板底边 8px
    text(labelX, by + 44, 'token', 'sm', 'end');
    text(labelX, by + 58, '位置', 'sm', 'end');
    text(labelX, by + 75, '完整 mask', 'sm', 'end');
    text(labelX, by + 91, 'Sample.loss_mask', 'sm', 'end');
    text(labelX, by + 107, 'get_batch 平移后', 'sm', 'end');
    let x = x0;
    row.forEach((i, k) => {
      const msg = m.messages[i];
      const [a, b] = q.msgRanges[i];
      if (k > 0) {
        x += 14;
        line(x - 8, by + 8, x - 8, by + blockH - 16, 'aux');
      }
      const role = msg.role === 'assistant' && msg.tool_calls ? 'assistant（tool_calls）' : msg.role;
      text(x, by + 20, `m${i} ${role}`, 'pt');
      text(x + textWidth(`m${i} ${role}`, FONT.pt) + 10, by + 20, NOTES[i], 'cap');
      for (let p = a; p <= b; p += 1) {
        const t = q.tokens[p];
        const w = cellW(t.piece);
        const inResp = p >= q.firstOne;
        const cls = q.mask[p] ? 'acc1' : inResp ? 'neutral' : 'ghost';
        rect(x, by + 30, w, 20, cls, 3);
        text(x + w / 2, by + 44, glyph(t.piece), 'mono', 'middle');
        text(x + w / 2, by + 60, String(p), 'pos', 'middle');
        rect(x, by + 64, w, 14, q.mask[p] ? 'acc1' : 'ghost', 2);
        text(x + w / 2, by + 75, String(q.mask[p]), 'bit', 'middle');
        if (inResp) {
          const li = p - q.firstOne;
          rect(x, by + 80, w, 14, q.lossMask[li] ? 'acc1' : 'neutral', 2);
          text(x + w / 2, by + 91, String(q.lossMask[li]), 'bit', 'middle');
        } else {
          text(x + w / 2, by + 91, '·', 'pos', 'middle');
        }
        rect(x, by + 96, w, 14, q.shifted[p] ? 'acc1' : 'ghost', 2);
        text(x + w / 2, by + 107, String(q.shifted[p]), 'bit', 'middle');
        x += w + gap;
      }
    });
  });

  // ---- 放大：平移的逐位对应 ----
  const zy = zoomTop;
  rect(20, zy, W - 40, 234, 'zoom', 8);
  text(30, zy + 20, `放大：F.pad(loss_mask, (prompt_length−1, 1)) = F.pad(loss_mask, (${q.P - 1}, 1))，位置 p 的 1 表示"用 p 处 logits 预测 token p+1"`, 'pt');
  const zoomStrip = (sx, sy, positions, label) => {
    const w = 76;
    text(sx, sy - 6, label, 'cap');
    text(sx - 6, sy + 14, 'token', 'sm', 'end');
    text(sx - 6, sy + 40, 'loss_mask[i]', 'sm', 'end');
    text(sx - 6, sy + 92, '平移后 [p]', 'sm', 'end');
    positions.forEach((p, k) => {
      const x = sx + k * (w + 6);
      const inRange = p < q.T;
      if (inRange) {
        const t = q.tokens[p];
        rect(x, sy, w, 20, q.mask[p] ? 'acc1' : p >= q.firstOne ? 'neutral' : 'ghost', 3);
        text(x + w / 2, sy + 14, glyph(t.piece), 'mono', 'middle');
        text(x + w / 2, sy + 30, `p=${p}`, 'pos', 'middle');
        if (p >= q.firstOne) {
          const li = p - q.firstOne;
          rect(x, sy + 32, w, 14, q.lossMask[li] ? 'acc1' : 'neutral', 2);
          text(x + w / 2, sy + 43, `i=${li}:${q.lossMask[li]}`, 'bit', 'middle');
        } else {
          text(x + w / 2, sy + 43, '左补 0', 'pos', 'middle');
        }
        rect(x, sy + 80, w, 16, q.shifted[p] ? 'acc1' : 'ghost', 2);
        text(x + w / 2, sy + 92, `${q.shifted[p]}`, 'bit', 'middle');
        if (p >= q.firstOne && k > 0) line(x + w / 2, sy + 48, x - 6 - w / 2 + 2, sy + 78, q.lossMask[p - q.firstOne] ? 'main' : 'aux');
      } else {
        rect(x, sy + 80, w, 16, 'acc2', 2);
        text(x + w / 2, sy + 92, '越界', 'sm', 'middle');
      }
    });
  };
  const posA = range(6).map((k) => q.firstOne - 2 + k);
  zoomStrip(130, zy + 58, posA, `位置 ${posA[0]}–${posA.at(-1)}：m2 模板头 ↵（p=${q.P - 1}）预测首个目标 Hello`);
  const posB = range(5).map((k) => q.T - 5 + k);
  zoomStrip(700, zy + 58, posB, `位置 ${posB[0]}–${posB.at(-1)}：最后一个 ↵ 没有下一 token，右补 0`);
  text(30, zy + 178, `T=${q.T} · 首个 1 在 ${q.firstOne} → prompt_length P=${q.P}，response_length R=${q.R} · Σloss_mask=${q.targets}，response 内另有 ${q.contextInResponse} 个 mask=0 的上下文 token`, 'tx');
  text(30, zy + 198, `sft_loss_function 取 logits 位置 [${q.nllWindow[0]}, ${q.nllWindow[1]}) 的 ${q.R} 个 logprob，与 Sample.loss_mask 相乘 → ${q.targets} 项 NLL；平移后的 mask 作为 loss_mask 传入模型前向`, 'tx');
  text(30, zy + 218, `平移后为 1 的位置：${q.shiftedOnes.slice(0, 3).join(', ')} 与 ${q.shiftedOnes[3]}–${q.shiftedOnes.at(-1)}；训练目标 token 位置：${q.trainedPositions.slice(0, 3).join(', ')} 与 ${q.trainedPositions[3]}–${q.trainedPositions.at(-1)}`, 'cap');
  return c.finish();
}

const abbreviate = (s) => glyph(s.replace(/\{"name": "get_weather", "arguments": \{"city": "Paris"\}\}/, '{…}'));

export function renderLanes(m = model()) {
  const L = m.lanes;
  const Q = m.q35;
  const W = 1200;
  const H = 904;
  const c = canvas(W, H, 'slime 多轮 loss mask 四条 lane 与 qwen3_5 字符投影', '上表列出同一段两轮对话在 qwen、qwen3、qwen3_5、distill_qwen 四条 lane 下被训练的文本与 think 块差异；下方两条字符级放大复现 qwen3_5 的结束规则（紧接的换行计入 span）和前缀规则（只排除 <think> 与其后换行），以及跨越 span 起点的换行 token 按有交集即置 1。');
  const { rect, text } = c;
  text(24, 32, '同一段对话在真实 Qwen3 模板下：四条 lane 训练的文本不同', 'ti');
  text(24, 54, 'qwen / qwen3 逐段套模板后拼接；qwen3_5 整段渲染后用 offsets 投影；distill_qwen 只取首条与末条 · 蓝：被训练 · 橙：与整段模板渲染不一致或取决于词表', 'su');

  // ---- 上表 ----
  const ty = 70;
  rect(20, ty, W - 40, 336, 'panel', 8);
  const cols = [34, 150, 440, 760];
  text(cols[0], ty + 22, 'lane', 'pt');
  text(cols[1], ty + 22, '渲染方式', 'pt');
  text(cols[2], ty + 22, 'm2 被训练的文本', 'pt');
  text(cols[3], ty + 22, 'm4 被训练的文本（tool call 体略作 {…}）', 'pt');
  const rows = [
    {
      name: 'qwen（默认）', how: [`每条消息单独套模板，`, `assistant 前 ${m.qwen.G} 个 token 置 0`],
      m2: abbreviate(L.qwen.texts[0]), m4: abbreviate(L.qwen.texts[1]), cls: 'acc2',
      note: `think 块 ${L.qwen.thinkBlocks} 个；m4 缺整段渲染里的空 think 块 · T=${L.qwen.T}，目标 ${L.qwen.targets}`,
    },
    {
      name: 'qwen3', how: ['在虚拟 user 前缀之后套模板，', '再裁掉前缀'],
      m2: abbreviate(L.qwen3.texts[0]), m4: abbreviate(L.qwen3.texts[1]), cls: 'acc2',
      note: `think 块 ${L.qwen3.thinkBlocks} 个：每轮都被当成"最后一个 user 之后"而插入并训练；m2 与整段渲染不一致`,
    },
    {
      name: 'qwen3_5', how: ['整段渲染一次，字符 span', '经 offsets 投影到 token'],
      m2: abbreviate(L.qwen35.charTexts[0]), m4: abbreviate(L.qwen35.charTexts[1]), cls: 'acc1',
      note: `think 块 ${L.qwen35.thinkBlocks} 个，只在最后一轮；token 序列与整段渲染逐项相等（源码有相等校验）· 字符级目标 ${L.qwen35.charTargets} 个`,
    },
    {
      name: 'distill_qwen', how: ['首条消息加生成提示为 prompt，', '最后一条 content 为 response'],
      m2: '（中间消息不进入输出）', m4: `content 为空串 → response ${L.distill.responseTokens} 个 token`, cls: 'acc2', plain: true,
      note: `共 ${L.distill.T} 个 token 全为 0；response_length=${L.distill.R}，loss_mask[-0:] 仍保留 ${L.distill.tailLen} 位 → converter 的长度断言失败`,
    },
  ];
  rows.forEach((r, k) => {
    const ry = ty + 36 + k * 74;
    rect(28, ry, W - 56, 66, k === 2 ? 'ghost' : 'neutral', 6);
    const boxCls = r.plain ? 'ghost' : 'acc1';
    text(cols[0], ry + 22, r.name, 'pt');
    text(cols[1], ry + 20, r.how[0], 'sm');
    text(cols[1], ry + 35, r.how[1], 'sm');
    rect(cols[2] - 6, ry + 8, Math.ceil(textWidth(r.m2, r.plain ? FONT.sm : FONT.mono)) + 12, 20, boxCls, 3);
    text(cols[2], ry + 22, r.m2, r.plain ? 'sm' : 'mono');
    rect(cols[3] - 6, ry + 8, Math.ceil(textWidth(r.m4, r.plain ? FONT.sm : FONT.mono)) + 12, 20, k === 2 ? 'acc1' : 'acc2', 3);
    text(cols[3], ry + 22, r.m4, r.plain ? 'sm' : 'mono');
    text(cols[2] - 6, ry + 54, r.note, 'cap');
  });

  // ---- 字符级放大 ----
  const charStrip = (sy, win, toks, label, { splitToks = null } = {}) => {
    const [a, b] = win;
    const cw = 17;
    const sx = 170;
    rect(20, sy, W - 40, splitToks ? 232 : 160, 'zoom', 8);
    text(30, sy + 20, label, 'pt');
    text(sx - 8, sy + 46, '字符', 'sm', 'end');
    text(sx - 8, sy + 64, 'char_mask', 'sm', 'end');
    text(sx - 8, sy + 96, 'token [start,end)', 'sm', 'end');
    text(sx - 8, sy + 124, 'Δ前缀和', 'sm', 'end');
    text(sx - 8, sy + 142, 'token bit', 'sm', 'end');
    for (let p = a; p < b; p += 1) {
      const x = sx + (p - a) * cw;
      rect(x, sy + 32, cw - 1, 18, Q.charMask[p] ? 'acc1' : 'ghost', 2);
      text(x + cw / 2, sy + 45, glyph(Q.text[p]), 'mono', 'middle');
      text(x + cw / 2, sy + 64, String(Q.charMask[p]), 'bit', 'middle');
    }
    const tokRow = (list, ty2, bitY, dashed) => {
      list.forEach((t) => {
        const x = sx + (t.start - a) * cw;
        const w = (t.end - t.start) * cw - 1;
        const straddles = Q.straddle && t.start === Q.straddle.start && t.end === Q.straddle.end;
        const cls = straddles || dashed ? (t.bit ? 'acc2' : 'acc2d') : t.bit ? 'acc1' : 'ghost';
        rect(x, ty2, w, 20, cls, 3);
        const label = w > textWidth(glyph(t.piece), FONT.mono) + 4 ? glyph(t.piece) : '';
        if (label) text(x + w / 2, ty2 + 14, label, 'mono', 'middle');
        const d = Q.prefix[t.end] - Q.prefix[t.start];
        text(x + w / 2, bitY, `${d}`, 'mono', 'middle');
        text(x + w / 2, bitY + 18, `${t.bit}`, 'bit', 'middle');
      });
    };
    tokRow(toks, sy + 82, sy + 124, false);
    if (splitToks) {
      const diff = splitToks.filter((t) => !toks.some((u) => u.start === t.start && u.end === t.end));
      text(sx - 8, sy + 176, '若 ↵↵ 拆成两个 token', 'sm', 'end');
      tokRow(diff, sy + 162, sy + 200, true);
    }
  };
  const s2 = Q.spans[0];
  const s4 = Q.spans[1];
  charStrip(422, Q.tailWin, Q.tailTokens, `结束规则（m2）：找到 <|im_end|> 后，紧接的 ↵ 也计入 span，游标移到 ${s2.spanEnd}；下一条 <|im_start|>user 不训练`);
  charStrip(592, Q.headWin, Q.headTokens, `前缀规则（m4）：content 以 <think>↵ 开头时只排除这 8 个字符，mask 从字符 ${s4.maskStart} 开始`, { splitToks: Q.headTokensSplit });
  const st = Q.straddle;
  text(30, 844, `跨界 token：↵↵ 占字符 [${st.start},${st.end})，其中 ${st.start} 属于被排除的前缀、${st.start + 1} 在 span 内，前缀和差为 ${Q.prefix[st.end] - Q.prefix[st.start]} > 0，于是整个 token 置 ${st.bit}`, 'tx');
  text(30, 864, `↵↵ 是一个预分词片段（Qwen2 预分词正则）；它是否为单个词表项取决于 tokenizer，本机未核验。拆开时变为 0 与 1，被训练的字符不变`, 'tx');
  text(30, 888, '模板：vLLM@199cb9b9 qwen3.jinja（Qwen3-0.6B 夹具）· id：SGLang v0.5.15.post1 Qwen3 tokenizer 夹具 · 规则：slime@4c193f1f mask_utils.py', 'su');
  return c.finish();
}

const here = dirname(fileURLToPath(import.meta.url));
const defaultOutput = join(here, '..', '..', '..', 'wiki', '02_engineering', '04_posttrain_frameworks', 'slime', 'assets');
const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  const outputDir = process.argv[2] ? process.argv[2] : defaultOutput;
  mkdirSync(outputDir, { recursive: true });
  const m = model();
  writeFileSync(join(outputDir, 'slime_sft_loss_mask_replay.svg'), `${renderReplay(m).svg}\n`, 'utf8');
  writeFileSync(join(outputDir, 'slime_sft_loss_mask_lanes.svg'), `${renderLanes(m).svg}\n`, 'utf8');
}
