// 锁住 slime SFT loss mask 原理图的可执行契约：图上每个数字都由同一段对话经 Qwen3 模板子集、
// SGLang tokenizer 夹具与 slime/utils/mask_utils.py 的复现推导，并且必须与
// 28_slime_sft_path_and_loss_mask_analysis.md 正文引用的数值一致。
//
// 运行：node --test tools/figs/svg/lib/slime_sft_loss_mask_figures.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

import {
  FIXTURES, MESSAGES, VOCAB, fPad, laneQwen, laneQwen3, laneQwen35, model, pretokenize, pyTail,
  renderLanes, renderQwen3, renderReplay, responseLength, systemMessageLength, textWidth,
} from '../slime_sft_loss_mask_figures.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const generator = join(here, '..', 'slime_sft_loss_mask_figures.mjs');
const slimeDir = join(here, '..', '..', '..', '..', 'wiki', '02_engineering', '04_posttrain_frameworks', 'slime');
const pagePath = join(slimeDir, '28_slime_sft_path_and_loss_mask_analysis.md');
const SVGS = ['slime_sft_loss_mask_replay.svg', 'slime_sft_loss_mask_lanes.svg'];

// jinja2 3.1.6 按 transformers apply_chat_template 的环境（ImmutableSandboxedEnvironment(trim_blocks, lstrip_blocks)、
// 自定义 tojson）渲染 vllm-project/vllm@199cb9b9 rust/src/chat/tests/templates/qwen3.jinja 的输出，逐字抄录
const JINJA_RENDERS = {
  full: '<|im_start|>system\nYou have tools.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello<|im_end|>\n<|im_start|>user\nWeather in Paris?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call><|im_end|>\n',
  m0: '<|im_start|>system\nYou have tools.<|im_end|>\n',
  m2: '<|im_start|>assistant\nHello<|im_end|>\n',
  m4: '<|im_start|>assistant\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call><|im_end|>\n',
  prefix_m2: '<|im_start|>user\nFOR CALCULATING LOSS MASK ONLY<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nHello<|im_end|>\n',
  prefix_m4: '<|im_start|>user\nFOR CALCULATING LOSS MASK ONLY<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call><|im_end|>\n',
  m0_prefix: '<|im_start|>system\nYou have tools.<|im_end|>\n<|im_start|>user\nFOR CALCULATING LOSS MASK ONLY<|im_end|>\n',
  test_two_users_gen: '<|im_start|>user\nFOR TESTING ONLY<|im_end|>\n<|im_start|>user\nFOR TESTING ONLY<|im_end|>\n<|im_start|>assistant\n',
};

// 用 slime@4c193f1f 的 slime/utils/mask_utils.py 副本（只去掉 transformers import）在同一模板与夹具 tokenizer 上跑出的 mask
const PYTHON_MASKS = {
  qwen: '000000000000000000111000000000000111111111111111111111',
  qwen3: '00000000000000000011111110000000000001111111111111111111111111',
  qwen3_5: '0000000000000000001110000000000000111111111111111111111111',
};

function viewBox(svg) {
  const match = svg.match(/viewBox="0 0 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)"/);
  assert.ok(match, 'SVG 必须声明 viewBox');
  return { w: Number(match[1]), h: Number(match[2]) };
}

// 面板（panel/zoom）内的元素与面板上下边框至少留 4px，防止末行压线
function assertPanelPadding(svg, minGap = 4) {
  const rects = [...svg.matchAll(/<rect x="(-?[\d.]+)" y="(-?[\d.]+)" width="([\d.]+)" height="([\d.]+)" rx="[\d.]+" class="([\w]+)"\/>/g)]
    .map(([, x, y, w, h, cls]) => ({ x: Number(x), y: Number(y), w: Number(w), h: Number(h), cls }));
  const panels = rects.filter((r) => r.cls === 'panel' || r.cls === 'zoom');
  assert.ok(panels.length > 0, '图中应有面板');
  for (const p of panels) {
    for (const r of rects) {
      if (r === p || r.cls === 'panel' || r.cls === 'zoom') continue;
      const inside = r.x >= p.x && r.x + r.w <= p.x + p.w && r.y >= p.y && r.y + r.h <= p.y + p.h;
      if (!inside) continue;
      assert.ok(p.y + p.h - (r.y + r.h) >= minGap, `rect 底边压到面板底边：rect y=${r.y} h=${r.h}，面板 y=${p.y} h=${p.h}`);
      assert.ok(r.y - p.y >= minGap, `rect 顶边压到面板顶边：rect y=${r.y}，面板 y=${p.y}`);
    }
  }
}

function assertInsideCanvas(svg) {
  const { w, h } = viewBox(svg);
  for (const [, x, y, rw, rh] of svg.matchAll(
    /<rect[^>]*?x="(-?\d+(?:\.\d+)?)"[^>]*?y="(-?\d+(?:\.\d+)?)"[^>]*?width="(-?\d+(?:\.\d+)?)"[^>]*?height="(\d+(?:\.\d+)?)"/g,
  )) {
    assert.ok(Number(rw) > 0, `rect 宽度必须为正 ${rw}`);
    assert.ok(Number(x) >= 0 && Number(y) >= 0, `rect 左上越界 ${x},${y}`);
    assert.ok(Number(x) + Number(rw) <= w && Number(y) + Number(rh) <= h, `rect 右下越界 ${x}+${rw},${y}+${rh}`);
  }
  for (const [, x, y] of svg.matchAll(/<text[^>]*?x="(-?\d+(?:\.\d+)?)" y="(-?\d+(?:\.\d+)?)"/g)) {
    assert.ok(Number(x) >= 0 && Number(x) <= w && Number(y) >= 0 && Number(y) <= h, `text 锚点越界 ${x},${y}`);
  }
}

test('夹具 tokenizer：预分词片段与 SGLang 期望 id 一一对应', () => {
  for (const f of FIXTURES) assert.equal(pretokenize(f.text).length, f.ids.length, f.name);
  assert.equal(VOCAB.get('<|im_start|>'), 151644);
  assert.equal(VOCAB.get('<|im_end|>'), 151645);
  assert.equal(VOCAB.get('\n'), 198);
  assert.equal(VOCAB.get('assistant'), 77091);
  assert.equal(VOCAB.get('Hello'), 9707);
  assert.equal(VOCAB.get('"}}\n'), 95642);
  assert.equal(VOCAB.has('\n\n'), false, '↵↵ 不在夹具里，不能伪造 id');
});

test('Qwen3 模板子集与 jinja2 渲染逐字一致', () => {
  const P = { role: 'user', content: 'FOR CALCULATING LOSS MASK ONLY' };
  const T = { role: 'user', content: 'FOR TESTING ONLY' };
  assert.equal(renderQwen3(MESSAGES), JINJA_RENDERS.full);
  assert.equal(renderQwen3([MESSAGES[0]]), JINJA_RENDERS.m0);
  assert.equal(renderQwen3([MESSAGES[2]]), JINJA_RENDERS.m2);
  assert.equal(renderQwen3([MESSAGES[4]]), JINJA_RENDERS.m4);
  assert.equal(renderQwen3([P, MESSAGES[2]]), JINJA_RENDERS.prefix_m2);
  assert.equal(renderQwen3([P, MESSAGES[4]]), JINJA_RENDERS.prefix_m4);
  assert.equal(renderQwen3([MESSAGES[0], P]), JINJA_RENDERS.m0_prefix);
  assert.equal(renderQwen3([T, T], { addGenerationPrompt: true }), JINJA_RENDERS.test_two_users_gen);
});

test('三条多轮 lane 与 slime 源码副本的输出一致', () => {
  assert.deepEqual(systemMessageLength(), { systemMessageLength: 0, genTokenLength: 3 });
  assert.equal(laneQwen(MESSAGES).mask.join(''), PYTHON_MASKS.qwen);
  assert.equal(laneQwen3(MESSAGES).mask.join(''), PYTHON_MASKS.qwen3);
  assert.equal(laneQwen35(MESSAGES).mask.join(''), PYTHON_MASKS.qwen3_5);
  // get_response_lengths 与 loss_mask[-0:] 的 Python 语义
  assert.equal(responseLength([0, 0, 1, 0]), 2);
  assert.equal(responseLength([0, 0]), 0);
  assert.deepEqual(pyTail([0, 0, 0], 0), [0, 0, 0]);
  // F.pad 负值裁剪
  assert.deepEqual(fPad([1, 1], -1, 1), [1, 0]);
});

test('qwen lane 回放：长度、平移与 NLL 窗口', () => {
  const q = model().qwen;
  assert.deepEqual(q.unknownInQwen, [], '图 1 的每个 token 都必须有夹具 id');
  assert.deepEqual(q.msgRanges.map(([a, b]) => b - a + 1), [9, 6, 6, 9, 24]);
  assert.equal(q.T, 54);
  assert.equal(q.firstOne, 18);
  assert.equal(q.P, 18);
  assert.equal(q.R, 36);
  assert.equal(q.targets, 24);
  assert.equal(q.contextInResponse, 12);
  assert.equal(q.shifted.length, q.T);
  assert.equal(q.shifted.at(-1), 0);
  for (let p = 0; p < q.T - 1; p += 1) assert.equal(q.shifted[p], q.mask[p + 1], `平移后位置 ${p} 应等于 mask[${p + 1}]`);
  assert.deepEqual(q.nllWindow, [17, 53]);
  const m = model();
  assert.equal(m.lanes.qwen35.sameAsFull, true);
  assert.equal(m.lanes.qwen.sameAsFull, false);
  assert.equal(m.lanes.qwen3.sameAsFull, false);
  assert.deepEqual([m.lanes.qwen.thinkBlocks, m.lanes.qwen3.thinkBlocks, m.lanes.qwen35.thinkBlocks], [0, 2, 1]);
  assert.equal(m.q35.straddle.piece, '\n\n');
  assert.equal(m.q35.straddle.bit, 1);
  assert.deepEqual(m.lanes.distill, { T: 12, promptLen: 12, responseTokens: 0, R: 0, tailLen: 12, converterAssertFails: true });
  assert.deepEqual(m.q35.headTokensSplit.filter((t) => t.start >= m.q35.straddle.start && t.end <= m.q35.straddle.end).map((t) => t.bit), [0, 1]);
});

test('生成器产出两张图，且与已跟踪的 SVG 一致、文字不越出画布', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'slime-sft-mask-'));
  try {
    const run = spawnSync(process.execPath, [generator, outputDir], { encoding: 'utf8' });
    assert.equal(run.status, 0, run.stderr || run.stdout);
    for (const name of SVGS) {
      const svg = await readFile(join(outputDir, name), 'utf8');
      assertInsideCanvas(svg);
      assertPanelPadding(svg);
      assert.doesNotMatch(svg, /\[\[/, 'SVG 不得泄漏 wikilink 标记');
      const tracked = await readFile(join(slimeDir, 'assets', name), 'utf8');
      assert.equal(tracked, svg, `已跟踪的 ${name} 必须由当前生成器重新生成`);
    }
    const m = model();
    for (const r of [renderReplay(m), renderLanes(m)]) {
      for (const b of r.boxes) {
        assert.ok(b.x0 >= 0 && b.x1 <= r.W, `文字估算宽度越界：${b.s}（${b.x0.toFixed(0)}–${b.x1.toFixed(0)}）`);
      }
    }
    const replay = await readFile(join(outputDir, SVGS[0]), 'utf8');
    assert.match(replay, /F\.pad\(loss_mask, \(17, 1\)\)/);
    assert.match(replay, /prompt_length P=18，response_length R=36/);
    const lanes = await readFile(join(outputDir, SVGS[1]), 'utf8');
    assert.match(lanes, /mask 从字符 188 开始/);
    assert.match(lanes, /游标移到 113/);
    assert.ok(textWidth('abc', 10) === 18);
  } finally {
    await rm(outputDir, { recursive: true, force: true });
  }
});

test('正文引用的数值与模型一致', async () => {
  const page = await readFile(pagePath, 'utf8');
  const m = model();
  const q = m.qwen;
  const [s2, s4] = m.q35.spans;
  const st = m.q35.straddle;
  const ones = q.shiftedOnes;
  const trained = q.trainedPositions;
  for (const needle of [
    q.mask.join(''),
    `\`system_message_length=${q.S}\` 与 \`gen_token_length=${q.G}\``,
    `依次得到 ${q.msgRanges.map(([a, b]) => b - a + 1).join('、')} 个 token，共 ${q.T} 个 token`,
    `\`Hello\` 在位置 ${q.firstOne}，是首个 1`,
    `\`prompt_length=${q.P}\`、\`response_length=${q.R}\``,
    `位置 ${q.msgRanges[3][0]}–${q.msgRanges[4][0] + q.G - 1} 是第二个 user 轮与第二个模板头`,
    `${q.R} 个 response token 里有 ${q.targets} 个目标 token，另 ${q.contextInResponse} 个只作上下文`,
    `\`F.pad(loss_mask, (${q.P - 1}, 1))\``,
    `平移后为 1 的位置是 ${ones.slice(0, 3).join('、')} 与 ${ones[3]}–${ones.at(-1)}`,
    `位置 ${q.P - 1} 是 m2 模板头的 \`↵\``,
    `位置 ${q.T - 1} 是最后一个 \`↵\``,
    `logits 窗口 \`[${q.nllWindow[0]}, ${q.nllWindow[1]})\`，得到 ${q.R} 个 logprob`,
    `留下 ${q.targets} 项`,
    `位置 ${trained.slice(0, 3).join('、')} 与 ${trained[3]}–${trained.at(-1)}，共 ${q.targets} 个`,
    `-\\frac{1}{${q.targets}}`,
    `共 ${m.lanes.qwen.T} 个 token、${m.lanes.qwen.targets} 个目标 token`,
    `字符级共有 ${m.lanes.qwen35.charTargets} 个目标字符`,
    `m2 是字符 ${s2.spanEnd}`,
    `m4 的 mask 从字符 ${s4.maskStart} 开始`,
    `占字符 \`[${st.start}, ${st.end})\``,
    `就变成 0 与 1`,
    `response 内 ${q.contextInResponse} 个 mask=0`,
    `得到 ${m.lanes.distill.T} 个 token 的全 0 mask，\`response_length=${m.lanes.distill.R}\``,
    `首个 1 后移到 m4 的 \`<tool_call>\`（位置 ${q.skipResponse.firstOne}），\`response_length\` 缩短为 ${q.skipResponse.R}，目标 token 剩 ${q.skipResponse.targets} 个`,
  ]) {
    assert.ok(page.includes(needle), `正文必须出现 ${needle}`);
  }
  for (const name of SVGS) assert.ok(page.includes(`assets/${name}`), `正文必须引用 ${name}`);
});
