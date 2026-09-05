// EP 页面与三张生成 SVG 的双向契约。
//
// 运行：node --test tools/figs/svg/lib/megatron_ep_figures.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readdirSync } from 'node:fs';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { spawnSync } from 'node:child_process';

const here = dirname(fileURLToPath(import.meta.url));
const script = join(here, '..', 'megatron_ep_figures.mjs');
const repoRoot = join(here, '..', '..', '..', '..');
const page = join(
  repoRoot, 'wiki', '02_engineering', '02_train_frameworks', 'megatron-lm',
  '14_megatron_ep_analysis.md',
);
const assetDir = join(dirname(page), 'assets');
const assetNames = [
  'megatron_ep_route_compute_combine.svg',
  'megatron_ep_dispatcher_variants.svg',
  'megatron_ep_flex_backends.svg',
];

function findBrowserExecutable() {
  const candidates = [process.env.PUPPETEER_EXECUTABLE_PATH];
  if (process.platform === 'win32') {
    const cache = join(homedir(), '.cache', 'puppeteer', 'chrome');
    if (existsSync(cache)) {
      for (const entry of readdirSync(cache, { withFileTypes: true }).filter((x) => x.isDirectory()).sort().reverse()) {
        candidates.push(join(cache, entry.name, 'chrome-win64', 'chrome.exe'));
      }
    }
    candidates.push('C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe');
  } else {
    candidates.push('/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser');
  }
  return candidates.find((candidate) => candidate && existsSync(candidate));
}

const EXPECTED_CONTRACT = {
  tokensGlobal: 4,
  tokensLocalPerRank: 2,
  experts: 4,
  topk: 2,
  ep: 2,
  tp: 1,
  edges: 8,
  remoteEdges: 4,
  remoteUniqueRankCopies: 3,
  capacityAtHalf: 1,
  droppedEdgesAtHalf: 0,
  capacityAtOnePointFive: 2,
  slotsPerOwnedExpertAfterA2A: 4,
  realSlotsPerOwnedExpert: 2,
  zeroSlotsPerOwnedExpert: 2,
  backends: ['DeepEP', 'DeepEPv2', 'HybridEP', 'NCCL-EP'],
  inferenceSibling: {
    field: 'inference_moe_token_dispatcher_type',
    values: ['nccl', 'nvls'],
    classes: ['NCCLAllGatherDispatcher', 'NVLSAllGatherVDispatcher'],
    selector: 'InferenceMode.is_active',
    owners: [
      '30_megatron_rl_posttraining_consistency_analysis',
      '31_megatron_inference_engine_analysis',
    ],
  },
};

function markdownContract(markdown) {
  const match = markdown.match(/<!-- megatron-ep-figure-contract: (\{[^\n]+\}) -->/);
  assert.ok(match, 'Markdown 必须包含 megatron-ep-figure-contract');
  return JSON.parse(match[1]);
}

function svgContract(svg) {
  const match = svg.match(
    /<metadata id="megatron-ep-figure-contract">(\{[^<]+\})<\/metadata>/,
  );
  assert.ok(match, 'SVG 必须包含 megatron-ep-figure-contract metadata');
  return JSON.parse(match[1]);
}

function section(markdown, start, end) {
  const startAt = markdown.indexOf(start);
  const endAt = markdown.indexOf(end, startAt + start.length);
  assert.notEqual(startAt, -1, `缺少正文段落 ${start}`);
  assert.notEqual(endAt, -1, `缺少正文段落边界 ${end}`);
  return markdown.slice(startAt, endAt);
}

test('Markdown 与 SVG 共享同一 token/variant 契约', async () => {
  const markdown = await readFile(page, 'utf8');
  assert.deepEqual(markdownContract(markdown), EXPECTED_CONTRACT);

  for (const value of ['allgather', 'alltoall', 'flex', 'deepep', 'deepepv2', 'hybridep', 'ncclep']) {
    assert.match(markdown, new RegExp(`\\b${value}\\b`));
  }
  for (const manager of ['_DeepepManager', '_DeepepV2Manager', '_HybridEPManager', '_NCCLEPManager']) {
    assert.match(markdown, new RegExp(manager));
  }
  for (const name of assetNames) assert.match(markdown, new RegExp(`assets/${name}`));
  for (const bodyFact of [
    'T_{\\mathrm{global}}=4',
    'T_{\\mathrm{local}}=2',
    'K=T_{\\mathrm{global}}\\cdot k=4\\cdot2=8',
    'K_{\\mathrm{remote}}=4',
    'input_splits=[2,2]',
    'rank 0 的未按专家展开的收件是 $[t_0,t_1,t_3]$',
    'rank 1 是 $[t_0,t_1,t_2]$',
    '各收 3 token',
    '本 rank 四条有效 edge（全组八条）',
    'inference_moe_token_dispatcher_type ∈ {nccl, nvls}',
    'NCCLAllGatherDispatcher',
    'NVLSAllGatherVDispatcher',
    '[[30_megatron_rl_posttraining_consistency_analysis]]',
    '[[31_megatron_inference_engine_analysis]]',
  ]) {
    assert.ok(markdown.includes(bodyFact), `正文同例事实漂移：${bodyFact}`);
  }

  const capacity = section(markdown, '### 2.9', '## 3. 代码实现分析');
  for (const capacityFact of [
    'f_{\\mathrm{cap}}=0.5',
    'C=\\left\\lceil(2\\cdot2/4)\\cdot0.5\\right\\rceil=1',
    '一条也不丢',
    'f_{\\mathrm{cap}}=1.5',
    'C=\\left\\lceil(2\\cdot2/4)\\cdot1.5\\right\\rceil=2',
    '每个本地专家',
    '**2 真 + 2 零**',
  ]) {
    assert.ok(capacity.includes(capacityFact), `容量同例事实漂移：${capacityFact}`);
  }

  const deepEP = section(markdown, '### 2.4 DeepEP', '### 2.5 HybridEP');
  const deepEPv2 = section(markdown, '### 2.6 DeepEPv2', '### 2.7 NCCL-EP');
  const hybridEP = section(markdown, '### 2.5 HybridEP', '### 2.6 DeepEPv2');
  const ncclEP = section(markdown, '### 2.7 NCCL-EP', '### 2.8');
  for (const [name, lane] of Object.entries({ DeepEP: deepEP, DeepEPv2: deepEPv2, HybridEP: hybridEP, 'NCCL-EP': ncclEP })) {
    for (const stage of ['路由器', 'expert-major', 'MLP', 'combine', '反向']) {
      assert.match(lane, new RegExp(stage, 'i'), `${name} lane 缺 ${stage}`);
    }
  }
  assert.match(deepEP, /MCore ↔ DeepEP@af9a040 依赖边界/);
  assert.match(deepEP, /使 CPU 取得计数的同步/);
  assert.match(deepEP, /最大 160/);
  assert.match(deepEP, /routed_experts_compute → TEGroupedMLP\.forward\/SequentialMLP\.forward/);
  assert.match(deepEP, /Buffer\.combine\(x, handle=handle, \.\.\.\)/);
  assert.match(deepEP, /不传 `topk_weights`/);
  assert.match(deepEP, /权重梯度数据/);
  assert.match(deepEPv2, /do_expand=False/);
  assert.match(deepEPv2, /do_cpu_sync=True/);
  assert.match(deepEPv2, /ranks≤1024、experts≤2048、experts\/rank≤256/);
  assert.match(hybridEP, /MCore ↔ HybridEP 依赖边界/);
  assert.match(hybridEP, /DtoH/);
  assert.match(hybridEP, /64 对齐/);
  assert.match(hybridEP, /依赖内被丢弃/);
  assert.match(hybridEP, /累计进 `over_budget`/);
  assert.match(hybridEP, /不丢弃路由边的整步重算/);
  assert.match(hybridEP, /整个 MoE 层的 CUDA Graph 已经完成捕获/);
  assert.match(hybridEP, /才抛 `RuntimeError`/);
  assert.match(ncclEP, /MCore ↔ Transformer Engine 依赖边界/);
  assert.match(ncclEP, /SM100\+/);
  assert.match(ncclEP, /溢出会直接报错/);
  assert.match(ncclEP, /transformer_engine\.pytorch\.ep\.ep_dispatch\(buffer,/);
  assert.match(ncclEP, /transformer_engine\.pytorch\.ep\.ep_combine\(buffer,/);
  assert.doesNotMatch(ncclEP, /EpBuffer\.ep_(?:dispatch|combine)/);
  assert.doesNotMatch(markdown, /backend-defined/);
});

test('生成器、tracked SVG 与 Markdown contract 不允许单边漂移', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'megatron-ep-figures-'));
  try {
    const result = spawnSync(process.execPath, [script, outputDir], { encoding: 'utf8' });
    assert.equal(result.status, 0, result.stderr || result.stdout);
    for (const name of assetNames) {
      const generated = await readFile(join(outputDir, name), 'utf8');
      const tracked = await readFile(join(assetDir, name), 'utf8');
      assert.equal(tracked, generated, `${name} 与生成器不一致`);
      assert.deepEqual(svgContract(generated), EXPECTED_CONTRACT, `${name} contract 漂移`);
    }
  } finally {
    await rm(outputDir, { recursive: true, force: true });
  }
});

test('Flex 图是四条独立 lane，数字全部来自共同算例', async () => {
  const svg = await readFile(join(assetDir, 'megatron_ep_flex_backends.svg'), 'utf8');
  for (const backend of EXPECTED_CONTRACT.backends) {
    assert.match(svg, new RegExp(`[①②③④] ${backend}`), `${backend} 没有独立 lane`);
  }
  for (const term of [
    'T_global=4', 'T_local=2/rank', 'router=[2,4]',
    '8 edge / 4 remote / 3 remote rank-copy',
    'MCore ↔ DeepEP 边界', 'MCore ↔ DeepEPv2 边界', 'MCore ↔ HybridEP 边界',
    'MCore ↔ TE 边界', 'Backward：', 'expert-major',
    'TEGrouped/Sequential 在此消费 p', 'Buffer.combine(x, handle)',
    'forward 不传 route weights', 'rank 超预算：依赖内 drop',
    'handle flag → over_budget', '常规 PagedStashRunner：整步重跑',
    'RuntimeError，禁止动态 fallback', 'ep_dispatch(buffer,…)', 'ep_combine(buffer,…)',
    'f=.5 → C=1/丢 0 真边', 'f=1.5 + pad → C=2', '4=2 真+2 零 slots',
  ]) {
    assert.ok(svg.includes(term), `Flex 图缺少 ${term}`);
  }
  assert.doesNotMatch(svg, /EpBuffer\.ep_(?:dispatch|combine)/);
  assert.doesNotMatch(svg, /backend-defined/);
});

test('推理 sibling 是训练 dispatcher 的正交选择轴，并在正文与 dispatcher 图双向可见', async () => {
  const markdown = await readFile(page, 'utf8');
  const svg = await readFile(join(assetDir, 'megatron_ep_dispatcher_variants.svg'), 'utf8');
  const expected = EXPECTED_CONTRACT.inferenceSibling;

  assert.deepEqual(markdownContract(markdown).inferenceSibling, expected);
  assert.deepEqual(svgContract(svg).inferenceSibling, expected);
  for (const visibleFact of [
    '正交推理 sibling 轴',
    '不是训练 dispatcher 的第四值',
    'inference_moe_token_dispatcher_type ∈ {nccl,nvls}',
    'NCCLAllGatherDispatcher',
    'NVLSAllGatherVDispatcher',
    'InferenceMode.is_active()',
    '30_megatron_rl_posttraining_consistency_analysis',
    '31_megatron_inference_engine_analysis',
  ]) {
    assert.ok(svg.includes(visibleFact), `dispatcher 图缺少推理 sibling 边界：${visibleFact}`);
  }
  for (const choice of ['nccl', 'nvls']) {
    assert.match(
      svg,
      new RegExp(`<path data-inference-branch="${choice}"`),
      `dispatcher 图必须把 ${choice} 画成从配置选择点分出的独立分支`,
    );
  }
  assert.match(markdown, /正交的\*\*推理分支轴\*\*/);
  assert.match(markdown, /不能把它误并入 `\{allgather, alltoall, flex\}`/);
});

test('HybridEP 与 NCCL-EP 都直接产出 expert-major，禁止声称 HybridEP 唯一跳过本地重排', async () => {
  const markdown = await readFile(page, 'utf8');
  const flexSvg = await readFile(join(assetDir, 'megatron_ep_flex_backends.svg'), 'utf8');
  const hybridEP = section(markdown, '### 2.5 HybridEP', '### 2.6 DeepEPv2');
  const ncclEP = section(markdown, '### 2.7 NCCL-EP', '### 2.8');

  assert.match(hybridEP, /直接[^\n]*expert-major[^\n]*不调用 MCore 二次本地重排/);
  assert.match(ncclEP, /直接[^\n]*expert-major[^\n]*不调用 MCore 二次本地重排/);
  for (const artifact of [markdown, flexSvg]) {
    assert.doesNotMatch(
      artifact,
      /(?:HybridEP|四(?:个 manager|条 lane))[^\n]*(?:唯一|only)[^\n]*(?:local permute|本地重排)/i,
      '不得把 HybridEP 写成唯一省去 MCore 二次 local permute 的 manager',
    );
  }
});

test('三张可见图都锁定 local/global token 与 capacity 结果', async () => {
  for (const name of assetNames) {
    const svg = await readFile(join(assetDir, name), 'utf8');
    for (const visibleFact of ['T_global=4', 'T_local=2', 'f=.5', 'C=1', 'f=1.5', 'C=2']) {
      assert.ok(svg.includes(visibleFact), `${name} 的可见标签缺少 ${visibleFact}`);
    }
    assert.deepEqual(svgContract(svg), EXPECTED_CONTRACT, `${name} capacity contract 漂移`);
  }
});

test('真实浏览器渲染：文字不裁切且文字包围盒不重叠', async (t) => {
  const executablePath = findBrowserExecutable();
  const root = resolve(here, '../../../..');
  const puppeteerEntry = [
    join(root, 'tools', 'mkdocs-site', 'node_modules', 'puppeteer-core', 'lib', 'esm', 'puppeteer', 'puppeteer-core.js'),
    join(root, 'tools', 'html2md', 'node_modules', 'puppeteer-core', 'lib', 'esm', 'puppeteer', 'puppeteer-core.js'),
  ].find((candidate) => existsSync(candidate));
  if (!executablePath || !existsSync(puppeteerEntry)) {
    t.skip('本机没有可用 Chromium/puppeteer-core');
    return;
  }
  const { default: puppeteer } = await import(pathToFileURL(puppeteerEntry).href);
  const browser = await puppeteer.launch({
    executablePath, headless: true, args: ['--disable-gpu', '--no-sandbox'],
  });
  try {
    for (const name of assetNames) {
      const svgText = await readFile(join(assetDir, name), 'utf8');
      const browserPage = await browser.newPage();
      await browserPage.setContent(svgText);
      const audit = await browserPage.evaluate(() => {
        const svg = document.querySelector('svg');
        const view = svg.viewBox.baseVal;
        const boxes = [...svg.querySelectorAll('text')].map((element) => {
          const box = element.getBBox();
          return {
            text: element.textContent,
            x: box.x,
            y: box.y,
            right: box.x + box.width,
            bottom: box.y + box.height,
          };
        });
        const clipped = boxes.filter((box) => (
          box.x < view.x - 0.5 || box.y < view.y - 0.5
          || box.right > view.x + view.width + 0.5
          || box.bottom > view.y + view.height + 0.5
        ));
        const overlaps = [];
        for (let i = 0; i < boxes.length; i += 1) {
          for (let j = i + 1; j < boxes.length; j += 1) {
            const width = Math.min(boxes[i].right, boxes[j].right) - Math.max(boxes[i].x, boxes[j].x);
            const height = Math.min(boxes[i].bottom, boxes[j].bottom) - Math.max(boxes[i].y, boxes[j].y);
            if (width > 1 && height > 1) overlaps.push([boxes[i].text, boxes[j].text]);
          }
        }
        return { clipped, overlaps };
      });
      await browserPage.close();
      assert.deepEqual(audit.clipped, [], `${name} 有文字越出 viewBox`);
      assert.deepEqual(audit.overlaps, [], `${name} 有文字包围盒重叠`);
    }
  } finally {
    await browser.close();
  }
});
