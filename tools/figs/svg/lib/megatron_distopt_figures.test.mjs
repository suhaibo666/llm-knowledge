// Distributed Optimizer 页面与四张外部 SVG 的双向契约。
//
// - 只改 SVG：重新生成结果与 tracked asset 不同，失败。
// - 只改 Markdown 中的结构化同例契约：生成器嵌入新的摘要，tracked asset 与结果不同，失败。
//   无关措辞不参与摘要。
// - 改坏正文关键变体/数字：页面断言失败。
//
// 运行：node --test tools/figs/svg/lib/megatron_distopt_figures.test.mjs

import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { existsSync, readdirSync } from 'node:fs';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import assert from 'node:assert/strict';
import { test } from 'node:test';

const here = dirname(fileURLToPath(import.meta.url));
const script = join(here, '..', 'megatron_distributed_optimizer_figures.mjs');
const pageDir = join(
  here, '..', '..', '..', '..', 'wiki', '02_engineering', '02_train_frameworks',
  'megatron-lm',
);
const assetDir = join(pageDir, 'assets');
const pagePath = join(pageDir, '16_megatron_distributed_optimizer_analysis.md');
const assetNames = [
  'megatron_distopt_flat_buffer_ranges.svg',
  'megatron_distopt_rs_update_ag.svg',
  'megatron_distopt_live_paths.svg',
  'megatron_distopt_fp32_rs_hsdp.svg',
];

function findBrowserExecutable() {
  const candidates = [process.env.PUPPETEER_EXECUTABLE_PATH];
  if (process.platform === 'win32') {
    const cache = join(homedir(), '.cache', 'puppeteer', 'chrome');
    if (existsSync(cache)) {
      for (const entry of readdirSync(cache, { withFileTypes: true }).filter((item) => item.isDirectory()).sort().reverse()) {
        candidates.push(join(cache, entry.name, 'chrome-win64', 'chrome.exe'));
      }
    }
    candidates.push('C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe');
  } else {
    candidates.push('/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser');
  }
  return candidates.find((candidate) => candidate && existsSync(candidate));
}

async function readGenerated(outputDir, name) {
  const generated = await readFile(join(outputDir, name), 'utf8');
  const tracked = await readFile(join(assetDir, name), 'utf8');
  assert.equal(normalizeEol(tracked), normalizeEol(generated), `${name} 与生成器或 Markdown 已漂移`);
  return generated;
}

test('四张 SVG 可重生成，且同例契约摘要使 Markdown/图单边修改失败', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'megatron-distopt-contract-'));
  try {
    const result = spawnSync(process.execPath, [script, outputDir], { encoding: 'utf8' });
    assert.equal(result.status, 0, result.stderr || result.stdout);

    const generated = new Map();
    for (const name of assetNames) generated.set(name, await readGenerated(outputDir, name));

    const markdown = await readFile(pagePath, 'utf8');
    const contractMatch = markdown.match(
      /<!-- distopt-figure-contract:start -->([\s\S]*?)<!-- distopt-figure-contract:end -->/,
    );
    assert.ok(contractMatch, 'Markdown 必须有 distopt-figure-contract block');
    const expectedHash = createHash('sha256').update(contractMatch[1].trim()).digest('hex');
    const planes = generated.get('megatron_distopt_fp32_rs_hsdp.svg');
    assert.match(
      planes,
      new RegExp(`data-contract-sha256="${expectedHash}"`),
      '四-lane 图必须嵌入当前 Markdown 同例契约摘要',
    );
    assert.match(
      generated.get('megatron_distopt_live_paths.svg'),
      new RegExp(`data-contract-sha256="${expectedHash}"`),
      'live-path 图必须嵌入含 LayerWise 同例的 Markdown 契约摘要',
    );
  } finally {
    await rm(outputDir, { recursive: true, force: true });
  }
});

test('同一 N=16 例独立复演 all-reduce、standard RS、custom FP32 RS 与 HSDP', async () => {
  const planes = await readFile(join(assetDir, 'megatron_distopt_fp32_rs_hsdp.svg'), 'utf8');
  assert.match(planes, /N=16 · d=4 · q\[0,2\) \| p\[2,7\) \| r\[7,13\) \| pad\[13,16\)/);
  assert.match(planes, /forward → loss → gradient-ready \/ backward handoff/);

  const lanes = [
    [
      'ordinary-all-reduce',
      [
        'all-reduce', 'plain non-LayerWise optimizer', 'force+k&gt;1: one instance',
        'global only if k=1', 'inter local-shard AR',
        '所有 DP rank 都不持有全局 full gradient', '80 B/rank',
      ],
    ],
    ['standard-reduce-scatter', ['reduce-scatter', 'rank1 local owner', '[4,8)', '24 B/rank', 'parameter all-gather']],
    [
      'custom-fp32-accumulation-reduce-scatter',
      [
        'FP32 accumulation', 'all_to_all_single', 'A1 full-size temp', 'custom handle.wait',
        'downcast / copy', 'one bucket / bucket group',
        'num_distributed_optimizer_instances = 1', '+48 B temp', 'predecessor',
      ],
    ],
    [
      'multi-instance-hsdp',
      [
        'HSDP', 'instances', 'I0={0,1,2,3}', 'I1={4,5,6,7}',
        '{0,4} {1,5} {2,6} {3,7}', 'rank1 &amp; rank5 own [4,8)',
        'dense bucket-group collection: 1 shared stream',
        'expert bucket-group collection: another shared stream',
        'not one stream per bucket group',
        '24 B intra RS + 8 B inter AR + 24 B AG = 56 B/rank',
      ],
    ],
  ];

  for (const [id, terms] of lanes) {
    const match = planes.match(new RegExp(`<g data-plane="${id}">([\\s\\S]*?)<\\/g>`));
    assert.ok(match, `缺少独立 lane ${id}`);
    for (const term of terms) assert.ok(match[1].includes(term), `${id} 缺少 ${term}`);
  }
});

test('ordinary AR 不吞掉 LayerWise；两种 LayerWise layout 都闭合到参数重建', async () => {
  const page = await readFile(pagePath, 'utf8');
  const planes = await readFile(join(assetDir, 'megatron_distopt_fp32_rs_hsdp.svg'), 'utf8');
  const livePaths = await readFile(join(assetDir, 'megatron_distopt_live_paths.svg'), 'utf8');
  const ordinary = planes.match(/<g data-plane="ordinary-all-reduce">([\s\S]*?)<\/g>/);
  assert.ok(ordinary, '缺少 ordinary all-reduce lane');
  assert.ok(
    ordinary[1].includes('plain non-LayerWise optimizer'),
    'ordinary lane 必须把 full-update 语义收窄到 plain non-LayerWise optimizer',
  );
  assert.doesNotMatch(
    ordinary[1],
    /non-DistOpt:\s*full update/,
    '不得再把所有 use_distributed_optimizer=False 路径泛化为 non-DistOpt full update',
  );

  const layerWiseLanes = [
    [
      'layerwise-decoupled',
      [
        'same q(2), p(5), r(6)',
        'effective use_distributed_optimizer=False',
        'gradient all-reduce',
        'whole-param owner update',
        'variable-size allgather_params',
        'owned sizes [2,5,6,0]',
        '39 B/rank',
        'no equal-shard padding',
        'consumer wait',
        'num_distributed_optimizer_instances = 1',
      ],
    ],
    [
      'layerwise-layout',
      [
        'same q(2), p(5), r(6)',
        'effective use_distributed_optimizer=True',
        'gradient reduce-scatter (not AR)',
        'whole-param owner update',
        'padded equal-shard layout',
        'Nlayout=256',
        'padding=243 over raw 13',
        'buffer all-gather reconstruction',
        'all_gather_into_tensor',
        'RS+AG=768 B/rank',
        'num_distributed_optimizer_instances = 1',
      ],
    ],
  ];
  for (const [id, terms] of layerWiseLanes) {
    const lane = livePaths.match(new RegExp(`<g data-plane="${id}">([\\s\\S]*?)<\\/g>`));
    assert.ok(lane, `缺少独立 lane ${id}`);
    for (const term of terms) assert.ok(lane[1].includes(term), `${id} 缺少 ${term}`);
  }

  const contract = page.match(
    /<!-- distopt-figure-contract:start -->([\s\S]*?)<!-- distopt-figure-contract:end -->/,
  );
  assert.ok(contract, 'Markdown 必须有 distopt-figure-contract block');
  for (const term of [
    'ordinary AR=plain non-LayerWise only',
    'LayerWise decoupled=AR→whole-param owner update→variable-size allgather_params',
    'LayerWise layout=RS→whole-param owner update→padded buffer AG',
    'decoupled owners=[q],[p],[r],[]',
    'layout owners=[r],[p],[q],[]',
    'padded layout=4*64=256',
    'padding=243',
  ]) {
    assert.ok(contract[1].includes(term), `结构化同例契约缺少 ${term}`);
  }

  const ordinarySection = page.match(/#### 2\.4\.1 ([\s\S]*?)(?=#### 2\.4\.2 )/);
  assert.ok(ordinarySection, '缺少 §2.4.1 ordinary all-reduce 数据面说明');
  assert.ok(
    ordinarySection[1].includes('plain non-LayerWise optimizer'),
    '§2.4.1 必须显式限定 ordinary full-update lane',
  );
  assert.doesNotMatch(
    ordinarySection[1],
    /因此非 DistOpt 可在每个 replica 做同一 update/,
    '§2.4.1 不得再把 LayerWise decoupled 混进 ordinary non-DistOpt update 语义',
  );

  const layerWiseSection = page.match(/### 2\.5 ([\s\S]*?)(?=### 2\.6 )/);
  assert.ok(layerWiseSection, '缺少 §2.5 LayerWise 两数据面回放');
  for (const term of [
    'q(2),p(5),r(6)',
    'allgather_params',
    'all_gather_into_tensor',
    '各多 243 slots',
    'partition_buckets',
    'overlap_param_gather_with_optimizer_step',
    'padded layout 还拒绝 FP8/FP4 param gather',
  ]) {
    assert.ok(layerWiseSection[1].includes(term), `§2.5 缺少 LayerWise 约束 ${term}`);
  }
});

test('parameter AG 的三个真实 dispatch owner 与 consumer wait 可追踪', async () => {
  const page = await readFile(pagePath, 'utf8');
  const lifecycle = await readFile(join(assetDir, 'megatron_distopt_rs_update_ag.svg'), 'utf8');

  for (const term of [
    'align_param_gather=False',
    'finish_param_sync()',
    '懒 dispatch 当前 bucket',
    'align_param_gather=True',
    'param_sync_func',
    'ChainedOptimizer._step',
    'start_param_sync(force_dispatch=True)',
    '不 dispatch parameter AG',
  ]) {
    assert.ok(page.includes(term), `正文缺少 parameter AG 生命周期语义 ${term}`);
  }

  for (const term of [
    'UNALIGNED OVERLAP',
    'finish_param_sync',
    'lazy dispatch current AG',
    'ALIGNED OVERLAP',
    'param_sync_func',
    'OPTIMIZER-STEP OVERLAP',
    'force_dispatch=True',
    'zero_grad_buffer / optimizer.zero_grad 只清 gradient',
  ]) {
    assert.ok(lifecycle.includes(term), `生命周期图缺少 ${term}`);
  }

  assert.doesNotMatch(
    lifecycle,
    /zero_grad \/ first bucket/,
    'zero_grad 不得再被画成首个 parameter AG dispatch owner',
  );
});

test('HSDP stream 属于 dense/expert collection，forced AR 只全局化 local shard', async () => {
  const page = await readFile(pagePath, 'utf8');
  const planes = await readFile(join(assetDir, 'megatron_distopt_fp32_rs_hsdp.svg'), 'utf8');

  for (const term of [
    'dense `bucket_groups` 集合创建一条 stream',
    '`expert_parallel_bucket_groups` 集合另创建一条',
    '不是每个 bucket group 独占一条 stream',
    '所有 DP rank 都不持有全局 full gradient',
    'force_all_reduce+k>1 bytes=48+8+24=80 B',
  ]) {
    assert.ok(page.includes(term), `正文缺少 HSDP/forced-AR 约束 ${term}`);
  }

  for (const term of [
    'dense bucket-group collection: 1 shared stream',
    'expert bucket-group collection: another shared stream',
    'not one stream per bucket group',
    'full gradient 只在 intra-instance group 内成立',
    '跨 instance 只 all-reduce local shard',
  ]) {
    assert.ok(planes.includes(term), `四-lane 图缺少 HSDP/forced-AR 约束 ${term}`);
  }
  assert.doesNotMatch(planes, /each bucket group[^<]{0,40}dedicated/i);
});

test('Markdown 覆盖选择轴、硬约束、finalizer 和 owner 链接', async () => {
  const page = await readFile(pagePath, 'utf8');
  for (const term of [
    'use_torch_fsdp2',
    'get_megatron_optimizer',
    'LayerWiseDistributedOptimizer',
    'NonuniformTPDistributedDataParallel',
    'reduce_scatter_with_fp32_accumulation',
    'all_to_all_single',
    'num_distributed_optimizer_instances',
    'average_in_collective',
    'param_name_patterns_for_fp32_local_accumulation',
    'pad_buckets_for_high_nccl_busbw',
    'bucket_size',
    'num_buckets',
    'param_sync_func',
  ]) {
    assert.ok(page.includes(term), `正文缺少活跃选择/配置 ${term}`);
  }

  for (const finalizerTerm of [
    '_allreduce_conditional_embedding_grads',
    '_allreduce_router_grads',
    '_allreduce_non_tensor_model_parallel_grads',
    '_allreduce_word_embedding_grads',
    '_allreduce_position_embedding_grads',
    '_update_router_expert_bias',
    'reset_model_temporary_tensors',
    'num_tokens is not None',
  ]) {
    assert.ok(page.includes(finalizerTerm), `finalize_model_grads 闭环缺少 ${finalizerTerm}`);
  }

  for (const owner of [
    '[[22_megatron_memory_optimization_analysis]]',
    '[[23_megatron_precision_cudagraph_fusion_analysis]]',
    '[[25_megatron_nonuniform_tp_analysis]]',
    '[[26_megatron_optimizer_step_internals_deepdive]]',
    '[[36_megatron_fsdp_analysis]]',
  ]) {
    assert.ok(page.includes(owner), `正文缺少 sibling owner ${owner}`);
  }

  for (const asset of assetNames) assert.ok(page.includes(`assets/${asset}`), `正文未嵌入 ${asset}`);

  // 保留现有页面入口；最新特性契约的顺序另检查最小实例先于选型展开。
  for (const heading of [
    '## 1. 特性概览',
    '## 2. 分布式优化器详细方案',
    '## 3. 代码实现分析',
    '## 4. 配套机制',
    '## 5. 约束、适用场景与趋势',
    '## 6. 配置契约',
    '## Related Pages',
  ]) {
    assert.ok(page.includes(heading), `页面缺少房子形状的一节 ${heading}`);
  }
  assert.ok(page.includes('```mermaid'), '缺少类与所有权图');
  assert.ok(page.indexOf('### 2.1 最小示例') < page.indexOf('### 2.2 从真实选型点'), '应从最小range实例再展开系统选型');
  // 覆盖清单指派给本页的唯一配置字段，必须留在配置契约里。
  assert.ok(page.includes('`param_sync_func`'), '配置契约缺少 param_sync_func');
  // 面向流程的内容守恒记账归 changelog，不再作为页面内容。
  assert.doesNotMatch(
    page,
    /kept \+ deepened|corrected \+ deepened|相对旧 HEAD 的内容守恒/,
    '内容守恒记账属于 changelog，不应回到页面',
  );
});

test('图与正文都区分等分原语、真实布局和仅通信的padding', async () => {
  const page = await readFile(pagePath, 'utf8');
  assert.ok(page.includes('不是当前 native layout builder'), '正文须声明N=16的抽象层级');
  assert.ok(page.includes('padding 会参加 collective，不获得 main/state 或更新权'), '正文不得让padding参与optimizer');
  for (const name of assetNames) {
    const svg = await readFile(join(assetDir, name), 'utf8');
    assert.match(svg, /等分原语示例/);
    assert.match(svg, /64\/128对齐/);
  }
  const planes = await readFile(join(assetDir, 'megatron_distopt_fp32_rs_hsdp.svg'), 'utf8');
  assert.ok(planes.includes('overlap: one bucket / bucket group'), '单bucket约束须限定async/overlap');
  const livePaths = await readFile(join(assetDir, 'megatron_distopt_live_paths.svg'), 'utf8');
  assert.ok(livePaths.includes('依赖API契约对照，未核验PyTorch内核'), '依赖边界必须在图中可见');
});

test('真实浏览器渲染：四图文字不裁切且包围盒不重叠', async (t) => {
  const executablePath = findBrowserExecutable();
  const root = resolve(here, '../../../..');
  const puppeteerEntry = [
    join(root, 'tools', 'mkdocs-site', 'node_modules', 'puppeteer-core', 'lib', 'esm', 'puppeteer', 'puppeteer-core.js'),
    join(root, 'tools', 'html2md', 'node_modules', 'puppeteer-core', 'lib', 'esm', 'puppeteer', 'puppeteer-core.js'),
  ].find((candidate) => existsSync(candidate));
  if (!executablePath || !puppeteerEntry) {
    t.skip('本机没有可用 Chromium/puppeteer-core');
    return;
  }

  const { default: puppeteer } = await import(pathToFileURL(puppeteerEntry).href);
  const browser = await puppeteer.launch({
    executablePath, headless: true, args: ['--disable-gpu', '--no-sandbox'],
  });
  try {
    for (const name of assetNames) {
      const svg = await readFile(join(assetDir, name), 'utf8');
      const browserPage = await browser.newPage();
      await browserPage.setContent(svg);
      const audit = await browserPage.evaluate(() => {
        const rootSvg = document.querySelector('svg');
        const view = rootSvg.viewBox.baseVal;
        const boxes = [...rootSvg.querySelectorAll('text')].map((element) => {
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

// Git checkouts may use CRLF; compare SVG content with only line endings normalized.
function normalizeEol(text) { return text.replace(/\r\n/g, '\n'); }
