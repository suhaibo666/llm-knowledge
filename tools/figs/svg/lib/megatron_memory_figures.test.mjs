// 锁住 22 号页两张图的可执行契约：offload 时序由 simulateOffload() 复刻
// fine_grained_activation_offload.py 的组提交 / 预取规则算出，paged stash 的页数与去向由
// solveStash() 复刻 paged_stash.py 的预定公式与 ops/paged_stash.py 的复制判定算出；
// 全部与 22_megatron_memory_optimization_analysis.md 正文引用的数值逐个对齐。
//
// 运行：node --test tools/figs/svg/lib/megatron_memory_figures.test.mjs

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFile, mkdir } from 'node:fs/promises';
import { existsSync, readdirSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { resolve, join, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { OFFLOAD_CFG, STASH_CFG, simulateOffload, solveStash, buildFigures } from '../megatron_memory_figures.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../../../..');
const pageDir = join(root, 'wiki/02_engineering/02_train_frameworks/megatron-lm');
const pagePath = join(pageDir, '22_megatron_memory_optimization_analysis.md');
const figures = buildFigures();
const normalizeEol = (s) => s.replace(/\r\n/g, '\n');

test('offload 仿真：margin = 组名去重数，预取距离等于 margin，本例零阻塞', () => {
  const s = simulateOffload();
  assert.equal(s.margin, OFFLOAD_CFG.names.length);
  const kept = s.groups.filter((g) => !g.offload).map((g) => g.id);
  assert.deepEqual(kept, ['L3.core_attn', 'L3.expert_fc1'], '每个名字前向顺序里最后一组留在 GPU');
  assert.equal(s.stall, 0);
  assert.equal(s.exposedIfNoMargin, (96 + 128) / OFFLOAD_CFG.pcieMBPerUnit);
  assert.equal(s.peakNoOffloadMB, 3 * (96 + 128));
  assert.ok(s.peakMB < s.peakNoOffloadMB);
  // 反向里每个被 offload 的组，其 reload 由后一个（前向序）组的 start 反向触发
  for (const g of s.groups.filter((x) => x.offload)) assert.ok(g.h2d && g.h2d[1] <= g.bwd[0], `${g.id} 的 reload 未在反向前完成`);
  // margin=0 时最后两组无人预取：对照值就是它们的拷贝时长之和
  assert.equal(s.exposedIfNoMargin, 14);
});

test('paged stash 解算：预定页数、步 A 命中、步 B 溢写或 overflow', () => {
  const t = solveStash();
  assert.equal(t.avgTokens, Math.floor(STASH_CFG.maxNumTokens / STASH_CFG.capacityFactor));
  assert.equal(t.cudaTokens, Math.floor(t.peakAvg * STASH_CFG.cudaFactor));
  assert.equal(t.cudaPages, Math.ceil(t.cudaTokens / STASH_CFG.pageSize));
  assert.equal(t.stepA.overflow, 0);
  assert.ok(t.stepA.totalNeed <= t.cudaPages);
  assert.equal(t.stepBNoHost.overflow, 1, 'cpu factor 0 时偏斜步必须 overflow');
  assert.equal(t.stepBHost.overflow, 0);
  assert.equal(t.stepBHost.hostSpill, 1, '有宿主页时偏斜步只溢写');
  assert.equal(t.maxTries, 2);
});

test('正文引用的数值与图同源，且图文件与生成器一致', async () => {
  const p = normalizeEol(await readFile(pagePath, 'utf8'));
  const s = simulateOffload();
  const t = solveStash();
  for (const term of [
    `margin=${s.margin}`,
    `峰值从 ${s.peakNoOffloadMB} MB 降到 ${s.peakMB} MB`,
    `暴露 ${s.exposedIfNoMargin} 个单位`,
    `avg=${t.avgTokens}`,
    `${t.peakAvg} 行`,
    `${t.cudaTokens} 行`,
    `${t.cudaPages} 页`,
    `[${STASH_CFG.stepA.join(', ')}]`,
    `[${STASH_CFG.stepB.join(', ')}]`,
    `${t.stepBNoHost.totalNeed} 页`,
    `${t.hostPages} 个宿主页`,
    `num_tries < ${t.maxTries}`,
  ]) assert.ok(p.includes(term), `正文与求解结果漂移: ${term}`);
  for (const term of ['PipelineOffloadManager', 'ChunkOffloadHandler', 'post_warmup_callback', 'saved_tensors_hooks', 'PagedStashBuffer', 'paged_stash_copy_kernel', 'PagedStashRunner', 'prepare_for_rerun', 'ChunkedOptimizerStateOffloader', 'ncclMemAlloc', 'get_cpu_offload_context']) {
    assert.ok(p.includes(term), `关键契约缺失: ${term}`);
  }
  for (const [name, expected] of Object.entries(figures)) {
    assert.ok(p.includes(`assets/${name}`), `正文未引用 ${name}`);
    assert.equal(normalizeEol(await readFile(join(pageDir, 'assets', name), 'utf8')), normalizeEol(expected), `图/生成器漂移 ${name}`);
  }
});

test('真实浏览器渲染：文字不越界，并留下截图', async () => {
  const cache = join(homedir(), '.cache/puppeteer/chrome');
  const candidates = [process.env.PUPPETEER_EXECUTABLE_PATH];
  if (existsSync(cache)) for (const x of readdirSync(cache).sort().reverse()) candidates.push(join(cache, x, 'chrome-win64/chrome.exe'));
  candidates.push('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', '/usr/bin/chromium');
  const executablePath = candidates.find((c) => c && existsSync(c));
  assert.ok(executablePath, '需要真实浏览器，禁止静默跳过渲染');
  const { default: puppeteer } = await import(pathToFileURL(join(root, 'tools/mkdocs-site/node_modules/puppeteer-core/lib/esm/puppeteer/puppeteer-core.js')));
  const browser = await puppeteer.launch({ executablePath, headless: true, args: ['--disable-gpu', '--no-sandbox'] });
  const out = join(tmpdir(), 'megatron-memory-figures-render');
  await mkdir(out, { recursive: true });
  try {
    const page = await browser.newPage();
    for (const name of Object.keys(figures)) {
      await page.goto(pathToFileURL(join(pageDir, 'assets', name)).href);
      const issues = await page.evaluate(() => {
        const svg = document.querySelector('svg');
        const view = svg.viewBox.baseVal;
        const problems = [];
        for (const t of svg.querySelectorAll('text')) {
          const b = t.getBBox();
          if (b.x < 0 || b.y < 0 || b.x + b.width > view.width + 1 || b.y + b.height > view.height + 1) problems.push(`越界: ${t.textContent}`);
        }
        return problems;
      });
      assert.deepEqual(issues, [], `${name}: ${issues.join(' | ')}`);
      await page.setViewport({ width: 1240, height: 1400 });
      await page.screenshot({ path: join(out, name.replace('.svg', '.png')), fullPage: true });
    }
  } finally {
    await browser.close();
  }
});
