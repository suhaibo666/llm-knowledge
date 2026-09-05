// 锁住 21 号页融合阶梯图的可执行契约：图上每个字节数、kernel 数都由 solveExample() 从
// T/H/dtype 算出，并与 21_megatron_fusion_operators_analysis.md 正文引用的数值逐个对齐；
// 只改正文、不改图，本测试必须红。
//
// 运行：node --test tools/figs/svg/lib/megatron_fusion_figures.test.mjs

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFile, mkdir } from 'node:fs/promises';
import { existsSync, readdirSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { resolve, join, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { solveExample, buildFigures } from '../megatron_fusion_figures.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../../../..');
const pageDir = join(root, 'wiki/02_engineering/02_train_frameworks/megatron-lm');
const pagePath = join(pageDir, '21_megatron_fusion_operators_analysis.md');
const figures = buildFigures();
const normalizeEol = (s) => s.replace(/\r\n/g, '\n');

test('算例字节数由 T/H/dtype 推出，且各级之间的差额自洽', () => {
  const e = solveExample();
  assert.equal(e.y, e.T * e.twoH * e.bf16);
  assert.equal(e.half, e.y / 2);
  assert.equal(e.eager.kernels, 3);
  assert.equal(e.eager.read, e.y + e.bias + 3 * e.half);
  assert.equal(e.eager.write, e.y + 2 * e.half);
  assert.equal(e.fused.regions, 1);
  assert.equal(e.fused.savedForBackward, e.y + e.bias);
  assert.equal(e.fp8Store.savedForBackward, e.y / 2 + e.bias, 'fp8 存储把 input 字节减半');
  assert.equal(e.weighted.savedForBackward - e.fused.savedForBackward, e.weights);
  assert.equal(e.crossEntropy.unfusedAllReduce - e.crossEntropy.fusedAllReduce, 1);
});

test('正文引用的数值与图同源，且图文件与生成器一致', async () => {
  const p = normalizeEol(await readFile(pagePath, 'utf8'));
  const e = solveExample();
  for (const term of [
    `y[T=${e.T}, 2H=${e.twoH}]`,
    `${e.eager.kernels} 个逐算子 kernel`,
    `读 ${e.eager.read} B、写 ${e.eager.write} B`,
    `物化 ${e.eager.materialized} B`,
    `读 ${e.fused.read} B、写 ${e.fused.write} B`,
    `保存 ${e.fused.savedForBackward} B`,
    `降到 ${e.fp8Store.savedForBackward} B`,
    `多保存 ${e.weighted.extraSaved} B`,
    `${e.crossEntropy.unfusedAllReduce} 次 all-reduce`,
    `${e.crossEntropy.fusedAllReduce} 次`,
  ]) assert.ok(p.includes(term), `正文与求解结果漂移: ${term}`);
  for (const term of ['jit_fuser', 'BiasGeGLUFunction', 'fp8_input_store', 'weighted_bias_swiglu_impl', '_is_fused_impl_supported', 'GroupedTensor', 'fused_vocab_parallel_cross_entropy', 'MHC_FORCE_BACKEND', 'dsa_kernel_backend', 'disable_jit_fuser']) {
    assert.ok(p.includes(term), `关键契约缺失: ${term}`);
  }
  for (const [name, expected] of Object.entries(figures)) {
    assert.ok(p.includes(`assets/${name}`), `正文未引用 ${name}`);
    assert.equal(normalizeEol(await readFile(join(pageDir, 'assets', name), 'utf8')), normalizeEol(expected), `图/生成器漂移 ${name}`);
  }
});

test('真实浏览器渲染：文字包围盒不越出所在方框，并留下截图', async () => {
  const cache = join(homedir(), '.cache/puppeteer/chrome');
  const candidates = [process.env.PUPPETEER_EXECUTABLE_PATH];
  if (existsSync(cache)) for (const x of readdirSync(cache).sort().reverse()) candidates.push(join(cache, x, 'chrome-win64/chrome.exe'));
  candidates.push('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', '/usr/bin/chromium');
  const executablePath = candidates.find((c) => c && existsSync(c));
  assert.ok(executablePath, '需要真实浏览器，禁止静默跳过渲染');
  const { default: puppeteer } = await import(pathToFileURL(join(root, 'tools/mkdocs-site/node_modules/puppeteer-core/lib/esm/puppeteer/puppeteer-core.js')));
  const browser = await puppeteer.launch({ executablePath, headless: true, args: ['--disable-gpu', '--no-sandbox'] });
  const out = join(tmpdir(), 'megatron-fusion-figures-render');
  await mkdir(out, { recursive: true });
  try {
    const page = await browser.newPage();
    for (const name of Object.keys(figures)) {
      await page.goto(pathToFileURL(join(pageDir, 'assets', name)).href);
      const issues = await page.evaluate(() => {
        const svg = document.querySelector('svg');
        const view = svg.viewBox.baseVal;
        const problems = [];
        const rects = [...svg.querySelectorAll('rect')].map((r) => r.getBBox());
        for (const t of svg.querySelectorAll('text')) {
          const b = t.getBBox();
          if (b.x < 0 || b.y < 0 || b.x + b.width > view.width + 1 || b.y + b.height > view.height + 1) problems.push(`越界: ${t.textContent}`);
          const inside = rects.some((r) => b.x >= r.x - 1 && b.y >= r.y - 1 && b.x + b.width <= r.x + r.width + 1 && b.y + b.height <= r.y + r.height + 1);
          const isFree = ['ti', 'su', 'cap', 'rank', 'pt'].includes(t.getAttribute('class'));
          if (!inside && !isFree) problems.push(`文字溢出方框: ${t.textContent}`);
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
