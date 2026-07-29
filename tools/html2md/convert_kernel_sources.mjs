import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import puppeteer from 'puppeteer-core';
import TurndownService from 'turndown';
import { gfm } from 'turndown-plugin-gfm';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const rawDir = path.join(root, 'raw', '02_engineering', '05_gpu_kernel');
const wikiDir = path.join(root, 'wiki', '02_engineering', '05_gpu_kernel');
const assetsDir = path.join(wikiDir, 'assets');
const edge = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';

const files = [
  { source: 'cuda_gemm_final.html', output: 'cuda_gemm_kernel_analysis.md' },
  { source: 'cuda_nonmatmul_kernels_final.html', output: 'cuda_nonmatmul_kernels_analysis.md' },
  { source: 'ascend_kernels.html', output: 'ascend_kernel_execution_model_analysis.md' },
];

const td = new TurndownService({
  headingStyle: 'atx',
  hr: '---',
  bulletListMarker: '-',
  codeBlockStyle: 'fenced',
  fence: '```',
  emDelimiter: '*',
  strongDelimiter: '**',
});
td.use(gfm);
td.addRule('kernel-figure', {
  filter: (node) => node.nodeName === 'IMG' && node.getAttribute('data-kernel-figure'),
  replacement: (_content, node) => `\n\n![${node.getAttribute('alt')}](${node.getAttribute('src')})\n\n`,
});

const browser = await puppeteer.launch({
  executablePath: edge,
  headless: 'new',
  args: ['--no-sandbox', '--force-color-profile=srgb'],
});

fs.mkdirSync(assetsDir, { recursive: true });

for (const cfg of files) {
  const sourcePath = path.join(rawDir, cfg.source);
  const outputPath = path.join(wikiDir, cfg.output);
  const stem = path.basename(cfg.output, '.md');
  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 1600, deviceScaleFactor: 2 });
  await page.goto(pathToFileURL(sourcePath).href, { waitUntil: 'networkidle0', timeout: 60000 }).catch(() => {});
  await page.addStyleTag({ content: '*,*::before,*::after{animation:none!important;transition:none!important}' });
  await page.evaluate(async () => { try { await document.fonts.ready; } catch {} });

  const figures = await page.evaluate(() => Array.from(document.querySelectorAll('svg')).map((svg, index) => {
    svg.setAttribute('data-kernel-figure-index', String(index));
    const label = svg.getAttribute('aria-label') || svg.querySelector('text')?.textContent || `diagram ${index + 1}`;
    return { index, label: label.replace(/\s+/g, ' ').trim().slice(0, 120) };
  }));

  for (const figure of figures) {
    const name = `${stem}_fig${figure.index + 1}.png`;
    const element = await page.$(`[data-kernel-figure-index="${figure.index}"]`);
    await element.screenshot({ path: path.join(assetsDir, name) });
    figure.src = `assets/${name}`;
  }

  const bodyHtml = await page.evaluate((figureMeta) => {
    for (const figure of figureMeta) {
      const svg = document.querySelector(`[data-kernel-figure-index="${figure.index}"]`);
      const img = document.createElement('img');
      img.setAttribute('data-kernel-figure', '1');
      img.setAttribute('src', figure.src);
      img.setAttribute('alt', figure.label || `diagram ${figure.index + 1}`);
      svg.replaceWith(img);
    }
    document.querySelectorAll('script,noscript,style,.section-tag').forEach((node) => node.remove());
    document.querySelectorAll('pre').forEach((pre) => {
      const code = document.createElement('code');
      code.textContent = pre.textContent;
      pre.replaceChildren(code);
    });
    document.querySelectorAll('.callout,.lead,.formula,.info,.tip,.warn,.note').forEach((node) => {
      const quote = document.createElement('blockquote');
      quote.innerHTML = node.innerHTML;
      node.replaceWith(quote);
    });
    document.querySelectorAll('.divider').forEach((node) => node.replaceWith(document.createElement('hr')));
    document.querySelectorAll('span.sn,span.num').forEach((span) => {
      span.textContent = `${span.textContent.trim()} `;
    });
    return document.body.innerHTML;
  }, figures);

  let markdown = td.turndown(bodyHtml)
    .replace(/\\_/g, '_')
    .replace(/\n[ \t]+\n/g, '\n\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim() + '\n';
  fs.writeFileSync(outputPath, markdown, 'utf8');
  console.log(`${cfg.source} -> ${cfg.output}: ${figures.length} figures, ${Buffer.byteLength(markdown)} bytes`);
  await page.close();
}

await browser.close();
