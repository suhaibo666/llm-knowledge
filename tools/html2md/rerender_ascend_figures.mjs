import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import puppeteer from 'puppeteer-core';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const source = path.join(root, 'raw', '02_engineering', '05_gpu_kernel', 'ascend_kernels.html');
const assets = path.join(root, 'wiki', '02_engineering', '05_gpu_kernel', 'assets');
const browser = await puppeteer.launch({
  executablePath: 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
  headless: 'new',
  args: ['--no-sandbox', '--force-color-profile=srgb'],
});
const page = await browser.newPage();
await page.setViewport({ width: 1400, height: 1600, deviceScaleFactor: 2 });
await page.goto(pathToFileURL(source).href, { waitUntil: 'networkidle0', timeout: 60000 }).catch(() => {});
await page.addStyleTag({ content: '*,*::before,*::after{animation:none!important;transition:none!important}' });
await page.evaluate(async () => { try { await document.fonts.ready; } catch {} });

await page.evaluate(() => {
  const svgs = document.querySelectorAll('svg');
  svgs[3].querySelector('text').setAttribute('y', '16');
  svgs[4].setAttribute('viewBox', '0 0 1100 300');
});

for (const index of [3, 4]) {
  const svg = (await page.$$('svg'))[index];
  await svg.screenshot({ path: path.join(assets, `ascend_kernel_execution_model_analysis_fig${index + 1}.png`) });
}

await page.close();
await browser.close();
