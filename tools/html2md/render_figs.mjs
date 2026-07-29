// Standalone flowchart renderer for the GLM-5 deep-dive wiki pages.
// Reuses the same headless-Edge + 2x screenshot approach as convert.mjs, but for
// hand-authored HTML/CSS+SVG flowcharts instead of full HTML reports.
//
// Input : every *.html in .html2md/figs/  (each contains one or more elements
//         with class "diagram" carrying a data-name="<basename_figN>" attribute)
// Output: default wiki/01_theory/01_models/zhipu_glm/assets/<data-name>.png (deviceScaleFactor 2)
//         override with env FIGS_OUT=<dir> (absolute, or relative to repo root)
//
// Usage : node render_figs.mjs            # render all figs/*.html
//         node render_figs.mjs arch data  # only figs whose filename contains "arch" or "data"
//         FIGS_OUT=wiki/01_theory/06_distributed_parallelism/assets node render_figs.mjs dp_
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import puppeteer from 'puppeteer-core';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const FIGS_DIR = path.resolve(__dirname, '..', 'figs');
const REPO_ROOT = path.resolve(__dirname, '..', '..');
const DEFAULT_ASSETS = path.resolve(REPO_ROOT, 'wiki', '01_theory', '01_models', 'zhipu_glm', 'assets');
const ASSETS = process.env.FIGS_OUT
  ? (path.isAbsolute(process.env.FIGS_OUT) ? process.env.FIGS_OUT : path.resolve(REPO_ROOT, process.env.FIGS_OUT))
  : DEFAULT_ASSETS;

const REVEAL_FIX = `*,*::before,*::after{animation:none!important;transition:none!important}`;

async function main() {
  const filter = process.argv.slice(2);
  fs.mkdirSync(ASSETS, { recursive: true });
  if (!fs.existsSync(FIGS_DIR)) { console.error('no figs dir:', FIGS_DIR); process.exit(1); }
  let files = fs.readdirSync(FIGS_DIR).filter((f) => f.endsWith('.html'));
  if (filter.length) files = files.filter((f) => filter.some((k) => f.includes(k)));
  if (!files.length) { console.error('no matching html in', FIGS_DIR); process.exit(1); }

  const browser = await puppeteer.launch({
    executablePath: EDGE, headless: 'new',
    args: ['--no-sandbox', '--force-color-profile=srgb'],
  });
  let count = 0;
  for (const f of files) {
    const abs = path.join(FIGS_DIR, f);
    const page = await browser.newPage();
    await page.setViewport({ width: 1500, height: 2200, deviceScaleFactor: 2 });
    await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'light' }]);
    await page.goto(pathToFileURL(abs).href, { waitUntil: 'networkidle0', timeout: 60000 }).catch(() => {});
    await page.addStyleTag({ content: REVEAL_FIX });
    await page.evaluate(async () => { try { await document.fonts.ready; } catch (e) {} });
    await new Promise((r) => setTimeout(r, 300));

    const names = await page.evaluate(() =>
      Array.from(document.querySelectorAll('.diagram')).map((el) => el.getAttribute('data-name')));
    for (const name of names) {
      if (!name) { console.warn('  .diagram without data-name in', f); continue; }
      const el = await page.$(`.diagram[data-name="${name}"]`);
      if (!el) continue;
      const out = path.join(ASSETS, `${name}.png`);
      await el.screenshot({ path: out });
      console.log('OK ', name + '.png');
      count++;
    }
    await page.close();
  }
  await browser.close();
  console.log(`\nDONE ${count} figures -> ${ASSETS}`);
}
main().catch((e) => { console.error(e); process.exit(1); });
