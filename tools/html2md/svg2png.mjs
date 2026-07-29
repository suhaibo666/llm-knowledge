// One-off: rasterize standalone .svg files to .png via headless Edge at 2x,
// on a white background (matches the house PNG convention). Keeps the .svg as source.
// Usage: node svg2png.mjs <file1.svg> <file2.svg> ...
import fs from 'node:fs';
import path from 'node:path';
import puppeteer from 'puppeteer-core';

const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const files = process.argv.slice(2);
if (!files.length) { console.error('usage: node svg2png.mjs <file.svg> ...'); process.exit(1); }

const browser = await puppeteer.launch({
  executablePath: EDGE, headless: 'new',
  args: ['--no-sandbox', '--force-color-profile=srgb'],
});
for (const f of files) {
  const abs = path.resolve(f);
  let svg = fs.readFileSync(abs, 'utf8');
  svg = svg.slice(svg.indexOf('<svg')); // strip xml decl / doctype / comments before root
  const page = await browser.newPage();
  await page.setViewport({ width: 2200, height: 1600, deviceScaleFactor: 2 });
  await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'light' }]);
  await page.setContent(
    `<!doctype html><html><head><meta charset="utf-8">
     <style>html,body{margin:0;padding:0;background:#fff}#wrap{display:inline-block;background:#fff;padding:8px}svg{display:block}</style>
     </head><body><div id="wrap">${svg}</div></body></html>`,
    { waitUntil: 'networkidle0', timeout: 60000 }
  );
  await page.evaluate(async () => { try { await document.fonts.ready; } catch (e) {} });
  await new Promise((r) => setTimeout(r, 300));
  const el = (await page.$('#wrap')) || (await page.$('svg'));
  const out = abs.replace(/\.svg$/i, '.png');
  await el.screenshot({ path: out });
  console.log('OK', path.basename(out));
  await page.close();
}
await browser.close();
console.log('DONE');
