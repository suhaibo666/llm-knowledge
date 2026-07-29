// Render each *.html in this dir to a 2x PNG by element-screenshotting its <figure>.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import puppeteer from 'puppeteer-core';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';

const files = process.argv.slice(2);
if (files.length === 0) {
  for (const f of fs.readdirSync(__dirname)) if (f.endsWith('.html')) files.push(path.join(__dirname, f));
}

const browser = await puppeteer.launch({
  executablePath: EDGE,
  headless: true,
  args: ['--no-sandbox', '--force-color-profile=srgb', '--hide-scrollbars'],
});
const page = await browser.newPage();
await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'light' }]);
await page.setViewport({ width: 1100, height: 800, deviceScaleFactor: 2 });

for (const f of files) {
  const abs = path.resolve(f);
  await page.goto(pathToFileURL(abs).href, { waitUntil: 'networkidle0' });
  const el = (await page.$('figure')) || (await page.$('svg'));
  const out = abs.replace(/\.html$/, '.png');
  await el.screenshot({ path: out });
  console.log('wrote', out);
}
await browser.close();
