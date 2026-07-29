// HTML -> Markdown converter for the llm-knowledge wiki.
// Renders each HTML page in headless Edge, rasterizes every inline-SVG / CSS
// diagram to a PNG (preserving exact CSS-variable colors + fonts), then converts
// the remaining body to clean Markdown via Turndown + GFM.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import puppeteer from 'puppeteer-core';
import TurndownService from 'turndown';
import { gfm } from 'turndown-plugin-gfm';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const WIKI = path.resolve(__dirname, '..', '..', 'wiki');
const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';

// figureSelector: elements to rasterize.  captionSel: caption element WITHIN a figure (hidden before
// shot, used as the visible caption + alt).  If null, no caption.
const TF = '02_engineering/02_train_frameworks';
const FILES = [
  { f: `${TF}/async_collective_tensor_deep_dive.html`,        figureSelector: 'div.diagram', captionSel: '.diagram-cap' },
  { f: `${TF}/comm_compute_overlap_analysis.html`,            figureSelector: 'div.diagram', captionSel: '.diagram-cap' },
  { f: `${TF}/deepseek_v4_context_parallel_analysis.html`,    figureSelector: 'div.diagram', captionSel: '.diagram-cap' },
  { f: `${TF}/deepseek_v4_tensor_parallel_analysis.html`,     figureSelector: 'div.diagram', captionSel: '.diagram-cap' },
  { f: `${TF}/distributed_optimizer_deep_dive.html`,          figureSelector: 'div.diagram', captionSel: '.diagram-cap' },
  { f: `${TF}/megatron_pp_parallelism_analysis.html`,         figureSelector: 'div.diagram', captionSel: '.diagram-cap' },
  { f: `${TF}/mindformers_moe_token_dispatcher_analysis.html`, figureSelector: 'figure',     captionSel: 'figcaption' },
  { f: `${TF}/muon_sharded_hsdp_report.html`,                 figureSelector: 'div.figure',  captionSel: '.figure-caption' },
  { f: `02_engineering/05_gpu_kernel/gpu_kernel_guide.html`,  figureSelector: '.tier-diagram, .fa-flow', captionSel: null },
];

const REVEAL_FIX = `
*,*::before,*::after{animation:none!important;transition:none!important}
[class*="reveal"]{opacity:1!important;transform:none!important;filter:none!important;visibility:visible!important}
.reveal{opacity:1!important;transform:none!important}
`;

function buildTurndown() {
  const td = new TurndownService({
    headingStyle: 'atx', hr: '---', bulletListMarker: '-',
    codeBlockStyle: 'fenced', fence: '```', emDelimiter: '*',
    strongDelimiter: '**', linkStyle: 'inlined',
  });
  td.use(gfm);
  // Rasterized figures: emit image + italic caption line.
  td.addRule('figure', {
    filter: (node) => node.nodeName === 'IMG' && node.getAttribute('data-fig'),
    replacement: (_c, node) => {
      const src = node.getAttribute('src');
      const cap = node.getAttribute('data-cap') || '';
      const alt = node.getAttribute('alt') || cap || 'diagram';
      return `\n\n![${alt}](${src})\n\n` + (cap ? `*${cap}*\n\n` : '');
    },
  });
  return td;
}

// Runs in the browser. Marks figures + hides their captions; returns caption metadata.
function prepFigures(figureSelector, captionSel) {
  const figs = Array.from(document.querySelectorAll(figureSelector));
  return figs.map((el, i) => {
    el.setAttribute('data-figindex', String(i));
    let caption = '';
    if (captionSel) {
      const c = el.querySelector(captionSel);
      if (c) { caption = c.textContent.replace(/\s+/g, ' ').trim(); c.style.display = 'none'; }
    }
    return { index: i, caption };
  });
}

// Runs in the browser. Replaces figures with <img> placeholders and normalizes the DOM, then
// returns the cleaned body HTML.
function cleanDom(metas) {
  // 1. figures -> <img data-fig>
  for (const m of metas) {
    const el = document.querySelector(`[data-figindex="${m.index}"]`);
    if (!el) continue;
    const img = document.createElement('img');
    img.setAttribute('data-fig', '1');
    img.setAttribute('src', m.src);
    img.setAttribute('alt', m.caption || 'diagram');
    if (m.caption) img.setAttribute('data-cap', m.caption);
    el.replaceWith(img);
  }
  // 2. drop scripts + redundant section tags
  document.querySelectorAll('script,noscript').forEach((n) => n.remove());
  document.querySelectorAll('.section-tag').forEach((n) => n.remove());
  // 2b. normalize every <pre> to <pre><code>plain text</code> so Turndown fences it.
  // Some pages use bare <pre> with syntax-highlight <span>s (no <code>) -> would leak as escaped prose.
  document.querySelectorAll('pre').forEach((pre) => {
    const text = pre.textContent;
    pre.innerHTML = '';
    const code = document.createElement('code');
    code.textContent = text;
    pre.appendChild(code);
  });
  // 2c. pre-formatted leaf <div>s (e.g. .math-block with white-space:pre, no real <pre>) -> fenced code
  document.querySelectorAll('div,section,p').forEach((el) => {
    if (el.querySelector('pre,div,section,table,ul,ol,img')) return; // leaf text containers only
    const ws = (getComputedStyle(el).whiteSpace || '');
    const isPre = ws.startsWith('pre') || el.classList.contains('math-block');
    if (!isPre) return;
    const text = el.textContent;
    if (!text.includes('\n')) return;
    const pre = document.createElement('pre');
    const code = document.createElement('code');
    code.textContent = text;
    pre.appendChild(code);
    el.replaceWith(pre);
  });
  // 3. callouts / leads / formulas -> blockquote
  document.querySelectorAll('.callout,.lead,.formula,.info,.tip,.warn,.note').forEach((n) => {
    const bq = document.createElement('blockquote');
    bq.innerHTML = n.innerHTML;
    n.replaceWith(bq);
  });
  // 4. dividers -> hr
  document.querySelectorAll('.divider').forEach((n) => n.replaceWith(document.createElement('hr')));
  // 5. TOC -> plain nested list (internal #anchors don't survive to MD)
  document.querySelectorAll('.toc').forEach((toc) => {
    const anchors = Array.from(toc.querySelectorAll('a'));
    if (!anchors.length) { toc.remove(); return; }
    const wrap = document.createElement('div');
    const title = document.createElement('p');
    title.innerHTML = '<strong>目录</strong>';
    wrap.appendChild(title);
    const ul = document.createElement('ul');
    let lastLi = null;
    for (const a of anchors) {
      let txt = a.textContent.replace(/\s+/g, ' ').trim();
      txt = txt.replace(/^(\d{1,2})(?=[^\d\s])/, '$1 '); // "00问题" -> "00 问题"
      if (!txt) continue;
      const li = document.createElement('li');
      li.textContent = txt;
      if (a.classList.contains('sub') && lastLi) {
        let sub = lastLi.querySelector('ul');
        if (!sub) { sub = document.createElement('ul'); lastLi.appendChild(sub); }
        sub.appendChild(li);
      } else { ul.appendChild(li); lastLi = li; }
    }
    wrap.appendChild(ul);
    toc.replaceWith(wrap);
  });
  // 6. heading number spans get a trailing space; h1 subtitle span -> emphasized line
  document.querySelectorAll('span.sn,span.num').forEach((s) => { s.textContent = s.textContent.trim() + ' '; });
  document.querySelectorAll('h1').forEach((h1) => {
    let sub = '';
    const span = h1.querySelector('span');
    if (span) { sub = span.textContent.replace(/\s+/g, ' ').trim(); span.remove(); }
    const br = h1.querySelector('br');
    if (br) { // text after <br> is a subtitle too
      let after = '', n = br.nextSibling;
      while (n) { after += n.textContent || ''; const nx = n.nextSibling; n.remove(); n = nx; }
      after = after.replace(/\s+/g, ' ').trim();
      if (after && !sub) sub = after;
    }
    h1.querySelectorAll('br').forEach((b) => b.remove());
    h1.textContent = h1.textContent.replace(/\s+/g, ' ').trim();
    if (sub) {
      const p = document.createElement('p');
      const em = document.createElement('em');
      em.textContent = sub; p.appendChild(em);
      h1.after(p);
    }
  });
  return document.body.innerHTML;
}

async function main() {
  const browser = await puppeteer.launch({
    executablePath: EDGE, headless: 'new',
    args: ['--no-sandbox', '--force-color-profile=srgb'],
  });
  const td = buildTurndown();
  const summary = [];
  for (const cfg of FILES) {
    const abs = path.join(WIKI, cfg.f);
    const base = path.basename(cfg.f, '.html');
    const dir = path.dirname(abs);
    const assetsDir = path.join(dir, 'assets');
    fs.mkdirSync(assetsDir, { recursive: true });

    const page = await browser.newPage();
    await page.setViewport({ width: 1280, height: 1600, deviceScaleFactor: 2 });
    await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'light' }]);
    await page.goto(pathToFileURL(abs).href, { waitUntil: 'networkidle0', timeout: 60000 }).catch(() => {});
    await page.addStyleTag({ content: REVEAL_FIX });
    await page.evaluate(async () => { try { await document.fonts.ready; } catch (e) {} });
    await new Promise((r) => setTimeout(r, 400));

    const metas = await page.evaluate(prepFigures, cfg.figureSelector, cfg.captionSel);
    for (const m of metas) {
      const name = `${base}_fig${m.index + 1}.png`;
      const out = path.join(assetsDir, name);
      const el = await page.$(`[data-figindex="${m.index}"]`);
      if (el) {
        await el.screenshot({ path: out });
        m.src = `assets/${name}`;
      } else {
        m.src = '';
      }
    }
    const bodyHtml = await page.evaluate(cleanDom, metas);
    await page.close();

    let md = td.turndown(bodyHtml);
    md = md
      .replace(/\\_/g, '_')           // un-escape snake_case underscores (turndown over-escapes; code blocks are never escaped)
      .replace(/\n[ \t]+\n/g, '\n\n') // drop whitespace-only lines
      .replace(/\n{3,}/g, '\n\n')
      .trim() + '\n';
    const mdPath = path.join(dir, `${base}.md`);
    fs.writeFileSync(mdPath, md, 'utf8');
    summary.push({ file: cfg.f, figures: metas.length, mdBytes: Buffer.byteLength(md), md: path.relative(WIKI, mdPath) });
    console.log(`OK  ${cfg.f}  -> ${metas.length} png, ${Buffer.byteLength(md)} md bytes`);
  }
  await browser.close();
  fs.writeFileSync(path.join(__dirname, 'summary.json'), JSON.stringify(summary, null, 2));
  console.log('\nDONE');
}
main().catch((e) => { console.error(e); process.exit(1); });
