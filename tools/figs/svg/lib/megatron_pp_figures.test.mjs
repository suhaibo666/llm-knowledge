// PP 页面与五张脚本生成图的结构化契约。
// 运行：node --test tools/figs/svg/lib/megatron_pp_figures.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readdirSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { dirname, join, resolve } from 'node:path';
import { homedir } from 'node:os';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '../../../..');
const scriptDir = join(root, 'tools', 'figs', 'svg');
const pagePath = join(root, 'wiki', '02_engineering', '02_train_frameworks', 'megatron-lm', '15_megatron_pp_schedulers_analysis.md');
const assetDir = join(root, 'wiki', '02_engineering', '02_train_frameworks', 'megatron-lm', 'assets');
const figures = [
  ['megatron_vpp_vs_1f1b.mjs', 'megatron_pp_vpp_vs_1f1b.svg'],
  ['megatron_pp_p2p_overlap.mjs', 'megatron_pp_p2p_overlap.svg'],
  ['megatron_combined_1f1b.mjs', 'megatron_pp_combined_1f1b.svg'],
  ['megatron_pp_p2p_transports.mjs', 'megatron_pp_p2p_transports.svg'],
  ['megatron_pp_multimodule_bridge.mjs', 'megatron_pp_multimodule_bridge.svg'],
];
const normalize = (s) => s.replace(/^\uFEFF/, '').replace(/\r\n/g, '\n').trimEnd();

function generate(script) {
  const result = spawnSync(process.execPath, [join(scriptDir, script)], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr || result.stdout);
  assert.match(result.stdout, /^<svg[\s>]/);
  return normalize(result.stdout);
}

function contract(svg) {
  const match = svg.match(/data-contract="([^"]+)"/);
  assert.ok(match, 'SVG 必须声明 data-contract');
  return Object.fromEntries(match[1].split(';').map((item) => {
    const at = item.indexOf('=');
    return [item.slice(0, at), item.slice(at + 1)];
  }));
}

function validatePageAgainstFigures(page, svgByAsset) {
  const vpp = contract(svgByAsset.get('megatron_pp_vpp_vs_1f1b.svg'));
  const overlap = contract(svgByAsset.get('megatron_pp_p2p_overlap.svg'));
  const combined = contract(svgByAsset.get('megatron_pp_combined_1f1b.svg'));
  const transports = contract(svgByAsset.get('megatron_pp_p2p_transports.svg'));
  const bridge = contract(svgByAsset.get('megatron_pp_multimodule_bridge.svg'));

  for (const [, asset] of figures) assert.match(page, new RegExp(`assets/${asset.replaceAll('.', '\\.')}`));
  for (const term of [
    'no-pipeline', 'non-interleaved', 'VPP', 'combined-1F1B',
    'MultiModuleProcessGroupCollection', 'BridgeCommunicator',
    'ring_exchange', 'batch_isend_irecv', 'isend/irecv', 'NCCL', 'UCC',
  ]) assert.ok(page.includes(term), `Markdown 缺少 variant/data-plane 名称：${term}`);

  assert.ok(page.includes(`$${vpp.standard_span}t_f$`), 'Markdown 与标准 1F1B span 漂移');
  assert.ok(page.includes(`$${Number(vpp.vpp_span) / Number(vpp.v)}t_f$`), 'Markdown 与 VPP span 漂移');
  assert.ok(page.includes('`[' + vpp.standard_live + ']`'), 'Markdown 与普通 peak-live 漂移');
  assert.ok(page.includes('`[' + vpp.vpp_live + ']`'), 'Markdown 与 VPP peak-live 漂移');
  assert.ok(page.includes(`$2m(Pv-1)=${vpp.vpp_messages}$`), 'Markdown 与 VPP message 数漂移');
  assert.ok(page.includes(`$112/48=7/3$`), 'Markdown 缺少精确 VPP message 比');

  assert.ok(page.includes(`$${overlap.sync_span}t$`), 'Markdown 与同步 overlap 算例漂移');
  assert.ok(page.includes(`$${overlap.async_compute}t$`), 'Markdown 与 async compute 区间漂移');
  assert.ok(page.includes(`$${overlap.async_completion}t$`), 'Markdown 与 async drain 漂移');
  assert.ok(page.includes(`$${overlap.saved}t$`), 'Markdown 与 exposed saving 漂移');

  assert.ok(page.includes('`[' + combined.live + ']`'), 'Markdown 与 combined peak-live 漂移');
  assert.ok(page.includes(`PP messages 仍为 ${combined.messages}`), 'Markdown 与 combined PP message 数漂移');
  assert.ok(page.includes('combine_bwd(m0)') && page.includes('combine_fwd(m1)'), 'Markdown 缺少 combined F/B 节点闭环');

  assert.equal(transports.shape_words, '3');
  assert.equal(transports.shape_bytes_per_direction, '24');
  assert.ok(page.includes('3 个 int64，即 24 B'), 'Markdown 与 shape metadata 契约漂移');
  assert.ok(page.includes('CPU 发起的全设备同步') && page.includes('request'), 'Markdown 缺少 wait/sync 成本');

  assert.ok(page.includes(`src DP=${bridge.fanin_src_dp} → dst DP=${bridge.fanin_dst_dp}`), 'Markdown 与 fan-in contract 漂移');
  assert.ok(page.includes(`src DP=${bridge.fanout_src_dp} → dst DP=${bridge.fanout_dst_dp}`), 'Markdown 与 fan-out contract 漂移');
  assert.ok(page.includes('split [2,2]') && page.includes('`cat`'), 'Markdown 缺少 batch split/cat');
  assert.ok(page.includes('destination leader') && page.includes('broadcast'), 'Markdown 缺少 leader/broadcast');
  assert.ok(page.includes(`加权最长路径恰有 $P=${bridge.host_P}$`) && page.includes(`峰值记录 \`[${bridge.host_live}]\``), 'Markdown 与 Multi-Module host schedule 漂移');
  assert.ok(page.includes('读图时'), '图与页面必须给出陌生读者线');
}

function validateBridgeReviewContract(page, svg) {
  const bridge = contract(svg);
  assert.equal(bridge.split_peer_form, 'direct');
  assert.equal(bridge.split_sample_form, 'grouped');
  assert.equal(bridge.split_sample_input, '0,3,1,2');
  assert.equal(bridge.split_sample_peers, '2');
  assert.equal(bridge.split_sample_output, '3,3');
  assert.equal(bridge.split_sum_rule, 'batch_dim');
  assert.equal(bridge.source_leader, 'group[-1]');
  assert.equal(bridge.destination_leader, 'group[0]');

  assert.match(svg, /src[^<]*group\[-1\]/, 'Bridge 图未显示 source leader 的 group[-1] 选择');
  assert.match(svg, /dst[^<]*group\[0\]/, 'Bridge 图未显示 destination leader 的 group[0] 选择');
  assert.match(svg, /\[0,3,1,2\][^<]*\[3,3\]/, 'Bridge 图未回放逐样本 metadata 的 peer 聚合');

  assert.ok(page.includes('`group[-1]`') && page.includes('`group[0]`'), 'Markdown 缺少 leader 方向非对称');
  assert.match(page, /长度等于 peer 数[^\n]*直接/, 'Markdown 缺少 per-peer split metadata 分支');
  assert.match(page, /长度大于 peer 数[^\n]*整除/, 'Markdown 缺少逐样本 metadata 聚合条件');
  assert.match(page, /\[0,3,1,2\][^\n]*\[3,3\]/, 'Markdown 缺少 [0,3,1,2] → [3,3] 回放');
  assert.match(page, /总和[^\n]*batch[^\n]*ValueError/, 'Markdown 缺少 split metadata 总和失败边界');
  assert.match(page, /长度[^\n]*ValueError/, 'Markdown 缺少 split metadata 长度失败边界');
}

function validateHyperConnectionsConservation(page, svg) {
  const hyper = contract(svg);
  assert.equal(hyper.hyper_hidden, 'hidden_size*num_residual_streams');
  assert.equal(hyper.hyper_scope, 'intermediate_pp_boundary');
  assert.equal(hyper.hyper_recv_gate, 'physical_pp_rank_gt_0');
  assert.equal(hyper.hyper_send_gate, 'physical_pp_rank_lt_last');
  assert.equal(hyper.hyper_endpoints, 'hidden_size');
  assert.equal(hyper.hyper_flexible_vpp, 'TODO');
  assert.match(svg, /hidden_size \* num_residual_streams/, 'P2P boundary 图未显示 Hyper Connections hidden 维扩展');
  assert.match(svg, /flexible VPP TODO/, 'P2P boundary 图未显示 flexible VPP TODO 边界');
  assert.match(svg, /recv[^<]*physical PP rank &gt; 0/, 'P2P boundary 图未显示 recv 的中间边界 gate');
  assert.match(svg, /send[^<]*physical PP rank &lt; PP last/, 'P2P boundary 图未显示 send 的中间边界 gate');
  assert.match(
    page,
    /Hyper Connections[^\n]*`hidden_size \* num_residual_streams`[^\n]*flexible VPP[^\n]*TODO/,
    'Markdown 缺少 Hyper Connections 边界 shape 与 flexible-VPP TODO',
  );
  // 守恒锚点绑在正文的可验证证据上，而不是绑在一节「旧 HEAD 内容守恒」的过程记录上：
  // 守恒是写给协调者的义务，页面历史归 changelog，页面本身只该留下 shape 断言的锁。
  assert.match(
    page,
    /test_pp2_flexible_vpp_mhc_send_recv_match/,
    '正文未登记 flexible-VPP boundary shape 的锁定测试',
  );
  assert.match(
    page,
    /test_pp4_flexible_vpp_mhc_all_consecutive_match/,
    '正文未登记 flexible-VPP boundary shape 的锁定测试',
  );
}

function accentFamilies(svg) {
  const colors = [...svg.matchAll(/#[0-9a-fA-F]{6}/g)].map((m) => m[0].slice(1));
  const families = new Set();
  for (const hex of colors) {
    const [r, g, b] = [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
    const max = Math.max(r, g, b), min = Math.min(r, g, b), delta = max - min;
    const saturation = max === 0 ? 0 : delta / max;
    if (saturation < 0.35 || delta === 0) continue;
    let hue;
    if (max === r) hue = 60 * (((g - b) / delta) % 6);
    else if (max === g) hue = 60 * ((b - r) / delta + 2);
    else hue = 60 * ((r - g) / delta + 4);
    if (hue < 0) hue += 360;
    if (hue >= 195 && hue <= 235) families.add('blue');
    else if (hue >= 15 && hue <= 40) families.add('orange');
    else families.add(`unexpected-${Math.round(hue)}`);
  }
  return families;
}

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

test('五张生成图与 tracked SVG 逐字一致，并带可访问描述与双强调色', async () => {
  for (const [script, asset] of figures) {
    const generated = generate(script);
    const tracked = normalize(await readFile(join(assetDir, asset), 'utf8'));
    assert.equal(generated, tracked, `${asset} 不是 ${script} 的当前生成结果`);
    assert.match(generated, /role="img"/);
    assert.match(generated, /aria-label="[^"]{30,}"/);
    const families = accentFamilies(generated);
    assert.ok(families.size <= 2, `${asset} 强调色族超过 2：${[...families].join(', ')}`);
    assert.deepEqual([...families].sort(), ['blue', 'orange']);
  }
});

test('Markdown 的结构化算例、variant lane 与 SVG data-contract 一致', async () => {
  const page = await readFile(pagePath, 'utf8');
  const svgByAsset = new Map();
  for (const [script, asset] of figures) svgByAsset.set(asset, generate(script));
  validatePageAgainstFigures(page, svgByAsset);
});

test('Bridge split metadata 两种形式与 leader 方向非对称在正文和图中同源', async () => {
  const page = await readFile(pagePath, 'utf8');
  const svg = generate('megatron_pp_multimodule_bridge.mjs');
  validateBridgeReviewContract(page, svg);
});

test('旧 HEAD 的 Hyper Connections boundary shape 与 flexible-VPP TODO 守恒', async () => {
  const page = await readFile(pagePath, 'utf8');
  const svg = generate('megatron_pp_p2p_overlap.mjs');
  validateHyperConnectionsConservation(page, svg);
});

test('Hyper Connections 图单边漂移会失败：fixed boundary contract 不能只改 SVG', async () => {
  const page = await readFile(pagePath, 'utf8');
  const svg = generate('megatron_pp_p2p_overlap.mjs');
  const figureOnlyMutation = svg.replace(
    'hyper_hidden=hidden_size*num_residual_streams',
    'hyper_hidden=hidden_size',
  );
  assert.notEqual(figureOnlyMutation, svg);
  assert.throws(
    () => validateHyperConnectionsConservation(page, figureOnlyMutation),
    /Expected values to be strictly equal/,
  );
});

test('正文单边漂移会失败：VPP message 数不能只改 Markdown', async () => {
  const page = await readFile(pagePath, 'utf8');
  const svgByAsset = new Map();
  for (const [script, asset] of figures) svgByAsset.set(asset, generate(script));
  const bodyOnlyMutation = page.replace('$2m(Pv-1)=112$', '$2m(Pv-1)=113$');
  assert.notEqual(bodyOnlyMutation, page);
  assert.throws(() => validatePageAgainstFigures(bodyOnlyMutation, svgByAsset), /message 数漂移/);
});

test('图单边漂移会失败：VPP data-contract 不能只改 SVG', async () => {
  const page = await readFile(pagePath, 'utf8');
  const svgByAsset = new Map();
  for (const [script, asset] of figures) svgByAsset.set(asset, generate(script));
  const name = 'megatron_pp_vpp_vs_1f1b.svg';
  svgByAsset.set(name, svgByAsset.get(name).replace('vpp_messages=112', 'vpp_messages=113'));
  assert.throws(() => validatePageAgainstFigures(page, svgByAsset), /message 数漂移/);
});

test('真实浏览器渲染：文字不出 viewBox，文字包围盒不互相覆盖', async (t) => {
  const executablePath = findBrowserExecutable();
  const puppeteerEntry = join(root, 'tools', 'mkdocs-site', 'node_modules', 'puppeteer-core', 'lib', 'esm', 'puppeteer', 'puppeteer-core.js');
  if (!executablePath || !existsSync(puppeteerEntry)) {
    t.skip('本机没有可用 Chromium/puppeteer-core');
    return;
  }
  const { default: puppeteer } = await import(pathToFileURL(puppeteerEntry).href);
  let browser;
  try {
    browser = await puppeteer.launch({ executablePath, headless: true, args: ['--disable-gpu', '--no-sandbox'] });
  } catch (error) {
    t.skip(`浏览器不可启动：${error.message}`);
    return;
  }
  try {
    for (const [script, asset] of figures) {
      const page = await browser.newPage();
      await page.setContent(generate(script));
      const audit = await page.evaluate(() => {
        const svg = document.querySelector('svg');
        const view = svg.viewBox.baseVal;
        const boxes = [...svg.querySelectorAll('text')].map((element) => {
          const box = element.getBBox();
          return { text: element.textContent, x: box.x, y: box.y, right: box.x + box.width, bottom: box.y + box.height };
        });
        const clipped = boxes.filter((box) => (
          box.x < view.x - 0.5 || box.y < view.y - 0.5
          || box.right > view.x + view.width + 0.5 || box.bottom > view.y + view.height + 0.5
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
      await page.close();
      assert.deepEqual(audit.clipped, [], `${asset} 有文字越出 viewBox`);
      assert.deepEqual(audit.overlaps, [], `${asset} 有文字包围盒重叠`);
    }
  } finally {
    await browser.close();
  }
});
