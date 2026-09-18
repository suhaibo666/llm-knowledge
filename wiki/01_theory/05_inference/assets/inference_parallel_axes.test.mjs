import assert from 'node:assert/strict';
import { copyFile, mkdtemp, mkdir, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import test from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';

const assets = dirname(fileURLToPath(import.meta.url));
const wiki = join(assets, '..');
const pages = [
  '27_prefill_context_parallelism_analysis.md',
  '28_decode_context_parallelism_analysis.md',
  '29_chunked_pipeline_parallelism_analysis.md',
];
const figures = [
  '27_pcp_query_kv_layout.svg',
  '28_dcp_kv_merge.svg',
  '29_cpp_stage_chunk_timeline.svg',
];

function scheduleFromPage(markdown) {
  const rows = markdown.split('\n').filter((line) => /^\|\s*[1-6]\s*\|/.test(line));
  assert.equal(rows.length, 6, 'CPP page must contain all six time slots');
  return rows.map((line, index) => {
    const cells = line.split('|').slice(1, 5).map((cell) => cell.trim());
    assert.equal(Number(cells[0]), index + 1, 'CPP page time slots must be ordered');
    return cells.slice(1).map((cell) => {
      if (cell === '空') return null;
      const match = cell.match(/^\$C_(\d+)\$$/);
      assert.ok(match, `unexpected CPP page cell: ${cell}`);
      return Number(match[1]);
    });
  });
}

function scheduleFromSvg(svg) {
  const cells = [...svg.matchAll(/<rect class="(blue|ghost)" x="(\d+)" y="(\d+)" width="\d+" height="\d+" rx="5"\/><text class="(?:head|small)" x="[^"]+" y="[^"]+" text-anchor="middle">(C\d+|空)<\/text>/g)]
    .map((match) => ({ kind: match[1], x: Number(match[2]), y: Number(match[3]), label: match[4] }));
  assert.equal(cells.length, 18, 'CPP SVG must have a full 3 × 6 grid');
  const xs = [...new Set(cells.map((cell) => cell.x))].sort((a, b) => a - b);
  const ys = [...new Set(cells.map((cell) => cell.y))].sort((a, b) => a - b);
  assert.equal(xs.length, 6);
  assert.equal(ys.length, 3);
  return xs.map((x) => ys.map((y) => {
    const matches = cells.filter((cell) => cell.x === x && cell.y === y);
    assert.equal(matches.length, 1, `CPP SVG cell at (${x}, ${y})`);
    const cell = matches[0];
    if (cell.kind === 'ghost') {
      assert.equal(cell.label, '空');
      return null;
    }
    assert.match(cell.label, /^C[1-4]$/);
    return Number(cell.label.slice(1));
  }));
}

test('PCP, DCP and CPP figures stay consistent with their actual wiki pages', async () => {
  // Import a copy: the generator writes SVGs when imported, so keep this test
  // from modifying the checked-in figures while still executing its real code.
  const temporary = await mkdtemp(join(tmpdir(), 'inference-parallel-axes-'));
  const temporaryAssets = join(temporary, 'assets');
  try {
    await mkdir(temporaryAssets);
    await copyFile(join(assets, 'inference_parallel_axes.mjs'), join(temporaryAssets, 'inference_parallel_axes.mjs'));
    await Promise.all(pages.map((page) => copyFile(join(wiki, page), join(temporary, page))));
    const generator = await import(pathToFileURL(join(temporaryAssets, 'inference_parallel_axes.mjs')).href);
    const markdown = await Promise.all(pages.map((page) => readFile(join(wiki, page), 'utf8')));
    const svg = await Promise.all(figures.map((figure) => readFile(join(temporaryAssets, figure), 'utf8')));
    for (let index = 0; index < figures.length; index += 1) {
      assert.equal(svg[index], await readFile(join(assets, figures[index]), 'utf8'), `${figures[index]} must match its generator`);
      assert.ok(markdown[index].includes(`assets/${figures[index]}`), `${pages[index]} must display its checked figure`);
    }

    const pcpRows = [...markdown[0].matchAll(/^\|\s*rank\s+([01])\s*\|\s*\$(\d)\$\s*\|\s*\$([\d,]+)\$\s*\|\s*\$(\d)\$/gm)];
    assert.equal(pcpRows.length, 4, 'PCP page must specify all four query positions');
    const scores = [0, 0];
    for (const row of pcpRows) {
      const rank = Number(row[1]);
      const position = Number(row[2]);
      const visible = row[3].split(',').map(Number);
      const count = Number(row[4]);
      assert.deepEqual(visible, Array.from({ length: position + 1 }, (_, i) => i));
      assert.equal(count, visible.length);
      scores[rank] += count;
      assert.ok(svg[0].includes(`q${position} ← K/V ${visible.join('、')}`));
    }
    assert.deepEqual(scores, [3, 7]);
    assert.ok(svg[0].includes('有效分数项：1 + 2 = 3'));
    assert.ok(svg[0].includes('有效分数项：3 + 4 = 7'));
    assert.ok(svg[0].includes('输出顺序：o0，o1，o2，o3'));
    for (const claim of ['$1+2=3$', '$3+4=7$', '$1+4=5$', '$2+3=5$']) assert.ok(markdown[0].includes(claim));

    const dcp = generator.dcpStats();
    assert.ok(Math.abs(dcp.result - 14 / 5) < 1e-12);
    assert.ok(Math.abs(dcp.direct - dcp.result) < 1e-12);
    assert.ok(Math.abs(dcp.naive - 8 / 3) < 1e-12);
    assert.deepEqual(dcp.local.map((part) => part.ids), [[0, 2], [1, 3]]);
    assert.ok(svg[1].includes('ℓ = 5/2，u = 7，o3 = 2.8 = 14/5'));
    assert.ok(svg[1].includes('(2 + 10/3)/2 = 8/3'));
    for (const claim of ['$14/5$', '$8/3$', 'rank 0：$0,2$', 'rank 1：$1,3$']) assert.ok(markdown[1].includes(claim));

    const schedule = generator.cppSchedule(3, 4);
    assert.equal(schedule.slots, 6);
    assert.equal(schedule.busy, 12);
    assert.equal(schedule.bubbles, 6);
    assert.equal(schedule.ratio, 1 / 3);
    const timeMajor = Array.from({ length: schedule.slots }, (_, slot) => schedule.grid.map((stage) => stage[slot]));
    assert.deepEqual(scheduleFromPage(markdown[2]), timeMajor, 'CPP page table must match generator schedule');
    assert.deepEqual(scheduleFromSvg(svg[2]), timeMajor, 'CPP SVG cells must match generator schedule');
    for (let stage = 0; stage < 3; stage += 1) {
      for (let chunk = 1; chunk <= 4; chunk += 1) {
        const slot = timeMajor.findIndex((column) => column[stage] === chunk);
        assert.equal(slot, stage + chunk - 1);
        if (stage > 0) assert.equal(timeMajor[slot - 1][stage - 1], chunk, 'upstream activation must be ready');
        if (chunk > 1) assert.equal(timeMajor[slot - 1][stage], chunk - 1, 'prior local KV must be ready');
      }
    }
    assert.equal(timeMajor[5][2], 4, 'full prompt completes only at final stage and chunk');
    for (const claim of ['$3\\times4=12$', '$3\\times6=18$', '空格为 $6$', '空格比例 $1/3$']) assert.ok(markdown[2].includes(claim));
    assert.ok(svg[2].includes('忙格 12，空格 6，理想气泡 33.3%'));
    assert.ok(svg[2].includes('完成边界：S2 的 C4，格 6'));
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
});
