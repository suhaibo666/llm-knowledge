import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { model, render } from '../megatron_pp_gpipe_figures.mjs';

const page = new URL('../../../../wiki/02_engineering/02_train_frameworks/megatron-lm/15_megatron_pp_schedulers_analysis.md', import.meta.url);
const asset = new URL('./assets/megatron_pp_gpipe_to_1f1b.svg', page);

test('same work and messages; GPipe retains m records while 1F1B consumes earlier records', () => {
  for (const [pp, m] of [[1, 1], [2, 4], [4, 8], [4, 12]]) {
    const [g, f] = model(pp, m);
    assert.equal(g.span, f.span);
    assert.equal(g.span, 2 * (m + pp - 1));
    assert.deepEqual(g.peaks, Array(pp).fill(m));
    assert.deepEqual(f.peaks, Array.from({ length: pp }, (_, r) => pp - r));
    for (const p of [g, f]) for (const cells of p.rows) {
      assert.equal(cells.filter(Boolean).length, 2 * m);
      for (let mb = 0; mb < m; mb++) {
        assert.ok(cells.findIndex(x => x?.mb === mb && x.f) < cells.findIndex(x => x?.mb === mb && !x.f));
      }
    }
  }
});

test('rendered asset and page independently preserve the simulated comparison', async () => {
  const text = await readFile(page, 'utf8');
  assert.equal((await readFile(asset, 'utf8')).replace(/\r\n/g, '\n').trimEnd(), render());
  const [g, f] = model();
  const gp = text.slice(text.indexOf('### 2.2 GPipe'), text.indexOf('### 2.3 1F1B'));
  const one = text.slice(text.indexOf('### 2.3 1F1B'), text.indexOf('### 2.4 VPP'));
  assert.ok(gp.includes(`[${g.peaks}]`) && one.includes(`[${f.peaks}]`));
  for (const section of [gp, one]) {
    assert.ok(section.includes(`$${g.span}t_f$`));
    assert.ok(section.includes(`${g.messages}`));
  }
  assert.ok(text.includes('assets/megatron_pp_gpipe_to_1f1b.svg'));
});
