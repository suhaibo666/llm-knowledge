import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {model, draw, out} from './vllm_kv_slot_address_layout.mjs';

const page = readFileSync(new URL('../../../wiki/02_engineering/03_infer_frameworks/vllm/10_vllm_attention_backends_analysis.md', import.meta.url), 'utf8');
const list = a => '[' + a.join(', ') + ']';

test('dense LBNHC offset agrees with the slot-based closed form', () => {
  const m = model();
  assert.deepEqual(m.slots, [453, 210, 211]);
  assert.equal(m.block * m.Bk + m.offset, m.slot);
  assert.equal(m.k, m.kFormula);
  assert.equal(m.v - m.k, m.D * m.e);
  // LBHNC closed form: (b·H·Bk·2D + h·Bk·2D + o·2D + d)·e; same slot, different bytes.
  const {block: b, H, Bk, D, h, offset: o, d, e} = m;
  assert.equal(m.kLBHNC, (b * H * Bk * 2 * D + h * Bk * 2 * D + o * 2 * D + d) * e);
  assert.notEqual(m.kLBHNC, m.k);
});

test('page prose uses the numbers the figure is drawn from', () => {
  const m = model();
  assert.ok(page.includes(`\`slot_mapping\` | \`${list(m.slots)}\``), 'running-example slots drifted');
  assert.ok(page.includes(`${m.block}×${m.Bk}+${m.offset}=${m.slot}`), 'A19 slot arithmetic drifted');
  for (const [name, value] of Object.entries({s_b: m.strides.sB, s_h: m.strides.sH, s_o: m.strides.sO, s_c: m.strides.sC})) {
    assert.ok(page.includes(`${name}=${value}`), `${name} drifted`);
  }
  assert.ok(page.includes(`**${m.k} B**`), 'K byte offset drifted');
  assert.ok(page.includes(`**${m.v} B**`), 'V byte offset drifted');
  assert.ok(page.includes(`s_o=${m.lbhnc.sO}`) && page.includes(`s_h=${m.lbhnc.sH}`), 'LBHNC strides drifted');
  assert.ok(page.includes(`**${m.kLBHNC} B**`) && page.includes(`**${m.vLBHNC} B**`), 'LBHNC offsets drifted');
  assert.ok(page.includes(`cache[${m.block}, 0, ${m.offset}, ${m.kIndex}]`), 'K index drifted');
  assert.ok(page.includes(`cache[${m.block}, 0, ${m.offset}, ${m.vIndex}]`), 'V index drifted');
  assert.ok(page.includes(m.expandB.join(',')) && page.includes(m.expandA.join(',')), 'manager→kernel expansion drifted');
});

test('checked-in SVG is current and contains real text', () => {
  const svg = draw();
  assert.equal(readFileSync(out, 'utf8').trim(), svg.trim());
  assert.ok(svg.includes('<text') && !svg.includes('<foreignObject'));
});
