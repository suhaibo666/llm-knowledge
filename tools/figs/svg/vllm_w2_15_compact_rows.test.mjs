import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {replay,draw} from './vllm_w2_15_compact_rows.mjs';
const page=readFileSync(new URL('../../../wiki/02_engineering/03_infer_frameworks/vllm/11_vllm_model_runner_v1_analysis.md',import.meta.url),'utf8');
const list=a=>'['+a.join(',')+']';
test('compact/swap carry record identity and prose matches the replay',()=>{
 const r=replay();
 assert.deepEqual(r.compact.map(x=>x.id),['A','B']);assert.deepEqual(r.ordered.map(x=>x.id),['B','A']);
 assert.equal(r.ordered[0],r.holes[2]);assert.equal(r.ordered[1],r.holes[0]);
 assert.match(page,/尾部 B 从 row 2 移到 row 1/);assert.match(page,/交换 row 0、1/);
 assert.deepEqual(r.ordered.map(x=>[x.temp,x.lora,x.blocks]),[[0.6,7,[28]],[0,0,[12,13]]]);
});
test('swap_states arguments agree between the reorder replay, the SVG and the page',()=>{
 const r=replay();
 assert.deepEqual(r.swaps,[[1,0]]);
 const [i1,i2]=r.swaps[0];
 assert.ok(draw().includes(`swap_states(${i1}, ${i2})`),'SVG swap label drifted');
 assert.ok(page.includes(`swap_states(${i1},${i2})`),'page §2.3 swap_states call drifted');
 assert.ok(page.includes(`(${i1},${i2},SWAP)`),'page §2.4 BatchUpdate move drifted');
 assert.ok(page.includes(`swap_states(${i1}, ${i2})`),'page figure-spec comment drifted');
 assert.doesNotMatch(page,/swap_states\(0, ?1\)|SWAP\(0,1\)|swap\(0,1\)/);
});
test('prose transformation table is computed from the same rows as SVG',()=>{
 const r=replay();
 for(const [label,value] of [
  ['`req_indices`',r.reqIndices],['`positions`',r.positions],
  ['CPU flattened token indices',r.tokenIndices],['`input_ids`',r.ids],
  ['`query_start_loc`',r.qsl],['`seq_lens`',r.seqLens]]) {
   assert.ok(page.includes(`| ${label} | \`${list(value)}\` |`),`${label} drifted`);
 }
 assert.ok(page.includes(`token LoRA mapping \`${list(r.loraTokens)}\``));
 assert.match(page,/row stride，即教学 `max_model_len`，为 32/);
 const prevMap=new Map([['B',2]]);const prev=r.ordered.map(x=>prevMap.get(x.id)??-1);
 assert.ok(page.includes(`prev_positions=${list(prev)}`));
});
test('checked-in SVG is current and contains real text',()=>{
 const file=new URL('../../../wiki/02_engineering/03_infer_frameworks/vllm/assets/vllm_w2_15_compact_rows.svg',import.meta.url);
 assert.equal(readFileSync(file,'utf8').trim(),draw().trim());
 assert.ok(draw().includes('<text'));assert.ok(!draw().includes('<foreignObject'));
});
