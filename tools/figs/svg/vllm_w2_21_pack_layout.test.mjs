import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {example,hex,unpack,render} from './vllm_w2_21_pack_layout.mjs';
const page=readFileSync(new URL('../../../wiki/02_engineering/03_infer_frameworks/vllm/21_vllm_quantization_analysis.md',import.meta.url),'utf8');
test('AWQ reverse indexing and input packing preserve the complete logical matrix',()=>{
 const d=example(); assert.deepEqual(d.recovered,d.codes);
 for(let n=0;n<8;n++) assert.deepEqual(unpack(d.standardWords[n]),d.codes.map(row=>row[n]));
 assert.deepEqual(d.awqSlots[0],[0,2,4,6,1,3,5,7]);
 assert.equal(hex(d.awqWords[0]),'0x75316420');assert.equal(hex(d.standardWords[0]),'0x76543210');
 for(const value of [hex(d.awqWords[0]),hex(d.standardWords[0]),String(d.y)]) {
   assert.ok(page.includes(value),`page must preserve computed ${value}`);
   assert.ok(render().includes(value),`figure must preserve computed ${value}`);
 }
 assert.equal(d.y,-4.5);
});
test('E4M3FN conversion values, bytes and output agree with the actual page table',()=>{
 const d=example();assert.equal(d.scale,.5);
 assert.deepEqual(d.fp8.map(x=>x.value),[1.125,2.25,4.5,448]);
 assert.deepEqual(d.fp8.map(x=>x.byte.toString(16)),['39','41','49','7e']);
 assert.equal(d.reference,3.828125);assert.equal(d.quantized,3.9375);
 assert.ok(page.includes(`| E4M3FN 最近可表示值 $q$ | ${d.fp8.map(x=>x.value).join(' | ')} |`));
 assert.ok(page.includes(`| 编码 byte（十六进制） | ${d.fp8.map(x=>x.byte.toString(16)).join(' | ')} |`));
 assert.ok(page.includes(`\`${d.reference}\``));assert.ok(page.includes(`\`${d.quantized}\``));
});
