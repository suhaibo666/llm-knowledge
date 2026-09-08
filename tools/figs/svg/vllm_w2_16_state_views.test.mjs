import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {replay,drawStable,drawUbatch} from './vllm_w2_16_state_views.mjs';
const root=new URL('../../../wiki/02_engineering/03_infer_frameworks/vllm/',import.meta.url);
const page=readFileSync(new URL('12_vllm_model_runner_v2_analysis.md',root),'utf8');
const arr=x=>'['+x.join(',')+']';
test('remove leaves state holes; step order is an indirection',()=>{
 const r=replay();assert.deepEqual(r.map,{A:3,B:1});assert.deepEqual(r.free,[0,2]);assert.deepEqual(r.idx,[1,3]);
 for(const s of [`idx_mapping=${arr(r.idx)}`,`query_start_loc=${arr(r.qsl)}`,`seq_lens=${arr(r.seq)}`]) assert.ok(page.includes(s),s);
 assert.match(page,/X 完成后 row 2 归还，A 和 B 都留在原行/);
});
test('cross-boundary request removes only future query, not prior ubatch history',()=>{
 const r=replay();assert.deepEqual(r.first,{qsl:[0,1,2],seq:[6,19],tokens:2});assert.deepEqual(r.second,{qsl:[0,1],seq:[20],tokens:1});
 for(const part of [r.first,r.second])for(const s of [`query_start_loc=${arr(part.qsl)}`,`seq_lens=${arr(part.seq)}`])assert.ok(page.includes(s),s);
 assert.match(page,/A18 已在前一份计算，属于当前历史，不能再扣一次/);
});
test('staged descriptors reconstruct rows and sampled updates match prose',()=>{
 const rows={3:[12],1:[]},index=[3,1],start=[1,0],content=[13,14,28],ends=[2,3];
 index.forEach((row,i)=>content.slice(i?ends[i-1]:0,ends[i]).forEach((v,j)=>rows[row][start[i]+j]=v));
 assert.deepEqual(rows,{3:[12,13,14],1:[28]});
 for(const s of ['indices=[3,1]','starts=[1,0]','contents=[13,14,28]','cu_lens=[2,3]'])assert.ok(page.includes(s));
 const updated=[[5,1,0,6,206],[18,2,0,20,120]].map(([computed,q,rejected,total,last])=>({computed:computed+q-rejected,total:total+1,last}));
 for(const x of updated)assert.ok(page.includes(`computed=${x.computed},total=${x.total},last=${x.last}`));
});
test('both checked-in SVGs match generators and keep text as text',()=>{
 for(const [name,draw]of [['stable_rows',drawStable],['ubatch',drawUbatch]]){
  assert.equal(readFileSync(new URL(`assets/vllm_w2_16_${name}.svg`,root),'utf8').trim(),draw().trim());
  assert.ok(draw().includes('<text'));assert.ok(!draw().includes('<foreignObject'));
 }
});
