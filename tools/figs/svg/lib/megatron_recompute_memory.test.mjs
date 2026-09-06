import assert from 'node:assert/strict';
import {test} from 'node:test';
import {readFile} from 'node:fs/promises';
import {ledger,simulate,comparison,render,simulateChoice,choiceModels,renderChoices,mlaExample} from '../megatron_recompute_memory.mjs';
const pageUrl=new URL('../../../../wiki/02_engineering/02_train_frameworks/megatron-lm/18_megatron_recompute_analysis.md',import.meta.url);
const MiB=2**20,GiB=2**30;

test('tensor inventory follows linear and quadratic dimension scaling',()=>{
  const a=ledger(),b=ledger({s:4096});
  assert.equal(a.layer,512*MiB);
  assert.equal(a.boundary,16*MiB);
  assert.equal(a.internal,496*MiB);
  for(const [key,value] of Object.entries(a.tensors)) assert.equal(b.tensors[key],value*(key==='P'?4:2));
  assert.equal(ledger({bytes:4}).layer,2*a.layer);
  assert.throws(()=>ledger({h:4097}));
});

test('choice window follows last-consumer releases and supports budget decisions',async()=>{
  const page=await readFile(pageUrl,'utf8');
  for(const layers of [1,7,24])for(const pending of [1,4]){
    for(const strategy of ['none','core','block'])for(const selected of [0,layers]){
      const m=simulateChoice({layers,pending,strategy,selected});
      assert.equal(m.events.at(-1).total,0);
      assert.equal(m.events.at(-1).count,0);
      assert.ok(m.events.every(x=>x.saved>=0&&x.replay>=0));
      if(strategy==='block'&&selected===layers)assert.equal(m.peak,simulate({layers,pending,group:1}).peak);
    }
  }
  const core=simulateChoice({strategy:'core'});
  const replay=core.events.find(x=>x.phase==='core-replayed');
  assert.equal(core.initial-replay.saved,176*MiB);
  assert.equal(replay.replay,272*MiB);
  assert.equal(core.peak-core.initial,96*MiB);
  for(const row of choiceModels()){
    const line=page.split('\n').find(x=>x.startsWith(`| ${row.name}（w=4） |`));
    assert.ok(line?.includes(`| ${row.initial/GiB} GiB | ${row.peak/GiB} GiB |`),`choice row drift: ${row.name}`);
  }
  const firstBlock=budget=>Array.from({length:25},(_,selected)=>({selected,...simulateChoice({strategy:'block',selected})})).find(m=>m.peak<=budget*GiB).selected;
  assert.equal(firstBlock(20),15);
  assert.equal(firstBlock(3),24);
  assert.ok(core.peak<=26*GiB&&core.initial>20*GiB);
  assert.ok(simulate({group:2}).peak<=3*GiB);
  for(const x of ['**176 MiB**','**272 MiB**','**96 MiB**','20.875 GiB','3.4375 GiB','**26 GiB**','**20 GiB**','**3 GiB**'])assert.ok(page.includes(x),`budget prose drift ${x}`);
  assert.equal((await readFile(new URL('assets/megatron_recompute_choices.svg',pageUrl),'utf8')).replaceAll('\r\n','\n').trim(),renderChoices().trim());
});

test('MLA worked boundary distinguishes retained input, output storage and replay GEMMs',async()=>{
  const page=await readFile(pageUrl,'utf8'),e=mlaExample();
  assert.deepEqual(e,{retained:48,query:64,key:64,value:32,replayFlops:256});
  assert.equal(mlaExample({heads:4}).retained,e.retained);
  assert.equal(mlaExample({heads:4}).query,2*e.query);
  for(const term of [`${e.retained} B 与 ${e.query+e.key+e.value} B`,`**${e.replayFlops} FLOPs**`,'contiguous()','112 B'])assert.ok(page.includes(term),`MLA prose drift ${term}`);
});

test('storage lifecycle handles full groups, partial tails and drains all microbatches',()=>{
  for(const layers of [1,7,24]) for(const pending of [1,4]) for(const group of [1,2,4,30]) {
    const a=ledger(),m=simulate({layers,pending,group,account:a});
    assert.equal(m.initial,pending*Math.ceil(layers/group)*a.boundary);
    assert.equal(m.events.filter(e=>e.phase==='replay-layer').length,layers*pending);
    assert.equal(m.events.filter(e=>e.phase==='backward-layer').length,layers*pending);
    assert.equal(m.events.at(-1).count,0);
    assert.equal(m.events.at(-1).total,0);
    assert.ok(m.events.every(e=>e.saved>=0&&e.replay>=0));
    if(layers%group===0) assert.equal(m.peak,m.initial+group*a.layer);
  }
  assert.throws(()=>simulate({group:0}));
  assert.throws(()=>simulate({pending:1.5}));
});

test('page, table and figure share numerical results; changing prose alone fails',async()=>{
  const page=await readFile(pageUrl,'utf8');
  for(const row of comparison()) {
    const line=page.split('\n').find(x=>x.startsWith(`| ${row.name} |`));
    assert.ok(line,`missing table row ${row.name}`);
    assert.ok(line.includes(`| ${row.saved/GiB} GiB |`),`retention mismatch ${row.name}`);
    if(row.saving) assert.ok(line.includes(`| ${row.saving/GiB} GiB |`),`saving mismatch ${row.name}`);
  }
  const ms=[1,2,4].map(group=>simulate({group}));
  for(const nums of [ms.map(m=>m.initial/MiB),ms.map(m=>m.events.find(e=>e.phase==='segment-replayed').replay/MiB),ms.map(m=>m.peak/MiB)])
    assert.ok(page.includes(`**${nums.join('、')} MiB**`),`prose drift ${nums}`);
  const svg=await readFile(new URL('assets/megatron_recompute_memory.svg',pageUrl),'utf8');
  assert.equal(svg.replaceAll('\r\n','\n').trim(),render().trim());
  assert.ok(page.includes('assets/megatron_recompute_memory.svg'));
});
