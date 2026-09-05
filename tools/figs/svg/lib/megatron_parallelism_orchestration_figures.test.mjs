import assert from 'node:assert/strict';
import { test } from 'node:test';
import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync, mkdirSync, readdirSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { tmpdir, homedir } from 'node:os';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { findBrowserExecutable } from '../../../docs-site/listeners.mjs';
import { CFG, cases, grids, derived, fmt, own, gridEnum, rankFigure, gridFigure, derivedFigure } from '../megatron_parallelism_orchestration_figures.mjs';
const root = fileURLToPath(new URL('../../../../', import.meta.url));
const pagePath = join(root,'wiki/02_engineering/02_train_frameworks/megatron-lm/17_megatron_parallelism_orchestration_analysis.md');
const assetDir = join(root,'wiki/02_engineering/02_train_frameworks/megatron-lm/assets');
const page = readFileSync(pagePath,'utf8');
const assets = [['ranks',rankFigure],['grid',gridFigure],['derived',derivedFigure]];

test('两种order的dense/expert均覆盖world一次且PP组完全对齐',()=>{
  for(const c of cases) assert.deepEqual(c.groups.flat().toSorted((a,b)=>a-b),Array.from({length:CFG.world},(_,i)=>i));
  assert.deepEqual(cases[0].pp,cases[1].pp);
  assert.deepEqual(cases[2].pp,cases[3].pp);
  assert.notDeepEqual(cases[0].pp,cases[2].pp);
  for(let i=0;i<2;i++) assert.deepEqual(grids[i].groups,cases[i].groups);
});

test('页面实例表逐行锁定sizes/mask/所有成员与R5的PP；改正文会失败',()=>{
  for(const c of cases) {
    const label=`${c.alternative?'alternative':'default'} ${c.expert?'expert':'dense'}`;
    const row=page.split('\n').find(l=>l.startsWith(`| ${label} `));
    assert.ok(row,`missing ${label}`);
    assert.ok(row.includes(`${fmt(c.sizes)} / ${c.mask.map(Number).join('')}`),`${label} input drift`);
    assert.ok(row.includes(c.groups.map(fmt).join(';')),`${label} enumeration drift`);
    assert.ok(row.includes(fmt(own(c.pp))),`${label} PP drift`);
  }
  assert.ok(page.includes(`world_size=${CFG.world}`));
  assert.ok(page.includes(`TP=${CFG.tp}, CP=${CFG.cp}, PP=${CFG.pp}`));
  assert.ok(page.includes(`ETP=${CFG.etp}, EP=${CFG.ep}, PP=${CFG.pp}`));
  for(const g of grids) {
    assert.ok(page.includes(`shape=${fmt(g.shape)}`));
    assert.ok(page.includes(g.axes.join(',')));
    assert.ok(page.includes(fmt(g.sourceAxes)) && page.includes(fmt(g.targetAxes)));
  }
  const d=derived();
  for(const key of ['hierarchical CP','dynamic DP×CP','partial DP','partial expert DP']) {
    const row=page.split('\n').find(l=>l.startsWith(`| ${key} /`));
    assert.ok(row,key);
    const wanted=key==='hierarchical CP'?d.hcp:key==='dynamic DP×CP'?d.dynamic:key==='partial DP'?d.partial:d.exptPartial;
    for(const level of wanted) assert.ok(row.includes(fmt(own(level))),`${key} ${fmt(own(level))}`);
  }
});

test('冻结源码原函数CPU复演：RankGenerator与NumPy moveaxis核对独立JS解算',t=>{
  const source=process.env.MEGATRON_SOURCE ?? resolve(root,'../Megatron-LM');
  if(!existsSync(join(source,'megatron/core/parallel_state.py'))) { t.skip('无冻结checkout'); return; }
  const py=String.raw`
from __future__ import annotations
import ast,json,sys,types,logging
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]); ns={'__builtins__':__builtins__,'np':np,'logging':logging}
tree=ast.parse((root/'megatron/core/parallel_state.py').read_text(encoding='utf-8'))
selected=[n for n in tree.body if getattr(n,'name','') in ['RankGenerator','generate_masked_orthogonal_rank_groups']]
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*selected],type_ignores=[])),'<frozen-ranks>','exec'),ns)
tree=ast.parse((root/'megatron/core/hyper_comm_grid.py').read_text(encoding='utf-8'))
cls=next(n for n in tree.body if getattr(n,'name','')=='HyperCommGrid')
selected=[n for n in cls.body if getattr(n,'name','') in ['_gen_rank_enum_for','_order_dims_for']]
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*selected],type_ignores=[])),'<frozen-grid>','exec'),ns)
out=[]
for alt in [False,True]:
 for expert in [False,True]:
  order='tp-cp-ep-pp-dp' if alt else 'tp-cp-ep-dp-pp'
  r=ns['RankGenerator'](tp=1 if expert else 2,ep=4 if expert else 1,dp=2,pp=2,cp=1 if expert else 2,order=order)
  out.append({'groups':r.get_ranks('ep' if expert else 'dp-cp'),'pp':r.get_ranks('pp')})
gr=[]
for shape,names,dims in [([2,2,2,2],['tp','cp','dp','pp'],['cp','dp']),([1,4,2,2],['expt_tp','ep','expt_dp','pp'],['ep'])]:
 ordered,_=ns['_order_dims_for'](None,names,dims)
 gr.append(ns['_gen_rank_enum_for'](types.SimpleNamespace(rank_offset=0),shape,names,ordered))
print(json.dumps({'rank':out,'grid':gr}))
`;
  const result=spawnSync('python',['-c',py,source],{encoding:'utf8'});
  assert.equal(result.status,0,result.stderr);
  const actual=JSON.parse(result.stdout);
  assert.deepEqual(actual.rank,cases.map(c=>({groups:c.groups,pp:c.pp})));
  assert.deepEqual(actual.grid,grids.map(g=>g.groups));
});

test('资产均由脚本重建且正文嵌入，PGC与后端边界有明示',()=>{
  for(const [name,fn] of assets) {
    assert.equal(normalizeEol(readFileSync(join(assetDir,`megatron_orchestration_${name}.svg`),'utf8')),normalizeEol(fn()));
    assert.ok(page.includes(`assets/megatron_orchestration_${name}.svg`));
  }
  for(const word of ['new_subgroups_by_enumeration','PyTorch','共享键机制','没有跨rank事务回滚保证','get_attr_wrapped_model','high_priority_stream_groups','local_rank']) assert.ok(page.includes(word),word);
});

test('真实Chromium渲染三图：文字bbox不相交，不出viewBox，保存审阅截图',async()=>{
  const entry=join(root,'tools/mkdocs-site/node_modules/puppeteer-core/lib/esm/puppeteer/puppeteer-core.js');
  const {default:puppeteer}=await import(pathToFileURL(entry).href);
  const cache=join(homedir(),'.cache/puppeteer/chrome');
  const cached=existsSync(cache)?readdirSync(cache).toSorted().toReversed().map(n=>join(cache,n,'chrome-win64/chrome.exe')).find(existsSync):undefined;
  const executablePath=process.env.PUPPETEER_EXECUTABLE_PATH ?? cached ?? await findBrowserExecutable();
  const browser=await puppeteer.launch({executablePath,headless:true,args:['--disable-gpu','--no-sandbox']});
  const out=join(tmpdir(),'megatron-feature-16-18-20260905'); mkdirSync(out,{recursive:true});
  try {
    for(const [name,fn] of assets) {
      const tab=await browser.newPage();await tab.setViewport({width:1180,height:1000,deviceScaleFactor:1});
      await tab.setContent(fn());
      const audit=await tab.evaluate(()=>{
        const svg=document.querySelector('svg'),v=svg.viewBox.baseVal;
        const bs=[...svg.querySelectorAll('text')].map(e=>{const b=e.getBBox();return {s:e.textContent,x:b.x,y:b.y,r:b.x+b.width,b:b.y+b.height};});
        const clipped=bs.filter(b=>b.x<v.x||b.y<v.y||b.r>v.width||b.b>v.height);
        const overlaps=[];
        for(let i=0;i<bs.length;i++)for(let j=i+1;j<bs.length;j++) if(Math.min(bs[i].r,bs[j].r)-Math.max(bs[i].x,bs[j].x)>1&&Math.min(bs[i].b,bs[j].b)-Math.max(bs[i].y,bs[j].y)>1)overlaps.push([bs[i].s,bs[j].s]);
        return {clipped,overlaps};
      });
      await (await tab.$('svg')).screenshot({path:join(out,`17_${name}.png`)});
      assert.deepEqual(audit,{clipped:[],overlaps:[]},name);
      await tab.close();
    }
  } finally {await browser.close();}
});

// Git checkouts may use CRLF; compare SVG content with only line endings normalized.
function normalizeEol(text) { return text.replace(/\r\n/g, '\n'); }
