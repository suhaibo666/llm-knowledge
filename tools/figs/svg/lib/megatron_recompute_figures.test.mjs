import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { existsSync,readdirSync } from 'node:fs';
import { homedir,tmpdir } from 'node:os';
import { resolve,join,dirname } from 'node:path';
import { fileURLToPath,pathToFileURL } from 'node:url';
import { solveExample,buildFigures } from '../megatron_recompute_figures.mjs';

const root=resolve(dirname(fileURLToPath(import.meta.url)),'../../../..');
const pageDir=join(root,'wiki/02_engineering/02_train_frameworks/megatron-lm');
const pagePath=join(pageDir,'18_megatron_recompute_analysis.md');
const figures=buildFigures();

test('same input arithmetic and finite-difference parameter gradients',()=>{
 const e=solveExample();
 assert.deepEqual(e.chunks,[[0,1],[2,3]]);
 assert.deepEqual(e.block,[[0],[1]]);
 assert.deepEqual(e.flags,[false,false,true,true]);
 assert.equal(e.epUniformSegments,5);
 assert.equal(e.epBlockSegments,3);
 const objective=(x,w,v)=>e.size*(x*w*e.mask*v)**2/2;
 const eps=1e-4;
 const fd=(axis)=>{
  const plus=[e.x,e.w,e.v],minus=[...plus];plus[axis]+=eps;minus[axis]-=eps;
  return (objective(...plus)-objective(...minus))/(2*eps);
 };
 assert.ok(Math.abs(fd(0)/e.size-e.dx)<1e-5);
 assert.ok(Math.abs(fd(1)-e.dw)<1e-5);
 assert.ok(Math.abs(fd(2)-e.dv)<1e-5);
 assert.equal(e.shardSize*e.tp,e.size);
 assert.equal(e.fullBytes/e.tp,e.shardBytes);
 assert.equal(e.carrierBytes,e.fullBytes*e.nstreams);
});

test('Markdown numbers and selectors agree with computed example, no one-sided drift',async()=>{
 const p=await readFile(pagePath,'utf8'),e=solveExample();
 for(const term of [
  `${e.size} 个元素、${e.fullBytes} B`,`${e.shardSize} 个元素、${e.shardBytes} B`,
  `${e.carrierSize} 个元素、${e.carrierBytes} B`,
  `y=${e.y}, z=${e.z}, loss=${e.loss}`,`\`${e.dx}\``,
  `dw=${e.dw}, dv=${e.dv}`,`${e.chunks.length} 个区域、${e.layers} 次层前向回放`,
  `${e.block.length} 个区域、${e.group} 次层前向回放`,
  `w=${e.layers},k=${e.group}，四个标志为 ${e.flags.map(x=>x?'True':'False').join(',')}`,
  `EP+MTP uniform n=1：${e.layers} 个 decoder 段 + 1 个 MTP 段 = ${e.epUniformSegments} 段`,
  `EP+MTP block n=${e.group}：${e.group} 个 decoder 段 + 1 个 MTP 段 = ${e.epBlockSegments} 段`,
 ])assert.ok(p.includes(term),`正文与求解结果漂移: ${term}`);
 const selectors=['core_attn','moe_act','layernorm','mla_up_proj','mlp','moe','shared_experts','mhc','gdn','gdn_norm_out'];
 for(const term of selectors){assert.ok(p.includes(`\`${term}\``));assert.ok(Object.values(figures).some(svg=>svg.includes(`data-plane="${term}"`)),`缺图lane ${term}`);}
 for(const term of ['moe_layer_recompute','num_microbatches_with_partial_activation_checkpoints','RecomputeSegment','supports_hybrid_recompute_kwargs','activation_recompute_in_mlp','BEFORE_COMBINE_BWD','StorageImpl'])assert.ok(p.includes(term),`关键契约缺失: ${term}`);
 const mhc=figures['megatron_recompute_mhc.svg'];
 const choice=mhc.match(/data-cell="mhc-post-branches-0"([\s\S]*?)<\/g>/)[1];
 const retained=mhc.match(/data-cell="mhc-post-branches-1"([\s\S]*?)<\/g>/)[1];
 assert.ok(choice.includes('互斥 A:')&&choice.includes('互斥 B:'),'mHC 两条互斥分支必须在同一选择列');
 assert.ok(retained.includes('原 RNG')&&retained.includes('discard'),'下一列必须是共同保存/丢弃状态');
 assert.ok(!retained.includes('dropout&gt;0'),'不得把互斥分支 B 误画成 A 的下一生命周期阶段');
 for(const [name,expected] of Object.entries(figures)){
  assert.ok(p.includes(`assets/${name}`),`正文未引用 ${name}`);
  assert.equal(normalizeEol(await readFile(join(pageDir,'assets',name),'utf8')),normalizeEol(expected),`图/生成器漂移 ${name}`);
 }
});

test('actual browser render: cell containment, text overlap, screenshots',async()=>{
 const cache=join(homedir(),'.cache/puppeteer/chrome');
 const candidates=[process.env.PUPPETEER_EXECUTABLE_PATH];
 if(existsSync(cache))for(const x of readdirSync(cache).sort().reverse())candidates.push(join(cache,x,'chrome-win64/chrome.exe'));
 candidates.push('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe','/usr/bin/chromium');
 const executablePath=candidates.find(p=>p&&existsSync(p));
 assert.ok(executablePath,'需要真实浏览器，禁止静默跳过渲染');
 const {default:puppeteer}=await import(pathToFileURL(join(root,'tools/mkdocs-site/node_modules/puppeteer-core/lib/esm/puppeteer/puppeteer-core.js')));
 const browser=await puppeteer.launch({executablePath,headless:true,args:['--disable-gpu','--no-sandbox']});
 const out=join(tmpdir(),'megatron-feature-16-18-20260905/recompute-render');
 await mkdir(out,{recursive:true});
 try{
  const p=await browser.newPage();
  for(const name of Object.keys(figures)){
   await p.goto(pathToFileURL(join(pageDir,'assets',name)).href);
   const result=await p.evaluate(()=>{
    const svg=document.querySelector('svg'),view=svg.viewBox.baseVal;
    const issues=[];
    for(const cell of document.querySelectorAll('[data-cell]')){
     const r=cell.querySelector('rect').getBBox();
     const texts=[...cell.querySelectorAll('text')].map(x=>({s:x.textContent,b:x.getBBox()}));
     for(const {s,b} of texts)if(b.x<r.x||b.y<r.y||b.x+b.width>r.x+r.width||b.y+b.height>r.y+r.height)issues.push('cell overflow '+s);
     for(let i=0;i<texts.length;i++)for(let j=i+1;j<texts.length;j++){
      const a=texts[i].b,b=texts[j].b;
      if(a.x<b.x+b.width&&a.x+a.width>b.x&&a.y<b.y+b.height&&a.y+a.height>b.y)issues.push('text overlap '+texts[i].s);
     }
    }
    for(const t of document.querySelectorAll('text')){const b=t.getBBox();if(b.x<0||b.y<0||b.x+b.width>view.width||b.y+b.height>view.height)issues.push('canvas overflow '+t.textContent);}
    return {issues,width:view.width,height:view.height};
   });
   assert.deepEqual(result.issues,[],`${name} 排版错误`);
   await p.setViewport({width:result.width,height:result.height,deviceScaleFactor:1});
   await p.screenshot({path:join(out,name.replace('.svg','.png')),fullPage:true});
  }
  await writeFile(join(out,'render-result.json'),JSON.stringify({figures:Object.keys(figures),cellOverflow:0,textOverlap:0},null,2));
  process.stdout.write(`rendered screenshots: ${out}\n`);
 }finally{await browser.close();}
});

// Git checkouts may use CRLF; compare SVG content with only line endings normalized.
function normalizeEol(text) { return text.replace(/\r\n/g, '\n'); }
