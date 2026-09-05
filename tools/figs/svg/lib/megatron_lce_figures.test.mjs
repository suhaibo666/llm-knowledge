import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {example,evaluate,stats,merge,memory,render} from '../megatron_lce_figures.mjs';
const pageURL=new URL('../../../../wiki/02_engineering/02_train_frameworks/megatron-lm/24_megatron_linear_cross_entropy_analysis.md',import.meta.url);
const close=(a,b,eps=1e-9)=>assert.ok(Math.abs(a-b)<eps,`${a} != ${b}`);
const matrixClose=(a,b)=>a.forEach((row,i)=>row.forEach((v,j)=>close(v,b[i][j])));
test('块合并与完整 softmax 等价，包括负 logits 和尾块',()=>{
  for(const z of [[1,-2,3,-1,0],[-3,-4,-5],[12,0,-14,1,2,3]])for(const width of [1,2,3]){
    const blocks=Array.from({length:Math.ceil(z.length/width)},(_,i)=>stats(z.slice(i*width,(i+1)*width)));
    const s=blocks.reduce(merge,{m:0,a:0}),ref=stats(z);
    close(s.m+Math.log(s.a),ref.m+Math.log(ref.a));
    z.forEach(x=>close(Math.exp(x-s.m)/s.a,Math.exp(x-ref.m)/ref.a));
  }
});
test('实际三次 TP 归约的块 MAX、指数和与梯度 owner 恢复同一结果',()=>{
  const tp=evaluate({p:2}),dp=evaluate();assert.deepEqual(tp.collectives,[4,2,2]);
  tp.reducedBlockMax[0].forEach((x,i)=>close(Math.exp(x),[6,8][i]));
  tp.reducedBlockMax[1].forEach((x,i)=>close(Math.exp(x),[8,6][i]));
  // Replay entry/triton ordering: reduced block maxima select one reference;
  // local backup maxima reweight local sums, then SUM over ranks.
  tp.Z.forEach((_,i)=>{const m=Math.max(0,...tp.reducedBlockMax[i]);const a=tp.ranks.reduce((s,r)=>s+r.blocks[i].reduce((q,b)=>q+b.a*Math.exp(b.m-m),0),0);close(m+Math.log(a)-tp.Z[i][example.labels[i]],dp.losses[i]);});
  matrixClose(tp.dX,dp.dX);matrixClose(tp.dW,dp.dW);
  assert.deepEqual(tp.spOwners,[0,1]);assert.deepEqual(tp.ranks.map(r=>r.rows),[[0,1,2,3],[4,5,6,7]]);
  const spReturned=tp.spOwners.map((rank,i)=>tp.localDX.reduce((sum,r)=>sum.map((x,h)=>x+r[i][h]),[0,0]));
  matrixClose(spReturned,dp.dX);
});
test('重算后的 dX/dW 符合独立完整 CE 有限差分，含 mean 与 ignore',()=>{
  const objective=(X,W,labels,reduction)=>{
    let sum=0,valid=0;
    X.forEach((row,i)=>{if(labels[i]===-100)return;valid++;const z=W.map(w=>w.reduce((s,x,h)=>s+x*row[h],0));sum+=Math.log(z.reduce((s,x)=>s+Math.exp(x),0))-z[labels[i]];});
    return reduction==='mean'?sum/valid:sum;
  };
  for(const reduction of ['sum','mean'])for(const labels of [example.labels,[6,-100]]){
    const X=structuredClone(example.X),W=structuredClone(example.W),e=evaluate({X,W,labels,reduction,p:2}),eps=1e-5;
    for(const [v,grad] of [[X,e.dX],[W,e.dW]])for(let i=0;i<v.length;i++)for(let h=0;h<v[i].length;h++){
      const orig=v[i][h];v[i][h]=orig+eps;const hi=objective(X,W,labels,reduction);v[i][h]=orig-eps;const lo=objective(X,W,labels,reduction);v[i][h]=orig;close(grad[i][h],(hi-lo)/(2*eps),1e-8);
    }
  }
});
function checkPage(md){md=md.replaceAll('`','');const m=memory(),e=evaluate();assert.ok(md.includes(`**${e.losses[0].toFixed(6)}**`));assert.ok(md.includes(`**${m.globalLogits} MiB，约 2.49 GB**`));assert.ok(md.includes(`每卡词表 **${m.vp}** 项、前向 **${m.K}** 块`));
  const rows=[['普通输出头完整本地 BF16 logits',m.logits],['普通/native 留存的 FP32 softmax',m.softmax],['完整 BF16 hidden',m.hidden],['最终最大值与指数和',m.finalStats],['前向两张块统计表',m.partialStats],['反向 _d_logits 块',m.dLogits],['FP32 d_hidden',m.dHidden],['FP32 _delta_hidden',m.deltaHidden],['BF16 d_weight',m.dWeight]];
  for(const [label,value]of rows){const row=md.split(/\r?\n/).find(l=>l.startsWith(`| ${label} |`));assert.ok(row?.includes(`**${value} MiB**`),`${label}: ${value}`);}
  assert.ok(md.includes(`_max_backup **${m.maxBackup} MiB**`));
  for(const term of ['cross_entropy_loss_fusion','cross_entropy_fusion_impl','LCE_FWD_VOCAB_SPLIT_SIZE','LCE_BWD_VOCAB_SPLIT_SIZE','all-reduce 后切片','总共三次前向 all-reduce','没有零分母保护'])assert.ok(md.includes(term),term);
}
test('读取真实 Markdown，逐对象锁定显存与算例结果',()=>{const md=readFileSync(pageURL,'utf8');for(const block of md.matchAll(/^\$\$\r?\n[\s\S]*?^\$\$/gm))assert.ok(!block[0].includes('`'),'display math must not contain code markup');checkPage(md);});
test('只改正文中的一个显存数值会失败',()=>assert.throws(()=>checkPage(readFileSync(pageURL,'utf8').replace('**48 MiB**','**49 MiB**'))));
test('生成的外部 SVG 与页面所引用文件严格一致',()=>{const md=readFileSync(pageURL,'utf8');for(const [name,svg]of Object.entries(render())){assert.ok(md.includes(`](assets/${name})`));assert.equal(readFileSync(new URL(`../../../../wiki/02_engineering/02_train_frameworks/megatron-lm/assets/${name}`,import.meta.url),'utf8').replace(/\r\n/g,'\n'),svg);assert.ok(!svg.includes('NaN'));}});
