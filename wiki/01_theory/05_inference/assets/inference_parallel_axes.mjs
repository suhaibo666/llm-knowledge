// Regenerate with: node wiki/01_theory/05_inference/assets/inference_parallel_axes.mjs
import { writeFileSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const dir = dirname(fileURLToPath(import.meta.url));
const esc = (s) => String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const bg = '#ffffff', ink = '#162033', muted = '#53627a', lineColor = '#b7c4d4', blue = '#e8f2ff', blueDark = '#1757a5', orange = '#fff1db', orangeDark = '#a65e0c';
const style = `<style>text{font-family:Inter,"Noto Sans CJK SC","PingFang SC",sans-serif;fill:${ink}}.title{font-size:26px;font-weight:750}.head{font-size:17px;font-weight:700}.body{font-size:15px}.small{font-size:13px;fill:${muted}}.num{font-variant-numeric:tabular-nums}.box{fill:#fff;stroke:${lineColor};stroke-width:1.5}.blue{fill:${blue};stroke:${blueDark};stroke-width:1.5}.orange{fill:${orange};stroke:${orangeDark};stroke-width:1.5}.ghost{fill:#f6f8fb;stroke:${lineColor};stroke-width:1}.arrow{stroke:${blueDark};stroke-width:2;fill:none;marker-end:url(#arrow)}.aux{stroke:${lineColor};stroke-width:1.5;fill:none;marker-end:url(#arrowGray)}.dash{stroke-dasharray:5 5}</style>`;
const defs = `<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9" fill="${blueDark}"/></marker><marker id="arrowGray" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9" fill="${lineColor}"/></marker></defs>`;
const svg = (w,h,body) => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" role="img">${style}${defs}<rect width="${w}" height="${h}" fill="${bg}"/>${body}</svg>\n`;
const r = (x,y,w,h,cls='box',rad=9) => `<rect class="${cls}" x="${x}" y="${y}" width="${w}" height="${h}" rx="${rad}"/>`;
const t = (x,y,s,cls='body',anchor='start') => `<text class="${cls}" x="${x}" y="${y}" text-anchor="${anchor}">${esc(s)}</text>`;
const path = (d,cls='arrow') => `<path class="${cls}" d="${d}"/>`;

export function pcpFigure() {
  const owned = [[0,1],[2,3]];
  const visible = Array.from({length:4},(_,i)=>Array.from({length:i+1},(_,j)=>j));
  const scores = owned.map(ids=>ids.reduce((sum,i)=>sum+visible[i].length,0));
  const parts = [t(40,48,'PCP：局部 Query，临时全量 KV','title'),t(40,77,'教学例子 · 4 个 prompt 位置 · 2 张卡 · 因果可见集合','small')];
  const cards = [40,625];
  owned.forEach((ids,rank)=>{
    const x=cards[rank];
    parts.push(r(x,115,535,382,'box'),t(x+22,145,`Rank ${rank} · 本地生成位置 ${ids.join('、')}`,'head'));
    ids.forEach((i,k)=>{
      const y=174+k*67;
      parts.push(r(x+22,y,190,47,'blue'),t(x+36,y+29,`Q${i} / K${i} / V${i}`,'body'));
      parts.push(path(`M${x+216} ${y+24} H${x+246}`,'aux'));
      parts.push(r(x+252,y,260,47,'ghost'),t(x+265,y+29,`q${i} ← K/V ${visible[i].join('、')}`,'body'));
    });
    parts.push(r(x+22,328,490,62,'orange'),t(x+37,355,`有效分数项：${ids.map(i=>visible[i].length).join(' + ')} = ${scores[rank]}`,'head'),t(x+37,377,'后段 Query 的历史更长','small'));
    parts.push(r(x+22,411,490,60,'blue'),t(x+37,438,'当前层 All-gather K/V：临时可见 0、1、2、3','body'),t(x+37,458,'持久 KV 仍可按位置分片','small'));
  });
  parts.push(path('M307 503 C307 535 460 542 490 568','arrow'));
  parts.push(path('M893 503 C893 535 740 542 710 568','arrow'));
  parts.push(t(600,546,'位置顺序恢复','small','middle'));
  parts.push(r(320,568,560,62,'blue'),t(600,596,'输出顺序：o0，o1，o2，o3','head','middle'),t(600,617,'完成条件：远端 KV 到齐 → 本地 Q 算完 → 顺序恢复','small','middle'));
  parts.push(t(600,672,'本例连续切分负载 3 : 7；head-tail 可变成 5 : 5，但需重排位置。','small','middle'));
  return svg(1200,700,parts.join(''));
}

export function dcpStats() {
  const scores=[0,Math.log(2),0,0], values=[0,2,4,6], groups=[[0,2],[1,3]];
  const local=groups.map(ids=>{
    const m=Math.max(...ids.map(i=>scores[i]));
    const ell=ids.reduce((a,i)=>a+Math.exp(scores[i]-m),0);
    const u=ids.reduce((a,i)=>a+Math.exp(scores[i]-m)*values[i],0);
    return {ids,m,ell,u,o:u/ell};
  });
  const m=Math.max(...local.map(x=>x.m));
  const ell=local.reduce((a,x)=>a+Math.exp(x.m-m)*x.ell,0);
  const u=local.reduce((a,x)=>a+Math.exp(x.m-m)*x.u,0);
  const direct=values.reduce((a,v,i)=>a+Math.exp(scores[i])*v,0)/scores.reduce((a,s)=>a+Math.exp(s),0);
  const naive=local.reduce((a,x)=>a+x.o,0)/local.length;
  if(Math.abs(u/ell-direct)>1e-12) throw new Error('DCP recombination failed');
  return {local,m,ell,u,result:u/ell,direct,naive};
}
export function dcpFigure() {
  const x=dcpStats();
  const parts=[t(40,48,'DCP：序列分片，按全局分母合并','title'),t(40,78,'同一个 q3 · 分数 0、ln2、0、0 · value 0、2、4、6','small')];
  [40,625].forEach((left,rank)=>{
    const p=x.local[rank];
    parts.push(r(left,112,535,265,'box'),t(left+22,146,`Rank ${rank} · KV 位置 ${p.ids.join('、')}`,'head'));
    const vals=p.ids.map(i=>`k${i}: s=${i===1?'ln2':'0'}, v=${[0,2,4,6][i]}`);
    vals.forEach((v,i)=>{parts.push(r(left+22,169+i*50,491,40,'blue'),t(left+36,195+i*50,v,'body'));});
    const stats=rank===0?'m=0，ℓ=2，u=4，局部 o=2':'m=ln2，ℓ=3/2，u=5，局部 o=10/3';
    parts.push(r(left+22,285,491,65,'orange'),t(left+36,314,stats,'body'),t(left+36,338,rank===0?'合并时该卡权重 × 1/2':'合并时该卡权重 × 1','small'));
  });
  parts.push(path('M307 382 C307 430 475 445 510 464','arrow'),path('M892 382 C892 430 725 445 690 464','arrow'));
  parts.push(r(330,466,540,100,'blue'),t(600,496,'共同最大值 m=ln2','head','middle'),t(600,522,`ℓ = 5/2，u = 7，o3 = ${x.result.toFixed(1)} = 14/5`,'body','middle'),t(600,548,'完整四项权重 1、2、1、1，直接计算也得 14/5','small','middle'));
  parts.push(r(278,593,644,53,'ghost'),t(600,625,'不能简单平均局部输出： (2 + 10/3)/2 = 8/3','body','middle'));
  parts.push(t(600,681,'当前 token 位置 3 的 KV 存 Rank 1；下一轮须等本轮 KV 写入及跨卡合并完成。','small','middle'));
  return svg(1200,707,parts.join(''));
}

export function cppSchedule(P=3,M=4) {
  const slots=M+P-1;
  const grid=Array.from({length:P},()=>Array(slots).fill(null));
  for(let s=0;s<P;s++) for(let j=0;j<M;j++) grid[s][s+j]=j+1;
  const busy=grid.flat().filter(Boolean).length, bubbles=P*slots-busy;
  if(busy!==P*M || bubbles!==P*(P-1)) throw new Error('CPP schedule count failed');
  return {grid,slots,busy,bubbles,ratio:bubbles/(P*slots)};
}
export function cppFigure() {
  const x=cppSchedule(), parts=[t(40,48,'CPP：3 个层 Stage × 4 个连续 Chunk','title'),t(40,78,'教学假设：每个 Stage/Chunk 任务耗时相同，跨级激活在格边界就绪','small')];
  const gx=210,gy=178,cw=145,ch=92;
  for(let slot=0;slot<x.slots;slot++){
    const px=gx+slot*cw;
    parts.push(t(px+cw/2,145,`时间格 ${slot+1}`,'head','middle'));
  }
  x.grid.forEach((row,s)=>{
    parts.push(t(165,gy+s*ch+51,`Stage ${s}`,'head','end'));
    row.forEach((chunk,slot)=>{
      const px=gx+slot*cw,py=gy+s*ch;
      parts.push(r(px+3,py+3,cw-8,ch-8,chunk?'blue':'ghost',5));
      if(chunk){parts.push(t(px+cw/2,py+38,`C${chunk}`,'head','middle'),t(px+cw/2,py+62,`本地 KV 至 C${chunk}`,'small','middle'));}
      else parts.push(t(px+cw/2,py+52,'空','small','middle'));
    });
  });
  // Highlight the two independent prerequisites of Stage 1 / Chunk 2.
  parts.push(path('M490 264 L503 274','arrow'));
  parts.push(path('M493 316 L503 316','arrow'));
  parts.push(r(75,497,550,105,'box'),t(100,531,'S1 的 C2 在格 3 才能开始','head'),t(100,558,'须等 S0 的 C2 激活（格 2）','body'),t(100,581,'也须等 S1 的 C1 历史 KV（格 2）','body'));
  parts.push(r(651,497,465,105,'orange'),t(675,531,'完成边界：S2 的 C4，格 6','head'),t(675,558,`忙格 ${x.busy}，空格 ${x.bubbles}，理想气泡 ${Math.round(x.ratio*1000)/10}%`,'body'),t(675,581,'等时均衡假设；后块实际可能更慢','small'));
  parts.push(t(596,652,'横向：同一 Stage 先前 Chunk 写 KV　　纵向：同一 Chunk 从上游 Stage 收激活','small','middle'));
  return svg(1190,680,parts.join(''));
}

const files=[['27_pcp_query_kv_layout.svg',pcpFigure()],['28_dcp_kv_merge.svg',dcpFigure()],['29_cpp_stage_chunk_timeline.svg',cppFigure()]];
for(const [name,content] of files) writeFileSync(join(dir,name),content);

// Regress the numerical labels against the actual page, not only the SVG.
const md27=readFileSync(join(dir,'../27_prefill_context_parallelism_analysis.md'),'utf8');
const md28=readFileSync(join(dir,'../28_decode_context_parallelism_analysis.md'),'utf8');
const md29=readFileSync(join(dir,'../29_chunked_pipeline_parallelism_analysis.md'),'utf8');
if(!md27.includes('$1+2=3$') || !md27.includes('$3+4=7$') || !md27.includes('$1+4=5$')) throw new Error('PCP prose/example mismatch');
if(!md28.includes('$14/5$') || !md28.includes('$8/3$') || Math.abs(dcpStats().result-14/5)>1e-12 || Math.abs(dcpStats().naive-8/3)>1e-12) throw new Error('DCP prose/example mismatch');
if(!md29.includes('空格为 $6$') || !md29.includes('空格比例 $1/3$') || cppSchedule().bubbles!==6) throw new Error('CPP prose/schedule mismatch');
console.log('Generated 3 inference parallelism figures; page/example checks passed.');
