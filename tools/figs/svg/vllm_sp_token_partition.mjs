#!/usr/bin/env node
// Conceptual rank × token layout, not a numerical simulation.
// Source: vllm @ 199cb9b964822e59ab9b58d88e7be31eb419a2ae
// vllm/compilation/passes/fusion/sequence_parallelism.py:
// _SequenceParallelPatternHelper; FirstAllReduceRMSNormPattern.register.
// Pattern returns (rmsnorm, all_reduce); replacement returns
// (all_gather(rmsnorm(reduce_scatter(input))), reduce_scatter(input)).
// Example inputs reuse the page's vectors; p1 here is a GEMM contribution,
// not an existing residual. Every cell is a full H-dimensional token vector.
// Generate: node tools/figs/svg/vllm_sp_token_partition.mjs
import {mkdir, writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
const target = new URL('../../../wiki/02_engineering/03_infer_frameworks/vllm/assets/vllm_sp_token_partition.svg', import.meta.url);
const W=1400, H=920, xs=[40,380,720,1060];
const out=[];
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
const text=(x,y,s,c='body',anchor='start')=>out.push(`<text class="${c}" x="${x}" y="${y}" text-anchor="${anchor}">${esc(s)}</text>`);
const rect=(x,y,w,h,c,rx=7)=>out.push(`<rect class="${c}" x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}"/>`);
const sub=['₀','₁','₂','₃'];
function grid(x,y,kind,{shard=false,accent=false}={}){
  for(let t=0;t<4;t++) text(x+59+t*57,y-12,`t${t}`,'dim','middle');
  for(let r=0;r<2;r++){
    text(x,y+r*44+24,`R${r}`,'rank');
    for(let t=0;t<4;t++){
      const owned=!shard||(r===0?t<2:t>=2);
      const cx=x+34+t*57;
      rect(cx,y+r*44,51,36,owned?(accent?'cell acc1':'cell neutral'):'cell ghost',5);
      const label=owned?(kind==='p'?`p${sub[r]}`:`${kind}${sub[t]}`):'—';
      text(cx+25.5,y+r*44+24,label,owned?'value':'dim','middle');
    }
  }
}
function arrow(from,to,y,label,main=false){
 const start=from+279,end=to-13;
 out.push(`<path class="arrow ${main?'main':'aux'}" d="M ${start} ${y} H ${end}" marker-end="url(#${main?'blue':'gray'})"/>`);
 text((start+end)/2,y-13,label,main?'arrow-label blue':'arrow-label','middle');
}
function stage(x,y,title,dim){text(x,y,title,'stage');text(x,y+25,dim,'dim');}
out.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="title desc">`);
out.push('<title id="title">SP 如何按 token 分摊 RMSNorm，并分别返回 y 与 residual u</title>');
out.push('<desc id="desc">T=4、H=4、TP=2 的教学布局。上方 AllReduce 后，两 rank 对全部 token 重复 RMSNorm，返回完整 y 和 u。下方 ReduceScatter 后，R0 只处理 t0、t1，R1 只处理 t2、t3；AllGather 恢复两 rank 的完整 y，residual u 仍为每 rank 2×4。每个格子包含一个 token 的全部 hidden 列。</desc>');
out.push(`<style>
text{font-family:"Segoe UI","Microsoft YaHei","PingFang SC",system-ui,sans-serif;fill:#2A313B}
.title{font-size:25px;font-weight:700}.panel-title{font-size:21px;font-weight:700}.stage{font-size:18px;font-weight:650}
.body{font-size:15px}.dim{font-size:14px;fill:#697280}.rank{font-size:15px;font-weight:650}.value{font-size:18px}
.arrow-label{font-size:13px;fill:#697280}.blue{fill:#2563EB}.orange{fill:#8A4A11;font-size:16px}.note{font-size:15px;fill:#173F87}
.neutral{fill:#fff;stroke:#C7CCD3;stroke-width:1.1}.ghost{fill:#F4F5F7;stroke:#D5DAE1;stroke-dasharray:3 3}
.acc1{fill:#EAF1FD;stroke:#2563EB;stroke-width:1.4}.acc2{fill:#FCF1E6;stroke:#C3651F;stroke-width:1.1}
.panel{fill:#FCFDFF;stroke:#D5DAE1}.arrow{fill:none}.arrow.main{stroke:#2563EB;stroke-width:2.3}.arrow.aux{stroke:#8A939F;stroke-width:1.6}
</style><defs><marker id="blue" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#2563EB"/></marker><marker id="gray" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#8A939F"/></marker></defs>`);
rect(0,0,W,H,'neutral',0);
text(28,37,'Sequence Parallelism：按 token 分摊 Norm，分别追踪 y 与 residual u','title');
text(28,69,'教学输入：T=4，H=4，TP=2；各 token 的贡献 p₀=(1,2,3,4)，p₁=(1,0,−1,−2)；w=(1,2,1,2)，ε=10⁻⁶');
text(28,95,'每格 = 一个 token 的完整 H=4 向量；uₜ = 两 rank 贡献之和，yₜ = RMSNorm(uₜ)；灰虚格 = 此 rank 不持有该 token。');
for(const [sp,py] of [[false,120],[true,493]]){
 rect(20,py,1360,366,'panel',12);
 text(40,py+31,sp?'SP 改写 · token 分片后做本地 Norm':'未改写 · 两个 rank 都做完整 Norm','panel-title');
 const y=py+67;
 stage(xs[0],y,'① GEMM partial','每 rank：4×4');
 stage(xs[1],y,sp?'② ReduceScatter':'② AllReduce',sp?'沿 token 轴 dim=0；每 rank：2×4':'求和并复制；每 rank：4×4');
 stage(xs[2],y,sp?'③ 本地 RMSNorm':'③ RMSNorm',sp?'仅处理本 rank 的 2 个 token':'各自处理全部 4 个 token');
 stage(xs[3],y,sp?'④ AllGather → 返回':'④ 返回 (y, u)',sp?'仅聚合 y，沿 token 轴 dim=0':'y、u 都是每 rank 完整张量');
 const gy=y+58;
 grid(xs[0],gy,'p');
 grid(xs[1],gy,'u',{shard:sp,accent:sp});
 grid(xs[2],gy,'y',{shard:sp,accent:sp});
 grid(xs[3],gy,'y');
 arrow(xs[0],xs[1],gy+39,'sum',sp);
 arrow(xs[1],xs[2],gy+39,'Norm',sp);
 arrow(xs[2],xs[3],gy+39,sp?'AG':'返回',sp);
 text(xs[3]+34,gy+103,'y：每 rank 4×4','dim');
 text(xs[3],gy+124,sp?'residual u：每 rank 2×4':'residual u：每 rank 4×4','stage');
 grid(xs[3],gy+150,'u',{shard:sp,accent:sp});
 text(xs[0],gy+109,'p₀、p₁：GEMM 的规约贡献');
 text(xs[0],gy+132,'这里没有已有 residual 输入。','dim');
 if(sp){
  text(xs[1],gy+109,'R0：t0、t1；R1：t2、t3','note');
  text(xs[1],gy+132,'切分 token；H 列保持完整。','dim');
  rect(xs[0],gy+161,945,62,'acc1',8);
  text(xs[0]+16,gy+186,'每个 token 只在一个 rank 上做 Norm；AllGather 后，两 rank 重新拿到全部 y₀…y₃。','note');
  text(xs[0]+16,gy+209,'返回值中的 u 不参与这次 AllGather，继续保留 token 分片。','note');
 }else{
  rect(xs[1],gy+111,600,49,'acc2',8);
  text(xs[1]+14,gy+141,'两个 rank 重复计算同一批 token 的 RMSNorm。','orange');
  text(xs[0],gy+196,'返回约定：第一个输出是 norm 结果 y；第二个输出 u 用作后续 residual。');
  text(xs[0],gy+221,'两条方案使用相同输入；差别是中间 token 的归属与 residual 的形状。','dim');
 }
}
text(28,879,'适用边界：仅示意 FirstAllReduceRMSNormPattern。H=4 是布局教学例；真实 SP 启用还受设备、hidden size 与 token 阈值约束。','dim');
text(28,902,'源码：vllm/compilation/passes/fusion/sequence_parallelism.py · FirstAllReduceRMSNormPattern.register · baseline 199cb9b96482','dim');
out.push('</svg>');
await mkdir(new URL('.',target),{recursive:true});
await writeFile(target,out.join('\n')+'\n');
console.log(fileURLToPath(target));
