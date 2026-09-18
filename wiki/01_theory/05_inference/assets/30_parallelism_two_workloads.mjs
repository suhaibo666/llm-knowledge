// Regenerate with: node wiki/01_theory/05_inference/assets/30_parallelism_two_workloads.mjs
import {readFileSync,writeFileSync} from 'node:fs';
import {dirname,join} from 'node:path';
import {fileURLToPath} from 'node:url';
const dir=dirname(fileURLToPath(import.meta.url));
const kvBytes=2*16*1*64*2, longTokens=128*1024, shortTokens=2*1024;
const longMiB=kvBytes*longTokens/2**20, shortMiB=kvBytes*shortTokens/2**20;
const longPerGpu=longMiB/8, shortPerGpu=shortMiB*(64/8);
if(longMiB!==512||shortMiB!==8||longPerGpu!==64||shortPerGpu!==64) throw new Error('KV ledger changed');
const page=readFileSync(join(dir,'../30_inference_parallelism_composition_analysis.md'),'utf8');
for(const claim of ['$512/8=64\\ \\mathrm{MiB}$','$8\\cdot8=64\\ \\mathrm{MiB}$','$256\\ \\mathrm{MiB}$']) if(!page.includes(claim)) throw new Error(`Page lost computed claim: ${claim}`);
const E=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
const t=(x,y,s,c='text',a='start')=>`<text class="${c}" x="${x}" y="${y}" text-anchor="${a}">${E(s)}</text>`;
const r=(x,y,w,h,c='card')=>`<rect class="${c}" x="${x}" y="${y}" width="${w}" height="${h}" rx="8"/>`;
const p=[];
p.push(t(38,45,'同八卡、同 512 MiB 逻辑 KV：并行轴随请求形状变化','title'));
p.push(t(38,73,'教学模型 · 16 层 / 1 KV head / d=64 / fp16 K 与 V / 每卡 256 MiB 持久 KV 预算','small'));
p.push(r(38,104,1124,305,'panel'));
p.push(t(62,138,'A · 一个 128K 长请求','head'));
p.push(t(62,163,'候选布局：PCP2 × TP4；DCP4 在每行复用 TP ranks','small'));
for(let row=0;row<2;row++){
  p.push(t(70,223+row*87,`PCP ${row}`,'head'));
  for(let col=0;col<4;col++){
    const x=184+col*227,y=184+row*87;
    p.push(r(x,y,206,72,'blue'));
    p.push(t(x+103,y+28,`TP / DCP rank ${col}`,'body','middle'));
    p.push(t(x+103,y+53,`持久 KV ${longPerGpu} MiB`,'small','middle'));
  }
  for(let col=0;col<3;col++) p.push(t(401+col*227,228+row*87,'↔','comm','middle'));
}
p.push(t(130,275,'↕','comm','middle'));
p.push(t(184,379,'同一请求：PCP 跨行分 Q；DCP 在行内分 KV 并合并局部 attention 统计','small'));
p.push(r(38,428,1124,175,'panel'));
p.push(t(62,462,'B · 六十四个 2K 短请求','head'));
p.push(t(62,487,'候选布局：DP8；每副本 8 个独立请求','small'));
for(let col=0;col<8;col++){
  const x=68+col*136;
  p.push(r(x,507,124,72,'ghost'));
  p.push(t(x+62,535,`DP ${col}`,'body','middle'));
  p.push(t(x+62,557,`8 请求 · ${shortPerGpu} MiB`,'small','middle'));
}
p.push(r(226,623,748,58,'orange'));
p.push(t(600,649,'每卡持久 KV 均为 64 MiB；A 的同一请求须跨卡通信，B 的请求彼此独立','body','middle'));
p.push(t(600,670,'临时聚合、权重和实际性能仍需另算；网格本身不证明 backend 支持','small','middle'));
const style=`<style>text{font-family:Inter,"Noto Sans CJK SC","PingFang SC",sans-serif;fill:#162033}.title{font-size:25px;font-weight:750}.head{font-size:17px;font-weight:700}.body{font-size:14px}.small{font-size:12.5px;fill:#53627a}.comm{font-size:20px;fill:#1757a5;font-weight:700}.panel{fill:#fff;stroke:#b7c4d4;stroke-width:1.4}.blue{fill:#e8f2ff;stroke:#1757a5;stroke-width:1.4}.ghost{fill:#f6f8fb;stroke:#b7c4d4;stroke-width:1.2}.orange{fill:#fff1db;stroke:#a65e0c;stroke-width:1.4}</style>`;
writeFileSync(join(dir,'30_parallelism_two_workloads.svg'),`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 705" role="img">${style}<rect width="1200" height="705" fill="#fff"/>${p.join('')}</svg>\n`);
console.log(`Generated workload figure: ${longMiB} MiB long, ${shortMiB} MiB/request short, ${longPerGpu} MiB/GPU both layouts.`);
