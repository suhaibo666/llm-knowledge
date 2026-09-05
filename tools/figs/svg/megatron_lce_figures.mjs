// Figure specification: same X[2,2], W[8,2], labels[2] for all lanes.
// Figure 1: explicit probability cells, chunk statistics and backward consumption.
// Figure 2: DP/TP/SP ownership, block-MAX then two SUMs, dX reduction/slicing.
// Gray = existing tensors; blue = reconstructible statistics; orange = retained/replayed work.
// This is a real-arithmetic teaching model, NOT a CuTe launch/performance simulator.
import {writeFileSync} from 'node:fs';
import {fileURLToPath, pathToFileURL} from 'node:url';

export const example={X:[[1,0],[0,1]],W:Array.from({length:8},(_,j)=>[Math.log(j+1),Math.log(8-j)]),labels:[6,1],chunk:2};
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
export function stats(z){const m=Math.max(...z);return {m,a:z.reduce((s,x)=>s+Math.exp(x-m),0)};}
export function merge(a,b){const m=Math.max(a.m,b.m);return {m,a:a.a*Math.exp(a.m-m)+b.a*Math.exp(b.m-m)};}
export function evaluate({X=example.X,W=example.W,labels=example.labels,chunk=2,p=1,reduction='sum',ignore=-100}={}){
  if(W.length%p||!Number.isInteger(chunk)||chunk<1)throw Error('equal vocab shards and positive chunks required');
  if(!['none','sum','mean'].includes(reduction))throw Error('invalid reduction');
  const Z=X.map(x=>W.map(w=>dot(x,w))), vp=W.length/p,valid=labels.filter(y=>y!==ignore).length;
  const ranks=Array.from({length:p},(_,r)=>({rank:r,rows:Array.from({length:vp},(_,j)=>r*vp+j),blocks:Z.map(z=>Array.from({length:Math.ceil(vp/chunk)},(_,k)=>stats(z.slice(r*vp+k*chunk,r*vp+Math.min((k+1)*chunk,vp)))))}));
  const totals=Z.map((_,i)=>ranks.flatMap(r=>r.blocks[i]).reduce(merge,{m:0,a:0}));
  const P=Z.map((row,i)=>row.map(z=>Math.exp(z-totals[i].m)/totals[i].a));
  const losses=Z.map((row,i)=>labels[i]===ignore?0:totals[i].m+Math.log(totals[i].a)-row[labels[i]]);
  const scale=reduction==='mean'?1/valid:1;
  const D=P.map((row,i)=>row.map((v,j)=>labels[i]===ignore?0:scale*(v-Number(j===labels[i]))));
  const localDX=ranks.map(r=>X.map((_,i)=>X[0].map((_,h)=>r.rows.reduce((s,j)=>s+D[i][j]*W[j][h],0))));
  const dX=X.map((_,i)=>X[0].map((_,h)=>localDX.reduce((s,r)=>s+r[i][h],0)));
  const dW=W.map((_,j)=>X[0].map((_,h)=>X.reduce((s,x,i)=>s+D[i][j]*x[h],0)));
  const K=ranks[0].blocks[0].length;
  const reducedBlockMax=Z.map((_,i)=>Array.from({length:K},(_,k)=>Math.max(...ranks.map(r=>r.blocks[i][k].m))));
  const localAcc=ranks.map(r=>Z.map((_,i)=>r.blocks[i].reduce((s,b)=>s+b.a*Math.exp(b.m-totals[i].m),0)));
  return {Z,P,D,totals,losses,loss:losses.reduce((a,b)=>a+b,0)*scale,ranks,localDX,dX,dW,reducedBlockMax,localAcc,valid,collectives:p>1?[X.length*K,X.length,X.length]:[],spOwners:X.map((_,i)=>Math.floor(i*p/X.length))};
}
export function memory({N=8192,V=152064,d=8192,p=8,Cf=3072,Cb=3072}={}){
  const vp=V/p,K=Math.ceil(vp/Cf),M=2**20;
  return {N,V,d,p,Cf,Cb,vp,K,globalLogits:2*N*V/M,logits:2*N*vp/M,softmax:4*N*vp/M,hidden:2*N*d/M,localHidden:2*N*d/p/M,finalStats:8*N/M,partialStats:8*N*K/M,maxBackup:4*N*K/M,dLogits:2*N*Cb/M,dHidden:4*N*d/M,deltaHidden:4*N*d/M,dWeight:2*vp*d/M};
}
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
function canvas(w,h,title){const parts=[`<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img"><title>${esc(title)}</title><style>text{font-family:'Microsoft YaHei','Noto Sans CJK SC',sans-serif;fill:#233044;font-size:17px}.title{font-size:25px;font-weight:700}.label{font-weight:700}.small{font-size:15px}.neutral{fill:#fff;stroke:#cbd5e1}.ghost{fill:#f8fafc;stroke:#e2e8f0}.acc1{fill:#eff6ff;stroke:#60a5fa}.acc2{fill:#fff7ed;stroke:#fb923c}.arrow{stroke:#64748b;stroke-width:2;fill:none;marker-end:url(#a)}</style><defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10z" fill="#64748b"/></marker></defs><rect width="100%" height="100%" fill="white"/>`];
  return {text(x,y,s,c=''){parts.push(`<text x="${x}" y="${y}" class="${c}">${esc(s)}</text>`);},rect(x,y,w,h,c='neutral'){parts.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="5" class="${c}"/>`);},arrow(x1,y1,x2,y2){parts.push(`<path d="M${x1} ${y1} L${x2} ${y2}" class="arrow"/>`);},done(){return parts.join('')+'</svg>';}};
}
const list=(a,n=3)=>a.map(x=>Number(x.toFixed(n))).join(', ');
function cellRow(c,x,y,vals,{denom=null,target=-1,width=57}={}){vals.forEach((v,j)=>{c.rect(x+j*width,y,width-4,32,j===target?'acc2':'neutral');c.text(x+j*width+8,y+22,String(v));});if(denom)c.text(x+vals.length*width+4,y+23,`/ ${denom}`);}

export function mechanismSVG(){const e=evaluate(), c=canvas(1180,1030,'同一交叉熵目标：保存完整概率，或保存统计量再重算');
  c.text(25,38,'从完整概率留存，到分块重建', 'title');
  c.text(25,69,`X₀=(1,0), X₁=(0,1)；Wⱼ=(ln(j+1),ln(8−j))；标签 y=(${example.labels})`);
  c.text(25,96,'数学例子：N=2, d=2, V=8；橙格是标签项。实际 GPU 形状须满足对齐条件。','small');
  c.text(25,139,'exp(Z₀)');cellRow(c,145,116,e.Z[0].map(x=>Math.round(Math.exp(x))),{target:6});
  c.text(25,181,'exp(Z₁)');cellRow(c,145,158,e.Z[1].map(x=>Math.round(Math.exp(x))),{target:1});
  c.arrow(670,151,724,151);c.rect(745,110,408,94,'acc1');c.text(761,139,`每行 Σexp(Z)=${Math.round(Math.exp(e.totals[0].m)*e.totals[0].a)}；目标 exp(Zᵧ)=7`);c.text(761,170,`loss = ln(36/7) = ${e.losses[0].toFixed(6)}`);
  const lanes=[
    ['普通 CE · TP=2','XWᵀ → 每卡 Z[2,4]','MAX[2] → SUM[2] → SUM[2]','保留 P[2,4] FP32 → dZ；输出层求本地 dW、归约 dX'],
    ['native · TP=2','XWᵀ → 每卡 Z[2,4]','MAX[2] → SUM[4]（目标项＋指数和）','保留 P[2,4] FP32 → BF16 dZ；输出层求本地 dW、归约 dX'],
    ['te · TP=2','XWᵀ → 每卡 Z[2,4]','TE CE(logits, labels, TP group) → loss','依赖 autograd → dZ → dX / dW；内部未核验'],
  ];
  lanes.forEach((l,i)=>{const y=230+i*93;c.rect(25,y,1128,81,i===2?'ghost':'neutral');c.text(40,y+25,l[0],'label');c.text(240,y+25,l[1]);c.text(670,y+25,l[2],'small');c.text(240,y+58,l[3]);});
  c.text(40,525,'TE：训练入口禁用；图中的 loss 表示目标契约，不是运行验证。','small');
  c.rect(25,546,1128,282,'acc1');c.text(40,575,'linear 前向 · 无 TP：每两列一块，只写块统计量','label');
  e.ranks[0].blocks[0].forEach((b,k)=>{let x=45+k*215;c.rect(x,594,198,86);c.text(x+10,619,`词表 ${2*k}–${2*k+1}`);c.text(x+10,645,`m=ln${Math.round(Math.exp(b.m))}, a=${b.a.toFixed(3)}`);c.text(x+10,670,`指数和=${Math.round(Math.exp(b.m)*b.a)}`,'small');});
  c.text(934,622,'同样处理 X₁');c.text(934,650,'得到逆序块','small');
  c.arrow(450,688,450,715);c.text(40,748,`合并：m=ln${Math.round(Math.exp(e.totals[0].m))}，a=${e.totals[0].a}；q=ln7 → 两份 loss=${e.losses[0].toFixed(6)}`);
  c.text(40,783,'ctx：X、W、y、m、a、有效数；完整 Z / P 不保留','label');
  c.text(40,812,'合并不变量：exp(m′)·a′ = exp(m₁)·a₁ + exp(m₂)·a₂','small');
  c.rect(25,846,1128,161,'acc2');c.text(40,875,'linear 反向 · sum 目标：重算 Z 块 → P 块 → D 块 → 立即求梯度','label');
  c.text(40,916,'D₀ =');cellRow(c,115,893,e.D[0].map(x=>Math.round(x*36)),{denom:36,width:57});
  c.text(678,918,'每块 dX += DₖWₖ；dWₖ = DₖᵀX');
  c.text(40,953,`D₁ = (${e.D[1].map(x=>Math.round(x*36)).join(', ')}) / 36`);
  c.text(40,984,'支付：HBM 中的 D 块 + FP32 dX/ΔX；输出头矩阵乘 3 遍 → 4 遍','small');
  return c.done();
}
export function parallelSVG(){const e=evaluate({p:2}),single=evaluate(),c=canvas(1180,1040,'linear 的 DP TP SP 前反向所有权');
  c.text(25,38,'同一 loss，三种 hidden 所有权', 'title');c.text(25,71,'N=2, d=2, V=8, C=2；W₀₋₃ 属 rank 0，W₄₋₇ 属 rank 1；标签 (6,1) 在 TP 内复制');
  c.rect(25,92,1128,116,'ghost');c.text(40,121,'无 TP / 一个 DP 副本内部','label');c.text(40,152,'X[2,2] + W[8,2] → 4 个词表块 → 合并统计 → 两份 loss');c.text(40,186,`反向：4 块 D → dX[2,2]、dW[8,2]；本核无 DP collective（loss=${single.losses[0].toFixed(6)}）`);
  c.text(25,246,'TP 前向：两卡有完整 X；先按块编号 MAX，再重标定本地指数和','label');
  e.ranks.forEach((r,i)=>{let x=25+i*582;c.rect(x,264,546,201);c.text(x+15,293,`rank ${i}：X₀、X₁；W 行 ${r.rows[0]}–${r.rows.at(-1)}`,'label');r.blocks.forEach((bs,t)=>{c.text(x+15,328+t*45,`X${t} 的块 m：(${bs.map(b=>'ln'+Math.round(Math.exp(b.m))).join(', ')})`);c.text(x+15,352+t*45,`块 a：(${list(bs.map(b=>b.a))})`,'small');});c.text(x+15,447,'先备份本地 m，才能用原缩放恢复指数和','small');});
  c.arrow(297,473,297,495);c.arrow(870,473,870,495);
  c.rect(25,508,1128,143,'acc1');c.text(40,537,`① MAX[2,2]：X₀ → (${e.reducedBlockMax[0].map(x=>'ln'+Math.round(Math.exp(x))).join(', ')})；X₁ → (${e.reducedBlockMax[1].map(x=>'ln'+Math.round(Math.exp(x))).join(', ')})`,'label');
  c.text(40,571,`② 专用流 SUM 目标项[2]：rank 0=(0,ln7)，rank 1=(ln7,0)`);
  c.text(40,601,`③ 当前流 SUM 指数和[2]：rank 0=(${list(e.localAcc[0])})，rank 1=(${list(e.localAcc[1])})`);
  c.text(40,632,`事件等待完成 → 每卡 (m=ln8,a=${e.totals[0].a},q=ln7) → loss=(${e.losses.map(x=>x.toFixed(6)).join(', ')})`,'small');
  c.rect(25,672,1128,127,'acc2');c.text(40,700,'TP 反向：每卡收到相同 g=(1,1)，只计算本地词表的梯度','label');c.text(40,732,`rank 0 的 dX₀=(${list(e.localDX[0][0])})；rank 1 的 dX₀=(${list(e.localDX[1][0])})`);
  c.text(40,764,`SUM all-reduce FP32 dX[2,2] → dX₀=(${list(e.dX[0])})；dW[4,2] 各留原 owner`);
  c.rect(25,820,1128,192);c.text(40,851,'SP 在上述 TP 路径外，再增加聚合与切回','label');
  c.text(40,889,'入口：rank 0 只有 X₀[1,2]');c.text(630,889,'入口：rank 1 只有 X₁[1,2]');
  c.arrow(240,895,240,909);c.arrow(845,895,845,909);c.text(395,890,'all-gather → 两卡复制','small');c.text(40,935,'两卡保存 global_hidden[2,2] → 执行上述三次前向归约与完整 dX all-reduce');
  c.text(40,958,'反向归约后：rank 0 取 dX₀ 并 clone；rank 1 取 dX₁ 并 clone → 转回输入 dtype');
  c.text(40,990,'边界：源码是 all-reduce 后切片；SP 不把词表核限制为仅处理本地 token。','small');
  return c.done();
}
export function render(){return {'megatron_lce_mechanism.svg':mechanismSVG(),'megatron_lce_parallel.svg':parallelSVG()};}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){for(const [name,svg] of Object.entries(render()))writeFileSync(fileURLToPath(new URL(`../../../wiki/02_engineering/02_train_frameworks/megatron-lm/assets/${name}`,import.meta.url)),svg);}
