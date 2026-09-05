// Figure specification: one GEMM input is replayed in every recipe lane.
// The left column fixes identity/shape, the middle marks the TE evidence boundary,
// the right column shows output and backward obligations. Payload bars are bit counts,
// not TE allocations. The launch figure separates CPU submissions from device work;
// the slot figure is a discrete liveness replay of the pinned PP2 test order.
import { writeFile, mkdir } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
export const root = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');
export const assetDir = resolve(root, 'wiki/02_engineering/02_train_frameworks/megatron-lm/assets');
export function model() {
  const m=128,k=128,n=128, elements=m*k;
  const order=['F0','F1','B0','F2','B1','F3','B2','B3'];
  const live=new Set(), values=[], intervals=[]; let peak=0;
  for (let t=0;t<order.length;t++) {
    const id=Number(order[t].slice(1));
    if(order[t][0]==='F'){ if(live.has(id)) throw Error('duplicate forward'); live.add(id); intervals[id]={id,start:t,slot:id%2}; }
    else { if(!live.delete(id)) throw Error('backward before forward'); intervals[id].end=t+1; }
    values.push(live.size); peak=Math.max(peak,live.size);
  }
  if(live.size)throw Error('undrained schedule');
  return {m,k,n,elements,flops:2*m*k*n,bf16:elements*16/8/1024,fp8:elements*8/8/1024,fp4:elements*4/8/1024,roundtrip:elements*16/8/1024*2,order,values,intervals,peak,submissions:[['GEMM','bias','GeLU'],['GEMM','bias + GeLU'],['graph.replay']]};
}
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
const txt=(x,y,s,cls='')=>`<text x="${x}" y="${y}" class="${cls}">${esc(s)}</text>`;
const rect=(x,y,w,h,cls='neutral')=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="6" class="${cls}"/>`;
const line=(x,y,x2,y2,cls='aux')=>`<path d="M${x},${y} L${x2},${y2}" class="arrow ${cls}" marker-end="url(#head)"/>`;
function svg(w,h,title,body){return `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img"><title>${esc(title)}</title><style>text{font-family:'Microsoft YaHei','Segoe UI',sans-serif;font-size:15px;fill:#2A313B}.title{font-size:23px;font-weight:700}.small{font-size:13px;fill:#626B77}.bold{font-weight:700}.neutral{fill:#fff;stroke:#C7CCD3}.ghost{fill:#F7F6F3;stroke:#DDD9D2}.acc1{fill:#EAF1FD;stroke:#2563EB}.acc2{fill:#FCF1E6;stroke:#C3651F}.arrow{fill:none;stroke:#98A1AD;stroke-width:1.3}.main{stroke:#2563EB;stroke-width:2}.dash{fill:none;stroke:#C3651F;stroke-dasharray:5 4}</style><defs><marker id="head" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="none" stroke="#98A1AD"/></marker></defs><rect width="100%" height="100%" fill="#fff"/>${txt(26,37,title,'title')}${body}</svg>\n`;}
export function quantization(d=model()){
 const lanes=[['delayed','历史 amax → delayed scale','历史与规约状态'],['tensorwise','当前整张量 → current scale','当前统计成本'],['blockwise','分块缩放 → block recipe','尺度 + 方向布局'],['mxfp8','微缩放 → MX recipe','对齐 + 列向表示'],['FP8 custom','工厂 → CustomRecipe','数值/保存由工厂定义'],['nvfp4','NVFP4 recipe → packed bytes','scale/amax 不随 bytes 消失'],['FP4 custom','工厂 → CustomRecipe','实际位宽/算法另验']];
 let b=txt(26,65,`每条 lane：X[${d.m},${d.k}]、W[${d.k},${d.n}] → 同一 GEMM → Z[${d.m},${d.n}]`,'small');
 b+=rect(218,85,552,510,'ghost')+txt(238,109,'TE 依赖边界：这里标接口语义，不模拟 TE 内核','bold');
 b+=txt(28,109,'Megatron 选择', 'bold')+txt(802,109,'消费与反向', 'bold');
 lanes.forEach(([name,step,cost],i)=>{let y=126+i*64;b+=rect(26,y,168,51)+txt(37,y+23,name,'bold')+txt(37,y+43,'同一 X、W','small')+line(195,y+26,234,y+26,'main')+rect(240,y,510,51,i===4||i===6?'ghost':'acc1')+txt(250,y+22,step)+txt(250,y+42,cost,'small')+line(752,y+26,790,y+26,'main')+rect(800,y,278,51)+txt(812,y+22,'Z → bias/GeLU → A → loss')+txt(812,y+42,'BWD：GZ·Wᵀ 与 Xᵀ·GZ','small');});
 b+=txt(26,623,'单份操作数的名义载荷（不含 S、padding、转置、高精度副本）','bold');
 [['BF16',d.bf16,'ghost'],['FP8',d.fp8,'acc1'],['packed FP4',d.fp4,'acc2']].forEach(([name,kib,cl],i)=>{let y=642+i*41;b+=txt(26,y+22,name)+rect(157,y,kib*12,27,cl)+txt(165+kib*12,y+20,`${kib} KiB`)+txt(645,y+20,i?'另加 S；custom 不承诺此位宽':'原始表示');});
 b+=txt(26,792,`不变量：输出形状不变；前向乘加仍为 ${d.flops.toLocaleString('en-US')} FLOPs。缩放布局与反向保存由 TE/工厂决定。`,'small');
 return svg(1104,818,'量化改变表示，不抹去反向依赖',b);
}
export function launch(d=model()){
 let b=txt(26,66,`X → GEMM → Z → bias → U → GeLU → A。U 为 BF16 ${d.bf16} KiB；横向仅表顺序，不表耗时；纯融合对照须对齐 GeLU 近似。`,'small');
 const lane=[['eager 基础',['GEMM','bias','GeLU'],`${d.submissions[0].length} 次操作提交`,true],['eager + 融合',['GEMM','bias + GeLU'],`${d.submissions[1].length} 次操作提交`,false],['Graph + 融合',['GEMM','bias + GeLU'],`${d.submissions[2].length} 次图提交`,false]];
 lane.forEach(([name,ops,count,intermediate],i)=>{let y=108+i*97;b+=txt(26,y+26,name,'bold')+txt(26,y+51,count,'small');if(i===2)b+=rect(219,y-10,601,66,'acc1');ops.forEach((op,j)=>{let x=230+j*207;b+=rect(x,y,178,45)+txt(x+15,y+28,op);if(j<ops.length-1)b+=line(x+181,y+22,x+202,y+22,'main');});b+=txt(853,y+23,intermediate?`U 写 + 读 = ${d.roundtrip} KiB`:'免去独立 U 往返',intermediate?'bold':'small')+txt(853,y+45,i===2?'图内设备工作仍存在':'反向仍保存所需输入','small');});
 b+=txt(26,419,'捕获边界扩展：同一 attention → router → experts/通信 → 合并输出','bold');
 const scopes=[['partial',['图内','图内','eager','eager'],'BWD 沿边界返回'],['drop-and-pad',['图内','图内','固定专家容量','图内'],'丢弃 / padding 成本'],['HybridEP whole',['TE 图内','TE 图内','rank 容量 + stash','TE 图内'],'溢出硬失败；固定调度'],['full_iteration',['整步图','整步图','须可捕获','整步图'],'整步 F/B；optimizer 图外']];
 scopes.forEach(([name,parts,note],i)=>{let y=444+i*70;b+=txt(26,y+27,name,'bold');parts.forEach((p,j)=>{let x=218+j*164;b+=rect(x,y,151,46,p==='eager'?'ghost':'neutral')+txt(x+9,y+27,p,'small');if(j<parts.length-1)b+=line(x+152,y+22,x+161,y+22);});b+=txt(892,y+18,note,'small')+txt(892,y+39,'输出 → loss → backward','small');});
 b+=txt(26,755,'图上提交数是教学模型，非测量 kernel 数；TE 内部的量化/融合/捕获由依赖实现，不能由此换算加速比。','small');
 return svg(1220,780,'融合减少中间流量，Graph 减少主机提交',b);
}
export function slots(d=model()){
 let b=txt(26,65,'冻结源码 PP2 测试顺序；同槽下一次前向必须晚于上次反向完成。不是实际时间比例。','small');
 const x0=180,cw=101;
 d.order.forEach((s,i)=>{b+=rect(x0+i*cw,91,cw-7,45,s[0]==='F'?'acc1':'ghost')+txt(x0+i*cw+28,120,s,'bold');});
 b+=txt(26,119,'执行顺序','bold');
 d.values.forEach((v,i)=>{b+=rect(x0+i*cw,157,cw-7,35)+txt(x0+i*cw+40,181,v);});b+=txt(26,181,'未完成微批','bold');
 for(let slot=0;slot<d.peak;slot++){let y=228+slot*94;b+=txt(26,y+38,`固定槽位 ${slot}`,'bold')+rect(x0,y,cw*d.order.length-7,61,'ghost');for(const span of d.intervals.filter(i=>i.slot===slot)){let x=x0+span.start*cw;b+=rect(x+3,y+5,(span.end-span.start)*cw-14,49,'acc1')+txt(x+18,y+35,`microbatch ${span.id}：F${span.id} → B${span.id}`);}}
 b+=rect(26,436,950,69,'acc2')+txt(44,463,`${d.intervals.length} 个微批，峰值 ${d.peak} 个存活槽位；BWD 完成后才释放本次保存状态。`,'bold')+txt(44,490,'图/地址长期存在；随微批更新的是其中的数据。先复用再反向会覆盖尚需使用的激活。','small');
 return svg(1020,532,'固定地址如何承载变化的微批总数',b);
}
// Lifecycle figure spec: each lane starts with the same X. Local follows a
// forward row then a reverse backward row, exposing copies, address ownership,
// the last-layer clone and the ready event. TE marks sample slots and its callable
// dependency; full-iteration encloses F/loss/B but leaves optimizer outside.
export const lifecycleContracts = {
 local:'固定输入复制 → 前向 surface/末层 clone → loss 梯度复制 → backward/main_grad/event',
 te:'sample slots → 依赖 callable → loss/backward → 梯度消费者',
 full:'整步静态输入 → 图内 F/B → 图外 optimizer',
};
export function lifecycle(d=model()) {
 const w=1260,x0=26,step=244,nw=225;
 let b=txt(26,67,`三条通路使用同一 X[${d.m},${d.k}]；每条通路内的精度、融合与激活近似固定。箭头表示依赖，不表示耗时。`,'small');
 function box(i,y,title,detail,cls='neutral'){const x=x0+i*step;return rect(x,y,nw,58,cls)+txt(x+11,y+24,title,'bold')+txt(x+11,y+46,detail,'small');}
 function forward(y,items){let out='';items.forEach(([a,c,cl],i)=>{out+=box(i,y,a,c,cl);if(i<items.length-1)out+=line(x0+i*step+nw+1,y+29,x0+(i+1)*step-3,y+29,'main');});return out;}
 function backward(y,items){let out='';items.forEach(([a,c,cl],i)=>{out+=box(i,y,a,c,cl);if(i<items.length-1)out+=line(x0+(i+1)*step-3,y+29,x0+i*step+nw+2,y+29,'main');});return out;}
 b+=txt(26,110,'local：固定地址由 runner 持有；调用者的地址可以变化','bold');
 b+=forward(128,[['运行时 X','新的张量数据'],['固定输入表面','异址 copy_；合法别名可省','acc1'],['fwd_graph.replay','写固定输出 surface','acc1'],['外部可用 A','末层 clone，防覆盖','acc2'],['后续层 → loss','自动微分产生 G_A']]);
 b+=line(x0+4*step+nw/2,188,x0+4*step+nw/2,221,'main');
 b+=backward(225,[['梯度消费者','等待事件 / 完成梯度同步'],['main_grad + event','wgrad 累加，记录 ready','acc1'],['bwd_graph.replay','计算 dgrad / wgrad','acc1'],['固定输出梯度表面','新 G_A 异址则 copy_','acc2'],['运行时 G_A','来自 loss backward']]);
 b+=txt(26,310,'先记录首次 F/B 顺序，再共享池捕获；前向表面可复用，反向消费前不得覆盖。','small');
 b+=txt(26,356,'transformer_engine：helper 安装图；依赖 callable 承载前后向','bold');
 b+=forward(374,[['sample X / args','helper 构造静态输入'],['有界 sample slots','按 F/B 顺序证明可复用','acc1'],['make_graphed_callables','TE 捕获与缓冲优化','acc2'],['layer.cuda_graphs','按层 / 微批槽位安装'],['运行时 X → A → loss','选定槽位 callable 执行']]);
 b+=line(x0+4*step+nw/2,434,x0+4*step+nw/2,467,'main');
 b+=backward(471,[['梯度消费者','同步 / optimizer 后续消费'],['dgrad / wgrad','返回或累加参数梯度'],['TE backward','内部保存由依赖管理','acc2'],['同槽 callable 的反向','按原 F/B 对应关系执行','acc1'],['loss 的自动微分','到达图 callable 边界']]);
 b+=txt(26,556,'橙框为 TE 证据边界：Megatron 可核实输入、顺序与安装；不把 local 的缓冲/event 实现照搬给 TE。','small');
 b+=txt(26,601,'full_iteration：数据复制在图外，整步 F/loss/B 在图内','bold');
 b+=forward(619,[['各微批的 X','data_read 读完整步输入'],['StaticBufferLoader','复制流写入，当前流等待','acc1'],['整步 CUDA Graph','forward → loss → backward','acc1'],['保存的 result 容器','重放结果交训练循环'],['optimizer.step（图外）','随后参数同步 / 下一次 GEMM','acc2']]);
 b+=txt(26,706,'训练与验证各持有图、静态输入和结果；共享 mempool 不等于 optimizer 已成为这张 F/B 图的一部分。','small');
 return svg(w,735,'同一输入如何走到可消费的梯度',b);
}

export const renderers={megatron_precision_quantization:quantization,megatron_precision_launch:launch,megatron_precision_graph_slots:slots,megatron_precision_graph_lifecycle:lifecycle};
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)){
 await mkdir(assetDir,{recursive:true});
 for(const [name,render] of Object.entries(renderers))if(!process.argv[2]||process.argv[2]===name) await writeFile(resolve(assetDir,name+'.svg'),render());
}
