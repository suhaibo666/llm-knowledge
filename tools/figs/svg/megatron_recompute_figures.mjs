// Figure specification (written before rendering implementation):
// All lanes use microbatch A, t0..t3, X0[4,1,4], L0..L3.
// 1. Primitive: input -> original forward -> retained/discarded state -> replay -> grads.
//    Ordinary, CWO, TP-sharded saved input, and the TE boundary remain distinct.
// 2. Full: uniform and block show actual forward grouping and reverse replay order;
//    Hybrid/GPT/MTP and hand-replayed EP plans expose their different dispatch rules.
// 3. Normal selective: one lane per selected region. CP and EP movement is inside
//    the checkpoint rectangle exactly when replay must repeat it.
// 4. Discard selective: producer output -> real consumer -> release -> consumer-grad
//    trigger -> restore producer -> consumer backward -> producer backward.
// 5. mHC: registration dependencies survive a group; replay is in forward order;
//    eager/EP barrier/fixed-address TE slot/full-iteration capture have separate lanes.
// Neutral is the default; blue marks replay, amber marks unavoidable cost/boundary.

import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';

export function solveExample() {
  const S=4,B=1,H=4,tp=2,bytes=2,layers=4,group=2,nstreams=2;
  const size=S*B*H, x=2,w=3,mask=2,v=5,y=x*w*mask,z=y*v;
  const loss=size*z*z/2,dx=z*v*w*mask,dw=size*z*v*x*mask,dv=size*z*y;
  const chunks=Array.from({length:Math.ceil(layers/group)},(_,i)=>
    Array.from({length:Math.min(group,layers-i*group)},(_,j)=>i*group+j));
  const block=Array.from({length:group},(_,i)=>[i]);
  const tokens=Array.from({length:S},(_,i)=>`t${i}`);
  const owners=tokens.map((t,i)=>({token:t,expert:i%2}));
  return {S,B,H,tp,size,bytes,fullBytes:size*bytes,shardSize:size/tp,
    shardBytes:size*bytes/tp,layers,group,nstreams,carrierSize:size*nstreams,
    carrierBytes:size*nstreams*bytes,x,w,mask,v,y,z,loss,dx,dw,dv,chunks,block,tokens,owners,
    epUniformSegments:layers+1,epBlockSegments:group+1,
    flags:tokens.map((_,i)=>i%layers>=group)};
}

const escape = s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
const W=1240, margin=24, labelW=162, colW=247, gap=10;
const css=`text{font-family:'Segoe UI','Microsoft YaHei',sans-serif;fill:#1f2937;font-size:16px}
.title{font-size:23px;font-weight:650}.subtitle{fill:#64748b;font-size:15px}
.cap{fill:#64748b;font-size:14px}.rowname{font-weight:650;font-size:16px}
.neutral{fill:#fff;stroke:#cbd5e1}.ghost{fill:#f8fafc;stroke:#e2e8f0}
.acc1{fill:#eff6ff;stroke:#2563eb}.acc2{fill:#fff7ed;stroke:#d97706}
.arrow{fill:none;stroke:#2563eb;stroke-width:2;marker-end:url(#a)}
.aux{stroke:#94a3b8;stroke-width:1.2}.box{rx:7;stroke-width:1.2}`;

function text(lines,x,y,cls='') {
  return lines.map((s,i)=>`<text class="${cls}" x="${x}" y="${y+i*23}">${escape(s)}</text>`).join('');
}
function wrap(lines,limit=27){
 return lines.flatMap(line=>{
  let result=[],part='',units=0;
  for(const token of line.match(/[A-Za-z0-9_.\/-]+|./gu)||[]){
   const n=[...token].reduce((a,ch)=>a+(/[^\x00-\x7F]/.test(ch)?2:1),0);
   if(units+n>limit&&part){result.push(part.trim());part='';units=0;}
   part+=token;units+=n;
  }
  if(part)result.push(part);
  return result;
 });
}
function lane(id,label,cells,y,h=158) {
  let s=`<g data-plane="${id}"><rect class="ghost box" x="${margin}" y="${y}" width="1192" height="${h}"/>`;
  s+=text(wrap(label,18),margin+12,y+28,'rowname');
  cells.forEach((c,i)=>{
    const x=margin+labelW+i*(colW+gap);
    s+=`<g data-cell="${id}-${i}"><rect class="${c.kind||'neutral'} box" x="${x}" y="${y+12}" width="${colW}" height="${h-24}"/>`;
    s+=text(c.lines,x+12,y+38);
    s+='</g>';
    if(i<cells.length-1)s+=`<path class="arrow" d="M ${x+colW} ${y+h/2} h ${gap-2}"/>`;
  });
  return s+'</g>';
}
const cell=(...lines)=>({lines});
const replay=(...lines)=>({lines,kind:'acc1'});
const cost=(...lines)=>({lines,kind:'acc2'});
function primitiveLayout(e,y){
 let s=`<g data-layout="tp-saved-identity"><rect class="ghost box" x="24" y="${y}" width="1192" height="183"/>`;
 const matrix=(x,top,ids,cols,klass)=>ids.map((id,i)=>{
  const xx=x+(i%cols)*32,yy=top+Math.floor(i/cols)*24;
  return `<rect class="${klass}" x="${xx}" y="${yy}" width="30" height="22"/>${text([`x${id}`],xx+3,yy+16,'cap')}`;
 }).join('');
 s+=text(['同一 X 的元素身份：先完整前向，再切保存面；回放前恢复'],40,y+25,'rowname');
 s+=text([`X0[4,1,4] · ${e.fullBytes} B`],40,y+53,'cap');
 s+=matrix(40,y+64,Array.from({length:e.size},(_,i)=>i),e.H,'neutral');
 s+=`<path class="arrow" d="M 182 ${y+106} H 313"/>`;
 s+=text(['原前向之后 copy'],190,y+96,'cap');
 s+=text([`rank0 [0,${e.shardSize}) · ${e.shardBytes} B`],330,y+57,'cap');
 s+=matrix(330,y+64,Array.from({length:e.shardSize},(_,i)=>i),e.shardSize,'acc2');
 s+=text([`rank1 [${e.shardSize},${e.size}) · ${e.shardBytes} B`],330,y+111,'cap');
 s+=matrix(330,y+119,Array.from({length:e.shardSize},(_,i)=>i+e.shardSize),e.shardSize,'acc2');
 s+=`<path class="arrow" d="M 601 ${y+105} H 786"/>`;
 s+=text(['backward: TP all-gather'],609,y+93,'cap');
 s+=text([`每 rank 重建 X0 · ${e.fullBytes} B`],805,y+53,'cap');
 s+=matrix(805,y+64,Array.from({length:e.size},(_,i)=>i),e.H,'acc1');
 s+=text(['detach → replay f','得到 dX0 与 dθ','非半个输入直接计算'],968,y+82);
 return s+'</g>';
}
function figure(title,subtitle,headers,rows,foot,layout=null) {
  let y=122,s='';
  rows.forEach(r=>{
   const cells=r.cells.map(c=>({...c,lines:wrap(c.lines)}));
   const h=Math.max(r.h||158,Math.max(...cells.map(c=>c.lines.length))*23+42);
   s+=lane(r.id,r.label,cells,y,h);y+=h+12;
  });
  if(layout){s+=layout(y);y+=197;}
  const height=y+58;
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${height}" role="img" aria-label="${escape(title)}"><style>${css}</style><defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#2563eb"/></marker></defs><rect width="${W}" height="${height}" fill="#fff"/>${text([title],margin,34,'title')}${text([subtitle],margin,61,'subtitle')}${headers.map((h,i)=>text([h],margin+labelW+i*(colW+gap)+12,99,'rowname')).join('')}${s}${text(foot,margin,y+17,'cap')}</svg>\n`;
}

export function buildFigures(){
 const e=solveExample(), figs={};
 const same=`microbatch A · X0[${e.S},${e.B},${e.H}] · t0…t${e.S-1} · BF16 · TP=${e.tp} · SP off`;
 figs['megatron_recompute_primitive.svg']=figure('输入留存与反向恢复：输出身份也是状态',same,
 ['原前向 f → g','前反向间隔','loss backward / replay','消费者梯度 → 参数交付'],[
  {id:'ordinary',label:['ordinary','checkpoint'],cells:[cell('X → f(no-grad) → Y','Y → g → Z → loss','保存 X + 原前向 RNG'),cell('X: '+e.size+' el / '+e.fullBytes+' B','f 内部图不保留','g 保存的 Y 仍存活'),replay('dY 到达 checkpoint','恢复 RNG → f(grad)','还原环境 RNG → B(f)'),cell('返回 dX，累积 dθ','g 先算 dφ 与 dY','finalize grads → optimizer')]},
  {id:'without-output',label:['Checkpoint','WithoutOutput'],cells:[cell('X → f(no-grad) → Y','g 的 backward 保存 Y','g 完成后，hook 绑定 Z'),cost('Y storage.resize_(0)','保留旧 Tensor / views','X + RNG + metadata 留下'),replay('dZ hook → replay f','旧 StorageImpl 共享新值','g backward 前恢复 Y'),cell('B(g) → dφ,dY','CWO B(f) → dθ,dX','ctx/replay 引用清空')]},
  {id:'tp-saved-input',label:['distribute_saved','_activations'],cells:[cell('f 原前向读完整 X','之后 flatten / new copy',`rank0 [0,${e.shardSize})`),cost(`rank1 [${e.shardSize},${e.size})`,`每 rank ${e.shardSize} el / ${e.shardBytes} B`,'只切 args[0]，非全部输入'),replay('TP all-gather → full X',`恢复 [${e.S},${e.B},${e.H}] → f(grad)`,'AG 后按 stream 顺序使用'),cell('B(f) 返回完整输入梯度','原层 TP 通信照常执行','+AG，回放峰值重现 full X')]},
  {id:'te-dependency',label:['FP8 / FP4','TE checkpoint'],cells:[cell('量化 full / MLP / MoE','传函数、位置输入、TP组','RNG getter、分存开关'),cost('Megatron → TE 边界','版本 ≥1.5 新 kwargs','更旧版本用位置签名'),replay('TE checkpoint 返回 Y','回放/量化元数据：','TE 合约，非本仓内部证明'),cell('按调用接口回传梯度','延迟 FP8 MTP 外层 context','内部容量/数值须依赖验证')]}
 ],[`局部可算例：x=${e.x}, w=${e.w}, mask=${e.mask}, v=${e.v} → y=${e.y}, z=${e.z}, loss=${e.loss}, dx=${e.dx}, dw=${e.dw}, dv=${e.dv}。`,`数值为 ${e.size} 个元素共享标量权重的解析演示；RNG 恢复不代表通信或 optimizer 已完成。`],y=>primitiveLayout(e,y));
 const groupName=c=>c.map(i=>`L${i}`).join('→');
 figs['megatron_recompute_full.svg']=figure('同一四层，分组策略与模型入口共同决定 replay',same,
 ['原前向选择 / 区域','保留边界','反向重建顺序','结果与限制'],[
  {id:'uniform',label:['full / uniform',`n=${e.group}`],cells:[cell(...e.chunks.map(c=>`checkpoint ${groupName(c)}`),'每块 no-grad 原前向'),cell(...e.chunks.map(c=>`保留 X${c[0]} + RNG`),`${e.chunks.length} 个区域`),replay(...[...e.chunks].reverse().map(c=>`replay ${groupName(c)} → B`),`${e.layers} 次层前向回放`),cost('少保留 chunk 边界','chunk 越大，重建图越大','仅 chunk 尾可取中间特征')]},
  {id:'block',label:['full / block',`n=${e.group}`],cells:[cell(...e.block.map(c=>`checkpoint ${groupName(c)}`),'L2→L3 正常保存内部图'),cell('保留 X0,X1 + RNG','尾部 L2,L3 内部图保留',`${e.block.length} 个区域`),replay('正常 B(L3),B(L2)','replay L1→B，再 L0→B',`${e.group} 次层前向回放`),cost('只支付选中层 replay','量化输入无 grad：窗口后移','不能承诺永远物理前 N 层')]},
  {id:'hybrid',label:['GPT / Hybrid','sibling entry'],cells:[cell('GPT: 自有 checkpointed','Hybrid: 共享 recompute.py','同一 L0…L3 分组规则'),cell('TransformerLayer: 全 kwargs','能力位包装: 保留路由输入','裸 Mamba: 窄 kwargs'),replay('重建同类型层与原输出','包装返回 hidden,context','裸层返回 hidden'),cost('能力位避免循环 import','GPT block eager 漏 padding','共享 Hybrid 分支已保留')]},
  {id:'mtp',label:['ordinary MTP','Hybrid MTP'],cells:[cell('X4 + decoder_input','普通: proj + MTP layer','Hybrid: 内部 HybridStack'),cell('普通 uniform 要 n=1','普通 block 留原图','Hybrid 委托内部 full'),replay('ordinary uniform replay','ordinary block: warn/skip','Hybrid: 内部 stack replay'),cost('dX4 必须返回 decoder','不是所有 MTP 都 skip block','MTP loss 缩放仍由训练层管')]},
  {id:'ep-overlap-full',label:['EP overlap full','Recompute','Segment'],h:206,cells:[cell('含 MTP=1，公共配置要求：','uniform n=1: L0|L1|L2|L3',`+MTP = ${e.epUniformSegments} 段`,'block n=2: L0|L1',`+MTP = ${e.epBlockSegments} 段；L2,L3 正常图`),cell('每段 capture input/state/RNG','no-grad 节点后清 transient','不走 checkpoint / 不分存','MTP bridge old grad carrier','无量化 block 窗口后移'),replay('段尾 backward 前 replay','attn→dispatch→MLP→combine','→mhc_post→mtp_post','重建节点图后逐节点 B','MTP 两方法均独立 replay'),cost('replay A2A 暴露；原 overlap 在','新 bridge leaf 接旧梯度','段头 B 后 release input','零 dropout / 非 delayed FP8','低层可多层 ≠ 公共配置放行')]}
 ],[`四层例：uniform=${e.chunks.length} 区域/${e.layers} 回放；block=${e.block.length} 区域/${e.group} 回放；PP w=${e.layers},k=${e.group} → ${e.flags.map(x=>x?'True':'False').join(',')}。`,'分组数字不等于完整显存；每条路线最后都需要输入/参数梯度完成并经 finalizer 交付。']);
 const ep0=e.owners.filter(o=>o.expert===0).map(o=>o.token).join(','), ep1=e.owners.filter(o=>o.expert===1).map(o=>o.token).join(',');
 figs['megatron_recompute_selective_normal.svg']=figure('selective / normal：回放区域内的计算与通信一起重做',same+' · L0 内替换对应子模块',
 ['保存输入 → 原前向区域','输出 / 跨间隔状态','上游梯度到达后的 replay','恢复后 backward / 新增成本'],[
  {id:'core_attn',label:['core_attn'],cells:[cell('Q,K,V + mask + enum','QKᵀ→softmax/dropout→PV','DSA tensor kwargs 位置传递'),cell('输出 O → output projection','保留 Q/K/V；内部图不保留','QKV 投影在 checkpoint 外'),replay('dO → 重跑 core attention','同 RNG / mask / extra tensors','backend CP 通信若在内也重跑'),cost('B(attention) → dQ,dK,dV','再由投影 B 得 dX0,dW','fused attention 可能无大收益')]},
  {id:'mlp',label:['mlp','dense only'],cells:[cell('U = pre-MLP norm(X)','FC1 → activation → FC2','区域内 TP collective'),cell('M → residual / 后层','保存 U + RNG','FC1/act 内部图释放'),replay('dM → replay 整 dense MLP','重做两组 GEMM 与通信','MLP 输出图重建'),cost('B(FC2),B(act),B(FC1)','得到 dU,dW1,dW2','MoE layer 不由 mlp 值启用')]},
  {id:'moe',label:['moe','full MoE region'],cells:[cell(`U: ${e.tokens.join(',')}`,'route / permutation / dispatch',`${ep0}→E0; ${ep1}→E1`),cell('expert→combine→token order','M + shared output','保留 U 与路由输入元数据'),replay('dM → replay route/dispatch','重建 expert 顺序与概率','expert→combine 恢复 token'),cost('B(combine)→B(expert)','→B(dispatch/router)→dU,dW','+EP A2A；overlap 入口禁 moe')]},
  {id:'shared_experts',label:['shared_experts'],cells:[cell('同 U → shared MLP','FC1→activation→FC2','routed branch 在区域外'),cell('S 与 routed output 汇合','保存 U + RNG','共享支路内部图释放'),replay('dS → replay shared MLP','不重发 routed tokens','恢复共享分支图'),cost('B(shared) → dU,dWshared','+本支路 GEMM/TP 通信','不能 shared-expert-overlap')]},
  {id:'gdn',label:['gdn','GDN / KDA'],h:205,cells:[cell('X0 按 CP token 分片','in-proj → CP→HP A2A','所有 token / 半数 heads','conv → delta rule'),cell('norm → HP→CP A2A','out-proj → O','rank0 t0,t3 / rank1 t1,t2','保存 X0，内部图释放'),replay('dO → replay 完整区域','两向 A2A + conv + delta','+norm + out-proj','chunkwise kernel: 外部边界'),cost('B(out-proj)→CP→HP','B(norm/delta/conv)','HP→CP→B(in-proj)','dX0,dW；两向 replay 通信')]}
 ],['GDN/KDA headwise 例：CP rank0=t0,t3；rank1=t1,t2。HP 后每 rank 拥有 t0…t3 的半数 heads。','kernel 内部归依赖合约；此图只证明本仓包装、布局和调用顺序，非 kernel 内实现测量。']);
 figs['megatron_recompute_selective_discard.svg']=figure('selective / output-discard：消费者反向读 storage 前必须恢复',same+' · t0…t3 身份始终不变',
 ['保存输入 → producer','consumer 完成后 discard','consumer-grad trigger → replay','consumer B → producer B'],[
  {id:'layernorm',label:['layernorm'],cells:[cell('X / residual → norm','U → attention / MLP','真实 norm，Identity 跳过'),cost('丢 U storage / 保留 X','后继必须直接保存 U','量化消费者保存 original input'),replay('attention/MLP 输出 hook','先 replay norm → 恢复 U','图条件可能禁 pre-MLP norm'),cell('B(consumer) 读取 U','dU → B(norm) → dX,dθ','不重做整个投影') ]},
  {id:'moe_act',label:['moe_act','non-fused'],cells:[cell('routed FC1 输出 A','A,bias,probs → bias_act','B → FC2'),cost('FC2 F 后释放 B storage','保存 A,bias,permuted_probs','不重算外层 EP dispatch'),replay('FC2 输出 hook → act replay','恢复 B 的旧 storage/views','FC2 backward 现在可读 B'),cell('B(FC2)→dB,dW2','B(act)→dA,dprobs','再 B(FC1)→dU,dW1')]},
  {id:'moe_act-fused',label:['moe_act','TE fused sibling'],cells:[cell('A→scaled act→FC2','_make_fused_ops 组装 TE op','activation_recompute_in_mlp'),cost('inspect.signature 有此参数','才向 TE op 传重算开关','缺参数也构建 op'),replay('Megatron → TE op boundary','回放和存储: 依赖实现','本仓未证明同 CWO 的收益'),cell('接口交付 FC2/FC1 梯度','额外计算/容量不编造','同 token 与概率乘法契约')]},
  {id:'mla_up_proj',label:['mla_up_proj'],cells:[cell('compressed Q / KV','k_pos + RoPE','up projection + RoPE→QKV'),cost('QKV → core attention','core F 后释放 Q/K/V','压缩输入继续保存'),replay('core O hook → up+RoPE','恢复 Q/K/V 所有输出','attention B 前完成'),cell('B(core)→dQ,dK,dV','B(up+RoPE)→dlatents,dW','额外 up GEMM，不只 softmax')]},
  {id:'gdn_norm_out',label:['gdn_norm_out','GDN / KDA'],h:182,cells:[cell('D + gate 在 HP 布局','all tokens / half heads','norm → HP→CP A2A','N → out projection'),cost('out-proj F 后释放 N','保留 D/gate 与 inverse map','rank0 t0,t3 / rank1 t1,t2','不与整个 gdn 同选'),replay('out O hook → norm + A2A','再次恢复 CP token 布局','N storage 恢复','无 in-proj/conv/delta replay'),cell('B(out-proj) 读 N→dN,dW','B(restore): CP→HP','B(norm)→dD,dgate','再走原 delta/conv B')]},
  {id:'mhc',label:['mhc','grouped CWO'],cells:[cell('C0 多 stream → aggregate','norm / residual / expand','checkpoint 输入相互依赖'),cost('组尾统一 discard','每 checkpoint 自有 RNG','组边界保留'),replay('统一 hook 或 schedule barrier','按原 forward 注册顺序恢复','固定 slot 详见 mHC 图'),cell('消费者 B→各 CWO B','梯度回到 streams/mappings','不能独立任意触发') ]}
 ],['CWO 额外减少的是后继保存的 output storage；producer input、实际副本和静态 graph slots 仍占显存。','普通 checkpoint 不能替代这个 storage 生命周期；合法配置也不保证所有后端都真正采用同一机制。']);
 figs['megatron_recompute_mhc.svg']=figure('mHC：恢复次序、梯度 barrier 与固定地址',same+` · ${e.nstreams} streams: C0[4,1,${e.H*e.nstreams}], ${e.carrierSize} el / ${e.carrierBytes} B`,
 ['原前向 producer / consumer','前反向间隔 owner','loss backward 前恢复','消费与释放边界'],[
  {id:'mhc-eager',label:['eager group',`L0→L1; L2→L3`],h:182,cells:[cell('compute_mappings 正常保留','C→aggregate→norm→attn','residual/post→norm→MLP','按顺序注册 CWO'),cost('每组 manager 保存 checkpoints','每个 producer 独立 RNG','组尾保留 / 其余统一释放','不能给所有 CWO 一个 RNG'),replay('组尾梯度 unified hook','recompute_now → 全组','按原 F 顺序逐个恢复','先 producer，再 dependent'),cell('consumer B 可读旧 storage','各 CWO B → streams/dW','ctx 清理不等于 arena 释放','容量受组边界和活读者限制')]},
  {id:'mhc-post-branches',label:['post BDA','exclusive A / B'],h:190,cells:[cell('互斥 A: dropout=0 或 eval','fused/reference h_post-BDA','互斥 B: dropout>0 且 train','h_res→h_post expand→BDA'),cell('只选 A 或 B 包入 CWO','保留所选 op 输入 + 原 RNG','组尾 discard 所选 op 输出','输入/RNG 跨前反向间隔'),replay('A: replay fused/reference op','B: replay 顺序 op','每 checkpoint 恢复同 RNG','重建被选 producer 的输出'),cell('恢复同 n-stream 输出','后继反向 → 所选 producer 反向','返回 mapping/input grads','full graph 不允许非零 dropout')]},
  {id:'mhc-ep-barrier',label:['EP overlap','explicit barrier'],cells:[cell('正常 compute/comm 节点','manager 随在途 plan 独立','组尾 discard_all_outputs'),cost('recompute node 在 compute stream','BEFORE_COMBINE_BWD','所有 producer 仅此 phase'),replay('recompute_until → 全组','先于 mhc_post/combine B','另一 phase 也仍全组 replay'),cell('恢复 → mhc_post B→combine B','顺序由节点/stream 依赖保证','不在 replay 点回收 slot')]},
  {id:'mhc-te-arena',label:['TE attention split','fixed-address arena'],h:182,cells:[cell('eager aggregate producer','直接写 slot.writer','graph consumer 绑定同地址','input norm + attention 捕获'),cost('slot 保留物理 allocation','discard 仅逻辑失效','checkpoint 持有输入与 RNG','不 resize captured storage'),replay('验证 address/shape/dtype','replay 直接写同 slot','旧逻辑 tensor storage 衔接','不能新 tensor 换掉 graph 输入'),cell('captured attention B 读 slot','mHC post B 也可能读取','recompute 返回非 last-reader','非 Hybrid/attn-only/mhc-only')]},
  {id:'mhc-full-graph',label:['full_iteration','captured lifecycle'],cells:[cell('CWO 在 warmup bypass','capture 包含 producer','discard + replay + backward'),cost('full graph 固定地址','零 attention/hidden dropout','local graph + mhc 被拒绝'),replay('graph replay 重放已捕获操作','不是普通 checkpoint wrapper','普通 checkpoint capture bypass'),cell('图内 B → grads','图调度完成后 finalizer','optimizer-ready 另有同步') ]}
 ],['BEFORE_ATTN_BWD 是另一个调用参数；所有 producer 仍只注册 BEFORE_COMBINE_BWD，尚无 phase 分区。','每次恢复只证明后继能够继续读取；最后 reader、梯度同步和 optimizer 更新由各自 owner 承接。']);
 return figs;
}

if(process.argv[1] && resolve(process.argv[1])===fileURLToPath(import.meta.url)){
 const root=resolve(dirname(fileURLToPath(import.meta.url)),'../../..');
 const out=process.argv[2]||join(root,'wiki/02_engineering/02_train_frameworks/megatron-lm/assets');
 await mkdir(out,{recursive:true});
 for(const [name,svg] of Object.entries(buildFigures())) await writeFile(join(out,name),svg,'utf8');
 process.stdout.write(`generated ${Object.keys(buildFigures()).length} recompute figures\n`);
}
