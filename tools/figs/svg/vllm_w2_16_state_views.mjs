// 图规格1：左为4行persistent state，X退出后row2保留空洞；右为当步两行batch。
// B的state row1经idx_mapping进入batch row0，A的state row3进入batch row1；箭头不移动左表。
// 图规格2：全batch token位置0..3为B5/A18/A19/padding，分界在2。
// 下方两块显示各ubatch的局部qsl与seq；A18先算完，第二块A19仍拥有完整20位置历史。
// 教学参数与vLLM 199cb9b9的寻址/截断规则；不运行vLLM、不模拟并发耗时。
import {pathToFileURL} from 'node:url';
export function replay() {
 const free=[0,1,2,3], map={};for(const id of ['A','X','B'])map[id]=free.pop();free.push(map.X);delete map.X;
 const order=['B','A'],idx=order.map(id=>map[id]),queries=[1,2],computed=[5,18];
 const qsl=[0];for(const q of queries)qsl.push(qsl.at(-1)+q);
 const seq=computed.map((n,i)=>n+queries[i]);
 const slice=(rs,re,ts,te)=>{
  const localQ=qsl.slice(rs,re+1).map(x=>Math.min(te-ts,Math.max(0,x-ts)));
  const localS=seq.slice(rs,re);localS[localS.length-1]-=Math.max(0,qsl[re]-te);
  return {qsl:localQ,seq:localS,tokens:Math.max(0,Math.min(te,3)-ts)};
 };
 return {map,free,order,idx,qsl,seq,first:slice(0,2,0,2),second:slice(1,2,2,4)};
}
const esc=x=>String(x).replaceAll('&','&amp;').replaceAll('<','&lt;');
function canvas(w,h,title){
 const parts=[`<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}"><title>${esc(title)}</title><style>text{font-family:Arial,'PingFang SC','Microsoft YaHei',sans-serif;fill:#2A313B;font-size:20px}.neutral{fill:#fff;stroke:#C7CCD3}.ghost{fill:#F7F6F3;stroke:#DDD9D2}.acc1{fill:#EAF1FD;stroke:#2563EB}.acc2{fill:#FCF1E6;stroke:#C3651F}.heading{font-size:24px;font-weight:600}.cap{font-size:19px;fill:#6B7280}.arrow{stroke:#2563EB;stroke-width:2.5;fill:none}</style><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#2563EB"/></marker></defs><rect width="${w}" height="${h}" fill="white"/>`];
 const text=(x,y,s,c='')=>parts.push(`<text x="${x}" y="${y}" class="${c}">${esc(s)}</text>`);
 const box=(x,y,w,h,c='neutral')=>parts.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" class="${c}"/>`);
 const arrow=(x,y,xx,yy)=>parts.push(`<path d="M${x} ${y}L${xx} ${yy}" class="arrow" marker-end="url(#a)"/>`);
 return {parts,text,box,arrow,end:()=>parts.join('\n')+'\n</svg>'};
}
export function drawStable(){
 const r=replay(),g=canvas(1040,475,'MRV2：稳定状态行与每步batch视图');const {text,box,arrow}=g;
 text(30,34,'请求不因排序搬家：只改变 batch_idx → state_idx','heading');
 text(30,79,'持久状态表（X 已退出）','heading');text(600,79,'本步执行视图 [B, A]','heading');
 const cols=[70,75,120,160],heads=['state','请求','computed','block table'];let x=30;
 heads.forEach((h,i)=>{box(x,100,cols[i],35,'ghost');text(x+8,125,h);x+=cols[i]});
 const rows=[['0','—','—','—'],['1','B','5','[28]'],['2','空位','—','—'],['3','A','18','[12,13]']];
 rows.forEach((a,j)=>{let xx=30;const y=135+j*45;a.forEach((v,i)=>{box(xx,y,cols[i],45,j===1||j===3?'neutral':'ghost');text(xx+8,y+29,v);xx+=cols[i]})});
 const rhs=[['batch','请求','state','query'],['0','B',r.idx[0],'1'],['1','A',r.idx[1],'2']];
 rhs.forEach((a,j)=>{let xx=600;const y=100+j*55;a.forEach((v,i)=>{box(xx,y,100,55,j===0?'ghost':i===2?'acc1':'neutral');text(xx+9,y+34,v);xx+=100;})});
 arrow(460,202,592,182);arrow(460,292,592,237);
 text(600,303,`idx_mapping = [${r.idx.join(',')}]`,'heading');
 text(30,364,'state row 2 的空洞保留；新请求可复用该槽，A/B 原行不变。','cap');
 text(30,404,'GPU 按 mapping 读取：B 的 last_sampled=205，A 的 prompt[18:20]=118,119。');
 text(30,447,`得到 input_ids=[205,118,119]；Query 边界=[${r.qsl.join(',')}]；seq_lens=[${r.seq.join(',')}]。`);
 return g.end();
}
export function drawUbatch(){
 const r=replay(),g=canvas(1040,610,'MRV2：跨微批次请求只截掉未来Query');const{text,box,arrow}=g;
 text(30,35,'同一 A 横跨两个微批次：第二段保留第一段已算出的历史','heading');
 text(30,77,'教学切分：3 个真实 token 补到 4；边界在扁平位置 2（不代表默认触发阈值）','cap');
 const labels=['B5','A18','A19','padding'];
 labels.forEach((s,i)=>{text(100+i*230,115,`flat ${i}`,'cap');box(80+i*230,128,205,58,i===3?'ghost':i===1?'acc1':'neutral');text(140+i*230,165,s)});
 g.parts.push('<path d="M525 112V202" stroke="#C3651F" stroke-width="2" stroke-dasharray="5 4"/>');
 text(110,223,'U0：token [0,2)');text(590,223,'U1：token [2,4)');
 arrow(285,237,285,262);arrow(760,237,760,262);
 box(30,276,475,245);box(535,276,475,245);
 text(50,312,'U0：B5、A18','heading');text(555,312,'U1：A19、padding','heading');
 text(50,355,`query_start_loc = [${r.first.qsl.join(',')}]`);text(555,355,`query_start_loc = [${r.second.qsl.join(',')}]`);
 text(50,394,`seq_lens = [${r.first.seq.join(',')}]`);text(555,394,`seq_lens = [${r.second.seq.join(',')}]`);
 text(50,436,'A 的未来 A19 尚未算：20 − 1 = 19');text(555,436,'A18 已在 U0 算过：仍计入 20');
 text(50,477,'B 可见 0–5；A18 可见 A0–A18','cap');text(555,477,'A19 可见 A0–A19；padding 不产出','cap');
 text(30,562,'拼回 B5 / A18 / A19 的 hidden states 后，只对完整 batch 采样一次。');
 text(30,597,'不能按“该微批次只有一个 A Query”把 U1 的历史又减掉 A18。','cap');
 return g.end();
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href) console.log(process.argv[2]==='ubatch'?drawUbatch():drawStable());
