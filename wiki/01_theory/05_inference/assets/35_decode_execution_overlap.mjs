// Regenerate with: node wiki/01_theory/05_inference/assets/35_decode_execution_overlap.mjs
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const dir = dirname(fileURLToPath(import.meta.url));
const page = readFileSync(join(dir, '../35_inference_execution_optimization_analysis.md'), 'utf8');
const duration = { M: 2, B: 1, U: 2, C: 2, V: 3, J: 1, R: 1 };
const serial = Object.values(duration).reduce((a, b) => a + b, 0);
const start = {
  M: 0,
  B: duration.M,
  U: duration.M + duration.B,
};
start.C = start.U + duration.U;
start.V = start.C; // V does not depend on C, but follows U in this teaching schedule.
start.J = Math.max(start.C + duration.C, start.V + duration.V);
start.R = start.J + duration.J;
const overlapped = start.R + duration.R;
if (serial !== 12 || overlapped !== 10 || start.J !== 8) throw new Error('Teaching schedule changed');
for (const claim of [
  '串行总长 $2+1+2+2+3+1+1=12$',
  '$\\max(7,8)=8$',
  '总长 10',
  '从写入的 $t=2$ 起，到 $V$ 最后读取完成的 $t=8$',
]) if (!page.includes(claim)) throw new Error(`Page lost schedule claim: ${claim}`);

const esc = s => String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
const T = (x, y, s, klass='text', anchor='start') => `<text class="${klass}" x="${x}" y="${y}" text-anchor="${anchor}">${esc(s)}</text>`;
const R = (x, y, w, h, klass) => `<rect class="${klass}" x="${x}" y="${y}" width="${w}" height="${h}" rx="7"/>`;
const L = (x1, y1, x2, y2, klass='rule') => `<line class="${klass}" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/>`;
const P = (d, klass='dep') => `<path class="${klass}" d="${d}"/>`;
const x0=260, scale=84, X=t=>x0+t*scale;
const out=[];
out.push(T(36,46,'同一 decode 步骤：串行 12 → 部分重叠 10（教学时间单位）','title'));
out.push(T(36,73,'工作量不变；只有通信 C 与独立计算 V 可并行，J 必须等待两者','sub'));

// Shared grid and axis, so both schedules can be checked against the same clock.
for(let i=0;i<=12;i++){
  out.push(L(X(i),103,X(i),688,'grid'));
  out.push(T(X(i),96,String(i),'tick','middle'));
}
out.push(T(45,96,'时间单位 →','tick'));

out.push(R(26,118,1284,143,'panel'));
out.push(T(45,148,'串行','section'));
out.push(T(45,171,'所有片段依次执行','small'));
const serialParts=[
  ['M','元数据 M',0,2,'cpu'],['B','写 B',2,3,'cpu'],
  ['U','计算 U',3,5,'gpu'],['C','通信 C',5,7,'comm'],
  ['V','计算 V',7,10,'gpu'],['J','合并 J',10,11,'gpu'],['R','消费结果',11,12,'cpu'],
];
for(const [,label,a,b,kind] of serialParts){
  out.push(R(X(a)+2,178,(b-a)*scale-4,52,kind));
  out.push(T((X(a)+X(b))/2,209,label,'box','middle'));
}
out.push(T(1278,243,'完成 12','end','end'));

out.push(R(26,280,1284,413,'panel'));
out.push(T(45,313,'部分重叠','section'));
out.push(T(45,338,'CPU','lane'));
out.push(T(45,428,'GPU 计算','lane'));
out.push(T(45,518,'通信','lane'));

const box=(a,b,y,label,kind)=>{out.push(R(X(a)+2,y,(b-a)*scale-4,55,kind));out.push(T((X(a)+X(b))/2,y+34,label,'box','middle'));};
box(start.M,start.M+duration.M,317,'元数据 M','cpu');
box(start.B,start.B+duration.B,317,'写 B','cpu');
box(start.R,start.R+duration.R,317,'消费结果','cpu');
box(5,7,317,'独立入队准备','dashed');
box(start.U,start.U+duration.U,407,'计算 U','gpu');
box(start.V,start.V+duration.V,407,'计算 V','gpu');
box(start.J,start.J+duration.J,407,'合并 J','gpu');
box(start.C,start.C+duration.C,497,'通信 C','comm');

// Explicit prerequisite arrows and the join, rather than adjacent colored bars alone.
out.push(P(`M ${X(3)} 372 V 404`)); // input write -> GPU U
out.push(P(`M ${X(5)} 462 V 494`)); // U -> C
out.push(P(`M ${X(7)} 552 V 479 H ${X(8)-9} V 465`)); // C -> J
out.push(P(`M ${X(9)} 404 V 375`)); // J -> CPU visible result
out.push(T(X(5)+10,482,'U 产出载荷','arrowNote'));
out.push(T(X(8)-10,476,'C 与 V 汇合','arrowNote','end'));
out.push(T(X(9)+12,394,'结果可见','arrowNote'));
out.push(T(1278,587,'完成 10','end','end'));

out.push(L(44,603,1292,603,'rule'));
out.push(T(45,627,'缓冲生命周期','section'));
const life=(a,b,y,label,color)=>{out.push(L(X(a),y,X(b),y,color));out.push(T(X(b)+9,y+5,label,'life'));};
life(2,8,626,'B：最后读完才可覆盖','lifeBlue');
life(5,7,651,'通信载荷：C 完成后复用','lifeOrange');
life(9,10,676,'结果：J 完成后消费','lifeGreen');
out.push(T(45,684,'下一轮依赖本轮 token 的最终元数据须等结果；图重放也不能跨过此边界','small'));

const style=`<style>
text{font-family:Inter,"Noto Sans CJK SC","PingFang SC",sans-serif;fill:#162033}
.title{font-size:25px;font-weight:750}.sub{font-size:14px;fill:#526174}.section{font-size:17px;font-weight:700}.small{font-size:12px;fill:#526174}.lane{font-size:14px;font-weight:650}.tick{font-size:12px;fill:#69768a}.box{font-size:13px;font-weight:650}.end{font-size:15px;font-weight:700;fill:#1757a5}.arrowNote{font-size:11px;fill:#596577}.life{font-size:11.5px;fill:#526174}
.panel{fill:#fff;stroke:#c1cad7;stroke-width:1.2}.grid{stroke:#e8ecf2;stroke-width:1}.rule{stroke:#d0d8e3;stroke-width:1.2}
.cpu{fill:#f6f8fb;stroke:#b7c4d4;stroke-width:1.3}.gpu{fill:#e8f2ff;stroke:#1757a5;stroke-width:1.3}.comm{fill:#fff1db;stroke:#a65e0c;stroke-width:1.3}.dashed{fill:#f7f9fc;stroke:#8d9aab;stroke-width:1.4;stroke-dasharray:5 4}
.dep{fill:none;stroke:#24374f;stroke-width:1.8;marker-end:url(#arrow)}.lifeBlue{stroke:#1d5ba6;stroke-width:5;stroke-linecap:round}.lifeOrange{stroke:#af690d;stroke-width:5;stroke-linecap:round}.lifeGreen{stroke:#23806b;stroke-width:5;stroke-linecap:round}
</style>`;
const defs='<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#24374f"/></marker></defs>';
writeFileSync(join(dir,'35_decode_execution_overlap.svg'),`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1340 716" role="img">${defs}${style}<rect width="1340" height="716" fill="#fff"/>${out.join('')}</svg>\n`);
console.log(`Generated decode timeline: serial=${serial}, overlap=${overlapped}, join=${start.J}`);
