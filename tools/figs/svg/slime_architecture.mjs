// Conceptual responsibility map, pinned to THUDM/slime@681b3adca54105d5ecd3fb822fa0dc58a427e0f9.
// Four bands contain the same eight modules as slime 01. Band arrows show the
// main dependency direction, not runtime order or a strict no-skip layer rule.
// Optional extensions and cross-cutting controls connect at the right boundary.
// Run: node tools/figs/svg/slime_architecture.mjs > wiki/02_engineering/04_posttrain_frameworks/slime/assets/slime_architecture.svg
const out=[];
const escape=s=>s.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
const text=(x,y,s,cls='label',anchor='start')=>out.push(`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${escape(s)}</text>`);
const rect=(x,y,w,h,cls='neutral')=>out.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="7" class="${cls}"/>`);
const arrow=(d,cls='main')=>out.push(`<path d="${d}" class="arrow ${cls}" marker-end="url(#${cls})"/>`);
out.push(`<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="810" viewBox="0 0 1120 810" role="img" aria-labelledby="title desc"><title id="title">slime 四层八模块静态架构</title><desc id="desc">场景入口、应用编排、后训练合同、后端适配四层向下连接外部运行时。右侧表示可选工作流扩展和横切观测恢复。箭头不表示训练时间顺序。</desc><defs><marker id="main" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#2563EB"/></marker><marker id="aux" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#C3651F"/></marker></defs><style>text{font-family:system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;fill:#2A313B}.title{font-size:24px;font-weight:700}.heading{font-size:18px;font-weight:650}.label{font-size:17px;font-weight:600}.sub{font-size:14px;fill:#5B6470}.neutral{fill:#fff;stroke:#C7CCD3;stroke-width:1.4}.ghost{fill:#F7F6F3;stroke:#DDD9D2;stroke-width:1.2}.acc1{fill:#EAF1FD;stroke:#2563EB;stroke-width:1.4}.acc2{fill:#FCF1E6;stroke:#C3651F;stroke-width:1.4}.arrow{fill:none}.main{stroke:#2563EB;stroke-width:2.2}.aux{stroke:#C3651F;stroke-width:1.5;stroke-dasharray:5 4}</style><rect width="1120" height="810" fill="white"/>`);
text(24,34,'slime：四层软件责任与八个模块','title');
text(24,58,'静态能力视图 · 主线为依赖方向；跨层调用与返回数据见正文','sub');
const layers=[
 ['1  场景入口层',82,[['配置与场景装配','CLI / YAML / 后端参数合并']]],
 ['2  应用编排层',214,[['迭代控制','轮次 / 阶段 / 等待与完成'],['资源与生命周期','GPU 放置 / actors / server groups']]],
 ['3  后训练合同层',346,[['生成与评测','轨迹组 / reward / eval'],['样本与批次','Sample / 身份 / mask / DP 计划']]],
 ['4  后端适配层',478,[['训练执行适配','角色 / model / optimizer'],['推理服务适配','server / router / HTTP 控制'],['权重发布','转换 / 传输 / 版本可见性']]],
];
for(const [title,y,items] of layers){
 rect(24,y,790,108,'ghost');text(42,y+26,title,'heading');
 const gap=12, width=(754-gap*(items.length-1))/items.length;
 items.forEach(([name,sub],i)=>{const x=42+i*(width+gap);rect(x,y+40,width,54,name==='样本与批次'||name==='权重发布'?'acc1':'neutral');text(x+width/2,y+62,name,'label','middle');text(x+width/2,y+83,sub,'sub','middle');});
 arrow(`M419 ${y+108} V${y+129}`);
}
rect(24,610,790,78,'neutral');text(42,636,'外部运行时边界','heading');text(42,662,'Ray：资源与远程对象    Megatron：训练执行    SGLang：生成服务    PyTorch：张量','sub');
arrow('M419 688 V711');
rect(24,714,790,62,'ghost');text(42,740,'设备与通信基础设施','heading');text(42,762,'GPU / 设备运行时 / 集合通信 / 网络 / 文件系统','sub');
rect(858,214,238,112,'acc2');text(877,241,'横切能力','heading');text(877,268,'日志 · trace · 指标 · debug','sub');text(877,293,'检查 · 健康监测 · 恢复','sub');text(877,315,'挂接控制与执行边界','sub');
arrow('M858 254 H818','aux');
arrow('M868 326 H834 V532 H818','aux');
rect(858,346,238,110,'neutral');text(877,373,'侧接扩展','heading');text(877,400,'custom generate / rollout','sub');text(877,425,'agent · reward · environment','sub');text(877,446,'在 Sample 等合同处接入','sub');
arrow('M858 389 H818','aux');
text(858,633,'外部库内部实现','label');text(858,657,'不是本页已核验的源码','sub');text(858,689,'主线：软件责任逐层落实','sub');text(858,713,'虚线：可选或横切接入','sub');text(858,737,'蓝框：两个关键交界','sub');
text(24,800,'源码基线：THUDM/slime @ 681b3adca541 · 分层为依据实现重建的设计分析','sub');
out.push('</svg>');console.log(out.join('\n'));
