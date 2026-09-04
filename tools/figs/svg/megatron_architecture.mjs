// 生成 Megatron-LM 静态软件分层与能力框图。
//
// 本图只表达“代码责任位于哪一层、向哪一层依赖”，不表达一次训练的时间流程。
// 页面通过外部 SVG 引用它：
//   node tools/figs/svg/megatron_architecture.mjs > wiki/.../assets/megatron_architecture.svg

const W = 1600;
const H = 980;
const MAIN_X = 30;
const MAIN_W = 1150;
const SIDE_X = 1205;
const SIDE_W = 365;
const TOP = 88;
const GAP = 10;
const HEADER_W = 220;

const layers = [
  {
    title: '01 场景入口层',
    summary: 'root scripts · examples · tools',
    height: 88,
    items: [
      ['预训练与基础模型实验', 'pretrain_*.py'],
      ['自定义 MCore 框架', 'examples/simple MCore loop'],
      ['推理与在线服务', 'examples/inference'],
      ['后训练与 RL', 'post_training · train_rl.py'],
      ['数据、转换与诊断', 'tools'],
    ],
  },
  {
    title: '02 应用编排层',
    summary: '生命周期、完成语义与场景策略',
    height: 105,
    items: [
      ['训练控制', 'megatron/training'],
      ['推理控制', 'megatron/inference'],
      ['后训练控制', 'megatron/post_training'],
      ['RL 控制', 'megatron/rl'],
      ['弹性训练', 'megatron/elastification'],
    ],
  },
  {
    title: '03 MCore 可组合组件层',
    summary: 'models · transformer · ssm',
    detail: 'datasets · tokenizers · inference',
    height: 118,
    items: [
      ['配置与规格', 'config · ModuleSpec'],
      ['模型结构', 'models · transformer · ssm'],
      ['数据与 tokenizer', 'datasets · tokenizers'],
      ['推理与导出组件', 'inference · export · post_training'],
    ],
  },
  {
    title: '04 MCore 分布式执行层',
    summary: 'parallel_state · TP · PP · CP · EP',
    detail: 'distributed · optimizer · dist-ckpt',
    height: 132,
    items: [
      ['拓扑与通信域', 'parallel_state · process groups'],
      ['并行算子、时序与重叠', 'tensor · pipeline · context'],
      ['参数执行、提交与迁移', 'distributed · optimizer'],
      ['状态与恢复', 'dist_checkpointing · rerun state'],
    ],
  },
  {
    title: '05 加速适配层',
    summary: 'extensions · fusions · quantization',
    height: 96,
    items: [
      ['后端适配', 'Transformer Engine · local'],
      ['融合算子', 'fusions'],
      ['低精度与量化', 'FP8 · FP4 · quantization'],
      ['图与运行时适配', 'CUDA Graph · backend hooks'],
    ],
  },
  {
    title: '06 基础框架层',
    summary: 'PyTorch · torch.distributed',
    height: 70,
    items: [
      ['张量与自动微分', 'PyTorch'],
      ['进程组与 collective', 'torch.distributed'],
    ],
  },
  {
    title: '07 设备与通信基础设施层',
    summary: 'CUDA · NCCL · NVIDIA GPU',
    height: 70,
    items: [
      ['设备执行', 'CUDA · NVIDIA GPU'],
      ['跨卡与跨机通信', 'NCCL · NVLink · InfiniBand'],
    ],
  },
];

const esc = (value) => String(value)
  .replaceAll('&', '&amp;')
  .replaceAll('<', '&lt;')
  .replaceAll('>', '&gt;');

const out = [];
const put = (line) => out.push(line);

function text(x, y, value, cls, anchor = 'start') {
  put(`  <text class="${cls}" x="${x}" y="${y}" text-anchor="${anchor}">${esc(value)}</text>`);
}

function box(x, y, width, height, title, code) {
  put(`  <rect class="cap-box" x="${x}" y="${y}" width="${width}" height="${height}" rx="9"/>`);
  text(x + width / 2, y + height / 2 - 4, title, 'cap-title', 'middle');
  text(x + width / 2, y + height / 2 + 16, code, 'cap-code', 'middle');
}

function arrow(x, y1, y2, dashed = false) {
  const cls = dashed ? 'arrow dashed' : 'arrow';
  put(`  <line class="${cls}" x1="${x}" y1="${y1}" x2="${x}" y2="${y2}" marker-end="url(#arrow)"/>`);
}

let currentY = TOP;
const layerPositions = [];

for (const [index, layer] of layers.entries()) {
  const y = currentY;
  layerPositions.push({ y, height: layer.height });
  put(`  <g data-layer-index="${index + 1}">`);
  put(`    <rect class="layer" x="${MAIN_X}" y="${y}" width="${MAIN_W}" height="${layer.height}" rx="12"/>`);
  put(`    <rect class="layer-head" x="${MAIN_X}" y="${y}" width="${HEADER_W}" height="${layer.height}" rx="12"/>`);
  text(MAIN_X + 16, y + 28, layer.title, 'layer-title');
  text(MAIN_X + 16, y + 49, layer.summary, 'layer-summary');
  if (layer.detail) text(MAIN_X + 16, y + 67, layer.detail, 'layer-summary');

  const contentX = MAIN_X + HEADER_W + 14;
  const contentW = MAIN_W - HEADER_W - 28;
  const itemGap = 8;
  const itemW = (contentW - itemGap * (layer.items.length - 1)) / layer.items.length;
  const itemY = y + 12;
  const itemH = layer.height - 24;
  layer.items.forEach(([title, code], itemIndex) => {
    box(contentX + itemIndex * (itemW + itemGap), itemY, itemW, itemH, title, code);
  });
  put('  </g>');

  if (index < layers.length - 1) {
    const nextY = y + layer.height + GAP;
    arrow(MAIN_X + 15, y + layer.height + 2, nextY - 2);
  }
  currentY += layer.height + GAP;
}

const stackBottom = currentY - GAP;

// 外部生态是侧向适配，不伪装成主依赖栈中的一层。
put(`  <rect class="side-panel external" x="${SIDE_X}" y="${TOP}" width="${SIDE_W}" height="318" rx="14"/>`);
text(SIDE_X + 18, TOP + 30, '外部生态适配', 'side-title');
text(SIDE_X + 18, TOP + 50, '非主层级 · 在明确接口处接入', 'side-sub');
const integrations = [
  ['Energon', '→ 数据组件'],
  ['ModelOpt', '→ 后训练与模型规格'],
  ['TRT-LLM', '→ 导出组件'],
  ['NVRx', '→ 可靠性与重启'],
];
integrations.forEach(([name, target], index) => {
  const y = TOP + 69 + index * 52;
  put(`  <rect class="integration" x="${SIDE_X + 18}" y="${y}" width="${SIDE_W - 36}" height="42" rx="8"/>`);
  text(SIDE_X + 34, y + 18, name, 'integration-title');
  text(SIDE_X + SIDE_W - 34, y + 18, target, 'integration-target', 'end');
});
text(SIDE_X + SIDE_W / 2, TOP + 298, 'Energon · ModelOpt · TRT-LLM · NVRx', 'side-foot', 'middle');

// Lite 有自己的 runtime 和 native model；只在 primitive 边界与 MCore 共享或适配。
const liteY = TOP + 338;
const liteH = stackBottom - liteY;
put(`  <rect class="side-panel" x="${SIDE_X}" y="${liteY}" width="${SIDE_W}" height="${liteH}" rx="14"/>`);
text(SIDE_X + 18, liteY + 30, 'Megatron Lite 独立实验纵切', 'side-title');
text(SIDE_X + 18, liteY + 50, '不是 megatron.training 的下层模块', 'side-sub');
const liteItems = [
  ['公开 API', 'megatron.lite'],
  ['Runtime contracts', 'create_runtime · ModelHandle'],
  ['Native models', 'model protocols'],
  ['Training primitives', 'parallel · optimizer · checkpoint'],
];
liteItems.forEach(([title, code], index) => {
  const y = liteY + 67 + index * 67;
  box(SIDE_X + 18, y, SIDE_W - 36, 54, title, code);
  if (index < liteItems.length - 1) arrow(SIDE_X + SIDE_W / 2, y + 56, y + 65);
});
text(SIDE_X + SIDE_W / 2, liteY + liteH - 45, 'native models · runtime · primitives', 'side-foot', 'middle');
text(SIDE_X + SIDE_W / 2, liteY + liteH - 24, '虚线连接 MCore parallel / optimizer primitives', 'side-foot', 'middle');

const distributed = layerPositions[3];
const liteConnectorY = liteY + 67 + 3 * 67 + 27;
put(`  <path class="arrow dashed" d="M ${SIDE_X + 18} ${liteConnectorY} H ${MAIN_X + MAIN_W + 7} V ${distributed.y + distributed.height / 2}" marker-end="url(#arrow)"/>`);

const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Megatron-LM 静态软件分层与能力框图">
  <style>
    text{font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif}
    .canvas{fill:#fff;stroke:#E4E7EC;stroke-width:1.2}
    .title{font-size:24px;font-weight:700;fill:#1F2430}
    .subtitle{font-size:13px;fill:#7A808A}
    .layer{fill:#F8FAFC;stroke:#CBD5E1;stroke-width:1.2}
    .layer-head{fill:#EAF1FD;stroke:#2563EB;stroke-width:1.4}
    .layer-title{font-size:14px;font-weight:700;fill:#173F87}
    .layer-summary{font-size:10.5px;fill:#506784}
    .cap-box{fill:#fff;stroke:#C7CCD3;stroke-width:1.1}
    .cap-title{font-size:12.5px;font-weight:700;fill:#2A313B}
    .cap-code{font-size:9.5px;fill:#707986;font-family:"Cascadia Mono",Consolas,"Microsoft YaHei",monospace}
    .side-panel{fill:#F8FAFC;stroke:#94A3B8;stroke-width:1.2}
    .side-panel.external{fill:#FFF9F3;stroke:#C3651F}
    .side-title{font-size:14px;font-weight:700;fill:#2A313B}
    .side-sub{font-size:10.5px;fill:#7A808A}
    .side-foot{font-size:9.8px;fill:#7A808A}
    .integration{fill:#fff;stroke:#D9B18A;stroke-width:1.1}
    .integration-title{font-size:12px;font-weight:700;fill:#8A4A11}
    .integration-target{font-size:10px;fill:#8A6A4A}
    .arrow{fill:none;stroke:#2563EB;stroke-width:1.8}
    .arrow.dashed{stroke:#94A3B8;stroke-width:1.4;stroke-dasharray:5 4}
    .legend{font-size:10.5px;fill:#6B7280}
  </style>
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#64748B"/>
    </marker>
  </defs>
  <rect class="canvas" x="0.6" y="0.6" width="${W - 1.2}" height="${H - 1.2}" rx="16"/>
  <text class="title" x="${MAIN_X}" y="38">Megatron-LM 静态软件分层与能力框图</text>
  <text class="subtitle" x="${MAIN_X}" y="62">同一划分轴：由用户意图向设备执行逐层下降；层内方框表示代码能力，侧栏表示非主层级的适配关系。</text>
${out.join('\n')}
  <line class="arrow" x1="${MAIN_X + 8}" y1="${stackBottom + 24}" x2="${MAIN_X + 58}" y2="${stackBottom + 24}" marker-end="url(#arrow)"/>
  <text class="legend" x="${MAIN_X + 68}" y="${stackBottom + 28}">实线：主依赖</text>
  <line class="arrow dashed" x1="${MAIN_X + 176}" y1="${stackBottom + 24}" x2="${MAIN_X + 226}" y2="${stackBottom + 24}" marker-end="url(#arrow)"/>
  <text class="legend" x="${MAIN_X + 236}" y="${stackBottom + 28}">虚线：共享或适配</text>
  <text class="legend" x="${MAIN_X + 405}" y="${stackBottom + 28}">每个能力块第二行给出代表代码归属；时序和状态完成边界见后续动态图。</text>
</svg>`;

console.log(svg);
