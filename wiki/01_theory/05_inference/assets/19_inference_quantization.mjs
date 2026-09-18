import { readFileSync, writeFileSync } from "node:fs";

const x = [-1.2, -0.4, 0.2, 0.7, 3.8];
const qmin = -7;
const qmax = 7;
const fixed = (v, n = 6) => Number(v.toFixed(n)).toString();
const escapeXml = (s) => String(s).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");

function roundEven(v) {
  const floor = Math.floor(v);
  const fraction = v - floor;
  if (Math.abs(fraction - 0.5) < 1e-10) return floor % 2 === 0 ? floor : floor + 1;
  return Math.round(v);
}

function quantize(values, scales) {
  const codes = values.map((value, i) => Math.max(qmin, Math.min(qmax, roundEven(value / scales[i]))));
  const recon = codes.map((code, i) => code * scales[i]);
  const errors = recon.map((value, i) => Math.abs(value - values[i]));
  return {
    codes,
    recon,
    errors,
    mae: errors.reduce((sum, value) => sum + value, 0) / values.length,
    max: Math.max(...errors),
  };
}

const outlierIndex = x.length - 1;
const wideMax = Math.max(...x.map(Math.abs));
const narrowMax = Math.max(...x.slice(0, outlierIndex).map(Math.abs));
if (Math.abs(x[outlierIndex]) <= narrowMax) throw new Error("The last input must remain the teaching outlier.");
const sWide = wideMax / qmax;
const sNarrow = narrowMax / qmax;
const routes = [
  {
    title: "全张量：覆盖离群值",
    color: "#2563eb",
    scaleLabel: "s = " + fixed(wideMax) + " / " + qmax + " = " + fixed(sWide),
    scaleCount: 1,
    result: quantize(x, x.map(() => sWide)),
  },
  {
    title: "窄范围：末项被 clip",
    color: "#ea580c",
    scaleLabel: "s = " + fixed(narrowMax) + " / " + qmax + " = " + fixed(sNarrow),
    scaleCount: 1,
    result: quantize(x, x.map(() => sNarrow)),
  },
  {
    title: "两组：末项单独 scale",
    color: "#2563eb",
    scaleLabel: "s0 = " + fixed(sNarrow) + "（前四项）；s1 = " + fixed(sWide) + "（末项）",
    scaleCount: 2,
    result: quantize(x, [sNarrow, sNarrow, sNarrow, sNarrow, sWide]),
  },
];

function list(values) {
  return "(" + values.map((value) => fixed(value, 3)).join(", ") + ")";
}

function arrow(x1, y1, x2, y2, color) {
  return '<path d="M ' + x1 + " " + y1 + " L " + x2 + " " + y2 + '" stroke="' + color + '" stroke-width="2.2" fill="none" marker-end="url(#arrow)"/>';
}

function card(x0, y0, w, h, title, lines, color, fill = "#ffffff") {
  let out = '<rect x="' + x0 + '" y="' + y0 + '" width="' + w + '" height="' + h + '" rx="12" fill="' + fill + '" stroke="' + color + '" stroke-width="2"/>';
  out += '<text x="' + (x0 + 16) + '" y="' + (y0 + 28) + '" class="title" fill="' + color + '">' + escapeXml(title) + "</text>";
  lines.forEach((line, i) => {
    out += '<text x="' + (x0 + 16) + '" y="' + (y0 + 56 + i * 23) + '" class="body">' + escapeXml(line) + "</text>";
  });
  return out;
}

let svg = '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="860" viewBox="0 0 1600 860">';
svg += '<style>.title{font:700 17px "Noto Sans CJK SC","PingFang SC",Arial,sans-serif}.body{font:14px "Noto Sans CJK SC","PingFang SC",Arial,sans-serif;fill:#172033}.small{font:12px "Noto Sans CJK SC","PingFang SC",Arial,sans-serif;fill:#475569}.route{font:700 19px "Noto Sans CJK SC","PingFang SC",Arial,sans-serif}</style>';
svg += '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/></marker></defs>';
svg += '<rect width="1600" height="860" fill="#f8fafc"/>';
svg += '<text x="28" y="34" class="route" fill="#0f172a">同一向量 x = ' + escapeXml(list(x)) + ' 的可复算量化路径</text>';
svg += '<text x="28" y="60" class="small">共同约定：对称整数码 q ∈ {' + qmin + ',…,' + qmax + '}，z=0，最近偶数舍入；橙色路径表示末项截断的成本。</text>';
svg += '<line x1="28" y1="78" x2="1572" y2="78" stroke="#cbd5e1"/>';

const yRows = [108, 358, 608];
routes.forEach((route, routeIndex) => {
  const y = yRows[routeIndex];
  const r = route.result;
  svg += '<text x="28" y="' + (y + 27) + '" class="route" fill="' + route.color + '">' + escapeXml(route.title) + "</text>";
  svg += card(28, y + 42, 255, 162, "输入 x", [
    list(x),
    "共同输入；末项为离群值",
    "编码：q = clip(roundEven(x / s))",
  ], route.color, "#eff6ff");
  const scaleLines = routeIndex === 2
    ? ["s0 = " + fixed(sNarrow) + "（前四项）", "s1 = " + fixed(sWide) + "（末项）", "粒度：前四项 / 末项"]
    : [route.scaleLabel, "粒度：全张量", "代码边界：[-7, 7]"];
  if (routeIndex === 2) scaleLines.push("代码边界：[-7, 7]");
  svg += card(340, y + 42, 258, 162, "选择 scale", scaleLines, route.color);
  svg += card(655, y + 42, 218, 162, "整数代码 q", [
    list(r.codes),
    routeIndex === 1 ? "末项：round(" + fixed(x[outlierIndex]) + " / s) 后 clip 至 " + qmax : "每项均在代码范围内",
    "每个元素占 4 bit（教学格式）",
  ], route.color, routeIndex === 1 ? "#fff7ed" : "#ffffff");
  svg += card(930, y + 42, 305, 162, "还原 xhat = q × s", [
    list(r.recon),
    "用编码时相同的 scale",
    routeIndex === 1 ? "末项只能还原为 " + fixed(r.recon[outlierIndex]) : "范围决定网格间距",
  ], route.color);
  svg += card(1292, y + 42, 280, 162, "绝对误差", [
    list(r.errors),
    "MAE = " + fixed(r.mae),
    "max = " + fixed(r.max) + "；scale 数 = " + route.scaleCount,
  ], route.color, routeIndex === 1 ? "#fff7ed" : "#f8fafc");
  svg += arrow(284, y + 123, 339, y + 123, "#64748b");
  svg += arrow(599, y + 123, 654, y + 123, "#64748b");
  svg += arrow(874, y + 123, 929, y + 123, "#64748b");
  svg += arrow(1236, y + 123, 1291, y + 123, "#64748b");
});
svg += '<text x="28" y="838" class="small">读图：扩大 scale 可避免截断但会增大步长；缩小 scale 会细化常见值却可能截断离群值；分组降低前四项误差，同时多存一个 scale。</text>';
svg += "</svg>";

const page = readFileSync(new URL("../19_inference_quantization_analysis.md", import.meta.url), "utf8");
const texInput = "x=(" + x.map((value) => fixed(value)).join(",\\,") + ").";
const claims = [
  texInput,
  "$s=" + fixed(wideMax) + "/" + qmax + "=" + fixed(sWide) + "$",
  "$s=" + fixed(narrowMax) + "/" + qmax + "=" + fixed(sNarrow) + "$",
  ...routes.map(({ result }) => "$" + fixed(result.mae) + " / " + fixed(result.max) + "$"),
];
for (const claim of claims) if (!page.includes(claim)) throw new Error("Quantization page and figure disagree: " + claim);
writeFileSync(new URL("19_inference_quantization.svg", import.meta.url), svg);
