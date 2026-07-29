# tools/ — 知识库维护工具

| 内容 | 用途 |
|---|---|
| `check_links.py` | Obsidian 链接健康检查（broken/ambiguous/裸 index/孤儿页）。`python tools/check_links.py`，`--json` 全量，`--strict` 做门禁 |
| `figs/*.html` + `figstyle.css` | wiki 内 png/svg 图表的**可编辑源**（dp_*、glm5_*、longcat2、training_reliability 等；deepep/ 子目录同） |
| `html2md/*.mjs` | html→md 转换与图表渲染脚本（convert / convert_kernel_sources / fix_links / gen_pp_fig） |
| `batch_invariance_demo.py` | 配套 `wiki/.../batch_invariance_guide.md` 的可执行 demo |

## 图表再生

```
cd tools/html2md && npm install   # 需网络；渲染依赖见 package.json
node gen_pp_fig.mjs               # 具体脚本用法见各脚本头部注释
```

输出 png 放回 wiki 对应 `assets/` 目录。历史工作目录 `.html2md/`（gitignored）仍在本地，与本目录内容同源。

## 已知问题

- **目录深度迁移导致的路径修正**：`.html2md/*.mjs` 原先直接位于仓库根下一层（`repo/.html2md/`），多个脚本用 `path.resolve(__dirname, '..')` 定位仓库根（`convert.mjs`、`convert_kernel_sources.mjs`、`fix_links.mjs`、`render_figs.mjs`、`rerender_ascend_figures.mjs`），`gen_pp_fig.mjs` 用 `path.join(__dirname, 'figs')` 定位同级 `figs/` 子目录。入库后脚本移到 `tools/html2md/`（仓库根下两层），且 `figs/` 改为与 `html2md/` 同级的 `tools/figs/`（而非嵌套其下）。若不修正，上述六个脚本会把仓库根误判为 `tools/`，`gen_pp_fig.mjs` 会在 `tools/html2md/figs/` 下找不到目录。**已在本目录的副本中修正**（多加一级 `..`，`gen_pp_fig.mjs` 的 `figs` 路径改为 `path.resolve(__dirname, '..', 'figs', ...)`）；`.html2md/` 里的历史副本未改动，因此两边不再逐字节相同，但脚本逻辑与产出一致。
- **deepep 渲染脚本**：`.html2md/deepep_figs/render.mjs` 依赖脚本与 `*.html` 同目录（用 `__dirname` 枚举同级 `.html` 文件），已连同 `fig{1,2,3}.html` 一并复制到 `tools/figs/deepep/render.mjs`，路径关系不变，无需修正。
- **渲染依赖 Edge**：`render_figs.mjs` / `gen_pp_fig.mjs` 等脚本硬编码 `C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe` 作为 puppeteer-core 的 `executablePath`，仅在装有该路径 Edge 的 Windows 机器上可跑；其余环境需自行改路径或换 `puppeteer`（自带 Chromium）。
