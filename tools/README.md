# tools/ — 知识库维护工具

| 内容 | 用途 |
|---|---|
| `check_links.py` | Obsidian 链接健康检查（broken/ambiguous/裸 index/孤儿页）。`python tools/check_links.py`，`--json` 全量，`--strict` 做门禁 |
| `check_math.py` | Obsidian/MathJax 公式检查。`python tools/check_math.py --strict <文件或目录>` 检查指定内容，`python tools/check_math.py --changed --strict` 检查 Git 变更 |
| `docs-site/` | 锁定版本的 Quartz 本地文档启动器、配置、补丁、单元测试与浏览器验收脚本 |
| `figs/*.html` + `figstyle.css` | wiki 内 png/svg 图表的**可编辑源**（dp_*、glm5_*、longcat2、training_reliability 等；deepep/ 子目录同） |
| `html2md/*.mjs` | html→md 转换与图表渲染脚本（convert / convert_kernel_sources / fix_links / gen_pp_fig） |
| `batch_invariance_demo.py` | 配套 `wiki/.../batch_invariance_guide.md` 的可执行 demo |

## 图表再生

```
cd tools/html2md && npm install   # 需网络；渲染依赖见 package.json
node gen_pp_fig.mjs               # 具体脚本用法见各脚本头部注释
```

输出 png 放回 wiki 对应 `assets/` 目录。历史工作目录 `.html2md/`（gitignored）仍在本地，与本目录内容同源。

## 本地文档启动器

`tools/docs-site/` 负责把 `wiki/` 原样交给 Quartz 5，并将可变运行时、输出、staging 和 npm 缓存限制在 `.cache/llm-knowledge-docs/`。根命令为 `npm run docs`、`npm run docs:build`、`npm run docs:test` 和 `npm run docs:repair`。

仓库跟踪的配置层包含三类适配：

1. HTTP 与热更新 WebSocket 都固定绑定 `127.0.0.1`，且启动前检查成对端口。
2. 移除远程预连接和默认远程 OG 资源，使用系统字体、MathJax SVG，并把 Mermaid 及完整依赖树 vendoring 到本地静态目录。
3. 兼容本 vault 的 Obsidian 路径后缀 wikilink、相对媒体资源、无 frontmatter 页面与 breadcrumbs、无可选 Footer 的精简布局和 Node 22+ 日志接口，同时以严格补丁上下文阻止静默版本漂移。

网页服务只解决人类浏览与本地搜索；Agent/MCP 查询仍走 qmd，二者读取同一份 `wiki/`，互不替代。

升级 Quartz 或任一社区插件必须是显式维护操作：更新 `runtime-manifest.json` 与 `quartz.lock.json` 的 commit，逐项重建版本限定补丁，更新相应单元测试，并运行 `npm run docs:test`。不要直接修改 `.cache/llm-knowledge-docs/quartz`，因为运行时完整性检查会拒绝未记录的变化并要求执行 `npm run docs:repair`。

## 已知问题

- **目录深度迁移导致的路径修正**：`.html2md/*.mjs` 原先直接位于仓库根下一层（`repo/.html2md/`），多个脚本用 `path.resolve(__dirname, '..')` 定位仓库根（`convert.mjs`、`convert_kernel_sources.mjs`、`fix_links.mjs`、`render_figs.mjs`、`rerender_ascend_figures.mjs`），`gen_pp_fig.mjs` 用 `path.join(__dirname, 'figs')` 定位同级 `figs/` 子目录。入库后脚本移到 `tools/html2md/`（仓库根下两层），且 `figs/` 改为与 `html2md/` 同级的 `tools/figs/`（而非嵌套其下）。若不修正，上述六个脚本会把仓库根误判为 `tools/`，`gen_pp_fig.mjs` 会在 `tools/html2md/figs/` 下找不到目录。**已在本目录的副本中修正**（多加一级 `..`，`gen_pp_fig.mjs` 的 `figs` 路径改为 `path.resolve(__dirname, '..', 'figs', ...)`）；`.html2md/` 里的历史副本未改动，因此两边不再逐字节相同，但脚本逻辑与产出一致。
- **deepep 渲染脚本**：`.html2md/deepep_figs/render.mjs` 依赖脚本与 `*.html` 同目录（用 `__dirname` 枚举同级 `.html` 文件），已连同 `fig{1,2,3}.html` 一并复制到 `tools/figs/deepep/render.mjs`，路径关系不变，无需修正。
- **渲染依赖 Edge**：`render_figs.mjs` / `gen_pp_fig.mjs` 等脚本硬编码 `C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe` 作为 puppeteer-core 的 `executablePath`，仅在装有该路径 Edge 的 Windows 机器上可跑；其余环境需自行改路径或换 `puppeteer`（自带 Chromium）。
- **冒烟测试记录**：2026-07-29 已端到端验证——`npm install` 成功；`gen_pp_fig.mjs` 再生的 `dp_pipeline_parallel.html` 与原件字节一致；`render_figs.mjs` 渲出的 PNG 与 `wiki/01_theory/06_distributed_parallelism/assets/dp_pipeline_parallel_fig1.png` 字节一致（369645 bytes）。
