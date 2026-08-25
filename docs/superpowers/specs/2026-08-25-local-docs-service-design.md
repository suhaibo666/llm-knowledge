# 本地 Markdown 文档站设计

## 1. 背景与结论

`llm-knowledge` 当前以 `wiki/` 作为 Obsidian 内容树，并通过 `qmd` 为 Agent 提供检索。仓库没有面向人的浏览器文档站：`qmd` 没有网页 UI，现有 Puppeteer 只用于抓取与图表渲染，也不负责把 Markdown 组织成可导航的网站。

本次采用 **Quartz 5** 作为站点引擎，并在仓库外部依赖与仓库内容之间增加一个很薄的启动层。Quartz 原生支持 Obsidian Flavored Markdown、最短路径双链、callout、Mermaid、LaTeX、目录树、全文搜索和页内目录，能直接读取现有 `wiki/`，不需要迁移或生成第二份 Markdown。

工具基线固定为：

- 仓库：`jackyzha0/quartz`
- 版本：`v5.0.0`
- commit：`ab346fa66a895e12d63a308e70ce330ba795822a`
- 本机前置条件：Node.js 22 及以上、npm 10.9.2 及以上、Git

## 2. 目标与非目标

### 2.1 目标

1. 在仓库根目录执行一个命令即可启动本地文档服务。
2. `wiki/**/*.md` 与相邻 assets 是唯一内容源，站点层不得改写它们。
3. 在浏览器中正确呈现当前仓库使用的 Obsidian 双链、路径双链、标题锚点、callout、Mermaid、LaTeX、代码块、表格、内联 HTML 和相对图片。
4. 提供代码仓文档常见的左侧目录树、顶部搜索、正文、右侧页内目录、面包屑和深色模式。
5. 修改 Markdown 后自动重建并刷新浏览器。
6. 首次初始化后可在无网络环境下重复启动。
7. 依赖、缓存和生成站点不得污染 Git 工作树。

### 2.2 非目标

- 不修改、复制或批量规范化 `wiki/` 中的 Markdown。
- 不展示 `raw/`、`docs/` 或仓库其他目录。
- 不提供公网部署、局域网共享、认证、编辑器或评论系统。
- 不在首版把 `qmd` 包装成 HTTP 搜索服务；浏览器搜索使用 Quartz 内置索引。
- 不展示知识图谱、backlinks、最近文章、RSS、站点地图或社交分享卡。
- 不把 Quartz 源码、`node_modules`、生成 HTML 或运行缓存提交进本仓库。

## 3. 方案比较

### 3.1 Quartz 5（采用）

Quartz 的 Obsidian 模板与本库方言直接匹配，尤其是 `[[bare_name]]`、`[[path/index|显示名]]`、`[[page#heading]]`、`> [!type]`、Mermaid 和 dollar 定界符公式。它还自带目录浏览、中文全文搜索、页内目录和热重载。站点适配集中在配置与启动脚本，不需要维护 Markdown AST 转换器。

代价是 Quartz 目前以完整仓库形式分发，而非普通 npm library。启动层需要在首次运行时获取一个固定版本，并管理独立缓存。

### 3.2 MkDocs Material（不采用）

MkDocs Material 的传统 docs 外观很成熟，也支持 Mermaid、公式和 admonition，但它的原生 admonition 使用 `!!!` 语法；Obsidian callout 与双链仍需自定义预处理或第三方插件。对 405 篇包含双链、141 篇包含 callout 的现有内容而言，这会把主要风险转移到自研兼容层。

### 3.3 VitePress（不采用）

VitePress 可直接复用 Node.js，文档 UI 也合适，但需要组合或实现双链解析、Obsidian callout、Mermaid 和搜索适配。相比 Quartz，它没有为本库带来足以抵消额外维护成本的收益。

## 4. 总体架构

数据流保持单向：

1. 启动器定位仓库根目录与只读内容目录 `wiki/`。
2. 启动器确认本地 Quartz runtime 缓存存在且 commit 正确；首次运行时从固定 tag 初始化缓存。
3. 启动器把仓库内受版本控制的 Quartz 配置复制到 runtime 工作目录，并对固定 commit 应用受版本控制的 loopback 补丁。
4. 补丁只把 Quartz HTTP server 与热重载 WebSocket 的监听 host 从默认的所有网卡收窄为 `127.0.0.1`；补丁上下文不匹配时立即停止。
5. Quartz 以 `wiki/` 为显式 content directory 构建到被忽略的输出目录。
6. Quartz preview server 监视 `wiki/` 与站点配置的变化。
7. 启动器在 HTTP 健康检查成功后打开默认浏览器。

`wiki/` 不经过同步目录、符号链接或临时改写。这样避免 Windows symlink 权限问题，也消除了“网站副本落后于知识库”的状态分叉。

## 5. 仓库组件

### 5.1 根命令

根目录新增私有 `package.json`，提供以下稳定入口：

- `npm run docs`：初始化依赖并启动服务，默认地址 `http://127.0.0.1:8080`。
- `npm run docs:build`：只执行全量静态构建，用于验收与 CI 式检查。
- `npm run docs:test`：执行启动层测试与浏览器冒烟测试。
- `npm run docs:repair`：只重建仓库专用 Quartz runtime，不触碰 `wiki/` 或系统级缓存。

端口可通过 `npm run docs -- --port 8081` 显式覆盖。服务地址仍固定为 loopback，不提供 `--host 0.0.0.0` 快捷入口。

### 5.2 启动器

`tools/docs-site/` 保存跨平台 Node.js 启动代码。启动器职责严格限定为：

- 校验 Node、npm、Git 与 `wiki/index.md`；
- 管理固定位置的 disposable runtime；
- 校验 Quartz commit，避免静默使用漂移版本；
- 幂等校验并应用 `tools/docs-site/patches/quartz-v5-loopback.patch`；
- 在首次运行时执行 clone、`npm ci` 与所需插件安装；
- 把受控配置同步到 runtime；
- 启动 Quartz 并转发退出码与 Ctrl+C；
- 等待本地 HTTP 服务就绪后打开浏览器。

启动器不解析 Markdown，不实现页面路由，也不接管 Quartz 的 watcher。

### 5.3 Runtime 与缓存

runtime 位于仓库内一个被 Git 忽略的专用缓存目录，例如 `.cache/llm-knowledge-docs/quartz/`；生成站点位于同一缓存树下。不得复用或删除用户的通用 npm、Git、Quartz 或系统缓存。

首次初始化采用“临时目录构建完成后原子切换”的方式，避免网络中断留下看似可用的半成品。发现 commit 不匹配时启动器停止并提示 `npm run docs:repair`，不自动递归删除未知目录。repair 在解析和验证绝对路径后只替换专用 runtime；旧 runtime 先移动到同一缓存根内的隔离目录，成功后再报告清理结果。

仓库跟踪 Quartz 核心 commit、loopback 补丁、站点配置及插件锁定信息。runtime 的允许状态只有“固定 commit + 精确匹配的已应用补丁”；出现其他已跟踪文件差异时停止并提示 repair。完成首次初始化后，普通启动不得访问网络。

### 5.4 Quartz 配置

配置从 Quartz Obsidian 模板裁剪，保留：

- Obsidian Flavored Markdown；
- GitHub Flavored Markdown；
- `shortest` 双链解析；
- Mermaid、LaTeX 和语法高亮；
- content page 与 folder page；
- Explorer、Search、Table of Contents、Breadcrumbs、Page Title、Darkmode；
- 中文 locale 与系统中英文混排字体栈。

关闭 Graph、Backlinks、Popover、Reader Mode、Recent Notes、Comments、RSS、sitemap、OG image、CNAME、Canvas、Bases、encrypted pages、远程字体和所有 analytics。页面入口使用 `wiki/index.md`，数字前缀目录按名称自然排序。浏览器页面只引用本地生成的 CSS、JavaScript、字体回退和内容 assets，不依赖 CDN。

站点标题为 `LLM Knowledge Wiki`。默认主题只做必要的 docs 布局调整，不在首版引入独立设计系统或大规模自定义组件。

### 5.5 搜索边界

浏览器搜索使用 Quartz 的客户端全文索引，因为它直接对应已生成页面，并支持中文分词。现有 `qmd` 继续承担 Agent/MCP 的 BM25 检索，两者职责如下：

- Quartz Search：人在浏览器内找页面并跳转；
- `qmd`：Agent 查找、读取并综合知识库内容。

首版不让网页进程调用 `qmd`，从而保持站点为静态内容加本地文件服务器，无额外 API、数据库生命周期或 Windows shell 兼容问题。

## 6. 内容兼容策略

站点对内容采取“解析时兼容、源文件零修改”原则：

- 裸基名双链按 Obsidian shortest 策略解析；仓库已有 `check_links.py` 保证基名唯一。
- 路径双链从 `wiki/` 根解析，别名与标题锚点保留。
- 表格内转义别名 `[[target\|label]]` 必须正确显示。
- `> [!note]`、`> [!warning]`、`> [!contradiction]`、`> [!deprecated]` 等 callout 由 Quartz OFM 插件渲染；未知类型使用安全的默认 callout 样式。
- ` ```mermaid ` 围栏由 Mermaid 插件在页面内渲染。
- `$...$` 和独占行的 `$$` 块由 KaTeX 渲染，沿用本仓库公式规范。
- `assets/...` 相对图片及 SVG 按 Markdown 文件所在目录解析并复制。
- 合法内联 HTML 保留；脚本型 HTML 不作为受支持内容。

若某一语法在 Quartz 中存在真实不兼容，优先增加站点侧插件或配置；只有在内容本身违反 `CLAUDE.md` 时才允许另行修正文档，且不把这种修复混入站点实现。

## 7. 启动与错误处理

正常启动顺序为：版本检查、runtime 检查、必要时初始化、配置同步、Quartz build/serve、HTTP 健康检查、打开浏览器。日志明确区分首次安装、构建和服务阶段。

错误边界：

- 前置版本不足：在 clone 或安装前失败，并显示检测值和最低要求。
- 首次运行无网络：保留可诊断日志，不留下已宣告可用的 runtime。
- runtime commit 漂移：停止并提示显式 repair 操作，不悄悄升级或降级。
- loopback 补丁不适用或 runtime 出现额外差异：停止并提示 repair，不以宽松文本替换继续运行。
- 端口占用：非零退出并提示 `--port` 用法。
- Quartz 构建失败：透传文件与错误，禁止以部分站点继续服务。
- 浏览器无法自动打开：服务保持运行并打印 URL；这不是构建失败。
- Ctrl+C：终止 Quartz 子进程并返回终端，不遗留后台服务。

## 8. 安全与可恢复性

- HTTP server 仅监听 `127.0.0.1`，不暴露到局域网。
- 站点不提供认证，因为它没有网络暴露面。
- 运行目录只允许落在经解析并校验的仓库专用缓存路径内。
- 所有清理操作只针对该专用 runtime；删除前验证绝对路径仍位于缓存根下。
- Quartz 与插件版本固定，不在普通启动时执行自动升级。
- HTTP 与 WebSocket 两个监听器都经固定版本补丁显式绑定到 `127.0.0.1`；验收不能只检查日志中的 URL。
- `wiki/` 只作为输入，启动器不以写权限需求为前提。

## 9. 测试与验收

### 9.1 自动测试

启动层测试至少覆盖：

- 仓库根目录与缓存路径解析；
- Node/npm 版本门槛；
- 参数与端口校验；
- runtime commit 检查；
- loopback 补丁首次应用、重复应用和上下文漂移拒绝；
- 浏览器打开失败不杀死服务；
- 子进程退出码和 Ctrl+C 转发。

浏览器冒烟测试复用 `tools/html2md` 已安装的 `puppeteer-core` 与本机 Edge/Chrome，检查：

- 首页、左侧目录、搜索入口和深色模式按钮存在；
- 一篇代表页的裸双链、路径双链和标题锚点可导航；
- callout 生成结构化容器；
- Mermaid 生成 SVG；
- 行内与块级公式生成 KaTeX DOM；
- PNG/SVG 图片加载成功；
- 页面没有向 loopback 地址以外发起资源请求；
- 页面没有未捕获的浏览器错误或失败资源。

### 9.2 全量验收

最终必须执行：

1. `python tools/check_links.py --strict`；
2. Quartz 对全部 `wiki/` 内容的干净构建；
3. 启动本地服务并运行浏览器冒烟测试；
4. 从系统监听表确认 HTTP 与 WebSocket 只绑定 `127.0.0.1`；
5. `git diff --check`；
6. 对比 Git 状态，确认 `wiki/` 没有任何变更，runtime、输出和依赖均未被跟踪。

验收成功标准：用户从干净 checkout 首次执行 `npm run docs` 能完成初始化并打开首页；后续在已初始化机器上离线启动成功；现有核心 Markdown 方言在浏览器中可读、可导航、可搜索。

## 10. 文档与维护

根 `README.md` 增加本地文档站的前置条件、启动命令、默认地址、端口覆盖、首次联网说明和缓存修复入口。`tools/README.md` 增加站点工具职责，但不在根 README 维护页面数量或深层目录树。

升级 Quartz 必须是显式维护动作：更新固定 commit，重新生成/核验插件锁定信息，执行全量构建与浏览器冒烟测试，并在变更说明中记录兼容性结果。
