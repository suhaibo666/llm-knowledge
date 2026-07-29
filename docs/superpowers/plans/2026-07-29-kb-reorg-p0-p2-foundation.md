# 知识库整改 P0–P2:工具、快速止血、图源入库 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立链接检查器与基线,清除工作产物与琐碎坏链,图表源入库恢复可再生——为 P3+ 的内容合并铺路。

**Architecture:** 全部为机械性操作(建工具、删文件、修文本、搬文件),不做任何内容合并。唯一新代码是 `tools/check_links.py`(Obsidian 链接健康检查),TDD 开发,它是后续所有阶段的验收门。

**Tech Stack:** Python 3 + pytest(检查器);git;PowerShell/bash。工作目录:`E:\97-codes\torch_parallel\llm-knowledge`。

**Spec:** `docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md`(§4 链接治理、§7 阶段表)

**约束:** 编辑 wiki 正文时只删/只转义,不改写内容语义;所有 grep 计数在执行时以实际输出为准(盘点数字可能有 ± 小偏差,以工具输出为事实)。

---

### Task 1: 开分支 + 用 TDD 写 `tools/check_links.py`

**Files:**
- Create: `tools/check_links.py`
- Test: `tools/test_check_links.py`

- [ ] **Step 1: 确认 main 最新并开阶段分支**

```bash
cd /e/97-codes/torch_parallel/llm-knowledge
git checkout main && git pull 2>/dev/null; git status --short   # 应无未提交改动(docs/reports/ 除外)
git checkout -b reorg/p0-p2
mkdir -p tools
```

- [ ] **Step 2: 写失败测试**

创建 `tools/test_check_links.py`:

```python
"""check_links.py 的行为测试。用临时目录构造微型 wiki。"""
from pathlib import Path

from check_links import scan, target_of


def make(tmp_path: Path, files: dict) -> Path:
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return tmp_path


def test_target_of_variants():
    assert target_of("alpha|显示名") == "alpha"
    assert target_of("alpha#章节") == "alpha"
    assert target_of("dir/alpha.md") == "dir/alpha"
    # 末尾反斜杠是本库实际存在的坏链形态,规范化后仍应无法解析(报 broken)
    assert "/" in target_of("alpha\\")


def test_broken_and_ok(tmp_path):
    wiki = make(tmp_path, {
        "alpha.md": "see [[bravo]] and [[missing_page]]",
        "bravo.md": "back [[alpha]]",
        "index.md": "[[alpha]] [[bravo]]",
    })
    r, n = scan(wiki)
    assert n == 3
    assert len(r["broken"]) == 1 and "missing_page" in r["broken"][0]


def test_alias_anchor_and_path(tmp_path):
    wiki = make(tmp_path, {
        "sub/alpha.md": "[[bravo|B]] [[bravo#x]] [[../charlie]]",
        "sub/bravo.md": "",
        "charlie.md": "",
        "index.md": "[[sub/alpha]] [[sub/bravo]] [[charlie]]",
    })
    r, _ = scan(wiki)
    assert r["broken"] == []


def test_code_blocks_ignored(tmp_path):
    wiki = make(tmp_path, {
        "alpha.md": "```\n[[ghost]]\n```\n还有 `[[ghost2]]` 行内\n[[bravo]]",
        "bravo.md": "",
        "index.md": "[[alpha]] [[bravo]]",
    })
    r, _ = scan(wiki)
    assert r["broken"] == []


def test_bare_index_and_ambiguous(tmp_path):
    wiki = make(tmp_path, {
        "x/index.md": "[[index]]",       # 裸 index:既 bare 又 ambiguous
        "y/index.md": "",
        "index.md": "[[x/index]] [[y/index]]",   # 路径限定:合法
    })
    r, _ = scan(wiki)
    assert len(r["bare_index"]) == 1
    assert len(r["ambiguous"]) == 1
    assert r["broken"] == []


def test_orphan(tmp_path):
    wiki = make(tmp_path, {
        "alpha.md": "[[bravo]]",
        "bravo.md": "",
        "charlie.md": "",                 # 无入链且不在 index → 孤儿
        "index.md": "[[alpha]]",
    })
    r, _ = scan(wiki)
    assert r["orphans"] == ["charlie.md"]
```

- [ ] **Step 3: 运行测试确认失败**

```bash
cd tools && python -m pytest test_check_links.py -q
```

Expected: FAIL,`ModuleNotFoundError: No module named 'check_links'`(若无 pytest:`pip install pytest`)

- [ ] **Step 4: 写实现**

创建 `tools/check_links.py`:

```python
#!/usr/bin/env python3
"""Obsidian [[wiki-link]] 健康检查器(llm-knowledge 专用)。

检查项(扫 wiki/**/*.md,围栏代码块与行内代码不计):
  broken     - 目标解析不到任何 .md(含末尾反斜杠等畸形链接)
  ambiguous  - 裸基名命中多个文件(典型:[[index]] 命中 56 个)
  bare_index - 裸 [[index]] 链接(规则要求路径限定)
  orphans    - 无入链且未被任何 index.md 提及的非 index 页

用法:
  python tools/check_links.py            # 摘要
  python tools/check_links.py --json     # 完整清单(基线存档用)
  python tools/check_links.py --strict   # broken/ambiguous/bare_index>0 时退出码 1
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

WIKI_LINK_RE = re.compile(r"\[\[([^\[\]\n]+?)\]\]")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")


def visible_text(md: str) -> str:
    """去掉围栏代码块与行内代码后的正文。"""
    lines, in_fence = [], False
    for line in md.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            lines.append(line)
    return INLINE_CODE_RE.sub("", "\n".join(lines))


def target_of(raw: str) -> str:
    """[[a/b#sec|label]] -> a/b;反斜杠归一为 /;去 .md 后缀。"""
    t = raw.split("|", 1)[0].split("#", 1)[0].strip().replace("\\", "/")
    if t.lower().endswith(".md"):
        t = t[:-3]
    return t


def scan(wiki: Path):
    pages = sorted(wiki.rglob("*.md"))
    by_base: dict[str, list[Path]] = defaultdict(list)
    for p in pages:
        by_base[p.stem].append(p)

    report = {"broken": [], "ambiguous": [], "bare_index": [], "orphans": []}
    inbound: dict[Path, int] = defaultdict(int)

    for page in pages:
        text = visible_text(page.read_text(encoding="utf-8", errors="replace"))
        for m in WIKI_LINK_RE.finditer(text):
            t = target_of(m.group(1))
            if not t:
                continue
            loc = f"{page.relative_to(wiki).as_posix()} -> [[{m.group(1)}]]"
            if t == "index":
                report["bare_index"].append(loc)
            if "/" in t:
                # 路径限定:先按 wiki 根,再按当前页相对路径,最后唯一后缀匹配兜底
                cands = [wiki / (t + ".md"), page.parent / (t + ".md")]
                hit = next((c for c in cands if c.exists()), None)
                if hit is None:
                    tails = [p for p in pages if p.as_posix().endswith("/" + t + ".md")]
                    hit = tails[0] if len(tails) == 1 else None
                if hit is None:
                    report["broken"].append(loc)
                else:
                    inbound[hit.resolve()] += 1
            else:
                hits = by_base.get(t, [])
                if not hits:
                    report["broken"].append(loc)
                elif len(hits) > 1:
                    report["ambiguous"].append(loc)
                else:
                    inbound[hits[0].resolve()] += 1

    index_text = "\n".join(
        p.read_text(encoding="utf-8", errors="replace")
        for p in pages
        if p.name == "index.md"
    )
    report["orphans"] = [
        p.relative_to(wiki).as_posix()
        for p in pages
        if p.name != "index.md"
        and inbound[p.resolve()] == 0
        and p.stem not in index_text
    ]
    return report, len(pages)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wiki", default="wiki")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    report, n_pages = scan(Path(a.wiki))
    if a.json:
        print(json.dumps({"pages": n_pages, **report}, ensure_ascii=False, indent=1))
    else:
        print(f"pages={n_pages}")
        for k in ("broken", "ambiguous", "bare_index", "orphans"):
            print(f"{k}={len(report[k])}")
            for loc in report[k][:20]:
                print(f"  {loc}")
            if len(report[k]) > 20:
                print(f"  ... +{len(report[k]) - 20} more (--json 看全量)")
    bad = sum(len(report[k]) for k in ("broken", "ambiguous", "bare_index"))
    return 1 if (a.strict and bad) else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: 运行测试确认通过**

```bash
cd tools && python -m pytest test_check_links.py -q
```

Expected: `6 passed`

- [ ] **Step 6: 跑真实 wiki 存基线**

```bash
cd /e/97-codes/torch_parallel/llm-knowledge
python tools/check_links.py
python tools/check_links.py --json > docs/research/2026-07-29-linkcheck-baseline.json
```

Expected 量级(以实际输出为准并记录):pages≈400、broken≈160、bare_index≈71、ambiguous 数百(裸基名撞 index 等)、orphans 个位数。

- [ ] **Step 7: Commit**

```bash
git add tools/check_links.py tools/test_check_links.py docs/research/2026-07-29-linkcheck-baseline.json
git commit -m "tools: add Obsidian wiki-link health checker with baseline"
```

---

### Task 2: 删调试残留 + gitignore

**Files:**
- Delete: `torch_compile_debug/`(整目录,5 组 PyTorch 调试日志,untracked)
- Modify: `.gitignore`

- [ ] **Step 1: 确认内容后删除**

```bash
ls torch_compile_debug/          # 应只有 run_2026_07_26_*/ 调试日志目录
rm -rf torch_compile_debug
```

- [ ] **Step 2: gitignore 防复发**(torch demo 一跑就会在 CWD 重新生成)

在 `.gitignore` 末尾追加:

```
# torch.compile 运行时自动生成的调试输出
torch_compile_debug/
```

- [ ] **Step 3: Commit**

```bash
git add .gitignore && git commit -m "chore: drop torch_compile_debug leftovers and ignore future ones"
```

---

### Task 3: raw/ 层次归位(_ingest 施工单 + wanka 分析页)

**Files:**
- Move: `raw/_ingest/{INGEST_MANIFEST_block1_tim.md, changelog_entry_2026-07-25.md, fetch_block1_tim_sources.ps1}` → `docs/research/`
- Move: `raw/02_engineering/wanka_determinism_reliability_deep_analysis.md` → 视核验结果

- [ ] **Step 1: 迁移 _ingest(流程产物离开源材料层)**

```bash
git mv raw/_ingest/INGEST_MANIFEST_block1_tim.md docs/research/
git mv raw/_ingest/changelog_entry_2026-07-25.md docs/research/
git mv raw/_ingest/fetch_block1_tim_sources.ps1 docs/research/
rmdir raw/_ingest
grep -rn "_ingest" wiki/ docs/ --include="*.md" | grep -v superpowers   # 找引用旧路径处
```

对 grep 命中的引用(已知至少 `wiki/01_theory/04_posttraining/index.md` 提到 `raw/_ingest/INGEST_MANIFEST_block1_tim.md`)把路径改为 `docs/research/INGEST_MANIFEST_block1_tim.md`。

- [ ] **Step 2: 核验 wanka 文件性质再归位**

```bash
head -50 raw/02_engineering/wanka_determinism_reliability_deep_analysis.md
ls wiki/02_engineering/07_training_reliability/
```

判定规则(二选一,在 commit message 里写明依据):
- 若它是**已拆分进** `07_training_reliability/` 4 页的源汇编稿(开头是多来源清单/汇编说明)→ `git mv` 到 `docs/research/`(定位:源账本);
- 若它是**独立成稿的分析页**(有自己的论述结构,4 页未覆盖)→ `git mv` 到 `wiki/02_engineering/07_training_reliability/` 并在该目录 `index.md` 补一行条目。

- [ ] **Step 3: 验证 + Commit**

```bash
python tools/check_links.py    # broken 不得高于基线
git add -A && git commit -m "refactor: move ingest worksheets and stray analysis out of raw/"
```

---

### Task 4: 审计产物清理(55MB)+ demo 脚本归位 + reports docx

**Files:**
- Delete (git rm): `docs/audits/`(整目录)
- Move: `docs/batch_invariance_demo.py` → `tools/`
- Delete (无 git 兜底,先核验): `docs/reports/fx_graph_construction_and_transformation_design_report.docx`
- Modify: `.gitignore`

- [ ] **Step 1: 审计产物从工作区删除**(git 历史可追溯,~55MB)

```bash
git rm -r -q docs/audits
git mv docs/batch_invariance_demo.py tools/batch_invariance_demo.py
grep -rn "batch_invariance_demo" wiki/ --include="*.md"
```

grep 命中处(预期 `wiki/02_engineering/04_posttrain_frameworks/batch_invariance_guide.md` 有路径引用)把 `docs/batch_invariance_demo.py` 改为 `tools/batch_invariance_demo.py`。

- [ ] **Step 2: gitignore 防复发**

`.gitignore` 追加:

```
# 审计流水线产物(账本 jsonl 动辄十几 MB),只留本地,不入库
docs/audits/
```

- [ ] **Step 3: reports docx 删除前核验**(untracked,删了不可恢复)

```bash
ls -la docs/reports/
wc -l wiki/02_engineering/01_ai_frameworks/03_aot_autograd/fx_graph_construction_and_transformation_analysis.md
```

确认同主题 wiki 页存在(≈600 行)后:`rm -r docs/reports`。若 wiki 页不存在或行数异常(<100 行),**停止并向用户报告**,不删。

- [ ] **Step 4: Commit**

```bash
python tools/check_links.py    # broken 不升
git add -A && git commit -m "chore: drop 55MB audit artifacts, relocate demo script, remove duplicated docx"
```

---

### Task 5: 移除 113 处 `[[correction_report]]` 审计标注

**Files:**
- Modify: 约 20+ 个 wiki 页(以脚本输出为准)
- Create(用后即删): `tools/tmp_strip_correction.py`

- [ ] **Step 1: 写一次性清理脚本**

```python
# tools/tmp_strip_correction.py — 删含 [[correction_report]] 的行,压缩多余空行。用后删除。
import pathlib
import re

removed = 0
for p in pathlib.Path("wiki").rglob("*.md"):
    text = p.read_text(encoding="utf-8")
    if "[[correction_report]]" not in text:
        continue
    lines = text.splitlines(keepends=True)
    kept = [l for l in lines if "[[correction_report]]" not in l]
    removed += len(lines) - len(kept)
    p.write_text(re.sub(r"\n{3,}", "\n\n", "".join(kept)), encoding="utf-8")
print("removed lines:", removed)
```

- [ ] **Step 2: 执行并人工复查残块**

```bash
python tools/tmp_strip_correction.py     # 预期 removed lines ≈ 113
grep -rn "\[!correction\]" wiki/ --include="*.md"
```

多行 callout 被删掉首行后可能留下孤立的 `> [!correction]` 后续行:逐个打开 grep 命中处,整块删除只剩审计口径说明的残余引用行;**若某块残余里含对读者有用的技术性 caveat(讲机制而非讲审计过程),保留文字、去掉 callout 外壳**。

- [ ] **Step 3: 顺手修同类跨层链接**

`wiki/02_engineering/01_ai_frameworks/19_torch_compile_end_to_end/labs/README.md` 里的 `[[demo_delivery_report_2026-07-29]]`(目标在已删除的 docs/audits 下)改为纯文本 `demo 交付报告(审计产物,见 git 历史)`。

- [ ] **Step 4: 验证 + Commit**

```bash
python tools/check_links.py    # broken 应比基线少 ≈114
rm tools/tmp_strip_correction.py
git add -A && git commit -m "docs: strip correction_report audit callouts (113 dead cross-layer links)"
```

---

### Task 6: 修剩余琐碎坏链(示例转义/反斜杠/乱码/真实缺页)

**Files:**(逐处编辑,以 `--json` 清单为准)
- Modify: `wiki/changelog.md`、`wiki/03_posttraining/00_posttraining_source_reading_guide.md`、`wiki/02_engineering/02_train_frameworks/megatron-lm/index.md`、`wiki/02_engineering/02_train_frameworks/megatron-lm/megatron_moe_training_optimization_report.md`、`wiki/01_theory/01_models/deepseek/deepseek_v4_architecture_diagrams.md`、`wiki/01_theory/01_models/moonshot_kimi/kimi_k2.5_analysis.md`、`wiki/01_theory/01_models/moonshot_kimi/index.md`、`wiki/01_theory/01_models/scaling_laws_analysis.md`、`wiki/01_theory/01_models/index.md`、`wiki/02_engineering/01_ai_frameworks/02_dynamo/dynamo_pass_methodology.md`、`wiki/02_engineering/01_ai_frameworks/04_inductor/scheduler_analysis.md`、`wiki/02_engineering/01_ai_frameworks/05_codegen_backends/mlir/mlir_core_concepts.md`

- [ ] **Step 1: 取当前 broken 全量清单**

```bash
python tools/check_links.py --json > /tmp/lc.json   # Windows 下用 scratchpad 路径亦可
```

- [ ] **Step 2: 按四类逐处修**(每处修法固定,不改语义)

1. **末尾反斜杠(≈9 处)**:`00_posttraining_source_reading_guide.md` 的 5 处 `[[03_posttraining/12_kimi...analysis\]]`、`megatron-lm/index.md` 的 3 处 `[[../...\]]` → 删掉链接末尾的 `\`。
2. **示例文本被当链接(≈16 处)**:changelog.md 里的 `[[wiki link]]`/`[[link]]`/`[[wiki]]`/`[[SUMMARY]]`/`[[*]]`/`[[*.html]]`/`[[maybe_unused]]` 等、`megatron_moe_training_optimization_report.md` 的 8 处 `[[link]]`、`deepseek_v4_architecture_diagrams.md` 的 4 处 `[[wiki link]]` → 整个示例包进反引号(`` `[[wiki link]]` ``)。
3. **代码签名被误解析(3 处)**:`dynamo_pass_methodology.md`(`fx.GraphModule, list[Tensor`)、`scheduler_analysis.md`(`list[BaseSchedulerNode`)、`mlir_core_concepts.md`(`0, 1, 2, 3`)→ 找到原句,把整个类型签名/下标表达式包进反引号。
4. **乱码链接(≈6 处,changelog.md)**:打开命中行看上下文;能从上下文推出原意就恢复为纯文本,推不出就替换为 `(原文编码损坏,见 git 历史)`。

- [ ] **Step 3: 真实缺页转 gap 标注(4 处)**

1. `scaling_laws_analysis.md` 的 `[[scaling_laws_for_transfer_analysis]]` → 改纯文本 + 在 `wiki/01_theory/01_models/index.md` 的 Knowledge Gaps(无此节则新建)加一行 `scaling_laws_for_transfer_analysis — 待摄入(raw 有源)`。
2. `moonshot_kimi/index.md` 的 `[[Kimi VL]]`、`[[Kimi Audio]]` → 改纯文本 `Kimi VL(待建)` / `Kimi Audio(待建)`。
3. changelog.md 的 `[[llm_parallelism_analysis]]`(历史记录,页已改名)→ 包反引号。
4. `kimi_k2.5_analysis.md` 的 2 处 `[[01_theory/01_models]]`(指向目录)→ 改 `[[01_theory/01_models/index]]`。

- [ ] **Step 4: 验证 + Commit**

```bash
python tools/check_links.py
```

Expected: broken ≤ 5(残余逐条看:属于 P4+ 迁移范围的留着并记录,不属于的当场修)。

```bash
git add -A && git commit -m "docs: fix trivial broken links (escapes, backslashes, mojibake, gap markers)"
```

---

### Task 7: 修 README.md 与 wiki/index.md 失真

**Files:**
- Modify: `README.md`(整文件替换)
- Modify: `wiki/index.md`(两处)

- [ ] **Step 1: README.md 整体替换为**(不再维护具体页数,防再腐):

````markdown
# LLM Knowledge Base

LLM 训练与推理技术知识库,由 Claude Code Agent 维护。

## 结构

```
raw/            # 原始源材料(论文 PDF / 文章 / 图表源),只读
wiki/           # 生成的分析页(Obsidian vault)
├── 01_theory/        # 理论:模型家族 / 预训练 / SFT / 后训练对齐 / 推理技术 / 分布式并行
├── 02_engineering/   # 工程:AI框架 / 训练框架 / 推理框架 / 后训练框架 / GPU Kernel / 自动并行 / 训练可靠性
└── 03_posttraining/  # 后训练纵向学习域(整改中:将并入功能树,阅读路线迁往 wiki/courses/)
docs/           # 流程文档(specs / plans / research)
tools/          # 维护工具(链接检查器、图表源与再生脚本、demo)
```

各域页面清单见 [wiki/index.md](wiki/index.md)。

## 使用

页面间用 `[[wiki link]]` 交叉引用,Obsidian 打开 `wiki/` 浏览;或 `cd llm-knowledge && claude` 直接提问。

## 维护

按 [CLAUDE.md](CLAUDE.md) 定义的 Workflow 由 Agent 维护。当前结构整改:
`docs/superpowers/specs/2026-07-29-llm-knowledge-reorg-design.md`。
````

- [ ] **Step 2: wiki/index.md 修两处失真**(完整重建留给 P7,这里只除假)

1. 删掉「## 目录结构」整节的 ASCII 树(其中 `cudagraphs/`、`inductor/`、`mlir/` 三个目录不存在),替换为一句:`目录结构以文件系统与各域 index 为准;本索引只维护下方领域总览。`
2. 「02 工程实现」表:`AI框架 | ... | 45 |` 改为 `178`;并在表尾补一行 `分布式并行理论 | [[01_theory/06_distributed_parallelism/index]] | 8 | 活跃`(加到「01 理论研究」表)。

- [ ] **Step 3: 验证 + Commit**

```bash
python tools/check_links.py     # 不升
git add README.md wiki/index.md && git commit -m "docs: fix stale README and wiki index (phantom dirs, wrong counts)"
```

---

### Task 8: P2 图表源入库(可再生性)

**Files:**
- Create: `tools/figs/`(从 `.html2md/figs/` 复制)、`tools/figs/deepep/`、`tools/html2md/`、`tools/README.md`

- [ ] **Step 1: 复制源文件(不含 node_modules)**

```bash
mkdir -p tools/figs/deepep tools/html2md
cp .html2md/figs/*.html .html2md/figs/figstyle.css tools/figs/
cp .html2md/deepep_figs/*.html tools/figs/deepep/
cp .html2md/*.mjs tools/html2md/
ls .html2md/package.json 2>/dev/null && cp .html2md/package.json tools/html2md/
```

- [ ] **Step 2: 写 `tools/README.md`**

````markdown
# tools/ — 知识库维护工具

| 内容 | 用途 |
|---|---|
| `check_links.py` | Obsidian 链接健康检查(broken/ambiguous/裸 index/孤儿页)。`python tools/check_links.py`,`--json` 全量,`--strict` 做门禁 |
| `figs/*.html` + `figstyle.css` | wiki 内 png/svg 图表的**可编辑源**(dp_*、glm5_*、longcat2、training_reliability 等;deepep/ 子目录同) |
| `html2md/*.mjs` | html→md 转换与图表渲染脚本(convert / convert_kernel_sources / fix_links / gen_pp_fig) |
| `batch_invariance_demo.py` | 配套 `wiki/.../batch_invariance_guide.md` 的可执行 demo |

## 图表再生

```
cd tools/html2md && npm install   # 需网络;渲染依赖见 package.json
node gen_pp_fig.mjs               # 具体脚本用法见各脚本头部注释
```

输出 png 放回 wiki 对应 `assets/` 目录。历史工作目录 `.html2md/`(gitignored)仍在本地,与本目录内容同源。
````

- [ ] **Step 3: 再生冒烟测试(best-effort,不做门禁)**

```bash
cd tools/html2md && npm install && node gen_pp_fig.mjs; cd ../..
```

成功 → 确认输出图与 wiki 内既有 png 一致(目测)。失败(无网络/缺依赖)→ 把确切报错记进 `tools/README.md` 新增「已知问题」小节,继续。

- [ ] **Step 4: Commit**

```bash
git add tools/ && git commit -m "tools: check in figure sources and render scripts (figures reproducible)"
```

---

### Task 9: 阶段门验收 + 合回 main

- [ ] **Step 1: 全量验收**

```bash
python -m pytest tools/test_check_links.py -q          # 6 passed
python tools/check_links.py --json > docs/research/2026-07-29-linkcheck-post-p1p2.json
python tools/check_links.py                            # broken ≤ 5,较基线 -155 左右
git add docs/research/2026-07-29-linkcheck-post-p1p2.json
git commit -m "docs: record post-P1/P2 link-check stats"
git status --short                                     # 干净
```

- [ ] **Step 2: 合回 main 删分支**

```bash
git checkout main && git pull 2>/dev/null
git merge --no-ff reorg/p0-p2 -m "merge: kb-reorg P0-P2 (link checker, cleanup, figure sources)"
git branch -d reorg/p0-p2
```

- [ ] **Step 3: 在 `wiki/changelog.md` 当前季度节追加一条**(格式沿用文件内既有条目):P0–P2 完成内容一句话 + 指向 spec。commit:`docs: changelog for reorg P0-P2`。
