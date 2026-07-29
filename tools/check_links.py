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
