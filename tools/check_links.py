#!/usr/bin/env python3
"""Obsidian [[wiki-link]] 健康检查器(llm-knowledge 专用)。

检查项(扫 wiki/**/*.md,围栏代码块与行内代码不计):
  broken     - 目标解析不到任何 .md(含末尾反斜杠等畸形链接;表格转义别名 \\| 不算)
  ambiguous  - 裸基名命中多个文件(典型:[[index]] 命中 56 个)
  bare_index - 裸 [[index]] 链接(规则要求路径限定)
  orphans    - 无入链且未被任何 index.md 提及的非 index 页
  stale_section - [[页面]] 紧跟 §N 时,目标页并无该顶层节(§N 是纯文本,不受 wikilink 检查保护)

用法:
  python tools/check_links.py            # 摘要
  python tools/check_links.py --json     # 完整清单(基线存档用)
  python tools/check_links.py --strict   # broken/ambiguous/bare_index/stale_section>0 时退出码 1
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
# [[页面]] 后紧跟的 §N —— 只认紧邻形态,避免把「链接…一句话…§N 指本页」误判
SECTION_REF_RE = re.compile(r"\A\s*(§\s*[〇一二三四五六七八九十\d][\d.〇一二三四五六七八九十]*"
                            r"(?:\s*[、/\-–~]\s*§?\s*[〇一二三四五六七八九十\d][\d.〇一二三四五六七八九十]*)*)")
TOP_SECTION_RE = re.compile(r"^##\s+([〇一二三四五六七八九十]+|\d+)\s*[.、]")
CN_NUM = {"〇": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
          "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _cn_to_int(tok: str):
    """把「三」「十」「十一」这类中文序号转成整数;转不了返回 None。"""
    if tok.isdigit():
        return int(tok)
    if tok in CN_NUM:
        return CN_NUM[tok]
    if tok.startswith("十") and len(tok) == 2 and tok[1] in CN_NUM:
        return 10 + CN_NUM[tok[1]]
    if len(tok) == 3 and tok[1] == "十" and tok[0] in CN_NUM and tok[2] in CN_NUM:
        return CN_NUM[tok[0]] * 10 + CN_NUM[tok[2]]
    return None


def top_sections(path: Path) -> set:
    """目标页现有的顶层节号集合(## N. / ## 一、两种风格)。"""
    out = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = TOP_SECTION_RE.match(line)
        if m:
            n = _cn_to_int(m.group(1))
            if n is not None:
                out.add(n)
    return out


def cited_sections(tail: str) -> list:
    """从链接后紧跟的文本里取出被引的顶层节号。"""
    m = SECTION_REF_RE.match(tail)
    if not m:
        return []
    nums = []
    for tok in re.findall(r"[〇一二三四五六七八九十\d][\d.〇一二三四五六七八九十]*", m.group(1)):
        head = tok.split(".")[0]
        n = _cn_to_int(head)
        if n is not None:
            nums.append(n)
    return nums


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
    """[[a/b#sec|label]] -> a/b;表格转义别名 [[a\\|label]] 视同 [[a|label]];反斜杠归一为 /;去 .md 后缀。"""
    raw = raw.replace("\\|", "|")
    t = raw.split("|", 1)[0].split("#", 1)[0].strip().replace("\\", "/")
    if t.lower().endswith(".md"):
        t = t[:-3]
    return t


def _check_sections(report, page, wiki, m, text, target: Path) -> None:
    """链接后若紧跟 §N,校验目标页确有该顶层节。"""
    tail = text[m.end():m.end() + 48]
    # 窗口止于行尾:跨行会把下一行的 §N 误算成本链接的引用
    tail = tail.splitlines()[0] if tail else ""
    nums = cited_sections(tail)
    if not nums:
        return
    have = top_sections(target)
    if not have:                       # 目标页没有编号小节(如纯 index),不评判
        return
    for n in nums:
        if n not in have:
            report["stale_section"].append(
                f"{page.relative_to(wiki).as_posix()} -> [[{m.group(1)}]] §{n}"
                f" (该页顶层节为 {sorted(have)})"
            )


def scan(wiki: Path):
    pages = sorted(wiki.rglob("*.md"))
    by_base: dict[str, list[Path]] = defaultdict(list)
    for p in pages:
        by_base[p.stem].append(p)

    report = {"broken": [], "ambiguous": [], "bare_index": [], "orphans": [], "stale_section": []}
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
                    _check_sections(report, page, wiki, m, text, hit)
            else:
                hits = by_base.get(t, [])
                if not hits:
                    report["broken"].append(loc)
                elif len(hits) > 1:
                    report["ambiguous"].append(loc)
                else:
                    inbound[hits[0].resolve()] += 1
                    _check_sections(report, page, wiki, m, text, hits[0])

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
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wiki", default="wiki")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    wiki = Path(a.wiki)
    if not wiki.is_dir():
        print(f"error: wiki dir not found: {wiki}", file=sys.stderr)
        return 2
    report, n_pages = scan(wiki)
    if n_pages == 0:
        print(f"error: no md pages under {wiki}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps({"pages": n_pages, **report}, ensure_ascii=False, indent=1))
    else:
        print(f"pages={n_pages}")
        for k in ("broken", "ambiguous", "bare_index", "stale_section", "orphans"):
            print(f"{k}={len(report[k])}")
            for loc in report[k][:20]:
                print(f"  {loc}")
            if len(report[k]) > 20:
                print(f"  ... +{len(report[k]) - 20} more (--json 看全量)")
    bad = sum(len(report[k]) for k in ("broken", "ambiguous", "bare_index", "stale_section"))
    return 1 if (a.strict and bad) else 0


if __name__ == "__main__":
    sys.exit(main())
