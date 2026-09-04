#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_locators — 验证 wiki 页面里仍保留的旧式 `path:line` 引用。

这是条件式遗留检查器，不是新代码分析页的引用生成规范。新页面默认用冻结基线加
`path::qualified.symbol` 的稳定源码导读；只有改动范围仍含显式行号引用时才运行本工具。

原理（全程只读，不 fetch、不动任何 checkout 的工作区/HEAD）：
  1. 从页面头部（前 HEADER_LINES 行）解析基线 `owner/repo@<hex>`（宽容匹配历史写法，
     见 docs/radar/watchlist.yaml 头注释吐槽的格式清单）；
  2. 经 watchlist.yaml 的 checkout 字段找到本地旁置检出；
  3. `git -C <checkout> ls-tree -r <commit>` 拿该 commit 的文件清单（每 (repo,commit) 一次），
     `git show <commit>:<path>` 数行数（每被引文件一次，带缓存）——所以即使 checkout HEAD
     已领先基线几百个 commit，验证仍然精确；
  4. 引用分类：
        pass               路径存在且行号在界内
        missing_file       ERROR   该 commit 树中无此路径（含 basename 唯一化失败）
        out_of_range       warning 行号超出该 commit 下的文件行数（v1 验不出行内漂移，
                                   那是 v2 内容锚点的活）
        ambiguous          warning bare 文件名在树中对应多个路径
        unanchored_page    warning 页面含代码引用但没有可解析的基线头
        unresolved_repo    warning 基线仓不在 watchlist 或无本地 checkout
        unverifiable       warning 页面钉了本机验不了的仓，引用无法归属判定（≠缺失）
        commit_unavailable warning 本地 checkout 里没有该 commit（需要 fetch，本工具不代劳）
        local              (info)  路径存在于本仓工作区（tools/…、wiki/… 自引），按工作区验行号

退出码：错误>0 → 1；--strict 时警告也计入。默认排除 wiki/changelog*（历史日志按写入
当时状态锁定，宪法明言不随迁移回写）。

用法：
    python tools/check_locators.py                 # 全库报告
    python tools/check_locators.py --dir wiki/02_engineering/02_train_frameworks/megatron-lm
    python tools/check_locators.py --strict
"""
from __future__ import annotations

import argparse
import io
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict

from check_math import collect_changed_markdown
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER_LINES = 40

CITE_RE = re.compile(
    r"(?<![\w/])"
    r"((?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+"
    r"\.(?:py|pyi|pyx|c|cc|cpp|cxx|h|hpp|cu|cuh|rs|go|ts|tsx|js|mjs|sh|bash|yaml|yml|json|proto|cmake))"
    r":(\d+)(?:-(\d+))?"
)
SLUG_BASE_RE = re.compile(r"([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)@([0-9a-f]{7,40})\b")
WORD_BASE_RE = re.compile(r"([A-Za-z0-9_.-]{2,})\s*@\s*([0-9a-f]{7,40})\b")
BRANCH_WORDS = {"main", "master", "dev", "trunk"}


def load_watchlist():
    import yaml

    with io.open(ROOT / "docs/radar/watchlist.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    entries = []
    for r in data.get("repos", []):
        co = r.get("checkout")
        entries.append(
            {
                "name": r.get("name", ""),
                "repo": (r.get("repo") or "").strip(),
                "checkout": (ROOT / co).resolve() if co else None,
            }
        )
    return entries


def _entry_keys(e):
    keys = {e["name"].lower(), e["repo"].lower()}
    if "/" in e["repo"]:
        keys.add(e["repo"].split("/", 1)[1].lower())
    if e["checkout"] is not None:
        keys.add(e["checkout"].name.lower())
    keys.discard("")
    return keys


def parse_baselines(header_text: str, entries):
    """返回 [(entry, commit)]，去重；解析不出仓名的 hash 忽略。"""
    found, seen = [], set()

    def push(name, commit):
        name = name.lower()
        for e in entries:
            if name in _entry_keys(e):
                key = (e["repo"], commit)
                if key not in seen:
                    seen.add(key)
                    found.append((e, commit))
                return True
        return False

    unresolved = []
    for m in SLUG_BASE_RE.finditer(header_text):
        if not push(m.group(1), m.group(2)):
            unresolved.append(m.group(1))
    for m in WORD_BASE_RE.finditer(header_text):
        word = m.group(1)
        if "/" in word:
            continue  # slug 形式上面已处理
        if word.lower() in BRANCH_WORDS:
            # `verl main @ 254a23ed`：分支词前面那个词才是仓名
            prefix = header_text[: m.start(1)].rstrip()
            prev = re.search(r"([A-Za-z0-9_.\-]+)[`\s，。:：]*$", prefix)
            word = prev.group(1) if prev else word
        if not push(word, m.group(2)):
            unresolved.append(word)
    return found, unresolved


class GitView:
    """对 (checkout, commit) 的只读视图：文件清单 + 行数，全部缓存。"""

    def __init__(self):
        self._tree = {}
        self._lines = {}
        self._commit_ok = {}

    def _run(self, checkout, args):
        return subprocess.run(
            ["git", "-C", str(checkout)] + args,
            capture_output=True,
            timeout=120,
        )

    def commit_exists(self, checkout, commit):
        key = (checkout, commit)
        if key not in self._commit_ok:
            r = self._run(checkout, ["cat-file", "-e", commit + "^{commit}"])
            self._commit_ok[key] = r.returncode == 0
        return self._commit_ok[key]

    def tree(self, checkout, commit):
        key = (checkout, commit)
        if key not in self._tree:
            r = self._run(checkout, ["ls-tree", "-r", "--name-only", commit])
            paths = r.stdout.decode("utf-8", "replace").splitlines() if r.returncode == 0 else []
            by_base = defaultdict(list)
            for p in paths:
                by_base[p.rsplit("/", 1)[-1]].append(p)
            self._tree[key] = (set(paths), by_base)
        return self._tree[key]

    def nlines(self, checkout, commit, path):
        key = (checkout, commit, path)
        if key not in self._lines:
            r = self._run(checkout, ["show", f"{commit}:{path}"])
            self._lines[key] = (
                r.stdout.count(b"\n") + (0 if r.stdout.endswith(b"\n") or not r.stdout else 1)
                if r.returncode == 0
                else None
            )
        return self._lines[key]


def resolve_in_tree(path, tree_paths, by_base):
    """返回 (resolved_path|None, 'exact'|'basename'|'suffix'|'ambiguous'|'missing')."""
    path = path.lstrip("./")
    if path in tree_paths:
        return path, "exact"
    if "/" not in path:
        cands = by_base.get(path, [])
    else:
        cands = [p for p in tree_paths if p.endswith("/" + path)]
    if len(cands) == 1:
        return cands[0], "basename" if "/" not in path else "suffix"
    if len(cands) > 1:
        return None, "ambiguous"
    return None, "missing"


def audit_page(md_path: Path, entries, gv: GitView):
    """返回 findings: [(category, detail)]；category ∈ 上述分类。"""
    text = io.open(md_path, encoding="utf-8", errors="replace").read()
    header = "\n".join(text.splitlines()[:HEADER_LINES])
    baselines, unresolved_names = parse_baselines(header, entries)
    # slug 形式（owner/repo@hex）另扫全文——“一页钉两个仓”的第二个基线常钉在小节里；
    # 词形式仍只认头部，避免正文顺带提到别仓 commit 时误判归属。
    body_slugs = chr(10).join(f"{a}@{b}" for a, b in SLUG_BASE_RE.findall(text))
    more, _ = parse_baselines(body_slugs, entries)
    seen_bl = {(e["repo"], c) for e, c in baselines}
    for e, c in more:
        if (e["repo"], c) not in seen_bl:
            baselines.append((e, c)); seen_bl.add((e["repo"], c))

    cites = []
    for m in CITE_RE.finditer(text):
        pre = text[max(0, m.start() - 8) : m.start()]
        if "://" in pre:
            continue
        cites.append((m.group(1), int(m.group(2)), int(m.group(3) or m.group(2))))
    if not cites:
        return []

    findings = []
    usable = []
    had_unresolved = bool(unresolved_names)
    for e, commit in baselines:
        if e["checkout"] is None or not e["checkout"].exists():
            findings.append(("unresolved_repo", f"{md_path}: {e['repo']}@{commit[:9]} 无本地 checkout"))
            had_unresolved = True
            continue
        if not gv.commit_exists(e["checkout"], commit):
            findings.append(
                ("commit_unavailable", f"{md_path}: {e['repo']}@{commit[:9]} 不在 {e['checkout'].name} 本地对象库")
            )
            had_unresolved = True
            continue
        usable.append((e, commit))
    for name in unresolved_names:
        findings.append(("unknown_repo", f"{md_path}: 基线仓 `{name}` 不在 watchlist"))

    if not usable:
        if not baselines and not unresolved_names:
            findings.append(("unanchored_page", f"{md_path}: 含 {len(cites)} 处代码引用但无可解析基线头"))
        return findings

    for path, lo, hi in cites:
        verdict = None
        for e, commit in usable:
            tree_paths, by_base = gv.tree(e["checkout"], commit)
            resolved, how = resolve_in_tree(path, tree_paths, by_base)
            if how == "ambiguous":
                verdict = ("ambiguous", f"{md_path}: `{path}:{lo}` 在 {e['repo']} 树中多处同名")
                continue
            if resolved is None:
                continue
            n = gv.nlines(e["checkout"], commit, resolved)
            if n is not None and hi > n:
                verdict = ("out_of_range", f"{md_path}: `{path}:{lo}{'-%d' % hi if hi != lo else ''}` 超界（{e['repo']}@{commit[:9]} 下 {resolved} 共 {n} 行）")
            else:
                verdict = ("pass", None)
            break
        if verdict is None:
            local = ROOT / path
            if local.exists():
                try:
                    n = sum(1 for _ in io.open(local, encoding="utf-8", errors="replace"))
                except OSError:
                    n = None
                verdict = ("pass", None) if (n is None or hi <= n) else (
                    "out_of_range", f"{md_path}: `{path}:{lo}` 超界（本仓工作区 {n} 行）")
            elif had_unresolved:
                # 页面钉了一个本机验不了的仓——引用可能属于它：不可验证，不是缺失
                verdict = ("unverifiable", f"{md_path}: `{path}:{lo}` 无法验证（页面另钉有不可解析的基线仓）")
            else:
                verdict = ("missing_file", f"{md_path}: `{path}:{lo}` 不在任何页面基线的树中")
        findings.append(verdict)
    return findings


# 三档分级：
#   ERROR —— 引用的路径压根不在任何基线树里，确定性缺陷。
#   WARN  —— 页面或配置能修的：行号超界、同名歧义、缺基线头、仓不在 watchlist。
#            `out_of_range` 只约束仍在用 `path:line` 的存量；宪法已把行号定为可选，
#            新页面改用稳定符号锚点后这一类会自然归零。
#   ENV   —— 本机没克隆、对象库里没这个 commit：作者改文档也修不掉。默认单独列出、
#            不计入告警与退出码；要连它一起卡，加 --include-env。
ERROR_CATS = {"missing_file"}
WARN_CATS = {"out_of_range", "ambiguous", "unanchored_page", "unknown_repo"}
ENV_CATS = {"unresolved_repo", "commit_unavailable", "unverifiable"}
REPORT_ORDER = [
    "missing_file", "out_of_range", "ambiguous", "unanchored_page", "unknown_repo",
    "unresolved_repo", "commit_unavailable", "unverifiable",
]


def _is_changelog(rel: str) -> bool:
    return rel == "wiki/changelog.md" or rel.startswith("wiki/changelog/")


def pages_to_audit(scan_root, explicit=None):
    """审计面：显式列表（--changed）优先，否则扫目录；changelog 两条路径都豁免。"""
    if explicit is None:
        candidates = sorted(Path(scan_root).rglob("*.md"))
    else:
        candidates = sorted(explicit)
    pages = []
    for md in candidates:
        try:
            rel = md.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            rel = md.as_posix()
        if _is_changelog(rel):
            continue
        pages.append(md)
    return pages


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default="wiki", help="扫描目录（默认 wiki）")
    ap.add_argument("--strict", action="store_true", help="警告也计入退出码")
    ap.add_argument("--changed", action="store_true", help="只审计 git 中已改动的 Markdown")
    ap.add_argument("--include-env", action="store_true", help="环境缺口也计入告警与退出码")
    ap.add_argument("--examples", type=int, default=12, help="每类最多打印几条明细")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    entries = load_watchlist()
    gv = GitView()
    counts = Counter()
    details = defaultdict(list)
    pages = 0

    explicit = collect_changed_markdown(ROOT) if args.changed else None
    for md in pages_to_audit(ROOT / args.dir, explicit):
        pages += 1
        for cat, detail in audit_page(md, entries, gv):
            counts[cat] += 1
            if detail:
                details[cat].append(detail)

    print(f"pages scanned: {pages}")
    total = sum(counts.values())
    print(f"citations audited: {total}  (pass={counts['pass']})")
    env_header = False
    for cat in REPORT_ORDER:
        if cat in ENV_CATS and not env_header:
            print("--- 环境缺口（本机没有该仓/该 commit，非页面缺陷）---")
            env_header = True
        print(f"{cat}={counts[cat]}")
        for d in details[cat][: args.examples]:
            print(f"    {d}")
        if len(details[cat]) > args.examples:
            print(f"    … 另 {len(details[cat]) - args.examples} 条")

    errors = sum(counts[c] for c in ERROR_CATS)
    warnings = sum(counts[c] for c in WARN_CATS)
    env = sum(counts[c] for c in ENV_CATS)
    if args.include_env:
        warnings += env
    print(f"\nerrors={errors} warnings={warnings} env={env}")
    if errors or (args.strict and warnings):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
