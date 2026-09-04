#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_coverage — 覆盖率的独立枚举轴：配置面 ↔ 页面归属对账。

planning-codebase-analysis 的 coverage matrix 只能覆盖「发现图里有的东西」；本工具提供一个
**不经过分析者判断、可机器枚举**的对照面——框架的用户可见配置项（dataclass 字段）。发现图漏掉
的能力通常还留着一个 flag，会在这里现形。

数据文件 docs/coverage/<domain>.yaml：
    domain / wiki_dir / repo(watchlist name) / commit / sources: [{file, class}]
    flags:
      - name: <字段名>
        from: <类名>
        owner: <页面 basename|null>     # null 且未 excluded = gap
        auto: true                       # owner 由 --generate 依「域内唯一提及」自动建议
        candidates: [...]                # 多页提及、无法自动定夺时列出
        excluded: <理由>                 # 显式排除（例如纯调试开关）

模式：
    --generate   从冻结 commit AST 枚举字段（git show，不动工作区），grep 域内页面自动建议
                 owner。**承载人工决定的行保留不动**——定了 owner（非 auto）或写了 excluded；
                 `owner: null` + candidates 只是机器建议的快照，会随当前页面内容一并刷新。
    默认         三查：
                 C1 gap            flag 无 owner 且未 excluded          （warning；--strict 计入）
                 C2 stale_owner    人工 owner 页面并未提及该 flag        （warning）
                 C3 unknown_page   owner/candidates 指向域内不存在的页    （error）
                 另列出「域内无任何 flag 归属的页」（info——概念页可以合法拥有零个 flag）。

用法：
    python tools/check_coverage.py docs/coverage/megatron-lm.yaml --generate
    python tools/check_coverage.py docs/coverage/megatron-lm.yaml [--strict]
"""
from __future__ import annotations

import argparse
import ast
import io
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path):
    import yaml

    with io.open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(path, data):
    import yaml

    with io.open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, width=100)


def find_checkout(repo_name):
    wl = load_yaml(ROOT / "docs/radar/watchlist.yaml")
    for r in wl.get("repos", []):
        if r.get("name") == repo_name:
            co = r.get("checkout")
            return (ROOT / co).resolve() if co else None
    return None


def enumerate_fields(checkout, commit, file, cls):
    """AST 枚举冻结 commit 下某类的注解字段（含默认值行），不含方法。"""
    r = subprocess.run(
        ["git", "-C", str(checkout), "show", f"{commit}:{file}"],
        capture_output=True, timeout=120,
    )
    if r.returncode != 0:
        raise SystemExit(f"git show {commit}:{file} 失败：{r.stderr.decode('utf-8', 'replace')[:200]}")
    tree = ast.parse(r.stdout.decode("utf-8", "replace"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            return [
                st.target.id
                for st in node.body
                if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name)
            ]
    raise SystemExit(f"{file} 中未找到 class {cls}")


def domain_pages(wiki_dir):
    d = ROOT / wiki_dir
    return sorted(
        p.stem for p in d.glob("*.md") if p.name != "index.md"
    )


def pages_mentioning(wiki_dir, flag):
    """域内哪些内容页以词边界提及该 flag（含 --kebab-case CLI 形式）。"""
    d = ROOT / wiki_dir
    pat = re.compile(r"(?<![\w-])(?:--)?" + re.escape(flag).replace("_", "[_-]") + r"(?![\w-])")
    hits = []
    for p in sorted(d.glob("*.md")):
        if p.name == "index.md":
            continue
        if pat.search(io.open(p, encoding="utf-8", errors="replace").read()):
            hits.append(p.stem)
    return hits


def _is_human_decided(row):
    """该行是否承载人工决定——决定了才保留，只是机器建议就该随页面内容刷新。

    人工决定 = 定了 owner（且非 auto 建议）或写了 excluded 理由。
    `owner: null` 配一串 candidates 只是「当时哪些页提到过」的快照：页面改了它就过期，
    冻住它会让后续归属判断基于陈旧候选。
    """
    if row.get("excluded"):
        return True
    return bool(row.get("owner")) and not row.get("auto", False)


def generate(cfg_path):
    cfg = load_yaml(cfg_path)
    checkout = find_checkout(cfg["repo"])
    if checkout is None or not checkout.exists():
        raise SystemExit(f"watchlist 中 {cfg['repo']} 无可用 checkout")
    old = {f["name"]: f for f in cfg.get("flags") or []}
    flags, seen = [], set()
    for src in cfg["sources"]:
        for name in enumerate_fields(checkout, cfg["commit"], src["file"], src["class"]):
            if name in seen:
                continue
            seen.add(name)
            prev = old.get(name)
            if prev is not None and _is_human_decided(prev):
                flags.append(prev)  # 人工决定保留不动
                continue
            hits = pages_mentioning(cfg["wiki_dir"], name)
            row = {"name": name, "from": src["class"]}
            if len(hits) == 1:
                row.update(owner=hits[0], auto=True)
            elif len(hits) > 1:
                row.update(owner=None, candidates=hits)
            else:
                row["owner"] = None
            flags.append(row)
    removed = [n for n in old if n not in seen]
    cfg["flags"] = flags
    dump_yaml(cfg_path, cfg)
    print(f"enumerated {len(flags)} flags from {len(cfg['sources'])} classes @ {cfg['commit'][:9]}")
    undecided = sum(1 for f in flags if f.get("candidates") and not f.get("owner"))
    unmentioned = sum(1 for f in flags
                      if not f.get("owner") and not f.get("candidates") and not f.get("excluded"))
    print(f"auto-owned={sum(1 for f in flags if f.get('owner') and f.get('auto'))} "
          f"manual={sum(1 for f in flags if f.get('owner') and not f.get('auto'))} "
          f"undecided(多页提及待定)={undecided} "
          f"unmentioned(全域未提及)={unmentioned} "
          f"excluded={sum(1 for f in flags if f.get('excluded'))}")
    if removed:
        print(f"dropped (no longer in source): {removed}")


def check(cfg_path, strict, examples):
    cfg = load_yaml(cfg_path)
    pages = set(domain_pages(cfg["wiki_dir"]))
    gaps, stale, unknown, owned_pages = [], [], [], set()
    for f in cfg.get("flags") or []:
        owner = f.get("owner")
        for ref in ([owner] if owner else []) + list(f.get("candidates") or []):
            if ref not in pages:
                unknown.append(f"{f['name']} -> {ref}")
            else:
                owned_pages.add(ref)
        if owner:
            # 与 --generate 的 pages_mentioning 同口径：词边界 + 认 CLI 的 kebab 形式。
            # 两处若不一致，只写 `--tensor-model-parallel-size` 的页会被误判成 stale。
            if not f.get("auto") and owner in pages and owner not in pages_mentioning(
                cfg["wiki_dir"], f["name"]
            ):
                stale.append(f"{f['name']} -> {owner}")
        elif not f.get("excluded"):
            gaps.append(f["name"] + ("（多页提及待定）" if f.get("candidates") else "（全域未提及）"))
    silent = sorted(pages - owned_pages)

    print(f"flags={len(cfg.get('flags') or [])}  pages={len(pages)}")
    for label, rows in [("C1 gap（无归属且未排除）", gaps), ("C2 stale_owner", stale), ("C3 unknown_page", unknown)]:
        print(f"{label}: {len(rows)}")
        for r in rows[:examples]:
            print(f"    {r}")
        if len(rows) > examples:
            print(f"    … 另 {len(rows) - examples} 条")
    print(f"info 无 flag 归属的页（概念页可合法为零）: {len(silent)}")
    for r in silent[:examples]:
        print(f"    {r}")
    errors = len(unknown)
    warnings = len(gaps) + len(stale)
    print(f"\nerrors={errors} warnings={warnings}")
    return 1 if errors or (strict and warnings) else 0


_cache = {}


def _page_text(wiki_dir, stem):
    key = (wiki_dir, stem)
    if key not in _cache:
        p = ROOT / wiki_dir / (stem + ".md")
        _cache[key] = io.open(p, encoding="utf-8", errors="replace").read() if p.exists() else ""
    return _cache[key]


def main(argv=None):
    # 必须早于 argparse：--help 与用法错误都在 parse_args 里直接写 stdout，
    # 而首行 docstring 含 GBK 编不出的字符（↔），重配晚一步就是 UnicodeEncodeError。
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config", help="docs/coverage/<domain>.yaml")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--examples", type=int, default=15)
    args = ap.parse_args(argv)
    if args.generate:
        generate(Path(args.config))
        return 0
    return check(Path(args.config), args.strict, args.examples)


if __name__ == "__main__":
    sys.exit(main())
