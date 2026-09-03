#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_feature_tree — 代码仓功能树的 ownership manifest ↔ 冻结 commit 双向对账（只读）。

feature-tree-analysis 技能的机械门禁。manifest（宿主为本 wiki 时在 docs/feature-tree/<domain>.yaml，
独立交付时在输出目录的 feature-tree.yaml）是树（nodes）、叶子（leaves）、代码归属、排除表、状态与评审
记录（reviews）的唯一权威数据；本工具在冻结 commit 下枚举三个机械面并与之求差集，不改动任何文件。
字段说明见 skills/feature-tree-analysis/references/tree-method.md §5。相对路径一律相对 manifest 所在
目录解析；`checkout:` 显式路径优先，否则 `repo:` 走 docs/radar/watchlist.yaml 的 checkout。

检查项（X1 有错时其余检查不再执行）：
    X1 schema            键白名单/类型/枚举；commit 必须 40 位 hex；leaves/nodes 非空；include 必须是
                         非空 glob 列表；排除项必须带 reason；spec/delivery 阶段 spec_dir 必填   error
    V0 scope             每个 include 必须在冻结 commit 下命中 ≥1 文件；范围不能为空            error
    F1 unowned_file      范围内文件无叶子认领且未被排除                                        warning
    F2 phantom_file      叶子认领的文件/glob 在冻结 commit 下不存在                             error
    F3 bad_entry_anchor  叶子入口 path::qualified.symbol 文件不存在，或可选 path:line 越界       error
    F4 stale_exclusion   排除 glob 在范围内命中零文件                                          warning
    G1 flag_gap          AST 枚举的字段（Class.field）无叶子认领且未排除                       warning
    G2 unknown_flag      认领/排除了枚举面里没有的字段                                        error
    G3 ambiguous_flag    裸字段名在多个类中出现，须写 Class.field                              error
    E1 entry_gap         surfaces.entries 条目无叶子认领且未排除                              warning
    E2 unknown_entry     叶子认领了 surfaces.entries 里没有的条目                             error
    T1 parent_chain      节点/叶子的父链必须逐级解析到已声明节点，parent 必须等于 id 前缀      error
    T2 empty_node        非叶节点必须至少有一个子节点或叶子                                    error
    S1 status_spec       spec'd/verified 却无 spec，或 spec 页不存在                            error
    S2 overview_mismatch 总览页叶子行表与 manifest 的 ID/状态不一致                           error
    S3 spec_anchor       spec 页里没有包含叶子 id 的标题                                      error
    V1 review_missing    verified 叶子没有 PASS 评审记录；评审指向未知叶子                     error
    V2 not_verified      delivery 阶段仍有非 verified 叶子                                     error
    D1 duplicate_id      节点/叶子 id 重复                                                    error

阶段（--phase）：proposal（默认；建树/对账）、spec（规格波次；spec_dir 必填，spec'd/verified 须有真实
页面与标题锚）、delivery（终门禁；全部叶子 verified + PASS 评审；隐含 --strict）。
对账为零只是机械下限：文件级认领看不见同一文件里的第二个行为，未枚举的面对本工具不可见。

用法：
    python tools/check_feature_tree.py <manifest.yaml> [--phase proposal|spec|delivery] [--strict] [--examples N]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import io
import re
import sys
from collections import defaultdict
from pathlib import Path

from check_coverage import enumerate_fields
from check_locators import GitView, load_watchlist

SEVERITY = {
    "X1": "error",
    "V0": "error",
    "F1": "warning", "F2": "error", "F3": "error", "F4": "warning",
    "G1": "warning", "G2": "error", "G3": "error",
    "E1": "warning", "E2": "error",
    "T1": "error", "T2": "error",
    "S1": "error", "S2": "error", "S3": "error",
    "V1": "error", "V2": "error",
    "D1": "error",
}
LABELS = {
    "X1": "schema", "V0": "scope",
    "F1": "unowned_file", "F2": "phantom_file", "F3": "bad_entry_anchor", "F4": "stale_exclusion",
    "G1": "flag_gap", "G2": "unknown_flag", "G3": "ambiguous_flag",
    "E1": "entry_gap", "E2": "unknown_entry",
    "T1": "parent_chain", "T2": "empty_node",
    "S1": "status_spec", "S2": "overview_mismatch", "S3": "spec_anchor",
    "V1": "review_missing", "V2": "not_verified",
    "D1": "duplicate_id",
}
PHASES = ("proposal", "spec", "delivery")
STATUSES = ("planned", "spec'd", "verified")
VERDICTS = ("PASS", "REJECT")

_SEG = r"[a-z0-9]+(?:[-_][a-z0-9]+)*"
NODE_ID_RE = re.compile(rf"^{_SEG}(?:/{_SEG})*$")
LEAF_ID_RE = re.compile(rf"^{_SEG}(?:/{_SEG})+$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SYMBOL_ENTRY_RE = re.compile(r"([^:]+)::([^:]+)")
LINE_ENTRY_RE = re.compile(r"(.+?):(\d+)(?:-(\d+))?")
ROW_ID_RE = re.compile(r"^`([^`]+)`$")
HEADING_RE = re.compile(r"^#{1,6}\s+(.*)$")

TOP_KEYS = {"domain", "repo", "checkout", "commit", "overview", "spec_dir",
            "surfaces", "nodes", "leaves", "exclusions", "reviews"}
SURFACE_KEYS = {"files", "flags", "entries"}
FILES_KEYS = {"include"}
FLAG_SRC_KEYS = {"file", "class"}
NODE_KEYS = {"id", "name", "responsibility", "parent"}
LEAF_KEYS = {"id", "name", "definition", "entry", "spec", "status", "owns"}
OWNS_KEYS = {"files", "flags", "entries"}
EXCL_KEYS = {"files": "glob", "flags": "name", "entries": "name"}
REVIEW_KEYS = {"leaf", "date", "reviewer", "r1", "r2", "r3", "r4", "r5", "r6", "verdict"}

_GLOB_CACHE: dict = {}


# ---------------------------------------------------------------------------- helpers

def _glob_re(pattern: str):
    """`**` 跨目录（`**/` 可匹配零级），`*`/`?` 不跨 `/`，其余字面匹配。"""
    if pattern not in _GLOB_CACHE:
        out, i = [], 0
        while i < len(pattern):
            if pattern.startswith("**/", i):
                out.append("(?:.*/)?")
                i += 3
            elif pattern.startswith("**", i):
                out.append(".*")
                i += 2
            elif pattern[i] == "*":
                out.append("[^/]*")
                i += 1
            elif pattern[i] == "?":
                out.append("[^/]")
                i += 1
            else:
                out.append(re.escape(pattern[i]))
                i += 1
        _GLOB_CACHE[pattern] = re.compile("".join(out))
    return _GLOB_CACHE[pattern]


def glob_match(pattern: str, path: str) -> bool:
    return _glob_re(pattern).fullmatch(path) is not None


def _is_str(v) -> bool:
    return isinstance(v, str) and v.strip() != ""


def _is_text_or_date(v) -> bool:
    return _is_str(v) or isinstance(v, (_dt.date, _dt.datetime))


def load_manifest(path: Path):
    import yaml

    with io.open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {} if data is None else data


def resolve_checkout(cfg: dict, base: Path) -> Path:
    co = cfg.get("checkout")
    if co:
        p = Path(co)
        return (p if p.is_absolute() else base / p).resolve()
    name = cfg.get("repo")
    for e in load_watchlist():
        if e["name"] == name and e["checkout"]:
            return e["checkout"]
    raise SystemExit(f"watchlist 中无 {name} 的 checkout")


def _rel(base: Path, p):
    if not p:
        return None
    q = Path(p)
    return q if q.is_absolute() else base / q


def parse_leaf_rows(text: str) -> dict:
    """总览页叶子行表：首列反引号 ID、末列状态，≥6 列的表格行。"""
    rows = {}
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) < 6:
            continue
        m = ROW_ID_RE.fullmatch(cells[0])
        if m:
            rows[m.group(1)] = cells[-1]
    return rows


def _parent_of(node_id: str):
    return node_id.rsplit("/", 1)[0] if "/" in node_id else None


# ---------------------------------------------------------------------------- X1 schema

def _str_list(v, where: str, errs: list, allow_empty: bool = False) -> bool:
    ok = isinstance(v, list) and (allow_empty or bool(v)) and all(_is_str(x) for x in v)
    if not ok:
        errs.append(f"{where}: must be a {'' if allow_empty else 'non-empty '}list of non-empty strings")
    return ok


def _unknown_keys(obj: dict, allowed, where: str, errs: list):
    for k in obj:
        if k not in allowed:
            errs.append(f"{where}: unknown key '{k}'")


def validate_schema(cfg, phase: str) -> list:
    errs: list = []
    if not isinstance(cfg, dict):
        return ["manifest: must be a mapping"]
    _unknown_keys(cfg, TOP_KEYS, "manifest", errs)
    if not _is_str(cfg.get("domain")):
        errs.append("domain: required non-empty string")
    if not (_is_str(cfg.get("checkout")) or _is_str(cfg.get("repo"))):
        errs.append("checkout or repo: one of them is required")
    commit = cfg.get("commit")
    if not (isinstance(commit, str) and COMMIT_RE.fullmatch(commit)):
        errs.append("commit: must be the frozen 40-hex commit (not HEAD, a branch, or a short hash)")
    for k in ("overview", "spec_dir"):
        if k in cfg and not _is_str(cfg[k]):
            errs.append(f"{k}: must be a non-empty string")
    if phase in ("spec", "delivery") and not _is_str(cfg.get("spec_dir")):
        errs.append(f"spec_dir: required in phase {phase}")

    surfaces = cfg.get("surfaces")
    if not isinstance(surfaces, dict):
        errs.append("surfaces: required mapping with files.include")
    else:
        _unknown_keys(surfaces, SURFACE_KEYS, "surfaces", errs)
        files = surfaces.get("files")
        if not isinstance(files, dict):
            errs.append("surfaces.files.include: required non-empty list of globs")
        else:
            _unknown_keys(files, FILES_KEYS, "surfaces.files", errs)
            if "include" not in files:
                errs.append("surfaces.files.include: required non-empty list of globs")
            else:
                _str_list(files["include"], "surfaces.files.include", errs)
        flags = surfaces.get("flags") or []
        if not isinstance(flags, list):
            errs.append("surfaces.flags: must be a list of {file, class}")
        else:
            for i, src in enumerate(flags):
                if (not isinstance(src, dict) or set(src) - FLAG_SRC_KEYS
                        or not all(_is_str(src.get(k)) for k in FLAG_SRC_KEYS)):
                    errs.append(f"surfaces.flags[{i}]: must be {{file, class}}")
        entries = surfaces.get("entries")
        if entries is not None and _str_list(entries, "surfaces.entries", errs, allow_empty=True):
            for d in sorted({e for e in entries if entries.count(e) > 1}):
                errs.append(f"surfaces.entries: duplicate '{d}'")

    nodes = cfg.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        errs.append("nodes: required non-empty list (the non-leaf tree: id, name, responsibility, parent)")
    else:
        for i, n in enumerate(nodes):
            where = f"nodes[{i}]"
            if not isinstance(n, dict):
                errs.append(f"{where}: must be a mapping")
                continue
            _unknown_keys(n, NODE_KEYS, where, errs)
            nid = n.get("id")
            if not (isinstance(nid, str) and NODE_ID_RE.fullmatch(nid)):
                errs.append(f"{where}: id {nid!r} must be a lowercase slug path")
            for k in ("name", "responsibility"):
                if not _is_str(n.get(k)):
                    errs.append(f"{where} ({nid}): {k} required non-empty")
            if n.get("parent") is not None and not _is_str(n.get("parent")):
                errs.append(f"{where} ({nid}): parent must be null or a node id")

    leaves = cfg.get("leaves")
    if not isinstance(leaves, list) or not leaves:
        errs.append("leaves: required non-empty list")
    else:
        for i, leaf in enumerate(leaves):
            where = f"leaves[{i}]"
            if not isinstance(leaf, dict):
                errs.append(f"{where}: must be a mapping")
                continue
            _unknown_keys(leaf, LEAF_KEYS, where, errs)
            lid = leaf.get("id")
            if not (isinstance(lid, str) and LEAF_ID_RE.fullmatch(lid)):
                errs.append(f"{where}: id {lid!r} must be a lowercase slug path with at least two segments")
            if not _is_str(leaf.get("name")):
                errs.append(f"{where} ({lid}): name required non-empty")
            if not _is_str(leaf.get("entry")):
                errs.append(f"{where} ({lid}): entry required (path::qualified.symbol; path:line is legacy-compatible)")
            status = leaf.get("status", "planned")
            if status not in STATUSES:
                errs.append(f"{where} ({lid}): status {status!r} not in {STATUSES}")
            for k in ("spec", "definition"):
                if k in leaf and not _is_str(leaf[k]):
                    errs.append(f"{where} ({lid}): {k} must be a non-empty string")
            owns = leaf.get("owns", {})
            if not isinstance(owns, dict):
                errs.append(f"{where} ({lid}): owns must be a mapping")
            else:
                _unknown_keys(owns, OWNS_KEYS, f"{where} ({lid}).owns", errs)
                for k in OWNS_KEYS:
                    if k in owns:
                        _str_list(owns[k], f"{where} ({lid}).owns.{k}", errs, allow_empty=True)

    excl = cfg.get("exclusions", {}) or {}
    if not isinstance(excl, dict):
        errs.append("exclusions: must be a mapping")
    else:
        _unknown_keys(excl, EXCL_KEYS, "exclusions", errs)
        for k, field in EXCL_KEYS.items():
            rows = excl.get(k) or []
            if not isinstance(rows, list):
                errs.append(f"exclusions.{k}: must be a list")
                continue
            for i, row in enumerate(rows):
                if not isinstance(row, dict) or set(row) - {field, "reason"} or not _is_str(row.get(field)):
                    errs.append(f"exclusions.{k}[{i}]: must be {{{field}, reason}}")
                    continue
                if not _is_str(row.get("reason")):
                    errs.append(f"exclusions.{k}[{i}] ({row.get(field)}): reason required")

    reviews = cfg.get("reviews", []) or []
    if not isinstance(reviews, list):
        errs.append("reviews: must be a list")
    else:
        for i, rv in enumerate(reviews):
            if not isinstance(rv, dict):
                errs.append(f"reviews[{i}]: must be a mapping")
                continue
            _unknown_keys(rv, REVIEW_KEYS, f"reviews[{i}]", errs)
            missing = sorted(k for k in REVIEW_KEYS if not _is_text_or_date(rv.get(k)))
            if missing:
                errs.append(f"reviews[{i}]: missing or empty {missing}")
            if rv.get("verdict") not in VERDICTS:
                errs.append(f"reviews[{i}]: verdict must be PASS or REJECT")
    return errs


# ---------------------------------------------------------------------------- audit

def audit(manifest_path, phase: str = "proposal") -> dict:
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}")
    manifest_path = Path(manifest_path)
    base = manifest_path.resolve().parent
    cfg = load_manifest(manifest_path)
    report = {k: [] for k in SEVERITY}
    report["X1"] = validate_schema(cfg, phase)
    if report["X1"]:
        return report

    checkout = resolve_checkout(cfg, base)
    commit = cfg["commit"]
    gv = GitView()
    if not checkout.exists() or not gv.commit_exists(checkout, commit):
        raise SystemExit(f"commit {commit[:12]} 在 {checkout} 中不可用（checkout 不存在或未含该 commit）")

    surfaces = cfg["surfaces"]
    nodes = cfg["nodes"]
    leaves = cfg["leaves"]
    exclusions = cfg.get("exclusions") or {}
    node_ids = [n["id"] for n in nodes]
    leaf_ids = [leaf["id"] for leaf in leaves]

    def owns(leaf, key):
        return list(((leaf.get("owns") or {}).get(key)) or [])

    # D1 —— id 唯一（节点与叶子共用一个命名空间）
    seen = set()
    for i in node_ids + leaf_ids:
        if i in seen and i not in report["D1"]:
            report["D1"].append(i)
        seen.add(i)

    # T1/T2 —— 树：父链逐级存在、parent = id 前缀、非叶节点非空
    node_set = set(node_ids)
    for n in nodes:
        nid, expected, declared = n["id"], _parent_of(n["id"]), n.get("parent")
        if declared != expected:
            report["T1"].append(f"{nid}: parent {declared!r} != {expected!r}")
        if expected is not None and expected not in node_set:
            report["T1"].append(f"{nid}: missing parent node {expected}")
    for lid in leaf_ids:
        parent = _parent_of(lid)
        if parent not in node_set:
            report["T1"].append(f"{lid}: missing parent node {parent}")
    children = defaultdict(int)
    for i in node_ids + leaf_ids:
        parent = _parent_of(i)
        if parent:
            children[parent] += 1
    for nid in node_ids:
        if children[nid] == 0:
            report["T2"].append(nid)

    # V0/F1/F2/F4 —— 文件面
    tree_paths, _ = gv.tree(checkout, commit)
    in_scope_set = set()
    for g in surfaces["files"]["include"]:
        hits = {p for p in tree_paths if glob_match(g, p)}
        if not hits:
            report["V0"].append(f"include '{g}' matches no file at the frozen commit")
        in_scope_set |= hits
    if not in_scope_set:
        report["V0"].append("scope is empty: no file matched surfaces.files.include")
    in_scope = sorted(in_scope_set)
    excluded = set()
    for row in exclusions.get("files") or []:
        hits = {p for p in in_scope if glob_match(row["glob"], p)}
        if not hits:
            report["F4"].append(row["glob"])
        excluded |= hits
    claimed = set()
    for leaf in leaves:
        for pat in owns(leaf, "files"):
            hits = [p for p in tree_paths if glob_match(pat, p)]
            if not hits:
                report["F2"].append(f"{leaf['id']} -> {pat}")
            claimed.update(hits)
    for p in in_scope:
        if p not in claimed and p not in excluded:
            report["F1"].append(p)

    # F3 —— 默认验证稳定 path::symbol 的文件；遗留 path:line 继续兼容并校验范围
    for leaf in leaves:
        entry = leaf["entry"]
        symbol = SYMBOL_ENTRY_RE.fullmatch(entry)
        line_ref = LINE_ENTRY_RE.fullmatch(entry)
        if symbol:
            if symbol.group(1) not in tree_paths:
                report["F3"].append(f"{leaf['id']} -> {entry}")
            continue
        if not line_ref:
            report["F3"].append(f"{leaf['id']} -> {entry}")
            continue
        n = gv.nlines(checkout, commit, line_ref.group(1))
        line = int(line_ref.group(2))
        if n is None or line < 1 or line > n:
            report["F3"].append(f"{leaf['id']} -> {entry}")

    # G1/G2/G3 —— flag 面，身份 = Class.field
    enumerated = []  # qualified names in enumeration order
    by_bare = defaultdict(list)
    for src in surfaces.get("flags") or []:
        for field in enumerate_fields(checkout, commit, src["file"], src["class"]):
            q = f"{src['class']}.{field}"
            if q not in enumerated:
                enumerated.append(q)
                by_bare[field].append(q)
    qualified = set(enumerated)

    def resolve_flag(name):
        """返回 (qualified | None, error | None)。"""
        if name in qualified:
            return name, None
        if "." in name:
            return None, "unknown"
        cands = by_bare.get(name, [])
        if len(cands) == 1:
            return cands[0], None
        if len(cands) > 1:
            return None, f"ambiguous ({', '.join(cands)})"
        return None, "unknown"

    flag_claims = set()
    for leaf in leaves:
        for name in owns(leaf, "flags"):
            q, err = resolve_flag(name)
            if q:
                flag_claims.add(q)
            elif err == "unknown":
                report["G2"].append(f"{leaf['id']} -> {name}")
            else:
                report["G3"].append(f"{leaf['id']} -> {name} {err[len('ambiguous '):]}")
    excluded_flags = set()
    for row in exclusions.get("flags") or []:
        q, err = resolve_flag(row["name"])
        if q:
            excluded_flags.add(q)
        elif err == "unknown":
            report["G2"].append(f"exclusion -> {row['name']}")
        else:
            report["G3"].append(f"exclusion -> {row['name']} {err[len('ambiguous '):]}")
    for q in enumerated:
        if q not in flag_claims and q not in excluded_flags:
            report["G1"].append(q)

    # E1/E2 —— 入口面：分析者枚举的符号级清单
    entries = list(surfaces.get("entries") or [])
    entry_set = set(entries)
    entry_claims = set()
    for leaf in leaves:
        for name in owns(leaf, "entries"):
            if name not in entry_set:
                report["E2"].append(f"{leaf['id']} -> {name}")
            entry_claims.add(name)
    excluded_entries = {row["name"] for row in exclusions.get("entries") or []}
    for name in entries:
        if name not in entry_claims and name not in excluded_entries:
            report["E1"].append(name)

    # S1/S3 —— 状态与规格页
    spec_dir = _rel(base, cfg.get("spec_dir"))
    for leaf in leaves:
        lid, status, spec = leaf["id"], leaf.get("status", "planned"), leaf.get("spec")
        if status != "planned" and not spec:
            report["S1"].append(f"{lid}: status {status} without spec")
            continue
        if spec and spec_dir is not None:
            page = str(spec).split("#", 1)[0]
            page_path = spec_dir / (page if page.endswith(".md") else page + ".md")
            if not page_path.exists():
                shown = Path(str(cfg["spec_dir"])) / page_path.name
                report["S1"].append(f"{lid} -> {shown.as_posix()} missing")
                continue
            text = io.open(page_path, encoding="utf-8", errors="replace").read()
            headings = [m.group(1) for m in (HEADING_RE.match(l) for l in text.splitlines()) if m]
            if not any(lid in h for h in headings):
                report["S3"].append(f"{lid}: no heading containing the leaf id in {page_path.name}")

    # V1/V2 —— 评审记录与交付门
    leaf_set = set(leaf_ids)
    pass_by_leaf = defaultdict(int)
    for rv in cfg.get("reviews") or []:
        if rv["leaf"] in leaf_set and rv["verdict"] == "PASS":
            pass_by_leaf[rv["leaf"]] += 1
    for leaf in leaves:
        if leaf.get("status", "planned") == "verified" and not pass_by_leaf[leaf["id"]]:
            report["V1"].append(f"{leaf['id']}: no PASS review")
    for rv in cfg.get("reviews") or []:
        if rv["leaf"] not in leaf_set:
            report["V1"].append(f"review for unknown leaf {rv['leaf']}")
    if phase == "delivery":
        for leaf in leaves:
            status = leaf.get("status", "planned")
            if status != "verified":
                report["V2"].append(f"{leaf['id']}: status {status}")

    # S2 —— 总览页叶子行表 vs manifest
    overview = _rel(base, cfg.get("overview"))
    if overview is not None:
        if not overview.exists():
            report["S2"].append(f"overview {cfg.get('overview')} missing")
        else:
            rows = parse_leaf_rows(io.open(overview, encoding="utf-8", errors="replace").read())
            manifest_status = {leaf["id"]: leaf.get("status", "planned") for leaf in leaves}
            for lid, status in rows.items():
                if lid not in manifest_status:
                    report["S2"].append(f"{lid}: in overview, not in manifest")
                elif status != manifest_status[lid]:
                    report["S2"].append(
                        f"{lid}: overview status {status} != manifest {manifest_status[lid]}"
                    )
            for lid in manifest_status:
                if lid not in rows:
                    report["S2"].append(f"{lid}: in manifest, not in overview")
    return report


def counts(report: dict):
    errors = sum(len(v) for k, v in report.items() if SEVERITY[k] == "error")
    warnings = sum(len(v) for k, v in report.items() if SEVERITY[k] == "warning")
    return errors, warnings


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("manifest", help="feature-tree manifest yaml")
    ap.add_argument("--phase", choices=PHASES, default="proposal",
                    help="proposal (default) | spec | delivery (implies --strict)")
    ap.add_argument("--strict", action="store_true", help="warnings also fail")
    ap.add_argument("--examples", type=int, default=15)
    args = ap.parse_args(argv)
    strict = args.strict or args.phase == "delivery"
    path = Path(args.manifest)
    report = audit(path, phase=args.phase)
    if not report["X1"]:
        cfg = load_manifest(path)
        leaves = cfg["leaves"]
        by_status = defaultdict(int)
        for leaf in leaves:
            by_status[leaf.get("status", "planned")] += 1
        status_txt = " ".join(f"{k}={v}" for k, v in sorted(by_status.items()))
        print(f"phase={args.phase} nodes={len(cfg['nodes'])} leaves={len(leaves)} {status_txt} @ {cfg['commit'][:9]}")
    else:
        print(f"phase={args.phase} manifest rejected by schema")
    for code in SEVERITY:
        rows = report[code]
        print(f"{code} {LABELS[code]} ({SEVERITY[code]}): {len(rows)}")
        for r in rows[: args.examples]:
            print(f"    {r}")
        if len(rows) > args.examples:
            print(f"    … 另 {len(rows) - args.examples} 条")
    errors, warnings = counts(report)
    print(f"\nerrors={errors} warnings={warnings}")
    return 1 if errors or (strict and warnings) else 0


if __name__ == "__main__":
    sys.exit(main())
