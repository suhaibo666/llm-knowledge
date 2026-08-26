#!/usr/bin/env python3
"""上游雷达：追踪代码仓库演进、模型发布与前沿论文，产出一份周报。

刻意的边界：**只报告，不写 wiki 分析页**。本库的价值在于每条断言都有可核验
定位符；无人值守产出的机制级结论没人复核，会污染这个前提。雷达负责把
「有什么变了、哪些已让现有页面的基线过期」摆出来，要不要落成分析页、怎么落，
仍然走 skills/source-faithful-analysis 的人工流程。

用法：
    python tools/radar.py                 # 看最近 7 天，写 docs/radar/<日期>.md
    python tools/radar.py --since 14      # 改时间窗
    python tools/radar.py --dry-run       # 只打印，不落盘、不更新 state
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RADAR_DIR = REPO_ROOT / "docs" / "radar"
WATCHLIST = RADAR_DIR / "watchlist.yaml"
STATE_FILE = RADAR_DIR / "state.json"

USER_AGENT = "llm-knowledge-radar/1 (+https://github.com/suhaibo666/llm-knowledge)"
HTTP_TIMEOUT = 30

HOST_URL = {
    "github": "https://github.com/{repo}.git",
    "gitee": "https://gitee.com/{repo}.git",
}


# --------------------------------------------------------------------------- io
def load_watchlist(path: Path = WATCHLIST) -> dict:
    try:
        import yaml
    except ImportError:  # pragma: no cover - environment guard
        raise SystemExit("radar 需要 PyYAML：pip install pyyaml")
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_state(path: Path = STATE_FILE) -> dict:
    if not path.exists():
        return {"repos": {}, "vendors": {}, "arxiv": {}}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save_state(state: dict, path: Path = STATE_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=1, sort_keys=True)
        handle.write("\n")


def fetch_json(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
        return response.read().decode("utf-8")


# ---------------------------------------------------------------------- repos
def ls_remote(url: str, pattern: str) -> dict:
    """`git ls-remote` —— 不消耗 GitHub API 配额，gitee 也通用。"""
    result = subprocess.run(
        ["git", "ls-remote", url, pattern],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "ls-remote failed")
    refs = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            sha, ref = parts
            refs[ref] = sha
    return refs


def compare_commits(repo: str, base: str, head: str):
    """GitHub compare：拿到落后多少个提交，以及最近几条标题。失败返回 None。"""
    url = "https://api.github.com/repos/%s/compare/%s...%s" % (repo, base, head)
    try:
        data = fetch_json(url)
    except Exception:
        return None
    commits = [
        c.get("commit", {}).get("message", "").splitlines()[0]
        for c in data.get("commits", [])
    ]
    return {"ahead_by": data.get("ahead_by"), "titles": commits[-8:]}


RELEASE_TAG = re.compile(r"^v?\d+\.\d+")
MAX_TRACKED_TAGS = 80


def release_tags(refs: dict) -> list:
    """只保留发布形态的 tag，并限量。

    pytorch 这类仓库有 6600+ 个 tag，其中 2751 个是 `viable/strict/*`、
    1000+ 个是 `ciflow/*` —— 全存进 state.json 会让它涨到 286KB 且每周提交，
    而这些 CI 产物对「上游发了什么版本」毫无信息量。
    """

    names = {ref.split("/", 2)[2].removesuffix("^{}") for ref in refs}
    releases = [n for n in names if "/" not in n and RELEASE_TAG.match(n)]

    def natural(tag):
        return [int(p) if p.isdigit() else p
                for p in re.split(r"(\d+)", tag.lstrip("v"))]

    return sorted(releases, key=natural)[-MAX_TRACKED_TAGS:]


def check_repo(entry: dict, state: dict, errors: list) -> dict | None:
    host = entry.get("host", "github")
    url = HOST_URL[host].format(repo=entry["repo"])
    branch = entry.get("branch", "main")
    key = "%s:%s" % (host, entry["repo"])

    try:
        heads = ls_remote(url, "refs/heads/" + branch)
        tags = ls_remote(url, "refs/tags/*")
    except Exception as exc:
        errors.append("仓库 %s：%s" % (entry["name"], exc))
        return None

    head_sha = heads.get("refs/heads/" + branch)
    if not head_sha:
        errors.append("仓库 %s：分支 %s 不存在" % (entry["name"], branch))
        return None

    tag_names = release_tags(tags)
    previous = state["repos"].get(key, {})
    new_tags = [t for t in tag_names if t not in set(previous.get("tags", []))]
    # 首次运行时全部 tag 都是「新」的，那不是信号，跳过
    first_run = "tags" not in previous

    finding = {
        "name": entry["name"],
        "category": entry.get("category", "未分类"),
        "repo": entry["repo"],
        "host": host,
        "branch": branch,
        "head": head_sha,
        "moved": previous.get("head") not in (None, head_sha),
        "new_tags": [] if first_run else new_tags[-10:],
        "kb_baseline": entry.get("kb_baseline"),
        "kb_entry": entry.get("kb_entry"),
        "drift": None,
    }

    baseline = entry.get("kb_baseline")
    if baseline and host == "github" and not head_sha.startswith(baseline):
        finding["drift"] = compare_commits(entry["repo"], baseline, head_sha)

    state["repos"][key] = {"head": head_sha, "tags": tag_names, "checked": utc_now_iso()}
    return finding


# -------------------------------------------------------------------- vendors
def check_vendor(entry: dict, since: datetime, state: dict, errors: list) -> dict | None:
    org = entry["org"]
    url = (
        "https://huggingface.co/api/models?author=%s&sort=createdAt&direction=-1&limit=30"
        % urllib.parse.quote(org)
    )
    try:
        models = fetch_json(url)
    except Exception as exc:
        errors.append("厂商 %s：%s" % (entry["name"], exc))
        return None

    seen = set(state["vendors"].get(org, {}).get("seen", []))
    fresh = []
    for model in models:
        created = model.get("createdAt", "")
        model_id = model.get("id", "")
        if not created or not model_id:
            continue
        if parse_iso(created) < since:
            continue
        if model_id in seen:
            continue
        fresh.append({"id": model_id, "created": created[:10],
                      "downloads": model.get("downloads", 0)})

    state["vendors"][org] = {
        "seen": sorted(seen | {m.get("id", "") for m in models if m.get("id")}),
        "checked": utc_now_iso(),
    }
    if not fresh:
        return None
    return {"name": entry["name"], "org": org,
            "kb_entry": entry.get("kb_entry"), "models": fresh}


# --------------------------------------------------------------------- arxiv
ATOM = {"a": "http://www.w3.org/2005/Atom"}


def check_arxiv(entry: dict, since: datetime, state: dict, errors: list) -> dict | None:
    url = (
        "https://export.arxiv.org/api/query?search_query=%s"
        "&sortBy=submittedDate&sortOrder=descending&max_results=25"
        % urllib.parse.quote(entry["query"])
    )
    try:
        root = ET.fromstring(fetch_text(url))
    except Exception as exc:
        errors.append("论文主题 %s：%s" % (entry["name"], exc))
        return None

    seen = set(state["arxiv"].get(entry["name"], {}).get("seen", []))
    papers, all_ids = [], []
    for node in root.findall("a:entry", ATOM):
        raw_id = node.find("a:id", ATOM).text or ""
        match = re.search(r"abs/(\d{4}\.\d{4,5})", raw_id)
        if not match:
            continue
        paper_id = match.group(1)
        all_ids.append(paper_id)
        published = node.find("a:published", ATOM).text or ""
        if not published or parse_iso(published) < since:
            continue
        if paper_id in seen:
            continue
        papers.append({
            "id": paper_id,
            "date": published[:10],
            "title": " ".join((node.find("a:title", ATOM).text or "").split()),
        })

    state["arxiv"][entry["name"]] = {
        "seen": sorted(seen | set(all_ids)), "checked": utc_now_iso(),
    }
    if not papers:
        return None
    # 每个主题最多列 8 条，并如实说明截断了多少——静默截断会让读者
    # 以为「本期就这些」。
    cap = 8
    truncated = max(0, len(papers) - cap)
    return {"name": entry["name"], "papers": papers[:cap], "truncated": truncated}


# --------------------------------------------------------------------- utils
def parse_iso(value: str) -> datetime:
    text = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# -------------------------------------------------------------------- report
def render(findings, vendors, papers, errors, since_days, today) -> str:
    lines = [
        "# 上游雷达 · %s" % today,
        "",
        "> 由 `tools/radar.py` 自动生成，时间窗 **最近 %d 天**。" % since_days,
        "> 本文件**只报告事实**（谁动了、动了多少、链接在哪），不含机制结论——"
        "要落成分析页请走 [`source-faithful-analysis`](../../skills/source-faithful-analysis/SKILL.md)，"
        "并在合并后同步更新 `docs/radar/watchlist.yaml` 里的 `kb_baseline`。",
        "",
    ]

    stale = [f for f in findings if f.get("drift") and f["drift"].get("ahead_by")]
    lines += ["## 一、KB 基线已过期的仓库", ""]
    if stale:
        lines += ["| 仓库 | KB 基线 | 上游已领先 | KB 入口 |", "|---|---|---:|---|"]
        for f in sorted(stale, key=lambda x: -(x["drift"]["ahead_by"] or 0)):
            lines.append("| %s | `%s` | %d 个提交 | [%s](../../%s) |" % (
                f["name"], (f["kb_baseline"] or "")[:12],
                f["drift"]["ahead_by"], Path(f["kb_entry"]).name, f["kb_entry"]))
        lines.append("")
        for f in sorted(stale, key=lambda x: -(x["drift"]["ahead_by"] or 0)):
            lines += ["<details><summary>%s 最近提交</summary>" % f["name"], ""]
            lines += ["- %s" % t for t in f["drift"]["titles"]]
            lines += ["", "</details>", ""]
    else:
        lines += ["本期没有仓库的 KB 基线落后（或未配置 `kb_baseline`）。", ""]

    lines += ["## 二、仓库活动", ""]
    by_category: dict[str, list] = {}
    for f in findings:
        by_category.setdefault(f["category"], []).append(f)
    for category in sorted(by_category):
        lines += ["### %s" % category, "",
                  "| 仓库 | 分支 HEAD | 自上次是否变化 | 新 tag |", "|---|---|---|---|"]
        for f in by_category[category]:
            lines.append("| [%s](https://%s.com/%s) | `%s` | %s | %s |" % (
                f["name"], f["host"], f["repo"], f["head"][:9],
                "是" if f["moved"] else "否",
                ", ".join("`%s`" % t for t in f["new_tags"]) or "—"))
        lines.append("")

    lines += ["## 三、模型厂商新发布", ""]
    if vendors:
        for v in vendors:
            lines += ["### %s（`%s`）" % (v["name"], v["org"]), ""]
            for m in v["models"]:
                lines.append("- `%s` — %s · [HF](https://huggingface.co/%s)"
                             % (m["created"], m["id"], m["id"]))
            lines += ["", "KB 入口：[%s](../../%s)" % (Path(v["kb_entry"]).name, v["kb_entry"]), ""]
    else:
        lines += ["本期无新模型。", ""]

    lines += ["## 四、前沿论文", ""]
    if papers:
        for group in papers:
            lines += ["### %s" % group["name"], ""]
            for p in group["papers"]:
                lines.append("- `%s` [%s](https://arxiv.org/abs/%s) — %s"
                             % (p["date"], p["id"], p["id"], p["title"]))
            if group.get("truncated"):
                lines.append("- *（另有 %d 篇命中未列出，调窄 watchlist 里的 query "
                             "或用 --since 缩短时间窗）*" % group["truncated"])
            lines.append("")
    else:
        lines += ["本期无新论文命中。", ""]

    lines += ["## 五、本次采集失败项", ""]
    if errors:
        lines += ["以下来源本次没取到，**不代表它们没有变化**：", ""]
        lines += ["- %s" % e for e in errors]
        lines.append("")
    else:
        lines += ["无。所有来源均成功采集。", ""]

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------- main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", type=int, default=7, help="时间窗天数（默认 7）")
    parser.add_argument("--dry-run", action="store_true", help="只打印，不落盘")
    parser.add_argument("--watchlist", type=Path, default=WATCHLIST)
    args = parser.parse_args(argv)

    watchlist = load_watchlist(args.watchlist)
    state = load_state()
    since = datetime.now(timezone.utc) - timedelta(days=args.since)
    errors: list[str] = []

    findings = [f for f in (check_repo(r, state, errors) for r in watchlist.get("repos", [])) if f]
    vendors = [v for v in (check_vendor(v, since, state, errors) for v in watchlist.get("vendors", [])) if v]
    papers = [p for p in (check_arxiv(a, since, state, errors) for a in watchlist.get("arxiv", [])) if p]

    today = datetime.now().strftime("%Y-%m-%d")
    report = render(findings, vendors, papers, errors, args.since, today)

    if args.dry_run:
        sys.stdout.write(report)
        return 0

    RADAR_DIR.mkdir(parents=True, exist_ok=True)
    out = RADAR_DIR / ("%s.md" % today)
    out.write_text(report, encoding="utf-8", newline="\n")
    save_state(state)
    print("雷达报告已写入 %s" % out.relative_to(REPO_ROOT))
    print("  仓库 %d · 新模型厂商 %d · 论文主题命中 %d · 采集失败 %d"
          % (len(findings), len(vendors), len(papers), len(errors)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
