# -*- coding: utf-8 -*-
"""check_locators 的单测：基线解析、引用抽取、对冻结 commit 的验证。

用 tmp_path 里的真实小 git 仓做 fixture——验证逻辑走的就是 git ls-tree / git show，
mock 会把最容易出错的边界（basename 解析、行数统计）藏起来。
"""
import subprocess
from pathlib import Path

import pytest

import check_locators as cl


# ---------- 纯函数：解析 ----------

ENTRIES = [
    {"name": "Megatron-LM", "repo": "NVIDIA/Megatron-LM", "checkout": None},
    {"name": "verl", "repo": "volcengine/verl", "checkout": None},
    {"name": "torch_npu", "repo": "Ascend/pytorch", "checkout": None},
]


def test_parse_slug_baseline():
    found, unresolved = cl.parse_baselines(
        "> **源码基线**：`NVIDIA/Megatron-LM@71092579522a12522d9f323ae180c9825d01928a`（`dev`，2026-08-27）",
        ENTRIES,
    )
    assert [(e["repo"], c[:8]) for e, c in found] == [("NVIDIA/Megatron-LM", "71092579")]
    assert unresolved == []


def test_parse_name_branch_baseline():
    # `verl main @ 254a23ed` —— 分支词前面那个词才是仓名
    found, _ = cl.parse_baselines("> 代码基准：verl main @ 254a23edc62f25eb", ENTRIES)
    assert [(e["name"], c[:8]) for e, c in found] == [("verl", "254a23ed")]


def test_parse_name_only_baseline():
    found, _ = cl.parse_baselines(
        "> **Source baseline**: torch_npu v2.7.1@b3c8a815b4bf6f8ec28b418aa9ec4281", ENTRIES
    )
    # v2.7.1 不是分支词，直接词匹配失败后应落到 torch_npu……v1 允许解析失败进 unresolved，
    # 但 slug 缺失的 torch_npu 页至少不能被错误解析成别的仓。
    assert all(e["name"] != "Megatron-LM" for e, _ in found)


def test_parse_dedup_and_multi_repo():
    txt = (
        "`NVIDIA/Megatron-LM@71092579522a1252` 与 `volcengine/verl@254a23edc62f25eb`；"
        "又提一次 `NVIDIA/Megatron-LM@71092579522a1252`"
    )
    found, _ = cl.parse_baselines(txt, ENTRIES)
    assert len(found) == 2


def test_citation_extraction():
    text = (
        "见 `megatron/core/pipeline_parallel/schedules.py:938-965` 与 `combined_1f1b.py:35`；"
        "链接 https://x.com/a/b.py:1 不算；`config.yaml:3` 算。"
    )
    cites = [(m.group(1), m.group(2)) for m in cl.CITE_RE.finditer(text)
             if "://" not in text[max(0, m.start() - 8):m.start()]]
    assert ("megatron/core/pipeline_parallel/schedules.py", "938") in cites
    assert ("combined_1f1b.py", "35") in cites
    assert ("config.yaml", "3") in cites
    assert all(p != "b.py" for p, _ in cites)


# ---------- 真 git 仓：验证 ----------

@pytest.fixture()
def repo(tmp_path):
    r = tmp_path / "upstream"
    r.mkdir()

    def git(*args):
        subprocess.run(["git", "-C", str(r)] + list(args), check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (r / "pkg").mkdir()
    (r / "pkg" / "core.py").write_text("l1\nl2\nl3\nl4\nl5\n", encoding="utf-8")
    (r / "unique_name.py").write_text("a\nb\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "init")
    commit = subprocess.run(
        ["git", "-C", str(r), "rev-parse", "HEAD"], check=True, capture_output=True
    ).stdout.decode().strip()
    return r, commit


def _audit(tmp_path, repo, page_text):
    r, commit = repo
    entry = {"name": "up", "repo": "org/up", "checkout": r}
    page = tmp_path / "page.md"
    page.write_text(page_text.replace("COMMIT", commit), encoding="utf-8")
    return cl.audit_page(page, [entry], cl.GitView())


def test_pass_and_out_of_range_and_missing(tmp_path, repo):
    findings = _audit(
        tmp_path, repo,
        "> 源码基线：`org/up@COMMIT`\n\n"
        "对 `pkg/core.py:2` 和 `pkg/core.py:4-5`；超界 `pkg/core.py:99`；"
        "不存在 `pkg/ghost.py:1`。",
    )
    cats = [c for c, _ in findings]
    assert cats.count("pass") == 2
    assert cats.count("out_of_range") == 1
    assert cats.count("missing_file") == 1


def test_bare_basename_resolves_unique(tmp_path, repo):
    findings = _audit(
        tmp_path, repo,
        "> 源码基线：`org/up@COMMIT`\n\n裸文件名 `unique_name.py:2` 应按 basename 唯一解析。",
    )
    assert [c for c, _ in findings] == ["pass"]


def test_unanchored_page(tmp_path, repo):
    r, _ = repo
    entry = {"name": "up", "repo": "org/up", "checkout": r}
    page = tmp_path / "page.md"
    page.write_text("没有基线头，但引用了 `pkg/core.py:1`。", encoding="utf-8")
    findings = cl.audit_page(page, [entry], cl.GitView())
    assert [c for c, _ in findings] == ["unanchored_page"]


def test_commit_unavailable(tmp_path, repo):
    findings = _audit(
        tmp_path, repo,
        "> 源码基线：`org/up@deadbeefdeadbeefdeadbeefdeadbeefdeadbeef`\n\n`pkg/core.py:1`。",
    )
    assert [c for c, _ in findings] == ["commit_unavailable"]


def test_verified_range_uses_pinned_commit_not_worktree(tmp_path, repo):
    """checkout 工作区前进后，验证仍按页面钉的 commit。"""
    r, commit = repo

    def git(*args):
        subprocess.run(["git", "-C", str(r)] + list(args), check=True, capture_output=True)

    (r / "pkg" / "core.py").write_text("only1\n", encoding="utf-8")  # 工作区缩成 1 行
    git("add", "-A")
    git("commit", "-qm", "shrink")
    findings = _audit(
        tmp_path, (r, commit),
        "> 源码基线：`org/up@COMMIT`\n\n`pkg/core.py:5` 在钉住的 commit 里仍是合法行。",
    )
    assert [c for c, _ in findings] == ["pass"]


def test_unverifiable_when_page_pins_unresolvable_repo(tmp_path, repo):
    """页面另钉了一个 watchlist 之外的仓：查无此路径应降级为 unverifiable，不是 missing_file。"""
    findings = _audit(
        tmp_path, repo,
        "> 源码基线：`org/up@COMMIT`；辅仓 `someone/unknown-repo@deadbeefdead`" + chr(10) + chr(10)
        + "`pkg/core.py:1` 与 `their/only_in_unknown.py:7`。",
    )
    cats = sorted(c for c, _ in findings)
    assert "unverifiable" in cats and "missing_file" not in cats


# ---------- 分级：页面缺陷 vs 环境缺口 ----------

def test_unknown_repo_is_a_page_defect_not_an_environment_gap(tmp_path, repo):
    """页面钉了一个 watchlist 里没有的仓——这是页面/配置能修的，不是本机环境问题。"""
    r, commit = repo
    entry = {"name": "up", "repo": "org/up", "checkout": r}
    page = tmp_path / "page.md"
    page.write_text(
        "> 源码基线：`NoSuchRepo@" + commit + "`\n\n引用 `pkg/core.py:1`。",
        encoding="utf-8",
    )

    cats = [c for c, _ in cl.audit_page(page, [entry], cl.GitView())]

    assert "unknown_repo" in cats
    assert "unresolved_repo" not in cats


def test_category_tiers_partition_every_emitted_category():
    """三档必须互斥且穷尽，否则会有类别既不计数也不打印。"""
    tiers = [cl.ERROR_CATS, cl.WARN_CATS, cl.ENV_CATS]
    union = set().union(*tiers)

    assert sum(len(tier) for tier in tiers) == len(union), "分档之间有重叠"
    assert union == set(cl.REPORT_ORDER) - {"pass"}


def test_environment_gaps_are_not_page_defects():
    """本机没克隆、对象库里没这个 commit——作者改文档也修不掉，不该算页面告警。"""
    assert cl.ENV_CATS == {"unresolved_repo", "commit_unavailable", "unverifiable"}
    assert not (cl.ENV_CATS & cl.WARN_CATS)
    assert "out_of_range" in cl.WARN_CATS
    assert cl.ERROR_CATS == {"missing_file"}


def test_explicit_pages_override_directory_scan(tmp_path):
    """--changed / 显式文件列表要能把审计面收窄到给定页面。"""
    page = tmp_path / "only.md"
    page.write_text("# only\n", encoding="utf-8")
    other = tmp_path / "other.md"
    other.write_text("# other\n", encoding="utf-8")

    assert cl.pages_to_audit(tmp_path, [page]) == [page]
    assert sorted(cl.pages_to_audit(tmp_path, None)) == sorted([other, page])


def test_changelog_stays_excluded_from_explicit_lists(tmp_path):
    """changelog 历来豁免（正文大量引用历史行号），显式列表也不该把它拉回来。"""
    changelog = cl.ROOT / "wiki" / "changelog.md"

    assert cl.pages_to_audit(cl.ROOT, [changelog]) == []
