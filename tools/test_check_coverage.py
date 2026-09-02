# -*- coding: utf-8 -*-
"""check_coverage 的单测：AST 字段枚举、词边界提及匹配、三查分类。"""
import subprocess
import textwrap

import pytest

import check_coverage as cc


@pytest.fixture()
def repo(tmp_path):
    r = tmp_path / "up"
    r.mkdir()

    def git(*args):
        subprocess.run(["git", "-C", str(r)] + list(args), check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (r / "cfg.py").write_text(textwrap.dedent('''\
        from dataclasses import dataclass

        @dataclass
        class DemoConfig:
            alpha_size: int = 1
            beta_mode: str = "x"

            def method(self):          # 方法不是字段
                pass

        class Other:
            gamma: int = 0
    '''), encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "init")
    commit = subprocess.run(
        ["git", "-C", str(r), "rev-parse", "HEAD"], check=True, capture_output=True
    ).stdout.decode().strip()
    return r, commit


def test_enumerate_fields_only_annassign(repo):
    r, commit = repo
    assert cc.enumerate_fields(r, commit, "cfg.py", "DemoConfig") == ["alpha_size", "beta_mode"]


def test_enumerate_missing_class_exits(repo):
    r, commit = repo
    with pytest.raises(SystemExit):
        cc.enumerate_fields(r, commit, "cfg.py", "Nope")


@pytest.fixture()
def wiki(tmp_path, monkeypatch):
    d = tmp_path / "wiki" / "demo"
    d.mkdir(parents=True)
    (d / "index.md").write_text("alpha_size 出现在 index 不算", encoding="utf-8")
    (d / "10_a_analysis.md").write_text("讲 `alpha_size` 与 --beta-mode（kebab 形式）", encoding="utf-8")
    (d / "20_b_analysis.md").write_text("也提 beta_mode；但 alpha_size_extra 是别的词", encoding="utf-8")
    monkeypatch.setattr(cc, "ROOT", tmp_path)
    return "wiki/demo"


def test_pages_mentioning_word_boundary_and_kebab(wiki):
    assert cc.pages_mentioning(wiki, "alpha_size") == ["10_a_analysis"]          # extra 词不算、index 不算
    assert cc.pages_mentioning(wiki, "beta_mode") == ["10_a_analysis", "20_b_analysis"]  # kebab 也算


def test_check_three_classes(tmp_path, wiki):
    import yaml
    cfgp = tmp_path / "cov.yaml"
    cfgp.write_text(yaml.safe_dump({
        "domain": "demo", "wiki_dir": wiki, "repo": "n/a", "commit": "0" * 40,
        "sources": [],
        "flags": [
            {"name": "alpha_size", "from": "C", "owner": "10_a_analysis", "auto": True},
            {"name": "ghost_flag", "from": "C", "owner": None},                       # C1 未提及
            {"name": "beta_mode", "from": "C", "owner": None,
             "candidates": ["10_a_analysis", "20_b_analysis"]},                       # C1 待定
            {"name": "gone_flag", "from": "C", "owner": "20_b_analysis"},             # C2 手工 owner 但页面没提
            {"name": "bad_ref", "from": "C", "owner": "99_missing_analysis"},         # C3
            {"name": "debug_only", "from": "C", "owner": None, "excluded": "调试开关"},  # 排除不算 gap
        ],
    }, allow_unicode=True), encoding="utf-8")
    rc = cc.check(cfgp, strict=False, examples=10)
    assert rc == 1  # C3 是 error


def test_generate_preserves_manual_rows(tmp_path, repo, wiki, monkeypatch):
    import yaml
    r, commit = repo
    monkeypatch.setattr(cc, "find_checkout", lambda name: r)
    cfgp = tmp_path / "cov.yaml"
    cfgp.write_text(yaml.safe_dump({
        "domain": "demo", "wiki_dir": wiki, "repo": "up", "commit": commit,
        "sources": [{"file": "cfg.py", "class": "DemoConfig"}],
        "flags": [{"name": "alpha_size", "from": "DemoConfig",
                   "owner": "20_b_analysis", "note": "人工指定"}],   # 无 auto → 保留
    }, allow_unicode=True), encoding="utf-8")
    cc.generate(cfgp)
    data = yaml.safe_load(cfgp.read_text(encoding="utf-8"))
    by = {f["name"]: f for f in data["flags"]}
    assert by["alpha_size"]["owner"] == "20_b_analysis"      # 人工行未被自动建议覆盖
    assert by["beta_mode"]["candidates"] == ["10_a_analysis", "20_b_analysis"]


def test_generate_preserves_excluded_rows(tmp_path, repo, wiki, monkeypatch):
    """`excluded` 是人工决定，即便没有 owner 也不能被重新生成覆盖。"""
    import yaml
    r, commit = repo
    monkeypatch.setattr(cc, "find_checkout", lambda name: r)
    cfgp = tmp_path / "cov.yaml"
    cfgp.write_text(yaml.safe_dump({
        "domain": "demo", "wiki_dir": wiki, "repo": "up", "commit": commit,
        "sources": [{"file": "cfg.py", "class": "DemoConfig"}],
        "flags": [{"name": "alpha_size", "from": "DemoConfig",
                   "owner": None, "excluded": "内部调试开关"}],
    }, allow_unicode=True), encoding="utf-8")
    cc.generate(cfgp)
    by = {f["name"]: f for f in yaml.safe_load(cfgp.read_text(encoding="utf-8"))["flags"]}
    assert by["alpha_size"]["excluded"] == "内部调试开关"
    assert by["alpha_size"].get("owner") is None            # 未被自动建议改写


def test_generate_refreshes_undecided_rows(tmp_path, repo, wiki, monkeypatch):
    """owner=null 且未 excluded 的行只是机器建议，不是人工决定——重新生成时必须刷新。

    回归用例：候选清单会随页面内容变化（新页写了该 flag、旧页删了提及）。若把这类行
    当人工行冻住，候选就永远停在首次生成时的快照，据此定归属会定错。
    """
    import yaml
    r, commit = repo
    monkeypatch.setattr(cc, "find_checkout", lambda name: r)
    cfgp = tmp_path / "cov.yaml"
    cfgp.write_text(yaml.safe_dump({
        "domain": "demo", "wiki_dir": wiki, "repo": "up", "commit": commit,
        "sources": [{"file": "cfg.py", "class": "DemoConfig"}],
        # 陈旧快照：alpha_size 当时只被 20 提及、beta_mode 当时只有 10
        "flags": [
            {"name": "alpha_size", "from": "DemoConfig", "owner": None,
             "candidates": ["20_b_analysis"]},
            {"name": "beta_mode", "from": "DemoConfig", "owner": None,
             "candidates": ["10_a_analysis"]},
        ],
    }, allow_unicode=True), encoding="utf-8")
    cc.generate(cfgp)
    by = {f["name"]: f for f in yaml.safe_load(cfgp.read_text(encoding="utf-8"))["flags"]}
    # 现网页面里 alpha_size 只有 10 提及 → 应刷新为 auto owner，而不是留着陈旧的 20
    assert by["alpha_size"]["owner"] == "10_a_analysis"
    assert by["alpha_size"].get("auto") is True
    assert "candidates" not in by["alpha_size"]
    # beta_mode 现在两页都提及 → 候选清单应刷新为两页
    assert by["beta_mode"]["candidates"] == ["10_a_analysis", "20_b_analysis"]
