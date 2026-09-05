# -*- coding: utf-8 -*-
"""check_coverage 的单测：AST 字段枚举、词边界提及匹配、配置三查与变体门禁。"""
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

def test_c2_uses_same_matcher_as_generate(tmp_path, wiki):
    """C2 判定「owner 页面是否提及该 flag」必须与 --generate 的匹配器同口径。

    回归用例：页面只写 CLI 的 kebab 形式（`--beta-mode`）时，pages_mentioning 认它、
    C2 若改用下划线裸子串就会误报 stale_owner —— 两处口径不一致会把正确的人工归属
    判成陈旧，进而诱导把 owner 改到错的页上。
    """
    import yaml
    cfgp = tmp_path / "cov.yaml"
    cfgp.write_text(yaml.safe_dump({
        "domain": "demo", "wiki_dir": wiki, "repo": "n/a", "commit": "0" * 40,
        "sources": [],
        # 10_a_analysis 只写了 --beta-mode（kebab），没有 beta_mode 字面量
        "flags": [{"name": "beta_mode", "from": "C", "owner": "10_a_analysis"}],
    }, allow_unicode=True), encoding="utf-8")
    assert cc.pages_mentioning(wiki, "beta_mode") == ["10_a_analysis", "20_b_analysis"]
    rc = cc.check(cfgp, strict=True, examples=10)
    assert rc == 0, "kebab 形式的提及不应被判成 stale_owner"


def test_strict_rejects_nested_live_variant_missing_from_page_and_principle_figure(tmp_path, wiki):
    """二级 selector 不能被上层 umbrella 名称吞掉。

    回归用例：页面和图只写了 Flex/DeepEP，但冻结基线的 backend selector
    还有 HybridEP。旧 gate 忽略 variant_axes，会错误返回 0。
    """
    import yaml

    page = tmp_path / wiki / "10_a_analysis.md"
    page.write_text(
        "# Demo\n\n`moe_flex_dispatcher_backend` 选择 Flex 的 DeepEP。\n\n"
        "![dispatcher](assets/dispatcher.svg)\n",
        encoding="utf-8",
    )
    assets = page.parent / "assets"
    assets.mkdir()
    (assets / "dispatcher.svg").write_text(
        "<svg xmlns='http://www.w3.org/2000/svg'><text>Flex DeepEP</text></svg>",
        encoding="utf-8",
    )

    cfgp = tmp_path / "cov.yaml"
    cfgp.write_text(yaml.safe_dump({
        "domain": "demo", "wiki_dir": wiki, "repo": "n/a", "commit": "0" * 40,
        "sources": [], "flags": [],
        "variant_axes": [{
            "owner": "10_a_analysis",
            "selector": "moe_flex_dispatcher_backend",
            "source": "cfg.py::DemoConfig.moe_flex_dispatcher_backend",
            "variants": [
                {"name": "deepep", "page_terms": ["DeepEP"],
                 "figure_terms": ["DeepEP"], "figures": ["assets/dispatcher.svg"]},
                {"name": "hybridep", "page_terms": ["HybridEP"],
                 "figure_terms": ["HybridEP"], "figures": ["assets/dispatcher.svg"]},
            ],
        }],
    }, allow_unicode=True), encoding="utf-8")

    assert cc.check(cfgp, strict=True, examples=10) == 1

    # 只补页面名称仍不够：算法 data plane 还没有进入可视化 lane。
    page.write_text(page.read_text(encoding="utf-8") + "\nHybridEP 是另一条 live backend。\n", encoding="utf-8")
    assert cc.check(cfgp, strict=True, examples=10) == 1

    # 页面与它嵌入的原理图同时覆盖后才能通过。
    (assets / "dispatcher.svg").write_text(
        "<svg xmlns='http://www.w3.org/2000/svg'><text>Flex DeepEP HybridEP</text></svg>",
        encoding="utf-8",
    )
    assert cc.check(cfgp, strict=True, examples=10) == 0


def test_variant_figure_path_in_prose_does_not_count_as_an_embed(tmp_path, wiki):
    """A named asset is not evidence that readers can actually see the figure."""
    import yaml

    page = tmp_path / wiki / "10_a_analysis.md"
    page.write_text(
        "# Demo\n\n`backend` supports DeepEP and HybridEP. "
        "The asset lives at `assets/dispatcher.svg`, but is not embedded.\n",
        encoding="utf-8",
    )
    assets = page.parent / "assets"
    assets.mkdir()
    (assets / "dispatcher.svg").write_text(
        "<svg xmlns='http://www.w3.org/2000/svg'><text>DeepEP HybridEP</text></svg>",
        encoding="utf-8",
    )
    cfgp = tmp_path / "cov.yaml"
    cfgp.write_text(yaml.safe_dump({
        "domain": "demo", "wiki_dir": wiki, "repo": "n/a", "commit": "0" * 40,
        "sources": [], "flags": [],
        "variant_axes": [{
            "owner": "10_a_analysis", "selector": "backend", "source": "cfg.py::backend",
            "variants": [{
                "name": "hybridep", "page_terms": ["HybridEP"],
                "figure_terms": ["HybridEP"], "figures": ["assets/dispatcher.svg"],
            }],
        }],
    }, allow_unicode=True), encoding="utf-8")

    assert cc.check(cfgp, strict=True, examples=10) == 1


def test_variant_axis_cannot_omit_its_figure_contract(tmp_path, wiki):
    """A live algorithm variant must not pass merely because only page_terms were declared.

    Reviewers found this exact blind spot for inference EP, Hyper Connections, and
    LayerWise DistOpt: the prose named the sibling while another unrelated figure
    made the page look illustrated.  Every variant-axis entry must therefore name
    the rendered asset and the visible terms that prove that variant reached it.
    """
    import yaml

    page = tmp_path / wiki / "10_a_analysis.md"
    page.write_text(
        "# Demo\n\n`get_optimizer` selects LayerWiseDistributedOptimizer as a live sibling.\n",
        encoding="utf-8",
    )
    cfgp = tmp_path / "cov.yaml"
    cfgp.write_text(yaml.safe_dump({
        "domain": "demo", "wiki_dir": wiki, "repo": "n/a", "commit": "0" * 40,
        "sources": [], "flags": [],
        "variant_axes": [{
            "owner": "10_a_analysis",
            "selector": "get_optimizer",
            "source": "optimizer.py::get_optimizer",
            "variants": [{
                "name": "layer-wise",
                "page_terms": ["LayerWiseDistributedOptimizer"],
            }],
        }],
    }, allow_unicode=True), encoding="utf-8")

    assert cc.check(cfgp, strict=True, examples=10) == 1


def test_variant_term_hidden_in_svg_metadata_is_not_visible_figure_evidence(tmp_path, wiki):
    """figure_terms must be reader-visible SVG text, not metadata or aria-label only."""
    import yaml

    page = tmp_path / wiki / "10_a_analysis.md"
    page.write_text(
        "# Demo\n\n`backend` supports HybridEP.\n\n![dispatcher](assets/dispatcher.svg)\n",
        encoding="utf-8",
    )
    assets = page.parent / "assets"
    assets.mkdir()
    (assets / "dispatcher.svg").write_text(
        "<svg xmlns='http://www.w3.org/2000/svg' aria-label='HybridEP'>"
        "<metadata>HybridEP</metadata><text>generic Flex lane</text></svg>",
        encoding="utf-8",
    )
    cfgp = tmp_path / "cov.yaml"
    cfgp.write_text(yaml.safe_dump({
        "domain": "demo", "wiki_dir": wiki, "repo": "n/a", "commit": "0" * 40,
        "sources": [], "flags": [],
        "variant_axes": [{
            "owner": "10_a_analysis", "selector": "backend", "source": "cfg.py::backend",
            "variants": [{
                "name": "hybridep", "page_terms": ["HybridEP"],
                "figure_terms": ["HybridEP"], "figures": ["assets/dispatcher.svg"],
            }],
        }],
    }, allow_unicode=True), encoding="utf-8")

    assert cc.check(cfgp, strict=True, examples=10) == 1


def test_help_survives_a_legacy_codepage_console():
    """帮助文本在 reconfigure 之前就被打印，非 GBK 字符（如 ↔）会让 --help 直接崩。"""
    import os
    import subprocess
    import sys
    from pathlib import Path

    env = {**os.environ, "PYTHONIOENCODING": "gbk"}
    for checker in ("check_coverage.py", "check_locators.py"):
        script = Path(__file__).resolve().parent / checker
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            errors="replace",
            env=env,
        )
        assert result.returncode == 0, f"{checker}: {result.stderr[-400:]}"
        assert "UnicodeEncodeError" not in result.stderr
