"""check_assets.py 的行为测试。全部用 tmp_path 造微型页面,不绑定真实文章。"""

from pathlib import Path

import check_assets


def make(tmp_path: Path, files: dict) -> Path:
    """按 {相对路径: 文本} 落盘;值为 None 时只建空文件(当资源用)。"""
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"" if text is None else text.encode("utf-8"))
    return tmp_path


def page(tmp_path: Path, body: str, files: dict | None = None) -> Path:
    make(tmp_path, {"note.md": body, **(files or {})})
    return tmp_path / "note.md"


def codes(path: Path) -> list[str]:
    return [item.code for item in check_assets.check_file(path)]


# ── 引用抽取(纯文本,不碰文件系统) ──────────────────────────────────────────

def test_iter_references_covers_image_html_and_resource_link():
    text = (
        "![图](assets/fig.png)\n"
        '<img src="assets/inline.svg">\n'
        '<source srcset="assets/a.webp 1x, assets/b.webp 2x">\n'
        "[规格书](assets/spec.pdf)\n"
    )
    refs = list(check_assets.iter_references(text))
    assert [(r.target, r.line) for r in refs] == [
        ("assets/fig.png", 1),
        ("assets/inline.svg", 2),
        ("assets/a.webp", 3),
        ("assets/b.webp", 3),
        ("assets/spec.pdf", 4),
    ]


def test_iter_references_keeps_page_links_out():
    """相对 .md / 锚点链接是 check_links.py 的地盘,本检查器不碰。"""
    text = "[同域页面](other.md) 与 [本页小节](#结论) 与 [[wiki/page]]\n"
    assert list(check_assets.iter_references(text)) == []


def test_iter_references_skips_external_and_scheme_targets():
    text = (
        "![远程](https://example.com/a.png)\n"
        "![远程](http://example.com/b.png)\n"
        "![协议相对](//cdn.example.com/c.png)\n"
        "[邮件](mailto:someone@example.com)\n"
        "[本地源码](file:///e:/repo/x.py#L17)\n"
        "[内联数据](data:image/png;base64,AAAA)\n"
    )
    assert list(check_assets.iter_references(text)) == []


def test_iter_references_handles_brackets_parens_titles_and_encoding():
    text = (
        "![① scan(log 步,1→2) 与 [a,b) 区间](assets/fig%201.png)\n"
        '![带标题](assets/fig2.png "示意图")\n'
        "![尖括号](<assets/fig 3.png>)\n"
        "![带片段](assets/fig4.svg#frag)\n"
    )
    assert [r.target for r in check_assets.iter_references(text)] == [
        "assets/fig%201.png",
        "assets/fig2.png",
        "assets/fig 3.png",
        "assets/fig4.svg#frag",
    ]


def test_iter_references_ignores_fenced_and_inline_code():
    text = (
        "```markdown\n"
        "![示例](assets/ghost.png)\n"
        "```\n"
        "行内 `![示例](assets/ghost2.png)` 不算\n"
        "~~~\n"
        "![示例](assets/ghost3.png)\n"
        "~~~\n"
        "![真的](assets/real.png)\n"
    )
    assert [(r.target, r.line) for r in check_assets.iter_references(text)] == [
        ("assets/real.png", 8)
    ]


def test_iter_references_skips_srcset_with_data_uri():
    text = '<source srcset="data:image/png;base64,AAAA 1x, assets/b.png 2x">\n'
    assert list(check_assets.iter_references(text)) == []


# ── 存在性判定 ────────────────────────────────────────────────────────────

def test_existing_assets_produce_no_diagnostics(tmp_path):
    note = page(
        tmp_path,
        "![图](assets/fig.png)\n[规格书](assets/spec.pdf)\n",
        {"assets/fig.png": None, "assets/spec.pdf": None},
    )
    assert check_assets.check_file(note) == []


def test_missing_asset_is_an_error_with_line_number(tmp_path):
    note = page(tmp_path, "开头\n\n![图](assets/fig.png)\n")
    diagnostics = check_assets.check_file(note)
    assert len(diagnostics) == 1
    only = diagnostics[0]
    assert (only.code, only.severity, only.line) == ("ASSET001", "error", 3)
    assert "assets/fig.png" in only.message


def test_paths_resolve_against_the_page_directory(tmp_path):
    make(
        tmp_path,
        {
            "sub/note.md": "![上层](../assets/up.png) ![同级](fig.png) ![缺](../assets/no.png)\n",
            "assets/up.png": None,
            "sub/fig.png": None,
        },
    )
    diagnostics = check_assets.check_file(tmp_path / "sub" / "note.md")
    assert [item.code for item in diagnostics] == ["ASSET001"]
    assert "../assets/no.png" in diagnostics[0].message


def test_percent_encoded_and_directory_targets(tmp_path):
    note = page(
        tmp_path,
        "![空格](assets/one%20fig.png)\n![目录](assets)\n",
        {"assets/one fig.png": None},
    )
    diagnostics = check_assets.check_file(note)
    assert [item.code for item in diagnostics] == ["ASSET001"]
    assert "'assets'" in diagnostics[0].message


def test_html_img_and_srcset_existence(tmp_path):
    note = page(
        tmp_path,
        '<img src="assets/have.svg">\n'
        '<source srcset="assets/have.svg 1x, assets/gone.svg 2x">\n',
        {"assets/have.svg": None},
    )
    diagnostics = check_assets.check_file(note)
    assert [item.code for item in diagnostics] == ["ASSET001"]
    assert (diagnostics[0].line, "assets/gone.svg" in diagnostics[0].message) == (2, True)


def test_case_mismatch_is_a_warning_not_an_error(tmp_path):
    """大小写不符在 Windows 本地能渲染,推到 Linux 站点就 404 —— 必须报出来。"""
    note = page(tmp_path, "![图](assets/fig.png)\n", {"assets/Fig.PNG": None})
    diagnostics = check_assets.check_file(note)
    assert [(item.code, item.severity) for item in diagnostics] == [
        ("ASSET101", "warning")
    ]
    assert "Fig.PNG" in diagnostics[0].message


def test_case_mismatch_in_a_directory_component(tmp_path):
    note = page(tmp_path, "![图](Assets/fig.png)\n", {"assets/fig.png": None})
    assert codes(note) == ["ASSET101"]


def test_diagnostics_are_sorted_by_line_then_message(tmp_path):
    note = page(tmp_path, "![a](z.png) ![b](a.png)\n![c](z.png)\n")
    diagnostics = check_assets.check_file(note)
    assert [(item.line, "z.png" in item.message) for item in diagnostics] == [
        (1, False),
        (1, True),
        (2, True),
    ]


# ── 扫描与 CLI ────────────────────────────────────────────────────────────

def test_collect_explicit_paths_walks_directories(tmp_path):
    make(tmp_path, {"a/one.md": "", "a/b/two.md": "", "a/b/skip.txt": ""})
    found = check_assets._collect_explicit_paths([str(tmp_path)])
    assert {p.name for p in found} == {"one.md", "two.md"}


def test_cli_exit_codes_and_summary(tmp_path, monkeypatch, capsys):
    clean = make(tmp_path / "ok", {"note.md": "![图](assets/fig.png)\n", "assets/fig.png": None})
    monkeypatch.setattr("sys.argv", ["check_assets.py", str(clean)])
    assert check_assets.main() == 0
    assert "Checked 1 Markdown file(s): 0 error(s), 0 warning(s)." in capsys.readouterr().out

    broken = make(tmp_path / "bad", {"note.md": "![图](assets/fig.png)\n"})
    monkeypatch.setattr("sys.argv", ["check_assets.py", str(broken)])
    assert check_assets.main() == 1
    out = capsys.readouterr().out
    assert "[ERROR ASSET001]" in out
    assert "Checked 1 Markdown file(s): 1 error(s), 0 warning(s)." in out


def test_cli_strict_turns_warnings_into_failure(tmp_path, monkeypatch, capsys):
    wiki = make(tmp_path, {"note.md": "![图](assets/fig.png)\n", "assets/FIG.png": None})
    monkeypatch.setattr("sys.argv", ["check_assets.py", str(wiki)])
    assert check_assets.main() == 0
    assert "0 error(s), 1 warning(s)." in capsys.readouterr().out

    monkeypatch.setattr("sys.argv", ["check_assets.py", "--strict", str(wiki)])
    assert check_assets.main() == 1


def test_changed_collector_is_reused_from_check_math():
    """变更集不另造轮子:直接复用 check_math 的收集器。"""
    import check_math

    assert check_assets.collect_changed_markdown is check_math.collect_changed_markdown


def test_cli_changed_flag_checks_the_collected_files(tmp_path, monkeypatch):
    note = make(tmp_path, {"note.md": "![图](assets/fig.png)\n"}) / "note.md"
    monkeypatch.setattr(
        check_assets, "collect_changed_markdown", lambda repo_root: [note]
    )
    monkeypatch.setattr("sys.argv", ["check_assets.py", "--changed"])
    assert check_assets.main() == 1
