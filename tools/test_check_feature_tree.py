# -*- coding: utf-8 -*-
"""check_feature_tree 的单测：schema/真空防护、Class.field 身份、nodes 树模型、phase 门禁、双向对账。"""
import subprocess
import textwrap

import pytest
import yaml

import check_feature_tree as cft


@pytest.fixture()
def repo(tmp_path):
    r = tmp_path / "up"
    r.mkdir()

    def git(*args):
        subprocess.run(["git", "-C", str(r)] + list(args), check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (r / "pkg").mkdir()
    (r / "pkg" / "a.py").write_text(textwrap.dedent('''\
        from dataclasses import dataclass

        @dataclass
        class Cfg:
            alpha: int = 1
            beta: str = "x"

        @dataclass
        class Other:
            alpha: int = 0

        def main():
            return Cfg()
    '''), encoding="utf-8")
    (r / "pkg" / "b.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    (r / "tests").mkdir()
    (r / "tests" / "test_a.py").write_text("def test_x():\n    assert True\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "init")
    commit = subprocess.run(
        ["git", "-C", str(r), "rev-parse", "HEAD"], check=True, capture_output=True
    ).stdout.decode().strip()
    return r, commit


def base_manifest(r, commit):
    return {
        "domain": "demo",
        "checkout": str(r),
        "commit": commit,
        "surfaces": {
            "files": {"include": ["pkg/**", "tests/**"]},
            "flags": [{"file": "pkg/a.py", "class": "Cfg"}],
            "entries": ["pkg/a.py::main"],
        },
        "nodes": [
            {"id": "core", "name": "core", "responsibility": "runs and helps", "parent": None},
        ],
        "leaves": [
            {
                "id": "core/run", "name": "run", "definition": "builds the config and runs",
                "entry": "pkg/a.py::main", "status": "planned",
                "owns": {"files": ["pkg/a.py"], "flags": ["alpha", "beta"],
                         "entries": ["pkg/a.py::main"]},
            },
            {
                "id": "core/help", "name": "helper", "entry": "pkg/b.py::helper", "status": "planned",
                "owns": {"files": ["pkg/b.py"]},
            },
        ],
        "exclusions": {
            "files": [{"glob": "tests/**", "reason": "tests: referenced from test anchors"}],
        },
    }


def write_manifest(tmp_path, data, name="ft.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return p


def spec_page(tmp_path, name, *leaf_ids):
    d = tmp_path / "specs"
    d.mkdir(exist_ok=True)
    body = "# spec\n\n" + "".join(f"### {lid} something\n- Position: x\n\n" for lid in leaf_ids)
    (d / f"{name}.md").write_text(body, encoding="utf-8")


def review(leaf, verdict="PASS"):
    return {"leaf": leaf, "date": "2026-09-02", "reviewer": "r", "r1": "3/3", "r2": "pass",
            "r3": "pass", "r4": "pass", "r5": "pass", "r6": "pass", "verdict": verdict}


# ---------------------------------------------------------------- happy path / reconciliation

def test_clean_manifest_has_no_findings(repo, tmp_path):
    r, commit = repo
    p = write_manifest(tmp_path, base_manifest(r, commit))
    report = cft.audit(p)
    assert cft.counts(report) == (0, 0), report
    assert cft.main([str(p), "--strict"]) == 0


def test_stable_symbol_entries_are_the_default_manifest_anchor(repo, tmp_path):
    r, commit = repo
    m = base_manifest(r, commit)
    m["leaves"][0]["entry"] = "pkg/a.py::main"
    m["leaves"][1]["entry"] = "pkg/b.py::helper"
    report = cft.audit(write_manifest(tmp_path, m))
    assert report["F3"] == []


def test_unowned_file_is_warning_and_strict_fails(repo, tmp_path):
    r, commit = repo
    m = base_manifest(r, commit)
    del m["leaves"][1]  # pkg/b.py 无人认领
    p = write_manifest(tmp_path, m)
    report = cft.audit(p)
    assert report["F1"] == ["pkg/b.py"]
    assert cft.counts(report) == (0, 1)
    assert cft.main([str(p)]) == 0
    assert cft.main([str(p), "--strict"]) == 1


def test_phantom_file_and_bad_legacy_line_entry_are_errors(repo, tmp_path):
    r, commit = repo
    m = base_manifest(r, commit)
    m["leaves"][1]["owns"]["files"] = ["pkg/nope.py"]
    m["leaves"][1]["entry"] = "pkg/b.py:999"
    p = write_manifest(tmp_path, m)
    report = cft.audit(p)
    assert report["F2"] == ["core/help -> pkg/nope.py"]
    assert report["F3"] == ["core/help -> pkg/b.py:999"]
    assert report["F1"] == ["pkg/b.py"]  # 幻影认领不算认领
    assert cft.counts(report)[0] == 2


def test_entry_gap_and_unknown_entry(repo, tmp_path):
    r, commit = repo
    m = base_manifest(r, commit)
    m["surfaces"]["entries"] = ["pkg/a.py::main", "pkg/b.py::helper"]
    m["leaves"][1]["owns"]["entries"] = ["pkg/b.py::nope"]
    p = write_manifest(tmp_path, m)
    report = cft.audit(p)
    assert report["E1"] == ["pkg/b.py::helper"]
    assert report["E2"] == ["core/help -> pkg/b.py::nope"]


def test_overview_table_mismatch(repo, tmp_path):
    r, commit = repo
    m = base_manifest(r, commit)
    m["overview"] = "overview.md"
    (tmp_path / "overview.md").write_text(textwrap.dedent('''\
        | ID | Function point | One-line definition | Entry | Spec anchor | Status |
        |---|---|---|---|---|---|
        | `core/run` | run | runs | `pkg/a.py:12` | — | spec'd |
        | `core/extra` | extra | not in manifest | `x` | — | planned |
    '''), encoding="utf-8")
    p = write_manifest(tmp_path, m)
    report = cft.audit(p)
    assert sorted(report["S2"]) == sorted([
        "core/run: overview status spec'd != manifest planned",
        "core/extra: in overview, not in manifest",
        "core/help: in manifest, not in overview",
    ])


def test_missing_commit_exits(repo, tmp_path):
    r, commit = repo
    m = base_manifest(r, commit)
    m["commit"] = "deadbeef" * 5
    p = write_manifest(tmp_path, m)
    with pytest.raises(SystemExit):
        cft.audit(p)


# ---------------------------------------------------------------- P1-3 schema / vacuum protection

def _schema_errors(tmp_path, m):
    report = cft.audit(write_manifest(tmp_path, m))
    return report["X1"]


def test_schema_rejects_unknown_keys_at_every_level(repo, tmp_path):
    r, commit = repo
    m = base_manifest(r, commit)
    m["bogus"] = 1
    m["surfaces"]["files"]["includ"] = ["pkg/**"]
    m["leaves"][0]["own"] = {}
    errs = _schema_errors(tmp_path, m)
    assert any("unknown key 'bogus'" in e for e in errs)
    assert any("unknown key 'includ'" in e for e in errs)
    assert any("unknown key 'own'" in e for e in errs)


def test_schema_rejects_empty_leaves_and_non_frozen_commit(repo, tmp_path):
    r, commit = repo
    m = base_manifest(r, commit)
    m["leaves"] = []
    m["commit"] = "HEAD"
    errs = _schema_errors(tmp_path, m)
    assert any("leaves" in e and "non-empty" in e for e in errs)
    assert any("commit" in e and "40" in e for e in errs)


def test_schema_rejects_include_as_string_and_missing_include(repo, tmp_path):
    r, commit = repo
    m = base_manifest(r, commit)
    m["surfaces"]["files"]["include"] = "pkg/**"
    assert any("include" in e for e in _schema_errors(tmp_path, m))
    m["surfaces"]["files"] = {}
    assert any("include" in e for e in _schema_errors(tmp_path, m))


def test_schema_requires_exclusion_reason(repo, tmp_path):
    r, commit = repo
    m = base_manifest(r, commit)
    m["exclusions"]["files"] = [{"glob": "tests/**"}]
    assert any("reason" in e for e in _schema_errors(tmp_path, m))
    m["exclusions"]["files"] = [{"glob": "tests/**", "reason": ""}]
    assert any("reason" in e for e in _schema_errors(tmp_path, m))


def test_schema_rejects_bad_ids_status_and_node_fields(repo, tmp_path):
    r, commit = repo
    m = base_manifest(r, commit)
    m["leaves"][0]["id"] = "Core/Run"
    m["leaves"][1]["status"] = "done"
    m["nodes"][0]["responsibility"] = ""
    errs = _schema_errors(tmp_path, m)
    assert any("Core/Run" in e for e in errs)
    assert any("status" in e and "done" in e for e in errs)
    assert any("responsibility" in e for e in errs)


def test_schema_errors_short_circuit_other_checks(repo, tmp_path):
    r, commit = repo
    m = base_manifest(r, commit)
    m["leaves"] = "not a list"
    report = cft.audit(write_manifest(tmp_path, m))
    assert report["X1"]
    assert all(v == [] for k, v in report.items() if k != "X1")
    assert cft.counts(report)[0] >= 1


def test_no_leaves_with_reasonless_star_exclusion_cannot_zero(repo, tmp_path):
    r, commit = repo
    m = base_manifest(r, commit)
    m["leaves"] = []
    m["exclusions"] = {"files": [{"glob": "**"}]}
    p = write_manifest(tmp_path, m)
    assert cft.main([str(p), "--strict"]) == 1


def test_include_matching_nothing_and_empty_scope_are_errors(repo, tmp_path):
    r, commit = repo
    m = base_manifest(r, commit)
    m["surfaces"]["files"]["include"] = ["nothing/**"]
    report = cft.audit(write_manifest(tmp_path, m))
    assert any("nothing/**" in e and "matches no file" in e for e in report["V0"])
    assert any("scope is empty" in e for e in report["V0"])
    m["surfaces"]["files"]["include"] = ["pkg/**", "typo/**"]
    report = cft.audit(write_manifest(tmp_path, m))
    assert [e for e in report["V0"] if "typo/**" in e]
    assert not [e for e in report["V0"] if "scope is empty" in e]


def test_stale_exclusion_is_warning(repo, tmp_path):
    r, commit = repo
    m = base_manifest(r, commit)
    m["exclusions"]["files"].append({"glob": "vendor/**", "reason": "third-party"})
    report = cft.audit(write_manifest(tmp_path, m))
    assert report["F4"] == ["vendor/**"]
    assert cft.counts(report) == (0, 1)


# ---------------------------------------------------------------- P1-4 flag identity

def test_flag_gap_and_unknown_flag_use_qualified_names(repo, tmp_path):
    r, commit = repo
    m = base_manifest(r, commit)
    m["leaves"][0]["owns"]["flags"] = ["alpha", "gamma"]
    report = cft.audit(write_manifest(tmp_path, m))
    assert report["G1"] == ["Cfg.beta"]
    assert report["G2"] == ["core/run -> gamma"]


def test_ambiguous_bare_flag_is_error_and_qualified_names_resolve(repo, tmp_path):
    r, commit = repo
    m = base_manifest(r, commit)
    m["surfaces"]["flags"].append({"file": "pkg/a.py", "class": "Other"})
    report = cft.audit(write_manifest(tmp_path, m))
    assert report["G3"] == ["core/run -> alpha (Cfg.alpha, Other.alpha)"]
    m["leaves"][0]["owns"]["flags"] = ["Cfg.alpha", "Other.alpha", "beta"]
    report = cft.audit(write_manifest(tmp_path, m))
    assert report["G1"] == [] and report["G2"] == [] and report["G3"] == []


def test_excluded_flag_is_not_a_gap_and_ambiguous_exclusion_is_error(repo, tmp_path):
    r, commit = repo
    m = base_manifest(r, commit)
    m["leaves"][0]["owns"]["flags"] = ["alpha"]
    m["exclusions"]["flags"] = [{"name": "Cfg.beta", "reason": "debug-only"}]
    assert cft.audit(write_manifest(tmp_path, m))["G1"] == []
    m["surfaces"]["flags"].append({"file": "pkg/a.py", "class": "Other"})
    m["leaves"][0]["owns"]["flags"] = ["Cfg.alpha", "Other.alpha"]
    m["exclusions"]["flags"] = [{"name": "beta", "reason": "x"}, {"name": "alpha", "reason": "y"}]
    report = cft.audit(write_manifest(tmp_path, m))
    assert report["G3"] == ["exclusion -> alpha (Cfg.alpha, Other.alpha)"]


# ---------------------------------------------------------------- P1-1 tree model

def test_leaf_without_parent_node_is_error(repo, tmp_path):
    r, commit = repo
    m = base_manifest(r, commit)
    m["leaves"][1]["id"] = "extra/help"
    report = cft.audit(write_manifest(tmp_path, m))
    assert report["T1"] == ["extra/help: missing parent node extra"]


def test_node_parent_must_match_id_prefix_and_exist(repo, tmp_path):
    r, commit = repo
    m = base_manifest(r, commit)
    m["nodes"].append({"id": "core/sub", "name": "sub", "responsibility": "x", "parent": "other"})
    m["nodes"].append({"id": "deep/er", "name": "d", "responsibility": "x", "parent": "deep"})
    m["leaves"][1]["id"] = "core/sub/help"
    report = cft.audit(write_manifest(tmp_path, m))
    assert "core/sub: parent 'other' != 'core'" in report["T1"]
    assert "deep/er: missing parent node deep" in report["T1"]


def test_node_without_children_is_error(repo, tmp_path):
    r, commit = repo
    m = base_manifest(r, commit)
    m["nodes"].append({"id": "empty", "name": "e", "responsibility": "nothing here", "parent": None})
    report = cft.audit(write_manifest(tmp_path, m))
    assert report["T2"] == ["empty"]


def test_duplicate_ids_across_nodes_and_leaves_are_errors(repo, tmp_path):
    r, commit = repo
    m = base_manifest(r, commit)
    m["leaves"][1]["id"] = "core/run"
    m["nodes"].append({"id": "core", "name": "again", "responsibility": "x", "parent": None})
    report = cft.audit(write_manifest(tmp_path, m))
    assert sorted(report["D1"]) == ["core", "core/run"]


# ---------------------------------------------------------------- P1-2 phases

def test_status_requires_existing_spec_page_and_anchor(repo, tmp_path):
    r, commit = repo
    m = base_manifest(r, commit)
    m["spec_dir"] = "specs"
    m["leaves"][0]["status"] = "spec'd"
    p = write_manifest(tmp_path, m)
    assert cft.audit(p)["S1"] == ["core/run: status spec'd without spec"]
    m["leaves"][0]["spec"] = "10_core_analysis#core-run"
    p = write_manifest(tmp_path, m)
    assert cft.audit(p)["S1"] == ["core/run -> specs/10_core_analysis.md missing"]
    spec_page(tmp_path, "10_core_analysis", "core/other")
    assert cft.audit(p)["S1"] == []
    assert cft.audit(p)["S3"] == ["core/run: no heading containing the leaf id in 10_core_analysis.md"]
    spec_page(tmp_path, "10_core_analysis", "core/run")
    assert cft.audit(p)["S3"] == []


def test_spec_and_delivery_phases_require_spec_dir(repo, tmp_path):
    r, commit = repo
    p = write_manifest(tmp_path, base_manifest(r, commit))
    assert cft.audit(p, phase="proposal")["X1"] == []
    assert any("spec_dir" in e for e in cft.audit(p, phase="spec")["X1"])
    assert any("spec_dir" in e for e in cft.audit(p, phase="delivery")["X1"])


def test_delivery_requires_every_leaf_verified_with_pass_review(repo, tmp_path):
    r, commit = repo
    m = base_manifest(r, commit)
    m["spec_dir"] = "specs"
    spec_page(tmp_path, "10_core_analysis", "core/run", "core/help")
    for leaf in m["leaves"]:
        leaf["spec"] = "10_core_analysis"
    m["leaves"][0]["status"] = "verified"
    m["leaves"][1]["status"] = "spec'd"
    m["reviews"] = [review("core/run", "REJECT"), review("core/nope")]
    p = write_manifest(tmp_path, m)
    report = cft.audit(p, phase="delivery")
    assert report["V1"] == ["core/run: no PASS review", "review for unknown leaf core/nope"]
    assert report["V2"] == ["core/help: status spec'd"]
    m["leaves"][1]["status"] = "verified"
    m["reviews"] = [review("core/run"), review("core/help")]
    p = write_manifest(tmp_path, m)
    report = cft.audit(p, phase="delivery")
    assert cft.counts(report) == (0, 0), report
    assert cft.main([str(p), "--phase", "delivery"]) == 0


def test_verified_leaf_needs_pass_review_in_every_phase(repo, tmp_path):
    r, commit = repo
    m = base_manifest(r, commit)
    m["spec_dir"] = "specs"
    spec_page(tmp_path, "10_core_analysis", "core/run")
    m["leaves"][0].update(status="verified", spec="10_core_analysis")
    p = write_manifest(tmp_path, m)
    assert cft.audit(p)["V1"] == ["core/run: no PASS review"]


def test_delivery_phase_implies_strict(repo, tmp_path):
    r, commit = repo
    m = base_manifest(r, commit)
    m["spec_dir"] = "specs"
    spec_page(tmp_path, "10_core_analysis", "core/run")
    m["leaves"] = [m["leaves"][0]]
    m["leaves"][0].update(status="verified", spec="10_core_analysis")
    m["reviews"] = [review("core/run")]
    p = write_manifest(tmp_path, m)  # pkg/b.py unowned -> F1 warning only
    assert cft.main([str(p)]) == 0
    assert cft.main([str(p), "--phase", "delivery"]) == 1


# ---------------------------------------------------------------- misc

def test_symbol_entry_must_exist_at_the_frozen_commit(repo, tmp_path):
    r, commit = repo
    m = base_manifest(r, commit)
    m["leaves"][0]["entry"] = "pkg/a.py::Cfg.alpha"   # class attribute, qualified
    m["leaves"][1]["entry"] = "pkg/b.py::helper"
    assert cft.audit(write_manifest(tmp_path, m))["F3"] == []
    m["leaves"][0]["entry"] = "pkg/a.py::nope"        # file exists, symbol does not
    m["leaves"][1]["entry"] = "pkg/nope.py::helper"   # file does not exist
    report = cft.audit(write_manifest(tmp_path, m))
    assert report["F3"] == ["core/run -> pkg/a.py::nope", "core/help -> pkg/nope.py::helper"]


def _mini_repo(tmp_path, name, files):
    r = tmp_path / name
    r.mkdir()

    def git(*args):
        subprocess.run(["git", "-C", str(r)] + list(args), check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    for rel, text in files.items():
        p = r / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "init")
    commit = subprocess.run(
        ["git", "-C", str(r), "rev-parse", "HEAD"], check=True, capture_output=True
    ).stdout.decode().strip()
    return r, commit


def test_main_guard_pseudo_symbol_is_accepted_only_where_the_guard_exists(tmp_path):
    r, commit = _mini_repo(tmp_path, "m", {
        "demo.py": "print('x')\n\nif __name__ == \"__main__\":\n    print('run')\n",
        "lib.py": "X = 1\n",
    })
    m = {
        "domain": "m", "checkout": str(r), "commit": commit,
        "surfaces": {"files": {"include": ["*.py"]}},
        "nodes": [{"id": "demo", "name": "demo", "responsibility": "runs", "parent": None}],
        "leaves": [
            {"id": "demo/run", "name": "run", "entry": "demo.py::__main__", "owns": {"files": ["demo.py"]}},
            {"id": "demo/lib", "name": "lib", "entry": "lib.py::__main__", "owns": {"files": ["lib.py"]}},
        ],
    }
    assert cft.audit(write_manifest(tmp_path, m))["F3"] == ["demo/lib -> lib.py::__main__"]


def test_legacy_line_anchor_is_a_warning(repo, tmp_path):
    r, commit = repo
    m = base_manifest(r, commit)
    m["leaves"][0]["entry"] = "pkg/a.py:12"
    m["leaves"][1]["entry"] = "pkg/b.py:1"
    report = cft.audit(write_manifest(tmp_path, m))
    assert report["F3"] == []
    assert report["F5"] == ["core/run -> pkg/a.py:12", "core/help -> pkg/b.py:1"]
    assert cft.counts(report) == (0, 2)


def test_optional_branch_and_date_are_allowed(repo, tmp_path):
    r, commit = repo
    m = base_manifest(r, commit)
    m["branch"] = "main"
    m["date"] = "2026-09-02"
    assert cft.audit(write_manifest(tmp_path, m))["X1"] == []


def test_symbol_entry_in_non_python_file_uses_identifier_match(tmp_path):
    r, commit = _mini_repo(tmp_path, "js", {"cli.mjs": "export function render(x) { return x }\n"})
    m = {
        "domain": "js", "checkout": str(r), "commit": commit,
        "surfaces": {"files": {"include": ["*.mjs"]}},
        "nodes": [{"id": "cli", "name": "cli", "responsibility": "render", "parent": None}],
        "leaves": [{"id": "cli/render", "name": "render", "entry": "cli.mjs::render",
                    "owns": {"files": ["cli.mjs"]}}],
    }
    assert cft.audit(write_manifest(tmp_path, m))["F3"] == []
    m["leaves"][0]["entry"] = "cli.mjs::renderer"
    assert cft.audit(write_manifest(tmp_path, m))["F3"] == ["cli/render -> cli.mjs::renderer"]


def test_glob_semantics():
    assert cft.glob_match("tools/**", "tools/a/b.py")
    assert cft.glob_match("tools/*.py", "tools/a.py")
    assert not cft.glob_match("tools/*.py", "tools/a/b.py")
    assert cft.glob_match("**/*.py", "x/y/z.py")
    assert cft.glob_match("**/*.py", "z.py")
    assert cft.glob_match("**", "anything/at/all")
    assert cft.glob_match("pkg/a.py", "pkg/a.py")
    assert not cft.glob_match("pkg/a.py", "pkg/a.pyc")


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        cft.main(["--help"])
    assert exc.value.code == 0
    assert "--phase" in capsys.readouterr().out
