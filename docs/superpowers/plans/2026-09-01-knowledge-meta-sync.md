# Knowledge Metadata Synchronization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refresh the reader-facing repository documentation and add a deterministic, tested synchronization gate that keeps `wiki/index.md` counts current while preventing README link and volatile-claim drift.

**Architecture:** A dependency-free Python tool inventories `wiki/**/*.md`, rewrites only numeric payloads inside explicit count markers in `wiki/index.md`, and validates local README links plus narrowly defined volatile phrases. Repository docs and a pull-request workflow make `--write` the local repair path and `--check` the read-only gate; descriptions, featured-page choices, changelog entries, and radar baselines remain human/agent-owned.

**Tech Stack:** Python 3.13 standard library, pytest, Markdown/HTML comment markers, GitHub Actions YAML.

**Spec:** `docs/superpowers/specs/2026-09-01-knowledge-meta-sync-design.md`

## Global Constraints

- Freeze implementation semantics to repository baseline `main @ 97a23022d31a72b0b513497e2bfa577f10e6cf76` plus the approved design commit.
- Do not add third-party dependencies, pre-commit, a metadata manifest, or CI write-back.
- `--write` may change only the integer payload inside a valid `knowledge-meta:count` marker.
- README descriptions and featured-page selection remain human-authored; the tool validates but never generates them.
- Count recursively under the marked directory, include nested `index.md`, and exclude files named `SUPERSEDED.md`.
- Preserve UTF-8 content outside marker payloads byte-for-byte and write atomically only after all markers parse.
- Run tests red first, then green; each task ends in an independently reviewable commit.
- Use the repository constitution and `skills/maintaining-llm-knowledge/SKILL.md`; no formula or diagram edits are expected.

---

### Task 1: Count-marker inventory and synchronization core

**Files:**
- Create: `tools/sync_knowledge_meta.py`
- Create: `tools/test_sync_knowledge_meta.py`

**Interfaces:**
- Produces: `Diagnostic(file: Path, line: int, rule: str, message: str)`.
- Produces: `CountMarker(path: PurePosixPath, value: int, value_start: int, value_end: int, line: int)`.
- Produces: `SyncSummary(marker_count: int, stale_count: int, updated_count: int, readme_link_count: int, diagnostics: tuple[Diagnostic, ...])`.
- Produces: `count_markdown_pages(wiki_root: Path, relative_dir: PurePosixPath) -> int`.
- Produces: `synchronize_count_markers(index_path: Path, wiki_root: Path, *, write: bool) -> tuple[int, int, int, tuple[Diagnostic, ...]]`.
- Produces: `run(repo_root: Path, *, write: bool) -> SyncSummary` and `main(argv: Sequence[str] | None = None) -> int`.

- [ ] **Step 1: Write failing inventory and marker tests**

Add fixtures that create a temporary `wiki/` with nested content, nested `index.md`, and a
`SUPERSEDED.md`. Cover correct counting, stale read-only detection, write behavior, idempotence,
duplicate paths, missing directories, absolute paths, `..`, malformed markers, and the guarantee
that a parse error leaves the file byte-identical. Also cover an `unmanaged-count` error for a
page-count table row whose count cell is a bare integer instead of a marker.

```python
def marker(path: str, value: int) -> str:
    return (
        f"<!-- knowledge-meta:count path={path} -->{value}"
        "<!-- /knowledge-meta:count -->"
    )


def test_write_updates_only_marker_payload_and_is_idempotent(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    index = repo / "wiki" / "index.md"
    index.write_text(
        f"| Domain | [[domain/index]] | {marker('domain', 1)} | active |\n",
        encoding="utf-8",
    )
    before = index.read_text(encoding="utf-8")

    first = synchronize_count_markers(index, repo / "wiki", write=True)
    after = index.read_text(encoding="utf-8")
    second = synchronize_count_markers(index, repo / "wiki", write=True)

    assert first[2] == 1
    assert marker("domain", 3) in after
    assert after.replace("3<!--", "1<!--") == before
    assert second[2] == 0
    assert index.read_text(encoding="utf-8") == after
```

- [ ] **Step 2: Run focused tests and verify the red state**

Run:

```bash
python3 -m pytest tools/test_sync_knowledge_meta.py -q
```

Expected: collection fails because `tools.sync_knowledge_meta` does not exist.

- [ ] **Step 3: Implement the marker parser and inventory core**

Use frozen dataclasses, an explicit marker regex, `PurePosixPath` validation, and one preflight pass
before any replacement.

```python
COUNT_MARKER_RE = re.compile(
    r"<!-- knowledge-meta:count path=(?P<path>[^\s>]+) -->"
    r"(?P<value>\d+)"
    r"<!-- /knowledge-meta:count -->"
)


@dataclass(frozen=True)
class Diagnostic:
    file: Path
    line: int
    rule: str
    message: str


def count_markdown_pages(wiki_root: Path, relative_dir: PurePosixPath) -> int:
    target = wiki_root.joinpath(*relative_dir.parts)
    return sum(
        1
        for path in target.rglob("*.md")
        if path.name != "SUPERSEDED.md" and path.is_file()
    )
```

For atomic writes, create a temporary sibling with `tempfile.NamedTemporaryFile(delete=False,
dir=index_path.parent)`, copy the original mode with `os.chmod`, then `os.replace` only after the
complete replacement string is ready. Clean up the temporary path on exceptions.

- [ ] **Step 4: Add the mutually exclusive CLI skeleton**

```python
parser = argparse.ArgumentParser(description=__doc__)
mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument("--write", action="store_true")
mode.add_argument("--check", action="store_true")
```

At this task boundary `run()` invokes marker synchronization and returns zero README links; Task 2
fills the README validator. Exit non-zero for stale markers in check mode, unmanaged count rows, or
any diagnostic.

- [ ] **Step 5: Run focused tests and verify green**

Run:

```bash
python3 -m pytest tools/test_sync_knowledge_meta.py -q
python3 tools/sync_knowledge_meta.py --check
```

Expected: fixture tests pass; the real-repository command exits non-zero with `unmanaged-count`
diagnostics until Task 3 introduces markers. This known integration red state is not wired into CI
until Task 4.

- [ ] **Step 6: Commit the synchronization core**

```bash
git add tools/sync_knowledge_meta.py tools/test_sync_knowledge_meta.py
git commit -m "feat: add knowledge metadata count synchronization"
```

---

### Task 2: README link and volatile-claim contract

**Files:**
- Modify: `tools/sync_knowledge_meta.py`
- Modify: `tools/test_sync_knowledge_meta.py`

**Interfaces:**
- Consumes: `Diagnostic` and `SyncSummary` from Task 1.
- Produces: `visible_markdown(text: str) -> str`.
- Produces: `validate_readme(readme_path: Path, repo_root: Path) -> tuple[int, tuple[Diagnostic, ...]]`.
- Updates: `run()` combines marker and README diagnostics in stable file/line/rule order.

- [ ] **Step 1: Write failing README contract tests**

Cover valid relative links, fragments, external URLs, pure anchors, missing targets, fenced-code and
inline-code examples, and each prohibited volatile phrase. Include negative controls for Node 22,
seven-day radar windows, dates, and other non-page numbers.

```python
@pytest.mark.parametrize(
    "claim",
    [
        "全库共 **409 篇** Markdown 分析页",
        "全量 409 页首次构建约 30 秒",
        "| 二级目录 | 篇数 | 讲什么 |",
        "全库被引最多的工程页（65 次入链）",
        "按被库内其它页面引用的次数排序挑选",
    ],
)
def test_readme_rejects_volatile_repository_claims(tmp_path: Path, claim: str) -> None:
    repo = make_repo(tmp_path)
    readme = repo / "README.md"
    readme.write_text(claim, encoding="utf-8")
    _, diagnostics = validate_readme(readme, repo)
    assert any(item.rule == "readme-volatile-claim" for item in diagnostics)
```

- [ ] **Step 2: Run the new tests and verify the red state**

Run:

```bash
python3 -m pytest tools/test_sync_knowledge_meta.py -q
```

Expected: failures show `validate_readme` and README diagnostics are absent.

- [ ] **Step 3: Implement README-visible-text and link validation**

Reuse the repository checkers' conservative convention: strip fenced blocks and inline code before
matching live links. Capture line numbers from match offsets. Accept `http:`, `https:`, `mailto:`,
pure `#fragment`, and empty targets; for local paths remove query/fragment, URL-decode them, reject
paths escaping the repository, and require the resolved target to exist.

```python
MARKDOWN_LINK_RE = re.compile(r"\[[^\]\n]+\]\((?P<target>[^)\n]+)\)")
VOLATILE_README_RULES = (
    re.compile(r"全库共\s*\**\d+\s*篇"),
    re.compile(r"全量\s*\d+\s*页"),
    re.compile(r"\|[^\n]*篇数[^\n]*\|"),
    re.compile(r"\d+\s*次入链"),
    re.compile(r"被引最多|按[^\n]*引用[^\n]*次数排序"),
)
```

When removing code, preserve newlines and replace stripped characters with spaces so diagnostic
line numbers remain stable.

- [ ] **Step 4: Integrate README checks into both CLI modes**

`--write` must update valid count markers first, then validate the post-write repository. README
errors remain errors and are never auto-repaired. Print:

```text
markers=<N> stale=<N> updated=<N> readme_links=<N> errors=<N>
```

Then print each diagnostic as `path:line: rule: message`.

- [ ] **Step 5: Run focused tests and verify green**

Run:

```bash
python3 -m pytest tools/test_sync_knowledge_meta.py -q
```

Expected: all synchronization-tool tests pass, including negative controls.

- [ ] **Step 6: Commit the README contract**

```bash
git add tools/sync_knowledge_meta.py tools/test_sync_knowledge_meta.py
git commit -m "feat: validate README metadata contract"
```

---

### Task 3: Apply the one-time README and master-index refresh

**Files:**
- Modify: `README.md`
- Modify: `wiki/index.md`

**Interfaces:**
- Consumes: count-marker and README contracts from Tasks 1–2.
- Produces: a real repository for which `python3 tools/sync_knowledge_meta.py --check` exits zero.

- [ ] **Step 1: Add explicit count markers to every existing count-bearing master-index row**

Wrap only the numeric cell payloads. Use these current expected values:

```text
01_theory/01_models=67
01_theory/01_models/deepseek=22
01_theory/01_models/moonshot_kimi=15
01_theory/01_models/zhipu_glm=12
01_theory/01_models/alibaba_qwen=6
01_theory/01_models/meituan_longcat=3
01_theory/02_pretraining=7
01_theory/03_sft=1
01_theory/04_posttraining=22
01_theory/05_inference=1
01_theory/06_distributed_parallelism=9
02_engineering/01_pytorch=148
02_engineering/01_pytorch/02_compile_stack/04_inductor=36
02_engineering/01_pytorch/03_runtime_graphs=12
02_engineering/01_pytorch/02_compile_stack/05_codegen_backends/mlir=8
02_engineering/02_train_frameworks=68
02_engineering/02_train_frameworks/megatron-lm=28
02_engineering/02_train_frameworks/torchtitan=24
02_engineering/02_train_frameworks/mindformers=3
02_engineering/02_train_frameworks/mindspeed=6
02_engineering/03_infer_frameworks=33
02_engineering/03_infer_frameworks/vllm=25
02_engineering/04_posttrain_frameworks=49
02_engineering/04_posttrain_frameworks/verl=17
02_engineering/04_posttrain_frameworks/slime=21
02_engineering/05_gpu_kernel=17
02_engineering/05_gpu_kernel/triton=9
02_engineering/06_auto_parallel=2
02_engineering/07_training_reliability=5
```

- [ ] **Step 2: Replace dated count notes with one stable generated-count note**

Set `最后更新` to `2026-09-01` and replace both recount-history paragraphs with:

```markdown
> 页面数由 `tools/sync_knowledge_meta.py` 按目录递归统计（含各级 `index.md`，不含
> `SUPERSEDED.md`）；精确数字以本表的自动维护结果为准。
```

- [ ] **Step 3: Refresh README stable content**

Make the approved edits:

- agent-neutral maintainer description naming Claude Code and Codex;
- no repository total or per-domain `篇数` columns;
- no fixed full-build page count;
- “核心文章索引” described as editorially selected, not live inbound ranking;
- no “most-linked” or exact inbound-count statements;
- retired vLLM link replaced with `03_vllm_architecture_overview_analysis.md`, titled “vLLM 架构与请求生命周期”; 
- exact constitution quality commands, including the new metadata check;
- maintenance wording split between `CLAUDE.md` constitution and `skills/` workflows;
- scheduled radar sentence scoped to the maintainer's local machine.

- [ ] **Step 4: Run write mode, inspect the diff, and prove idempotence**

Run:

```bash
python3 tools/sync_knowledge_meta.py --write
git diff -- README.md wiki/index.md
git diff --check
git diff -- README.md wiki/index.md | shasum -a 256 > /tmp/knowledge-meta-before.sha
python3 tools/sync_knowledge_meta.py --write
git diff -- README.md wiki/index.md | shasum -a 256 > /tmp/knowledge-meta-after.sha
cmp /tmp/knowledge-meta-before.sha /tmp/knowledge-meta-after.sha
```

The hash comparison proves the second run added no changes while preserving the intended refresh
diff. Expected tool summary on the second run: `updated=0 errors=0`.

- [ ] **Step 5: Run documentation-focused gates**

Run:

```bash
python3 tools/sync_knowledge_meta.py --check
python3 tools/check_links.py --strict
python3 tools/check_math.py --changed --strict
```

Expected: metadata errors=0; links broken=0, ambiguous=0, bare_index=0, stale_section=0,
orphans=0; math errors=0 and warnings=0.

- [ ] **Step 6: Commit the refreshed documents**

```bash
git add README.md wiki/index.md
git commit -m "docs: refresh repository reader indexes"
```

---

### Task 4: Integrate the synchronization gate into repository workflow and CI

**Files:**
- Modify: `CLAUDE.md`
- Modify: `skills/maintaining-llm-knowledge/SKILL.md`
- Modify: `tools/README.md`
- Modify: `README.md`
- Create: `.github/workflows/knowledge-quality.yml`
- Modify: `tools/test_sync_knowledge_meta.py`

**Interfaces:**
- Consumes: `python tools/sync_knowledge_meta.py --write|--check` from Tasks 1–2.
- Produces: a pull-request and `main` push gate named `Knowledge quality`.
- Produces: repository instructions that consistently assign generated counts to the tool and
  semantic descriptions to agents.

- [ ] **Step 1: Write failing repository-integration contract tests**

Add a test that reads the real repository files and requires the quality gate and workflow hooks:

```python
def test_repository_documents_and_ci_name_sync_commands() -> None:
    repo = Path(__file__).resolve().parents[1]
    assert "sync_knowledge_meta.py --check" in (repo / "CLAUDE.md").read_text()
    skill = (repo / "skills/maintaining-llm-knowledge/SKILL.md").read_text()
    assert "sync_knowledge_meta.py --write" in skill
    assert "sync_knowledge_meta.py --check" in skill
    workflow = (repo / ".github/workflows/knowledge-quality.yml").read_text()
    assert "pull_request:" in workflow
    assert "sync_knowledge_meta.py --check" in workflow
```

- [ ] **Step 2: Run the integration test and verify the red state**

Run:

```bash
python3 -m pytest tools/test_sync_knowledge_meta.py::test_repository_documents_and_ci_name_sync_commands -q
```

Expected: failure because the CI workflow and documentation hooks do not yet exist.

- [ ] **Step 3: Update constitution, maintenance skill, and tool documentation**

Add the metadata check to `CLAUDE.md` before the link gate. Add a final metadata synchronization
step to both the ingest and maintenance workflows in the skill. Document `--write`, `--check`,
marker ownership, and README check-only behavior in `tools/README.md`. Keep the commands exactly
consistent across files.

- [ ] **Step 4: Add the read-only GitHub Actions workflow**

Create:

```yaml
name: Knowledge quality

on:
  pull_request:
  push:
    branches:
      - main

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
        with:
          fetch-depth: 0
      - uses: actions/setup-python@v6
        with:
          python-version: "3.13"
      - name: Check generated knowledge metadata
        run: python tools/sync_knowledge_meta.py --check
      - name: Check wiki links
        run: python tools/check_links.py --strict
      - name: Check wiki math
        run: python tools/check_math.py wiki --strict
      - name: Test maintenance tools
        run: python -m pytest tools/test_sync_knowledge_meta.py tools/test_check_links.py tools/test_check_math.py
```

- [ ] **Step 5: Run focused and integration tests**

Run:

```bash
python3 -m pytest tools/test_sync_knowledge_meta.py tools/test_check_links.py tools/test_check_math.py -q
python3 tools/sync_knowledge_meta.py --check
```

Expected: all selected tests pass and metadata errors=0.

- [ ] **Step 6: Commit workflow integration**

```bash
git add CLAUDE.md skills/maintaining-llm-knowledge/SKILL.md tools/README.md README.md \
  .github/workflows/knowledge-quality.yml tools/test_sync_knowledge_meta.py
git commit -m "ci: enforce knowledge metadata freshness"
```

---

### Task 5: Full verification and delivery audit

**Files:**
- Verify only; modify a preceding task's owned file if a gate exposes a defect.

**Interfaces:**
- Consumes: all artifacts from Tasks 1–4.
- Produces: fresh evidence for every acceptance criterion in the approved design.

- [ ] **Step 1: Prove deterministic metadata state**

Run:

```bash
python3 tools/sync_knowledge_meta.py --check
python3 tools/sync_knowledge_meta.py --write
git status --short
```

Expected: check exits zero, write reports `updated=0`, and write creates no new diff.

- [ ] **Step 2: Run all constitution gates**

Run:

```bash
python3 tools/check_links.py --strict
python3 tools/check_math.py --changed --strict
python3 -m pytest tools/
npm run docs:test
```

Expected: all commands exit zero. Record exact pass counts and any environment-specific skip count.

- [ ] **Step 3: Audit acceptance criteria directly**

Run targeted searches:

```bash
rg -n "全库共|全量 [0-9]+ 页|篇数|次入链|被引最多|引用.*次数排序" README.md
python3 -c 'from pathlib import Path; from tools.sync_knowledge_meta import validate_readme; print(validate_readme(Path("README.md"), Path.cwd()))'
git diff --check HEAD~4..HEAD
```

Expected: the volatile-pattern search has no matches, README validation has no diagnostics, and diff
check is clean.

- [ ] **Step 4: Inspect final history and worktree**

Run:

```bash
git log --oneline --decorate -6
git status --short --branch
```

Expected: design/plan plus four implementation commits are visible; the worktree is clean and the
branch is ahead of `origin/main` only by the intended commits.
