import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = REPO_ROOT / "skills/source-faithful-analysis"
CORE = SKILL_ROOT / "SKILL.md"
REFERENCES = SKILL_ROOT / "references"
SOURCE_FIDELITY = REFERENCES / "source-fidelity.md"
CODEBASE = REFERENCES / "codebase.md"
PAPER = REFERENCES / "paper.md"
GENERAL = REFERENCES / "general.md"
MECHANISM = REFERENCES / "document-types/mechanism-analysis.md"
FEATURE = REFERENCES / "document-types/feature-analysis.md"
FEATURE_REVIEW = REFERENCES / "reviews/feature-analysis.md"
PARALLEL = REFERENCES / "parallel-agent-contract.md"
PAGE_REVIEW = REFERENCES / "page-review-rubric.md"
EVALS = SKILL_ROOT / "evals/evals.json"
LIVE_EVALS = SKILL_ROOT / "evals/scenarios.md"
CONSTITUTION = REPO_ROOT / "CLAUDE.md"
FEATURE_TREE = REPO_ROOT / "skills/feature-tree-analysis/SKILL.md"
FEATURE_TEMPLATE = REPO_ROOT / "skills/feature-tree-analysis/references/feature-point-template.md"
FEATURE_SPEC_WRITER = REPO_ROOT / "skills/feature-tree-analysis/references/spec-writer-contract.md"


def _text(path):
    return path.read_text(encoding="utf-8")


def _squash(text):
    return " ".join(text.split())


def test_router_is_lean_and_selects_source_profile_and_review_separately():
    raw = _text(CORE)
    text = _squash(raw).lower()

    assert len(raw.splitlines()) <= 95
    for heading in ("## Source fidelity", "## Canonical analysis contract", "## Workflow"):
        assert heading not in raw
    for dimension in ("source pack", "document profile", "review profile"):
        assert dimension in text


def test_router_uses_exact_resolvable_paths_for_all_primary_profiles():
    raw = _text(CORE)
    expected = (
        "references/source-fidelity.md",
        "references/codebase.md",
        "references/paper.md",
        "references/general.md",
        "references/document-types/mechanism-analysis.md",
        "references/document-types/feature-analysis.md",
        "references/document-types/software-architecture.md",
        "references/reviews/feature-analysis.md",
        "references/reviews/software-architecture.md",
    )
    for relative in expected:
        assert f"`{relative}`" in raw
        assert (SKILL_ROOT / relative).is_file(), f"unresolved routed file: {relative}"


def test_shared_kernel_owns_source_fidelity_and_boundary_drift_once():
    router = _text(CORE)
    kernel = _squash(_text(SOURCE_FIDELITY)).lower()

    for phrase in (
        "verified source evidence",
        "frozen baseline",
        "source fact",
        "analyst inference",
        "conflicts stay visible",
        "stable code anchor",
        "approved boundary",
        "owning output contract",
        "owning approval workflow",
        "planning-codebase-analysis",
        "feature-tree-analysis",
    ):
        assert phrase in kernel, f"shared kernel lost {phrase!r}"
    assert "verified source evidence" not in router.lower()
    assert "frozen baseline" not in router.lower()


def test_mechanism_profile_owns_the_causal_page_contract():
    mechanism = _squash(_text(MECHANISM)).lower()
    for phrase in (
        "central thesis",
        "background / problem",
        "obvious alternative",
        "mechanism and evidence",
        "constraints and failure boundary",
        "state model",
        "execution trace",
        "hop-walk",
    ):
        assert phrase in mechanism, f"mechanism profile lost {phrase!r}"


def test_code_pages_default_to_stable_symbol_anchors_not_line_number_citations():
    kernel = _squash(_text(SOURCE_FIDELITY)).lower()
    codebase = _squash(_text(CODEBASE)).lower()
    parallel = _squash(_text(PARALLEL)).lower()
    review = _squash(_text(PAGE_REVIEW)).lower()

    assert "stable code anchor" in kernel
    assert "path + qualified symbol" in codebase
    assert "line numbers are optional" in codebase
    assert "compact source-reading route" in codebase
    assert "do not attach `file:line` to every claim or semantic hop" in parallel
    assert "line-number citations are not required" in review


def test_legacy_line_checker_is_conditional_and_feature_specs_use_symbol_anchors():
    constitution = _text(CONSTITUTION)
    always_on_gates = constitution.split("```bash", 1)[1].split("```", 1)[0]
    feature_tree = _squash(_text(FEATURE_TREE)).lower()
    template = _squash(_text(FEATURE_TEMPLATE)).lower()

    assert "python tools/check_locators.py" not in always_on_gates
    assert "conditional legacy gate" in constitution
    assert "stable source anchors" in feature_tree
    assert "path::qualified.symbol" in template


def test_codebase_pack_owns_traceable_execution_evidence_not_page_shape():
    pack = _squash(_text(CODEBASE)).lower()
    required = (
        "actual call path",
        "caller → callee",
        "crossing object",
        "pre-state",
        "post-state",
        "state owner",
        "local/remote",
        "sync/async",
        "blocking/non-blocking",
        "completion signal",
        "externally visible",
        "partial side effects",
        "rollback",
        "collapse pure forwarding helpers",
    )
    for phrase in required:
        assert phrase in pack, f"execution evidence contract lost {phrase!r}"
    for heading in ("## Code mechanism contract", "## Architecture overview contract"):
        assert heading not in _text(CODEBASE)
    assert "## Evidence roles" not in _text(CODEBASE)
    assert "conflicts stay visible" not in _text(CODEBASE).lower()


def test_codebase_depth_profiles_are_conditional_evidence_rules():
    pack = _squash(_text(CODEBASE)).lower()
    for phrase in (
        "multiple live implementations",
        "data representation",
        "asynchronous, concurrent",
        "runtime verification",
        "only when the trigger is present",
    ):
        assert phrase in pack


def test_concrete_feature_profile_preserves_user_added_contract_and_delegates_trace():
    profile = _squash(_text(FEATURE)).lower()
    codebase = _squash(_text(CODEBASE)).lower()

    for phrase in (
        "primitive → system",
        "smallest meaningful example",
        "class / ownership view",
        "ascii caller tree",
        "aggregate cost",
        "operating envelope",
    ):
        assert phrase in profile

    delegated_trace_terms = (
        "local/remote",
        "sync/async",
        "blocking/non-blocking",
        "completion signal",
        "partial side effects",
        "retry/idempotency",
        "rollback",
        "path::qualified.symbol",
        "transitive/elided",
    )
    for term in delegated_trace_terms:
        assert term in codebase, f"codebase pack no longer owns {term!r}"
        assert term not in profile, f"feature profile duplicates codebase-owned {term!r}"


def test_feature_profile_keeps_domain_depth_conditional_and_review_separate():
    raw = _text(FEATURE)
    universal, conditional = raw.lower().split("## conditional depth", 1)

    assert "tensor parallel" not in universal
    assert "sequence parallel" not in universal
    assert "asynchronous or concurrent" not in conditional
    assert "retry/idempotency" not in conditional
    assert "## feature-page completion review" not in raw.lower()
    review = _text(FEATURE_REVIEW)
    assert len([line for line in review.splitlines() if line.startswith("-")]) == 11


def test_variant_set_needs_an_enumeration_basis_and_sibling_axis_check():
    """A page organized around one controlling field inherits that field's blind spot:
    the four cp_comm_type values were replayed while linear_cp_mode and MambaContextParallel
    — sibling axes for other layer types — went unmentioned."""
    feature = _squash(_text(FEATURE)).lower()
    for phrase in ("enumeration basis", "sibling selection axes", "selection sites"):
        assert phrase in feature, f"feature profile lost variant enumeration rule {phrase!r}"
    # Variants must carry motivation and limit, not only cost accounting.
    assert "pressure it answers and the resource that caps it" in feature

    review = _squash(_text(FEATURE_REVIEW)).lower()
    assert "enumeration basis" in review
    assert "sibling selection axis" in review


def test_dependency_boundary_is_an_evidence_boundary():
    """Third-party internals may be reported as published contract, never narrated as read."""
    kernel = _squash(_text(SOURCE_FIDELITY)).lower()
    assert "dependency boundary" in kernel
    for phrase in ("what the analyzed source", "published contract", "never narrated as if read"):
        assert phrase in kernel, f"shared kernel lost dependency-boundary rule {phrase!r}"

    review = _squash(_text(FEATURE_REVIEW)).lower()
    assert "third-party dependency is narrated as verified execution" in review


def test_restructuring_an_existing_unit_must_conserve_what_it_owned():
    """Six coverage-owned config names once left the wiki entirely during a rewrite,
    and the host gate could not see it. Conservation is a method rule, not a gate rule."""
    kernel = _squash(_text(SOURCE_FIDELITY)).lower()
    assert "restructuring an existing unit is a conservation problem" in kernel
    for phrase in ("kept, corrected", "rehomed", "silent loss"):
        assert phrase in kernel, f"shared kernel lost conservation rule {phrase!r}"

    review = _squash(_text(FEATURE_REVIEW)).lower()
    assert "nor rehomed to a named owner" in review


def test_algorithmic_units_require_a_replayable_principle_figure_without_forcing_plain_crud():
    profiles = {
        "feature": _squash(_text(FEATURE)).lower(),
        "mechanism": _squash(_text(MECHANISM)).lower(),
    }

    for name, profile in profiles.items():
        for phrase in (
            "algorithmic implementation",
            "at least one principle figure",
            "drawing-wiki-figures",
            "partitioning",
            "routing",
            "packing",
            "scheduling",
            "ordinary crud",
        ):
            assert phrase in profile, f"{name} profile lost algorithmic visual contract {phrase!r}"

    for name, profile in profiles.items():
        for phrase in (
            "local compute",
            "data/state/ownership movement",
            "communication",
            "synchronization",
            "forward/backward differences",
            "incremental cost",
        ):
            assert phrase in profile, f"{name} profile lost variant closure {phrase!r}"

    assert "outside figure-medium rules" in profiles["feature"]


def test_algorithmic_review_gate_rejects_missing_or_non_replayable_figures():
    base = _squash(_text(PAGE_REVIEW)).lower()
    feature = _squash(_text(FEATURE_REVIEW)).lower()

    for phrase in (
        "algorithm replay",
        "figure-trigger",
        "rendered figure",
        "smallest example",
        "stranger-reader line",
        "class diagram",
        "caller tree",
        "ownership inventory",
        "code excerpt",
        "prose",
        "table",
        "no trigger",
        "selected document profile",
        "local compute",
        "data/state/ownership movement",
        "communication",
        "synchronization",
        "forward/backward differences",
        "incremental cost",
    ):
        assert phrase in base, f"base review lost algorithmic visual gate {phrase!r}"

    assert "## the five checks" in base

    for phrase in (
        "same concrete example",
        "distinct live variant",
        "data plane",
        "local compute",
        "communication",
        "incremental cost",
    ):
        assert phrase in feature, f"feature review lost variant replay gate {phrase!r}"


def test_source_analysis_evals_encode_source_profile_and_depth_as_separate_axes():
    payload = json.loads(_text(EVALS))
    assert payload["skill_name"] == "source-faithful-analysis"
    by_id = {item["id"]: item for item in payload["evals"]}
    assert set(by_id) == {1, 2, 3, 4, 5, 6, 7, 8}
    assert all(item["prompt"] and item["expected_output"] for item in payload["evals"])

    expected_routing = {
        1: ("generic-code-mechanism", {"codebase"}, "mechanism-analysis", set(), set()),
        2: ("async-distributed-trace", {"codebase"}, "mechanism-analysis", {"async-concurrent"}, set()),
        3: ("data-lifecycle", {"codebase"}, "mechanism-analysis", {"data-representation"}, set()),
        4: ("ordinary-software-feature", {"codebase"}, "feature-analysis", set(), {"training", "parallel-distributed", "companion-without-evidence"}),
        5: ("parallel-training-feature", {"codebase"}, "feature-analysis", {"training", "parallel-distributed", "algorithmic-visual"}, {"companion-without-evidence"}),
        6: ("conditional-depth-routing", {"codebase"}, "feature-analysis", set(), {"fixed-cost-dimensions", "empty-conditional-sections", "companion-without-evidence"}),
        7: ("algorithmic-implementation-visual-gate", {"codebase"}, "feature-analysis", {"algorithmic-visual", "parallel-distributed"}, {"prose-only-algorithm", "caller-tree-as-principle-figure"}),
        8: ("algorithmic-nondistributed-visual-gate", {"codebase"}, "feature-analysis", {"algorithmic-visual"}, {"parallel-distributed", "prose-only-algorithm"}),
    }
    for scenario_id, (kind, packs, profile, depth, forbidden) in expected_routing.items():
        scenario = by_id[scenario_id]
        assert scenario["kind"] == kind
        assert set(scenario["source_packs"]) == packs
        assert scenario["document_profile"] == profile
        assert set(scenario["conditional_depth"]) == depth
        assert set(scenario["forbidden_depth"]) == forbidden
        assert "active_profiles" not in scenario
        assert "forbidden_profiles" not in scenario

    assert "原理图" in by_id[5]["expected_output"]
    assert "原理图" in by_id[8]["expected_output"]


def test_feature_profile_has_reproducible_live_behavior_scenarios():
    scenarios = _text(LIVE_EVALS)
    headings = (
        "## S1 — Parallel training feature",
        "## S2 — Ordinary software feature",
        "## S3 — Asynchronous stateful feature",
        "## S4 — Algorithmic implementation visual gate",
        "## Run log",
    )
    offsets = [scenarios.index(heading) for heading in headings]
    assert offsets == sorted(offsets)
    for index in range(4):
        section = scenarios[offsets[index] : offsets[index + 1]]
        assert "Prompt:" in section
        assert "Green result:" in section

    run_log = scenarios[offsets[-1] :]
    for run in (
        "S1 baseline",
        "S2 baseline",
        "S3 baseline",
        "S4 baseline",
        "S1 green",
        "S2 green",
        "S3 green",
        "S4 green",
    ):
        assert run in run_log

    s4 = scenarios[offsets[3] : offsets[4]].lower()
    for family in ("pp", "tp", "cp", "ep", "packed-dataset"):
        assert family in s4
    assert "s1 visual rerun" in run_log.lower()


def test_source_type_packs_instantiate_evidence_without_redefining_page_profiles():
    paper = _text(PAPER)
    general = _text(GENERAL)

    assert len(paper.splitlines()) <= 115
    assert len(general.splitlines()) <= 55
    for text in (paper, general):
        assert "## Essence checklist" not in text
        assert "## Doc structure" not in text
        assert "## Canonical analysis contract" not in text

    for phrase in ("arXiv id + VERSION + date", "released artifact", "机制 ↔ 源码", "✅ 官方已发布", "❌ 无任何实现"):
        assert phrase in paper
    for phrase in ("Spec / RFC / standard / contract", "Dataset / schema / table", "Running system / incident / logs / metrics", "Business / financial / market report"):
        assert phrase in general


def test_parallel_contract_obeys_selected_profile_and_delegates_figure_rules():
    parallel = _text(PARALLEL)
    flat = _squash(parallel).lower()
    assert len(parallel.splitlines()) <= 95
    assert "required document profile" in flat
    assert "selected document profile" in flat
    assert "SVG→PNG" not in parallel
    assert "<div class=\"diagram\"" not in parallel
    assert "active house figure skill" in parallel
    assert "completion or visibility" in parallel


def test_feature_tree_remains_an_independent_workflow_but_imports_shared_fidelity():
    raw = _text(FEATURE_TREE)
    flat = _squash(raw).lower()
    shared_reference = "../source-faithful-analysis/references/source-fidelity.md"

    assert f"`{shared_reference}`" in raw
    assert (FEATURE_TREE.parent / shared_reference).resolve().is_file()
    for phrase in ("manifest", "checker", "approval", "wave", "baseline bump"):
        assert phrase in flat
    assert "document-types/feature-analysis.md" not in raw
    assert "codebase pack" not in _text(FEATURE_TEMPLATE).lower()


def test_nested_profile_and_feature_tree_references_resolve_from_their_files():
    references = {
        FEATURE: ("../source-fidelity.md", "../codebase.md", "../reviews/feature-analysis.md"),
        FEATURE_REVIEW: ("../page-review-rubric.md", "../codebase.md"),
        LIVE_EVALS: (
            "../SKILL.md",
            "../references/codebase.md",
            "../references/document-types/feature-analysis.md",
        ),
        PAGE_REVIEW: ("source-fidelity.md",),
        FEATURE_TEMPLATE: ("../../source-faithful-analysis/references/source-fidelity.md",),
        FEATURE_SPEC_WRITER: (
            "../../source-faithful-analysis/references/source-fidelity.md",
            "../../source-faithful-analysis/references/parallel-agent-contract.md",
        ),
    }
    for owner, relative_paths in references.items():
        raw = _text(owner)
        for relative in relative_paths:
            assert f"`{relative}`" in raw, f"{owner} does not name {relative}"
            assert (owner.parent / relative).resolve().is_file(), f"{owner}: unresolved {relative}"


def test_progressive_disclosure_routes_fit_their_budgets():
    routes = {
        "code feature": ((CORE, SOURCE_FIDELITY, CODEBASE, FEATURE), 500),
        "code mechanism": ((CORE, SOURCE_FIDELITY, CODEBASE, MECHANISM), 500),
        "paper mechanism": ((CORE, SOURCE_FIDELITY, PAPER, MECHANISM), 460),
        "general artifact": ((CORE, SOURCE_FIDELITY, GENERAL, MECHANISM), 400),
        "delegated analysis": ((CORE, SOURCE_FIDELITY, PARALLEL), 360),
    }
    for route, (files, limit) in routes.items():
        actual = sum(len(_text(path).splitlines()) for path in files)
        assert actual <= limit, f"{route} route is {actual} lines (limit {limit})"
