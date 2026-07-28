from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from docs.audits.pytorch_graph_series.tools.audit_graph_docs import (
    audit_manifest,
    build_ledger_rows,
)


CONTENT_CLASSES = {
    "assertion",
    "question",
    "instruction",
    "provenance",
    "limitation",
    "nonassertive",
}
NONASSERTIVE_CLASSES = {"question", "instruction", "nonassertive"}
EVIDENCE_CLASSES = {"S", "R", "I", "M", "B"}
RUNTIME_BASELINE_FIELDS = {
    "python",
    "torch",
    "torch_git",
    "platform",
    "cuda_available",
}
RUNTIME_PRODUCER_ROLES = {
    "producer_script",
    "producer_script_sha256",
    "producer_command",
    "producer_exit_code",
}
REQUIRED_FIELDS = {
    "schema_version",
    "claim_id",
    "page",
    "start_line",
    "end_line",
    "text_sha256",
    "content_class",
    "evidence_class",
    "status",
    "evidence",
    "parent_claim_ids",
    "runtime_evidence",
    "presentation",
    "notes",
}


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _derived_claim_id(
    parent_id: object, kind: str, ordinal: int, text: str
) -> str:
    payload = "\0".join(
        ["course-claim-v2", str(parent_id), kind, str(ordinal), text]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _error(code: str, claim_id: object, detail: str) -> dict[str, object]:
    return {"code": code, "claim_id": str(claim_id), "detail": detail}


def _safe_repo_path(root: Path, raw_path: object) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path:
        return None
    candidate = (root / Path(raw_path.replace("/", str(Path("/"))))).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def validate_source_checkout(
    source_root: Path, pinned_baseline: str
) -> list[dict[str, object]]:
    """Require evidence locators to be read from the exact clean checkout."""
    claim_id = "<source-checkout>"
    try:
        head = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        return [
            _error(
                "source_checkout_unreadable",
                claim_id,
                f"cannot inspect source checkout: {error}",
            )
        ]

    errors: list[dict[str, object]] = []
    if head != pinned_baseline:
        errors.append(
            _error(
                "source_checkout_commit_mismatch",
                claim_id,
                f"expected {pinned_baseline}, found {head}",
            )
        )
    if status.strip():
        errors.append(
            _error(
                "source_checkout_dirty",
                claim_id,
                "source checkout has tracked or untracked changes",
            )
        )
    return errors


def load_course_claim_decisions(directory: Path) -> list[dict[str, object]]:
    decisions: list[dict[str, object]] = []
    for path in sorted(directory.glob("*.jsonl")):
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: decision must be an object")
            decisions.append(value)
    return decisions


def reconcile_course_claim_decisions(
    claim_rows: Sequence[Mapping[str, object]],
    decisions: Sequence[Mapping[str, object]],
) -> tuple[
    list[dict[str, object]],
    list[str],
    list[str],
    list[dict[str, object]],
]:
    """Re-key unchanged reviewed decisions after page line/hash movement.

    Exact current IDs win. Otherwise a prior decision may move only when its
    page plus text hash identifies exactly one current claim. Changed text,
    ambiguous duplicate text, and duplicate target selection remain explicit
    review work; this helper never assigns truth to a new or edited claim.
    """

    claims_by_id = {str(row["id"]): row for row in claim_rows}
    claims_by_text: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(
        list
    )
    for row in claim_rows:
        key = (str(row["page"]), _text_sha256(str(row.get("text", ""))))
        claims_by_text[key].append(row)

    selected: list[tuple[Mapping[str, object], Mapping[str, object]]] = []
    old_to_new: dict[str, str] = {}
    claimed_targets: dict[str, str] = {}
    stale_ids: list[str] = []
    errors: list[dict[str, object]] = []
    for decision in decisions:
        old_id = str(decision.get("claim_id", ""))
        expected_hash = str(decision.get("text_sha256", ""))
        claim = claims_by_id.get(old_id)
        if claim is not None and expected_hash == _text_sha256(
            str(claim.get("text", ""))
        ):
            candidates = [claim]
        else:
            candidates = claims_by_text.get(
                (str(decision.get("page", "")), expected_hash), []
            )
        if len(candidates) != 1:
            stale_ids.append(old_id)
            code = (
                "reconcile_ambiguous_text"
                if len(candidates) > 1
                else "reconcile_unmatched_decision"
            )
            errors.append(
                _error(
                    code,
                    old_id,
                    "decision text does not select exactly one current claim",
                )
            )
            continue
        target = candidates[0]
        new_id = str(target["id"])
        if new_id in claimed_targets:
            stale_ids.append(old_id)
            errors.append(
                _error(
                    "reconcile_duplicate_target",
                    old_id,
                    f"also selected by {claimed_targets[new_id]}",
                )
            )
            continue
        claimed_targets[new_id] = old_id
        old_to_new[old_id] = new_id
        selected.append((decision, target))

    migrated: list[dict[str, object]] = []
    for decision, claim in selected:
        value = copy.deepcopy(dict(decision))
        value.update(
            claim_id=str(claim["id"]),
            page=str(claim["page"]),
            start_line=int(claim["source_start_line"]),
            end_line=int(claim["source_end_line"]),
            text_sha256=_text_sha256(str(claim.get("text", ""))),
            unit_kind=str(claim.get("kind", "")),
        )
        parents = value.get("parent_claim_ids")
        if isinstance(parents, list):
            value["parent_claim_ids"] = [
                old_to_new.get(str(parent), str(parent)) for parent in parents
            ]
        evidence = value.get("evidence")
        if isinstance(evidence, list):
            for item in evidence:
                if isinstance(item, dict) and "parent_claim_id" in item:
                    parent = str(item["parent_claim_id"])
                    item["parent_claim_id"] = old_to_new.get(parent, parent)
        migrated.append(value)

    missing_ids = sorted(set(claims_by_id).difference(claimed_targets))
    return migrated, sorted(stale_ids), missing_ids, errors


def build_course_claim_rows(
    repo_root: Path, source_root: Path, manifest_path: Path
) -> list[dict[str, object]]:
    records = audit_manifest(repo_root, source_root, manifest_path)
    rows: list[dict[str, object]] = []
    for row in build_ledger_rows(records):
        kind = str(row["kind"])
        if kind == "claim_candidate":
            rows.append(row)
            continue
        if kind == "markdown_table":
            payload = row.get("payload")
            if not isinstance(payload, Mapping):
                continue
            data_rows = payload.get("data_rows")
            if not isinstance(data_rows, list):
                continue
            page = repo_root / str(row["page"])
            page_lines = page.read_text(encoding="utf-8").splitlines()
            first_data_line = int(row["source_start_line"]) + 2
            for index, cells in enumerate(data_rows):
                line_number = first_data_line + index
                if line_number < 1 or line_number > len(page_lines):
                    raise ValueError(
                        f"table row outside page: {row['page']}:{line_number}"
                    )
                text = page_lines[line_number - 1]
                expanded = dict(row)
                expanded.update(
                    id=_derived_claim_id(
                        row["id"], "table_row_claim", index, text
                    ),
                    kind="table_row_claim",
                    source_start_line=line_number,
                    source_end_line=line_number,
                    source_line=line_number,
                    text=text,
                    payload={
                        "parent_unit_id": row["id"],
                        "header_cells": payload.get("header_cells", []),
                        "cells": cells,
                        "row_index": index,
                    },
                )
                rows.append(expanded)
            continue
        if kind.startswith("code_"):
            payload = row.get("payload")
            if not isinstance(payload, Mapping):
                continue
            content = str(payload.get("content", ""))
            if not content.strip():
                continue
            language = str(payload.get("language", ""))
            text = f"{language}\n{content}"
            expanded = dict(row)
            expanded.update(
                id=_derived_claim_id(row["id"], "code_claim", 0, text),
                kind="code_claim",
                text=text,
                payload={**payload, "parent_unit_id": row["id"]},
            )
            rows.append(expanded)
    return rows


def _validate_source_evidence(
    decision: Mapping[str, object],
    source_root: Path,
    pinned_baseline: str,
) -> list[dict[str, object]]:
    claim_id = decision.get("claim_id", "")
    evidence = decision.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        return [_error("source_evidence_missing", claim_id, "S requires evidence[]")]
    errors: list[dict[str, object]] = []
    for index, item in enumerate(evidence):
        if not isinstance(item, Mapping):
            errors.append(
                _error(
                    "source_evidence_invalid",
                    claim_id,
                    f"evidence[{index}] must be an object",
                )
            )
            continue
        if item.get("baseline") != pinned_baseline:
            errors.append(
                _error(
                    "source_baseline_mismatch",
                    claim_id,
                    f"evidence[{index}] baseline is not the pinned commit",
                )
            )
        candidate = _safe_repo_path(source_root, item.get("path"))
        if candidate is None or not candidate.is_file():
            errors.append(
                _error(
                    "source_path_missing",
                    claim_id,
                    f"evidence[{index}] path does not resolve inside source root",
                )
            )
            continue
        try:
            start_line = int(item["start_line"])
            end_line = int(item["end_line"])
        except (KeyError, TypeError, ValueError):
            errors.append(
                _error(
                    "source_line_invalid",
                    claim_id,
                    f"evidence[{index}] requires integer start/end lines",
                )
            )
            continue
        line_count = len(
            candidate.read_text(encoding="utf-8", errors="replace").splitlines()
        )
        if start_line < 1 or end_line < start_line or end_line > line_count:
            errors.append(
                _error(
                    "source_line_out_of_bounds",
                    claim_id,
                    f"evidence[{index}] lines {start_line}-{end_line}/{line_count}",
                )
            )
        if end_line - start_line + 1 > 30:
            errors.append(
                _error(
                    "source_range_too_wide",
                    claim_id,
                    f"evidence[{index}] spans more than 30 source lines",
                )
            )
        if len(str(item.get("supports", "")).strip()) < 40:
            errors.append(
                _error(
                    "source_support_explanation_missing",
                    claim_id,
                    f"evidence[{index}] lacks a claim-specific explanation",
                )
            )
    return errors


def _validate_runtime_evidence(
    decision: Mapping[str, object], repo_root: Path, *, required: bool
) -> list[dict[str, object]]:
    claim_id = decision.get("claim_id", "")
    runtime = decision.get("runtime_evidence")
    if not isinstance(runtime, list):
        return [
            _error(
                "runtime_evidence_invalid",
                claim_id,
                "runtime_evidence must be an array",
            )
        ]
    if required and not runtime:
        return [
            _error(
                "runtime_evidence_missing",
                claim_id,
                "R/M evidence requires at least one runtime record",
            )
        ]
    errors: list[dict[str, object]] = []
    for index, item in enumerate(runtime):
        if not isinstance(item, Mapping):
            errors.append(
                _error(
                    "runtime_evidence_invalid",
                    claim_id,
                    f"runtime_evidence[{index}] must be an object",
                )
            )
            continue
        raw_script = item.get("script")
        script = _safe_repo_path(repo_root, raw_script)
        if script is None or not script.is_file():
            errors.append(
                _error(
                    "runtime_script_missing",
                    claim_id,
                    f"runtime_evidence[{index}] script is missing",
                )
            )
        expected_script_hash = str(item.get("script_sha256", ""))
        if not expected_script_hash:
            errors.append(
                _error(
                    "runtime_script_hash_missing",
                    claim_id,
                    f"runtime_evidence[{index}] requires script_sha256",
                )
            )
        elif script is not None and script.is_file():
            observed_script_hash = hashlib.sha256(script.read_bytes()).hexdigest()
            if expected_script_hash != observed_script_hash:
                errors.append(
                    _error(
                        "runtime_script_hash_mismatch",
                        claim_id,
                        f"runtime_evidence[{index}] script hash is stale",
                    )
                )
        command = item.get("command")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(token, str) and token for token in command)
        ):
            errors.append(
                _error(
                    "runtime_command_missing",
                    claim_id,
                    f"runtime_evidence[{index}] requires a tokenized command",
                )
            )
        elif isinstance(raw_script, str):
            normalized_script = raw_script.replace("\\", "/").casefold()
            normalized_command = [
                token.replace("\\", "/").casefold() for token in command
            ]
            invokes_script = any(
                token == normalized_script
                or token.endswith(f"/{normalized_script}")
                for token in normalized_command
            )
            if not invokes_script:
                errors.append(
                    _error(
                        "runtime_command_script_mismatch",
                        claim_id,
                        f"runtime_evidence[{index}] command does not invoke script",
                    )
                )
        artifacts = item.get("artifacts")
        declared_artifacts: set[str] = set()
        if not isinstance(artifacts, list) or not artifacts:
            errors.append(
                _error(
                    "runtime_artifact_missing",
                    claim_id,
                    f"runtime_evidence[{index}] requires artifacts[]",
                )
            )
        else:
            for artifact in artifacts:
                if isinstance(artifact, str):
                    declared_artifacts.add(artifact)
                candidate = _safe_repo_path(repo_root, artifact)
                if candidate is None or not candidate.exists():
                    errors.append(
                        _error(
                            "runtime_artifact_missing",
                            claim_id,
                            f"runtime artifact is missing: {artifact!r}",
                        )
                    )
                elif candidate.suffix.lower() == ".md":
                    errors.append(
                        _error(
                            "runtime_artifact_not_generated",
                            claim_id,
                            f"Markdown documentation is not runtime output: {artifact!r}",
                        )
                    )
        checks = item.get("artifact_checks")
        if not isinstance(checks, list) or not checks:
            errors.append(
                _error(
                    "runtime_artifact_check_missing",
                    claim_id,
                    f"runtime_evidence[{index}] requires artifact_checks[]",
                )
            )
        else:
            producer_roles: Counter[str] = Counter()
            for check_index, check in enumerate(checks):
                if not isinstance(check, Mapping):
                    errors.append(
                        _error(
                            "runtime_artifact_check_invalid",
                            claim_id,
                            f"artifact_checks[{check_index}] must be an object",
                        )
                    )
                    continue
                role = check.get("role")
                if role in RUNTIME_PRODUCER_ROLES:
                    producer_roles[str(role)] += 1
                raw_path = check.get("path")
                if raw_path not in declared_artifacts:
                    errors.append(
                        _error(
                            "runtime_artifact_check_undeclared",
                            claim_id,
                            f"checked artifact is not listed in artifacts[]: {raw_path!r}",
                        )
                    )
                candidate = _safe_repo_path(repo_root, raw_path)
                if candidate is None or not candidate.is_file():
                    continue
                expected_hash = str(check.get("sha256", ""))
                observed_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
                if expected_hash != observed_hash:
                    errors.append(
                        _error(
                            "runtime_artifact_hash_mismatch",
                            claim_id,
                            f"artifact_checks[{check_index}] hash is stale",
                        )
                    )
                if candidate.suffix.lower() != ".json":
                    errors.append(
                        _error(
                            "runtime_artifact_check_not_json",
                            claim_id,
                            f"checked artifact must be JSON: {raw_path!r}",
                        )
                    )
                    continue
                selector = check.get("selector")
                if not isinstance(selector, str) or not selector.startswith("/"):
                    errors.append(
                        _error(
                            "runtime_artifact_selector_invalid",
                            claim_id,
                            f"artifact_checks[{check_index}] requires a JSON pointer",
                        )
                    )
                    continue
                try:
                    selected: object = json.loads(
                        candidate.read_text(encoding="utf-8")
                    )
                    for token in selector.split("/")[1:]:
                        token = token.replace("~1", "/").replace("~0", "~")
                        if isinstance(selected, list):
                            selected = selected[int(token)]
                        elif isinstance(selected, Mapping):
                            selected = selected[token]
                        else:
                            raise KeyError(token)
                except (json.JSONDecodeError, KeyError, IndexError, ValueError, TypeError):
                    errors.append(
                        _error(
                            "runtime_artifact_selector_missing",
                            claim_id,
                            f"artifact_checks[{check_index}] pointer does not resolve",
                        )
                    )
                else:
                    if "observed" not in check or check.get("observed") != selected:
                        errors.append(
                            _error(
                                "runtime_artifact_observation_mismatch",
                                claim_id,
                                f"artifact_checks[{check_index}] observed value is stale",
                            )
                        )
                    elif role == "producer_script":
                        normalized_selected = str(selected).replace("\\", "/").casefold()
                        normalized_raw = str(raw_script).replace("\\", "/").casefold()
                        if (
                            normalized_selected != normalized_raw
                            and not normalized_selected.endswith(
                                f"/{normalized_raw}"
                            )
                            and Path(normalized_selected).name
                            != Path(normalized_raw).name
                        ):
                            errors.append(
                                _error(
                                    "runtime_producer_script_mismatch",
                                    claim_id,
                                    f"artifact_checks[{check_index}] names another script",
                                )
                            )
                    elif role == "producer_script_sha256":
                        if selected != expected_script_hash:
                            errors.append(
                                _error(
                                    "runtime_producer_script_hash_mismatch",
                                    claim_id,
                                    f"artifact_checks[{check_index}] has another script hash",
                                )
                            )
                    elif role == "producer_command":
                        if selected != command:
                            errors.append(
                                _error(
                                    "runtime_producer_command_mismatch",
                                    claim_id,
                                    f"artifact_checks[{check_index}] has another command",
                                )
                            )
                    elif role == "producer_exit_code" and selected != 0:
                        errors.append(
                            _error(
                                "runtime_producer_exit_code_invalid",
                                claim_id,
                                f"artifact_checks[{check_index}] producer did not exit zero",
                            )
                        )
                if len(str(check.get("supports", "")).strip()) < 20:
                    errors.append(
                        _error(
                            "runtime_artifact_support_missing",
                            claim_id,
                            f"artifact_checks[{check_index}] lacks claim-specific support",
                        )
                    )
            for role in sorted(RUNTIME_PRODUCER_ROLES):
                count = producer_roles[role]
                if count == 0:
                    errors.append(
                        _error(
                            f"runtime_{role}_check_missing",
                            claim_id,
                            f"runtime_evidence[{index}] lacks {role}",
                        )
                    )
                elif count > 1:
                    errors.append(
                        _error(
                            "runtime_producer_check_duplicate",
                            claim_id,
                            f"runtime_evidence[{index}] repeats {role}",
                        )
                    )
        baseline = item.get("runtime_baseline")
        if not isinstance(baseline, Mapping) or not RUNTIME_BASELINE_FIELDS.issubset(
            baseline
        ):
            errors.append(
                _error(
                    "runtime_baseline_incomplete",
                    claim_id,
                    f"runtime_evidence[{index}] baseline is incomplete",
                )
            )
        if not str(item.get("supports", "")).strip():
            errors.append(
                _error(
                    "runtime_support_explanation_missing",
                    claim_id,
                    f"runtime_evidence[{index}] has no supports explanation",
                )
            )
    return errors


def _inference_graph_errors(
    decisions_by_id: Mapping[str, Mapping[str, object]]
) -> list[dict[str, object]]:
    errors: list[dict[str, object]] = []
    graph: dict[str, list[str]] = {}
    for claim_id, decision in decisions_by_id.items():
        if decision.get("evidence_class") != "I":
            continue
        parents = decision.get("parent_claim_ids")
        if not isinstance(parents, list) or not parents:
            errors.append(
                _error(
                    "inference_parent_missing",
                    claim_id,
                    "I evidence requires parent_claim_ids",
                )
            )
            graph[claim_id] = []
            continue
        support = decision.get("evidence")
        support_by_parent: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        if isinstance(support, list):
            for item in support:
                if isinstance(item, Mapping) and isinstance(
                    item.get("parent_claim_id"), str
                ):
                    support_by_parent[str(item["parent_claim_id"])].append(item)
        expected_parents = [str(parent) for parent in parents]
        if set(support_by_parent) != set(expected_parents) or any(
            len(support_by_parent[parent]) != 1 for parent in set(expected_parents)
        ):
            errors.append(
                _error(
                    "inference_support_missing",
                    claim_id,
                    "I evidence requires exactly one rationale for every parent claim",
                )
            )
        else:
            for parent in expected_parents:
                explanation = str(
                    support_by_parent[parent][0].get("supports", "")
                ).strip()
                if len(explanation) < 40:
                    errors.append(
                        _error(
                            "inference_support_not_claim_specific",
                            claim_id,
                            f"parent {parent} rationale is too weak",
                        )
                    )
        graph[claim_id] = []
        for parent in parents:
            parent_id = str(parent)
            parent_decision = decisions_by_id.get(parent_id)
            if parent_decision is None:
                errors.append(
                    _error(
                        "inference_parent_missing",
                        claim_id,
                        f"parent claim does not exist: {parent_id}",
                    )
                )
                continue
            if parent_decision.get("evidence_class") not in {"S", "R", "I"}:
                errors.append(
                    _error(
                        "inference_parent_invalid",
                        claim_id,
                        f"parent {parent_id} is not S/R/I evidence",
                    )
                )
            if parent_decision.get("status") != "verified-current":
                errors.append(
                    _error(
                        "inference_parent_unverified",
                        claim_id,
                        f"parent {parent_id} is not verified-current",
                    )
                )
            graph[claim_id].append(parent_id)

    state: dict[str, int] = {}
    cycle_nodes: set[str] = set()

    def visit(node: str, stack: list[str]) -> None:
        current = state.get(node, 0)
        if current == 1:
            if node in stack:
                cycle_nodes.update(stack[stack.index(node) :])
            return
        if current == 2:
            return
        state[node] = 1
        stack.append(node)
        for parent in graph.get(node, []):
            if parent in graph:
                visit(parent, stack)
        stack.pop()
        state[node] = 2

    for node in graph:
        visit(node, [])
    for node in sorted(cycle_nodes):
        errors.append(
            _error("inference_cycle", node, "inference parent graph contains a cycle")
        )
    return errors


def validate_course_claim_decisions(
    claim_rows: Sequence[Mapping[str, object]],
    decisions: Sequence[Mapping[str, object]],
    repo_root: Path,
    source_root: Path,
    pinned_baseline: str,
) -> list[dict[str, object]]:
    errors: list[dict[str, object]] = []
    claims_by_id = {str(row["id"]): row for row in claim_rows}
    decisions_by_id: dict[str, Mapping[str, object]] = {}
    decision_counts: Counter[str] = Counter()

    for decision in decisions:
        claim_id = str(decision.get("claim_id", ""))
        decision_counts[claim_id] += 1
        decisions_by_id.setdefault(claim_id, decision)
        missing = sorted(REQUIRED_FIELDS.difference(decision))
        if missing:
            errors.append(
                _error(
                    "decision_schema_missing_fields",
                    claim_id,
                    f"missing fields: {', '.join(missing)}",
                )
            )
            continue
        claim = claims_by_id.get(claim_id)
        if claim is None:
            errors.append(
                _error(
                    "unknown_claim_decision",
                    claim_id,
                    "decision does not select a manifest claim",
                )
            )
            continue
        if (
            decision.get("page") != claim.get("page")
            or int(decision.get("start_line", -1))
            != int(claim.get("source_start_line", -2))
            or int(decision.get("end_line", -1))
            != int(claim.get("source_end_line", -2))
        ):
            errors.append(
                _error(
                    "claim_span_mismatch",
                    claim_id,
                    "decision page/span does not match the canonical claim",
                )
            )
        if decision.get("text_sha256") != _text_sha256(str(claim.get("text", ""))):
            errors.append(
                _error(
                    "text_hash_mismatch",
                    claim_id,
                    "decision text hash is stale",
                )
            )

        content_class = decision.get("content_class")
        evidence_class = decision.get("evidence_class")
        if content_class not in CONTENT_CLASSES:
            errors.append(
                _error(
                    "invalid_content_class",
                    claim_id,
                    f"unsupported content class: {content_class!r}",
                )
            )
        if evidence_class is not None and evidence_class not in EVIDENCE_CLASSES:
            errors.append(
                _error(
                    "invalid_evidence_class",
                    claim_id,
                    f"unsupported evidence class: {evidence_class!r}",
                )
            )

        if content_class in NONASSERTIVE_CLASSES:
            if evidence_class is not None:
                errors.append(
                    _error(
                        "nonassertive_has_evidence_class",
                        claim_id,
                        "nonassertive content must not be labelled S/R/I/M/B",
                    )
                )
            if (
                decision.get("status") != "not-applicable"
                or decision.get("presentation") != "nonassertive"
            ):
                errors.append(
                    _error(
                        "nonassertive_status_invalid",
                        claim_id,
                        "nonassertive content requires not-applicable/nonassertive",
                    )
                )
        elif evidence_class is None:
            errors.append(
                _error(
                    "assertion_evidence_missing",
                    claim_id,
                    "assertion/provenance/limitation requires S/R/I/M/B",
                )
            )

        if not isinstance(decision.get("evidence"), list):
            errors.append(
                _error("evidence_not_array", claim_id, "evidence must be an array")
            )
        if not isinstance(decision.get("parent_claim_ids"), list):
            errors.append(
                _error(
                    "parent_claim_ids_not_array",
                    claim_id,
                    "parent_claim_ids must be an array",
                )
            )
        if evidence_class == "S":
            errors.extend(
                _validate_source_evidence(decision, source_root, pinned_baseline)
            )
        if evidence_class in {"R", "M", "B"}:
            errors.extend(
                _validate_runtime_evidence(decision, repo_root, required=True)
            )
        elif isinstance(decision.get("runtime_evidence"), list):
            errors.extend(
                _validate_runtime_evidence(decision, repo_root, required=False)
            )
        if evidence_class == "B":
            if decision.get("status") != "unresolved":
                errors.append(
                    _error(
                        "blocked_status_invalid",
                        claim_id,
                        "B evidence must leave the blocked execution claim unresolved",
                    )
                )
            if decision.get("presentation") not in {"qualified", "blocked"}:
                errors.append(
                    _error(
                        "blocked_presentation_not_qualified",
                        claim_id,
                        "B evidence cannot be presented as authoritative/executed",
                    )
                )
            if not str(decision.get("notes", "")).strip():
                errors.append(
                    _error(
                        "blocked_reason_missing",
                        claim_id,
                        "B evidence requires an explicit blocker",
                    )
                )
        elif evidence_class in {"S", "R", "I", "M"} and decision.get(
            "status"
        ) != "verified-current":
            errors.append(
                _error(
                    "verified_evidence_status_invalid",
                    claim_id,
                    "S/R/I/M evidence requires verified-current status",
                )
            )
        if evidence_class == "M" and decision.get("presentation") not in {
            "qualified",
            "codegen-only",
        }:
            errors.append(
                _error(
                    "mock_presentation_not_qualified",
                    claim_id,
                    "M evidence must be presented as codegen-only/qualified",
                )
            )

    for claim_id in claims_by_id:
        count = decision_counts[claim_id]
        if count == 0:
            errors.append(
                _error(
                    "missing_claim_decision",
                    claim_id,
                    "canonical claim has no decision",
                )
            )
        elif count > 1:
            errors.append(
                _error(
                    "duplicate_claim_decision",
                    claim_id,
                    f"canonical claim has {count} decisions",
                )
            )
    errors.extend(_inference_graph_errors(decisions_by_id))
    return errors


def merge_course_claim_ledger(
    claim_rows: Sequence[Mapping[str, object]],
    decisions: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    decisions_by_id = {
        str(decision["claim_id"]): decision for decision in decisions
    }
    ledger: list[dict[str, object]] = []
    for row in claim_rows:
        claim_id = str(row["id"])
        decision = decisions_by_id[claim_id]
        merged = dict(row)
        merged["claim_id"] = claim_id
        for field in REQUIRED_FIELDS.difference({"claim_id"}):
            merged[field] = decision[field]
        ledger.append(merged)
    return ledger


def write_course_claim_ledger(
    path: Path, ledger: Sequence[Mapping[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in ledger:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_course_claim_summary(
    path: Path,
    ledger: Sequence[Mapping[str, object]],
    errors: Sequence[Mapping[str, object]],
) -> None:
    evidence = Counter(str(row.get("evidence_class")) for row in ledger)
    content = Counter(str(row.get("content_class")) for row in ledger)
    unit_kinds = Counter(str(row.get("kind")) for row in ledger)
    statuses = Counter(str(row.get("status")) for row in ledger)
    by_page: dict[str, Counter[str]] = defaultdict(Counter)
    for row in ledger:
        by_page[str(row["page"])][str(row.get("evidence_class"))] += 1
    error_counts = Counter(str(error["code"]) for error in errors)
    output = [
        "# Course Claim Evidence Summary",
        "",
        f"- Claim candidates: **{len(ledger)}**",
        f"- Validation errors: **{len(errors)}**",
        "",
        "## Evidence classes",
        "",
        "| Class | Count |",
        "|---|---:|",
    ]
    output.extend(f"| `{key}` | {value} |" for key, value in sorted(evidence.items()))
    output.extend(
        [
            "",
            "## Content classes",
            "",
            "| Class | Count |",
            "|---|---:|",
        ]
    )
    output.extend(f"| `{key}` | {value} |" for key, value in sorted(content.items()))
    output.extend(
        [
            "",
            "## Claim unit kinds",
            "",
            "| Kind | Count |",
            "|---|---:|",
        ]
    )
    output.extend(
        f"| `{key}` | {value} |" for key, value in sorted(unit_kinds.items())
    )
    output.extend(
        [
            "",
            "## Statuses",
            "",
            "| Status | Count |",
            "|---|---:|",
        ]
    )
    output.extend(
        f"| `{key}` | {value} |" for key, value in sorted(statuses.items())
    )
    output.extend(
        [
            "",
            "## Page coverage",
            "",
            "| Page | S | R | I | M | B | none |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for page, counts in sorted(by_page.items()):
        output.append(
            f"| `{page}` | {counts['S']} | {counts['R']} | {counts['I']} | "
            f"{counts['M']} | {counts['B']} | {counts['None']} |"
        )
    output.extend(
        [
            "",
            "## Validation gates",
            "",
            "| Error code | Count |",
            "|---|---:|",
        ]
    )
    if error_counts:
        output.extend(
            f"| `{key}` | {value} |" for key, value in sorted(error_counts.items())
        )
    else:
        output.append("| all gates | 0 |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(output) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and build the numbered course claim ledger."
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--decisions-dir", type=Path, required=True)
    parser.add_argument("--pinned-baseline", required=True)
    parser.add_argument("--ledger-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--errors-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    source_root = args.source_root.resolve()
    claims = build_course_claim_rows(
        repo_root, source_root, args.manifest.resolve()
    )
    decisions = load_course_claim_decisions(args.decisions_dir.resolve())
    errors = [
        *validate_source_checkout(source_root, args.pinned_baseline),
        *validate_course_claim_decisions(
            claims,
            decisions,
            repo_root,
            source_root,
            args.pinned_baseline,
        ),
    ]
    ledger = (
        merge_course_claim_ledger(claims, decisions)
        if not errors
        else []
    )
    if ledger:
        write_course_claim_ledger(args.ledger_output.resolve(), ledger)
    args.errors_output.parent.mkdir(parents=True, exist_ok=True)
    args.errors_output.write_text(
        json.dumps(errors, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_course_claim_summary(args.summary_output.resolve(), ledger, errors)
    print(
        json.dumps(
            {
                "claims": len(claims),
                "decisions": len(decisions),
                "errors": len(errors),
                "ledger_rows": len(ledger),
            },
            sort_keys=True,
        )
    )
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
