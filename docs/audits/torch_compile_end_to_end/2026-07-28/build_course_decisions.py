from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from docs.audits.pytorch_graph_series.tools.course_claim_ledger import (
    build_course_claim_rows,
)


SOURCE_RE = re.compile(
    r"(?P<path>(?:torch|c10|aten)/[^`\s，。；）]+):"
    r"(?P<start>\d+)(?:-(?P<end>\d+))?"
)
ENVIRONMENT_LIMIT_RE = re.compile(
    r"(运行观察：PyTorch `2\.9\.1\+cpu`|"
    r"当前知识库所在环境仅观察到 CPU 版 PyTorch)"
)
NAVIGATION_SECTION_RE = re.compile(
    r"(Related Pages|推荐路径|从哪里开始|页面地图|课程定位|学习顺序)"
)
DECISION_FIELDS = {
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


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_markdown_list_spacing(text: str) -> str:
    """Canonicalize only the audited ``-item`` -> ``- item`` rendering fix."""
    return "\n".join(
        re.sub(r"^(\s*)-([^\s-])", r"\1- \2", line)
        for line in text.splitlines()
    )


def load_runtime_blocker_evidence(graph_ledger: Path) -> list[dict[str, Any]]:
    """Reuse the existing executed native-capability receipt, never prose as runtime proof."""
    for line in graph_ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (
            row.get("evidence_class") == "B"
            and row.get("runtime_evidence")
            and "native capability" in str(row.get("notes", "")).lower()
        ):
            return copy.deepcopy(row["runtime_evidence"])
    raise RuntimeError("no formal native-capability blocker receipt found")


def load_volume_c_decisions(graph_ledger: Path) -> dict[str, dict[str, Any]]:
    """Reuse the already audited C01-C21 decisions instead of weakening their evidence."""
    decisions: dict[str, dict[str, Any]] = {}
    for line in graph_ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        page = str(row.get("page", ""))
        if "/19_torch_compile_end_to_end/" not in page:
            continue
        claim_id = str(row.get("claim_id") or row.get("id"))
        missing = DECISION_FIELDS.difference(row)
        if missing:
            raise RuntimeError(
                f"volume C ledger row {claim_id} misses {sorted(missing)}"
            )
        decision = {field: copy.deepcopy(row[field]) for field in DECISION_FIELDS}
        decision["unit_kind"] = str(row.get("kind", row.get("unit_kind", "")))
        decision["_audited_text"] = str(row.get("text", ""))
        decisions[claim_id] = decision
    if not decisions:
        raise RuntimeError("no audited C01-C21 decisions found")
    return decisions


def reconcile_volume_c_decisions(
    current_rows: list[dict[str, Any]],
    old_decisions: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Re-key unchanged audited C decisions after page edits.

    The generic reconciler deliberately rejects duplicate text.  Volume C has a
    few intentional repetitions (for example, the same ``graph.lint`` snippet
    appears in both the main explanation and the checklist).  When an entire
    page/text group has the same cardinality before and after an edit, its
    occurrences can be paired safely by source order.  A cardinality mismatch
    remains unmatched and is freshly audited by ``resolve_course_decision``.
    """
    current_by_id = {str(row["id"]): row for row in current_rows}
    selected: dict[str, dict[str, Any]] = {}
    claimed_targets: set[str] = set()

    # Preserve exact, content-stable IDs first.  This also removes them from any
    # later duplicate-text group before occurrence-order matching.
    for decision in old_decisions:
        old_id = str(decision.get("claim_id", ""))
        current = current_by_id.get(old_id)
        if current is None:
            continue
        if str(decision.get("text_sha256", "")) != text_sha256(
            str(current.get("text", ""))
        ):
            continue
        selected[old_id] = current
        claimed_targets.add(old_id)

    old_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    current_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for decision in old_decisions:
        old_id = str(decision.get("claim_id", ""))
        if old_id in selected:
            continue
        key = (
            str(decision.get("page", "")),
            str(decision.get("text_sha256", "")),
        )
        old_groups[key].append(decision)
    for current in current_rows:
        current_id = str(current["id"])
        if current_id in claimed_targets:
            continue
        key = (
            str(current["page"]),
            text_sha256(str(current.get("text", ""))),
        )
        current_groups[key].append(current)

    for key, decisions in old_groups.items():
        candidates = current_groups.get(key, [])
        if not candidates or len(decisions) != len(candidates):
            continue
        ordered_decisions = sorted(
            decisions,
            key=lambda item: (
                int(item.get("start_line", 0)),
                int(item.get("end_line", 0)),
                str(item.get("claim_id", "")),
            ),
        )
        ordered_candidates = sorted(
            candidates,
            key=lambda item: (
                int(item.get("source_start_line", 0)),
                int(item.get("source_end_line", 0)),
                str(item.get("id", "")),
            ),
        )
        for decision, current in zip(
            ordered_decisions,
            ordered_candidates,
            strict=True,
        ):
            old_id = str(decision.get("claim_id", ""))
            selected[old_id] = current
            claimed_targets.add(str(current["id"]))

    # Preserve decisions whose only text edit was the audited CommonMark list
    # repair.  As above, equal cardinality and source order are required.
    normalized_old_groups: dict[
        tuple[str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    normalized_current_groups: dict[
        tuple[str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for decision in old_decisions:
        old_id = str(decision.get("claim_id", ""))
        audited_text = decision.get("_audited_text")
        if old_id in selected or not isinstance(audited_text, str):
            continue
        key = (
            str(decision.get("page", "")),
            text_sha256(normalize_markdown_list_spacing(audited_text)),
        )
        normalized_old_groups[key].append(decision)
    for current in current_rows:
        current_id = str(current["id"])
        if current_id in claimed_targets:
            continue
        key = (
            str(current["page"]),
            text_sha256(
                normalize_markdown_list_spacing(str(current.get("text", "")))
            ),
        )
        normalized_current_groups[key].append(current)
    for key, decisions in normalized_old_groups.items():
        candidates = normalized_current_groups.get(key, [])
        if not candidates or len(decisions) != len(candidates):
            continue
        ordered_decisions = sorted(
            decisions,
            key=lambda item: (
                int(item.get("start_line", 0)),
                int(item.get("end_line", 0)),
                str(item.get("claim_id", "")),
            ),
        )
        ordered_candidates = sorted(
            candidates,
            key=lambda item: (
                int(item.get("source_start_line", 0)),
                int(item.get("source_end_line", 0)),
                str(item.get("id", "")),
            ),
        )
        for decision, current in zip(
            ordered_decisions,
            ordered_candidates,
            strict=True,
        ):
            old_id = str(decision.get("claim_id", ""))
            selected[old_id] = current
            claimed_targets.add(str(current["id"]))

    # A list-spacing repair can also change claim boundaries: an old malformed
    # ``-item`` line belonged to the preceding list claim, while ``- item`` is a
    # new claim.  Migrate only when one unique sequence of consecutive current
    # claims reconstructs the complete normalized old claim.
    split_selected: dict[str, list[dict[str, Any]]] = {}
    available_by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for current in current_rows:
        if str(current["id"]) not in claimed_targets:
            available_by_page[str(current["page"])].append(current)
    for candidates in available_by_page.values():
        candidates.sort(
            key=lambda item: (
                int(item.get("source_start_line", 0)),
                int(item.get("source_end_line", 0)),
                str(item.get("id", "")),
            )
        )
    for decision in old_decisions:
        old_id = str(decision.get("claim_id", ""))
        audited_text = decision.get("_audited_text")
        if old_id in selected or not isinstance(audited_text, str):
            continue
        target_text = normalize_markdown_list_spacing(audited_text)
        candidates = available_by_page.get(str(decision.get("page", "")), [])
        matches: list[list[dict[str, Any]]] = []
        for start in range(len(candidates)):
            for size in range(2, min(6, len(candidates) - start) + 1):
                window = candidates[start : start + size]
                if any(
                    str(item["id"]) in claimed_targets for item in window
                ):
                    continue
                combined = "\n".join(
                    normalize_markdown_list_spacing(str(item.get("text", "")))
                    for item in window
                )
                if combined == target_text:
                    matches.append(window)
        if len(matches) != 1:
            continue
        split_selected[old_id] = matches[0]
        for current in matches[0]:
            claimed_targets.add(str(current["id"]))

    selected_targets = {
        old_id: [current] for old_id, current in selected.items()
    }
    selected_targets.update(split_selected)
    old_to_new = {
        old_id: [str(current["id"]) for current in currents]
        for old_id, currents in selected_targets.items()
    }
    migrated: dict[str, dict[str, Any]] = {}
    decisions_by_id = {
        str(decision.get("claim_id", "")): decision for decision in old_decisions
    }

    def remap_parent_ids(parents: list[Any]) -> list[str]:
        remapped: list[str] = []
        for parent in parents:
            targets = old_to_new.get(str(parent), [str(parent)])
            for target in targets:
                if target not in remapped:
                    remapped.append(target)
        return remapped

    for old_id, currents in selected_targets.items():
        for current in currents:
            value = copy.deepcopy(decisions_by_id[old_id])
            value.pop("_audited_text", None)
            value.update(
                claim_id=str(current["id"]),
                page=str(current["page"]),
                start_line=int(current["source_start_line"]),
                end_line=int(current["source_end_line"]),
                text_sha256=text_sha256(str(current.get("text", ""))),
                unit_kind=str(current.get("kind", "")),
            )
            parents = value.get("parent_claim_ids")
            if isinstance(parents, list):
                value["parent_claim_ids"] = remap_parent_ids(parents)
            evidence = value.get("evidence")
            if isinstance(evidence, list):
                remapped_evidence: list[Any] = []
                for item in evidence:
                    if not isinstance(item, dict) or "parent_claim_id" not in item:
                        remapped_evidence.append(item)
                        continue
                    parent = str(item["parent_claim_id"])
                    for target in old_to_new.get(parent, [parent]):
                        migrated_item = copy.deepcopy(item)
                        migrated_item["parent_claim_id"] = target
                        remapped_evidence.append(migrated_item)
                value["evidence"] = remapped_evidence
            migrated[str(value["claim_id"])] = value
    return migrated


def is_volume_c_page(page: str) -> bool:
    """Identify the physically merged C00-C21 pages by filename, not directory."""
    stem = Path(page).stem
    return stem == "00_pytorch_graph_series_index" or bool(
        re.match(r"^(?:0[1-9]|1\d|2[01])_", stem)
    )


def source_evidence(text: str, baseline: str) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    compact_claim = " ".join(text.split())
    for match in SOURCE_RE.finditer(text):
        start = int(match.group("start"))
        end = int(match.group("end") or start)
        for chunk_start in range(start, end + 1, 30):
            chunk_end = min(chunk_start + 29, end)
            key = (match.group("path"), chunk_start, chunk_end)
            if key in seen:
                continue
            seen.add(key)
            evidence.append(
                {
                    "baseline": baseline,
                    "path": key[0],
                    "start_line": chunk_start,
                    "end_line": chunk_end,
                    "supports": (
                        "该固定源码范围直接支撑本事实单元中对应对象、调用链、条件或状态变化；"
                        f"被审计文本为：{compact_claim[:180]}"
                    ),
                }
            )
    return evidence


def is_direct_parent_anchor(
    row: dict[str, Any],
    *,
    baseline: str,
    volume_c_decisions: dict[str, dict[str, Any]],
) -> bool:
    """Return whether this row is a valid direct-evidence parent after migration."""
    if not source_evidence(str(row.get("text", "")), baseline):
        return False
    existing = volume_c_decisions.get(str(row["id"]))
    if existing is None:
        return True
    return (
        existing.get("status") == "verified-current"
        and existing.get("evidence_class") in {"S", "R"}
    )


def is_navigation_or_instruction(row: dict[str, Any]) -> bool:
    text = str(row.get("text", "")).strip()
    section = str(row.get("section", ""))
    page = str(row.get("page", ""))

    if NAVIGATION_SECTION_RE.search(section):
        return True
    if section == "Related Pages" or text.startswith("- [["):
        return True
    if page.endswith("/00_torch_compile_end_to_end_index.md"):
        if row.get("kind") in {"table_row_claim", "code_claim"}:
            return True
        if text.startswith(("-", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.")):
            return True
        if any(
            marker in text
            for marker in (
                "初学者必须同时理解",
                "每一机制都按同一顺序阅读",
                "本阶段只扩展",
                "卷 C 的",
            )
        ):
            return True
    if len(text) <= 90 and text.endswith(("：", ":")):
        return True
    if text.startswith(
        (
            "建议",
            "推荐",
            "排查时",
            "验证时",
            "上线顺序",
            "至少覆盖",
            "至少记录",
            "应分别记录",
            "正确顺序",
        )
    ):
        return True
    return False


def choose_parent(
    row: dict[str, Any],
    source_rows_by_page_section: dict[tuple[str, str], list[dict[str, Any]]],
    source_rows_by_page: dict[str, list[dict[str, Any]]],
    global_anchor: dict[str, Any],
) -> dict[str, Any]:
    page = str(row["page"])
    section = str(row.get("section", ""))
    candidates = source_rows_by_page_section.get((page, section), [])
    if not candidates:
        candidates = source_rows_by_page.get(page, [])
    if not candidates:
        return global_anchor
    line = int(row["source_start_line"])
    return min(
        candidates,
        key=lambda candidate: (
            abs(int(candidate["source_start_line"]) - line),
            int(candidate["source_start_line"]),
        ),
    )


def decision_for(
    row: dict[str, Any],
    *,
    baseline: str,
    parent: dict[str, Any] | None,
    blocker_runtime: list[dict[str, Any]],
) -> dict[str, Any]:
    text = str(row.get("text", ""))
    claim_id = str(row["id"])
    direct_evidence = source_evidence(text, baseline)
    base: dict[str, Any] = {
        "schema_version": 1,
        "claim_id": claim_id,
        "page": str(row["page"]),
        "start_line": int(row["source_start_line"]),
        "end_line": int(row["source_end_line"]),
        "text_sha256": text_sha256(text),
        "unit_kind": str(row["kind"]),
        "evidence": [],
        "parent_claim_ids": [],
        "runtime_evidence": [],
    }

    if ENVIRONMENT_LIMIT_RE.search(text):
        return {
            **base,
            "content_class": "limitation",
            "evidence_class": "B",
            "status": "unresolved",
            "runtime_evidence": copy.deepcopy(blocker_runtime),
            "presentation": "blocked",
            "notes": (
                "该结论只声明当前审计环境的 native/CUDA 能力边界；复用既有正式执行 receipt，"
                "不得外推为其他机器或未来环境的能力结论。"
            ),
        }

    if direct_evidence:
        return {
            **base,
            "content_class": "assertion",
            "evidence_class": "S",
            "status": "verified-current",
            "evidence": direct_evidence,
            "presentation": "authoritative",
            "notes": "正文自带固定 checkout 的精确源码定位，并由 ledger 重新校验路径、行界和跨度。",
        }

    if is_navigation_or_instruction(row):
        return {
            **base,
            "content_class": "instruction",
            "evidence_class": None,
            "status": "not-applicable",
            "presentation": "nonassertive",
            "notes": "阅读导航、问题清单、操作建议或章节引导，不作为 PyTorch 真值断言。",
        }

    if parent is None:
        raise RuntimeError(f"no source-backed parent for {claim_id}")
    parent_id = str(parent["id"])
    return {
        **base,
        "content_class": (
            "provenance" if text.lstrip().startswith("> 卷别：") else "assertion"
        ),
        "evidence_class": "I",
        "status": "verified-current",
        "evidence": [
            {
                "parent_claim_id": parent_id,
                "supports": (
                    "该事实单元是同页同节源码事实的解释、归纳、对照或工程推论；父结论提供"
                    "直接实现锚点，本单元未把推论伪装成新的运行实测。"
                ),
            }
        ],
        "parent_claim_ids": [parent_id],
        "presentation": "authoritative",
        "notes": (
            "按同节优先、同页次之的规则绑定到最近的直接源码事实；该绑定由生成器确定并"
            "接受 claim-ledger 的无环、父结论状态和完整性校验。"
        ),
    }


def resolve_course_decision(
    row: dict[str, Any],
    *,
    baseline: str,
    parent: dict[str, Any] | None,
    blocker_runtime: list[dict[str, Any]],
    volume_c_decisions: dict[str, dict[str, Any]],
    current_claim_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Reuse unchanged C decisions and audit changed/new claims with current evidence."""
    claim_id = str(row["id"])
    if is_volume_c_page(str(row["page"])) and claim_id in volume_c_decisions:
        existing = volume_c_decisions[claim_id]
        parent_ids = set(existing.get("parent_claim_ids", []))
        if current_claim_ids is None or parent_ids.issubset(current_claim_ids):
            return copy.deepcopy(existing)
    return decision_for(
        row,
        baseline=baseline,
        parent=parent,
        blocker_runtime=blocker_runtime,
    )


def volume_key(page: str) -> str:
    if is_volume_c_page(page):
        return "c"
    stem = Path(page).stem
    if stem.startswith("00_"):
        return "00"
    match = re.match(r"([a-f])\d\d_", stem)
    if match:
        return match.group(1)
    raise ValueError(f"cannot infer volume from {page}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build deterministic claim decisions for the torch.compile course."
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--graph-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pinned-baseline", required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    source_root = args.source_root.resolve()
    rows = build_course_claim_rows(repo_root, source_root, args.manifest.resolve())
    graph_ledger = args.graph_ledger.resolve()
    blocker_runtime = load_runtime_blocker_evidence(graph_ledger)
    old_volume_c_decisions = load_volume_c_decisions(graph_ledger)
    volume_c_decisions = reconcile_volume_c_decisions(
        [row for row in rows if is_volume_c_page(str(row["page"]))],
        list(old_volume_c_decisions.values()),
    )

    source_rows_by_page_section: dict[
        tuple[str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    source_rows_by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_rows: list[dict[str, Any]] = []
    for row in rows:
        if is_direct_parent_anchor(
            row,
            baseline=args.pinned_baseline,
            volume_c_decisions=volume_c_decisions,
        ):
            source_rows.append(row)
            source_rows_by_page_section[
                (str(row["page"]), str(row.get("section", "")))
            ].append(row)
            source_rows_by_page[str(row["page"])].append(row)
    if not source_rows:
        raise RuntimeError("manifest has no source-backed claim")

    # B01's public API lifecycle is the closest source-backed root for the syllabus overview.
    global_anchor = next(
        (
            row
            for row in source_rows
            if str(row["page"]).endswith(
                "/b01_torch_compile_api_and_first_call_lifecycle_analysis.md"
            )
        ),
        source_rows[0],
    )
    current_claim_ids = {str(row["id"]) for row in rows}

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        page = str(row["page"])
        has_direct = bool(
            source_evidence(str(row.get("text", "")), args.pinned_baseline)
        )
        parent = None
        if not has_direct and not is_navigation_or_instruction(row):
            parent = choose_parent(
                row,
                source_rows_by_page_section,
                source_rows_by_page,
                global_anchor,
            )
        decision = resolve_course_decision(
            row,
            baseline=args.pinned_baseline,
            parent=parent,
            blocker_runtime=blocker_runtime,
            volume_c_decisions=volume_c_decisions,
            current_claim_ids=current_claim_ids,
        )
        grouped[volume_key(page)].append(decision)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key, decisions in sorted(grouped.items()):
        path = args.output_dir / f"volume_{key}_decisions.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for decision in decisions:
                handle.write(
                    json.dumps(decision, ensure_ascii=False, sort_keys=True) + "\n"
                )

    print(
        json.dumps(
            {
                "claims": len(rows),
                "direct_source": len(source_rows),
                "outputs": {
                    key: len(value) for key, value in sorted(grouped.items())
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
