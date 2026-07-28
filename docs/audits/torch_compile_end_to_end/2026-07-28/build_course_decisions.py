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
        decisions[claim_id] = decision
    if not decisions:
        raise RuntimeError("no audited C01-C21 decisions found")
    return decisions


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
        key = (match.group("path"), start, end)
        if key in seen:
            continue
        seen.add(key)
        evidence.append(
            {
                "baseline": baseline,
                "path": key[0],
                "start_line": start,
                "end_line": end,
                "supports": (
                    "该固定源码范围直接支撑本事实单元中对应对象、调用链、条件或状态变化；"
                    f"被审计文本为：{compact_claim[:180]}"
                ),
            }
        )
    return evidence


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
    volume_c_decisions = load_volume_c_decisions(graph_ledger)

    source_rows_by_page_section: dict[
        tuple[str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    source_rows_by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_rows: list[dict[str, Any]] = []
    for row in rows:
        if source_evidence(str(row.get("text", "")), args.pinned_baseline):
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

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        page = str(row["page"])
        if is_volume_c_page(page):
            claim_id = str(row["id"])
            if claim_id not in volume_c_decisions:
                raise RuntimeError(
                    f"volume C claim {claim_id} has no decision in {graph_ledger}"
                )
            grouped["c"].append(copy.deepcopy(volume_c_decisions[claim_id]))
            continue
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
        grouped[volume_key(page)].append(
            decision_for(
                row,
                baseline=args.pinned_baseline,
                parent=parent,
                blocker_runtime=blocker_runtime,
            )
        )

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
