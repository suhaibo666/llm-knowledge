from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
BUILDER_PATH = (
    REPO_ROOT
    / "docs"
    / "audits"
    / "torch_compile_end_to_end"
    / "2026-07-28"
    / "build_course_decisions.py"
)
SPEC = importlib.util.spec_from_file_location("torch_compile_course_builder", BUILDER_PATH)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class VolumeCDecisionCompatibilityTests(unittest.TestCase):
    def test_changed_source_backed_claim_gets_a_fresh_decision(self) -> None:
        row = {
            "id": "changed-c-claim",
            "page": (
                "wiki/02_engineering/01_ai_frameworks/"
                "19_torch_compile_end_to_end/09_aotautograd_joint_forward_backward_graphs.md"
            ),
            "source_start_line": 100,
            "source_end_line": 101,
            "kind": "claim_candidate",
            "text": "Fresh evidence: torch/fx/graph.py:10-20.",
        }

        decision = builder.resolve_course_decision(
            row,
            baseline="e8f97c1a",
            parent=None,
            blocker_runtime=[],
            volume_c_decisions={},
        )

        self.assertEqual(decision["claim_id"], "changed-c-claim")
        self.assertEqual(decision["evidence_class"], "S")
        self.assertEqual(decision["status"], "verified-current")
        self.assertEqual(decision["evidence"][0]["baseline"], "e8f97c1a")

    def test_unchanged_volume_c_claim_reuses_the_audited_decision(self) -> None:
        existing = {
            "schema_version": 1,
            "claim_id": "unchanged-c-claim",
            "page": (
                "wiki/02_engineering/01_ai_frameworks/"
                "19_torch_compile_end_to_end/13_pattern_expression_and_matcher_engine.md"
            ),
            "start_line": 1,
            "end_line": 1,
            "text_sha256": "abc",
            "unit_kind": "claim_candidate",
            "content_class": "assertion",
            "evidence_class": "S",
            "status": "verified-current",
            "evidence": [{"baseline": "old-audit"}],
            "parent_claim_ids": [],
            "runtime_evidence": [],
            "presentation": "authoritative",
            "notes": "audited",
        }
        row = {
            "id": "unchanged-c-claim",
            "page": (
                "wiki/02_engineering/01_ai_frameworks/"
                "19_torch_compile_end_to_end/13_pattern_expression_and_matcher_engine.md"
            ),
            "source_start_line": 1,
            "source_end_line": 1,
            "kind": "claim_candidate",
            "text": "The unchanged text is not regenerated.",
        }

        decision = builder.resolve_course_decision(
            row,
            baseline="e8f97c1a",
            parent=None,
            blocker_runtime=[],
            volume_c_decisions={"unchanged-c-claim": existing},
        )

        self.assertEqual(decision, existing)
        self.assertIsNot(decision, existing)

    def test_volume_c_decision_with_a_stale_parent_is_regenerated(self) -> None:
        row = {
            "id": "c-inference",
            "page": (
                "wiki/02_engineering/01_ai_frameworks/"
                "19_torch_compile_end_to_end/14_dead_code_topology_and_effect_order.md"
            ),
            "source_start_line": 20,
            "source_end_line": 20,
            "kind": "claim_candidate",
            "text": "The reverse scan relies on the current users index.",
        }
        parent = {
            "id": "current-parent",
            "page": row["page"],
            "source_start_line": 18,
            "source_end_line": 18,
            "kind": "claim_candidate",
            "text": "Source: torch/fx/graph.py:2740-2756.",
        }
        stale = {
            "claim_id": "c-inference",
            "parent_claim_ids": ["removed-parent"],
        }

        decision = builder.resolve_course_decision(
            row,
            baseline="e8f97c1a",
            parent=parent,
            blocker_runtime=[],
            volume_c_decisions={"c-inference": stale},
            current_claim_ids={"c-inference", "current-parent"},
        )

        self.assertEqual(decision["evidence_class"], "I")
        self.assertEqual(decision["parent_claim_ids"], ["current-parent"])

    def test_wide_navigation_locator_is_split_into_formal_evidence_ranges(self) -> None:
        evidence = builder.source_evidence(
            "See torch/fx/graph.py:1-65.",
            "e8f97c1a",
        )

        self.assertEqual(
            [
                (item["start_line"], item["end_line"])
                for item in evidence
            ],
            [(1, 30), (31, 60), (61, 65)],
        )
        self.assertTrue(
            all(item["end_line"] - item["start_line"] + 1 <= 30 for item in evidence)
        )

    def test_migrated_inference_with_locator_is_not_a_direct_parent_anchor(
        self,
    ) -> None:
        row = {
            "id": "inference-with-locator",
            "page": (
                "wiki/02_engineering/01_ai_frameworks/"
                "19_torch_compile_end_to_end/01_graph_ir_motivation_and_taxonomy.md"
            ),
            "text": "An interpretation of torch/fx/graph.py:1-10.",
        }

        self.assertFalse(
            builder.is_direct_parent_anchor(
                row,
                baseline="e8f97c1a",
                volume_c_decisions={
                    "inference-with-locator": {"evidence_class": "I"}
                },
            )
        )
        self.assertTrue(
            builder.is_direct_parent_anchor(
                row,
                baseline="e8f97c1a",
                volume_c_decisions={
                    "inference-with-locator": {
                        "evidence_class": "S",
                        "status": "verified-current",
                    }
                },
            )
        )

    def test_volume_c_reconciliation_preserves_runtime_evidence_and_parent_chain(
        self,
    ) -> None:
        page = (
            "wiki/02_engineering/01_ai_frameworks/"
            "19_torch_compile_end_to_end/13_pattern_expression_and_matcher_engine.md"
        )
        current_rows = [
            {
                "id": "new-parent",
                "page": page,
                "source_start_line": 10,
                "source_end_line": 10,
                "kind": "claim_candidate",
                "text": "Observed runtime match count.",
            },
            {
                "id": "new-child",
                "page": page,
                "source_start_line": 11,
                "source_end_line": 11,
                "kind": "claim_candidate",
                "text": "This explains the observed count.",
            },
        ]
        old_decisions = [
            {
                "claim_id": "old-parent",
                "page": page,
                "text_sha256": builder.text_sha256(current_rows[0]["text"]),
                "evidence_class": "R",
                "parent_claim_ids": [],
                "evidence": [],
                "runtime_evidence": [{"script": "labs/observed.py"}],
            },
            {
                "claim_id": "old-child",
                "page": page,
                "text_sha256": builder.text_sha256(current_rows[1]["text"]),
                "evidence_class": "I",
                "parent_claim_ids": ["old-parent"],
                "evidence": [{"parent_claim_id": "old-parent"}],
                "runtime_evidence": [],
            },
        ]

        reconciled = builder.reconcile_volume_c_decisions(
            current_rows,
            old_decisions,
        )

        self.assertEqual(set(reconciled), {"new-parent", "new-child"})
        self.assertEqual(reconciled["new-parent"]["evidence_class"], "R")
        self.assertEqual(
            reconciled["new-parent"]["runtime_evidence"],
            [{"script": "labs/observed.py"}],
        )
        self.assertEqual(
            reconciled["new-child"]["parent_claim_ids"],
            ["new-parent"],
        )
        self.assertEqual(
            reconciled["new-child"]["evidence"][0]["parent_claim_id"],
            "new-parent",
        )

    def test_volume_c_reconciliation_maps_duplicate_text_by_occurrence_order(
        self,
    ) -> None:
        page = (
            "wiki/02_engineering/01_ai_frameworks/"
            "19_torch_compile_end_to_end/02_fx_graph_core_data_model.md"
        )
        text = "python\ngm.graph.lint()\ngm.recompile()"
        current_rows = [
            {
                "id": "new-first",
                "page": page,
                "source_start_line": 300,
                "source_end_line": 303,
                "kind": "code_claim",
                "text": text,
            },
            {
                "id": "new-second",
                "page": page,
                "source_start_line": 500,
                "source_end_line": 503,
                "kind": "code_claim",
                "text": text,
            },
        ]
        old_decisions = [
            {
                "claim_id": "old-first",
                "page": page,
                "start_line": 278,
                "end_line": 281,
                "text_sha256": builder.text_sha256(text),
                "evidence_class": "R",
                "parent_claim_ids": [],
                "evidence": [],
                "runtime_evidence": [{"observed": "first occurrence"}],
            },
            {
                "claim_id": "old-second",
                "page": page,
                "start_line": 459,
                "end_line": 462,
                "text_sha256": builder.text_sha256(text),
                "evidence_class": None,
                "parent_claim_ids": [],
                "evidence": [],
                "runtime_evidence": [],
            },
        ]

        reconciled = builder.reconcile_volume_c_decisions(
            current_rows,
            old_decisions,
        )

        self.assertEqual(
            reconciled["new-first"]["runtime_evidence"],
            [{"observed": "first occurrence"}],
        )
        self.assertEqual(reconciled["new-first"]["evidence_class"], "R")
        self.assertIsNone(reconciled["new-second"]["evidence_class"])

    def test_volume_c_reconciliation_preserves_evidence_across_list_spacing_fix(
        self,
    ) -> None:
        page = (
            "wiki/02_engineering/01_ai_frameworks/"
            "19_torch_compile_end_to_end/19_buffer_liveness_memory_planning_and_reuse.md"
        )
        old_text = "-ordinary reuse is linear"
        current_text = "- ordinary reuse is linear"
        current_rows = [
            {
                "id": "new-list-claim",
                "page": page,
                "source_start_line": 245,
                "source_end_line": 245,
                "kind": "claim_candidate",
                "text": current_text,
            }
        ]
        old_decisions = [
            {
                "claim_id": "old-list-claim",
                "page": page,
                "start_line": 230,
                "end_line": 230,
                "text_sha256": builder.text_sha256(old_text),
                "_audited_text": old_text,
                "evidence_class": "R",
                "parent_claim_ids": [],
                "evidence": [],
                "runtime_evidence": [{"observed": "reuse"}],
            }
        ]

        reconciled = builder.reconcile_volume_c_decisions(
            current_rows,
            old_decisions,
        )

        self.assertEqual(
            reconciled["new-list-claim"]["runtime_evidence"],
            [{"observed": "reuse"}],
        )
        self.assertNotIn("_audited_text", reconciled["new-list-claim"])

    def test_volume_c_reconciliation_preserves_evidence_when_list_fix_splits_claim(
        self,
    ) -> None:
        page = (
            "wiki/02_engineering/01_ai_frameworks/"
            "19_torch_compile_end_to_end/19_buffer_liveness_memory_planning_and_reuse.md"
        )
        old_text = "- first complexity fact\n-second complexity fact"
        current_rows = [
            {
                "id": "new-first-fact",
                "page": page,
                "source_start_line": 242,
                "source_end_line": 242,
                "kind": "claim_candidate",
                "text": "- first complexity fact",
            },
            {
                "id": "new-second-fact",
                "page": page,
                "source_start_line": 243,
                "source_end_line": 243,
                "kind": "claim_candidate",
                "text": "- second complexity fact",
            },
        ]
        old_decisions = [
            {
                "claim_id": "old-combined-facts",
                "page": page,
                "start_line": 226,
                "end_line": 227,
                "text_sha256": builder.text_sha256(old_text),
                "_audited_text": old_text,
                "evidence_class": "R",
                "parent_claim_ids": [],
                "evidence": [],
                "runtime_evidence": [{"observed": "both facts"}],
            }
        ]

        reconciled = builder.reconcile_volume_c_decisions(
            current_rows,
            old_decisions,
        )

        self.assertEqual(
            set(reconciled),
            {"new-first-fact", "new-second-fact"},
        )
        self.assertTrue(
            all(
                decision["runtime_evidence"] == [{"observed": "both facts"}]
                for decision in reconciled.values()
            )
        )


if __name__ == "__main__":
    unittest.main()
