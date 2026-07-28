from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from docs.audits.pytorch_graph_series.tools.course_claim_ledger import (
    build_course_claim_rows,
    merge_course_claim_ledger,
    reconcile_course_claim_decisions,
    validate_source_checkout,
    validate_course_claim_decisions,
    write_course_claim_ledger,
)


PINNED = "e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _claim(claim_id: str, text: str, line: int = 1) -> dict[str, object]:
    return {
        "id": claim_id,
        "page": "wiki/course/01_page.md",
        "source_start_line": line,
        "source_end_line": line,
        "text": text,
    }


def _decision(
    claim_id: str,
    text: str,
    *,
    evidence_class: str | None = "S",
    content_class: str = "assertion",
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "claim_id": claim_id,
        "page": "wiki/course/01_page.md",
        "start_line": 1,
        "end_line": 1,
        "text_sha256": _sha256(text),
        "content_class": content_class,
        "evidence_class": evidence_class,
        "status": "verified-current",
        "evidence": [
            {
                "path": "torch/fx/graph.py",
                "start_line": 1,
                "end_line": 1,
                "baseline": PINNED,
                "supports": "The source line directly supports this fixture claim.",
            }
        ],
        "parent_claim_ids": [],
        "runtime_evidence": [],
        "presentation": "authoritative",
        "notes": "fixture",
    }


class CourseClaimLedgerValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.source_root = self.root / "p"
        (self.source_root / "torch" / "fx").mkdir(parents=True)
        (self.source_root / "torch" / "fx" / "graph.py").write_text(
            "SUPPORTED = True\n", encoding="utf-8"
        )
        (self.root / "wiki" / "course").mkdir(parents=True)
        (self.root / "wiki" / "course" / "01_page.md").write_text(
            "# 01 · Page\n", encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_requires_exactly_one_decision_for_every_claim(self) -> None:
        claims = [_claim("a", "alpha"), _claim("b", "beta", line=2)]
        decisions = [_decision("a", "alpha"), _decision("a", "alpha")]

        errors = validate_course_claim_decisions(
            claims, decisions, self.root, self.source_root, PINNED
        )

        self.assertEqual(
            ["duplicate_claim_decision", "missing_claim_decision"],
            sorted(error["code"] for error in errors),
        )

    def test_rejects_unknown_evidence_class_and_stale_text_hash(self) -> None:
        claim = _claim("a", "alpha")
        decision = _decision("a", "alpha")
        decision["evidence_class"] = "N"
        decision["text_sha256"] = _sha256("old text")

        errors = validate_course_claim_decisions(
            [claim], [decision], self.root, self.source_root, PINNED
        )

        self.assertEqual(
            {"invalid_evidence_class", "text_hash_mismatch"},
            {error["code"] for error in errors},
        )

    def test_source_evidence_requires_pinned_existing_in_bounds_locator(self) -> None:
        claim = _claim("a", "alpha")
        decision = _decision("a", "alpha")
        source = decision["evidence"][0]
        source["baseline"] = "wrong"
        source["end_line"] = 2

        errors = validate_course_claim_decisions(
            [claim], [decision], self.root, self.source_root, PINNED
        )

        self.assertEqual(
            {"source_baseline_mismatch", "source_line_out_of_bounds"},
            {error["code"] for error in errors},
        )

    def test_source_evidence_rejects_broad_non_auditable_ranges(self) -> None:
        source = self.source_root / "torch" / "fx" / "graph.py"
        source.write_text(
            "".join(f"SUPPORTED_{line} = True\n" for line in range(1, 41)),
            encoding="utf-8",
        )
        claim = _claim("a", "alpha")
        decision = _decision("a", "alpha")
        decision["evidence"][0].update(start_line=1, end_line=31)

        errors = validate_course_claim_decisions(
            [claim], [decision], self.root, self.source_root, PINNED
        )

        self.assertEqual(
            ["source_range_too_wide"],
            [error["code"] for error in errors],
        )

    def test_runtime_evidence_requires_real_script_artifact_and_baseline(self) -> None:
        claim = _claim("a", "alpha")
        decision = _decision("a", "alpha", evidence_class="R")
        decision["evidence"] = []
        decision["runtime_evidence"] = [
            {
                "script": "labs/missing.py",
                "artifacts": ["labs/missing.json"],
                "runtime_baseline": {},
                "supports": "fixture",
            }
        ]

        errors = validate_course_claim_decisions(
            [claim], [decision], self.root, self.source_root, PINNED
        )

        self.assertEqual(
            {
                "runtime_artifact_check_missing",
                "runtime_artifact_missing",
                "runtime_baseline_incomplete",
                "runtime_command_missing",
                "runtime_script_missing",
                "runtime_script_hash_missing",
            },
            {error["code"] for error in errors},
        )

    def test_inference_requires_existing_parent_and_acyclic_chain(self) -> None:
        claims = [_claim("a", "alpha"), _claim("b", "beta", line=2)]
        first = _decision("a", "alpha", evidence_class="I")
        first["evidence"] = []
        first["parent_claim_ids"] = ["b"]
        second = _decision("b", "beta", evidence_class="I")
        second["start_line"] = 2
        second["end_line"] = 2
        second["evidence"] = []
        second["parent_claim_ids"] = ["a", "missing"]

        errors = validate_course_claim_decisions(
            claims, [first, second], self.root, self.source_root, PINNED
        )

        self.assertEqual(
            {
                "inference_cycle",
                "inference_parent_missing",
                "inference_support_missing",
            },
            {error["code"] for error in errors},
        )

    def test_nonassertive_has_no_evidence_class_and_blocked_is_qualified(self) -> None:
        claims = [_claim("a", "question?"), _claim("b", "blocked", line=2)]
        question = _decision(
            "a", "question?", evidence_class=None, content_class="question"
        )
        question.update(
            status="not-applicable",
            evidence=[],
            runtime_evidence=[],
            presentation="nonassertive",
        )
        blocked = _decision("b", "blocked", evidence_class="B")
        blocked.update(
            start_line=2,
            end_line=2,
            evidence=[],
            runtime_evidence=[],
            status="unresolved",
            presentation="authoritative",
            notes="Environment is blocked.",
        )

        errors = validate_course_claim_decisions(
            claims, [question, blocked], self.root, self.source_root, PINNED
        )

        self.assertEqual(
            [
                "runtime_evidence_missing",
                "blocked_presentation_not_qualified",
            ],
            [error["code"] for error in errors],
        )

    def test_evidence_class_status_contract_is_enforced(self) -> None:
        claims = [_claim("a", "source"), _claim("b", "blocked", line=2)]
        source = _decision("a", "source")
        source["status"] = "unresolved"
        blocked = _decision("b", "blocked", evidence_class="B")
        blocked.update(
            start_line=2,
            end_line=2,
            evidence=[],
            runtime_evidence=[],
            status="verified-current",
            presentation="blocked",
            notes="No compatible device.",
        )

        errors = validate_course_claim_decisions(
            claims, [source, blocked], self.root, self.source_root, PINNED
        )

        self.assertEqual(
            {
                "blocked_status_invalid",
                "runtime_evidence_missing",
                "verified_evidence_status_invalid",
            },
            {error["code"] for error in errors},
        )

    def test_merge_preserves_decision_evidence_and_writes_jsonl(self) -> None:
        claim = _claim("a", "alpha")
        decision = _decision("a", "alpha")

        ledger = merge_course_claim_ledger([claim], [decision])

        self.assertEqual("S", ledger[0]["evidence_class"])
        self.assertEqual("a", ledger[0]["claim_id"])
        with tempfile.TemporaryDirectory() as output_dir:
            output = Path(output_dir) / "ledger.jsonl"
            write_course_claim_ledger(output, ledger)
            parsed = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(ledger, parsed)

    def test_reconcile_rekeys_only_unique_unchanged_claim_text(self) -> None:
        current = [
            {
                **_claim("new-parent", "source", line=10),
                "kind": "claim_candidate",
            },
            {
                **_claim("new-child", "inference", line=20),
                "kind": "claim_candidate",
            },
            {
                **_claim("new-claim", "new text", line=30),
                "kind": "code_claim",
            },
        ]
        parent = _decision("old-parent", "source")
        parent.update(start_line=1, end_line=1)
        child = _decision("old-child", "inference", evidence_class="I")
        child.update(
            start_line=2,
            end_line=2,
            evidence=[
                {
                    "parent_claim_id": "old-parent",
                    "supports": (
                        "The source parent establishes the property used by this "
                        "specific derived inference."
                    ),
                }
            ],
            parent_claim_ids=["old-parent"],
        )
        changed = _decision("old-changed", "obsolete text")
        changed.update(start_line=3, end_line=3)

        migrated, stale, missing, errors = reconcile_course_claim_decisions(
            current, [parent, child, changed]
        )

        self.assertEqual(["old-changed"], stale)
        self.assertEqual(["new-claim"], missing)
        self.assertEqual(
            ["reconcile_unmatched_decision"],
            [error["code"] for error in errors],
        )
        by_id = {decision["claim_id"]: decision for decision in migrated}
        self.assertEqual({"new-parent", "new-child"}, set(by_id))
        self.assertEqual(10, by_id["new-parent"]["start_line"])
        self.assertEqual(
            ["new-parent"], by_id["new-child"]["parent_claim_ids"]
        )
        self.assertEqual(
            "new-parent",
            by_id["new-child"]["evidence"][0]["parent_claim_id"],
        )
        self.assertEqual("claim_candidate", by_id["new-parent"]["unit_kind"])

    def test_reconcile_refuses_duplicate_text_and_duplicate_target(self) -> None:
        duplicate_claims = [
            {
                **_claim("first", "same", line=1),
                "kind": "claim_candidate",
            },
            {
                **_claim("second", "same", line=2),
                "kind": "claim_candidate",
            },
        ]
        ambiguous = _decision("old", "same")
        migrated, stale, missing, errors = reconcile_course_claim_decisions(
            duplicate_claims, [ambiguous]
        )
        self.assertEqual([], migrated)
        self.assertEqual(["old"], stale)
        self.assertEqual(["first", "second"], missing)
        self.assertEqual("reconcile_ambiguous_text", errors[0]["code"])

        single_claim = [
            {
                **_claim("current", "same", line=1),
                "kind": "claim_candidate",
            }
        ]
        first = _decision("old-a", "same")
        second = _decision("old-b", "same")
        migrated, stale, missing, errors = reconcile_course_claim_decisions(
            single_claim, [first, second]
        )
        self.assertEqual(["current"], [item["claim_id"] for item in migrated])
        self.assertEqual(["old-b"], stale)
        self.assertEqual([], missing)
        self.assertEqual("reconcile_duplicate_target", errors[0]["code"])

    def test_source_checkout_requires_exact_clean_pinned_commit(self) -> None:
        subprocess.run(
            ["git", "init", "--quiet", str(self.source_root)],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(self.source_root),
                "config",
                "user.email",
                "fixture@example.invalid",
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.source_root), "config", "user.name", "Fixture"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.source_root), "add", "torch/fx/graph.py"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.source_root), "commit", "--quiet", "-m", "fixture"],
            check=True,
        )
        head = subprocess.run(
            ["git", "-C", str(self.source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        self.assertEqual([], validate_source_checkout(self.source_root, head))
        mismatch = validate_source_checkout(self.source_root, "0" * 40)
        self.assertEqual(
            ["source_checkout_commit_mismatch"],
            [item["code"] for item in mismatch],
        )

        (self.source_root / "torch" / "fx" / "graph.py").write_text(
            "SUPPORTED = False\n", encoding="utf-8"
        )
        dirty = validate_source_checkout(self.source_root, head)
        self.assertEqual(["source_checkout_dirty"], [item["code"] for item in dirty])

    def test_course_claim_rows_include_atomic_table_rows_and_code_fences(self) -> None:
        page = self.root / "wiki" / "course" / "01_page.md"
        page.write_text(
            "# 01 · Page\n\n"
            "A prose assertion.\n\n"
            "| Kind | Meaning |\n"
            "|---|---|\n"
            "| FX | program graph |\n"
            "| AOT | forward/backward graphs |\n\n"
            "```text\n"
            "fw(x) -> (y, saved_x)\n"
            "```\n",
            encoding="utf-8",
        )
        manifest = self.root / "manifest.json"
        manifest.write_text(
            json.dumps(["wiki/course/01_page.md"]),
            encoding="utf-8",
        )

        rows = build_course_claim_rows(self.root, self.source_root, manifest)

        self.assertEqual(
            ["claim_candidate", "table_row_claim", "table_row_claim", "code_claim"],
            [row["kind"] for row in rows],
        )
        self.assertEqual([3, 7, 8, 10], [row["source_start_line"] for row in rows])
        self.assertEqual(4, len({row["id"] for row in rows}))
        self.assertIn("program graph", rows[1]["text"])
        self.assertIn("saved_x", rows[3]["text"])

    def test_inference_requires_claim_specific_parent_rationales(self) -> None:
        claims = [_claim("parent", "source"), _claim("child", "inference", line=2)]
        parent = _decision("parent", "source")
        child = _decision("child", "inference", evidence_class="I")
        child.update(
            start_line=2,
            end_line=2,
            evidence=[
                {
                    "parent_claim_id": "parent",
                    "supports": (
                        "The parent establishes the fresh-graph construction; "
                        "the child only derives the resulting identity boundary."
                    ),
                }
            ],
            parent_claim_ids=["parent"],
        )

        errors = validate_course_claim_decisions(
            claims, [parent, child], self.root, self.source_root, PINNED
        )

        self.assertEqual([], errors)

    def test_runtime_artifact_check_binds_hash_pointer_and_observed_value(self) -> None:
        script = self.root / "labs" / "run.py"
        script.parent.mkdir()
        script.write_text("print('ok')\n", encoding="utf-8")
        artifact = self.root / "labs" / "result.json"
        command = ["python", str(script.resolve())]
        script_sha256 = hashlib.sha256(script.read_bytes()).hexdigest()
        artifact.write_text(
            json.dumps(
                {
                    "producer": {
                        "script": "labs/run.py",
                        "script_sha256": script_sha256,
                        "command": command,
                        "exit_code": 0,
                    },
                    "result": {"matches": True},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        claim = _claim("a", "runtime")
        decision = _decision("a", "runtime", evidence_class="R")
        decision["evidence"] = []
        decision["runtime_evidence"] = [
            {
                "script": "labs/run.py",
                "script_sha256": script_sha256,
                "command": command,
                "artifacts": ["labs/result.json"],
                "artifact_checks": [
                    {
                        "role": "producer_script",
                        "path": "labs/result.json",
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                        "selector": "/producer/script",
                        "observed": "labs/run.py",
                        "supports": "The artifact identifies the script that produced this run.",
                    },
                    {
                        "role": "producer_script_sha256",
                        "path": "labs/result.json",
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                        "selector": "/producer/script_sha256",
                        "observed": script_sha256,
                        "supports": "The artifact binds the producer to the reviewed script bytes.",
                    },
                    {
                        "role": "producer_command",
                        "path": "labs/result.json",
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                        "selector": "/producer/command",
                        "observed": command,
                        "supports": "The artifact records the exact tokenized execution command.",
                    },
                    {
                        "role": "producer_exit_code",
                        "path": "labs/result.json",
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                        "selector": "/producer/exit_code",
                        "observed": 0,
                        "supports": "The producing command completed successfully with exit code zero.",
                    },
                    {
                        "path": "labs/result.json",
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                        "selector": "/result/matches",
                        "observed": True,
                        "supports": "The executed result records the claimed equality.",
                    }
                ],
                "runtime_baseline": {
                    "python": "fixture",
                    "torch": "fixture",
                    "torch_git": "fixture",
                    "platform": "fixture",
                    "cuda_available": False,
                },
                "supports": "The fixture command produced the checked result artifact.",
            }
        ]

        errors = validate_course_claim_decisions(
            [claim], [decision], self.root, self.source_root, PINNED
        )

        self.assertEqual([], errors)

    def test_runtime_artifact_check_rejects_stale_or_non_runtime_evidence(self) -> None:
        script = self.root / "labs" / "run.py"
        script.parent.mkdir()
        script.write_text("print('ok')\n", encoding="utf-8")
        artifact = self.root / "labs" / "result.json"
        command = ["python", str(script.resolve())]
        script_sha256 = hashlib.sha256(script.read_bytes()).hexdigest()
        artifact.write_text(
            json.dumps(
                {
                    "producer": {
                        "script": "labs/run.py",
                        "script_sha256": script_sha256,
                        "command": command,
                        "exit_code": 0,
                    },
                    "result": {"matches": True},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        documentation = self.root / "labs" / "README.md"
        documentation.write_text("# Not runtime output\n", encoding="utf-8")
        claim = _claim("a", "runtime")
        valid = _decision("a", "runtime", evidence_class="R")
        valid["evidence"] = []
        valid["runtime_evidence"] = [
            {
                "script": "labs/run.py",
                "script_sha256": script_sha256,
                "command": command,
                "artifacts": ["labs/result.json"],
                "artifact_checks": [
                    {
                        "role": "producer_script",
                        "path": "labs/result.json",
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                        "selector": "/producer/script",
                        "observed": "labs/run.py",
                        "supports": "The artifact identifies the script that produced this run.",
                    },
                    {
                        "role": "producer_script_sha256",
                        "path": "labs/result.json",
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                        "selector": "/producer/script_sha256",
                        "observed": script_sha256,
                        "supports": "The artifact binds the producer to the reviewed script bytes.",
                    },
                    {
                        "role": "producer_command",
                        "path": "labs/result.json",
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                        "selector": "/producer/command",
                        "observed": command,
                        "supports": "The artifact records the exact tokenized execution command.",
                    },
                    {
                        "role": "producer_exit_code",
                        "path": "labs/result.json",
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                        "selector": "/producer/exit_code",
                        "observed": 0,
                        "supports": "The producing command completed successfully with exit code zero.",
                    },
                    {
                        "path": "labs/result.json",
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                        "selector": "/result/matches",
                        "observed": True,
                        "supports": "The selected result directly records this runtime claim.",
                    }
                ],
                "runtime_baseline": {
                    "python": "fixture",
                    "torch": "fixture",
                    "torch_git": "fixture",
                    "platform": "fixture",
                    "cuda_available": False,
                },
                "supports": "The fixture command generated the checked JSON observation.",
            }
        ]

        stale_hash = deepcopy(valid)
        stale_hash["runtime_evidence"][0]["artifact_checks"][0]["sha256"] = "0" * 64
        stale_script = deepcopy(valid)
        stale_script["runtime_evidence"][0]["script_sha256"] = "0" * 64
        wrong_command = deepcopy(valid)
        wrong_command["runtime_evidence"][0]["command"] = ["python", "other.py"]
        missing_producer = deepcopy(valid)
        missing_producer["runtime_evidence"][0]["artifact_checks"] = [
            check
            for check in missing_producer["runtime_evidence"][0]["artifact_checks"]
            if check.get("role") != "producer_script"
        ]
        missing_pointer = deepcopy(valid)
        missing_pointer["runtime_evidence"][0]["artifact_checks"][0][
            "selector"
        ] = "/result/missing"
        stale_observation = deepcopy(valid)
        stale_observation["runtime_evidence"][0]["artifact_checks"][0][
            "observed"
        ] = False
        markdown = deepcopy(valid)
        markdown["runtime_evidence"][0]["artifacts"] = ["labs/README.md"]
        markdown["runtime_evidence"][0]["artifact_checks"][0].update(
            path="labs/README.md",
            sha256=hashlib.sha256(documentation.read_bytes()).hexdigest(),
            selector="/ignored",
        )

        cases = [
            (stale_hash, "runtime_artifact_hash_mismatch"),
            (stale_script, "runtime_script_hash_mismatch"),
            (wrong_command, "runtime_command_script_mismatch"),
            (missing_producer, "runtime_producer_script_check_missing"),
            (missing_pointer, "runtime_artifact_selector_missing"),
            (stale_observation, "runtime_artifact_observation_mismatch"),
            (markdown, "runtime_artifact_not_generated"),
        ]
        for decision, expected_code in cases:
            with self.subTest(expected_code=expected_code):
                errors = validate_course_claim_decisions(
                    [claim], [decision], self.root, self.source_root, PINNED
                )
                self.assertIn(expected_code, {error["code"] for error in errors})


if __name__ == "__main__":
    unittest.main()
