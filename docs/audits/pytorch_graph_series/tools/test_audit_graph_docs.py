from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from docs.audits.pytorch_graph_series.tools import audit_graph_docs

audit_manifest = audit_graph_docs.audit_manifest
extract_locators = audit_graph_docs.extract_locators
load_semantic_decisions = audit_graph_docs.load_semantic_decisions
parse_markdown = audit_graph_docs.parse_markdown
resolve_locator = audit_graph_docs.resolve_locator
write_coverage_ledger = audit_graph_docs.write_coverage_ledger


class ParseMarkdownTests(unittest.TestCase):
    def write_temp_markdown(self, text: str) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        temp_dir = tempfile.TemporaryDirectory()
        path = Path(temp_dir.name) / "page.md"
        path.write_text(text, encoding="utf-8")
        return temp_dir, path

    def test_nested_headings_fences_links_and_locators(self) -> None:
        temp_dir, path = self.write_temp_markdown(
            "\n".join(
                [
                    "# Title",
                    "## Section",
                    "Text [[target#anchor|label]] `torch/fx/graph.py:10-20`.",
                    "### Detail",
                    "```python",
                    "print('x')",
                    "```",
                    "```mermaid",
                    "flowchart LR",
                    '    A["x"] --> B["y"]',
                    "```",
                ]
            )
        )
        self.addCleanup(temp_dir.cleanup)

        records = parse_markdown(path)
        headings = [record for record in records if record["kind"] == "heading"]
        fences = [record for record in records if record["kind"] == "code_fence"]
        links = [record for record in records if record["kind"] == "wikilink"]
        locators = [record for record in records if record["kind"] == "locator"]

        self.assertEqual(
            [record["heading_path"] for record in headings],
            [["Section"], ["Section", "Detail"]],
        )
        self.assertEqual([record["language"] for record in fences], ["python", "mermaid"])
        self.assertTrue(all(record["balanced"] for record in fences))
        self.assertEqual(fences[1]["figure_classification"], "mermaid")
        self.assertEqual(links[0]["target"], "target")
        self.assertEqual(links[0]["anchor"], "anchor")
        self.assertEqual(links[0]["label"], "label")
        self.assertEqual(locators[0]["path"], "torch/fx/graph.py")
        self.assertEqual(locators[0]["start_line"], 10)
        self.assertEqual(locators[0]["end_line"], 20)

    def test_unbalanced_fence_is_reported(self) -> None:
        temp_dir, path = self.write_temp_markdown(
            "# Title\n## Section\n```python\nprint('missing close')\n"
        )
        self.addCleanup(temp_dir.cleanup)

        fences = [
            record for record in parse_markdown(path) if record["kind"] == "code_fence"
        ]
        self.assertEqual(len(fences), 1)
        self.assertFalse(fences[0]["balanced"])
        self.assertEqual(fences[0]["start_line"], 3)
        self.assertEqual(fences[0]["end_line"], 4)

    def test_table_with_escaped_pipe_is_one_table_and_preserves_cells(self) -> None:
        temp_dir, path = self.write_temp_markdown(
            "\n".join(
                [
                    "# Title",
                    "## Operators",
                    "| op | `\\|=` |",
                    "| --- | --- |",
                    "| merge | value |",
                ]
            )
        )
        self.addCleanup(temp_dir.cleanup)

        tables = [
            record for record in parse_markdown(path) if record["kind"] == "markdown_table"
        ]

        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0]["start_line"], 3)
        self.assertEqual(tables[0]["end_line"], 5)
        self.assertEqual(tables[0]["header_cells"], ["op", "`\\|=`"])
        self.assertEqual(tables[0]["column_count"], 2)
        self.assertEqual(tables[0]["data_row_count"], 1)
        self.assertEqual(tables[0]["data_rows"], [["merge", "value"]])
        self.assertEqual(
            tables[0]["raw"],
            "| op | `\\|=` |\n| --- | --- |\n| merge | value |",
        )

    def test_pipe_prose_and_pipe_inside_fence_are_not_markdown_tables(self) -> None:
        temp_dir, path = self.write_temp_markdown(
            "\n".join(
                [
                    "# Title",
                    "## Notes",
                    "A | B is prose, not a table.",
                    "```text",
                    "| fake | table |",
                    "| --- | --- |",
                    "```",
                ]
            )
        )
        self.addCleanup(temp_dir.cleanup)

        records = parse_markdown(path)
        self.assertFalse(any(record["kind"] == "markdown_table" for record in records))
        self.assertTrue(
            any(
                record["kind"] == "claim_candidate"
                and record["text"] == "A | B is prose, not a table."
                for record in records
            )
        )

    def test_inline_and_reference_images_are_inventory_units(self) -> None:
        temp_dir, path = self.write_temp_markdown(
            "\n".join(
                [
                    "# Title",
                    "## Figures",
                    "![inline diagram](assets/inline.png \"Inline\")",
                    "![reference diagram][fig-one]",
                    "[fig-one]: assets/reference.svg \"Reference\"",
                ]
            )
        )
        self.addCleanup(temp_dir.cleanup)

        images = [record for record in parse_markdown(path) if record["kind"] == "image"]

        self.assertEqual(
            [(record["syntax"], record["alt"], record["target"]) for record in images],
            [
                ("inline", "inline diagram", "assets/inline.png"),
                ("reference", "reference diagram", "assets/reference.svg"),
            ],
        )

    def test_multiple_inline_images_have_distinct_columns_and_keep_prose(self) -> None:
        temp_dir, path = self.write_temp_markdown(
            "\n".join(
                [
                    "# Title",
                    "## Figures",
                    "Before ![first](a.png) between ![second](b.png) after.",
                ]
            )
        )
        self.addCleanup(temp_dir.cleanup)

        records = parse_markdown(path)
        images = [record for record in records if record["kind"] == "image"]
        claims = [record for record in records if record["kind"] == "claim_candidate"]

        self.assertEqual(
            [
                (
                    image["target"],
                    image["source_line"],
                    image["source_start_column"],
                    image["source_end_column"],
                )
                for image in images
            ],
            [
                ("a.png", 3, 8, 22),
                ("b.png", 3, 32, 47),
            ],
        )
        self.assertEqual(
            [(claim["start_line"], claim["end_line"], claim["text"]) for claim in claims],
            [(3, 3, "Before ![first](a.png) between ![second](b.png) after.")],
        )
        image_rows = audit_graph_docs.build_ledger_rows(
            [
                {
                    "page": "wiki/page.md",
                    "page_sha256": "a" * 64,
                    **image,
                }
                for image in images
            ]
        )
        self.assertEqual(len({row["id"] for row in image_rows}), 2)

    def test_thematic_breaks_are_not_claim_candidates(self) -> None:
        temp_dir, path = self.write_temp_markdown(
            "# Title\n## Section\n---\n***\n___\nActual claim.\n"
        )
        self.addCleanup(temp_dir.cleanup)

        claims = [
            record for record in parse_markdown(path) if record["kind"] == "claim_candidate"
        ]

        self.assertEqual(
            [(claim["start_line"], claim["end_line"], claim["text"]) for claim in claims],
            [(6, 6, "Actual claim.")],
        )

    def test_spaced_thematic_breaks_and_tilde_fences_follow_markdown_blocks(self) -> None:
        temp_dir, path = self.write_temp_markdown(
            "\n".join(
                [
                    "# Title",
                    "## Evidence",
                    "- - -",
                    "* * *",
                    "_ _ _",
                    "~~~python",
                    "print('inside')",
                    "~~~",
                    "```text",
                    "```not-a-closing-fence",
                    "```",
                ]
            )
        )
        self.addCleanup(temp_dir.cleanup)

        records = parse_markdown(path)
        claims = [record for record in records if record["kind"] == "claim_candidate"]
        fences = [record for record in records if record["kind"] == "code_fence"]

        self.assertEqual(claims, [])
        self.assertEqual([record["language"] for record in fences], ["python", "text"])
        self.assertEqual(fences[0]["content"], "print('inside')")
        self.assertEqual(fences[1]["content"], "```not-a-closing-fence")
        self.assertTrue(all(record["balanced"] for record in fences))

    def test_list_items_are_atomic_claims_with_continuations_and_navigation(self) -> None:
        temp_dir, path = self.write_temp_markdown(
            "\n".join(
                [
                    "# Title",
                    "## Claims",
                    "1. First claim.",
                    "   Its continuation belongs to the first claim.",
                    "2. Second claim.",
                    "- [[graph_index]]",
                    "- [Local contents](#claims)",
                ]
            )
        )
        self.addCleanup(temp_dir.cleanup)

        records = parse_markdown(path)
        claims = [record for record in records if record["kind"] == "claim_candidate"]
        navigation = [record for record in records if record["kind"] == "navigation"]

        self.assertEqual(
            [
                (record["start_line"], record["end_line"], record["text"])
                for record in claims
            ],
            [
                (
                    3,
                    4,
                    "1. First claim.\n   Its continuation belongs to the first claim.",
                ),
                (5, 5, "2. Second claim."),
            ],
        )
        self.assertEqual(
            [
                (record["start_line"], record["end_line"], record["text"])
                for record in navigation
            ],
            [
                (6, 6, "- [[graph_index]]"),
                (7, 7, "- [Local contents](#claims)"),
            ],
        )

    def test_table_cell_may_end_in_an_escaped_pipe(self) -> None:
        temp_dir, path = self.write_temp_markdown(
            "\n".join(
                [
                    "# Title",
                    "## Operators",
                    "| key | value\\|",
                    "| --- | --- |",
                    "| a | b |",
                ]
            )
        )
        self.addCleanup(temp_dir.cleanup)

        tables = [
            record for record in parse_markdown(path) if record["kind"] == "markdown_table"
        ]

        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0]["header_cells"], ["key", "value\\|"])
        self.assertEqual(tables[0]["data_rows"], [["a", "b"]])

    def test_locator_occurrences_keep_markdown_position_and_target_range(self) -> None:
        temp_dir, path = self.write_temp_markdown(
            "\n".join(
                [
                    "# Title",
                    "## Sources",
                    "torch/fx/graph.py:10 and torch/fx/graph.py:20.",
                    "torch/fx/graph.py:10 then torch/fx/graph.py:10.",
                    "torch/fx/graph.py:10.",
                ]
            )
        )
        self.addCleanup(temp_dir.cleanup)

        locators = [
            record for record in parse_markdown(path) if record["kind"] == "locator"
        ]

        self.assertEqual(
            [
                (
                    locator["source_line"],
                    locator["source_start_column"],
                    locator["source_end_column"],
                    locator["target_start_line"],
                    locator["target_end_line"],
                )
                for locator in locators
            ],
            [
                (3, 1, 20, 10, 10),
                (3, 26, 45, 20, 20),
                (4, 1, 20, 10, 10),
                (4, 27, 46, 10, 10),
                (5, 1, 20, 10, 10),
            ],
        )

    def test_lab_heading_creates_one_experiment_span(self) -> None:
        temp_dir, path = self.write_temp_markdown(
            "\n".join(
                [
                    "# Title",
                    "## Lab: fusion measurement",
                    "Set up the benchmark.",
                    "### Result",
                    "The result is stable.",
                    "## Follow-up",
                    "Normal prose.",
                ]
            )
        )
        self.addCleanup(temp_dir.cleanup)

        experiments = [
            record for record in parse_markdown(path) if record["kind"] == "experiment"
        ]

        self.assertEqual(len(experiments), 1)
        self.assertEqual(experiments[0]["heading"], "Lab: fusion measurement")
        self.assertEqual((experiments[0]["start_line"], experiments[0]["end_line"]), (2, 5))

    def test_prose_paragraphs_become_claim_candidates(self) -> None:
        temp_dir, path = self.write_temp_markdown(
            "\n".join(
                [
                    "# Title",
                    "## Evidence",
                    "The compiler applies this pass before lowering.",
                    "",
                    "This paragraph gives a second independently auditable claim.",
                    "```python",
                    "not a claim",
                    "```",
                ]
            )
        )
        self.addCleanup(temp_dir.cleanup)

        claims = [
            record for record in parse_markdown(path) if record["kind"] == "claim_candidate"
        ]

        self.assertEqual(
            [(claim["start_line"], claim["end_line"], claim["text"]) for claim in claims],
            [
                (3, 3, "The compiler applies this pass before lowering."),
                (5, 5, "This paragraph gives a second independently auditable claim."),
            ],
        )


class LocatorTests(unittest.TestCase):
    def test_extracts_full_shorthand_l_prefix_and_duplicates(self) -> None:
        locators = extract_locators(
            "See torch/fx/graph.py:10-20, runtime_wrappers.py:L30-L40, "
            "and torch/fx/graph.py:10-20 again."
        )
        self.assertEqual(
            [(item["path"], item["start_line"], item["end_line"]) for item in locators],
            [
                ("torch/fx/graph.py", 10, 20),
                ("runtime_wrappers.py", 30, 40),
                ("torch/fx/graph.py", 10, 20),
            ],
        )

    def test_resolve_full_path_and_leave_shorthand_for_manual_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_root = Path(temp_dir)
            source_file = source_root / "torch" / "fx" / "graph.py"
            source_file.parent.mkdir(parents=True)
            source_file.write_text("one\ntwo\nthree\n", encoding="utf-8")

            valid = resolve_locator(
                {
                    "raw": "torch/fx/graph.py:2-3",
                    "path": "torch/fx/graph.py",
                    "start_line": 2,
                    "end_line": 3,
                },
                source_root,
            )
            shorthand = resolve_locator(
                {
                    "raw": "graph.py:2",
                    "path": "graph.py",
                    "start_line": 2,
                    "end_line": 2,
                },
                source_root,
            )
            out_of_bounds = resolve_locator(
                {
                    "raw": "torch/fx/graph.py:4",
                    "path": "torch/fx/graph.py",
                    "start_line": 4,
                    "end_line": 4,
                },
                source_root,
            )

        self.assertEqual(valid["resolution"], "path_and_line_valid")
        self.assertEqual(shorthand["resolution"], "needs_manual_resolution")
        self.assertEqual(out_of_bounds["resolution"], "line_out_of_bounds")


class ManifestAuditTests(unittest.TestCase):
    def test_audit_manifest_enriches_locators_and_keeps_page_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo_root = root / "repo"
            source_root = root / "source"
            page = repo_root / "wiki" / "page.md"
            source = source_root / "torch" / "fx" / "graph.py"
            page.parent.mkdir(parents=True)
            source.parent.mkdir(parents=True)
            page.write_text(
                "# Page\n## Section\nSee torch/fx/graph.py:1 and [[other]].\n",
                encoding="utf-8",
            )
            source.write_text("line\n", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(["wiki/page.md"]),
                encoding="utf-8",
            )

            records = audit_manifest(repo_root, source_root, manifest)

        self.assertEqual(records[0]["kind"], "page")
        locator = next(record for record in records if record["kind"] == "locator")
        self.assertEqual(locator["resolution"], "path_and_line_valid")
        self.assertEqual(locator["page"], "wiki/page.md")

    def test_coverage_ledger_has_one_row_per_auditable_unit(self) -> None:
        records = [
            {"kind": "page", "page": "wiki/page.md", "source_line": 1},
            {
                "kind": "heading",
                "page": "wiki/page.md",
                "level": 2,
                "heading_path": ["Section"],
                "source_line": 2,
            },
            {
                "kind": "code_fence",
                "page": "wiki/page.md",
                "language": "python",
                "heading_path": ["Section"],
                "start_line": 3,
                "end_line": 5,
                "balanced": True,
            },
            {
                "kind": "locator",
                "page": "wiki/page.md",
                "heading_path": ["Section"],
                "source_line": 6,
                "raw": "torch/fx/graph.py:1",
                "resolution": "path_and_line_valid",
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "ledger.md"
            write_coverage_ledger(output, records)
            text = output.read_text(encoding="utf-8")

        self.assertEqual(text.count("| `wiki/page.md` |"), 3)
        self.assertIn("heading_h2", text)
        self.assertIn("code_python", text)
        self.assertIn("source_locator", text)

    def test_coverage_ledger_applies_line_scoped_semantic_decisions(self) -> None:
        records = [
            {
                "kind": "heading",
                "page": "wiki/page.md",
                "level": 2,
                "heading_path": ["Corrected section"],
                "source_line": 10,
            },
            {
                "kind": "locator",
                "page": "wiki/page.md",
                "heading_path": ["Corrected section"],
                "source_line": 12,
                "raw": "torch/fx/graph.py:1",
                "resolution": "path_and_line_valid",
            },
            {
                "kind": "heading",
                "page": "wiki/page.md",
                "level": 2,
                "heading_path": ["Pending section"],
                "source_line": 20,
            },
        ]
        decisions = [
            {
                "page": "wiki/page.md",
                "start_line": 10,
                "end_line": 19,
                "status": "corrected",
                "current_result": "semantic_correction_required",
                "destinations": [
                    {
                        "path": "wiki/new_page.md",
                        "anchor_text": "Replacement",
                        "anchor_occurrence": 1,
                    }
                ],
                "destination_role": "authoritative_import",
                "action": "rewrite",
                "notes": "F-001",
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "ledger.md"
            write_coverage_ledger(output, records, decisions)
            text = output.read_text(encoding="utf-8")

        corrected_row = next(
            line for line in text.splitlines() if "| 10 | heading_h2 |" in line
        )
        pending_row = next(
            line for line in text.splitlines() if "| 20 | heading_h2 |" in line
        )
        self.assertIn("| semantic_correction_required | corrected |", corrected_row)
        self.assertIn("wiki/new_page.md", corrected_row)
        self.assertIn("| authoritative_import | rewrite | F-001 |", corrected_row)
        self.assertIn("| not_semantically_audited | unresolved |", pending_row)

    def test_load_semantic_decisions_combines_multiple_files_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first.json"
            second = root / "second.json"
            first.write_text(
                json.dumps([{"page": "wiki/first.md", "start_line": 1, "end_line": 2}]),
                encoding="utf-8",
            )
            second.write_text(
                json.dumps([{"page": "wiki/second.md", "start_line": 3, "end_line": 4}]),
                encoding="utf-8",
            )

            decisions = load_semantic_decisions([first, second])

        self.assertEqual(
            [decision["page"] for decision in decisions],
            ["wiki/first.md", "wiki/second.md"],
        )

    def test_load_semantic_decisions_accepts_jsonl_and_json_array(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            jsonl_path = root / "claims.jsonl"
            json_path = root / "claims.json"
            jsonl_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "page": "wiki/first.md",
                                "start_line": 3,
                                "end_line": 3,
                                "status": "verified-current",
                            }
                        ),
                        json.dumps(
                            {
                                "page": "wiki/second.md",
                                "start_line": 5,
                                "end_line": 5,
                                "status": "corrected",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            json_path.write_text(
                json.dumps(
                    [
                        {
                            "page": "wiki/third.md",
                            "start_line": 7,
                            "end_line": 7,
                            "status": "unresolved",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            decisions = load_semantic_decisions([jsonl_path, json_path])

        self.assertEqual(
            [decision["page"] for decision in decisions],
            ["wiki/first.md", "wiki/second.md", "wiki/third.md"],
        )

    def test_inventory_summary_reports_new_structural_kinds(self) -> None:
        records = [
            {
                "kind": "page",
                "page": "wiki/page.md",
                "source_line": 1,
                "line_count": 20,
            },
            {"kind": "markdown_table", "page": "wiki/page.md"},
            {"kind": "image", "page": "wiki/page.md"},
            {"kind": "experiment", "page": "wiki/page.md"},
            {"kind": "claim_candidate", "page": "wiki/page.md"},
            {"kind": "navigation", "page": "wiki/page.md"},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "summary.md"
            audit_graph_docs._write_summary(output, records)
            text = output.read_text(encoding="utf-8")

        self.assertIn(
            "| Page | Lines | H2 | H3 | Code | Mermaid | Tables | Images | "
            "Experiments | Claims | Navigation | Locators | Links | Unbalanced |",
            text,
        )
        page_row = next(line for line in text.splitlines() if "`wiki/page.md`" in line)
        self.assertIn("| 1 | 1 | 1 | 1 | 1 |", page_row)


class CanonicalLedgerTests(unittest.TestCase):
    PAGE = "wiki/page.md"
    PAGE_HASH = "a" * 64

    def ledger_records(self) -> list[dict[str, object]]:
        return [
            {
                "kind": "heading",
                "page": self.PAGE,
                "page_sha256": self.PAGE_HASH,
                "level": 2,
                "heading_path": ["Evidence"],
                "source_line": 2,
            },
            {
                "kind": "claim_candidate",
                "page": self.PAGE,
                "page_sha256": self.PAGE_HASH,
                "heading_path": ["Evidence"],
                "start_line": 3,
                "end_line": 4,
                "text": "Claim with |, `code`, and a newline.\nSecond line.",
            },
        ]

    def test_jsonl_rows_round_trip_special_text(self) -> None:
        rows = audit_graph_docs.build_ledger_rows(self.ledger_records())
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "coverage_ledger.jsonl"
            audit_graph_docs.write_ledger_jsonl(path, rows)
            restored = audit_graph_docs.read_ledger_jsonl(path)

        self.assertEqual(restored, rows)
        claim_row = next(row for row in restored if row["kind"] == "claim_candidate")
        self.assertEqual(
            claim_row["text"], "Claim with |, `code`, and a newline.\nSecond line."
        )

    def test_duplicate_stable_ids_fail(self) -> None:
        duplicate = self.ledger_records()[0].copy()
        with self.assertRaisesRegex(ValueError, "duplicate stable ledger ID"):
            audit_graph_docs.build_ledger_rows([self.ledger_records()[0], duplicate])

    def test_locator_occurrences_have_distinct_ids_and_markdown_source_spans(self) -> None:
        common = {
            "kind": "locator",
            "page": self.PAGE,
            "page_sha256": self.PAGE_HASH,
            "heading_path": ["Evidence"],
            "raw": "torch/fx/graph.py:10",
            "path": "torch/fx/graph.py",
            "start_line": 10,
            "end_line": 10,
            "target_start_line": 10,
            "target_end_line": 10,
            "resolution": "path_and_line_valid",
        }
        records = [
            {
                **common,
                "source_line": 3,
                "source_start_column": 1,
                "source_end_column": 20,
            },
            {
                **common,
                "source_line": 3,
                "source_start_column": 27,
                "source_end_column": 46,
            },
            {
                **common,
                "source_line": 4,
                "source_start_column": 1,
                "source_end_column": 20,
            },
        ]

        rows = audit_graph_docs.build_ledger_rows(records)

        self.assertEqual(len({row["id"] for row in rows}), 3)
        self.assertEqual(
            [
                (
                    row["source_start_line"],
                    row["source_end_line"],
                    row["source_start_column"],
                    row["source_end_column"],
                    row["payload"]["target_start_line"],
                    row["payload"]["target_end_line"],
                )
                for row in rows
            ],
            [
                (3, 3, 1, 20, 10, 10),
                (3, 3, 27, 46, 10, 10),
                (4, 4, 1, 20, 10, 10),
            ],
        )

    def test_canonical_payload_preserves_each_auditable_record_kind(self) -> None:
        records = [
            {
                "kind": "heading",
                "page": self.PAGE,
                "page_sha256": self.PAGE_HASH,
                "level": 2,
                "title": "Evidence",
                "heading_path": ["Evidence"],
                "source_line": 2,
            },
            {
                "kind": "code_fence",
                "page": self.PAGE,
                "page_sha256": self.PAGE_HASH,
                "language": "python",
                "marker": "```",
                "figure_classification": None,
                "content": "print('|')\nprint('done')",
                "balanced": True,
                "heading_path": ["Evidence"],
                "start_line": 3,
                "end_line": 6,
            },
            {
                "kind": "markdown_table",
                "page": self.PAGE,
                "page_sha256": self.PAGE_HASH,
                "heading_path": ["Evidence"],
                "start_line": 7,
                "end_line": 9,
                "header_cells": ["op", "`\\|=`"],
                "delimiter_cells": ["---", "---"],
                "data_rows": [["merge", "value"]],
                "raw": "| op | `\\|=` |\n| --- | --- |\n| merge | value |",
                "column_count": 2,
                "data_row_count": 1,
            },
            {
                "kind": "image",
                "page": self.PAGE,
                "page_sha256": self.PAGE_HASH,
                "syntax": "inline",
                "alt": "graph",
                "target": "graph.png",
                "title": "Graph",
                "heading_path": ["Evidence"],
                "source_line": 10,
                "source_start_column": 1,
                "source_end_column": 27,
            },
            {
                "kind": "experiment",
                "page": self.PAGE,
                "page_sha256": self.PAGE_HASH,
                "heading": "Lab: benchmark",
                "heading_path": ["Lab: benchmark"],
                "start_line": 11,
                "end_line": 14,
            },
            {
                "kind": "locator",
                "page": self.PAGE,
                "page_sha256": self.PAGE_HASH,
                "heading_path": ["Evidence"],
                "source_line": 15,
                "source_start_column": 5,
                "source_end_column": 24,
                "raw": "torch/fx/graph.py:10",
                "path": "torch/fx/graph.py",
                "start_line": 10,
                "end_line": 10,
                "target_start_line": 10,
                "target_end_line": 10,
                "resolution": "path_and_line_valid",
                "resolved_path": "E:/pytorch/torch/fx/graph.py",
                "source_line_count": 100,
            },
            {
                "kind": "claim_candidate",
                "page": self.PAGE,
                "page_sha256": self.PAGE_HASH,
                "heading_path": ["Evidence"],
                "start_line": 16,
                "end_line": 16,
                "text": "Auditable prose.",
            },
        ]

        rows = audit_graph_docs.build_ledger_rows(records)
        payloads = {row["kind"]: row["payload"] for row in rows}
        table_row = next(row for row in rows if row["kind"] == "markdown_table")

        self.assertEqual(payloads["heading_h2"], {"level": 2, "title": "Evidence"})
        self.assertEqual(
            payloads["code_python"]["content"], "print('|')\nprint('done')"
        )
        self.assertEqual(
            payloads["markdown_table"]["data_rows"], [["merge", "value"]]
        )
        self.assertEqual(payloads["markdown_table"]["header_cells"], ["op", "`\\|=`"])
        self.assertEqual(payloads["image_inline"]["alt"], "graph")
        self.assertEqual(payloads["image_inline"]["target"], "graph.png")
        self.assertEqual(payloads["experiment"]["heading"], "Lab: benchmark")
        self.assertEqual(
            payloads["source_locator"]["resolved_path"],
            "E:/pytorch/torch/fx/graph.py",
        )
        self.assertEqual(payloads["source_locator"]["target_start_line"], 10)
        self.assertEqual(payloads["source_locator"]["target_end_line"], 10)
        self.assertEqual(payloads["claim_candidate"]["text"], "Auditable prose.")
        self.assertEqual(table_row["locator"], "")

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "ledger.jsonl"
            audit_graph_docs.write_ledger_jsonl(output, rows)
            restored = audit_graph_docs.read_ledger_jsonl(output)
        self.assertEqual(restored, rows)

    def test_write_and_read_reject_duplicate_canonical_ids(self) -> None:
        rows = audit_graph_docs.build_ledger_rows(self.ledger_records())
        duplicate_rows = [*rows, dict(rows[0])]
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "ledger.jsonl"
            with self.assertRaisesRegex(ValueError, "duplicate stable ledger ID"):
                audit_graph_docs.write_ledger_jsonl(output, duplicate_rows)
            output.write_text(
                "\n".join(json.dumps(row) for row in duplicate_rows) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate stable ledger ID"):
                audit_graph_docs.read_ledger_jsonl(output)

    def test_canonical_schema_is_validated_on_write_read_and_summary(self) -> None:
        malformed = {
            "id": "not-a-stable-id",
            "kind": "claim_candidate",
            "status": "not-a-status",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "ledger.jsonl"
            with self.assertRaisesRegex(ValueError, "invalid canonical ledger row"):
                audit_graph_docs.write_ledger_jsonl(output, [malformed])
            output.write_text(json.dumps(malformed) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid canonical ledger row"):
                audit_graph_docs.read_ledger_jsonl(output)
        with self.assertRaisesRegex(ValueError, "invalid canonical ledger row"):
            audit_graph_docs.summarize_ledger([malformed])

    def test_decision_provenance_distinguishes_explicit_unresolved(self) -> None:
        without_decision = audit_graph_docs.build_ledger_rows(self.ledger_records())
        explicit_unresolved = audit_graph_docs.build_ledger_rows(
            self.ledger_records(),
            claim_decisions=[
                {
                    "page": self.PAGE,
                    "start_line": 3,
                    "end_line": 4,
                    "status": "unresolved",
                }
            ],
        )
        claim_without = next(
            row for row in without_decision if row["kind"] == "claim_candidate"
        )
        claim_explicit = next(
            row for row in explicit_unresolved if row["kind"] == "claim_candidate"
        )

        self.assertFalse(claim_without["decision_applied"])
        self.assertIsNone(claim_without["decision_selector"])
        self.assertTrue(claim_explicit["decision_applied"])
        self.assertEqual(claim_explicit["decision_category"], "claim")
        self.assertEqual(
            audit_graph_docs.summarize_ledger(explicit_unresolved)["decision_totals"],
            {"assigned": 1, "unassigned": 1},
        )

        with self.assertRaisesRegex(ValueError, "unmatched claim decision"):
            audit_graph_docs.build_ledger_rows(
                self.ledger_records(),
                claim_decisions=[
                    {"claim_id": "0" * 64, "status": "verified-current"}
                ],
            )

    def test_broad_ranges_only_assign_headings_and_units_require_exact_selectors(self) -> None:
        records = [
            self.ledger_records()[0],
            {
                "kind": "markdown_table",
                "page": self.PAGE,
                "page_sha256": self.PAGE_HASH,
                "heading_path": ["Evidence"],
                "start_line": 5,
                "end_line": 7,
                "header_cells": ["key", "value"],
                "delimiter_cells": ["---", "---"],
                "data_rows": [["a", "b"]],
                "raw": "| key | value |\n|---|---|\n| a | b |",
                "column_count": 2,
                "data_row_count": 1,
            },
        ]
        range_decision = {
            "page": self.PAGE,
            "start_line": 1,
            "end_line": 10,
            "status": "verified-current",
        }
        ranged = audit_graph_docs.build_ledger_rows(
            records, decisions=[range_decision]
        )
        heading = next(row for row in ranged if row["kind"] == "heading_h2")
        table = next(row for row in ranged if row["kind"] == "markdown_table")

        self.assertEqual(heading["status"], "verified-current")
        self.assertTrue(heading["decision_applied"])
        self.assertEqual(table["status"], "unresolved")
        self.assertFalse(table["decision_applied"])

        exact = audit_graph_docs.build_ledger_rows(
            records,
            decisions=[range_decision],
            unit_decisions=[
                {
                    "unit_id": table["id"],
                    "status": "corrected",
                    "action": "rewrite",
                }
            ],
        )
        exact_table = next(row for row in exact if row["kind"] == "markdown_table")
        self.assertEqual(exact_table["status"], "corrected")
        self.assertEqual(exact_table["decision_category"], "unit")
        self.assertTrue(exact_table["decision_applied"])

        with self.assertRaisesRegex(ValueError, "unmatched unit decision"):
            audit_graph_docs.build_ledger_rows(
                records,
                unit_decisions=[
                    {"unit_id": "f" * 64, "status": "verified-current"}
                ],
            )

    def test_coverage_ledger_renders_id_span_and_claim_or_unit(self) -> None:
        rows = audit_graph_docs.build_ledger_rows(self.ledger_records())
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "coverage_ledger.md"
            write_coverage_ledger(output, rows)
            text = output.read_text(encoding="utf-8")

        header = next(line for line in text.splitlines() if line.startswith("| id |"))
        self.assertIn("| source_span |", header)
        self.assertIn("| claim_or_unit |", header)
        self.assertIn("3-4", text)
        self.assertIn("Claim with \\|, `code`, and a newline.<br/>Second line.", text)

    def test_overlapping_claim_decisions_fail(self) -> None:
        claim_id = audit_graph_docs.build_ledger_rows(self.ledger_records())[1]["id"]
        decisions = [
            {"claim_id": claim_id, "status": "verified-current"},
            {"claim_id": claim_id, "status": "corrected"},
        ]

        with self.assertRaisesRegex(ValueError, "overlapping claim decisions"):
            audit_graph_docs.build_ledger_rows(
                self.ledger_records(), claim_decisions=decisions
            )

    def test_claim_candidate_is_not_upgraded_by_broad_range_decision(self) -> None:
        rows = audit_graph_docs.build_ledger_rows(
            self.ledger_records(),
            decisions=[
                {
                    "page": self.PAGE,
                    "start_line": 1,
                    "end_line": 10,
                    "status": "verified-current",
                }
            ],
        )

        claim = next(row for row in rows if row["kind"] == "claim_candidate")
        self.assertEqual(claim["status"], "unresolved")

    def test_unresolved_range_is_candidate_only_and_does_not_assign_heading(self) -> None:
        rows = audit_graph_docs.build_ledger_rows(
            self.ledger_records(),
            decisions=[
                {
                    "page": self.PAGE,
                    "start_line": 1,
                    "end_line": 10,
                    "status": "unresolved",
                    "destinations": [
                        {
                            "path": "wiki/target.md",
                            "anchor_text": "Candidate",
                            "anchor_occurrence": 1,
                        }
                    ],
                    "destination_role": "candidate_only",
                }
            ],
        )

        heading = next(row for row in rows if row["kind"] == "heading_h2")
        self.assertEqual(heading["status"], "unresolved")
        self.assertFalse(heading["decision_applied"])
        self.assertEqual(heading["destinations"], [])
        self.assertEqual(heading["destination_role"], "unassigned")

    def test_summary_uses_canonical_rows_not_rendered_markdown(self) -> None:
        rows = audit_graph_docs.build_ledger_rows(
            self.ledger_records(),
            claim_decisions=[
                {
                    "page": self.PAGE,
                    "start_line": 3,
                    "end_line": 4,
                    "status": "verified-current",
                }
            ],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "coverage_ledger.md"
            write_coverage_ledger(output, rows)
            output.write_text("| forged | verified-current |\n", encoding="utf-8")

        self.assertEqual(
            audit_graph_docs.summarize_ledger(rows),
            {
                "rows": 2,
                "kind_totals": {"claim_candidate": 1, "heading_h2": 1},
                "status_totals": {"unresolved": 1, "verified-current": 1},
                "decision_totals": {"assigned": 1, "unassigned": 1},
            },
        )

    def test_cli_summary_totals_equal_canonical_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo_root = root / "repo"
            source_root = root / "source"
            page = repo_root / "wiki" / "page.md"
            page.parent.mkdir(parents=True)
            page.write_text(
                "# Page\n## Evidence\nThis is an auditable claim.\n",
                encoding="utf-8",
            )
            evidence_path = repo_root / "src" / "implementation.py"
            evidence_path.parent.mkdir(parents=True)
            evidence_path.write_text(
                "CLAIM_IS_SUPPORTED = True\n",
                encoding="utf-8",
            )
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(["wiki/page.md"]), encoding="utf-8")
            inventory_jsonl = root / "inventory.jsonl"
            jsonl = root / "ledger.jsonl"
            summary = root / "summary.md"
            markdown_ledger = root / "coverage_ledger.md"
            claim_decisions = root / "claim_decisions.jsonl"
            claim_row = next(
                row
                for row in audit_graph_docs.build_ledger_rows(
                    audit_graph_docs.audit_manifest(
                        repo_root,
                        source_root,
                        manifest,
                    )
                )
                if row["kind"] == "claim_candidate"
            )
            claim_decisions.write_text(
                json.dumps(
                    {
                        "claim_id": claim_row["id"],
                        "page": claim_row["page"],
                        "page_sha256": claim_row["page_sha256"],
                        "start_line": claim_row["source_start_line"],
                        "end_line": claim_row["source_end_line"],
                        "source_start_column": None,
                        "source_end_column": None,
                        "claim_text_sha256": hashlib.sha256(
                            str(claim_row["text"]).encode("utf-8")
                        ).hexdigest(),
                        "status": "verified-current",
                        "claimed_baseline": "unknown",
                        "current_result": "claim_specific_source_evidence",
                        "destinations": [
                            {
                                "path": "wiki/page.md",
                                "anchor_text": "Evidence",
                                "anchor_occurrence": 1,
                            }
                        ],
                        "destination_role": "authoritative_import",
                        "action": "migrate",
                        "blocker": "",
                        "correction_ids": [],
                        "range_context": [],
                        "notes": "Exact test evidence.",
                        "evidence": [
                            {
                                "kind": "source",
                                "path": "src/implementation.py",
                                "baseline": "test-baseline",
                                "start_line": 1,
                                "end_line": 1,
                                "sha256": hashlib.sha256(
                                    evidence_path.read_bytes()
                                ).hexdigest(),
                                "supports": (
                                    "The exact fixture claim is supported."
                                ),
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "docs.audits.pytorch_graph_series.tools.audit_graph_docs",
                    "--repo-root",
                    str(repo_root),
                    "--source-root",
                    str(source_root),
                    "--manifest",
                    str(manifest),
                    "--jsonl-output",
                    str(inventory_jsonl),
                    "--ledger-jsonl-output",
                    str(jsonl),
                    "--summary-output",
                    str(summary),
                    "--ledger-output",
                    str(markdown_ledger),
                    "--claim-decisions",
                    str(claim_decisions),
                ],
                capture_output=True,
                check=False,
                cwd=Path(__file__).resolve().parents[4],
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            cli_summary = json.loads(result.stdout)
            canonical_summary = audit_graph_docs.summarize_ledger(
                audit_graph_docs.read_ledger_jsonl(jsonl)
            )
            claim_row = next(
                row
                for row in audit_graph_docs.read_ledger_jsonl(jsonl)
                if row["kind"] == "claim_candidate"
            )
            inventory_rows = [
                json.loads(line)
                for line in inventory_jsonl.read_text(encoding="utf-8").splitlines()
                if line
            ]

        self.assertEqual(cli_summary["kind_totals"], canonical_summary["kind_totals"])
        self.assertEqual(cli_summary["status_totals"], canonical_summary["status_totals"])
        self.assertEqual(claim_row["status"], "verified-current")
        self.assertEqual(inventory_rows[0]["kind"], "page")
        self.assertFalse(any("status" in row for row in inventory_rows))

    def test_cli_applies_exact_unit_decisions_and_keeps_legacy_inventory_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo_root = root / "repo"
            source_root = root / "source"
            page = repo_root / "wiki" / "page.md"
            page.parent.mkdir(parents=True)
            page.write_text(
                "# Page\n## Evidence\n| key | value |\n|---|---|\n| a | b |\n",
                encoding="utf-8",
            )
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(["wiki/page.md"]), encoding="utf-8")
            inventory = root / "inventory.jsonl"
            ledger = root / "ledger.jsonl"
            summary = root / "summary.md"
            markdown_ledger = root / "coverage_ledger.md"
            unit_decisions = root / "unit_decisions.jsonl"
            unit_decisions.write_text(
                json.dumps(
                    {
                        "page": "wiki/page.md",
                        "start_line": 3,
                        "end_line": 5,
                        "kind": "markdown_table",
                        "status": "corrected",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "docs.audits.pytorch_graph_series.tools.audit_graph_docs",
                    "--repo-root",
                    str(repo_root),
                    "--source-root",
                    str(source_root),
                    "--manifest",
                    str(manifest),
                    "--jsonl-output",
                    str(inventory),
                    "--ledger-jsonl-output",
                    str(ledger),
                    "--summary-output",
                    str(summary),
                    "--ledger-output",
                    str(markdown_ledger),
                    "--unit-decisions",
                    str(unit_decisions),
                ],
                capture_output=True,
                check=False,
                cwd=Path(__file__).resolve().parents[4],
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            table = next(
                row
                for row in audit_graph_docs.read_ledger_jsonl(ledger)
                if row["kind"] == "markdown_table"
            )
            inventory_rows = [
                json.loads(line)
                for line in inventory.read_text(encoding="utf-8").splitlines()
                if line
            ]

        self.assertEqual(table["status"], "corrected")
        self.assertEqual(table["decision_category"], "unit")
        self.assertTrue(table["decision_applied"])
        self.assertEqual(inventory_rows[0]["kind"], "page")
        self.assertIn("markdown_table", {row["kind"] for row in inventory_rows})


class DestinationValidationTests(unittest.TestCase):
    COURSE_ROOT = (
        "wiki/02_engineering/01_ai_frameworks/"
        "19_torch_compile_end_to_end"
    )

    def make_repo(self) -> tuple[tempfile.TemporaryDirectory[str], Path, str]:
        temp_dir = tempfile.TemporaryDirectory()
        repo_root = Path(temp_dir.name)
        relative = f"{self.COURSE_ROOT}/01_graph_ir_motivation_and_taxonomy.md"
        target = repo_root / relative
        target.parent.mkdir(parents=True)
        target.write_text(
            "\n".join(
                [
                    "# Page",
                    "## Exact target",
                    "### Repeated target",
                    "## Other",
                    "### Repeated target",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return temp_dir, repo_root, relative

    @staticmethod
    def destination(path: str, anchor: str, occurrence: int | None = 1) -> dict[str, object]:
        result: dict[str, object] = {
            "path": path,
            "anchor_text": anchor,
        }
        if occurrence is not None:
            result["anchor_occurrence"] = occurrence
        return result

    def row(
        self,
        *,
        destinations: object,
        role: str = "authoritative_import",
        status: str = "corrected",
    ) -> dict[str, object]:
        return {
            "id": "row-1",
            "status": status,
            "destinations": destinations,
            "destination_role": role,
        }

    def test_valid_exact_heading_and_occurrence_pass(self) -> None:
        temp_dir, repo_root, relative = self.make_repo()
        self.addCleanup(temp_dir.cleanup)

        errors = audit_graph_docs.validate_destinations(
            [
                self.row(
                    destinations=[
                        self.destination(relative, "Exact target"),
                        self.destination(relative, "Repeated target", 2),
                    ]
                )
            ],
            repo_root,
        )

        self.assertEqual(errors, [])

    def test_missing_path_reports_precise_error(self) -> None:
        temp_dir, repo_root, _ = self.make_repo()
        self.addCleanup(temp_dir.cleanup)

        errors = audit_graph_docs.validate_destinations(
            [
                self.row(
                    destinations=[
                        self.destination(
                            f"{self.COURSE_ROOT}/99_missing.md",
                            "Exact target",
                        )
                    ]
                )
            ],
            repo_root,
        )

        self.assertEqual(errors[0]["code"], "missing_destination_path")
        self.assertEqual(errors[0]["row_index"], 0)
        self.assertEqual(errors[0]["destination_index"], 0)

    def test_missing_anchor_reports_precise_error(self) -> None:
        temp_dir, repo_root, relative = self.make_repo()
        self.addCleanup(temp_dir.cleanup)

        errors = audit_graph_docs.validate_destinations(
            [
                self.row(
                    destinations=[
                        self.destination(relative, "Does not exist")
                    ]
                )
            ],
            repo_root,
        )

        self.assertEqual(errors[0]["code"], "missing_destination_anchor")
        self.assertEqual(errors[0]["anchor_text"], "Does not exist")

    def test_duplicate_anchor_requires_occurrence(self) -> None:
        temp_dir, repo_root, relative = self.make_repo()
        self.addCleanup(temp_dir.cleanup)

        errors = audit_graph_docs.validate_destinations(
            [
                self.row(
                    destinations=[
                        self.destination(relative, "Repeated target", None)
                    ]
                )
            ],
            repo_root,
        )

        self.assertEqual(errors[0]["code"], "ambiguous_destination_anchor")
        self.assertEqual(errors[0]["occurrences"], 2)

    def test_unique_anchor_still_requires_explicit_occurrence(self) -> None:
        temp_dir, repo_root, relative = self.make_repo()
        self.addCleanup(temp_dir.cleanup)

        errors = audit_graph_docs.validate_destinations(
            [
                self.row(
                    destinations=[
                        self.destination(relative, "Exact target", None)
                    ]
                )
            ],
            repo_root,
        )

        self.assertEqual(errors[0]["code"], "missing_anchor_occurrence")

    def test_occurrence_must_be_positive_and_in_range(self) -> None:
        temp_dir, repo_root, relative = self.make_repo()
        self.addCleanup(temp_dir.cleanup)

        invalid = audit_graph_docs.validate_destinations(
            [
                self.row(
                    destinations=[
                        self.destination(relative, "Repeated target", 0),
                        self.destination(relative, "Repeated target", 3),
                    ]
                )
            ],
            repo_root,
        )

        self.assertEqual(
            [error["code"] for error in invalid],
            [
                "invalid_anchor_occurrence",
                "anchor_occurrence_out_of_range",
            ],
        )

    def test_bare_destination_string_is_rejected(self) -> None:
        temp_dir, repo_root, _ = self.make_repo()
        self.addCleanup(temp_dir.cleanup)

        errors = audit_graph_docs.validate_destinations(
            [
                {
                    "id": "row-1",
                    "status": "corrected",
                    "destination": "01_graph_ir_motivation_and_taxonomy.md",
                    "destination_role": "authoritative_import",
                }
            ],
            repo_root,
        )

        self.assertEqual(errors[0]["code"], "bare_destination_string")

    def test_unresolved_content_cannot_be_an_authoritative_course_import(self) -> None:
        temp_dir, repo_root, relative = self.make_repo()
        self.addCleanup(temp_dir.cleanup)
        destination = [self.destination(relative, "Exact target")]

        invalid = audit_graph_docs.validate_destinations(
            [
                self.row(
                    destinations=destination,
                    status="unresolved",
                    role="authoritative_import",
                )
            ],
            repo_root,
        )
        valid = audit_graph_docs.validate_destinations(
            [
                self.row(
                    destinations=destination,
                    status="unresolved",
                    role="candidate_only",
                )
            ],
            repo_root,
        )

        self.assertEqual(
            invalid[0]["code"], "unresolved_authoritative_target"
        )
        self.assertEqual(valid, [])

    def test_aliases_normalize_numbered_and_renamed_pages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "aliases.json"
            prefix = f"{self.COURSE_ROOT}/"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "aliases": {
                            "01_graph_ir_motivation_and_taxonomy.md": (
                                prefix
                                + "01_graph_ir_motivation_and_taxonomy.md"
                            ),
                            "12_fx_graph_editing_primitives.md": (
                                prefix
                                + "12_fx_graph_editing_primitives_and_invariants.md"
                            ),
                            "13_patternexpr_and_patternmatcher.md": (
                                prefix
                                + "13_pattern_expression_and_matcher_engine.md"
                            ),
                        },
                        "manual_split_required": {
                            "20_debugging_observability_and_verification_labs.md": {
                                "reason": "removed aggregate page"
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            aliases = audit_graph_docs.load_destination_aliases(path)

        self.assertEqual(
            audit_graph_docs.resolve_destination_alias(
                "01_graph_ir_motivation_and_taxonomy.md", aliases
            ),
            (
                f"{self.COURSE_ROOT}/"
                "01_graph_ir_motivation_and_taxonomy.md"
            ),
        )
        self.assertTrue(
            audit_graph_docs.resolve_destination_alias(
                "12_fx_graph_editing_primitives.md", aliases
            ).endswith("12_fx_graph_editing_primitives_and_invariants.md")
        )
        self.assertTrue(
            audit_graph_docs.resolve_destination_alias(
                "13_patternexpr_and_patternmatcher.md", aliases
            ).endswith("13_pattern_expression_and_matcher_engine.md")
        )
        with self.assertRaisesRegex(ValueError, "manual_split_required"):
            audit_graph_docs.resolve_destination_alias(
                "20_debugging_observability_and_verification_labs.md",
                aliases,
            )

    def test_repository_alias_table_covers_final_numbered_course(self) -> None:
        repository_root = Path(__file__).resolve().parents[4]
        alias_path = (
            repository_root
            / "docs/audits/pytorch_graph_series/2026-07-27"
            / "destination_aliases.json"
        )
        aliases = audit_graph_docs.load_destination_aliases(alias_path)
        mapping = aliases["aliases"]

        for number in range(22):
            prefix = f"{number:02}_"
            matching = [
                target
                for target in mapping.values()
                if Path(target).name.startswith(prefix)
            ]
            self.assertTrue(matching, prefix)
            self.assertTrue(all((repository_root / target).is_file() for target in matching))
        self.assertTrue(
            mapping["12_fx_graph_editing_primitives.md"].endswith(
                "12_fx_graph_editing_primitives_and_invariants.md"
            )
        )
        self.assertTrue(
            mapping["13_patternexpr_and_patternmatcher.md"].endswith(
                "13_pattern_expression_and_matcher_engine.md"
            )
        )
        self.assertNotIn(
            "20_debugging_observability_and_verification_labs.md",
            mapping,
        )

    def test_repository_range_decisions_have_valid_final_destinations(self) -> None:
        repository_root = Path(__file__).resolve().parents[4]
        decision_root = (
            repository_root
            / "docs/audits/pytorch_graph_series/2026-07-23"
        )
        decisions = audit_graph_docs.load_semantic_decisions(
            sorted(decision_root.glob("semantic_decisions_*.json"))
        )
        final_path = re.compile(
            r"^wiki/02_engineering/01_ai_frameworks/"
            r"19_torch_compile_end_to_end/"
            r"(?:0[0-9]|1[0-9]|2[01])_[^/]+\.md$"
        )

        self.assertEqual(len(decisions), 139)
        self.assertEqual(
            sum(len(decision["destinations"]) for decision in decisions),
            238,
        )
        self.assertFalse(
            any("destination" in decision for decision in decisions)
        )
        self.assertTrue(
            all(
                final_path.fullmatch(destination["path"])
                for decision in decisions
                for destination in decision["destinations"]
            )
        )
        self.assertTrue(
            all(
                decision["destination_role"] == "candidate_only"
                for decision in decisions
                if decision["status"] == "unresolved"
            )
        )
        self.assertEqual(
            audit_graph_docs.validate_destinations(
                decisions, repository_root
            ),
            [],
        )

    def test_repository_exact_units_close_all_nonclaim_structure(self) -> None:
        repository_root = Path(__file__).resolve().parents[4]
        audit_root = (
            repository_root
            / "docs/audits/pytorch_graph_series/2026-07-23"
        )
        migration_root = (
            repository_root
            / "docs/audits/pytorch_graph_series/2026-07-27"
        )
        records = audit_graph_docs.audit_manifest(
            repository_root,
            repository_root / "__missing_pytorch_source__",
            audit_root / "audit_manifest.json",
        )
        range_decisions = audit_graph_docs.load_semantic_decisions(
            sorted(audit_root.glob("semantic_decisions_*.json"))
        )
        unit_decisions = audit_graph_docs.load_semantic_decisions(
            [migration_root / "legacy_unit_decisions.jsonl"]
        )
        rows = audit_graph_docs.build_ledger_rows(
            records,
            range_decisions,
            unit_decisions=unit_decisions,
        )
        destination_errors = audit_graph_docs.validate_destinations(
            [*range_decisions, *unit_decisions],
            repository_root,
        )
        closure = audit_graph_docs.summarize_destination_closure(
            rows, destination_errors
        )

        self.assertEqual(len(unit_decisions), 1602)
        self.assertEqual(
            len({decision["unit_id"] for decision in unit_decisions}),
            len(unit_decisions),
        )
        self.assertTrue(
            all(
                decision["status"] == "unresolved"
                and decision["action"] == "retain-quarantined"
                and decision["destination_role"] == "retain_quarantined"
                for decision in unit_decisions
            )
        )
        self.assertEqual(destination_errors, [])
        self.assertEqual(closure["unassigned_structural_units"], 0)
        self.assertGreater(closure["pending_claim_candidates"], 0)

    def test_canonical_rows_use_destinations_only_and_reject_bare_decisions(self) -> None:
        records = [
            {
                "kind": "heading",
                "page": "wiki/page.md",
                "page_sha256": "a" * 64,
                "level": 2,
                "title": "Evidence",
                "heading_path": ["Evidence"],
                "source_line": 2,
            }
        ]
        rows = audit_graph_docs.build_ledger_rows(records)

        self.assertIn("destinations", rows[0])
        self.assertIn("destination_role", rows[0])
        self.assertNotIn("destination", rows[0])
        with self.assertRaisesRegex(ValueError, "bare destination"):
            audit_graph_docs.build_ledger_rows(
                records,
                decisions=[
                    {
                        "page": "wiki/page.md",
                        "start_line": 1,
                        "end_line": 3,
                        "status": "corrected",
                        "destination": "target.md",
                    }
                ],
            )

    def test_legacy_migration_is_idempotent_and_refuses_anchor_guessing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            target_path = "wiki/01_target.md"
            target = repo_root / target_path
            target.parent.mkdir(parents=True)
            target.write_text("# Page\n## Only target\n", encoding="utf-8")
            aliases = {
                "schema_version": 1,
                "aliases": {"old.md": target_path},
                "manual_split_required": {
                    "removed.md": {"reason": "semantic split"}
                },
            }
            legacy = [
                {
                    "page": "wiki/legacy.md",
                    "start_line": 1,
                    "end_line": 2,
                    "status": "corrected",
                    "destination": "old.md",
                    "action": "rewrite",
                }
            ]

            migrated = audit_graph_docs.migrate_legacy_destination_decisions(
                legacy, aliases, repo_root
            )
            migrated_twice = (
                audit_graph_docs.migrate_legacy_destination_decisions(
                    migrated, aliases, repo_root
                )
            )

            self.assertEqual(migrated_twice, migrated)
            self.assertNotIn("destination", migrated[0])
            self.assertEqual(
                migrated[0]["destinations"],
                [
                    {
                        "path": target_path,
                        "anchor_text": "Only target",
                        "anchor_occurrence": 1,
                    }
                ],
            )
            self.assertEqual(
                migrated[0]["destination_role"], "authoritative_import"
            )

            target.write_text(
                "# Page\n## First possible target\n## Second possible target\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "ambiguous legacy destination"
            ):
                audit_graph_docs.migrate_legacy_destination_decisions(
                    legacy, aliases, repo_root
                )
            with self.assertRaisesRegex(ValueError, "manual_split_required"):
                audit_graph_docs.migrate_legacy_destination_decisions(
                    [
                        {
                            **legacy[0],
                            "destination": "removed.md",
                        }
                    ],
                    aliases,
                    repo_root,
                )

    def test_legacy_migration_subcommand_writes_canonical_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo_root = root / "repo"
            target = repo_root / "wiki/target.md"
            target.parent.mkdir(parents=True)
            target.write_text("# Page\n## Exact target\n", encoding="utf-8")
            aliases = root / "aliases.json"
            aliases.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "aliases": {"old.md": "wiki/target.md"},
                        "manual_split_required": {},
                    }
                ),
                encoding="utf-8",
            )
            source = root / "legacy.json"
            source.write_text(
                json.dumps(
                    [
                        {
                            "page": "wiki/legacy.md",
                            "start_line": 1,
                            "end_line": 2,
                            "status": "corrected",
                            "destination": "old.md",
                            "action": "rewrite",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            first_output = root / "migrated.json"
            second_output = root / "migrated_twice.json"
            command = [
                sys.executable,
                "-m",
                "docs.audits.pytorch_graph_series.tools.audit_graph_docs",
                "migrate-destinations",
                "--repo-root",
                str(repo_root),
                "--aliases",
                str(aliases),
                "--input",
                str(source),
                "--output",
                str(first_output),
            ]
            first = subprocess.run(
                command,
                capture_output=True,
                check=False,
                cwd=Path(__file__).resolve().parents[4],
                text=True,
            )
            command[command.index(str(source))] = str(first_output)
            command[command.index(str(first_output), command.index("--output"))] = (
                str(second_output)
            )
            second = subprocess.run(
                command,
                capture_output=True,
                check=False,
                cwd=Path(__file__).resolve().parents[4],
                text=True,
            )

            first_rows = json.loads(first_output.read_text(encoding="utf-8"))
            second_rows = json.loads(second_output.read_text(encoding="utf-8"))

        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(first_rows, second_rows)
        self.assertNotIn("destination", first_rows[0])

    def test_destination_closure_keeps_claims_pending_separately(self) -> None:
        rows = [
            {
                "kind": "heading_h2",
                "decision_applied": True,
                "destinations": [
                    self.destination("wiki/page.md", "Heading")
                ],
            },
            {
                "kind": "markdown_table",
                "decision_applied": False,
                "destinations": [],
            },
            {
                "kind": "claim_candidate",
                "decision_applied": False,
                "destinations": [],
            },
        ]

        summary = audit_graph_docs.summarize_destination_closure(rows, [])

        self.assertEqual(summary["unassigned_structural_units"], 1)
        self.assertEqual(summary["pending_claim_candidates"], 1)
        self.assertEqual(summary["bare_destination_strings"], 0)

    def test_cli_validates_candidate_decisions_even_when_range_is_not_applied(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo_root = root / "repo"
            source_root = root / "source"
            source_root.mkdir()
            relative = "wiki/page.md"
            page = repo_root / relative
            page.parent.mkdir(parents=True)
            page.write_text("# Page\n## Exact target\nText.\n", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps([relative]), encoding="utf-8")
            decisions = root / "decisions.json"
            decisions.write_text(
                json.dumps(
                    [
                        {
                            "page": relative,
                            "start_line": 1,
                            "end_line": 3,
                            "status": "unresolved",
                            "destinations": [
                                self.destination(relative, "Exact target")
                            ],
                            "destination_role": "authoritative_import",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            errors_output = root / "destination_errors.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "docs.audits.pytorch_graph_series.tools.audit_graph_docs",
                    "--repo-root",
                    str(repo_root),
                    "--source-root",
                    str(source_root),
                    "--manifest",
                    str(manifest),
                    "--jsonl-output",
                    str(root / "inventory.jsonl"),
                    "--summary-output",
                    str(root / "summary.md"),
                    "--ledger-output",
                    str(root / "ledger.md"),
                    "--decisions",
                    str(decisions),
                    "--destination-errors-output",
                    str(errors_output),
                ],
                capture_output=True,
                check=False,
                cwd=Path(__file__).resolve().parents[4],
                text=True,
            )

            summary = json.loads(result.stdout)
            errors = json.loads(errors_output.read_text(encoding="utf-8"))

        self.assertEqual(result.returncode, 2)
        self.assertEqual(summary["destination_validation_errors"], 1)
        self.assertEqual(
            errors[0]["code"], "unresolved_authoritative_target"
        )


class LegacyClaimAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.repo_root = self.root / "repo"
        self.source_root = self.root / "source"
        self.source_root.mkdir(parents=True)
        self.legacy_relative = "wiki/legacy.md"
        self.target_relative = "wiki/target.md"
        self.report_relative = "docs/corrections_foundations.md"
        legacy = self.repo_root / self.legacy_relative
        legacy.parent.mkdir(parents=True)
        legacy.write_text(
            "\n".join(
                [
                    "# Legacy",
                    "## Topic",
                    "> [!correction] F-001: use the corrected mechanism.",
                    "",
                    "The original mechanism is asserted here.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        target = self.repo_root / self.target_relative
        target.write_text(
            "# Target\n## Correct mechanism\nCurrent explanation.\n",
            encoding="utf-8",
        )
        report = self.repo_root / self.report_relative
        report.parent.mkdir(parents=True)
        report.write_text(
            "\n".join(
                [
                    "# Corrections",
                    "## Required corrections",
                    "### F-001 — Exact correction",
                    "The replacement is documented here.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        self.source_relative = "torch/example.py"
        source = self.repo_root / self.source_relative
        source.parent.mkdir(parents=True)
        source.write_text("SUPPORTED = True\n", encoding="utf-8")
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(
            json.dumps([self.legacy_relative]),
            encoding="utf-8",
        )
        self.records = audit_graph_docs.audit_manifest(
            self.repo_root,
            self.source_root,
            self.manifest,
        )
        self.rows = [
            row
            for row in audit_graph_docs.build_ledger_rows(self.records)
            if row["kind"] == "claim_candidate"
        ]
        self.callout, self.original = self.rows
        self.destination = {
            "path": self.target_relative,
            "anchor_text": "Correct mechanism",
            "anchor_occurrence": 1,
        }
        self.catalog = audit_graph_docs.load_correction_catalog(
            [self.repo_root / self.report_relative],
            self.repo_root,
        )

    @staticmethod
    def text_hash(row: dict[str, object]) -> str:
        return hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest()

    def identity_fields(self, row: dict[str, object]) -> dict[str, object]:
        return {
            "claim_id": row["id"],
            "page": row["page"],
            "page_sha256": row["page_sha256"],
            "start_line": row["source_start_line"],
            "end_line": row["source_end_line"],
            "source_start_column": row["source_start_column"],
            "source_end_column": row["source_end_column"],
            "claim_text_sha256": self.text_hash(row),
        }

    def unresolved_decision(
        self, row: dict[str, object]
    ) -> dict[str, object]:
        return {
            **self.identity_fields(row),
            "status": "unresolved",
            "claimed_baseline": "unknown",
            "current_result": "claim_specific_evidence_missing",
            "destinations": [self.destination],
            "destination_role": "candidate_only",
            "action": "retain-quarantined",
            "blocker": "No claim-specific source, runtime, or historical evidence.",
            "evidence": [],
            "correction_ids": [],
            "range_context": [],
            "notes": "The exact legacy claim remains quarantined.",
        }

    def corrected_decision(self) -> dict[str, object]:
        report_entry = self.catalog["F-001"]
        return {
            **self.identity_fields(self.callout),
            "status": "corrected",
            "claimed_baseline": "unknown",
            "current_result": "local_correction_callout_and_report",
            "destinations": [self.destination],
            "destination_role": "candidate_only",
            "action": "retain-with-correction",
            "blocker": "",
            "correction_ids": ["F-001"],
            "range_context": [],
            "notes": "The callout is the correction pointer, not imported folklore.",
            "evidence": [
                {
                    "kind": "local_callout",
                    "path": self.legacy_relative,
                    "page_sha256": self.callout["page_sha256"],
                    "start_line": 3,
                    "end_line": 3,
                    "callout_type": "correction",
                    "correction_ids": ["F-001"],
                },
                {
                    "kind": "correction_report",
                    "correction_id": "F-001",
                    "path": report_entry["path"],
                    "anchor_text": report_entry["anchor_text"],
                    "anchor_occurrence": 1,
                    "source_line": report_entry["source_line"],
                    "report_sha256": report_entry["report_sha256"],
                },
            ],
        }

    def local_disposition(
        self,
        *,
        correction_id: str = "F-001",
        decision: dict[str, object] | None = None,
    ) -> dict[str, object]:
        selected = decision or self.corrected_decision()
        return {
            "correction_id": correction_id,
            "disposition": "local-claim-corrected",
            "claim_refs": [
                {
                    field: selected[field]
                    for field in (
                        "claim_id",
                        "page",
                        "page_sha256",
                        "start_line",
                        "end_line",
                        "source_start_column",
                        "source_end_column",
                        "claim_text_sha256",
                    )
                }
            ],
        }

    def catalog_only_disposition(
        self,
        *,
        reason: str = (
            "The current legacy page already states the qualified mechanism; "
            "the catalog entry adds background rather than correcting a false assertion."
        ),
        false_assertion_audit: str = "none-found",
    ) -> dict[str, object]:
        legacy = self.repo_root / self.legacy_relative
        return {
            "correction_id": "F-001",
            "disposition": "catalog-only/no-local-target",
            "reason": reason,
            "false_assertion_audit": false_assertion_audit,
            "accurate_spans": [
                {
                    "kind": "current-accurate-span",
                    "path": self.legacy_relative,
                    "start_line": 1,
                    "end_line": 2,
                    "baseline": "frozen legacy snapshot",
                    "sha256": hashlib.sha256(legacy.read_bytes()).hexdigest(),
                    "supports": (
                        "The exact current span was inspected and contains no "
                        "contradicted mechanism."
                    ),
                }
            ],
        }

    def test_claim_decisions_require_exact_identity_and_one_to_one(self) -> None:
        first = self.unresolved_decision(self.callout)
        errors = audit_graph_docs.validate_claim_decisions(
            self.rows,
            [first],
            self.repo_root,
            self.catalog,
        )
        self.assertEqual(
            [error["code"] for error in errors],
            ["missing_claim_decision"],
        )

        wrong_hash = self.unresolved_decision(self.original)
        wrong_hash["page_sha256"] = "f" * 64
        errors = audit_graph_docs.validate_claim_decisions(
            self.rows,
            [first, wrong_hash, dict(first)],
            self.repo_root,
            self.catalog,
        )
        self.assertEqual(
            {error["code"] for error in errors},
            {"claim_page_hash_mismatch", "duplicate_claim_decision"},
        )

    def test_unresolved_claim_cannot_be_imported_or_migrated(self) -> None:
        decision = self.unresolved_decision(self.original)
        decision.update(
            {
                "destination_role": "authoritative_import",
                "action": "migrate",
                "blocker": "",
            }
        )

        errors = audit_graph_docs.validate_claim_decisions(
            [self.original],
            [decision],
            self.repo_root,
            self.catalog,
        )

        self.assertEqual(
            {error["code"] for error in errors},
            {
                "unresolved_claim_imported",
                "invalid_unresolved_claim_action",
                "missing_claim_blocker",
            },
        )

    def test_verified_claim_rejects_range_or_locator_as_evidence(self) -> None:
        decision = self.unresolved_decision(self.original)
        decision.update(
            {
                "status": "verified-current",
                "current_result": "claimed_verified",
                "destination_role": "authoritative_import",
                "action": "migrate",
                "blocker": "",
                "evidence": [
                    {
                        "kind": "range_decision",
                        "page": self.legacy_relative,
                        "start_line": 1,
                        "end_line": 6,
                    },
                    {
                        "kind": "source_locator",
                        "path": self.source_relative,
                        "start_line": 1,
                        "end_line": 1,
                    },
                ],
            }
        )

        errors = audit_graph_docs.validate_claim_decisions(
            [self.original],
            [decision],
            self.repo_root,
            self.catalog,
        )
        self.assertEqual(
            [error["code"] for error in errors],
            [
                "non_claim_specific_verified_evidence",
                "non_claim_specific_verified_evidence",
            ],
        )

        source_path = self.repo_root / self.source_relative
        decision["evidence"] = [
            {
                "kind": "source",
                "path": self.source_relative,
                "baseline": "e8f97c1a6ef8cbcdd0a946606bc1e924e4f07e52",
                "start_line": 1,
                "end_line": 1,
                "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "supports": "The exact claim is supported by this source range.",
            }
        ]
        self.assertEqual(
            audit_graph_docs.validate_claim_decisions(
                [self.original],
                [decision],
                self.repo_root,
                self.catalog,
            ),
            [],
        )

    def test_corrected_claim_requires_catalog_and_exact_local_callout(self) -> None:
        decision = self.corrected_decision()
        self.assertEqual(
            audit_graph_docs.validate_claim_decisions(
                [self.callout],
                [decision],
                self.repo_root,
                self.catalog,
            ),
            [],
        )

        without_callout = self.corrected_decision()
        without_callout["evidence"] = without_callout["evidence"][1:]
        invalid_id = self.corrected_decision()
        invalid_id["correction_ids"] = ["F-999"]
        missing_id = self.corrected_decision()
        missing_id["correction_ids"] = []
        errors = audit_graph_docs.validate_claim_decisions(
            [self.callout],
            [without_callout],
            self.repo_root,
            self.catalog,
        ) + audit_graph_docs.validate_claim_decisions(
            [self.callout],
            [invalid_id],
            self.repo_root,
            self.catalog,
        ) + audit_graph_docs.validate_claim_decisions(
            [self.callout],
            [missing_id],
            self.repo_root,
            self.catalog,
        )
        self.assertEqual(
            {error["code"] for error in errors},
            {
                "corrected_without_local_callout",
                "invalid_correction_id",
                "correction_id_callout_mismatch",
                "missing_correction_id",
            },
        )

    def test_correction_dispositions_require_exactly_one_entry_per_catalog_id(
        self,
    ) -> None:
        valid = self.local_disposition()
        self.assertEqual(
            audit_graph_docs.validate_correction_dispositions(
                {
                    "schema_version": 1,
                    "dispositions": [valid],
                },
                self.catalog,
                self.rows,
                [self.corrected_decision()],
                self.repo_root,
            ),
            [],
        )

        errors = audit_graph_docs.validate_correction_dispositions(
            {
                "schema_version": 1,
                "dispositions": [valid, dict(valid)],
            },
            self.catalog,
            self.rows,
            [self.corrected_decision()],
            self.repo_root,
        )
        self.assertEqual(
            {error["code"] for error in errors},
            {"duplicate_correction_disposition"},
        )

        errors = audit_graph_docs.validate_correction_dispositions(
            {"schema_version": 1, "dispositions": []},
            self.catalog,
            self.rows,
            [self.corrected_decision()],
            self.repo_root,
        )
        self.assertEqual(
            [error["code"] for error in errors],
            ["missing_correction_disposition"],
        )

    def test_local_disposition_rejects_wrong_id_adjacency(self) -> None:
        wrong_catalog = {
            "F-002": {
                **self.catalog["F-001"],
                "correction_id": "F-002",
            }
        }
        wrongly_attributed_decision = self.corrected_decision()
        wrongly_attributed_decision["correction_ids"] = ["F-002"]
        errors = audit_graph_docs.validate_correction_dispositions(
            {
                "schema_version": 1,
                "dispositions": [
                    self.local_disposition(correction_id="F-002")
                ],
            },
            wrong_catalog,
            self.rows,
            [wrongly_attributed_decision],
            self.repo_root,
        )
        self.assertEqual(
            [error["code"] for error in errors],
            ["disposition_claim_missing_exact_callout"],
        )

    def test_catalog_only_disposition_requires_exact_none_found_audit(
        self,
    ) -> None:
        invalid = self.catalog_only_disposition(
            reason="",
            false_assertion_audit="false-assertion-found",
        )
        invalid["accurate_spans"] = []
        errors = audit_graph_docs.validate_correction_dispositions(
            {
                "schema_version": 1,
                "dispositions": [invalid],
            },
            self.catalog,
            self.rows,
            [],
            self.repo_root,
        )
        self.assertEqual(
            {error["code"] for error in errors},
            {
                "catalog_only_false_assertion_not_cleared",
                "catalog_only_missing_accurate_span",
                "catalog_only_missing_reason",
            },
        )

        self.assertEqual(
            audit_graph_docs.validate_correction_dispositions(
                {
                    "schema_version": 1,
                    "dispositions": [
                        self.catalog_only_disposition()
                    ],
                },
                self.catalog,
                self.rows,
                [],
                self.repo_root,
            ),
            [],
        )

        errors = audit_graph_docs.validate_correction_dispositions(
            {
                "schema_version": 1,
                "dispositions": [
                    self.catalog_only_disposition()
                ],
            },
            self.catalog,
            self.rows,
            [self.corrected_decision()],
            self.repo_root,
        )
        self.assertEqual(
            [error["code"] for error in errors],
            ["catalog_only_has_local_claim"],
        )

    def test_generator_is_conservative_and_claim_specific(self) -> None:
        ranges = [
            {
                "page": self.legacy_relative,
                "start_line": 2,
                "end_line": 5,
                "status": "corrected",
                "claimed_baseline": "unknown",
                "current_result": "semantic_correction_required",
                "destinations": [self.destination],
                "destination_role": "authoritative_import",
                "action": "rewrite",
                "notes": "F-001: exact correction.",
            }
        ]

        decisions = audit_graph_docs.generate_legacy_claim_decisions(
            self.rows,
            ranges,
            self.catalog,
            self.repo_root,
        )

        self.assertEqual(len(decisions), 2)
        self.assertEqual(decisions[0]["claim_id"], self.callout["id"])
        self.assertEqual(decisions[0]["status"], "corrected")
        self.assertEqual(decisions[0]["destination_role"], "candidate_only")
        self.assertEqual(decisions[1]["status"], "unresolved")
        self.assertEqual(decisions[1]["action"], "retain-quarantined")
        self.assertIn("broad range", decisions[1]["blocker"])
        self.assertEqual(
            audit_graph_docs.validate_claim_decisions(
                self.rows,
                decisions,
                self.repo_root,
                self.catalog,
            ),
            [],
        )

    def test_unit_generator_rebuilds_ids_from_the_frozen_page_hash(self) -> None:
        base_rows = audit_graph_docs.build_ledger_rows(self.records)
        units = audit_graph_docs.generate_legacy_unit_decisions(
            base_rows,
            self.repo_root,
        )
        rebuilt = audit_graph_docs.build_ledger_rows(
            self.records,
            unit_decisions=units,
        )

        self.assertEqual(len(units), 1)
        self.assertEqual(units[0]["status"], "unresolved")
        self.assertEqual(
            units[0]["destination_role"], "retain_quarantined"
        )
        self.assertEqual(
            audit_graph_docs.summarize_destination_closure(
                rebuilt,
                audit_graph_docs.validate_destinations(
                    units, self.repo_root
                ),
            )["unassigned_structural_units"],
            0,
        )

    def test_claim_closure_separates_decisions_from_truth_status(self) -> None:
        decisions = [
            self.corrected_decision(),
            self.unresolved_decision(self.original),
        ]
        rows = audit_graph_docs.build_ledger_rows(
            [
                {
                    "kind": "claim_candidate",
                    "page": row["page"],
                    "page_sha256": row["page_sha256"],
                    "heading_path": row["section"].split(" > "),
                    "start_line": row["source_start_line"],
                    "end_line": row["source_end_line"],
                    "text": row["text"],
                }
                for row in self.rows
            ],
            claim_decisions=decisions,
        )
        errors = audit_graph_docs.validate_claim_decisions(
            self.rows,
            decisions,
            self.repo_root,
            self.catalog,
        )
        summary = audit_graph_docs.summarize_claim_closure(
            rows,
            decisions,
            errors,
            [],
            self.catalog,
        )

        self.assertEqual(summary["claim_candidates"], 2)
        self.assertEqual(summary["claim_decisions"], 2)
        self.assertEqual(summary["unaudited_claim_candidates"], 0)
        self.assertEqual(summary["unresolved_claims_imported_by_new_series"], 0)
        self.assertEqual(summary["corrected_without_local_callout"], 0)
        self.assertEqual(summary["correction_catalog_ids"], 1)
        self.assertEqual(
            summary["status_totals"],
            {"corrected": 1, "unresolved": 1},
        )

    def test_generate_legacy_claims_cli_writes_exact_page_and_ledger_outputs(
        self,
    ) -> None:
        ranges = self.root / "ranges.json"
        ranges.write_text(
            json.dumps(
                [
                    {
                        "page": self.legacy_relative,
                        "start_line": 2,
                        "end_line": 5,
                        "status": "corrected",
                        "destinations": [self.destination],
                        "destination_role": "authoritative_import",
                        "action": "rewrite",
                        "notes": "F-001: exact correction.",
                    }
                ]
            ),
            encoding="utf-8",
        )
        output_dir = self.root / "claim_decisions"
        ledger = self.root / "claim_ledger.jsonl"
        closure = self.root / "claim_closure.json"
        errors = self.root / "claim_errors.json"
        dispositions = self.root / "correction_dispositions.json"
        dispositions.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "dispositions": [self.local_disposition()],
                }
            ),
            encoding="utf-8",
        )

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "docs.audits.pytorch_graph_series.tools.audit_graph_docs",
                "generate-legacy-claims",
                "--repo-root",
                str(self.repo_root),
                "--source-root",
                str(self.source_root),
                "--manifest",
                str(self.manifest),
                "--decisions",
                str(ranges),
                "--correction-reports",
                str(self.repo_root / self.report_relative),
                "--correction-dispositions",
                str(dispositions),
                "--output-dir",
                str(output_dir),
                "--claim-ledger-output",
                str(ledger),
                "--closure-output",
                str(closure),
                "--errors-output",
                str(errors),
            ],
            capture_output=True,
            check=False,
            cwd=Path(__file__).resolve().parents[4],
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        page_decisions = load_semantic_decisions(
            [output_dir / "legacy.jsonl"]
        )
        ledger_rows = audit_graph_docs.read_ledger_jsonl(ledger)
        summary = json.loads(closure.read_text(encoding="utf-8"))
        self.assertEqual(len(page_decisions), 2)
        self.assertEqual(len(ledger_rows), 2)
        self.assertEqual(json.loads(errors.read_text(encoding="utf-8")), [])
        self.assertEqual(summary["claim_candidates"], 2)
        self.assertEqual(summary["unaudited_claim_candidates"], 0)
        self.assertEqual(summary["correction_dispositions"], 1)
        self.assertEqual(summary["correction_disposition_validation_errors"], 0)
        self.assertEqual(summary["correction_ids_without_disposition"], [])

    def test_main_cli_reports_claim_identity_errors_before_application(
        self,
    ) -> None:
        decisions = [
            self.corrected_decision(),
            self.unresolved_decision(self.original),
        ]
        decisions[1]["page_sha256"] = "f" * 64
        decision_path = self.root / "claim_decisions.jsonl"
        decision_path.write_text(
            "".join(
                json.dumps(decision) + "\n" for decision in decisions
            ),
            encoding="utf-8",
        )
        errors_output = self.root / "claim_errors.json"
        closure_output = self.root / "claim_closure.json"
        dispositions_path = self.root / "correction_dispositions.json"
        dispositions_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "dispositions": [self.local_disposition()],
                }
            ),
            encoding="utf-8",
        )

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "docs.audits.pytorch_graph_series.tools.audit_graph_docs",
                "--repo-root",
                str(self.repo_root),
                "--source-root",
                str(self.source_root),
                "--manifest",
                str(self.manifest),
                "--jsonl-output",
                str(self.root / "inventory.jsonl"),
                "--ledger-jsonl-output",
                str(self.root / "ledger.jsonl"),
                "--claim-ledger-jsonl-output",
                str(self.root / "claim_ledger.jsonl"),
                "--summary-output",
                str(self.root / "summary.md"),
                "--ledger-output",
                str(self.root / "ledger.md"),
                "--claim-decisions",
                str(decision_path),
                "--correction-reports",
                str(self.repo_root / self.report_relative),
                "--correction-dispositions",
                str(dispositions_path),
                "--claim-errors-output",
                str(errors_output),
                "--claim-closure-output",
                str(closure_output),
            ],
            capture_output=True,
            check=False,
            cwd=Path(__file__).resolve().parents[4],
            text=True,
        )

        self.assertEqual(result.returncode, 2)
        errors = json.loads(errors_output.read_text(encoding="utf-8"))
        closure = json.loads(closure_output.read_text(encoding="utf-8"))
        self.assertEqual(
            [error["code"] for error in errors],
            ["claim_page_hash_mismatch"],
        )
        self.assertEqual(closure["claim_validation_errors"], 1)
        self.assertEqual(closure["unaudited_claim_candidates"], 2)

    def test_repository_legacy_claim_artifacts_close_the_frozen_manifest(
        self,
    ) -> None:
        repository_root = Path(__file__).resolve().parents[4]
        audit_root = (
            repository_root
            / "docs/audits/pytorch_graph_series/2026-07-23"
        )
        migration_root = (
            repository_root
            / "docs/audits/pytorch_graph_series/2026-07-27"
        )
        records = audit_graph_docs.audit_manifest(
            repository_root,
            repository_root.parent / "p",
            audit_root / "audit_manifest.json",
        )
        ranges = load_semantic_decisions(
            sorted(audit_root.glob("semantic_decisions_*.json"))
        )
        units = load_semantic_decisions(
            [migration_root / "legacy_unit_decisions.jsonl"]
        )
        decision_files = sorted(
            (migration_root / "legacy_claim_decisions").glob("*.jsonl")
        )
        decisions = load_semantic_decisions(decision_files)
        catalog = audit_graph_docs.load_correction_catalog(
            sorted(audit_root.glob("corrections_*.md")),
            repository_root,
        )
        dispositions = audit_graph_docs.load_correction_dispositions(
            migration_root / "legacy_correction_dispositions.json"
        )
        base_rows = audit_graph_docs.build_ledger_rows(records, ranges)
        claim_rows = [
            row
            for row in base_rows
            if row["kind"] == "claim_candidate"
        ]
        claim_errors = audit_graph_docs.validate_claim_decisions(
            claim_rows,
            decisions,
            repository_root,
            catalog,
        )
        disposition_errors = (
            audit_graph_docs.validate_correction_dispositions(
                dispositions,
                catalog,
                claim_rows,
                decisions,
                repository_root,
            )
        )
        destination_errors = audit_graph_docs.validate_destinations(
            [*ranges, *units, *decisions],
            repository_root,
        )
        rows = audit_graph_docs.build_ledger_rows(
            records,
            ranges,
            claim_decisions=decisions,
            unit_decisions=units,
        )
        closure = audit_graph_docs.summarize_claim_closure(
            rows,
            decisions,
            claim_errors,
            destination_errors,
            catalog,
            dispositions,
            disposition_errors,
        )
        destination_closure = (
            audit_graph_docs.summarize_destination_closure(
                rows,
                destination_errors,
            )
        )
        canonical_claim_rows = [
            row for row in rows if row["kind"] == "claim_candidate"
        ]
        saved_ledger = audit_graph_docs.read_ledger_jsonl(
            migration_root / "legacy_claim_ledger.jsonl"
        )
        saved_closure = json.loads(
            (migration_root / "legacy_claim_closure.json").read_text(
                encoding="utf-8"
            )
        )
        regenerated = audit_graph_docs.generate_legacy_claim_decisions(
            claim_rows,
            ranges,
            catalog,
            repository_root,
        )

        self.assertEqual(len(decision_files), 28)
        self.assertEqual(len(catalog), 94)
        self.assertEqual(len(claim_rows), 2190)
        self.assertEqual(len(decisions), 2190)
        self.assertEqual(
            len({decision["claim_id"] for decision in decisions}),
            2190,
        )
        self.assertEqual(claim_errors, [])
        self.assertEqual(disposition_errors, [])
        self.assertEqual(destination_errors, [])
        self.assertEqual(
            closure["status_totals"],
            {"corrected": 91, "unresolved": 2099},
        )
        self.assertEqual(closure["unaudited_claim_candidates"], 0)
        self.assertEqual(
            closure["unresolved_claims_imported_by_new_series"],
            0,
        )
        self.assertEqual(
            closure["corrected_without_local_callout"],
            0,
        )
        self.assertEqual(
            closure["correction_ids_referenced_by_claims"],
            84,
        )
        self.assertEqual(
            closure["correction_ids_without_local_claim"],
            [
                "A-016",
                "A-018",
                "I-004",
                "P-002",
                "P-003",
                "P-004",
                "P-007",
                "P-008",
                "P-016",
                "P-018",
            ],
        )
        self.assertEqual(closure["correction_dispositions"], 94)
        self.assertEqual(closure["unique_correction_dispositions"], 94)
        self.assertEqual(
            closure["correction_disposition_totals"],
            {
                "catalog-only/no-local-target": 10,
                "local-claim-corrected": 84,
            },
        )
        self.assertEqual(
            closure["correction_disposition_validation_errors"],
            0,
        )
        self.assertEqual(closure["correction_ids_without_disposition"], [])
        self.assertEqual(
            destination_closure["unassigned_structural_units"],
            0,
        )
        self.assertEqual(
            destination_closure["pending_claim_candidates"],
            0,
        )
        self.assertEqual(saved_ledger, canonical_claim_rows)
        self.assertEqual(
            {
                key: saved_closure[key]
                for key in (
                    "manifest_pages",
                    "claim_candidates",
                    "claim_decisions",
                    "unique_claim_decisions",
                    "unaudited_claim_candidates",
                    "claim_validation_errors",
                    "destination_validation_errors",
                    "correction_dispositions",
                    "unique_correction_dispositions",
                    "correction_disposition_validation_errors",
                    "correction_ids_without_disposition",
                    "correction_disposition_totals",
                    "status_totals",
                )
            },
            {
                key: closure[key]
                for key in (
                    "manifest_pages",
                    "claim_candidates",
                    "claim_decisions",
                    "unique_claim_decisions",
                    "unaudited_claim_candidates",
                    "claim_validation_errors",
                    "destination_validation_errors",
                    "correction_dispositions",
                    "unique_correction_dispositions",
                    "correction_disposition_validation_errors",
                    "correction_ids_without_disposition",
                    "correction_disposition_totals",
                    "status_totals",
                )
            },
        )
        self.assertEqual(
            json.loads(
                (
                    migration_root / "legacy_claim_errors.json"
                ).read_text(encoding="utf-8")
            ),
            [],
        )
        self.assertEqual(
            {
                decision["claim_id"]: decision
                for decision in decisions
            },
            {
                decision["claim_id"]: decision
                for decision in regenerated
            },
        )
        self.assertTrue(
            all(
                decision["action"] == "retain-quarantined"
                and decision["destination_role"]
                in {"candidate_only", "retain_quarantined"}
                and decision["blocker"]
                and decision["evidence"] == []
                for decision in decisions
                if decision["status"] == "unresolved"
            )
        )
        self.assertTrue(
            all(
                decision["action"] == "retain-with-correction"
                and decision["destination_role"] != "authoritative_import"
                and decision["correction_ids"]
                for decision in decisions
                if decision["status"] == "corrected"
            )
        )


if __name__ == "__main__":
    unittest.main()
