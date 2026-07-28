from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from docs.audits.pytorch_graph_series.tools.delivery_gates import (
    validate_markdown_fences_and_mermaid,
    validate_markdown_links,
    validate_local_corrections,
    validate_numbered_course,
    validate_related_pages,
    validate_wikilinks,
)


class DeliveryGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.course = self.root / "wiki" / "course"
        self.course.mkdir(parents=True)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _write_course(self) -> Path:
        paths: list[str] = []
        for number in range(22):
            name = (
                "00_pytorch_graph_series_index.md"
                if number == 0
                else f"{number:02d}_page_{number:02d}.md"
            )
            relative = f"wiki/course/{name}"
            paths.append(relative)
            if number == 0:
                body = (
                    "# 00 · Index\n\n"
                    "## Map\n\n"
                    "[[01_page_01]]\n\n"
                    "## Related Pages\n\n"
                    "- [[01_page_01]]\n"
                )
            else:
                previous = (
                    "00_pytorch_graph_series_index"
                    if number == 1
                    else f"{number - 1:02d}_page_{number - 1:02d}"
                )
                next_line = (
                    "- 返回总索引：[[00_pytorch_graph_series_index]]"
                    if number == 21
                    else f"- 下一篇：[[{number + 1:02d}_page_{number + 1:02d}]]"
                )
                body = (
                    f"# {number:02d} · Page\n\n"
                    "## Mechanism\n\n"
                    "Text.\n\n"
                    "## 学习顺序\n\n"
                    f"- 上一篇：[[{previous}]]\n"
                    f"{next_line}\n\n"
                    "## Related Pages\n\n"
                    "- [[00_pytorch_graph_series_index]]\n"
                )
            (self.root / relative).write_text(body, encoding="utf-8")
        manifest = self.root / "course_manifest.json"
        manifest.write_text(
            json.dumps(paths, ensure_ascii=False), encoding="utf-8"
        )
        return manifest

    def test_numbered_course_requires_00_through_21_h1_and_navigation(self) -> None:
        manifest = self._write_course()

        errors = validate_numbered_course(self.root, manifest)

        self.assertEqual([], errors)
        page = self.course / "10_page_10.md"
        page.write_text(
            page.read_text(encoding="utf-8").replace("# 10 ·", "# Wrong"),
            encoding="utf-8",
        )
        errors = validate_numbered_course(self.root, manifest)
        self.assertIn("course_h1_number_mismatch", {item["code"] for item in errors})

    def test_related_pages_must_be_the_last_h2(self) -> None:
        path = self.course / "one.md"
        path.write_text(
            "# One\n\n## Related Pages\n\n- [[two]]\n\n## Later\n",
            encoding="utf-8",
        )

        errors = validate_related_pages([path])

        self.assertEqual(["related_pages_not_final"], [item["code"] for item in errors])

    def test_wikilinks_validate_target_and_exact_anchor(self) -> None:
        target = self.course / "target.md"
        target.write_text("# Target\n\n## Exact heading\n", encoding="utf-8")
        source = self.course / "source.md"
        source.write_text(
            "# Source\n\n"
            "[[course/target#Exact heading]]\n"
            "[[course/target#Missing heading]]\n"
            "[[missing_page]]\n",
            encoding="utf-8",
        )

        errors = validate_wikilinks(self.root, [source])

        self.assertEqual(
            {"wikilink_anchor_missing", "wikilink_target_missing"},
            {item["code"] for item in errors},
        )

    def test_wikilinks_resolve_repo_docs_and_ignore_inline_code_examples(self) -> None:
        docs = self.root / "docs"
        docs.mkdir()
        (docs / "correction_report.md").write_text(
            "# Corrections\n", encoding="utf-8"
        )
        source = self.course / "source.md"
        source.write_text(
            "# Source\n\n"
            "[[correction_report]]\n"
            "`Callable[[list[Node]], list[Node]]`\n"
            "`[[placeholder_link]]`\n",
            encoding="utf-8",
        )

        errors = validate_wikilinks(self.root, [source])

        self.assertEqual([], errors)

    def test_wikilink_anchor_may_contain_inline_code_or_single_brackets(self) -> None:
        target = self.course / "target.md"
        target.write_text(
            "# Target\n\n"
            "## 8. `torch.compile` dynamic=[True]\n",
            encoding="utf-8",
        )
        source = self.course / "source.md"
        source.write_text(
            "# Source\n\n"
            "[[course/target#8. `torch.compile` dynamic=[True]]]\n",
            encoding="utf-8",
        )

        errors = validate_wikilinks(self.root, [source])

        self.assertEqual([], errors)

    def test_fences_and_mermaid_require_balanced_nonempty_known_diagram(self) -> None:
        valid = self.course / "valid.md"
        valid.write_text(
            "# Valid\n\n"
            "```mermaid\n"
            "%% a comment\n"
            "flowchart LR\n"
            "  A --> B\n"
            "```\n",
            encoding="utf-8",
        )
        invalid = self.course / "invalid.md"
        invalid.write_text(
            "# Invalid\n\n"
            "```mermaid\n"
            "not-a-mermaid-directive\n"
            "```\n\n"
            "```python\n"
            "print('unterminated')\n",
            encoding="utf-8",
        )

        self.assertEqual([], validate_markdown_fences_and_mermaid([valid]))
        errors = validate_markdown_fences_and_mermaid([invalid])

        self.assertEqual(
            {"mermaid_unknown_directive", "unbalanced_code_fence"},
            {item["code"] for item in errors},
        )

    def test_markdown_links_validate_local_files_and_anchors(self) -> None:
        target = self.course / "target.md"
        target.write_text("# Target\n\n## Exact heading\n", encoding="utf-8")
        source = self.course / "source.md"
        source.write_text(
            "# Source\n\n"
            "[`valid`](target.md#Exact%20heading)\n"
            "![asset](missing.png)\n"
            "[bad anchor](target.md#Missing%20heading)\n"
            "`[example](not-a-real-link.md)`\n"
            "[external](https://example.com)\n",
            encoding="utf-8",
        )

        errors = validate_markdown_links(self.root, [source])

        self.assertEqual(
            {"markdown_link_anchor_missing", "markdown_link_target_missing"},
            {item["code"] for item in errors},
        )

    def test_corrected_range_requires_ids_and_local_callout(self) -> None:
        page = self.course / "legacy.md"
        page.write_text(
            "# Legacy\n\n"
            "## Old section\n"
            "> [!correction] X-001：本区段按固定基线纠错；"
            "现行结论见 [[course/target#Exact heading]]，逐项说明见 [[correction_report]]。\n"
            "Old text.\n",
            encoding="utf-8",
        )
        decision_path = self.root / "semantic.json"
        decision_path.write_text(
            json.dumps(
                [
                    {
                        "page": "wiki/course/legacy.md",
                        "start_line": 3,
                        "end_line": 5,
                        "status": "corrected",
                        "notes": "X-001: replacement",
                    }
                ]
            ),
            encoding="utf-8",
        )

        self.assertEqual(
            [], validate_local_corrections(self.root, [decision_path])
        )
        page.write_text(
            page.read_text(encoding="utf-8").replace("X-001", "X-002"),
            encoding="utf-8",
        )
        errors = validate_local_corrections(self.root, [decision_path])
        self.assertEqual(
            ["corrected_range_without_local_callout"],
            [item["code"] for item in errors],
        )


if __name__ == "__main__":
    unittest.main()
