"""Regression checks for index verification dates and configurable freshness."""
import contextlib
import io
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import build_index


class FreshnessTests(unittest.TestCase):
    def test_age_boundary_and_custom_threshold(self):
        meta = {"last_verified": "2026-07-17"}
        today = date(2026, 10, 15)
        self.assertFalse(build_index.stale(meta, "models/test.md", today))
        self.assertTrue(build_index.stale(meta, "models/test.md", today, stale_days=89))
        self.assertFalse(build_index.stale(meta, "models/test.md", today, stale_days=91))
        self.assertFalse(build_index.stale(meta, "models/test.md", today, stale_days=10**12))

    def test_scope_and_invalid_dates(self):
        today = date(2026, 10, 15)
        for meta in ({}, {"last_verified": "invalid"}):
            self.assertTrue(build_index.stale(meta, "setup/test.md", today))
            self.assertFalse(build_index.stale(meta, "concepts/test.md", today))
            self.assertTrue(build_index.stale(meta, "llm-apps/test.md", today, "gpt-5.1"))

    def test_index_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(build_index, "ROOT", Path(tmp)):
            folder = Path(tmp) / "models"
            folder.mkdir()
            (folder / "example.md").write_text('''---
title: Example
last_verified: 2000-01-01
status: draft
tags: [example]
use_cases:
  - test freshness
---
# Example
''', encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()) as output:
                build_index.main(["--stale", "30"])
            index = (Path(tmp) / "INDEX.md").read_text(encoding="utf-8")
            self.assertIn("| Status | Verified |", index)
            self.assertIn("| draft | 2000-01-01 |", index)
            self.assertIn("not verified in 30 days", output.getvalue())
            self.assertIn("models/example.md (2000-01-01)", output.getvalue())
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                build_index.main(["--strict", "--stale", "30"])

    def test_llms_txt(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(build_index, "ROOT", Path(tmp)):
            for folder, title, use in (("models", "Example", "pick a model"), ("scenarios", "Chooser", "start here")):
                (Path(tmp) / folder).mkdir()
                (Path(tmp) / folder / f"{title.lower()}.md").write_text(
                    f"---\ntitle: {title}\ncategory: {folder}\nuse_cases:\n  - {use}\n  - second use\n---\n# {title}\n",
                    encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                build_index.main([])
            lines = (Path(tmp) / "llms.txt").read_text(encoding="utf-8").splitlines()
        self.assertTrue(lines[0].startswith("# "))
        self.assertTrue(lines[2].startswith("> "))
        # Same category order as INDEX.md (scenarios first), one link line per topic with its first use case.
        self.assertLess(lines.index("## scenarios"), lines.index("## models"))
        self.assertIn("- [Chooser](scenarios/chooser.md): start here", lines)
        self.assertIn("- [Example](models/example.md): pick a model", lines)

    def test_invalid_cli_thresholds(self):
        for value in ("-1", "abc", "1.5"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                build_index.main(["--stale", value])
            self.assertEqual(error.exception.code, 2)
        self.assertEqual(build_index.nonnegative_days("0"), 0)


if __name__ == "__main__":
    unittest.main()
