"""Regression checks for check_code.py and check_links.py (stdlib only, no network)."""
import contextlib
import importlib.machinery
import importlib.util
import io
import sys
import unittest
from unittest.mock import patch

import check_code
import check_links


class CheckCodeTests(unittest.TestCase):
    def test_imported_names(self):
        names = check_code.imported_names("import os.path\nfrom google import genai\nfrom numpy import *\n")
        self.assertEqual(names, {"os.path", "google", "google.genai", "numpy"})
        self.assertEqual(check_code.imported("from google import genai\n"), {"google"})
        self.assertIsNone(check_code.imported_names("def broken(:\n"))

    def test_network_sdk_inside_namespace_package(self):
        self.assertTrue(check_code.needs_network({"google", "google.genai"}))
        self.assertTrue(check_code.needs_network({"google.genai.types"}))
        self.assertTrue(check_code.needs_network({"openai"}))
        self.assertFalse(check_code.needs_network({"google", "google.protobuf", "numpy", "numpy.linalg"}))

    def test_missing_submodule_of_namespace_package(self):
        namespace = importlib.machinery.ModuleSpec("ns", None, is_package=True)
        regular = importlib.machinery.ModuleSpec("pkg", object(), origin="/pkg/__init__.py")
        specs = {"ns": namespace, "ns.present": regular, "pkg": regular}
        with patch.object(importlib.util, "find_spec", side_effect=specs.get):
            # Under a namespace package, a name that is not a submodule cannot be imported.
            self.assertEqual(check_code.namespace_missing({"ns", "ns.present", "ns.absent"}), ["ns.absent"])
            # Under a regular package, `from pkg import name` may be an attribute: not checked.
            self.assertEqual(check_code.namespace_missing({"pkg", "pkg.helper"}), [])
        # Never resolve below a regular package: find_spec("pkg.sub") would import (run) pkg.
        with patch.object(importlib.util, "find_spec", side_effect=specs.get) as find_spec:
            check_code.namespace_missing({"pkg.sub.name"})
        self.assertEqual([c.args[0] for c in find_spec.call_args_list], ["pkg"])

    def test_file_arguments_resolve_from_cwd_or_repo_root(self):
        import os
        from pathlib import Path
        target = (check_code.ROOT / "glossary.md").resolve()
        old = os.getcwd()
        try:
            os.chdir(check_code.ROOT / "concepts")
            for mod in (check_code, check_links):
                self.assertEqual(mod.locate("../glossary.md"), target)  # relative to the current folder
                self.assertEqual(mod.locate("glossary.md"), target)     # relative to the repo root
                self.assertIsNone(mod.locate(str(Path(os.devnull))))    # outside the repo
        finally:
            os.chdir(old)

    def test_skip_check_marker(self):
        text = ("```python\nprint(1)\n```\n<!-- skip-check: fragment -->\n```python\nundefined()\n```\n")
        self.assertEqual(check_code.blocks(text), ["print(1)\n"])


    def test_cli_rejects_missing_file_and_bad_jobs(self):
        for argv in (["check_code.py", "no-such-topic.md"], ["check_code.py", "--jobs", "0"]):
            with patch.object(sys, "argv", argv), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as error:
                check_code.main()
            self.assertEqual(error.exception.code, 2)


class CheckLinksTests(unittest.TestCase):
    def test_cli_rejects_missing_file(self):
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(check_links.main(["no-such-topic.md"]), 2)
        self.assertIn("no-such-topic.md", err.getvalue())

    def test_trailing_punctuation(self):
        self.assertEqual(check_links.clean("https://example.org/a)."), "https://example.org/a")
        self.assertEqual(check_links.clean("https://example.org/a\","), "https://example.org/a")

    def test_balanced_parentheses_kept(self):
        for url in ("https://en.wikipedia.org/wiki/Phi_(language_model)",
                    "https://doi.org/10.1016/0893-6080(89)90014-2"):
            self.assertEqual(check_links.clean(url), url)
            self.assertEqual(check_links.clean(url + ")."), url)

    def test_urls_of_sources_only(self):
        text = ("---\ntitle: T\nsources:\n  - https://en.wikipedia.org/wiki/Phi_(language_model)\n"
                "  - Paper (https://arxiv.org/abs/1234.5678).\n---\nBody https://not-a-source.example\n")
        self.assertEqual(check_links.urls_of(text), ["https://en.wikipedia.org/wiki/Phi_(language_model)",
                                                     "https://arxiv.org/abs/1234.5678"])


if __name__ == "__main__":
    unittest.main()
