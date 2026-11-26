"""Regression tests for Python snippets in Markdown fences."""
import unittest

from check_code import blocks


class PythonFenceTests(unittest.TestCase):
    def test_indented_fences_preserve_relative_python_indentation(self):
        text = "  ```python \t\r\n  if True:\r\n      print('first')\r\n  ``` \t\r\n"
        text += "```py\r\nprint('second')\r\n```"
        self.assertEqual(blocks(text), ["if True:\n    print('first')\n", "print('second')\n"])

    def test_skip_directive_applies_only_to_next_fence(self):
        text = "  <!-- skip-check: incomplete snippet -->\r\n\r\n"
        text += "  ```py \t\r\nthis =\r\n  ```\r\n"
        text += "```python\r\nprint('checked')\r\n```\r\n"
        self.assertEqual(blocks(text), ["print('checked')\n"])

    def test_backticks_in_python_do_not_close_the_fence(self):
        code = "value = '''\n```\n```` with text\n'''\n"
        self.assertEqual(blocks(f"````python\n{code}`````\n"), [code])

    def test_python_inside_a_non_python_fence_is_ignored(self):
        text = "````markdown\n```python\nignored\n```\n````\n"
        text += "~~~py\nprint('checked')\n~~~~\n"
        self.assertEqual(blocks(text), ["print('checked')\n"])

    def test_inline_and_unclosed_fences_are_ignored(self):
        self.assertEqual(blocks("Prose with ```python\nignored\n```\n"), [])
        self.assertEqual(blocks("```python\nignored\n"), [])

    def test_fence_with_extra_info_is_ignored(self):
        self.assertEqual(blocks("```python fragment\nignored\n```\n"), [])


if __name__ == "__main__":
    unittest.main()
