"""The agent directory must track the same topics as the Markdown index."""
import re
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import build_index


class AgentDirectoryTests(unittest.TestCase):
    def test_generation_order_metadata_and_refresh(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(build_index, 'ROOT', Path(tmp)):
            root = Path(tmp)
            fixtures = (
                ('models/provider.md', 'Provider', 'models', 'call a model'),
                ('scenarios/chooser.md', 'Chooser', 'scenarios', 'choose an approach'),
                ('concepts/foundation.md', 'Foundation', 'concepts', ''),
            )
            for rel, title, category, use in fixtures:
                path = root / rel
                path.parent.mkdir(exist_ok=True)
                path.write_text(
                    f'---\ntitle: {title}\ncategory: {category}\ntags: [example]\n'
                    'status: draft\nlast_verified: 2026-10-15\n'
                    + (f'use_cases:\n  - {use}\n' if use else '') + '---\n', encoding='utf-8')
            build_index.build(today=date(2026, 10, 15))
            output = (root / 'llms.txt').read_text(encoding='utf-8')
            self.assertTrue(output.startswith('# AI Knowledge Base\n\n> '))
            self.assertIn('[Topic index](INDEX.md)', output)
            self.assertIn('[Agent instructions](AGENTS.md)', output)
            self.assertLess(output.index('## scenarios'), output.index('## concepts'))
            self.assertLess(output.index('## concepts'), output.index('## models'))
            self.assertIn('[Foundation](concepts/foundation.md): example', output)
            self.assertIn('status: draft; verified: 2026-10-15', output)
            index_paths = re.findall(r'\]\(([^)]+\.md)\)', (root / 'INDEX.md').read_text())
            agent_paths = re.findall(r'\]\(([^)]+/[^)]+\.md)\)', output)
            self.assertEqual(set(index_paths), set(agent_paths))
            self.assertEqual(len(agent_paths), len(fixtures))
            build_index.build(today=date(2026, 10, 15))
            self.assertEqual((root / 'llms.txt').read_text(), output)
            (root / 'models/provider.md').unlink()
            build_index.build(today=date(2026, 10, 15))
            self.assertNotIn('models/provider.md', (root / 'llms.txt').read_text())


if __name__ == '__main__':
    unittest.main()
