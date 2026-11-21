"""Regression tests for URL cleanup and HTTP fallback in check_links."""
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import check_links


class LinkCheckerTests(unittest.TestCase):
    def test_balanced_parentheses_survive_cleanup(self):
        self.assertEqual(
            check_links.clean_url('https://example.org/Phi_(language_model)).'),
            'https://example.org/Phi_(language_model)',
        )

    def test_trailing_prose_punctuation_is_removed(self):
        self.assertEqual(check_links.clean_url('https://example.org/page.,;'),
                         'https://example.org/page')

    @patch('check_links.urllib.request.urlopen')
    def test_head_success_does_not_download_body(self, open_url):
        response = open_url.return_value.__enter__.return_value
        response.status = 204
        self.assertEqual(check_links.status('https://example.org/ok'), 204)
        self.assertEqual(open_url.call_count, 1)
        self.assertEqual(open_url.call_args.args[0].get_method(), 'HEAD')

    @patch('check_links.urllib.request.urlopen')
    def test_head_rejected_falls_back_to_get(self, open_url):
        response = unittest.mock.MagicMock()
        response.__enter__.return_value.status = 200
        open_url.side_effect = [HTTPError('https://example.org', 405, 'Method Not Allowed', {}, None), response]
        self.assertEqual(check_links.status('https://example.org'), 200)
        self.assertEqual([call.args[0].get_method() for call in open_url.call_args_list],
                         ['HEAD', 'GET'])

    @patch('check_links.urllib.request.urlopen')
    def test_network_failure_is_reported(self, open_url):
        open_url.side_effect = URLError('offline')
        result = check_links.status('https://example.org')
        self.assertIn('URLError', result)
        self.assertEqual(open_url.call_count, 2)


if __name__ == '__main__':
    unittest.main()
