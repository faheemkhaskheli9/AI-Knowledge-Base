from django.test import SimpleTestCase

from webapp.views import load_topics, render_body


class SiteTests(SimpleTestCase):
    def test_index_lists_every_topic(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.context["shown"], len(load_topics()))

    def test_search_and_category_filter(self):
        r = self.client.get("/", {"q": "attention", "category": "concepts"})
        self.assertTrue(r.context["shown"] > 0)
        self.assertEqual([name for name, _ in r.context["groups"]], ["concepts"])

    def test_topic_page_and_404(self):
        self.assertEqual(self.client.get("/t/activation-functions/").status_code, 200)
        self.assertEqual(self.client.get("/t/no-such-topic/").status_code, 404)

    def test_wikilinks_outside_code_only(self):
        topics = {"a": {"title": "Topic A"}}
        html = render_body("See [[a]] and [[gone]].\n\n```\nx = [[a]]\n```\n", topics)
        self.assertIn('<a href="/t/a/">Topic A</a>', html)
        self.assertIn('<span class="missing">gone</span>', html)
        self.assertIn("x = [[a]]", html)
