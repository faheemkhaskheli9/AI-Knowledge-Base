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


class EntryPointPageTests(SimpleTestCase):
    """glossary.md, learning-paths.md, INDEX.md and llms.txt have read-only pages (#126)."""

    def test_glossary_page(self):
        r = self.client.get("/glossary/")
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, "<strong>Activation function</strong>")
        topics = load_topics()  # [[activation-functions]] -> topic page, labelled with its title
        self.assertContains(r, f'<a href="/t/activation-functions/">{topics["activation-functions"]["title"]}</a>', html=True)
        self.assertContains(r, 'href="/learning-paths/"')  # its [learning-paths.md](learning-paths.md) link
        self.assertNotContains(r, 'class="missing"')

    def test_learning_paths_page(self):
        r = self.client.get("/learning-paths/")
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, "Learning paths")
        self.assertContains(r, 'href="/t/ml-fundamentals/"')
        self.assertContains(r, 'href="/glossary/"')
        self.assertNotContains(r, 'class="missing"')

    def test_index_md_page(self):
        r = self.client.get("/index/")
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, "scenario-chooser")
        # INDEX.md rows are relative Markdown links (scenarios/agriculture.md) -> topic pages.
        self.assertContains(r, 'href="/t/agriculture/"')
        self.assertNotContains(r, 'href="scenarios/agriculture.md"')

    def test_llms_txt_is_plain_text(self):
        r = self.client.get("/llms.txt")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r["Content-Type"].startswith("text/plain"))
        body = b"".join(r.streaming_content).decode()
        self.assertTrue(body.startswith("# AI Knowledge Base"))

    def test_header_links_entry_points(self):
        r = self.client.get("/")
        for href in ("/glossary/", "/learning-paths/", "/index/", "/llms.txt"):
            self.assertContains(r, f'href="{href}"')

    def test_topic_links_to_entry_points_resolve(self):
        text = ("See [terms](../glossary.md#a), [paths](../learning-paths.md), "
                "[index](../INDEX.md) and [list](../llms.txt).")
        html = render_body(text, load_topics())
        for href in ("/glossary/#a", "/learning-paths/", "/index/", "/llms.txt"):
            self.assertIn(f'href="{href}"', html)
            self.assertEqual(self.client.get(href.split("#")[0]).status_code, 200)

    def test_no_other_root_files(self):
        for url in ("/README.md", "/AGENTS.md", "/t/glossary/", "/t/INDEX/", "/glossary.md", "/settings.py"):
            self.assertEqual(self.client.get(url).status_code, 404, url)
