"""Browse the Markdown topics. The .md files stay the source of truth."""
import re
from collections import defaultdict

import markdown
from django.http import FileResponse, Http404
from django.shortcuts import render
from django.urls import reverse
from django.utils.html import escape

from build_index import GLOSSARY, LEARNING_PATHS, ROOT, SKIP, frontmatter

WIKILINK = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]+))?\]\]")
CODE = re.compile(r"(<pre>.*?</pre>|<code>.*?</code>)", re.S)
MD_HREF = re.compile(r'href="(?:[^"]*/)?([\w-]+)\.md(#[^"]*)?"')
LLMS_HREF = re.compile(r'href="(?:[^"]*/)?llms\.txt"')
# The only repo-root files the site serves besides topics: file name -> URL name.
# Fixed list, no file name ever comes from the request, so nothing else is reachable.
PAGES = {GLOSSARY: "glossary", LEARNING_PATHS: "learning_paths", "INDEX.md": "index_md"}
LLMS_TXT = "llms.txt"


def load_topics():
    # ponytail: re-reads every file per request so edits show up live; cache it if that gets slow
    topics = {}
    for path in sorted(ROOT.rglob("*.md")):
        rel = path.relative_to(ROOT).as_posix()
        if path.name in SKIP or rel.startswith(".") or "/" not in rel:
            continue
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        meta = frontmatter(text) or {}
        topics[path.stem] = {
            "title": path.stem,
            "category": rel.split("/")[0],
            **meta,
            "stem": path.stem,
            "rel": rel,
            "text": text,
        }
    return topics


def index(request):
    topics = load_topics()
    q = request.GET.get("q", "").strip()
    category = request.GET.get("category", "")
    tag = request.GET.get("tag", "")
    words = q.lower().split()
    groups, counts = defaultdict(list), defaultdict(int)
    for t in topics.values():
        counts[t["category"]] += 1
        if category and t["category"] != category:
            continue
        if tag and tag not in t.get("tags", []):
            continue
        if words and not all(w in t["text"].lower() for w in words):
            continue
        groups[t["category"]].append(t)
    return render(request, "index.html", {
        "groups": sorted(groups.items()),
        "counts": sorted(counts.items()),
        "shown": sum(len(v) for v in groups.values()),
        "total": len(topics),
        "q": q, "category": category, "tag": tag,
    })


def render_body(text, topics):
    body = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)
    html = markdown.markdown(body, extensions=["fenced_code", "tables", "toc", "sane_lists"])

    def wikilink(m):
        stem, label = m.group(1).strip(), m.group(2)
        if stem not in topics:
            return f'<span class="missing">{escape(label or stem)}</span>'
        return f'<a href="{reverse("topic", args=[stem])}">{escape(label or topics[stem]["title"])}</a>'

    def md_href(m):
        stem, anchor = m.group(1), m.group(2) or ""
        if f"{stem}.md" in PAGES:  # glossary.md, learning-paths.md, INDEX.md: never topics (SKIP)
            return f'href="{reverse(PAGES[stem + ".md"])}{anchor}"'
        if stem not in topics:
            return m.group(0)
        return f'href="{reverse("topic", args=[stem])}{anchor}"'

    # Leave code alone: `[[1, 2]]` in a snippet is a list, not a link.
    parts = CODE.split(html)
    for i in range(0, len(parts), 2):
        parts[i] = MD_HREF.sub(md_href, WIKILINK.sub(wikilink, parts[i]))
        parts[i] = LLMS_HREF.sub(f'href="{reverse("llms_txt")}"', parts[i])
    return "".join(parts)


def topic(request, stem):
    topics = load_topics()
    if stem not in topics:
        raise Http404(f"No topic {stem!r}")
    t = topics[stem]
    backlinks = sorted(
        (o for o in topics.values() if o["stem"] != stem and f"[[{stem}]]" in o["text"]),
        key=lambda o: o["title"],
    )
    return render(request, "topic.html", {"t": t, "body": render_body(t["text"], topics), "backlinks": backlinks})


def page(request, name):
    """glossary.md, learning-paths.md or INDEX.md, rendered like a topic. `name` comes from urls.py."""
    if name not in PAGES:
        raise Http404(f"No page {name!r}")
    text = (ROOT / name).read_text(encoding="utf-8").replace("\r\n", "\n")
    title = next((l[2:].strip() for l in text.splitlines() if l.startswith("# ")), name)
    return render(request, "page.html", {"name": name, "title": title, "body": render_body(text, load_topics())})


def llms_txt(request):
    return FileResponse(open(ROOT / LLMS_TXT, "rb"), content_type="text/plain; charset=utf-8")
