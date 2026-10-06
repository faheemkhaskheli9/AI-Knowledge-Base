"""Browse the Markdown topics. The .md files stay the source of truth."""
import re
from collections import defaultdict

import markdown
from django.http import Http404
from django.shortcuts import render
from django.urls import reverse
from django.utils.html import escape

from build_index import ROOT, SKIP, frontmatter

WIKILINK = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]+))?\]\]")
CODE = re.compile(r"(<pre>.*?</pre>|<code>.*?</code>)", re.S)
MD_HREF = re.compile(r'href="(?:[^"]*/)?([\w-]+)\.md(#[^"]*)?"')


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
        stem = m.group(1)
        if stem not in topics:
            return m.group(0)
        return f'href="{reverse("topic", args=[stem])}{m.group(2) or ""}"'

    # Leave code alone: `[[1, 2]]` in a snippet is a list, not a link.
    parts = CODE.split(html)
    for i in range(0, len(parts), 2):
        parts[i] = MD_HREF.sub(md_href, WIKILINK.sub(wikilink, parts[i]))
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
