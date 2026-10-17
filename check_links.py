"""Check that every URL in topic `sources` frontmatter resolves.

    python check_links.py                 # every topic
    python check_links.py models/x.md     # only these files

Prints each non-2xx URL under its file. 403 and 429 are warnings (sites that
block bots or rate-limit); any other HTTP error or a network failure is an
error and makes the exit code 1. Stdlib only. Network-bound, so CI runs it
weekly and on demand (.github/workflows/links.yml), not on every push.
"""
import os
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from build_index import ROOT, SKIP, frontmatter

TIMEOUT = 20
WARN = {403, 429}
HEADERS = {  # some sites answer 406 without browser-like Accept headers
    "User-Agent": "Mozilla/5.0 (link checker; +https://github.com/faheemkhaskheli9/AI-Knowledge-Base)",
    "Accept": "text/html,application/xhtml+xml,application/pdf,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
ANNOTATE = os.environ.get("GITHUB_ACTIONS") == "true"  # ::warning:: / ::error:: lines in the run summary


def topics(files):
    if files:
        return [ROOT / f for f in files]
    return [p for p in sorted(ROOT.rglob("*.md"))
            if p.name not in SKIP and not p.relative_to(ROOT).as_posix().startswith(".")]


def clean(url):
    """Drop trailing punctuation picked up from prose; keep a `)` that closes one in the URL (Wikipedia, DOIs)."""
    while url:
        if url[-1] in ".,;\"'":
            url = url[:-1]
        elif url[-1] == ")" and url.count(")") > url.count("("):
            url = url[:-1]
        else:
            return url
    return url


def urls_of(text):
    srcs = (frontmatter(text) or {}).get("sources", [])
    srcs = [srcs] if isinstance(srcs, str) else srcs
    return [clean(u) for s in srcs for u in re.findall(r"https?://\S+", s)]


def status(url):
    """HTTP status after redirects, or the exception text if the request failed."""
    for method in ("HEAD", "GET"):  # some servers reject HEAD with 403/405
        try:
            req = urllib.request.Request(url, method=method, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return r.status
        except urllib.error.HTTPError as e:
            code = e.code
        except Exception as e:  # DNS, TLS, timeout, connection reset
            code = f"{type(e).__name__}: {e}"
    return code


def main(files):
    absent = [f for f in files if not (ROOT / f).is_file()]
    if absent:
        print(f"no such file (paths are relative to {ROOT}): {', '.join(absent)}", file=sys.stderr)
        return 2
    by_file = {p.relative_to(ROOT).as_posix(): urls_of(p.read_text(encoding="utf-8")) for p in topics(files)}
    unique = sorted({u for urls in by_file.values() for u in urls})
    with ThreadPoolExecutor(16) as pool:
        result = dict(zip(unique, pool.map(status, unique)))
    errors = warnings = 0
    for rel, urls in by_file.items():
        bad = [(u, result[u]) for u in urls if not (isinstance(result[u], int) and result[u] < 300)]
        if bad:
            print(rel)
        for url, code in bad:
            warn = code in WARN
            warnings += warn
            errors += not warn
            print(f"  {'WARN' if warn else 'FAIL'} {code} {url}")
            if ANNOTATE:
                print(f"::{'warning' if warn else 'error'} file={rel}::{code} {url}")
    print(f"{len(unique)} unique URLs: {errors} failed, {warnings} warnings")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
