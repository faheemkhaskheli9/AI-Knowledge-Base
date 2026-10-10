# Roadmap

Improvement plan for the knowledge base, ordered by value. Audit of 2026-10-06:
320 topics, all `status: draft`; every file has the template sections; 0 broken
`[[links]]` but nothing enforced it; 30 topics had no inbound link, so an agent
following `## Related` never reached them; coverage is lopsided (concepts 113,
ml 105 vs models 13, setup 7, deployment 9).

This file is the short ordered plan. Each task names its GitHub issue, which
holds the detail and any leftover acceptance criteria.

## Tasks

1. [x] **Link integrity in the build.** `build_index.py` fails on a broken
   `[[link]]` (code blocks ignored) and warns on orphan topics. Wire the 30
   orphans into their parent topic's `## Related`. Issue #3 (still open for a
   weekly lychee check of `sources:` URLs).
2. [x] **Run the code.** A `check_code.py` that extracts the python block(s)
   of each from-scratch topic and runs them, so the "minimal and runnable"
   rule in AGENTS.md is tested rather than assumed. Start with NumPy-only files.
   Done: `check_code.py` runs every topic whose imports are installed and skips
   the rest; 147 pass. Still unrun: topics needing torch, API keys or downloads
   (a CI job with torch would cover the torch ones). Issue #12 (closed);
   repo hygiene for it in #100 (closed).
3. [x] **Status promotion.** Move a topic `draft -> stable` once its code runs
   (task 2) and its sources resolve. Today the column carries no signal.
   Issue #11 (still open: the most-used API-calling topics are not yet stable).
4. [x] **Staleness report.** `build_index.py` lists `models/`, `setup/` and
   `deployment/` files whose `last_verified` is older than 90 days, the ones
   whose facts (model IDs, prices, versions) rot. Issue #4 (still open for a
   `Verified` column and a `--stale N` flag).
5. [ ] **Fill thin categories.** Add applied topics before more from-scratch
   ones: `models/` (embedding and reranker models, open-weight vision LLMs),
   `setup/` (vLLM/llama.cpp server on Windows, Docker GPU), `deployment/`
   (LLM gateways and rate limits, batch inference, caching). Tracker #98,
   one `topic:*` issue per topic, plus #6, #7 and #10 (A2A, context
   engineering, agent skills).
6. [x] **Scenario links.** Each `scenarios/` file links the from-scratch and
   overview topics it relies on; check every overview topic is reachable from
   at least one scenario. Done: `build_index.py` warns on overview topics no
   `scenarios/` file links; the 98 it found are routed from `scenario-chooser.md`.
   Issue #102 (closed).
7. [x] **Browsable index.** Group `INDEX.md` by category with a count per
   section; one 320-row table is hard to scan. Done: one section per category,
   scenarios first, with a linked contents line. Issue #103 (closed).
8. [x] **Cross-link the Software Engineering KB** from deployment and LLM-app
   topics. Issue #104 (closed).
9. [x] **CI for the Django webapp.** Issue #105 (closed).
10. [ ] **Verify flagged facts.** Replace or remove the "not verified" flags
    and the Claude sampling-parameter note. Issues #5, #8.
11. [ ] **Agent entry points.** `llms.txt`, a glossary and learning paths.
    Issues #13, #96, #97.
