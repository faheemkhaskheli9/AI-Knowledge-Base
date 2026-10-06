# Roadmap

Improvement plan for the knowledge base, ordered by value. Audit of 2026-10-06:
320 topics, all `status: draft`; every file has the template sections; 0 broken
`[[links]]` but nothing enforced it; 30 topics had no inbound link, so an agent
following `## Related` never reached them; coverage is lopsided (concepts 113,
ml 105 vs models 13, setup 7, deployment 9).

## Tasks

1. [x] **Link integrity in the build.** `build_index.py` fails on a broken
   `[[link]]` (code blocks ignored) and warns on orphan topics. Wire the 30
   orphans into their parent topic's `## Related`.
2. [ ] **Run the code.** A `check_code.py` that extracts the python block(s)
   of each from-scratch topic and runs them, so the "minimal and runnable"
   rule in AGENTS.md is tested rather than assumed. Start with NumPy-only files.
3. [ ] **Status promotion.** Move a topic `draft -> stable` once its code runs
   (task 2) and its sources resolve. Today the column carries no signal.
4. [ ] **Staleness report.** `build_index.py` lists `models/`, `setup/` and
   `deployment/` files whose `last_verified` is older than 90 days, the ones
   whose facts (model IDs, prices, versions) rot.
5. [ ] **Fill thin categories.** Add applied topics before more from-scratch
   ones: `models/` (embedding and reranker models, open-weight vision LLMs),
   `setup/` (vLLM/llama.cpp server on Windows, Docker GPU), `deployment/`
   (LLM gateways and rate limits, batch inference, caching).
6. [ ] **Scenario links.** Each `scenarios/` file links the from-scratch and
   overview topics it relies on; check every overview topic is reachable from
   at least one scenario.
7. [ ] **Browsable index.** Group `INDEX.md` by category with a count per
   section; one 320-row table is hard to scan.
