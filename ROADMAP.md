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
   orphans into their parent topic's `## Related`. Issue #3: done; CI also
   fails when `INDEX.md`/`llms.txt` are stale, and `.github/workflows/links.yml`
   runs `check_links.py` weekly (and on demand) over every `sources:` URL,
   failing on dead links and warning on 403/429.
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
   Checked 2026-10-15: `scenario-chooser` is stable; `model-selection`,
   `rag-basics`, `agents` and `tool-calling` stay draft because `check_code.py`
   skips their code (needs API keys or network), so it has not run clean. They
   need a run in an environment with keys before promotion.
4. [x] **Staleness report.** `build_index.py` lists `models/`, `setup/` and
   `deployment/` files whose `last_verified` is older than 90 days, the ones
   whose facts (model IDs, prices, versions) rot. Issue #4 closed: the index now
   includes a `Verified` column and `--stale N` configures the threshold.
5. [x] **Fill thin categories.** Add applied topics before more from-scratch
   ones: `models/` (embedding and reranker models, open-weight vision LLMs),
   `setup/` (vLLM/llama.cpp server on Windows, Docker GPU), `deployment/`
   (LLM gateways and rate limits, batch inference, caching). Tracker #98,
   one `topic:*` issue per topic, plus #6, #7 and #10 (A2A, context
   engineering, agent skills).
   Done: tracker #98 and #6, #7, #10 closed (models 27, setup 18,
   deployment 36 topics as of 2026-10-24).
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
10. [x] **Verify flagged facts.** Replace or remove the "not verified" flags
    and the Claude sampling-parameter note. Issues #5, #8. Done 2026-10-15:
    flags replaced with cited facts (Gemini 3.5 Flash-Lite ID; SD 3.5, Qwen-Image-2.1,
    SmolLM2, Llama 4, InsightFace and SAM 2 licences) or removed; Claude 4.7+
    sampling-parameter deprecation and the Sonnet 4.5 retirement (marked
    `valid-until: 2026-11-30`) documented in decoding-and-sampling,
    anthropic-claude and api-sdk-setup.
11. [x] **Agent entry points.** `llms.txt`, a glossary and learning paths.
    Issues #13, #96, #97. #13 done: `build_index.py` writes `llms.txt` from the
    same rows as `INDEX.md` and CI diff-checks both. #96 and #97 done 2026-10-17:
    `glossary.md` (`- **Term** - definition. [[topic]]`, alphabetical) and
    `learning-paths.md` (ordered paths by role); `build_index.py` fails on a
    broken `[[link]]` in either and lists both under `## Optional` in `llms.txt`.
