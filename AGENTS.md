# Instructions for AI agents (Claude Code, Codex, Cursor, Copilot, ...)

This repo is a reference knowledge base on AI technologies: models, concepts,
setup/code, deployment, and the real-world scenarios each one fits.

## Consulting it

1. Read `INDEX.md`. Match your task against the **Tags** and **Use when** columns.
2. Read the **full** matched file(s), not just the index row. Follow `[[links]]`
   in the `## Related` section when they apply.
3. For "which approach/model for X?" start in `scenarios/`, which maps
   problems to topic files.
4. Volatile facts (model names, versions, prices, context sizes) are only as
   fresh as the file's `last_verified` date. If it is older than ~3 months
   and the decision depends on it, check the linked official source first.
5. Mention which file(s) informed your work.

Quick search without reading the index: `grep -ril "<keyword>" --include=*.md .`

## Adding or editing a topic

1. Copy `_template.md` into the right category folder. Use one topic per file
   and a kebab-case filename.
2. Fill in every section. Code must be minimal and runnable, and use the
   official SDK/API.
3. Cite official sources for volatile facts. Set `last_verified` to today.
   Do not invent version numbers, benchmarks or prices; leave them out instead.
4. Link a new topic from at least one existing topic's `## Related` (usually
   its overview topic), or agents following links never reach it. A new
   overview (not from-scratch) topic also needs a link from a `scenarios/` file,
   usually a row in `scenarios/scenario-chooser.md`.
5. Run `python build_index.py` (stdlib only). It regenerates `INDEX.md`, fails
   on missing frontmatter or a broken `[[link]]`, and warns on orphan topics
   and on overview topics no scenario links to. CI runs it with `--strict`,
   which turns those two warnings into failures (stale warnings stay warnings).
6. Run `python check_code.py <your file>`. It runs the file's python blocks (needs
   numpy etc. installed). Mark a deliberate fragment with `<!-- skip-check: reason -->`
   above its fence; give a slow benchmark `<!-- check-timeout: 900 -->`.
7. Use LF line endings and UTF-8.

### Status
- `draft`: new, or its code has not been run by `check_code.py`.
- `stable`: `check_code.py` ran its code clean and it cites sources (`python check_code.py --promote`
  sets this). It means the code runs, not that the facts are current: check `last_verified`.
  A topic with no python block (`check_code.py` reports "no code") becomes `stable` when every
  URL in `sources` resolves and someone other than its author has read the whole file and found
  no wrong claims; change only its `status:` line and say who reviewed it in the commit.
- `needs-review`: was `stable` and its code now fails, or its facts need re-checking.
Topics whose code needs torch, API keys or downloads stay `draft` until run in a full environment.

`python build_index.py` also warns about topics not verified in 90 days when they sit in
`models/`, `setup/` or `deployment/`, or anywhere else name an API model ID (`claude-...`,
`gpt-...`, `gemini-...`). Example code copies its model ID from the table in
`models/<provider>.md`, so when an ID is retired that table says what replaces it.

A fact with a known end date (an introductory price, a deprecation date) gets
`<!-- valid-until: YYYY-MM-DD -->` on its line; `build_index.py` warns once that date is
14 days away or has passed, whatever the file's `last_verified` says.

Planned improvements live in `ROADMAP.md`.

## Categories

| Folder | Holds |
|---|---|
| `concepts/` | How AI works: transformers, embeddings, training, fine-tuning, quantization, diffusion, ... |
| `models/` | Model families and providers: what they are good at, how to call them |
| `llm-apps/` | Building with LLMs: prompting, RAG, tool use, agents, MCP, evals, guardrails |
| `ml/` | Classical ML and deep learning workflows (tabular, time series, recommenders) |
| `cv/` | Computer vision tasks and models |
| `speech/` | Speech-to-text, text-to-speech, audio |
| `setup/` | Local environments: GPU/CUDA, Ollama, llama.cpp, Hugging Face, API SDKs |
| `deployment/` | Serving, scaling, cost/latency, MLOps, monitoring |
| `scenarios/` | Industry/problem -> recommended approach, linking the files above |

## Sibling knowledge bases
| Repo | Holds | `webapp/` |
|---|---|---|
| `faheemkhaskheli9/AI-Knowledge-Base` (this one) | AI models, concepts, setup, deployment, scenarios | yes |
| `faheemkhaskheli9/Software-Engineering-KnowledgeBase` | system design: load balancing, caching, queues, rate limits, resilience, SLOs, observability, security, testing | no |
| `faheemkhaskheli9/Personal-Knowledge-Base` (private) | the owner's profile, CVs and per-project notes | yes |
| `faheemkhaskheli9/Cyber-Security-Knowledge-Base` | security topics | yes |

If a task could use another one and it is not in the session, attach it with `add_repo` (read access
is enough unless you need to push) and clone it next to this one. Where there is a `webapp/`, it is
for manual viewing and editing (`python3 webapp/server.py --open`).
An AI system's general system-design side belongs to the SE KB. Link its topics as plain
"See also (SE KB): <GitHub URL>" lines, never `[[links]]` (the build checks those against this
repo only).
