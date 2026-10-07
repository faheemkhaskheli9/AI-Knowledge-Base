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
   its overview topic), or agents following links never reach it.
5. Run `python build_index.py` (stdlib only). It regenerates `INDEX.md`, fails
   on missing frontmatter or a broken `[[link]]`, and warns on orphan topics.
6. Run `python check_code.py <your file>`. It runs the file's python blocks (needs
   numpy etc. installed). Mark a deliberate fragment with `<!-- skip-check: reason -->`
   above its fence; give a slow benchmark `<!-- check-timeout: 900 -->`.
7. Use LF line endings and UTF-8.

### Status
- `draft`: new, or its code has not been run by `check_code.py`.
- `stable`: `check_code.py` ran its code clean and it cites sources (`python check_code.py --promote`
  sets this). It means the code runs, not that the facts are current: check `last_verified`.
- `needs-review`: was `stable` and its code now fails, or its facts need re-checking.
Topics whose code needs torch, API keys or downloads stay `draft` until run in a full environment.

`python build_index.py` also warns about `models/`, `setup/` and `deployment/` topics not
verified in 90 days.

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
This repo is one of three: `faheemkhaskheli9/Cyber-Security-Knowledge-Base`, `faheemkhaskheli9/Personal-Knowledge-Base`,
`faheemkhaskheli9/AI-Knowledge-Base`. If a task could use the others and they are not in the session, attach them
with `add_repo` (read access is enough unless you need to push) and clone them next to this one. Each has a
`webapp/` for manual viewing and editing (`python3 webapp/server.py --open`).
