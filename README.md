# AI Knowledge Base

A reference on AI technologies covering models, core concepts, setup and code,
deployment, and the scenarios each one fits. It is written as plain Markdown
so that people and AI coding agents (Claude Code, Codex, Cursor, Copilot)
can look things up before they build.

- **Start here:** [INDEX.md](INDEX.md) lists every topic with its tags and use cases.
- **By problem:** [scenarios/](scenarios/) maps an industry or task to the
  recommended approach.
- **Terms:** [glossary.md](glossary.md) defines core terms (KV cache, LoRA, MCP, RAG, ...), each
  linked to the topic that explains it.
- **Learning:** [learning-paths.md](learning-paths.md) gives ordered reading paths by role
  (beginner, LLM engineer, agent builder, ML engineer, CV, speech, MLOps).
- **For agents:** [AGENTS.md](AGENTS.md) explains how to consult the base and how to extend it;
  [llms.txt](llms.txt) is a compact one-line-per-topic index ([llmstxt.org](https://llmstxt.org/) format),
  generated with INDEX.md.

## Layout

One topic per file, grouped by category (`concepts/`, `models/`, `llm-apps/`,
`ml/`, `cv/`, `speech/`, `setup/`, `deployment/`, `scenarios/`). Each file
opens with YAML frontmatter (`tags`, `use_cases`, `status`, `last_verified`,
`sources`) followed by the same sections: Summary, Key concepts, When to use,
Setup & code, Trade-offs, Gotchas, Related, References.

## Which knowledge base answers what

- **This repo (AI KB):** which model, library or AI technique to use and how
  to set it up: LLM APIs, RAG, agents, fine-tuning, CV, speech, local
  inference, model serving and cost.
- **Software Engineering KB**
  (https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase):
  the general system around the model: architecture, APIs, databases,
  caching, queues, load balancing, rate limiting, resilience, SLOs,
  observability, deployment, security and testing.

Deployment and LLM-app topics here end their `## Related` section with
"See also (SE KB)" links to the matching SE KB topic.

## Contributing

Copy [_template.md](_template.md) to start a topic, then run
`python build_index.py` to regenerate the index. It also fails on a broken
`[[link]]` in `glossary.md` or `learning-paths.md`; add a glossary term for a
new core concept. Facts that change over time,
such as model names, prices and context sizes, must cite a source and carry a
`last_verified` date. The generated index includes that date in its **Verified**
column; `stable` is not a guarantee that facts are still current.

```bash
python build_index.py --stale 30          # warn on volatile topics older than 30 days
python build_index.py --strict --stale 90 # also fail on orphan/unreached topics
python -m unittest test_build_index_freshness
```

Staleness warnings cover `models/`, `setup/`, `deployment/`, and topics naming
API model IDs. The default threshold is 90 days; missing or malformed dates
also warn. These warnings do not fail the build, including with `--strict`.

## Web app (view and edit)

```bash
python3 webapp/server.py --open     # http://localhost:8773 , stdlib only, no install
```

![The AI KB web app showing the topic on RAG basics](docs/screenshots/webapp.png)

**Where to use it**
- Choosing an approach or model: open `scenarios/scenario-chooser.md`, follow the `[[links]]`,
  and search across all topics without leaving the page.
- Adding or fixing a topic: edit it in place, press **Rebuild INDEX.md**, then review the
  diff and commit with git.
- On your own machine only. It is a local tool for the person who maintains the KB, not a
  site to host.

**Why use it instead of GitHub or an editor**
- Rendered Markdown with working `[[wiki-links]]`, which GitHub does not resolve.
- One search box over every file, with no install: it is a single stdlib Python script.
- Safe edits: writes stay inside the repo and refuse to overwrite a file changed on disk.

Browse the file tree, read rendered Markdown (links and `[[wiki-links]]` work), full-text search, edit raw
text (Ctrl+S saves) and create new files. Edits write straight to the files here; review and commit with git.
The **Rebuild INDEX.md** button runs `build_index.py` after you add or edit topics.
It listens on 127.0.0.1 only, rejects foreign Host/Origin headers, needs a per-run token for writes, only touches
text files inside the repo, and refuses to overwrite a file that changed on disk since you opened it.

## Read-only Django site

A read-only Django site over the same Markdown files, with search, category
and tag filters, `[[links]]` and backlinks. It has no database, so edits to the
files show up on refresh.

```bash
pip install -r requirements.txt
python manage.py runserver      # http://127.0.0.1:8000
python manage.py test webapp
```
