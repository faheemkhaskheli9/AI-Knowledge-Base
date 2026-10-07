# AI Knowledge Base

A reference on AI technologies covering models, core concepts, setup and code,
deployment, and the scenarios each one fits. It is written as plain Markdown
so that people and AI coding agents (Claude Code, Codex, Cursor, Copilot)
can look things up before they build.

- **Start here:** [INDEX.md](INDEX.md) lists every topic with its tags and use cases.
- **By problem:** [scenarios/](scenarios/) maps an industry or task to the
  recommended approach.
- **For agents:** [AGENTS.md](AGENTS.md) explains how to consult the base and how to extend it.

## Layout

One topic per file, grouped by category (`concepts/`, `models/`, `llm-apps/`,
`ml/`, `cv/`, `speech/`, `setup/`, `deployment/`, `scenarios/`). Each file
opens with YAML frontmatter (`tags`, `use_cases`, `status`, `last_verified`,
`sources`) followed by the same sections: Summary, Key concepts, When to use,
Setup & code, Trade-offs, Gotchas, Related, References.

## Contributing

Copy [_template.md](_template.md) to start a topic, then run
`python build_index.py` to regenerate the index. Facts that change over time,
such as model names, prices and context sizes, must cite a source and carry a
`last_verified` date.

## Web app (view and edit)

```bash
python3 webapp/server.py --open     # http://localhost:8773 , stdlib only, no install
```

Browse the file tree, read rendered Markdown (links and `[[wiki-links]]` work), full-text search, edit raw
text (Ctrl+S saves) and create new files. Edits write straight to the files here; review and commit with git.
The **Rebuild INDEX.md** button runs `build_index.py` after you add or edit topics.
It listens on 127.0.0.1 only, rejects foreign Host/Origin headers, needs a per-run token for writes, only touches
text files inside the repo, and refuses to overwrite a file that changed on disk since you opened it.
