---
title: Coding agents (Claude Code, Codex, Cursor) and project instruction files
category: llm-apps
tags: [coding-agents, claude-code, codex, cursor, agents-md, claude-md, mcp, subagents]
use_cases:
  - "set up Claude Code or Codex on a repo with the right project instructions"
  - "write a CLAUDE.md / AGENTS.md that makes agents follow our conventions"
  - "choose between Claude Code, Codex CLI, Cursor and Copilot for a team"
  - "run a coding agent safely in CI or on untrusted repositories"
  - "give a coding agent access to internal docs and tools via MCP"
status: draft
last_verified: 2026-10-03
sources:
  - https://code.claude.com/docs/en/overview
  - https://code.claude.com/docs/en/memory
  - https://agents.md/
  - https://github.com/openai/codex
  - https://docs.cursor.com/
---

# Coding agents (Claude Code, Codex, Cursor)

## Summary
Coding agents are LLM agents with file-edit, shell, search and git tools that work inside a repository: they read code, plan, edit, run tests and iterate. The leading ones are Claude Code (terminal/IDE/CI), OpenAI Codex (CLI/cloud/IDE), Cursor (AI IDE) and GitHub Copilot's agent mode. Their output quality depends heavily on project instruction files, tests the agent can run, and permission settings.

## Key concepts
- **Instruction files**: Claude Code reads `CLAUDE.md` (project, user-level `~/.claude/CLAUDE.md`, and nested directories); **`AGENTS.md`** is a cross-tool convention read by Codex, Cursor, Copilot and many others (see agents.md). When both are needed, keep one source of truth and have `CLAUDE.md` import or point to `AGENTS.md` (Claude supports `@path` imports - check the memory docs). Cursor also has its own rules directory.
- **What to put in them**: build/test/lint commands (exact), project layout, code style that linters do not enforce, branch/commit rules, "never do" list, gotchas (generated files, env vars). Keep short and specific; they load into every session's context.
- **Permissions and modes**: ask-before-edit, auto-accept edits, plan mode (read-only planning), allowlists for commands; sandboxing and approval policies in Codex. Avoid "skip all permissions" outside isolated environments.
- **Extensibility (Claude Code)**: slash commands, **skills**, **subagents** (isolated context, restricted tools), **hooks** (deterministic scripts at lifecycle events), **MCP servers** ([[model-context-protocol]]), plugins. Cursor and Codex support MCP too.
- **Headless/CI use**: Claude Code and Codex have non-interactive modes; the Claude Agent SDK embeds the same harness in your code ([[agent-frameworks]]).
- **Verification loop**: tests, type checks, linters, and a reviewable diff are what make autonomy safe.

## When to use / scenarios
- Feature work, bug fixes, refactors, test writing, migrations, code review, codebase Q&A.
- Team rollout: commit `AGENTS.md`/`CLAUDE.md`, shared allowlist settings, and MCP config so everyone's agent behaves the same.
- CI bots that fix lint, triage issues or review PRs (read-only by default).
- Less suitable: tasks without a verification signal, secrets-heavy environments without isolation, or work where nobody will review the diff.

## Setup & code
Minimal `AGENTS.md` (also usable as the body of `CLAUDE.md`):
```markdown
# Project instructions
## Commands
- Install: `uv sync`   Test: `uv run pytest -q`   Lint: `uv run ruff check .`
## Layout
- `src/app/` application code; `tests/` mirrors it; `migrations/` generated, never edit by hand
## Conventions
- Type hints required; no new dependencies without asking
- Small commits, imperative messages; never commit to `main` directly
## Do not
- Touch `.env` or print secrets; run destructive DB commands
```
`CLAUDE.md` that reuses it:
```markdown
@AGENTS.md
## Claude-specific
- Prefer plan mode for changes touching more than 3 files
```
Install and run (verify current commands in each tool's docs):
```bash
# Claude Code: see https://code.claude.com/docs/en/overview for the current installer
claude                      # interactive session in the repo
claude -p "fix the failing test in tests/test_api.py"   # headless one-shot
# Codex CLI: https://github.com/openai/codex  (e.g. npm i -g @openai/codex ; then: codex)
```
Add an MCP server to Claude Code: `claude mcp add docs -- python server.py` ([[model-context-protocol]]).

## Choosing / trade-offs
| Tool | Strength |
|---|---|
| Claude Code | Terminal-first, strong agentic loop, subagents/hooks/skills/MCP, SDK for embedding |
| Codex | CLI plus cloud tasks in sandboxes, OpenAI models, AGENTS.md native |
| Cursor | IDE-centric flow: inline edits, tab completion, background agents |
| Copilot agent mode | Deep GitHub/PR integration, enterprise controls |
- **IDE-integrated vs terminal agent**: IDE for tight edit loops; terminal/headless for autonomy, scripting and CI.
- **Autonomy vs review**: more autonomy needs better tests and sandboxing.
- **Vendor lock-in**: keep conventions in `AGENTS.md` and tools in MCP so you can switch.
- **Cost**: long sessions consume many tokens; cache and use smaller models for routine edits ([[prompt-caching-and-cost]]).

## Gotchas
- Bloated instruction files get ignored or waste context; prune and keep rules verifiable.
- Instruction files conflict across levels (user, project, nested); know the precedence.
- Agents confidently claim tests pass without running them; require the command output.
- Untrusted repos, issues and dependencies can contain injected instructions ([[prompt-injection]]); do not run agents with broad credentials on them.
- Auto-approve flags with network and secrets access are how incidents happen ([[guardrails-and-safety]]).
- Large generated or vendored directories blow context; exclude them via ignore files.
- Stale commands in instruction files cause repeated failures; update them when tooling changes.

## Related
- [[agents]] - the underlying loop.
- [[model-context-protocol]] - extend agents with tools and docs.
- [[agent-frameworks]] - Claude Agent SDK for custom harnesses.
- [[multi-agent-systems]] - subagent patterns.
- [[software-engineering]] - scenario guidance.
- [[prompt-injection]] - security for repo-reading agents.

## References
- Claude Code docs: https://code.claude.com/docs/en/overview
- Claude Code memory (CLAUDE.md): https://code.claude.com/docs/en/memory
- AGENTS.md convention: https://agents.md/
- Codex CLI: https://github.com/openai/codex
- Cursor docs: https://docs.cursor.com/
