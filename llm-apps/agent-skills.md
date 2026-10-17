---
title: Agent skills (SKILL.md) and progressive disclosure
category: llm-apps
tags: [agent-skills, skill-md, progressive-disclosure, claude-code, claude-agent-sdk, agents-md, claude-md]
use_cases:
  - "package our team's code-review or release procedure so a coding agent follows it on demand"
  - "give an agent dozens of procedures without filling its context window"
  - "decide whether a capability belongs in a skill, an MCP server or CLAUDE.md/AGENTS.md"
  - "write a skill description that triggers reliably"
  - "use pre-built document skills (xlsx, pptx, docx, pdf) through the Claude API"
status: draft
last_verified: 2026-10-15
sources:
  - https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview
  - https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices
  - https://platform.claude.com/docs/en/build-with-claude/skills-guide
  - https://code.claude.com/docs/en/skills
  - https://agentskills.io
  - https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills
---

# Agent skills (SKILL.md) and progressive disclosure

## Summary
An agent skill is a folder with a `SKILL.md` file (YAML frontmatter + Markdown instructions) and optional scripts, templates and reference files. The agent sees only each skill's `name` and `description` up front; when a request matches, it reads the body, and it opens bundled files or runs bundled scripts only if the task needs them. This **progressive disclosure** lets an agent carry many procedures at almost no context cost. Skills started in Claude (Claude Code, Claude API, claude.ai, Agent SDK) and follow the Agent Skills open standard (agentskills.io), which other agent tools can read.

## Key concepts
- **Three loading levels** (Anthropic docs): (1) metadata, always in the system prompt, about 100 tokens per skill; (2) the SKILL.md body, loaded when triggered, recommended under 5k tokens; (3) bundled files and scripts, loaded or executed on demand. A script's code never enters context, only its output.
- **Frontmatter (portable fields)**: `name` (required; max 64 chars; lowercase letters, numbers, hyphens; no XML tags; must not contain "anthropic" or "claude") and `description` (required; non-empty; max 1024 chars; no XML tags). The standard also allows `license`, `compatibility`, `metadata`, `allowed-tools`. Uploads via claude.ai or the Skills API reject other fields.
- **Claude Code extensions**: `disable-model-invocation` (only the user can run it, for side-effecting workflows), `user-invocable: false` (only the model can), `allowed-tools`, `context: fork` (run in a sub-agent), `paths`, `model`, `when_to_use`, argument substitution (`$ARGUMENTS`). Custom slash commands were merged into skills.
- **The description is the trigger**: the model decides to load a skill by matching the request against `description`, so it must say *what* the skill does and *when* to use it, with the words users actually say.
- **Locations**: Claude Code reads `~/.claude/skills/<name>/SKILL.md` (personal), `.claude/skills/` (project) and plugin skills; the Claude API uses uploaded skills (`/v1/skills`) or pre-built ones (`xlsx`, `pptx`, `docx`, `pdf`) inside the code-execution container; claude.ai takes zip uploads. Skills do not sync across these surfaces.
- **Runtime differs by surface**: on the Claude API the skill container has no network and no runtime package installs; in Claude Code a skill has the same access as any local program.

## When to use / scenarios
- Repeatable procedures with a known shape: "release a version", "write a migration", "review a PR against our checklist", "produce the monthly finance workbook".
- Organization knowledge that is long but rarely needed all at once: API style guides, schema references, brand rules.
- Deterministic steps better done by a script than by generated code: form filling, validation, file conversion.
- Not for always-true project facts (build command, repo layout): those go in CLAUDE.md / AGENTS.md, which are always loaded.
- Not for live access to external systems: that is an MCP server or tool ([[model-context-protocol]], [[tool-calling]]). A skill can teach the agent *how* to use those tools well.

## Setup & code
A skill folder:
```
release-notes/
  SKILL.md            # frontmatter + instructions (level 2)
  TEMPLATE.md         # read only when writing notes (level 3)
  scripts/changelog.py  # run, not read (level 3)
```
`SKILL.md`:
```markdown
---
name: release-notes
description: Draft release notes from merged PRs and the changelog. Use when the user asks for release notes, a changelog entry, or "what shipped" for a version or date range.
---
# Release notes
1. Run `python scripts/changelog.py <from-tag> <to-tag>` to list merged PRs.
2. Group by Features / Fixes / Breaking changes; drop internal-only PRs.
3. Fill TEMPLATE.md; link each item to its PR.
```
Runnable sketch of how a host implements discovery: parse frontmatter, validate the portable constraints, build the level-1 listing for the system prompt, and load a body only when triggered (a real agent lets the model choose; keyword overlap stands in here):
```python
import re, tempfile
from pathlib import Path

root = Path(tempfile.mkdtemp())
for name, desc, body in [
    ("release-notes", "Draft release notes from merged PRs. Use when the user asks for release notes or a changelog.", "1. Run changelog.py ..."),
    ("sql-style", "Our SQL style guide. Use when writing or reviewing SQL queries or migrations.", "Use snake_case; CTEs over subqueries ..."),
]:
    (root / name).mkdir()
    (root / name / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {desc}\n---\n# {name}\n{body}\n")

def parse(path):
    m = re.match(r"---\n(.*?)\n---\n(.*)", path.read_text(), re.S)
    meta = dict(line.split(": ", 1) for line in m.group(1).splitlines())
    return meta, m.group(2)

def validate(meta, folder):
    errs = []
    n, d = meta.get("name", ""), meta.get("description", "")
    if not re.fullmatch(r"[a-z0-9-]{1,64}", n): errs.append("name: 1-64 chars, lowercase/digits/hyphens")
    if any(w in n for w in ("anthropic", "claude")): errs.append("name: reserved word")
    if not d or len(d) > 1024 or re.search(r"<[^>]+>", d): errs.append("description: 1-1024 chars, no XML tags")
    if n != folder: errs.append("name should match folder name")
    return errs

skills = {}
for f in sorted(root.glob("*/SKILL.md")):
    meta, body = parse(f)
    assert not validate(meta, f.parent.name), validate(meta, f.parent.name)
    skills[meta["name"]] = (meta["description"], f)

level1 = "\n".join(f"- {n}: {d}" for n, (d, _) in skills.items())        # always in the system prompt
print("Available skills:\n" + level1, f"\n(~{len(level1)//4} tokens for {len(skills)} skills)")

request = "please review this SQL migration"
words = set(re.findall(r"[a-z]+", request.lower()))
best = max(skills, key=lambda n: len(words & set(re.findall(r"[a-z]+", skills[n][0].lower()))))
print(f"\nTriggered: {best}\n" + parse(skills[best][1])[1])                # level 2 loaded on demand
```
Using a pre-built skill on the Claude API (model ID from [[anthropic-claude]]; up to 20 skills per request; requires the code-execution tool):
<!-- skip-check: needs an Anthropic API key -->
```python
import anthropic
client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-opus-5-5",
    max_tokens=4096,
    container={"skills": [{"type": "anthropic", "skill_id": "xlsx", "version": "latest"}]},
    messages=[{"role": "user", "content": "Build a 12-month budget workbook with a summary sheet"}],
    tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
)
```

## Choosing / trade-offs
| Put it in... | When | Context cost | Example |
|---|---|---|---|
| CLAUDE.md / AGENTS.md | Always-true facts and conventions for this repo | Every turn | "Run `pytest -q`; never edit `generated/`" |
| Skill | A procedure or reference needed for some tasks | ~100 tokens until triggered | Release checklist, SQL style guide, PDF form filling |
| MCP server / tool | Live data or actions in an external system | Tool schemas every turn (unless deferred) | Query Jira, read a database |
| Sub-agent | A task that needs its own clean context | Only its returned summary | Broad codebase search |

- **Skill vs MCP**: MCP gives *capabilities* (connectivity); a skill gives *know-how* (procedure, conventions, scripts). They combine: a skill can say which MCP tools to call, in what order.
- **Instructions vs scripts inside a skill**: prose for judgment calls; scripts for anything that must be exact or repeatable.
- **Model-invoked vs user-invoked**: let the model trigger read-only or advisory skills; require explicit user invocation for deploys, emails, payments.
- **One big skill vs several small ones**: small, single-purpose skills trigger more precisely; a big one wastes level-2 tokens on unused sections (split into referenced files instead).

## Gotchas
- A vague description ("Helps with documents") never triggers or triggers on everything. Name the file types, verbs and user phrasings; test with real requests.
- SKILL.md bodies that grow past a few thousand tokens defeat progressive disclosure; move detail into referenced files one level deep.
- Skills are code you execute: a malicious or compromised skill can direct tool calls or exfiltrate data. Install only from trusted sources and audit scripts and any URLs they fetch ([[prompt-injection]], [[sandboxed-code-execution]]).
- Surface differences: Claude Code-only frontmatter fields cause upload errors on claude.ai / the Skills API; API skills have no network.
- Skills on one surface are not available on another; keep one source folder and publish to each.
- An "Agent Card skill" in [[a2a-protocol]] is an advertised capability of a remote agent, not a SKILL.md package.

## Related
- [[coding-agents]] - Claude Code, Codex and Cursor, where skills, CLAUDE.md and AGENTS.md live.
- [[model-context-protocol]] - connectivity to external systems; complements skills.
- [[context-engineering]] - progressive disclosure is one of its core techniques.
- [[agent-memory]] - procedural memory stored as files and skills.
- [[agent-frameworks]] - the Claude Agent SDK loads skills from the filesystem.
- [[sandboxed-code-execution]] - where skill scripts run.
- [[prompt-engineering]] - writing the instructions inside a skill.
- [[dspy-and-prompt-optimization]] - optimizing prompts automatically instead of by hand.

## References
- Agent Skills overview (levels, frontmatter limits, security): https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview
- Skill authoring best practices: https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices
- Using skills with the API: https://platform.claude.com/docs/en/build-with-claude/skills-guide
- Skills in Claude Code: https://code.claude.com/docs/en/skills
- Agent Skills open standard: https://agentskills.io
- Anthropic engineering, Equipping agents for the real world with Agent Skills: https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills
