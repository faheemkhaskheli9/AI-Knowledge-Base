---
title: Design tooling for Claude (Figma MCP and plugin equivalents)
category: llm-apps
tags: [design, figma, mcp, ui, design-to-code, accessibility, playwright, claude-code, design-plugins]
use_cases:
  - "turn a Figma design into React/HTML code with Claude Code"
  - "let Claude read our design system tokens and components from Figma"
  - "have an agent check a UI for accessibility and layout problems"
  - "find a Claude-usable replacement for a Sketch/Figma plugin workflow"
  - "generate a FigJam diagram or Figma screens from an agent"
status: draft
last_verified: 2026-10-04
sources:
  - https://developers.figma.com/docs/figma-mcp-server/remote-server-installation/
  - https://developers.figma.com/docs/figma-mcp-server/tools-and-prompts/
  - https://help.figma.com/hc/en-us/articles/39888612464151-Claude-Code-and-Figma-Set-up-the-MCP-server
  - https://github.com/goabstract/Awesome-Design-Tools/blob/master/Awesome-Design-Plugins.md
  - https://github.com/microsoft/playwright-mcp
---

# Design tooling for Claude (Figma MCP and plugin equivalents)

## Summary
Design-app plugins (Sketch, Figma, Adobe XD) are GUI extensions that run inside
those apps; an LLM agent cannot install or call them. To give Claude design
capabilities, use **MCP servers** (above all Figma's official MCP server) and
command-line tools the agent can run. This file maps the plugin categories of
the Awesome-Design-Plugins list to what Claude can actually use.

## Key concepts
- **Figma MCP server**: Figma's official MCP server. The **remote** version
  (`https://mcp.figma.com/mcp`, OAuth) needs no desktop app and has the most
  tools. The **desktop** version (`http://127.0.0.1:3845/mcp`) runs from the
  Figma desktop app.
- **Read tools**: `get_design_context` (code-ready context for a layer or
  selection), `get_metadata` (sparse XML outline), `get_screenshot`,
  `get_variable_defs` (variables/styles = design tokens), `search_design_system`,
  `get_code_connect_map`, `get_figjam`.
- **Write tools** (remote only): `use_figma` (general create/edit/inspect),
  `generate_figma_design`, `create_new_file`, `upload_assets`,
  `generate_diagram` (Mermaid to FigJam).
- **Code Connect** maps Figma components to your real code components, so the
  generated code imports `<Button>` instead of re-drawing it.
- **The plugin list is a catalogue of needs, not tools**: use its categories as
  a checklist of design workflows, then pick the agent-side equivalent below.

## When to use / scenarios
- Design-to-code: a frontend team hands Claude Code a Figma frame link and
  gets components that use the existing design tokens.
- Design-system audits: pull variables and components, then compare them with
  the codebase's theme file.
- Diagramming: an agent turns an architecture description into a FigJam board.
- UI QA: an agent drives the running app in a browser, takes screenshots and
  runs accessibility checks.
- **Not needed** when there is no Figma source. For greenfield UI, prompt with
  a clear brief plus the project's design rules (an Agent Skill or CLAUDE.md
  section), and verify in a browser.

### Plugin category -> Claude-usable equivalent
| Awesome-Design-Plugins category | What Claude can use |
|---|---|
| Code Export, Website & HTML Export, Developers Handoff | Figma MCP `get_design_context` + Code Connect |
| Style Management, Color Management, Typeface | Figma MCP `get_variable_defs`, `search_design_system`; tokens in code |
| Symbols & Components Management, UI Kits | Code Connect (`get_code_connect_map`, `add_code_connect_map`) |
| Accessibility | Playwright MCP + `@axe-core/cli` or Lighthouse on the running page |
| Presentation & Preview, Prototyping, Mockup | Run the app; Playwright MCP screenshots; `get_screenshot` for the design side |
| Data Generation | Claude writes realistic fixtures directly (or Faker in code) |
| User Flows, Map Generation, charts | `generate_diagram` (Mermaid -> FigJam), or Mermaid in docs |
| Images Management, Import, Export | `upload_assets` / `download_assets` (remote server) |
| Translation & Localization, Text Management | Claude edits i18n files directly ([[structured-output]] for string tables) |
| Rename Helper, Align & Arrange, Layout & Padding, Resize | `use_figma` (write access, remote only) |
| Version Control, Collaboration | git for code; Figma's own version history for files |

## Setup & code
Claude Code, Figma remote server (preferred: plugin, which also bundles Agent
Skills for common Figma workflows):
```bash
claude plugin install figma@claude-plugins-official
```
Manual alternative (`--scope user` makes it available in every project):
```bash
claude mcp add --scope user --transport http figma https://mcp.figma.com/mcp
# start a new Claude Code session -> /mcp -> figma -> Authenticate -> Allow Access
```
Desktop server instead (Figma desktop app running, MCP server enabled in it):
```bash
claude mcp add --transport http figma-desktop http://127.0.0.1:3845/mcp
```
Then prompt with a frame link, e.g. *"Implement this frame as a React component
using our existing tokens: https://www.figma.com/design/<file>?node-id=1-2"*.

Browser-side checks the agent can run:
```bash
claude mcp add playwright -- npx @playwright/mcp@latest   # agent can open pages, click, screenshot
npx @axe-core/cli http://localhost:3000                    # accessibility violations
npx lighthouse http://localhost:3000 --only-categories=accessibility --output=json
```

## Choosing / trade-offs
- **Remote vs desktop Figma server**: remote has more tools (all write tools,
  library search, asset download) and works without the app; desktop works off
  the selection in the open app and keeps traffic local.
- **Figma MCP vs screenshot-only**: a pasted screenshot gives Claude the look
  but not the tokens, spacing values or component names, so the code drifts
  from the design system. Use MCP whenever a Figma file exists.
- **Code Connect setup cost** pays off on a mature component library; skip it
  for a one-off landing page.

## Gotchas
- Large selections blow up context. Pick a single frame, or call
  `get_metadata` first and then `get_design_context` on the needed nodes.
- Output follows the Figma file's quality: unnamed layers, absolute positioning
  and missing auto-layout produce poor code. Clean the frame first.
- Write tools change real files. Use a scratch file or branch, as with any
  agent write access ([[guardrails-and-safety]]).
- Design files and their text are untrusted input to the model
  ([[prompt-injection]]).
- The Awesome-Design-Plugins list is mostly Sketch plugins and is no longer
  actively maintained. Use it to see which workflows exist, not to find current tools.

## Related
- [[model-context-protocol]] - how MCP servers plug into Claude.
- [[coding-agents]] - Claude Code setup the Figma server attaches to.
- [[computer-use-agents]] - fallback for design apps with no API/MCP.
- [[guardrails-and-safety]] - scoping write access.

## References
- Figma MCP remote setup: https://developers.figma.com/docs/figma-mcp-server/remote-server-installation/
- Figma MCP tools: https://developers.figma.com/docs/figma-mcp-server/tools-and-prompts/
- Figma help, Claude Code: https://help.figma.com/hc/en-us/articles/39888612464151-Claude-Code-and-Figma-Set-up-the-MCP-server
- Playwright MCP: https://github.com/microsoft/playwright-mcp
- Awesome Design Plugins: https://github.com/goabstract/Awesome-Design-Tools/blob/master/Awesome-Design-Plugins.md
