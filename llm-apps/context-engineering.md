---
title: Context engineering for agents
category: llm-apps
tags: [context-engineering, compaction, context-rot, progressive-disclosure, tool-result-clearing, sub-agents, context-editing]
use_cases:
  - "stop a long-running coding or research agent from degrading as its context fills up"
  - "decide what goes into the context window on each turn of an agent loop"
  - "cut token cost of an agent whose tool results are huge (logs, web pages, SQL results)"
  - "split a big investigation across sub-agents so the lead agent stays focused"
status: draft
last_verified: 2026-10-15
sources:
  - https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
  - https://platform.claude.com/docs/en/build-with-claude/context-editing
  - https://platform.claude.com/docs/en/build-with-claude/compaction
  - https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview
---

# Context engineering for agents

## Summary
Context engineering is deciding, on every model call, which tokens the model sees: system prompt, tool definitions, retrieved data, message history and tool results. Prompt engineering is about writing one good instruction; context engineering is about curating the whole window as an agent runs for many turns. The goal is the smallest set of high-signal tokens, because accuracy drops as context grows ("context rot") and every token costs money and latency. The main levers are compaction, tool-result trimming, just-in-time retrieval (progressive disclosure), external notes, and sub-agent isolation.

## Key concepts
- **Context is a budget, not a bucket**: a 1M-token window does not mean 1M useful tokens; recall and instruction-following degrade with length and with distractors ([[long-context]]).
- **Anatomy of the window**: stable prefix (system prompt, tool schemas, skill metadata) -> retrieved/working data -> conversation and tool results. Keep the prefix stable for caching ([[prompt-caching-and-cost]]); manage the growing tail.
- **Tool-result trimming / clearing**: old tool outputs are rarely needed verbatim. Truncate on return, keep only the last N results, replace older ones with a placeholder. The cheapest form of compaction.
- **Compaction**: when the window nears a threshold, summarize older turns (decisions, open questions, file paths, IDs) and continue from the summary plus recent turns. Tune the summary prompt for recall first, then precision.
- **Just-in-time retrieval / progressive disclosure**: keep references (file paths, URLs, query names, skill names) in context and load content only when needed through tools. Agent skills use three levels: metadata always, SKILL.md when triggered, bundled files on demand ([[agent-skills]]).
- **Offloading**: write big outputs to files or a store and return a short handle plus a preview; let the agent `grep`/`read` slices.
- **Structured notes**: the agent maintains a progress or NOTES file outside the window and re-reads it after compaction or in a new session ([[agent-memory]]).
- **Sub-agent isolation**: a sub-agent explores in its own fresh window and returns a condensed result (often a short summary) to the lead agent, so the lead's context holds conclusions, not search noise ([[multi-agent-systems]]).
- **Tool design is context design**: fewer, non-overlapping tools with tight descriptions and concise, paginated outputs ([[tool-calling]]).

## When to use / scenarios
- Agents that run tens to hundreds of turns: coding agents, research/deep-search agents, SRE/incident agents reading logs ([[coding-agents]]).
- Tools that return large payloads (HTML, logs, query results, PDFs).
- Cost or latency is dominated by input tokens re-sent each turn.
- Quality drops late in long sessions (forgets constraints, repeats work, loses track of files).
- Not worth much for single-shot calls or short chats: write a good prompt ([[prompt-engineering]]) and stop there.

## Setup & code
Runnable sketch of a context manager for an agent loop: trims tool results on arrival, offloads big ones to files, clears old results, and compacts old turns once a token budget is exceeded. The summarizer is a stub; in production it is an LLM call or the provider's compaction API.
```python
import json, os, tempfile

def tokens(msg):                         # rough estimate: ~4 characters per token
    return len(json.dumps(msg)) // 4

class Context:
    def __init__(self, budget=400, keep_tool_results=2, max_result_chars=400, keep_recent=4):
        self.system = {"role": "system", "content": "You are a log-triage agent. Cite file paths."}
        self.msgs, self.summary = [], []
        self.budget, self.keep_tool, self.max_chars, self.keep_recent = budget, keep_tool_results, max_result_chars, keep_recent
        self.offload_dir = tempfile.mkdtemp()

    def add_tool_result(self, name, text):
        if len(text) > self.max_chars:                       # offload + preview (just-in-time later)
            path = os.path.join(self.offload_dir, f"{name}-{len(self.msgs)}.txt")
            open(path, "w").write(text)
            text = text[: self.max_chars] + f"\n...[truncated, full output in {path}; read slices as needed]"
        self.msgs.append({"role": "tool", "name": name, "content": text})
        self._clear_old_tool_results()
        self._compact_if_needed()

    def add(self, role, text):
        self.msgs.append({"role": role, "content": text}); self._compact_if_needed()

    def _clear_old_tool_results(self):
        tool_idx = [i for i, m in enumerate(self.msgs) if m["role"] == "tool"]
        for i in tool_idx[: -self.keep_tool]:
            self.msgs[i] = {**self.msgs[i], "content": "[old tool result cleared]"}

    def _compact_if_needed(self):
        if sum(map(tokens, self.window())) <= self.budget or len(self.msgs) <= self.keep_recent:
            return
        old, self.msgs = self.msgs[: -self.keep_recent], self.msgs[-self.keep_recent:]
        self.summary += [m["content"][:80] for m in old if m["role"] in ("user", "assistant")]  # stub summarizer

    def window(self):
        head = [self.system] + ([{"role": "user", "content": "Summary of earlier work: " + " | ".join(self.summary)}]
                               if self.summary else [])
        return head + self.msgs

ctx = Context()
ctx.add("user", "Find why checkout errors spiked at 14:00 and propose a fix.")
for step in range(6):
    ctx.add("assistant", f"Step {step}: grep logs for 5xx in service-{step}.")
    ctx.add_tool_result("grep_logs", f"service-{step}: " + "ERROR timeout to payments-db " * 60)
ctx.add("assistant", "Root cause: payments-db connection pool exhausted after 13:58 deploy.")

win = ctx.window()
raw = 6 * len("ERROR timeout to payments-db " * 60) // 4
print("messages in window:", len(win), "| tokens:", sum(map(tokens, win)), "| raw tool tokens avoided:", raw)
print(win[1]["content"][:120])
```
Server-side equivalents on the Claude API (beta): **context editing** clears old tool results or thinking blocks by rule, and **compaction** replaces older turns with a model-written summary. Example from the context-editing docs (model ID from [[anthropic-claude]]):
<!-- skip-check: needs an Anthropic API key -->
```python
import anthropic
client = anthropic.Anthropic()
response = client.beta.messages.create(
    model="claude-opus-5-5",
    max_tokens=4096,
    messages=[{"role": "user", "content": "Create a simple command line calculator app using Python"}],
    tools=[{"type": "text_editor_20250728", "name": "str_replace_based_edit_tool", "max_characters": 10000},
           {"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
    betas=["context-management-2025-06-27"],
    context_management={"edits": [{
        "type": "clear_tool_uses_20250919",
        "trigger": {"type": "input_tokens", "value": 30000},     # start clearing above this
        "keep": {"type": "tool_uses", "value": 3},               # keep the 3 most recent tool results
        "clear_at_least": {"type": "input_tokens", "value": 5000},
        "exclude_tools": ["web_search"],
    }]},
)
```
Compaction has its own beta header and request shape (on-demand vs at-a-threshold); follow the compaction docs rather than copying parameters from memory.

## Choosing / trade-offs
| Technique | Cost | Loses | Pick when |
|---|---|---|---|
| Truncate/offload tool results | none | detail unless re-read | always; first line of defence |
| Clear old tool results | none, but breaks cache at the cleared point | old raw outputs | many tool calls whose results are consumed once |
| Compaction (summarize) | one extra LLM call | nuance, exact wording | long sessions that must continue in one thread |
| External notes file | a few tool calls | nothing if notes are good | multi-session or multi-hour tasks |
| Sub-agents | more total tokens than one agent | the sub-agent's raw trail | broad, parallelizable search or exploration |
| Just-in-time retrieval | latency of extra tool calls | nothing | large corpora, repos, skill libraries |
| Pre-loading (stuff it all in) | tokens every turn | focus | small, fixed reference data that is always needed |

- **Caching vs editing**: every edit to earlier context invalidates the prompt cache from that point; batch edits (clear at least N tokens) so the saving beats the re-cache cost ([[prompt-caching-and-cost]]).
- **Retrieval vs long context**: retrieve when the corpus is large or changing; load it whole when it is small and every part matters ([[long-context]], [[rag-basics]]).
- **Compaction vs fresh session + notes**: compaction keeps continuity; a fresh session with a handoff note is more predictable and easier to debug.

## Gotchas
- Summaries silently drop the one constraint that mattered (a user's "never touch prod", an ID). Keep a pinned "constraints" block outside what gets compacted, and evaluate summaries ([[agent-evaluation]]).
- Clearing a tool result the model still needs causes repeated calls; exclude tools whose results are referenced later (e.g. the plan or spec file).
- Truncation without a pointer is data loss; always say where the full output lives and how to fetch a slice.
- Dozens of tools or MCP servers burn context before the first user turn; load tools on demand or expose fewer ([[model-context-protocol]]).
- Sub-agents that return raw transcripts defeat the purpose; specify the return format and length.
- Injected content in tool results survives compaction if the summarizer copies it as an instruction ([[prompt-injection]]).
- Token estimates by character count are rough; use the provider's token-counting endpoint for budgets that matter.

## Related
- [[prompt-engineering]] - writing the instructions; this topic manages everything around them.
- [[agent-memory]] - persistent tiers (episodic/semantic) that context engineering reads from and writes to.
- [[long-context]] - why bigger windows still degrade, and when to use them anyway.
- [[prompt-caching-and-cost]] - stable prefixes and the cost of editing context.
- [[agent-skills]] - progressive disclosure packaged as skill folders.
- [[multi-agent-systems]] - sub-agent isolation and orchestrator-worker patterns.
- [[coding-agents]] - where compaction, notes files and sub-agents are used daily.
- [[a2a-protocol]] - delegating to a remote agent returns only its condensed result.
- [[dspy-and-prompt-optimization]] - optimizing the instruction and demo part of the context automatically.
- [[tool-calling]] - concise tool outputs are the cheapest context saving.
- [[prompt-compression]] - token-level and sentence-level pruning of long context and tool outputs.

## References
- Anthropic, Effective context engineering for AI agents (2025-09-29): https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- Claude API context editing: https://platform.claude.com/docs/en/build-with-claude/context-editing
- Claude API compaction: https://platform.claude.com/docs/en/build-with-claude/compaction
- Agent Skills progressive disclosure: https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview
