---
title: Agent memory (working, episodic, semantic)
category: llm-apps
tags: [agent-memory, working-memory, episodic-memory, semantic-memory, context-management, personalization]
use_cases:
  - "give a personal assistant memory of user preferences across sessions"
  - "stop a long-running coding or research agent from overflowing its context window"
  - "keep an auditable record of what a healthcare or finance agent decided and why"
  - "decide what belongs in a vector store versus a database for agent state"
  - "forget or expire stored user data to meet retention rules"
status: draft
last_verified: 2026-10-03
sources:
  - https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool
  - https://arxiv.org/abs/2304.03442
  - https://langchain-ai.github.io/langgraph/concepts/memory/
---

# Agent memory (working, episodic, semantic)

## Summary
LLMs are stateless; "memory" is state your application stores and re-injects. Split it into tiers by lifetime and query pattern: working (this session), episodic (specific past events, exact lookup), semantic (distilled facts and preferences, similarity lookup). Use only the tiers the agent needs, and treat memory as a privacy and security surface.

## Key concepts
- **Working memory**: the current context window plus session state (messages, scratchpad, plan). Managed by trimming, summarizing, and offloading large tool outputs to files. Hot cache (in-process or Redis) for session state.
- **Episodic memory**: time-ordered records of what happened (conversations, tool calls, decisions). Query by id, user, date, exact keyword - a document/relational DB, not only vectors. Keep source records for audit; do not rely on the summary as the record of truth.
- **Semantic memory**: long-term facts and patterns ("prefers aisle seats", "uses metric units") extracted from episodes, stored with embeddings for similarity search ([[vector-databases]]).
- **Procedural memory**: learned instructions/skills, often stored as files or prompts (CLAUDE.md-style notes, [[coding-agents]]).
- **Write path**: decide what to store (extraction by LLM at end of session or via a `remember` tool), dedupe, resolve contradictions (newer wins; keep history), score importance.
- **Read path**: retrieve top-k relevant memories per turn, inject into the prompt within a token budget, measure whether they help.
- **Lifecycle**: importance scoring, decay, consolidation of episodes into semantic facts, retention limits, deletion on request.
- **Tools**: Anthropic's memory tool (client-side file directory the model reads/writes), LangGraph stores/checkpointers, Mem0/Letta-style libraries, or your own DB.

## When to use / scenarios
- Personal assistants and support agents that should remember a user across sessions.
- Long tasks (coding, research) that exceed one context window: notes files and summaries.
- Regulated agents: episodic log for audit and replay.
- Not needed: stateless single-shot tools. Start with working memory; add episodic when you need "what happened on date X"; add semantic only when cross-session personalization justifies embedding infrastructure.

| Agent type | Working | Episodic | Semantic |
|---|---|---|---|
| Single-session transcriber | yes | optional | no |
| Record-lookup / history review | yes | yes | optional |
| Advisory / personalization | yes | yes | yes |

## Setup & code
Minimal two-tier memory: episodic in SQLite (exact) + a `remember`/`recall` tool pair for durable facts (swap the fact store for a vector DB when facts grow):
```python
# pip install anthropic   (sqlite3 is stdlib)
import sqlite3, time, anthropic
db = sqlite3.connect("memory.db")
db.execute("CREATE TABLE IF NOT EXISTS facts(user TEXT, fact TEXT, ts REAL)")
db.execute("CREATE TABLE IF NOT EXISTS episodes(user TEXT, role TEXT, text TEXT, ts REAL)")

def remember(user, fact):
    db.execute("INSERT INTO facts VALUES (?,?,?)", (user, fact, time.time())); db.commit()
def recall(user, limit=10):
    return [r[0] for r in db.execute("SELECT fact FROM facts WHERE user=? ORDER BY ts DESC LIMIT ?", (user, limit))]

client = anthropic.Anthropic()
def chat(user, text, history):
    facts = "\n".join(f"- {f}" for f in recall(user))
    system = f"You are a helpful assistant.\nKnown facts about the user (may be outdated):\n{facts}"
    history.append({"role": "user", "content": text})
    r = client.messages.create(model="claude-sonnet-5-5", max_tokens=500, system=system, messages=history[-20:])  # trim working memory
    out = r.content[0].text
    history.append({"role": "assistant", "content": out})
    db.execute("INSERT INTO episodes VALUES (?,?,?,?)", (user, "user", text, time.time())); db.commit()
    return out
```
Extract facts after the session with a structured-output call ([[structured-output]]) and call `remember` on each, instead of trusting the main loop to decide.

## Choosing / trade-offs
- **Summarize vs retrieve vs store full**: summaries are cheap but lossy; retrieval is precise but needs good embeddings; full logs are complete but costly to inject.
- **LLM-managed memory (model decides what to save) vs deterministic pipeline**: model-managed is flexible but inconsistent; pipelines are predictable.
- **Vector vs relational**: vectors for fuzzy "relevant to this", SQL for exact/time/audit. Most real agents need both ([[rag-basics]]).
- **Per-user vs shared memory**: shared team state needs consistency and access control; isolate tenants.
- **Context compaction vs fresh session with summary** for long runs ([[agents]]).

## Gotchas
- Memory poisoning: injected text can be stored and replayed later ([[prompt-injection]]); do not store raw untrusted content as instructions.
- Stale or contradicted facts: store timestamps and prefer newest; allow user edit/delete.
- PII retention and right-to-erasure: have a delete path and retention policy ([[ai-security-privacy-compliance]]).
- Irrelevant memories injected every turn waste tokens and distract; gate by relevance score and budget.
- Cross-user leakage from missing tenant filters.
- Embeddings of memories must be regenerated if the embedding model changes ([[embedding-models]]).

## Related
- [[agents]] - where memory is consumed.
- [[vector-databases]] - semantic tier storage.
- [[rag-basics]] - same retrieval mechanics.
- [[multi-agent-systems]] - shared state between agents.
- [[long-context]] - alternative to retrieval for small histories.
- [[prompt-caching-and-cost]] - cache the stable memory prefix.

## References
- Anthropic memory tool: https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool
- Park et al., Generative Agents: https://arxiv.org/abs/2304.03442
- LangGraph memory concepts: https://langchain-ai.github.io/langgraph/concepts/memory/
