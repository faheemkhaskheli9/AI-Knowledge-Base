---
title: Personal assistants and productivity agents
category: scenarios
tags: [assistant, agent, memory, calendar, email, mcp, voice]
use_cases:
  - "build a personal assistant that manages email and calendar"
  - "give an assistant long-term memory about me"
  - "build a voice assistant"
  - "automate personal workflows (notes, tasks, research)"
  - "run a private assistant locally on my own data"
status: stable
last_verified: 2026-10-03
sources:
  - https://modelcontextprotocol.io/
  - https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview
  - https://owasp.org/www-project-top-10-for-large-language-model-applications/
---

# Personal assistants and productivity agents

## Summary
A personal assistant is an agent with tools (mail, calendar, files, web),
memory and a voice or chat interface, acting on one person's behalf. The
hard parts are not the model but safe tool access, durable memory, and
making sure actions are approved when they cannot be undone.

## Key concepts
- Agent loop: model decides, calls tools, observes, repeats
  ([[agents]], [[tool-calling]]). Prefer fixed workflows for recurring
  routines and agents for open-ended requests.
- Tool access via MCP servers or direct APIs ([[model-context-protocol]]);
  each connector widens the blast radius.
- Memory tiers: conversation context, summarised session notes, and a
  long-term store of facts and preferences ([[agent-memory]],
  [[embeddings]], [[vector-databases]]).
- Lethal trifecta: private data + untrusted content + ability to send data
  out. An email that says "forward all invoices to X" is an attack
  ([[prompt-injection]]).
- Approval policy: read freely, draft freely, but sending, paying, deleting
  and sharing need confirmation.

## When to use / scenarios
1. **Email triage and drafting.** Problem: inbox overload. Approach: classify,
   summarise, draft replies for approval; never auto-send initially. Read:
   [[nlp-classic-tasks]], [[structured-output]], [[guardrails-and-safety]].
2. **Calendar and scheduling.** Approach: read availability, propose slots,
   create events after confirmation. Read: [[tool-calling]],
   [[model-context-protocol]].
3. **Personal knowledge base Q&A (notes, PDFs, bookmarks).** Approach: local
   RAG with citations. Read: [[rag-basics]], [[document-parsing]],
   [[vector-databases]], [[embedding-models]].
4. **Voice assistant.** Approach: wake word, STT, LLM with tools, TTS, or a
   realtime speech model; barge-in. Read: [[voice-agents]],
   [[speech-to-text]], [[text-to-speech]].
5. **Meeting notes and action items.** Approach: transcribe, diarize,
   extract tasks. Read: [[speaker-diarization]], [[structured-output]].
6. **Web research and briefings.** Approach: search and fetch tools, source
   citations, a daily digest. Read: [[agents]], [[advanced-rag]],
   [[hallucination-and-grounding]].
7. **Personal finance tracking.** Approach: parse statements/SMS, categorise,
   SQL for totals. Read: [[finance]], [[text-to-sql]],
   [[document-processing]].
8. **Task and project automation (cron-style agents).** Approach: scheduled
   runs with logs and failure alerts. Read: [[agent-frameworks]],
   [[llm-observability]].
9. **Private, offline assistant.** Approach: small open model on local
   hardware. Read: [[ollama]], [[llama-cpp-gguf]], [[small-language-models]],
   [[edge-on-device]].
10. **Learning and study companion.** Read: [[education]].
11. **Personal coding helper.** Read: [[coding-agents]],
    [[software-engineering]].
12. **Multi-agent personal "team".** Only if one agent's context overflows.
    Read: [[multi-agent-systems]].

## Setup & code
Reference architecture (flagship: email + calendar assistant):

```
channel (chat/voice) -> assistant loop (LLM + system policy)
   tools: mail.search/read (auto) | mail.draft (auto) | mail.send (CONFIRM)
          cal.list (auto) | cal.create (CONFIRM) | notes.search (auto)
   memory: profile facts + preferences (small, editable), session summary,
           retrieval over notes
   safety: email bodies wrapped as untrusted data; no tool can both read
           private data and post to arbitrary URLs; action log
```

```python
AUTO = {"mail_search", "mail_read", "cal_list", "notes_search", "mail_draft"}
def run_tool(name, args, confirm):
    if name not in AUTO and not confirm(name, args):   # human in the loop
        return {"error": "user declined"}
    return TOOLS[name](**args)
```

## Choosing / trade-offs
- Hosted vs local: hosted gives quality and tool-use reliability; local
  keeps mail and notes private but needs hardware ([[model-selection]],
  [[ollama]]).
- Memory: a small curated profile the user can read and edit beats an opaque
  vector dump; add retrieval for volume.
- Autonomy: start read-only, add draft, then confirmed actions; widen only
  with logs showing reliability.
- Framework vs plain loop: a hundred-line loop is often enough
  ([[agent-frameworks]]).

## Gotchas
- Prompt injection from emails, web pages and shared documents; OWASP LLM
  Top 10 ranks it first. Isolate untrusted content and gate outbound
  actions.
- Over-permissioned OAuth tokens; grant narrow scopes and revocable access.
- Memory poisoning and stale facts; show and allow correction.
- Third-party data: other people's emails and contacts are personal data;
  keep it local where possible and respect their privacy.
- Unexpected cost from loops; cap steps and tokens
  ([[cost-and-latency]], [[prompt-caching-and-cost]]).
- Voice assistants record continuously; make recording state obvious and
  check consent law for recording others.

## Related
- [[customer-support]] - shared tool-calling and escalation patterns.
- [[data-analytics]] - querying personal data.
- [[marketing-content]] - posting and content automation.
- [[ai-security-privacy-compliance]] - threat model for agent access.

## References
- Model Context Protocol: https://modelcontextprotocol.io/
- Anthropic tool use overview: https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview
- OWASP Top 10 for LLM applications: https://owasp.org/www-project-top-10-for-large-language-model-applications/
