---
title: Prompt caching and LLM cost control
category: llm-apps
tags: [prompt-caching, cost, latency, batch-api, token-budget, model-routing]
use_cases:
  - "cut the cost of a chatbot that resends a long system prompt and documents every turn"
  - "make a RAG or coding agent with large static context faster and cheaper"
  - "process 100k documents overnight at lower cost"
  - "estimate and cap monthly LLM spend for a SaaS feature"
  - "route easy requests to a small model and hard ones to a strong model"
status: draft
last_verified: 2026-10-03
sources:
  - https://platform.claude.com/docs/en/build-with-claude/prompt-caching
  - https://platform.claude.com/docs/en/build-with-claude/batch-processing
  - https://platform.openai.com/docs/guides/prompt-caching
---

# Prompt caching and LLM cost control

## Summary
LLM cost scales with tokens in and out. Prompt caching lets the provider reuse the processed static prefix of a prompt (system prompt, tools, documents, conversation history) so repeated calls cost far less and respond faster. Combine it with batching, model routing, output limits and context trimming to cut spend without hurting quality.

## Key concepts
- **Prefix caching (Anthropic)**: mark content with `cache_control: {"type": "ephemeral"}`; either top-level (automatic) or on specific blocks (up to 4 breakpoints). Order is tools -> system -> messages; everything up to the breakpoint must be byte-identical to hit.
- **TTL**: default 5 minutes (refreshed on hit); optional `"ttl": "1h"`.
- **Pricing multipliers** (verified in docs): 5-minute cache write 1.25x base input, 1-hour write 2x, cache read typically 0.1x of base input (some newer models list lower read multipliers; check the pricing page for your model). Minimum cacheable prompt length is model-specific (documented values range from 512 to 4,096 tokens); below it caching is silently skipped.
- **Usage fields**: `usage.cache_creation_input_tokens`, `usage.cache_read_input_tokens`, `usage.input_tokens` (uncached remainder), `usage.output_tokens`. Total input = sum of the first three.
- **OpenAI**: caching of long shared prefixes is automatic (no flag) with discounted cached-token pricing; keep the stable prefix first. See their guide for thresholds.
- **Batch APIs**: asynchronous batch processing at a discount (Anthropic Message Batches, OpenAI Batch) for non-interactive workloads.
- **Other levers**: smaller model for easy steps (routing/cascades), `max_tokens` and concise-output instructions, prune/summarize history, retrieval instead of stuffing, structured output to avoid verbose text, semantic response caching, count tokens before sending.

## When to use / scenarios
- Long static system prompt, tool definitions, few-shot examples, or reference documents reused across calls.
- Multi-turn chat and agent loops: the growing history is re-sent every step ([[agents]]).
- Contextual-retrieval ingestion: the full document cached while each chunk is contextualized ([[advanced-rag]]).
- Bulk offline jobs (classification, summarization, embeddings, evals): use a batch API ([[llm-evaluation]]).
- Not useful when prompts differ at the start every time, or traffic is so sparse the cache expires between calls.

## Setup & code
```python
# pip install anthropic
import anthropic
client = anthropic.Anthropic()
MODEL = "claude-sonnet-5-5"
BIG_DOC = open("handbook.txt").read()          # thousands of tokens, static

def ask(q: str):
    r = client.messages.create(
        model=MODEL, max_tokens=500,
        system=[{"type": "text", "text": "Answer from the handbook only."},
                {"type": "text", "text": BIG_DOC, "cache_control": {"type": "ephemeral"}}],  # breakpoint
        messages=[{"role": "user", "content": q}])
    u = r.usage
    print(f"uncached={u.input_tokens} written={u.cache_creation_input_tokens} read={u.cache_read_input_tokens}")
    return r.content[0].text

ask("What is the leave policy?")   # first call writes the cache
ask("How do I claim expenses?")    # within the TTL: cache read
```
Simple router (cheap model first, escalate on low confidence):
```python
def route(q):
    cheap = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=300,
                                   messages=[{"role": "user", "content": q}])
    return cheap.content[0].text   # escalate to a stronger MODEL if a validator/judge rejects it
```
Track cost per request from `usage` and log it per feature/tenant ([[llm-observability]], [[cost-and-latency]]).

## Choosing / trade-offs
- **5-minute vs 1-hour TTL**: 1h costs more to write but pays off when requests come in minutes apart, not seconds.
- **Cache vs shrink**: caching reduces cost of repeated tokens; trimming removes tokens entirely - do both.
- **Small model vs large**: route by task difficulty; verify the cheap path with evals before shipping.
- **Batch vs real-time**: batch is cheaper but results arrive later (hours at worst); never for interactive UX.
- **Long context vs RAG**: with caching, stuffing a modest corpus can beat building RAG ([[long-context]], [[rag-basics]]); for large or changing corpora RAG still wins.

## Gotchas
- Any change before the breakpoint (timestamps, user names, reordered tools, changed system text) invalidates the cache; put dynamic content after it.
- Silent miss below the minimum length: check `cache_read_input_tokens` rather than assuming.
- Concurrent first requests can all pay the write; warm the cache with one call first.
- Extended thinking or toggled features can change the cached prefix on some models; verify with usage fields.
- Output tokens usually cost more than input and are not cacheable: cap `max_tokens` and avoid rambling formats.
- Prices and multipliers change; re-check the pricing page before budgeting.
- Agent loops without step/token caps are the most common cost incident.

## Related
- [[cost-and-latency]] - deployment-level levers.
- [[agents]] - history growth in loops.
- [[advanced-rag]] - contextual retrieval depends on caching.
- [[llm-observability]] - per-request cost tracking.
- [[model-selection]] - routing between models.
- [[long-context]] - when to stuff vs retrieve.

## References
- Anthropic prompt caching: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- Anthropic batch processing: https://platform.claude.com/docs/en/build-with-claude/batch-processing
- OpenAI prompt caching: https://platform.openai.com/docs/guides/prompt-caching
