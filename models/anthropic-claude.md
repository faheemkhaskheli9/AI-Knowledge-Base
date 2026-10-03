---
title: Anthropic Claude models
category: models
tags: [claude, anthropic, llm, api, agents, coding, long-context]
use_cases:
  - "build a coding or agentic assistant on the Claude API"
  - "pick between Claude tiers for a support-ticket triage pipeline on a budget"
  - "run Claude through AWS Bedrock or Google Vertex for a regulated bank or hospital"
  - "analyse 500-page contracts in one prompt for a legal-tech product"
status: draft
last_verified: 2026-10-03
sources:
  - https://platform.claude.com/docs/en/about-claude/models/overview
  - https://platform.claude.com/docs/en/about-claude/pricing
  - https://platform.claude.com/docs/en/about-claude/model-deprecations
---

# Anthropic Claude models

## Summary
Claude is Anthropic's closed-weight LLM family, served via the Claude API and
through Amazon Bedrock, Google Cloud Vertex AI and Microsoft Foundry. It is
strong at agentic coding, long-horizon tool use and long documents. All
current models take text and image input and produce text.

## Key concepts
- Tiers: a largest/slowest tier (Fable), a high-end tier (Opus), a balanced
  tier (Sonnet) and a fastest/cheapest tier (Haiku).
- Model IDs are pinned snapshots; the docs say every ID, including dateless
  ones from the 4.6 generation on, is a pinned snapshot.
- Thinking: newer models use adaptive thinking steered by the `effort`
  parameter; the manual `thinking.type="enabled"` + `budget_tokens` mode is not
  accepted on later models (Haiku 4.5 still uses extended thinking).
- Batch API is 50% off; prompt-cache reads cost a fraction of input price.

Current IDs as of 2026-10 (source: models overview page):

| Tier | API ID | Context | Max output | $/MTok in / out |
|---|---|---|---|---|
| Fable 5.1 (largest) | `claude-fable-5-1` | 1M | 128K | 10 / 50 |
| Opus 5.5 | `claude-opus-5-5` | 1M | 128K | 4 / 20 |
| Sonnet 5.5 (balanced) | `claude-sonnet-5-5` | 1M | 128K | 2 / 10 |
| Haiku 4.5 (fastest) | `claude-haiku-4-5-20251001` (alias `claude-haiku-4-5`) | 200K | 64K | 1 / 5 |

Haiku 4.5 retirement: not sooner than 2026-10-15 (check the deprecations page).
Older Opus 4.x/5, Sonnet 4.6/5 and Fable 5 are listed as legacy but available.

## When to use / scenarios
- Agentic coding, multi-step tool use, computer-use style workflows.
- Long-document work (contracts, filings, codebases) using the 1M window.
- Enterprise teams that must stay inside AWS/GCP/Azure: same models via
  Bedrock/Vertex/Foundry (IDs differ, see overview page).
- High-volume classification/extraction: Haiku tier; escalate hard cases up.
- Not for: on-prem/air-gapped needs (no open weights) - see [[meta-llama]],
  [[qwen]], [[mistral]]; embeddings (Anthropic has no embedding model, see
  [[embedding-models]]).

## Setup & code
```bash
pip install anthropic
export ANTHROPIC_API_KEY=...
```
```python
import anthropic

client = anthropic.Anthropic()
msg = client.messages.create(
    model="claude-sonnet-5-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Summarise this invoice policy in 3 bullets: ..."}],
)
print(msg.content[0].text)
```

## Choosing / trade-offs
- Start on Opus 5.5 per Anthropic's guidance; move to Fable only if evals still
  fall short at higher effort; drop to Sonnet/Haiku when latency or cost
  dominate and evals hold.
- Higher `effort` and thinking raise quality and cost/latency.
- Use cache + batch for repeated prefixes and offline jobs (see
  [[prompt-caching-and-cost]]).

## Gotchas
- Pricing, IDs and retirement dates change; re-check the overview page.
- The newer tokenizer yields ~555k words per 1M tokens (fewer than older models).
- Bedrock/Vertex IDs and retirement dates differ from the first-party API.
- Max output above 128K needs the Batches API plus a beta header (per docs).
- Pin the ID in production; do not rely on aliases for pre-4.6 models.

## Related
- [[model-selection]] - choosing across providers.
- [[tool-calling]] - Claude tool use.
- [[prompt-caching-and-cost]] - cutting cost on repeated context.
- [[reasoning-models]] - adaptive/extended thinking concepts.
- [[long-context]] - using large windows well.

## References
- https://platform.claude.com/docs/en/about-claude/models/overview
- https://platform.claude.com/docs/en/about-claude/pricing
- https://platform.claude.com/docs/en/about-claude/model-deprecations
