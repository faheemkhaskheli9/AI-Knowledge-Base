---
title: Model selection guide
category: models
tags: [model-selection, open-weights, api, cost, latency, privacy, evaluation]
use_cases:
  - "decide between a closed API and a self-hosted open-weight model"
  - "choose the cheapest model that passes our quality bar for ticket classification"
  - "pick a model for a privacy-sensitive hospital or bank deployment"
  - "plan model routing: small model by default, large model for hard cases"
status: draft
last_verified: 2026-10-03
sources:
  - https://platform.claude.com/docs/en/about-claude/models/overview
  - https://developers.openai.com/api/docs/models
  - https://ai.google.dev/gemini-api/docs/pricing
  - https://api-docs.deepseek.com/quick_start/pricing
---

# Model selection guide

## Summary
Choose a model by running your own eval on candidates from each tier, not by
leaderboard rank. The main axes are quality, cost, latency, privacy/control and
license. Names and prices change monthly; the decision procedure does not.

## Key concepts
- Every major provider offers tiers: largest (slow, costly), balanced, fastest/
  cheapest. List-price spread in 2026-10 is large: output per 1M tokens ranges
  from about $0.5 (OpenAI efficient tier) through $10 (balanced tiers of
  Anthropic/OpenAI) to $50 (top tiers), per each provider's pricing page;
  DeepSeek Flash is about $0.6-1.2.
- Closed API: best frontier quality, zero ops, data goes to a vendor.
- Open weights: control, privacy, fine-tuning, fixed hardware cost; quality
  gap to frontier shrinks but exists on hard agentic work.
- Evals decide: see [[llm-evaluation]].

## When to use / scenarios
Rules of thumb (verify with an eval):
- Prototype: balanced tier of any API ([[anthropic-claude]], [[openai-gpt]],
  [[google-gemini]]).
- Hard reasoning/agentic coding: largest tier, then try to downshift.
- High-volume extraction/classification: fastest tier, or tuned small/open model
  ([[small-language-models]], [[deepseek]], [[qwen]]).
- Long documents: models with 1M context (Claude, GPT-6, Gemini, DeepSeek list
  1M or 1.05M); still test recall ([[long-context]]).
- Multimodal video/audio: Gemini ([[google-gemini]]).
- Private data / air-gapped: open weights on your hardware ([[meta-llama]],
  [[mistral]], [[qwen]]); or cloud-private endpoints (Bedrock/Vertex).
- EU residency preference: [[mistral]] or EU cloud regions.
- RAG: pair an LLM with [[embedding-models]] and [[rerankers]].
- Images: [[image-generation-models]].

## Setup & code
Minimal routing: try cheap model first, escalate on low confidence.
```python
from openai import OpenAI

cheap = OpenAI(base_url="https://api.deepseek.com", api_key="...")
def answer(q: str) -> str:
    r = cheap.chat.completions.create(model="deepseek-flash", messages=[{"role": "user", "content": q}])
    text = r.choices[0].message.content
    return text if "UNSURE" not in text else escalate(q)  # escalate() calls a larger model
```
Build a 50-200 item eval set first and score each candidate on it.

## Choosing / trade-offs
| Pressure | Pushes toward |
|---|---|
| Max quality | largest closed tier |
| Lowest cost at volume | fastest tier, DeepSeek Flash, tuned small model |
| Latency (voice, autocomplete) | small/fast tier, local model |
| Data must stay in-house | open weights self-hosted |
| License freedom | Apache/MIT ([[model-licenses]]) |
| No GPU team | API |
Break-even for self-hosting needs sustained high utilisation; below that APIs
are usually cheaper.

## Gotchas
- Leaderboards saturate and are gamed; your eval is the truth.
- Total cost includes retries, thinking tokens, long prompts and output length.
- Vendor lock-in: keep prompts and a provider-agnostic wrapper; many providers
  accept the OpenAI format.
- Models retire (for example Haiku 4.5 not before 2026-10-15); plan migrations.
- Jurisdiction and data-processing terms matter for API choice
  ([[ai-security-privacy-compliance]]).

## Related
- [[model-licenses]] - what you may do with open weights.
- [[cost-and-latency]] - optimisation.
- [[scenario-chooser]] - problem to approach mapping.
- [[prompt-caching-and-cost]] - lowering API spend.

## References
- https://platform.claude.com/docs/en/about-claude/models/choosing-a-model
- https://developers.openai.com/api/docs/models
- https://api-docs.deepseek.com/quick_start/pricing
