---
title: OpenAI GPT models
category: models
tags: [openai, gpt, llm, api, image-generation, realtime]
use_cases:
  - "build a chatbot or agent on the OpenAI API"
  - "route high-volume classification to the cheapest OpenAI tier"
  - "add image generation and editing to a marketing tool"
  - "add a voice/realtime assistant to a call-centre product"
status: draft
last_verified: 2026-10-03
sources:
  - https://developers.openai.com/api/docs/models
  - https://developers.openai.com/api/docs/guides/image-generation
  - https://developers.openai.com/api/docs/guides/embeddings
---

# OpenAI GPT models

## Summary
OpenAI serves closed-weight GPT models through its API (Responses and Chat
Completions style endpoints), plus image, realtime/audio, embedding and
transcription models. The lineup is tiered by capability and price.

## Key concepts
- Tiers (flagship / near-flagship / efficient) with 1.05M-token context on the
  GPT-6 family per the models page.
- Specialized offerings exist (cybersecurity, life-sciences for approved
  organisations, realtime/audio, image generation).
- The same `openai` SDK works with many OpenAI-compatible providers (DeepSeek,
  vLLM, Ollama) by changing `base_url`.

Current IDs as of 2026-10 (source: developers.openai.com models page):

| Tier | ID | Context | $/MTok in / out |
|---|---|---|---|
| Most capable | `gpt-6-astra` | 1.05M | 10 / 50 |
| Near-top, cheaper | `gpt-6.1-sol` | 1.05M | 2 / 10 |
| Most efficient | `gpt-6-luna` | 1.05M | 0.1 / 0.5 |
| Image | `gpt-image-2.5-sunburst`, `gpt-image-2.5-flare` | - | see pricing page |
| Embeddings | `text-embedding-3-small` / `-large` | 8,192 input | see pricing page |

The models page did not show a dedicated open-weight section; I could not
verify current OpenAI open-weight model names.

## When to use / scenarios
- General assistants, structured-output extraction, tool-using agents.
- High-volume cheap tasks (tagging, routing, moderation-style) on the
  efficient tier.
- Image generation/editing inside products (see [[image-generation-models]]).
- Voice agents via realtime models (see [[voice-agents]]).
- Not for: data that must never leave your network - use open weights
  ([[meta-llama]], [[qwen]], [[deepseek]]).

## Setup & code
```bash
pip install openai
export OPENAI_API_KEY=...
```
```python
from openai import OpenAI

client = OpenAI()
r = client.responses.create(model="gpt-6.1-sol", input="Classify: 'card declined twice' -> billing|tech|other")
print(r.output_text)
```

## Choosing / trade-offs
- Sol-class for most production work; Astra only where evals show a gap; Luna
  for bulk. Evaluate on your own data ([[llm-evaluation]]).
- Reasoning effort and long outputs raise cost and latency.

## Gotchas
- Model names change fast and old ones are deprecated; pin and monitor
  deprecation notices.
- Prices above are list prices from the models page on 2026-10-03; cached-input
  and batch discounts differ.
- Image models struggle with precise text placement per the docs.

## Related
- [[model-selection]] - cross-provider decision guide.
- [[structured-output]] - JSON schema outputs.
- [[embedding-models]] - OpenAI embeddings vs alternatives.
- [[cost-and-latency]] - managing spend.

## References
- https://developers.openai.com/api/docs/models
- https://developers.openai.com/api/docs/guides/image-generation
- https://developers.openai.com/api/docs/guides/embeddings
