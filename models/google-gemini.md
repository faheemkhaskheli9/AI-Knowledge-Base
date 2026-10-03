---
title: Google Gemini models
category: models
tags: [gemini, google, llm, multimodal, embeddings, video, api]
use_cases:
  - "build a multimodal app that reads PDFs, images and video with the Gemini API"
  - "run cheap high-volume summarisation using a Flash-Lite tier"
  - "add video generation to a media-production tool"
  - "embed text, images and audio into one vector space for search"
status: draft
last_verified: 2026-10-03
sources:
  - https://ai.google.dev/gemini-api/docs/models
  - https://ai.google.dev/gemini-api/docs/pricing
---

# Google Gemini models

## Summary
Gemini is Google's closed multimodal model family, served by the Gemini API
(Google AI Studio) and Vertex AI. It spans fast "Flash" models, cheaper
"Flash-Lite" models and "Pro" models, plus specialised image, video, music, TTS
and embedding models.

## Key concepts
- Flash = speed/price workhorse; Flash-Lite = cheapest; Pro = deepest reasoning.
- Preview vs stable: Pro was still a `-preview` ID on 2026-10-03; preview IDs
  can change or be retired quickly.
- Native multimodality: text, image, audio, video input; separate models for
  generating images, video, music and speech.
- Open-weight sibling family: Gemma (see [[small-language-models]]).

Current IDs as of 2026-10 (source: models + pricing pages):

| Model | ID | Status | $/1M in / out |
|---|---|---|---|
| Gemini 3.8 Flash | `gemini-3.8-flash` | stable | 0.75 / 3.75 (to 2026-12-31) |
| Gemini 3.5 Flash-Lite | (listed; ID not verified) | - | 0.30 / 2.50 |
| Gemini 3.1 Pro | `gemini-3.1-pro-preview` | preview | 2.00 / 12.00 (<=200k ctx) |
| Nano Banana 2 (image) | `gemini-3.1-flash-image` | - | see pricing |
| Gemini Embedding 2 | (see docs) | - | text 0.20 |
| Veo 3.1 / Veo 3.1 Lite | video generation | - | see pricing |

The docs state prices rise on 2027-01-01 and that earlier Flash versions are
deprecated. I did not verify context-window sizes.

## When to use / scenarios
- Video/audio/PDF understanding in one API call.
- Cost-sensitive bulk jobs: Flash-Lite, plus Batch/Flex at 50% off.
- Media pipelines needing image + video generation under one vendor.
- Google Cloud shops wanting Vertex AI governance.
- Not for: strict on-prem; use Gemma or other open weights.

## Setup & code
```bash
pip install google-genai
export GEMINI_API_KEY=...
```
```python
from google import genai

client = genai.Client()
r = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="Summarise the key risks in three bullets: ...",
)
print(r.text)
```

## Choosing / trade-offs
- Flash for default; Pro when reasoning quality matters and preview risk is
  acceptable; Flash-Lite for volume.
- Long prompts (>200k) cost more on Pro.
- Free tier is generous but data may be used to improve products; use paid tier
  for sensitive data (check current terms).

## Gotchas
- Fast version churn: your pinned ID may be deprecated within months.
- Preview models carry no stability guarantees.
- Prices listed are promotional through 2026-12-31 for Flash.

## Related
- [[model-selection]] - cross-provider guide.
- [[multimodal-models]] - how multimodal models work.
- [[embedding-models]] - Gemini Embedding options.
- [[image-generation-models]] - Nano Banana and others.

## References
- https://ai.google.dev/gemini-api/docs/models
- https://ai.google.dev/gemini-api/docs/pricing
