---
title: Mistral models
category: models
tags: [mistral, open-weights, apache-2, ocr, voxtral, embeddings, europe]
use_cases:
  - "use a European-hosted LLM API for GDPR-sensitive workloads"
  - "self-host an Apache-2.0 multimodal model for internal tooling"
  - "run OCR on scanned invoices with a dedicated OCR model"
  - "deploy a small Ministral model on a laptop or edge box"
status: draft
last_verified: 2026-10-03
sources:
  - https://docs.mistral.ai/getting-started/models/models_overview/
---

# Mistral models

## Summary
Mistral AI (France) ships a mix of open-weight models (mostly Apache 2.0) and
"Premier" API-only models, via its La Plateforme API and cloud marketplaces.
The lineup covers general LLMs, small edge models, OCR, speech, code,
embeddings and moderation.

## Key concepts
- Open-weight (Apache 2.0): Mistral Large 3, Mistral Small 4, Ministral 3
  (3B/8B/14B), Voxtral Mini Transcribe Realtime, Shieldstral 1.0.
- Mistral Medium 3.5 is listed under a "Modified MIT" license; read its terms
  before commercial use (I could not retrieve the model card).
- Premier (API-only): OCR 4.1/OCR 3, Voxtral Mini Transcribe 2, Codestral,
  Mistral Embed, Codestral Embed, Mistral Moderation 2.
- The catalogue page also lists third-party Z.ai GLM 5.3 (`zai-glm-5-3`, open,
  1M context).

Current IDs as of 2026-10 (source: models overview page):

| Model | API ID | License |
|---|---|---|
| Mistral Medium 3.5 | `mistral-medium-2604` | Modified MIT |
| Mistral Small 4 | `mistral-small-2603` | Apache 2.0 |
| Mistral Large 3 | `mistral-large-2512` | Apache 2.0 |
| Ministral 3 14B / 8B / 3B | `ministral-3-14b-2512`, `-8b-`, `-3b-` | Apache 2.0 |

Pricing and context windows were not shown on the fetched page; check the
docs/pricing page.

## When to use / scenarios
- EU data-residency preference or vendor diversification away from US labs.
- Self-hosting with a clean Apache 2.0 license (Small 4, Large 3, Ministral).
- Document digitisation: dedicated OCR models ([[ocr]], [[document-parsing]]).
- Speech: Voxtral for transcription/TTS ([[speech-to-text]]).
- Not for: the absolute frontier on agentic coding; benchmark against
  [[anthropic-claude]] / [[openai-gpt]] first.

## Setup & code
```bash
pip install mistralai
export MISTRAL_API_KEY=...
```
```python
import os
from mistralai import Mistral

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
r = client.chat.complete(
    model="mistral-small-2603",
    messages=[{"role": "user", "content": "Extract the total from: 'Invoice total EUR 1,240.50'"}],
)
print(r.choices[0].message.content)
```
(If the SDK major version differs, check the docs; the 1.x client shown uses
`Mistral(...).chat.complete`.)

## Choosing / trade-offs
- Small 4 for cost-efficient general use; Large 3 for quality with open weights;
  Ministral for edge ([[small-language-models]]).
- Open weights give control; API gives zero ops and Premier-only features.

## Gotchas
- Dated IDs (`-2603`) are snapshots; aliases such as `-latest` may move.
- "Premier" models have a more restrictive commercial license than Apache ones.
- Self-hosting Large 3 needs serious GPU memory ([[inference-servers-vllm]]).

## Related
- [[model-licenses]] - Apache vs modified-MIT vs community licenses.
- [[model-selection]] - when to pick Mistral.
- [[small-language-models]] - Ministral for edge.

## References
- https://docs.mistral.ai/getting-started/models/models_overview/
