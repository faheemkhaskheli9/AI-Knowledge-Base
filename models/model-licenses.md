---
title: Model licenses and commercial use
category: models
tags: [license, apache-2, mit, community-license, open-weights, commercial-use, compliance]
use_cases:
  - "check whether an open-weight model can be used in a commercial product"
  - "choose a permissively licensed model for software we ship to customers"
  - "review licensing risk before fine-tuning and redistributing a model"
  - "decide if generated images from an open image model can be sold"
status: draft
last_verified: 2026-10-03
sources:
  - https://huggingface.co/Qwen/Qwen3.8-27B
  - https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
  - https://huggingface.co/google/gemma-4-31B-it
  - https://huggingface.co/black-forest-labs/FLUX.2-dev
  - https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
  - https://docs.mistral.ai/getting-started/models/models_overview/
---

# Model licenses and commercial use

## Summary
"Open weights" does not mean "open source". Licenses range from permissive
(Apache 2.0, MIT) to custom community or non-commercial terms. Always read the
license on the exact checkpoint you use; the same vendor mixes licenses. This
is guidance, not legal advice.

## Key concepts
- Permissive (Apache 2.0, MIT): commercial use, modification, redistribution;
  Apache adds an explicit patent grant; both require notices.
- Custom community licenses (Llama-style): free for most users but add
  conditions such as attribution/naming, an acceptable-use policy and thresholds
  for very large companies. I could not load the Llama text, so read it directly.
- Non-commercial licenses (FLUX.2 [dev]): research/personal; commercial
  deployment needs a separate agreement.
- API terms are separate: closed models (Claude, GPT, Gemini) are governed by
  provider terms and usage policies, not a weights license.
- Weights license is distinct from training-data and output ownership issues.

Verified examples (2026-10):

| Model | License | Source |
|---|---|---|
| Qwen3.8-27B, Qwen3-Embedding/Reranker-8B | Apache 2.0 | model cards ([[qwen]]) |
| DeepSeek-V4.1-Flash | MIT | model card ([[deepseek]]) |
| Gemma 4 | Apache 2.0 | model card ([[small-language-models]]) |
| Mistral Large 3, Small 4, Ministral 3 | Apache 2.0 | Mistral docs ([[mistral]]) |
| Mistral Medium 3.5 | "Modified MIT" | Mistral docs; read terms |
| Mistral Premier models (OCR, Codestral, Embed) | API-only | Mistral docs |
| FLUX.2 [klein] 4B | Apache 2.0 | card ([[image-generation-models]]) |
| FLUX.2 [dev] | FLUX Non-Commercial License | card |
| Llama (Meta) | Llama Community License (custom) | [[meta-llama]]; terms unverified |
| Phi-4 family | MIT (secondary source) | verify on cards |

Licenses can differ between repos of one family; Qwen's org page does not state
licenses, so open each card.

## When to use / scenarios
- Shipping a model inside a product or on-prem for customers: prefer Apache/MIT.
- SaaS using hosted open models: check acceptable-use and size thresholds.
- Fine-tune and publish derivatives: confirm redistribution and naming rules.
- Image products: confirm the checkpoint allows commercial use of outputs.

## Setup & code
Record the license programmatically before adopting a model:
```python
from huggingface_hub import model_info

info = model_info("Qwen/Qwen3.8-27B")
print(info.card_data.license if info.card_data else "no card data")
```
```bash
pip install huggingface_hub
```
Keep a MODEL_LICENSES.md in your repo with name, revision, license, link, date.

## Choosing / trade-offs
- Apache/MIT: least friction, slightly narrower model choice at the frontier.
- Community licenses: often strong models; accept ongoing compliance work.
- Closed API: no weights license issue, but vendor terms and data handling apply.

## Gotchas
- Gated repos require accepting terms; that acceptance is a contract.
- A permissive model fine-tuned on restricted data inherits those data terms.
- Derivative or quantized copies on HF may carry the original's restrictions.
- Licenses can change for newer versions of the same family.
- Regulatory duties (EU AI Act, privacy law) sit on top of the license
  ([[ai-security-privacy-compliance]]).

## Related
- [[model-selection]] - where license fits with cost and quality.
- [[meta-llama]], [[qwen]], [[mistral]], [[deepseek]] - per-family notes.

## References
- https://huggingface.co/Qwen/Qwen3.8-27B
- https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
- https://huggingface.co/black-forest-labs/FLUX.2-dev
