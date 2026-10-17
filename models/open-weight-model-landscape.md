---
title: Open-weight model landscape (dated table across text, vision, audio, video)
category: models
tags: [open-weights, landscape, model-releases, licences, llm, vision, audio, video, image-generation, self-hosting]
use_cases:
  - "see which open-weight models are current before choosing one to self-host"
  - "find an open model with a commercial-friendly licence for text, vision, speech or video"
  - "refresh our internal list of approved open models each quarter"
  - "compare open-weight options across modalities for an on-prem AI platform"
status: draft
last_verified: 2026-10-15
sources:
  - https://huggingface.co/docs/huggingface_hub/guides/search
  - https://github.com/MoonshotAI/Kimi-K3
  - https://github.com/zai-org/GLM-5
  - https://github.com/MiniMax-AI/MiniMax-M3
  - https://github.com/openai/gpt-oss
  - https://github.com/QwenLM/Qwen3-Coder
  - https://github.com/ByteDance-Seed/Depth-Anything-3
  - https://github.com/openai/whisper
  - https://github.com/Wan-Video/Wan2.2
  - https://github.com/Lightricks/LTX-2
---

# Open-weight model landscape (dated table across text, vision, audio, video)

## Summary
A snapshot of notable open-weight models by modality, each row with the date it
was last checked and where the fact came from. Open releases move monthly, so
this file is a starting map, not a ranking: use it to shortlist, then read the
model card and licence, and run your own evals. The "How to refresh" section
describes how to update the table.

## Key concepts
- **Open weights is not open source**: licences range from Apache 2.0 / MIT to
  community licences with revenue thresholds and non-commercial terms
  ([[model-licenses]]).
- **Total vs active parameters**: MoE models need memory for all experts
  ([[mixture-of-experts]]).
- **"Checked" column**: the date someone last confirmed the row against the
  source in brackets. Rows from other files in this base inherit that file's
  `last_verified` date.
- Rows list what was verifiable; blank or "not verified" means leave it out of
  decisions until you read the card.

### Text and reasoning LLMs
| Model | Org | Size (total / active) | Licence | Checked (source) |
|---|---|---|---|---|
| Kimi K3 | Moonshot AI | 2.8T / ~104B, 1M ctx, + vision | Kimi K3 License (MIT-based; MaaS and attribution thresholds) | 2026-10-15 (GitHub README, LICENSE) |
| GLM-5.3 / GLM-5.3-Flash | Z.ai | 744B / 40B; 320B / 18B | custom (5.3) / MIT (Flash), per news reports | 2026-10-15 (GitHub README; licences from news) |
| MiniMax M3 | MiniMax | ~428B / ~23B, 1M ctx, + image/video | not verified | 2026-10-15 (GitHub README) |
| DeepSeek V4.1-Flash | DeepSeek | 552B MoE | MIT | 2026-10-03 ([[deepseek]]) |
| Qwen3.8-27B | Alibaba | 27B, + image/video | Apache 2.0 | 2026-10-03 ([[qwen]]) |
| Mistral Large 3, Small 4, Ministral 3 (3B/8B/14B) | Mistral AI | - | Apache 2.0 | 2026-10-03 ([[mistral]]) |
| Gemma 4 (E2B, E4B, 12B, 26B-A4B, 31B) | Google | up to 31B | Apache 2.0 | 2026-10-03 ([[small-language-models]]) |
| gpt-oss-120b / gpt-oss-20b | OpenAI | 117B / 5.1B; 21B / 3.6B | Apache 2.0 (2025 release) | 2026-10-15 (GitHub README) |
| Llama 4 Scout / Maverick | Meta | 17B active | Llama Community License | 2026-10-03 ([[meta-llama]]) |

### Code
| Model | Org | Size | Licence | Checked (source) |
|---|---|---|---|---|
| Qwen3-Coder-480B-A35B, 30B-A3B, Qwen3-Coder-Next | Alibaba | see name; Next on an 80B-A3B base | not stated in README | 2026-10-15 (GitHub README; [[code-models]]) |
| Devstral 2 / Devstral Small 2 | Mistral AI | - | modified MIT / Apache 2.0 | 2026-10-15 (Mistral announcement, via search) |

### Vision, image and 3D
| Model | Task | Licence | Checked (source) |
|---|---|---|---|
| Qwen3.8-27B, Kimi K3, MiniMax M3, Gemma 4 | vision-language (image in) | see text table | as above |
| FLUX.2 [klein] 4B / FLUX.2 [dev] 32B | text-to-image | Apache 2.0 / non-commercial | 2026-10-03 ([[image-generation-models]]) |
| Depth Anything 3 | depth, pose, 3D Gaussians | SMALL, BASE, METRIC, MONO Apache 2.0; larger CC BY-NC 4.0 | 2026-10-15 (GitHub; [[depth-estimation]]) |
| SAM 2 | promptable segmentation | permissive per repo | 2026-10-03 ([[segmentation]]) |

### Audio and speech
| Model | Task | Licence | Checked (source) |
|---|---|---|---|
| Whisper (tiny to large, turbo) | speech-to-text | MIT (code and weights) | 2026-10-15 (GitHub README) |
| Voxtral Mini Transcribe Realtime | speech-to-text | Apache 2.0 | 2026-10-03 ([[mistral]]) |
| Kokoro (82M) | text-to-speech | Apache 2.0 | 2026-10-03 ([[text-to-speech]]) |

### Video generation
| Model | Task | Size | Licence | Checked (source) |
|---|---|---|---|---|
| Wan2.2 T2V/I2V-A14B, TI2V-5B, S2V-14B, Animate-14B | text/image/speech to video, 480P-720P | A14B: 27B total, 14B active per step; TI2V 5B dense | Apache 2.0 | 2026-10-15 (GitHub README) |
| LTX-2.5 | joint audio + video from text/image | 22B transformer + Gemma-based text encoder | LTX-2.x Community License: free under US$10M annual revenue, paid licence above | 2026-10-15 (GitHub README, LICENSE-2_x) |

## When to use / scenarios
- Before choosing a self-hosted model for a new project: shortlist here, then
  go to the vendor file ([[qwen]], [[chinese-open-weight-models]], [[mistral]]).
- Platform teams maintaining an "approved models" list with licence status.
- Quarterly reviews: re-run the refresh steps and record what changed.
- Not for: picking the single best model; use [[model-selection]] and your own
  evals. Not a source for API-only models ([[anthropic-claude]],
  [[openai-gpt]], [[google-gemini]]).

## Setup & code
How to refresh the table:
1. For each org, list its newest repos on Hugging Face (script below) and its
   GitHub org; note new families, not every fine-tune or quant.
2. Open the model card and LICENSE file; copy size, context, modality and
   licence only from there. If a fact comes from news, say so in the row.
3. Update the row's "Checked" date and source; update the vendor file too.
4. Flag rows older than 90 days (runnable check below) and re-verify them first.

Runnable staleness check over the table rows:
```python
from datetime import date

rows = {  # model -> last checked date, copied from the tables above
    "Kimi K3": "2026-10-15", "DeepSeek V4.1-Flash": "2026-10-03",
    "Llama 4 Scout / Maverick": "2026-10-03", "Wan2.2": "2026-10-15",
}

def stale(rows: dict, today: date, max_age_days: int = 90) -> list[tuple[str, int]]:
    out = []
    for model, checked in rows.items():
        age = (today - date.fromisoformat(checked)).days
        if age > max_age_days:
            out.append((model, age))
    return sorted(out, key=lambda x: -x[1])

print("stale today:", stale(rows, date(2026, 10, 15)))
print("stale in Feb 2027:", stale(rows, date(2027, 2, 1)))
assert stale(rows, date(2026, 10, 15)) == []
```

Listing an org's most recently updated repos with `huggingface_hub`:
<!-- skip-check: needs network access to the Hugging Face Hub -->
```python
from huggingface_hub import HfApi

api = HfApi()
for org in ["moonshotai", "zai-org", "MiniMaxAI", "Qwen", "deepseek-ai", "mistralai", "google", "meta-llama"]:
    for m in api.list_models(author=org, sort="lastModified", direction=-1, limit=5):
        print(org, m.id, m.last_modified, m.pipeline_tag)
```

## Choosing / trade-offs
- **Licence vs capability**: the strongest open models increasingly carry
  custom licences (revenue thresholds, attribution, security reviews); Apache /
  MIT options are often a step behind but simpler to ship.
- **Size vs hardware**: trillion-parameter MoEs need multi-GPU nodes; 20-30B
  dense or small-active MoE models fit one large GPU when quantized
  ([[quantization]], [[gpu-cloud-options]]).
- **Generalist vs specialist**: one multimodal LLM can replace several models,
  but dedicated ASR, depth or video models are usually cheaper and better at
  their task.

## Gotchas
- Secondary sources (blogs, news, aggregators) often disagree on dates, sizes
  and licences; trust the card and LICENSE file.
- Licences change between point releases; re-check on upgrade.
- "Coming soon" weights and API-only "preview" models are not open releases.
- HF listing parameter counts may include vision encoders or MTP layers.
- Benchmarks on cards are vendor-run; see [[reading-ai-benchmarks]].

## Related
- [[model-selection]] - choosing between open and closed models.
- [[model-licenses]] - licence types behind the Licence column.
- [[chinese-open-weight-models]] - Kimi, GLM, MiniMax details.
- [[code-models]] - code-specialised models and benchmarks.
- [[depth-estimation]], [[gaussian-splatting-and-nerf]] - vision/3D rows.
- [[speech-to-text]], [[text-to-speech]] - audio rows.
- [[image-generation-models]] - image rows.
- [[huggingface-transformers]] - loading open models.

## References
- Hugging Face Hub search API: https://huggingface.co/docs/huggingface_hub/guides/search
- Kimi K3: https://github.com/MoonshotAI/Kimi-K3
- GLM-5 family: https://github.com/zai-org/GLM-5
- MiniMax M3: https://github.com/MiniMax-AI/MiniMax-M3
- gpt-oss: https://github.com/openai/gpt-oss
- Qwen3-Coder: https://github.com/QwenLM/Qwen3-Coder
- Depth Anything 3: https://github.com/ByteDance-Seed/Depth-Anything-3
- Whisper: https://github.com/openai/whisper
- Wan2.2: https://github.com/Wan-Video/Wan2.2
- LTX-2: https://github.com/Lightricks/LTX-2
