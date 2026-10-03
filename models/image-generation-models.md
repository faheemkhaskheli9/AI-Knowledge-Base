---
title: Image generation models
category: models
tags: [image-generation, flux, stable-diffusion, gpt-image, nano-banana, diffusers]
use_cases:
  - "generate product mockups and marketing images from text prompts"
  - "self-host an open image model for an e-commerce catalogue tool"
  - "edit existing photos with masks via an API"
  - "fine-tune an image model with LoRA on a brand style"
status: draft
last_verified: 2026-10-03
sources:
  - https://developers.openai.com/api/docs/guides/image-generation
  - https://huggingface.co/black-forest-labs
  - https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
  - https://huggingface.co/black-forest-labs/FLUX.2-dev
  - https://huggingface.co/stabilityai
  - https://ai.google.dev/gemini-api/docs/models
---

# Image generation models

## Summary
Image generation comes as closed APIs (OpenAI, Google) and open-weight diffusion
models you run yourself (FLUX, Stable Diffusion). APIs are simplest and strongest
at text-in-image and editing; open models give control, LoRA fine-tuning and
fixed cost.

## Key concepts
- Text-to-image vs image editing (reference images, masks).
- Open models run through `diffusers`; see [[diffusion-models]] for theory.
- License differs per checkpoint, even within one vendor.

Current options as of 2026-10:

| Model | Access | Notes (source) |
|---|---|---|
| `gpt-image-2.5-sunburst` | OpenAI API | recommended for editing precision |
| `gpt-image-2.5-flare` | OpenAI API | faster, everyday use; sizes 1024x1024, 1536x1024, 1024x1536 |
| Nano Banana 2 (`gemini-3.1-flash-image`), Nano Banana 2 Lite (`gemini-3.1-flash-lite-image`) | Gemini API | |
| FLUX.2 [klein] 4B / 9B | open | 4B is Apache 2.0, about 13GB VRAM |
| FLUX.2 [dev] 32B | open, FLUX Non-Commercial License | commercial deployment needs separate terms |
| Stable Diffusion 3.5 Large/Medium/Turbo | open (Stability) | license not verified; read the card |
| Qwen-Image-2.1 | open | license not verified, see [[qwen]] |

## When to use / scenarios
- Marketing/creative assets, ad variations ([[marketing-content]]).
- E-commerce: background replacement, lifestyle shots ([[ecommerce-retail]]).
- Self-hosted for private or high-volume use.
- Not for: faithful photo editing of people without consent policies; add
  provenance/moderation ([[guardrails-and-safety]]).

## Setup & code
```bash
pip install openai
```
```python
import base64
from openai import OpenAI

r = OpenAI().images.generate(model="gpt-image-2.5-flare", prompt="Flat-lay of a leather wallet on linen, soft light")
open("out.png", "wb").write(base64.b64decode(r.data[0].b64_json))
```
Open model (needs recent diffusers from git, per the card):
```python
import torch
from diffusers import Flux2KleinPipeline

pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
pipe("A cat holding a sign that says hello world", height=1024, width=1024,
     guidance_scale=1.0, num_inference_steps=4).images[0].save("cat.png")
```
(The OpenAI base64 field is an assumption for GPT image models; check the guide.)

## Choosing / trade-offs
- API: best prompt adherence and text rendering, pay per image, no GPU.
- Open: LoRA brand styles, no per-image fee, need 13GB+ VRAM for klein-4B.
- Check license before commercial use ([[model-licenses]]).

## Gotchas
- Text rendering and multi-image consistency remain imperfect (OpenAI docs).
- FLUX [dev] is non-commercial by default; klein 4B is Apache 2.0.
- Model names churn; pin versions.

## Related
- [[diffusion-models]] - how they work.
- [[model-licenses]] - commercial caveats.
- [[multimodal-models]] - unified models that also output images.

## References
- https://developers.openai.com/api/docs/guides/image-generation
- https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
- https://huggingface.co/black-forest-labs/FLUX.2-dev
