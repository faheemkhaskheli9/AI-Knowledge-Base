---
title: Video generation models
category: models
tags: [video-generation, text-to-video, image-to-video, wan, hunyuanvideo, ltx, veo, gemini-omni, sora, diffusers]
use_cases:
  - "generate short product or ad clips from a text prompt"
  - "animate a still product photo into a 5-second video"
  - "self-host an open video model so footage never leaves our servers"
  - "control camera motion or character pose in generated video"
  - "estimate the GPU needed to run an open video model locally"
status: draft
last_verified: 2026-10-15
sources:
  - https://github.com/Wan-Video/Wan2.2
  - https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers
  - https://huggingface.co/tencent/HunyuanVideo-1.5
  - https://github.com/Lightricks/LTX-2
  - https://ai.google.dev/gemini-api/docs/video
  - https://ai.google.dev/gemini-api/docs/deprecations
  - https://developers.openai.com/docs/deprecations
  - https://huggingface.co/docs/diffusers/main/en/api/pipelines/wan
---

# Video generation models

## Summary
Video generation models turn text, an image, or other video into short clips
(typically seconds long). Closed APIs (Google's Veo / Gemini Omni line) give
the best quality with no GPU; open-weight models (Alibaba's Wan, Tencent's
HunyuanVideo, Lightricks' LTX) run on your own GPUs, support LoRA fine-tuning
and control inputs, but need a lot of VRAM and minutes per clip. Licences vary
widely, and closed models have been retired at short notice.

## Key concepts
- **How they work**: almost all are latent diffusion (or flow-matching)
  transformers over video. A 3D VAE compresses frames in space and time, a DiT
  denoises the latent tokens, the VAE decodes back to pixels ([[diffusion-models]]).
  Cost grows with resolution x frames, and full attention is quadratic in that
  token count, so length and resolution are the main cost drivers.
- **Modes**: text-to-video (T2V), image-to-video (I2V, the image becomes the
  first frame), first+last frame interpolation, video-to-video/editing,
  extension of an existing clip, and joint audio+video (LTX-2, Veo 3.x).
- **Control methods**: reference images for identity/style; camera trajectory
  conditioning; pose/depth/edge control video (ControlNet-style); character
  animation and replacement (e.g. Wan-Animate); LoRA for a brand style or a
  subject; negative prompts and guidance scale.
- **Distilled variants** (few-step, fp8/fp4 quantized) trade some quality for
  large speed-ups; most families ship them or have community versions.

Options as of 2026-10 (verify licence and status on the linked page before use):

| Model | Access | Notes |
|---|---|---|
| Wan 2.2 (T2V-A14B, I2V-A14B, TI2V-5B, Animate-14B) | open weights (Alibaba) | reported as Apache 2.0; TI2V-5B targets 720p at 24 fps on a single consumer GPU such as an RTX 4090 (model card). Later Wan versions (2.5, 2.6) were API-only; open status of newer ones is unclear |
| HunyuanVideo 1.5 (Tencent) | open weights | about 8.3B parameters; Tencent Hunyuan Community License, which excludes the EU, UK and South Korea - read it before commercial use |
| LTX-2 family (Lightricks) | open weights | joint audio+video DiT; LTX-2 Community License (free commercial use with conditions; check revenue terms); distilled and fp8/fp4 checkpoints |
| Veo 3.1 / Fast / Lite (`veo-3.1-generate-preview`, `veo-3.1-fast-generate-preview`, `veo-3.1-lite-generate-preview`) | Gemini API (retired) | shut down on 2026-10-22; use `gemini-omni-1.1-flash` (Gemini deprecations page) |
| Gemini Omni 1.1 Flash (`gemini-omni-1.1-flash`) | Gemini API | Google's current video generation line; see [[google-gemini]] and the video docs for call shape |
| Sora 2 / Sora 2 Pro (OpenAI) | retired | OpenAI's deprecations page lists the Videos API and Sora models removed on 2026-09-24 with no replacement listed; do not start new work on it |

## When to use / scenarios
- Marketing: ad variants, social clips, product spins from a packshot (I2V)
  ([[marketing-content]], [[ecommerce-retail]]).
- Pre-visualisation and storyboards for film, games and architecture.
- Education/training explainers ([[education]]).
- Synthetic training data for vision models (with care about domain gap); for
  action-conditioned simulation see [[world-models]].
- Not for: long-form narrative with consistent characters across minutes (still
  weak), anything needing accurate text, physics or brand-safe faces without
  review, or footage of real people without consent. Talking-head avatars and
  dubbing are usually better served by dedicated lip-sync tools plus
  [[text-to-speech]].

## Setup & code
Why video is expensive - latent token count and attention cost grow fast with
length and resolution (numpy, runs offline; compression factors are typical
values, not a specific model's):
```python
import numpy as np

def latent_tokens(seconds, fps, height, width, t_down=4, s_down=8, patch=2):
    frames = int(seconds * fps)
    lat_t = 1 + (frames - 1) // t_down          # causal 3D VAE keeps frame 0 separate
    lat_h, lat_w = height // s_down // patch, width // s_down // patch
    return lat_t * lat_h * lat_w

rows = []
for secs in (2, 5, 10):
    for (h, w) in ((480, 832), (720, 1280)):
        n = latent_tokens(secs, 24, h, w)
        rows.append((secs, f"{w}x{h}", n, n ** 2))
base = rows[0][3]
for secs, res, n, attn in rows:
    print(f"{secs:>3}s {res:>9}: {n:>7,} tokens, attention cost x{attn / base:6.1f}")
costs = np.array([r[3] for r in rows], dtype=float)
assert np.all(np.diff(costs[[0, 2, 4]]) > 0)    # longer clip -> more attention work
```

Open model with diffusers (Wan 2.2 TI2V-5B; needs a CUDA GPU and a large
download; follow the model card for current arguments):
<!-- skip-check: needs GPU, torch and a multi-GB model download -->
```python
import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()                     # lower VRAM, slower

frames = pipe(
    prompt="Slow dolly shot of a ceramic mug on a wooden table, morning light, steam rising",
    negative_prompt="blurry, distorted, text, watermark",
    height=704, width=1280, num_frames=121,         # about 5 s at 24 fps
    guidance_scale=5.0, num_inference_steps=50,
).frames[0]
export_to_video(frames, "mug.mp4", fps=24)
```
Hosted APIs are asynchronous: you submit a job, poll an operation until it is
done, then download the file. Check the provider's video guide for the current
model ID and parameters ([[google-gemini]]).

## Choosing / trade-offs
- **API vs open weights**: APIs win on quality, audio, and zero GPU ops; open
  weights win on data control, LoRA/brand fine-tuning, control inputs and
  per-clip cost at volume. Closed models have been retired within months
  (Sora API, Veo previews), so wrap the provider behind your own interface.
- **Model size vs hardware**: small models (around 5B) target a single 24GB
  consumer GPU with offloading; 14B-class and mixture-of-experts models want
  data-centre GPUs or multi-GPU, or quantized/distilled checkpoints
  ([[gpu-cloud-options]], [[quantization]]).
- **Speed vs quality**: distilled few-step checkpoints and lower resolution
  drafts for iteration; full steps and upscalers for final renders.
- **Licence**: Apache 2.0 (Wan 2.2 as reported) is the least restrictive;
  community licences may restrict territories, revenue or fine-tune
  distribution ([[model-licenses]]).

## Gotchas
- Minutes per clip on a single GPU is normal; plan an async job queue, not a
  synchronous HTTP call.
- Prompt adherence drops with long, multi-event prompts; generate shots
  separately and edit them together.
- Hands, text, fast motion and object permanence still fail often; review
  every clip before publishing.
- Some licences exclude whole regions (HunyuanVideo's excludes the EU, UK and
  South Korea); a model that is fine for a demo may be unusable for your users.
- Generated video can be deepfake-adjacent: add consent checks, moderation and
  provenance (C2PA metadata, provider watermarks) ([[guardrails-and-safety]]).
- Preview model IDs churn fast; check the deprecations page before shipping.

## Related
- [[diffusion-models]] - the underlying generative method.
- [[image-generation-models]] - still images, often the first frame for I2V.
- [[world-models]] - interactive, action-conditioned video for agents and simulation.
- [[music-and-audio-generation-models]] - soundtrack and sound effects for clips.
- [[google-gemini]] - Veo / Gemini Omni IDs and deprecations.
- [[model-licenses]] - commercial terms of open weights.
- [[video-analytics]] - understanding video rather than generating it.
- [[gpu-cloud-options]] - renting GPUs for open models.

## References
- Wan 2.2 code and model cards: https://github.com/Wan-Video/Wan2.2, https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers
- HunyuanVideo 1.5: https://huggingface.co/tencent/HunyuanVideo-1.5
- LTX-2: https://github.com/Lightricks/LTX-2
- Gemini API video generation: https://ai.google.dev/gemini-api/docs/video
- Gemini API deprecations: https://ai.google.dev/gemini-api/docs/deprecations
- OpenAI deprecations (Sora / Videos API): https://developers.openai.com/docs/deprecations
- Diffusers Wan pipeline: https://huggingface.co/docs/diffusers/main/en/api/pipelines/wan
