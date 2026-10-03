---
title: Diffusion models
category: concepts
tags: [diffusion, text-to-image, diffusers, stable-diffusion, latent-diffusion, denoising]
use_cases:
  - "generate product images or marketing visuals from text prompts"
  - "fine-tune an image model on a brand style with LoRA"
  - "edit or inpaint images (remove background objects, change clothing colour) for an e-commerce catalogue"
  - "synthesise training images for a rare-defect detector in manufacturing"
  - "run text-to-image locally within a limited GPU memory budget"
status: draft
last_verified: 2026-10-03
sources:
  - https://arxiv.org/abs/2006.11239
  - https://arxiv.org/abs/2112.10752
  - https://arxiv.org/abs/2207.12598
  - https://huggingface.co/docs/diffusers
---

# Diffusion models

## Summary
Diffusion models generate data by learning to reverse a gradual noising process: start from random noise and iteratively denoise into an image (or audio, video, molecule). Latent diffusion runs this in a compressed latent space, making high-resolution text-to-image practical, and is the basis of Stable Diffusion-style and many video generators.

## Key concepts
- **Forward / reverse process**: training adds noise at random strengths; the network learns to predict the noise (or velocity / clean image). Generation runs the reverse for N steps.
- **Latent diffusion**: a VAE compresses images to a small latent grid; denoising happens there, then the VAE decodes.
- **Denoiser backbone**: a U-Net in older models, a diffusion transformer (DiT) in many newer ones.
- **Text conditioning**: a text encoder feeds cross-attention; **classifier-free guidance** (guidance scale) trades prompt adherence against diversity.
- **Schedulers/samplers**: DDIM, Euler, DPM-Solver etc.; fewer steps = faster. Distilled models reach decent quality in very few steps.
- **Conditioning add-ons**: ControlNet (edges, depth, pose), IP-Adapter (reference image), inpainting, img2img (strength parameter).
- **Customisation**: LoRA or DreamBooth fine-tuning of a style or subject with a small dataset.
- Flow-matching is a closely related formulation used by several recent open models.

## When to use / scenarios
- Image/video generation, editing, upscaling, synthetic data.
- Real-world: fashion retailer producing lifestyle backgrounds; game studio concept art; manufacturer generating synthetic defects for [[object-detection]] training.
- Not when: you need exact text rendering, precise logos, or legally certain provenance without checking licences (see [[model-licenses]]); use templates or design tools instead.

## Setup & code
`pip install diffusers transformers accelerate torch` and a CUDA GPU (CPU works but is very slow). Model IDs and licences change; pick one via [[image-generation-models]] and accept its terms on Hugging Face if gated.

```python
import torch
from diffusers import AutoPipelineForText2Image

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
# pipe.enable_model_cpu_offload()   # use instead of .to("cuda") on small-VRAM GPUs
image = pipe("studio photo of a leather backpack on a white background",
             num_inference_steps=30, guidance_scale=7.0,
             generator=torch.Generator("cuda").manual_seed(0)).images[0]
image.save("backpack.png")
```

## Choosing / trade-offs
- Steps vs speed: 20-30 steps is typical; distilled models cut latency sharply at some quality cost.
- Guidance scale: higher follows the prompt more but oversaturates; lower is more natural and varied.
- Hosted API (easy, per-image cost, content policies) vs local (control, fine-tuning, privacy, GPU needed).
- LoRA for style/subject vs ControlNet for structure vs full fine-tune (rarely needed).

## Gotchas
- fp16 on some VAEs yields black or NaN images; use the model's recommended VAE or higher precision.
- Reproducibility needs a fixed seed and the same scheduler and library version.
- Hands, in-image text and counting remain weak in many models.
- Training-data copyright, likeness and licence terms vary by model and jurisdiction.
- Add your own moderation for public-facing products; safety filters differ between local and hosted use.
- Memory: SDXL-class models need roughly 8-12 GB VRAM in fp16 without offloading (approximate).

## Related
- [[image-generation-models]] - which model families to pick.
- [[multimodal-models]] - understanding (not generating) images.
- [[fine-tuning-and-peft]] - LoRA concept shared with LLMs.
- [[marketing-content]] - typical commercial use.
- [[gpu-cuda-setup]] - getting the GPU stack working.

## References
- Ho et al., DDPM: https://arxiv.org/abs/2006.11239
- Rombach et al., Latent Diffusion: https://arxiv.org/abs/2112.10752
- Ho & Salimans, Classifier-free guidance: https://arxiv.org/abs/2207.12598
- Diffusers docs: https://huggingface.co/docs/diffusers
