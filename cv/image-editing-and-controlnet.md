---
title: Image editing, inpainting and ControlNet
category: cv
tags: [image-editing, inpainting, outpainting, controlnet, ip-adapter, instruction-editing, flux-kontext, qwen-image-edit, diffusers]
use_cases:
  - "remove or replace an object in a product photo using a mask"
  - "generate images that follow the pose, edges or depth of a reference photo"
  - "keep a brand style or product look consistent across generated images"
  - "edit a photo with a plain-language instruction like 'make it night time'"
  - "extend an image beyond its borders for a new aspect ratio"
status: draft
last_verified: 2026-10-15
sources:
  - https://huggingface.co/docs/diffusers/using-diffusers/inpaint
  - https://huggingface.co/docs/diffusers/using-diffusers/controlnet
  - https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter
  - https://huggingface.co/docs/diffusers/api/pipelines/flux
  - https://huggingface.co/docs/diffusers/api/pipelines/qwenimage
  - https://github.com/black-forest-labs/flux
  - https://github.com/QwenLM/Qwen-Image
  - https://arxiv.org/abs/2302.05543
  - https://arxiv.org/abs/2308.06721
  - https://arxiv.org/abs/2211.09800
---

# Image editing, inpainting and ControlNet

## Summary
Diffusion models can edit as well as generate. Inpainting regenerates only a masked region; outpainting extends the canvas. ControlNet conditions generation on a structure map (edges, depth, pose) so the output keeps a reference layout; IP-Adapter conditions on a reference image's content or style. Instruction-based editors (FLUX.1 Kontext, Qwen-Image-Edit, and the image APIs in [[image-generation-models]]) take an image plus a sentence and return the edited image, which is now the simplest route for most edits. Pick by how precisely you must control where and what changes.

## Key concepts
- Mask convention: white (1) = regenerate, black (0) = keep. `strength` (img2img/inpaint) sets how far from the original the region may move; 1.0 ignores the original pixels under the mask.
- Paste-back: inpainting models re-encode the whole image through the VAE, which subtly changes unmasked pixels; composite the result back onto the original with a feathered mask to keep untouched areas bit-identical.
- ControlNet: a trainable copy of the encoder fed a control image (Canny edges, depth, OpenPose skeleton, segmentation, scribble); `controlnet_conditioning_scale` trades prompt freedom for structure fidelity. Several ControlNets can be combined.
- IP-Adapter: small adapter (about 100MB per the diffusers docs) adding image-prompt cross-attention to a frozen base; `set_ip_adapter_scale` (1.0 full, around 0.5 balances text and image per the docs).
- Instruction editing: the model reads the source image as context and follows a text instruction; no mask needed, but less spatial control. Some (Qwen-Image-Edit-2509/2511) accept several reference images.
- Control maps and masks often come from other CV models: edges (Canny), depth estimators, pose models ([[face-and-pose]]), and SAM masks ([[segment-anything-sam2]]).

## When to use / scenarios
- E-commerce: replace backgrounds, remove props, generate lifestyle scenes while keeping the product pixels ([[ecommerce-retail]]). Mask the product, inpaint the rest, paste the product back.
- Marketing: keep a pose or layout fixed across variants (ControlNet), keep a style or mascot consistent (IP-Adapter or a LoRA), localise text in banners (instruction editors) - [[marketing-content]].
- Real estate/interiors: restyle rooms while keeping geometry (depth or edge ControlNet).
- Photo cleanup: object removal, outpainting to new aspect ratios.
- Not for: pixel-exact retouching or colour-managed print work (use an image editor), photorealistic edits of real people without consent and provenance controls ([[guardrails-and-safety]]), or document/ID image changes (fraud risk).

## Setup & code
Runnable core with numpy: build a feathered mask, derive an edge control map, and composite an "inpainted" result so pixels outside the mask are unchanged.
```python
import numpy as np

rng = np.random.default_rng(0)
H, W = 64, 96
original = rng.uniform(0, 1, (H, W, 3))
original[20:44, 30:60] = [0.9, 0.1, 0.1]           # the object we want to replace

def box_blur(a, r):
    k = 2 * r + 1
    p = np.pad(a, r, mode="edge")
    c = p.cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / k ** 2

mask = np.zeros((H, W))
mask[18:46, 28:62] = 1                              # white = regenerate (dilated a little past the object)
soft = np.clip(box_blur(mask, 3), 0, 1)             # feathered edge for blending

# Control map: gradient magnitude edges (Canny-like stand-in) of the original, as ControlNet input.
gray = original.mean(2)
gy, gx = np.gradient(gray)
edges = (np.hypot(gx, gy) > 0.25).astype(np.uint8) * 255

generated = rng.uniform(0, 1, (H, W, 3)) * 0.2 + 0.4   # stand-in for the model's full-frame output
result = soft[..., None] * generated + (1 - soft[..., None]) * original

outside = soft == 0
assert np.allclose(result[outside], original[outside])   # untouched pixels are bit-identical
assert not np.allclose(result[30, 45], original[30, 45]) # the masked region changed
print("edge pixels:", int((edges > 0).sum()), "feathered px:", int(((soft > 0) & (soft < 1)).sum()))
```

Inpainting with diffusers (`pip install diffusers transformers accelerate torch`), following the diffusers inpaint guide:
<!-- skip-check: downloads model weights, needs torch and a GPU -->
```python
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image

pipe = AutoPipelineForInpainting.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
image, mask = load_image("product.png"), load_image("mask.png")   # white = repaint
out = pipe(prompt="a marble countertop, soft daylight", image=image, mask_image=mask,
           strength=0.99).images[0]
out.save("inpainted.png")
```

ControlNet (Canny edges) plus IP-Adapter on SDXL, from the diffusers ControlNet and IP-Adapter guides:
<!-- skip-check: downloads model weights, needs torch and a GPU -->
```python
import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from diffusers.utils import load_image

controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, torch_dtype=torch.float16).to("cuda")
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
pipe.set_ip_adapter_scale(0.5)

edges = load_image("canny_edges.png")          # e.g. cv2.Canny output of the layout reference
style = load_image("brand_style.png")
img = pipe("a living room, scandinavian style", image=edges, ip_adapter_image=style,
           controlnet_conditioning_scale=0.5).images[0]
img.save("controlled.png")
```

Instruction-based editing with an open model (Qwen-Image-Edit, Apache 2.0), from the Qwen-Image README; large model, needs a big GPU or offloading:
<!-- skip-check: large model download, needs torch and a GPU -->
```python
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

pipe = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2511", torch_dtype=torch.bfloat16).to("cuda")
out = pipe(image=[Image.open("room.jpg").convert("RGB")], prompt="Make it night time with warm lamp light",
           true_cfg_scale=4.0, negative_prompt=" ", num_inference_steps=40,
           generator=torch.manual_seed(0)).images[0]
out.save("edited.png")
```
Hosted alternatives (OpenAI image edits with optional mask, Gemini image models) are listed in [[image-generation-models]].

## Choosing / trade-offs
| Approach | Control | Effort | Notes |
|---|---|---|---|
| Instruction editing (API or Qwen-Image-Edit / FLUX.1 Kontext) | medium: "what", not exact "where" | lowest | best default for one-off edits and text changes |
| Inpainting with a mask | high on "where" | need a mask (SAM, brush, detector) | product-safe with paste-back |
| ControlNet | high on structure (pose, edges, depth) | need a control map | many checkpoints per base model; base-model specific |
| IP-Adapter | style/subject from an image | low | weaker identity than a LoRA; stack with ControlNet |
| LoRA fine-tune ([[fine-tuning-and-peft]]) | strongest style/subject consistency | needs training images | per base model |

- Licences differ per checkpoint: per the black-forest-labs/flux README, FLUX.1 Fill, Canny, Depth, Redux and Kontext [dev] use the FLUX.1-dev Non-Commercial License (commercial use needs a BFL licence); Qwen-Image (including the edit models) is Apache 2.0 per its repo; the ControlNet and IP-Adapter code repos are Apache 2.0, while each base model and adapter checkpoint carries its own terms ([[model-licenses]]).
- Hosted API vs self-hosted: APIs need no GPU and handle instruction edits well; self-host for privacy, volume, LoRAs and exact pipelines.
- ControlNets, IP-Adapters and LoRAs are tied to one base model family (SD1.5, SDXL, FLUX); switching base means re-sourcing all of them.

## Gotchas
- Visible seams: feather and dilate the mask a few pixels; mismatched lighting needs a slightly larger mask so the model can blend.
- Unmasked pixels drift through the VAE round trip; always paste back the original outside the mask when fidelity matters (product, logo, text).
- Mask polarity and resolution mismatches (mask not the same size as the image) silently produce wrong edits.
- Too-high `controlnet_conditioning_scale` copies artefacts of the control map; too low ignores it. Sweep a few values.
- Instruction editors may change things you did not ask for (faces, text, colour); compare before/after and constrain with a mask where supported.
- Kontext-family pipelines document an integrity/safety checker to run after generation; build moderation and provenance (C2PA/watermarks) into any public-facing editor.
- GPU memory: FLUX and Qwen-Image editors are large; use CPU offload or quantization ([[quantization]]) and expect slower runs.

## Related
- [[diffusion-models]] - how denoising, guidance and latent diffusion work.
- [[image-generation-models]] - which generation/editing models and APIs exist.
- [[segment-anything-sam2]] - masks for inpainting from clicks or text.
- [[open-vocabulary-detection]] - find the object to mask from a text prompt.
- [[fine-tuning-and-peft]] - LoRA for consistent styles and subjects.
- [[face-and-pose]] - pose maps for OpenPose ControlNets.
- [[guardrails-and-safety]] - moderation and misuse controls.
- [[marketing-content]] - typical commercial workflows.

## References
- Diffusers guides: inpainting https://huggingface.co/docs/diffusers/using-diffusers/inpaint, ControlNet https://huggingface.co/docs/diffusers/using-diffusers/controlnet, IP-Adapter https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter
- Diffusers FLUX (Fill, Control, Kontext) and Qwen-Image pipelines: https://huggingface.co/docs/diffusers/api/pipelines/flux, https://huggingface.co/docs/diffusers/api/pipelines/qwenimage
- FLUX.1 model licences table: https://github.com/black-forest-labs/flux
- Qwen-Image and Qwen-Image-Edit: https://github.com/QwenLM/Qwen-Image
- Zhang et al., ControlNet: https://arxiv.org/abs/2302.05543
- Ye et al., IP-Adapter: https://arxiv.org/abs/2308.06721
- Brooks et al., InstructPix2Pix: https://arxiv.org/abs/2211.09800
