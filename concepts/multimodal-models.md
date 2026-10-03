---
title: Multimodal models
category: concepts
tags: [multimodal, vision-language, vlm, clip, image-input, audio, anthropic]
use_cases:
  - "extract fields from invoices, receipts or ID photos with a vision-capable LLM"
  - "answer questions about charts, screenshots or medical-device photos"
  - "build image search with text queries using CLIP-style embeddings"
  - "add voice input and spoken output to an assistant"
  - "choose between a hosted multimodal API and an open vision-language model for privacy-sensitive documents"
status: draft
last_verified: 2026-10-03
sources:
  - https://arxiv.org/abs/2103.00020
  - https://arxiv.org/abs/2304.08485
  - https://docs.claude.com/en/docs/build-with-claude/vision
  - https://huggingface.co/docs/transformers/main/en/tasks/image_text_to_text
---

# Multimodal models

## Summary
Multimodal models process more than text: images, documents, audio, and sometimes video, either as input, output or both. Most "vision LLMs" turn an image into tokens via a vision encoder and feed them into an ordinary language model, so you can prompt about pictures the same way you prompt about text.

## Key concepts
- **Vision encoder + projector + LLM** (LLaVA-style): a ViT encodes patches; a projector maps them into the LLM's embedding space; the LLM reads image tokens beside text tokens.
- **Contrastive dual encoders (CLIP)**: image and text encoders trained so matching pairs share a vector space; powers zero-shot classification and image search ([[embeddings]]).
- **Image tokens cost context**: images consume tokens roughly in proportion to resolution; providers resize or tile large images.
- **Native multimodal vs stitched**: some models are trained end to end on mixed data (can also output images or audio); others chain separate models (speech-to-text -> LLM -> text-to-speech).
- **Document understanding**: PDFs/screenshots can be passed as images; layout, tables and charts are read visually, but dense small text may need OCR or higher resolution.
- **Outputs**: most return text; image generation is typically a separate diffusion model ([[diffusion-models]]).

## When to use / scenarios
- Documents with layout, tables or handwriting where text extraction loses structure.
- Screenshots and UI understanding, chart Q&A, product photo tagging, accessibility captions.
- Real-world: insurer reads damage photos plus claim text; retailer auto-tags catalogue images; clinic assistant reads scanned forms (with consent and compliance controls).
- Not when: pure text is available (cheaper), exact OCR transcription is required at scale (use [[ocr]] tools and verify), or precise measurement/counting is critical (VLMs are weak there; use [[object-detection]]).

## Setup & code
`pip install anthropic`; set `ANTHROPIC_API_KEY`. Pick the model ID from the current models page.

```python
import base64, anthropic

client = anthropic.Anthropic()
with open("invoice.png", "rb") as f:
    data = base64.standard_b64encode(f.read()).decode()

resp = client.messages.create(
    model="<a vision-capable Claude model>",
    max_tokens=500,
    messages=[{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
        {"type": "text", "text": "Return vendor, date and total as JSON."},
    ]}],
)
print(resp.content[0].text)
```
Open-source route: Transformers `image-text-to-text` pipelines with an open VLM; see [[vision-language-models]] and [[huggingface-transformers]].

## Choosing / trade-offs
- Hosted API (best quality, easy, data leaves your boundary) vs open VLM (private, tunable, more engineering and weaker on hard tasks).
- Image resolution: higher helps small text but costs tokens and latency.
- One multimodal call vs a pipeline (OCR + LLM): the pipeline is cheaper and auditable; the single call handles layout better.
- Voice: native audio models give low latency and prosody; STT+LLM+TTS is more controllable ([[voice-agents]]).

## Gotchas
- Models hallucinate details in unreadable regions instead of saying they are unreadable; ask for "unreadable" markers.
- Counting objects, precise spatial relations and small-text reading are unreliable.
- Images can carry prompt injections (text inside a picture) ([[prompt-injection]]).
- Medical, ID and face images raise privacy and regulatory issues; check [[ai-security-privacy-compliance]].
- Per-request image limits (size, count, formats) differ by provider; check the docs.
- Validate extracted numbers (totals, dates) with deterministic code.

## Related
- [[vision-language-models]] - CV-side detail on VLM tasks.
- [[embeddings]] - CLIP-style joint embeddings.
- [[document-parsing]] - when to parse before prompting.
- [[ocr]] - dedicated text extraction.
- [[speech-to-text]] - audio input path.
- [[diffusion-models]] - generating images.

## References
- Radford et al., CLIP: https://arxiv.org/abs/2103.00020
- Liu et al., LLaVA: https://arxiv.org/abs/2304.08485
- Claude vision docs: https://docs.claude.com/en/docs/build-with-claude/vision
- Transformers image-text-to-text: https://huggingface.co/docs/transformers/main/en/tasks/image_text_to_text
