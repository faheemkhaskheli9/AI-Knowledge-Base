---
title: Vision-language models (VLMs)
category: cv
tags: [vlm, multimodal, clip, image-understanding, captioning, vqa, document-understanding]
use_cases:
  - "answer questions about images or screenshots with an LLM"
  - "extract structured data from receipts, forms or charts without a custom model"
  - "tag or search a photo library by natural-language description"
  - "caption images or generate alt text at scale"
  - "run an image-understanding model locally for privacy"
status: draft
last_verified: 2026-10-03
sources:
  - https://docs.anthropic.com/en/docs/build-with-claude/vision
  - https://huggingface.co/docs/transformers/tasks/image_text_to_text
  - https://github.com/mlfoundations/open_clip
---

# Vision-language models (VLMs)

## Summary
VLMs take images (and often video or PDFs) plus text and produce text. Hosted multimodal LLMs (Claude, GPT, Gemini families) and open models (Qwen-VL, Llama vision, others) handle captioning, visual question answering, chart/screenshot/document reading and flexible zero-shot classification without task-specific training. CLIP-style models instead embed images and text into one space for search and zero-shot matching.

## Key concepts
- Generative VLM: vision encoder + LLM; input image(s) plus a prompt, output free text or JSON.
- Contrastive models (CLIP, SigLIP): separate image and text encoders; cosine similarity gives zero-shot classification and image search; no text generation.
- Images cost tokens (size-dependent); downscale where fine detail is not needed. Check each provider's image limits.
- Grounding: some VLMs return boxes/points for referenced objects; accuracy varies, so verify before relying on coordinates.
- Structured output: ask for JSON schema output for extraction (see [[structured-output]]).
- Weaknesses: counting, precise spatial reasoning, small text, charts with dense data, hallucinated details.

## When to use / scenarios
- Back-office: invoices, forms, ID documents, tables; prototype extraction fast (compare with [[ocr]] and [[document-parsing]]).
- Ecommerce/media: attribute tagging, alt text, moderation triage, visual search with CLIP embeddings ([[embedding-models]]).
- Support/software: understand screenshots and UI bugs; computer-use style agents ([[agents]]).
- Accessibility: describe images for blind users (review for accuracy).
- Not for: high-volume, fixed-label tasks where a small trained model is 100x cheaper ([[image-classification]], [[object-detection]]); precise measurement or counting; medical diagnosis.

## Setup & code
Hosted API (Anthropic SDK, base64 image):
```bash
pip install anthropic
```
```python
import anthropic, base64

client = anthropic.Anthropic()                      # ANTHROPIC_API_KEY in env
img = base64.standard_b64encode(open("receipt.jpg", "rb").read()).decode()
msg = client.messages.create(
    model="MODEL_ID",                               # pick a current vision-capable model from the docs
    max_tokens=500,
    messages=[{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img}},
        {"type": "text", "text": "Return the merchant, date and total as JSON."},
    ]}],
)
print(msg.content[0].text)
```
Local: Hugging Face `transformers` image-text-to-text pipeline with an open VLM (see the task guide; choose a model that fits your VRAM, see [[huggingface-transformers]]). CLIP: `pip install open_clip_torch`.

## Choosing / trade-offs
- Hosted frontier VLM: best quality and zero setup; per-image cost, data leaves your network (see [[ai-security-privacy-compliance]]).
- Open VLM locally (via Ollama, vLLM, transformers): privacy and fixed cost; needs a GPU and lower accuracy on hard documents.
- CLIP/SigLIP embeddings: cheap, fast search and zero-shot classification; no reasoning or extraction.
- Distil: use a VLM to pre-label, then train a small specialised model for volume.
- Image resolution vs cost/latency; crop regions of interest rather than sending full pages.

## Gotchas
- Confident hallucinations of text and numbers in images; validate extracted fields (checksums, totals, schema) and keep a human in the loop for high stakes.
- Counting objects and reading small or rotated text is unreliable.
- Prompt injection via text embedded in images; treat image content as untrusted ([[prompt-injection]]).
- Image tokens inflate cost and context; batching many full-resolution pages is expensive.
- Do not send sensitive images (faces, medical, IDs) to a third-party API without a legal basis ([[face-and-pose]]).
- Model IDs and limits change; check the provider docs before hardcoding.
- Evaluate on your own labelled set ([[llm-evaluation]]) rather than public benchmarks.

## Related
- [[multimodal-models]] - how these models are built.
- [[ocr]] - dedicated text extraction.
- [[document-parsing]] - PDF/document pipelines.
- [[embedding-models]] - CLIP-style embeddings.
- [[structured-output]] - JSON extraction.

## References
- https://docs.anthropic.com/en/docs/build-with-claude/vision
- https://huggingface.co/docs/transformers/tasks/image_text_to_text
- https://github.com/mlfoundations/open_clip
