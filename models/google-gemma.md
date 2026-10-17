---
title: Google Gemma open-weight models
category: models
tags: [gemma, gemma-4, google, open-weights, apache-2, on-device, embeddinggemma, shieldgemma, medgemma, ollama]
use_cases:
  - "self-host an Apache-2.0 multimodal model on one GPU for internal tools"
  - "run an assistant on a phone or laptop with a 2-4B effective-parameter model"
  - "build on-device semantic search with a small open embedding model"
  - "add an open image-safety classifier in front of a user-upload feature"
status: draft
last_verified: 2026-10-15
sources:
  - https://ai.google.dev/gemma/docs
  - https://ai.google.dev/gemma/docs/core
  - https://ai.google.dev/gemma/docs/core/model_card_4
  - https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/
  - https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12B/
  - https://ai.google.dev/gemma/docs/gemma-3n
  - https://ai.google.dev/gemma/docs/embeddinggemma
  - https://ai.google.dev/gemma/docs/shieldgemma
  - https://developers.google.com/health-ai-developer-foundations/medgemma/model-card
  - https://ai.google.dev/gemma/terms
  - https://ai.google.dev/gemma/docs/integrations/ollama
  - https://ollama.com/library/gemma4
---

# Google Gemma open-weight models

## Summary
Gemma is Google's open-weight model family, built from the same research as
Gemini but downloadable and self-hostable. The current generation, Gemma 4,
is multimodal (text, image, audio in), comes in sizes from on-device "E2B" to a
31B dense model, and is released under Apache 2.0. Specialised variants cover
embeddings, safety classification and medicine.

## Key concepts
- Gemma 4 sizes (model card): E2B, E4B, 12B, 26B A4B and 31B, each as
  pre-trained and instruction-tuned checkpoints. The launch blog listed four
  sizes; the 12B (encoder-free multimodal, runs in about 16 GB of VRAM or
  unified memory per Google) was added later.
- "E" = effective parameters: E2B/E4B use per-layer embedding tricks so the
  memory-resident model is smaller than the full parameter count; built for
  phones and edge devices. 128K context.
- 26B A4B is a mixture-of-experts model with about 4B active parameters per
  token ([[mixture-of-experts]]); 31B is dense. Larger sizes go up to 256K
  context.
- Licences differ by generation: Gemma 4 and EmbeddingGemma 2 are Apache 2.0;
  earlier generations (Gemma 1-3, Gemma 3n) use the Gemma Terms of Use with a
  prohibited-use policy; MedGemma uses the Health AI Developer Foundations
  terms ([[model-licenses]]).

Variants as of 2026-10-15 (sources: ai.google.dev Gemma docs, Google blogs):

| Variant | What it is | Licence |
|---|---|---|
| Gemma 4 (E2B, E4B, 12B, 26B A4B, 31B) | general multimodal LLM, 140+ languages | Apache 2.0 |
| Gemma 3n (E2B, E4B) | earlier on-device model (MatFormer) | Gemma Terms of Use |
| EmbeddingGemma | 300M text embedding model built on Gemma 3 | Gemma Terms of Use |
| EmbeddingGemma 2 | Gemma 4-based embeddings for text, image, audio, video | Apache 2.0 |
| ShieldGemma 2 | 4B image safety classifier (ShieldGemma 1: 2B/9B/27B text) | check card |
| MedGemma / MedGemma 1.5 | medical text and image models (4B multimodal, 27B) | HAI-DEF terms |
| CodeGemma, PaliGemma | older code and vision-language variants | check card |

Ollama tags (ollama.com/library/gemma4): `gemma4:e2b`, `gemma4:e4b`,
`gemma4:12b`, `gemma4:26b`, `gemma4:31b`, plus quantized (`-q4_K_M`, `-q8_0`)
and Apple MLX builds. A `-cloud` tag runs on Ollama's hosted service, not
locally.

## When to use / scenarios
- Self-hosting with a clean commercial licence: Gemma 4 under Apache 2.0 for
  internal assistants, document Q&A, image understanding ([[rag-basics]]).
- On-device apps (offline note-taking, in-car voice, kiosks) with E2B/E4B
  ([[edge-on-device]], [[small-language-models]]).
- Local semantic search and RAG with EmbeddingGemma ([[embedding-models]]).
- Moderating user-uploaded images with ShieldGemma 2 ([[guardrails-and-safety]]).
- Healthcare prototypes with MedGemma, after reading the HAI-DEF terms; it is
  not a cleared medical device.
- Not for: frontier-level reasoning or very long agentic tasks without
  testing; use [[google-gemini]] (hosted) or another frontier API.

## Setup & code
Sizing check (runs offline). An MoE model must hold all experts in memory,
even though only a few run per token:
```python
SIZES = {  # (total params, active params per token) from the Gemma 4 model card names
    "gemma4 26B A4B (MoE)": (26e9, 4e9),
    "gemma4 31B (dense)": (31e9, 31e9),
    "gemma4 12B (dense)": (12e9, 12e9),
}

for name, (total, active) in SIZES.items():
    mem_q4 = total * 4 / 8 / 1e9          # 4-bit weights must all be resident
    flops_ratio = active / total          # compute per token vs a dense model of the same size
    print(f"{name}: ~{mem_q4:.0f} GB at 4-bit, compute per token = {flops_ratio:.0%} of total params")
```

Local chat with Ollama ([[ollama]]):
```bash
ollama pull gemma4:e4b
pip install ollama
```
<!-- skip-check: needs a running Ollama server and a model download -->
```python
import ollama

r = ollama.chat(
    model="gemma4:e4b",
    messages=[{"role": "user", "content": "Summarise in one line: the 3pm delivery moved to Friday."}],
)
print(r["message"]["content"])
```
With Transformers, the Gemma 4 cards load via `AutoModelForMultimodalLM`
(e.g. `google/gemma-4-31B-it`); older Gemma repos ask you to accept the Gemma
Terms of Use before download ([[huggingface-transformers]]). Google also documents LiteRT-LM for Android/iOS
deployment.

## Choosing / trade-offs
- E2B/E4B: phones, laptops without a GPU. 12B: a single 16 GB GPU or Apple
  silicon laptop. 26B A4B: faster per token than 31B but needs similar memory.
  31B dense: best quality, needs a 24 GB+ GPU at 4-bit or a multi-GPU server
  ([[inference-servers-vllm]]).
- Gemma 4 (Apache 2.0) vs Gemma 3 (Gemma Terms of Use): prefer 4 for new
  commercial work; the older terms carry use restrictions that flow down to
  fine-tunes.
- Gemma vs Gemini: open weights and local data control vs a hosted frontier
  model with no ops.

## Gotchas
- "E4B" is not a 4B model in file size; check the actual download size and
  memory on your device.
- MoE 26B A4B saves compute, not memory.
- Licence depends on the exact checkpoint: a Gemma 3-based EmbeddingGemma or
  ShieldGemma is not Apache 2.0 just because Gemma 4 is.
- Context window maximums (128K / 256K) are rarely usable on consumer hardware
  because of KV-cache memory.
- Quantized community builds can use the wrong chat template; prefer the
  official Ollama library or Google's own releases.

## Related
- [[small-language-models]] - Gemma among other small models.
- [[microsoft-phi]] - the other main MIT/Apache small-model family.
- [[google-gemini]] - the hosted sibling family.
- [[model-licenses]] - Apache 2.0 vs Gemma Terms of Use.
- [[ollama]] - easiest way to run Gemma locally.
- [[edge-on-device]] - mobile and edge deployment.
- [[mixture-of-experts]] - why 26B A4B is fast but large.

## References
- https://ai.google.dev/gemma/docs
- https://ai.google.dev/gemma/docs/core/model_card_4
- https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/
- https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12B/
- https://ai.google.dev/gemma/docs/gemma-3n
- https://ai.google.dev/gemma/docs/embeddinggemma/model_card
- https://ai.google.dev/gemma/docs/shieldgemma
- https://developers.google.com/health-ai-developer-foundations/medgemma/model-card
- https://developers.google.com/edge/litert-lm/models/gemma-4
- https://ai.google.dev/gemma/terms
- https://ai.google.dev/gemma/docs/integrations/ollama
- https://ollama.com/library/gemma4/tags
