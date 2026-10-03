---
title: Small language models for edge and local use
category: models
tags: [slm, gemma, phi, smollm, ministral, edge, local, on-device]
use_cases:
  - "run an offline assistant on a laptop or phone without sending data to a cloud API"
  - "deploy a 1-4B model on a Raspberry Pi or kiosk for a retail or factory scenario"
  - "fine-tune a small model for one narrow classification task cheaply"
  - "pick a permissively licensed small model for an embedded product"
status: draft
last_verified: 2026-10-03
sources:
  - https://huggingface.co/google/gemma-4-31B-it
  - https://huggingface.co/google
  - https://huggingface.co/HuggingFaceTB
  - https://docs.mistral.ai/getting-started/models/models_overview/
  - https://en.wikipedia.org/wiki/Phi_(language_model)
---

# Small language models for edge and local use

## Summary
Small language models (SLMs, roughly 0.1B-15B parameters) run on laptops,
phones and small GPUs. They trade breadth for privacy, offline use, latency and
near-zero marginal cost, and become strong after task-specific fine-tuning.

## Key concepts
- Quantization (4-bit GGUF etc.) shrinks memory 3-4x; see [[quantization]].
- "Effective" parameters: Gemma 4 E2B/E4B are 2.3B/4.5B effective params with
  128K context, built for devices; the 12B, 26B-A4B (MoE, 3.8B active) and 31B
  sizes have 256K context. License: Apache 2.0 (Gemma 4 card).
- Mistral Ministral 3 (3B, 8B, 14B; text + vision; Apache 2.0).
- Microsoft Phi-4 family: mini 3.8B (128K ctx), multimodal 5.6B, reasoning 14B,
  reasoning-vision 15B (Mar 2026), all MIT per secondary sources; verify on the
  model cards, as I could not load the Microsoft HF org page.
- Hugging Face SmolLM3-3B (6 languages, long context, function calling) and
  SmolLM2 135M/360M/1.7B; license not verified, check the card.
- Qwen also ships small sizes ([[qwen]]); Llama 3.2 1B/3B ([[meta-llama]]).

## When to use / scenarios
- Privacy-critical on-device use (clinic tablets, field-service apps).
- Offline products: in-car, kiosks, industrial HMIs ([[edge-on-device]]).
- High-volume narrow tasks (intent routing, PII tagging, log triage) where a
  tuned 1-4B model beats paying for a frontier API.
- Draft/speculative-decoding helpers for larger models.
- Not for: open-ended reasoning or long agentic work; use [[model-selection]].

## Setup & code
```bash
ollama pull gemma3:4b    # check `ollama search`/library for the current tag
pip install ollama
```
```python
import ollama

r = ollama.chat(model="gemma3:4b", messages=[{"role": "user", "content": "Tag intent: 'where is my parcel'"}])
print(r["message"]["content"])
```
The tag above is an example; confirm the current Gemma/Phi tag in the Ollama
library ([[ollama]]). For Transformers, `google/gemma-4-31B-it` loads with
`AutoModelForMultimodalLM`; use the E2B/E4B repos for small hardware.

## Choosing / trade-offs
- Quality rises steeply from 1B to 4B to 8B; pick the largest your memory/latency
  budget allows.
- Fine-tune with LoRA ([[fine-tuning-and-peft]]) for a narrow task.
- MIT/Apache models avoid license friction in shipped products
  ([[model-licenses]]).

## Gotchas
- Small models hallucinate and follow long instructions poorly; constrain with
  [[structured-output]] and verify.
- Context windows quoted are maximums; KV cache memory limits usable length on
  devices.
- Phone runtimes lag desktop ones; test on the target hardware.

## Related
- [[llama-cpp-gguf]] - GGUF runtime.
- [[ollama]] - easiest local runner.
- [[edge-on-device]] - deployment concerns.
- [[quantization]] - fitting models into memory.

## References
- https://huggingface.co/google/gemma-4-31B-it
- https://huggingface.co/HuggingFaceTB
- https://docs.mistral.ai/getting-started/models/models_overview/
