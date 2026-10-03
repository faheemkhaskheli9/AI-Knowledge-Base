---
title: Qwen models
category: models
tags: [qwen, alibaba, open-weights, apache-2, multilingual, embeddings, vllm]
use_cases:
  - "self-host a strong Apache-2.0 multilingual LLM for internal assistants"
  - "build a Chinese/English customer-service bot on open weights"
  - "use Qwen3 embedding and reranker models for a RAG stack"
  - "serve a 27B multimodal model on a single large GPU with vLLM"
status: draft
last_verified: 2026-10-03
sources:
  - https://huggingface.co/Qwen
  - https://huggingface.co/Qwen/Qwen3.8-27B
  - https://huggingface.co/Qwen/Qwen3-Embedding-8B
  - https://huggingface.co/Qwen/Qwen3-Reranker-8B
---

# Qwen models

## Summary
Qwen is Alibaba's model family, published openly on Hugging Face: LLMs from
small to very large, multimodal/vision, image generation, guard models,
embeddings and rerankers. Many checkpoints use Apache 2.0, making Qwen a
default choice for permissively licensed self-hosting.

## Key concepts
- Recent repos on the Qwen HF org (2026-10): Qwen3.8-27B, Qwen3.8-Flash-Next
  (180B), a multi-trillion-parameter Qwen3.8 flagship, Qwen-Image-2.1,
  Qwen3Guard-Stream, Qwen-Drive-1.0. Licenses vary per repo; check each card.
- Qwen3.8-27B (verified): Apache 2.0, 27B, native multimodal (image/video),
  hybrid Gated DeltaNet + attention, 262,144-token native context extensible
  to 1M, thinking on by default with `reasoning_effort` control.
- Qwen3-Embedding-8B and Qwen3-Reranker-8B: Apache 2.0, 32k context, 100+
  languages; embedding output sizes 32-4096 dims.

## When to use / scenarios
- Permissive-license self-hosting (no MAU clauses) - see [[model-licenses]].
- Multilingual products, especially Chinese, and agentic/coding tasks.
- A full open RAG stack: Qwen embeddings + Qwen reranker + Qwen LLM
  ([[embedding-models]], [[rerankers]]).
- Not for: workloads needing a US/EU vendor SLA; use hosted APIs, or check your
  compliance rules on provenance ([[ai-security-privacy-compliance]]).

## Setup & code
```bash
pip install vllm
vllm serve "Qwen/Qwen3.8-27B"      # OpenAI-compatible server, port 8000
```
```python
from openai import OpenAI

c = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
r = c.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=[{"role": "user", "content": "Translate to Urdu: payment received"}],
)
print(r.choices[0].message.content)
```
Transformers route: `AutoProcessor` + `AutoModelForMultimodalLM` from
`Qwen/Qwen3.8-27B` (per the model card).

## Choosing / trade-offs
- 27B fits one high-memory GPU in 8-bit or 4-bit ([[quantization]]); the
  flagship MoE models need multi-GPU clusters.
- Thinking mode improves reasoning but costs tokens/latency; lower
  `reasoning_effort` for simple tasks.

## Gotchas
- Not every Qwen repo is Apache 2.0; always read the card.
- Requires a recent `transformers`/`vllm`; old versions miss new architectures.
- Parameter counts on HF listings include vision encoders and may differ from
  names.

## Related
- [[meta-llama]], [[deepseek]], [[mistral]] - other open-weight families.
- [[inference-servers-vllm]] - serving.
- [[model-selection]] - open vs closed.

## References
- https://huggingface.co/Qwen
- https://huggingface.co/Qwen/Qwen3.8-27B
- https://huggingface.co/Qwen/Qwen3-Embedding-8B
