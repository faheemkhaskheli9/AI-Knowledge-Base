---
title: Embedding models (API and open)
category: models
tags: [embeddings, mteb, voyage, openai, gemini, qwen3-embedding, sentence-transformers, rag]
use_cases:
  - "choose an embedding model for a RAG system over company documents"
  - "embed multilingual support tickets for semantic search"
  - "run a free open-source embedding model locally for private documents"
  - "reduce vector storage cost with shortened or quantized embeddings"
status: draft
last_verified: 2026-10-03
sources:
  - https://developers.openai.com/api/docs/guides/embeddings
  - https://docs.voyageai.com/docs/embeddings
  - https://ai.google.dev/gemini-api/docs/pricing
  - https://huggingface.co/Qwen/Qwen3-Embedding-8B
  - https://mteb-leaderboard.hf.space/
---

# Embedding models (API and open)

## Summary
Embedding models turn text (and sometimes images/audio) into vectors for
semantic search, RAG, clustering and deduplication. Pick by retrieval quality
on your own data, language coverage, dimension/cost and where it can run. MTEB
is the public leaderboard but not a substitute for your own eval.

## Key concepts
- Query vs document: many models want an `input_type`/instruction prefix.
- Dimensions: Matryoshka-style models can be truncated (OpenAI `dimensions`
  parameter; Qwen3-Embedding 32-4096; Voyage 4 series 256/512/1024/2048).
- Quantized outputs (int8/binary) cut vector-DB cost (Voyage supports them).
- Never mix vectors from different models in one index.

Current options as of 2026-10:

| Model | Type | Notes (source) |
|---|---|---|
| `text-embedding-3-small` / `-large` | API | 1536 / 3072 dims, 8,192 input tokens (OpenAI) |
| `voyage-4-large`, `voyage-4`, `voyage-4-lite` | API | 32K tokens, 1024 dims default; `voyage-code-4`, `voyage-finance-2`, `voyage-law-2` domain models; `voyage-4-nano` open on HF |
| Gemini Embedding 2 | API | text, image, audio and video; text $0.20/1M tokens (Google pricing) |
| Qwen3-Embedding-8B | open, Apache 2.0 | 32K context, up to 4096 dims, 100+ languages, MTEB multilingual 70.58 at #1 on 2025-06-05 (stale) |
| Mistral Embed, Codestral Embed | API (Premier) | see [[mistral]] |

I could not fetch the live MTEB ranking; check it before deciding. Cohere
embed and others exist but were not verified here.

## When to use / scenarios
- RAG over PDFs/wikis ([[rag-basics]], [[advanced-rag]]).
- Domain search: legal/finance/code models (Voyage) when the domain matches.
- Private/on-prem: Qwen3-Embedding or other open models via sentence-transformers.
- Multimodal retrieval over images/video: Gemini Embedding 2.
- Not for: ranking final results precisely; add a reranker ([[rerankers]]).

## Setup & code
```bash
pip install sentence-transformers
```
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Qwen/Qwen3-Embedding-8B")  # large; use a smaller Qwen3-Embedding size on small GPUs
docs = model.encode(["Refund policy: 30 days", "Shipping takes 5 days"])
q = model.encode(["how long do I have to return an item?"])
print(model.similarity(q, docs))
```
API version:
```python
from openai import OpenAI
v = OpenAI().embeddings.create(model="text-embedding-3-small", input="hello", dimensions=512)
print(len(v.data[0].embedding))
```

## Choosing / trade-offs
- API: no GPUs, pay per token, data leaves your network.
- Open: fixed cost, privacy, tunable; need GPU for 8B-class models.
- Smaller dims save storage/latency with a modest quality loss; test it.
- Re-embedding the corpus is costly: choose a model you can live with.

## Gotchas
- Leaderboard overfitting: top MTEB scores may not transfer; run a 50-200 query
  eval on your data ([[llm-evaluation]]).
- Truncating long inputs silently hurts recall; chunk sensibly.
- Normalize vectors if your DB uses dot product vs cosine.

## Related
- [[embeddings]] - the concept.
- [[vector-databases]] - where vectors live.
- [[rerankers]] - second-stage ranking.
- [[model-licenses]] - open model licenses.

## References
- https://developers.openai.com/api/docs/guides/embeddings
- https://docs.voyageai.com/docs/embeddings
- https://huggingface.co/Qwen/Qwen3-Embedding-8B
- https://mteb-leaderboard.hf.space/
