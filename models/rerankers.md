---
title: Rerankers
category: models
tags: [reranker, cross-encoder, cohere-rerank, qwen3-reranker, rag, retrieval]
use_cases:
  - "improve RAG answer quality by reranking the top 50 retrieved chunks"
  - "rerank search results for an e-commerce catalogue query"
  - "run a self-hosted reranker for confidential legal documents"
  - "cut LLM cost by sending only the best 5 chunks to the model"
status: draft
last_verified: 2026-10-03
sources:
  - https://docs.cohere.com/docs/rerank
  - https://huggingface.co/Qwen/Qwen3-Reranker-8B
  - https://docs.voyageai.com/docs/embeddings
---

# Rerankers

## Summary
A reranker scores (query, document) pairs jointly with a cross-encoder or LLM,
giving more accurate ordering than vector similarity alone. Typical use: fast
embedding retrieval returns 50-100 candidates, the reranker keeps the best 3-10
for the LLM.

## Key concepts
- Two-stage retrieval: bi-encoder (cheap, recall) then reranker (precise).
- Cost scales with number of candidates times document length.
- Context limit applies to query + document combined (Cohere docs).

Current options as of 2026-10:

| Model | Type | Notes (source) |
|---|---|---|
| `rerank-v4.0-pro` | Cohere API | multilingual, highest quality |
| `rerank-v4.0-fast` | Cohere API | low latency/high throughput |
| `rerank-v3.5`, `rerank-english-v3.0`, `rerank-multilingual-v3.0` | Cohere API | 4096-token per-document limit |
| Qwen3-Reranker (up to 8B) | open, Apache 2.0 | 32k context, 100+ languages, instruction-aware |

Voyage also sells rerankers and others (BGE etc.) exist; I did not verify their
current IDs.

## When to use / scenarios
- RAG where top-k retrieval is noisy ([[advanced-rag]]).
- Enterprise search over mixed-quality documents.
- Multilingual retrieval.
- Not needed when the corpus is tiny or latency budget is under ~100 ms and
  embeddings already perform well; measure first.

## Setup & code
```bash
pip install cohere
export CO_API_KEY=...
```
```python
import cohere

co = cohere.ClientV2()
r = co.rerank(
    model="rerank-v4.0-fast",
    query="refund window for damaged items",
    documents=["Shipping takes 5 days", "Damaged items can be refunded within 30 days"],
    top_n=1,
)
print(r.results[0].index, r.results[0].relevance_score)
```
Open-weight route (sentence-transformers):
```python
from sentence_transformers import CrossEncoder
ce = CrossEncoder("Qwen/Qwen3-Reranker-8B")  # per the model card; smaller sizes exist
print(ce.predict([("refund window", "Damaged items can be refunded within 30 days")]))
```

## Choosing / trade-offs
- API reranker: easy, per-search pricing, data leaves your network.
- Open reranker: GPU needed, full control.
- More candidates raise recall and latency; typical sweet spot 30-100.
- Fast vs pro tier: pick fast unless your eval shows a gap.

## Gotchas
- Truncated documents lose the answer; chunk to fit the limit.
- Scores are relative per query, not comparable across queries.
- Verify the CrossEncoder snippet works with your sentence-transformers version
  (Qwen's card lists it as the simplest route, requires recent transformers).

## Related
- [[embedding-models]] - first-stage retrieval.
- [[rag-basics]] - pipeline context.
- [[llm-evaluation]] - measuring the gain.

## References
- https://docs.cohere.com/docs/rerank
- https://huggingface.co/Qwen/Qwen3-Reranker-8B
