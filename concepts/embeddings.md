---
title: Embeddings
category: concepts
tags: [embeddings, vectors, cosine-similarity, semantic-search, sentence-transformers]
use_cases:
  - "build semantic search over a company knowledge base or product catalogue"
  - "cluster customer feedback to find recurring complaint themes"
  - "detect near-duplicate support tickets or listings"
  - "retrieve relevant chunks for a RAG assistant in a legal or healthcare setting"
  - "add a cheap similarity-based recommender to an e-commerce site"
status: draft
last_verified: 2026-10-03
sources:
  - https://www.sbert.net/
  - https://arxiv.org/abs/1908.10084
  - https://huggingface.co/spaces/mteb/leaderboard
---

# Embeddings

## Summary
An embedding maps text (or images, audio, code) to a fixed-length vector so that semantically similar items are close in vector space. It turns "find things that mean the same" into fast nearest-neighbour math, and underlies semantic search, RAG retrieval, clustering, deduplication and recommendation.

## Key concepts
- **Dense vector**: typically a few hundred to a few thousand floats per item.
- **Similarity**: cosine similarity (or dot product on normalised vectors); L2 is equivalent for unit vectors.
- **Bi-encoder**: query and document are embedded separately (fast, indexable). A **cross-encoder / reranker** scores pairs jointly (slower, more accurate); see [[rerankers]].
- **Task prefixes / instructions**: many models need "query:" vs "passage:" style prefixes; read the model card.
- **Dimension and truncation**: some models support Matryoshka truncation to shorter vectors with small quality loss.
- **ANN index**: HNSW/IVF indexes give approximate nearest neighbours at scale (see [[vector-databases]]).
- **Sparse vs dense**: BM25/keyword catches exact terms and IDs; combine with dense (hybrid) for best retrieval.
- Embeddings from different models are not comparable. Changing model means re-embedding everything.

## When to use / scenarios
- Retrieval for RAG, FAQ matching, "similar items", intent routing, clustering, anomaly detection on text.
- Real-world: insurer matches a claim description to past claims; retailer finds similar products by description and image embeddings; a hospital searches clinical guidelines (with appropriate privacy controls).
- Not suited: exact lookups (use SQL/keyword), numeric reasoning, or when negation and fine distinctions matter (add a reranker).

## Setup & code
`pip install sentence-transformers`

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")
docs = ["Reset your password from the login page",
        "Our refund policy lasts 30 days",
        "How to change account email"]
doc_vecs = model.encode(docs, normalize_embeddings=True)
q = model.encode("I forgot my login", normalize_embeddings=True)
scores = util.cos_sim(q, doc_vecs)[0]
print(docs[int(scores.argmax())], float(scores.max()))
```

For production pick a model from the MTEB leaderboard that matches your language and domain, then evaluate on your own labelled queries (see [[embedding-models]]).

## Choosing / trade-offs
- API embeddings (no infra, per-token cost, data leaves your boundary) vs local open models (private, GPU/CPU cost).
- Dimension: higher gives slightly better recall but more storage and slower search.
- Chunk size: small chunks are precise but lose context; large chunks dilute the vector.
- Fine-tune embeddings only after hybrid search + reranking are tuned.

## Gotchas
- Mixing models between indexing and querying silently returns garbage.
- Forgetting to normalise when using dot product.
- Leaderboard rank on English benchmarks does not predict performance in Urdu, legal or code domains; test yourself.
- Truncation: inputs longer than the model's max length are cut silently.
- Short queries vs long passages are asymmetric; use the model's query/passage conventions.
- Embeddings can leak information about the source text; treat vectors as sensitive data.

## Related
- [[embedding-models]] - which models to choose.
- [[rag-basics]] - main consumer of embeddings.
- [[vector-databases]] - storing and searching vectors.
- [[rerankers]] - improving top-k precision.
- [[tokenization]] - token limits of embedding models.
- [[anomaly-detection]] - embedding distance as an outlier signal.

## References
- Sentence-Transformers docs: https://www.sbert.net/
- Reimers & Gurevych, Sentence-BERT: https://arxiv.org/abs/1908.10084
- MTEB leaderboard: https://huggingface.co/spaces/mteb/leaderboard
