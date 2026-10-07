---
title: Reciprocal rank fusion (RRF) from scratch (hybrid BM25 + dense search)
category: llm-apps
tags: [reciprocal-rank-fusion, rrf, hybrid-search, rank-fusion, bm25, dense-retrieval, rag, retrieval, from-scratch]
use_cases:
  - "merge keyword (BM25) and vector search results into one ranked list for RAG"
  - "see why adding raw scores from different retrievers fails and rank-based fusion does not"
  - "pick between RRF and normalized score fusion for hybrid search"
  - "explain hybrid search and RRF in an interview"
status: stable
last_verified: 2026-10-04
sources:
  - https://doi.org/10.1145/1571941.1572114
  - https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html
---

# Reciprocal rank fusion (RRF) from scratch (hybrid BM25 + dense search)

## Summary
Hybrid search runs a keyword retriever (BM25) and a vector retriever, then merges their results. The scores cannot simply be added: BM25 scores are unbounded and often 0-30, while cosine similarities sit in a narrow band near 0.3. Reciprocal rank fusion ignores scores and uses only ranks: `RRF(d) = Σ 1 / (k + rank_i(d))`, with `k = 60` by default. It is about eight lines of NumPy. In a simulation where each retriever finds a different subset of the relevant documents, recall@10 is 0.45 for BM25 alone and 0.43 for dense alone. Adding raw scores gives 0.45, no better than BM25, because BM25's larger scale dominates. Min-max normalization gives 0.55, RRF with `k = 60` gives 0.53, and RRF with `k = 1` gives 0.64 on this data.

## Key concepts
- **Rank, not score.** Each retriever contributes `1 / (k + rank)`. Scales, distributions and units no longer matter, so any retrievers can be fused, including ones that return no scores.
- **The constant `k`.** It controls how fast credit decays with rank. Large `k` (60, from the original paper) flattens the curve, so documents found by both retrievers at middling ranks win. Small `k` gives most of the credit to each retriever's top few results.
- **Agreement bonus.** A document ranked 5th by both retrievers scores `2/65`, more than one ranked 1st by one and absent from the other (`1/61`). RRF rewards consensus.
- **Score-based fusion.** The alternative is to normalize each retriever's scores (min-max or z-score) and take a weighted sum. It keeps the score gaps that RRF throws away, but depends on the normalization and on outliers in each result list.
- **Fusion depth.** Fuse the top N from each retriever (often 50-100), not just the top 10, so documents one retriever ranks moderately can still surface.

## When to use / scenarios
- Learning: the merge step inside every "hybrid search" option in vector databases and search engines.
- Practice: RAG over documents with product codes, names or error messages (BM25 wins) and paraphrased questions (dense wins) ([[advanced-rag]]). Merging results from several query rewrites (RAG-Fusion), several indexes, or several embedding models.
- Interviews: "how do you combine BM25 and embeddings", "why not just add the scores".
- Not for: final ranking quality on its own. Fusion produces candidates; a cross-encoder reranker usually orders the top ones. Also not needed when one retriever clearly dominates on your evaluation set.

## Setup & code
NumPy only. Runs in under a second.

```python
import numpy as np

rng = np.random.default_rng(0)
n_docs, n_queries, n_rel = 2000, 200, 6


def rankings():
    """Simulate one query: a lexical (BM25-like) and a dense retriever, each blind to some relevant docs."""
    rel = rng.choice(n_docs, n_rel, replace=False)
    lexical = rng.gamma(2, 2, n_docs)                  # BM25-like scores, unbounded, ~0-25
    dense = rng.normal(0.30, 0.08, n_docs)             # cosine similarities, squashed near 0.3
    lexical[rel[:4]] += rng.uniform(8, 20, 4)          # keyword matches: lexical finds these
    dense[rel[2:]] += rng.uniform(0.15, 0.35, 4)       # paraphrases: dense finds these
    return set(rel), lexical, dense


def rrf(*score_lists, k=60):
    """Reciprocal rank fusion: sum over retrievers of 1 / (k + rank)."""
    fused = np.zeros(n_docs)
    for s in score_lists:
        ranks = np.empty(n_docs)
        ranks[np.argsort(-s)] = np.arange(1, n_docs + 1)
        fused += 1 / (k + ranks)
    return fused


def minmax(s):
    return (s - s.min()) / (s.max() - s.min())


methods = {
    "lexical only": lambda l, d: l,
    "dense only": lambda l, d: d,
    "raw score sum": lambda l, d: l + d,
    "min-max sum": lambda l, d: minmax(l) + minmax(d),
    "RRF k=60": lambda l, d: rrf(l, d),
    "RRF k=1": lambda l, d: rrf(l, d, k=1),
}
recall = {m: [] for m in methods}
for _ in range(n_queries):
    rel, lex, den = rankings()
    for m, f in methods.items():
        top = np.argsort(-f(lex, den))[:10]
        recall[m].append(len(rel & set(top)) / n_rel)
for m, r in recall.items():
    print(f"{m:14s} recall@10 = {np.mean(r):.3f}")
```

Output (Python 3.14, NumPy 2.5):
```
lexical only   recall@10 = 0.446
dense only     recall@10 = 0.434
raw score sum  recall@10 = 0.452
min-max sum    recall@10 = 0.549
RRF k=60       recall@10 = 0.532
RRF k=1        recall@10 = 0.643
```

Each retriever sees four of the six relevant documents, two of them shared, so a good fusion should beat both. Raw score addition barely moves from BM25 alone: dense scores stay below about 1, while BM25 noise alone reaches close to 20. Both min-max and RRF recover the documents only one retriever found. On this simulation `k = 1` beats the default `k = 60`, because each retriever's top few results are reliable and the long tail is pure noise, so top ranks deserve most of the credit. Real data will differ; tune `k` on your own labeled queries rather than trusting either number.

## Choosing / trade-offs
- **RRF vs normalized score fusion.** RRF needs no tuning beyond `k`, is robust to outlier scores, and works with any retriever. A tuned weighted sum of normalized scores can beat it when you have labeled queries, because it keeps how confident each retriever is. Start with RRF; move to weighted fusion if an evaluation shows a gain.
- **`k`.** 60 is a safe default from the original paper. Smaller values suit retrievers with sharp, reliable top ranks. Check recall@k or nDCG on 50-100 labeled queries ([[llm-evaluation]]).
- **Weights.** Weighted RRF (`w_i / (k + rank_i)`) lets you trust one retriever more, for example BM25 on code search.
- **Built-in options.** Elasticsearch, OpenSearch, Weaviate, Qdrant, Milvus and Azure AI Search offer RRF or score fusion for hybrid queries. Use them instead of fusing in application code when your store supports it ([[vector-databases]]).
- **After fusion.** Send the top 20-50 fused documents to a cross-encoder reranker, then optionally [[mmr-diversity-reranking-from-scratch]] to remove near-duplicates.

## Gotchas
- Documents missing from one retriever's list get no contribution from it. Fuse deep enough lists, or a document ranked 101st by BM25 counts the same as one that does not match at all.
- Ranks start at 1, not 0. With `k = 0` and 0-based ranks the top document divides by zero.
- De-duplicate by document ID before fusing. The same chunk returned under two IDs splits its credit and drops in the ranking.
- RRF scores have no absolute meaning. Do not threshold on them to decide "nothing relevant found"; use the reranker score or the retrievers' own scores for that.
- Min-max normalization depends on the best and worst document in each list, so one outlier compresses everything else. Z-scores or fusing fixed-depth lists reduce this.
- Fusing more weak retrievers is not free. Each one adds noise as well as recall; check the evaluation after each addition.

## Related
- [[advanced-rag]] - hybrid retrieval, reranking and query rewriting around this step.
- [[bm25-from-scratch]] - the lexical retriever being fused.
- [[rag-basics]] - dense retrieval, the other half of hybrid search.
- [[mmr-diversity-reranking-from-scratch]] - diversity reranking applied after fusion.
- [[learning-to-rank]] - learned combination of many ranking signals, the next step beyond fusion.
- [[vector-databases]] - stores with built-in hybrid search.

## References
- Cormack, Clarke and Buettcher (2009), "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods", SIGIR: https://doi.org/10.1145/1571941.1572114
- Elasticsearch reference, reciprocal rank fusion: https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html
