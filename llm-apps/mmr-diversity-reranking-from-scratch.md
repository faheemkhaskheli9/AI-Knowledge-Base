---
title: Maximal Marginal Relevance (MMR) for diverse retrieval from scratch (RAG dedup, lambda trade-off)
category: llm-apps
tags: [mmr, maximal-marginal-relevance, diversity, reranking, rag, retrieval, near-duplicates, embeddings, numpy, from-scratch]
use_cases:
  - "stop RAG top-k from filling the context with near-duplicate chunks"
  - "return search or recommendation results that cover several aspects of a query"
  - "tune the lambda / fetch_k settings of a vector store's MMR search"
  - "explain the relevance vs diversity trade-off in an interview"
status: draft
last_verified: 2026-10-04
sources:
  - https://doi.org/10.1145/290941.291025
  - https://python.langchain.com/api_reference/core/vectorstores/langchain_core.vectorstores.base.VectorStore.html
---

# Maximal Marginal Relevance (MMR) for diverse retrieval from scratch (RAG dedup, lambda trade-off)

## Summary
Pure similarity top-k returns the k chunks closest to the query, and those are often near-copies of each other: the same paragraph from five versions of a document, or one FAQ answer repeated across pages. That wastes the LLM's context and hides other relevant facts. Maximal Marginal Relevance (Carbonell and Goldstein, 1998) builds the result list greedily. At each step it adds the candidate with the best `λ · sim(query, d) − (1 − λ) · max_{s ∈ selected} sim(d, s)`, rewarding relevance and penalizing redundancy with what is already chosen. Below, a query has 5 relevant aspects, and one of them has 12 near-duplicate chunks. Similarity top-5 returns 5 chunks of that one aspect (mean pairwise similarity 0.98). MMR with `λ = 0.5` returns one chunk from each of the 5 aspects, while mean relevance drops only from 0.88 to 0.76.

## Key concepts
- **Greedy selection.** Start with the most relevant candidate. Then repeatedly add `argmax_d [λ·rel(d) − (1−λ)·max_s sim(d, s)]`. The cost is `O(k · fetch_k)` similarity computations, which is negligible next to the retrieval itself.
- **`λ` (lambda_mult).** `λ = 1` is plain similarity ranking, and `λ = 0` picks only for dissimilarity. Useful values sit around 0.5-0.7. LangChain's default `lambda_mult` is 0.5.
- **`fetch_k`.** Run MMR over the top `fetch_k` by similarity, not the whole corpus. This keeps diversity from pulling in irrelevant documents: with `λ = 0` over everything, you would get random off-topic chunks.
- **Max, not mean.** The redundancy term uses the similarity to the closest already-selected item, so one near-duplicate is enough to suppress a candidate.
- **Same embedding space.** Relevance and redundancy are both cosine similarities from the same embedding model, so they are on comparable scales and `λ` means something.

## When to use / scenarios
- Practice: RAG over corpora with heavy duplication, such as versioned docs, email threads with quoted replies, support macros, scraped pages with shared boilerplate, and overlapping chunk windows. Also multi-aspect questions ("pros and cons of X"), search-result and news diversification, and recommendation carousels that should not show five near-identical items. Most vector stores and frameworks expose it as `max_marginal_relevance_search` / `search_type="mmr"`.
- Learning: the simplest diversity-aware ranking, and a template for any "utility minus redundancy" greedy selection.
- Interviews: "my RAG context is full of duplicates", "relevance vs diversity".
- Not for: exact duplicates, which you should remove at ingestion with hashing or MinHash ([[locality-sensitive-hashing-from-scratch]]) rather than at query time; single-fact lookups where the top chunk is all you need; or when a cross-encoder reranker is the bottleneck for quality. Rerank for relevance first, then apply MMR over the reranked list ([[advanced-rag]]).

## Setup & code
NumPy only. Runs in under a second.

```python
import numpy as np

rng = np.random.default_rng(0)
d = 64
unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)

query = unit(rng.normal(size=d))
# 5 sub-topics ("aspects") relevant to the query; aspect 0 is the closest and has many near-duplicates
aspects = unit(query + 0.9 * unit(rng.normal(size=(5, d))))
aspects[0] = unit(query + 0.5 * unit(rng.normal(size=d)))
sizes = [12, 3, 3, 3, 3]
docs, label = [], []
for a, (vec, size) in enumerate(zip(aspects, sizes)):
    for _ in range(size):
        docs.append(unit(vec + 0.15 * unit(rng.normal(size=d))))
        label.append(a)
docs.extend(unit(rng.normal(size=(30, d))))      # off-topic noise
label.extend([-1] * 30)
docs, label = np.array(docs), np.array(label)


def mmr(query, docs, k, lam=0.5, fetch=None):
    """Maximal Marginal Relevance: pick argmax lam*sim(q,d) - (1-lam)*max_s sim(d,s) over selected s."""
    rel = docs @ query
    cand = list(np.argsort(rel)[::-1][:fetch or len(docs)])
    chosen = [cand.pop(0)]
    while len(chosen) < k and cand:
        red = (docs[cand] @ docs[chosen].T).max(1)
        scores = lam * rel[cand] - (1 - lam) * red
        chosen.append(cand.pop(int(np.argmax(scores))))
    return chosen


def report(name, idx):
    rel = docs[idx] @ query
    sims = docs[idx] @ docs[idx].T
    pairwise = sims[np.triu_indices(len(idx), 1)].mean()
    print(f"{name:<22} aspects {str(label[idx].tolist()):<17} distinct {len(set(label[idx]) - {-1})}  "
          f"mean rel {rel.mean():.3f}  mean pairwise sim {pairwise:.3f}")


k = 5
report("top-k by similarity", list(np.argsort(docs @ query)[::-1][:k]))
for lam in [0.9, 0.7, 0.5, 0.3, 0.0]:
    report(f"MMR lambda={lam}", mmr(query, docs, k, lam, fetch=20))
```

Output (Python 3.14, NumPy 2.5):
```
top-k by similarity    aspects [0, 0, 0, 0, 0]   distinct 1  mean rel 0.882  mean pairwise sim 0.979
MMR lambda=0.9         aspects [0, 0, 0, 0, 0]   distinct 1  mean rel 0.882  mean pairwise sim 0.979
MMR lambda=0.7         aspects [0, 4, 0, 0, 0]   distinct 2  mean rel 0.863  mean pairwise sim 0.858
MMR lambda=0.5         aspects [0, 4, 3, 1, 2]   distinct 5  mean rel 0.758  mean pairwise sim 0.573
MMR lambda=0.3         aspects [0, 4, 3, 1, 2]   distinct 5  mean rel 0.752  mean pairwise sim 0.560
MMR lambda=0.0         aspects [0, 4, 3, 1, 2]   distinct 5  mean rel 0.752  mean pairwise sim 0.560
```

The 12 near-duplicates of aspect 0 take every top-5 slot under plain similarity, and `λ = 0.9` is too weak to change that. The transition is sharp: between 0.7 and 0.5, the redundancy penalty of a near-duplicate (similarity ≈ 0.98) outweighs its relevance advantage, and MMR switches to one chunk per aspect. Even `λ = 0` returns only relevant chunks here, because `fetch_k = 20` keeps the candidate pool on-topic. Remove `fetch` and run MMR over the whole corpus: `λ = 0` returns 4 off-topic chunks out of 5 (mean relevance 0.11), and even `λ = 0.5` lets 2 in (mean relevance 0.55). Where the switch happens depends on how tight your duplicates are and how far apart your aspects sit, so tune `λ` on real queries.

## Choosing / trade-offs
- **`λ`.** Start at 0.5-0.7. Go higher for factoid QA (relevance matters most) and lower for exploratory, multi-aspect or summarization queries. Evaluate with answer quality or aspect coverage on a labelled set ([[llm-evaluation]]), not by eyeballing a single query.
- **`fetch_k`.** Typically 3-5× `k` (LangChain's default is `fetch_k=20` for `k=4`). Too small and there is nothing diverse to pick; too large and MMR can surface weakly relevant chunks.
- **MMR vs dedup at ingestion.** Exact and near-exact duplicates are cheaper to drop once, at indexing time. MMR handles the semantic overlap that remains, such as different wording of the same fact.
- **MMR vs alternatives.** Determinantal point processes (DPPs) model diversity jointly rather than greedily, but are heavier. Clustering candidates and taking the best chunk per cluster is simpler and works when aspects are well separated. Cross-encoder reranking improves relevance but does not reduce redundancy, so the two combine well.

## Gotchas
- Use the same normalized embeddings for both terms. Mixing a reranker's relevance score (different scale) with cosine redundancy makes `λ` meaningless unless you rescale the scores to a comparable range.
- MMR changes the ordering, so position-sensitive metrics (MRR, nDCG on relevance labels) will drop even when answers improve. Measure end-task quality.
- Overlapping chunk windows (for example 20% overlap) create built-in near-duplicates. Either expect MMR to fight them or reduce the overlap.
- Some vector stores compute MMR on stored embeddings, others re-embed; some normalize, others do not. Check what your store does before copying `λ` values between them.
- A diverse set can still be all wrong if retrieval itself failed. MMR only reorders what similarity search fetched, so it cannot fix bad recall.

## Related
- [[rag-basics]] - the retrieval step MMR sits on top of.
- [[advanced-rag]] - reranking, hybrid search and context assembly around MMR.
- [[vector-databases]] - stores that expose MMR search, and their parameters.
- [[bm25-from-scratch]] - the lexical retriever whose candidates can also be diversified.
- [[embeddings]] - the similarity space both MMR terms use.

## References
- Carbonell and Goldstein (1998), "The use of MMR, diversity-based reranking for reordering documents and producing summaries", SIGIR: https://doi.org/10.1145/290941.291025
- LangChain `VectorStore.max_marginal_relevance_search` (parameters `k`, `fetch_k`, `lambda_mult`): https://python.langchain.com/api_reference/core/vectorstores/langchain_core.vectorstores.base.VectorStore.html
