---
title: Product quantization for vector search from scratch (PQ codebooks, ADC lookup tables, rerank)
category: llm-apps
tags: [product-quantization, pq, vector-search, ann, faiss, ivf-pq, compression, embeddings, numpy, from-scratch]
use_cases:
  - "compress an embedding index 16-64x so it fits in RAM"
  - "understand what IVF-PQ / PQ settings in FAISS, Milvus or a vector DB actually do"
  - "pick the number of PQ subquantizers (m) for a recall / memory budget"
  - "explain product quantization and asymmetric distance computation in an interview"
status: draft
last_verified: 2026-10-04
sources:
  - https://doi.org/10.1109/TPAMI.2010.57
  - https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
  - https://faiss.ai/
---

# Product quantization for vector search from scratch (PQ codebooks, ADC lookup tables, rerank)

## Summary
Product quantization (PQ) compresses each vector into `m` bytes. It splits the `d` dimensions into `m` sub-vectors, runs k-means with 256 centroids in each subspace, and stores only the index of the nearest centroid per subspace. To search, the query stays exact. You build an `m × 256` table of query-to-centroid distances once, and the approximate distance to every stored vector becomes `m` table lookups added together (asymmetric distance computation, ADC). Below, 20,000 64-d vectors (5.12 MB as float32) are compressed to 0.08-0.32 MB. With `m = 16` (16x smaller), the true nearest neighbour is in the PQ top-10 for 98.5% of queries. Fetching the top 100 by PQ and reranking them with exact distances recovers 100% of the true top-10. At `m = 4` (64x smaller), recall falls to 0.48 without reranking and 0.80 with it.

## Key concepts
- **Subspace codebooks.** Split vectors into `m` chunks of `d/m` dims and train one k-means (`k = 256`, so one byte per code) per chunk. The implied codebook has `256^m` centroids, a product of small ones. That is how a few KB of codebooks can represent a fine grid.
- **Encoding.** Each vector becomes `m` uint8 codes: 4 bytes per float dimension compress down to `m` bytes per vector. For 64-d float32 that is 256 bytes → 16 bytes at `m = 16`.
- **ADC (asymmetric distance).** Compare the exact query with the quantized database vectors. Per query, precompute `table[j][c] = ||q_j − centroid_jc||²`. Then `dist(q, x) ≈ Σ_j table[j][code_j(x)]`. This is cheaper than decoding vectors, and more accurate than also quantizing the query (symmetric distance).
- **Rerank.** PQ distances are approximate, so close candidates get swapped. Take the top `R` by PQ, then re-score them with the original vectors (from disk or a slower store) to recover most of the lost recall.
- **IVF-PQ.** Production indexes first route the query to a few coarse k-means cells (IVF) and only scan those lists' PQ codes. They usually encode the residual `x − cell_centroid`, which makes PQ more accurate.

## When to use / scenarios
- Learning: what the `M`, `nbits`, `nlist` and `nprobe` knobs mean in FAISS `IndexIVFPQ`, Milvus `IVF_PQ`, or the PQ / compression options in other vector DBs.
- Practice: RAG or semantic-search indexes over tens of millions to billions of embeddings that do not fit in RAM as float32. Also image or product similarity search, and dedup at scale. PQ often sits behind HNSW or IVF as the storage format.
- Interviews: "how does FAISS fit a billion vectors on one machine", "PQ vs scalar quantization vs LSH".
- Not for: corpora under ~1M vectors, where a flat or HNSW index over float32/float16 fits and is exact or nearly so ([[vector-databases]]). Not for when recall must be exact without a rerank stage. For cheaper but coarser compression, binary or 1-bit quantization with Hamming distance also works ([[locality-sensitive-hashing-from-scratch]]).

## Setup & code
NumPy only. Runs in about 12 seconds, mostly in k-means training (`m` codebooks × 20 iterations).

```python
import numpy as np

rng = np.random.default_rng(0)
d, n_base, n_query = 64, 20000, 200
proj = rng.normal(size=(16, d))                              # real embeddings have low intrinsic dimension
make = lambda n: (rng.normal(size=(n, 16)) @ proj + 0.3 * rng.normal(size=(n, d))).astype(np.float32)
base, queries = make(n_base), make(n_query)


def kmeans(x, k, iters=20):
    c = x[rng.choice(len(x), k, replace=False)].copy()
    for _ in range(iters):
        dist = (x ** 2).sum(1)[:, None] - 2 * x @ c.T + (c ** 2).sum(1)[None]
        a = dist.argmin(1)
        for j in range(k):
            pts = x[a == j]
            if len(pts):
                c[j] = pts.mean(0)
    return c


def pq_train(x, m, k=256):
    """One k-means codebook per subspace: m codebooks of shape (k, d/m)."""
    return [kmeans(s, k) for s in np.split(x, m, axis=1)]


def pq_encode(x, books):
    codes = []
    for s, c in zip(np.split(x, len(books), axis=1), books):
        dist = (s ** 2).sum(1)[:, None] - 2 * s @ c.T + (c ** 2).sum(1)[None]
        codes.append(dist.argmin(1))
    return np.stack(codes, 1).astype(np.uint8)              # n x m bytes


def pq_search(q, codes, books, k=10):
    """Asymmetric distance: exact query vs quantized base, via m lookup tables."""
    tables = [((c - s) ** 2).sum(1) for s, c in zip(np.split(q, len(books)), books)]  # m x 256
    dist = sum(t[codes[:, j]] for j, t in enumerate(tables))
    return np.argsort(dist)[:k]


true_nn = [np.argsort(((base - q) ** 2).sum(1))[:10] for q in queries]
train = base[rng.choice(n_base, 5000, replace=False)]
print(f"float32 index: {base.nbytes / 1e6:.2f} MB")
for m in [4, 8, 16]:
    books = pq_train(train, m)
    codes = pq_encode(base, books)
    recall1 = np.mean([nn[0] in pq_search(q, codes, books, 10) for q, nn in zip(queries, true_nn)])
    recall10 = np.mean([len(set(nn) & set(pq_search(q, codes, books, 10))) / 10 for q, nn in zip(queries, true_nn)])
    rerank = np.mean([len(set(nn) & set(cand[np.argsort(((base[cand] - q) ** 2).sum(1))[:10]])) / 10
                      for q, nn in zip(queries, true_nn) for cand in [pq_search(q, codes, books, 100)]])
    print(f"m={m:>2}: codes {codes.nbytes / 1e6:.2f} MB ({base.nbytes // codes.nbytes}x smaller)  "
          f"1-recall@10 {recall1:.3f}  10-recall@10 {recall10:.3f}  +exact rerank of top-100 {rerank:.3f}")
```

Output (Python 3.14, NumPy 2.5):
```
float32 index: 5.12 MB
m= 4: codes 0.08 MB (64x smaller)  1-recall@10 0.480  10-recall@10 0.285  +exact rerank of top-100 0.802
m= 8: codes 0.16 MB (32x smaller)  1-recall@10 0.815  10-recall@10 0.487  +exact rerank of top-100 0.977
m=16: codes 0.32 MB (16x smaller)  1-recall@10 0.985  10-recall@10 0.704  +exact rerank of top-100 1.000
```

`1-recall@10` (is the single true nearest neighbour in the returned 10?) is the metric FAISS reports, and it is far more forgiving than `10-recall@10` (how many of the true top-10 were returned). At `m = 16`, PQ almost always finds the nearest neighbour but orders the next few loosely: 0.70 of the true top-10. Fetching 100 candidates and reranking them exactly closes the gap completely. The reranked column is the number to compare with an exact index. Each extra byte per vector buys accuracy, and the rerank depth `R` is the cheap second knob.

The data here has 16 intrinsic dimensions plus noise, roughly how real embeddings behave. An earlier run used 50 Gaussian blobs with unit noise in all 64 dims, so there was no low-dimensional structure inside a blob and every neighbour was about equally far. On that data, the same code at `m = 8` got only 0.21 1-recall@10 (0.63 after rerank). PQ works when there is structure to quantize.

## Choosing / trade-offs
- **`m` (bytes per vector).** Memory grows linearly with `m`, and so does table-lookup time. `d/m` of 4-8 dims per subspace is a common sweet spot, and `d` must divide by `m`. FAISS's fast PQ kernels prefer specific `m` values (and 4-bit "fast scan" variants); check the index docs.
- **PQ vs scalar quantization (SQ8 / float16).** SQ8 keeps 1 byte per dimension (4x compression), needs no training, and loses very little recall. PQ reaches 16-64x but needs training data and a rerank. Use SQ when 4x is enough.
- **PQ vs binary quantization.** 1 bit per dimension (32x) with Hamming distance is extremely fast on SIMD. It works well for some modern embedding models, especially with a rerank, and badly for others. PQ is usually more accurate at the same byte budget.
- **Flat PQ vs IVF-PQ vs HNSW+PQ.** Flat PQ (this code) still scans every code, which is O(N) but cheap per item. IVF-PQ scans only `nprobe` lists. HNSW over PQ codes trades graph memory for latency. Use a library (FAISS `index_factory("IVF4096,PQ16")`, Milvus, Qdrant/Weaviate PQ options) rather than this code in production.
- **OPQ.** Learning a rotation before splitting balances variance across subspaces. It helps when dimensions are correlated or have uneven variance, at the cost of an extra training step.

## Gotchas
- Train codebooks on a sample from the same embedding model and domain you will index. Codebooks trained on other data, or before a model upgrade, silently tank recall. Re-train and re-encode when the embedding model changes.
- You need roughly `≥ 30-100 × 256` training vectors per codebook, so at least about 10k. Training on a few hundred leaves empty or degenerate centroids.
- Normalize the way your metric needs. For cosine similarity, L2-normalize before training and encoding; L2 on unit vectors then ranks like cosine. For inner product, use the IP variant of the tables.
- Report the recall metric you actually care about. `1-recall@10` can look excellent while `10-recall@10`, which drives RAG context quality, is mediocre, as above.
- PQ distances are biased low and noisy. Never threshold on raw PQ distances (for example "similarity > 0.8"); rerank exactly first.
- Keep the original vectors (on disk, or a float16 copy) if you plan to rerank. Throwing them away makes the accuracy loss permanent.

## Related
- [[vector-databases]] - where PQ shows up as an index option, and when HNSW/flat is enough.
- [[k-means-from-scratch]] - the clustering step each subspace codebook uses.
- [[locality-sensitive-hashing-from-scratch]] - the other classic way to compress and bucket vectors.
- [[knn-from-scratch]] - exact nearest neighbours, the baseline PQ approximates.
- [[embeddings]] - what is being compressed and why its intrinsic dimension matters.
- [[quantization]] - the same compression idea applied to model weights.

## References
- Jégou, Douze and Schmid (2011), "Product Quantization for Nearest Neighbor Search", IEEE TPAMI: https://doi.org/10.1109/TPAMI.2010.57
- FAISS wiki, index types and the index factory: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
- FAISS documentation: https://faiss.ai/
