---
title: Locality-sensitive hashing from scratch (random hyperplanes for cosine ANN search)
category: ml
tags: [lsh, locality-sensitive-hashing, approximate-nearest-neighbors, ann, simhash, random-projection, cosine-similarity, vector-search, embeddings, numpy, from-scratch]
use_cases:
  - "implement random-hyperplane LSH (SimHash) for approximate cosine nearest-neighbour search in NumPy"
  - "see the recall vs candidates-scanned trade-off controlled by bits per table (K) and number of tables (L)"
  - "understand how vector databases avoid scanning every embedding before reaching for HNSW or IVF"
  - "explain the LSH collision probability and the K/L amplification trick in an interview"
status: draft
last_verified: 2026-10-04
sources:
  - https://www.cs.princeton.edu/courses/archive/spr04/cos598B/bib/CharikarEstim.pdf
  - http://www.mit.edu/~andoni/LSH/
  - http://infolab.stanford.edu/~ullman/mmds/ch3n.pdf
  - https://github.com/facebookresearch/faiss/wiki
---

# Locality-sensitive hashing from scratch (random hyperplanes for cosine ANN search)

## Summary
Exact nearest-neighbour search compares the query with every stored vector. Locality-sensitive hashing (LSH) replaces that scan with hash lookups designed so that similar vectors tend to collide. For cosine similarity, the hash is random-hyperplane LSH (Charikar 2002, also called SimHash): draw a random Gaussian vector `r` and record `sign(r · x)`. Two vectors at angle `φ` land on the same side with probability `1 − φ/π`. Concatenating K such bits makes a bucket key that only close vectors share, and building L independent tables gives a close neighbour L chances to collide. The candidates found are then re-ranked exactly. On 50,000 clustered 128-d vectors this implementation reaches 95.4% recall@10 while scanning 3.9% of the data, at 1.3 ms per query vs 3.9 ms for an exact NumPy scan. With fewer tables it is far faster and far less accurate (63% recall at 0.7% scanned). The measured bit agreement (0.525) matches the `1 − φ/π` formula (0.525).

## Key concepts
- **LSH family.** A hash family is locality-sensitive if close points collide with probability `p1` and far points with a lower probability `p2`. Random hyperplanes are such a family for angular distance: `P[h(x) = h(y)] = 1 − θ(x, y)/π`.
- **AND (K bits per table).** A key of K bits collides with probability `p^K`. Larger K shrinks buckets, so fewer false candidates, but also fewer true neighbours.
- **OR (L tables).** A neighbour is found if it collides in any table: probability `1 − (1 − p^K)^L`. Larger L restores recall at the cost of memory and candidates.
- **S-curve.** Together K and L turn the smooth `p` into a steep step: pairs above a similarity threshold almost always collide, pairs below almost never do. The threshold is roughly where `p^K ≈ 1/L`.
- **Re-ranking.** LSH only produces candidates. Exact similarity is computed on the candidates, so precision among returned items is perfect; the only error is a missed neighbour (recall).
- **Other families.** MinHash for Jaccard similarity of sets (near-duplicate documents), p-stable projections for Euclidean distance (E2LSH).

## When to use / scenarios
- Learning: the simplest sub-linear vector search, and the idea behind binary embeddings and SimHash near-duplicate detection.
- Streaming or distributed settings where items must be bucketed independently without building a global graph (LSH buckets shard trivially).
- Near-duplicate detection on text or web pages with MinHash or SimHash signatures.
- Interviews: "how do you search a billion embeddings", "what do K and L do", "LSH vs HNSW vs IVF-PQ".
- Not for: production vector search on a single machine. HNSW and IVF-PQ (FAISS, pgvector, a vector database) give better recall per millisecond and per byte; see [[vector-databases]].

## Setup & code
`pip install numpy`. About 20 seconds (index build dominates).

```python
import time
from collections import defaultdict

import numpy as np


class HyperplaneLSH:
    """Random-hyperplane LSH for cosine similarity (Charikar 2002).
    L tables, each hashing a vector to a K-bit signature sign(R x)."""

    def __init__(self, dim, K=12, L=16, seed=0):
        rng = np.random.default_rng(seed)
        self.R = rng.normal(size=(L, K, dim))
        self.pow2 = 1 << np.arange(K)
        self.tables = [defaultdict(list) for _ in range(L)]

    def _keys(self, X):
        bits = np.einsum("lkd,nd->nlk", self.R, X) > 0          # (n, L, K)
        return bits @ self.pow2                                  # (n, L) integer bucket ids

    def index(self, X):
        self.X = X / np.linalg.norm(X, axis=1, keepdims=True)
        for i, row in enumerate(self._keys(self.X)):
            for table, key in zip(self.tables, row):
                table[key].append(i)

    def query(self, q, k=10):
        q = q / np.linalg.norm(q)
        cand = set()
        for table, key in zip(self.tables, self._keys(q[None])[0]):
            cand.update(table.get(key, ()))
        cand = np.fromiter(cand, int)
        if len(cand) == 0:
            return cand, 0
        sims = self.X[cand] @ q                                 # exact re-rank of candidates only
        return cand[np.argsort(-sims)[:k]], len(cand)


# Clustered synthetic "embeddings": 200 topics, 50k vectors, 128-d.
rng = np.random.default_rng(1)
N, D = 50_000, 128
centers = rng.normal(size=(200, D))
X = centers[rng.integers(0, 200, N)] + 0.6 * rng.normal(size=(N, D))
Q = centers[rng.integers(0, 200, 200)] + 0.6 * rng.normal(size=(200, D))
Xn = X / np.linalg.norm(X, axis=1, keepdims=True)

t0 = time.perf_counter()
truth = [np.argsort(-(Xn @ (q / np.linalg.norm(q))))[:10] for q in Q]
t_exact = (time.perf_counter() - t0) / len(Q)

# Collision probability check: P[same bit] = 1 - angle/pi
a, b = Xn[0], Xn[1]
r = rng.normal(size=(100_000, D))
print(f"bit agreement {np.mean((r @ a > 0) == (r @ b > 0)):.3f} vs 1 - angle/pi "
      f"{1 - np.arccos(a @ b) / np.pi:.3f}")

for K, L in [(16, 8), (12, 16), (10, 32)]:
    lsh = HyperplaneLSH(D, K, L)
    lsh.index(X)
    t0 = time.perf_counter()
    rec, ncand = [], []
    for q, tr in zip(Q, truth):
        got, nc = lsh.query(q)
        rec.append(len(set(got) & set(tr)) / 10)
        ncand.append(nc)
    t_q = (time.perf_counter() - t0) / len(Q)
    print(f"K={K:2d} L={L:2d}: recall@10 {np.mean(rec):.3f} | candidates scanned "
          f"{np.mean(ncand) / N:5.1%} of {N} | {t_q * 1e3:.2f} ms/query vs exact {t_exact * 1e3:.2f} ms")
```

Output (numpy 2.5; timings from one laptop CPU run and will vary):
```
bit agreement 0.525 vs 1 - angle/pi 0.525
K=16 L= 8: recall@10 0.177 | candidates scanned  0.1% of 50000 | 0.05 ms/query vs exact 3.89 ms
K=12 L=16: recall@10 0.632 | candidates scanned  0.7% of 50000 | 0.22 ms/query vs exact 3.89 ms
K=10 L=32: recall@10 0.954 | candidates scanned  3.9% of 50000 | 1.31 ms/query vs exact 3.89 ms
```

The first line checks the theory: two unrelated vectors (angle about 85°) agree on 52.5% of 100,000 random hyperplanes, exactly `1 − φ/π`. The three configurations then trace the trade-off. Sixteen bits per key with 8 tables makes tiny buckets (0.1% of the data scanned, 80× faster than exact) but finds only 18% of the true top-10. Ten bits with 32 tables finds 95% of them while scanning 4% of the data, about 3× faster than the exact scan in this Python implementation. The exact baseline here is a single vectorised matrix-vector product, which is hard to beat at 50,000 vectors; LSH's advantage grows with N, because the exact scan is linear while bucket sizes stay roughly constant if K grows with `log N`.

## Choosing / trade-offs
- **K and L.** Pick K so that buckets hold a manageable number of points, then raise L until recall is acceptable. Memory grows linearly with L (each table stores every id), which is LSH's main cost.
- **Multi-probe LSH.** Also probe buckets whose key differs in the least-confident bits. It reaches the same recall with far fewer tables.
- **LSH vs HNSW.** HNSW (graph-based) usually needs much less memory and gives higher recall at the same latency; it is the default in most vector databases. LSH is simpler, has provable guarantees, inserts in O(L), and shards without coordination.
- **LSH vs IVF-PQ.** IVF clusters the data (k-means) and scans the nearest clusters; PQ compresses vectors. They dominate on billion-scale, memory-bound search (FAISS).
- **Binary codes as embeddings.** Storing the bits themselves (sign of random or learned projections) and ranking by Hamming distance compresses vectors 32× from float32. Many vector databases now offer binary quantisation as an option.

## Gotchas
- Normalise vectors if you want cosine similarity; random-hyperplane hashing ignores length, but the exact re-rank does not.
- Mean-centre embeddings that share a large common component. If every vector points in a similar direction, most hyperplanes put all of them on the same side and buckets become huge.
- Empty candidate sets happen for outlying queries; fall back to a bigger probe or an exact scan rather than returning nothing.
- Recall depends on the data distribution. Clustered real embeddings behave very differently from uniform random vectors, so measure on your data, never on a synthetic benchmark like this one.
- Python dict-of-lists tables are slow and memory-hungry; real implementations use sorted arrays of keys or integer hash tables.
- Benchmark against an optimised exact baseline (one BLAS matrix product, or FAISS `IndexFlatIP`). Brute force is fast up to surprisingly large N.

## Related
- [[vector-databases]] - HNSW, IVF and the managed stores that use them.
- [[knn-from-scratch]] - exact nearest neighbours, the baseline LSH approximates.
- [[distance-metrics-and-similarity]] - cosine, Euclidean and Jaccard, and which hash family fits each.
- [[embeddings]] - where the vectors come from.
- [[curse-of-dimensionality]] - why exact space-partitioning trees fail in high dimensions and hashing does not.
- [[label-propagation-from-scratch]] - kNN graphs at scale need approximate neighbours.

## References
- Charikar (2002), "Similarity Estimation Techniques from Rounding Algorithms", STOC: https://www.cs.princeton.edu/courses/archive/spr04/cos598B/bib/CharikarEstim.pdf
- Andoni and Indyk, LSH algorithm and implementation (E2LSH) page: http://www.mit.edu/~andoni/LSH/
- Leskovec, Rajaraman and Ullman, Mining of Massive Datasets, Chapter 3 "Finding Similar Items": http://infolab.stanford.edu/~ullman/mmds/ch3n.pdf
- FAISS wiki (index types, including `IndexLSH` and HNSW): https://github.com/facebookresearch/faiss/wiki
