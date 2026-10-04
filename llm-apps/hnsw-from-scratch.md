---
title: HNSW vector index from scratch (layered graph, greedy search, ef, neighbour-selection heuristic)
category: llm-apps
tags: [hnsw, approximate-nearest-neighbor, ann, vector-search, vector-database, graph-index, recall, ef-search, rag, numpy, from-scratch]
use_cases:
  - "understand what M, ef_construction and ef_search do before tuning a vector database"
  - "trade recall against latency for semantic search or RAG retrieval"
  - "explain why HNSW recall drops on clustered embeddings and how the neighbour heuristic fixes it"
  - "measure recall@k of an ANN index against brute-force search on my own data"
status: draft
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/1603.09320
  - https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
  - https://github.com/facebookresearch/faiss/wiki/Indexing-1M-vectors
---

# HNSW vector index from scratch (layered graph, greedy search, ef, neighbour-selection heuristic)

## Summary
Hierarchical Navigable Small World (HNSW) is the graph index behind most vector databases (pgvector, Qdrant, Weaviate, Milvus, Elasticsearch/OpenSearch, FAISS `IndexHNSWFlat`, hnswlib). Every vector is a node linked to about `M` near neighbours. A few nodes are also copied into sparser upper layers. A query enters at the top, walks greedily toward the target on each layer, then runs a best-first search with a candidate list of size `ef` on the bottom layer. This file builds the index in about 100 lines of Python, measures recall@10 against brute force, and shows that the paper's neighbour-selection heuristic, not the layers, is what keeps recall high on clustered data like real embeddings.

## Key concepts
- **Layers.** Each node gets a level `floor(−ln(U) · mL)` with `mL = 1/ln(M)`, so about `1/M` of the nodes reach layer 1, `1/M²` layer 2, and so on. Upper layers are a coarse road network for getting close fast.
- **Greedy search with `ef`.** Keep a min-heap of candidates to expand and a max-heap of the best `ef` results. Stop when the closest unexpanded candidate is farther than the worst result. `ef` is the beam width: higher means better recall and more distance computations.
- **`M` and `M0`.** Each node keeps up to `M` links on upper layers and `2M` on layer 0. Memory grows with `M`. Recall and build time grow with it too.
- **`ef_construction`.** The beam width used while inserting. It decides how good the candidate neighbour list is for each new node. It matters once, at build time.
- **Neighbour-selection heuristic.** Instead of linking to the `M` nearest candidates, keep a candidate only if it is closer to the new node than to any neighbour already kept. Links then point in diverse directions, including across to neighbouring clusters, rather than all into the node's own tight cluster.

## When to use / scenarios
- Semantic search and RAG retrieval over 10⁴ to 10⁹ vectors with sub-10 ms latency and recall above 0.95.
- When vectors and the graph fit in RAM. HNSW is fast because every hop is a random memory access. For larger-than-RAM corpora, use IVF-PQ or a disk-based graph (DiskANN). See [[product-quantization-from-scratch]].
- Not for a few thousand vectors: brute force with one matrix multiply is exact and fast enough.
- Not when the index changes constantly with deletes. HNSW handles inserts well, but deletes are usually soft (tombstones) and degrade the graph until a rebuild.

## Setup & code
NumPy only. Builds a 5,000-vector, 64-dimensional index in about 5 s (pure Python loops; real libraries are C++ and SIMD), then compares the heuristic with plain nearest-`M` linking.

```python
import heapq
import time
import numpy as np


class HNSW:
    def __init__(self, dim, M=16, ef_construction=100, simple=False, seed=0):
        self.M, self.M0, self.efc, self.simple = M, 2 * M, ef_construction, simple
        self.mL = 1 / np.log(M)                          # level multiplier from the paper
        self.rng = np.random.default_rng(seed)
        self.X = np.empty((0, dim), dtype=np.float32)
        self.graph = []                                  # graph[layer][node] -> list of neighbours
        self.entry, self.top = None, -1
        self.dist_count = 0

    def _d(self, q, ids):
        self.dist_count += len(ids)
        return ((self.X[ids] - q) ** 2).sum(1)

    def _search_layer(self, q, entry, ef, layer):
        """Greedy best-first search keeping the ef closest nodes found so far."""
        d0 = self._d(q, [entry])[0]
        visited = {entry}
        cand = [(d0, entry)]                             # min-heap: closest unexpanded node first
        best = [(-d0, entry)]                            # max-heap: worst of the current ef results on top
        while cand:
            d, c = heapq.heappop(cand)
            if d > -best[0][0]:                          # closest candidate is worse than the worst result: stop
                break
            nbrs = [n for n in self.graph[layer][c] if n not in visited]
            visited.update(nbrs)
            if not nbrs:
                continue
            for dn, n in zip(self._d(q, nbrs), nbrs):
                if len(best) < ef or dn < -best[0][0]:
                    heapq.heappush(cand, (dn, n))
                    heapq.heappush(best, (-dn, n))
                    if len(best) > ef:
                        heapq.heappop(best)
        return sorted((-d, n) for d, n in best)

    def _select(self, q, cands, M):
        """Neighbour-selection heuristic: keep a candidate only if it is closer to q than to every kept one."""
        if self.simple:
            return [c for _, c in cands[:M]]
        kept = []
        for d, c in cands:
            if len(kept) == M:
                break
            if all(((self.X[c] - self.X[k]) ** 2).sum() > d for k in kept):
                kept.append(c)
        return kept

    def add(self, x):
        i = len(self.X)
        self.X = np.vstack([self.X, x[None].astype(np.float32)])
        level = int(-np.log(self.rng.random()) * self.mL)   # exponentially rarer higher layers
        while len(self.graph) <= level:
            self.graph.append({})
        for l in range(level + 1):
            self.graph[l][i] = []
        if self.entry is None:
            self.entry, self.top = i, level
            return
        ep = self.entry
        for l in range(self.top, level, -1):            # descend greedily (ef=1) through upper layers
            ep = self._search_layer(x, ep, 1, l)[0][1]
        for l in range(min(level, self.top), -1, -1):
            cands = self._search_layer(x, ep, self.efc, l)
            Mmax = self.M0 if l == 0 else self.M
            self.graph[l][i] = self._select(x, cands, self.M)
            for n in self.graph[l][i]:                   # bidirectional links, pruned back to Mmax
                self.graph[l][n].append(i)
                if len(self.graph[l][n]) > Mmax:
                    nb = self.graph[l][n]
                    d = ((self.X[nb] - self.X[n]) ** 2).sum(1)
                    self.graph[l][n] = self._select(self.X[n], sorted(zip(d, nb)), Mmax)
            ep = cands[0][1]
        if level > self.top:
            self.entry, self.top = i, level

    def search(self, q, k=10, ef=50):
        ep = self.entry
        for l in range(self.top, 0, -1):
            ep = self._search_layer(q, ep, 1, l)[0][1]
        return [n for _, n in self._search_layer(q, ep, max(ef, k), 0)[:k]]


rng = np.random.default_rng(0)
N, D, Q, k = 5000, 64, 200, 10
centers = rng.normal(size=(100, D)) * 0.5             # 100 tight, overlapping clusters
X = (centers[rng.integers(0, 100, N)] + rng.normal(size=(N, D)) * 0.3).astype(np.float32)
queries = (centers[rng.integers(0, 100, Q)] + rng.normal(size=(Q, D)) * 0.3).astype(np.float32)
truth = [set(np.argsort(((X - q) ** 2).sum(1))[:k]) for q in queries]   # exact brute-force top-k

for simple in [False, True]:
    t = time.perf_counter()
    index = HNSW(D, M=8, ef_construction=64, simple=simple)
    for x in X:
        index.add(x)
    print(f"{'simple nearest-M' if simple else 'heuristic'} links: built in {time.perf_counter() - t:.1f}s, "
          f"nodes per layer {[len(g) for g in index.graph]}")
    for ef in [10, 20, 50, 100]:
        index.dist_count = 0
        recall = np.mean([len(truth[j] & set(index.search(q, k, ef))) / k for j, q in enumerate(queries)])
        print(f"  ef={ef:3d}  recall@{k} {recall:.3f}  distance evals/query {index.dist_count / Q:5.0f}"
              f"  ({index.dist_count / Q / N:.1%} of brute force)")
```

Output (Python 3.14, NumPy 2.5):
```
heuristic links: built in 5.4s, nodes per layer [5000, 651, 88, 13, 2]
  ef= 10  recall@10 0.942  distance evals/query   126  (2.5% of brute force)
  ef= 20  recall@10 0.996  distance evals/query   160  (3.2% of brute force)
  ef= 50  recall@10 1.000  distance evals/query   249  (5.0% of brute force)
  ef=100  recall@10 1.000  distance evals/query   571  (11.4% of brute force)
simple nearest-M links: built in 3.4s, nodes per layer [5000, 651, 88, 13, 2]
  ef= 10  recall@10 0.587  distance evals/query    96  (1.9% of brute force)
  ef= 20  recall@10 0.645  distance evals/query   117  (2.3% of brute force)
  ef= 50  recall@10 0.794  distance evals/query   188  (3.8% of brute force)
  ef=100  recall@10 0.872  distance evals/query   368  (7.4% of brute force)
```

With the heuristic, `ef=20` finds 99.6% of the true top-10 while computing distances to 3.2% of the corpus. The layer sizes follow the `1/M` rule: 651 of 5,000 nodes on layer 1 with `M=8`. With plain nearest-`M` links on the same data, recall stays at 0.59 to 0.87 even with 5 to 10 times the beam width. Each cluster becomes an island whose links all point inward, and the search cannot leave the cluster it entered. Both variants have identical layers, so the layers are not what fails. Uniform random data hides this effect, which is why a benchmark on your own embeddings matters more than published numbers.

## Choosing / trade-offs
- **`M`.** hnswlib's guidance: 12 to 48 suits most use cases, and memory is roughly `M × 8–10` bytes per element on top of the vectors. Raise `M` for high-dimensional or hard data; lower it to save memory.
- **`ef_construction`.** Higher builds a better graph more slowly, with diminishing returns. hnswlib's check: with `ef = ef_construction`, recall for `M` nearest neighbours should exceed 0.9.
- **`ef` (search).** The runtime knob: must be at least `k`. Sweep it on a held-out query set and pick the smallest value that meets your recall target. It can differ per query.
- **HNSW vs IVF / PQ.** HNSW gives the best recall/latency in RAM but stores full vectors plus links. IVF-PQ compresses vectors 10 to 50 times for billion-scale corpora at some recall cost. Many systems combine them (HNSW as the IVF coarse quantiser, or HNSW over PQ codes). See [[vector-databases]].
- **HNSW vs LSH.** LSH has guarantees and cheap inserts but needs far more candidates for the same recall on real embeddings. See [[locality-sensitive-hashing-from-scratch]].

## Gotchas
- Build time is the cost people forget: inserting is a search per node, so a 10M-vector index takes minutes to hours. Build in parallel and snapshot the index; do not rebuild on every deploy.
- Filtered search (metadata `WHERE` + vector) can wreck recall if the filter is applied after the graph walk. Check how your database handles restrictive filters (pre-filtering, filtered traversal, or fallback to brute force).
- The metric must match the embedding model. For cosine similarity, normalise vectors and use inner product or L2; mixing them silently returns different neighbours.
- Deletes are tombstones in most implementations. Heavy churn leaves dead nodes the search still walks through; rebuild or compact periodically.
- Measure recall on your own data and queries against exact search. Recall on in-distribution queries can be much higher than on the rare queries users actually care about.

## Related
- [[vector-databases]] - which database to use and how they expose HNSW parameters.
- [[product-quantization-from-scratch]] - compressing vectors for corpora too big for RAM.
- [[locality-sensitive-hashing-from-scratch]] - the hashing alternative to graph indexes.
- [[distance-metrics-and-similarity]] - cosine vs inner product vs L2.
- [[embeddings]] - where the vectors come from.
- [[rag-basics]] - the retrieval step HNSW speeds up.

## References
- Malkov & Yashunin (2016), "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs": https://arxiv.org/abs/1603.09320
- hnswlib, "HNSW algorithm parameters": https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
- FAISS wiki, "Indexing 1M vectors": https://github.com/facebookresearch/faiss/wiki/Indexing-1M-vectors
