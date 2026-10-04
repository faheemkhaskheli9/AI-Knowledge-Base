---
title: Non-negative matrix factorization (NMF) from scratch (multiplicative updates, parts-based topics)
category: ml
tags: [nmf, non-negative-matrix-factorization, matrix-factorization, topic-modeling, dimensionality-reduction, multiplicative-updates, numpy, from-scratch, ml-basics]
use_cases:
  - "factor a non-negative matrix (counts, TF-IDF, spectra, pixel intensities) into additive parts"
  - "extract interpretable topics from a document-term matrix without LDA"
  - "see why NMF components are interpretable and PCA components are not"
  - "explain NMF and Lee-Seung multiplicative updates in an interview"
status: draft
last_verified: 2026-10-04
sources:
  - https://doi.org/10.1038/44565
  - https://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization
  - https://scikit-learn.org/stable/modules/decomposition.html#non-negative-matrix-factorization-nmf-or-nnmf
---

# Non-negative matrix factorization (NMF) from scratch (multiplicative updates, parts-based topics)

## Summary
NMF approximates a non-negative matrix `X (n × p)` as `W H`, with `W (n × k) ≥ 0` and `H (k × p) ≥ 0`. Because nothing can be subtracted, every row of `X` is built by adding up parts. In a document-term matrix, the rows of `H` become topics (weighted word lists) and `W` gives each document's topic mix. Lee and Seung's multiplicative updates make this about ten lines of NumPy. Below, 300 Poisson-sampled documents over 16 words are generated from 4 hidden topics. NMF recovers all 4 topics with cosine similarity 1.00 to the truth, and the relative reconstruction error falls from 0.72 to 0.12 within 50 iterations. PCA on the same matrix gives a first component that contrasts "politics" against "markets" and "genes", with 66% of its top-4 loadings negative, so it does not read as a topic.

## Key concepts
- **Objective.** Minimize `||X − WH||²_F` subject to `W, H ≥ 0`. The problem is non-convex, so the result depends on initialization, but each step is easy.
- **Multiplicative updates.** `H ← H ∘ (WᵀX) / (WᵀWH)`, then `W ← W ∘ (XHᵀ) / (WHHᵀ)`. These are gradient steps with a per-element step size chosen so the update becomes a ratio. Entries that start positive stay non-negative, and the loss never increases.
- **Parts-based.** With only additions allowed, components tend to be sparse, localized pieces: words of one topic, facial features, or the spectrum of one chemical. PCA's orthogonal, signed components describe contrasts instead.
- **Rank `k`.** Choose it like a number of clusters: look at the reconstruction-error elbow, topic coherence, and whether the components make sense to a person.
- **Scale ambiguity.** `WH = (WD)(D⁻¹H)` for any positive diagonal `D`. Normalize the rows of `H` (or columns of `W`) before comparing or reporting them.

## When to use / scenarios
- Learning: a minimal example of constrained matrix factorization, and how a constraint creates interpretability.
- Practice: topic extraction on TF-IDF from support tickets, reviews or survey answers. NMF on TF-IDF is a fast, strong baseline that often reads cleaner than LDA on short texts. Also used for spectral unmixing (chemometrics, hyperspectral imaging), audio source separation on magnitude spectrograms, gene-expression signatures, and recommendations from implicit feedback such as play counts.
- Interviews: "NMF vs PCA vs LDA", "why are NMF components interpretable".
- Not for: data with meaningful negative values (use PCA or ICA, see [[pca-from-scratch]] and [[fastica-from-scratch]]); when you need a probabilistic generative model with priors (use LDA); or semantic topics on long or nuanced text, where embedding-based clustering (BERTopic-style, [[topic-modeling]]) usually wins.

## Setup & code
NumPy only. Runs in under a second.

```python
import numpy as np

rng = np.random.default_rng(0)
vocab = ["goal", "match", "league", "coach", "vote", "election", "senate", "party",
         "stock", "market", "shares", "profit", "gene", "cell", "protein", "dna"]
n_docs, n_topics = 300, 4
H_true = np.zeros((n_topics, len(vocab)))
for t in range(n_topics):
    H_true[t, 4 * t:4 * t + 4] = rng.uniform(1, 3, 4)       # each topic owns 4 words
H_true += 0.05                                               # small background rate
W_true = rng.dirichlet(np.full(n_topics, 0.3), n_docs) * 40  # docs mix a few topics
X = rng.poisson(W_true @ H_true).astype(float)               # doc x word count matrix


def nmf(X, k, iters=500, eps=1e-10):
    """Lee-Seung multiplicative updates for min ||X - WH||_F, W, H >= 0."""
    W = rng.uniform(0.1, 1, (X.shape[0], k))
    H = rng.uniform(0.1, 1, (k, X.shape[1]))
    errs = []
    for i in range(iters):
        H *= (W.T @ X) / (W.T @ W @ H + eps)
        W *= (X @ H.T) / (W @ H @ H.T + eps)
        if i in (0, 9, 49, 499):
            errs.append((i + 1, np.linalg.norm(X - W @ H) / np.linalg.norm(X)))
    return W, H, errs


W, H, errs = nmf(X, n_topics)
print("relative error ||X-WH||/||X|| by iteration:", ", ".join(f"{i}: {e:.3f}" for i, e in errs))
print("\nNMF topics (top 4 words):")
for h in H:
    print("  ", [vocab[j] for j in np.argsort(h)[::-1][:4]])
sim = (H / np.linalg.norm(H, axis=1, keepdims=True)) @ (H_true / np.linalg.norm(H_true, axis=1, keepdims=True)).T
print("best cosine match of each NMF topic to a true topic:", np.round(sim.max(1), 3))

Xc = X - X.mean(0)
_, _, Vt = np.linalg.svd(Xc, full_matrices=False)
print("\nPCA component 1 (signed loadings, not a topic):")
print("  ", {vocab[j]: round(float(v), 2) for j, v in enumerate(Vt[0]) if abs(v) > 0.15})
print(f"negative entries: PCA top-4 components {np.mean(Vt[:4] < 0):.0%}, NMF H {np.mean(H < 0):.0%}")
```

Output (Python 3.14, NumPy 2.5):
```
relative error ||X-WH||/||X|| by iteration: 1: 0.718, 10: 0.164, 50: 0.120, 500: 0.120

NMF topics (top 4 words):
   ['gene', 'protein', 'dna', 'cell']
   ['election', 'vote', 'party', 'senate']
   ['goal', 'match', 'league', 'coach']
   ['market', 'shares', 'stock', 'profit']
best cosine match of each NMF topic to a true topic: [1. 1. 1. 1.]

PCA component 1 (signed loadings, not a topic):
   {'vote': 0.47, 'election': 0.5, 'senate': 0.37, 'party': 0.43, 'market': -0.19, 'shares': -0.18, 'gene': -0.2, 'protein': -0.18}
negative entries: PCA top-4 components 66%, NMF H 0%
```

NMF converges in about 50 iterations. The remaining 0.12 error is Poisson sampling noise, which no rank-4 model can explain. The recovered topics come out in arbitrary order, which is the usual permutation ambiguity, but each one matches a true topic exactly. A rank-4 SVD reconstructs `X` at least as well, since it is optimal in squared error (Eckart-Young), but its axes are signed contrasts. Component 1 reads as "politics, and not markets or genes", which is useful for variance but not as a list of themes.

## Choosing / trade-offs
- **NMF vs PCA/SVD.** PCA is optimal in squared error, unique, and fast, but its components are hard to interpret. NMF gives up a little error and uniqueness for additive, readable parts. Use PCA for compression and preprocessing, NMF for explanation.
- **NMF vs LDA.** LDA is a probabilistic model with Dirichlet priors and gives proper topic distributions. NMF on TF-IDF is faster, deterministic given a seed, and often just as coherent. Try both on a sample and read the topics.
- **Loss.** Frobenius (here) suits roughly Gaussian noise. KL or Itakura-Saito divergence (`beta_loss` in scikit-learn) better fits counts and audio power spectra.
- **Solver.** Multiplicative updates are simple but can stall. Coordinate descent (scikit-learn's default `solver="cd"`) or HALS converges faster. Add L1 or L2 penalties (`alpha_W`, `alpha_H`, `l1_ratio`) for sparser or smoother factors.
- **Library.** `sklearn.decomposition.NMF(n_components=k, init="nndsvda")` for production. Its `nndsvd` initialization is SVD-based and more stable than random starts.

## Gotchas
- Initialize with strictly positive values. A zero entry in `W` or `H` stays zero forever under multiplicative updates. Keep the `eps` in the denominators.
- Results vary with the seed: run a few restarts and keep the lowest error, or use `nndsvd` init. Topic order also changes between runs, so match topics by cosine similarity, not by index.
- Preprocessing dominates topic quality. Remove stop words and extremely common terms, and use TF-IDF or length-normalized counts, or the longest documents and most frequent words will dominate the factors.
- Picking `k` by reconstruction error alone always favours a larger `k`. Also check topic coherence and whether topics duplicate each other.
- `X` must be non-negative, and so must any imputed values. Centering or standardizing the data first breaks NMF, because it makes values negative.
- Projecting new documents means solving for `W` with `H` fixed (`model.transform` in scikit-learn), not refitting. Refitting reshuffles the topics.

## Related
- [[topic-modeling]] - LDA, BERTopic and when NMF fits among them.
- [[pca-from-scratch]] - the unconstrained, signed factorization NMF is compared with here.
- [[matrix-factorization-from-scratch]] - the recommender cousin (latent factors from ratings).
- [[fastica-from-scratch]] - another way to get interpretable sources, via independence instead of non-negativity.
- [[dimensionality-reduction]] - where NMF sits among reduction methods.

## References
- Lee and Seung (1999), "Learning the parts of objects by non-negative matrix factorization", Nature: https://doi.org/10.1038/44565
- Lee and Seung (2000), "Algorithms for Non-negative Matrix Factorization", NeurIPS: https://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization
- scikit-learn user guide, NMF: https://scikit-learn.org/stable/modules/decomposition.html#non-negative-matrix-factorization-nmf-or-nnmf
