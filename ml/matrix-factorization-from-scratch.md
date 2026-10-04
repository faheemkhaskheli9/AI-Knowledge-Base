---
title: Matrix factorization recommender from scratch (ALS and SGD)
category: ml
tags: [matrix-factorization, collaborative-filtering, als, alternating-least-squares, funk-svd, sgd, recommender-systems, latent-factors, numpy, from-scratch, ml-basics]
use_cases:
  - "implement alternating least squares and SGD matrix factorization for explicit ratings in NumPy"
  - "predict missing user-item ratings and rank unseen items for a user"
  - "see why regularization and rank decide whether MF beats a global-mean baseline"
  - "explain ALS vs SGD, biases and cold start in a recommender-systems interview"
status: draft
last_verified: 2026-10-04
sources:
  - https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf
  - https://doi.org/10.1109/ICDM.2008.22
  - https://sifter.org/~simon/journal/20061211.html
---

# Matrix factorization recommender from scratch (ALS and SGD)

## Summary
Matrix factorization (MF) fills in a sparse user × item ratings matrix by assuming it is close to low rank: each user and each item gets a short vector of latent factors, and a predicted rating is their dot product plus the global mean. Alternating least squares (ALS) fixes one side and solves a small ridge regression for each row of the other, alternating. SGD updates both sides one rating at a time. On a synthetic 300 × 200 matrix with 10% of ratings observed, well-regularised ALS cuts test RMSE from 0.957 (predict the mean) to 0.519. Without regularisation it overfits to 1.294, worse than the baseline.

## Key concepts
- **Model.** `r̂_ui = μ + b_u + b_i + p_u · q_i`, with global mean `μ`, optional user/item biases, and `k`-dimensional factors `p_u`, `q_i`.
- **Objective.** Squared error on observed ratings only, plus `λ (‖p_u‖² + ‖q_i‖²)`. Missing entries are unknown, not zero; this is not an SVD of a zero-filled matrix.
- **ALS.** With `Q` fixed, each user's factors have a closed form `p_u = (Q_uᵀ Q_u + λ n_u I)⁻¹ Q_uᵀ r_u`, where `Q_u` holds the factors of items the user rated. Then swap roles. Every half-step is exact, so the loss never increases, and each user (item) solve is independent and parallel.
- **Weighted λ.** Scaling λ by the number of ratings `n_u` (Zhou et al., 2008) keeps heavy and light users regularised comparably.
- **SGD (Funk SVD).** For each observed rating, compute the error `e` and step `p_u += lr (e q_i − λ p_u)`, `q_i += lr (e p_u − λ q_i)`. Cheap per step and easy to extend (biases, implicit feedback, side features).
- **Ranking.** Recommend by scoring every unseen item for a user (`μ + Q p_u`) and taking the top N.

## When to use / scenarios
- Learning: low-rank structure, ridge regression inside an alternating scheme, and the gap between training and test error on sparse data.
- Interviews: "how does collaborative filtering work", "ALS vs SGD", "why regularise", "how do you handle cold start".
- Products: ratings prediction, "you may also like" lists, item embeddings for similarity search, a strong baseline before neural recommenders.
- Implicit feedback (clicks, plays, purchases): the same ALS with confidence weights (Hu, Koren and Volinsky, 2008), as in `implicit` or Spark MLlib.
- Not for: brand-new users or items with no interactions (cold start), or when content, context and sequence matter. Use content features, two-tower models or sequential recommenders there ([[recommender-systems]]).

## Setup & code
`pip install numpy`. ALS runs in a few seconds; the pure-Python SGD loop takes about a minute.

```python
import numpy as np


def make_ratings(n_users, n_items, k, density, rng):
    """Synthetic explicit ratings 1..5 from a true low-rank model plus noise."""
    U, V = rng.normal(0, 1, (n_users, k)), rng.normal(0, 1, (n_items, k))
    full = 3 + U @ V.T / np.sqrt(k) + rng.normal(0, 0.3, (n_users, n_items))
    mask = rng.random((n_users, n_items)) < density
    u, i = np.nonzero(mask)
    return u, i, np.clip(full[u, i], 1, 5)


def als(u, i, r, n_users, n_items, k, lam, iters, rng):
    """Alternating least squares with biases folded out via the global mean."""
    mu = r.mean()
    P, Q = rng.normal(0, 0.1, (n_users, k)), rng.normal(0, 0.1, (n_items, k))
    by_user = [np.nonzero(u == a)[0] for a in range(n_users)]
    by_item = [np.nonzero(i == b)[0] for b in range(n_items)]
    I = lam * np.eye(k)
    for _ in range(iters):
        for a, idx in enumerate(by_user):                 # fix Q, each user is a ridge regression
            Qa = Q[i[idx]]
            P[a] = np.linalg.solve(Qa.T @ Qa + I * max(len(idx), 1), Qa.T @ (r[idx] - mu))
        for b, idx in enumerate(by_item):                 # fix P, each item is a ridge regression
            Pb = P[u[idx]]
            Q[b] = np.linalg.solve(Pb.T @ Pb + I * max(len(idx), 1), Pb.T @ (r[idx] - mu))
    return mu, P, Q


def sgd_mf(u, i, r, n_users, n_items, k, lam, lr, epochs, rng):
    """Funk-style SGD with user/item biases."""
    mu = r.mean()
    P, Q = rng.normal(0, 0.1, (n_users, k)), rng.normal(0, 0.1, (n_items, k))
    bu, bi = np.zeros(n_users), np.zeros(n_items)
    for _ in range(epochs):
        for n in rng.permutation(len(r)):
            a, b = u[n], i[n]
            e = r[n] - (mu + bu[a] + bi[b] + P[a] @ Q[b])
            bu[a] += lr * (e - lam * bu[a]); bi[b] += lr * (e - lam * bi[b])
            P[a], Q[b] = P[a] + lr * (e * Q[b] - lam * P[a]), Q[b] + lr * (e * P[a] - lam * Q[b])
    return mu, bu, bi, P, Q


rmse = lambda pred, y: np.sqrt(np.mean((pred - y) ** 2))
rng = np.random.default_rng(0)
nU, nI = 300, 200
u, i, r = make_ratings(nU, nI, k=5, density=0.1, rng=rng)
test = rng.random(len(r)) < 0.2
tr, te = ~test, test
print(f"{len(r)} ratings ({len(r) / (nU * nI):.0%} of the matrix), train {tr.sum()}, test {te.sum()}")

mu = r[tr].mean()
print(f"baseline global mean: test RMSE {rmse(np.full(te.sum(), mu), r[te]):.3f}")

for k, lam in [(5, 0.0), (5, 0.1), (20, 0.1), (20, 0.5)]:
    m, P, Q = als(u[tr], i[tr], r[tr], nU, nI, k, lam, 15, np.random.default_rng(1))
    p_tr = m + np.sum(P[u[tr]] * Q[i[tr]], 1)
    p_te = np.clip(m + np.sum(P[u[te]] * Q[i[te]], 1), 1, 5)
    print(f"ALS k={k:2d} lambda={lam}: train RMSE {rmse(p_tr, r[tr]):.3f} | test RMSE {rmse(p_te, r[te]):.3f}")

for lr, epochs in [(0.01, 40), (0.05, 100)]:
    m, bu, bi, P, Q = sgd_mf(u[tr], i[tr], r[tr], nU, nI, 5, 0.05, lr, epochs, np.random.default_rng(1))
    p_te = np.clip(m + bu[u[te]] + bi[i[te]] + np.sum(P[u[te]] * Q[i[te]], 1), 1, 5)
    print(f"SGD k=5 with biases, lr={lr}, {epochs} epochs: test RMSE {rmse(p_te, r[te]):.3f}")

# top-N for one user: rank items they have not rated
m, P, Q = als(u[tr], i[tr], r[tr], nU, nI, 5, 0.1, 15, np.random.default_rng(1))
seen = set(i[u == 0])
scores = m + Q @ P[0]
print("top-5 unseen items for user 0:", [int(b) for b in np.argsort(-scores) if b not in seen][:5])
```

Output (numpy 2.5):
```
6044 ratings (10% of the matrix), train 4824, test 1220
baseline global mean: test RMSE 0.957
ALS k= 5 lambda=0.0: train RMSE 0.410 | test RMSE 1.294
ALS k= 5 lambda=0.1: train RMSE 0.335 | test RMSE 0.519
ALS k=20 lambda=0.1: train RMSE 0.275 | test RMSE 0.611
ALS k=20 lambda=0.5: train RMSE 0.913 | test RMSE 0.931
SGD k=5 with biases, lr=0.01, 40 epochs: test RMSE 0.971
SGD k=5 with biases, lr=0.05, 100 epochs: test RMSE 0.490
top-5 unseen items for user 0: [24, 22, 34, 163, 122]
```

Each user has rated about 16 items on average, so fitting 5 free factors per user with no penalty overfits: train RMSE 0.41 but test RMSE 1.29, worse than always predicting the mean. A small weighted penalty (λ = 0.1) gives test RMSE 0.519, close to the 0.3 noise floor plus clipping. Raising the rank to 20 at the same λ fits training data better (0.275) but generalises worse (0.611), and λ = 0.5 over-shrinks the factors so the model is barely better than the mean (0.931). SGD needs care: with `lr = 0.01` and small initial factors it is still stuck near zero after 40 epochs (0.971, worse than baseline). With `lr = 0.05` for 100 epochs it reaches 0.490, slightly better than ALS here thanks to the bias terms.

## Choosing / trade-offs
- **ALS vs SGD.** ALS has no learning rate, converges in 10–20 sweeps and parallelises per user/item (Spark MLlib uses it), and handles implicit-feedback weighting over all items efficiently. SGD is simpler to extend with biases, side features and other losses, but needs learning-rate and epoch tuning.
- **Rank k and λ.** Tune together on a held-out split. Higher rank needs stronger regularisation; the best pair here was k = 5, λ = 0.1, matching the true rank.
- **Biases.** Always add user and item biases (or at least the global mean). Much of the rating signal is "this user rates high" and "this item is popular".
- **Evaluation.** RMSE for rating prediction; for top-N recommendation use ranking metrics (precision@k, recall@k, NDCG) on a time-based split ([[learning-to-rank]]).
- **Library.** `surprise` (SVD, SVD++), `implicit` (ALS and BPR for implicit feedback), Spark MLlib `ALS` at scale, or a PyTorch embedding model when you need side features.

## Gotchas
- Do not zero-fill missing ratings and run plain SVD: the model then learns that unrated means a rating of 0.
- With λ = 0, a user or item with fewer ratings than `k` makes `Qᵤᵀ Qᵤ` singular and `np.linalg.solve` fails (it did for k = 20 here). Any λ > 0 fixes it.
- Random train/test splits leak the future in real logs. Split by time for anything you will deploy.
- SGD from small initial factors sits near a saddle point at first; too low a learning rate looks like "MF does not work" (0.971 above).
- Clip predictions to the rating scale before scoring RMSE.
- Cold start: a new user has no factors. Fall back to popularity, ask for a few seed ratings, or use a content or hybrid model.
- Popular items dominate top-N lists. Exclude already-seen items and consider diversity or popularity debiasing.

## Related
- [[recommender-systems]] - overview, implicit feedback, two-tower and sequential models, libraries.
- [[ridge-and-lasso-from-scratch]] - each ALS half-step is one ridge regression per user or item.
- [[pca-from-scratch]] - low-rank approximation of a fully observed matrix (SVD/PCA).
- [[embeddings]] - learned item and user vectors used for similarity search.
- [[learning-to-rank]] - ranking metrics and losses for top-N recommendation.
- [[word2vec-skip-gram-from-scratch-numpy]] - another factorisation of a co-occurrence matrix trained with SGD.

## References
- Koren, Bell and Volinsky (2009), "Matrix Factorization Techniques for Recommender Systems", IEEE Computer: https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf
- Hu, Koren and Volinsky (2008), "Collaborative Filtering for Implicit Feedback Datasets", ICDM: https://doi.org/10.1109/ICDM.2008.22
- Simon Funk (2006), "Netflix Update: Try This at Home": https://sifter.org/~simon/journal/20061211.html
