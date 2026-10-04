---
title: k-nearest neighbours from scratch (vectorised distances, argpartition, voting, choosing k)
category: ml
tags: [knn, k-nearest-neighbors, distance, euclidean, lazy-learning, feature-scaling, numpy, scikit-learn, from-scratch, ml-basics]
use_cases:
  - "implement k-NN classification from scratch for learning or an interview"
  - "compute pairwise distances fast in NumPy without Python loops"
  - "see why feature scaling and the choice of k matter for k-NN"
  - "match scikit-learn's KNeighborsClassifier with a few lines of NumPy"
status: draft
last_verified: 2026-10-04
sources:
  - https://scikit-learn.org/stable/modules/neighbors.html
  - https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
  - https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html
---

# k-nearest neighbours from scratch (vectorised distances, argpartition, voting, choosing k)

## Summary
k-NN has no training step: it stores the training set and, for each query, takes a vote among the `k` closest stored points. All of the work is at prediction time, so a from-scratch version is mostly about computing distances fast. One matrix product (`‖a‖² − 2a·b + ‖b‖²`) plus `argpartition` gives an implementation that agrees with scikit-learn on every test point.

## Key concepts
- **Lazy learner.** Fit = store `(X, y)`. Predict = find neighbours and vote. Training is free, but prediction costs `O(n·d)` per query with brute force.
- **Vectorised distance.** `‖a − b‖² = ‖a‖² − 2a·b + ‖b‖²`. The middle term is one `(m, d) × (d, n)` matrix product, far faster than an `(m, n, d)` broadcast and far lighter on memory. Clip tiny negative values caused by rounding before taking a square root.
- **argpartition.** You need the `k` smallest distances per row, not a full sort. `np.argpartition(d, k)` does it in `O(n)` per row instead of `O(n log n)`.
- **Voting.** Majority vote over neighbour labels. Distance-weighted voting (`w = 1/d`) lets near neighbours count more; regression averages neighbour targets instead.
- **k as a bias-variance knob.** `k = 1` memorises the training set (low bias, high variance). Large `k` smooths toward the majority class (high bias). At `k = n` every prediction is the overall majority ([[bias-variance-and-learning-curves]]).
- **Scale dependence.** Distances add up raw feature differences, so a feature measured in thousands drowns one measured in fractions ([[feature-scaling-and-normalization]]).

## When to use / scenarios
- Learning: the simplest non-parametric classifier, and a clean way to practise vectorised NumPy.
- Interviews: writing the distance trick, explaining why scaling and `k` matter, and the cost at prediction time.
- Small to medium tabular data with a meaningful distance, as a quick baseline ([[svm-knn-naive-bayes]]).
- Retrieval over embeddings is k-NN at scale: there you use an approximate index (FAISS, HNSW) rather than brute force ([[distance-metrics-and-similarity]]).
- Not for: high-dimensional raw features where all distances look alike ([[curse-of-dimensionality]]), or latency-sensitive serving over millions of points without an index.

## Setup & code
`pip install numpy scikit-learn`. Runs in about a second on CPU.

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def sq_dists(A, B):
    """Pairwise squared Euclidean distances via ||a||^2 - 2 a.b + ||b||^2."""
    d = (A ** 2).sum(1)[:, None] - 2 * A @ B.T + (B ** 2).sum(1)[None]
    return np.maximum(d, 0)                       # clip tiny negatives from rounding


def knn_predict(Xtr, ytr, Xte, k=5, weighted=False):
    d = sq_dists(Xte, Xtr)
    idx = np.argpartition(d, k, axis=1)[:, :k]    # k nearest, unordered, O(n) per row
    w = 1 / (np.sqrt(np.take_along_axis(d, idx, 1)) + 1e-12) if weighted \
        else np.ones(idx.shape)
    n_cls = ytr.max() + 1
    votes = np.zeros((len(Xte), n_cls))
    np.add.at(votes, (np.arange(len(Xte))[:, None], ytr[idx]), w)
    return votes.argmax(1)


X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

print("raw features   acc:", (knn_predict(Xtr, ytr, Xte) == yte).mean().round(3))
sc = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
ours = knn_predict(Xtr_s, ytr, Xte_s)
sk = KNeighborsClassifier(n_neighbors=5).fit(Xtr_s, ytr).predict(Xte_s)
print("scaled features acc:", (ours == yte).mean().round(3))
print("agreement with sklearn:", (ours == sk).mean())

for k in (1, 3, 5, 15, 51, 151):
    acc = (knn_predict(Xtr_s, ytr, Xte_s, k) == yte).mean()
    print(f"k={k:>3}: test acc={acc:.3f}")
```

Output (numpy 2.5, scikit-learn 1.9):
```
raw features   acc: 0.912
scaled features acc: 0.947
agreement with sklearn: 1.0
k=  1: test acc=0.947
k=  3: test acc=0.942
k=  5: test acc=0.947
k= 15: test acc=0.942
k= 51: test acc=0.947
k=151: test acc=0.883
```

Standardising the 30 features lifts accuracy from 0.912 to 0.947 with no other change, and the from-scratch predictions agree with scikit-learn on every test point. Accuracy is flat for `k` from 1 to 51 on this easy dataset and only drops once `k = 151` starts pulling in a large share of the 398 training points. Pick `k` by cross-validation on the training split, not on this test set.

## Choosing / trade-offs
- **Brute force vs tree vs approximate index.** Brute force is exact and best for small `n` or high `d`. KD-trees and ball trees speed up exact search in low dimensions (roughly `d < 20`). For millions of embeddings, use an approximate index (HNSW, IVF in FAISS) and accept slightly imperfect recall.
- **Uniform vs distance weights.** Distance weighting helps near class boundaries and with larger `k`. Watch for exact duplicates (distance 0): add a small epsilon or let the duplicate's label win.
- **Metric.** Euclidean on standardised features is the default. Use cosine for embeddings and text vectors, Manhattan for robustness to single large differences ([[distance-metrics-and-similarity]]).
- **k-NN vs a trained model.** k-NN adapts to new data instantly (just add rows) but costs memory and prediction time. A trained model (logistic regression, gradient boosting) is usually more accurate on tabular data and constant-cost to serve ([[gradient-boosting-tabular]]).

## Gotchas
- Fit the scaler on the training split only. Scaling with test statistics is leakage ([[data-leakage-and-validation-splits]]).
- Prefer odd `k` for binary problems so a vote cannot tie. For ties with more classes, `argmax` picks the lowest class id, which is a silent bias.
- When predicting on the training set itself, each point is its own nearest neighbour at distance 0, so training accuracy at `k = 1` is always 100% and means nothing.
- The full `(m, n)` distance matrix can be huge (10k queries × 1M points is 80 GB in float64). Process queries in chunks.
- `‖a‖² − 2a·b + ‖b‖²` loses precision when points are far from the origin and close to each other. Centre the data (scaling does) or use float64.
- Categorical features one-hot encoded next to numeric ones get an arbitrary weight in the distance. Consider Gower distance or a learned embedding ([[categorical-encoding]]).
- Imbalanced classes: the majority class dominates votes as `k` grows. Use distance weighting, class-balanced resampling or a smaller `k` ([[imbalanced-data]]).

## Related
- [[svm-knn-naive-bayes]] - k-NN next to SVMs and naive Bayes, with scikit-learn usage.
- [[distance-metrics-and-similarity]] - choosing the distance and approximate nearest-neighbour indexes.
- [[feature-scaling-and-normalization]] - required before any distance-based method.
- [[curse-of-dimensionality]] - why k-NN degrades in high dimensions.
- [[k-means-from-scratch]] - uses the same vectorised distance computation.
- [[naive-bayes-from-scratch]] - another simple baseline classifier built from scratch.

## References
- scikit-learn, Nearest neighbors: https://scikit-learn.org/stable/modules/neighbors.html
- scikit-learn KNeighborsClassifier API: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
- NumPy `argpartition`: https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html
