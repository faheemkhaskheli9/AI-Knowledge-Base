---
title: Isolation Forest from scratch (random isolation trees, path length, anomaly score)
category: ml
tags: [isolation-forest, anomaly-detection, outlier-detection, random-trees, path-length, unsupervised-learning, numpy, scikit-learn, from-scratch, ml-basics]
use_cases:
  - "implement Isolation Forest from scratch and check it against scikit-learn"
  - "understand why anomalies have short paths in random trees"
  - "score outliers in tabular data with no labels"
  - "explain the c(n) normaliser and the 0.5 score threshold in an interview"
status: draft
last_verified: 2026-10-04
sources:
  - https://doi.org/10.1109/ICDM.2008.17
  - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
  - https://scikit-learn.org/stable/modules/outlier_detection.html
---

# Isolation Forest from scratch (random isolation trees, path length, anomaly score)

## Summary
Isolation Forest finds anomalies by how easily they can be separated, not by how far they are from anything. Each tree picks a random feature and a random split value inside that feature's range, again and again, until every point sits alone. Rare, far-away points get isolated after a few splits; points inside dense regions need many. Averaging the path length over 100 trees built on random subsamples of 256 points gives the score. The NumPy version below has the same ROC AUC as scikit-learn (0.949 vs 0.947) on 2-D blobs with scattered outliers, and shows how the method degrades when irrelevant features are added.

## Key concepts
- **Isolation tree.** Recursively choose a random feature `f` and a uniform random split `s` in `[min, max]` of the current points. Stop when a node holds one point or the depth limit `ceil(log2 ψ)` is reached.
- **Path length `h(x)`.** Number of edges from the root to the leaf `x` lands in, plus `c(n)` for a leaf that still holds `n` points (the expected depth of the subtree that was never built).
- **Normaliser `c(n) = 2H(n−1) − 2(n−1)/n`**, with `H(i) ≈ ln i + 0.5772`. It is the average path length of an unsuccessful search in a binary search tree of `n` points, so `E[h]/c(ψ)` is comparable across subsample sizes.
- **Score `s = 2^(−E[h(x)] / c(ψ))`.** Close to 1: anomaly. About 0.5 or less: normal. The 0.5 boundary is a rule of thumb, not a calibrated probability.
- **Subsampling.** Each tree sees only `ψ = 256` points. Small samples reduce *swamping* (normal points near anomalies looking anomalous) and *masking* (dense groups of anomalies hiding each other), and make training linear in the number of trees.
- **No distances, no density.** Cost is O(t·ψ·log ψ) to train and O(t·log ψ) per scored point, independent of dataset size.

## When to use / scenarios
- Learning: a model where randomness is the whole algorithm, and a direct contrast with distance-based outlier detectors ([[knn-from-scratch]], [[dbscan-from-scratch]]).
- Interviews: "how does Isolation Forest work", "why random splits", "what does the score mean", "what is `contamination`".
- Unlabelled fraud, intrusion or sensor-fault screening on tabular data with a modest number of mostly relevant features ([[anomaly-detection]]).
- A fast first baseline before density models ([[gmm-em-from-scratch]], [[kernel-methods-and-density-estimation]]) or autoencoders ([[autoencoder-from-scratch-numpy]]).
- Not for: anomalies defined by context or order (time series need features like lags first); data with many irrelevant features (see below); local anomalies sitting between clusters of very different density, where LOF can do better.

## Setup & code
`pip install numpy scikit-learn`. Runs in a few seconds on CPU (pure-Python tree traversal, so it slows down linearly with rows scored).

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score


def c(n):
    """Average path length of an unsuccessful BST search over n points (normaliser)."""
    return 2 * (np.log(n - 1) + np.euler_gamma) - 2 * (n - 1) / n if n > 2 else float(n == 2)


def build(X, depth, max_depth, rng):
    if depth >= max_depth or len(X) <= 1 or np.all(X == X[0]):
        return ("leaf", len(X))
    spread = X.max(0) > X.min(0)
    f = rng.choice(np.flatnonzero(spread))                # random feature that still varies
    s = rng.uniform(X[:, f].min(), X[:, f].max())         # random split inside its range
    left = X[:, f] < s
    return ("node", f, s, build(X[left], depth + 1, max_depth, rng), build(X[~left], depth + 1, max_depth, rng))


def path_length(x, node, depth=0):
    if node[0] == "leaf":
        return depth + c(node[1])                         # unfinished subtree: add its expected depth
    _, f, s, l, r = node
    return path_length(x, l if x[f] < s else r, depth + 1)


class IForest:
    def __init__(self, n_trees=100, psi=256, seed=0):
        self.n_trees, self.psi, self.rng = n_trees, psi, np.random.default_rng(seed)

    def fit(self, X):
        self.psi = min(self.psi, len(X))
        max_depth = int(np.ceil(np.log2(self.psi)))
        self.trees = [build(X[self.rng.choice(len(X), self.psi, replace=False)], 0, max_depth, self.rng)
                      for _ in range(self.n_trees)]
        return self

    def score(self, X):
        """Anomaly score s in (0, 1): ~1 anomaly, <=0.5 normal."""
        E = np.array([[path_length(x, t) for t in self.trees] for x in X]).mean(1)
        return 2 ** (-E / c(self.psi))


rng = np.random.default_rng(0)
Xn, _ = make_blobs(n_samples=1000, centers=[[0, 0], [5, 5]], cluster_std=1.0, random_state=0)
Xa = rng.uniform(-6, 11, size=(50, 2))                   # scattered anomalies
X = np.vstack([Xn, Xa]); y = np.r_[np.zeros(1000), np.ones(50)]

ours = IForest(seed=0).fit(X).score(X)
sk = -IsolationForest(n_estimators=100, max_samples=256, random_state=0).fit(X).score_samples(X)
print(f"ROC AUC: ours {roc_auc_score(y, ours):.3f} | sklearn {roc_auc_score(y, sk):.3f}")
print(f"rank correlation ours vs sklearn: {np.corrcoef(ours.argsort().argsort(), sk.argsort().argsort())[0, 1]:.3f}")
print(f"mean score: normal {ours[y == 0].mean():.3f} | anomalies {ours[y == 1].mean():.3f}")
print("score at cluster centre vs far point:", IForest(seed=0).fit(X).score(np.array([[0, 0], [10, -5]])).round(3))

# irrelevant noise features dilute the random feature choice
for k in [0, 8, 32]:
    Xk = np.hstack([X, rng.normal(size=(len(X), k))])
    a = roc_auc_score(y, IForest(seed=0).fit(Xk).score(Xk))
    print(f"+{k:2d} noise features: AUC {a:.3f}")
```

Output (numpy 2.5, scikit-learn 1.9):
```
ROC AUC: ours 0.949 | sklearn 0.947
rank correlation ours vs sklearn: 0.937
mean score: normal 0.436 | anomalies 0.640
score at cluster centre vs far point: [0.395 0.787]
+ 0 noise features: AUC 0.949
+ 8 noise features: AUC 0.796
+32 noise features: AUC 0.578
```

The from-scratch forest ranks points almost as scikit-learn does (Spearman 0.94; the two use different random trees, so the scores are not identical) and detects the uniform outliers equally well, AUC 0.949 vs 0.947. Normal points average a score of 0.436 and anomalies 0.640; a point at a cluster centre scores 0.395 and one far outside 0.787, on either side of the 0.5 rule of thumb. Some of the 50 "anomalies" were drawn inside the blobs, which is why AUC is not 1. Adding 8 pure-noise features drops AUC to 0.796 and 32 drops it to 0.578, close to chance: most random splits now land on features that carry no signal, so every point needs about the same number of splits.

## Choosing / trade-offs
- **Isolation Forest vs LOF / k-NN distance.** Isolation Forest is faster and scales to millions of rows; LOF handles clusters of different density better but is O(n²) without an index.
- **vs density models (GMM, KDE).** Density models give a likelihood and work well in low dimensions with a sensible shape; Isolation Forest makes no distributional assumption.
- **vs autoencoder reconstruction error.** Autoencoders suit high-dimensional correlated inputs (images, many sensors); Isolation Forest suits tabular features that each mean something.
- **Hyperparameters.** `n_estimators` 100 is usually enough; `max_samples` 256 is the paper's default; `contamination` only sets the threshold for `predict`, not the scores.
- **Extended Isolation Forest** uses random hyperplanes instead of axis-parallel splits and removes the axis-aligned artefacts in the score map.

## Gotchas
- scikit-learn's `score_samples` returns the *negative* score (lower = more anomalous); `decision_function` is shifted so that 0 is the threshold. Negate before computing AUC.
- `contamination="auto"` uses the paper's 0.5 cut-off; a number such as 0.01 forces exactly that fraction of training points to be flagged, whether or not they are anomalies.
- Irrelevant features hurt quickly (AUC 0.95 to 0.58 above). Select features with domain knowledge or use a feature-bagging variant ([[curse-of-dimensionality]]).
- Scaling does not matter (splits are drawn inside each feature's range), but categorical codes do: integer-encoded categories create meaningless "ranges".
- Duplicated normal values (many identical rows) can make rare-but-legitimate values look anomalous; check flagged points before acting.
- Scores are relative to the training data. Retrain when the normal distribution drifts.

## Related
- [[anomaly-detection]] - the wider toolbox: LOF, One-Class SVM, autoencoders, PyOD.
- [[random-forest-from-scratch]] - the supervised forest whose tree-building recursion this mirrors.
- [[decision-tree-from-scratch]] - greedy splits, in contrast to the random splits here.
- [[dbscan-from-scratch]] - density-based clustering that labels low-density points as noise.
- [[gmm-em-from-scratch]] - likelihood-based outlier scores.
- [[autoencoder-from-scratch-numpy]] - reconstruction error as an anomaly score.

## References
- Liu, Ting and Zhou (2008), "Isolation Forest", ICDM: https://doi.org/10.1109/ICDM.2008.17
- scikit-learn `IsolationForest`: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
- scikit-learn user guide, novelty and outlier detection: https://scikit-learn.org/stable/modules/outlier_detection.html
