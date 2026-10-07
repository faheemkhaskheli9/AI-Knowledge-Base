---
title: Curse of dimensionality
category: ml
tags: [curse-of-dimensionality, high-dimensional-data, distance-concentration, knn, feature-selection, dimensionality-reduction, sparsity, scikit-learn]
use_cases:
  - "understand why k-NN, clustering or distance-based search degrades with many features"
  - "decide whether to reduce or select features before a distance-based model"
  - "explain why a model with thousands of features and few rows overfits"
  - "pick a model that copes with wide data (genomics, text, sensor arrays)"
status: stable
last_verified: 2026-10-04
sources:
  - https://scikit-learn.org/stable/modules/neighbors.html
  - https://scikit-learn.org/stable/modules/feature_selection.html
  - https://link.springer.com/chapter/10.1007/3-540-49257-7_15
  - https://hastie.su.domains/ElemStatLearn/
---

# Curse of dimensionality

## Summary
As the number of features grows, the volume of the feature space grows exponentially, so any fixed dataset covers it more and more thinly. Two practical effects follow. Distances concentrate: the nearest and farthest points end up almost equally far away, which breaks k-NN, k-means, kernel methods and vector search on raw features. And the number of samples needed to learn a flexible function grows exponentially. The cure is to reduce the effective dimension (feature selection, projection, learned embeddings) or to use models with a strong bias that do not rely on local neighbourhoods.

## Key concepts
- **Sparsity of space.** To cover 10% of a unit hypercube's volume, a sub-cube needs edge `0.1^(1/d)`: 0.32 in 2-D, 0.79 in 10-D, 0.98 in 100-D. "Local" neighbourhoods are no longer local.
- **Distance concentration.** For many i.i.d. features, `(d_max − d_min)/d_min → 0` as dimension grows (Beyer et al., 1999). Nearest-neighbour ranking becomes close to random.
- **Noise features.** Irrelevant features add the same amount to every distance and drown out the few informative ones. This, not raw dimension, is the usual cause in practice.
- **Intrinsic dimension.** Real data (images, text embeddings) often lie near a much lower-dimensional manifold. Methods that exploit that structure escape the curse; methods on raw coordinates do not.
- **p ≫ n.** With more features than rows, a flexible model can fit the training set exactly by chance. Regularisation or feature selection is mandatory.
- **Which models suffer most.** Local, distance-based methods (k-NN, RBF kernels, k-means, DBSCAN, density estimation). Linear models with regularisation, tree ensembles and neural networks with learned representations suffer much less.

## When to use / scenarios
- k-NN, clustering or vector search results look random after adding features: suspect noise features and distance concentration.
- Wide tabular or omics data (hundreds to tens of thousands of columns, a few hundred rows): plan regularisation and feature selection up front.
- Choosing between a distance-based model and a tree/linear model for a wide dataset: prefer the latter unless you first build a good low-dimensional representation.
- NOT a reason to throw away features before trying anything: gradient boosting and L1/L2 linear models often handle hundreds of features well as-is ([[gradient-boosting-tabular]], [[linear-models]]).

## Setup & code
`pip install numpy scikit-learn`. Runs on CPU in a few seconds.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

rng = np.random.default_rng(0)

# 1. Distance concentration: nearest and farthest neighbours look alike in high d
for d in [2, 10, 100, 1000]:
    X = rng.random((1000, d))
    dist = np.linalg.norm(X[1:] - X[0], axis=1)
    print(f"d={d:5d}  (max-min)/min = {(dist.max() - dist.min()) / dist.min():.3f}")

# 2. k-NN accuracy as noise features are added to 5 informative ones
X_inf, y = make_classification(n_samples=500, n_features=5, n_informative=5,
                               n_redundant=0, random_state=0)
for n_noise in [0, 20, 100, 500]:
    X = np.hstack([X_inf, rng.normal(size=(500, n_noise))])
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier())
    print(f"noise={n_noise:4d}  kNN CV acc = {cross_val_score(knn, X, y, cv=5).mean():.3f}")

# 3. Fixes: PCA is unsupervised and keeps the noise directions; supervised selection works
from sklearn.feature_selection import SelectKBest, f_classif

X = np.hstack([X_inf, rng.normal(size=(500, 500))])
for name, step in [("PCA(20)", PCA(n_components=20)),
                   ("SelectKBest(10)", SelectKBest(f_classif, k=10))]:
    pipe = make_pipeline(StandardScaler(), step, KNeighborsClassifier())
    print(f"noise=500 + {name:15s} kNN CV acc = {cross_val_score(pipe, X, y, cv=5).mean():.3f}")
```

Output (numpy 2.5, scikit-learn 1.9):
```
d=    2  (max-min)/min = 164.171
d=   10  (max-min)/min = 2.479
d=  100  (max-min)/min = 0.430
d= 1000  (max-min)/min = 0.111
noise=   0  kNN CV acc = 0.928
noise=  20  kNN CV acc = 0.728
noise= 100  kNN CV acc = 0.634
noise= 500  kNN CV acc = 0.524
noise=500 + PCA(20)         kNN CV acc = 0.534
noise=500 + SelectKBest(10) kNN CV acc = 0.772
```
500 noise columns push k-NN to chance (0.52). PCA does not help here: the noise has as much variance as the signal, so the top components are mostly noise. Supervised selection, fitted inside the pipeline so each CV fold selects on its own training split, recovers most of the accuracy.

## Choosing / trade-offs
- **Feature selection** (filter, L1, tree importances) keeps original, interpretable columns and removes noise; it needs labels and must sit inside cross-validation ([[feature-selection]]).
- **Projection** (PCA, random projection) is cheap and label-free; it helps when the signal lives in the high-variance directions, and fails when noise dominates variance ([[dimensionality-reduction]]).
- **Learned embeddings** (autoencoders, pretrained encoders) give the best low-dimensional space for images, text and audio; they need data or a pretrained model ([[embeddings]]).
- **Switch model family** instead of reducing dimension: regularised linear models and gradient-boosted trees pick out the useful features themselves.
- **Distance choice.** Cosine similarity on normalised embeddings, or Manhattan distance, concentrate a bit less than Euclidean on raw features, but no metric rescues pure noise ([[distance-metrics-and-similarity]]).

## Gotchas
- Selecting features on the full dataset before cross-validation leaks the labels and makes a pure-noise dataset look predictive. Always select inside the pipeline ([[data-leakage-and-validation-splits]]).
- Unscaled features make the curse worse: one large-range column dominates every distance. Standardise before any distance-based method.
- One-hot encoding a high-cardinality column can add thousands of sparse dimensions at once. Use target or hashed encodings, or a tree model ([[categorical-encoding]]).
- High dimension is not automatically bad: in 768-D text embeddings the intrinsic dimension is low and nearest-neighbour search works well. The curse bites on raw, noisy, independent features.
- t-SNE/UMAP plots of high-dimensional data are for looking, not for distances. Do not cluster or measure on their 2-D output.
- With p ≫ n, near-perfect training accuracy is expected even on random labels. Judge only by cross-validated scores.

## Related
- [[feature-selection]] - filter, wrapper and embedded methods to drop noise features.
- [[dimensionality-reduction]] - PCA, random projection, UMAP and when each fits.
- [[distance-metrics-and-similarity]] - how metric choice interacts with dimension.
- [[svm-knn-naive-bayes]] - k-NN and kernel methods, the models hit hardest.
- [[bias-variance-and-learning-curves]] - why many features and few rows means high variance.
- [[embeddings]] - learned low-dimensional representations for unstructured data.

## References
- Beyer et al., When Is "Nearest Neighbor" Meaningful? (ICDT 1999): https://link.springer.com/chapter/10.1007/3-540-49257-7_15
- Hastie, Tibshirani, Friedman, The Elements of Statistical Learning, sec. 2.5: https://hastie.su.domains/ElemStatLearn/
- scikit-learn nearest neighbours: https://scikit-learn.org/stable/modules/neighbors.html
- scikit-learn feature selection: https://scikit-learn.org/stable/modules/feature_selection.html
