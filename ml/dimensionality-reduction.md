---
title: Dimensionality reduction (PCA, t-SNE, UMAP)
category: ml
tags: [dimensionality-reduction, pca, t-sne, umap, truncated-svd, visualization, unsupervised, embeddings, scikit-learn]
use_cases:
  - "visualise high-dimensional embeddings or customer features in a 2D plot"
  - "compress hundreds of correlated features before clustering or a linear model"
  - "shrink embedding vectors to cut vector-database storage and search cost"
  - "check whether classes are separable before building a classifier"
status: stable
last_verified: 2026-10-03
sources:
  - https://scikit-learn.org/stable/modules/decomposition.html
  - https://scikit-learn.org/stable/modules/manifold.html
  - https://umap-learn.readthedocs.io/
  - https://distill.pub/2016/misread-tsne/
---

# Dimensionality reduction (PCA, t-SNE, UMAP)

## Summary
Dimensionality reduction maps data with many features to a few new ones while keeping the structure that matters. PCA finds the linear directions of greatest variance; it is fast, deterministic, reversible and the right tool for compression and preprocessing. t-SNE and UMAP are non-linear methods that preserve local neighbourhoods and are mainly for 2D/3D visualisation of things like embeddings. Use PCA to transform data for a model; use t-SNE/UMAP to look at it.

## Key concepts
- **Curse of dimensionality.** With many features, points become sparse and distances similar, which hurts k-NN, clustering and some models; reducing dimensions helps.
- **PCA.** Rotates data onto orthogonal components ordered by variance. `explained_variance_ratio_` tells how much each keeps; pick the number of components that keeps e.g. 95% (`PCA(n_components=0.95)`).
- **Standardise before PCA** unless features share units, or PCA just finds the column with the largest numbers.
- **TruncatedSVD.** PCA-like reduction that works directly on sparse matrices (TF-IDF); this is classic latent semantic analysis.
- **t-SNE.** Non-linear, keeps close points close; good 2D pictures, slow on large data, stochastic, and distances between clusters and cluster sizes are not meaningful. Key knob: `perplexity` (roughly neighbours considered, 5-50).
- **UMAP.** Like t-SNE but faster, scales to millions, keeps somewhat more global structure, and can `transform` new points. Separate package `umap-learn`. Knobs: `n_neighbors`, `min_dist`.
- **Feature selection vs extraction.** Selection keeps a subset of original columns (interpretable, e.g. Lasso, see [[linear-models]]); extraction creates new combined columns (PCA), which are harder to explain.
- **Supervised reduction.** LDA (`LinearDiscriminantAnalysis`) finds directions that best separate known classes.

## When to use / scenarios
- Embedding exploration: plot sentence or image embeddings in 2D with UMAP/t-SNE to see topics, duplicates or label noise, see [[embeddings]].
- Preprocessing for clustering on wide data (survey responses, gene expression, sensor arrays): PCA to 10-50 components, then cluster, see [[clustering]].
- Compressing embeddings for retrieval (e.g. 1024 → 256 dims) to reduce storage and latency in a [[vector-databases]] index; check recall after.
- Removing multicollinearity before a linear model, or denoising (reconstruct from top components).
- Not when: a tree/boosting model is the downstream model (they handle many features fine and PCA destroys interpretability), or features are few already.

## Setup & code
```bash
pip install scikit-learn
# optional, for UMAP:
pip install umap-learn
```
```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_digits(return_X_y=True)            # 1797 images x 64 pixels
pca = PCA(n_components=0.95).fit(StandardScaler().fit_transform(X))
print("components for 95% variance:", pca.n_components_, "of", X.shape[1])

# PCA as preprocessing inside a pipeline (fit on train folds only).
for name, model in [
    ("raw 64 dims", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
    ("PCA 20 dims", make_pipeline(StandardScaler(), PCA(20), LogisticRegression(max_iter=2000))),
]:
    print(name, "CV acc:", round(cross_val_score(model, X, y, cv=5).mean(), 3))

# t-SNE for a 2D picture (no transform for new points; for plotting only).
X2 = TSNE(n_components=2, perplexity=30, random_state=0).fit_transform(X)
print("t-SNE output shape:", X2.shape)
# import matplotlib.pyplot as plt; plt.scatter(*X2.T, c=y, s=4, cmap="tab10"); plt.show()

# UMAP equivalent (needs umap-learn):
# import umap; X2 = umap.UMAP(n_neighbors=15, min_dist=0.1).fit_transform(X)
```
Here 40 of 64 components keep 95% of the variance, and a model on 20 components scores about 0.90 against 0.92 on all 64 pixels: a third of the inputs for two points of accuracy.

## Choosing / trade-offs
- **PCA** for preprocessing, compression and anything that must be reproducible and applied to new data. Linear, so it misses curved structure.
- **UMAP** for visualising large datasets and when you need to embed new points the same way; tune `n_neighbors` up for more global structure.
- **t-SNE** for the cleanest local-structure plots on small to medium data (up to tens of thousands of points); slow beyond that.
- **TruncatedSVD** for sparse text matrices; PCA would densify them.
- **Autoencoders** (see [[neural-network-fundamentals]]) for non-linear compression of images or signals when PCA loses too much.

## Gotchas
- Fitting PCA on all data before splitting leaks test information; put it inside a `Pipeline`.
- t-SNE/UMAP plots invite over-reading: gaps between clusters, cluster sizes and relative positions are mostly not meaningful. Run several perplexities/seeds before concluding.
- Do not train a downstream model on t-SNE output; it has no stable mapping for new data.
- Clusters that appear only in the 2D plot may not exist in the original space; validate in the original or PCA space.
- PCA components are mixes of all features; "component 1" rarely has a clean business meaning.
- Matryoshka-trained embedding models can be truncated directly to fewer dimensions; for others, truncating without PCA hurts quality, see [[embedding-models]].

## Related
- [[clustering]] - usually the next step after reducing dimensions.
- [[embeddings]] - the most common high-dimensional data to reduce and visualise.
- [[feature-engineering]] - selecting and creating features before reducing them.
- [[linear-models]] - Lasso as an interpretable alternative (feature selection).
- [[vector-databases]] - storage and latency gains from smaller vectors.

## References
- scikit-learn decomposition (PCA, TruncatedSVD): https://scikit-learn.org/stable/modules/decomposition.html
- scikit-learn manifold learning (t-SNE): https://scikit-learn.org/stable/modules/manifold.html
- UMAP documentation: https://umap-learn.readthedocs.io/
- Wattenberg et al., *How to Use t-SNE Effectively*: https://distill.pub/2016/misread-tsne/
