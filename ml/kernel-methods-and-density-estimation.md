---
title: Kernel methods and kernel density estimation (kernel ridge, Nystroem, random Fourier features, KDE)
category: ml
tags: [kernel-methods, kernel-trick, rbf-kernel, kernel-ridge-regression, nystroem, random-fourier-features, kernel-approximation, kernel-density-estimation, kde, bandwidth, scikit-learn]
use_cases:
  - "fit a smooth nonlinear regression or classifier on a few thousand rows without a neural network"
  - "scale an RBF-kernel model to hundreds of thousands of rows with a kernel approximation"
  - "estimate the probability density of a variable to flag rare values or sample synthetic ones"
  - "replace a histogram with a smooth density estimate for a report or anomaly threshold"
status: draft
last_verified: 2026-10-04
sources:
  - https://scikit-learn.org/stable/modules/kernel_ridge.html
  - https://scikit-learn.org/stable/modules/kernel_approximation.html
  - https://scikit-learn.org/stable/modules/density.html
  - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
  - https://papers.nips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html
---

# Kernel methods and kernel density estimation (kernel ridge, Nystroem, random Fourier features, KDE)

## Summary
A kernel k(x, x') measures similarity between two points; kernel methods turn linear algorithms (ridge regression, SVMs, PCA) into nonlinear ones by working with these similarities instead of raw features (the kernel trick). They give strong, smooth fits on small and medium tabular or signal data with few hyperparameters, but exact methods scale as O(n²) memory and O(n³) time. Kernel approximations (Nystroem, random Fourier features) map data into a few hundred explicit features so a linear model can mimic the kernel model at a fraction of the cost. The same kernel idea gives kernel density estimation (KDE): a smooth estimate of a probability density built by placing a small bump on every data point.

## Key concepts
- **Kernel trick.** A kernel equals an inner product in some (possibly infinite) feature space, so an algorithm that only needs inner products can run in that space without building it. The model is a weighted sum of kernels centred on training points.
- **Common kernels.** RBF/Gaussian `exp(-gamma·||x-x'||²)` (smooth, default), Laplacian, polynomial, linear; Matérn and periodic kernels in [[bayesian-and-gaussian-processes]]. `gamma` (or length scale) sets how far each point's influence reaches.
- **Kernel ridge regression (KRR).** Ridge regression in kernel space; closed-form solve of an n x n system. Same predictions as the mean of a Gaussian process with matching kernel and noise, but no uncertainty. Hyperparameters: `alpha` (regularisation) and kernel parameters.
- **Kernel SVM vs KRR.** SVMs use hinge/epsilon-insensitive loss and give sparse support vectors; KRR uses squared loss, is faster to fit at moderate n and has no sparsity ([[svm-knn-naive-bayes]]).
- **Nystroem approximation.** Pick m landmark points, build features from kernels to the landmarks; adapts to the data, usually more accurate than RFF for the same m.
- **Random Fourier features (`RBFSampler`).** Random projections + cosine that approximate the RBF kernel in expectation; data-independent, very fast, needs more components for the same accuracy.
- **KDE.** density(x) = (1/n)·Σ K((x - xᵢ)/h)/h. The **bandwidth h** matters far more than the kernel shape: too small gives spiky noise, too large blurs real modes. Choose by cross-validated log-likelihood or rules of thumb (Scott, Silverman).
- **Curse of dimensionality.** Both distance-based kernels and KDE degrade as dimensions grow; KDE is reliable in roughly 1-6 dimensions.

## When to use / scenarios
- Small to medium tabular or sensor data (up to ~10-20k rows) with smooth nonlinear relationships: calibration curves, physical response surfaces, dose-response ([[manufacturing-iot]]).
- Larger data where you still want an RBF-like model: Nystroem/RFF + a linear model, trainable with SGD on millions of rows.
- Features for a linear model inside a pipeline, cheaper than a neural network and easier to tune.
- KDE for anomaly scores on one or a few variables (low density = rare), smooth histograms in reports, and drawing synthetic samples that follow the observed distribution ([[anomaly-detection]]).
- NOT for wide, heterogeneous tabular data with categorical columns: gradient boosting usually wins ([[gradient-boosting-tabular]]).
- NOT for images, text or audio from raw inputs: use deep networks or pretrained embeddings (a kernel on top of embeddings can still work).
- When you need predictive uncertainty too, use a Gaussian process instead of KRR.

## Setup & code
```bash
pip install scikit-learn numpy
```
Nonlinear regression with exact kernel ridge vs two kernel approximations, then KDE with a cross-validated bandwidth:
```python
import time

import numpy as np
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.pipeline import make_pipeline

# Nonlinear regression: y = sin(x0) * cos(x1) + noise.
rng = np.random.default_rng(0)
X = rng.uniform(-3, 3, size=(6000, 2))
y = np.sin(X[:, 0]) * np.cos(X[:, 1]) + rng.normal(0, 0.1, len(X))
X_tr, y_tr, X_te, y_te = X[:5000], y[:5000], X[5000:], y[5000:]

models = {
    "linear ridge": Ridge(alpha=1.0),
    "exact kernel ridge": KernelRidge(kernel="rbf", gamma=0.5, alpha=0.1),        # O(n^2) memory, O(n^3) fit
    "Nystroem(300)+ridge": make_pipeline(Nystroem(gamma=0.5, n_components=300, random_state=0), Ridge(alpha=0.1)),
    "RFF(300)+ridge": make_pipeline(RBFSampler(gamma=0.5, n_components=300, random_state=0), Ridge(alpha=0.1)),
}
for name, m in models.items():
    t = time.perf_counter()
    m.fit(X_tr, y_tr)
    print(f"{name:20s} R2 {r2_score(y_te, m.predict(X_te)):.3f}  fit {time.perf_counter() - t:.2f}s")

# Kernel density estimation: pick the bandwidth by cross-validated log-likelihood.
data = np.concatenate([rng.normal(0, 1, 400), rng.normal(5, 0.5, 200)])[:, None]
grid = GridSearchCV(KernelDensity(kernel="gaussian"), {"bandwidth": np.logspace(-1.5, 0.5, 20)}, cv=5)
kde = grid.fit(data).best_estimator_
print("best bandwidth:", round(kde.bandwidth, 3))
pts = np.array([[0.0], [2.5], [5.0]])
print("density at 0, 2.5, 5:", np.exp(kde.score_samples(pts)).round(3))
print("3 samples:", kde.sample(3, random_state=0).ravel().round(2))
```
Output with scikit-learn 1.9.0 (timings from one CPU run; the gap grows as O(n³) with more rows):
```text
linear ridge         R2 0.003  fit 0.02s
exact kernel ridge   R2 0.962  fit 1.13s
Nystroem(300)+ridge  R2 0.962  fit 0.07s
RFF(300)+ridge       R2 0.961  fit 0.03s
best bandwidth: 0.28
density at 0, 2.5, 5: [0.237 0.021 0.225]
3 samples: [-1.28  5.62 -1.01]
```
The approximations match exact KRR at ~1/15 of the fit time. KDE finds both modes (0 and 5) and a low density in the gap at 2.5. `score_samples` returns log-density. For a quick 1-D estimate with an automatic bandwidth rule, `scipy.stats.gaussian_kde(values)` is a one-liner.

## Choosing / trade-offs
- **Exact KRR / SVC** up to ~10-20k rows; beyond that memory (n² floats) and time explode.
- **Nystroem** as the default approximation; **RFF** when data arrives in a stream or you want features independent of the training set. Increase `n_components` until the validation score stops improving (a few hundred to a few thousand).
- **Kernel approx + `SGDClassifier`/`SGDRegressor`** for millions of rows or online updates ([[online-learning-and-concept-drift]]).
- **KRR vs GP.** Same mean prediction; a GP also gives uncertainty and learns kernel parameters by marginal likelihood, at the same O(n³) cost.
- **KDE vs histogram vs GMM.** Histograms are simplest and fine for reports with many points; KDE is smooth and nonparametric; a [[gaussian-mixture-models-and-em]] model is compact, works in more dimensions and gives cluster structure. For high-dimensional densities use [[normalizing-flows-and-energy-based-models]] or an autoencoder-based score.
- **KDE bandwidth.** Cross-validated likelihood adapts to the data; Scott/Silverman rules over-smooth multi-modal data.

## Gotchas
- Unscaled features make RBF kernels meaningless: one column in metres and one in millimetres means distance is dominated by the latter. Standardise first, inside the pipeline.
- `gamma` and `alpha` interact strongly; tune them together on a log grid ([[hyperparameter-tuning]]).
- `KernelRidge` keeps all training data and predicts in O(n) per point; slow at serving time for large n.
- `RBFSampler` and `Nystroem` are random; fix `random_state` and note that few components give a noisy, under-fitting approximation.
- KDE leaks probability mass past natural bounds (negative prices, ages below 0); transform the variable (log) or reflect at the boundary.
- KDE bandwidth chosen on the full dataset and then used to score the same points for anomaly detection inflates densities; score held-out points or use leave-one-out.
- KDE in more than a handful of dimensions needs exponentially more data; densities become nearly uniform and anomaly scores lose meaning.

## Related
- [[svm-knn-naive-bayes]] - kernel SVMs and the kernel trick for classification.
- [[bayesian-and-gaussian-processes]] - kernels with uncertainty and learned length scales.
- [[linear-models]] - ridge regression, the base that kernels lift to nonlinear fits.
- [[gaussian-mixture-models-and-em]] - parametric density estimation.
- [[anomaly-detection]] - density-based anomaly scores.
- [[dimensionality-reduction]] - kernel PCA and the curse of dimensionality.

## References
- scikit-learn, Kernel ridge regression: https://scikit-learn.org/stable/modules/kernel_ridge.html
- scikit-learn, Kernel approximation (Nystroem, RBFSampler): https://scikit-learn.org/stable/modules/kernel_approximation.html
- scikit-learn, Density estimation (KernelDensity): https://scikit-learn.org/stable/modules/density.html
- SciPy, `gaussian_kde`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
- Rahimi & Recht, Random Features for Large-Scale Kernel Machines (NeurIPS 2007): https://papers.nips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html
