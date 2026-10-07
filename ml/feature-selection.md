---
title: Feature selection
category: ml
tags: [feature-selection, filter-methods, wrapper-methods, embedded-methods, rfe, mutual-information, permutation-importance, lasso, boruta, scikit-learn]
use_cases:
  - "cut a wide table of hundreds of columns down to the ones that actually help the model"
  - "make a model cheaper to serve and easier to explain by dropping useless features"
  - "find which sensor readings or survey questions matter for predicting an outcome"
  - "remove redundant, constant or leaky columns before training"
  - "select features without leaking the test set into the choice"
status: stable
last_verified: 2026-10-04
sources:
  - https://scikit-learn.org/stable/modules/feature_selection.html
  - https://scikit-learn.org/stable/modules/permutation_importance.html
  - https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html
  - https://www.jmlr.org/papers/v3/guyon03a.html
---

# Feature selection

## Summary
Feature selection keeps a subset of the original columns and drops the rest. It makes models cheaper to train and serve, easier to explain, less exposed to drift in columns that never mattered, and sometimes more accurate when there are many noisy features and few rows. There are three families: *filter* methods score each feature alone (fast), *wrapper* methods search subsets by retraining the model (slow, model-specific), and *embedded* methods let the model's own training pick (Lasso, tree importances). The selection must happen inside cross-validation, or it leaks and the reported score is optimistic.

## Key concepts
- **Selection vs extraction.** Selection keeps original columns (interpretable); extraction builds new ones such as PCA components ([[dimensionality-reduction]]).
- **Cleanup first (always worth doing).** Drop constant and near-constant columns (`VarianceThreshold`), exact duplicates, ID-like columns (unique per row), and anything only known *after* the target event (leakage).
- **Filter methods.** Score each feature against the target, keep the top k: `f_classif`/`f_regression` (linear relation), `mutual_info_classif`/`mutual_info_regression` (any relation, noisier), chi-squared for non-negative counts. Fast, model-agnostic, but blind to interactions and redundancy (two copies of the same strong feature both score high).
- **Wrapper methods.** Train the model on candidate subsets: `RFE`/`RFECV` (recursively drop the weakest by the model's coefficients or importances), `SequentialFeatureSelector` (greedy forward/backward by CV score). Capture interactions, cost many model fits.
- **Embedded methods.** Selection is a side effect of training: L1 (Lasso, L1-logistic) drives coefficients to exactly zero ([[linear-models]]); tree ensembles give importances ([[decision-trees-and-random-forests]]). `SelectFromModel` turns either into a selector.
- **Permutation importance.** Shuffle one column on *validation* data and measure the score drop. Model-agnostic and measured on held-out data, unlike impurity importance (which favours high-cardinality columns). Correlated features share credit, so both can look unimportant.
- **Boruta-style shadow features.** Add shuffled copies of every column; keep only real features that beat the best shadow consistently. A cheap, principled "is this better than noise" test.
- **Stability.** Selections change between resamples when features are correlated; check how often each feature is chosen across CV folds before trusting a short list.

## When to use / scenarios
- Wide tabular data: hundreds of engineered features, sensor channels or survey items, and a need for a smaller, explainable model ([[feature-engineering]]).
- Regulated scoring (credit, insurance, healthcare): fewer features means fewer to justify and audit ([[finance]], [[healthcare]], [[model-interpretability]]).
- Serving cost: each feature is a pipeline, a lookup and a drift monitor; dropping 80% of them that add nothing is a real saving.
- Genomics and other p ≫ n data: L1 or filter + wrapper on few samples (with nested CV).
- NOT needed for most gradient boosting on moderate width: boosting already ignores useless columns; selection mainly saves cost, rarely accuracy ([[gradient-boosting-tabular]]).
- NOT for raw images, audio or text: deep nets learn their own features; use embeddings and [[dimensionality-reduction]] instead.

## Setup & code
```bash
pip install scikit-learn numpy
```
Ten informative features hidden among 90 noise columns. Compare filter, embedded (L1), wrapper (RFECV) and permutation importance, all inside a pipeline so cross-validation scores stay honest:
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFECV, SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=400, n_features=100, n_informative=10,
                           n_redundant=0, n_clusters_per_class=1,
                           shuffle=False, random_state=0)  # cols 0-9 informative

base = LogisticRegression(max_iter=2000)
pipes = {
    "all 100 features": make_pipeline(StandardScaler(), base),
    "filter: top-10 MI": make_pipeline(StandardScaler(),
                                       SelectKBest(mutual_info_classif, k=10), base),
    "embedded: L1": make_pipeline(StandardScaler(), SelectFromModel(
        LogisticRegression(penalty="l1", C=0.1, solver="liblinear")), base),
}
for name, p in pipes.items():                     # selection is refit inside every fold
    print(f"{name:20s} CV acc {cross_val_score(p, X, y, cv=5).mean():.3f}")

# Wrapper: recursive elimination with CV choosing how many features to keep.
Xs = StandardScaler().fit_transform(X)
rfe = RFECV(LogisticRegression(max_iter=2000), step=5, cv=5).fit(Xs, y)
print("RFECV kept", rfe.n_features_, "features; informative hit:",
      np.sum(rfe.support_[:10]), "/ 10")

# Permutation importance on held-out data.
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)
model = pipes["all 100 features"].fit(X_tr, y_tr)
pi = permutation_importance(model, X_te, y_te, n_repeats=10, random_state=0)
top = np.argsort(pi.importances_mean)[::-1][:10]
print("top-10 by permutation:", sorted(top.tolist()))
```
The selected pipelines should roughly match "all 100 features" with a tenth of the columns (L1 slightly ahead). RFECV keeps only true informative columns; permutation importance finds about half of them plus some noise columns, which is why stability across folds is worth checking.

## Choosing / trade-offs
- **Start cheap.** Cleanup + one embedded method (L1 for linear models, `SelectFromModel` on a forest/boosting model) covers most projects.
- **Filter** when features number in the thousands and you need a fast first cut; follow with a wrapper or embedded method on the survivors.
- **Wrapper (RFECV, sequential)** when the final model is fixed and you can afford many fits; best at finding a small set, worst at cost and overfitting the CV score.
- **Permutation importance** for explaining or pruning a finished model of any type; group correlated features and permute them together, or the scores mislead.
- **L1 vs tree importances.** L1 picks one from a group of correlated features arbitrarily; elastic net keeps the group ([[linear-models]]). Impurity importance is biased toward high-cardinality and continuous columns; prefer permutation or SHAP ([[model-interpretability]]).
- **Selection vs regularisation alone.** If you only care about accuracy, a well-regularised model on all features is often as good; select when cost, explainability or drift surface matters.

## Gotchas
- Selecting on the full dataset and then cross-validating leaks: with enough noise columns, some correlate with the target by chance and the CV score looks great. Put the selector in the `Pipeline`, or use nested CV.
- Univariate filters miss features that only matter together (XOR-like interactions) and keep redundant duplicates.
- Mutual information estimates are noisy on small data; fix `random_state` and average across seeds.
- Scale features before L1 or coefficient-based RFE; otherwise selection reflects units, not signal.
- A feature with zero importance in one model may matter in another; re-select when you change model families.
- Importance is not causation: a selected feature may proxy for the real cause or for a protected attribute ([[causal-inference-and-uplift]]).
- Dropping a column that downstream rules, reports or fairness checks rely on; coordinate before removing it from the pipeline.

## Related
- [[feature-engineering]] - building and cleaning features before selecting them.
- [[dimensionality-reduction]] - extraction (PCA, UMAP) instead of selection.
- [[linear-models]] - Lasso and elastic net as embedded selectors.
- [[model-interpretability]] - permutation importance and SHAP.
- [[model-evaluation-and-metrics]] - cross-validation and nested CV.
- [[automl]] - tools that search features and models together.

## References
- scikit-learn, Feature selection: https://scikit-learn.org/stable/modules/feature_selection.html
- scikit-learn, Permutation importance: https://scikit-learn.org/stable/modules/permutation_importance.html
- `SequentialFeatureSelector`: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html
- Guyon & Elisseeff, An Introduction to Variable and Feature Selection (JMLR 2003): https://www.jmlr.org/papers/v3/guyon03a.html
