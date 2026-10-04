---
title: Feature scaling and normalization (standard, min-max, robust, quantile)
category: ml
tags: [feature-scaling, standardization, normalization, standardscaler, minmaxscaler, robustscaler, quantiletransformer, preprocessing, scikit-learn, ml-basics]
use_cases:
  - "decide whether a model needs its input features scaled, and with which scaler"
  - "fix a k-NN, SVM or neural net that performs badly on raw tabular features"
  - "speed up slow or non-converging gradient-based training on unscaled data"
  - "scale features with outliers or heavy tails without letting them dominate"
status: draft
last_verified: 2026-10-04
sources:
  - https://scikit-learn.org/stable/modules/preprocessing.html
  - https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
  - https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html
---

# Feature scaling and normalization (standard, min-max, robust, quantile)

## Summary
Feature scaling puts numeric columns on comparable ranges before training. Models that measure distances (k-NN, k-means, RBF SVMs) or train by gradient descent (linear models, neural nets) are dominated by whichever feature has the largest numbers, so on raw data they can lose 30 points of accuracy or take hundreds of times more iterations to converge. Tree models split one feature at a time and do not care. The scaler is learned from data, so it is fit on the training split only, inside a pipeline.

## Key concepts
- **Standardization (z-score).** `(x − mean) / std` per column, giving mean 0 and std 1. The default for linear models, SVMs, PCA and neural nets (`StandardScaler`).
- **Min-max scaling.** `(x − min) / (max − min)` into `[0, 1]`. Useful when a bounded range is required (pixel-like inputs, some distance metrics). One outlier squashes every other value into a narrow band (`MinMaxScaler`).
- **Robust scaling.** `(x − median) / IQR`. Outliers no longer set the scale, though they stay in the data as large values (`RobustScaler`).
- **Max-abs scaling.** Divides by the largest absolute value and keeps zeros as zeros, so it works on sparse matrices (`MaxAbsScaler`).
- **Non-linear transforms.** `log1p`, Box-Cox / Yeo-Johnson (`PowerTransformer`) and rank-based `QuantileTransformer` reshape skewed distributions, not just their range. They change the relationships a linear model sees.
- **Row normalization is something else.** `Normalizer` scales each *sample* to unit length (for cosine-like comparisons of TF-IDF or embeddings). It is not a per-feature scaler.
- **Why gradient descent cares.** Features on very different scales make the loss surface a long, narrow valley. The learning rate that is safe for the steep direction is tiny for the flat one, so convergence crawls ([[gradient-descent]]).
- **Why distances care.** Euclidean distance sums squared differences. A feature measured in thousands drowns out one measured in fractions ([[distance-metrics-and-similarity]]).

## When to use / scenarios
- **Always scale for:** k-NN, k-means, SVM (especially RBF), PCA, regularised linear/logistic regression (the penalty treats all weights alike, so their scales must be comparable), neural nets, Gaussian processes.
- **No need for:** decision trees, random forests and gradient-boosted trees ([[gradient-boosting-tabular]]). Scaling neither helps nor hurts them.
- **Outliers or heavy tails** (income, transaction amounts, counts): log or Yeo-Johnson first, or `RobustScaler`, or `QuantileTransformer` when only the rank order matters.
- **Sparse inputs** (bag-of-words, one-hot): `MaxAbsScaler` or no scaling. `StandardScaler` subtracts the mean and destroys sparsity (use `with_mean=False`).
- **Interpretability**: standardised inputs make linear coefficients comparable as "effect per standard deviation" ([[linear-models]]).

## Setup & code
`pip install scikit-learn`. Runs on CPU in a few seconds. Compares scalers across models on the wine dataset, whose feature stds range from 0.12 to 314.

```python
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

X, y = load_wine(return_X_y=True)
print("feature ranges: min std %.3f, max std %.1f" % (X.std(0).min(), X.std(0).max()))

scalers = {"none": None, "standard": StandardScaler(), "minmax": MinMaxScaler(),
           "robust": RobustScaler(), "quantile": QuantileTransformer(n_quantiles=100)}
models = {"kNN": KNeighborsClassifier(), "SVM-rbf": SVC(),
          "LogReg": LogisticRegression(max_iter=5000), "Tree": DecisionTreeClassifier(random_state=0)}

print(f"{'':10}" + "".join(f"{s:>10}" for s in scalers))
for mname, m in models.items():
    row = []
    for s in scalers.values():
        pipe = make_pipeline(s, m) if s is not None else m  # scaler is refit inside each CV fold
        row.append(cross_val_score(pipe, X, y, cv=5).mean())
    print(f"{mname:10}" + "".join(f"{v:10.3f}" for v in row))

# Gradient-based solver: iterations to converge with raw vs standardised inputs
lr = LogisticRegression(solver="lbfgs", max_iter=10000)
print("lbfgs iterations raw:", lr.fit(X, y).n_iter_[0])
print("lbfgs iterations scaled:", lr.fit(StandardScaler().fit_transform(X), y).n_iter_[0])
```

Output (scikit-learn 1.9):
```
feature ranges: min std 0.124, max std 314.0
                none  standard    minmax    robust  quantile
kNN            0.691     0.949     0.950     0.938     0.950
SVM-rbf        0.663     0.983     0.978     0.983     0.978
LogReg         0.961     0.983     0.978     0.989     0.972
Tree           0.888     0.888     0.888     0.888     0.888
lbfgs iterations raw: 3108
lbfgs iterations scaled: 15
```

k-NN and the RBF SVM gain about 30 points from any scaler. Which scaler matters much less than scaling at all. The tree is unchanged. Logistic regression reaches a similar accuracy raw, but needs about 200 times more solver iterations.

For mixed tabular data, scale only the numeric columns:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

pre = ColumnTransformer([
    ("num", StandardScaler(), ["age", "income"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["city"]),
])
pipe = make_pipeline(pre, LogisticRegression(max_iter=5000))
```

## Choosing / trade-offs
- **StandardScaler**: the default. Not bounded, sensitive to outliers through the mean and std.
- **MinMaxScaler**: bounded output, but the most outlier-sensitive. A new value outside the training range maps outside `[0, 1]` (`clip=True` caps it).
- **RobustScaler**: best when a few extreme values exist and should keep their order without setting the scale.
- **QuantileTransformer / PowerTransformer**: when the distribution shape matters (skewed amounts for a linear model or a net). Quantile output is uniform or normal whatever the input, at the cost of distorting distances between values. With little data it overfits the training quantiles.
- **Neural nets**: standardise inputs, then rely on normalisation layers for the hidden activations ([[normalization-layers]]). Regression *targets* with large values also train better standardised; invert the transform on predictions (`TransformedTargetRegressor`).

## Gotchas
- Fitting the scaler on the whole dataset before splitting leaks test-set statistics into training. Put it in a `Pipeline` so it is refit in each CV fold ([[data-leakage-and-validation-splits]]).
- Saving the model without the fitted scaler. Serving then feeds raw values to a model trained on scaled ones and predicts garbage. Persist the whole pipeline.
- Scaling one-hot or binary columns is usually harmless but pointless. Scaling ID-like integers does not make them meaningful features.
- A constant column has std 0. `StandardScaler` leaves it at 0 rather than dividing by zero, but it is a feature to drop.
- `Normalizer` is not a per-column scaler. Using it in place of `StandardScaler` silently changes the problem.
- L1/L2 penalties are unfair on unscaled data: a small-scale feature needs a large weight to matter, and the penalty shrinks large weights hardest, so the model under-uses it. Its coefficients cannot be compared either.
- Drift: the scaler freezes training means and stds. If production data shifts, scaled inputs leave the range the model saw ([[online-learning-and-concept-drift]]).

## Related
- [[feature-engineering]] - the wider preprocessing pipeline this step sits in.
- [[gradient-descent]] - why ill-conditioned inputs slow training.
- [[distance-metrics-and-similarity]] - why distance-based models need comparable scales.
- [[data-leakage-and-validation-splits]] - fitting transforms inside the CV fold.
- [[normalization-layers]] - the in-network counterpart (batch norm, layer norm).
- [[categorical-encoding]] - the non-numeric side of the same ColumnTransformer.

## References
- scikit-learn, Preprocessing data: https://scikit-learn.org/stable/modules/preprocessing.html
- scikit-learn, Compare the effect of different scalers on data with outliers: https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
- scikit-learn, Importance of feature scaling: https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html
