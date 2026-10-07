---
title: Linear and logistic regression (Ridge, Lasso, ElasticNet)
category: ml
tags: [linear-regression, logistic-regression, ridge, lasso, elasticnet, regularization, coefficients, scikit-learn, baseline]
use_cases:
  - "build a fast, explainable baseline for churn, credit or price prediction"
  - "explain to a regulator or manager which inputs push a prediction up or down"
  - "predict house or product prices from a handful of numeric features"
  - "select a small set of useful features out of hundreds with Lasso"
status: stable
last_verified: 2026-10-03
sources:
  - https://scikit-learn.org/stable/modules/linear_model.html
  - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
  - https://www.statlearning.com/
---

# Linear and logistic regression (Ridge, Lasso, ElasticNet)

## Summary
Linear models predict with a weighted sum of the features: `y = w·x + b` for regression, and `sigmoid(w·x + b)` for logistic regression (binary classification; softmax for multi-class). They train in seconds, need little data, and each weight says how much a feature moves the prediction, which makes them the default baseline and the default choice when a model must be explained. Regularisation (Ridge = L2, Lasso = L1, ElasticNet = both) keeps the weights from overfitting.

## Key concepts
- **Ordinary least squares (OLS).** Minimises squared error; has a closed-form solution. `LinearRegression` in scikit-learn.
- **Logistic regression.** Fits log-odds as a linear function, trained with cross-entropy (log loss). Despite the name it is a classifier, and its probabilities are usually well calibrated.
- **Ridge (L2).** Adds `alpha * sum(w²)`. Shrinks all weights, stabilises correlated features, never sets a weight to exactly zero.
- **Lasso (L1).** Adds `alpha * sum(|w|)`. Drives weak weights to exactly zero, so it doubles as feature selection.
- **ElasticNet.** Mix of L1 and L2 (`l1_ratio`); keeps groups of correlated features together where Lasso would pick one at random.
- **Regularisation strength.** `alpha` in Ridge/Lasso (bigger = simpler); `C` in `LogisticRegression` is the inverse (smaller = simpler). Tune it with the built-in CV variants (`RidgeCV`, `LassoCV`, `LogisticRegressionCV`).
- **Scaling matters.** Penalties compare weight sizes, so features must be on the same scale (`StandardScaler`) or the penalty is unfair.
- **Non-linearity by features.** A linear model can fit curves and interactions if you add them as features (`PolynomialFeatures`, `SplineTransformer`, log transforms); see [[feature-engineering]].
- **Interpretation.** With standardised inputs, a coefficient is the change in prediction (or log-odds) per one standard deviation of that feature, holding the others fixed. `exp(coef)` in logistic regression is an odds ratio.

## When to use / scenarios
- First model on any tabular problem: it sets the bar that [[gradient-boosting-tabular]] has to beat.
- Credit scoring, insurance pricing, clinical risk scores: regulators and clinicians want per-feature weights and monotone, explainable behaviour.
- High-dimensional sparse data (bag-of-words text, one-hot categories with thousands of levels): logistic regression on TF-IDF is still a strong text classifier, see [[nlp-classic-tasks]].
- Small data (hundreds of rows): fewer parameters, less overfitting.
- Not when: interactions and thresholds dominate the signal (use trees/boosting), or the input is images/audio/raw text sequences (use deep learning).

## Setup & code
```bash
pip install scikit-learn
```
```python
import numpy as np
from sklearn.datasets import load_breast_cancer, make_regression
from sklearn.linear_model import LassoCV, LogisticRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Regression: 50 features, only 5 matter.
X, y = make_regression(n_samples=500, n_features=50, n_informative=5,
                       noise=10, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)
for name, reg in [("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13))),
                  ("lasso", LassoCV(cv=5, random_state=0))]:
    model = make_pipeline(StandardScaler(), reg).fit(X_tr, y_tr)
    nonzero = int((model[-1].coef_ != 0).sum())
    print(f"{name}: MAE={mean_absolute_error(y_te, model.predict(X_te)):.1f}, "
          f"non-zero weights={nonzero}/50")

# Classification: logistic regression with readable odds ratios.
data = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target,
                                          stratify=data.target, random_state=0)
clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
clf.fit(X_tr, y_tr)
print("ROC-AUC:", round(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]), 3))
coefs = clf[-1].coef_[0]
for i in np.argsort(-np.abs(coefs))[:3]:
    print(f"{data.feature_names[i]}: odds x{np.exp(coefs[i]):.2f} per +1 std")
```
On this data Lasso keeps 19 of 50 weights and gets a slightly lower MAE than Ridge, which keeps all 50; the odds ratios are for the positive class (benign), so values below 1 push toward malignant.

## Choosing / trade-offs
- **Ridge vs Lasso vs ElasticNet.** Ridge when most features carry some signal or are correlated; Lasso when you expect few relevant features and want a sparse model; ElasticNet when both (correlated groups plus many irrelevant features).
- **Linear vs boosting.** Boosting usually wins on accuracy for tabular data with interactions; linear wins on explainability, training speed, extrapolation stability and tiny datasets. Check the gap on validation data before giving up explainability.
- **scikit-learn vs statsmodels.** scikit-learn for prediction pipelines; `statsmodels` when you need p-values, confidence intervals and classical inference on the coefficients.
- **Solver** (`LogisticRegression`): default `lbfgs` is fine for most; `saga` for L1/ElasticNet on large sparse data; `liblinear` for small data with L1.

## Gotchas
- Forgetting to scale makes the penalty favour features with large units, and makes coefficients incomparable.
- `LogisticRegression` is regularised by default (`C=1.0`); results differ from textbook unpenalised logistic regression. Set `C` deliberately.
- Correlated features make individual coefficients unstable (they can flip sign between samples) even when predictions are fine; do not read causation into them.
- A convergence warning means increase `max_iter` or scale the features, not ignore it.
- Linear models extrapolate linearly forever; a price model can predict negative prices for out-of-range inputs.
- One-hot encoding every level plus an intercept is collinear; regularisation hides this, plain OLS does not (use `drop="first"` for unpenalised models).

## Related
- [[ml-fundamentals]] - regularisation and the bias-variance trade-off these models illustrate.
- [[feature-engineering]] - scaling, encoding and creating non-linear features.
- [[classic-ml-scikit-learn]] - pipelines and the wider scikit-learn workflow.
- [[model-interpretability]] - explaining models whose weights are not directly readable.
- [[neural-network-fundamentals]] - a neural network is stacked logistic regressions with non-linearities.
- [[lda-from-scratch]] - linear discriminant analysis built by hand as classifier and Fisher projection.

## References
- scikit-learn linear models: https://scikit-learn.org/stable/modules/linear_model.html
- LogisticRegression API: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- *An Introduction to Statistical Learning*, chapters 3, 4 and 6: https://www.statlearning.com/
