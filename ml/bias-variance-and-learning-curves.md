---
title: Bias-variance trade-off and learning curves
category: ml
tags: [bias-variance, learning-curve, validation-curve, overfitting, underfitting, diagnostics, model-capacity, scikit-learn]
use_cases:
  - "decide whether collecting more data will improve my model"
  - "tell whether a model is underfitting or overfitting and what to change"
  - "explain why a very flexible model does worse on new data than a simpler one"
  - "pick model complexity (tree depth, polynomial degree, regularization) from a plot"
status: stable
last_verified: 2026-10-04
sources:
  - https://scikit-learn.org/stable/modules/learning_curve.html
  - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html
  - https://www.statlearning.com/
---

# Bias-variance trade-off and learning curves

## Summary
A model's expected error on new data splits into three parts: **bias²** (error from wrong assumptions, so the model is too simple), **variance** (error from being sensitive to which training sample it saw, so it is too flexible), and **irreducible noise**. Learning curves plot training and validation score against training-set size. They are the cheapest way to tell which part dominates, and so whether to get more data, use a bigger model, or regularise.

## Key concepts
- **Decomposition (squared loss).** `E[(y − ŷ)²] = bias² + variance + σ²`. Bias² is how far the *average* model (over many training sets) is from the truth. Variance is how much individual fits scatter around that average. σ² is label noise, which no model removes.
- **Capacity.** Raising complexity (degree, depth, number of features, smaller regularisation) usually lowers bias and raises variance. The best validation error sits in between.
- **Learning curve** (x = number of training samples). A high-bias model has train and validation scores that converge quickly to a *low* plateau, so more data will not help. A high-variance model has a large train/validation gap that shrinks as data grows, so more data helps.
- **Validation curve** (x = one hyperparameter). It shows train and validation score as capacity changes, and where the gap opens up. [[ml-fundamentals]] shows one for tree depth.
- **The gap is variance, the plateau is bias.** Read the gap between the curves, and how far the validation curve is from the score you need.
- **Deep networks break the U-shape.** Very over-parameterised models can show double descent ([[generalization-in-deep-learning]]). The learning-curve diagnosis still works in practice.

## When to use / scenarios
- Before paying for more labelling (medical images, annotated documents): a learning curve that has flattened says extra labels are wasted.
- A model plateaus at a disappointing score. A small gap means it is underfitting: add features, use a more flexible model or reduce regularisation.
- Training accuracy is near 100% but validation is much lower. That is overfitting: regularise, simplify, augment ([[data-augmentation]]) or get more data.
- Explaining to stakeholders why "the most accurate model on training data" is not the one to ship.
- Not a substitute for a proper held-out estimate ([[model-selection-and-comparison]]). It is a diagnostic, not a final score.

## Setup & code
```bash
pip install "scikit-learn>=1.4" numpy
```
```python
import numpy as np
from sklearn.model_selection import learning_curve, ShuffleSplit
from sklearn.datasets import load_digits
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# 1) Learning curves: train vs validation score as the training set grows.
X, y = load_digits(return_X_y=True)
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
for name, model in [("GaussianNB", GaussianNB()), ("SVC(rbf)", SVC(gamma=0.001))]:
    sizes, tr, va = learning_curve(model, X, y, cv=cv,
                                   train_sizes=[0.1, 0.3, 0.6, 1.0], n_jobs=-1)
    for n, a, b in zip(sizes, tr.mean(1), va.mean(1)):
        print(f"{name:10s} n={n:4d} train={a:.3f} val={b:.3f} gap={a - b:.3f}")

# 2) Empirical bias^2 / variance: refit on many fresh training sets.
rng = np.random.default_rng(0)
f = lambda x: np.sin(2 * np.pi * x)
x_test = np.linspace(0, 1, 200)[:, None]
for degree in [1, 3, 12]:
    preds = []
    for _ in range(200):
        x = rng.uniform(0, 1, (30, 1))
        y_ = f(x[:, 0]) + rng.normal(0, 0.3, 30)
        m = make_pipeline(PolynomialFeatures(degree), LinearRegression()).fit(x, y_)
        preds.append(m.predict(x_test))
    preds = np.array(preds)
    bias2 = ((preds.mean(0) - f(x_test[:, 0])) ** 2).mean()
    var = preds.var(0).mean()
    print(f"degree={degree:2d} bias^2={bias2:.3f} variance={var:.3f} noise=0.090")
```
Output (scikit-learn 1.9):

| Model | n=143 train / val | n=1437 train / val |
|---|---|---|
| GaussianNB | 0.945 / 0.761 | 0.861 / 0.856 |
| SVC (rbf) | 1.000 / 0.926 | 0.999 / 0.993 |

GaussianNB converges to about 0.86 with almost no gap. That is high bias, and more digits will not help, so a better model will. SVC's gap shrinks from 0.074 to 0.006 as data grows, and it is the model to keep.

| Degree | bias² | variance |
|---|---|---|
| 1 | 0.200 | 0.023 |
| 3 | 0.005 | 0.017 |
| 12 | 0.125 | 26.0 |

With 30 points, degree 1 underfits the sine and degree 12 explodes in variance (its average fit is also off, because of wild swings at the edges). Degree 3 minimises the sum.

## Choosing / trade-offs
| Learning curve shows | Diagnosis | Do |
|---|---|---|
| Both scores low, small gap | High bias | More features, more flexible model, less regularisation, train longer (neural nets) |
| Train high, val low, gap shrinking with n | High variance | More data, augmentation, regularisation, simpler model, ensembling/bagging |
| Both high, small gap | Good fit | Stop. Check the held-out test set |
| Val flat while train keeps rising | Variance, and more data has stopped helping | Regularise or change features, not more of the same data |

- **Bagging lowers variance, boosting lowers bias** ([[ensemble-methods]]). Pick the ensemble that attacks your problem.
- **Cost.** `learning_curve` refits `n_splits × len(train_sizes)` times. Use 3 to 5 sizes and a fast model or a subsample for big data.

## Gotchas
- **Noise is not bias.** If labels are noisy (inter-annotator disagreement), every model plateaus below 100%. Estimate the noise ceiling (e.g. human agreement) before chasing bias ([[label-noise-and-data-cleaning]]).
- **Leakage hides variance.** Preprocessing fitted on all data, or duplicate rows across splits, makes the validation curve look too good. Keep preprocessing in a pipeline ([[data-leakage-and-validation-splits]]).
- **Wrong split for the data.** Time series or grouped data need `TimeSeriesSplit` / `GroupKFold` inside `learning_curve`. Random shuffling understates variance.
- **Small validation folds are noisy.** Plot the standard deviation across folds as well as the mean, or use more splits, before reading a 0.01 difference.
- **Score direction.** scikit-learn's error scorers are negated (`neg_mean_squared_error`). A curve going "up" means less error.

## Related
- [[ml-fundamentals]] - overfitting, underfitting and a validation curve over tree depth.
- [[generalization-in-deep-learning]] - double descent and why huge networks still generalise.
- [[regularization-in-deep-learning]] - the variance-reducing toolbox for neural nets.
- [[ensemble-methods]] - bagging for variance, boosting for bias.
- [[hyperparameter-tuning]] - searching the capacity knob once the diagnosis is clear.
- [[model-selection-and-comparison]] - honest final estimates after the diagnosis.

## References
- scikit-learn, validation and learning curves: https://scikit-learn.org/stable/modules/learning_curve.html
- scikit-learn `learning_curve` API: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html
- James, Witten, Hastie, Tibshirani, *An Introduction to Statistical Learning*, section 2.2.2 (the bias-variance trade-off): https://www.statlearning.com/
