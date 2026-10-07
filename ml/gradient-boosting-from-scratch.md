---
title: Gradient boosting from scratch (fitting residuals, shrinkage, staged predictions)
category: ml
tags: [gradient-boosting, boosting, residuals, shrinkage, learning-rate, early-stopping, decision-tree, ensemble, numpy, scikit-learn, from-scratch, ml-basics]
use_cases:
  - "implement gradient boosting from scratch for learning or an interview"
  - "understand why each boosting tree fits the residuals of the previous ones"
  - "see how learning rate and number of trees trade off, and why boosting needs early stopping"
  - "reproduce scikit-learn's GradientBoostingRegressor exactly with a short loop"
status: stable
last_verified: 2026-10-04
sources:
  - https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting
  - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
  - https://jerryfriedman.su.domains/ftp/trebst.pdf
---

# Gradient boosting from scratch (fitting residuals, shrinkage, staged predictions)

## Summary
Gradient boosting builds a strong model by adding many small trees one at a time, each trained to correct what the current ensemble still gets wrong. For squared error, "what it gets wrong" is exactly the residual `y − F(x)`, which is also the negative gradient of the loss, hence the name. A 20-line loop around scikit-learn's `DecisionTreeRegressor` reproduces `GradientBoostingRegressor` to the last bit, and makes the learning-rate / number-of-trees trade-off and the need for early stopping easy to see.

## Key concepts
- **Start from a constant.** `F₀(x) = mean(y)`, the best single number under squared error.
- **Fit the negative gradient.** At step `m`, compute `rᵢ = −∂L/∂F(xᵢ)`. For `L = ½(y − F)²` that is `yᵢ − F(xᵢ)`, the residual. Fit a small regression tree `hₘ` to `(X, r)`.
- **Shrinkage.** Update `F ← F + η·hₘ` with learning rate `η < 1`. Small steps mean more trees but much better generalisation (Friedman, 2001).
- **Weak learners.** Shallow trees (depth 3 to 6) are the norm. Depth sets the order of feature interactions each tree can capture; boosting adds them up.
- **Other losses.** Only the residual line changes. Log-loss for classification: fit `y − sigmoid(F)` and then reset each leaf to a Newton step. Absolute error: fit `sign(y − F)`. XGBoost and LightGBM use the second derivative too ([[gradient-boosting-tabular]]).
- **Boosting vs bagging.** A random forest trains deep trees independently and averages them to cut variance. Boosting trains shallow trees in sequence to cut bias, and can overfit if run too long ([[ensemble-methods]]).

## When to use / scenarios
- Learning: the cleanest way to see "gradient descent in function space".
- Interviews: "how does gradient boosting work", "what does the learning rate do", "boosting vs random forest".
- Debugging a production GBM: staged predictions show where validation loss turns up and how many trees you actually need.
- For real tabular work use LightGBM, XGBoost, CatBoost or scikit-learn's `HistGradientBoosting*`, which bin features and are orders of magnitude faster ([[gradient-boosting-tabular]]).

## Setup & code
`pip install numpy scikit-learn`. Runs in a few seconds on CPU. The tree learner is borrowed from scikit-learn; [[decision-tree-from-scratch]] shows how to write it.

```python
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


class GBRegressor:
    """Gradient boosting for squared error: each tree fits the current residuals
    (the negative gradient of 0.5 * (y - F)^2)."""

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n, self.lr, self.depth = n_estimators, learning_rate, max_depth

    def fit(self, X, y):
        self.f0 = y.mean()                          # best constant model
        F = np.full(len(y), self.f0)
        self.trees = []
        rng = np.random.RandomState(0)              # one shared stream, like sklearn's
        for _ in range(self.n):
            residual = y - F                        # negative gradient
            tree = DecisionTreeRegressor(max_depth=self.depth, random_state=rng).fit(X, residual)
            F += self.lr * tree.predict(X)          # shrinkage
            self.trees.append(tree)
        return self

    def staged_predict(self, X):
        F = np.full(len(X), self.f0)
        for tree in self.trees:
            F = F + self.lr * tree.predict(X)
            yield F

    def predict(self, X):
        *_, F = self.staged_predict(X)
        return F


X, y = load_diabetes(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

ours = GBRegressor(n_estimators=300).fit(Xtr, ytr)
sk = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=3,
                               random_state=0).fit(Xtr, ytr)
print("max |ours - sklearn| on test:", np.abs(ours.predict(Xte) - sk.predict(Xte)).max())

mse = lambda a, b: np.mean((a - b) ** 2)
print("baseline (predict mean) test MSE:", round(mse(np.full(len(yte), ytr.mean()), yte), 1))
for lr in (1.0, 0.1, 0.01):
    m = GBRegressor(n_estimators=300, learning_rate=lr).fit(Xtr, ytr)
    test = [mse(F, yte) for F in m.staged_predict(Xte)]
    train = mse(m.predict(Xtr), ytr)
    best = int(np.argmin(test))
    print(f"lr={lr:<5} train MSE={train:7.1f}  test MSE @300={test[-1]:7.1f}  "
          f"best test MSE={test[best]:7.1f} at {best + 1} trees")
```

Output (numpy 2.5, scikit-learn 1.9):
```
max |ours - sklearn| on test: 0.0
baseline (predict mean) test MSE: 5101.5
lr=1.0   train MSE=    0.0  test MSE @300= 5330.9  best test MSE= 4120.4 at 2 trees
lr=0.1   train MSE=  148.2  test MSE @300= 3967.3  best test MSE= 3263.4 at 15 trees
lr=0.01  train MSE= 1663.0  test MSE @300= 3372.9  best test MSE= 3293.1 at 194 trees
```

The loop matches scikit-learn exactly. The learning-rate sweep shows the core trade-off on this small (309-row) dataset. With `η = 1` the ensemble memorises the training set (train MSE 0.0) and after 300 trees is worse on test than just predicting the mean. With `η = 0.1` the best test MSE comes at 15 trees and then climbs by about 700 as the model overfits. With `η = 0.01` it needs about 190 trees to reach a similar best score but degrades much more slowly afterwards, so the number of trees matters less. The "best at N trees" figures here are read off the test set for illustration; in real use pick N on a validation split ([[data-leakage-and-validation-splits]]).

## Choosing / trade-offs
- **Learning rate vs trees.** Halving `η` roughly doubles the trees needed. Pick the smallest `η` your training-time budget allows (0.05 to 0.1 is common), then set the number of trees with early stopping (`n_iter_no_change` in scikit-learn, `early_stopping_rounds` in XGBoost/LightGBM).
- **Tree depth.** Depth 1 (stumps) gives an additive model with no interactions. Depth 3 to 6 covers most tabular problems. Deeper trees overfit faster per round.
- **Subsampling (stochastic gradient boosting).** Fitting each tree on a random 50 to 80% of rows (`subsample`) adds noise that usually improves generalisation and speeds training.
- **Exact vs histogram splits.** This loop searches every threshold. Histogram-based libraries bin each feature into about 256 buckets, which is far faster on large data with little accuracy loss.

## Gotchas
- Boosting overfits if you just add trees. Always watch a validation curve or use early stopping; unlike a random forest, more trees is not "free".
- `F` must be updated with the *shrunk* tree output (`η·h`), both in training and in prediction, or the two drift apart.
- The tree is fit to residuals, not to `y`. Fitting each tree to `y` gives a weird average of similar trees, not boosting.
- To match scikit-learn bit for bit you need the same random stream: it passes one `RandomState` to every tree, which breaks ties between equally good splits. Giving each tree `random_state=0` instead matched for the first 10 trees here and then drifted by up to 26 units.
- `criterion="friedman_mse"` is deprecated for `DecisionTreeRegressor` in scikit-learn 1.9 (it was always identical to `squared_error`); leave the default.
- For classification, do not fit trees to `y − p` and stop: the leaf values need a Newton step (`Σr / Σp(1−p)`), otherwise convergence is slow. Library implementations do this for you.
- Feature scaling is not needed for tree-based boosting, but target scale matters: the learning rate acts on the target's units.

## Related
- [[gradient-boosting-tabular]] - XGBoost, LightGBM and CatBoost for real tabular work.
- [[decision-tree-from-scratch]] - the base learner, written by hand.
- [[ensemble-methods]] - boosting next to bagging, random forests and stacking.
- [[decision-trees-and-random-forests]] - the bagging alternative.
- [[bias-variance-and-learning-curves]] - boosting reduces bias; too many rounds add variance.
- [[gradient-descent]] - the same idea in parameter space instead of function space.

## References
- scikit-learn, Gradient boosting: https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting
- scikit-learn GradientBoostingRegressor API: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
- Friedman, "Greedy Function Approximation: A Gradient Boosting Machine" (2001): https://jerryfriedman.su.domains/ftp/trebst.pdf
