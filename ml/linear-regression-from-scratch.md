---
title: Linear regression from scratch (normal equation, ridge closed form, gradient descent in NumPy)
category: ml
tags: [linear-regression, ordinary-least-squares, normal-equation, ridge, lstsq, gradient-descent, numpy, scikit-learn, from-scratch, ml-basics]
use_cases:
  - "implement linear regression from scratch for learning or an interview"
  - "derive and solve the normal equation without inverting X^T X"
  - "write ridge regression in closed form and match scikit-learn's alpha"
  - "fit linear regression with gradient descent and map the weights back to unscaled features"
status: stable
last_verified: 2026-10-04
sources:
  - https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares
  - https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification
  - https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
---

# Linear regression from scratch (normal equation, ridge closed form, gradient descent in NumPy)

## Summary
Linear regression predicts `ŷ = Xw + b` and picks `w, b` that minimise the squared error. Unlike almost every other model, it has an exact closed-form solution (the normal equation), so the same fit can be computed three ways: a least-squares solve, a ridge-penalised linear solve, and plain gradient descent. Writing all three in about 25 lines of NumPy and matching scikit-learn to 1e-13 shows when to solve and when to iterate, and why the bias, scaling and conditioning matter.

## Key concepts
- **Model and loss.** Add a column of ones to `X` so the bias is just `w₀`. The loss is `L(w) = ‖Xw − y‖² / n` (MSE). It is a convex quadratic bowl, so there is one global minimum (unless columns are collinear).
- **Normal equation.** Setting `∇L = 2Xᵀ(Xw − y)/n = 0` gives `XᵀX w = Xᵀy`. It is the maximum-likelihood fit under Gaussian noise ([[maximum-likelihood-and-map-estimation]]).
- **Solve, don't invert.** `np.linalg.inv(X.T @ X)` squares the condition number of `X` and loses precision. `np.linalg.lstsq` (SVD) or a QR solve works on `X` directly and also handles rank-deficient `X` by returning the minimum-norm solution.
- **Ridge closed form.** Adding `λ‖w‖²` gives `(XᵀX + λI) w = Xᵀy`. The `+λI` makes the matrix invertible even with collinear features. Leave the bias out of the penalty (zero that diagonal entry). It is MAP with a Gaussian prior on `w`.
- **Gradient descent.** `w ← w − η · 2Xᵀ(Xw − y)/n`. Each step costs `O(nd)` rather than the `O(nd² + d³)` of a solve, and it works on mini-batches ([[gradient-descent]]).
- **Scaling.** GD's speed depends on the conditioning of `XᵀX`. Standardise features, fit, then map back: `w_raw = w_s / σ`, `b_raw = b_s − Σ w_s·μ/σ` ([[feature-scaling-and-normalization]]).
- **R².** `1 − SS_res / SS_tot`: the fraction of variance explained compared with predicting the mean. On training data it never decreases as features are added.

## When to use / scenarios
- Learning: the first model where "loss → gradient → solution" can be checked exactly, and the base for [[logistic-regression-from-scratch]].
- Interviews: deriving the normal equation, explaining why not to invert, and the ridge closed form.
- Small, custom fits where a library is overkill or unavailable: calibration curves, sensor offset/gain fits, trend lines inside a script.
- Production: use `sklearn.linear_model.LinearRegression` / `Ridge` / `RidgeCV`, or `statsmodels` when you need standard errors and p-values ([[linear-models]], [[generalized-linear-models]]).

## Setup & code
`pip install numpy scikit-learn`. Runs on CPU in well under a second.

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge

X, y = make_regression(n_samples=500, n_features=5, noise=10, random_state=0)
n, d = X.shape
Xb = np.c_[np.ones(n), X]                      # bias column

# 1) Normal equation, solved stably (never invert X^T X explicitly).
w_lstsq, *_ = np.linalg.lstsq(Xb, y, rcond=None)

# 2) Ridge closed form: (X^T X + lam*I) w = X^T y, bias not penalised.
lam = 10.0
P = lam * np.eye(d + 1); P[0, 0] = 0
w_ridge = np.linalg.solve(Xb.T @ Xb + P, Xb.T @ y)

# 3) Batch gradient descent on mean squared error (features standardised).
mu, sd = X.mean(0), X.std(0)
Xs = np.c_[np.ones(n), (X - mu) / sd]
w = np.zeros(d + 1)
for _ in range(500):
    grad = 2 / n * Xs.T @ (Xs @ w - y)
    w -= 0.1 * grad
w_gd = np.r_[w[0] - (w[1:] * mu / sd).sum(), w[1:] / sd]   # undo scaling

sk = LinearRegression().fit(X, y)
sk_w = np.r_[sk.intercept_, sk.coef_]
print("lstsq  vs sklearn max diff:", f"{np.abs(w_lstsq - sk_w).max():.1e}")
print("GD     vs sklearn max diff:", f"{np.abs(w_gd - sk_w).max():.1e}")
rk = Ridge(alpha=lam).fit(X, y)
print("ridge  vs sklearn max diff:", f"{np.abs(w_ridge - np.r_[rk.intercept_, rk.coef_]).max():.1e}")
r2 = 1 - ((Xb @ w_lstsq - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print("train R^2:", round(r2, 4))
```

Output (numpy 2.5, scikit-learn 1.9):
```
lstsq  vs sklearn max diff: 9.9e-14
GD     vs sklearn max diff: 4.3e-14
ridge  vs sklearn max diff: 4.3e-14
train R^2: 0.991
```

All three agree with scikit-learn to floating-point precision. The ridge match works because scikit-learn's `Ridge(alpha)` minimises the *summed* squared error plus `alpha‖w‖²` with an unpenalised intercept, which is exactly the system solved above. Gradient descent converges fully here because the standardised problem is well conditioned; on raw features with very different scales the same 500 steps would not.

## Choosing / trade-offs
- **Closed form vs GD.** A solve is exact and needs no learning rate, but costs `O(nd² + d³)` and needs `X` in memory. Prefer it up to a few thousand features. Beyond that, or for streaming data, use (mini-batch) GD or `SGDRegressor`.
- **lstsq vs solve on XᵀX.** `lstsq` (SVD) is the safest for ill-conditioned or rank-deficient `X`. `solve(XᵀX + λI, Xᵀy)` is faster and fine once ridge has made the matrix well conditioned. Cholesky is faster still for large `d` with ridge.
- **OLS vs ridge.** OLS is unbiased but its variance explodes with collinear or many features. Ridge trades a little bias for much lower variance. Pick `λ` by cross-validation (`RidgeCV`) ([[bias-variance-and-learning-curves]]).
- **Squared loss vs robust losses.** MSE is dominated by outliers. Huber or absolute loss ([[quantile-regression]]) has no closed form, so it needs an iterative solver.

## Gotchas
- `np.linalg.inv(X.T @ X) @ X.T @ y` is the textbook formula and the wrong code: it fails or returns noise on near-collinear features. Use `lstsq` or `solve`.
- Forgetting the ones column fits a line through the origin. Libraries add the intercept for you (`fit_intercept=True`); from-scratch code must do it itself.
- Penalising the bias in ridge shrinks predictions toward 0 instead of toward the mean of `y`. Zero the bias entry of the penalty, or centre `X` and `y` first.
- Ridge `λ` depends on the loss scale: `alpha` on the summed loss equals `alpha / n` on the mean loss. Mismatches are the usual reason a from-scratch ridge "disagrees" with a library.
- Ridge is not scale invariant: unstandardised features get penalised unequally. Standardise before ridge or lasso.
- GD learning rate too high makes the loss oscillate and blow up to inf/NaN. With standardised features, `η ≈ 0.1` on the mean loss is a safe start.
- High train R² is not evidence of a good model. Check held-out error and the residual plots ([[regression-metrics-and-residual-analysis]]).

## Related
- [[linear-models]] - production OLS, ridge, lasso and elastic net in scikit-learn.
- [[logistic-regression-from-scratch]] - the same linear model with a sigmoid and log-loss, no closed form.
- [[gradient-descent]] - step size, momentum and convergence in general.
- [[feature-scaling-and-normalization]] - why GD on raw features crawls, and how to undo scaling.
- [[regression-metrics-and-residual-analysis]] - evaluating the fit beyond R².
- [[math-for-machine-learning]] - the linear algebra behind the normal equation.

## References
- scikit-learn, Ordinary Least Squares: https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares
- scikit-learn, Ridge regression (objective with `alpha`): https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification
- NumPy `linalg.lstsq`: https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
