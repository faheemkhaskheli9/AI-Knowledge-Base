---
title: Ridge and lasso from scratch (closed-form ridge, coordinate descent and soft-thresholding in NumPy)
category: ml
tags: [ridge, lasso, l1, l2, regularization, coordinate-descent, soft-thresholding, sparsity, feature-selection, linear-regression, numpy, scikit-learn, from-scratch, ml-basics]
use_cases:
  - "implement ridge regression and lasso from scratch"
  - "understand why L1 gives sparse weights and L2 does not"
  - "write coordinate descent with soft-thresholding and check it against scikit-learn"
status: draft
last_verified: 2026-10-04
sources:
  - https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification
  - https://scikit-learn.org/stable/modules/linear_model.html#lasso
  - https://www.jstatsoft.org/article/view/v033i01
  - https://hastie.su.domains/StatLearnSparsity/
---

# Ridge and lasso from scratch (closed-form ridge, coordinate descent and soft-thresholding in NumPy)

## Summary
Ridge and lasso are linear regression with a penalty on the weights. Ridge adds `α‖w‖²`, which has a closed-form solution `(XᵀX + αI)⁻¹Xᵀy`: one line of NumPy. Lasso adds `α‖w‖₁`, which has no closed form but is solved by coordinate descent, where each one-dimensional update is a **soft-threshold** that sets small weights exactly to zero. That zeroing is why lasso selects features and ridge only shrinks them. The from-scratch versions match scikit-learn's `Ridge` to `1e-13` and `Lasso` to `2e-9` on the diabetes dataset.

## Key concepts
- **Ridge objective.** `‖y − Xw − b‖² + α‖w‖²`. Setting the gradient to zero gives `w = (XᵀX + αI)⁻¹Xᵀy` on centred data. Adding `αI` also makes the system invertible when features are collinear.
- **Lasso objective** (scikit-learn's scaling). `(1/2n)‖y − Xw − b‖² + α‖w‖₁`. The `|w|` term has a kink at 0, so there is no closed form and no ordinary gradient there.
- **Coordinate descent.** Optimise one weight at a time with the others fixed. For weight `j`: compute `ρ = xⱼᵀr/n` with feature `j`'s contribution added back into the residual `r`, then `wⱼ = S(ρ, α) / (xⱼᵀxⱼ/n)`.
- **Soft-thresholding.** `S(z, α) = sign(z) · max(|z| − α, 0)`. Any coordinate whose correlation with the residual is below `α` becomes exactly 0.
- **Do not penalise the bias.** Centre `X` and `y`, fit `w`, then recover `b = ȳ − x̄ᵀw`.
- **Geometry.** The L1 ball has corners on the axes, so the loss contours usually touch it at a point where some weights are 0. The L2 ball is round and does not ([[regularization-in-deep-learning]]).
- **Bayesian view.** Ridge is the MAP estimate with a Gaussian prior on `w`; lasso uses a Laplace prior ([[maximum-likelihood-and-map-estimation]]).

## When to use / scenarios
- Learning: the closed-form vs iterative contrast, and the clearest demonstration of what L1 vs L2 does.
- Interviews: "why does lasso give sparse solutions", "derive the ridge solution", "what is soft-thresholding".
- Ridge: many correlated features, `d` close to or above `n`, or whenever plain least squares has huge unstable weights.
- Lasso: you expect few relevant features and want the model to pick them ([[feature-selection]]).
- Not for: non-linear relationships (use trees or add basis functions, see [[generalized-additive-models-and-splines]]), or groups of strongly correlated features where you need all of them (use ridge or elastic net).

## Setup & code
`pip install numpy scikit-learn`. Runs in about 2 seconds.

```python
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso, Ridge


def fit_ridge(X, y, alpha):
    """Minimise ||y - Xw - b||^2 + alpha ||w||^2 (bias not penalised)."""
    xm, ym = X.mean(0), y.mean()
    Xc, yc = X - xm, y - ym                      # centring removes the bias from the penalty
    w = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ yc)
    return w, ym - xm @ w


def soft_threshold(z, g):
    return np.sign(z) * np.maximum(np.abs(z) - g, 0.0)


def fit_lasso(X, y, alpha, iters=1000, tol=1e-8):
    """Minimise (1/2n)||y - Xw - b||^2 + alpha ||w||_1 by cyclic coordinate descent."""
    n, d = X.shape
    xm, ym = X.mean(0), y.mean()
    Xc, yc = X - xm, y - ym
    w = np.zeros(d)
    col_sq = (Xc ** 2).sum(0) / n
    r = yc.copy()                                # residual y - Xw, kept up to date
    for it in range(iters):
        w_old = w.copy()
        for j in range(d):
            r += Xc[:, j] * w[j]                 # remove feature j's contribution
            rho = Xc[:, j] @ r / n
            w[j] = soft_threshold(rho, alpha) / col_sq[j]
            r -= Xc[:, j] * w[j]
        if np.abs(w - w_old).max() < tol:
            break
    return w, ym - xm @ w, it + 1


X, y = load_diabetes(return_X_y=True)            # 442 patients, 10 standardised features

w, b = fit_ridge(X, y, alpha=1.0)
sk = Ridge(alpha=1.0).fit(X, y)
print(f"ridge max |w - sklearn| = {np.abs(w - sk.coef_).max():.1e}, "
      f"bias diff = {abs(b - sk.intercept_):.1e}")

w, b, iters = fit_lasso(X, y, alpha=0.5)
sk = Lasso(alpha=0.5, tol=1e-10, max_iter=100000).fit(X, y)
print(f"lasso max |w - sklearn| = {np.abs(w - sk.coef_).max():.1e} after {iters} sweeps")

print("alpha  lasso nonzero  ridge nonzero  ridge min|w|")
for a in (0.01, 0.1, 0.5, 1.0, 5.0):
    wl = fit_lasso(X, y, a)[0]
    wr = fit_ridge(X, y, 2 * len(y) * a)[0]   # same penalty scale as the (1/2n) lasso loss
    print(f"{a:<6} {np.count_nonzero(wl):>8} {np.count_nonzero(wr):>14} {np.abs(wr).min():>13.2f}")

# correlated features: lasso picks one, ridge splits the weight
rng = np.random.default_rng(0)
z = rng.normal(size=500)
Xd = np.column_stack([z, z + 0.01 * rng.normal(size=500), rng.normal(size=500)])
yd = 3 * z + 0.1 * rng.normal(size=500)
print("duplicated feature  ridge:", fit_ridge(Xd, yd, 10.0)[0].round(2),
      " lasso:", fit_lasso(Xd, yd, 0.1)[0].round(2))
```

Output (numpy 2.5, scikit-learn 1.9):
```
ridge max |w - sklearn| = 1.1e-13, bias diff = 0.0e+00
lasso max |w - sklearn| = 2.4e-09 after 24 sweeps
alpha  lasso nonzero  ridge nonzero  ridge min|w|
0.01         10             10          1.77
0.1           7             10          0.65
0.5           4             10          0.15
1.0           3             10          0.08
5.0           0             10          0.02
duplicated feature  ridge: [1.49 1.48 0.  ]  lasso: [2.81 0.09 0.  ]
```

Both solvers agree with scikit-learn to rounding error, and coordinate descent converges in 24 sweeps over the 10 features. As `α` grows, lasso drops features one by one (10, 7, 4, 3, then none), while ridge keeps all 10 and only shrinks them towards 0. With two near-duplicate copies of the true feature, ridge splits the weight evenly (1.49 and 1.48) and lasso puts almost all of it on one (2.81). Which copy lasso picks is arbitrary, so do not read lasso's choice among correlated features as importance.

## Choosing / trade-offs
- **Ridge vs lasso vs elastic net.** Ridge for prediction with many correlated features. Lasso for a sparse, readable model. Elastic net (`l1_ratio` between 0 and 1) when features come in correlated groups and you still want sparsity ([[linear-models]]).
- **Closed form vs iterative.** The ridge solve costs `O(nd² + d³)`: fine up to a few thousand features. Beyond that, use iterative solvers (`solver="sag"`, `"sparse_cg"`) or SGD.
- **Choosing α.** Use `RidgeCV` / `LassoCV`, which compute a whole regularisation path cheaply with warm starts. Search on a log scale ([[hyperparameter-tuning]]).
- **Lasso for selection, then refit.** Lasso shrinks the surviving weights too. Refitting plain least squares on the selected features (the "relaxed lasso") removes that bias.

## Gotchas
- **Scale the features first.** The penalty treats all weights equally, so a feature measured in thousands gets penalised far less than one measured in fractions ([[feature-scaling-and-normalization]]). The diabetes features here are already standardised.
- Check the library's loss scaling before copying an `α`. scikit-learn's `Ridge` uses the plain sum of squares; `Lasso` uses `1/(2n)` times it. The same number means very different strengths in the two.
- Do not penalise the intercept. Centre the data or fit the bias separately; penalising it pulls predictions towards 0.
- Lasso picks at most `n` features when `d > n`, and with correlated features its choice is unstable across bootstrap samples. Check stability before reporting "the selected features".
- Coordinate descent's `tol` is on the change in weights, not the loss. Too loose a tolerance stops early with slightly wrong weights and visible gaps vs another solver.
- Fit the scaler and choose `α` inside cross-validation folds, not on the full data, or the score is leaked ([[data-leakage-and-validation-splits]]).

## Related
- [[linear-regression-from-scratch]] - the unpenalised model: normal equations and gradient descent.
- [[linear-models]] - Ridge, Lasso, ElasticNet and logistic regression in scikit-learn.
- [[feature-selection]] - lasso next to filter and wrapper methods.
- [[regularization-in-deep-learning]] - weight decay, the L2 penalty in neural networks.
- [[bias-variance-and-learning-curves]] - what the penalty strength trades off.
- [[svm-from-scratch]] - another model trained with a penalised convex objective.

## References
- scikit-learn User Guide, Ridge regression: https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification
- scikit-learn User Guide, Lasso: https://scikit-learn.org/stable/modules/linear_model.html#lasso
- Friedman, Hastie and Tibshirani, Regularization Paths for Generalized Linear Models via Coordinate Descent (J. Stat. Software, 2010): https://www.jstatsoft.org/article/view/v033i01
- Hastie, Tibshirani and Wainwright, Statistical Learning with Sparsity: https://hastie.su.domains/StatLearnSparsity/
