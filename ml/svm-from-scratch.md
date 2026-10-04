---
title: Linear SVM from scratch (hinge loss, soft margin, subgradient descent in NumPy)
category: ml
tags: [svm, support-vector-machine, hinge-loss, soft-margin, max-margin, subgradient-descent, regularization, numpy, scikit-learn, from-scratch, ml-basics]
use_cases:
  - "implement a linear SVM from scratch for learning or an interview"
  - "understand the hinge loss, the margin and what C controls in an SVM"
  - "train an SVM in the primal with subgradient descent and match scikit-learn"
status: draft
last_verified: 2026-10-04
sources:
  - https://scikit-learn.org/stable/modules/svm.html
  - https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
  - https://home.ttic.edu/~nati/Publications/PegasosMPB.pdf
---

# Linear SVM from scratch (hinge loss, soft margin, subgradient descent in NumPy)

## Summary
A linear support vector machine finds the hyperplane `w·x + b = 0` that separates two classes with the widest margin, while paying a penalty `C` for each point that sits inside the margin or on the wrong side. In the primal form this is just L2-regularised hinge loss, which you can minimise with subgradient descent in about ten lines of NumPy. On a standardised breast-cancer dataset the from-scratch model reaches the same objective value and the same test predictions as scikit-learn's `SVC(kernel="linear")` for small `C`.

## Key concepts
- **Labels are ±1.** The maths uses `y ∈ {−1, +1}` so that `y·(w·x + b)` (the functional margin) is positive exactly when a point is classified correctly.
- **Soft-margin objective.** `½‖w‖² + C·Σ max(0, 1 − yᵢ(w·xᵢ + b))`. The first term widens the margin (the margin width is `2/‖w‖`); the second is the **hinge loss**, which is zero for points at least one unit past the boundary.
- **C is inverse regularisation.** Small `C` tolerates margin violations and gives a wide margin with many support vectors. Large `C` punishes every violation and gives a narrow margin that chases outliers. It plays the same role as `1/λ` in ridge or `C` in scikit-learn's `LogisticRegression`.
- **Subgradient.** Hinge loss has a kink at margin 1, so it has no gradient there. A valid subgradient is `w − C·Σ yᵢxᵢ` over the points with margin `< 1`, and `−C·Σ yᵢ` for `b`. Points with margin `≥ 1` contribute nothing.
- **Support vectors.** Only points on or inside the margin (`yᵢ(w·xᵢ + b) ≤ 1`) shape the solution. Delete every other point and the hyperplane does not move.
- **Hinge vs log-loss.** Logistic regression uses `log(1 + e^(−margin))`, a smooth version of the hinge that never reaches zero, so every point contributes and the model outputs probabilities. The SVM gives a sparser solution and only a score ([[logistic-regression-from-scratch]]).

## When to use / scenarios
- Learning: the hinge loss, the margin and `C` in their simplest setting before kernels.
- Interviews: "write the SVM loss and its gradient", "what is a support vector", "what does C do".
- High-dimensional sparse data (bag-of-words text, TF-IDF) where a linear SVM is a fast, strong baseline ([[nlp-classic-tasks]]).
- Not for: non-linear boundaries on low-dimensional data (use an RBF kernel SVM or gradient boosting, see [[svm-knn-naive-bayes]] and [[gradient-boosting-tabular]]), or when you need calibrated probabilities ([[probability-calibration]]).

## Setup & code
`pip install numpy scikit-learn`. Runs in a few seconds on CPU.

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def objective(w, b, X, y, C):
    margins = y * (X @ w + b)
    return 0.5 * w @ w + C * np.maximum(0, 1 - margins).sum()


def fit_linear_svm(X, y, C=1.0, epochs=2000, lr0=0.1):
    """Soft-margin linear SVM in the primal: subgradient descent on hinge loss.
    y must be in {-1, +1}."""
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for t in range(1, epochs + 1):
        margins = y * (X @ w + b)
        viol = margins < 1                          # points inside the margin or misclassified
        gw = w - C * (y[viol, None] * X[viol]).sum(0)
        gb = -C * y[viol].sum()
        lr = lr0 / np.sqrt(t) / (1 + C * n / 100)    # decaying step; scale down for large C*n
        w, b = w - lr * gw, b - lr * gb
    return w, b


X, y01 = load_breast_cancer(return_X_y=True)
y = np.where(y01 == 1, 1, -1)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
sc = StandardScaler().fit(Xtr)
Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

for C in (0.01, 0.1, 1.0):
    w, b = fit_linear_svm(Xtr, ytr, C=C)
    sk = SVC(kernel="linear", C=C).fit(Xtr, ytr)
    ours = np.sign(Xte @ w + b)
    n_sv = (ytr * (Xtr @ w + b) <= 1 + 1e-3).sum()
    print(f"C={C:<5} acc ours={np.mean(ours == yte):.3f} sklearn={sk.score(Xte, yte):.3f} "
          f"agree={np.mean(ours == sk.predict(Xte)):.3f} "
          f"obj ours={objective(w, b, Xtr, ytr, C):.2f} sklearn={objective(sk.coef_[0], sk.intercept_[0], Xtr, ytr, C):.2f} "
          f"margin-SVs ours={n_sv} sklearn={sk.n_support_.sum()} |w|={np.linalg.norm(w):.2f}")
```

Output (numpy 2.5, scikit-learn 1.9):
```
C=0.01  acc ours=0.959 sklearn=0.959 agree=1.000 obj ours=0.65 sklearn=0.65 margin-SVs ours=92 sklearn=92 |w|=0.66
C=0.1   acc ours=0.959 sklearn=0.959 agree=1.000 obj ours=2.89 sklearn=2.89 margin-SVs ours=40 sklearn=44 |w|=1.37
C=1.0   acc ours=0.959 sklearn=0.953 agree=0.971 obj ours=15.73 sklearn=14.20 margin-SVs ours=15 sklearn=27 |w|=3.29
```

For `C = 0.01` and `0.1` the from-scratch solver reaches the same objective as scikit-learn's exact dual solver (libsvm) and predicts the same label for every test point. Raising `C` from 0.01 to 1 shrinks the margin (`‖w‖` grows from 0.66 to 3.29) and cuts the support vectors from 92 to 27. At `C = 1` subgradient descent has not converged after 2000 steps (objective 15.73 vs the optimum 14.20). This is the main weakness of the method: the larger `C·n` is, the stiffer the problem and the slower plain subgradient descent becomes. Here the extra error happened to cost nothing on the test set, but do not count on that.

## Choosing / trade-offs
- **Primal subgradient vs dual solver.** Subgradient descent (and its stochastic form, Pegasos) scales to millions of rows and is easy to write, but converges slowly near the optimum. Coordinate descent in the dual (liblinear, `LinearSVC`) or SMO (libsvm, `SVC`) gives exact solutions. Use `LinearSVC` or `SGDClassifier(loss="hinge")` in practice.
- **Linear vs kernel SVM.** The primal form here only gives linear boundaries. Kernels (RBF, polynomial) need the dual, where the data enters only through dot products, and cost `O(n²)` memory, so they stop being practical beyond tens of thousands of rows ([[kernel-methods-and-density-estimation]]).
- **SVM vs logistic regression.** Similar accuracy in most linear problems. Choose logistic regression when you need probabilities or a smooth loss for a gradient-based pipeline; choose the SVM for a sparse set of support vectors or a max-margin bias on separable data.
- **Tuning C.** Search on a log grid (`0.001` to `100`) with cross-validation ([[hyperparameter-tuning]]).

## Gotchas
- Labels must be `−1/+1`. Using `0/1` makes every class-0 point contribute zero to the margin term and silently trains a wrong model.
- Standardise features first. The margin is measured in raw feature units, so an unscaled feature dominates `‖w‖` ([[feature-scaling-and-normalization]]).
- Do not regularise the bias. The `½‖w‖²` term covers `w` only; folding `b` into `w` with a column of ones changes the problem.
- A fixed step size makes subgradient descent oscillate around the optimum forever. Use a decaying step (`∝ 1/√t` or `1/(λt)` as in Pegasos) or average the iterates.
- `C` is a sum-of-losses weight, so its best value depends on the number of rows. Formulations with a mean loss use `λ = 1/(C·n)` instead. Check which one a library uses before copying a value.
- `decision_function` values are distances, not probabilities. Wrap the model in `CalibratedClassifierCV` if you need probabilities ([[probability-calibration]]).
- The "support vector" count from a not-fully-converged primal solver depends on the tolerance you use for "margin ≤ 1". Compare counts only after convergence.

## Related
- [[svm-knn-naive-bayes]] - kernel SVMs in scikit-learn next to k-NN and naive Bayes.
- [[logistic-regression-from-scratch]] - the same linear model with log-loss instead of hinge loss.
- [[kernel-methods-and-density-estimation]] - the kernel trick that turns this into a non-linear model.
- [[linear-models]] - where linear SVMs sit among the other linear models.
- [[gradient-descent]] - the optimiser underneath, and why step-size schedules matter.
- [[feature-scaling-and-normalization]] - required before any margin-based model.

## References
- scikit-learn, Support Vector Machines: https://scikit-learn.org/stable/modules/svm.html
- scikit-learn SVC API: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
- Shalev-Shwartz et al., "Pegasos: Primal Estimated sub-GrAdient SOlver for SVM": https://home.ttic.edu/~nati/Publications/PegasosMPB.pdf
