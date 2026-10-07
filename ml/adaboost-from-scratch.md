---
title: AdaBoost from scratch (decision stumps, sample re-weighting and exponential loss in NumPy)
category: ml
tags: [adaboost, boosting, decision-stump, weak-learner, sample-weights, exponential-loss, samme, ensemble, numpy, scikit-learn, from-scratch, ml-basics]
use_cases:
  - "implement AdaBoost with decision stumps from scratch"
  - "understand how boosting re-weights misclassified samples"
  - "compare a hand-written AdaBoost to scikit-learn's AdaBoostClassifier"
status: stable
last_verified: 2026-10-04
sources:
  - https://scikit-learn.org/stable/modules/ensemble.html#adaboost
  - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
  - https://hastie.su.domains/ElemStatLearn/
---

# AdaBoost from scratch (decision stumps, sample re-weighting and exponential loss in NumPy)

## Summary
AdaBoost builds a strong classifier from a sequence of weak ones, usually one-split decision trees ("stumps"). Each round fits a stump to **weighted** data, gives it a vote `α = ½ log((1 − err)/err)` based on its weighted error, then multiplies the weight of every misclassified sample by `e^{α}` so the next stump focuses on them. The final prediction is the sign of the weighted vote. It is the first boosting algorithm and equals forward stagewise fitting of the exponential loss. About 50 lines of NumPy reach 0.971 test accuracy on the breast-cancer dataset and pick exactly the same first stumps, thresholds and votes as scikit-learn.

## Key concepts
- **Weak learner.** Anything slightly better than chance on weighted data. A stump picks one feature, one threshold and one polarity.
- **Weighted error.** `err = Σ wᵢ · [h(xᵢ) ≠ yᵢ]` with weights summing to 1. A stump fitted to weights, not to counts, is the whole trick.
- **Vote.** `α = ½ log((1 − err)/err)`: large when `err` is small, zero at `err = 0.5`, negative below chance (so stop there).
- **Re-weighting.** `wᵢ ← wᵢ · exp(−α yᵢ h(xᵢ))`, then normalise. With labels in `{−1, +1}`, correct samples shrink and mistakes grow by the same factor.
- **Exponential loss.** AdaBoost greedily minimises `Σ exp(−yᵢ F(xᵢ))` where `F = Σ αₘ hₘ`. Gradient boosting generalises the idea to any differentiable loss ([[gradient-boosting-from-scratch]]).
- **Margin.** `y F(x)` keeps growing after training error hits 0, which is why test accuracy can still improve with more rounds.
- **SAMME.** The multiclass version used by scikit-learn: `α = log((1 − err)/err) + log(K − 1)`. For two classes it is exactly twice the classic `α`, which does not change the sign of the vote.

## When to use / scenarios
- Learning: the cleanest example of boosting and of training a model on sample weights.
- Interviews: "how does AdaBoost pick weights", "bagging vs boosting", "why is AdaBoost sensitive to label noise".
- Small, clean tabular problems where a quick, interpretable ensemble of stumps is enough; also the Viola-Jones face detector.
- Production: prefer gradient-boosted trees (XGBoost, LightGBM, CatBoost, scikit-learn `HistGradientBoostingClassifier`) for accuracy and speed ([[gradient-boosting-tabular]]).
- Not for: noisy labels or many outliers. The exponential loss keeps up-weighting points it cannot fit.

## Setup & code
`pip install numpy scikit-learn`. Runs in about 3 seconds.

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def fit_stump(X, y, w):
    """Best weighted-error threshold split; y in {-1, +1}. Predicts s if x > t else -s."""
    best = (np.inf, 0, 0.0, 1)
    for j in range(X.shape[1]):
        order = np.argsort(X[:, j])
        xs, ys, ws = X[order, j], y[order], w[order]
        # error of "predict +1 right of the cut": positives left + negatives right
        pos_left = np.concatenate([[0], np.cumsum(ws * (ys == 1))])
        neg_left = np.concatenate([[0], np.cumsum(ws * (ys == -1))])
        err = pos_left + (neg_left[-1] - neg_left)          # cut after i items, i = 0..n
        valid = np.concatenate([[True], xs[1:] != xs[:-1], [True]])   # never split ties
        for s, e in ((1, err), (-1, 1 - err)):              # flipped polarity
            e = np.where(valid, e, np.inf)                   # mask after flipping, not before
            i = int(np.argmin(e))
            if e[i] < best[0]:
                t = xs[0] - 1 if i == 0 else xs[-1] if i == len(xs) else (xs[i - 1] + xs[i]) / 2
                best = (e[i], j, t, s)
    return best


def stump_predict(X, j, t, s):
    return np.where(X[:, j] > t, s, -s)


def fit_adaboost(X, y, rounds=200):
    w = np.full(len(y), 1 / len(y))
    model = []
    for _ in range(rounds):
        err, j, t, s = fit_stump(X, y, w)
        err = max(err, 1e-10)
        if err >= 0.5:                                       # no better than chance: stop
            break
        alpha = 0.5 * np.log((1 - err) / err)
        pred = stump_predict(X, j, t, s)
        w *= np.exp(-alpha * y * pred)                       # up-weight mistakes
        w /= w.sum()
        model.append((alpha, j, t, s))
    return model


def decision(model, X):
    return sum(a * stump_predict(X, j, t, s) for a, j, t, s in model)


X, y01 = load_breast_cancer(return_X_y=True)
y = 2 * y01 - 1
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

model = fit_adaboost(Xtr, ytr, rounds=200)
for m in (1, 10, 50, 200):
    tr = np.mean(np.sign(decision(model[:m], Xtr)) == ytr)
    te = np.mean(np.sign(decision(model[:m], Xte)) == yte)
    print(f"rounds={m:3d} train acc={tr:.3f} test acc={te:.3f}")

sk = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200,
                        random_state=0).fit(Xtr, ytr)
ours = np.sign(decision(model, Xte))
print(f"sklearn test acc={sk.score(Xte, yte):.3f} agree={np.mean(ours == sk.predict(Xte)):.3f}")
print("ours    (alpha, feature, threshold):",
      [(round(float(a), 3), j, round(float(t), 3)) for a, j, t, s in model[:3]])
print("sklearn (alpha/2, feature, threshold):",   # SAMME's alpha has no 1/2 factor
      [(round(float(a) / 2, 3), int(e.tree_.feature[0]), round(float(e.tree_.threshold[0]), 3))
       for a, e in zip(sk.estimator_weights_[:3], sk.estimators_[:3])])
```

Output (numpy 2.5, scikit-learn 1.9):
```
rounds=  1 train acc=0.932 test acc=0.889
rounds= 10 train acc=0.982 test acc=0.930
rounds= 50 train acc=1.000 test acc=0.959
rounds=200 train acc=1.000 test acc=0.971
sklearn test acc=0.959 agree=0.977
ours    (alpha, feature, threshold): [(1.31, 22, 106.1), (1.057, 27, 0.142), (0.794, 21, 23.35)]
sklearn (alpha/2, feature, threshold): [(1.31, 22, 106.1), (1.057, 27, 0.142), (0.794, 21, 23.35)]
```

One stump already gets 0.889 test accuracy; 200 of them get 0.971. Training accuracy reaches 1.000 by round 50, yet test accuracy keeps rising afterwards: the margins keep growing even when no training point is misclassified. The first three stumps match scikit-learn's exactly. Later rounds drift apart because near-equal splits break ties differently, so the two models agree on 97.7% of test points rather than all of them. A first version of `fit_stump` set invalid (tied) cuts to `inf` **before** computing the flipped error `1 − err`, which turned them into `−inf` and made every round pick the same useless split, with 0.456 accuracy. Masking after the flip fixed it.

## Choosing / trade-offs
- **AdaBoost vs gradient boosting.** AdaBoost is gradient boosting with exponential loss and a line search. Log-loss gradient boosting is more robust to noise and gives better probabilities; histogram GBDTs are much faster ([[ensemble-methods]], [[gradient-boosting-tabular]]).
- **Boosting vs bagging.** Boosting fits learners in sequence to the previous mistakes and mainly reduces bias. Bagging and random forests fit them in parallel on bootstrap samples and mainly reduce variance ([[random-forest-from-scratch]], [[bias-variance-and-learning-curves]]).
- **Weak learner depth.** Stumps give an additive model with no feature interactions. Depth-2 or 3 trees capture interactions but overfit sooner.
- **Rounds and learning rate.** Shrinking each vote (`learning_rate < 1`) needs more rounds but usually generalises better. Pick the number of rounds on validation data.

## Gotchas
- Labels must be `−1/+1` for the update `exp(−α y h)` to work. With `0/1` labels the weights of correct and wrong samples move the wrong way.
- Stop when `err ≥ 0.5`, and clamp `err` away from 0: a perfect stump gives `α = ∞`.
- Never place a threshold between two equal feature values. The split cannot actually separate them and the computed error is wrong.
- Mislabelled points get exponentially large weights and drag later stumps towards them. Inspect the largest final weights; they are often label errors ([[label-noise-and-data-cleaning]]).
- `sign(F)` is a class, not a probability. Use `predict_proba` with care, or calibrate ([[probability-calibration]]).
- scikit-learn removed the `algorithm="SAMME.R"` option; current versions use SAMME only. Code that passes it fails on recent releases.
- Stumps do not need feature scaling, but one-hot encode or ordinal-encode categoricals first ([[categorical-encoding]]).

## Related
- [[gradient-boosting-from-scratch]] - boosting with any differentiable loss by fitting residuals.
- [[decision-tree-from-scratch]] - the tree learner a stump is the depth-1 case of.
- [[random-forest-from-scratch]] - the bagging counterpart.
- [[ensemble-methods]] - bagging, boosting and stacking compared.
- [[gradient-boosting-tabular]] - XGBoost, LightGBM and CatBoost for production.
- [[loss-functions]] - exponential loss next to log-loss and hinge loss.

## References
- scikit-learn User Guide, AdaBoost: https://scikit-learn.org/stable/modules/ensemble.html#adaboost
- scikit-learn AdaBoostClassifier API: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
- Hastie, Tibshirani and Friedman, The Elements of Statistical Learning, ch. 10 (Boosting and Additive Trees): https://hastie.su.domains/ElemStatLearn/
