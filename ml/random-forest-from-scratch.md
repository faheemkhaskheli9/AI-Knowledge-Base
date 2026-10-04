---
title: Random forest from scratch (bagging, feature subsampling and out-of-bag score)
category: ml
tags: [random-forest, bagging, bootstrap, ensemble, out-of-bag, feature-subsampling, variance-reduction, numpy, scikit-learn, from-scratch, ml-basics]
use_cases:
  - "implement a random forest from scratch for learning or an interview"
  - "understand why averaging many decision trees beats one tree"
  - "compute an out-of-bag score instead of holding out a validation set"
status: draft
last_verified: 2026-10-04
sources:
  - https://scikit-learn.org/stable/modules/ensemble.html#random-forests
  - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
  - https://link.springer.com/article/10.1023/A:1010933404324
---

# Random forest from scratch (bagging, feature subsampling and out-of-bag score)

## Summary
A random forest trains many deep decision trees, each on a bootstrap sample of the rows and with only a random subset of features considered at every split, then averages their predicted probabilities. Each tree overfits in its own way, so averaging cancels much of the variance while keeping the low bias of a deep tree. The forest is about 30 lines on top of a CART tree; on the breast-cancer dataset it lifts test accuracy from 0.906 (one tree) to 0.942, and the out-of-bag rows give a free validation score.

## Key concepts
- **Bootstrap (bagging).** Each tree sees `n` rows drawn with replacement. About `1 − 1/e ≈ 63.2%` of distinct rows appear in a given sample; the other ~36.8% are that tree's **out-of-bag (OOB)** rows.
- **Feature subsampling.** At every split a tree considers only `max_features` random features (`√d` is the classic default for classification). This is what separates a random forest from plain bagging: it stops every tree from splitting on the same strong feature first, which decorrelates the trees.
- **Why averaging helps.** For `B` trees, each with variance `σ²` and pairwise correlation `ρ`, the variance of the average is `ρσ² + (1 − ρ)σ²/B`. More trees shrink the second term; only lower correlation shrinks the first. Feature subsampling attacks `ρ`.
- **Deep, unpruned trees.** Each tree should have low bias and high variance; the ensemble removes the variance. Pruning individual trees mostly throws away the bias advantage ([[bias-variance-and-learning-curves]]).
- **OOB score.** Predict every row using only the trees whose bootstrap did not contain it, then score those predictions. It is close to a cross-validation estimate at no extra training cost.
- **Soft voting.** Average `predict_proba` across trees, then take the argmax. scikit-learn does this; majority voting of hard labels throws away confidence.

## When to use / scenarios
- Learning: the cleanest example of variance reduction by ensembling ([[ensemble-methods]]).
- Interviews: "how is a random forest different from bagging", "what is OOB error", "why don't more trees overfit".
- A strong tabular baseline with almost no tuning, robust to unscaled features and outliers.
- Not for: squeezing out the last points of accuracy on tabular data (gradient boosting usually wins, see [[gradient-boosting-from-scratch]] and [[gradient-boosting-tabular]]), extrapolating beyond the training range in regression, or very high-dimensional sparse text.

## Setup & code
`pip install numpy scikit-learn`. The tree itself comes from scikit-learn so the forest logic stays visible; swap in the CART from [[decision-tree-from-scratch]] (adding a random feature subset inside its split search) to make it fully from scratch. Runs in a few seconds.

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class RandomForest:
    """Bagging + random feature subsets per split, on top of a CART tree."""

    def __init__(self, n_trees=100, max_features="sqrt", seed=0):
        self.n_trees, self.max_features = n_trees, max_features
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y):
        n = len(X)
        self.classes_ = np.unique(y)
        self.trees, self.oob_masks = [], []
        for _ in range(self.n_trees):
            idx = self.rng.integers(0, n, n)                 # bootstrap: n draws with replacement
            oob = np.ones(n, bool)
            oob[idx] = False                                 # ~36.8% of rows never drawn
            tree = DecisionTreeClassifier(                   # swap in your own CART here
                max_features=self.max_features,              # feature subsampling at every split
                random_state=int(self.rng.integers(1 << 31)))
            self.trees.append(tree.fit(X[idx], y[idx]))
            self.oob_masks.append(oob)
        # out-of-bag score: each row is predicted only by trees that never saw it
        votes = np.zeros((n, len(self.classes_)))
        for tree, oob in zip(self.trees, self.oob_masks):
            votes[oob] += tree.predict_proba(X[oob])
        seen = votes.sum(1) > 0
        self.oob_score_ = np.mean(self.classes_[votes[seen].argmax(1)] == y[seen])
        return self

    def predict_proba(self, X):
        return np.mean([t.predict_proba(X) for t in self.trees], axis=0)   # soft voting

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

tree = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr)
print(f"single tree        test acc={tree.score(Xte, yte):.3f}")
for mf in (None, "sqrt"):
    rf = RandomForest(n_trees=200, max_features=mf).fit(Xtr, ytr)
    print(f"ours max_features={str(mf):<5} test acc={np.mean(rf.predict(Xte) == yte):.3f} oob={rf.oob_score_:.3f}")
sk = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=0).fit(Xtr, ytr)
print(f"sklearn            test acc={sk.score(Xte, yte):.3f} oob={sk.oob_score_:.3f}")

# variance of the ensemble vs number of trees
rf = RandomForest(n_trees=200).fit(Xtr, ytr)
for k in (1, 5, 25, 200):
    p = np.mean([t.predict_proba(Xte) for t in rf.trees[:k]], axis=0)
    print(f"trees={k:<4} test acc={np.mean(rf.classes_[p.argmax(1)] == yte):.3f}")
```

Output (numpy 2.5, scikit-learn 1.9):
```
single tree        test acc=0.906
ours max_features=None  test acc=0.924 oob=0.962
ours max_features=sqrt  test acc=0.942 oob=0.962
sklearn            test acc=0.953 oob=0.970
trees=1    test acc=0.918
trees=5    test acc=0.936
trees=25   test acc=0.947
trees=200  test acc=0.942
```

Bagging alone (`max_features=None`) improves on one tree (0.906 to 0.924), and adding per-split feature subsampling improves it again (0.942). scikit-learn's forest uses the same algorithm with different random draws and lands at 0.953; on 171 test rows one row is 0.006, so the gap between the two forests is two rows and not a real difference. Accuracy climbs fast for the first few trees and then flattens; the dip from 25 to 200 trees is one test row and is noise, not overfitting. The OOB scores sit a little above the test scores here, which is within the noise of a 171-row test set.

## Choosing / trade-offs
- **`max_features`.** Lower values decorrelate trees more but make each tree weaker. `√d` for classification and `d/3` or `1.0` for regression are good starting points; tune it before anything else.
- **Number of trees.** More trees never increase variance, they only cost time and memory. Stop when the OOB score stops moving (often 200 to 500).
- **Tree depth / `min_samples_leaf`.** Leave trees deep by default. Raise `min_samples_leaf` for noisy targets or to shrink the model on disk.
- **Random forest vs gradient boosting.** Forests reduce variance of low-bias trees and are hard to misconfigure. Boosting reduces bias of shallow trees, usually scores higher, but needs a learning rate, early stopping and more tuning ([[ensemble-methods]]).
- **Random forest vs extra-trees.** `ExtraTreesClassifier` also randomises the split threshold, which decorrelates trees further and trains faster, often at similar accuracy.

## Gotchas
- Re-using one random state for every tree makes all trees identical apart from the bootstrap. Give each tree its own seed, as above.
- The OOB score is only defined for rows that were out-of-bag for at least one tree. With very few trees some rows have no OOB prediction; skip them (the `seen` mask) rather than counting them as wrong.
- Impurity-based feature importances (`feature_importances_`) favour high-cardinality and continuous features. Use permutation importance on held-out data for decisions ([[model-interpretability]]).
- Forests cannot predict outside the range of training targets in regression; a leaf average is always between observed values. Use a linear model or boosting with a linear base for trends ([[time-series-forecasting]]).
- Class imbalance: the bootstrap preserves the imbalance. Use `class_weight="balanced_subsample"` or balanced bootstraps ([[imbalanced-data]]).
- Big forests are big files. 500 deep trees on a million rows can be gigabytes; cap depth or leaf size before shipping.

## Related
- [[decision-tree-from-scratch]] - the CART base learner each tree uses.
- [[decision-trees-and-random-forests]] - the scikit-learn API, tuning and feature importance.
- [[ensemble-methods]] - bagging, boosting and stacking side by side.
- [[gradient-boosting-from-scratch]] - the other way to combine trees, by fitting residuals.
- [[bias-variance-and-learning-curves]] - why averaging deep trees works.

## References
- scikit-learn, Forests of randomized trees: https://scikit-learn.org/stable/modules/ensemble.html#random-forests
- scikit-learn RandomForestClassifier API: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- Breiman, "Random Forests", Machine Learning 45, 2001: https://link.springer.com/article/10.1023/A:1010933404324
