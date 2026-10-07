---
title: Decision tree from scratch (Gini impurity, best-split search, recursive CART in NumPy)
category: ml
tags: [decision-tree, cart, gini-impurity, entropy, information-gain, recursive-partitioning, numpy, scikit-learn, from-scratch, ml-basics]
use_cases:
  - "implement a decision tree classifier from scratch for learning or an interview"
  - "understand how a tree chooses a feature and threshold with Gini impurity"
  - "see why max_depth and min_samples control overfitting in trees"
  - "match a from-scratch tree's splits to scikit-learn's DecisionTreeClassifier"
status: stable
last_verified: 2026-10-04
sources:
  - https://scikit-learn.org/stable/modules/tree.html
  - https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
  - https://scikit-learn.org/stable/modules/tree.html#mathematical-formulation
---

# Decision tree from scratch (Gini impurity, best-split search, recursive CART in NumPy)

## Summary
A decision tree classifier asks a sequence of yes/no questions of the form `x[f] <= t` and predicts the majority class of the leaf a sample lands in. CART builds it greedily: at each node, try every feature and every threshold, keep the split that lowers impurity the most, and recurse on both halves until a stopping rule fires. About 45 lines of NumPy pick the same root split as scikit-learn and reach the same accuracy. Writing it once explains what every tree ensemble (random forest, gradient boosting) is built from.

## Key concepts
- **Impurity.** Gini `G = 1 − Σ p_c²` is the chance two random samples from the node have different classes: 0 for a pure node, 0.5 for a 50/50 binary node. Entropy `H = −Σ p_c log₂ p_c` behaves almost identically ([[information-theory-for-ml]]).
- **Split gain.** `gain = G(parent) − (n_L·G(L) + n_R·G(R)) / n`. The children are weighted by their size so that splitting off one sample cannot look great.
- **Threshold search.** Sort the node's values of one feature. Only midpoints between consecutive *distinct* values can change the partition, so there are at most `n − 1` candidates per feature. Trying all features gives `O(d · n log n)` per node.
- **Greedy recursion.** The best split now is chosen without look-ahead. A globally optimal tree is NP-hard to find, so every practical library is greedy.
- **Stopping and pruning.** Stop at `max_depth`, at fewer than `min_samples_split` samples, at a pure node, or when no split improves impurity. Without these limits a tree grows until every leaf is pure and memorises the training set. Cost-complexity pruning (`ccp_alpha`) cuts back a grown tree instead.
- **Leaves.** A classification leaf stores class counts: predict the majority, or the class fractions as probabilities. A regression tree uses variance (MSE) as the impurity and predicts the leaf mean.
- **Axis-aligned.** Every split is perpendicular to one feature axis, so trees need no feature scaling, but a diagonal boundary takes a staircase of many splits.

## When to use / scenarios
- Learning: the building block of [[decision-trees-and-random-forests]] and [[gradient-boosting-tabular]]. Reading this code makes `max_depth`, `min_samples_leaf` and feature importance concrete.
- Interviews: computing Gini by hand, writing the split search, explaining why deep trees overfit ([[bias-variance-and-learning-curves]]).
- Rule extraction: a shallow tree (depth 2-4) is a readable set of if/else rules for a business or compliance audience ([[model-interpretability]]).
- Production: use `sklearn.tree.DecisionTreeClassifier`, or better a random forest or gradient boosting for accuracy. The from-scratch version is quadratic in node size and pure Python.

## Setup & code
`pip install numpy scikit-learn`. Runs on CPU in a few seconds (the pure-Python split loop is the slow part).

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def gini(y):
    p = np.bincount(y) / len(y)
    return 1 - (p ** 2).sum()


def best_split(X, y):
    """Scan every feature and every midpoint; return the largest impurity drop."""
    n, best = len(y), (0.0, None, None)
    parent = gini(y)
    for f in range(X.shape[1]):
        order = np.argsort(X[:, f])
        xs, ys = X[order, f], y[order]
        for i in range(1, n):
            if xs[i] == xs[i - 1]:
                continue                       # can't split between equal values
            left, right = ys[:i], ys[i:]
            child = (i * gini(left) + (n - i) * gini(right)) / n
            if parent - child > best[0]:
                best = (parent - child, f, (xs[i] + xs[i - 1]) / 2)
    return best


def build(X, y, depth=0, max_depth=4, min_samples=2):
    leaf = {"value": np.bincount(y, minlength=2).argmax()}
    if depth == max_depth or len(y) < min_samples or gini(y) == 0:
        return leaf
    gain, f, t = best_split(X, y)
    if f is None:
        return leaf
    m = X[:, f] <= t
    return {"f": f, "t": t,
            "l": build(X[m], y[m], depth + 1, max_depth, min_samples),
            "r": build(X[~m], y[~m], depth + 1, max_depth, min_samples)}


def predict_one(node, x):
    while "f" in node:
        node = node["l"] if x[node["f"]] <= node["t"] else node["r"]
    return node["value"]


X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state=0)
tree = build(X_tr, y_tr, max_depth=4)
ours = np.array([predict_one(tree, x) for x in X_te])
sk = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X_tr, y_tr)
print("root split ours:   ", tree["f"], round(tree["t"], 4))
print("root split sklearn:", sk.tree_.feature[0], round(sk.tree_.threshold[0], 4))
print("test acc ours:", round((ours == y_te).mean(), 3),
      " sklearn:", round(sk.score(X_te, y_te), 3))
print("prediction agreement:", round((ours == sk.predict(X_te)).mean(), 3))
```

Output (numpy 2.5, scikit-learn 1.9):
```
root split ours:    22 106.1
root split sklearn: 22 106.1
test acc ours: 0.916  sklearn: 0.909
prediction agreement: 0.979
```

The root split is identical (feature 22, "worst perimeter", at 106.1). Deeper nodes occasionally differ: when two features tie on gain, scikit-learn breaks the tie through a random feature permutation (`random_state`), while this code keeps the first one found. That is why 2% of test predictions differ and the accuracies differ by one sample. Neither number is "better"; both are the same greedy algorithm.

## Choosing / trade-offs
- **Gini vs entropy.** They pick the same split in the vast majority of cases. Gini avoids a log and is the default. Use `log_loss`/entropy only if you have measured a difference.
- **Pre-pruning vs post-pruning.** `max_depth` / `min_samples_leaf` are cheap and easy to tune. Cost-complexity pruning (`ccp_alpha`, chosen by cross-validation) can keep a deep branch where it helps and cut it elsewhere.
- **One tree vs an ensemble.** A single tree is readable but high variance: a small change in data can change the root split. Random forests average many decorrelated trees, boosting adds shallow trees sequentially. Both trade readability for accuracy ([[ensemble-methods]]).
- **Exact vs histogram splits.** Exact search sorts every feature at every node. Histogram-based learners (LightGBM, `HistGradientBoosting`) bin features into ~256 buckets and scan the bins: much faster on large data, nearly the same splits.

## Gotchas
- Calling `gini` on each candidate's slices, as above, makes the scan `O(n²)` per feature. Real implementations keep running class counts and update them as the threshold moves, which makes it `O(n)` after sorting.
- An unbounded tree hits 100% training accuracy. That number says nothing; always check held-out data.
- Impurity-based feature importance is biased toward continuous and high-cardinality features, which offer more thresholds. Prefer permutation importance ([[model-interpretability]]).
- `<=` vs `<` and midpoint vs left-value thresholds change which side boundary samples go to. Match the library's convention when comparing results.
- scikit-learn trees need numeric input. Label-encoding an unordered category imposes a fake order; one-hot or target encoding is usually safer ([[categorical-encoding]]).
- Class imbalance makes the majority class win most leaves. Use `class_weight="balanced"` or adjust the threshold on predicted probabilities ([[imbalanced-data]]).

## Related
- [[decision-trees-and-random-forests]] - production trees and forests in scikit-learn.
- [[gradient-boosting-tabular]] - shallow trees added sequentially, the strongest tabular baseline.
- [[ensemble-methods]] - bagging and boosting built on top of this tree.
- [[information-theory-for-ml]] - entropy and information gain behind the split criterion.
- [[bias-variance-and-learning-curves]] - why depth controls over- and underfitting.
- [[k-means-from-scratch]] - the other greedy, locally optimal from-scratch algorithm.

## References
- scikit-learn, Decision Trees (CART, criteria, pruning): https://scikit-learn.org/stable/modules/tree.html
- scikit-learn DecisionTreeClassifier API: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
- scikit-learn, Decision trees mathematical formulation (impurity and split quality): https://scikit-learn.org/stable/modules/tree.html#mathematical-formulation
