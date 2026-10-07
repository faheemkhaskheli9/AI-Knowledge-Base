---
title: Decision trees and random forests
category: ml
tags: [decision-tree, random-forest, ensemble, bagging, extra-trees, feature-importance, tabular, scikit-learn]
use_cases:
  - "build a strong tabular model with almost no tuning or preprocessing"
  - "turn a model into readable if-then rules a business team can check"
  - "classify loan applications or customer churn with mixed numeric and categorical inputs"
  - "get a quick out-of-bag accuracy estimate without a separate validation set"
status: stable
last_verified: 2026-10-03
sources:
  - https://scikit-learn.org/stable/modules/tree.html
  - https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees
  - https://scikit-learn.org/stable/modules/permutation_importance.html
---

# Decision trees and random forests

## Summary
A decision tree splits the data with a sequence of yes/no questions on one feature at a time ("income < 40k?") until the leaves are pure enough to predict. A single tree is readable but overfits easily. A random forest averages hundreds of trees, each trained on a bootstrap sample with a random subset of features per split, which cancels most of that variance. Forests need no scaling, handle non-linearity and interactions natively, and are a strong default with almost no tuning; [[gradient-boosting-tabular]] usually edges them out after tuning.

## Key concepts
- **Split criterion.** Classification trees pick the split that most reduces impurity (Gini or entropy); regression trees reduce squared error.
- **Depth and leaves control capacity.** `max_depth`, `min_samples_leaf`, `max_leaf_nodes`; an unlimited tree memorises the training set.
- **Cost-complexity pruning** (`ccp_alpha`): grow a full tree, then prune branches that do not pay for their complexity.
- **Bagging.** Train each model on a bootstrap sample (rows drawn with replacement) and average; reduces variance, not bias.
- **Feature subsampling** (`max_features`): each split considers only a random subset of features, which decorrelates the trees. This is what makes a forest better than bagged trees.
- **Out-of-bag (OOB) score.** Each tree skips ~37% of rows; scoring each row with only the trees that did not see it gives a free validation estimate (`oob_score=True`).
- **Extra trees** (`ExtraTreesClassifier`): random split thresholds as well as random features; faster, sometimes as accurate.
- **Bagging vs boosting.** Forests build deep trees independently and average them (low bias, reduce variance). Boosting builds shallow trees sequentially, each fixing the previous errors (reduce bias); see [[gradient-boosting-tabular]].

## When to use / scenarios
- Tabular classification or regression where you want a good model in minutes: churn, credit default, lead scoring, equipment failure.
- When a stakeholder needs rules: a depth-3 tree printed with `export_text` is a policy people can audit.
- Mixed feature scales and skewed distributions: trees are invariant to monotone transforms, so no scaling.
- Robust baseline for comparing against boosting or deep learning.
- Not when: data is images/audio/text sequences (deep learning), you need smooth extrapolation beyond the training range (trees predict flat outside it, as in price trend forecasting), or the model must be tiny and very fast at inference (a forest of 500 deep trees is large).

## Setup & code
```bash
pip install scikit-learn
```
```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

data = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target,
                                          stratify=data.target, random_state=0)

# A shallow tree as readable rules.
tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X_tr, y_tr)
print(export_text(tree, feature_names=list(data.feature_names)))
print("tree test acc:", round(tree.score(X_te, y_te), 3))

# A forest: more accurate, with a free out-of-bag estimate.
forest = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                oob_score=True, n_jobs=-1, random_state=0)
forest.fit(X_tr, y_tr)
print("forest OOB acc:", round(forest.oob_score_, 3),
      "test acc:", round(forest.score(X_te, y_te), 3))

# Permutation importance on held-out data (more honest than feature_importances_).
imp = permutation_importance(forest, X_te, y_te, n_repeats=10, random_state=0)
for i in imp.importances_mean.argsort()[::-1][:3]:
    print(f"{data.feature_names[i]}: {imp.importances_mean[i]:.3f}")
```

## Choosing / trade-offs
- **Single tree vs forest.** Tree when the rules themselves are the deliverable; forest when accuracy matters more than a readable model.
- **Forest vs gradient boosting.** Forests are harder to get badly wrong (defaults work, little tuning, parallel training). Boosting is usually a few points better once tuned and handles larger data faster with histogram methods. Try both.
- **n_estimators.** More trees never overfit, they only cost time and memory; accuracy plateaus, often by 200-500.
- **max_features.** Lower = more diverse trees, less variance; scikit-learn defaults are `"sqrt"` for classification and all features (`1.0`) for regression.
- **min_samples_leaf.** Raising it to 2-10 smooths probabilities and shrinks the model with little accuracy loss.

## Gotchas
- Impurity-based `feature_importances_` is computed on training data and favours high-cardinality features (IDs, continuous noise). Use `permutation_importance` on a held-out set, see [[model-interpretability]].
- Forest probabilities are often poorly calibrated (pushed away from 0 and 1); calibrate before using them as real probabilities, see [[model-evaluation-and-metrics]].
- A fully grown forest on a large dataset can be gigabytes; limit depth or leaf size before deploying.
- Trees cannot extrapolate: a regression forest trained on prices up to 100 never predicts 120.
- scikit-learn trees need numeric input; encode categoricals first (ordinal encoding is fine for trees), see [[feature-engineering]].
- Decision boundaries are axis-aligned steps; a diagonal relationship needs many splits that a linear model gets in one weight.

## Related
- [[gradient-boosting-tabular]] - the boosted-tree alternative, usually the top tabular model.
- [[linear-models]] - the other default baseline, explainable through weights instead of rules.
- [[hyperparameter-tuning]] - tuning depth, leaf size and features per split.
- [[model-interpretability]] - permutation importance and partial dependence for forests.
- [[anomaly-detection]] - isolation forest reuses the random-tree idea for outliers.

## References
- scikit-learn decision trees: https://scikit-learn.org/stable/modules/tree.html
- scikit-learn forests of randomized trees: https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees
- Permutation importance: https://scikit-learn.org/stable/modules/permutation_importance.html
- Breiman, *Random Forests* (2001): https://doi.org/10.1023/A:1010933404324
