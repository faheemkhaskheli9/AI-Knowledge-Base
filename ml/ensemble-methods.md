---
title: Ensemble methods (voting, bagging, stacking)
category: ml
tags: [ensemble, voting, bagging, stacking, blending, model-averaging, scikit-learn]
use_cases:
  - "squeeze extra accuracy out of several models that each do reasonably well"
  - "combine a gradient-boosting model and a linear model for a more stable prediction"
  - "reduce the variance of an unstable model such as a deep decision tree"
  - "decide whether a stacked ensemble is worth its serving and maintenance cost"
status: stable
last_verified: 2026-10-03
sources:
  - https://scikit-learn.org/stable/modules/ensemble.html
  - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html
---

# Ensemble methods (voting, bagging, stacking)

## Summary
An ensemble combines several models so that their errors partly cancel out. **Bagging** trains copies of one model on bootstrap resamples and averages them (random forests are bagged trees). **Boosting** trains models in sequence, each fixing the last one's errors (see [[gradient-boosting-tabular]]). **Voting** averages different model types, and **stacking** trains a small "meta" model to learn how to weigh them. Ensembles usually add a few points of accuracy and stability, at the cost of slower prediction and more to maintain.

## Key concepts
- **Why it works.** Averaging models whose errors are not perfectly correlated reduces variance. Diversity matters more than the strength of any single member.
- **Bagging (bootstrap aggregating).** Each model sees a resample of rows (and optionally a subset of features); averaging smooths unstable, high-variance learners. `BaggingClassifier`, `RandomForestClassifier`, `ExtraTreesClassifier`.
- **Boosting.** Sequential, reduces bias; covered in [[gradient-boosting-tabular]].
- **Hard vs soft voting.** Hard voting takes the majority label; soft voting averages predicted probabilities and is usually better when members are calibrated.
- **Stacking.** Base models produce out-of-fold predictions (via internal cross-validation) that become features for a meta-model, usually a regularised logistic or linear regression.
- **Blending.** A simpler stacking variant that fits the meta-model on a single held-out set instead of out-of-fold predictions; easier, but wastes data.
- **Seed / snapshot averaging.** Averaging the same model trained with different random seeds (or checkpoints of one neural network) is the cheapest ensemble and often helps deep learning.

## When to use / scenarios
- Kaggle-style or high-stakes tabular problems (credit default, demand forecasting, churn) where the last 0.5-2% matters and serving several models is acceptable.
- Medical or risk models where stability across retrains is valued: averaged models flip fewer predictions between versions.
- Combining models that see different inputs (a text model and a tabular model) by stacking their scores.
- Not when: latency or memory budgets are tight (edge, real-time bidding); the gain on validation is within noise; one well-tuned boosting model is already close to the ceiling. Distil the ensemble into one model if you need its accuracy but not its cost.

## Setup & code
```bash
pip install scikit-learn
```
```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
members = [
    ("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
    ("svm", make_pipeline(StandardScaler(), SVC(probability=True, random_state=0))),
    ("rf", RandomForestClassifier(n_estimators=300, random_state=0)),
    ("hgb", HistGradientBoostingClassifier(random_state=0)),
]
models = dict(members)
models["soft vote"] = VotingClassifier(members, voting="soft")
models["stack"] = StackingClassifier(members, final_estimator=LogisticRegression(),
                                     cv=5)
for name, model in models.items():
    s = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print(f"{name:10s} ROC-AUC {s.mean():.4f} +/- {s.std():.4f}")
```
Here single models score 0.992-0.996 ROC-AUC, and the voting and stacking ensembles (about 0.996) slightly beat the best single model with a smaller spread across folds: a typical result when members are already strong. Larger gains appear when members are diverse and individually weaker.

## Choosing / trade-offs
- **Voting vs stacking.** Soft voting has nothing to overfit and is the safe default; stacking can learn that one model deserves more weight but needs enough data for its internal CV to be reliable.
- **Meta-model.** Keep it simple (logistic/linear regression, maybe with `passthrough=False`); a powerful meta-model overfits the base predictions.
- **Diversity over count.** Three different model families beat ten variants of the same boosting model. Check the correlation of their out-of-fold predictions.
- **Cost.** Prediction latency and memory grow with every member, and each must be retrained, monitored and versioned. Measure the gain against that.

## Gotchas
- Fitting a stacker's meta-model on in-sample base predictions leaks: base models look perfect on their own training data. Always use out-of-fold predictions (`StackingClassifier` does this via `cv`).
- Comparing an ensemble's CV score to single models tuned on the same folds overstates the gain; use a final hold-out or nested CV ([[hyperparameter-tuning]]).
- Soft voting with uncalibrated members (Naive Bayes, SVM scores) lets the most over-confident member dominate.
- Ensembles make [[model-interpretability]] harder; SHAP on a stack is slow and hard to explain to stakeholders.
- A gain inside the fold-to-fold standard deviation is noise, not an improvement.

## Related
- [[decision-trees-and-random-forests]] - bagging applied to trees.
- [[gradient-boosting-tabular]] - boosting, usually the strongest single tabular model.
- [[model-evaluation-and-metrics]] - calibration and cross-validation used to judge ensembles.
- [[hyperparameter-tuning]] - nested CV for an honest comparison.
- [[svm-knn-naive-bayes]] - diverse base models to combine.
- [[adaboost-from-scratch]] - AdaBoost built by hand: stumps, sample re-weighting, exponential loss.

## References
- scikit-learn ensemble guide: https://scikit-learn.org/stable/modules/ensemble.html
- StackingClassifier API: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html
