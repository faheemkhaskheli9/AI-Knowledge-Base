---
title: Machine learning fundamentals
category: concepts
tags: [machine-learning, supervised, unsupervised, overfitting, bias-variance, generalization, train-test-split, baseline]
use_cases:
  - "decide whether a business problem is a classification, regression, clustering or ranking task"
  - "explain to stakeholders why a model scores 99% in training but fails in production"
  - "set up a correct train/validation/test split before building any model"
  - "decide whether a problem needs ML at all or a rule/SQL query is enough"
status: stable
last_verified: 2026-10-03
sources:
  - https://developers.google.com/machine-learning/crash-course
  - https://scikit-learn.org/stable/tutorial/basic/tutorial.html
  - https://scikit-learn.org/stable/modules/learning_curve.html
  - https://www.statlearning.com/
---

# Machine learning fundamentals

## Summary
Machine learning fits a function from examples instead of hand-writing rules: given inputs X and (usually) known outputs y, an algorithm picks parameters that minimise a loss on training data, and the real test is how well it does on data it has never seen (generalisation). This file covers the vocabulary and the few principles every ML project depends on: problem framing, data splits, overfitting vs underfitting, the bias-variance trade-off and baselines. Read it before [[classic-ml-scikit-learn]] or [[neural-network-fundamentals]].

## Key concepts
- **Learning types.** Supervised (labelled y: classification for categories, regression for numbers), unsupervised (no y: clustering, dimensionality reduction, anomaly detection), self-supervised (labels made from the data itself, e.g. next-token prediction, see [[pretraining-and-scaling-laws]]), and reinforcement learning (reward from an environment).
- **Features, labels, samples.** A row is a sample, columns are features, the target is the label. Most ML effort goes into the data, not the algorithm (see [[feature-engineering]]).
- **Model, loss, optimiser.** The model is a parameterised function; the loss measures error (MSE for regression, cross-entropy for classification); the optimiser (closed form, gradient descent, tree splitting) changes parameters to reduce it.
- **Generalisation.** Training error always looks good. Only error on held-out data estimates real performance.
- **Train / validation / test.** Fit on train, choose models and hyperparameters on validation (or cross-validation), report once on test. Touching the test set while tuning makes it a second validation set.
- **Overfitting vs underfitting.** Overfit: low train error, high validation error (memorised noise). Underfit: both errors high (model too simple or features too weak).
- **Bias-variance trade-off.** Error = bias² (wrong assumptions) + variance (sensitivity to the particular training sample) + irreducible noise. More capacity lowers bias and raises variance; more data and regularisation lower variance.
- **Regularisation.** Anything that limits effective capacity: L1/L2 penalties, tree depth limits, early stopping, dropout (see [[deep-learning-training]]).
- **Hyperparameters vs parameters.** Parameters are learned (weights); hyperparameters are set before training (learning rate, tree depth, k in k-NN) and tuned on validation data.
- **No free lunch.** No algorithm wins everywhere; rough defaults: tabular → gradient boosting, images/audio/text → deep learning, small/explainable → linear models.

## When to use / scenarios
- Signals ML fits: a decision is repeated many times, examples with known outcomes exist (or can be labelled), the rules are too many or fuzzy to write by hand, and some error rate is acceptable.
- Typical framings: churn (binary classification), demand (regression or [[time-series-forecasting]]), ticket routing (multi-class, see [[nlp-classic-tasks]]), customer segments (clustering), fraud with few labels ([[anomaly-detection]]), "what to show next" ([[recommender-systems]]).
- NOT ML: the rule is known and stable (tax brackets → code), there is no historical data and no way to label it, or every error is unacceptable and must be explained deterministically. For language tasks with no training data, calling an LLM ([[prompt-engineering]]) is often faster than training anything.

## Setup & code
```bash
pip install scikit-learn
```
Diagnose over/underfitting by comparing train and validation scores as capacity grows:
```python
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.tree import DecisionTreeClassifier

X, y = make_classification(n_samples=2000, n_features=20, n_informative=5,
                           flip_y=0.1, random_state=0)
# Hold out a test set first; never look at it while tuning.
X_trval, X_test, y_trval, y_test = train_test_split(X, y, test_size=0.2,
                                                    stratify=y, random_state=0)

print("baseline:", DummyClassifier().fit(X_trval, y_trval).score(X_test, y_test))

depths = [1, 2, 4, 8, 16, None]
train_sc, val_sc = validation_curve(
    DecisionTreeClassifier(random_state=0), X_trval, y_trval,
    param_name="max_depth", param_range=depths, cv=5)
for d, tr, va in zip(depths, train_sc.mean(1), val_sc.mean(1)):
    print(f"depth={d}: train={tr:.2f} val={va:.2f}")   # gap grows = overfitting

best = depths[val_sc.mean(1).argmax()]
model = DecisionTreeClassifier(max_depth=best, random_state=0).fit(X_trval, y_trval)
print("test:", model.score(X_test, y_test))           # report once
```

## Choosing / trade-offs
- **Simple vs flexible model.** Start simple (linear/logistic, shallow tree). Move to flexible models only when validation error, not training error, says the simple one underfits.
- **More data vs better model.** A learning curve (`sklearn.model_selection.learning_curve`) that still rises with more data says collect data; a flat one with a train/val gap says regularise; both low says better features or a stronger model.
- **Split strategy.** Random split for i.i.d. rows; time-ordered split when predicting the future; group split (`GroupKFold`) when rows share an entity such as a customer or patient.
- **Interpretability vs accuracy.** Regulated domains (credit, healthcare) may prefer a slightly weaker explainable model.

## Gotchas
- **Data leakage** is the most common reason a model looks great offline and fails live: preprocessing fit on all data, features that are only known after the outcome, or duplicate entities across splits.
- Plain accuracy on imbalanced data is misleading (99% "not fraud" is useless); see [[model-evaluation-and-metrics]].
- Train and production distributions drift apart over time; a model is not finished at deploy (see [[mlops-lifecycle]]).
- Correlation is not causation: a churn model tells you who will leave, not what will make them stay.
- Set random seeds and record splits, or results cannot be reproduced (see [[experiment-tracking]]).
- Always compare against a dumb baseline (majority class, last value, mean); many "models" do not beat it.

## Related
- [[classic-ml-scikit-learn]] - the practical toolkit for these ideas on tabular data.
- [[model-evaluation-and-metrics]] - how to measure generalisation correctly.
- [[feature-engineering]] - turning raw data into useful inputs.
- [[neural-network-fundamentals]] - the deep-learning branch of the same ideas.
- [[gradient-boosting-tabular]] - the usual strongest tabular model.
- [[exploratory-data-analysis]] - the EDA pass to run before any modelling.

## References
- Google Machine Learning Crash Course: https://developers.google.com/machine-learning/crash-course
- scikit-learn tutorial: https://scikit-learn.org/stable/tutorial/basic/tutorial.html
- Validation and learning curves: https://scikit-learn.org/stable/modules/learning_curve.html
- James, Witten, Hastie, Tibshirani, *An Introduction to Statistical Learning* (free PDF): https://www.statlearning.com/
