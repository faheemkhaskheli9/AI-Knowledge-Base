---
title: Multiclass classification strategies (softmax, one-vs-rest, one-vs-one)
category: ml
tags: [multiclass, softmax, one-vs-rest, one-vs-one, logistic-regression, svm, top-k-accuracy, macro-f1, scikit-learn]
use_cases:
  - "classify into more than two categories (product type, intent, document class)"
  - "turn a binary-only model such as an SVM into a multiclass one"
  - "choose metrics for a 10- or 1000-class problem"
  - "handle hundreds of classes or a label set that keeps growing"
status: draft
last_verified: 2026-10-04
sources:
  - https://scikit-learn.org/stable/modules/multiclass.html
  - https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  - https://scikit-learn.org/stable/modules/model_evaluation.html#top-k-accuracy-score
---

# Multiclass classification strategies (softmax, one-vs-rest, one-vs-one)

## Summary
Multiclass classification assigns each example exactly one of K > 2 classes. A model either handles K classes natively (softmax/multinomial logistic regression, trees, gradient boosting, neural networks), or it is built from binary classifiers: one-vs-rest (K models) or one-vs-one (K(K-1)/2 models). The choice affects calibration, training cost and how the model scales to many classes. The metric (macro vs micro averaging, top-k) affects what "good" means.

## Key concepts
- **Multiclass vs multi-label.** Multiclass: exactly one label per row, probabilities sum to 1. Multi-label: any number of labels per row, one independent sigmoid each ([[multi-label-and-multi-task-learning]]). Mixing them up is the most common modeling bug.
- **Softmax (multinomial).** One model outputs K scores, and softmax turns them into probabilities that sum to 1. Trained with cross-entropy. This is what scikit-learn's `LogisticRegression` fits for multiclass `y` with its default `lbfgs` solver, and what every neural classifier uses.
- **One-vs-rest (OvR / OvA).** K binary models, "class k vs everything else". Predict the class with the highest score. Each model sees an imbalanced problem (1 vs K-1), and the K scores are not jointly calibrated.
- **One-vs-one (OvO).** One model per pair of classes, prediction by voting. Each model trains on only two classes' rows, so each fit is small. This is why `SVC` uses it internally. The number of models grows quadratically: 45 for 10 classes, ~500k for 1000.
- **Native multiclass.** Decision trees, random forests, k-NN and naive Bayes handle K classes directly. Gradient boosting builds K trees per round (one per class) with a softmax loss.
- **Averaging.** Per-class precision/recall/F1 are combined by **macro** (mean over classes, every class counts equally), **weighted** (by support), or **micro** (pooled counts; equals accuracy for single-label multiclass).
- **Top-k accuracy.** Correct if the true class is among the k highest-scoring. Use it when the output feeds a human picker or a reranker.

## When to use / scenarios
- Support-ticket routing, intent detection, product categorization, document type, species, fault type.
- Handwritten digits and image classification ([[image-classification]]).
- Hundreds or thousands of classes (product taxonomy, ICD codes): softmax with a neural net, or embeddings + nearest neighbor ([[metric-learning-and-few-shot]]). OvO is out of the question. A hierarchical classifier (department -> category -> item) helps when the taxonomy is a tree.
- Classes added often: embeddings plus k-NN or retrieval avoids retraining a softmax head per new class.
- Few labeled examples per class with a text input: try zero-/few-shot LLM classification before training ([[structured-output]]).

## Setup & code
```bash
pip install "scikit-learn>=1.5"
```
Softmax vs OvR vs OvO on the 10-class digits dataset:
```python
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state=0)

models = {
    "softmax (multinomial LR)": LogisticRegression(max_iter=2000),
    "one-vs-rest LinearSVC": OneVsRestClassifier(LinearSVC()),
    "one-vs-one LinearSVC": OneVsOneClassifier(LinearSVC()),
}
for name, clf in models.items():
    pipe = make_pipeline(StandardScaler(), clf).fit(X_tr, y_tr)
    n_est = len(getattr(pipe[-1], "estimators_", [None]))
    print(f"{name:26s} acc={pipe.score(X_te, y_te):.3f}  binary models={n_est}")

soft = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X_tr, y_tr)
proba = soft.predict_proba(X_te)
print("top-2 accuracy:", round(top_k_accuracy_score(y_te, proba, k=2), 3))
print(classification_report(y_te, soft.predict(X_te), digits=3).splitlines()[-2])
```
Output (scikit-learn 1.9.0):
```
softmax (multinomial LR)   acc=0.969  binary models=1
one-vs-rest LinearSVC      acc=0.951  binary models=10
one-vs-one LinearSVC       acc=0.976  binary models=45
top-2 accuracy: 0.998
   macro avg      0.971     0.969     0.969       450
```
In PyTorch, a multiclass head is `nn.Linear(d, K)` with `nn.CrossEntropyLoss()` on raw logits and integer targets. Do not add a softmax layer before the loss ([[loss-functions]]).

## Choosing / trade-offs
- **Need probabilities** (thresholds, cost-sensitive decisions, ranking): softmax. OvR scores do not sum to 1, and normalizing them does not make them calibrated ([[probability-calibration]]).
- **Kernel SVM, small/medium data:** OvO (built into `SVC`). It trains on smaller problems and is often slightly more accurate, at quadratic model count.
- **Binary-only learner or per-class tuning:** OvR. It is linear in K, parallel (`n_jobs`), and each class's model can be inspected or thresholded separately.
- **Many classes:** softmax neural net, or embeddings + k-NN. With very large K (100k+ items), sampled softmax or retrieval-style two-tower training ([[recommender-systems]]).
- **Metric:** macro-F1 when rare classes matter as much as common ones. Weighted-F1 or accuracy when the class mix matches production and only the overall rate matters. Top-k when a human or reranker picks from a short list. Always look at the confusion matrix ([[model-evaluation-and-metrics]]).

## Gotchas
- Labels that look numeric (`1..5` ratings, digit codes) passed to a regressor or treated as ordered by accident. For truly ordered classes, see [[ordinal-regression]].
- Class imbalance is worse with many classes: a handful of classes can hold 90% of rows. Use `class_weight="balanced"`, stratified splits, and macro metrics ([[imbalanced-data]]).
- `predict_proba` on `SVC` uses an internal 5-fold Platt scaling. It is slow, and its argmax can disagree with `predict`. Use `decision_function` for ranking, or calibrate explicitly.
- `OneVsRestClassifier` on a 2D binary indicator `y` silently does **multi-label**, not multiclass. Check the shape of `y`.
- Unseen classes at predict time get forced into a known class. Add an "other/unknown" class or a confidence threshold with OOD checks ([[out-of-distribution-detection]]).
- Accuracy on a 1000-class problem with a dominant class looks high while most classes are never predicted. Count how many classes have recall 0.
- Stratify the split so every class appears in train and test; with rare classes use `StratifiedKFold` ([[data-leakage-and-validation-splits]]).

## Related
- [[multi-label-and-multi-task-learning]] - when one row can have several labels.
- [[ordinal-regression]] - when the classes have an order.
- [[svm-knn-naive-bayes]] - learners whose multiclass behavior differs (OvO in SVC, native in k-NN/NB).
- [[linear-models]] - logistic regression details and regularization.
- [[probability-calibration]] - making multiclass probabilities trustworthy.
- [[model-evaluation-and-metrics]] - confusion matrices, macro/micro averaging.
- [[imbalanced-data]] - rare classes.

## References
- scikit-learn, Multiclass and multioutput algorithms: https://scikit-learn.org/stable/modules/multiclass.html
- scikit-learn, Logistic regression (multinomial): https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
- scikit-learn, Top-k accuracy score: https://scikit-learn.org/stable/modules/model_evaluation.html#top-k-accuracy-score
- Rifkin & Klautau, "In Defense of One-Vs-All Classification", JMLR 2004: https://www.jmlr.org/papers/v5/rifkin04a.html
