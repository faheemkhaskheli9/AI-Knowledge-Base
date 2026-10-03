---
title: Multi-label classification and multi-task learning
category: ml
tags: [multi-label, multi-task, classifier-chain, binary-relevance, threshold-tuning, hamming-loss, shared-encoder, multioutput, scikit-learn]
use_cases:
  - "tag a document, image or ticket with several labels at once"
  - "predict several related targets (churn, upsell, complaint) from one model"
  - "pick a decision threshold per label instead of 0.5 for all"
  - "share one encoder across tasks to save compute and help small tasks"
status: draft
last_verified: 2026-10-03
sources:
  - https://scikit-learn.org/stable/modules/multiclass.html
  - https://arxiv.org/abs/1706.05098
---

# Multi-label classification and multi-task learning

## Summary
Multi-label classification assigns any number of labels to one input (a
support ticket can be both "billing" and "urgent"), unlike multi-class, which
picks exactly one. Multi-task learning trains one model on several related
targets at once, usually a shared encoder with one head per task. Both turn
"many separate models" into one system, and both need per-label thresholds
and per-label metrics to be judged honestly.

## Key concepts
- Output layer: multi-label uses one sigmoid per label with binary
  cross-entropy, never a softmax (softmax forces labels to compete).
- Binary relevance: one independent classifier per label. Simple, strong,
  ignores label correlations.
- Classifier chains: label k's model also sees predictions for labels 1..k-1,
  capturing correlations. Order matters; average an ensemble of random orders.
- Label powerset: treat each label combination as a class. Captures all
  correlations, explodes with many labels.
- Metrics: micro-F1 (pooled, dominated by frequent labels), macro-F1 (mean per
  label, sensitive to rare ones), Hamming loss (fraction of wrong label bits),
  subset accuracy (whole label set exactly right, very strict).
- Thresholds: 0.5 is rarely optimal per label; tune each on validation data
  for the metric you report.
- Multi-task: hard parameter sharing (shared trunk, task heads) is the default.
  The loss is a weighted sum; tasks that conflict can hurt each other
  (negative transfer).

## When to use / scenarios
- Content tagging: articles, products, images with several attributes
  ([[nlp-classic-tasks]], [[image-classification]]).
- Customer support: topic + urgency + sentiment from one ticket encoder.
- Healthcare: several diagnoses or findings per record or scan.
- Marketing/CRM: churn, upsell and complaint risk from one customer model.
- Self-driving/vision: detection, segmentation and depth from one backbone.
- NOT for: exactly one answer per input (multi-class with softmax); unrelated
  tasks with plenty of data each (separate models are easier to own and ship);
  hundreds of thousands of labels (extreme multi-label: embeddings + nearest
  neighbors, see [[metric-learning-and-few-shot]]).

## Setup & code
```bash
pip install scikit-learn numpy
```
```python
import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, hamming_loss, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier

X, Y = make_multilabel_classification(n_samples=3000, n_features=30, n_classes=8,
                                      n_labels=2, random_state=0)
X_tr, X_rest, Y_tr, Y_rest = train_test_split(X, Y, test_size=0.4, random_state=0)
X_val, X_te, Y_val, Y_te = train_test_split(X_rest, Y_rest, test_size=0.5, random_state=0)

base = LogisticRegression(max_iter=2000)
br = MultiOutputClassifier(base).fit(X_tr, Y_tr)             # binary relevance
chains = [ClassifierChain(base, order="random", random_state=i).fit(X_tr, Y_tr)
          for i in range(5)]

def report(name, P, thr=0.5):
    Yp = (P >= thr).astype(int)
    print(f"{name:26s} microF1={f1_score(Y_te, Yp, average='micro'):.3f} "
          f"macroF1={f1_score(Y_te, Yp, average='macro'):.3f} "
          f"hamming={hamming_loss(Y_te, Yp):.3f} subset_acc={accuracy_score(Y_te, Yp):.3f}")

proba = lambda m, X: np.column_stack([p[:, 1] for p in m.predict_proba(X)])
P_br = proba(br, X_te)
report("binary relevance @0.5", P_br)
report("chain ensemble @0.5", np.mean([c.predict_proba(X_te) for c in chains], axis=0))

# Per-label threshold tuned on validation for F1
P_val, grid = proba(br, X_val), np.linspace(0.05, 0.95, 19)
thr = np.array([grid[np.argmax([f1_score(Y_val[:, j], P_val[:, j] >= t) for t in grid])]
                for j in range(Y.shape[1])])
report("binary relevance tuned", P_br, thr)
```
Output (scikit-learn 1.9.0):

| Method | micro-F1 | macro-F1 | Hamming | subset acc |
|---|---|---|---|---|
| Binary relevance, 0.5 | 0.545 | 0.537 | 0.189 | 0.317 |
| Chain ensemble (5), 0.5 | 0.508 | 0.500 | 0.193 | 0.347 |
| Binary relevance, tuned thresholds | 0.607 | 0.603 | 0.209 | 0.197 |

The metrics disagree, and that is the lesson. Tuned thresholds (0.20-0.45,
all below 0.5 because each label is present only 18-35% of the time) raise F1
by about 6 points but make more bit errors and fewer exact sets. Chains model
label correlations and win on subset accuracy, but lose on F1 here. Pick the
metric your product needs first, then the method.

For deep multi-task models, the pattern is a shared encoder and a dict of
heads in PyTorch, with the loss `sum(w_k * loss_k)`; see
[[pytorch-basics]] for the training loop.

## Choosing / trade-offs
- Few labels, little correlation: binary relevance with tuned thresholds.
- Strong label dependencies and exact-set correctness matters: chain ensemble
  or label powerset (few labels only).
- Text/images with many labels: one fine-tuned encoder with a multi-label
  sigmoid head beats per-label models in cost and usually quality.
- Multi-task vs separate models: share when tasks are related and some have
  little data; separate when teams, release cycles or data sizes differ a lot.
- Loss weighting across tasks: start equal on normalized losses; try
  uncertainty weighting or GradNorm only if one task dominates.

## Gotchas
- `train_test_split` with `stratify` does not handle label sets; rare labels
  can vanish from validation. Check per-label counts in each split
  (iterative stratification in scikit-multilearn handles it).
- Reporting only micro-F1 hides that rare labels are never predicted. Always
  show per-label precision/recall.
- Tuning thresholds on the test set inflates results; use a validation split.
- Missing labels are not negatives: if annotators tagged only some labels,
  mask unknown ones out of the loss.
- In multi-task training, a task with a larger loss scale or more data
  dominates the shared trunk. Monitor each task's validation metric.

## Related
- [[model-evaluation-and-metrics]] - F1, thresholds, per-class reporting.
- [[imbalanced-data]] - rare labels are an imbalance problem per label.
- [[nlp-classic-tasks]] - text tagging and classification.
- [[image-classification]] - multi-label image tags.
- [[neural-network-fundamentals]] - sigmoid vs softmax outputs.
- [[customer-support]] - ticket routing with several labels.

## References
- scikit-learn multiclass and multioutput algorithms: https://scikit-learn.org/stable/modules/multiclass.html
- Ruder, An Overview of Multi-Task Learning in Deep Neural Networks: https://arxiv.org/abs/1706.05098
