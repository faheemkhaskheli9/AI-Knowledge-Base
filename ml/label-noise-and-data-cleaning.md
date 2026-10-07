---
title: Label noise and data cleaning
category: ml
tags: [label-noise, confident-learning, cleanlab, data-quality, mislabeled-data, out-of-fold, data-centric-ai, deduplication]
use_cases:
  - "find mislabeled rows in a training set before retraining"
  - "decide which annotations to send back for review"
  - "train a usable model on crowdsourced or weakly labeled data"
  - "audit a dataset for duplicates, outliers and label errors"
status: stable
last_verified: 2026-10-03
sources:
  - https://arxiv.org/abs/1911.00068
  - https://arxiv.org/abs/2103.14749
  - https://docs.cleanlab.ai/
---

# Label noise and data cleaning

## Summary
Real labels are wrong more often than people think; even benchmark test sets
contain a few percent errors. Label noise lowers accuracy, and noise in the
test set makes evaluation itself wrong. Confident learning finds likely
mislabeled rows using a model's out-of-fold predicted probabilities, so a
reviewer can check the top suspects instead of the whole dataset. Fixing data
often beats tuning the model.

## Key concepts
- Noise types: random (uniform flips), class-conditional (cat often labeled
  dog), and instance-dependent (hard or ambiguous examples). The last is most
  common and hardest.
- Out-of-fold probabilities: each row is scored by a model that never trained
  on it (`cross_val_predict`). In-sample probabilities memorize the noise.
- Confident learning: per-class threshold = average self-confidence of rows
  given that label; a row is a suspect if another class clears its threshold
  and beats the given label. Rank suspects by low self-confidence.
- Options after detection: relabel (best), drop, or down-weight suspects.
- Noise-robust training: label smoothing, robust losses (generalized
  cross-entropy, symmetric CE), co-teaching, early stopping (nets fit clean
  rows first, noise later).
- Other data issues worth the same audit: exact/near duplicates (leak across
  splits), outliers, and ambiguous rows with more than one valid label.

## When to use / scenarios
- Crowdsourced, outsourced or auto-generated labels (weak supervision, LLM
  labels, see [[data-labeling-and-synthetic-data]]).
- Healthcare, legal, finance: labels from busy experts with disagreement.
- Support ticket and product categorization where taxonomies changed over
  time.
- Before a big retrain or before trusting a test set to compare models.
- NOT for: tiny datasets where you can just review everything; cases where
  "noise" is really two valid labels (switch to multi-label, see
  [[multi-label-and-multi-task-learning]]).

## Setup & code
```bash
pip install scikit-learn numpy   # or: pip install cleanlab
```
```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)
X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# Corrupt 15% of training labels
y_noisy = y_tr.copy()
flip = rng.random(len(y_tr)) < 0.15
y_noisy[flip] = (y_tr[flip] + rng.integers(1, 10, flip.sum())) % 10

model = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))

# Simplified confident learning on out-of-fold probabilities
P = cross_val_predict(model(), X_tr, y_noisy, cv=5, method="predict_proba")
t = np.array([P[y_noisy == k, k].mean() for k in range(P.shape[1])])  # per-class thresholds
above = P >= t
best = np.where(above, P, -1).argmax(1)
suspect = above.any(1) & (best != y_noisy)

tp = (suspect & flip).sum()
print(f"noisy={flip.sum()} flagged={suspect.sum()} "
      f"precision={tp/suspect.sum():.3f} recall={tp/flip.sum():.3f}")
acc = lambda Xf, yf: model().fit(Xf, yf).score(X_te, y_te)
print(f"test acc clean={acc(X_tr, y_tr):.3f} noisy={acc(X_tr, y_noisy):.3f} "
      f"noisy minus flagged={acc(X_tr[~suspect], y_noisy[~suspect]):.3f}")
```
Output (scikit-learn 1.9.0): of 181 corrupted rows, 143 rows are flagged with
precision 0.923 and recall 0.729. Test accuracy is 0.972 with clean labels,
0.939 with noisy labels and 0.954 after dropping the flagged rows: cleaning
recovers about half the loss. Relabeling the suspects instead of dropping them
would recover more.

`cleanlab.filter.find_label_issues(labels, pred_probs)` implements the full
method (joint count matrix, several ranking options) and also finds outliers
and near duplicates through `cleanlab.Datalab`.

## Choosing / trade-offs
- Budget for review: rank suspects and send the top N to humans; this is the
  highest-value use.
- No review budget: drop suspects if data is plentiful; down-weight them if
  data is scarce.
- Deep nets on big noisy data: noise-robust losses or co-teaching, plus early
  stopping on a clean validation set.
- Better base model, better detection: detection quality is bounded by the
  model producing the probabilities; use the strongest one you can afford,
  including pretrained embeddings plus a linear model.

## Gotchas
- Using in-sample probabilities finds almost nothing: the model has memorized
  the wrong labels. Always out-of-fold.
- Auto-dropping removes hard but correct examples (rare classes, edge cases)
  too. Precision here was 92%, not 100%; check what is being dropped per class.
- Cleaning the training set but not the test set leaves evaluation noisy;
  audit the test set with a human pass.
- Systematic noise (one annotator always confuses two classes) is better fixed
  at the source: guidelines, examples, or merging the classes.
- Duplicates across train and test inflate scores silently; deduplicate before
  splitting.

## Related
- [[data-labeling-and-synthetic-data]] - where labels come from.
- [[semi-supervised-and-active-learning]] - choosing what to label next.
- [[model-evaluation-and-metrics]] - noisy test sets distort metrics.
- [[imbalanced-data]] - rare classes get flagged more often.
- [[conformal-prediction-and-uncertainty]] - another use of held-out probabilities.

## References
- Northcutt et al., Confident Learning: https://arxiv.org/abs/1911.00068
- Northcutt et al., Pervasive Label Errors in Test Sets: https://arxiv.org/abs/2103.14749
- cleanlab documentation: https://docs.cleanlab.ai/
