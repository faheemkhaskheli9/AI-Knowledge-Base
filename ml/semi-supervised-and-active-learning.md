---
title: Semi-supervised and active learning
category: ml
tags: [semi-supervised, active-learning, self-training, pseudo-labelling, label-propagation, uncertainty-sampling, few-labels, scikit-learn]
use_cases:
  - "train a classifier when only a few hundred examples are labelled"
  - "choose which examples to send to annotators to get the most accuracy per label"
  - "use a large pool of unlabelled data to improve a small labelled set"
  - "cut labelling cost for document review or image annotation"
status: stable
last_verified: 2026-10-03
sources:
  - https://scikit-learn.org/stable/modules/semi_supervised.html
  - https://modal-python.readthedocs.io/en/latest/
---

# Semi-supervised and active learning

## Summary
Two ways to get a good model when labels are expensive and unlabelled data
is cheap. Semi-supervised learning uses the unlabelled data directly
(pseudo-labels, label propagation, consistency training). Active learning
chooses which examples a human should label next, so each label adds the
most information. They combine well: label smartly, then use the rest.

## Key concepts
- Self-training / pseudo-labelling: train on labelled data, predict the
  unlabelled pool, add high-confidence predictions as labels, repeat.
- Label propagation / spreading: build a similarity graph over all points
  and spread labels along it. Works when classes form clusters ("cluster assumption").
- Consistency regularisation (deep learning): predictions should not change
  under augmentation (FixMatch, Mean Teacher).
- Pretrained embeddings are the strongest semi-supervised trick today:
  self-supervised features + a small labelled head.
- Active learning strategies: uncertainty sampling (least confidence,
  smallest margin, entropy), query-by-committee, diversity/core-set
  sampling, expected model change.
- Pool-based loop: model -> score pool -> pick a batch -> human labels ->
  retrain -> stop when the learning curve flattens or budget runs out.

## When to use / scenarios
- Legal e-discovery / document review: technology-assisted review is
  active learning in production.
- Medical imaging and industrial defects: expert labels are costly.
- Text classification for a new intent taxonomy: hundreds of labels, millions of messages.
- Content moderation: route uncertain items to reviewers, label them, retrain.
- NOT worth it when labels are cheap (crowdsourcing, logged outcomes), or
  when a zero/few-shot LLM is already accurate enough; then just label
  with the LLM and review (see [[data-labeling-and-synthetic-data]]).

## Setup & code
```bash
pip install scikit-learn
```
```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier

X, y = load_digits(return_X_y=True)
X = X / 16.0
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
rng = np.random.default_rng(0)
labelled = rng.choice(len(y_tr), 50, replace=False)   # only 50 labels
y_semi = np.full_like(y_tr, -1)                        # -1 = unlabelled
y_semi[labelled] = y_tr[labelled]

base = LogisticRegression(max_iter=2000)
print("supervised, 50 labels :", round(base.fit(X_tr[labelled], y_tr[labelled]).score(X_te, y_te), 3))
st = SelfTrainingClassifier(LogisticRegression(max_iter=2000), threshold=0.9).fit(X_tr, y_semi)
print("self-training         :", round(st.score(X_te, y_te), 3))
ls = LabelSpreading(kernel="knn", n_neighbors=7).fit(X_tr, y_semi)
print("label spreading       :", round(ls.score(X_te, y_te), 3))

# Active learning: uncertainty sampling vs random, 10 rounds x 10 labels
def run(strategy):
    idx = list(labelled[:20])
    for _ in range(10):
        clf = LogisticRegression(max_iter=2000).fit(X_tr[idx], y_tr[idx])
        pool = np.setdiff1d(np.arange(len(y_tr)), idx)
        if strategy == "uncertainty":
            margin = np.sort(clf.predict_proba(X_tr[pool]), axis=1)
            pick = pool[np.argsort(margin[:, -1] - margin[:, -2])[:10]]   # smallest top-2 margin
        else:
            pick = rng.choice(pool, 10, replace=False)
        idx += list(pick)                                                  # "ask the oracle"
    return round(LogisticRegression(max_iter=2000).fit(X_tr[idx], y_tr[idx]).score(X_te, y_te), 3)

print("active (120 labels)   :", run("uncertainty"))
print("random (120 labels)   :", run("random"))
```
Output (scikit-learn 1.9.0), test accuracy:

| Method | Labels | Accuracy |
|---|---|---|
| Supervised logistic regression | 50 | 0.831 |
| Self-training (threshold 0.9) | 50 | 0.817 |
| Label spreading (kNN graph) | 50 | 0.928 |
| Active learning, margin sampling | 120 | 0.933 |
| Random sampling | 120 | 0.904 |

Self-training did *not* help here: a weak base model's confident mistakes
became training labels. Label spreading did, because digits form tight
clusters. Margin sampling beat random by ~3 points at the same label budget.
`LabelSpreading` printed a divide-by-zero RuntimeWarning for points with no
labelled neighbours; it is harmless here but a sign `n_neighbors` is small.

## Choosing / trade-offs
- Start with pretrained embeddings + logistic regression; often beats any
  semi-supervised trick on raw features.
- Self-training: any classifier, easy; needs a calibrated, reasonably good
  base model or it amplifies errors. Use a high threshold and check on a holdout.
- Graph methods: strong when the embedding space clusters by class; O(n^2)
  memory with the RBF kernel, so use `kernel="knn"` on large pools.
- Active learning: biggest wins early in labelling; batch-mode with a
  diversity term avoids picking 10 near-duplicates. Libraries: modAL, or
  built into Label Studio / Prodigy / Argilla workflows.
- LLM as the labeller and active learning to choose what humans verify is a
  common modern combination.

## Gotchas
- Keep a randomly sampled, human-labelled test set. Actively chosen labels
  are biased toward hard cases and give pessimistic, non-representative metrics.
- Uncertainty sampling picks outliers and junk; filter or mix in random samples.
- Class imbalance: the pool may contain very few positives; seed with keyword
  or rule hits so the model sees the rare class at all.
- Pseudo-labels leak into evaluation if the test set is part of the unlabelled pool.
- Annotator throughput, not the algorithm, is often the real bottleneck;
  batch sizes should match how reviewers work.

## Related
- [[data-labeling-and-synthetic-data]] - labelling tools, LLM-assisted labelling.
- [[autoencoders-and-self-supervised-learning]] - pretrained features for few-label tasks.
- [[embeddings]] - embed, then train a small head.
- [[imbalanced-data]] - rare classes in the pool.
- [[bayesian-and-gaussian-processes]] - uncertainty estimates that drive query selection.
- [[legal]] - technology-assisted review.

## References
- scikit-learn semi-supervised: https://scikit-learn.org/stable/modules/semi_supervised.html
- modAL active learning: https://modal-python.readthedocs.io/en/latest/
- Settles, Active Learning Literature Survey (2009); Sohn et al., FixMatch (NeurIPS 2020).
