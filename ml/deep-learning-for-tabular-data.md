---
title: Deep learning for tabular data (TabPFN, tabular nets)
category: ml
tags: [tabular, deep-learning, tabpfn, foundation-model, in-context-learning, mlp, ft-transformer, entity-embeddings, pytorch]
use_cases:
  - "get a strong classifier on a small tabular dataset with no tuning"
  - "decide whether a neural net can beat gradient boosting on my table"
  - "combine tabular features with text or image inputs in one model"
  - "learn embeddings for high-cardinality categorical columns"
status: draft
last_verified: 2026-10-04
sources:
  - https://github.com/PriorLabs/TabPFN
  - https://docs.priorlabs.ai/models
  - https://arxiv.org/abs/2106.11959
---

# Deep learning for tabular data (TabPFN, tabular nets)

## Summary
On ordinary tables, tuned gradient-boosted trees are still the default and
the bar to beat. Two neural approaches are worth knowing: tabular foundation
models such as TabPFN, which predict in one forward pass by reading the
training set as context (no training, no tuning), and trained tabular nets
(MLP with embeddings, FT-Transformer) which matter mainly when tables are
combined with text, images or sequences in one end-to-end model.

## Key concepts
- Why trees usually win: tables have heterogeneous, irregular,
  non-smooth features and uninformative columns; trees handle these natively.
- In-context learning for tables: TabPFN is a transformer pretrained on
  millions of synthetic datasets; `fit` stores the data and `predict` runs
  attention over training rows and test rows together.
- Size limits: each TabPFN checkpoint supports a maximum number of rows,
  features and classes, and cost grows with the training set because it is
  re-read at prediction time. Limits have grown a lot between releases;
  read the current Models page.
- Licences differ per checkpoint: the code is Apache 2.0, TabPFN-2 weights
  are Apache 2.0 plus attribution, while newer default weights (2.5 onward)
  are non-commercial and need a commercial licence for production.
- Trained tabular nets: numeric features scaled (quantile/standard),
  categoricals mapped to learned embeddings, then an MLP, ResNet-style MLP or
  FT-Transformer (one token per feature, then transformer layers).
- Multimodal: concatenate a tabular embedding with a text/image encoder
  output and train jointly.

## When to use / scenarios
- Small data (hundreds to tens of thousands of rows) in clinical studies,
  lab experiments, B2B sales or pilot projects: try TabPFN first, it often
  matches or beats tuned GBMs with zero tuning.
- Product listing with title, photo and attributes; claims with notes and
  structured fields: a joint neural model over all inputs.
- Very high-cardinality IDs (users, items, stores): learned entity embeddings
  reusable in other models ([[recommender-systems]]).
- NOT the default for a large plain table with a tight latency budget: use
  [[gradient-boosting-tabular]]. Not when the TabPFN weight licence does not
  fit a commercial use case and the v2 weights fall short.

## Setup & code
```bash
pip install tabpfn scikit-learn   # needs PyTorch; a GPU is much faster for larger data
```
```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state=0)

clf = TabPFNClassifier()          # sklearn API; "fit" stores the context, no gradient steps
clf.fit(X_tr, y_tr)
print("AUC:", roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))
```
The default checkpoint is downloaded on first use and newer ones require a
one-time licence acceptance on Hugging Face; the repo README shows how to load
the TabPFN-2 weights instead. `TabPFNRegressor` works the same way.

A trained tabular MLP with categorical embeddings (PyTorch sketch):
```python
import torch, torch.nn as nn

class TabMLP(nn.Module):
    def __init__(self, n_num, cat_cardinalities, emb_dim=8, hidden=128, n_out=1):
        super().__init__()
        self.embs = nn.ModuleList(nn.Embedding(c, emb_dim) for c in cat_cardinalities)
        d_in = n_num + emb_dim * len(cat_cardinalities)
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_out))

    def forward(self, x_num, x_cat):          # x_cat: LongTensor of category indices
        embs = [e(x_cat[:, i]) for i, e in enumerate(self.embs)]
        return self.net(torch.cat([x_num, *embs], dim=1))

model = TabMLP(n_num=5, cat_cardinalities=[10, 300])
print(model(torch.randn(4, 5), torch.randint(0, 10, (4, 2))).shape)   # torch.Size([4, 1])
```
Train it with the usual loop from [[pytorch-basics]]; reserve index 0 of each
embedding for "unknown/rare" categories seen at inference.

## Choosing / trade-offs
- Default for tabular: [[gradient-boosting-tabular]] (fast, cheap, robust).
- Small data, no time to tune, licence fits: TabPFN; check its prediction
  latency since the training set is re-processed per prediction batch.
- Need squeezed accuracy: ensemble GBM + TabPFN or a tabular net; diverse
  model types average well ([[ensemble-methods]]).
- Mixed modalities or end-to-end with other nets: trained tabular net.
- Cannot ship a large model: TabPFN can be distilled into a small MLP or tree
  ensemble ([[knowledge-distillation-and-compression]]).

## Gotchas
- Comparing a tuned neural net against a default GBM (or the reverse) is the
  most common unfair benchmark; tune both or neither.
- Unscaled numeric features and raw high-cardinality IDs break MLPs; trees
  are insensitive to both.
- TabPFN beyond its supported rows/features needs subsampling or extensions;
  check the limits of the checkpoint in use.
- Licence: confirm the weight licence before a commercial deployment, not after.
- Neural nets on tables need early stopping and careful validation; they
  overfit small tables easily ([[deep-learning-training]]).

## Related
- [[gradient-boosting-tabular]] - the default to beat.
- [[automl]] - automated search that may include neural models.
- [[feature-engineering]] - encoding and scaling inputs.
- [[embeddings]] - learned entity embeddings for categories.
- [[pytorch-basics]] - training loop for the MLP above.
- [[model-licenses]] - checking weight licences.

## References
- TabPFN repo and licences: https://github.com/PriorLabs/TabPFN
- TabPFN checkpoints and limits: https://docs.priorlabs.ai/models
- Hollmann et al., "Accurate predictions on small data with a tabular foundation model", Nature (2025)
- Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data" (FT-Transformer): https://arxiv.org/abs/2106.11959
- Grinsztajn et al., "Why do tree-based models still outperform deep learning on tabular data?": https://arxiv.org/abs/2207.08815
