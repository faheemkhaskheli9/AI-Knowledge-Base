---
title: Tabular foundation models (TabPFN, TabICL, in-context tabular learning)
category: ml
tags: [tabular, foundation-model, in-context-learning, tabpfn, tabicl, mitra, autogluon, small-data, gradient-boosting, tabarena]
use_cases:
  - "get a strong baseline on a small tabular dataset in minutes with no hyperparameter tuning"
  - "decide whether TabPFN or TabICL beats XGBoost/LightGBM on my table before committing"
  - "check whether a tabular foundation model's licence and latency allow production use"
  - "add a pretrained tabular model to an AutoGluon ensemble"
status: draft
last_verified: 2026-10-15
sources:
  - https://github.com/PriorLabs/TabPFN
  - https://github.com/PriorLabs/tabpfn-extensions
  - https://github.com/soda-inria/tabicl
  - https://arxiv.org/abs/2502.05564
  - https://arxiv.org/abs/2506.16791
  - https://auto.gluon.ai/stable/tutorials/tabular/tabular-foundational-models.html
---

# Tabular foundation models (TabPFN, TabICL, in-context tabular learning)

## Summary
A tabular foundation model (TFM) is a transformer pretrained on millions of
synthetic tables so that, given a new dataset's training rows as context, it
predicts test rows in one forward pass: `fit` stores the data, `predict` does
the "learning". There is no gradient training and no hyperparameter search on
your data. On small and medium tables they often match or beat tuned gradient
boosting; on large tables, under tight latency budgets, or where weight
licences forbid commercial use, gradient boosting remains the default.

## Key concepts
- **Prior-fitted networks (PFNs).** Pretraining samples synthetic datasets
  from a prior (random causal graphs, random functions, noise), and trains the
  network to predict held-out rows given the rest. The result approximates
  Bayesian prediction under that prior, so it is well calibrated by default.
- **In-context learning (ICL) for tables.** Attention runs across features
  (within a row) and across rows (test rows attend to training rows). Cost
  grows with the number of training rows, and the training set is re-read for
  every prediction batch unless the model caches it (TabICL offers a KV cache).
- **Main models (Oct 2026).**
  - **TabPFN** (Prior Labs). Code Apache 2.0. TabPFN-2 weights: Prior Labs
    licence (Apache 2.0 plus attribution). TabPFN-2.5, 2.6, 3 and 3.5 weights:
    non-commercial; TabPFN-3.5 is the package default. The README lists
    TabPFN-3.5 up to 1,000,000 rows and 20,000 features, a GPU recommended
    (about 8 GB VRAM, 16 GB for some large sets), and a CPU cap of 5,000 samples
    for v3+ (1,000 for older) unless overridden. A hosted API client
    (`tabpfn-client`) serves users without a GPU.
  - **TabICL / TabICLv2** (Inria Soda). BSD-3-Clause code and open
    pretraining. The README reports strong results with 300 to 100,000
    training rows and up to 2,000 features, KV caching for repeated
    prediction, and optional extras for forecasting, SHAP and fine-tuning.
  - **Mitra** (Amazon, in AutoGluon). Weights under Apache 2.0, pretrained on
    synthetic data with an emphasis on fine-tuning; runs zero-shot or
    fine-tuned inside AutoGluon next to TabPFNv2 and TabICL.
- **Zero-shot vs fine-tuned.** All of these work zero-shot; fine-tuning the
  TFM on one large dataset can add accuracy at the cost of training time.
- **Ensembling.** TFMs make errors different from trees, so a GBM + TFM blend
  often beats either alone ([[ensemble-methods]]).
- **Extensions.** `tabpfn-extensions` adds SHAP explanations, outlier
  detection and synthetic data, embeddings, and many-class wrappers beyond the
  built-in class limit.

## When to use / scenarios
- **Beats or matches gradient boosting when:** a few hundred to tens of
  thousands of rows, mostly numeric or low-cardinality features, and no time
  to tune. TabArena (NeurIPS 2025 benchmark) found foundation models
  especially strong on smaller datasets; check the live leaderboard at
  tabarena.ai for current standings.
- Clinical or lab studies, pilot projects, B2B churn or lead scoring with
  hundreds of customers, materials or process experiments: small data where
  tuning a GBM overfits the validation split.
- Quick feasibility check: "is there any signal in this table?" in minutes.
- Calibrated probabilities on small data without a separate calibration step
  (still verify with [[probability-calibration]]).
- **Not when:** millions of rows and a GBM already works; a hard latency or
  CPU-only budget at serving time; heavy free text or very high-cardinality
  IDs (use [[deep-learning-for-tabular-data]] or engineered features);
  commercial use where only non-commercial weights fit your accuracy target;
  strict time-ordered data where the model's assumption of exchangeable rows
  hides drift (validate with a time split anyway).

## Setup & code
```bash
pip install tabpfn          # Python 3.10+, pulls PyTorch; GPU recommended
pip install tabicl          # BSD-3-Clause alternative
pip install "autogluon.tabular[mitra,tabicl,tabpfn]"   # inside AutoGluon
```
<!-- skip-check: needs torch and a model checkpoint download -->
```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from tabicl import TabICLClassifier

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state=0)

for name, clf in [
    ("tabpfn-default", TabPFNClassifier()),                     # non-commercial weights
    ("tabpfn-v2", TabPFNClassifier.create_default_for_version(ModelVersion.V2)),
    ("tabicl", TabICLClassifier()),                             # BSD-3-Clause
]:
    clf.fit(X_tr, y_tr)                                          # stores context only
    print(name, roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))
```

The core mechanism, without a pretrained network: a predictor that "fits" by
storing rows and predicts by attention over them. The second half is the
comparison harness to run before choosing a TFM over a GBM: repeated CV on
learning-curve subsamples, same folds for every model.
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

class AttentionICL:
    """fit = store context; predict = softmax attention from test rows to train rows."""
    def __init__(self, temperature=1.0):
        self.t = temperature
    def fit(self, X, y):
        self.sc = StandardScaler().fit(X)
        self.K, self.V = self.sc.transform(X), np.eye(2)[y]   # keys and one-hot values
        return self
    def predict_proba(self, X):
        Q = self.sc.transform(X)
        d2 = ((Q[:, None, :] - self.K[None, :, :]) ** 2).sum(-1)  # distance as negative score
        s = -d2 / (self.t * Q.shape[1])
        w = np.exp(s - s.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
        return w @ self.V                                    # attention-weighted labels

X, y = make_classification(n_samples=3000, n_features=12, n_informative=6,
                           class_sep=0.8, random_state=0)
rng = np.random.default_rng(0)
for n in (100, 300, 1000, 3000):                              # learning curve
    idx = rng.choice(len(y), n, replace=False)
    res = {"icl": [], "gbm": []}
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X[idx], y[idx]):
        Xtr, ytr, Xte, yte = X[idx][tr], y[idx][tr], X[idx][te], y[idx][te]
        res["icl"].append(roc_auc_score(yte, AttentionICL(0.5).fit(Xtr, ytr).predict_proba(Xte)[:, 1]))
        gbm = HistGradientBoostingClassifier(random_state=0).fit(Xtr, ytr)
        res["gbm"].append(roc_auc_score(yte, gbm.predict_proba(Xte)[:, 1]))
    print(n, {k: round(float(np.mean(v)), 3) for k, v in res.items()})
```
The toy ICL model is a kernel smoother, not TabPFN; it shows why such models
need no training and why their cost scales with context size. Swap in
`TabPFNClassifier` / `TabICLClassifier` and a tuned GBM to make the real
decision on your own data.

## Choosing / trade-offs
| Situation | Lean towards |
|---|---|
| < ~10k rows, little time, licence fits | TabPFN or TabICL zero-shot |
| Commercial product, no licence deal | TabICL (BSD-3), TabPFN-2 weights, Mitra (Apache 2.0), or GBM |
| Millions of rows, CPU serving, ms latency | [[gradient-boosting-tabular]] |
| Max accuracy, offline scoring | GBM + TFM ensemble, e.g. via [[automl]] (AutoGluon) |
| Need a small, fast deployable model | Distil the TFM into a GBM or MLP ([[knowledge-distillation-and-compression]]) |
| No GPU at all | TabPFN hosted client (data leaves your network) or GBM |

- **Latency and cost:** prediction cost scales with training rows times test
  rows; batch predictions and use caching where offered.
- **Privacy:** the hosted TabPFN client sends your rows to a third party;
  check data terms before using it on personal data
  ([[ai-security-privacy-compliance]]).
- **Explainability:** TFMs have no tree structure; use SHAP via the
  extensions or [[model-interpretability]] tools, which is slower than
  TreeSHAP on a GBM.

## Gotchas
- Unfair benchmarks: a default GBM vs a TFM on one split proves little. Use
  repeated CV, tune the GBM, and compare on the same folds
  ([[model-selection-and-comparison]]).
- Licence drift: newer TabPFN releases changed weight licences; the default
  checkpoint in a `pip install` may be non-commercial. Pin the version and
  the checkpoint, and record the licence ([[model-licenses]]).
- Exceeding the pretraining limits (rows, features, classes) silently or
  with an override degrades accuracy; subsample, select features
  ([[feature-selection]]), or use the many-class wrapper.
- CPU guardrails: TabPFN refuses large datasets on CPU by default; the
  override makes it run, not fast.
- Leakage still applies: the context must not contain rows from the test
  period or the same entity ([[data-leakage-and-validation-splits]]).
- Rows are treated as exchangeable; with drift, weighting recent rows or a
  time split matters more than the model choice.
- First call downloads a checkpoint (some need a Hugging Face licence
  acceptance); air-gapped deployments must pre-fetch and pin it.

## Related
- [[deep-learning-for-tabular-data]] - trained tabular nets and multimodal tables.
- [[gradient-boosting-tabular]] - the default to beat.
- [[automl]] - AutoGluon ensembles that include TabPFN, TabICL and Mitra.
- [[ensemble-methods]] - blending TFMs with trees.
- [[model-licenses]] - checking weight licences before production.
- [[probability-calibration]] - verifying calibrated outputs.
- [[transformers-and-attention]] - the architecture underneath.

## References
- TabPFN repository, licences, limits, hosted client: https://github.com/PriorLabs/TabPFN
- TabPFN extensions: https://github.com/PriorLabs/tabpfn-extensions
- Hollmann et al., "Accurate predictions on small data with a tabular foundation model", Nature (2025)
- TabICL repository (TabICL and TabICLv2): https://github.com/soda-inria/tabicl
- Qu et al., "TabICL: A Tabular Foundation Model for In-Context Learning on Large Data": https://arxiv.org/abs/2502.05564
- Erickson et al., "TabArena: A Living Benchmark for Machine Learning on Tabular Data": https://arxiv.org/abs/2506.16791
- AutoGluon tabular foundation models tutorial: https://auto.gluon.ai/stable/tutorials/tabular/tabular-foundational-models.html
