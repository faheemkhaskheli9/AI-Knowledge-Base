---
title: Learning to rank
category: ml
tags: [learning-to-rank, ltr, lambdarank, lambdamart, ndcg, search-ranking, lightgbm, xgboost, pairwise, listwise]
use_cases:
  - "order search results or product listings by relevance using click or rating data"
  - "re-rank candidates from a retriever or recommender with many features"
  - "optimize NDCG or MRR directly instead of predicting a score per item"
  - "rank leads, tickets or job candidates for a team to work through top-down"
status: draft
last_verified: 2026-10-03
sources:
  - https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html
  - https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html
  - https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/
---

# Learning to rank

## Summary
Learning to rank (LTR) trains a model to order a list of items for a query
(a search, a user, a session) so the most relevant come first. Instead of
predicting each item's label in isolation, it optimizes the order within each
query group, usually for NDCG. Gradient-boosted LambdaMART (LightGBM, XGBoost)
is the standard, strong baseline for feature-based ranking.

## Key concepts
- Query group: the set of candidates ranked together. Only order within a
  group matters; scores are not comparable across groups.
- Approaches: pointwise (regress/classify each item), pairwise (learn which of
  two items is better, RankNet), listwise (optimize a list metric, LambdaRank/
  LambdaMART, which weights pair swaps by their effect on NDCG).
- Labels: graded relevance (0-4) from human judges, or implicit feedback
  (clicks, purchases, dwell time), which is biased by position.
- Metrics: NDCG@k (graded, discounted by position), MRR (first relevant item),
  MAP and recall@k. Always "@k" for the number of slots users actually see.
- Two-stage systems: a cheap retriever (BM25, vector search, candidate
  generator) returns hundreds; the ranker orders the top of that list.
- Neural alternatives: cross-encoder rerankers for text relevance
  ([[rerankers]]), two-tower models for retrieval ([[recommender-systems]]).

## When to use / scenarios
- E-commerce search and category pages ([[ecommerce-retail]]).
- Final ranking stage of a recommender, with user, item and context features.
- Job/candidate matching, lead scoring for sales queues, support ticket triage.
- RAG: rank retrieved chunks using features beyond text similarity (freshness,
  source authority, click data), alongside a reranker ([[advanced-rag]]).
- NOT for: one global score where items never compete in a list (plain
  classification/regression); pure text-to-text relevance with no other
  features (a cross-encoder reranker is simpler and better).

## Setup & code
```bash
pip install lightgbm scikit-learn numpy
```
```python
import numpy as np
import lightgbm as lgb
from sklearn.metrics import ndcg_score

rng = np.random.default_rng(0)

# Synthetic search logs: queries x 20 candidates, graded relevance 0-4.
# Relevance depends on query-relative features, so absolute scores mislead.
def make(nq, docs=20):
    X, y, groups = [], [], []
    for _ in range(nq):
        f = rng.normal(size=(docs, 5))
        bias = rng.normal(0, 2)                     # per-query offset in feature 0
        f[:, 0] += bias
        s = (f[:, 0] - bias) + 0.5 * f[:, 1] * f[:, 2] + rng.normal(0, 0.5, docs)
        rel = np.digitize(s, np.quantile(s, [0.5, 0.75, 0.9, 0.97]))
        X.append(f); y.append(rel); groups.append(docs)
    return np.vstack(X), np.concatenate(y), np.array(groups)

X_tr, y_tr, g_tr = make(600)
X_te, y_te, g_te = make(200)

def mean_ndcg(scores, y, groups, k=10):
    out, i = [], 0
    for g in groups:
        out.append(ndcg_score([y[i:i + g]], [scores[i:i + g]], k=k)); i += g
    return np.mean(out)

ranker = lgb.LGBMRanker(objective="lambdarank", n_estimators=300,
                        learning_rate=0.05, verbose=-1, random_state=0)
ranker.fit(X_tr, y_tr, group=g_tr)          # rows must be sorted by query
pointwise = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                              verbose=-1, random_state=0).fit(X_tr, y_tr)

print(f"NDCG@10 lambdarank={mean_ndcg(ranker.predict(X_te), y_te, g_te):.3f} "
      f"pointwise={mean_ndcg(pointwise.predict(X_te), y_te, g_te):.3f} "
      f"random={mean_ndcg(rng.random(len(y_te)), y_te, g_te):.3f}")
```
Output (LightGBM 4.7.0, scikit-learn 1.9.0): NDCG@10 is 0.881 for LambdaRank,
0.726 for a pointwise regressor on the same features, 0.391 for random order.
The pointwise model is hurt by the per-query offset, which the ranker ignores
because it only compares items within a query. Normalizing features per query
(rank or z-score within the group) narrows that gap, and is worth doing for
both.

## Choosing / trade-offs
- Tabular features (price, CTR, recency, BM25 score, embeddings similarity):
  LambdaMART in LightGBM/XGBoost. Fast, strong, explainable with SHAP.
- Mostly text relevance: cross-encoder reranker; feed its score as one feature
  to LambdaMART if you also have business features.
- Very large scale with user histories: neural rankers (DLRM-style, sequence
  models) once GBDT has plateaued.
- Labels: human judgments are clean but costly; clicks are plentiful but need
  position-bias correction (randomized swaps, inverse propensity weighting).
- Metric choice drives the objective: NDCG for graded relevance, MRR when only
  the first good result matters.

## Gotchas
- `group` must list the size of each query block in the order the rows
  appear. Unsorted rows silently train on mixed-up groups.
- Split train/validation/test by query (`GroupKFold`), never by row, or the
  same query leaks across sets.
- Queries where every candidate has the same label teach nothing; drop them.
- Training only on items that were shown (logged) means the ranker never
  learns about items the old system hid. Mix in some exploration.
- LightGBM's default `label_gain` covers labels up to 30; larger integer
  labels need a custom `label_gain`.
- Offline NDCG gains do not always become online gains; confirm with an A/B
  test ([[statistics-and-ab-testing]]).

## Related
- [[gradient-boosting-tabular]] - the model family behind LambdaMART.
- [[recommender-systems]] - candidate generation before ranking.
- [[rerankers]] - neural text rerankers, often a feature or alternative.
- [[advanced-rag]] - ranking retrieved context.
- [[model-evaluation-and-metrics]] - NDCG, MRR, MAP.
- [[ecommerce-retail]] - search and listing ranking.

## References
- LightGBM LGBMRanker: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html
- XGBoost learning to rank tutorial: https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html
- Burges, From RankNet to LambdaRank to LambdaMART: https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/
