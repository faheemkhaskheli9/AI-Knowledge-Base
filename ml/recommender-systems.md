---
title: Recommender systems
category: ml
tags: [recommendation, collaborative-filtering, two-tower, ranking, implicit, embeddings, cold-start]
use_cases:
  - "recommend products to shoppers on an ecommerce site"
  - "build a 'because you watched' feed for a video or music app"
  - "suggest related articles or courses based on user history"
  - "solve the cold-start problem for new users and new items"
  - "rank search results or feed items by predicted engagement"
status: draft
last_verified: 2026-10-03
sources:
  - https://benfred.github.io/implicit/
  - https://www.tensorflow.org/recommenders
  - https://developers.google.com/machine-learning/recommendation
---

# Recommender systems

## Summary
Recommenders predict which items a user will engage with from interaction history, item content and context. Production systems are usually two-stage: a cheap candidate generator retrieves hundreds of items, then a ranker scores them with richer features. Start with popularity and item-item similarity; add learned embeddings or a ranker only when logged data and metrics justify it.

## Key concepts
- Feedback types: explicit (ratings) vs implicit (clicks, purchases, watch time); implicit is far more common and treats absence as weak signal, not dislike.
- Collaborative filtering: matrix factorisation (ALS, BPR) learns user and item vectors from interactions.
- Content-based: similarity from item text/image embeddings (see [[embedding-models]]); solves item cold start.
- Two-tower retrieval: user and item encoders trained contrastively; serve item vectors from an ANN index (see [[vector-databases]]).
- Ranking stage: gradient boosting or neural ranker with LambdaRank/pairwise loss on user, item and context features.
- Metrics: offline Recall@K, NDCG@K, MAP; online A/B on CTR, conversion, retention. Also diversity, coverage, novelty.
- Cold start: new users (use popularity, onboarding, context), new items (content features, exploration/bandits).

## When to use / scenarios
- Ecommerce: "customers also bought", personalised home page; Media: next-video/song; Education: next lesson; Marketplaces/jobs: matching.
- Sparse data (< a few thousand interactions): popularity + rules + content similarity beat learned models.
- LLM-era option: use an LLM to explain or re-rank a short candidate list, or embed item text; do not use an LLM as the main retrieval over millions of items (cost, latency).
- NOT for: pure search with a query (see [[advanced-rag]] / search stacks), or one-off segmentation (clustering in [[classic-ml-scikit-learn]]).

## Setup & code
```bash
pip install implicit scipy numpy
```
```python
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

# rows=users, cols=items, values=interaction strength
rng = np.random.default_rng(0)
user_items = csr_matrix((rng.random((100, 50)) > 0.9).astype(np.float32))

model = AlternatingLeastSquares(factors=32, regularization=0.1, iterations=15)
model.fit(user_items)
ids, scores = model.recommend(0, user_items[0], N=5, filter_already_liked_items=True)
print(ids, scores)
```
Check the `implicit` docs for the current `fit`/`recommend` signature, which has changed across releases (older versions expected an item-user matrix).

## Choosing / trade-offs
- Popularity / item-item co-occurrence: cheapest, hard to beat on sparse data; start here.
- ALS/BPR (implicit): fast, scalable CF with no features; weak on cold start.
- Two-tower + ANN: scalable retrieval with side features; needs more data and engineering (TensorFlow Recommenders, PyTorch).
- Ranker (LightGBM LambdaRank): best place to add context and business features; see [[gradient-boosting-tabular]].
- Sequence models (SASRec-style transformers): capture order of actions; costlier.
- Exploration (bandits) trades short-term CTR for learning; needed to escape feedback loops.

## Gotchas
- Random train/test split leaks the future; split by time and evaluate next-item prediction.
- Offline metrics on logged data are biased by the previous recommender (popularity and exposure bias); confirm with A/B tests.
- Optimising clicks produces clickbait and filter bubbles; add diversity and long-term metrics.
- Filter already-seen/purchased and out-of-stock items at serving time.
- Embeddings go stale; schedule retraining and index rebuilds.
- Privacy: behavioural data is personal data; honour consent and deletion requests.

## Related
- [[embedding-models]] - item/user text embeddings for content-based retrieval.
- [[vector-databases]] - ANN serving of item vectors.
- [[gradient-boosting-tabular]] - ranking stage.
- [[ecommerce-retail]] - scenario context.
- [[experiment-tracking]] - compare offline runs.

## References
- https://developers.google.com/machine-learning/recommendation
- https://benfred.github.io/implicit/
- https://www.tensorflow.org/recommenders
