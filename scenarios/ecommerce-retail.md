---
title: E-commerce and retail AI
category: scenarios
tags: [ecommerce, retail, recommendations, search, demand-forecasting, catalog]
use_cases:
  - "add product recommendations to a store"
  - "build semantic or conversational product search"
  - "generate or enrich product descriptions and catalog attributes"
  - "forecast demand and optimise inventory"
  - "detect fake reviews, fraud or returns abuse"
status: stable
last_verified: 2026-10-03
sources:
  - https://scikit-learn.org/stable/
  - https://eur-lex.europa.eu/eli/reg/2022/2065/oj
---

# E-commerce and retail AI

## Summary
Retail has abundant structured data (catalog, orders, clicks) and measurable
outcomes (conversion, AOV, returns). Classic ML covers recommendations,
forecasting and fraud; LLMs add catalog enrichment, conversational search and
support. A/B testing is the arbiter.

## Key concepts
- Cold start: new users and new items have no history; use content
  embeddings and popularity fallbacks ([[recommender-systems]]).
- Search quality is retrieval quality: hybrid lexical + vector, then rerank
  ([[advanced-rag]], [[rerankers]]).
- Catalog data is messy; attribute normalisation is an extraction task
  ([[structured-output]]).
- Metrics: offline (NDCG, recall@k, WAPE for forecasts) must be confirmed by
  online A/B tests.
- Personalisation uses personal data; consent and profiling rules apply
  (check GDPR and local consumer law).

## When to use / scenarios
1. **Product recommendations.** Problem: low cross-sell. Approach: popularity
   and co-purchase baseline -> two-tower embeddings + ranker. Read:
   [[recommender-systems]], [[embeddings]], [[gradient-boosting-tabular]].
2. **Semantic / natural-language search.** Approach: hybrid retrieval,
   rerank, query rewriting by LLM, facets from extracted attributes. Read:
   [[vector-databases]], [[embedding-models]], [[rerankers]].
3. **Conversational shopping assistant.** Approach: agent with catalog search,
   inventory and cart tools; guardrails on price claims. Read: [[agents]],
   [[tool-calling]], [[guardrails-and-safety]].
4. **Product description and attribute generation.** Approach: LLM from
   structured specs and images, human spot checks, no unsupported claims.
   Read: [[vision-language-models]], [[structured-output]],
   [[marketing-content]].
5. **Demand forecasting / replenishment.** Approach: hierarchical time series,
   GBM with promo and calendar features, prediction intervals. Read:
   [[time-series-forecasting]], [[gradient-boosting-tabular]].
6. **Dynamic pricing.** Approach: elasticity models plus guardrails; check
   consumer-protection and competition rules. Read: [[classic-ml-scikit-learn]].
7. **Visual search and shelf analytics.** Approach: embeddings of product
   images; detection for shelf compliance. Read: [[image-classification]],
   [[object-detection]], [[video-analytics]].
8. **Support and returns.** Read: [[customer-support]].
9. **Review analysis and fake-review detection.** Approach: aspect-based
   sentiment, anomaly detection on reviewer behaviour. Read:
   [[nlp-classic-tasks]], [[anomaly-detection]].
10. **Fraud and chargebacks.** Read: [[finance]], [[anomaly-detection]].
11. **Virtual try-on and product imagery.** Read: [[image-generation-models]],
    [[model-licenses]].

## Setup & code
Reference architecture (flagship: recommendations):

```
events (view, cart, buy) -> warehouse -> nightly training
  candidate gen: item embeddings / co-purchase ANN  (~hundreds)
  ranker: GBM on user, item, context features        (top 20)
  business rules: stock, margin, diversity, already-bought filter
  serving: cached per-user lists + real-time rerank; log impressions
  A/B test; monitor CTR, conversion, coverage
```

```python
import numpy as np
# minimal item-to-item recs from embeddings (cosine), as a cold-start fallback
def similar(item_vecs, i, k=10):
    v = item_vecs / np.linalg.norm(item_vecs, axis=1, keepdims=True)
    s = v @ v[i]
    return np.argsort(-s)[1:k + 1]
```

## Choosing / trade-offs
- Build vs buy: managed recommendation/search services speed launch; custom
  wins when you have data volume and differentiation.
- LLM vs classic for catalog: LLM for messy text, classic models for ranking
  and forecasting.
- Freshness vs cost: batch scores are cheap; real-time features cost more.

## Gotchas
- Popularity bias and filter bubbles; add diversity and exploration.
- Training on impressions-only data biases the ranker; log what was shown.
- LLM-generated descriptions that invent specs (waterproof, certified) create
  legal exposure; ground in the spec sheet.
- Review and endorsement rules: fake or AI-generated reviews are restricted in
  some jurisdictions (check FTC and EU consumer rules).
- Price personalisation and transparency duties vary; check local law and the
  EU Digital Services Act where it applies to your marketplace.
- Seasonality and promotions break naive forecasts; hold out a full season.

## Related
- [[data-analytics]] - funnel and cohort analysis.
- [[ai-security-privacy-compliance]] - customer data handling.
- [[mlops-lifecycle]] - retraining and monitoring recommenders.
- [[cost-and-latency]] - serving embeddings at scale.

## References
- scikit-learn docs: https://scikit-learn.org/stable/
- Regulation (EU) 2022/2065 (Digital Services Act): https://eur-lex.europa.eu/eli/reg/2022/2065/oj
