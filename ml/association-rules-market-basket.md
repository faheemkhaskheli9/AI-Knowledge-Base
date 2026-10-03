---
title: Association rules and market basket analysis
category: ml
tags: [association-rules, market-basket, frequent-itemsets, apriori, fp-growth, support, confidence, lift, mlxtend]
use_cases:
  - "find products that are frequently bought together"
  - "suggest cross-sell bundles or 'customers also bought' items"
  - "plan store layout or promotions from transaction data"
  - "find co-occurring symptoms, errors or events in logs"
status: draft
last_verified: 2026-10-03
sources:
  - https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/fpgrowth/
  - https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/
---

# Association rules and market basket analysis

## Summary
Association rule mining finds itemsets that occur together in transactions
and turns them into rules such as {diapers} -> {beer}. It is unsupervised,
explainable and cheap, which makes it a good first step for cross-sell,
bundling and store layout before building a full recommender.

## Key concepts
- Transaction = a set of items (a basket, a session, a patient's codes).
- Support(X) = share of transactions containing X. Filters rare itemsets.
- Confidence(X -> Y) = support(X and Y) / support(X) = P(Y | X).
- Lift(X -> Y) = confidence / support(Y). Lift > 1 means X makes Y more
  likely than its base rate; lift ~1 means independence. Rank by lift (or
  leverage/conviction), not by confidence alone, because popular items have
  high confidence with everything.
- Apriori: level-wise search using "subsets of frequent itemsets are
  frequent". FP-Growth: builds a prefix tree, no candidate generation; faster
  on large data, same output.
- Sequential pattern mining (PrefixSpan) when order matters (A then B).

## When to use / scenarios
- Retail/grocery: bundles, shelf placement, coupon pairing.
- E-commerce: a "frequently bought together" widget for a cold catalogue
  with no user history yet.
- Healthcare: co-occurring diagnoses or drug combinations (as hypotheses).
- IT ops/security: alert or error codes that fire together.
- NOT for: personalised ranking per user (use [[recommender-systems]]),
  or proving one item drives sales of another (that is causal, see
  [[causal-inference-and-uplift]]).

## Setup & code
```bash
pip install mlxtend pandas
```
```python
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

baskets = [
    ["bread", "milk"], ["bread", "diapers", "beer", "eggs"], ["milk", "diapers", "beer", "cola"],
    ["bread", "milk", "diapers", "beer"], ["bread", "milk", "diapers", "cola"],
    ["bread", "milk"], ["diapers", "beer"], ["milk", "cola"],
]
te = TransactionEncoder()
df = pd.DataFrame(te.fit(baskets).transform(baskets), columns=te.columns_)

itemsets = fpgrowth(df, min_support=0.25, use_colnames=True)
rules = association_rules(itemsets, metric="lift", min_threshold=1.2)
cols = ["antecedents", "consequents", "support", "confidence", "lift"]
print(rules.sort_values("lift", ascending=False)[cols].round(2).to_string(index=False))
```
Output (mlxtend 0.25.0), top rows:

| antecedents | consequents | support | confidence | lift |
|---|---|---|---|---|
| {diapers, milk} | {cola} | 0.25 | 0.67 | 1.78 |
| {diapers} | {beer} | 0.50 | 0.80 | 1.60 |
| {beer} | {diapers} | 0.50 | 1.00 | 1.60 |
| {milk} | {cola} | 0.38 | 0.50 | 1.33 |

{beer} -> {diapers} has confidence 1.0: every beer basket had diapers.
Rules come in both directions with the same lift but different confidence.

## Choosing / trade-offs
- `min_support` is the main knob: too high misses niche pairs, too low
  explodes the number of itemsets (memory and time). Start around 0.5-1% on
  real retail data and cap `max_len` at 2-3.
- FP-Growth over Apriori for anything beyond toy data; for very large data
  use Spark MLlib `FPGrowth`.
- Pairwise co-occurrence counts with lift (a SQL self-join on order_id) is
  often enough and easier to maintain than full itemset mining.
- Item-to-item embeddings (item2vec) generalise better to rare items;
  rules are easier to explain to merchandisers.

## Gotchas
- Very popular items (bags, bread) appear in every rule; filter them or
  rank by lift.
- High lift on tiny support is noise; require a minimum count, not just a ratio.
- Rules are correlation, not causation: bundling A with B may not raise B's sales.
- One-hot baskets with thousands of SKUs are memory-hungry; roll up to
  category level or use a sparse DataFrame.
- Time windows matter: a rule from last year's assortment may not hold now.

## Related
- [[recommender-systems]] - personalised recommendations beyond co-occurrence.
- [[clustering]] - segment customers or baskets instead of items.
- [[ecommerce-retail]] - cross-sell and bundling scenarios.
- [[statistics-and-ab-testing]] - test whether a bundle actually lifts sales.

## References
- mlxtend FP-Growth: https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/fpgrowth/
- mlxtend association_rules: https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/
- Agrawal & Srikant, Fast Algorithms for Mining Association Rules (VLDB 1994); Han et al., Mining Frequent Patterns without Candidate Generation (SIGMOD 2000).
