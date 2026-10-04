---
title: Apriori from scratch (frequent itemsets, candidate pruning, support, confidence, lift)
category: ml
tags: [apriori, association-rules, frequent-itemsets, market-basket-analysis, support, confidence, lift, candidate-generation, pruning, python, from-scratch, ml-basics]
use_cases:
  - "implement the Apriori algorithm in plain Python to find frequent itemsets and association rules"
  - "find which products are bought together in transaction data (market basket analysis)"
  - "understand why the Apriori property makes frequent-itemset mining tractable"
  - "explain support, confidence and lift, and why confidence alone misleads, in an interview"
status: draft
last_verified: 2026-10-04
sources:
  - https://www.vldb.org/conf/1994/P487.PDF
  - https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/
  - https://doi.org/10.1145/335191.335372
---

# Apriori from scratch (frequent itemsets, candidate pruning, support, confidence, lift)

## Summary
Apriori finds every itemset that appears in at least a minimum fraction of transactions, then turns those itemsets into rules of the form "baskets with X also tend to contain Y". It rests on one property: every subset of a frequent itemset must itself be frequent. The algorithm grows itemsets one item at a time and only counts candidates whose subsets have all survived. Below, 2000 synthetic baskets over 10 items contain four planted buying patterns. At 5% minimum support, Apriori counts 139 candidate itemsets instead of the 1013 a brute-force pass over every multi-item set would need, and finds 57 frequent itemsets. Ranking the rules by lift recovers the planted patterns, with lift 1.8 to 2.9. A rule like `apples → bread` has 37% confidence but a lift of 1.03, which means no association: bread is in 36% of baskets anyway.

## Key concepts
- **Support.** `supp(X)` = the fraction of transactions that contain every item in X.
- **Apriori property (anti-monotonicity).** If X is infrequent, every superset of X is infrequent. Equivalently, every subset of a frequent set is frequent. This is what lets the search skip most of the 2ⁿ itemsets.
- **Level-wise search.** Level k: *join* frequent (k−1)-itemsets into k-item candidates, *prune* any candidate with an infrequent (k−1)-subset, then scan the data once to count the survivors. Stop when a level is empty.
- **Confidence.** `conf(X → Y) = supp(X ∪ Y) / supp(X)`, an estimate of `P(Y | X)`.
- **Lift.** `lift(X → Y) = conf(X → Y) / supp(Y)`. A value of 1 means X and Y are independent, above 1 means they co-occur more than chance, and below 1 means they substitute for each other. Lift is symmetric in X and Y.
- **Rule generation.** Every split of a frequent itemset into a non-empty LHS and RHS is a candidate rule. Its support values are all already known, so this step needs no further data scans.

## When to use / scenarios
- Learning: a clean example of search-space pruning with a monotone property, which is the same idea behind many data-mining algorithms.
- Interviews: "explain support/confidence/lift", "why can't you enumerate all itemsets", "Apriori vs FP-growth".
- Practice: retail and e-commerce basket analysis (cross-sell, shelf placement, bundles), "frequently bought together" widgets on small catalogues, co-occurring symptoms or diagnoses in clinical records, co-occurring alarms in IT or industrial logs, and web navigation patterns.
- Not for: large catalogues or long baskets (use FP-growth or ECLAT, which avoid candidate generation); personalised recommendations (use collaborative filtering or matrix factorisation); ordered events (use sequential-pattern mining such as PrefixSpan); or causal claims ("putting X near Y increases sales" needs an experiment).

## Setup & code
Standard library only. Runs instantly.

```python
from collections import Counter
from itertools import combinations
import random


def apriori(transactions, min_support):
    """Return {frozenset itemset: support} for every itemset with support >= min_support."""
    n = len(transactions)
    transactions = [frozenset(t) for t in transactions]
    counts = Counter(item for t in transactions for item in t)
    level = {frozenset([i]): c / n for i, c in counts.items() if c / n >= min_support}
    frequent, k, scanned = dict(level), 2, 0
    while level:
        prev = set(level)
        # join: union pairs of frequent (k-1)-itemsets that differ by one item
        cands = {a | b for a in prev for b in prev if len(a | b) == k}
        # prune: every (k-1)-subset must itself be frequent (the Apriori property)
        cands = {c for c in cands if all(frozenset(s) in prev for s in combinations(c, k - 1))}
        scanned += len(cands)
        counts = Counter(c for t in transactions for c in cands if c <= t)
        level = {c: cnt / n for c, cnt in counts.items() if cnt / n >= min_support}
        frequent.update(level)
        k += 1
    return frequent, scanned


def rules(frequent, min_confidence):
    """Rules X -> Y with confidence = supp(X u Y) / supp(X) and lift = confidence / supp(Y)."""
    out = []
    for itemset, supp in frequent.items():
        for r in range(1, len(itemset)):
            for lhs in map(frozenset, combinations(itemset, r)):
                rhs = itemset - lhs
                conf = supp / frequent[lhs]
                if conf >= min_confidence:
                    out.append((lhs, rhs, supp, conf, conf / frequent[rhs]))
    return sorted(out, key=lambda r: -r[4])


# synthetic shop: planted patterns plus random noise baskets
random.seed(0)
items = ["bread", "milk", "eggs", "butter", "beer", "chips", "diapers", "coffee", "sugar", "apples"]
baskets = []
for _ in range(2000):
    b = set(random.sample(items, random.randint(1, 3)))
    r = random.random()
    if r < 0.25: b |= {"bread", "butter"}
    elif r < 0.40: b |= {"beer", "chips"}
    elif r < 0.50: b |= {"diapers", "beer"}
    elif r < 0.65: b |= {"coffee", "sugar", "milk"}
    baskets.append(b)

freq, scanned = apriori(baskets, min_support=0.05)
n_items = len(items)
print(f"{len(baskets)} baskets, {n_items} items; frequent itemsets: {len(freq)}")
print(f"candidates counted after level 1: {scanned} (brute force over all itemsets of size >= 2: {2**n_items - n_items - 1})")
by_size = Counter(len(s) for s in freq)
print("frequent itemsets by size:", dict(sorted(by_size.items())))
print("\nlhs -> rhs                     support  confidence  lift")
for lhs, rhs, s, c, l in rules(freq, min_confidence=0.6)[:8]:
    print(f"{', '.join(sorted(lhs)) + ' -> ' + ', '.join(sorted(rhs)):<30} {s:>7.3f}  {c:>10.2f}  {l:>4.2f}")

# confidence alone misleads when the consequent is common anyway
bread = freq[frozenset(["bread"])]
r = [x for x in rules(freq, 0.0) if x[0] == frozenset(["apples"]) and x[1] == frozenset(["bread"])][0]
print(f"\napples -> bread: confidence {r[3]:.2f}, but P(bread) = {bread:.2f}, lift {r[4]:.2f}")
```

Output (Python 3.14):
```
2000 baskets, 10 items; frequent itemsets: 57
candidates counted after level 1: 139 (brute force over all itemsets of size >= 2: 1013)
frequent itemsets by size: {1: 10, 2: 41, 3: 6}

lhs -> rhs                     support  confidence  lift
milk, sugar -> coffee            0.160        0.89  2.85
coffee, sugar -> milk            0.160        0.90  2.73
coffee, milk -> sugar            0.160        0.87  2.73
apples, butter -> bread          0.053        0.79  2.19
apples, bread -> butter          0.053        0.75  1.96
chips, diapers -> beer           0.062        0.75  1.83
bread -> butter                  0.247        0.69  1.79
butter -> bread                  0.247        0.64  1.79

apples -> bread: confidence 0.37, but P(bread) = 0.36, lift 1.03
```

All 10 single items clear 5% support. Pruning then leaves 139 candidates to count across levels 2 and 3, where enumerating every itemset of two or more items would need 1013. The real saving grows exponentially with the catalogue size. The top rules by lift are the planted patterns: the coffee/sugar/milk triple (lift 2.7 to 2.9, confidence about 0.9) and bread/butter (lift 1.79). `chips, diapers → beer` (confidence 0.75, lift 1.83) was never planted as a three-item pattern. It appears because chips and diapers each come with beer through two separate patterns, so a rule can join two independent habits. `apples → bread` shows why confidence alone misleads: 37% of apple baskets contain bread, but so do 36% of all baskets, and the lift of 1.03 says apples tell you nothing.

## Choosing / trade-offs
- **Apriori vs FP-growth vs ECLAT.** Apriori is simple and scans the data once per level, and its candidate sets explode with low support or long baskets. FP-growth compresses the data into a prefix tree and mines it without generating candidates, which is usually much faster. ECLAT uses vertical tid-lists and set intersections, which is fast when those lists fit in memory. All three return the same itemsets.
- **Minimum support.** Too high and only the obvious best-sellers survive. Too low and the output and runtime explode. Start where the top few hundred itemsets survive, and use absolute counts on large data.
- **Ranking metric.** Lift favours rare items (a pair seen twice can have huge lift). Combine a minimum support with lift, or use leverage (`supp(X∪Y) − supp(X)·supp(Y)`), conviction, or a significance test.
- **Libraries.** `mlxtend.frequent_patterns` (apriori, fpgrowth, association_rules) works on a one-hot pandas DataFrame. Spark MLlib `FPGrowth` handles distributed data. R's `arules` is the most complete toolkit.

## Gotchas
- Thresholds decide the answer. Report the minimum support and confidence next to every rule list, and test several values.
- High confidence with lift ≈ 1 is a popular consequent, not an association. Always check lift (or leverage) before acting on a rule.
- Many rules are redundant: every subset split of the same frequent triple shows up. Keep only the closed or maximal itemsets, or deduplicate rules by itemset.
- Transactions are sets. Quantities, prices and order are discarded. Decide whether "bought 5 cans" and "bought 1 can" should look the same.
- Rare but valuable items (luxury goods, rare diseases) fall below the threshold. Mine them separately or use per-item support thresholds.
- Association is not causation, and a rule found by searching thousands of itemsets is a multiple-comparisons result. Validate on a holdout period before changing the shop.
- The naive join above compares every pair of (k−1)-itemsets, which is O(L²). Real implementations join only sets that share their first k−2 items (sorted), and count candidates with a hash tree or bitsets.

## Related
- [[association-rules-market-basket]] - library-level market basket analysis with mlxtend and how to read the results.
- [[matrix-factorization-from-scratch]] - personalised recommendations rather than global co-occurrence rules.
- [[recommender-systems]] - where rule mining sits among recommendation approaches.
- [[naive-bayes-from-scratch]] - another counting-based method built on conditional probabilities.

## References
- Agrawal and Srikant (1994), "Fast algorithms for mining association rules", VLDB: https://www.vldb.org/conf/1994/P487.PDF
- mlxtend user guide, apriori and association_rules: https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/
- Han, Pei and Yin (2000), "Mining frequent patterns without candidate generation" (FP-growth), SIGMOD: https://doi.org/10.1145/335191.335372
