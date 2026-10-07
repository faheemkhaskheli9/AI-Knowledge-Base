---
title: Categorical feature encoding (one-hot, ordinal, target, hashing, embeddings)
category: ml
tags: [categorical-features, one-hot, ordinal-encoding, target-encoding, hashing-trick, entity-embeddings, high-cardinality, scikit-learn, catboost, feature-engineering]
use_cases:
  - "turn text categories like city, product or merchant ID into model inputs"
  - "handle a categorical column with thousands of levels without blowing up memory"
  - "use target encoding without leaking the label into the features"
  - "deal with categories at prediction time that never appeared in training"
status: stable
last_verified: 2026-10-04
sources:
  - https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features
  - https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html
  - https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder_cross_val.html
  - https://catboost.ai/docs/en/concepts/algorithm-main-stages_cat-to-numberic
  - https://arxiv.org/abs/1604.06737
---

# Categorical feature encoding (one-hot, ordinal, target, hashing, embeddings)

## Summary
Most models need numbers, but much business data is categorical: country, product, merchant, device, ZIP code. The right encoding depends on how many levels a column has, whether the levels have an order, and which model consumes them. One-hot suits a few levels. Ordinal suits ordered levels and tree models. Cross-fitted target encoding or hashing suits high cardinality. Learned embeddings suit neural networks. A wrong choice either explodes memory or leaks the label into the features.

## Key concepts
- **Cardinality.** The number of distinct levels. Low (under ~15) favours one-hot. High (hundreds to millions: user ID, ZIP, SKU) needs a compact encoding.
- **Nominal vs ordinal.** Nominal levels have no order (colour, city). Ordinal levels do (S < M < L, rating bands). Ordinal codes `0, 1, 2` are only meaningful for ordered levels, and for tree models, which can split any code range.
- **One-hot.** One binary column per level. Safe and interpretable for linear models and neural nets. Width grows with cardinality. `drop="first"` matters only for unregularised linear models (collinearity).
- **Target (mean) encoding.** Replace a level by a smoothed mean of the target for that level, shrunk toward the global mean when the level is rare. Compact and strong, but it uses the label, so it **must be cross-fitted**: each row gets an encoding computed from the other folds.
- **Hashing trick.** Hash each level into one of k buckets (`FeatureHasher`). Fixed width, no fitted vocabulary, handles unseen levels. Collisions merge unrelated levels.
- **Rare and unseen levels.** Group levels below a frequency threshold into an "infrequent" bucket. Decide in advance what an unseen level maps to at serving time (infrequent bucket, global mean, zero vector).
- **Entity embeddings.** A neural network learns a dense vector per level (`nn.Embedding`), trained end-to-end with the task. The vectors can be reused as features for other models.
- **Native categorical support.** LightGBM, CatBoost, XGBoost (`enable_categorical`) and sklearn's `HistGradientBoosting*` (`categorical_features="from_dtype"`) split on categories directly. CatBoost uses ordered target statistics, which is target encoding with built-in leakage protection.

## When to use / scenarios
- Retail and e-commerce: SKU, brand, store and category IDs in demand or conversion models.
- Fraud and risk: merchant ID, device type, IP country, BIN in transaction scoring.
- Ads and recommenders: user, item, campaign and publisher IDs with millions of levels (hashing or embeddings).
- Insurance and credit: region, occupation, vehicle model in pricing models, where target encoding is common.
- Not needed when you use a gradient-boosting library with native categorical handling: pass the column as a category dtype ([[gradient-boosting-tabular]]).
- Not this file for free text (use [[embeddings]] or TF-IDF from [[nlp-classic-tasks]]).

## Setup & code
```bash
pip install "scikit-learn>=1.9" pandas numpy
```
```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder

rng = np.random.default_rng(0)
n = 5000
city = rng.integers(0, 500, n).astype(str)              # high cardinality
size = rng.choice(["S", "M", "L"], n)                    # ordered
city_effect = rng.normal(size=500)
logit = city_effect[city.astype(int)] + pd.Series(size).map({"S": -1, "M": 0, "L": 1}).to_numpy()
y = (logit + rng.normal(size=n) > 0).astype(int)
X = pd.DataFrame({"city": city, "size": size})

def score(city_enc):
    pre = ColumnTransformer([
        ("city", city_enc, ["city"]),
        ("size", OrdinalEncoder(categories=[["S", "M", "L"]]), ["size"]),
    ])
    pipe = make_pipeline(pre, LogisticRegression(max_iter=1000))
    return cross_val_score(pipe, X, y, cv=5, scoring="roc_auc").mean()

folds = StratifiedKFold(5, shuffle=True, random_state=0)   # sklearn>=1.9: pass a splitter, not random_state
print("one-hot:", round(score(OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5)), 3))
print("target: ", round(score(TargetEncoder(target_type="binary", cv=folds)), 3))

# Leak demo on a pure-noise ID column (one row per level):
noise = pd.DataFrame({"id": np.arange(n).astype(str)})
te = TargetEncoder(target_type="binary", cv=folds)
leaky = te.fit(noise, y).transform(noise)   # encodes each row with its own label
safe = te.fit_transform(noise, y)           # cross-fitted
print(np.corrcoef(leaky[:, 0], y)[0, 1], np.corrcoef(safe[:, 0], y)[0, 1])
```
Output with scikit-learn 1.9: one-hot AUC 0.831, target-encoded AUC 0.837, leaky correlation 1.0 vs cross-fitted 0.0. The leaky version would look perfect in training and be useless in production.

In scikit-learn 1.9, `TargetEncoder`'s `shuffle` and `random_state` are deprecated. Pass a CV splitter via `cv=` instead.

## Choosing / trade-offs
| Situation | Encoding |
|---|---|
| Few levels, linear model or neural net | One-hot |
| Ordered levels | `OrdinalEncoder` with explicit `categories` |
| Any cardinality, tree model | Native categorical support, or `OrdinalEncoder` |
| High cardinality, linear model | Cross-fitted `TargetEncoder`, or one-hot with `min_frequency`/`max_categories` |
| Huge or open-ended vocabulary, streaming | `FeatureHasher` |
| Neural network on tabular data | `nn.Embedding` per column ([[deep-learning-for-tabular-data]]) |

- **Target encoding vs one-hot.** Target encoding is one column per feature and often scores better at high cardinality. It hides per-level effects from interpretation and needs cross-fitting. One-hot is transparent and leak-free.
- **Hashing vs vocabulary.** Hashing needs no fitted state and never meets an unseen level, which suits online learning ([[online-learning-and-concept-drift]]). A vocabulary is collision-free and invertible, so you can read which level a weight belongs to.
- **Embedding size.** A common heuristic is `min(50, (levels + 1) // 2)` dimensions. Tune it like any hyperparameter.

## Gotchas
- **Target encoding fit on the training rows it then encodes** is label leakage. `TargetEncoder.fit_transform` cross-fits. `fit(X, y).transform(X)` does not. Use `fit_transform` on training data and `transform` only on new data ([[data-leakage-and-validation-splits]]).
- **Fitting encoders outside the pipeline.** Encoding the full dataset before the train/test split leaks vocabulary and target statistics. Put encoders in a `Pipeline`/`ColumnTransformer`.
- **`handle_unknown="error"` is the default** in `OneHotEncoder`. The first unseen level in production crashes the service. Set `handle_unknown="ignore"` or `"infrequent_if_exist"`.
- **Ordinal codes in a linear model** impose a fake order and equal spacing (city 7 is "between" city 6 and city 8).
- **Integer IDs stored as numbers** (ZIP 94103, store 17) get treated as continuous. Cast to string or category first.
- **Drift in level frequencies.** New products and merchants keep appearing. Monitor the share of unseen/infrequent levels at serving time ([[mlops-lifecycle]], [[online-learning-and-concept-drift]]).
- **Group splits.** If the same customer appears in train and test, target encoding on customer ID will look great offline and fail on new customers. Split by group.

## Related
- [[feature-engineering]] - the wider preprocessing pipeline this encoding step sits in.
- [[data-leakage-and-validation-splits]] - why encoders must be fit per fold.
- [[gradient-boosting-tabular]] - native categorical handling in LightGBM, CatBoost and XGBoost.
- [[deep-learning-for-tabular-data]] - entity embeddings inside tabular neural nets.
- [[classic-ml-scikit-learn]] - `Pipeline` and `ColumnTransformer` basics.
- [[recommender-systems]] - ID embeddings at very high cardinality.

## References
- scikit-learn, encoding categorical features: https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features
- `TargetEncoder` API and cross-fitting example: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html
- CatBoost ordered target statistics: https://catboost.ai/docs/en/concepts/algorithm-main-stages_cat-to-numberic
- Guo & Berkhahn, "Entity Embeddings of Categorical Variables" (2016): https://arxiv.org/abs/1604.06737
- Micci-Barreca, "A preprocessing scheme for high-cardinality categorical attributes" (SIGKDD Explorations, 2001): https://doi.org/10.1145/507533.507538
