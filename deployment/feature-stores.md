---
title: Feature stores (online/offline features, point-in-time correctness, Feast)
category: deployment
tags: [feature-store, feast, online-store, offline-store, point-in-time, training-serving-skew, mlops, real-time-ml]
use_cases:
  - "serve the same customer features to a fraud model in training and in real-time scoring"
  - "build a training set without leaking future feature values into the labels"
  - "look up user features in a few milliseconds at inference time"
  - "decide whether we need Feast or a managed feature store at all"
status: draft
last_verified: 2026-10-15
sources:
  - https://docs.feast.dev/
  - https://docs.feast.dev/getting-started/concepts/point-in-time-joins
  - https://github.com/feast-dev/feast
  - https://pypi.org/project/feast/
  - https://www.databricks.com/blog/tecton-joining-databricks-power-real-time-data-personalized-ai-agents
---

# Feature stores (online/offline features, point-in-time correctness, Feast)

## Summary
A feature store keeps ML features defined once and served two ways: an **offline store** with full history for building training sets with point-in-time-correct joins, and an **online store** with the latest value per entity for millisecond lookups at inference time. It exists to stop two classic production bugs: label leakage (training on feature values that were not yet known) and training/serving skew (the model sees differently computed features in production). Feast is the main open-source option; cloud platforms ship managed ones.

## Key concepts
- **Entity:** the key features hang off (customer_id, merchant_id, device_id). **Feature view:** a group of features for an entity from one source, with a schema and a TTL.
- **Event timestamp:** when a feature value became true in the world. Every feature row needs one; it drives correctness.
- **Point-in-time join:** for each training row (entity + prediction timestamp), take the latest feature value at or before that timestamp, and none older than the TTL. A plain "latest value" join leaks the future.
- **Offline store:** warehouse/lake tables (Parquet, BigQuery, Snowflake, Redshift, Spark, Postgres, DuckDB...). Large, slow, historical.
- **Online store:** key-value store with the latest values (Redis, DynamoDB, Bigtable, Cassandra, Postgres, SQLite for local dev...). Small, fast.
- **Materialisation:** batch job copying the newest offline values into the online store (Feast: `materialize` / `materialize_incremental`). **Push / stream ingestion:** write fresh values straight to the online store from a stream ([[streaming-ml-pipelines]]).
- **On-demand features:** computed at request time from request data plus stored features (e.g. distance between this transaction's location and the customer's home).
- **Registry:** the catalogue of definitions, so features are discoverable, versioned and reused across teams.
- **Training/serving skew:** offline and online values for the same entity and instant differ because two code paths compute them. Single definitions plus a parity check catch it.

## When to use / scenarios
- Real-time models that need per-entity history at request time: fraud and risk scoring, credit decisions ([[finance]]), recommendations and personalisation ([[recommender-systems]], [[ecommerce-retail]]), ad ranking, ETA and dispatch.
- Several teams/models reusing the same features (customer lifetime value, 7-day spend) and fighting over definitions.
- LLM apps that personalise prompts with user context (plan, recent orders, risk tier) fetched by key: an online store fits; the Feast registry also has vector-store integrations for RAG, but a dedicated [[vector-databases]] setup is the usual choice for document retrieval.
- Not needed when: models are batch-scored (a warehouse table and SQL with careful as-of joins is enough), there is one model and one team, or features come only from the request itself. Start with disciplined SQL and add a feature store when reuse or online serving forces it.

## Setup & code
Point-in-time correctness, the online store and a skew check, shown with pandas (`merge_asof` is the core of a point-in-time join):
```python
import pandas as pd

# Offline store: feature values over time (event_timestamp = when the value became true).
features = pd.DataFrame({
    "customer_id": [1, 1, 1, 2, 2],
    "event_timestamp": pd.to_datetime(["2026-01-01", "2026-01-10", "2026-01-20", "2026-01-05", "2026-01-25"]),
    "txn_count_7d": [2, 9, 30, 1, 4],
})
# Training labels ("entity dataframe"): what we want to predict, and WHEN the prediction would have been made.
labels = pd.DataFrame({
    "customer_id": [1, 1, 2, 2],
    "event_timestamp": pd.to_datetime(["2026-01-12", "2026-01-21", "2026-01-06", "2026-03-01"]),
    "is_fraud": [0, 1, 0, 0],
})

# WRONG: join the latest value per customer -> leaks the future into training rows.
latest = features.sort_values("event_timestamp").groupby("customer_id").tail(1)
naive = labels.merge(latest[["customer_id", "txn_count_7d"]], on="customer_id")

# RIGHT: point-in-time join = for each label row, the last feature value at or before its timestamp,
# ignoring values older than the feature's TTL (here 30 days).
pit = pd.merge_asof(labels.sort_values("event_timestamp"), features.sort_values("event_timestamp"),
                    on="event_timestamp", by="customer_id", direction="backward",
                    tolerance=pd.Timedelta(days=30))
out = pit.merge(naive[["customer_id", "event_timestamp", "txn_count_7d"]],
                on=["customer_id", "event_timestamp"], suffixes=("_pit", "_naive"))
print(out.sort_values(["customer_id", "event_timestamp"]).to_string(index=False))

# Online store: only the latest value per entity, keyed for low-latency lookup at serving time.
online = {row.customer_id: {"txn_count_7d": row.txn_count_7d, "ts": row.event_timestamp}
          for row in latest.itertuples()}
now = pd.Timestamp("2026-01-26")
def get_online(customer_id, ttl=pd.Timedelta(days=30)):
    r = online.get(customer_id)
    return r["txn_count_7d"] if r and now - r["ts"] <= ttl else None   # stale -> None, model must handle it
print("online lookup at serving time:", {c: get_online(c) for c in (1, 2, 3)})

# Training/serving skew check: the same feature definition must give the same value offline and online.
offline_now = pd.merge_asof(pd.DataFrame({"customer_id": [1, 2], "event_timestamp": [now, now]}),
                            features.sort_values("event_timestamp"), on="event_timestamp", by="customer_id")
skew = {int(r.customer_id): (int(r.txn_count_7d), get_online(r.customer_id)) for r in offline_now.itertuples()}
print("offline vs online at the same instant:", skew, "OK" if all(a == b for a, b in skew.values()) else "SKEW")
```
The naive join gives the 2026-01-12 row a value of 30 that only appeared on 2026-01-20: a model trained on it looks great offline and fails in production. The 2026-03-01 row gets no value because the last one is older than the 30-day TTL.

Feast (feast 0.67.0 on PyPI as of 2026-10-15). `feast init my_repo` scaffolds a repo with `feature_store.yaml` (provider, registry, online and offline store config); definitions are plain Python, and `feast apply` registers them:
<!-- skip-check: needs feast installed and a feature repo with data files -->
```python
# my_repo/features.py
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Int64

customer = Entity(name="customer", join_keys=["customer_id"])
customer_stats_source = FileSource(path="data/customer_stats.parquet", timestamp_field="event_timestamp")
customer_stats = FeatureView(
    name="customer_stats",
    entities=[customer],
    ttl=timedelta(days=30),                       # bounds the point-in-time lookback
    schema=[Field(name="txn_count_7d", dtype=Int64)],
    source=customer_stats_source,
)
```
<!-- skip-check: needs feast installed and an applied feature repo -->
```python
from datetime import datetime, timezone
import pandas as pd
from feast import FeatureStore

store = FeatureStore(repo_path="my_repo")
labels = pd.DataFrame({"customer_id": [1, 2],
                       "event_timestamp": pd.to_datetime(["2026-01-12", "2026-01-06"], utc=True)})
train_df = store.get_historical_features(              # point-in-time join in the offline store
    entity_df=labels, features=["customer_stats:txn_count_7d"]).to_df()

store.materialize_incremental(end_date=datetime.now(timezone.utc))   # offline -> online
online = store.get_online_features(                     # low-latency lookup at inference time
    features=["customer_stats:txn_count_7d"], entity_rows=[{"customer_id": 1}]).to_dict()
```
Streaming freshness: define a `PushSource` and call `store.push(...)` from your stream processor, or use `StreamFeatureView` with a `KafkaSource`/`KinesisSource`.

## Choosing / trade-offs
| Option | Fits | Watch out for |
|---|---|---|
| Plain SQL + cache (no feature store) | Batch scoring, one team, few online features | Hand-written as-of joins; skew between SQL and serving code |
| Feast (open source, Apache 2.0) | You want control, already run a warehouse and Redis/DynamoDB; pluggable stores | You operate materialisation jobs, the registry and the online store yourself; streaming transformations are thinner than commercial platforms |
| Platform feature stores (Databricks, Vertex AI, SageMaker, Hopsworks) | You are already on that platform ([[managed-cloud-ai-platforms]]) | Lock-in; check online latency and cost per read for your QPS |
| Commercial real-time feature platforms | Heavy streaming aggregations (sliding windows over events) with low freshness lag | Cost; Tecton was acquired by Databricks in 2025, so check its standalone roadmap |

- **Freshness vs cost:** hourly batch materialisation is cheap; second-level freshness needs stream ingestion ([[streaming-ml-pipelines]]).
- **Online store choice:** Redis for lowest latency, DynamoDB/Bigtable for managed scale, Postgres when QPS is modest and you want one fewer system.
- **Compute features where:** in the feature store's transformation engine (one definition, less flexible) vs upstream in dbt/Spark/Flink (flexible, but then the store only stores).

## Gotchas
- Using ingestion or processing time instead of event time as the timestamp silently reintroduces leakage.
- TTL too short gives NULLs in training that never occur online (or vice versa); check null rates offline vs online ([[missing-data-and-imputation]]).
- Late-arriving data: a backfilled value with an old event timestamp changes historical training sets after the fact; snapshot training sets you ship models on.
- Materialisation lag means the online value can be older than the offline "as-of now" value; that gap is skew too. Monitor online freshness per feature view.
- Entity key mismatches (int vs string IDs, different casing) produce silent misses that look like "new customer".
- Point-in-time joins on large entity dataframes are expensive; sample label rows or push the join into the warehouse engine.
- A feature store does not fix bad features: leakage via a feature that encodes the label (refund issued -> fraud) still needs review ([[data-leakage-and-validation-splits]]).
- Online stores often hold PII; apply access control, retention and deletion requests there too ([[ai-security-privacy-compliance]]).

## Related
- [[feature-engineering]] - designing the features themselves.
- [[data-leakage-and-validation-splits]] - leakage that point-in-time joins prevent, and leakage they do not.
- [[streaming-ml-pipelines]] - computing fresh features from event streams.
- [[mlops-lifecycle]], [[experiment-tracking]] - versioned training sets and models.
- [[online-learning-and-concept-drift]] - monitoring feature distributions over time.
- [[ab-testing-and-canary-for-models]] - rolling out a model that uses new features.
- [[recommender-systems]], [[anomaly-detection]] - typical consumers of online features.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/data/batch-and-stream-processing.md - batch vs stream pipelines behind the stores.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/data/data-warehousing-and-olap.md - the offline store side.

## References
- Feast documentation: https://docs.feast.dev/
- Feast point-in-time joins: https://docs.feast.dev/getting-started/concepts/point-in-time-joins
- Feast source (API used above: `FeatureStore.get_historical_features`, `materialize_incremental`, `get_online_features`, `push`): https://github.com/feast-dev/feast
- Feast on PyPI (version): https://pypi.org/project/feast/
- Databricks announcement on Tecton: https://www.databricks.com/blog/tecton-joining-databricks-power-real-time-data-personalized-ai-agents
