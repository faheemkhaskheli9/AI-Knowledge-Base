---
title: Polars and DuckDB for ML feature preparation
category: ml
tags: [polars, duckdb, pandas, dataframes, sql, parquet, arrow, feature-engineering, lazy-evaluation, out-of-core, data-preparation]
use_cases:
  - "speed up a slow pandas feature pipeline on tens of millions of rows"
  - "build training features from Parquet files that do not fit in memory on a laptop"
  - "compute per-customer aggregate features as of a cutoff date without leakage"
  - "decide between pandas, Polars and DuckDB for data preparation in an ML project"
status: draft
last_verified: 2026-10-15
sources:
  - https://docs.pola.rs/user-guide/lazy/execution
  - https://docs.pola.rs/user-guide/concepts/streaming
  - https://docs.pola.rs/releases/upgrade/2/
  - https://duckdb.org/docs/current/guides/python/sql_on_pandas
  - https://duckdb.org/docs/current/guides/python/execute_sql
  - https://pandas.pydata.org/docs/user_guide/pyarrow.html
---

# Polars and DuckDB for ML feature preparation

## Summary
Polars (a Rust DataFrame library with a lazy query optimiser) and DuckDB (an in-process analytical SQL database) are the two common replacements for pandas when feature preparation gets slow or data outgrows memory. Both are columnar, multi-threaded, read Parquet/CSV directly, push filters and column selection down into the file scan, and can process larger-than-memory data. Both speak Apache Arrow, so data moves between Polars, DuckDB, pandas and NumPy cheaply. Model training still happens in scikit-learn, XGBoost or PyTorch; these tools build the feature table that feeds them ([[feature-engineering]]).

## Key concepts
- **Columnar + vectorised.** Operations run over whole columns in compiled code across all cores, instead of pandas' mostly single-threaded execution.
- **Lazy evaluation (Polars).** `pl.scan_parquet(...)` builds a query plan; `.collect()` optimises it (predicate and projection pushdown, common-subexpression elimination) and runs it. `LazyFrame.explain()` shows the optimised plan.
- **Expressions (Polars).** `pl.col("amount").filter(...).sum()` inside `group_by().agg()` or `with_columns()` replaces pandas' `apply`/lambda patterns and runs in parallel.
- **Streaming.** Polars' streaming engine (`collect(engine="streaming")`) processes data in batches for larger-than-memory inputs; the Polars 2 upgrade guide says lazy queries now default to it. DuckDB spills large joins, sorts and aggregations to disk automatically.
- **In-process SQL (DuckDB).** `duckdb.sql("SELECT ... FROM 'data/*.parquet'")` needs no server. It can query in-memory pandas, Polars and Arrow objects by their Python variable name, and results convert with `.df()` (pandas), `.pl()` (Polars), `.arrow()` or `.fetchnumpy()`.
- **Arrow interchange.** Both use Arrow memory, so DuckDB-to-Polars and Polars-to-Arrow conversions are often zero-copy. pandas can also use Arrow-backed dtypes.
- **Point-in-time correctness.** Aggregate only events before each label's cutoff time; both tools express this as a filter or an as-of join (`join_asof` in Polars, `ASOF JOIN` in DuckDB) ([[data-leakage-and-validation-splits]]).

## When to use / scenarios
- Churn, fraud or propensity models built from event logs: per-entity counts, sums, recency and window features over 10M-1B rows on one machine.
- Recommender and ranking pipelines that join large interaction tables with item metadata ([[recommender-systems]], [[learning-to-rank]]).
- Time-series feature tables (lags, rolling stats per series) for global forecasting models ([[time-series-forecasting]]).
- Data cleaning and EDA on files that crash pandas with out-of-memory errors ([[exploratory-data-analysis]]).
- Prefer **DuckDB** when the team thinks in SQL, the logic is joins/window functions, or you query a folder of Parquet files ad hoc (also used by LLM data agents, see [[data-analytics]]).
- Prefer **Polars** when you want a DataFrame API with type-checked expressions in Python code and fast single-machine pipelines.
- NOT needed for small data (under a few million rows) where pandas is fast enough and every library already accepts it.
- NOT a replacement for a warehouse or Spark when data is many terabytes, needs concurrent writers, or must be shared across a team; use the warehouse and pull an extract.

## Setup & code
```bash
pip install polars duckdb pyarrow pandas
```
One million synthetic transactions written to Parquet; the same per-customer features as of a cutoff date are computed with Polars (lazy), DuckDB (SQL) and pandas, checked to be identical, then joined to labels and turned into a NumPy training matrix. Tested with polars 2.0.0 and duckdb 1.5.6; timings vary by machine (on a shared 4-core container Polars and DuckDB were roughly 3-5x faster than pandas, and the gap grows with data size and with string and join-heavy work).
```python
import time
from datetime import date, timedelta
import numpy as np
import pandas as pd
import polars as pl
import duckdb

rng = np.random.default_rng(0)
n = 1_000_000
start = np.datetime64("2026-01-01T00:00:00", "us")
tx = pl.DataFrame({
    "customer_id": rng.integers(0, 50_000, n),
    "ts": start + rng.integers(0, 180 * 24 * 3600, n).astype("timedelta64[s]").astype("timedelta64[us]"),
    "amount": rng.gamma(2.0, 30.0, n).round(2),
    "channel": rng.choice(["web", "app", "store"], n),
})
tx.write_parquet("tx.parquet")
labels = pl.DataFrame({"customer_id": np.arange(0, 50_000, 5),
                       "cutoff": [date(2026, 5, 1)] * 10_000})
cutoff = date(2026, 5, 1)

# Polars lazy: scan Parquet, filter before the cutoff (no leakage), aggregate per customer.
t0 = time.perf_counter()
feats_pl = (
    pl.scan_parquet("tx.parquet")
    .filter(pl.col("ts") < pl.lit(cutoff).cast(pl.Datetime("us")))
    .group_by("customer_id")
    .agg(
        n_tx=pl.len(),
        spend=pl.col("amount").sum(),
        avg_amount=pl.col("amount").mean(),
        app_share=(pl.col("channel") == "app").mean(),
        spend_30d=pl.col("amount").filter(
            pl.col("ts") >= pl.lit(cutoff - timedelta(days=30)).cast(pl.Datetime("us"))).sum(),
        last_ts=pl.col("ts").max(),
    )
    .with_columns(days_since_last=(pl.lit(cutoff).cast(pl.Datetime("us")) - pl.col("last_ts")).dt.total_days())
    .drop("last_ts")
    .sort("customer_id")
    .collect()
)
t_pl = time.perf_counter() - t0

# DuckDB: the same features in SQL, reading the same Parquet file.
t0 = time.perf_counter()
feats_db = duckdb.sql("""
    SELECT customer_id,
           count(*)                                         AS n_tx,
           sum(amount)                                      AS spend,
           avg(amount)                                      AS avg_amount,
           avg((channel = 'app')::DOUBLE)                   AS app_share,
           coalesce(sum(amount) FILTER (WHERE ts >= DATE '2026-05-01' - INTERVAL 30 DAY), 0) AS spend_30d,
           floor(epoch(TIMESTAMP '2026-05-01' - max(ts)) / 86400)::BIGINT AS days_since_last
    FROM 'tx.parquet'
    WHERE ts < DATE '2026-05-01'
    GROUP BY customer_id ORDER BY customer_id
""").pl()
t_db = time.perf_counter() - t0

# pandas: the eager baseline.
t0 = time.perf_counter()
df = pd.read_parquet("tx.parquet")
df = df[df["ts"] < pd.Timestamp(cutoff)]
recent = df["ts"] >= pd.Timestamp(cutoff) - pd.Timedelta(days=30)
feats_pd = (df.assign(is_app=(df["channel"] == "app").astype(float), recent_amt=df["amount"].where(recent, 0.0))
              .groupby("customer_id")
              .agg(n_tx=("amount", "size"), spend=("amount", "sum"), avg_amount=("amount", "mean"),
                   app_share=("is_app", "mean"), spend_30d=("recent_amt", "sum"), last_ts=("ts", "max"))
              .reset_index())
t_pd = time.perf_counter() - t0
print(f"rows {n:,}: polars {t_pl:.2f}s  duckdb {t_db:.2f}s  pandas {t_pd:.2f}s")

# All three agree.
assert feats_pl.height == feats_db.height == len(feats_pd)
for col in ["n_tx", "spend", "avg_amount", "app_share", "spend_30d"]:
    assert np.allclose(feats_pl[col].to_numpy(), feats_db[col].to_numpy()), col
    assert np.allclose(feats_pl[col].to_numpy(), feats_pd[col].to_numpy()), col
assert (feats_pl["days_since_last"].to_numpy() == feats_db["days_since_last"].to_numpy()).all()

# DuckDB queries in-memory Polars/pandas frames by variable name (zero-copy via Arrow).
train = duckdb.sql("SELECT l.customer_id, f.* EXCLUDE (customer_id) FROM labels l "
                   "LEFT JOIN feats_pl f USING (customer_id) ORDER BY l.customer_id").pl()
X = train.drop("customer_id").fill_null(0).to_numpy()     # hand to scikit-learn / XGBoost
print("training matrix:", X.shape, X.dtype)
```
Notes on the example: the cutoff filter is applied before aggregation, so no post-cutoff event leaks into features; `days_since_last` is computed the same way in both engines on purpose (see Gotchas); and the final join is done by DuckDB directly on the Polars frames `labels` and `feats_pl`.

## Choosing / trade-offs
| Need | pandas | Polars | DuckDB |
|---|---|---|---|
| Interface | DataFrame, huge ecosystem | DataFrame expressions | SQL (plus a relational Python API) |
| Speed on large data | slowest, mostly single-threaded | fast, multi-threaded | fast, multi-threaded |
| Larger than memory | no (chunk by hand) | streaming engine | yes, spills to disk |
| Library compatibility | everything accepts it | convert with `.to_pandas()` / `.to_numpy()` | convert with `.df()` / `.pl()` |
| Persistence | files | files | files, or a `.duckdb` database file |

- **Mix them.** Use DuckDB for heavy SQL joins over Parquet, Polars for expression-heavy transforms, and convert to pandas or NumPy only at the model boundary.
- **Index-free model.** Polars and DuckDB have no row index; code that relies on pandas index alignment must be rewritten with explicit joins.
- **Learning cost.** Polars' API differs from pandas (no `inplace`, no index, expression contexts); DuckDB needs SQL fluency. A mixed team may standardise on SQL.
- **Serving.** For online features, the same logic must run at request time; a feature store or a shared SQL definition avoids training/serving skew ([[feature-stores]]).

## Gotchas
- Row order is not guaranteed after `group_by` or joins in Polars (especially with the streaming engine) or in SQL without `ORDER BY`. Sort explicitly before comparing or exporting.
- Date arithmetic semantics differ: DuckDB's `date_diff('day', a, b)` counts calendar-day boundaries crossed, while Polars' `(b - a).dt.total_days()` truncates the elapsed duration. The same feature name can silently mean different numbers.
- NumPy `datetime64[s]` is rejected by Polars; cast to `us`, `ms` or `ns` first.
- Nulls vs NaN: Polars and DuckDB use real nulls; NaN is a float value. Aggregations treat them differently, and `.to_numpy()` turns nulls into NaN (or object dtype) depending on type. Fill nulls deliberately before training.
- Polars has had breaking releases (the 2.0 upgrade guide lists many). Pin the version in `requirements.txt` and read the upgrade guide before bumping.
- `pl.DataFrame.apply`/`map_elements` with a Python lambda runs row by row and throws away the speed; use expressions.
- DuckDB's default in-memory database disappears at process exit; pass a file path to `duckdb.connect()` to persist tables.
- A DuckDB database file allows one writer process at a time; it is not a multi-user server.
- Don't benchmark on cached, tiny data: measure on realistic sizes and count conversion time to pandas/NumPy too.

## Related
- [[feature-engineering]] - what features to build; this file is about the tooling to build them fast.
- [[data-leakage-and-validation-splits]] - point-in-time cutoffs that the example applies.
- [[exploratory-data-analysis]] - first look at data before building features.
- [[time-series-forecasting]] - lag and rolling features at scale.
- [[classic-ml-scikit-learn]] - models that consume the resulting NumPy matrix.
- [[gradient-boosting-tabular]] - the usual model on aggregated tabular features.
- [[data-analytics]] - DuckDB as the engine behind LLM data-analysis agents.
- [[text-to-sql]] - generating DuckDB SQL from natural language.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/data/data-warehousing-and-olap.md
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/data/batch-and-stream-processing.md

## References
- Polars user guide, lazy API and query execution: https://docs.pola.rs/user-guide/lazy/execution
- Polars user guide, streaming: https://docs.pola.rs/user-guide/concepts/streaming
- Polars 2 upgrade guide: https://docs.pola.rs/releases/upgrade/2/
- DuckDB, SQL on pandas (replacement scans): https://duckdb.org/docs/current/guides/python/sql_on_pandas
- DuckDB, executing SQL in Python: https://duckdb.org/docs/current/guides/python/execute_sql
- pandas, PyArrow functionality: https://pandas.pydata.org/docs/user_guide/pyarrow.html
