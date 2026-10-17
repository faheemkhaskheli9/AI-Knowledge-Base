---
title: Streaming ML pipelines (Kafka, stream processing, real-time features and inference)
category: deployment
tags: [streaming, kafka, flink, spark-structured-streaming, quix-streams, real-time-features, event-time, watermarks, online-inference]
use_cases:
  - "score card transactions for fraud within a second using features from the last 5 minutes"
  - "compute real-time features from a Kafka topic and push them to an online feature store"
  - "run an ML model or LLM classifier on every event in a stream"
  - "decide between Flink, Spark Structured Streaming, Kafka Streams and a Python stream library"
status: draft
last_verified: 2026-10-15
sources:
  - https://kafka.apache.org/documentation/
  - https://nightlies.apache.org/flink/flink-docs-stable/docs/concepts/time/
  - https://spark.apache.org/docs/latest/streaming/
  - https://github.com/quixio/quix-streams
  - https://pypi.org/project/quixstreams/
  - https://docs.feast.dev/
---

# Streaming ML pipelines (Kafka, stream processing, real-time features and inference)

## Summary
A streaming ML pipeline consumes events from a log such as Apache Kafka, computes features over time windows per key (card, user, device), and either scores each event with a model right away or writes fresh features to an online store for a separate inference service. It is how fraud detection, real-time personalisation and monitoring react within seconds instead of after a nightly batch. The hard parts are not the model: event time vs arrival time, late and duplicate events, state, and keeping streaming features identical to the ones used in training.

## Key concepts
- **Log / topic:** Kafka (or Kinesis, Pub/Sub, Redpanda, Pulsar) stores ordered, replayable events split into partitions. Ordering holds only within a partition, so key messages by the entity whose features you compute.
- **Consumer group and offsets:** parallelism = number of partitions; committed offsets decide what is reprocessed after a crash.
- **Event time vs processing time:** windows must use the time the event happened, or replays and backlogs produce different features than live traffic.
- **Watermark and allowed lateness:** the processor's estimate of "event time has reached T"; events older than the watermark are late and are dropped, sent to a side output, or used to update an already emitted result.
- **Windows:** tumbling (fixed, non-overlapping), hopping/sliding (overlapping, e.g. "last 5 minutes" at every event), session (gap-based).
- **Keyed state and checkpoints:** per-key aggregates live in the processor's state store (RocksDB in Flink and Kafka Streams) and are checkpointed so a restart resumes without recomputing history.
- **Delivery semantics:** at-least-once is the default; make sinks idempotent (dedupe by event ID, upsert by key) or use the engine's exactly-once mode (Kafka transactions for Kafka-to-Kafka, Flink checkpoints with transactional sinks).
- **Two inference patterns:** (1) *embedded*: the stream job loads the model and scores each event; (2) *feature pipeline + service*: the stream job only updates an online store ([[feature-stores]]) and a request/response service scores when asked. LLM calls in a stream are a third variant: async, batched, rate-limited calls to an API or self-hosted server.

## When to use / scenarios
- Fraud, AML and risk scoring on payments ([[finance]]), account takeover and anomaly alerts on logs ([[security-defensive]], [[anomaly-detection]]).
- Real-time recommendation and ranking features: items viewed in the last 10 minutes, session intent ([[recommender-systems]], [[ecommerce-retail]]).
- IoT and manufacturing: sensor windows feeding predictive-maintenance models ([[manufacturing-iot]]).
- LLM enrichment of event streams: classify or extract fields from support tickets, reviews or chat messages as they arrive, writing results to a topic ([[structured-output]]).
- Monitoring models themselves: streaming prediction logs into drift detectors ([[online-learning-and-concept-drift]]).
- Not needed when hourly or daily freshness is fine: batch features plus a request-time model is simpler, cheaper and easier to debug. Many "real-time" requirements are met by request-time features from the request payload alone.

## Setup & code
Stdlib/numpy simulation of the core mechanics: an out-of-order topic with a duplicate, an idempotent consumer, event-time sliding-window features per key, a watermark with allowed lateness, and per-event scoring by a stand-in model.
```python
from collections import defaultdict
import numpy as np

rng = np.random.default_rng(1)

# A "topic": card transactions with event time (seconds), arriving slightly out of order, with one redelivery.
events = []
for i in range(400):
    t = i * 3 + int(rng.integers(0, 5))
    events.append({"id": f"e{i}", "card": f"c{int(rng.integers(0, 5))}", "amount": float(rng.gamma(2, 20)), "ts": t})
events.append({"id": "e900", "card": "c3", "amount": 900.0, "ts": 1190})          # burst on one card
events += [{"id": f"e90{k}", "card": "c3", "amount": 850.0, "ts": 1192 + k} for k in range(1, 4)]
events.append(events[10])                                                             # at-least-once: duplicate
arrival = sorted(events, key=lambda e: e["ts"] + rng.integers(0, 40))                # network reordering

WINDOW_S, LATENESS_S = 300, 30          # 5-minute sliding window, 30 s allowed lateness
state = defaultdict(list)                # per-card (ts, amount) still needed (keyed state)
seen, watermark, late, preds = set(), 0, 0, []

def features(card, now):
    # sliding event-time window (now - 5 min, now]; out-of-order events are fine
    amts = [a for t, a in state[card] if now - WINDOW_S < t <= now]
    return {"n_5m": len(amts), "sum_5m": sum(amts), "max_5m": max(amts, default=0.0)}

def score(f, amount):                        # stand-in model: logistic score on window features
    z = -6 + 0.0025 * f["sum_5m"] + 0.002 * amount
    return float(1 / (1 + np.exp(-z)))

for e in arrival:
    if e["id"] in seen:                      # idempotent consumer: drop redeliveries
        continue
    seen.add(e["id"])
    watermark = max(watermark, e["ts"] - LATENESS_S)
    if e["ts"] < watermark:                  # too late for the window: count it, send to a side output
        late += 1
        continue
    state[e["card"]].append((e["ts"], e["amount"]))
    state[e["card"]] = [x for x in state[e["card"]] if x[0] > watermark - WINDOW_S]   # evict old state
    f = features(e["card"], e["ts"])         # features as of this event's time (no future data)
    p = score(f, e["amount"])
    preds.append((e["id"], e["card"], round(p, 3)))   # in production: produce to a "scores" topic

alerts = [p for p in preds if p[2] > 0.5]
print(f"processed {len(preds)} events, dropped {len(events) - len(seen)} duplicate(s), {late} too late")
print("alerts (p > 0.5):", alerts)
now = watermark + LATENESS_S
print("online features now:", {c: {k: round(v) for k, v in features(c, now).items()} for c in sorted(state)})
```

The same window feature with Quix Streams, a pure-Python Kafka stream library (API as in quixstreams 3.28; assumes a local Kafka broker and producers that key messages by card):
<!-- skip-check: needs quixstreams installed and a running Kafka broker -->
```python
from datetime import timedelta
from quixstreams import Application
import quixstreams.dataframe.windows.aggregations as agg

app = Application(broker_address="localhost:9092", consumer_group="card-features",
                  auto_offset_reset="earliest")
txns = app.topic("transactions", value_deserializer="json")       # message key = card id
out = app.topic("card-features-5m", value_serializer="json")

sdf = app.dataframe(txns)
sdf = (
    sdf.tumbling_window(duration_ms=timedelta(minutes=5), grace_ms=timedelta(seconds=30))
       .agg(sum_5m=agg.Sum("amount"), n_5m=agg.Count(), max_5m=agg.Max("amount"))
       .current()            # emit on every update; .final() emits once the window closes
)
sdf = sdf.to_topic(out)       # downstream: push to the online feature store or score here
app.run()
```
In Feast, a `PushSource` plus `FeatureStore.push(...)` writes these values to the online store; `StreamFeatureView` with a `KafkaSource` declares a stream-backed feature view ([[feature-stores]]).

## Choosing / trade-offs
| Engine | Fits | Costs |
|---|---|---|
| Apache Flink (Java/SQL, PyFlink) | Large, stateful, event-time-heavy jobs; exactly-once state; Flink SQL for windowed features | Operating a cluster (or paying for managed Flink); JVM skills |
| Spark Structured Streaming | Teams already on Spark/Databricks; same code for batch backfills and streaming | Micro-batch by default (the Spark guide cites latencies down to about 100 ms), so higher latency than record-at-a-time engines |
| Kafka Streams / ksqlDB | JVM services that read and write Kafka only; no separate cluster | Kafka-only; JVM |
| Python libraries (Quix Streams and similar) | Python ML teams, moderate throughput, models in-process | Smaller ecosystem; check state, scaling and exactly-once support for your load |
| Plain consumer + model service | One simple feature, low volume | You rebuild windows, state and recovery yourself |

- **Embedded model vs model service:** embedding gives the lowest latency and no network hop, but couples model deploys to stream job deploys and makes GPU use awkward; a separate service ([[serving-with-fastapi]], [[inference-servers-vllm]]) scales and rolls out independently ([[ab-testing-and-canary-for-models]]).
- **Freshness vs cost:** sliding windows that update on every event are the most expensive; hopping windows (e.g. 5-minute window every 30 s) or tumbling windows plus request-time adjustments are cheaper.
- **Same code for training:** compute training features by replaying the topic (or the archived events) through the same job, or by a point-in-time join over the logged online features; separate batch SQL that "should match" is the classic skew source.
- **LLM calls in streams:** use async I/O with a concurrency cap and batching; route through a gateway with retries and fallbacks ([[llm-gateways-and-routing]]); write failures to a dead-letter topic instead of blocking the partition.

## Gotchas
- Processing-time windows look fine live and break on replay or catch-up after an outage, when hours of events arrive in minutes.
- Hot keys (one merchant or bot account with most of the traffic) overload a single partition and its state.
- Repartitioning (changing the key) creates an internal topic and a network shuffle; plan keys up front.
- A slow model call blocks the partition and grows consumer lag; set timeouts, and alert on lag, not only on errors.
- Exactly-once inside Kafka does not cover side effects: an email sent or an external API called during processing can still run twice. Make them idempotent ([[tool-calling]] has the same issue for agents).
- Watermarks that are too strict drop real data; too lax and results arrive late. Measure the lateness distribution before choosing allowed lateness.
- Schema changes break consumers silently; use a schema registry and compatible evolution rules.
- State size grows with keys times window length; set TTLs on state for keys that go quiet.
- Replaying a topic to rebuild features needs retention long enough for your longest window plus backfill needs; compacted or tiered storage changes what is available.

## Related
- [[feature-stores]] - online/offline stores the stream writes to; point-in-time training sets.
- [[online-learning-and-concept-drift]] - updating models and detecting drift on streams.
- [[anomaly-detection]], [[cusum-change-point-detection-from-scratch]] - streaming detectors.
- [[serving-with-fastapi]], [[inference-servers-vllm]] - the request/response side of inference.
- [[ab-testing-and-canary-for-models]] - rolling out a new streaming model safely.
- [[opentelemetry-genai-tracing]] - tracing LLM calls made from stream processors.
- [[mlops-lifecycle]] - where streaming pipelines fit in the lifecycle.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/data/batch-and-stream-processing.md - batch vs stream, windows and watermarks in general.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/system-design/message-queues-and-streams.md - Kafka, partitions and consumer groups.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/distributed-systems/idempotency-and-exactly-once.md - delivery semantics and idempotent sinks.

## References
- Apache Kafka documentation: https://kafka.apache.org/documentation/
- Apache Flink, timely stream processing (event time, watermarks): https://nightlies.apache.org/flink/flink-docs-stable/docs/concepts/time/
- Spark Structured Streaming: https://spark.apache.org/docs/latest/streaming/
- Quix Streams source and docs links: https://github.com/quixio/quix-streams; PyPI: https://pypi.org/project/quixstreams/
- Feast (push and stream sources): https://docs.feast.dev/
- Akidau, Chernyak, Lax, Streaming Systems (O'Reilly, 2018) - event time, watermarks and windows.
