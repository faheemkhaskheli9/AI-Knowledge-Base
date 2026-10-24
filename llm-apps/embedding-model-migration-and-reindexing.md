---
title: Embedding model migration and re-indexing (dual writes, backfill, alias cut-over)
category: llm-apps
tags: [embedding-migration, re-indexing, re-embedding, blue-green, dual-write, backfill, collection-alias, named-vectors, qdrant, pgvector, vector-database, retrieval-evaluation, rollback]
use_cases:
  - "switch our RAG index to a newer or fine-tuned embedding model without search downtime"
  - "our hosted embedding model is being retired; re-embed millions of chunks and cut over safely"
  - "prove a new embedding model retrieves better on our own queries before moving traffic"
  - "roll back to the old embeddings if the new model makes search worse"
  - "change vector dimension or distance metric in pgvector or Qdrant on a live table"
status: draft
last_verified: 2026-10-24
sources:
  - https://github.com/qdrant/landing_page/blob/master/qdrant-landing/content/documentation/tutorials-operations/embedding-model-migration.md
  - https://github.com/qdrant/landing_page/blob/master/qdrant-landing/content/documentation/manage-data/collections.md
  - https://github.com/pgvector/pgvector/blob/master/README.md
  - https://github.com/AmenRa/ranx
---

# Embedding model migration and re-indexing (dual writes, backfill, alias cut-over)

## Summary
Vectors from two embedding models live in different spaces: a query embedded with the new model cannot be searched against documents embedded with the old one, even when the dimensions match. Changing the embedder (a better model, a fine-tune, a provider retiring a model, a new dimension or quantization) therefore means re-embedding the whole corpus and switching every query at the same moment. This topic is the zero-downtime procedure: build the new index beside the old one, dual-write new data to both, backfill old data idempotently, compare retrieval quality on your own labelled queries, cut over atomically with an alias, keep the old index for rollback, then clean up. It follows Qdrant's migration tutorial and maps the same steps to pgvector.

## Key concepts
- **The model is part of the index schema**: record the model name and revision, dimension, distance metric, normalization and any query/document prefix with every index ([[embedding-and-reranker-serving]]). Any change to these is a migration, not a config tweak.
- **Two layouts for side-by-side vectors**:
  - *Blue-green collections*: a second collection or table sized for the new model. Works everywhere; duplicates payloads, and deletes and partial updates need extra handling during migration.
  - *Second vector on the same record*: Qdrant named vectors (adding or removing a named vector on an existing collection is available from Qdrant 1.18 per its docs) or a second column in pgvector (`ALTER TABLE ... ADD COLUMN embedding_v2 vector(<dim>)`). No payload copy; rollback is switching which vector the query uses.
- **Dual writes**: from the moment the new index exists, every insert or update is embedded with both models and written to both places, so the backfill only has to cover data that existed before. Both writes must succeed; failures go to a retry or dead-letter queue.
- **Idempotent backfill**: a background job scrolls old records in batches, re-embeds the stored source text, and writes to the new index without overwriting newer dual-written rows (Qdrant's insert-only upsert mode; in SQL, `WHERE embedding_v2 IS NULL` or a version check). Store the scroll offset so it resumes after a crash. Re-embedding needs the original text, so keep it in the payload or primary database.
- **Deletes and partial updates during backfill**: Qdrant's tutorial notes that its blue-green procedure only works as-is for upserts. A delete that runs before the backfill copies a record lets the backfill resurrect it; pause deletes and partial updates during backfill or record tombstones and apply them after.
- **Completeness check**: before switching, count records missing the new vector (should be 0) or compare counts between collections.
- **Quality gate**: run the same labelled queries against both indexes and compare Recall@k, MRR and nDCG@k; read the per-query regressions, not only the average ([[rag-evaluation]]). Without labels you can only measure how much the results change (top-k overlap), not whether they improved.
- **Atomic cut-over**: queries address an alias (Qdrant collection aliases, a database view, or one config value read by every query service) that you repoint in one operation; Qdrant applies multiple alias actions atomically. Query-side model and index must switch together.
- **Rollback window**: keep dual writes on after cut-over; while the old index still receives every write, rollback is repointing the alias. Turning off dual writes or deleting the old vector is the point of no return.

## When to use / scenarios
- Upgrading to a stronger general embedder or a domain fine-tune ([[embedding-fine-tuning]], [[embedding-models]]).
- A hosted embedding model has a published retirement date; plan the re-embed well before it ([[model-selection]]).
- Changing dimension (Matryoshka truncation, a smaller model), precision (half precision, binary quantization) or distance metric to cut memory ([[vector-databases]]).
- Changing chunking: new chunk boundaries also need a full rebuild, and the same dual-index procedure applies ([[chunking-strategies]]).
- Not needed when only the reranker or the generator model changes; those run at query time on retrieved text ([[rerankers]]).
- If re-embedding the corpus is not possible (no access to source text, vendor-owned vectors), a query-side adapter keeps document vectors fixed ([[embedding-fine-tuning]]).

## Setup & code
pgvector, second-column layout (DDL from the pgvector README; build the index concurrently so writes are not blocked):
```sql
ALTER TABLE chunks ADD COLUMN embedding_v2 vector(<new_dim>);     -- the new model's output dimension
-- dual-write in the application, then backfill in batches:
UPDATE chunks SET embedding_v2 = $1 WHERE id = $2 AND embedding_v2 IS NULL;
CREATE INDEX CONCURRENTLY chunks_embedding_v2_hnsw ON chunks USING hnsw (embedding_v2 vector_cosine_ops);
SELECT count(*) FROM chunks WHERE embedding_v2 IS NULL;          -- completeness check, expect 0
-- cut over: queries switch to ORDER BY embedding_v2 <=> $query_v2 (with the new model's query embedding)
-- after the rollback window: DROP INDEX ...; ALTER TABLE chunks DROP COLUMN embedding;
```
Take `<new_dim>` from the new model's card. pgvector's HNSW index supports up to 2,000 dimensions for `vector` and 4,000 for `halfvec` per its README, so a larger model needs half-precision indexing, binary quantization or subvector indexing. Index builds are much faster when the graph fits in `maintenance_work_mem`.

Qdrant: create the new collection (or add a named vector), dual-write, backfill with insert-only upserts, then swap the alias in one request (method names from the Qdrant Python client; see the tutorial for the full code in each SDK).

Runnable with NumPy only: a synthetic corpus, an "old" and a "new" embedder that live in different spaces, the whole migration (dual writes during an idempotent resumable backfill, a delete tombstone, completeness check, quality gate, alias swap and rollback) and the metrics used to decide.
```python
import numpy as np

rng = np.random.default_rng(0)
N_DOCS, N_TOPICS, LATENT = 600, 60, 32

# Synthetic world: each doc and query has a latent meaning vector; relevant = same topic.
topic_centres = rng.normal(size=(N_TOPICS, LATENT))
doc_topic = rng.integers(0, N_TOPICS, N_DOCS)
doc_latent = topic_centres[doc_topic] + 0.6 * rng.normal(size=(N_DOCS, LATENT))

def make_embedder(dim, noise, seed):
    """An 'embedding model': its own random projection of the latent meaning plus model-specific noise."""
    proj = np.random.default_rng(seed).normal(size=(LATENT, dim))
    def embed(latent, rng_noise):
        v = latent @ proj + noise * rng_noise.normal(size=(len(latent), dim))
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    return embed, {"dim": dim, "seed": seed}

old_embed, old_meta = make_embedder(dim=64, noise=4.0, seed=1)   # weaker model
new_embed, new_meta = make_embedder(dim=64, noise=2.0, seed=2)   # stronger model, SAME dim, different space

class Index:
    def __init__(self, name, meta):
        self.name, self.meta, self.vecs = name, meta, {}
    def upsert(self, doc_id, vec, insert_only=False):
        if insert_only and doc_id in self.vecs:
            return False                                  # never overwrite a newer dual-written row
        self.vecs[doc_id] = vec
        return True
    def delete(self, doc_id):
        self.vecs.pop(doc_id, None)
    def search(self, q, k):
        ids = np.fromiter(self.vecs, dtype=int)
        mat = np.stack([self.vecs[i] for i in ids])
        return ids[np.argsort(-(mat @ q))[:k]].tolist()

source_text = {i: doc_latent[i] for i in range(N_DOCS)}   # stands in for the stored chunk text
noise_rng = np.random.default_rng(42)
old_ix = Index("chunks_v1", old_meta)
for i in range(N_DOCS):
    old_ix.upsert(i, old_embed(source_text[i][None], noise_rng)[0])
aliases = {"chunks": old_ix}                              # queries only ever use the alias

# 1. New index + dual writes. 2. Backfill in batches with a persisted offset, live traffic interleaved.
new_ix = Index("chunks_v2", new_meta)
def write(doc_id, latent):                                # the application's write path during migration
    source_text[doc_id] = latent
    for ix, emb in ((old_ix, old_embed), (new_ix, new_embed)):
        ix.upsert(doc_id, emb(latent[None], noise_rng)[0])

tombstones, offset, BATCH = set(), 0, 50
live_rng = np.random.default_rng(7)
while offset < N_DOCS:
    batch = [i for i in range(offset, min(offset + BATCH, N_DOCS)) if i in source_text and i not in tombstones]
    vecs = new_embed(np.stack([source_text[i] for i in batch]), noise_rng) if batch else []
    for i, v in zip(batch, vecs):
        new_ix.upsert(i, v, insert_only=True)
    offset += BATCH                                       # persist this in real life to resume after a crash
    if live_rng.random() < 0.5:                           # an update to an existing doc arrives mid-backfill
        d = int(live_rng.integers(0, N_DOCS))
        write(d, topic_centres[doc_topic[d]] + 0.6 * live_rng.normal(size=LATENT))
    if offset == 300:                                     # a delete arrives: record a tombstone too
        old_ix.delete(5); new_ix.delete(5); source_text.pop(5); tombstones.add(5)
for d in tombstones:                                      # re-apply deletes after backfill
    new_ix.delete(d)
missing = set(old_ix.vecs) - set(new_ix.vecs)
print(f"completeness: old={len(old_ix.vecs)} new={len(new_ix.vecs)} missing={len(missing)}")
assert not missing and set(new_ix.vecs) == set(old_ix.vecs)

# 3. Quality gate on labelled queries (relevant = same topic), same k for both setups.
n_q = 120
q_topic = rng.integers(0, N_TOPICS, n_q)
q_latent = topic_centres[q_topic] + 0.6 * rng.normal(size=(n_q, LATENT))
relevant = [{i for i in old_ix.vecs if doc_topic[i] == t} for t in q_topic]

def metrics(ranked, rel, k=10):
    hits = [1 if d in rel else 0 for d in ranked[:k]]
    recall = sum(hits) / max(len(rel), 1)
    first = next((r for r, h in enumerate(hits, 1) if h), None)
    mrr = 1 / first if first else 0.0
    dcg = sum(h / np.log2(r + 1) for r, h in enumerate(hits, 1))
    idcg = sum(1 / np.log2(r + 1) for r in range(1, min(len(rel), k) + 1))
    return recall, mrr, dcg / idcg if idcg else 0.0

def evaluate(ix, embed, k=10):
    qv = embed(q_latent, np.random.default_rng(99))
    runs = [ix.search(q, k) for q in qv]
    return runs, np.array([metrics(r, rel, k) for r, rel in zip(runs, relevant)])

old_runs, old_m = evaluate(old_ix, old_embed)
new_runs, new_m = evaluate(new_ix, new_embed)
_, cross_m = evaluate(old_ix, new_embed)                 # the classic bug: new query model, old index
for name, m in (("old model / old index", old_m), ("new model / new index", new_m), ("new query / OLD index", cross_m)):
    print(f"{name:24s} recall@10={m[:, 0].mean():.3f} MRR={m[:, 1].mean():.3f} nDCG@10={m[:, 2].mean():.3f}")
chance = 10 / len(old_ix.vecs)                            # expected recall@10 of a random ranking
print(f"random-ranking recall@10 ~ {chance:.3f}: the cross-space search is close to random, and raises no error")

delta = new_m[:, 2] - old_m[:, 2]
worst = np.argsort(delta)[:3]
print(f"per-query nDCG: {int((delta < 0).sum())} of {n_q} queries got worse; read these first: {worst.tolist()}")
overlap = np.mean([len(set(a) & set(b)) / len(a) for a, b in zip(old_runs, new_runs)])
print(f"top-10 overlap old vs new (the only signal without labels): {overlap:.2f}")

# 4. Cut over atomically if the pre-agreed bar is met; keep dual writes on for rollback.
BAR = 0.0                                                 # decide the minimum nDCG gain before looking at results
if new_m[:, 2].mean() - old_m[:, 2].mean() > BAR:
    aliases["chunks"] = new_ix                            # one atomic swap; query embedder switches with it
print("alias 'chunks' ->", aliases["chunks"].name, aliases["chunks"].meta)
aliases["chunks"] = old_ix                                # rollback is the same swap while dual writes run
print("rolled back to", aliases["chunks"].name)
```

## Choosing / trade-offs
| Choice | Pushes toward | Cost |
|---|---|---|
| Blue-green second collection/table | Any store, also changing sharding, replication or index params | Payload copied twice; deletes and partial updates need pausing or tombstones |
| Second vector on the same record (Qdrant named vector, pgvector column) | Store supports it, large payloads | Index on the same record set; the table/collection temporarily holds both vectors |
| Re-embed from stored text in the payload | Self-contained, simple job | Payload must hold the exact text that was embedded |
| Re-embed from the primary database | Payload is thin or text has changed since | Join source rows to index IDs; snapshot consistency |
| Full re-embed vs only changed chunks | Model change needs everything; chunking change needs everything; metadata-only change needs nothing | Embedding compute ([[batch-inference-and-batch-apis]]) or provider batch API cost |
| Cut over by alias | Atomic, no code deploy | Every query path must use the alias and the matching query model |

- **Throughput of the backfill**: rate-limit it so it does not starve query-time embedding; run it on a separate embedding deployment or a batch API ([[embedding-and-reranker-serving]], [[llm-api-rate-limits-and-retries]]).
- **Bigger is not free**: a model with larger vectors costs memory, disk and possibly latency; compare those on the new index next to quality.
- **Semantic and prompt caches keyed on embeddings must be rebuilt too**, or they return stale hits from the old space ([[semantic-caching]]).

## Gotchas
- Same dimension does not mean compatible: two models with equal output dimensions still produce unrelated spaces. Mixing them gives near-random results with no error.
- Query and document prefixes or instructions (for models that need them) are part of the model contract; a missing prefix on one side degrades recall silently.
- Switching the alias but not the query-side model (or the reverse) is the most common cut-over bug; deploy both from one config value.
- A delete or partial update that lands during backfill can be undone by the backfill unless deletes are paused or tombstoned.
- Backfilling with plain upserts overwrites rows that dual writes already refreshed with newer text; use insert-only or a version check.
- An HNSW index built while rows are still being backfilled is fine in pgvector with `CREATE INDEX CONCURRENTLY`, but a build that runs out of `maintenance_work_mem` slows sharply; plan memory for the rebuild.
- Averages hide regressions: a new model can win overall and lose on a whole query class (product codes, another language). Read the worst queries.
- Benchmarks such as MTEB do not predict your corpus; the gate must use your own queries ([[reading-ai-benchmarks]]).
- Deleting the old vector or turning off dual writes ends the rollback option; snapshot the old collection first if you may need it.
- Hybrid search: sparse/BM25 scores are unaffected, but fusion weights tuned for the old dense model may need retuning ([[advanced-rag]]).

## Related
- [[vector-databases]] - stores, indexes and aliases this procedure runs on.
- [[embedding-models]] - choosing the model you migrate to.
- [[embedding-fine-tuning]] - a fine-tuned embedder always needs this migration.
- [[embedding-and-reranker-serving]] - running the backfill without hurting query traffic; model identity on every vector.
- [[rag-evaluation]] - building the labelled query set and retrieval metrics for the quality gate.
- [[rag-basics]] - the pipeline whose index is being rebuilt.
- [[chunking-strategies]] - chunking changes need the same rebuild.
- [[semantic-caching]] - caches keyed on old-space vectors must be invalidated.
- [[batch-inference-and-batch-apis]] - cheap bulk re-embedding.
- [[ab-testing-and-canary-for-models]] - an online check after the offline gate.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/data/schema-migrations.md - expand/contract migrations, the same pattern for relational schemas.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/data/vector-databases-for-engineers.md - operating vector stores.

## References
- Qdrant, Migrate to a New Embedding Model with Zero Downtime (blue-green and named-vector procedures, insert-only backfill, quality check): https://github.com/qdrant/landing_page/blob/master/qdrant-landing/content/documentation/tutorials-operations/embedding-model-migration.md
- Qdrant, Collections (collection aliases and atomic switch, update vector schema): https://github.com/qdrant/landing_page/blob/master/qdrant-landing/content/documentation/manage-data/collections.md
- pgvector README (index dimension limits, `CREATE INDEX CONCURRENTLY`, `maintenance_work_mem`, half precision, binary quantization, subvector indexing): https://github.com/pgvector/pgvector/blob/master/README.md
- ranx, IR evaluation library (Recall, MRR, nDCG): https://github.com/AmenRa/ranx
