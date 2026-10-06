---
title: Vector databases and indexes
category: llm-apps
tags: [vector-database, pgvector, qdrant, chroma, faiss, hnsw, similarity-search]
use_cases:
  - "pick a vector store for a RAG app already running on Postgres"
  - "store 50 million embeddings with metadata filtering at low latency"
  - "prototype semantic search locally with no server"
  - "multi-tenant document search where each customer sees only their data"
  - "add semantic memory to an agent"
status: draft
last_verified: 2026-10-03
sources:
  - https://github.com/pgvector/pgvector
  - https://qdrant.tech/documentation/
  - https://docs.trychroma.com/
  - https://github.com/facebookresearch/faiss
---

# Vector databases and indexes

## Summary
A vector store holds embeddings and returns nearest neighbours fast, usually with metadata filters. The choice is mostly operational: reuse what you already run (Postgres -> pgvector), use a purpose-built engine at scale (Qdrant, Milvus, Weaviate, Pinecone), or an embedded library for prototypes (Chroma, FAISS, LanceDB).

## Key concepts
- **Distance metric**: cosine, dot product, L2; use the metric your embedding model was trained for (normalized vectors: cosine == dot).
- **Exact vs approximate (ANN)**: brute force is exact and fine up to ~100k-1M vectors; **HNSW** (graph, high recall, memory-hungry) and **IVF** (clusters, smaller) trade recall for speed. Tune `ef_search`/`nprobe` for recall.
- **Quantization**: scalar/product/binary quantization cuts memory 4-32x with small recall loss; rescoring with full vectors recovers it.
- **Filtering**: pre-filter vs post-filter; selective filters can hurt ANN recall, so check how the engine handles filtered HNSW.
- **Hybrid search**: dense + sparse/BM25 in one engine ([[advanced-rag]]).
- **Dimension and model are part of the schema**: changing embedder = re-embed ([[embedding-models]]).

## When to use / scenarios
- Any RAG ([[rag-basics]]), semantic search, dedup, recommendations, agent semantic memory ([[agent-memory]]).
- Under ~1M chunks, a single Postgres or an in-process index is enough; do not add infrastructure early.
- Not needed for exact-match/keyword-only search (use SQL/Elasticsearch).

## Setup & code
| Option | Type | Strengths | Watch out |
|---|---|---|---|
| **pgvector** | Postgres extension | One DB for rows + vectors, SQL joins, transactions, RLS for tenancy; HNSW/IVFFlat | Scale/latency tuning is yours; very large indexes need RAM |
| **Qdrant** | Dedicated engine (Rust), OSS + cloud | Strong filtering, sparse+dense hybrid, quantization, multi-tenancy | Another service to run |
| **Chroma** | Embedded/server, OSS | Simplest API, great for prototypes | Less suited to very large/production-critical loads |
| **FAISS** | Library | Fastest raw ANN, many index types, GPU | No persistence/filtering/CRUD layer; you build it |
| **Milvus / Weaviate** | Dedicated, OSS + cloud | Scale-out, built-in hybrid/modules | Heavier ops |
| **Pinecone** | Managed | Zero ops, serverless | Vendor lock-in, cost; data leaves your network |
| **LanceDB / sqlite-vec / DuckDB vss** | Embedded | Local-first, file-based | Smaller ecosystem |
| **Elasticsearch/OpenSearch** | Search engine | Best lexical+vector hybrid in one | Heavier to run |

pgvector:
```python
# pip install "psycopg[binary]" pgvector numpy ; Postgres with: CREATE EXTENSION vector;
import numpy as np, psycopg
from pgvector.psycopg import register_vector

conn = psycopg.connect("postgresql://user:pw@localhost/db", autocommit=True)
conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
register_vector(conn)
conn.execute("CREATE TABLE IF NOT EXISTS chunks (id text PRIMARY KEY, body text, tenant text, embedding vector(384))")
conn.execute("CREATE INDEX IF NOT EXISTS chunks_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)")
conn.execute("INSERT INTO chunks VALUES (%s,%s,%s,%s) ON CONFLICT (id) DO NOTHING",
             ("a1", "hello", "acme", np.random.rand(384)))
rows = conn.execute("SELECT id, body FROM chunks WHERE tenant=%s ORDER BY embedding <=> %s LIMIT 5",
                    ("acme", np.random.rand(384))).fetchall()
```
Qdrant (local mode):
```python
# pip install qdrant-client
from qdrant_client import QdrantClient, models
q = QdrantClient(":memory:")            # or QdrantClient(url="http://localhost:6333")
q.create_collection("docs", vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE))
q.upsert("docs", points=[models.PointStruct(id=1, vector=[0.1]*384, payload={"tenant": "acme"})])
hits = q.query_points("docs", query=[0.1]*384, limit=5,
        query_filter=models.Filter(must=[models.FieldCondition(key="tenant", match=models.MatchValue(value="acme"))])).points
```

## Choosing / trade-offs
- **Already on Postgres and < ~10M vectors**: pgvector. **Heavy filtering, hybrid, or large scale, self-hosted**: Qdrant/Milvus. **No ops**: managed (Pinecone, cloud Qdrant). **Prototype/notebook**: Chroma or FAISS.
- **Memory vs recall**: HNSW is fastest but RAM-bound; quantize or use disk-based indexes for billions.
- **Multi-tenancy**: filter on tenant payload (Qdrant, pgvector with RLS) vs collection-per-tenant (isolation, but overhead). See [[ai-security-privacy-compliance]].
- **Cost**: managed pricing scales with vectors and queries; self-hosting scales with RAM.

## Gotchas
- Create the ANN index after bulk load (or expect slow builds); check the query actually uses it (`EXPLAIN`).
- Wrong metric/ops class (`vector_l2_ops` vs cosine) silently degrades results.
- Upsert by deterministic id for idempotent ingestion ([[rag-basics]]).
- Filter + ANN can return fewer than k results; over-fetch or tune.
- Deleting source docs must delete their vectors (privacy/right-to-erasure).
- Benchmarks on public datasets rarely reflect your data; test recall against exact search on a sample.

## Related
- [[rag-basics]] - consumer of the store.
- [[advanced-rag]] - hybrid and reranking.
- [[embedding-models]] - dimensions and metrics.
- [[agent-memory]] - semantic memory tier.
- [[embeddings]] - concept background.
- [[hnsw-from-scratch]] - the HNSW index most vector DBs use, built by hand.

## References
- https://github.com/pgvector/pgvector
- https://qdrant.tech/documentation/
- https://docs.trychroma.com/
- https://github.com/facebookresearch/faiss
