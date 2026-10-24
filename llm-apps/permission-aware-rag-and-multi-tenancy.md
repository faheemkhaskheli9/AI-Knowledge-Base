---
title: Permission-aware RAG and multi-tenant retrieval (document ACLs, tenant isolation, pre-filtering)
category: llm-apps
tags: [rag, access-control, acl, multi-tenancy, tenant-isolation, metadata-filtering, qdrant, pgvector, row-level-security, weaviate, pinecone, data-leakage, owasp-llm08]
use_cases:
  - "make an internal RAG assistant answer only from documents the asking employee is allowed to open"
  - "serve many customers from one vector database without one tenant's chunks reaching another tenant's prompt"
  - "choose between a collection per tenant, a namespace per tenant and a shared collection with a tenant filter"
  - "keep search results correct after a user loses access to a SharePoint or Drive folder"
status: draft
last_verified: 2026-10-24
sources:
  - https://qdrant.tech/documentation/manage-data/multitenancy/
  - https://www.postgresql.org/docs/current/ddl-rowsecurity.html
  - https://github.com/pgvector/pgvector
  - https://docs.weaviate.io/weaviate/manage-collections/multi-tenancy
  - https://docs.pinecone.io/guides/index-data/implement-multitenancy
  - https://genai.owasp.org/llmrisk/llm082025-vector-and-embedding-weaknesses/
---

# Permission-aware RAG and multi-tenant retrieval (document ACLs, tenant isolation, pre-filtering)

## Summary
A RAG system copies documents out of systems that enforce permissions (Drive, SharePoint, Confluence, a ticketing tool, a database) into a vector index that, by default, does not. Without extra work any user can retrieve any chunk, and the LLM will happily quote it. Permission-aware retrieval stores who may read each chunk (tenant ID, group or user ACL) next to its vector, applies that as a filter inside the vector search (pre-filtering), and decides it from the authenticated identity on the server, never from the prompt. Multi-tenancy is the same problem at the customer level, with a choice of physical isolation (collection or namespace per tenant) or logical isolation (one collection with an indexed tenant field). OWASP lists cross-tenant and unauthorized retrieval under LLM08:2025, Vector and Embedding Weaknesses.

## Key concepts
- **Authorization happens at retrieval, not in the prompt**: telling the model "only use documents the user may see" does nothing; any chunk that enters the context can leak. Filter before the chunk reaches the prompt, and keep the filter out of the model's control ([[prompt-injection]]).
- **Pre-filter vs post-filter**: pre-filtering restricts the candidate set inside the ANN search, so top-k are the best *allowed* chunks. Post-filtering takes the global top-k and then drops forbidden ones; it returns fewer than k results, often zero for a user with access to a small slice, and the forbidden chunks have already left the database. Engines with filtered HNSW (Qdrant, Weaviate, pgvector iterative scans) make pre-filtering the default path.
- **ACL metadata on every chunk**: copy the source document's tenant ID and allowed principals (groups, users) onto every chunk at ingestion, as an indexed payload field, and filter with "tenant = T AND allowed_groups ANY OF user's groups". Index the field, or filtered search degrades to a scan.
- **Early vs late binding**: early binding stores ACLs in the index and trusts them at query time (fast, but stale until re-synced). Late binding re-checks each retrieved document against the source system or a permissions cache before use (always current, slower). Common practice: early binding for the filter plus a late check on the few final chunks, and an ACL sync job on permission-change events.
- **Isolation levels for tenants**: (1) separate database, collection or index per tenant, (2) namespace or shard per tenant (Pinecone namespaces, Weaviate tenants, Qdrant shard keys), (3) one shared collection with a tenant field filtered on every query (Qdrant payload partitioning, Postgres row-level security). Stronger isolation costs more per tenant and makes cross-tenant queries harder.
- **Qdrant payload partitioning**: Qdrant's multitenancy guide recommends one collection with a tenant payload field for many tenants, a keyword payload index with `is_tenant=True` (a storage hint that keeps a tenant's points together, not an access control), and optionally `hnsw_config` `m=0` with `payload_m=16` to build per-tenant graphs instead of a global one. Version 1.16 added tiered multitenancy: small tenants share a fallback shard and large ones are promoted to a dedicated shard.
- **Postgres row-level security (RLS)**: with pgvector the chunks are ordinary rows, so `ENABLE ROW LEVEL SECURITY` plus a `CREATE POLICY ... USING (tenant_id = current_setting('app.tenant_id'))` makes the database add the predicate to every query. Superusers, roles with `BYPASSRLS`, and the table owner (unless `FORCE ROW LEVEL SECURITY` is set) bypass policies.
- **Weaviate and Pinecone**: Weaviate's multi-tenancy gives each tenant its own shard; every request names a tenant, and tenants can be `ACTIVE`, `INACTIVE` (on disk) or `OFFLOADED` (cloud storage). Pinecone's guide uses one namespace per tenant in a serverless index; every data-plane operation targets exactly one namespace and offboarding is deleting it.
- **Everything derived from documents inherits their ACLs**: caches, summaries, extracted entities, knowledge-graph nodes, fine-tuning data, eval sets and traces all carry the content; each needs the same tenant and permission scope or it becomes a side channel.

## When to use / scenarios
- Internal knowledge assistants over HR, finance, legal or engineering documents where employees have different rights ([[rag-basics]], [[legal]], [[finance]]).
- B2B SaaS that adds "chat with your data" for each customer on shared infrastructure ([[vector-databases]], [[multi-lora-adapter-serving]] for per-tenant models).
- Support bots that read a customer's own tickets and orders: the user ID comes from the session, never from the message ([[customer-support]]).
- Agents with search tools: the tool runs with the end user's identity, not a service account that sees everything ([[tool-calling]], [[agents]]).
- Not needed for a single-tenant corpus everyone may read (public docs, a product manual); a plain index is fine.
- Not a substitute for data minimisation: content no user should see through chat (secrets, raw PII) is better excluded or redacted at ingestion ([[pii-redaction-and-pseudonymization]]).

## Setup & code
Qdrant: one shared collection with a tenant index and per-tenant HNSW graphs, then a query whose filter comes from the authenticated session. Parameter names are from the Qdrant multitenancy guide; check them against the client version you install.
<!-- skip-check: needs a running Qdrant server -->
```python
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")
client.create_collection(
    collection_name="docs",
    vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
    hnsw_config=models.HnswConfigDiff(payload_m=16, m=0),   # per-tenant graphs, no global graph
)
client.create_payload_index(
    collection_name="docs",
    field_name="tenant_id",
    field_schema=models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, is_tenant=True),
)
client.create_payload_index(collection_name="docs", field_name="allowed_groups",
                            field_schema=models.PayloadSchemaType.KEYWORD)

def search(session, query_vector, k=5):
    # tenant and groups come from the verified token on the server, never from the prompt or the client body
    flt = models.Filter(must=[
        models.FieldCondition(key="tenant_id", match=models.MatchValue(value=session["tenant_id"])),
        models.FieldCondition(key="allowed_groups", match=models.MatchAny(any=session["groups"])),
    ])
    return client.query_points("docs", query=query_vector, query_filter=flt, limit=k).points
```
Postgres with pgvector: row-level security puts the tenant predicate into every query, including ones a developer forgets to filter. The app connects as a non-owner role without `BYPASSRLS` and sets the tenant per transaction.
```sql
CREATE TABLE chunks (
  id bigserial PRIMARY KEY,
  tenant_id text NOT NULL,
  allowed_groups text[] NOT NULL,
  body text NOT NULL,
  embedding vector(768) NOT NULL
);
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON chunks (tenant_id);
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_and_groups ON chunks
  USING (tenant_id = current_setting('app.tenant_id')
         AND allowed_groups && string_to_array(current_setting('app.groups'), ','));

-- per request, inside a transaction:
BEGIN;
SELECT set_config('app.tenant_id', 'acme', true), set_config('app.groups', 'eng,all-staff', true);
SET LOCAL hnsw.iterative_scan = strict_order;   -- pgvector 0.8+: keep scanning when the filter removes candidates
SELECT id, body FROM chunks ORDER BY embedding <=> $1 LIMIT 5;
COMMIT;
```
Runnable, NumPy only: a toy index with two tenants and group ACLs, comparing no filter (leaks), post-filter (correct but starves), pre-filter (correct and full), a cache keyed without the tenant (leaks), and a late-binding check that catches a revoked permission the index still holds.
```python
import numpy as np

rng = np.random.default_rng(0)
DIM, N = 32, 400
vecs = rng.normal(size=(N, DIM))
vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
tenants = np.where(np.arange(N) < 360, "big-co", "small-co")        # small-co owns 10% of the chunks
groups = [("all-staff",) if i % 4 else ("finance",) for i in range(N)]  # 1 in 4 chunks is finance-only
chunk_acl = {i: (tenants[i], set(groups[i])) for i in range(N)}

def allowed(i, user):
    t, g = chunk_acl[i]
    return t == user["tenant"] and bool(g & user["groups"])

def top_k(q, k, candidates=None):
    idx = np.arange(N) if candidates is None else np.asarray(candidates)
    scores = vecs[idx] @ q
    return [int(i) for i in idx[np.argsort(-scores)[:k]]]

def no_filter(q, user, k):
    return top_k(q, k)

def post_filter(q, user, k):
    return [i for i in top_k(q, k) if allowed(i, user)]      # forbidden chunks already left the DB

def pre_filter(q, user, k):
    return top_k(q, k, [i for i in range(N) if allowed(i, user)])

users = {
    "big-co staff": {"tenant": "big-co", "groups": {"all-staff"}},
    "small-co staff": {"tenant": "small-co", "groups": {"all-staff"}},
    "small-co finance": {"tenant": "small-co", "groups": {"all-staff", "finance"}},
}
K, Q = 5, 200
queries = rng.normal(size=(Q, DIM))
queries /= np.linalg.norm(queries, axis=1, keepdims=True)
print(f"{'user':18s} {'strategy':12s} {'leaked/query':>12s} {'results/query':>13s}")
for name, user in users.items():
    for strat in (no_filter, post_filter, pre_filter):
        res = [strat(q, user, K) for q in queries]
        leaked = np.mean([sum(not allowed(i, user) for i in r) for r in res])
        print(f"{name:18s} {strat.__name__:12s} {leaked:12.2f} {np.mean([len(r) for r in res]):13.2f}")

# A response or semantic cache keyed only on the question text serves one tenant's answer to another.
cache = {}
def answer(question, user, key_has_tenant):
    key = (user["tenant"], frozenset(user["groups"]), question) if key_has_tenant else question
    if key not in cache:
        cache[key] = ("answer built from", user["tenant"])
    return cache[key]
for flag in (False, True):
    cache.clear()
    answer("What is our Q3 revenue?", users["big-co staff"], flag)
    got = answer("What is our Q3 revenue?", users["small-co staff"], flag)
    print(f"cache key includes tenant={flag}: small-co sees {got}")

# Late binding: the index still says chunk 3 is all-staff, but the source system revoked it.
source_of_truth = dict(chunk_acl)
source_of_truth[3] = ("big-co", {"finance"})
def late_check(ids, user):
    return [i for i in ids if source_of_truth[i][0] == user["tenant"] and source_of_truth[i][1] & user["groups"]]
user = users["big-co staff"]
hits = pre_filter(vecs[3], user, K)
print("pre-filter on stale index:", hits[:3], "| after late check:", late_check(hits, user)[:3])
```
Typical output: no filter leaks most results for small-co users, post-filter returns well under five chunks for them (often zero), pre-filter returns five allowed chunks for everyone, the tenant-blind cache hands small-co big-co's answer, and the late check drops chunk 3.

## Choosing / trade-offs
| Layout | Isolation | Cost per tenant | Cross-tenant query | Pick when |
|---|---|---|---|---|
| Database or cluster per tenant | Strongest, separate keys and backups | Highest | Not possible | Regulated customers, contractual isolation, per-tenant region |
| Collection / index per tenant | Strong; a missing filter cannot leak | High overhead with thousands of tenants | Fan-out | Tens to hundreds of large tenants |
| Namespace / tenant shard (Pinecone, Weaviate, Qdrant shard keys) | Strong per request; one API call reaches one tenant | Low to moderate | Fan-out | Many tenants, simple offboarding by delete |
| Shared collection + indexed tenant filter / RLS | Logical; relies on every query being filtered | Lowest | Easy | Very many small tenants; enforce with RLS or a single query wrapper |

- **Where to enforce**: one retrieval service that takes an identity and adds the filter itself is safer than filters scattered through app code. With Postgres, RLS makes the database enforce it even for ad-hoc queries.
- **Group explosion**: users in hundreds of groups make "ANY OF" filters large; resolve groups at login, cache them per session, and prefer coarse document-level ACLs on chunks over per-paragraph ones.
- **ACL freshness vs cost**: event-driven ACL sync (webhooks or change feeds from the source system) keeps early-bound ACLs current; a nightly full sync leaves a window where revoked users still retrieve. Add a late check when that window is unacceptable.
- **Recall under restrictive filters**: very selective filters can starve approximate search. Qdrant switches to a payload-index plan for small filtered sets; pgvector needs iterative scans or a higher `hnsw.ef_search`; check recall per tenant size in your eval ([[rag-evaluation]]).

## Gotchas
- **Tenant ID from the request body or the prompt**: a user can change it. Derive tenant and groups on the server from the verified token or session.
- **Post-filtering in app code**: looks safe in a demo with one tenant; in production small tenants get empty answers, and forbidden chunks are fetched, logged and possibly cached before being dropped.
- **Caches without scope**: exact, semantic and prompt caches keyed only on the question serve one tenant's answer to another; include tenant and permission set in the key, or cache only public content ([[semantic-caching]]).
- **Derived artifacts lose ACLs**: summaries, GraphRAG entities, extracted FAQs and fine-tuned models built from many documents mix their permissions; restrict them to the intersection of readers or keep them per tenant ([[graphrag-and-knowledge-graphs]], [[fine-tuning-and-peft]]).
- **Traces and eval sets**: observability tools store full prompts with retrieved chunks; scope trace access by tenant, redact, or log chunk IDs instead of text ([[llm-observability]]).
- **Deleted and revoked content**: removing a document in the source system must delete its chunks (and cached answers) in the index; test it, since many sync connectors only add and update.
- **Owner role bypasses RLS**: running the app as the table owner, a superuser or a `BYPASSRLS` role silently disables policies; use a dedicated role and `FORCE ROW LEVEL SECURITY`.
- **`is_tenant` is not access control**: in Qdrant it only optimises storage layout; the filter on every query is what isolates tenants.
- **Indirect prompt injection across tenants**: in a shared index, a malicious tenant's document that reaches another tenant's prompt can carry instructions; tenant filtering is also an injection defence ([[prompt-injection]]).

## Related
- [[rag-basics]] - the pipeline this adds authorization to.
- [[vector-databases]] - filtered search, payload indexes and the engines compared.
- [[advanced-rag]] - hybrid search and reranking, which must apply the same filter on every retriever.
- [[semantic-caching]] - cache keys that include tenant and permission scope.
- [[pii-redaction-and-pseudonymization]] - removing data no user should retrieve through chat.
- [[prompt-injection]] - why the filter must not be under the model's control.
- [[guardrails-and-safety]] - output checks as a second line, not the access control.
- [[ai-security-privacy-compliance]] - data-protection duties for the stored chunks.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/security/authorization-models.md - RBAC, ABAC and ReBAC models behind the ACLs.

## References
- Qdrant, Configure multitenancy (payload partitioning, `is_tenant`, `payload_m`/`m=0`, tiered multitenancy): https://qdrant.tech/documentation/manage-data/multitenancy/
- PostgreSQL documentation, Row Security Policies (`ENABLE`/`FORCE ROW LEVEL SECURITY`, `BYPASSRLS`, owner bypass): https://www.postgresql.org/docs/current/ddl-rowsecurity.html
- pgvector README (HNSW, filtering, iterative index scans, `hnsw.ef_search`): https://github.com/pgvector/pgvector
- Weaviate, Multi-tenancy operations (tenant shards, activity status): https://docs.weaviate.io/weaviate/manage-collections/multi-tenancy
- Pinecone, Implement multitenancy using namespaces: https://docs.pinecone.io/guides/index-data/implement-multitenancy
- OWASP GenAI Security Project, LLM08:2025 Vector and Embedding Weaknesses: https://genai.owasp.org/llmrisk/llm082025-vector-and-embedding-weaknesses/
