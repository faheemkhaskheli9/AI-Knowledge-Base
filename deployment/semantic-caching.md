---
title: Semantic caching for LLM responses
category: deployment
tags: [semantic-cache, caching, embeddings, cost, latency, redisvl, invalidation, llmops]
use_cases:
  - "cut LLM cost for a support bot that gets the same questions phrased many ways"
  - "serve repeated FAQ answers from a cache without serving a wrong answer to a near-miss question"
  - "pick a similarity threshold and TTL for an embedding-keyed response cache"
  - "invalidate cached LLM answers when the docs, prompt or model change"
status: draft
last_verified: 2026-10-15
sources:
  - https://redis.io/docs/latest/develop/ai/redisvl/user_guide/how_to_guides/llmcache/
  - https://github.com/redis/redis-vl-python
  - https://github.com/zilliztech/GPTCache
  - https://platform.claude.com/docs/en/build-with-claude/prompt-caching
---

# Semantic caching for LLM responses

## Summary
A semantic cache stores past LLM answers keyed by the embedding of the prompt, and returns a stored answer when a new prompt is "close enough" (cosine similarity above a threshold). It can skip the model call entirely for repeated questions phrased differently, saving cost and latency, but every hit is a bet that the old answer is still right for the new question. The design work is choosing what to cache, the threshold, the cache key scope and how entries get invalidated.

## Key concepts
- **Exact cache vs semantic cache.** Exact: hash of the normalised prompt; zero false hits, low hit rate. Semantic: nearest-neighbour search over prompt embeddings; higher hit rate, some false hits. Run exact first, semantic second.
- **Not the same as provider prompt caching.** Prompt caching ([[prompt-caching-and-cost]]) reuses the provider's computation for a repeated *prefix* and still generates a fresh answer. A semantic cache returns an *old answer* and never calls the model.
- **Threshold.** Similarity (or distance) cut-off. Lower similarity threshold = more hits and more wrong answers. RedisVL's `SemanticCache` expresses it as `distance_threshold` in Redis cosine distance units (0 to 2, lower = stricter; default 0.1 in redisvl 0.28).
- **Near misses.** Embeddings put "refund policy for annual plans" and "... monthly plans" very close, yet the answers differ. Numbers, negations, entity names, dates and units are where false hits come from.
- **Scope (namespace / filters).** Anything that changes the right answer must be part of the key or a filter: model ID, prompt/template version, tenant, user permissions, locale, retrieved document set version.
- **Invalidation.** TTL per entry, plus explicit purge of a namespace when the source data, prompt or model changes. Event-driven purge beats long TTLs for facts that change.
- **Metrics.** Hit rate, false-hit rate (sampled and judged), latency saved, cost saved, and the embedding call cost on every miss.

## When to use / scenarios
- High-volume, repetitive, mostly static questions: support FAQ bots, internal HR/IT helpdesks, product Q&A, documentation assistants ([[customer-support]]).
- Deterministic-ish tasks with canonical outputs: classification of recurring texts, translation of fixed UI strings, query rewriting.
- Expensive calls where a slightly generic answer is acceptable (suggestions, summaries of the same public page).
- Do NOT use (or use exact-match only) for: personalised answers, anything reading live data (balances, stock, order status), agent tool-calling turns with side effects, multi-turn chats where the answer depends on history, regulated advice where a wrong-but-similar answer is a compliance issue.
- If the repeated part is a long shared prompt prefix rather than the whole question, use [[prompt-caching-and-cost]] instead.

## Setup & code
Self-contained numpy demo: exact layer, semantic layer with namespace and TTL filters, explicit invalidation, and a threshold sweep that shows the hit-rate vs correctness trade-off. The toy hashed-word embedding stands in for a real model from [[embedding-models]].
```python
import hashlib, re, time
import numpy as np

DIM = 512

def embed(text: str) -> np.ndarray:
    """Toy embedding: hashed word unigrams + bigrams, L2-normalised. Swap for a real model."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    grams = words + [a + "_" + b for a, b in zip(words, words[1:])]
    v = np.zeros(DIM)
    for g in grams:
        h = int(hashlib.md5(g.encode()).hexdigest(), 16)
        v[h % DIM] += 1.0 if (h >> 9) & 1 else -1.0
    n = np.linalg.norm(v)
    return v / n if n else v

class SemanticCache:
    def __init__(self, threshold=0.85, ttl_s=3600.0):
        self.threshold, self.ttl_s = threshold, ttl_s
        self.exact = {}                                   # normalised text -> index
        self.vecs, self.rows = np.empty((0, DIM)), []     # rows: (namespace, prompt, answer, ts)

    @staticmethod
    def _norm(text):
        return " ".join(re.findall(r"[a-z0-9]+", text.lower()))

    def get(self, ns, prompt, now=None):
        now = time.time() if now is None else now
        i = self.exact.get((ns, self._norm(prompt)))      # 1) cheap exact layer
        if i is not None and now - self.rows[i][3] < self.ttl_s:
            return self.rows[i][2], 1.0, "exact"
        if not self.rows:
            return None, 0.0, "miss"
        sims = self.vecs @ embed(prompt)                  # 2) cosine similarity (unit vectors)
        ok = np.array([r[0] == ns and now - r[3] < self.ttl_s for r in self.rows])
        sims = np.where(ok, sims, -1.0)                   # namespace + TTL filter BEFORE ranking
        j = int(np.argmax(sims))
        if sims[j] >= self.threshold:
            return self.rows[j][2], float(sims[j]), "semantic"
        return None, float(sims[j]), "miss"

    def put(self, ns, prompt, answer, now=None):
        now = time.time() if now is None else now
        self.exact[(ns, self._norm(prompt))] = len(self.rows)
        self.vecs = np.vstack([self.vecs, embed(prompt)])
        self.rows.append((ns, prompt, answer, now))

    def invalidate(self, ns):
        """Drop a namespace, e.g. after the docs or the prompt template change."""
        keep = [i for i, r in enumerate(self.rows) if r[0] != ns]
        self.vecs, self.rows = self.vecs[keep], [self.rows[i] for i in keep]
        self.exact = {(r[0], self._norm(r[1])): i for i, r in enumerate(self.rows)}

# Namespace = everything that changes the right answer: model, prompt version, tenant, locale.
NS = "claude-haiku-4-5|prompt-v3|tenant-42|en"
cache = SemanticCache(threshold=0.6)
cache.put(NS, "How do I reset my password?", "Settings > Security > Reset password.")
cache.put(NS, "What is the refund policy for annual plans?", "Annual plans: pro-rata refund within 30 days.")

for q in ["how do i reset my password",            # exact after normalisation
          "How can I reset my password please?",   # paraphrase -> semantic hit
          "What is the refund policy for monthly plans?",  # near miss: WRONG answer served
          "Where is your office?"]:
    ans, score, kind = cache.get(NS, q)
    print(f"{kind:8s} {score:5.2f}  {q!r} -> {ans!r}")
print("other tenant:", cache.get("claude-haiku-4-5|prompt-v3|tenant-7|en", "How do I reset my password?")[2])
cache.invalidate(NS)
print("after invalidate:", cache.get(NS, "How do I reset my password?")[2])

# Hit rate vs correctness: sweep the threshold on labelled pairs (same answer or not).
pairs = [("how do i reset my password", "how can i reset my password", 1),
         ("how do i reset my password", "reset password steps", 1),
         ("cancel my subscription", "how do i cancel my subscription", 1),
         ("refund policy for annual plans", "refund policy for monthly plans", 0),
         ("reset my password", "reset my username", 0),
         ("price of the pro plan in usd", "price of the pro plan in eur", 0)]
sims = np.array([embed(a) @ embed(b) for a, b, _ in pairs]); same = np.array([p[2] for p in pairs])
for t in (0.3, 0.5, 0.7, 0.9):
    hit = sims >= t
    prec = f"{same[hit].mean():.2f}" if hit.any() else "n/a"
    print(f"threshold {t:.1f}: hit rate {hit.mean():.2f}, share of hits that are correct {prec}")
```
The demo deliberately serves a wrong answer for "monthly plans": that is the failure mode to measure before launch.

Production version with RedisVL (needs a Redis instance with the query engine and a vectorizer; API as in redisvl 0.28):
<!-- skip-check: needs a running Redis server and an embedding model download -->
```python
from redisvl.extensions.cache.llm import SemanticCache

cache = SemanticCache(
    name="support-faq-v3",              # one index per prompt/model version is a simple namespace
    redis_url="redis://localhost:6379",
    distance_threshold=0.1,              # Redis cosine distance, 0..2, lower = stricter
    ttl=24 * 3600,                       # seconds; per-entry override via store(..., ttl=)
)
hits = cache.check(prompt="How can I reset my password?")
if hits:
    answer = hits[0]["response"]
else:
    answer = call_llm("How can I reset my password?")   # your model call
    cache.store(prompt="How can I reset my password?", response=answer,
                metadata={"model": "claude-haiku-4-5", "prompt_version": "v3"})
```
`filterable_fields` plus `filter_expression` on `check()` restrict hits to a tenant or user group; check the RedisVL docs for the field schema.

## Choosing / trade-offs
| Decision | Pushes one way | Pushes the other way |
|---|---|---|
| Threshold strict vs loose | Wrong answers are costly (money, legal, medical) -> strict, or exact-only | Answers are generic and cheap to be slightly off -> looser |
| What to embed | Only the user question: high hit rate | Question + key context (e.g. product, plan): fewer false hits |
| TTL short vs long | Source data changes often | Static docs, versioned namespaces that are purged on change |
| Store | In-process dict/numpy: one replica, small | Redis/vector DB ([[vector-databases]]): shared across replicas, filters, eviction |
| Verify on hit | Cheap re-ranker or small LLM check "does this answer fit this question?" lowers false hits | Adds latency and cost to every hit |
| Cache LLM output vs retrieval output | Cache final answer: biggest saving | Cache retrieved chunks only: answer still fresh, smaller saving |

Tune the threshold on a labelled set of real query pairs from logs ([[llm-observability]]): for each candidate threshold, measure hit rate and the share of hits a judge or human marks correct ([[llm-evaluation]]). Pick the loosest threshold that keeps false hits below what the product can tolerate.

## Gotchas
- A cache hit bypasses guardrails, retrieval and logging unless you route hits through them too; log hits as their own span ([[opentelemetry-genai-tracing]]).
- Cross-tenant leaks: one shared index without a tenant filter can serve customer A's answer (with their data) to customer B. Never cache answers that contain user-specific data in a shared namespace ([[ai-security-privacy-compliance]]).
- Cache poisoning: a user who can get a bad answer stored (prompt injection, jailbreak) serves it to everyone. Only store answers that passed output checks; consider storing only answers with positive feedback.
- Changing the embedding model invalidates every stored vector; rebuild the index.
- Thresholds do not transfer between embedding models, or between similarity and distance conventions (similarity 0.9 is distance 0.1 only for cosine).
- Temperature > 0 answers are samples; caching one freezes it for everyone, which may be fine or may hide variability users expected.
- Measure the miss path: every miss now pays an embedding call plus a vector search before the LLM call.
- Streaming UIs: a cached answer arrives all at once; either stream it out in chunks or accept the different feel.

## Related
- [[prompt-caching-and-cost]] - provider prefix caching; complementary, not a replacement.
- [[cost-and-latency]] - where caching sits among other cost levers.
- [[embeddings]], [[embedding-models]] - choosing the model that defines "similar".
- [[vector-databases]] - the store behind a shared semantic cache.
- [[llm-gateways-and-routing]] - gateways that offer caching as a feature.
- [[distance-metrics-and-similarity]] - cosine similarity vs distance.
- [[llm-observability]], [[opentelemetry-genai-tracing]] - measuring hit rate and false hits.
- [[ab-testing-and-canary-for-models]] - rolling out a cache or a new threshold safely.
- [[permission-aware-rag-and-multi-tenancy]] - why cache keys need tenant and permission scope.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/system-design/caching-strategies.md - TTLs, invalidation and cache-aside in general.

## References
- RedisVL LLM cache guide: https://redis.io/docs/latest/develop/ai/redisvl/user_guide/how_to_guides/llmcache/
- RedisVL source (`redisvl.extensions.cache.llm.SemanticCache`): https://github.com/redis/redis-vl-python
- GPTCache (open-source semantic cache library): https://github.com/zilliztech/GPTCache
- Anthropic prompt caching (for contrast): https://platform.claude.com/docs/en/build-with-claude/prompt-caching
