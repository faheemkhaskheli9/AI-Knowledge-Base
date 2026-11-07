---
title: Agent memory lifecycle (importance, decay, context budget, retention, audit)
category: llm-apps
tags: [agent-memory, importance-scoring, decay, contradiction-detection, context-budget, retention, consolidation, audit-trail, multi-tenant, right-to-erasure]
use_cases:
  - "rank stored memories so stale or superseded facts stop surfacing"
  - "decide how many retrieved memories to inject per query within a token budget"
  - "memory store keeps growing and needs a retention, pruning and consolidation policy"
  - "multi-tenant agent memory for patients or customers that must never leak across owners"
  - "prove who accessed which memory and when for a HIPAA or GDPR audit, and honour deletion requests"
  - "diagnose why an agent forgets, misremembers or misses exact facts despite having memory"
status: draft
last_verified: 2026-11-07
sources:
  - https://learn.microsoft.com/en-us/training/modules/aaai-design-multi-agent-memory-azure-cosmos-db/
  - https://arxiv.org/abs/2304.03442
  - https://arxiv.org/abs/2307.03172
---

# Agent memory lifecycle (importance, decay, context budget, retention, audit)

## Summary
Storing memories is the easy half; [[agent-memory]] covers the tiers. This topic covers what happens to a memory after it is written: its importance decays, gets boosted by use, and drops when a newer memory contradicts it; only a budgeted, blended-score subset is injected per query; old detail is consolidated, archived and pruned on a schedule; and every access and deletion is isolated per owner and audited. Critical facts are exempt from every lossy step.

## Key concepts
- **Mutable importance.** Score each memory 1-10 at write time, then `current = base * (1 - rate) ** months + min(access_count * k, cap)`, clamped to the scale. Decay rate is domain-tuned (preferences decay fast, diagnoses slowly). Cap the access boost, or a memory popular once stays important forever.
- **Contradiction suppresses, never deletes.** A new memory of the same type and owner, very similar (cosine ~0.85+) and newer than an old one, supersedes it: drop the old one to the floor and record `contradicted_by`. The record stays for audit.
- **Critical tag.** Hard constraints (allergies, confirmed refusals, legal holds) carry an explicit `critical: true` and skip decay, pruning and consolidation. A big boost is not the same guarantee; decay math can still erode it.
- **Blended ranking.** `score = w1*similarity + w2*importance + w3*recency` (the relevance/importance/recency retrieval of Park et al., Generative Agents). Weight importance up for safety domains, recency up for preference tracking.
- **Encoding policy.** After each turn, an extraction pass asks "what new information about this entity was revealed?" (not "what should be remembered?"), returns 0-3 observations with a type and importance. Skip acknowledgements and routine process text.
- **Context budget by query class.** Classify the query (simple factual / history-dependent / complex) and look up a token budget per content type (system, memories, documents, history, tool results, response reserve). If the total overflows, scale all allocations down proportionally instead of zeroing one.
- **Hybrid selection pool.** Fill ~70% of slots by blended rank and ~30% with the most recent items not already picked, so what just happened is always visible; then re-sort chronologically.
- **Retention tiers by owner activity, not record age.** Active owner: keep everything. Semi-active: consolidate low-importance episodes, keep high-importance ones. Inactive: archive to cold storage. Past the retention limit: flag for (manually approved) deletion.
- **Consolidation is deliberately lossy.** Cluster an owner's episodic memories by similarity, and only for clusters of 3+ let an LLM extract what stays constant into one semantic memory that lists the source ids it replaced.
- **Isolation in two layers.** Storage partition key = owner id, and a session-bound owner id (from the auth context, never a request parameter) checked by one service wrapper on every operation; reject on mismatch, never silently re-scope.
- **Audit is separate infrastructure.** Log timestamp, actor, owner, operation (read/write/delete), record ids and why, in its own store with a longer retention than the memories (investigations lag years).
- **Erasure is cascade + verify.** Delete across every tier and every derived embedding/index, then run a verification query. "Issued deletes" and "verified empty" are two required steps.

## When to use / scenarios
- Long-lived assistants and support agents with months of memories ([[personal-assistants]], [[customer-support]]).
- Regulated memory (clinical, financial) needing bounded retention, audit and erasure ([[healthcare]], [[ai-security-privacy-compliance]]).
- Retrieval returns semantically similar but outdated facts, or context cost climbs while quality plateaus.
- Not needed: single-session agents, or a few dozen user facts that fit in the prompt every time ([[long-context]]).

## Setup & code
Decay, contradiction suppression and hybrid budgeted selection, stdlib only:
```python
import math, time
from dataclasses import dataclass, field

MONTH = 30 * 86400

@dataclass
class Memory:
    id: str; owner: str; kind: str; text: str; vec: list; base: float
    ts: float; access_count: int = 0; critical: bool = False
    contradicted_by: str | None = None; tokens: int = 40

def cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / (math.hypot(*a) * math.hypot(*b))

def importance(m: Memory, now: float, rate=0.05, k=0.5, cap=2.0) -> float:
    if m.contradicted_by: return 1.0                      # floor, kept for audit
    if m.critical: return 10.0                            # exempt from decay
    months = (now - m.ts) / MONTH
    return max(1.0, min(10.0, m.base * (1 - rate) ** months + min(m.access_count * k, cap)))

def suppress_contradicted(new: Memory, store: list[Memory], threshold=0.85):
    for old in store:
        if (old.owner == new.owner and old.kind == new.kind and old.ts < new.ts
                and not old.critical and cos(old.vec, new.vec) >= threshold):
            old.contradicted_by = new.id

def select(query_vec, store, owner, budget_tokens, now, w=(0.5, 0.3, 0.2), recent_share=0.3):
    # scope to the session's owner; superseded memories stay stored but are never injected
    mine = [m for m in store if m.owner == owner and not m.contradicted_by]
    def score(m):
        recency = math.exp(-(now - m.ts) / MONTH)
        return w[0] * cos(query_vec, m.vec) + w[1] * importance(m, now) / 10 + w[2] * recency
    avg = sum(m.tokens for m in mine) / max(len(mine), 1)
    n = int(budget_tokens // avg) if mine else 0
    n_ranked = n - int(n * recent_share)
    ranked = sorted(mine, key=score, reverse=True)[:n_ranked]
    rest = sorted((m for m in mine if m not in ranked), key=lambda m: m.ts, reverse=True)
    picked = ranked + rest[: n - n_ranked]
    for m in picked: m.access_count += 1
    return sorted(picked, key=lambda m: m.ts)             # chronological for the prompt

now = time.time()
jan = Memory("m1", "p1", "med", "tolerates metformin well", [1, 0, 0.1], 6, now - 3 * MONTH)
mar = Memory("m2", "p1", "med", "GI distress on metformin, switched formulation", [1, 0, 0.2], 7, now)
allergy = Memory("m3", "p1", "allergy", "penicillin allergy", [0, 1, 0], 9, now - 40 * MONTH, critical=True)
store = [jan, allergy]
suppress_contradicted(mar, store); store.append(mar)
assert jan.contradicted_by == "m2" and importance(jan, now) == 1.0
assert importance(allergy, now) == 10.0                   # old but critical: no decay
picked = select([1, 0, 0.2], store, "p1", budget_tokens=80, now=now)
assert mar in picked and allergy in picked and jan not in picked
```
Measure the injection size instead of guessing it: replay a labelled query set at 0/3/5/10 injected memories and record accuracy, tokens and latency per size; pick the knee of the curve, not the largest size that fits.

## Choosing / trade-offs
- **Decay rate**: fast keeps memory current but forgets stable facts; validate against how often facts in your domain actually go stale.
- **Weights**: importance-heavy for safety/compliance, recency-heavy for preferences, similarity-heavy for open Q&A. Tune on the eval curve.
- **Compress before injecting**: summarizing or distilling (strip hedging, keep the factual core) cuts tokens per fact; keep the full record in the store for audit and exact lookup.
- **Consolidate vs keep**: consolidation lowers storage and retrieval noise but loses per-event detail; never consolidate critical or audit-relevant episodes.
- **Batch vs inline lifecycle**: pruning and consolidation touch data at rest; run them as scheduled jobs (weekly/monthly), not on the request path.
- **Auto vs approved deletion**: compliance-sensitive domains should require a human approval step for retention-expired deletes.

## Gotchas
Name the failure before fixing it; each needs a different tool:

| Failure | Symptom | Fix |
|---|---|---|
| Sliding-window amnesia | Early facts vanish as a fixed window advances | Pin critical facts outside the window |
| Summary drift | Re-summarizing a summary compounds errors | Validate consolidation against sources; prefer raw recent episodes |
| Vector-only recall | Misses exact ids, names, doses that embed close to other values | Hybrid (vector + keyword) search plus reranking ([[advanced-rag]]) |
| Entity continuity loss | Memories attach to the wrong person across sessions | Tag every memory with a stable entity id |

- Ranking by raw similarity alone surfaces confident but stale memories and excludes recent ones worded differently.
- Pruning by age alone discards old but still-critical facts; check the critical tag before any lifecycle action, not the score.
- Deleting contradicted memories loses the record of what was believed and when.
- Audit logs pruned on the data's schedule cannot prove anything in a delayed investigation; log reads as well as writes.
- Erasure that skips derived embeddings or indexes leaves the data recoverable by similarity search.
- Isolation only in application code: one missed `owner_id` filter is a cross-tenant leak.
- Consolidating clusters of one or two memories turns a coincidence into a "pattern".

## Related
- [[agent-memory]] - the tiers this lifecycle runs over.
- [[vector-databases]] - semantic tier storage, filtering and deletion.
- [[advanced-rag]] - hybrid search and reranking for exact-match recall.
- [[long-context]] - why more injected context is not automatically better.
- [[ai-security-privacy-compliance]] - retention, DPAs and erasure obligations.
- [[pii-redaction-placeholders]] - keeping identifiers out of what is sent to the model.

## References
- Microsoft Learn, Design multi-agent memory architectures with Azure Cosmos DB: https://learn.microsoft.com/en-us/training/modules/aaai-design-multi-agent-memory-azure-cosmos-db/
- Park et al., Generative Agents (importance, recency, relevance retrieval): https://arxiv.org/abs/2304.03442
- Liu et al., Lost in the Middle: https://arxiv.org/abs/2307.03172
