---
title: RAG evaluation (retrieval metrics, faithfulness, context precision and recall, Ragas)
category: llm-apps
tags: [rag, evaluation, recall-at-k, mrr, ndcg, faithfulness, groundedness, context-precision, context-recall, answer-relevancy, ragas, golden-set]
use_cases:
  - "measure whether our RAG chatbot's wrong answers come from bad retrieval or bad generation"
  - "compare two chunking or embedding choices on retrieval recall before shipping one"
  - "score RAG answers for faithfulness to the retrieved documents in CI"
  - "build a labelled question set for evaluating a RAG system over company documents"
status: draft
last_verified: 2026-10-24
sources:
  - https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/
  - https://arxiv.org/abs/2309.15217
  - https://arxiv.org/abs/2311.09476
  - https://arxiv.org/abs/2104.08663
  - https://doi.org/10.1145/582415.582418
---

# RAG evaluation (retrieval metrics, faithfulness, context precision and recall, Ragas)

## Summary
A RAG answer can be wrong for two separate reasons: the retriever did not return the passage that holds the answer, or the generator had the passage and still answered badly (ignored it, contradicted it, or added unsupported claims). RAG evaluation scores the two stages separately: retrieval with ranking metrics against labelled relevant passages (recall@k, MRR, nDCG, context precision), and generation with claim-level checks against the retrieved context (faithfulness) and against the question (answer relevancy) or a reference answer (correctness). Splitting the score tells you which component to fix. Frameworks such as Ragas compute the LLM-judged metrics; the ranking metrics are a few lines of code and need no model at all.

## Key concepts
- **Golden set**: questions with the IDs of the passages (or documents) that answer them and, ideally, a reference answer. A modest set drawn from real user queries is worth more for decisions than a large synthetic one; synthetic questions generated from your documents are useful for coverage, but check a sample by hand ([[llm-evaluation]]).
- **Retrieval metrics** (need relevance labels, no LLM):
  - *Recall@k*: share of a question's relevant passages that appear in the top k. The main number for RAG, because the generator only sees the top k.
  - *Hit rate@k*: share of questions with at least one relevant passage in the top k.
  - *MRR*: mean of 1/rank of the first relevant passage; rewards putting it first.
  - *nDCG@k*: discounted cumulative gain normalized by the ideal ranking; supports graded relevance (Jarvelin and Kekalainen; used by BEIR).
  - *Context precision* (Ragas): mean of precision@i over the ranks i that hold a relevant chunk, divided by the number of relevant chunks in the top K; it rewards relevant chunks ranked above irrelevant ones.
- **Generation metrics** (need an LLM or NLI judge):
  - *Faithfulness / groundedness*: split the answer into claims, check each against the retrieved context; score = supported claims / all claims (Ragas definition). Catches hallucination even when the answer happens to be right.
  - *Context recall* (Ragas): split the *reference* answer into claims and check how many the retrieved context supports; a label-light proxy for retrieval recall when you have reference answers but no passage IDs.
  - *Answer relevancy* (Ragas): generate questions from the answer, embed them, average cosine similarity to the real question; penalizes evasive or padded answers, not factual errors.
  - *Answer correctness*: compare to the reference answer (exact match for short facts, an LLM judge with a rubric otherwise; [[llm-as-judge-and-eval-driven-development]]).
- **Diagnosis grid**: low recall -> fix chunking, embeddings, hybrid search or reranking ([[advanced-rag]]); good recall but low faithfulness -> fix the prompt, citation instructions or model; good both but low correctness -> labels, question ambiguity or missing documents.
- **Paired comparison**: compare two pipeline variants on the same questions and bootstrap the per-question difference; small golden sets give wide intervals ([[statistics-and-ab-testing]]).

## When to use / scenarios
- Before choosing chunk size, embedding model, hybrid search or a reranker: run the retrieval metrics for each option on the same golden set ([[chunking-strategies]], [[embedding-models]], [[rerankers]]).
- A support or policy chatbot gives confident wrong answers: faithfulness tells you whether it is inventing or the index lacks the passage ([[hallucination-and-grounding]]).
- CI gate on prompt, model or index changes: retrieval metrics are cheap and deterministic, so run them on every change; run the LLM-judged metrics on a sample or nightly.
- Production monitoring: sample live traces, score faithfulness without references, alert on drops ([[llm-observability]]).
- Not a replacement for end-to-end user outcomes (resolution rate, escalations); use those for the final A/B decision ([[ab-testing-and-canary-for-models]]).
- For agentic RAG with many tool calls, score the trajectory as well ([[agent-evaluation]]).

## Setup & code
Runnable, stdlib only: retrieval metrics on a labelled set, Ragas-style context precision, claim-level faithfulness with a pluggable judge (a word-overlap stand-in here, an LLM in production), a stage diagnosis per question and a paired bootstrap between two retrievers.
```python
import math
import random
import re

# Golden set: question -> relevant chunk IDs (graded: 2 = answers it, 1 = partly relevant).
golden = {
    "q1": {"refund-policy#2": 2, "refund-policy#3": 1},
    "q2": {"shipping#1": 2},
    "q3": {"warranty#4": 2, "warranty#5": 2},
    "q4": {"returns#1": 2},
}
# Ranked chunk IDs returned by two retriever variants for each question.
runs = {
    "dense": {"q1": ["faq#9", "refund-policy#2", "refund-policy#3", "shipping#1"],
              "q2": ["shipping#2", "shipping#3", "faq#1", "shipping#1"],
              "q3": ["warranty#4", "faq#2", "warranty#7", "warranty#8"],
              "q4": ["faq#5", "faq#6", "faq#7", "faq#8"]},
    "hybrid+rerank": {"q1": ["refund-policy#2", "refund-policy#3", "faq#9", "faq#1"],
                      "q2": ["shipping#1", "shipping#2", "faq#1", "faq#3"],
                      "q3": ["warranty#4", "warranty#5", "faq#2", "faq#3"],
                      "q4": ["faq#5", "returns#1", "faq#7", "faq#8"]},
}

def recall_at_k(ranked, rel, k):
    return len(set(ranked[:k]) & set(rel)) / len(rel)

def hit_at_k(ranked, rel, k):
    return float(any(c in rel for c in ranked[:k]))

def reciprocal_rank(ranked, rel):
    return next((1 / (i + 1) for i, c in enumerate(ranked) if c in rel), 0.0)

def ndcg_at_k(ranked, rel, k):
    dcg = sum((2 ** rel.get(c, 0) - 1) / math.log2(i + 2) for i, c in enumerate(ranked[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0

def context_precision(ranked, rel, k):
    """Ragas definition: sum of precision@i * v_i over the top K, / relevant items in the top K."""
    hits, total = 0, 0.0
    for i, c in enumerate(ranked[:k], start=1):
        if c in rel:
            hits += 1
            total += hits / i
    return total / hits if hits else 0.0

K = 3
per_question = {}
for name, run in runs.items():
    rows = {q: dict(recall=recall_at_k(run[q], rel, K), hit=hit_at_k(run[q], rel, K),
                    rr=reciprocal_rank(run[q], rel), ndcg=ndcg_at_k(run[q], rel, K),
                    ctx_p=context_precision(run[q], rel, K)) for q, rel in golden.items()}
    per_question[name] = rows
    means = {m: sum(r[m] for r in rows.values()) / len(rows) for m in next(iter(rows.values()))}
    print(f"{name:14s} " + "  ".join(f"{m}@{K}={v:.2f}" if m != "rr" else f"MRR={v:.2f}"
                                     for m, v in means.items()))

# Faithfulness: split the answer into claims, ask a judge whether the context supports each one.
STOP = {"the", "a", "an", "of", "to", "is", "are", "in", "within", "for", "and", "you", "your", "can", "be", "it", "on"}

def claims(answer):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]

def overlap_judge(claim, context):
    """Stand-in judge: every content word of the claim appears in the context. Use an LLM or NLI model in production."""
    words = set(re.findall(r"[a-z0-9]+", claim.lower())) - STOP
    ctx = set(re.findall(r"[a-z0-9]+", context.lower()))
    return words <= ctx

def faithfulness(answer, contexts, judge=overlap_judge):
    cs = claims(answer)
    supported = [judge(c, " ".join(contexts)) for c in cs]
    return sum(supported) / len(cs), [c for c, ok in zip(cs, supported) if not ok]

contexts = ["Refunds are issued within 14 days of receiving the returned item.",
            "Refunds go back to the original payment method."]
answer = ("Refunds are issued within 14 days of receiving the returned item. "
          "Refunds go back to the original payment method. Shipping costs are refunded too.")
score, unsupported = faithfulness(answer, contexts)
print(f"faithfulness={score:.2f} unsupported={unsupported}")

# Stage diagnosis: which component to fix for each question.
def diagnose(recall, faith, correct):
    if recall == 0:
        return "retrieval miss: fix chunking/embeddings/hybrid/rerank"
    if faith < 1.0:
        return "generation adds unsupported claims: fix prompt/model"
    return "ok" if correct else "context fine, answer wrong: check labels or question"

print(diagnose(per_question["dense"]["q4"]["recall"], 1.0, False))
print(diagnose(1.0, score, True))

# Paired bootstrap over questions: is hybrid+rerank really better on recall@K?
def paired_bootstrap(a, b, n=5000, seed=0):
    rnd = random.Random(seed)
    diffs = [y - x for x, y in zip(a, b)]
    stats = sorted(sum(rnd.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    return sum(diffs) / len(diffs), stats[int(0.025 * n)], stats[int(0.975 * n)]

qs = sorted(golden)
mean, lo, hi = paired_bootstrap([per_question["dense"][q]["recall"] for q in qs],
                                [per_question["hybrid+rerank"][q]["recall"] for q in qs])
print(f"recall@{K} gain {mean:+.2f}, 95% CI [{lo:+.2f}, {hi:+.2f}] on {len(qs)} questions "
      "(a 4-question set is far too small to decide)")
```
With Ragas, the LLM-judged metrics take the same fields (`user_input`, `response`, `retrieved_contexts`, `reference`) and an LLM wrapper; the metric classes and the `llm_factory` helper are from its current docs, check them before pinning a version:
<!-- skip-check: needs ragas, an LLM API key and network -->
```python
import asyncio
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRecall, Faithfulness

JUDGE_MODEL = "<judge-model-id>"          # copy a current ID from models/openai-gpt.md
llm = llm_factory(JUDGE_MODEL, client=AsyncOpenAI())

async def main():
    sample = dict(user_input="How long do refunds take?",
                  response="Refunds are issued within 14 days. Shipping is refunded too.",
                  retrieved_contexts=["Refunds are issued within 14 days of receiving the returned item."])
    print((await Faithfulness(llm=llm).ascore(**sample)).value)
    print((await ContextRecall(llm=llm).ascore(
        user_input=sample["user_input"], retrieved_contexts=sample["retrieved_contexts"],
        reference="Refunds are issued within 14 days of the return arriving.")).value)

asyncio.run(main())
```

## Choosing / trade-offs
| Metric | Needs | Cost | Use it for |
|---|---|---|---|
| Recall@k, hit rate, MRR, nDCG | relevant-passage labels | free, deterministic | every retrieval change, CI |
| Context precision | relevance labels or an LLM judge per chunk | free with labels | rerankers and ordering |
| Context recall (Ragas) | reference answers, LLM judge | one judge call per sample | retrieval check when passage labels are missing |
| Faithfulness | LLM or NLI judge, no reference | claim extraction + checks | hallucination, prompt and model changes, production sampling |
| Answer relevancy | LLM + embeddings | several calls | evasive or padded answers |
| Answer correctness | reference answer + judge | one judge call | release decisions |

- **Passage labels vs reference answers**: passage IDs make retrieval metrics exact but break when you re-chunk (IDs change); label at document or span level and map spans to chunks, or use reference answers with context recall.
- **Which k**: evaluate at the k the generator actually receives after reranking, plus a larger k for the first-stage retriever (recall@50 before reranking to top 5).
- **Judge choice**: a strong judge model is more accurate and costs more; calibrate any judge against a few dozen human labels before trusting its scores ([[llm-as-judge-and-eval-driven-development]]).
- **Framework vs own code**: Ragas, ARES-style trained judges, or eval platforms save work for LLM-judged metrics; keep ranking metrics in your own code so they stay deterministic and version-free.

## Gotchas
- A high answer-correctness score with low faithfulness means the model answered from its own memory; it will fail on documents newer than its training data.
- Retrieval labels go stale when documents change; tie each label to a document version or re-check after reindexing.
- Re-chunking changes chunk IDs, so recall from before and after a chunking change is not comparable unless relevance is labelled at document or span level.
- Duplicated or near-duplicate chunks inflate recall@k without adding information; deduplicate or use MMR ([[mmr-diversity-reranking-from-scratch]]).
- Questions generated by an LLM from a single chunk are easy for retrieval (they reuse the chunk's words); mix in real user queries with paraphrases, typos and multi-hop questions.
- LLM-judged scores move when the judge model or its prompt changes; pin both, store them with results, and rerun the baseline when you change them.
- Averages hide failure clusters; slice by question type, product or document source.
- Answer relevancy can be high for a fluent wrong answer; never use it alone.

## Related
- [[rag-basics]] - the pipeline these metrics measure.
- [[advanced-rag]] - hybrid search, reranking and query rewriting to fix low recall.
- [[llm-evaluation]] - golden sets, scoring ladder and eval harnesses in general.
- [[llm-as-judge-and-eval-driven-development]] - calibrating the judge behind faithfulness and correctness.
- [[hallucination-and-grounding]] - why faithfulness fails and how citations help.
- [[chunking-strategies]] - the choice most often compared with recall@k.
- [[rerankers]] - improving context precision and MRR.
- [[reciprocal-rank-fusion-from-scratch]] - hybrid retrieval variant to compare.
- [[statistics-and-ab-testing]] - confidence intervals for small golden sets.
- [[llm-observability]] - sampling production traces for online faithfulness scoring.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/testing/testing-strategy-and-pyramid.md - where eval suites fit in a test strategy.

## References
- Ragas metrics (faithfulness, context precision, context recall, answer relevancy definitions and APIs): https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/
- Es et al., RAGAS: Automated Evaluation of Retrieval Augmented Generation: https://arxiv.org/abs/2309.15217
- Saad-Falcon et al., ARES: An Automated Evaluation Framework for RAG Systems: https://arxiv.org/abs/2311.09476
- Thakur et al., BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models (nDCG@10 as the headline metric): https://arxiv.org/abs/2104.08663
- Jarvelin and Kekalainen, Cumulated gain-based evaluation of IR techniques, ACM TOIS 2002 (nDCG): https://doi.org/10.1145/582415.582418
