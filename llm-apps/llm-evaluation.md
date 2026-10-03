---
title: LLM evaluation (evals, LLM-as-judge, golden sets)
category: llm-apps
tags: [evaluation, evals, llm-as-judge, golden-set, regression, rag-eval, agent-eval]
use_cases:
  - "measure whether a prompt change improved our support bot"
  - "build a golden test set for a RAG system over legal documents"
  - "score free-text answers automatically with an LLM judge in CI"
  - "compare two models for a classification or extraction task before switching"
  - "evaluate an agent's tool trajectory, not just its final answer"
status: draft
last_verified: 2026-10-03
sources:
  - https://platform.claude.com/docs/en/test-and-evaluate/develop-tests
  - https://platform.openai.com/docs/guides/evals
  - https://arxiv.org/abs/2306.05685
---

# LLM evaluation

## Summary
Evals are repeatable tests that tell you whether an LLM feature works and whether a change made it better or worse. Build a small golden set from real cases, score with the cheapest reliable method (exact match, code, then LLM judge, then humans), and run it on every prompt, model or retrieval change. No evals means every change is a guess.

## Key concepts
- **Golden set**: 30-200 representative cases (inputs + expected output or criteria), including edge cases, adversarial inputs and "should refuse / no answer" cases. Source from production logs; version it; keep a held-out portion.
- **Scoring ladder**: (1) deterministic - exact match, regex, JSON schema validity, code/unit tests, SQL result equality; (2) heuristic - retrieval recall@k, citation overlap; (3) **LLM-as-judge** with a rubric; (4) human review for calibration.
- **LLM-as-judge**: a model scores output against explicit per-case criteria, returning `{passed, score, rationale, criteria_met, criteria_missed}`. Validate the returned shape strictly; malformed or out-of-range scores are errors, not silently coerced. Pass criteria per call, thresholds as config.
- **Judge biases** (Zheng et al., arXiv 2306.05685): position bias, verbosity bias, self-preference. Mitigate by swapping order in pairwise tests, using a different/stronger judge model, and calibrating against human labels.
- **Component vs end-to-end**: RAG: retrieval (recall@k, MRR) and generation (faithfulness/groundedness, answer relevance, correctness) separately ([[rag-basics]]). Agents: task success, tool-call correctness, steps, cost ([[agents]]).
- **Online evals**: sample production traffic, user feedback, shadow tests ([[llm-observability]]).
- **Regression gate in CI**: fail the build if score drops past a threshold.

## When to use / scenarios
- Before launching any LLM feature, and on each prompt/model/chunking/retriever change.
- Model migration: replay the golden set on candidate models and compare quality, latency, cost ([[model-selection]]).
- Safety: red-team sets for [[prompt-injection]] and [[guardrails-and-safety]].
- Not a substitute for human spot checks on subjective quality.

## Setup & code
Minimal harness with deterministic checks plus an LLM judge:
```python
# pip install anthropic pydantic
import json, anthropic
from pydantic import BaseModel
client = anthropic.Anthropic()
MODEL = "claude-sonnet-5-5"

class Verdict(BaseModel):
    score: int            # 1-5
    rationale: str
    criteria_missed: list[str]

def judge(question: str, answer: str, criteria: list[str]) -> Verdict:
    r = client.messages.parse(
        model=MODEL, max_tokens=500, output_format=Verdict,
        system="You are a strict grader. Score 1-5 against ALL criteria. Do not reward length.",
        messages=[{"role": "user", "content":
            f"<question>{question}</question>\n<answer>{answer}</answer>\n<criteria>\n" + "\n".join(criteria) + "\n</criteria>"}])
    v = r.parsed_output
    if not 1 <= v.score <= 5:
        raise ValueError("judge returned out-of-range score")
    return v

GOLDEN = [{"q": "Refund window?", "must_contain": ["30 days"], "criteria": ["States 30-day window", "No invented exceptions"]}]

def run(system_under_test, pass_threshold=4):
    results = []
    for c in GOLDEN:
        a = system_under_test(c["q"])
        det = all(s in a for s in c["must_contain"])          # cheap deterministic gate first
        v = judge(c["q"], a, c["criteria"]) if det else None
        results.append({"q": c["q"], "det": det, "score": v.score if v else 0})
    passed = sum(r["det"] and r["score"] >= pass_threshold for r in results)
    print(json.dumps(results, indent=1), f"\npass rate {passed}/{len(results)}")
    return passed / len(results)
```
Frameworks: Promptfoo, Inspect, DeepEval, Ragas, OpenAI Evals, LangSmith/Braintrust/Langfuse datasets; use them once the in-house harness gets painful.

## Choosing / trade-offs
- **Deterministic vs judge**: deterministic is cheap and unambiguous; use a judge only for what code cannot check.
- **Judge strength**: a stronger or different-family judge is more reliable but costs more; sample or batch.
- **Size vs cost**: start with 30-50 cases; grow when a failure appears in production (add it as a case).
- **Synthetic vs real data**: LLM-generated cases scale but skew easy; mix with real queries.
- **Binary vs scalar scores**: pass/fail per criterion is more stable than 1-10 scores.

## Gotchas
- Judging with the same model that wrote the answer inflates scores.
- A constant or keyword-matching fake judge in tests must be content-dependent so both pass and fail paths run.
- Overfitting prompts to a small golden set; keep a held-out slice.
- Non-determinism: run multiple samples or set low temperature; report variance, not one number.
- Empty/degenerate input should return "nothing to score" (None), not a fake fail.
- Data leakage: golden answers inside the prompt or RAG corpus.
- Metrics without user context (e.g. fluency) rarely track business outcomes; tie to task success.

## Related
- [[rag-basics]] - retrieval and faithfulness metrics.
- [[agents]] - trajectory evaluation.
- [[guardrails-and-safety]] - safety test sets.
- [[prompt-engineering]] - what evals iterate.
- [[llm-observability]] - production monitoring.
- [[hallucination-and-grounding]] - groundedness checks.

## References
- Anthropic, Define success and build evaluations: https://platform.claude.com/docs/en/test-and-evaluate/develop-tests
- OpenAI Evals guide: https://platform.openai.com/docs/guides/evals
- Zheng et al., Judging LLM-as-a-Judge: https://arxiv.org/abs/2306.05685
