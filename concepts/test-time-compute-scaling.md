---
title: Test-time compute scaling
category: concepts
tags: [test-time-compute, inference-scaling, best-of-n, self-consistency, verifiers, reward-models, tree-search, thinking-budget, effort]
use_cases:
  - "raise accuracy on hard maths, coding or planning questions by spending more inference compute"
  - "decide between sampling many answers, a bigger model, or a higher thinking effort"
  - "use unit tests or a verifier to pick the best of several generated solutions"
  - "set a per-request compute budget that adapts to question difficulty"
  - "estimate how many samples are worth paying for before accuracy plateaus"
status: draft
last_verified: 2026-10-15
sources:
  - https://arxiv.org/abs/2408.03314
  - https://arxiv.org/abs/2203.11171
  - https://arxiv.org/abs/2110.14168
  - https://arxiv.org/abs/2305.20050
  - https://arxiv.org/abs/2407.21787
  - https://arxiv.org/abs/2305.10601
  - https://arxiv.org/abs/2501.19393
  - https://platform.claude.com/docs/en/build-with-claude/effort
---

# Test-time compute scaling

## Summary
Test-time compute scaling means buying accuracy by spending more computation per query at inference instead of training a bigger model. The main levers are sequential (longer reasoning, controlled by a thinking budget or effort level), parallel (sample N answers and pick one by majority vote or a verifier), and search (expand partial solutions guided by a reward model). It helps most on problems with checkable answers and moderate difficulty, and returns diminish quickly once the model rarely finds the right answer at all or the selector cannot recognise it.

## Key concepts
- **Two ways to spend compute**: *sequential* (one longer chain of thought, revisions) and *parallel* (independent samples). Snell et al. found the best split depends on question difficulty: revisions suit easier questions, search/parallel suits harder ones, and a difficulty-adaptive ("compute-optimal") allocation beat best-of-N at equal compute.
- **Coverage vs selection**: with N samples, *coverage* (pass@N: at least one sample is correct) keeps rising with N (Brown et al., "Large Language Monkeys"), but the gain you get depends on picking the right one. With an exact verifier (unit tests, proof checker) you get coverage; with majority vote or a learned reward model you get much less.
- **Self-consistency**: sample several reasoning paths, take the most common final answer (Wang et al.). Needs a canonical, comparable answer (a number, a label); plateaus once the plurality answer stabilises ([[self-consistency-voting-from-scratch]]).
- **Best-of-N with a verifier**: score each sample with an outcome reward model (ORM, scores the final answer; Cobbe et al.) or a process reward model (PRM, scores each step; Lightman et al. "Let's verify step by step"). An imperfect verifier can be "reward-hacked" as N grows: it picks confident wrong answers.
- **Search**: beam search or tree search over reasoning steps guided by a PRM ([[mcts-from-scratch]]); Tree of Thoughts lets the model propose and evaluate partial thoughts.
- **Thinking budgets**: reasoning models expose a dial (Claude `output_config={"effort": ...}` with levels `low`/`medium`/`high`/`xhigh`/`max` on current models; budget tokens on older ones). Effort is a behavioral signal, not a strict budget ([[reasoning-models]]). "Budget forcing" (s1) shows the same idea for open models: cut thinking short or append "Wait" to extend it.
- **Cost**: parallel sampling multiplies output tokens by N; thinking tokens are billed as output. Price it before shipping ([[llm-cost-calculator]]).

## When to use / scenarios
- Code generation with tests: generate N candidates, run tests in a sandbox, return the first that passes ([[sandboxed-code-execution]], [[coding-agents]]).
- Maths, quantitative finance checks, unit conversions: self-consistency over numeric answers.
- Agent planning steps where an error cascades: raise effort on the planning call only ([[agents]]).
- Offline data generation and labelling: spend compute once to produce better training data ([[synthetic-data-for-llm-training]]).
- Uncertainty signal: low agreement among samples flags answers for human review ([[hallucination-and-grounding]]).
- Not when: latency-critical UIs, open-ended writing with no way to compare answers, or tasks where the model is near 0% or near 100% per-sample accuracy (nothing to gain). A stronger model or better retrieval often beats more samples from a weak one.

## Setup & code
A numpy simulation of the three selection rules on synthetic "questions" of varying difficulty. Each question has a per-sample success probability `p`; wrong samples fall on a few distractor answers; the reward model is a noisy scorer. It shows coverage rising with N, majority vote plateauing, and a noisy verifier sitting in between.

```python
import numpy as np

rng = np.random.default_rng(0)
Q = 2000                                        # questions
p = rng.beta(1.2, 1.8, size=Q)                  # per-sample chance of the correct answer (difficulty mix)
N_MAX, K_WRONG = 64, 4                          # samples per question, number of distractor answers

# answer 0 = correct, 1..K_WRONG = distractors (some distractors are more attractive than others)
distractor_w = rng.dirichlet(np.ones(K_WRONG) * 0.7, size=Q)
correct = rng.random((Q, N_MAX)) < p[:, None]
wrong_ans = np.array([rng.choice(np.arange(1, K_WRONG + 1), size=N_MAX, p=w) for w in distractor_w])
answers = np.where(correct, 0, wrong_ans)

# noisy outcome reward model: correct answers score higher on average, with overlap
rm_score = rng.normal(loc=np.where(answers == 0, 1.0, 0.0), scale=1.0)

def evaluate(n):
    a, s = answers[:, :n], rm_score[:, :n]
    coverage = (a == 0).any(axis=1).mean()                         # oracle verifier (tests pass)
    counts = np.stack([(a == k).sum(axis=1) for k in range(K_WRONG + 1)], axis=1)
    tie_break = rng.random(counts.shape) * 0.1                     # random tie-break, no bias to correct
    majority = ((counts + tie_break).argmax(axis=1) == 0).mean()   # self-consistency
    best_rm = (a[np.arange(Q), s.argmax(axis=1)] == 0).mean()      # best-of-n by reward model
    return coverage, majority, best_rm

print(" n   oracle(pass@n)  majority  best-of-n(RM)")
for n in (1, 2, 4, 8, 16, 32, 64):
    cov, maj, rm = evaluate(n)
    print(f"{n:2d}   {cov:12.3f}  {maj:8.3f}  {rm:13.3f}")

# Where does extra compute help? Gain from n=1 to n=16 by difficulty bucket
buckets = np.digitize(p, [0.1, 0.3, 0.6, 0.9])
names = ["p<0.1", "0.1-0.3", "0.3-0.6", "0.6-0.9", ">0.9"]
single = (answers[:, 0] == 0)
a16 = answers[:, :16]
c16 = np.stack([(a16 == k).sum(1) for k in range(K_WRONG + 1)], 1)
maj16 = (c16 + rng.random(c16.shape) * 0.1).argmax(1) == 0
for b, name in enumerate(names):
    m = buckets == b
    if m.any():
        print(f"{name:8s} n={m.sum():4d}  single={single[m].mean():.2f}  majority@16={maj16[m].mean():.2f}")
```

Typical reading of the output: majority vote helps most for mid-difficulty questions (per-sample accuracy above the most attractive distractor) and can hurt when a distractor is more likely than the correct answer; the oracle column is the ceiling a perfect verifier would reach.

Parallel sampling plus an exact verifier against the Claude API. Current Claude models reject `temperature`, so diversity comes from independent calls (and optionally varied prompts); effort sets the per-call thinking depth. Model ID from [[anthropic-claude]].

<!-- skip-check: needs ANTHROPIC_API_KEY and network -->
```python
import anthropic, re

client = anthropic.Anthropic()

def solve(question, effort="medium"):
    r = client.messages.create(
        model="claude-sonnet-5-5", max_tokens=16000,
        output_config={"effort": effort},
        messages=[{"role": "user", "content": question + "\nEnd with 'ANSWER: <integer>'."}])
    text = "".join(b.text for b in r.content if b.type == "text")
    m = re.search(r"ANSWER:\s*(-?\d+)", text)
    return (int(m.group(1)) if m else None), r.usage.output_tokens

def verify(ans):                    # replace with a real checker: unit tests, a solver, a schema
    return ans is not None and ans % 7 == 0

def best_of_n(question, n=4, effort="medium"):
    spent = 0
    for _ in range(n):              # sequential early exit; use a thread pool for parallel calls
        ans, toks = solve(question, effort)
        spent += toks
        if verify(ans):
            return ans, spent
    return None, spent              # escalate: raise effort, switch model, or ask a human

print(best_of_n("Find the smallest 3-digit multiple of 7 whose digits sum to 13."))
```

## Choosing / trade-offs
- **More samples vs a bigger model**: Snell et al. report that on easier and medium problems, test-time compute on a smaller model could match a much larger one at equal FLOPs, while on the hardest problems pretraining a larger model was the better use of compute. Check on your own eval set.
- **Majority vote vs verifier**: voting needs no extra model but only works with comparable answers; an exact verifier is best when it exists; a learned reward model generalises but can be gamed at large N.
- **Parallel vs sequential**: parallel calls cut wall-clock latency (run concurrently) but multiply cost; longer thinking costs less overhead per useful token but adds latency.
- **Fixed vs adaptive budget**: route easy queries to low effort and N=1, hard ones to higher effort or more samples; difficulty can be estimated from a cheap first pass or sample agreement ([[llm-gateways-and-routing]]).
- **Early stopping**: stop sampling once k samples agree or a verifier passes; this saves most of the cost of fixed N ([[self-consistency-voting-from-scratch]]).

## Gotchas
- Coverage numbers (pass@N) overstate what you can ship unless you have an oracle-quality verifier.
- Reward models over-optimised by large N pick fluent wrong answers; monitor accuracy as N grows, not just the reward.
- Answer normalisation matters: "1/2", "0.5" and "50%" must vote together.
- Samples from the same prompt are correlated; without temperature control (current Claude models) vary the prompt or framing if diversity is too low.
- Higher effort does not reliably make visible answers longer or shorter; it changes thinking volume. Set `max_tokens` large enough for thinking plus the answer.
- Test-time scaling can amplify a leaked or contaminated eval: check gains on held-out data.

## Related
- [[reasoning-models]] - models trained to use long chains of thought; effort and thinking controls.
- [[self-consistency-voting-from-scratch]] - majority voting implemented step by step.
- [[decoding-and-sampling]] - sampling settings behind parallel samples.
- [[mcts-from-scratch]] - tree search used in search-based scaling.
- [[reward-modeling-and-verifiers]] - outcome and process reward models used to select or guide samples.
- [[grpo-from-scratch-numpy]] - RL training that produces long-reasoning models.
- [[llm-cost-calculator]] - pricing N samples and higher effort.
- [[llm-evaluation]] - measuring whether the extra compute pays off.
- [[synthetic-data-for-llm-training]] - spending inference compute to create training data.

## References
- Snell et al. (2024), Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters: https://arxiv.org/abs/2408.03314
- Wang et al. (2023), Self-Consistency Improves Chain of Thought Reasoning: https://arxiv.org/abs/2203.11171
- Cobbe et al. (2021), Training Verifiers to Solve Math Word Problems: https://arxiv.org/abs/2110.14168
- Lightman et al. (2023), Let's Verify Step by Step: https://arxiv.org/abs/2305.20050
- Brown et al. (2024), Large Language Monkeys: Scaling Inference Compute with Repeated Sampling: https://arxiv.org/abs/2407.21787
- Yao et al. (2023), Tree of Thoughts: https://arxiv.org/abs/2305.10601
- Muennighoff et al. (2025), s1: Simple test-time scaling: https://arxiv.org/abs/2501.19393
- Claude effort parameter: https://platform.claude.com/docs/en/build-with-claude/effort
