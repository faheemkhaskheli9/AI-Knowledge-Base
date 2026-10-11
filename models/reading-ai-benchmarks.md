---
title: Reading AI benchmarks and leaderboards
category: models
tags: [benchmarks, leaderboards, mmlu, gpqa, swe-bench, arc-agi, lmarena, contamination, saturation, evaluation]
use_cases:
  - "judge whether a vendor's benchmark claim should change our model choice"
  - "decide if a 3-point gap between two models on a leaderboard is real"
  - "understand what MMLU, GPQA, SWE-bench or ARC-AGI actually measure"
  - "explain why we build our own eval set instead of trusting leaderboard rank"
status: draft
last_verified: 2026-10-11
sources:
  - https://arxiv.org/abs/2009.03300
  - https://arxiv.org/abs/2406.01574
  - https://arxiv.org/abs/2311.12022
  - https://arxiv.org/abs/2107.03374
  - https://arxiv.org/abs/2403.07974
  - https://arxiv.org/abs/2310.06770
  - https://arxiv.org/abs/2103.03874
  - https://arxiv.org/abs/1911.01547
  - https://arcprize.org/arc-agi
  - https://arxiv.org/abs/2501.14249
  - https://arxiv.org/abs/2403.04132
  - https://arena.ai/how-it-works
  - https://arxiv.org/abs/2311.09783
---

# Reading AI benchmarks and leaderboards

## Summary
A benchmark score is a measurement of one narrow task format under one
prompt and scoring setup, not a measure of "intelligence" or of your use
case. This topic explains what the common benchmarks test, why scores stop
being informative (saturation, contamination), how arena-style leaderboards
work, and why your own eval set is the deciding evidence. It carries no model
scores on purpose: they go stale within months.

## Key concepts
- **What each benchmark measures** (per its own paper):
  - MMLU: multiple-task accuracy over 57 subjects (elementary maths, US
    history, computer science, law, ...). MMLU-Pro raises the answer options
    from four to ten, adds harder reasoning questions, removes trivial and
    noisy ones, and reports lower accuracy and less prompt sensitivity.
  - GPQA: 448 multiple-choice biology, physics and chemistry questions
    written by PhD-level experts and designed to be "Google-proof" (skilled
    non-experts with web access did poorly).
  - HumanEval: functional correctness of programs written from docstrings,
    checked by tests. LiveCodeBench keeps adding fresh problems from LeetCode,
    AtCoder and CodeForces contests, so you can score only problems dated
    after a model's training cutoff.
  - SWE-bench: 2,294 real GitHub issues from 12 Python repositories; the model
    must edit the codebase to resolve the issue. Closer to real work than a
    single function, but Python-only and depends heavily on the agent
    scaffold around the model.
  - MATH: 12,500 competition mathematics problems with worked solutions.
  - ARC / ARC-AGI: abstract grid puzzles meant to test skill-acquisition on
    novel tasks using only basic priors, not language or cultural knowledge;
    "easy for humans, hard for AI".
  - Humanity's Last Exam: 2,500 closed-ended expert questions built because
    older tests such as MMLU were near saturation.
- **Saturation**: when top models cluster near the ceiling, the benchmark
  cannot separate them. The MMLU-Pro and HLE papers were written for exactly
  this reason. Each generation of benchmark is a reaction to the last one
  saturating.
- **Contamination**: test items (or near copies) leaked into training data, so
  the score measures memory, not skill. Detection is hard for closed models;
  one paper's TS-Guessing probe hides a wrong answer option and asks the model
  to fill it in, and reported that some commercial models could do so on
  MMLU. Time-split benchmarks (LiveCodeBench) and private test sets are the
  structural defences.
- **Arena-style leaderboards**: users compare two anonymous model answers to
  the same prompt and vote; votes feed a pairwise-comparison ranking (see
  [[elo-and-bradley-terry-leaderboard-from-scratch]] for the maths). This
  measures human preference on whatever users happen to ask, including style
  and formatting, not correctness on your task.
- **Setup matters as much as the model**: prompt format, few-shot count,
  sampling (pass@k with k samples vs one try), tool access and scaffold all
  change a number. A claim without these details is not comparable.

## When to use / scenarios
- A vendor launch post claims a win: check which benchmark, which setup, and
  whether the benchmark still separates models.
- Shortlisting: use public benchmarks only to pick 2-4 candidates, then run
  your own eval ([[model-selection]], [[llm-evaluation]]).
- Match the benchmark to the task: SWE-bench-like for coding agents,
  GPQA/HLE-like for expert reasoning, arena preference for chat tone, ARC-AGI
  for novel-puzzle generalisation. None of these predicts, say, invoice
  extraction accuracy.
- Do NOT use a leaderboard rank as the decision, or to justify a cost
  difference without a task-level test.

## Setup & code
A bootstrap check for whether a gap on a small eval set is noise. The data
below is synthetic toy data, not real model results.
```python
import random

random.seed(0)
N = 100
# per-item correctness on the same 100 questions (1 = right)
a = [1] * 72 + [0] * 28
b = [1] * 66 + [0] * 34
random.shuffle(a); random.shuffle(b)

def boot_diff(a, b, n=5000):
    diffs = []
    for _ in range(n):
        idx = [random.randrange(len(a)) for _ in a]  # resample items, paired
        diffs.append(sum(a[i] - b[i] for i in idx) / len(idx))
    diffs.sort()
    return diffs[int(0.025 * n)], diffs[int(0.975 * n)]

lo, hi = boot_diff(a, b)
print(f"gap = {sum(a)/N - sum(b)/N:+.2f}, 95% CI = [{lo:+.2f}, {hi:+.2f}]")
print("noise" if lo <= 0 <= hi else "real gap")
# expected output:
# gap = +0.06, 95% CI = [-0.06, +0.18]
# noise
```
Six points on 100 questions is within noise. In real use score both models on
the same items and resample those paired items, as above.

## Choosing / trade-offs
| Evidence | Good for | Weakness |
|---|---|---|
| Static public benchmark | Cheap first filter, comparing across vendors | Saturation, contamination, setup differences |
| Time-split / private benchmark | Resisting contamination | Smaller, fewer models tested |
| Arena preference votes | Chat quality as users feel it | Style bias, user mix is not your mix |
| Your own eval set | Your real task and data | Costs effort to build and label |
Pull toward your own eval as stakes rise or as the task gets unusual.

## Gotchas
- A headline gap on a few hundred items can be noise; report intervals.
- "Same benchmark" claims can differ in prompt, shots, sampling, tools and
  grading, so numbers from two vendors' blog posts are not directly comparable.
- Self-reported vendor numbers are not independently reproduced; prefer
  results with a public harness you can rerun.
- Old benchmarks keep being quoted after they saturate; check the date and
  whether the benchmark still discriminates.
- Optimising for a leaderboard (training on its style or items) improves the
  rank without improving your task.
- Arena rankings can move with response style and length, not only accuracy.
- Keep your own eval set private and out of prompts and fine-tuning data, or it
  contaminates too.

## Related
- [[model-selection]] - the decision procedure these numbers feed into.
- [[llm-evaluation]] - how to build and run your own eval set.
- [[elo-and-bradley-terry-leaderboard-from-scratch]] - how pairwise votes become a ranking.
- [[agent-evaluation]] - SWE-bench, tau-bench and other agent benchmarks in depth.

## References
- MMLU: https://arxiv.org/abs/2009.03300 ; MMLU-Pro: https://arxiv.org/abs/2406.01574
- GPQA: https://arxiv.org/abs/2311.12022
- HumanEval (Codex paper): https://arxiv.org/abs/2107.03374 ; LiveCodeBench: https://arxiv.org/abs/2403.07974
- SWE-bench: https://arxiv.org/abs/2310.06770 ; MATH: https://arxiv.org/abs/2103.03874
- ARC: https://arxiv.org/abs/1911.01547 ; https://arcprize.org/arc-agi
- Humanity's Last Exam: https://arxiv.org/abs/2501.14249
- Chatbot Arena: https://arxiv.org/abs/2403.04132 ; https://arena.ai/how-it-works
- Contamination detection: https://arxiv.org/abs/2311.09783
