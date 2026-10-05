---
title: Self-consistency (majority voting over sampled reasoning paths) from scratch
category: llm-apps
tags: [self-consistency, majority-voting, chain-of-thought, test-time-compute, answer-extraction, confidence, adaptive-sampling, best-of-n, numpy, from-scratch]
use_cases:
  - "improve LLM accuracy on math, logic or extraction questions by sampling several answers and voting"
  - "get a cheap confidence score for an LLM answer from agreement between samples"
  - "decide how many samples to draw and when to stop early to control cost"
status: draft
last_verified: 2026-10-05
sources:
  - https://arxiv.org/abs/2203.11171
  - https://arxiv.org/abs/2305.11860
---

# Self-consistency (majority voting over sampled reasoning paths) from scratch

## Summary
Self-consistency (Wang et al. 2023) samples several chain-of-thought completions at non-zero temperature, extracts the final answer from each, and returns the most common answer. Many different reasoning paths can reach the same correct answer, while wrong paths tend to scatter across different wrong answers. The mode of the *answer* distribution is therefore often right even when the single most likely *path* (what greedy decoding returns) is wrong. This file simulates questions as sets of reasoning paths with model probabilities. It implements answer extraction, majority and probability-weighted voting, vote share as a confidence signal, and early stopping, and measures accuracy against the number of samples.

## Key concepts
- **Paths vs answers.** Greedy decoding approximates the most likely *path*. Self-consistency estimates the most likely *answer*, which sums probability over all paths that end in it. These can differ, and that gap is the whole gain.
- **Answer extraction is part of the method.** Votes only count if equivalent answers compare equal: "42", "42.0", "**42**" and "$42" must map to one key. Use a fixed output format ("The answer is X") and a normaliser.
- **Convergence.** As the sample count grows, majority voting converges to the mode of the answer distribution. More samples cannot fix a question whose most likely answer is wrong.
- **Vote share as confidence.** The fraction of samples that agree with the winner is a well-ordered confidence score at no extra cost: high agreement means high accuracy.
- **Adaptive sampling.** Stop drawing samples once agreement is clear (Aggarwal et al. 2023 make this rigorous with a stopping criterion). Easy questions stop early and hard ones use the full budget.

## When to use / scenarios
- Math, arithmetic, logic and multiple-choice questions, or extraction with a short canonical answer (a number, date, label, entity). Anything where answers can be compared exactly.
- Batch jobs where accuracy matters more than cost: grading, data labelling, evaluation pipelines ([[llm-evaluation]]).
- Routing and abstention: send low-agreement items to a human or a stronger model, and auto-accept high-agreement ones ([[hallucination-and-grounding]]).
- Not for open-ended text (summaries, emails), where no two samples match exactly. Use a verifier or reward model for best-of-n, an LLM judge, or the "universal self-consistency" variant that asks a model to pick the most consistent response.
- Less needed with reasoning models ([[reasoning-models]]), which already spend test-time compute internally. Voting on top still helps on hard benchmarks but multiplies an already large cost.

## Setup & code
NumPy and the standard library, under a second. A real run would call an LLM `n` times at temperature around 0.7 and pass each completion through `extract_answer`. The simulation replaces the LLM. Each of 2000 questions has 30 reasoning paths with Dirichlet-distributed probabilities. Each path ends in the correct answer (`0`) with probability `1 - difficulty`, otherwise in one of three distractors. Difficulty is uniform in 0.2–0.7.

```python
import re
from collections import Counter

import numpy as np

rng = np.random.default_rng(0)


def extract_answer(text):
    """Pull the final numeric answer out of a chain-of-thought completion; None if there is none."""
    m = re.search(r"answer is\s*[:\-]?\s*\**\s*\$?(-?[\d,]*\.?\d+)", text, re.I)
    nums = [m.group(1)] if m else re.findall(r"-?[\d,]*\.?\d+", text)
    if not nums:
        return None
    x = float(nums[-1].replace(",", ""))
    return str(int(x)) if x.is_integer() else f"{x:g}"           # "42", "42.0" and "42.00" vote together


assert extract_answer("3 + 4 = 7, so 7 * 6 = 42. The answer is 42.") == "42"
assert extract_answer("Total: $1,200.00") == "1200"
assert extract_answer("The answer is **42.0**") == "42"
assert extract_answer("I am not sure.") is None
print("answer extraction ok")


def make_question(difficulty):
    """A question = K reasoning paths with model probabilities; each path ends in the right answer or a distractor."""
    K = 30
    probs = rng.dirichlet(np.full(K, 0.5))
    right = rng.random(K) < 1 - difficulty
    answers = np.where(right, 0, rng.integers(1, 4, K))        # answer 0 is correct, 1-3 are distractors
    return probs, answers


def sample(q, n):
    probs, answers = q
    idx = rng.choice(len(probs), n, p=probs)
    return answers[idx], np.log(probs[idx])


def greedy(q):
    probs, answers = q
    return answers[probs.argmax()]                              # temperature 0 ~ the single most likely path


def vote(ans, logp=None):
    if logp is None:
        return Counter(ans.tolist()).most_common(1)[0][0]
    w = Counter()
    for a, lp in zip(ans.tolist(), logp):
        w[a] += np.exp(lp)                                       # weight each vote by its path probability
    return max(w, key=w.get)


qs = [make_question(d) for d in rng.uniform(0.2, 0.7, 2000)]
print(f"greedy (1 path)                acc {np.mean([greedy(q) == 0 for q in qs]):.3f}")
for n in (1, 5, 10, 20, 40):
    draws = [sample(q, n) for q in qs]
    maj = np.mean([vote(a) == 0 for a, _ in draws])
    wtd = np.mean([vote(a, lp) == 0 for a, lp in draws])
    print(f"self-consistency n={n:<3}         acc {maj:.3f}   prob-weighted {wtd:.3f}")
oracle = np.mean([np.bincount(a, weights=p, minlength=4).argmax() == 0 for p, a in qs])
print(f"mode of the answer distribution acc {oracle:.3f}   (what voting converges to)")

# Vote share as a confidence signal, and early stopping on it
draws = [sample(q, 40) for q in qs]
share = np.array([Counter(a.tolist()).most_common(1)[0][1] / 40 for a, _ in draws])
correct = np.array([vote(a) == 0 for a, _ in draws])
for lo, hi in ((0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)):
    sel = (share >= lo) & (share < hi)
    print(f"vote share {lo:.1f}-{min(hi, 1):.1f}: {sel.sum():4d} questions, accuracy {correct[sel].mean():.3f}")


def adaptive(ans, min_n=5, stop_share=0.8):
    """Stop once the leading answer holds `stop_share` of the votes after at least `min_n` samples."""
    for n in range(min_n, len(ans) + 1):
        top, c = Counter(ans[:n].tolist()).most_common(1)[0]
        if c / n >= stop_share:
            return top, n
    return top, len(ans)


res = [adaptive(a) for a, _ in draws]
print(f"adaptive stop (>=80% agree after 5): acc {np.mean([r[0] == 0 for r in res]):.3f}, "
      f"mean samples {np.mean([r[1] for r in res]):.1f} of 40")
```

Output (Python 3.14, NumPy 2.5):
```
answer extraction ok
greedy (1 path)                acc 0.561
self-consistency n=1           acc 0.544   prob-weighted 0.544
self-consistency n=5           acc 0.676   prob-weighted 0.616
self-consistency n=10          acc 0.741   prob-weighted 0.657
self-consistency n=20          acc 0.770   prob-weighted 0.677
self-consistency n=40          acc 0.797   prob-weighted 0.664
mode of the answer distribution acc 0.817   (what voting converges to)
vote share 0.0-0.5:  629 questions, accuracy 0.545
vote share 0.5-0.7:  791 questions, accuracy 0.847
vote share 0.7-0.9:  465 questions, accuracy 0.989
vote share 0.9-1.0:  115 questions, accuracy 1.000
adaptive stop (>=80% agree after 5): acc 0.786, mean samples 25.4 of 40
```

How to read it:
- Greedy (the single most likely path) is right 56% of the time. One sample at temperature 1 is slightly worse (54%), which is why single-sample accuracy is reported at temperature 0.
- Majority voting climbs quickly: 68% at 5 samples, 74% at 10, 80% at 40. It approaches the 82% ceiling set by the mode of the answer distribution. Most of the gain comes in the first 5–10 samples.
- Probability-weighted voting is *worse* (66% at 40). Sampling already draws high-probability paths more often, so weighting by probability again counts them twice and pulls the result back towards the greedy path. Wang et al. found plain majority voting about as good as weighting by length-normalised probability; use plain majority unless you have a calibrated verifier.
- Vote share is a strong confidence signal. Below 50% agreement accuracy is 55%; above 70% it is 99%. Auto-accept the high-agreement third of the data and route the rest.
- Stopping once 80% of at least 5 samples agree keeps nearly all the accuracy (78.6% vs 79.7%) with 37% fewer samples (25.4 vs 40). Easy questions stop at 5; hard ones run to the cap.
- The simulation fixes the shape of the effect, not its size. Real gains depend on the model and task. The paper reports double-digit gains on arithmetic benchmarks with 40 samples, and smaller gains for stronger models.

## Choosing / trade-offs
- **Number of samples.** 5–10 gets most of the benefit; 20–40 for evaluation or high-stakes batches. Cost is linear in `n`, and with a shared prompt the input tokens can be cached ([[prompt-caching-and-cost]]). Some APIs return `n` completions per request.
- **Temperature.** Around 0.5–1.0. Too low and samples repeat the greedy path (no diversity, no gain); too high and paths get incoherent. Top-p sampling is fine ([[decoding-strategies-from-scratch-numpy]]).
- **Majority vote vs verifier.** Voting needs no extra model but only works with exact-match answers. A verifier or reward model picks among samples by quality and works for open-ended output, but needs to be trained or prompted and can be gamed.
- **Self-consistency vs a reasoning model.** Voting over a cheaper model can match a single call to a pricier one on some tasks. Compare cost per correct answer, not per call.

## Gotchas
- Ties: decide a rule (fewest tokens, first seen, or a tie-break sample) and log ties. With few samples and many distractors, ties are common.
- Count unparseable answers (`None`) as their own bucket or drop them, but do not let "no answer" win the vote.
- Normalise answers per task: units, fractions vs decimals, case and whitespace for labels, date formats. A missed normalisation splits the correct vote and lets a wrong answer win.
- Agreement is not correctness. When the model is confidently wrong (a common misconception in its training data), all samples agree on the wrong answer. Vote share ranks questions by confidence but is not calibrated; calibrate it on a labelled set ([[platt-and-isotonic-calibration-from-scratch]]).
- Samples from one prompt are correlated. Varying the few-shot examples or phrasing between samples adds diversity and can help more than extra samples.

## Related
- [[decoding-strategies-from-scratch-numpy]] - temperature and top-p sampling, the source of path diversity.
- [[decoding-and-sampling]] - generation parameters and best-of-n with a verifier.
- [[reasoning-models]] - models that spend test-time compute internally instead of through external voting.
- [[hallucination-and-grounding]] - using sample agreement to detect unreliable answers.
- [[platt-and-isotonic-calibration-from-scratch]] - turn vote share into a calibrated probability.
- [[llm-evaluation]] - measuring whether the extra samples pay for themselves.

## References
- Wang et al. (2023), "Self-Consistency Improves Chain of Thought Reasoning in Language Models", ICLR: https://arxiv.org/abs/2203.11171
- Aggarwal et al. (2023), "Let's Sample Step by Step: Adaptive-Consistency for Efficient Reasoning and Coding with LLMs", EMNLP: https://arxiv.org/abs/2305.11860
