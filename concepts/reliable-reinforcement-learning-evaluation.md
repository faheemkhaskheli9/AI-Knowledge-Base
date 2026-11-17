---
title: Reliable reinforcement learning evaluation (seeds, IQM, uncertainty)
category: concepts
tags: [reinforcement-learning, evaluation, reproducibility, bootstrap, iqm, confidence-intervals, rliable, ablation, generalization]
use_cases:
  - "compare reinforcement learning algorithms without mistaking seed noise for progress"
  - "measure whether a learning agent transfers to unseen tasks and retains previous skills"
  - "design a reproducible experiment on brain-inspired learning, memory, or planning"
status: draft
last_verified: 2026-11-17
sources:
  - https://papers.neurips.cc/paper_files/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html
  - https://ojs.aaai.org/index.php/AAAI/article/view/11694
  - https://research.google/blog/rliable-towards-reliable-evaluation-reporting-in-reinforcement-learning/
  - https://github.com/google-research/rliable
  - https://rll.berkeley.edu/deeprlcourse/
---

# Reliable reinforcement learning evaluation (seeds, IQM, uncertainty)

## Summary

A good score from one training seed is weak evidence. Initialization, stochastic environments, exploration, evaluation episodes and benchmark selection all influence reported RL performance. To compare algorithms, pre-register the hypothesis and failure criterion, use independently trained seeds, keep tests held out, match interaction and search budgets, and report uncertainty. The paper *Deep Reinforcement Learning at the Edge of the Statistical Precipice* proposes interquartile mean (IQM), performance profiles and stratified bootstrap intervals for few-run RL benchmarks.

## Key concepts

- **Training seeds are the independent units.** Ten evaluation episodes from one checkpoint are not ten independently trained policies. Track training seeds, evaluation seeds, and all failed runs separately.
- **Hold out task structure.** Freeze unseen layouts, changed dynamics and novel combinations before training. Using the test distribution to tune hyperparameters undermines transfer claims.
- **Match interaction and compute.** Compare environment steps, replay sampling, gradients, model-fitting steps, imagined/planning rollouts, hardware and wall time. Extra search compute must be disclosed.
- **Ablate the claimed mechanism.** Compare the candidate with no-learning/random, frozen weights, memory-only retrieval and a strong traditional learning baseline. Test replay versus no-replay under matched budgets.
- **Evaluate learning, not answer storage.** Track repeated errors before and after corrective feedback; perturb situations, delay retention tests, and test unseen action/task compositions. Explicitly distinguish weight changes from episodic retrieval.
- **Report both learning and generalization.** Show per-task success, worst-task score, area under the learning curve, final return, transfer, forgetting, compute, and robustness over independent runs.
- **Estimate uncertainty correctly.** On a fixed suite of tasks, bootstrap independent training runs within each task. IQM describes the middle half of normalized benchmark scores, not an unqualified global intelligence measure.
- **Pre-register.** Record seeds, task split, normalization, budget, primary outcome, smallest practical effect, interval method and acceptable failure rate before seeing results.

## When to use / scenarios

- Studying continual learning, reward-based credit assignment, world models, neural plasticity, planning and embodied control.
- Auditing any claim about a system's transfer, adaptation, retention or learning from mistakes.
- Comparing an LLM tool agent with a model that actually updates a policy from feedback.
- Use ordinary deterministic unit tests where appropriate; infer neither human-level intelligence nor biological plausibility from a single simulator score.

## Setup & code

**Synthetic, educational example:** scores below are invented, normalized, and on two fixed tasks. This is not an experimental result. The stdlib-only script computes a difference in IQM and resamples runs within tasks. For actual multi-task RL studies use [RLiable](https://github.com/google-research/rliable); consider hierarchical resampling when claiming generalization to new task populations.

```python
from random import Random

scores = {
    "baseline": {"maze": [0.45, 0.52, 0.49, 0.53, 0.46],
                 "transfer": [0.21, 0.22, 0.17, 0.25, 0.19]},
    "candidate": {"maze": [0.56, 0.58, 0.52, 0.49, 0.60],
                  "transfer": [0.31, 0.34, 0.28, 0.25, 0.30]},
}

def iqm(per_task):
    values = sorted(v for runs in per_task.values() for v in runs)
    k = len(values) // 4
    middle = values[k:len(values) - k]
    return sum(middle) / len(middle)

rng = Random(17)

def resample(per_task):
    return {task: [rng.choice(runs) for _ in runs]
            for task, runs in per_task.items()}

observed = iqm(scores["candidate"]) - iqm(scores["baseline"])
deltas = sorted(
    iqm(resample(scores["candidate"])) - iqm(resample(scores["baseline"]))
    for _ in range(2000)
)
lo, hi = deltas[int(0.025 * len(deltas))], deltas[int(0.975 * len(deltas))]
print(f"synthetic IQM difference: {observed:.3f}; "
      f"bootstrap 95% interval [{lo:.3f}, {hi:.3f}]")
assert 0.07 < observed < 0.08
```

On the synthetic input, the difference is about 0.073 and the illustrative interval about [0.040, 0.107]. These are not real benchmarks.

## Choosing / trade-offs

- **One task, many seeds:** report independent training-run distributions, learning curves, effect size and interval. IQM across unrelated tasks is not needed.
- **Multiple tasks, few seeds:** report per-task returns, IQM on appropriately normalized scores, performance profiles and stratified uncertainty.
- **Unseen-task population:** use a principled task sampling design; uncertainty conditional on fixed tasks is not the same as uncertainty across arbitrary unseen tasks.
- **Model-based planning:** report simulation/planning work and real-world interaction costs separately.
- **Paired-seed experiment:** sharing seed integers does not guarantee paired outcomes; only use paired inference when the randomization coupling warrants it.

## Gotchas

- Selecting the best seed/checkpoint or tuning on test tasks inflates reported success.
- Repeated evaluation rollouts do not replace independent training runs.
- Truncated episodes and terminated episodes have different bootstrapping semantics.
- Excluding crashes, divergence or NaN runs systematically advantages unstable models.
- Bootstrap confidence intervals cannot fix leakage, weak baselines, hardcoded solutions or unfair compute budgets.
- If the effect interval is compatible with practically irrelevant improvement, say **inconclusive**.

## Related

- [[reinforcement-learning]] - algorithms and task setup.
- [[continual-learning]] - retention and forgetting.
- [[meta-learning]] - held-out task-family adaptation.
- [[agent-evaluation]] - independent task outcome and trajectory scoring.
- [[experiment-tracking]] - seeds, artifacts and version metadata.
- [[world-models]] - imagined rollouts, planning and prediction errors.
- [[statistics-and-ab-testing]] - experimental design.

## References

- Agarwal et al., *Deep Reinforcement Learning at the Edge of the Statistical Precipice*, NeurIPS 2021: https://papers.neurips.cc/paper_files/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html
- Henderson et al., *Deep Reinforcement Learning That Matters*, AAAI 2018: https://ojs.aaai.org/index.php/AAAI/article/view/11694
- Google Research RLiable guidance: https://research.google/blog/rliable-towards-reliable-evaluation-reporting-in-reinforcement-learning/
- RLiable code: https://github.com/google-research/rliable
- UC Berkeley CS285: https://rll.berkeley.edu/deeprlcourse/
