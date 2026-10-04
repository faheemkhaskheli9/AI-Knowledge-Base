---
title: Thompson sampling vs UCB1 vs epsilon-greedy from scratch (Bernoulli bandits, regret)
category: ml
tags: [multi-armed-bandits, thompson-sampling, ucb, epsilon-greedy, exploration-exploitation, regret, beta-bernoulli, ab-testing, numpy, from-scratch, ml-basics]
use_cases:
  - "implement Thompson sampling, UCB1 and epsilon-greedy for click-through-rate bandits in NumPy"
  - "compare bandit policies by cumulative regret and how often they lock onto the best arm"
  - "see why pure greedy fails and why UCB1's exploration bonus can be too large in practice"
  - "explain exploration vs exploitation and bandits vs A/B tests in an interview"
status: draft
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/1707.02038
  - https://doi.org/10.1023/A:1013689704352
  - https://tor-lattimore.com/downloads/book/book.pdf
---

# Thompson sampling vs UCB1 vs epsilon-greedy from scratch (Bernoulli bandits, regret)

## Summary
A multi-armed bandit chooses repeatedly among K options with unknown reward rates, such as ad creatives, headlines or prices, and has to balance trying arms to learn about them (exploration) against playing the best one so far (exploitation). Performance is measured by **regret**: the reward lost compared with always playing the best arm. Below, four policies play 10 Bernoulli arms with click rates drawn from U(0.1, 0.6), for 5,000 rounds over 200 random problems. Thompson sampling, which samples each arm's rate from a Beta posterior and plays the highest sample, has the lowest regret (100) and plays the best arm 68% of the time. ε-greedy (ε = 0.1) reaches 176, and its regret keeps growing linearly. Pure greedy averages 278, but it is all-or-nothing: in 47% of runs it locks onto a bad arm forever. Textbook UCB1 (319) is the worst at this horizon, because its `√(2 ln t / n)` bonus is huge compared with gaps of a few percentage points.

## Key concepts
- **Regret.** `R_T = Σ_t (μ* − μ_{a_t})`, the expected reward lost. A good policy has regret that grows like `log T`. A policy that keeps exploring at a fixed rate (ε-greedy) or that can lock in on a bad arm (greedy) has regret that grows linearly in T.
- **ε-greedy.** Play the best empirical arm, except with probability ε play a random one. It is simple, but it never stops exploring at rate ε. Decaying ε fixes this, at the cost of a schedule to tune.
- **UCB1.** Play `argmax  x̄_a + √(2 ln t / n_a)`, which is optimism in the face of uncertainty. It has a worst-case `O(log T)` guarantee, but the constant is large: rewards in [0, 1] with small gaps mean many forced pulls of bad arms.
- **Thompson sampling.** Keep a posterior per arm (for clicks, `Beta(1 + wins, 1 + losses)`), draw one sample from each, and play the argmax. Each arm is played with the probability that it is the best, so exploration fades automatically as the posteriors sharpen.
- **Bandit vs A/B test.** An A/B test explores uniformly and then commits, which is best for a clean estimate of the effect. A bandit shifts traffic to winners as it learns, which is best for total reward collected during the experiment.

## When to use / scenarios
- Learning: the cleanest setting for exploration vs exploitation, and the base case for contextual bandits and reinforcement learning ([[reinforcement-learning]]).
- Interviews: "how would you choose among 10 ad creatives", "Thompson sampling vs UCB", "when to use a bandit instead of an A/B test", "what is regret".
- Practice: headline and thumbnail testing, ad creative rotation, recommendation slot exploration, dynamic pricing among discrete prices, and choosing among LLM prompts or models by user feedback. Use it when the cost of showing a loser during the test matters, and when arms can be added or removed continuously ([[multi-armed-bandits]]).
- Not for: decisions that need an unbiased effect estimate or a regulatory-grade test (use a fixed-allocation A/B test: [[statistics-and-ab-testing]]); rewards that arrive long after the action, such as 30-day retention (the bandit learns too slowly, so use a proxy metric or batch updates); or cases where each user's best arm depends on features (use contextual bandits such as LinUCB or Thompson with a linear model).

## Setup & code
NumPy only. 200 runs × 4 policies × 5,000 rounds takes about 40 seconds.

```python
import numpy as np

rng = np.random.default_rng(0)
K, T, RUNS = 10, 5000, 200


def run(policy, p):
    counts, wins = np.zeros(K), np.zeros(K)
    regret = np.zeros(T)
    best = p.max()
    a_post, b_post = np.ones(K), np.ones(K)          # Beta(1,1) priors for Thompson
    for t in range(T):
        if policy == "greedy":
            arm = t if t < K else np.argmax(wins / counts)
        elif policy == "eps-greedy 0.1":
            arm = t if t < K else (rng.integers(K) if rng.random() < 0.1 else np.argmax(wins / counts))
        elif policy == "UCB1":
            arm = t if t < K else np.argmax(wins / counts + np.sqrt(2 * np.log(t) / counts))
        else:  # Thompson sampling
            arm = np.argmax(rng.beta(a_post, b_post))
        r = rng.random() < p[arm]
        counts[arm] += 1
        wins[arm] += r
        a_post[arm] += r
        b_post[arm] += 1 - r
        regret[t] = best - p[arm]                       # expected (pseudo-)regret
    return np.cumsum(regret), counts


policies = ["greedy", "eps-greedy 0.1", "UCB1", "Thompson"]
res = {k: [] for k in policies}
picks = {k: [] for k in policies}
for _ in range(RUNS):
    p = rng.uniform(0.1, 0.6, K)                        # unknown click-through rates
    for k in policies:
        cum, counts = run(k, p)
        res[k].append(cum)
        picks[k].append(counts[np.argmax(p)] / T)

print(f"{K} Bernoulli arms, p ~ U(0.1, 0.6), {T} rounds, {RUNS} runs")
print("policy           regret@500  regret@5000  best-arm share  runs regret>150")
for k in policies:
    c = np.array(res[k])
    print(f"{k:<15} {c[:, 499].mean():>11.1f}  {c[:, -1].mean():>11.1f}  {np.mean(picks[k]):>14.3f}"
          f"  {np.mean(c[:, -1] > 150):>15.3f}")
```

Output (Python 3.14, NumPy 2.5):
```
10 Bernoulli arms, p ~ U(0.1, 0.6), 5000 rounds, 200 runs
policy           regret@500  regret@5000  best-arm share  runs regret>150
greedy                 31.7        278.2           0.353            0.465
eps-greedy 0.1         31.8        175.5           0.550            0.465
UCB1                   67.6        319.2           0.435            1.000
Thompson               39.5        100.4           0.675            0.075
```

The last column shows the shape of each failure. Greedy is not uniformly mediocre. It does very well in half the runs and in the other half commits to an arm whose first pulls happened to win, then never looks back. ε-greedy rescues those runs, but it pays ε × (average gap) on every round forever, so its regret grows by about 18 per 500 rounds and never levels off. UCB1 starts slowest (67.6 at round 500), because it must pull every arm until its confidence bound falls below the leader's. With gaps around 0.02-0.05, that takes hundreds of pulls per arm, so every run ends with regret above 150. Thompson explores about as much as greedy early on (39.5 at round 500), yet only 7.5% of its runs exceed 150 regret, because exploration of an arm stops once the posterior is confident that the arm is worse.

## Choosing / trade-offs
- **Default: Thompson sampling.** It is near-optimal in practice, has no tuning knob beyond the prior, handles batched or delayed updates gracefully (sample many times from the same posterior), and extends to Gaussian rewards, contextual models and Bayesian neural nets.
- **UCB variants.** UCB1's bound is distribution-free and loose. KL-UCB and UCB1-Tuned use the reward variance and close most of the gap to Thompson, and they are deterministic, which makes them easier to debug and audit.
- **ε-greedy.** Use it only as a baseline or when the action set is huge and posteriors are impractical. Decay ε, for example `ε_t = min(1, c·K / t)`.
- **Non-stationary rates.** Click rates drift. Use discounted counts (multiply `a`, `b` by γ < 1 each round) or a sliding window, or the bandit stays locked on yesterday's winner ([[online-learning-and-concept-drift]]).
- **Libraries.** Vowpal Wabbit (contextual bandits), MABWiser and the bandit tooling in experimentation platforms. The from-scratch policies above are small enough to run in production as-is for a non-contextual case.

## Gotchas
- Greedy with optimistic initial values (start every arm at 1.0) is a cheap fix for greedy's lock-in, but it depends on the reward scale.
- Report regret distributions, not only means. A policy with good average regret can still have a heavy tail of bad runs (greedy here).
- Bandit traffic splits are not random samples, so a naive conversion-rate comparison at the end is biased. Use inverse-propensity weighting if you need effect estimates from bandit logs.
- Delayed feedback: update counts only when the reward arrives, and keep sampling from the current posterior in the meantime. Do not count a pending reward as a 0.
- Real CTRs are often close together (0.01 vs 0.012). Expect thousands of impressions per arm before any policy separates them. Simulate with realistic rates before promising results.
- Thompson sampling with too strong a prior (for example Beta(50, 50)) explores far too long. Keep priors weak, or fit them to historical arms.

## Related
- [[multi-armed-bandits]] - bandit algorithms and libraries, contextual bandits.
- [[statistics-and-ab-testing]] - fixed-allocation experiments and when they are preferable.
- [[q-learning-from-scratch]] - exploration in full reinforcement learning with states.
- [[bayesian-linear-regression-from-scratch]] - the posterior used by linear Thompson sampling.
- [[online-learning-and-concept-drift]] - handling drifting reward rates.

## References
- Russo, Van Roy, Kazerouni, Osband and Wen (2018), "A Tutorial on Thompson Sampling": https://arxiv.org/abs/1707.02038
- Auer, Cesa-Bianchi and Fischer (2002), "Finite-time Analysis of the Multiarmed Bandit Problem" (UCB1), Machine Learning: https://doi.org/10.1023/A:1013689704352
- Lattimore and Szepesvári, "Bandit Algorithms" (book PDF): https://tor-lattimore.com/downloads/book/book.pdf
