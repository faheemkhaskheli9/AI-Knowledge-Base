---
title: Multi-armed and contextual bandits
category: ml
tags: [bandits, multi-armed-bandit, contextual-bandit, exploration-exploitation, epsilon-greedy, ucb, thompson-sampling, linucb, regret, vowpal-wabbit, numpy]
use_cases:
  - "pick the best headline, banner or email subject while traffic is live, wasting less traffic than an A/B test"
  - "personalise which offer or recommendation to show using user context"
  - "explore new items in a recommender instead of always showing the current top items"
  - "tune a price, bid or notification time online from click or conversion feedback"
  - "evaluate a new selection policy offline from logged data before deploying it"
status: draft
last_verified: 2026-10-04
sources:
  - https://tor-lattimore.com/downloads/book/book.pdf
  - https://arxiv.org/abs/1707.02038
  - https://arxiv.org/abs/1003.0146
  - https://vowpalwabbit.org/
---

# Multi-armed and contextual bandits

## Summary
A bandit repeatedly chooses one of several actions ("arms"), observes a reward for that action only, and must balance *exploring* uncertain arms with *exploiting* the best one so far. It is the one-step special case of reinforcement learning: actions do not change future states. Multi-armed bandits find the single best arm (best headline, best creative); *contextual* bandits pick the best arm per user or situation from features (personalised offers, recommendations). Compared with a fixed A/B test, bandits shift traffic toward winners while the experiment runs, at the cost of weaker statistical inference.

## Key concepts
- **Regret.** Cumulative reward lost versus always playing the best arm. Good algorithms have regret growing like log(T) or √T; pure exploitation or fixed uniform exploration grows linearly.
- **ε-greedy.** With probability ε pick a random arm, else the best estimate. Simple and robust; wastes exploration on clearly bad arms. Decay ε over time.
- **UCB (upper confidence bound).** Pick the arm with the highest `mean + c·sqrt(ln t / n_arm)`: optimism under uncertainty. Deterministic, no prior needed.
- **Thompson sampling.** Keep a posterior per arm (Beta for click/no-click, Normal for continuous rewards), sample one value from each, pick the max. Explores in proportion to the chance an arm is best; usually the best practical default and easy to batch.
- **Contextual bandits.** Reward depends on context x: learn a model per arm (or one model over (x, arm) features) and explore around its uncertainty. *LinUCB* and *linear Thompson sampling* use Bayesian linear regression; neural or tree models use ε-greedy or bootstrapped ensembles.
- **Propensities and off-policy evaluation.** Log the probability with which each shown action was chosen. Then *inverse propensity scoring* (IPS) and *doubly robust* estimators evaluate a new policy on old logs without deploying it. Without propensities, logs from a deterministic policy cannot evaluate a different one.
- **Non-stationarity.** Click rates drift (novelty, seasonality). Use a sliding window or discounting so old evidence fades.

## When to use / scenarios
- Marketing and content: headline, thumbnail, subject-line, landing-page and ad-creative selection where every impression has a cost ([[marketing-content]]).
- E-commerce and recommendations: slot-level exploration of new items, cold-start promotion, personalised offers and coupons ([[ecommerce-retail]], [[recommender-systems]]).
- Pricing, bidding and notification-time tuning with fast feedback.
- Model or prompt routing: send traffic to the LLM or prompt variant that performs best on live feedback ([[llm-evaluation]]).
- NOT when you need a clean causal estimate with confidence intervals for a decision (launch or not): run a fixed-allocation A/B test ([[statistics-and-ab-testing]]).
- NOT when actions affect future states (inventory depletion, multi-step dialogues): that is full RL ([[reinforcement-learning]]).
- NOT when rewards arrive days later and decisions cannot wait; long delays erode the bandit's advantage over a test.

## Setup & code
```bash
pip install numpy
```
Simulate 5 headlines with click rates between 2% and 5% for 20,000 impressions and compare ε-greedy, UCB1 and Beta-Bernoulli Thompson sampling by regret:
```python
import numpy as np

p = np.array([0.020, 0.030, 0.035, 0.040, 0.050])   # true click rates, unknown to the agent
T, K, runs = 20_000, len(p), 20

def run(policy, seed):
    rng = np.random.default_rng(seed)
    n, s = np.zeros(K), np.zeros(K)                  # pulls and clicks per arm
    regret = 0.0
    for t in range(1, T + 1):
        if policy == "eps-greedy":
            eps = min(1.0, 50 / t)                     # decaying exploration
            a = rng.integers(K) if rng.random() < eps else np.argmax(s / np.maximum(n, 1))
        elif policy == "ucb1":
            a = t - 1 if t <= K else np.argmax(s / n + np.sqrt(2 * np.log(t) / n))
        else:                                          # thompson: sample from Beta posteriors
            a = np.argmax(rng.beta(1 + s, 1 + n - s))
        r = rng.random() < p[a]
        n[a] += 1; s[a] += r
        regret += p.max() - p[a]
    return regret, n[-1] / T

res = {}
for pol in ["eps-greedy", "ucb1", "thompson"]:
    out = np.array([run(pol, seed) for seed in range(runs)])
    res[pol] = out[:, 0].mean()
    print(f"{pol:11s} regret {out[:, 0].mean():6.1f} clicks lost   "
          f"best-arm share {out[:, 1].mean():.0%}")
print(f"uniform A/B split would lose {T * (p.max() - p.mean()):.0f} clicks")
assert res["thompson"] < T * (p.max() - p.mean()) / 2
```
Thompson sampling typically loses the fewest clicks and ends with most traffic on the best headline; UCB1 explores more conservatively; all three beat an equal split, which pays the full gap on every impression.

Contextual version: LinUCB keeps one ridge regression per arm and adds an uncertainty bonus.
```python
import numpy as np

rng = np.random.default_rng(0)
d, K, T, alpha = 5, 4, 5000, 1.0
theta = rng.normal(size=(K, d))                       # hidden per-arm weights
A = np.stack([np.eye(d)] * K); b = np.zeros((K, d))
reward = oracle = 0.0
for t in range(T):
    x = rng.normal(size=d)                            # user/context features
    A_inv = np.linalg.inv(A)
    est = np.einsum("kij,kj->ki", A_inv, b) @ x       # per-arm predicted reward
    bonus = alpha * np.sqrt(np.einsum("i,kij,j->k", x, A_inv, x))
    a = np.argmax(est + bonus)
    r = theta[a] @ x + rng.normal(scale=0.1)
    A[a] += np.outer(x, x); b[a] += r * x
    reward += r; oracle += (theta @ x).max()        # oracle knows theta
print(f"LinUCB avg reward {reward / T:.2f} vs oracle {oracle / T:.2f}")
```
For production contextual bandits, **Vowpal Wabbit** (`--cb_explore_adf` with ε-greedy, softmax, bagging or SquareCB exploration) handles large sparse features, logs propensities and has built-in off-policy evaluation; cloud personalisation services are built on the same idea.

## Choosing / trade-offs
- **Bandit vs A/B test.** Bandit: minimise cost during the experiment, many variants, short-lived content, decision is "serve the best". A/B: unbiased effect sizes, guardrail metrics, a one-time ship decision. A common hybrid: A/B for launches, bandit for continuous creative rotation.
- **Thompson vs UCB vs ε-greedy.** Thompson for binary or Gaussian rewards and batched updates; UCB when you want determinism and simple proofs; ε-greedy when the reward model is a black box (any regressor) and you just need some exploration.
- **Multi-armed vs contextual.** Contextual only pays off when the best arm truly differs by segment and you have enough traffic per context region; otherwise it explores a larger space for nothing.
- **Linear vs neural/tree reward models.** Linear (LinUCB, linear TS) gives calibrated uncertainty cheaply; complex models need bootstrapping or ensembles to estimate uncertainty ([[bayesian-deep-learning-and-uncertainty]]).

## Gotchas
- Not logging propensities makes later offline evaluation impossible; record the probability of the chosen action every time.
- Bandit allocation biases naive per-arm averages and p-values (adaptive sampling); do not run a standard t-test on bandit data ([[statistics-and-ab-testing]]).
- Optimising clicks alone drifts toward clickbait; reward on the business outcome (conversion, retention) or add guardrails.
- Delayed rewards: arms with slow conversions look worse early. Update on matured rewards or model the delay.
- A hard-coded exploration floor of 0 for "known bad" arms means you never notice when they improve; keep minimal exploration under non-stationarity.
- Feedback loops: a recommender that only shows its top picks only learns about its top picks ([[recommender-systems]]).
- Too many arms for the traffic: with 100 variants and few thousand impressions, every algorithm is mostly exploring; prune or share information across arms with features.

## Related
- [[statistics-and-ab-testing]] - fixed-allocation experiments and their inference.
- [[reinforcement-learning]] - the multi-step generalisation; bandits are its one-step case.
- [[recommender-systems]] - exploration and cold start.
- [[causal-inference-and-uplift]] - uplift modelling and off-policy evaluation share propensity weighting.
- [[bayesian-and-gaussian-processes]] - posteriors behind Thompson sampling and GP bandits.
- [[black-box-and-evolutionary-optimization]] - Bayesian optimisation is a bandit over a continuous space.

## References
- Lattimore & Szepesvári, Bandit Algorithms (book, free PDF): https://tor-lattimore.com/downloads/book/book.pdf
- Russo et al., A Tutorial on Thompson Sampling: https://arxiv.org/abs/1707.02038
- Li et al., A Contextual-Bandit Approach to Personalized News Article Recommendation (LinUCB): https://arxiv.org/abs/1003.0146
- Vowpal Wabbit (contextual bandits): https://vowpalwabbit.org/
