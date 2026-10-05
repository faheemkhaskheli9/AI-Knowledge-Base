---
title: Elo and Bradley-Terry leaderboards from scratch (pairwise model comparisons, MM algorithm, bootstrap CIs)
category: llm-apps
tags: [elo, bradley-terry, pairwise-comparison, leaderboard, chatbot-arena, llm-evaluation, llm-as-judge, ranking, bootstrap, mm-algorithm, numpy, from-scratch]
use_cases:
  - "rank LLMs, prompts or model versions from pairwise A/B preferences (human or LLM-as-judge)"
  - "build an internal arena-style leaderboard with confidence intervals"
  - "understand why Chatbot Arena moved from online Elo to a Bradley-Terry fit"
  - "decide whether two models are really different or within noise on a pairwise eval"
status: draft
last_verified: 2026-10-05
sources:
  - https://doi.org/10.2307/2334029
  - https://doi.org/10.1214/aos/1079120141
  - https://lmsys.org/blog/2023-12-07-leaderboard/
  - https://lmsys.org/blog/2024-08-28-style-control/
  - https://arxiv.org/abs/2403.04132
---

# Elo and Bradley-Terry leaderboards from scratch (pairwise model comparisons, MM algorithm, bootstrap CIs)

## Summary
Absolute scores for open-ended LLM output are hard to define, but "which of these two answers is better?" is easy to ask a person or a judge model. A leaderboard turns many such pairwise votes into one rating per model. Online Elo, from chess, updates two ratings after each game. The Bradley-Terry (BT) model fits all votes at once by maximum likelihood. Both assume `P(i beats j) = 1 / (1 + 10^((R_j − R_i)/400))`, but they behave differently on a fixed set of votes. This file implements both in NumPy on simulated votes from 8 models with known strengths, where the strongest models join halfway, as new releases do. Online Elo's result depends on the order of the votes and on its step size `k`; BT does not, is closer to the truth, and gives bootstrap confidence intervals. That is why LMSYS's Chatbot Arena switched to BT in December 2023.

## Key concepts
- **Bradley-Terry model.** Each item has a strength `p_i`, and `P(i beats j) = p_i / (p_i + p_j)`. With `p_i = e^{R_i/s}` and `s = 400/ln 10` this is exactly the Elo formula. It is a logistic regression on rating differences.
- **Online Elo.** After each game, `R_i += k(S − E)`, where `S` is the result (1/0) and `E` the predicted win probability. It is stochastic gradient ascent on the BT log-likelihood with a constant step, so it never settles: recent games weigh more, and the final numbers depend on the order of the games.
- **BT maximum likelihood.** Fit all ratings at once. Hunter's MM update, `p_i ← W_i / Σ_j n_ij/(p_i + p_j)` (`W_i` = wins of `i`, `n_ij` = games between `i` and `j`), increases the likelihood every step and needs no learning rate. Ratings are only defined up to a common shift, so fix the mean.
- **Bootstrap CIs.** Resample the votes with replacement, refit, and take percentiles. Two models with overlapping intervals are not reliably ordered.
- **Connectivity.** BT ratings only compare models linked by a chain of games. A model never compared with the rest has no defined rating.

## When to use / scenarios
- Internal model selection: compare candidate models, fine-tunes or prompt versions on your own prompts with pairwise human or LLM-as-judge preferences ([[llm-evaluation]]).
- Arena-style public or team leaderboards, where models are added over time and votes arrive in no particular order.
- Ranking anything judged in pairs: summaries, retrieval results, generated images, agent trajectories.
- Not when a reference answer or a checkable outcome exists (unit tests, exact match, extraction accuracy): score each output directly; it is cheaper and less noisy.
- Not for adaptive, live matchmaking where the rating must move as players change (games, chess): there, online Elo or Glicko is the point, not a flaw.

## Setup & code
NumPy only, runs in about 3 s. Eight models with known ratings; 6,000 votes drawn from the BT model. The three strongest models (`m-a` to `m-c`) only appear in the second half. Online Elo is run with `k = 32` (a common chess value) and with a small `k = 4`. BT is fitted by the MM algorithm, with 200 bootstrap refits for 95% intervals.

```python
import numpy as np

rng = np.random.default_rng(0)
names = ["m-a", "m-b", "m-c", "m-d", "m-e", "m-f", "m-g", "m-h"]
true = np.array([1250, 1200, 1180, 1100, 1060, 1050, 980, 900], float)   # true ratings, Elo scale
M = len(names)
SCALE = 400 / np.log(10)                     # Elo: P(i beats j) = 1 / (1 + 10^((Rj - Ri)/400))


def p_win(ri, rj):
    return 1 / (1 + 10 ** ((rj - ri) / 400))


def battles(n, drift=True):
    """Simulated arena votes. Strong models join late and get many battles (as on a real leaderboard)."""
    out = []
    for t in range(n):
        live = np.arange(M) if (not drift or t > n // 2) else np.arange(3, M)  # m-a..m-c arrive halfway
        i, j = rng.choice(live, 2, replace=False)
        out.append((i, j, rng.random() < p_win(true[i], true[j])))
    return np.array(out, dtype=int)


def online_elo(b, k=32, init=1000):
    r = np.full(M, float(init))
    for i, j, a_wins in b:
        e = p_win(r[i], r[j])
        r[i] += k * (a_wins - e)
        r[j] -= k * (a_wins - e)
    return r


def bradley_terry(b, iters=200, init=1000):
    """MLE of Bradley-Terry strengths with Hunter's (2004) MM updates; returned on the Elo scale."""
    wins = np.zeros((M, M))
    for i, j, a_wins in b:
        wins[i, j] += a_wins
        wins[j, i] += 1 - a_wins
    n = wins + wins.T                                 # games between each pair
    w = wins.sum(1)
    p = np.ones(M)
    for _ in range(iters):
        p = w / (n / (p[:, None] + p[None, :])).sum(1)
        p /= np.exp(np.log(p).mean())                 # fix the scale: ratings are only defined up to a shift
    r = SCALE * np.log(p)
    return r - r.mean() + init


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


def centred(r):
    return r - r.mean()


b = battles(6000)
print("model  true  | Elo k=32  Elo k=4 | BT MLE  95% bootstrap CI")
elo32, elo4, bt = online_elo(b, 32), online_elo(b, 4), bradley_terry(b)
boot = np.array([bradley_terry(b[rng.integers(0, len(b), len(b))]) for _ in range(200)])
lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
tc = centred(true) + 1000
for m in range(M):
    print(f"{names[m]:5s} {tc[m]:5.0f} | {elo32[m]:8.0f} {elo4[m]:8.0f} | {bt[m]:6.0f}  [{lo[m]:5.0f}, {hi[m]:5.0f}]")

print("\nmean abs error vs truth (centred) and rank agreement:")
for name, r in (("Elo k=32", elo32), ("Elo k=4", elo4), ("BT MLE", bt)):
    print(f"{name:9s} MAE {np.abs(centred(r) - centred(true)).mean():5.1f}   Spearman {spearman(r, true):.3f}")

# Order dependence: same votes, shuffled order
spread = {}
for name, fn in (("Elo k=32", lambda x: online_elo(x, 32)), ("Elo k=4", lambda x: online_elo(x, 4)),
                 ("BT MLE", bradley_terry)):
    runs = np.array([fn(b[rng.permutation(len(b))]) for _ in range(50)])
    spread[name] = runs.std(0).mean()
print("\nstd of a model's rating over 50 shuffles of the same votes:",
      ", ".join(f"{k} {v:.1f}" for k, v in spread.items()))
```

Output (Python 3.14, NumPy 2.5):
```
model  true  | Elo k=32  Elo k=4 | BT MLE  95% bootstrap CI
m-a    1160 |     1177     1163 |   1153  [ 1130,  1179]
m-b    1110 |     1152     1130 |   1133  [ 1103,  1157]
m-c    1090 |     1037     1061 |   1085  [ 1060,  1110]
m-d    1010 |     1063     1017 |   1004  [  989,  1018]
m-e     970 |      947      963 |    971  [  952,   987]
m-f     960 |      873      941 |    968  [  951,   982]
m-g     890 |      834      889 |    872  [  858,   888]
m-h     810 |      916      835 |    815  [  796,   831]

mean abs error vs truth (centred) and rank agreement:
Elo k=32  MAE  54.8   Spearman 0.905
Elo k=4   MAE  14.0   Spearman 1.000
BT MLE    MAE   9.3   Spearman 1.000

std of a model's rating over 50 shuffles of the same votes: Elo k=32 49.3, Elo k=4 15.3, BT MLE 0.0
```

How to read it:
- True ratings are centred on 1000 to match the fits, since only differences are identified.
- **Elo with k = 32 misranks.** It puts the weakest model `m-h` (true 810) at 916, above `m-f` and `m-g`, and swaps `m-c` and `m-d`. Its MAE is 55 rating points. With a large step, each rating mostly reflects the last few dozen games.
- **A smaller k helps but does not fix the method.** `k = 4` gets the order right with MAE 14, but the late arrivals `m-a` to `m-c` start at 1000 and have only had half the games to climb; with fewer votes they would still be too low.
- **Order dependence.** Shuffling the same 6,000 votes moves each Elo rating by 49 points (k = 32) or 15 (k = 4) on average. BT gives identical numbers for every order, because it only sees the win counts.
- **BT** is within 9 points of the truth on average, and 7 of the 8 true ratings fall inside its 95% bootstrap intervals (`m-g` misses by 2 points, about what 95% coverage predicts). The intervals of `m-e` and `m-f` overlap heavily (true gap 10): on this data they are a tie, and a leaderboard should say so.

## Choosing / trade-offs
- **Online Elo vs BT.** For ranking static models from a fixed pool of votes, use BT. Chatbot Arena first stabilised online Elo by averaging it over 1,000 shuffles of the votes, found the resulting intervals too wide, and switched to BT in December 2023. Online Elo suits players whose skill changes over time; its order dependence is a feature there.
- **Ties.** Arena votes allow ties. Count a tie as half a win for each side (simple), or use a model with an explicit tie parameter (Rao-Kupper, Davidson).
- **Votes needed.** Interval width shrinks roughly with `1/√(votes per model)`. Add votes until the models you need to separate have non-overlapping intervals, and pair close models more often (active sampling) rather than sampling pairs uniformly.
- **Confounders.** LLM-judge and human preferences depend on answer length, formatting and position ([[llm-evaluation]]). Chatbot Arena adds style features (answer length, markdown headers, bold text, list items) to the BT regression to report a style-controlled ranking, and the ranking changes when it does. Swap answer order in judge prompts and log length so you can do the same.
- **Fitting tool.** MM is a few lines and needs no learning rate. Equivalently, fit `LogisticRegression(fit_intercept=False)` on rows with `+1` for the first model and `−1` for the second; a small L2 penalty keeps ratings finite when a model wins or loses every game.

## Gotchas
- A model that won (or lost) every game has an infinite MLE rating. In the code above, a model with no wins gets `p = 0` after one MM step and the log turns every rating into `-inf`/NaN; a model with no losses climbs a little more on every iteration. Add a weak prior (a few pseudo-games against an average model) or regularise.
- Ratings from different fits or different vote pools are not comparable: the anchor (here, mean = 1000) is arbitrary. Report differences or win probabilities against a fixed reference model.
- Uniform pair sampling wastes votes on obvious mismatches. On a real arena the pairing scheme also biases which prompts each model sees, so condition on prompt categories when you can.
- One vote per pair from a strongly biased judge is not independent evidence. Average both answer orders per comparison before counting it as a win.
- Do not compute Elo once in vote order and publish it. As shown, a reshuffle can reorder the leaderboard.

## Related
- [[llm-evaluation]] - LLM-as-judge, pairwise evals and judge biases that feed this leaderboard.
- [[dpo-loss-from-scratch-numpy]] - the same Bradley-Terry likelihood used for preference training.
- [[logistic-regression-from-scratch]] - BT is a logistic regression on rating differences.
- [[thompson-sampling-and-ucb-from-scratch]] - choosing which pairs to compare next (active sampling).
- [[reciprocal-rank-fusion-from-scratch]] - another way to merge rankings, without a probability model.

## References
- Bradley & Terry (1952), "Rank analysis of incomplete block designs: I. The method of paired comparisons", Biometrika 39: https://doi.org/10.2307/2334029
- Hunter (2004), "MM algorithms for generalized Bradley-Terry models", Annals of Statistics 32(1): https://doi.org/10.1214/aos/1079120141
- LMSYS (2023-12-07), "Chatbot Arena: New models & Elo system update" (move from online Elo to Bradley-Terry): https://lmsys.org/blog/2023-12-07-leaderboard/
- LMSYS (2024-08-28), "Does style matter? Disentangling style and substance in Chatbot Arena": https://lmsys.org/blog/2024-08-28-style-control/
- Chiang et al. (2024), "Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference": https://arxiv.org/abs/2403.04132
