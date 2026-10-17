---
title: Reward modeling and verifiers
category: concepts
tags: [reward-model, verifier, prm, orm, bradley-terry, best-of-n, reward-hacking, rlvr, llm-as-judge]
use_cases:
  - "train a reward model from thumbs-up/down or pairwise preference data to rerank chatbot answers"
  - "pick the best of n sampled solutions to a math or coding task with a verifier"
  - "decide between an outcome reward model, a process reward model and unit-test verification for RL training"
  - "detect and limit reward hacking when optimising a model against a learned reward"
status: draft
last_verified: 2026-10-15
sources:
  - https://arxiv.org/abs/2110.14168
  - https://arxiv.org/abs/2211.14275
  - https://arxiv.org/abs/2305.20050
  - https://arxiv.org/abs/2312.08935
  - https://arxiv.org/abs/2408.15240
  - https://arxiv.org/abs/2210.10760
  - https://arxiv.org/abs/2403.13787
  - https://huggingface.co/docs/trl/reward_trainer
---

# Reward modeling and verifiers

## Summary
A reward model or verifier scores a model output so you can rank, filter or train on it. Learned reward models predict human preference (used in RLHF and best-of-n reranking); verifiers check correctness, either with a learned model or a program (unit tests, exact-match answers). The key choices are what to score (the final outcome or each reasoning step), learned versus programmatic checks, and how hard to optimise against the score, because any imperfect reward gets exploited ("reward hacking").

## Key concepts
- **Bradley-Terry reward model**: an LLM with a scalar head trained on (prompt, chosen, rejected) pairs with loss `-log sigmoid(r(chosen) - r(rejected))`. Only differences in reward are meaningful.
- **Outcome reward model (ORM)**: scores a whole solution (e.g. "is the final answer right?"). Cheap labels; credit assignment is coarse.
- **Process reward model (PRM)**: scores each reasoning step. Lightman et al. found process supervision beat outcome supervision for selecting solutions on MATH and released PRM800K step labels; Uesato et al. found similar final-answer accuracy but fewer reasoning errors with process feedback. Step labels are expensive; Math-Shepherd derives them automatically from rollouts.
- **Generative verifiers / LLM-as-judge**: the verifier is itself an LLM that writes a critique or a yes/no token, so it can use chain-of-thought and voting (GenRM).
- **Programmatic verifiers**: unit tests, compilers, SQL result comparison, symbolic math equivalence, schema validation. Basis of RL with verifiable rewards (RLVR; see [[rlhf-and-preference-optimization]]).
- **Uses**: reranking (best-of-n), filtering synthetic data (rejection sampling), RL training signal, guiding search over steps ([[reasoning-models]]).
- **Reward hacking / over-optimisation**: optimising hard against a proxy reward raises the proxy while the true objective peaks and then falls (Gao et al.). Classic exploits: length, flattery, format tricks, tests that check too little.

## When to use / scenarios
- Preference data exists (A/B choices, thumbs) and you want to rerank or train toward it: Bradley-Terry RM.
- Tasks with checkable answers (code, maths, SQL, extraction against a gold table): programmatic verifier first; it is cheaper and harder to fool than a learned one.
- Long multi-step reasoning where wrong intermediate steps matter (tutoring, audits): PRM or step-level judge.
- Real-world: a coding assistant samples 8 patches and keeps the first that passes the repo's tests; a tutoring product flags the first wrong step in a student-style solution; a support bot reranks drafts with an RM trained on agent edits.
- Not when: there is no reliable signal (a noisy RM adds bias, not quality), or a simple rule or eval metric already captures what you want.

## Setup & code
A Bradley-Terry reward model learned from biased labels, best-of-n against it, and a unit-test verifier that is too weak. NumPy and stdlib only.

```python
import numpy as np

rng = np.random.default_rng(0)


# --- 1. A Bradley-Terry reward model learned from pairwise preferences -------------
# Each response has two features: quality q and length l (both standardised).
# The true reward likes quality and dislikes very long answers.
def true_reward(q, l):
    return q - 0.5 * np.maximum(l, 0) ** 2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


n = 5000
qa, la, qb, lb = rng.normal(size=(4, n))
# Labellers mostly follow the true reward but also favour longer answers (a known bias).
p_a = sigmoid(true_reward(qa, la) - true_reward(qb, lb) + 1.5 * (la - lb))
y = (rng.random(n) < p_a).astype(float)        # 1 if response A was preferred

# Linear reward model r(x) = w . [q, l]; BT loss = -log sigmoid(r(chosen) - r(rejected)).
dx = np.stack([qa - qb, la - lb], axis=1)
w = np.zeros(2)
for _ in range(2000):                           # plain gradient descent on the BT loss
    w -= 0.5 * dx.T @ (sigmoid(dx @ w) - y) / n
print("learned reward weights [quality, length]:", np.round(w, 2))
acc = np.mean((dx @ w > 0) == (y == 1))
print(f"pairwise accuracy on its own training pairs: {acc:.0%}")

# --- 2. Best-of-n against the learned (proxy) reward: Goodhart in action ------------
for n_samples in (1, 4, 16, 64, 256):
    q, l = rng.normal(size=(2, 4000, n_samples))
    proxy = w[0] * q + w[1] * l
    pick = proxy.argmax(axis=1)                 # choose the sample the RM likes best
    rows = np.arange(4000)
    print(f"best-of-{n_samples:<3d} proxy reward {proxy[rows, pick].mean():5.2f}   "
          f"true reward {true_reward(q[rows, pick], l[rows, pick]).mean():5.2f}")


# --- 3. An outcome verifier for code: run the candidate against unit tests ----------
candidates = {
    "correct": "def median(xs):\n    s = sorted(xs)\n    m = len(s) // 2\n"
               "    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2",
    "off_by_one": "def median(xs):\n    s = sorted(xs)\n    return s[len(s) // 2]",
    "hacks_the_test": "def median(xs):\n    return 2",
}
weak_tests = [([1, 2, 3], 2)]                   # a verifier that is too easy to game
strong_tests = weak_tests + [([4, 1, 3, 2], 2.5), ([7], 7), ([5, 5, 1], 5)]


def verify(src, tests):
    ns = {}
    exec(src, ns)                               # real systems: run in a sandbox with limits
    return all(ns["median"](list(x)) == want for x, want in tests)


for name, src in candidates.items():
    print(f"{name:15s} weak verifier: {verify(src, weak_tests)!s:5s}  "
          f"strong verifier: {verify(src, strong_tests)}")
```
The labellers' length bias leaks into the reward model, so best-of-n raises the proxy reward steadily while the true reward first rises and then falls. The weak verifier accepts a solution that hard-codes the test answer; more and more varied tests close that hole.

For real models, TRL's `RewardTrainer` trains a sequence-classification model on chosen/rejected pairs (see its docs for the dataset format of your TRL version). Score candidates with `AutoModelForSequenceClassification` and take the argmax for best-of-n. Run code verifiers in a sandbox ([[sandboxed-code-execution]]).

## Choosing / trade-offs
| Signal | Label cost | Hackability | Best for |
|---|---|---|---|
| Programmatic verifier | Low once written | Low if tests are thorough | Code, maths, SQL, formats |
| ORM | Low (final answer) | Medium | Reranking, data filtering |
| PRM | High (step labels) or rollout-derived | Medium | Long reasoning, search |
| Bradley-Terry RM | Medium (pairs) | High (style, length) | Open-ended quality, tone |
| LLM-as-judge | None to train | Medium; judge biases | Fast start, rubrics |

- Optimisation pressure: best-of-n with small n, or RL with a KL penalty to a reference model, limits over-optimisation; large n or long RL runs need a held-out gold signal to catch the peak.
- One general RM vs task-specific RMs: general RMs miss domain errors; task RMs need their own data.

## Gotchas
- Pairwise accuracy of an RM says little about how it behaves under optimisation; track a held-out true metric while optimising.
- RMs and judges favour length, confident tone and their own model family's style; control for length in evaluation.
- Tests written by the same model that writes the code can share its bugs; keep hidden tests the policy never sees.
- Never `exec` model-written code in the serving process; the demo does it only on fixed strings.
- RM scores drift as the policy moves off the RM's training distribution; refresh preference data from the current policy.
- Public RM leaderboards (e.g. RewardBench) measure agreement with their pairs, not your domain.

## Related
- [[rlhf-and-preference-optimization]] - where reward models are used for training, and RLVR.
- [[grpo-from-scratch-numpy]] - an RL loop driven by a verifier reward.
- [[dpo-loss-from-scratch-numpy]] - preference optimisation with an implicit reward model.
- [[reasoning-models]] - verifiers for test-time search and training.
- [[self-consistency-voting-from-scratch]] - majority voting, the verifier-free alternative to best-of-n.
- [[elo-and-bradley-terry-leaderboard-from-scratch]] - the same Bradley-Terry model used for leaderboards.
- [[llm-evaluation]] - LLM-as-judge and its biases.
- [[sandboxed-code-execution]] - running code verifiers safely.
- [[coding-agents]] - test-based verification in agent loops.

## References
- Cobbe et al., Training Verifiers to Solve Math Word Problems: https://arxiv.org/abs/2110.14168
- Uesato et al., Solving math word problems with process- and outcome-based feedback: https://arxiv.org/abs/2211.14275
- Lightman et al., Let's Verify Step by Step: https://arxiv.org/abs/2305.20050
- Wang et al., Math-Shepherd: https://arxiv.org/abs/2312.08935
- Zhang et al., Generative Verifiers: https://arxiv.org/abs/2408.15240
- Gao et al., Scaling Laws for Reward Model Overoptimization: https://arxiv.org/abs/2210.10760
- Lambert et al., RewardBench: https://arxiv.org/abs/2403.13787
- TRL Reward Trainer: https://huggingface.co/docs/trl/reward_trainer
