---
title: DPO loss from scratch in NumPy (beta as the KL leash, match with the RLHF optimum)
category: concepts
tags: [dpo, direct-preference-optimization, rlhf, bradley-terry, preference-data, kl-divergence, alignment, numpy, from-scratch]
use_cases:
  - "understand what the DPO loss computes before running TRL's DPOTrainer"
  - "choose beta for DPO and predict how far the policy drifts from the SFT model"
  - "explain why DPO needs no reward model and no sampling during training"
  - "debug a DPO run where the chosen responses' log-probabilities go down"
status: draft
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/2305.18290
  - https://arxiv.org/abs/2310.12036
  - https://arxiv.org/abs/2402.13228
  - https://huggingface.co/docs/trl/dpo_trainer
---

# DPO loss from scratch in NumPy (beta as the KL leash, match with the RLHF optimum)

## Summary
Direct Preference Optimization (DPO) trains a policy on pairs of (chosen, rejected) responses with a single classification-style loss: `-log σ(β [(log π(y_w) − log π_ref(y_w)) − (log π(y_l) − log π_ref(y_l))])`. It needs no reward model and no sampling, because the KL-regularised RLHF objective has a closed-form optimum `π* ∝ π_ref · exp(r / β)` that lets the reward be written in terms of the policy. This file implements the loss and its gradient for a toy log-linear policy and checks the claim numerically: DPO trained on noisy pairwise labels lands on the same reward and KL as the closed-form RLHF optimum computed with the true reward, for every `β`. It also reproduces a known DPO quirk: the log-probability of the chosen responses usually goes down.

## Key concepts
- **Bradley-Terry preferences.** Annotators prefer `y_w` over `y_l` with probability `σ(r(y_w) − r(y_l))`. A reward model is a logistic regression on that difference.
- **Implicit reward.** DPO substitutes `r(x, y) = β log(π(y|x) / π_ref(y|x)) + const`. The constant (the partition function) cancels inside a pair, which is why DPO never needs it.
- **The loss** is binary cross-entropy on the implicit-reward margin of each pair. Its gradient is `β σ(−margin) (∇log π(y_w) − ∇log π(y_l))`: pairs the model already ranks correctly get little weight.
- **β is the KL leash.** The optimum is `π_ref · exp(r / β)`. Small β lets the policy move far from the reference (high reward, high KL). Large β keeps it close.
- **Only the margin is constrained.** DPO raises the chosen response *relative* to the rejected one. Both can fall in absolute probability, with mass moving to responses that appear in no pair.

## When to use / scenarios
- Aligning an SFT model on pairwise preference data (helpfulness, tone, format, refusals) with a modest GPU budget: DPO trains like supervised fine-tuning, with one extra frozen reference model forward pass.
- Offline preference data you already have (A/B choices from users, rankings from labellers, LLM-as-judge comparisons).
- Not when the reward is verifiable (unit tests, exact math answers). Online RL with sampled responses (GRPO, PPO) uses that signal directly, see [[rlhf-and-preference-optimization]] and [[ppo-from-scratch]].
- Not when the pairs come from a very different model than the one you train. DPO only sees the responses in the data, so off-distribution pairs teach little about the policy's own outputs. Regenerate pairs from the current policy (iterative / online DPO).

## Setup & code
NumPy only, runs in about a second. The policy is log-linear over 8 candidate responses per prompt, so the exact RLHF optimum can be written down and compared.

```python
import numpy as np

rng = np.random.default_rng(0)
P, K, D = 200, 8, 6                      # prompts, candidate responses per prompt, features
phi = rng.normal(size=(P, K, D))         # features of each (prompt, response)
w_true = rng.normal(size=D)              # hidden "human" reward r = phi @ w_true
theta_ref = rng.normal(scale=0.5, size=D)  # SFT / reference policy


def logp(theta):
    z = phi @ theta
    return z - np.log(np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)) - z.max(1, keepdims=True)


def sample_pairs(n):
    """Two responses drawn from the reference policy, labelled by a Bradley-Terry annotator."""
    pr = np.exp(logp(theta_ref))
    x = rng.integers(0, P, n)
    a = np.array([rng.choice(K, p=pr[i]) for i in x])
    b = np.array([rng.choice(K, p=pr[i]) for i in x])
    keep = a != b
    x, a, b = x[keep], a[keep], b[keep]
    r = phi @ w_true
    a_wins = rng.random(len(x)) < 1 / (1 + np.exp(-(r[x, a] - r[x, b])))
    return x, np.where(a_wins, a, b), np.where(a_wins, b, a)   # prompt, chosen, rejected


def dpo(beta, x, yw, yl, steps=500):
    lr = 0.5 / beta**2                               # curvature scales with beta^2
    theta = theta_ref.copy()
    lref = logp(theta_ref)
    for _ in range(steps):
        lp = logp(theta)
        margin = beta * ((lp[x, yw] - lref[x, yw]) - (lp[x, yl] - lref[x, yl]))
        g = 1 / (1 + np.exp(margin))                 # sigma(-margin): weight of each pair
        # d log pi(y)/d theta = phi_y - E_pi[phi]; the expectation cancels inside a pair
        grad = -beta * (g[:, None] * (phi[x, yw] - phi[x, yl])).mean(0)
        theta -= lr * grad
    return theta


def report(name, theta, x, yw, yl):
    lp, lref = logp(theta), logp(theta_ref)
    p, pref = np.exp(lp), np.exp(lref)
    reward = (p * (phi @ w_true)).sum(1).mean()
    kl = (p * (lp - lref)).sum(1).mean()
    m = (lp[x, yw] - lref[x, yw]) - (lp[x, yl] - lref[x, yl])
    acc = np.mean(m > 0) + 0.5 * np.mean(m == 0)     # ties count as a coin flip
    dchosen = np.mean(lp[x, yw] - lref[x, yw])
    print(f"{name:12s} true reward {reward:5.2f}  KL(pi||ref) {kl:5.2f}  "
          f"held-out pref acc {acc:.2f}  mean d log pi(chosen) {dchosen:+.2f}")


tr, te = sample_pairs(2000), sample_pairs(2000)
report("reference", theta_ref, *te)
for beta in [0.25, 0.5, 1.0, 2.0]:
    report(f"dpo b={beta}", dpo(beta, *tr), *te)
    # closed-form KL-regularised RLHF optimum with the TRUE reward: pi* ∝ pi_ref exp(r / beta)
    report(f"rlhf* b={beta}", theta_ref + w_true / beta, *te)
```

Output (Python 3.14, NumPy 2.5):
```
reference    true reward -0.79  KL(pi||ref)  0.00  held-out pref acc 0.50  mean d log pi(chosen) +0.00
dpo b=0.25   true reward  2.75  KL(pi||ref)  2.77  held-out pref acc 0.82  mean d log pi(chosen) -7.53
rlhf* b=0.25 true reward  2.75  KL(pi||ref)  2.77  held-out pref acc 0.82  mean d log pi(chosen) -7.13
dpo b=0.5    true reward  2.53  KL(pi||ref)  2.19  held-out pref acc 0.82  mean d log pi(chosen) -2.54
rlhf* b=0.5  true reward  2.52  KL(pi||ref)  2.16  held-out pref acc 0.82  mean d log pi(chosen) -2.35
dpo b=1.0    true reward  1.84  KL(pi||ref)  1.22  held-out pref acc 0.82  mean d log pi(chosen) -0.46
rlhf* b=1.0  true reward  1.79  KL(pi||ref)  1.16  held-out pref acc 0.82  mean d log pi(chosen) -0.39
dpo b=2.0    true reward  0.77  KL(pi||ref)  0.41  held-out pref acc 0.82  mean d log pi(chosen) +0.13
rlhf* b=2.0  true reward  0.71  KL(pi||ref)  0.38  held-out pref acc 0.82  mean d log pi(chosen) +0.14
```

Each DPO row matches the closed-form RLHF optimum for the same β to within the noise of 2,000 labelled pairs. DPO never saw the true reward or sampled a response. β trades reward against drift: from β = 2 to β = 0.25 the true reward rises from 0.77 to 2.75 while KL from the reference grows sevenfold. Held-out preference accuracy stays at 0.82 for every β because β only rescales the implicit reward, it does not change the ranking. 0.82 is the ceiling set by the annotators' own Bradley-Terry noise. The last column shows the quirk: for small β the chosen responses lose log-probability on average, because the policy moves its mass to the highest-reward responses, which are rarely the ones sampled into pairs.

The `lr = 0.5 / β²` line keeps the comparison fair. The loss curvature scales with β², so with a fixed learning rate and step count, small-β runs stop early and look like they have *less* KL, the opposite of their optimum. Real runs with a fixed LR hit the same effect.

## Choosing / trade-offs
- **β.** TRL's default is `beta=0.1`. Lower it to move further from the SFT model, raise it when the model drifts (verbosity, broken formatting, forgotten skills). Watch the KL or the reward margins TRL logs, not only the loss.
- **DPO vs PPO/GRPO.** DPO is cheaper and more stable: no reward model, no generation in the loop. Online RL explores the policy's own outputs and can beat DPO when a reliable reward exists. See [[rlhf-and-preference-optimization]].
- **Variants.** IPO replaces the log-sigmoid with a squared loss so the margin cannot grow without bound on deterministic preferences. Conservative DPO / label smoothing assumes a fraction of labels are flipped. DPOP adds a penalty that stops the chosen log-probability from falling. KTO works with unpaired thumbs-up/down data. ORPO and SimPO drop the reference model.
- **Reference model.** Usually the SFT checkpoint. Its log-probabilities can be precomputed once to save memory (`precompute_ref_log_probs` in TRL).
- **Data.** Pairs sampled from the model being trained (on-policy) work better than pairs from another model. Fewer high-margin, clean pairs beat many near-ties.

## Gotchas
- `log π(y|x)` is the **sum** of token log-probabilities over the response only. Including prompt tokens, or averaging instead of summing, silently changes the objective (averaging is a different method, closer to SimPO).
- A falling chosen log-probability is expected, not a bug, as long as the margin grows and generations look right. If generations degrade, raise β, add an SFT term on chosen responses, or use DPOP.
- On data where the same response always wins, the log-sigmoid keeps pushing the margin toward infinity. Stop early, use IPO or add label smoothing.
- Long chosen responses get larger summed log-probability gaps, so DPO can learn "longer is better". Check length statistics of chosen vs rejected before training, and length-normalise or balance the data if they differ.
- The tokenizer and chat template must be identical to SFT. A template mismatch changes every log-probability and the reference ratio is meaningless.
- With LoRA, the reference model can be the base model with adapters disabled, which TRL does automatically, so you do not need a second copy in memory.

## Related
- [[rlhf-and-preference-optimization]] - where DPO sits among RLHF, GRPO and other preference methods, with TRL setup.
- [[ppo-from-scratch]] - the online RL alternative DPO replaces.
- [[logistic-regression-from-scratch]] - DPO is logistic regression on implicit-reward differences.
- [[information-theory-for-ml]] - KL divergence, the quantity β controls.
- [[fine-tuning-and-peft]] - SFT and LoRA, the usual step before DPO.
- [[lora-from-scratch-numpy]] - the adapters most DPO runs train.

## References
- Rafailov et al. (2023), "Direct Preference Optimization: Your Language Model is Secretly a Reward Model": https://arxiv.org/abs/2305.18290
- Azar et al. (2023), "A General Theoretical Paradigm to Understand Learning from Human Preferences" (IPO): https://arxiv.org/abs/2310.12036
- Pal et al. (2024), "Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive": https://arxiv.org/abs/2402.13228
- Hugging Face TRL, DPO Trainer: https://huggingface.co/docs/trl/dpo_trainer
