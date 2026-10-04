---
title: GRPO from scratch in NumPy (group-relative advantages, clipped ratio, k3 KL, zero-advantage groups)
category: concepts
tags: [grpo, group-relative-policy-optimization, rlvr, verifiable-rewards, ppo, policy-gradient, kl-divergence, reasoning-models, rlhf, numpy, from-scratch]
use_cases:
  - "understand what GRPO computes before running TRL's GRPOTrainer on a reasoning or math task"
  - "pick the group size, KL coefficient and learning rate for RL with verifiable rewards"
  - "debug a GRPO run whose reward stops moving because every group is all-right or all-wrong"
  - "explain how GRPO replaces PPO's value network with a per-prompt group baseline"
status: draft
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/2402.03300
  - https://arxiv.org/abs/2503.20783
  - https://arxiv.org/abs/2503.14476
  - https://huggingface.co/docs/trl/grpo_trainer
---

# GRPO from scratch in NumPy (group-relative advantages, clipped ratio, k3 KL, zero-advantage groups)

## Summary
Group Relative Policy Optimization (GRPO) is PPO without a value network. For each prompt it samples a group of `G` responses, scores them with a reward (often a verifiable 0/1 check: right answer, tests pass), and uses the reward minus the group mean, optionally divided by the group std, as every token's advantage. The policy then takes PPO-style clipped steps, with an optional KL penalty to a reference model. This file trains a toy log-linear policy on 256 "math problems" with one right answer each and shows the behaviours that matter in real runs: groups where every sample gets the same reward carry no gradient, the KL coefficient caps how far the policy can move, and a too-large step collapses the policy into a state it cannot learn out of.

## Key concepts
- **Group baseline.** `A_i = r_i − mean(r_1..r_G)`. The other samples for the *same prompt* are the baseline, so no critic is trained. With 0/1 rewards, a right answer in a group where most are wrong gets a large positive advantage.
- **Std scaling.** The original GRPO divides by the group std. That rescales every prompt's advantages to unit size, which acts as a larger, prompt-dependent learning rate and up-weights prompts the model almost always or almost never solves (the "difficulty bias" Dr. GRPO removes).
- **Zero-advantage groups.** If all `G` samples are right, or all wrong, every advantage is 0 and the prompt contributes nothing. Prompts that are too hard or already solved are wasted compute.
- **Clipped ratio.** As in PPO, `min(ρA, clip(ρ, 1−ε, 1+ε)A)` with `ρ = π/π_old` stops one batch from moving any response's probability too far. It only bites when the batch is reused for more than one step (`num_iterations > 1` in TRL).
- **k3 KL estimator.** `π_ref/π − log(π_ref/π) − 1` per sampled token: unbiased, always positive, computable from the sampled tokens alone. Its gradient with respect to `log π` is `1 − π_ref/π`.

## When to use / scenarios
- Teaching a model a skill whose output can be checked automatically: math answers, code that must pass tests, JSON that must validate, tool calls that must succeed. This is RL with verifiable rewards (RLVR), the recipe behind DeepSeek-R1-style reasoning models.
- When PPO's critic is the memory or stability problem. GRPO keeps only the policy (plus an optional frozen reference).
- Not for offline preference pairs with no reward function. That is DPO's job, see [[dpo-loss-from-scratch-numpy]].
- Not when the base model almost never solves the task. With a 0/1 reward and success rate near 0, nearly every group is all-wrong and GRPO learns nothing. SFT on solved examples first, or use an easier curriculum.

## Setup & code
NumPy only, runs in about a second. Each "response" is one of 16 candidate answers, so `log π(y)` is a single log-softmax and the gradient `∇ log π(y) = φ_y − E_π[φ]` is exact. In an LLM the same per-sample weight is applied to every token of the response.

```python
import numpy as np

rng = np.random.default_rng(0)
P, K, D = 256, 16, 8                     # prompts, candidate answers per prompt, features
phi = rng.normal(size=(P, K, D))         # features of each (prompt, answer)
correct = (phi @ rng.normal(size=D)).argmax(1)   # one verifiable right answer per prompt
theta_ref = rng.normal(scale=0.3, size=D)        # the SFT / reference policy


def logp(theta, idx=slice(None)):
    z = phi[idx] @ theta
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def grpo(steps=300, G=8, B=32, lr=0.1, eps=0.2, beta=0.04, mu=2, std_norm=True, seed=1):
    r_ = np.random.default_rng(seed)
    theta = theta_ref.copy()
    for step in range(steps + 1):
        if step % 100 == 0:
            p = np.exp(logp(theta))
            acc = p[np.arange(P), correct].mean()
            kl = (p * (logp(theta) - logp(theta_ref))).sum(1).mean()
        x = r_.choice(P, B, replace=False)
        pold = np.exp(logp(theta, x))
        y = np.array([r_.choice(K, G, p=pr) for pr in pold])         # G samples per prompt
        r = (y == correct[x, None]).astype(float)                     # verifiable 0/1 reward
        A = r - r.mean(1, keepdims=True)                              # group-relative baseline
        if std_norm:
            A = A / (r.std(1, keepdims=True) + 1e-4)
        dead = np.mean(r.std(1) == 0)                                 # all-right or all-wrong groups
        if step % 100 == 0:
            print(f"  step {step:3d}  expected acc {acc:.2f}  KL {kl:.3f}  zero-adv groups {dead:.2f}")
        lold = np.log(pold[np.arange(B)[:, None], y])
        lref = logp(theta_ref, x)[np.arange(B)[:, None], y]
        for _ in range(mu):                                           # mu updates per batch -> ratio != 1
            lpa = logp(theta, x)
            lp = lpa[np.arange(B)[:, None], y]
            ratio = np.exp(lp - lold)
            clipped = ((A > 0) & (ratio > 1 + eps)) | ((A < 0) & (ratio < 1 - eps))
            w = np.where(clipped, 0.0, ratio * A)                     # d/dlogpi of min(rA, clip(r)A)
            w = w - beta * (1 - np.exp(lref - lp))                    # d/dlogpi of the k3 KL estimate
            # d log pi(y)/d theta = phi_y - E_pi[phi]
            g = phi[x[:, None], y] - (np.exp(lpa)[..., None] * phi[x]).sum(1)[:, None]
            theta += lr * (w[..., None] * g).mean((0, 1))
    return theta


for name, kw in [("GRPO (std-normalised, beta=0.04)", {}),
                 ("Dr. GRPO-style (no std division)", {"std_norm": False}),
                 ("GRPO, beta=0.5", {"beta": 0.5}),
                 ("GRPO, G=2", {"G": 2}),
                 ("GRPO, lr=0.5", {"lr": 0.5})]:
    print(name)
    grpo(**kw)
```

Output (Python 3.14, NumPy 2.5):
```
GRPO (std-normalised, beta=0.04)
  step   0  expected acc 0.05  KL 0.000  zero-adv groups 0.72
  step 100  expected acc 0.71  KL 2.713  zero-adv groups 0.16
  step 200  expected acc 0.76  KL 2.922  zero-adv groups 0.47
  step 300  expected acc 0.80  KL 3.078  zero-adv groups 0.47
Dr. GRPO-style (no std division)
  step   0  expected acc 0.05  KL 0.000  zero-adv groups 0.72
  step 100  expected acc 0.44  KL 1.457  zero-adv groups 0.03
  step 200  expected acc 0.63  KL 2.340  zero-adv groups 0.16
  step 300  expected acc 0.69  KL 2.636  zero-adv groups 0.34
GRPO, beta=0.5
  step   0  expected acc 0.05  KL 0.000  zero-adv groups 0.72
  step 100  expected acc 0.19  KL 0.355  zero-adv groups 0.38
  step 200  expected acc 0.18  KL 0.351  zero-adv groups 0.34
  step 300  expected acc 0.20  KL 0.407  zero-adv groups 0.25
GRPO, G=2
  step   0  expected acc 0.05  KL 0.000  zero-adv groups 0.97
  step 100  expected acc 0.52  KL 1.843  zero-adv groups 0.72
  step 200  expected acc 0.68  KL 2.589  zero-adv groups 0.72
  step 300  expected acc 0.73  KL 2.823  zero-adv groups 0.69
GRPO, lr=0.5
  step   0  expected acc 0.05  KL 0.000  zero-adv groups 0.72
  step 100  expected acc 0.21  KL 4.309  zero-adv groups 1.00
  step 200  expected acc 0.22  KL 4.302  zero-adv groups 1.00
  step 300  expected acc 0.22  KL 4.299  zero-adv groups 0.94
```

How to read it:
- **Default run.** Accuracy climbs from 0.05 to 0.80. At step 0, 72% of groups are all-wrong and teach nothing; at the end, 47% are all-right and teach nothing. In both regimes, a large share of generation compute is wasted.
- **No std division** learns more slowly here at the same learning rate. Its advantages are at most 1 in size instead of about 2.6 for a lone correct sample in 8, so the step is smaller. That is a rescaling, not a worse algorithm: the Dr. GRPO paper argues the std term biases which prompts get weight, and TRL exposes it as `scale_rewards`.
- **β = 0.5** pins the policy near the reference (KL about 0.4) and accuracy stalls near 0.2. The KL term is a leash on the whole run, not only a stabiliser.
- **G = 2** still learns, but most groups are dead at every stage (97% at the start) because two samples agree far more often than eight. Small groups waste fewer tokens per prompt but give a much sparser signal.
- **lr = 0.5** collapses. Within 100 steps the policy puts nearly all its mass on one answer per prompt: right for 21% of prompts, wrong for the rest. Every group is now all-right or all-wrong, the gradient is zero, and the run is stuck for good. Real GRPO runs show this as entropy collapsing to near zero while reward plateaus.

## Choosing / trade-offs
- **Group size `G`.** TRL defaults to `num_generations=8`. Larger groups give more non-zero advantages on hard prompts at a linear cost in generation, which dominates GRPO's run time.
- **KL coefficient β.** TRL now defaults to `beta=0.0` (no KL term, no reference model in memory), following recent reasoning-RL work. Add a small β when the model drifts in style or forgets general skills; large β stops learning, as above.
- **Loss aggregation.** For LLMs the per-token losses must be averaged. TRL offers `loss_type="grpo"` (per-sequence mean, which favours short correct and long wrong answers), `"dr_grpo"` (divide by a constant) and `"dapo"` (token-level mean over the batch, TRL's default). This choice changes how response length evolves during training.
- **Std scaling.** `scale_rewards="group"` (the original), `"batch"`, or off. Turning it off makes the step size depend on the raw reward scale, so retune the learning rate.
- **GRPO vs PPO.** PPO's critic gives per-token credit assignment and works with a single sample per prompt. GRPO trades that for simplicity and memory and needs several samples per prompt. See [[ppo-from-scratch]].
- **GRPO vs DPO.** DPO learns from fixed preference pairs without sampling. GRPO samples from the current policy and needs a reward function. With a reliable automatic reward, online GRPO usually goes further. See [[dpo-loss-from-scratch-numpy]].

## Gotchas
- Filter the prompt set before training. Drop prompts the model always solves or never solves at `G` samples (DAPO's "dynamic sampling"); they cost generation time and contribute zero gradient.
- Reward hacking is the default failure. A reward that checks only the final answer format, or only that tests run, will be gamed. Read samples, not just the reward curve.
- Watch entropy (TRL logs it) and the fraction of groups whose rewards all match. Entropy falling fast while reward is flat is the collapse shown above: lower the learning rate or raise the sampling temperature.
- The clip only matters when a batch is reused (`num_iterations > 1`) or when generation and training run on slightly different weights. With one update per batch, the ratio is 1 and the loss reduces to plain REINFORCE with a group baseline.
- Generation and training must use the same tokenizer, chat template and sampling settings that the reward assumes. A vLLM generation engine with different numerics can make `π_old` drift from the trainer's view of the same weights.

## Related
- [[rlhf-and-preference-optimization]] - where GRPO sits among RLHF methods, with TRL setup.
- [[ppo-from-scratch]] - the clipped objective GRPO keeps and the critic it removes.
- [[reinforce-policy-gradient-from-scratch]] - the score-function gradient underneath, and baselines.
- [[dpo-loss-from-scratch-numpy]] - the offline preference alternative.
- [[information-theory-for-ml]] - the KL divergence the β term penalises.
- [[fine-tuning-and-peft]] - the SFT step that usually comes first.

## References
- Shao et al. (2024), "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" (introduces GRPO): https://arxiv.org/abs/2402.03300
- Liu et al. (2025), "Understanding R1-Zero-Like Training: A Critical Perspective" (Dr. GRPO): https://arxiv.org/abs/2503.20783
- Yu et al. (2025), "DAPO: An Open-Source LLM Reinforcement Learning System at Scale": https://arxiv.org/abs/2503.14476
- Hugging Face TRL, GRPO Trainer: https://huggingface.co/docs/trl/grpo_trainer
