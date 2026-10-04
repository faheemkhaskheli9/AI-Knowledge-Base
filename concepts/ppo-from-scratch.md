---
title: PPO from scratch (clipped surrogate, GAE, CartPole in NumPy)
category: concepts
tags: [ppo, proximal-policy-optimization, policy-gradient, actor-critic, gae, trust-region, clipping, reinforcement-learning, cartpole, numpy, from-scratch, deep-learning-basics]
use_cases:
  - "implement PPO's clipped surrogate objective and GAE in NumPy without an RL library"
  - "see why PPO clips the probability ratio: what goes wrong when you reuse a batch for many epochs without it"
  - "explain the PPO objective, the ratio, GAE lambda and the KL / clip-fraction diagnostics in an interview"
  - "understand the RL step behind RLHF before reading an LLM PPO implementation"
status: draft
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/1707.06347
  - https://arxiv.org/abs/1506.02438
  - https://spinningup.openai.com/en/latest/algorithms/ppo.html
  - https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
---

# PPO from scratch (clipped surrogate, GAE, CartPole in NumPy)

## Summary
Proximal Policy Optimization (Schulman et al., 2017) is the actor-critic method that [[actor-critic-from-scratch]] points toward. It collects a batch with the current policy, then takes many minibatch gradient steps on that same batch. Reusing data that way is what makes it sample-efficient, but after a few steps the policy is no longer the one that collected the data. PPO keeps it close: the objective uses the probability ratio `r = π_new(a|s) / π_old(a|s)` and stops rewarding any change that pushes `r` outside `[1 − ε, 1 + ε]`. This file implements PPO-clip with GAE on the same NumPy CartPole simulator and logistic policy as the REINFORCE and actor-critic files, then runs it with and without the clip at three actor learning rates. At a small step size the clip never activates and both versions match (176 vs 175 steps). At a large one, the biggest single policy update without the clip has KL 1.32 from the data-collecting policy. With the clip it stays at 0.014, and the worst seed finishes at 193 steps against 103.

## Key concepts
- **Ratio.** `r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)`, computed as `exp(log π_θ − log π_old)`. The plain surrogate `E[r_t · A_t]` has the same gradient as the policy gradient at `θ = θ_old`, and lets you keep optimising the same batch afterwards.
- **Clipped surrogate.** `L = E[min(r_t A_t, clip(r_t, 1 − ε, 1 + ε) A_t)]`, with `ε = 0.2` by default. For a good action (`A > 0`) there is no further gain once `r > 1 + ε`; for a bad action (`A < 0`) none once `r < 1 − ε`. The `min` makes it a pessimistic bound: moves that make the objective worse are never clipped.
- **What the clip does to the gradient.** A sample whose ratio is out of range on the profitable side contributes zero gradient. That is all the implementation needs (the `out` mask below).
- **GAE.** `A_t = Σ (γλ)^k δ_{t+k}` with TD errors `δ_t = r_t + γV(s_{t+1}) − V(s_t)`. `λ = 0` is one-step TD, `λ = 1` is Monte Carlo; PPO's default is `λ = 0.95`. Computed backwards in one pass per episode.
- **Value target.** `A_t + V(s_t)` (the λ-return). The critic regresses onto it with squared error.
- **Trust region, cheaply.** TRPO enforces `KL(π_old || π_new) ≤ δ` with a second-order constrained step. PPO gets similar behaviour with first-order SGD and a clip, which is why it became the default.

## When to use / scenarios
- Learning: the algorithm behind most on-policy deep RL baselines and the original RLHF pipeline (InstructGPT).
- Interviews: "write the PPO objective", "why clip the ratio instead of the gradient", "what does λ in GAE trade off", "what are clip fraction and approximate KL for", "PPO vs TRPO".
- Reading LLM RL code (TRL, OpenRLHF): the token-level loss is this same clipped ratio, with a KL penalty to a reference model added to the reward.
- Not for: production RL (use Stable-Baselines3 or CleanRL PPO, see [[reinforcement-learning]]), or sample-starved continuous control where off-policy SAC/TD3 learn far faster.

## Setup & code
`pip install numpy`. About 1.5 minutes on CPU (30 runs of 30 iterations × 2048 steps, pure-Python step loop).

```python
import numpy as np

# CartPole dynamics (Barto, Sutton & Anderson 1983; same constants as Gymnasium CartPole-v1)
G, MC, MP, LEN, FMAG, TAU = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
SCALE = np.array([2.4, 3.0, 0.21, 3.0])


def cartpole_step(s, a):
    x, xd, th, thd = s
    f = FMAG if a == 1 else -FMAG
    ct, st = np.cos(th), np.sin(th)
    tmp = (f + MP * LEN * thd**2 * st) / (MC + MP)
    thacc = (G * st - ct * tmp) / (LEN * (4 / 3 - MP * ct**2 / (MC + MP)))
    xacc = tmp - MP * LEN * thacc * ct / (MC + MP)
    s = np.array([x + TAU * xd, xd + TAU * xacc, th + TAU * thd, thd + TAU * thacc])
    done = abs(s[0]) > 2.4 or abs(s[2]) > 12 * np.pi / 180
    return s, done


def actor_feats(s):
    return np.append(s / SCALE, 1.0)


def critic_feats(s):
    z = s / SCALE
    return np.concatenate([[1.0], z, np.outer(z, z)[np.triu_indices(4)]])


def collect(rng, theta, w, n_steps=2048, gamma=0.99, lam=0.95, max_len=500):
    """Roll out n_steps with the current policy; return arrays + GAE advantages."""
    XA, XC, A, LOGP, ADV, RET, lengths = [], [], [], [], [], [], []
    s, t = rng.uniform(-0.05, 0.05, 4), 0
    ep = []
    for i in range(n_steps):
        xa, xc = actor_feats(s), critic_feats(s)
        p = 1 / (1 + np.exp(-theta @ xa))
        a = int(rng.random() < p)
        s2, term = cartpole_step(s, a)
        t += 1
        trunc = t >= max_len
        ep.append((xa, xc, a, np.log(p if a else 1 - p), w @ xc))
        if term or trunc or i == n_steps - 1:
            # bootstrap from V(s') unless the pole really fell
            v_last = 0.0 if term else w @ critic_feats(s2)
            vals = np.array([e[4] for e in ep] + [v_last])
            deltas = 1.0 + gamma * vals[1:] - vals[:-1]
            adv, g = np.zeros(len(ep)), 0.0
            for k in reversed(range(len(ep))):
                g = deltas[k] + gamma * lam * g
                adv[k] = g
            for e in ep:
                XA.append(e[0]); XC.append(e[1]); A.append(e[2]); LOGP.append(e[3])
            ADV.extend(adv)
            RET.extend(adv + vals[:-1])
            if term or trunc:
                lengths.append(t)
            s, t, ep = rng.uniform(-0.05, 0.05, 4), 0, []
        else:
            s = s2
    return (np.array(XA), np.array(XC), np.array(A), np.array(LOGP),
            np.array(ADV), np.array(RET), lengths)


def train(seed, clip, iters=30, epochs=10, batch=256, lr_a=0.05, lr_c=0.05):
    """clip=0.2 -> PPO-clip; clip=None -> same reuse of data with the plain ratio objective."""
    rng = np.random.default_rng(seed)
    theta, w = np.zeros(5), np.zeros(15)
    history, kls = [], []
    for _ in range(iters):
        XA, XC, A, LOGP_old, ADV, RET, lengths = collect(rng, theta, w)
        history.append(np.mean(lengths) if lengths else 500)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)     # per-batch advantage normalisation
        for _ in range(epochs):
            for idx in np.array_split(rng.permutation(len(A)), len(A) // batch):
                xa, a, adv = XA[idx], A[idx], ADV[idx]
                p = 1 / (1 + np.exp(-xa @ theta))
                logp = np.where(a == 1, np.log(p), np.log(1 - p))
                ratio = np.exp(logp - LOGP_old[idx])
                # d log pi / d theta for a Bernoulli-logistic policy = (a - p) x
                g = ratio * adv
                if clip is not None:
                    # gradient is zero where the clipped term is the active min
                    out = ((adv > 0) & (ratio > 1 + clip)) | ((adv < 0) & (ratio < 1 - clip))
                    g = np.where(out, 0.0, g)
                theta += lr_a * ((g * (a - p))[:, None] * xa).mean(0)
                # critic: least squares onto the GAE return
                xc = XC[idx]
                w += lr_c * ((RET[idx] - xc @ w)[:, None] * xc).mean(0) / 10
        # how far did this iteration move the policy? KL(old || new) on the collected states
        p_old = np.where(A == 1, np.exp(LOGP_old), 1 - np.exp(LOGP_old))
        p_new = 1 / (1 + np.exp(-XA @ theta))
        kls.append(np.mean(p_old * np.log(p_old / p_new) + (1 - p_old) * np.log((1 - p_old) / (1 - p_new))))
    return np.array(history), np.array(kls)


for lr_a in [0.05, 0.5, 2.0]:
    for name, clip in [("ppo-clip 0.2", 0.2), ("no clip", None)]:
        runs = [train(seed, clip, lr_a=lr_a) for seed in range(5)]
        H, KL = np.array([r[0] for r in runs]), np.array([r[1] for r in runs])
        print(f"lr {lr_a:4.2f} {name:12s}: iter 10 {H[:, 9].mean():6.1f} | last 5 iters "
              f"{H[:, -5:].mean():6.1f} (worst seed {H[:, -5:].mean(1).min():5.1f}) | "
              f"max KL per update {KL.max():.3f}")
```

Output (numpy 2.5, 5 seeds each; numbers are mean episode length out of 500):
```
lr 0.05 ppo-clip 0.2: iter 10   47.5 | last 5 iters  175.7 (worst seed 164.8) | max KL per update 0.003
lr 0.05 no clip     : iter 10   48.5 | last 5 iters  175.0 (worst seed 148.5) | max KL per update 0.003
lr 0.50 ppo-clip 0.2: iter 10  203.0 | last 5 iters  224.9 (worst seed 189.6) | max KL per update 0.008
lr 0.50 no clip     : iter 10  179.4 | last 5 iters  251.0 (worst seed 179.2) | max KL per update 0.420
lr 2.00 ppo-clip 0.2: iter 10  222.3 | last 5 iters  265.4 (worst seed 192.8) | max KL per update 0.014
lr 2.00 no clip     : iter 10  130.6 | last 5 iters  225.8 (worst seed 102.5) | max KL per update 1.320
```

At `lr 0.05` ten epochs move the policy so little (KL 0.003) that the ratio never leaves `[0.8, 1.2]`. The clip does nothing and the two runs match. Raising the step size is where PPO earns its name. Without the clip the largest single iteration moves the policy by KL 0.42 and then 1.32: the policy at the end of the epochs is very different from the one that collected the batch, so its advantages no longer describe it. At `lr 2.0` that shows up as a bad early phase (131 steps at iteration 10 vs 222) and a seed that ends at 103. With the clip, the KL per update stays below 0.015 even at a 40× larger step size, and the clipped run is the best configuration overall (265). The clip did not make learning faster. It made a larger step size safe. The unclipped `lr 0.5` row ends higher (251 vs 225), which is within seed noise for 5 seeds, so read the KL column, not single averages. The linear-logistic policy, not PPO, limits how long any of these runs keep the pole up; library PPO with an MLP policy is the usual way to reach the 500-step cap.

## Choosing / trade-offs
- **ε (clip range).** 0.1 to 0.3; 0.2 is the standard default. Smaller is safer and slower. Many implementations anneal it.
- **Epochs and minibatches.** More epochs extract more from each batch and push the ratio harder. Typical: 3 to 10 epochs, 4 to 32 minibatches. If the clip fraction (share of samples with an out-of-range ratio) exceeds about 0.2 to 0.3, cut epochs or the learning rate.
- **Early stopping on KL.** Spinning Up and many libraries stop the epoch loop once the approximate KL passes a target (for example 0.01 to 0.02). It is a second guard on top of the clip, which on its own does not strictly bound the KL.
- **PPO-clip vs PPO-penalty vs TRPO.** The KL-penalty variant adds `−β·KL` with an adaptive β. TRPO solves the constrained step exactly with conjugate gradients. Clip is the simplest and usually performs as well.
- **PPO vs SAC/TD3.** PPO is on-policy: each batch is used for one iteration and then discarded. Off-policy methods reuse a replay buffer and need far fewer environment steps, but PPO is easier to tune and to parallelise over many cheap simulators.
- **For LLMs.** PPO needs a critic as large as the policy. GRPO replaces the critic with the mean reward of a group of samples per prompt, and DPO skips RL entirely; see [[rlhf-and-preference-optimization]].

## Gotchas
- `π_old` is frozen. Store the log-probabilities at collection time and never recompute them after the policy changes; the ratio must be 1 at the start of every iteration.
- Normalise advantages per batch (zero mean, unit variance). Without it, the scale of `A` changes with the episode length and the effective step size drifts.
- The `min` is not symmetric. A bad action whose probability already rose (`A < 0`, `r > 1 + ε`) is not clipped, so the objective can always undo a harmful change.
- Clipping does not bound the KL. A sample can leave the clip range in a single minibatch step and the clip only stops further pushing. Watch the measured KL, as above.
- Truncation is not termination: bootstrap from `V(s')` at the step limit or at the end of the rollout buffer. The code above does both (`v_last`).
- Many reported PPO gains come from implementation details (orthogonal init, observation normalisation, value clipping, learning-rate annealing, gradient-norm clipping). The ICLR blog post on the 37 implementation details lists them; reproduce a baseline before trusting a comparison.

## Related
- [[actor-critic-from-scratch]] - the same CartPole, policy and critic features, with one update per sample.
- [[reinforce-policy-gradient-from-scratch]] - the plain policy gradient PPO builds on.
- [[reinforcement-learning]] - PPO, SAC and A2C with maintained libraries.
- [[rlhf-and-preference-optimization]] - PPO, GRPO and DPO applied to language models.
- [[q-learning-from-scratch]] - the off-policy, value-based alternative.
- [[information-theory-for-ml]] - KL divergence, the quantity PPO keeps small.

## References
- Schulman et al. (2017), "Proximal Policy Optimization Algorithms": https://arxiv.org/abs/1707.06347
- Schulman et al. (2015), "High-Dimensional Continuous Control Using Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
- OpenAI Spinning Up, "Proximal Policy Optimization": https://spinningup.openai.com/en/latest/algorithms/ppo.html
- Huang et al. (2022), "The 37 Implementation Details of Proximal Policy Optimization" (ICLR blog track): https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
