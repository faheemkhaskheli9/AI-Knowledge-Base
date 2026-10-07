---
title: Actor-critic from scratch (CartPole, learned value baseline, TD error)
category: concepts
tags: [actor-critic, policy-gradient, value-function, td-learning, advantage, baseline, reinforcement-learning, cartpole, numpy, from-scratch, deep-learning-basics]
use_cases:
  - "implement an actor-critic agent in NumPy without an RL library"
  - "compare a learned V(s) baseline (Monte Carlo advantage) with one-step TD actor-critic on CartPole"
  - "see the bias-variance trade-off between Monte Carlo returns and bootstrapped TD targets"
  - "explain advantage, TD error and why A2C/PPO use a critic in an interview"
status: stable
last_verified: 2026-10-04
sources:
  - http://incompleteideas.net/book/the-book-2nd.html
  - https://arxiv.org/abs/1602.01783
  - https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
  - https://gymnasium.farama.org/environments/classic_control/cart_pole/
---

# Actor-critic from scratch (CartPole, learned value baseline, TD error)

## Summary
An actor-critic agent has two parts. The actor is a policy `π_θ(a | s)`. The critic is a value function `V_w(s)` that predicts the return from a state. The critic replaces the noisy return in the policy gradient with an advantage: how much better an action turned out than the critic expected. This file extends [[reinforce-policy-gradient-from-scratch]] on the same CartPole simulator and logistic policy, adding a linear critic on 15 quadratic state features. With the critic as a Monte Carlo baseline (`G_t − V(s_t)`, updated after each episode), the last 50 of 600 episodes average 356 steps over 5 seeds, against 203 for REINFORCE with a per-episode standardised baseline. With the critic bootstrapped one step (TD error `r + γV(s') − V(s)`, updated every step), the average is 254 and the seeds spread from 164 to 331. TD targets have lower variance but inherit the critic's bias, and a linear critic is quite biased here.

## Key concepts
- **Advantage.** `A(s, a) = Q(s, a) − V(s)`. The policy gradient becomes `E[∇ log π(a | s) · A(s, a)]`. Subtracting `V(s)` leaves it unbiased and removes the part of the return that the state alone explains.
- **Critic training.** Regress `V_w(s)` onto a target with squared error. For a linear critic `V = w · φ(s)` the update is `w += lr · (target − V(s)) · φ(s)`.
- **Monte Carlo advantage.** Target = the observed reward-to-go `G_t`. Unbiased, high variance, and the update must wait until the episode ends.
- **TD error.** `δ = r + γ V(s') − V(s)` (just `r − V(s)` at a terminal state). It estimates the advantage from a single step, so updates can happen online, but any error in `V` becomes bias in the policy gradient.
- **n-step and GAE.** Bootstrapping after n steps, or the exponentially weighted mix in Generalized Advantage Estimation (λ between 0 and 1), interpolates between TD (λ = 0) and Monte Carlo (λ = 1). PPO uses GAE.
- **Two learning rates.** The critic usually learns faster than the actor, so the actor follows a reasonably accurate advantage.

## When to use / scenarios
- Learning: the step from REINFORCE to A2C, PPO and SAC, which are all actor-critic methods.
- Interviews: "what is the advantage", "why does a critic reduce variance", "TD vs Monte Carlo bias and variance", "what does GAE's λ do", "why does PPO have a value loss".
- Long or continuing tasks where waiting for episode ends is impractical; any setting that needs per-step updates.
- Not for: production training (use PPO or SAC from Stable-Baselines3 or CleanRL, see [[reinforcement-learning]]), or small discrete problems ([[q-learning-from-scratch]] is simpler).

## Setup & code
`pip install numpy`. Runs in about a minute on CPU (pure-Python step loop, 10 runs of 600 episodes).

<!-- check-timeout: 900 -->
```python
import numpy as np

# CartPole dynamics (Barto, Sutton & Anderson 1983; same constants as Gymnasium CartPole-v1)
G, MC, MP, LEN, FMAG, TAU = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02


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


SCALE = np.array([2.4, 3.0, 0.21, 3.0])


def actor_feats(s):
    return np.append(s, 1.0)


def critic_feats(s):
    """Quadratic features of the scaled state: a linear critic on these can bend V(s)."""
    z = s / SCALE
    return np.concatenate([[1.0], z, np.outer(z, z)[np.triu_indices(4)]])


def run(rng, mode, episodes=600, lr_a=0.01, lr_c=0.05, gamma=0.99, max_steps=500):
    """mode: 'mc' = REINFORCE with learned V(s) baseline (update after episode)
             'td' = one-step actor-critic (update every step with the TD error)."""
    theta, w = np.zeros(5), np.zeros(15)
    lengths = []
    for _ in range(episodes):
        s = rng.uniform(-0.05, 0.05, 4)
        traj = []
        for t in range(max_steps):
            xa, xc = actor_feats(s), critic_feats(s)
            p = 1 / (1 + np.exp(-theta @ xa))
            a = int(rng.random() < p)
            s2, done = cartpole_step(s, a)
            if mode == "td":
                target = 1.0 if done else 1.0 + gamma * w @ critic_feats(s2)   # bootstrap from V(s')
                delta = target - w @ xc                                           # TD error = advantage estimate
                w += lr_c * delta * xc
                theta += lr_a * delta * (a - p) * xa
            else:
                traj.append((xa, xc, a, p))
            s = s2
            if done:
                break
        lengths.append(t + 1)
        if mode == "mc":
            T = len(traj)
            Gt = np.array([(gamma ** np.arange(T - k)).sum() for k in range(T)])
            for (xa, xc, a, p), g in zip(traj, Gt):
                adv = g - w @ xc
                w += lr_c * adv * xc / 10                                          # MC targets are larger/noisier
                theta += lr_a * adv * (a - p) * xa / 10
    return theta, np.array(lengths)


def evaluate(theta, seed, n=20):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        s, t = rng.uniform(-0.05, 0.05, 4), 0
        while t < 500:
            p = 1 / (1 + np.exp(-theta @ actor_feats(s)))
            s, done = cartpole_step(s, int(rng.random() < p))
            t += 1
            if done:
                break
        out.append(t)
    return np.mean(out)


for mode in ["mc", "td"]:
    finals = []
    for seed in range(5):
        theta, L = run(np.random.default_rng(seed), mode)
        finals.append((L[:50].mean(), L[-50:].mean(), evaluate(theta, 100 + seed)))
    f = np.array(finals)
    print(f"{mode}: first 50 eps {f[:, 0].mean():6.1f} | last 50 eps {f[:, 1].mean():6.1f} "
          f"(per seed {np.round(f[:, 1]).astype(int).tolist()}) | eval {f[:, 2].mean():6.1f} / 500")
```

Output (numpy 2.5):
```
mc: first 50 eps   25.7 | last 50 eps  356.3 (per seed [359, 345, 378, 338, 362]) | eval  398.8 / 500
td: first 50 eps   22.7 | last 50 eps  253.9 (per seed [191, 331, 291, 164, 293]) | eval  241.9 / 500
```

Both agents start near random play (about 23 steps). With the critic as a Monte Carlo baseline, the last 50 episodes average 356 steps and the five seeds are tightly grouped (338 to 378). The "eval" column re-runs each final policy on 20 fresh episodes and gives 399 of a possible 500. The same policy class trained with REINFORCE and a standardised per-episode baseline reached 203. A state-dependent baseline separates "this state was good" from "this action was good" better than a single number per episode. The one-step TD version learns online from every step but ends lower (254) and varies more across seeds (164 to 331). Its advantage `r + γV(s') − V(s)` is only as good as the critic, and a linear function of 15 quadratic features cannot represent CartPole's value function well near the failure boundary, so the policy gradient is biased. With a neural-network critic or n-step returns the balance usually shifts toward bootstrapping, which is why A2C and PPO use multi-step returns or GAE rather than one-step TD.

## Choosing / trade-offs
- **Advantage estimator.** Monte Carlo: unbiased, noisy, episodic only. One-step TD: low variance, biased by the critic, fully online. n-step returns or GAE (PPO's default λ = 0.95) sit in between and are the usual choice.
- **Critic capacity.** The critic's error becomes the actor's bias. Linear critics need good features; neural critics need their own tuning and often a separate network or a shared trunk with two heads.
- **Learning rates.** Critic faster than actor. If the actor moves faster, it chases a stale advantage and oscillates.
- **On-policy actor-critic (A2C, PPO)** vs **off-policy (SAC, TD3, DDPG).** Off-policy methods learn `Q(s, a)` from a replay buffer and are far more sample-efficient, especially for continuous control; on-policy methods are simpler and more stable.
- **Parallel environments.** A2C/A3C average gradients over many environment copies to cut variance, which plays the role of a larger batch.

## Gotchas
- Truncation is not termination. When an episode is cut at the step limit (500 here), bootstrap from `V(s')`; only a real failure has target `r`. Treating timeouts as failures teaches the critic that the 500th step is bad. Gymnasium returns `terminated` and `truncated` separately for this reason.
- The advantage is a constant weight for the actor. With autograd, call `.detach()` on it, or the actor's loss will also push the critic.
- Monte Carlo returns are large (up to about 100 with γ = 0.99), so the update needs a smaller step than TD errors (the `/ 10` above). Normalising advantages per batch, as PPO does, avoids hand-tuning this.
- Unscaled state features make the linear critic diverge: CartPole's pole angle is in radians (±0.21) while velocities reach several units. Scale inputs first (`SCALE` above).
- A good critic loss does not mean a good policy. Track episode return, not value loss, to judge progress.
- Report several seeds. The TD agent above ranges from 164 to 331 steps under identical settings.

## Related
- [[reinforce-policy-gradient-from-scratch]] - the same CartPole and policy without a critic.
- [[q-learning-from-scratch]] - TD learning of action values, the critic's cousin.
- [[reinforcement-learning]] - PPO, SAC and A2C with maintained libraries.
- [[rlhf-and-preference-optimization]] - PPO's actor-critic setup applied to LLMs, and GRPO, which drops the critic.
- [[linear-regression-from-scratch]] - the critic update is online least squares on the TD target.
- [[kalman-filter-from-scratch]] - another case of correcting a prediction with an error signal step by step.

## References
- Sutton and Barto, Reinforcement Learning: An Introduction, 2nd ed. (2018), Chapters 6 and 13.5: http://incompleteideas.net/book/the-book-2nd.html
- Mnih et al. (2016), "Asynchronous Methods for Deep Reinforcement Learning" (A3C/A2C): https://arxiv.org/abs/1602.01783
- OpenAI Spinning Up, "Intro to Policy Optimization": https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
- Gymnasium CartPole environment: https://gymnasium.farama.org/environments/classic_control/cart_pole/
