---
title: REINFORCE policy gradient from scratch (CartPole, reward-to-go, baseline)
category: concepts
tags: [reinforce, policy-gradient, reinforcement-learning, log-derivative-trick, baseline, variance-reduction, cartpole, stochastic-policy, numpy, from-scratch, deep-learning-basics]
use_cases:
  - "implement the REINFORCE policy-gradient algorithm in NumPy without an RL library"
  - "balance CartPole with a logistic policy trained from episode returns"
  - "see how a baseline and reward-to-go cut policy-gradient variance"
  - "explain the log-derivative trick and why PPO/GRPO add baselines and clipping in an interview"
status: stable
last_verified: 2026-10-04
sources:
  - https://doi.org/10.1007/BF00992696
  - http://incompleteideas.net/book/the-book-2nd.html
  - https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
  - https://gymnasium.farama.org/environments/classic_control/cart_pole/
---

# REINFORCE policy gradient from scratch (CartPole, reward-to-go, baseline)

## Summary
Policy-gradient methods learn a policy `π_θ(a | s)` directly, with no Q-table. REINFORCE runs an episode, then raises the log-probability of each action in proportion to the return that followed it. Over many episodes, actions that led to long, rewarding runs become more likely. Below, a 5-parameter logistic policy learns to balance CartPole (simulated in about 15 lines with the classic equations) from +1 reward per step. With raw returns the gradient is so noisy that after 600 episodes the pole stays up for only 38 steps on average (random play: 22). Standardising the reward-to-go, which acts as a baseline, raises that to 203, with every seed between 167 and 237. Variance reduction is the main idea behind REINFORCE's descendants: actor-critic, PPO, and the GRPO used to train reasoning LLMs.

## Key concepts
- **Objective.** `J(θ) = E_τ~π_θ [R(τ)]`, the expected return over trajectories the policy generates.
- **Log-derivative trick.** `∇J = E[Σ_t ∇ log π_θ(a_t | s_t) · G_t]`. It needs only samples and the gradient of the log-probability of the actions taken, not the environment's dynamics or reward function.
- **Reward-to-go.** Weight each action by the return from that step on, `G_t = Σ_{k≥t} γ^{k−t} r_k`, not by the whole episode's return. An action cannot affect rewards that came before it.
- **Baseline.** Subtracting any `b(s_t)` from `G_t` leaves the gradient unbiased but can cut its variance a lot. The episode mean (as here) is the simplest; a learned value function `V(s)` gives actor-critic, and `G_t − V(s_t)` is the advantage.
- **Logistic policy gradient.** For `π(right | s) = sigmoid(θ · s)`, `∇ log π(a | s) = (a − p) s`, the same form as the logistic-regression gradient.
- **On-policy.** Each update uses fresh episodes from the current policy, so old data cannot be reused without importance weighting.

## When to use / scenarios
- Learning: the root of every policy-gradient method, from actor-critic to PPO and RLHF, in about 40 lines.
- Interviews: "derive the policy gradient", "why does a baseline not add bias", "REINFORCE vs Q-learning", "why is PPO clipped", "what does GRPO use as its baseline".
- Problems with continuous or very large action spaces, or where a stochastic policy is itself the goal (mixed strategies, sampling text from an LLM).
- Not for: sample-limited real systems (REINFORCE is very sample-inefficient; use off-policy SAC or model-based RL), small discrete problems ([[q-learning-from-scratch]] is simpler and exact), or production training (use PPO from a maintained library, see [[reinforcement-learning]]).

## Setup & code
`pip install numpy`. Runs in about 5 seconds on CPU.

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


def policy(theta, s):
    """Logistic policy: P(push right | s) = sigmoid(theta . [s, 1])."""
    p = 1 / (1 + np.exp(-(theta @ np.append(s, 1.0))))
    return p


def episode(theta, rng, max_steps=500):
    s = rng.uniform(-0.05, 0.05, 4)
    feats, acts, probs = [], [], []
    for _ in range(max_steps):
        p = policy(theta, s)
        a = int(rng.random() < p)
        feats.append(np.append(s, 1.0)); acts.append(a); probs.append(p)
        s, done = cartpole_step(s, a)
        if done:
            break
    return np.array(feats), np.array(acts), np.array(probs)


def reinforce(rng, episodes=600, lr=0.01, gamma=0.99, baseline=True):
    theta = np.zeros(5)
    lengths = []
    for _ in range(episodes):
        Xf, A, P = episode(theta, rng)
        T = len(A)
        lengths.append(T)
        # reward +1 per step -> discounted reward-to-go G_t
        Gt = np.array([(gamma ** np.arange(T - t)).sum() for t in range(T)])
        adv = (Gt - Gt.mean()) / (Gt.std() + 1e-8) if baseline else Gt   # baseline + scale: lower variance
        grad_logp = (A - P)[:, None] * Xf                                  # d/dtheta log pi(a|s) for a Bernoulli-logistic policy
        theta += lr * (adv[:, None] * grad_logp).sum(0)
    return theta, np.array(lengths)


rng = np.random.default_rng(0)
print(f"random policy (theta=0): mean episode length {np.mean([len(episode(np.zeros(5), rng)[1]) for _ in range(100)]):.1f}")
for baseline in [False, True]:
    finals = []
    for seed in range(5):
        theta, L = reinforce(np.random.default_rng(seed), baseline=baseline)
        greedy = np.mean([len(episode(theta, np.random.default_rng(100 + i))[1]) for i in range(20)])
        finals.append((L[:50].mean(), L[-50:].mean(), greedy))
    f = np.array(finals)
    print(f"baseline={str(baseline):5s}: mean length first 50 eps {f[:, 0].mean():6.1f} | last 50 eps {f[:, 1].mean():6.1f} "
          f"(per seed {np.round(f[:, 1]).astype(int).tolist()}) | eval {f[:, 2].mean():6.1f} / 500")
```

Output (numpy 2.5):
```
random policy (theta=0): mean episode length 22.3
baseline=False: mean length first 50 eps   21.6 | last 50 eps   38.1 (per seed [36, 23, 35, 47, 48]) | eval   36.3 / 500
baseline=True : mean length first 50 eps   26.7 | last 50 eps  202.9 (per seed [210, 237, 167, 194, 206]) | eval  184.3 / 500
```

A random policy keeps the pole up for 22 steps. Plain REINFORCE, weighting each action by its raw reward-to-go, barely improves in 600 episodes (38 steps), and one seed is still near random (23). With +1 reward per step every `G_t` is positive, so every action taken gets reinforced. The signal that separates good actions from bad is a small difference between large numbers, buried in noise. Standardising `G_t` per episode subtracts a baseline and fixes the scale: early actions that preceded a long survival get positive weight, late ones just before the fall get negative weight. The same policy, learning rate and episode budget now reach 203 steps, and every seed lands between 167 and 237. The "eval" column re-runs each final policy on 20 fresh episodes (still sampling actions) and confirms the gap: 184 vs 36 out of a maximum of 500.

## Choosing / trade-offs
- **Baseline choice.** Per-episode or per-batch mean is free and helps a lot. A learned `V(s)` (actor-critic, A2C) helps more on long episodes. GRPO, used for LLM reasoning RL, uses the mean reward of a group of sampled answers to the same prompt instead of a value network.
- **Batch size.** Average the gradient over several episodes per update for less variance at the cost of fewer updates; one episode per update, as above, is the noisiest setting.
- **Step size.** Policy gradients are very sensitive to it: one large step can collapse the policy to deterministic and stop exploration. PPO clips the probability ratio and TRPO constrains the KL divergence for exactly this reason.
- **Discount γ.** Below 1 reduces variance by down-weighting distant rewards, at the cost of bias toward short-term reward.
- **Policy class.** Logistic or softmax-linear on hand-made features for teaching; an MLP for real tasks; a Gaussian with learned mean and std for continuous actions.
- **Policy gradient vs value-based.** Policy gradients handle continuous actions and stochastic policies naturally; value methods (Q-learning, DQN) are more sample-efficient on discrete actions and can learn off-policy from replay.

## Gotchas
- Using the whole-episode return for every step (not reward-to-go) adds variance with no benefit.
- Positive-only rewards with no baseline reinforce everything; learning then depends on tiny differences and stalls (38 steps above).
- The sign of the update is ascent: `θ += lr · ∇J`. With an autograd library you minimise `−Σ log π(a|s) · A`, and getting the sign wrong trains the agent to fail.
- Do not backpropagate through the advantage. It is a constant weight; in PyTorch call `.detach()` on it.
- Entropy collapse: the policy becomes deterministic early and stops exploring. Add an entropy bonus or lower the learning rate.
- Results vary a lot by seed. Report several seeds (5 above) and the spread, not one curve.
- CartPole caps episodes at 500 steps (`max_steps`); report it as "x / 500", otherwise lengths are not comparable.

## Related
- [[reinforcement-learning]] - PPO, SAC and DQN with maintained libraries and environments.
- [[q-learning-from-scratch]] - the value-based alternative on a tabular problem.
- [[rlhf-and-preference-optimization]] - PPO and GRPO, policy gradients applied to LLM fine-tuning.
- [[logistic-regression-from-scratch]] - the same `(y − p) x` gradient, here weighted by return.
- [[multi-armed-bandits]] - the one-step special case, where gradient-bandit algorithms are REINFORCE.
- [[optimizers-from-scratch-numpy]] - why noisy gradients need small steps or momentum.

## References
- Williams (1992), "Simple statistical gradient-following algorithms for connectionist reinforcement learning", Machine Learning 8: https://doi.org/10.1007/BF00992696
- Sutton and Barto, Reinforcement Learning: An Introduction, 2nd ed. (2018), Chapter 13: http://incompleteideas.net/book/the-book-2nd.html
- OpenAI Spinning Up, "Intro to Policy Optimization": https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
- Gymnasium CartPole environment: https://gymnasium.farama.org/environments/classic_control/cart_pole/
