---
title: Deep Q-network (DQN) from scratch in NumPy (replay buffer, target network, double DQN, CartPole)
category: concepts
tags: [dqn, deep-q-network, double-dqn, experience-replay, target-network, q-learning, function-approximation, deadly-triad, cartpole, reinforcement-learning, numpy, from-scratch, deep-learning-basics]
use_cases:
  - "implement DQN with experience replay, a target network and double-DQN targets in plain NumPy"
  - "balance CartPole with a neural-network Q-function without Gym or PyTorch"
  - "see what breaks when you remove the target network or the replay buffer"
  - "explain why naive Q-learning with a neural network diverges in a reinforcement-learning interview"
status: draft
last_verified: 2026-10-04
sources:
  - https://doi.org/10.1038/nature14236
  - https://arxiv.org/abs/1509.06461
  - https://gymnasium.farama.org/environments/classic_control/cart_pole/
  - https://docs.cleanrl.dev/rl-algorithms/dqn/
---

# Deep Q-network (DQN) from scratch in NumPy (replay buffer, target network, double DQN, CartPole)

## Summary
DQN is Q-learning with a neural network in place of the Q-table, so it works when states are continuous or too numerous to enumerate. Plugging a network into the Q-learning update diverges on its own. DQN adds two stabilisers: a replay buffer that trains on random past transitions instead of the latest correlated ones, and a slowly moving target network that computes the TD targets. Below, a 4-64-64-2 MLP written in NumPy learns CartPole from 60,000 environment steps. Its best checkpoint balances the pole for the full 500 steps on all 3 seeds. Without a target network, Q-values blow up to about 7×10⁸, against a true maximum of 100, and the policy never lasts more than 10 steps. The final weights of the working agent score 362, 500 and 98, which shows how unstable DQN training is even when it works.

## Key concepts
- **Q-function approximation.** A network `Q(s, ·; θ)` outputs one value per action. The greedy policy picks the arg-max.
- **TD target.** `y = r + γ (1 − done) · max_a' Q(s', a'; θ⁻)`. Train `Q(s, a; θ)` toward `y` with a regression loss. Only the taken action's output gets a gradient.
- **Experience replay.** Store `(s, a, r, s', done)` in a ring buffer and train on uniformly sampled minibatches. This breaks the correlation between consecutive steps and reuses each transition many times.
- **Target network.** `θ⁻` is a lagged copy of `θ`, refreshed by a hard copy every C steps or by a soft (Polyak) update `θ⁻ ← τθ + (1 − τ)θ⁻`. Without it the target moves with every gradient step and the regression chases itself.
- **Double DQN.** The max in the target overestimates, because noise in Q makes the max biased upward. Double DQN picks `a* = argmax Q(s', ·; θ)` with the online net and evaluates it with the target net.
- **Deadly triad.** Function approximation, bootstrapping (targets built from your own estimates) and off-policy data together can diverge. DQN does not remove the risk, only damps it.
- **Termination vs truncation.** A pole that falls is a real terminal state (no future value). Hitting the 500-step time limit is not, so the target must still bootstrap from `s'`.

## When to use / scenarios
- Learning: the bridge from tabular [[q-learning-from-scratch]] to deep RL, and the base that Rainbow, QR-DQN and R2D2 extend.
- Interviews: "why a replay buffer", "why a target network", "what is overestimation bias", "DQN vs policy gradients", "why can't DQN do continuous actions".
- Practice: discrete-action control with cheap simulators, such as Atari-style games, job and packet scheduling, recommendation and ad-slot sequencing simulators, HVAC or traffic-light control in simulation, and simple robotics with a few discrete motor commands.
- Not for: continuous actions (use SAC, TD3 or PPO); tasks where a simulator is unavailable and real interaction is costly (consider offline RL or bandits first); or problems where supervised learning on logged decisions already works. For real projects, use a maintained implementation (Stable-Baselines3, CleanRL) over hand-written code.

## Setup & code
NumPy only. The CartPole dynamics are implemented inline with the same equations and constants as Gymnasium's `CartPole-v1`, so no Gym install is needed. The full script takes about 3 minutes on a laptop CPU (9 training runs of 60k steps).

```python
import numpy as np

# ---------- CartPole dynamics (Barto, Sutton & Anderson 1983; same constants as Gymnasium CartPole-v1) ----------
G, MC, MP, L, FMAG, TAU = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02


class CartPole:
    def __init__(self, rng):
        self.rng = rng

    def reset(self):
        self.s, self.t = self.rng.uniform(-0.05, 0.05, 4), 0
        return self.s.copy()

    def step(self, a):
        x, xd, th, thd = self.s
        force = FMAG if a == 1 else -FMAG
        c, s = np.cos(th), np.sin(th)
        tmp = (force + MP * L * thd**2 * s) / (MC + MP)
        thacc = (G * s - c * tmp) / (L * (4 / 3 - MP * c**2 / (MC + MP)))
        xacc = tmp - MP * L * thacc * c / (MC + MP)
        self.s = np.array([x + TAU * xd, xd + TAU * xacc, th + TAU * thd, thd + TAU * thacc])
        self.t += 1
        fell = abs(self.s[0]) > 2.4 or abs(self.s[2]) > 12 * np.pi / 180
        return self.s.copy(), 1.0, fell, self.t >= 500           # obs, reward, terminated, truncated


# ---------- Q-network: 4 -> 64 -> 64 -> 2, ReLU, trained with Adam on the Huber TD loss ----------
def init(rng, sizes=(4, 64, 64, 2)):
    return [(rng.normal(0, np.sqrt(2 / i), (i, o)), np.zeros(o)) for i, o in zip(sizes, sizes[1:])]


def forward(params, x):
    hs = [x]
    for k, (W, b) in enumerate(params):
        x = x @ W + b
        if k < len(params) - 1:
            x = np.maximum(x, 0)
        hs.append(x)
    return x, hs


def backward(params, hs, dout):
    grads = []
    for k in reversed(range(len(params))):
        W, _ = params[k]
        grads.append((hs[k].T @ dout, dout.sum(0)))
        dout = (dout @ W.T) * (hs[k] > 0) if k > 0 else None
    return grads[::-1]


class Adam:
    def __init__(self, params, lr=1e-3):
        self.lr, self.t = lr, 0
        self.m = [[np.zeros_like(p) for p in layer] for layer in params]
        self.v = [[np.zeros_like(p) for p in layer] for layer in params]

    def step(self, params, grads):
        self.t += 1
        out = []
        for (W, b), g, m, v in zip(params, grads, self.m, self.v):
            new = []
            for p, gp, mp, vp in zip((W, b), g, m, v):
                mp[:] = 0.9 * mp + 0.1 * gp
                vp[:] = 0.999 * vp + 0.001 * gp**2
                mh, vh = mp / (1 - 0.9**self.t), vp / (1 - 0.999**self.t)
                new.append(p - self.lr * mh / (np.sqrt(vh) + 1e-8))
            out.append(tuple(new))
        return out


def dqn(seed, total_steps=60_000, gamma=0.99, batch=64, buffer_size=50_000, tau=0.005, lr=5e-4,
        double=True, use_target=True, use_replay=True):
    rng = np.random.default_rng(seed)
    env = CartPole(rng)
    online = init(rng)
    target = [(W.copy(), b.copy()) for W, b in online]
    opt = Adam(online, lr)
    S, A, Rw, S2, D = (np.zeros((buffer_size, 4)), np.zeros(buffer_size, int), np.zeros(buffer_size),
                       np.zeros((buffer_size, 4)), np.zeros(buffer_size))
    size = ptr = steps = 0
    returns, best, best_score = [], online, -1.0
    while steps < total_steps:
        s, ret, done = env.reset(), 0.0, False
        while not done:
            eps = max(0.05, 1 - steps / 10_000)                     # linear epsilon decay
            a = int(rng.integers(2)) if rng.random() < eps else int(forward(online, s[None])[0].argmax())
            s2, r, term, trunc = env.step(a)
            S[ptr], A[ptr], Rw[ptr], S2[ptr], D[ptr] = s, a, r, s2, float(term)  # truncation is NOT terminal
            ptr, size = (ptr + 1) % buffer_size, min(size + 1, buffer_size)
            s, ret, done, steps = s2, ret + r, term or trunc, steps + 1
            if size >= 1000:
                idx = rng.integers(size, size=batch) if use_replay else (np.arange(ptr - batch, ptr) % buffer_size)
                tgt_net = target if use_target else online
                q_next_t = forward(tgt_net, S2[idx])[0]
                if double:                                          # online net picks, target net evaluates
                    a_star = forward(online, S2[idx])[0].argmax(1)
                    q_next = q_next_t[np.arange(batch), a_star]
                else:
                    q_next = q_next_t.max(1)
                y = Rw[idx] + gamma * (1 - D[idx]) * q_next
                q, hs = forward(online, S[idx])
                err = q[np.arange(batch), A[idx]] - y
                dq = np.zeros_like(q)
                dq[np.arange(batch), A[idx]] = np.clip(err, -1, 1) / batch   # Huber loss gradient
                online = opt.step(online, backward(online, hs, dq))
                if use_target:                                      # Polyak (soft) target update
                    target = [(tau * W + (1 - tau) * Wt, tau * b + (1 - tau) * bt)
                              for (W, b), (Wt, bt) in zip(online, target)]
            if steps % 5000 == 0:                                   # checkpoint the best greedy policy
                score = evaluate(online, seed=steps, n=5)
                if score >= best_score:
                    best, best_score = online, score
        returns.append(ret)
    q_max = np.abs(forward(online, S[:size])[0]).max()               # true values are <= 1/(1-gamma) = 100
    return online, best, q_max


def evaluate(params, seed, n=20):
    env, total = CartPole(np.random.default_rng(seed)), []
    for _ in range(n):
        s, ret, done = env.reset(), 0.0, False
        while not done:
            s, r, term, trunc = env.step(int(forward(params, s[None])[0].argmax()))
            ret, done = ret + r, term or trunc
        total.append(ret)
    return np.mean(total)


print("greedy return over 20 fresh episodes (max 500), 3 seeds, 60k env steps each")
print(f"{'variant':<34} {'final weights':<16} {'best checkpoint':<16} max|Q|")
for name, kw in [("double DQN, soft target, replay", {}),
                 ("no target network, no double", {"use_target": False, "double": False}),
                 ("no replay (last 64 transitions)", {"use_replay": False})]:
    rows = [dqn(seed, **kw) for seed in range(3)]
    final = "/".join(f"{evaluate(f, 100 + i):.0f}" for i, (f, _, _) in enumerate(rows))
    best = "/".join(f"{evaluate(b, 100 + i):.0f}" for i, (_, b, _) in enumerate(rows))
    qmax = "/".join(f"{q:.0f}" for _, _, q in rows)
    print(f"{name:<34} {final:<16} {best:<16} {qmax}")
```

Output (Python 3.14, NumPy 2.5):
```
greedy return over 20 fresh episodes (max 500), 3 seeds, 60k env steps each
variant                            final weights    best checkpoint  max|Q|
double DQN, soft target, replay    362/500/98       500/500/500      108/120/173
no target network, no double       9/10/9           9/10/9           711811188/738901371/665547422
no replay (last 64 transitions)    216/222/484      344/492/500      95/98/95
```

Each cell lists the 3 seeds. The full agent's best checkpoint solves CartPole on every seed, but its final weights score 98 on one seed. That is the well-known DQN failure mode where performance collapses after it peaks, and it is why real training loops evaluate periodically and keep the best checkpoint. Removing the target network (and bootstrapping from the online net with a plain max) makes Q-values explode to about 7×10⁸. A reward of 1 per step with γ = 0.99 bounds every true value at 100, so this is pure divergence, and the policy degenerates into pushing one way. The max|Q| column also shows mild overestimation in the working agent (108 to 173, above the bound of 100), even with double-DQN targets. Training on only the latest 64 transitions still learns something on this easy task, but less reliably: 2 of 3 best checkpoints fall short of 500. On harder tasks such as Atari, replay matters far more. With only 3 seeds and 60k steps, treat these as qualitative effects, not benchmark numbers.

## Choosing / trade-offs
- **Hard vs soft target updates.** Hard copies every C steps (the original Atari DQN used C = 10,000 updates) or Polyak averaging every step (τ around 0.005). Soft updates change the targets more smoothly. Both work when tuned.
- **DQN vs policy-gradient and actor-critic methods.** DQN is off-policy and sample-efficient through replay, but only for discrete actions. PPO is more stable and handles continuous actions, at the cost of sample efficiency. SAC is the usual choice for continuous control.
- **Extensions.** Rainbow combines double DQN, dueling heads, prioritised replay, n-step returns, distributional value heads and noisy nets. Prioritised replay and n-step returns usually give the largest gains.
- **Buffer size.** Too small and samples are correlated and old skills are forgotten. Too large and the buffer is dominated by stale behaviour. 10⁴ to 10⁶ transitions is typical.
- **Exploration.** Epsilon-greedy with linear decay is the standard baseline. Too fast a decay locks in a poor policy, and too slow a decay wastes steps.
- **Libraries.** Stable-Baselines3 `DQN` and CleanRL `dqn.py` (single file, well documented) for real use. Gymnasium for environments.

## Gotchas
- Mark only true terminal states as `done`. Treating the time limit as terminal teaches the agent that the world ends at step 500 and biases values near the limit.
- Gradients flow only through `Q(s, a)` for the taken action. A loss over all outputs, or gradients through the target, silently trains the wrong thing.
- Use the Huber loss or gradient clipping. Squared TD errors early in training produce huge gradient steps.
- The training-episode return during training is measured with epsilon-greedy exploration. Evaluate the greedy policy separately on fresh episodes.
- Performance can collapse after it peaks. Checkpoint on evaluation score instead of keeping the last weights, and report results across several seeds.
- Rewards and observations with large scales destabilise the regression. Clip or normalise rewards (Atari DQN clipped to ±1) and normalise observations.
- Training on every step starts only after the buffer holds enough data (here 1000 transitions). Training on a nearly empty buffer overfits the first few episodes.

## Related
- [[q-learning-from-scratch]] - the tabular version, with exact values from value iteration.
- [[reinforcement-learning]] - deep RL algorithms, libraries and environments in practice.
- [[reinforce-policy-gradient-from-scratch]] - the policy-gradient alternative to value-based methods.
- [[actor-critic-from-scratch]] - combines a learned value function with a policy.
- [[neural-network-from-scratch-numpy]] - the MLP forward/backward pass used for the Q-network.
- [[optimizers-from-scratch-numpy]] - Adam and the other update rules.

## References
- Mnih et al. (2015), "Human-level control through deep reinforcement learning", Nature: https://doi.org/10.1038/nature14236
- van Hasselt, Guez and Silver (2015), "Deep reinforcement learning with double Q-learning": https://arxiv.org/abs/1509.06461
- Gymnasium CartPole-v1 (dynamics, termination and truncation): https://gymnasium.farama.org/environments/classic_control/cart_pole/
- CleanRL DQN documentation: https://docs.cleanrl.dev/rl-algorithms/dqn/
