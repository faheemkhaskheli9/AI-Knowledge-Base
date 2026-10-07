---
title: Tabular Q-learning and SARSA from scratch (cliff walking, value iteration)
category: concepts
tags: [q-learning, sarsa, temporal-difference, td-learning, value-iteration, bellman-equation, epsilon-greedy, reinforcement-learning, gridworld, numpy, from-scratch, deep-learning-basics]
use_cases:
  - "implement tabular Q-learning and SARSA with epsilon-greedy exploration in NumPy"
  - "check a learned Q-table against the exact optimum from value iteration"
  - "see the on-policy vs off-policy difference on the cliff-walking gridworld"
  - "explain the Bellman equation, TD targets and exploration in a reinforcement-learning interview"
status: stable
last_verified: 2026-10-04
sources:
  - http://incompleteideas.net/book/the-book-2nd.html
  - https://doi.org/10.1007/BF00992698
  - https://gymnasium.farama.org/environments/toy_text/cliff_walking/
---

# Tabular Q-learning and SARSA from scratch (cliff walking, value iteration)

## Summary
Q-learning learns `Q(s, a)`, the expected return of taking action `a` in state `s` and acting well afterwards, from trial and error alone, without a model of the environment. After every step it nudges `Q(s, a)` toward a temporal-difference (TD) target `r + γ max_a' Q(s', a')`. SARSA is the same update with the action the agent actually takes next in place of the max. On the 4 × 12 cliff-walking gridworld, Q-learning recovers the exact optimal value from value iteration (−13) and learns the shortest path along the cliff edge. SARSA learns a longer, safer path, and while still exploring it earns about twice the reward (−26 vs −50 per episode) because exploration near the cliff costs Q-learning falls.

## Key concepts
- **MDP.** States, actions, a transition function, rewards and a discount `γ`. The goal is a policy that maximises expected discounted return.
- **Bellman optimality equation.** `Q*(s, a) = E[r + γ max_a' Q*(s', a')]`. Value iteration applies it as an update until it stops changing; that needs the model.
- **TD update.** `Q(s, a) += α (target − Q(s, a))`. It learns from single transitions and bootstraps from its own current estimates.
- **Q-learning (off-policy).** Target `r + γ max_a' Q(s', a')`: it learns the value of the greedy policy while following an exploratory one.
- **SARSA (on-policy).** Target `r + γ Q(s', a')` with the `a'` actually chosen: it learns the value of the exploratory policy it follows, including its occasional random moves.
- **ε-greedy exploration.** With probability ε take a random action, otherwise the best one. Without exploration the agent locks onto the first path that works.
- **Terminal states.** The target is just `r` at the end of an episode; bootstrapping past a terminal state leaks value.

## When to use / scenarios
- Learning: the core ideas behind all of RL (Bellman equations, bootstrapping, exploration vs exploitation, on- vs off-policy) in about 60 lines.
- Interviews: "Q-learning vs SARSA", "what does γ do", "why does DQN need a target network and replay buffer", "on- vs off-policy".
- Small discrete problems: inventory or pricing toys, simple games, routing on small graphs, tuning a controller over a handful of discretised states.
- Not for: large or continuous state spaces (use function approximation: DQN, PPO, SAC via [[reinforcement-learning]]), one-step decisions with no state transitions ([[multi-armed-bandits]]), or problems where trial-and-error in the real system is unsafe or expensive without a simulator.

## Setup & code
`pip install numpy`. Runs in a few seconds on CPU.

```python
import numpy as np

# 4x12 "cliff walking" gridworld (Sutton & Barto, Example 6.6)
H, W = 4, 12
START, GOAL = (3, 0), (3, 11)
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]               # up, right, down, left


def step(s, a):
    r, c = s[0] + MOVES[a][0], s[1] + MOVES[a][1]
    r, c = min(max(r, 0), H - 1), min(max(c, 0), W - 1)
    if r == 3 and 0 < c < 11:                             # fell off the cliff
        return START, -100, False
    return (r, c), -1, (r, c) == GOAL


def eps_greedy(Q, s, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(4))
    q = Q[s]
    return int(rng.choice(np.flatnonzero(q == q.max())))  # break ties randomly


def run(method, episodes, alpha, gamma, eps, rng):
    Q = np.zeros((H, W, 4))
    returns = []
    for _ in range(episodes):
        s, G, done = START, 0, False
        a = eps_greedy(Q, s, eps, rng)
        while not done:
            s2, r, done = step(s, a)
            G += r
            a2 = eps_greedy(Q, s2, eps, rng)
            if method == "q":                             # off-policy: bootstrap from the greedy action
                target = r + gamma * Q[s2].max() * (not done)
            else:                                         # SARSA, on-policy: bootstrap from the action actually taken
                target = r + gamma * Q[s2][a2] * (not done)
            Q[s][a] += alpha * (target - Q[s][a])
            s, a = s2, a2
        returns.append(G)
    return Q, np.array(returns)


def greedy_path(Q):
    s, path = START, [START]
    for _ in range(100):
        s, _, done = step(s, int(Q[s].argmax()))
        path.append(s)
        if done:
            break
    return path


def value_iteration(gamma, tol=1e-9):
    """Exact optimal Q via the Bellman optimality equation (the model is known here)."""
    Q = np.zeros((H, W, 4))
    while True:
        Q_new = np.zeros_like(Q)
        for r in range(H):
            for c in range(W):
                if (r, c) == GOAL:
                    continue
                for a in range(4):
                    s2, rew, done = step((r, c), a)
                    Q_new[r, c, a] = rew + gamma * Q[s2].max() * (not done)
        if np.abs(Q_new - Q).max() < tol:
            return Q_new
        Q = Q_new


Q_star = value_iteration(gamma=1.0)
print(f"optimal return from start (value iteration): {Q_star[START].max():.0f}")

for method in ["q", "sarsa"]:
    curves, final = [], []
    for seed in range(10):
        Q, ret = run(method, 500, alpha=0.5, gamma=1.0, eps=0.1, rng=np.random.default_rng(seed))
        curves.append(ret); final.append(Q)
    curves = np.array(curves)
    path = greedy_path(final[0])
    print(f"{method:5s}: mean return while exploring, last 100 episodes {curves[:, -100:].mean():.1f} | "
          f"greedy path length {len(path) - 1} | min row of path {min(p[0] for p in path)}")

Q, _ = run("q", 500, 0.5, 1.0, 0.1, np.random.default_rng(0))
print(f"Q-learning greedy Q(start) {Q[START].max():.1f} vs optimal {Q_star[START].max():.1f}")
```

Output (numpy 2.5):
```
optimal return from start (value iteration): -13
q    : mean return while exploring, last 100 episodes -49.5 | greedy path length 13 | min row of path 2
sarsa: mean return while exploring, last 100 episodes -26.2 | greedy path length 17 | min row of path 0
Q-learning greedy Q(start) -13.0 vs optimal -13.0
```

Value iteration, which knows the rules, says the best possible return is −13: one step up, eleven along the row just above the cliff, one step down. Q-learning never sees the rules but its learned `Q(start)` matches −13 exactly, and its greedy path is that 13-step route along row 2, right next to the cliff. SARSA's greedy path takes 17 steps and climbs to the top row (row 0), far from the edge. The reason is in the targets. Q-learning values the greedy policy, which never steps off the cliff. SARSA values the ε-greedy policy it actually runs, which slips off the cliff with a random move 10% of the time, so cliff-adjacent cells look bad to it. While exploring, that makes SARSA earn −26 per episode against −50 for Q-learning, which keeps falling off its edge-hugging route. Set ε to 0 at deployment and Q-learning's path is the better one; keep exploring (or run in a world with real slips) and SARSA's is.

## Choosing / trade-offs
- **Q-learning vs SARSA.** Q-learning learns the optimal policy regardless of exploration and can learn from logged or replayed data (off-policy). SARSA is safer when the agent must keep exploring in the real system. Expected SARSA (average over the policy's action probabilities) has lower variance than either.
- **Learning rate α.** Constant α (0.1–0.5) tracks a changing environment; decaying α is needed for formal convergence in stochastic environments.
- **Discount γ.** Close to 1 for long-horizon tasks with episodes that end; below 1 for continuing tasks so returns stay finite. It also changes what "optimal" means.
- **Exploration.** ε-greedy with ε decayed over time is the default. Optimistic initial values, softmax (Boltzmann), and count-based bonuses (UCB-style) explore more systematically.
- **Model-free vs model-based.** If you know or can learn the transition model, value iteration or planning (Dyna, MCTS) needs far fewer real interactions.
- **Beyond tables.** Replace the table with a neural network (DQN) once states are too many to enumerate; that adds instability that replay buffers and target networks address.

## Gotchas
- Forgetting `(not done)` on terminal transitions bootstraps past the end and inflates values.
- `argmax` on an all-zero row always picks action 0. Break ties randomly early in training or exploration is biased.
- The on-policy vs off-policy difference shows up in the returns *while exploring*. Evaluate the greedy policy separately with ε = 0.
- Q-learning's max over noisy estimates is biased upward (maximisation bias). Double Q-learning keeps two tables to fix it; DQN variants do the same.
- Reward design steers everything. The −1 per step here is what makes the agent want a short path; a sparse reward only at the goal learns far more slowly.
- Results vary a lot between seeds. Average over several runs before comparing methods, as the code does with 10.
- Tabular methods do not generalise: an unvisited state keeps its initial value.

## Related
- [[reinforcement-learning]] - deep RL (DQN, PPO, SAC), libraries and environments.
- [[multi-armed-bandits]] - exploration vs exploitation with a single state.
- [[rlhf-and-preference-optimization]] - RL applied to fine-tuning language models.
- [[hmm-from-scratch]] - another dynamic program (Viterbi) over a sequence of states.
- [[optimizers-from-scratch-numpy]] - the TD update is a stochastic step toward a moving target.
- [[black-box-and-evolutionary-optimization]] - an alternative when you can only score whole policies.

## References
- Sutton and Barto, Reinforcement Learning: An Introduction, 2nd ed. (2018), Chapter 6 and Example 6.6: http://incompleteideas.net/book/the-book-2nd.html
- Watkins and Dayan (1992), "Q-learning", Machine Learning 8: https://doi.org/10.1007/BF00992698
- Gymnasium CliffWalking environment: https://gymnasium.farama.org/environments/toy_text/cliff_walking/
