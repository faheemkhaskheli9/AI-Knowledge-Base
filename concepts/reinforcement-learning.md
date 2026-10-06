---
title: Reinforcement learning
category: concepts
tags: [reinforcement-learning, rl, q-learning, dqn, policy-gradient, ppo, gymnasium, stable-baselines3, mdp]
use_cases:
  - "train an agent that learns a sequence of decisions from rewards (game, robot, controller)"
  - "optimise a control or scheduling policy in a simulator"
  - "understand the RL behind RLHF and reasoning-model training"
  - "decide whether a problem needs RL or plain supervised learning / optimisation"
status: draft
last_verified: 2026-10-03
sources:
  - http://incompleteideas.net/book/the-book-2nd.html
  - https://gymnasium.farama.org/
  - https://stable-baselines3.readthedocs.io/
  - https://spinningup.openai.com/en/latest/
  - https://arxiv.org/abs/1707.06347
---

# Reinforcement learning

## Summary
Reinforcement learning (RL) trains an **agent** to choose **actions** in an **environment** so as to maximise long-run **reward**, learning from trial and error rather than labelled answers. It fits sequential decision problems where an action changes what happens next: games, robotics, control, recommendation sequences, and the fine-tuning of LLMs from feedback. It is data-hungry and fragile, so the first question is always whether a simulator exists and whether a simpler method would do.

## Key concepts
- **MDP.** The formal setting: states `s`, actions `a`, transition `P(s'|s,a)`, reward `r`, discount `gamma` (0.9-0.999) that weights future reward. An **episode** runs from reset to a terminal state or time limit.
- **Policy** `pi(a|s)`: what the agent does. **Value** `V(s)` / **action value** `Q(s,a)`: expected discounted return from there. The goal is a policy with maximal return.
- **Exploration vs exploitation.** The agent must try uncertain actions to discover better ones; epsilon-greedy (random action with probability epsilon, decayed over training) is the simplest scheme.
- **Value-based.** Learn Q and act greedily. **Q-learning** updates `Q(s,a) += alpha * (r + gamma * max Q(s',.) - Q(s,a))` (temporal-difference learning). **DQN** replaces the table with a neural network plus a replay buffer and a target network; discrete actions only.
- **Policy-based.** Optimise the policy directly by gradient ascent on return (REINFORCE). **Actor-critic** adds a learned value function to reduce variance. **PPO** (clipped policy updates) is the robust default; **SAC/TD3** are the usual choices for continuous control.
- **On- vs off-policy.** On-policy (PPO, A2C) learns only from data of the current policy: stable, sample-hungry. Off-policy (DQN, SAC) reuses old data from a replay buffer: more sample-efficient, more tuning.
- **Model-free vs model-based.** Model-free learns from interaction alone; model-based learns or uses a model of the dynamics to plan (AlphaZero, MuZero, Dreamer).
- **Offline RL / bandits.** Offline RL learns from a fixed log without interaction. **Contextual bandits** are the one-step special case (choose an action, get a reward, no state change), the right tool for many recommendation and A/B-style problems.
- **RL for LLMs.** RLHF, DPO-style preference tuning and RL with verifiable rewards for reasoning reuse these ideas, with the LLM as the policy ([[rlhf-and-preference-optimization]], [[reasoning-models]]).

## When to use / scenarios
- **A simulator or cheap environment exists** and the decision is sequential: game AI, robot locomotion/manipulation in simulation then sim-to-real, HVAC or data-centre cooling control, traffic-signal timing, inventory and resource scheduling.
- **Bandits** for picking among options with immediate feedback: headline/creative selection, dynamic pricing tests, recommendation exploration ([[recommender-systems]]).
- **Fine-tuning LLMs** against a reward model or verifier.
- **Not when:** labelled correct actions exist (use supervised learning / imitation), the problem is a one-shot optimisation with known objective and constraints (use an OR solver: linear/integer programming, OR-Tools), or you can only experiment on the live system with real costs and no simulator (consider offline RL or bandits, carefully).

## Setup & code
```bash
pip install gymnasium numpy
```
Tabular Q-learning on Gymnasium's FrozenLake (slippery 4x4 grid; reward 1 for reaching the goal, 0 otherwise):
```python
import numpy as np
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=True)
rng = np.random.default_rng(0)
Q = np.zeros((env.observation_space.n, env.action_space.n))
alpha, gamma = 0.1, 0.99
episodes = 20_000

for ep in range(episodes):
    eps = max(0.05, 1 - ep / (0.8 * episodes))     # explore a lot early, little later
    s, _ = env.reset(seed=int(rng.integers(1e9)))
    done = False
    while not done:
        a = env.action_space.sample() if rng.random() < eps else int(Q[s].argmax())
        s2, r, terminated, truncated, _ = env.step(a)
        target = r + gamma * Q[s2].max() * (not terminated)   # no bootstrap past a terminal state
        Q[s, a] += alpha * (target - Q[s, a])
        s, done = s2, terminated or truncated

def success_rate(policy, n=1000):
    wins = 0
    for i in range(n):
        s, _ = env.reset(seed=i)
        done = False
        while not done:
            s, r, terminated, truncated, _ = env.step(policy(s))
            done = terminated or truncated
        wins += r
    return wins / n

print(f"random policy: {success_rate(lambda s: env.action_space.sample()):.2f}")
print(f"greedy Q policy: {success_rate(lambda s: int(Q[s].argmax())):.2f}")
```
On a test run (gymnasium 1.0.0, numpy 2.5) the random policy reached the goal about 2% of the time and the learned greedy policy about 75%; the ice is slippery, so no policy succeeds every time.

For larger or continuous state spaces use a maintained library instead of writing DQN/PPO yourself. Stable-Baselines3 (PyTorch) trains a PPO agent in a few lines:
```bash
pip install stable-baselines3
```
```python
from stable_baselines3 import PPO
model = PPO("MlpPolicy", "CartPole-v1", verbose=0, seed=0).learn(total_timesteps=50_000)
model.save("ppo_cartpole")
```

## Choosing / trade-offs
- **Algorithm.** Discrete actions: PPO first, DQN if samples are expensive. Continuous actions: SAC (sample-efficient) or PPO (simpler, stabler). Small discrete state space: tabular Q-learning.
- **Library.** Stable-Baselines3 for reliable single-machine baselines; RLlib (Ray) for distributed training; CleanRL for readable single-file implementations to learn from or modify; TRL for RL on language models.
- **Reward design vs imitation.** A hand-written reward is easy to start but gets gamed; if demonstrations exist, behaviour cloning or imitation learning is faster and safer, often followed by RL fine-tuning.
- **Sim fidelity vs cost.** A fast, simplified simulator gives many samples but a sim-to-real gap; domain randomisation (vary friction, delays, sensor noise) narrows it.

## Gotchas
- **Reward hacking.** The agent maximises the reward you wrote, not the one you meant (circling to farm a bonus, exploiting a simulator bug). Watch rollouts, not just the curve.
- Results vary a lot with the random seed; report several seeds before concluding that one setting beats another.
- Distinguish `terminated` (true end, value 0 after it) from `truncated` (time limit, should still bootstrap); mixing them biases value estimates. Gymnasium returns them separately for this reason.
- Normalise observations and rewards for neural-network agents; unscaled inputs are a common silent failure.
- Sparse rewards (only at the goal) can mean the agent never sees a reward; use shaping, curricula or demonstrations.
- An agent evaluated with exploration still on (epsilon > 0, stochastic policy) looks worse than it is; evaluate the greedy/deterministic policy separately.
- Offline RL overestimates actions the log never tried; use methods built for it (CQL, IQL), not plain off-policy algorithms.

## Related
- [[ml-fundamentals]] - where RL sits next to supervised and unsupervised learning.
- [[neural-network-fundamentals]] - function approximators for deep RL.
- [[rlhf-and-preference-optimization]] - RL applied to aligning LLMs.
- [[reasoning-models]] - RL with verifiable rewards for reasoning.
- [[recommender-systems]] - bandits for exploration in recommendations.
- [[computer-use-agents]] - LLM agents, a different meaning of "agent".
- [[dqn-from-scratch-numpy]] - DQN built in NumPy: replay buffer, target network, double DQN on CartPole.
- [[mcts-from-scratch]] - Monte Carlo tree search (UCT) built by hand on tic-tac-toe.

## References
- Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed.: http://incompleteideas.net/book/the-book-2nd.html
- Gymnasium (environment API): https://gymnasium.farama.org/
- Stable-Baselines3 docs: https://stable-baselines3.readthedocs.io/
- OpenAI Spinning Up (deep RL introduction): https://spinningup.openai.com/en/latest/
- Schulman et al., Proximal Policy Optimization: https://arxiv.org/abs/1707.06347
