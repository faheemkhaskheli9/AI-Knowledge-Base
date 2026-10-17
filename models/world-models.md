---
title: World models (predictive and interactive)
category: models
tags: [world-models, video-world-models, cosmos, genie, sana-wm, dreamer, v-jepa, model-based-rl, robotics, simulation]
use_cases:
  - "generate synthetic driving or robot video to augment training data"
  - "let a robot plan by imagining the outcome of actions before acting"
  - "prototype an explorable game level from a single image and a prompt"
  - "decide between a physics simulator and a learned world model"
  - "run an open camera-controlled world model on one GPU"
status: draft
last_verified: 2026-10-15
sources:
  - https://arxiv.org/abs/1803.10122
  - https://arxiv.org/abs/2301.04104
  - https://arxiv.org/abs/2506.09985
  - https://arxiv.org/abs/2605.15178
  - https://www.nvidia.com/en-us/ai/cosmos/
  - https://docs.nvidia.com/cosmos/latest/
  - https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/
  - https://blog.google/innovation-and-ai/models-and-research/google-deepmind/project-genie/
---

# World models (predictive and interactive)

## Summary
A world model learns how an environment changes: given the current state (or
video frames) and an action, it predicts what happens next. Agents use it to
plan or train "in imagination" instead of in the real world, and generative
versions render the prediction as video you can steer interactively. Uses
include robot planning, synthetic training data for robots and autonomous
vehicles, and game/simulation prototyping. They are not physics engines:
predictions drift over long horizons and can violate physics.

## Key concepts
- **Core interface**: `next_state = f(state, action)` (often probabilistic).
  Video world models condition a video generator on actions: keyboard/controller
  input, camera trajectory, or robot commands.
- **Families**:
  - *Latent-dynamics models for RL*: learn compact latent dynamics and train a
    policy inside them (Ha & Schmidhuber's "World Models", DreamerV3).
    Small, fast, task-specific ([[reinforcement-learning]]).
  - *Predictive representation models*: predict in embedding space rather than
    pixels (V-JEPA 2, which Meta also uses for zero-shot robot planning).
  - *World foundation models for physical AI*: large video models pre-trained
    for robotics and driving. NVIDIA Cosmos has Predict (future video from
    text/image/video), Transfer (sim or control inputs -> photoreal video) and
    Reason (a vision-language model for physical reasoning), released under the
    NVIDIA Open Model License (NVIDIA Cosmos page).
  - *Interactive video world models*: generate an explorable world in real
    time as the user moves. Google DeepMind's Genie 3 (announced August 2025;
    text prompt -> navigable world at 720p, 24 fps, per DeepMind) is
    available as the Project Genie research prototype to Google AI Ultra
    subscribers in the US, with sessions limited to 60 seconds (Google blog).
    NVIDIA's SANA-WM (2026 paper) is an open 2.6B camera-controlled model that
    generates up to one-minute 720p video from an image and a 6-DoF camera path,
    built for single-GPU efficiency (arXiv 2605.15178).
- **Compounding error**: each predicted step feeds the next, so errors grow
  with horizon. Planning uses short horizons and re-plans from real
  observations (model-predictive control).
- **Memory/consistency**: revisiting a place should look the same; many video
  world models lack persistent 3D memory and drift when the camera loops back.

## When to use / scenarios
- Robotics: plan actions by imagining outcomes; augment demonstrations with
  varied lighting/backgrounds (Cosmos Transfer style); pair with policies from
  [[vision-language-action-models]] ([[manufacturing-iot]]).
- Autonomous driving: generate rare scenarios (weather, near-misses) for
  training and testing perception.
- Games and media: explorable concept prototypes, previs, interactive
  experiences; for fixed clips use [[video-generation-models]].
- RL research: sample-efficient learning when real interaction is costly.
- Not for: safety certification or anything needing exact physics (use a
  physics simulator: MuJoCo, Isaac Sim and similar), long sessions needing
  stable world state, or shipping a game engine replacement.

## Setup & code
The core loop in numpy (runs offline): learn a dynamics model from random
interactions, use it for model-predictive control, and measure how prediction
error compounds with horizon.
```python
import numpy as np

rng = np.random.default_rng(0)
A_true = np.array([[1.0, 0.1], [0.0, 0.95]])     # position, velocity
B_true = np.array([[0.0], [0.1]])

def env_step(s, a):
    return A_true @ s + B_true @ a + 0.01 * rng.standard_normal(2)

# 1. Collect random experience
S, U, S2 = [], [], []
s = np.zeros(2)
for _ in range(500):
    a = rng.uniform(-1, 1, size=1)
    s2 = env_step(s, a)
    S.append(s); U.append(a); S2.append(s2)
    s = s2 if abs(s2[0]) < 5 else np.zeros(2)
X = np.hstack([np.array(S), np.array(U)])
W, *_ = np.linalg.lstsq(X, np.array(S2), rcond=None)   # learned world model
model = lambda s, a: np.concatenate([s, a]) @ W

# 2. Compounding error: open-loop imagined rollout vs reality
s_real = s_img = np.array([0.0, 0.0])
errs = []
for h in range(1, 41):
    a = np.array([np.sin(h / 5)])
    s_real, s_img = env_step(s_real, a), model(s_img, a)
    errs.append(np.linalg.norm(s_real - s_img))
print("imagination error at horizon 1/10/40:", [round(errs[i], 3) for i in (0, 9, 39)])

# 3. Plan with the model (random-shooting MPC), act in the real env, re-plan
goal, s = 2.0, np.zeros(2)
for t in range(60):
    plans = rng.uniform(-1, 1, size=(256, 10, 1))
    costs = np.zeros(256)
    for i, plan in enumerate(plans):
        x = s
        for a in plan:
            x = model(x, a)
            costs[i] += (x[0] - goal) ** 2 + 0.01 * a[0] ** 2
    s = env_step(s, plans[costs.argmin(), 0])    # execute only the first action
print(f"final position {s[0]:.2f} (goal {goal})")
assert abs(s[0] - goal) < 0.3
```
Large world models are used through their own repos and checkpoints (NVIDIA
Cosmos docs, the SANA code in NVLabs' Sana repository); they need data-centre
or high-end consumer GPUs and multi-GB downloads.

## Choosing / trade-offs
| Need | Lean toward |
|---|---|
| Exact, verifiable physics; safety testing | classical simulator (optionally with learned rendering on top) |
| Photoreal variety for perception training | world foundation model (e.g. Cosmos Transfer/Predict) |
| Sample-efficient control on one task | latent-dynamics RL (Dreamer-style) |
| Planning from video without pixel generation | embedding-space predictor (V-JEPA 2 style) |
| Interactive explorable demo | interactive video world model (Genie, SANA-WM) |

- **Open vs closed**: Cosmos and SANA-WM publish weights/code you can
  fine-tune on your robot or camera rig; Genie 3 is a closed research
  prototype with no general API.
- **Horizon vs fidelity**: minute-long generation trades off against drift
  and compute; short-horizon prediction with frequent re-planning is more
  reliable for control.
- **Licence**: check each checkpoint's licence; code and weights are often
  licensed differently ([[model-licenses]]).

## Gotchas
- Generated "data" inherits the model's blind spots; validate on real-world
  test sets and never evaluate on synthetic data alone.
- Imagined rollouts can be physically wrong yet look plausible; a policy will
  exploit model errors if trained only in imagination.
- Action conditioning is often coarse (camera paths, a few discrete keys);
  fine robot joint control needs task-specific fine-tuning.
- Latency: real-time interaction needs distilled/few-step models and strong
  GPUs; reported throughput figures are vendor benchmarks on specific hardware.
- The term "world model" is used loosely in marketing for ordinary video
  generators; check that the model actually takes actions as input.

## Related
- [[vision-language-action-models]] - robot policies that world models help train and plan for.
- [[video-generation-models]] - non-interactive text/image-to-video.
- [[reinforcement-learning]] - model-based RL and policy learning.
- [[diffusion-models]] - the generator inside most video world models.
- [[edge-on-device]] - running models next to robots.
- [[manufacturing-iot]] - industrial robotics scenarios.

## References
- Ha & Schmidhuber, World Models (2018): https://arxiv.org/abs/1803.10122
- Hafner et al., DreamerV3 (2023): https://arxiv.org/abs/2301.04104
- Assran et al., V-JEPA 2 (2025): https://arxiv.org/abs/2506.09985
- Zhu et al., SANA-WM (2026): https://arxiv.org/abs/2605.15178
- NVIDIA Cosmos: https://www.nvidia.com/en-us/ai/cosmos/ and docs https://docs.nvidia.com/cosmos/latest/
- Google DeepMind, Genie 3: https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/
- Google, Project Genie: https://blog.google/innovation-and-ai/models-and-research/google-deepmind/project-genie/
