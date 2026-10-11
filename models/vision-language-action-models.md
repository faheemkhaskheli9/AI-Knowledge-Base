---
title: Vision-language-action (VLA) and robot foundation models
category: models
tags: [vla, robotics, robot-foundation-model, openvla, pi0, gr00t, smolvla, lerobot, action-tokenization, sim-to-real, imitation-learning]
use_cases:
  - "pick an open robot policy model to fine-tune for a pick-and-place arm"
  - "understand how a VLM is turned into a policy that outputs robot actions"
  - "estimate how many demonstrations and how much GPU a VLA fine-tune needs"
  - "plan the safety envelope around a learned policy on a factory cell"
status: draft
last_verified: 2026-10-11
sources:
  - https://arxiv.org/abs/2307.15818
  - https://arxiv.org/abs/2406.09246
  - https://arxiv.org/abs/2410.24164
  - https://arxiv.org/abs/2501.09747
  - https://arxiv.org/abs/2506.01844
  - https://arxiv.org/abs/2310.08864
  - https://arxiv.org/abs/2503.14734
  - https://github.com/Physical-Intelligence/openpi
  - https://github.com/NVIDIA/Isaac-GR00T
  - https://huggingface.co/lerobot/smolvla_base
  - https://ai.google.dev/gemini-api/docs/robotics-overview
  - https://www.automate.org/robotics/blogs/updated-iso-10218-faq
---

# Vision-language-action (VLA) and robot foundation models

## Summary
A vision-language-action model takes camera images, robot state and a language
instruction and outputs robot actions (end-effector or joint commands). It
starts from a pretrained vision-language model, so it inherits object and
instruction understanding from web data, and is then trained on robot
demonstrations. RT-2 (2023) named the category; open models you can download
and fine-tune now include OpenVLA, Physical Intelligence's pi0 family,
NVIDIA GR00T and Hugging Face SmolVLA. They are imitation-learned policies:
good at generalising a skill to new objects and phrasing, not certified
controllers.

## Key concepts
- **Action tokenisation (discrete)**: RT-2 discretises each continuous action
  dimension (6-DoF end-effector delta, gripper) into 256 uniform bins and maps
  bins to existing vocabulary tokens, so the VLM predicts actions as text.
  OpenVLA uses the same per-dimension binning.
- **Compressed tokenisation (FAST)**: per-dimension, per-timestep bins work
  poorly for high-frequency dexterous control. FAST applies a discrete cosine
  transform to a chunk of actions, quantises the coefficients and compresses
  them with BPE; the paper reports matching diffusion VLAs with up to 5x less
  training time. FAST+ is a universal tokenizer trained on 1M trajectories.
- **Continuous action heads**: pi0 attaches a flow-matching "action expert" to
  the VLM and emits continuous action chunks; GR00T N1 pairs a VLM ("System
  2") with a diffusion-transformer action module ("System 1"). These avoid quantisation error and suit fast control loops.
- **Action chunking**: predict several future steps at once and execute them
  while the next chunk is computed. SmolVLA adds an asynchronous inference
  stack for this so control rate is not capped by model latency.
- **Cross-embodiment data**: Open X-Embodiment pooled data from 22 robots
  (21 institutions, 527 skills); OpenVLA was trained on 970k real
  demonstrations from it. A new robot still needs its own demonstrations.
- **Hierarchy**: a reasoning model plans and checks ("embodied reasoning"),
  a VLA executes. Google's Gemini Robotics ER models play the planner role and
  are callable through the Gemini API; they are not open weights.

## Open and API models (2026-10)
| Model | Size | Access | Notes (source) |
|---|---|---|---|
| OpenVLA | 7B | open weights + code | Llama 2 + DINOv2/SigLIP encoders; LoRA fine-tune on consumer GPUs, quantised serving (paper) |
| pi0, pi0-FAST, pi0.5 (`openpi`) | - | Apache-2.0 repo | base and DROID/ALOHA/LIBERO fine-tuned checkpoints; inference > 8 GB, LoRA fine-tune > 22.5 GB, full fine-tune > 70 GB GPU memory (repo README) |
| NVIDIA Isaac GR00T N1.7 | 3B | code Apache 2.0, weights NVIDIA Open Model License | humanoid and arm embodiments; inference 16 GB+ GPU, fine-tune 40 GB+ recommended; LeRobot v2 data plus `meta/modality.json` (repo README) |
| SmolVLA (`lerobot/smolvla_base`) | 0.5B | open, LeRobot | trains on one GPU, runs on consumer GPUs or CPU (paper, model card) |
| Gemini Robotics ER (`gemini-robotics-er-2-preview`) | - | Gemini API preview | embodied reasoning/planning, not a low-level controller (Gemini API docs) |

## When to use / scenarios
- Manipulation with variety: many SKUs, changing bins, language-specified
  tasks ("put the red connectors in tray B") where scripting every case is
  impractical.
- Research and prototyping on low-cost arms (SO-100 class) with LeRobot data.
- Not for fixed, repetitive motions: a programmed trajectory with classic
  vision (pose estimation, [[object-detection]]) is cheaper, faster and
  certifiable.
- Not as a safety function. The policy runs inside a safety-rated envelope
  (see Gotchas), never as the thing that stops the robot hitting a person.
- Planning-only problems (task sequencing, reading a scene) may need just a
  VLM ([[vision-language-models]]) driving a classical motion planner.

## Setup & code
Action tokenisation in NumPy: RT-2 style 256-bin per-dimension tokens versus a
FAST-style DCT over an action chunk.
```python
import numpy as np

rng = np.random.default_rng(0)
T, D = 50, 7                      # 50 control steps, 7-DoF action chunk
t = np.linspace(0, 1, T)[:, None]
chunk = np.sin(2 * np.pi * t * rng.uniform(0.5, 2, D)) * 0.8  # smooth, in [-1, 1]

# RT-2 / OpenVLA: 256 uniform bins per dimension per step
low, high, n_bins = -1.0, 1.0, 256
bins = np.clip(((chunk - low) / (high - low) * n_bins).astype(int), 0, n_bins - 1)
decoded = low + (bins + 0.5) * (high - low) / n_bins
print("binning tokens:", bins.size, " max error:", np.abs(decoded - chunk).max().round(4))

# FAST idea: DCT along time, round scaled coefficients, keep the non-zeros
n = np.arange(T)
dct = np.cos(np.pi / T * (n[None, :] + 0.5) * n[:, None]) * np.sqrt(2 / T)
dct[0] /= np.sqrt(2)              # orthonormal DCT-II matrix
coef = np.round(dct @ chunk * 10)  # scale sets the precision/length trade-off
recon = dct.T @ (coef / 10)
print("DCT non-zero coefficients:", int(np.count_nonzero(coef)),
      " max error:", np.abs(recon - chunk).max().round(4))
assert np.count_nonzero(coef) < bins.size  # smooth motion compresses
```
FAST then runs BPE over the integer coefficients, so the token count is lower
still. Fine-tuning SmolVLA on your own LeRobot dataset (per its model card;
needs a GPU and a recorded dataset on the Hub):
<!-- skip-check: shell commands; needs a GPU and a LeRobot dataset -->
```bash
pip install "lerobot[smolvla]"
lerobot-train --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=$HF_USER/my_pick_place --policy.repo_id=$HF_USER/my_policy \
  --output_dir=./outputs/run1 --job_name=run1 \
  --policy.device=cuda --steps=100000 --batch_size=4
```

## Choosing / trade-offs
- **Size vs latency**: a 3-7B VLA needs a 16 GB-class GPU next to the robot
  and limits how often a new chunk arrives; SmolVLA-class models trade
  generality for speed and cheaper fine-tuning ([[edge-on-device]]).
- **Discrete vs continuous actions**: binned tokens reuse the LLM training
  stack and are simple; flow/diffusion heads and FAST handle high-frequency,
  dexterous control better.
- **Licence**: openpi is Apache-2.0; GR00T code is Apache 2.0 but its weights
  use the NVIDIA Open Model License. Read each model card's terms before
  shipping ([[model-licenses]]).
- **Data budget**: demonstrations (teleoperation) are the dominant cost. Start
  from the closest pretrained embodiment checkpoint, collect task-specific
  episodes, and measure real success rate as you add data; the count needed
  depends on task variety, so no published number transfers.

## Gotchas
- **Sim-to-real gap**: policies trained or evaluated only in simulation meet
  different lighting, friction, latency and camera placement on hardware.
  Domain randomisation and a small real dataset help; always gate on real
  trials. Simulation benchmarks such as LIBERO rank models, they do not
  certify them.
- **Embodiment mismatch**: action spaces differ (joint vs end-effector,
  gripper encoding, control frequency). Normalise actions with the statistics
  the checkpoint expects, or the policy outputs nonsense.
- **Safety limits**: learned policies have no guarantees. Keep the robot's
  safety-rated functions (monitored stop, speed and separation monitoring,
  power and force limiting) in the controller. ISO 10218-1/-2:2025 now carries
  the collaborative-robot requirements formerly in ISO/TS 15066; follow your
  integrator's risk assessment, and the EU Machinery Regulation / AI Act where
  they apply ([[ai-security-privacy-compliance]]).
- **Language is an attack surface**: an instruction or a sign in the camera
  view can redirect the policy ([[prompt-injection]]); restrict who can issue
  commands and bound the workspace.
- Reported success rates come from the authors' own robots and tasks; they
  rarely transfer as numbers.

## Related
- [[vision-language-models]] - the perception backbone a VLA starts from.
- [[reinforcement-learning]] - RL fine-tuning of policies and the sim-to-real literature.
- [[multimodal-models]] - how image and text tokens share one model.
- [[edge-on-device]] - running the policy next to the robot.
- [[manufacturing-iot]] - industrial scenarios and safety context.

## References
- Brohan et al., RT-2 (2023): https://arxiv.org/abs/2307.15818
- Kim et al., OpenVLA (2024): https://arxiv.org/abs/2406.09246
- Black et al., pi0 (2024): https://arxiv.org/abs/2410.24164
- Pertsch et al., FAST (2025): https://arxiv.org/abs/2501.09747
- Shukor et al., SmolVLA (2025): https://arxiv.org/abs/2506.01844
- Open X-Embodiment (2023): https://arxiv.org/abs/2310.08864
- NVIDIA, GR00T N1 (2025): https://arxiv.org/abs/2503.14734
- openpi repo: https://github.com/Physical-Intelligence/openpi
- Isaac GR00T repo: https://github.com/NVIDIA/Isaac-GR00T
- SmolVLA model card: https://huggingface.co/lerobot/smolvla_base
- Gemini Robotics ER: https://ai.google.dev/gemini-api/docs/robotics-overview
- ISO 10218:2025 FAQ (A3): https://www.automate.org/robotics/blogs/updated-iso-10218-faq
