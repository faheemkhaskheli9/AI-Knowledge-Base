---
title: Automotive and robotics AI
category: scenarios
tags: [automotive, robotics, autonomous-driving, adas, perception, vla, simulation, sim-to-real, edge, functional-safety, ros2]
use_cases:
  - "build a perception stack (detection, segmentation, tracking) for a vehicle or mobile robot"
  - "pick between a scripted robot program, a classic perception + planner stack and a VLA policy"
  - "train or test a driving or manipulation policy in simulation before touching hardware"
  - "fit a model into an embedded GPU's latency and power budget"
  - "add a voice or language interface to a car or robot without compromising safety"
status: draft
last_verified: 2026-10-15
sources:
  - https://www.iso.org/standard/68383.html
  - https://www.iso.org/standard/77490.html
  - https://www.sae.org/standards/content/j3016_202104/
  - https://carla.org/
  - https://mujoco.org/
  - https://developer.nvidia.com/isaac/sim
  - https://docs.ros.org/en/rolling/
  - https://github.com/huggingface/lerobot
  - https://eur-lex.europa.eu/eli/reg/2024/1689/oj
---

# Automotive and robotics AI

## Summary
Vehicles and robots are where AI meets physics: perception (cameras,
lidar, radar) feeds planning and control under hard real-time, power and
safety constraints. The classic stack (perception -> tracking -> prediction
-> planning -> control) is still what ships in safety-critical products;
learned end-to-end policies, including vision-language-action (VLA) models,
are spreading in manipulation and research driving stacks. Simulation is
the main training and test ground, and every learned component sits inside
a deterministic safety envelope.

## Key concepts
- **Modular vs end-to-end**: modular stacks are testable per component and
  map onto safety standards; end-to-end policies (camera -> action) handle
  long-tail variety better but are harder to verify. Hybrids are common: a
  learned planner or VLA bounded by rule-based limits
  ([[vision-language-action-models]]).
- **Perception tasks**: 2D/3D detection, segmentation, tracking across
  frames, sensor fusion and state estimation ([[object-detection]],
  [[segmentation]], [[video-analytics]],
  [[hidden-markov-models-and-kalman-filters]]).
- **Functional safety and SOTIF**: ISO 26262 covers hazards from
  malfunctioning electrical/electronic systems in road vehicles; ISO 21448
  (SOTIF) covers hazards from a function that works as designed but is
  insufficient, which is where ML perception errors live. Industrial robots
  follow ISO 10218 (see [[vision-language-action-models]]).
- **Autonomy levels**: SAE J3016 defines driving-automation levels 0-5; the
  level decides who is responsible for monitoring and what the system must
  handle.
- **Edge budget**: inference runs on an embedded accelerator with a fixed
  latency, power and thermal budget, often without network. Quantise, prune
  and compile for the target ([[edge-on-device]], [[quantization]],
  [[knowledge-distillation-and-compression]]).
- **Simulation and sim-to-real**: simulators (CARLA for driving, MuJoCo and
  NVIDIA Isaac Sim for robots) generate data and scenarios that are rare or
  dangerous to collect; domain randomisation narrows the reality gap, real
  trials close it ([[reinforcement-learning]], [[transfer-learning-and-domain-adaptation]]).
- **Data engine**: fleet or robot logs -> mine hard cases (disagreement,
  disengagements, near misses) -> label -> retrain -> regression-test in sim
  and replay ([[semi-supervised-and-active-learning]], [[data-labeling-and-synthetic-data]]).

## When to use / scenarios
1. **Perception for ADAS or a mobile robot.** Approach: pretrained
   detector/segmenter fine-tuned on your sensors, tracking with a Kalman
   filter, evaluated per scenario class (night, rain, occlusion). Read:
   [[object-detection]], [[segmentation]],
   [[hungarian-algorithm-tracking-from-scratch]], [[kalman-filter-from-scratch]].
2. **Manipulation with high variety (warehouse picking, kitting).**
   Approach: VLA or imitation-learned policy fine-tuned on teleoperated
   demonstrations, inside a safety-rated controller. Read:
   [[vision-language-action-models]], [[multimodal-models]].
3. **Fixed, repetitive robot motions.** Approach: programmed trajectories +
   classic vision for part pose; no learned policy needed. Read:
   [[manufacturing-iot]], [[face-and-pose]] (pose estimation ideas),
   [[ransac-homography-from-scratch]].
4. **Policy learning in simulation.** Approach: RL or imitation in a
   simulator with domain randomisation, then small real fine-tune. Read:
   [[reinforcement-learning]], [[ppo-from-scratch]],
   [[physics-informed-neural-networks]] (fast simulation surrogates).
5. **Scenario-based testing and validation.** Approach: generate
   variations of logged and synthetic scenarios in simulation, measure
   failure rates per scenario class; feed failures back to the data engine.
   Read: [[data-labeling-and-synthetic-data]], [[out-of-distribution-detection]].
6. **Predictive maintenance for fleets and robot cells.** Read:
   [[manufacturing-iot]], [[anomaly-detection]], [[survival-analysis]].
7. **In-cabin and robot voice assistants.** Approach: on-device speech
   recognition and a small model for commands; cloud LLM only for
   non-safety features. Read: [[voice-agents]], [[speech-to-text]],
   [[small-language-models]].
8. **High-level task planning.** Approach: an LLM/VLM plans steps and
   checks scene state, a classical planner or VLA executes. Read:
   [[agents]], [[vision-language-models]].
9. **Engineering copilots** over requirements, test logs and standards
   documents. Read: [[rag-basics]], [[document-processing]].
- **When NOT to use a learned model**: as the only safety function
  (emergency stop, speed and separation monitoring), or where a scripted
  solution meets the spec; a learned component adds a validation burden.

## Setup & code
Reference architecture (flagship: learned policy on a robot):

```
sensors -> perception (detect/segment/track, on-device) -> state
  -> policy (VLA / planner, 1-10 Hz chunks)   [learned, best effort]
  -> safety filter (speed, force, workspace limits)  [deterministic]
  -> controller (50-1000 Hz)  -> actuators
  safety-rated stop and monitoring in the controller/PLC, independent of the AI
  logs -> hard-case mining -> sim regression suite -> retrain
```

Runnable numpy sketch of the edge constraint: the policy's inference is
slower than the control loop, so actions arrive in chunks asynchronously; a
deterministic filter clips speeds and a watchdog holds position when the plan
runs out.

```python
import numpy as np

rng = np.random.default_rng(1)
CONTROL_HZ, CHUNK = 50, 25            # controller runs at 50 Hz; policy emits 25-step chunks
V_MAX, STALE_STEPS = 0.25, 30         # joint speed limit (rad/s); hold-to-stop if no fresh plan
dt = 1 / CONTROL_HZ

def policy_chunk(q):                   # stand-in for a VLA: noisy, sometimes too fast
    target = np.array([0.8, -0.4, 0.3])
    v = (target - q) * rng.uniform(1, 6) + rng.normal(0, 0.2, 3)
    return np.repeat(v[None], CHUNK, axis=0)

def safety_filter(v):                  # deterministic, outside the learned model
    return np.clip(v, -V_MAX, V_MAX)

q, plan, age, stops, clipped, v_peak = np.zeros(3), None, 0, 0, 0, 0.0
latency_steps = rng.integers(5, 40, size=100)   # inference time in control ticks; tail is slow
pending, k = None, 0
for step in range(600):
    if pending is None:                          # start async inference on the current state
        pending, ready_at = policy_chunk(q.copy()), step + latency_steps[k % 100]; k += 1
    if step >= ready_at:                         # new chunk arrives
        plan, age, pending = pending, 0, None
    if plan is None or age >= min(CHUNK, STALE_STEPS):
        v = np.zeros(3); stops += 1              # watchdog: plan exhausted -> hold position
    else:
        raw = plan[age]; v = safety_filter(raw); clipped += int(np.any(raw != v)); age += 1
    q = q + v * dt
    v_peak = max(v_peak, np.abs(v).max())

print("final q:", q.round(3), "(target 0.8, -0.4, 0.3; open-loop chunks overshoot)")
print("hold ticks:", stops, "clipped ticks:", clipped, "peak speed:", round(v_peak, 3))
assert v_peak <= V_MAX and stops > 0 and clipped > 0
```

ROS 2 is the usual middleware for wiring these stages together on robots;
LeRobot provides datasets, teleoperation recording and training scripts for
low-cost arms (see [[vision-language-action-models]] for a fine-tune command).

## Choosing / trade-offs
- **Modular vs end-to-end**: modular when you must certify, debug per
  component, or reuse an existing safety case; end-to-end when variety
  dominates and you can collect large amounts of demonstration data.
- **Model size vs latency and power**: bigger models generalise better but
  lower the control rate; distil or pick small models for the in-loop path,
  keep large ones for offline labelling and planning.
- **Simulation vs real data**: sim is cheap, safe and scalable; real data
  is the ground truth. Budget for both and track the sim-to-real gap as a
  metric.
- **On-board vs cloud**: anything in the control loop runs on-board;
  cloud is for fleet learning, map updates and non-critical features.
- **Open vs proprietary stacks**: open models and simulators speed up
  prototyping; check licences for commercial and automotive use
  ([[model-licenses]]).

## Gotchas
- Average metrics hide the long tail: a detector with high mAP can still
  miss rare classes (debris, unusual vehicles) that cause incidents; report
  per scenario class ([[nms-iou-and-map-from-scratch]]).
- Sensor and calibration drift (a bumped camera, new firmware) silently
  degrades perception; monitor input statistics ([[online-learning-and-concept-drift]]).
- Action chunks executed open loop overshoot when the world changes during
  inference (the sketch above overshoots its target); keep chunks short or
  re-plan on new observations.
- Timing jitter on shared GPUs breaks control assumptions; measure tail
  latency on the target hardware, not the dev box ([[cost-and-latency]]).
- Simulation success rarely transfers as a number; gate releases on real
  trials and replayed logs.
- Language and visual inputs are an attack surface (adversarial patches,
  injected instructions); restrict who can command the system
  ([[adversarial-examples-and-robustness]], [[prompt-injection]]).
- Regulation: vehicle type approval, machinery rules and the EU AI Act
  (safety components of regulated products can be high-risk) apply; involve
  safety and homologation engineers early ([[ai-security-privacy-compliance]]).

## Related
- [[vision-language-action-models]] - VLA models, action tokenisation, robot safety standards.
- [[manufacturing-iot]] - factory cells, visual inspection, edge deployment.
- [[edge-on-device]] - fitting models onto embedded accelerators.
- [[reinforcement-learning]] - sim-based policy learning.
- [[low-budget-ai]] - low-cost arms and open models for prototyping.

## References
- ISO 26262-1:2018, road vehicles functional safety (vocabulary): https://www.iso.org/standard/68383.html
- ISO 21448:2022, safety of the intended functionality (SOTIF): https://www.iso.org/standard/77490.html
- SAE J3016, levels of driving automation: https://www.sae.org/standards/content/j3016_202104/
- CARLA open-source driving simulator: https://carla.org/
- MuJoCo physics engine: https://mujoco.org/
- NVIDIA Isaac Sim: https://developer.nvidia.com/isaac/sim
- ROS 2 documentation: https://docs.ros.org/en/rolling/
- Hugging Face LeRobot: https://github.com/huggingface/lerobot
- Regulation (EU) 2024/1689 (AI Act): https://eur-lex.europa.eu/eli/reg/2024/1689/oj
