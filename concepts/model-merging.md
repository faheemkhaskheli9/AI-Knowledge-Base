---
title: Model merging (SLERP, task arithmetic, TIES, DARE)
category: concepts
tags: [model-merging, task-arithmetic, slerp, ties, dare, mergekit, fine-tuning, weight-averaging]
use_cases:
  - "combine a code fine-tune and a chat fine-tune of the same base model into one checkpoint without retraining"
  - "merge several LoRA or full fine-tunes of one base so we serve a single model instead of many"
  - "recover general ability lost after fine-tuning on a narrow domain by merging back toward the base"
  - "decide whether model merging or multi-task fine-tuning fits our compute budget"
status: draft
last_verified: 2026-10-15
sources:
  - https://arxiv.org/abs/2212.04089
  - https://arxiv.org/abs/2306.01708
  - https://arxiv.org/abs/2311.03099
  - https://arxiv.org/abs/2203.05482
  - https://arxiv.org/abs/2403.13257
  - https://github.com/arcee-ai/mergekit
---

# Model merging (SLERP, task arithmetic, TIES, DARE)

## Summary
Model merging builds one model by combining the weights of several fine-tunes of the same base model, with no gradient training and no training data. It is cheap (mergekit runs on a CPU or a modest GPU) and lets you fold skills from separate fine-tunes into one checkpoint, but it only works when the models share an architecture and, in practice, a common ancestor, and the result must be evaluated because merges can silently lose a skill.

## Key concepts
- **Task vector**: `tau = w_finetuned - w_base`. Fine-tuning moves weights by a task vector; merging manipulates and adds these vectors (Ilharco et al., task arithmetic).
- **Linear / model soup**: weighted average of whole weights. Averaging several fine-tunes of one base from the same run often matches or beats the best single one (Wortsman et al., model soups). Averaging dilutes each task vector by its weight.
- **SLERP**: spherical linear interpolation between exactly two models, per tensor. Interpolates along the arc between the two weight vectors, which keeps their norm better than a straight average; `t=0` is one model, `t=1` the other.
- **Task arithmetic**: `w = w_base + lambda * sum(tau_i)`. Adding vectors adds skills; subtracting one ("negation") removes a behaviour. `lambda` is tuned on a validation set.
- **TIES-merging**: three steps on task vectors: **trim** each to its largest-magnitude fraction (`density`), **elect** a sign per parameter by total mass, then **disjoint mean** of the entries that agree with that sign. Reduces interference from redundant small updates and sign conflicts.
- **DARE**: randomly **drop** a fraction `p` of each task vector's entries and **rescale** the rest by `1/(1-p)`. Works when fine-tuning deltas are highly redundant (typical for LLM SFT); usually combined with TIES sign election (`dare_ties`) or a linear sum (`dare_linear`).
- **Passthrough / frankenmerges**: stacking layers from different models into a deeper model. Not weight averaging; usually needs further training to be good.
- **Linear mode connectivity**: averaging only works when the models sit in one low-loss basin, which in practice means they were fine-tuned from the same pretrained checkpoint.

## When to use / scenarios
- You have several fine-tunes of one base (e.g. a domain SFT, an instruction SFT, a tool-use SFT) and want one deployable model.
- A narrow fine-tune hurt general ability: interpolate back toward the base (or the instruct model) and pick the point on your eval that keeps most of both.
- Cheap exploration: try many combinations before paying for multi-task fine-tuning.
- Real-world: a legal-tech team merges a contract-clause fine-tune with the vendor's instruct model to keep chat quality; a support team merges per-language fine-tunes of one base.
- Not when: models have different architectures, tokenizers or unrelated pretraining (use [[knowledge-distillation-and-compression]] instead); you need guaranteed per-task quality (keep separate models or LoRA adapters and route; see [[fine-tuning-and-peft]]); you have the data and compute for proper multi-task training, which is usually the stronger option.

## Setup & code
The merge rules themselves on a toy problem: a linear model whose two "fine-tunes" learned different features. NumPy only; real merges apply the same functions per tensor.

```python
import numpy as np

rng = np.random.default_rng(0)
d = 40
base = rng.normal(size=d)                      # "pretrained" weights


shared = np.arange(20, d)                      # features every task sees


def make_task(own):
    """Inputs use the task's own features plus shared ones; the task needs new
    weights on its own features and the base weights on the shared ones."""
    target = base.copy()
    target[own] += rng.normal(scale=2.0, size=len(own))
    X = np.zeros((60, d))
    cols = np.concatenate([own, shared])
    X[:, cols] = rng.normal(size=(60, len(cols)))
    return X, X @ target + rng.normal(scale=1.0, size=60)  # label noise


def finetune(X, y, steps=300, lr=0.01):
    w = base.copy()
    for _ in range(steps):
        w -= lr * X.T @ (X @ w - y) / len(y)
    return w


def mse(w, task):
    X, y = task
    return float(np.mean((X @ w - y) ** 2))


task_a = make_task(np.arange(0, 10))
task_b = make_task(np.arange(10, 20))
w_a, w_b = finetune(*task_a), finetune(*task_b)
tau_a, tau_b = w_a - base, w_b - base          # task vectors


def linear(ws, weights):
    return sum(c * w for c, w in zip(weights, ws))


def slerp(w0, w1, t, eps=1e-8):
    """Spherical interpolation of two weight vectors (per tensor in practice)."""
    u0, u1 = w0 / np.linalg.norm(w0), w1 / np.linalg.norm(w1)
    omega = np.arccos(np.clip(u0 @ u1, -1.0, 1.0))
    if omega < eps:                            # nearly parallel: fall back to lerp
        return (1 - t) * w0 + t * w1
    return (np.sin((1 - t) * omega) * w0 + np.sin(t * omega) * w1) / np.sin(omega)


def task_arithmetic(taus, lam):
    return base + lam * sum(taus)


def ties(taus, density, lam):
    """Trim small entries, elect a sign per parameter, average agreeing entries."""
    T = np.stack(taus)
    k = int(density * T.shape[1])
    trimmed = np.zeros_like(T)
    for i, t in enumerate(T):                  # 1. trim: keep top-k |values| per task
        keep = np.argsort(-np.abs(t))[:k]
        trimmed[i, keep] = t[keep]
    sign = np.sign(trimmed.sum(axis=0))        # 2. elect: sign of the summed mass
    agree = (np.sign(trimmed) == sign) & (trimmed != 0)
    merged = (trimmed * agree).sum(0) / np.maximum(agree.sum(0), 1)  # 3. disjoint mean
    return base + lam * merged


def dare(tau, drop, rng):
    """Drop a random fraction of the delta and rescale the rest by 1/(1-drop)."""
    mask = rng.random(tau.shape) >= drop
    return tau * mask / (1.0 - drop)


merges = {
    "fine-tuned A only": w_a,
    "fine-tuned B only": w_b,
    "linear average": linear([w_a, w_b], [0.5, 0.5]),
    "SLERP t=0.5": slerp(w_a, w_b, 0.5),
    "task arithmetic lam=1": task_arithmetic([tau_a, tau_b], 1.0),
    "TIES density=0.3 lam=1": ties([tau_a, tau_b], 0.3, 1.0),
    "DARE p=0.5 + sign election": ties([dare(t, 0.5, rng) for t in (tau_a, tau_b)], 1.0, 1.0),
}
print(f"{"method":28s} {'MSE task A':>11s} {'MSE task B':>11s}")
for name, w in merges.items():
    print(f"{name:28s} {mse(w, task_a):11.2f} {mse(w, task_b):11.2f}")
```
In this toy the two task vectors live on different features, so task arithmetic and TIES keep both skills while plain averaging and SLERP halve each one. DARE does badly here because each task vector is concentrated in 10 weights; dropping half of them destroys signal. DARE relies on LLM fine-tuning deltas being highly redundant, so check it on your own eval rather than assuming it helps.

Real merges use **mergekit** (LGPL-3.0, `pip install mergekit`), which reads a YAML config and writes a Hugging Face checkpoint. It supports `linear`, `slerp`, `task_arithmetic`, `ties`, `dare_ties`, `dare_linear`, `passthrough` and more; see the repo's merge-method guide for the parameters each one takes.

```yaml
# ties.yml  ->  mergekit-yaml ties.yml ./merged --cuda
merge_method: ties
base_model: org/base-7b
models:
  - model: org/base-7b-code-sft
    parameters: {density: 0.5, weight: 0.5}
  - model: org/base-7b-chat-sft
    parameters: {density: 0.5, weight: 0.5}
parameters: {normalize: true}
dtype: bfloat16
```
Model names above are placeholders. For LoRA adapters, PEFT's `add_weighted_adapter` offers linear, TIES and DARE combinations without materialising full models.

## Choosing / trade-offs
| Method | Models | Good for | Watch out |
|---|---|---|---|
| Linear / soup | 2+ | Fine-tunes of the same run or very similar tasks | Dilutes distinct skills |
| SLERP | exactly 2 | Smooth blend of two related models | Not defined for 3+ (mergekit has `multislerp`) |
| Task arithmetic | 2+ with base | Adding / negating skills | Needs `lambda` tuning; interference on shared weights |
| TIES | 2+ with base | Many task vectors with conflicts | `density` and weights need tuning |
| DARE (+TIES) | 2+ with base | Redundant SFT deltas | Random; can wreck concentrated deltas |

- Merge vs multi-task fine-tune: merging is near-free and needs no data; joint training is more reliable when data and compute exist.
- Merge vs routing adapters: merging gives one fast model; routing keeps each task's full quality but adds a router and serving complexity.

## Gotchas
- Same base is a hard requirement in practice; merging models from different pretraining runs usually produces garbage even with identical shapes.
- Tokenizer and chat template differences (added tokens, resized embeddings) must be reconciled; mergekit has options for this, but check the template you ship.
- Merged models can top leaderboards by overfitting benchmarks the parents were tuned on; evaluate on your own held-out tasks ([[llm-evaluation]]).
- Safety and refusal behaviour from an aligned parent can weaken after merging; re-run safety evals.
- Licences combine: a merge inherits the most restrictive terms of its parents ([[model-licenses]]).
- Merging quantized weights is lossy; merge in fp16/bf16, then quantize ([[quantization]]).

## Related
- [[fine-tuning-and-peft]] - produces the fine-tunes and LoRA adapters being merged.
- [[lora-from-scratch-numpy]] - a LoRA adapter is a low-rank task vector.
- [[knowledge-distillation-and-compression]] - the option when architectures differ.
- [[continual-learning]] - merging as a way to limit catastrophic forgetting.
- [[rlhf-and-preference-optimization]] - merged models still need alignment checks.
- [[model-licenses]] - licence terms of merged parents.
- [[llm-evaluation]] - the only way to know a merge worked.

## References
- Ilharco et al., Editing Models with Task Arithmetic: https://arxiv.org/abs/2212.04089
- Yadav et al., TIES-Merging: https://arxiv.org/abs/2306.01708
- Yu et al., Language Models are Super Mario (DARE): https://arxiv.org/abs/2311.03099
- Wortsman et al., Model soups: https://arxiv.org/abs/2203.05482
- Goddard et al., Arcee's MergeKit: https://arxiv.org/abs/2403.13257
- mergekit repository and merge-method guide: https://github.com/arcee-ai/mergekit
- Shoemake, Animating rotation with quaternion curves (SIGGRAPH 1985) - origin of SLERP.
