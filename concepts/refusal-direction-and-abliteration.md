---
title: Refusal direction and abliteration (difference-of-means, directional ablation, weight orthogonalization)
category: concepts
tags: [abliteration, refusal-direction, activation-steering, directional-ablation, weight-orthogonalization, representation-engineering, mechanistic-interpretability, alignment, red-teaming, numpy, from-scratch]
use_cases:
  - "understand how abliterated open-weight models are produced and what they lose"
  - "find a linear direction for a behaviour (refusal, style, language) in a model's residual stream"
  - "steer a model's behaviour at inference by adding or removing an activation direction"
  - "assess how robust an open-weight model's safety tuning is before release"
status: draft
last_verified: 2026-10-19
sources:
  - https://arxiv.org/abs/2406.11717
  - https://huggingface.co/blog/mlabonne/abliteration
  - https://arxiv.org/abs/2308.10248
  - https://arxiv.org/abs/2310.01405
  - https://github.com/mlabonne/llm-course
---

# Refusal direction and abliteration (difference-of-means, directional ablation, weight orthogonalization)

## Summary
Arditi et al. (2024) showed that in many chat models, refusing a request is mediated by one
direction `r` in the residual stream: the difference between mean activations on prompts the
model refuses and prompts it answers. Projecting `r` out of every activation (directional
ablation) stops refusals; adding it (activation addition) makes the model refuse harmless
prompts. **Abliteration** makes the ablation permanent by orthogonalizing every weight matrix that
writes to the residual stream against `r`, so no training is needed. The same recipe finds and
edits other behaviour directions. This file builds the whole procedure on a toy NumPy residual
stream and checks that weight orthogonalization equals ablation at inference.

## Key concepts
- **Residual stream.** Each layer reads the residual vector and adds its output back. A
  direction that no layer can write to cannot influence anything downstream.
- **Difference of means.** `r = mean(acts | refused prompts) − mean(acts | answered prompts)`,
  taken at one layer and one token position (usually the last prompt token), then normalized.
  Candidates from several layers are scored on held-out prompts and the best one is kept.
- **Directional ablation.** `x' = x − (x·r̂) r̂` on every layer output, at inference (hooks).
- **Activation addition.** `x' = x + α r̂` at one layer induces the behaviour (steering vectors).
- **Weight orthogonalization.** For every matrix `W` that writes to the residual stream
  (embedding, attention output, MLP output), `W' = W − r̂ r̂ᵀ W`. Its outputs then have no
  component along `r̂`, which is the same as ablating at inference, baked into the weights.
- **Healing.** Abliteration usually costs some general capability; the abliterated model is often
  followed by a short DPO or SFT run to recover quality (Labonne's write-up does this).

## When to use / scenarios
- Interpretability research: test whether a behaviour is linearly represented, and where.
- Red-teaming and release reviews of open-weight models: refusal training that a single direction
  removes is not a safety control against anyone holding the weights; plan safeguards outside the
  weights ([[guardrails-and-safety]], [[red-teaming-llm-apps]]).
- Steering style or persona (formality, language, verbosity) with a direction instead of a fine-tune.
- Not a way to fix over-refusal in a product: abliteration removes refusals indiscriminately,
  including ones you need. Fix over-refusal with prompts, data and preference tuning on
  borderline examples ([[rlhf-and-preference-optimization]]).
- Not for closed API models: it needs the weights and activations.
- Removing safeguards can break a model's licence or acceptable-use policy and the law where you
  deploy; you are responsible for what an abliterated model produces ([[model-licenses]]).

## Setup & code
NumPy only, runs in under a second. A two-layer toy residual stream in which "harmful" inputs
trigger a layer that writes a refusal direction, read out by a refusal head.

```python
import numpy as np

rng = np.random.default_rng(0)
d, n = 64, 400

def unit(v): return v / np.linalg.norm(v)

harm_feat = unit(rng.normal(size=d))          # input feature present in harmful prompts
r_true = unit(rng.normal(size=d))             # direction the model uses for "refuse"

# Toy model: embedding E, two residual layers writing W_k @ x, refusal head reads x . r_true
E = np.eye(d) + 0.02 * rng.normal(size=(d, d))
W1 = 3.0 * np.outer(r_true, harm_feat) + 0.02 * rng.normal(size=(d, d))   # detects harm, writes r
W2 = 0.02 * rng.normal(size=(d, d))

def forward(x, E, W1, W2, ablate=None, add=None):
    h = E @ x
    if ablate is not None: h = h - (h @ ablate) * ablate
    acts = []
    for W in (W1, W2):
        h = h + W @ h
        if ablate is not None: h = h - (h @ ablate) * ablate
        if add is not None and not acts: h = h + add
        acts.append(h)
    return h, acts

def batch_forward(X, *a, **k):
    out = [forward(x, *a, **k) for x in X]
    return np.array([o[0] for o in out]), [np.array([o[1][i] for o in out]) for i in range(2)]

def refuses(H): return (H @ r_true) > 1.5

harmful = 0.15 * rng.normal(size=(n, d)) + harm_feat
harmless = 0.15 * rng.normal(size=(n, d))
tr, te = slice(0, n // 2), slice(n // 2, n)

# 1) Find the direction: difference of means at layer 1, on training prompts only
_, acts_h = batch_forward(harmful[tr], E, W1, W2)
_, acts_b = batch_forward(harmless[tr], E, W1, W2)
r_hat = unit(acts_h[0].mean(0) - acts_b[0].mean(0))
print("cos(r_hat, r_true) =", round(float(r_hat @ r_true), 3))

H_h, _ = batch_forward(harmful[te], E, W1, W2)
H_b, _ = batch_forward(harmless[te], E, W1, W2)
print("refusal rate  harmful %.2f  harmless %.2f" % (refuses(H_h).mean(), refuses(H_b).mean()))

# 2) Directional ablation at inference
H_h_abl, _ = batch_forward(harmful[te], E, W1, W2, ablate=r_hat)
print("ablated: harmful refusal rate %.2f" % refuses(H_h_abl).mean())

# 3) Weight orthogonalization: same effect, no hooks
P = np.eye(d) - np.outer(r_hat, r_hat)
E_o, W1_o, W2_o = P @ E, P @ W1, P @ W2
H_h_orth, _ = batch_forward(harmful[te], E_o, W1_o, W2_o)
H_b_orth, _ = batch_forward(harmless[te], E_o, W1_o, W2_o)
print("orthogonalized: harmful refusal rate %.2f" % refuses(H_h_orth).mean())
print("max |ablated - orthogonalized| =", float(np.abs(H_h_abl - H_h_orth).max()))

# Side effect on harmless prompts: output change outside the removed direction
drift = np.linalg.norm((H_b_orth - H_b) @ P, axis=1) / np.linalg.norm(H_b, axis=1)
print("harmless relative drift (off-direction) %.3f" % drift.mean())

# 4) Activation addition: push harmless prompts along r_hat at layer 1
alpha = float((acts_h[0] - acts_b[0]).mean(0) @ r_hat)
H_b_add, _ = batch_forward(harmless[te], E, W1, W2, add=alpha * r_hat)
print("steered harmless refusal rate %.2f" % refuses(H_b_add).mean())

assert r_hat @ r_true > 0.9
assert refuses(H_h).mean() > 0.95 and refuses(H_b).mean() < 0.05
assert refuses(H_h_orth).mean() < 0.05
assert np.allclose(H_h_abl, H_h_orth, atol=1e-8)
assert refuses(H_b_add).mean() > 0.9
```

On a real model the steps are the same: run a few hundred refused and answered instructions,
cache residual activations at the last prompt token for every layer, build one candidate `r̂` per
layer, pick the one that best removes refusals on held-out prompts while keeping a benchmark
score, then orthogonalize the embedding and every attention-output and MLP-output matrix
(TransformerLens hooks or plain PyTorch on the `o_proj` / `down_proj` weights).

## Choosing / trade-offs
- **Inference-time ablation vs orthogonalized weights.** Hooks are reversible and per-request but
  need a custom runtime; orthogonalized weights run anywhere (vLLM, GGUF) and cannot be undone
  without the original weights.
- **Layer choice.** Middle layers usually give the cleanest direction; early ones capture surface
  features of the prompt, late ones are close to the output tokens. Select on held-out prompts,
  not on the prompts used to compute `r̂`.
- **Ablate vs subtract a scaled vector.** Projection removes the component for every input;
  subtracting `α r̂` shifts all inputs by the same amount and can push answered prompts off-distribution.
- **Abliteration vs fine-tuning.** Abliteration is minutes of compute and no data labels, but blunt
  and lossy; fine-tuning or DPO changes behaviour selectively and keeps quality better.
- **Steering vs prompting.** A steering vector is cheaper per token than a long system prompt and
  harder to override from user text, but needs the weights and per-model tuning of `α`.

## Gotchas
- The direction fits the prompt set: a narrow or templated set of "refused" prompts yields a
  direction for that template, not for refusal. Use varied prompts and check transfer on held-out ones.
- Abliterated models lose some capability (benchmark drops are commonly reported); evaluate
  before and after, and expect to heal with DPO or SFT.
- Removing refusal does not add knowledge or skill, and the model may still decline in other words;
  some behaviours are spread over several directions.
- Forgetting a writer (tied embeddings, biases of output projections, extra norms in some
  architectures) leaves a path for `r̂` back into the stream; verify activations along `r̂` are ~0.
- Downloaded "abliterated" or "uncensored" checkpoints are third-party weights: check licence,
  provenance and safetensors format, and treat them as untrusted ([[ai-security-privacy-compliance]]).
- Results on one model family do not transfer; recompute the direction per model and per version.

## Related
- [[mechanistic-interpretability]] - activation patching, probes and features: the toolkit this builds on.
- [[sparse-autoencoders-for-interpretability]] - finding many directions at once instead of one by contrast.
- [[rlhf-and-preference-optimization]] - how refusal behaviour is trained in, and DPO for healing.
- [[guardrails-and-safety]] - controls outside the weights, which abliteration cannot remove.
- [[red-teaming-llm-apps]] - testing a model's safety behaviour before release.
- [[model-merging]] - the other training-free way to edit open weights.
- [[model-licenses]] - acceptable-use terms that removing safeguards can breach.

## References
- Arditi et al., "Refusal in Language Models Is Mediated by a Single Direction" (2024): https://arxiv.org/abs/2406.11717
- M. Labonne, "Uncensor any LLM with abliteration" (Hugging Face blog): https://huggingface.co/blog/mlabonne/abliteration
- Turner et al., "Steering Language Models With Activation Engineering" (2023): https://arxiv.org/abs/2308.10248
- Zou et al., "Representation Engineering: A Top-Down Approach to AI Transparency" (2023): https://arxiv.org/abs/2310.01405
- M. Labonne, LLM Course, "New trends: interpretability": https://github.com/mlabonne/llm-course
