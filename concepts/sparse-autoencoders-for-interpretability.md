---
title: Sparse autoencoders for interpretability (dictionary learning on activations, features, steering)
category: concepts
tags: [sparse-autoencoder, sae, dictionary-learning, mechanistic-interpretability, superposition, topk-sae, jumprelu, steering, saelens, gemma-scope, neuronpedia, numpy]
use_cases:
  - "decompose a model's activations into human-interpretable features"
  - "find a feature for a concept (a language, a topic, refusal, a code bug) inside an LLM and steer with it"
  - "load a published sparse autoencoder for an open model and inspect which features fire on a prompt"
  - "train a small sparse autoencoder and judge whether it is any good"
status: draft
last_verified: 2026-10-15
sources:
  - https://transformer-circuits.pub/2023/monosemantic-features
  - https://transformer-circuits.pub/2024/scaling-monosemanticity/
  - https://transformer-circuits.pub/2022/toy_model/index.html
  - https://arxiv.org/abs/2406.04093
  - https://arxiv.org/abs/2407.14435
  - https://arxiv.org/abs/2408.05147
  - https://github.com/decoderesearch/SAELens
---

# Sparse autoencoders for interpretability (dictionary learning on activations, features, steering)

## Summary
A sparse autoencoder (SAE) is a wide, one-hidden-layer autoencoder trained on a model's internal activations so that each activation is rebuilt from a *few* active "latents" out of thousands. Because networks store more concepts than they have neurons (superposition), single neurons are polysemantic; the SAE's decoder rows ("features") tend to line up with single concepts instead. That gives a vocabulary for reading a layer (which features fire on this prompt?) and for writing to it (add a feature's direction to steer behaviour). This file builds a TopK SAE in NumPy on synthetic superposed features, measures what "good" means, and shows how to load published SAEs with SAELens. The broader toolkit (probes, patching) is in [[mechanistic-interpretability]].

## Key concepts
- **Superposition.** With `n` sparse concepts and only `d < n` dimensions, a model can store each concept as a nearly orthogonal direction and tolerate the small interference, because few concepts are active at once. Each neuron then mixes many concepts.
- **Dictionary learning.** Model each activation `x ≈ b_dec + Σ_i z_i · d_i` with a dictionary of unit vectors `d_i` (decoder rows) and a sparse, non-negative code `z`. The encoder `z = σ(W_enc(x − b_dec) + b_enc)` predicts the code in one step.
- **Dictionary width.** Number of latents `m`, usually a multiple of the layer width (published SAEs range from a few thousand to millions of latents). Wider dictionaries split coarse features into finer ones ("feature splitting").
- **Sparsity mechanism.**
  - *L1 penalty* on `z` with a ReLU (original Anthropic recipe): simple, but shrinks activations toward zero and needs a tuned coefficient.
  - *TopK*: keep only the `k` largest pre-activations (Gao et al.); sets L0 directly, no shrinkage, fewer dead latents.
  - *JumpReLU*: per-latent learned threshold (used by Gemma Scope); similar goals to TopK.
- **Quality metrics.** L0 (mean active latents per token), fraction of variance unexplained (FVU), and, most important, *downstream loss*: the model's loss when the layer is replaced by the SAE reconstruction. Also count *dead latents* (never fire) and check interpretability by reading top-activating examples.
- **Features vs latents.** A latent is a unit of the SAE; calling it a "feature" is a hypothesis that it tracks one concept. Verify by intervening, not just by reading examples.
- **Steering.** Adding `α · d_i` to the residual stream (or clamping `z_i` high) pushes the model toward that concept. Anthropic's "Golden Gate" demo in Scaling Monosemanticity is the well-known example.
- **Variants.** Transcoders (predict an MLP's output from its input) and crosscoders (one dictionary across layers or models) use the same sparse-dictionary idea for circuit tracing and model diffing.

## When to use / scenarios
- LLM safety and auditing research: find features for deception, refusal, sycophancy, PII, or a dangerous topic, then monitor or ablate them ([[guardrails-and-safety]]).
- Model debugging: see what a layer "thinks" on failing prompts, e.g. a wrong-language feature firing in a translation model.
- Model diffing: compare features before and after fine-tuning to see what changed ([[fine-tuning-and-peft]]).
- Explaining an open model where published SAEs exist (Gemma Scope for Gemma models, community SAEs for GPT-2 and Llama-family models), browsable on Neuronpedia.
- NOT when you just need "is concept X decodable?": a linear probe is far cheaper ([[mechanistic-interpretability]]).
- NOT for closed API models: you need the weights and activations.
- NOT as a reliable production behaviour control: prompting, fine-tuning ([[rlhf-and-preference-optimization]]) and guardrails are more predictable than steering.

## Setup & code
NumPy only. 48 ground-truth sparse features are packed into 16 dimensions, which makes every "neuron" polysemantic. A TopK SAE with 96 latents and manual gradients plus Adam recovers the feature directions, and adding a learned decoder direction to an activation turns its latent on (steering).
```python
import numpy as np

rng = np.random.default_rng(0)
n_true, d, n = 48, 16, 20000           # 48 ground-truth features squeezed into 16 dims
W_true = rng.normal(size=(n_true, d))
W_true /= np.linalg.norm(W_true, axis=1, keepdims=True)
active = rng.random((n, n_true)) < 0.04            # each feature fires ~4% of the time
F = active * rng.uniform(0.5, 1.5, (n, n_true))    # sparse feature magnitudes
X = F @ W_true                                     # "activations": features in superposition

# Neurons are polysemantic: each dimension of X mixes many features.
corr = np.abs(np.corrcoef(np.c_[X, F].T)[:d, d:])
print(f"features a single neuron correlates with (|r|>0.1): {np.mean((corr > 0.1).sum(1)):.1f}")

m, k, lr = 96, 3, 1e-3                             # dictionary 6x wider than d, TopK sparsity
W_enc = rng.normal(0, 0.1, (d, m)); W_dec = W_enc.T.copy()
W_dec /= np.linalg.norm(W_dec, axis=1, keepdims=True)
b_enc = np.zeros(m); b_dec = X.mean(0)
params = [W_enc, b_enc, W_dec, b_dec]
mom = [np.zeros_like(p) for p in params]; vel = [np.zeros_like(p) for p in params]

def encode(x):
    pre = (x - b_dec) @ W_enc + b_enc
    thresh = np.partition(pre, -k, axis=1)[:, -k][:, None]
    z = np.where(pre >= thresh, np.maximum(pre, 0), 0.0)   # keep top-k, ReLU
    return z

for step in range(1, 3001):
    xb = X[rng.integers(0, n, 512)]
    z = encode(xb)
    xh = z @ W_dec + b_dec
    g_xh = 2 * (xh - xb) / len(xb)                 # d(MSE)/d(x_hat), summed over dims
    g_z = (g_xh @ W_dec.T) * (z > 0)
    grads = [(xb - b_dec).T @ g_z, g_z.sum(0), z.T @ g_xh, g_xh.sum(0) - (g_z @ W_enc.T).sum(0)]
    for i, (p, g) in enumerate(zip(params, grads)):  # Adam
        mom[i] = 0.9 * mom[i] + 0.1 * g; vel[i] = 0.999 * vel[i] + 0.001 * g * g
        p -= lr * (mom[i] / (1 - 0.9**step)) / (np.sqrt(vel[i] / (1 - 0.999**step)) + 1e-8)
    W_dec /= np.linalg.norm(W_dec, axis=1, keepdims=True)  # unit-norm decoder rows

z = encode(X); X_hat = z @ W_dec + b_dec
fvu = ((X - X_hat) ** 2).sum() / ((X - X.mean(0)) ** 2).sum()
dead = (z.max(0) == 0).sum()
cos = W_true @ W_dec.T                             # 48 x 96 cosine similarities
recovered = (cos.max(1) > 0.9).sum()
print(f"unexplained variance {fvu:.3f}, L0 {np.mean((z > 0).sum(1)):.1f}, dead latents {dead}/{m}")
print(f"true features recovered by some latent (cos > 0.9): {recovered}/{n_true}")

# Each latent should be monosemantic: find the latent matching feature 0 and check what fires it.
j = cos[0].argmax()
fires = z[:, j] > 0
print(f"latent {j}: precision for feature 0 = {active[fires, 0].mean():.2f}, recall = {fires[active[:, 0]].mean():.2f}")

# Steering: add the latent's decoder direction to an activation where feature 0 is off.
x0 = X[~active[:, 0]][:1]
print("latent before/after steering:", round(float(encode(x0)[0, j]), 2),
      round(float(encode(x0 + 2.0 * W_dec[j])[0, j]), 2))
assert recovered >= 30 and fvu < 0.3
```
Typical output: each raw neuron correlates with about 25 of the 48 features; the SAE explains about 88% of the variance with exactly 3 active latents, a few latents die, every true feature has a latent with cosine > 0.9, and the matching latent fires mostly when its feature is present. Precision is below 1 because `k=3` is larger than the ~2 features active per sample, so the extra slot is filled by near-miss latents; lowering `k` or widening the dictionary trades reconstruction against cleanliness. Real SAEs show the same trade-off at much larger scale.

Loading a published SAE with SAELens (`pip install sae-lens transformer-lens`) and steering GPT-2 small. `SAE.from_pretrained` returns the SAE object in SAELens 6.x (it returned a tuple in 5.x and earlier). Release names and SAE ids come from SAELens's pretrained SAE list:
<!-- skip-check: needs torch and downloads GPT-2 small and SAE weights from Hugging Face -->
```python
from sae_lens import SAE
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2")
sae = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id="blocks.8.hook_resid_pre")
hook = sae.cfg.metadata.hook_name                    # "blocks.8.hook_resid_pre"

_, cache = model.run_with_cache("The Golden Gate Bridge is in San")
z = sae.encode(cache[hook])                          # [batch, tokens, d_sae]
vals, ids = z[0, -1].topk(5)
print("top latents on last token:", ids.tolist())    # look them up on neuronpedia.org

latent = int(ids[0])
def steer(act, hook):
    return act + 8.0 * sae.W_dec[latent]             # add the feature's direction everywhere

with model.hooks(fwd_hooks=[(hook, steer)]):
    print(model.generate("I think that", max_new_tokens=20, verbose=False))
```
To train your own SAE on an open model, SAELens has a training runner (cache activations, choose architecture `topk`/`jumprelu`/`standard`, width, and L0 or L1 target); budget many millions of tokens of activations for a usable result.

## Choosing / trade-offs
- **Reuse vs train.** If a published SAE covers your model and layer, use it; training one well on a large model costs serious GPU time and activation storage.
- **Architecture.** TopK or JumpReLU give better reconstruction at a given L0 than L1 + ReLU in the papers above and are the common modern choices; L1 is simplest to implement and explain.
- **Width.** Wider = finer, more specific features and better reconstruction, but more dead latents, more absorbed/split features, and more to label. Narrow = coarse, general features.
- **L0.** Low L0 = more interpretable latents but worse reconstruction (downstream loss rises); high L0 = faithful but latents become denser and harder to read.
- **Layer.** Middle layers of an LLM tend to hold the more abstract concepts; early layers are token-level, late layers are output-prediction oriented.
- **SAE steering vs a diff-of-means steering vector.** SAE features are named and reusable; a contrastive steering vector ([[mechanistic-interpretability]]) needs no SAE and is often just as effective for one behaviour.

## Gotchas
- Low MSE is not enough: always measure the model's loss with the reconstruction spliced in. A small FVU can still destroy downstream behaviour.
- The reconstruction error ("dark matter") is not noise; it can carry information the SAE missed, so conclusions about "all features" of a layer are incomplete.
- Feature absorption and splitting: a general feature ("starts with S") may be silently absorbed into specific ones ("snake"), so it fails to fire where you expect.
- Labels from top-activating examples (often auto-written by an LLM) are hypotheses; check with interventions and with examples at low and mid activation, not just the top.
- Normalise the decoder rows (or fold norms) consistently; otherwise activation magnitudes are not comparable across latents.
- Steering strength is model- and layer-specific; too large a coefficient produces repetitive or broken text. Evaluate side effects on held-out prompts.
- SAEs are trained for one model checkpoint and one hook point; they do not transfer to a fine-tuned model without checking reconstruction quality first.
- Activation datasets are large (tokens x width x 4 bytes); store in fp16/bf16 and stream.

## Related
- [[mechanistic-interpretability]] - probes, activation patching and where SAEs fit in the toolkit.
- [[autoencoder-from-scratch-numpy]] - the plain autoencoder an SAE extends with sparsity.
- [[autoencoders-and-self-supervised-learning]] - autoencoder families and objectives.
- [[transformers-and-attention]] - the residual stream that SAEs are usually trained on.
- [[model-interpretability]] - input-attribution methods (SHAP, permutation importance) for comparison.
- [[guardrails-and-safety]] - where feature monitoring and steering could plug in.
- [[nmf-from-scratch]] - another non-negative sparse decomposition, for classical data.

## References
- Elhage et al. (2022), Toy Models of Superposition: https://transformer-circuits.pub/2022/toy_model/index.html
- Bricken et al. (2023), Towards Monosemanticity: Decomposing Language Models With Dictionary Learning: https://transformer-circuits.pub/2023/monosemantic-features
- Templeton et al. (2024), Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet: https://transformer-circuits.pub/2024/scaling-monosemanticity/
- Gao et al. (2024), Scaling and evaluating sparse autoencoders (TopK SAEs): https://arxiv.org/abs/2406.04093
- Rajamanoharan et al. (2024), Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders: https://arxiv.org/abs/2407.14435
- Lieberum et al. (2024), Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2: https://arxiv.org/abs/2408.05147
- SAELens: https://github.com/decoderesearch/SAELens
- Neuronpedia (browse published SAE features): https://www.neuronpedia.org/
