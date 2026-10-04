---
title: Mechanistic interpretability (probes, patching, sparse autoencoders)
category: concepts
tags: [mechanistic-interpretability, interpretability, linear-probe, activation-patching, sparse-autoencoder, superposition, steering, forward-hooks, transformerlens, nnsight, pytorch]
use_cases:
  - "find out whether a neural network internally represents a concept such as sentiment, truthfulness or a protected attribute"
  - "locate which layer or component of a model is responsible for a behaviour"
  - "extract human-readable features from a model's activations with a sparse autoencoder"
  - "steer a model's output by adding a direction to its activations"
  - "read and modify intermediate activations of a PyTorch or transformer model"
status: draft
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/1610.01644
  - https://arxiv.org/abs/2202.05262
  - https://transformer-circuits.pub/2022/toy_model/index.html
  - https://transformer-circuits.pub/2023/monosemantic-features
  - https://github.com/TransformerLensOrg/TransformerLens
  - https://nnsight.net/
---

# Mechanistic interpretability (probes, patching, sparse autoencoders)

## Summary
Mechanistic interpretability studies the *internals* of a trained network (its activations, directions and circuits) to explain how it computes an output, rather than only attributing the output to input features. The core toolkit is small: *linear probes* test whether information is linearly readable from a layer; *activation patching* swaps activations between two runs to find which components cause a behaviour; *sparse autoencoders* (SAEs) decompose activations into many sparse, often human-interpretable features; *steering* adds a direction to activations to change behaviour. All of it rests on reading and writing activations with forward hooks. It complements, rather than replaces, input-attribution methods like SHAP ([[model-interpretability]]).

## Key concepts
- **Residual stream and activations.** In a transformer each layer reads from and adds to a shared residual vector per token ([[transformers-and-attention]]); most techniques read or edit that vector, or the outputs of attention heads and MLPs.
- **Linear representation hypothesis.** Many concepts (sentiment, truth, language, position) appear as *directions* in activation space. A logistic regression on activations that predicts the concept well is evidence the information is present and linearly decodable.
- **Probing caveat.** A probe shows information is *there*, not that the model *uses* it. Compare against a control (random labels or a randomly initialised model) and confirm causally with patching or ablation.
- **Activation patching (causal tracing).** Run a clean input and a corrupted input; copy one component's activation from the clean run into the corrupted run and measure how much of the clean output returns. Components that restore the output are causally important. *Ablation* (zeroing or mean-replacing a component) is the cruder cousin.
- **Superposition.** Networks pack more features than they have neurons by using nearly orthogonal directions, so single neurons are *polysemantic* (fire for unrelated things). This is why neuron-by-neuron inspection usually fails.
- **Sparse autoencoders / dictionary learning.** Train a wide autoencoder (dictionary often 4-64x the layer width) to reconstruct activations with an L1 penalty (or TopK) on its hidden code. Each hidden unit tends to fire for one interpretable concept. Quality is judged by reconstruction error, sparsity (L0), and the model's loss when activations are replaced by the SAE reconstruction.
- **Steering vectors.** The difference of mean activations between two prompt sets (e.g. positive vs negative) gives a direction; adding it at inference shifts behaviour. SAE features can be used as steering directions too.
- **Circuits.** A circuit is a small subgraph of heads and MLPs that implements a behaviour (e.g. induction heads that copy earlier tokens). Found by patching and path analysis.

## When to use / scenarios
- Auditing: does a credit or hiring model encode a protected attribute even after it was removed from inputs? Probe for it ([[guardrails-and-safety]]).
- Debugging: a model fails on a pattern; patching localises the failure to a layer, which guides targeted data or fine-tuning fixes ([[debugging-neural-network-training]]).
- LLM safety research: detecting deception or refusal directions, monitoring activations for jailbreak attempts, steering away from unwanted behaviour ([[prompt-injection]]).
- Science and medicine models: checking whether a model learned a known physical or biological quantity, not a shortcut.
- NOT for explaining a tabular model to a business user: SHAP, permutation importance and partial dependence answer "which inputs mattered" more directly ([[model-interpretability]]).
- NOT a production control by itself: steering and activation monitors are research-grade; pair them with evaluation and ordinary guardrails ([[llm-evaluation]]).

## Setup & code
```bash
pip install torch scikit-learn numpy
```
A plain-PyTorch walkthrough of the three core moves on a small MLP trained on the digits dataset: capture activations with a forward hook, fit a linear probe for a concept the model was never trained on (is the digit even?), patch an activation from one input into another, and train a small sparse autoencoder on the hidden layer:
```python
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

torch.manual_seed(0)
X, y = load_digits(return_X_y=True)
X = torch.tensor(X / 16.0, dtype=torch.float32); y = torch.tensor(y)

model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
opt = torch.optim.Adam(model.parameters(), 1e-3)
for _ in range(200):
    opt.zero_grad(); F.cross_entropy(model(X), y).backward(); opt.step()

# 1) Read activations with a forward hook.
acts = {}
hook = model[3].register_forward_hook(lambda m, i, o: acts.__setitem__("h2", o.detach()))
with torch.no_grad():
    model(X)
hook.remove()
H = acts["h2"].numpy()

# 2) Linear probe: is "even digit" linearly decodable? Compare with shuffled labels.
even = (y % 2 == 0).numpy()
probe = cross_val_score(LogisticRegression(max_iter=2000), H, even, cv=5).mean()
control = cross_val_score(LogisticRegression(max_iter=2000), H,
                          even[torch.randperm(len(even)).numpy()], cv=5).mean()
print(f"probe acc {probe:.2f}  vs shuffled-label control {control:.2f}")

# 3) Activation patching: copy layer-2 activation of a '3' into a run on a '7'.
src, dst = X[y == 3][:1], X[y == 7][:1]
with torch.no_grad():
    clean = model[:4](src)                       # activation after layer 2 for the '3'
    patched = model[4](clean)                    # finish the forward pass from there
    print("pred on '7' input:", model(dst).argmax().item(),
          "-> after patching layer 2 from a '3':", patched.argmax().item())

# 4) Sparse autoencoder on the hidden activations (dictionary 4x wider, L1 sparsity).
class SAE(nn.Module):
    def __init__(self, d, k):
        super().__init__(); self.enc = nn.Linear(d, k); self.dec = nn.Linear(k, d)
    def forward(self, h):
        z = F.relu(self.enc(h)); return self.dec(z), z

Ht = acts["h2"]
sae = SAE(64, 256); opt = torch.optim.Adam(sae.parameters(), 1e-3)
for _ in range(2000):
    rec, z = sae(Ht)
    loss = F.mse_loss(rec, Ht) + 1e-3 * z.abs().sum(1).mean()
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    rec, z = sae(Ht)
    fvu = ((rec - Ht) ** 2).sum() / ((Ht - Ht.mean(0)) ** 2).sum()
    l0 = (z > 0).float().sum(1).mean()
    spliced = (model[4](rec).argmax(1) == y).float().mean()
print(f"SAE: unexplained variance {fvu:.3f}, active features/input {l0:.1f} of 256, "
      f"model acc with SAE reconstruction {spliced:.2f}")
assert probe > control + 0.2
```
The probe reads "even vs odd" well above the shuffled control even though the model was only trained on digit identity, because the parity is a function of the identity it does encode, which is exactly the probing caveat: decodable is not the same as used. Patching a whole layer from a '3' turns the prediction into a 3, so everything downstream reads that layer. The SAE reconstructs the layer with only a handful of active features per input, and the model keeps its accuracy on the spliced reconstruction.

For transformers, use a library that names every hook point instead of hand-writing hooks: **TransformerLens** (`HookedTransformer`, `run_with_cache`, `run_with_hooks`) for supported open models, **nnsight** for wrapping arbitrary Hugging Face models (also remotely), and **SAELens** for training and loading published SAEs (e.g. Gemma Scope for Gemma 2).

## Choosing / trade-offs
- **Question "is X represented?"** Linear probe with a control task. Cheap; correlational only.
- **Question "which component causes Y?"** Activation patching or ablation over layers/heads. Needs carefully matched clean/corrupted prompts.
- **Question "what features does this layer use?"** Sparse autoencoder. Expensive to train well on large models (many tokens, large dictionaries); reuse published SAEs when they exist for your model.
- **Goal "change behaviour without fine-tuning"?** Steering vector from contrastive prompt sets. Quick to try; effect size and side effects vary, so evaluate on held-out prompts.
- **Hand-written hooks vs a library.** Plain `register_forward_hook` works for any PyTorch model; TransformerLens/nnsight save time on transformers and make patching sweeps short.

## Gotchas
- A high probe accuracy can come from the probe itself learning the task; use simple (linear) probes and compare against control tasks or a random-weights model.
- Patching results depend on the corruption: Gaussian noise, swapped names and mean ablation can rank components differently. State the corruption you used.
- SAE features are not guaranteed to be interpretable or complete; dead features (never fire) and "dense" features are common. Check reconstruction loss *inside the model*, not just MSE.
- Explanations of features from top-activating examples are suggestive, not proof; verify by intervening on the feature.
- Activation caches for long contexts and big models are huge (layers x tokens x width); hook only the layers you need and move tensors to CPU.
- Hooks persist until removed; a forgotten hook silently changes every later forward pass. Use the handle's `.remove()` or a context manager.
- Results on a small model or one checkpoint may not transfer to a larger or retrained one.

## Related
- [[model-interpretability]] - SHAP, permutation importance and input attribution.
- [[transformers-and-attention]] - residual stream, attention heads and MLP blocks.
- [[autoencoders-and-self-supervised-learning]] - the autoencoder idea behind SAEs.
- [[adversarial-examples-and-robustness]] - shortcut features found by probing.
- [[guardrails-and-safety]] - activation monitoring and steering as safety tools.
- [[hallucination-and-grounding]] - probes for truthfulness directions.

## References
- Alain & Bengio, Understanding intermediate layers using linear classifier probes: https://arxiv.org/abs/1610.01644
- Meng et al., Locating and Editing Factual Associations in GPT (causal tracing): https://arxiv.org/abs/2202.05262
- Elhage et al., Toy Models of Superposition: https://transformer-circuits.pub/2022/toy_model/index.html
- Bricken et al., Towards Monosemanticity (sparse autoencoders): https://transformer-circuits.pub/2023/monosemantic-features
- TransformerLens: https://github.com/TransformerLensOrg/TransformerLens
- nnsight: https://nnsight.net/
