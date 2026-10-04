---
title: Weight initialization (Xavier/Glorot, He/Kaiming, orthogonal)
category: concepts
tags: [deep-learning, initialization, xavier, glorot, kaiming, he-init, orthogonal, lecun, vanishing-gradients, exploding-gradients, residual, pytorch]
use_cases:
  - "pick an initialization scheme for a custom network built from scratch"
  - "debug activations or gradients that shrink to zero or blow up with depth"
  - "initialize residual branches so a very deep network trains from step one"
  - "check what PyTorch layers use as their default init"
status: draft
last_verified: 2026-10-04
sources:
  - https://pytorch.org/docs/stable/nn.init.html
  - https://proceedings.mlr.press/v9/glorot10a.html
  - https://arxiv.org/abs/1502.01852
  - https://arxiv.org/abs/1312.6120
  - https://arxiv.org/abs/1901.09321
---

# Weight initialization (Xavier/Glorot, He/Kaiming, orthogonal)

## Summary
Initialization sets the starting weights before the first gradient step. If the weights are too small, activations and gradients shrink layer by layer until nothing reaches the early layers. If they are too large, everything explodes into NaN. Good schemes scale the random weights by the layer's fan-in or fan-out so the signal variance stays roughly constant through depth. Framework defaults are usually fine for shallow nets and pretrained models, but deep custom nets, unusual activations and residual stacks need a deliberate choice.

## Key concepts
- **Variance preservation.** For `y = W x` with `n_in` inputs, `Var(y) ≈ n_in · Var(W) · Var(x)`. Choosing `Var(W) ≈ 1 / n_in` keeps the activation scale stable across layers. Doing the same for gradients would need `1 / n_out`. Xavier averages the two.
- **Symmetry breaking.** All-zero (or all-equal) weights make every unit in a layer compute the same thing and receive the same gradient forever. Weights must be random. Biases can start at zero.
- **Schemes.**
  - **Xavier / Glorot**: `Var(W) = 2 / (n_in + n_out)`. Designed for tanh/sigmoid (symmetric, roughly linear near 0).
  - **He / Kaiming**: `Var(W) = 2 / n_in`. The factor 2 makes up for ReLU zeroing half the inputs. Use for ReLU, Leaky ReLU (gain adjusted for the slope), GELU and SiLU.
  - **LeCun**: `Var(W) = 1 / n_in`. Used with SELU, and the basis of many JAX/Flax defaults.
  - **Orthogonal**: a random orthogonal matrix times a gain. Preserves norms exactly in linear nets; common for RNN recurrent weights and RL policy nets.
  - **Truncated normal, small std** (for example 0.02): the standard for transformers (BERT, GPT-2), with output projections often scaled down further by depth.
- **Gain.** `nn.init.calculate_gain(nonlinearity)` returns the multiplier per activation (relu: √2, tanh: 5/3, linear/sigmoid: 1, leaky_relu: √(2 / (1 + a²))).
- **Residual networks.** In `x + f(x)` each block adds variance, so deep stacks grow unless the branch starts small. Common fixes: zero-init the last BatchNorm γ in each ResNet block, zero-init or depth-scale the output projection (GPT-2 scales by 1/√(2·n_layers)), or Fixup/SkipInit without normalisation.
- **Normalisation reduces sensitivity.** BatchNorm/LayerNorm re-scale activations every layer, so init matters less in normalised nets, but not zero. Unnormalised deep nets depend on it heavily.

## When to use / scenarios
- Custom MLP/CNN with ReLU-family activations: Kaiming normal or uniform, `mode="fan_in"`, biases zero. PyTorch's `nn.Linear`/`nn.Conv*` default is a Kaiming-uniform variant with a = √5, which works but is smaller than He init.
- tanh/sigmoid networks: Xavier.
- Transformer from scratch: normal(0, 0.02) for embeddings and linear layers, scaled-down residual output projections, LayerNorm weight 1 and bias 0. Copy the reference model's `_init_weights`.
- LSTM/GRU: orthogonal recurrent weights, Xavier input weights, forget-gate bias set to 1.
- Fine-tuning a pretrained model: only the new head is initialised (small normal or zeros for the final classifier); never re-init the backbone.
- NOT worth hand-tuning for tree models, linear models fit by closed form, or any pretrained network you use as is.

## Setup & code
```bash
pip install torch
```
Track activation scale through a 30-layer ReLU MLP under three inits:
```python
import torch
from torch import nn

def deep_mlp(init, depth=30, width=512):
    layers = []
    for _ in range(depth):
        lin = nn.Linear(width, width)
        init(lin.weight)
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.ReLU()]
    return nn.Sequential(*layers)

inits = {
    "std=0.01": lambda w: nn.init.normal_(w, std=0.01),
    "xavier": nn.init.xavier_normal_,
    "kaiming": lambda w: nn.init.kaiming_normal_(w, nonlinearity="relu"),
}
torch.manual_seed(0)
x = torch.randn(256, 512)
for name, init in inits.items():
    torch.manual_seed(0)
    h, stds = x, []
    with torch.no_grad():
        for layer in deep_mlp(init):
            h = layer(h)
            if isinstance(layer, nn.ReLU):
                stds.append(h.std().item())
    print(f"{name:9s} activation std at layers 1/10/30: "
          f"{stds[0]:.3g} / {stds[9]:.3g} / {stds[29]:.3g}")
# Observed (torch 2.13 CPU): std=0.01 -> 0.13 / 9e-09 / 6e-25 (collapses),
# xavier -> 0.59 / 0.027 / 1e-05 (shrinks with ReLU), kaiming -> 0.83 / 0.85 / 0.48 (stays order 1).
```
Applying a scheme to a whole model:
```python
def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 2))
model.apply(init_weights)
```

## Choosing / trade-offs
- **Normal vs uniform.** Same variance gives near-identical results; follow the reference implementation.
- **fan_in vs fan_out.** `fan_in` preserves forward activations (the usual choice); `fan_out` preserves backward gradients (torchvision ResNets use it for conv layers). Either works with normalisation.
- **Rely on defaults vs explicit init.** For shallow or normalised nets, defaults are fine and explicit init is noise. For deep unnormalised nets, custom activations or reproducing a paper, write the init out.
- **Zero-init residual outputs** speeds early training of deep ResNets/transformers, but makes the network start as (nearly) the identity, so check that gradients still reach the branch weights.

## Gotchas
- Initialising everything with `torch.zeros` or a constant: the model trains, but all units in a layer stay identical. Loss plateaus fast.
- Calling init inside `forward` or after loading a checkpoint overwrites the trained weights. Init once, in `__init__` or via `model.apply` before training.
- `model.apply(init_fn)` touches every submodule, including pretrained ones and embeddings. Scope it to the new modules when fine-tuning.
- Kaiming's `nonlinearity` argument defaults to `leaky_relu` with a = 0, which equals ReLU. Pass `"relu"` explicitly for clarity, and the right slope for Leaky ReLU.
- The final classification layer with a large init gives overconfident initial logits and a large starting loss. A healthy start for C classes is loss ≈ ln(C); much higher means the head init or the loss pairing is off ([[debugging-neural-network-training]]).
- Embedding layers default to N(0, 1), which is large for transformers; most LLM code re-inits them to std 0.02.

## Related
- [[activation-functions]] - each scheme's gain is derived for a specific activation.
- [[normalization-layers]] - reduce, but do not remove, sensitivity to init.
- [[neural-network-fundamentals]] - vanishing and exploding gradients.
- [[debugging-neural-network-training]] - checking initial loss and per-layer activation stats.
- [[deep-learning-training]] - LR warmup also protects against a poor start.

## References
- PyTorch `torch.nn.init`: https://pytorch.org/docs/stable/nn.init.html
- Glorot & Bengio, Understanding the difficulty of training deep feedforward neural networks: https://proceedings.mlr.press/v9/glorot10a.html
- He et al., Delving Deep into Rectifiers (Kaiming init, PReLU): https://arxiv.org/abs/1502.01852
- Saxe et al., Exact solutions to the nonlinear dynamics of learning in deep linear networks (orthogonal init): https://arxiv.org/abs/1312.6120
- Zhang et al., Fixup Initialization: https://arxiv.org/abs/1901.09321
