---
title: Vanishing and exploding gradients (and gradient clipping)
category: concepts
tags: [vanishing-gradients, exploding-gradients, gradient-clipping, weight-initialization, residual-connections, normalization, lstm, pytorch, deep-learning-basics]
use_cases:
  - "find out why the early layers of a deep network are not learning"
  - "fix a loss that suddenly jumps to NaN or inf during training"
  - "add gradient clipping to an RNN, LSTM or transformer training loop"
  - "choose initialization and activations so a deep MLP trains at all"
status: draft
last_verified: 2026-10-04
sources:
  - https://proceedings.mlr.press/v9/glorot10a.html
  - https://arxiv.org/abs/1502.01852
  - https://arxiv.org/abs/1211.5063
  - https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
---

# Vanishing and exploding gradients (and gradient clipping)

## Summary
Backpropagation multiplies one Jacobian per layer (or per time step in an RNN). If those factors are mostly below 1, the gradient shrinks exponentially with depth and the early layers stop learning (vanishing). If they are mostly above 1, it grows exponentially and the loss blows up to inf/NaN (exploding). Modern deep learning is largely a set of fixes for this: ReLU-family activations, variance-preserving initialization, normalization layers, residual connections, gated RNNs and gradient clipping.

## Key concepts
- **Chain rule as a product.** `∂L/∂h₁ = ∂L/∂h_L · Π ∂h_{k+1}/∂h_k`. With 50 factors of 0.9 the gradient is ~0.005×; with 50 factors of 1.1 it is ~117×.
- **Saturating activations.** Sigmoid's derivative is at most 0.25, and both sigmoid and tanh have near-zero derivative once inputs are large. Stacks of them vanish. ReLU has derivative 1 for positive inputs.
- **Initialization scale.** Weights that are too small shrink the activations and gradients layer by layer; too large and they explode. Xavier/Glorot (`Var = 2/(fan_in+fan_out)`) suits tanh; Kaiming/He (`Var = 2/fan_in`) suits ReLU ([[weight-initialization]]).
- **Residual connections.** `h + f(h)` gives the gradient an identity path, so it passes many layers almost unchanged ([[residual-and-skip-connections]]).
- **Normalization.** BatchNorm/LayerNorm keep activations at a stable scale through depth ([[normalization-layers]]).
- **RNNs over time.** The same weight matrix is multiplied at every step, so long sequences vanish or explode along time. LSTM/GRU gates add an additive memory path ([[recurrent-neural-networks-lstm-gru]]).
- **Gradient clipping.** Rescale the gradient when its global norm exceeds a threshold: `g ← g · min(1, max_norm/‖g‖)`. It caps exploding updates. It does nothing for vanishing gradients.

## When to use / scenarios
- Deep MLP or CNN where the loss barely moves and first-layer gradient norms are orders of magnitude smaller than last-layer ones: vanishing.
- Loss spikes, then inf/NaN, often a few hundred steps in, or right after the learning rate warms up: exploding.
- RNN/LSTM, seq2seq, RL policy gradients and transformer pre-training: add clipping by default (max norm 0.5 to 1.0 is a common start).
- Do NOT reach for clipping to fix a loss that diverges from step one. That is usually a too-high learning rate, unscaled inputs or a bug ([[debugging-neural-network-training]]).

## Setup & code
`pip install torch` (CPU is fine). The first part trains nothing: it measures first- vs last-layer gradient norms of a 20-layer MLP under four activation/initialization choices.

```python
import torch
import torch.nn as nn

torch.manual_seed(0)


def grad_norms(act, init, depth=20, width=256):
    layers = []
    for _ in range(depth):
        lin = nn.Linear(width, width)
        init(lin.weight)
        nn.init.zeros_(lin.bias)
        layers += [lin, act()]
    net = nn.Sequential(*layers)
    x = torch.randn(512, width)
    net(x).pow(2).mean().backward()
    lins = [m for m in net if isinstance(m, nn.Linear)]
    return lins[0].weight.grad.norm().item(), lins[-1].weight.grad.norm().item()


normal_small = lambda w: nn.init.normal_(w, std=0.01)
normal_big = lambda w: nn.init.normal_(w, std=0.2)
kaiming = lambda w: nn.init.kaiming_normal_(w, nonlinearity="relu")
xavier = lambda w: nn.init.xavier_normal_(w)

for name, act, init in [
    ("sigmoid + N(0,0.01)", nn.Sigmoid, normal_small),
    ("tanh + xavier", nn.Tanh, xavier),
    ("relu + N(0,0.2)", nn.ReLU, normal_big),
    ("relu + kaiming", nn.ReLU, kaiming),
]:
    first, last = grad_norms(act, init)
    print(f"{name:22s} first-layer grad {first:.2e}  last-layer grad {last:.2e}")

# Gradient clipping in a training step
model = nn.LSTM(32, 64, batch_first=True)
head = nn.Linear(64, 1)
params = list(model.parameters()) + list(head.parameters())
opt = torch.optim.AdamW(params, lr=1e-3)
x, y = torch.randn(16, 200, 32), torch.randn(16, 1)
out, _ = model(x)
loss = nn.functional.mse_loss(head(out[:, -1]), y)
opt.zero_grad()
loss.backward()
total = torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)  # returns pre-clip norm
opt.step()
print(f"pre-clip grad norm {total:.3f}")
```

Output (torch 2.13 CPU):
```
sigmoid + N(0,0.01)    first-layer grad 0.00e+00  last-layer grad 1.25e-01
tanh + xavier          first-layer grad 9.91e-03  last-layer grad 1.45e-02
relu + N(0,0.2)        first-layer grad 2.52e+13  last-layer grad 1.01e+14
relu + kaiming         first-layer grad 3.48e-01  last-layer grad 1.35e+00
pre-clip grad norm 0.498
```
Sigmoid with small weights: the first layer gets exactly zero (underflow). ReLU with `std=0.2` (Kaiming would be ~0.088): gradients near 1e13. Matching init to activation keeps all layers in the same range.

## Choosing / trade-offs
- **Clip by norm vs by value.** `clip_grad_norm_` rescales the whole gradient and keeps its direction; it is the standard. `clip_grad_value_` clamps each element and changes the direction; use it only when a few parameters spike.
- **Clip threshold.** Log the pre-clip norm (the return value) for a few hundred steps and set `max_norm` near its typical value. Clipping on every step means the threshold is too low and acts as a hidden learning-rate cut.
- **Fix the architecture first.** Residuals + normalization + correct init solve vanishing gradients at the source. Clipping is a safety net for occasional spikes.
- **Pre-LN vs Post-LN transformers.** Pre-LN (norm before each sub-layer) keeps gradients stable at depth and trains without long warmup. Post-LN can reach slightly better quality but is more fragile.

## Gotchas
- With mixed precision (`GradScaler`), call `scaler.unscale_(optimizer)` before `clip_grad_norm_`, otherwise you clip the scaled gradients ([[efficient-training-mixed-precision]]).
- With gradient accumulation, clip once after the last micro-batch, right before `optimizer.step()`, not after every `backward()`.
- Clip the parameters of the whole model in one call. Clipping each sub-module separately changes the relative scale between them.
- Dead ReLUs look like vanishing gradients (zero grads in some units) but have a different cause: large negative pre-activations. Leaky ReLU/GELU or a lower learning rate helps ([[activation-functions]]).
- A NaN can come from the data (a NaN feature, log of zero in the loss), not from exploding gradients. Check `torch.isfinite` on inputs and loss before tuning clipping. `torch.autograd.set_detect_anomaly(True)` points at the op that produced it (slow; debug only).
- Adam normalises each parameter's step by its gradient scale, which hides small vanishing gradients in the update size but not in the signal quality. Do not read "Adam trains it" as "the gradients are healthy".

## Related
- [[weight-initialization]] - Xavier/Kaiming derivations and choosing init per activation.
- [[residual-and-skip-connections]] - the identity path that lets very deep nets train.
- [[normalization-layers]] - BatchNorm/LayerNorm/RMSNorm for stable activation scale.
- [[recurrent-neural-networks-lstm-gru]] - gating as the RNN fix for vanishing gradients over time.
- [[backpropagation-and-autograd]] - where the product of Jacobians comes from.
- [[debugging-neural-network-training]] - full checklist when loss stalls or goes NaN.

## References
- Glorot & Bengio, Understanding the difficulty of training deep feedforward neural networks (2010): https://proceedings.mlr.press/v9/glorot10a.html
- He et al., Delving Deep into Rectifiers (Kaiming init, 2015): https://arxiv.org/abs/1502.01852
- Pascanu, Mikolov, Bengio, On the difficulty of training recurrent neural networks (2013): https://arxiv.org/abs/1211.5063
- PyTorch `clip_grad_norm_`: https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
