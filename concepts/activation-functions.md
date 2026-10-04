---
title: Activation functions (ReLU, GELU, SiLU, sigmoid, tanh, softmax, GLU)
category: concepts
tags: [deep-learning, activation-function, relu, leaky-relu, gelu, silu, swish, sigmoid, tanh, softmax, glu, swiglu, dead-relu, pytorch]
use_cases:
  - "choose the hidden-layer activation for a new MLP, CNN or transformer"
  - "fix dead ReLU units or vanishing gradients in a deep network"
  - "pick the right output activation for regression, binary, multi-label or multi-class"
  - "understand why LLM feed-forward blocks use SwiGLU instead of ReLU"
status: draft
last_verified: 2026-10-04
sources:
  - https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
  - https://arxiv.org/abs/1606.08415
  - https://arxiv.org/abs/1710.05941
  - https://arxiv.org/abs/2002.05202
  - https://d2l.ai/chapter_multilayer-perceptrons/mlp.html
---

# Activation functions (ReLU, GELU, SiLU, sigmoid, tanh, softmax, GLU)

## Summary
An activation function is the element-wise non-linearity applied after each linear layer. Without it, any stack of linear layers collapses into one linear map, so the network could only learn linear functions. The choice affects how well gradients flow through depth, how fast training converges and what the output means. In practice there are two separate decisions: the hidden activation (ReLU, GELU or SiLU almost always) and the output activation, which is fixed by the task and the loss.

## Key concepts
- **Why non-linearity.** `W2(W1 x) = (W2 W1) x`: two linear layers equal one. A non-linearity between them is what lets an MLP approximate arbitrary functions ([[neural-network-fundamentals]]).
- **Hidden activations.**
  - **ReLU** `max(0, x)`: cheap, gradient 1 for x > 0 so it does not saturate on the positive side. Default for CNNs and MLPs. Units stuck at x < 0 get zero gradient and can "die".
  - **Leaky ReLU / PReLU** `max(ax, x)`, a ≈ 0.01 (PReLU learns a): keeps a small gradient for x < 0, avoiding dead units.
  - **ELU / SELU**: smooth negative side; SELU with LeCun init and `AlphaDropout` gives self-normalising MLPs. Rarely needed now that normalisation layers exist.
  - **GELU** `x·Φ(x)`: smooth ReLU-like curve; default in BERT, GPT-2 and ViT.
  - **SiLU / Swish** `x·sigmoid(x)`: similar shape to GELU; used in EfficientNet, YOLO and many LLMs.
  - **Mish, hard-swish**: mobile-friendly or niche variants; small gains, not worth chasing first.
- **Gated linear units (GLU family).** `GLU(x) = (x W) ⊙ σ(x V)`. **SwiGLU** (SiLU gate) and **GeGLU** (GELU gate) replace the ReLU/GELU FFN in most modern LLMs (LLaMA, Mistral, PaLM). They use three weight matrices instead of two, so the hidden width is usually cut to ~2/3 to keep the parameter count equal.
- **Saturating activations.** **Sigmoid** (0..1) and **tanh** (-1..1) have gradients near zero for large |x|, which causes vanishing gradients in deep stacks. They survive where a bounded value is the point: gates in LSTM/GRU, attention-style gating, and output layers.
- **Output activations follow the task and loss.**

  | Task | Output layer | Loss (PyTorch) |
  |---|---|---|
  | Regression | none (linear) | `MSELoss`, `HuberLoss`, `L1Loss` |
  | Binary / multi-label | none in the model; sigmoid at inference | `BCEWithLogitsLoss` |
  | Multi-class (one label) | none in the model; softmax at inference | `CrossEntropyLoss` |
  | Positive value (rate, variance) | softplus or exp | Poisson / Gaussian NLL |
  | Bounded value in a range | sigmoid or tanh, rescaled | MSE |

- **Softmax** turns a vector of logits into a probability distribution. It is invariant to adding a constant, which is why stable implementations subtract the max; `log_softmax` is more stable than `log(softmax)`. Temperature (`softmax(z / T)`) sharpens or flattens it ([[decoding-and-sampling]]).

## When to use / scenarios
- New MLP or CNN: ReLU. Switch to Leaky ReLU if you see many dead units, or to SiLU/GELU if a reference architecture uses it.
- Transformer encoder or ViT: GELU. Decoder LLM FFN: SwiGLU (copy the reference implementation's hidden-width rule).
- Recurrent cells and gates: sigmoid for gates, tanh for the candidate state; do not swap these for ReLU.
- Output layer: let the loss decide (table above). The most common bug is adding a softmax or sigmoid before a loss that already applies it.
- NOT a tuning priority: after picking a sensible default, learning rate, data and normalisation matter far more than activation choice ([[deep-learning-training]]).

## Setup & code
```bash
pip install torch
```
Count dead ReLU units vs Leaky ReLU / GELU in a deep MLP after one aggressive step:
```python
import torch
from torch import nn

torch.manual_seed(0)
X = torch.randn(2048, 64)
y = (X[:, :8].sum(1) > 0).long()

def mlp(act):
    layers, d = [], 64
    for _ in range(8):
        layers += [nn.Linear(d, 256), act()]
        d = 256
    return nn.Sequential(*layers, nn.Linear(d, 2))

for name, act in [("relu", nn.ReLU), ("leaky", nn.LeakyReLU), ("gelu", nn.GELU)]:
    torch.manual_seed(0)
    model = mlp(act)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)       # deliberately too high
    loss = nn.functional.cross_entropy(model(X), y)
    opt.zero_grad(); loss.backward(); opt.step()

    dead, total, h = 0, 0, X
    with torch.no_grad():
        for layer in model[:-1]:
            h = layer(h)
            if isinstance(layer, act):
                dead += (h.abs().max(0).values < 1e-6).sum().item()  # never active on any input
                total += h.shape[1]
    print(f"{name:6s} dead units after one big step: {dead}/{total}")
# Observed (torch 2.13 CPU): relu 401/2048, leaky 0/2048, gelu 0/2048.
```
A SwiGLU feed-forward block as used in LLaMA-style models:
```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.w_down(nn.functional.silu(self.w_gate(x)) * self.w_up(x))

print(SwiGLU(512, int(2 / 3 * 4 * 512))(torch.randn(2, 10, 512)).shape)  # (2, 10, 512)
```

## Choosing / trade-offs
- **ReLU vs GELU/SiLU.** ReLU is fastest and fine for most CNNs/MLPs. GELU/SiLU are smooth and give small, consistent gains in transformers; the cost is a few extra FLOPs, negligible next to matmuls.
- **Plain FFN vs GLU.** SwiGLU improves LLM quality at equal parameters (Shazeer 2020) but adds a third matrix and kernel launch; for small models or CNNs it is not worth the complexity.
- **Smooth vs piecewise-linear for deployment.** ReLU and hard-swish quantise and fuse easily on edge hardware; GELU sometimes needs an approximation (`approximate="tanh"`) on accelerators.
- **Match the pretrained model.** When fine-tuning, never change the activation of a pretrained network: the weights were learned for it.

## Gotchas
- `nn.Softmax` before `nn.CrossEntropyLoss`, or `nn.Sigmoid` before `nn.BCEWithLogitsLoss`, applies the function twice: training still runs but learns badly. Output raw logits.
- Softmax over the wrong dimension (`dim=0` over the batch) silently produces nonsense probabilities; always pass `dim=-1` explicitly.
- Many dead ReLUs usually mean a too-high learning rate or a large negative bias, not that ReLU is wrong; lower LR or check initialisation ([[weight-initialization]]) before switching activation.
- Sigmoid/tanh hidden layers in a deep feed-forward net train slowly or stall (vanishing gradients); use them only for gates and bounded outputs.
- The init gain must match the activation (He for ReLU-family, Xavier for tanh/sigmoid); see [[weight-initialization]].
- `nn.ReLU(inplace=True)` saves memory but breaks autograd if another branch needs the pre-activation value.

## Related
- [[neural-network-fundamentals]] - where activations sit in a layer and why depth needs them.
- [[weight-initialization]] - init gains are derived per activation.
- [[normalization-layers]] - keep pre-activations in the range where the activation is useful.
- [[loss-functions]] - which output activation each loss already includes.
- [[transformers-and-attention]] - GELU and SwiGLU in the feed-forward block.
- [[cnn-and-rnn-architectures]] - ReLU in CNNs, sigmoid/tanh gates in LSTM/GRU.

## References
- PyTorch non-linear activations: https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
- Hendrycks & Gimpel, Gaussian Error Linear Units (GELUs): https://arxiv.org/abs/1606.08415
- Ramachandran et al., Searching for Activation Functions (Swish): https://arxiv.org/abs/1710.05941
- Shazeer, GLU Variants Improve Transformer: https://arxiv.org/abs/2002.05202
- *Dive into Deep Learning*, Multilayer Perceptrons: https://d2l.ai/chapter_multilayer-perceptrons/mlp.html
