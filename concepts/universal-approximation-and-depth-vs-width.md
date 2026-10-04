---
title: Universal approximation theorem and depth vs width
category: concepts
tags: [universal-approximation, depth, width, expressivity, mlp, depth-separation, overparameterization, pytorch, deep-learning-basics]
use_cases:
  - "explain what the universal approximation theorem does and does not promise"
  - "decide whether to make a network wider or deeper"
  - "show why stacking linear layers without activations adds no power"
  - "size a small MLP for a regression or classification problem"
status: draft
last_verified: 2026-10-04
sources:
  - https://www.deeplearningbook.org/contents/mlp.html
  - https://link.springer.com/article/10.1007/BF02551274
  - https://arxiv.org/abs/1602.04485
  - https://arxiv.org/abs/1709.02540
  - https://pytorch.org/docs/stable/nn.html
---

# Universal approximation theorem and depth vs width

## Summary
The universal approximation theorem says a feed-forward network with a single hidden layer and a non-linear activation can approximate any continuous function on a bounded region to any accuracy, given enough hidden units. It is an existence result. It does not say how many units are needed, whether gradient descent will find the weights, or whether the network will generalise. Depth matters because some functions need exponentially many units in a shallow network but only a modest number in a deep one. In practice depth buys parameter efficiency, and residual connections and normalisation make depth trainable.

## Key concepts
- **The theorem (width version).** Cybenko (1989) for sigmoids, Hornik (1991) for any bounded non-constant activation, Leshno et al. (1993) for any non-polynomial activation including ReLU: one hidden layer of enough width is dense in the continuous functions on a compact set.
- **What it does not say.** Nothing about the required width (it can be exponential in the input dimension), about learnability by SGD, about sample complexity, or about extrapolation outside the training region.
- **Non-linearity is essential.** A stack of linear layers is a single linear map (`W3·W2·W1 = W`). Activations are what make depth add power ([[activation-functions]]).
- **ReLU nets are piecewise linear.** A ReLU network splits input space into linear regions. Depth multiplies the number of regions it can create, while width adds to it roughly linearly. That is one formal sense in which depth is more expressive.
- **Depth separation.** There are functions that a deep, narrow network represents with few units but that any shallow network needs exponentially many units to approximate (Telgarsky 2016, the "sawtooth" construction).
- **Width version for deep nets.** Fixed-width networks are also universal as depth grows, if the width is at least about the input dimension plus a small constant (Lu et al. 2017). Narrower than that, some functions cannot be approximated at any depth.
- **Inductive bias.** Depth suits compositional structure (edges → parts → objects, characters → words → phrases). This, more than raw expressivity, is why deep models work on images, audio and text.

## When to use / scenarios
- Explaining to a team why "a neural net can learn anything" is not a reason to skip feature design, data work or a simple baseline.
- Sizing a small MLP for tabular regression or a function-approximation problem (surrogate models, control, physics-informed nets): start with 2-4 hidden layers of moderate width rather than one huge layer.
- Diagnosing underfitting: if training loss stays high, add capacity (depth first, with residuals for deep stacks) or check optimisation before blaming the data ([[debugging-neural-network-training]]).
- Do NOT reach for a deep net on small tabular data because of this theorem. Gradient-boosted trees usually win there ([[deep-learning-for-tabular-data]]).

## Setup & code
`pip install torch`. Runs on CPU in about a minute. Fits a wiggly 1-D function, `sin(3x) + 0.3·cos(9x)`, with MLPs of different widths and depths, all trained for the same 3,000 Adam steps.

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
x = torch.linspace(-3, 3, 512).unsqueeze(1)
y = torch.sin(3 * x) + 0.3 * torch.cos(9 * x)


def mlp(width, depth):
    layers, d_in = [], 1
    for _ in range(depth):
        layers += [nn.Linear(d_in, width), nn.Tanh()]
        d_in = width
    return nn.Sequential(*layers, nn.Linear(d_in, 1))


def fit(model, steps=3000):
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
    return loss.item()


print(f"{'shape':>18} {'params':>7} {'train MSE':>10}")
for width, depth in [(4, 1), (16, 1), (64, 1), (256, 1), (16, 2), (16, 3), (32, 3)]:
    torch.manual_seed(0)
    m = mlp(width, depth)
    n = sum(p.numel() for p in m.parameters())
    print(f"{f'width {width} x depth {depth}':>18} {n:7d} {fit(m):10.5f}")

# Linear layers without activations collapse to one linear map
torch.manual_seed(0)
deep_linear = nn.Sequential(*[nn.Linear(1 if i == 0 else 64, 64) for i in range(4)], nn.Linear(64, 1))
print(f"5-layer linear net (no activations) MSE: {fit(deep_linear):.5f}")
```

Output (torch 2.13 CPU):
```
             shape  params  train MSE
 width 4 x depth 1      13    0.07322
width 16 x depth 1      49    0.03174
width 64 x depth 1     193    0.00878
width 256 x depth 1     769    0.02664
width 16 x depth 2     321    0.00696
width 16 x depth 3     593    0.00041
width 32 x depth 3    2209    0.00760
5-layer linear net (no activations) MSE: 0.53037
```

What the numbers show:
- Wider single layers fit better, as the theorem promises, up to a point. Width 256 does *worse* than width 64 in the same step budget: the extra units exist, but the optimiser has not used them yet. Existence is not learnability.
- `16 x 3` (593 parameters) fits about 20x better than the best single layer, with fewer parameters than `256 x 1`.
- More is not monotonically better: `32 x 3` is worse than `16 x 3` here. With a fixed number of steps, results depend on optimisation and the seed. Compare several seeds before drawing conclusions.
- The 5-layer linear net is no better than a straight line (MSE ≈ the variance of `y` around its linear trend), however many layers it has.

## Choosing / trade-offs
- **Deeper**: more parameter-efficient and a better fit for compositional data. It is harder to optimise (vanishing gradients, sensitivity to initialisation), so beyond a few layers it needs residual connections and normalisation ([[residual-and-skip-connections]], [[vanishing-and-exploding-gradients]]). It also adds sequential latency per layer.
- **Wider**: easier to optimise, more parallel on GPUs, and in the very wide limit training becomes close to a kernel method. It uses more parameters for the same function.
- **Rule of thumb for MLPs**: 2-4 hidden layers, widths of 64-512, tuned together with learning rate and regularisation, beat both one huge layer and a very deep plain stack. Scaling laws for large models suggest that, within a sensible range of shapes, the total parameter count matters more than the exact depth/width ratio ([[pretraining-and-scaling-laws]]).

## Gotchas
- Reading the theorem as "neural nets can learn any function from data". It is about representation, not about training or generalisation.
- Approximation holds on a bounded input region. Outside the training range, a ReLU net extrapolates linearly and a tanh net goes flat. Neither recovers a periodic or exponential trend.
- Forgetting the activation between layers (a common bug in hand-written `nn.Sequential`). The model silently becomes linear.
- Comparing shapes at a fixed step count and one seed, then crediting the architecture for what was an optimisation difference.
- A very narrow bottleneck layer (width below the input dimension) limits what any later depth can recover.
- Huge capacity on little data memorises it. Add regularisation and validation, not just units ([[regularization-in-deep-learning]]).

## Related
- [[neural-network-fundamentals]] - layers, activations and training in full.
- [[perceptron-and-linear-separability]] - the single-neuron limit that hidden layers overcome.
- [[activation-functions]] - the non-linearity the theorem requires.
- [[residual-and-skip-connections]] - what makes deep stacks trainable.
- [[loss-landscapes-and-flat-minima]] - why overparameterised nets are still optimisable.
- [[generalization-in-deep-learning]] - why huge networks still generalise.

## References
- Goodfellow, Bengio, Courville, Deep Learning, sec. 6.4 Architecture Design: https://www.deeplearningbook.org/contents/mlp.html
- Cybenko, Approximation by Superpositions of a Sigmoidal Function (1989): https://link.springer.com/article/10.1007/BF02551274
- Telgarsky, Benefits of Depth in Neural Networks (2016): https://arxiv.org/abs/1602.04485
- Lu et al., The Expressive Power of Neural Networks: A View from the Width (2017): https://arxiv.org/abs/1709.02540
- PyTorch `torch.nn`: https://pytorch.org/docs/stable/nn.html
