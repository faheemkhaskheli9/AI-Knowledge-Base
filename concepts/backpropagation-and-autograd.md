---
title: Backpropagation and automatic differentiation (autograd)
category: concepts
tags: [deep-learning, backpropagation, autograd, automatic-differentiation, chain-rule, computational-graph, gradient-check, custom-gradient, gradient-accumulation, pytorch, jax]
use_cases:
  - "understand how a framework computes gradients for any model"
  - "write a custom autograd function or operation with its own backward pass"
  - "verify hand-written gradients with a numerical gradient check"
  - "debug None gradients, in-place errors or graph memory leaks in PyTorch"
  - "accumulate gradients over several micro-batches"
status: draft
last_verified: 2026-10-04
sources:
  - https://pytorch.org/docs/stable/notes/autograd.html
  - https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
  - https://pytorch.org/docs/stable/notes/extending.html
  - https://docs.jax.dev/en/latest/automatic-differentiation.html
  - https://www.deeplearningbook.org/contents/mlp.html
---

# Backpropagation and automatic differentiation (autograd)

## Summary
Backpropagation computes the gradient of the loss with respect to every parameter in one backward sweep. It applies the chain rule from the loss back to the inputs, reusing intermediate results so the cost stays about that of a couple of forward passes. Frameworks implement it as reverse-mode automatic differentiation: during the forward pass they record a graph of operations, and `loss.backward()` walks that graph in reverse. You rarely write gradients by hand. You do need the mental model to read errors, write custom ops, manage memory, and spot why a parameter gets no gradient.

## Key concepts
- **Computational graph.** Each tensor op becomes a node that stores what it needs for its local derivative (for example the inputs to a multiply). In PyTorch the graph is built dynamically on every forward pass ("define by run").
- **Chain rule, backwards.** For `L = f(g(h(x)))`, `dL/dx = f'·g'·h'`. Reverse mode starts with `dL/dL = 1` and multiplies by each node's local Jacobian-vector product on the way back. Each parameter's `.grad` is the sum over all paths that reach it.
- **Reverse vs forward mode.** Reverse mode costs one backward pass per *scalar output*, which is ideal for a scalar loss and millions of parameters. Forward mode costs one pass per *input*, which suits few inputs and many outputs (Jacobians of small functions). JAX exposes both (`jax.grad`, `jax.jvp`, `jax.vjp`); PyTorch has `torch.func.jvp` / `jacrev` / `jacfwd`.
- **Leaf tensors and `requires_grad`.** Parameters are leaf tensors with `requires_grad=True`; their gradients are stored in `.grad`. Intermediate tensors get gradients only if you call `.retain_grad()`.
- **Gradients accumulate.** `.backward()` *adds* into `.grad`. That is why the loop calls `optimizer.zero_grad()` each step, and also why gradient accumulation over micro-batches works.
- **Graph lifetime.** The graph and its saved activations are freed after `.backward()` unless `retain_graph=True`. Saved activations are the main memory cost of training. Activation checkpointing trades recompute for memory ([[efficient-training-mixed-precision]]).
- **Stopping gradients.** `torch.no_grad()` / `torch.inference_mode()` skip graph building (eval, inference). `tensor.detach()` cuts one tensor out of the graph (targets in RL/TD learning, stop-gradient in self-supervised methods). `jax.lax.stop_gradient` is the JAX equivalent.
- **Vanishing and exploding gradients** come straight from the chain rule: a long product of factors < 1 shrinks to zero, and > 1 explodes. Activations, init, normalisation, residuals and clipping all exist to keep that product near 1 ([[activation-functions]], [[weight-initialization]], [[normalization-layers]]).

## When to use / scenarios
- Every gradient-trained model uses this implicitly; read this file when you need to go below `loss.backward()`.
- Custom op with a known or more stable gradient (a fused kernel, a straight-through estimator for quantisation, gradient reversal for domain adaptation): `torch.autograd.Function`.
- Effective batch larger than memory: accumulate gradients over k micro-batches, scale the loss by 1/k, step once.
- Higher-order derivatives (gradient penalties in WGAN-GP, physics-informed losses, MAML): `create_graph=True` / `torch.autograd.grad`, or nested `jax.grad` ([[physics-informed-neural-networks]], [[meta-learning]]).
- Per-sample gradients (differential privacy, influence analysis): `torch.func.vmap(grad(...))` or JAX `vmap`.
- NOT needed for gradient-free models (trees, k-NN) or black-box tuning ([[black-box-and-evolutionary-optimization]]).

## Setup & code
```bash
pip install torch
```
Backprop by hand for a 2-layer net, checked against autograd and a finite-difference gradient:
```python
import torch

torch.manual_seed(0)
x = torch.randn(8, 3, dtype=torch.float64)
y = torch.randn(8, 1, dtype=torch.float64)
W1 = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
W2 = torch.randn(4, 1, dtype=torch.float64, requires_grad=True)

def loss_fn(W1, W2):
    h = torch.tanh(x @ W1)
    return ((h @ W2 - y) ** 2).mean()

# 1) autograd
loss = loss_fn(W1, W2)
loss.backward()

# 2) manual chain rule, step by step backwards
with torch.no_grad():
    a = x @ W1; h = torch.tanh(a); out = h @ W2
    d_out = 2 * (out - y) / out.numel()      # dL/d_out
    dW2 = h.T @ d_out                        # through out = h @ W2
    d_h = d_out @ W2.T
    d_a = d_h * (1 - h ** 2)                 # tanh' = 1 - tanh^2
    dW1 = x.T @ d_a
print("manual == autograd:", torch.allclose(dW1, W1.grad), torch.allclose(dW2, W2.grad))

# 3) numerical gradient check (use float64)
print("gradcheck:", torch.autograd.gradcheck(loss_fn, (W1, W2)))
```
A custom autograd function (straight-through estimator for rounding):
```python
class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out                       # pretend round() is the identity

w = torch.tensor([0.3, 1.7], requires_grad=True)
RoundSTE.apply(w).sum().backward()
print(w.grad)                                 # tensor([1., 1.]); plain round() gives zeros
```
Gradient accumulation over 4 micro-batches:
```python
model = torch.nn.Linear(10, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
accum = 4
opt.zero_grad()
for i in range(16):
    xb, yb = torch.randn(8, 10), torch.randn(8, 1)
    loss = torch.nn.functional.mse_loss(model(xb), yb) / accum   # scale so the sum is a mean
    loss.backward()                                              # grads add up
    if (i + 1) % accum == 0:
        opt.step()
        opt.zero_grad()
```

## Choosing / trade-offs
- **PyTorch eager autograd vs JAX transforms.** PyTorch builds a graph per call and is the easier to debug. JAX differentiates pure functions (`jax.grad(f)`), composes with `jit`/`vmap`, and makes higher-order and per-example gradients trivial. See [[jax-and-flax]].
- **Memory vs compute.** Keeping all activations is fastest. Checkpointing recomputes segments in backward and cuts activation memory a lot for ~20-30% more compute.
- **Custom backward vs letting autograd differentiate.** Write one only for numerical stability, speed (fused kernels) or a deliberate surrogate gradient. Otherwise it is extra code to get wrong; always `gradcheck` it.

## Gotchas
- Missing `optimizer.zero_grad()` makes gradients sum across steps. Training looks like it has a huge, growing LR.
- `param.grad is None` after backward means the parameter was not used in the loss, was used through a `.detach()`, `.item()`, `.numpy()` or `torch.no_grad()` block, or was replaced by a non-leaf (for example `w = w.cuda()` after creating `w` with `requires_grad`).
- "one of the variables needed for gradient computation has been modified by an inplace operation": an in-place op (`x += 1`, `relu_(inplace=True)`, `x[idx] = ...`) overwrote a value saved for backward. Use the out-of-place version.
- Accumulating `total_loss += loss` (not `loss.item()`) for logging keeps every step's graph alive and leaks memory.
- Calling `.backward()` twice on the same graph fails unless `retain_graph=True`; usually the real fix is to compute both losses and sum them before one backward.
- Non-differentiable ops (`argmax`, `round`, indexing with computed integer indices, sampling) pass zero or no gradient. Use a relaxation (softmax, Gumbel-softmax), the reparameterisation trick or a straight-through estimator.
- `gradcheck` in float32 fails spuriously; run it in float64 on small inputs.

## Related
- [[neural-network-fundamentals]] - the forward pass, losses and gradient-descent update that backprop serves.
- [[deep-learning-training]] - optimizers that consume the gradients, clipping, schedules.
- [[debugging-neural-network-training]] - per-layer gradient norms, NaN hunting.
- [[pytorch-basics]] - the standard training loop built on autograd.
- [[jax-and-flax]] - functional autodiff with `grad`, `vmap`, `jit`.
- [[math-for-machine-learning]] - derivatives, Jacobians and the chain rule.

## References
- PyTorch autograd mechanics: https://pytorch.org/docs/stable/notes/autograd.html
- PyTorch autograd tutorial: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
- Extending PyTorch (custom `autograd.Function`): https://pytorch.org/docs/stable/notes/extending.html
- JAX automatic differentiation: https://docs.jax.dev/en/latest/automatic-differentiation.html
- Goodfellow, Bengio & Courville, *Deep Learning*, ch. 6.5 Back-Propagation: https://www.deeplearningbook.org/contents/mlp.html
