---
title: JAX and Flax
category: ml
tags: [jax, flax, nnx, optax, xla, jit, grad, vmap, tpu, deep-learning, numpy]
use_cases:
  - "train a model on TPUs or scale it across many accelerators"
  - "write fast numerical code with NumPy syntax and automatic differentiation"
  - "vectorize a per-example function (per-sample gradients, ensembles) with vmap"
  - "speed up a scientific simulation or custom loss with jit compilation"
status: draft
last_verified: 2026-10-04
sources:
  - https://docs.jax.dev/en/latest/
  - https://flax.readthedocs.io/en/stable/
  - https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/training/optimizer.html
  - https://optax.readthedocs.io/
---

# JAX and Flax

## Summary
JAX is NumPy with composable function transformations: `grad` (derivatives),
`jit` (compile with XLA), `vmap` (auto-vectorize) and sharding across devices.
It has no layers or optimizers itself; Flax adds neural-network modules (the
current API is Flax NNX) and Optax adds optimizers. It is the main framework
on TPUs and for large-scale training at Google, and common in scientific ML.

## Key concepts
- **`jax.numpy`** mirrors NumPy, but arrays are immutable: `x.at[i].set(v)`
  instead of `x[i] = v`.
- **Pure functions.** Transformations assume no side effects: no global state,
  no in-place mutation, no Python `print` you rely on inside `jit`
  (use `jax.debug.print`).
- **`jax.grad` / `jax.value_and_grad`** differentiate a scalar-valued function
  with respect to its first argument (or `argnums`).
- **`jax.jit`** traces the function once per input shape/dtype and compiles
  it. Python `if` on traced values fails; use `jnp.where` or `jax.lax.cond`.
- **`jax.vmap`** maps a function written for one example over a batch axis.
- **Explicit randomness.** No global seed: pass a key
  (`jax.random.key(0)`) and `split` it for every new random draw.
- **Flax NNX** modules are Python objects holding `nnx.Param` state;
  `nnx.Optimizer(model, optax_tx, wrt=nnx.Param)` updates them, and
  `nnx.jit` / `nnx.grad` are the module-aware transforms.

## When to use / scenarios
- TPU training (Google Cloud), or very large models sharded over many
  devices.
- Research needing unusual derivatives: per-example gradients, Hessians,
  meta-learning, differentiable simulators (see
  [[physics-informed-neural-networks]]).
- Fast array code that is too slow in NumPy and does not need a full
  deep-learning stack.
- NOT for: everyday fine-tuning of open models, where tools and checkpoints
  are PyTorch-first (see [[pytorch-basics]], [[huggingface-transformers]]);
  teams wanting `fit()` convenience (Keras 3 on the JAX backend gives that,
  see [[keras-and-tensorflow]]).

## Setup & code
```bash
pip install -U jax flax optax          # CPU; GPU: pip install -U "jax[cuda12]"
```
```python
import jax, jax.numpy as jnp, optax
from flax import nnx

key = jax.random.key(0)
X = jax.random.normal(key, (1000, 20))
y = (X[:, 0] + X[:, 1] > 0).astype(jnp.int32)

class MLP(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.l1 = nnx.Linear(20, 64, rngs=rngs)
        self.l2 = nnx.Linear(64, 2, rngs=rngs)
    def __call__(self, x):
        return self.l2(nnx.relu(self.l1(x)))

model = MLP(nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adamw(1e-3), wrt=nnx.Param)

@nnx.jit
def train_step(model, optimizer, xb, yb):
    def loss_fn(model):
        logits = model(xb)
        return optax.softmax_cross_entropy_with_integer_labels(logits, yb).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss

for epoch in range(5):
    for i in range(0, len(X), 64):
        loss = train_step(model, optimizer, X[i:i + 64], y[i:i + 64])
    acc = (model(X).argmax(-1) == y).mean()
    print(epoch, float(loss), float(acc))
```
Pure JAX, no Flax: `jax.grad(lambda w: jnp.mean((X @ w - t) ** 2))(w)`.

## Choosing / trade-offs
- **JAX vs PyTorch.** JAX: functional style, strong compiler, best TPU
  support, clean higher-order derivatives. PyTorch: eager-first, easier
  debugging, far larger model zoo and hiring pool.
- **Flax NNX vs Linen.** NNX (stateful Python objects) is the current
  recommended API; Linen (`init`/`apply` with separate param trees) is the
  older one still found in many codebases and checkpoints.
- **jit granularity.** Jit the whole train step, not tiny helpers; each
  compiled call has dispatch overhead and each new shape triggers a recompile.

## Gotchas
- The first call to a jitted function is slow (compilation); time the second.
- Changing batch shapes (e.g. the last, smaller batch) recompiles; drop or pad
  the remainder.
- JAX preallocates most GPU memory at start-up; set
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` when sharing a GPU.
- Default dtype is float32 even for Python floats; float64 needs
  `jax.config.update("jax_enable_x64", True)`.
- Reusing a random key gives identical "random" numbers; always split.
- Out-of-bounds indexing does not raise inside jit; it clamps silently.
- Native Windows is CPU-only; use WSL2 for GPU (see [[windows-wsl-setup]]).

## Related
- [[pytorch-basics]] - the eager-mode alternative.
- [[keras-and-tensorflow]] - Keras 3 can run on JAX with `fit()`.
- [[distributed-training]] - sharding and data/model parallelism concepts.
- [[math-for-machine-learning]] - gradients and autodiff behind `grad`.
- [[physics-informed-neural-networks]] - a common JAX use case.

## References
- https://docs.jax.dev/en/latest/
- https://flax.readthedocs.io/en/stable/
- https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/training/optimizer.html
- https://optax.readthedocs.io/
