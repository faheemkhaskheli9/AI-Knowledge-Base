---
title: Autograd engine from scratch (a scalar reverse-mode autodiff in pure Python)
category: concepts
tags: [autograd, automatic-differentiation, reverse-mode, backpropagation, computation-graph, chain-rule, micrograd, pytorch, from-scratch, dl-basics]
use_cases:
  - "build a tiny autograd engine from scratch to understand how loss.backward() works"
  - "explain the computation graph, topological sort and gradient accumulation in an interview"
  - "train a small neural network with no libraries except the Python standard library"
status: draft
last_verified: 2026-10-04
sources:
  - https://github.com/karpathy/micrograd
  - https://pytorch.org/docs/stable/notes/autograd.html
  - https://www.jmlr.org/papers/v18/17-468.html
---

# Autograd engine from scratch (a scalar reverse-mode autodiff in pure Python)

## Summary
An autograd engine records every arithmetic operation as a node in a graph while the forward pass runs, then walks that graph backwards applying the chain rule to get the gradient of the output with respect to every input. Doing it for scalars takes about 60 lines of plain Python: a `Value` class that stores its data, its gradient, its parents and a closure that pushes gradient to those parents. The engine below matches PyTorch's gradients to six decimal places and trains a 2-8-1 MLP on XOR with no other library.

## Key concepts
- **Computation graph.** Each operation creates a new `Value` that keeps references to its inputs (`_parents`). The graph is built implicitly by ordinary Python expressions; this is the "define-by-run" style PyTorch uses.
- **Local derivative closure.** Each operation knows only its own derivative: `+` passes the gradient through unchanged, `*` multiplies it by the other operand, `tanh` multiplies by `1 − tanh²`. The closure `_backward` applies that rule.
- **Chain rule as message passing.** `self.grad += local_derivative * out.grad`. The upstream gradient `out.grad` must be complete before a node passes it on.
- **Topological order.** A depth-first search lists nodes so that every node comes after its inputs. Running `_backward` in reverse of that list guarantees each node's gradient is fully summed before it is used.
- **Accumulate, don't assign.** A value used twice (`a` in `a*b + a**2`) receives gradient from both uses, so gradients are summed with `+=`. That is also why every training loop must zero gradients between steps.
- **Reverse mode vs forward mode.** Reverse mode costs about one extra forward pass to get the gradient with respect to *all* inputs of a scalar output, which is exactly the shape of a loss. Forward mode is cheaper when there are few inputs and many outputs ([[backpropagation-and-autograd]]).

## When to use / scenarios
- Learning: the cleanest way to see that backprop is just the chain rule plus bookkeeping.
- Interviews: "implement backward for multiply", "why do gradients accumulate", "why do you call `zero_grad()`".
- Debugging intuition for PyTorch errors such as "Trying to backward through the graph a second time" or in-place ops breaking autograd.
- Not for: real training. Scalar nodes are millions of times slower than tensor kernels. Use PyTorch, JAX or TensorFlow ([[pytorch-basics]], [[jax-and-flax]]).

## Setup & code
Standard library only for the engine; `pip install torch` only to check the gradients. Runs in about a second.

```python
import math
import random


class Value:
    """A scalar that remembers how it was computed, so it can backpropagate."""

    def __init__(self, data, parents=(), op=""):
        self.data, self.grad = float(data), 0.0
        self._parents, self._op = parents, op
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += out.grad                 # d(a+b)/da = 1
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad    # d(a*b)/da = b
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):                         # k is a plain number
        out = Value(self.data ** k, (self,), f"**{k}")
        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")
        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        order, seen = [], set()
        def visit(v):                             # topological sort: parents before children
            if v not in seen:
                seen.add(v)
                for p in v._parents:
                    visit(p)
                order.append(v)
        visit(self)
        self.grad = 1.0
        for v in reversed(order):                 # children first, so out.grad is complete
            v._backward()

    __radd__ = __add__
    __rmul__ = __mul__
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __truediv__(self, other): return self * other ** -1


# 1) Check one gradient against PyTorch
import torch
a, b = Value(2.0), Value(-3.0)
f = (a * b + a ** 2).tanh() * a + b / a
f.backward()
ta = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
tb = torch.tensor(-3.0, dtype=torch.float64, requires_grad=True)
tf = torch.tanh(ta * tb + ta ** 2) * ta + tb / ta
tf.backward()
print(f"f={f.data:.6f} torch={tf.item():.6f}")
print(f"df/da={a.grad:.6f} torch={ta.grad.item():.6f}  df/db={b.grad:.6f} torch={tb.grad.item():.6f}")


# 2) Train a tiny MLP (2-8-1, tanh) on XOR with nothing but Value
random.seed(0)
def neuron(n_in): return [Value(random.uniform(-1, 1)) for _ in range(n_in)] + [Value(0.0)]
hidden = [neuron(2) for _ in range(8)]
out_n = neuron(8)
params = [p for n in hidden + [out_n] for p in n]

def forward(x):
    h = [(sum(w * xi for w, xi in zip(n[:-1], x)) + n[-1]).tanh() for n in hidden]
    return (sum(w * hi for w, hi in zip(out_n[:-1], h)) + out_n[-1]).tanh()

X = [(0, 0), (0, 1), (1, 0), (1, 1)]
Y = [-1, 1, 1, -1]
for step in range(201):
    loss = sum((forward(x) - y) ** 2 for x, y in zip(X, Y)) * 0.25
    for p in params:
        p.grad = 0.0                              # grads accumulate, so reset each step
    loss.backward()
    for p in params:
        p.data -= 0.2 * p.grad
    if step % 50 == 0:
        print(f"step {step:3d} loss {loss.data:.4f}")
print("predictions", [round(forward(x).data, 3) for x in X], "params", len(params))
```

Output (Python 3.14, torch 2.13):
```
f=-3.428055 torch=-3.428055
df/da=-0.072726 torch=-0.072726  df/db=0.782603 torch=0.782603
step   0 loss 1.1784
step  50 loss 0.0883
step 100 loss 0.0238
step 150 loss 0.0122
step 200 loss 0.0079
predictions [-0.932, 0.911, 0.908, -0.898] params 33
```

The value and both partial derivatives agree with PyTorch to every printed digit, including `a`, which appears three times in `f` and so collects gradient along three paths. The 33-parameter network learns XOR (a problem no linear model can solve, see [[perceptron-and-linear-separability]]) in 200 steps of plain gradient descent.

## Choosing / trade-offs
- **Scalar vs tensor nodes.** Scalar nodes make every rule obvious but create one Python object per multiply. Real engines make each node a whole tensor op (matmul, conv) with a hand-written vectorised backward, which is the same idea at a coarser grain ([[neural-network-from-scratch-numpy]]).
- **Closures vs an op table.** Storing a closure per node is the shortest code. Frameworks store an op type plus saved tensors instead, which lets them free memory, serialise graphs and run backward on another device.
- **Dynamic vs static graphs.** Rebuilding the graph on every forward pass (PyTorch, this engine) handles Python control flow for free. Tracing once and compiling (JAX `jit`, `torch.compile`) is faster but needs shapes and branches to stay fixed.

## Gotchas
- Forgetting to zero gradients. The second `backward()` adds to the first, so the step uses the sum of two gradients. PyTorch has the same behaviour and the same fix (`optimizer.zero_grad()`).
- Using `=` instead of `+=` in a `_backward`. Any value used more than once gets only the gradient from its last use. The PyTorch comparison above catches this immediately, because `a` is reused.
- Running `_backward` in the wrong order (or in plain creation order) passes on a partial gradient. Always topologically sort.
- Recursion depth. The recursive DFS fails on graphs deeper than Python's recursion limit (about 1000); a long unrolled RNN hits it. Use an explicit stack for deep graphs.
- `__pow__` here handles a constant exponent only. `x ** y` with both as `Value` needs the `log` rule, `d/dy x^y = x^y · ln x`, which is undefined for `x ≤ 0`.
- `__radd__` is needed because `sum()` starts from the integer `0`, and `0 + Value` calls `Value.__radd__`.

## Related
- [[backpropagation-and-autograd]] - the theory and PyTorch's autograd API, including custom `autograd.Function`.
- [[neural-network-from-scratch-numpy]] - the same backward pass done by hand per layer on whole matrices.
- [[gradient-descent]] - the update rule the training loop applies.
- [[perceptron-and-linear-separability]] - why XOR needs a hidden layer.
- [[pytorch-basics]] - the production engine this toy mirrors.

## References
- Karpathy, micrograd (the design this example follows): https://github.com/karpathy/micrograd
- PyTorch, Autograd mechanics: https://pytorch.org/docs/stable/notes/autograd.html
- Baydin et al., "Automatic Differentiation in Machine Learning: a Survey", JMLR 2018: https://www.jmlr.org/papers/v18/17-468.html
