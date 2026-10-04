---
title: Knowledge distillation from scratch in NumPy (soft targets, temperature, T² scaling)
category: concepts
tags: [knowledge-distillation, distillation, teacher-student, soft-targets, temperature, model-compression, numpy, from-scratch]
use_cases:
  - "train a small student model to imitate a large teacher's output distribution"
  - "see why soft targets carry more information than one-hot labels when labeled data is scarce"
  - "get the temperature and T² gradient scaling of the distillation loss right"
  - "explain knowledge distillation in an interview"
status: draft
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/1503.02531
  - https://arxiv.org/abs/1910.01108
---

# Knowledge distillation from scratch in NumPy (soft targets, temperature, T² scaling)

## Summary
Knowledge distillation trains a small student to match a large teacher's predicted probabilities, not only the one-hot labels. Softened with a temperature `T`, the teacher's output says "class 1, but a bit like 0 and 2", and that similarity structure is extra training signal for each example. Below, a 128-unit teacher MLP is trained on 3000 labeled points of a noisy 3-class spiral (98.9% test accuracy). An 8-unit student is then trained on only 60 points. With hard labels it reaches 92.1%. Distilled from the teacher's `T = 4` soft targets on the same 60 points it reaches 96.7%, averaged over five seeds. The whole thing is a two-layer MLP with hand-written backprop and one extra loss term.

## Key concepts
- **Soft targets.** `q = softmax(z_teacher / T)`. At `T = 1` a confident teacher outputs nearly one-hot vectors. Raising `T` flattens the distribution so the relative sizes of the wrong-class probabilities become visible.
- **Distillation loss.** `L = (1 − α) · CE(y, softmax(z_s)) + α · T² · CE(q, softmax(z_s / T))`. The student uses the same `T` as the teacher in the soft term and `T = 1` at inference.
- **T² scaling.** The gradient of the soft term with respect to the student logits is `(p_T − q_T) / T`, which shrinks as `T` grows. Multiplying by `T²` keeps its size comparable to the hard-label term, so `α` means the same thing at any temperature.
- **Why it helps.** Each soft target is a full distribution instead of one class index. The teacher has already smoothed noisy labels and encoded which classes are close, so the student needs fewer examples to learn the same decision boundary.
- **Transfer set.** The soft term needs no labels, only teacher outputs. In practice the student is often distilled on a large unlabeled or synthetic set, which is where most of the gain comes from in real systems.

## When to use / scenarios
- Learning: the core idea behind DistilBERT, TinyBERT, small "distilled" LLMs and many on-device vision models.
- Practice: shrinking a model for latency or cost ([[cost-and-latency]]) while keeping most of its accuracy. Training a small classifier from an LLM's labels, a form of distillation on generated data ([[data-labeling-and-synthetic-data]]). Ensembles distilled into one model.
- Interviews: "why use temperature", "why multiply by T²", "distillation vs quantization vs pruning".
- Not for: when you have no good teacher, or the student is already as large as the teacher. Not a fix for a student architecture that cannot represent the task at all.

## Setup & code
NumPy only. Runs in about 20 seconds (16 training runs of full-batch gradient descent).

```python
import numpy as np

rng = np.random.default_rng(0)


def spirals(n, classes=3, noise=0.4):
    t = rng.uniform(0.2, 1, n)
    c = rng.integers(0, classes, n)
    ang = 4 * t + 2 * np.pi * c / classes + rng.normal(0, noise, n)
    return np.c_[t * np.cos(ang), t * np.sin(ang)], c


def softmax(z, T=1.0):
    z = z / T
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def init(h):
    return [rng.normal(0, 1, (2, h)), np.zeros(h), rng.normal(0, 1 / np.sqrt(h), (h, 3)), np.zeros(3)]


def logits(p, X):
    a = np.tanh(X @ p[0] + p[1])
    return a, a @ p[2] + p[3]


def train(p, X, targets, T=1.0, alpha=0.0, hard=None, steps=3000, lr=0.5):
    """Minimize (1-alpha) * CE(hard labels) + alpha * T^2 * CE(soft targets at temperature T)."""
    Y = np.eye(3)[hard] if hard is not None else 0
    for _ in range(steps):
        a, z = logits(p, X)
        g = alpha * T * (softmax(z, T) - targets) if alpha else 0   # T^2 * dCE_soft/dz = T (p_T - q_T)
        if alpha < 1:
            g = g + (1 - alpha) * (softmax(z) - Y)
        g /= len(X)
        da = (g @ p[2].T) * (1 - a ** 2)
        for i, grad in enumerate([X.T @ da, da.sum(0), a.T @ g, g.sum(0)]):
            p[i] -= lr * grad
    return p


def acc(p, X, y):
    return np.mean(logits(p, X)[1].argmax(1) == y)


X_big, y_big = spirals(3000)
X_small, y_small = spirals(60)
X_test, y_test = spirals(5000)

teacher = train(init(128), X_big, None, alpha=0.0, hard=y_big, steps=2000)
print(f"teacher (128 hidden, 3000 labels)        test acc {acc(teacher, X_test, y_test):.3f}")

T = 4.0
soft = softmax(logits(teacher, X_small)[1], T)
print("teacher soft targets at T=4 for one point:", np.round(soft[0], 3), "label", y_small[0])

for name, kw in [("hard labels only", dict(alpha=0.0)),
                 ("distilled, alpha=0.7", dict(alpha=0.7, T=T)),
                 ("distilled, alpha=1.0", dict(alpha=1.0, T=T))]:
    s = []
    for seed in range(5):
        rng = np.random.default_rng(seed + 1)
        st = train(init(8), X_small, soft, hard=y_small, **kw)
        s.append(acc(st, X_test, y_test))
    print(f"student (8 hidden, 60 pts) {name:21s} test acc {np.mean(s):.3f} +/- {np.std(s):.3f}")
```

Output (Python 3.14, NumPy 2.5):
```
teacher (128 hidden, 3000 labels)        test acc 0.989
teacher soft targets at T=4 for one point: [0.123 0.757 0.121] label 1
student (8 hidden, 60 pts) hard labels only      test acc 0.921 +/- 0.008
student (8 hidden, 60 pts) distilled, alpha=0.7  test acc 0.967 +/- 0.002
student (8 hidden, 60 pts) distilled, alpha=1.0  test acc 0.964 +/- 0.002
```

With 60 hard labels the student fits a jagged boundary and gets 92%. The teacher's soft targets on the same 60 points tell it how close each point is to the other arms of the spiral, which closes most of the gap to the teacher and also cuts the seed-to-seed spread by 4×. Pure distillation (`α = 1`, no labels used) is almost as good as the mix, because this teacher is very accurate. With a weaker teacher, keeping some hard-label weight matters more.

## Choosing / trade-offs
- **Temperature.** `T` between 2 and 10 is the usual range. Too low and the soft targets are nearly one-hot; too high and they become uniform and carry no class signal. Tune it on a validation set.
- **α.** Start around 0.5-0.9. Lower it when the teacher is weak or the labels are much better than the teacher.
- **Logit vs feature distillation.** Matching output probabilities (here) is architecture-agnostic. Matching hidden states or attention maps (FitNets, TinyBERT) gives more signal but needs a mapping between teacher and student layers.
- **LLM distillation.** Sequence-level distillation trains on teacher-generated text, which is simple and works through an API. Token-level distillation matches the teacher's full next-token distribution and needs logit access. Check the teacher model's license and terms before training on its outputs.
- **Distillation vs quantization vs pruning.** Quantization ([[int8-quantization-from-scratch-numpy]]) keeps the architecture and lowers precision; it is cheap and the first thing to try. Distillation changes the architecture and needs training, but can give larger speedups. They combine well.

## Gotchas
- Forgetting the `T²` factor makes the soft term vanish as `T` grows, and the run silently becomes plain hard-label training.
- Use the same temperature for the teacher's targets and the student's soft-term logits, and `T = 1` for the student at inference.
- Compute teacher outputs in eval mode (no dropout, frozen batch-norm statistics) and cache them. Recomputing them every step wastes most of the training time.
- A student far smaller than the teacher may fail to learn from it at all (the "capacity gap"). An intermediate teacher-assistant model can help.
- The student inherits the teacher's mistakes and biases. Evaluate it against ground truth, not against agreement with the teacher.
- Gains are largest when labeled data is scarce or noisy, as in this demo. With abundant clean labels the improvement is usually a few points at most.

## Related
- [[knowledge-distillation-and-compression]] - the broader overview: pruning, distillation variants, model compression.
- [[neural-network-from-scratch-numpy]] - the MLP and backprop this builds on.
- [[int8-quantization-from-scratch-numpy]] - the cheaper compression step to try first.
- [[loss-functions]] - cross-entropy and KL divergence, the terms in the distillation loss.
- [[probability-calibration]] - temperature scaling, the same knob used for calibration.

## References
- Hinton, Vinyals and Dean (2015), "Distilling the Knowledge in a Neural Network": https://arxiv.org/abs/1503.02531
- Sanh et al. (2019), "DistilBERT, a distilled version of BERT": https://arxiv.org/abs/1910.01108
