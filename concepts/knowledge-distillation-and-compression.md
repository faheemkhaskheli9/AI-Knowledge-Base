---
title: Knowledge distillation and model compression
category: concepts
tags: [distillation, knowledge-distillation, teacher-student, pruning, compression, small-models, soft-labels]
use_cases:
  - "shrink a large model into a smaller, faster one for production or edge devices"
  - "train a small task model using a large LLM as the labeller/teacher"
  - "cut inference cost of a fine-tuned transformer without losing much accuracy"
  - "choose between distillation, pruning and quantization for a latency budget"
status: draft
last_verified: 2026-10-03
sources:
  - https://arxiv.org/abs/1503.02531
  - https://arxiv.org/abs/1910.01108
  - https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html
---

# Knowledge distillation and model compression

## Summary
Knowledge distillation trains a small "student" model to imitate a large
"teacher", usually by matching the teacher's softened output probabilities
rather than only the hard labels. Together with pruning (removing weights)
and quantization (lower-precision weights), it is the main way to make a model
cheaper and faster to serve. Its biggest practical win is letting a strong
teacher label large amounts of unlabelled data for a small model.

## Key concepts
- Soft targets: teacher probabilities carry "dark knowledge" (which wrong
  classes are similar). A temperature T > 1 softens them.
- Loss: KL(student/T || teacher/T) x T^2, often mixed with ordinary
  cross-entropy on true labels (weight alpha).
- Transfer set: inputs the teacher labels for the student; can be unlabelled
  in-domain data or synthetic data. More of it is usually the main lever.
- Variants: logit distillation (Hinton), feature/hidden-state matching
  (DistilBERT, TinyBERT), self-distillation, and sequence-level distillation
  for LLMs (train on the teacher's generated outputs).
- LLM distillation in practice: generate responses or rationales with a large
  model, then fine-tune a small model on them ([[fine-tuning-and-peft]]).
  Check the teacher's licence and terms of use first.
- Pruning: unstructured (zero individual weights, needs sparse kernels to be
  faster) vs structured (drop heads, channels, layers; speeds up on any hardware).
- Quantization: fewer bits per weight; usually the first and cheapest step
  ([[quantization]]). The three combine.

## When to use / scenarios
- Edge/mobile: a small CNN or transformer distilled from a big one
  ([[edge-on-device]]).
- High-volume classification: replace per-call LLM classification with a
  small model trained on LLM labels ([[nlp-classic-tasks]]).
- Search/ranking: distil a cross-encoder reranker into a bi-encoder or smaller reranker.
- Latency-bound serving where quantization alone is not enough.
- NOT for: when quantizing or switching to an existing small pretrained model
  already meets the budget - try those first; and a student cannot learn what
  its capacity or the transfer set does not cover.

## Setup & code
```bash
pip install torch scikit-learn
```
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
X, y = load_digits(return_X_y=True)
X, y = torch.tensor(X / 16.0, dtype=torch.float32), torch.tensor(y)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
X_few, y_few = X_tr[:100], y_tr[:100]   # the student only gets 100 labels

def mlp(hidden):
    return nn.Sequential(nn.Linear(64, hidden), nn.ReLU(), nn.Linear(hidden, 10))

def train(model, inputs, loss_fn, epochs=300):
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(model(inputs)).backward()
        opt.step()
    return (model(X_te).argmax(1) == y_te).float().mean().item()

teacher = mlp(512)                       # stands in for a big pretrained model
acc = train(teacher, X_tr, lambda out: F.cross_entropy(out, y_tr))
print(f"teacher          {acc:.3f}")
with torch.no_grad():
    t_logits = teacher.eval()(X_tr)      # teacher labels the whole (unlabelled) transfer set

T = 4.0
def kd_loss(s_logits):
    return F.kl_div(F.log_softmax(s_logits / T, 1), F.softmax(t_logits / T, 1),
                    reduction="batchmean") * T * T   # T^2 keeps the gradient scale

for seed in range(3):
    torch.manual_seed(seed)
    alone = train(mlp(16), X_few, lambda out: F.cross_entropy(out, y_few))
    torch.manual_seed(seed)
    kd = train(mlp(16), X_tr, kd_loss)
    print(f"seed {seed}: student on 100 labels {alone:.3f}   distilled {kd:.3f}")
```
Output (torch 2.13 CPU): teacher 0.972; the 16-unit student reaches
0.84-0.86 on its 100 labels and 0.96-0.97 when distilled from the teacher over
the full transfer set - near teacher accuracy at ~3% of the parameters.

For transformers, the same loss drops into a Hugging Face `Trainer` subclass
that overrides `compute_loss` and runs the frozen teacher on each batch.

## Choosing / trade-offs
- Order of attack: quantize -> pick an existing smaller model -> distil ->
  prune. Each step costs more engineering than the last.
- Distillation needs a training pipeline and a transfer set; quantization
  needs neither.
- Student size: DistilBERT halves BERT-base's layers, is 40% smaller and 60%
  faster, and keeps about 97% of its language-understanding score (per the paper).
- Structured pruning gives real speed-ups; unstructured sparsity mostly saves
  memory unless the runtime has sparse kernels.

## Gotchas
- On a small, fully labelled dataset, plain logit distillation often does not
  beat training the student directly: an overfit teacher's training-set
  logits are near one-hot and carry little extra signal. In a check on
  scikit-learn digits it scored the same or slightly lower across seeds.
  Distillation pays off with more unlabelled data or a teacher that was
  pretrained on far more data than you have.
- Forgetting the T^2 factor makes the soft loss vanish as T grows.
- Teacher errors are copied; filter low-confidence teacher outputs.
- Evaluate the student on real held-out labels, never on teacher agreement alone.
- Using a commercial LLM's outputs to train a competing model may be barred
  by its terms; check [[model-licenses]].
- Pruned or distilled models can lose accuracy on rare slices that the
  average metric hides; check per-segment.

## Related
- [[quantization]] - the other main compression lever.
- [[fine-tuning-and-peft]] - training the student on teacher outputs.
- [[small-language-models]] - many are distilled from larger siblings.
- [[edge-on-device]] - deployment target for compressed models.
- [[data-labeling-and-synthetic-data]] - LLM-generated transfer sets.
- [[neural-network-fundamentals]] - softmax, cross-entropy, training loop basics.

## References
- Hinton, Vinyals, Dean, Distilling the Knowledge in a Neural Network: https://arxiv.org/abs/1503.02531
- Sanh et al., DistilBERT: https://arxiv.org/abs/1910.01108
- PyTorch pruning tutorial: https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html
