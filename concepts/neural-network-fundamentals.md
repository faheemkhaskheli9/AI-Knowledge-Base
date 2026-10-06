---
title: Neural network fundamentals
category: concepts
tags: [deep-learning, neural-network, mlp, backpropagation, gradient-descent, activation, loss-function, pytorch]
use_cases:
  - "understand how a neural network learns before training one on my own data"
  - "choose the output layer and loss for classification, multi-label or regression"
  - "explain backpropagation and gradient descent to a team new to deep learning"
  - "build a small MLP from scratch to check that shapes, loss and gradients behave"
status: draft
last_verified: 2026-10-03
sources:
  - https://www.deeplearningbook.org/
  - https://d2l.ai/
  - https://pytorch.org/tutorials/beginner/basics/intro.html
  - https://pytorch.org/docs/stable/nn.html#loss-functions
---

# Neural network fundamentals

## Summary
A neural network is a stack of layers, each a linear transform followed by a non-linear activation, trained end to end by gradient descent: compute a loss on a batch, backpropagate to get the gradient of the loss for every weight, and nudge the weights downhill. Stacking layers lets the network learn its own features from raw inputs, which is why deep learning dominates images, audio and text. This file covers the core mechanics; [[deep-learning-training]] covers making training work, and [[cnn-and-rnn-architectures]] / [[transformers-and-attention]] cover the main architectures.

## Key concepts
- **Neuron / layer.** A dense (fully connected, `nn.Linear`) layer computes `y = xW + b`. Stacking dense layers with activations gives a multi-layer perceptron (MLP).
- **Activations.** ReLU (default for hidden layers), GELU/SiLU (transformers, modern CNNs), sigmoid and tanh (gates, outputs). Without a non-linearity, any stack of layers collapses to one linear map.
- **Universal approximation.** A wide enough MLP can approximate any continuous function; depth makes it far more parameter-efficient in practice.
- **Forward pass** produces predictions; the **loss** compares them to targets.
- **Output layer + loss pairs:**
  - Regression: linear output + MSE (`nn.MSELoss`) or Huber/L1 for outliers.
  - Binary or multi-label: one logit per label + `nn.BCEWithLogitsLoss` (sigmoid included).
  - Multi-class (one label): one logit per class + `nn.CrossEntropyLoss` (softmax included).
- **Backpropagation.** The chain rule applied from the loss back through every layer gives `∂loss/∂w` for all weights in one backward pass. Frameworks do it automatically (autograd).
- **Gradient descent.** `w ← w − lr · ∂loss/∂w`. Mini-batch SGD uses a batch of samples per step; an **epoch** is one pass over the training set. Adam/AdamW adapt the step per parameter (see [[deep-learning-training]]).
- **Initialisation.** Weights start small and random (He/Kaiming for ReLU, Xavier/Glorot for tanh) so signals neither explode nor vanish; PyTorch layers do this by default.
- **Vanishing / exploding gradients.** Deep stacks multiply many derivatives; fixes are ReLU-family activations, normalisation layers, residual connections and gradient clipping.
- **Representation learning.** Early layers learn generic features (edges, sub-words), later layers task-specific ones, which is what makes transfer learning and [[embeddings]] work.

## When to use / scenarios
- Unstructured inputs: images ([[image-classification]], [[object-detection]]), audio ([[speech-to-text]]), text ([[nlp-classic-tasks]]), and combinations ([[multimodal-models]]).
- Very large datasets where learned features beat hand-made ones; sequence and set data; embeddings for search and recommendation.
- Using a pretrained model (fine-tuning or just its embeddings) is usually better than training from scratch on small data.
- NOT the first choice for: small or medium tabular datasets (gradient boosting usually wins, see [[gradient-boosting-tabular]]), problems that need a short audited rule, or when an LLM API already solves the language task.

## Setup & code
```bash
pip install torch   # pick the CUDA/CPU build at https://pytorch.org/get-started/locally/
```
A two-layer MLP on a non-linear problem, showing every step of the loop:
```python
import torch
from torch import nn

torch.manual_seed(0)
# Class depends non-linearly on (x1, x2): no straight line separates it.
X = torch.randn(2000, 2)
y = ((X[:, 0] ** 2 + X[:, 1] ** 2) > 1.4).long()          # inside vs outside a circle
X_tr, y_tr, X_va, y_va = X[:1600], y[:1600], X[1600:], y[1600:]

model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(),
                      nn.Linear(32, 32), nn.ReLU(),
                      nn.Linear(32, 2))                      # 2 logits -> CrossEntropy
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

for epoch in range(200):
    model.train()
    for i in range(0, len(X_tr), 64):                       # mini-batches
        xb, yb = X_tr[i:i + 64], y_tr[i:i + 64]
        loss = loss_fn(model(xb), yb)                         # forward + loss
        opt.zero_grad()
        loss.backward()                                       # backprop
        opt.step()                                            # gradient step
    if epoch % 50 == 0:
        model.eval()
        with torch.no_grad():
            acc = (model(X_va).argmax(1) == y_va).float().mean()
        print(f"epoch {epoch} loss {loss.item():.3f} val_acc {acc:.3f}")
```
Swap `nn.Sequential(nn.Linear(2, 2))` in to see a linear model fail on the same data (val accuracy stays near the majority rate).

## Choosing / trade-offs
- **Width vs depth.** Deeper nets learn hierarchical features with fewer parameters but are harder to optimise without residual connections and normalisation.
- **From scratch vs pretrained.** From scratch needs lots of data and compute; fine-tuning a pretrained backbone works with hundreds to thousands of labels (see [[fine-tuning-and-peft]], [[pytorch-basics]]).
- **Model size vs latency.** Bigger models are more accurate and slower; distillation, pruning and [[quantization]] trade a little accuracy back for speed.
- **Framework.** PyTorch is the default for research and most open models; JAX for some large-scale research; Keras 3 for a high-level API over several backends.

## Gotchas
- Applying `softmax`/`sigmoid` before `CrossEntropyLoss`/`BCEWithLogitsLoss` double-applies it and trains badly; those losses expect raw logits.
- Unscaled inputs (one feature in thousands, another in 0-1) slow or stall training; standardise inputs.
- Forgetting `optimizer.zero_grad()` accumulates gradients across steps.
- Forgetting `model.eval()` at validation keeps dropout on and batch-norm statistics updating.
- A network that cannot overfit a single small batch has a bug (wrong loss, labels, shapes or learning rate), not a data problem.
- Loss going to NaN usually means too high a learning rate, a log of zero, or fp16 overflow.

## Related
- [[ml-fundamentals]] - generalisation, overfitting and splits apply unchanged.
- [[deep-learning-training]] - optimisers, learning-rate schedules, regularisation and normalisation.
- [[cnn-and-rnn-architectures]] - layers specialised for images and sequences.
- [[transformers-and-attention]] - the architecture behind LLMs.
- [[pytorch-basics]] - the production training loop (DataLoader, GPU, AMP, checkpoints).
- [[universal-approximation-and-depth-vs-width]] - what one hidden layer can represent and why depth still helps.

## References
- Goodfellow, Bengio, Courville, *Deep Learning* (free online): https://www.deeplearningbook.org/
- *Dive into Deep Learning* (interactive, PyTorch code): https://d2l.ai/
- PyTorch basics tutorial: https://pytorch.org/tutorials/beginner/basics/intro.html
- PyTorch loss functions: https://pytorch.org/docs/stable/nn.html#loss-functions
