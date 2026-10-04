---
title: LoRA from scratch in NumPy (low-rank adapters, rank choice, merging)
category: concepts
tags: [lora, low-rank-adaptation, peft, fine-tuning, adapters, low-rank, parameter-efficient, numpy, from-scratch, deep-learning-basics]
use_cases:
  - "implement LoRA (low-rank adaptation) on a frozen layer in NumPy without PEFT"
  - "see why a rank-r adapter with 6% of the parameters can beat full fine-tuning on little data"
  - "pick a LoRA rank and alpha and understand the B = 0 initialisation"
  - "explain LoRA, adapter merging and why it adds no inference latency in an interview"
status: draft
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/2106.09685
  - https://huggingface.co/docs/peft/main/en/conceptual_guides/lora
  - https://arxiv.org/abs/2305.14314
---

# LoRA from scratch in NumPy (low-rank adapters, rank choice, merging)

## Summary
LoRA (Low-Rank Adaptation) fine-tunes a pretrained model without touching its weights. It freezes each adapted matrix `W` and learns a low-rank update, `W' = W + (α/r)·B·A`, where `B` is `d_out × r` and `A` is `r × d_in`, with `r` much smaller than either dimension. Below, a pretrained 64×64 tanh layer is adapted to a new task whose true weight change has rank 2, using only 100 labelled examples. Full fine-tuning trains 4,096 parameters, memorises the training set (train MSE 0.0000) and reaches a test MSE of 0.069. LoRA rank 2 trains 256 parameters (6%) and reaches 0.0098, seven times lower. Rank 1 is too small (0.074), and ranks 4 and 16 have spare capacity that fits the label noise (0.034 to 0.048). The low-rank constraint helps as a regulariser, not only as a memory saving, when the real update is low-rank and data is scarce. After training, `B·A` folds back into `W`, so inference costs nothing extra.

## Key concepts
- **Frozen base, trainable delta.** Only `A` and `B` get gradients and optimiser state. That saves most of the memory of fine-tuning: no gradients or Adam moments for the base weights.
- **Parameter count.** `r·(d_in + d_out)` instead of `d_in·d_out`. For a 4096×4096 attention projection at `r = 16`, that is 131k instead of 16.8M (0.8%).
- **Initialisation.** `B = 0` and `A` random, so `B·A = 0` and training starts exactly at the pretrained model. If both were random, step 0 would already be a damaged model.
- **Scaling α/r.** The update is multiplied by `α/r`, so changing `r` does not force a new learning rate. Common defaults are `α = r` or `α = 2r`.
- **Gradients.** With `G = ∂L/∂W'` (the ordinary weight gradient), `∂L/∂B = (α/r)·G·Aᵀ` and `∂L/∂A = (α/r)·Bᵀ·G`. LoRA is a reparameterisation; backprop through the layer is unchanged.
- **Merging.** `W_merged = W + (α/r)·B·A` is an ordinary dense matrix: same shape, same latency as the base model. Keep the adapter separate instead to swap tasks on one base model.
- **Intrinsic rank.** The method assumes the weight change a downstream task needs is low-rank. The demo builds that in; on real LLMs, small ranks (8 to 64) work for most instruction and style tuning.

## When to use / scenarios
- Learning: what PEFT libraries do inside `LoraConfig`, in about 30 lines.
- Interviews: "why initialise B to zero", "what does alpha do", "does LoRA slow down inference", "LoRA vs full fine-tune vs QLoRA", "which matrices to adapt".
- Fine-tuning large models on one GPU, keeping one base model with many small task adapters, or small-data adaptation where full fine-tuning overfits.
- Not for: teaching a model substantial new knowledge or a new language (full fine-tuning or continued pretraining does better there), or production training code (use Hugging Face PEFT, see [[fine-tuning-and-peft]]).

## Setup & code
`pip install numpy`. Runs in about 2 seconds on CPU.

```python
import numpy as np

D_IN, D_H, D_OUT = 64, 64, 8


def forward(X, W1, W2):
    H = np.tanh(X @ W1.T)
    return H, H @ W2.T


def finetune(X, Y, W1, W2, rank=None, alpha=8.0, steps=300, lr=1e-2, seed=0):
    """Adapt W1 to new data. rank=None: full fine-tune of W1. rank=r: LoRA, W1 frozen, learn B @ A."""
    rng = np.random.default_rng(seed)
    if rank is None:
        params = [W1.copy()]
    else:
        params = [np.zeros((D_H, rank)), rng.normal(0, 1 / np.sqrt(D_IN), (rank, D_IN))]  # B = 0: start at the pretrained model
    scale = 1.0 if rank is None else alpha / rank
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]

    def eff(ps):
        return ps[0] if rank is None else W1 + scale * ps[0] @ ps[1]

    for t in range(1, steps + 1):
        W = eff(params)
        H, P = forward(X, W, W2)
        dP = 2 * (P - Y) / len(X)
        dZ = (dP @ W2) * (1 - H**2)
        G = dZ.T @ X                                    # dLoss/dW_eff
        grads = [G] if rank is None else [scale * G @ params[1].T, scale * params[0].T @ G]
        for i, g in enumerate(grads):                   # Adam
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g**2
            params[i] -= lr * (m[i] / (1 - 0.9**t)) / (np.sqrt(v[i] / (1 - 0.999**t)) + 1e-8)
    return eff(params), sum(p.size for p in params)     # eff() is the merged weight: W1 + scale * B @ A


rng = np.random.default_rng(0)
# "pretrained" network = the task-A teacher; task B shifts W1 by a rank-2 update
W1_pre = rng.normal(0, 1 / np.sqrt(D_IN), (D_H, D_IN))
W2 = rng.normal(0, 1 / np.sqrt(D_H), (D_OUT, D_H))
delta = 0.5 * rng.normal(0, 1, (D_H, 2)) @ rng.normal(0, 1 / np.sqrt(D_IN), (2, D_IN))
W1_task = W1_pre + delta

X_tr = rng.normal(0, 1, (100, D_IN))                    # only 100 labelled examples for task B
Y_tr = forward(X_tr, W1_task, W2)[1] + rng.normal(0, 0.1, (100, D_OUT))
X_te = rng.normal(0, 1, (2000, D_IN))
Y_te = forward(X_te, W1_task, W2)[1]

mse = lambda W: np.mean((forward(X_te, W, W2)[1] - Y_te) ** 2)
tr_mse = lambda W: np.mean((forward(X_tr, W, W2)[1] - Y_tr) ** 2)
print(f"pretrained, no fine-tune: test MSE {mse(W1_pre):.4f}")
for rank in [None, 1, 2, 4, 16]:
    res = [finetune(X_tr, Y_tr, W1_pre, W2, rank=rank, seed=s) for s in range(3)]
    n = res[0][1]
    name = "full fine-tune" if rank is None else f"LoRA rank {rank}"
    print(f"{name:15s}: trainable params {n:5d} ({n / W1_pre.size:6.1%} of W1) | train MSE {np.mean([tr_mse(W) for W, _ in res]):.4f} "
          f"| test MSE per seed {[round(float(mse(W)), 4) for W, _ in res]}")
```

Output (numpy 2.5):
```
pretrained, no fine-tune: test MSE 0.1126
full fine-tune : trainable params  4096 (100.0% of W1) | train MSE 0.0000 | test MSE per seed [0.0687, 0.0687, 0.0687]
LoRA rank 1    : trainable params   128 (  3.1% of W1) | train MSE 0.0435 | test MSE per seed [0.074, 0.0764, 0.0839]
LoRA rank 2    : trainable params   256 (  6.2% of W1) | train MSE 0.0064 | test MSE per seed [0.0098, 0.0098, 0.0098]
LoRA rank 4    : trainable params   512 ( 12.5% of W1) | train MSE 0.0028 | test MSE per seed [0.0401, 0.0341, 0.0433]
LoRA rank 16   : trainable params  2048 ( 50.0% of W1) | train MSE 0.0000 | test MSE per seed [0.0432, 0.043, 0.0478]
```

The pretrained layer scores 0.113 on the new task. Full fine-tuning has 4,096 free parameters and 800 noisy targets, so it fits the noise (train MSE 0.0000, below the 0.01 noise variance) and improves the test score only to 0.069. LoRA rank 2 matches the true rank of the change: it cannot represent the noise, stops at a train MSE near the noise level (0.0064) and reaches 0.0098 on every seed. Rank 1 underfits, since it cannot express a rank-2 change. Ranks 4 and 16 have spare directions that absorb noise, so their train error falls below the noise level and their test error rises to 0.03 to 0.05. That is still better than full fine-tuning, but not by much. On real tasks the true rank is unknown, so rank is a hyperparameter: sweep a few values on a validation set. Full fine-tuning gives the same result on each seed because it has no random initialisation; LoRA's random `A` makes runs differ.

## Choosing / trade-offs
- **Rank.** Start at 8 or 16 for LLM instruction tuning. Increase it if training loss plateaus well above full fine-tuning; decrease it, or add dropout on the adapter input, if validation loss rises while training loss keeps falling.
- **Alpha.** It acts as a learning-rate multiplier on the adapter. Keep `α/r` fixed when sweeping `r`. rsLoRA scales by `α/√r` instead, which keeps large ranks stable.
- **Which matrices.** The original paper adapted only the attention `W_q` and `W_v`. Current practice usually targets all linear layers (attention and MLP) for better quality at a few times the adapter size.
- **QLoRA.** Store the frozen base in 4-bit NF4 and train LoRA adapters in bf16 on top. A 65B model then fine-tunes on one 48 GB GPU at close to 16-bit quality ([[quantization]]).
- **Variants.** DoRA splits magnitude and direction, LoRA+ uses a larger learning rate for `B`, and PiSSA initialises from the top singular vectors of `W`. They give small gains over a well-tuned plain LoRA.
- **Merge or keep separate.** Merge for single-task serving (zero overhead). Keep adapters separate to serve many tasks from one base in one batch (S-LoRA, vLLM multi-LoRA).

## Gotchas
- Initialising both `A` and `B` randomly starts training from a corrupted model, and the first steps spend effort undoing that.
- With `B = 0`, the gradient of `A` is zero at step 0 (`Bᵀ·G = 0`); only `B` moves at first. That is expected, not a bug.
- Merging into a quantized base then re-quantizing loses some of the adapter's effect. Merge into a 16-bit copy of the base when you need exact results.
- A higher rank is not automatically better: above the needed rank it overfits small datasets (ranks 4 and 16 above).
- LoRA usually wants a higher learning rate than full fine-tuning (commonly 1e-4 to 2e-4 for LLMs, against around 1e-5 for full fine-tuning). Reusing a full fine-tuning rate makes LoRA look weak.
- Fine-tuning still updates layer norms and the head if you leave them trainable. Count trainable parameters (`model.print_trainable_parameters()` in PEFT) to confirm only the adapters train.

## Related
- [[fine-tuning-and-peft]] - LoRA, QLoRA and other PEFT methods with Hugging Face PEFT and TRL.
- [[neural-network-from-scratch-numpy]] - the dense-layer backprop that LoRA reparameterises.
- [[optimizers-from-scratch-numpy]] - the Adam update used here.
- [[pca-from-scratch]] - low-rank structure via SVD, the same idea applied to data instead of weight updates.
- [[matrix-factorization-from-scratch]] - learning a low-rank product `B·A` by gradient descent in a recommender.
- [[quantization]] - the 4-bit base model in QLoRA.

## References
- Hu et al. (2021), "LoRA: Low-Rank Adaptation of Large Language Models": https://arxiv.org/abs/2106.09685
- Hugging Face PEFT documentation, LoRA conceptual guide: https://huggingface.co/docs/peft/main/en/conceptual_guides/lora
- Dettmers et al. (2023), "QLoRA: Efficient Finetuning of Quantized LLMs": https://arxiv.org/abs/2305.14314
