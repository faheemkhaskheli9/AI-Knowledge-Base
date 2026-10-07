---
title: Information theory for ML (entropy, cross-entropy, KL divergence, mutual information)
category: concepts
tags: [information-theory, entropy, cross-entropy, kl-divergence, mutual-information, perplexity, bits, nats, log-loss, scipy, scikit-learn]
use_cases:
  - "understand why classification and language models are trained with cross-entropy"
  - "read perplexity, bits-per-character or log loss numbers and compare them"
  - "know what the KL term in a VAE, distillation or RLHF objective is doing"
  - "measure non-linear dependence between a feature and a target with mutual information"
  - "pick the direction of KL (forward vs reverse) for a fitting or distillation objective"
status: stable
last_verified: 2026-10-04
sources:
  - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html
  - https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html
  - https://www.inference.org.uk/itprnn/book.pdf
  - https://www.deeplearningbook.org/contents/prob.html
---

# Information theory for ML (entropy, cross-entropy, KL divergence, mutual information)

## Summary
Four quantities from information theory sit under most of ML. **Entropy** `H(p)` is the average surprise of a distribution. **Cross-entropy** `H(p, q)` is the average surprise when reality follows `p` but you predict with `q`; it is exactly the log loss used to train classifiers and language models. **KL divergence** `KL(p||q) = H(p,q) - H(p)` is the extra cost of using `q` instead of `p`; it appears in VAEs, distillation, RLHF and variational inference. **Mutual information** `I(X;Y)` measures how much knowing one variable reduces uncertainty about another, including non-linear dependence that correlation misses.

## Key concepts
- **Surprise.** An event with probability `p` carries `-log p` units of information. Base 2 gives **bits**, natural log gives **nats** (frameworks use nats; divide by `ln 2` for bits).
- **Entropy** `H(p) = -Σ p log p`. Maximal for a uniform distribution (`log K` for K classes), zero for a certain outcome. Decision trees split to reduce it (information gain) ([[decision-trees-and-random-forests]]).
- **Cross-entropy** `H(p, q) = -Σ p log q`. With one-hot labels it reduces to `-log q(true class)`: the negative log-likelihood. Minimising it = maximum likelihood ([[loss-functions]]).
- **KL divergence** `KL(p||q) = Σ p log(p/q) >= 0`, zero only when `p = q`. Not symmetric and not a distance. Since `H(p)` is fixed by the data, minimising cross-entropy is minimising KL to the data.
- **Forward vs reverse KL.** `KL(p||q)` (forward, what maximum likelihood minimises) is mass-covering: `q` spreads to cover every mode of `p`. `KL(q||p)` (reverse, used in variational inference and the RLHF penalty) is mode-seeking: `q` locks onto one mode.
- **Perplexity** `= exp(cross-entropy in nats)`: the effective number of equally likely choices the model is unsure between. Uniform over K classes has perplexity K ([[pretraining-and-scaling-laws]]).
- **Mutual information** `I(X;Y) = H(Y) - H(Y|X) = KL(p(x,y) || p(x)p(y))`. Zero iff independent; captures any dependence. Estimating it for continuous variables needs nearest-neighbour estimators (what scikit-learn uses).
- **Where they show up.** Softmax + cross-entropy training; label smoothing (mix target with uniform); knowledge distillation (KL between teacher and student softmax) ([[knowledge-distillation-and-compression]]); VAE ELBO (reconstruction + KL to prior) ([[autoencoders-and-self-supervised-learning]]); RLHF/DPO (KL penalty to the reference model) ([[rlhf-and-preference-optimization]]); InfoNCE contrastive loss (lower bound on MI); feature selection by MI ([[feature-selection]]).

## When to use / scenarios
- Interpreting a training curve: a 3-class classifier stuck at loss `ln 3 ≈ 1.0986` is guessing uniformly ([[debugging-neural-network-training]]).
- Comparing language models: perplexity or bits-per-byte, only on the same tokenizer and text ([[tokenization]]).
- Writing a custom loss that mixes a likelihood with a "stay close to this distribution" term.
- Screening features for non-linear relationships with the target before modelling.
- Drift monitoring: KL or Jensen-Shannon divergence between training and live feature histograms ([[online-learning-and-concept-drift]]).
- NOT a model by itself; these are measuring tools and loss ingredients.

## Setup & code
```bash
pip install numpy scipy scikit-learn
```
```python
import numpy as np
from scipy.stats import entropy
from scipy.special import rel_entr, log_softmax
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import log_loss, mutual_info_score

p = np.array([0.5, 0.25, 0.25])          # true distribution
q = np.array([0.8, 0.1, 0.1])            # model distribution

H_p = entropy(p, base=2)
CE = -(p * np.log2(q)).sum()
KL = entropy(p, q, base=2)               # scipy: entropy(pk, qk) is KL(p || q)
print("H(p) = %.3f bits | CE(p,q) = %.3f | KL(p||q) = %.3f | H+KL = %.3f" % (H_p, CE, KL, H_p + KL))
print("KL(q||p) = %.3f  (not symmetric)" % entropy(q, p, base=2))
print("rel_entr sum (nats) = %.4f == KL nats %.4f" % (rel_entr(p, q).sum(), entropy(p, q)))

# cross-entropy loss == log loss == mean negative log-likelihood (nats); perplexity = exp(CE)
y = np.array([0, 2, 1, 0])
logits = np.array([[2.0, 0.5, 0.1], [0.2, 0.3, 1.5], [0.1, 2.2, 0.3], [1.0, 1.0, 1.0]])
nll = -log_softmax(logits, axis=1)[np.arange(4), y].mean()
print("manual CE %.4f | sklearn log_loss %.4f | perplexity %.3f" % (
    nll, log_loss(y, np.exp(log_softmax(logits, axis=1)), labels=[0, 1, 2]), np.exp(nll)))
print("uniform guess over 3 classes: CE %.4f = ln 3" % np.log(3))

# mutual information catches a non-linear dependence that correlation misses
rng = np.random.default_rng(0)
x_signal = rng.uniform(-1, 1, 2000)
x_noise = rng.uniform(-1, 1, 2000)
target = (np.abs(x_signal) > 0.5).astype(int)          # depends on |x|, correlation ~ 0
print("corr(signal, target) %.3f" % np.corrcoef(x_signal, target)[0, 1])
print("MI estimates (nats):", mutual_info_classif(np.column_stack([x_signal, x_noise]), target, random_state=0).round(3))
print("MI of discrete labels with themselves %.4f = H(target) %.4f" % (
    mutual_info_score(target, target), entropy(np.bincount(target))))
```
Output with scipy 1.18.0, scikit-learn 1.9.0:
```text
H(p) = 1.500 bits | CE(p,q) = 1.822 | KL(p||q) = 0.322 | H+KL = 1.822
KL(q||p) = 0.278  (not symmetric)
rel_entr sum (nats) = 0.2231 == KL nats 0.2231
manual CE 0.5274 | sklearn log_loss 0.5274 | perplexity 1.694
uniform guess over 3 classes: CE 1.0986 = ln 3
corr(signal, target) 0.003
MI estimates (nats): [0.692 0.003]
MI of discrete labels with themselves 0.6929 = H(target) 0.6929
```
Cross-entropy = entropy + KL holds exactly; the hand-written loss matches `log_loss`; correlation sees nothing in `x_signal` while MI finds almost the full target entropy (0.69 nats ≈ 1 bit for a balanced binary target).

In PyTorch: `F.cross_entropy(logits, y)` takes raw logits; `F.kl_div(input, target)` expects `input` as **log**-probabilities and computes `KL(target || exp(input))`.

## Choosing / trade-offs
- **Bits vs nats.** Same thing, different units. Report bits-per-character/byte for compression-style comparisons, nats for raw framework losses; never compare across them unconverted.
- **KL vs Jensen-Shannon vs Wasserstein** for comparing distributions: KL is infinite when `q=0` where `p>0` (needs smoothing); JS is symmetric and bounded (`<= ln 2`); Wasserstein respects distances between bins (good for ordered histograms).
- **Forward vs reverse KL** in an objective: forward when missing a mode is costly (cover everything, accept blur); reverse when producing implausible samples is costly (pick one good mode).
- **MI vs correlation for feature screening.** MI finds any dependence but is a noisy estimate with high variance on small samples and says nothing about direction or shape; correlation is cheap and stable but linear only. Use MI to screen, then look at the relationship.

## Gotchas
- `log(0)`: clip probabilities (`np.clip(q, 1e-12, 1)`) or work in log-space with `log_softmax`; computing `softmax` then `log` underflows.
- `scipy.stats.entropy` normalises its inputs to sum to 1 silently; passing counts is fine, passing unnormalised scores you did not mean to normalise is a bug.
- Perplexity numbers from different tokenizers are not comparable: a model with a bigger vocabulary predicts fewer, harder tokens. Use bits-per-byte across tokenizers.
- `F.kl_div` argument order and log-space input are the classic distillation bug; with `reduction="mean"` it also averages over classes, so use `reduction="batchmean"` for the true KL.
- MI estimates from `mutual_info_classif` vary with `n_neighbors` and the random seed (noise is added to break ties); set `random_state` and compare features relative to each other, not to an absolute threshold.
- Label-smoothed or soft targets mean the minimum achievable cross-entropy is their entropy, not 0; a loss that "won't go below 0.5" may already be optimal.

## Related
- [[math-for-machine-learning]] - the probability and log-space background.
- [[loss-functions]] - cross-entropy, BCE, focal and KL losses in practice.
- [[decision-trees-and-random-forests]] - entropy and information gain as split criteria.
- [[feature-selection]] - mutual-information-based feature scoring.
- [[knowledge-distillation-and-compression]] - KL between teacher and student outputs.
- [[autoencoders-and-self-supervised-learning]] - VAE KL term and contrastive (InfoNCE) objectives.
- [[rlhf-and-preference-optimization]] - KL penalty to a reference policy.
- [[probability-calibration]] - log loss as a proper scoring rule for probabilities.

## References
- `scipy.stats.entropy`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html
- `mutual_info_classif`: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html
- MacKay, Information Theory, Inference, and Learning Algorithms: https://www.inference.org.uk/itprnn/book.pdf
- Goodfellow, Bengio, Courville, Deep Learning, ch. 3 (Probability and Information Theory): https://www.deeplearningbook.org/contents/prob.html
