---
title: Hidden Markov model from scratch (forward-backward, Viterbi, Baum-Welch EM)
category: ml
tags: [hmm, hidden-markov-model, forward-algorithm, forward-backward, viterbi, baum-welch, expectation-maximization, dynamic-programming, sequence-models, numpy, from-scratch, ml-basics]
use_cases:
  - "implement the forward, Viterbi and Baum-Welch algorithms for a discrete HMM in NumPy"
  - "decode hidden regimes (fair/loaded, bull/bear, speech/silence) from an observed sequence"
  - "understand why HMM probabilities need scaling or log space on long sequences"
  - "explain Viterbi vs posterior decoding and EM local optima in an interview"
status: stable
last_verified: 2026-10-04
sources:
  - https://doi.org/10.1109/5.18626
  - https://web.stanford.edu/~jurafsky/slp3/A.pdf
  - https://hmmlearn.readthedocs.io/en/latest/
---

# Hidden Markov model from scratch (forward-backward, Viterbi, Baum-Welch EM)

## Summary
A hidden Markov model (HMM) explains a sequence of observations with a hidden state that follows a Markov chain. Each state has its own distribution over observations. Three dynamic-programming algorithms do all the work. The forward algorithm computes the likelihood of a sequence, Viterbi finds the most likely hidden path, and forward-backward plus Baum-Welch EM learns the parameters from unlabelled sequences. The NumPy version below matches a brute-force sum over all 1,024 paths to 6 decimals. It decodes the "dishonest casino" (fair vs loaded die) and recovers the true transition matrix from a random start, but only when EM starts well.

## Key concepts
- **Parameters.** Initial distribution `π` (K), transition matrix `A[i, j] = P(z_t = j | z_{t−1} = i)` (K×K), emission matrix `B[i, m] = P(x_t = m | z_t = i)` (K×M).
- **Forward variable** `α_t(j) = P(x_1..x_t, z_t = j)`, with recursion `α_t = (α_{t−1} A) ⊙ B[:, x_t]`. Summing `α_T` gives `P(x)` in O(T·K²) instead of O(K^T).
- **Scaling.** `α` shrinks geometrically and underflows past a few hundred steps. Normalise it every step and keep the normalisers `c_t`: `log P(x) = Σ log c_t`.
- **Backward variable** `β_t(i) = P(x_{t+1}..x_T | z_t = i)`. `γ_t ∝ α_t ⊙ β_t` is the posterior over the state at each step.
- **Viterbi.** The same recursion with `max` in place of `sum`, done in log space, plus back-pointers to read out the single best path.
- **Baum-Welch.** EM: the E-step computes `γ` and the pair posteriors `ξ_t(i, j)`; the M-step sets `A` to expected transition counts and `B` to expected emission counts, both normalised. Log-likelihood never decreases, but it converges to a local optimum.
- **Label switching.** States learned by EM come out in arbitrary order. Match them to meaning afterwards (here by which state emits more sixes).

## When to use / scenarios
- Learning: the cleanest example of dynamic programming over a latent sequence, and of EM on something richer than a mixture model ([[gmm-em-from-scratch]]).
- Interviews: "derive the forward algorithm", "Viterbi vs forward", "why scale", "how does Baum-Welch work".
- Regime detection: market regimes, machine health states, user activity modes, CpG islands in DNA, part-of-speech tagging baselines.
- Small, interpretable sequence models where labelled data is scarce and the number of hidden states is known.
- Not for: long-range dependencies (the Markov assumption forgets everything but the last state); rich sequence labelling with features, where CRFs or neural models ([[rnn-from-scratch-numpy]], [[lstm-from-scratch-numpy]]) do better; continuous-state tracking, where a Kalman filter fits ([[hidden-markov-models-and-kalman-filters]]).

## Setup & code
`pip install numpy`. Runs in a few seconds on CPU.

```python
import itertools
import numpy as np


def forward(obs, pi, A, B):
    """Scaled forward pass. Returns alpha (normalised per step), scales c, log p(obs)."""
    T, K = len(obs), len(pi)
    alpha, c = np.zeros((T, K)), np.zeros(T)
    a = pi * B[:, obs[0]]
    for t in range(T):
        if t:
            a = (alpha[t - 1] @ A) * B[:, obs[t]]
        c[t] = a.sum(); alpha[t] = a / c[t]               # rescale so it never underflows
    return alpha, c, np.log(c).sum()


def backward(obs, A, B, c):
    T, K = len(obs), A.shape[0]
    beta = np.ones((T, K))
    for t in range(T - 2, -1, -1):
        beta[t] = A @ (B[:, obs[t + 1]] * beta[t + 1]) / c[t + 1]
    return beta


def viterbi(obs, pi, A, B):
    """Most likely hidden path, in log space."""
    lA, lB = np.log(A), np.log(B)
    d = np.log(pi) + lB[:, obs[0]]
    back = np.zeros((len(obs), len(pi)), dtype=int)
    for t in range(1, len(obs)):
        s = d[:, None] + lA                               # s[i, j]: best path ending i, then i -> j
        back[t] = s.argmax(0); d = s.max(0) + lB[:, obs[t]]
    path = [int(d.argmax())]
    for t in range(len(obs) - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    return np.array(path[::-1])


def baum_welch(obs, K, M, iters, rng, stay=0.5):
    """EM for a discrete HMM from a random start. Returns params and log-likelihood per iteration."""
    pi = np.full(K, 1 / K)
    A = stay * np.eye(K) + (1 - stay) * rng.dirichlet(np.ones(K), K)   # stay>0.5 = "sticky" start
    B = rng.dirichlet(np.ones(M), K)
    hist = []
    for _ in range(iters):
        alpha, c, ll = forward(obs, pi, A, B); hist.append(ll)
        beta = backward(obs, A, B, c)
        gamma = alpha * beta                              # p(state_t | obs)
        xi = (alpha[:-1, :, None] * A[None] * (B[:, obs[1:]].T * beta[1:])[:, None, :]) / c[1:, None, None]
        pi = gamma[0]
        A = xi.sum(0) / gamma[:-1].sum(0)[:, None]
        B = np.stack([gamma[obs == m].sum(0) for m in range(M)], 1) / gamma.sum(0)[:, None]
    return pi, A, B, np.array(hist)


def sample(T, pi, A, B, rng):
    z, x = np.zeros(T, int), np.zeros(T, int)
    z[0] = rng.choice(len(pi), p=pi)
    for t in range(T):
        if t:
            z[t] = rng.choice(len(pi), p=A[z[t - 1]])
        x[t] = rng.choice(B.shape[1], p=B[z[t]])
    return z, x


# "occasionally dishonest casino": state 0 fair die, state 1 loaded die (six half the time)
pi = np.array([0.5, 0.5])
A = np.array([[0.95, 0.05], [0.10, 0.90]])
B = np.array([[1 / 6] * 6, [0.1] * 5 + [0.5]])
rng = np.random.default_rng(0)

# 1) forward algorithm == brute-force sum over all 2^T hidden paths
_, x_short = sample(10, pi, A, B, rng)
brute = sum(pi[z[0]] * B[z[0], x_short[0]] * np.prod([A[z[t - 1], z[t]] * B[z[t], x_short[t]] for t in range(1, 10)])
            for z in itertools.product(range(2), repeat=10))
print(f"log p(x), T=10: forward {forward(x_short, pi, A, B)[2]:.6f} | brute force over 1024 paths {np.log(brute):.6f}")

# 2) decoding: Viterbi path vs per-step posterior argmax, on 2000 rolls
z, x = sample(2000, pi, A, B, rng)
alpha, c, ll = forward(x, pi, A, B)
post = alpha * backward(x, A, B, c)
print(f"log p(x), T=2000: {ll:.1f}  (unscaled product would be ~1e{ll / np.log(10):.0f}, below float64 min 1e-308)")
print(f"state accuracy: Viterbi {(viterbi(x, pi, A, B) == z).mean():.3f} | posterior argmax {(post.argmax(1) == z).mean():.3f} | always-fair {(z == 0).mean():.3f}")

# 3) Baum-Welch: EM only finds a local optimum, so the start matters
for stay in [0.0, 0.9]:
    runs = [baum_welch(x, 2, 6, 200, np.random.default_rng(s), stay) for s in range(5)]
    print(f"start stay={stay}: final log-lik of 5 restarts", [round(float(r[3][-1]), 1) for r in runs])
pi_h, A_h, B_h, hist = max(runs, key=lambda r: r[3][-1])
if B_h[0, 5] > B_h[1, 5]:                                 # put the loaded state second
    A_h, B_h = A_h[::-1, ::-1], B_h[::-1]
print(f"best run: iter 1 {hist[0]:.1f} -> iter 200 {hist[-1]:.1f} | never decreased: {bool(np.all(np.diff(hist) > -1e-8))} | true params {ll:.1f}")
print("learned A:\n", A_h.round(3))
print("learned P(six): fair", B_h[0, 5].round(3), "| loaded", B_h[1, 5].round(3))
```

Output (numpy 2.5):
```
log p(x), T=10: forward -18.321325 | brute force over 1024 paths -18.321325
log p(x), T=2000: -3518.4  (unscaled product would be ~1e-1528, below float64 min 1e-308)
state accuracy: Viterbi 0.787 | posterior argmax 0.817 | always-fair 0.713
start stay=0.0: final log-lik of 5 restarts [-3528.6, -3529.4, -3529.5, -3515.1, -3529.6]
start stay=0.9: final log-lik of 5 restarts [-3512.7, -3512.6, -3512.6, -3512.6, -3529.5]
best run: iter 1 -3766.3 -> iter 200 -3512.6 | never decreased: True | true params -3518.4
learned A:
 [[0.956 0.044]
 [0.088 0.912]]
learned P(six): fair 0.159 | loaded 0.444
```

The scaled forward pass gives the same log-likelihood as summing all 1,024 hidden paths. On 2,000 rolls the unscaled probability would be about 10⁻¹⁵²⁸, far below the smallest float64, which is why scaling (or log space) is required. Viterbi labels 78.7% of rolls correctly and posterior argmax 81.7%, against 71.3% for "always fair". Posterior decoding maximises the expected number of correct states. Viterbi maximises the probability of the whole path, so it scores lower per step but always returns a path the model allows. A loaded run of a few rolls looks like chance, so no decoder gets close to 100%. Baum-Welch from a near-uniform start (`stay=0`) stalls around −3529 in 4 of 5 runs, a solution where the states do not separate fair from loaded. A "sticky" start with 0.9 on the diagonal reaches −3512.6 in 4 of 5 runs and recovers `A` within 0.02 of the truth. The learned model fits slightly better than the true parameters (−3512.6 vs −3518.4) because it is the maximum-likelihood fit to this one finite sample.

## Choosing / trade-offs
- **Viterbi vs posterior decoding.** Use Viterbi when you need one consistent segmentation (gene structure, speech alignment). Use per-step posteriors when you need per-step confidence or the best per-step accuracy.
- **Number of states.** Choose by held-out log-likelihood or BIC (parameters grow as K² + K·M), not training likelihood, which always rises with K.
- **Emission model.** Categorical for symbols; Gaussian or GMM per state for continuous values (`hmmlearn.GaussianHMM`, `GMMHMM`).
- **HMM vs CRF vs neural.** HMMs are generative and learn without labels. CRFs are discriminative and use rich overlapping features but need labels. RNNs and transformers model long-range context but need much more data.
- **Library.** `hmmlearn` (scikit-learn-style API) for Gaussian, GMM and categorical HMMs; `pomegranate` for more general probabilistic models.

## Gotchas
- Never multiply raw probabilities across a long sequence. Scale `α`/`β` per step or work in log space with `logsumexp`.
- EM's result depends on the start (4 of 5 uniform starts failed above). Use several restarts and keep the best log-likelihood, and start the transition matrix sticky when states are known to persist.
- A zero in the initial `A` or `B` stays zero forever under EM. Initialise with strictly positive values, or add pseudo-counts (a Dirichlet prior) in the M-step.
- One long sequence vs many short ones: for many sequences, sum the expected counts across sequences before normalising; do not concatenate them, which invents transitions at the joins.
- `log(0)` in Viterbi gives `-inf`, which is correct (forbidden transition) but turns into `nan` if multiplied. Keep Viterbi in pure additions.
- The Markov assumption means state durations are geometric. If real regimes have a typical length, use an explicit-duration (semi-Markov) model.

## Related
- [[hidden-markov-models-and-kalman-filters]] - library usage, Gaussian HMMs and the continuous-state counterpart.
- [[gmm-em-from-scratch]] - EM on a mixture model; an HMM is a mixture whose component follows a Markov chain.
- [[probabilistic-graphical-models]] - HMMs as the simplest dynamic Bayesian network.
- [[time-series-forecasting]] - regime features from HMM posteriors.
- [[rnn-from-scratch-numpy]] - a learned continuous hidden state instead of a discrete one.
- [[sequence-to-sequence-and-ctc]] - CTC uses the same forward-backward recursion over alignments.

## References
- Rabiner (1989), "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition", Proc. IEEE: https://doi.org/10.1109/5.18626
- Jurafsky and Martin, Speech and Language Processing, Appendix A (Hidden Markov Models): https://web.stanford.edu/~jurafsky/slp3/A.pdf
- hmmlearn documentation: https://hmmlearn.readthedocs.io/en/latest/
