---
title: Speculative decoding from scratch in NumPy (draft-then-verify, exact target distribution)
category: concepts
tags: [speculative-decoding, speculative-sampling, draft-model, llm-inference, latency, rejection-sampling, decoding, numpy, from-scratch, llm-basics]
use_cases:
  - "implement speculative sampling (accept with min(1, p/q), resample from max(0, p - q)) in NumPy"
  - "check empirically that speculative decoding samples exactly from the target model"
  - "see how acceptance rate and draft length k set the tokens generated per target forward pass"
  - "explain speculative decoding, draft models, Medusa/EAGLE and their limits in an interview"
status: stable
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/2211.17192
  - https://arxiv.org/abs/2302.01318
  - https://docs.vllm.ai/en/latest/features/spec_decode.html
  - https://huggingface.co/blog/assisted-generation
---

# Speculative decoding from scratch in NumPy (draft-then-verify, exact target distribution)

## Summary
LLM decoding is slow because every token needs a full forward pass of the big model, and at batch size 1 that pass is limited by memory bandwidth, not compute. Scoring 5 tokens in one pass costs about the same as scoring 1. Speculative decoding uses this: a small **draft** model proposes k tokens, the big **target** model scores all of them in one pass, and a rejection-sampling rule keeps a prefix of the drafts. The rule is chosen so the output has **exactly** the target model's distribution, so quality is unchanged. Below, toy bigram "models" over 50 tokens generate 20,000 tokens. With a good draft (81% acceptance) and k = 8, each target call yields 4.56 tokens. With a poor draft (46% acceptance), the gain drops to 1.68-1.86 tokens per call. In every setting, the generated transitions match the target as closely as plain sampling does (total variation 0.086-0.096 vs 0.090), while sampling from the draft alone is far off (0.22-0.58).

## Key concepts
- **Draft and verify.** The draft generates `x1..xk` autoregressively, recording its distributions `q_i`. The target runs once on `prefix + x1..xk` and returns its distributions `p_1..p_{k+1}` at every position.
- **Acceptance rule.** For each draft token in order, accept `x_i` with probability `min(1, p_i(x_i) / q_i(x_i))`. On the first rejection, sample a replacement from the residual `max(0, p_i − q_i)` (renormalized) and discard the rest of the drafts. If all k are accepted, sample one bonus token from `p_{k+1}`. Accepted-or-resampled tokens are distributed exactly as `p_i`.
- **Acceptance rate α.** The expected acceptance probability is `Σ_x min(p(x), q(x)) = 1 − TV(p, q)`, so it measures how close the draft is to the target. With a constant α, expected tokens per target call are `(1 − α^{k+1}) / (1 − α)`; at α = 0.81 and k = 2 that is 2.47, matching the 2.46 measured below.
- **Greedy case.** With temperature 0, the rule reduces to "keep drafts while they equal the target's argmax", which is the simplest form to implement and debug.
- **Where the speed-up comes from.** The draft must be much cheaper than the target (often 10-50× fewer parameters, or a few extra heads on the target itself), and the target's k+1-token verification pass must cost about as much as a 1-token pass. Both hold at small batch sizes, so speculative decoding improves latency, not throughput under heavy batching.

## When to use / scenarios
- Learning: a clean example of rejection sampling that preserves a distribution exactly, and of why LLM inference is memory-bound ([[kv-cache-from-scratch-numpy]], [[decoding-strategies-from-scratch-numpy]]).
- Interviews: "how does speculative decoding keep quality unchanged", "what limits the speed-up", "draft model vs Medusa vs EAGLE vs prompt lookup".
- Practice: interactive chat and coding assistants where per-user latency matters, and long outputs that copy from the input (code edits, RAG answers), where n-gram "prompt lookup" drafts are nearly free and accept often. Enable it in the inference server rather than writing it yourself ([[inference-servers-vllm]]).
- Not for: throughput-bound batch jobs where the GPU is already compute-saturated (verification of rejected tokens is wasted work); drafts with very different tokenizers from the target (tokens must align); or creative high-temperature sampling where acceptance rates drop.

## Setup & code
NumPy only. Runs in about 4 seconds. The "models" are bigram tables: the target is a fixed random table and the draft is the same table with Gaussian noise added to its logits, so a larger noise means a worse draft.

```python
import numpy as np

rng = np.random.default_rng(0)
V = 50                                                   # vocabulary size


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


# Toy "models": next-token distribution depends on the previous token (bigram tables).
target_logits = rng.normal(0, 2.0, (V, V))
P = softmax(target_logits)                               # big, slow target model


def draft_table(noise):
    return softmax(target_logits + rng.normal(0, noise, (V, V)))   # small model ~ noisy copy


def sample(p):
    return rng.choice(len(p), p=p)


def speculative(Q, n_tokens, k):
    """Generate n_tokens with draft table Q proposing k tokens per target call."""
    seq, target_calls, accepted, judged = [0], 0, 0, 0
    while len(seq) - 1 < n_tokens:
        ctx, drafts, qs = seq[-1], [], []
        for _ in range(k):                               # draft k tokens autoregressively
            q = Q[ctx]
            t = sample(q)
            drafts.append(t)
            qs.append(q)
            ctx = t
        target_calls += 1                                # ONE target pass scores all k+1 positions
        ctxs = [seq[-1]] + drafts
        ps = [P[c] for c in ctxs]
        for i, t in enumerate(drafts):
            judged += 1
            if rng.random() < min(1.0, ps[i][t] / qs[i][t]):     # accept with prob min(1, p/q)
                seq.append(t)
                accepted += 1
            else:                                        # reject: resample from max(0, p - q)
                resid = np.maximum(ps[i] - qs[i], 0)
                seq.append(sample(resid / resid.sum()))
                break
        else:
            seq.append(sample(ps[k]))                    # all k accepted: free bonus token
    return seq[1:n_tokens + 1], target_calls, accepted, judged


def vanilla(n_tokens, table=P):
    seq = [0]
    for _ in range(n_tokens):
        seq.append(sample(table[seq[-1]]))
    return seq[1:]


def bigram_tv(seq):
    """Mean total-variation distance between empirical transitions and the target table P."""
    C = np.zeros((V, V))
    prev = [0] + seq[:-1]
    np.add.at(C, (prev, seq), 1)
    rows = C.sum(1) > 0
    emp = C[rows] / C[rows].sum(1, keepdims=True)
    w = C[rows].sum(1) / C.sum()
    return float((0.5 * np.abs(emp - P[rows]).sum(1) * w).sum())


N = 20000
print(f"vocab {V}, {N} tokens per run")
print(f"vanilla sampling: target calls {N}, bigram TV vs target {bigram_tv(vanilla(N)):.3f}")
print("draft noise  k  accept rate  tokens/target call  target calls  bigram TV")
for noise in [0.5, 1.0, 2.0]:
    Q = draft_table(noise)
    for k in [2, 4, 8]:
        seq, calls, acc, judged = speculative(Q, N, k)
        print(f"{noise:>11} {k:>2}  {acc / judged:>11.3f}  {N / calls:>18.2f}  {calls:>12}  {bigram_tv(seq):>9.3f}")
    print(f"  (draft alone, noise {noise}: bigram TV vs target {bigram_tv(vanilla(N, Q)):.3f})")
```

Output (Python 3.14, NumPy 2.5):
```
vocab 50, 20000 tokens per run
vanilla sampling: target calls 20000, bigram TV vs target 0.090
draft noise  k  accept rate  tokens/target call  target calls  bigram TV
        0.5  2        0.809                2.46          8118      0.087
        0.5  4        0.813                3.45          5800      0.088
        0.5  8        0.815                4.56          4384      0.087
  (draft alone, noise 0.5: bigram TV vs target 0.221)
        1.0  2        0.665                2.11          9484      0.096
        1.0  4        0.661                2.58          7749      0.094
        1.0  8        0.658                2.85          7011      0.086
  (draft alone, noise 1.0: bigram TV vs target 0.359)
        2.0  2        0.461                1.68         11937      0.093
        2.0  4        0.461                1.82         10988      0.090
        2.0  8        0.463                1.86         10754      0.092
  (draft alone, noise 2.0: bigram TV vs target 0.581)
```

The TV column is the correctness check. With 20,000 tokens over 2,500 transitions, even exact sampling from the target leaves a TV of about 0.09 from finite-sample noise. Every speculative run lands at that floor, whereas sampling from the draft alone does not. The speed-up depends almost entirely on α: at α ≈ 0.46, drafting 8 tokens instead of 2 adds only 0.18 tokens per call, because the chance of accepting a long run (`α^k`) collapses. The table counts target calls only. Real wall-clock gain is lower because the draft also costs time (k draft passes per round), so measured speed-ups are typically 2-3× for good draft/target pairs, not the raw tokens-per-call figure.

## Choosing / trade-offs
- **Separate draft model.** A smaller model from the same family with the same tokenizer (for example a 1B draft for a 70B target). It is easy to set up, but it uses extra memory and its acceptance depends on the domain.
- **Self-drafting heads (Medusa, EAGLE).** Extra lightweight heads trained on the target's hidden states predict several future tokens. They accept more often than an independent small model and need no second model, but they require a training step per target model.
- **Prompt lookup / n-gram drafting.** Propose the continuation of the last few tokens' earlier occurrence in the prompt. It costs almost nothing and does well on code editing, summarization with quotes and RAG answers that copy the context.
- **Choosing k.** Pick k so that `α^k` is still meaningful; around 3-5 for α ≈ 0.7-0.8. Larger k wastes draft compute and target verification on tokens that will be rejected. Some servers adapt k from the measured acceptance rate.
- **Batching.** At large batch sizes the target is compute-bound, so verifying rejected tokens costs real throughput. Servers often turn speculation off, or shorten k, as load rises.

## Gotchas
- The acceptance test needs the draft's full distribution `q`, not just its sampled token. Sampling from the draft with one temperature and verifying with another breaks exactness; apply the same temperature, top-p and top-k processing to both `p` and `q`.
- On rejection, sample from the residual `max(0, p − q)`, not from `p`. Resampling from `p` over-weights tokens the draft already proposed and biases the output.
- Draft and target must share a tokenizer (or you need token alignment logic). Different vocabularies make `p(x_i)` meaningless.
- The KV cache must be rolled back to the last accepted position after a rejection, for both models. Forgetting this produces subtly wrong continuations rather than an error.
- Measure wall-clock latency on your real traffic. A high tokens-per-call figure can still give little speed-up if the draft is slow or runs on the same GPU and competes for bandwidth.
- With greedy decoding, outputs should be token-for-token identical with and without speculation. If they differ, the cause is usually numerical non-determinism in batched verification, which is worth knowing before blaming the algorithm.

## Related
- [[decoding-strategies-from-scratch-numpy]] - temperature, top-k and top-p, which must be applied identically to draft and target.
- [[kv-cache-from-scratch-numpy]] - why decoding is memory-bound and how the cache is rolled back.
- [[decoding-and-sampling]] - overview of decoding methods.
- [[inference-servers-vllm]] - production speculative decoding settings.
- [[small-language-models]] - candidate draft models.

## References
- Leviathan, Kalman and Matias (2023), "Fast Inference from Transformers via Speculative Decoding": https://arxiv.org/abs/2211.17192
- Chen et al. (2023), "Accelerating Large Language Model Decoding with Speculative Sampling": https://arxiv.org/abs/2302.01318
- vLLM documentation, speculative decoding: https://docs.vllm.ai/en/latest/features/spec_decode.html
- Hugging Face blog, "Assisted Generation": https://huggingface.co/blog/assisted-generation
