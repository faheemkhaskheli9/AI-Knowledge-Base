---
title: Multi-token prediction (MTP)
category: concepts
tags: [multi-token-prediction, mtp, speculative-decoding, medusa, deepseek-v3, pretraining, inference-speed]
use_cases:
  - "understand why a model card says it was trained with multi-token prediction and what that buys at inference"
  - "speed up decoding of a model that ships MTP heads by using them as a built-in draft model"
  - "decide between MTP heads, Medusa heads and a separate draft model for speculative decoding"
  - "evaluate whether adding an auxiliary multi-token loss is worth it in our own pretraining run"
status: draft
last_verified: 2026-10-15
sources:
  - https://arxiv.org/abs/2404.19737
  - https://arxiv.org/abs/2412.19437
  - https://arxiv.org/abs/2401.10774
  - https://arxiv.org/abs/1811.03115
  - https://arxiv.org/abs/2211.17192
---

# Multi-token prediction (MTP)

## Summary
Multi-token prediction trains a language model to predict several future tokens at each position (t+1, t+2, ... t+k) instead of only the next one, using extra output heads or small extra modules on a shared trunk. It is used for two things: a denser training signal that can improve the main model (especially on code), and cheap drafts for self-speculative decoding, where the extra heads propose tokens that the normal next-token head then verifies, giving more tokens per forward pass with identical output.

## Key concepts
- **Objective**: loss = sum over offsets i = 1..k of cross-entropy for token t+i, all from the hidden state at position t. Offset 1 is ordinary next-token prediction.
- **Parallel heads (Gloeckle et al., Meta)**: k independent output heads on a shared transformer trunk, trained from scratch together. The paper reports gains that grow with model size and are strongest on coding benchmarks, plus faster inference via self-speculative decoding.
- **Sequential MTP modules (DeepSeek-V3)**: each extra module takes the previous depth's representation plus the embedding of the next true token, so the causal chain is kept. Used mainly as a training objective; the module can be dropped at inference or reused as a speculative draft.
- **Medusa**: adds extra decoding heads to an already-trained model (backbone frozen or lightly tuned) purely for speed, verifying a tree of candidate continuations in one pass with tree attention. Not a pretraining objective.
- **Self-speculative decoding**: the heads draft tokens t+2..t+k; one forward pass verifies them with the next-token head; accept the longest matching prefix. With greedy verification (or proper rejection sampling) the output distribution equals the base model's. See [[speculative-decoding-from-scratch-numpy]].
- **Why later heads are weaker**: head i must guess token t+i without seeing tokens t+1..t+i-1, so it predicts a marginal over all paths. Acceptance falls with offset, which caps the speed-up.
- **Lineage**: blockwise parallel decoding (Stern et al., 2018) proposed the same draft-and-verify idea for sequence models.

## When to use / scenarios
- You serve a model that ships MTP or Medusa heads: enable them as the draft in your inference server to cut latency without a second model.
- Latency-bound single-user generation (code completion, chat) where GPUs are memory-bandwidth bound and spare compute per pass is available.
- Pretraining your own code or general LM: an auxiliary MTP loss is one option to test against plain next-token training.
- Real-world: an IDE completion service decoding with MTP drafts; a self-hosted DeepSeek deployment using its MTP module for speculative decoding.
- Not when: high-throughput batch serving where GPUs are already compute-bound (speculation adds wasted work); very small models where the paper found MTP can hurt; outputs with high entropy (creative sampling) where drafts are rarely accepted.

## Setup & code
Self-speculative greedy decoding with ideal MTP heads on a toy Markov-chain "language". NumPy only. Head i is the best any head can do from the current token alone: the i-step marginal `P^i`.

```python
import numpy as np

rng = np.random.default_rng(0)
V = 50
# Toy "language": a peaked Markov chain. P[a, b] = p(next=b | current=a).
logits = rng.normal(size=(V, V)) + 3.0 * np.eye(V)[rng.permutation(V)]
P = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)


def heads(k):
    """Ideal MTP head i predicts token t+i from the current token: the i-step marginal P^i.
    It cannot condition on the tokens in between, which is why later heads are weaker."""
    return [np.linalg.matrix_power(P, i) for i in range(1, k + 1)]


def greedy_next(tok):                     # the base model's own next-token choice
    return int(P[tok].argmax())


def generate(start, n_tokens, k):
    """Self-speculative greedy decoding: draft k tokens with the heads in one pass,
    then verify them with the next-token head (one batched pass in a real model)."""
    H = heads(k)
    out, passes = [start], 0
    while len(out) < n_tokens + 1:
        cur = out[-1]
        draft = [int(h[cur].argmax()) for h in H]  # one forward pass gives all k drafts
        passes += 1
        prev = cur
        for i, d in enumerate(draft):
            if i == 0 or d == greedy_next(prev):    # head 1 is the base model itself
                out.append(d)
                prev = d
            else:                                   # first mismatch: keep the verified token
                out.append(greedy_next(prev))
                break
    return out[1:n_tokens + 1], passes


for k in (1, 2, 3, 4):
    total = 0
    for start in range(V):                     # average over many prompts
        reference, _ = generate(start, 40, k=1)  # plain one-token-per-pass greedy
        toks, passes = generate(start, 40, k)
        assert toks == reference               # greedy verification never changes the output
        total += passes
    print(f"heads={k}: {40 * V / total:.2f} tokens per forward pass")

H = heads(4)
greedy_path = [int(P[a].argmax()) for a in range(V)]
for i, h in enumerate(H, start=1):
    # how often head i's guess matches what greedy decoding actually produces i steps ahead
    match = 0
    for a in range(V):
        t = a
        for _ in range(i):
            t = greedy_path[t]
        match += int(h[a].argmax()) == t
    print(f"head {i} agrees with the greedy path for {match}/{V} start tokens")
```
Two things to notice: the output is identical to plain greedy decoding (the assert), and each extra head adds less than the one before because its guesses agree with the real continuation less often.

For production, the inference server does this for you: vLLM and SGLang support speculative decoding with draft models, EAGLE-style heads and, for some model families, the checkpoint's own MTP module. Method names and supported models change between releases, so check the speculative-decoding page of your server's docs for your version ([[inference-servers-vllm]]).

## Choosing / trade-offs
| Draft source | Extra training | Extra memory | Notes |
|---|---|---|---|
| MTP heads from pretraining | None (shipped with model) | Small | Only if the checkpoint includes them |
| Medusa / EAGLE heads | Train heads on your model | Small | Works on any model you can tune |
| Separate small draft model | None if one exists | A second model | Needs the same tokenizer |
| n-gram / prompt lookup | None | None | Good when output copies the input |

- More heads: more potential tokens per pass, but lower acceptance per extra head and more verification work.
- Training with MTP: possible quality gain, more training compute per step; results depend on scale and domain.

## Gotchas
- Speed-up depends on acceptance rate, which depends on the task and sampling temperature; measure on your real traffic.
- Under large batch sizes, speculative decoding can lower total throughput even when it lowers per-request latency.
- Sampling (temperature > 0) needs the rejection-sampling acceptance rule to keep the distribution exact; naive "accept if equal to argmax" changes outputs.
- Quantising or fine-tuning the main model without updating the MTP/Medusa heads lowers acceptance.
- "Trained with MTP" on a model card does not mean the heads are in the released weights or supported by your server.

## Related
- [[speculative-decoding-from-scratch-numpy]] - the draft-and-verify algorithm and its exactness proof.
- [[decoding-and-sampling]] - greedy vs sampling, which changes the acceptance rule.
- [[inference-servers-vllm]] - enabling speculative decoding in production.
- [[kv-cache-from-scratch-numpy]] - why decoding is memory-bound, which is what speculation exploits.
- [[pretraining-and-scaling-laws]] - MTP as a pretraining objective.
- [[deepseek]] - DeepSeek-V3 uses an MTP training objective.
- [[diffusion-language-models]] - another route to generating several tokens per step.
- [[cost-and-latency]] - where latency gains matter.

## References
- Gloeckle et al., Better & Faster Large Language Models via Multi-token Prediction: https://arxiv.org/abs/2404.19737
- DeepSeek-AI, DeepSeek-V3 Technical Report (MTP section): https://arxiv.org/abs/2412.19437
- Cai et al., Medusa: https://arxiv.org/abs/2401.10774
- Stern et al., Blockwise Parallel Decoding: https://arxiv.org/abs/1811.03115
- Leviathan et al., Fast Inference from Transformers via Speculative Decoding: https://arxiv.org/abs/2211.17192
