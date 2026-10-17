---
title: Diffusion language models (dLLMs)
category: concepts
tags: [diffusion-language-models, dllm, masked-diffusion, llada, mdlm, block-diffusion, parallel-decoding, non-autoregressive]
use_cases:
  - "decide whether a diffusion LLM is worth trying instead of an autoregressive model for low-latency code generation"
  - "understand how a masked diffusion model generates text and why fewer denoising steps can lower quality"
  - "fill in the middle of a document or code file with a model that sees both sides of the gap"
  - "explain the trade-offs of text diffusion vs next-token prediction to a team evaluating vendors"
status: draft
last_verified: 2026-10-15
sources:
  - https://arxiv.org/abs/2107.03006
  - https://arxiv.org/abs/2310.16834
  - https://arxiv.org/abs/2406.07524
  - https://arxiv.org/abs/2502.09992
  - https://arxiv.org/abs/2503.09573
  - https://arxiv.org/abs/2506.17298
  - https://deepmind.google/models/gemini-diffusion/
  - https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/
---

# Diffusion language models (dLLMs)

## Summary
Diffusion language models generate text by starting from a fully masked (or noised) sequence and repeatedly denoising all positions in parallel, instead of emitting one token at a time left to right. The draw is speed (many tokens per forward pass), bidirectional context (natural infilling and editing) and a tunable quality/latency knob (the number of denoising steps). The costs are a factorization error when many tokens are fixed at once, weaker caching than autoregressive models, and a much younger ecosystem; most deployed LLMs remain autoregressive.

## Key concepts
- **Discrete diffusion**: a forward process corrupts tokens (most commonly replacing them with a `[MASK]` "absorbing" state with probability growing over time t); a model learns to reverse it (D3PM, Austin et al.).
- **Masked diffusion LMs (MDLM, LLaDA)**: the training loss reduces to a weighted masked-language-modelling loss over random mask ratios, with a bidirectional transformer (no causal mask). Sampling starts fully masked and unmasks positions over several steps.
- **Score-entropy / SEDD**: an alternative discrete-diffusion objective that models ratios between token probabilities; one of the results that made text diffusion competitive on perplexity.
- **Remasking strategy**: at each step, which predicted tokens to keep and which to mask again. Low-confidence remasking (keep the most confident predictions) is common.
- **Factorization error**: in one step the model predicts each masked position independently. Fixing many positions at once can combine individually likely tokens into an unlikely sequence. More steps reduce it; the demo below shows this.
- **Block (semi-autoregressive) diffusion**: generate blocks left to right, diffusing within each block. Restores KV caching across blocks and variable-length output (Arriola et al.).
- **State of the field**: LLaDA trained an 8B masked diffusion model from scratch and reported results competitive with autoregressive models of similar size; Inception Labs' Mercury is a commercial diffusion LLM aimed at code with vendor-reported high throughput; Google DeepMind has shown Gemini Diffusion as an experimental model and later released DiffusionGemma, an open Apache-2.0 26B mixture-of-experts text-diffusion model built on Gemma 4, which Google says is faster but lower quality than standard Gemma 4. Treat speed claims as vendor-reported until you measure them.

## When to use / scenarios
- Latency-critical generation of short to medium outputs (code edits, completions) where a vendor's dLLM meets your quality bar.
- Infilling and editing: filling a gap in code or text given both left and right context, constrained templates, or regenerating a span in place.
- Research on controllable or non-left-to-right generation.
- Real-world: an IDE that offers fast multi-line edits from a diffusion coding model; a form-filling tool that regenerates one field in the middle of a document.
- Not when: you need the strongest available reasoning, long-form generation, tool use and ecosystem support (structured output, mature inference servers) today; pick an autoregressive model from [[model-selection]]. For pure speed on an autoregressive model, try [[multi-token-prediction]] or speculative decoding first.

## Setup & code
Masked-diffusion sampling with an ideal denoiser over a tiny "language" of 12 words. The denoiser returns the exact posterior for each masked position; the sampler unmasks the most confident positions each step and samples them independently. NumPy only.

```python
import numpy as np

rng = np.random.default_rng(0)
# The "data distribution": uniform over a few 6-letter words.
corpus = ["planet", "planes", "plants", "player", "prance", "grapes", "graphs", "groves",
          "stones", "stoves", "states", "string"]
L, MASK = 6, None
alphabet = sorted(set("".join(corpus)))


def denoiser(seq):
    """Ideal masked denoiser: for every masked position, the exact posterior over letters
    given the unmasked letters. A trained transformer approximates this in one forward pass."""
    consistent = [w for w in corpus if all(s is MASK or s == c for s, c in zip(seq, w))]
    consistent = consistent or corpus      # already off-distribution: fall back to the prior
    probs = {}
    for i, s in enumerate(seq):
        if s is MASK:
            counts = np.array([sum(w[i] == a for w in consistent) for a in alphabet], float)
            probs[i] = counts / counts.sum()
    return probs


def sample(steps):
    """Masked-diffusion sampling: start fully masked, and at each step unmask the
    most confident positions, sampling each one independently from its marginal."""
    seq = [MASK] * L
    per_step = int(np.ceil(L / steps))
    for _ in range(steps):
        probs = denoiser(seq)
        if not probs:
            break
        order = sorted(probs, key=lambda i: -probs[i].max())   # confidence-based order
        for i in order[:per_step]:                             # parallel, independent picks
            seq[i] = alphabet[rng.choice(len(alphabet), p=probs[i])]
    return "".join(seq)


for steps in (1, 2, 3, 6):
    outs = [sample(steps) for _ in range(2000)]
    valid = np.mean([o in corpus for o in outs])
    print(f"{steps} denoising step(s), {L / steps:.0f} tokens per step: "
          f"{valid:.0%} valid words, e.g. {outs[:4]}")
```
Every marginal the denoiser returns is exact, yet unmasking all six letters in one step mixes letters from different words. With one token per step the sampler is as accurate as left-to-right generation, but gains no parallelism. Real dLLMs live between these extremes, and the steps-vs-quality curve is the main knob to tune.

Open checkpoints such as LLaDA are published on Hugging Face with custom generation code (`trust_remote_code`); follow the model card for the exact sampler arguments (steps, block length, remasking).

## Choosing / trade-offs
| | Autoregressive | Masked diffusion |
|---|---|---|
| Tokens per forward pass | 1 (more with speculative decoding) | Many; set by steps |
| Context | Left only (causal) | Both sides (bidirectional) |
| KV cache | Standard, very effective | Not directly; block diffusion and approximate caches help |
| Output length | Natural stop token | Fixed canvas or block-wise |
| Likelihood | Exact | Bound (ELBO) |
| Ecosystem | Mature | Young |

- Fewer steps: faster, more incoherent combinations. More steps: closer to AR quality, less speed advantage.
- Each diffusion step attends over the whole sequence, so a step costs more than one AR decode step; the win comes from needing fewer steps than tokens.

## Gotchas
- Quoted tokens/second compare different hardware, batch sizes and step counts; benchmark at your own quality target.
- Perplexity numbers for diffusion models are usually upper bounds and not directly comparable with AR perplexity.
- A fixed generation length must be chosen up front in plain masked diffusion; too long wastes compute on padding, too short truncates.
- Classifier-free guidance, temperature and remasking settings interact; defaults from one model card do not transfer.
- Inference-server support (batching, streaming, structured output) lags autoregressive models; check before committing.

## Related
- [[diffusion-models]] - diffusion for images, where the idea comes from.
- [[ddpm-diffusion-from-scratch-numpy]] - continuous diffusion built by hand.
- [[multi-token-prediction]] - the autoregressive route to several tokens per step.
- [[speculative-decoding-from-scratch-numpy]] - exact parallel drafting for AR models.
- [[decoding-and-sampling]] - sampling knobs that also apply per denoising step.
- [[transformers-and-attention]] - the bidirectional transformer used as the denoiser.
- [[autoencoders-and-self-supervised-learning]] - masked-language-model training, the closest relative of the loss.

## References
- Austin et al., Structured Denoising Diffusion Models in Discrete State-Spaces (D3PM): https://arxiv.org/abs/2107.03006
- Lou, Meng, Ermon, Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution (SEDD): https://arxiv.org/abs/2310.16834
- Sahoo et al., Simple and Effective Masked Diffusion Language Models (MDLM): https://arxiv.org/abs/2406.07524
- Nie et al., Large Language Diffusion Models (LLaDA): https://arxiv.org/abs/2502.09992
- Arriola et al., Block Diffusion: https://arxiv.org/abs/2503.09573
- Inception Labs, Mercury: Ultra-Fast Language Models Based on Diffusion: https://arxiv.org/abs/2506.17298
- Google DeepMind, Gemini Diffusion: https://deepmind.google/models/gemini-diffusion/
- Google, DiffusionGemma announcement: https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/
