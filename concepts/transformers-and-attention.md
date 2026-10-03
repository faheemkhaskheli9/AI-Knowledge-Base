---
title: Transformers and attention
category: concepts
tags: [transformer, attention, self-attention, kv-cache, flash-attention, pytorch]
use_cases:
  - "explain to a team why LLM inference memory grows with context length"
  - "implement a small attention block from scratch to debug shapes in a custom model"
  - "pick an encoder, decoder or encoder-decoder architecture for a classification, chat or translation product"
  - "reduce latency of a chatbot serving long conversations (KV cache, GQA)"
  - "decide between a BERT-style encoder and an LLM for document tagging in a legal or insurance workflow"
status: draft
last_verified: 2026-10-03
sources:
  - https://arxiv.org/abs/1706.03762
  - https://arxiv.org/abs/2205.14135
  - https://arxiv.org/abs/2305.13245
  - https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
---

# Transformers and attention

## Summary
The transformer is the neural architecture behind nearly all modern LLMs, vision transformers and many speech and diffusion models. Each token's representation is updated by a weighted mix of other tokens' representations (attention), followed by a per-token MLP. It replaced RNNs because all positions train in parallel and any token can reach any other in one step.

## Key concepts
- **Self-attention**: each token produces a query (Q), key (K) and value (V). Output = softmax(QK^T / sqrt(d)) V. Cost is O(n^2) in sequence length n.
- **Multi-head**: several attention maps in parallel on lower-dim slices, then concatenated.
- **Block**: attention + MLP, each wrapped in residual connection and normalisation (pre-norm in most modern models; RMSNorm is common).
- **Causal mask**: in decoder-only models a token only attends to earlier tokens, enabling next-token prediction.
- **Positional information**: attention is order-blind; positions are injected (sinusoidal, learned, or rotary/RoPE in most current open models).
- **Three layouts**: encoder-only (BERT; embeddings, classification), decoder-only (GPT/Llama/Claude style; generation), encoder-decoder (T5, Whisper; translation, speech).
- **KV cache**: at generation time K and V of past tokens are stored so each new token costs one step, not a full recompute. Cache size ~ layers x kv_heads x head_dim x tokens x 2 x bytes. This, not weights, often limits batch size and context.
- **MQA / GQA**: share K/V across groups of query heads to shrink the KV cache with small quality loss.
- **FlashAttention**: exact attention computed in tiles to avoid materialising the n x n matrix; faster and memory-lean. Use it via PyTorch SDPA or library flags.

## When to use / scenarios
- Any text generation, chat, summarisation or code task: decoder-only transformer.
- Embeddings, classification, NER on large volumes with tight latency: small encoder-only model (cheaper than an LLM).
- Translation or speech-to-text: encoder-decoder.
- Not ideal: very long sequences with strict linear cost (consider state-space/hybrid models) or small tabular data (use gradient boosting, see [[gradient-boosting-tabular]]).
- Real-world: a bank tagging 1M support emails/day uses a fine-tuned encoder; a coding assistant uses a decoder LLM with a large KV cache budget.

## Setup & code
Minimal causal self-attention using the fused PyTorch kernel (`pip install torch`):

```python
import torch, torch.nn as nn, torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        self.h = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):                      # x: (batch, seq, d_model)
        b, n, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(b, n, self.h, d // self.h).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # uses Flash kernels when available
        return self.out(y.transpose(1, 2).reshape(b, n, d))

x = torch.randn(2, 10, 64)
print(CausalSelfAttention()(x).shape)  # torch.Size([2, 10, 64])
```

KV-cache sizing back-of-envelope: `2 * layers * kv_heads * head_dim * tokens * bytes_per_value` per sequence.

## Choosing / trade-offs
- Full attention (best quality) vs sliding-window/sparse (cheaper, loses far context).
- MHA vs GQA/MQA: GQA is the usual default for serving; MHA only if training from scratch with memory to spare.
- Bigger context raises KV memory linearly and attention compute quadratically (see [[long-context]]).
- Encoder vs decoder for classification: encoder is 10-100x cheaper; LLM needs no labelled data.

## Gotchas
- Forgetting the causal mask during training leaks future tokens (suspiciously low loss).
- Softmax in fp16 can overflow; use bf16 or fused kernels.
- Quoting "context window" ignores that quality often degrades before the limit.
- Padding tokens must be masked in batched encoder use.
- Attention weights are not a faithful explanation of model decisions.

## Related
- [[tokenization]] - what the sequence elements actually are.
- [[long-context]] - scaling attention to huge inputs.
- [[mixture-of-experts]] - replaces the MLP half of the block with sparse experts.
- [[quantization]] - shrinks weights and KV cache.
- [[decoding-and-sampling]] - how the next token is picked from the output.
- [[pytorch-basics]] - tensor/training fundamentals.

## References
- Vaswani et al., "Attention Is All You Need": https://arxiv.org/abs/1706.03762
- Dao et al., FlashAttention: https://arxiv.org/abs/2205.14135
- Ainslie et al., GQA: https://arxiv.org/abs/2305.13245
- PyTorch SDPA docs: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
