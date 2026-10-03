---
title: Mixture of experts (MoE)
category: concepts
tags: [moe, sparse-models, routing, experts, mixtral, scaling]
use_cases:
  - "understand why a large MoE model is cheap per token but still needs lots of GPU memory"
  - "size GPUs for serving an open MoE model such as Mixtral-style or DeepSeek-style models"
  - "choose between a dense and a MoE model for a latency-sensitive product"
  - "explain active vs total parameters to a procurement or finance team"
  - "fine-tune or quantize an MoE model without breaking routing"
status: draft
last_verified: 2026-10-03
sources:
  - https://arxiv.org/abs/1701.06538
  - https://arxiv.org/abs/2101.03961
  - https://arxiv.org/abs/2401.04088
  - https://huggingface.co/blog/moe
---

# Mixture of experts (MoE)

## Summary
An MoE transformer replaces each dense MLP with many parallel "expert" MLPs plus a small router that sends each token to only the top-k experts. Total parameters (knowledge capacity) grow a lot while per-token compute stays near that of a much smaller dense model.

## Key concepts
- **Router (gate)**: linear layer producing scores per expert; top-k (commonly k = 1 or 2, some designs add always-on shared experts) are selected and their outputs combined by gate weights.
- **Active vs total parameters**: compute and speed follow active parameters; memory follows total. Mixtral 8x7B has about 47B total, about 13B active per token (per the Mixtral paper).
- **Load balancing**: an auxiliary loss (or bias-based balancing) keeps tokens spread across experts; otherwise a few experts collapse into doing all the work.
- **Experts are not topic specialists**: they specialise on syntactic/token-level patterns more than neat human topics.
- **Expert parallelism**: experts are sharded across GPUs; routing causes all-to-all communication.
- **Fine-grained experts**: newer designs use many small experts and shared experts for better specialisation.

## When to use / scenarios
- You want quality of a large model with lower per-token latency/FLOPs and have enough aggregate GPU memory (multi-GPU server or large-memory node).
- High-throughput batched serving where all experts stay busy.
- Not ideal: single small GPU or laptop (you must still load all experts), tiny batch sizes (routing sparsity gives little speedup), or simple fine-tuning budgets.
- Real-world: a platform team serves an open MoE model for a coding assistant to many concurrent users; per-token cost is lower than an equal-quality dense model.

## Setup & code
Toy top-2 MoE layer to see the mechanism (`pip install torch`):

```python
import torch, torch.nn as nn, torch.nn.functional as F

class MoE(nn.Module):
    def __init__(self, d=32, n_experts=4, k=2):
        super().__init__()
        self.k = k
        self.router = nn.Linear(d, n_experts)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
            for _ in range(n_experts))

    def forward(self, x):                       # x: (tokens, d)
        w, idx = F.softmax(self.router(x), -1).topk(self.k, dim=-1)
        w = w / w.sum(-1, keepdim=True)         # renormalise over chosen experts
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            for slot in range(self.k):
                mask = idx[:, slot] == e
                if mask.any():
                    out[mask] += w[mask, slot, None] * expert(x[mask])
        return out

print(MoE()(torch.randn(10, 32)).shape)
```
Serving real MoE checkpoints: use vLLM or Transformers with `device_map="auto"`; see [[inference-servers-vllm]].

## Choosing / trade-offs
- Dense: simpler, predictable memory, easier fine-tuning. MoE: better quality per FLOP, harder to host and tune.
- Memory-bound small-batch decoding is limited by loading expert weights, so latency gains are smaller than FLOP counts suggest.
- Quantizing MoE saves memory but routing/gate layers are sensitive; keep the router in higher precision if the toolkit allows.

## Gotchas
- "Active parameters" is the wrong number for VRAM planning; use total.
- Token dropping/capacity limits in some training setups; imbalance causes quality loss.
- LoRA on MoE: target attention (and optionally experts) deliberately; naive "all linear" can create huge adapters.
- Benchmark comparisons across MoE and dense models need equal-cost framing (active params, total params, or serving cost).
- Routing is non-deterministic under batch composition in some kernels, causing small output variation.

## Related
- [[transformers-and-attention]] - the block whose MLP is replaced.
- [[pretraining-and-scaling-laws]] - MoE changes the scaling accounting.
- [[quantization]] - to fit total parameters in memory.
- [[inference-servers-vllm]] - serving with expert parallelism.
- [[deepseek]] - family known for large MoE models.
- [[mistral]] - Mixtral popularised open MoE.

## References
- Shazeer et al., Sparsely-Gated MoE: https://arxiv.org/abs/1701.06538
- Fedus et al., Switch Transformer: https://arxiv.org/abs/2101.03961
- Jiang et al., Mixtral of Experts: https://arxiv.org/abs/2401.04088
- Hugging Face, "Mixture of Experts Explained": https://huggingface.co/blog/moe
