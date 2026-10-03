---
title: Pretraining and scaling laws
category: concepts
tags: [pretraining, scaling-laws, chinchilla, data, compute, next-token-prediction]
use_cases:
  - "estimate the GPU budget and token count to train a small domain language model from scratch"
  - "decide whether to pretrain, continue-pretrain or just fine-tune for a specialised domain such as legal or medical text"
  - "explain to stakeholders why a bigger model needs more data, not just more parameters"
  - "choose a model size for a cost-sensitive product given inference volume"
  - "plan data curation and deduplication for a corpus in a low-resource language"
status: draft
last_verified: 2026-10-03
sources:
  - https://arxiv.org/abs/2001.08361
  - https://arxiv.org/abs/2203.15556
  - https://arxiv.org/abs/2005.14165
---

# Pretraining and scaling laws

## Summary
Pretraining trains a model on a huge unlabelled corpus with a simple objective (next-token prediction for decoder LLMs, masked-token for encoders), giving it general language and world knowledge. Scaling laws are empirical power-law fits that predict loss from parameters, data and compute, and guide how to spend a training budget.

## Key concepts
- **Objective**: causal LM predicts token t+1 from tokens <= t; loss is cross-entropy. Everything else (chat, tools) is layered on after.
- **Scaling laws (Kaplan 2020)**: loss falls as a smooth power law in model size N, dataset size D and compute C.
- **Chinchilla (Hoffmann 2022)**: for a fixed training-compute budget, parameters and tokens should be scaled roughly together (on the order of 20 tokens per parameter was the headline finding); many earlier models were under-trained.
- **Inference-aware training**: because inference cost scales with parameters but not tokens, modern small models are deliberately trained on far more tokens than compute-optimal, trading training cost for cheaper serving.
- **Training compute rule of thumb**: FLOPs ~ 6 x parameters x tokens.
- **Data quality dominates**: filtering, deduplication, language/domain mixing and decontamination of benchmarks matter as much as size.
- **Stages**: pretraining -> (continued pretraining / annealing on high-quality data) -> supervised fine-tuning -> preference optimisation (see [[rlhf-and-preference-optimization]]).
- **Emergent abilities**: some capabilities appear to jump with scale; metric choice often explains apparent discontinuities.

## When to use / scenarios
- Pretrain from scratch: rare; only with unique data at scale (new language, proprietary modality) and serious compute.
- Continued pretraining: adapt an open base model to a domain corpus (e.g. clinical notes, financial filings) when fine-tuning on Q&A is not enough.
- Usually better: pick an existing base model and use [[fine-tuning-and-peft]] or RAG.
- Real-world: a research lab training a 1B model for Urdu; a bank continuing pretraining on internal documents under strict data residency.

## Setup & code
Back-of-envelope budget calculator (pure Python):

```python
def budget(params_b, tokens_b, gpu_tflops=300, mfu=0.4):
    flops = 6 * params_b * 1e9 * tokens_b * 1e9
    gpu_seconds = flops / (gpu_tflops * 1e12 * mfu)
    return gpu_seconds / 3600                 # GPU-hours

print(round(budget(1, 20)), "GPU-hours for a 1B model on 20B tokens")
print(round(budget(1, 200)), "GPU-hours if trained 10x longer")
```
`gpu_tflops` and `mfu` (model FLOPs utilisation) are assumptions; replace with your hardware's real numbers.

## Choosing / trade-offs
- Compute-optimal (cheapest to train) vs over-trained small model (cheapest to serve at high volume).
- Pretrain vs continue-pretrain vs fine-tune vs RAG: cost rises left to right in the first three; RAG avoids training for knowledge that changes.
- More unique data beats repeated epochs; a few epochs of repetition is tolerable, many are not.

## Gotchas
- Scaling-law constants depend on architecture, data and tokenizer; extrapolate cautiously.
- Benchmark contamination inflates scores.
- Loss improvements do not map linearly to task quality.
- Training instabilities (loss spikes) and checkpoint/data-order reproducibility cost real time.
- Licences and copyright of corpora are legal risks, not just technical ones.

## Related
- [[transformers-and-attention]] - the architecture being trained.
- [[tokenization]] - tokens-per-parameter is counted in tokens.
- [[fine-tuning-and-peft]] - the usual alternative to pretraining.
- [[mixture-of-experts]] - changes the active-vs-total parameter accounting.
- [[small-language-models]] - over-trained small models.
- [[gpu-cloud-options]] - where compute is rented.

## References
- Kaplan et al., Scaling Laws for Neural Language Models: https://arxiv.org/abs/2001.08361
- Hoffmann et al., Chinchilla: https://arxiv.org/abs/2203.15556
- Brown et al., GPT-3: https://arxiv.org/abs/2005.14165
