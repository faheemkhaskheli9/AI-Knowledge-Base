---
title: Decoding and sampling
category: concepts
tags: [temperature, top-p, top-k, beam-search, speculative-decoding, generate, determinism]
use_cases:
  - "make an extraction or classification pipeline as deterministic as possible"
  - "tune creativity of a marketing-copy generator without gibberish"
  - "speed up local LLM inference with speculative decoding"
  - "get diverse candidates for best-of-n or self-consistency voting"
  - "debug repetitive or looping output from an open model"
status: draft
last_verified: 2026-10-03
sources:
  - https://huggingface.co/docs/transformers/generation_strategies
  - https://huggingface.co/blog/assisted-generation
  - https://arxiv.org/abs/1904.09751
  - https://arxiv.org/abs/2211.17192
---

# Decoding and sampling

## Summary
A language model outputs a probability distribution over the next token; the decoding strategy decides which token to emit. Settings such as temperature, top-p and top-k control the trade-off between determinism and diversity; beam search and speculative decoding address quality and speed.

## Key concepts
- **Greedy**: always take the most likely token. Deterministic-ish, can loop or be bland.
- **Temperature**: divides logits before softmax. Low (near 0) = peaked/deterministic; 1 = model's raw distribution; high = flatter, more random.
- **Top-k**: sample only from the k most likely tokens.
- **Top-p (nucleus)**: sample from the smallest set whose cumulative probability >= p (Holtzman et al.). Adapts to confidence. Typical advice: tune temperature or top-p, not both aggressively.
- **Min-p** and **repetition/frequency penalties**: other filters offered by some engines to curb junk and loops.
- **Beam search**: keep the b best partial sequences; suits translation/summarisation with a clear target, tends to produce generic text in open-ended chat.
- **Speculative decoding**: a small draft model proposes several tokens, the large model verifies them in one pass; accepted tokens are exact samples of the target distribution, so quality is unchanged while latency drops.
- **Constrained / structured decoding**: mask tokens so output follows a JSON schema or grammar ([[structured-output]]).
- **Best-of-n / self-consistency**: sample many answers, vote or rank ([[reasoning-models]]).

## When to use / scenarios
- Extraction, classification, SQL, code: temperature 0 (or low) and structured decoding.
- Brainstorming, copywriting, role-play: moderate temperature with top-p.
- Evaluation harnesses: fix settings and seeds so runs are comparable.
- Serving latency on self-hosted models: speculative decoding when a compatible small draft exists.
- Real-world: a bank's KYC extractor at temperature 0; an ad agency generating 20 slogan variants at higher temperature.

## Setup & code
`pip install transformers torch`

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "Qwen/Qwen2.5-1.5B-Instruct"
draft = "Qwen/Qwen2.5-0.5B-Instruct"            # same tokenizer family required
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, device_map="auto")
assistant = AutoModelForCausalLM.from_pretrained(draft, device_map="auto")

inputs = tok("Explain nucleus sampling in one sentence.", return_tensors="pt").to(model.device)

# sampling
out = model.generate(**inputs, do_sample=True, temperature=0.7, top_p=0.9, max_new_tokens=60)
print(tok.decode(out[0], skip_special_tokens=True))

# greedy + speculative (assisted) decoding
out = model.generate(**inputs, do_sample=False, assistant_model=assistant, max_new_tokens=60)
print(tok.decode(out[0], skip_special_tokens=True))
```
Hosted APIs expose `temperature`, `top_p`, `top_k` (provider dependent) and `max_tokens`; some newer models restrict or ignore sampling parameters, so check the provider's docs.

## Choosing / trade-offs
- Temperature 0 vs sampling: reproducibility and accuracy vs variety. Even temperature 0 is not guaranteed bit-identical across runs on hosted APIs or GPUs.
- Beam width: more beams = more compute and generic output.
- Speculative decoding: big wins when the draft agrees often (code, boilerplate, low batch sizes); little gain at large batch or poor draft match. Extra memory for the draft.
- Penalties fix loops but can hurt legitimate repetition (code, names, tables).

## Gotchas
- `do_sample=False` ignores temperature/top_p in Transformers (a warning appears); set `do_sample=True` to use them.
- Setting temperature 0 literally can raise a division error in some libraries; use greedy mode.
- Draft and target must share a tokenizer/vocabulary.
- Setting both low temperature and low top-p double-restricts; changes are hard to attribute.
- Missing stop tokens or too small `max_new_tokens` truncate output mid-JSON.
- Determinism claims: seeds help locally, but batching and non-deterministic kernels still introduce drift.

## Related
- [[transformers-and-attention]] - produces the logits being sampled.
- [[structured-output]] - constrained decoding for JSON.
- [[reasoning-models]] - best-of-n and self-consistency.
- [[inference-servers-vllm]] - speculative decoding and sampling params at scale.
- [[hallucination-and-grounding]] - low temperature helps but does not fix hallucination.
- [[cost-and-latency]] - decoding choices affect latency.
- [[speculative-decoding-from-scratch-numpy]] - speculative decoding built in NumPy, exact to the target distribution.

## References
- Transformers generation strategies: https://huggingface.co/docs/transformers/generation_strategies
- Assisted generation: https://huggingface.co/blog/assisted-generation
- Holtzman et al., Nucleus sampling: https://arxiv.org/abs/1904.09751
- Leviathan et al., Speculative decoding: https://arxiv.org/abs/2211.17192
