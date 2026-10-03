---
title: Meta Llama models
category: models
tags: [llama, meta, open-weights, moe, self-hosting, community-license]
use_cases:
  - "self-host an open-weight LLM for on-prem document Q&A"
  - "fine-tune a Llama model on proprietary support transcripts"
  - "check whether Llama's license allows our commercial product"
  - "run an open multimodal model behind our firewall for a hospital"
status: draft
last_verified: 2026-10-03
sources:
  - https://huggingface.co/meta-llama
  - https://dev.meta.ai/llama/docs/model-cards-and-prompt-formats/
---

# Meta Llama models

## Summary
Llama is Meta's family of open-weight LLMs, distributed on Hugging Face under a
custom "Llama Community License" (not an OSI open-source license). Llama 4 is
the newest open generation visible on Meta's Hugging Face org: natively
multimodal mixture-of-experts models. Web reports say Meta's newest frontier
model (April 2026) is closed and API-only, so do not assume future Llama
releases stay open.

## Key concepts
- Llama 4 Scout: 17B active parameters, 16 experts. Llama 4 Maverick: 17B
  active, 128 experts (source: Meta HF org page).
- Older generations remain popular and easy to run: Llama 3.1 (8B-405B), 3.2
  (1B/3B text, 11B/90B vision), 3.3 (70B text).
- Companion safety models: Llama Guard, Prompt Guard.
- Access is gated: accept the license on the HF repo page, then `huggingface-cli
  login`.
- License is custom: it carries conditions (attribution/naming, acceptable-use
  policy, a very-large-company clause). I could not load the license text in
  this session, so read it before shipping; see [[model-licenses]].

## When to use / scenarios
- On-prem or VPC deployments where data cannot go to an API (healthcare,
  finance, government).
- Fine-tuning on domain data with LoRA ([[fine-tuning-and-peft]]).
- Edge/local use with small Llama 3.2 1B/3B ([[small-language-models]]).
- Not for: projects needing a permissive license with no conditions - prefer
  Apache/MIT models such as [[qwen]], [[mistral]] (Apache variants), [[deepseek]].
- Check whether newer open models (Qwen, DeepSeek, Gemma) beat Llama 4 on your
  evals; in 2026 several do on cost/quality.

## Setup & code
```bash
pip install transformers accelerate
huggingface-cli login   # after accepting the license on the model page
```
```python
from transformers import pipeline

pipe = pipeline("text-generation", model="meta-llama/Llama-3.3-70B-Instruct", device_map="auto")
out = pipe([{"role": "user", "content": "List 3 risks of self-hosting LLMs."}], max_new_tokens=200)
print(out[0]["generated_text"][-1]["content"])
```
For serving use vLLM ([[inference-servers-vllm]]) or Ollama ([[ollama]]).

## Choosing / trade-offs
- MoE models need memory for all experts even though few are active; check
  VRAM for the total parameter count, not the active count.
- 70B dense needs multi-GPU or aggressive quantization ([[quantization]]).
- Hosted Llama (cloud providers) trades privacy for zero ops.

## Gotchas
- "Open weights" is not "open source"; conditions apply.
- Gated downloads fail in CI without a token.
- Context-length claims differ by deployment; verify on the model card.
- Chat templates differ between Llama versions; use `apply_chat_template`.

## Related
- [[model-licenses]] - license comparison.
- [[model-selection]] - open vs closed.
- [[huggingface-transformers]] - loading models.
- [[mixture-of-experts]] - why active != total params.

## References
- https://huggingface.co/meta-llama
- https://dev.meta.ai/llama/docs/model-cards-and-prompt-formats/
