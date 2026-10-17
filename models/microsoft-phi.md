---
title: Microsoft Phi small models
category: models
tags: [phi, phi-4, microsoft, slm, on-device, mit-license, foundry-local, onnx, reasoning]
use_cases:
  - "run a private assistant on a Windows laptop with no cloud calls"
  - "ship an MIT-licensed small model inside a commercial desktop or mobile app"
  - "add step-by-step math reasoning to an offline tutoring app"
  - "fine-tune a 3.8B model for one narrow extraction task"
status: draft
last_verified: 2026-10-15
sources:
  - https://azure.microsoft.com/en-us/products/phi
  - https://huggingface.co/collections/microsoft/phi-4
  - https://huggingface.co/microsoft/phi-4
  - https://huggingface.co/microsoft/Phi-4-mini-instruct
  - https://huggingface.co/microsoft/Phi-4-reasoning-vision-15B
  - https://arxiv.org/abs/2503.01743
  - https://azure.microsoft.com/en-us/blog/reasoning-reimagined-introducing-phi-4-mini-flash-reasoning/
  - https://learn.microsoft.com/en-us/windows/ai/foundry-local/get-started
  - https://learn.microsoft.com/en-us/azure/foundry-local/how-to/how-to-integrate-with-inference-sdks
---

# Microsoft Phi small models

## Summary
Phi is Microsoft's family of small open-weight language models (about 3.8B to
15B parameters), trained heavily on curated and synthetic data to punch above
their size on reasoning. Microsoft releases them under the MIT licence and
targets on-device and edge use, with ONNX builds and a local runtime
(Foundry Local). The current generation is Phi-4.

## Key concepts
- Licence: Microsoft's Phi page says Phi models are open source under MIT;
  the model cards checked (phi-4, Phi-4-mini-instruct,
  Phi-4-reasoning-vision-15B, and the Phi-4-multimodal ONNX build) say MIT.
- Small models trade breadth of knowledge for size: strong at reasoning and
  instruction following for their size, weaker at world facts. Pair with
  retrieval for factual tasks ([[rag-basics]]).
- On-device paths: ONNX Runtime builds on Hugging Face, Foundry Local (local
  service with an OpenAI-compatible endpoint), Ollama/llama.cpp GGUF builds.
- Phi Silica is a separate Windows-built-in model behind a Limited Access
  Feature token; Microsoft Learn says it is due to be replaced by a model that
  does not need the token. It is not the downloadable Phi family.

Phi-4 family as of 2026-10-15 (sources: Hugging Face model cards and the
microsoft/phi-4 collection, Microsoft blogs):

| Model | Hugging Face repo | Params | Notes |
|---|---|---|---|
| Phi-4 | `microsoft/phi-4` | 14B dense | text; released 2024-12-12 |
| Phi-4-mini | `microsoft/Phi-4-mini-instruct` | 3.8B | 128K context, 200K vocab, function calling |
| Phi-4-multimodal | `microsoft/Phi-4-multimodal-instruct` | 5.6B | text + image + speech input |
| Phi-4-reasoning / -plus | `microsoft/Phi-4-reasoning`, `microsoft/Phi-4-reasoning-plus` | 14B | reasoning-tuned Phi-4 |
| Phi-4-mini-reasoning | `microsoft/Phi-4-mini-reasoning` | 3.8B | math-focused reasoning |
| Phi-4-mini-flash-reasoning | `microsoft/Phi-4-mini-flash-reasoning` | 3.8B | hybrid architecture for faster long generations |
| Phi-4-reasoning-vision | `microsoft/Phi-4-reasoning-vision-15B` | 15B | text + image in, reasoning on/off at runtime; 2026-03 |

Foundry Local's alias for the default Phi model is `phi-4-mini`. I found no
Phi-5 release as of this date.

## When to use / scenarios
- Offline or privacy-bound assistants: field-service laptops, clinic tablets,
  air-gapped plants ([[edge-on-device]]).
- Commercial products that embed a model and need a simple permissive licence
  (MIT) ([[model-licenses]]).
- Narrow high-volume tasks (intent routing, extraction, tagging) after LoRA
  fine-tuning ([[fine-tuning-and-peft]]).
- Math and logic tutoring offline with the reasoning variants.
- Not for: broad factual Q&A without retrieval, long agentic coding, or many
  languages at high quality; use a larger model ([[model-selection]]). For
  other small families see [[small-language-models]] and [[google-gemma]].

## Setup & code
Fit check before you download (runs offline). Weight memory is roughly
params x bits / 8; add KV cache and runtime overhead on top:
```python
MODELS = {"Phi-4-mini (3.8B)": 3.8e9, "Phi-4-multimodal (5.6B)": 5.6e9,
          "Phi-4 (14B)": 14e9, "Phi-4-reasoning-vision (15B)": 15e9}

def weight_gb(params, bits):
    return params * bits / 8 / 1e9

for name, p in MODELS.items():
    row = ", ".join(f"{b}-bit {weight_gb(p, b):.1f} GB" for b in (16, 8, 4))
    print(f"{name}: {row}")
```

Local run with Foundry Local (Windows/macOS) through its OpenAI-compatible
endpoint. The SDK API changed between versions; this follows the legacy
`foundry_local` package documented on Microsoft Learn, so check the current
quickstart if it fails:
```bash
pip install foundry-local-sdk openai
```
<!-- skip-check: downloads a model and starts a local service -->
```python
import openai
from foundry_local import FoundryLocalManager

alias = "phi-4-mini"
manager = FoundryLocalManager(alias)  # starts the service, downloads and loads the model
client = openai.OpenAI(base_url=manager.endpoint, api_key=manager.api_key)
r = client.chat.completions.create(
    model=manager.get_model_info(alias).id,
    messages=[{"role": "user", "content": "Extract the date: 'Service visit booked for 3 March'"}],
)
print(r.choices[0].message.content)
```
With Hugging Face Transformers, load `microsoft/Phi-4-mini-instruct` as a
causal LM ([[huggingface-transformers]]); for GGUF runtimes see [[ollama]] and
[[llama-cpp-gguf]].

## Choosing / trade-offs
- 3.8B mini for phones/laptops and CPU; 14B Phi-4 / reasoning when a single
  consumer GPU is available and quality matters more.
- Reasoning variants produce long thinking traces: better answers on math,
  more tokens and latency. Flash-reasoning targets that latency.
- Multimodal (5.6B) when you need speech or image input on-device; otherwise
  the text mini is smaller and simpler.
- Phi vs Gemma vs Qwen small models: all permissive; pick by evals on your task
  and by runtime support on your target hardware.

## Gotchas
- Small models hallucinate facts confidently; ground them and constrain output
  ([[structured-output]]).
- Quoted context (128K) is a maximum; KV-cache memory on a device usually
  limits you far below it ([[kv-cache-from-scratch-numpy]]).
- Community GGUF/quantized repos may carry their own licence notes and may not
  match the official chat template; prefer Microsoft's ONNX builds or
  well-known converters.
- MIT covers the weights; dependencies (runtime, tokenizer libs) have their own
  licences, and you still own output safety ([[guardrails-and-safety]]).
- Foundry Local SDK package and API names changed across versions; pin the
  version you test with.

## Related
- [[small-language-models]] - Phi alongside other small model families.
- [[google-gemma]] - the main open-weight alternative at similar sizes.
- [[edge-on-device]] - ONNX, mobile and edge deployment.
- [[quantization]] - shrinking weights to fit devices.
- [[model-licenses]] - why MIT matters for shipped products.
- [[reasoning-models]] - what the reasoning variants change.

## References
- https://azure.microsoft.com/en-us/products/phi
- https://huggingface.co/collections/microsoft/phi-4
- https://huggingface.co/microsoft/phi-4
- https://huggingface.co/microsoft/Phi-4-mini-instruct
- https://huggingface.co/microsoft/Phi-4-reasoning-vision-15B
- Microsoft, "Phi-4-Mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture-of-LoRAs" (2025), https://arxiv.org/abs/2503.01743
- https://azure.microsoft.com/en-us/blog/reasoning-reimagined-introducing-phi-4-mini-flash-reasoning/
- https://learn.microsoft.com/en-us/windows/ai/foundry-local/get-started
- https://learn.microsoft.com/en-us/azure/foundry-local/how-to/how-to-integrate-with-inference-sdks
- https://learn.microsoft.com/en-us/azure/foundry-local/reference/reference-sdk-legacy
