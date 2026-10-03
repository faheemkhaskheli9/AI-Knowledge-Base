---
title: Hugging Face Hub and Transformers
category: setup
tags: [huggingface, transformers, hub, pipeline, safetensors, cache, hf-cli, auth]
use_cases:
  - "download an open model from Hugging Face and run it locally in Python"
  - "run a quick sentiment/NER/summarization baseline with a pipeline"
  - "access a gated model (e.g. Llama) with an access token"
  - "move the model cache to a bigger drive or make a machine work offline"
status: draft
last_verified: 2026-10-03
sources:
  - https://huggingface.co/docs/huggingface_hub/guides/cli
  - https://huggingface.co/docs/transformers/
  - https://huggingface.co/docs/huggingface_hub/guides/manage-cache
---

# Hugging Face Hub and Transformers

## Summary
The Hugging Face Hub hosts models, datasets and demo Spaces; the `transformers` library loads those models in PyTorch with a uniform API (tokenizers, models, pipelines, generation). It is the default way to obtain and run open-weight models outside of GGUF/Ollama runtimes, and the base for fine-tuning with PEFT/TRL.

## Key concepts
- Repo id `org/name` identifies a model; revisions (branch, tag, commit hash) pin versions.
- `AutoTokenizer` / `AutoModelForCausalLM` (and other Auto classes) pick the architecture from the repo config.
- `pipeline(task, model=...)` wraps tokenizer + model + post-processing for common tasks.
- Weights are usually `.safetensors` (safe to load, unlike pickle `.bin`).
- Gated models require accepting a license on the model page, then authenticating with a token.
- Cache: downloads are stored once, deduplicated, under the Hub cache directory.

## When to use / scenarios
- Baselines and prototypes: classification, NER, embeddings, speech, vision pipelines.
- Loading a chat LLM for experiments, with `device_map="auto"` and 4/8-bit quantization.
- Pre-downloading models into a Docker image or an air-gapped server.
- For production LLM serving prefer [[inference-servers-vllm]]; for CPU/laptop use [[llama-cpp-gguf]] or [[ollama]].

## Setup & code
```bash
uv pip install transformers accelerate huggingface_hub   # plus torch, see gpu-cuda-setup
hf auth login          # paste a token from huggingface.co/settings/tokens
hf auth whoami
hf download <org>/<model> --local-dir ./models/m   # prefetch for offline use
hf cache ls
```
(`hf` ships with `huggingface_hub`; standalone installers exist at https://hf.co/cli/install.sh and install.ps1. `hf download --dry-run` previews.)

Pipeline:
```python
from transformers import pipeline
clf = pipeline("sentiment-analysis")           # default model; pin one in real code
print(clf("The onboarding flow was painless."))
```
Chat/causal LLM:
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
mid = "<org>/<instruct-model>"
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16, device_map="auto")
msgs = [{"role": "user", "content": "Explain KV cache in two sentences."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=128)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
```
Environment variables:
```bash
export HF_TOKEN=...          # token for non-interactive auth (CI); never commit it
export HF_HOME=/data/hf      # relocate cache root
export HF_HUB_OFFLINE=1      # use only cached files
```
Windows PowerShell: `$env:HF_HOME = "D:\hf"` (persist with `setx`).

## Choosing / trade-offs
- transformers vs vLLM/TGI: transformers is flexible and good for research and small batch; servers win on throughput.
- `bfloat16` (Ampere+) vs `float16`: bf16 is more stable; older GPUs need fp16.
- Quantized loading (bitsandbytes, etc.) saves VRAM but adds dependencies and slows some paths; see [[quantization]].
- Pin `revision=` for reproducibility vs tracking `main` for updates.

## Gotchas
- Default cache is under the user home directory; large models fill the system drive, so set `HF_HOME` early.
- On Windows the cache uses symlinks when allowed, otherwise copies (more disk); enabling Developer Mode helps.
- Pipelines without an explicit `model=` download a default that can change; pin it.
- `trust_remote_code=True` executes code from the repo; only use for repos you have reviewed.
- Forgetting `apply_chat_template` (or using the wrong template) quietly degrades instruct models.
- Gated-model 401/403: accept the license on the web page with the same account as the token.

## Related
- [[python-env-uv]] - installing torch/transformers.
- [[gpu-cuda-setup]] - memory sizing and CUDA.
- [[fine-tuning-and-peft]] - training on top of Hub models.
- [[tokenization]] - what tokenizers do.
- [[model-licenses]] - gated and restricted licenses.
- [[embedding-models]] - sentence-transformers via the Hub.

## References
- Hub CLI: https://huggingface.co/docs/huggingface_hub/guides/cli
- Transformers docs: https://huggingface.co/docs/transformers/
- Cache guide: https://huggingface.co/docs/huggingface_hub/guides/manage-cache
