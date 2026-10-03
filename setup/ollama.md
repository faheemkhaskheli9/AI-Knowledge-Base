---
title: Ollama (local LLM runner)
category: setup
tags: [ollama, local-llm, gguf, openai-compatible, modelfile, llama.cpp]
use_cases:
  - "run an open-weight LLM locally with one command"
  - "give my app a local OpenAI-style endpoint so no data leaves the machine"
  - "prototype a RAG app offline with local chat and embedding models"
  - "set up a private assistant for a small office on one shared machine"
status: draft
last_verified: 2026-10-03
sources:
  - https://github.com/ollama/ollama/blob/main/docs/api.md
  - https://docs.ollama.com/
  - https://github.com/ollama/ollama
---

# Ollama (local LLM runner)

## Summary
Ollama downloads, manages and serves open-weight models behind a local HTTP server (default port 11434) with a simple CLI. It wraps llama.cpp-style runtimes and handles quantized model files, GPU offload and model lifecycle for you. It is the fastest route to a private, local model for development and light production.

## Key concepts
- Model tags: `name:size-quant` pulled from the Ollama library (`ollama pull`); weights are quantized GGUF-like blobs.
- Server: background service exposing a native API (`/api/generate`, `/api/chat`, `/api/embed`, `/api/tags`, ...) plus an OpenAI-compatible surface under `/v1`.
- Models load into memory on first request and unload after an idle period (default keep-alive 5 minutes, `OLLAMA_KEEP_ALIVE`).
- Modelfile: declarative recipe (base model, system prompt, parameters) built with `ollama create`.
- Env vars: `OLLAMA_HOST` (bind address), `OLLAMA_MODELS` (storage dir), `OLLAMA_KEEP_ALIVE`.

## When to use / scenarios
- Laptop/desktop prototyping without API costs.
- Privacy-sensitive data (clinic notes, legal drafts) that must stay on-prem; see [[ai-security-privacy-compliance]].
- Offline demos, CI tests against a small local model.
- Not for high-concurrency production serving: use [[inference-servers-vllm]]. Not for fine-grained quantization control: use [[llama-cpp-gguf]].

## Setup & code
Install: download the installer from https://ollama.com/download (Windows, macOS) or follow the Linux instructions at the same page. Then:
```bash
ollama pull <model-tag>        # tags listed at https://ollama.com/library
ollama run <model-tag>         # interactive chat
ollama list                    # downloaded models
ollama ps                      # loaded models
```
Native API:
```bash
curl http://localhost:11434/api/chat -d '{
  "model": "<model-tag>",
  "messages": [{"role": "user", "content": "Say hi"}],
  "stream": false
}'
```
OpenAI-compatible client (the key is required by the SDK but ignored):
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
r = client.chat.completions.create(
    model="<model-tag>",
    messages=[{"role": "user", "content": "Say hi"}],
)
print(r.choices[0].message.content)
```
Modelfile:
```
FROM <model-tag>
SYSTEM You are a concise assistant.
PARAMETER temperature 0.2
```
```bash
ollama create concise -f Modelfile
```
Expose on the LAN (Linux/macOS `export`, Windows `setx` then restart the service):
```bash
export OLLAMA_HOST=0.0.0.0:11434
```

## Choosing / trade-offs
- Ollama (ease) vs llama.cpp directly (flags, custom quants, newest features first) vs LM Studio (GUI).
- Ollama vs vLLM: Ollama is single-user friendly and runs on CPU/Apple Silicon; vLLM maximizes GPU throughput under many concurrent requests.
- Quant level in the tag trades memory for quality; pick the largest that fits ([[gpu-cuda-setup]]).

## Gotchas
- Default context window is modest regardless of the model's maximum; raise `num_ctx` (options or Modelfile `PARAMETER num_ctx`) for long prompts, which increases memory.
- Binding to 0.0.0.0 exposes an unauthenticated API; put it behind a reverse proxy/firewall.
- Streaming is on by default for the native API; set `"stream": false` for a single JSON.
- If a model does not fit in VRAM it silently spills to CPU and gets slow; check `ollama ps`.
- Tool calling and structured output depend on the specific model's support.

## Related
- [[llama-cpp-gguf]] - the underlying runtime and GGUF files.
- [[small-language-models]] - models that suit local hardware.
- [[inference-servers-vllm]] - when you outgrow it.
- [[quantization]] - what the tags mean.
- [[api-sdk-setup]] - OpenAI SDK usage.

## References
- API docs: https://github.com/ollama/ollama/blob/main/docs/api.md
- https://docs.ollama.com/
