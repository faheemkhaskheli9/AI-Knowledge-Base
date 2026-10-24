---
title: LM Studio and Jan (desktop apps for local LLMs)
category: setup
tags: [lm-studio, jan, local-llm, desktop-app, openai-compatible, gguf, mlx, llama.cpp]
use_cases:
  - "chat with an open-weight model on my laptop without using the command line"
  - "point an app written for the OpenAI SDK at a model running on my own machine"
  - "let non-technical staff try local models with a GUI before we commit to a server"
  - "compare a few quantized models side by side on my own hardware"
status: draft
last_verified: 2026-10-15
sources:
  - https://lmstudio.ai/docs/developer
  - https://www.lmstudio.ai/docs/api/openai-api
  - https://lmstudio.ai/changelog/lmstudio-v0.3.29
  - https://jan.ai/docs/desktop/api-server
  - https://github.com/janhq/jan
---

# LM Studio and Jan (desktop apps for local LLMs)

## Summary
LM Studio and Jan are desktop apps (Windows, macOS, Linux) that download quantized open-weight models, run them on your CPU/GPU and give you a chat window. Both can also start a local HTTP server that speaks the OpenAI API, so code written for the OpenAI SDK works by changing `base_url`. Use them for GUI-first local experimentation; use [[ollama]] or [[llama-cpp-gguf]] for scripted setups and [[vllm-local-setup]] for GPU throughput.

## Key concepts
- Engines: both run GGUF models through llama.cpp; LM Studio also runs MLX models on Apple Silicon. Model files come from Hugging Face, searched inside the app.
- Local server:
  - LM Studio: start it from the Developer tab or headless with `lms server start`; it listens on `127.0.0.1:1234` by default, base URL `http://localhost:1234/v1`.
  - Jan: Settings > Local API Server > Start Server; default `http://127.0.0.1:1337/v1`. Recent builds ask you to set an API key that clients send as `Authorization: Bearer <key>`.
- OpenAI-compatible routes in LM Studio's docs: `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, and `/v1/responses` (added in 0.3.29). LM Studio also has its own REST API and SDKs; see its developer docs.
- Model id: the `model` field must match an id returned by `GET /v1/models` (the app's identifier, not necessarily the Hugging Face repo name).
- Licensing: Jan is open source (see the repo's LICENSE); LM Studio is a free but closed-source app with its own terms. Model weights keep their own licences ([[model-licenses]]).

## When to use / scenarios
- A consultant demos a private document assistant to a law firm on a laptop, no cloud calls ([[legal]]).
- A developer tests whether a 4-bit 8B model is good enough before paying for a hosted API ([[small-language-models]]).
- Office staff want a ChatGPT-like window that works offline.
- Not for: shared multi-user serving or CI (headless servers are simpler: [[ollama]], [[inference-servers-vllm]]); exact control over quantization or build flags ([[llama-cpp-gguf]]).

## Setup & code
Install from the official sites (lmstudio.ai, jan.ai), download a model in the app, then start the server.

LM Studio headless (the `lms` CLI ships with the app):
```bash
lms server start            # optional: --port 1234
curl http://localhost:1234/v1/models
```
Jan: enable the server in Settings > Local API Server, then:
```bash
curl http://127.0.0.1:1337/v1/models -H "Authorization: Bearer <your-jan-key>"
```
OpenAI SDK against either server (the key is ignored by LM Studio unless you enabled auth):
<!-- skip-check: needs a running LM Studio or Jan server and the openai package -->
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")  # Jan: :1337, your key
model = client.models.list().data[0].id
r = client.chat.completions.create(model=model,
                                   messages=[{"role": "user", "content": "One-line haiku about RAM"}])
print(r.choices[0].message.content)
```
What the wire protocol looks like, runnable with the stdlib only: a fake OpenAI-compatible server on a free local port and a raw HTTP client. Swapping the fake for LM Studio or Jan changes only host and port.
```python
import http.client, json, threading
from http.server import BaseHTTPRequestHandler, HTTPServer

class FakeOpenAI(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        if self.path == "/v1/models":
            self._send({"object": "list", "data": [{"id": "local-model", "object": "model"}]})
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        last = req["messages"][-1]["content"]
        self._send({"object": "chat.completion", "model": req["model"],
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": f"echo: {last}"}}]})

srv = HTTPServer(("127.0.0.1", 0), FakeOpenAI)
threading.Thread(target=srv.serve_forever, daemon=True).start()
host, port = srv.server_address

def call(method, path, payload=None):
    c = http.client.HTTPConnection(host, port, timeout=5)
    c.request(method, path, body=json.dumps(payload) if payload else None,
              headers={"Content-Type": "application/json", "Authorization": "Bearer dummy"})
    return json.loads(c.getresponse().read())

model = call("GET", "/v1/models")["data"][0]["id"]
out = call("POST", "/v1/chat/completions",
           {"model": model, "messages": [{"role": "user", "content": "hello"}]})
print(model, "->", out["choices"][0]["message"]["content"])
srv.shutdown()
```

## Choosing / trade-offs
| Need | Pick |
|---|---|
| Polished GUI, MLX on a Mac, model search and per-model settings | LM Studio |
| Open-source app you can audit or fork, assistants plus optional cloud providers in one UI | Jan |
| Scriptable service, Docker, CI, LAN sharing with less clicking | [[ollama]] |
| Raw flags, custom quantization, newest llama.cpp features | [[llama-cpp-gguf]] |
| Many concurrent users on NVIDIA GPUs | [[vllm-local-setup]] then [[inference-servers-vllm]] |

- GUI apps make the context length, GPU offload layers and quantization visible per model, which is good for learning, but settings live in the app rather than in version control.
- A desktop app runs only while a user is logged in; for an always-on endpoint use a service.

## Gotchas
- Context length is a per-model load setting; long prompts silently truncate or fail if it is left low. Raising it costs memory ([[long-context]]).
- "Serve on local network" (LM Studio) or a non-127.0.0.1 host (Jan) exposes the API to the LAN; add auth and a firewall rule, never port-forward it to the internet ([[ai-security-privacy-compliance]]).
- The `model` string must match `/v1/models`; copying a Hugging Face repo id often fails.
- OpenAI compatibility is partial: tool calling, JSON mode and vision depend on the loaded model and app version. Test the exact feature you need.
- If a model does not fit in VRAM, layers fall back to the CPU and generation slows sharply; pick a smaller quant ([[quantization]], [[gpu-cuda-setup]]).
- Ports differ (1234 vs 1337 vs Ollama's 11434); a "connection refused" is usually the wrong port or the server toggle being off.

## Related
- [[ollama]] - CLI/service alternative with the same OpenAI-style surface.
- [[llama-cpp-gguf]] - the engine both apps use for GGUF models.
- [[vllm-local-setup]] - GPU server when one user is no longer enough.
- [[docker-for-ml]] - running local servers in containers.
- [[comfyui]] - the equivalent GUI route for local image generation.
- [[api-sdk-setup]] - OpenAI SDK basics reused with a local `base_url`.
- [[small-language-models]] - models that fit desktop hardware.
- [[quantization]] - what the GGUF quant names mean.
- [[open-webui]] - self-hosted web UI when a whole team needs access.

## References
- LM Studio developer docs: https://lmstudio.ai/docs/developer
- LM Studio OpenAI compatibility: https://www.lmstudio.ai/docs/api/openai-api
- LM Studio 0.3.29 changelog (Responses API): https://lmstudio.ai/changelog/lmstudio-v0.3.29
- Jan local API server: https://jan.ai/docs/desktop/api-server
- Jan source: https://github.com/janhq/jan
