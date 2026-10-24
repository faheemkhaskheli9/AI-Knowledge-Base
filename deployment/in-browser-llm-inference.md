---
title: In-browser inference (WebLLM, Transformers.js, ONNX Runtime Web, WebGPU)
category: deployment
tags: [webgpu, wasm, webllm, mlc-llm, transformers-js, onnxruntime-web, browser, client-side, on-device, privacy, web-worker, cross-origin-isolation, prompt-api]
use_cases:
  - "run a small LLM or embedding model entirely in the user's browser so text never leaves the device"
  - "add offline semantic search or classification to a web app without paying for inference servers"
  - "ship a browser extension that summarizes pages with a local model"
  - "decide between in-browser inference, a local runner like Ollama and a server API for a web feature"
status: draft
last_verified: 2026-10-24
sources:
  - https://github.com/mlc-ai/web-llm
  - https://github.com/huggingface/transformers.js
  - https://github.com/huggingface/transformers.js/blob/main/packages/transformers/docs/source/guides/webgpu.md
  - https://github.com/huggingface/transformers.js/blob/main/packages/transformers/docs/source/guides/dtypes.md
  - https://github.com/microsoft/onnxruntime/tree/main/js/web
  - https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API
  - https://github.com/webmachinelearning/prompt-api
---

# In-browser inference (WebLLM, Transformers.js, ONNX Runtime Web, WebGPU)

## Summary
Models can run inside a web page: the browser downloads quantized weights once, caches them, and executes them on the user's GPU through **WebGPU** or on the CPU through **WebAssembly**. **WebLLM** (MLC) runs chat LLMs compiled to MLC format behind an OpenAI-style `engine.chat.completions.create` API; **Transformers.js** (Hugging Face) runs ONNX exports of transformers models (text, vision, audio, embeddings, small LLMs) with a `pipeline()` API like the Python library; both sit on lower-level runtimes (TVM/WebGPU for WebLLM, **ONNX Runtime Web** for Transformers.js). The payoffs are privacy, zero server cost and offline use; the costs are a large first download, uneven device support and small models only.

## Key concepts
- **WebGPU vs WASM**: WebGPU gives GPU compute from JavaScript and is available only in secure contexts (HTTPS or localhost) and from workers via `WorkerNavigator.gpu`. WASM runs on the CPU everywhere; multi-threaded WASM needs `SharedArrayBuffer`, which needs a **cross-origin isolated** page (`Cross-Origin-Opener-Policy: same-origin` plus `Cross-Origin-Embedder-Policy: require-corp` or `credentialless`), otherwise ONNX Runtime Web falls back to one thread.
- **Support is uneven**: ONNX Runtime Web's compatibility table lists its WebGPU backend for Chrome/Edge on Windows, Android and macOS, and not for Safari, iOS or Firefox at the time of writing; Transformers.js's WebGPU guide says some browsers still need a flag. Feature-detect `navigator.gpu` and fall back to WASM or a server.
- **WebLLM**: `CreateMLCEngine(modelId, {initProgressCallback})` downloads and loads a prebuilt model (`prebuiltAppConfig.model_list` lists them, with `vram_required_MB`, `low_resource_required` and `required_features` such as `shader-f16`); custom models are compiled with MLC LLM into weights plus a `model_lib` WASM. It supports streaming, JSON mode and seeds; its README marks function calling as work in progress. The `model` field in a request is ignored: switch models with `engine.reload()`.
- **Transformers.js**: `pipeline(task, modelId, {device: "webgpu", dtype: "q4"})`; default device in the browser is WASM (CPU), default `dtype` is `fp32` on WebGPU and `q8` on WASM; other dtypes include `fp16`, `q4` and `q4f16`, depending on which ONNX files the model repo ships. Models come from Hugging Face repos tagged `transformers.js` or from your own path (`env.localModelPath`, `env.allowRemoteModels = false`).
- **Caching**: weights are stored per origin (WebLLM: Cache API by default, or IndexedDB, OPFS); the browser can evict them under storage pressure, and a different origin downloads again.
- **Workers**: run the engine in a Web Worker (WebLLM `WebWorkerMLCEngineHandler` / `CreateWebWorkerMLCEngine`) so the UI thread does not freeze; a Service Worker can keep the model loaded across page visits but the browser may kill it at any time.
- **Integrity**: WebLLM can verify model config, WASM and tokenizer files against SRI hashes (`integrity` on a model record). Pin model revisions and self-host artifacts for production.
- **Browser-provided models**: the W3C Web Machine Learning Community Group's Prompt API explainer proposes a JavaScript API to a model shipped by the browser or OS, experimentally available in Chrome and Edge per the explainer. No download for you, but no choice of model and not cross-browser.

## When to use / scenarios
- Privacy-sensitive text the user does not want to upload: drafting, redaction, classification or search over local notes, mail or documents ([[pii-redaction-and-pseudonymization]], [[ai-security-privacy-compliance]]).
- Free-tier or high-volume features where per-request server cost dominates: embeddings for client-side semantic search, zero-shot classification, speech-to-text of short clips ([[embedding-models]], [[speech-to-text]], [[low-budget-ai]]).
- Offline-capable apps and browser extensions (summarize the current page, rewrite a selection) ([[small-language-models]]).
- Demos and education pages that must run without a backend.
- Not for large models or long contexts: the user's GPU memory, download size and battery set the ceiling; use a server ([[inference-servers-vllm]]) or an API ([[model-selection]]).
- Not when you need consistent latency or quality across users; device spread is huge. A desktop runner such as [[ollama]] or [[lm-studio-and-jan]] fits power users on one machine; native mobile apps fit [[edge-on-device]].

## Setup & code
Install the JavaScript packages (or import from a CDN in a plain page):
```bash
npm install @mlc-ai/web-llm            # chat LLMs on WebGPU
npm install @huggingface/transformers  # pipelines on WebGPU or WASM via ONNX Runtime Web
```
WebLLM chat with streaming, inside a worker-friendly module (API from the WebLLM README; pick an ID from `prebuiltAppConfig.model_list`):
```javascript
import { CreateMLCEngine, prebuiltAppConfig } from "@mlc-ai/web-llm";

if (!navigator.gpu) throw new Error("WebGPU unavailable: fall back to WASM or the server API");
const small = prebuiltAppConfig.model_list.filter((m) => m.low_resource_required);
const engine = await CreateMLCEngine(small[0].model_id, {
  initProgressCallback: (p) => console.log(p.text),        // first load downloads the weights
});
const chunks = await engine.chat.completions.create({
  messages: [{ role: "user", content: "Summarize: ..." }],
  stream: true,
  stream_options: { include_usage: true },
});
let reply = "";
for await (const chunk of chunks) reply += chunk.choices[0]?.delta.content || "";
```
Transformers.js embeddings on WebGPU, falling back to WASM (from the Transformers.js WebGPU and dtype guides):
```javascript
import { pipeline } from "@huggingface/transformers";

const device = navigator.gpu ? "webgpu" : "wasm";
const extractor = await pipeline("feature-extraction", "mixedbread-ai/mxbai-embed-xsmall-v1", { device });
const vectors = await extractor(["refund for damaged item", "shipping times"], { pooling: "mean", normalize: true });
console.log(vectors.tolist());
```
Runnable, stdlib only: a local dev server that sends the headers in-browser inference needs (cross-origin isolation for multi-threaded WASM, `application/wasm` MIME type, long caching for immutable model shards), checked with a real request; and a pre-flight check that picks a model tier for a client from the capabilities the page reports (WebGPU present, `shader-f16`, storage quota estimate). The model catalogue values are placeholders: read real sizes and requirements from WebLLM's `model_list` or the model repo.
```python
import http.client
import http.server
import json
import threading
from pathlib import Path
import tempfile

ISOLATION = {"Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp"}

class ModelAppHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".wasm": "application/wasm", ".mjs": "text/javascript", ".onnx": "application/octet-stream"}

    def end_headers(self):
        for k, v in ISOLATION.items():
            self.send_header(k, v)
        if self.path.startswith("/models/"):           # versioned, immutable shards: let the browser keep them
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        else:
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, *args):
        pass

root = Path(tempfile.mkdtemp())
(root / "models").mkdir()
(root / "index.html").write_text("<script type=module>console.log(self.crossOriginIsolated)</script>")
(root / "models" / "shard-00001.bin").write_bytes(b"\0" * 16)
(root / "ort.wasm").write_bytes(b"\0asm\1\0\0\0")

handler = lambda *a, **kw: ModelAppHandler(*a, directory=str(root), **kw)
server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
for path in ["/index.html", "/ort.wasm", "/models/shard-00001.bin"]:
    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1])   # local loopback only
    conn.request("GET", path)
    r = conn.getresponse()
    r.read()
    h = r.headers
    print(f"{path:26s} {h['Content-Type']:24s} COOP={h['Cross-Origin-Opener-Policy']} "
          f"COEP={h['Cross-Origin-Embedder-Policy']} cache={h['Cache-Control']}")
    assert h["Cross-Origin-Opener-Policy"] == "same-origin"
    conn.close()
server.shutdown()

# Pre-flight: the page posts what it detected; pick the largest tier that fits, else fall back.
CATALOGUE = [  # placeholder numbers for the demo, not measured model sizes
    {"tier": "chat-3b-q4f16", "download_mb": 2000, "vram_mb": 3000, "features": ["shader-f16"], "backend": "webgpu"},
    {"tier": "chat-1b-q4f32", "download_mb": 900, "vram_mb": 1200, "features": [], "backend": "webgpu"},
    {"tier": "embed-small-q8", "download_mb": 40, "vram_mb": 0, "features": [], "backend": "wasm"},
]

def choose(caps, catalogue=CATALOGUE, headroom=0.5):
    """caps: {'webgpu': bool, 'features': [...], 'quota_mb': navigator.storage.estimate() quota - usage,
              'gpu_budget_mb': app's own estimate or a user setting}"""
    for m in catalogue:
        if m["backend"] == "webgpu" and not caps["webgpu"]:
            continue
        if any(f not in caps["features"] for f in m["features"]):
            continue
        if m["download_mb"] > caps["quota_mb"] * headroom:     # leave room so the browser does not evict it
            continue
        if m["vram_mb"] > caps.get("gpu_budget_mb", 0) and m["backend"] == "webgpu":
            continue
        return m["tier"]
    return "server-api"

clients = {
    "desktop chrome, discrete GPU": {"webgpu": True, "features": ["shader-f16"], "quota_mb": 20000, "gpu_budget_mb": 6000},
    "laptop, no f16":               {"webgpu": True, "features": [], "quota_mb": 8000, "gpu_budget_mb": 2000},
    "browser without WebGPU":       {"webgpu": False, "features": [], "quota_mb": 4000},
    "private window, tiny quota":   {"webgpu": True, "features": ["shader-f16"], "quota_mb": 50, "gpu_budget_mb": 4000},
}
print(json.dumps({name: choose(c) for name, c in clients.items()}, indent=2))
```
In the page, the inputs come from `navigator.gpu` (and `adapter.features.has("shader-f16")` on the adapter from `navigator.gpu.requestAdapter()`), `navigator.storage.estimate()`, and `self.crossOriginIsolated` to confirm the headers took effect.

## Choosing / trade-offs
| Option | Runs | Good for | Watch out for |
|---|---|---|---|
| WebLLM | MLC-compiled LLMs on WebGPU | Chat, streaming, JSON mode with an OpenAI-like API | WebGPU required; custom models need MLC compilation; function calling marked WIP |
| Transformers.js | ONNX models on WebGPU or WASM | Embeddings, classification, NER, Whisper speech-to-text, vision, small LLMs | Model must have an ONNX export in the right dtype; WebGPU support varies |
| ONNX Runtime Web directly | Any ONNX graph | Your own non-transformers models (detectors, tabular) | You write pre/post-processing; WebGPU operator coverage is narrower than WASM |
| Browser built-in model (Prompt API) | The browser's own model | No download, tiny pages | Experimental, browser-specific, no model choice |
| Server or hosted API | Your GPUs or a provider | Large models, consistent latency and quality | Cost per request, data leaves the device |

- **Download vs quality**: every user pays the download once per origin and again after eviction; a 4-bit 1B-class chat model is a large download compared with a web page, and small models answer worse than API models. Use the browser for narrow tasks (embed, classify, rewrite) and keep open-ended chat on a server unless privacy forces it.
- **Hybrid**: run in the browser when the device qualifies, fall back to the server otherwise, behind one interface; WebLLM's OpenAI-style API makes the client code the same shape as a server call ([[llm-gateways-and-routing]]).
- **WebGPU vs WASM**: WebGPU for anything generative or larger than an encoder; quantized WASM (`q8`) is often enough for small encoders and works in more browsers.

## Gotchas
- **No secure context, no WebGPU**: plain `http://` on a LAN IP has no `navigator.gpu`; use HTTPS or `localhost` in development.
- **COEP breaks third-party embeds**: `require-corp` blocks cross-origin images, scripts and iframes that do not send CORP/CORS headers. Try `credentialless`, or isolate only the page that runs the model.
- **First load looks like a hang**: show download progress (`initProgressCallback`, Transformers.js `progress_callback`) and let users opt in before downloading large weights, especially on metered connections.
- **Storage eviction and private windows**: cached weights can disappear; private windows often have small quotas and clear on close. Call `navigator.storage.persist()` where appropriate and handle a re-download.
- **Main-thread freezes**: running inference on the UI thread blocks input; put the engine in a worker.
- **Tab memory limits**: a model that fits the GPU can still crash the tab; mobile browsers are the tightest. Test on the lowest device you support.
- **Model licence applies to every download**: you are distributing the weights to every visitor; check the licence permits it and show the required notices ([[model-licenses]]).
- **Client-side means user-controlled**: users can read the weights and system prompt and alter outputs. Never put secrets in the prompt or trust client-side moderation for anything that matters; enforce policy on the server ([[guardrails-and-safety]], [[prompt-injection]]).
- **Supply chain**: loading weights and WASM from third-party CDNs at runtime means they can change under you; pin versions, self-host artifacts and use WebLLM's `integrity` hashes.

## Related
- [[edge-on-device]] - native on-device runtimes (ONNX Runtime, Core ML, TensorRT, mobile).
- [[small-language-models]] - the model sizes that fit a browser.
- [[quantization]] - why 4-bit and 8-bit weights make browser inference possible.
- [[onnx-and-triton-inference-server]] - exporting to ONNX, the format Transformers.js runs.
- [[embedding-models]] - choosing a small embedder for client-side search.
- [[ollama]] - local runner for users who can install software.
- [[streaming-and-async-llm-apps]] - streaming UX that also applies to local generation.
- [[ai-security-privacy-compliance]] - when keeping data on the device is the requirement.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/frontend/web-performance.md - download size, caching and main-thread work in web apps.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/frontend/offline-first-and-service-workers.md - service workers, storage quotas and offline caching.

## References
- WebLLM README (CreateMLCEngine, streaming, workers, cache backends, integrity, custom models): https://github.com/mlc-ai/web-llm
- WebLLM model list and `ModelRecord` fields (`vram_required_MB`, `low_resource_required`, `required_features`): https://github.com/mlc-ai/web-llm/blob/main/src/config.ts
- Transformers.js README (pipeline API, `device`, `dtype`, `env` settings): https://github.com/huggingface/transformers.js
- Transformers.js WebGPU guide: https://github.com/huggingface/transformers.js/blob/main/packages/transformers/docs/source/guides/webgpu.md
- Transformers.js dtypes guide: https://github.com/huggingface/transformers.js/blob/main/packages/transformers/docs/source/guides/dtypes.md
- ONNX Runtime Web README (execution providers, browser compatibility table): https://github.com/microsoft/onnxruntime/tree/main/js/web
- MDN, WebGPU API (secure context, workers, adapters): https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API
- MDN, `crossOriginIsolated` (COOP/COEP headers, SharedArrayBuffer): https://developer.mozilla.org/en-US/docs/Web/API/Window/crossOriginIsolated
- Prompt API explainer, W3C Web Machine Learning Community Group: https://github.com/webmachinelearning/prompt-api
