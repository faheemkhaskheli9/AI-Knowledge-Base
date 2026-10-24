---
title: Python model-serving frameworks (Ray Serve, BentoML, LitServe)
category: deployment
tags: [ray-serve, bentoml, litserve, model-serving, dynamic-batching, adaptive-batching, model-composition, multiplexing, autoscaling, fractional-gpu, inference-pipeline, openai-compatible]
use_cases:
  - "serve a custom PyTorch, scikit-learn or Hugging Face model behind an HTTP API with request batching"
  - "chain several models (OCR, embedder, classifier, LLM) into one inference service and scale each step separately"
  - "pack two small models onto one GPU instead of paying for a GPU each"
  - "choose between Ray Serve, BentoML, LitServe and plain FastAPI for a model API"
  - "put an OpenAI-compatible API in front of vLLM with autoscaling across several GPU nodes"
status: draft
last_verified: 2026-10-24
sources:
  - https://github.com/ray-project/ray/blob/master/doc/source/serve/index.md
  - https://github.com/ray-project/ray/blob/master/doc/source/serve/key-concepts.md
  - https://github.com/ray-project/ray/blob/master/doc/source/serve/advanced-guides/dyn-req-batch.md
  - https://github.com/ray-project/ray/blob/master/doc/source/serve/autoscaling-guide.md
  - https://github.com/ray-project/ray/blob/master/doc/source/serve/resource-allocation.md
  - https://github.com/ray-project/ray/blob/master/doc/source/serve/model-multiplexing.md
  - https://github.com/ray-project/ray/blob/master/doc/source/serve/llm/index.md
  - https://github.com/bentoml/BentoML/blob/main/README.md
  - https://github.com/bentoml/BentoML/blob/main/docs/source/get-started/adaptive-batching.rst
  - https://github.com/bentoml/BentoML/blob/main/docs/source/build-with-bentoml/parallelize-requests.rst
  - https://github.com/bentoml/BentoML/blob/main/docs/source/get-started/model-composition.rst
  - https://github.com/Lightning-AI/LitServe/blob/main/README.md
  - https://github.com/Lightning-AI/LitServe/blob/main/src/litserve/api.py
  - https://github.com/Lightning-AI/LitServe/blob/main/src/litserve/server.py
---

# Python model-serving frameworks (Ray Serve, BentoML, LitServe)

## Summary
An LLM engine such as vLLM already batches and schedules tokens, but many production models are not LLMs (embedders, classifiers, OCR, detectors, diffusion) or are a pipeline of several models plus business logic. Ray Serve, BentoML and LitServe are Python frameworks that turn that inference code into an HTTP service and add what a hand-written FastAPI app lacks: server-side dynamic batching, worker or replica processes per device, multi-model composition, autoscaling and packaging. This topic compares the three, shows the minimal service in each, and includes a stdlib implementation of the dynamic-batching loop they all share so you can reason about its latency/throughput knobs.

## Key concepts
- **Dynamic (adaptive) batching**: requests arrive one at a time; the server queues them and calls the model with a list once either a size cap or a wait timeout is hit, then splits the outputs back per request. It raises GPU utilization for models that vectorize well, at the cost of up to one timeout of extra latency. Names per framework:
  - Ray Serve: `@serve.batch(max_batch_size=..., batch_wait_timeout_s=...)` on an `async` method that takes and returns a list. Documented defaults: `max_batch_size` 10, `batch_wait_timeout_s` 0.01 s, `max_concurrent_batches` 1. `batch_size_fn` lets the cap count tokens, pixels or graph nodes instead of requests. Both knobs can be changed at runtime (`set_max_batch_size`, `set_batch_wait_timeout_s`) from `reconfigure`.
  - BentoML: `@bentoml.api(batchable=True, max_batch_size=..., max_latency_ms=...)`; off by default. The dispatcher adapts the batch window to recent traffic and processing times while respecting both caps. Request order inside a batch is not guaranteed, and a batchable API takes exactly one parameter (group several with a Pydantic model).
  - LitServe: `LitAPI(max_batch_size=..., batch_timeout=...)` (defaults 1 and 0.0, so off); override `batch()`/`unbatch()` to stack and split tensors. `batch_timeout` must be smaller than the server's request `timeout`.
- **Replicas and workers**: Ray Serve runs each deployment as N replica actors (`num_replicas`, or `num_replicas="auto"`) placed anywhere in a Ray cluster. BentoML runs `workers` processes per service (default 1, or `"cpu_count"`), each loading its own model copy. LitServe runs `workers_per_device` processes on each of `devices` for an `accelerator` (`"auto"`, `"cuda"`, `"cpu"`, `"mps"`).
- **Resources and GPU sharing**: Ray Serve takes `ray_actor_options={"num_gpus": 0.5}`, so two replicas are scheduled onto one GPU; you must check that both models fit in its memory together. BentoML uses `resources={"gpu": N}` and `bentoml.server_context.worker_index` to pin workers to devices; LitServe assigns devices to workers itself.
- **Composition**: Ray Serve binds deployments into an application (`A.bind(B.bind())`) and calls them through a `DeploymentHandle`, each deployment scaling independently; one deployment is the HTTP ingress (optionally a FastAPI app). BentoML composes services with `bentoml.depends(OtherService)`. LitServe keeps the whole pipeline in one `LitAPI.setup()`/`predict()`.
- **Autoscaling (Ray Serve)**: `num_replicas="auto"` is documented as `target_ongoing_requests: 2`, `max_ongoing_requests: 5`, `min_replicas: 1`, `max_replicas: 100`. A hand-written `autoscaling_config` without `"auto"` has a base `max_replicas` of 1, so it does not scale until you raise it. `min_replicas: 0` allows scale to zero with cold-start latency on the first request.
- **Model multiplexing (Ray Serve)**: `@serve.multiplexed(max_num_models_per_replica=...)` loads models by ID on demand inside a shared replica pool with LRU eviction; clients pick the model with the `serve_multiplexed_model_id` header. Useful for many small per-tenant models.
- **LLM layers on top**: Ray Serve LLM (`pip install "ray[llm]"`) wraps vLLM or SGLang engines in an OpenAI-compatible app with multi-node parallelism, prefix-aware routing, LoRA and prefill/decode disaggregation. LitServe ships `OpenAISpec` and `OpenAIEmbeddingSpec` to expose an OpenAI-shaped endpoint for your own `predict`. BentoML provides vLLM example services (BentoVLLM).
- **Packaging**: BentoML builds a versioned "Bento" (`bentoml build`) and a Docker image from it (`bentoml containerize`). Ray Serve deploys from a YAML config (`serve run`/`serve deploy`) to a Ray cluster, on Kubernetes via KubeRay. LitServe is a Python script you containerize yourself (or deploy on its vendor's cloud).

## When to use / scenarios
- A non-LLM model (embedder, reranker, image classifier, ASR) needs server-side batching and you do not want to adopt Triton's model-repository format ([[onnx-and-triton-inference-server]]).
- A document pipeline (layout -> OCR -> extraction LLM -> validation) where steps need different hardware and scale separately: Ray Serve composition or BentoML `depends` ([[document-processing]]).
- Many small per-customer models that do not each deserve a GPU: Ray Serve multiplexing or fractional GPUs.
- Several vLLM replicas across nodes behind one OpenAI-compatible endpoint with autoscaling, when you already run Ray: Ray Serve LLM ([[kubernetes-for-llm-serving]]).
- Not for a single LLM on one machine: run `vllm serve` or SGLang directly ([[inference-servers-vllm]], [[sglang-and-tensorrt-llm]]); these frameworks add a hop and do not make token generation faster. LitServe's README says it is not a vLLM or Ollama alternative out of the box.
- Not needed when the model is a hosted API: a thin FastAPI app with retries is enough ([[serving-with-fastapi]], [[llm-api-rate-limits-and-retries]]).

## Setup & code
Minimal service in each framework (from their docs; model code elided). These need the framework installed and start a server, so they are not run here:
<!-- skip-check: needs ray, bentoml or litserve installed and starts a server -->
```python
# Ray Serve: pip install "ray[serve]"; run with `serve run app:app`
from typing import List
from ray import serve

@serve.deployment(ray_actor_options={"num_gpus": 0.5}, num_replicas="auto")
class Embedder:
    def __init__(self):
        self.model = load_model()                      # your loader

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.01)
    async def __call__(self, texts: List[str]) -> List[list]:
        return self.model.encode(texts).tolist()       # one output per input, same order

app = Embedder.bind()

# BentoML: pip install bentoml; run with `bentoml serve`, package with `bentoml build`
import bentoml

@bentoml.service(resources={"gpu": 1}, workers=1)
class Embed:
    def __init__(self):
        self.model = load_model()

    @bentoml.api(batchable=True, max_batch_size=32, max_latency_ms=100)
    def encode(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()

# LitServe: pip install litserve; run with `python server.py`
import litserve as ls

class EmbedAPI(ls.LitAPI):
    def setup(self, device):
        self.model = load_model(device)

    def predict(self, texts):                          # receives the batched list when max_batch_size > 1
        return self.model.encode(texts).tolist()

if __name__ == "__main__":
    server = ls.LitServer(EmbedAPI(max_batch_size=32, batch_timeout=0.01), accelerator="auto", workers_per_device=1)
    server.run(host="127.0.0.1", port=8000)            # run() defaults to host 0.0.0.0
```
Ray Serve LLM, OpenAI-compatible vLLM deployment (shape of the Ray docs example; replace the model and accelerator):
<!-- skip-check: needs ray[llm], vLLM and GPUs -->
```python
from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app

llm_config = LLMConfig(
    model_loading_config={"model_id": "my-model", "model_source": "<org>/<hf-model>"},
    deployment_config={"autoscaling_config": {"min_replicas": 1, "max_replicas": 2}},
    accelerator_type="<gpu-type>",
    engine_kwargs={"tensor_parallel_size": 2},       # passed through to vLLM
)
serve.run(build_openai_app({"llm_configs": [llm_config]}), blocking=True)
# query with any OpenAI client at http://localhost:8000/v1
```
Runnable, stdlib only: the dynamic-batching loop these frameworks implement (size cap or wait timeout, whichever first; outputs routed back per request). Run it to see how the timeout trades per-request latency for batch size under bursty versus sparse arrivals.
```python
import asyncio
import random
import time

class MicroBatcher:
    """Collect single requests into batches: flush at max_batch_size or batch_wait_s after the first item."""
    def __init__(self, model_fn, max_batch_size, batch_wait_s):
        self.model_fn, self.max_batch_size, self.batch_wait_s = model_fn, max_batch_size, batch_wait_s
        self.queue = asyncio.Queue()
        self.batch_sizes = []

    async def submit(self, item):
        fut = asyncio.get_running_loop().create_future()
        await self.queue.put((item, fut))
        return await fut

    async def run(self):
        while True:
            item, fut = await self.queue.get()               # block until the first request
            batch = [(item, fut)]
            deadline = time.monotonic() + self.batch_wait_s
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(), remaining))
                except asyncio.TimeoutError:
                    break
            self.batch_sizes.append(len(batch))
            outputs = await self.model_fn([x for x, _ in batch])
            if len(outputs) != len(batch):                   # the contract every framework enforces
                raise ValueError("model must return one output per input")
            for (_, f), out in zip(batch, outputs):
                f.set_result(out)

def make_model(fixed_s, per_item_s):
    """Fake vectorized model: a fixed launch cost plus a small per-item cost (measure yours)."""
    async def model(xs):
        await asyncio.sleep(fixed_s + per_item_s * len(xs))
        return [x * 2 for x in xs]
    return model

async def load(batcher, n_requests, mean_gap_s, seed):
    rng = random.Random(seed)
    latencies = []

    async def one(i):
        t0 = time.monotonic()
        assert await batcher.submit(i) == i * 2
        latencies.append(time.monotonic() - t0)

    tasks = []
    for i in range(n_requests):
        tasks.append(asyncio.create_task(one(i)))
        await asyncio.sleep(rng.expovariate(1 / mean_gap_s))  # Poisson arrivals
    await asyncio.gather(*tasks)
    latencies.sort()
    return latencies

async def experiment(max_batch_size, batch_wait_s, mean_gap_s, fixed_s, per_item_s, n=200, seed=0):
    b = MicroBatcher(make_model(fixed_s, per_item_s), max_batch_size, batch_wait_s)
    worker = asyncio.create_task(b.run())
    t0 = time.monotonic()
    lat = await load(b, n, mean_gap_s, seed)
    wall = time.monotonic() - t0
    worker.cancel()
    p50, p95 = lat[len(lat) // 2], lat[int(len(lat) * 0.95)]
    mean_bs = sum(b.batch_sizes) / len(b.batch_sizes)
    return {"max_bs": max_batch_size, "wait_ms": batch_wait_s * 1e3, "gap_ms": mean_gap_s * 1e3,
            "batches": len(b.batch_sizes), "mean_bs": round(mean_bs, 1),
            "p50_ms": round(p50 * 1e3, 1), "p95_ms": round(p95 * 1e3, 1), "req_per_s": round(n / wall, 1)}

async def main():
    fixed_s, per_item_s = 0.005, 0.0002          # synthetic: launch cost >> per-item cost; not a measurement
    for gap in (fixed_s / 10, fixed_s * 2):     # bursty (gap << launch cost) vs sparse (gap > launch cost)
        for max_bs, wait in ((1, 0.0), (16, 0.002), (16, 0.02)):
            print(await experiment(max_bs, wait, gap, fixed_s, per_item_s))

asyncio.run(main())
```
Reading the output: with bursty traffic, batching collapses many model calls into few and raises throughput; with sparse traffic, a long wait adds latency without filling batches. That is why Ray Serve's default wait is short and BentoML adapts its window.

## Choosing / trade-offs
| Need | Ray Serve | BentoML | LitServe | Plain FastAPI |
|---|---|---|---|---|
| Server-side batching | `@serve.batch`, token-aware via `batch_size_fn` | Adaptive, latency-capped | `max_batch_size` + `batch_timeout` | Write it yourself |
| Multi-model pipeline, separate scaling | Deployments + handles, per-deployment autoscaling | `bentoml.depends` services | One process; scale the whole API | Separate services |
| Multi-node, cluster scheduling | Yes (Ray cluster, KubeRay) | Via Kubernetes or its cloud | Per-node; multi-node on its cloud | Via Kubernetes |
| Share a GPU between models | Fractional `num_gpus` | Workers per GPU | `workers_per_device` | Manual |
| Many per-tenant models | Multiplexing with LRU | Manual | Manual | Manual |
| Packaging | YAML config to a Ray cluster | Bento + generated Docker image | Your Dockerfile | Your Dockerfile |
| Operational weight | Highest (a Ray cluster) | Medium | Lowest | Lowest |

- **Already on Ray (training, data, RL)**: Ray Serve reuses the cluster and Ray Data for batch jobs ([[batch-inference-and-batch-apis]]).
- **Small team, one or two models, Docker deploys**: BentoML or LitServe; LitServe is closest to plain Python, BentoML gives packaging and versioned artifacts ([[model-registry-and-versioning]]).
- **Max throughput for one ONNX/TensorRT model**: Triton, with these frameworks only as a pre/post-processing layer ([[onnx-and-triton-inference-server]]).
- **LLM engines**: put the engine's own server first; reach for Ray Serve LLM only for multi-node, multi-model routing or when the rest of the stack is Ray ([[llm-gateways-and-routing]]).

## Gotchas
- A batched method must return exactly one output per input in the same order; returning a single array for the whole batch, or dropping failed items, breaks the split back to requests.
- Batching only helps when the model vectorizes. A Python loop over the batch inside `predict` keeps the latency cost of waiting and removes the gain.
- Each worker or replica loads its own copy of the model: `workers=4` on BentoML means four copies in RAM or VRAM ([[model-weight-loading-and-cold-starts]]).
- Ray fractional GPUs decide placement; they are not documented as memory isolation. Two replicas with `num_gpus: 0.5` must both fit in the GPU's memory, so measure each model's footprint first.
- Ray Serve `autoscaling_config` without `num_replicas="auto"` has a base `max_replicas` of 1: autoscaling appears configured but never adds replicas.
- `max_ongoing_requests` caps queueing per replica; set it relative to `target_ongoing_requests` and the batch size, or requests queue at the proxy instead of forming batches.
- BentoML adaptive batching does not keep request order inside a batch, and a synchronous caller in another service sends one request at a time, so it never fills a batch.
- LitServe `run()` binds `0.0.0.0` by default and has no built-in auth for self-hosting; put it behind a gateway ([[ai-security-privacy-compliance]]).
- BentoML collects anonymous usage data by default; opt out with `BENTOML_DO_NOT_TRACK=True` or `--do-not-track`.
- Framework "Nx faster than FastAPI" claims come from the vendors' own benchmarks on specific tasks; measure your model with your traffic ([[llm-load-testing-and-benchmarking]]).

## Related
- [[serving-with-fastapi]] - the baseline these frameworks build on or replace.
- [[inference-servers-vllm]] - the LLM engine Ray Serve LLM and BentoML examples wrap.
- [[onnx-and-triton-inference-server]] - the model-repository alternative for high-throughput single models.
- [[embedding-and-reranker-serving]] - batching embedders and rerankers, a common use of these frameworks.
- [[kubernetes-for-llm-serving]] - KubeRay, KServe and GPU autoscaling on a cluster.
- [[multi-lora-adapter-serving]] - many fine-tunes on one base model, the LLM analogue of multiplexing.
- [[model-weight-loading-and-cold-starts]] - replica start-up time that limits autoscaling.
- [[batch-inference-and-batch-apis]] - offline jobs on the same models.
- [[llm-load-testing-and-benchmarking]] - measure batch size, latency and throughput before tuning.
- [[mlops-lifecycle]] - where serving sits in the model lifecycle.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/reliability/capacity-planning-and-autoscaling.md - scaling policies and headroom.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/reliability/backpressure-and-load-shedding.md - what to do when the batch queue grows faster than it drains.

## References
- Ray Serve overview: https://github.com/ray-project/ray/blob/master/doc/source/serve/index.md
- Ray Serve key concepts (deployments, applications, ingress, DeploymentHandle): https://github.com/ray-project/ray/blob/master/doc/source/serve/key-concepts.md
- Ray Serve dynamic request batching (`@serve.batch` parameters and defaults): https://github.com/ray-project/ray/blob/master/doc/source/serve/advanced-guides/dyn-req-batch.md
- Ray Serve autoscaling guide (`num_replicas="auto"` defaults): https://github.com/ray-project/ray/blob/master/doc/source/serve/autoscaling-guide.md
- Ray Serve resource allocation (fractional GPUs): https://github.com/ray-project/ray/blob/master/doc/source/serve/resource-allocation.md
- Ray Serve model multiplexing: https://github.com/ray-project/ray/blob/master/doc/source/serve/model-multiplexing.md
- Ray Serve LLM: https://github.com/ray-project/ray/blob/master/doc/source/serve/llm/index.md
- BentoML README (services, build, containerize, usage tracking): https://github.com/bentoml/BentoML/blob/main/README.md
- BentoML adaptive batching: https://github.com/bentoml/BentoML/blob/main/docs/source/get-started/adaptive-batching.rst
- BentoML workers: https://github.com/bentoml/BentoML/blob/main/docs/source/build-with-bentoml/parallelize-requests.rst
- BentoML model composition (`bentoml.depends`): https://github.com/bentoml/BentoML/blob/main/docs/source/get-started/model-composition.rst
- LitServe README: https://github.com/Lightning-AI/LitServe/blob/main/README.md
- LitServe `LitAPI` and `LitServer` arguments (source): https://github.com/Lightning-AI/LitServe/blob/main/src/litserve/api.py, https://github.com/Lightning-AI/LitServe/blob/main/src/litserve/server.py
