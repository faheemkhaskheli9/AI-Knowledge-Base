---
title: ONNX export and Triton Inference Server
category: deployment
tags: [onnx, onnxruntime, triton, tensorrt, dynamic-batching, model-repository, config-pbtxt, kserve-v2, gpu-serving]
use_cases:
  - "serve an image classifier or embedding model on GPU with high throughput"
  - "export a PyTorch model to ONNX and serve it without Python in the request path"
  - "batch many small concurrent inference requests together on one GPU"
  - "host PyTorch, ONNX, TensorRT and scikit-learn/XGBoost models behind one inference server"
status: draft
last_verified: 2026-10-15
sources:
  - https://github.com/triton-inference-server/server
  - https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md
  - https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md
  - https://github.com/triton-inference-server/server/blob/main/docs/user_guide/batcher.md
  - https://docs.pytorch.org/docs/stable/onnx.html
  - https://onnxruntime.ai/docs/
---

# ONNX export and Triton Inference Server

## Summary
ONNX is a framework-neutral model format. Export a PyTorch (or scikit-learn, TensorFlow) model once and run it with ONNX Runtime on CPU, CUDA, TensorRT, OpenVINO and other execution providers. NVIDIA Triton Inference Server (BSD-3, open source) serves models from a directory "model repository" over HTTP/gRPC. It supports ONNX, TensorRT, PyTorch, OpenVINO, Python and tree-model (FIL) backends, and adds **dynamic batching**, concurrent model instances, ensembles and Prometheus metrics. Together they are a standard way to serve non-LLM models (vision, embeddings, rerankers, tabular, speech) at high throughput. For LLMs prefer [[inference-servers-vllm]] or [[sglang-and-tensorrt-llm]].

## Key concepts
- **ONNX export**: `torch.onnx.export(model, example_inputs, "model.onnx", dynamo=True, dynamic_shapes=...)` traces the model into an ONNX graph. Mark batch (and sequence) dimensions dynamic, or the model only accepts the example shape. Verify outputs against PyTorch with a tolerance. Basics are in [[model-checkpointing-and-export]].
- **ONNX Runtime execution providers**: the same `.onnx` runs on `CPUExecutionProvider`, `CUDAExecutionProvider`, `TensorrtExecutionProvider`, and others. Graph optimizations and quantization are applied at load or offline ([[quantization]]).
- **Model repository** layout Triton loads:
  ```
  model_repository/
    resnet50/
      config.pbtxt
      1/model.onnx        # numeric version directories
      2/model.onnx
  ```
- **`config.pbtxt`**: backend/platform, `max_batch_size`, input/output names, dtypes, dims, `instance_group` (how many copies on which GPUs), `dynamic_batching`. For ONNX and TensorRT, Triton can auto-complete much of the config from the model; `--disable-auto-complete-config` turns that off.
- **Dynamic batching**: Triton holds requests up to `max_queue_delay_microseconds` and merges them into one batch (optionally preferring `preferred_batch_size`). The GPU does one large matmul instead of many small ones, so throughput rises a lot for a small latency cost. This differs from LLM continuous batching, which interleaves decode steps.
- **Instance groups**: several copies of a model on one GPU overlap work; `KIND_CPU` / `KIND_GPU` choose the device.
- **Ensembles / BLS**: chain preprocessing (Python backend), model and postprocessing server-side.
- **Protocol**: HTTP/REST and gRPC following the KServe v2 inference protocol (default ports 8000 HTTP, 8001 gRPC, 8002 metrics). Clients use `tritonclient`.

## When to use / scenarios
- E-commerce image tagging or visual search: many concurrent single-image requests to a CNN/ViT on one GPU. Dynamic batching multiplies throughput.
- Embedding or reranker service for RAG ([[embedding-models]], [[rerankers]]): export the encoder to ONNX (optionally TensorRT), serve with Triton, batch small requests.
- Manufacturing defect detection or video analytics ([[object-detection]], [[video-analytics]]): ensembles for decode -> preprocess -> detect -> NMS.
- Fraud scoring with gradient-boosted trees at high QPS: Triton FIL backend next to deep models.
- CPU-only or edge deployment without Python: ONNX Runtime directly in the app, no server ([[edge-on-device]]).
- Not needed for low traffic: a FastAPI app with ONNX Runtime is simpler ([[serving-with-fastapi]]). Not the first pick for LLM chat serving.

## Setup & code
Export and check (PyTorch side):
<!-- skip-check: needs torch, torchvision and onnxruntime -->
```python
import numpy as np, torch, torchvision, onnxruntime as ort
model = torchvision.models.resnet18(weights=None).eval()
x = torch.randn(1, 3, 224, 224)
batch = torch.export.Dim("batch", min=1, max=64)
torch.onnx.export(model, (x,), "model.onnx", dynamo=True,
                  input_names=["input"], output_names=["logits"],
                  dynamic_shapes={"x": {0: batch}})
sess = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
xb = np.random.randn(8, 3, 224, 224).astype(np.float32)
ref = model(torch.from_numpy(xb)).detach().numpy()
out = sess.run(None, {"input": xb})[0]
print("max abs diff:", np.abs(out - ref).max())       # expect ~1e-5 in fp32
```
`model_repository/resnet18/config.pbtxt`:
```
name: "resnet18"
backend: "onnxruntime"
max_batch_size: 64
input  [ { name: "input"  data_type: TYPE_FP32 dims: [ 3, 224, 224 ] } ]
output [ { name: "logits" data_type: TYPE_FP32 dims: [ 1000 ] } ]
instance_group [ { count: 2 kind: KIND_GPU } ]
dynamic_batching {
  preferred_batch_size: [ 16, 32 ]
  max_queue_delay_microseconds: 2000
}
```
Run Triton from NGC (use a current `<yy.mm>` tag) and call it:
```bash
docker run --gpus=1 --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $PWD/model_repository:/models nvcr.io/nvidia/tritonserver:<yy.mm>-py3 \
  tritonserver --model-repository=/models
curl localhost:8000/v2/health/ready
uv pip install "tritonclient[http]"
```
<!-- skip-check: needs a running Triton server -->
```python
import numpy as np, tritonclient.http as httpclient
client = httpclient.InferenceServerClient("localhost:8000")
x = np.random.randn(4, 3, 224, 224).astype(np.float32)
inp = httpclient.InferInput("input", x.shape, "FP32"); inp.set_data_from_numpy(x)
res = client.infer("resnet18", [inp], outputs=[httpclient.InferRequestedOutput("logits")])
print(res.as_numpy("logits").argmax(axis=1))
```
What dynamic batching buys, as a numpy simulation: Poisson arrivals of single-item requests, a GPU whose batch cost is a fixed overhead plus a small per-item cost, and a scheduler that waits up to `max_delay` to fill a batch.
```python
import numpy as np

rng = np.random.default_rng(0)
RATE = 800                       # requests/s
arrivals = np.cumsum(rng.exponential(1 / RATE, size=4000))
def gpu_time(n):                 # fixed launch/overhead + per-item cost (seconds)
    return 0.004 + 0.0002 * n

def serve(max_batch, max_delay):
    t_free, i, lat = 0.0, 0, []
    while i < len(arrivals):
        start = max(t_free, arrivals[i])
        deadline = start + max_delay           # wait for more requests up to max_delay
        j = i + 1
        while j < len(arrivals) and j - i < max_batch and arrivals[j] <= deadline:
            j += 1
        launch = max(start, arrivals[j - 1]) if j - i == max_batch else (deadline if max_delay else start)
        launch = max(launch, t_free)
        done = launch + gpu_time(j - i)
        lat.extend(done - arrivals[i:j]); t_free, i = done, j
    lat = np.array(lat)
    return len(arrivals) / t_free, np.percentile(lat, 50) * 1e3, np.percentile(lat, 99) * 1e3

for mb, d in [(1, 0.0), (32, 0.0), (32, 0.002), (32, 0.010)]:
    thr, p50, p99 = serve(mb, d)
    print(f"max_batch={mb:>2} delay={d*1e3:>4.0f}ms  throughput={thr:6.0f}/s  p50={p50:7.1f}ms  p99={p99:7.1f}ms")
```
Without batching the GPU (about 240 single requests/s here) cannot keep up with 800 requests/s and the queue grows without bound. Batching keeps up. At this load, requests already queue up while the GPU is busy, so batches fill without any added delay and a larger `max_delay` only adds latency. A queue delay helps at lower or bursty load, where it collects a few more requests per launch and raises GPU efficiency. Tune `max_queue_delay_microseconds` against your latency SLO, starting at zero or small values.

## Choosing / trade-offs
| Option | Pick when |
|---|---|
| ONNX Runtime in-process (FastAPI, Java, C#, mobile) | low/medium QPS, CPU, edge, simplest ops |
| Triton + ONNX backend | GPU, many concurrent small requests, several models, need batching and metrics |
| Triton + TensorRT backend | lowest latency on NVIDIA; build engines per GPU type and precision |
| TorchServe / plain PyTorch | export to ONNX fails (dynamic control flow) and you accept lower throughput |
| KServe in front of Triton | Kubernetes-native rollout and autoscaling ([[kubernetes-for-llm-serving]]) |

- ONNX vs native PyTorch: ONNX is portable and often faster, but exporting models with data-dependent control flow or custom ops can fail or need rewrites.
- TensorRT: fastest on NVIDIA, but engines are tied to GPU architecture and TensorRT version; rebuild on upgrade.
- Larger `max_queue_delay` and `preferred_batch_size` give more throughput and higher latency; set from the p99 SLO, not by guessing.

## Gotchas
- Exporting without dynamic batch dimensions produces a model that only accepts batch size 1; Triton then cannot batch it.
- `max_batch_size > 0` means `dims` in `config.pbtxt` exclude the batch dimension; getting this wrong is the most common load error.
- Input/output names in the config must match the ONNX graph exactly (check with Netron or `onnxruntime` session `get_inputs()`).
- fp16/INT8 conversion can shift outputs; compare against the fp32 reference on real data, not random tensors.
- Preprocessing mismatch (resize, normalization, channel order) between training and the serving path silently degrades accuracy. Put preprocessing in the ensemble or test it end-to-end.
- Triton containers pair with specific CUDA drivers; check the NGC release notes for the minimum driver.
- Pin the NGC image tag; `latest`-style upgrades change backends and behavior.

## Related
- [[model-checkpointing-and-export]] - saving and exporting models (ONNX, torch.export).
- [[edge-on-device]] - ONNX Runtime, TensorRT and Core ML on devices.
- [[serving-with-fastapi]] - simpler in-process serving.
- [[kubernetes-for-llm-serving]] - KServe/Kubernetes deployment around Triton.
- [[sglang-and-tensorrt-llm]] - TensorRT-LLM for LLMs, which Triton can also host.
- [[batch-inference-and-batch-apis]] - offline scoring instead of online serving.
- [[quantization]] - fp16/INT8 for faster inference.
- [[image-classification]], [[object-detection]], [[embedding-models]] - typical models served this way.
- [[ray-serve-bentoml-and-litserve]] - Python-first serving frameworks as an alternative or a pre/post-processing layer.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/system-design/load-balancing.md - spreading requests over Triton replicas.

## References
- Triton Inference Server repository (backends, features, licence, current release): https://github.com/triton-inference-server/server
- Triton model repository: https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md
- Triton model configuration: https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md
- Triton dynamic batcher: https://github.com/triton-inference-server/server/blob/main/docs/user_guide/batcher.md
- PyTorch ONNX export: https://docs.pytorch.org/docs/stable/onnx.html
- ONNX Runtime docs (execution providers): https://onnxruntime.ai/docs/
