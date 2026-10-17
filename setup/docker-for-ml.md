---
title: Docker for ML (CUDA images, GPU access, slim images, model caching)
category: setup
tags: [docker, containers, cuda, nvidia-container-toolkit, gpu, dockerfile, multi-stage, model-cache]
use_cases:
  - "package my PyTorch inference service in a Docker image that can see the GPU"
  - "fix a container that cannot see the GPU (could not select device driver)"
  - "shrink a 15 GB ML image and stop re-downloading model weights on every start"
  - "give the team a reproducible GPU dev environment that matches production"
status: draft
last_verified: 2026-10-15
sources:
  - https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
  - https://hub.docker.com/r/nvidia/cuda/
  - https://docs.docker.com/build/building/multi-stage/
  - https://docs.docker.com/build/cache/optimize/
  - https://huggingface.co/docs/huggingface_hub/guides/manage-cache
  - https://docs.vllm.ai/en/latest/deployment/docker.html
---

# Docker for ML (CUDA images, GPU access, slim images, model caching)

## Summary
A container pins the OS, Python, CUDA user-space libraries and your code so a model runs the same on a laptop, a CI runner and a GPU server. The host still provides the NVIDIA driver; the NVIDIA Container Toolkit injects it into containers when you pass `--gpus`. The main design choices are the base image, how to keep the image small, and where model weights live (outside the image, in a mounted cache).

## Key concepts
- Host vs container: the kernel driver stays on the host; the container carries CUDA runtime libraries. The container's CUDA version must be supported by the host driver ([[gpu-cuda-setup]]).
- NVIDIA Container Toolkit: host package that lets Docker/containerd/Podman expose GPUs. Configure Docker with `nvidia-ctk runtime configure --runtime=docker` and restart the daemon. Podman uses CDI instead.
- `nvidia/cuda` image flavours, each adding to the previous one:
  - `base`: CUDA runtime (cudart) only.
  - `runtime`: adds CUDA math libraries and NCCL.
  - `devel`: adds headers and compiler tools; use it in a build stage only.
  - cuDNN comes as separate tags. The `latest` tag is deprecated, so pin a full tag (CUDA version, flavour, OS).
- PyTorch pip wheels bundle their own CUDA runtime, so a plain `python:3.x-slim` base plus a CUDA wheel is often enough for inference; you need a `nvidia/cuda` `devel` stage only to compile CUDA extensions.
- Multi-stage builds: compile/install in a fat stage, copy only the result (a venv) into a slim final stage.
- Layer cache: order Dockerfile steps from least to most frequently changed (OS packages, then lockfile + dependencies, then code).
- Model weights are data, not code: mount a volume at the Hugging Face cache (`HF_HOME`) or pre-download into a volume, instead of baking tens of GB into each image tag.

## When to use / scenarios
- Shipping an inference API ([[serving-with-fastapi]]) or vLLM server ([[inference-servers-vllm]]) to a cloud GPU VM or Kubernetes.
- A research team whose results differ between machines because of CUDA/cuDNN/PyTorch mismatches.
- A hospital deploying an on-prem model where the image is scanned and signed before release ([[ai-security-privacy-compliance]]).
- Not needed for a single-developer notebook ([[python-env-uv]] is lighter), or for hosted APIs. On Windows, run Docker with the WSL2 backend ([[windows-wsl-setup]]).

## Setup & code
Host (Linux) after installing the toolkit package per NVIDIA's guide:
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:<cuda-version>-base-<os> nvidia-smi   # pick a tag from Docker Hub
```
Slim inference image with uv and a mounted model cache (pin your own versions):
```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:<uv-version> /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

FROM python:3.12-slim
RUN useradd -m app
WORKDIR /app
COPY --from=build /app /app
ENV PATH="/app/.venv/bin:$PATH" HF_HOME=/models
USER app
CMD ["python", "-m", "src.server"]
```
```bash
docker build -t myorg/infer:1.0.0 .
docker run --rm --gpus all -p 8000:8000 \
  -v hf-cache:/models -e HF_TOKEN \
  myorg/infer:1.0.0
```
Prebuilt servers follow the same pattern (from the vLLM docs): `docker run --runtime nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface -e HF_TOKEN -p 8000:8000 --ipc=host vllm/vllm-openai:<tag> --model <hf-org>/<model>`. See [[vllm-local-setup]].

A runnable linter for the rules above (stdlib only): flags unpinned bases, baked-in weights, root user and secrets in `ENV`.
```python
import re

def lint_dockerfile(text):
    issues, user_set = [], False
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        up = line.upper()
        if up.startswith("FROM "):
            image = line.split()[1]
            if image.lower() != "scratch" and (":" not in image.split("/")[-1] or image.endswith(":latest")):
                issues.append((n, f"unpinned base image {image}"))
        if up.startswith("USER ") and line.split()[1] not in ("root", "0"):
            user_set = True
        if re.search(r"\.(safetensors|bin|gguf|pt|ckpt)\b", line) and up.startswith(("COPY", "ADD")):
            issues.append((n, "model weights copied into the image; mount a cache volume instead"))
        if up.startswith(("ENV", "ARG")) and re.search(r"(TOKEN|KEY|SECRET|PASSWORD)\s*=\s*\S+", line, re.I):
            issues.append((n, "secret value in ENV/ARG ends up in image layers; pass it at run time"))
    if not user_set:
        issues.append((0, "container runs as root; add a non-root USER"))
    return issues

bad = """FROM nvidia/cuda:latest
ENV HF_TOKEN=hf_abc123
COPY model.safetensors /app/
CMD python serve.py
"""
good = """FROM python:3.12-slim
ENV HF_HOME=/models
USER app
CMD ["python", "-m", "src.server"]
"""
for name, df in [("bad", bad), ("good", good)]:
    found = lint_dockerfile(df)
    print(name, "->", found or "ok")
assert len(lint_dockerfile(bad)) == 4 and lint_dockerfile(good) == []
```

## Choosing / trade-offs
| Base | Use when | Cost |
|---|---|---|
| `python:3.x-slim` + PyTorch CUDA wheel | Inference, no custom CUDA kernels | Wheel is large, but no duplicate system CUDA |
| `nvidia/cuda:*-runtime-*` | Libraries that expect system CUDA/NCCL | Bigger; you manage Python yourself |
| `nvidia/cuda:*-devel-*` (build stage) | Compiling extensions (flash-attention style kernels) | Very large; never ship it as the final stage |
| Framework images (NGC PyTorch, `vllm/vllm-openai`) | Fast start, vendor-tested versions | Large, opinionated, slower security patch cycle |

- Weights in the image: immutable, works offline and air-gapped, but every tag is huge and pulls are slow. Weights in a volume or object storage: small images, but you need a warm-up/download step and cache management.
- `--ipc=host` vs `--shm-size`: PyTorch DataLoader workers and some servers need large shared memory; `--ipc=host` is simplest, an explicit `--shm-size` is more isolated.

## Gotchas
- `docker run` without `--gpus all` (or a CDI device) gives a container with no GPU and no error until your code calls CUDA.
- A newer CUDA image on an older host driver fails at start-up; upgrade the driver or pick an older tag ([[gpu-cuda-setup]]).
- `COPY . .` before installing dependencies busts the cache on every code change; add a `.dockerignore` for `.venv`, data and checkpoints.
- Secrets in `ENV`/`ARG` or a copied `.env` stay in image history; use run-time env vars or BuildKit secret mounts.
- The `latest` tag on `nvidia/cuda` is deprecated; floating tags also make builds non-reproducible.
- Default Docker on Linux runs as root and bind-mounted caches end up root-owned; set `USER` and matching volume permissions.
- Exposing an inference port with `-p 8000:8000` binds all host interfaces; use `-p 127.0.0.1:8000:8000` for local-only.

## Related
- [[gpu-cuda-setup]] - driver and CUDA version matching on the host.
- [[python-env-uv]] - lockfiles that the Dockerfile installs from.
- [[vllm-local-setup]] - the vLLM container in practice.
- [[inference-servers-vllm]] - production serving inside these images.
- [[serving-with-fastapi]] - the app you usually package.
- [[huggingface-transformers]] - the Hub cache that you mount as a volume.
- [[mlops-lifecycle]] - where images fit in build, registry and deploy.
- [[gpu-cloud-options]] - where the images run.
- [[jupyter-colab-and-kaggle]] - container-free notebook alternatives.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/reliability/containers-and-docker.md - general container practice.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/reliability/kubernetes-basics.md - running GPU containers on a cluster.

## References
- NVIDIA Container Toolkit install guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- `nvidia/cuda` images: https://hub.docker.com/r/nvidia/cuda/
- Docker multi-stage builds: https://docs.docker.com/build/building/multi-stage/
- Docker build cache: https://docs.docker.com/build/cache/optimize/
- Hugging Face cache: https://huggingface.co/docs/huggingface_hub/guides/manage-cache
- vLLM Docker deployment: https://docs.vllm.ai/en/latest/deployment/docker.html
