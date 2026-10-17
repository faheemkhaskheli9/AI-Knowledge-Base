---
title: ComfyUI (node-based local image and video generation)
category: setup
tags: [comfyui, stable-diffusion, flux, diffusion, image-generation, video-generation, workflows, custom-nodes, lora]
use_cases:
  - "generate product images locally on my own GPU without a cloud API"
  - "build a repeatable image pipeline (text-to-image, upscale, inpaint) I can share as one file"
  - "call a local image-generation workflow from my Python backend"
  - "share one folder of checkpoints and LoRAs between ComfyUI and other tools"
status: draft
last_verified: 2026-10-15
sources:
  - https://github.com/Comfy-Org/ComfyUI
  - https://docs.comfy.org/installation/manual_install
  - https://docs.comfy.org/development/comfyui-server/comms_routes
  - https://docs.comfy.org/installation/install_custom_node
  - https://comfyui-wiki.com/install/install-comfyui/install-comfyui-on-windows
---

# ComfyUI (node-based local image and video generation)

## Summary
ComfyUI is an open-source (GPL-3.0) app that runs diffusion models (Stable Diffusion family, FLUX, video models and others) locally, where you build the pipeline as a graph of nodes in the browser: load model, encode prompt, sample, decode, save. A graph (workflow) is a JSON file you can share, version and also submit to ComfyUI's local HTTP API, so the same pipeline serves both experimentation and automation. Model choice lives in [[image-generation-models]]; how diffusion works lives in [[diffusion-models]].

## Key concepts
- Nodes and links: each node is an operation (`CheckpointLoaderSimple`, `CLIPTextEncode`, `KSampler`, `VAEDecode`, `SaveImage`, ...); outputs connect to typed inputs (MODEL, CLIP, VAE, CONDITIONING, LATENT, IMAGE).
- Execution: only nodes needed for an output run, and unchanged nodes are cached between runs, so tweaking the sampler does not reload the checkpoint.
- Model folders under `ComfyUI/models/`: `checkpoints/`, `loras/`, `vae/`, `controlnet/`, `upscale_models/`, `text_encoders/`, `diffusion_models/` and more; the folder decides which loader node lists a file. `extra_model_paths.yaml` (rename the shipped `.example`) points ComfyUI at folders you already have.
- Workflow formats: the UI format (with layout) is what you save and share; images saved by ComfyUI embed the workflow in their metadata so dropping a PNG onto the canvas restores it. The API format (export via the menu, "Export (API)") is the compact `{node_id: {class_type, inputs}}` graph that the `/prompt` endpoint accepts.
- Server: the UI talks to a local server (default `http://127.0.0.1:8188`). `POST /prompt` queues a workflow and returns a `prompt_id`; `/ws` streams progress; `/history/{prompt_id}` lists outputs; `/view` returns a file.
- Custom nodes: Python packages dropped into `custom_nodes/` (often installed with ComfyUI-Manager). They run arbitrary code with your user's permissions.
- Install options: Desktop app, Windows portable build (bundled Python), or manual install from git; ComfyUI also supports non-NVIDIA backends (see the README).

## When to use / scenarios
- A marketing team iterates on product shots with a fixed brand LoRA and ControlNet pose, then hands the workflow file to a designer ([[marketing-content]]).
- An e-commerce backend generates background variations for catalogue images overnight on an on-prem GPU ([[ecommerce-retail]]).
- An artist chains upscaling, inpainting and video interpolation steps that one-click UIs do not expose.
- Not for: one-off images where a hosted API is simpler ([[image-generation-models]]); multi-tenant production at scale without a queueing/serving layer in front ([[serving-with-fastapi]]); learning the maths ([[ddpm-diffusion-from-scratch-numpy]]).

## Setup & code
Manual install (Linux/WSL2/macOS; install the GPU build of PyTorch first per [[gpu-cuda-setup]]):
```bash
git clone https://github.com/Comfy-Org/ComfyUI
cd ComfyUI
uv venv && source .venv/bin/activate
uv pip install torch torchvision torchaudio --torch-backend=auto
uv pip install -r requirements.txt
python main.py                     # UI at http://127.0.0.1:8188
```
Put a checkpoint in `models/checkpoints/` (check its licence, [[model-licenses]]), load the default workflow, press Run.

Build and check an API-format text-to-image graph (runnable, stdlib only). The node class names are those of ComfyUI's default workflow; the checkpoint filename is a placeholder.
```python
import json

def txt2img(ckpt, prompt, negative="", seed=0, steps=20, cfg=7.0, w=1024, h=1024):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0],
            "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler",
            "scheduler": "normal", "denoise": 1.0}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "demo"}},
    }

def check_links(graph):
    """Every [node_id, output_index] input must point at an existing node; return a run order."""
    deps = {nid: [v[0] for v in n["inputs"].values() if isinstance(v, list)] for nid, n in graph.items()}
    missing = [(nid, d) for nid, ds in deps.items() for d in ds if d not in graph]
    assert not missing, f"dangling links: {missing}"
    order, seen = [], set()
    def visit(n, stack=()):
        assert n not in stack, f"cycle at {n}"
        if n not in seen:
            for d in deps[n]:
                visit(d, stack + (n,))
            seen.add(n)
            order.append(n)
    for n in graph:
        visit(n)
    return order

g = txt2img("<your-checkpoint>.safetensors", "studio photo of a ceramic mug, soft light", seed=42)
order = check_links(g)
print("run order:", " -> ".join(g[n]["class_type"] for n in order))
payload = json.dumps({"prompt": g, "client_id": "my-backend"})
print("payload bytes:", len(payload))
g["5"]["inputs"]["seed"] = 43          # vary only the seed; the checkpoint stays cached server-side
```
Submit it to a running ComfyUI (only works with the server up and the checkpoint present):
<!-- skip-check: needs a running ComfyUI server on localhost:8188 -->
```python
import json, time, urllib.request
BASE = "http://127.0.0.1:8188"
req = urllib.request.Request(f"{BASE}/prompt", data=payload.encode(),
                             headers={"Content-Type": "application/json"})
pid = json.load(urllib.request.urlopen(req))["prompt_id"]
while True:
    hist = json.load(urllib.request.urlopen(f"{BASE}/history/{pid}"))
    if pid in hist:
        print(hist[pid]["outputs"])        # filenames; fetch them via /view?filename=...&type=output
        break
    time.sleep(1)
```

## Choosing / trade-offs
| Option | Good for | Cost |
|---|---|---|
| ComfyUI | Full control, complex multi-step graphs, video models, API automation | Steeper learning curve; graphs get large |
| Simpler local UIs (form-based) | Quick prompting, beginners | Less control over pipelines |
| `diffusers` in Python ([[diffusion-models]]) | Code-first pipelines, tests, custom training | You write the glue that ComfyUI nodes give you |
| Hosted image APIs ([[image-generation-models]]) | No GPU, low volume, newest proprietary models | Per-image cost, data leaves your network |

- Desktop/portable vs manual install: packaged builds are easiest; manual install fits servers, WSL and Docker ([[docker-for-ml]]).
- VRAM decides which models and resolutions run; ComfyUI can offload to system RAM but becomes much slower ([[gpu-cuda-setup]], [[quantization]]).

## Gotchas
- Custom nodes and downloaded workflows can execute arbitrary Python; install only from sources you trust, review new nodes, and prefer `.safetensors` over pickle-based checkpoint formats ([[ai-security-privacy-compliance]]).
- The server has no authentication; `--listen` exposes it to the network. Keep it on `127.0.0.1` or behind an authenticating reverse proxy.
- Saved PNGs embed the full workflow and prompt in metadata; strip metadata before publishing images if prompts are confidential.
- A UI-format workflow sent to `/prompt` fails with node errors; export the API format.
- Workflows reference model filenames; on another machine missing files or different folder names break them. Share a list of required models with their sources.
- Custom nodes pin their own dependencies and can break each other or a ComfyUI update; keep one environment per setup and update deliberately.
- Model licences differ (some prohibit commercial use or need attribution); check before shipping outputs ([[model-licenses]]).

## Related
- [[image-generation-models]] - which models to load and their licences.
- [[diffusion-models]] - what the sampler, CFG and VAE nodes do.
- [[image-editing-and-controlnet]] - inpainting and ControlNet, common ComfyUI graphs.
- [[video-generation-models]] - video models you can run in ComfyUI.
- [[gpu-cuda-setup]] - drivers, PyTorch build and VRAM.
- [[docker-for-ml]] - running ComfyUI headless in a container with a model volume.
- [[lm-studio-and-jan]] - the analogous GUI route for local LLMs.
- [[jupyter-colab-and-kaggle]] - trying ComfyUI on a hosted GPU before buying one.
- [[fine-tuning-and-peft]] - LoRA, which ComfyUI loads with a LoRA loader node.
- [[marketing-content]] - typical commercial use.

## References
- ComfyUI repository and README: https://github.com/Comfy-Org/ComfyUI
- Manual install: https://docs.comfy.org/installation/manual_install
- Server routes and WebSocket messages: https://docs.comfy.org/development/comfyui-server/comms_routes
- Custom nodes: https://docs.comfy.org/installation/install_custom_node
- Windows install options (community wiki): https://comfyui-wiki.com/install/install-comfyui/install-comfyui-on-windows
