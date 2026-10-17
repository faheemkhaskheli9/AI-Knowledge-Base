---
title: Jupyter, Google Colab and Kaggle notebooks
category: setup
tags: [jupyter, jupyterlab, colab, kaggle, notebooks, free-gpu, reproducibility, ipykernel]
use_cases:
  - "get a free GPU to try fine-tuning a small model without buying hardware"
  - "set up JupyterLab locally with the same environment as my scripts"
  - "make a notebook that a reviewer can re-run and get the same numbers"
  - "run a Kaggle competition notebook and keep my work when the session ends"
status: draft
last_verified: 2026-10-15
sources:
  - https://jupyterlab.readthedocs.io/en/latest/getting_started/installation.html
  - https://ipython.readthedocs.io/en/stable/install/kernel_install.html
  - https://research.google.com/colaboratory/faq.html
  - https://colab.research.google.com/signup
  - https://www.kaggle.com/docs/notebooks
  - https://pytorch.org/docs/stable/notes/randomness.html
---

# Jupyter, Google Colab and Kaggle notebooks

## Summary
Notebooks mix code, output and notes, which suits exploration, teaching and quick experiments. JupyterLab runs locally against your own Python environment; Google Colab and Kaggle Notebooks are hosted Jupyter-style environments with free (limited, not guaranteed) CPU/GPU/TPU time. The main risks are lost work when a hosted session ends and results that cannot be reproduced because of hidden state, unpinned packages or unseeded randomness.

## Key concepts
- Kernel: the process that runs a notebook's code. Each environment ([[python-env-uv]]) needs `ipykernel` registered as a kernel to appear in JupyterLab.
- Hidden state: cells can run out of order; the only trustworthy result is "Restart kernel and run all".
- Hosted runtimes are ephemeral VMs: files on the local disk vanish when the session ends or idles out. Persist to Google Drive (Colab), Kaggle Datasets/Notebook output (Kaggle) or a Hub repo.
- Free tiers:
  - Colab: the FAQ says free resources are not guaranteed and not unlimited, usage limits change over time, notebooks run for at most 12 hours depending on availability, and idle runtimes are disconnected. Paid plans (Pay As You Go, Pro, Pro+) use compute units.
  - Kaggle: GPU and TPU accelerators are available with weekly quotas and per-session time limits; your current quota is shown in your account/notebook settings and changes over time. Phone verification is required to enable accelerators and internet access.
  - Exact GPU models, VRAM and hour quotas change; read them from the runtime (`nvidia-smi`) and the official pages instead of blog posts.
- Reproducibility levers: pinned package versions, fixed seeds, recorded hardware/CUDA versions, data versioned or hashed, and run-all-from-scratch before sharing.

## When to use / scenarios
- A student or small startup tries QLoRA on a small model with a free GPU before renting one ([[fine-tuning-and-peft]], [[gpu-cloud-options]]).
- A data scientist explores a churn dataset ([[exploratory-data-analysis]]) and hands the cleaned pipeline to engineering.
- Kaggle competitions, where data is already attached and submissions come from notebook outputs.
- Workshops: one Colab link, no installs for attendees.
- Not for: production services, long training runs (sessions end), or private/regulated data you may not upload to a third party ([[ai-security-privacy-compliance]]). Move code into modules and scripts once it stabilises ([[experiment-tracking]] for logging runs).

## Setup & code
Local JupyterLab inside a project environment:
```bash
uv venv && source .venv/bin/activate
uv pip install jupyterlab ipykernel
python -m ipykernel install --user --name myproj --display-name "Python (myproj)"
jupyter lab
```
Colab: check the GPU and mount Drive for persistence (the mount call only works inside Colab).
```bash
!nvidia-smi
```
<!-- skip-check: only runs inside Google Colab -->
```python
from google.colab import drive
drive.mount("/content/drive")
CKPT_DIR = "/content/drive/MyDrive/runs/exp1"   # save checkpoints here, not /content
```
Kaggle: attach datasets in the right-hand panel (read-only under `/kaggle/input`), write outputs to `/kaggle/working`, and turn on the accelerator in notebook settings.

A reproducibility header to paste as the first cell (runnable: stdlib + numpy). It seeds RNGs, records the environment and fingerprints the input data so a reviewer can tell whether a rerun used the same inputs.
```python
import hashlib, importlib, json, platform, random, sys
import numpy as np

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
rng = np.random.default_rng(SEED)            # prefer an explicit Generator in new code
try:
    torch = importlib.import_module("torch")   # optional; works without PyTorch
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False    # see PyTorch "Reproducibility" notes
except ImportError:
    torch = None

def fingerprint(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]

def env_report():
    rep = {"python": sys.version.split()[0], "platform": platform.platform(),
           "numpy": np.__version__, "seed": SEED}
    if torch is not None:
        rep["torch"] = torch.__version__
        rep["cuda"] = torch.version.cuda
        rep["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return rep

data = rng.normal(size=(1000, 4))              # stand-in for your loaded dataset
print(json.dumps({**env_report(), "data_sha256": fingerprint(data)}, indent=1))

# Same seed -> same numbers: the property a rerun relies on.
a = np.random.default_rng(SEED).normal(size=5)
b = np.random.default_rng(SEED).normal(size=5)
assert np.array_equal(a, b)
```
Then pin what you used, for example `pip freeze > requirements-lock.txt` (or `uv pip compile`) and commit it next to the notebook.

## Choosing / trade-offs
| Option | Good for | Watch out for |
|---|---|---|
| Local JupyterLab | Private data, your own GPU, long runs, version control | You maintain drivers and envs ([[gpu-cuda-setup]]) |
| Google Colab (free) | Zero setup, sharing a link, Drive integration | Not guaranteed GPUs, session caps and idle disconnects |
| Colab paid / Colab Enterprise | Longer and more reliable sessions | Compute-unit billing; check current plan terms |
| Kaggle Notebooks | Competitions, public datasets, weekly free accelerator quota | Quotas, public-by-default sharing choices, internet toggle |
| Cloud GPU VM + JupyterLab | Big models, full control | Pay per hour; shut it down ([[gpu-cloud-options]]) |

- Notebook vs script: notebooks for exploration and reports; scripts/modules plus tests for anything that runs twice. Tools such as `jupytext` or `nbconvert --to script` help move code out.
- Full GPU determinism costs speed (deterministic kernels, no cuDNN autotuning); for most work, fixed seeds plus recorded versions are enough.

## Gotchas
- `!pip install` without versions gives a different environment next week; pin versions in the install cell.
- Installing a package that upgrades a preinstalled one (numpy, torch) on Colab may require a runtime restart, and can break other preinstalled libraries.
- Use `%pip install` rather than `!pip install` locally so the package goes into the kernel's environment, not whichever `pip` is on PATH.
- Secrets pasted into cells end up in saved notebooks and shared links; use Colab's Secrets panel, Kaggle Secrets, or environment variables.
- Saving checkpoints to `/content` or `/kaggle/working` only, then losing the session, loses the run; write to persistent storage periodically.
- Output cells can contain private data; clear outputs before committing or sharing (`nbstripout` as a pre-commit hook).
- Free GPUs vary by session; a script tuned for one GPU's memory may OOM on another. Check with `nvidia-smi` each time.

## Related
- [[python-env-uv]] - environments and lockfiles behind a local kernel.
- [[gpu-cuda-setup]] - checking the GPU from inside a notebook.
- [[docker-for-ml]] - the fully pinned alternative to a notebook VM.
- [[experiment-tracking]] - logging notebook runs beyond the notebook.
- [[gpu-cloud-options]] - paid GPUs when free tiers are not enough.
- [[fine-tuning-and-peft]] - the typical free-GPU workload.
- [[huggingface-transformers]] - downloading models into the session cache.
- [[lm-studio-and-jan]] - GUI local alternative for chatting with models.

## References
- JupyterLab install: https://jupyterlab.readthedocs.io/en/latest/getting_started/installation.html
- Kernels for environments: https://ipython.readthedocs.io/en/stable/install/kernel_install.html
- Colab FAQ (limits, free tier): https://research.google.com/colaboratory/faq.html
- Colab plans: https://colab.research.google.com/signup
- Kaggle Notebooks docs: https://www.kaggle.com/docs/notebooks
- PyTorch reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
