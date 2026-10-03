---
title: Python environments with uv
category: setup
tags: [python, uv, venv, pip, dependencies, lockfile, pyproject]
use_cases:
  - "set up a reproducible Python environment for a new AI/LLM project"
  - "install PyTorch with the right CUDA build using uv"
  - "pin Python version and dependencies so a teammate or CI gets the same environment"
  - "stop mixing conda, pip and system Python on a Windows dev machine"
status: draft
last_verified: 2026-10-03
sources:
  - https://docs.astral.sh/uv/getting-started/installation/
  - https://docs.astral.sh/uv/guides/integration/pytorch/
---

# Python environments with uv

## Summary
uv is a fast Python package and project manager (written in Rust) that replaces pip, venv, pip-tools and pyenv for most workflows. It creates virtual environments, installs and locks dependencies, and downloads Python interpreters. For AI work it matters because heavy wheels (torch, CUDA libs) install quickly and a lockfile makes environments reproducible.

## Key concepts
- Project mode: `pyproject.toml` + `uv.lock`; `uv add`, `uv sync`, `uv run`.
- pip-compatible mode: `uv venv` + `uv pip install ...` (no lockfile, drop-in for pip).
- uv manages interpreters: `uv python install 3.12`; pin with `.python-version`.
- Indexes: PyTorch CUDA wheels live on `download.pytorch.org/whl/<cuXXX>`, not PyPI; use `--torch-backend` or an explicit index.
- `uv run <cmd>` auto-syncs the env first, so no manual activation is needed.

## When to use / scenarios
- Any new Python project (RAG app, fine-tuning repo, FastAPI service).
- CI and Docker builds that need fast, deterministic installs (`uv sync --frozen`).
- Scripts with inline dependencies (`uv run script.py` with PEP 723 metadata).
- Not ideal: environments that need non-Python native stacks (CUDA toolkit, compilers, MKL) managed together. Conda/mamba is still reasonable there, though pip wheels for torch bundle CUDA runtime so it is rarely needed.

## Setup & code
Install uv:
```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```
```powershell
# Windows
winget install --id=astral-sh.uv -e
# or: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
Update: `uv self update` (standalone installer only; otherwise use the package manager you installed with).

New project:
```bash
uv init my-app && cd my-app
uv python install 3.12
uv add anthropic openai httpx
uv run python -c "import anthropic; print(anthropic.__version__)"
uv sync            # recreate env from uv.lock on another machine
```
Throwaway env, pip style:
```bash
uv venv --python 3.12
source .venv/bin/activate        # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```
PyTorch with CUDA (needs uv 0.5.3+ for the pyproject form):
```bash
uv pip install torch torchvision --torch-backend=auto     # auto-detects driver
uv pip install torch --torch-backend=cu130                # or pick explicitly
```
Project form in `pyproject.toml` (index name/CUDA tag per the PyTorch docs; replace cu130 with the tag you need):
```toml
[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu130", marker = "sys_platform == 'linux' or sys_platform == 'win32'" }]
```

## Choosing / trade-offs
- uv vs conda: uv is faster and simpler; conda wins when you need non-Python binary deps or an existing conda-only workflow.
- uv vs poetry/pdm: uv covers locking and venvs with less configuration; poetry has a larger plugin ecosystem.
- `uv pip` (no lock) for quick experiments; project mode for anything shared.

## Gotchas
- A project-mode `uv run` writes `uv.lock` and `.venv` into the repo; add `.venv` to `.gitignore` and commit `uv.lock`.
- Installing `torch` from plain PyPI on Windows gives a CPU build; you must use the CUDA index or `--torch-backend`.
- `--torch-backend=auto` inspects the installed driver; inside Docker build stages without a GPU, pass an explicit tag.
- Do not `pip install` into a uv venv without `uv pip`; bare `pip` may be absent unless created with `--seed`.
- Windows: if the PowerShell installer is blocked, check the execution policy before retrying.

## Related
- [[gpu-cuda-setup]] - choosing the CUDA build to match your driver.
- [[windows-wsl-setup]] - uv inside WSL2.
- [[huggingface-transformers]] - what you typically install next.
- [[mlops-lifecycle]] - reproducibility in the wider lifecycle.

## References
- uv install docs: https://docs.astral.sh/uv/getting-started/installation/
- uv PyTorch guide: https://docs.astral.sh/uv/guides/integration/pytorch/
