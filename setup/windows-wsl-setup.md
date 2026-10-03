---
title: Windows and WSL2 for AI development
category: setup
tags: [windows, wsl2, ubuntu, cuda, docker, vscode, gpu]
use_cases:
  - "set up a Linux-style AI dev environment on a Windows 11 laptop"
  - "use my NVIDIA GPU from inside WSL2 for PyTorch or vLLM-style tooling"
  - "decide between native Windows Python and WSL2 for a project"
  - "fix slow file access or disk bloat in WSL"
status: draft
last_verified: 2026-10-03
sources:
  - https://learn.microsoft.com/en-us/windows/wsl/install
  - https://learn.microsoft.com/en-us/windows/wsl/setup/environment
  - https://docs.nvidia.com/cuda/wsl-user-guide/index.html
---

# Windows and WSL2 for AI development

## Summary
WSL2 runs a real Linux kernel on Windows, giving Linux-only tooling (vLLM, many CUDA extensions, bash scripts) a native-feeling home while keeping Windows apps. With a recent NVIDIA Windows driver, GPUs are exposed to WSL2 so PyTorch can use CUDA. For many AI tasks, native Windows also works, so choose per project.

## Key concepts
- WSL2 (default for new installs) is a lightweight VM with its own ext4 virtual disk; WSL1 is a translation layer and is not what you want for GPU work.
- GPU: the NVIDIA driver is installed on Windows only; WSL2 gets a passthrough `libcuda`. Do not install a Linux display/driver package inside WSL.
- Filesystems: Windows drives appear under `/mnt/c`; Linux files live in `\\wsl$\<Distro>` (or `\\wsl.localhost\`).
- Resource limits are set in `%UserProfile%\.wslconfig` (global) and `/etc/wsl.conf` (per distro).

## When to use / scenarios
- Tools that are Linux-first (vLLM, Triton, many research repos, Docker with GPU).
- Matching a Linux production environment on a Windows workstation.
- Native Windows is fine for: hosted-API apps, Ollama, llama.cpp binaries, plain PyTorch, VS Code work.
- Not for: needing the full GPU for display-heavy loads, or tiny-RAM machines (the VM takes memory).

## Setup & code
Install (PowerShell as Administrator, then reboot):
```powershell
wsl --install                # Ubuntu by default
wsl --list --online          # other distros
wsl --install -d <DistroName>
wsl --list --verbose         # confirm VERSION is 2
wsl --update
```
Inside Ubuntu:
```bash
sudo apt update && sudo apt install -y build-essential git curl
curl -LsSf https://astral.sh/uv/install.sh | sh
nvidia-smi                   # should show the Windows GPU
uv venv && source .venv/bin/activate
uv pip install torch --torch-backend=auto
python -c "import torch; print(torch.cuda.is_available())"
```
Work in the Linux home (`~/projects`), not `/mnt/c/...`. Open it from Windows with VS Code's WSL extension: `code .` inside WSL.

Optional resource caps in `%UserProfile%\.wslconfig` (check the Microsoft "Advanced settings configuration" page for current keys):
```ini
[wsl2]
memory=24GB
processors=8
```
then `wsl --shutdown` to apply.

Docker: Docker Desktop with the WSL2 backend (or Docker Engine inside WSL) plus the NVIDIA Container Toolkit for `--gpus` containers; follow NVIDIA/Docker docs for your setup.

## Choosing / trade-offs
- WSL2 vs native Windows: WSL2 for Linux parity and fewer build problems; native for max simplicity and direct access to Windows apps.
- WSL2 vs dual boot: WSL2 is convenient with a small GPU/virtualization overhead; dual boot gives full bare-metal speed and RAM.
- Docker in WSL vs Docker Desktop: Desktop is easier; in-WSL engine is lighter.

## Gotchas
- Cross-OS file access is slow: a repo on `/mnt/c` is much slower than in `~`, and `git` status can crawl.
- The WSL virtual disk grows but does not shrink automatically; large model caches need cleanup and compaction.
- WSL2 may default to about half the host RAM; large models need an explicit `memory=` setting.
- Do not install the Linux NVIDIA driver in WSL; it breaks the passthrough.
- Line endings: keep repos LF; CRLF checked out by Windows git breaks shell scripts.
- `wsl --install` only works when WSL is not already installed; otherwise use `wsl --install -d <Distro>`.
- Virtualization must be enabled in BIOS/UEFI.

## Related
- [[gpu-cuda-setup]] - driver and wheel matching.
- [[python-env-uv]] - environments inside WSL.
- [[inference-servers-vllm]] - Linux-only serving stack.
- [[ollama]] - runs natively on Windows too.

## References
- Install WSL: https://learn.microsoft.com/en-us/windows/wsl/install
- WSL dev environment: https://learn.microsoft.com/en-us/windows/wsl/setup/environment
- CUDA on WSL: https://docs.nvidia.com/cuda/wsl-user-guide/index.html
