---
title: Sandboxed code execution for agents
category: llm-apps
tags: [sandbox, code-execution, gvisor, firecracker, microvm, docker, e2b, modal, code-interpreter, egress, secrets, agent-security]
use_cases:
  - "let an LLM agent run the Python it writes without risking the host"
  - "choose between Docker, gVisor, Firecracker microVMs and a hosted sandbox for a coding agent"
  - "lock down network egress and secrets for a data-analysis agent"
  - "set CPU, memory and time limits on model-generated code"
status: draft
last_verified: 2026-10-11
sources:
  - https://gvisor.dev/docs/
  - https://github.com/firecracker-microvm/firecracker
  - https://docs.docker.com/engine/containers/resource_constraints/
  - https://docs.docker.com/reference/cli/docker/container/run/
  - https://e2b.dev/
  - https://modal.com/docs/guide/sandboxes
  - https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool
  - https://developers.openai.com/api/docs/guides/tools-code-interpreter
  - https://developers.openai.com/api/docs/guides/tools-shell
  - https://docs.python.org/3/library/resource.html
---

# Sandboxed code execution for agents

## Summary
Coding agents, data-analysis agents and computer-use agents run code a model
wrote, often after reading untrusted input. Treat that code as hostile: run it
in an isolated runtime with no secrets, a deny-by-default network, and hard
CPU, memory, process and time limits. Isolation strength rises from plain
containers (shared host kernel) to gVisor (a user-space kernel in front of the
host) to microVMs such as Firecracker (a separate guest kernel behind
hardware virtualisation). Hosted sandboxes (E2B, Modal, the providers' own
code-execution tools) sell that isolation as an API.

## Key concepts
- **Threat model**: prompt injection ([[prompt-injection]]) turns "run the
  model's code" into "run the attacker's code". Assume the code will try to
  read secrets, call out to the network, fill the disk, fork-bomb, or escape.
- **Containers** (Docker, Podman): namespaces and cgroups on the host kernel.
  Cheap and fast, but a kernel bug is a shared escape path. Harden with a
  non-root user, dropped capabilities, `no-new-privileges`, read-only root
  filesystem and resource flags.
- **gVisor**: an application kernel written in Go that runs in user space and
  serves the sandboxed process's system calls, so the host kernel sees far
  fewer of them. Plugs into Docker and Kubernetes as the `runsc` runtime.
- **Firecracker microVMs**: minimalist VMM from AWS (built for Lambda and
  Fargate) that runs each workload in a lightweight VM with its own kernel,
  excluding unneeded devices to shrink the attack surface. E2B's sandboxes run
  on a Firecracker-based runtime; Modal sandboxes offer gVisor or a VM
  runtime.
- **Provider-hosted execution**: Anthropic's code execution tool runs in a
  sandboxed container with no internet access; OpenAI's Code Interpreter and
  hosted shell run in containers you can size (`memory_limit`) and, for the
  shell, give a domain allowlist (`network_policy`).
- **Egress policy**: default deny. If the agent must install packages or call
  an API, allow named domains through a proxy, never `0.0.0.0/0`. Egress is
  the exfiltration channel for anything the sandbox can read.
- **Secrets**: do not put API keys in the sandbox environment or filesystem.
  Keep credentials in the host-side tool layer and expose narrow tools
  ([[tool-calling]]); if code must call a service, mint a short-lived,
  scoped token per task.
- **Resource limits**: wall-clock timeout, CPU time, memory, process count,
  file size and disk quota, plus an output-size cap on what returns to the
  model.

## When to use / scenarios
- Data-analysis agents that write pandas or plotting code over user files.
- Coding agents running tests and builds ([[coding-agents]]): one disposable
  sandbox per task or branch.
- Computer-use agents: a throwaway VM or container with a virtual display
  ([[computer-use-agents]]).
- Evaluation harnesses executing model-generated solutions
  ([[agent-evaluation]]).
- Not needed for pure function calling where the model only picks arguments
  for code you wrote; validate the arguments instead ([[guardrails-and-safety]]).

## Setup & code
Minimal stdlib runner: separate interpreter in isolated mode, empty
environment, scratch directory, timeout, and POSIX rlimits where available.
This limits accidents; it is **not** a security boundary on its own, so put it
inside one of the runtimes below.
```python
import importlib
import importlib.util
import os
import subprocess
import sys
import tempfile

HAS_RLIMIT = importlib.util.find_spec("resource") is not None  # POSIX only

def _limits():
    resource = importlib.import_module("resource")
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))                   # CPU seconds
    resource.setrlimit(resource.RLIMIT_AS, (512 * 2**20, 512 * 2**20))  # address space
    resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 2**20, 10 * 2**20))  # bytes per file
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))               # processes

def run_untrusted(code, timeout=10, max_output=10_000):
    with tempfile.TemporaryDirectory() as workdir:
        try:
            p = subprocess.run(
                [sys.executable, "-I", "-c", code],  # -I: ignore env vars and user site
                cwd=workdir,
                env={"PATH": os.defpath},              # no inherited secrets
                capture_output=True, text=True, timeout=timeout,
                preexec_fn=_limits if HAS_RLIMIT else None,
            )
            return {"ok": p.returncode == 0, "stdout": p.stdout[:max_output],
                    "stderr": p.stderr[-max_output:]}
        except subprocess.TimeoutExpired:
            return {"ok": False, "stdout": "", "stderr": f"killed after {timeout}s"}

os.putenv("FAKE_TOKEN", "sk-test")  # pretend the host has a secret
print(run_untrusted("print(sum(range(10)))"))
r = run_untrusted("import os as o; print(o.environ.get('FAKE_TOKEN'))")
assert r["stdout"].strip() == "None", r
r = run_untrusted("while True: pass", timeout=2)
assert not r["ok"]
print("secret hidden, runaway loop stopped")
```
A hardened container for the same job (needs Docker on Linux; add
`--runtime=runsc` once gVisor is installed):
<!-- skip-check: shell command; needs Docker and an image -->
```bash
docker run --rm --network none --read-only --tmpfs /tmp:size=64m \
  --memory 512m --cpus 1 --pids-limit 64 \
  --cap-drop ALL --security-opt no-new-privileges --user 65534:65534 \
  -v "$PWD/job:/job:ro" python:3.13-slim timeout 30 python -I /job/main.py
```

## Choosing / trade-offs
| Option | Isolation | Start-up and ops | Fits |
|---|---|---|---|
| Subprocess + rlimits | none against a determined attacker | trivial | trusted code, CI of your own scripts |
| Hardened container | shared kernel | fast, ubiquitous | single-tenant, internal agents |
| gVisor (`runsc`) | user-space kernel between code and host | drop-in for Docker/K8s, some syscall and I/O overhead | multi-tenant code execution on your cluster |
| Firecracker / Kata microVM | separate guest kernel, hardware virtualisation | needs KVM hosts and image tooling | untrusted multi-tenant code at scale |
| Hosted sandbox (E2B, Modal, provider tools) | vendor's microVM/gVisor | API call, per-use billing, data leaves your network | fastest path; check residency and egress controls |

- Per-task disposable sandboxes beat long-lived ones: state from one user or
  injected document cannot leak into the next.
- Provider code-execution tools are simplest but run in the provider's
  environment with the provider's package set and network rules; your own
  sandbox is needed for private data that cannot leave your network.

## Gotchas
- `timeout` on the parent kills the child but not grandchildren it spawned;
  kill the whole process group or the container.
- Memory limits via `RLIMIT_AS` break some runtimes (JIT, large mmap) and do
  not exist on Windows or macOS in the same form; use cgroups or the container
  flag in production.
- Mounting the Docker socket, the repo's `.git`, cloud credential files or
  `~/.ssh` into the sandbox hands over the host.
- An allowlisted domain that hosts user content (paste sites, object storage,
  GitHub gists) is still an exfiltration route.
- Output returned to the model is another injection channel: truncate it and
  treat it as data.
- Log every executed command and its result for audit and incident response
  ([[llm-observability]]).

## Related
- [[coding-agents]] - agents that run tests and builds in the sandbox.
- [[computer-use-agents]] - disposable VMs for GUI automation.
- [[prompt-injection]] - why model-written code must be treated as hostile.
- [[guardrails-and-safety]] - the wider control layers around actions.
- [[agents]] - the tool loop the sandbox executes for.
- [[ai-security-privacy-compliance]] - data residency and audit duties.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/security/secrets-management.md - keep credentials out of the execution environment.

## References
- gVisor documentation: https://gvisor.dev/docs/
- Firecracker: https://github.com/firecracker-microvm/firecracker
- Docker resource constraints: https://docs.docker.com/engine/containers/resource_constraints/
- docker run reference: https://docs.docker.com/reference/cli/docker/container/run/
- E2B: https://e2b.dev/
- Modal Sandboxes: https://modal.com/docs/guide/sandboxes
- Anthropic code execution tool: https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool
- OpenAI Code Interpreter: https://developers.openai.com/api/docs/guides/tools-code-interpreter
- OpenAI hosted shell (network allowlist): https://developers.openai.com/api/docs/guides/tools-shell
- Python `resource` module: https://docs.python.org/3/library/resource.html
