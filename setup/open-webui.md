---
title: Open WebUI (self-hosted chat UI for Ollama and OpenAI-compatible servers)
category: setup
tags: [open-webui, chat-ui, self-hosted, ollama, vllm, openai-compatible, docker, rbac, rag, filters, pipes, mcp, api-keys]
use_cases:
  - "give my team a private ChatGPT-style web UI over our own Ollama or vLLM server"
  - "put one chat UI in front of a local model and a hosted API with per-user logins"
  - "add a company-wide filter that redacts emails before prompts reach the model"
  - "call our self-hosted models through Open WebUI's OpenAI-compatible API from a script"
status: draft
last_verified: 2026-10-24
sources:
  - https://github.com/open-webui/open-webui/blob/main/README.md
  - https://github.com/open-webui/docs/blob/main/docs/reference/env-configuration.mdx
  - https://github.com/open-webui/docs/blob/main/docs/reference/api-endpoints.md
  - https://github.com/open-webui/docs/blob/main/docs/features/authentication-access/api-keys.md
  - https://github.com/open-webui/docs/blob/main/docs/features/authentication-access/rbac/roles.md
  - https://github.com/open-webui/docs/blob/main/docs/features/extensibility/plugin/index.mdx
  - https://github.com/open-webui/docs/blob/main/docs/features/extensibility/plugin/functions/filter.mdx
  - https://github.com/open-webui/docs/blob/main/docs/license.mdx
---

# Open WebUI (self-hosted chat UI for Ollama and OpenAI-compatible servers)

## Summary
Open WebUI is a self-hosted web application that gives users a ChatGPT-style interface over models you run or rent: local Ollama models and any OpenAI-compatible API (vLLM, LM Studio, llama.cpp server, a gateway, or a hosted provider). It adds what a bare model server lacks: user accounts with roles and groups, chat history, document upload with built-in RAG, web search, tools and MCP servers, and Python plugins (Filters, Pipes, Actions) that run on the server. It installs with `pip install open-webui` or one Docker command, and also exposes its own OpenAI-compatible API, so scripts can use the same models and permissions as the UI.

## Key concepts
- **Connections**: Open WebUI is a client of model servers. `OLLAMA_BASE_URL` points at Ollama (several hosts with `OLLAMA_BASE_URLS`, `;`-separated); `OPENAI_API_BASE_URL`/`OPENAI_API_BASE_URLS` plus `OPENAI_API_KEY`/`OPENAI_API_KEYS` add OpenAI-compatible backends. Models from all connections appear in one picker.
- **Data volume**: everything (users, chats, settings, uploaded files, the default SQLite DB) lives under `/app/backend/data` in the container. Mount it as a volume (`-v open-webui:/app/backend/data`) or you lose it when the container is recreated. `DATABASE_URL` switches to PostgreSQL (install `open-webui[all]` for Postgres support with pip); changing it does not migrate data.
- **Persistent config (`ConfigVar`)**: many environment variables are read only on the first start and then stored in the database; later edits to the env var are ignored unless `ENABLE_PERSISTENT_CONFIG=False` (then UI changes are lost on restart). This is the most common "my env var does nothing" cause.
- **Users and roles**: the first account created becomes admin. New sign-ups default to the `pending` role (`DEFAULT_USER_ROLE`) until an admin approves them; `ENABLE_SIGNUP` toggles self-registration. Groups grant per-model and per-feature permissions.
- **`WEBUI_SECRET_KEY`**: signs sessions (JWTs) and encrypts stored secrets such as OAuth and MCP tokens. Standard launchers generate one into a `.webui_secret_key` file; if a recreated container loses that file, users are logged out and encrypted tokens can no longer be decrypted. Set it explicitly (`openssl rand -hex 32`).
- **API**: `POST /api/chat/completions` is OpenAI-compatible and serves every model Open WebUI knows (Ollama, OpenAI-compatible, and Function models); `GET /api/models` lists them. Authenticate with `Authorization: Bearer <key>`; API keys must be enabled by an admin (`ENABLE_API_KEYS`, default off) and non-admins need the API Keys permission. A key acts as its user with that user's permissions.
- **Plugins run arbitrary Python on the server**: Tools (functions the model can call), Filters (`inlet` before the model, `request` on every outgoing model call, `stream` on chunks, `outlet` after), Pipes (custom "models", e.g. an agent or another provider) and Actions (buttons). Only admins can create Functions; the docs warn that community listings are not vetted.
- **Licence**: since v0.6.6 the Open WebUI License is BSD-3-based with a branding clause: you may not remove the "Open WebUI" branding unless you have 50 or fewer users in a 30-day period, have written permission as a substantive contributor, or have an enterprise licence. Code up to v0.6.5 stays BSD-3.

## When to use / scenarios
- A team or company wants a private chat assistant over self-hosted models ([[ollama]], [[vllm-local-setup]]) with logins, history and document chat, without building a front end.
- A single UI over several backends: a local model for sensitive data and a hosted API (or an [[llm-gateways-and-routing]] endpoint) for hard questions, with per-group model access.
- Quick internal pilots of RAG over uploaded documents before investing in a custom pipeline ([[rag-basics]]).
- Not when the product is customer-facing with your own UX: build on the model API with [[serving-with-fastapi]] and [[streaming-and-async-llm-apps]]. For one person on one laptop, a desktop app is simpler ([[lm-studio-and-jan]]).
- Not as your only control for regulated data: add [[pii-redaction-and-pseudonymization]], logging and review per [[ai-security-privacy-compliance]].

## Setup & code
From the README (Docker; the UI is then at http://localhost:3000):
```bash
# Ollama already running on this machine
docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway \
  -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main
# Bundled Ollama in the same container, NVIDIA GPU
docker run -d -p 3000:8080 --gpus=all -v ollama:/root/.ollama -v open-webui:/app/backend/data \
  --name open-webui --restart always ghcr.io/open-webui/open-webui:ollama
# Or with pip (Python 3.11), UI on http://localhost:8080
pip install open-webui && open-webui serve
```
Point it at a vLLM server and pin the settings you care about (env var names from the configuration reference):
```bash
docker run -d -p 3000:8080 -v open-webui:/app/backend/data --name open-webui \
  -e OPENAI_API_BASE_URL=http://vllm-host:8000/v1 -e OPENAI_API_KEY=<vllm --api-key value> \
  -e WEBUI_SECRET_KEY="$(openssl rand -hex 32)" -e ENABLE_SIGNUP=false \
  ghcr.io/open-webui/open-webui:main
```
Call it from a script with an API key (Settings > Account, after an admin enables API keys):
<!-- skip-check: needs a running Open WebUI instance and an API key -->
```python
import requests

BASE, KEY = "http://localhost:3000", "YOUR_API_KEY"
headers = {"Authorization": f"Bearer {KEY}"}
models = requests.get(f"{BASE}/api/models", headers=headers, timeout=30).json()
print([m["id"] for m in models["data"]])
r = requests.post(f"{BASE}/api/chat/completions", headers=headers, timeout=120, json={
    "model": models["data"][0]["id"],
    "messages": [{"role": "user", "content": "Summarize our VPN policy in two lines."}],
})
print(r.json()["choices"][0]["message"]["content"])
```
Runnable: a Filter function in the shape the docs give (class `Filter`, pydantic `Valves`, async `inlet`/`outlet`). It masks email addresses in user messages before they reach the model and appends a notice to replies. Paste the class (without the test code below it) into Admin Panel > Functions.
```python
import asyncio
import re
from typing import Optional

from pydantic import BaseModel

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

class Filter:
    class Valves(BaseModel):
        placeholder: str = "[EMAIL]"
        notice: str = ""

    def __init__(self):
        self.valves = self.Valves()

    async def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        for m in body.get("messages", []):
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                m["content"] = EMAIL.sub(self.valves.placeholder, m["content"])
        return body

    async def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        if self.valves.notice:
            for m in body.get("messages", []):
                if m.get("role") == "assistant" and not m["content"].endswith(self.valves.notice):
                    m["content"] += "\n\n" + self.valves.notice
        return body

# Local test, not part of the plugin
f = Filter()
f.valves.notice = "(emails were masked before reaching the model)"
body = {"model": "local", "messages": [
    {"role": "user", "content": "Forward this to jane.doe@example.com and ops@corp.example"}]}
out = asyncio.run(f.inlet(body))
print(out["messages"][0]["content"])
assert "@" not in out["messages"][0]["content"]
reply = {"messages": [{"role": "assistant", "content": "Done."}]}
print(asyncio.run(f.outlet(reply))["messages"][0]["content"])
print(asyncio.run(f.outlet(reply))["messages"][0]["content"].count("masked"), "notice (outlet is idempotent)")
```

## Choosing / trade-offs
| Need | Open WebUI | Alternative |
|---|---|---|
| Team chat UI over own models, logins, RAG | Good fit, one container | Custom app if UX is the product |
| Single user, offline laptop | Works, but heavier | [[lm-studio-and-jan]], [[ollama]] CLI |
| Many concurrent users | PostgreSQL via `DATABASE_URL`, Redis (`REDIS_URL`) for multi-worker/multi-node | A dedicated serving stack behind it ([[inference-servers-vllm]]) |
| Central cost control and routing across providers | Point Open WebUI at a gateway as its OpenAI connection | [[llm-gateways-and-routing]] |
| Custom logic (redaction, routing, agents) | Filters and Pipes, admin-installed Python | Logic in a gateway or your own service, outside the UI |

- **Built-in RAG vs your own pipeline**: the built-in document chat is quick to try and configurable (several vector databases and extraction engines), but retrieval quality is tuned through admin settings rather than code; measure it like any RAG system ([[rag-evaluation]]) and move to a dedicated pipeline when you need control.
- **Env vars vs Admin UI**: with persistent config on (default), the first start seeds the database and the UI is the source of truth. For GitOps-style config, set `ENABLE_PERSISTENT_CONFIG=False` and treat the env file as the source of truth.

## Gotchas
- Forgetting the `/app/backend/data` volume loses all users and chats on the next `docker rm`.
- "Server connection error" usually means the container cannot reach Ollama at `localhost`; inside Docker use `host.docker.internal` (with `--add-host=host.docker.internal:host-gateway` on Linux) or `--network=host` (then the UI is on port 8080).
- Changing a `ConfigVar` env var after the first start has no effect; change it in the Admin UI or set `ENABLE_PERSISTENT_CONFIG=False`.
- The first person to open a fresh, internet-reachable instance becomes admin. Create the admin account before exposing it, then keep sign-up off or new users pending.
- Not setting `WEBUI_SECRET_KEY` explicitly works until the container is recreated without the key file; then sessions and stored OAuth/MCP tokens break.
- Functions, Tools and Pipelines execute arbitrary code with the server's permissions; review the code of anything you import and restrict Workspace access to trusted admins.
- When metering via the API, a reply that used tools makes several model calls: `prompt_tokens`/`completion_tokens` cover only the last call, `input_tokens`/`output_tokens` the whole reply (per the API docs).
- Keep the branding if you exceed 50 users in 30 days and have no enterprise licence or permission; the licence is not plain BSD-3 for current releases ([[model-licenses]] covers model licences, not this one).
- `HF_HUB_OFFLINE=1` stops it downloading embedding or speech models from the Hub in air-gapped installs; pre-load those models first.

## Related
- [[ollama]] - the most common local backend behind Open WebUI.
- [[vllm-local-setup]] - an OpenAI-compatible GPU server to connect as a backend.
- [[lm-studio-and-jan]] - single-user desktop alternatives.
- [[llm-gateways-and-routing]] - one OpenAI-compatible endpoint across providers to plug in.
- [[docker-for-ml]] - volumes, GPU flags and container networking.
- [[rag-basics]] - what the built-in document chat does under the hood.
- [[model-context-protocol]] - MCP servers as tools for chats.
- [[pii-redaction-and-pseudonymization]] - stronger redaction than the example filter.
- [[guardrails-and-safety]] - input and output checks to add as filters.
- [[open-weight-model-landscape]] - which models to serve behind it.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/security/authentication-and-sessions.md - sessions, SSO and account lifecycle for an internal web app.

## References
- Open WebUI README (features, pip and Docker install, Ollama connection troubleshooting, offline mode): https://github.com/open-webui/open-webui/blob/main/README.md
- Environment variable reference (`ConfigVar` persistence, `OLLAMA_BASE_URL`, `OPENAI_API_BASE_URLS`, `ENABLE_SIGNUP`, `DEFAULT_USER_ROLE`, `WEBUI_SECRET_KEY`, `DATABASE_URL`, `ENABLE_API_KEYS`): https://github.com/open-webui/docs/blob/main/docs/reference/env-configuration.mdx
- API endpoints (`/api/models`, `/api/chat/completions`, usage fields with tools): https://github.com/open-webui/docs/blob/main/docs/reference/api-endpoints.md
- API keys (admin toggle, group permission, key acts as its user): https://github.com/open-webui/docs/blob/main/docs/features/authentication-access/api-keys.md
- Roles (first user is admin, pending users): https://github.com/open-webui/docs/blob/main/docs/features/authentication-access/rbac/roles.md
- Plugins overview and security warning (Tools, Functions, Pipelines): https://github.com/open-webui/docs/blob/main/docs/features/extensibility/plugin/index.mdx
- Filter functions (`inlet`, `request`, `stream`, `outlet`, Valves): https://github.com/open-webui/docs/blob/main/docs/features/extensibility/plugin/functions/filter.mdx
- Open WebUI License (branding clause, 50-user threshold, BSD-3 up to v0.6.5): https://github.com/open-webui/docs/blob/main/docs/license.mdx
