---
title: A2A (Agent2Agent) protocol
category: llm-apps
tags: [a2a, agent2agent, agent-card, multi-agent, interoperability, json-rpc, linux-foundation]
use_cases:
  - "let our support agent hand a task to a partner company's billing agent built on a different framework"
  - "publish an internal agent so other teams' agents can discover and call it"
  - "decide whether to expose a capability over MCP or A2A"
  - "track a long-running task delegated to a remote agent with streaming or push updates"
status: draft
last_verified: 2026-10-15
sources:
  - https://github.com/a2aproject/A2A
  - https://github.com/a2aproject/A2A/blob/main/docs/specification.md
  - https://aaif.io/blog/a2a-v1-0-a-builder-s-guide-part-1-discovery-tasks-and-clients
  - https://linuxfoundation.org/press/a2a-protocol-surpasses-150-organizations-lands-in-major-cloud-platforms-and-sees-enterprise-production-use-in-first-year
---

# A2A (Agent2Agent) protocol

## Summary
A2A is an open protocol for one AI agent to discover, call and track work done by another agent that runs as an independent, opaque service, possibly from another vendor or framework. An agent publishes an **Agent Card** (JSON at a well-known URL) describing its skills, endpoints and auth; clients send it **messages** and get back **tasks** with status, streaming updates and **artifacts**. Google contributed it; it is an Apache-2.0 project under the Linux Foundation, and the specification reached 1.0 in 2026. Use MCP to give an agent tools; use A2A when the thing on the other end is itself an agent.

## Key concepts
- **Client agent / remote agent (server)**: the client delegates; the remote agent does the work without exposing its prompts, memory or tools ("opacity").
- **Agent Card**: public JSON at `/.well-known/agent-card.json` (v0.3 used `agent.json`). Fields include `name`, `description`, `version`, `supportedInterfaces`, `capabilities` (`streaming`, `pushNotifications`, `extensions`, extended card), `defaultInputModes`/`defaultOutputModes`, `skills` (each with `id`, `name`, `description`, `tags`, `examples`), `securitySchemes`/`security`, and optional `signatures`.
- **supportedInterfaces** (new in 1.0, replaces the single `url`): an ordered list of `{url, protocolBinding, protocolVersion}`; the first entry is preferred. Bindings: JSON-RPC 2.0, gRPC and HTTP+JSON (REST).
- **Message**: one turn with `role` (`ROLE_USER` / `ROLE_AGENT`), `messageId` and `parts` (text, file by bytes or URL, structured data).
- **Task**: the unit of work with an `id`, optional `contextId` (groups related tasks/messages into one conversation), a status and artifacts. States: `TASK_STATE_SUBMITTED`, `WORKING`, `INPUT_REQUIRED`, `AUTH_REQUIRED`, `COMPLETED`, `FAILED`, `CANCELED`, `REJECTED` (all prefixed `TASK_STATE_`).
- **Artifact**: an output of a task (document, image, JSON) made of parts, with an `artifactId`.
- **Operations** (JSON-RPC method names in 1.0): `SendMessage`, `SendStreamingMessage`, `GetTask`, `ListTasks`, `CancelTask`, `SubscribeToTask`, push-notification config CRUD (`CreateTaskPushNotificationConfig`, ...), `GetExtendedAgentCard`. REST equivalents: `POST /message:send`, `GET /tasks/{id}`.
- **Interaction styles**: request/response (poll `GetTask`), streaming over SSE, or webhooks (push notifications) for long tasks.
- **Versioning**: clients send `A2A-Version: 1.0`; a missing value is treated as 0.3.
- **SDKs**: Python `pip install a2a-sdk`, JS `@a2a-js/sdk`, Go, Java, .NET, Rust (see the repo README).

## When to use / scenarios
- Cross-organization delegation: a travel agent asks an airline's agent to rebook; a procurement agent asks a supplier's agent for a quote.
- An enterprise with many teams building agents on different stacks (ADK, LangGraph, Claude Agent SDK, Bedrock) that need one way to call each other.
- Long-running delegated work where the caller needs status, cancellation, follow-up input (`INPUT_REQUIRED`) and results as artifacts.
- Not needed when all agents live in one process and one framework: use the framework's handoff/sub-agent mechanism ([[multi-agent-systems]], [[agent-frameworks]]).
- Not the right tool for exposing a database, API or file system to a model: that is [[model-context-protocol]] or plain [[tool-calling]].

## Setup & code
A2A is JSON over HTTP, so the core flow fits in the stdlib. This runnable sketch starts a toy remote agent on localhost that serves an Agent Card and answers `SendMessage` over JSON-RPC, then a client discovers it and calls it. Real agents should use the official SDK, which handles streaming, tasks storage, auth and validation.
```python
# Stdlib-only illustration of A2A 1.0 discovery + SendMessage (not the official SDK).
import http.client, json, threading, uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

TASKS = {}

class RemoteAgent(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if self.path == "/.well-known/agent-card.json":
            base = f"http://127.0.0.1:{self.server.server_port}"
            self._send({
                "name": "fx-quote-agent", "description": "Quotes currency conversions.", "version": "1.0.0",
                "supportedInterfaces": [{"url": f"{base}/a2a", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}],
                "capabilities": {"streaming": False, "pushNotifications": False},
                "defaultInputModes": ["text/plain"], "defaultOutputModes": ["application/json"],
                "skills": [{"id": "quote", "name": "FX quote", "description": "Convert an amount between currencies",
                            "tags": ["finance"], "examples": ["100 USD to EUR"]}],
            })
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if req.get("method") != "SendMessage":
            return self._send({"jsonrpc": "2.0", "id": req.get("id"), "error": {"code": -32601, "message": "Method not found"}})
        text = req["params"]["message"]["parts"][0]["text"]          # e.g. "100 USD to EUR"
        amount, src, _, dst = text.split()
        rate = {("USD", "EUR"): 0.9}.get((src, dst), 1.0)            # toy rate table
        task = {"id": str(uuid.uuid4()), "contextId": str(uuid.uuid4()),
                "status": {"state": "TASK_STATE_COMPLETED"},
                "artifacts": [{"artifactId": "a1", "name": "quote",
                               "parts": [{"data": {"amount": float(amount) * rate, "currency": dst}}]}]}
        TASKS[task["id"]] = task
        self._send({"jsonrpc": "2.0", "id": req["id"], "result": {"task": task}})

srv = HTTPServer(("127.0.0.1", 0), RemoteAgent)
threading.Thread(target=srv.serve_forever, daemon=True).start()
def call(method, url, body=None, headers=None):              # tiny localhost-only HTTP helper
    host, path = url.removeprefix("http://").split("/", 1)
    conn = http.client.HTTPConnection(host); conn.request(method, "/" + path, body, headers or {})
    return json.loads(conn.getresponse().read())

base = f"http://127.0.0.1:{srv.server_port}"

# --- client agent: 1) discover, 2) pick interface + skill, 3) send message
card = call("GET", f"{base}/.well-known/agent-card.json")
iface = card["supportedInterfaces"][0]                               # first = preferred
assert any(s["id"] == "quote" for s in card["skills"])
rpc = {"jsonrpc": "2.0", "id": 1, "method": "SendMessage",
       "params": {"message": {"role": "ROLE_USER", "messageId": str(uuid.uuid4()),
                              "parts": [{"text": "100 USD to EUR"}]}}}
resp = call("POST", iface["url"], json.dumps(rpc), {"Content-Type": "application/json", "A2A-Version": "1.0"})
task = resp["result"]["task"]
print(card["name"], task["status"]["state"], task["artifacts"][0]["parts"][0]["data"])
srv.shutdown()
```
With the SDK (`pip install a2a-sdk`), you implement an agent executor and let the SDK's server app serve the card and the bindings; follow the samples in the repo, since class names changed between 0.3 and 1.0.

## Choosing / trade-offs
| Question | MCP | A2A |
|---|---|---|
| What is on the other end? | A tool/data server, no reasoning of its own | An autonomous agent with its own model, tools and memory |
| Unit of exchange | Tool call -> result | Message -> task with lifecycle, artifacts, follow-ups |
| Long-running work | Optional extensions | Core: task states, streaming, push notifications, cancel |
| Discovery | Configured by the host | Agent Card at a well-known URL |
| Typical use | Give one agent capabilities | Let agents from different teams/vendors collaborate |

- They compose: an A2A remote agent typically uses MCP servers internally.
- **A2A vs in-framework handoffs**: A2A adds network hops, auth and serialization; use it at organizational or deployment boundaries, not between two functions in one service.
- **Binding**: JSON-RPC is the common default; gRPC for high-throughput internal traffic; HTTP+JSON for REST-style gateways and caching proxies.
- **Sync vs streaming vs push**: poll for short tasks; SSE when a client waits for progress; push notifications when the task outlives the connection.
- **Version**: 0.3 and 1.0 differ (card path, `supportedInterfaces`, method names, enum style); check which version each peer implements.

## Gotchas
- A remote agent's replies and artifacts are untrusted input: they can carry prompt injection back into your agent ([[prompt-injection]]). Treat them like tool results.
- Agent Cards are self-descriptions; do not trust a card's claims without authenticating the server (TLS, signed cards where supported) and authorizing the skill per caller.
- Auth is declared in the card (`securitySchemes`) but enforced by you; avoid forwarding the end user's token to third-party agents without scoped delegation.
- Delegating without a budget can loop (agent A asks B asks A); set depth limits, timeouts and cost caps ([[guardrails-and-safety]]).
- Many tutorials and libraries still show 0.3 (`/.well-known/agent.json`, `message/send`, `kind` fields); match the spec version your peers speak.
- Adoption figures (for example the Linux Foundation's "150+ organizations" in April 2026) count supporters, not production users.

## Related
- [[model-context-protocol]] - agent-to-tool protocol; A2A is agent-to-agent.
- [[multi-agent-systems]] - orchestration patterns that A2A can carry across services.
- [[agent-frameworks]] - frameworks with A2A client/server support.
- [[agents]] - the single-agent loop each A2A server wraps.
- [[agent-skills]] - "skills" in an Agent Card are advertised capabilities, unrelated to SKILL.md packages.
- [[context-engineering]] - a remote agent's condensed reply keeps the caller's context small.
- [[prompt-injection]] - remote agent output is an injection surface.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/security/oauth2-and-oidc.md - authenticating agent-to-agent calls.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/distributed-systems/idempotency-and-exactly-once.md - safe retries of delegated tasks.

## References
- A2A repository (Linux Foundation, Apache-2.0, SDK list): https://github.com/a2aproject/A2A
- A2A specification 1.0: https://github.com/a2aproject/A2A/blob/main/docs/specification.md
- AAIF, A2A v1.0 builder's guide part 1: https://aaif.io/blog/a2a-v1-0-a-builder-s-guide-part-1-discovery-tasks-and-clients
- Linux Foundation press release, 150+ organizations (April 2026): https://linuxfoundation.org/press/a2a-protocol-surpasses-150-organizations-lands-in-major-cloud-platforms-and-sees-enterprise-production-use-in-first-year
