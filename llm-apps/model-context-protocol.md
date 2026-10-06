---
title: Model Context Protocol (MCP)
category: llm-apps
tags: [mcp, tools, resources, prompts, fastmcp, stdio, streamable-http, integration]
use_cases:
  - "expose our internal database and ticketing API to Claude and other AI clients"
  - "write a Python MCP server with tools for a finance or HR system"
  - "connect an agent to GitHub, Slack or a file system through existing MCP servers"
  - "make one tool integration reusable across Claude Code, Cursor and custom agents"
  - "secure a remote MCP server for a team"
status: draft
last_verified: 2026-10-03
sources:
  - https://modelcontextprotocol.io/specification/latest
  - https://github.com/modelcontextprotocol/python-sdk
  - https://platform.claude.com/docs/en/agents-and-tools/mcp-connector
---

# Model Context Protocol (MCP)

## Summary
MCP is an open JSON-RPC protocol that standardizes how AI applications (hosts/clients) connect to external tools, data and prompt templates (servers). Write an integration once as an MCP server and any MCP-capable client (Claude, Claude Code, Cursor, VS Code, OpenAI Agents SDK, custom agents) can use it, instead of N custom adapters.

## Key concepts
- **Roles**: *host* (the LLM app) contains *clients*; each client connects to one *server*.
- **Server primitives**: **tools** (model-invoked functions), **resources** (readable data/URIs, application-controlled), **prompts** (user-invoked templates). Clients may offer **elicitation** (server asks the user for input); the spec also defines optional extensions such as Tasks for long-running work.
- **Transports**: **stdio** (client spawns a local subprocess; good for local tools) and **Streamable HTTP** (remote servers, multi-client; auth required). Older HTTP+SSE is legacy.
- **Spec versions** are date-stamped; the latest seen at verification was 2026-07-28. Clients and servers negotiate capabilities, so check the version your SDK supports.
- **Python SDK**: `pip install "mcp[cli]"`. The widely used high-level API is `FastMCP` (`from mcp.server.fastmcp import FastMCP`); the repository README's newest examples use a renamed `MCPServer` class, so check the README for your installed version.
- **Tool schema** is generated from type hints and docstrings - write them for the model.
- **MCP connector**: the Anthropic Messages API can call remote MCP servers directly, without your own client.

## When to use / scenarios
- You want the same integration usable from several AI clients or by several teams.
- Using off-the-shelf servers (filesystem, git, databases, SaaS apps) in a coding agent ([[coding-agents]]).
- Giving an agent controlled, audited access to internal systems ([[agents]]).
- Not needed for a single app with two functions: plain [[tool-calling]] is simpler. Beware loading dozens of servers - tool definitions eat context.

## Setup & code
Server (stdio, local) with a tool, a resource and a prompt:
```python
# pip install "mcp[cli]"      -> run: python server.py  (or: mcp dev server.py to use the Inspector)
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("expenses")

@mcp.tool()
def total_spend(category: str, month: str) -> float:
    """Total spend for a category in a month (YYYY-MM). Read-only."""
    data = {("food", "2026-09"): 412.5}
    return data.get((category, month), 0.0)

@mcp.resource("policy://{name}")
def policy(name: str) -> str:
    """Company expense policy text by name."""
    return f"Policy {name}: receipts required above 25 USD."

@mcp.prompt()
def review(report: str) -> str:
    return f"Review this expense report for policy violations:\n{report}"

if __name__ == "__main__":
    mcp.run()                              # stdio; use mcp.run(transport="streamable-http") for remote
```
Register in a client (Claude Code example): `claude mcp add expenses -- python server.py`. For desktop clients, add the command to the client's MCP JSON config.

Minimal client (official SDK):
```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(command="python", args=["server.py"])
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            print([t.name for t in (await s.list_tools()).tools])
            print(await s.call_tool("total_spend", {"category": "food", "month": "2026-09"}))
asyncio.run(main())
```
The Claude Agent SDK can also host in-process MCP servers via `create_sdk_mcp_server` and `@tool` ([[agent-frameworks]]).

## Choosing / trade-offs
- **stdio vs Streamable HTTP**: stdio = zero auth setup, one user, local; HTTP = shared, remote, needs OAuth/token auth and hosting.
- **MCP vs direct function tools**: MCP adds reuse and ecosystem at the cost of a process/protocol layer; direct tools win for private, single-app logic.
- **Tools vs resources**: use resources for read-only data the app decides to attach; tools when the model should decide to fetch/act.
- **Many small servers vs one**: fewer, curated tools beat exposing every API endpoint.
- **Hosted/third-party servers**: convenient but you trust their code and data handling.

## Gotchas
- Servers are arbitrary code with your credentials: vet before installing; prefer read-only scopes; the spec says tool descriptions from untrusted servers must be treated as untrusted.
- Tool descriptions and results are prompt-injection surfaces, including "tool poisoning" and rug-pulls when a server changes tools later ([[prompt-injection]]).
- Never write to stdout in a stdio server (it is the protocol channel); log to stderr.
- Return concise, structured results; huge payloads bloat context. Many tools degrade selection accuracy - filter what each agent sees.
- Remote servers: use proper OAuth, per-user authorization, and audit logs; do not pass through the client's token to downstream APIs blindly.
- SDK APIs are moving (FastMCP vs MCPServer naming, transport names); pin versions.

## Related
- [[tool-calling]] - the underlying mechanism.
- [[agents]] - consumers of MCP tools.
- [[coding-agents]] - Claude Code/Cursor MCP configs.
- [[agent-frameworks]] - framework MCP support.
- [[prompt-injection]] - MCP-specific attack paths.
- [[guardrails-and-safety]] - approvals and least privilege.
- [[design-tools-for-claude]] - Figma MCP and plugin options for giving Claude design context.

## References
- MCP specification: https://modelcontextprotocol.io/specification/latest
- Python SDK: https://github.com/modelcontextprotocol/python-sdk
- Anthropic MCP connector: https://platform.claude.com/docs/en/agents-and-tools/mcp-connector
