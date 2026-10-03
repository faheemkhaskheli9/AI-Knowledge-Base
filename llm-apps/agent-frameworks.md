---
title: Agent frameworks (LangGraph, LlamaIndex, Claude Agent SDK, OpenAI Agents SDK, plain loop)
category: llm-apps
tags: [langgraph, llamaindex, claude-agent-sdk, openai-agents-sdk, orchestration, frameworks]
use_cases:
  - "choose a framework for a production support agent with human approvals"
  - "build a document-heavy RAG agent quickly"
  - "embed Claude Code-style autonomous coding/file tools into my own app"
  - "decide whether I need a framework at all for a tool-calling bot"
  - "add durable state, retries and resumability to a long-running agent"
status: draft
last_verified: 2026-10-03
sources:
  - https://code.claude.com/docs/en/agent-sdk/python
  - https://openai.github.io/openai-agents-python/
  - https://langchain-ai.github.io/langgraph/
  - https://docs.llamaindex.ai/
---

# Agent frameworks

## Summary
A framework packages the agent loop, tool plumbing, state, tracing and multi-agent patterns. The right choice depends on how much control, durability and built-in tooling you need. For many apps a 40-line loop on the provider SDK ([[agents]]) is enough; reach for a framework when you need persistence, graphs, handoffs or ready-made tools.

## Key concepts
- **Plain loop** (provider SDK + your registry): maximum transparency, no dependency churn.
- **Claude Agent SDK** (`pip install claude-agent-sdk`; `query()` + `ClaudeAgentOptions`): exposes the Claude Code agent harness as a library - built-in file/shell/web tools, permissions modes, subagents, hooks, MCP servers, in-process tools via `@tool` and `create_sdk_mcp_server`.
- **OpenAI Agents SDK** (`pip install openai-agents`): small primitives - `Agent`, `Runner`, tools (`@function_tool`), **handoffs**, **guardrails**, tracing, sessions; provider-agnostic via LiteLLM-style adapters.
- **LangGraph**: agents as a state graph (nodes, edges, conditional routing) with checkpointing, durable execution, human-in-the-loop interrupts, streaming. Part of the LangChain ecosystem; LangSmith for tracing.
- **LlamaIndex**: data/RAG-centric - loaders, indexes, query engines, agent workflows over your documents ([[rag-basics]], [[document-parsing]]).
- **Others**: Pydantic AI (typed, validation-first), CrewAI/AutoGen-style role-based multi-agent, Semantic Kernel, Haystack.
- Frameworks change fast; APIs and class names drift between versions. Pin and read current docs.

## When to use / scenarios
| Need | Pick |
|---|---|
| One agent, few tools, want zero magic | plain loop |
| Coding/file/shell agent, Claude Code behavior in your app (CI bots, internal devtools) | Claude Agent SDK |
| Triage + specialist handoffs, tracing out of the box, small API | OpenAI Agents SDK |
| Complex branching, retries, durable/resumable runs, approval interrupts | LangGraph |
| Heavy document ingestion, indexes, hybrid retrieval pipelines | LlamaIndex |
| Typed outputs and validation as the core | Pydantic AI |

## Setup & code
Claude Agent SDK (needs Claude Code runtime available per the SDK docs):
```python
# pip install claude-agent-sdk
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions, tool, create_sdk_mcp_server

@tool("add", "Add two numbers", {"a": float, "b": float})
async def add(args):
    return {"content": [{"type": "text", "text": str(args["a"] + args["b"])}]}

calc = create_sdk_mcp_server(name="calc", version="1.0.0", tools=[add])

async def main():
    opts = ClaudeAgentOptions(mcp_servers={"calc": calc}, allowed_tools=["mcp__calc__add"])
    async for msg in query(prompt="What is 5 + 3?", options=opts):
        print(msg)
asyncio.run(main())
```
OpenAI Agents SDK:
```python
# pip install openai-agents
from agents import Agent, Runner, function_tool

@function_tool
def get_order(order_id: str) -> str:
    """Look up an order status."""
    return f"{order_id}: shipped"

agent = Agent(name="Support", instructions="Help with orders.", tools=[get_order])
print(Runner.run_sync(agent, "Status of A123?").final_output)
```
LangGraph (prebuilt ReAct agent; API names per current docs):
```python
# pip install langgraph langchain-anthropic
from langgraph.prebuilt import create_react_agent
def get_order(order_id: str) -> str:
    """Look up an order status."""
    return "shipped"
agent = create_react_agent("anthropic:claude-sonnet-5-5", tools=[get_order])
print(agent.invoke({"messages": [("user", "Status of A123?")]})["messages"][-1].content)
```

## Choosing / trade-offs
- **Control vs speed**: frameworks save weeks on persistence/tracing but hide prompts and add upgrade risk.
- **Lock-in**: agent logic expressed as graph/framework objects is harder to move; keep tools as plain functions and expose them via MCP ([[model-context-protocol]]) to stay portable.
- **Model/provider flexibility**: LangGraph and LlamaIndex are broadly provider-neutral; each vendor SDK is best on its own models.
- **Durability**: if runs last minutes-hours or need human approval mid-run, prefer checkpointed graphs or a workflow engine.
- **Team familiarity and ecosystem** often decide ties.

## Gotchas
- Abstractions obscure the actual prompt and token use; enable tracing and inspect raw requests ([[llm-observability]]).
- Default agent prompts are generic and sometimes verbose; override them.
- Version churn: tutorials older than a few months may use removed APIs.
- Giving a framework agent shell/file tools is a security decision: sandbox and set permission modes ([[guardrails-and-safety]]).
- Adding a framework to a problem that a chain solves is the most common over-engineering.

## Related
- [[agents]] - the loop every framework wraps.
- [[multi-agent-systems]] - handoffs and supervisors.
- [[model-context-protocol]] - portable tool layer.
- [[coding-agents]] - Claude Code and Agent SDK relationship.
- [[llm-evaluation]] - evaluate regardless of framework.

## References
- Claude Agent SDK (Python): https://code.claude.com/docs/en/agent-sdk/python
- OpenAI Agents SDK: https://openai.github.io/openai-agents-python/
- LangGraph: https://langchain-ai.github.io/langgraph/
- LlamaIndex: https://docs.llamaindex.ai/
