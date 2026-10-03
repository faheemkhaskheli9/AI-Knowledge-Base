---
title: Structured output (JSON schema and Pydantic)
category: llm-apps
tags: [structured-output, json-schema, pydantic, extraction, validation, instructor]
use_cases:
  - "extract invoice line items into a typed database schema"
  - "classify support tickets into a fixed enum with confidence"
  - "turn free-text clinical or legal notes into validated records"
  - "make an LLM call that downstream code can parse without regex"
  - "generate API request payloads from natural language"
status: draft
last_verified: 2026-10-03
sources:
  - https://platform.claude.com/docs/en/build-with-claude/structured-outputs
  - https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use
  - https://platform.openai.com/docs/guides/structured-outputs
---

# Structured output (JSON schema and Pydantic)

## Summary
Structured output makes the model return data matching a JSON Schema so code can consume it directly. Modern APIs enforce the schema with constrained decoding, which beats "please reply in JSON" plus retry loops. Pair it with Pydantic for types and a second, semantic validation layer.

## Key concepts
- **Schema-constrained decoding**: output is guaranteed to parse and match the schema (types, required fields, enums). It does NOT guarantee the values are correct.
- **Anthropic**: `client.messages.parse(..., output_format=PydanticModel)` returns `response.parsed_output`; raw schemas go in `output_config={"format": {"type": "json_schema", "schema": ...}}`. GA, no beta header (as of verification). `strict: true` on tool definitions gives the same guarantee for tool-call arguments.
- **OpenAI**: `client.chat.completions.parse(model=..., response_format=PydanticModel)` -> `message.parsed`; or `response_format={"type":"json_schema",...}` / strict function calling.
- **Supported-subset limits**: recursion, numeric bounds (`minimum`/`maximum`), string length limits, external `$ref` are not enforced by the API (the Anthropic SDK can strip them and validate client-side). Keep schemas flat and small.
- **Refusals / truncation**: a safety refusal or hitting `max_tokens` can yield output that does not match; check `stop_reason`.
- **Library route**: `instructor`, `pydantic-ai`, `outlines` wrap any provider with retries and validation; also the only option for models without native constrained decoding (local models: grammar/`guided_json` in vLLM, llama.cpp GBNF).

## When to use / scenarios
- Extraction: invoices, receipts, résumés, KYC documents -> typed records.
- Routing/classification: `Literal["billing","tech","other"]` + short reason.
- Agent glue: planner emits a typed plan; tool args validated before execution.
- Not needed for free-form chat answers; use plain text and optional citations.

## Setup & code
```python
# pip install anthropic pydantic
from typing import Literal
from pydantic import BaseModel, Field
import anthropic

class Ticket(BaseModel):
    category: Literal["billing", "technical", "other"]
    urgency: int = Field(description="1 (low) to 5 (critical)")
    summary: str
    needs_human: bool

client = anthropic.Anthropic()
r = client.messages.parse(
    model="claude-sonnet-5-5", max_tokens=500,
    messages=[{"role": "user", "content": "Classify: 'I was charged twice and my card is now blocked!!'"}],
    output_format=Ticket,
)
t = r.parsed_output          # Ticket instance
assert 1 <= t.urgency <= 5   # semantic check the schema could not enforce
```
OpenAI:
```python
# pip install openai pydantic
from openai import OpenAI
c = OpenAI()
res = c.chat.completions.parse(
    model="<current-chat-model>",
    messages=[{"role": "user", "content": "Classify: 'Charged twice, card blocked'"}],
    response_format=Ticket,
)
t = res.choices[0].message.parsed
```

## Choosing / trade-offs
- **Native structured output vs tool-call-as-schema**: native is simpler for "just give me JSON"; tool calls fit when the model also decides whether to act ([[tool-calling]]).
- **Strict schema vs permissive + validate-and-retry**: strict removes parse failures; retry loops still needed for semantic errors. Feed the validation error back to the model once or twice, then fail loudly.
- **Rich schema vs small schema**: field descriptions steer quality (they are prompts), but large/deep schemas raise latency and can hit complexity limits; split into two calls.
- **Reasoning field first**: add a `reasoning` string before the answer fields to improve accuracy on hard classification (decoding is left-to-right).
- **Local models**: grammar-constrained decoding is robust but slower to compile; keep enums short.

## Gotchas
- "Valid JSON" is not "true": dates, currencies, IDs can be hallucinated. Cross-check against source text (substring check for extracted quotes).
- Use `null`/Optional and an explicit "unknown" enum value, or the model will invent values to fill required fields.
- Pydantic constraints (`ge`, `max_length`) may be dropped from the API schema; re-validate after parsing.
- Enum value capitalization can drift in some modes; normalize.
- Streaming + schema: partial JSON is not parseable until complete; use SDK streaming helpers.
- Changing the schema invalidates any cached schema/prefix ([[prompt-caching-and-cost]]).

## Related
- [[tool-calling]] - strict tool args use the same machinery.
- [[prompt-engineering]] - field descriptions and examples.
- [[llm-evaluation]] - measure field-level accuracy against labeled data.
- [[document-parsing]] - upstream step that turns PDFs into text for extraction.
- [[text-to-sql]] - structured output for query generation.

## References
- https://platform.claude.com/docs/en/build-with-claude/structured-outputs
- https://platform.openai.com/docs/guides/structured-outputs
- Instructor: https://python.useinstructor.com/
