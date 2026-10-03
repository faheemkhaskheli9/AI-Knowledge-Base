---
title: Prompt engineering
category: llm-apps
tags: [prompting, system-prompt, few-shot, chain-of-thought, xml-tags, anthropic, openai]
use_cases:
  - "write a reliable system prompt for a customer-support chatbot that stays in scope"
  - "extract fields from invoices or clinical notes with few-shot examples"
  - "reduce hallucinations in a legal-document summarizer"
  - "iterate on prompts safely with a regression test set"
  - "migrate a prompt from one model family to another"
status: draft
last_verified: 2026-10-03
sources:
  - https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview
  - https://platform.openai.com/docs/guides/prompt-engineering
---

# Prompt engineering

## Summary
Prompt engineering is writing the instructions, context and examples that make an LLM do a task reliably. Most quality gains come from clear task framing, good examples, structured context and a test set - not clever tricks. Treat prompts as code: version them, test them, and change one thing at a time.

## Key concepts
- **System vs user turn**: the system prompt holds role, rules, output format and tone (stable, cacheable); the user turn holds the per-request data.
- **Be explicit**: state the goal, audience, constraints, and what "done" looks like. Say what to do, not only what to avoid. Give the reason behind a rule; models generalize from reasons.
- **Few-shot examples**: 3-5 diverse, realistic input/output pairs beat paragraphs of description. Include an edge case and a "no answer" case. Wrap in tags such as `<example>`.
- **Structure with delimiters**: XML-style tags (`<document>`, `<question>`) separate instructions from untrusted data; works well on Claude and fine on GPT models.
- **Put long documents first, the question last** (long-context tasks) and ask the model to quote relevant passages before answering.
- **Reasoning**: for hard problems let the model think before answering (extended/adaptive thinking on reasoning models, or a `<thinking>` scratchpad then `<answer>`). Do not force step-by-step on trivial tasks.
- **Prefill / output priming**: start the assistant turn with `{` or a tag to force format (check the model still supports prefill; prefer structured outputs, see [[structured-output]]).
- **Prompt as spec + eval set**: every prompt change is judged on a fixed test set ([[llm-evaluation]]).

## When to use / scenarios
- Any LLM feature: it is the first and cheapest lever, before RAG, fine-tuning or agents.
- Support bot: persona, escalation rules, refusal policy, tone, answer-only-from-context rule.
- Extraction (invoices, résumés, clinical notes): schema + 3 examples + "use null if absent".
- Classification/routing: label definitions with boundary examples, then force a label enum.
- Not enough: if the model lacks knowledge, use [[rag-basics]]; if style/format cannot be reached, see [[fine-tuning-and-peft]].

## Setup & code
```python
# pip install anthropic
import anthropic
client = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY
MODEL = "claude-sonnet-5-5"             # any current model id

SYSTEM = """You are a support assistant for Acme Bank.
Answer ONLY from the <context>. If the answer is not there, say you don't know
and offer a human handoff. Never give investment advice.
Reply in 3 sentences or fewer, plain language."""

def ask(context: str, question: str) -> str:
    r = client.messages.create(
        model=MODEL, max_tokens=500, system=SYSTEM,
        messages=[{"role": "user", "content":
            f"<context>\n{context}\n</context>\n<question>{question}</question>"}],
    )
    return r.content[0].text
```
Few-shot block inside the system prompt:
```
<example><input>Card declined at store</input><output>{"intent":"card_issue"}</output></example>
<example><input>What's the weather?</input><output>{"intent":"out_of_scope"}</output></example>
```
OpenAI equivalent: `client.chat.completions.create(model=..., messages=[{"role":"system",...},{"role":"user",...}])` (or the Responses API with `instructions=`).

## Choosing / trade-offs
- **More examples vs shorter prompt**: examples improve consistency but cost tokens each call; cache the static prefix ([[prompt-caching-and-cost]]).
- **One big prompt vs prompt chain**: chains (extract -> validate -> write) are easier to debug and test; one prompt is cheaper/faster. Split when a step fails independently.
- **Reasoning on/off**: thinking improves multi-step logic, math and planning, adds latency/cost. Use a small model without thinking for classification.
- **Model-specific tuning**: older tricks (ALL CAPS, "think step by step" everywhere) can over-trigger on newer models; re-test when changing model ([[model-selection]]).

## Gotchas
- Prompts that pass 5 hand-tried cases fail on the 6th; build a 30-100 case set early.
- Contradictory instructions (be brief / be thorough) yield inconsistent output; resolve priority explicitly.
- Examples leak: the model copies example wording and label distribution. Vary them.
- User-supplied text inside the prompt is an injection vector ([[prompt-injection]]); delimit it and never let it overwrite system rules.
- Temperature 0 is not fully deterministic; do not rely on it for exactness. Some reasoning models fix or ignore sampling params ([[decoding-and-sampling]]).
- Silent truncation: check `stop_reason == "max_tokens"`.

## Related
- [[structured-output]] - enforce format instead of begging for it.
- [[llm-evaluation]] - test prompts against a golden set.
- [[prompt-caching-and-cost]] - keep stable parts at the front to cache.
- [[reasoning-models]] - when to rely on built-in thinking.
- [[hallucination-and-grounding]] - grounding instructions.

## References
- Anthropic prompt engineering overview: https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview
- OpenAI prompt engineering guide: https://platform.openai.com/docs/guides/prompt-engineering
