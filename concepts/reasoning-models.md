---
title: Reasoning models and test-time compute
category: concepts
tags: [reasoning, test-time-compute, chain-of-thought, extended-thinking, rlvr, anthropic]
use_cases:
  - "improve accuracy on multi-step math, code or planning tasks by letting the model think longer"
  - "enable extended thinking on Claude for a hard analysis step while controlling cost"
  - "decide when a reasoning model is overkill for a simple extraction or classification task"
  - "build a fraud or compliance review that needs deliberate multi-step checks"
  - "route easy queries to a fast model and hard ones to a reasoning model"
status: draft
last_verified: 2026-10-03
sources:
  - https://arxiv.org/abs/2201.11903
  - https://arxiv.org/abs/2408.03314
  - https://arxiv.org/abs/2501.12948
  - https://docs.claude.com/en/docs/build-with-claude/extended-thinking
---

# Reasoning models and test-time compute

## Summary
Reasoning models are trained (largely with reinforcement learning on verifiable tasks) to produce a long internal chain of thought before the final answer. Spending more tokens at inference time ("test-time compute") buys accuracy on hard problems, trading latency and cost for quality, and is controlled by a thinking budget or effort setting.

## Key concepts
- **Chain-of-thought (CoT)**: writing intermediate steps improves multi-step accuracy; reasoning models do this natively and at length.
- **Test-time scaling**: accuracy improves with more thinking tokens, parallel samples (best-of-n, majority vote) or search with a verifier; returns diminish.
- **Training**: RL with verifiable rewards (math answers, unit tests) teaches self-checking, backtracking and exploration (e.g. DeepSeek-R1; see [[rlhf-and-preference-optimization]]).
- **Thinking tokens are billed** as output tokens and count toward limits, even when only a summary is shown.
- **Extended thinking (Claude)**: request param `thinking={"type": "enabled", "budget_tokens": N}`; the budget has a documented minimum and must be below `max_tokens`. Newer Claude models may offer adaptive or effort-based controls, so check the current docs for your model before hardcoding.
- **Interleaved with tools**: thinking blocks must be passed back unmodified in tool-use loops; see the docs.
- Visible reasoning is not guaranteed to be a faithful account of why the model answered.

## When to use / scenarios
- Hard maths, algorithmic coding, debugging, planning, multi-document analysis, ambiguous policy checks.
- Agents where one wrong step cascades ([[agents]]).
- Real-world: insurance claims adjudication rules with many conditions; code review of a tricky concurrency bug; financial reconciliation anomalies.
- Not when: simple extraction, classification, short chat, latency-critical UI, or high-volume cheap tasks. A standard model with a good prompt is cheaper and often equal.

## Setup & code
`pip install anthropic`, set `ANTHROPIC_API_KEY`. Choose the model ID from the current models page; do not copy a stale name.

```python
import anthropic

client = anthropic.Anthropic()
MODEL = "<a Claude model that supports extended thinking>"   # see docs.claude.com model list

resp = client.messages.create(
    model=MODEL,
    max_tokens=8000,                                   # must exceed budget_tokens
    thinking={"type": "enabled", "budget_tokens": 4000},
    messages=[{"role": "user", "content": "Is 2^61 - 1 prime? Show your checks."}],
)
for block in resp.content:
    if block.type == "thinking":
        print("[thinking]", block.thinking[:200])
    elif block.type == "text":
        print(block.text)
```
Open models: reasoning-tuned checkpoints (for example DeepSeek-R1 distills, Qwen reasoning variants) run via Transformers/vLLM and emit their reasoning in the output, often between think tags.

## Choosing / trade-offs
- Budget: start small (a few thousand tokens), raise only while task accuracy on your eval improves.
- Reasoning model vs standard model + CoT prompting: reasoning model is more reliable on hard tasks, costlier and slower.
- Single long think vs self-consistency (n samples + vote): voting parallelises latency but multiplies cost.
- Route by difficulty with a cheap classifier or a fast model first.

## Gotchas
- Billing surprises: thinking tokens can dwarf the visible answer.
- Streaming and timeouts: long thinking needs streaming or generous timeouts.
- Some sampling params (temperature, forced tool choice) are restricted when thinking is on; check docs.
- Do not display raw reasoning to end users as a justification; it may be wrong or unsafe.
- "Overthinking": simple questions can get worse or just slower with high budgets.
- Do not put chain-of-thought into training data or prompts blindly; follow vendor guidance on thinking blocks.

## Related
- [[rlhf-and-preference-optimization]] - how reasoning is trained.
- [[prompt-engineering]] - CoT prompting for non-reasoning models.
- [[anthropic-claude]] - model line-up and API details.
- [[agents]] - where reasoning helps most.
- [[cost-and-latency]] - budgeting thinking tokens.
- [[decoding-and-sampling]] - best-of-n and self-consistency.

## References
- Wei et al., Chain-of-Thought: https://arxiv.org/abs/2201.11903
- Snell et al., Scaling test-time compute: https://arxiv.org/abs/2408.03314
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- Claude extended thinking: https://docs.claude.com/en/docs/build-with-claude/extended-thinking
