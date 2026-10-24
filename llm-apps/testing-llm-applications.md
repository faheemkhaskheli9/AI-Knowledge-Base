---
title: Testing LLM applications (fake clients, recorded responses, contract tests, live smoke tests)
category: llm-apps
tags: [testing, unit-tests, pytest, unittest, mocking, fake-llm, vcrpy, pytest-recording, cassettes, snapshot-testing, contract-tests, promptfoo, deepeval, ci, flaky-tests]
use_cases:
  - "write fast, deterministic CI tests for code that calls an LLM API without paying for or depending on the API"
  - "test retries, JSON parsing, tool dispatch and fallbacks around a model call"
  - "record real model responses once and replay them in tests without leaking the API key into fixtures"
  - "decide what belongs in unit tests, in recorded integration tests, in nightly live tests and in evals"
status: draft
last_verified: 2026-10-24
sources:
  - https://docs.python.org/3/library/unittest.mock.html
  - https://docs.pytest.org/en/stable/how-to/monkeypatch.html
  - https://vcrpy.readthedocs.io/en/latest/
  - https://github.com/kiwicom/pytest-recording
  - https://www.promptfoo.dev/docs/configuration/expected-outputs/
  - https://deepeval.com/docs/getting-started
---

# Testing LLM applications (fake clients, recorded responses, contract tests, live smoke tests)

## Summary
Most code in an LLM application is not the model: it builds prompts, retrieves context, parses and validates output, dispatches tools, retries, falls back and stores results. That code can and should be tested like any other code, deterministically and offline, by replacing the model with a fake client that returns scripted responses. Real model behaviour is covered by a smaller set of recorded-response tests (cassettes replayed in CI), a few live smoke tests on a schedule, and evals, which measure quality statistically rather than pass/fail. Mixing these up gives either flaky, expensive CI or tests that never exercise the model at all.

## Key concepts
- **Four layers**: (1) *unit tests* with a fake LLM: prompt assembly, parsing, validation, retries, tool routing, error paths; (2) *contract tests*: output schemas, tool schemas and prompt templates checked for accidental change; (3) *recorded integration tests*: real HTTP responses captured once into cassettes and replayed; (4) *live tests and evals*: real calls, run nightly or before release, scored with tolerant assertions or eval metrics ([[llm-evaluation]]).
- **Fake at your own seam**: wrap the SDK in a small interface (`complete(messages, **kw) -> str` or a typed response) and inject it. A fake that implements the interface is simpler and more stable than patching SDK internals with `unittest.mock.patch`, which breaks when the SDK changes its module layout.
- **Scripted fakes record calls**: a fake returns a queue of responses (including malformed JSON, refusals, tool calls, timeouts and rate-limit errors) and records each request, so a test can assert both on the output and on what was sent (model, temperature, prompt content, tool list).
- **Cassettes (VCR)**: vcrpy intercepts HTTP from `requests`, `httpx`, `aiohttp` and others, writes request/response pairs to a YAML cassette on the first run and replays them afterwards. pytest-recording wraps it as `@pytest.mark.vcr` with `--record-mode` (`none`, `once`, `new_episodes`, `all`, `rewrite`). Filter auth headers (`authorization`, `x-api-key`) with `filter_headers`, or the key is committed to the repo.
- **Assert properties, not strings**: generated text changes across model versions and even across identical calls. Assert what must hold: parses as JSON, matches the schema, cites only chunk IDs that were in the context, stays under a length, contains a required field, contains no PII, calls the expected tool with valid arguments ([[structured-output]]).
- **Snapshot (golden) tests for prompts**: rendering the prompt template for fixed inputs and comparing it with a stored snapshot catches unreviewed prompt edits; review the diff and update the snapshot on purpose ([[prompt-management-and-versioning]]).
- **Determinism is not guaranteed**: temperature 0 or a fixed `seed` reduces variation but providers do not promise identical outputs across calls or model updates; live tests need tolerance, retries are not a fix for a failing assertion.
- **Eval tools as test runners**: promptfoo runs a YAML matrix of prompts, providers and test cases with assertions such as `equals`, `contains`, `regex`, `is-json`, `javascript`, `python`, `similar` and `llm-rubric`; DeepEval provides pytest-style `assert_test` with metric thresholds and a `deepeval test run` command. Both fit layer 4, not the offline unit layer.

## When to use / scenarios
- Any LLM feature going into a CI pipeline: support bot, extraction pipeline, RAG service, agent with tools ([[rag-basics]], [[agents]]).
- Refactoring prompt-building, retrieval or output-parsing code and needing confidence that behaviour did not change.
- Testing failure handling that is hard to trigger live: malformed output, refusals, 429 and 5xx responses, timeouts, context-length errors ([[llm-api-rate-limits-and-retries]]).
- Upgrading the model or SDK: re-record cassettes, run live smoke tests and evals, compare ([[llm-as-judge-and-eval-driven-development]]).
- Not a quality measure: a green unit suite says the plumbing works, not that answers are good. Use evals, judges and human review for that ([[user-feedback-and-human-review]]).

## Setup & code
Recorded-response test with pytest-recording (vcrpy underneath). The first run with `--record-mode=once` and a real key writes `tests/cassettes/<test>.yaml`; later runs replay it offline. In CI use `--record-mode=none`, so a missing cassette fails instead of calling the API.
```bash
pip install pytest pytest-recording openai
pytest --record-mode=once tests/test_summary.py   # record locally, once
pytest --record-mode=none                         # CI: replay only, never hit the network
```
<!-- skip-check: needs the openai package, an API key on first record, and pytest -->
```python
import json
import pytest
from openai import OpenAI

@pytest.fixture(scope="module")
def vcr_config():
    # never write credentials into cassettes; also drop org/project headers if you send them
    return {"filter_headers": ["authorization", "x-api-key", "openai-organization", "openai-project"]}

@pytest.mark.vcr
def test_summary_is_valid_json():
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-6-luna",            # copy the current ID from models/openai-gpt.md
        messages=[{"role": "user", "content": 'Return JSON {"summary": str} for: The order arrived broken.'}],
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    assert set(data) == {"summary"} and 0 < len(data["summary"]) < 300
```
A promptfoo config for the live/eval layer (run with `npx promptfoo@latest eval`). Deterministic assertions first; `llm-rubric` only where a property cannot be checked in code.
```yaml
prompts: ["Classify the ticket into billing, shipping or other. Reply with JSON {\"label\": ...}.\n\n{{ticket}}"]
providers: ["openai:gpt-6-luna"]
tests:
  - vars: {ticket: "I was charged twice for order 1182"}
    assert:
      - type: is-json
      - type: javascript
        value: JSON.parse(output).label === 'billing'
  - vars: {ticket: "Parcel stuck in customs for 9 days"}
    assert:
      - type: contains
        value: shipping
```
Runnable, stdlib only: the code under test (prompt building, JSON parsing with one repair retry, a fallback, tool dispatch with argument validation) and its unit tests with a scripted fake client, including a prompt snapshot check and the error paths.
```python
import hashlib
import json
import unittest

# ---------- code under test ----------
LABELS = {"billing", "shipping", "other"}
SYSTEM = "You classify support tickets. Reply with JSON {\"label\": one of billing|shipping|other, \"reason\": str}."

class LLMError(Exception):
    pass

def build_messages(ticket, retry_note=None):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"<ticket>\n{ticket.strip()}\n</ticket>"}]
    if retry_note:
        msgs.append({"role": "user", "content": retry_note})
    return msgs

def parse(text):
    data = json.loads(text)
    if data.get("label") not in LABELS or not isinstance(data.get("reason"), str):
        raise ValueError(f"bad schema: {data!r}")
    return data

def classify(client, ticket, model="small-model"):
    note = None
    for _ in range(2):                              # one repair retry, then fall back
        try:
            text = client.complete(build_messages(ticket, note), model=model, temperature=0)
            return parse(text)
        except (ValueError, json.JSONDecodeError) as e:
            note = f"Your last reply was invalid ({type(e).__name__}). Reply with the JSON object only."
        except LLMError:
            break
    return {"label": "other", "reason": "fallback", "fallback": True}

TOOLS = {"refund": {"order_id": str, "amount": float}}

def dispatch(tool_call, handlers):
    name, args = tool_call["name"], tool_call["arguments"]
    if name not in TOOLS:
        raise PermissionError(f"model requested unknown tool {name!r}")
    schema = TOOLS[name]
    if set(args) != set(schema) or not all(isinstance(args[k], t) for k, t in schema.items()):
        raise ValueError(f"bad arguments for {name}: {args!r}")
    return handlers[name](**args)

# ---------- test double ----------
class FakeLLM:
    """Returns scripted replies in order; an Exception instance in the script is raised instead."""
    def __init__(self, *script):
        self.script, self.calls = list(script), []
    def complete(self, messages, **kw):
        self.calls.append({"messages": messages, **kw})
        reply = self.script.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

PROMPT_SNAPSHOT = hashlib.sha256(json.dumps(build_messages("example ticket")).encode()).hexdigest()

# ---------- tests ----------
class ClassifyTests(unittest.TestCase):
    def test_happy_path_and_request_shape(self):
        fake = FakeLLM('{"label": "billing", "reason": "double charge"}')
        self.assertEqual(classify(fake, "  Charged twice  ")["label"], "billing")
        call = fake.calls[0]
        self.assertEqual((call["model"], call["temperature"]), ("small-model", 0))
        self.assertIn("<ticket>\nCharged twice\n</ticket>", call["messages"][1]["content"])

    def test_repairs_invalid_json_once(self):
        fake = FakeLLM("Sure! Here you go: {label: billing}", '{"label": "shipping", "reason": "late"}')
        self.assertEqual(classify(fake, "Where is my parcel?")["label"], "shipping")
        self.assertEqual(len(fake.calls), 2)
        self.assertIn("invalid", fake.calls[1]["messages"][-1]["content"])

    def test_rejects_label_outside_enum_then_falls_back(self):
        fake = FakeLLM('{"label": "refunds", "reason": "x"}', '{"label": "urgent", "reason": "x"}')
        out = classify(fake, "Refund please")
        self.assertTrue(out.get("fallback"))
        self.assertEqual(len(fake.calls), 2)

    def test_provider_error_falls_back_without_retry_loop(self):
        fake = FakeLLM(LLMError("503"))
        self.assertTrue(classify(fake, "hello").get("fallback"))
        self.assertEqual(len(fake.calls), 1)

    def test_prompt_snapshot_unchanged(self):
        # fails on any edit to SYSTEM or build_messages: review the prompt change, then update the snapshot
        now = hashlib.sha256(json.dumps(build_messages("example ticket")).encode()).hexdigest()
        self.assertEqual(now, PROMPT_SNAPSHOT)

class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.refunds = []
        self.handlers = {"refund": lambda order_id, amount: self.refunds.append((order_id, amount)) or "ok"}

    def test_valid_tool_call(self):
        self.assertEqual(dispatch({"name": "refund", "arguments": {"order_id": "A1", "amount": 9.5}}, self.handlers), "ok")
        self.assertEqual(self.refunds, [("A1", 9.5)])

    def test_unknown_tool_is_refused(self):
        with self.assertRaises(PermissionError):
            dispatch({"name": "delete_account", "arguments": {}}, self.handlers)

    def test_wrong_argument_types_never_reach_handler(self):
        with self.assertRaises(ValueError):
            dispatch({"name": "refund", "arguments": {"order_id": "A1", "amount": "all of it"}}, self.handlers)
        self.assertEqual(self.refunds, [])

result = unittest.main(argv=["llm-tests", "-v"], exit=False).result
assert result.wasSuccessful(), "unit tests failed"
```
In a real project these live in `tests/` and run under pytest, which also collects `unittest.TestCase` classes; the fake replaces the SDK wrapper through dependency injection or a pytest fixture.

## Choosing / trade-offs
| Layer | Runs | Cost | Catches | Misses |
|---|---|---|---|---|
| Unit tests with fakes | Every commit, offline, milliseconds | None | Parsing, retries, fallbacks, tool routing, prompt assembly bugs | Anything the real model does differently |
| Contract / snapshot tests | Every commit | None | Unreviewed prompt and schema changes | Whether the change is good |
| Recorded (cassette) tests | Every commit, replayed | One recording | SDK and API shape changes at re-record, realistic payloads | Model drift after recording |
| Live smoke tests | Nightly or pre-release | Small API spend | Model or provider changes, auth, quotas | Rare failures (sample is small) |
| Evals | On prompt/model change, scheduled | Larger API spend | Quality regressions with statistics | Plumbing bugs |

- **Fake vs mock library**: a hand-written fake at your own interface survives SDK upgrades; `unittest.mock` or `respx`/`responses`-style HTTP mocks are handy for one-off error cases but couple tests to transport details.
- **Cassette freshness**: cassettes go stale silently; re-record on SDK or model upgrades and on a schedule, and review cassette diffs like code.
- **Where LLM-judged assertions belong**: `llm-rubric` and DeepEval metrics call a model, cost money and vary; keep them out of the per-commit gate unless the suite is small and thresholds have margin ([[llm-as-judge-and-eval-driven-development]]).

## Gotchas
- **API keys in cassettes**: vcrpy records request headers unless filtered; add `filter_headers` before the first recording and scan cassettes for secrets in CI. Prompts and responses in cassettes may also contain customer data; record with synthetic inputs.
- **CI silently recording**: a record mode that allows new requests plus a key in the CI environment turns tests into live calls; use `--record-mode=none` and no key in the unit-test job.
- **Exact-string assertions on generated text**: they pass today and fail on the next model update; assert schema and properties.
- **Fakes that always succeed**: a fake that returns the perfect answer tests nothing; script malformed JSON, out-of-enum labels, empty strings, refusals, tool calls with wrong types, and provider errors.
- **Patching the wrong target**: `mock.patch` must target the name where it is looked up (`myapp.llm.OpenAI`, not `openai.OpenAI`), a common reason a "mocked" test still calls the API.
- **Streaming and async paths untested**: the happy path test uses the non-streaming call while production streams; give the fake a streaming method and test partial-chunk handling ([[streaming-and-async-llm-apps]]).
- **Retries hiding failures**: a test that retries a live call until it passes measures luck; set a pass rate over N runs in evals instead.
- **Tool side effects in tests**: live agent tests that call real tools (email, payments, databases) need sandboxes or stubs ([[sandboxed-code-execution]], [[agent-evaluation]]).

## Related
- [[llm-evaluation]] - the statistical layer above these tests.
- [[llm-as-judge-and-eval-driven-development]] - eval gates in CI and judge calibration.
- [[structured-output]] - schemas the contract tests check.
- [[prompt-management-and-versioning]] - prompt versions and the snapshot tests that guard them.
- [[llm-api-rate-limits-and-retries]] - the retry and fallback logic worth unit-testing.
- [[agent-evaluation]] - testing multi-step agents with replayable tasks.
- [[tool-calling]] - tool schemas and dispatch that tests must cover.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/testing/unit-testing-and-test-doubles.md - fakes, stubs and mocks in general.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/testing/snapshot-and-golden-testing.md - golden files and snapshot review.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/testing/flaky-tests.md - why nondeterministic tests erode CI.

## References
- Python docs, `unittest.mock` (patch targets, `side_effect`): https://docs.python.org/3/library/unittest.mock.html
- pytest docs, monkeypatching and fixtures: https://docs.pytest.org/en/stable/how-to/monkeypatch.html
- vcrpy documentation (cassettes, record modes, `filter_headers`): https://vcrpy.readthedocs.io/en/latest/
- pytest-recording (`@pytest.mark.vcr`, `--record-mode`, `vcr_config`): https://github.com/kiwicom/pytest-recording
- promptfoo, Assertions and metrics (`is-json`, `javascript`, `llm-rubric` and others): https://www.promptfoo.dev/docs/configuration/expected-outputs/
- DeepEval, Getting started (`assert_test`, `deepeval test run`): https://deepeval.com/docs/getting-started
