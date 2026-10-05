---
title: Grammar-constrained decoding from scratch (DFA token masks, dead-state pruning, guaranteed-valid output)
category: llm-apps
tags: [constrained-decoding, structured-output, json-mode, logit-masking, finite-state-machine, regex, grammar, tokenization, outlines, xgrammar, llama-cpp, numpy, from-scratch]
use_cases:
  - "guarantee an LLM's output matches a JSON schema, regex or grammar instead of retrying on parse errors"
  - "understand how structured outputs, JSON mode, GBNF grammars and Outlines work under the hood"
  - "debug a constrained generation that stops early, loops, or produces odd values"
  - "decide between prompting plus validation, provider structured outputs and local grammar decoding"
status: draft
last_verified: 2026-10-05
sources:
  - https://arxiv.org/abs/2307.09702
  - https://arxiv.org/abs/2411.15100
  - https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md
---

# Grammar-constrained decoding from scratch (DFA token masks, dead-state pruning, guaranteed-valid output)

## Summary
Constrained decoding makes a language model's output match a formal language (a regex, a JSON schema, a context-free grammar) by construction. At each step it sets the logits of every token that would break the format to −∞ before sampling, so invalid text cannot be produced. The format is compiled into an automaton over characters, and for each automaton state the set of allowed tokens is precomputed, which makes the run-time cost a lookup (Willard & Louf 2023). This file builds a small DFA for a JSON object, a toy BPE-like vocabulary with multi-character tokens, and a noisy stand-in model. It shows that masking alone is not enough: tokens that lead into states with no way to finish create dead ends, and those must be pruned.

## Key concepts
- **Format as an automaton.** Regexes and JSON schemas without recursion compile to a deterministic finite automaton (DFA) over characters. Recursive grammars (nested JSON, code) need a pushdown automaton; the masking idea is the same.
- **Token-level transitions.** Tokens are multi-character strings, so a token is allowed in state `s` if walking all its characters from `s` stays inside the DFA. Tokens can cross structural boundaries (`", "`, `true}`), which is why masks cannot be written by hand per field.
- **Precomputed masks.** `allowed[state] = {token: next_state}` is built once per format and vocabulary. Decoding then costs one lookup and one masked softmax per token.
- **Dead states.** A token can lead to a state that is valid as a prefix but from which no token sequence in the vocabulary reaches an accepting state (for example, the next required character exists only inside tokens that do not start with it). Prune them by backward reachability from the accepting states.
- **End of sequence.** EOS is allowed only in accepting states, and must be allowed there; otherwise the model is forced to keep writing.
- **Distortion.** Masking renormalises over allowed tokens at each step. This is not the model's distribution conditioned on the output being valid; the model can be pushed into continuations it would never choose, such as odd values in a field.

## When to use / scenarios
- Extraction and tool calls where downstream code parses the output: JSON for an API, function arguments, labels from a fixed set, dates in one format ([[structured-output]], [[tool-calling]]).
- Local and self-hosted models, which follow formats less reliably than frontier APIs: llama.cpp GBNF grammars, Outlines, XGrammar, and the structured-output options of vLLM and SGLang.
- Small fine-tuned models in pipelines where one malformed record breaks a batch job.
- Hosted APIs: use the provider's structured-output or strict tool-schema feature, which does this server-side, rather than building it.
- Not a substitute for semantic validation. A grammar guarantees the shape (an integer), not the meaning (a plausible age). Validate values separately.
- Not for free-form text with a loose format. A prompt plus a parser with one retry is simpler and does not distort the output.

## Setup & code
NumPy only, a second or so. The target is `{"age": <1-3 digits>, "ok": true|false}`. The DFA is built by hand; the vocabulary of 31 tokens includes whole-prefix tokens, boundary-crossing tokens and wrong-habit tokens (`null`, `yes`). The stand-in model favours in-format continuations but has noise and some bad habits.

```python
import numpy as np

rng = np.random.default_rng(0)

# Target format: {"age": <1-3 digits>, "ok": true|false}
# A hand-built DFA over characters; state -> {char: next_state}. ACCEPT is the final state.
PREFIX = '{"age": '
DFA = {}
for i, ch in enumerate(PREFIX):
    DFA[i] = {ch: i + 1}
D = "0123456789"
P = len(PREFIX)
DFA[P] = {d: P + 1 for d in D}                                  # first digit
DFA[P + 1] = {**{d: P + 2 for d in D}, ",": P + 4}             # 1 digit, or more
DFA[P + 2] = {**{d: P + 3 for d in D}, ",": P + 4}
DFA[P + 3] = {",": P + 4}                                     # at most 3 digits
s = P + 4
for ch in ' "ok": ':
    DFA[s] = {ch: s + 1}
    s += 1
DFA[s] = {"t": 100, "f": 200}
for base, word in ((100, "rue}"), (200, "alse}")):
    for i, ch in enumerate(word):
        DFA[base + i] = {ch: base + i + 1}
ACCEPT = {104, 205}


def walk(state, token):
    """Feed a multi-character token through the DFA; None if it leaves the language."""
    for ch in token:
        state = DFA.get(state, {}).get(ch)
        if state is None:
            return None
    return state


# Toy vocabulary with multi-char tokens, like a BPE vocabulary, including tokens that span a boundary
VOCAB = ['{"', 'age', '":', ' ', '{"age": ', '0', '1', '2', '3', '4', '5', '7', '9', '42', '100', '1000',
         ',', ', "', 'ok', '": ', 'true', 'false', '}', 'true}', 'false}', ' true', '"', 'yes', 'null', '\n', '<eos>']
EOS = VOCAB.index("<eos>")

# Precompute the token mask per DFA state: the trick that makes constrained decoding cheap at run time
STATES = sorted(set(DFA) | ACCEPT)
NEXT = {st: {t: walk(st, tok) for t, tok in enumerate(VOCAB) if tok != "<eos>" and walk(st, tok) is not None}
        for st in STATES}

# Prune tokens that lead to a state from which no token sequence reaches ACCEPT
live = set(ACCEPT)
while True:
    grown = live | {st for st, nxt in NEXT.items() if set(nxt.values()) & live}
    if grown == live:
        break
    live = grown
PRUNED = {st: {t: s for t, s in nxt.items() if s in live} for st, nxt in NEXT.items()}


def fake_lm_logits(text):
    """Stand-in for a model that mostly knows the format but slips, likes 'null'/'yes', and rambles after '}'."""
    logits = rng.normal(0, 1.0, len(VOCAB))
    state = walk(0, text)
    if state is not None:
        for t in NEXT[state]:
            logits[t] += 6.0                       # in-format continuations are favoured...
    logits[VOCAB.index("null")] += 1.5             # ...but some wrong habits are strong too
    logits[VOCAB.index("yes")] += 1.5
    logits[VOCAB.index("\n")] += 1.0
    if text.endswith("}"):
        logits[EOS] += 3.0
    return logits


def generate(mask_table=None, max_tokens=20):
    """mask_table=None: plain sampling. Otherwise only tokens in mask_table[state] (plus EOS at ACCEPT) may be sampled."""
    text, state = "", 0
    for steps in range(1, max_tokens + 1):
        logits = fake_lm_logits(text)
        if mask_table is not None:
            allowed = list(mask_table[state]) + ([EOS] if state in ACCEPT else [])
            if not allowed:
                return text, steps, "dead end"
            mask = np.full(len(VOCAB), -np.inf)
            mask[allowed] = 0
            logits = logits + mask
        p = np.exp(logits - logits.max())
        t = rng.choice(len(VOCAB), p=p / p.sum())
        if t == EOS:
            break
        text += VOCAB[t]
        if mask_table is not None:
            state = mask_table[state][t]
    return text, steps, "ok"


def valid(text):
    return walk(0, text) in ACCEPT


print("dead states reachable by a token:", sorted({s for nxt in NEXT.values() for s in nxt.values()} - live))
for name, table in (("unconstrained", None), ("mask, no pruning", NEXT), ("mask + pruning", PRUNED)):
    outs = [generate(table) for _ in range(2000)]
    ok = np.mean([valid(t) for t, _, _ in outs])
    dead = np.mean([r == "dead end" for _, _, r in outs])
    print(f"{name:17s}: valid {ok:6.1%}  dead ends {dead:5.1%}  mean tokens {np.mean([n for _, n, _ in outs]):4.1f}"
          f"  e.g. {outs[1][0]!r}")

print("allowed after '{\"age\": 4':  ", [VOCAB[t] for t in PRUNED[walk(0, '{"age": 4')]])
print("allowed after '{\"age\": 100':", [VOCAB[t] for t in PRUNED[walk(0, '{"age": 100')]])
```

Output (Python 3.14, NumPy 2.5):
```
dead states reachable by a token: [6, 17]
unconstrained    : valid   7.6%  dead ends  0.0%  mean tokens 15.8  e.g. '{"age": 51, "ok": true}1yes, "yes}'
mask, no pruning : valid  56.0%  dead ends 44.0%  mean tokens  9.4  e.g. '{"age": 7, "ok"'
mask + pruning   : valid 100.0%  dead ends  0.0%  mean tokens 11.3  e.g. '{"age": 410, "ok": false}'
allowed after '{"age": 4':   ['0', '1', '2', '3', '4', '5', '7', '9', '42', ',', ', "']
allowed after '{"age": 100': [',', ', "']
```

How to read it:
- Unconstrained, only 7.6% of samples are valid, even though each step favours in-format tokens by +6 logits. Errors compound over about a dozen tokens, and the most common failure is the one in the example: a correct object followed by extra text because the model did not stop.
- Masking without pruning raises validity to 56% but 44% of generations hit a dead end. The token `"` after `{"age` leads to a state (6) that needs `:` next, and no token starts with `:`: the colon only appears inside `":` and `": `, which start with `"`. The mask said `"` was a legal prefix, and it was, but nothing in the vocabulary can complete it. State 17 has the same problem after `"ok`.
- After pruning dead states, 100% of samples are valid and none dead-end. The allowed-token lists show the mechanics: after `{"age": 4` digits, `42` and both comma forms are allowed but not `100` (four digits); after three digits only the comma tokens remain.
- The valid outputs show the distortion: ages like 410 and 974 are well-formed and wrong. The mask enforces the 1–3 digit rule; the noisy model fills it with digits. Shape is guaranteed, value is not.

## Choosing / trade-offs
- **Provider structured outputs vs local grammars.** With a hosted API, use its structured-output or strict tool mode: nothing to build, and the provider handles tokenisation details. Locally, llama.cpp (GBNF grammars or JSON schema), Outlines, XGrammar and the guided-decoding options in vLLM and SGLang implement this with optimised mask computation.
- **Regex/DFA vs full grammar.** Flat schemas compile to a DFA with precomputed masks, the fastest case. Recursive JSON, code and nested structures need a pushdown automaton; libraries split tokens into context-independent ones (precomputed) and a few context-dependent ones checked at run time.
- **Constrain vs validate-and-retry.** Constraining costs mask computation and a compile step per schema, and can distort values. Prompting plus a parser with one retry costs extra calls on failures only. With a strong model and a simple schema the failure rate may be low enough that retry is cheaper; with small local models, constrain.
- **Schema design.** Field order matters because generation is left to right: put a short reasoning or evidence field before the answer field if the answer depends on it. Enums are cheaper to constrain and validate than free strings.
- **Strictness.** Very tight constraints (exact lengths, numeric ranges as regexes) increase distortion. Constrain the structure and types, then validate ranges and cross-field rules in code.

## Gotchas
- Dead ends from tokenisation, as above: any hand-rolled masking must prune states that cannot reach acceptance, or it will stall or truncate output.
- Forgetting to allow EOS in accepting states, or allowing it elsewhere. The first makes the model ramble inside the grammar (whitespace, extra digits); the second produces truncated objects.
- Whitespace in grammars: if the grammar allows unlimited whitespace, some models emit newlines until the token limit. Bound it or disallow it.
- Token healing at the prompt boundary: if the prompt ends mid-token (for example with `{"`), the most natural continuation token may be masked. Ending the prompt on a clean boundary avoids odd first tokens.
- Mask compile time for large schemas and vocabularies (100k+ tokens) can be seconds; cache compiled grammars per schema.
- Guaranteed shape hides model confusion. A model that does not know the answer still fills every required field. Allow `null` or an explicit "unknown" value in the schema where absence is legitimate.

## Related
- [[structured-output]] - provider features and libraries that apply this for you.
- [[decoding-strategies-from-scratch-numpy]] - the sampling step that the mask plugs into.
- [[tool-calling]] - function arguments, the most common constrained format.
- [[bpe-tokenizer-from-scratch]] - why tokens are multi-character strings that cross format boundaries.
- [[linear-chain-crf-from-scratch]] - another place where allowed transitions constrain a sequence.

## References
- Willard & Louf (2023), "Efficient Guided Generation for Large Language Models" (Outlines): https://arxiv.org/abs/2307.09702
- Dong et al. (2024), "XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models": https://arxiv.org/abs/2411.15100
- llama.cpp, GBNF grammar guide: https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md
