---
title: Prompt compression (LLMLingua, LongLLMLingua, LLMLingua-2, extractive context pruning)
category: llm-apps
tags: [prompt-compression, llmlingua, longllmlingua, llmlingua-2, context-pruning, token-budget, rag, cost, latency, lost-in-the-middle]
use_cases:
  - "cut the input tokens of a RAG prompt that stuffs 20 retrieved chunks into every call"
  - "fit long meeting transcripts or tool outputs into a smaller model's context window"
  - "compress retrieved context per question without re-chunking or re-indexing the corpus"
  - "decide between prompt compression, prompt caching and a smaller top-k for an expensive LLM pipeline"
status: draft
last_verified: 2026-10-24
sources:
  - https://github.com/microsoft/LLMLingua
  - https://github.com/microsoft/LLMLingua/blob/main/DOCUMENT.md
  - https://aclanthology.org/2023.emnlp-main.825/
  - https://aclanthology.org/2024.acl-long.91/
  - https://aclanthology.org/2024.findings-acl.57/
---

# Prompt compression (LLMLingua, LongLLMLingua, LLMLingua-2, extractive context pruning)

## Summary
Prompt compression shortens the prompt before it reaches the target LLM by dropping the parts that carry little information: whole documents or sentences (coarse, extractive) and individual low-information tokens (fine). The LLMLingua family from Microsoft does this with a small model: LLMLingua scores tokens by a small causal LM's perplexity, LongLLMLingua makes that question-aware for long-context and RAG prompts, and LLMLingua-2 trains a BERT-size token classifier (keep/drop) on compressions distilled from a large LLM. It is a cost and latency lever for long, mostly-context prompts that change on every call, the case prompt caching cannot help; it costs a compressor call, some answer quality, and readable prompts.

## Key concepts
- **What gets compressed**: the LLMLingua docs split a prompt into *instruction*, *context* (documents, demonstrations) and *question*, and note that instructions and questions are sensitive to compression while context is much less so. Compress the context; pass instruction and question through.
- **Coarse vs fine**: coarse (context-level, sentence-level) filtering drops whole documents or sentences that rank low for the question; fine (token-level) filtering removes individual tokens inside what is kept. The `PromptCompressor.compress_prompt` call does both, controlled by `use_context_level_filter`, `use_sentence_level_filter` and `use_token_level_filter`.
- **LLMLingua (EMNLP 2023)**: a small causal LM computes per-token perplexity; tokens the small model finds predictable are dropped, iteratively over segments (`iterative_size`), with a budget controller spreading the ratio across prompt parts.
- **LongLLMLingua (ACL 2024)**: question-aware coarse ranking (`rank_method="longllmlingua"`, `condition_in_question`), document reordering (`reorder_context="sort"`) so the most relevant documents are not lost in the middle, and dynamic per-document ratios (`dynamic_context_compression_ratio`).
- **LLMLingua-2 (ACL 2024 Findings)**: token classification with an XLM-RoBERTa-large or multilingual BERT-base encoder (`use_llmlingua2=True`), trained on data distilled from a large LLM; task-agnostic, bidirectional context, and reported by its authors as 3x-6x faster than LLMLingua. `force_tokens` and `force_reserve_digit` keep characters the task depends on.
- **`rate` and `target_token`**: `rate` is compressed size divided by original size (0.33 keeps about a third); `target_token` overrides it with an absolute token budget. Actual ratios vary with the tokenizer.
- **Extractive alternatives without a model**: a smaller top-k, a reranker cut-off, or sentence-level selection by query overlap are all prompt compression in the coarse sense, and they are the baseline any learned compressor must beat.
- **Compression is lossy and unreadable**: token-level output drops function words and fragments numbers and names (see the outputs in the LLMLingua README), so a human cannot audit it easily and the target model must tolerate broken text.

## When to use / scenarios
- RAG prompts dominated by retrieved context that differs per question, where prompt caching gives little because the prefix is not stable ([[rag-basics]], [[prompt-caching-and-cost]]).
- Meeting transcripts, call logs and long tool outputs fed to a model with a small context window or a per-token price that matters ([[long-document-summarization]], [[speech-to-text]]).
- Agent loops whose tool results bloat the context; compress or summarize the tool output before appending it ([[context-engineering]]).
- Not as the first lever: lower top-k, add a reranker, trim boilerplate in the documents and cache the static prefix first; they are cheaper and keep the prompt readable ([[advanced-rag]], [[rerankers]], [[cost-and-latency]]).
- Not for prompts whose exact wording matters: contracts, code, structured data, legal or medical quotes, or anything the answer must cite verbatim ([[hallucination-and-grounding]]).
- Not when the bottleneck is output tokens or a reasoning model's thinking tokens; compression only shortens input.

## Setup & code
Install the package (MIT licence; it pulls in `transformers`, `torch`, `tiktoken` and `nltk`):
```bash
pip install llmlingua
```
Library calls, from the project README and `DOCUMENT.md`. The first run downloads the compressor model from Hugging Face; the default `PromptCompressor()` loads a 7B causal LM on CUDA, so pass a small model or LLMLingua-2 on CPU-only machines.
<!-- skip-check: needs llmlingua, torch and a model download from Hugging Face -->
```python
from llmlingua import PromptCompressor

docs = ["...retrieved chunk 1...", "...retrieved chunk 2...", "...retrieved chunk 3..."]
question = "What is the refund window for damaged items?"

# LLMLingua-2: token classifier, task-agnostic, runs on CPU.
compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True,
    device_map="cpu",
)
out = compressor.compress_prompt(docs, rate=0.33, force_tokens=["\n", "?"], force_reserve_digit=True)
print(out["origin_tokens"], "->", out["compressed_tokens"], out["ratio"])
print(out["compressed_prompt"])

# LongLLMLingua: question-aware ranking, reordering and dynamic ratios for RAG.
long_compressor = PromptCompressor(device_map="cuda")          # default model: a 7B causal LM
out = long_compressor.compress_prompt(
    docs,
    question=question,
    rate=0.55,
    condition_in_question="after_condition",
    reorder_context="sort",
    dynamic_context_compression_ratio=0.3,
    condition_compare=True,
    context_budget="+100",
    rank_method="longllmlingua",
)
prompt = out["compressed_prompt"]   # send this, plus the uncompressed instruction, to the target LLM
```
Runnable, stdlib only: a question-aware extractive compressor (sentence ranking by IDF-weighted overlap with the question, a token budget, most relevant sentences moved to the start and end), a fine pass that drops stopwords but keeps digits, IDs and capitalised names, and the check that matters in production: do the facts the answer needs survive, compared with naive truncation at the same budget.
```python
import math
import re
from collections import Counter

STOP = set("a an the of to in on for and or but is are was were be been it its this that these those with as by at "
           "from we our you your they their there here which who whom also very just so such than then into about "
           "over can could would should may might will shall do does did has have had not no".split())

def words(text):
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9\-./$%]*", text)

def n_tokens(text):
    return len(text.split())          # stand-in for the target model's tokenizer

def sentences(doc):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", doc) if s.strip()]

def rank_sentences(docs, question):
    sents = [(d, i, s) for d, doc in enumerate(docs) for i, s in enumerate(sentences(doc))]
    df = Counter(w for _, _, s in sents for w in {x.lower() for x in words(s)})
    n = len(sents)
    q = {w.lower() for w in words(question)} - STOP
    def score(s):
        ws = {w.lower() for w in words(s)}
        return sum(math.log(1 + n / df[w]) for w in q & ws)
    return sorted(sents, key=lambda t: score(t[2]), reverse=True), score

def keep_token(tok):
    return (tok.lower() not in STOP) or bool(re.search(r"\d", tok))

def fine_prune(sentence):
    """Token-level pass: drop stopwords, never digits, IDs or capitalised names."""
    toks = sentence.split()
    kept = [t for i, t in enumerate(toks) if keep_token(t) or (t[:1].isupper() and i > 0)]
    return " ".join(kept)

def compress(docs, question, budget, fine=True):
    ranked, score = rank_sentences(docs, question)
    chosen, used = [], 0
    for d, i, s in ranked:
        s2 = fine_prune(s) if fine else s
        cost = n_tokens(s2)
        if score(s) == 0 or used + cost > budget:
            continue
        chosen.append((score(s), d, i, s2))
        used += cost
    # "sort"-style reorder: best sentence first, second best last, the rest in between (lost-in-the-middle).
    chosen.sort(key=lambda t: t[0], reverse=True)
    ordered = chosen[:1] + sorted(chosen[2:], key=lambda t: (t[1], t[2])) + chosen[1:2]
    return "\n".join(s for _, _, _, s in ordered)

def truncate(docs, budget):
    toks = " ".join(docs).split()
    return " ".join(toks[:budget])

docs = [
    "Welcome to the Acme help centre. Our team is here to help you with any question you might have about orders.",
    "Shipping is free for orders over $50. Standard delivery takes 3-5 business days in the EU.",
    "Returns policy. You can return most items within 30 days of delivery for a full refund.",
    "Damaged items must be reported within 14 days with a photo. Refunds for damaged items are issued to the "
    "original payment method within 5 business days. Contact claims via form CL-204.",
    "Gift cards cannot be refunded. Our offices are closed on public holidays and the team will reply later.",
]
cases = [
    ("How many days do I have to report a damaged item and how is the refund paid?", ["14", "original", "CL-204"]),
    ("What is the minimum order value for free shipping?", ["$50"]),
    ("Can I get a refund on a gift card?", ["Gift", "cannot"]),
]

original = "\n".join(docs)
budget = n_tokens(original) // 3
print(f"original context: {n_tokens(original)} tokens, budget {budget}")
for question, facts in cases:
    for name, ctx in [("truncate", truncate(docs, budget)),
                      ("extractive", compress(docs, question, budget, fine=False)),
                      ("extractive+fine", compress(docs, question, budget))]:
        kept = [f for f in facts if f in ctx]
        print(f"{name:16s} tokens={n_tokens(ctx):3d} rate={n_tokens(ctx) / n_tokens(original):.2f} "
              f"facts {len(kept)}/{len(facts)}  {question[:40]}")
print("\ncompressed for Q1:\n" + compress(docs, cases[0][0], budget))
```
Treat the numbers above as a test harness, not a result: swap `compress` for an LLMLingua call, `n_tokens` for the target model's tokenizer, `facts` for your eval set's answers, and compare end-task accuracy and cost per answer, not only the ratio ([[rag-evaluation]]).

## Choosing / trade-offs
| Lever | Saves | Costs | Pick when |
|---|---|---|---|
| Lower top-k / reranker cut-off | Input tokens, readable | May drop the needed chunk | Always try first; tune with recall@k |
| Prompt caching | Price and latency of a stable prefix | Nothing on varying context | Long system prompt, tools, shared documents |
| Extractive sentence selection | Tokens, stays readable and quotable | Misses facts that do not share words with the question | Citation-heavy answers, audits |
| LLMLingua-2 | Tokens at higher ratios, task-agnostic, CPU-friendly encoder | A compressor call per request, broken text | Transcripts, mixed RAG context, many languages (multilingual model) |
| LongLLMLingua | Tokens with question-aware ranking and reordering | Small causal LM on GPU per request, more knobs | Multi-document QA with many retrieved chunks |
| Summarize with an LLM | Tokens, readable | An extra LLM call, can add errors | Agent tool outputs, long histories ([[context-engineering]]) |

- **Where the compressor runs**: it sits on the request path, so its latency adds to time-to-first-token. Compressing a long prompt with a GPU model to save a few hundred tokens on a cheap API can cost more time than it saves; measure end to end ([[llm-load-testing-and-benchmarking]]).
- **Ratio**: higher compression saves more but raises error rates on numbers, names and negations; pick the ratio per task from an eval sweep, not from a paper.
- **Compress once vs per question**: question-agnostic compression (LLMLingua-2) can be done at indexing time and cached with the chunk; question-aware compression (LongLLMLingua) has to run per request.

## Gotchas
- **Compressing the instruction or question**: instruction and question go through uncompressed; compressing them changes what the model is asked. Pass them as `instruction`/`question`, not inside `context`.
- **Numbers and IDs fragment**: token-level pruning can cut digits out of amounts and order IDs (the README's own example output shows "1 boxes x00 oflters"). Use `force_reserve_digit=True` and `force_tokens`, or keep such fields outside compression.
- **Docs and code disagree on a parameter name**: `DOCUMENT.md` shows `ratio=` in the LongLLMLingua example, while `compress_prompt` in `llmlingua/prompt_compressor.py` takes `rate`; it also lists `rank_method` defaulting to `"longllmlingua"` where the code default is `"llmlingua"`. Pass `rate` and `rank_method` explicitly.
- **Default model is large**: `PromptCompressor()` defaults to a 7B causal LM with `device_map="cuda"`; on a CPU box it fails or crawls. Name the model and device.
- **Breaks prompt caching**: compressed text changes whenever retrieval changes, and compressing the static prefix makes it vary across versions. Keep the cached prefix uncompressed and compress only the variable tail ([[prompt-caching-and-cost]]).
- **Citations and grounding**: compressed passages no longer match the source text, so quote-level citation checks fail. Keep chunk IDs outside compression and cite the original chunk ([[hallucination-and-grounding]]).
- **Safety filters see different text**: run PII redaction and injection checks on the original text, before compression; pruning can split a pattern a regex or classifier would have caught ([[pii-redaction-and-pseudonymization]], [[prompt-injection]]).
- **Reported gains are benchmark-specific**: the 20x compression and RAG improvement figures are the authors' results on their benchmarks; re-measure on your own eval set ([[reading-ai-benchmarks]]).

## Related
- [[prompt-caching-and-cost]] - the other input-cost lever; works on stable prefixes where compression does not help.
- [[cost-and-latency]] - where compression sits among routing, caching, batching and smaller models.
- [[context-engineering]] - compaction and tool-output trimming for agents.
- [[advanced-rag]] - reranking and contextual retrieval, the extractive alternatives.
- [[rag-evaluation]] - measuring whether compression lost the facts answers need.
- [[long-context]] - lost-in-the-middle, which LongLLMLingua's reordering targets.
- [[long-document-summarization]] - abstractive alternative when the whole document matters.
- [[information-theory-for-ml]] - self-information and perplexity, the signal LLMLingua uses.

## References
- LLMLingua repository and README (install, PromptCompressor examples, MIT licence): https://github.com/microsoft/LLMLingua
- LLMLingua documentation (`DOCUMENT.md`, parameters, structured and JSON compression): https://github.com/microsoft/LLMLingua/blob/main/DOCUMENT.md
- `compress_prompt` signature and `rate` definition: https://github.com/microsoft/LLMLingua/blob/main/llmlingua/prompt_compressor.py
- Jiang et al., "LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models", EMNLP 2023: https://aclanthology.org/2023.emnlp-main.825/
- Jiang et al., "LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression", ACL 2024: https://aclanthology.org/2024.acl-long.91/
- Pan et al., "LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression", Findings of ACL 2024: https://aclanthology.org/2024.findings-acl.57/
