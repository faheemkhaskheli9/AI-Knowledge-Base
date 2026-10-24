---
title: Long-document summarization (stuff, map-reduce, refine, hierarchical)
category: llm-apps
tags: [summarization, map-reduce, refine, hierarchical-summarization, long-context, chunking, faithfulness, citations]
use_cases:
  - "summarize a 300-page contract, report or book that is longer than the model's context window"
  - "produce a daily digest of thousands of support tickets or meeting transcripts"
  - "decide between pasting the whole document into a long-context model and map-reduce"
  - "write summaries that cite the section each claim came from and check them for hallucinations"
status: draft
last_verified: 2026-10-24
sources:
  - https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/long-context-tips
  - https://cookbook.openai.com/examples/summarizing_long_documents
  - https://arxiv.org/abs/2307.03172
  - https://arxiv.org/abs/2310.00785
---

# Long-document summarization (stuff, map-reduce, refine, hierarchical)

## Summary
Summarizing text longer than a comfortable prompt needs a strategy: put it all in one long-context call ("stuff"), summarize chunks independently and then combine them (map-reduce, recursively for very long inputs), or walk the document while updating a running summary (refine). The choice trades cost, latency, parallelism and how well cross-chunk connections survive. Whatever the strategy, define the summary's purpose and length up front, keep chunk-to-source references so each claim can be cited, and check faithfulness, because summaries drop and invent details in ways a quick read misses.

## Key concepts
- **Stuff (single call)**: the whole document plus instructions in one prompt. Simplest and keeps global context, now feasible for many documents with long-context models ([[long-context]]). Put the document first and the instructions and question after it, as provider long-context guides advise.
- **Map-reduce**: split into chunks, summarize each in parallel (map), then summarize the summaries (reduce). If the combined summaries still do not fit, collapse them in groups and repeat, giving a tree (hierarchical summarization).
- **Refine (incremental)**: summarize chunk 1, then feed the running summary plus chunk 2, and so on. Keeps narrative continuity but is sequential and early content gets compressed away.
- **Lost in the middle**: models use information at the start and end of a long context better than in the middle (Liu et al.), one reason very long single-call summaries can under-weight middle sections.
- **Granularity control**: a target length per level (for example 150 words per chunk, 500 for the final) and a fixed output structure (headings, bullet facts, decisions, open issues) make results comparable and stop the reduce step from paraphrasing into vagueness.
- **Query-focused vs generic**: "summarize for a lawyer checking termination clauses" is a different and easier task than "summarize". If there is a question, consider retrieval first ([[rag-basics]]).
- **Faithfulness**: the main failure is unsupported claims, then omissions. Evaluate both: check each summary claim against its source chunks, and check that required items (a rubric or key-fact list) are present ([[llm-as-judge-and-eval-driven-development]]).

## When to use / scenarios
- Legal and finance: contract or filing summaries with section citations ([[legal]], [[finance]]).
- Support and operations: daily digests of tickets, incidents or chat logs, as a batch job ([[customer-support]], [[batch-inference-and-batch-apis]]).
- Meetings and calls: transcript to decisions and action items after speech-to-text ([[speech-to-text]]).
- Research: book- or report-length summaries with a chapter-level tree you can drill into.
- Corpus-wide themes across many documents ("what are the main complaints this quarter"): map-reduce works, but [[graphrag-and-knowledge-graphs]] or [[topic-modeling]] may answer better.
- Not needed when the document fits comfortably and cost is acceptable: one well-structured call is simpler and usually at least as good.
- Not a substitute for extraction: if you need specific fields (dates, amounts, parties), use [[structured-output]] instead of a prose summary.

## Setup & code
Map-reduce with a real model (model IDs from [[anthropic-claude]]; any provider works the same way). The cheap model maps, the stronger one reduces.
<!-- skip-check: needs ANTHROPIC_API_KEY and network -->
```python
import anthropic
from concurrent.futures import ThreadPoolExecutor

client = anthropic.Anthropic()
MAP = "Summarize this section in at most 120 words. Keep names, numbers and dates exactly. Section id: {sid}"
REDUCE = ("Combine these section summaries into one summary of at most 400 words with headings "
          "Key points, Decisions, Risks. After each claim cite the section id(s) in brackets.")

def call(model, system, text, max_tokens=600):
    r = client.messages.create(model=model, max_tokens=max_tokens, system=system,
                               messages=[{"role": "user", "content": text}])
    return r.content[0].text

def summarize(chunks):                               # chunks: list of (section_id, text)
    with ThreadPoolExecutor(8) as pool:
        parts = list(pool.map(lambda c: f"[{c[0]}] " + call("claude-haiku-4-5", MAP.format(sid=c[0]), c[1]), chunks))
    return call("claude-sonnet-5-5", REDUCE, "\n\n".join(parts), max_tokens=1200)
```
Runnable, stdlib only: the plumbing around the model calls. A stand-in extractive "model" (keeps the highest-scoring sentences) makes it deterministic, so you can see token-budget chunking, recursive collapse until the input fits, section-id propagation for citations, and a support check that flags any summary sentence not found in its cited sources. Swap `fake_llm` for a real call.
```python
import re
from collections import Counter

doc = []
topics = ["pricing", "termination", "liability", "data protection", "support levels", "renewal"]
for s, t in enumerate(topics * 5):
    doc.append(f"Section {s}. The {t} clause applies from day {s * 3}. "
               f"Either party may raise {t} issues in writing. Filler text about {t} follows here. " * 3)
text = "\n".join(doc)

def n_tokens(s):                                     # stand-in for the model's tokenizer
    return len(s.split())

def chunk(text, max_tokens=150):
    """Split on paragraphs, packing them up to max_tokens; ids record where each chunk came from."""
    chunks, cur, cur_ids = [], [], []
    for i, para in enumerate(text.split("\n")):
        if cur and n_tokens(" ".join(cur + [para])) > max_tokens:
            chunks.append((tuple(cur_ids), " ".join(cur))); cur, cur_ids = [], []
        cur.append(para); cur_ids.append(i)
    chunks.append((tuple(cur_ids), " ".join(cur)))
    return chunks

def sentences(s):
    return [x.strip() for x in re.split(r"(?<=\.)\s+", s) if x.strip()]

def fake_llm(passage, k=2):
    """Stand-in summarizer: keep the k sentences with the most frequent content words."""
    sents = list(dict.fromkeys(sentences(passage)))
    freq = Counter(w for w in re.findall(r"[a-z]+", passage.lower()) if len(w) > 4)
    return " ".join(sorted(sents, key=lambda x: -sum(freq[w] for w in re.findall(r"[a-z]+", x.lower())))[:k])

def map_reduce(chunks, budget=150, level=0):
    summaries = [(ids, fake_llm(t)) for ids, t in chunks]           # map (parallel in practice)
    total = sum(n_tokens(s) for _, s in summaries)
    print(f"level {level}: {len(chunks)} chunks -> {total} tokens of summaries")
    if total <= budget or len(summaries) == 1:
        return summaries
    groups, cur = [], []                                            # collapse in budget-sized groups
    for ids, s in summaries:
        if cur and n_tokens(" ".join(x for _, x in cur + [(ids, s)])) > budget:
            groups.append(cur); cur = []
        cur.append((ids, s))
    groups.append(cur)
    merged = [(tuple(i for ids, _ in g for i in ids), " ".join(s for _, s in g)) for g in groups]
    return map_reduce(merged, budget, level + 1)

final = map_reduce(chunk(text))      # top level now fits the budget: one last reduce call merges it
paras = text.split("\n")
for ids, summary in final:
    sources = " ".join(paras[i] for i in ids)
    unsupported = [s for s in sentences(summary) if s not in sources]   # real systems: an NLI or judge check
    print(f"cites paragraphs {ids[0]}-{ids[-1]}: {summary[:90]}...")
    print("unsupported sentences:", len(unsupported))
```

## Choosing / trade-offs
| Strategy | Pick when | Watch out for |
|---|---|---|
| Stuff (one long-context call) | document fits the window with room for output; cross-section reasoning matters | cost per call scales with length; middle sections may be under-used; use prompt caching if you ask several questions of the same document ([[prompt-caching-and-cost]]) |
| Map-reduce / hierarchical | longer than the window, or you need speed via parallel calls, or a drill-down tree | details shared across chunks get lost; reduce step can blur specifics; more total calls |
| Refine | narrative order matters (stories, chronological logs) | sequential latency; early content over-compressed; one bad step propagates |
| Extract-then-summarize | you know the key facts to capture | needs a schema; misses what the schema omits |

- **Chunk size**: larger chunks keep context and need fewer calls; smaller chunks keep detail. Split on document structure (sections, headings) rather than fixed character counts ([[chunking-strategies]]).
- **Model per stage**: a small cheap model for the map step and a stronger one for the reduce step is a common cost lever; check quality on your eval set ([[model-selection]]).
- **Batch vs interactive**: digests and archive summaries run fine as batch jobs at a discount; interactive "summarize this PDF" needs streaming and a single-call path where possible ([[streaming-and-async-llm-apps]]).

## Gotchas
- No stated purpose or length gives generic, padded summaries; say who reads it and what they decide with it.
- The reduce step invents connections between chunks that are not in the text. Require citations to section ids and verify them.
- Numbers, names and negations are the most often corrupted details; tell the map step to copy them exactly, and spot-check them.
- Overlapping chunks cause duplicated points in the final summary; deduplicate in the reduce prompt or use structure-aligned chunks without overlap.
- ROUGE against a reference summary rewards wording overlap, not faithfulness; use claim-level support checks and rubric grading instead ([[bleu-and-rouge-from-scratch]], [[llm-evaluation]]).
- Scanned PDFs and tables need parsing first; summarizing OCR noise produces confident nonsense ([[document-parsing]]).
- Summaries of untrusted documents can carry injected instructions into downstream agents; treat document text as data ([[prompt-injection]]).

## Related
- [[long-context]] - when the whole document fits and how models use long inputs.
- [[chunking-strategies]] - splitting documents along their structure.
- [[prompt-engineering]] - stating purpose, length and output structure.
- [[structured-output]] - extraction when you need fields, not prose.
- [[hallucination-and-grounding]] - unsupported claims and citation grounding.
- [[llm-as-judge-and-eval-driven-development]] - grading summaries for faithfulness and coverage.
- [[batch-inference-and-batch-apis]] - running large summarization jobs cheaply.
- [[context-engineering]] - compaction, which is summarization of an agent's own history.

## References
- Anthropic long-context prompting tips: https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/long-context-tips
- OpenAI Cookbook, summarizing long documents: https://cookbook.openai.com/examples/summarizing_long_documents
- Liu et al., Lost in the Middle: How Language Models Use Long Contexts: https://arxiv.org/abs/2307.03172
- Chang et al., BooookScore: a systematic exploration of book-length summarization: https://arxiv.org/abs/2310.00785
