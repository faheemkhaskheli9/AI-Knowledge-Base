---
title: Recursive text splitter from scratch (separator hierarchy, greedy packing, overlap, chunk-quality checks)
category: llm-apps
tags: [chunking, text-splitter, recursive-character-text-splitter, chunk-size, chunk-overlap, rag, document-processing, langchain, python, from-scratch]
use_cases:
  - "split documents into chunks for RAG without cutting sentences or facts in half"
  - "choose chunk size and overlap and check what they do to the chunks before indexing"
  - "understand what LangChain's RecursiveCharacterTextSplitter actually does with separators and overlap"
status: stable
last_verified: 2026-10-05
sources:
  - https://github.com/langchain-ai/langchain/tree/master/libs/text-splitters
  - https://arxiv.org/abs/2005.11401
---

# Recursive text splitter from scratch (separator hierarchy, greedy packing, overlap, chunk-quality checks)

## Summary
Before a document can be embedded for RAG, it has to be cut into chunks that fit the embedding model and say one thing each. The simplest cut, every N characters, slices through sentences and through exactly the facts a query is looking for. The recursive splitter popularised by LangChain fixes that: split on the coarsest boundary that exists (blank line, then newline, then sentence end, then space, then any character), recurse only into pieces still too long, then pack neighbouring pieces greedily up to the size limit. This file implements both splitters in plain Python on a synthetic manual with known "fact" sentences and measures how many facts and sentences survive whole, how chunks end, and what overlap really costs.

## Key concepts
- **Separator hierarchy.** Try `"\n\n"` (paragraphs), then `"\n"`, `". "`, `" "`, and finally `""` (characters). A piece is split further only if it is longer than the limit, so most chunks end on a paragraph or sentence boundary.
- **Greedy packing.** After splitting, adjacent pieces are merged until the next one would exceed `size`. Without this step you would get thousands of tiny one-sentence chunks.
- **Overlap.** Carry the tail of the previous chunk into the next, so text near a boundary appears in both. In a recursive splitter the overlap is made of whole pieces, so it only happens when the pieces are shorter than the overlap.
- **Length unit.** Characters here (as in LangChain's default `len`). Embedding models limit tokens, roughly 3–4 characters per token for English and fewer for other scripts; count tokens with the model's tokenizer when you are near the limit.
- **Chunk quality checks.** Before indexing, measure: are known facts intact in at least one chunk, do chunks end at sentence boundaries, how much extra text does overlap store, and how variable are chunk sizes.

## When to use / scenarios
- Default chunker for RAG over prose: manuals, policies, wikis, transcripts, support articles ([[rag-basics]]).
- Pre-processing before BM25 or hybrid indexing, where a chunk is the unit that gets a score ([[bm25-from-scratch]], [[reciprocal-rank-fusion-from-scratch]]).
- A baseline before trying structure-aware chunking (by Markdown heading, HTML section, code function) or semantic chunking ([[advanced-rag]]).
- Not for tables, code or forms on its own: split those by structure first (rows, functions, fields), using the parsed layout from [[document-parsing]].

## Setup & code
Standard library only, instant. The synthetic manual has 12 sections of paragraphs. Each paragraph contains one fact sentence such as "The cache limit for region R7 is 412 requests per minute." A fact counts as intact if some chunk contains the whole sentence.

```python
import random
import re

random.seed(0)


def fixed_split(text, size, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, max(len(text) - overlap, 1), step)]


def recursive_split(text, size, overlap=0, seps=("\n\n", "\n", ". ", " ", "")):
    """Split on the coarsest separator that works, recurse into pieces that are still too long, then pack
    neighbouring pieces greedily up to `size` characters, carrying `overlap` characters of whole pieces forward."""
    sep, rest = seps[0], seps[1:]
    pieces = list(text) if sep == "" else [p + sep for p in text.split(sep)]
    pieces[-1] = pieces[-1][:len(pieces[-1]) - len(sep)] if sep else pieces[-1]   # no separator after the last piece
    atoms = []
    for p in pieces:
        atoms += [p] if len(p) <= size or not rest else recursive_split(p, size, 0, rest)
    chunks, cur = [], []
    for a in atoms:
        if cur and sum(map(len, cur)) + len(a) > size:
            chunks.append("".join(cur))
            while cur and (sum(map(len, cur)) > overlap or sum(map(len, cur)) + len(a) > size):
                cur.pop(0)                                              # keep a tail of whole pieces as overlap
        cur.append(a)
    if cur:
        chunks.append("".join(cur))
    return [c for c in chunks if c.strip()]


# Synthetic manual: sections of paragraphs; one "fact" sentence per paragraph is the thing a query needs.
WORDS = ("system user request service value config token cache server client data model index query "
         "result error policy network storage backup region update version limit access").split()


def sentence():
    return " ".join(random.choice(WORDS) for _ in range(random.randint(8, 22))).capitalize() + "."


facts, paras = [], []
for sec in range(12):
    paras.append(f"## Section {sec + 1}")
    for _ in range(random.randint(3, 6)):
        f = f"The {random.choice(WORDS)} limit for region R{len(facts)} is {random.randint(100, 999)} requests per minute."
        facts.append(f)
        body = [sentence() for _ in range(random.randint(2, 7))]
        body.insert(random.randrange(len(body) + 1), f)
        paras.append(" ".join(body))
doc = "\n\n".join(paras)
sentences = [s for s in re.split(r"(?<=\.) |\n\n", doc) if s.endswith(".")]
print(f"{len(doc)} chars, {len(paras)} blocks, {len(sentences)} sentences, {len(facts)} facts")


def report(name, chunks):
    whole_facts = sum(any(f in c for c in chunks) for f in facts) / len(facts)
    whole_sents = sum(any(s in c for c in chunks) for s in sentences) / len(sentences)
    clean_end = sum(c.rstrip().endswith(".") for c in chunks) / len(chunks)        # chunk ends at a sentence end
    stored = sum(map(len, chunks)) / len(doc)
    sizes = sorted(map(len, chunks))
    print(f"{name:<28} {len(chunks):4d} chunks  size min/median/max {sizes[0]:4d}/{sizes[len(sizes) // 2]:4d}/{sizes[-1]:4d}"
          f"  facts intact {whole_facts:4.0%}  sentences intact {whole_sents:4.0%}  ends cleanly {clean_end:4.0%}"
          f"  stored {stored:.2f}x")


for size in (300, 800):
    print(f"\nchunk size {size} chars")
    report("fixed, no overlap", fixed_split(doc, size))
    report(f"fixed, overlap {size // 5}", fixed_split(doc, size, size // 5))
    report("recursive, no overlap", recursive_split(doc, size))
    report(f"recursive, overlap {size // 5}", recursive_split(doc, size, size // 5))

assert all(len(c) <= 300 for c in recursive_split(doc, 300, 60))
assert "".join(recursive_split(doc, 300)).split() == doc.split()   # no overlap: chunks tile the text (bar blank chunks)
```

Output (Python 3.14):
```
28700 chars, 68 blocks, 300 sentences, 56 facts

chunk size 300 chars
fixed, no overlap              96 chunks  size min/median/max  200/ 300/ 300  facts intact  77%  sentences intact  69%  ends cleanly   3%  stored 1.00x
fixed, overlap 60             120 chunks  size min/median/max  140/ 300/ 300  facts intact 100%  sentences intact  86%  ends cleanly   3%  stored 1.25x
recursive, no overlap         125 chunks  size min/median/max   61/ 247/ 300  facts intact 100%  sentences intact 100%  ends cleanly  94%  stored 1.00x
recursive, overlap 60         125 chunks  size min/median/max   61/ 247/ 300  facts intact 100%  sentences intact 100%  ends cleanly  94%  stored 1.00x

chunk size 800 chars
fixed, no overlap              36 chunks  size min/median/max  700/ 800/ 800  facts intact  95%  sentences intact  88%  ends cleanly   3%  stored 1.00x
fixed, overlap 160             45 chunks  size min/median/max  540/ 800/ 800  facts intact 100%  sentences intact 100%  ends cleanly   4%  stored 1.25x
recursive, no overlap          50 chunks  size min/median/max  289/ 614/ 797  facts intact 100%  sentences intact 100%  ends cleanly  80%  stored 1.00x
recursive, overlap 160         50 chunks  size min/median/max  289/ 626/ 799  facts intact 100%  sentences intact 100%  ends cleanly  80%  stored 1.00x
```

How to read it:
- Fixed-size chunks of 300 characters cut 23% of the facts in half, and 97% of chunks end mid-sentence. A query for "region R7 limit" then retrieves a chunk that holds the region but not the number.
- Overlap of 20% rescues every fact here, but stores 1.25x the text (25% more embeddings and index space) and still leaves 14% of sentences broken at size 300. Overlap repairs fixed-size cuts at a cost; it does not prevent them.
- The recursive splitter keeps every fact and sentence intact with no extra storage. It produces more, smaller chunks (median 247 of 300 allowed) because it stops at a paragraph boundary rather than filling every chunk to the limit.
- Overlap does nothing for the recursive splitter in this document: the same 125 chunks at size 300, because a whole paragraph is longer than the 60-character overlap, so no whole piece fits in the carry-over. LangChain's splitter behaves the same way. If you rely on overlap, check it is actually happening.
- At size 800, 20% of recursive chunks end without a full stop. Those are chunks that end with a section header (`## Section 4`) whose body went into the next chunk, so the header is separated from its text. Structure-aware splitting (split by heading first, keep the heading with its section) fixes this.

## Choosing / trade-offs
- **Chunk size.** Small chunks (100–300 tokens) give precise matches and let more chunks fit in the prompt, but each one has less context. Large chunks (500–1000 tokens) keep context together but dilute the embedding, so a single fact in a long chunk can rank badly. Measure retrieval recall on real questions over 2–3 sizes rather than picking one from a blog post.
- **Recursive vs structure-aware vs semantic.** Recursive is the right default for prose. For Markdown, HTML or PDFs with headings, split by heading first and add the heading path (`Manual > Limits > Regions`) to each chunk. Semantic chunking (cut where adjacent sentence embeddings diverge) costs an embedding call per sentence and often gives small gains over recursive on clean documents.
- **Overlap.** 10–20% for fixed-size or token-based splitting. With a sentence-aware splitter, a smaller overlap or none is usually fine; check the stored ratio.
- **Parent-child retrieval.** Index small chunks for matching, then return the larger parent section to the LLM. This separates "what to match" from "what to read" and removes most of the size dilemma ([[advanced-rag]]).

## Gotchas
- Character limits are not token limits. A 1000-character chunk of English is about 250 tokens, but code, URLs, numbers or non-Latin scripts can be two or three times that. Measure with the embedding model's tokenizer, or the model silently truncates.
- `". "` misses sentence ends like "e.g. " (false split), "?", "!", quotes and many languages (Chinese, Japanese and Urdu full stops). Add the separators your text uses, or use a sentence segmenter.
- PDF extraction often puts a newline at the end of every visual line, so `"\n"` becomes a meaningless separator and `"\n\n"` disappears. Normalise whitespace (join lines, keep blank lines) before splitting.
- Tables and code split mid-row or mid-function become noise. Detect them and keep them as whole chunks, or convert tables to a row-per-line format first.
- Store metadata with each chunk: source document, section heading, page, character offsets. Without it you cannot cite sources or deduplicate, and re-chunking means re-parsing.
- Changing the chunker changes every chunk ID and embedding. Version the chunking config with the index so you know what produced it.

## Related
- [[rag-basics]] - where chunking sits in the retrieval pipeline.
- [[advanced-rag]] - parent-child retrieval, contextual chunk headers, semantic chunking and re-ranking.
- [[document-parsing]] - getting clean text and structure out of PDFs before splitting.
- [[bm25-from-scratch]] - keyword scoring over the chunks this produces.
- [[vector-databases]] - where chunk embeddings and their metadata are stored.
- [[embeddings]] - the models whose input limit sets the chunk size.

## References
- LangChain text splitters (`RecursiveCharacterTextSplitter` source and separators): https://github.com/langchain-ai/langchain/tree/master/libs/text-splitters
- Lewis et al. (2020), "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (retrieval over fixed passages): https://arxiv.org/abs/2005.11401
