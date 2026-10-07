---
title: Semantic chunking from scratch (embedding-distance breakpoints for RAG)
category: llm-apps
tags: [semantic-chunking, chunking, text-splitting, rag, embeddings, cosine-distance, langchain, numpy, from-scratch]
use_cases:
  - "split long documents into topic-coherent chunks for a RAG index"
  - "decide whether semantic chunking is worth the embedding cost over a fixed-size splitter"
  - "understand what the breakpoint threshold in SemanticChunker controls"
status: stable
last_verified: 2026-10-06
sources:
  - https://github.com/FullStackRetrieval-com/RetrievalTutorials
  - https://python.langchain.com/api_reference/experimental/text_splitter/langchain_experimental.text_splitter.SemanticChunker.html
  - https://research.trychroma.com/evaluating-chunking
---

# Semantic chunking from scratch (embedding-distance breakpoints for RAG)

## Summary
Semantic chunking splits text where the meaning changes, not every N characters. It embeds each sentence (optionally with its neighbours), computes the cosine distance between consecutive sentences, and cuts where the distance is unusually high, e.g. above the 80th–95th percentile of all distances in the document. The approach was popularised by Greg Kamradt's "5 levels of text splitting" and is implemented as LangChain's `SemanticChunker`. This file implements it with a size cap and a pluggable embedder, and compares chunk topic purity against fixed-size chunks on a three-topic document.

## Key concepts
- **Sentence windows.** Each sentence is embedded together with `window` neighbours on each side. That smooths noise from short sentences; `window=0` embeds sentences alone.
- **Distance curve.** `d_i = 1 - cos(e_i, e_{i+1})`. Peaks are topic boundaries.
- **Threshold.** A percentile of the document's own distances, so it adapts to each document. Other options are mean + k·std, IQR, or a fixed value. The percentile sets roughly how many cuts you get: the 80th percentile cuts at about 20% of sentence gaps.
- **Size cap.** Without `max_sents` (or a token cap), one long uniform section becomes one huge chunk that blows the embedding model's context or dilutes retrieval. Always bound chunk size.
- **Cost.** One embedding call per sentence at indexing time, on top of the chunk embeddings, so roughly 5–20x the indexing embedding cost of fixed-size chunking.

## When to use / scenarios
- Documents that switch topics without headings: transcripts, emails, chat logs, FAQs pasted into one blob, OCR output ([[document-parsing]]).
- Small corpora where indexing cost does not matter and retrieval quality does ([[rag-basics]], [[advanced-rag]]).
- Not when structure is available: Markdown/HTML headings, code, legal sections. Split on structure first ([[recursive-text-splitter-from-scratch]]), which is free and usually as good or better.
- Not by default: benchmark results are mixed. Chroma's chunking evaluation found that a well-tuned recursive splitter is competitive with semantic methods. Measure on your own queries before paying for it.

## Setup & code
NumPy only. The stand-in embedder hashes word stems so the example runs offline. In practice, pass any real embedding function (OpenAI, Voyage, `sentence-transformers`) as `embed`.

```python
import re
import zlib
import numpy as np


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def embed(texts, dim=512):
    """Stand-in embedder: hashed bag of word stems, L2-normalised. Swap for a real model in practice."""
    out = np.zeros((len(texts), dim))
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z]+", t.lower()):
            if len(w) > 3:
                out[i, zlib.crc32(w[:5].encode()) % dim] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def semantic_chunks(text, embed=embed, window=1, percentile=80, max_sents=8):
    """Split where the cosine distance between neighbouring sentence windows spikes
    (Kamradt-style). window=1 compares sentence i (+ neighbours) to sentence i+1 (+ neighbours)."""
    sents = sentences(text)
    ctx = [" ".join(sents[max(0, i - window):i + window + 1]) for i in range(len(sents))]
    E = embed(ctx)
    dist = 1 - (E[:-1] * E[1:]).sum(axis=1)                # distance between i and i+1
    cut = np.percentile(dist, percentile)
    chunks, cur = [], [sents[0]]
    for i, d in enumerate(dist):
        if d > cut or len(cur) >= max_sents:               # size cap keeps chunks bounded
            chunks.append(" ".join(cur))
            cur = []
        cur.append(sents[i + 1])
    chunks.append(" ".join(cur))
    return chunks, dist, cut


doc = (
    "Our refund policy allows returns within thirty days. Refunds are issued to the original payment method. "
    "Refund requests need the order number and a reason. Opened software is not eligible for refunds. "
    "Shipping is free for orders over fifty dollars. Standard shipping takes three to five business days. "
    "Express shipping arrives the next business day. Shipping to remote areas may take longer. "
    "Passwords must contain at least twelve characters. Accounts lock after five failed password attempts. "
    "Two-factor authentication protects accounts from stolen passwords. Reset passwords from the account settings page."
)
truth = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]               # topic of each sentence

chunks, dist, cut = semantic_chunks(doc, window=0, percentile=75)
print("distances:", " ".join(f"{d:.2f}" for d in dist), f"| cut > {cut:.2f}")
for c in chunks:
    print(f"- [{len(sentences(c))} sents] {c[:70]}...")


def purity(chunks):
    """Share of chunks whose sentences all come from one topic."""
    sents = sentences(doc)
    idx, pure = 0, 0
    for c in chunks:
        k = len(sentences(c))
        pure += len(set(truth[idx:idx + k])) == 1
        idx += k
    return pure / len(chunks)


fixed = [" ".join(sentences(doc)[i:i + 5]) for i in range(0, 12, 5)]   # naive fixed 5-sentence chunks
print(f"topic-pure chunks: semantic {purity(chunks):.0%}, fixed-size {purity(fixed):.0%}")

assert len(" ".join(chunks).split()) == len(doc.split())   # nothing lost or duplicated
```

Output (Python 3.14, NumPy 2.5):
```
distances: 0.83 0.82 0.83 1.00 0.85 0.66 0.80 1.00 0.85 0.71 0.54 | cut > 0.85
- [4 sents] Our refund policy allows returns within thirty days. Refunds are issue...
- [4 sents] Shipping is free for orders over fifty dollars. Standard shipping take...
- [4 sents] Passwords must contain at least twelve characters. Accounts lock after...
topic-pure chunks: semantic 100%, fixed-size 33%
```

How to read it:
- The two topic boundaries (refunds to shipping, shipping to passwords) are the only gaps at distance 1.00. Those sentence pairs share no word stems. The 75th-percentile threshold (0.85) cuts exactly there, so all three chunks are single-topic.
- Fixed 5-sentence chunks straddle both boundaries, and only 1 of 3 chunks is single-topic. A query about shipping would retrieve a chunk half about refunds.
- Within-topic distances are high too (0.66–0.85), because a bag-of-words embedder sees little overlap between short sentences. A real sentence-embedding model gives a much wider gap between within-topic and cross-topic distances. With it, `window=1` and a higher percentile (90–95) are the usual settings.
- The threshold is relative: the same percentile always produces about the same *number* of cuts, even for a single-topic document. Combine it with a minimum distance or a minimum chunk size if that matters.

## Choosing / trade-offs
- **Semantic vs recursive vs fixed.** Recursive splitting on structure is the default ([[recursive-text-splitter-from-scratch]]). Semantic chunking helps on unstructured prose with topic shifts. Fixed-size with overlap is the baseline to beat. Evaluate retrieval recall on real questions ([[llm-evaluation]]).
- **Threshold type.** Percentile (LangChain default 95) is the easiest to tune. Standard-deviation and IQR thresholds behave better on documents with skewed distance distributions. A fixed distance threshold only works with one embedding model.
- **Embedding model.** Use the same model family as retrieval, or a cheap small model for chunking only. The chunker only needs relative distances, so a small local model is usually enough.
- **Alternatives.** LLM-based chunking (ask a model for boundaries or propositions) gives better boundaries at far higher cost. Late chunking (embed the whole document with a long-context model, then pool per chunk) keeps cross-chunk context. Contextual retrieval (prepend a generated summary to each chunk) addresses the same "chunk lost its context" problem ([[advanced-rag]]).

## Gotchas
- The sentence splitter drives everything. Abbreviations ("e.g.", "Dr."), decimals and missing punctuation in transcripts break regex splitting. Use a proper sentence segmenter (`pysbd`, spaCy) for real data.
- Very short sentences ("Yes.", "See below.") have noisy embeddings and create false boundaries. Windows (`window >= 1`) or merging very short sentences fixes it.
- Without a size cap, chunks can exceed the embedder's token limit and be silently truncated.
- Chunk boundaries move whenever the document or embedding model changes, so incremental re-indexing is harder than with fixed offsets. Store chunk text and IDs, not offsets.
- The extra embedding calls at indexing time are easy to forget in cost estimates ([[prompt-caching-and-cost]]).

## Related
- [[recursive-text-splitter-from-scratch]] - the structure-based splitter to try first.
- [[rag-basics]] - where chunks go after splitting.
- [[advanced-rag]] - contextual retrieval, late chunking and other fixes for chunks that lose context.
- [[mmr-diversity-reranking-from-scratch]] - another cosine-similarity tool in the same pipeline.
- [[document-parsing]] - getting clean text with structure before chunking.

## References
- Greg Kamradt, "5 Levels of Text Splitting" (RetrievalTutorials): https://github.com/FullStackRetrieval-com/RetrievalTutorials
- LangChain `SemanticChunker`: https://python.langchain.com/api_reference/experimental/text_splitter/langchain_experimental.text_splitter.SemanticChunker.html
- Chroma Research, "Evaluating Chunking Strategies for Retrieval": https://research.trychroma.com/evaluating-chunking
