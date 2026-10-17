---
title: Chunking strategies for RAG (fixed, recursive, semantic, late, contextual)
category: llm-apps
tags: [chunking, text-splitting, rag, recursive-splitter, semantic-chunking, late-chunking, contextual-retrieval, chunk-size, overlap, retrieval-eval]
use_cases:
  - "choose a chunking strategy and chunk size for a RAG index over manuals or contracts"
  - "fix retrieval that returns the right passage but without the product or section it belongs to"
  - "evaluate chunk size, overlap and splitter choice on our own questions before re-indexing"
  - "decide whether semantic chunking, late chunking or contextual retrieval is worth its cost"
status: draft
last_verified: 2026-10-15
sources:
  - https://research.trychroma.com/evaluating-chunking
  - https://arxiv.org/abs/2409.04701
  - https://github.com/jina-ai/late-chunking
  - https://www.anthropic.com/news/contextual-retrieval
  - https://github.com/langchain-ai/langchain/tree/master/libs/text-splitters
---

# Chunking strategies for RAG (fixed, recursive, semantic, late, contextual)

## Summary
Chunking decides what a retriever can find: a fact cut in half, or a chunk
that says "the warranty is 24 months" without saying which product, is lost
before ranking starts. The usual ladder is fixed-size, then recursive
(structure-aware) splitting as the default, then semantic chunking, late
chunking or contextual retrieval when evals show chunks lack context.
Chroma's chunking study found the strategy has a significant effect on
retrieval and that some popular defaults perform relatively poorly, so choose
by measuring on your own questions, not by a rule of thumb.

## Key concepts
- **Fixed-size**: every N characters or tokens, optional overlap. Fast and
  predictable; cuts sentences, tables and facts mid-way.
- **Recursive / structure-aware**: split on the coarsest boundary that exists
  (headings, blank lines, newlines, sentence ends, spaces), recurse into
  pieces still too long, pack neighbours up to the size limit. LangChain's
  `RecursiveCharacterTextSplitter`; built by hand in
  [[recursive-text-splitter-from-scratch]]. Markdown/HTML/code splitters do
  the same with format-specific separators.
- **Semantic chunking**: embed sentences, cut where the distance between
  neighbours jumps. More coherent chunks, costs an embedding pass; see
  [[semantic-chunking-from-scratch]].
- **Late chunking** (Günther et al., 2024): run a long-context embedding model
  over the whole document (or a long window), then mean-pool token vectors
  per chunk span. Each chunk vector carries document context, with no
  extra training and no LLM calls; needs an embedder that exposes token
  embeddings and a long enough context.
- **Contextual retrieval** (Anthropic, 2024): an LLM writes a short
  chunk-specific context ("this chunk is from Product Alpha's guide, warranty
  section") that is prepended before embedding and before BM25 indexing.
  Anthropic reported the top-20 retrieval failure rate falling from 5.7% to
  3.7% with contextual embeddings, to 2.9% adding contextual BM25, and to 1.9%
  with reranking on top; prompt caching keeps the per-chunk calls cheap
  ([[advanced-rag]], [[prompt-caching-and-cost]]).
- **Parent-document / small-to-big**: index small chunks for precise matching,
  return the parent section to the LLM.
- **Metadata over text tricks**: title, section path, page and date stored
  with each chunk (and optionally prefixed) are the cheapest form of context.
- **Token-level evaluation** (Chroma): measure precision, recall and IoU of
  retrieved tokens against the gold excerpt, so oversized chunks pay for the
  irrelevant text they carry.

## When to use / scenarios
- Default for prose and documentation: recursive splitting by headings and
  paragraphs, a few hundred tokens, small overlap; then measure.
- Manuals, catalogues, policy books with repeated templates per section:
  chunks lose the product or clause name; add section path metadata or
  contextual retrieval.
- Contracts and filings with long cross-references: parent-document retrieval
  or late chunking.
- Transcripts and chat logs without structure: semantic chunking or
  speaker-turn windows.
- Tables, slides, charts: chunking text is the wrong unit; see
  [[multimodal-rag]] and [[document-parsing]].
- Skip chunking when the whole corpus fits in the context window and can be
  cached ([[long-context]]).

## Setup & code
A small chunking eval in NumPy and stdlib: a synthetic manual where every
section repeats the same template, so a fact sentence never names its product.
It compares fixed-size, fixed with overlap, sentence-packed (a simplified
recursive splitter) and sentence-packed plus a context prefix. "Answerable@3"
counts a query as answered when a top-3 chunk holds the whole fact sentence
and names the product it belongs to. TF-IDF stands in for a real embedder.

```python
import re
import numpy as np

rng = np.random.default_rng(0)
PRODUCTS = ["alpha", "bravo", "cobalt", "delta", "ember", "falcon"]
ATTRS = ["warranty", "battery", "weight", "charger", "firmware"]
FILLER = ["The device ships in a recyclable box.", "Keep the unit away from water.",
          "Support is available on weekdays.", "Read the safety notes before first use.",
          "The housing is made of aluminium.", "Store the unit in a dry place."]

sections, facts = [], []
for p in PRODUCTS:
    sents = [f"{p.title()} product guide."]
    for a in ATTRS:
        sents += list(rng.choice(FILLER, size=2, replace=False))
        fact = f"The {a} value for this model is {rng.integers(10, 99)} units."
        sents.append(fact)
        facts.append((p, a, fact))
    sections.append((p, " ".join(sents)))
doc = "\n\n".join(s for _, s in sections)

def fixed(text, size=160, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]

def sentence_packed(text, size=160):
    out = []
    for para in text.split("\n\n"):
        cur = ""
        for s in re.split(r"(?<=\.)\s+", para):
            if cur and len(cur) + 1 + len(s) > size:
                out.append(cur)
                cur = s
            else:
                cur = f"{cur} {s}".strip()
        out.append(cur)
    return out

def with_context(chunks):
    # contextual retrieval stand-in: prefix the section's product name (an LLM writes this in practice)
    out, current = [], ""
    for c in chunks:
        m = re.search(r"(\w+) product guide", c)
        current = m.group(1).lower() if m else current
        out.append(f"[{current} product guide] {c}")
    return out

def tokens(s):
    return re.findall(r"[a-z]+", s.lower())

def evaluate(chunks, k=3):
    vocab = sorted({t for c in chunks for t in tokens(c)})
    idx = {t: i for i, t in enumerate(vocab)}
    M = np.zeros((len(chunks), len(vocab)))
    for r, c in enumerate(chunks):
        for t in tokens(c):
            M[r, idx[t]] += 1
    idf = np.log(len(chunks) / (1 + (M > 0).sum(0))) + 1
    M = M * idf
    M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    intact = hits = 0
    for p, a, fact in facts:
        intact += any(fact in c for c in chunks)
        q = np.zeros(len(vocab))
        for t in [p, a]:
            if t in idx:
                q[idx[t]] = idf[idx[t]]
        top = np.argsort(-(M @ q))[:k]
        hits += any(fact in chunks[i] and p in chunks[i].lower() for i in top)
    n = len(facts)
    return intact / n, hits / n, sum(map(len, chunks)) / len(doc)

for name, ch in [("fixed 160", fixed(doc)), ("fixed 160 +40 overlap", fixed(doc, overlap=40)),
                 ("sentence-packed 160", sentence_packed(doc)),
                 ("sentence-packed + context", with_context(sentence_packed(doc)))]:
    intact, rec, cost = evaluate(ch)
    print(f"{name:27s} chunks={len(ch):3d} facts intact={intact:.2f} answerable@3={rec:.2f} index size x{cost:.2f}")
```
Typical output: fixed-size chunks split about a quarter of the facts; overlap
and sentence packing keep them whole but answerable@3 stays low, because the
chunk with the fact does not say which product it is about; the context
prefix fixes that for a modest index-size increase. Swap in your own
documents, real queries, your embedder and recall@k or token IoU against
gold excerpts ([[llm-evaluation]]).

Library equivalents: `langchain-text-splitters` (`RecursiveCharacterTextSplitter`,
`MarkdownHeaderTextSplitter`) and LlamaIndex node parsers; the Chroma report
describes the token-level metrics and links its evaluation code. Late chunking
reference code is in `jina-ai/late-chunking`.

## Choosing / trade-offs
| Strategy | Index-time cost | Helps when | Watch out |
|---|---|---|---|
| Fixed-size (+overlap) | lowest | uniform text, quick baseline | split facts; overlap duplicates text |
| Recursive / structure-aware | low | documents with headings and paragraphs | needs clean parsing upstream |
| Semantic | one embedding per sentence | topic shifts without markup | threshold tuning; tiny or huge chunks |
| Late chunking | one long-context pass per document | chunks that depend on earlier text (pronouns, "this model") | needs token-level outputs from a long-context embedder |
| Contextual retrieval | one LLM call per chunk (cacheable) | templated or cross-referencing documents | LLM cost and re-index time; context can hallucinate |
| Parent-document | extra storage | precise match, broad answer context | larger prompts at query time |

- **Chunk size**: smaller chunks match precisely but carry less context;
  larger ones dilute the embedding and cost more tokens per retrieved chunk.
  Sweep a few sizes with the same eval set.
- **Overlap** protects facts at boundaries but duplicates text in the index
  and in the top-k; sentence-aware splitting often makes it unnecessary.
- **Re-indexing cost**: every chunking change means re-embedding the corpus;
  version the chunker config with the index.

## Gotchas
- The chunker's size unit (characters vs tokens) and the embedder's token
  limit differ; text past the limit is silently truncated by many embedders.
- Parsing errors (headers and footers in the middle of text, broken tables,
  two-column PDFs read across) look like chunking problems; fix parsing first.
- Evaluating chunks by eye on three examples misleads; use a fixed query set
  with gold passages and compare recall@k, then answer quality.
- Overlap inflates recall@k when duplicates of the same passage fill the
  top-k; dedupe before measuring.
- Contextual prefixes generated by an LLM can be wrong; spot-check a sample.
- Changing chunking without re-running retrieval evals hides regressions in
  questions that used to work.

## Related
- [[recursive-text-splitter-from-scratch]] - the recursive splitter and overlap, built by hand.
- [[semantic-chunking-from-scratch]] - embedding-distance breakpoints, built by hand.
- [[rag-basics]] - where chunking sits in the pipeline.
- [[advanced-rag]] - contextual retrieval, parent-document, hybrid search and reranking.
- [[multimodal-rag]] - when the content is tables, charts or page images.
- [[document-parsing]] - clean text before splitting.
- [[embedding-models]] - token limits and long-context embedders.
- [[late-interaction-retrieval]] - token-level vectors instead of pooled chunks.
- [[prompt-caching-and-cost]] - makes per-chunk context generation affordable.
- [[llm-evaluation]] - retrieval metrics and golden sets.
- [[long-context]] - when not to chunk at all.

## References
- Chroma, Evaluating Chunking Strategies for Retrieval: https://research.trychroma.com/evaluating-chunking
- Günther et al., Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models: https://arxiv.org/abs/2409.04701
- Late chunking code: https://github.com/jina-ai/late-chunking
- Anthropic, Introducing Contextual Retrieval: https://www.anthropic.com/news/contextual-retrieval
- LangChain text splitters: https://github.com/langchain-ai/langchain/tree/master/libs/text-splitters
