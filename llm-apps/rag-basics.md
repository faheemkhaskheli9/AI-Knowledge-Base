---
title: Retrieval-augmented generation (RAG) basics
category: llm-apps
tags: [rag, retrieval, chunking, embeddings, citations, grounding]
use_cases:
  - "build a chatbot that answers questions from our company wiki and PDFs"
  - "internal policy assistant for HR or compliance with cited answers"
  - "customer-support bot grounded in product manuals and past tickets"
  - "private document Q&A for lawyers or clinicians where data cannot leave the network"
  - "search over a research-paper collection with source links"
status: draft
last_verified: 2026-10-03
sources:
  - https://arxiv.org/abs/2005.11401
  - https://platform.claude.com/docs/en/build-with-claude/embeddings
  - https://platform.openai.com/docs/guides/embeddings
---

# Retrieval-augmented generation (RAG) basics

## Summary
RAG retrieves relevant passages from your own corpus at question time and puts them in the prompt, so the model answers from current, private knowledge instead of its weights. It is the default way to add knowledge to an LLM: cheaper and fresher than fine-tuning, and it supports citations. Quality is mostly decided by ingestion and retrieval, not the generator.

## Key concepts
- **Pipeline**: load/parse -> clean -> chunk -> embed -> index (vector DB) -> at query: embed query -> top-k search -> (rerank) -> build prompt with numbered sources -> generate -> cite.
- **Chunking**: 200-800 tokens with 10-20% overlap is a common start; split on structure (headings, paragraphs, table rows) before size. Keep metadata (source, page, section, date, ACL).
- **Embeddings**: same model for documents and queries; cosine similarity. See [[embedding-models]], [[embeddings]].
- **Idempotent ingestion**: chunk id = hash(source, position, content) so re-ingesting never duplicates and edits replace stale text.
- **Citation-or-fallback**: attach sources from retrieval metadata (not parsed from model text); on an empty/low-score retrieval, return a fixed "I don't know" without calling the LLM.
- **Context budget**: top-k of 3-8 chunks is usual; more context is not better ("lost in the middle"). Measure whether extra chunks help.
- **Retrieval and generation are evaluated separately** ([[llm-evaluation]]): recall@k / MRR for retrieval, faithfulness and answer correctness for generation.

## When to use / scenarios
- Knowledge changes often, is private, or too large for the context window; answers must be traceable.
- Enterprise search/FAQ, support deflection, onboarding assistants, legal/clinical reference lookup.
- Not RAG: whole corpus fits in context and is cached ([[long-context]], [[prompt-caching-and-cost]]); structured data questions ([[text-to-sql]]); style or behavior change ([[fine-tuning-and-peft]]).

## Setup & code
```python
# pip install anthropic chromadb sentence-transformers
import hashlib, anthropic, chromadb
from sentence_transformers import SentenceTransformer

emb = SentenceTransformer("all-MiniLM-L6-v2")
db = chromadb.PersistentClient(path="./rag_db").get_or_create_collection("docs")
llm = anthropic.Anthropic()

def chunks(text, size=800, overlap=120):
    i = 0
    while i < len(text):
        yield text[i:i + size]
        i += size - overlap                       # guarantees forward progress

def ingest(source: str, text: str):
    for n, c in enumerate(chunks(text)):
        cid = hashlib.sha256(f"{source}|{n}|{c}".encode()).hexdigest()[:32]
        db.upsert(ids=[cid], documents=[c], embeddings=[emb.encode(c).tolist()],
                  metadatas=[{"source": source, "n": n}])

def answer(q: str, k=4, max_dist=0.8):
    r = db.query(query_embeddings=[emb.encode(q).tolist()], n_results=k)
    docs, metas, dists = r["documents"][0], r["metadatas"][0], r["distances"][0]
    hits = [(d, m) for d, m, x in zip(docs, metas, dists) if x <= max_dist]
    if not hits:                                  # short-circuit: never ask the LLM on a miss
        return "I could not find this in the documents.", []
    ctx = "\n\n".join(f"[{i+1}] (source: {m['source']}) {d}" for i, (d, m) in enumerate(hits))
    msg = llm.messages.create(
        model="claude-sonnet-5-5", max_tokens=600,
        system="Answer only from the numbered context. Cite like [1]. If not answerable, say so.",
        messages=[{"role": "user", "content": f"<context>\n{ctx}\n</context>\nQuestion: {q}"}])
    return msg.content[0].text, [m["source"] for _, m in hits]
```
(Chroma's default distance and the `max_dist` threshold depend on the collection's metric; calibrate on real queries.)

## Choosing / trade-offs
- **Chunk size**: small = precise but loses context; large = more context but diluted embeddings and cost. Tune on your eval set.
- **Hosted vs local embeddings**: hosted = quality/zero ops; local = privacy, no per-call cost ([[ollama]]).
- **Vector DB**: see [[vector-databases]]; start with pgvector if you already run Postgres, Chroma/FAISS for prototypes.
- **Naive vs advanced RAG**: add hybrid search, reranking and query rewriting only after measuring failures ([[advanced-rag]]).
- **Agentic RAG** (model decides when/what to search) handles multi-hop questions at higher latency/cost ([[agents]]).

## Gotchas
- Garbage parsing in, garbage answers out: tables, scans and multi-column PDFs need real parsing ([[document-parsing]]).
- Retrieved text is untrusted and can carry injected instructions ([[prompt-injection]]).
- Access control must be enforced at retrieval time (filter by user/tenant), not in the prompt.
- Changing the embedding model requires re-embedding everything; store the model name in metadata.
- Stale index: schedule re-ingestion and delete removed documents.
- Without a "no answer" path the model will make something up ([[hallucination-and-grounding]]).
- Evaluate on real user questions, including unanswerable ones.

## Related
- [[advanced-rag]] - hybrid search, reranking, GraphRAG, contextual retrieval.
- [[vector-databases]] - where to store vectors.
- [[embedding-models]] - choosing an embedder.
- [[document-parsing]] - upstream text quality.
- [[llm-evaluation]] - measuring retrieval and faithfulness.
- [[agent-memory]] - RAG as long-term memory.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/system-design/search-systems.md - keyword search and ranking infrastructure.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/data/batch-and-stream-processing.md - keeping the index in sync with source data.

## References
- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks": https://arxiv.org/abs/2005.11401
- Anthropic embeddings guide: https://platform.claude.com/docs/en/build-with-claude/embeddings
- OpenAI embeddings guide: https://platform.openai.com/docs/guides/embeddings
