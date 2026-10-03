---
title: Advanced RAG (hybrid search, reranking, query rewriting, GraphRAG, contextual retrieval)
category: llm-apps
tags: [rag, hybrid-search, bm25, reranking, query-rewriting, graphrag, contextual-retrieval]
use_cases:
  - "improve a RAG chatbot that misses exact product codes and error messages"
  - "answer multi-hop questions across many contracts or reports"
  - "raise retrieval recall on a large technical documentation set"
  - "reduce wrong-chunk answers in a healthcare or legal knowledge base"
  - "summarize themes across a whole corpus, not just single passages"
status: draft
last_verified: 2026-10-03
sources:
  - https://www.anthropic.com/news/contextual-retrieval
  - https://arxiv.org/abs/2404.16130
  - https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking
  - https://qdrant.tech/documentation/concepts/hybrid-queries/
---

# Advanced RAG

## Summary
Techniques layered on basic RAG ([[rag-basics]]) when measured retrieval recall or answer faithfulness is too low: lexical+vector hybrid search, rerankers, query rewriting, contextualized chunks, and graph-based retrieval. Add them one at a time, driven by an eval set, because each adds latency and moving parts.

## Key concepts
- **Hybrid search**: run BM25/keyword and dense vector search, merge with Reciprocal Rank Fusion (RRF: `score = sum 1/(k + rank)`, k~60). Keywords catch exact IDs, names, error codes; vectors catch paraphrase.
- **Reranking**: retrieve wide (top 50-100), rescore with a cross-encoder or reranker API, keep top 5-10. Usually the best quality gain per effort ([[rerankers]]).
- **Query rewriting**: rewrite conversational follow-ups into standalone queries; **multi-query** (several paraphrases); **HyDE** (embed a hypothetical answer); **decomposition** for multi-part questions; step-back queries.
- **Contextual retrieval** (Anthropic): before embedding, have an LLM prepend a short chunk-specific context ("This chunk is from the 2025 ACME 10-K, Revenue section...") to each chunk, for both embeddings and BM25. Anthropic reported large drops in failed retrievals, further improved by reranking; prompt caching makes the per-chunk LLM calls cheap.
- **Parent-document / small-to-big**: match on small chunks, return the surrounding section.
- **GraphRAG**: LLM extracts entities/relations into a knowledge graph, builds community summaries; good for "what are the main themes" and multi-hop relational questions across a corpus (Microsoft GraphRAG, arXiv 2404.16130).
- **Metadata filtering and routing**: filter by date/tenant/doc type; route queries to different indexes.
- **Agentic RAG**: the model iteratively searches, reads, and re-queries ([[agents]]).

## When to use / scenarios
- Support docs full of SKUs/error codes: hybrid search.
- Long legal/financial filings where chunks lack context: contextual retrieval + rerank.
- Chat assistant with follow-ups: query rewriting using history.
- Investigative/analyst questions over many linked entities (fraud rings, supply chains): GraphRAG.
- Not worth it when basic RAG already hits your recall target or the corpus is tiny.

## Setup & code
Hybrid (BM25 + vector) with RRF, then rerank; minimal and library-light:
```python
# pip install rank-bm25 sentence-transformers numpy
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

docs = ["Error E4021: disk quota exceeded", "How to raise storage limits", "Billing FAQ"]
bm25 = BM25Okapi([d.lower().split() for d in docs])
emb = SentenceTransformer("all-MiniLM-L6-v2")
D = emb.encode(docs, normalize_embeddings=True)
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rrf(rankings, k=60):
    s = {}
    for r in rankings:
        for pos, i in enumerate(r):
            s[i] = s.get(i, 0) + 1 / (k + pos + 1)
    return sorted(s, key=s.get, reverse=True)

def search(q, wide=20, top=3):
    lex = np.argsort(-bm25.get_scores(q.lower().split()))[:wide]
    vec = np.argsort(-(D @ emb.encode(q, normalize_embeddings=True)))[:wide]
    cand = rrf([lex.tolist(), vec.tolist()])[:wide]
    scores = ce.predict([(q, docs[i]) for i in cand])
    return [docs[cand[i]] for i in np.argsort(-scores)[:top]]

print(search("E4021 quota"))
```
Contextual retrieval step (Anthropic SDK; cache the whole document with `cache_control` so each chunk call is cheap):
```python
import anthropic
c = anthropic.Anthropic()
def contextualize(doc: str, chunk: str) -> str:
    r = c.messages.create(model="claude-haiku-4-5-20251001", max_tokens=120, messages=[{
        "role": "user", "content": [
            {"type": "text", "text": f"<document>\n{doc}\n</document>", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": f"<chunk>\n{chunk}\n</chunk>\nWrite 1-2 sentences situating this chunk in the document to improve search retrieval. Output only the context."}]}])
    return r.content[0].text + "\n" + chunk     # embed and BM25-index this string
```
Many vector DBs do hybrid natively (Qdrant sparse+dense with fusion, Weaviate, Elasticsearch/OpenSearch, Postgres FTS + pgvector) - see [[vector-databases]].

## Choosing / trade-offs
- **Reranker latency** (tens-hundreds ms) vs quality: always worth it for top-k <= 100; use a hosted rerank API or a small local cross-encoder.
- **Contextual retrieval** adds one-time ingest cost (LLM call per chunk) but no query-time cost.
- **GraphRAG** has heavy indexing cost and complexity; use it for global/relational questions, not plain lookup. Lighter alternative: entity metadata + filters.
- **Rewriting** adds an LLM hop per query; skip for single-turn search boxes.
- **Fine-tuned embedder** helps domain jargon but needs labeled pairs ([[embedding-models]]).

## Gotchas
- Fusing raw scores from BM25 and cosine is meaningless; fuse ranks (RRF) or normalize.
- Rewriting can drift from user intent; keep the original query as one of the retrieval queries.
- A reranker's max input length truncates long chunks silently.
- Evaluate each addition against the same eval set; stack only what helps ([[llm-evaluation]]).
- Re-index when you change the contextualization prompt; store prompt version.

## Related
- [[rag-basics]] - baseline pipeline.
- [[vector-databases]] - native hybrid support.
- [[rerankers]] - model choices.
- [[prompt-caching-and-cost]] - makes contextual retrieval affordable.
- [[agents]] - agentic/iterative retrieval.

## References
- Anthropic, Introducing Contextual Retrieval: https://www.anthropic.com/news/contextual-retrieval
- Edge et al., From Local to Global (GraphRAG): https://arxiv.org/abs/2404.16130
- Qdrant hybrid queries: https://qdrant.tech/documentation/concepts/hybrid-queries/
