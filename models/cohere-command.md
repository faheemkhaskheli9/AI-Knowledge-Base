---
title: Cohere Command, Embed and Rerank
category: models
tags: [cohere, command-a, embed, rerank, rag, citations, enterprise, private-deployment]
use_cases:
  - "build an enterprise RAG assistant that answers with citations to source documents"
  - "add multilingual semantic search plus reranking to an internal document portal"
  - "deploy an LLM inside our own VPC or cloud account for a regulated bank"
  - "use one vendor for the generator, embeddings and reranker in a RAG stack"
status: draft
last_verified: 2026-10-15
sources:
  - https://docs.cohere.com/docs/models
  - https://docs.cohere.com/docs/rag-complete-example
  - https://docs.cohere.com/docs/cohere-embed
  - https://docs.cohere.com/docs/rerank
  - https://docs.cohere.com/v2/docs/reranking-best-practices
  - https://docs.cohere.com/docs/cohere-works-everywhere
  - https://cohere.com/models-overview
  - https://huggingface.co/CohereLabs/c4ai-command-a-03-2025
  - https://cohere.com/research/papers/command-a-technical-report.pdf
---

# Cohere Command, Embed and Rerank

## Summary
Cohere (Canada) sells an enterprise-focused stack: Command generative models,
Embed embedding models and Rerank cross-encoders, all from one API and SDK.
Its chat endpoint takes a `documents` list and returns citation spans pointing
back to those documents, which makes it a natural fit for grounded RAG.
Models are also available on major clouds and for private deployment.

## Key concepts
- Three model types, one SDK (`cohere.ClientV2`): `chat` (Command), `embed`,
  `rerank`. Older code uses `cohere.Client` (v1 API); new code should use v2.
- Grounded chat: pass `documents=[...]` (strings or objects with metadata);
  the reply's `message.citations` lists spans (`start`, `end`, `text`) and
  the source documents for each.
- Embeddings are asymmetric: embed the corpus with
  `input_type="search_document"` and queries with `input_type="search_query"`.
- Rerank scores query and document together; long documents are chunked
  automatically and the context limit covers query + document ([[rerankers]]).
- Weights: Command A (111B) and its Reasoning / Translate / Vision siblings are
  published by Cohere Labs under CC-BY-NC (non-commercial) plus an acceptable
  use policy. Cohere's models overview lists Command A+ as open under Apache 2.0;
  confirm on its model card before relying on that.

Current IDs as of 2026-10-15 (source: docs.cohere.com models page, rerank
best-practices page and the RAG guide; read via search snippets because the
docs host was unreachable from my environment):

| Model | API ID | Context / limits | Notes |
|---|---|---|---|
| Command A+ | `command-a-plus-05-2026` | 128K in, 64K max out | text + image in; MoE; reasoning, agents, translation |
| Command A | `command-a-03-2025` | 256K in, 8K max out | text only |
| Command R7B | `command-r7b-12-2024` | 128K in, 4K max out | small, cheap |
| Embed 5 Pro | `embed-v5.0-pro` | 128K | multimodal, 100+ languages (blog + RAG guide) |
| Embed 4 | `embed-v4.0` | 128K | text + images; 256/512/1024/1536 dims (default 1536) |
| Embed English 3 | `embed-english-v3.0` | 512 tokens | 1024 dims |
| Rerank 4 Pro / Fast | `rerank-v4.0-pro`, `rerank-v4.0-fast` | 32,768 tokens | multilingual |
| Rerank 3.5 | `rerank-v3.5` | 4,096 tokens | |

`command-r-plus-04-2024` was deprecated on 2025-09-15. An Embed 5 Fast tier
exists per Cohere's blog, but I could not confirm its API ID. Prices were not
in the sources I could read; see cohere.com/pricing.

## When to use / scenarios
- Enterprise search and knowledge assistants that must show which document
  each claim came from (legal, policy, internal wiki) ([[rag-basics]]).
- Regulated industries (banking, public sector) that want the model inside
  their own cloud account or on-prem; Cohere offers cloud-marketplace and
  private deployments ([[managed-cloud-ai-platforms]]).
- Multilingual retrieval: Embed and Rerank multilingual models
  ([[embedding-models]]).
- Adding a reranker to an existing RAG stack, even if the generator is another
  vendor's model.
- Not for: the strongest agentic coding or frontier reasoning; compare with
  [[anthropic-claude]] / [[openai-gpt]]. Not for commercial self-hosting of
  Command A weights (CC-BY-NC) without a licence from Cohere.

## Setup & code
```bash
pip install cohere
export CO_API_KEY=...
```
<!-- skip-check: needs a Cohere API key and network -->
```python
import cohere

co = cohere.ClientV2()  # reads CO_API_KEY
docs = [
    "Refunds are issued within 14 days of a return being received.",
    "Shipping to the EU takes 3-5 working days.",
    "Gift cards cannot be refunded.",
]
query = "How long do refunds take?"

ranked = co.rerank(model="rerank-v4.0-pro", query=query, documents=docs, top_n=2)
top = [docs[r.index] for r in ranked.results]

resp = co.chat(
    model="command-a-03-2025",
    messages=[{"role": "user", "content": query}],
    documents=top,
)
print(resp.message.content[0].text)
for c in resp.message.citations or []:
    print(c.start, c.end, c.text)
```

Rendering citation spans as footnotes (runs offline; the dicts mimic the
`start` / `end` / `sources` fields of a v2 response):
```python
answer = "Refunds are issued within 14 days. Gift cards cannot be refunded."
citations = [
    {"start": 0, "end": 33, "sources": ["doc_0"]},
    {"start": 35, "end": 64, "sources": ["doc_2"]},
]

def with_footnotes(text, cites):
    out, last = [], 0
    for c in sorted(cites, key=lambda c: c["start"]):
        out.append(text[last:c["end"]])
        out.append("[" + ",".join(s.split("_")[1] for s in c["sources"]) + "]")
        last = c["end"]
    out.append(text[last:])
    return "".join(out)

print(with_footnotes(answer, citations))
```

## Choosing / trade-offs
- Command A+ for multimodal input and agentic work; Command A when you need the
  256K input window on text; R7B for cheap, high-volume tasks.
- Embed 4/5 vs open embedders: Cohere gives multimodal, long-context embeddings
  with no ops; open models avoid per-call cost and data egress
  ([[embedding-models]]).
- Rerank Pro vs Fast: quality vs latency/throughput; measure the gain on your
  queries ([[llm-evaluation]]).
- Cohere API vs cloud marketplace vs private deployment: data residency and
  procurement usually decide, not model quality.

## Gotchas
- Mixing `input_type` (embedding queries as documents) quietly lowers recall.
- Changing embedding model or dimension means re-embedding the whole corpus.
- Rerank v3.x has a 4,096-token limit; long chunks are split and you pay for
  each piece.
- Old tutorials use `cohere.Client` and v1 parameters (`connectors`,
  `preamble`); v2 uses `messages` and `documents`.
- On Azure and other clouds each model can have its own endpoint and key;
  create one client per model.
- CC-BY-NC weights on Hugging Face are for research; production use needs the
  API or a commercial agreement.

## Related
- [[rerankers]] - Rerank in context with open alternatives.
- [[embedding-models]] - how Cohere Embed compares.
- [[rag-basics]] - the pipeline these three models fill.
- [[advanced-rag]] - hybrid search, query rewriting, reranking.
- [[hallucination-and-grounding]] - why citations matter.
- [[model-licenses]] - CC-BY-NC vs Apache 2.0 weights.
- [[model-selection]] - cross-provider guide.

## References
- https://docs.cohere.com/docs/models
- https://docs.cohere.com/docs/rag-complete-example
- https://docs.cohere.com/docs/cohere-embed
- https://docs.cohere.com/docs/rerank
- https://docs.cohere.com/v2/docs/reranking-best-practices
- https://docs.cohere.com/docs/cohere-works-everywhere
- https://cohere.com/blog/embed-5
- https://cohere.com/models-overview
- https://huggingface.co/CohereLabs/c4ai-command-a-03-2025
- Cohere, "Command A: An Enterprise-Ready Large Language Model" (2025), https://cohere.com/research/papers/command-a-technical-report.pdf
