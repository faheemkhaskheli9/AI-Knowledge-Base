---
title: GraphRAG and knowledge graphs for retrieval
category: llm-apps
tags: [graphrag, knowledge-graph, entity-extraction, community-detection, leiden, lightrag, neo4j, multi-hop]
use_cases:
  - "answer 'what are the main themes across these 10,000 reports' questions that plain RAG cannot"
  - "answer multi-hop questions over linked entities (companies, people, contracts, suppliers)"
  - "build a knowledge graph from unstructured documents with an LLM"
  - "decide whether GraphRAG is worth its indexing cost over vector RAG"
status: draft
last_verified: 2026-10-15
sources:
  - https://arxiv.org/abs/2404.16130
  - https://github.com/microsoft/graphrag
  - https://pypi.org/project/graphrag/
  - https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/
  - https://arxiv.org/abs/2410.05779
---

# GraphRAG and knowledge graphs for retrieval

## Summary
GraphRAG uses an LLM to extract entities and relations from documents into a knowledge graph, groups the graph into communities, and pre-writes a summary of each community. At query time it answers either **globally** (map-reduce over community summaries, for "what are the main themes / risks across the corpus") or **locally** (start from entities in the question and walk their neighbourhood, for multi-hop questions about specific things). It beats plain vector RAG on corpus-wide sensemaking and relational questions, at the cost of an expensive, LLM-heavy indexing step. For simple fact lookup, vector or hybrid RAG is cheaper and usually as good ([[rag-basics]], [[advanced-rag]]).

## Key concepts
- **Knowledge graph (KG)**: nodes = entities (person, org, product, place, concept) with descriptions; edges = typed relations with descriptions and source chunk IDs (provenance).
- **Extraction**: an LLM reads each chunk and emits entities, relations and optionally claims. Repeated "gleaning" passes pick up missed items. Then **entity resolution** merges duplicates ("IBM", "International Business Machines").
- **Communities**: hierarchical clustering of the graph (Microsoft GraphRAG uses Leiden). Each community gets an LLM-written report: key entities, relations, findings. Higher levels summarize lower ones.
- **Global search**: send the question to many community reports in parallel (map), collect partial answers with relevance scores, then combine (reduce). Answers "whole corpus" questions vector top-k cannot, because no single chunk holds the answer.
- **Local search**: link question entities to graph nodes, gather neighbours, relations, community reports and the source chunks, then answer. Good for "how is X connected to Y".
- **Lighter variants**: LightRAG (graph + vector dual-level retrieval with incremental updates), Microsoft's LazyGraphRAG and "basic"/DRIFT search modes, or a hand-modelled graph in a graph database queried with Cypher (text-to-Cypher, similar to [[text-to-sql]]).
- **Two ways to get a graph**: *LLM-extracted* (fast to start, noisy) vs *curated/structured* (from your CRM, ERP, ontology; accurate, needs data engineering). Many production systems combine a curated core with LLM-extracted edges.

## When to use / scenarios
- Analyst and investigative work over many documents: "summarize the main allegations across all case files", "what risks recur across supplier audits".
- Relational, multi-hop questions: fraud rings, beneficial ownership, supply-chain exposure ("which of our products depend on suppliers in region R"), drug-gene-disease links, contract obligations across parties.
- Domains that already have entities and relations (CRM, ERP, ontologies, citations) where the graph is cheap to build from structured data.
- Not for: FAQ/support lookup, single-document QA, fast-changing corpora where re-indexing cost hurts, or small corpora that fit in the context window ([[long-context]]).

## Setup & code
Microsoft GraphRAG (MIT; 3.3.0 on PyPI at verification, Python >= 3.11; the repo says it is largely in maintenance mode):
```
python -m pip install graphrag
graphrag init                    # writes settings.yaml and prompts; set your model and API key
graphrag index                   # LLM extraction + communities + reports (the expensive step)
graphrag query "What are the top themes in this corpus?"            # global (default)
graphrag query "Who is X and what are their main relationships?" --method local
```
Run `graphrag init` again between minor versions (it overwrites config and prompts, so back them up) and tune the extraction prompts for your domain before trusting results.

The mechanics, without an LLM: extract triples, build the graph, detect communities, summarize them, then answer a multi-hop question locally and a theme question globally. Extraction and summaries are stubs here; in production they are LLM calls with [[structured-output]].
```python
import re
from collections import Counter, defaultdict

chunks = [
    "Acme Corp buys lithium from Borealis Mining.", "Borealis Mining operates in Region North.",
    "Region North faces export restrictions.", "Acme Corp sells batteries to VoltCar.",
    "Zenith Labs partners with Helix Bio.", "Helix Bio runs trials with Zenith Labs in Region South.",
]
REL = re.compile(r"^([A-Z]\w+(?: [A-Z]\w+)*) (buys \w+ from|operates in|faces|sells \w+ to|partners with|runs trials with) ([\w ]+?)\.$")

# 1) extraction (LLM in practice): (head, relation, tail, source chunk)
triples = [(m[1], m[2], m[3].split(" in ")[0], i) for i, c in enumerate(chunks) if (m := REL.match(c))]
graph = defaultdict(set)
for h, r, t, i in triples:
    graph[h].add((r, t, i)); graph[t].add((f"<-{r}", h, i))

# 2) communities by label propagation (GraphRAG uses hierarchical Leiden)
label = {n: n for n in graph}
for _ in range(10):
    for n in sorted(graph):
        label[n] = Counter(label[nb] for _, nb, _ in graph[n]).most_common(1)[0][0]
communities = defaultdict(set)
for n, l in label.items(): communities[l].add(n)

# 3) community reports (LLM summary in practice)
reports = {c: "Entities: " + ", ".join(sorted(ns)) + ". Facts: " +
              "; ".join(chunks[i] for h, _, t, i in triples if h in ns) for c, ns in communities.items()}

# 4) local search: multi-hop walk from the question's entities, keep provenance
def local(entity, hops=3):
    seen, frontier, evidence = {entity}, [entity], set()
    for _ in range(hops):
        nxt = []
        for n in frontier:
            for r, nb, i in graph[n]:
                evidence.add(i)
                if nb not in seen: seen.add(nb); nxt.append(nb)
        frontier = nxt
    return [chunks[i] for i in sorted(evidence)]

print("Q: Is Acme Corp exposed to export restrictions?")
for line in local("Acme Corp"): print("  evidence:", line)

# 5) global search: map over community reports, reduce to themes
print("Q: What are the main themes?")
for c, rep in reports.items(): print(f"  community {c!r}: {rep[:90]}...")
print("communities:", len(communities))
```
No single chunk says "Acme is exposed to export restrictions"; the three-hop path Acme -> Borealis -> Region North -> restrictions does, which is the case where vector top-k on the question tends to miss the middle hops.

## Choosing / trade-offs
| Approach | Indexing cost | Best at | Weak at |
|---|---|---|---|
| Vector / hybrid RAG | Embeddings only | Lookup, "find the passage" | Corpus-wide themes, multi-hop joins |
| GraphRAG global | LLM pass over every chunk + community reports | Sensemaking across the whole corpus | Precise facts; cost to keep fresh |
| GraphRAG local / LightRAG | LLM extraction | Entity-centric multi-hop questions | Extraction errors propagate |
| Curated KG + text-to-Cypher | Data engineering, schema design | Exact relational answers, audit | Unstructured nuance; schema upkeep |
| Long context, no index | Tokens per query | Small corpora | Large or growing corpora |

- **Cost**: indexing makes LLM calls proportional to corpus size (extraction, gleaning, summaries). Pilot on a sample, measure cost per document, and use a cheaper model for extraction if quality holds ([[cost-and-latency]]).
- **Freshness**: full re-indexing is expensive; prefer approaches with incremental updates for changing corpora.
- **Where the graph lives**: Parquet/files (GraphRAG default) for batch analytics; a graph database when you need online queries, access control and joins with business data.
- **Graph RAG vs GNNs**: GraphRAG retrieves text through a graph; [[graph-neural-networks]] learn on graph structure (link prediction, node classification). Different problems.

## Gotchas
- Extraction quality is the ceiling: duplicate entities, hallucinated relations, and missing edges all flow into answers. Sample and review extracted triples; keep source chunk IDs for every edge ([[hallucination-and-grounding]]).
- Default prompts are generic; domain entity types (drug, clause, part number) need prompt tuning, which can be optimized with a labelled sample ([[dspy-and-prompt-optimization]]).
- Global search fans out to many reports per question: query cost and latency are much higher than vector RAG.
- Evaluate against a vector/hybrid baseline on your own questions before committing; many "GraphRAG wins" are on sensemaking questions, not lookups ([[llm-evaluation]]).
- Access control: community reports mix facts from many documents, which can leak content across permission boundaries; build per-tenant graphs or filter at extraction time.
- Injected instructions in source documents can become "facts" in the graph and summaries ([[prompt-injection]]).

## Related
- [[advanced-rag]] - hybrid search, reranking and contextual retrieval to try first.
- [[rag-basics]] - the baseline pipeline.
- [[vector-databases]] - many store both vectors and graph-like metadata.
- [[text-to-sql]] - same pattern as text-to-Cypher over a curated graph.
- [[structured-output]] - reliable entity/relation extraction.
- [[graph-neural-networks]] - learning on graphs, a different use of graph structure.
- [[context-engineering]] - community reports are a pre-computed form of compaction.
- [[dspy-and-prompt-optimization]] - tune extraction prompts against a labelled sample.
- [[multi-agent-systems]] - map-reduce global search resembles orchestrator-worker fan-out.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/data/graph-databases.md - graph database modelling and query languages.

## References
- Edge et al., From Local to Global: A Graph RAG Approach to Query-Focused Summarization: https://arxiv.org/abs/2404.16130
- Microsoft GraphRAG repository: https://github.com/microsoft/graphrag
- GraphRAG on PyPI: https://pypi.org/project/graphrag/
- Microsoft Research blog, GraphRAG: https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/
- Guo et al., LightRAG: Simple and Fast Retrieval-Augmented Generation: https://arxiv.org/abs/2410.05779
