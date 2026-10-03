---
title: Data analytics and BI with AI
category: scenarios
tags: [analytics, bi, text-to-sql, dashboards, forecasting, data-quality]
use_cases:
  - "let business users ask questions of the database in natural language"
  - "automate report and insight narration"
  - "clean and classify messy data with an LLM"
  - "forecast metrics and detect anomalies in KPIs"
  - "build an analyst agent that runs code on data"
status: draft
last_verified: 2026-10-03
sources:
  - https://pandas.pydata.org/docs/
  - https://duckdb.org/docs/
  - https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview
---

# Data analytics and BI with AI

## Summary
AI reduces the distance between a question and an answer: natural-language
to SQL, auto-narrated dashboards, LLM-assisted data cleaning, and forecasting
or anomaly detection on metrics. The central rule: the model plans and
explains, while code and the database compute. Numbers must come from
queries, not from the model.

## Key concepts
- Semantic layer: metric definitions ("active customer", "net revenue") live
  in one governed place; text-to-SQL accuracy depends on it more than on the
  model ([[text-to-sql]]).
- Least privilege: read-only database role, row-level security, query
  timeouts and cost limits.
- Verification: show the SQL and the result, and let users inspect it; run
  generated code in a sandbox.
- Statistics still apply: LLMs will state causal stories for correlations;
  require uncertainty and sample sizes.
- Data governance: PII columns masked before the model sees them
  ([[ai-security-privacy-compliance]]).

## When to use / scenarios
1. **Natural-language BI ("ask your data").** Problem: analyst backlog.
   Approach: schema + metric docs in context, retrieval of relevant tables,
   SQL generation, validation, execution, explanation. Read: [[text-to-sql]],
   [[rag-basics]], [[structured-output]], [[tool-calling]].
2. **Analyst agent that writes and runs code.** Approach: sandboxed Python
   with pandas/DuckDB, iterate on errors, return charts and the code. Read:
   [[agents]], [[coding-agents]], [[agent-frameworks]].
3. **Automated report narration.** Approach: compute stats in code, LLM
   writes the commentary from those numbers only. Read:
   [[hallucination-and-grounding]], [[prompt-engineering]].
4. **Data cleaning and normalisation.** Approach: LLM for fuzzy cases
   (company names, addresses, categories), deterministic rules first, batch
   with sampling checks. Read: [[structured-output]],
   [[cost-and-latency]], [[small-language-models]].
5. **Free-text analysis (surveys, tickets, reviews).** Approach: embed and
   cluster, LLM labels clusters, quantify. Read: [[embeddings]],
   [[nlp-classic-tasks]].
6. **KPI forecasting.** Approach: baselines (seasonal naive) then GBM or
   specialised models; intervals. Read: [[time-series-forecasting]],
   [[gradient-boosting-tabular]].
7. **Metric anomaly alerts and root-cause hints.** Approach: detect on
   segments, LLM proposes hypotheses from related dimensions. Read:
   [[anomaly-detection]].
8. **Predictive models (churn, propensity).** Read:
   [[gradient-boosting-tabular]], [[classic-ml-scikit-learn]],
   [[experiment-tracking]].
9. **Document-to-table extraction for analysis.** Read:
   [[document-processing]], [[document-parsing]].
10. **Training-data labelling for analytics models.** Read:
    [[data-labeling-and-synthetic-data]].
11. **Domain dashboards.** Read: [[finance]], [[ecommerce-retail]],
    [[manufacturing-iot]].

## Setup & code
Reference architecture (flagship: NL-to-SQL assistant):

```
question -> retrieve relevant tables/metrics/example queries (vector index)
 -> LLM writes SQL (dialect + schema + rules in prompt)
 -> validate: parse, allow only SELECT, add LIMIT, EXPLAIN cost check
 -> run on read-only replica (timeout) -> result rows
 -> LLM explains from result rows; show SQL + table + chart
 -> thumbs feedback -> verified Q->SQL pairs become few-shot/eval set
```

```python
import duckdb, sqlglot
def safe_run(sql, con):
    tree = sqlglot.parse_one(sql)
    assert tree.key == "select", "read-only queries only"
    return con.execute(f"SELECT * FROM ({sql}) LIMIT 1000").df()
con = duckdb.connect("warehouse.duckdb", read_only=True)
```

## Choosing / trade-offs
- Text-to-SQL vs curated dashboards: free-form flexibility vs guaranteed
  definitions; many teams expose NL on top of a semantic layer only.
- Big model vs small: complex joins need a strong model; simple lookups run
  on cheaper ones ([[model-selection]]).
- LLM cleaning vs rules: rules are free and deterministic, so use LLMs for
  the residue.
- Local analysis vs warehouse: DuckDB handles surprisingly large files on a
  laptop.

## Gotchas
- Plausible but wrong SQL (wrong join grain, double counting); evaluate on
  a gold set of questions with known answers ([[llm-evaluation]]).
- Ambiguous metric names produce different, equally confident answers.
- Prompt injection via data values (a text cell saying "ignore previous
  instructions") ([[prompt-injection]]).
- Sensitive columns leaking into prompts or logs; mask first.
- Executing generated code without a sandbox.
- Narratives that overstate small effects; include sample sizes.

## Related
- [[llm-observability]] - logging queries, cost and failures.
- [[vector-databases]] - indexing schema and example queries.
- [[personal-assistants]] - personal-scale analytics (budgets, health logs).

## References
- pandas docs: https://pandas.pydata.org/docs/
- DuckDB docs: https://duckdb.org/docs/
- Anthropic tool use overview: https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview
