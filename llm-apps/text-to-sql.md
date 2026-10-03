---
title: Text-to-SQL
category: llm-apps
tags: [text-to-sql, nl2sql, sql, analytics, schema-linking, read-only, bi]
use_cases:
  - "let business users ask questions of the sales database in plain English"
  - "build a safe analytics chatbot over a data warehouse"
  - "generate SQL from natural language with schema context and validation"
  - "evaluate how accurate our NL-to-SQL assistant really is"
  - "add a read-only query tool to an agent"
status: draft
last_verified: 2026-10-03
sources:
  - https://yale-lily.github.io/spider
  - https://bird-bench.github.io/
  - https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
  - https://docs.python.org/3/library/sqlite3.html
---

# Text-to-SQL

## Summary
Text-to-SQL translates a natural-language question into a SQL query, runs it, and returns or explains the result. Frontier LLMs do well on clean schemas but real-world accuracy depends on schema context, business definitions, validation and safe execution. Treat the model as an untrusted query author behind a read-only, limited, audited connection.

## Key concepts
- **Schema context**: give table/column names, types, keys/relations, short descriptions, sample values, and a few example question-SQL pairs. For big schemas, **schema linking** retrieves only the relevant tables/columns (embedding search over schema docs, [[rag-basics]]).
- **Business semantics**: define metrics ("active customer", "revenue") in a semantic layer or glossary; otherwise the model guesses. Views or a semantic layer (dbt, Cube, LookML) reduce ambiguity.
- **Generate -> validate -> execute -> repair**: parse with a SQL parser (sqlglot), allow only `SELECT`, `EXPLAIN` first, run with limits, feed errors back for one or two repair attempts.
- **Agentic variant**: expose `list_tables`, `describe_table`, `run_query` as tools ([[tool-calling]]) so the model explores before querying ([[agents]]).
- **Safety**: read-only DB role, row/column-level permissions per end user, statement timeout, `LIMIT`, no multi-statements, allowlisted schemas. Query results are untrusted text ([[prompt-injection]]).
- **Evaluation**: execution accuracy (does the result match the gold query's result), not string match. Benchmarks: Spider, BIRD; build your own from real questions ([[llm-evaluation]]).
- **MCP**: database MCP servers expose the same tools to many clients ([[model-context-protocol]]).

## When to use / scenarios
- Self-serve analytics for sales, ops, finance teams on a warehouse; internal BI chat.
- Developer productivity: draft queries for review.
- Customer-facing "ask your data" features (needs strict tenant isolation).
- Not suited to: writes/DDL, ambiguous business definitions with no semantic layer, or high-stakes numbers without review - prefer curated dashboards or parameterized queries.

## Setup & code
Read-only guarded generator on SQLite (swap driver for Postgres/Snowflake; use a read-only role there):
```python
# pip install anthropic sqlglot
import sqlite3, sqlglot, anthropic
from sqlglot import exp

client = anthropic.Anthropic()
MODEL = "claude-sonnet-5-5"
conn = sqlite3.connect("file:shop.db?mode=ro", uri=True)       # read-only connection

def schema_text() -> str:
    rows = conn.execute("SELECT sql FROM sqlite_master WHERE type='table'").fetchall()
    return "\n".join(r[0] for r in rows)

def to_sql(question: str, error: str = "") -> str:
    r = client.messages.create(model=MODEL, max_tokens=400,
        system=f"Write ONE SQLite SELECT query answering the question. Output only SQL.\nSchema:\n{schema_text()}",
        messages=[{"role": "user", "content": question + (f"\nPrevious attempt failed: {error}" if error else "")}])
    return r.content[0].text.strip().strip("`").removeprefix("sql").strip()

def validate(sql: str) -> str:
    tree = sqlglot.parse_one(sql, read="sqlite")                 # raises on multiple/invalid statements
    if not isinstance(tree, exp.Select):
        raise ValueError("only SELECT allowed")
    if not tree.args.get("limit"):
        sql = f"SELECT * FROM ({sql}) LIMIT 200"
    return sql

def ask(question: str):
    err = ""
    for _ in range(3):
        try:
            sql = validate(to_sql(question, err))
            return sql, conn.execute(sql).fetchall()
        except Exception as e:
            err = str(e)                                         # repair loop
    raise RuntimeError(f"failed: {err}")
```
Note: `parse_one` of a statement list raises; also grant the DB role only SELECT so validation is a second layer, not the only one.

## Choosing / trade-offs
- **One-shot vs agentic exploration**: one-shot is faster/cheaper on small schemas; agentic handles large, messy schemas and self-correction at higher latency and cost.
- **Schema in prompt vs retrieved**: full schema is simplest up to a few dozen tables (cache it, [[prompt-caching-and-cost]]); retrieval needed beyond that.
- **Semantic layer vs raw tables**: a semantic layer boosts accuracy and consistency but needs upkeep.
- **Explain vs answer**: show the SQL and assumptions to users who can verify it; hide it only when results are low-stakes.
- **Fine-tuned small model vs prompted frontier model**: fine-tuning ([[fine-tuning-and-peft]]) helps for fixed schemas and cost/latency; prompting is stronger on novel schemas.

## Gotchas
- Plausible but wrong SQL (bad joins, fan-out duplicates, wrong date boundaries, NULL handling) returns confident wrong numbers; show the query and test on a golden question set.
- Ambiguous questions: let the model ask a clarifying question instead of guessing.
- Dialect mismatches (date functions, quoting); state the dialect explicitly.
- Prompt-injected text in data (comments, names) can steer later queries; keep the DB role read-only and scoped.
- Cost blowups from unbounded scans: statement timeouts, partition filters, `EXPLAIN` cost checks.
- Sensitive columns: mask or exclude them from the schema the model sees, enforce at the DB layer, and do not trust prompt rules ([[guardrails-and-safety]]).
- Execution accuracy on benchmarks overstates real-world performance on your schema.

## Related
- [[tool-calling]] - query tool design.
- [[agents]] - exploratory SQL agents.
- [[structured-output]] - structured query plans.
- [[llm-evaluation]] - execution-accuracy tests.
- [[data-analytics]] - scenario guidance.
- [[model-context-protocol]] - database servers.

## References
- Spider benchmark: https://yale-lily.github.io/spider
- BIRD benchmark: https://bird-bench.github.io/
- sqlglot: https://github.com/tobymao/sqlglot
