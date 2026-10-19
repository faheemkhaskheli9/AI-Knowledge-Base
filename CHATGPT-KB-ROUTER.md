# ChatGPT knowledge-base router

Use this file as the **public entry point** for choosing among the owner's topic-specific GitHub reference repositories. It is a retrieval map, **not** an automatic ChatGPT integration or a replacement for live sources. Always follow the actual repository's current default branch.

## Which repository to consult

| Task | Knowledge base | Start with |
|---|---|---|
| AI/ML, deep learning, LLMs, RAG, agents, MCP, computer vision, speech, model selection, evaluation, AI deployment | [AI Knowledge Base](https://github.com/faheemkhaskheli9/AI-Knowledge-Base) | [AGENTS.md](AGENTS.md), [INDEX.md](INDEX.md); optionally [llms.txt](llms.txt) or [scenarios/](scenarios/) |
| Software engineering, Python and API design, architecture, system design, scaling, databases, testing, reliability, DevOps | [Software Engineering Knowledge Base](https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase) | [AGENTS.md](https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/AGENTS.md), [INDEX.md](https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/INDEX.md), then matching topics |
| Cybersecurity, secure coding, threat modeling, OWASP, secrets, IAM, cloud/container security, SOC, incident response, AI/LLM security | [Cyber Security Knowledge Base](https://github.com/faheemkhaskheli9/Cyber-Security-Knowledge-Base) | [README.md](https://github.com/faheemkhaskheli9/Cyber-Security-Knowledge-Base/blob/main/README.md) (topic index), matching `kb/` files; `CLAUDE.md` is Claude-specific |
| Medicine, symptoms, drug safety, clinical evidence, healthcare AI | [Medical Knowledge Base](https://github.com/faheemkhaskheli9/Medical-Knowledge-Base) | [AGENTS.md](https://github.com/faheemkhaskheli9/Medical-Knowledge-Base/blob/HEAD/AGENTS.md), [KB-INDEX.md](https://github.com/faheemkhaskheli9/Medical-Knowledge-Base/blob/HEAD/KB-INDEX.md), matching clinical reference files |
| User-specific work, project decisions, CV, career and preferences **only when relevant and authorized** | [Personal Knowledge Base](https://github.com/faheemkhaskheli9/Personal-Knowledge-Base) (**private**) | `CONTEXT.md`, `INDEX.md`, then the minimum required files; only if access is granted |

The Personal KB is private. Its contents must not be reproduced, copied into public repos, or used to infer sensitive personal details unless the user's request requires them. If access is unavailable, skip it rather than guessing.

## Retrieval protocol

1. Classify the task and choose the smallest applicable set of KBs. Do **not** fetch all repositories by default. For mixed tasks, combine sources: for example, secure LLM deployments need AI + software engineering + cybersecurity; clinical AI adds the medical KB.
2. **In the current session**, fetch each applicable KB's instructions/index and the actual matched reference pages using the connected GitHub integration, or a public GitHub file URL if permitted. An index or memory alone is not the underlying evidence.
3. Follow the retrieved repo's rules, scope, links and source provenance. Prefer primary official sources for facts that change (model versions, licensing, vulnerabilities, APIs, regulatory rules, medication guidance, safety requirements). Check the item's date and reverify material stale facts.
4. Cite/link the actual files consulted; distinguish owner's notes and drafts from current verified evidence. If the KB is missing, unavailable, stale or conflicts with current primary sources, say so and use trustworthy alternatives. **Never claim you accessed a repo that you did not retrieve.**
5. Use live application source and git history as the authority for implementation state; knowledge bases are design/reference material and may be out of date.
6. Never put patient-identifiable information, secrets, unpublished private context or personal records in these public KBs. For cybersecurity, limit offensive instructions to authorized defensive/educational work.

## Paste into ChatGPT Project instructions

```text
When answering tasks relevant to my reference knowledge bases, begin with
https://github.com/faheemkhaskheli9/AI-Knowledge-Base/blob/main/CHATGPT-KB-ROUTER.md
and consult only the matching repositories:
- AI: https://github.com/faheemkhaskheli9/AI-Knowledge-Base
- Engineering: https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase
- Security: https://github.com/faheemkhaskheli9/Cyber-Security-Knowledge-Base
- Medical: https://github.com/faheemkhaskheli9/Medical-Knowledge-Base
- Personal context (private, only when relevant and accessible):
  https://github.com/faheemkhaskheli9/Personal-Knowledge-Base

Read the matched repo's AGENTS/README and index files, then the relevant
full topics during this session. Use cross-domain KBs for overlapping work.
Cite the actual file paths read; validate changeable facts with current
authoritative primary sources. Never invent KB contents or imply a repository
was read if it was not. Keep private information private. If access is
unavailable, explain the limitation and continue with reliable sources.
For code work, check the live source, issues and tests before relying on notes.
```

## Access and persistence limitations

- A ChatGPT Project can hold these instructions for chats **inside that Project**. Memory/custom instructions can preserve the preference, but cannot enforce always-on repository reads.
- A connected GitHub integration needs permission to each repository, especially the private Personal KB. Public URLs are a read-only fallback where supported.
- Repository edits are **not automatically synchronized into every conversation**. Retrieve current files when needed. If GitHub access is not available in a session, say so.
