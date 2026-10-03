---
title: Tokenization
category: concepts
tags: [tokenizer, bpe, tiktoken, sentencepiece, token-count, huggingface]
use_cases:
  - "estimate API cost and context usage for documents before sending them to an LLM"
  - "chunk documents for a RAG pipeline by token count instead of characters"
  - "support Urdu, Arabic or other non-English text in an LLM product without exploding token costs"
  - "add special tokens or a chat template when fine-tuning an open model"
  - "debug an LLM that miscounts letters or fails at arithmetic on numbers"
status: draft
last_verified: 2026-10-03
sources:
  - https://huggingface.co/docs/transformers/main_classes/tokenizer
  - https://huggingface.co/docs/transformers/chat_templating
  - https://github.com/openai/tiktoken
  - https://arxiv.org/abs/1508.07909
---

# Tokenization

## Summary
Tokenization converts text into integer IDs from a fixed vocabulary of sub-word pieces; the model never sees raw characters. It determines cost (billing is per token), effective context length, multilingual efficiency and some odd failure modes (spelling, arithmetic).

## Key concepts
- **Sub-word algorithms**: BPE (most GPT/Llama-style models, often byte-level), WordPiece (BERT), Unigram/SentencePiece (T5, many multilingual models).
- **Vocabulary**: typically tens of thousands to a few hundred thousand entries. Larger vocab = shorter sequences but bigger embedding matrix.
- **Byte-level fallback**: any text is representable; rare scripts just cost many tokens.
- **Special tokens**: BOS/EOS, padding, role markers. Chat models use a **chat template** that wraps messages in model-specific special tokens.
- **Tokenizer is part of the model**: a model must be used with its own tokenizer; vendors' tokenizers differ, so counts differ between providers.
- Rule of thumb for English: ~4 characters or ~0.75 words per token. Other scripts are often 2-5x worse.

## When to use / scenarios
- Cost and limit planning: count tokens before calling an API.
- RAG chunking: size chunks in tokens to fit embedding model limits.
- Fine-tuning: apply the exact chat template of the base model.
- Real-world: a Pakistani telecom support bot in Urdu sees much higher token usage than English; budget for it or choose a model with a better multilingual tokenizer.
- Not needed: when calling a hosted API with short prompts and no cost sensitivity.

## Setup & code
`pip install transformers tiktoken`

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")       # OpenAI-family encoding
ids = enc.encode("Tokenization is subtle.")
print(len(ids), enc.decode(ids))

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
msgs = [{"role": "user", "content": "Hello"}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
print(text)                                      # shows the model's special-token layout
print(len(tok(text).input_ids))
```

For Anthropic models use the API's token-counting endpoint (see the official docs) rather than a local approximation; tiktoken is only exact for OpenAI encodings.

## Choosing / trade-offs
- Count locally (fast, approximate for other vendors) vs via provider endpoint (exact, network call).
- Extending a vocabulary for a new language needs embedding resizing and continued pretraining; usually cheaper to pick a model whose tokenizer already covers the language.
- Character-chunking is simple but ignores model limits; token chunking is accurate.

## Gotchas
- Leading spaces matter: " hello" and "hello" are different tokens.
- Numbers split inconsistently, hurting arithmetic; use tools/code for math.
- "How many r in strawberry" fails because of sub-word pieces, not lack of intelligence.
- Applying a chat template twice, or omitting it, quietly degrades fine-tuned model quality.
- Setting `pad_token = eos_token` for training can teach the model never to stop if EOS is masked out of the loss; check labels.
- Token counts from one vendor do not transfer to another.

## Related
- [[transformers-and-attention]] - what consumes the token IDs.
- [[embeddings]] - input embedding table is indexed by token ID.
- [[fine-tuning-and-peft]] - chat templates and special tokens in training data.
- [[prompt-caching-and-cost]] - token counts drive cost.
- [[rag-basics]] - chunking by tokens.

## References
- Hugging Face tokenizers: https://huggingface.co/docs/transformers/main_classes/tokenizer
- Chat templating: https://huggingface.co/docs/transformers/chat_templating
- tiktoken: https://github.com/openai/tiktoken
- Sennrich et al., BPE: https://arxiv.org/abs/1508.07909
