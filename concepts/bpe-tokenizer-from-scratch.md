---
title: Byte-level BPE tokenizer from scratch (train, encode, decode)
category: concepts
tags: [bpe, byte-pair-encoding, tokenizer, tokenization, subword, byte-level, pre-tokenization, llm, python, from-scratch, deep-learning-basics]
use_cases:
  - "implement byte-level BPE training, encoding and decoding in plain Python"
  - "understand why LLM tokenizers split text with a regex before merging"
  - "see why non-English text and emoji cost more tokens per character"
  - "explain how GPT-style tokenizers work and where their quirks come from in an interview"
status: stable
last_verified: 2026-10-04
sources:
  - https://aclanthology.org/P16-1162/
  - https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
  - https://github.com/karpathy/minbpe
  - https://huggingface.co/learn/llm-course/chapter6/5
---

# Byte-level BPE tokenizer from scratch (train, encode, decode)

## Summary
Byte-pair encoding (BPE) builds a subword vocabulary by starting from single symbols and repeatedly merging the most frequent adjacent pair into a new token. Byte-level BPE, used by GPT-2 and later GPT models, Llama 3 and many others, starts from the 256 byte values, so any UTF-8 text encodes without an unknown token and decodes back exactly. The pure-Python version below learns 40 merges on a toy corpus. It shows why real tokenizers first split text with a regex: without it, one "token" swallows a whole 75-character sentence. It also shows why accented letters and emoji cost several tokens each when the training data never saw them.

## Key concepts
- **Base vocabulary.** The 256 bytes. Every string is a byte sequence, so coverage is complete and no `<unk>` is needed.
- **Training loop.** Count adjacent pairs across the corpus, merge the most frequent into a new id, record the merge, repeat until the vocabulary reaches its target size (tens of thousands to a few hundred thousand in real models).
- **Merge table.** An ordered dictionary `(a, b) → new_id`. Its order is the tokenizer: encoding replays merges by rank.
- **Encoding.** Start from bytes, then repeatedly apply the earliest-learned merge present in the sequence until none applies. Greedy longest-match gives different results.
- **Pre-tokenization.** Split text into chunks (words with their leading space, numbers, punctuation runs, whitespace) with a regex, and only merge within a chunk. This stops merges across word boundaries and keeps " cat" the same token wherever it appears.
- **Counting distinct chunks.** Train on a `Counter` of chunks, not the raw text: repeated words cost one entry, which makes training much faster.
- **Decoding.** Concatenate each token's bytes and decode UTF-8. Individual tokens may be partial characters; only the full sequence is guaranteed valid.

## When to use / scenarios
- Learning: where token counts, context limits and per-token prices come from, and why models struggle with spelling, digits and reversed strings.
- Interviews: "how does BPE work", "BPE vs WordPiece vs Unigram", "why byte-level", "why does my non-English prompt cost more".
- Practice: estimating token counts, training a domain tokenizer (code, a new language, chemistry strings) for a small model, debugging odd tokenization.
- Not for: production training on large corpora. Use Hugging Face `tokenizers` (Rust, minutes for gigabytes) or SentencePiece; for an existing model always use that model's own tokenizer ([[tokenization]]).

## Setup & code
Standard library only. Runs in about a second.

```python
import re
from collections import Counter

# GPT-2-style pre-tokenizer (simplified): merges never cross a word/space/punctuation boundary
SPLIT = re.compile(r" ?[^\W\d_]+| ?\d+| ?[^\w\s]+|\s+")


def merge(ids, pair, new_id):
    out, j = [], 0
    while j < len(ids):
        if j < len(ids) - 1 and (ids[j], ids[j + 1]) == pair:
            out.append(new_id); j += 2
        else:
            out.append(ids[j]); j += 1
    return out


def train_bpe(text, vocab_size, pretokenize=True):
    """Byte-level BPE: start from the 256 byte values, repeatedly merge the most frequent adjacent pair."""
    chunks = SPLIT.findall(text) if pretokenize else [text]
    words = Counter(tuple(w.encode("utf-8")) for w in chunks)        # count each distinct chunk once
    merges, vocab = {}, {b: bytes([b]) for b in range(256)}
    for new_id in range(256, vocab_size):
        pairs = Counter()
        for w, n in words.items():
            for p in zip(w, w[1:]):
                pairs[p] += n
        if not pairs:
            break
        pair = max(pairs, key=pairs.get)                  # ties: first seen wins
        words = Counter({tuple(merge(list(w), pair, new_id)): n for w, n in words.items()})
        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
    return merges, vocab


def encode(text, merges):
    out = []
    for chunk in SPLIT.findall(text):
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            pair = min(zip(ids, ids[1:]), key=lambda p: merges.get(p, float("inf")))  # earliest-learned merge first
            if pair not in merges:
                break
            ids = merge(ids, pair, merges[pair])
        out += ids
    return out


def decode(ids, vocab):
    return b"".join(vocab[t] for t in ids).decode("utf-8", errors="replace")


corpus = ("the cat sat on the mat. the dog sat on the log. "
          "a cat and a dog met on the mat and then the cat ran. ") * 20 + \
         "low lower lowest newer newest wider widest. "

raw, raw_vocab = train_bpe(corpus, 256 + 40, pretokenize=False)
print("no pre-tokenizer, longest token:", repr(max((raw_vocab[m] for m in raw.values()), key=len).decode()))

merges, vocab = train_bpe(corpus, 256 + 40)
print("first 10 merges:", [vocab[m].decode() for m in list(merges.values())[:10]])
print("longest learned tokens:", sorted((vocab[m].decode() for m in merges.values()), key=len)[-4:])

for s in ["the cat sat on the mat.", "the lowest newest cat", "naïve café 🙂"]:
    ids = encode(s, merges)
    assert decode(ids, vocab) == s                        # lossless round trip, any Unicode
    print(f"{s!r}: {len(s.encode())} bytes -> {len(ids)} tokens: {[vocab[t] for t in ids]}")

n_bytes, n_tok = len(corpus.encode()), len(encode(corpus, merges))
print(f"corpus compression: {n_bytes} bytes -> {n_tok} tokens ({n_bytes / n_tok:.2f} bytes/token)")
```

Output (Python 3.14):
```
no pre-tokenizer, longest token: 'cat sat on the mat. the dog sat on the log. a cat and a dog met on the mat '
first 10 merges: ['th', 'the', 'at', ' the', ' a', ' c', ' cat', ' o', ' on', ' m']
longest learned tokens: [' newe', ' wide', ' lower', ' lowest']
'the cat sat on the mat.': 23 bytes -> 7 tokens: [b'the', b' cat', b' sat', b' on', b' the', b' mat', b'.']
'the lowest newest cat': 21 bytes -> 5 tokens: [b'the', b' lowest', b' newe', b'st', b' cat']
'naïve café 🙂': 17 bytes -> 16 tokens: [b'n', b'a', b'\xc3', b'\xaf', b'v', b'e', b' c', b'a', b'f', b'\xc3', b'\xa9', b' ', b'\xf0', b'\x9f', b'\x99', b'\x82']
corpus compression: 2064 bytes -> 593 tokens (3.48 bytes/token)
```

Trained on the raw string, BPE memorises the repeated sentence and ends up with a 75-character token: great compression on this corpus, useless for anything else. With the regex pre-tokenizer the merges are subwords: "th", "the", " the", " cat". Note the leading space is part of the word, so "the" at the start of a line and " the" mid-sentence are different tokens. Common words become one token each, and the rarer comparative forms share pieces (" lowest", but " newe" + "st"). Text the corpus never saw falls back to raw bytes: "ï", "é" and "🙂" cost 2, 2 and 4 tokens, which is why languages and scripts under-represented in a tokenizer's training data use more tokens (and money and context) for the same content. Every string still round-trips exactly.

## Choosing / trade-offs
- **Vocabulary size.** Larger vocabularies give shorter sequences (cheaper attention, more text per context window) but a bigger embedding and output layer and rarer, less-trained tokens. GPT-2 used about 50k; recent models use 100k–260k to cover more languages and code.
- **BPE vs WordPiece vs Unigram.** BPE merges by frequency. WordPiece (BERT) merges by a likelihood score. Unigram (SentencePiece) starts large and prunes, and can sample several segmentations for regularisation. Quality differences are small next to vocabulary size and training data.
- **Byte-level vs character-level base.** Byte-level never sees an unknown symbol. SentencePiece BPE starts from characters with byte fallback for rare ones; both work.
- **Pre-tokenizer regex.** Decides how digits are split (one digit per token helps arithmetic), whether contractions split, and how whitespace and code indentation are handled. It matters as much as the merge algorithm.
- **Tooling.** `tiktoken` to count tokens for OpenAI models, Hugging Face `tokenizers` to train fast BPE/WordPiece/Unigram, `sentencepiece` for language-agnostic training without pre-splitting by spaces.

## Gotchas
- The same word gets different tokens with and without a leading space, and with different capitalisation. Prompt formatting changes token counts and sometimes behaviour.
- Splitting a token sequence at an arbitrary point can cut a UTF-8 character in half. Decode the whole sequence, or buffer bytes when streaming.
- Encoding must replay merges by rank. Greedy longest-match over the vocabulary produces different, non-canonical ids the model never saw in training.
- Token counts differ between tokenizers. Never estimate one model's cost with another model's tokenizer.
- Special tokens (`<|endoftext|>`, chat role markers) must be handled outside BPE and never produced from user text, or users can inject them.
- The naive loop here is O(merges × corpus). Real trainers update pair counts incrementally with a priority queue.
- Tokenization explains many LLM quirks: miscounting letters, weak digit arithmetic, trailing-space sensitivity. Check the tokens before blaming the model.

## Related
- [[tokenization]] - tokenizer families, library usage and practical token counting.
- [[char-mlp-language-model-from-scratch-numpy]] - a character-level model that skips subword tokenization.
- [[word2vec-skip-gram-from-scratch-numpy]] - learns embeddings for a fixed word vocabulary.
- [[transformer-block-from-scratch-numpy]] - the model that consumes the token ids.
- [[prompt-caching-and-cost]] - why tokens per request drive cost and latency.
- [[embeddings]] - the embedding table indexed by token id.

## References
- Sennrich, Haddow and Birch (2016), "Neural Machine Translation of Rare Words with Subword Units", ACL: https://aclanthology.org/P16-1162/
- Radford et al. (2019), "Language Models are Unsupervised Multitask Learners" (GPT-2, byte-level BPE): https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- Karpathy, minbpe (minimal BPE reference implementation): https://github.com/karpathy/minbpe
- Hugging Face LLM course, "Byte-Pair Encoding tokenization": https://huggingface.co/learn/llm-course/chapter6/5
