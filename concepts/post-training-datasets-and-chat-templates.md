---
title: Post-training datasets and chat templates (formats, loss masking, dedup, decontamination)
category: concepts
tags: [post-training, sft, instruction-dataset, preference-data, chat-template, chatml, sharegpt, loss-masking, deduplication, minhash, decontamination, trl, axolotl]
use_cases:
  - "turn a ShareGPT or OpenAI-style conversation dataset into training samples for SFT"
  - "make sure fine-tuning trains only on the assistant turns, not on the prompt"
  - "remove duplicate and near-duplicate examples from an instruction dataset"
  - "check an SFT dataset for overlap with the benchmarks I report"
  - "fix a fine-tuned model that never stops generating or prints role tokens"
status: draft
last_verified: 2026-10-19
sources:
  - https://huggingface.co/docs/transformers/main/en/chat_templating
  - https://huggingface.co/docs/trl/sft_trainer
  - https://docs.axolotl.ai/docs/dataset-formats/conversation.html
  - https://github.com/mlabonne/llm-datasets
  - https://github.com/mlabonne/llm-course
  - https://github.com/huggingface/datatrove
  - https://github.com/MinishLab/semhash
  - https://www.cs.princeton.edu/courses/archive/spring13/cos598C/broder97resemblance.pdf
  - https://arxiv.org/abs/2107.06499
---

# Post-training datasets and chat templates (formats, loss masking, dedup, decontamination)

## Summary
Post-training (SFT and preference tuning) uses small, structured datasets: conversations of
`(role, content)` turns for SFT, and `(prompt, chosen, rejected)` triples for DPO-style
preference tuning. Before training, each conversation is stored in a format (ShareGPT,
OpenAI/HF `messages`), rendered into one string by the model's **chat template** (ChatML,
Llama, Alpaca...), tokenized, and given a **loss mask** so only assistant tokens are learned.
Quality comes from filtering: exact and near-duplicate removal (MinHash or embeddings),
n-gram decontamination against eval sets, then judge or reward-model scoring. This file
implements format conversion, ChatML rendering with an assistant-only mask, MinHash
near-dedup and n-gram decontamination in plain Python.

## Key concepts
- **Storage formats.** OpenAI/HF `messages`: `[{"role": "user"|"assistant"|"system", "content": ...}]`
  (what `apply_chat_template` and TRL's `SFTTrainer` expect). ShareGPT: `[{"from": "human"|"gpt"|"system", "value": ...}]`.
  Preference data: `prompt`, `chosen`, `rejected` (each a string or a messages list).
- **Chat template.** A Jinja template in `tokenizer_config.json` that turns messages into the exact
  string, special tokens included, the model saw in training. ChatML wraps each turn as
  `<|im_start|>role\ncontent<|im_end|>\n`. Alpaca uses `### Instruction:` / `### Response:` headers.
  Training and inference must use the same template.
- **Generation prompt.** At inference the template ends with the opening of an assistant turn
  (`add_generation_prompt=True`) so the model continues as the assistant. In training data the
  assistant turn is complete, and its end token (`<|im_end|>` / EOS) must be in the labels or
  the model never learns to stop.
- **Loss masking.** Label positions for system and user tokens are set to `-100` (ignored by the
  cross-entropy loss), so the model learns to answer, not to predict prompts. TRL exposes this as
  `assistant_only_loss=True` (needs a template with `{% generation %}` markers) and
  `completion_only_loss` for prompt/completion datasets.
- **Exact dedup** hashes a normalized copy of each sample. **Near dedup** (paraphrases, template
  variants) uses MinHash: the fraction of equal minimum hashes over word shingles estimates
  Jaccard similarity; LSH banding finds candidate pairs without comparing all pairs. Embedding
  similarity (e.g. semhash) catches semantic duplicates MinHash misses.
- **Decontamination** removes training samples that share long n-grams with benchmark items
  you will report on; otherwise the score measures memorization (Lee et al. found near-duplicates
  inside and across common corpora and their test sets).
- **Enhancement before filtering.** Seed data is usually expanded with synthetic generation,
  verified answers (unit tests, solvers), rejection sampling of several answers, and
  chain-of-thought rewrites; see [[synthetic-data-for-llm-training]].

## When to use / scenarios
- Preparing any SFT or DPO run (TRL, Unsloth, Axolotl, hosted fine-tuning): convert to the
  trainer's format and check the rendered sample once by eye.
- A fine-tuned model that rambles past its answer, prints `<|im_end|>` or `### Response:`, or
  ignores the system prompt: almost always a template or mask mismatch, not a data-quality problem.
- Merging several public instruction datasets: dedup across them, then decontaminate against
  your eval set before training.
- Not for pretraining-scale corpora (billions of documents): use a pipeline library
  (datatrove, NeMo Curator) with distributed MinHash; the logic below is the same but needs sharding.
- Not when prompting or RAG solves the task: see [[fine-tuning-and-peft]] for when to fine-tune at all.

## Setup & code
Standard library only. Converts ShareGPT to messages, renders ChatML with an assistant-only
mask (characters stand in for tokens), removes near-duplicates with MinHash + LSH, and drops
samples contaminated by an eval set.

```python
import hashlib, re

# 1) ShareGPT -> OpenAI/HF messages
ROLE = {"system": "system", "human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant"}

def sharegpt_to_messages(conv):
    return [{"role": ROLE[t["from"]], "content": t["value"].strip()} for t in conv["conversations"]]

# 2) ChatML rendering with an assistant-only loss mask.
# Real code: tokenizer.apply_chat_template(msgs, return_assistant_tokens_mask=True, return_dict=True)
def render_chatml(messages, add_generation_prompt=False):
    text, mask = "", []
    for m in messages:
        head, body = f"<|im_start|>{m['role']}\n", f"{m['content']}<|im_end|>\n"
        train = m["role"] == "assistant"
        text += head + body
        mask += [0] * len(head) + [int(train)] * len(body)   # end token is trained too
    if add_generation_prompt:
        text += "<|im_start|>assistant\n"; mask += [0] * len("<|im_start|>assistant\n")
    return text, mask

sample = {"conversations": [
    {"from": "system", "value": "You are concise."},
    {"from": "human", "value": "Capital of France?"},
    {"from": "gpt", "value": "Paris."}]}
msgs = sharegpt_to_messages(sample)
text, mask = render_chatml(msgs)
trained = "".join(c for c, k in zip(text, mask) if k)
print(repr(text))
print("trained on:", repr(trained))
assert trained == "Paris.<|im_end|>\n"
prompt, _ = render_chatml(msgs[:2], add_generation_prompt=True)
assert prompt.endswith("<|im_start|>assistant\n")

# 3) Exact + near-duplicate removal with MinHash and LSH banding
def normalize(s): return re.sub(r"\W+", " ", s.lower()).strip()

def shingles(s, k=3):
    w = normalize(s).split()
    return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}

def minhash(sh, num_perm=64):
    return [min(int.from_bytes(hashlib.blake2b(f"{p}:{x}".encode(), digest_size=8).digest(), "big")
                for x in sh) for p in range(num_perm)]

def jaccard(a, b): return len(a & b) / len(a | b)

def dedup(texts, threshold=0.7, bands=16, rows=4):
    seen_exact, buckets, kept, sigs = set(), {}, [], {}
    for i, t in enumerate(texts):
        h = hashlib.sha256(normalize(t).encode()).hexdigest()
        if h in seen_exact: continue
        seen_exact.add(h)
        sig = minhash(shingles(t), bands * rows)
        cands = set()
        for b in range(bands):
            key = (b, tuple(sig[b * rows:(b + 1) * rows]))
            cands |= set(buckets.get(key, []))
        est = lambda j: sum(x == y for x, y in zip(sig, sigs[j])) / len(sig)
        if any(est(j) >= threshold for j in cands): continue
        for b in range(bands):
            buckets.setdefault((b, tuple(sig[b * rows:(b + 1) * rows])), []).append(i)
        sigs[i] = sig; kept.append(i)
    return kept

data = [
    "Explain how a hash map handles collisions using chaining and open addressing in detail.",
    "explain how a hash map handles collisions using chaining and open addressing in detail!",   # exact after normalize
    "Explain how a hash map handles collisions using chaining and open addressing, in detail please.",  # near
    "Write a haiku about autumn leaves falling on a quiet river.",
    "What is the derivative of x squared times sine of x?",
]
kept = dedup(data)
print("kept:", kept, "true J(0,2) =", round(jaccard(shingles(data[0]), shingles(data[2])), 2))
assert kept == [0, 3, 4]

# 4) n-gram decontamination against an eval set
def ngrams(s, n=8):
    w = normalize(s).split()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}

eval_set = ["What is the derivative of x squared times sine of x? Show each step."]
eval_ngrams = set().union(*(ngrams(e) for e in eval_set))
clean = [data[i] for i in kept if not (ngrams(data[i]) & eval_ngrams)]
print("after decontamination:", len(clean), "of", len(kept))
assert len(clean) == 2
```

With the Hugging Face stack the same steps are: `datasets.map` to convert formats,
`tokenizer.apply_chat_template` to render, `SFTConfig(assistant_only_loss=True)` in TRL to
mask, and datatrove or semhash for dedup at scale.

## Choosing / trade-offs
- **Which template.** Fine-tuning an instruct model: keep its own template. Fine-tuning a base
  model: any template works, but pick one whose special tokens already exist in the tokenizer
  (or add them and resize embeddings) and ship it in `tokenizer_config.json` so inference servers
  apply it.
- **Assistant-only vs full-sequence loss.** Masking prompts is the default and helps when prompts
  are long or repetitive. Training on everything can help when user turns are themselves good
  text (multi-turn chat data), at the cost of the model learning to imitate users.
- **MinHash vs embeddings.** MinHash: cheap, exact-ish lexical overlap, scales to billions with
  LSH. Embedding dedup: catches paraphrases and translations, needs a model and a vector index,
  and a threshold tuned by looking at pairs near it.
- **Threshold.** Lower Jaccard thresholds remove more data, including legitimately similar
  samples (same task, different numbers). Inspect a random set of removed pairs before trusting it.
- **Quantity vs quality.** A few thousand clean, diverse, verified samples often beat a large
  noisy mix for style or format tuning; skills and knowledge-heavy tasks need more and more varied data.
- **Framework.** TRL for control and the latest methods, Unsloth for single-GPU speed and memory
  ([[unsloth-finetuning]]), Axolotl for YAML-configured multi-GPU runs with many dataset formats built in.

## Gotchas
- Missing end-of-turn token in the labels: the model never learns to stop. Check that the last
  trained token of each assistant turn is `<|im_end|>` or EOS.
- Template drift: training with one template and serving with another (or with Ollama's
  default) silently degrades quality. Print one fully rendered sample from training and one
  request from the server and compare.
- `pad_token = eos_token` plus masking all pad positions also masks the real EOS; use a separate
  pad token or mask by attention, not by token id.
- Two BOS tokens: templates often add BOS, and so does the tokenizer when called again on the
  rendered string. Tokenize with `add_special_tokens=False` after rendering.
- ShareGPT dumps often contain empty turns, two assistant turns in a row, or a trailing user
  turn with no answer; validate role alternation before converting.
- Dedup within each dataset but not across them leaves cross-source copies, and some public
  instruction sets share sources.
- Decontaminate with your exact eval prompts; benchmark paraphrases need embedding similarity,
  and the eval set itself should never be used to generate synthetic data.
- Licences: datasets distilled from API models inherit those providers' terms; check before a
  commercial release ([[model-licenses]]).

## Related
- [[fine-tuning-and-peft]] - SFT with LoRA/QLoRA on the data prepared here.
- [[synthetic-data-for-llm-training]] - generating and enhancing the samples before filtering.
- [[rlhf-and-preference-optimization]] - what preference (chosen/rejected) data trains.
- [[dpo-loss-from-scratch-numpy]] - the loss preference pairs feed.
- [[unsloth-finetuning]] - single-GPU training that applies chat templates for you.
- [[tokenization]] - special tokens and how the rendered string becomes ids.
- [[locality-sensitive-hashing-from-scratch]] - the banding trick behind MinHash LSH.
- [[reading-ai-benchmarks]] - why contamination makes benchmark scores unreliable.
- [[model-licenses]] - terms that follow distilled data.

## References
- Hugging Face, Chat templates: https://huggingface.co/docs/transformers/main/en/chat_templating
- TRL, SFTTrainer (dataset formats, `assistant_only_loss`): https://huggingface.co/docs/trl/sft_trainer
- Axolotl, conversation dataset formats: https://docs.axolotl.ai/docs/dataset-formats/conversation.html
- M. Labonne, LLM Datasets (curated post-training datasets and tools): https://github.com/mlabonne/llm-datasets
- M. Labonne, LLM Course, "Post-training datasets": https://github.com/mlabonne/llm-course
- datatrove (large-scale filtering and MinHash dedup): https://github.com/huggingface/datatrove
- semhash (embedding near-dedup and decontamination): https://github.com/MinishLab/semhash
- A. Broder, "On the resemblance and containment of documents" (1997).
- Lee et al., "Deduplicating Training Data Makes Language Models Better" (2021): https://arxiv.org/abs/2107.06499
