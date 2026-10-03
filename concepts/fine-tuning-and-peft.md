---
title: Fine-tuning and PEFT (LoRA, QLoRA)
category: concepts
tags: [fine-tuning, lora, qlora, peft, trl, sft, bitsandbytes]
use_cases:
  - "fine-tune a 7B model on a single 24GB GPU for a company-specific tone and format"
  - "teach a small model to emit strict JSON for an extraction pipeline in insurance or logistics"
  - "adapt an open model to Urdu or Roman-Urdu customer messages"
  - "serve many customer-specific adapters from one base model"
  - "decide whether to fine-tune or just use RAG and prompting"
status: draft
last_verified: 2026-10-03
sources:
  - https://huggingface.co/docs/peft
  - https://huggingface.co/docs/trl/sft_trainer
  - https://arxiv.org/abs/2106.09685
  - https://arxiv.org/abs/2305.14314
---

# Fine-tuning and PEFT (LoRA, QLoRA)

## Summary
Fine-tuning continues training a pretrained model on your examples to change its behaviour (style, format, task skill, domain vocabulary). Parameter-efficient methods such as LoRA train small added matrices instead of all weights, cutting memory and storage by orders of magnitude; QLoRA does this on top of a 4-bit quantised base so large models fit on one consumer GPU.

## Key concepts
- **Full fine-tuning**: update all weights. Best ceiling, but needs memory for weights + gradients + optimizer states (several times model size) and produces a full copy per task.
- **LoRA**: freeze W, learn low-rank update BA (rank r, scaled by alpha/r) on chosen layers (usually attention projections, often all linear layers). Adapters are MBs and can be merged into the base or hot-swapped.
- **QLoRA**: base model in 4-bit NF4 (via bitsandbytes), LoRA adapters in higher precision, paged optimizers.
- **SFT**: supervised fine-tuning on (prompt, response) or chat-formatted data; loss usually only on assistant tokens.
- **What fine-tuning is good at**: format, tone, narrow task skill, distilling a big model into a small one. **Poor at**: injecting lots of new, changing facts (use RAG).
- **Data > hyperparameters**: a few hundred to a few thousand clean, diverse examples often beat tens of thousands of noisy ones.
- Alignment after SFT: see [[rlhf-and-preference-optimization]].

## When to use / scenarios
- Prompting/few-shot is not reliable enough, or prompts are too long/costly at your volume.
- You need a small, cheap, private model with consistent output (support triage, invoice field extraction, SQL generation for a fixed schema).
- Real-world: a clinic tagging notes on-prem; a law firm drafting in house style; a retailer classifying product titles.
- Not when: knowledge changes frequently (RAG), few examples exist (prompting), or a hosted API already meets quality and privacy needs.

## Setup & code
`pip install transformers peft trl bitsandbytes datasets accelerate` (bitsandbytes needs a supported NVIDIA GPU; check its docs for other platforms). Library APIs move fast; pin versions in real projects.

```python
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

name = "Qwen/Qwen2.5-0.5B-Instruct"            # swap for your base model
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16)   # drop for plain LoRA
model = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb)
tok = AutoTokenizer.from_pretrained(name)

# dataset with a "messages" column: [{"role": "user", ...}, {"role": "assistant", ...}]
ds = load_dataset("trl-lib/Capybara", split="train[:1000]")

peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules="all-linear", task_type="CAUSAL_LM")
trainer = SFTTrainer(model=model, train_dataset=ds, peft_config=peft_cfg,
                     processing_class=tok,
                     args=SFTConfig(output_dir="out", per_device_train_batch_size=2,
                                    gradient_accumulation_steps=8, learning_rate=2e-4,
                                    num_train_epochs=1, bf16=True, logging_steps=10))
trainer.train()
trainer.save_model("out/adapter")               # adapter only (small)
```
Merge for deployment: load the base in 16-bit, `PeftModel.from_pretrained(...)`, then `merge_and_unload()`.

## Choosing / trade-offs
- Full FT vs LoRA: LoRA matches full FT on most narrow tasks at a fraction of the memory; full FT can win for big distribution shifts or continued pretraining.
- LoRA vs QLoRA: QLoRA saves memory (about 4x on weights) but is slower and slightly lossy; use LoRA if the 16-bit base fits.
- Rank r: 8-64 typical; raise only if underfitting.
- Hosted fine-tuning APIs (no GPUs, less control, vendor lock-in) vs self-hosted (control, privacy, ops cost).

## Gotchas
- Wrong or missing chat template makes a fine-tune behave worse than the base model.
- Training on prompt tokens (no completion-only loss) wastes capacity and can leak prompt style.
- Learning rates for LoRA (about 1e-4 to 2e-4) are far higher than for full FT (about 1e-5).
- Overfitting on small sets: hold out a validation set and a task-specific eval ([[llm-evaluation]]).
- Catastrophic forgetting of general ability; mix in some general data if it matters.
- Merging an adapter trained on a 4-bit base into a 16-bit base introduces small mismatch; evaluate after merging.
- Training data containing PII will be memorised.

## Related
- [[quantization]] - the 4-bit part of QLoRA.
- [[rlhf-and-preference-optimization]] - DPO and friends after SFT.
- [[pretraining-and-scaling-laws]] - when continued pretraining is the right tool.
- [[rag-basics]] - the alternative for knowledge injection.
- [[small-language-models]] - common targets of fine-tuning.
- [[data-labeling-and-synthetic-data]] - producing training data.
- [[huggingface-transformers]] - environment setup.

## References
- PEFT docs: https://huggingface.co/docs/peft
- TRL SFTTrainer: https://huggingface.co/docs/trl/sft_trainer
- Hu et al., LoRA: https://arxiv.org/abs/2106.09685
- Dettmers et al., QLoRA: https://arxiv.org/abs/2305.14314
