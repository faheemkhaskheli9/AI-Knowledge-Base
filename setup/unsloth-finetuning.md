---
title: Unsloth fine-tuning setup
category: setup
tags: [unsloth, lora, qlora, fine-tuning, sft, trl, colab, single-gpu, gguf, ollama, llama.cpp, grpo]
use_cases:
  - "fine-tune a small open model with QLoRA on a free Colab GPU"
  - "LoRA fine-tune a 7B-8B model on a single consumer NVIDIA GPU"
  - "export a fine-tuned model to GGUF and run it in Ollama or llama.cpp"
  - "train a model to answer in our company's format from a few thousand examples"
  - "run GRPO or DPO on an open model without a multi-GPU cluster"
status: draft
last_verified: 2026-10-15
sources:
  - https://github.com/unslothai/unsloth
  - https://github.com/unslothai/notebooks
  - https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf
  - https://huggingface.co/docs/trl/main/en/unsloth_integration
  - https://arxiv.org/abs/2305.14314
---

# Unsloth fine-tuning setup

## Summary
Unsloth is an open-source library that speeds up and slims down LoRA, QLoRA and full fine-tuning (plus RL methods such as GRPO and DPO) of open LLMs. It does this with hand-written kernels and memory-saving patches over Hugging Face Transformers, PEFT and TRL. Its README claims about 2x faster training with 70% less VRAM and no accuracy loss; per-notebook figures range from 1.5x to 2x faster and 20% to 80% less memory. That makes it the usual route for fine-tuning on one GPU or a free Colab notebook. It saves LoRA adapters, merged 16-bit weights, or GGUF files (via llama.cpp) ready for [[ollama]] and [[llama-cpp-gguf]].

## Key concepts
- **FastLanguageModel**: Unsloth's loader. `from_pretrained(model_name, max_seq_length, load_in_4bit=True)` loads a model (4-bit gives QLoRA), and `get_peft_model(model, r, target_modules, lora_alpha, ...)` attaches LoRA adapters. FastVisionModel does the same for vision models.
- **QLoRA vs LoRA**: QLoRA freezes a 4-bit (bitsandbytes NF4) base and trains 16-bit adapters. It uses far less VRAM and gives slightly lower quality than LoRA on a 16-bit base (`load_in_16bit=True`). Theory: [[fine-tuning-and-peft]], [[lora-from-scratch-numpy]], [[quantization]].
- **Pre-quantized repos**: the `unsloth/...-bnb-4bit` repos on Hugging Face download faster and avoid quantizing on load.
- **`use_gradient_checkpointing="unsloth"`**: Unsloth's offloaded checkpointing, which lets you use longer sequences and bigger batches in the same VRAM.
- **Training loop**: TRL's `SFTTrainer`/`SFTConfig` (or GRPO/DPO trainers) runs on the patched model. Unsloth is not a separate trainer API.
- **Chat formatting**: training text must use the model's own chat template, or the fine-tune learns the wrong format. `train_on_responses_only` (from `unsloth.chat_templates`) masks the user turns so only assistant tokens are learned.
- **Save methods**: `save_pretrained` saves adapters only. `save_pretrained_merged(..., save_method="merged_16bit" | "merged_4bit")` saves merged weights. `save_pretrained_gguf(dir, tokenizer, quantization_method="q4_k_m")` and `push_to_hub_gguf` write GGUF; the default is `fast_quantized` (q8_0), and other options include `f16`, `q5_k_m` and `q8_0`.
- **Licence**: the core package is Apache-2.0. Some optional components, such as the Unsloth Studio UI, are AGPL-3.0.

## When to use / scenarios
- Teach an open model a format, tone, task or tool-calling style from a few hundred to tens of thousands of examples, then serve it locally: support-reply drafting, extraction to JSON, domain classification.
- Budget training: one consumer GPU, a free Colab T4 or a single rented cloud GPU ([[gpu-cloud-options]]).
- Fine-tune, then run offline: export GGUF for laptops, edge boxes and air-gapped sites ([[edge-on-device]]).
- Reasoning RL experiments (GRPO) on small models.
- Not for: adding fresh or changing knowledge (use [[rag-basics]]). Not for models you can only reach through an API: use [[hosted-fine-tuning-apis]] for those. Not on Apple silicon without NVIDIA/AMD/Intel GPUs: use [[mlx-on-apple-silicon]]. For multi-node pretraining-scale jobs, see [[distributed-training]].

## Setup & code
Install (Linux or WSL, from the README; Windows needs `torchao` installed first, and AMD, Intel and Blackwell GPUs have their own guides):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv unsloth_env --python 3.13 && source unsloth_env/bin/activate
uv pip install unsloth --torch-backend=auto
```
In Colab, open one of the free notebooks linked from the README; they pin compatible `transformers` and `trl` versions in their first cell. Check your GPU and driver first ([[gpu-cuda-setup]]).

Prepare data as chat JSONL. This stdlib block validates and splits it, and it runs as is:
```python
import json, random

raw = [  # your examples; one conversation per record
    {"messages": [{"role": "system", "content": "Answer as JSON."},
                  {"role": "user", "content": f"Order {i}: 2 shirts, size M"},
                  {"role": "assistant", "content": json.dumps({"order": i, "item": "shirt", "qty": 2, "size": "M"})}]}
    for i in range(50)
]

def validate(rec):
    msgs = rec.get("messages", [])
    roles = [m["role"] for m in msgs if m["role"] != "system"]
    assert roles and roles[-1] == "assistant", "conversation must end with an assistant turn"
    assert all(a != b for a, b in zip(roles, roles[1:])), "user/assistant turns must alternate"
    assert all(isinstance(m.get("content"), str) and m["content"].strip() for m in msgs), "empty content"
    return rec

data = [validate(r) for r in raw]
random.Random(0).shuffle(data)
n_val = max(1, len(data) // 10)
splits = {"valid.jsonl": data[:n_val], "train.jsonl": data[n_val:]}
for name, rows in splits.items():
    with open(name, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(name, len(rows))
```
QLoRA SFT and GGUF export (pattern from the official notebooks):
<!-- skip-check: needs an NVIDIA/AMD/Intel GPU, torch and a model download -->
```python
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct", max_seq_length=2048, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=3407)

ds = load_dataset("json", data_files={"train": "train.jsonl"})["train"]
ds = ds.map(lambda r: {"text": tokenizer.apply_chat_template(r["messages"], tokenize=False)})

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=ds,
    args=SFTConfig(dataset_text_field="text", per_device_train_batch_size=2,
                   gradient_accumulation_steps=4, max_steps=60, learning_rate=2e-4,
                   optim="adamw_8bit", logging_steps=1, output_dir="outputs", report_to="none"))
trainer.train()

model.save_pretrained("lora_adapter"); tokenizer.save_pretrained("lora_adapter")   # adapters only
model.save_pretrained_gguf("model_gguf", tokenizer, quantization_method="q4_k_m")  # GGUF via llama.cpp
```
Then load the `.gguf` file in llama.cpp (`llama-cli -m <file>.gguf`) or with an Ollama Modelfile (`FROM ./<file>.gguf`, then `ollama create`). Check that the chat template carried over.

## Choosing / trade-offs
| Choice | Pick A when | Pick B when |
|---|---|---|
| QLoRA (4-bit base) vs LoRA (16-bit base) | VRAM is tight (Colab, 8-16 GB cards), or the model is large for your GPU | You have the VRAM and want the last bit of quality |
| LoRA vs full fine-tuning (`full_finetuning=True`) | Most adaptation tasks; cheap; adapters are swappable | Large distribution shift, and you have a big GPU and lots of data |
| Rank `r` 8-16 vs 32-128 | Format and style tasks | Harder tasks with more data; watch overfitting |
| Unsloth vs plain TRL + PEFT | One GPU, speed and VRAM matter | Multi-node training, or an unsupported architecture ([[huggingface-transformers]]) |
| Unsloth vs hosted fine-tuning | You need the open weights, offline serving or GGUF | You want no GPU operations and the base model is closed |
| GGUF quant `q4_k_m` vs `q8_0` / `f16` | Laptops and CPU serving | Quality-critical serving, or as an intermediate for later re-quantizing |

## Gotchas
- Chat template mismatch is the top failure. Train with the same template that inference uses (the GGUF and Ollama Modelfile included), or the model rambles or ignores the system prompt.
- Train on responses only. Otherwise loss on long user prompts dominates and the model learns to echo them.
- Library versions move fast. The notebooks pin `transformers` and `trl`; mixing in versions from another guide is the usual cause of import errors.
- `max_steps=60` in examples is a smoke test. Real runs set `num_train_epochs`, hold out a validation set, and evaluate on task metrics ([[llm-evaluation]]), not just loss.
- Overfitting on small datasets shows up as copied training answers. Use fewer epochs, a lower rank or more varied data.
- GGUF export compiles or downloads llama.cpp and needs free disk space for the merged 16-bit copy. Small Colab or Kaggle disks fill up.
- A model that works inside Unsloth can behave differently after GGUF export (template, quant, sampling defaults). Re-test the exported file.
- Respect the base model's licence and acceptable-use policy ([[model-licenses]]). Gated repos need a Hugging Face token. Never hard-code tokens in notebooks you share.
- Pre-quantized `bnb-4bit` weights are for training. Serve from the merged or GGUF output, not from the 4-bit training checkpoint.

## Related
- [[fine-tuning-and-peft]] - when to fine-tune, LoRA/QLoRA theory and data preparation.
- [[lora-from-scratch-numpy]] - what the adapters compute.
- [[quantization]] - 4-bit NF4 and GGUF quant types.
- [[llama-cpp-gguf]], [[ollama]] - running the exported GGUF.
- [[gpu-cuda-setup]] - drivers, CUDA and VRAM sizing.
- [[huggingface-transformers]] - the underlying stack and Hub tokens.
- [[hosted-fine-tuning-apis]] - managed alternative.
- [[rlhf-and-preference-optimization]] - DPO and GRPO after SFT.
- [[mlx-on-apple-silicon]] - the Mac equivalent.

## References
- Unsloth README (install, licences, notebooks, speed and memory claims): https://github.com/unslothai/unsloth
- Unsloth notebooks (FastLanguageModel, SFTTrainer and save/GGUF code): https://github.com/unslothai/notebooks
- Saving to GGUF: https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf
- TRL Unsloth integration: https://huggingface.co/docs/trl/main/en/unsloth_integration
- Dettmers et al. (2023), QLoRA: https://arxiv.org/abs/2305.14314
