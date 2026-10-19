# Learning paths

Ordered reading lists through existing topics, one per role. Read the steps in
order; each one builds on the ones before it. Skip a step you already know, and
follow a topic's `## Related` links when you need more depth. Terms are defined
in [glossary.md](glossary.md); for "which approach for X?" start at
[scenarios/scenario-chooser.md](scenarios/scenario-chooser.md) instead.
Not a topic file: `build_index.py` leaves it out of INDEX.md but fails on a broken `[[link]]` here.

Paths: [Beginner](#beginner) | [LLM engineer](#llm-engineer) | [Agent builder](#agent-builder) |
[ML engineer](#ml-engineer) | [Computer vision](#computer-vision) | [Speech and audio](#speech-and-audio) |
[MLOps and deployment](#mlops-and-deployment) | [LLM scientist](#llm-scientist) |
[Internals from scratch](#internals-from-scratch)

## Beginner

From "what is machine learning" to calling an LLM and answering from your own documents.

1. [[ml-fundamentals]] - the vocabulary: supervised vs unsupervised, train/test, overfitting.
2. [[math-for-machine-learning]] - just enough linear algebra, calculus and probability to follow the rest.
3. [[classic-ml-scikit-learn]] - train and evaluate a first model end to end with scikit-learn.
4. [[model-evaluation-and-metrics]] - pick a metric that matches the problem before trusting a score.
5. [[data-leakage-and-validation-splits]] - the most common reason good offline scores fail in production.
6. [[neural-network-fundamentals]] - layers, activations and loss: what a network is.
7. [[gradient-descent]] - how every network (and many classic models) is trained.
8. [[pytorch-basics]] - tensors, autograd and a training loop in the framework most topics use.
9. [[tokenization]] - how text becomes tokens, which sets context limits and cost.
10. [[embeddings]] - vectors that capture meaning; the basis of search and RAG.
11. [[transformers-and-attention]] - the architecture behind every modern LLM.
12. [[api-sdk-setup]] - get a key and make a first call to a hosted LLM.
13. [[prompt-engineering]] - get reliable behaviour from a model without training it.
14. [[rag-basics]] - answer from your own documents instead of the model's memory.
15. [[model-selection]] - choose a model by quality, cost, latency and privacy.

## LLM engineer

Building and shipping an LLM feature: prompts, retrieval, evals, safety and cost.

1. [[tokenization]] - tokens are the unit of context, latency and price.
2. [[decoding-and-sampling]] - what temperature and top-p change and when to set them.
3. [[model-selection]] - pick a starting model and know when to switch.
4. [[api-sdk-setup]] - SDK setup, keys and the request shape every later topic assumes.
5. [[prompt-engineering]] - system prompts, examples and structure.
6. [[structured-output]] - get JSON your code can parse instead of prose.
7. [[tool-calling]] - let the model call your functions.
8. [[rag-basics]] - the retrieve-then-generate pattern most LLM features need.
9. [[chunking-strategies]] - how splitting documents decides what retrieval can find.
10. [[embedding-models]] - choose the model that turns chunks into vectors.
11. [[vector-databases]] - store and search those vectors at scale.
12. [[advanced-rag]] - hybrid search, query rewriting and contextual retrieval when basic RAG misses.
13. [[rerankers]] - reorder a shortlist for precision at low cost.
14. [[llm-evaluation]] - build a golden set so changes are measured, not guessed.
15. [[llm-as-judge-and-eval-driven-development]] - grade open-ended outputs and gate changes on evals.
16. [[hallucination-and-grounding]] - make answers cite sources and say "I don't know".
17. [[prompt-injection]] - the main security risk once a model reads untrusted text.
18. [[guardrails-and-safety]] - input and output checks around the model.
19. [[streaming-and-async-llm-apps]] - stream tokens and run calls concurrently for responsive UIs.
20. [[prompt-caching-and-cost]] - cut cost and latency on repeated prompt prefixes.
21. [[cost-and-latency]] - the full set of levers once usage grows.
22. [[llm-observability]] - trace every call to debug and monitor in production.
23. [[fine-tuning-and-peft]] - when prompting and RAG are not enough, and how to fine-tune cheaply.

## Agent builder

From one tool call to long-running, multi-agent systems. Do the LLM engineer path through step 7 first.

1. [[tool-calling]] - the primitive every agent is built on.
2. [[structured-output]] - reliable arguments and results between steps.
3. [[workflows-vs-agents]] - use a fixed workflow unless the task really needs autonomy.
4. [[agents]] - the agent loop, planning and stopping conditions.
5. [[agent-frameworks]] - what frameworks add over a plain loop, and when to skip them.
6. [[model-context-protocol]] - connect tools and data through standard MCP servers.
7. [[context-engineering]] - decide what enters the context window at each step.
8. [[agent-memory]] - keep state beyond one context window.
9. [[agent-skills]] - package instructions and scripts the agent loads on demand.
10. [[web-search-and-grounding]] - give the agent fresh information with citations.
11. [[sandboxed-code-execution]] - run model-written code without risking the host.
12. [[prompt-injection]] - defend an agent that reads untrusted pages and tool results.
13. [[computer-use-agents]] - operate GUIs and browsers when there is no API.
14. [[coding-agents]] - agents that edit repositories, and the instruction files that steer them.
15. [[multi-agent-systems]] - when splitting work across agents helps and when it hurts.
16. [[a2a-protocol]] - let agents from different vendors talk to each other.
17. [[long-running-agents]] - checkpoints, approvals and budgets for work that takes hours.
18. [[agent-evaluation]] - measure task success and trajectories, not just final answers.

## ML engineer

Tabular and classic ML to production quality, then deep-learning training.

1. [[exploratory-data-analysis]] - understand the data before modelling it.
2. [[data-leakage-and-validation-splits]] - set up validation that predicts production performance.
3. [[feature-engineering]] - build informative features from raw data.
4. [[categorical-encoding]] - turn categories into numbers without leaking the target.
5. [[missing-data-and-imputation]] - handle gaps without biasing the model.
6. [[linear-models]] - the fast, interpretable baseline.
7. [[decision-trees-and-random-forests]] - non-linear models that need little preprocessing.
8. [[gradient-boosting-tabular]] - the usual winner on tabular data.
9. [[model-evaluation-and-metrics]] - choose and read the right metric.
10. [[imbalanced-data]] - rare classes, where accuracy misleads.
11. [[hyperparameter-tuning]] - tune efficiently instead of by hand.
12. [[probability-calibration]] - make predicted probabilities usable for decisions.
13. [[model-interpretability]] - explain predictions to users and regulators.
14. [[time-series-forecasting]] - forecasting, which needs its own validation and models.
15. [[deep-learning-training]] - optimizers, regularization and normalization for networks.
16. [[learning-rate-schedules]] - the single setting that most often makes or breaks training.
17. [[debugging-neural-network-training]] - what to check when the loss will not go down.
18. [[experiment-tracking]] - record runs so results can be compared and reproduced.
19. [[online-learning-and-concept-drift]] - keep a deployed model accurate as data changes.
20. [[mlops-lifecycle]] - the path from notebook to monitored production model.

## Computer vision

From convolutions to detection, segmentation, vision-language models and generation.

1. [[convolution-arithmetic-and-receptive-field]] - how convolution layers see an image.
2. [[cnn-and-rnn-architectures]] - the standard CNN building blocks.
3. [[residual-and-skip-connections]] - why deep vision networks (ResNet, U-Net) train.
4. [[data-augmentation]] - the cheapest accuracy gain for image models.
5. [[transfer-learning-and-domain-adaptation]] - start from a pretrained backbone, not from scratch.
6. [[image-classification]] - the first task to get working end to end.
7. [[object-detection]] - boxes and labels: models, data and metrics.
8. [[nms-iou-and-map-from-scratch]] - the post-processing and metric every detector uses.
9. [[segmentation]] - per-pixel labels: semantic and instance.
10. [[segment-anything-sam2]] - promptable masks without training your own model.
11. [[open-vocabulary-detection]] - detect objects named in text, with no labelled data.
12. [[image-retrieval-clip-search]] - search images by text or by example with CLIP embeddings.
13. [[vision-language-models]] - ask questions about images in natural language.
14. [[ocr]] - read text from images and scans.
15. [[document-layout-and-table-extraction]] - recover structure and tables from documents.
16. [[video-analytics]] - detection and tracking over video streams.
17. [[diffusion-models]] - how modern image generators work.
18. [[image-generation-models]] - choose a generator for a product.
19. [[edge-on-device]] - run vision models on phones and embedded devices.

## Speech and audio

From audio features to transcription, synthesis and real-time voice agents.

1. [[mel-spectrogram-and-mfcc-from-scratch]] - the audio features nearly every speech model takes as input.
2. [[sequence-to-sequence-and-ctc]] - the two ways models align audio frames to text.
3. [[speech-to-text]] - choose and run an ASR model.
4. [[word-error-rate-from-scratch]] - measure transcription quality the standard way.
5. [[voice-activity-detection-from-scratch]] - find where speech is, before transcribing it.
6. [[speaker-diarization]] - who spoke when, for meetings and calls.
7. [[speech-translation]] - speech in one language to text or speech in another.
8. [[text-to-speech]] - choose and run a TTS model.
9. [[audio-classification-and-sound-events]] - label non-speech sounds and events.
10. [[wake-word-detection]] - always-on keyword spotting on device.
11. [[voice-agents]] - combine STT, an LLM and TTS into a real-time assistant.
12. [[speech-to-speech-realtime-models]] - native audio models and when they beat the cascade.
13. [[voice-cloning-and-ethics]] - consent and safeguards before cloning a voice.

## MLOps and deployment

From a reproducible environment to serving, scaling, monitoring and compliance.

1. [[python-env-uv]] - reproducible Python environments and lockfiles.
2. [[gpu-cuda-setup]] - drivers, CUDA and framework versions that work together.
3. [[docker-for-ml]] - package a model with GPU access in a container.
4. [[model-checkpointing-and-export]] - save and export weights safely (safetensors, ONNX).
5. [[experiment-tracking]] - know which run produced which model.
6. [[model-registry-and-versioning]] - version models and prompts with rollback.
7. [[serving-with-fastapi]] - wrap a model or LLM call in an HTTP API.
8. [[quantization]] - shrink a model to fit cheaper hardware.
9. [[inference-servers-vllm]] - serve open-weight LLMs efficiently.
10. [[onnx-and-triton-inference-server]] - serve non-LLM models across frameworks.
11. [[gpu-cloud-options]] - where to rent GPUs and how the options differ.
12. [[kubernetes-for-llm-serving]] - autoscale model servers on a cluster.
13. [[managed-cloud-ai-platforms]] - use Bedrock, Vertex AI or Azure instead of running models yourself.
14. [[batch-inference-and-batch-apis]] - cheaper processing when results can wait.
15. [[llm-gateways-and-routing]] - one entry point for many providers, with fallbacks.
16. [[semantic-caching]] - reuse answers to similar queries.
17. [[llm-observability]] - trace, log and monitor calls in production.
18. [[opentelemetry-genai-tracing]] - standard trace attributes for LLMs and agents.
19. [[ab-testing-and-canary-for-models]] - roll out new models and prompts safely.
20. [[ai-security-privacy-compliance]] - data handling, PII and access control basics.
21. [[ai-regulation-checklist]] - EU AI Act obligations and the ISO/IEC 42001 management standard.
22. [[mlops-lifecycle]] - tie it together into a repeatable lifecycle.

## LLM scientist

Training and adapting LLMs yourself: data, SFT, preference tuning, evaluation, compression and
model editing. Follows the "LLM Scientist" track of https://github.com/mlabonne/llm-course.

1. [[transformers-and-attention]] - the decoder-only architecture being trained.
2. [[tokenization]] - vocabularies, special tokens and why they matter for chat formats.
3. [[decoding-and-sampling]] - greedy, beam, temperature and top-p: how outputs are produced.
4. [[pretraining-and-scaling-laws]] - data, compute and model size at pretraining scale.
5. [[distributed-training]] - DDP, FSDP and DeepSpeed ZeRO across GPUs.
6. [[efficient-training-mixed-precision]] - bf16, gradient checkpointing and memory savings.
7. [[synthetic-data-for-llm-training]] - generate and enhance instruction and preference data.
8. [[post-training-datasets-and-chat-templates]] - formats, chat templates, loss masks, dedup, decontamination.
9. [[fine-tuning-and-peft]] - SFT with full fine-tuning, LoRA and QLoRA.
10. [[unsloth-finetuning]] - fast single-GPU SFT and GRPO, export to GGUF.
11. [[rlhf-and-preference-optimization]] - reward models, PPO, DPO and their trade-offs.
12. [[reward-modeling-and-verifiers]] - scoring outputs for RL and rejection sampling.
13. [[rl-with-verifiable-rewards]] - GRPO-style RL for reasoning models.
14. [[reading-ai-benchmarks]] - what benchmark scores measure and how contamination skews them.
15. [[llm-evaluation]] - golden sets, LLM-as-judge and regression evals.
16. [[quantization]] - GGUF, GPTQ, AWQ and bitsandbytes for cheaper inference.
17. [[model-merging]] - SLERP, TIES and DARE without further training.
18. [[mechanistic-interpretability]] - probes, patching and features inside the model.
19. [[refusal-direction-and-abliteration]] - find and edit a behaviour direction in the weights.
20. [[test-time-compute-scaling]] - spend more inference compute for better reasoning.

## Internals from scratch

Small NumPy or pure-Python builds of the pieces inside a modern LLM, in dependency order.

1. [[neural-network-from-scratch-numpy]] - forward pass, backprop and training in a few dozen lines.
2. [[autograd-engine-from-scratch]] - how frameworks compute gradients automatically.
3. [[optimizers-from-scratch-numpy]] - SGD, momentum and Adam side by side.
4. [[bpe-tokenizer-from-scratch]] - train and run the tokenizer LLMs use.
5. [[word2vec-skip-gram-from-scratch-numpy]] - the first widely used learned embeddings.
6. [[self-attention-from-scratch-numpy]] - scaled dot-product and multi-head attention.
7. [[rope-rotary-embeddings-from-scratch-numpy]] - how modern LLMs encode position.
8. [[transformer-block-from-scratch-numpy]] - assemble attention, MLP, norms and residuals.
9. [[kv-cache-from-scratch-numpy]] - why generation is fast after the first token.
10. [[decoding-strategies-from-scratch-numpy]] - greedy, beam, temperature and top-p sampling.
11. [[speculative-decoding-from-scratch-numpy]] - faster generation with a draft model.
12. [[lora-from-scratch-numpy]] - low-rank fine-tuning and merging.
13. [[int8-quantization-from-scratch-numpy]] - the arithmetic behind quantized weights.
14. [[dpo-loss-from-scratch-numpy]] - preference tuning without a reward model.
15. [[grpo-from-scratch-numpy]] - the RL objective used for reasoning models.
