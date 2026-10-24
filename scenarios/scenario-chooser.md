---
title: Scenario chooser (entry point)
category: scenarios
tags: [routing, decision-tree, chooser, rag, fine-tuning, local-vs-api]
use_cases:
  - "I have an AI task and do not know which approach or which KB files to read"
  - "decide between prompting, RAG and fine-tuning for my project"
  - "decide between a hosted API and a local open-weights model"
  - "map a task type (extraction, classification, forecasting, vision, speech) to the right files"
status: stable
last_verified: 2026-10-17
sources:
  - https://docs.anthropic.com/en/docs/build-with-claude/overview
  - https://platform.openai.com/docs/guides/optimizing-llm-accuracy
---

# Scenario chooser (entry point)

## Summary
Start here when you have a task but not an approach. Section 1 routes by task
type, section 2 settles the cross-cutting choices (prompt vs RAG vs fine-tune,
API vs local, LLM vs classic ML), section 3 routes by industry and section 4
routes the stages of building and training your own model. Each row names the
KB files to read in full next.

## Key concepts
- Cheapest thing that meets the quality bar wins: prompt first, then RAG or
  tools, then fine-tuning, then training from scratch. Move down only when an
  eval ([[llm-evaluation]]) shows the cheaper rung failing.
- Not every problem needs an LLM. Tabular numbers, forecasts and anomalies are
  usually better served by classic ML ([[gradient-boosting-tabular]],
  [[time-series-forecasting]], [[anomaly-detection]]).
- Build the eval set before the system. Without it you cannot choose between
  any of the options below.
- Knowledge problems (facts the model lacks) -> retrieval. Behaviour problems
  (style, format, a narrow skill) -> prompting, then fine-tuning.

## When to use / scenarios

### 1. By task type
| I want to ... | Recommended approach | Read |
|---|---|---|
| Generate or rewrite text (emails, copy, summaries) | Prompted LLM, few-shot examples, evals on a rubric | [[prompt-engineering]], [[model-selection]], [[marketing-content]] |
| Summarize a document longer than the context window, or thousands of tickets into a digest | One long-context call if it fits; map-reduce with section citations otherwise, checked for unsupported claims | [[long-document-summarization]], [[long-context]], [[chunking-strategies]] |
| Extract fields from text or documents | LLM with schema-constrained output; OCR/parsing first if scanned | [[structured-output]], [[document-parsing]], [[ocr]], [[document-processing]] |
| Classify text (intent, sentiment, routing, spam) | Few-shot LLM to start; embeddings + logistic regression or a small fine-tuned model at volume | [[nlp-classic-tasks]], [[embeddings]], [[classic-ml-scikit-learn]], [[small-language-models]] |
| Answer questions over my documents | RAG: parse, chunk, embed, retrieve, rerank, cite | [[rag-basics]], [[advanced-rag]], [[vector-databases]], [[embedding-models]], [[rerankers]] |
| Answer from internal documents that users have different rights to, or from many customers' data in one system | Tenant and ACL metadata on every chunk, enforced as a pre-filter (or Postgres RLS) from the server-side identity; caches and traces scoped the same way | [[permission-aware-rag-and-multi-tenancy]], [[vector-databases]], [[rag-basics]] |
| Self-host embeddings and a reranker for private or high-volume search | TEI or vLLM pooling server, token-budget batching, vector cache keyed by model revision | [[embedding-and-reranker-serving]], [[embedding-models]], [[rerankers]] |
| Retrieval misses domain jargon even with hybrid search and a reranker | Fine-tune an open embedder on query-passage pairs with in-batch and hard negatives, judged by recall@k on held-out documents, then re-embed | [[embedding-fine-tuning]], [[embedding-models]], [[advanced-rag]] |
| Answer questions over my database | Text-to-SQL with a read-only role and schema context | [[text-to-sql]], [[data-analytics]] |
| Automate a multi-step workflow / act on systems | Tool-calling agent, human approval for side effects | [[agents]], [[tool-calling]], [[model-context-protocol]], [[agent-frameworks]], [[guardrails-and-safety]] |
| Write or fix code | Coding agent with tests as the feedback loop | [[coding-agents]], [[software-engineering]] |
| Answer with current facts from the web, cite sources, run deep research | Hosted web search tool or your own search API, fetched pages treated as untrusted data | [[web-search-and-grounding]], [[prompt-injection]], [[rag-basics]] |
| Test whether an agent completes tasks reliably | Replayable sandbox tasks, outcome check plus trajectory and budget scoring | [[agent-evaluation]], [[llm-evaluation]] |
| Judge a vendor's benchmark claim or a leaderboard rank | Read what the benchmark measures, check saturation and contamination, then run your own eval set | [[reading-ai-benchmarks]], [[model-selection]] |
| Check a fine-tuned, merged or quantized open model kept its general ability | Same fixed benchmark suite (lm-evaluation-harness or lighteval) on base and new checkpoint with identical settings, differences judged against standard errors | [[lm-evaluation-harness-and-lighteval]], [[quantizing-llms-for-serving]], [[llm-evaluation]] |
| Understand images or screenshots | Vision-language model; classic CV if the label set is fixed and volume is high | [[vision-language-models]], [[multimodal-models]], [[image-classification]], [[object-detection]], [[segmentation]] |
| Read text from images / scans | OCR engine or VLM; compare on your documents | [[ocr]], [[document-parsing]] |
| Analyse video or camera streams | Detection + tracking models, edge inference | [[video-analytics]], [[object-detection]], [[edge-on-device]], [[manufacturing-iot]] |
| Transcribe or caption audio | Speech-to-text, diarization if multi-speaker | [[speech-to-text]], [[speaker-diarization]] |
| Build a voice assistant / phone bot | STT + LLM + TTS pipeline or a realtime model | [[voice-agents]], [[text-to-speech]], [[speech-to-text]] |
| Forecast a number over time (demand, load) | Time-series models, gradient boosting with lag features | [[time-series-forecasting]], [[gradient-boosting-tabular]] |
| Forecast many series with short histories, no per-series fitting | Zero-shot time-series foundation model (Chronos, TimesFM, Moirai), backtested against seasonal naive and ETS; check the weight licence | [[time-series-foundation-models]], [[time-series-forecasting]], [[model-licenses]] |
| Predict a label/value from a table (churn, credit, price) | Gradient boosting baseline; LLM only for text columns | [[gradient-boosting-tabular]], [[classic-ml-scikit-learn]], [[experiment-tracking]] |
| Predict when something happens (churn date, failure, default) | Survival models that handle censored rows | [[survival-analysis]], [[gradient-boosting-tabular]] |
| Measure whether an action caused an outcome / target an offer | A/B test first; causal ML or uplift models on the data | [[statistics-and-ab-testing]], [[causal-inference-and-uplift]], [[model-evaluation-and-metrics]] |
| Tell whether a metric change or model gain is real | Power analysis up front, then a test with a confidence interval | [[statistics-and-ab-testing]] |
| Predict with honest uncertainty on small data | Gaussian process or Bayesian model; conformal intervals otherwise | [[bayesian-and-gaussian-processes]] |
| Put guaranteed intervals / label sets around any model, escalate unsure cases | Split conformal prediction on a held-out calibration set | [[conformal-prediction-and-uncertainty]], [[model-evaluation-and-metrics]] |
| Train across organizations or devices without pooling data | Federated averaging; add differential privacy for a real privacy claim | [[federated-learning-and-differential-privacy]], [[ai-security-privacy-compliance]] |
| Recognize or match items from a few examples; add classes without retraining | Pretrained embeddings + nearest prototype; metric learning with many training classes | [[metric-learning-and-few-shot]], [[embeddings]], [[face-and-pose]] |
| Model very long sequences (audio, genomics, logs) or cut KV-cache cost | State space / hybrid SSM-attention models | [[state-space-models]], [[long-context]] |
| Detect hidden regimes or smooth/fuse noisy sensor readings over time | HMM for discrete states, Kalman filter for continuous state | [[hidden-markov-models-and-kalman-filters]], [[anomaly-detection]] |
| Order search results, listings or candidates by relevance | LambdaMART (LightGBM/XGBoost ranker) on query-grouped data | [[learning-to-rank]], [[rerankers]], [[recommender-systems]] |
| Tag one input with several labels, or predict several targets from one model | Per-label sigmoid outputs with tuned thresholds; shared encoder with task heads | [[multi-label-and-multi-task-learning]], [[model-evaluation-and-metrics]] |
| Find and fix mislabeled training data | Confident learning on out-of-fold probabilities, then human review | [[label-noise-and-data-cleaning]], [[data-labeling-and-synthetic-data]] |
| Make a model smaller or cheaper to serve | Quantize first, then distil into a small student | [[quantization]], [[knowledge-distillation-and-compression]], [[edge-on-device]] |
| Run a small model inside the user's browser (private, offline, no server cost) | WebLLM for chat on WebGPU, Transformers.js for embeddings and classifiers with WASM fallback, server API when the device does not qualify | [[in-browser-llm-inference]], [[small-language-models]], [[edge-on-device]] |
| Cut input tokens of RAG or transcript prompts whose context changes every call | Lower top-k and rerank first; then extractive or LLMLingua-2 compression of the context only, checked on end-task accuracy | [[prompt-compression]], [[prompt-caching-and-cost]], [[rag-evaluation]] |
| Fit a bigger open model or more concurrent users on our GPUs | FP8 (or W4A16 when memory-bound) checkpoint from llm-compressor plus FP8 KV cache in vLLM, checked on our eval set | [[quantizing-llms-for-serving]], [[quantization]], [[inference-servers-vllm]] |
| Serve a fine-tune per customer or per task without a GPU each | One base model with many LoRA adapters (vLLM `--enable-lora`, LoRAX), slots sized from active tenants | [[multi-lora-adapter-serving]], [[fine-tuning-and-peft]], [[inference-servers-vllm]] |
| Recommend items / personalise a feed | Embedding retrieval + ranker; popularity baseline first | [[recommender-systems]], [[embeddings]], [[ecommerce-retail]] |
| Find items bought together / bundles | Frequent itemsets and association rules, ranked by lift | [[association-rules-market-basket]], [[recommender-systems]] |
| Keep a model current as data changes | Drift monitoring + scheduled retrain; online learning if labels arrive fast | [[online-learning-and-concept-drift]], [[mlops-lifecycle]] |
| Detect anomalies / fraud / failures | Unsupervised or semi-supervised detectors plus rules | [[anomaly-detection]], [[finance]], [[manufacturing-iot]], [[security-defensive]] |
| Generate or edit images | Diffusion model API or local; check licence and provenance | [[image-generation-models]], [[diffusion-models]], [[model-licenses]] |
| Understand or train the objective behind current image/video generators, cut sampling steps | Flow matching with a straight (rectified) path; reflow or distillation for few-step sampling | [[flow-matching-and-rectified-flow]], [[diffusion-models]], [[ddpm-diffusion-from-scratch-numpy]] |
| Prepare an SFT or preference dataset for fine-tuning an LLM | Messages format, the model's chat template, assistant-only loss, MinHash dedup, decontaminate against evals | [[post-training-datasets-and-chat-templates]], [[synthetic-data-for-llm-training]], [[fine-tuning-and-peft]] |
| Find or edit a behaviour direction (refusal, style) in an open-weight model | Difference-of-means direction; ablate or add it at inference, or orthogonalize the weights | [[refusal-direction-and-abliteration]], [[mechanistic-interpretability]] |
| Label data / create training data | LLM-assisted labelling with human review | [[data-labeling-and-synthetic-data]] |
| Train with few labels and many unlabelled examples | Pretrained embeddings, active learning to pick labels, label spreading / pseudo-labels | [[semi-supervised-and-active-learning]], [[data-labeling-and-synthetic-data]] |
| Build a personal assistant with memory | Agent + memory store + calendar/mail tools | [[personal-assistants]], [[agent-memory]] |
| Drive a browser or desktop app that has no API | Computer-use agent in a sandbox, approval for side effects | [[computer-use-agents]], [[agents]] |
| Turn designs (Figma) into UI code with Claude | Design MCP server or plugin, then check the result in a browser | [[design-tools-for-claude]], [[coding-agents]] |
| Group customers or items without labels (segments) | Scale features, then k-means / HDBSCAN / Gaussian mixture; inspect in 2D | [[clustering]], [[gaussian-mixture-models-and-em]], [[dimensionality-reduction]], [[distance-metrics-and-similarity]] |
| Find the themes in a pile of text | Topic model: BERTopic on embeddings, LDA/NMF on bag-of-words | [[topic-modeling]], [[word-embeddings-word2vec]], [[embeddings]] |
| Pick which variant to show while still learning (offers, layouts) | Multi-armed or contextual bandit | [[multi-armed-bandits]], [[statistics-and-ab-testing]] |
| Learn a behaviour by trial and error (control, games) | Reinforcement learning in a simulator | [[reinforcement-learning]] |
| Predict ordered grades, counts or a range instead of one number | Ordinal regression; Poisson/Tweedie GLM for counts; quantile models for intervals | [[ordinal-regression]], [[generalized-linear-models]], [[quantile-regression]] |
| Flag inputs unlike the training data and abstain on them | OOD score checked on a held-out shift set; uncertainty-aware models | [[out-of-distribution-detection]], [[bayesian-deep-learning-and-uncertainty]] |
| Optimise something you cannot differentiate (simulator, config) | Evolutionary or other black-box optimiser | [[black-box-and-evolutionary-optimization]], [[hyperparameter-tuning]] |
| Build a fast surrogate for a physical simulation | Physics-informed network or neural operator | [[physics-informed-neural-networks]] |
| Learn from relations in a graph (fraud rings, molecules, links) | Graph neural network | [[graph-neural-networks]] |
| Label AI-generated images, audio or video, or check whether media is AI-made | C2PA manifests plus invisible watermarks; detectors only as a weak signal | [[ai-watermarking-and-content-provenance]], [[image-generation-models]] |
| Pick between a fixed pipeline and an agent | Workflow (chain, router, parallel) unless the path is genuinely dynamic | [[workflows-vs-agents]], [[agents]] |
| Keep a long agent run on track as its context fills | Compaction, notes files, sub-agents, just-in-time retrieval | [[context-engineering]], [[long-running-agents]], [[agent-memory]] |
| Make an agent survive crashes and wait for approvals | Durable execution with checkpoints, budgets and human approval steps | [[long-running-agents]], [[agents]] |
| Package a team procedure an agent loads on demand | Agent skill (SKILL.md) with progressive disclosure | [[agent-skills]], [[coding-agents]], [[model-context-protocol]] |
| Let agents from different vendors or frameworks hand tasks to each other | A2A protocol between agents; MCP for agent-to-tool | [[a2a-protocol]], [[multi-agent-systems]], [[model-context-protocol]] |
| Stream answers to a UI or run many LLM calls concurrently | SSE streaming and async clients with bounded concurrency | [[streaming-and-async-llm-apps]], [[serving-with-fastapi]] |
| Find how much traffic a model server handles within a latency SLO | Open-loop rate sweep measuring TTFT, TPOT and goodput on your prompt lengths | [[llm-load-testing-and-benchmarking]], [[inference-servers-vllm]], [[cost-and-latency]] |
| Pick a chunk size and strategy for a RAG index | Recursive chunks as baseline; semantic, late or contextual chunking when evals show gains | [[chunking-strategies]], [[rag-basics]], [[recursive-text-splitter-from-scratch]] |
| Answer corpus-wide questions ("main themes across all reports") | GraphRAG / knowledge-graph retrieval | [[graphrag-and-knowledge-graphs]], [[advanced-rag]] |
| Answer over charts, tables and page images | Multimodal RAG on page images or parsed tables | [[multimodal-rag]], [[document-layout-and-table-extraction]], [[vision-language-models]] |
| Extract tables and layout from PDFs | Layout model plus table-structure model; VLM fallback | [[document-layout-and-table-extraction]], [[document-parsing]], [[ocr]] |
| Gate prompt and model changes on evals in CI | LLM-as-judge calibrated against human labels, eval-driven development | [[llm-as-judge-and-eval-driven-development]], [[llm-evaluation]] |
| Write fast CI tests for code that calls an LLM, without paying for or depending on the API | Fake LLM client at your own interface for parsing, retries, fallbacks and tool dispatch; replayed cassettes for real payloads; live smoke tests and evals on a schedule | [[testing-llm-applications]], [[structured-output]], [[llm-evaluation]] |
| Change prompts without a code deploy, with review, rollback and a record of which version answered | Prompt registry (or prompts in Git) with immutable versions, a `production` label moved only after an eval gate, version logged on every trace | [[prompt-management-and-versioning]], [[llm-as-judge-and-eval-driven-development]], [[llm-observability]] |
| Stop bulk LLM jobs failing on 429 rate limits or 529/5xx overloads | Retry only retryable errors, capped exponential backoff with jitter and `retry-after`, client-side token bucket shared by workers; batch API if results can wait | [[llm-api-rate-limits-and-retries]], [[batch-inference-and-batch-apis]], [[llm-gateways-and-routing]] |
| Send customer text to an LLM API without exposing names, emails or card numbers | Detect PII (patterns + checksums + NER, e.g. Presidio), replace with typed placeholders, restore in the answer server-side; measure recall on your data | [[pii-redaction-and-pseudonymization]], [[ai-security-privacy-compliance]], [[guardrails-and-safety]] |
| Find out whether wrong RAG answers come from retrieval or generation | Golden questions with relevant-passage labels; recall@k, MRR and nDCG for retrieval, claim-level faithfulness and correctness for answers | [[rag-evaluation]], [[llm-evaluation]], [[rag-basics]], [[advanced-rag]] |
| Get OpenAI-style tool calls from a self-hosted open model | vLLM `--enable-auto-tool-choice` with the family's `--tool-call-parser` and a tool-aware chat template (llama-server `--jinja`, Ollama `tools`), then a tool-use eval | [[tool-calling-with-self-hosted-models]], [[tool-calling]], [[inference-servers-vllm]], [[ollama]] |
| Cut inter-token latency of a self-hosted LLM without changing outputs | Speculative decoding: MTP or EAGLE head first, n-gram for copy-heavy outputs; check acceptance length at production load | [[speculative-decoding-in-serving]], [[speculative-decoding-from-scratch-numpy]], [[llm-load-testing-and-benchmarking]] |
| Stop long prompts from spiking inter-token latency, or keep long chat/agent KV cache past GPU memory | Chunked prefill and prefix caching first; then KV offloading to CPU/disk (vLLM OffloadingConnector, LMCache) and, for strict TTFT/ITL SLOs on fast interconnects, prefill/decode disaggregation (NixlConnector, llm-d, Dynamo); it does not raise throughput | [[disaggregated-prefill-and-kv-cache-offloading]], [[inference-servers-vllm]], [[llm-load-testing-and-benchmarking]] |
| Flag unsafe prompts and replies, or jailbreak attempts, with a classifier model | Content classifier (Llama Guard, ShieldGemma, moderation API) on input and output, attack detector (Prompt Guard) on untrusted text, or a policy-following judge (Granite Guardian, gpt-oss-safeguard) for your own rules; thresholds set on your labelled traffic | [[safety-classifier-models]], [[guardrails-and-safety]], [[prompt-injection]] |
| Run PyTorch, vLLM, llama.cpp or Ollama on an AMD GPU | ROCm with the tool's ROCm build (vLLM ROCm wheels or image, llama.cpp `GGML_HIP`); check the GPU's gfx target; override or Vulkan for unlisted cards | [[amd-rocm-setup]], [[gpu-cuda-setup]], [[vllm-local-setup]] |
| Run PyTorch or a local LLM on an Intel Arc GPU, Core Ultra laptop GPU or NPU | PyTorch XPU wheels (`device="xpu"`) for development; OpenVINO GenAI on GPU/NPU for local inference; vLLM XPU image, llama.cpp SYCL or Ollama (Vulkan) for serving; not the archived IPEX/IPEX-LLM | [[intel-gpu-and-openvino-setup]], [[amd-rocm-setup]], [[edge-on-device]] |
| Give a team a private ChatGPT-style web UI over self-hosted or mixed models | Open WebUI in Docker with a persisted data volume, connected to Ollama, vLLM or a gateway; explicit `WEBUI_SECRET_KEY`, sign-up off or pending | [[open-webui]], [[ollama]], [[llm-gateways-and-routing]] |
| New LLM replicas take minutes to start, or scale-to-zero wakes too slowly | Measure each start-up phase; shared HF cache with `HF_HUB_OFFLINE=1`, stream weights from object storage (vLLM `runai_streamer`), persist the compile cache, sleep idle engines instead of stopping them | [[model-weight-loading-and-cold-starts]], [[kubernetes-for-llm-serving]], [[inference-servers-vllm]] |
| Dashboard and alerts for self-hosted LLM servers, or find why latency rose | vLLM/SGLang Prometheus metrics (queue, KV-cache usage, preemptions, TTFT, inter-token latency) plus DCGM exporter GPU metrics and XID alerts | [[inference-server-and-gpu-metrics]], [[inference-servers-vllm]], [[kubernetes-for-llm-serving]] |
| Serve a custom model or multi-model pipeline (embedder, classifier, OCR, LLM) with server-side batching and per-step scaling | Ray Serve when already on Ray or multi-node; BentoML for packaged Docker services; LitServe for the thinnest Python layer; the LLM engine's own server for a single LLM | [[ray-serve-bentoml-and-litserve]], [[serving-with-fastapi]], [[onnx-and-triton-inference-server]] |
| Switch a live RAG index to a new or fine-tuned embedding model without downtime | Second collection or vector column, dual writes, idempotent backfill, quality gate on labelled queries, atomic alias swap, old index kept for rollback | [[embedding-model-migration-and-reindexing]], [[vector-databases]], [[rag-evaluation]] |
| Use thumbs up/down and expert review to find and fix bad answers in production | Feedback as scores on traces, response-rate-aware comparison with intervals, stratified review queue, judge calibrated by kappa, reviewed failures exported as eval cases | [[user-feedback-and-human-review]], [[llm-observability]], [[llm-as-judge-and-eval-driven-development]] |
| Tune prompts of a multi-step pipeline automatically | DSPy-style prompt optimization against a metric | [[dspy-and-prompt-optimization]], [[prompt-engineering]] |
| Search a photo library by text or find near-duplicate images | CLIP-style embeddings in a vector index | [[image-retrieval-clip-search]], [[vector-databases]], [[embeddings]] |
| Detect objects named in plain text, no training data | Open-vocabulary detector (Grounding DINO, OWLv2, YOLO-World) | [[open-vocabulary-detection]], [[object-detection]] |
| Get a pixel mask from a click or a box, in images or video | Segment Anything (SAM 2 / SAM 3) | [[segment-anything-sam2]], [[segmentation]] |
| Estimate depth from one image | Monocular depth model | [[depth-estimation]], [[edge-on-device]] |
| Reconstruct a 3D scene from photos or video | 3D Gaussian Splatting or NeRF | [[gaussian-splatting-and-nerf]], [[3d-generation-models]] |
| Edit, inpaint or pose-control generated images | Inpainting and ControlNet on a diffusion model | [[image-editing-and-controlnet]], [[comfyui]], [[diffusion-models]] |
| Generate 3D assets, music, sound or video | Text/image-to-3D, music and video generation models; check licences | [[3d-generation-models]], [[music-and-audio-generation-models]], [[video-generation-models]], [[model-licenses]] |
| Simulate environments or generate training video for robots and driving | World models | [[world-models]], [[vision-language-action-models]] |
| Analyse music (tempo, beats, key, tags) | MIR features and pretrained audio models | [[music-information-retrieval]], [[audio-classification-and-sound-events]] |
| Translate speech into subtitles or another spoken language | Speech translation model or STT + MT cascade | [[speech-translation]], [[speech-to-text]] |
| Wake a device on a spoken keyword | Small on-device keyword-spotting model | [[wake-word-detection]], [[edge-on-device]] |
| Choose a native speech-to-speech model vs an STT-LLM-TTS cascade | Compare latency, control and cost on your calls | [[speech-to-speech-realtime-models]], [[voice-agents]] |
| Get a strong tabular baseline on a small dataset with no tuning | Tabular foundation model (TabPFN, TabICL), then compare to boosting | [[tabular-foundation-models]], [[gradient-boosting-tabular]] |
| Speed up a slow pandas feature pipeline | Polars or DuckDB | [[polars-and-duckdb-for-ml]], [[feature-engineering]] |
| Score events in real time with fresh features | Streaming pipeline plus an online feature store | [[streaming-ml-pipelines]], [[feature-stores]], [[anomaly-detection]] |
| Serve the same features in training and in production | Feature store with point-in-time joins | [[feature-stores]], [[data-leakage-and-validation-splits]] |

### 2. Cross-cutting decisions
- **Prompt vs RAG vs fine-tune.** Missing or changing facts, need citations ->
  [[rag-basics]]. Consistent format or tone, narrow task, latency/cost at high
  volume -> [[fine-tuning-and-peft]] (after prompting and [[structured-output]]
  fail). Often combined: RAG for facts, small fine-tune for behaviour.
  No GPUs, or a checkable-answer task for reinforcement fine-tuning ->
  a provider-hosted tune: [[hosted-fine-tuning-apis]].
- **API vs local model.** API: best quality, no GPU ops, per-token cost, data
  leaves your boundary (check the provider's retention and DPA terms). Local:
  data control, fixed cost at scale, offline use, more ops work. See
  [[model-selection]], [[ollama]], [[llama-cpp-gguf]], [[inference-servers-vllm]],
  [[gpu-cloud-options]], [[ai-security-privacy-compliance]]. Same hosted models
  inside your cloud account (residency, private networking, existing billing):
  [[managed-cloud-ai-platforms]].
- **Which model family.** [[anthropic-claude]], [[openai-gpt]], [[google-gemini]]
  (hosted); [[meta-llama]], [[mistral]], [[qwen]], [[deepseek]] (open weights,
  check [[model-licenses]]). Hard multi-step problems: [[reasoning-models]].
- **One agent vs workflow.** If the steps are known, write a fixed pipeline
  with LLM calls; use an [[agents]] loop only when the path is genuinely
  dynamic. Multiple agents: [[multi-agent-systems]] only when one cannot cope.
- **Long document vs retrieval.** Few documents that fit the window ->
  stuff them in ([[long-context]], [[prompt-caching-and-cost]]); large or
  changing corpus -> RAG.
- **Cost and latency.** [[cost-and-latency]], [[prompt-caching-and-cost]],
  [[quantization]], [[small-language-models]]. Several providers or models
  behind one endpoint, with fallbacks: [[llm-gateways-and-routing]].
- **Which serving stack.** [[inference-servers-vllm]] by default; compare
  [[sglang-and-tensorrt-llm]] for throughput, [[onnx-and-triton-inference-server]]
  for non-LLM models, [[kubernetes-for-llm-serving]] on a cluster. Local
  install: [[vllm-local-setup]], [[docker-for-ml]]. Size replicas and compare
  stacks with a load test: [[llm-load-testing-and-benchmarking]].
- **Shipping a model or prompt change.** Version it in a registry
  ([[model-registry-and-versioning]]), roll out with shadow or canary traffic
  ([[ab-testing-and-canary-for-models]]), trace it with
  [[opentelemetry-genai-tracing]].
- **Which open-weight model.** Start at [[open-weight-model-landscape]], then
  [[google-gemma]], [[microsoft-phi]], [[chinese-open-weight-models]],
  [[code-models]] (IDE completion). Other hosted APIs: [[xai-grok]],
  [[cohere-command]] (enterprise RAG, embed, rerank).
- **Running models on your own machine.** Desktop apps [[lm-studio-and-jan]];
  Macs [[mlx-on-apple-silicon]]; images and video [[comfyui]]; free GPUs
  [[jupyter-colab-and-kaggle]].
- **Trust.** Hallucination risk -> [[hallucination-and-grounding]]; untrusted
  input -> [[prompt-injection]], [[guardrails-and-safety]].

### 3. By industry
[[customer-support]], [[document-processing]], [[healthcare]], [[finance]],
[[ecommerce-retail]], [[education]], [[legal]], [[manufacturing-iot]],
[[software-engineering]], [[security-defensive]], [[marketing-content]],
[[data-analytics]], [[personal-assistants]], [[hr-and-recruiting]],
[[insurance]], [[real-estate]], [[telecom]], [[agriculture]],
[[automotive-robotics]], [[logistics-and-supply-chain]],
[[government-public-sector]], [[media-and-gaming]], [[cybersecurity-soc]].
By constraint: [[low-budget-ai]], [[startup-mvp-with-ai]].

### 4. Building and training your own model
| Stage | Read |
|---|---|
| Learn the foundations | [[ml-fundamentals]], [[math-for-machine-learning]], [[information-theory-for-ml]], [[maximum-likelihood-and-map-estimation]], [[gradient-descent]], [[perceptron-and-linear-separability]], [[universal-approximation-and-depth-vs-width]] |
| Explore and prepare tabular data | [[exploratory-data-analysis]], [[missing-data-and-imputation]], [[feature-engineering]], [[categorical-encoding]], [[feature-scaling-and-normalization]], [[feature-selection]], [[curse-of-dimensionality]] |
| Validate without leakage, pick the metric | [[data-leakage-and-validation-splits]], [[model-selection-and-comparison]], [[bias-variance-and-learning-curves]], [[regression-metrics-and-residual-analysis]], [[probability-calibration]], [[imbalanced-data]], [[multiclass-classification-strategies]] |
| Pick and tune a tabular model | [[linear-models]], [[generalized-additive-models-and-splines]], [[decision-trees-and-random-forests]], [[ensemble-methods]], [[svm-knn-naive-bayes]], [[kernel-methods-and-density-estimation]], [[probabilistic-graphical-models]], [[deep-learning-for-tabular-data]], [[hyperparameter-tuning]], [[automl]] |
| Explain a model's predictions | [[model-interpretability]], [[saliency-maps-and-neural-attribution]], [[mechanistic-interpretability]] |
| Pick a deep-learning framework | [[pytorch-basics]], [[keras-and-tensorflow]], [[jax-and-flax]], [[tensor-shapes-broadcasting-and-einsum]] |
| Design a neural network | [[neural-network-fundamentals]], [[activation-functions]], [[weight-initialization]], [[normalization-layers]], [[residual-and-skip-connections]], [[cnn-and-rnn-architectures]], [[convolution-arithmetic-and-receptive-field]], [[recurrent-neural-networks-lstm-gru]], [[sequence-to-sequence-and-ctc]], [[backpropagation-and-autograd]], [[loss-functions]], [[neural-architecture-search]] |
| Train it well | [[deep-learning-training]], [[optimizers]], [[learning-rate-schedules]], [[batch-size-and-gradient-noise]], [[regularization-in-deep-learning]], [[data-augmentation]], [[data-loading-pipelines]], [[generalization-in-deep-learning]], [[loss-landscapes-and-flat-minima]] |
| Training fails (NaNs, flat loss) | [[debugging-neural-network-training]], [[vanishing-and-exploding-gradients]] |
| Train bigger or faster, save and export | [[efficient-training-mixed-precision]], [[distributed-training]], [[model-checkpointing-and-export]] |
| Reuse a pretrained model on new data | [[transfer-learning-and-domain-adaptation]], [[continual-learning]], [[meta-learning]], [[autoencoders-and-self-supervised-learning]] |
| Harden a model against attacks | [[adversarial-examples-and-robustness]] |
| Train a generative model | [[generative-adversarial-networks]], [[normalizing-flows-and-energy-based-models]] |
| Understand how LLMs work inside | [[transformers-and-attention]], [[attention-variants-and-efficient-attention]], [[positional-encodings]], [[tokenization]], [[decoding-and-sampling]], [[mixture-of-experts]], [[pretraining-and-scaling-laws]], [[rlhf-and-preference-optimization]] |
| Fine-tune an open model cheaply | [[unsloth-finetuning]], [[pytorch-lightning-and-accelerate]], [[fine-tuning-and-peft]] |
| Make training data, rewards and verifiers | [[synthetic-data-for-llm-training]], [[reward-modeling-and-verifiers]], [[rl-with-verifiable-rewards]] |
| Combine fine-tunes or spend more compute at inference | [[model-merging]], [[test-time-compute-scaling]], [[reasoning-models]] |
| Newer LLM architectures and training objectives | [[multi-token-prediction]], [[diffusion-language-models]] |
| Look inside a trained model's features | [[sparse-autoencoders-for-interpretability]], [[mechanistic-interpretability]] |

## Setup & code
Environment first: [[python-env-uv]], [[api-sdk-setup]], and for local
inference [[gpu-cuda-setup]] (AMD: [[amd-rocm-setup]]), [[huggingface-transformers]],
[[windows-wsl-setup]]. A minimal routing helper an agent could use:

```python
def route(task: dict) -> str:
    if task["data"] == "tabular" and task["goal"] in {"predict", "forecast"}:
        return "ml/gradient-boosting-tabular"      # not an LLM problem
    if task["needs_private_docs"]:
        return "llm-apps/rag-basics"
    if task["output"] == "fields":
        return "llm-apps/structured-output"
    return "llm-apps/prompt-engineering"           # default: prompt first
```

## Choosing / trade-offs
- Quality vs cost: start with the strongest model to find the ceiling, then
  step down until the eval breaks ([[model-selection]]).
- Control vs effort: local models buy privacy and fixed cost, and cost you
  GPU operations, updates and usually some quality.
- Flexibility vs reliability: agents are flexible; fixed pipelines are
  testable. Prefer the pipeline until it cannot cover the cases.

## Gotchas
- Choosing fine-tuning to inject knowledge: it teaches style, not reliable
  facts. Use retrieval.
- Skipping the baseline: a rules or regression baseline often matches an LLM
  on structured problems at a fraction of the cost.
- No eval set: every approach "looks fine" on five hand-picked examples.
- Regulated domains ([[healthcare]], [[finance]], [[legal]]) add legal
  constraints that dominate the technical choice. Read those files first.

## Related
- [[llm-evaluation]] - how to decide that a rung is good enough.
- [[mlops-lifecycle]] - running the chosen approach in production.
- [[llm-observability]] - monitoring after launch.

## References
- Anthropic, Build with Claude overview: https://docs.anthropic.com/en/docs/build-with-claude/overview
- OpenAI, Optimizing LLM accuracy (prompt vs RAG vs fine-tune): https://platform.openai.com/docs/guides/optimizing-llm-accuracy
