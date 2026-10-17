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
| Extract fields from text or documents | LLM with schema-constrained output; OCR/parsing first if scanned | [[structured-output]], [[document-parsing]], [[ocr]], [[document-processing]] |
| Classify text (intent, sentiment, routing, spam) | Few-shot LLM to start; embeddings + logistic regression or a small fine-tuned model at volume | [[nlp-classic-tasks]], [[embeddings]], [[classic-ml-scikit-learn]], [[small-language-models]] |
| Answer questions over my documents | RAG: parse, chunk, embed, retrieve, rerank, cite | [[rag-basics]], [[advanced-rag]], [[vector-databases]], [[embedding-models]], [[rerankers]] |
| Answer questions over my database | Text-to-SQL with a read-only role and schema context | [[text-to-sql]], [[data-analytics]] |
| Automate a multi-step workflow / act on systems | Tool-calling agent, human approval for side effects | [[agents]], [[tool-calling]], [[model-context-protocol]], [[agent-frameworks]], [[guardrails-and-safety]] |
| Write or fix code | Coding agent with tests as the feedback loop | [[coding-agents]], [[software-engineering]] |
| Answer with current facts from the web, cite sources, run deep research | Hosted web search tool or your own search API, fetched pages treated as untrusted data | [[web-search-and-grounding]], [[prompt-injection]], [[rag-basics]] |
| Test whether an agent completes tasks reliably | Replayable sandbox tasks, outcome check plus trajectory and budget scoring | [[agent-evaluation]], [[llm-evaluation]] |
| Judge a vendor's benchmark claim or a leaderboard rank | Read what the benchmark measures, check saturation and contamination, then run your own eval set | [[reading-ai-benchmarks]], [[model-selection]] |
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
| Recommend items / personalise a feed | Embedding retrieval + ranker; popularity baseline first | [[recommender-systems]], [[embeddings]], [[ecommerce-retail]] |
| Find items bought together / bundles | Frequent itemsets and association rules, ranked by lift | [[association-rules-market-basket]], [[recommender-systems]] |
| Keep a model current as data changes | Drift monitoring + scheduled retrain; online learning if labels arrive fast | [[online-learning-and-concept-drift]], [[mlops-lifecycle]] |
| Detect anomalies / fraud / failures | Unsupervised or semi-supervised detectors plus rules | [[anomaly-detection]], [[finance]], [[manufacturing-iot]], [[security-defensive]] |
| Generate or edit images | Diffusion model API or local; check licence and provenance | [[image-generation-models]], [[diffusion-models]], [[model-licenses]] |
| Understand or train the objective behind current image/video generators, cut sampling steps | Flow matching with a straight (rectified) path; reflow or distillation for few-step sampling | [[flow-matching-and-rectified-flow]], [[diffusion-models]], [[ddpm-diffusion-from-scratch-numpy]] |
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
| Pick a chunk size and strategy for a RAG index | Recursive chunks as baseline; semantic, late or contextual chunking when evals show gains | [[chunking-strategies]], [[rag-basics]], [[recursive-text-splitter-from-scratch]] |
| Answer corpus-wide questions ("main themes across all reports") | GraphRAG / knowledge-graph retrieval | [[graphrag-and-knowledge-graphs]], [[advanced-rag]] |
| Answer over charts, tables and page images | Multimodal RAG on page images or parsed tables | [[multimodal-rag]], [[document-layout-and-table-extraction]], [[vision-language-models]] |
| Extract tables and layout from PDFs | Layout model plus table-structure model; VLM fallback | [[document-layout-and-table-extraction]], [[document-parsing]], [[ocr]] |
| Gate prompt and model changes on evals in CI | LLM-as-judge calibrated against human labels, eval-driven development | [[llm-as-judge-and-eval-driven-development]], [[llm-evaluation]] |
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
  install: [[vllm-local-setup]], [[docker-for-ml]].
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
inference [[gpu-cuda-setup]], [[huggingface-transformers]],
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
