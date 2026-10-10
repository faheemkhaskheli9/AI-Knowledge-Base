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
last_verified: 2026-10-03
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
| Understand images or screenshots | Vision-language model; classic CV if the label set is fixed and volume is high | [[vision-language-models]], [[multimodal-models]], [[image-classification]], [[object-detection]], [[segmentation]] |
| Read text from images / scans | OCR engine or VLM; compare on your documents | [[ocr]], [[document-parsing]] |
| Analyse video or camera streams | Detection + tracking models, edge inference | [[video-analytics]], [[object-detection]], [[edge-on-device]], [[manufacturing-iot]] |
| Transcribe or caption audio | Speech-to-text, diarization if multi-speaker | [[speech-to-text]], [[speaker-diarization]] |
| Build a voice assistant / phone bot | STT + LLM + TTS pipeline or a realtime model | [[voice-agents]], [[text-to-speech]], [[speech-to-text]] |
| Forecast a number over time (demand, load) | Time-series models, gradient boosting with lag features | [[time-series-forecasting]], [[gradient-boosting-tabular]] |
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

### 2. Cross-cutting decisions
- **Prompt vs RAG vs fine-tune.** Missing or changing facts, need citations ->
  [[rag-basics]]. Consistent format or tone, narrow task, latency/cost at high
  volume -> [[fine-tuning-and-peft]] (after prompting and [[structured-output]]
  fail). Often combined: RAG for facts, small fine-tune for behaviour.
- **API vs local model.** API: best quality, no GPU ops, per-token cost, data
  leaves your boundary (check the provider's retention and DPA terms). Local:
  data control, fixed cost at scale, offline use, more ops work. See
  [[model-selection]], [[ollama]], [[llama-cpp-gguf]], [[inference-servers-vllm]],
  [[gpu-cloud-options]], [[ai-security-privacy-compliance]].
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
- **Trust.** Hallucination risk -> [[hallucination-and-grounding]]; untrusted
  input -> [[prompt-injection]], [[guardrails-and-safety]].

### 3. By industry
[[customer-support]], [[document-processing]], [[healthcare]], [[finance]],
[[ecommerce-retail]], [[education]], [[legal]], [[manufacturing-iot]],
[[software-engineering]], [[security-defensive]], [[marketing-content]],
[[data-analytics]], [[personal-assistants]].

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
