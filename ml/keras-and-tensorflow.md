---
title: Keras 3 and TensorFlow
category: ml
tags: [keras, tensorflow, deep-learning, fit, multi-backend, jax, pytorch, tflite, savedmodel]
use_cases:
  - "train a standard neural network quickly with model.fit() instead of a hand-written loop"
  - "maintain or migrate an existing tf.keras / TensorFlow 2 codebase"
  - "write a model once and run it on JAX, PyTorch or TensorFlow"
  - "export a model to TFLite / LiteRT for Android or a microcontroller"
status: draft
last_verified: 2026-10-04
sources:
  - https://keras.io/getting_started/
  - https://keras.io/keras_3/
  - https://github.com/keras-team/keras
  - https://www.tensorflow.org/guide
---

# Keras 3 and TensorFlow

## Summary
Keras is a high-level deep-learning API: define a model from layers, `compile()`
it with an optimizer, loss and metrics, then `fit()`, `evaluate()` and
`predict()`. Keras 3 is multi-backend: the same model runs on JAX, PyTorch or
TensorFlow, picked with the `KERAS_BACKEND` environment variable. TensorFlow
is still the backend you meet in older codebases and in the TFLite/LiteRT
mobile and edge toolchain, but new research code mostly uses PyTorch.

## Key concepts
- **Three ways to build a model.** `keras.Sequential([...])` for a plain stack;
  the functional API (`inputs -> layers -> keras.Model(inputs, outputs)`) for
  branches, multiple inputs/outputs and shared layers; subclassing
  `keras.Model` when the forward pass needs Python control flow.
- **compile / fit.** `compile(optimizer, loss, metrics)` wires training;
  `fit(x, y, validation_split=..., callbacks=[...])` runs the loop with
  batching, shuffling and validation built in.
- **Callbacks.** `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`,
  `TensorBoard` hook into the loop without editing it.
- **Backend.** Set `KERAS_BACKEND` to `jax`, `torch` or `tensorflow` *before*
  `import keras`. Built-in layers work on every backend; custom ops must use
  `keras.ops` (not `tf.*` or `torch.*`) to stay portable.
- **Saving.** `model.save("m.keras")` stores architecture, weights and
  optimizer state; `keras.models.load_model` restores it. `model.export()`
  writes a TensorFlow SavedModel for TF Serving / TFLite.
- **Data.** `fit` accepts NumPy arrays, `tf.data.Dataset`, a PyTorch
  `DataLoader` or a `keras.utils.PyDataset`, on any backend.

## When to use / scenarios
- Standard architectures (MLP, CNN, small transformers) where `fit()` saves
  the boilerplate of a training loop; teaching and quick baselines.
- Existing TensorFlow 2 / `tf.keras` projects, and pipelines that end in
  TFLite/LiteRT on Android or microcontrollers (see [[edge-on-device]]).
- Pretrained models via KerasHub (`keras_hub`) for vision and text.
- NOT for: most new research or fine-tuning of open LLMs, where the ecosystem
  (Hugging Face, vLLM, PEFT) is PyTorch-first (see [[pytorch-basics]],
  [[huggingface-transformers]]); tabular data (see [[gradient-boosting-tabular]]).

## Setup & code
Install Keras plus one backend (see [[python-env-uv]], [[gpu-cuda-setup]]):
```bash
pip install keras torch        # or: keras jax  /  keras tensorflow
```
```python
import os
os.environ["KERAS_BACKEND"] = "torch"   # must be set before importing keras
import keras
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(1000, 20)).astype("float32")
y = (X[:, 0] + X[:, 1] > 0).astype("int32")

model = keras.Sequential([
    keras.Input(shape=(20,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(2),                    # logits
])
model.compile(
    optimizer=keras.optimizers.AdamW(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
model.fit(X, y, epochs=10, batch_size=64, validation_split=0.2,
          callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
print(model.evaluate(X, y, verbose=0))
model.save("model.keras")                     # reload: keras.models.load_model("model.keras")
```

## Choosing / trade-offs
- **Keras vs raw PyTorch.** Keras is less code for standard models and gives
  backend portability; PyTorch gives full control and the largest model
  ecosystem. Keras on the `torch` backend is a middle path.
- **Which backend.** JAX is often fastest (XLA compilation, TPUs); PyTorch
  mixes with PyTorch code and models; TensorFlow is needed for SavedModel,
  TF Serving and the TFLite converter.
- **`fit()` vs a custom loop.** Override `train_step` for unusual training
  (GANs, contrastive losses) and keep callbacks; write a fully custom loop
  only when the training logic does not fit the step abstraction.

## Gotchas
- Setting `KERAS_BACKEND` after `import keras` has no effect; the backend is
  fixed at first import.
- `tf.keras` in TensorFlow 2.16+ resolves to Keras 3. Old Keras 2 code that
  relies on removed APIs needs `pip install tf_keras` and
  `TF_USE_LEGACY_KERAS=1`, or a port.
- `from_logits=True` must match the last layer: no softmax with it, softmax
  without it. A mismatch trains but gives wrong probabilities.
- Custom layers using `tf.*` ops break on JAX/PyTorch backends; use `keras.ops`.
- `validation_split` takes the *last* fraction of the data before shuffling;
  shuffle first, or pass explicit `validation_data`, when data is ordered.
- `.h5` saving is legacy; use `.keras`.

## Related
- [[pytorch-basics]] - the explicit-loop alternative and the larger model ecosystem.
- [[jax-and-flax]] - the JAX backend underneath, used directly.
- [[neural-network-fundamentals]] - what the layers and losses do.
- [[deep-learning-training]] - optimizers, regularization and callbacks explained.
- [[edge-on-device]] - TFLite/LiteRT export targets.

## References
- https://keras.io/getting_started/
- https://keras.io/keras_3/
- https://github.com/keras-team/keras
- https://www.tensorflow.org/guide
