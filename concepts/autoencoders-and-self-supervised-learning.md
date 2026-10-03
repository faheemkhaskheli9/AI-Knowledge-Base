---
title: Autoencoders and self-supervised learning
category: concepts
tags: [autoencoder, vae, self-supervised, contrastive-learning, simclr, masked-modeling, representation-learning, pytorch]
use_cases:
  - "learn useful features from lots of unlabelled images, signals or logs when labels are scarce"
  - "detect anomalies by how badly a model reconstructs a sample"
  - "compress high-dimensional data into a small learned code for search or clustering"
  - "understand how foundation models are pretrained without human labels"
status: draft
last_verified: 2026-10-03
sources:
  - https://www.deeplearningbook.org/contents/autoencoders.html
  - https://arxiv.org/abs/1312.6114
  - https://arxiv.org/abs/2002.05709
  - https://arxiv.org/abs/2111.06377
---

# Autoencoders and self-supervised learning

## Summary
Self-supervised learning (SSL) trains a network on a task whose labels come from the data itself: reconstruct the input, predict a masked part, or tell whether two augmented views come from the same sample. The network has to learn useful features to solve it, and those features then transfer to real tasks with few labels. **Autoencoders** are the simplest case (compress, then reconstruct); **contrastive** methods (SimCLR, CLIP) and **masked modelling** (BERT, MAE) are what modern foundation models use.

## Key concepts
- **Autoencoder (AE).** Encoder maps input to a small **latent code**; decoder reconstructs the input; trained on reconstruction loss (MSE for continuous data). The bottleneck forces it to keep only the important structure; a linear AE learns the same subspace as PCA.
- **Denoising AE.** Corrupt the input (noise, masking) and reconstruct the clean version; learns more robust features.
- **Variational AE (VAE).** The encoder outputs a distribution (mean, variance) and the loss adds a KL term that keeps latents close to a standard normal, so you can sample new data from the latent space. A generative model, now mostly superseded by [[diffusion-models]] for image quality but still used as their latent compressor.
- **Contrastive learning.** Pull embeddings of two augmentations of the same sample together and push other samples apart (InfoNCE loss). SimCLR/MoCo for images; CLIP for image-text pairs ([[multimodal-models]]). Augmentation choice defines what the features ignore.
- **Non-contrastive SSL.** BYOL, DINO and similar avoid negative pairs with a teacher-student setup; DINO-style backbones are strong general vision features.
- **Masked modelling.** Hide part of the input and predict it: masked tokens (BERT), next token (GPT, see [[pretraining-and-scaling-laws]]), masked image patches (MAE).
- **Linear probe.** Standard SSL evaluation: freeze the encoder, train a linear classifier on its features with few labels.

## When to use / scenarios
- Lots of unlabelled data, few labels: medical images, satellite imagery, industrial sensor streams, call-centre audio. Pretrain with SSL, then fine-tune or linear-probe.
- **Anomaly detection by reconstruction:** train an AE on normal machine vibrations, network traffic or transaction sequences; high reconstruction error flags unusual cases ([[anomaly-detection]]).
- Learned compression for retrieval or clustering of domain data where generic [[embeddings]] are weak.
- **Usually not from scratch:** for images and text, a public pretrained backbone (DINO-family, CLIP, BERT-family) already did the SSL on far more data. Train your own only when your domain is far from public data (radar, spectra, proprietary sensors) and you have a lot of it.

## Setup & code
```bash
pip install torch
```
```python
import torch
from torch import nn

torch.manual_seed(0)
# "Normal" data lies near a 2-D surface inside 20-D space; anomalies do not.
W = torch.randn(2, 18) * 0.5                 # fixed: defines the "normal" surface
def normal(n):
    z = torch.randn(n, 2)
    return torch.cat([z, torch.sin(z) @ W], dim=1)

train, test_ok, test_bad = normal(4000), normal(200), torch.randn(200, 20)

ae = nn.Sequential(                 # encoder 20 -> 2, decoder 2 -> 20
    nn.Linear(20, 32), nn.ReLU(), nn.Linear(32, 2),
    nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 20))
opt = torch.optim.Adam(ae.parameters(), lr=1e-2)
for epoch in range(300):
    loss = nn.functional.mse_loss(ae(train), train)
    opt.zero_grad(); loss.backward(); opt.step()

with torch.no_grad():
    err = lambda x: ((ae(x) - x) ** 2).mean(dim=1)
    threshold = err(train).quantile(0.99)       # 1% false-alarm budget
    print(f"train loss {loss.item():.4f}")
    print(f"flagged normal:    {(err(test_ok) > threshold).float().mean():.2f}")
    print(f"flagged anomalies: {(err(test_bad) > threshold).float().mean():.2f}")
```
The autoencoder learns the 2-D structure of normal data, so it flags almost none of the new normal points (the threshold allows up to ~1%) and all of the random anomalies. The same pattern (train on normal, threshold the reconstruction error at a chosen false-alarm rate) is used on real sensor or log data with a convolutional or recurrent encoder ([[cnn-and-rnn-architectures]]).

For contrastive pretraining of an image encoder, start from a library rather than writing the loss yourself: e.g. Lightly (`pip install lightly`) implements SimCLR, MoCo, BYOL and DINO on top of PyTorch.

## Choosing / trade-offs
- **AE vs PCA.** PCA is faster, deterministic and enough for roughly linear structure; an AE captures non-linear structure and handles images/sequences with conv or recurrent layers ([[dimensionality-reduction]]).
- **AE vs VAE.** Plain or denoising AE for compression and anomaly scores; VAE when you need to sample or interpolate in a smooth latent space.
- **Contrastive vs masked.** Contrastive features are immediately good for retrieval and linear probes but depend heavily on augmentations and batch size; masked modelling scales well, needs less augmentation design, and its features often need fine-tuning rather than a linear probe.
- **Pretrain yourself vs use a public backbone.** Own SSL costs GPU-days to weeks and expertise; a public backbone plus fine-tuning is the default unless the domain gap is large.

## Gotchas
- An AE with too large a bottleneck (or too powerful a decoder) learns the identity and reconstructs anomalies well too; keep the code small or add noise/masking.
- Reconstruction-error anomaly detectors flag anything unusual, including harmless new normal behaviour (a sensor recalibration); review alerts and retrain as normal drifts.
- Choose the anomaly threshold on validation data of normal cases at a target false-alarm rate, not by eye.
- Contrastive learning with small batches has few negatives and can collapse or learn weak features; use memory queues (MoCo) or non-contrastive methods.
- Augmentations encode assumptions: colour jitter teaches colour invariance, which is wrong if colour is the signal (e.g. some medical stains, ripeness grading).
- Evaluate SSL features on the downstream task, not on the pretext loss; a lower reconstruction loss does not guarantee better features.

## Related
- [[neural-network-fundamentals]] - encoders and decoders are ordinary networks.
- [[anomaly-detection]] - reconstruction error as one family of detectors.
- [[dimensionality-reduction]] - PCA and other non-learned compression.
- [[embeddings]] - what SSL encoders produce and how they are used.
- [[multimodal-models]] - CLIP-style contrastive image-text training.
- [[diffusion-models]] - generative models that use VAE latents.
- [[pretraining-and-scaling-laws]] - self-supervised next-token pretraining for LLMs.

## References
- Goodfellow et al., *Deep Learning*, chapter 14 (Autoencoders): https://www.deeplearningbook.org/contents/autoencoders.html
- Kingma & Welling, Auto-Encoding Variational Bayes (VAE): https://arxiv.org/abs/1312.6114
- Chen et al., SimCLR: https://arxiv.org/abs/2002.05709
- He et al., Masked Autoencoders (MAE): https://arxiv.org/abs/2111.06377
