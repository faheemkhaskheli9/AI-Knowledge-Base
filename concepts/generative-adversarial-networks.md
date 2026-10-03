---
title: Generative adversarial networks (GANs)
category: concepts
tags: [gan, generative-model, generator, discriminator, dcgan, stylegan, conditional-gan, mode-collapse, pytorch]
use_cases:
  - "generate realistic synthetic images or tabular rows to augment a small dataset"
  - "do fast single-step image-to-image translation or super-resolution"
  - "understand GANs and when diffusion models have replaced them"
status: draft
last_verified: 2026-10-03
sources:
  - https://arxiv.org/abs/1406.2661
  - https://arxiv.org/abs/1511.06434
  - https://arxiv.org/abs/1701.07875
  - https://arxiv.org/abs/1912.04958
  - https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
---

# Generative adversarial networks (GANs)

## Summary
A GAN trains two networks against each other: a **generator** turns random noise into samples, and a **discriminator** tries to tell those samples from real data. As the discriminator gets better at spotting fakes, the generator gets better at fooling it, until generated samples look like the data. GANs gave the first photorealistic image synthesis (StyleGAN) and still win on single-step speed, but for most new image generation work [[diffusion-models]] have replaced them because they train more stably and cover the data better.

## Key concepts
- **Minimax game.** The discriminator `D` minimises classification loss on real-vs-fake; the generator `G` maximises `D`'s mistakes. Training alternates one `D` step and one `G` step. No explicit likelihood is computed.
- **Non-saturating loss.** In practice `G` is trained to make `D` label fakes as real (`BCE(D(G(z)), 1)`) rather than to minimise `D`'s success, which gives usable gradients early on.
- **Mode collapse.** `G` finds a few outputs that fool `D` and produces only those, ignoring the rest of the data distribution. The central GAN failure.
- **Stabilisers.** DCGAN conventions (strided convs, batch norm, LeakyReLU in `D`, Adam with beta1 = 0.5); **Wasserstein loss with gradient penalty** (WGAN-GP); **spectral normalisation** on `D`; two learning rates (TTUR).
- **Conditional GAN.** Feed a label or input image to both networks to control the output: class-conditional generation, pix2pix (paired image translation), CycleGAN (unpaired translation).
- **StyleGAN family.** Maps noise to a style space that modulates each layer; state of the art for single-domain faces/objects, with a controllable latent space.
- **Evaluation.** No loss value tells you quality. Use FID (distance between Inception features of real and generated sets, lower is better) plus looking at samples, and check diversity, not just realism.

## When to use / scenarios
- **Real-time or on-device generation** where one forward pass matters: super-resolution (ESRGAN-style), face/photo enhancement, style transfer in video.
- **Narrow, single-domain generators** with plenty of data: product renders, textures, faces, defect images for training an inspection model.
- **Tabular synthetic data** for privacy-preserving sharing or augmentation (CTGAN in the SDV library), see [[data-labeling-and-synthetic-data]].
- **Adversarial loss as a component:** the decoder of image VAEs/tokenisers used by latent diffusion and some vocoders in [[text-to-speech]] are trained with a GAN loss for sharpness.
- **Not when:** you want text-to-image or broad, diverse generation (use [[diffusion-models]]), you need likelihoods/density estimates (VAEs, flows), or you have little data (GANs memorise or collapse; augment with ADA or use a pretrained diffusion model).

## Setup & code
```bash
pip install torch
```
A minimal GAN learning a 2-D ring distribution (the same loop scales to images with conv layers):
```python
import torch
from torch import nn

torch.manual_seed(0)
def real(n):                                   # target: a ring of radius 2 with a little noise
    angle = torch.rand(n, 1) * 2 * torch.pi
    return torch.cat([angle.cos(), angle.sin()], dim=1) * 2 + 0.05 * torch.randn(n, 2)

G = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))
D = nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(0.2), nn.Linear(64, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1))
opt_g = torch.optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.999))
bce = nn.BCEWithLogitsLoss()                   # D outputs logits, not probabilities
ones, zeros = torch.ones(256, 1), torch.zeros(256, 1)

for step in range(3000):
    # 1) Discriminator: real -> 1, fake -> 0 (detach so G is not updated here)
    fake = G(torch.randn(256, 8))
    loss_d = bce(D(real(256)), ones) + bce(D(fake.detach()), zeros)
    opt_d.zero_grad(); loss_d.backward(); opt_d.step()
    # 2) Generator: make D call fakes real (non-saturating loss)
    loss_g = bce(D(fake), ones)
    opt_g.zero_grad(); loss_g.backward(); opt_g.step()

with torch.no_grad():
    s = G(torch.randn(2000, 8))
    radius, angle = s.norm(dim=1), torch.atan2(s[:, 1], s[:, 0])
print(f"D loss {loss_d.item():.2f}  G loss {loss_g.item():.2f}")
print(f"sample radius: mean {radius.mean():.2f} (target 2.00), std {radius.std():.2f}")
print(f"angle coverage: {torch.histc(angle, bins=8, min=-torch.pi, max=torch.pi).gt(0).sum().item()}/8 sectors")
```
On a test run (torch 2.13, CPU, under a minute) samples landed on the ring (mean radius about 1.99, std about 0.07) and covered all 8 angular sectors, i.e. no mode collapse. `D loss` near 1.39 (= 2 ln 2) and `G loss` near 0.69 (= ln 2) mean the discriminator is guessing, which is the equilibrium you want; it does not by itself prove good samples, so the script also checks the samples directly.

For images, start from the PyTorch DCGAN tutorial (linked below) or a maintained StyleGAN implementation rather than designing the architecture yourself.

## Choosing / trade-offs
- **GAN vs diffusion.** GAN: one forward pass (milliseconds), sharp outputs, unstable training, weaker diversity. Diffusion: many denoising steps (distillation narrows the gap), stable training, better coverage and text conditioning. New projects default to diffusion unless latency is the binding constraint.
- **GAN vs VAE.** VAE trains stably and gives a smooth latent space and a likelihood bound, but samples are blurrier; GAN samples are sharper. Combining them (VAE + adversarial loss) is common ([[autoencoders-and-self-supervised-learning]]).
- **Loss.** Standard non-saturating BCE with spectral norm is a fine start; switch to WGAN-GP or hinge loss if training oscillates or collapses.
- **Synthetic tabular data.** CTGAN handles mixed types but is not automatically private; for formal guarantees use differentially private synthesisers.

## Gotchas
- Generator and discriminator losses do not go down like a normal loss; judge progress by samples and FID, saved at checkpoints.
- If `D` wins outright (its loss near 0), `G` gets no useful gradient; lower `D`'s capacity or learning rate, add noise/label smoothing, or use spectral norm.
- Mode collapse can look like great samples: always look at a large grid of samples and check class/attribute coverage.
- Forgetting `.detach()` on the fake batch in the `D` step updates `G` with the wrong objective.
- Batch norm in `D` leaks batch statistics between real and fake batches; keep real and fake in separate batches (as above) or use other normalisation.
- GANs trained on small datasets memorise training images; check nearest neighbours in the training set before claiming novelty or privacy.
- FID depends on the sample count and the feature extractor; compare FID numbers only when computed the same way.

## Related
- [[diffusion-models]] - the successor for most image generation.
- [[autoencoders-and-self-supervised-learning]] - VAEs, the other classic generative model.
- [[cnn-and-rnn-architectures]] - the conv layers inside image GANs.
- [[deep-learning-training]] - optimisers and normalisation used to stabilise training.
- [[data-labeling-and-synthetic-data]] - synthetic data, including CTGAN for tables.
- [[image-generation-models]] - practical image generation tools today.

## References
- Goodfellow et al., Generative Adversarial Nets: https://arxiv.org/abs/1406.2661
- Radford et al., DCGAN: https://arxiv.org/abs/1511.06434
- Arjovsky et al., Wasserstein GAN: https://arxiv.org/abs/1701.07875
- Karras et al., StyleGAN2: https://arxiv.org/abs/1912.04958
- PyTorch DCGAN tutorial: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
