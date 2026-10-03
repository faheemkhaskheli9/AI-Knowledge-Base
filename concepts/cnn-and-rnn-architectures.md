---
title: CNN and RNN architectures
category: concepts
tags: [deep-learning, cnn, convolution, resnet, rnn, lstm, gru, sequence-models, 1d-cnn, pytorch]
use_cases:
  - "understand how convolutional networks process images before fine-tuning one"
  - "choose between a CNN, LSTM/GRU or transformer for sensor, audio or text sequences"
  - "build a small CNN for a custom image or spectrogram classification task"
  - "classify or forecast from time-series sensor windows with a 1D CNN or LSTM"
status: draft
last_verified: 2026-10-03
sources:
  - https://d2l.ai/chapter_convolutional-neural-networks/index.html
  - https://d2l.ai/chapter_recurrent-neural-networks/index.html
  - https://pytorch.org/docs/stable/nn.html#convolution-layers
  - https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
  - https://arxiv.org/abs/1512.03385
---

# CNN and RNN architectures

## Summary
Convolutional neural networks (CNNs) and recurrent neural networks (RNNs: LSTM, GRU) are the two classic deep-learning architectures that build structure into the network: CNNs share small filters across positions, which suits images, spectrograms and local patterns in signals; RNNs carry a hidden state through a sequence step by step, which suits ordered data. Transformers ([[transformers-and-attention]]) have replaced RNNs for most language tasks and compete with CNNs in vision, but CNNs remain the workhorse for edge vision and both remain strong, cheap choices for small sensor and audio models.

## Key concepts
- **Convolution.** A small kernel (e.g. 3x3) slides over the input computing dot products; the same weights are used at every position (weight sharing), so the layer detects a pattern anywhere (translation equivariance) with few parameters.
- **Channels and feature maps.** `Conv2d(in_ch, out_ch, k)` learns `out_ch` filters; early layers detect edges/textures, deeper layers parts and objects.
- **Stride, padding, pooling.** Stride and pooling (max/average) downsample to grow the **receptive field** and cut compute; padding keeps spatial size.
- **Classic CNN block:** conv → BatchNorm → ReLU, repeated, with downsampling; a global average pool and a linear head at the end.
- **Key CNN families:** ResNet (residual connections, the standard backbone), MobileNet/EfficientNet (depthwise-separable convs for phones and edge), ConvNeXt (modernised CNN competitive with vision transformers), U-Net (encoder-decoder for [[segmentation]]).
- **1D convolutions** (`Conv1d`) slide along time: fast, parallel models for sensor windows, audio waveforms and short text.
- **RNN.** `h_t = f(W x_t + U h_{t-1})`: one hidden state updated per step. Plain RNNs suffer vanishing gradients over long sequences.
- **LSTM / GRU.** Gated cells (input/forget/output gates; GRU merges them) that keep information over hundreds of steps. GRU is smaller and often as accurate.
- **Bidirectional RNNs** read the sequence both ways when the whole sequence is available (tagging, classification), not for real-time forecasting.
- **Seq2seq and attention.** Encoder-decoder RNNs with attention were the bridge to transformers, which drop recurrence and attend over all positions in parallel.

## When to use / scenarios
- **CNN:** image classification, detection and segmentation on edge devices ([[image-classification]], [[object-detection]]); defect inspection in manufacturing; medical imaging; spectrogram-based audio classification (machine sounds, keyword spotting).
- **1D CNN / LSTM / GRU:** predictive maintenance from vibration or sensor windows, human-activity recognition from phone/wearable IMU data, ECG classification, small on-device models where a transformer is too heavy.
- **Transformer instead:** text and language in almost all cases ([[nlp-classic-tasks]]), long-range dependencies, large datasets or when strong pretrained models exist; vision transformers when data and compute are large.
- **Not deep learning at all:** short tabular time series with few series often do better with statistical or boosted models ([[time-series-forecasting]]).
- Prefer fine-tuning a pretrained CNN (ResNet, EfficientNet, ConvNeXt from `torchvision`/`timm`) over training from scratch unless the data is unusual (e.g. raw sensor channels).

## Setup & code
```bash
pip install torch
```
A small CNN for 1-channel 28x28 images and a GRU classifier for sensor windows; both run on random data to check shapes:
```python
import torch
from torch import nn

class SmallCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(1, 32), block(32, 64))   # 28 -> 14 -> 7
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(64, n_classes))

    def forward(self, x):                  # x: (batch, 1, 28, 28)
        return self.head(self.features(x))

class SensorGRU(nn.Module):
    def __init__(self, n_channels=6, hidden=64, n_classes=5):
        super().__init__()
        self.gru = nn.GRU(n_channels, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):                  # x: (batch, time, channels)
        out, h = self.gru(x)
        return self.head(h[-1])            # last layer's final hidden state

print(SmallCNN()(torch.randn(8, 1, 28, 28)).shape)   # torch.Size([8, 10])
print(SensorGRU()(torch.randn(8, 128, 6)).shape)     # torch.Size([8, 5])
```
Train either with the loop in [[deep-learning-training]] or [[pytorch-basics]]. For pretrained CNNs: `torchvision.models.resnet50(weights="IMAGENET1K_V2")`, then replace `model.fc` with a new `nn.Linear`.

## Choosing / trade-offs
- **CNN vs ViT for images.** CNNs are more data-efficient and cheaper at the edge; vision transformers scale better with very large data and pretraining. With a good pretrained checkpoint either works; pick by deployment hardware and latency.
- **LSTM/GRU vs 1D CNN vs transformer for sequences.** 1D CNNs (or temporal conv nets) are fast and parallel; GRU/LSTM are compact and natural for streaming one step at a time; transformers win on long context and large data but cost more memory (quadratic in length).
- **Depth.** Use residual blocks once a CNN exceeds a handful of layers; stacking more than 2-3 RNN layers rarely helps.
- **Input representation for audio.** Log-mel spectrogram + 2D CNN is a strong, simple baseline; raw-waveform models need more data (see [[speech-to-text]] for speech).

## Gotchas
- Shape errors are the usual first bug: PyTorch convs expect `(batch, channels, height, width)`, RNNs default to `(time, batch, features)` unless `batch_first=True`.
- Pretrained CNNs expect the same input size range and normalisation (ImageNet mean/std) they were trained with; skipping it silently costs accuracy.
- The `nn.LSTM/GRU` `dropout` argument applies only between stacked layers; with `num_layers=1` it does nothing (and warns).
- Feeding padded variable-length sequences without `pack_padded_sequence` lets padding influence the final hidden state.
- Train/serve mismatch in sliding-window sensor data: overlapping windows from the same recording split across train and validation leak; split by recording, device or person.
- RNNs process steps sequentially, so they are slow to train on long sequences on GPUs compared with convs or attention.
- Exploding gradients in RNNs: always use gradient clipping.

## Related
- [[neural-network-fundamentals]] - layers, losses and backprop these architectures build on.
- [[deep-learning-training]] - BatchNorm, LayerNorm, clipping and schedules used here.
- [[transformers-and-attention]] - the architecture that superseded RNNs for language.
- [[image-classification]] - fine-tuning pretrained CNNs in practice.
- [[time-series-forecasting]] - when classic or boosted models beat sequence nets.

## References
- *Dive into Deep Learning*, CNN chapter: https://d2l.ai/chapter_convolutional-neural-networks/index.html
- *Dive into Deep Learning*, RNN chapter: https://d2l.ai/chapter_recurrent-neural-networks/index.html
- PyTorch convolution layers: https://pytorch.org/docs/stable/nn.html#convolution-layers
- PyTorch LSTM: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
- He et al., Deep Residual Learning for Image Recognition (ResNet): https://arxiv.org/abs/1512.03385
