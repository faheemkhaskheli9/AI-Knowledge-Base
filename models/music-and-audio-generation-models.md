---
title: Music and audio generation models
category: models
tags: [music-generation, text-to-music, sound-effects, sfx, lyria, elevenlabs, stable-audio, musicgen, ace-step, yue, licensing]
use_cases:
  - "generate royalty-free background music for videos or a game"
  - "create sound effects from text descriptions for an app"
  - "self-host an open-weight music model for commercial use"
  - "check whether we can sell or copyright AI-generated music"
  - "add a soundtrack to generated video clips"
status: draft
last_verified: 2026-10-15
sources:
  - https://ai.google.dev/gemini-api/docs/realtime-music-generation
  - https://elevenlabs.io/music-api
  - https://elevenlabs.io/docs/overview/capabilities/sound-effects
  - https://stability.ai/license
  - https://github.com/facebookresearch/audiocraft
  - https://github.com/ACE-Step/ACE-Step-1.5
  - https://arxiv.org/abs/2503.08638
  - https://www.copyright.gov/ai/Copyright-and-Artificial-Intelligence-Part-2-Copyrightability-Report.pdf
---

# Music and audio generation models

## Summary
Generative audio covers text-to-music (instrumental or full songs with
vocals), text-to-sound-effects (SFX), and real-time/interactive music. Hosted
APIs (Google Lyria, ElevenLabs Music and Sound Effects) are the easy path;
open-weight models (ACE-Step, YuE, Stable Audio Open, MusicGen) can be
self-hosted. The hard part is usually not quality but rights: the licence on
the weights, the vendor's terms for outputs, and whether outputs can be
copyrighted at all. Speech is a separate topic ([[text-to-speech]]).

## Key concepts
- **Two main architectures**: (1) autoregressive language models over discrete
  audio tokens from a neural codec (MusicGen over EnCodec tokens; YuE is a
  LLaMA-style LM for lyrics-to-song); (2) latent diffusion over a compressed
  audio latent (Stable Audio, ACE-Step uses diffusion with a compression
  autoencoder). See [[diffusion-models]], [[transformers-and-attention]].
- **Residual vector quantization (RVQ)**: neural codecs encode each audio frame
  with a stack of codebooks, each quantizing the residual of the previous one;
  more codebooks = higher fidelity, more tokens to generate. The demo below
  shows the idea.
- **Conditioning**: text prompt (genre, mood, instruments, BPM), lyrics with
  section tags, a melody or reference audio, duration; real-time models take
  steerable prompts while streaming.
- **Rights have three layers**: weights licence (can you run it commercially),
  vendor output terms (who may use the output and where), copyright law (can
  anyone own the output). They are independent.

Options as of 2026-10 (verify on the linked page; licences differ per checkpoint):

| Model | Access | Notes |
|---|---|---|
| Lyria RealTime (`lyria-realtime-exp`) | Gemini API, experimental | streaming instrumental music over a WebSocket session, steerable while playing (Gemini docs). Newer Lyria 3 song models exist; IDs not verified here |
| Eleven Music | ElevenLabs API (`music.compose`) | songs with or without vocals; vendor says broad commercial use on paid plans, film/TV and large game studios need Enterprise - read the Music terms |
| ElevenLabs Sound Effects | ElevenLabs API | text-to-SFX with optional duration; confirm output rights in the terms |
| Stable Audio Open (Stability AI) | open weights | SFX and short music; Stability AI Community License - free commercial use under a revenue threshold, Enterprise licence above it (license page) |
| MusicGen (Meta, AudioCraft) | open weights | code MIT, weights CC-BY-NC 4.0: **no commercial use** of the weights |
| ACE-Step 1.5 | open weights | full songs with lyrics; MIT per the GitHub repo (some third-party pages say Apache 2.0); the card says outputs may be used commercially; LoRA from a few songs |
| YuE (HKUST / M-A-P) | open weights | lyrics-to-song up to several minutes (paper); some checkpoints Apache 2.0, at least one card lists CC-BY-NC 4.0 - check per checkpoint |
| Suno, Udio | consumer apps | no general official public API; both were sued by major labels in 2024 and have since signed licensing settlements with some labels - check current terms |

## When to use / scenarios
- Background music for social videos, ads, podcasts and generated clips
  ([[video-generation-models]], [[marketing-content]]).
- Game and app SFX prototyping; adaptive/interactive music via real-time models.
- Placeholder tracks during film/game production, replaced by licensed or
  composed music later.
- Not for: anything that must imitate a named artist's voice or style (legal
  and platform risk), or music you need to register and exclusively own
  without meaningful human authorship (see Gotchas). For stock music with
  clear rights, a licensed library may be simpler.

## Setup & code
RVQ, the tokenizer idea behind codec-based music models (numpy only, runs
offline): each extra codebook quantizes what the previous ones missed.
```python
import numpy as np

rng = np.random.default_rng(0)
sr, frame = 8000, 32
t = np.arange(sr * 2) / sr                         # 2 s synthetic "music"
audio = (0.6 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 330 * t)
         * (t % 0.5 < 0.25) + 0.05 * rng.standard_normal(t.size))
X = audio[: t.size // frame * frame].reshape(-1, frame)   # frames as vectors

def kmeans(data, k, iters=15):
    c = data[rng.choice(len(data), k, replace=False)]
    for _ in range(iters):
        idx = ((data[:, None, :] - c[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            if np.any(idx == j):
                c[j] = data[idx == j].mean(0)
    return c, idx

residual, recon, errors = X.copy(), np.zeros_like(X), []
for level in range(4):                              # 4 codebooks of 64 entries each
    codebook, codes = kmeans(residual, 64)
    q = codebook[codes]
    recon += q
    residual -= q
    errors.append(float(((X - recon) ** 2).mean()))
    print(f"codebooks={level + 1}  tokens/frame={level + 1}  MSE={errors[-1]:.5f}")
assert all(a > b for a, b in zip(errors, errors[1:]))   # more codebooks, lower error
```

Hosted API (ElevenLabs Python SDK; check its docs for the current model ID and
plan-dependent rights):
<!-- skip-check: needs ELEVENLABS_API_KEY and network -->
```python
from elevenlabs.client import ElevenLabs

client = ElevenLabs()                               # reads ELEVENLABS_API_KEY
track = client.music.compose(
    prompt="Upbeat lo-fi hip hop, warm keys, vinyl crackle, no vocals",
    music_length_ms=30_000,
)
with open("track.mp3", "wb") as f:
    for chunk in track:
        f.write(chunk)
```

## Choosing / trade-offs
- **API vs open weights**: APIs give the best vocals and song structure with
  vendor-defined rights; open weights give privacy, fixed cost and fine-tuning,
  but you carry the licence and training-data risk yourself.
- **Licence first**: MusicGen weights are non-commercial; Stability's
  community licence has a revenue threshold; permissive licences (ACE-Step,
  some YuE checkpoints) are the easiest for products ([[model-licenses]]).
- **Indemnity**: enterprise plans of some vendors include legal coverage;
  open models never do.
- **Instrumental vs vocals**: instrumental beds and SFX are mature; full songs
  with intelligible lyrics are newer and vary more in quality.
- **Real-time vs offline**: real-time streaming models suit interactive apps
  and live performance; offline generation gives longer, more structured songs.

## Gotchas
- **Copyright of outputs**: the US Copyright Office's 2025 report says
  purely AI-generated material is not protected and prompts alone are not
  enough; human selection, arrangement or modification can be. Others may be
  free to copy a track you generated. Other countries differ.
- **"Commercial use allowed" is not exclusive ownership**: vendors may grant
  the same rights to similar outputs for other users.
- **Training-data litigation** around music models is ongoing; terms and
  availability can change after settlements.
- **Content ID**: generated tracks can still match existing recordings in
  platform fingerprinting, or be claimed by others who uploaded similar output.
- **Free tiers** of hosted tools often exclude commercial use.
- **Artist imitation**: prompts naming real artists are often blocked by
  vendors and risky even when they work (right of publicity, voice cloning
  laws) ([[guardrails-and-safety]]).
- **Loudness and format**: normalise loudness and check sample rate/stereo
  before mixing with video or voice.

## Related
- [[text-to-speech]] - spoken voice rather than music or SFX.
- [[video-generation-models]] - pairing soundtracks with generated video.
- [[diffusion-models]] - latent diffusion behind several audio models.
- [[model-licenses]] - reading weight licences.
- [[mel-spectrogram-and-mfcc-from-scratch]] - audio features used to evaluate and condition models.
- [[google-gemini]] - Google's model family, including Lyria.

## References
- Gemini API, Lyria RealTime music generation: https://ai.google.dev/gemini-api/docs/realtime-music-generation
- ElevenLabs Music API: https://elevenlabs.io/music-api
- ElevenLabs Sound Effects: https://elevenlabs.io/docs/overview/capabilities/sound-effects
- Stability AI licences: https://stability.ai/license
- Meta AudioCraft (MusicGen): https://github.com/facebookresearch/audiocraft
- ACE-Step 1.5: https://github.com/ACE-Step/ACE-Step-1.5
- Yuan et al., YuE (2025): https://arxiv.org/abs/2503.08638
- US Copyright Office, Copyright and AI Part 2: Copyrightability (2025): https://www.copyright.gov/ai/Copyright-and-Artificial-Intelligence-Part-2-Copyrightability-Report.pdf
