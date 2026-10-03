---
title: Speaker diarization
category: speech
tags: [diarization, pyannote, speaker-identification, transcription, whisperx, meetings]
use_cases:
  - "label who said what in a meeting or interview transcript"
  - "separate agent and customer speech in call recordings"
  - "split a podcast or panel recording into per-speaker segments"
  - "combine Whisper transcription with speaker labels"
  - "count the number of speakers in an audio file"
status: draft
last_verified: 2026-10-03
sources:
  - https://github.com/pyannote/pyannote-audio
  - https://github.com/m-bain/whisperX
  - https://huggingface.co/pyannote/speaker-diarization-community-1
---

# Speaker diarization

## Summary
Diarization answers "who spoke when": it segments audio into speaker-homogeneous turns labelled SPEAKER_00, SPEAKER_01 and so on, without knowing real identities. It is combined with speech-to-text to produce speaker-attributed transcripts. The leading open toolkit is pyannote.audio; hosted STT APIs often include diarization as an option.

## Key concepts
- Pipeline stages: voice activity detection, speaker embedding of segments, clustering into speakers, overlap handling.
- Labels are anonymous per file; mapping to named people needs speaker identification (enrolment embeddings) or manual naming.
- Overlapped speech (people talking at once) is the hardest part and the main source of error.
- Metric: diarization error rate (DER) = missed speech + false alarm + speaker confusion. pyannote reports DER from roughly 8.9% to 46.8% across datasets for its open community-1 pipeline, so results vary hugely by audio.
- Speaker count hints: `num_speakers`, `min_speakers`, `max_speakers` improve clustering when known.
- Alignment: merge diarization turns with word timestamps from STT; WhisperX packages transcription, word alignment and pyannote diarization.
- Multichannel recordings (one channel per participant, typical of phone calls) make diarization trivial: transcribe per channel.

## When to use / scenarios
- Meetings, interviews, podcasts, depositions: speaker-labelled minutes and summaries.
- Call centres: agent vs customer talk time, compliance scripts, sentiment per speaker (prefer stereo recording where possible).
- Journalism and research: qualitative interview analysis.
- Healthcare consultations: doctor/patient attribution (privacy and consent rules apply).
- Not for: real-time live captions with speaker labels at very low latency (needs streaming diarization from a vendor), or authenticating identity (a different, security-sensitive task).

## Setup & code
```bash
pip install pyannote.audio
```
The open pipeline requires accepting the model conditions on Hugging Face and an access token (per the pyannote README). Use a GPU if available.
```python
import torch
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1",
                                    token="HF_TOKEN")      # read from an env var in real code
pipeline.to(torch.device("cuda"))                           # omit on CPU (slower)
output = pipeline("meeting.wav")
# The README shows the result object; iterate its speaker turns (check the current output API
# in the repo, as it changed between major versions).
print(output)
```
For a one-step transcript with speakers, see WhisperX (`m-bain/whisperX`), which chains faster-whisper, alignment and pyannote.

## Choosing / trade-offs
- pyannote open pipeline: free to run locally, good quality, needs HF token and acceptance of terms; pyannoteAI hosts a higher-accuracy premium pipeline.
- Hosted STT with built-in diarization (AssemblyAI, Deepgram, Google, Azure etc.): one call, scalable, per-minute cost, data leaves your network.
- Alternatives: NVIDIA NeMo (Sortformer/MSDD models), SpeechBrain; evaluate on your audio.
- Known number of speakers: pass it; unknown: let the model estimate and review.
- Separate channels beat any model: record per-participant tracks when you can.

## Gotchas
- Wrong or varying speaker counts on short clips; very short turns ("yeah") get misassigned.
- Overlap and crosstalk produce attribution errors in the transcript; do not use labels as legal evidence without review.
- Speaker labels are not consistent across files; do not assume SPEAKER_00 is the same person in the next recording.
- Audio quality (far-field mics, echo, music) degrades DER sharply.
- Voice embeddings are biometric data in many jurisdictions; obtain consent and avoid persistent voiceprints without a legal basis ([[face-and-pose]] covers the parallel legal risks).
- Pretrained pipeline names and output APIs change; read the current README and model card.
- Diarization plus STT timestamps can misalign by hundreds of milliseconds; use word-level alignment.

## Related
- [[speech-to-text]] - transcription stage.
- [[voice-agents]] - live calls where channels are already separate.
- [[customer-support]] - call analytics scenario.
- [[huggingface-transformers]] - model and token management.
- [[ai-security-privacy-compliance]] - voice data handling.

## References
- https://github.com/pyannote/pyannote-audio
- https://huggingface.co/pyannote/speaker-diarization-community-1
- https://github.com/m-bain/whisperX
