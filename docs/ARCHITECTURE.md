# Architecture

## Data Flow

```
                    ┌─────────────┐
                    │  Audio File  │
                    │  (wav/mp3)   │
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
    ┌─────────────────┐     ┌─────────────────┐
    │   Qwen3-ASR     │     │    pyannote      │
    │   (1.7B)        │     │  community-1     │
    │                 │     │                  │
    │  Input: audio   │     │  Input: waveform │
    │  Output: words  │     │  Output: turns   │
    │  + timestamps   │     │  (start/end/     │
    │                 │     │   speaker)       │
    └────────┬────────┘     └────────┬─────────┘
             │                       │
             │  ┌────────────────────┘
             │  │
             ▼  ▼
    ┌─────────────────────┐
    │ Punctuation Model   │
    │                     │
    │ Russian: rubert     │
    │ English: multilingual│
    │                     │
    │ Restores: . ? ! ,   │
    │ + Capitalization    │
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │  assign_speakers()  │
    │                     │
    │  Interval-overlap   │
    │  algorithm          │
    │                     │
    │  Input: words +     │
    │         turns       │
    │  Output: segments   │
    │  with speaker labels│
    └──────────┬──────────┘
               │
    ┌──────────┴──────────┐
    │                     │
    ▼         ▼         ▼         ▼
  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐
  │ SRT │  │ VTT │  │ JSON│  │ TXT │
  └─────┘  └─────┘  └─────┘  └─────┘
```

## Modules

### `asr.py`
Wraps `Qwen3ASRModel` for speech recognition with word-level timestamps. Uses the transformers backend (not vLLM). Caches models in memory for repeated calls.

### `punctuate.py`
Restores punctuation and capitalization to raw ASR output. Language-specific models:
- Russian: `markusiko/rubert-base-punctuation`
- English: `oliverguhr/fullstop-punctuation-multilingual-base`

### `diarize.py`
Wraps `pyannote.audio.Pipeline` for speaker diarization. Loads audio via `soundfile` (avoids torchcodec issues on MetaX). Returns DataFrame with speaker turns.

### `merge.py`
Pure Python implementation of interval-overlap speaker assignment (ported from whisperX). For each word, finds the diarization turn with maximum temporal overlap. Groups consecutive same-speaker words into Segments.

### `writers/`
Output format writers. Each takes a `TranscriptionResult` and writes to a file. Supported formats: SRT, VTT, JSON, TXT.

### `cli.py`
Typer-based CLI with two commands:
- `transcribe`: Full pipeline (ASR → punctuation → diarize → merge → output)
- `info`: Environment diagnostics

### `schemas.py`
Pydantic models for type-safe data flow between modules.

### `config.py`
Settings loaded from environment variables with `QWEN_` prefix.

## Key Design Decisions

1. **No vLLM**: vLLM is broken on MetaX for long audio. Transformers backend is reliable.
2. **No whisperX import**: Copy the interval-overlap algorithm, don't import the package (its ASR backend doesn't work on MetaX).
3. **pyannote community-1 only**: Version 3.1 gives garbage output on MetaX stub-fbank stack.
4. **soundfile for audio loading**: Avoids torchcodec dependency issues on MetaX.
5. **Model caching**: Avoids reloading 1.7B+ models on repeated CLI calls.
6. **Language-specific punctuation**: Different models for different languages (Russian vs English).
