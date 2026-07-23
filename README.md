# qwen-transkrib

Multilingual audio transcription with speaker diarization. Supports **Qwen3-ASR** and **GigaAM v3** (Russian ASR, WER 2.76%) backends. Uses pyannote for speaker identification and language-specific punctuation restoration.

## Quickstart

```bash
pip install uv
git clone https://github.com/GRomR1/qwen-transkrib.git
cd qwen-transkrib
uv sync
./apply_patches.sh  # Required for MetaX GPU
uv run qwen-transkrib transcribe audio.wav --context "Google, Microsoft, Amazon"
```

## Architecture

```
Audio File
    │
    ├──► Silero VAD ──► Speech segments (split at pauses)
    │
    ├──► Qwen3-ASR-1.7B ──► Words with timestamps (per chunk)
    │
    ├──► Punctuation Model ──► Restored punctuation & capitalization
    │   (Russian: kontur-ai/sbert_punc_case_ru)
    │   (English: fullstop-punctuation-multilingual)
    │
    ├──► pyannote community-1 ──► Speaker turns (start, end, speaker)
    │
    └──► assign_speakers() ──► Segments with speaker labels
                                    │
                                    ├──► SRT
                                    ├──► VTT
                                    ├──► JSON
                                    └──► TXT
```

## Requirements

- Python 3.12
- GPU: MetaX C500/C550/C650 or NVIDIA CUDA
- RAM: 16 GB minimum, 32 GB recommended
- Disk: 10 GB for models

## Installation

### uv (recommended)

```bash
pip install uv
uv sync
./apply_patches.sh  # Apply MetaX-specific patches
```

### pip

```bash
pip install .
# Apply patches manually if on MetaX
```

### Docker

```bash
# Build
docker build -t qwen-transkrib .

# Run
docker run --gpus all -v $(pwd)/data:/data qwen-transkrib transcribe /data/audio.wav

# Or use Docker Compose
docker compose up asr
```

## Usage

### CLI

```bash
# Basic transcription (Russian)
uv run qwen-transkrib transcribe audio.wav

# English transcription
uv run qwen-transkrib transcribe audio.wav --language English

# With diarization and context for term correction
uv run qwen-transkrib transcribe audio.wav --context "Google, Microsoft, Amazon"

# All options
uv run qwen-transkrib transcribe input.webm \
  --output-dir ./out \
  --language Russian \
  --format srt,json,txt \
  --context "Google, Microsoft, Amazon" \
  --device cuda:0 \
  --punct \
  --vad

# Without VAD (process entire file at once)
uv run qwen-transkrib transcribe audio.wav --no-vad

# Without punctuation restoration
uv run qwen-transkrib transcribe audio.wav --no-punct

# Benchmark WER against reference dataset
uv run qwen-transkrib bench bond005/sberdevices_golos_10h_crowd -n 50 --backend gigaam

# Show environment info
uv run qwen-transkrib info
```

### Python API

```python
from pathlib import Path
from qwen_transkrib import transcribe_file, diarize_file, assign_speakers, Settings

settings = Settings()
words, text, lang = transcribe_file(Path("audio.wav"), settings, context="Google, Microsoft")
diar = diarize_file(Path("audio.wav"), settings)
segments = assign_speakers(words, diar)
```

## Benchmarking

The `bench` command measures ASR accuracy (WER) against reference datasets from HuggingFace.

```bash
# Benchmark GigaAM on Golos Crowd (Russian)
uv run qwen-transkrib bench bond005/sberdevices_golos_10h_crowd -n 50 --backend gigaam

# Benchmark Qwen3-ASR (default)
uv run qwen-transkrib bench bond005/sberdevices_golos_10h_crowd -n 50

# Custom dataset and split
uv run qwen-transkrib bench my-org/my-dataset --split validation -n 100
```

Output:
```
WER:      13.78%
Sub:      27
Ins:      0
Del:      8
Ref words: 254
Samples:  50
```

**Note:** Benchmark strips punctuation for fair comparison across models. WER varies by sample count — official GigaAM WER on full Golos Crowd test set is 2.76%.

## Supported Languages

| Language | ASR Model | Punctuation Model |
|----------|-----------|-------------------|
| Russian | Qwen3-ASR-1.7B / GigaAM-v3 | kontur-ai/sbert_punc_case_ru (Qwen3) / built-in (GigaAM) |
| English | Qwen3-ASR-1.7B | oliverguhr/fullstop-punctuation-multilingual-base |

## Configuration

### Environment Variables

```bash
# Required for diarization (HuggingFace token)
export HF_TOKEN=your_token_here

# Optional: Model settings
export QWEN_ASR_MODEL=Qwen/Qwen3-ASR-1.7B
export QWEN_ALIGNER_MODEL=Qwen/Qwen3-ForcedAligner-0.6B
export QWEN_LANGUAGE=Russian
export QWEN_DEVICE=cuda:0
```

### .env file

Create `.env` in project root:
```
HF_TOKEN=your_token_here
```

## MetaX GPU Support

For MetaX C500/C550/C650 GPUs, apply the required patches:

```bash
./apply_patches.sh
```

This patches:
1. `transformers` cache_utils.py - fixes empty tensor initialization
2. `pyannote` wespeaker - fixes torch.vmap storage issue
3. `pyannote` io.py - suppresses noisy torchcodec warning (falls back to torchaudio)

### torchaudio-stub

The `packages/torchaudio-stub` provides a minimal torchaudio replacement that works with MetaX GPUs. It includes:
- `load`/`save` via soundfile (no ffmpeg dependency)
- `MelSpectrogram` transform (required by GigaAM)
- Stub implementations for other transforms

This avoids the MetaX torchaudio compatibility issues (mel spectrogram channel mismatch, missing transforms).

## Output Formats

- **SRT** - SubRip subtitles (for video players)
- **VTT** - WebVTT subtitles (for web)
- **JSON** - Full transcription with word-level timestamps
- **TXT** - Plain text

## Troubleshooting

### Warnings in logs

The following warnings are harmless and can be ignored:

- **`SyntaxWarning: invalid escape sequence`** (from nagisa) — third-party library issue, suppressed automatically
- **`UserWarning: torchcodec is not installed correctly`** — expected on MetaX GPUs; pyannote falls back to built-in audio decoding

### Punctuation not working

Punctuation is **enabled by default**. If your output lacks punctuation:

1. Check that you're not using `--no-punct`
2. Ensure the punctuation model downloaded successfully on first run
3. Try running with `--punct` explicitly

### Diarization fails with `HF_TOKEN not set`

Set your HuggingFace token:
```bash
export HF_TOKEN=your_token_here
```
Or create a `.env` file with `HF_TOKEN=your_token_here`.

### GigaAM backend

GigaAM v3 (CTC revision) is supported as an alternative ASR backend for Russian. Uses HuggingFace transformers for text transcription (no native gigaam package required for basic use). Includes built-in punctuation and capitalization.

Benchmarked WER on Golos Crowd:

| Backend | WER | Sub | Ins | Del | Size |
|---------|:---:|:---:|:---:|:---:|:----:|
| **GigaAM-v3** (CTC) | **2.76%** | 6 | 0 | 1 | 240M params |
| Qwen3-ASR-1.7B | 57.48% | 138 | 2 | 6 | 1.7B params |

```bash
# Transcribe with GigaAM
uv run qwen-transkrib transcribe audio.wav --backend gigaam

# With VAD for long audio
uv run qwen-transkrib transcribe long_audio.wav --backend gigaam --vad

# Benchmark GigaAM WER
uv run qwen-transkrib bench bond005/sberdevices_golos_10h_crowd -n 50 --backend gigaam

# Benchmark Qwen3-ASR WER
uv run qwen-transkrib bench bond005/sberdevices_golos_10h_crowd -n 50
```

**Note:** For word-level timestamps, install the native gigaam package: `uv sync --extra gigaam`.

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run tests
uv run pytest tests/ -v

# Lint
uv run ruff check src/

# Type check
uv run mypy src/
```

## License

Apache-2.0
