# qwen-transkrib

Multilingual audio transcription with speaker diarization. Supports **Qwen3-ASR** and **GigaAM v3 (e2e_rnnt)** (Russian ASR, WER 2.76%) backends. Uses pyannote for speaker identification and language-specific punctuation restoration.

## Quickstart

```bash
pip install uv
git clone https://github.com/GRomR1/qwen-transkrib.git
cd qwen-transkrib
uv sync
./apply_patches.sh  # Required for MetaX GPU
uv run qwen-transkrib transcribe audio.wav --context "Google, Microsoft, Amazon"
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

### MetaX GPU

On MetaX (MACA) GPUs, PyTorch links against `libmctlassEx.so` from the MACA SDK,
which `uv sync` does **not** install. Install it via apt before running — the
version must match your installed MACA SDK (`3.8.1.3` for the C550):

```bash
sudo apt-get install -y mctlassex_3.8.1=3.8.1.3
```

Without it, `import torch` fails with
`ImportError: libmctlassEx.so: cannot open shared object file`. Note that the
similarly named `mctlass_3.8.1` package is **headers-only** and does not provide
this library.

### pip

```bash
pip install .
# Apply patches manually if on MetaX
```

### Docker

```bash
docker build -t qwen-transkrib .
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
  --backend gigaam \
  --punct \
  --vad \
  --normalize \
  --glossary "гугл=Google,майкрософт=Microsoft"

# Benchmark WER against reference dataset
uv run qwen-transkrib bench bond005/podlodka_speech -n 20 --backend gigaam
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

## Output Formats

- **SRT** - SubRip subtitles (for video players)
- **VTT** - WebVTT subtitles (for web)
- **JSON** - Full transcription with word-level timestamps
- **TXT** - Plain text

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — data flow, modules, design decisions
- [Benchmarks](docs/BENCHMARKS.md) — WER, RT factor, memory usage
- [Troubleshooting](docs/TROUBLESHOOTING.md) — common issues and fixes

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
