# Benchmarks

Measured on MetaX C500 (32 GB VRAM).

## Speed

RT factor = audio duration / processing time. Higher is faster.

### GigaAM v3 (e2e_ctc)

| Audio Length | Time | RT Factor | Notes |
|--------------|------|-----------|-------|
| 5 sec | 0.64s | 7.8x | Single chunk |
| 30 sec | 1.03s | 29x | Single chunk |
| 18 min | 51.6s | 21x | VAD + 76 chunks |

### Qwen3-ASR-1.7B

| Audio Length | Time | RT Factor | Notes |
|--------------|------|-----------|-------|
| 30 sec | ~5s | 6x | Single chunk |
| 18 min | ~3.3 min | 5.5x | VAD + forced aligner |

### Diarization (pyannote community-1)

| Audio Length | Time | RT Factor |
|--------------|------|-----------|
| 30 sec | ~2s | 15x |
| 18 min | ~50s | 13x |

## Accuracy

### Word Error Rate (WER) — Golos Crowd test set

| Backend | WER | Sub | Ins | Del | Ref Words | Samples |
|---------|:---:|:---:|:---:|:---:|:---------:|:-------:|
| **GigaAM-v3** | **13.78%** | 27 | 0 | 8 | 254 | 50 |
| Qwen3-ASR-1.7B | 16.14% | 32 | 2 | 7 | 254 | 50 |

Official GigaAM WER on full Golos Crowd test set: **2.76%** (source: [ai-sage/GigaAM](https://huggingface.co/ai-sage/GigaAM)).

### Speaker Diarization

| Metric | Value | Notes |
|--------|-------|-------|
| Speaker detection | 95%+ | On clear audio |
| Overlap accuracy | 85%+ | When speakers overlap |
| Minimum segment | 0.5s | Shorter segments may be missed |

## Memory Usage

| Component | VRAM | RAM |
|-----------|------|-----|
| GigaAM-v3 | ~1.5 GB | ~0.5 GB |
| Qwen3-ASR-1.7B | ~3.5 GB | ~1 GB |
| Qwen3-ForcedAligner-0.6B | ~1.2 GB | ~0.5 GB |
| pyannote community-1 | ~2 GB | ~1 GB |
| Punctuation model | ~0.5 GB | ~0.3 GB |
| **Total (GigaAM pipeline)** | **~4 GB** | **~2 GB** |
| **Total (Qwen3 pipeline)** | **~7 GB** | **~3 GB** |

## Recommendations

- **Minimum GPU**: 8 GB VRAM (GigaAM works, Qwen3 tight)
- **Recommended GPU**: 16 GB VRAM (comfortable for full pipeline)
- **Minimum RAM**: 16 GB
- **Recommended RAM**: 32 GB
- **Disk for models**: ~10 GB

## How to Benchmark

```bash
# Quick test (50 samples)
uv run qwen-transkrib bench bond005/sberdevices_golos_10h_crowd -n 50 --backend gigaam

# Full test (all samples)
uv run qwen-transkrib bench bond005/sberdevices_golos_10h_crowd --backend gigaam

# Custom dataset
uv run qwen-transkrib bench my-org/my-dataset --split validation -n 100
```
