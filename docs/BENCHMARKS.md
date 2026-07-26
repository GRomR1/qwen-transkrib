# Benchmarks

Measured on MetaX C500 (32 GB VRAM).

## Speed

RT factor = audio duration / processing time. Higher is faster.

### GigaAM v3 (e2e_rnnt)

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

### Word Error Rate (WER) — Podlodka Speech test set (long audio)

Primary benchmark dataset. 20 samples, 8-60 seconds each, Russian podcast audio.

| Backend | WER | Sub | Ins | Del | Ref Words | Samples |
|---------|:---:|:---:|:---:|:---:|:---------:|:-------:|
| **GigaAM-v3 (e2e_rnnt) + native** | **7.74%** | 53 | 24 | 11 | 1137 | 20 |
| GigaAM-v3 (e2e_rnnt) + HF | 8.09% | 45 | 38 | 9 | 1137 | 20 |
| Qwen3-ASR-1.7B | 12.40% | 105 | 24 | 12 | 1137 | 20 |

Note: Native gigaam requires patch (see `patches/apply_gigaam_patch.py`). Official GigaAM WER on full Golos Crowd test set: **2.76%**.

### Word Error Rate (WER) — Golos Crowd test set (short commands)

| Backend | WER | Sub | Ins | Del | Ref Words | Samples |
|---------|:---:|:---:|:---:|:---:|:---------:|:-------:|
| **GigaAM-v3 (e2e_rnnt)** | **13.15%** | 49 | 1 | 16 | 502 | 100 |
| GigaAM-v3 (e2e_rnnt, 50 samples) | **12.99%** | 24 | 0 | 9 | 254 | 50 |
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
# Primary benchmark (Podlodka Speech - long audio)
uv run qwen-transkrib bench bond005/podlodka_speech -n 20 --backend gigaam

# Full benchmark (all 20 samples)
uv run qwen-transkrib bench bond005/podlodka_speech --backend gigaam

# Legacy benchmark (Golos Crowd - short commands)
uv run qwen-transkrib bench bond005/sberdevices_golos_10h_crowd -n 50 --backend gigaam

# Custom dataset
uv run qwen-transkrib bench my-org/my-dataset --split validation -n 100
```
