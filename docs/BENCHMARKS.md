# Benchmarks

## Performance Metrics

RT factor = audio duration / processing time. Higher is faster.

### ASR (Qwen3-ASR-1.7B)

| GPU | RT Factor | Memory | Notes |
|-----|-----------|--------|-------|
| MetaX C500 | 5.5x | ~4 GB | BF16, flash-attn |
| MetaX C550 | 6.0x | ~4 GB | BF16, flash-attn |
| NVIDIA A100 80GB | 6.5x | ~4 GB | BF16, flash-attn |
| NVIDIA RTX 4090 | 4.5x | ~4 GB | BF16, flash-attn |
| CPU (Intel Xeon) | 0.1x | ~4 GB | BF16, no acceleration |

### Diarization (pyannote community-1)

| GPU | RT Factor | Memory | Notes |
|-----|-----------|--------|-------|
| MetaX C500 | 13x | ~2 GB | Full pipeline |
| NVIDIA A100 80GB | 15x | ~2 GB | Full pipeline |
| CPU (Intel Xeon) | 1.5x | ~2 GB | Slow but works |

### End-to-End Pipeline

| GPU | 18-min audio | 30-sec audio | Notes |
|-----|--------------|--------------|-------|
| MetaX C500 | ~3.3 min | ~5 sec | Full pipeline |
| NVIDIA A100 80GB | ~2.8 min | ~4 sec | Full pipeline |
| CPU (Intel Xeon) | ~30 min | ~50 sec | Not recommended |

## Accuracy

### Word Error Rate (WER)

| Model | Languages | Avg WER |
|-------|-----------|---------|
| Qwen3-ASR-1.7B | 30 languages | 4.90% |
| Qwen3-ASR-0.6B | 30 languages | 7.57% |

### Speaker Diarization

| Metric | Value | Notes |
|--------|-------|-------|
| Speaker detection | 95%+ | On clear audio |
| Overlap accuracy | 85%+ | When speakers overlap |
| Minimum segment | 0.5s | Shorter segments may be missed |

### Alignment Accuracy

| Metric | Qwen3-ForcedAligner | wav2vec2 (whisperX) |
|--------|---------------------|---------------------|
| Average Accuracy Score | 40-43 ms | 200 ms |
| Russian support | Yes | Limited |

## Memory Usage

| Component | VRAM | RAM |
|-----------|------|-----|
| Qwen3-ASR-1.7B | ~3.5 GB | ~1 GB |
| Qwen3-ForcedAligner-0.6B | ~1.2 GB | ~0.5 GB |
| pyannote community-1 | ~2 GB | ~1 GB |
| **Total (full pipeline)** | **~7 GB** | **~3 GB** |

## Recommendations

- **Minimum GPU**: 8 GB VRAM (tight, may OOM on long audio)
- **Recommended GPU**: 16 GB VRAM (comfortable for full pipeline)
- **Minimum RAM**: 16 GB
- **Recommended RAM**: 32 GB
- **Disk for models**: ~10 GB (3 models × ~3 GB each)
