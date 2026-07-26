# Test Fixtures

Test audio samples are downloaded from verified HuggingFace datasets on first use.

## Verified Datasets

### bond005/podlodka_speech (Primary)

Russian podcast audio (tech domain). Longer clips for testing VAD chunking.

- **Source**: [bond005/podlodka_speech](https://huggingface.co/datasets/bond005/podlodka_speech)
- **Durations**: 8-60 seconds per sample
- **Test split**: 20 samples with transcriptions
- **Use case**: Long audio, VAD chunking, context carry-over

```bash
uv run qwen-transkrib bench bond005/podlodka_speech -n 20 --backend gigaam
```

### bond005/sberdevices_golos_10h_crowd

Short Russian voice commands. Crowdsourced recordings.

- **Source**: [bond005/sberdevices_golos_10h_crowd](https://huggingface.co/datasets/bond005/sberdevices_golos_10h_crowd)
- **Durations**: 0.7-27.5 seconds per sample
- **Test split**: 18.8K samples
- **Use case**: Short commands, quick smoke tests

```bash
uv run qwen-transkrib bench bond005/sberdevices_golos_10h_crowd -n 50 --backend gigaam
```

## Why these datasets?

- Verified transcriptions
- Publicly available on HuggingFace
- Cover different audio types (long/short, natural/commands)
- Used in published benchmarks
