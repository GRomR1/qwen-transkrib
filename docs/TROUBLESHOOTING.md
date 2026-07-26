# Troubleshooting

## Installation

### `uv sync` fails with version conflicts
**Cause**: qwen-asr 0.0.6 requires exact versions.
**Solution**: Delete `.venv` and reinstall from scratch:
```bash
rm -rf .venv
uv sync
```

### Missing dependencies for GigaAM
**Cause**: GigaAM HF model requires hydra, omegaconf, sentencepiece.
**Solution**: Already included in pyproject.toml. Run `uv sync` to install.

### Native gigaam installation fails on MetaX
**Cause**: Native gigaam requires `onnxruntime==1.23.*` (NVIDIA CUDA), incompatible with MetaX.
**Solution**: Install with `--no-deps` and patch for compatibility:
```bash
uv pip install --no-deps "gigaam @ git+https://github.com/salute-developers/GigaAM.git"
uv run python patches/apply_gigaam_patch.py
```
The patch sets `strict=False` in the model loading code to handle checkpoint version mismatches.

## Runtime

### "pyannote model requires authentication"
**Cause**: pyannote models are gated on HuggingFace.
**Solution**: Set `HF_TOKEN` environment variable:
```bash
export HF_TOKEN=hf_your_token_here
```
Get yours at: https://huggingface.co/settings/tokens

### "CUDA out of memory"
**Cause**: Multiple large models loaded simultaneously.
**Solution**:
- Use `--backend gigaam` (240M params vs 1.7B)
- Use `--device cpu` for small files
- Close other GPU applications

### Diarization returns only 1 speaker
**Cause**: Wrong pyannote model version.
**Solution**: Use `pyannote/speaker-diarization-community-1` (default):
```bash
qwen-transkrib transcribe audio.wav --diarization-model pyannote/speaker-diarization-community-1
```

### GigaAM "Too long wav file" error
**Cause**: Audio chunk exceeds 25s limit.
**Solution**: Use VAD chunking (default) or reduce `--max-segment-sec`:
```bash
qwen-transkrib transcribe long_audio.wav --backend gigaam --vad
```

## MetaX GPU

### GPU not detected
**Cause**: MACA drivers not installed.
**Solution**: Ensure MetaX drivers are installed and `/dev/mxcd` exists:
```bash
ls -la /dev/mxcd
```

### Apply patches before first run
```bash
./apply_patches.sh
```

## Performance Tips

1. **Use GigaAM for Russian**: 240M params, faster, lower WER than Qwen3-ASR.
2. **Use flash-attn**: Reduces memory and speeds up inference.
3. **VAD enabled by default**: Splits long audio at natural pauses.
4. **GPU memory**: ~4 GB for GigaAM pipeline, ~7 GB for Qwen3 pipeline.
