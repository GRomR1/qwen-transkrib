# Troubleshooting

## Common Issues

### "No module named 'torchaudio'"
**Cause**: torchaudio has no wheel for MetaX torch 2.10+metax.
**Solution**: This is expected. pyannote falls back to soundfile automatically. No action needed.

### "pyannote model requires authentication"
**Cause**: pyannote models are gated on HuggingFace.
**Solution**: Set `HF_TOKEN` environment variable:
```bash
export HF_TOKEN=hf_your_token_here
```
Get yours at: https://huggingface.co/settings/tokens

### "transformers version mismatch"
**Cause**: qwen-asr 0.0.6 requires transformers==4.57.6 exactly.
**Solution**: Do not upgrade transformers. Reinstall with exact pin:
```bash
uv sync  # or pip install transformers==4.57.6
```

### "CUDA out of memory"
**Cause**: Two large models (ASR + aligner) loaded simultaneously.
**Solution**: 
- Use `--device cpu` for small files
- Close other GPU applications
- Use the 0.6B model variant instead of 1.7B

### Diarization returns only 1 speaker
**Cause**: Wrong pyannote model version (3.1 instead of community-1).
**Solution**: Ensure using `pyannote/speaker-diarization-community-1`:
```bash
qwen-transkrib transcribe audio.wav --diarization-model pyannote/speaker-diarization-community-1
```

### "IndexError: Dimension out of range" with transformers
**Cause**: Known issue with transformers cache_utils and Qwen3-ASR.
**Solution**: Ensure transformers==4.57.6. If persists, check for conflicting patches.

### Audio format not supported
**Cause**: Missing ffmpeg or libsndfile.
**Solution**:
```bash
# Ubuntu/Debian
sudo apt install ffmpeg libsndfile1

# macOS
brew install ffmpeg libsndfile
```

## MetaX-Specific Issues

### GPU not detected
**Cause**: MACA drivers not installed or MXCD device not available.
**Solution**: Ensure MetaX drivers are installed and `/dev/mxcd` exists:
```bash
ls -la /dev/mxcd
```

### torch.cuda.is_available() returns False
**Cause**: torch not compiled with MetaX/MACA support.
**Solution**: Use MetaX-specific torch wheel:
```bash
pip install torch==2.10.0+metax3.8.1.0
```

### Slow performance on MetaX
**Cause**: flash-attn not installed or not working.
**Solution**: Install flash-attn for MetaX:
```bash
pip install flash-attn>=2.6
```

## Performance Tips

1. **Use flash-attn**: Reduces memory usage and speeds up inference.
2. **Batch processing**: Process multiple files in one session to amortize model loading.
3. **Shorter audio**: Split long files (>20 min) for better results.
4. **GPU memory**: Ensure 16GB+ VRAM for full pipeline (ASR + aligner + diarization).
