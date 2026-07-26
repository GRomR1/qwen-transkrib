"""Pytest configuration and fixtures."""

import logging
import os
import tempfile
from pathlib import Path

import pytest


def pytest_configure(config):
    """Configure logging for tests."""
    # Suppress transformers INFO/WARNING spam
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers.generation").setLevel(logging.ERROR)


def requires_gpu(func):
    """Decorator to skip test if no GPU is available."""
    return pytest.mark.skipif(
        not os.environ.get("CUDA_VISIBLE_DEVICES") and not os.path.exists("/dev/nvidia0"),
        reason="No GPU available"
    )(func)


@pytest.fixture
def short_audio_path():
    """Download a short audio sample from bond005/podlodka_speech for testing.

    Uses the first test sample (~8s) from the verified dataset.
    Caches the file in a temp directory for the test session.
    """
    import soundfile as sf
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    # Cache directory for test session
    cache_dir = Path(tempfile.gettempdir()) / "qwen_transkrib_test_fixtures"
    cache_dir.mkdir(exist_ok=True)

    cached_file = cache_dir / "podlodka_speech_sample_0.wav"
    if cached_file.exists():
        return str(cached_file)

    # Download and extract first sample from test set
    local = hf_hub_download(
        repo_id="bond005/podlodka_speech",
        filename="data/test-00000-of-00001.parquet",
        repo_type="dataset",
    )
    table = pq.read_table(local, columns=["audio"])

    audio_struct = table.column("audio")[0].as_py()
    audio_bytes = audio_struct.get("bytes")
    if not audio_bytes:
        pytest.skip("Could not extract audio from dataset")

    # Write to WAV atomically (temp file + rename)
    import io
    data, sr = sf.read(io.BytesIO(audio_bytes))
    temp_file = cache_dir / "podlodka_speech_sample_0.tmp.wav"
    try:
        sf.write(str(temp_file), data, sr)
        temp_file.replace(cached_file)
    finally:
        if temp_file.exists():
            temp_file.unlink()

    return str(cached_file)
