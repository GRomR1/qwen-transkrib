"""Pytest configuration and fixtures."""

import logging
import os

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
    """Path to test audio file."""
    return "tests/fixtures/ru_30s.wav"
