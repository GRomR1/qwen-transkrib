"""CLI integration tests."""

from __future__ import annotations

from typer.testing import CliRunner

from qwen_transkrib.cli import app

runner = CliRunner()


def test_help():
    """CLI shows help text."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Multilingual audio transcription" in result.output


def test_transcribe_help():
    """Transcribe command shows help."""
    result = runner.invoke(app, ["transcribe", "--help"])
    assert result.exit_code == 0
    assert "--language" in result.output
    assert "--diarize" in result.output
    assert "--format" in result.output


def test_info():
    """Info command runs successfully."""
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "Qwen Transkrib" in result.output
    assert "Python:" in result.output
    assert "PyTorch:" in result.output
