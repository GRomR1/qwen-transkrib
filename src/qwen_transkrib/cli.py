"""CLI interface for qwen-transkrib."""

from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path

from qwen_transkrib.normalize import TextNormalizer

# Suppress vllm INFO/DEBUG spam - set env BEFORE any imports
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

# Suppress transformers logging spam
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Set root logger to WARNING to suppress all INFO/WARNING from libraries
logging.getLogger().setLevel(logging.WARNING)

# Set specific loggers to WARNING
for name in ("vllm", "transformers", "transformers.generation", "transformers.generation.utils"):
    logging.getLogger(name).setLevel(logging.ERROR)
    logging.getLogger(name).propagate = False

# Suppress known harmless warnings from third-party libraries
warnings.filterwarnings("ignore", message=".*degrees of freedom.*")
warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")
warnings.filterwarnings("ignore", message=".*Transformers v4.*")
warnings.filterwarnings("ignore", message=".*generation flags.*")
warnings.filterwarnings("ignore", message=".*pad_token_id.*")
warnings.filterwarnings("ignore", message=".*flash_attn.*")
warnings.filterwarnings("ignore", message=".*memory efficient attention.*")
warnings.filterwarnings("ignore", message=".*invalid escape sequence.*", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Load .env file if present (for HF_TOKEN etc.)
load_dotenv()

from qwen_transkrib import __version__
from qwen_transkrib.asr import create_backend, transcribe_file, transcribe_file_vad
from qwen_transkrib.config import Settings
from qwen_transkrib.correct import CorrectionModel
from qwen_transkrib.diarize import diarize_file
from qwen_transkrib.merge import assign_speakers
from qwen_transkrib.punctuate import PunctuationModel
from qwen_transkrib.schemas import TranscriptionResult, Word
from qwen_transkrib.writers import get_writer

# Cache for models (loaded once per session)
_punct_cache: dict[str, PunctuationModel] = {}
_correct_cache: dict[str, CorrectionModel] = {}


def _get_punct_model(language: str = "Russian") -> PunctuationModel:
    """Get or create cached punctuation model for a language."""
    if language not in _punct_cache:
        _punct_cache[language] = PunctuationModel(language=language, device="cpu")
    return _punct_cache[language]


def _get_correct_model(device: str = "cuda:0") -> CorrectionModel:
    """Get or create cached correction model."""
    if device not in _correct_cache:
        _correct_cache[device] = CorrectionModel(device=device)
    return _correct_cache[device]

app = typer.Typer(
    add_completion=False,
    help="Audio transcription with speaker diarization. Supports Qwen3-ASR and GigaAM v3 backends.",
    no_args_is_help=True,
)
console = Console()


def _parse_glossary(raw: str) -> dict[str, str]:
    """Parse 'src=dst,src2=dst2' glossary string into a dict."""
    glossary: dict[str, str] = {}
    if not raw.strip():
        return glossary
    for pair in raw.split(","):
        pair = pair.strip()
        if "=" in pair:
            src, dst = pair.split("=", 1)
            glossary[src.strip()] = dst.strip()
    return glossary


@app.command()
def transcribe(
    input: Path = typer.Argument(
        ..., help="Path to audio file (wav, mp3, webm)", exists=True, readable=True
    ),
    output_dir: Path = typer.Option(Path("./out"), "-o", "--output-dir", help="Output directory"),
    language: str = typer.Option("Russian", "-l", "--language", help="Audio language (Title Case)"),
    model: str = typer.Option("Qwen/Qwen3-ASR-1.7B", help="ASR model name (Qwen3 only)"),
    aligner: str = typer.Option("Qwen/Qwen3-ForcedAligner-0.6B", help="Forced aligner model"),
    punct_flag: bool = typer.Option(
        True, "--punct/--no-punct", help="Restore punctuation"
    ),
    correct_flag: bool = typer.Option(
        False, "--correct/--no-correct",
        help="Experimental: LLM post-processing to fix ASR errors (slow, GPU recommended)",
    ),
    normalize_flag: bool = typer.Option(
        True, "--normalize/--no-normalize",
        help="Number normalization (пять → 5) and glossary post-processing",
    ),
    glossary: str = typer.Option(
        "",
        "--glossary",
        help="Comma-separated term corrections (e.g. 'гугл=Google,майкрософт=Microsoft')",
    ),
    vad_flag: bool = typer.Option(
        True, "--vad/--no-vad",
        help="VAD-based chunking: split audio at pause boundaries (recommended)"
    ),
    diarize_flag: bool = typer.Option(
        True, "--diarize/--no-diarize", help="Enable speaker diarization"
    ),
    diarization_model: str = typer.Option(
        "pyannote/speaker-diarization-community-1",
        help="Diarization model (do NOT change to 3.1/3.0 — broken on MetaX)",
    ),
    fmt: str = typer.Option("srt,json,txt", "--format", help="Comma-separated output formats"),
    device: str = typer.Option("cuda:0", "--device", help="Device (cuda:0, cpu)"),
    context: str = typer.Option(
        "",
        "--context",
        help="Comma-separated hotwords for term correction (e.g. 'Google, Microsoft, Amazon')",
    ),
    backend: str = typer.Option(
        "asr", "--backend", "-b",
        help="ASR backend: 'asr' (Qwen3-ASR) or 'gigaam' (e2e_rnnt w/ built-in punctuation)",
    ),
) -> None:
    """Transcribe audio with optional speaker diarization."""
    # Check HF_TOKEN early for faster failure
    if diarize_flag and not os.environ.get("HF_TOKEN"):
        console.print(
            "[yellow]Warning: HF_TOKEN not set. "
            "Diarization requires HuggingFace auth token.[/yellow]"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    settings = Settings(
        asr_model=model,
        aligner_model=aligner,
        language=language,
        device=device,
        diarization_model=diarization_model,
    )

    # Parse comma-separated format string
    formats = [f.strip() for f in fmt.split(",")]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        if backend == "gigaam":
            # ---- GigaAM path (e2e_rnnt with built-in punctuation) ----
            from qwen_transkrib.asr import create_backend
            asr = create_backend("gigaam")

            task = progress.add_task("Transcribing (GigaAM e2e_rnnt)...", total=None)
            words, full_text = asr.transcribe_words(str(input))
            detected_language = language
            all_words = words
            progress.update(task, description="Transcription complete")

            # GigaAM e2e_rnnt already includes punctuation — skip if user didn't explicitly opt out
            punct_text = full_text

            # Number normalization + glossary (lightweight, 0.5-2pp WER improvement)
            if normalize_flag:
                norm = TextNormalizer(glossary=_parse_glossary(glossary))
                punct_text = norm.normalize(punct_text)

            # LLM correction (optional)
            if correct_flag:
                task = progress.add_task("Correcting ASR errors...", total=None)
                correct_model = _get_correct_model(device)
                punct_text = correct_model.correct(punct_text)
                progress.update(task, description="ASR errors corrected")

        else:
            # ---- Qwen3-ASR path (original) ----
            if vad_flag:
                task = progress.add_task("Transcribing (VAD chunking)...", total=None)
                words, full_text, detected_language = transcribe_file_vad(
                    input, settings, context=context
                )
                progress.update(task, description="VAD transcription complete")
            else:
                task = progress.add_task("Transcribing audio...", total=None)
                words, full_text, detected_language = transcribe_file(
                    input, settings, context=context
                )
                progress.update(task, description="Transcription complete")
            all_words = words

            # Step 2: Restore punctuation
            punct_text = full_text
            if punct_flag:
                task = progress.add_task("Restoring punctuation...", total=None)
                punct_model = _get_punct_model(language)
                punct_text = punct_model.restore(full_text)
                progress.update(task, description="Punctuation restored")

            # Number normalization + glossary
            if normalize_flag:
                norm = TextNormalizer(glossary=_parse_glossary(glossary))
                punct_text = norm.normalize(punct_text)

            # Step 2.5: LLM error correction
            if correct_flag:
                task = progress.add_task("Correcting ASR errors...", total=None)
                correct_model = _get_correct_model(device)
                punct_text = correct_model.correct(punct_text)
                progress.update(task, description="ASR errors corrected")

            # Step 3: Diarize (Qwen3 always has proper word timestamps)
            if diarize_flag and not words:
                diarize_flag = False

        # Step 4: Diarize (shared for both backends)
        speakers: list[str] = []
        if diarize_flag:
            task = progress.add_task("Running speaker diarization...", total=None)
            diar = diarize_file(input, settings)
            progress.update(task, description="Diarization complete")
        else:
            diar = None

        # Step 5: Merge
        if diar is not None and not diar.empty and all_words:
            task = progress.add_task("Assigning speakers to words...", total=None)
            segments = assign_speakers(all_words, diar, gap_threshold=settings.gap_threshold)
            speakers = sorted(set(s.speaker for s in segments))
            progress.update(task, description="Speaker assignment complete")

            # Apply punctuation to each segment if not already punctuated
            if backend != "gigaam" and punct_flag:
                for seg in segments:
                    seg.text = punct_model.restore(seg.text)

            if correct_flag:
                for seg in segments:
                    seg.text = correct_model.correct(seg.text)
        else:
            from qwen_transkrib.schemas import Segment

            segments = (
                [
                    Segment(
                        start=all_words[0].start if all_words else 0,
                        end=all_words[-1].end if all_words else 0,
                        speaker="SPEAKER_00",
                        text=punct_text,
                        words=all_words,
                    )
                ]
                if all_words
                else []
            )
            speakers = ["SPEAKER_00"] if segments else []

        # Build result
        result = TranscriptionResult(
            audio_path=str(input.resolve()),
            duration_sec=all_words[-1].end if all_words else 0.0,
            language=detected_language,
            model=settings.asr_model if backend != "gigaam" else "GigaAM-v3 (e2e_rnnt)",
            diarization_model=diarization_model if diarize_flag else None,
            speakers=speakers,
            segments=segments,
            full_text=punct_text,
        )

        # Step 5: Write outputs
        for f in formats:
            task = progress.add_task(f"Writing {f.upper()}...", total=None)
            writer = get_writer(f)
            out_path = output_dir / f"{input.stem}.{f}"
            writer(result, out_path)
            progress.update(task, description=f"Written {out_path}")

    console.print(f"\n[green]Done![/green] Output files in {output_dir}/")
    for f in formats:
        console.print(f"  - {input.stem}.{f}")


@app.command()
def info() -> None:
    """Show environment info: Python, torch, GPU, models."""
    import sys

    import torch

    console.print("[bold]Qwen Transkrib[/bold] v" + __version__)
    console.print()

    # Python
    console.print(f"[bold]Python:[/bold] {sys.version}")

    # PyTorch
    console.print(f"[bold]PyTorch:[/bold] {torch.__version__}")

    # CUDA / MACA
    if torch.cuda.is_available():
        console.print("[bold]CUDA:[/bold] Available")
        console.print(f"[bold]GPU:[/bold] {torch.cuda.get_device_name(0)}")
        try:
            props = torch.cuda.get_device_properties(0)
            vram = getattr(props, "total_mem", None) or getattr(props, "total_memory", None)
            if vram:
                console.print(f"[bold]VRAM:[/bold] {vram / 1e9:.1f} GB")
        except Exception:
            pass
    else:
        console.print("[bold]CUDA:[/bold] Not available (CPU mode)")

    # Key packages
    console.print()
    console.print("[bold]Key packages:[/bold]")
    try:
        from importlib.metadata import version as pkg_version

        console.print(f"  qwen-asr: {pkg_version('qwen-asr')}")
    except Exception:
        console.print("  qwen-asr: [red]not installed[/red]")

    try:
        import pyannote.audio

        console.print(f"  pyannote-audio: {pyannote.audio.__version__}")
    except ImportError:
        console.print("  pyannote-audio: [red]not installed[/red]")

    try:
        import transformers

        console.print(f"  transformers: {transformers.__version__}")
    except ImportError:
        console.print("  transformers: [red]not installed[/red]")


@app.command()
def bench(
    dataset: str = typer.Argument(
        ...,
        help="HF dataset name (e.g. bond005/sberdevices_golos_10h_crowd)",
    ),
    split: str = typer.Option("test", "--split", "-s", help="Dataset split"),
    max_samples: int | None = typer.Option(
        None, "--max-samples", "-n", help="Limit samples (quick test)",
    ),
    model: str = typer.Option("Qwen/Qwen3-ASR-1.7B", help="Qwen3-ASR model name (ignored when --backend gigaam)"),
    device: str = typer.Option("cuda:0", "--device", help="Device"),
    language: str = typer.Option("Russian", "-l", "--language"),
    backend: str = typer.Option(
        "asr", "--backend", "-b",
        help="ASR backend: 'asr' (Qwen3-ASR) or 'gigaam'",
    ),
) -> None:
    """Benchmark ASR WER against a reference dataset from HuggingFace."""
    from qwen_transkrib.bench import run_bench
    from qwen_transkrib.asr import create_backend
    from qwen_transkrib.config import Settings

    if backend == "gigaam":
        asr = create_backend("gigaam")
        console.print(f"Benchmark: [bold]{dataset}[/bold] (split={split})")
        console.print(f"ASR: GigaAM-v3 (e2e_rnnt) | device={device}")
    else:
        settings = Settings(
            asr_model=model,
            language=language,
            device=device,
        )
        asr = create_backend("asr", settings)
        console.print(f"Benchmark: [bold]{dataset}[/bold] (split={split})")
        console.print(f"ASR: {model} | device={device} | language={language}")

    def _transcribe(path: str) -> str:
        return asr.transcribe(path)

    if max_samples:
        console.print(f"Samples: {max_samples} (limited)")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading dataset & transcribing...", total=None)
        results = run_bench(dataset, split, _transcribe, max_samples)
        progress.update(task, description="Done")

    console.print()
    console.print("[bold]Results:[/bold]")
    console.print(f"  WER:      [cyan]{results['wer']:.2f}%[/cyan]")
    console.print(f"  Sub:      {results['substitutions']}")
    console.print(f"  Ins:      {results['insertions']}")
    console.print(f"  Del:      {results['deletions']}")
    console.print(f"  Ref words: {results['ref_words']}")
    console.print(f"  Samples:  {results['samples']}")


if __name__ == "__main__":
    app()
