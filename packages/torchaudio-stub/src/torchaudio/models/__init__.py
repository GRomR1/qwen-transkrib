"""torchaudio.models stub for MetaX.

torchaudio is stubbed on MetaX GPUs (no native build). This package only exists
so that `from torchaudio import models` succeeds — it is imported unconditionally
by pyannote.audio (see models/segmentation/SSeRiouSS.py). The wav2vec2 path is not
used by the default diarization pipeline; if it is ever reached, fail loudly with a
clear message instead of a confusing AttributeError.
"""

from __future__ import annotations


def wav2vec2_model(*args, **kwargs):  # noqa: ANN002, ANN003, ANN201
    raise NotImplementedError(
        "torchaudio.models.wav2vec2_model is not available: torchaudio is a stub on "
        "MetaX GPUs. This pyannote model (SSeRiouSS/wav2vec2) is unsupported here; "
        "use the default diarization pipeline instead."
    )
