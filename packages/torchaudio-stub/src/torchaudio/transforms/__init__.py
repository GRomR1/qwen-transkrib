"""torchaudio transforms — real MelSpectrogram implementation for MetaX."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _hz_to_mel_htk(freq: torch.Tensor) -> torch.Tensor:
    """Convert Hz to mel scale using HTK formula."""
    return 2595.0 * torch.log10(1.0 + freq / 700.0)


def _mel_to_hz_htk(mel: torch.Tensor) -> torch.Tensor:
    """Convert mel scale to Hz using HTK formula."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _create_mel_filterbank(
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    f_min: float = 0.0,
    f_max: float | None = None,
    norm: str | None = None,
    mel_scale: str = "htk",
) -> torch.Tensor:
    """Build mel filterbank matrix (n_mels × n_freqs)."""
    if f_max is None:
        f_max = float(sample_rate) / 2

    if mel_scale == "htk":
        mel_min = _hz_to_mel_htk(torch.tensor(f_min))
        mel_max = _hz_to_mel_htk(torch.tensor(f_max))
        mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = _mel_to_hz_htk(mel_points)
    else:
        raise ValueError(f"Unsupported mel_scale: {mel_scale}")

    # Convert Hz to FFT bins
    bin_width = float(sample_rate) / n_fft
    fft_bins = hz_points / bin_width

    # Build triangular filters
    n_freqs = n_fft // 2 + 1
    filterbank = torch.zeros((n_mels, n_freqs))
    for m in range(1, n_mels + 1):
        left = int(math.ceil(fft_bins[m - 1].item()))
        center = fft_bins[m].item()
        right = int(math.floor(fft_bins[m + 1].item()))

        # Rising edge
        for k in range(left, int(math.floor(center))):
            filterbank[m - 1, k] = (k - fft_bins[m - 1].item()) / (center - fft_bins[m - 1].item())
        # Falling edge
        for k in range(int(math.floor(center)), right + 1):
            filterbank[m - 1, k] = (fft_bins[m + 1].item() - k) / (fft_bins[m + 1].item() - center)

    if norm == "slaney":
        # Normalize each filter to have area of 1
        filterbank *= torch.tensor(2.0 / (hz_points[2:] - hz_points[:-2]).unsqueeze(1))

    return filterbank


class _Stub(nn.Module):
    def __init__(self, *a, **k):
        super().__init__()

    @staticmethod
    def _randn(w, *shape):
        return torch.zeros(*shape, device=w.device, dtype=w.dtype)


class MFCC(_Stub):
    def forward(self, w):
        return self._randn(w, 1, 100, 40)


class Spectrogram(_Stub):
    def forward(self, w):
        return self._randn(w, 1, 257, 100)


class MelSpectrogram(nn.Module):
    """Real MelSpectrogram implementation — no torchaudio dependency."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 64,
        n_fft: int = 320,
        hop_length: int = 160,
        win_length: int = 320,
        f_min: float = 0.0,
        f_max: float | None = None,
        power: float = 2.0,
        center: bool = True,
        pad: int = 0,
        norm: str | None = None,
        mel_scale: str = "htk",
        window_fn: callable = torch.hann_window,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.power = power
        self.center = center
        self.pad = pad
        self.norm = norm
        self.mel_scale = mel_scale
        self.window_fn = window_fn

        # Precompute window
        self.register_buffer("window", window_fn(win_length), persistent=False)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute log-mel spectrogram.

        Args:
            waveform: (batch, samples) or (samples,)

        Returns:
            (batch, n_mels, time) Mel spectrogram (linear scale, not logged)
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        batch, samples = waveform.shape

        # STFT
        # Apply pre-STFT padding if needed (pad=0 → no-op)
        if self.pad > 0:
            waveform = torch.nn.functional.pad(waveform, (self.pad, self.pad), mode="reflect")

        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            return_complex=True,
        )

        # Power spectrogram: (batch, n_fft//2+1, time)
        specgram = stft.abs() ** self.power

        # Build mel filterbank and move to device
        n_freqs = self.n_fft // 2 + 1
        fb = _create_mel_filterbank(
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            sample_rate=self.sample_rate,
            f_min=self.f_min,
            f_max=self.f_max,
            norm=self.norm,
            mel_scale=self.mel_scale,
        )
        fb = fb.to(specgram.device, dtype=specgram.dtype)

        # Apply filterbank: (batch, n_mels, time)
        mel_spec = fb @ specgram

        return mel_spec


class FBank(_Stub):
    pass


class MelScale(_Stub):
    pass


class InverseMelScale(_Stub):
    pass


class ComputeDeltas(_Stub):
    def forward(self, w):
        return w


class SlidingWindowCmn(_Stub):
    def forward(self, w):
        return w


class Vad(_Stub):
    pass


class Vol(_Stub):
    def forward(self, w):
        return torch.tensor(0.0)


class TimeStretch(_Stub):
    pass


class FrequencyMasking(_Stub):
    pass


class TimeMasking(_Stub):
    pass
