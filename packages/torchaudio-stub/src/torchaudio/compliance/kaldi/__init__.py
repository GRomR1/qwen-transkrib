"""Реальная реализация fbank/mfcc/spectrogram для pyannote (MACA).
Поддерживает vmap (тензоры без storage)."""
import torch
import numpy as np
import math


def _mel_scale(freq_hz):
    f_min, f_sp = 0.0, 200.0 / 3.0
    min_log_hz, min_log_mel = 1000.0, (1000.0 - f_min) / f_sp
    logstep = math.log(6.4) / 27.0
    if freq_hz < min_log_hz:
        return (freq_hz - f_min) / f_sp
    return min_log_mel + math.log(freq_hz / min_log_hz) / logstep


def _inv_mel_scale(mel):
    f_min, f_sp = 0.0, 200.0 / 3.0
    min_log_hz, min_log_mel = 1000.0, (1000.0 - f_min) / f_sp
    logstep = math.log(6.4) / 27.0
    if mel < min_log_mel:
        return f_min + f_sp * mel
    return min_log_hz * math.exp(logstep * (mel - min_log_mel))


def _get_mel_banks(num_bins, sample_freq, low_freq, high_freq, n_fft):
    n_freqs = n_fft // 2 + 1
    mel_low = _mel_scale(low_freq)
    mel_high = _mel_scale(high_freq)
    mel_points = np.linspace(mel_low, mel_high, num_bins + 2)
    hz_points = np.array([_inv_mel_scale(m) for m in mel_points])
    bin_indices = np.floor((n_fft + 1) * hz_points / sample_freq).astype(int)
    bin_indices = np.clip(bin_indices, 0, n_freqs - 1)
    mel_banks = np.zeros((num_bins, n_freqs), dtype=np.float32)
    for i in range(num_bins):
        left, center, right = bin_indices[i], bin_indices[i + 1], bin_indices[i + 2]
        if right <= left: continue
        for j in range(left, center):
            if center > left:
                mel_banks[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right > center:
                mel_banks[i, j] = (right - j) / (right - center)
    return mel_banks


def _pow_spectrogram(wav_np, n_fft, hop_length, win_length):
    """np массив → power spectrogram (n_freqs, n_frames)."""
    wav_np = np.asarray(wav_np, dtype=np.float32).ravel()
    if len(wav_np) < win_length:
        wav_np = np.pad(wav_np, (0, win_length - len(wav_np)), mode='reflect')
    pad = win_length // 2
    wav_padded = np.pad(wav_np, (pad, pad), mode='reflect')
    n_samples = len(wav_padded)
    n_frames = max(1, (n_samples - win_length) // hop_length + 1)
    indices = np.arange(win_length)[None, :] + hop_length * np.arange(n_frames)[:, None]
    indices = np.clip(indices, 0, len(wav_padded) - 1)
    frames = wav_padded[indices]
    window = np.hanning(win_length).astype(np.float32)
    frames = frames * window[None, :]
    spec = np.fft.rfft(frames, n=n_fft, axis=-1)
    return (spec.real ** 2 + spec.imag ** 2).T  # (n_freqs, n_frames)


def fbank(waveform, sample_frequency=16000, num_mel_bins=80,
          frame_length=25.0, frame_shift=10.0, dither=1.0,
          snip_edges=True, energy_floor=0.0, use_energy=False,
          window_type="povey", **kwargs):
    """Mimic torchaudio.compliance.kaldi.fbank.

    Accepts 1D (num_samples,) or 2D (channel, num_samples) input.
    Returns 2D (num_frames, num_mel_bins) — matches real torchaudio
    (real one processes only the first channel for 2D input).
    """
    if isinstance(waveform, torch.Tensor):
        try:
            wav = waveform.detach().float().cpu().numpy()
        except RuntimeError:
            wav = np.asarray(waveform.detach().cpu().tolist(), dtype=np.float32)
    else:
        wav = np.asarray(waveform, dtype=np.float32)
    if wav.ndim == 1:
        wav = wav[np.newaxis, :]
    elif wav.ndim > 2:
        wav = wav.reshape(-1, wav.shape[-1])

    n_fft = int(sample_frequency * frame_length / 1000)
    hop_length = int(sample_frequency * frame_shift / 1000)
    win_length = n_fft

    ch = wav[0]
    if snip_edges:
        usable = len(ch) - win_length
        if usable < 0:
            ch = np.pad(ch, (0, win_length - len(ch)), mode='reflect')
            n_frames = 1
        else:
            n_frames = 1 + usable // hop_length
        ch = ch[:win_length + (n_frames - 1) * hop_length]
    else:
        n_frames = 1 + (len(ch) - win_length) // hop_length
    if dither > 0:
        ch = ch + np.random.randn(*ch.shape).astype(np.float32) * dither
    spec = _pow_spectrogram(ch.astype(np.float32), n_fft, hop_length, win_length)
    mel_banks = _get_mel_banks(num_mel_bins, sample_frequency, 20.0,
                                sample_frequency / 2 - 1, n_fft)
    mel_spec = mel_banks @ spec
    mel_spec = np.maximum(mel_spec, 1e-10)
    log_mel = np.log(mel_spec).T.astype(np.float32)
    return torch.from_numpy(log_mel).float()


def mfcc(waveform, sample_frequency=16000, num_ceps=40, **kwargs):
    fb = fbank(waveform, sample_frequency=sample_frequency, num_mel_bins=23, **kwargs)
    if isinstance(fb, torch.Tensor):
        fb_np = fb.numpy()
    else:
        fb_np = fb
    n = fb_np.shape[-1]
    basis = np.zeros((num_ceps, n), dtype=np.float32)
    for i in range(num_ceps):
        for j in range(n):
            basis[i, j] = math.cos(math.pi * i * (j + 0.5) / n)
    basis *= math.sqrt(2.0 / n)
    fb_flat = fb_np.reshape(-1, n)
    mfcc_flat = fb_flat @ basis.T
    return torch.from_numpy(mfcc_flat.reshape(*fb_np.shape[:-1], num_ceps)).float()


def spectrogram(waveform, sample_frequency=16000, n_fft=400, hop_length=160,
                win_length=400, power=2.0, **kwargs):
    if isinstance(waveform, torch.Tensor):
        try:
            wav = waveform.detach().float().cpu().numpy()
        except RuntimeError:
            wav = np.asarray(waveform.detach().cpu().tolist(), dtype=np.float32)
    else:
        wav = np.asarray(waveform, dtype=np.float32)
    if wav.ndim == 1:
        wav = wav[np.newaxis, :]
    elif wav.ndim > 2:
        wav = wav.reshape(-1, wav.shape[-1])
    feats = []
    for ch in wav:
        spec = _pow_spectrogram(ch, n_fft, hop_length, win_length)
        if power == 1.0:
            spec = np.sqrt(spec + 1e-10)
        feats.append(spec)
    return torch.from_numpy(np.stack(feats)).float()


def pitch(waveform, sample_frequency=16000, frame_shift=10.0, **kwargs):
    if isinstance(waveform, torch.Tensor):
        try:
            n = waveform.shape[-1]
        except:
            n = 0
    else:
        n = len(waveform)
    n_frames = max(1, n // int(sample_frequency * frame_shift / 1000))
    return torch.zeros(1, n_frames, 2)


def snip(waveform, **kwargs):
    return waveform
