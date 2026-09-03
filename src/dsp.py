"""
src/dsp.py — STFT / ISTFT utilities for the speech enhancement pipeline.

This module is deliberately the FIRST piece of the project to get right.
Everything downstream (dataset, model, training, inference) depends on
stft() and istft() being perfect inverses of each other. If the sanity
check at the bottom of this file fails, stop and fix this file before
touching anything else.
"""

import numpy as np
import soundfile as sf
import torch
import torchaudio

# ---------------------------------------------------------------------------
# Config (kept local to this module — configs/config.yaml can override later)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
N_FFT = 512          # ~32ms window at 16kHz
HOP_LENGTH = 128      # ~8ms hop (75% overlap)
WIN_LENGTH = 512

# NOTE: window must be created once and reused for both stft() and istft().
# If you ever create a new Hann window per call, minor floating point
# differences can break perfect reconstruction on some torch versions.
_window = torch.hann_window(WIN_LENGTH)


def load_audio(path, sr=SAMPLE_RATE):
    """
    Load a wav file, resample to `sr` if needed, and return mono waveform.

    Uses soundfile instead of torchaudio.load() on purpose: torchaudio >=2.9
    routes .load()/.save() through an optional `torchcodec` backend and
    raises ImportError if it isn't installed. soundfile is already a
    required dependency (requirements.txt) and has no extra backend to
    install, so this keeps setup simple and avoids a version-dependent
    crash on a fresh environment.
    """
    data, orig_sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)  # (channels, samples)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)  # collapse stereo -> mono

    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)

    return wav.squeeze(0)


def stft(wav, device=None):
    """Waveform (T,) -> complex spectrogram (F, T')."""
    if device is not None:
        wav = wav.to(device)
    window = _window.to(wav.device)
    return torch.stft(
        wav, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        window=window, return_complex=True
    )


def istft(spec, length=None):
    """Complex spectrogram (F, T') -> waveform (T,). `length` trims padding."""
    window = _window.to(spec.device)
    return torch.istft(
        spec, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        window=window, length=length
    )


def magnitude_phase(spec):
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    return mag, phase


def reconstruct_complex(mag, phase):
    return mag * torch.exp(1j * phase)


# ---------------------------------------------------------------------------
# Sanity check — run this file directly: `python -m src.dsp`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        wav_path = sys.argv[1]
        wav = load_audio(wav_path)
    else:
        # No file given: synthesize a test signal (sine + noise) so the
        # check still runs even before you've downloaded the dataset.
        print("No wav path given, generating a synthetic test signal instead.")
        t = torch.linspace(0, 2, int(SAMPLE_RATE * 2))
        wav = 0.5 * torch.sin(2 * torch.pi * 220 * t) + 0.05 * torch.randn_like(t)

    spec = stft(wav)
    recon = istft(spec, length=wav.shape[-1])

    max_err = (wav - recon).abs().max().item()
    is_close = torch.allclose(wav, recon, atol=1e-4)

    print(f"Input length:  {wav.shape[-1]} samples")
    print(f"Recon length:  {recon.shape[-1]} samples")
    print(f"Max abs error: {max_err:.2e}")
    print(f"allclose(atol=1e-4): {is_close}")

    if is_close:
        print("✅ STFT -> ISTFT reconstruction is lossless. Safe to proceed.")
    else:
        print("❌ Reconstruction failed. Check n_fft/hop_length/window settings "
              "before building anything else on top of this.")