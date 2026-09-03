"""
src/baseline.py — Classical spectral subtraction baseline.

Gives a non-neural point of comparison for the U-Net (roadmap milestones
2 and 5): if the trained model can't beat this, something in training is
wrong before blaming architecture choices.
"""

import torch

from src.dsp import stft, istft, magnitude_phase, reconstruct_complex


def spectral_subtraction(noisy_wav, noise_estimate_frames=6, alpha=2.0, beta=0.01):
    """
    Estimate the noise spectrum from the first few frames (assumed noise-only,
    e.g. leading silence before speech starts) and subtract it from every frame.

    noisy_wav: 1D waveform tensor.
    noise_estimate_frames: how many leading STFT frames to average for the
        noise estimate. VoiceBank+DEMAND clips generally have a short silent
        lead-in, but this is a rough estimate, not ground truth — that's the
        whole point of comparing against a learned model.
    alpha: over-subtraction factor. Higher = more aggressive noise removal,
        more risk of "musical noise" artifacts.
    beta: spectral floor (fraction of original magnitude) to prevent negative
        magnitudes and reduce musical noise from over-subtraction.
    """
    spec = stft(noisy_wav)
    mag, phase = magnitude_phase(spec)

    n_frames = mag.shape[-1]
    n_estimate = min(noise_estimate_frames, n_frames)
    noise_mag = mag[..., :n_estimate].mean(dim=-1, keepdim=True)

    enhanced_mag = mag - alpha * noise_mag
    enhanced_mag = torch.clamp(enhanced_mag, min=beta * mag)

    enhanced_spec = reconstruct_complex(enhanced_mag, phase)
    return istft(enhanced_spec, length=noisy_wav.shape[-1])


# ---------------------------------------------------------------------------
# Smoke test — run `python -m src.baseline`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    sr = 16000
    t = torch.linspace(0, 2, sr * 2)
    clean = 0.5 * torch.sin(2 * torch.pi * 220 * t)
    # Zero out a short lead-in so the first few frames are noise-only, matching
    # the assumption spectral_subtraction's noise estimate depends on (real
    # VoiceBank+DEMAND clips have this silence naturally; a continuous tone
    # does not, which would make this self-test invalid).
    clean[: sr // 4] = 0.0
    noise = 0.3 * torch.randn_like(clean)
    noisy = clean + noise

    enhanced = spectral_subtraction(noisy)

    noisy_err = (noisy - clean).pow(2).mean().sqrt().item()
    enhanced_err = (enhanced - clean).pow(2).mean().sqrt().item()

    print(f"RMSE noisy vs clean:    {noisy_err:.4f}")
    print(f"RMSE enhanced vs clean: {enhanced_err:.4f}")
    assert enhanced_err < noisy_err, "Spectral subtraction should reduce error vs raw noisy signal"
    print("✅ Baseline reduces error relative to unprocessed noisy signal.")