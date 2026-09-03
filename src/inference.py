"""
src/inference.py — Run the trained model on a single noisy .wav file.

Usage:
    python -m src.inference --input noisy.wav --output enhanced.wav
    python -m src.inference --input noisy.wav --output enhanced.wav --checkpoint checkpoints/unet_se.pt
"""

import argparse
import os

import soundfile as sf
import torch
import yaml

from src.dsp import load_audio, stft, istft, magnitude_phase, reconstruct_complex, SAMPLE_RATE
from src.model import UNetSE


def load_config(path="configs/config.yaml"):
    if os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


def enhance_file(input_path, output_path, checkpoint=None, base_ch=None, device=None):
    cfg = load_config()
    train_cfg = cfg.get("train", {})

    checkpoint = checkpoint or train_cfg.get("checkpoint_path", "checkpoints/unet_se.pt")
    base_ch = base_ch or train_cfg.get("base_ch", 32)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"No checkpoint found at '{checkpoint}'. Train a model first with "
            f"`python -m src.train`, or pass --checkpoint pointing to one."
        )

    model = UNetSE(base_ch=base_ch).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    wav = load_audio(input_path).to(device)
    spec = stft(wav)
    mag, phase = magnitude_phase(spec)
    mag_log = torch.log1p(mag).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        mask = model(mag_log).squeeze()

    enhanced_mag = mask * mag
    enhanced_spec = reconstruct_complex(enhanced_mag, phase)
    enhanced_wav = istft(enhanced_spec, length=wav.shape[-1]).cpu()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    # soundfile, not torchaudio.save() — same torchcodec-backend issue as
    # load_audio() in dsp.py works around, but on the write side.
    sf.write(output_path, enhanced_wav.numpy(), SAMPLE_RATE)
    print(f"[inference] wrote enhanced audio to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Enhance a noisy speech file with the trained U-Net.")
    parser.add_argument("--input", required=True, help="Path to noisy input .wav")
    parser.add_argument("--output", required=True, help="Path to write enhanced .wav")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path from config")
    args = parser.parse_args()

    enhance_file(args.input, args.output, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()