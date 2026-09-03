"""
src/data_loader.py — PyTorch Dataset for paired (noisy, clean) speech.

Key design decision (see README §5.5 note): raw wav clips have variable
length. Rather than padding every spectrogram to the longest clip in the
dataset (wasteful) or writing a custom collate_fn (fiddly), this dataset
crops/pads each waveform to a FIXED number of samples *before* taking the
STFT. That way every item returned has identical tensor shape and the
default DataLoader collate_fn just works.
"""

import os
import random

import torch
from torch.utils.data import Dataset

from src.dsp import load_audio, stft, magnitude_phase, SAMPLE_RATE


class NoisyCleanDataset(Dataset):
    """
    Expects noisy_dir and clean_dir to contain files with MATCHING filenames
    (this is how VoiceBank+DEMAND ships). Run 01_dataset_audit.ipynb first to
    confirm there are zero filename mismatches before training.

    Args:
        noisy_dir, clean_dir: paths to the wav folders (str or Path).
        segment_seconds: fixed clip length in seconds. Clips longer than this
            are randomly cropped (train) or center-cropped (eval); clips
            shorter are zero-padded. 2.0s is a common default for VoiceBank+DEMAND.
        train: if True, crop randomly (data augmentation via random offset).
            If False, crop deterministically from the start — reproducible eval.
    """

    def __init__(self, noisy_dir, clean_dir, segment_seconds=2.0, train=True):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.train = train
        self.segment_len = int(segment_seconds * SAMPLE_RATE)

        noisy_files = set(os.listdir(noisy_dir))
        clean_files = set(os.listdir(clean_dir))
        common = sorted(noisy_files & clean_files)

        missing_noisy = clean_files - noisy_files
        missing_clean = noisy_files - clean_files
        if missing_noisy or missing_clean:
            print(f"[NoisyCleanDataset] WARNING: {len(missing_noisy)} files only in "
                  f"clean_dir, {len(missing_clean)} only in noisy_dir — skipping those, "
                  f"using {len(common)} matched pairs.")

        self.files = common

    def __len__(self):
        return len(self.files)

    def _fit_length(self, wav):
        """Crop or pad a 1D waveform tensor to exactly self.segment_len samples."""
        n = wav.shape[-1]
        if n == self.segment_len:
            return wav
        if n > self.segment_len:
            if self.train:
                start = random.randint(0, n - self.segment_len)
            else:
                start = 0
            return wav[start:start + self.segment_len]
        # n < segment_len: zero-pad at the end
        pad = self.segment_len - n
        return torch.nn.functional.pad(wav, (0, pad))

    def __getitem__(self, idx):
        fname = self.files[idx]
        noisy_wav = load_audio(os.path.join(self.noisy_dir, fname))
        clean_wav = load_audio(os.path.join(self.clean_dir, fname))

        # Align lengths BEFORE cropping so the same time offset in noisy/clean
        # corresponds to the same audio content.
        min_len = min(noisy_wav.shape[-1], clean_wav.shape[-1])
        noisy_wav, clean_wav = noisy_wav[:min_len], clean_wav[:min_len]

        # Use one shared random crop offset for both, not two independent ones.
        n = noisy_wav.shape[-1]
        if n > self.segment_len:
            start = random.randint(0, n - self.segment_len) if self.train else 0
            noisy_wav = noisy_wav[start:start + self.segment_len]
            clean_wav = clean_wav[start:start + self.segment_len]
        elif n < self.segment_len:
            pad = self.segment_len - n
            noisy_wav = torch.nn.functional.pad(noisy_wav, (0, pad))
            clean_wav = torch.nn.functional.pad(clean_wav, (0, pad))

        noisy_spec = stft(noisy_wav)
        clean_spec = stft(clean_wav)

        noisy_mag, noisy_phase = magnitude_phase(noisy_spec)
        clean_mag, _ = magnitude_phase(clean_spec)

        # Linear magnitudes: the mask is only meaningful multiplied onto linear
        # magnitude, so log-compression happens downstream (see train.py /
        # inference.py), not here.
        return noisy_mag, clean_mag, noisy_phase

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    ds = NoisyCleanDataset(
        "data/raw/noisy_trainset_28spk_wav",
        "data/raw/clean_trainset_28spk_wav",
        segment_seconds=2.0,
        train=True,
    )
    print(f"Dataset size: {len(ds)} pairs")

    loader = DataLoader(ds, batch_size=4, shuffle=True)
    noisy_mag, clean_mag, noisy_phase = next(iter(loader))
    print("noisy_mag  shape:", noisy_mag.shape)   # (B, F, T), linear magnitude
    print("clean_mag  shape:", clean_mag.shape)
    print("noisy_phase shape:", noisy_phase.shape)
    assert noisy_mag.shape == clean_mag.shape, "noisy/clean mag shapes must match!"
    print("✅ Batch shapes are consistent — DataLoader is ready for training.")