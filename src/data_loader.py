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
import re

import torch
from torch.utils.data import Dataset

from src.dsp import load_audio, stft, magnitude_phase, SAMPLE_RATE

SPEAKER_RE = re.compile(r"^(p\d+)_")


def split_train_val_speakers(noisy_dir, clean_dir, num_val_speakers=2, seed=42):
    """
    Carve a held-out validation split out of a training directory pair by
    SPEAKER, not by file — VoiceBank+DEMAND filenames are p<speaker>_<utt>.wav,
    and utterances from the same speaker are acoustically correlated, so a
    per-file random split would leak speaker identity into "validation."

    Deterministic given the same (num_val_speakers, seed): sorts the speaker
    IDs first (directory listing order isn't guaranteed stable across
    filesystems) before sampling, so the same seed always holds out the same
    speakers.

    Returns (train_files, val_files, val_speakers) — plain filename lists
    (no directory prefix), val_speakers sorted for a readable, stable record
    of exactly what was held out.
    """
    noisy_files = set(os.listdir(noisy_dir))
    clean_files = set(os.listdir(clean_dir))
    common = sorted(noisy_files & clean_files)

    by_speaker = {}
    for fname in common:
        m = SPEAKER_RE.match(fname)
        if not m:
            raise ValueError(f"Filename doesn't match expected p<speaker>_<utt>.wav pattern: {fname}")
        by_speaker.setdefault(m.group(1), []).append(fname)

    speakers = sorted(by_speaker)
    if num_val_speakers >= len(speakers):
        raise ValueError(f"num_val_speakers={num_val_speakers} >= {len(speakers)} total speakers")

    val_speakers = sorted(random.Random(seed).sample(speakers, num_val_speakers))
    val_speaker_set = set(val_speakers)

    train_files = [f for s in speakers if s not in val_speaker_set for f in by_speaker[s]]
    val_files = [f for s in val_speakers for f in by_speaker[s]]

    return sorted(train_files), sorted(val_files), val_speakers


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
        file_list: optional explicit list of filenames to use instead of every
            matched file in noisy_dir/clean_dir — e.g. a speaker-disjoint
            subset from split_train_val_speakers(). When given, it's trusted
            as already-valid (present in both dirs); the missing-file check
            below only runs when discovering files from the directories.
    """

    def __init__(self, noisy_dir, clean_dir, segment_seconds=2.0, train=True, file_list=None):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.train = train
        self.segment_len = int(segment_seconds * SAMPLE_RATE)

        if file_list is not None:
            self.files = sorted(file_list)
        else:
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