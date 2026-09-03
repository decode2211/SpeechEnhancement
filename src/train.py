"""
src/train.py — Training loop for the U-Net mask predictor.

Run with:  python -m src.train
Configure via configs/config.yaml (paths, batch size, epochs, lr).

Changes vs the README skeleton:
- Loads paths/hyperparameters from configs/config.yaml instead of hardcoding,
  so this file doesn't need to be edited when you move to a new machine.
- Creates checkpoints/ if it doesn't exist (torch.save fails otherwise).
- Adds a validation pass each epoch and only saves the checkpoint when val
  loss improves, so checkpoints/unet_se.pt is always your best model, not
  just whatever epoch happened to run last. Validation is a speaker-disjoint
  split carved out of the TRAINING set (src.data_loader.split_train_val_speakers),
  never the test directories — those are read only by evaluate.py, once, at
  the end. Using the test set for checkpoint selection would leak its score
  into which epoch gets picked as "best", biasing the final reported metrics.
- Adds tqdm progress bars — with ~11.5k training pairs, a silent loop with
  no feedback for minutes looks hung.
"""

import csv
import json
import os
import time

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_loader import NoisyCleanDataset, split_train_val_speakers
from src.model import UNetSE
from src.losses import combined_loss


def load_config(path="configs/config.yaml"):
    if os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f)
    print(f"[train] WARNING: {path} not found, using built-in defaults.")
    return {}


def run_epoch(model, loader, optimizer, device, si_sdr_weight, train=True):
    model.train() if train else model.eval()
    running_loss = 0.0

    with torch.set_grad_enabled(train):
        for noisy_mag, clean_mag, _noisy_phase in tqdm(loader, desc="train" if train else "val", leave=False):
            noisy_mag = noisy_mag.unsqueeze(1).to(device)   # (B,1,F,T), linear magnitude
            clean_mag = clean_mag.unsqueeze(1).to(device)

            # Network sees log-compressed input; mask is applied to LINEAR
            # magnitude (a 0-1 multiplicative mask is only meaningful there);
            # loss is then compared in log domain to match clean_mag's scale.
            mask = model(torch.log1p(noisy_mag))
            pred_mag = mask * noisy_mag

            loss = combined_loss(torch.log1p(pred_mag), torch.log1p(clean_mag), si_sdr_weight=si_sdr_weight)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

    return running_loss / max(len(loader), 1)


def main():
    cfg = load_config()
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})

    # Test directories are intentionally not read here — train.py never sees
    # them. They exist only for evaluate.py's one-time final scoring; reusing
    # them for validation/checkpoint-selection here would let that final
    # score leak into which epoch gets picked as "best".
    noisy_train_dir = data_cfg.get("noisy_train_dir", "data/raw/noisy_trainset_28spk_wav")
    clean_train_dir = data_cfg.get("clean_train_dir", "data/raw/clean_trainset_28spk_wav")

    segment_seconds = train_cfg.get("segment_seconds", 2.0)
    batch_size = train_cfg.get("batch_size", 8)
    epochs = train_cfg.get("epochs", 30)
    lr = train_cfg.get("lr", 1e-3)
    si_sdr_weight = train_cfg.get("si_sdr_weight", 0.0)
    base_ch = train_cfg.get("base_ch", 32)
    checkpoint_path = train_cfg.get("checkpoint_path", "checkpoints/unet_se.pt")
    log_path = train_cfg.get("log_path", "results/train_log.csv")
    num_workers = train_cfg.get("num_workers", 2)
    val_speakers_count = train_cfg.get("val_speakers", 2)
    val_split_seed = train_cfg.get("val_split_seed", 42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device: {device}")

    train_files, val_files, held_out_speakers = split_train_val_speakers(
        noisy_train_dir, clean_train_dir, num_val_speakers=val_speakers_count, seed=val_split_seed
    )
    print(f"[train] held-out val speakers (seed={val_split_seed}): {held_out_speakers}")

    val_split_path = os.path.join(os.path.dirname(log_path) or ".", "val_split.json")
    os.makedirs(os.path.dirname(val_split_path) or ".", exist_ok=True)
    with open(val_split_path, "w") as f:
        json.dump({
            "seed": val_split_seed,
            "num_val_speakers": val_speakers_count,
            "val_speakers": held_out_speakers,
            "train_files": len(train_files),
            "val_files": len(val_files),
        }, f, indent=2)

    train_dataset = NoisyCleanDataset(
        noisy_train_dir, clean_train_dir, segment_seconds=segment_seconds, train=True,
        file_list=train_files,
    )
    val_dataset = NoisyCleanDataset(
        noisy_train_dir, clean_train_dir, segment_seconds=segment_seconds, train=False,
        file_list=val_files,
    )
    print(f"[train] train pairs: {len(train_dataset)}, val pairs: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    model = UNetSE(base_ch=base_ch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    best_val_loss = float("inf")
    best_epoch = None

    with open(log_path, "w", newline="") as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(["epoch", "train_loss", "val_loss", "is_best", "wall_clock_sec"])
        run_start = time.time()

        for epoch in range(epochs):
            train_loss = run_epoch(model, train_loader, optimizer, device, si_sdr_weight, train=True)
            val_loss = run_epoch(model, val_loader, optimizer, device, si_sdr_weight, train=False)

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                torch.save(model.state_dict(), checkpoint_path)

            print(f"Epoch {epoch + 1}/{epochs} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}"
                  + ("  <- new best, saved checkpoint" if is_best else ""))

            log_writer.writerow([epoch + 1, train_loss, val_loss, is_best, round(time.time() - run_start, 1)])
            log_file.flush()  # survive a crash/kill mid-run with partial progress intact

    print(f"[train] done. Best val_loss: {best_val_loss:.4f} at epoch {best_epoch}. "
          f"Checkpoint: {checkpoint_path}. Log: {log_path}")


if __name__ == "__main__":
    main()