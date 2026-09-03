"""
src/evaluate.py — PESQ / STOI / SI-SDR evaluation over the full VoiceBank+DEMAND test set.

Run with:
    python -m src.evaluate --mode noisy       # unprocessed input, the floor
    python -m src.evaluate --mode baseline    # classical spectral subtraction
    python -m src.evaluate --mode model [--checkpoint checkpoints/unet_se.pt]

Runs on FULL-LENGTH test utterances (not fixed 2-second segments) because PESQ
and STOI are defined per-utterance and truncating changes the score. This
intentionally does NOT use NoisyCleanDataset (which crops/pads for batching);
evaluation processes one file at a time instead.
"""

import argparse
import csv
import os

import torch
import yaml
from tqdm import tqdm

from src.dsp import load_audio, stft, istft, magnitude_phase, reconstruct_complex, SAMPLE_RATE
from src.model import UNetSE
from src.baseline import spectral_subtraction
from src.losses import si_sdr_loss

try:
    from pesq import pesq
    from pystoi import stoi
except ImportError as e:
    raise ImportError(
        "pesq and pystoi are required for evaluation. Install with: "
        "pip install pesq pystoi"
    ) from e


def load_config(path="configs/config.yaml"):
    if os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f)
    print(f"[evaluate] WARNING: {path} not found, using built-in defaults.")
    return {}


def si_sdr(pred_wav, target_wav, eps=1e-8):
    """Scale-Invariant SDR in dB (higher is better) — the metric, not the loss.

    Reuses losses.si_sdr_loss (which returns -SI-SDR, since that's minimized
    during training) and flips the sign back to report the actual metric.
    """
    return -si_sdr_loss(pred_wav, target_wav, eps=eps).item()


def evaluate_pair(clean_wav, enhanced_wav, sr=SAMPLE_RATE):
    """Compute PESQ (wideband), STOI, and SI-SDR for one enhanced/clean pair.

    clean_wav, enhanced_wav: 1D torch tensors, same sample rate. Truncated to
    equal length before scoring — all three metrics require matching lengths.
    """
    min_len = min(clean_wav.shape[-1], enhanced_wav.shape[-1])
    clean_wav = clean_wav[:min_len]
    enhanced_wav = enhanced_wav[:min_len]

    clean_np = clean_wav.numpy()
    enhanced_np = enhanced_wav.numpy()

    try:
        pesq_score = pesq(sr, clean_np, enhanced_np, "wb")
        pesq_failed = False
    except Exception:
        # pesq raises on near-silent/degenerate clips rather than returning NaN
        pesq_score = None
        pesq_failed = True

    stoi_score = stoi(clean_np, enhanced_np, sr, extended=False)
    sisdr_score = si_sdr(enhanced_wav, clean_wav)

    return {"PESQ": pesq_score, "STOI": stoi_score, "SI-SDR": sisdr_score,
            "pesq_failed": pesq_failed}


def enhance_with_model(model, noisy_wav, device):
    spec = stft(noisy_wav.to(device))
    mag, phase = magnitude_phase(spec)
    mag_log = torch.log1p(mag).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        mask = model(mag_log).squeeze()

    enhanced_mag = mask * mag
    enhanced_spec = reconstruct_complex(enhanced_mag, phase)
    return istft(enhanced_spec, length=noisy_wav.shape[-1]).cpu()


def run_evaluation(enhance_fn, name, data_cfg, limit=None):
    """Walk the test set, apply `enhance_fn` to every noisy clip, score each
    result against the matching clean clip.

    enhance_fn: callable(noisy_wav: Tensor) -> enhanced_wav: Tensor, both 1D.
    name: label used for the printed summary and the output CSV filename
        (results/<name>_metrics.csv).
    limit: evaluate only the first N matched files, for a quick smoke run.

    Returns (rows, summary) — per-file dicts and the averaged summary dict.
    """
    noisy_test_dir = data_cfg.get("noisy_test_dir", "data/raw/noisy_testset_wav")
    clean_test_dir = data_cfg.get("clean_test_dir", "data/raw/clean_testset_wav")

    clean_files = sorted(os.listdir(clean_test_dir))
    noisy_files = set(os.listdir(noisy_test_dir))
    files = [f for f in clean_files if f in noisy_files]
    if limit is not None:
        files = files[:limit]

    rows = []
    pesq_skipped = 0
    for fname in tqdm(files, desc=f"evaluate[{name}]"):
        clean_wav = load_audio(os.path.join(clean_test_dir, fname))
        noisy_wav = load_audio(os.path.join(noisy_test_dir, fname))

        enhanced_wav = enhance_fn(noisy_wav)
        metrics = evaluate_pair(clean_wav, enhanced_wav)

        if metrics.pop("pesq_failed"):
            pesq_skipped += 1

        rows.append({"filename": fname, **metrics})

    summary = _summarize(rows, pesq_skipped, len(files))
    _print_summary(name, summary)
    _save_csv(rows, name)
    return rows, summary


def _summarize(rows, pesq_skipped, total_files):
    def avg(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    return {
        "PESQ": avg("PESQ"),
        "STOI": avg("STOI"),
        "SI-SDR": avg("SI-SDR"),
        "pesq_skipped": pesq_skipped,
        "total_files": total_files,
    }


def _print_summary(name, summary):
    print(f"\n{'=' * 50}")
    print(f"SUMMARY — {name}  ({summary['total_files']} files, "
          f"{summary['pesq_skipped']} PESQ skips)")
    print(f"{'=' * 50}")
    print(f"  PESQ:   {summary['PESQ']:.3f}")
    print(f"  STOI:   {summary['STOI']:.3f}")
    print(f"  SI-SDR: {summary['SI-SDR']:.3f} dB")


def _save_csv(rows, name, out_dir="results"):
    if not rows:
        return
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}_metrics.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[evaluate] per-file results saved to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate speech enhancement on the VoiceBank+DEMAND test set."
    )
    parser.add_argument("--mode", required=True, choices=["noisy", "baseline", "model"],
                         help="noisy = unprocessed input (the floor), "
                              "baseline = classical spectral subtraction, "
                              "model = trained U-Net checkpoint")
    parser.add_argument("--checkpoint", default=None,
                         help="Override checkpoint path from config (mode=model only)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Evaluate only the first N test files (for a quick run)")
    args = parser.parse_args()

    cfg = load_config()
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})

    if args.mode == "noisy":
        enhance_fn = lambda wav: wav

    elif args.mode == "baseline":
        enhance_fn = spectral_subtraction

    else:  # model
        checkpoint_path = args.checkpoint or train_cfg.get("checkpoint_path", "checkpoints/unet_se.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"No checkpoint found at '{checkpoint_path}'. Train a model first with "
                f"`python -m src.train`, or pass --checkpoint pointing to one."
            )

        base_ch = train_cfg.get("base_ch", 32)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = UNetSE(base_ch=base_ch).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        model.eval()
        print(f"[evaluate] loaded checkpoint: {checkpoint_path}")

        def enhance_fn(noisy_wav):
            return enhance_with_model(model, noisy_wav, device)

    run_evaluation(enhance_fn, name=args.mode, data_cfg=data_cfg, limit=args.limit)


if __name__ == "__main__":
    main()
