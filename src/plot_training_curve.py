"""
src/plot_training_curve.py — plot train/val loss vs epoch from a training log.

Reads the CSV produced by src/train.py (epoch, train_loss, val_loss, is_best,
wall_clock_sec) and saves a PNG. Takes the CSV path as an argument so the
same script plots a local run's results/train_log.csv or one pulled back
from a Kaggle run without any code changes.

Usage:
    python -m src.plot_training_curve results/train_log.csv
    python -m src.plot_training_curve path/to/other_run_log.csv --output results/other_curve.png
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")  # write straight to a file, no display needed on a server/Kaggle
import matplotlib.pyplot as plt


def load_log(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no data rows yet (training may still be on epoch 1)")
    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [float(r["train_loss"]) for r in rows]
    val_loss = [float(r["val_loss"]) for r in rows]
    return epochs, train_loss, val_loss


def plot_training_curve(csv_path, output_path):
    epochs, train_loss, val_loss = load_log(csv_path)
    best_epoch = epochs[val_loss.index(min(val_loss))]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, marker="o", label="train loss")
    ax.plot(epochs, val_loss, marker="o", label="val loss")
    ax.axvline(best_epoch, color="gray", linestyle="--", alpha=0.6,
               label=f"best epoch ({best_epoch})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (log-magnitude MSE)")
    ax.set_title("Training curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"[plot_training_curve] saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot train/val loss vs epoch from a train_log.csv.")
    parser.add_argument("csv_path", help="Path to the training log CSV (written by src/train.py)")
    parser.add_argument("--output", default="results/training_curve.png",
                         help="Where to save the plot (default: results/training_curve.png)")
    args = parser.parse_args()
    plot_training_curve(args.csv_path, args.output)


if __name__ == "__main__":
    main()
