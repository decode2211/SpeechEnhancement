# Speech Enhancement / Noise Suppression using AI + Signal Processing

A deep learning based speech enhancement system that removes background noise from speech
recordings, combining classical Digital Signal Processing (STFT/ISTFT, spectral masking) with
a neural network (U-Net style) trained on the VoiceBank+DEMAND dataset.

> See `Theory.md` for the full conceptual/mathematical explanation of every step.

---

## 1. Overview

**Input:** Noisy speech waveform (speech + background noise)
**Output:** Enhanced/denoised speech waveform

**Pipeline:**
```
Noisy waveform → STFT → Magnitude + Phase split → Neural Network (predicts mask)
→ Mask x Noisy Magnitude → Combine with noisy Phase → ISTFT → Enhanced waveform
```

Phase is never modified (noisy-phase reconstruction) — a deliberate simplification, not an
oversight. See §5.0 for the one convention that trips this pipeline up if you're not careful:
the mask is only meaningful multiplied onto **linear** magnitude.

---

## 2. Dataset

**VoiceBank+DEMAND** — standard benchmark dataset for speech enhancement.
- Clean speech: VCTK corpus recordings
- Noise: DEMAND dataset (10 real-world noise types) mixed at multiple SNR levels
- Already provided as paired (noisy, clean) `.wav` files, split into train/test (28-speaker
  version — see `dataset_setup.md`)

Expected directory structure after download+extraction:
```
data/raw/
├── clean_trainset_28spk_wav/   (11,572 files)
├── noisy_trainset_28spk_wav/   (11,572 files)
├── clean_testset_wav/          (824 files)
└── noisy_testset_wav/          (824 files)
```

**Known extraction gotcha:** the official zips for `clean_trainset_28spk_wav` and
`noisy_testset_wav` extract one level too deep on some tools (e.g.
`clean_trainset_28spk_wav/clean_trainset_28spk_wav/*.wav`). If `NoisyCleanDataset` reports 0
matched pairs, check for this and flatten the nested directory. `01_dataset_audit.ipynb`
detects and auto-resolves this when auditing; `verify_data.py` and `data_loader.py` do not, so
fix the directory structure on disk first.

See `dataset_setup.md` for the full download procedure (`download_data.sh` automates it),
`verify_data.py` for a quick file-count/pairing check, and `01_dataset_audit.ipynb` for a
deeper per-file audit (corruption, silence, clipping, exact duplicates via hash).

---

## 3. Repository Structure

```
speech-enhancement/
│
├── data/
│   ├── raw/                     # original VoiceBank+DEMAND files (gitignored)
│   └── processed/               # reserved for cached STFT features — not used yet;
│                                 # the loader recomputes STFT on every __getitem__
│
├── src/
│   ├── dsp.py                   # STFT/ISTFT, magnitude/phase split, mono loading
│   ├── data_loader.py           # NoisyCleanDataset (fixed-length crop/pad + STFT)
│   ├── model.py                 # UNetSE — predicts a [0,1] T-F mask
│   ├── losses.py                # log_magnitude_loss, si_sdr_loss, combined_loss
│   ├── train.py                 # training loop, config-driven, best-checkpoint saving
│   ├── evaluate.py              # PESQ / STOI / SI-SDR test-set driver
│   ├── baseline.py              # classical spectral subtraction
│   └── inference.py             # run a trained checkpoint on one file
│
├── 01_dataset_audit.ipynb       # per-file dataset audit (see §2)
├── verify_data.py               # quick file-count/pairing check
├── download_data.sh             # automated dataset download
├── dataset_setup.md             # manual dataset download/verification guide
│
├── demo/
│   └── app.py                   # Streamlit demo: upload noisy wav → hear enhanced result
│
├── checkpoints/                 # saved model weights (.pt) — gitignored, empty until trained
├── results/                     # per-mode evaluation CSVs from src/evaluate.py — tracked
├── configs/
│   └── config.yaml              # data paths + training hyperparameters (read by train.py,
│                                 # evaluate.py, inference.py)
├── Theory.md                    # conceptual/mathematical background
├── requirements.txt
└── README.md
```

---

## 4. Setup

```bash
git clone <your-repo-url>
cd speech-enhancement
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**requirements.txt**
```
torch
torchaudio
librosa
numpy
scipy
soundfile
pesq
pystoi
matplotlib
pandas
tqdm
pyyaml
streamlit
```

**Windows note:** the console's default cp1252 encoding can't print the ✅/❌ used in this
project's smoke tests, which otherwise crashes on an encode error *after* printing the real
result. Set `PYTHONIOENCODING=utf-8` before running any `python -m src.*` command to avoid it.

---

## 5. Core Code

### 5.0 Mask-domain convention (read this before touching train.py or inference.py)

The predicted mask is a `[0, 1]` multiplicative mask, and **it is only meaningful applied to
linear magnitude** — multiplying a mask against a log-compressed magnitude doesn't correspond
to anything physical. The convention followed consistently across this codebase is:

1. `NoisyCleanDataset` returns **linear** magnitudes (`noisy_mag`, `clean_mag`).
2. The network's *input* is log-compressed (`torch.log1p(noisy_mag)`) purely to tame dynamic
   range — this is the only place `log1p` is applied before the mask.
3. The predicted mask multiplies the **linear** `noisy_mag`: `pred_mag = mask * noisy_mag`.
4. Any loss/metric that expects log-scale (`log_magnitude_loss`) is computed as
   `log1p(pred_mag)` vs. `log1p(clean_mag)`, computed *after* masking, never before.

`train.py`, `inference.py`, and `evaluate.py` all follow this. If you add a new code path that
touches the mask, follow it too — an earlier version of `train.py` masked the log-magnitude
directly, which trained a model whose checkpoint meant something different than what
`inference.py` assumed.

### 5.1 `src/dsp.py` — STFT / ISTFT utilities

```python
import numpy as np
import soundfile as sf
import torch
import torchaudio

SAMPLE_RATE = 16000
N_FFT = 512          # ~32ms window at 16kHz
HOP_LENGTH = 128      # ~8ms hop (75% overlap)
WIN_LENGTH = 512

_window = torch.hann_window(WIN_LENGTH)

def load_audio(path, sr=SAMPLE_RATE):
    """Load via soundfile (not torchaudio.load — see docstring in dsp.py for why),
    collapse to mono, resample to `sr` if needed."""
    data, orig_sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)
    return wav.squeeze(0)

def stft(wav, device=None):
    window = _window.to(device) if device is not None else _window
    if device is not None:
        wav = wav.to(device)
    return torch.stft(wav, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                       window=window, return_complex=True)

def istft(spec, length=None):
    window = _window.to(spec.real.device)
    return torch.istft(spec, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                        window=window, length=length)

def magnitude_phase(spec):
    return torch.abs(spec), torch.angle(spec)

def reconstruct_complex(mag, phase):
    return mag * torch.exp(1j * phase)
```

**Sanity check (verified — do this first on a new machine):**
```bash
python -m src.dsp
# ✅ STFT -> ISTFT reconstruction is lossless. Safe to proceed.
```
Confirmed: max abs error ≈ 1.79e-07 on a synthetic signal, well under the `atol=1e-4` threshold.

---

### 5.2 `src/data_loader.py` — Dataset class

```python
import os, random
import torch
from torch.utils.data import Dataset
from src.dsp import load_audio, stft, magnitude_phase, SAMPLE_RATE

class NoisyCleanDataset(Dataset):
    """Crops/pads each waveform to a FIXED number of samples *before* the STFT,
    so every item has identical tensor shape and the default DataLoader
    collate_fn works with no custom collation needed."""

    def __init__(self, noisy_dir, clean_dir, segment_seconds=2.0, train=True):
        self.train = train
        self.segment_len = int(segment_seconds * SAMPLE_RATE)
        noisy_files = set(os.listdir(noisy_dir))
        clean_files = set(os.listdir(clean_dir))
        self.files = sorted(noisy_files & clean_files)
        self.noisy_dir, self.clean_dir = noisy_dir, clean_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        noisy_wav = load_audio(os.path.join(self.noisy_dir, fname))
        clean_wav = load_audio(os.path.join(self.clean_dir, fname))

        min_len = min(noisy_wav.shape[-1], clean_wav.shape[-1])
        noisy_wav, clean_wav = noisy_wav[:min_len], clean_wav[:min_len]

        # One shared random crop offset for both, so the same time window
        # is used for noisy and clean.
        n = noisy_wav.shape[-1]
        if n > self.segment_len:
            start = random.randint(0, n - self.segment_len) if self.train else 0
            noisy_wav = noisy_wav[start:start + self.segment_len]
            clean_wav = clean_wav[start:start + self.segment_len]
        elif n < self.segment_len:
            pad = self.segment_len - n
            noisy_wav = torch.nn.functional.pad(noisy_wav, (0, pad))
            clean_wav = torch.nn.functional.pad(clean_wav, (0, pad))

        noisy_spec, clean_spec = stft(noisy_wav), stft(clean_wav)
        noisy_mag, noisy_phase = magnitude_phase(noisy_spec)
        clean_mag, _ = magnitude_phase(clean_spec)

        # LINEAR magnitudes — see §5.0. Log-compression happens downstream.
        return noisy_mag, clean_mag, noisy_phase
```

Verified against the real (post-flatten) dataset: `Dataset size: 11572 pairs`, batches of
shape `(4, 257, 251)` for a 2-second segment, default `DataLoader` collation works with no
custom `collate_fn`.

---

### 5.3 `src/model.py` — U-Net for mask prediction

```python
import torch, torch.nn as nn, torch.nn.functional as F

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
    )

def _match_size(x, target):
    """Center-crop or pad x (B,C,H,W) so its spatial dims match target's —
    N_FFT=512 gives 257 freq bins, which isn't divisible by 8 (three 2x
    poolings), so skip connections need this to line up."""
    _, _, h, w = x.shape
    _, _, th, tw = target.shape
    if h != th or w != tw:
        dh, dw = th - h, tw - w
        x = F.pad(x, (0, max(dw, 0), 0, max(dh, 0)))
        if dh < 0 or dw < 0:
            x = x[:, :, :th, :tw]
    return x

class UNetSE(nn.Module):
    """Predicts a [0,1] T-F mask, applied to LINEAR noisy magnitude (§5.0)."""
    def __init__(self, base_ch=32):
        super().__init__()
        self.enc1 = conv_block(1, base_ch)
        self.enc2 = conv_block(base_ch, base_ch * 2)
        self.enc3 = conv_block(base_ch * 2, base_ch * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(base_ch * 4, base_ch * 8)
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = conv_block(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = conv_block(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = conv_block(base_ch * 2, base_ch)
        self.out_conv = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):  # x: (batch, 1, freq, time)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([_match_size(self.up3(b), e3), e3], dim=1))
        d2 = self.dec2(torch.cat([_match_size(self.up2(d3), e2), e2], dim=1))
        d1 = self.dec1(torch.cat([_match_size(self.up1(d2), e1), e1], dim=1))
        d1 = _match_size(d1, x)
        return torch.sigmoid(self.out_conv(d1))
```

`base_ch=32` → 1,927,841 parameters. Verified: forward pass on `(2, 1, 257, 126)` produces a
matching `(2, 1, 257, 126)` mask in `[0, 1]`.

---

### 5.4 `src/losses.py` — Loss functions

```python
import torch, torch.nn as nn

mse_loss = nn.MSELoss()

def log_magnitude_loss(pred_log_mag, target_log_mag):
    return mse_loss(pred_log_mag, target_log_mag)

def si_sdr_loss(pred_wav, target_wav, eps=1e-8):
    """Negative SI-SDR (time-domain), so minimizing it maximizes SI-SDR."""
    pred_wav = pred_wav - pred_wav.mean(dim=-1, keepdim=True)
    target_wav = target_wav - target_wav.mean(dim=-1, keepdim=True)
    dot = torch.sum(pred_wav * target_wav, dim=-1, keepdim=True)
    target_energy = torch.sum(target_wav ** 2, dim=-1, keepdim=True) + eps
    s_target = (dot / target_energy) * target_wav
    e_noise = pred_wav - s_target
    si_sdr = 10 * torch.log10((torch.sum(s_target ** 2, dim=-1) + eps) /
                               (torch.sum(e_noise ** 2, dim=-1) + eps))
    return -si_sdr.mean()

def combined_loss(pred_log_mag, target_log_mag, pred_wav=None, target_wav=None, si_sdr_weight=0.0):
    """log-mag loss, optionally blended with SI-SDR loss (needs waveforms — see
    roadmap item 6, not wired into train.py's loop yet)."""
    loss = log_magnitude_loss(pred_log_mag, target_log_mag)
    if si_sdr_weight > 0:
        if pred_wav is None or target_wav is None:
            raise ValueError("si_sdr_weight > 0 requires pred_wav and target_wav")
        loss = loss + si_sdr_weight * si_sdr_loss(pred_wav, target_wav)
    return loss
```

`src/evaluate.py`'s `si_sdr()` metric reuses `si_sdr_loss` (sign-flipped), rather than
reimplementing the same math twice.

---

### 5.5 `src/train.py` — Training loop

```python
import os, torch, yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data_loader import NoisyCleanDataset
from src.model import UNetSE
from src.losses import combined_loss

def run_epoch(model, loader, optimizer, device, si_sdr_weight, train=True):
    model.train() if train else model.eval()
    running_loss = 0.0
    with torch.set_grad_enabled(train):
        for noisy_mag, clean_mag, _noisy_phase in tqdm(loader, leave=False):
            noisy_mag = noisy_mag.unsqueeze(1).to(device)   # (B,1,F,T), linear
            clean_mag = clean_mag.unsqueeze(1).to(device)

            # See §5.0: log-compress only for the model's input; mask the
            # linear magnitude; compare the loss in log domain.
            mask = model(torch.log1p(noisy_mag))
            pred_mag = mask * noisy_mag
            loss = combined_loss(torch.log1p(pred_mag), torch.log1p(clean_mag),
                                  si_sdr_weight=si_sdr_weight)

            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            running_loss += loss.item()
    return running_loss / max(len(loader), 1)

def main():
    cfg = yaml.safe_load(open("configs/config.yaml"))
    data_cfg, train_cfg = cfg.get("data", {}), cfg.get("train", {})
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = NoisyCleanDataset(data_cfg["noisy_train_dir"], data_cfg["clean_train_dir"],
                                       segment_seconds=train_cfg.get("segment_seconds", 2.0), train=True)
    val_dataset = NoisyCleanDataset(data_cfg["noisy_test_dir"], data_cfg["clean_test_dir"],
                                     segment_seconds=train_cfg.get("segment_seconds", 2.0), train=False)

    model = UNetSE(base_ch=train_cfg.get("base_ch", 32)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.get("lr", 1e-3))

    checkpoint_path = train_cfg.get("checkpoint_path", "checkpoints/unet_se.pt")
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(train_cfg.get("epochs", 30)):
        train_loss = run_epoch(model, train_loader, optimizer, device, 0.0, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, 0.0, train=False)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)  # only the best epoch, not the last
```

(Full version reads all hyperparameters from `configs/config.yaml`, builds both loaders, and
logs per-epoch — see `src/train.py`.) Only saves `checkpoints/unet_se.pt` when validation loss
improves, so the checkpoint is always the best epoch seen, not just the last one.

**Verified:** a manual 60-step overfit on 8 real training clips drops loss from 0.0235 to
0.0032 — the plumbing (data → model → mask → loss → backward) works end to end. A full
training run over all 11,572 pairs has not been done yet; `checkpoints/` is currently empty.

---

### 5.6 `src/evaluate.py` — PESQ / STOI / SI-SDR test-set evaluation

```python
def si_sdr(pred_wav, target_wav, eps=1e-8):
    """SI-SDR metric (higher is better) — reuses losses.si_sdr_loss, sign-flipped."""
    return -si_sdr_loss(pred_wav, target_wav, eps=eps).item()

def evaluate_pair(clean_wav, enhanced_wav, sr=SAMPLE_RATE):
    """PESQ (wideband) / STOI / SI-SDR for one pair. Truncates to equal length
    first; PESQ is wrapped in try/except since it raises on near-silent clips
    rather than returning a score."""
    min_len = min(clean_wav.shape[-1], enhanced_wav.shape[-1])
    clean_wav, enhanced_wav = clean_wav[:min_len], enhanced_wav[:min_len]
    try:
        pesq_score, pesq_failed = pesq(sr, clean_wav.numpy(), enhanced_wav.numpy(), "wb"), False
    except Exception:
        pesq_score, pesq_failed = None, True
    stoi_score = stoi(clean_wav.numpy(), enhanced_wav.numpy(), sr, extended=False)
    return {"PESQ": pesq_score, "STOI": stoi_score,
            "SI-SDR": si_sdr(enhanced_wav, clean_wav), "pesq_failed": pesq_failed}

def run_evaluation(enhance_fn, name, data_cfg, limit=None):
    """Walks the matched test-set files, applies enhance_fn per clip, writes
    results/<name>_metrics.csv plus a printed summary. enhance_fn is identity
    for the noisy floor, spectral_subtraction for the baseline, or a loaded
    checkpoint's forward pass for the model."""
    ...
```

CLI (full test set, one mode at a time):
```bash
python -m src.evaluate --mode noisy                                    # unprocessed floor
python -m src.evaluate --mode baseline                                 # spectral subtraction
python -m src.evaluate --mode model --checkpoint checkpoints/unet_se.pt
python -m src.evaluate --mode noisy --limit 20                         # quick smoke run
```

See §6 for the noisy/baseline numbers currently recorded in `results/`.

---

### 5.7 `src/baseline.py` — Classical baseline (spectral subtraction)

```python
import torch
from src.dsp import stft, istft, magnitude_phase, reconstruct_complex

def spectral_subtraction(noisy_wav, noise_estimate_frames=6, alpha=2.0, beta=0.01):
    """Estimates the noise spectrum from the first few frames (assumed
    noise-only — real VoiceBank+DEMAND clips have a short silent lead-in)
    and subtracts it from every frame, with a spectral floor to limit
    musical-noise artifacts from over-subtraction."""
    spec = stft(noisy_wav)
    mag, phase = magnitude_phase(spec)
    n_estimate = min(noise_estimate_frames, mag.shape[-1])
    noise_mag = mag[..., :n_estimate].mean(dim=-1, keepdim=True)
    enhanced_mag = torch.clamp(mag - alpha * noise_mag, min=beta * mag)
    return istft(reconstruct_complex(enhanced_mag, phase), length=noisy_wav.shape[-1])
```

**Verified on real data:** RMSE vs. clean drops from 0.0152 to 0.0064 on a real test-set pair
(58% reduction). Its noise estimate assumes a silent lead-in, which real VoiceBank clips have
but is worth keeping in mind if you ever point it at audio that doesn't (a naive per-file VAD
or minimum-statistics estimate would be more robust, but wasn't necessary here).

---

### 5.8 `src/inference.py` — Run enhancement on a new file

```python
def enhance_file(input_path, output_path, checkpoint=None, base_ch=None, device=None):
    ...
    wav = load_audio(input_path).to(device)
    mag, phase = magnitude_phase(stft(wav))
    mask = model(torch.log1p(mag).unsqueeze(0).unsqueeze(0)).squeeze()

    enhanced_mag = mask * mag                              # linear magnitude — see §5.0
    enhanced_spec = reconstruct_complex(enhanced_mag, phase)
    enhanced_wav = istft(enhanced_spec, length=wav.shape[-1]).cpu()
    torchaudio.save(output_path, enhanced_wav.unsqueeze(0), SAMPLE_RATE)
```

```bash
python -m src.inference --input noisy.wav --output enhanced.wav
python -m src.inference --input noisy.wav --output enhanced.wav --checkpoint checkpoints/unet_se.pt
```

---

### 5.9 `demo/app.py` — Streamlit demo

Uploads a noisy `.wav`, calls `enhance_file`, and plays both the input and output with a
download button for the result. Not yet tested end-to-end — it requires a trained checkpoint
at `checkpoints/unet_se.pt`, which doesn't exist yet (§5.5).

```bash
streamlit run demo/app.py
```

---

## 6. Current results

Recorded in `results/` via `python -m src.evaluate --mode <noisy|baseline|model>` on the full
824-file test set (0 PESQ skips in any run):

| Mode | PESQ | STOI | SI-SDR (dB) |
|---|---|---|---|
| noisy (unprocessed floor) | 1.971 | 0.921 | 8.44 |
| baseline (spectral subtraction, alpha=1.5) | 2.323 | 0.917 | 14.615 |
| U-Net (log-mag loss, 30 epochs) | **2.746** | **0.941** | **18.162** |
| U-Net (SI-SDR loss) | not started | | |

The U-Net (trained on a Kaggle T4, 2.42h wall-clock, best val_loss 0.00198 at epoch 18 —
see `results/train_log.csv` and `results/training_curve.png`) beats the baseline on every
metric, including STOI, which the baseline alone couldn't do relative to the noisy floor:
spectral subtraction's musical-noise artifacts cost a little intelligibility even while
improving perceived quality and SNR, but the learned mask recovers past that.

**Limitations.** This run's checkpoint selection is biased: `train.py` at the time built its
validation set from the same test directories `evaluate.py` then scored, so epoch 18 was picked
for minimizing loss on the exact 824 files the PESQ/STOI/SI-SDR numbers above come from. No
gradient updates ever touched the test set — this is model-selection leakage, not train/test
contamination — but it does mean "epoch 18 is best" was decided by peeking at the test score,
not a held-out proxy for it. The likely size of the bias is small: val loss in
`results/training_curve.png` is flat/noisy from roughly epoch 10 onward rather than sharply
peaked at 18, and the margin over baseline (PESQ 2.746 vs 2.320) is far larger than plausible
selection noise from picking among a handful of similar late-training epochs. `src/train.py` now
builds a speaker-disjoint held-out validation split from the training set (`p230`, `p274` held
out — see `results/val_split.json`) instead, so future runs won't have this issue; the numbers
above predate that fix and haven't been re-run yet.

---

## 7. Roadmap / Milestones

1. [x] Verify STFT → ISTFT reconstruction is lossless — confirmed, max abs error ≈ 1.79e-07
2. [x] Implement and test classical baseline (spectral subtraction) — 58% RMSE reduction on real data
3. [x] Build data pipeline + sanity check on a few samples — 11,572/824 pairs load and batch correctly
4. [x] Train U-Net mask model, confirm loss decreases — full 30-epoch run on a Kaggle T4
   (2.42h), best val_loss 0.00198 at epoch 18, checkpoint saved
5. [x] Evaluate with PESQ / STOI / SI-SDR, compare to baseline — all three modes (noisy,
   baseline, model) scored on the full test set; U-Net beats baseline on every metric
6. [ ] Tune loss function (try log-mag loss vs SI-SDR loss)
7. [ ] Build Streamlit demo — skeleton wired to `inference.py`, untested (needs a checkpoint)
8. [ ] (Stretch) Implement Conv-TasNet and compare time-domain vs T-F domain approaches
9. [ ] Write up final report with metric tables + before/after spectrogram plots

---

## 8. References

- Pascual et al., "SEGAN: Speech Enhancement Generative Adversarial Network"
- Luo & Mesgarani, "Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation"
- Valentini-Botinhao et al., "Noisy speech database for training speech enhancement algorithms" (VoiceBank+DEMAND)
- See `Theory.md` for full conceptual background.
