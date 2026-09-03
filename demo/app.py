"""
demo/app.py — Streamlit demo: upload a noisy .wav, hear the enhanced result.

Run with:
    streamlit run demo/app.py
"""

import os
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# allow `from src...` imports when running via `streamlit run demo/app.py`
# from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dsp import load_audio, stft, magnitude_phase
from src.inference import enhance_file


def plot_spectrograms(noisy_path, enhanced_path):
    """Side-by-side log-magnitude spectrograms, same dB scale on both panels
    so the noise reduction is visually comparable rather than each panel
    auto-scaling to its own range."""
    noisy_mag, _ = magnitude_phase(stft(load_audio(noisy_path)))
    enhanced_mag, _ = magnitude_phase(stft(load_audio(enhanced_path)))
    noisy_db = 20 * np.log10(noisy_mag.numpy() + 1e-8)
    enhanced_db = 20 * np.log10(enhanced_mag.numpy() + 1e-8)
    vmin, vmax = -80, max(noisy_db.max(), enhanced_db.max())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, data, title in zip(axes, (noisy_db, enhanced_db), ("Noisy", "Enhanced")):
        im = ax.imshow(data, origin="lower", aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("frame")
    axes[0].set_ylabel("frequency bin")
    fig.colorbar(im, ax=axes, label="dB")
    return fig

st.set_page_config(page_title="AI Speech Enhancement Demo")
st.title("AI Speech Enhancement Demo")
st.caption("U-Net mask predictor trained on VoiceBank+DEMAND. Upload a noisy .wav file.")

checkpoint_path = "checkpoints/unet_se.pt"
if not os.path.exists(checkpoint_path):
    st.error(
        f"No trained model found at `{checkpoint_path}`. Run `python -m src.train` "
        f"first, then restart this demo."
    )
    st.stop()

uploaded_file = st.file_uploader("Upload a noisy audio file (.wav)", type=["wav"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
        tmp_in.write(uploaded_file.read())
        tmp_in_path = tmp_in.name

    st.subheader("Noisy Input")
    st.audio(tmp_in_path)

    output_path = os.path.join(tempfile.gettempdir(), "enhanced_output.wav")

    with st.spinner("Enhancing..."):
        try:
            enhance_file(tmp_in_path, output_path, checkpoint=checkpoint_path)
        except Exception as e:
            st.error(f"Enhancement failed: {e}")
            st.stop()

    st.subheader("Enhanced Output")
    st.audio(output_path)

    st.subheader("Spectrograms (before / after)")
    st.pyplot(plot_spectrograms(tmp_in_path, output_path))

    with open(output_path, "rb") as f:
        st.download_button("Download enhanced audio", f, file_name="enhanced.wav")