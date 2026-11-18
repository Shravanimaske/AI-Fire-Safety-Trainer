"""
Load a WAV file and plot its spectrogram (frequency content over time).

Usage:
    python ai/audio/visualize_spectrogram.py data/samples/sample_voice.wav

Saves the plot as a PNG next to showing it on screen.
"""

import sys
import os
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt


def visualize_spectrogram(path):
    data, sample_rate = sf.read(path)

    if data.ndim > 1:
        data = data[:, 0]  # just use the first channel if stereo

    plt.figure(figsize=(12, 5))

    # plt.specgram computes and plots a spectrogram in one step.
    # It works by chopping the signal into short overlapping windows
    # and running an FFT (Fourier Transform) on each one - this is the
    # "how" behind turning a 1D waveform into a 2D time-vs-frequency image.
    #
    # NFFT: how many samples per FFT window - bigger = better frequency
    #       resolution, worse time resolution (a fundamental tradeoff).
    # Fs: the sample rate, needed so frequency axis labels are in Hz.
    # noverlap: how much consecutive windows overlap - more overlap =
    #           smoother-looking result, more compute.
    spectrum, freqs, t, im = plt.specgram(
        data,
        NFFT=1024,
        Fs=sample_rate,
        noverlap=512,
        cmap="viridis",
    )

    plt.title(f"Spectrogram: {os.path.basename(path)}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Intensity (dB)")

    # Human speech mostly lives below ~8000 Hz - zoom in there so the
    # interesting detail isn't squeezed into a tiny sliver of the plot.
    plt.ylim(0, 8000)

    plt.tight_layout()

    output_path = os.path.splitext(path)[0] + "_spectrogram.png"
    plt.savefig(output_path, dpi=100)
    print(f"Saved spectrogram plot to: {output_path}")

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ai/audio/visualize_spectrogram.py <path_to_wav_file>")
        sys.exit(1)

    visualize_spectrogram(sys.argv[1])