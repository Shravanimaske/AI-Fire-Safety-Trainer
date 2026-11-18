"""
Load a WAV file and plot its waveform (amplitude vs. time).

Usage:
    python ai/audio/visualize_waveform.py data/samples/sample_voice.wav

Saves the plot as a PNG next to showing it on screen.
"""

import sys
import os
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt


def visualize_waveform(path):
    data, sample_rate = sf.read(path)

    if data.ndim > 1:
        # If stereo, just plot the first channel for simplicity here.
        data = data[:, 0]

    # Build a time axis: one time value per sample, evenly spaced.
    # e.g. for 44100 Hz, sample 0 is at t=0, sample 44100 is at t=1.0s
    num_samples = len(data)
    time_axis = np.linspace(0, num_samples / sample_rate, num_samples)

    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, data, linewidth=0.5)
    plt.title(f"Waveform: {os.path.basename(path)}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = os.path.splitext(path)[0] + "_waveform.png"
    plt.savefig(output_path, dpi=100)
    print(f"Saved waveform plot to: {output_path}")

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ai/audio/visualize_waveform.py <path_to_wav_file>")
        sys.exit(1)

    visualize_waveform(sys.argv[1])