"""
Full audio preprocessing pipeline: raw audio in -> clean, STT-ready audio out.

Steps:
    1. Load audio (any format soundfile supports)
    2. Convert to mono if stereo
    3. Resample to 16kHz (the rate Whisper expects)
    4. Apply noise suppression
    5. Normalize amplitude

Usage:
    python ai/preprocessing/pipeline.py data/samples/phone_voice.wav
"""

import sys
import os
import numpy as np
import soundfile as sf
import librosa
import noisereduce as nr

TARGET_SAMPLE_RATE = 16000


def preprocess_audio(path, apply_denoise=True):
    print(f"Processing: {path}")

    # --- Step 1: Load ---
    data, original_rate = sf.read(path)
    print(f"  Loaded: {original_rate} Hz, "
          f"{'stereo' if data.ndim > 1 else 'mono'}, "
          f"{len(data) / original_rate:.2f}s")

    # --- Step 2: Downmix to mono if needed ---
    if data.ndim > 1:
        data = data.mean(axis=1)
        print("  Downmixed to mono")

    # --- Step 3: Resample to 16kHz ---
    if original_rate != TARGET_SAMPLE_RATE:
        data = librosa.resample(
            data.astype(np.float32),
            orig_sr=original_rate,
            target_sr=TARGET_SAMPLE_RATE,
        )
        print(f"  Resampled to {TARGET_SAMPLE_RATE} Hz")

    # --- Step 4: Noise suppression ---
    if apply_denoise:
        data = nr.reduce_noise(y=data, sr=TARGET_SAMPLE_RATE, stationary=True)
        print("  Applied noise suppression")

    # --- Step 5: Normalize amplitude ---
    # Scale so the loudest peak is at 0.9 (leaving a little headroom to
    # avoid clipping), rather than leaving amplitude wherever it ended up
    # after the previous steps. This keeps volume consistent across
    # different input recordings.
    peak = np.max(np.abs(data))
    if peak > 0:
        data = data / peak * 0.9
    print("  Normalized amplitude")

    base, ext = os.path.splitext(path)
    output_path = f"{base}_clean{ext}"
    sf.write(output_path, data, TARGET_SAMPLE_RATE)
    print(f"Saved clean audio to: {output_path}")

    return output_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ai/preprocessing/pipeline.py <path_to_wav_file>")
        sys.exit(1)

    preprocess_audio(sys.argv[1])