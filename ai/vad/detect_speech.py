"""
Detect speech segments in an audio file using Silero VAD, and save a
trimmed version containing only the speech.

Usage:
    python ai/vad/detect_speech.py data/samples/phone_voice_clean.wav
"""

import sys
import os
import numpy as np
import soundfile as sf
import torch
from silero_vad import load_silero_vad, get_speech_timestamps

VAD_SAMPLE_RATE = 16000  # Silero VAD expects 16kHz (or 8kHz) audio


def detect_speech(path):
    data, sample_rate = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)

    if sample_rate != VAD_SAMPLE_RATE:
        print(f"  WARNING: file is {sample_rate} Hz, but Silero VAD expects "
              f"{VAD_SAMPLE_RATE} Hz. Run this through the preprocessing "
              f"pipeline first (ai/preprocessing/pipeline.py).")
        return

    # Silero VAD's Python API wants a torch tensor, float32
    audio_tensor = torch.from_numpy(data.astype(np.float32))

    print("Loading Silero VAD model...")
    model = load_silero_vad()

    print("Detecting speech segments...")
    # get_speech_timestamps returns a list of dicts like:
    #   {'start': <sample_index>, 'end': <sample_index>}
    # marking where speech was detected, in SAMPLES (not seconds) by default.
    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        sampling_rate=VAD_SAMPLE_RATE,
        return_seconds=False,
        threshold=0.25,  # tuned experimentally for this project's typical
                          # recording volume - see Issue #4 notes on
                          # cross-machine variance in recurrent VAD models
        min_silence_duration_ms=400,  # IMPORTANT: default is only 100ms,
                          # which is shorter than a normal pause between
                          # words - this caused speech to be cut off
                          # mid-sentence at natural pauses. 400ms tolerates
                          # real pauses without ending the segment early.
        speech_pad_ms=100,  # pad 100ms onto the start/end of each detected
                          # segment, so soft word edges aren't clipped.
    )

    if not speech_timestamps:
        print("  No speech detected in this file.")
        return

    print(f"  Found {len(speech_timestamps)} speech segment(s):")
    for i, seg in enumerate(speech_timestamps):
        start_s = seg["start"] / VAD_SAMPLE_RATE
        end_s = seg["end"] / VAD_SAMPLE_RATE
        print(f"    Segment {i+1}: {start_s:.2f}s -> {end_s:.2f}s "
              f"({end_s - start_s:.2f}s duration)")

    # Build trimmed audio: concatenate only the detected speech segments,
    # dropping everything else (silence, noise between segments).
    trimmed_chunks = [data[seg["start"]:seg["end"]] for seg in speech_timestamps]
    trimmed_audio = np.concatenate(trimmed_chunks)

    original_duration = len(data) / VAD_SAMPLE_RATE
    trimmed_duration = len(trimmed_audio) / VAD_SAMPLE_RATE
    print(f"  Original duration: {original_duration:.2f}s")
    print(f"  Trimmed duration:  {trimmed_duration:.2f}s "
          f"({100 * trimmed_duration / original_duration:.1f}% kept)")

    base, ext = os.path.splitext(path)
    output_path = f"{base}_trimmed{ext}"
    sf.write(output_path, trimmed_audio, VAD_SAMPLE_RATE)
    print(f"Saved trimmed audio to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ai/vad/detect_speech.py <path_to_wav_file>")
        sys.exit(1)

    detect_speech(sys.argv[1])