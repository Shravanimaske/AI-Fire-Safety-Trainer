"""
Load a WAV and resample it to a target sample sample rate 
"""

import sys 
import os 
import soundfile as sf 
import librosa 
import numpy as np 

def resample_audio(path, target_rate = 16000):
    data, original_rate =sf.read(path)

    if data.ndim>1:
        data = data[:, 0]

    print(f"Original: {original_rate} Hz, {len(data)} samples"
          f"({len(data) / original_rate:.2f})") 

    if original_rate == target_rate:
        print("Already at target rate - nothing to do.")
        return 

    resampled = librosa.resample(
        data.astype(np.float32),
        orig_sr= original_rate,
        target_sr = target_rate, 
    )

    print(f"Resampled: {target_rate} Hz, {len(resampled)} samples,"
          f"{len(resampled) / target_rate: .2f} seconds")


    base, ext = os.path.splitext(path)
    output_path = f"{base}_resampled_{target_rate} hz{ext}"
    sf.write(output_path, resampled, target_rate)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv)<2:
        print("Usage: python ai/audio/resample_audio.py<path_to_wav_file> [target_rate]")
        sys.exit(1)

    input_path = sys.argv[1]
    target = int(sys.argv[2]) if len(sys.argv)>2 else 16000

    resample_audio(input_path, target)

