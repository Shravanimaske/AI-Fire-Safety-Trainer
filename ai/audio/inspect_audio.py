"""
Load a wav file and report its core Properties 

Usage : Python ai/Audio/inpenct_audio.py data/samples/samples_voice.wav

"""

import sys
import soundfile as sf
import numpy as np 

def inspect_audio(path):

    data, sample_rate = sf.read(path)
    num_channels = 1 if data.ndim == 1 else data.shape[1]
    num_samples = data.shape[0]
    duration_seconds = num_samples /sample_rate


    max_amplitude = np.max(np.abs(data))
    mean_amplitude = np.mean(np.abs(data))

    print(f"file: {path}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Channels: {num_channels}")
    print(f"Duration: {duration_seconds:.2f} seconds")
    print(f"Total samples: {num_samples}")
    print(f"Max amplitude: {max_amplitude: .4f}(range is 0.0 to 1.0)")
    print(f"Mean amplitude: {mean_amplitude: .4f}")

    if max_amplitude< 0.01:
        print(" WARNING: this file appears to be silent or near-silent.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ai/audio/inspect_audio.py<path_to_wav_file>")
        sys.exit(1)

    inspect_audio(sys.argv[1])