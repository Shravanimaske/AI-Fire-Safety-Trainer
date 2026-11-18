# Load a wave file and play it back through your speakers 


import sys
import soundfile as sf 
import sounddevice as sd

def play_audio(path):
    data, sample_rate = sf.read(path)

    print(f"Playing: {path}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {len(data)/ sample_rate: .2f} seconds")

    sd.play(data, sample_rate)
    sd.wait()

    print("Plaback Finished.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ai/audio/play_audio.py<path_to_wav_files>")
        sys.exit(1)

    play_audio(sys.argv[1])    