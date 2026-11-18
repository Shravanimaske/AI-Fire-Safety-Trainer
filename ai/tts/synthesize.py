"""
Convert text to spoken audio using Piper TTS.

Usage:
    python ai/tts/synthesize.py "If you discover a fire, stay calm."
    python ai/tts/synthesize.py "aapko turant alarm bajana chahiye" hi
"""


# ****************************************************************************************************************************

import sys
import os
import time
import wave
from piper import PiperVoice

VOICE_PATHS = {
    "en": os.path.join("models", "piper_voices", "en_US-lessac-medium.onnx"),
    "hi": os.path.join("models", "piper_voices", "hi_IN-priyamvada-medium.onnx"),
}


def synthesize(text, language="en", output_path=None):
    model_path = VOICE_PATHS.get(language)
    if not model_path or not os.path.exists(model_path):
        print(f"ERROR: no voice model found for language '{language}' "
              f"at expected path: {model_path}")
        return None

    if output_path is None:
        output_path = f"data/samples/tts_output_{language}.wav"

    print(f"Loading Piper voice ({language}): {model_path}...")
    start = time.time()
    voice = PiperVoice.load(model_path)
    print(f"Voice loaded in {time.time() - start:.2f}s")

    print(f"\nSynthesizing: \"{text}\"")
    start = time.time()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Piper writes directly to an open WAV file handle - it streams
    # audio chunks as it generates them, rather than returning one big
    # array, which matters for later real-time streaming use.
    with wave.open(output_path, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)

    elapsed = time.time() - start

    # Measure the resulting audio's duration to compute a real-time
    # factor - how much longer/shorter than the actual speech duration
    # generation took. Important latency metric for live use.
    with wave.open(output_path, "rb") as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        audio_duration = frames / rate

    print(f"Generated {audio_duration:.2f}s of audio in {elapsed:.2f}s "
          f"(real-time factor: {elapsed/audio_duration:.2f}x)")
    print(f"Saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python ai/tts/synthesize.py "text to speak" [language]')
        sys.exit(1)

    text = sys.argv[1]
    lang = sys.argv[2] if len(sys.argv) > 2 else "en"
    synthesize(text, language=lang)



# *********************************************************************************************************
