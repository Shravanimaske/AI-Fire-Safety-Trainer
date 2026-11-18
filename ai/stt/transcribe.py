

import sys
import time
from faster_whisper import WhisperModel


def transcribe(path, model_size="base", language=None):
    print(f"Loading Whisper model: {model_size}")
    load_start = time.time()

    
    
    model = WhisperModel(model_size="base", device="cpu", compute_type="int8")
    load_time = time.time() - load_start
    print(f"  Model loaded in {load_time:.2f}s")

    print(f"\nTranscribing: {path}")
    if language:
        print(f"  Language forced to: {language}")
    transcribe_start = time.time()

 
    segments, info = model.transcribe(path, beam_size=10, language=language)

    print(f"  Detected language: {info.language} "
          f"(probability: {info.language_probability:.2f})")

    full_text = ""
    print("\n  Segments:")
    for segment in segments:
        print(f"    [{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        full_text += " " + segment.text.strip()

    transcribe_time = time.time() - transcribe_start

    print(f"\n  Full transcript: \"{full_text.strip()}\"")
    print(f"\n  Transcription time: {transcribe_time:.2f}s")

    return full_text.strip()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ai/stt/transcribe.py <path_to_wav_file> [model_size] [language]")
        sys.exit(1)

    audio_path = sys.argv[1]
    size = sys.argv[2] if len(sys.argv) > 2 else "large"
    lang = sys.argv[3] if len(sys.argv) > 3 else None

    transcribe(audio_path, size, lang)

    