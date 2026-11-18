"""
Synthesize text to speech.
Usage:
    python ai/tts/synthesize_hinglish.py "Boss को बता देना kal मैं leave पर रहूँगा"
"""

import sys
import os
import re
import time
import numpy as np
import soundfile as sf
from piper import PiperVoice

# --------------------------------------------------------------------------
# Detect CUDA once, at import time.
# --------------------------------------------------------------------------
try:
    import onnxruntime
    _CUDA_AVAILABLE = "CUDAExecutionProvider" in onnxruntime.get_available_providers()
except Exception:
    _CUDA_AVAILABLE = False

print(f"[TTS] onnxruntime CUDA available: {_CUDA_AVAILABLE}")

# --------------------------------------------------------------------------
# Paths (anchored to this file, so cwd doesn't matter)
# --------------------------------------------------------------------------
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_MODULE_DIR, "..", ".."))

VOICE_PATHS = {
    "en": os.path.join(_PROJECT_ROOT, "ai", "models", "piper_voices",
                       "en_US-lessac-medium.onnx"),
    "hi": os.path.join(_PROJECT_ROOT, "ai", "models", "piper_voices",
                       "hi_IN-priyamvada-medium.onnx"),
}

DEVANAGARI_PATTERN = re.compile(r"[\u0900-\u097F]")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[।.?!])\s+")

# --------------------------------------------------------------------------
# MODULE-LEVEL CACHE — persists for the life of the process.
# This is THE fix for the 7-second latency.
# --------------------------------------------------------------------------
_VOICE_CACHE = {}


def _load_voice(lang):
    """Load a Piper voice once, keep it in memory forever."""
    if lang in _VOICE_CACHE:
        return _VOICE_CACHE[lang]

    path = VOICE_PATHS.get(lang)
    if not path or not os.path.exists(path):
        print(f"[TTS] ERROR: voice model missing for '{lang}': {path}")
        return None
    if not os.path.exists(path + ".json"):
        print(f"[TTS] ERROR: voice config missing: {path}.json")
        return None

    t0 = time.time()
    try:
        voice = PiperVoice.load(path, use_cuda=_CUDA_AVAILABLE)
        print(f"[TTS] Loaded '{lang}' voice (cuda={_CUDA_AVAILABLE}) "
              f"in {time.time()-t0:.2f}s")
    except Exception as e:
        print(f"[TTS] CUDA load failed ({e}); retrying on CPU")
        voice = PiperVoice.load(path, use_cuda=False)
        print(f"[TTS] Loaded '{lang}' voice (cpu) in {time.time()-t0:.2f}s")

    _VOICE_CACHE[lang] = voice
    return voice


def warmup():
    """Force-load all voices ahead of the first user request.
    Call this from FastAPI's lifespan startup (see notes below)."""
    for lang in VOICE_PATHS:
        _load_voice(lang)


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------
def split_into_sentences(text):
    sentences = SENTENCE_SPLIT_PATTERN.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def detect_script_mix(text):
    words = text.split()
    if not words:
        return "en"
    devanagari_words = sum(1 for w in words if DEVANAGARI_PATTERN.search(w))
    ratio = devanagari_words / len(words)
    if ratio == 0:
        return "en"
    if ratio >= 0.9:
        return "hi"
    return "mixed"


def segment_by_script(text):
    words = text.split()
    segments = []
    current_words = []
    current_lang = None

    for word in words:
        word_lang = "hi" if DEVANAGARI_PATTERN.search(word) else "en"
        if current_lang is None:
            current_lang = word_lang
        if word_lang != current_lang:
            segments.append((" ".join(current_words), current_lang))
            current_words = []
            current_lang = word_lang
        current_words.append(word)

    if current_words:
        segments.append((" ".join(current_words), current_lang))

    return segments


def synthesize_segment(voice, text):
    audio_chunks = []
    for audio_chunk in voice.synthesize(text):
        arr = np.frombuffer(audio_chunk.audio_int16_bytes, dtype=np.int16)
        audio_chunks.append(arr)

    if not audio_chunks:
        return np.array([], dtype=np.float32), voice.config.sample_rate

    combined = np.concatenate(audio_chunks).astype(np.float32) / 32768.0
    return combined, voice.config.sample_rate


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def synthesize_auto(text, output_path="data/samples/tts_auto_output.wav"):
    print(f"[TTS] Input text length: {len(text)} chars")
    t_start = time.time()

    sentences = split_into_sentences(text) 
    print(f"Sentence : {sentences} \n")
    print(f"[TTS] Split into {len(sentences)} sentence(s)")

    audio_pieces = []
    target_rate = None
    sentence_gap = None

    for idx, sentence in enumerate(sentences):
        mode = detect_script_mix(sentence)
        t_sent = time.time()

        if mode == "mixed":
            segs = segment_by_script(sentence)
            for seg_idx, (seg_text, lang) in enumerate(segs):
                voice = _load_voice(lang)
                if voice is None:
                    continue
                audio, rate = synthesize_segment(voice, seg_text)
                if target_rate is None:
                    target_rate = rate
                    sentence_gap = np.zeros(int(0.05 * rate), dtype=np.float32)
                audio_pieces.append(audio)
                # tiny pause only between script segments, not after the last
                if seg_idx < len(segs) - 1:
                    audio_pieces.append(np.zeros(int(0.03 * rate), dtype=np.float32))
        else:
            voice = _load_voice(mode)
            if voice is None:
                print(f"  WARNING: no voice for '{mode}', skipping sentence")
                continue
            audio, rate = synthesize_segment(voice, sentence)
            if target_rate is None:
                target_rate = rate
                sentence_gap = np.zeros(int(0.05 * rate), dtype=np.float32)
            audio_pieces.append(audio)

        # Insert silence BETWEEN sentences only — never after the last one.
        if idx < len(sentences) - 1 and sentence_gap is not None:
            audio_pieces.append(sentence_gap)

        dt = time.time() - t_sent
        if dt > 0.5:
            print(f"  [slow sentence {idx+1}] {dt:.2f}s :: {sentence[:60]!r}")

    if not audio_pieces:
        print("[TTS] ERROR: no audio generated.")
        return None

    final_audio = np.concatenate(audio_pieces)
    elapsed = time.time() - t_start

    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    sf.write(output_path, final_audio, target_rate)

    duration = len(final_audio) / target_rate
    print(f"[TTS] Generated {duration:.2f}s of audio in {elapsed:.2f}s "
          f"(real-time factor: {elapsed/duration:.2f}x)")
    print(f"[TTS] Saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python ai/tts/synthesize_hinglish.py "text"')
        sys.exit(1)
    synthesize_auto(sys.argv[1])