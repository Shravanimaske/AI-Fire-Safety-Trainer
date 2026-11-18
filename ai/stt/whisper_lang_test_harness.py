#!/usr/bin/env python3
"""
whisper_lang_test_harness.py

Compares Whisper's language auto-detection against a forced language code
(e.g. "mr" for Marathi) on a batch of audio files, to quantify how often
auto-detect misfires (e.g. Marathi -> Hindi confusion) and how much that
degrades transcription accuracy.

USAGE
-----
1. Put your test audio files in a folder, e.g. ./test_audio/
   Supported formats: anything ffmpeg can read (wav, mp3, m4a, ogg...)

2. (Optional but recommended) Create a reference transcript file so the
   script can compute Word Error Rate (WER) / Character Error Rate (CER).
   This should be a JSON file mapping filename -> ground-truth text, e.g.:

   {
     "sample1.wav": "कागदाच्या डब्यात छोटी आग लागली आहे.",
     "sample2.wav": "अग्निशामक यंत्र वापरण्यापूर्वी तुम्ही काय तपासाल?"
   }

   Save this as references.json in the same folder as the audio, or pass
   --references /path/to/references.json

3. Run:
   python whisper_lang_test_harness.py --audio-dir ./test_audio --forced-lang mr

   Add --model medium (or large-v3) to test whether a bigger checkpoint
   fixes the confusion, e.g.:
   python whisper_lang_test_harness.py --audio-dir ./test_audio --forced-lang mr --model medium

OUTPUT
------
- Per-file comparison: auto-detected language + confidence, auto transcript,
  forced-language transcript, and (if references given) WER/CER for both.
- A summary table at the end: how often auto-detect got the language wrong,
  and average WER/CER for auto vs forced.
- Full results also dumped to results.json in the audio dir for later analysis.

INSTALL
-------
pip install openai-whisper jiwer --break-system-packages
# ffmpeg must also be installed and on PATH
"""

import argparse
import json
import sys
import time
from pathlib import Path

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}


def load_whisper(model_name: str):
    try:
        import whisper
    except ImportError:
        print(
            "ERROR: openai-whisper is not installed.\n"
            "Install it with: pip install openai-whisper --break-system-packages",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Loading Whisper model '{model_name}' (this may take a while on first run)...")
    return whisper.load_model(model_name)


def try_load_jiwer():
    try:
        import jiwer
        return jiwer
    except ImportError:
        print(
            "NOTE: jiwer not installed, so WER/CER will be skipped.\n"
            "Install it with: pip install jiwer --break-system-packages\n"
        )
        return None


def transcribe(model, audio_path: str, language: str | None):
    """Run whisper.transcribe, returning (text, detected_language, detected_lang_prob)."""
    kwargs = {"task": "transcribe", "fp16": False}
    if language is not None:
        kwargs["language"] = language

    result = model.transcribe(audio_path, **kwargs)
    detected_lang = result.get("language")

    # Whisper doesn't return a language probability directly from transcribe(),
    # so if we're in auto-detect mode, run detect_language() separately to get it.
    lang_prob = None
    if language is None:
        try:
            import whisper
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
            _, probs = model.detect_language(mel)
            lang_prob = probs.get(detected_lang)
        except Exception as e:
            print(f"  (could not compute language probability: {e})")

    return result.get("text", "").strip(), detected_lang, lang_prob


def score(jiwer_lib, hypothesis: str, reference: str):
    if jiwer_lib is None or not reference:
        return None, None
    try:
        wer = jiwer_lib.wer(reference, hypothesis)
        cer = jiwer_lib.cer(reference, hypothesis)
        return wer, cer
    except Exception as e:
        print(f"  (scoring failed: {e})")
        return None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio-dir", required=True, help="Folder containing audio test files")
    parser.add_argument("--forced-lang", default="mr", help="Language code to force (default: mr = Marathi)")
    parser.add_argument("--model", default="small", help="Whisper model size: tiny/base/small/medium/large-v3")
    parser.add_argument("--references", default=None, help="Path to references.json (filename -> ground truth text)")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    if not audio_dir.is_dir():
        print(f"ERROR: {audio_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    ref_path = Path(args.references) if args.references else audio_dir / "references.json"
    references = {}
    if ref_path.exists():
        with open(ref_path, "r", encoding="utf-8") as f:
            references = json.load(f)
        print(f"Loaded {len(references)} reference transcript(s) from {ref_path}")
    else:
        print(f"No references.json found at {ref_path} — will skip WER/CER scoring.\n")

    audio_files = sorted(
        p for p in audio_dir.iterdir() if p.suffix.lower() in AUDIO_EXTS
    )
    if not audio_files:
        print(f"ERROR: no audio files found in {audio_dir} (looked for {AUDIO_EXTS})", file=sys.stderr)
        sys.exit(1)

    model = load_whisper(args.model)
    jiwer_lib = try_load_jiwer()

    results = []
    lang_mismatches = 0
    auto_wers, forced_wers = [], []

    for audio_path in audio_files:
        fname = audio_path.name
        print(f"\n=== {fname} ===")
        reference = references.get(fname)

        t0 = time.time()
        auto_text, detected_lang, lang_prob = transcribe(model, str(audio_path), language=None)
        auto_time = time.time() - t0
        print(f"[auto-detect]  lang={detected_lang} (p={lang_prob})  ({auto_time:.1f}s)")
        print(f"[auto-detect]  text: {auto_text}")

        t0 = time.time()
        forced_text, _, _ = transcribe(model, str(audio_path), language=args.forced_lang)
        forced_time = time.time() - t0
        print(f"[forced={args.forced_lang}]  text: {forced_text}  ({forced_time:.1f}s)")

        if detected_lang != args.forced_lang:
            lang_mismatches += 1
            print(f"  >>> LANGUAGE MISMATCH: auto-detected '{detected_lang}' vs expected '{args.forced_lang}'")

        auto_wer, auto_cer = score(jiwer_lib, auto_text, reference)
        forced_wer, forced_cer = score(jiwer_lib, forced_text, reference)
        if auto_wer is not None:
            print(f"  WER/CER  auto={auto_wer:.2f}/{auto_cer:.2f}   forced={forced_wer:.2f}/{forced_cer:.2f}")
            auto_wers.append(auto_wer)
            forced_wers.append(forced_wer)

        results.append({
            "file": fname,
            "reference": reference,
            "auto_detected_lang": detected_lang,
            "auto_detected_lang_prob": lang_prob,
            "auto_text": auto_text,
            "forced_lang": args.forced_lang,
            "forced_text": forced_text,
            "auto_wer": auto_wer,
            "auto_cer": auto_cer,
            "forced_wer": forced_wer,
            "forced_cer": forced_cer,
        })

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Files tested: {len(audio_files)}")
    print(f"Language auto-detect mismatches (vs '{args.forced_lang}'): {lang_mismatches}/{len(audio_files)}")
    if auto_wers:
        print(f"Average WER  — auto-detect: {sum(auto_wers)/len(auto_wers):.3f}   forced: {sum(forced_wers)/len(forced_wers):.3f}")
    else:
        print("No WER/CER computed (add a references.json to enable this).")

    out_path = audio_dir / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()