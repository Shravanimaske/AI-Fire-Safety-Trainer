"""
Run STT evaluation: transcribe audio files and score them against ground
truth transcripts, producing a summary report.

Expects a simple folder structure:
    data/evaluation/<language>/<name>.wav
    data/evaluation/<language>/<name>.txt   (ground truth transcript)

Usage:
    python ai/evaluation/run_evaluation.py small
    python ai/evaluation/run_evaluation.py large-v3
"""

import sys
import os
import glob
import time
from faster_whisper import WhisperModel
from wer import compute_wer, compute_cer

EVAL_DATA_DIR = os.path.join("data", "evaluation")


def load_eval_pairs():
    """
    Find every (audio, ground_truth) pair under data/evaluation/<lang>/.
    Returns a list of dicts: {language, audio_path, reference_text}
    """
    pairs = []
    if not os.path.isdir(EVAL_DATA_DIR):
        print(f"No evaluation data found at {EVAL_DATA_DIR}/")
        print("Create it with structure: data/evaluation/<language>/<name>.wav "
              "+ <name>.txt (ground truth)")
        return pairs

    for lang_dir in sorted(os.listdir(EVAL_DATA_DIR)):
        full_lang_dir = os.path.join(EVAL_DATA_DIR, lang_dir)
        if not os.path.isdir(full_lang_dir):
            continue

        for wav_path in sorted(glob.glob(os.path.join(full_lang_dir, "*.wav"))):
            txt_path = os.path.splitext(wav_path)[0] + ".txt"
            if not os.path.exists(txt_path):
                print(f"  WARNING: no ground truth found for {wav_path}, skipping")
                continue
            with open(txt_path, "r", encoding="utf-8") as f:
                reference_text = f.read().strip()

            pairs.append({
                "language": lang_dir,
                "audio_path": wav_path,
                "reference_text": reference_text,
            })

    return pairs


def run_evaluation(model_size="small"):
    pairs = load_eval_pairs()
    if not pairs:
        return

    print(f"Loading Whisper model: {model_size}...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    results = []
    for pair in pairs:
        print(f"\nEvaluating: {pair['audio_path']} ({pair['language']})")

        start = time.time()
        segments, info = model.transcribe(
            pair["audio_path"], beam_size=5, language=pair["language"]
        )
        hypothesis = "".join(seg.text for seg in segments).strip()
        elapsed = time.time() - start

        wer_result = compute_wer(pair["reference_text"], hypothesis)
        cer_result = compute_cer(pair["reference_text"], hypothesis)

        print(f"  Reference:  {pair['reference_text']}")
        print(f"  Hypothesis: {hypothesis}")
        print(f"  WER: {wer_result['wer']:.3f}  CER: {cer_result['cer']:.3f}  "
              f"Time: {elapsed:.2f}s")

        results.append({
            "language": pair["language"],
            "audio": os.path.basename(pair["audio_path"]),
            "wer": wer_result["wer"],
            "cer": cer_result["cer"],
            "time_seconds": elapsed,
        })

    # --- Summary report ---
    print("\n" + "=" * 70)
    print(f"SUMMARY  (model: {model_size})")
    print("=" * 70)
    print(f"{'Language':<12} {'File':<25} {'WER':>8} {'CER':>8} {'Time(s)':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['language']:<12} {r['audio']:<25} {r['wer']:>8.3f} "
              f"{r['cer']:>8.3f} {r['time_seconds']:>10.2f}")

    avg_wer = sum(r["wer"] for r in results) / len(results)
    avg_cer = sum(r["cer"] for r in results) / len(results)
    avg_time = sum(r["time_seconds"] for r in results) / len(results)
    print("-" * 70)
    print(f"{'AVERAGE':<12} {'':<25} {avg_wer:>8.3f} {avg_cer:>8.3f} {avg_time:>10.2f}")


if __name__ == "__main__":
    size = sys.argv[1] if len(sys.argv) > 1 else "small"
    run_evaluation(size)