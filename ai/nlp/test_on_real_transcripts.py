"""
Test the text processing pipeline (normalize + classify) against real
transcripts already collected during Phases 5-7 evaluation.

"""

import os
import glob 
import sys

sys.path.insert(0,os.path.dirname(__file__))
from normalize import normalize_text
from intent import classify_intent


EVAL_DATA_DIR = os.path.join("data","evaluation")

def run():
    txt_files = glob.glob(os.path.join(EVAL_DATA_DIR, "*", "*.txt"))

    if not txt_files:
        print(f"No transcripts found under {EVAL_DATA_DIR}/")
        return

    print(f"Testing text pipeline against {len(txt_files)} real transcript(s)\n")


    for path in sorted(txt_files):
        with open(path, "r", encoding = "utf-8") as f:
            text = f.read().strip()

        norm = normalize_text(text)
        result = classify_intent(norm["lowercase"])

        print(f"File: {path}")

        print(f"Text: {text[:100]}{'...' if len(text)> 100 else''}")
        print(f"Intent: {result['matched_keywords']}")
        print(f"Matched Keywords: {result['matched_keywords']}")
        print()

if __name__ == "__main__":
    run()
