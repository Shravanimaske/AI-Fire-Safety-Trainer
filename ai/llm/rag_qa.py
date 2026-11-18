

""""""
import sys
import os
import time
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nlp"))
from retrieve import retrieve
from language_codes import get_flores_code, get_language_name, is_supported

from llama_cpp import Llama

MODEL_PATH = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
TRANSLATE_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "translate.py")
import sys

if sys.platform == "win32":
    TRANSLATE_ENV_PYTHON = os.path.join(
        os.path.dirname(__file__), "..", "..", "translate_env", "Scripts", "python.exe"
    )
else:
    TRANSLATE_ENV_PYTHON = os.path.join(
        os.path.dirname(__file__), "..", "..", "translate_env", "bin", "python"
    )

SYSTEM_PROMPT = (
    "You are a fire safety training assistant for Thermax factory "
    "workers. Answer questions clearly and concisely using ONLY the "
    "provided reference material. If the reference material does not "
    "contain the answer, say so honestly rather than guessing."
)

CONFIDENCE_THRESHOLD = 0.55


def load_model():
    return Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_threads=None,
        n_gpu_layers=-1,
        verbose=False,
    )


def _call_translate_subprocess(text, src_flores, tgt_flores):
    """
    Run translate.py using translate_env's ISOLATED Python interpreter,
    with EXPLICIT FLORES-200 language codes - works for any supported
    language, not just Hindi.
    """
    python_exe = os.path.abspath(TRANSLATE_ENV_PYTHON)

    if not os.path.exists(python_exe):
        print(f"  WARNING: translate_env Python not found at {python_exe} "
              f"- falling back to original text")
        return text

    result = subprocess.run(
        [python_exe, TRANSLATE_SCRIPT_PATH, "--translate", text,
         "--src-lang", src_flores, "--tgt-lang", tgt_flores],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    if result.returncode != 0:
        print(f"  WARNING: translation subprocess failed: {result.stderr[-500:]}")
        return text

    return result.stdout.strip()


def translate_question_to_english(question, language):
    """Translate the user's question to English BEFORE retrieval.
    `language` is a Whisper-style code (e.g. "hi", "mr", "bn") - looked
    up in language_codes.py to get the actual FLORES-200 code needed."""
    if language == "en":
        return question

    if not is_supported(language):
        print(f"  WARNING: language '{language}' is not in our supported "
              f"language map - skipping translation, using original text "
              f"(retrieval quality may suffer)")
        return question

    flores_code = get_flores_code(language)
    lang_name = get_language_name(language)
    print(f"  [translating question {lang_name} ({language}) -> en for retrieval...]")
    translated = _call_translate_subprocess(question, flores_code, "eng_Latn")
    print(f"  [translated question]: {translated}")
    return translated


def translate_answer_back(english_answer, language):
    """Translate the finished English answer back to the target language."""
    if language == "en":
        return english_answer

    if not is_supported(language):
        print(f"  WARNING: language '{language}' not supported for "
              f"translation - returning English answer as fallback")
        return english_answer

    flores_code = get_flores_code(language)
    lang_name = get_language_name(language)
    print(f"\n[TRANSLATING answer to {lang_name} ({language})...]")
    return _call_translate_subprocess(english_answer, "eng_Latn", flores_code)

def answer_with_rag(model, question, top_k=3, language="en"):
    """
    Full round-trip: translate question to English (if needed) ->
    retrieve + generate in English -> translate answer back.
    """
    english_question = translate_question_to_english(question, language)

    results = retrieve(english_question, top_k=top_k, verbose=True)

    best_score = results[0].score if results else 0.0
    low_confidence = best_score < CONFIDENCE_THRESHOLD

    if low_confidence:
        print(f"  [confidence check] best score {best_score:.2f} < "
              f"{CONFIDENCE_THRESHOLD} -> LOW CONFIDENCE")
        fallback_en = ("I don't have reliable information on this in my "
                        "training materials. Please check with your "
                        "supervisor or the official documentation.")
        return translate_answer_back(fallback_en, language)

    print(f"  [confidence check] best score {best_score:.2f} >= "
          f"{CONFIDENCE_THRESHOLD} -> confidence OK")

    context_text = "\n\n".join(
        f"[{r.payload['title']}]\n{r.payload['text']}" for r in results
    )

    prompt = (
        f"Reference material:\n{context_text}\n\n"
        f"Question: {english_question}\n\n"
        f"Answer the question using only the reference material above. "
        f"Respond in English. "
        f"IMPORTANT: use SHORT, SIMPLE sentences. Avoid combining multiple "
        f"instructions with 'and' in one sentence. "
        f"CRITICAL: do NOT begin your answer with 'Yes' or 'No', even if "
        f"the question is phrased as a yes/no question. Instead, state the "
        f"correct action directly and plainly. For example, instead of "
        f"'No, you should not finish your task and leave', write 'Leave "
        f"immediately. Do not finish your task first.' This avoids "
        f"ambiguity about what a Yes/No answer refers to."
    )

    start = time.time()
    response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=300,
        temperature=0.3,
    )
    elapsed = time.time() - start

    english_answer = response["choices"][0]["message"]["content"]
    print(f"\n[GROUNDED (English), {elapsed:.1f}s]\n{english_answer}")

    return translate_answer_back(english_answer, language)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python ai/llm/rag_qa.py "your question here" [language]')
        sys.exit(1)

    question = sys.argv[1]
    lang = sys.argv[2] if len(sys.argv) > 2 else "en"

    model = load_model()
    print(f"Question: {question}")
    answer = answer_with_rag(model, question, language=lang)
    print(f"\nFINAL ANSWER:\n{answer}")
    model.close()