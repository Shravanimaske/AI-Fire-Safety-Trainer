"""
import torch

import transformers.tokenization_utils

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
transformers.tokenization_utils.PreTrainedTokenizerBase = PreTrainedTokenizerBase

import sys
import types

if "transformers.onnx" not in sys.modules:
    _onnx_stub = types.ModuleType("transformers.onnx")

    class OnnxConfig:
        pass


    class OnnxConfigWithPast(OnnxConfig):
        pass
    
    class OnnxSeq2SeqConfigWithPast:
        pass


    class PatchingSpec:
        pass

    _onnx_stub.OnnxConfig = OnnxConfig
    _onnx_stub.OnnxConfigWithPast = OnnxConfigWithPast
    _onnx_stub.OnnxSeq2SeqConfigWithPast = OnnxSeq2SeqConfigWithPast
    _onnx_stub.PatchingSpec = PatchingSpec
    sys.modules["transformers.onnx"] = _onnx_stub
    _onnx_utils_stub = types.ModuleType("transformers.onnx.utils")

    def compute_effective_axis_dimension(*args, **kwargs):
       
        raise NotImplementedError(
            "compute_effective_axis_dimension is a stub - ONNX export "
            "is not supported in this environment."
        )

    _onnx_utils_stub.compute_effective_axis_dimension = compute_effective_axis_dimension
    sys.modules["transformers.onnx.utils"] = _onnx_utils_stub
    _onnx_stub.utils = _onnx_utils_stub  


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

EN_INDIC_MODEL = "naklitechie/indictrans2-en-indic-dist-200M"
INDIC_EN_MODEL = "ai4bharat/indictrans2-indic-en-dist-200M"

_models = {}
_processor = None


def _load(direction):
    global _processor
    if direction not in _models:
        model_name = EN_INDIC_MODEL if direction == "en_to_hi" else INDIC_EN_MODEL
        print(f"Loading IndicTrans2 ({direction}): {model_name}...", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True,torch_dtype=torch.float16, 
            low_cpu_mem_usage=True)
        _models[direction] = (model, tokenizer)
        print("Loaded.", file=sys.stderr)

    if _processor is None:
        _processor = IndicProcessor(inference=True)

    return _models[direction][0], _models[direction][1], _processor


def _run_translation(model, tokenizer, processor, text, src_lang, tgt_lang):
    batch = processor.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
    inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt")

    with torch.no_grad():
        generated_tokens = model.generate(**inputs, max_length=256, num_beams=5)

    decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return processor.postprocess_batch(decoded, lang=tgt_lang)[0]


def translate_to_hindi(english_text):
    model, tokenizer, processor = _load("en_to_hi")
    return _run_translation(model, tokenizer, processor, english_text,
                             src_lang="eng_Latn", tgt_lang="hin_Deva")


def translate_to_english(hindi_text):
    model, tokenizer, processor = _load("hi_to_en")
    return _run_translation(model, tokenizer, processor, hindi_text,
                             src_lang="hin_Deva", tgt_lang="eng_Latn")


if __name__ == "__main__":
    import sys as _sys

    # Windows console/subprocess UTF-8 compatibility 
    _sys.stdout.reconfigure(encoding="utf-8") 
    _sys.stderr.reconfigure(encoding="utf-8")

    if "--translate" in _sys.argv:
        text_idx = _sys.argv.index("--translate") + 1
        text_to_translate = _sys.argv[text_idx]

        direction = "en_to_hi"
        if "--direction" in _sys.argv:
            direction = _sys.argv[_sys.argv.index("--direction") + 1]

        if direction == "hi_to_en":
            result = translate_to_english(text_to_translate)
        else:
            result = translate_to_hindi(text_to_translate)

        print(result)
        _sys.exit(0) 

    print("=== English -> Hindi ===")
    for sentence in ["Stay calm and raise the alarm.",
                      "Move to a safe position."]:
        result = translate_to_hindi(sentence)
        print(f"EN: {sentence}")
        print(f"HI: {result}\n")

    print("=== Hindi -> English ===")
    for sentence in ["आग लगने की स्थिति में क्या करना चाहिए?"]:
        result = translate_to_english(sentence)
        print(f"HI: {sentence}")
        print(f"EN: {result}\n")

"""

"""
Dedicated English<->Indic-language translation using IndicTrans2
(AI4Bharat), running in an ISOLATED virtual environment (translate_env)
with its own pinned transformers==4.40.0.

WHY AN ISOLATED ENVIRONMENT: IndicTransToolkit and sentence-transformers
(used elsewhere in this project for BGE-M3 retrieval) have genuinely
incompatible transformers version requirements - no single version
satisfies both. This module is only ever invoked via subprocess, using
translate_env's own Python interpreter (see ai/llm/rag_qa.py's
TRANSLATE_ENV_PYTHON), never imported directly into the main app.

Supports translation between English and any of the Indic languages
listed in ai/nlp/language_codes.py, using FLORES-200 language codes
(e.g. "hin_Deva", "mar_Deva", "ben_Beng") - not hardcoded to Hindi.

CLI usage (used by rag_qa.py's subprocess call):
    python translate.py --translate "text" --src-lang eng_Latn --tgt-lang mar_Deva
    python translate.py --translate "text" --direction hi_to_en   (legacy, Hindi only)
"""

import torch

# --- Compatibility shim #1 ---
# IndicTransToolkit's collator.py imports PreTrainedTokenizerBase from
# the wrong module path for newer transformers versions (5.x+). Not
# strictly needed now that we're pinned to transformers==4.40.0 in
# translate_env, but kept as defense-in-depth in case that pin ever
# changes.
import transformers.tokenization_utils
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
transformers.tokenization_utils.PreTrainedTokenizerBase = PreTrainedTokenizerBase

# --- Compatibility shim #2 ---
# IndicTrans2's remote model code imports several classes from the
# transformers.onnx module tree, which was removed in transformers 5.x.
# Also kept as defense-in-depth under the 4.40.0 pin.
import sys
import types

if "transformers.onnx" not in sys.modules:
    _onnx_stub = types.ModuleType("transformers.onnx")

    class OnnxConfig:
        pass

    class OnnxConfigWithPast(OnnxConfig):
        pass

    class OnnxSeq2SeqConfigWithPast(OnnxConfigWithPast):
        pass

    class PatchingSpec:
        pass

    _onnx_stub.OnnxConfig = OnnxConfig
    _onnx_stub.OnnxConfigWithPast = OnnxConfigWithPast
    _onnx_stub.OnnxSeq2SeqConfigWithPast = OnnxSeq2SeqConfigWithPast
    _onnx_stub.PatchingSpec = PatchingSpec
    sys.modules["transformers.onnx"] = _onnx_stub

    _onnx_utils_stub = types.ModuleType("transformers.onnx.utils")

    def compute_effective_axis_dimension(*args, **kwargs):
        raise NotImplementedError(
            "compute_effective_axis_dimension is a stub - ONNX export "
            "is not supported in this environment."
        )

    _onnx_utils_stub.compute_effective_axis_dimension = compute_effective_axis_dimension
    sys.modules["transformers.onnx.utils"] = _onnx_utils_stub
    _onnx_stub.utils = _onnx_utils_stub

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

EN_INDIC_MODEL = "naklitechie/indictrans2-en-indic-dist-200M"
INDIC_EN_MODEL = "ai4bharat/indictrans2-indic-en-dist-200M"

_models = {}
_processor = None


def _load(direction):
    """direction is 'en_to_hi' or 'hi_to_en' - these are really just
    internal keys selecting WHICH underlying model to use (the en-indic
    many-to-many model, or the indic-en many-to-many model) - not
    limited to Hindi specifically, despite the naming (kept for
    backward compatibility with earlier code)."""
    global _processor
    if direction not in _models:
        model_name = EN_INDIC_MODEL if direction == "en_to_hi" else INDIC_EN_MODEL
        print(f"Loading IndicTrans2 ({direction}): {model_name}...", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, trust_remote_code=True,
            torch_dtype=torch.float16, low_cpu_mem_usage=True,
        )
        _models[direction] = (model, tokenizer)
        print("Loaded.", file=sys.stderr)

    if _processor is None:
        _processor = IndicProcessor(inference=True)

    return _models[direction][0], _models[direction][1], _processor


def _run_translation(model, tokenizer, processor, text, src_lang, tgt_lang):
    batch = processor.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
    inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt")

    with torch.no_grad():
        generated_tokens = model.generate(**inputs, max_length=256, num_beams=5)

    decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return processor.postprocess_batch(decoded, lang=tgt_lang)[0]


def translate_text(text, src_lang, tgt_lang):
    """
    Generic translation between English and ANY supported Indic
    language, using FLORES-200 codes directly (e.g. "hin_Deva",
    "mar_Deva", "ben_Beng") - not hardcoded to Hindi.
    """
    if src_lang == tgt_lang:
        return text

    if src_lang == "eng_Latn":
        model, tokenizer, processor = _load("en_to_hi")
    else:
        model, tokenizer, processor = _load("hi_to_en")

    return _run_translation(model, tokenizer, processor, text,
                             src_lang=src_lang, tgt_lang=tgt_lang)


# Backward-compatible wrappers (existing callers elsewhere may still
# use these Hindi-specific names).
def translate_to_hindi(english_text):
    return translate_text(english_text, "eng_Latn", "hin_Deva")


def translate_to_english(indic_text, src_lang="hin_Deva"):
    return translate_text(indic_text, src_lang, "eng_Latn")


if __name__ == "__main__":
    import sys as _sys

    # Windows console/subprocess UTF-8 compatibility
    _sys.stdout.reconfigure(encoding="utf-8")
    _sys.stderr.reconfigure(encoding="utf-8")

    if "--translate" in _sys.argv:
        text_idx = _sys.argv.index("--translate") + 1
        text_to_translate = _sys.argv[text_idx]

        # Generic interface: --src-lang / --tgt-lang take FLORES codes
        # directly (e.g. "hin_Deva", "mar_Deva", "eng_Latn").
        src_lang = "eng_Latn"
        tgt_lang = "hin_Deva"
        if "--src-lang" in _sys.argv:
            src_lang = _sys.argv[_sys.argv.index("--src-lang") + 1]
        if "--tgt-lang" in _sys.argv:
            tgt_lang = _sys.argv[_sys.argv.index("--tgt-lang") + 1]

        # Legacy interface: --direction hi_to_en / en_to_hi (Hindi only)
        if "--direction" in _sys.argv:
            direction = _sys.argv[_sys.argv.index("--direction") + 1]
            if direction == "hi_to_en":
                src_lang, tgt_lang = "hin_Deva", "eng_Latn"
            else:
                src_lang, tgt_lang = "eng_Latn", "hin_Deva"

        result = translate_text(text_to_translate, src_lang, tgt_lang)
        print(result)
        _sys.exit(0)

    print("=== Testing multiple languages ===")
    tests = [
        ("Stay calm and raise the alarm.", "eng_Latn", "hin_Deva"),
        ("Stay calm and raise the alarm.", "eng_Latn", "mar_Deva"),
        ("Stay calm and raise the alarm.", "eng_Latn", "ben_Beng"),
    ]
    for text, src, tgt in tests:
        result = translate_text(text, src, tgt)
        print(f"[{src} -> {tgt}] {text}")
        print(f"  -> {result}\n")
