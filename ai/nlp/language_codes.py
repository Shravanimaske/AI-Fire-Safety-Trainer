"""
Maps Whisper/faster-whisper's detected language codes (ISO 639-1, e.g.
"hi", "mr") to IndicTrans2's FLORES-200 codes (e.g. "hin_Deva",
"mar_Deva"), which is what the translation model actually needs.

This is the single source of truth for "which languages does this
project actually support end-to-end" - if a language isn't in this
map, we can't translate it, even if Whisper can transcribe it.

KNOWN GAP, not yet verified: Whisper's exact language coverage for
Odia and Assamese specifically hasn't been confirmed - these are less
commonly included in general-purpose STT training data than the other
languages here. Test with real audio before assuming they work.
"""

# Whisper ISO code -> (FLORES-200 code, human-readable name)
WHISPER_TO_FLORES = {
    "en": ("eng_Latn", "English"),
    "hi": ("hin_Deva", "Hindi"),
    "mr": ("mar_Deva", "Marathi"),
    "bn": ("ben_Beng", "Bengali"),
    "gu": ("guj_Gujr", "Gujarati"),
    "kn": ("kan_Knda", "Kannada"),
    "ml": ("mal_Mlym", "Malayalam"),
    "pa": ("pan_Guru", "Punjabi"),
    "ta": ("tam_Taml", "Tamil"),
    "te": ("tel_Telu", "Telugu"),
    "or": ("ory_Orya", "Odia"),   # unconfirmed Whisper coverage - test before relying on
    "as": ("asm_Beng", "Assamese"),  # unconfirmed Whisper coverage - test before relying on
}


def get_flores_code(whisper_code):
    """Returns the FLORES-200 code for a Whisper language code, or None
    if this language isn't supported by our translation pipeline."""
    entry = WHISPER_TO_FLORES.get(whisper_code)
    return entry[0] if entry else None


def get_language_name(whisper_code):
    entry = WHISPER_TO_FLORES.get(whisper_code)
    return entry[1] if entry else whisper_code


def is_supported(whisper_code):
    return whisper_code in WHISPER_TO_FLORES


if __name__ == "__main__":
    print("Supported languages:")
    for code, (flores, name) in WHISPER_TO_FLORES.items():
        print(f"  {code:4s} -> {flores:12s} ({name})")

    print("\nTest lookups:")
    for code in ["hi", "mr", "en", "xx"]:
        print(f"  get_flores_code('{code}') = {get_flores_code(code)}")


        