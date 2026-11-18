"""
Language detection for Indian languages.

Strategy (much more reliable than langdetect for Indic text):
  1. Unicode script decides the language family. Almost every Indian language
     has its own script, so script alone identifies Tamil, Telugu, Bengali,
     Gujarati, Kannada, Punjabi, Malayalam and Odia.
  2. Devanagari is shared by Hindi and Marathi, so those two are separated
     with Marathi/Hindi marker words and Marathi-specific letters (e.g. ळ).
  3. Bengali vs Assamese is split on the Assamese-only letters ৰ and ৱ.
  4. Latin-only text is returned as "en" (same behaviour as before).
  5. Text mixing Indic script with Latin words returns "mixed" (same as before).

Returned codes match language_codes.WHISPER_TO_FLORES.
"""

import re

# (start, end, language code) - Unicode blocks
SCRIPT_RANGES = [
    (0x0900, 0x097F, "devanagari"),  # Hindi / Marathi
    (0x0980, 0x09FF, "bn"),          # Bengali / Assamese
    (0x0A00, 0x0A7F, "pa"),          # Gurmukhi (Punjabi)
    (0x0A80, 0x0AFF, "gu"),          # Gujarati
    (0x0B00, 0x0B7F, "or"),          # Odia
    (0x0B80, 0x0BFF, "ta"),          # Tamil
    (0x0C00, 0x0C7F, "te"),          # Telugu
    (0x0C80, 0x0CFF, "kn"),          # Kannada
    (0x0D00, 0x0D7F, "ml"),          # Malayalam
]

MIXED_THRESHOLD = 0.9   # below this share of Indic-script words -> "mixed"

STRIP_CHARS = ".,!?;:()[]{}\"'\u2018\u2019\u201c\u201d\u0964\u0965-"  # includes danda (।)

# Words that are Marathi-only (not used in Hindi)
MARATHI_WORDS = {
    "आहे", "आहेत", "आहेस", "नाही", "नाहीत", "काय", "मला", "तुला", "तुम्ही",
    "आम्ही", "माझ्या", "माझे", "माझी", "माझा", "तुमच्या", "तुमचे", "तुमची",
    "करा", "करावे", "करावा", "करू", "कसे", "कसा", "कशी", "होते", "होता",
    "होती", "आणि", "पण", "किंवा", "मध्ये", "साठी", "पासून", "झाले", "झाला",
    "झाली", "त्याला", "तिला", "त्यांना", "काही", "कृपया", "म्हणजे", "लागली",
    "लागल्यास", "जर", "मी", "तुम्हाला", "आपल्याला", "असे", "येथे", "तेथे",
    "सांगा", "सांगू", "कुठे", "केव्हा", "किती", "होय", "नको", "हवे", "हवा",
    "हवी", "द्या", "घ्या", "जावे", "यावे", "असेल", "असतात", "असते", "वाटते",
}

# Words that are Hindi-only (not used in Marathi)
HINDI_WORDS = {
    "है", "हैं", "नहीं", "क्या", "मुझे", "आप", "हम", "मेरा", "मेरी", "मेरे",
    "और", "लेकिन", "में", "के", "की", "को", "से", "करना", "चाहिए", "कैसे",
    "था", "थी", "थे", "यह", "वह", "कि", "तुम", "मैं", "हूँ", "हूं", "रहा",
    "रही", "रहे", "कृपया", "बताइए", "बताएं", "कीजिए", "करें", "क्यों", "कहाँ",
    "कब", "कितना", "कितनी", "कौन", "अगर", "तो", "भी", "सकता", "सकती", "सकते",
    "होगा", "होगी", "होता", "होती", "लगने", "स्थिति",
}

MARATHI_SUFFIXES = ("च्या", "ाची", "ाचा", "ाचे", "ामध्ये", "ाला", "ांना", "ाने")


def _script_of(ch):
    cp = ord(ch)
    for start, end, name in SCRIPT_RANGES:
        if start <= cp <= end:
            return name
    return None


def _clean_words(text):
    words = (w.strip(STRIP_CHARS) for w in text.split())
    return [w for w in words if w]


def _devanagari_language(words):
    """Hindi vs Marathi for Devanagari text."""
    mr_score = 0
    hi_score = 0
    joined = " ".join(words)

    for w in words:
        if w in MARATHI_WORDS:
            mr_score += 2
        if w in HINDI_WORDS:
            hi_score += 2
        if w.endswith(MARATHI_SUFFIXES):
            mr_score += 1

    mr_score += 2 * joined.count("\u0933")          # ळ - Marathi, rare in Hindi
    hi_score += joined.count("\u095B") + joined.count("\u095E") \
        + joined.count("\u095C") + joined.count("\u095D")  # ज़ फ़ ड़ ढ़ (nukta letters)

    return "mr" if mr_score > hi_score else "hi"


def detect_language(text):
    if not text or not text.strip():
        return "en"

    words = _clean_words(text)
    if not words:
        return "en"

    script_chars = {}
    indic_words = 0
    for w in words:
        has_indic = False
        for ch in w:
            s = _script_of(ch)
            if s:
                script_chars[s] = script_chars.get(s, 0) + 1
                has_indic = True
        if has_indic:
            indic_words += 1

    if not script_chars:
        return "en"   # Latin / no Indic script

    if indic_words / len(words) < MIXED_THRESHOLD:
        return "mixed"

    dominant = max(script_chars, key=script_chars.get)

    if dominant == "devanagari":
        return _devanagari_language(words)

    if dominant == "bn":
        # Assamese uses ৰ (U+09F0) and ৱ (U+09F1), Bengali does not
        return "as" if ("\u09F0" in text or "\u09F1" in text) else "bn"

    return dominant


if __name__ == "__main__":
    test_cases = [
        ("what should I do if I discover a fire", "en"),
        ("aag lagne par kya karna chahiye", "en"),
        ("आग लगने की स्थिति में मुझे क्या करना चाहिए?", "hi"),
        ("आग लागल्यास मी काय करावे?", "mr"),
        ("मला डोकेदुखी आहे आणि मी काय करू?", "mr"),
        ("कर्करोगाची लक्षणे कोणती आहेत?", "mr"),
        ("कैंसर के लक्षण क्या हैं?", "hi"),
        ("தீ விபத்து ஏற்பட்டால் என்ன செய்ய வேண்டும்?", "ta"),
        ("అగ్ని ప్రమాదం జరిగితే నేను ఏమి చేయాలి?", "te"),
        ("আগুন লাগলে আমার কী করা উচিত?", "bn"),
        ("આગ લાગે તો મારે શું કરવું જોઈએ?", "gu"),
        ("ಬೆಂಕಿ ಅಪಘಾತ ಸಂಭವಿಸಿದರೆ ನಾನು ಏನು ಮಾಡಬೇಕು?", "kn"),
        ("ਅੱਗ ਲੱਗਣ 'ਤੇ ਮੈਨੂੰ ਕੀ ਕਰਨਾ ਚਾਹੀਦਾ ਹੈ?", "pa"),
        ("Boss को बता देना kal मैं leave पर रहूँगा", "mixed"),
    ]

    print("Running language detection tests...\n")
    for text, expected in test_cases:
        result = detect_language(text)
        status = "OK" if result == expected else "MISMATCH"
        print(f"[{status}] \"{text}\" -> {result} (expected {expected})")