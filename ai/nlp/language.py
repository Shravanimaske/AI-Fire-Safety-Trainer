

import re
from langdetect import detect, LangDetectException

DEVANAGARI_PATTERN = re.compile(r"[\u0900-\u097F]")

def _detect_script_mix(text):
   
    words = text.split()
    if not words:
        return "en"
    devanagari_words = sum(1 for w in words if DEVANAGARI_PATTERN.search(w))
    ratio = devanagari_words / len(words)
    if ratio == 0:
        return "en"
    elif ratio >= 0.9:
        return None  # single-script - let langdetect decide which language
    else:
        return "mixed"


def detect_language(text):
    
    script_result = _detect_script_mix(text)
    if script_result == "mixed":
        return "mixed"  
 
    try:
        detected = detect(text)
        return detected
    except LangDetectException:
        
        return script_result if script_result else "en"
 


if __name__ == "__main__":
    test_cases = [
        ("what should I do if I discover a fire", "en"),
        ("आग लगने की स्थिति में मुझे क्या करना चाहिए?", "hi"),
        ("Boss को बता देना kal मैं leave पर रहूँगा", "mixed"),
        ("aag lagne par kya karna chahiye", "en"), ]

    print("Running language detection tests...\n")
    for text, expected in test_cases:
        result = detect_language(text)
        status = "OK" if result == expected else "MISMATCH"
        print(f"[{status}] \"{text}\" -> {result} (expected {expected})")

        