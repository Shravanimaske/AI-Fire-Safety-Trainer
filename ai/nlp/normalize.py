"""
Text Normalization: Clean up raw STT output befor further processing. 
This includes removing filler words, punctuation, and other noise that may be present in the raw text. 
The goal is to produce a cleaner and more structured representation of the user's intent for downstream NLP tasks.


"""

import re

def normalize_text(text):
    """
     Clean raw STT text: 
     - Collapse multiple spaces into a single space
     - Strip leading and trailing whitespace
     - Remove Trailling ellipse left over from cut-off/ hallucinated speech 
     - Normalize multiple punctuation marks.
     - Lowecase (for matching purpose - we keep a separte "display"
     -version with original casing, since lowercasing loses information 
     like acronyms - PLC, SCADA, OEE - that might matter later)

     Returns a dict with both version, since different downstream downstream steps 
     need different things: keyword matching wants lowercase, but a 
     human-readable log LLM prromt wants the original casing preserved. 
    
    """


    original = text.strip()

    # Collapse multiple spaces into a single space
    cleaned = re.sub(r'\s+', ' ', original)

    # Remove trailing ellipses
    cleaned = re.sub(r'\.\.\.$', '', cleaned).strip()

    # Collapse repeated punctuation marks into a single instance
    cleaned = re.sub(r'([!?.,])\1+', r'\1', cleaned)

    # Convert to lowercase for matching purposes
    lowercase = cleaned.lower()

    return {"original": original, 
            "cleaned": cleaned, 
            "lowercase": lowercase}


if __name__ == "__main__":
    # Quick self-test with a few messy examples
    test_cases = [
        "Could you Explain the workflow?? ",
        "raw material head and cutting...  ",
        "   What is the OEE of the line???  ",
    ]

    for t in test_cases:
        result = normalize_text(t)
        print(f"Original: '{t!r}'")
        print(f"Cleaned: '{result['cleaned']}'")
        print(f"Lowercase: '{result['lowercase']}'")
        print()