"""
Safety validation layer - sits between the LLM and final output.

Two mechanisms, both deterministic (no reliance on LLM self-judgment,
per the lesson learned in Phase 14):

1. EMERGENCY SHORT-CIRCUIT: if intent is EMERGENCY, skip the LLM/RAG
   entirely and return a fixed, always-correct response immediately -
   now localized by language.

2. OUTPUT VALIDATION: scan the LLM's actual generated text for red-flag
   phrases that contradict known safety principles, regardless of what
   was asked or what the LLM "intended".

Usage:
    from ai.safety.validator import check_emergency_shortcut, validate_output
"""

import re

# ============================================================
# LAYER 1: EMERGENCY SHORT-CIRCUIT
# ============================================================

EMERGENCY_RESPONSE = {
    "en": (
        "This sounds like an emergency. Move to a safe position immediately, "
        "raise the alarm, and report it to your supervisor or emergency "
        "response team right away. If there is immediate danger to life, "
        "call your site's emergency number now."
    ),
    "hi": (
        "Yeh ek emergency lagti hai. Turant surakshit sthaan par jaayein, "
        "alarm bajayein, aur apne supervisor ya emergency response team ko "
        "turant report karein. Agar jaan ko khatra hai, to abhi apni site "
        "ke emergency number par call karein."
    ),
}


def check_emergency_shortcut(intent, language="en"):
    """
    If intent is EMERGENCY, return the fixed safe response immediately -
    do NOT route through LLM/RAG at all. This guarantees a correct,
    fast, and consistent response to the highest-stakes case, with zero
    dependency on model behavior.

    `language` should come from STT's detected language (Phase 5-7) in
    the real pipeline, NOT re-guessed from text - Roman-script Hindi is
    indistinguishable from English by script alone (see Phase 17 notes).

    Returns the response string if this is an emergency short-circuit,
    or None if normal processing (RAG+LLM) should proceed instead.
    """
    if intent == "EMERGENCY":
        return EMERGENCY_RESPONSE.get(language, EMERGENCY_RESPONSE["en"])
    return None


# ============================================================
# LAYER 2: OUTPUT VALIDATION
# ============================================================

DANGEROUS_PHRASES = [
    "you can skip",
    "no need to lock out",
    "no need for lockout",
    "you don't need ppe",
    "you dont need ppe",
    "safe to bypass",
    "okay to bypass",
    "you can bypass",
    "ignore the guard",
    "disable the guard",
    "remove the guard",
    "you can restart without",
    "safe to touch",
    "no need to evacuate",
    "not necessary to report",
    "don't need to raise the alarm",
    "dont need to raise the alarm",
]

HIGH_RISK_ACTION_PHRASES = [
    "start the machine", "start this machine", "power on the machine",
    "stop the machine", "shut down the machine", "power off the machine",
    "restart the", "reset the",
]


def validate_output(response_text, intent=None):
    """
    Scan the LLM's actual generated response text for safety issues.
    Independent of intent classification - checks what the model
    ACTUALLY SAID, not what was asked.

    Returns a dict: {status, reason, matched_phrases}
    status is one of: SAFE, HIGH_RISK, DANGEROUS
    """
    lower_text = response_text.lower()

    matched_dangerous = [p for p in DANGEROUS_PHRASES if p in lower_text]
    if matched_dangerous:
        return {
            "status": "DANGEROUS",
            "reason": "Response contains phrases contradicting known "
                      "safety principles.",
            "matched_phrases": matched_dangerous,
        }

    matched_action = [p for p in HIGH_RISK_ACTION_PHRASES if p in lower_text]
    if matched_action:
        return {
            "status": "HIGH_RISK",
            "reason": "Response involves a high-risk machine action - "
                      "requires explicit human confirmation before proceeding.",
            "matched_phrases": matched_action,
        }

    return {
        "status": "SAFE",
        "reason": "No safety concerns detected in response text.",
        "matched_phrases": [],
    }


if __name__ == "__main__":
    print("=== Testing emergency short-circuit ===")
    print(f"Intent=EMERGENCY, lang=en -> {check_emergency_shortcut('EMERGENCY', 'en')}\n")
    print(f"Intent=EMERGENCY, lang=hi -> {check_emergency_shortcut('EMERGENCY', 'hi')}\n")
    print(f"Intent=SAFETY_QUESTION -> {check_emergency_shortcut('SAFETY_QUESTION')}\n")

    print("=== Testing output validation ===")
    test_cases = [
        ("You can skip the lockout procedure if you're in a hurry.", "DANGEROUS"),
        ("It's safe to bypass the safety guard for quick access.", "DANGEROUS"),
        ("Please start the machine after checking the safety guard is closed.", "HIGH_RISK"),
        ("If you discover a fire, raise the alarm and evacuate safely.", "SAFE"),
        ("PPE stands for Personal Protective Equipment.", "SAFE"),
    ]

    for text, expected in test_cases:
        result = validate_output(text)
        status = "OK" if result["status"] == expected else "MISMATCH"
        print(f"[{status}] \"{text[:60]}...\"")
        print(f"    -> {result['status']} (expected {expected})")
        if result["matched_phrases"]:
            print(f"    matched: {result['matched_phrases']}")
        print()