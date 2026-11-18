"""
Rule-based intent classification for the AI VR trainer.

Classifies normalized text into one of the intent categories defined in
the original project spec. This is a first-pass, explainable classifier
using keyword matching - not an LLM. It exists to route/triage BEFORE
the more expensive LLM stage (Phase 13).

Usage:
    from ai.nlp.normalize import normalize_text
    from ai.nlp.intent import classify_intent

    result = normalize_text("what should I do in an emergency")
    intent_result = classify_intent(result["lowercase"])
"""

# Keywords/phrases associated with each intent. Order matters: we check
# in a specific priority order below (EMERGENCY and SAFETY checked first,
# since a false negative there is much worse than elsewhere - missing an
# emergency signal is a real safety issue, not just an inconvenience).
INTENT_KEYWORDS = {
    "EMERGENCY": [
        "this is an emergency", "we have an emergency", "help me",
        "there is a fire", "fire right now", "fire has started",
        "injured", "injury", "accident", "urgent help",
        "danger", "dangerous situation", "someone is hurt",
        # NOTE: bare "emergency" is deliberately excluded - too
        # ambiguous, since "emergency stop" (equipment) isn't an actual
        # emergency. Bare "fire" is ALSO deliberately excluded - this is
        # a fire safety TRAINING app, so nearly every legitimate
        # question mentions "fire" (e.g. "how do I use a fire
        # extinguisher"). Only specific, actively-urgent phrasing
        # ("there is a fire", "fire has started") should trigger the
        # emergency shortcut - not the mere topic of fire.
    ],
    "SAFETY_QUESTION": [
        "safety", "safe to", "ppe", "protective equipment",
        "guard", "caution", "hazard", "risk", "precaution",
        "emergency stop",  # equipment name, not an actual emergency -
                            # must be checked as a SAFETY_QUESTION match
                            # BEFORE the generic "emergency" keyword below
        "lockout", "tagout",
        # Fire-safety-specific phrasing - core to this app's domain
        "discover a fire", "discover smoke", "fire extinguisher",
        "use an extinguisher", "fire alarm", "evacuation", "evacuate",
        "fire triangle", "pass technique",
    ],
    "STOP_PROCEDURE": [
        "stop the machine", "shut down", "shutdown", "turn off",
        "power off", "how do i stop", "how to stop",
    ],
    "START_PROCEDURE": [
        "start the machine", "startup", "start up", "how do i start",
        "how to start", "turn on", "power on", "begin operation",
        "start this machine", "start this", "safely start",
    ],
    "TROUBLESHOOTING": [
        "not working", "error", "problem", "issue", "malfunction",
        "won't start", "wont start", "broken", "fault", "troubleshoot",
    ],
    "MAINTENANCE": [
        "maintenance", "service", "lubricate", "replace part",
        "inspection schedule", "calibrate", "preventive",
    ],
    "MACHINE_INFORMATION": [
        "what is this", "explain", "how does this work", "workflow",
        "what does this machine", "tell me about", "what is the",
        "how does", "specifications", "components",
    ],
    "GREETING": [
        "hello", "hi there", "good morning", "good afternoon",
        "how are you", "hey there",
    ],
}

# Order of priority when multiple intents match - safety-critical
# categories are checked first so they take precedence over a more
# generic match (e.g. "explain the emergency stop procedure" should be
# SAFETY_QUESTION, not just MACHINE_INFORMATION, even though "explain"
# is also a MACHINE_INFORMATION keyword).
PRIORITY_ORDER = [
    "EMERGENCY", "SAFETY_QUESTION", "STOP_PROCEDURE", "START_PROCEDURE",
    "TROUBLESHOOTING", "MAINTENANCE", "MACHINE_INFORMATION", "GREETING",
]


def classify_intent(lowercase_text):
    """
    Check the text against each intent's keywords, in priority order.
    Returns the FIRST matching intent (highest priority match), along
    with which keyword(s) triggered the match - important for
    debuggability, so we always know WHY a classification happened.

    Returns UNKNOWN if nothing matches, rather than forcing a guess -
    an honest "I don't know" is more useful downstream than a
    confidently wrong category.
    """
    matches_by_intent = {}

    for intent in PRIORITY_ORDER:
        keywords = INTENT_KEYWORDS[intent]
        matched_keywords = [kw for kw in keywords if kw in lowercase_text]
        if matched_keywords:
            matches_by_intent[intent] = matched_keywords

    if not matches_by_intent:
        return {
            "intent": "UNKNOWN",
            "matched_keywords": [],
            "all_matches": {},
        }

    # Return the highest-priority intent that matched
    for intent in PRIORITY_ORDER:
        if intent in matches_by_intent:
            return {
                "intent": intent,
                "matched_keywords": matches_by_intent[intent],
                "all_matches": matches_by_intent,  # useful for debugging
                                                     # ambiguous cases where
                                                     # multiple intents matched
            }


if __name__ == "__main__":
    test_cases = [
        ("hello, how are you today", "GREETING"),
        ("what is this machine used for", "MACHINE_INFORMATION"),
        ("how do i safely start this machine", "START_PROCEDURE"),
        ("is it safe to touch this part", "SAFETY_QUESTION"),
        ("someone is injured, this is an emergency", "EMERGENCY"),
        ("the machine is not working, there is an error", "TROUBLESHOOTING"),
        ("what is the maintenance schedule for this", "MAINTENANCE"),
        ("how do i stop the machine", "STOP_PROCEDURE"),
        ("explain the emergency stop procedure", "SAFETY_QUESTION"),  # priority test
        ("blah blah nonsense text with no clear intent", "UNKNOWN"),
    ]

    print("Running intent classifier self-tests...\n")
    passed = 0
    for text, expected in test_cases:
        result = classify_intent(text)
        status = "OK" if result["intent"] == expected else "MISMATCH"
        if result["intent"] == expected:
            passed += 1
        print(f"[{status}] \"{text}\"")
        print(f"       -> {result['intent']} (expected {expected})")
        if result["matched_keywords"]:
            print(f"       matched: {result['matched_keywords']}")
        print()

    print(f"Passed {passed}/{len(test_cases)} test cases.")