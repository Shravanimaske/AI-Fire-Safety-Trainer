"""
Full pipeline: detect language -> normalize -> classify intent ->
(emergency shortcut OR RAG+LLM with confidence gate, both in the
detected language) -> safety validation -> TTS -> final output.

Usage:
    python ai/safety/full_pipeline.py "what is PPE"
    python ai/safety/full_pipeline.py "आग लगने की स्थिति में मुझे क्या करना चाहिए?"
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nlp"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "llm"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "actions"))
sys.path.insert(0, os.path.dirname(__file__))

from normalize import normalize_text
from intent import classify_intent
from language import detect_language
from rag_qa import load_model, answer_with_rag
from validator import check_emergency_shortcut, validate_output
from synthesize_hinglish import synthesize_auto
from decide_action import decide_action


def process_question(model, raw_text, speak=True):
    print(f"\n{'='*70}")
    print(f"USER INPUT: {raw_text}")
    print(f"{'='*70}")

    # Step 0: detect language (Phase 17 fix) - in the REAL pipeline this
    # should come from Whisper's STT detection (Phase 5-7) instead of
    # re-guessing from text, since Roman-script Hindi can't be reliably
    # detected from text alone. For text-based testing, we detect here.
    language = detect_language(raw_text)
    print(f"[LANGUAGE] {language}")

    # Step 1: normalize (Phase 8)
    norm = normalize_text(raw_text)

    # Step 2: classify intent (Phase 8)
    intent_result = classify_intent(norm["lowercase"])
    intent = intent_result["intent"]
    print(f"[INTENT] {intent} (matched: {intent_result['matched_keywords']})")

    # Step 3: emergency short-circuit (Phase 15) - now language-aware
    final_response = check_emergency_shortcut(intent, language=language)
    if final_response:
        print(f"[SAFETY] Emergency short-circuit triggered - bypassing LLM/RAG")
        decision = "ALLOWED (pre-approved emergency response)"
    else:
        # Step 4: RAG + LLM with confidence gating AND language matching
        final_response = answer_with_rag(model, norm["cleaned"], language=language)

        # Step 5: safety validation on the actual generated output (Phase 15)
        validation = validate_output(final_response, intent=intent)
        safety_status = validation["status"]
        print(f"\n[SAFETY] Output validation: {validation['status']} - "
              f"{validation['reason']}")

        if safety_status == "DANGEROUS":
            decision = "BLOCKED - response withheld from user"
            print(f"\nDECISION: {decision}")
            print(f"(Blocked content, for debugging only: {final_response})")
            return
        elif safety_status == "HIGH_RISK":
            decision = "CONFIRMATION REQUIRED before this action proceeds"
        else:
            decision = "ALLOWED"

    print(f"\nFINAL RESPONSE:\n{final_response}")
    print(f"\nDECISION: {decision}")

    # Step 6: text-to-speech (Phase 17) - uses the response's ACTUAL
    # script composition (not the detected input language) as the
    # source of truth for voice selection, since that's what the TTS
    # engine actually needs to read correctly.
    audio_path = None
    if speak:
        print(f"\n[TTS] Synthesizing spoken response...")
        audio_path = synthesize_auto(final_response)
        print(f"[TTS] Audio ready at: {audio_path}")

    action_info = decide_action(intent, safety_status=safety_status)
 
    output_for_unity = {
        "response": final_response,
        "audio_path": audio_path,
        "intent": action_info["intent"],
        "action": action_info["action"],
        "expression": action_info["expression"],
        "requires_confirmation": action_info["requires_confirmation"],
        "language": language,
        "safety_status": safety_status,
    }
 
    print(f"\n{'='*70}")
    print("STRUCTURED OUTPUT (for Unity):")
    print(f"{'='*70}")
    print(json.dumps(output_for_unity, ensure_ascii=False, indent=2))
 
    return output_for_unity


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python ai/safety/full_pipeline.py "your question here"')
        sys.exit(1)

    question = sys.argv[1]
    model = load_model()

    process_question(model, question)

    model.close()