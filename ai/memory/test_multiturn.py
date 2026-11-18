"""
Test whether conversation memory actually improves multi-turn answers,
by comparing WITH memory vs WITHOUT memory on a genuine follow-up question.

Usage:
    python ai/memory/test_multiturn.py
"""

import sys
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_LLM_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "llm"))
sys.path.insert(0, _LLM_DIR)
sys.path.insert(0, _THIS_DIR)

# Fail loudly and clearly if the path is wrong, rather than a confusing
# ModuleNotFoundError - this tells us exactly what path was checked.
_rag_qa_path = os.path.join(_LLM_DIR, "rag_qa.py")
if not os.path.exists(_rag_qa_path):
    raise FileNotFoundError(
        f"Could not find rag_qa.py at expected path: {_rag_qa_path}\n"
        f"This script expects to live at ai/memory/test_multiturn.py, "
        f"with ai/llm/rag_qa.py as a sibling folder."
    )

from rag_qa import load_model, retrieve, SYSTEM_PROMPT, CONFIDENCE_THRESHOLD
from session import ConversationMemory


def answer_with_history(model, question, history_text=""):
    """Same as answer_with_rag, but optionally prepends conversation
    history to the prompt - this is the actual mechanism being tested."""
    results = retrieve(question, top_k=3, verbose=True)
    best_score = results[0].score if results else 0.0

    if best_score < CONFIDENCE_THRESHOLD:
        return ("I don't have reliable information on this in my training "
                "materials. Please check with your supervisor or the "
                "official documentation.")

    context_text = "\n\n".join(
        f"[{r.payload['title']}]\n{r.payload['text']}" for r in results
    )

    history_section = (
        f"Conversation so far:\n{history_text}\n\n" if history_text else ""
    )

    prompt = (
        f"{history_section}"
        f"Reference material:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        f"Answer the question using only the reference material above. "
        f"If the question refers back to something discussed earlier in "
        f"the conversation, use that context to understand what is being "
        f"asked."
    )

    response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=200,
        temperature=0.3,
    )
    return response["choices"][0]["message"]["content"]


def run_test():
    model = load_model()

    turn1_question = "what should I do if I discover a fire"
    turn2_question = "what about smoke instead"  # ambiguous WITHOUT context -
                                                    # "what about smoke" alone
                                                    # doesn't specify what
                                                    # aspect of smoke is
                                                    # being asked about

    print("=" * 70)
    print("TURN 1:", turn1_question)
    print("=" * 70)
    turn1_answer = answer_with_history(model, turn1_question)
    print(f"\n[TURN 1 ANSWER]\n{turn1_answer}")

    print("\n\n" + "=" * 70)
    print("TURN 2 WITHOUT MEMORY:", turn2_question)
    print("=" * 70)
    answer_no_memory = answer_with_history(model, turn2_question, history_text="")
    print(f"\n[NO MEMORY]\n{answer_no_memory}")

    print("\n\n" + "=" * 70)
    print("TURN 2 WITH MEMORY:", turn2_question)
    print("=" * 70)
    memory = ConversationMemory(max_turns=4)
    memory.add_turn("user", turn1_question)
    memory.add_turn("assistant", turn1_answer)
    history_text = memory.get_history_text()
    print(f"[Context being passed to LLM]:\n{history_text}\n")

    answer_with_memory_result = answer_with_history(
        model, turn2_question, history_text=history_text
    )
    print(f"\n[WITH MEMORY]\n{answer_with_memory_result}")

    print("\n\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"\nWITHOUT MEMORY:\n{answer_no_memory}")
    print(f"\nWITH MEMORY:\n{answer_with_memory_result}")

    model.close()


if __name__ == "__main__":
    run_test()