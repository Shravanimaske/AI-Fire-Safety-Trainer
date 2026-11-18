"""
Load the local Qwen GGUF model and test basic text generation.

Usage:
    python ai/llm/test_llm.py
"""

import time
from llama_cpp import Llama

MODEL_PATH = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf"


def load_model():
    print(f"Loading model: {MODEL_PATH}...")
    start = time.time()

    # n_ctx: context window size (how many tokens of conversation/prompt
    #   it can "remember" at once). 2048 is modest but fine for testing;
    #   we'll likely need more once RAG context gets added to prompts.
    # n_threads: how many CPU threads to use - None lets llama.cpp
    #   auto-detect based on your CPU.
    # verbose=False: suppresses llama.cpp's own internal logging spam.
    model = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_threads=None,
        verbose=False,
    )

    print(f"Model loaded in {time.time() - start:.2f}s")
    return model


def test_completion(model, prompt):
    print(f"\nPrompt: {prompt}")
    start = time.time()

    # Qwen models expect a specific chat message format, not raw text
    # completion - this matters for getting coherent, instruction-following
    # responses rather than the model just continuing the text randomly.
    response = model.create_chat_completion(
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        temperature=0.3,  # lower = more focused/deterministic,
                           # higher = more creative/random. 0.3 is a
                           # reasonable default for factual Q&A.
    )

    elapsed = time.time() - start
    answer = response["choices"][0]["message"]["content"]
    tokens_generated = response["usage"]["completion_tokens"]

    print(f"Response: {answer}")
    print(f"\nGenerated {tokens_generated} tokens in {elapsed:.2f}s "
          f"({tokens_generated/elapsed:.1f} tokens/sec)")

    return answer


if __name__ == "__main__":
    model = load_model()

    test_completion(model, "What is a fire triangle? Answer in 2 sentences.")

    # See rag_qa.py for why this explicit close() matters - avoids a
    # known llama-cpp-python bug during automatic shutdown cleanup.
    model.close()