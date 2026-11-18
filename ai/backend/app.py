"""
    uvicorn ai.backend.app:app --host 0.0.0.0 --port 8000

LLM (Qwen) and embeddings (BGE) are served remotely (OpenAI-compatible API,
e.g. through ngrok). Set LLM_BASE_URL to override the default URL.
"""

import sys
import os
import json
import time
import tempfile
import urllib.parse

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nlp"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "llm"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "actions"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stt"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "safety"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # backend folder first

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

from ai.nlp.normalize import normalize_text
from ai.nlp.intent import classify_intent
from ai.nlp.indic_language import detect_language
from ai.nlp import indic_language
from ai.safety.validator import check_emergency_shortcut, validate_output
from ai.tts.synthesize_hinglish import synthesize_auto
from ai.actions.decide_action import decide_action


# ---------------------------------------------------------------------------
# Remote model configuration (same endpoints as test.py)
# ---------------------------------------------------------------------------
URL = os.getenv("LLM_BASE_URL", "https://hula-datebook-unengaged.ngrok-free.dev").rstrip("/")
LLM_URL = "http://127.0.0.1:8080"
EMB_URL = "http://127.0.0.1:8081"

QWEN_ENDPOINT = f"{LLM_URL}/v1/chat/completions"
BGE_ENDPOINT = f"{EMB_URL}/v1/embeddings"

LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 128
LLM_TIMEOUT = (15, 300)    # (connect, read) seconds
EMBED_TIMEOUT = 180

LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "mr": "Marathi", "bn": "Bengali",
    "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam", "pa": "Punjabi",
    "ta": "Tamil", "te": "Telugu", "or": "Odia", "as": "Assamese",
    "mixed": "the same mix of languages the user used",
}

SYSTEM_PROMPT = (
    "You are a helpful, safe health-awareness assistant. "
    "Answer concisely and accurately. "
    "Reply ONLY in {language}, written in its native script "
    "(for Hindi use Devanagari, for Marathi use Devanagari, etc.). "
    "Never use Roman/Latin letters for Indian languages."
)

models = {}


def qwen_stream(messages, temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS):
    """Stream tokens from the remote Qwen server (SSE), yielding text chunks."""
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    with models["llm"].post(
        QWEN_ENDPOINT, json=payload, stream=True, timeout=LLM_TIMEOUT
    ) as response:
        response.raise_for_status()

        # SSE servers reply with `text/event-stream` (no charset).
        # requests then defaults to ISO-8859-1 per RFC 2616 — which
        # mangles every non-ASCII char (Hindi, Marathi, Tamil...).
        # Force UTF-8 to match what the server is actually sending.
        response.encoding = "utf-8"

        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                print("Skipping invalid SSE event:", data)
                continue
            choices = event.get("choices", [])
            if choices:
                content = choices[0].get("delta", {}).get("content", "")
                if content:
                    yield content


def answer_with_remote_llm(text, language="en"):
    """Replacement for rag_qa.answer_with_rag: asks the remote Qwen server."""
    messages = [
        {"role": "system",
         "content": SYSTEM_PROMPT.format(language=LANGUAGE_NAMES.get(language, "English"))},
        {"role": "user", "content": text},
    ]
    return "".join(qwen_stream(messages)).strip()


def embed_texts(texts):
    """Get BGE embeddings from the remote server. Use this for RAG retrieval."""
    response = models["llm"].post(
        BGE_ENDPOINT, json={"input": texts}, timeout=EMBED_TIMEOUT
    )
    response.raise_for_status()
    return [item["embedding"] for item in response.json()["data"]]


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting server...")
    print(f"  Language detector loaded from: {indic_language.__file__}")
    start = time.time()

    # Remote LLM: just a reusable HTTP session, nothing heavy to load.
    models["llm"] = requests.Session()
    try:
        embed_texts(["ping"])
        print(f"  Remote Qwen/BGE server reachable at {LLM_URL} | {EMB_URL}")
    except Exception as e:
        print(f"  WARNING: remote model server check failed: {e}")

    try:
        from faster_whisper import WhisperModel
        models["stt"] = WhisperModel("base", device="cpu", compute_type="int8",cpu_threads=4,num_workers=1)
        print("  STT (faster-whisper base) loaded")
    except Exception as e:
        print(f"  WARNING: STT model failed to load: {e}")
        models["stt"] = None

    from ai.tts.synthesize_hinglish import warmup as tts_warmup
    tts_warmup()
    print("  TTS voices warmed up")
    print(f"Startup finished in {time.time() - start:.1f}s")
    yield
    print("Shutting down, closing remote LLM session...")
    models["llm"].close()


app = FastAPI(title="AI VR Trainer Backend", lifespan=lifespan)

# CORS: lets Unity / browser clients call this API, and makes the
# custom X-* headers on /audio readable from JavaScript.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Transcript", "X-Intent", "X-Language", "X-Safety-Status",
        "X-Action-Type", "X-Expression", "X-Requires-Confirmation",
    ],
)


class ChatRequest(BaseModel):
    text: str


class TTSRequest(BaseModel):
    text: str
    language: str = None   # optional: en, hi, mr, ta, te, bn, gu, kn, pa, ...


class FileInp(BaseModel):
    file: str = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm_url": URL,
        "models_loaded": {
            "llm": models.get("llm") is not None,
            "stt": models.get("stt") is not None,
        },
    }


def _run_pipeline(raw_text, known_language=None):
    """Shared pipeline logic used by both /chat and /audio."""
    language = known_language if known_language else detect_language(raw_text)

    norm = normalize_text(raw_text)
    intent_result = classify_intent(norm["lowercase"])
    intent = intent_result["intent"]

    safety_status = "SAFE"
    final_response = check_emergency_shortcut(intent, language=language)

    if final_response:
        decision = "ALLOWED (pre-approved emergency response)"

    else:
        try:
            final_response = answer_with_remote_llm(norm["cleaned"], language=language) 
            print(f"Final RAW result : {final_response}")
        except requests.RequestException as e:
            raise HTTPException(status_code=502, detail=f"Remote LLM error: {e}")

        validation = validate_output(final_response, intent=intent)
        safety_status = validation["status"]

        if safety_status == "DANGEROUS":
            return {
                "response": None,
                "decision": "BLOCKED",
                "intent": intent,
                "language": language,
                "safety_status": safety_status,
            }
        elif safety_status == "HIGH_RISK":
            decision = "CONFIRMATION REQUIRED"
        else:
            decision = "ALLOWED"

    action_info = decide_action(intent, safety_status=safety_status)

    return {
        "response": final_response,
        "intent": action_info["intent"],
        "action": action_info["action"],
        "expression": action_info["expression"],
        "requires_confirmation": action_info["requires_confirmation"],
        "language": language,
        "safety_status": safety_status,
        "decision": decision,
    }


@app.post("/chat")
def chat(request: ChatRequest):
    """Text in, full pipeline JSON out (no audio)."""
    if models.get("llm") is None:
        raise HTTPException(status_code=503, detail="LLM client not initialised")
    return _run_pipeline(request.text)


@app.post("/embed")
def embed(request: ChatRequest):
    """Text in, BGE embedding out (handy for testing the remote embedding server)."""
    try:
        vectors = embed_texts([request.text])
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Remote embedding error: {e}")
    return {"dimension": len(vectors[0]), "embedding": vectors[0]}


@app.post("/tts")
def tts(request: TTSRequest):
    """Text in, audio file out."""
    output_path = tempfile.mktemp(suffix=".wav")
    language = request.language or detect_language(request.text)
    result_path = synthesize_auto(request.text, output_path=output_path)

    if result_path is None:
        raise HTTPException(status_code=500, detail="TTS synthesis failed")

    return FileResponse(result_path, media_type="audio/wav")


@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    """Audio in, transcript + detected language out."""
    if models.get("stt") is None:
        raise HTTPException(status_code=503, detail="STT model not loaded")

    temp_path = tempfile.mktemp(suffix=".wav")
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    segments, info = models["stt"].transcribe(temp_path, beam_size=5)
    text = "".join(seg.text for seg in segments).strip()

    os.remove(temp_path)

    return {
        "text": text,
        "language": info.language,
        "language_probability": info.language_probability,
    }


def _ascii_safe_header(value):
    """HTTP headers must be ASCII - Devanagari text needs URL-encoding
    before it can go into a header value."""
    return urllib.parse.quote(str(value))


@app.post("/audio")
async def audio_full_turn(file: UploadFile = File(...)):
    """
    The complete real-world flow: audio in, AUDIO OUT (not JSON).

    Whisper's own language tag is unreliable for Marathi (misdetected as
    Punjabi / Hindi in real tests), so we run our text-based detect_language()
    on the transcript and treat that as authoritative.
    """
    try:
        if models.get("stt") is None:
            raise HTTPException(status_code=503, detail="STT model not loaded")

        temp_path = tempfile.mktemp(suffix=".wav")
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        segments, info = models["stt"].transcribe(temp_path, beam_size=5)
        transcript = "".join(seg.text for seg in segments).strip()
        whisper_language = info.language
        os.remove(temp_path)

        text_detected_language = detect_language(transcript)

        print(f"[STT] Whisper's own tag: {whisper_language} "
              f"(probability: {info.language_probability:.2f})")
        print(f"[STT] Text-based correction: {text_detected_language}")

        if whisper_language != text_detected_language:
            print(f"[STT] MISMATCH - using text-based detection "
                  f"('{text_detected_language}') as authoritative, "
                  f"overriding Whisper's own tag ('{whisper_language}')")

        result = _run_pipeline(transcript, known_language=text_detected_language)

        if result.get("decision") == "BLOCKED" or not result.get("response"):
            raise HTTPException(
                status_code=422,
                detail="Response failed safety validation - no audio generated",
            )

        output_path = tempfile.mktemp(suffix=".wav")
        synthesize_auto(result["response"], output_path=output_path
                       )

        headers = {
            "X-Transcript": _ascii_safe_header(transcript),
            "X-Intent": _ascii_safe_header(result.get("intent", "")),
            "X-Language": _ascii_safe_header(result.get("language", "")),
            "X-Safety-Status": _ascii_safe_header(result.get("safety_status", "")),
            "X-Action-Type": _ascii_safe_header(result.get("action", {}).get("type", "")),
            "X-Expression": _ascii_safe_header(result.get("expression", "")),
            "X-Requires-Confirmation": str(result.get("requires_confirmation", False)).lower(),
        }

        return FileResponse(output_path, media_type="audio/wav", headers=headers)

    except HTTPException:
        raise

    except Exception as e:
        import traceback
        print(f"Unexpected error in /audio: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")



@app.post("/text_audio")
async def text_full_turn(req :  ChatRequest):
    """
    The complete real-world flow: audio in, AUDIO OUT (not JSON).

    Whisper's own language tag is unreliable for Marathi (misdetected as
    Punjabi / Hindi in real tests), so we run our text-based detect_language()
    on the transcript and treat that as authoritative.
    """
    try:
        
        text_input = req.text
        text_detected_language = detect_language(text_input)
        print(f"Language detection : {text_detected_language}")
        # print(f"[STT] Whisper's own tag: {whisper_language} "
        #       f"(probability: {info.language_probability:.2f})")
        # print(f"[STT] Text-based correction: {text_detected_language}")

        # if whisper_language != text_detected_language:
        #     print(f"[STT] MISMATCH - using text-based detection "
        #           f"('{text_detected_language}') as authoritative, "
        #           f"overriding Whisper's own tag ('{whisper_language}')")

        result = _run_pipeline(text_input, known_language=text_detected_language)
        print(f"[RESULT] : {result['response']}")
        if result.get("decision") == "BLOCKED" or not result.get("response"):
            raise HTTPException(
                status_code=422,
                detail="Response failed safety validation - no audio generated",
            )

        output_path = tempfile.mktemp(suffix=".wav")
        synthesize_auto(result["response"], output_path=output_path
                       )

        headers = {
            "X-Transcript": _ascii_safe_header(text_input),
            "X-Intent": _ascii_safe_header(result.get("intent", "")),
            "X-Language": _ascii_safe_header(result.get("language", "")),
            "X-Safety-Status": _ascii_safe_header(result.get("safety_status", "")),
            "X-Action-Type": _ascii_safe_header(result.get("action", {}).get("type", "")),
            "X-Expression": _ascii_safe_header(result.get("expression", "")),
            "X-Requires-Confirmation": str(result.get("requires_confirmation", False)).lower(),
        }

        return FileResponse(output_path, media_type="audio/wav", headers=headers)

    except HTTPException:
        raise

    except Exception as e:
        import traceback
        print(f"Unexpected error in /audio: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
