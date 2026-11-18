import subprocess, time, requests, atexit
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SERVER = BASE_DIR / "llama-b11695-bin-win-cuda-12.4-x64" / "llama-server.exe"
LLM = BASE_DIR / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"
EMBED = BASE_DIR / "models" / "bge-m3-Q4_K_M.gguf"

def start_server(cmd, port):
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    atexit.register(proc.terminate)          # stop server when script exits
    for _ in range(120):                     # wait until model is loaded
        try:
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                return proc
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError(f"Server on port {port} did not start")

# Chat model
start_server([str(SERVER), "-m", str(LLM), "-ngl", "99", "-c", "1024",
              "--host", "127.0.0.1", "--port", "8080"], 8080)

# Embedding model (for RAG)
start_server([str(SERVER), "-m", str(EMBED), "-ngl", "99", "--embedding",
              "--pooling", "cls", "-c", "8192",
              "--host", "127.0.0.1", "--port", "8081"], 8081)


print("Both servers ready. Press Ctrl+C to stop.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping servers...")