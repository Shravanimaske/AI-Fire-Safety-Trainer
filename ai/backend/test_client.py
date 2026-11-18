"""
Simple test client for the AI VR Trainer API - avoids PowerShell's
Unicode-mangling issue with curl -d by using Python's requests library,
which handles UTF-8 correctly by default.

Usage:
    python ai/backend/test_client.py "what is PPE"
    python ai/backend/test_client.py "आग लगने की स्थिति में क्या करना चाहिए?"
"""

import sys
import requests
import json

API_URL = "http://localhost:8001/chat"


def ask(question):
    response = requests.post(
        API_URL,
        json={"text": question},  # requests handles UTF-8 encoding correctly
    )

    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        return

    result = response.json()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python ai/backend/test_client.py "your question here"')
        sys.exit(1)

    ask(sys.argv[1])
