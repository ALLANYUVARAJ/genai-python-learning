import requests

resp = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.2",
        "prompt": "Hello from Python!",
        "stream": False  # return a single JSON object (no NDJSON)
    },
    timeout=60,
)
resp.raise_for_status()
print(resp.json()["response"])
