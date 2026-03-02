import requests

resp = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "llama3.2",
        "messages": [{"role": "user", "content": "Give 3 creative startup ideas."}],
        "options": {
            "temperature": 1.0,
            "top_p": 0.9,
            "max_tokens": 150
        },
        "stream": False
    },
    timeout=60,
)
resp.raise_for_status()
print(resp.json()["message"]["content"])
