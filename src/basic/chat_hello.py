import requests

resp = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "llama3.2",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in one line."}
        ],
        "stream": False
    },
    timeout=60,
)
resp.raise_for_status()
data = resp.json()
print(data["message"]["content"])
