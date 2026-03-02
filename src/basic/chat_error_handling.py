import requests

try:
    r = requests.post(
        "http://localhost:11434/api/chat",
        json={"model": "llama3.2", "messages": [{"role": "user", "content": "ping"}], "stream": False},
        timeout=10,
    )
    r.raise_for_status()
    print(r.json()["message"]["content"])

except requests.exceptions.ConnectionError:
    print("❌ Ollama is not running. Start it with: ollama serve &")

except requests.exceptions.Timeout:
    print("❌ Request timed out. Try increasing timeout or reducing prompt size.")

except requests.exceptions.HTTPError as he:
    print("❌ HTTP error:", he, "| Body:", r.text)

except Exception as e:
    print("❌ Unexpected error:", str(e))
