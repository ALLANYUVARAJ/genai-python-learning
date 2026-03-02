import requests

def count_tokens(text: str) -> int:
    r = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": text},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    # Some Ollama builds include token counts in embedding responses.
    # Fallback to len(embedding) if "tokens" is not present.
    return data.get("tokens") or len(data.get("embedding", []))

sample = "This is a sample text to estimate tokens in Ollama."
print("Estimated tokens:", count_tokens(sample))
