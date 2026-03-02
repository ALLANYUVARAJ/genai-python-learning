import requests

texts = [
    "Printer DADH handles duplex scanning.",
    "The fax card is required for sending faxes."
]

vectors = []
for t in texts:
    e = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": t},
        timeout=60,
    )
    e.raise_for_status()
    vectors.append(e.json()["embedding"])

print("Num vectors:", len(vectors), "| Dim:", len(vectors[0]))
