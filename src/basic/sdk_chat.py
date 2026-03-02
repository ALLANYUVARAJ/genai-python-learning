import ollama

r = ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "One-line greeting?"}]
)
print(r["message"]["content"])
