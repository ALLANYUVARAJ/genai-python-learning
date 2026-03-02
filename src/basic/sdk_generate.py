import ollama

# Non-streaming
r = ollama.generate(model="llama3.2", prompt="Hello from SDK.")
print(r["response"])

# Streaming
for part in ollama.generate(model="llama3.2", prompt="Stream with SDK.", stream=True):
    print(part["response"], end="", flush=True)
print()
