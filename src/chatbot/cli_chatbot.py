import json
import os
from pathlib import Path
import ollama

HISTORY_PATH = Path("data/chat_history.json")
MODEL = os.getenv("CHAT_MODEL", "llama3.2")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant.")

def load_history():
    if HISTORY_PATH.exists():
        try:
            return json.loads(HISTORY_PATH.read_text())
        except Exception:
            pass
    return [{"role": "system", "content": SYSTEM_PROMPT}]

def save_history(history):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(history, indent=2))

def main():
    print("CLI Chatbot (Ollama). Commands: /reset, /save, /exit")
    history = load_history()

    while True:
        try:
            user = input("You: ").strip()
            if not user:
                continue
            if user.lower() in ("/exit", "/quit"):
                save_history(history)
                print("Saved & exiting. Bye!")
                break
            if user.lower() == "/reset":
                history = [{"role": "system", "content": SYSTEM_PROMPT}]
                print("History cleared.")
                continue
            if user.lower() == "/save":
                save_history(history)
                print(f"History saved to {HISTORY_PATH}")
                continue

            history.append({"role": "user", "content": user})
            resp = ollama.chat(model=MODEL, messages=history)
            content = resp["message"]["content"]
            print(f"Bot: {content}\n")
            history.append({"role": "assistant", "content": content})
        except KeyboardInterrupt:
            print("\nCtrl+C detected. Saving history and exiting.")
            save_history(history)
            break

if __name__ == "__main__":
    main()
