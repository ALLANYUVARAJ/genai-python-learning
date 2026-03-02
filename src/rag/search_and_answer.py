import json
from pathlib import Path
import numpy as np
import ollama
import os
from typing import List, Tuple

INDEX_DIR = Path("data/index")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")

def load_index():
    vec_path = INDEX_DIR / "vectors.npz"
    meta_path = INDEX_DIR / "meta.jsonl"
    if not vec_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Index not found. Run: python src/rag/build_index.py")
    X = np.load(vec_path)["vectors"]
    metas = [json.loads(line) for line in meta_path.read_text(encoding="utf-8").splitlines()]
    return X, metas

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

def embed_query(q: str) -> np.ndarray:
    e = ollama.embeddings(model=EMBED_MODEL, prompt=q)["embedding"]
    return np.array(e, dtype=np.float32)

def top_k(query_vec: np.ndarray, X: np.ndarray, metas: List[dict], k: int = 4) -> List[Tuple[float, dict]]:
    sims = X @ (query_vec / (np.linalg.norm(query_vec) + 1e-9))
    idxs = np.argsort(sims)[::-1][:k]
    return [(float(sims[i]), metas[i]) for i in idxs]

def build_prompt(contexts: List[str], question: str) -> str:
    joined = "\n\n---\n\n".join(contexts)
    return (
        "You are a domain assistant. Use the context to answer concisely.\n\n"
        f"Context:\n{joined}\n\n"
        f"Question: {question}\n\n"
        "Instructions:\n"
        "- If unsure, say you don't know.\n"
        "- Cite specific terms when relevant (e.g., DADH, NVM, fax card).\n"
        "Answer:\n"
    )

def main():
    X, metas = load_index()
    while True:
        q = input("Query (or /exit): ").strip()
        if not q or q.lower() in ("/exit", "/quit"):
            break
        qv = embed_query(q)
        hits = top_k(qv, X, metas, k=4)
        ctx = [h[1]["text"] for h in hits]
        prompt = build_prompt(ctx, q)
        ans = ollama.generate(model=LLM_MODEL, prompt=prompt)
        print("\n--- Answer ---")
        print(ans["response"])
        print("\n--- Top context chunks ---")
        for score, m in hits:
            print(f"[{score:.3f}] {m['path']}#chunk{m['chunk_id']}")
        print()

if __name__ == "__main__":
    main()
