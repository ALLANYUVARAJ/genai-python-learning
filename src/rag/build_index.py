import os
import glob
import json
from pathlib import Path
import numpy as np
import ollama

DOCS_GLOB = os.getenv("DOCS_GLOB", "data/docs/**/*.*")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
INDEX_DIR = Path("data/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

def read_docs():
    files = [f for f in glob.glob(DOCS_GLOB, recursive=True)
             if f.lower().endswith((".txt", ".md"))]
    docs = []
    for f in files:
        try:
            text = Path(f).read_text(encoding="utf-8", errors="ignore")
            docs.append({"path": f, "text": text})
        except Exception as e:
            print(f"Skip {f}: {e}")
    return docs

def chunk_text(text, chunk_words=300, overlap_words=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_words]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        i += (chunk_words - overlap_words)
    return chunks

def embed_texts(texts):
    vecs = []
    for t in texts:
        e = ollama.embeddings(model=EMBED_MODEL, prompt=t)
        vecs.append(np.array(e["embedding"], dtype=np.float32))
    return np.vstack(vecs)

def main():
    print("Reading docs...")
    docs = read_docs()
    if not docs:
        print("No documents found in data/docs. Add .txt or .md files and retry.")
        return

    print("Chunking...")
    records = []
    all_chunks = []
    for d in docs:
        chunks = chunk_text(d["text"])
        for idx, ch in enumerate(chunks):
            records.append({"path": d["path"], "chunk_id": idx, "text": ch})
        all_chunks.extend([r["text"] for r in records if r["path"] == d["path"]])
    # NOTE: previous loop appended repeatedly; rebuild clean lists:
    records = []
    all_texts = []
    for d in docs:
        chunks = chunk_text(d["text"])
        for idx, ch in enumerate(chunks):
            records.append({"path": d["path"], "chunk_id": idx, "text": ch})
            all_texts.append(ch)

    print(f"Embedding {len(all_texts)} chunks...")
    X = embed_texts(all_texts)

    # Save vectors and metadata
    np.savez_compressed(INDEX_DIR / "vectors.npz", vectors=X)
    Path(INDEX_DIR / "meta.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records),
        encoding="utf-8"
    )
    print(f"Saved {X.shape[0]} vectors @ dim {X.shape[1]} to {INDEX_DIR}/vectors.npz")
    print(f"Saved metadata to {INDEX_DIR}/meta.jsonl")

if __name__ == "__main__":
    main()
