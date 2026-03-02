from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama
import os
from typing import List, Dict
import numpy as np

app = FastAPI(title="Ollama GenAI API", version="0.1.0")

CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# ----------- Schemas -----------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    stream: bool = False
    options: Dict = {}

class GenerateRequest(BaseModel):
    prompt: str
    stream: bool = False
    options: Dict = {}

class EmbedRequest(BaseModel):
    texts: List[str]

class RagRequest(BaseModel):
    question: str
    top_k: int = 4

# ----------- Utils -----------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

def load_index():
    from pathlib import Path
    import json
    vec_path = Path("data/index/vectors.npz")
    meta_path = Path("data/index/meta.jsonl")
    if not vec_path.exists() or not meta_path.exists():
        raise FileNotFoundError("RAG index files not found. Build with src/rag/build_index.py")
    X = np.load(vec_path)["vectors"]
    metas = [json.loads(line) for line in Path(meta_path).read_text(encoding="utf-8").splitlines()]
    return X, metas

# ----------- Routes -----------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        payload = {
            "model": CHAT_MODEL,
            "messages": [m.model_dump() for m in req.messages],
        }
        # options like temperature, top_p, max_tokens etc.
        if req.options:
            payload["options"] = req.options

        if req.stream:
            # stream via SDK; here we keep it simple: non-stream recommended for APIs
            chunks = []
            for part in ollama.chat(stream=True, **payload):
                chunks.append(part["message"]["content"])
            return {"message": {"role": "assistant", "content": "".join(chunks)}}
        else:
            resp = ollama.chat(**payload)
            return {"message": resp["message"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
def generate(req: GenerateRequest):
    try:
        payload = {
            "model": CHAT_MODEL,
            "prompt": req.prompt,
        }
        if req.options:
            payload["options"] = req.options

        if req.stream:
            chunks = []
            for part in ollama.generate(stream=True, **payload):
                chunks.append(part["response"])
            return {"response": "".join(chunks)}
        else:
            resp = ollama.generate(**payload)
            return {"response": resp["response"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embeddings")
def embeddings(req: EmbedRequest):
    try:
        vectors = []
        for t in req.texts:
            e = ollama.embeddings(model=EMBED_MODEL, prompt=t)
            vectors.append(e["embedding"])
        return {"vectors": vectors, "dim": len(vectors[0]) if vectors else 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/ask")
def rag_ask(req: RagRequest):
    try:
        X, metas = load_index()
        q_vec = np.array(ollama.embeddings(model=EMBED_MODEL, prompt=req.question)["embedding"], dtype=np.float32)
        sims = X @ (q_vec / (np.linalg.norm(q_vec) + 1e-9))
        idxs = np.argsort(sims)[::-1][:req.top_k]
        ctx = [metas[i]["text"] for i in idxs]
        prompt = (
            "Use the context to answer factually and concisely.\n\n"
            f"Context:\n{'\n\n---\n\n'.join(ctx)}\n\n"
            f"Question: {req.question}\n\nAnswer:\n"
        )
        ans = ollama.generate(model=CHAT_MODEL, prompt=prompt)
        return {
            "answer": ans["response"],
            "chunks": [
                {"score": float(sims[i]), "path": metas[i]["path"], "chunk_id": metas[i]["chunk_id"]}
                for i in idxs
            ],
        }
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
