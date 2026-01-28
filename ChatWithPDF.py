from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
from pathlib import Path
import json
import tempfile
import re

import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from collections import Counter

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = DATA_DIR / 'index.faiss'
DOCS_PATH = DATA_DIR / 'docs.json'

MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'  
EMB_MODEL = None
EMB_DIM = None

# -----------------------------
# FASTAPI & TEMPLATES
# -----------------------------
app = FastAPI(title="ChatWithPDF - RAG API")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# -----------------------------
# PDF UTILITIES
# -----------------------------
def load_text_from_pdf(path: str) -> str:
    text = []
    with open(path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for p in reader.pages:
            try:
                t = p.extract_text()
            except:
                t = None
            if t:
                text.append(t)
    return "\n\n".join(text)

def chunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> List[str]:
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunks.append(" ".join(tokens[i:i+chunk_size]))
        i += chunk_size - overlap
    return chunks

# -----------------------------
# STARTUP: LOAD MODEL
# -----------------------------
@app.on_event("startup")
def load_model():
    global EMB_MODEL, EMB_DIM
    EMB_MODEL = SentenceTransformer(MODEL_NAME)
    EMB_DIM = EMB_MODEL.get_sentence_embedding_dimension()

# -----------------------------
# INGEST ENDPOINT
# -----------------------------
@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload a PDF file")
    
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp.write(await file.read())
    temp.close()

    text = load_text_from_pdf(temp.name)
    if not text.strip():
        raise HTTPException(status_code=400, detail="PDF has no extractable text")

    chunks = chunk_text(text)
    embeddings = EMB_MODEL.encode(chunks, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(EMB_DIM)
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_PATH))

    docs = [{"id": i, "text": chunks[i]} for i in range(len(chunks))]
    json.dump(docs, open(DOCS_PATH, "w", encoding="utf-8"), ensure_ascii=False)

    if INDEX_PATH.exists() and DOCS_PATH.exists():
        return {"status": "ok", "chunks": len(chunks)}
    else:
        raise HTTPException(status_code=500, detail="Failed to save index or docs")

# -----------------------------
# RETRIEVAL WITH RE-RANKING
# -----------------------------
def load_index_and_docs():
    if not INDEX_PATH.exists() or not DOCS_PATH.exists():
        return None, None
    index = faiss.read_index(str(INDEX_PATH))
    docs = json.load(open(DOCS_PATH, "r", encoding="utf-8"))
    return index, docs

def retrieve(question: str, k: int):
    index, docs = load_index_and_docs()
    if index is None:
        raise RuntimeError("Index not found. Please ingest a PDF first.")

    q_emb = EMB_MODEL.encode([question], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    
    results = []
    for d, idx in zip(D[0], I[0]):
        if 0 <= idx < len(docs):
            results.append({"score": float(d), "text": docs[idx]["text"]})

    # Re-ranking: choose chunk with most words matching question
    q_words = set(re.findall(r"\w+", question.lower()))
    for r in results:
        r_words = set(re.findall(r"\w+", r["text"].lower()))
        r["relevance"] = len(q_words & r_words)
    
    results.sort(key=lambda x: (x["relevance"], x["score"]), reverse=True)
    return results

# -----------------------------
# QUERY MODEL
# -----------------------------
class Query(BaseModel):
    question: str
    top_k: int = 4

@app.post("/query")
async def query_api(req: Query):
    ctx = retrieve(req.question, req.top_k)
    if not ctx:
        return {"answer": "No relevant content found.", "contexts_used": []}
    
    best_chunk = ctx[0]
    answer = f"{best_chunk['text'][:300]}..." if len(best_chunk['text']) > 300 else best_chunk['text']

    return {"answer": answer, "contexts_used": ctx}

# -----------------------------
# STATUS
# -----------------------------
@app.get("/status")
async def status():
    return {"index_exists": INDEX_PATH.exists(), "docs_count": len(json.load(open(DOCS_PATH))) if DOCS_PATH.exists() else 0}

# -----------------------------
# HOME
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ChatWithPDF:app", host="0.0.0.0", port=8000, reload=True)
