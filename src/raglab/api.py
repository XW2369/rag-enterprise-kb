# src/raglab/api.py
"""FastAPI 封装：检索 + ReAct Agent，共 3 个路由。

启动（注意：不要加 --reload，reload 会重复加载模型）：
    .\\.venv\\Scripts\\python.exe -m uvicorn raglab.api:app --host 127.0.0.1 --port 8000
"""
import json
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from raglab.embed_index import Encoder, load_index
from raglab.react import react_agent
from raglab.retrieve import vector_search

PROC = Path(__file__).resolve().parents[2] / "data" / "processed"

app = FastAPI(title="Enterprise KB RAG + Agent API", version="0.1.0")

# ---- 启动时一次性加载（复用已建好的离线索引，不重新编码）----
print("Loading chunks & FAISS index ...")
CHUNKS = [json.loads(line) for line in (PROC / "chunks.jsonl").open(encoding="utf-8")]
INDEX = load_index(PROC / "index.faiss")
ENCODER = Encoder()
print(f"Ready: {len(CHUNKS)} chunks, {INDEX.ntotal} vectors")


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class AgentRequest(BaseModel):
    question: str
    max_steps: int = 5


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "chunks": len(CHUNKS), "vectors": INDEX.ntotal}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    """检索，返回 top-k 片段（抽取式，未接 LLM 生成）。"""
    hits = vector_search(req.question, INDEX, ENCODER, k=req.top_k)
    return {
        "query": req.question,
        "mode": "retrieval_only",
        "retrieval_topk": [
            {
                "rank": rank,
                "score": round(score, 4),
                "source_path": CHUNKS[idx]["source_path"],
                "content": " ".join(CHUNKS[idx]["text"].split())[:300],
            }
            for rank, (idx, score) in enumerate(hits, start=1)
        ],
    }


@app.post("/agent")
def agent(req: AgentRequest) -> dict:
    """跑一次 ReAct 循环，返回完整 Thought/Action/Observation trace。"""
    result = react_agent(req.question, INDEX, ENCODER, CHUNKS, max_steps=req.max_steps)
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
