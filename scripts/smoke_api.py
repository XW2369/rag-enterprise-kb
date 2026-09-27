"""API 冒烟测试：不需要起 uvicorn，不需要端口，直接调三个路由。

用法：
    .\\.venv\\Scripts\\python.exe scripts\\smoke_api.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient

from raglab.api import app


def main() -> None:
    client = TestClient(app)

    r = client.get("/health")
    print("[GET /health]", r.status_code, r.json())

    r = client.post("/ask", json={"question": "LoRA 的可训练参数占比是多少", "top_k": 3})
    print("[POST /ask]", r.status_code)
    for item in r.json()["retrieval_topk"]:
        print(f"   {item['rank']}. score={item['score']:.4f}  {item['source_path']}")
    ok_ask = r.status_code == 200 and len(r.json()["retrieval_topk"]) == 3

    r = client.post("/agent", json={"question": "LoRA 的可训练参数占比", "max_steps": 3})
    body = r.json()
    print("[POST /agent]", r.status_code, f"steps={body['steps']}")
    for item in body["trace"]:
        print(f"   step {item['step']}: {item.get('action')}")
    print("   answer:", body["answer"][:80])
    ok_agent = r.status_code == 200 and body["steps"] >= 1

    print()
    print("ASK_OK" if ok_ask else "ASK_FAIL", "|", "AGENT_OK" if ok_agent else "AGENT_FAIL")


if __name__ == "__main__":
    main()
