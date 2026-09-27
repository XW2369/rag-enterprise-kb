"""构建 RAG 索引：语料 -> 分块 -> 向量化 -> FAISS 落盘。

用法：
    .\\.venv\\Scripts\\python.exe scripts\\build_index.py --limit 50   # 先小规模验证
    .\\.venv\\Scripts\\python.exe scripts\\build_index.py               # 全量
"""
import argparse
import json
import time
from pathlib import Path

import faiss

from raglab.chunk import build_chunks
from raglab.embed_index import Encoder, build_faiss_index
from raglab.load_docs import load_corpus

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
WORKSPACE_ROOT = r"C:\Users\25466\Desktop\培训专业"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只取前 N 篇文档，0 表示全部")
    args = ap.parse_args()

    t0 = time.time()
    docs = load_corpus(WORKSPACE_ROOT)
    if args.limit:
        docs = docs[: args.limit]
    print(f"loaded docs: {len(docs)}")

    chunks = build_chunks(docs)
    print(f"chunks: {len(chunks)}")

    encoder = Encoder()
    vectors = encoder.encode_docs([c.text for c in chunks])
    print(f"vectors: {vectors.shape}")

    index = build_faiss_index(vectors)
    OUT.mkdir(parents=True, exist_ok=True)
    # ⚠ FAISS 的 C++ write_index 处理不了含中文的绝对路径，改用 Python 侧序列化
    (OUT / "index.faiss").write_bytes(faiss.serialize_index(index))

    with (OUT / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(
                json.dumps(
                    {
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "source_path": c.source_path,
                        "text": c.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"index built: {index.ntotal} vectors -> {OUT / 'index.faiss'}")
    print(f"elapsed: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
