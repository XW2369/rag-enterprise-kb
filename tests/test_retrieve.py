"""检索链路的单元测试（不依赖网络；索引已离线建好）。

跑：.venv\\Scripts\\python.exe -m pytest -q
"""
import json
from pathlib import Path

import numpy as np
import pytest

from raglab.chunk import Chunk, build_chunks, recursive_chunk
from raglab.load_docs import Doc
from raglab.retrieve import recall_at_k, rrf_fuse, tokenize

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"


# ---------------- 分块 ----------------

def test_tokenize_keeps_ascii_words_whole():
    """英文标识符不能被拆成字母（否则 BM25 在混合语料上直接失效）。"""
    tokens = tokenize("LoRA 的可训练参数 bge-small-zh-v1.5")
    assert "LoRA" in tokens
    assert "bge" in tokens and "small" in tokens
    assert "的" in tokens
    # 不该出现单字母 token
    assert not [t for t in tokens if len(t) == 1 and t.isascii() and t.isalpha()]


def test_recursive_chunk_respects_size():
    text = "啊" * 1000
    pieces = recursive_chunk(text, size=100, overlap=10)
    assert len(pieces) > 1
    assert all(len(p) <= 100 for p in pieces)


def test_chunks_keep_source_path():
    """没有 source_path 就无法评估召回率。"""
    docs = [Doc(doc_id="d1", source_path="/tmp/a.md", text="内容" * 200)]
    chunks = build_chunks(docs)
    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.source_path == "/tmp/a.md" for c in chunks)
    assert all(c.doc_id == "d1" for c in chunks)


# ---------------- 融合 ----------------

def test_rrf_fuse_prefers_docs_in_both_lists():
    vec = [(1, 0.9), (2, 0.8), (3, 0.7)]
    bm25 = [(2, 12.0), (4, 11.0), (1, 10.0)]
    fused = rrf_fuse([vec, bm25], k=60, top_n=3)
    order = [i for i, _ in fused]
    # 1 和 2 出现在两路里，应该排在最前
    assert set(order[:2]) == {1, 2}
    # 分数必须单调递减
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_fuse_empty_list():
    assert rrf_fuse([], k=60, top_n=5) == []


# ---------------- 召回指标 ----------------

def test_recall_at_k_hit_and_miss():
    assert recall_at_k(["a.md", "b.md"], "b.md") == 1
    assert recall_at_k(["a.md", "b.md"], "c.md") == 0
    assert recall_at_k([], "a.md") == 0


# ---------------- 离线产物 ----------------

def test_offline_index_and_chunks_exist():
    index_path = PROC / "index.faiss"
    chunks_path = PROC / "chunks.jsonl"
    assert index_path.exists(), "先跑 scripts/build_index.py"
    assert chunks_path.exists()
    n_chunks = sum(1 for _ in chunks_path.open(encoding="utf-8"))
    assert n_chunks >= 100

    import faiss

    index = faiss.deserialize_index(
        np.frombuffer(index_path.read_bytes(), dtype="uint8")
    )
    assert index.ntotal == n_chunks


def test_chunks_have_required_fields():
    with (PROC / "chunks.jsonl").open(encoding="utf-8") as f:
        first = json.loads(f.readline())
    for key in ("chunk_id", "doc_id", "source_path", "text"):
        assert key in first
        assert first[key]


@pytest.mark.parametrize("k", [1, 3, 5])
def test_doc_level_pool_size(k):
    """文档级聚合的 top-k 不会超过池子大小。"""
    chunks = [{"source_path": f"doc{i}.md"} for i in range(10)]
    from scripts.eval_recall import doc_level_topk

    ranked = [(i, 1.0 - i * 0.01) for i in range(10)]
    assert len(doc_level_topk(ranked, chunks, k=k)) == k
