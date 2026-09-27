# src/raglab/embed_index.py
"""Embedding + FAISS 索引。

模型：BAAI/bge-small-zh-v1.5（512 维，中文语料专用）
要点：
1. 归一化后内积 == 余弦，所以索引用 IndexFlatIP
2. BGE 官方约定：查询侧加指令前缀，文档侧不加（漏了会明显掉召回）
3. 模型已缓存在本地 → **强制离线模式**，避免每次启动都去 Hub 发 HEAD 请求
   （hf-mirror 抖动时会连报 `WinError 10060` 并重试 5 次，白等 1 分钟以上）
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


class Encoder:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def encode_docs(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """文档侧编码：不加前缀，L2 归一化。"""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

    def encode_query(self, query: str) -> np.ndarray:
        """查询侧编码：加 BGE 指令前缀。"""
        if not query.startswith(QUERY_PREFIX):
            query = QUERY_PREFIX + query
        return self.encode_docs([query])[0]


def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """归一化向量 -> 内积索引（等价于余弦检索）。"""
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def save_index(index: faiss.Index, path) -> None:
    """保存索引。

    ⚠ 不能用 faiss.write_index：它的 C++ 实现处理不了含中文的绝对路径
    （报 could not open ... for writing）。改用 Python 侧序列化。
    """
    from pathlib import Path

    Path(path).write_bytes(faiss.serialize_index(index))


def load_index(path) -> faiss.Index:
    from pathlib import Path

    # deserialize_index 要的是 uint8 ndarray，不是裸 bytes
    buf = np.frombuffer(Path(path).read_bytes(), dtype="uint8")
    return faiss.deserialize_index(buf)
