# src/raglab/retrieve.py
"""三路检索：向量检索 / BM25 / RRF 融合。

⚠ 这个模块是「库」，不是「脚本」——直接跑它不会做任何事，
   要跑就写 `if __name__ == "__main__":` 自测块，或用 scripts/eval_recall.py 调它。
"""
import re
from collections.abc import Sequence

import numpy as np

Ranked = list[tuple[int, float]]  # [(chunk_index, score), ...]

# 🔴 混合语料（中文 + 英文标识符）的切分规则：
#   连续 ASCII 字母数字下划线 → 整体保留为 1 个 token（否则 attention 会被切成 9 个字母）
#   单个汉字 → 1 个 token
#   ⚠ 语料与查询必须使用同一个 tokenizer，否则 BM25 统计口径不一致
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def vector_search(query: str, index, encoder, k: int = 20) -> Ranked:
    """向量检索。查询必须走 encode_query（带 BGE 指令前缀）。"""
    qv = encoder.encode_query(query).reshape(1, -1)
    scores, idx = index.search(np.ascontiguousarray(qv), k)
    return [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i != -1]


def bm25_search(query: str, bm25, k: int = 20) -> Ranked:
    """BM25 字面检索。"""
    scores = np.asarray(bm25.get_scores(tokenize(query)))
    order = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in order]


def rrf_fuse(rank_lists: Sequence[Ranked], k: int = 60, top_n: int = 5) -> Ranked:
    """Reciprocal Rank Fusion：用排名而非分数融合（不同检索器分数不可比）。

    score(d) = Σ_i 1 / (k + rank_i(d))
    """
    agg: dict[int, float] = {}
    for ranked in rank_lists:
        for rank, (chunk_idx, _score) in enumerate(ranked, start=1):
            agg[chunk_idx] = agg.get(chunk_idx, 0.0) + 1.0 / (k + rank)
    return sorted(agg.items(), key=lambda x: -x[1])[:top_n]


def recall_at_k(hit_sources: Sequence[str], gold_source: str) -> int:
    """top-k 里是否命中了 gold 文档（返回 1/0）。"""
    return int(any(s == gold_source for s in hit_sources))


if __name__ == "__main__":
    # 自测：加载索引，随便问一句，看能不能拿到片段
    import json
    from pathlib import Path

    from raglab.embed_index import Encoder, load_index

    root = Path(__file__).resolve().parents[2]
    chunks = [
        json.loads(line)
        for line in (root / "data" / "processed" / "chunks.jsonl").open(encoding="utf-8")
    ]
    index = load_index(root / "data" / "processed" / "index.faiss")
    encoder = Encoder()

    q = "LoRA 的可训练参数占比是多少"
    hits = vector_search(q, index, encoder, k=5)
    print(f"query: {q}")
    for rank, (i, s) in enumerate(hits, start=1):
        print(f"  {rank}. score={s:.4f}  {chunks[i]['source_path']}")
        print(f"      {chunks[i]['text'][:80]!r}")
