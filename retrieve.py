# src/raglab/retrieve.py

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from src.raglab.embed import embed_texts

from src.raglab.chunk import build_chunks
from src.raglab.load_docs import load_corpus


def vector_search(query: str, index: faiss.IndexFlatIP, encoder, k=20) -> list[tuple[int, float]]:
    """向量检索，query走encode_query带前缀"""
    q_emb = encoder.encode_query(query)
    q_emb = np.expand_dims(q_emb, axis=0)
    # faiss返回距离，索引id
    scores, ids = index.search(q_emb, k)
    res = []
    for idx, s in zip(ids[0], scores[0]):
        if idx != -1:
            res.append((int(idx), float(s)))
    return res


def bm25_search(query: str, bm25: BM25Okapi, k=20) -> list[tuple[int, float]]:
    """中文：字符级切分 list(text)"""
    tokenized_q = list(query)
    scores = bm25.get_scores(tokenized_q)
    # 按分数降序，取topk
    idx_score_pairs = [(i, scores[i]) for i in range(len(scores))]
    idx_score_pairs.sort(key=lambda x:x[1], reverse=True)
    return idx_score_pairs[:k]


def rrf_fuse(rank_lists: list[list[tuple[int,float]]], k=60) -> list[tuple[int, float]]:
    """RRF融合，只用排名，不用原始相似度分数
    score = sum( 1/(k + rank_i) )
    rank_i: 该文档在某一个排序列表里的名次（从1开始）
    """
    doc_rrf = {}
    for one_ranklist in rank_lists:
        for pos, (docid, _score) in enumerate(one_ranklist):
            rank = pos + 1  # rank从1开始
            rrf_s = 1.0 / (k + rank)
            if docid in doc_rrf:
                doc_rrf[docid] += rrf_s
            else:
                doc_rrf[docid] = rrf_s
    # 按RRF分数降序返回
    fused = sorted(doc_rrf.items(), key=lambda x:x[1], reverse=True)
    return fused


if __name__ == "__main__":
    # 1.加载文档、分块
    docs = load_corpus(root=".")
    chunks = build_chunks(docs)
    chunk_texts = [c.text for c in chunks]

    # 2.构建faiss向量索引
    emb_arr = embed_texts(chunk_texts)
    dim = emb_arr.shape[1]
    index = faiss.IndexFlatIP(dim)
    # 归一化，内积等价余弦相似度
    faiss.normalize_L2(emb_arr)
    index.add(emb_arr)

    # 3.构建BM25：字符切分
    tokenized_corpus = [list(t) for t in chunk_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    # 4.查询测试
    test_query = "企业知识库相关问题"
    vec_hits = vector_search(test_query, index, encoder=None, k=20)
    bm25_hits = bm25_search(test_query, bm25, k=20)
    fused_hits = rrf_fuse([vec_hits, bm25_hits], k=60)
    print(f"vector hits count: {len(vec_hits)}")
    print(f"bm25 hits count: {len(bm25_hits)}")
    print(f"RRF fused top5: {fused_hits[:5]}")
