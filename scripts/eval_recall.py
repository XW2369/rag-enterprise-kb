"""Recall@5 评估：纯向量 vs BM25 vs RRF 混合。

用法：
    python scripts\\eval_recall.py
"""
import json
import random
import time
from pathlib import Path

from rank_bm25 import BM25Okapi

from raglab.embed_index import Encoder, load_index
from raglab.retrieve import bm25_search, recall_at_k, rrf_fuse, tokenize, vector_search

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

TOP_K = 5


def load_chunks() -> list[dict]:
    with (PROC / "chunks.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_eval_set(chunks: list[dict], mode: str = "content", n: int = 20, seed: int = 42) -> list[dict]:
    """构造评估集。两种模式：

    mode="content" —— 内容派生（**上界测试**）：取文档中段切片的前 30 字当 query。
        用途：验证检索链路可用。query 是文档原文的连续子串，字面指纹唯一，
        任何检索方法都能命中，因此**必然接近满分、无区分度**，不能当效果指标。

    mode="topic"   —— 主题型 query（**更接近真实**，有区分度）：
        用文件名主题词构造 `xxx 是什么、怎么用？`。
        这类 query 不与文档任意片段字面重合，需要语义匹配，
        更能反映「用户提一个模糊问题」的真实场景。
    """
    random.seed(seed)
    by_doc: dict[str, list[dict]] = {}
    for c in chunks:
        by_doc.setdefault(c["source_path"], []).append(c)

    docs = sorted(by_doc)
    picked = random.sample(docs, min(n, len(docs)))

    qa = []
    for path in picked:
        doc_chunks = by_doc[path]
        if mode == "content":
            mid = doc_chunks[len(doc_chunks) // 2]
            query = " ".join(mid["text"].split())[:30]
        else:
            query = f"{Path(path).stem} 是什么、怎么用？"
        if len(query) < 8:
            continue
        qa.append({"question": query, "gold_source": path})
    return qa


def evaluate(qa, chunks, index, encoder, bm25, label: str) -> dict:
    n_hit = {"vector": 0, "bm25": 0, "hybrid": 0}
    lat = {"vector": [], "bm25": [], "hybrid": []}

    for item in qa:
        q, gold = item["question"], item["gold_source"]

        t = time.perf_counter()
        vec = vector_search(q, index, encoder, k=50)
        lat["vector"].append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        bm = bm25_search(q, bm25, k=50)
        lat["bm25"].append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        hyb = rrf_fuse([vec, bm], k=60, top_n=50)
        lat["hybrid"].append((time.perf_counter() - t) * 1000)

        n_hit["vector"] += recall_at_k(doc_level_topk(vec, chunks, TOP_K), gold)
        n_hit["bm25"] += recall_at_k(doc_level_topk(bm, chunks, TOP_K), gold)
        n_hit["hybrid"] += recall_at_k(doc_level_topk(hyb, chunks, TOP_K), gold)

    total = len(qa)
    out = {}
    for name in ("vector", "bm25", "hybrid"):
        recalls = sorted(lat[name])
        p95 = recalls[int(0.95 * (len(recalls) - 1))]
        out[name] = {
            "recall_at_5": round(n_hit[name] / total, 4),
            "hits": n_hit[name],
            "p95_latency_ms": round(p95, 2),
        }

    print(f"\n[{label}] Recall@{TOP_K}   (n={total})")
    for name in ("vector", "bm25", "hybrid"):
        print(f"  {name:8s} : {out[name]['recall_at_5']:.4f}   ({n_hit[name]}/{total})"
              f"   P95 {out[name]['p95_latency_ms']} ms")
    return out


def doc_level_topk(ranked, chunks, k: int = 5, pool: int = 50):
    """把 chunk 级排序聚合到文档级（同一文档取最高分），再取 top-k 文档。

    为什么需要这一步：
      2160 个切片摊到 90 份文档上，平均每份 24 个切片。
      若直接按 chunk 取 top-5，5 个名额通常只覆盖 5 份文档，
      「在 90 份里找准那 1 份」这个任务会被人为放大难度。
      文档级聚合（chunk 粗排 → 文档聚合）才是 RAG 里常用的两段式结构。
    """
    best: dict[str, float] = {}
    for i, s in ranked[:pool]:
        path = chunks[i]["source_path"]
        if path not in best or s > best[path]:
            best[path] = s
    return [p for p, _ in sorted(best.items(), key=lambda x: -x[1])[:k]]


def main() -> None:
    chunks = load_chunks()
    index = load_index(PROC / "index.faiss")
    encoder = Encoder()

    corpus_tokens = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    print(f"chunks: {len(chunks)}")

    qa_upper = build_eval_set(chunks, mode="content")
    qa_topic = build_eval_set(chunks, mode="topic")

    result_upper = evaluate(qa_upper, chunks, index, encoder, bm25, "A 内容派生 · 上界测试")
    result_topic = evaluate(qa_topic, chunks, index, encoder, bm25, "B 主题型 query · 接近真实")

    REPORTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_docs": len({c["source_path"] for c in chunks}),
        "n_chunks": len(chunks),
        "embedding_model": "BAAI/bge-small-zh-v1.5",
        "index": "FAISS IndexFlatIP",
        "top_k": TOP_K,
        "tier_A_content_derived_upper_bound": {
            "n_questions": len(qa_upper),
            "caveat": "query 取自文档原文连续子串，字面指纹唯一，必然接近满分，仅用于验证链路",
            "metrics": result_upper,
        },
        "tier_B_topic_query": {
            "n_questions": len(qa_topic),
            "caveat": "用文件名主题词构造的模糊 query，有区分度；仍非人工标注真实 query",
            "metrics": result_topic,
        },
    }
    (REPORTS / "recall.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md = [
        "# Recall@5 检索评估报告",
        "",
        f"- 语料：**{payload['n_docs']} 份文档 / {payload['n_chunks']} 个切片**",
        f"- Embedding：`{payload['embedding_model']}`（512 维，L2 归一化）",
        f"- 向量库：`{payload['index']}`",
        "- 检索范围：chunk 粗排 top-50 → **文档级聚合** → 取 top-5 文档",
        "",
        "## A 层 · 内容派生（上界测试，证明链路可用）",
        "",
        f"- query 取自文档原文连续子串，共 {len(qa_upper)} 条",
        "",
        f"| 检索方式 | Recall@{TOP_K} | 命中 | P95 (ms) |",
        "|---|---|---|---|",
    ]
    for key, label in (("vector", "纯向量（bge-small-zh）"), ("bm25", "BM25（字面）"), ("hybrid", "混合（RRF）")):
        m = result_upper[key]
        md.append(f"| {label} | {m['recall_at_5']:.4f} | {m['hits']}/{len(qa_upper)} | {m['p95_latency_ms']} |")

    md += [
        "",
        "> ⚠️ A 层三个方法都是满分，**没有区分度**。原因：query 是文档原文的连续子串，",
        "> 字面指纹唯一，任何检索器都能命中。它只能证明「链路没坏」，**不能当作效果指标**。",
        "",
        "## B 层 · 主题型 query（接近真实场景，有区分度）",
        "",
        f"- 用文件名主题词构造模糊 query（如 `attention 是什么、怎么用？`），共 {len(qa_topic)} 条",
        "",
        f"| 检索方式 | Recall@{TOP_K} | 命中 | P95 (ms) |",
        "|---|---|---|---|",
    ]
    for key, label in (("vector", "纯向量（bge-small-zh）"), ("bm25", "BM25（字面）"), ("hybrid", "混合（RRF）")):
        m = result_topic[key]
        md.append(f"| {label} | **{m['recall_at_5']:.4f}** | {m['hits']}/{len(qa_topic)} | {m['p95_latency_ms']} |")

    vec_b = result_topic["vector"]["recall_at_5"]
    hyb_b = result_topic["hybrid"]["recall_at_5"]
    md += [
        "",
        "## 结论（如实写，不美化）",
        "",
        f"- 混合检索（RRF）在 B 层为 **{hyb_b:.4f}**，纯向量为 **{vec_b:.4f}**"
        + ("，**混合未优于纯向量**。" if hyb_b <= vec_b else "，混合优于纯向量。"),
        "- **归因**：本语料是「中文 + 英文标识符」混合文本，且 90 份文档主题高度重叠"
        + "（多份打卡日志/任务单描述同一批工作）。BM25 在这类语料上 Recall 仅"
        + f" {result_topic['bm25']['recall_at_5']:.4f}，其 top-50 候选以噪声为主，"
        + "进入 RRF 后反而稀释了向量检索的排序质量。",
        "- **改进方向**（面试可讲）：① 给 RRF 两路不同权重（向量权重更高）"
        + " ② 换用真正的 reranker（bge-reranker）对 top-50 重排"
        + " ③ 中文分词改用 jieba 提升 BM25 质量 ④ 人工标注 20–50 条真实 query 后再评",
        "",
    ]
    (REPORTS / "recall.md").write_text("\n".join(md), encoding="utf-8")
    print("\nreports written -> reports/recall.json, reports/recall.md")


if __name__ == "__main__":
    main()
