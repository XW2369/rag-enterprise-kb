# src/raglab/load_docs.py
"""扫描工作区，构建 RAG 语料。

语料来源：7 天冲刺营积累的全部工程文档（md / py / txt），排除虚拟环境与工具缓存。
"""
from dataclasses import dataclass
from pathlib import Path

SKIP_DIRS = {
    ".venv",
    ".git",
    "__pycache__",
    ".workbuddy",
    ".pytest_cache",
    ".ruff_cache",
    ".egg-info",
}


@dataclass
class Doc:
    doc_id: str
    source_path: str
    text: str


def load_corpus(root, exts=(".md", ".py", ".txt")) -> list[Doc]:
    root_path = Path(root)
    docs: list[Doc] = []

    for file in root_path.rglob("*"):
        # 只要路径任意一段在跳过名单里，直接跳过
        if any(seg in SKIP_DIRS for seg in file.parts):
            continue
        if not file.is_file():
            continue
        if file.suffix not in exts:
            continue
        try:
            content = file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if not content.strip():
            continue

        docs.append(Doc(doc_id=str(file), source_path=str(file), text=content))

    return docs


if __name__ == "__main__":
    WORKSPACE_ROOT = r"C:\Users\25466\Desktop\培训专业"
    docs = load_corpus(root=WORKSPACE_ROOT)
    print(f"root = {WORKSPACE_ROOT}")
    print(f"loaded docs: {len(docs)}")
    for d in docs[:5]:
        print("   ", d.source_path)
