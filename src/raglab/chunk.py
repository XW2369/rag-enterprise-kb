# src/raglab/chunk.py
from dataclasses import dataclass

from raglab.load_docs import Doc, load_corpus


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source_path: str
    text: str


def recursive_chunk(text: str, size: int = 300, overlap: int = 50) -> list[str]:
    """简单递归字符切分，分隔符优先级：换行 > 空格 > 直接硬切"""
    separators = ["\n\n", "\n", " ", ""]

    def split_recursive(s: str, sep_idx: int) -> list[str]:
        chunks = []
        sep = separators[sep_idx]

        # 🔴 最后一级是「硬切」：不能再调 s.split("")（会抛 ValueError: empty separator）
        if sep == "":
            step = max(1, size - overlap)
            return [s[i:i + size] for i in range(0, len(s), step)]

        splits = s.split(sep)
        current = ""
        for part in splits:
            if len(current) + len(sep) + len(part) <= size:
                if current:
                    current += sep
                current += part
            else:
                if current:
                    chunks.append(current)
                # 当前分隔符切出来还是太长，往下一级分隔符递归
                if len(part) > size and sep_idx + 1 < len(separators):
                    sub_chunks = split_recursive(part, sep_idx + 1)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(part)
                # 滑动窗口overlap：保留末尾overlap长度文本作为下一段开头
                if len(current) > overlap:
                    current = current[-overlap:]
                else:
                    current = ""
        if current:
            chunks.append(current)
        return chunks

    return split_recursive(text, sep_idx=0)


def build_chunks(docs: list[Doc]) -> list[Chunk]:
    chunk_list: list[Chunk] = []
    chunk_counter = 0
    for doc in docs:
        text_pieces = recursive_chunk(doc.text, size=300, overlap=50)
        for piece in text_pieces:
            chunk_id = f"chunk_{chunk_counter}"
            chunk_list.append(Chunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                source_path=doc.source_path,
                text=piece
            ))
            chunk_counter += 1
    print(f"chunks: {len(chunk_list)}")
    return chunk_list


if __name__ == "__main__":
    WORKSPACE_ROOT = r"C:\Users\25466\Desktop\培训专业"
    docs = load_corpus(root=WORKSPACE_ROOT)
    print(f"loaded docs: {len(docs)}")
    chunks = build_chunks(docs)
    for c in chunks[:3]:
        print("   ", c.chunk_id, "|", c.source_path)
