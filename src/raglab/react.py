# src/raglab/react.py
"""手写 ReAct 循环：Thought → Action → Observation，直到 Answer。

⚠ 诚实声明（必须写进 README / 打卡）：
   本项目**未接大模型 API**（B 路径），因此 `Thought` 这一步由 `RuleBasedLLM` 占位：
   它用关键词规则从问题里抽检索词，而不是真正的语言模型推理。
   循环结构、工具调用、Observation 回灌、步数截断**都是真实运行的**，
   只要把 `llm` 换成 `OpenAICompatLLM`（见文末注释）就是完整的 ReAct Agent。
"""
import re
from collections.abc import Callable
from typing import Any

from raglab.retrieve import vector_search


def search_knowledge_base(query: str, index, encoder, chunks: list[dict], k: int = 3) -> str:
    """工具：知识库检索。返回 top-k 文本片段（Observation）。"""
    hits = vector_search(query, index, encoder, k=k)
    lines = []
    for idx, score in hits:
        c = chunks[idx]
        text = " ".join(c["text"].split())[:200]
        lines.append(f"[{c['source_path']}] score={score:.4f}\n{text}")
    return "\n".join(lines) if lines else "（无结果）"


class RuleBasedLLM:
    """占位 LLM：用关键词规则替代大模型决策。

    不是语言模型推理，只保证 ReAct 循环能真实跑起来。
    """

    def __call__(self, question: str, scratchpad: str) -> str:
        if "Observation:" in scratchpad:
            return "Answer: 已在知识库中检索到相关片段，见上方 Observation。"
        stopwords = {"是什么", "怎么用", "有哪些", "如何", "的", "了", "吗", "？", "?", "、"}
        terms = [t for t in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", question) if t not in stopwords]
        query = " ".join(terms[:6]) or question
        return f'Action: search_knowledge_base("{query}")'


# 接上真 LLM 时把它传给 react_agent 即可（需自备 API key）：
# class OpenAICompatLLM:
#     def __init__(self, base_url, api_key, model):
#         ...
#     def __call__(self, question, scratchpad) -> str: ...


def react_agent(
    user_query: str,
    index,
    encoder,
    chunks: list[dict],
    llm: Callable[[str, str], str] | None = None,
    max_steps: int = 5,
) -> dict[str, Any]:
    """手写 ReAct 主循环。

    每一步：llm 输出 -> 解析 Action -> 调工具 -> 把 Observation 回灌 scratchpad。
    """
    llm = llm or RuleBasedLLM()
    scratchpad = ""
    trace: list[dict[str, Any]] = []

    for step in range(1, max_steps + 1):
        output = llm(user_query, scratchpad).strip()
        thought = ""
        if output.startswith("Thought:"):
            head, _, rest = output.partition("\n")
            thought, output = head[len("Thought:"):].strip(), rest.strip()

        trace.append({"step": step, "thought": thought, "raw": output})

        if output.startswith("Answer:"):
            return {
                "question": user_query,
                "answer": output[len("Answer:"):].strip(),
                "steps": step,
                "trace": trace,
            }

        m = re.search(r'Action:\s*search_knowledge_base\("(.+?)"\)', output)
        if not m:
            trace[-1]["error"] = "无法解析 Action"
            return {
                "question": user_query,
                "answer": "解析失败，已终止。",
                "steps": step,
                "trace": trace,
            }

        action_query = m.group(1)
        observation = search_knowledge_base(action_query, index, encoder, chunks)
        trace[-1]["action"] = f"search_knowledge_base({action_query!r})"
        trace[-1]["observation"] = observation
        scratchpad += f"\nThought: {thought}\nAction: search_knowledge_base(\"{action_query}\")\nObservation: {observation}\n"

    return {
        "question": user_query,
        "answer": f"达到最大步数 {max_steps}，已截断。",
        "steps": max_steps,
        "trace": trace,
    }


if __name__ == "__main__":
    import json
    from pathlib import Path

    from raglab.embed_index import Encoder, load_index

    root = Path(__file__).resolve().parents[2]
    proc = root / "data" / "processed"
    chunks = [json.loads(line) for line in (proc / "chunks.jsonl").open(encoding="utf-8")]
    index = load_index(proc / "index.faiss")
    encoder = Encoder()

    result = react_agent("LoRA 的可训练参数占比是多少？", index, encoder, chunks)
    for item in result["trace"]:
        print(f"[step {item['step']}] action={item.get('action')}")
        if item.get("observation"):
            print("   obs:", " ".join(item["observation"].split())[:120], "...")
    print("\nAnswer:", result["answer"])
