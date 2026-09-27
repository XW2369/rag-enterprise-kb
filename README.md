# 05-企业知识库RAG系统

> Day6（补做）：RAG 全链路 + 手写 ReAct Agent + FastAPI ｜ CPU 环境（无显卡）
> 语料：本冲刺营 **98 份工程文档**（md / py / txt，92 KB 级）→ **2341 个切片**

---

## 一、它解决什么问题

把「7 天积累的全部工程文档」变成一个可检索的知识库：
**给一句话提问，返回最相关的文档片段**，并支持 Agent 以 ReAct 方式多步检索。

---

## 二、架构（数据流）

```
原始文档 (98 份)
    │  load_docs.load_corpus()         递归扫描，排除 .venv/.git/__pycache__/.workbuddy/.egg-info
    ▼
切片 (2341 个)
    │  chunk.recursive_chunk()         size=300, overlap=50；分隔符优先级 \n\n > \n > 空格 > 硬切
    │                                  ★ 每个切片保留 source_path 元数据（否则无法评估召回）
    ▼
向量 (2341 × 512, float32)
    │  embed_index.Encoder             BAAI/bge-small-zh-v1.5，L2 归一化
    ▼
FAISS 索引
    │  faiss.IndexFlatIP               归一化后内积 == 余弦
    ▼
检索                                       Agent（手写 ReAct）
    ├── vector_search  语义相似            ┌──────────────────────────┐
    ├── bm25_search    字面匹配            │ Thought → Action → Obs.  │
    └── rrf_fuse       RRF 排名融合        │ 工具：search_knowledge_  │
                                           │      base()              │
                                           │ max_steps=5 截断         │
                                           └──────────────────────────┘
    ▼
FastAPI 三路由：GET /health ｜ POST /ask ｜ POST /agent
```

---

## 三、快速开始

本项目使用 `.venv`（**不执行 activate，直接用解释器绝对路径**）：

```powershell
# 依赖（uv 建的 venv 不含 pip，用 uv pip）
uv pip install -e .

# 1) 建索引（CPU 约 80 秒）
.\.venv\Scripts\python.exe scripts\build_index.py

# 2) 召回率评估（两层）
.\.venv\Scripts\python.exe scripts\eval_recall.py

# 3) API 冒烟（不用起服务，直接调三个路由）
.\.venv\Scripts\python.exe scripts\smoke_api.py

# 4) 真起服务（另开一个窗口，保持不关）
.\.venv\Scripts\python.exe -m uvicorn raglab.api:app --host 127.0.0.1 --port 8000
# 然后另开窗口：
curl.exe -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\":\"LoRA 的可训练参数占比是多少\"}"

# 5) 测试
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

---

## 四、实测结果

| 指标 | 数值 |
|---|---|
| 语料文档数 | **98** 份 |
| 切片数 | **2341** |
| Embedding | `BAAI/bge-small-zh-v1.5`（512 维，L2 归一化） |
| 向量库 | FAISS `IndexFlatIP` |
| 建索引耗时 | **约 80 s**（CPU 无独显） |
| `pytest` | **11 passed** |
| `ruff check .` | **All checks passed!** |

### Recall@5（文档级：chunk 粗排 top-50 → 按文档聚合 → 取 top-5）

| 检索方式 | B 层：主题型 query | P95 延迟 |
|---|---|---|
| 纯向量（bge-small-zh） | **0.4500** | 15.09 ms |
| BM25（字面） | **0.1500** | 9.00 ms |
| 混合（RRF 融合） | **0.3000** | 0.08 ms |

> ⚠️ **混合检索未优于纯向量**，这是实测结果，不做美化。归因与改进方向见 `reports/recall.md`。

---

## 五、诚实声明（面试前必读）

1. **未接大模型生成。** 本项目走「检索 + 返回片段」路径（抽取式），**没有接 LLM 做答案生成**。
   → **简历上只能写「检索 + 片段召回」，不许写「生成式问答」或「RAG 问答系统」。**
2. **`react.py` 的 `Thought` 是占位实现。** 因无 LLM API，`thought` 由 `RuleBasedLLM` 用**关键词规则**产出，
   不是语言模型推理。**ReAct 的循环结构、工具调用、Observation 回灌、步数截断均为真实运行。**
   换成真 API（`OpenAICompatLLM`）即为完整 Agent。
3. **评估集是脚本构造的，不是人工标注。**
   - A 层「内容派生」：query 取自文档原文连续子串 → 三个方法必然满分，**无区分度**，仅用于验证链路可用
   - B 层「主题型 query」：用文件名主题词构造，有区分度，但仍**不是**真实用户 query
   → 改进方向：人工标注 20–50 条真实 query 后再评
4. **代码来源**：`load_docs.py` 的扫描结构、`chunk.py` 的 `recursive_chunk` 算法主体由学员实现；
   `embed_index.py` / `retrieve.py` / `react.py` / `api.py` / `scripts/*` / `tests/*` 由教练提供（学员逐行读懂并调通）。
5. **数据未入库**：`data/` 已被 `.gitignore` 排除（索引与切片需本地重建）。

---

## 六、已排掉的坑（供复用）

| 坑 | 现象 | 解法 |
|---|---|---|
| HF 镜像抖动卡启动 | `WinError 10060` 重试 5 次 | `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`（模型已缓存则强制离线） |
| FAISS 写中文绝对路径 | `could not open ... for writing` | `faiss.serialize_index()` + Python `write_bytes` |
| BGE 召回偏低 | — | 查询侧加前缀 `为这个句子生成表示以用于检索相关文章：`，文档侧不加 |
| 中英混合语料 BM25 失效 | 英文词被拆成字母 | 分词用 `[A-Za-z0-9_]+` 整词 + CJK 单字，**语料与查询必须同一 tokenizer** |
| `s.split("")` | `ValueError: empty separator` | 末级分隔符改为定长硬切 + overlap |
| uv venv 无 pip | `No module named pip` | 用 `uv pip install` |
| PowerShell 的 `curl` | 参数报错 | 写 `curl.exe` |

---

## 七、改进方向（TODO）

1. 接 LLM API，把「检索」升级为真正的「检索增强生成」
2. 加 reranker（`bge-reranker`）对 top-50 重排
3. RRF 两路加权（向量权重更高），或换 weighted fusion
4. 中文分词改用 jieba，提升 BM25 质量
5. 人工标注 20–50 条真实 query，重做评估
6. `Dockerfile` + `docker-compose.yml`（本次未做）
