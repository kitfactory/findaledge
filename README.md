# 🚀 FindaLedge: Simple Ensemble Search for RAG 🔍

**FindaLedge** is a Python library for building robust, hybrid search backends for Retrieval-Augmented Generation (RAG) and LLM applications. It unifies vector and keyword search, manages document ingestion, and provides a simple, powerful API.

✨ **Build powerful RAG search backends with ease!** ✨

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/findaledge.svg)](https://badge.fury.io/py/findaledge)

---

[🇯🇵 日本語版 README はこちら (Click here for Japanese README)](README_ja.md)

---

## 🤔 Why FindaLedge?

- Vector search (semantic) and keyword search (BM25) each have strengths and weaknesses.
- **FindaLedge** combines both (ensemble search) for best accuracy, with zero setup hassle.
- Handles all the plumbing: document loading, chunking, embedding, index sync, result fusion (RRF), and more!

## ✨ Features

| Feature | Description |
|---|---|
| 🎯 Hybrid Search | Combines vector & keyword search (BM25) with RRF fusion |
| 🔌 Flexible | Supports Chroma, FAISS, BM25s, OpenAI, Ollama, HuggingFace, etc. |
| 📚 Easy Ingestion | Add files, directories, or LangChain Documents instantly |
| 🔄 Auto Indexing | Indices are auto-created, updated, and persisted |
| 🧹 Simple API | Add, search, remove documents with one-liners |
| 🧩 LangChain Ready | Use as a Retriever in LangChain chains |
| 🧪 Full Test Suite | 100+ tests, pytest/uv compatible |

## ⚙️ Supported Environment

| Item | Supported |
|---|---|
| Python | 3.11+ (Windows/Powershell推奨) |
| OS | Windows, macOS, Linux |
| Vector DB | Chroma, FAISS (optional) |
| Embeddings | OpenAI, Ollama, HuggingFace, etc. |
| Agents SDK | [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) |
| Test | pytest, pytest-cov, uv |

## 🛠️ Quick Start

### 1. Install (with uv & venv recommended)

```bash
# Create and activate venv
python -m venv .venv
.venv\Scripts\Activate.ps1  # (Windows Powershell)

# Install uv (if not yet)
pip install uv

# Install dependencies
uv pip install -r requirements.txt
# or: uv pip install .
```

### 2. Set Environment Variables (optional)

```bash
$env:OPENAI_API_KEY="sk-..."  # For OpenAI
$env:FINDALEDGE_EMBEDDING_MODEL_NAME="text-embedding-3-small"
$env:FINDALEDGE_PERSIST_DIR="./my_data"
```

### 3. Basic Usage

```python
from findaledge import FindaLedge

ledge = FindaLedge()
ledge.add_document("docs/manual.txt")
results = ledge.search("What is the main topic?")
for r in results:
    print(r.document.page_content, r.score)
```

### 4. Run Tests

```bash
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
pytest
```

## 🏗️ Architecture (Layered)

| Layer | Class | Responsibility |
|---|---|---|
| Controller | FindaLedge | Unified API, orchestrates all below |
| UseCase | Finder | Hybrid search, RRF fusion |
| Gateway | ChromaDocumentStore, BM25sStore | Vector/BM25 storage |
| Function | DocumentLoader, DocumentSplitter, EmbeddingModelFactory | Loading, splitting, embedding |
| Data | LangchainDocument, SearchResult | Data objects |
| Utility | Tokenizer, config/env | Tokenize, config |

```plantuml
@startuml
FindaLedge --> Finder
FindaLedge --> DocumentLoader
FindaLedge --> DocumentSplitter
FindaLedge --> EmbeddingModelFactory
FindaLedge --> ChromaDocumentStore
FindaLedge --> BM25sStore
Finder --> SearchResult
@enduml
```

## 🧑‍💻 Main API (Class Table)

| Class | Role | Key Methods |
|---|---|---|
| FindaLedge | Facade/Controller | add_document, search, remove_document, get_context |
| Finder | UseCase (Hybrid) | search (RRF), find |
| ChromaDocumentStore | Gateway | add_documents, as_retriever |
| BM25sStore | Gateway | add_documents, as_retriever |
| EmbeddingModelFactory | Factory | create_embeddings |
| DocumentLoader | Loader | load_file, load_from_directory |
| DocumentSplitter | Splitter | split_documents |

## 📖 Documentation

- [Usage Guide](docs/usage_guide.md)
- [Architecture](docs/architecture.md)
- [Requirements](docs/requirements.md)
- [Function Spec](docs/function_spec.md)

## 🧪 Testing

- All tests pass (pytest/uv, Windows Powershell)
- Run: `pytest`
- Coverage: pytest-cov enabled

## 🤝 Contributing

Contributions welcome! Fork, branch, PR, and let's build better RAG search together 🚀

## 📜 License

MIT License. See [LICENSE](LICENSE).
