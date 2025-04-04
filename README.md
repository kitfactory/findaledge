# 🚀 FindaLedge: Simple Ensemble Search for RAG 🔍

**FindaLedge** is a Python library designed to simplify the creation and management of **ensemble search systems** for Retrieval-Augmented Generation (RAG) applications. It elegantly combines vector search and keyword search, abstracting away the complexities of setting up multiple document stores (like Chroma, FAISS, BM25), handling document ingestion, managing indices, and merging search results (using RRF).

✨ **Build powerful RAG search backends with ease!** ✨

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/findaledge.svg)](https://badge.fury.io/py/findaledge)

--- 

[🇯🇵 日本語版 README はこちら (Click here for Japanese README)](README_ja.md)

--- 

## 🤔 Why FindaLedge?

RAG systems often struggle with a single search method. Vector search excels at semantic similarity but might miss specific keywords or acronyms. Keyword search finds exact matches but lacks understanding of intent.

**Ensemble search**, combining both, offers superior accuracy. However, building it involves juggling:

*   Multiple databases (Vector DBs like Chroma/FAISS, Keyword indices like BM25)
*   Document loading, parsing, and chunking pipelines
*   Embedding generation and management
*   Synchronizing additions/deletions across stores
*   Complex result merging logic (like Reciprocal Rank Fusion - RRF)

🤯 **It's a lot to handle!**

**FindaLedge** takes care of all this heavy lifting with a **simple, unified interface**. You focus on your LLM application, while FindaLedge manages the search backend.

## ✨ Key Features

*   **🎯 Easy Ensemble Search:** Get the power of hybrid search (vector + keyword) out-of-the-box.
*   **🔌 Flexible Components:** Easily configure vector stores (`Chroma`, `FAISS`), keyword stores (`BM25`), and embedding models (`OpenAI`, `Ollama`, `SentenceTransformers`) via environment variables or arguments.
*   **📚 Effortless Document Ingestion:** Add documents from files or entire directories (`.md`, `.txt`, `.pdf`, `.docx`, etc.) with a single command (`add_document`). Automatic parsing and chunking included!
*   **🔄 Automatic Indexing & Persistence:** Indices (vector and keyword) are automatically created, updated, and saved locally. Load them instantly on the next run!
*   **🥇 Smart Result Merging:** Uses Reciprocal Rank Fusion (RRF) by default to intelligently combine and rank results from different search methods.
*   **🔗 Seamless LangChain Integration:** Use FindaLedge directly as a LangChain `BaseRetriever` (`as_retriever()`) in your existing chains (like `RetrievalQA`).
*   **🧹 Simple Add/Remove:** Easily add new documents or remove existing ones (`remove_document`) with automatic index updates.

## ⚙️ Supported Environment

*   **Python:** 3.10 or higher
*   **OS:** Tested on Windows, macOS, Linux
*   **Dependencies:** See `pyproject.toml`. Key ones include `langchain`, `bm25s-j`, `numpy`, `chromadb` (optional), `faiss-cpu` (optional), `sentence-transformers` (optional), `openai` (optional), `ollama` (optional).

## 🚀 Getting Started

### 1. Installation

```bash
pip install findaledge

# Or, if using optional dependencies like FAISS:
# pip install findaledge[faiss] # Example, adjust based on pyproject.toml
```

Make sure you have necessary dependencies for your chosen vector store and embedding model (e.g., `pip install chromadb openai sentence-transformers`).

### 2. Configuration (Environment Variables)

Set environment variables (optional, defaults are provided):

```bash
export OPENAI_API_KEY="your-openai-key" # If using OpenAI embeddings
export FINDALEDGE_PERSIST_DIRECTORY="./my_findaledge_data" # Where to save index data
export FINDALEDGE_VECTOR_STORE_PROVIDER="chroma"       # "chroma" or "faiss"
export FINDALEDGE_EMBEDDING_PROVIDER="openai"          # "openai", "ollama", "huggingface"
export FINDALEDGE_EMBEDDING_MODEL_NAME="text-embedding-3-small" # Model name
# export OLLAMA_BASE_URL="http://localhost:11434" # If using Ollama
```

### 3. Basic Usage

```python
from findaledge import FindaLedge
from langchain_core.documents import Document

# Initialize FindaLedge (uses env vars or defaults if not passed)
# It automatically loads existing data from persist_directory or creates new indices.
ledge = FindaLedge(
    # You can override env vars here, e.g.:
    # persist_directory="./custom_data",
    # vector_store_provider="faiss"
)

# --- Add Documents ---

# Add from a file path (auto-parses based on extension)
print("Adding file...")
ledge.add_document("path/to/your/report.md")

# Add all compatible files from a directory (recursively)
print("Adding directory...")
ledge.add_document("path/to/your/docs_folder/")

# Add a LangChain Document object directly
print("Adding Document object...")
doc = Document(page_content="This is a sample document.", metadata={"source": "manual", "id": "manual-doc-1"})
ledge.add_document(doc)

print("Documents added!")

# --- Search Documents ---

query = "What is the main topic of the report?"
print(f"\nSearching for: '{query}'")

# Hybrid search (default)
results = ledge.search(query, top_k=3)

print("\nSearch Results:")
for i, doc in enumerate(results):
    print(f"{i+1}. Score: {doc.metadata.get('relevance_score', 0):.4f} | Source: {doc.metadata.get('source', 'N/A')}")
    # print(f"   Content: {doc.page_content[:150]}...") # Uncomment to show content

# --- Use as LangChain Retriever ---

print("\nUsing as LangChain Retriever...")
retriever = ledge.as_retriever(search_mode="hybrid", top_k=5)

# Now use this retriever in any LangChain component, e.g., RetrievalQA
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI # Example LLM

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=retriever,
    return_source_documents=True
)

response = qa_chain.invoke({"query": query})
print("\nLangChain QA Answer:", response["result"])

# --- Remove a Document (Example) ---
# You need the parent document ID. Often the source path or an ID you provided.
# doc_id_to_remove = "manual-doc-1" # Get this from add_document result or metadata
# print(f"\nRemoving document: {doc_id_to_remove}")
# ledge.remove_document(doc_id_to_remove)
# print("Document removed.")

```

## 📖 Documentation

*   [Usage Guide (使い方ガイド)](docs/usage_guide.md)
*   [Architecture (アーキテクチャ設計書)](docs/architecture.md)
*   [Requirements (要件定義書)](docs/requirements.md)
*   [Function Specification (機能仕様書)](docs/function_spec.md)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature`).
3.  Make your changes.
4.  Run tests (`pytest`).
5.  Commit your changes (`git commit -am 'Add some feature'`).
6.  Push to the branch (`git push origin feature/your-feature`).
7.  Create a new Pull Request.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
