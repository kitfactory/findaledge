# FinderLedge 🧭✨

**Effortlessly build powerful ensemble retrieval systems for your RAG applications!**

FinderLedge simplifies the process of setting up and managing multiple document retrieval methods (like vector search and keyword search) and combining their results for more relevant and robust RAG context generation.

[![PyPI version](https://badge.fury.io/py/finderledge.svg)](https://badge.fury.io/py/finderledge)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🤔 Why FinderLedge?

Retrieval-Augmented Generation (RAG) often benefits from combining different search strategies:

*   **Vector Search:** Great for semantic similarity (finding documents with similar *meaning*).
*   **Keyword Search (like BM25):** Excels at finding documents containing specific terms or phrases.

Setting up multiple retrievers, managing their indices, and combining (reciprocal rank fusion - RRF) their results can be tedious and complex. 😩

**FinderLedge makes it super simple!** ✨ It automatically configures vector stores (like Chroma or FAISS) and keyword stores (BM25) for you. Just add your documents, and perform powerful hybrid searches with a single command. Focus on your application, not the retrieval plumbing! 🚀

## 🚀 Features

*   **Easy Initialization:** Get started with sensible defaults (Chroma + BM25) in one line.
*   **Flexible Configuration:** Easily swap vector stores (Chroma, FAISS), keyword stores (BM25), embedding models (OpenAI, Ollama, etc.), and persistence paths.
*   **Simple Document Loading:** Add documents from files or entire directories with automatic file type detection and parsing (powered by LangChain document loaders).
*   **Built-in Splitting:** Automatically splits documents into appropriate chunks based on content type.
*   **Hybrid Search (RRF):** Performs vector and keyword searches simultaneously and intelligently combines results using Reciprocal Rank Fusion (RRF) by default.
*   **Pure Search Modes:** Option to use only vector search or only keyword search.
*   **LangChain Integration:** Built on top of popular LangChain components.

## 🛠️ Installation

```bash
# Using pip
pip install finderledge

# Or using uv
uv pip install finderledge

# Install optional dependencies for specific features (e.g., OpenAI embeddings)
pip install finderledge[openai]
# or
uv pip install finderledge[openai]
```

## 💻 Basic Usage

```python
from finderledge import FinderLedge
import os

# --- Configuration (Optional: Set environment variables) ---
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
# os.environ["FINDERLEDGE_PERSIST_DIRECTORY"] = "./my_data_store"
# os.environ["FINDERLEDGE_VECTOR_STORE"] = "faiss" # Example: Use FAISS instead of Chroma
# os.environ["FINDERLEDGE_EMBEDDING_PROVIDER"] = "openai"

# --- Initialization ---
# Uses defaults or environment variables if set
# Default: Chroma vector store, BM25 keyword store, SentenceTransformer embeddings
print("Initializing FinderLedge...")
ledge = FinderLedge()
print("FinderLedge Initialized!")

# --- Add Documents --- 
# Create dummy files for the example
docs_dir = "example_docs"
os.makedirs(docs_dir, exist_ok=True)
with open(os.path.join(docs_dir, "doc1.txt"), "w") as f:
    f.write("This is the content of the first document about apples.")
with open(os.path.join(docs_dir, "doc2.md"), "w") as f:
    f.write("# Oranges\nOranges are a citrus fruit.")

print(f"Adding documents from {docs_dir}...")
# Add a single file
# ledge.add_document(os.path.join(docs_dir, "doc1.txt")) 
# Add all supported files in a directory (recursive by default)
ledge.add_document(docs_dir)
print("Documents added!")

# --- Search --- 
query = "Tell me about fruit"
print(f"\nSearching for: '{query}'")
# Performs hybrid search (vector + keyword + RRF) by default
results = ledge.search(query, top_k=3)

print("\nSearch Results:")
if results:
    for i, doc in enumerate(results):
        print(f"--- Result {i+1} ---")
        print(f"  Score: {doc.metadata.get('relevance_score', 'N/A'):.4f}") # RRF Score
        print(f"  Source: {doc.metadata.get('source', 'N/A')}")
        # Displaying parent doc content if split, otherwise the content itself
        parent_content = doc.metadata.get("parent_content", doc.page_content) 
        print(f"  Content: {parent_content[:150]}...") # Show limited content
else:
    print("No results found.")

# --- Clean up dummy files (optional) ---
# import shutil
# shutil.rmtree(docs_dir)

```

## ⚙️ Advanced Configuration

You can configure FinderLedge extensively via environment variables or directly during initialization:

```python
# Example: Initialize with FAISS vector store and OpenAI embeddings
ledge_advanced = FinderLedge(
    vector_store_provider="faiss",        # Use FAISS
    keyword_store_provider="bm25",        # Keep BM25
    embedding_provider="openai",          # Use OpenAI for embeddings
    embedding_model_name="text-embedding-3-small", # Specify model
    persist_directory="./my_faiss_store" # Custom persistence path
    # chunk_size=500,                    # Optional: Custom chunk size
    # chunk_overlap=50                    # Optional: Custom chunk overlap
)

# Search using only vector mode
results_vector = ledge_advanced.search(query, search_mode="vector", top_k=2)
```

See the `FinderLedge` class documentation for all available options.

## 🌍 Supported Environment

*   🐍 Python 3.10+

## 🙏 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📜 License

FinderLedge is licensed under the [MIT License](LICENSE).
