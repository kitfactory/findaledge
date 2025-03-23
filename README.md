# FinderLedge 🔍📚

A document context management library for OpenAI Agents SDK. Provides dynamic document context to agents for efficient information retrieval and navigation.

## ✨ Features

- **📄 Document Import & Auto-Indexing**: Import documents and automatically create searchable indexes
- **🔍 Hybrid Search**: Combines vector search (semantic similarity) and keyword search (BM25) for optimal results
- **📁 Directory-wide Database**: Create indexes from entire folders of documents
- **📑 Multiple Document Formats**: Support for text, PDF, Word, Markdown, and more
- **🧠 Embedding Similarity**: Semantic search using OpenAI or other embedding models
- **🔤 High-performance BM25**: Keyword-based search for precise term matching
- **💾 Persistent Indexing & Caching**: Save and reuse indexes for faster startup
- **🔌 Simple Search API**: Intuitive interface for document retrieval
- **🤖 OpenAI Agents SDK Integration**: Use as a tool or context provider for agents
- **🔧 SDK-Independent Usage**: Can be used standalone or with other frameworks

## 🚀 Installation

```bash
pip install finderledge
```

## 🏁 Quick Start

```python
from finderledge import FinderLedge

# Create an instance
ledge = FinderLedge(db_name="my_documents")

# Add documents
ledge.add_document("path/to/document.pdf")
ledge.add_directory("path/to/document_folder")

# Search for related content
results = ledge.find_related("query text", mode="hybrid")

# Get context for OpenAI Agents SDK
context = ledge.get_context("query text")

# Release resources when done
ledge.close()
```

## 🤖 OpenAI Agents SDK Integration

```python
from openai import OpenAI
from finderledge import FinderLedge

# Create a FinderLedge instance
ledge = FinderLedge(db_name="knowledge_base")
ledge.add_directory("path/to/documents")

# Register as a tool
@function_tool
def search_docs(query: str) -> str:
    results = ledge.find_related(query)
    return "\n\n".join([r.page_content for r in results])

# Create an agent with the tool
client = OpenAI()
assistant = client.beta.assistants.create(
    name="Document Assistant",
    instructions="You help users find information in documents.",
    model="gpt-4-turbo",
    tools=[search_docs.openai_schema],
)
```

## 💻 Supported Environments

- Python 3.9+
- Windows/macOS/Linux compatible

## 📜 License

MIT

## 👩‍💻 Development

Setup development environment:

```bash
git clone https://github.com/yourusername/finderledge.git
cd finderledge
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```
