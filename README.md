# Finderledge 🔍

A document search library using embeddings and BM25 for efficient and accurate document retrieval.

埋め込みとBM25を使用した、効率的で正確な文書検索ライブラリ。

## Features ✨

- Document management with chunking support
- Hybrid search using embeddings and BM25
- Efficient storage and retrieval of document embeddings
- Configurable tokenization and text processing
- Easy-to-use API for document search

- チャンク分割をサポートした文書管理
- 埋め込みとBM25を使用したハイブリッド検索
- 効率的な文書埋め込みの保存と取得
- 設定可能なトークン化とテキスト処理
- 使いやすい文書検索API

## Installation 🚀

### Prerequisites

- Python 3.8 or higher
- uv (fast Python package installer)

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install Finderledge

```bash
# Clone the repository
git clone https://github.com/yourusername/finderledge.git
cd finderledge

# Create and activate virtual environment
uv venv
. .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package with development dependencies
uv pip install -e ".[dev]"
```

## Quick Start 🎯

```python
from finderledge import Document, DocumentStore, EmbeddingStore, EmbeddingModel, Tokenizer, BM25, Finder

# Initialize components
document_store = DocumentStore("documents")
embedding_store = EmbeddingStore("embeddings")
embedding_model = EmbeddingModel()
tokenizer = Tokenizer()
bm25 = BM25()

# Create finder
finder = Finder(
    document_store=document_store,
    embedding_store=embedding_store,
    embedding_model=embedding_model,
    tokenizer=tokenizer,
    bm25=bm25
)

# Add document
doc = Document(
    id="doc1",
    title="Sample Document",
    content="This is a sample document for testing.",
    metadata={"author": "John Doe"}
)
finder.add_document(doc)

# Search documents
results = finder.search("sample document", top_k=5)
for doc, score in results:
    print(f"Document: {doc.title}, Score: {score}")
```

## Documentation 📚

For detailed documentation, please visit our [documentation page](https://finderledge.readthedocs.io/).

詳細なドキュメントについては、[ドキュメントページ](https://finderledge.readthedocs.io/)をご覧ください。

## Development 🛠️

### Running Tests

```bash
pytest
```

### Code Style

```bash
# Format code
black .
isort .

# Type checking
mypy .

# Linting
flake8
```

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

貢献を歓迎します！プルリクエストをお気軽に送信してください。

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

このプロジェクトはMITライセンスの下でライセンスされています - 詳細については[LICENSE](LICENSE)ファイルをご覧ください。

## Support 💬

If you have any questions or need help, please open an issue or contact us at support@finderledge.com.

ご質問やお困りの点がございましたら、issueを作成するか、support@finderledge.comまでご連絡ください。
