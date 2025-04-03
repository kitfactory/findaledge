"""
Finderledge - A document search and retrieval library
Finderledge - 文書検索・取得ライブラリ

This library provides functionality for searching and retrieving documents
using various embedding models and search algorithms.
このライブラリは、様々な埋め込みモデルと検索アルゴリズムを使用して
文書を検索・取得する機能を提供します。
"""

from . import env # envモジュールをインポート

# from .document import Document  # <-- 削除
# from .embedding import OpenAIEmbeddingModel # <-- 削除
from .text_splitter import TextSplitter
from .document_loader import DocumentLoader
from .document_store.document_store import BaseDocumentStore
# from .embedding_store import EmbeddingStore # <-- 削除
from .finder import Finder # <-- コメントアウト解除
# from .bm25 import BM25 # <-- 削除
from .tokenizer import Tokenizer
from .document_splitter import DocumentSplitter, DocumentType
from .document_store.vector_document_store import VectorDocumentStore
from .embeddings_factory import EmbeddingModelFactory

# 新しく追加した BM25sStore もインポートする (必要であれば)
from .document_store.bm25s import BM25sStore

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    # "Document",  # <-- 削除
    "BaseDocumentStore",
    "VectorDocumentStore",
    # "EmbeddingStore", # <-- 削除
    # "OpenAIEmbeddingModel", # <-- 削除
    "EmbeddingModelFactory",
    "TextSplitter",
    "DocumentLoader",
    "Finder", # <-- コメントアウト解除
    # "BM25", # <-- 削除
    "Tokenizer",
    "DocumentSplitter",
    "DocumentType",
    "BM25sStore", # 新しく追加したクラスを __all__ に追加 (必要であれば)
] 