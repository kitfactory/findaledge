"""
Finderledge - A document search and retrieval library
Finderledge - 文書検索・取得ライブラリ

This library provides functionality for searching and retrieving documents
using various embedding models and search algorithms.
このライブラリは、様々な埋め込みモデルと検索アルゴリズムを使用して
文書を検索・取得する機能を提供します。
"""

# from .document import Document  # <-- 削除
from .embedding import OpenAIEmbeddingModel
from .text_splitter import TextSplitter
from .document_loader import DocumentLoader
from .document_store.document_store import BaseDocumentStore
from .embedding_store import EmbeddingStore
from .finder import Finder
from .bm25 import BM25
from .tokenizer import Tokenizer
from .document_splitter import DocumentSplitter, DocumentType
from .document_store.vector_document_store import VectorDocumentStore

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    # "Document",  # <-- 削除
    "BaseDocumentStore",
    "VectorDocumentStore",
    "EmbeddingStore",
    "OpenAIEmbeddingModel",
    "TextSplitter",
    "DocumentLoader",
    "Finder",
    "BM25",
    "Tokenizer",
    "DocumentSplitter",
    "DocumentType"
] 