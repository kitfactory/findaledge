"""
Finderledge - A document search library using embeddings and BM25
Finderledge - 埋め込みとBM25を使用した文書検索ライブラリ

This library provides functionality for searching documents using a combination of embeddings and BM25.
このライブラリは、埋め込みとBM25を組み合わせて文書を検索する機能を提供します。
"""

from .document import Document
from .document_store import DocumentStore
from .embedding_store import EmbeddingStore
from .embedding_model import EmbeddingModel
from .tokenizer import Tokenizer
from .bm25 import BM25
from .finder import Finder

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    "Document",
    "DocumentStore",
    "EmbeddingStore",
    "EmbeddingModel",
    "Tokenizer",
    "BM25",
    "Finder"
] 