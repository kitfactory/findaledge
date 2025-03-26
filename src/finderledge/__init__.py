"""
Finderledge - A document search and retrieval library
Finderledge - 文書検索・取得ライブラリ

This library provides functionality for searching and retrieving documents
using various embedding models and search algorithms.
このライブラリは、様々な埋め込みモデルと検索アルゴリズムを使用して
文書を検索・取得する機能を提供します。
"""

from .document import Document
from .embedding import OpenAIEmbeddingModel
from .text_splitter import TextSplitter
from .document_loader import DocumentLoader
from .document_store import DocumentStore
from .embedding_store import EmbeddingStore
from .finder import Finder
from .bm25 import BM25
from .tokenizer import Tokenizer

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    "Document",
    "DocumentStore",
    "EmbeddingStore",
    "OpenAIEmbeddingModel",
    "TextSplitter",
    "DocumentLoader",
    "Finder",
    "BM25",
    "Tokenizer"
] 