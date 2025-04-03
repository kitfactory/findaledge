"""
Document splitter that selects and uses appropriate LangChain text splitters
適切なLangChainテキストスプリッターを選択して使用するドキュメントスプリッター

This module provides a document splitter that automatically selects
the most appropriate LangChain text splitter based on document metadata.
このモジュールは、ドキュメントのメタデータに基づいて最適な
LangChainテキストスプリッターを自動選択するドキュメントスプリッターを提供します。
"""

from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from enum import Enum, auto
import uuid

from langchain.text_splitter import (
    TextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
)
from langchain_text_splitters.html import HTMLSemanticPreservingSplitter
from langchain.schema import Document

class DocumentType(Enum):
    """
    Supported document types
    サポートされているドキュメントタイプ
    """
    TEXT = auto()
    MARKDOWN = auto()
    PYTHON = auto()
    HTML = auto()
    JSON = auto()

class DocumentSplitter:
    """
    Document splitter that selects appropriate text splitter based on document type
    ドキュメントタイプに基づいて適切なテキストスプリッターを選択するドキュメントスプリッター
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        **kwargs: Any
    ):
        """
        Initialize document splitter
        ドキュメントスプリッターを初期化

        Args:
            chunk_size (int): Size of text chunks
                テキストチャンクのサイズ
            chunk_overlap (int): Overlap between chunks
                チャンク間の重複
            **kwargs: Additional arguments for text splitters
                テキストスプリッターの追加引数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.kwargs = kwargs

        # Initialize splitter registry
        self.splitters = self._init_splitters()

    def _init_splitters(self) -> Dict[DocumentType, TextSplitter]:
        """
        Initialize text splitters for each document type
        各ドキュメントタイプのテキストスプリッターを初期化

        Returns:
            Dict[DocumentType, TextSplitter]: Dictionary mapping document types to their splitters
                ドキュメントタイプとそのスプリッターのマッピング辞書
        """
        return {
            DocumentType.TEXT: RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            ),
            DocumentType.MARKDOWN: MarkdownTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            ),
            DocumentType.PYTHON: PythonCodeTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            ),
            DocumentType.HTML: RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["</h1>", "</h2>", "</h3>", "</h4>", "</h5>", "</h6>", "</p>", "</div>", "\n\n", "\n", " ", ""]
            ),
            DocumentType.JSON: RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["}}", "}", "],", "]", ",", " ", ""]
            )
        }

    def _get_document_type(self, document: Document) -> DocumentType:
        """
        Determine document type from metadata or content
        メタデータまたはコンテンツからドキュメントタイプを判定

        Args:
            document (Document): Input document
                入力ドキュメント

        Returns:
            DocumentType: Determined document type
                判定されたドキュメントタイプ
        """
        # Check metadata for explicit type
        metadata = document.metadata or {}
        doc_type = metadata.get("type")
        if doc_type:
            try:
                return DocumentType[doc_type.upper()]
            except KeyError:
                pass

        # Check file extension if path is provided
        file_path = metadata.get("source")
        if file_path:
            ext = Path(file_path).suffix.lower()
            ext_map = {
                ".md": DocumentType.MARKDOWN,
                ".py": DocumentType.PYTHON,
                ".html": DocumentType.HTML,
                ".htm": DocumentType.HTML,
                ".json": DocumentType.JSON
            }
            if ext in ext_map:
                return ext_map[ext]

        # Try to detect content type
        content = document.content.strip()
        if content.startswith("<!DOCTYPE html") or content.startswith("<html"):
            return DocumentType.HTML
        if content.startswith("{") and content.endswith("}"):
            try:
                import json
                json.loads(content)
                return DocumentType.JSON
            except json.JSONDecodeError:
                pass

        # Default to TEXT type with recursive character splitting
        return DocumentType.TEXT

    def split_document(self, document: Document) -> List[Document]:
        """
        Split a document using the appropriate text splitter
        適切なテキストスプリッターを使用してドキュメントを分割

        Args:
            document (Document): The document to split
                           分割するドキュメント

        Returns:
            List[Document]: List of split documents
                       分割されたドキュメントのリスト
        """
        doc_type = self._get_document_type(document)
        splitter = self.splitters[doc_type]

        # Prepare base metadata
        base_metadata = document.metadata.copy()
        base_metadata["type"] = doc_type.name
        base_metadata["parent_id"] = document.metadata.get("id", "")
        base_metadata["is_split"] = True

        if isinstance(splitter, HTMLSemanticPreservingSplitter):
            split_docs = splitter.transform_documents([document])
            for doc in split_docs:
                doc.metadata.update(base_metadata)
            return split_docs
        else:
            split_texts = splitter.split_text(document.content)
            split_docs = []
            for text in split_texts:
                metadata = base_metadata.copy()
                split_docs.append(Document(id=str(uuid.uuid4()), content=text, metadata=metadata))
            return split_docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split multiple documents using appropriate text splitters
        適切なテキストスプリッターを使用して複数のドキュメントを分割

        Args:
            documents (List[Document]): Documents to split
                分割するドキュメント

        Returns:
            List[Document]: List of split documents
                分割されたドキュメントのリスト
        """
        split_docs = []
        for doc in documents:
            split_docs.extend(self.split_document(doc))
        return split_docs

# 使用例 / Usage examples:
"""
from langchain.schema import Document

# Create a document splitter
splitter = DocumentSplitter(chunk_size=1000, chunk_overlap=200)

# Split a markdown document
markdown_doc = Document(
    page_content="# Title\n\nSome markdown content",
    metadata={"source": "example.md"}
)
split_docs = splitter.split_document(markdown_doc)

# Split multiple documents of different types
documents = [
    Document(
        page_content="def example():\n    pass",
        metadata={"type": "python"}
    ),
    Document(
        page_content="<html><body>Content</body></html>",
        metadata={"source": "example.html"}
    )
]
split_docs = splitter.split_documents(documents)
""" 