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
import json

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

        # If no metadata hint, try content detection
        content = document.page_content.strip()
        if content.startswith("<!DOCTYPE html") or content.startswith("<html>"):
            return DocumentType.HTML
        if content.startswith("{") or content.startswith("["):
            try:
                json.loads(content)
                return DocumentType.JSON
            except json.JSONDecodeError:
                pass

        # Default to TEXT type with recursive character splitting
        return DocumentType.TEXT

    def _get_splitter_for_type(self, doc_type: DocumentType) -> TextSplitter:
        """
        Get the appropriate text splitter for a given document type
        指定されたドキュメントタイプに適したテキストスプリッターを取得

        Args:
            doc_type (DocumentType): The document type
                ドキュメントタイプ

        Returns:
            TextSplitter: The appropriate text splitter for the given document type
                指定されたドキュメントタイプに適したテキストスプリッター
        """
        return self.splitters[doc_type]

    def split_document(self, document: Document) -> List[Document]:
        """
        Split a single document based on its detected or specified type.
        検出された、または指定されたタイプに基づいて単一のドキュメントを分割します。
        """
        doc_type = self._get_document_type(document)
        splitter = self._get_splitter_for_type(doc_type)

        # Use page_content instead of content
        split_texts = splitter.split_text(document.page_content)

        # Create new Document objects for each split
        split_docs = []
        # Use original metadata id as parent_id if available, else generate one.
        parent_id = document.metadata.get("id", str(uuid.uuid4()))

        for i, text in enumerate(split_texts):
            # Start with a copy of the original metadata
            metadata = document.metadata.copy()
            # Update with split-specific info
            metadata["parent_id"] = parent_id
            metadata["split_index"] = i
            metadata["is_split"] = True
            # Assign a new unique ID to the split chunk itself
            metadata["id"] = f"{parent_id}_split_{i}"
            # Ensure the detected type is also in metadata if needed later
            metadata["doc_type_detected"] = doc_type.name
            split_docs.append(Document(page_content=text, metadata=metadata))

        return split_docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split multiple documents.
        複数のドキュメントを分割します。
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