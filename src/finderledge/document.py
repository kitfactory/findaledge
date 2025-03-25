"""
Document data model for storing and managing document information
文書情報を保存・管理するためのデータモデル

This module defines the Document class which represents a single document
in the system, including its content, metadata, and embeddings.
このモジュールは、コンテンツ、メタデータ、埋め込みを含む
システム内の単一の文書を表すDocumentクラスを定義します。
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
import time

class Document(BaseModel):
    """
    Document data model
    文書データモデル
    """

    id: str = Field(description="Document ID / 文書ID")
    title: Optional[str] = Field(default="", description="Document title / 文書タイトル")
    content: str = Field(description="Document content / 文書内容")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata / 文書のメタデータ")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp / 作成日時")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp / 最終更新日時")
    chunks: List[str] = Field(default_factory=list, description="List of text chunks / テキストチャンクのリスト")
    chunk_embeddings: List[List[float]] = Field(default_factory=list, description="List of chunk embeddings / チャンクの埋め込みベクトルのリスト")

    def update(self, title: Optional[str] = None, content: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Update document attributes
        文書の属性を更新する

        Args:
            title (Optional[str]): New title / 新しいタイトル
            content (Optional[str]): New content / 新しい内容
            metadata (Optional[Dict[str, Any]]): New metadata / 新しいメタデータ
        """
        if title is not None:
            self.title = title
        if content is not None:
            self.content = content
        if metadata is not None:
            self.metadata = metadata
        time.sleep(0.001)  # Ensure updated_at is different from created_at
        self.updated_at = datetime.now()

    def add_chunks(self, chunks: List[str]) -> None:
        """
        Add text chunks to the document
        文書にテキストチャンクを追加

        Args:
            chunks (List[str]): List of text chunks / テキストチャンクのリスト
        """
        self.chunks = chunks

    def add_chunk_embeddings(self, embeddings: List[List[float]]) -> None:
        """
        Add chunk embeddings to the document
        文書にチャンクの埋め込みを追加

        Args:
            embeddings (List[List[float]]): List of chunk embeddings / チャンクの埋め込みベクトルのリスト
        """
        self.chunk_embeddings = embeddings

    def clear_chunks(self) -> None:
        """
        Clear all text chunks and their embeddings
        全てのテキストチャンクとその埋め込みをクリア
        """
        self.chunks = []
        self.chunk_embeddings = []

    def to_dict(self) -> Dict:
        """
        Convert document to dictionary
        文書を辞書に変換

        Returns:
            Dict: Dictionary representation of the document / 文書の辞書表現
        """
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "chunks": self.chunks,
            "chunk_embeddings": self.chunk_embeddings
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Document":
        """
        Create document from dictionary
        辞書から文書を作成

        Args:
            data (Dict): Dictionary containing document data / 文書データを含む辞書

        Returns:
            Document: New document instance / 新しい文書インスタンス
        """
        if "created_at" in data:
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data) 