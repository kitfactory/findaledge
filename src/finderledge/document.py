"""
Document data model for storing and managing document information
文書情報を保存・管理するためのデータモデル

This module defines the Document class which represents a single document
in the system, including its content, metadata, and embeddings.
このモジュールは、コンテンツ、メタデータ、埋め込みを含む
システム内の単一の文書を表すDocumentクラスを定義します。
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
import time
import numpy as np

class Document(BaseModel):
    """
    Document data model
    文書データモデル
    """

    id: str = Field(..., description="Document ID / 文書ID")
    title: str = Field("", description="Document title / 文書タイトル")
    content: str = Field(..., description="Document content / 文書内容")
    metadata: Dict = Field(default_factory=dict, description="Document metadata / 文書メタデータ")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp / 作成タイムスタンプ")
    updated_at: datetime = Field(default_factory=datetime.now, description="Update timestamp / 更新タイムスタンプ")
    chunks: Dict[str, str] = Field(default_factory=dict, description="Document chunks / 文書チャンク")
    chunk_embeddings: Dict[str, List[float]] = Field(default_factory=dict, description="Chunk embeddings / チャンクの埋め込みベクトル")

    @field_validator("chunks", "chunk_embeddings", mode="before")
    def validate_dict(cls, v):
        """
        Validate dictionary fields
        辞書フィールドを検証

        Args:
            v: Value to validate / 検証する値

        Returns:
            Dict: Validated dictionary / 検証済みの辞書
        """
        if not isinstance(v, dict):
            return {}
        return v

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
        Add multiple chunks to the document
        文書に複数のチャンクを追加

        Args:
            chunks (List[str]): List of chunk texts / チャンクテキストのリスト
        """
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{self.id}_{i}"
            self.chunks[chunk_id] = chunk_text
        self.updated_at = datetime.now()

    def add_chunk(self, chunk_id: str, chunk_text: str) -> None:
        """
        Add a chunk to the document
        文書にチャンクを追加

        Args:
            chunk_id (str): Chunk ID / チャンクID
            chunk_text (str): Chunk text / チャンクテキスト
        """
        self.chunks[chunk_id] = chunk_text
        self.updated_at = datetime.now()

    def add_chunk_embeddings(self, embeddings: Union[List[List[float]], Dict[str, List[float]]]) -> None:
        """
        Add chunk embeddings to the document
        文書にチャンクの埋め込みを追加

        Args:
            embeddings (Union[List[List[float]], Dict[str, List[float]]]): List or dictionary of chunk embeddings / チャンクの埋め込みベクトルのリストまたは辞書

        Raises:
            ValueError: If number of embeddings doesn't match number of chunks
        """
        # 辞書型の場合はチャンクIDの順序でリストに変換
        if isinstance(embeddings, dict):
            # チャンクIDが一致するか確認
            if not all(chunk_id in self.chunks for chunk_id in embeddings):
                raise ValueError("Embedding chunk IDs do not match document chunk IDs")

            embeddings_list = [embeddings[chunk_id] for chunk_id in self.chunks]
        else:
            embeddings_list = embeddings

        # エンベディング数とチャンク数が一致するか確認
        if len(embeddings_list) != len(self.chunks):
            raise ValueError(f"Number of embeddings ({len(embeddings_list)}) must match number of chunks ({len(self.chunks)})")

        # エンベディングを正規化して保存
        for chunk_id, embedding in zip(self.chunks.keys(), embeddings_list):
            # 文字列形式の場合は数値に変換
            if isinstance(embedding, str):
                try:
                    embedding = [float(x) for x in embedding.split(",")]
                except ValueError:
                    raise ValueError(f"Invalid embedding format: {embedding}")

            # ベクトルを正規化
            norm = sum(x * x for x in embedding) ** 0.5
            if norm > 0:
                embedding = [x / norm for x in embedding]

            self.chunk_embeddings[chunk_id] = embedding

        self.updated_at = datetime.now()

    def clear_chunks(self) -> None:
        """
        Clear all chunks and their embeddings
        全てのチャンクとその埋め込みをクリア
        """
        self.chunks.clear()
        self.chunk_embeddings.clear()
        self.updated_at = datetime.now()

    def get_chunk_embedding(self, chunk_id: str) -> Optional[List[float]]:
        """
        Get embedding for a specific chunk
        特定のチャンクの埋め込みを取得

        Args:
            chunk_id (str): Chunk ID / チャンクID

        Returns:
            Optional[List[float]]: Chunk embedding / チャンクの埋め込みベクトル
        """
        return self.chunk_embeddings.get(chunk_id)

    def get_all_chunk_embeddings(self) -> List[List[float]]:
        """
        Get all chunk embeddings
        全てのチャンクの埋め込みを取得

        Returns:
            List[List[float]]: List of chunk embeddings / チャンクの埋め込みベクトルのリスト
        """
        return [self.chunk_embeddings[chunk_id] for chunk_id in self.chunks]

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

        # チャンクと埋め込みが空のリストの場合は空の辞書に変換
        if "chunks" in data and isinstance(data["chunks"], list):
            data["chunks"] = {}
        if "chunk_embeddings" in data and isinstance(data["chunk_embeddings"], list):
            data["chunk_embeddings"] = {}

        return cls(**data) 