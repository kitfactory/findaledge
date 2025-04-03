"""
Base Document Store Interface Module
ベースドキュメントストアインターフェースモジュール

This module defines the abstract base class for document stores.
このモジュールはドキュメントストアの抽象基底クラスを定義します。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Sequence
from datetime import datetime
from langchain.schema import BaseRetriever, Document

class BaseDocumentStore(ABC):
    """
    Abstract base class for document stores
    ドキュメントストアの抽象基底クラス

    This class defines the interface that all document store implementations must follow.
    このクラスは全てのドキュメントストア実装が従うべきインターフェースを定義します。

    Note:
        Document IDs are stored in the metadata dictionary of each Document with the key 'id'
        ドキュメントIDは各Documentのmetadataディクショナリのidキーとして保存されます
    """

    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize document store with implementation specific parameters
        実装固有のパラメータでドキュメントストアを初期化

        Args:
            **kwargs (Any): Implementation specific parameters
                実装固有のパラメータ
        """
        pass

    @abstractmethod
    def _ensure_storage(self, **kwargs: Any) -> None:
        """
        Ensure storage is ready for use (e.g., create directories, establish connections)
        ストレージが使用可能な状態であることを確認（ディレクトリの作成、接続の確立など）

        Args:
            **kwargs (Any): Implementation specific parameters
                実装固有のパラメータ
        """
        pass

    @abstractmethod
    def add_documents(self, documents: Sequence[Document], ids: Optional[Sequence[str]] = None) -> List[str]:
        """
        Add multiple documents to store
        複数の文書をストアに追加

        Args:
            documents (Sequence[Document]): Documents to add / 追加する文書のリスト
            ids (Optional[Sequence[str]], optional): Custom IDs for the documents. If not provided, 
                IDs will be generated / 文書のカスタムID。指定がない場合はIDが生成されます

        Returns:
            List[str]: List of document IDs for the added documents
                追加された文書のIDのリスト

        Note:
            The document IDs will be stored in the metadata dictionary of each Document
            with the key 'id'
            ドキュメントIDは各Documentのmetadataディクショナリのidキーとして保存されます
        """
        pass

    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Get document by ID
        IDで文書を取得

        Args:
            doc_id (str): Document ID / 文書ID

        Returns:
            Optional[Document]: Document if found, None otherwise. The document's ID
                will be in its metadata dictionary with key 'id'
                見つかった場合は文書、それ以外はNone。文書のIDはmetadataディクショナリの
                idキーに格納されています
        """
        pass

    @abstractmethod
    def update_document(self, doc_id: str, document: Document) -> None:
        """
        Update document in store
        ストア内の文書を更新

        Args:
            doc_id (str): ID of the document to update / 更新する文書のID
            document (Document): New document content and metadata / 新しい文書の内容とメタデータ
        """
        pass

    @abstractmethod
    def delete_document(self, doc_id: str) -> None:
        """
        Delete document from store
        ストアから文書を削除

        Args:
            doc_id (str): Document ID / 文書ID
        """
        pass

    @abstractmethod
    def list_documents(self) -> List[str]:
        """
        List all document IDs in store
        ストア内の全文書IDをリスト表示

        Returns:
            List[str]: List of document IDs / 文書IDのリスト
        """
        pass

    @abstractmethod
    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        """
        Get a LangChain retriever interface for this document store
        このドキュメントストアのLangChain retrieverインターフェースを取得

        Args:
            **kwargs (Any): Additional parameters for the retriever
                retrieverの追加パラメータ

        Returns:
            BaseRetriever: A LangChain retriever interface
                LangChain retrieverインターフェース
        """
        pass
