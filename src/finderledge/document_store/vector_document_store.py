"""
Vector Document Store Module
ベクトルドキュメントストアモジュール

This module provides a base implementation of a vector-based document store.
このモジュールはベクトルベースのドキュメントストアの基本実装を提供します。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Sequence
from langchain.schema import Document, BaseRetriever
from langchain.embeddings.base import Embeddings
from .document_store import BaseDocumentStore
from ..document_splitter import DocumentSplitter
import uuid

class VectorDocumentStore(BaseDocumentStore, ABC):
    """
    Abstract base class for vector document stores
    ベクトルドキュメントストアの抽象基底クラス

    This class provides common functionality for vector stores, including document splitting
    and metadata management.
    このクラスは、文書分割やメタデータ管理を含むベクトルストアの共通機能を提供します。
    """

    def __init__(
        self,
        embedding_function: Embeddings,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        **kwargs: Any
    ) -> None:
        """
        Initialize vector document store
        ベクトルドキュメントストアを初期化する

        Args:
            embedding_function (Embeddings): Function to generate embeddings
                埋め込みを生成する関数
            chunk_size (int, optional): Size of text chunks for splitting. Defaults to 1000.
                テキスト分割のチャンクサイズ。デフォルトは1000。
            chunk_overlap (int, optional): Overlap between chunks. Defaults to 200.
                チャンク間の重複。デフォルトは200。
            **kwargs (Any): Additional implementation-specific parameters.
                追加の実装固有パラメータ。
        """
        super().__init__(**kwargs)
        self.embedding_function = embedding_function
        self.document_splitter = DocumentSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def add_documents(
        self,
        documents: Sequence[Document],
        ids: Optional[Sequence[str]] = None
    ) -> List[str]:
        """
        Add documents to the store, handling splitting and metadata.
        ドキュメントをストアに追加し、分割とメタデータを処理します。

        Args:
            documents (Sequence[Document]): Documents to add
                追加するドキュメント
            ids (Optional[Sequence[str]], optional): Optional document IDs. If provided, must match the length of documents.
                オプションのドキュメントID。指定する場合、ドキュメントの数と一致する必要があります。

        Returns:
            List[str]: List of parent document IDs (or original IDs if not split).
                親ドキュメントID（または分割されなかった場合は元のID）のリスト。
        """
        if ids and len(ids) != len(documents):
            raise ValueError("Number of IDs must match number of documents")

        docs_to_add: List[Document] = []
        processed_parent_ids: List[str] = []

        for i, doc in enumerate(documents):
            # Assign or generate parent ID
            parent_id = ids[i] if ids else doc.metadata.get("id", str(uuid.uuid4()))
            # Ensure original doc metadata has the definitive ID
            doc.metadata["id"] = parent_id

            # Attempt to split the document
            splits = self.document_splitter.split_document(doc) 

            if len(splits) > 1:
                # Document was split
                # 1. Mark original document as parent
                parent_doc = doc # Keep original doc object
                parent_doc.metadata["is_parent"] = True
                parent_doc.metadata["split_count"] = len(splits) # Add split count
                docs_to_add.append(parent_doc)
                
                # 2. Add split chunks (metadata already set by splitter)
                docs_to_add.extend(splits)
                processed_parent_ids.append(parent_id)
            else:
                # Document was not split, add the original document as is
                # Ensure no potentially confusing parent/split metadata is present
                doc.metadata.pop("is_parent", None)
                doc.metadata.pop("is_split", None)
                doc.metadata.pop("parent_id", None)
                doc.metadata.pop("split_index", None)
                doc.metadata.pop("split_count", None)
                docs_to_add.append(doc)
                processed_parent_ids.append(parent_id) # Still track the ID
        
        # Add all collected documents (parents, splits, unsplit) to the underlying store at once
        if docs_to_add:
             # Extract IDs for the final list to add
             final_ids = [d.metadata["id"] for d in docs_to_add]
             self._add_documents(docs_to_add, final_ids)

        # Return the IDs of the original documents processed
        return processed_parent_ids

    @abstractmethod
    def _add_documents(
        self,
        documents: Sequence[Document],
        ids: Optional[Sequence[str]] = None
    ) -> List[str]:
        """
        Add documents to the store (implementation)
        ドキュメントをストアに追加する（実装）

        Args:
            documents (Sequence[Document]): Documents to add
                追加するドキュメント
            ids (Optional[Sequence[str]], optional): Optional document IDs
                オプションのドキュメントID

        Returns:
            List[str]: List of document IDs
                ドキュメントIDのリスト
        """
        pass

    def get_document(self, id_: str) -> Optional[Document]:
        """
        Get a document by ID
        IDでドキュメントを取得する

        Args:
            id_ (str): Document ID
                ドキュメントID

        Returns:
            Optional[Document]: Document if found, None otherwise
                見つかった場合はドキュメント、見つからない場合はNone
        """
        return self._get_document(id_)

    @abstractmethod
    def _get_document(self, id_: str) -> Optional[Document]:
        """
        Get a document by ID (implementation)
        IDでドキュメントを取得する（実装）

        Args:
            id_ (str): Document ID
                ドキュメントID

        Returns:
            Optional[Document]: Document if found, None otherwise
                見つかった場合はドキュメント、見つからない場合はNone
        """
        pass

    def get_parent_document(self, doc_id: str) -> Optional[Document]:
        """
        Get parent document by ID (either parent ID or split ID)
        IDで親文書を取得（親IDまたは分割ID）

        Args:
            doc_id (str): Document ID (parent or split).
                文書ID（親または分割）。

        Returns:
            Optional[Document]: Parent document if found, None otherwise.
                見つかった場合は親文書、それ以外はNone。
        """
        doc = self.get_document(doc_id)
        if not doc:
            return None

        if doc.metadata.get('is_split'):
            parent_id = doc.metadata.get('parent_id')
            if parent_id:
                return self.get_document(parent_id)
        
        if doc.metadata.get('is_parent'):
            return doc

        return None

    def get_split_documents(self, parent_id: str) -> List[Document]:
        """
        Get split documents for a parent document
        親ドキュメントの分割ドキュメントを取得する

        Args:
            parent_id (str): Parent document ID
                親ドキュメントID

        Returns:
            List[Document]: List of split documents
                分割ドキュメントのリスト
        """
        return self._get_split_documents(parent_id)

    @abstractmethod
    def _get_split_documents(self, parent_id: str) -> List[Document]:
        """
        Get split documents for a parent document (implementation)
        親ドキュメントの分割ドキュメントを取得する（実装）

        Args:
            parent_id (str): Parent document ID
                親ドキュメントID

        Returns:
            List[Document]: List of split documents
                分割ドキュメントのリスト
        """
        pass 