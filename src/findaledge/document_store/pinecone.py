"""
Pinecone Document Store Module
Pineconeドキュメントストアモジュール

This module provides a Pinecone-based implementation of the vector document store.
このモジュールはPineconeベースのベクトルドキュメントストアの実装を提供します。
"""

from typing import List, Optional, Sequence
from uuid import uuid4
import pinecone
from langchain.schema import Document, BaseRetriever
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from .vector_document_store import VectorDocumentStore

class PineconeDocumentStore(VectorDocumentStore):
    """
    Pinecone-based document store implementation
    Pineconeベースのドキュメントストア実装

    This class implements the vector document store using Pinecone as the backend.
    このクラスはPineconeをバックエンドとして使用するベクトルドキュメントストアを実装します。
    """

    def __init__(
        self,
        api_key: str,
        environment: str,
        index_name: str,
        embedding_function: Embeddings,
        namespace: str = "",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        **kwargs
    ) -> None:
        """
        Initialize the Pinecone document store
        Pineconeドキュメントストアを初期化

        Args:
            api_key (str): Pinecone API key
                PineconeのAPIキー
            environment (str): Pinecone environment
                Pinecone環境
            index_name (str): Name of the Pinecone index
                Pineconeインデックス名
            embedding_function (Embeddings): Function to generate embeddings
                埋め込みを生成する関数
            namespace (str, optional): Pinecone namespace. Defaults to "".
                Pineconeの名前空間。デフォルトは空文字。
            chunk_size (int, optional): Size of text chunks. Defaults to 1000.
                テキストチャンクのサイズ。デフォルトは1000。
            chunk_overlap (int, optional): Overlap between chunks. Defaults to 200.
                チャンク間の重複。デフォルトは200。
        """
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        
        # Initialize Pinecone
        pinecone.init(api_key=api_key, environment=environment)
        
        # Create Pinecone vector store
        self.pinecone_store = LangchainPinecone(
            embedding=embedding_function,
            index_name=index_name,
            namespace=namespace
        )

    def _add_documents(
        self,
        documents: Sequence[Document],
        ids: Optional[Sequence[str]] = None
    ) -> List[str]:
        """
        Add documents to Pinecone store
        Pineconeストアに文書を追加

        Args:
            documents (Sequence[Document]): Documents to add
                追加する文書
            ids (Optional[Sequence[str]], optional): Custom IDs for documents
                文書のカスタムID

        Returns:
            List[str]: List of document IDs
                文書IDのリスト
        """
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid4()) for _ in documents]
        
        # Add documents to Pinecone
        self.pinecone_store.add_documents(documents, ids=ids)
        
        # Return only parent document IDs
        return [id for id, doc in zip(ids, documents) if doc.metadata.get('is_parent', False)]

    def _get_document(self, doc_id: str) -> Optional[Document]:
        """
        Get document from Pinecone by ID
        IDによってPineconeから文書を取得

        Args:
            doc_id (str): Document ID
                文書ID

        Returns:
            Optional[Document]: Document if found, None otherwise
                見つかった場合は文書、それ以外はNone
        """
        results = self.pinecone_store.similarity_search(
            query="",
            k=1,
            filter={"id": doc_id}
        )
        return results[0] if results else None

    def _get_split_documents(self, parent_id: str) -> List[Document]:
        """
        Get split documents from Pinecone by parent ID
        親IDによってPineconeから分割文書を取得

        Args:
            parent_id (str): Parent document ID
                親文書ID

        Returns:
            List[Document]: List of split documents
                分割文書のリスト
        """
        results = self.pinecone_store.similarity_search(
            query="",
            filter={"parent_id": parent_id, "is_split": True}
        )
        return sorted(results, key=lambda x: x.metadata.get('split_index', 0))

    def delete_document(self, doc_id: str) -> None:
        """
        Delete document and its splits from Pinecone
        Pineconeから文書とその分割を削除

        Args:
            doc_id (str): Document ID
                文書ID
        """
        # Delete parent document
        self.pinecone_store.delete([doc_id])
        
        # Delete split documents
        splits = self._get_split_documents(doc_id)
        split_ids = [doc.metadata.get('id') for doc in splits]
        if split_ids:
            self.pinecone_store.delete(split_ids)

    def as_retriever(self, **kwargs) -> BaseRetriever:
        """
        Get retriever interface
        検索インターフェースを取得

        Returns:
            BaseRetriever: Retriever interface
                検索インターフェース
        """
        return self.pinecone_store.as_retriever(**kwargs) 