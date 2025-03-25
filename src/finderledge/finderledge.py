"""
FinderLedge - Document context management library for OpenAI Agents SDK
FinderLedge - OpenAI Agents SDKのための文書コンテキスト管理ライブラリ

This module provides a library for managing document contexts in OpenAI Agents SDK,
supporting features like automatic document indexing, hybrid search, and persistence.
このモジュールは、OpenAI Agents SDKで文書コンテキストを管理するための
ライブラリを提供し、自動文書インデックス作成、ハイブリッド検索、
永続化などの機能をサポートします。
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import chromadb
from bm25s import BM25, tokenize
from pydantic import BaseModel
import uuid

from .document import Document
from .embedding import EmbeddingModel
from .text_splitter import TextSplitter
from .document_loader import DocumentLoader

class FinderLedge:
    """
    Document context management system
    文書コンテキスト管理システム

    This class provides functionality for managing document contexts,
    including document indexing, search, and persistence.
    このクラスは、文書インデックス作成、検索、永続化を含む
    文書コンテキスト管理機能を提供します。
    """

    def __init__(
        self,
        db_name: str = "finderledge",
        persist_dir: str = "data",
        embedding_model: Optional[EmbeddingModel] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the FinderLedge system
        FinderLedgeシステムを初期化

        Args:
            db_name (str): Name of the database / データベース名
            persist_dir (str): Directory for persisting data / データを永続化するディレクトリ
            embedding_model (Optional[EmbeddingModel]): Embedding model to use / 使用する埋め込みモデル
            chunk_size (int): Size of text chunks / テキストチャンクのサイズ
            chunk_overlap (int): Overlap between chunks / チャンク間の重複
        """
        self.db_name = db_name
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.embedding_model = embedding_model or EmbeddingModel()
        self.text_splitter = TextSplitter(chunk_size, chunk_overlap)
        self.document_loader = DocumentLoader()

        # Initialize storage
        self.vector_store = chromadb.PersistentClient(
            path=str(self.persist_dir / "chroma")
        )
        self.collection = self.vector_store.get_or_create_collection(
            name=db_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.bm25_index = BM25([])
        self.documents: Dict[str, Document] = {}

        # Load persisted state if exists
        self._load_state()

    def add_document(
        self,
        content: Union[str, Document],
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a document to the system
        システムに文書を追加

        Args:
            content (Union[str, Document]): Document content or Document object / 文書内容または文書オブジェクト
            title (Optional[str]): Document title / 文書タイトル
            metadata (Optional[Dict[str, Any]]): Document metadata / 文書メタデータ

        Returns:
            str: Document ID / 文書ID
        """
        # Create document
        if isinstance(content, Document):
            doc = content
        else:
            doc = Document(
                id=str(uuid.uuid4()),
                content=content,
                title=title or f"Document {len(self.documents) + 1}",
                metadata=metadata or {}
            )

        # Split content into chunks
        chunks = self.text_splitter.split_text(doc.content)
        doc.add_chunks(chunks)

        # Generate embeddings for chunks
        embeddings = self.embedding_model.embed_documents(chunks)
        doc.add_chunk_embeddings(embeddings)

        # Add to vector store
        self.collection.add(
            ids=[f"{doc.id}_{i}" for i in range(len(chunks))],
            embeddings=embeddings,
            metadatas=[{"doc_id": doc.id, "chunk_index": i} for i in range(len(chunks))],
            documents=chunks
        )

        # Add to BM25 index
        self.bm25_index.add_document(doc.id, doc.content)

        # Store document
        self.documents[doc.id] = doc

        # Persist state
        self._persist_state()

        return doc.id

    def remove_document(self, doc_id: str) -> None:
        """
        Remove a document from the system
        システムから文書を削除

        Args:
            doc_id (str): Document ID / 文書ID
        """
        if doc_id not in self.documents:
            raise ValueError(f"Document not found: {doc_id}")

        doc = self.documents[doc_id]
        chunk_ids = [f"{doc_id}_{i}" for i in range(len(doc.chunks))]

        # Remove from vector store
        self.vector_store.delete(ids=chunk_ids)

        # Remove from BM25 index
        tokenized_chunks = [tokenize(chunk) for chunk in doc.chunks]
        self.bm25_index = BM25([])  # Reset index
        for doc_id, doc in self.documents.items():
            if doc_id != doc_id:  # Skip the document being removed
                tokenized_chunks = [tokenize(chunk) for chunk in doc.chunks]
                self.bm25_index = BM25(tokenized_chunks)

        # Remove from documents
        del self.documents[doc_id]

        # Persist state
        self._persist_state()

    def find_related_documents(
        self,
        query: str,
        k: int = 5,
        search_mode: str = "hybrid"
    ) -> List[Document]:
        """
        Find documents related to a query
        クエリに関連する文書を検索

        Args:
            query (str): Search query / 検索クエリ
            k (int): Number of results to return / 返す結果の数
            search_mode (str): Search mode ("hybrid", "vector", or "keyword") / 検索モード

        Returns:
            List[Document]: List of related documents / 関連文書のリスト
        """
        if search_mode not in ["hybrid", "vector", "keyword"]:
            raise ValueError("search_mode must be one of: hybrid, vector, keyword")

        # Get vector search results
        if search_mode in ["hybrid", "vector"]:
            query_embedding = self.embedding_model.embed_query(query)
            vector_results = self.vector_store.query(
                query_embeddings=[query_embedding],
                n_results=k
            )

        # Get keyword search results
        if search_mode in ["hybrid", "keyword"]:
            tokenized_query = tokenize(query)
            keyword_results = self.bm25_index.get_scores(tokenized_query)
            # Convert scores to document IDs
            doc_scores = {}
            for i, chunk in enumerate(self.bm25_index.corpus):
                doc_id = chunk.get("doc_id")
                if doc_id:
                    doc_scores[doc_id] = max(doc_scores.get(doc_id, 0), keyword_results[i])
            keyword_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]
            keyword_results = [doc_id for doc_id, _ in keyword_results]

        # Combine results
        doc_ids = set()
        if search_mode in ["hybrid", "vector"]:
            for metadata in vector_results["metadatas"][0]:
                doc_ids.add(metadata["doc_id"])
        if search_mode in ["hybrid", "keyword"]:
            for doc_id in keyword_results:
                doc_ids.add(doc_id)

        # Get documents
        return [self.documents[doc_id] for doc_id in doc_ids]

    def get_context(
        self,
        query: str,
        k: int = 5,
        search_mode: str = "hybrid"
    ) -> str:
        """
        Get context for a query
        クエリのコンテキストを取得

        Args:
            query (str): Query to get context for / コンテキストを取得するクエリ
            k (int): Number of documents to include / 含める文書の数
            search_mode (str): Search mode to use / 使用する検索モード

        Returns:
            str: Combined context from related documents / 関連文書からの結合されたコンテキスト
        """
        related_docs = self.find_related_documents(query, k, search_mode)
        return "\n\n".join(doc.content for doc in related_docs)

    def _persist_state(self) -> None:
        """
        Persist the current state to disk
        現在の状態をディスクに永続化
        """
        # Save documents
        docs_file = self.persist_dir / "documents.json"
        with open(docs_file, "w", encoding="utf-8") as f:
            json.dump(
                {doc_id: doc.model_dump() for doc_id, doc in self.documents.items()},
                f,
                ensure_ascii=False,
                indent=2
            )

        # Save BM25 index
        bm25_file = self.persist_dir / "bm25.json"
        with open(bm25_file, "w", encoding="utf-8") as f:
            json.dump({
                "corpus": self.bm25_index.corpus,
                "doc_freqs": self.bm25_index.doc_freqs,
                "doc_lens": self.bm25_index.doc_lens,
                "avg_doc_len": self.bm25_index.avg_doc_len,
                "k1": self.bm25_index.k1,
                "b": self.bm25_index.b
            }, f, ensure_ascii=False, indent=2)

    def _load_state(self) -> None:
        """
        Load persisted state from disk
        ディスクから永続化された状態を読み込む
        """
        # Load documents
        docs_file = self.persist_dir / "documents.json"
        if docs_file.exists():
            with open(docs_file, "r", encoding="utf-8") as f:
                docs_data = json.load(f)
                self.documents = {
                    doc_id: Document(**doc_data)
                    for doc_id, doc_data in docs_data.items()
                }

        # Load BM25 index
        bm25_file = self.persist_dir / "bm25.json"
        if bm25_file.exists():
            with open(bm25_file, "r", encoding="utf-8") as f:
                bm25_data = json.load(f)
                self.bm25_index = BM25(
                    corpus=bm25_data["corpus"],
                    doc_freqs=bm25_data["doc_freqs"],
                    doc_lens=bm25_data["doc_lens"],
                    avg_doc_len=bm25_data["avg_doc_len"],
                    k1=bm25_data["k1"],
                    b=bm25_data["b"]
                )

    def close(self) -> None:
        """
        Close the FinderLedge instance
        FinderLedgeインスタンスを閉じる
        """
        self._persist_state()
        self.vector_store = None
        self.bm25_index = None
        self.documents.clear()

    def get_langchain_retriever(self) -> Any:
        """
        Get a LangChain retriever interface
        LangChainのretrieverインターフェースを取得

        Returns:
            Any: LangChain retriever interface / LangChainのretrieverインターフェース
        """
        # TODO: Implement LangChain retriever interface
        raise NotImplementedError("LangChain retriever interface not implemented yet") 