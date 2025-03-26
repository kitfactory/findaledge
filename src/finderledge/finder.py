"""
Finder - A class for searching documents using embeddings and BM25
Finder - 埋め込みとBM25を使用して文書を検索するためのクラス

This class provides functionality for searching documents using a combination of embeddings and BM25.
このクラスは、埋め込みとBM25を組み合わせて文書を検索する機能を提供します。
"""

from dataclasses import dataclass
from typing import List, Optional, Union, Dict
import numpy as np
import os
import json
from .document import Document
from .document_store import DocumentStore
from .embedding_store import EmbeddingStore
from .embedding_model import EmbeddingModel
from .tokenizer import Tokenizer
from .bm25 import BM25

@dataclass
class SearchResult:
    """
    A class representing a search result
    検索結果を表すクラス

    Attributes:
        document (Document): The matched document
        score (float): The search score
    """
    document: Document
    score: float

class Finder:
    """
    A class for searching documents using embeddings and BM25
    埋め込みとBM25を使用して文書を検索するためのクラス

    Attributes:
        tokenizer (Tokenizer): Tokenizer for text processing
        embedding_model (EmbeddingModel): Model for generating embeddings
        document_store (DocumentStore): Store for managing documents
        embedding_store (EmbeddingStore): Store for managing embeddings
        bm25 (BM25): BM25 search engine
        storage_dir (str): Directory to store persistent data
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        embedding_model: EmbeddingModel,
        document_store: DocumentStore,
        embedding_store: EmbeddingStore,
        bm25: BM25,
        storage_dir: str
    ):
        """
        Initialize the finder
        finderを初期化

        Args:
            tokenizer (Tokenizer): Tokenizer for text processing
            embedding_model (EmbeddingModel): Model for generating embeddings
            document_store (DocumentStore): Store for managing documents
            embedding_store (EmbeddingStore): Store for managing embeddings
            bm25 (BM25): BM25 search engine
            storage_dir (str): Directory to store persistent data
        """
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.document_store = document_store
        self.embedding_store = embedding_store
        self.bm25 = bm25
        self.storage_dir = storage_dir
        self.document_contents = {}

        # 永続化ディレクトリの作成
        os.makedirs(storage_dir, exist_ok=True)
        self.bm25_path = os.path.join(storage_dir, "bm25.json")

        # BM25の状態を読み込む
        if os.path.exists(self.bm25_path):
            self.bm25.load(self.bm25_path)
            # 既存の文書を読み込む（BM25の順序を維持）
            self.document_contents = {}
            for doc_id in self.bm25.doc_ids:
                doc = self.document_store.get_document(doc_id)
                if doc:
                    self.document_contents[doc_id] = doc.content
        else:
            # 既存の文書を読み込む
            self.document_contents = {}
            for doc_id in self.document_store.list_documents():
                doc = self.document_store.get_document(doc_id)
                if doc:
                    self.document_contents[doc_id] = doc.content
            # BM25の状態が存在しない場合のみ更新
            documents = list(self.document_contents.values())
            doc_ids = list(self.document_contents.keys())
            self.bm25.fit(documents, doc_ids)
            self.bm25.save(self.bm25_path)

    def add_document(self, document: Document) -> None:
        """
        Add a document to the finder
        文書をfinderに追加

        Args:
            document (Document): Document to add

        Raises:
            ValueError: If document is None or has no content
        """
        if document is None:
            raise ValueError("Document cannot be None")
        if not document.content:
            raise ValueError("Document content cannot be empty")

        # 文書を保存
        self.document_store.add_document(document)
        self.document_contents[document.id] = document.content

        # 埋め込みを生成して保存
        embedding = self.embedding_model.embed_text(document.content)
        self.embedding_store.add_embedding(document.id, embedding)

        # BM25を更新
        self._update_bm25()

    def remove_document(self, document_id: str) -> None:
        """
        Remove a document from the finder
        文書をfinderから削除

        Args:
            document_id (str): ID of the document to remove

        Raises:
            ValueError: If document_id is not found
        """
        if document_id not in self.document_contents:
            raise ValueError(f"Document with ID {document_id} not found")

        # 文書を削除
        self.document_store.delete_document(document_id)
        del self.document_contents[document_id]

        # 埋め込みを削除
        self.embedding_store.delete_embedding(document_id)

        # BM25を更新
        self._update_bm25()

    def _update_bm25(self) -> None:
        """
        Update BM25 model and save its state
        BM25モデルを更新して状態を保存
        """
        # BM25を更新
        documents = list(self.document_contents.values())
        doc_ids = list(self.document_contents.keys())
        self.bm25.fit(documents, doc_ids)
        # 状態を保存
        self.bm25.save(self.bm25_path)

    def search(
        self,
        query: str,
        search_mode: str = "hybrid",
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Search for documents using the specified mode
        指定されたモードで文書を検索

        Args:
            query (str): Search query
            search_mode (str): Search mode ("hybrid", "semantic", or "keyword")
            top_k (int): Number of results to return

        Returns:
            List[SearchResult]: List of search results

        Raises:
            ValueError: If search_mode is invalid
        """
        if not query:
            return []

        if search_mode not in ["hybrid", "semantic", "keyword"]:
            raise ValueError(f"Unknown search mode: {search_mode}")

        # クエリの埋め込みを生成
        query_embedding = self.embedding_model.embed_text(query)

        # 文書IDのリストを取得（BM25の順序を使用）
        doc_ids = self.bm25.doc_ids
        if not doc_ids:
            return []

        # 各文書のスコアを計算
        scores = []
        for doc_id in doc_ids:
            if search_mode == "hybrid":
                # ハイブリッド検索：セマンティック検索とキーワード検索の結果を組み合わせる
                semantic_score = self._calculate_semantic_score(query_embedding, doc_id)
                tokens = self.tokenizer.tokenize(query)
                keyword_score = self.bm25.score(tokens, doc_id)
                # Reciprocal Rank Fusion (RRF)を使用してスコアを組み合わせる
                rrf_k = 60  # RRFのパラメータ
                rrf_semantic = 1 / (rrf_k + semantic_score)
                rrf_keyword = 1 / (rrf_k + keyword_score)
                score = rrf_semantic + rrf_keyword
            elif search_mode == "semantic":
                # セマンティック検索のみ
                score = self._calculate_semantic_score(query_embedding, doc_id)
            else:  # keyword
                # キーワード検索のみ
                tokens = self.tokenizer.tokenize(query)
                score = self.bm25.score(tokens, doc_id)

            scores.append((doc_id, score))

        # スコアでソート（BM25の順序を維持）
        scores.sort(key=lambda x: (-x[1], doc_ids.index(x[0])))

        # 上位k件の結果を返す
        results = []
        for doc_id, score in scores[:top_k]:
            document = self.document_store.get_document(doc_id)
            if document:
                results.append(SearchResult(document, score))

        return results

    def _calculate_semantic_score(self, query_embedding: np.ndarray, doc_id: str) -> float:
        """
        Calculate semantic similarity score
        セマンティック類似度スコアを計算

        Args:
            query_embedding (np.ndarray): Query embedding
            doc_id (str): Document ID

        Returns:
            float: Semantic similarity score
        """
        doc_embedding = self.embedding_store.get_embedding(doc_id)
        if doc_embedding is None:
            return 0.0

        # コサイン類似度を計算
        similarity = np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
        )
        return float(similarity)

    def to_dict(self) -> dict:
        """
        Convert the finder to a dictionary
        finderを辞書に変換

        Returns:
            dict: Dictionary representation of the finder
        """
        return {
            "document_store": self.document_store.to_dict(),
            "embedding_store": self.embedding_store.to_dict(),
            "bm25": self.bm25.to_dict()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Finder":
        """
        Create a finder from a dictionary
        辞書からfinderを作成

        Args:
            data (dict): Dictionary representation of the finder

        Returns:
            Finder: New finder instance
        """
        document_store = DocumentStore.from_dict(data["document_store"])
        embedding_store = EmbeddingStore.from_dict(data["embedding_store"])
        bm25 = BM25.from_dict(data["bm25"])

        return cls(
            tokenizer=None,  # トークナイザーは別途設定が必要
            embedding_model=None,  # 埋め込みモデルは別途設定が必要
            document_store=document_store,
            embedding_store=embedding_store,
            bm25=bm25
        ) 