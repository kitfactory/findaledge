"""
Finder - A class for searching documents using embeddings and BM25
Finder - 埋め込みとBM25を使用して文書を検索するためのクラス

This class provides functionality for searching documents using a combination of embeddings and BM25.
このクラスは、埋め込みとBM25を組み合わせて文書を検索する機能を提供します。
"""

from dataclasses import dataclass
from typing import List, Optional, Union
import numpy as np
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
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        embedding_model: EmbeddingModel,
        document_store: DocumentStore,
        embedding_store: EmbeddingStore,
        bm25: BM25
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
        """
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.document_store = document_store
        self.embedding_store = embedding_store
        self.bm25 = bm25
        self.documents = []

    def add_document(self, document: Document) -> None:
        """
        Add a document to the finder
        文書をfinderに追加

        Args:
            document (Document): Document to add
        """
        # 文書を保存
        self.document_store.add_document(document)

        # 埋め込みを生成して保存
        embedding = self.embedding_model.generate_embedding(document.content)
        self.embedding_store.add_embedding(document.id, embedding)

        # 文書を追加してBM25を更新
        self.documents.append(document.content)
        if self.tokenizer:
            tokenized_documents = [self.tokenizer.tokenize(doc) for doc in self.documents]
            self.bm25.fit(tokenized_documents)
        else:
            # トークナイザーがない場合は、単純に空白で分割
            tokenized_documents = [doc.split() for doc in self.documents]
            self.bm25.fit(tokenized_documents)

    def search(
        self,
        query: str,
        search_mode: str = "hybrid",
        top_k: int = 5
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
            raise ValueError(f"Invalid search mode: {search_mode}")

        # クエリの埋め込みを生成
        query_embedding = self.embedding_model.generate_embedding(query)

        # 文書の埋め込みを取得
        document_embeddings = []
        document_ids = []
        for doc_id in self.document_store.documents.keys():
            embedding = self.embedding_store.get_embedding(doc_id)
            if embedding is not None:
                document_embeddings.append(embedding)
                document_ids.append(doc_id)

        if not document_embeddings:
            return []

        # コサイン類似度を計算
        document_embeddings = np.array(document_embeddings)
        similarities = np.dot(document_embeddings, query_embedding) / (
            np.linalg.norm(document_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # BM25スコアを計算
        if self.tokenizer:
            tokenized_query = self.tokenizer.tokenize(query)
        else:
            tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # 結果を統合
        results = []
        for i, doc_id in enumerate(document_ids):
            document = self.document_store.get_document(doc_id)
            if document is None:
                continue

            if search_mode == "hybrid":
                # ハイブリッド検索：コサイン類似度とBM25スコアを組み合わせる
                score = 0.5 * similarities[i] + 0.5 * bm25_scores[i]
            elif search_mode == "semantic":
                # セマンティック検索：コサイン類似度のみを使用
                score = similarities[i]
            else:  # keyword
                # キーワード検索：BM25スコアのみを使用
                score = bm25_scores[i]

            results.append(SearchResult(document=document, score=float(score)))

        # スコアの降順でソート
        results.sort(key=lambda x: x.score, reverse=True)

        # top_k件を返す
        return results[:top_k]

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