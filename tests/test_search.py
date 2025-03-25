"""
Tests for the search functionality
検索機能のテスト
"""

import pytest
import numpy as np
import tempfile
import os
from finderledge import Document, DocumentStore, EmbeddingStore, EmbeddingModel, BM25, Finder

@pytest.fixture
def temp_dir():
    """
    Create a temporary directory for test storage
    テスト用の一時ディレクトリを作成
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def documents():
    """
    Create test documents
    テスト用の文書を作成
    """
    return [
        Document(
            id="doc1",
            title="Python Programming",
            content="Python is a popular programming language for data science and machine learning."
        ),
        Document(
            id="doc2",
            title="Machine Learning Basics",
            content="Machine learning is a subset of artificial intelligence that focuses on data analysis."
        ),
        Document(
            id="doc3",
            title="Data Science Overview",
            content="Data science combines statistics, programming, and domain expertise to analyze data."
        )
    ]

@pytest.fixture
def embedding_model():
    """
    Create a mock embedding model
    モックの埋め込みモデルを作成
    """
    class MockEmbeddingModel(EmbeddingModel):
        def generate_embedding(self, text: str) -> np.ndarray:
            # 簡単なモック実装：テキストの長さに基づいて固定の埋め込みを生成
            return np.array([len(text) / 100] * 10)

        def generate_embeddings(self, texts: list[str]) -> list[np.ndarray]:
            return [self.generate_embedding(text) for text in texts]

    return MockEmbeddingModel()

@pytest.fixture
def finder(documents, embedding_model, temp_dir):
    """
    Create a finder instance with test documents
    テスト文書を含むfinderインスタンスを作成
    """
    document_store = DocumentStore(storage_dir=os.path.join(temp_dir, "documents"))
    embedding_store = EmbeddingStore(store_dir=os.path.join(temp_dir, "embeddings"))
    bm25 = BM25()

    finder = Finder(
        tokenizer=None,  # テストでは使用しない
        embedding_model=embedding_model,
        document_store=document_store,
        embedding_store=embedding_store,
        bm25=bm25
    )

    # テスト文書を追加
    for doc in documents:
        finder.add_document(doc)

    return finder

def test_hybrid_search(finder):
    """
    Test hybrid search functionality
    ハイブリッド検索機能のテスト
    """
    # 検索クエリ
    query = "data science and machine learning"

    # 検索実行
    results = finder.search(query, search_mode="hybrid")

    # 結果の検証
    assert len(results) > 0
    assert all(hasattr(result, "score") for result in results)
    assert all(hasattr(result, "document") for result in results)
    assert all(isinstance(result.score, float) for result in results)

def test_semantic_search(finder):
    """
    Test semantic search functionality
    セマンティック検索機能のテスト
    """
    # 検索クエリ
    query = "data science and machine learning"

    # 検索実行
    results = finder.search(query, search_mode="semantic")

    # 結果の検証
    assert len(results) > 0
    assert all(hasattr(result, "score") for result in results)
    assert all(hasattr(result, "document") for result in results)
    assert all(isinstance(result.score, float) for result in results)

def test_keyword_search(finder):
    """
    Test keyword search functionality
    キーワード検索機能のテスト
    """
    # 検索クエリ
    query = "data science and machine learning"

    # 検索実行
    results = finder.search(query, search_mode="keyword")

    # 結果の検証
    assert len(results) > 0
    assert all(hasattr(result, "score") for result in results)
    assert all(hasattr(result, "document") for result in results)
    assert all(isinstance(result.score, float) for result in results)

def test_search_with_empty_query(finder):
    """
    Test search with empty query
    空のクエリでの検索テスト
    """
    results = finder.search("")
    assert len(results) == 0

def test_search_with_unknown_mode(finder):
    """
    Test search with unknown search mode
    未知の検索モードでのテスト
    """
    with pytest.raises(ValueError):
        finder.search("test query", search_mode="unknown")

def test_search_result_ordering(finder):
    """
    Test search result ordering
    検索結果の順序付けテスト
    """
    query = "data science"
    results = finder.search(query, search_mode="hybrid")

    # スコアの降順でソートされていることを確認
    scores = [result.score for result in results]
    assert scores == sorted(scores, reverse=True) 