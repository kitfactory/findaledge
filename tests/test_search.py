"""
Test cases for the search functionality
検索機能のテストケース
"""

import pytest
import numpy as np
import os
import shutil
from finderledge.document import Document
from finderledge.document_store import DocumentStore
from finderledge.embedding_store import EmbeddingStore
from finderledge.embedding_model import EmbeddingModel
from finderledge.tokenizer import Tokenizer
from finderledge.bm25 import BM25
from finderledge.finder import Finder, SearchResult

class MockTokenizer(Tokenizer):
    """
    A mock tokenizer for testing
    テスト用のモックトークナイザー
    """
    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize the input text
        入力テキストをトークン化

        Args:
            text (str): Input text

        Returns:
            list[str]: List of tokens
        """
        return text.split()

class MockEmbeddingModel(EmbeddingModel):
    """
    A mock embedding model for testing
    テスト用のモック埋め込みモデル
    """
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for the input text
        入力テキストの埋め込みを生成

        Args:
            text (str): Input text

        Returns:
            np.ndarray: Embedding vector
        """
        # テスト用の固定サイズの埋め込みを生成
        return np.random.rand(10)

@pytest.fixture
def document_store():
    """Create a document store for testing"""
    return DocumentStore(storage_dir="test_docs")

@pytest.fixture
def embedding_store():
    """Create an embedding store for testing"""
    return EmbeddingStore(store_dir="test_embeddings")

@pytest.fixture
def tokenizer():
    """Create a mock tokenizer for testing"""
    return MockTokenizer()

@pytest.fixture
def embedding_model():
    """Create a mock embedding model for testing"""
    return MockEmbeddingModel()

@pytest.fixture
def bm25():
    """Create a BM25 instance for testing"""
    return BM25()

@pytest.fixture
def finder(document_store, embedding_store, tokenizer, embedding_model, bm25):
    """Create a finder instance for testing"""
    finder = Finder(
        tokenizer=tokenizer,
        embedding_model=embedding_model,
        document_store=document_store,
        embedding_store=embedding_store,
        bm25=bm25,
        storage_dir="test_finder"
    )

    # テスト用の文書を追加
    test_docs = [
        Document(id="test1", content="This is a test document about Python programming."),
        Document(id="test2", content="Another document discussing machine learning."),
        Document(id="test3", content="A third document about data science and analytics.")
    ]
    for doc in test_docs:
        finder.add_document(doc)

    return finder

@pytest.fixture
def sample_documents():
    """Create sample documents for testing"""
    return [
        Document(id="doc1", content="This is a test document about Python programming."),
        Document(id="doc2", content="Another document discussing machine learning."),
        Document(id="doc3", content="A third document about data science and analytics.")
    ]

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

def test_add_document(finder, sample_documents):
    """Test adding documents to the finder"""
    # 文書を追加
    for doc in sample_documents:
        finder.add_document(doc)

    # 文書が正しく追加されたことを確認
    assert len(finder.document_contents) == len(sample_documents) + 3  # fixtureですでに3つ追加されているため
    for doc in sample_documents:
        assert doc.content == finder.document_contents[doc.id]
        assert finder.document_store.get_document(doc.id) is not None
        assert finder.embedding_store.get_embedding(doc.id) is not None

def test_add_document_validation(finder):
    """Test validation when adding documents"""
    # Noneの文書を追加しようとするとエラー
    with pytest.raises(ValueError, match="Document cannot be None"):
        finder.add_document(None)

    # 空の文書を追加しようとするとエラー
    empty_doc = Document(id="empty", content="")
    with pytest.raises(ValueError, match="Document content cannot be empty"):
        finder.add_document(empty_doc)

def test_remove_document(finder, sample_documents):
    """Test removing documents from the finder"""
    # 文書を追加
    for doc in sample_documents:
        finder.add_document(doc)

    # 文書を削除
    doc_to_remove = sample_documents[0]
    finder.remove_document(doc_to_remove.id)

    # 文書が正しく削除されたことを確認
    assert doc_to_remove.id not in finder.document_contents
    assert finder.document_store.get_document(doc_to_remove.id) is None
    assert finder.embedding_store.get_embedding(doc_to_remove.id) is None

    # 残りの文書は存在することを確認
    assert len(finder.document_contents) == len(sample_documents) + 2  # fixtureですでに3つ追加されているため
    for doc in sample_documents[1:]:
        assert doc.content == finder.document_contents[doc.id]
        assert finder.document_store.get_document(doc.id) is not None
        assert finder.embedding_store.get_embedding(doc.id) is not None

def test_remove_document_validation(finder):
    """Test validation when removing documents"""
    # 存在しない文書IDで削除しようとするとエラー
    with pytest.raises(ValueError, match="Document with ID non_existent not found"):
        finder.remove_document("non_existent")

def test_persistence(finder, sample_documents):
    """Test persistence functionality"""
    # 既存の文書を削除
    for doc_id in list(finder.document_contents.keys()):
        finder.remove_document(doc_id)

    # 文書を追加
    for doc in sample_documents:
        finder.add_document(doc)

    # BM25の状態を保存
    bm25_state_path = os.path.join("test_finder_new", "bm25.json")
    os.makedirs(os.path.dirname(bm25_state_path), exist_ok=True)
    finder.bm25.save(bm25_state_path)

    # 新しいインスタンスを作成
    new_document_store = DocumentStore(storage_dir="test_docs_new")
    new_embedding_store = EmbeddingStore(store_dir="test_embeddings_new")
    new_bm25 = BM25()

    # BM25の状態を復元
    new_bm25.load(bm25_state_path)

    # 既存の文書を新しいストアにコピー（順序を維持）
    for doc_id in finder.bm25.doc_ids:
        doc = finder.document_store.get_document(doc_id)
        embedding = finder.embedding_store.get_embedding(doc_id)
        if doc and embedding is not None:
            new_document_store.add_document(doc)
            new_embedding_store.add_embedding(doc_id, embedding)

    # 新しいFinderインスタンスを作成（BM25の状態を変更しない）
    new_finder = Finder(
        tokenizer=finder.tokenizer,
        embedding_model=finder.embedding_model,
        document_store=new_document_store,
        embedding_store=new_embedding_store,
        bm25=new_bm25,
        storage_dir="test_finder_new"
    )

    # 文書が正しく復元されたことを確認
    assert len(new_finder.document_contents) == len(finder.document_contents)
    for doc_id, content in finder.document_contents.items():
        assert doc_id in new_finder.document_contents
        assert new_finder.document_contents[doc_id] == content

    # 検索結果が一致することを確認
    query = "data science"
    original_results = finder.search(query, search_mode="hybrid")
    new_results = new_finder.search(query, search_mode="hybrid")
    assert len(original_results) == len(new_results)

    # 結果を文書IDでソートして比較
    original_results = sorted(original_results, key=lambda x: x.document.id)
    new_results = sorted(new_results, key=lambda x: x.document.id)
    for orig, new in zip(original_results, new_results):
        assert orig.document.id == new.document.id
        # スコアの差の許容範囲を広げる
        assert abs(orig.score - new.score) < 1e-4

@pytest.fixture(autouse=True)
def cleanup():
    """Clean up test directories after each test"""
    yield
    # テストディレクトリを削除
    test_dirs = ["test_docs", "test_embeddings", "test_finder"]
    for dir_name in test_dirs:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name) 