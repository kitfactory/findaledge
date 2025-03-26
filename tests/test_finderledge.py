"""
Tests for FinderLedge
FinderLedgeのテスト
"""

import pytest
from pathlib import Path
import shutil
import time
from finderledge.finderledge import FinderLedge
from finderledge.document import Document

@pytest.fixture
def temp_dir():
    """
    Create temporary directory for testing
    テスト用の一時ディレクトリを作成
    """
    temp_dir = Path("tests/temp")
    temp_dir.mkdir(exist_ok=True)
    yield temp_dir
    
    # ChromaDBのファイルを確実にクリーンアップ
    time.sleep(0.1)  # ファイルのロックを解除するために少し待機
    try:
        shutil.rmtree(temp_dir)
    except PermissionError:
        # ファイルがロックされている場合は、もう一度試行
        time.sleep(0.1)
        shutil.rmtree(temp_dir)

def test_finderledge_initialization(temp_dir):
    """
    Test FinderLedge initialization
    FinderLedgeの初期化テスト
    """
    finder = FinderLedge(
        db_name="test_db",
        persist_dir=temp_dir,
        embedding_model="text-embedding-3-small",
        chunk_size=100,
        chunk_overlap=20
    )
    
    assert finder.db_name == "test_db"
    assert finder.persist_dir == temp_dir
    assert finder.embedding_model == "text-embedding-3-small"
    assert finder.chunk_size == 100
    assert finder.chunk_overlap == 20
    assert finder.vector_store is not None
    assert finder.bm25_index is not None

def test_add_document(temp_dir):
    """
    Test adding document
    文書追加のテスト
    """
    finder = FinderLedge(persist_dir=temp_dir)
    doc = Document(
        content="This is a test document.",
        title="Test Document",
        metadata={"source": "test"}
    )
    
    finder.add_document(doc)
    assert len(finder.documents) == 1
    assert finder.documents[0].content == doc.content
    assert finder.documents[0].title == doc.title
    assert finder.documents[0].metadata == doc.metadata

def test_remove_document(temp_dir):
    """
    Test removing document
    文書削除のテスト
    """
    finder = FinderLedge(persist_dir=temp_dir)
    doc = Document(
        content="This is a test document.",
        title="Test Document"
    )
    
    finder.add_document(doc)
    assert len(finder.documents) == 1
    
    finder.remove_document(doc.id)
    assert len(finder.documents) == 0
    
    with pytest.raises(ValueError):
        finder.remove_document(doc.id)

def test_find_related_documents(temp_dir):
    """
    Test finding related documents
    関連文書検索のテスト
    """
    finder = FinderLedge(persist_dir=temp_dir)
    
    # テスト文書を追加
    docs = [
        Document(content="This is a test document about Python.", title="Python Doc"),
        Document(content="This is a test document about JavaScript.", title="JS Doc"),
        Document(content="This is a test document about Java.", title="Java Doc")
    ]
    
    for doc in docs:
        finder.add_document(doc)
    
    # ハイブリッド検索
    results = finder.find_related_documents("Python programming", search_mode="hybrid")
    assert len(results) > 0
    assert any("Python" in doc.content for doc in results)
    
    # ベクトル検索
    results = finder.find_related_documents("JavaScript development", search_mode="vector")
    assert len(results) > 0
    assert any("JavaScript" in doc.content for doc in results)
    
    # キーワード検索
    results = finder.find_related_documents("Java", search_mode="keyword")
    assert len(results) > 0
    assert any("Java" in doc.content for doc in results)
    
    # 無効な検索モード
    with pytest.raises(ValueError):
        finder.find_related_documents("test", search_mode="invalid")

def test_get_context(temp_dir):
    """
    Test getting context
    コンテキスト取得のテスト
    """
    finder = FinderLedge(persist_dir=temp_dir)
    
    # テスト文書を追加
    doc = Document(
        content="This is a test document about Python programming. Python is a popular programming language.",
        title="Python Doc"
    )
    finder.add_document(doc)
    
    context = finder.get_context("Python programming")
    assert isinstance(context, str)
    assert "Python" in context
    assert len(context) <= finder.chunk_size

def test_persistence(temp_dir):
    """
    Test persistence functionality
    永続化機能のテスト
    """
    # 最初のインスタンス
    finder1 = FinderLedge(persist_dir=temp_dir)
    doc = Document(
        content="This is a test document.",
        title="Test Document"
    )
    finder1.add_document(doc)
    finder1.close()
    
    # 新しいインスタンス
    finder2 = FinderLedge(persist_dir=temp_dir)
    assert len(finder2.documents) == 1
    assert finder2.documents[0].content == doc.content
    assert finder2.documents[0].title == doc.title
    finder2.close()

def test_close(temp_dir):
    """
    Test closing FinderLedge
    FinderLedgeのクローズテスト
    """
    finder = FinderLedge(persist_dir=temp_dir)
    doc = Document(
        content="This is a test document.",
        title="Test Document"
    )
    finder.add_document(doc)
    
    finder.close()
    
    with pytest.raises(RuntimeError):
        finder.add_document(doc)
    
    with pytest.raises(RuntimeError):
        finder.find_related_documents("test")

def test_get_langchain_retriever(temp_dir):
    """
    Test getting LangChain retriever
    LangChainリトリーバー取得のテスト
    """
    finder = FinderLedge(persist_dir=temp_dir)
    retriever = finder.get_langchain_retriever()
    assert retriever is not None
    finder.close() 